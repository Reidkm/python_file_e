import glob, cv2, datetime, os, sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import mxnet as mx
from gluoncv import model_zoo, data, utils
from message import Message
import threading
import time
import yaml
import glob
import multiprocessing as mp
from utils import printLog
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class TopEagle(object):
    def __init__(self, id, video_source, camera_para_file, fix_distortion, need_record=False):
        self.id = id
        self.video_source = video_source
        self.classes = ['goods', 'pallet']  # only one foreground class here

        # net_name = 'yolo3_darknet53_custom'
        # params_fname = './data/models/yolo3_darknet53_custom_best.params'
        # im_loader = data.transforms.presets.yolo.load_test

        net_name = 'faster_rcnn_resnet101_v1d_custom'
        params_fname = '../models/2nd-no_mixup_faster_rcnn_resnet101_v1d_custom_best.params'
        self.im_loader = data.transforms.presets.rcnn.load_test

        self.ctx = mx.gpu(0)
        self.net = model_zoo.get_model(net_name, pretrained_base=True, classes=self.classes)
        self.net.load_parameters(params_fname)
        self.net.collect_params().reset_ctx(self.ctx)
        self.cam = None
        self.stopFlag = False
        self.fix_distortion = fix_distortion
        self.out = None
        self.need_record = need_record
        with open(os.path.join('cfg', camera_para_file), 'r') as stream:
            try:
                dict_1 = yaml.load(stream)
                self.camera_matrix = np.array(dict_1['mtx'])
                print self.camera_matrix
                self.distortion = np.array(dict_1['dist'])
                print self.distortion
            except yaml.YAMLError as exc:
                print(exc)

    def cameraSwitch(self):
        self.cam = cv2.VideoCapture(self.video_source)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        start = datetime.datetime.now()
        while True:
            self.cam.read()
            end = datetime.datetime.now()
            if (end - start).total_seconds() > 1:
                break
        self.queue = mp.Queue(maxsize=2)
        imgreadthread = threading.Thread(target=self.streamworker)
        imgreadthread.start()

    def undistortion(self, image):
        h, w = image.shape[:2]
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion, (w, h), 1, (w, h))
        dst = cv2.undistort(
            image, self.camera_matrix, self.distortion, None,
            new_matrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def streamworker(self):
        while not self.stopFlag:
            is_opened, temp = self.cam.read()
            if temp is not None:
                if self.fix_distortion:
                    temp = self.undistortion(temp)

                if self.out is None and self.need_record:
                    h, w = temp.shape[:2]
                    print('record: w: {0}, h:{1}'.format(w,h))
                    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
                    self.out = cv2.VideoWriter(
                        'output_{0}_{1}.avi'.format(
                            self.id, datetime.datetime.now().strftime('%Y%m%d%H%M%S')), fourcc, 20.0, (w, h))

                self.queue.put(temp) if is_opened else None
                self.queue.get() if self.queue.qsize() > 1 else None

            time.sleep(0.001)
        if self.cam is not None:
            self.cam.release()
        if self.out is not None:
            self.out.release()

    def process(self, use_nn=True):

        box_color = {0:(0,255,0), 1:(0, 0, 255)}
        classes = {0:Message.DetectObj.DETECT_OBJ_TYPE_CARGO,
                1:Message.DetectObj.DETECT_OBJ_TYPE_CONTAINER }
        source_image = self.queue.get().copy()
        if source_image is None:
            return None
        if not use_nn:
            return (None, {'result':source_image, 'src':source_image})
        # mxnet is RGB
        # opencv is BGR
        net_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        # transform from array to NDArray and resize to match import size
        x, test_image = data.transforms.presets.rcnn.transform_test(mx.nd.array(net_image), 720, 1280)
        # print('Shape of pre-processed image:', x.shape, net_image.shape, test_image.shape, mor.shape)
        class_IDs, scores, bounding_boxs = self.net(x.copyto(self.ctx))
        # print "class_IDS"
        # print class_IDs
        # print "scoress"
        # print scores
        # print 'bounding_boxs'
        # print bounding_boxs
        objs0 = []
        list_box = bounding_boxs[0].asnumpy()
        result_img = test_image.copy()
        for i in range(len(list_box)):
            bbox = list_box[i]
            score = scores[0][i].asnumpy()[0]
            class_ID = int(class_IDs[0][i].asnumpy()[0])

            if score > 0.9:
                printLog('{0} {1} {2}'.format(bbox, score, class_ID))
                cv2.rectangle(test_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=box_color[class_ID], thickness=2)
                obj0 = Message.DetectObj(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), classes[class_ID])
                objs0.append(obj0)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, result_img = cv2.imencode('.jpg', result_img, encode_param)
        msg0 = Message(self.id, result_img, objs0)

        ########################## plot with viz
        # fig = Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.gca()
        # ax.axis('off')
        # ax.margins(x=0,y=0)
        # ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=classes, ax=ax)
        # # ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0])
        # # plt.show()
        # # plt.close()
        # canvas.draw()
        # width, height = fig.get_size_inches() * fig.get_dpi()
        # assert int(width) == 640 and int(height) == 480
        # frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        # plt.close()
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        return (msg0, {'result':test_image, 'src':source_image})

    def record(self, img):
        if self.out is not None and self.need_record:
            self.out.write(img)
        else:
            print('not ready for record')

    def videoSourceRelease(self):
        self.stopFlag = True
