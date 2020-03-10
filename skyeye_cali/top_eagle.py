import glob, cv2, datetime, os, sys
import numpy as np
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# import mxnet as mx
# from gluoncv import model_zoo, data, utils
# from message import Message
import threading
import time
import yaml
import glob
import multiprocessing as mp
# from utils import printLog
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class TopEagle(object):
    def __init__(self, camera_para_file, camModel='pinhole'):
        # self.id = id
        # self.classes = ['goods', 'pallet']  # only one foreground class here

        # net_name = 'yolo3_darknet53_custom'
        # params_fname = './data/models/yolo3_darknet53_custom_best.params'
        # im_loader = data.transforms.presets.yolo.load_test

        # net_name = 'faster_rcnn_resnet101_v1d_custom'
        # params_fname = '../models/2nd-no_mixup_faster_rcnn_resnet101_v1d_custom_best.params'
        # self.im_loader = data.transforms.presets.rcnn.load_test

        # self.ctx = mx.gpu(0)
        # self.net = model_zoo.get_model(net_name, pretrained_base=True, classes=self.classes)
        # self.net.load_parameters(params_fname)
        # self.net.collect_params().reset_ctx(self.ctx)
        self.cam = None
        self.stopFlag = False
        self.K = None
        self.D = None
        self.xi = None
        self.PerspectiveMatrix = None
        self.image_width = 1920
        self.image_height = 1080
        self.mapx = None
        self.mapy = None
        # self.fix_distortion = fix_distortion
        # self.out = None
        # self.need_record = need_record
        with open(os.path.join('cfg', camera_para_file), 'r') as stream:
            try:
                dict_1 = yaml.load(stream)
                self.cam_config = dict_1
                
                # self.camera_matrix = np.array(dict_1['mtx'])
                # print self.camera_matrix
                # self.distortion = np.array(dict_1['dist'])
                # print self.distortion
                self.video_source = dict_1['rtsp_stream']
                self.scale = dict_1['chessboard']['scale']
                self.board_w = dict_1['chessboard']['board_w']
                self.board_h = dict_1['chessboard']['board_h']
                self.square = dict_1['chessboard']['square']
            except yaml.YAMLError as exc:
                print(exc)
        if camModel == 'pinhole':
            self.pinholeParamPrepare(alpha=1.0)
        elif camModel == 'Mei':
            self.MeiParamPrepare()
        elif camModel == 'Scaramuzza':
            self.ScaramuzzaParamPrepare(alpha=2.1)            
        else:
            print'[ERROR] BAD CAMERA MODEL'

    def cameraSwitch(self):
        self.cam = cv2.VideoCapture(self.video_source)
        # self.cam.set(3, 1280)
        # self.cam.set(4, 720)
        start = datetime.datetime.now()
        while True:
            self.cam.read()
            end = datetime.datetime.now()
            if (end - start).total_seconds() > 1:
                break
        self.queue = mp.Queue(maxsize=2)
        imgreadthread = threading.Thread(target=self.streamworker)
        imgreadthread.start()

    def pinholeParamPrepare(self, alpha=1.0):
        k1 = float(self.cam_config['distortion_parameters']['k1'])
        k2 = float(self.cam_config['distortion_parameters']['k2'])
        p1 = float(self.cam_config['distortion_parameters']['p1'])
        p2 = float(self.cam_config['distortion_parameters']['p2'])
        k3 = 0.0
        fx = float(self.cam_config['projection_parameters']['fx'])
        fy = float(self.cam_config['projection_parameters']['fy'])
        cx = float(self.cam_config['projection_parameters']['cx'])
        cy = float(self.cam_config['projection_parameters']['cy'])       
        self.K =  np.array([[  fx,   0.0,   cx],
            [  0.0,   fy,   cy],
            [  0.0,   0.0,   1.0]])
        self.D = np.array([[k1,  k2,  p1,  p2, k3]])

        self.image_width  = int(self.cam_config['image_width'])
        self.image_height = int(self.cam_config['image_height'])
        inputImageSize  = (self.image_width, self.image_height)
        outputImageSize = (self.image_width, self.image_height)
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D, inputImageSize, alpha, outputImageSize)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, newcameramtx, (self.image_width, self.image_height), 5) 
        

    def MeiParamPrepare(self):
        xi = np.array(self.cam_config['mirror_parameters']['xi'])
        k1 = float(self.cam_config['distortion_parameters']['k1'])
        k2 = float(self.cam_config['distortion_parameters']['k2'])
        p1 = float(self.cam_config['distortion_parameters']['p1'])
        p2 = float(self.cam_config['distortion_parameters']['p2'])
        gamma1 = float(self.cam_config['projection_parameters']['gamma1'])
        gamma2 = float(self.cam_config['projection_parameters']['gamma2'])
        u0 =     float(self.cam_config['projection_parameters']['u0'])
        v0 =     float(self.cam_config['projection_parameters']['v0'])
     
        self.K =  np.array([[  gamma1,   0,   u0],
                            [  0,   gamma2,   v0],
                            [  0,   0,   1]])
        self.D = np.array([[k1, k2, p1, p2]])
        self.xi = xi

    def undistort_Mei(self, verbose=True):             
        img = self.queue.get().copy()
        if img is None:
            return None

        h,  w = img.shape[:2]
        gamma1 = self.K[0,0]
        gamma2 = self.K[1,1]
        u0 = self.K[0,2]
        v0 = self.K[1,2]
        flag = cv2.omnidir.RECTIFY_PERSPECTIVE
        # flag = cv2.omnidir.RECTIFY_CYLINDRICAL
        # flag = cv2.omnidir.RECTIFY_LONGLATI
        # flag = cv2.omnidir.RECTIFY_STEREOGRAPHIC
        Knew =  np.array([[  gamma1/4.0,   0,   u0], #change according to diff camera
                        [  0,   gamma2/4.0,   v0],
                        [  0,   0,   1]])

        dst = cv2.omnidir.undistortImage(img, self.K, self.D, self.xi, flag, Knew=Knew, new_size=(w,h))
        # dst = cv2.omnidir.undistortImage(img, self.K, self.D, self.xi, flag)
        return dst

    def ScaramuzzaParamPrepare(self, alpha=2.0):
        import sys
        import os
        sys.path.append(os.path.abspath('./scaramuzza_model'))
        from ocam_functions import get_ocam_model, create_perspective_undistortion_LUT

        sf = alpha # Parameter that affect the result of Scaramuzza's undistortion. Try to change it to see how it affects the result

        path_ocam = "./scaramuzza_model/Data/ocam_calibration.txt"
        o = get_ocam_model(path_ocam)
        mapx_persp, mapy_persp = create_perspective_undistortion_LUT(o, sf)
        mapx_persp_32 = mapx_persp.astype('float32')
        mapy_persp_32 = mapy_persp.astype('float32')
        self.mapx = mapx_persp_32
        self.mapy = mapy_persp_32

    def undistort(self, image, verbose=True):             
        # image = self.queue.get().copy()
        # if image is None:
        #     print'[ERROR] queue is empty'
        #     return None

        dst = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_CUBIC) #INTER_CUBIC/INTER_LINEAR
        return dst


    def streamworker(self):
        while not self.stopFlag:
            is_opened, temp = self.cam.read()
            if temp is not None:
                # if self.fix_distortion:
                #     temp = self.undistortion(temp)
                #     undist_img = self.undistort_pinhole(alpha=1.0)
    

                # if self.out is None and self.need_record:
                #     h, w = temp.shape[:2]
                #     print('record: w: {0}, h:{1}'.format(w,h))
                #     fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
                #     self.out = cv2.VideoWriter(
                #         'output_{0}_{1}.avi'.format(
                #             self.id, datetime.datetime.now().strftime('%Y%m%d%H%M%S')), fourcc, 20.0, (w, h))

                self.queue.put(temp) if is_opened else None
                self.queue.get() if self.queue.qsize() > 1 else None

            time.sleep(0.01)
        if self.cam is not None:
            self.cam.release()
        # if self.out is not None:
        #     self.out.release()
    def homoTrans(self, image):
        undist_img = self.undistort(image)
        # undist_img = self.undistort_Mei()
        # img_crop = cv2.resize(undist_img, (int(undist_img.shape[1]/2), int(undist_img.shape[0]/2)), interpolation=cv2.INTER_CUBIC) #INTER_NEAREST
        img_crop = undist_img
        img_homo = self.VideoTrans(img_crop)

        # camHeight,camWidth, unit = getHomoMatrix_usingVirtualPoint(img_crop, debug=True)
        # if PerspectiveMatrix is not None:
        #     img_homo = VideoTrans(img_crop, PerspectiveMatrix, camWidth, camHeight)
        #     cv2.imshow('homo', img_homo)
        return img_homo



    # def process(self, use_nn=True):

    #     box_color = {0:(0,255,0), 1:(0, 0, 255)}
    #     classes = {0:Message.DetectObj.DETECT_OBJ_TYPE_CARGO,
    #             1:Message.DetectObj.DETECT_OBJ_TYPE_CONTAINER }
    #     source_image = self.queue.get().copy()
    #     if source_image is None:
    #         return None
    #     if not use_nn:
    #         return (None, {'result':source_image, 'src':source_image})
    #     # mxnet is RGB
    #     # opencv is BGR
    #     net_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    #     # transform from array to NDArray and resize to match import size
    #     x, test_image = data.transforms.presets.rcnn.transform_test(mx.nd.array(net_image), 720, 1280)
    #     # print('Shape of pre-processed image:', x.shape, net_image.shape, test_image.shape, mor.shape)
    #     class_IDs, scores, bounding_boxs = self.net(x.copyto(self.ctx))
    #     # print "class_IDS"
    #     # print class_IDs
    #     # print "scoress"
    #     # print scores
    #     # print 'bounding_boxs'
    #     # print bounding_boxs
    #     objs0 = []
    #     list_box = bounding_boxs[0].asnumpy()
    #     result_img = test_image.copy()
    #     for i in range(len(list_box)):
    #         bbox = list_box[i]
    #         score = scores[0][i].asnumpy()[0]
    #         class_ID = int(class_IDs[0][i].asnumpy()[0])

    #         if score > 0.9:
    #             printLog('{0} {1} {2}'.format(bbox, score, class_ID))
    #             cv2.rectangle(test_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=box_color[class_ID], thickness=2)
    #             obj0 = Message.DetectObj(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), classes[class_ID])
    #             objs0.append(obj0)

    #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    #     _, result_img = cv2.imencode('.jpg', result_img, encode_param)
    #     msg0 = Message(self.id, result_img, objs0)

    #     ########################## plot with viz
    #     # fig = Figure()
    #     # canvas = FigureCanvas(fig)
    #     # ax = fig.gca()
    #     # ax.axis('off')
    #     # ax.margins(x=0,y=0)
    #     # ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=classes, ax=ax)
    #     # # ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0])
    #     # # plt.show()
    #     # # plt.close()
    #     # canvas.draw()
    #     # width, height = fig.get_size_inches() * fig.get_dpi()
    #     # assert int(width) == 640 and int(height) == 480
    #     # frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    #     # plt.close()
    #     test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    #     return (msg0, {'result':test_image, 'src':source_image})

    # def record(self, img):
    #     if self.out is not None and self.need_record:
    #         self.out.write(img)
    #     else:
    #         print('not ready for record')

    def videoSourceRelease(self):
        self.stopFlag = True

    def VideoTrans(self, im):
        im = cv2.warpPerspective(im, self.PerspectiveMatrix,(im.shape[1],im.shape[0]),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        return im

    def getHomoMatrix_usingVirtualPoint(self, image, debug=False):
        unit=None
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        image = self.undistort(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width, chnl = np.shape(image)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.board_w, self.board_h), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)


            x0 = (width/2.0 - 1) - self.scale*(self.board_w/2.0 - 0.5)
            x1 = (width/2.0 - 1) + self.scale*(self.board_w/2.0 - 0.5)
            y0 = (height/2.0 - 1) - self.scale*(self.board_h/2.0 - 0.5)
            y1 = (height/2.0 - 1) + self.scale*(self.board_h/2.0 - 0.5)
            cmDistance = (self.board_w - 1.0)*self.square
            pixelDistance = (x1 - x0)
            unit = cmDistance / pixelDistance
            if debug == True:
                img_disp = image.copy()
                print 'current frame theoretical unit = ', unit
                # Draw and display the corners
                cv2.drawChessboardCorners(img_disp, (self.board_w, self.board_h), corners2, ret)
                cv2.imshow('corner image', img_disp)
                corners2 = np.squeeze(corners2)
            objPts = np.float32([[x0,y0], [x1,y0], [x0,y1], [x1,y1]])
            imgPts = np.float32([corners2[0],corners2[self.board_w-1], corners2[(self.board_h-1)*self.board_w],corners2[(self.board_h-1)*self.board_w + self.board_w-1]])

            self.PerspectiveMatrix = cv2.getPerspectiveTransform(objPts, imgPts)

        else:
            print('[ERROR] not enough corners')

        return height, width, unit

