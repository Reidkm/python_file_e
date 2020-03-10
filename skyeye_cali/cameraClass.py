#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from option import Options
import cv2
import numpy as np
import imageio
import yaml
import math
from collections import OrderedDict
import order_pyyaml.yamlparser as yamlparser
import os
# left_down_camera & right_down_camera
class DownCamera(object):
    def __init__(self, name, yamlFilePath):
        # threading.Thread.__init__(self)
        self.id = name
        self.cfg = None
        self.yamlFilePath = yamlFilePath

# variables that need to be chaged in every calibration
        self.unit = None
        self.cen_point = None       #coordinate of red dot 'O' on the calibration board
        self.center_x_length = 30   #constant
        self.center_y_length = None #needs manual measurement;left/right side of forklift --> lane line distance

        self.loadYamlFile()

    def scaleMeasure(self, scaleBar, barsWindow, debug=False):
        scale = cv2.getTrackbarPos(scaleBar, barsWindow)# read trackbar positions for all
        self.cfg[self.id]['chessboard']['PerspectiveHeight'] = scale
        if debug == True:
            print 'scale = ',scale

        return scale

    def yamlPrint(self):
        print 'unit in calibraton file =         {0}'.format(self.cfg[self.id]['unit'])
        print 'center_y_length in calibraton file ={0}'.format(self.cfg[self.id]['center_y_length'])
        print 'PerspectiveHeight in calibraton file = {0}'.format(self.cfg[self.id]['chessboard']['PerspectiveHeight'])
        print ''

    def ManualMeasure(self):
        filename = os.path.dirname(__file__) + "/"+"reference_image"+"/" + str(self.id) +"_center_y_length" + ".bmp"
        im = cv2.imread(filename)
        if im is None:
            print '[ERROR] missing reference image !'
        cv2.imshow("manual_measurement", im)
        cv2.waitKey(1000)
        print 'refer to the image(manual_measurement),then measure and fill in below(Remenber the unit is cm), then enter . '
        center_y_length = float(input())
        self.cfg[self.id]['center_y_length'] = center_y_length
        self.saveYamlFile()
        self.loadYamlFile() # reload 
        cv2.destroyWindow("manual_measurement")


    def OnMouseMeasure(self, WindowName, img_homo):
        self.ShowCenPoint(img_homo)
        cv2.imshow(WindowName, img_homo)
        cv2.setMouseCallback(WindowName, self.GetPosCoordinate)

        
    def ShowCenPoint(self, image):
        cen_point = self.cfg[self.id]['cen_point'].split(',')
        if cen_point != None :
            cv2.circle(image, (int(cen_point[0]),int(cen_point[1])), 3, (0, 0, 255), -1)
        

    def GetPosCoordinate(self, event, x, y, flags, param):
        # if event==cv2.EVENT_MOUSEMOVE:
            # print "pixel cordinate [x,y]=[ {0},{1} ]".format(x,y)
        if event==cv2.EVENT_LBUTTONUP:#mouse event
            # <<"触发左键抬起事件"<<;
            self.cen_point = (x,y)
            print "pixel cordinate [x,y]=[ {0},{1} ]".format(x,y)
            self.cfg[self.id]['cen_point'] = '{0},{1}'.format(int(x),int(y))

            self.saveYamlFile()
            self.loadYamlFile() # reload 

            
            # events=[i for i in dir(cv2) if 'EVENT' in i]   # print all mouse event
            # print('=====All mouse event list are here:  =====')
            # print(events)
            # print('============================')

    def loadYamlFile(self):
        # with open('calibration.yaml','r') as stream:
        #     try:
        #         cali_config = yaml.load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        cali_config = yamlparser.ordered_yaml_load(self.yamlFilePath)
        self.cfg = cali_config

    def saveYamlFile(self):
        dumpfile = open(self.yamlFilePath, 'w')
        self.cfg[self.id]['center_x_length'] = self.center_x_length
        ret = yamlparser.ordered_yaml_dump(self.cfg, dumpfile, default_flow_style=False)
        dumpfile.close()
        print "[INFO] save yaml file "



class BackCamera(object):
    def __init__(self, name, yamlFilePath):
        self.id = name
        self.cfg = None
        self.yamlFilePath = yamlFilePath

# variables that need to be chaged in every calibration
        self.unit = None
        self.cen_point = [319,239] #constant;center point of pallet's front side
        self.center_z_length = 60  #constant;centimeter;center-->fork tips distance

        self.loadYamlFile()

    def scaleMeasure(self, scaleBar, barsWindow, debug=False):
        scale = cv2.getTrackbarPos(scaleBar, barsWindow)# read trackbar positions for all
        self.cfg[self.id]['chessboard']['PerspectiveHeight'] = scale
        if debug == True:
            print 'scale = ',scale

        return scale

    def yamlPrint(self):
        print 'unit in calibraton file =         {0}'.format(self.cfg[self.id]['unit'])
        print 'center_z_length in calibraton file ={0}'.format(self.cfg[self.id]['center_z_length'])
        print 'cen_point in calibration file = {0},{1}'.format(int(self.cfg[self.id]['cen_point'].split(',')[0]), int(self.cfg[self.id]['cen_point'].split(',')[1]))
        print 'PerspectiveHeight in calibraton file = {0}'.format(self.cfg[self.id]['chessboard']['PerspectiveHeight'])
        print ''

    def ManualMeasure(self):
        pass


    def OnMouseMeasure(self, WindowName, img_homo):
        cv2.imshow(WindowName, img_homo)

    def loadYamlFile(self):
        # with open('calibration.yaml','r') as stream:
        #     try:
        #         cali_config = yaml.load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        cali_config = yamlparser.ordered_yaml_load(self.yamlFilePath)
        self.cfg = cali_config

    def saveYamlFile(self):
        dumpfile = open(self.yamlFilePath, 'w')
        self.cfg[self.id]['cen_point'] = '{0},{1}'.format(self.cen_point[0],self.cen_point[1])
        self.cfg[self.id]['center_z_length'] = self.center_z_length
        ret = yamlparser.ordered_yaml_dump(self.cfg, dumpfile, default_flow_style=False)
        dumpfile.close()
        print "[INFO] save yaml file "

class FrontCamera(object):
    def __init__(self, name, yamlFilePath):
        self.id = name
        self.cfg = None
        self.yamlFilePath = yamlFilePath

# variables that need to be chaged in every calibration
        self.unit = None
        self.baseline_y = None #needs manual measurement;front side of forklift-->bottom line of video distance

        self.loadYamlFile()

    def scaleMeasure(self, scaleBar, barsWindow, debug=False):
        scale = cv2.getTrackbarPos(scaleBar, barsWindow)# read trackbar positions for all
        self.cfg[self.id]['chessboard']['PerspectiveHeight'] = scale
        if debug == True:
            print 'scale = ',scale

        return scale


    def yamlPrint(self):
        print 'unit in calibraton file =         {0}'.format(self.cfg[self.id]['unit'])
        print 'baseline_y in calibraton file = {0}'.format(self.cfg[self.id]['baseline_y'])
        print 'PerspectiveHeight in calibraton file = {0}'.format(self.cfg[self.id]['chessboard']['PerspectiveHeight'])
        print ''

    def ManualMeasure(self):
        filename = os.path.dirname(__file__) + "/"+"reference_image"+"/" + "front_camera_center_z_length" + ".bmp"
        im = cv2.imread(filename)
        if im is None:
            print '[ERROR] missing reference image !'
        cv2.imshow("manual_measurement", im)
        cv2.waitKey(1000)
        print 'refer to the image(manual_measurement),then measure and fill in below(Remenber the unit is cm), then enter . '
        baseline_y = float(input())
        self.cfg[self.id]['baseline_y'] = baseline_y
        self.saveYamlFile()
        self.loadYamlFile() # reload 
        cv2.destroyWindow("manual_measurement")

    def OnMouseMeasure(self, WindowName, img_homo):
        cv2.imshow(WindowName, img_homo)

    def loadYamlFile(self):
        # with open('calibration.yaml','r') as stream:
        #     try:
        #         cali_config = yaml.load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        cali_config = yamlparser.ordered_yaml_load(self.yamlFilePath)
        self.cfg = cali_config

    def saveYamlFile(self):
        dumpfile = open(self.yamlFilePath, 'w')
        ret = yamlparser.ordered_yaml_dump(self.cfg, dumpfile, default_flow_style=False)
        dumpfile.close()
        print "[INFO] save yaml file "



class Camera(object):
    def __init__(self, name, cfg):
        # threading.Thread.__init__(self)
        self.id = name
        self.unit = None
        self.cfg = cfg
        self.src_cam_video_source = self.cfg['source']
        self.height = self.cfg['image_height']
        self.width = self.cfg['image_width']
        self.cam = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.m_lane = None
        homo_offset_x = self.cfg['x_offset']
        homo_offset_y = self.cfg['y_offset']
        self.unit = float(self.cfg['unit'])
        self.chessboard = self.cfg['chessboard']['picture_name']
        self.PerspectiveHeight = self.cfg['chessboard']['PerspectiveHeight']
        self.chessboard_spec = self.cfg['chessboard']['spec']
        self.model = self.cfg['nn']['model']
        self.model_tar = self.cfg['nn']['model_param']
        self.scale_h = self.unit
        self.scale_v = self.unit

        self.centerLine = Line(Point(320, 480), Point(320, 0))
        self.centerBottomPoint = Point(320, 480)
        self.fix_start = { 'x': 0 if homo_offset_x < 0 else homo_offset_x,
                           'y': 0 if homo_offset_y < 0 else homo_offset_y}

        self.fix_end = { 'x': homo_offset_x + self.width if homo_offset_x < 0 else self.width,
                         'y': homo_offset_y + self.height if homo_offset_y < 0 else self.height}

        self.org_start = {'x': homo_offset_x * (-1) if homo_offset_x < 0 else 0,
                          'y': homo_offset_y * (-1) if homo_offset_y < 0 else 0}

        self.org_end = {'x': self.width if homo_offset_x < 0 else self.width - homo_offset_x,
                        'y': self.height if homo_offset_y < 0 else self.height - homo_offset_y}

        self.m_lane = None
        self.PerspectiveMatrix = None
        self.prepareHomo()
        self.processInit()
        # self.left_distance = data_pool(5, 3, sys.maxint, 320)
        # self.right_distance = data_pool(5, 3, sys.maxint, 320)
        # self.top_distance = data_pool(5, 3, 100, 480)

    def cameraSwitch(self) :
        self.cameraInit()
        return True


    def viz(self, y_pred, filename):
        h, w = y_pred.shape
        # -- binary label
        # binary = y_pred.astype(np.uint8)
        # imageio.imwrite('binary_{}.png'.format(filename), binary)
        # -- colored image
        result_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        tp = np.where(y_pred == 1)
        result_rgb[tp[0], tp[1], :] = 0, 255, 0  #
        imageio.imwrite(filename, result_rgb)

    def cameraInit(self) :
        self.cam = cv2.VideoCapture(self.src_cam_video_source)
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        start = datetime.datetime.now()
        while True:
            self.cam.read()
            end = datetime.datetime.now()
            if (end - start).total_seconds() > 1 :
                break


    def prepareHomo(self) :
        # --- prepare the Hormography matrix
        # added by Matt

        #parameters
        board_w = int(self.chessboard_spec.split('x')[0])
        board_h = int(self.chessboard_spec.split('x')[1])
        H_Z = float(self.PerspectiveHeight)    #viewing height
        H_filename = os.path.join(os.getcwd(), 'H_matrix_src_image', self.chessboard)
        # end param

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)# termination criteria

        im1 = cv2.imread(H_filename) #reads BGR

        gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)# Find the chess board corners

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        else :
    	    print("[H matrix process] :findChessboardCorners() ERROR")

        x0 = 0
        x1 = board_w-1
        y0 = 0
        y1 = board_h-1
        corners2 = np.squeeze(corners2)
        objPts = np.float32([[x0,y0],[x1,y0],[x0,y1],[x1,y1]])
        imgPts = np.float32([corners2[0],corners2[board_w-1],corners2[(board_h-1)*board_w],corners2[(board_h-1)*board_w + board_w-1]])
        self.PerspectiveMatrix = cv2.getPerspectiveTransform(objPts, imgPts)
        self.PerspectiveMatrix[2,2] = H_Z
        # --- end the Hormography matrix

    def processInit(self) :
        CWD_PATH = os.getcwd()
        resume = os.path.join(CWD_PATH, 'model_params', self.model_tar)
        self.m_lane = LaneSeg(model=self.model, param=resume, n_classes=2)

    def run_demo(self, using_nn = True, mirror = False):
        return 0;

    def cameraRelease(self):
        self.cam.release()
