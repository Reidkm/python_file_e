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
from cameraClass import *
from calibration import *





def main():
    # getting things ready
    yamlFilePath = 'calibration.yaml'
    args = Options().parse()
    height = args.demo_size
    width = 640
    # video_source = args.video_source
    flip_flag = False
    unit = None
    stage_1_window = 'stage_1'
    stage_2_window = 'stage_2'
    stage_3_window = 'stage_3'
    barsWindow = stage_2_window
    scaleBar = 'scaleBar'

    # configuration from yaml file
    # cali_config = loadYamlFile(yamlFilePath)

    camera_type = args.camera_type
     
    if camera_type == 'left_down_camera' or camera_type == 'right_down_camera':
        CalibrCam = DownCamera(camera_type, yamlFilePath)
    elif camera_type == 'front_camera' :
        CalibrCam = FrontCamera(camera_type, yamlFilePath)
    elif camera_type == 'back_camera':
        CalibrCam = BackCamera(camera_type, yamlFilePath)

    # from front_camera import FrontCamera
    # frontCamera = FrontCamera('front_camera',
    #                             cfg=self.calibration_info['front_camera'])

    # if args.camera_type == 'front_camera' : #or args.camera_type == 'back_camera'
    #     p1 = [0,0];p2 = [0,0];p3 = [0,0];p4 = [0,0]
    # else:
    #     p1 = cali_config[args.camera_type]['chessboard']['point'][0].split(',')
    #     p2 = cali_config[args.camera_type]['chessboard']['point'][1].split(',')
    #     p3 = cali_config[args.camera_type]['chessboard']['point'][2].split(',')
    #     p4 = cali_config[args.camera_type]['chessboard']['point'][3].split(',')

    camera_center     = CalibrCam.cfg[args.camera_type]['center_point']# used for camera mounting
    chessboard_size   = CalibrCam.cfg[args.camera_type]['chessboard']['spec'].split('x')
    PerspectiveHeight = CalibrCam.cfg[args.camera_type]['chessboard']['PerspectiveHeight']
    scale = PerspectiveHeight
    print 'use calibration board : ',CalibrCam.cfg[args.camera_type]['chessboard']['type']
    notice_1 = CalibrCam.cfg[args.camera_type]['notice_1']
    notice_2 = CalibrCam.cfg[args.camera_type]['notice_2']
    if notice_1 != 'none':
        print notice_1
    if notice_2 != 'none':
        print notice_2


    # x:camera_center[0] y:camera_center[1]
    # camera_center = args.camera_center.split(',')
    # w: chessboard_size[0]
    # h: chessboard_size[1]
    # PerspectiveHeight: chessboard_size[2]
    # chessboard_size = args.chessboard_size.split(',')

    # cam = cv2.VideoCapture(int(video_source))
    cam = cv2.VideoCapture(CalibrCam.cfg[CalibrCam.id]['source'])
    cam.set(3, width)
    cam.set(4, height)
    stopflag = False
    stage = 1

    board_w = int(chessboard_size[0]) #chessboard width
    board_h = int(chessboard_size[1]) #chessboard height
    square  = float(chessboard_size[2]) # chessboard square size
    # PerspectiveHeight = int(chessboard_size[2])
    # chessboard_position = args.chessboard_position.split(',')

    # p1_x = int(p1[0])
    # p1_y = int(p1[1])

    # p2_x = int(p2[0])
    # p2_y = int(p2[1])

    # p3_x = int(p3[0])
    # p3_y = int(p3[1])

    # p4_x = int(p4[0])
    # p4_y = int(p4[1])

    while not stopflag:
        ret, image = cam.read()
        # image = cv2.imread(CalibrCam.cfg[args.camera_type]['chessboard']['picture_name']) #reads BGR; only for testing
        if image is None:
            continue

#        if flip_flag == True:
#            image = cv2.flip(image, 1)

        imgTrans = image.copy()
        image_line = image.copy()
        cimg = image.copy()
        for i in range(0,len(camera_center)):
            image_line[int(camera_center[i].split(',')[1]) - 1, :, :] = STANDER_COLOR[i]
            image_line[:, int(camera_center[i].split(',')[0]) - 1, :] = STANDER_COLOR[i]

        if stage == 1: # mounting camera

            cv2.imshow(stage_1_window, image_line)

        elif stage == 2: # homo
            cv2.destroyWindow(stage_1_window)
            scale = CalibrCam.scaleMeasure(scaleBar, stage_2_window)

#            camHeight,camWidth,PerspectiveMatrix = getHomoMatrix(board_w,board_h,image,PerspectiveHeight)
            camHeight,camWidth,PerspectiveMatrix, unit = getHomoMatrix_usingVirtualPoint(board_w, board_h, square, image, scale,True)
            CalibrCam.yamlPrint()

            try :

                # src = VideoTrans(image,PerspectiveMatrix,camWidth,camHeight)
                # src = image

                # cv2.line(src,(p1_x,p1_y),(p2_x,p2_y),(0,0,255),1)
                # cv2.line(src,(p2_x,p2_y),(p3_x,p3_y),(0,0,255),1)
                # cv2.line(src,(p3_x,p3_y),(p4_x,p4_y),(0,0,255),1)
                # cv2.line(src,(p4_x,p4_y),(p1_x,p1_y),(0,0,255),1)

                image_line_trans = VideoTrans(imgTrans,PerspectiveMatrix,camWidth,camHeight)
                ImBackMapping = VideoTrans(image_line_trans,PerspectiveMatrix,camWidth,camHeight, flags=cv2.INTER_LINEAR)
                # image_line_trans[int(camera_center[1]), :, :] = (0,0,255)
                # image_line_trans[:, int(camera_center[0]), :] = (0,0,255)
                # image_line[int(camera_center[1]), :, :] = (0,0,255)
                # image_line[:, int(camera_center[0]), :] = (0,0,255)

                # for i in range(0,len(camera_center)):
                image_line_trans[int(camHeight/2 - 1), :, :] = STANDER_COLOR[i]
                image_line_trans[:, int(camWidth/2 - 1), :] = STANDER_COLOR[i] #make sure this line is aligned with the middle of chess board

                # showimg = np.concatenate((image_line, image_line_trans, src),axis=1)
                showimg = np.concatenate((image_line, image_line_trans, ImBackMapping),axis=1)
            except :
                showimg = image

            cv2.imshow(stage_2_window, showimg)
        elif stage == 3: #view
            img_3 = image.copy()
            cv2.destroyWindow(stage_2_window)
            img_file = cv2.imread('{0}.png'.format(args.camera_type))
            if img_file is None:
                print '[ERROR] no such file:  {0}.png '.format(args.camera_type)
            else:
#                camHeight,camWidth,PerspectiveMatrix = getHomoMatrix(board_w,board_h,img_file,PerspectiveHeight)
                camHeight,camWidth,PerspectiveMatrix,_ = getHomoMatrix_usingVirtualPoint(board_w, board_h, square, img_file, PerspectiveHeight, False)
                img_homo = VideoTrans(img_3,PerspectiveMatrix, camWidth, camHeight)
                # cv2.imshow('homo_view', img_homo)
                CalibrCam.OnMouseMeasure('homo_view', img_homo)


        key = cv2.waitKey(1)
        if key == ord('1'):
            stage = 1
            print 'Enter stage 1 '
            print 'You can adjust the camera mounting angle '

        elif key == ord('2'):
            stage = 2
            barPrepare(CalibrCam, scaleBar, barsWindow)
            print 'Enter stage 2'

        elif key == ord('3'):
            stage = 3
            print 'Enter stage 3'

        elif key == ord('s'):
            print 'Saving image'
            import time
#            timestamp = int(time.time())
            cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
            imageio.imwrite('{0}.png'.format(args.camera_type), cimg) #save image
            time.sleep(1)


            img_file = cv2.imread('{0}.png'.format(args.camera_type)) #reload the image
            if img_file is None:
                print '[ERROR] no such file:  {0}.png '.format(args.camera_type)
            # cv2.imshow('img_file',img_file)

            # auto unit measurement
            CalibrCam.cfg[args.camera_type]['unit'] = unit
            CalibrCam.saveYamlFile()
            CalibrCam.loadYamlFile() # reload 

            CalibrCam.ManualMeasure()


            get_Angle_Unit(board_w,board_h, square, img_file,PerspectiveHeight, camera_type, scale)

        elif key == 27 :
            print 'Ese'
            stopflag = True
        else :
            pass



if __name__=='__main__':
    main()
