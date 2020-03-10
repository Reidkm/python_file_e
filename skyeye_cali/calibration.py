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


STANDER_COLOR = [[0,0,255],[255,0,0],[0,255,0],[255,255,0],[0,255,255],[255,0,255]]

###########################################
###########################################
def getdUnit(img, board_w, board_h, square, debug):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    unit = None
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        if debug == True:
            cv2.drawChessboardCorners(img, (board_w,board_h), corners2,ret)
            cv2.imshow('findUnit',img)

        corners2 = np.squeeze(corners2)
        # if debug == True:
            # print 'p1 = ',corners2[0]
            # print 'p2 = ',corners2[board_w-1]
        # # pixelDistance = np.linalg.norm(corners2[0] - corners2[board_w-1])#distance between 2 points
        # pixelDistance = np.linalg.norm(corners2[0] - corners2[board_w*(board_h-1)])#distance between 2 points
        # if debug == True:
            # print 'pixelDistance =  ',pixelDistance
        # # cmDistance = (board_w - 1) * square
        # cmDistance = (board_h - 1) * square
        # if debug == True:
            # print 'square num=  ',(board_w - 1)
            # print 'cmDistance =  ',cmDistance

        # unit = cmDistance / pixelDistance
        # if debug == True:
            # print 'unit =  ',unit

        unit_x = ((board_w - 1) * square) / (np.linalg.norm(corners2[0] - corners2[board_w-1]))
        unit_y = ((board_h - 1) * square) / (np.linalg.norm(corners2[0] - corners2[board_w*(board_h - 1)]))

    else:
        print 'not enough corners !'

    return unit_x, unit_y


def getAngle(im_i,debug):
    hsv = cv2.cvtColor(im_i, cv2.COLOR_BGR2HSV)

	# construct a mask for the spepcific color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask_1 = cv2.inRange(hsv, (0,43,46), (10,255,255))
    mask_2 = cv2.inRange(hsv, (156,43,46), (180,255,255))
    mask = cv2.bitwise_or(mask_1,mask_2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    if debug is True:
        cv2.imshow('mask image',mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[1]
    center = None

	# only proceed if at least one contour was found
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if center is not None:
            cv2.circle(im_i, center, 5, (0, 0, 255), -1)

        c2 = min(cnts, key=cv2.contourArea)
        M2 = cv2.moments(c2)
        center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
        if center2 is not None:
            cv2.circle(im_i, center2, 5, (0, 0, 255), -1)
            middle_point = ((center[0] + center2[0])/2,(center[1] + center2[1])/2)
            print 'middle point = ',middle_point
            cv2.circle(im_i, middle_point, 5, (0, 255, 255), -1)

            num = abs(center[1] - center2[1])
            denom = abs(center[0] - center2[0])
#            angles = np.arctan(num/denom)
            angles = math.atan2(num,denom)
            inner_angle = np.degrees(angles)

            print 'inner_angle = ', inner_angle


    if debug is True:
        cv2.imshow("Frame", im_i)

    return inner_angle, middle_point

def get_Angle_Unit(board_w,board_h, square, image, PerspectiveHeight, camera_type, scale):

    # camHeight,camWidth,PerspectiveMatrix = getHomoMatrix(board_w,board_h,image,PerspectiveHeight)
    camHeight,camWidth,PerspectiveMatrix, _ = getHomoMatrix_usingVirtualPoint(board_w, board_h, square, image, scale, True)
    img_homo = VideoTrans(image,PerspectiveMatrix,camWidth,camHeight)

    # img_dis = img_homo.copy()
#    cv2.imshow('img_dis', img_dis)

    # unit = None

    if camera_type == 'left_back_camera' or camera_type == 'right_back_camera' :
       angle, middle_point = getAngle(img_homo,True)
    unit_x, unit_y = getdUnit(img_homo, board_w, board_h, square, True)
    print 'unit_x =  ',unit_x
    print 'unit_y =  ',unit_y

    return unit_x,unit_y

def VideoTrans(im,PerspectiveMatrix,width,height, flags=cv2.WARP_INVERSE_MAP ): #flags=cv2.INTER_LINEAR
    im = cv2.warpPerspective(im, PerspectiveMatrix, (width,height), flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return im

def getHomoMatrix(board_w,board_h,image,PerspectiveHeight):
    img_disp = image.copy()
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height,width,chnl = np.shape(image)
    PerspectiveMatrix = None

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_disp, (board_w,board_h), corners2,ret)
        cv2.imshow('corner image',img_disp)


        x0 = 0
        x1 = board_w-1
        y0 = 0
        y1 = board_h-1
        corners2 = np.squeeze(corners2)
        objPts = np.float32([[x0,y0],[x1,y0],[x0,y1],[x1,y1]])
        imgPts = np.float32([corners2[0],corners2[board_w-1],corners2[(board_h-1)*board_w],corners2[(board_h-1)*board_w + board_w-1]])

        PerspectiveMatrix = cv2.getPerspectiveTransform(objPts, imgPts)

        PerspectiveMatrix[2,2] = PerspectiveHeight

        w = width
        h = height
        PerspectiveImg = VideoTrans(image,PerspectiveMatrix,w,h)

        PerspectiveImg[:,int(round(w/2-1)),:] = [0,0,255]
        PerspectiveImg[int(round(h/2-1)),:,:] = [0,0,255]

        PerspectiveImg[:,int(round(396-1)),:] = [0,255,255]
        PerspectiveImg[int(round(304-1)),:,:] = [0,255,255]
#        cv2.imshow('PerspectiveImg',PerspectiveImg)
    else:
        print('[ERROR] not enough corners')

    return height,width,PerspectiveMatrix

def getHomoMatrix_usingVirtualPoint(board_w, board_h, square_size, image, scale, debug=False):
    # img_disp = image.copy()
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height,width,chnl = np.shape(image)
    PerspectiveMatrix = None
    unit=None

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        # cv2.drawChessboardCorners(img_disp, (board_w,board_h), corners2,ret)
        # cv2.imshow('corner image',img_disp)

        x0 = (width/2.0 - 1) - scale*(board_w/2.0 - 0.5)
        x1 = (width/2.0 - 1) + scale*(board_w/2.0 - 0.5)
        y0 = (height/2.0 - 1) - scale*(board_h/2.0 - 0.5)
        y1 = (height/2.0 - 1) + scale*(board_h/2.0 - 0.5)
        cmDistance = (board_w - 1.0)*square_size
        pixelDistance = (x1 - x0)
        unit = cmDistance/pixelDistance
        if debug == True:
            print 'current frame theoretical unit = ',unit
        corners2 = np.squeeze(corners2)
        objPts = np.float32([[x0,y0],[x1,y0],[x0,y1],[x1,y1]])
        imgPts = np.float32([corners2[0],corners2[board_w-1],corners2[(board_h-1)*board_w],corners2[(board_h-1)*board_w + board_w-1]])

        PerspectiveMatrix = cv2.getPerspectiveTransform(objPts, imgPts)

        # w = width
        # h = height
        # PerspectiveImg = VideoTrans(image,PerspectiveMatrix,w,h)

        # PerspectiveImg[:,int(round(w/2-1)),:] = [0,0,255]
        # PerspectiveImg[int(round(h/2-1)),:,:] = [0,0,255]

        # PerspectiveImg[:,int(round(396-1)),:] = [0,255,255]
        # PerspectiveImg[int(round(304-1)),:,:] = [0,255,255]
#        cv2.imshow('PerspectiveImg',PerspectiveImg)
    else:
        print('[ERROR] not enough corners')

    return height,width,PerspectiveMatrix, unit

def drawLine(img, line, color=(255,255,255), thickness=1):
    cv2.line(img,(int(line.p1.x),int(line.p1.y)),(int(line.p2.x),int(line.p2.y)),color,thickness)





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
                # image_line_trans[int(camera_center[1]), :, :] = (0,0,255)
                # image_line_trans[:, int(camera_center[0]), :] = (0,0,255)
                # image_line[int(camera_center[1]), :, :] = (0,0,255)
                # image_line[:, int(camera_center[0]), :] = (0,0,255)

                # for i in range(0,len(camera_center)):
                image_line_trans[int(camHeight/2 - 1), :, :] = STANDER_COLOR[i]
                image_line_trans[:, int(camWidth/2 - 1), :] = STANDER_COLOR[i] #make sure this line is aligned with the middle of chess board

                # showimg = np.concatenate((image_line, image_line_trans, src),axis=1)
                showimg = np.concatenate((image_line, image_line_trans),axis=1)
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

def barPrepare(CalibrCam, scaleBar, barsWindow):
    scale = int(CalibrCam.cfg[CalibrCam.id]['chessboard']['PerspectiveHeight'])
    
    cv2.namedWindow(barsWindow, flags = cv2.WINDOW_AUTOSIZE) # create window for the slidebars
    cv2.createTrackbar(scaleBar, barsWindow, 10, 100, nothing)# create the sliders
    cv2.setTrackbarPos(scaleBar, barsWindow, scale)# set initial values for sliders

    # 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass


if __name__=='__main__':
    main()
