#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from option import Options
import cv2
import numpy as np
import imageio
import yaml

STANDER_COLOR = [[0,0,255],[0,255,0],[255,0,0],[255,255,0],[0,255,255],[255,0,255]]

def findUnit(img, board_w, board_h, square, debug):
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
        print 'p1 = ',corners2[0]
        print 'p2 = ',corners2[board_w-1]
        pixelDistance = np.linalg.norm(corners2[0] - corners2[board_w-1])#distance between 2 points
        if debug == True:
            print 'pixelDistance =  ',pixelDistance
        cmDistance = (board_w - 1) * square
        if debug == True:
            print 'square num=  ',(board_w - 1)
            print 'cmDistance =  ',cmDistance
            print 'type(cmDistance) = ',type(cmDistance)
            
        unit = cmDistance / pixelDistance
        if debug == True:
            print 'unit =  ',unit
    
    return unit
    
    
def VideoTrans(im,PerspectiveMatrix,width,height):
    im = cv2.warpPerspective(im, PerspectiveMatrix,(width,height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return im

def getHomoMatrix(board_w,board_h, square, image,PerspectiveHeight):
    tmpImg = image.copy()
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height,width,chnl = np.shape(image)
    PerspectiveMatrix = None
    
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)
    
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cornerImg = cv2.drawChessboardCorners(image, (board_w,board_h), corners2,ret)
        cv2.imshow('corner image',cornerImg)


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
        PerspectiveImg = VideoTrans(tmpImg,PerspectiveMatrix,w,h)
        
#        PerspectiveImg[:,int(round(w/2-1)),:] = [0,0,255]
#        PerspectiveImg[int(round(h/2-1)),:,:] = [0,0,255]
#        
#        PerspectiveImg[:,int(round(396-1)),:] = [0,255,255]
#        PerspectiveImg[int(round(304-1)),:,:] = [0,255,255]
        cv2.imshow('PerspectiveImg',PerspectiveImg)
        RedColorFilter(PerspectiveImg,True)            

        findUnit(PerspectiveImg, board_w, board_h, square, True)

    else:
        print('[ERROR] not enough corners')
    
    return height,width,PerspectiveMatrix

def drawLine(img, line, color=(255,255,255), thickness=1):
    cv2.line(img,(int(line.p1.x),int(line.p1.y)),(int(line.p2.x),int(line.p2.y)),color,thickness)

def RedColorFilter(im_i,debug):
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
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print('========center=',center)
 
		# only proceed if the radius meets a minimum size
        if center is not None:
            cv2.circle(im_i, center, 5, (0, 0, 255), -1)
            
        c2 = min(cnts, key=cv2.contourArea)
        M2 = cv2.moments(c2)
        center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
        print('========center2=',center2)
        if center2 is not None:
            cv2.circle(im_i, center2, 5, (0, 0, 255), -1)
            middle_point = ((center[0] + center2[0])/2,(center[1] + center2[1])/2)
            print 'middle point = ',middle_point
            cv2.circle(im_i, middle_point, 5, (0, 255, 255), -1)
            
            num = abs(center[1] - center2[1])
            denom = abs(center[0] - center2[0])
            angles = np.arctan(num/denom)
            inner_angle = np.degrees(angles)
            
            print 'inner_angle = ', inner_angle
 
        
    if debug is True:
        cv2.imshow("Frame", im_i)
        
    return im_i,center

if __name__=='__main__':
    # getting things ready
    args = Options().parse()
    height = args.demo_size
    width = 640
    video_source = args.video_source

    # configuration from yaml file
    cali_config = None
    with open('calibration.yaml','r') as stream:
        try:
            cali_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    camera_center = cali_config[args.camera_type]['center_point']
    chessboard_size = cali_config[args.camera_type]['chessboard']['spec'].split('x')
    PerspectiveHeight = cali_config[args.camera_type]['chessboard']['PerspectiveHeight']
    print 'use calibration board : ',cali_config[args.camera_type]['chessboard']['type']
    notice_1 = cali_config[args.camera_type]['notice_1']
    notice_2 = cali_config[args.camera_type]['notice_2']
    if notice_1 != 'none':
        print notice_1
    if notice_2 != 'none':
        print notice_2



    
    stopflag = False
    stage = 1

    board_w = int(chessboard_size[0])
    board_h = int(chessboard_size[1])
    square  = float(chessboard_size[2])
    # PerspectiveHeight = int(chessboard_size[2])
    # chessboard_position = args.chessboard_position.split(',')


    image = cv2.imread('right_back.png')
    camHeight,camWidth,PerspectiveMatrix = getHomoMatrix(board_w,board_h, square, image,PerspectiveHeight)
    while not stopflag:
        key = cv2.waitKey(1)
        if key == ord('1'):
            stage = 1
            print 'enter stage 1'
        elif key == ord('2'):
            stage = 2
            print 'enter stage 2'
        elif key == ord('s'):
            print 'save image'
            
        elif key == 27 :
            print 'Ese'
            stopflag = True
        else :
            pass
            
