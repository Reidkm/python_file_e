#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datetime
import Queue
import traceback
import binascii
import math
import numpy as np
import cv2


"""
input  :
    color image
    color threshold
    True/False(print the information or not)
output :
    color image
    center point
"""
def ColorFilter(im_i,ColorLower, ColorUpper,debug):
    hsv = cv2.cvtColor(im_i, cv2.COLOR_BGR2HSV)

	# construct a mask for the spepcific color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv2.inRange(hsv, ColorLower, ColorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    cnts = cnts[1]
    center = None

	# only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
#        if debug is True:
#            print('c is :',c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # if debug is True:
            # print('========center=',center)

		# only proceed if the radius meets a minimum size
        if center is not None:
            cv2.circle(im_i, center, 5, (255, 255, 255), -1)

    return im_i,center,mask

"""
input  :
    color image
    color threshold
    True/False(print the information or not)
output :
    color image
    center point
"""
def RedColorFilter(im_i,debug):
    PARAM_areaThreshold = 1e3 #change the area threshold as you want

    hsv = cv2.cvtColor(im_i, cv2.COLOR_BGR2HSV)

	# construct a mask for the spepcific color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask_1 = cv2.inRange(hsv, (0,43,46), (10,255,255))
    mask_2 = cv2.inRange(hsv, (156,43,46), (180,255,255))
    mask = cv2.bitwise_or(mask_1,mask_2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # if debug is True:
        # cv2.imshow('mask image',mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[1]
    center = None

	# only proceed if at least one contour was found
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        # if debug is True:
        #    print('area of c is :',cv2.contourArea(c))
        if cv2.contourArea(c) > PARAM_areaThreshold:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if debug is True:
                print('========center=',center)

            # only proceed if the radius meets a minimum size
            if center is not None:
                cv2.circle(im_i, center, 5, (0, 0, 255), -1)


    # if debug is True:
    #     cv2.imshow("Frame", im_i)

    return im_i,center, mask

def printHex(content):
    return binascii.hexlify(bytearray(content))

    # return ':'.join("{:02x}".format(ord(str(c))) for c in content)

def printTick(tick, content):
    tick2 = datetime.datetime.now()
    print ' ', content, ' ', tick2 - tick
    tick = tick2

# from int to 2 bytes
# Matt
def intToBytearray(value):
    # High 3 bit in front, low 3 bit in back
    return bytearray([value/256, value%256])




def GetCrossAngle(l1, l2):
    arr_0 = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])
    arr_1 = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1)))) # 注意转成浮点数运算
    return np.arccos(cos_value) * (180/np.pi)

# >0 point is on anti-clockwise side
# =0: poit in on line or its extend part
# <0: point in on clockwise side
def PointOffLine(pt, line):
    return ((pt.x - line.p2.x) * (line.p1.y - line.p2.y) - (line.p1.x - line.p2.x) * (pt.y - line.p2.y))*1.0

def distanceBetweenPointAndLine(point, line):
    p1 = np.array([line.p1.x, line.p1.y])
    p2 = np.array([line.p2.x, line.p2.y])
    p3 = np.array([point.x, point.y])
    return np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)

def InnerLine(plus_lines, minus_lines):
    max_out_minus = 0
    max_out_minus_line = None
    for plus_line in plus_lines:
        out_minus = 0
        for minus_line in minus_lines:
            if PointOffLine(minus_line.p1, plus_line) <= 0 and PointOffLine(minus_line.p2, plus_line) <= 0:
                out_minus += 1
        if out_minus > max_out_minus:
            max_out_minus_line = plus_line

    max_out_plus = 0
    max_out_plus_line = None
    for minus_line in minus_lines:
        out_plus = 0
        for plus_line in plus_lines:
            if PointOffLine(plus_line.p1, minus_line) <= 0 and PointOffLine(plus_line.p2, minus_line) <= 0:
                out_plus += 1
        if out_plus > max_out_plus:
            max_out_plus_line = minus_line

    return (max_out_plus_line, max_out_minus_line)

def lineintersection(l1, l2):
    L1 = lineintersectionWithLine(l1)
    L2 = lineintersectionWithLine(l2)
    return intersection(L1, L2)

def lineintersectionWithLine(l):
    return lineintersectionWithPoint(l.p1, l.p2)

def lineintersectionWithPoint(p1, p2) :
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x * p2.y - p2.x * p1.y)
    return A, B, -C

def intersection(L1, L2) :
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else :
        return None

def drawLine(img, line, color=(255,255,255), thickness=1):
    cv2.line(img,(int(line.p1.x),int(line.p1.y)),(int(line.p2.x),int(line.p2.y)),color,thickness)

# def drawLine(img, p1, p2, color=(255,255,255), thickness=1):
#     cv2.line(img,(int(p1.x),int(p1.y)),(int(p2.x),int(p2.y)),color,thickness)


"""
input : BGR image
        debug( or not )
output: gray image
"""
def smoothImg(img,debug):
    smooth_img = np.copy(img)
    smoothed = np.zeros(img.shape,dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # smoothed[:,:,:] = 0
    ret,img = cv2.threshold(img,127,255,0)
    if debug is True:
        cv2.imshow('threshold',img)

#    contour smoothing parameters for gaussian filter
    filterRadius = 7
    filterSize = 2*filterRadius + 1
    sigma  = 20.0
#    find contours and store all contour points
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE,offset=(0,0))#CHAIN_APPROX_SIMPLE
    for j in range(len(contours)):
        contours[j] = np.squeeze(contours[j])
        if contours[j].ndim == 1:
            continue
#        extract x and y coordinates of points. we'll consider these as 1-D signals
#        add circular padding to 1-D signals
        length = len(contours[j]) + 2 * filterRadius
        idx    = (len(contours[j]) - filterRadius)
#        filter 1-D signals
#       x = []
#       y = []
        x = np.zeros(shape=(1,length),dtype=float)
        y = np.zeros(shape=(1,length),dtype=float)
        x = np.squeeze(x)
        y = np.squeeze(y)
        # print '+++++++++++++++++++++++++++++++'
        # print contours
        # print contours[j].shape
        # print '+++++++++++++++++++++++++'

        for i in range(0,length):
            x[i] = contours[j][(idx + i) % len(contours[j]),0]
            y[i] = contours[j][(idx + i) % len(contours[j]),1]

        #filter 1-D signals
        xFilt = cv2.GaussianBlur(x,(filterSize, filterSize),sigma,sigma)
        yFilt = cv2.GaussianBlur(y,(filterSize, filterSize),sigma,sigma)
        xFilt = np.squeeze(xFilt)
        yFilt = np.squeeze(yFilt)

#        build smoothed contour
        smooth = []
#        smoothContours = []
        for i in range(filterRadius,(len(contours[j]) + filterRadius),1):
            smooth.append( [xFilt[i], yFilt[i]] )


        smooth_list2np = np.array(smooth, dtype = int)
        smooth_list2np = np.squeeze(smooth_list2np)
#        smoothContours.append(smooth_list2np)#list
#        smoothContours = np.append(smoothContours,smooth)#array


        # hierarchy = np.squeeze(hierarchy)
        #
        # print hierarchy
        # if hierarchy.ndim == 1:
        #     continue
        if hierarchy[0][j][3] < 0:
#            color = (0,0)
            color = (255,255,255)
        else:
#            color = (255,255)
            color = (0,0,0)

#        smooth_img = cv2.drawContours(smoothed, [smoothContours[0]], 0, color, -1)
        smooth_img = cv2.drawContours(smoothed, [smooth_list2np], 0, color, -1)

#    smooth_img = cv2.drawContours(smoothed, contours, 3, color, -1)


    if debug is True:
        cv2.imshow('smooth',smooth_img)
    return smooth_img


"""
use list2array
input : BGR image
        debug( or not )
output: gray image
"""
# def smoothImage(img,debug):
#     smoothed = np.zeros(img.shape,dtype=np.uint8)
#     smoothed[:,:,:] = 0
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     ret,img = cv2.threshold(img,127,255,0)
#     if debug is True:
#         cv2.imshow('threshold',img)
#
# #    contour smoothing parameters for gaussian filter
#     filterRadius = 5
#     filterSize = 2*filterRadius + 1
#     sigma  = 10.0
# #    find contours and store all contour points
#     image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE,offset=(0,0))#CHAIN_APPROX_SIMPLE
#     for j in range(len(contours)):
#         contours[j] = np.squeeze(contours[j])
# #        extract x and y coordinates of points. we'll consider these as 1-D signals
# #        add circular padding to 1-D signals
#         length = len(contours[j]) + 2 * filterRadius
#         idx    = (len(contours[j]) - filterRadius)
# #        filter 1-D signals
#         x = []
#         y = []
#         for i in range(0,length):
#             x.append(contours[j][(idx + i) % len(contours[j]),0])
#             y.append(contours[j][(idx + i) % len(contours[j]),1])
#
#         x = np.array(x, dtype = float)
#         y = np.array(y, dtype = float)
#         #filter 1-D signals
#         xFilt = cv2.GaussianBlur(x,(filterSize, filterSize),sigma,sigma)
#         yFilt = cv2.GaussianBlur(y,(filterSize, filterSize),sigma,sigma)
#         xFilt = np.squeeze(xFilt)
#         yFilt = np.squeeze(yFilt)
#
# #        build smoothed contour
#         smooth = []
# #        smoothContours = []
#         for i in range(filterRadius,(len(contours[j]) + filterRadius),1):
#             smooth.append( [xFilt[i], yFilt[i]] )
#
#
#         smooth_list2np = np.array(smooth, dtype = int)
#         smooth_list2np = np.squeeze(smooth_list2np)
# #        smoothContours.append(smooth_list2np)#list
# #        smoothContours = np.append(smoothContours,smooth)#array
#
#
#         hierarchy = np.squeeze(hierarchy)
#
#
#         if hierarchy[j][3] < 0:
# #            color = (0,0)
#             color = (255,255,255)
#         else:
# #            color = (255,255)
#             color = (0,0,0)
#
# #        smooth_img = cv2.drawContours(smoothed, [smoothContours[0]], 0, color, -1)
#         smooth_img = cv2.drawContours(smoothed, [smooth_list2np], 0, color, -1)
#
# #    smooth_img = cv2.drawContours(smoothed, contours, 3, color, -1)
#
#
#     if debug is True:
#         cv2.imshow('smooth',smooth_img)
#     return smooth_img