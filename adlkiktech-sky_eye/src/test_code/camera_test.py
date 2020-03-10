#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import sys
import datetime

IS_RECORD = False
def openCam(id, height, width):
    cam = cv2.VideoCapture('rtsp://admin:kiktech2016@192.168.1.64:554/out.h264')
    cam.set(3, width)
    cam.set(4, height)
    return cam

if __name__ == '__main__':
    cams_string = sys.argv[1]
    test_cam = []
    for i in cams_string.split(',') :
        test_cam.append(int(i))

    cams = {}
    for i in test_cam:
        cams[i] = openCam(i, 480, 640)

    if IS_RECORD:
        fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
        out = cv2.VideoWriter('output.avi', fourcc, 10.0, (1280, 720))
    while(True) :
        for i in test_cam:
            cam = cams[i]
            ret_val, img = cam.read()
            if img is None:
                pass
            cv2.imshow("video{0}".format(i), img)
            print(datetime.datetime.now())
            if IS_RECORD:
                out.write(img)
        key = cv2.waitKey(2)
        if key == 27:
            if IS_RECORD:
                print('record release')
                out.release()
            for i in test_cam:
                cam = cams[i]
                cam.release()
                print('cam release')
            exit(0)
