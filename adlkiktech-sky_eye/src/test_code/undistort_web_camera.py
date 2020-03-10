#!/usr/bin/env python
import numpy as np
import cv2
import glob
import yaml

f = open('web_camera_para.yaml','r',encoding='utf-8')
cont = f.read()
dict_1 = yaml.load(cont)
camera_matrix = np.array(dict_1['mtx'])
distortion = np.array(dict_1['dist'])
print(camera_matrix)
print(distortion)

#rtsp://admin:kiktech2016@192.168.1.64:554/out.h264
cam = cv2.VideoCapture('rtsp://admin:kiktech2016@192.168.1.64:554/out.h264')

while True:
	ret, image = cam.read()
	if not (image is None):
		dst = cv2.undistort(image,camera_matrix, distortion, None, camera_matrix)
		cv2.imshow('Demo', dst)
		cv2.waitKey(1)
