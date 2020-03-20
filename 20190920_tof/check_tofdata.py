#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2




if __name__ == '__main__':
    line_point = np.array([[5,378],
                           [252,259],
                           [371,265],
                           [637,381],
                           [645,378],
                           [892,259],
                           [1011,265],
                           [1277,381]]) #创建二维数组line_point
    for tof_data_path in glob.glob(r'C:\Users\x\Desktop\kk\*.jpg'):
            src_image = cv2.imread(tof_data_path, cv2.IMREAD_ANYCOLOR)
            cv2.line(src_image,line_point[0],line_point[1],(0,0,255),2)
            cv2.line(src_image,line_point[2],line_point[3],(0,0,255),2)
            cv2.line(src_image,line_point[4],line_point[5],(0,0,255),2)
            cv2.line(src_image,line_point[6],line_point[7],(0,0,255),2)
            cv2.imshow('src_image', src_image)
            cv2.waitKey()
    
        

