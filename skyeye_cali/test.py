
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import os 

#print (time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()))

#print("{0}_{1}".format("hello",time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime())) )

# camera_index = 1
# path = '/home/reid/skyeye_cali/config/skyeye_stream_add.txt'

# (filepath,tempfilename) = os.path.split(path)

# test_homo_image_path = '{0}/test_homo_image/homo_{1}.png'.format(filepath,camera_index)

# print(test_homo_image_path)


import numpy as np
import cv2

cv2.namedWindow("img",0)
cap=cv2.VideoCapture(0)
while True:

    sucess,img=cap.read()

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    cv2.imshow("img",gray)

    

    k=cv2.waitKey(1)

    if k == 27:

        cv2.destroyAllWindows()
        break
    elif k==ord("s"):

        cv2.imwrite("image2.jpg",img)
        cv2.destroyAllWindows()
        break
    elif k==ord("k"):
        cv2.resizeWindow("img", 1280, 720)
        

cap.release()
