#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import imageio
import argparse
import os
import linecache
import time
import yaml

#def detectBoardCorner(inputImage,board_w,board_h):
    
#        return corners2

def undistort_pinhole(img, cam_config, alpha=1.0, verbose=True): 
    k1 = float(cam_config['distortion_parameters']['k1'])
    k2 = float(cam_config['distortion_parameters']['k2'])
    p1 = float(cam_config['distortion_parameters']['p1'])
    p2 = float(cam_config['distortion_parameters']['p2'])
    k3 = 0.0
    fx = float(cam_config['projection_parameters']['fx'])
    fy = float(cam_config['projection_parameters']['fy'])
    cx = float(cam_config['projection_parameters']['cx'])
    cy = float(cam_config['projection_parameters']['cy'])       
    K =  np.array([[  fx,   0.0,   cx],
           [  0.0,   fy,   cy],
           [  0.0,   0.0,   1.0]])
    D = np.array([[k1,  k2,  p1,  p2, k3]])

    h,  w = img.shape[:2]
    
#method 1:    
    '''  
    dst = np.zeros((h,w,3))
    cv2.imshow('img', img)
    dst = cv2.undistort(img, K, D)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
#method 2:    
    # alpha = 0.5
    inputImageSize = (w,h)
    outputImageSize = (w,h)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K, D, inputImageSize, alpha, outputImageSize)
    
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w,h), 5)
    
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
    # if verbose is True:
        # print 'newcameramtx = ',newcameramtx
        # img_2 = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('img', img_2)
        # dst_2 = cv2.resize(dst, (int(w/2), int(h/2)), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('dst', dst_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


def getVirtualPoint(inputImage,board_w,board_h) :
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgPts = np.float32([corners2[0],corners2[board_w-1],corners2[(board_h-1)*board_w],corners2[(board_h-1)*board_w + board_w-1]])
    return imgPts


def VideoTrans(im,PerspectiveMatrix,width,height, flags=cv2.WARP_INVERSE_MAP ): #flags=cv2.INTER_LINEAR
    im = cv2.warpPerspective(im, PerspectiveMatrix, (width,height), flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return im



def getHomoMatrix(homo_image,test_homo_image,stand_image,board_w,board_h):
    homo_image_imgPts = getVirtualPoint(homo_image,board_w,board_h)
    test_homo_imgPts = getVirtualPoint(test_homo_image,board_w,board_h)
    stand_image_imgPts = getVirtualPoint(stand_image,board_w,board_h)

    PerspectiveMatrix = cv2.getPerspectiveTransform(homo_image_imgPts, stand_image_imgPts)
    PerspectiveMatrix_new = cv2.getPerspectiveTransform(test_homo_imgPts, stand_image_imgPts)
    return PerspectiveMatrix ,PerspectiveMatrix_new

def resizedisplayWindow(window_size_index):
    if window_size_index == 0 :
        win_width = 640 
        win_height = 480 
    elif window_size_index == 1 :
        win_width = 1280 
        win_height = 720 
    elif window_size_index == 2 :
        win_width = 1920 
        win_height = 1080
    return  win_width,win_height
    


def main():

    board_w = 7
    board_h = 4

    #get camera index
    parser = argparse.ArgumentParser(description='skyeye calibration')
    #parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')

    
    parser.add_argument("index", type=int, default=0,help="camera index")

    args = parser.parse_args()
    #height = args.demo_size
    #print(args.index)
    camera_index = args.index

    print('camera number is {0} '.format(camera_index))
    
    camera_address = linecache.getline('./config/skyeye_stream_add.txt',camera_index)

    print('camera stream addtress is {0} '.format(camera_address))

    stand_image_path = './config/stand_image/stand_image.png'

    #path = '/home/reid/skyeye_cali/config/skyeye_stream_add.txt'

    #(filepath,tempfilename) = os.path.split(path)

    #homo_image_path= filepath+'/homo_image/'+str(camera_index)+'_homo.png'

    #homo_image = os.path.join(filepath,homo_image)

    #test_homo_image_path = filepath+'/test_homo_image/'+str(camera_index)+'_homo.png'

    test_homo_image_path ='./config/test_homo_image/homo_new_{0}.png'.format(camera_index)
    homo_image_path ='./config/homo_image/homo_{0}.png'.format(camera_index)


    #test_homo_image = os.path.join(filepath,test_homo_image)

    #/home/reid/skyeye_cali/config/homo_image

    # (filename,extension) = os.path.splitext(tempfilename)

    #/home/reid/skyeye_cali/config/test_homo_image
    

    # homo_image = '/home/reid/skyeye_cali/config/homo_image/left_back_camera.png'

    #test_homo_image = '/home/reid/skyeye_cali/config/test_homo_image/right_back_camera.png'

   #print (homo_image)

    if os.path.exists(test_homo_image_path) is False:
        test_homo_image_path = homo_image_path
        print('no new homo image')
    else :
        print('test_homo_image_path is {0}'.format(test_homo_image_path))
    print('stand_image_path is {0}'.format(stand_image_path))
    print('homo_image_path is {0}'.format(homo_image_path))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    homo_image=cv2.imread(homo_image_path)  
    test_homo_image=cv2.imread(test_homo_image_path)
    stand_image=cv2.imread(stand_image_path) 

    (PerspectiveMatrix ,PerspectiveMatrix_new) = getHomoMatrix(homo_image,test_homo_image,stand_image,board_w,board_h)

    
    #print("homo_image=%s" %homo_image)

    #print("test_homo_image=%s" %test_homo_image)
    


    cam = cv2.VideoCapture(int(camera_address))
    #cam = cv2.VideoCapture(1)
    window_index = 1
    window_size_index = 0

    #imgpoints = [] # 2d points in image plane.
    
    
    #cv2.namedWindow("window",0);
    #cv2.resizeWindow("window", 640, 480);

    yamlFilePath = 'config/camera_camera_calib.yaml'
    with open(yamlFilePath,'r') as stream:
        try:

            cam_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    
    cv2.namedWindow("skyeye",0)

    while True:
        ret, src_image = cam.read()
        if src_image is None:
            #print('no image')
            continue

        image = undistort_pinhole(src_image, cam_config)
        key  = cv2.waitKey(1)
        
        if window_index == 1 :
            #cv2.namedWindow("skyeye",0)
            
            #win_width,win_height = resizedisplayWindow(window_size_index)
            
            #cv2.resizeWindow("skyeye", win_width, win_height)


            gray = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Draw and display the corners
                cv2.drawChessboardCorners(src_image, (board_w,board_h), corners2,ret)

            #corners = detectBoardCorner(image,board_w,board_h)
            #imgpoints.append(corners)
            #cv2.drawChessboardCorners(image,(board_w,board_h),corners,True)
            
            cv2.imshow("skyeye",src_image)
            if key == ord('s') :
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imageio.imwrite('{0}_{1}.png'.format(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()),camera_index), src_image) #save image
        #time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime())
        elif window_index == 2 :
            #cv2.namedWindow("skyeye_homo",0)
            #win_width,win_height = resizedisplayWindow(window_size_index)
            #cv2.resizeWindow("skyeye", win_width, win_height)
            img_homo = VideoTrans(image,PerspectiveMatrix,image.shape[1],image.shape[0], flags=cv2.WARP_INVERSE_MAP )
            cv2.imshow("skyeye",img_homo)
            if ord('s') == key :
                img_homo = cv2.cvtColor(img_homo, cv2.COLOR_BGR2RGB)
                imageio.imwrite('{0}_homo_{1}.png'.format(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()),camera_index), img_homo) #save image
        elif window_index == 3 :
            #cv2.namedWindow("skyeye_test",0)
            #win_width,win_height = resizedisplayWindow(window_size_index)
            #cv2.resizeWindow("skyeye", win_width, win_height)
            img_homo_new = VideoTrans(image,PerspectiveMatrix_new,image.shape[1],image.shape[0], flags=cv2.WARP_INVERSE_MAP )
            cv2.imshow("skyeye",img_homo_new)
            if ord('s') == key :
                img_homo_new = cv2.cvtColor(img_homo_new, cv2.COLOR_BGR2RGB)
                imageio.imwrite('{0}_homo_new_{1}.png'.format(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()),camera_index), img_homo_new) #save image 
        if key == ord('1')   :
            window_index = 1
            #cv2.destroyWindow(skyeye_homo)
            #cv2.destroyWindow(skyeye_test)
        elif key == ord('2')  :
            window_index = 2
            #cv2.destroyWindow(skyeye)
            #cv2.destroyWindow(skyeye_test)
        elif key == ord('3')  :
            window_index = 3
            #cv2.destroyWindow(skyeye)
            #cv2.destroyWindow(skyeye_homo)
        elif key == ord('k') :
            
            #window_size_index = +1
            window_size_index  = window_size_index +1 

            

            if window_size_index > 2 :
                window_size_index = 0
            win_width,win_height = resizedisplayWindow(window_size_index)
            
            cv2.resizeWindow("skyeye", win_width, win_height)
            #print(window_size_index)

        elif  key == 27 :
            cv2.destroyAllWindows()
            cam.release()
            break
        else :
            pass
        
        # if window_index == 0:
        #     if ord('k') == cv2.waitKey(1) && window_size_index == 0
        #         cv2.resizeWindow("window", 1280, 720);
        #     cv2.imshow("resized",iamge)
        #     if  ord('1') == cv2.waitKey(1)
        #Python: cv2.resizeWindow(winname, width, height) 
         
        

    
    
    #height = args.demo_size
    #width = 640
    #video_source = args.video_source




if __name__=='__main__':
    main()

    

    
    

    



	
    
