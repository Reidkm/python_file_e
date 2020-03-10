# -*- coding: utf-8 -*-

# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

"""
Created on Sun Jun 17 19:59:33 2018
@author: mesut
"""
import yaml
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import glob
import time
import os

"""
If everything goes smoothly the script will print out something like this:
    
Found 36 images for calibration
DIM=(1600, 1200)
K=np.array([[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
D=np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])

"""
def calibrate_fisheye():
    CHECKERBOARD = (7,6)
    imagePath = 'fisheye/*.jpg'
    error_num = 0

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
#    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC 

#    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
#    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3), np.float32)
#    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp[:,0,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    print 'objp.shape = ',objp.shape

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(imagePath)
    print 'len(images) = ',len(images)

    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
            
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
#            cv2.drawChessboardCorners(img,CHECKERBOARD,corners,ret)
#            cv2.imshow('findCorners',img)
#            cv2.waitKey(100)

        else:
#            print 'error_num = ',error_num
            error_num+=1
            
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
#    print 'objp type = ',type(objpoints)
#    print 'objp type = ', (objpoints[0].shape)
#    
#    print 'corners type = ',type(imgpoints)
#    print 'corners type = ',(imgpoints[0].shape)
    
    
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs=None,
            tvecs=None,
            flags=calibration_flags,
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
            )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
     
    
#    xi=0.001   
#    retval, K, xi, D, rvecs, tvecs,idx = cv2.omnidir.calibrate(
#            objpoints,
#            imgpoints,
#            gray.shape[::-1],
#            K=K,
#            xi=xi,
#            D=D,
#            flags=0,
#            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-6)
#            )
#            
#    print 'retval', retval            
#    print 'K = ', K            
#    print 'xi', xi            
#    print 'D = ', D            
#    print 'idx', idx            
            

    
    # 反投影误差
#    total_error = 0
#    for i in range(len(objpoints)):
#        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
#
#        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#        total_error += error
#    print ("total error: ", total_error/len(objpoints))    
#    
def undistort_fisheye(img, balance, dim2=None, dim3=None):
#    DIM=_img_shape[::-1]
#    balance=1
#    dim2=None
#    dim3=None
    K=np.array([[187.14998779206752, 0.0, 504.8117676120353], 
                [0.0, 183.71778293950518, 373.77861313778044], 
                [0.0, 0.0, 1.0]])
    D=np.array([[0.10256359718633422], 
                [-0.04971826902086319], 
                [0.09067822908771175], 
                [-0.052017350675030216]])

    
#    K=np.array([[407.4366543152521, 0.0, 639.5], 
#                [0.0, 407.4366543152521, 359.5], 
#                [0.0, 0.0, 1.0]])
#    D=np.array([[0.0], [0.0], [0.0], [0.0]])    
    
#    img = cv2.imread(img_path)
    DIM = img.shape[:2][::-1]
#    print 'DIM = ',DIM

    
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
#    print 'dim1 = ',dim1
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2: #
        dim2 = dim1
        
    if not dim3:
        dim3 = dim1
        
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    data = {'dim1': dim1, 
            'dim2':dim2,
            'dim3': dim3,
            'K': np.asarray(K).tolist(), 
            'D':np.asarray(D).tolist(),
            'new_K':np.asarray(new_K).tolist(),
            'scaled_K':np.asarray(scaled_K).tolist(),
            'balance':balance}
    
    import json
    with open("fisheye_calibration_data.json", "w") as f:
        json.dump(data, f)
    
    
    cv2.imshow('img',img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#from Reid Wang    
def calibrate_pinhole():    
    w = 11
    h = 8
    side_length = 0.0825 #for intrincix only, ignore
    imagePath = 'image/*.png'

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1,2)
    
    objp=side_length * objp
#    print(objp)
    
    obj_points = []
    img_points = []
    
    images = glob.glob(imagePath)
    
    for fname in images:
        img = cv2.imread(fname)
    #    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray,(w,h),None)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            
#            cv2.drawChessboardCorners(img,(w,h),corners,ret)
#            cv2.imshow('findCorners',img)
#            cv2.waitKey(50)
            
    cv2.destroyAllWindows()
    
    RMS, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, 
                                                       img_points,
                                                       gray.shape[::-1],
                                                       None,
                                                       None
                                                       )
    
    print("RMS = ",RMS)
    print("K = ",K) #内参矩阵
    print("D = ",D)#畸变参数
    '''
('ret = ', 0.4014581915137537)
('mtx = ', array([[  1.00565303e+03,   0.00000000e+00,   6.10862313e+02],
       [  0.00000000e+00,   1.00650356e+03,   3.70139601e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]))
('dist = ', array([[-0.30784144,  0.11913999,  0.00116823,  0.00075992, -0.04898696]]))
('total error: ', 0.035674659722025916)

    '''
    #fw = open('/home/reid/Desktop/read_yaml_undistort/test.yaml','a',encoding='utf-8')
    #yaml.dump(mtx,fw)
    #yaml.dump(dist,fw)
    
    '''
    # 去畸变
    img2 = cv2.imread('/home/reid/Desktop/read_yaml_undistort/frame58.jpg')
    h,  w = img2.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h)) # 自由比例参数
    
    #mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)  # 获取映射方程
    
    #dst = cv2.remap(img2,mapx,mapy,cv2.INTER_CUBIC)        # 重映射后，图像变小了
    
    
    
    
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    # 根据前面ROI区域裁剪图片
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    
    
    cv2.imwrite('/home/reid/Desktop/tem_folder/webcamera_undistort/result11.jpg',dst)
    '''
    # 反投影误差
    total_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    print ("total error: ", total_error/len(obj_points))
    
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



def undistort_Mei(img, cam_config, verbose=True):             
    xi = np.array(cam_config['mirror_parameters']['xi'])
    k1 = float(cam_config['distortion_parameters']['k1'])
    k2 = float(cam_config['distortion_parameters']['k2'])
    p1 = float(cam_config['distortion_parameters']['p1'])
    p2 = float(cam_config['distortion_parameters']['p2'])
    gamma1 = float(cam_config['projection_parameters']['gamma1'])
    gamma2 = float(cam_config['projection_parameters']['gamma2'])
    u0 =     float(cam_config['projection_parameters']['u0'])
    v0 =     float(cam_config['projection_parameters']['v0'])

    
    K =  np.array([[  gamma1,   0,   u0],
                   [  0,   gamma2,   v0],
                   [  0,   0,   1]])
    D = np.array([[k1, k2, p1, p2]])

    h,  w = img.shape[:2]
    
    flag = cv2.omnidir.RECTIFY_PERSPECTIVE
#    flag = cv2.omnidir.RECTIFY_CYLINDRICAL
#    flag = cv2.omnidir.RECTIFY_LONGLATI
#    flag = cv2.omnidir.RECTIFY_STEREOGRAPHIC
#    Knew =  np.array([[  w/3.0,   0,   w/2.0 - 0],
#                      [  0,   h/3.0,   h/2.0 - 0],
#                      [  0,   0,   1]])
    Knew =  np.array([[  gamma1/4.0,   0,   u0], #change according to diff camera
                      [  0,   gamma2/4.0,   v0],
                      [  0,   0,   1]])
    cv2.imshow('img', img)
    # dst = cv2.omnidir.undistortImage(img, K, D, xi, flag, Knew=Knew, new_size=(w,h))
    dst = cv2.omnidir.undistortImage(img, K, D, xi, flag)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def VideoTrans(im,PerspectiveMatrix,width,height):
    im = cv2.warpPerspective(im, PerspectiveMatrix,(width,height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return im

def getHomoMatrix_usingVirtualPoint(board_w, board_h, square_size, image, scale, debug=False):
    img_disp = image.copy()
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
        cv2.drawChessboardCorners(img_disp, (board_w,board_h), corners2, ret)
        cv2.imshow('corner image',img_disp)

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
        
    
    
# =========================
# 批量undistort某个目录下的所有图片
# =========================
def imagesUndist():
    from top_eagle import TopEagle
    yamlFilePath = 'camera_camera_calib.yaml'
    imagesPath = '/home/matt/public/cam-calibrate/adlkiktech-sky_eye/src/jiaxing' + '/'

    camModel = 'pinhole' # Mei/pinhole/Scaramuzza
    # camModel = 'Scaramuzza' # Mei/pinhole/Scaramuzza
    eagle = TopEagle(yamlFilePath, camModel)

    f_list = os.listdir(imagesPath)
    for f in f_list:
        if f[-2: ] == 'py':
            continue
        print 'f = ', f

        img = cv2.imread(imagesPath + f)
        if img is None:
            print'[ERROR] no such file'
            continue
        dst = eagle.undistort(img)
        cv2.imwrite(imagesPath + f.split('.')[0] + '_undist_using_' + camModel + '.' + f.split('.')[1], dst)
        cv2.waitKey(100)
    cv2.destroyAllWindows

# =========================
# 实时显示homo之后的画面
# =========================
def main_video():    
    scale = 25.0
    board_w = 8
    board_h = 5
    square = 20.8 #cm

    cam = cv2.VideoCapture('rtsp://admin:kiktech2016@192.168.2.64:554/h264/ch1/main/av_stream')
    # cam = cv2.VideoCapture('rtsp://localhost/h264/ch1/main/av_stream')
    
    yamlFilePath = 'cfg/camera_camera_calib.yaml'
    with open(yamlFilePath,'r') as stream:
        try:

            cam_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc) 

    while True:
        t = time.time()
        ret, img = cam.read()
        if not (img is None):	
            cv2.imshow('origin', img)

            img2 = img.copy()    
            img2 = undistort_pinhole(img2, cam_config)
            # img_crop = cv2.resize(img2, (int(img2.shape[1]/2), int(img2.shape[0]/2)), interpolation=cv2.INTER_CUBIC) #INTER_NEAREST
            img_crop = img2
            # undistort_Mei(img, cam_config)
            cv2.imshow('undist', img_crop)
            camHeight,camWidth,PerspectiveMatrix, unit = getHomoMatrix_usingVirtualPoint(board_w, board_h, square, img_crop, scale, True)
            if PerspectiveMatrix is not None:
                img_homo = VideoTrans(img_crop, PerspectiveMatrix, camWidth, camHeight)
                cv2.imshow('homo', img_homo)


                # print('[INFO] elapsed time: {:.4f}'.format(time.time() - t))
                print('[INFO] approx. FPS: {:.4f}'.format(1/(time.time() - t)))
        
            key = cv2.waitKey(1) & 0xFF # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite('base_image.png',img)

        else:
            print '[ERROR] cam.read()'
    cam.release()
    cv2.destroyAllWindows
        


# =========================
# =========================
def main():
    from top_eagle import TopEagle
    yamlFilePath = 'camera_camera_calib.yaml'
    # homoImgPath = 'sky_eye_camera.png'
    homoImgPath = 'SFexpress_test/base.png'
    measureImgPath = 'SFexpress_test/9/150cm_1.png'
    # camModel = 'pinhole' # Mei/pinhole/Scaramuzza
    camModel = 'Scaramuzza' # Mei/pinhole/Scaramuzza
    # CHECKERBOARD = (8, 5)
    # SUQARE_SIZE = 20.8 # cm 
    CHECKERBOARD = (7, 4)
    SUQARE_SIZE = 14.0 # cm 
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    stopflag = False    
    eagle = TopEagle(yamlFilePath, camModel)
    eagle.cameraSwitch()

    im = cv2.imread(homoImgPath)
    camHeight, camWidth, unit = eagle.getHomoMatrix_usingVirtualPoint(im, debug=True)

    while not stopflag:
        t = time.time()
        # image = eagle.queue.get().copy()
        image = cv2.imread(measureImgPath)
        if image is None:
            print'[ERROR] queue is empty'
            continue
        im = image.copy()
        
        # 九宫格
        im[int(eagle.image_height/3):int(eagle.image_height/3)+3,:,:] = [0,0,255]
        im[int(2*eagle.image_height/3):int(2*eagle.image_height/3)+3,:,:] = [0,0,255]
        im[:,int(eagle.image_width/3):int(eagle.image_width/3)+3,:] = [0,0,255]
        im[:,int(2*eagle.image_width/3):int(2*eagle.image_width/3)+3,:] = [0,0,255]
        cv2.imshow('origin', im)
        homo_img = eagle.homoTrans(image)
        # undist_img = eagle.undistort(image)
        # img_crop = cv2.resize(undist_img, (int(undist_img.shape[1]/2), int(undist_img.shape[0]/2)), interpolation=cv2.INTER_CUBIC) #INTER_NEAREST
        img_crop = homo_img # homo_img

        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), subpix_criteria)
            for i in range(0, CHECKERBOARD[1]-1):
                for j in range(0+1, CHECKERBOARD[0]-1):
                    index = int(i * CHECKERBOARD[0] + j)
                    pixelDistance = np.linalg.norm(corners[index] - corners[index-1])#distance between 2 points
                    # print 'pixelDistance = ', pixelDistance
                    unit = SUQARE_SIZE * 1 / pixelDistance 
                    # print unit
            cv2.drawChessboardCorners(img_crop, CHECKERBOARD, corners, ret)
            cv2.imshow('realtime Corners',img_crop)
#            cv2.waitKey(100)

        else:
           print '[WARN] not enough corners '








        # camHeight, camWidth, unit = eagle.getHomoMatrix_usingVirtualPoint(image, debug=True)
        # if eagle.PerspectiveMatrix is not None:
        #     img_homo = eagle.VideoTrans(undist_img)
        #     cv2.imshow('homo', img_homo)
        


        # if eagle.PerspectiveMatrix is not None:
        #     homo_img = eagle.homoTrans()
        #     cv2.imshow('homo', homo_img)




        # print('[INFO] elapsed time: {:.4f}'.format(time.time() - t))
        # print('[INFO] approx. FPS: {:.4f}'.format(1/(time.time() - t)))


        key = cv2.waitKey(1)
        if key == ord("q"):
            stopflag = True
            eagle.videoSourceRelease()
    cv2.destroyAllWindows
    
    
if __name__ == "__main__":
    # imagesUndist()
    main_video()
    # main()    
    
    
