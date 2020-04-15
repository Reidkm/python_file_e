#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

# X = np.array([[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]])

# print(X)
# # X 是一个二维数组，维度为 0 ，1
# # 第 0 层 [] 表示第 0 维；第 1 层 [] 表示第 1 维；

# # X[n0,n1] 表示第 0 维 取第n0 个元素 ，第 1 维取第 n1 个元素
# print(X[:,:]<=10)
# # X[1:3,1:3] 表示第 0 维 取 (1:3)元素 ，第 1 维取第(1:3) 个元素
# print(X[1:3,1:3])

# # X[:n0,:n1] 表示第 0 维 取 第0 到 第n0 个元素 ，第 1 维取 第0 到 第n1 个元素
# print(X[:2,:2])
# # X[:,:n1] 表示第 0 维 取 全部元素 ，第 1 维取 第0 到第n1 个元素
# print(X[:,:2])

# # X[:,0]) 表示第 0 维 取全部 元素 ，第 1 维取第 0 个元素
# print(X[:,0])




#print(len(X))
#c.ravel()[np.flatnonzero(c)]
#print(X_3.ravel()[np.flatnonzero(X_3)])
#print(np.min(X_3[X_3!=0]))


#  First determine the fully available space before the minimum distance
def calculate_space(input_array):
    no_space = 1
    if np.min(input_array[input_array!=0]) > 11 :  # 11 is the pallet length 
        print('depth from 0 to {} and width from 0 to 20 is fully available space'.format(np.min(input_array[input_array!=0])))
        if np.nonzero(input_array)[0][0] >= 12 :
            print('depth from {} to 30 and width from 0 to {} is available space'.format(np.min(input_array[input_array!=0]),np.nonzero(input_array)[0][0]))
        elif np.nonzero(input_array)[0][-1] <= 8 :
            print('depth from {} to 30 and width from {} to 20 is available space'.format(np.min(input_array[input_array!=0]),np.nonzero(input_array)[0][-1]+2))
        else :
            for array_index in range(len(np.nonzero(input_array)[0])-1) :
                if np.nonzero(input_array)[0][array_index+1] -np.nonzero(input_array)[0][array_index] >=11 :
                    print('depth from {} to 30 and width from {} to {} is available space'.format(np.min(input_array[input_array!=0]),np.nonzero(input_array)[0][array_index]+2,np.nonzero(input_array)[0][array_index+1]))
                    no_space = 0
                    break
                array_index += 1
            if no_space == 1 :
                print('depth from {} to 30 no available space'.format(np.min(input_array[input_array!=0])))
    else:
        print('')

if __name__=='__main__':
    X_1 = np.array([0,0,18,0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    X_2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,13,0,0,0])
    X_3 = np.array([0,0,0,17,0,0,0,0,0,0,19,0,0,0,13,0,0,0,0,0])
    X_4= np.array([0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,23,0])
    calculate_space(X_4)
