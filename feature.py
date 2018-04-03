# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:15:21 2018

@author: whlyc
"""
from skimage.feature import hog  
import cv2 as cv 
import numpy as np
def getBar(img):
    ret,img = cv.threshold(img,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ft_bar = []
    for i in range(img.shape[0]):
        temp = np.array(img[i,:])
        ft_bar.append(len(temp[temp==1])/img.shape[1])
    for i in range(img.shape[1]):
        temp = np.array(img[:,i])
        ft_bar.append(len(temp[temp==1])/img.shape[0])
    return ft_bar

def getHog(img):
    normalize = True  
    visualize = False  
    block_norm = 'L2-Hys'  
    cells_per_block = [2,2]  
    pixels_per_cell = [10,10]  
    orientations = 3  
    ft_hog = hog(img, orientations, pixels_per_cell, cells_per_block, 
         block_norm, visualize, normalize)
    return ft_hog

