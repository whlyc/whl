# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:00:44 2018

@author: whlyc
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def findLicence(image):
    
    img = cv.GaussianBlur(image,(5,5),0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.Sobel(img,cv.CV_16S,1,0)
    img = cv.convertScaleAbs(img)
    ret,img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(17,3))
    img = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
    img_h,img_w= img.shape[0],img.shape[1]
    img_area = img_h*img_w
    plt.imshow(img,'gray')
    image1, contours, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    box = []
    possible_l = []

    for c in contours:
        rect = cv.minAreaRect(c)
        
        angle = rect[2]
        if rect[1][0]<rect[1][1]:
            if angle>-60:
                continue
            w = rect[1][1]
            h = rect[1][0]
        else:
            if angle<-30:
                continue
            w , h = rect[1][0],rect[1][1]
        box1 = cv.boxPoints(rect)
        box1 =np.int0(box1)
        if h==0:
            continue
        ratio =float(w) / float(h)
        if (ratio > 8 or ratio < 2):
            continue
        area = h*w
        if area<0.001*img_area or area>0.06*img_area:
            continue
    
    
        for i in range(4):
            if box1[i][0]<0:
                box1[i][0] = 0
            if box1[i][0]>image.shape[1]-1:
                box1[i][0] = image.shape[1]-1
            if box1[i][1]<0:
                box1[i][1] = 0
            if box1[i][1]>image.shape[0]-1:
                box1[i][1] = image.shape[0]-1
        xs = [i[0] for i  in box1]
        ys = [i[1] for i  in box1]
        x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
    
        height,width = y2-y1,x2-x1
        #temp[:,:5],temp[:,10:] = 0,0
        #temp[:5,5:10],temp[10:,5:10]=0,0
        temp = image[y1:y1+height,x1:x1+width].copy()
        #temp[:,:x1],temp[:,x1+width:] = 0,0
        #temp[:y1,x1:x1+width],temp[y1+height:,x1:x1+width]=0,0
        if angle>=-45:
            angle = angle
        else:
            angle = (90+angle)
        (t_h,t_w) = temp.shape[0:2]
        center = (t_w//2,t_h//2)
        M = cv.getRotationMatrix2D(center,angle,1.0)
        Rotated = cv.warpAffine(temp,M,(t_w,t_h))
        #,flags = cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE
        #k = cv.getRectSubPix(Rotated,(t_w,t_h),(t_w/2,t_h/2))
        #img[y1:y1+height,x1:x1+width]
        possible_l.append(Rotated)
        box.append(box1)

    return possible_l







        