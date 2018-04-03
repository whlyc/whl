# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:49:17 2018

@author: whlyc
"""
import os
import numpy as np
import cv2 as cv
import feature
class Dataset:
    def __init__(self,lir):
        self.dir = lir
        self.List = []
        self.x_train = []
        self.y_train = []
    def readList(self):
        docList = os.listdir(self.dir)
        for i in docList:
            self.List.append(self.dir+'/'+i)
    def readImg_bar(self):
        ite = 0
        for i in self.List:
            imgL = os.listdir(i)
            for filename in imgL:
                img = cv.imread(i+'/'+filename,0)
                self.x_train.append(feature.getBar(img))
                self.y_train.append(ite)
            ite =ite+1
    def readImg_hog(self):
        ite = 0
        for i in self.List:
            imgL = os.listdir(i)
            for filename in imgL:
                img = cv.imread(i+'/'+filename,0)
                self.x_train.append(feature.getHog(img))
                self.y_train.append(ite)
            ite =ite+1
    def getdata_bar(self):
        self.readList()
        self.readImg_bar()
        return self.x_train,self.y_train
    def getdata_hog(self):
        self.readList()
        self.readImg_hog()
        return self.x_train,self.y_train





