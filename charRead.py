# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 10:53:45 2018

@author: whlyc
"""
import os
import numpy as np
import cv2 as cv
import feature
class charRead:
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
                img1 = cv.resize(img,(30,60))
                self.x_train.append(img1)
                self.y_train.append(ite)
            ite =ite+1
    def getdata_bar(self):
        self.readList()
        self.readImg_bar()
        return self.x_train,self.y_train
