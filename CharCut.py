# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 10:13:21 2018

@author: whlyc
"""
province = ['川','鄂','赣','甘','贵','桂','黑','沪','冀','津','京','吉','辽',
            '鲁','蒙','闽','宁','青','琼','陕','苏','晋','皖','湘','新','豫',
            '渝','粤','云','藏','浙']
setdic=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E',
        'F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',
        'W','X','Y','Z']
import cut
import feature
from sklearn.externals import joblib
from keras.models import load_model  
from sklearn.svm import SVC
import numpy as np
import cv2 as cv
import findLP
import matplotlib.pyplot as plt
svm_l = joblib.load('model/svm_bar.pkl')
cnn_str = load_model('model/string_cnn.h5')
cnn_char = load_model('model/char_cnn.h5')
image = cv.imread('car/4.jpg')
possible = findLP.findLicence(image)
plt.imshow(possible[0])
judge_p = []
for i in range(len(possible)):
    possible[i]=cv.cvtColor(possible[i], cv.COLOR_BGR2GRAY)
    p = cv.resize(possible[i],(136,36))
    judge_p.append(feature.getBar(p))
    

isL = svm_l.predict(judge_p)
total = []

for car in range(len(possible)):
    if isL[car]==1:
        continue
    else:
      
        LPR = ''
        ret,img = cv.threshold(possible[car],0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
        plt.imshow(img,'gray')
        img1 = cut.scan(img)
        char = cut.cut(img1)
        for i in range(7):
            if i==0:
                char_this= cv.resize(char[i],(40,60))
                this_char = np.zeros((1,1,60,40))
                this_char[0,0,:,:] = char_this
                pred=cnn_str.predict(this_char)
                pred[pred>0.6] = 1
                for j in range(31):
                    if pred[0,j]==1:
                        break
                LPR+=province[j]
            else:
                char_this= cv.resize(char[i],(30,60))
                this_char= np.zeros((1,1,60,30))
                this_char[0,0,:,:] = char_this
                pred=cnn_char.predict(this_char)
                pred[pred>0.6] = 1
                for k in range(34):
                    if pred[0,k]==1:
                        break
                LPR+=setdic[k]

        total.append(LPR) 
total    
    plt.imshow(char_this)
    plt.imshow(char[5])
    
    
    
    
import tensorflow
    
import numpy as np
import theano.tensor as T
import theano
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = theano.function([x,y],z)
print(f(2,3))
    
x = T.dmatrices('x')    #64b
y = T.dmatrices('y')   
z = T.dot(x,y)
f = theano.function([x,y],z)
print(f(np.arange(12).reshape((3,4)),
        10*np.ones((4,3))))

    
import tensorflow as tf
x =tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(x,a)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))













