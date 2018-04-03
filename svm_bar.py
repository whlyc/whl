# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:57:27 2018

@author: whlyc
"""
import dataset
import feature
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
d_train = dataset.Dataset('C:/Users/whlyc/Desktop/svm/train')
x_train,y_train = d_train.getdata_bar()

d_test = dataset.Dataset('C:/Users/whlyc/Desktop/svm/test')
x_test,y_test = d_test.getdata_bar()

stdsc = StandardScaler()
#x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)

svm = SVC(kernel = 'rbf',random_state = 0,C = 20,gamma = 0.0058)
svm.fit(x_train,y_train)
y=svm.predict(x_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
confmat1 = confusion_matrix(y_test,y)

(accuracy_score(y_test,y),precision_score(y_test,y),
 recall_score(y_test,y),f1_score(y_test,y))

from sklearn.externals import joblib
joblib.dump(svm, 'model/svm_bar.pkl') 

clf = joblib.load('svm_bar.pkl') 
y1=clf.predict(x_test_std)
confmat2 = confusion_matrix(y_test,y1)




