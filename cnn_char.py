# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 10:18:36 2018

@author: whlyc
"""
setdic=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E',
        'F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',
        'W','X','Y','Z']

import charRead
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

char = charRead.charRead('C:/Users/whlyc/Desktop/annGray')
x,y = char.getdata_bar()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train_d = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_train_d[:,:,:,0] = x_train
x_test_d = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
x_test_d[:,:,:,0] = x_test
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)




from keras.utils import np_utils
from keras.models import load_model  
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 60, 30,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(34))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(x_train_d, y_train, epochs=3, batch_size=20,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test_d, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


model.save('model/char_cnn.h5')
model.save('char_cnn.h5')

char_cnn=load_model('char_cnn.h5')

x = cv.imread('testCH.jpg',0)
img = cv.resize(x,(30,60))

plt.imshow(img,'gray')

img_test = np.zeros((1,60,30,1))

img_test[0,:,:,0] = img

pred = char_cnn.predict(img_test)

for i in range(34):
    if pred[0,i]==1:
        break
setdic[i]



