# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:34:33 2018

@author: whlyc
"""
province = ['川','鄂','赣','甘','贵','桂','黑','沪','冀','津','京','吉','辽',
            '鲁','蒙','闽','宁','青','琼','陕','苏','晋','皖','湘','新','豫',
            '渝','粤','云','藏','浙']

import charRead
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
char = charRead.charRead('C:/Users/whlyc/Desktop/annCh')
x,y = char.getdata_bar()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train_d = np.zeros((x_train.shape[0],1,x_train.shape[1],x_train.shape[2]))
x_train_d[:,0,:,:] = x_train
x_test_d = np.zeros((x_test.shape[0],1,x_test.shape[1],x_test.shape[2]))
x_test_d[:,0,:,:] = x_test
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


from keras.utils import np_utils
from keras.models import load_model  
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
model = Sequential()

# Conv layer 1 output shape (32, 60, 40)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 60, 40),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 30, 20)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 30,20)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 15,10)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 15 * 10) = (9600), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(31))
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

model.save('string_cnn.h5')

string_cnn =  load_model('string_cnn.h5')

img = cv.imread('testSTR.jpg',0)
img= cv.resize(img,(40,60))
plt.imshow(img,'gray')
img_test = np.zeros((1,1,60,40))
img_test[0,0,:,:] = img
pred=string_cnn.predict(img_test)
for i in range(31):
    if pred[0,i]==1:
        break

province[i]










