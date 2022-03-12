# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:04:55 2020

@author: Haoti
"""

#%% Importing Modules

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd

#%% SECTION TO RUN WITH GPU

# Choose GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

#Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
# Config.gpu_options.allow_growth=True

#%% Load data
size_x = 6
size_y = 6

data_set  = 5000
data_new  = np.zeros([data_set, 6, 6])
stiff_new = np.zeros([data_set, 1])

mat_mtx = np.load('weave_material_update.npy')
pat_mtx = np.load('weave_pattern_update.npy')
phy_vec = np.load('physics_vector_update.npy')
phy_norm_vec = np.load('physics_norm_vector_update.npy')

#%% Normalize physics vector

# phy_norm_vec[:,0] = phy_vec[:,0] / np.max(phy_vec[:,0])
# phy_norm_vec[:,1] = phy_vec[:,1] / np.max(phy_vec[:,1])
# phy_norm_vec[:,2] = phy_vec[:,2] / np.max(phy_vec[:,2])

phy_norm_vec[:,0] = phy_vec[:,0] / 1e9
phy_norm_vec[:,1] = phy_vec[:,1] / 1e9
phy_norm_vec[:,2] = phy_vec[:,2] / 1e8
            
#%% Data Preprocessing

P_train, P_test, M_train, M_test, V_train, V_test = train_test_split(pat_mtx, mat_mtx, phy_norm_vec, test_size=0.2, random_state=64)
P_test,  P_cv,   M_test,  M_cv,   V_test,  V_cv   = train_test_split(P_test,  M_test,  V_test,  test_size=0.5, random_state=64)

input_train_p  = tf.reshape(P_train, [-1, 6, 6, 1])
input_train_m  = tf.reshape(M_train, [-1, 6, 6, 1])
output_train   = tf.reshape(V_train, [-1, 3])

input_cv_p  = tf.reshape(P_cv, [-1, 6, 6, 1])
input_cv_m  = tf.reshape(M_cv, [-1, 6, 6, 1])
output_cv   = tf.reshape(V_cv, [-1, 3])

input_test_p  = tf.reshape(P_test, [-1, 6, 6, 1])
input_test_m  = tf.reshape(M_test, [-1, 6, 6, 1])
output_test   = tf.reshape(V_test, [-1, 3])

#%% Define Convolutional Network Functions

def conv_relu_block(x,filt,names):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[2,2],strides=2,
                               padding='same', activation='linear', 
                               use_bias=True,name=names)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    return y

def conv_unit_block(x,filt,names):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[2,2],strides=1,
                               padding='same', activation='linear', 
                               use_bias=True,name=names)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    return y

def se_block(x,filt,ratio=16):
    
    init = x
    se_shape = (1, 1, filt)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filt // ratio, activation='relu', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.Dense(filt, activation='sigmoid', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)    
    se = tf.keras.layers.multiply([init, se])
    
    return se
    
def me_block(x,filt,ratio=16):
    
    init = x
    me_shape = (1, 1, filt)
    
    me = tf.keras.layers.GlobalMaxPooling2D()(init)
    me = tf.keras.layers.Reshape(me_shape)(me)
    me = tf.keras.layers.Dense(filt // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(me)
    me = tf.keras.layers.Dense(filt, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(me)
    
    me = tf.keras.layers.multiply([init, me])
    
    return me

def resnet_block(x,filt):

    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = se_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def maxnet_block(x,filt):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = me_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='relu', 
                              kernel_initializer='he_normal', use_bias=True,
                              name = names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)
    
    return y    
    
from tensorflow.keras import initializers

def dense_relu_block_1(x,filt):
    
    y = tf.keras.layers.Dense(filt, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='linear', use_bias=True, bias_initializer=initializers.Zeros())(x)
    
    y = tf.keras.layers.ReLU()(y)
    
    return y

#%% Set Up CNN Structures   

input_layer_1 = tf.keras.Input(shape=(6, 6, 1))
input_layer_2 = tf.keras.Input(shape=(6, 6, 1))

conv_11  = conv_relu_block(input_layer_1, 64, 'conv11')
conv_21  = conv_relu_block(conv_11, 128, 'conv21')
conv_31  = conv_relu_block(conv_21, 128, 'conv31')
reshape1 = tf.keras.layers.GlobalAveragePooling2D()(conv_31)

conv_12  = conv_relu_block(input_layer_2, 64, 'conv12')
conv_22  = conv_relu_block(conv_12, 128, 'conv22')
conv_32  = conv_relu_block(conv_22, 128, 'conv32')
reshape2 = tf.keras.layers.GlobalAveragePooling2D()(conv_32)

concat   = tf.keras.layers.Concatenate(axis=1)([reshape1, reshape2])

dense_1  = dense_relu_block_1(concat,  128)
dense_2  = dense_relu_block_1(dense_1, 128)
output_layer  = dense_relu_block_1(dense_2,  3)

model = tf.keras.models.Model([input_layer_1, input_layer_2], output_layer)
model.summary()

#%% Training the model

sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

epoch = 200
history = model.fit([input_train_p, input_train_m], output_train, batch_size=64, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_p, input_cv_m], output_cv))

predict = model.predict([input_test_p, input_test_m])
score = model.evaluate([input_test_p, input_test_m], output_test, verbose=1)
print('\n', 'Test accuracy', score)

#%% Generating history plots of training ##

# Summarize history for accuracy
fig_acc = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.yscale('log')
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('training_accuracy.png')

# Summarize history for loss
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')

# Summarize history for loss
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')

#%% Calculate the prediction error

[p1,p2] = predict.shape

error_0 = np.zeros(p1)
error_1 = np.zeros(p1)
error_2 = np.zeros(p1)

for ip in range(p1):
    
    error_0[ip] = (np.abs(predict[ip,0] - output_test[ip,0])) / output_test[ip,0]
    error_1[ip] = (np.abs(predict[ip,1] - output_test[ip,1])) / output_test[ip,1]
    error_2[ip] = (np.abs(predict[ip,2] - output_test[ip,2])) / output_test[ip,2]
    
error_0_ave = np.mean(error_0)
error_1_ave = np.mean(error_1)
error_2_ave = np.mean(error_2)

print("Error rate of E1 is: ",  error_0_ave)
print("Error rate of E2 is: ",  error_1_ave)
print("Error rate of G is: ",   error_2_ave)

#%% Save the model

model.save('Weave_bimat_CNN_update.h5')

#%% Plot the prediction

plt.figure()
plt.plot(output_test[:,0], 'r-')
plt.plot(predict[:,0], 'b-')
plt.title('E1 prediction')
plt.legend(['Real E1 data','Predicted E1 data'])

plt.figure()
plt.plot(output_test[:,1], 'r-')
plt.plot(predict[:,1], 'b-')
plt.title('E2 prediction')
plt.legend(['Real E2 data','Predicted E2 data'])

plt.figure()
plt.plot(output_test[:,2], 'r-')
plt.plot(predict[:,2], 'b-')
plt.title('G12 prediction')
plt.legend(['Real G12 data','Predicted G12 data'])