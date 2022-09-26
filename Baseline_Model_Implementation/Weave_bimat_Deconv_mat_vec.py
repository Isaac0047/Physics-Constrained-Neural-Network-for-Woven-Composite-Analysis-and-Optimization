import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
import re

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

data_set  = 9000
data_new  = np.zeros([data_set, 6, 6])
stiff_new = np.zeros([data_set, 1])

mat_mtx = np.load('weave_material.npy')
mat_vec = np.load('weave_material_vector.npy')
pat_mtx = np.load('weave_pattern.npy')
phy_vec = np.load('physics_vector.npy')
phy_norm_vec = np.load('physics_norm_vector.npy')

#%% Normalize physics vector

# phy_norm_vec[:,0] = phy_vec[:,0] / np.max(phy_vec[:,0])
# phy_norm_vec[:,1] = phy_vec[:,1] / np.max(phy_vec[:,1])
# phy_norm_vec[:,2] = phy_vec[:,2] / np.max(phy_vec[:,2])

phy_norm_vec[:,0] = phy_vec[:,0] / 1e9
phy_norm_vec[:,1] = phy_vec[:,1] / 1e9
phy_norm_vec[:,2] = phy_vec[:,2] / 1e8

#%%  

idx_3000 = np.random.randint(9000, size=3000)
idx_6000 = np.random.randint(9000, size=6000)

mat_mtx_3000 = mat_mtx[idx_3000, :, :]
mat_vec_3000 = mat_vec[idx_3000, :]
pat_mtx_3000 = pat_mtx[idx_3000, :, :]
phy_vec_3000 = phy_vec[idx_3000, :]
phy_norm_vec_3000 = phy_norm_vec[idx_3000, :]

mat_mtx_6000 = mat_mtx[idx_6000, :]
mat_vec_6000 = mat_vec[idx_6000, :]
pat_mtx_6000 = pat_mtx[idx_6000, :]
phy_vec_6000 = phy_vec[idx_6000, :]
phy_norm_vec_6000 = phy_norm_vec[idx_6000, :]

#%% Split the material vector

[m1,m2] = mat_vec.shape
mat_vec_1     = np.zeros((m1, int(m2/2)))
mat_vec_1_sum = np.zeros((m1)) 
mat_vec_2     = np.zeros((m1, int(m2/2)))
mat_vec_2_sum = np.zeros((m1))

def binary_encoder(vector, length):
    
    energy= 0
    
    for ii in range(length):
        
        energy = energy + 2**(ii) * vector[ii]
        
    return energy

for i in range(m1):
    
    mat_vec_1[i,:]   = mat_vec[i,0:int(m2/2)]
    mat_vec_2[i,:]   = mat_vec[i,int(m2/2):]
    mat_vec_1_sum[i] = binary_encoder(mat_vec_1[i,:], int(m2/2))
    mat_vec_2_sum[i] = binary_encoder(mat_vec_2[i,:], int(m2/2))

#%% Concatenate the material vector sum
mat_vec_sum = np.array([mat_vec_1_sum, mat_vec_2_sum]).T

#%% Normalize physics vector

# phy_norm_vec[:,0] = phy_vec[:,0] / np.max(phy_vec[:,0])
# phy_norm_vec[:,1] = phy_vec[:,1] / np.max(phy_vec[:,1])
# phy_norm_vec[:,2] = phy_vec[:,2] / np.max(phy_vec[:,2])

phy_norm_vec[:,0] = phy_vec[:,0] / 1e9
phy_norm_vec[:,1] = phy_vec[:,1] / 1e9
phy_norm_vec[:,2] = phy_vec[:,2] / 1e8

#%% Data Preprocessing

P_train, P_test, M_train, M_test, V_train, V_test = train_test_split(pat_mtx_6000, mat_vec_6000, phy_norm_vec_6000, test_size=0.2, random_state=64)
P_test,  P_cv,   M_test,  M_cv,   V_test,  V_cv   = train_test_split(P_test,  M_test,  V_test,       test_size=0.5, random_state=64)

input_train_m  = tf.reshape(M_train, [-1, 12])
input_train_v  = tf.reshape(V_train, [-1, 3])
input_train_p   = tf.reshape(P_train, [-1, 6, 6, 1])

input_cv_m     = tf.reshape(M_cv, [-1, 12])
input_cv_v     = tf.reshape(V_cv, [-1, 3])
input_cv_p     = tf.reshape(P_cv, [-1, 6, 6, 1])

input_test_m   = tf.reshape(M_test, [-1, 12])
input_test_v   = tf.reshape(V_test, [-1, 3])
input_test_p   = tf.reshape(P_test, [-1, 6, 6, 1])

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


def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear', 
                              kernel_initializer='he_normal', use_bias=True,
                              name = names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    return y    

def dense_sigmoid_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='sigmoid', 
                              kernel_initializer='he_normal', use_bias=True,
                              name = names)(x)
    
    #y = tf.keras.layers.BatchNormalization()(y)
    
    return y    


#%% Set Up CNN Structures   

input_layer_1 = tf.keras.Input(shape=(3))
dense_1       = dense_block(input_layer_1, 64, 'dense0')
# dropout_1     = tf.keras.layers.Dropout(0.5)(dense_1)
dense_2       = dense_block(dense_1, 128, 'dense1')
# dropout_2     = tf.keras.layers.Dropout(0.5)(dense_2)
dense_3       = dense_block(dense_2, 128, 'dense2')
# dropout_3     = tf.keras.layers.Dropout(0.5)(dense_3)

input_layer_2 = tf.keras.Input(shape=(6, 6, 1))
conv_12  = conv_relu_block(input_layer_2, 64, 'conv12')
conv_22  = conv_relu_block(conv_12, 128, 'conv22')
conv_32  = conv_relu_block(conv_22, 128, 'conv32')
flat     = tf.keras.layers.GlobalAveragePooling2D()(conv_32)

concat = tf.keras.layers.Concatenate(axis=1)([dense_3, flat])

dense_out_1 = dense_block(concat, 128, 'dense_out1')
dense_out_2 = dense_block(dense_out_1, 64, 'dense_out2')
# dense_out_3 = dense_block(dense_out_2, 32, 'dense_out3')

output_layer = dense_sigmoid_block(dense_out_2, 12, 'output')

# output_layer = tf.keras.backend.round(output_layer)

model = tf.keras.models.Model([input_layer_1,input_layer_2], output_layer)
model.summary()

#%% Training the model

def custom_loss_function(y_true, y_pred):
    losses = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    binary_loss = tf.reduce_sum(losses,[1,2])
    return binary_loss

import tensorflow.keras.backend as K

gamma = 0.1
def keras_custom_loss_function(y_true, y_pred):
    # custom_loss_value = K.mean(tf.reduce_sum(K.square(y_pred-y_true), axis=[0,1,2,3]) / m_x / m_y)
    custom_loss_value = K.mean(tf.reduce_sum(K.square(y_pred-y_true)))
    return custom_loss_value

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.8)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,   beta_2=0.999, decay=1e-6)
sgd  = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.6, nesterov=True)
sgd1 = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# model.compile(optimizer=sgd1, loss=keras_custom_loss_function, metrics=['accuracy'])
    
# model.compile(optimizer='adam', loss=custom_loss_function, metrics=['accuracy'])
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

# sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
# model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

epoch = 150
history = model.fit([input_train_v, input_train_p], input_train_m, batch_size=128, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_v,input_cv_p], input_cv_m))

predict = model.predict([input_test_v,input_test_p])
score   = model.evaluate([input_test_v, input_test_p], input_test_m, verbose=1)
print('\n', 'Test accuracy', score)

#%% Material Matrix Visualization

def mat_form(vec,n1,n2):
    mat_new = np.zeros((n1,n2))
    
    for i in range(n1):
        for j in range(n2):
            
            if vec[i] == 0 and vec[n1+j] == 0:
                mat_new[i][j] = 0
            elif vec[i] == 1 and vec[n1+j] == 1:
                mat_new[i][j] = 1
            elif vec[i] == 0 and vec[n1+j] == 1:
                mat_new[i][j] = 2
            elif vec[i] == 1 and vec[n1+j] == 0:
                mat_new[i][j] = 3
                
    return mat_new

#%% Print out accuracy

predict = model.predict([input_test_v,input_test_p])
predict = np.round(predict)

plt.figure()
plt.plot(predict[:,0],'r-')
plt.plot(input_test_m[:,0],'b-')
plt.title('First Mat Vec Prediction')
plt.legend(['predicted value','true value'])

plt.figure()
plt.plot(predict[:,1],'r-')
plt.plot(input_test_m[:,1],'b-')
plt.title('Second Mat Vec Prediction')
plt.legend(['predicted value','true value'])

#%% Binary Decoder Function

def binary_decoder(x, max_order=6):
    
    decoder_vec = np.zeros(max_order)
    
    for i in reversed(range(max_order)):
        
        if x >= 2**(i):
            decoder_vec[max_order-1-i] = 1
            x = x-2**(i)
        else:
            continue
        
    return np.flip(decoder_vec)
        
#%% Test Binary Decoder function

print('Decoder for 60 is:', binary_decoder(60))
print('Decoder for 45 is:', binary_decoder(45))
print('Decoder for 27 is:', binary_decoder(27))
print('Decoder for 11 is:', binary_decoder(11))
print('Decoder for  1 is:', binary_decoder(1))

#%% Convert the prediction back to material vector

mat_1 = predict[:,0]
mat_2 = predict[:,1]

mat_1_real = input_test_m[:,0]
mat_2_real = input_test_m[:,1]

mm = len(mat_1)

mat_1_decode = np.zeros((mm, 6))
mat_2_decode = np.zeros((mm, 6))

mat_1_real_decode = np.zeros((mm, 6))
mat_2_real_decode = np.zeros((mm, 6))

for ii in range(mm):
    mat_1_decode[ii,:] = binary_decoder(int(mat_1[ii]))
    mat_2_decode[ii,:] = binary_decoder(int(mat_2[ii]))
    
    mat_1_real_decode[ii,:] = binary_decoder(int(mat_1_real[ii]))
    mat_2_real_decode[ii,:] = binary_decoder(int(mat_2_real[ii]))

mat_con = np.concatenate((mat_1_decode, mat_2_decode), axis=1)
mat_con_real = np.concatenate((mat_1_real_decode, mat_2_real_decode), axis=1)

mat_diff = mat_con_real - mat_con

#%% Calculate prediction error
mat_error = np.count_nonzero(mat_diff) / mat_diff.shape[0] / mat_diff.shape[1]
# print('\n', 'Prediction Element-wise Error:', prediction_error_ave)

#%% Prediction accuracy test
# [p1, p2] = mat_diff.shape

# predict_round = tf.keras.backend.round(predict)

predict_error = np.zeros([mm, 1])

for k in range(mm):
    # predict_error[k] = tf.math.count_nonzero(tf.math.subtract(output_test[k,:,:,0],predict_round[k,:,:,0]))
    # predict1 = predict[k,:]
    predict1 = mat_form(mat_con[k,:], 6, 6)
    true_mat = mat_form(mat_con_real[k,:], 6, 6)
    
    # predict_round = tf.math.round(predict1)
    # predict_round = tf.keras.backend.cast(predict_round, dtype='float64')
    
    predict_error[k] = tf.math.count_nonzero(true_mat-predict1)
    
prediction_error_ave = np.mean(predict_error / 36) 
print('\n', 'Prediction Element-wise Error:', prediction_error_ave)

#%% Training the model

# The first dataset
Y_test_1 = input_test_m[0, :]
Y_test_1_mtx = mat_form(Y_test_1, 6, 6)

fig1_test = plt.figure()
plt.title('Weave Pattern')
plt.imshow(Y_test_1_mtx)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Weave_test_1.png')

#%%
predict_1 = predict[0, :]
predict_round = tf.math.round(predict_1)
# predict_1 = np.reshape(predict_1, [12])
predict_1_mtx = mat_form(predict_round, 6, 6)
# predict_1 = tf.keras.backend.round(predict_1)
fig1_pred=plt.figure()
plt.title('Trained Weave Pattern')
plt.imshow(predict_1_mtx)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Weave_predict_1.png')

#%% Plot the training loss and accruacy

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

#%% Summarize history for loss
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')

#%% Print out accuracy

generated_image = model.predict([input_test_v,input_test_p])
generated_image = np.round(generated_image)
accuracy = np.zeros((generated_image.shape[0],1))

# Convert the material vector to matrix

gen_image_mtx = np.zeros([mm,6,6])

for iii in range(mm):
    predict_norm = mat_con[iii,:]
    gen_image_mtx[iii,:,:] = mat_form(predict_norm, 6, 6)

#%% Validate prediction

new_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

# Prediction: p + M -> V

physics_prediction = new_model.predict([input_test_p, np.round(predict)])

error_rate_0 = np.mean(np.abs(physics_prediction[:,0] - input_test_v[:,0]) / input_test_v[:,0])
error_rate_1 = np.mean(np.abs(physics_prediction[:,1] - input_test_v[:,1]) / input_test_v[:,1])
error_rate_2 = np.mean(np.abs(physics_prediction[:,2] - input_test_v[:,2]) / input_test_v[:,2])

print('The absolute prediction error for E1 is: ', error_rate_0)
print('The absolute prediction error for E2 is: ', error_rate_1)
print('The absolute prediction error for G12 is: ', error_rate_2)