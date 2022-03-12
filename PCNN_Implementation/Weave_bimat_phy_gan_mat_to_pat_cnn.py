import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
import re

from scipy.interpolate import interp1d
from scipy import io

#%% SECTION TO RUN WITH GPU

# Choose GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True

#%% Define Parameters

boxW = 0.25 + 0.025*2
boxL = 0.25 + 0.025*2
boxH = 1

def Ogden_func(x, w):
    
    p1 = 2*w[0]/w[3] * (x**(w[0]-1) - x**(-0.5*w[3]-1))
    p2 = 2*w[1]/w[4] * (x**(w[1]-1) - x**(-0.5*w[4]-1))
    p3 = 2*w[2]/w[5] * (x**(w[2]-1) - x**(-0.5*w[5]-1))
    
    return p1 + p2 + p3

#%% Load data
size_x = 6
size_y = 6

data_set  = 9000
data_new  = np.zeros([data_set, 6, 6])
stiff_new = np.zeros([data_set, 1])

mat_mtx = np.load('weave_material_update.npy')
mat_vec = np.load('weave_material_vector_update.npy')
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


#%% Calculating the FFT max value

def fourier_transform(image):
    
    img_fft   = np.fft.fftshift(np.fft.fft2((image)))
    trans_img = abs(np.fft.ifft2(img_fft))
    norm_fft  = np.abs(img_fft)
    
    return img_fft, trans_img, norm_fft

mat_fft = np.zeros((mat_mtx.shape[0],mat_mtx.shape[1],mat_mtx.shape[2]))
pat_fft = np.zeros((pat_mtx.shape[0],pat_mtx.shape[1],pat_mtx.shape[2]))
mat_fft_max = np.zeros((mat_mtx.shape[0]))
pat_fft_max = np.zeros((mat_mtx.shape[0]))

for ii in range(mat_mtx.shape[0]):
    
    target_mat = mat_mtx[ii,:,:]
    img_fft_mat, trans_mat, norm_fft_mat = fourier_transform(target_mat)
    mat_fft[ii,:,:] = norm_fft_mat
    mat_fft_max[ii] = np.max(norm_fft_mat)
    
    target_pat = pat_mtx[ii,:,:]
    img_fft_pat, trans_pat, norm_fft_pat = fourier_transform(target_pat)
    pat_fft[ii,:,:] = norm_fft_pat
    pat_fft_max[ii] = np.max(norm_fft_pat)

#%% Calculating the Image ratio

def pattern_ratio(img):
    
    ratio = np.sum(img) / 36
    
    return ratio

mat_ratio = np.zeros((mat_mtx.shape[0]))
pat_ratio = np.zeros((pat_mtx.shape[0]))

for ii in range(pat_mtx.shape[0]):
    
    mat_ratio[ii] = np.sum(mat_mtx[ii]) / 36
    pat_ratio[ii] = np.sum(pat_mtx[ii]) / 36

#%% Data Preprocessing

P_train, P_test, M_train, M_test, V_train, V_test, F_train, F_test = train_test_split(pat_mtx, mat_vec, phy_norm_vec, pat_ratio, test_size=0.2, random_state=64)
P_test,  P_cv,   M_test,  M_cv,   V_test,  V_cv,   F_test,  F_cv   = train_test_split(P_test,  M_test,  V_test,       F_test,      test_size=0.5, random_state=64)

input_train_m  = tf.reshape(M_train, [-1, 12])
input_train_v  = tf.reshape(V_train, [-1, 3])
input_train_p  = tf.reshape(P_train, [-1, 6, 6, 1])
output_train   = tf.reshape(F_train, [-1, 1])

input_cv_m     = tf.reshape(M_cv, [-1, 12])
input_cv_v     = tf.reshape(V_cv, [-1, 3])
input_cv_p     = tf.reshape(P_cv, [-1, 6, 6, 1])
output_cv      = tf.reshape(F_cv, [-1, 1])

input_test_m   = tf.reshape(M_test, [-1, 12])
input_test_v   = tf.reshape(V_test, [-1, 3])
input_test_p   = tf.reshape(P_test, [-1, 6, 6, 1])
output_test    = tf.reshape(F_test, [-1, 1])

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
                              kernel_initializer='he_normal', use_bias=False,
                              name = names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)
    
    return y    

def deconv_norm_linear(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='valid',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='linear')(y)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def deconv_norm_sigmoid(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='valid',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='sigmoid')(y)
    
    # y = tf.keras.layers.BatchNormalization()(y)

    return y

def deconv_block(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='same',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

#%% Set up Network framework

batch_size = 64
n_noise    = 64

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x*tf.log(z+eps) + (1.-x)*tf.log(1.-z+eps)))

#%% Neural Network Parameters

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,   beta_2=0.999, decay=1e-6)
sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)

import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#%% Decoder models

def make_discriminator_model():
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], strides=(2,2), padding='same', input_shape=[6, 6, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

de_model = make_discriminator_model()
de_model.summary()

#%% Encoder models

def make_generator_model():
    
    input_layer_1 = tf.keras.Input(shape=(3))
    dense_1       = dense_block(input_layer_1, 64, 'dense0')
    dense_2       = dense_block(dense_1, 128, 'dense1')
    dense_3       = dense_block(dense_2, 128, 'dense2')

    input_layer_2 = tf.keras.Input(shape=(12))
    dense_21       = dense_block(input_layer_2, 64, 'dense20')
    dense_22       = dense_block(dense_21, 128, 'dense21')
    dense_23       = dense_block(dense_22, 128, 'dense22')
    # flat          = tf.keras.layers.GlobalAveragePooling2D()(conv_32)
    
    concat  = tf.keras.layers.Concatenate(axis=1)([dense_3, dense_23])
    reshape = tf.keras.layers.Reshape((1,1,256), input_shape=(256,))(concat)

    deconv_1 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=[2,2],
        strides=(1,1),padding='valid',activation='relu', use_bias=True,
        name='deconv1')(reshape)
    
    deconv_2 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=[2,2],
        strides=(1,1),padding='valid',activation='relu', use_bias=True,
        name='deconv2')(deconv_1)
    
    output_layer = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=[2,2],
        strides=(2,2),padding='valid',activation='sigmoid', use_bias=True,
        name='deconv3')(deconv_2)
     
    ge_model = tf.keras.models.Model([input_layer_1,input_layer_2], output_layer)
    
    return ge_model

ge_model = make_generator_model()
ge_model.summary()

#%% Load discriminator model

de_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

#%% Setting up Convolutional Neural Network

def define_gan(g_model, d_model):
    
    d_model.trainable = False
    
    input_1 = tf.keras.Input(shape=(3))
    input_2 = tf.keras.Input(shape=(12))
    
    inter_output = g_model([input_1, input_2])
    inter_output_round = tf.keras.backend.round(inter_output)
    
    output = d_model([inter_output,input_2])
    
    gan_model = tf.keras.models.Model(inputs=[input_1,input_2], outputs=[inter_output, output])
    
    return gan_model

gan_model = define_gan(ge_model, de_model)
gan_decision = gan_model([input_train_v,input_train_m])

#%% Train the AE network 

gan_model.compile(optimizer=opt, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_squared_error], metrics=['accuracy'], loss_weights=[1, 5])

epoch = 150
ae_history  = gan_model.fit([input_train_v,input_train_m], [input_train_p, input_train_v], batch_size=64, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_v,input_cv_m], [input_cv_p, input_cv_v]))

ae_predict  = gan_model.predict([input_test_v,input_test_m])

ae_score    = gan_model.evaluate([input_test_v,input_test_m], [input_test_p, input_test_v], verbose=1)
print('\n', 'Test accuracy', ae_score)

#%% Save the model

# gan_model.save('gan_mat_vec_to_pat.h5')

#%% Test if decoder model changes

de_pred = de_model([input_test_p,input_test_m])

plt.figure()

plt.plot(output_test)
plt.plot(de_pred)
plt.title('Decoder Model Checking')
plt.legend(['True Mat Ratio','Pred Mat Ratio'])

plt.figure()
plt.plot(ae_predict[1])
plt.plot(output_test)
plt.plot(ae_predict[1]-output_test)
plt.legend(['Predicted Ratio','True Ratio','Differences'])

#%% Training the model

# opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-6)
# sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
# model.compile(optimizer=sgd, loss={'model_4':'mean_squared_error', 'model_3':'mean_squared_error'}, metrics=['accuracy'])

# epoch = 40
# history = model.fit(input_train, [inter_train, output_train], batch_size=64, epochs=epoch, 
#                     steps_per_epoch=40, validation_data=(input_cv, [inter_cv, output_cv]))

# predict = model.predict(input_test)

# score = model.evaluate(input_test, [inter_test, output_test], verbose=1)
# print('\n', 'Test accuracy', score)

#%% Generating history plots of training

#Summarize history for accuracy
# fig_acc = plt.figure()
# plt.plot(ae_history.history['accuracy'])
# plt.plot(ae_history.history['val_accuracy'])
# plt.title('ae_model accuracy in training')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# fig_acc_log = plt.figure()
# plt.plot(ae_history.history['accuracy'])
# plt.plot(ae_history.history['val_accuracy'])
# plt.title('ae_model accuracy in training')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.yscale('log')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # Summarize history for loss
# fig_loss_log = plt.figure()
# plt.plot(ae_history.history['loss'])
# plt.plot(ae_history.history['val_loss'])
# plt.title('ae_model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.yscale('log')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# fig_loss = plt.figure()
# plt.plot(ae_history.history['loss'])
# plt.plot(ae_history.history['val_loss'])
# plt.title('ae_model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#%% Prediction accuracy test
[p1, p2, p3, p4] = ae_predict[0].shape
# predict_round = tf.keras.backend.round(predict)
predict_round = tf.math.round(ae_predict[0])
predict_round = tf.keras.backend.cast(predict_round, dtype='float64')
predict_error = np.zeros([p1, 1])

for k in range(p1):
    # predict_error[k] = tf.math.count_nonzero(tf.math.subtract(output_test[k,:,:,0],predict_round[k,:,:,0]))
    predict_error[k] = tf.math.count_nonzero(input_test_p[k,:,:,0]-predict_round[k,:,:,0])
    
prediction_error_ave = np.mean(predict_error / 36) 
print('\n', 'Prediction Element-wise Error:', prediction_error_ave)  
        

#%% Training the model

# The first dataset
Y_test_1 = input_test_p[0, :, :, 0]

fig1_test = plt.figure()
plt.title('Weave Pattern')
plt.imshow(Y_test_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Weave_test_1.png')

predict_1 = predict_round[0, :, :, 0]
# predict_1 = tf.keras.backend.round(predict_1)
fig1_pred=plt.figure()
plt.title('Trained Weave Pattern')
plt.imshow(predict_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Weave_predict_1.png')

# The second dataset
Y_test_1 = input_test_p[10, :, :, 0]

fig1_test = plt.figure()
plt.title('Weave Pattern')
plt.imshow(Y_test_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Weave_test_1.png')

predict_1 = predict_round[10, :, :, 0]
# predict_1 = tf.keras.backend.round(predict_1)
fig1_pred=plt.figure()
plt.title('Trained Weave Pattern')
plt.imshow(predict_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Weave_predict_1.png')

# The second dataset
Y_test_1 = input_test_p[100, :, :, 0]

fig1_test = plt.figure()
plt.title('Weave Pattern')
plt.imshow(Y_test_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Weave_test_1.png')

predict_1 = predict_round[100, :, :, 0]
# predict_1 = tf.keras.backend.round(predict_1)
fig1_pred=plt.figure()
plt.title('Trained Weave Pattern')
plt.imshow(predict_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Weave_predict_1.png')

#%% Validate prediction

new_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

# Prediction: p + M -> V

physics_prediction = new_model.predict([np.round(ae_predict[0]), input_test_m])

error_rate_0 = np.mean(np.abs(physics_prediction[:,0] - input_test_v[:,0]) / input_test_v[:,0])
error_rate_1 = np.mean(np.abs(physics_prediction[:,1] - input_test_v[:,1]) / input_test_v[:,1])
error_rate_2 = np.mean(np.abs(physics_prediction[:,2] - input_test_v[:,2]) / input_test_v[:,2])

print('The absolute prediction error for E1 is: ', error_rate_0)
print('The absolute prediction error for E2 is: ', error_rate_1)
print('The absolute prediction error for G12 is: ', error_rate_2)

        
        
        
        
        


        

