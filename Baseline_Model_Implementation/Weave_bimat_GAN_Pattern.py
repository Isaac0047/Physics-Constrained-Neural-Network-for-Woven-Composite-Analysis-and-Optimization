# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:32:29 2020

@author: Haoti

"""

#%% Import modules

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
import re
from numpy import vstack

#%% SECTION TO RUN WITH GPU 

# Choose GPU to use
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True

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

mat_mtx_6000 = mat_mtx[idx_6000, :, :]
mat_vec_6000 = mat_vec[idx_6000, :]
pat_mtx_6000 = pat_mtx[idx_6000, :, :]
phy_vec_6000 = phy_vec[idx_6000, :]
phy_norm_vec_6000 = phy_norm_vec[idx_6000, :]

#%% Data Preprocessing

P_train, P_test, M_train, M_test, V_train, V_test = train_test_split(pat_mtx_6000, mat_vec_6000, phy_norm_vec_6000, test_size=0.2, random_state=64)
P_test,  P_cv,   M_test,  M_cv,   V_test,  V_cv   = train_test_split(P_test,  M_test,  V_test,  test_size=0.5, random_state=64)

input_train_m  = tf.reshape(M_train, [-1, 12])
input_train_v  = tf.reshape(V_train, [-1, 3])
output_train   = tf.reshape(P_train, [-1, 6, 6, 1])

input_cv_m     = tf.reshape(M_cv, [-1, 12])
input_cv_v     = tf.reshape(V_cv, [-1, 3])
output_cv      = tf.reshape(P_cv, [-1, 6, 6, 1])

input_test_m   = tf.reshape(M_test, [-1, 12])
input_test_v   = tf.reshape(V_test, [-1, 3])
output_test    = tf.reshape(P_test, [-1, 6, 6, 1])

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

from tensorflow.keras import initializers

def dense_relu_block_1(x,filt):
    
    y = tf.keras.layers.Dense(filt, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='linear', use_bias=True, bias_initializer=initializers.Zeros())(x)
    
    y = tf.keras.layers.ReLU()(y)
    
    return y

#%% Set up Network framework

batch_size = 64
n_noise    = 64

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x*tf.log(z+eps) + (1.-x)*tf.log(1.-z+eps)))

#%% Defining models

def make_discriminator_model():
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], strides=(2,2), padding='same', input_shape=[6, 6, 1]))
    model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], strides=(2,2), padding='same'))
    model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=[3,3], strides=(2,2), padding='same'))
    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape([-1]))
    
    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

de_model = make_discriminator_model()
de_model.summary()

def make_generator_model():
    
    input_layer_1 = tf.keras.Input(shape=(3))
    dense_1       = dense_block(input_layer_1, 64, 'dense0')
    dense_2       = dense_block(dense_1, 128, 'dense1')
    dense_3       = dense_block(dense_2, 128, 'dense2')

    input_layer_2 = tf.keras.Input(shape=(12))
    dense_21       = dense_block(input_layer_2, 64, 'dense20')
    dense_22       = dense_block(dense_21, 128, 'dense21')
    dense_23       = dense_block(dense_22, 128, 'dense22')
    
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

#%% Test generator model

# generator = make_generator_model()
generated_image = ge_model([input_train_v, input_train_m], training=False)

#%% Test discriminator model

# discriminator = make_discriminator_model()
decision        = de_model(generated_image)

#%% Define loss function and optimizers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#%% Define Discriminator Loss

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#%% Define Generator Loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#%% Generate models

def define_gan(g_model, d_model):
    # make weights in discriminator not trainable
    
    d_model.trainable = False
    
    input_1 = tf.keras.Input(shape=(3,1))
    input_2 = tf.keras.Input(shape=(12,1))
    
    # inter_1 = tf.keras.Input(shape=(6,6,1))
    
    inter_output = g_model([input_1, input_2])
    
    output = d_model(inter_output)
    
    gan_model = tf.keras.models.Model([input_1,input_2], output)
    
    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return gan_model

gan_model = define_gan(ge_model, de_model)
gan_decision = gan_model([input_train_v,input_train_m])

#%% Select real samples

def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # print(ix)
    X  = dataset[ix,:]
    Y  = np.ones((n_samples, 1))
    return X, Y

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim, n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, dataset1, dataset2, n_samples):
    # generate points in latent space
    ix = np.random.randint(0, dataset1.shape[0], n_samples)
    
    x_input1 = dataset1[ix,:]
    x_input2 = dataset2[ix,:]
    # predict output
    X = g_model.predict([x_input1,x_input2])
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y    

#%% Create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

#%% Evaluate the discriminator, plot generated images, save generator model

def summarize_performance(epoch, g_model, d_model, dataset_P, dataset_M, dataset_V, latent_dim, n_samples=128):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset_P, n_samples)
    # evaluate discriminator on real examples
    _, acc_real    = d_model.evaluate(np.expand_dims(X_real, axis=3), y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, dataset_V, dataset_M, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake    = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

#%% Train the generator and discriminator

def train(g_model, d_model, gan_model, dataset_P, dataset_M, dataset_V, latent_dim, n_epochs=100, n_batch=128):
    
    bat_per_epo = int(dataset_P.shape[0] / n_batch)
    half_batch  = int(n_batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            
            X_real, y_real = generate_real_samples(dataset_P, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, dataset_V, dataset_M, half_batch)
            
            X, y = vstack((np.expand_dims(X_real, axis=3), X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            
            X_gan_1, y_gan_1 = generate_real_samples(dataset_V, n_batch)
            X_gan_2, y_gan_2 = generate_real_samples(dataset_M, n_batch)
            
            g_loss = gan_model.train_on_batch([X_gan_1,X_gan_2], y_gan_1)
            
            # print(d_loss)
            # print(g_loss[0])
            
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss[0]))
            
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset_P, dataset_M, dataset_V, latent_dim)
            

#%% Training process
# size of the latent space
latent_dim = 3
# load image data
datasetp = P_train         # 6-by-6
datasetm = M_train         # 6-by-6
datasetv = V_train         # 1-by-3

#%% create the discriminator
d_model = make_discriminator_model()
# create the generator
g_model = make_generator_model()
# create the gan
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, datasetp, datasetm, datasetv, latent_dim)

#%% Training sample validation

P1,  P2  = generate_real_samples(P_train, 32)
VM1, VM2 = generate_fake_samples(g_model, V_train, M_train, 32)
V1,  V2  = generate_real_samples(V_train, 32)
M1,  M2  = generate_real_samples(M_train, 32)

#%%
d_model.train_on_batch(np.expand_dims(P1, axis=3), np.ones((32,1)))

#%%
gan_model.train_on_batch([V1, M1], np.ones((32,1)))

#%% Print out accuracy

generated_image = g_model.predict([V_test,M_test])
generated_image = np.round(generated_image)
accuracy = np.zeros((generated_image.shape[0],1))

#%%
for i in range (generated_image.shape[0]):
    accuracy[i] = 1-np.sum(np.abs(generated_image[i,:,:,0]-P_test[i,:,:]))/36
    
Accuracy = np.mean(accuracy)
print('The prediction accuracy is', Accuracy)

# Prediction image sample
plt.title('ML prediction contour')
plt.imshow(generated_image[0,:,:,0])
plt.colorbar()
plt.grid(True)
plt.show()

# Real image sample
plt.title('Real contour')
plt.imshow(P_test[0,:,:])
plt.colorbar()
plt.grid(True)
plt.show()

#%% Validate prediction

new_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

# Prediction: p + M -> V

physics_prediction = new_model.predict([np.round(generated_image[:,:,:,0]), M_test])

error_rate_0 = np.mean(np.abs(physics_prediction[:,0] - V_test[:,0]) / V_test[:,0])
error_rate_1 = np.mean(np.abs(physics_prediction[:,1] - V_test[:,1]) / V_test[:,1])
error_rate_2 = np.mean(np.abs(physics_prediction[:,2] - V_test[:,2]) / V_test[:,2])

print('The absolute prediction error for E1 is: ', error_rate_0)
print('The absolute prediction error for E2 is: ', error_rate_1)
print('The absolute prediction error for G12 is: ', error_rate_2)



