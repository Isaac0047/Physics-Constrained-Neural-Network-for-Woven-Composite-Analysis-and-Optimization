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

def dense_sigmoid_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='sigmoid', 
                              kernel_initializer='he_normal', use_bias=True,
                              name = names)(x)
    
    #y = tf.keras.layers.BatchNormalization()(y)
    
    return y    

from tensorflow.keras import initializers

def dense_relu_block_1(x,filt):
    
    y = tf.keras.layers.Dense(filt, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='linear', use_bias=True, bias_initializer=initializers.Zeros())(x)
    # y = tf.keras.layers.BatchNormalization()(y)
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
    model.add(tf.keras.Input(shape=(12)))
    
    model.add(tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='relu', use_bias=True, bias_initializer=initializers.Zeros()))
    
    model.add(tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='relu', use_bias=True, bias_initializer=initializers.Zeros()))

    model.add(tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='relu', use_bias=True, bias_initializer=initializers.Zeros()))
    
    model.add(tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                              activation='sigmoid', use_bias=True, bias_initializer=initializers.Zeros()))
    
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

    input_layer_2 = tf.keras.Input(shape=(6, 6, 1))
    conv_12       = conv_relu_block(input_layer_2, 64, 'conv12')
    conv_22       = conv_relu_block(conv_12, 128, 'conv22')
    conv_32       = conv_relu_block(conv_22, 128, 'conv32')
    flat          = tf.keras.layers.GlobalAveragePooling2D()(conv_32)
    
    concat  = tf.keras.layers.Concatenate(axis=1)([dense_3, flat])
    dense_out_1 = dense_block(concat, 128,     'dense_out1')
    dense_out_2 = dense_block(dense_out_1, 64, 'dense_out2')
    # dense_out_3 = dense_block(dense_out_2, 64, 'dense_out3')

    output_layer = dense_sigmoid_block(dense_out_2, 12, 'output')
     
    ge_model = tf.keras.models.Model([input_layer_1,input_layer_2], output_layer)
    
    return ge_model

ge_model = make_generator_model()
ge_model.summary()

#%% Test generator model

# generator = make_generator_model()
generated_image = ge_model([input_train_v, input_train_p], training=False)

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
    input_2 = tf.keras.Input(shape=(6,6,1))
    
    # inter_1 = tf.keras.Input(shape=(6,6,1))
    
    inter_output = g_model([input_1, input_2])
    
    output = d_model(inter_output)
    
    gan_model = tf.keras.models.Model([input_1,input_2], output)
    
    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return gan_model

gan_model = define_gan(ge_model, de_model)
gan_decision = gan_model([input_train_v,input_train_p])

#%% Select real samples

def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # print('ix is: ', ix)
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
    x_input2 = dataset2[ix,:,:]
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

def summarize_performance(epoch, g_model, d_model, dataset_P, dataset_M, dataset_V, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset_M, n_samples)
    # evaluate discriminator on real examples
    _, acc_real    = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, dataset_V, dataset_P, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake    = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    # save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

#%% Train the generator and discriminator

def train(g_model, d_model, gan_model, dataset_P, dataset_M, dataset_V, latent_dim, n_epochs=100, n_batch=128):
    
    bat_per_epo = int(dataset_P.shape[0] / n_batch)
    half_batch  = int(n_batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            
            X_real, y_real = generate_real_samples(dataset_M, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, dataset_V, dataset_P, half_batch)
            
            # print(X_real.shape)
            # print(X_fake.shape)
            
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            
            X_gan_1, y_gan_1 = generate_real_samples(dataset_V, n_batch)
            X_gan_2, y_gan_2 = generate_real_samples(dataset_P, n_batch)
            
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
datasetm = M_train         # 1-by-12
datasetv = V_train         # 1-by-3

datasetm1 = M_train

#%% create the discriminator
d_model = make_discriminator_model()
# create the generator
g_model = make_generator_model()
# create the gan
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, datasetp, datasetm, datasetv, latent_dim)

#%% Training sample validation

# P1,  P2  = generate_real_samples(P_train, 32)
# VM1, VM2 = generate_fake_samples(g_model, V_train, P_train, 32)
# V1,  V2  = generate_real_samples(V_train, 32)
# M1,  M2  = generate_real_samples(M_train, 32)

#%%
# d_model.train_on_batch(np.expand_dims(P1, axis=3), np.ones((32,1)))

#%%
# gan_model.train_on_batch([V1, M1], np.ones((32,1)))

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

predict = g_model.predict([input_test_v,input_test_p])
predict = np.round(predict)

[p1,p2] = predict.shape
gen_image_mtx = np.zeros([p1,6,6])
true_mat_mtx  = np.zeros([p1,6,6])

for iii in range(p1):
    gen_image_mtx[iii,:,:] = mat_form(predict[iii,:], 6, 6)
    true_mat_mtx[iii,:,:]  = mat_form(input_test_m[iii,:], 6, 6)
    
accuracy = np.zeros((generated_image.shape[0],1))

#%%

for i in range (predict.shape[0]):
    accuracy[i] = 1-np.sum(np.abs(gen_image_mtx[i,:,:]-true_mat_mtx[i,:,:]))/36
    
Accuracy = np.mean(accuracy)
print('The prediction accuracy is', Accuracy)

# Prediction image sample
plt.title('ML prediction contour')
plt.imshow(gen_image_mtx[0,:,:])
plt.colorbar()
plt.grid(True)
plt.show()

# Real image sample
plt.title('Real contour')
plt.imshow(true_mat_mtx[0,:,:])
plt.colorbar()
plt.grid(True)
plt.show()

#%% Validate prediction

new_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

# Prediction: p + M -> V

physics_prediction = new_model.predict([P_test,np.round(predict)])

error_rate_0 = np.mean(np.abs(physics_prediction[:,0] - input_test_v[:,0]) / input_test_v[:,0])
error_rate_1 = np.mean(np.abs(physics_prediction[:,1] - input_test_v[:,1]) / input_test_v[:,1])
error_rate_2 = np.mean(np.abs(physics_prediction[:,2] - input_test_v[:,2]) / input_test_v[:,2])

print('The absolute prediction error for E1 is: ', error_rate_0)
print('The absolute prediction error for E2 is: ', error_rate_1)
print('The absolute prediction error for G12 is: ', error_rate_2)



