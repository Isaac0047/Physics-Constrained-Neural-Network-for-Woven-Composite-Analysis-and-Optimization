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

#%% Predefining parameters

data_set = 9000
size_x = 6
size_y = 6
n1     = 1
n2     = 1
physics_vector = np.zeros([data_set, 3])
physics_vector_norm = np.zeros([data_set, 3])
pattern_matrix = np.zeros([data_set, size_x*n1, size_y*n2])

#%% Matrix expand

def mtx_expand(x,n1,n2,size_x,size_y):
    x_new = tf.convert_to_tensor(x)  
    y  = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    y  = tf.keras.backend.repeat_elements(y, rep=n2, axis=1)
    return y

def vec_expand(x,n1,size_x):
    x_new = tf.convert_to_tensor(x)
    y = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    return y

#%% Load data from files

for k in range(data_set):
    
    k_s = str(k)
    mtx_name = 'Matrix_' + k_s + '.csv'
    mtx_path = 'C:\\Users\\Haoti\\.spyder-py3\\Weave_ML\\Data_set\\Random_Matrix\\' + mtx_name
    
    file_name = 'Abaqus_' + k_s + '.txt'
    file_path = 'C:\\Users\\Haoti\\.spyder-py3\\Weave_ML\\Data_set\\Effective_Inplane_properties\\' + file_name
    
    matrix = np.loadtxt(mtx_path, delimiter=',')
    
    lines = []
    i = 0
    
    with open(file_path, 'r') as fp:
        for i, line in enumerate(fp):
            if i>=11 and i<=17:
                lines.append(line)
                
    lines = str(lines)          
    lines_sep = re.split(r'\s', lines)
    
    E1s   = lines_sep[1][:-4]
    E1    = float(E1s)
    E1n   = E1 / 1e7
    
    v12s  = lines_sep[3][:-4]
    v12   = float(v12s)
    
    E2s   = lines_sep[5][:-4]
    E2    = float(E2s)
    E2n   = E2 / 1e7
    
    v21s  = lines_sep[7][:-4]
    v21   = float(v21s)
    
    G12s  = lines_sep[9][:-4]
    G12   = float(G12s)
    G12n  = G12 / 1e7 
    
    CTE_1s = lines_sep[11][:-4]
    CTE_1  = float(CTE_1s)
    
    CTE_2s = lines_sep[13][:-4]
    CTE_2  = float(CTE_2s)
    
    physics_vector[k,:] = [E1, E2, G12]
    physics_vector_norm[k,:] = [E1n, E2n, G12n]
    pattern_matrix[k,:,:] = matrix
    
#%% Print out parameters

# print(physics_vector.shape, physics_vector_norm.shape, pattern_matrix.shape)

#%% Data preprocessing     
X_train, X_test, Y_train, Y_test = train_test_split(physics_vector_norm, pattern_matrix, test_size=0.2, random_state=47)
X_test,  X_cv,   Y_test,  Y_cv   = train_test_split(X_test,              Y_test,         test_size=0.5, random_state=47)

input_train  = tf.reshape(X_train, [-1, 3])
output_train = tf.reshape(Y_train, [-1, 6, 6])

input_cv  = tf.reshape(X_train, [-1, 3])
output_cv = tf.reshape(Y_train, [-1, 6, 6])

input_test  = tf.reshape(X_test, [-1, 3])
output_test = tf.reshape(Y_test, [-1, 6, 6])

#%% Defining models

def make_discriminator_model():
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], strides=(2,2), padding='same', input_shape=[6, 6, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

model = make_discriminator_model()
model.summary()

#%%
def make_generator_model():
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.Input(shape=(3)))
    model.add(tf.keras.layers.Dense(1*1*32, activation='linear'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Dense(1*1*64, activation='linear'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Dense(1*1*64, activation='linear'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Reshape((1, 1, 64)))
    assert model.output_shape == (None, 1, 1, 64) # None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[3,3], strides=(1,1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[3,3], strides=(2,2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=1,  kernel_size=[3,3], strides=(1,1), padding='same', use_bias=True, activation='sigmoid'))
    
    return model

ge_model = make_generator_model()
ge_model.summary()

#%% Load discriminator model

de_model = tf.keras.models.load_model('Weave_CNN_model_E7_new.h5')

#%% Test generator model

generator = make_generator_model()
generated_image = generator(X_train, training=False)

#%% Test discriminator model

discriminator = make_discriminator_model()
decision      = discriminator(generated_image)

#%% Define loss function and optimizers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#%% Define Discriminator Loss

def discriminator_loss(real_output, fake_output):
    real_loss  = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss  = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#%% Define Generator Loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#%% Generate models

def define_gan(g_model, d_model):
    # make weights in discriminator not trainable
    d_model.trainable = False
    # connect them
    input_1 = tf.keras.Input(shape=(3))
    # add generator
    inter_output = g_model(input_1)
    # inter_output_round = tf.keras.backend.round(inter_output)
    # add discriminator
    output = d_model(inter_output)
    # compile model
    gan_model = tf.keras.models.Model(inputs=input_1,outputs=[inter_output, output])
    
    return gan_model

gan_model = define_gan(ge_model, de_model)
gan_decision = gan_model(input_train)

#%% Neural Network Parameters

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,   beta_2=0.999, decay=1e-6)
sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)

#%% Train the AE network 

gan_model.compile(optimizer=opt, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_squared_error], metrics=['accuracy'], loss_weights=[1, 0.1])

epoch = 150
ae_history  = gan_model.fit(input_train, [output_train, input_train], batch_size=64, epochs=epoch, 
                    steps_per_epoch=40, validation_data=(input_cv, [output_cv, input_cv]))

ae_predict  = gan_model.predict(input_test)

ae_score    = gan_model.evaluate(input_test, [output_test, input_test], verbose=1)
print('\n', 'Test accuracy', ae_score)  

#%% Print out accuracy

generated_image = ge_model.predict(input_test)
generated_image = np.round(generated_image)
accuracy = np.zeros((generated_image.shape[0],1))
predict_stiff   = de_model(generated_image)


for i in range (generated_image.shape[0]):
    accuracy[i] = 1-np.sum(np.abs(generated_image[i,:,:,0]-Y_test[i,:,:]))/36
    
Accuracy = np.mean(accuracy)
print('The prediction accuracy is', Accuracy)

# Prediction image sample
plt.title('SDF contour')
plt.imshow(generated_image[0,:,:,0])
plt.colorbar()
plt.grid(True)
plt.show()

# Real image sample
plt.title('SDF contour')
plt.imshow(Y_test[0,:,:])
plt.colorbar()
plt.grid(True)
plt.show()

#%% Evaluate the prediction error w.r.t mechanical properties

[p1,p2,p3,p4] = ae_predict[0].shape
predict_stiff = tf.cast(predict_stiff, tf.float64)

error_0 = np.zeros(p1)
error_1 = np.zeros(p1)
error_2 = np.zeros(p1)

for ip in range(p1):
    
    error_0[ip] = (np.abs(predict_stiff[ip][0] - input_test[ip][0])) / input_test[ip][0]
    error_1[ip] = (np.abs(predict_stiff[ip][1] - input_test[ip][1])) / input_test[ip][1]
    error_2[ip] = (np.abs(predict_stiff[ip][2] - input_test[ip][2])) / input_test[ip][2]
    
error_0_ave = np.mean(error_0)
error_1_ave = np.mean(error_1)
error_2_ave = np.mean(error_2)

print("Error rate of E1 is: ",  error_0_ave)
print("Error rate of E2 is: ",  error_1_ave)
print("Error rate of G is: ",   error_2_ave)

#%% Check the prediction error of decoder Network

de_predict = de_model(output_test)
de_predict = tf.cast(de_predict, tf.float64)

error_00 = np.zeros(p1)
error_11 = np.zeros(p1)
error_22 = np.zeros(p1)

for ip in range(p1):
    
    error_00[ip] = (np.abs(de_predict[ip][0] - input_test[ip][0])) / input_test[ip][0]
    error_11[ip] = (np.abs(de_predict[ip][1] - input_test[ip][1])) / input_test[ip][1]
    error_22[ip] = (np.abs(de_predict[ip][2] - input_test[ip][2])) / input_test[ip][2]

#%%
error_00_ave = np.mean(error_00)
error_11_ave = np.mean(error_11)
error_22_ave = np.mean(error_22)

print("Error rate of E1 is: ",  error_00_ave)
print("Error rate of E2 is: ",  error_11_ave)
print("Error rate of G is: ",   error_22_ave)

#%% 

plt.figure()
plt.plot(de_predict[:,0])
plt.plot(input_test[:,0])
plt.title('E1')

plt.figure()
plt.plot(de_predict[:,1])
plt.plot(input_test[:,1])
plt.title('E2')

plt.figure()
plt.plot(de_predict[:,2])
plt.plot(input_test[:,2])
plt.title('G12')

#%% Save the model

# gan_model.save('weave_single_mat_reverse.h5')