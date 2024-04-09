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

#%% Load discriminator model

pred_model = tf.keras.models.load_model('gan_mat_vec_to_pat_Phy.h5')

modulus_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

#%% Consider for a sequence example

mat_vec = [1,1,1,1,0,0,0,1,1,0,1,0]
mod     = [30,30,25]

mat_vec = tf.reshape(mat_vec, [-1, 12])
mod     = tf.reshape(mod, [-1,3])

pred = pred_model([mod, mat_vec])

#%% Check modulus

pred_mod = modulus_model([pred[0], mat_vec])

#%%
pred_pat_0 = tf.squeeze(pred[0], [0,-1])
plt.figure()
plt.imshow(np.round(pred_pat_0))
plt.colorbar()

#%% Check if the model has continuous fiber or yarn

def func_check(m):
    
    aa = True
    
    for i in range(6):
            
        if np.sum(m[i,:])==0 or np.sum(m[i,:])==6:
            aa = False
            break
        
    for j in range(6):
        
        if np.sum(m[:,j])==0 or np.sum(m[:,j])==6:
            aa = False
            break
            
    return aa

#%% Check if the function works

kk = func_check(np.round(pred_pat))
print(kk)

#%% Now Loop Over to find out the optimal

import random

steps = 300

mat_vec = [1,1,1,1,0,0,0,1,1,0,1,0]
mod     = [30,30,25]

mat_vec_tf = tf.reshape(mat_vec, [-1, 12])
#mod_tf     = tf.reshape(mod, [-1,3])

#bound = [0.1,0.2,0.3,0.5,0.8,1.3,2.1,3.6,5.7] # Design based on Fibonacci number
bound = np.linspace(0.1,5.1,26)
kk = False

#%%

kk = False
mod_new = [0,0,0]

for bo in bound: 
    
    print(bo)

    kk = func_check(np.round(pred_pat))
    
    if kk is True:
        print('We found it before looping i!')
        print('The new modulus is', mod_new)
        
        #pred_pat = tf.squeeze(pred_pat, [0,-1])
        
        plt.figure()
        plt.imshow(np.round(pred_pat))
        plt.colorbar()
    
        break
    
    for i in range(steps):
        
        mod0 = pred_mod[0][0] + random.uniform(-1,1)*bo
        mod1 = pred_mod[0][1] + random.uniform(-1,1)*bo
        mod2 = pred_mod[0][2] + random.uniform(-1,1)*bo
        
        mod_new = [mod0,mod1,mod2]
        
        mod_tf  = tf.reshape(mod_new, [-1,3])
        predict = pred_model([mod_tf, mat_vec_tf])
        #print(predict)
        
        pred_pat = tf.squeeze(predict[0], [0,-1])
        pred_pat = np.round(pred_pat)
        #print(pred_pat.shape)
        
        kk = func_check(np.round(pred_pat))
        
        if kk is True:
            print('Hey we found it!')
            
            plt.figure()
            plt.imshow(pred_pat)
            plt.colorbar()
        
            break
    
#%%

# plain_pat = [[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,1,0,1,0],[0,1,0,1,0,1]]
#plain_pat = [[0,1,0,1,1,1],[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,1,0,0,1],[0,1,1,1,1,1],[1,0,1,0,0,1]]
plain_pat = [[0,1,1,1,1,1],[1,0,1,0,1,1],[1,1,0,1,0,1],[1,0,1,1,1,1],[1,1,1,1,1,1],[0,0,1,0,0,1]]
plain_pat = tf.reshape(plain_pat, [-1,6,6,1])

pred_plain_mod = modulus_model([plain_pat, mat_vec])
print(pred_plain_mod)

    










