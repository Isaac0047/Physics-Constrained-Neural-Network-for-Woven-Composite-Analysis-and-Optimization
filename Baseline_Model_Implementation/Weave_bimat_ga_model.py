import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os

import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from time import time
import re
import numpy as np
import math

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
pat_mtx = np.load('weave_pattern.npy')
phy_vec = np.load('physics_vector.npy')
phy_norm_vec = np.load('physics_norm_vector.npy')

#%% Define CNN models

def mtx_expand(x,n1,n2,size_x,size_y):
    
    x_new = tf.convert_to_tensor(x)
    
    y  = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    y  = tf.keras.backend.repeat_elements(y, rep=n2, axis=1)
    
    return y

def vec_expand(x,n1,size_x):
    
    x_new = tf.convert_to_tensor(x)
    y = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    
    return y

#%% Initialize the input

# target = np.array([4200,2600,312])
target = phy_norm_vec[0,:]

start =  time()

#%% Load trained CNNe

new_model = tf.keras.models.load_model('Weave_bimat_CNN_mat_vec_update.h5')

# predict = new_model.predict(input)

#%% Expand matrices

def mtx_expand(x,n1,n2):
    
    x_new = tf.convert_to_tensor(x)
    
    y  = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=1)
    y  = tf.keras.backend.repeat_elements(y,     rep=n2, axis=2)
    
    return y

#%% Define the loss function

def f(X):
    X_new = X.reshape(1, 6, 6, 1)
    
    return np.linalg.norm(target - new_model.predict(X_new))

#%% Start the algorithm

varbound = np.random.randint(2, size=(6,6))
varbound = varbound.flatten()

algorithm_param = {'max_num_iteration': 500,\
                    'population_size':100,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

model = ga(function=f, dimension=36, variable_type='bool', 
           variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()

time_2 = time() - start

#%% Test the solution

solution = model.output_dict

sol = solution['variable'].reshape(1,6,6,1)
prediction = new_model.predict(sol)

plt_sol = np.squeeze(sol)

# Summarize history for loss
fig = plt.figure()
plt.title('Predicted Contour')
plt.imshow(plt_sol)
plt.colorbar()
plt.grid(True) 