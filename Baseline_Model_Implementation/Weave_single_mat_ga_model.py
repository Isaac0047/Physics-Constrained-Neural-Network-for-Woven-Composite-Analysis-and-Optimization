import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from time import time
import re

#%% Data Preprocessing

size_x = 6
size_y = 6
n1     = 1
n2     = 1

data_set  = 9000
data_new  = np.zeros([data_set, 6, 6])
stiff_new = np.zeros([data_set, 1])

vf_name = 'volume_fraction.txt'
vf_path = 'C:\\Users\\Haoti\\Documents\\MATLAB\weave_ML\\' + vf_name
vf      = np.loadtxt(vf_path)

stiff_name = 'stiffness.txt'
stiff_path = 'C:\\Users\\Haoti\\Documents\\MATLAB\weave_ML\\' + stiff_name
stiff      = np.loadtxt(stiff_path)

[s_m, s_n] = stiff.shape
physics_vector = np.zeros([data_set, 3])
physics_vector_norm = np.zeros([data_set, 3])
pattern_matrix = np.zeros([data_set, size_x*n1, size_y*n2])

start_1 = time()

def mtx_expand(x,n1,n2,size_x,size_y):
    
    x_new = tf.convert_to_tensor(x)
    
    # r1 = n1 * np.ones(size_x)
    # r2 = n2 * np.ones(size_y)
    
    y  = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    y  = tf.keras.backend.repeat_elements(y, rep=n2, axis=1)
    
    return y

def vec_expand(x,n1,size_x):
    
    x_new = tf.convert_to_tensor(x)
    y = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=0)
    
    return y

for k in range(data_set):
    
    k_s = str(k)
    mtx_name = 'Matrix_' + k_s + '.csv'
    mtx_path = 'C:\\Users\\Haoti\\.spyder-py3\\Weave_ML\\Data_set\\Random_Matrix\\' + mtx_name
    
    file_name = 'Abaqus_' + k_s + '.txt'
    file_path = 'C:\\Users\\Haoti\\.spyder-py3\\Weave_ML\\Data_set\\Effective_Inplane_properties\\' + file_name
    
    matrix = np.loadtxt(mtx_path, delimiter=',')
    matrix_expand = mtx_expand(matrix, n1, n2, size_x, size_y)
    # data   = np.loadtxt(file_path, delimiter=',')
    
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
    pattern_matrix[k,:,:] = matrix_expand


#%% Initialize the input

# target = np.array([4200,2600,312])
target = physics_vector_norm[0,:]

start =  time()

#%% Load trained CNN

new_model = tf.keras.models.load_model('Weave_CNN_model_E7.h5')

# predict = new_model.predict(input)

#%% Expand matrices

def mtx_expand(x,n1,n2):
    
    x_new = tf.convert_to_tensor(x)
    
    y  = tf.keras.backend.repeat_elements(x_new, rep=n1, axis=1)
    y  = tf.keras.backend.repeat_elements(y, rep=n2, axis=2)
    
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