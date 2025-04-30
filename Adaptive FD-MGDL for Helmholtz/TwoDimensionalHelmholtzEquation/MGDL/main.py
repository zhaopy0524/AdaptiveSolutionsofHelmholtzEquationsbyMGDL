# -*- coding: utf-8 -*-


from multigrade_dnn_main import multi_grade_dnn
from data_generate import generate_data

#set parameter

SGD = False
mini_batch = False

k = 50

#set structure for MGDL
mul_layers_dims =  [[2, 256, 256, 1],[256,256,1],[256,256,1],[256,256,1],[256,256,1],[256,256,1]]                  # this is the structure for MGDL
#set activation for MGDL for each grade
activation = ['sin','sin','relu','relu','relu','relu']
#set train epoch for each grade
mul_epochs = [400,3000,3000,2500,2500,1000]

stop_criterion = [1e-06,1e-07,1e-07,1e-08,1e-08,1e-08]

data = generate_data(k)  

max_learning_rate = [0.1,0.01,0.001,0.001,0.001,0.0001]
min_learning_rate = [0.01,0.001,0.0001,0.001,0.001,0.001]
multi_grade_dnn(data, stop_criterion, max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation, k)


