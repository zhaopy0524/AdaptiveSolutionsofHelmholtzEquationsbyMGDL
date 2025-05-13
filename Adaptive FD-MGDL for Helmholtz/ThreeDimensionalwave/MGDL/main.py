# -*- coding: utf-8 -*-


from multigrade_dnn_main import multi_grade_dnn
from data_generate import generate_data

#set parameter
SGD = False
mini_batch = False


#-------------------------------------------------------------------------------------k=20----------------------------------------------------------------------
k = 20
data = generate_data(k)
#set structure for MGDL
mul_layers_dims =  [[3, 256, 256, 2], [256, 256, 2], [256, 256, 2], [256, 256, 2]]                  # this is the structure for MGDL
#set activation for MGDL for each grade
activation = ['sin', 'sin', 'relu', 'relu']
#set train epoch for each grade
mul_epochs = [500, 1000, 1000, 1000]
stop_criterion = [1e-09, 1e-09, 1e-09, 1e-09]
max_learning_rate = [0.1, 0.1, 0.001, 0.001]
min_learning_rate = [0.01, 0.1, 0.001, 0.001]
multi_grade_dnn(data, stop_criterion, max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation, k)
