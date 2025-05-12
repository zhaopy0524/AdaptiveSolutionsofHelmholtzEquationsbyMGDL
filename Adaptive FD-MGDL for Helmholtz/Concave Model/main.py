# -*- coding: utf-8 -*-


from multigrade_dnn_main import multi_grade_dnn
from data_generate import generate_data


SGD = False
mini_batch = False


#set structure for MGDL
mul_layers_dims =  [[2, 256, 256, 1]]                  # this is the structure for MGDL
#set activation for MGDL for each grade
activation = ['sin']
#set train epoch for each grade
mul_epochs = [1000]

stop_criterion = [1e-09]

data = generate_data()  

#set max learning rate and min learning rate for each grade
MAXLr = [[1e-1],[1e-2],[1e-3],[1e-4],[1e-5]]
MINLr = [[1e-1],[1e-2],[1e-3],[1e-4],[1e-5]]
for i in range(0,5):
    for j in range(i,5):
        max_learning_rate = MAXLr[i]
        min_learning_rate = MINLr[j]
        multi_grade_dnn(data, stop_criterion, max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation)








