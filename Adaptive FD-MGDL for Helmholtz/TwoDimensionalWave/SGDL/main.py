# -*- coding: utf-8 -*-

from singlegrade_dnn_main import single_dnn_main 


#set parameter
SGD = False
mini_batch_size = False


#----------------------------------------------------------------------k=50----------------------------------------------------------
k = 50
#set structure for SGDL
layers_dims = [2, 256, 256, 256, 256, 2]                                                  # this is the structure for SGDL
#set train epoch
epochs = 8000
#set max learning rate and min learning rate 
max_learning_rate = 1e-1                                            # the maximum learning rate, denote as t_max in the paper
min_learning_rate = 1e-4                                            # the minimum learning rate, denote as t_min in the paper
single_dnn_main( layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, k )
