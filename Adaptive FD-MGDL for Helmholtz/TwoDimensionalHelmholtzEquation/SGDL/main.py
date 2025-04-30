# -*- coding: utf-8 -*-

from singlegrade_dnn_main import single_dnn_main 


#set parameter
k = 50


SGD = False
mini_batch_size = False


#set structure for SGDL
layers_dims = [2,256,256,1]                                                  # this is the structure for SGDL
#set train epoch
epochs = 500


#set max learning rate and min learning rate 
max_learning_rate = 1e-1                                            # the maximum learning rate, denote as t_max in the paper
min_learning_rate = 1e-1                                            # the minimum learning rate, denote as t_min in the paper


single_dnn_main( layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, k )
