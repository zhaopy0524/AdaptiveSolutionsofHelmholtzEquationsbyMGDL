# -*- coding: utf-8 -*-


import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import singlegrade_dnn_solving as dnn
from data_generate import generate_data

mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.grid'] = False
mpl.rcParams['grid.color'] = 'white'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['legend.facecolor'] = 'white'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['figure.autolayout'] = True

def format_with_comma(x, pos):
    return f"{int(x):,}"

def result_analysis(fullfilename):
    match = re.search(r"k=(\d+)", fullfilename)
    k = int(match.group(1))

    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)

    data = generate_data(k)

    # train_predict, _ = singlegrade_model_forward(data["train_X"], nn_parameter['layers_dims'], history['parameters'], nn_parameter["activation"] , nn_parameter["sinORrelu"])
    test_predict, _ = dnn.singlegrade_model_forward(data["test_X"], nn_parameter['layers_dims'], history['parameters'],
                                                    nn_parameter["activation"], nn_parameter["sinORrelu"])

    print("###########################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)

    print('train_rse is {}, test_rse is {}'.format(history['train_rses'][-1], dnn.rse(data["test_Y"], test_predict)))

    print('the train time is {}'.format(history["time"]))

    plt.plot(history["REC_FRQ_iter"], np.array(history["train_costs"]), label="train loss")
    plt.xlabel('Number of training epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('SGDL: loss(k={})'.format(k))
    plt.show()

    return


