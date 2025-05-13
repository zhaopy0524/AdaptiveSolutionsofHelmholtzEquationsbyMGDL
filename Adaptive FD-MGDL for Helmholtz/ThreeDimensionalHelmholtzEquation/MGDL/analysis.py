# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pickle
import numpy as np
import re
import multigrade_dnn_model as m_dnn

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
        trained_variable, nn_parameter, opt_parameter, prev_info = pickle.load(f)
        # trained_variable, nn_parameter, opt_parameter = pickle.load(f)

    print(
        "################################################################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)

    data = generate_data(k)

    test_rse, predict_test_Y = m_dnn.multigrade_dnn_model_predict(data, nn_parameter, opt_parameter, trained_variable)

    num_iter = trained_variable["REC_FRQ_iter"]

    train_rse = []
    # validation_rse = []
    TRAIN_loss = []
    # VALIDATION_loss = []
    MUL_EPOCH = []

    total_time = 0

    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        total_time = total_time + trained_variable['train_time'][i]
        train_rse.append(trained_variable['train_rse'][i][-1])
        #     validation_rse.append(trained_variable['validation_rse'][i][-1])
        if i == 0:
            current_epoch = trained_variable["REC_FRQ_iter"][i][-1]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i])
        #         VALIDATION_loss.extend(trained_variable['validation_costs'][i])
        else:
            current_epoch += trained_variable["REC_FRQ_iter"][i][-1]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i][1:])
    #         VALIDATION_loss.extend(trained_variable['validation_costs'][i][1:])

    print('the train rse for each grade is {}'.format(train_rse))
    # print('the validation rse for each grade is {}'.format(validation_rse))
    print('the test rse for each grade is  {}'.format(test_rse))
    print('the train times for each grade is {}'.format(trained_variable['train_time']))
    print('the total train times is {}'.format(total_time))

    # plt.figure(figsize=(10, 6)) # figure's size setting
    plt.plot(np.array(TRAIN_loss), label='Train loss')
    plt.legend(loc='upper right')
    plt.xlabel("Number of training epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.xlim([0, MUL_EPOCH[0]])

    plt.title('MGDL: Loss ($k$={})'.format(k))
    for i in range(0, len(nn_parameter["mul_layers_dims"]) - 1):
        plt.axvline(x=MUL_EPOCH[i], color='k', linestyle=':')
    plt.xticks([0] + [MUL_EPOCH[i] for i in range(len(MUL_EPOCH))], rotation=45)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))

    plt.tight_layout()
    plt.grid(True)

    loss_curve_filename = "{}MGDL-LOSS.png".format(k)
    plt.savefig(loss_curve_filename, format="png")

    print(f"Loss curve saved as {loss_curve_filename}")

    plt.show()

    return
