# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pickle
import numpy as np

from data_generate import generate_data


background_color = '#EAEAF2'
mpl.rcParams['axes.facecolor'] = background_color  
mpl.rcParams['axes.edgecolor'] = background_color  
mpl.rcParams['axes.grid'] = True  
mpl.rcParams['grid.color'] = 'white'  
mpl.rcParams['grid.linestyle'] = '-'  
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['legend.facecolor'] = background_color  
mpl.rcParams['xtick.color'] = 'black'  
mpl.rcParams['ytick.color'] = 'black' 
mpl.rcParams['axes.spines.top'] = False 
mpl.rcParams['axes.spines.right'] = False 
mpl.rcParams['axes.spines.left'] = False 
mpl.rcParams['axes.spines.bottom'] = False  
mpl.rcParams['xtick.bottom'] = False  
mpl.rcParams['ytick.left'] = False  
mpl.rcParams['figure.autolayout'] = True  

def format_with_comma(x, pos):
    return f"{int(x):,}"

def results_analysis(fullfilename):
    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter, prev_info = pickle.load(f)

    print(
        "################################################################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)

    data = generate_data()

    num_iter = trained_variable["REC_FRQ_iter"]

    TRAIN_loss = []
    MUL_EPOCH = []

    total_time = 0

    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        total_time = total_time + trained_variable['train_time'][i]
        if i == 0:
            current_epoch = trained_variable["REC_FRQ_iter"][i][-1]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i])
        else:
            current_epoch += trained_variable["REC_FRQ_iter"][i][-1]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i][1:])

    print('the train times for each grade is {}'.format(trained_variable['train_time']))
    print('the total train times is {}'.format(total_time))

    plt.figure(figsize=(8, 6))
    plt.plot(np.array(TRAIN_loss), label='Train loss')
    plt.legend(loc='upper right')
    plt.xlabel("Number of training epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.xlim([0, MUL_EPOCH[0]])

    plt.title('MGDL: Loss')
    for i in range(0, len(nn_parameter["mul_layers_dims"]) - 1):
        plt.axvline(x=MUL_EPOCH[i], color='k', linestyle=':')
    # plt.xticks([0] + [MUL_EPOCH[i] for i in range(len(MUL_EPOCH))], rotation=45)
    plt.xticks([0] + [MUL_EPOCH[i] for i in range(len(MUL_EPOCH))])

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))

    plt.tight_layout()
    plt.grid(True)

    loss_curve_filename = "MGDL-LOSS.png"
    plt.savefig(loss_curve_filename, format="png")
    print(f"Loss curve saved as {loss_curve_filename}")

    plt.show()

    mpl.rcParams['axes.grid'] = False
    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        num_iter = trained_variable["REC_FRQ_iter"][i]

        X1 = data['train_X1'] * 1000
        X2 = data['train_X2'] * 1000
        Z = trained_variable["train_predict"][i][0].reshape(data['num'], data['num'])
        Z = Z.real

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='jet')
        plt.colorbar()
        plt.title('MGDL: grade {}'.format(i + 1))

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))

        plt.gca().invert_yaxis()

        scatter_filename = "grade{}.eps".format(i + 1)
        plt.savefig(scatter_filename, format="eps")

        scatter_filename = "grade{}.png".format(i + 1)
        plt.savefig(scatter_filename, format="png")

        plt.show()

        plt.imshow(Z, cmap='jet', aspect='auto', extent=[0, 2000, 2000, 0])
        plt.xticks(np.arange(0, 2001, 200))
        plt.yticks(np.arange(0, 2001, 200))
        plt.colorbar()
        plt.title('MGDL: grade {}'.format(i + 1))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))
        scatter_filename = "grade'{}.eps".format(i + 1)
        plt.savefig(scatter_filename, format="eps")
        scatter_filename = "grade'{}.png".format(i + 1)
        plt.savefig(scatter_filename, format="png")

        plt.show()
    return