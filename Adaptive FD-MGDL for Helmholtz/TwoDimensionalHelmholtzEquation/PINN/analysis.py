# -*- coding: utf-8 -*-


import re
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

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
    k = 50

    with open(fullfilename, 'rb') as f:
        results = pickle.load(f)

    total_losses = results['total_losses']
    interior_losses = results['interior_losses']
    boundary_losses = results['boundary_losses']

    plt.plot(total_losses, label='Total loss')
    plt.plot(interior_losses, label='Interior loss')
    plt.plot(boundary_losses, label='Boundary loss')
    plt.xlabel('Number of training epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('PINN: Loss ($k$={})'.format(k))
    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))

    plt.grid(True)

    loss_curve_filename = "PINN6-LOSS.pdf"
    plt.savefig(loss_curve_filename, format="pdf")

    plt.show()

    X1 = results['X_test']
    X2 = results['Y_test']
    Z = results['u_pred_test']
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='$u(x_1, x_2)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('PINN-6 ($k={}$)'.format(k))
    scatter_filename = "PINN6.pdf"
    plt.savefig(scatter_filename, format="pdf")
    plt.show()

    print(f"Total training time: {results['training_time']:.2f} seconds")
    print(f"Relative L2 error on train data: {results['error_rel_train']:.4e}")
    print(f"Relative L2 error on test data: {results['error_rel_test']:.4e}")

    return

