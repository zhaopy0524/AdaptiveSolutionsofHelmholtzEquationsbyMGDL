# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp


def generate_data(k):
    np.random.seed(0)
    theta = np.pi / 4
    k1 = k * np.cos(theta)
    k2 = k * np.sin(theta)

    k_settings = {
        50: (301, 151),
        100: (501, 251),
        150: (701, 351),
        200: (701, 351)
    }

    if k in k_settings:
        ntrain, ntest = k_settings[k]
        if k in {100, 200}:
            k += 1

    h = 1 / (ntrain - 1)

    def u(x1, x2):
        return np.exp(1j * (k1 * x1 + k2 * x2))
    
    x1 = np.linspace(0, 1, ntrain)
    x2 = np.linspace(0, 1, ntrain)
    X1, X2 = np.meshgrid(x1, x2)
    train_X1 = np.array([X1.flatten()])
    train_X2 = np.array([X2.flatten()])
    train_X = np.vstack((train_X1, train_X2))
    true_Y = u(train_X1,train_X2)
    Y_real = true_Y.real
    Y_imag = true_Y.imag
    true_Y = np.vstack((Y_real, Y_imag))

    z1 = np.linspace(0, 1, ntest)
    z2 = np.linspace(0, 1, ntest)
    Z1, Z2 = np.meshgrid(z1, z2)
    Y = u(Z1, Z2)
    test_X1 = np.array([Z1.flatten()])
    test_X2 = np.array([Z2.flatten()])
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.array([Y.flatten()])
    Y_real = test_Y.real
    Y_imag = test_Y.imag
    test_Y = np.vstack((Y_real, Y_imag))

    u = np.zeros((ntrain, ntrain), dtype=complex)
    for i in range(ntrain):
        x = i * h
        u[i, 0] = np.exp(1j * k1 * x)
        u[i, ntrain - 1] = np.exp(1j * (k1 * x + k2))

    for j in range(ntrain):
        y = j * h
        u[ntrain - 1, j] = np.exp(1j * (k1 + k2 * y))
        u[0, j] = np.exp(1j * k2 * y)

    size = (ntrain - 2) ** 2
    A = sp.lil_matrix((size, size), dtype=complex)
    b = np.zeros(size, dtype=complex)

    for i in range(1, ntrain - 1):
        for j in range(1, ntrain - 1):
            idx = (i - 1) * (ntrain - 2) + (j - 1)
            coeff_center = (-4 + (h ** 2) * (k ** 2))
            A[idx, idx] = coeff_center

            if i - 1 == 0:
                b[idx] -= u[i - 1, j]
            else:
                neighbor_idx = (i - 2) * (ntrain - 2) + (j - 1)
                A[idx, neighbor_idx] += 1

            if i + 1 == ntrain - 1:
                b[idx] -= u[i + 1, j]
            else:
                neighbor_idx = i * (ntrain - 2) + (j - 1)
                A[idx, neighbor_idx] += 1

            if j + 1 == ntrain - 1:
                b[idx] -= u[i, j + 1]
            else:
                neighbor_idx = (i - 1) * (ntrain - 2) + j
                A[idx, neighbor_idx] += 1

            if j - 1 == 0:
                b[idx] -= u[i, j - 1]
            else:
                neighbor_idx = (i - 1) * (ntrain - 2) + (j - 2)
                A[idx, neighbor_idx] += 1

    A = A.tocsr()
    

    data = {}
    data['AA'] = A
    data['BB'] = b
    data['u'] = u
    data['train_X'] = train_X
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    data['true_Y'] = true_Y
    data['train_X1'] = X1
    data['train_X2'] = X2
    data['test_X1'] = Z1
    data['test_X2'] = Z2
    data['ntrain'] = ntrain
    data['ntest'] = ntest

    return data
