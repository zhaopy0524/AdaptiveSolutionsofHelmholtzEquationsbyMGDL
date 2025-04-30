# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import diags


def generate_data(k):
    np.random.seed(0)

    def u(x1, x2, x3):
        return np.sin(np.sqrt(3) / 3 * k * x1) * np.sin(np.sqrt(3) / 3 * k * x2) * np.sin(np.sqrt(3) / 3 * k * x3)

    num = 61
    ntrain = num
    ntest = 31
    if k == 40:
        num = 59
        ntrain = num
        ntest = 30
    h = 1 / (ntrain - 1)
    m = ntrain - 2
    mm = m ** 2
    n = m ** 3

    x1 = np.linspace(0, 1, ntrain)[1:-1]
    x2 = np.linspace(0, 1, ntrain)[1:-1]
    x3 = np.linspace(0, 1, ntrain)[1:-1]

    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    Y = u(X1, X2, X3)

    train_X1 = X1.flatten().reshape(1, -1)
    train_X2 = X2.flatten().reshape(1, -1)
    train_X3 = X3.flatten().reshape(1, -1)
    train_X = np.vstack((train_X1, train_X2, train_X3))
    true_Y = Y.flatten().reshape(1, -1)

    z1 = np.linspace(0, 1, ntest)
    z2 = np.linspace(0, 1, ntest)
    z3 = np.linspace(0, 1, ntest)
    Z1, Z2, Z3 = np.meshgrid(z1, z2, z3, indexing='ij')
    Y = u(Z1, Z2, Z3)
    test_X1 = Z1.flatten().reshape(1, -1)
    test_X2 = Z2.flatten().reshape(1, -1)
    test_X3 = Z3.flatten().reshape(1, -1)
    test_X = np.vstack((test_X1, test_X2, test_X3))
    test_Y = Y.flatten().reshape(1, -1)

    def right(k):
        q = np.zeros((1, train_X.shape[1]))
        for i in range(0, q.shape[1]):
            if (i + 1) % m == 0:
                q[0][i] -= np.sin(np.sqrt(3) / 3 * k) * np.sin(np.sqrt(3) / 3 * k * train_X[0][i]) * np.sin(
                    np.sqrt(3) / 3 * k * train_X[1][i])

            if (i + 1) % mm == 0:
                for j in range(0, m):
                    q[0][i - j] -= np.sin(np.sqrt(3) / 3 * k) * np.sin(np.sqrt(3) / 3 * k * train_X[0][i - j]) * np.sin(
                        np.sqrt(3) / 3 * k * train_X[2][i - j])

            if (i + 1) % n == 0:
                for l in range(0, mm):
                    q[0][i - l] -= np.sin(np.sqrt(3) / 3 * k) * np.sin(np.sqrt(3) / 3 * k * train_X[1][i - l]) * np.sin(
                        np.sqrt(3) / 3 * k * train_X[2][i - l])
        return q

    def left(k):
        main_diag = (k ** 2 * h ** 2 - 6) * np.ones(n)

        upper_diag = np.ones(n - 1)
        for i in range(0, upper_diag.shape[0]):
            if (i + 1) % m == 0:
                upper_diag[i] = 0

        upper2_diag = np.ones(n - m)
        for i in range(0, upper2_diag.shape[0]):
            if (i + 1) % mm == 0:
                for j in range(0, m):
                    upper2_diag[i - j] = 0

        upper3_diag = np.ones(n - mm)

        A = diags(
            diagonals=[main_diag, upper_diag, upper_diag, upper2_diag, upper2_diag, upper3_diag, upper3_diag],
            offsets=[0, 1, -1, m, -m, mm, -mm],
            format="csr"
        )

        return A

    def bc():
        x_full = np.linspace(0, 1, ntrain)
        X1_full, X2_full, X3_full = np.meshgrid(x_full, x_full, x_full, indexing='ij')

        cond1 = (X1_full == 0) | (X1_full == 1)
        cond2 = (X2_full == 0) | (X2_full == 1)
        cond3 = (X3_full == 0) | (X3_full == 1)
        is_boundary = cond1 | cond2 | cond3

        boundary_X1 = X1_full[is_boundary].reshape(1, -1)
        boundary_X2 = X2_full[is_boundary].reshape(1, -1)
        boundary_X3 = X3_full[is_boundary].reshape(1, -1)

        BC = np.vstack((boundary_X1, boundary_X2, boundary_X3))
        BC_Y = u(boundary_X1, boundary_X2, boundary_X3)
        BC_Y = BC_Y.flatten().reshape(1, -1)

        return BC, BC_Y

    AA = left(k)
    B = right(k)
    BB = B.T

    train_BC, train_BC_Y = bc()
    train_X = np.hstack((train_X, train_BC))
    true_Y = np.hstack((true_Y, train_BC_Y))

    data = {}
    data['AA'] = AA
    data['BB'] = BB
    data['BC_Y'] = train_BC_Y
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