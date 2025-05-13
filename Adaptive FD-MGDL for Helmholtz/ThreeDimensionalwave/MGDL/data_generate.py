# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import diags

def generate_data(k):
    
    np.random.seed(0)
    phi = np.pi / 3
    theta = np.pi / 8
    k1 = k * np.cos(phi) * np.cos(theta)
    k2 = k * np.cos(phi) * np.sin(theta)
    k3 = k * np.sin(phi)
    
    def u(x1, x2, x3):
        return np.exp(1j * (k1 * x1 + k2 * x2 + k3 * x3))
    
    num = 62
    ntrain = num
    h = 1/(ntrain-1)
    m = ntrain - 2
    mm = m**2
    mmm = m**3
    ntest = 32
    
    x1 = np.linspace(0, 1, ntrain)[1:-1]
    x2 = np.linspace(0, 1, ntrain)[1:-1]
    x3 = np.linspace(0, 1, ntrain)[1:-1]
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    train_X1 = np.array([X1.flatten()])
    train_X2 = np.array([X2.flatten()])
    train_X3 = np.array([X3.flatten()])
    train_X = np.vstack((train_X1, train_X2, train_X3))
    true_Y = u(train_X1,train_X2,train_X3)
    Y_real = true_Y.real
    Y_imag = true_Y.imag
    true_Y = np.vstack((Y_real, Y_imag))

    z1 = np.linspace(0, 1, ntest)
    z2 = np.linspace(0, 1, ntest)
    z3 = np.linspace(0, 1, ntest)
    Z1, Z2, Z3 = np.meshgrid(z1, z2, z3, indexing='ij')
    Y = u(Z1, Z2, Z3)
    test_X1 = np.array([Z1.flatten()])
    test_X2 = np.array([Z2.flatten()])
    test_X3 = np.array([Z3.flatten()])
    test_X = np.vstack((test_X1, test_X2, test_X3))
    test_Y = np.array([Y.flatten()])
    Y_real = test_Y.real
    Y_imag = test_Y.imag
    test_Y = np.vstack((Y_real, Y_imag))
    
    
    def left(k):
        main_diag = (k**2*h**2-6) * np.ones(mmm)
        
        upper_diag = np.ones(mmm - 1)
        for i in range(0,upper_diag.shape[0]):
            if (i+1) % m ==0:
                upper_diag[i] = 0
        
        upper2_diag = np.ones(mmm - m)
        for i in range(0,upper2_diag.shape[0]):
            if (i+1) % mm ==0:
                for j in range(0, m):
                    upper2_diag[i-j] = 0
        
        upper3_diag = np.ones(mmm - mm)
        
        A = diags(
            diagonals=[main_diag, upper_diag, upper_diag, upper2_diag, upper2_diag, upper3_diag, upper3_diag],
            offsets=[0, 1, -1, m, -m, mm, -mm],
            format="csr"
            )

        return A
    
    
    def right(k):
        q = np.zeros((1,train_X.shape[1]), dtype=complex)
        for i in range(0,q.shape[1]):
            if (i+1) % m ==0:
                q[0][i] -= np.exp(1j * (k1 * train_X[0][i] + k2 * train_X[1][i] + k3))
            
            if (i+1) % mm ==0:
                for j in range(0,m):
                    q[0][i-j] -= np.exp(1j * (k1 * train_X[0][i-j] + k2 + k3 * train_X[2][i-j]))
            
            if (i+1) % mmm ==0:
                for l in range(0,mm):
                    q[0][i-l] -= np.exp(1j * (k1 + k2 * train_X[1][i-l] + k3 * train_X[2][i-l]))
            
            if (i+1) % m ==1:
                q[0][i] -= np.exp(1j * (k1 * train_X[0][i] + k2 * train_X[1][i]))
            
            if (i+1) % mm ==1:
                for j in range(0,m):
                    q[0][i+j] -= np.exp(1j * (k1 * train_X[0][i+j] + k3 * train_X[2][i+j]))
            
            if i ==0:
                for l in range(0,mm):
                    q[0][i+l] -= np.exp(1j * (k2 * train_X[1][i+l] + k3 * train_X[2][i+l]))
                
        return q
    
    
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
        BC_Y = np.vstack((BC_Y.real, BC_Y.imag))
        
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

