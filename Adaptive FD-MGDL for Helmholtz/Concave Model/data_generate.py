# -*- coding: utf-8 -*-

import numpy as np

from my_fun import integrate_myfun
from uniform_aoxian_Helmholtz_pml import uniform_aoxian_Helmholtz_pml

def generate_data():
    np.random.seed(0)
    ntrain = 201 
    
    x1 = np.linspace(0, 2, ntrain)
    x2 = np.linspace(0, 2, ntrain)
    
    X1, X2 = np.meshgrid(x1, x2)
    train_X1 = np.array([X1.flatten()])
    train_X2 = np.array([X2.flatten()])
    train_X = np.vstack((train_X1, train_X2))
    
    f = 25

    h = 10
    NP = 20 #layer numbers of PML
    NX = 201
    NZ = 201
    NX1 = 51
    NX2 = 151
    NZ1 = 31
    NZ2 = 101
    NZ3 = 151
    v1 = 1500
    v2 = 2000
    v3 = 2500

    f01 = 25
    TX = NX + NP + NP
    TZ = NZ + NP + NP
    N = TX * TZ

    m = NP + 101
    n = NP + 81
    temp1 = (n-1) * TX + m
    b = np.zeros((N, 1))


    A1 = uniform_aoxian_Helmholtz_pml(NP, NX, NZ, NX1, NX2, NZ1, NZ2, NZ3, v1, v2, v3, h, f01, f)
    A1 = A1.tocsr()

    b[temp1 - 1], _ = integrate_myfun(f, -0.1, 0.1)
    
    

    data = {}
    data['AA'] = A1
    data['BB'] = b
    data['NP'] = NP
    data['NZ'] = NZ
    data['NX'] = NX
    data['TX'] = TX
    data['TZ'] = TZ
    data['train_X'] = train_X
    data['train_X1'] = X1
    data['train_X2'] = X2
    data['num'] = ntrain

    return data