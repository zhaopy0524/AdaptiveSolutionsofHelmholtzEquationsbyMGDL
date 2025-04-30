# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from aoxian_v_model import v_aoxian_model
from nonuiform_sxsy import nonuiform_sxsy

def uniform_aoxian_Helmholtz_pml(NP,NX,NZ,NX1,NX2,NZ1,NZ2,NZ3,v1,v2,v3,h,f01,f1):
    
    NPX = NP
    NPZ = NP
    f0 = f01
    f = f1
    TX = NX + NPX + NPX
    TZ = NZ + NPZ + NPZ
    N = TX * TZ
    
    r = 1
    
    ta = 0.7926
    ts = 0.0942
    tc = -0.0016
    
    # ta=(0.5461+1)/2             
    # ts=0.3752/4
    # tc=-1e-5
    
    # ta = 1
    # ts = 0
    # tc = 0
    
    rp = 1 / (r**2)
    
    A = sp.lil_matrix((N, N), dtype=complex)
    
    
    for n in range(NPZ + 2, NZ + NPZ - 1 + 1):
        for m in range(NPX + 2, NX + NPX - 1 + 1):
            temp = (n - 1) * TX + m
            
            nw = n - 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AC = -(1 / 2) * (1 + rp) * (1 - ta) * ((1 / h) ** 2) - tc * k ** 2
            A[temp - 1, temp - TX - 2] = AC  # 注意 Python 索引从 0 开始
            
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AS1 = (1 - ta - ta * rp) * ((1 / h) ** 2) - ts * k ** 2
            A[temp - 1, temp - TX - 1] = AS1
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AC = -(1 / 2) * (1 + rp) * (1 - ta) * ((1 / h) ** 2) - tc * k ** 2
            A[temp - 1, temp - TX] = AC
    
            nw = n
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AS2 = ((1 - ta) * rp - ta) * ((1 / h) ** 2) - ts * k ** 2
            A[temp - 1, temp - 2] = AS2
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            A0 = (2 * ta) * ((1 / h) ** 2) * (1 + rp) - (1 - 4 * ts - 4 * tc) * k ** 2
            A[temp - 1, temp - 1] = A0
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AS2 = ((1 - ta) * rp - ta) * ((1 / h) ** 2) - ts * k ** 2
            A[temp - 1, temp] = AS2
    
            nw = n + 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AC = -(1 / 2) * (1 + rp) * (1 - ta) * ((1 / h) ** 2) - tc * k ** 2
            A[temp - 1, temp + TX - 2] = AC
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AS1 = (1 - ta - ta * rp) * ((1 / h) ** 2) - ts * k ** 2
            A[temp - 1, temp + TX - 1] = AS1
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            AC = -(1 / 2) * (1 + rp) * (1 - ta) * ((1 / h) ** 2) - tc * k ** 2
            A[temp - 1, temp + TX] = AC
            
    n = 1
    for m in range(1, TX + 1):
        temp = (n - 1) * TX + m
        mw = m
        nw = n
    
        aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
        nw = n
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
        nw = n + 1
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
        if m == 1:
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX] = B9
        elif m == TX:
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
        else:
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp] = B6
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
            A[temp - 1, temp + TX] = B9

    n = TZ
    for m in range(1, TX + 1):
        temp = (n - 1) * TX + m
        mw = m
        nw = n
    
        aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
        nw = n - 1
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])
    
        nw = n
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
        if m == 1: 
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX] = B3
        elif m == TX: 
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX - 2] = B1
        else: 
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp] = B6
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX - 2] = B1
            A[temp - 1, temp - TX] = B3

    m = 1
    for n in range(2, TZ):
        temp = (n - 1) * TX + m
        mw = m
        nw = n
    
        aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
        nw = n - 1
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])
    
        nw = n
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
        nw = n + 1
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
        mw = m + 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
        A[temp - 1, temp - 1] = B5
        A[temp - 1, temp] = B6
        A[temp - 1, temp - TX - 1] = B2
        A[temp - 1, temp - TX] = B3
        A[temp - 1, temp + TX - 1] = B8
        A[temp - 1, temp + TX] = B9
        
    
    m = TX
    for n in range(2, TZ):
        temp = (n - 1) * TX + m
        mw = m
        nw = n
    
        aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
        nw = n - 1
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
        nw = n
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
        nw = n + 1
        mw = m - 1
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
    
        mw = m
        v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
        k = 2 * np.pi * f / v
        B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
        A[temp - 1, temp - 1] = B5
        A[temp - 1, temp - 2] = B4
        A[temp - 1, temp - TX - 1] = B2
        A[temp - 1, temp - TX - 2] = B1
        A[temp - 1, temp + TX - 1] = B8
        A[temp - 1, temp + TX - 2] = B7
        

    for n in range(2, NPZ + 2):
        for m in range(2, TX):
            temp = (n - 1) * TX + m
            mw = m
            nw = n
    
            aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
            nw = n - 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])
    
            nw = n
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
            nw = n + 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
            
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX] = B3
            A[temp - 1, temp - TX - 2] = B1
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
            A[temp - 1, temp + TX] = B9
    
            
    for n in range(NZ + NPZ, TZ):
        for m in range(2, TX):
            temp = (n - 1) * TX + m
            mw = m
            nw = n
    
            aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
            nw = n - 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])
    
            nw = n
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
            nw = n + 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX] = B3
            A[temp - 1, temp - TX - 2] = B1
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
            A[temp - 1, temp + TX] = B9        
        
        
    for n in range(NPZ + 2, NZ + NPZ - 1 + 1):
        for m in range(2, NPX + 2):
            temp = (n - 1) * TX + m
            mw = m
            nw = n
    
            aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
            nw = n - 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])
    
            nw = n
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
            nw = n + 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX] = B3
            A[temp - 1, temp - TX - 2] = B1
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
            A[temp - 1, temp + TX] = B9        
            

    for n in range(NPZ + 2, NZ + NPZ - 1 + 1):
        for m in range(NX + NPX, TX):
            temp = (n - 1) * TX + m
            mw = m
            nw = n
    
            aa, bb = nonuiform_sxsy(NX, NZ, NPX, NPZ, temp, f0, f)
    
            nw = n - 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B1 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[1]) + rp * ((1 / h) ** 2) * (aa[0] / bb[3])) + k ** 2 * tc * aa[0] * bb[4])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B2 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[4] / aa[3] - bb[4] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[3]) + k ** 2 * ts * aa[2] * bb[4])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B3 = -((1 - ta) * (1 / 2) * (((1 / h) ** 2) * (bb[4] / aa[3]) + rp * ((1 / h) ** 2) * (aa[4] / bb[3])) + k ** 2 * tc * aa[4] * bb[4])

            nw = n
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B4 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[0] / bb[1] - aa[0] / bb[3]) + k ** 2 * ts * aa[0] * bb[2])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B5 = -(ta * ((1 / h) ** 2) * (-bb[2] / aa[3] - bb[2] / aa[1]) + ta * rp * ((1 / h) ** 2) * (-aa[2] / bb[1] - aa[2] / bb[3]) + k ** 2 * (1 - 4 * ts - 4 * tc) * aa[2] * bb[2])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B6 = -(ta * ((1 / h) ** 2) * (bb[2] / aa[3]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (-aa[4] / bb[1] - aa[4] / bb[3]) + k ** 2 * ts * aa[4] * bb[2])
    
            nw = n + 1
            mw = m - 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B7 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[1]) + (1 - ta) * (1 / 2) * rp * ((1 / h) ** 2) * (aa[0] / bb[1])) + k ** 2 * tc * aa[0] * bb[0])
    
            mw = m
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B8 = -((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (-bb[0] / aa[3] - bb[0] / aa[1]) + ta * rp * ((1 / h) ** 2) * (aa[2] / bb[1]) + k ** 2 * ts * aa[2] * bb[0])
    
            mw = m + 1
            v = v_aoxian_model(NP, NX1, NX2, NZ1, NZ2, NZ3, mw, nw, v1, v2, v3)
            k = 2 * np.pi * f / v
            B9 = -(((1 - ta) * (1 / 2) * ((1 / h) ** 2) * (bb[0] / aa[3]) + (1 - ta) * (1 / 2) * ((1 / h) ** 2) * rp * (aa[4] / bb[1])) + k ** 2 * tc * aa[4] * bb[0])
    
            A[temp - 1, temp - 1] = B5
            A[temp - 1, temp] = B6
            A[temp - 1, temp - 2] = B4
            A[temp - 1, temp - TX - 1] = B2
            A[temp - 1, temp - TX] = B3
            A[temp - 1, temp - TX - 2] = B1
            A[temp - 1, temp + TX - 1] = B8
            A[temp - 1, temp + TX - 2] = B7
            A[temp - 1, temp + TX] = B9

    return A
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        