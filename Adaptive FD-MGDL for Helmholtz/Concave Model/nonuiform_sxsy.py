# -*- coding: utf-8 -*-


import numpy as np

def nonuiform_sxsy(NX,NZ,NPX,NPZ,n1,f0,f):
    
    b0 = 1.79
    w = 2 * np.pi * f
    aa = np.zeros(5, dtype=complex)
    bb = np.zeros(5, dtype=complex)
    
    m0 = ((2 * np.pi * b0 * f0) / w) * 1j
    
    flag = 1
    
    N = NX + NPX + NPX
    # 第一部分
    if flag == 1:
        for j in range(NPX + 1):
            if (n1 % N) == (NPX + 1 - j):
                aa[2] = 1 - m0 * (j / NPX) ** 2
                aa[0] = 1 - m0 * ((j + 1) / NPX) ** 2
                aa[1] = 1 - m0 * ((j + 0.5) / NPX) ** 2
                if aa[2] == 1:
                    aa[3] = 1
                    aa[4] = 1
                else:
                    aa[3] = 1 - m0 * ((j - 0.5) / NPX) ** 2
                    aa[4] = 1 - m0 * ((j - 1) / NPX) ** 2
                flag = 0
                break
    
    # 第二部分
    if flag == 1:
        for j in range(NPX):
            if (n1 % N) == (N - NPX + j):
                aa[2] = 1 - m0 * (j / NPX) ** 2
                aa[3] = 1 - m0 * ((j + 0.5) / NPX) ** 2
                aa[4] = 1 - m0 * ((j + 1) / NPX) ** 2
                if aa[2] == 1:
                    aa[0] = 1
                    aa[1] = 1
                else:
                    aa[0] = 1 - m0 * ((j - 1) / NPX) ** 2
                    aa[1] = 1 - m0 * ((j - 0.5) / NPX) ** 2
                flag = 0
                break
    
    # 第三部分
    if flag == 1:
        if (n1 % N) == 0:
            aa[2] = 1 - m0 * (NPX / NPX) ** 2
            aa[0] = 1 - m0 * ((NPX - 1) / NPX) ** 2
            aa[1] = 1 - m0 * ((NPX - 0.5) / NPX) ** 2
            aa[3] = 1 - m0 * ((NPX + 0.5) / NPX) ** 2
            aa[4] = 1 - m0 * ((NPX + 1) / NPX) ** 2
            flag = 0
    
    # 第四部分
    if flag == 1:
        if (n1 % N) > (NPX + 1) and (n1 % N) < (NPX + NX):
            aa[0] = 1
            aa[1] = 1
            aa[2] = 1
            aa[3] = 1
            aa[4] = 1
            flag = 0
    
    # 第五部分
    if flag == 1:
        print('error')
    
    # 重置 flag
    flag = 1
    
    # 第六部分
    if flag == 1:
        if (n1 > (N * (NPZ + 1))) and (n1 <= (N * (NZ + NPZ - 1))):
            bb[0] = 1
            bb[1] = 1
            bb[2] = 1
            bb[3] = 1
            bb[4] = 1
            flag = 0
    
    # 第七部分
    if flag == 1:
        if (n1 >= 1) and (n1 <= N * NPZ):
            for j in range(NPZ):
                if (n1 >= (1 + j * N)) and (n1 <= ((j + 1) * N)):
                    bb[2] = 1 - m0 * ((NPZ - j) / NPZ) ** 2
                    bb[1] = 1 - m0 * ((NPZ - j - 0.5) / NPZ) ** 2
                    bb[0] = 1 - m0 * ((NPZ - j - 1) / NPZ) ** 2
                    bb[3] = 1 - m0 * ((NPZ - j + 0.5) / NPZ) ** 2
                    bb[4] = 1 - m0 * ((NPZ - j + 1) / NPZ) ** 2
                    flag = 0
                    break
    
    # 第八部分
    if flag == 1:
        if (n1 >= (N * NPZ + 1)) and (n1 <= (N * (NPZ + 1))):
            bb[2] = 1
            bb[0] = 1
            bb[1] = 1
            bb[4] = 1 - m0 * (1 / NPZ) ** 2
            bb[3] = 1 - m0 * ((0.5) / NPZ) ** 2
            flag = 0
    
    # 第九部分
    if flag == 1:
        if (n1 >= (N * (NZ + NPZ - 1) + 1)) and (n1 <= (N * (NZ + NPZ))):
            bb[2] = 1
            bb[0] = 1 - m0 * (1 / NPZ) ** 2
            bb[1] = 1 - m0 * ((0.5) / NPZ) ** 2
            bb[3] = 1
            bb[4] = 1
            flag = 0
    
    # 第十部分
    if flag == 1:
        if (n1 > (N * (NZ + NPZ))) and (n1 <= (N * (NZ + NPZ + NPZ))):
            for j in range(1, NPZ + 1):
                if (n1 >= (1 + (j - 1 + NZ + NPZ) * N)) and (n1 <= ((NZ + NPZ + j) * N)):
                    bb[2] = 1 - m0 * (j / NPZ) ** 2
                    bb[0] = 1 - m0 * ((j + 1) / NPZ) ** 2
                    bb[1] = 1 - m0 * ((j + 0.5) / NPZ) ** 2
                    bb[3] = 1 - m0 * ((j - 0.5) / NPZ) ** 2
                    bb[4] = 1 - m0 * ((j - 1) / NPZ) ** 2
                    flag = 0
                    break
    
    # 第十一部分
    if flag == 1:
        print('error')
    
    
    return aa, bb

    
    