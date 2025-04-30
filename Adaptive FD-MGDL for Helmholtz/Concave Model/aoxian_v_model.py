# -*- coding: utf-8 -*-


def v_aoxian_model(NP,NX1,NX2,NZ1,NZ2,NZ3,m,n,v1,v2,v3):
    flag = 1
    
    if n <= (NP + NZ1):
        v = v1
        flag = 0
    
    if (flag == 1) and ((NP + NZ1) < n) and (n <= (NP + NZ2)):
        v = v2
        flag = 0
    
    if (flag == 1) and ((NP + NZ2) < n) and (n <= (NP + NZ3)):
        if ((NP + NX1) <= m) and (m <= (NP + NX2)):
            v = v2
            flag = 0
    
    if (flag == 1) and ((NP + NZ2) < n) and (n <= (NP + NZ3)):
        if m < (NP + NX1):
            v = v3
            flag = 0

    if (flag == 1) and ((NP + NZ2) < n) and (n <= (NP + NZ3)):
        if (NP + NX2) < m:
            v = v3
            flag = 0

    if (flag == 1) and (n > (NP + NZ3)):
        v = v3
        flag = 0
        
    return v