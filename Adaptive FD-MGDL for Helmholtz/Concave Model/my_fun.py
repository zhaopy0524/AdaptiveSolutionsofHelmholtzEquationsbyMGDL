# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate as integrate

def myfun(t, f):
    f0 = 25
    return (1 - 2 * np.pi**2 * f0**2 * t**2) * np.exp(-np.pi**2 * f0**2 * t**2) * np.cos(2 * np.pi * f * t)

def integrate_myfun(f, t_lower, t_upper):
    result, error = integrate.quad(myfun, t_lower, t_upper, args=(f,))
    return result, error