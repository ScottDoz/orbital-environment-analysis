# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:30:55 2021

@author: scott
"""

import numpy as np
import vectorized_kepler

def test_kepler_loop():
    
    
    # Test data
    CNT = 1e6
    e = np.linspace(0, 0.7, int(CNT), dtype=np.dtype("f8")) # C0 argument
    M = np.linspace(0, 2*np.pi, int(CNT), dtype=np.dtype("f8")) # C0 argument
    XTOL, RTOL = 1e-8, 1e-8
    MITR = 10  # other solver parameters
    
    
    # Call the function
    x_loop = vectorized_kepler.Kepler_loop(e=e, M=M, xtol=1E-8, rtol=1e-8, mitr=MITR)
    # Convert output to numpy
    x = np.asarray(x_loop)[:]
    
    
    # Check solition
    # f(x) = x - e*sin(x) - M = 0
    f = x - e*np.sin(x) - M
    
    
    return x, f