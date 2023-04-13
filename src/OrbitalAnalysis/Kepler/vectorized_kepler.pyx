# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 19:46:09 2021

@author: scott

Vectorize Kepler's equation using cython loop and scipy brentq solver

"""

import numpy as np
from scipy.optimize.cython_optimize cimport brentq

# import math from Cython
from libc cimport math


# Kepler's equation -----------------------------------------------------------
# Define functions representing Kepler's equation

# Structure for extra parameters of functions
ctypedef struct test_params:
    double e
    double M

# Kepler's equation for Elliptical orbits
# f(x) = x - e*sin(x) - M = 0 
cdef double func_E(double x, void *args):
    cdef test_params *myargs = <test_params *> args
    return x -myargs.e*math.sin(x) - myargs.M

# Kepler's equation for Hyperbolic orbits
# f(x) = e*sinh(x) - x - M = 0 
cdef double func_H(double x, void *args):
    cdef test_params *myargs = <test_params *> args
    return myargs.e*math.sinh(x) - x - myargs.M


# Cython wrapper function
cdef int Kepler_cython(double[:] e, double[:] M, double xtol, double rtol, int mitr, double[:] result):
    cdef Py_ssize_t i = 0
    cdef test_params myargs

    #for i in prange(c0.shape[0], nogil=True):
    while i < len(M):
        myargs.e = e[i]
        myargs.M = M[i]
        if e[i]<1.:
            # Elliptical orbit
            result[i] = brentq(func_E, 0., 2*np.pi, <test_params *> &myargs, xtol, rtol, mitr, NULL)
        elif e[i]>=1:
            # Hyperbolic orbit
            result[i] = brentq(func_H, 0., 2*np.pi, <test_params *> &myargs, xtol, rtol, mitr, NULL)
        i += 1
    return 0

# Python wrapper function
def Kepler_loop(e, M, xtol=1e-5, rtol=1e-5, mitr=10):
    '''
    Python wrapper to Kepler_cython Cython function.

    Vectorized implementation of Kepler's equation using scipy brent solver
    within a Cython loop.
    '''
    cdef double[:] result = np.empty_like(M)
    if not Kepler_cython(e, M, xtol, rtol, mitr, result):
        return result