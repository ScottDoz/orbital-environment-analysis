# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:10:08 2021

@author: scott

Orbital Mechanics Functions

This code is copied from the sr_tools project - under development.

"""

# Standard imports
import numpy as np
import pandas as pd
import copy
from numba import jit
import pdb
import time

import astropy
from astroquery.jplhorizons import Horizons
from astroquery.jplsbdb import SBDB
import poliastro
import pykep as pk

# Spice
import spiceypy as spice
import spiceypy.utils.support_types as stypes
from spiceypy.utils.libspicehelper import libspice

# Plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
import plotly.express as px

from skimage import measure
# import cv2
from skimage.feature import peak_local_max
import scipy.signal

# Package imports
from Kepler import vectorized_kepler

# Supress numpy warnings
np.seterr(invalid='ignore')


#%% ###########################################################################
#                  Orbital Elements Subroutines
#
# Algorithms from Curtis and other sources
# #############################################################################


#%% Kepler's Problem ----------------------------------------------------------

# Cythonized version
def Kepler(e,M,xtol=1E-8, rtol=1e-8, mitr=100):
    '''
    Solve Kepler's equation to convert mean anomaly M to eccentric anomaly E, 
    for a Keplerian orbit of eccentricity e.
    
    f(x) = E - e*sin(E) - M = 0 ; for 0 < e < 1 (Ellicptical);
    f(x) = e*sinh(F) - F = 0    ; for     e >= 1 (Hyperbolic)
    
    This is a vectorized function, accepting arrays of inputs. 
    It utilizes Cython code to loop over each element, and solve the equation
    using the scipy implementation of Brent's method (scipy.optimize.brentq). 

    Parameters
    ----------
    e, M : float or 1D array
        Eccentricity and mean anomaly to be converted.
        Can be combinations of float or arrays.
    xtol : float, optional
        DESCRIPTION. The default is 1E-8.
    rtol : float, optional
        DESCRIPTION. The default is 1e-8.
    mitr : float, optional
        Max iterations. The default is 10.

    Returns
    -------
    x : TYPE
        Eccentric anomaly.

    '''
    
    # Check if inputs are arrays
    if isinstance(M, (list, tuple, np.ndarray)):
        # Multiple epochs
        if isinstance(e, (list, tuple, np.ndarray)) == False:
            # Single object, multiple epochs
            # Convert e to an array the same size as M
            e = e*np.ones(len(M))
    
    else:
        # Single epoch
        
        if isinstance(e, (list, tuple, np.ndarray)):
            # Multiple objects at te same epoch.
            # Convert M to array of same size as e
            M = M*np.ones(len(E))
        
        else:
            # Both e and M are floats.
            # Convert them to arrays to match required inputs
            e = np.array([e])
            M = np.array([M])

    # Call Cython module function
    x_loop = vectorized_kepler.Kepler_loop(e=e, M=M, xtol=xtol, rtol=rtol, mitr=mitr)
    # Convert output to numpy
    x = np.asarray(x_loop)[:]

    # If single value, convert to float
    if len(x)==1:
        x = float(x)

    return x


# @jit(nopython=True) # FIXME: numba is not working on this function
def Kepler_Newton(e,M,units='rad',tol=1.0E-10, max_iter = 100):
    '''
    Depreciated! Use Kepler() function instead. Much faster.
    
    This function uses Newton's method to solve Kepler's equation for both Elliptical and
    Hyperbolic orbits to find the Eccentric and Hyperbolic anomalies:
    f(x) = E - e*sin(E) - M = 0 (Ellicptical);
    f(x) = e*sinh(F) - F = 0 (Hyperbolic)

    This method is vectorized, and can accept multiple input pairs of e,M.
    These can be a mixture of elliptical (e<1) and hyperbolic (e>1) orbits.
    In the case of hyperbolic orbits, the method returns the Hyperbolic anomaly F.

    Adapted from Palido & Peláez (2016)

    Reference:
        Palido & Peláez, 2016. An efficient code to solve the Kepler's equation for elliptic and 
        hyperbolic orbits.
        https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf

    # Example 1. Solve for single object
    # Known solution E = 3.47942 (Example 3_02 of Curtis, pg 597).
    # >> E,err = Kepler_Newton(e=0.37255,M=3.6029) # Solve kepler
    
    # Example 2. Solve for all asteroids in MPCORB
    # >> df = Datasets.load_dataset('MPCORB') # Load MPCORB data
    # >> e = np.array(df.e) # Load e vector
    # >> M = np.deg2rad(np.array(df.M)) # Load M vector (rad)
    # >> E,err = Kepler_E(e,M) # Solve kepler
    
    # Example 3. Solve for all asteroids in JPL_SBDB
    # >> df = Datasets.load_dataset('JPL_SBDB') # Load JPL_SBDB data
    # >> e = np.array(df.e) # Load e vector
    # >> M = np.deg2rad(np.array(df.M)) # Load M vector (rad)
    # >> E,err = Kepler_E(e,M) # Solve kepler
    
    Parameters
    ----------
    e : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    tol : float
        Tolerance.

    Returns
    -------
    E : TYPE
        DESCRIPTION.

    '''
    # Future improvements:
    
    # 1. Convert to C++ compiled version 
    # see example: with python wrapper
    # https://github.com/dfm/kepler.py
    
    # 2. Create javascript version for web app
    # see: https://stackoverflow.com/questions/5287814/solving-keplers-equation-computationally
    # and: http://www.jgiesen.de/kepler/kepler.html
    
    # Depretiation warning
    print('Warning! Kepler_Newton() is depreciated. Use Kepler() function instead. Much faster.')
    
    # Start timer
    start_time = time.time()
    
    # 0. Setup and argument checks
    
    # If single value, convert to numpy array
    float_flag = 0
    if type(e) == float:
        e = np.array([e])
        M = np.array([M])
        float_flag = 1 # Mark a flag to convert solution back to float
    
    # Check lengths of vectors are the same
    if (len(e) != len(M)):
        raise ValueError('Length of e and M must be equal.')
    
    # Convert M to radians
    if units in ['deg','degrees']:
        M = np.deg2rad(M)
        
    # Check for hyperbolic objects
    # TODO: Add workaround for hyperbolic objects
    # https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf
    hyp_flag = 0
    if max(e) > 1.:
        hyp_flag = 1
        
        print('Warning! Population includes hyperbolic objects.')
        print('         These will be set to NaN.')
        
        # Find indicies of hyperbolic asteroids
        hyp_mask = e>1. # Get array of booleans for hyperbolic objects
        
        # Split population into two vectors
        # Replace eccentricity of hyperbolic functions to allow use in the function
        
        # Eliptical objects
        e1 = copy.deepcopy(e)# e.copy() # Copy original e values
        e1[hyp_mask] = 0. # Set e=0 for hyperbolic objects
        
        # Eliptical objects
        e2 = copy.deepcopy(e)# e.copy() # Copy original e values
        e2[~hyp_mask] = 0. # Set e=0 for eliptical objects
        
    
    # 1. Choose initial estimate of root E following (Prussing & Conway, 1993).
    # If Me < pi, then E=Me+e/2
    # If Me > pi, then E=Me-e/2
    E = np.nan*np.zeros(len(M)) # Initialize with nan
    E[M < np.pi] = M[M < np.pi] + e[M < np.pi]/2 # For M < pi
    E[M >= np.pi] = M[M >= np.pi] - e[M >= np.pi]/2 # For Me > pi
    
    
    # 2. Define the function to be solved and its derivative 
    #    f(Ei) = Ei - esinEi - Me, and
    #    f'(Ei) = 1 - ecosEi
    # Define equivalent equation of Hyperbolic orbits
    @jit(nopython=True) # Use numba to improve speed
    def func_E(x,e,M):
        '''       
        Define and evaluate the functions f(x) representing Kepler's equation for Elliptical orbits:
        
        f(x) = x - e*sin(x) - M = 0 
        where x = E is the Eccentric anomaly; and
              M is the Mean anomaly
        
        Solution uses an iterative method:
        x_i+1 = x_i + dx
        
        where dx is computed from a modified Newton-Raphson algorithm:
            dx = -2*f(x)/(f'(x) + sign(f'(x))*sqrt( f'(x)^2 - 2*f(x)*f''(x) ) )
            f'(x)  = 1. - e*cos(x)
            f''(x) = e*sin(x)
        
        Reference:
        Palido & Peláez, 2016. An efficient code to solve the Kepler's equation for elliptic and 
        hyperbolic orbits.
        https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf
            
        Parameters
        ----------
        x : TYPE
            x = E Eccentric anomaly
        e : TYPE
            Eccentricity
        M : TYPE
            Mean anomaly.

        Returns
        -------
        fx : TYPE
            DESCRIPTION.
        dfx : TYPE
            DESCRIPTION.
        ratio : TYPE
            DESCRIPTION.

        '''
        
        # Keplers equation
        # From: https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf
        # (equations have been changed: x->M and y->x)
        
        # Eliptical case
        # Me = y - e*sin(y)
        # where x = E (eccentric anomaly)
        fx = x - e*np.sin(x) - M   # Function to solve f(x) = 0
        dfx = 1. - e*np.cos(x)     # 1st derivative f'(x)
        ddfx = e*np.sin(x)         # 2nd derivative f''(x)
        ratio = 2.*fx/( dfx + np.sign(dfx)*np.sqrt( np.abs( dfx**2 - 2*fx*ddfx ) ) )
        
        # Alternative approaches
        
        # Solution is based on different order approximations of the Taylor series
        # expansion of f(x - eps), where eps = x-E is the error in solution.
        # See: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
        
        # Let error eps = x-E. Take Taylor series expansion about x=E:
        # f(E) = f(x - eps) 
        #       = x - e*sin(x) - M - (1-e*cos(x))*eps       # 1st order terms
        #               + (1/2)*eps^2*e*sin(x)              # 2nd order terms
        #               - (1/6)*eps^3*e*cos(x)              # 3rd order terms
        #               + ...
        
        # # 1st degree taylor expansion approximation (10 iterations, ~2.1 seconds)
        # fx = x - e*np.sin(x) - M # F(x)
        # dfx = 1. - e*np.cos(x)
        # ratio = fx/dfx
        
        # # 2nd degree taylor expansion approximation (6 iterations, ~1.5 seconds)
        # fx = (x - e*np.sin(x) - M)
        # dfx = (1 - e*np.cos(x) - 0.5*e*np.sin(x)*(x - e*np.sin(x) - M)/(1 - e*np.cos(x)) )
        # ratio = fx/dfx
        
        # # 3rd degree Taylor expansion approximation (5 iterations, ~1.52 seconds)
        # # (2 step solution)
        # fx = (x - e*np.sin(x) - M)
        # eps_n = fx/(1 - e*np.cos(x) - 0.5*e*np.sin(x)*(x - e*np.sin(x) - M)/(1 - e*np.cos(x)) ) # 2nd-degree approx
        # dfx = (1 - e*np.cos(x) - 0.5*( e*np.sin(x) - (1./3.)*e*np.cos(x)*eps_n)*eps_n  )
        # ratio = fx/dfx
        
        return fx,dfx,ratio
    
    # Define equivalent equation of Hyperbolic orbits
    @jit(nopython=True) # Use numba to improve speed
    def func_H(x,e,M):
        '''       
        Define and evaluate the functions f(x) representing Kepler's equation for Hyperbolic orbits:
        
        f(x) = e*sinh(x) - x - M = 0 
        where x = F is the Hyperbolic anomaly; and
              M is the Mean anomaly
        
        Solution uses an iterative method:
        x_i+1 = x_i + dx
        
        where dx is computed from a modified Newton-Raphson algorithm:
            dx = -2*f(x)/(f'(x) + sign(f'(x))*sqrt( f'(x)^2 - 2*f(x)*f''(x) ) )
            f'(x)  = e*cosh(x) - 1. 
            f''(x) = e*sinh(x)
        
        Reference:
        Palido & Peláez, 2016 An efficient code to solve the Kepler's equation for elliptic and 
        hyperbolic orbits.
        https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf
            
        Parameters
        ----------
        x : TYPE
            x = F Hyperbolic anomaly
        e : TYPE
            Eccentricity
        M : TYPE
            Mean anomaly.

        Returns
        -------
        fx : TYPE
            DESCRIPTION.
        dfx : TYPE
            DESCRIPTION.
        ratio : TYPE
            DESCRIPTION.

        '''
        
        # Keplers equation
        # From: https://indico.esa.int/event/111/contributions/312/attachments/544/589/VRP_ICATT2016.pdf
        # (equations have been changed: x->M and y->x)
        
        # Hyperbolic case
        # Mh = e*sinh(y) - y
        # where x = H (Hyperbolic anomaly)
        fx = e*np.sinh(x) - x - M  # Function to solve f(y) = 0
        dfx = e*np.cosh(x) - 1.    # 1st derivative f'(x)
        ddfx = e*np.sinh(x)      # 2nd derivative f''(x)
        ratio = 2.*fx/( dfx + np.sign(dfx)*np.sqrt( np.abs( dfx**2 - 2*fx*ddfx ) ) )
        
        
        return fx,dfx,ratio
    
    
    # 3. Iteration loop 
    # Update initial guess incrementally following Newton-Rhapson method
    # E_i+1 = E_i - f(Ei)/f'(Ei)
    count = 0 # Initialize iteration count
    max_error = tol +  1. # Initialize max error
    ratio = np.ones(len(M)) # Initialize ratio f(Ei)/f'(Ei)
    tol = np.float64(tol) # Convert tolerance to float64
    while (abs(max_error) > tol):
        
        # Check if maximum iterations has been reached
        if count == max_iter:
            print('Max iterations reached ({})'.format(max_iter) )
            break
        
        if hyp_flag == 0:
            # Case1: No hyperbolic objects
        
            # Values for Eliptical objets
            f, df, ratio = func_E(E,e,M)
        
            
        elif hyp_flag == 1:
            # Case2: Hyperbolic objects
            
            # Values for Eliptical objects
            f, df, ratio = func_E(E,e1,M)
            
            # Values for Hyberbolic objecs
            f_H, df_H, ratio_H = func_H(E,e2,M)
            
            # Replace solutions for hyperbolic objects
            f[hyp_mask] = f_H[hyp_mask]
            df[hyp_mask] = df_H[hyp_mask]
            ratio[hyp_mask] = ratio_H[hyp_mask]
        
        # Compute the error
        max_ratio = max(abs(ratio))
        max_error = max(abs(f))
        
        # Increment the solution (E_i+1 = E_i - ratio)
        E -= ratio
        
        # Increment iteration count
        count += 1
    # End while loop
    
    # Rename the error
    err = f
    
    # Convert single float back to float
    if float_flag == 1:
        E = float(E)
        ratio = float(ratio)
        err = float(err)
    
    # Print convergence message
    if count < max_iter:            
        # print('Solution converged in {} iterations'.format(count) )
        pass
    
    # Print timing message
    end_time = time.time()
    # print("Time elapsed {} s".format(end_time - start_time) )
    
    return E, err


# Kepler's problem using cspice

def kepleq(e,w,om,M):
    ''' Vectorized form of kepler's equation using spiceypy.kepleq '''
    
    # Check if inputs are vectors
    multiepochflag = False
    if isinstance(M, (list, tuple, np.ndarray)):
        # M is provided as list
        multiepochflag = True
        # Check if e,w,om also provided as lists
        if isinstance(e, (list, tuple, np.ndarray)) == False:
            # Single object. e is provided as a float.
            # Convert e,w,om to vectors of the same length as M
            e = e*np.ones(len(M))
            w = w*np.ones(len(M))
            om = om*np.ones(len(M))
        
    # Convert to equinoctial elements
    # ML = M + w+om Mean longitude
    # H = ecc*sin(w+om) h component of equinoctial elements
    # K = ecc*cos(w+om) k component of equinoctial elements
    ML = M + w + om
    H = e*np.sin(w+om)
    K = e*np.cos(w+om)
    
    if multiepochflag == True:
        # List of values. Loop over each
        # Loop over each item
        F = np.zeros(len(M))*np.nan # Initialize
        for i in range(len(M)):
            try:
                F[i] = spice.kepleq(ML[i],H[i],K[i])
            except:
                F[i] = np.nan
    
    else:
        # Single item
        F = spice.kepleq(ML,H,K)
        
    # Convert solution to Eccentric anomaly
    # F = E - argument of periapse - longitude of ascending node.
    E = F + w + om
    
    return E

#%% Converting TA,M,E

# Two Directions for conversions
# Forwards: TA -> E -> M (deterministic computations)
# Backwards: M -> E -> TA (numeric computations)

# Forward conversions

def TA_to_M(TA,e):
    ''' Convert true anomaly to Mean anomaly '''
    
    
    # Input checks. Convert floats to arrays
    if type(e) in [int,float,np.float64]:
        e = np.array([e]) # Convert to float to array
    if type(TA) in [int,float,np.float64]:
        TA = np.array([TA]) # Convert to float to array
    
    # Input cases
    if (len(e)==1) & (len(TA)>1):
        # Single orbit, mutliple epochs
        # Repeat e to match length of TA
        e = np.repeat(e, len(TA))
    elif (len(TA)==1) & (len(e)>1):
        # Single epoch, mutliple orbits
        # Repeat TA to match length of e
        TA = np.repeat(TA, len(e)) 
    
    # Eq 3.3 of Curtis
    # Elliptical case
    M = 2*np.arctan(np.sqrt( (1-e)/(1+e) )*np.tan(TA/2) ) - e*np.sqrt(1-e**2)*np.sin(TA)/(1+e*np.cos(TA))
    
    # Fix for Hyperbolic eq 3.36 and 3.37
    F = np.log( ( np.sqrt(e + 1) + np.sqrt(e-1)*np.tan(TA/2) )/( np.sqrt(e+1) - np.sqrt(e-1)*np.tan(TA/2) ) )
    Mh = e*np.sinh(F) - F
    
    hyp_mask = e>1. # Get array of booleans for hyperbolic objects
    M[hyp_mask] = Mh[hyp_mask]
    
    M = np.mod(M, 2*np.pi) # Wrap to [0,2*pi]
    
    return M

def TA_to_E(TA,e):
    ''' Convert True anomaly to Eccentric enomaly '''
    
    # Input checks. Convert floats to arrays
    if type(e) in [int,float,np.float64]:
        e = np.array([e]) # Convert to float to array
    if type(TA) in [int,float,np.float64]:
        TA = np.array([TA]) # Convert to float to array
    
    # Input cases
    if (len(e)==1) & (len(TA)>1):
        # Single orbit, mutliple epochs
        # Repeat e to match length of TA
        e = np.repeat(len(TA))
    elif (len(TA)==1) & (len(e)>1):
        # Single epoch, mutliple orbits
        # Repeat TA to match length of e
        TA = np.repeat(TA,len(e)) 
    
    # Ellipctical orbit
    # Eq 3.10b of Curtis
    E = 2*np.arctan( np.sqrt((1-e)/(1+e))*np.tan(TA/2) )
    
    # TODO: Fix for Hyperbolic
    # Eq 3.35 (hyperbolic anomaly)
    
    return E


def E_to_M(E,e):
    ''' Convert Ecentric anomaly to Mean anomaly '''
    
    # Input checks. Convert floats to arrays
    if type(e) in [int,float,np.float64]:
        e = np.array([e]) # Convert to float to array
    if type(E) in [int,float,np.float64]:
        E = np.array([E]) # Convert to float to array
    
    # Input cases
    if (len(e)==1) & (len(E)>1):
        # Single orbit, mutliple epochs
        # Repeat e to match length of E
        e = np.repeat(len(E))
    elif (len(E)==1) & (len(e)>1):
        # Single epoch, mutliple orbits
        # Repeat TA to match length of e
        E = np.repeat(E,len(e)) 
    
    
    # Ellipctical orbit
    # Eq 3.11 of Curtis
    M = E - e*np.sin(E)
    
    # TODO: Fix for Hyperbolic
    
    return M

# Backward conversions

def M_to_E(M,e):
    '''
    Convert mean anomaly M to eccentric anomaly E.
    
    This is the solution to Kepler's equation. This function is a wrapper to kepler.

    Parameters
    ----------
    e : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    E : TYPE
        DESCRIPTION.

    '''
    
    E = kepleq(e,w,om,M)
    
    return E


#%% State Vector from COEs ----------------------------------------------------

def sv_from_coe(a,e,i,om,w,M,mu=1.32712440018E11,units='AU',E=None):
    '''
    Compute the state vector from orbital elements.
    
    Two cases:
    1. Single object
       - a,e,i,om,w are constant
       - M is a vector containing mean anomalies at different epochs.
       - Use regular solution method with a single transfomration matrix.
     2. Multiple objects
       - a,e,i,om,w,M are vectors
       - use adapted solution method to handle the N different coordinate transforms.
    
    # Example 2. Solve for all asteroids in MPCORB
    # >> df = Datasets.load_dataset('MPCORB') # Load MPCORB data
    # >> a = np.array(df.a) # Load a vector
    # >> e = np.array(df.e) # Load e vector
    # >> i = np.deg2rad(np.array(df.i)) # Load i vector
    # >> om = np.deg2rad(np.array(df.om)) # Load om vector
    # >> w = np.deg2rad(np.array(df.w)) # Load w vector
    # >> M = np.deg2rad(np.array(df.M)) # Load M vector (rad)
    # >> sv = sv_from_coe(a,e,i,om,w,M)
    
    Parameters
    ----------
    coe : TYPE
        Classical orbital elements (a,e,i,om,w,M).
    mu : TYPE
        Gravitational parameter of central body.
        Default is GM_sun = 1.32712440018E11 # Sun gravitational parameter (km^3/s^2)

    Returns
    -------
    sv : 1D or 2D array
        State vectors.

    '''
    
    # Setup and argument checks
    
    # Convert float inputs to vectors
    float_flag = 0
    if type(a) in [int,float,np.float64]:
        a = np.array([a])
        float_flag = 1
    if type(e) in [int,float,np.float64]:
        e = np.array([e])
        float_flag = 1
    if type(i) in [int,float,np.float64]:
        i = np.array([i])
        float_flag = 1
    if type(om) in [int,float,np.float64]:
        om = np.array([om])
        float_flag = 1
    if type(w) in [int,float,np.float64]:
        w = np.array([w])
        float_flag = 1
    if type(M) in [int,float,np.float64]:
        M = np.array([M])
        float_flag = 1
    
    # 4. Convert a to km (if required)
    if units == 'AU':
        AU = 149597870.700 # Astronomical unit (km)
        a = a*AU
    
    # Single-object, single epoch case
    if (len(a) == 1) & (len(M) == 1):
        # Use spice.conics function (faster)
        rp = a*(1.-e) # Perifocal distance
        elts = [rp,e,i,om,w,M,0.,mu]
        sv = spice.conics(elts,0.)
        
        # # Continue using function
        # E = Kepler(e,M) # Compute Ecentric anomaly
        # E = np.array([E])
        
        return sv
    
    # Multiple objects/epochs case:
    
    # Check for single object/multiple epochs case
    if (len(a)==1) & (len(M) > 1):
        # Reset a,e,i,om,w to be vectors with equal values
        a = np.ones(len(M))*a[0]
        e = np.ones(len(M))*e[0]
        i = np.ones(len(M))*i[0]
        om = np.ones(len(M))*om[0]
        w = np.ones(len(M))*w[0]
    
    # 1. Compute Eccentric anomaly
    if E is None:
        E = Kepler(e,M) # Compute Ecentric anomaly
    # Get copy of Hyperbolic anomaly
    F = E.copy()
    
    # Find any hyperbolic objects
    hyp_mask = e>1.
    
    # 2. Compute tan(TA/2)
    tan_TA2 = np.nan*np.zeros(len(E)) # Initialize
    
    # (Eliptical orbits)
    # tan(TA/2) = sqrt((e+1)/(1-e))*tan(E/2) # Eq. 3.10a of Curtis
    tan_TA2[~hyp_mask] = np.sqrt((e[~hyp_mask]+1)/(1-e[~hyp_mask]))*np.tan(E[~hyp_mask]/2)
    
    # (Hyperbolic orbits)
    # tan(TA/2) = sqrt((e+1)/(e-1))*tanh(F/2) # Eq. 3.41b of Curtis
    tan_TA2[hyp_mask] = np.sqrt((e[hyp_mask]+1)/(e[hyp_mask]-1))*np.tanh(F[hyp_mask]/2)
    
    # 3. Compute True anomaly TA
    TA = np.arctan(tan_TA2)*2.
    
    # 3. Compute angular momentum h2 = h^2
    # (Elliptical or Hyperbolic orbits)
    h2 = mu*a* np.abs(1. - e**2)
    # FIXME: Case for parabolic orbits? 
    
    # 4. Compute state vectors in perifocal frame
    # Basis vectors p = periapsis
    #               q = apoapsis
    # (For all orbits) Eq. 4.37-4.38 of Curtis
    r = (h2/mu)*1./(1. + e*np.cos(TA)) # Radial distance (km)
    r_px = r*np.cos(TA) # x position (km)
    r_py = r*np.sin(TA) # y position (km)
    v_px = -(mu/np.sqrt(h2))*np.sin(TA) # vx velocity (km/s)
    v_py = (mu/np.sqrt(h2))*(e + np.cos(TA)) # vy velocity (km/s) 
    
    # 5. Convert Perifocal to Heliocentric coordinates
    # There are two cases for compting this, dependent on if the inputs represent:
    # A. the same object at different epochs; or
    # B. a population of different objects at a single epoch.
    
    # Case A: Single Object -------------------------------------------------------------------
    # This is the usual case, where the transformation from perifocal to Heliocentric coordinates
    # can be described by a single transformation matrix.
    # X = [Q]x 
    # where:
    # x is a 3xN matrix of the x,y,z perifocal coordinates at each epoch;
    # X is a 3xN matrix of the X,Y,Z Heliocentric coordinates at each epoch; and
    # Q is a 3x3 transformation matrix representing 3 rotations.
    #
    
    
    # Case B: Multiple Objects ----------------------------------------------------------------
    # The above method needs to be adapted, since each object will have a unique transformation
    # matrix.
    # We define Q1, Q2, ... Q9 as vectors of the components of the transform matrices of each object.
    # Then, the components of the Heliocentric state vector can be computed as vectors.
    # X = Q1*x + Q2*y + Q3*0
    # Y = Q4*x + Q5*y + Q6*0
    # Z = Q7*x + Q8*y + Q9*0
    # where
    # x,y,z are vectors of the position components in perifocal frame (z component = 0)
    # X,Y,Z are vectors of the position components in Heliocentric frame
    
    # if len(a) > 1:
        # Multiple objects
        
    # Compute components of transform matrices
    Q1 = np.cos(om)*np.cos(w) - np.sin(om)*np.sin(w)*np.cos(i)
    Q2 = -np.cos(om)*np.sin(w) - np.sin(om)*np.cos(i)*np.cos(w)
    
    Q4 = np.sin(om)*np.cos(w) + np.cos(om)*np.cos(i)*np.sin(w)
    Q5 = -np.sin(om)*np.sin(w) + np.cos(om)*np.cos(i)*np.cos(w)
    
    Q7 = np.sin(i)*np.sin(w)
    Q8 = np.sin(i)*np.cos(w)
    
    # Compute components of position vector in Heliocentric frame
    r_x = Q1*r_px + Q2*r_py
    r_y = Q4*r_px + Q5*r_py
    r_z = Q7*r_px + Q8*r_py
    
    # Compute components of velocity vector in Heliocentric frame
    v_x = Q1*v_px + Q2*v_py
    v_y = Q4*v_px + Q5*v_py
    v_z = Q7*v_px + Q8*v_py
    
    # Return state vector
    sv = np.column_stack((r_x,r_y,r_z,v_x,v_y,v_z))
    
    return sv

#%% COEs from State Vector ----------------------------------------------------

def coe_from_sv(R,V,mu=1.32712440018E11,units='km'):
    '''
    Compute the orbital elements from a state vector.
    Implements algorithm 4.1 of Curtis (2005).
    
    Two cases:
    1. Single object

    2. Multiple objects
    

    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    mu : TYPE, optional
        DESCRIPTION. The default is 1.32712440018E11.
    units : TYPE, optional
        DESCRIPTION. The default is 'AU'.

    Returns
    -------
    a,e,i,om,w,TA : Orbital elements

    '''
    
    # Format inputs
    float_flag = 0 # Indicate single objects
    if R.shape == (3,):
        # Single input. Convert to 1x3 vector
        R = R.reshape(1,3)
        V = V.reshape(1,3)
        float_flag = 1 # Indicate single objects
    
    eps = 1.0E-15 # Set small number
    
    # Convert a to km (if required)
    if units == 'AU':
        AU = 149597870.700 # Astronomical unit (km)
        R = R*AU
        V = V*AU # Convert AU/s to km/s
    
        
    # Magnitudes
    r = np.linalg.norm(R,axis=-1)
    v = np.linalg.norm(V,axis=-1)
    vr = np.einsum('ij,ij->i',R,V)/r # Radial velocity = np.dot(R,V)/r 
    
    
    # Angular momentum
    H = np.cross(R,V) # Vector
    h = np.linalg.norm(H,axis=-1) # Magnitude
    
    # Inclination (eqn 4.7)
    i = np.arccos(H[:,2]/h)

    # Line of nodes
    N = np.cross(np.tile(np.array([0,0,1]),(len(R),1)),H) # N = H x K
    n = np.linalg.norm(N,axis=-1) # Magnitude
    # Define default direction for circular orbits
    ind = n==0
    N[ind,:] = np.array([1,0,0])
    n[ind] = 1.
    
    # Ascending node om
    om = np.zeros(len(R)) # Initialize to zero
    om[n!=0] = np.arccos(N[:,0][n!=0]/n[n!=0])
    # Quadrant check
    # if N[1] < 0.: om = 2*np.pi - om # 180 < om < 360 deg
    om[(n!=0) & (N[:,1]<0)] = 2*np.pi - om[(n!=0) & (N[:,1]<0)]
    # if n != 0.:
    #     om = np.arccos(N[0]/n)
    #     # Quadrant check using j component of N
    #     if N[1] < 0.:
    #         # 180 < om < 360 deg
    #         om = 2*np.pi - om
    # else:
    #     om = 0.
    
    # Eccentricity vector
    ecc = ( (v**2 - mu/r)[:, np.newaxis]*R - (r*vr)[:, np.newaxis]*V )/mu # Vector
    e = np.linalg.norm(ecc,axis=-1) # Eccentricity
    # ecc = (1/mu)*( (v**2 - mu/r)*R - r*vr*V ) # Vector
    # e = np.linalg.norm(ecc) # Eccentricity
    
    # Argument of periapsis
    w = np.zeros(len(R)) # Initialize to zero
    ind = e > eps
    w[ind] = np.arccos(np.einsum('ij,ij->i',N[ind,:],ecc[ind,:])/(n[ind]*e[ind])) 
    # Quadrant check
    # ecc_z = ecc[:,2] # z component of ecc vector
    w[(e>eps) & (ecc[:,2]<0)] = 2*np.pi - om[(e>eps) & (ecc[:,2]<0)]
    # if n != 0.:
    #     if e > eps:
    #         w = np.arccos(np.dot(N,ecc)/(n*e))
    #         # Quadrant check using k component of ecc vector
    #         if ecc[2] < 0.:
    #             # 180 < w < 360 deg
    #             w = 2*np.pi - w
    #     else:
    #         w = 0.
    # else:
    #     w = 0.
    
    # True anomaly
    TA = np.zeros(len(R))*np.nan # Initialize
    # Elliptical orbits
    ind = e>eps
    TA[ind] = np.arccos(np.einsum('ij,ij->i',ecc[ind,:],R[ind,:])/(e[ind]*r[ind])) 
    TA[(e>eps) & (vr<0)] = 2*np.pi - TA[(e>eps) & (vr<0)] # Quadrant check 180 < TA < 360 deg
    
    # Circular orbits
    # Find angle from line of nodes to R
    cp = np.cross(N,R)
    ind = e<=eps
    TA[ind] = np.arccos(np.einsum('ij,ij->i',N[ind,:],R[ind,:])/(n[ind]*r[ind]))
    TA[(e<=eps) & (cp[:,2]<0)] = 2*np.pi - TA[(e<=eps) & (cp[:,2]<0)]
    # if e > eps:
    #     TA = np.arccos(np.dot(ecc,R)/(e*r))
    #     # Quadrant check using radial velocity
    #     if vr < 0:
    #         # 180 < TA < 360 deg
    #         TA = 2*np.pi - TA
    # else:
    #     # Circular orbit
    #     # Find angle from line of nodes to R
    #     cp = np.cross(N,R)
    #     if cp[2] >= 0.:
    #         TA = np.arccos(np.dot(N,R)/(n*r))
    #     else:
    #         TA = 2*np.pi - np.arccos(np.dot(N,R)/(n*r))
    
    # Semi-major axis (a<0 for hyperbola)
    E = v**2/2 - mu/r
    a = np.zeros(len(R)) # Initialize
    a[E<0] = h[E<0]**2/(mu*(1-e[E<0]**2))
    # if E==0.:
    #     # Parabola
    #     a == 0.
    # else:
    #     a = h**2/(mu*(1-e**2))
    
    
    # Convert single length values to floats
    if len(a)==1:
        a = float(a)
        e = float(e)
        i = float(i)
        om = float(om)
        w = float(w)
        TA = float(TA)
    
    return a,e,i,om,w,TA

#%% Hyperbolic Asymptotes

def asymptote_to_ra_dec(S):
    '''
    Compute declination (delta) and right ascension (alpha) of launch asymptote
    
          | cos(delta)*cos(alpha) |
      S = | cos(delta)*sin(alpha) |
          |       sin(delta)      |
    
    S = Outgoing asymptote (unit vector)
    delta = DLA = Declination of Launch asymptote (angle above/below plane)
    alpha = RLA = Right Ascension of Launch Asymptote (angle from vernal equinox)
    see: https://www.yumpu.com/en/document/read/45293841/calculating-c3-rla-and-dla


    Parameters
    ----------
    S : TYPE
        Hyperbolic asymptote (unit vector)

    Returns
    -------
    ra : TYPE
        Right ascension of launch asymptote (rad)
    dec : TYPE
        Declination of launch asymptote (rad)

    '''

    
    # TODO: Vectorize
    
    if S.shape == (3,):
        # Single value. S = 1x3 vector
        
        # Normalize vector
        S = S/np.linalg.norm(S)
        
        # Compute declination (delta) and right ascension (alpha) of launch asymptote
        ra = np.arctan2(S[1], S[0])  # atan(Sy/Sx)
        dec = np.arcsin(S[2])
        
    else:
        # Multiple values S = Nx3 vector
        
        # Normalize vector
        S = S/np.linalg.norm(S, axis=-1)[:, np.newaxis]
        
        # Compute declination (delta) and right ascension (alpha) of launch asymptote
        ra = np.arctan2(S[0,:], S[1,:])
        dec = np.arcsin(S[2,:])
    
    return ra, dec

def ra_dec_to_asymptote(ra,dec,units='rad'):
    '''
    Compute declination (delta) and right ascension (alpha) of launch asymptote
    
          | cos(delta)*cos(alpha) |
      S = | cos(delta)*sin(alpha) |
          |       sin(delta)      |

    Parameters
    ----------
    ra : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.

    '''
    
    # Convert units to radians
    if units.lower() == 'deg':
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)
    
    # Convert float inputs to vectors
    if (type(ra) == float) or (type(ra) == int) or (type(ra) == np.float64):
        # Single input. Output 3x1 vector.
        S = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec) ])
    else:
        # Multiple inputs. Output Nx3 vector
        Sx = np.cos(dec)*np.cos(ra)
        Sy = np.cos(dec)*np.sin(ra)
        Sz = np.sin(dec)
        S = np.column_stack((Sx,Sy,Sz))
    
    return S


#%% ###########################################################################
#                            Orbital Parameters
# #############################################################################

def compute_Tisserand(a,e,i,ref_planet='Jupiter'):
    '''
    Compute the Tisserand parameter from Orbital Elements.

    Parameters
    ----------
    a : TYPE
        Semimajor axis (AU)
    e : TYPE
        Eccentricity
    i : float
        Incliantion (deg)
    ref_planet : TYPE, optional
        DESCRIPTION. The default is 'Jupiter'.

    Returns
    -------
    Tp : TYPE
        DESCRIPTION.

    '''
    # Get semimajor axis of reference planet
    if ref_planet == 'Jupiter':
        ap = 5.2
    elif ref_planet == 'Earth':
        ap = 1.
    else:
        # TODO: Add other planets.
        raise ValueError('Tisserand parameter only impplemented relative to Jupiter or Earth.')
    
    # Compute the Tisserand parameter
    Tp = ap/a + np.sqrt( (a/ap)*((1.-e)**2) )*np.cos(np.deg2rad(i))
    
    return Tp
