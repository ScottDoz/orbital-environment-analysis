# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:38:09 2022

@author: scott

Epoch Module
------------

Functions for dealing with times.
Create custom lists of epochs.

"""

import numpy as np
import spiceypy as spice

#%% Create epoch lists --------------------------------------------------------

def et_from_date_range(start_date,stop_date,step):
    '''
    Generate a vector of ephemeris times between a start and stop date with a
    fixed step size.

    Parameters
    ----------
    start_date : str
        Start date e.g. '2020-10-26 16:00:00.000'.
    stop_date : str
         Stop Date e.g. '2020-11-25 15:59:59.999'.
    step : float
        Step size (s).

    Returns
    -------
    et : 1xN array
        Array of ephemeris times.

    '''
    
    # Load LSK for time conversions
    from utils import get_data_home
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    
    # Convert start and stop dates to ephemeris time
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)    
    
    # Generate ephemeris times
    et = np.arange(start_et,stop_et,step); et = np.append(et,stop_et)
    
    return et

def et_from_access_df(dfa,step):
    '''
    Generate a vector of ephemeris times based on access periods with a set timestep.

    Parameters
    ----------
    dfa : TYPE
        Dataframe containing start and stop et of each access period.
    step : TYPE
        Step size (s).

    Returns
    -------
    et : 1xN array
        Array of ephemeris times.

    '''
    et = np.array([]) # Instantiate empty array
    
    # Loop through each access period and generate separate vector of times.
    for i,row in dfa.iterrows():
        eti = np.arange(row['Start'],row['Stop'],step); eti = np.append(eti,row['Stop'])
        et = np.concatenate((et, eti)) 
    
    
    return et

