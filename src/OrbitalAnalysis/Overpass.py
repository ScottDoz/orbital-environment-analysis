# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:40:50 2022

@author: scott

Overpass Module
-------------------
Functions dealing with the setup of the scenario.

- configuring and running GMAT scripts
- analyse statistics of the overpasses

Note: the NAIF of the spacecraft in the output ephemeris file is
NAIF = -10002001

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import pdb

from astropy.time import Time
import spiceypy as spice

# Module imports
from SatelliteData import query_norad
from utils import get_data_home
from Ephem import *
from Events import *
from Visualization import plot_time_windows
from GmatScenario import *


#%% Main Analysis Workflow

# 1. Inputs
#    - define Satellite COEs, and groundstation locations
#
# 2. Generate Spice Files
#    - configure and run GMAT script to generate required files
#    - alternatively, generate files using SPICE
#
# 3. Compute Lighting and Access
#    - satellite lighting, groundstation lighting
#    - line-of-sight access, visible access




#%% Scenario coverage and time vectors

def generate_et_vectors_from_GMAT_coverage(step, exclude_ends=False):
    '''
    Generate a vector of epochs (ET) with start and stop dates from the GMAT
    scenario

    Parameters
    ----------
    step : float
        Step size in seconds
    
    exclude_ends : Bool
        Flag to exclude the srart and stop times.
        Sometimes problems with interpolating ephemeris at ends.

    Returns
    -------
    et : 1xN numpy array.
        Vector of ephemeris times

    '''
    
    # Read the GMAT script to get coverage
    cov = get_GMAT_coverage()
    
    # Create time vector
    et = np.arange(cov['start_et'],cov['stop_et'],step)
    
    # Clip ends
    if exclude_ends == True:
        et = et[1:-1]
    
    return et


#%% Create Files

def create_files(sat,start_date,stop_date,step,method):
    
    # Main function to setup 
    
    # Convert start and stop dates to Ephemeris Time
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    
    
    # Create Satellite SPK file
    print('\nCreating Satellite SPK Files', flush=True)
    print('----------------------------')
    create_satellite_ephem(sat,start_et,stop_et,step,method=method)
    
    return



#%% Optical Analysis Workflow

def optical_analysis(start_date,stop_date,step, DATA_DIR=None):
    ''' 
    Main function to compute all lighting and access intervals for the defined
    scenario. Also compute metrics for the optical trackability of the satellite.
    '''
    
    print('\nRunning Optical Analysis', flush=True)
    print('-------------------------')
    
    # Convert start and stop dates to Ephemeris Time
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    
    # Generate ephemeris times covering GMAT scenario
    # step = 10.
    scenario_duration = stop_et - start_et # Length of scenario (s)
    # Create confinement window for scenario
    cnfine = spice.cell_double(2*100) # Initialize window of interest
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
    
    # Get list of stations
    # stations = ['DSS-43','DSS-14','DSS-63']
    stations = ['SSR-1','SSR-2','SSR-3']
    colors = ['black','black','black']
    # DSS-43 : Canberra 70m Dish
    # DSS-14 : Goldstone 70m
    # DSS-63 : Madrid 70m
    
    # Compute satellite lighting intervals
    print('Computing Satellite Lighting intervals', flush=True)
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # Loop through ground stations and compute lighting and access intervals
    access_list = [] # List of visible access interval of stations
    num_access_list = [] # List of numbers of access
    total_access_list = [] # List of total access durations
    shortest_access_list = [] # List of shortest access durations
    longest_access_list = [] # List of longest access durations
    avg_pass_list = [] # List of average pass durations
    avg_coverage_list = [] # List of average coverage fractions
    num_gaps_list = [] # List of number of gaps in access
    avg_interval1_list = [] # List of average interval durations (definition 1)
    avg_interval2_list = [] # List of agerage interval durations (definition 2)
    
    print('Computing Station Lighting and Access intervals', flush=True)
    print('Stations: {}'.format(stations),flush=True)
    for gs in tqdm(stations):
        
        # Compute station lighting intervals
        gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
    
        # Compute line-of-sight access intervals
        los_access = find_access(start_et,stop_et,station=gs)
        
        # Compute visible (constrained) access intervals
        access = constrain_access_by_lighting(los_access,gslight,satdark)
        
        # Compute non-access intervals (complement of access periods)
        gap = spice.wncomd(start_et,stop_et,access)
        
        # Compute access metrics (avg_pass, avg_coverage, avg_interval)
        dfvis = window_to_dataframe(access,timefmt='ET') # Load as dataframe
        num_access = len(dfvis) # Number of access intervals
        total_access = dfvis.Duration.sum() # Total duration of access (s)
        shortest_access = dfvis.Duration.min() # Shortest access duration (s)
        longest_access = dfvis.Duration.max() # Longest access duration (s)
        avg_pass = total_access / num_access # Average duration of access intervals (s)
        avg_coverage = total_access / scenario_duration # Fraction of coverage
        # Compute interval metrics avg_interval
        dfgap = window_to_dataframe(gap,timefmt='ET') # Load as dataframe
        num_gaps = len(dfgap)   # Number of gaps
        avg_interval1 = ( scenario_duration - total_access) / num_access # Definition 1
        avg_interval2 = dfgap.Duration.sum() / num_gaps # Definition 1

        # Append results to lists
        access_list.append(access) # Append to list of station access intervals
        num_access_list.append(num_access)
        total_access_list.append(total_access)
        shortest_access_list.append(shortest_access)
        longest_access_list.append(longest_access)
        avg_pass_list.append(avg_pass)
        avg_coverage_list.append(avg_coverage)
        num_gaps_list.append(num_gaps)
        avg_interval1_list.append(avg_interval1)
        avg_interval2_list.append(avg_interval2)
        
    # *** Should metrics be computed each station or on combined??? ***
    
    # Compute combined access intervals
    combinedaccess = access_list[0] # First station
    if len(access_list)>1:
        for win in access_list[1:]:
            combinedaccess = spice.wnunid(combinedaccess,win) # Union of intervals
    
    # Generate results for each station
    results = pd.DataFrame(columns=['Station',
                                    'num_access','total_access','shortest_access','longest_access','avg_pass',
                                    'avg_coverage','num_gaps','avg_interval1','avg_interval2'])
    results['Station'] = stations
    results['num_access'] = num_access_list
    results['total_access'] = total_access_list
    results['shortest_access'] = shortest_access_list
    results['longest_access'] = longest_access_list
    results['avg_pass'] = avg_pass_list
    results['avg_coverage'] = avg_coverage_list
    results['num_gaps'] = num_gaps_list
    results['avg_interval1'] = avg_interval1_list
    results['avg_interval2'] = avg_interval2_list
    
    # Exchange rows/columns
    results = results.T
    # results.columns = results.iloc[0].to_list()
    print('')
    print(results)
    
    # Plot the access periods
    plot_time_windows(access_list+[combinedaccess],stations+['Combined'],stations+['Combined'],colors+['red'])
    
    return results




#%% Notes on access

# Access in STK uses lighting constaints on the groundstation
# In DT.py, the ligthing condition is set at value '3'.
# In Riley's thesis, it is listed as 'ePenumbraorUmbra'.
# STK reference on constraints (link below) lists lighting condition option
# of Penumbra or Umbra: Partial sunlight of total shadow.
# See: https://help.agi.com/stk/11.0.1/Content/stk/constraints-02.htm

# In GMAT, there is no option to add constraints to access.
# To replicate:
# 1. Compute Access times from gs-Sat
# 2. Compute lighting conditions at station (using sun elevation)
# 3. Apply these constraints to the access times.




