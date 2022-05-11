# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:40:50 2022

@author: scott

Overpass Module
-------------------
Functions dealing with the setup of the scenario.

- analyse statistics of the overpasses

Note: the NAIF of the spacecraft in the output ephemeris file is
NAIF = -10002001

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from astropy.time import Time
import spiceypy as spice

from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, freeze_support, RLock

import pdb

# Module imports
from SatelliteData import query_norad
from utils import get_data_home
from Ephem import *
from Events import *
from Visualization import plot_time_windows
from GroundstationData import get_groundstations
# from GmatScenario import *


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
#    - save each set as csv file


#%% Main Workflow

def run_analysis(sat_dict,start_date,stop_date,step):
    ''' Main workflow for DIT score calculation. '''
    
    # Convert start and stop dates to Ephemeris Time
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)
    
    # Define a working folder to output the data
    out_dir = get_data_home()/'DITdata'
    # Check if directory exists and create
    if not os.path.exists(str(out_dir)):
        os.makedirs(str(out_dir))

    
    
    # 0. Check all generic kernels exist
    check_generic_kernels()
    
    # # 1. Create Ephemeris files
    # # - Satellite Ephemeris file (sat.bsp)
    # # - SSR stations
    # # - SSRD stations
    # create_ephem_files(sat_dict,start_date,stop_date,step,method='two-body') # From elements
    # # create_ephem_files(NORAD,start_date,stop_date,step,method='tle') # From TLE
    
    # 2. Compute satellite lighting intervals
    print('\nComputing Satellite Lighting intervals', flush=True)
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # 3. Compute SSR and SSRD Station Lighting and access
    dflos_ssr, dfvis_ssr = compute_station_access('SSR',start_et,stop_et,satdark,save=True)
    dflos_ssrd, dfvis_ssrd = compute_station_access('SSRD',start_et,stop_et,satdark,save=True)
    
    # 4. Optical Trackability
    # Compute from combined list of visible access dfvis_ssr
    # (Uses 49 SSR stations)
    optical_results = compute_tracking_stats(dfvis_ssr,scenario_duration)
    optical_score = optical_results['Score'].mean()
    
    # 5. Radar Trackability
    # Compute from combined list of line-of-sight access dflos_ssr
    # (Uses 49 SSR stations)
    radar_results = compute_tracking_stats(dflos_ssr,scenario_duration)
    radar_score = radar_results['Score'].mean() # Total radar score

    # 6. Optical Detectability
    compute_optical_detectability(dfvis_ssr)

    # Save results to dict ---------------------------------
    results = {'SSRD_los':dflos_ssrd, 'SSRD_vis':dfvis_ssrd,
               'SSR_los':dflos_ssr, 'SSR_vis':dfvis_ssr,
               'radar_results':radar_results,'optical_results':optical_results,
               'radar_score':radar_score,'optical_score':optical_score}
    
    
    # Print Trackability Results
    print('\nRadar Trackability')
    print(radar_results)
    print("\nOverall T Radar Score: {}\n".format(radar_score))
    
    print('\nOptical Trackability')
    print(optical_results)
    print("\nOverall T Optical Score: {}\n".format(optical_score))
    
    
    
    return results



#%% Subroutines

def check_generic_kernels():
    ''' Check to ensure all generic kernels needed for analysis are present. '''
    
    # Get kernel directory
    kernel_dir = get_data_home() / 'Kernels'
    
    # Check Leap second kernel
    filename = kernel_dir/'naif0012.tls'
    if filename.exists() == False:
        print("Missing Leap Second Kernel")
        SPICEKernels.download_lsk()
    
    # Check Planetary Constants Kernel
    filename = kernel_dir/'pck00010.tpc'
    if filename.exists() == False:
        print("Missing Planetary Constants Kernel")
        SPICEKernels.download_pck()
    
    # DE440 Solar System Ephemeris
    filename = kernel_dir/'de440s.bsp'
    if filename.exists() == False:
        print("Missing DE440s Solar System Ephemeris")
        SPICEKernels.download_planet_spk()
    
    # Earth binary PCK (Jan 2000 - Jun 2022)
    filename = kernel_dir/'earth_000101_220616_220323.bpc'
    if filename.exists() == False:
        print("Missing Earth Binary PCK")
        SPICEKernels.download_earth_binary_pck()
    
    # Earth topocentric frame text kernel
    filename = kernel_dir/'earth_topo_201023.tf'
    if filename.exists() == False:
        print("Missing Earth topocentric frame text kernel")
        SPICEKernels.download_earth_topo_tf()
    
    # Geophysical constants kernel
    filename = kernel_dir/'geophysical.ker'
    if filename.exists() == False:
        print("Missing Geophysical constants kernel")
        SPICEKernels.download_geophysical()
    
    return


def create_ephem_files(sat,start_date,stop_date,step,method):
    ''' Create ephemeris files for satellite and groundstation networks '''
    
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
    
    
    # Create Groundstation files
    # 2 separate networks
    # SSR - 49 stations used for tracking analysis
    # SSRD - 7 stations used for detectability analysis
    
    # SSR
    print('\nCreating SPK Files for SSR Groundstations', flush=True)
    print('------------------------------------------')
    df = get_groundstations(network='SSR') # Get details of network
    create_station_ephem(df, network_name='SSR') # Write SPK
    
    # SSRD
    print('\nCreating SPK Files for SSRD Groundstations', flush=True)
    print('------------------------------------------')
    df = get_groundstations(network='SSRD') # Get details of network
    create_station_ephem(df, network_name='SSRD') # Write SPK
    
    
    return

def compute_station_access(network,start_et,stop_et,satdark,save=False):
    
    # Loop through all stations and 
    
    # Define station names
    if network == 'SSR':
        # SSR Network contains 49 stations
        stations = ['SSR-'+str(i+1) for i in range(49)]
        stations = stations[:4] # FIXME: Shortened for testing.
    elif network == 'SSRD':
        # SSRD Network contains 7 stations
        stations = ['SSRD-'+str(i+1) for i in range(7)]
        
    # Define a working folder to output the data
    out_dir = get_data_home()/'DITdata'
    
    
    
    # Loop through stations
    # TODO: in paralellize this function
    
    print('Computing {} Station Lighting and Access intervals'.format(network), flush=True)
    print('Stations: {}'.format(stations),flush=True)
    dflos = pd.DataFrame(columns=['Station','ID','Start','Stop','Duration'])
    dfvis = pd.DataFrame(columns=['Station','ID','Start','Stop','Duration'])
    for gs in tqdm(stations):
        
        # Compute station lighting intervals
        # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
        gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-0.25)
    
        # Compute line-of-sight access intervals
        los_access = find_access(start_et,stop_et,station=gs)
        # Convert to dataframe
        dflos_i = window_to_dataframe(los_access,timefmt='ET') # Load as dataframe
        dflos_i.insert(0,'ID',dflos_i.index)
        dflos_i.insert(0,'Station',gs)
        dflos = dflos.append(dflos_i) # Append to global dataframe
        
        # Compute visible (constrained) access intervals
        access = constrain_access_by_lighting(los_access,gslight,satdark)
        # Convert to dataframe
        dfvis_i = window_to_dataframe(access,timefmt='ET') # Load as dataframe
        dfvis_i.insert(0,'ID',dfvis_i.index)
        dfvis_i.insert(0,'Station',gs)
        dfvis = dfvis.append(dfvis_i)
        
        # Compute non-access intervals (complement of access periods)
        gap = spice.wncomd(start_et,stop_et,access)
    
        # Save data to file
        if save==True:
            
            # Line-of-sight access 
            
            # Copy start and stop et to new columns
            dflos_i.insert(2,'Start_et',dflos_i.Start)
            dflos_i.insert(3,'Stop_et',dflos_i.Stop)
            # Convert Start times to calendar
            dt = spice.et2datetime(dflos_i.Start) # Datetime
            t = Time(dt, format='datetime', scale='utc') # Astropy Time object
            t_iso = t.iso # Times in iso
            dflos_i['Start'] = pd.to_datetime(t_iso)
            # Convert Stop times to calendar
            dt = spice.et2datetime(dflos_i.Stop) # Datetime
            t = Time(dt, format='datetime', scale='utc') # Astropy Time object
            t_iso = t.iso # Times in iso
            dflos_i['Stop'] = pd.to_datetime(t_iso)
            # Save access data to file
            filename = gs + "_los_access_intervals.csv"
            dflos_i.to_csv(str(out_dir/filename),index=False)
            
            
            # Visible access 
            
            # Copy start and stop et to new columns
            dfvis_i.insert(2,'Start_et',dfvis_i.Start)
            dfvis_i.insert(3,'Stop_et',dfvis_i.Stop)
            # Convert Start times to calendar
            dt = spice.et2datetime(dfvis_i.Start) # Datetime
            t = Time(dt, format='datetime', scale='utc') # Astropy Time object
            t_iso = t.iso # Times in iso
            dfvis_i['Start'] = pd.to_datetime(t_iso)
            # Convert Stop times to calendar
            dt = spice.et2datetime(dfvis_i.Stop) # Datetime
            t = Time(dt, format='datetime', scale='utc') # Astropy Time object
            t_iso = t.iso # Times in iso
            dfvis_i['Stop'] = pd.to_datetime(t_iso)
            # Save access data to file
            filename = gs + "_vis_access_intervals.csv"
            dfvis_i.to_csv(str(out_dir/filename),index=False)
        
    
    return dflos, dfvis

def compute_optical_detectability(dfvis_ssr):
    
    # Find station with largest total access.
    dfgroup = dfvis_ssr[['Station','Duration']].groupby(['Station']).sum()
    best_station = dfgroup['Duration'].idxmax()
    
    # Extract access for that station
    df = dfvis_ssr[dfvis_ssr['Station'] == best_station]
    
    # Create time vector sampling all access periods
    step = 10 # Timestep (s)
    et = [] # Empty array
    for ind,row in df.iterrows():
        et_new = np.arange(row['Start']-2*step,row['Stop']+2*step,step)
        et += list(et_new)
    et = np.array(et) # Convert to numpy array
    et = np.sort(np.unique(et))  # Sort array and remove duplicates
    
    # Get Topocentric ephemeris relative to this station at these times
    dftopo = get_ephem_TOPO(et,groundstations=[best_station])
    dftopo = dftopo[0] # Select first station
    
    # Compute visual magnitude
    Rsat = 1 # Radius of satellite (m)
    msat = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12) # With airmass
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,include_airmass=False) # Without airmass
    
    # Constraints
    cutoff_mag = 15. # Maximum magnitude for visibility
    
    # Compute contrained stats
    max_mag = np.nanmax(msat[msat<=cutoff_mag])  # Maximum (dimest) magnitude
    min_mag = np.nanmin(msat[msat<=cutoff_mag])  # Minimum (brightest) magnitude 
    avg_mag = np.nanmean(msat[msat<=cutoff_mag]) # Mean magnitude
    
    
    # Generate plots
    import matplotlib.pyplot as plt
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Visual Magnitude')
    # Magnitude vs elevation
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat,'.b')
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat2,'.k')
    ax1.plot([0,90],[max_mag,max_mag],'-r')
    ax1.set_xlabel("Elevation (deg)")
    ax1.set_ylabel("Visual Magnitude (mag)")
    ax1.invert_yaxis() # Invert y axis
    # Az/El
    ax2.plot(dftopo['ET'],msat,'-b')
    ax2.plot(dftopo['ET'],msat2,'-k')
    # Max/min/mean
    ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[max_mag,max_mag],'-r')
    ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[min_mag,min_mag],'-r')
    ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[avg_mag,avg_mag],'-r')
    ax2.invert_yaxis() # Invert y axis
    ax2.set_xlabel("Epoch (ET)")
    ax2.set_ylabel("Visual Magnitude (mag)")
    fig.show()
    
    
    return

#%% Optical Analysis Workflow

def trackability_analysis(start_date,stop_date,step, DATA_DIR=None):
    ''' 
    Main function to compute all lighting and access intervals for the defined
    scenario. Also compute metrics for the optical and radar trackability of the 
    satellite.
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
    # stations = ['SSR-1','SSR-2','SSR-3']
    # stations = ['SSR-1','SSR-2','SSR-3','SSR-4','SSR-5','SSR-6','SSR-7']
    stations = ['SSR-'+str(i+1) for i in range(49)]
    colors = ['black']*len(stations)
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
    dflos_global = pd.DataFrame(columns=['Station','ID','Start','Stop','Duration'])
    dfvis_global = pd.DataFrame(columns=['Station','ID','Start','Stop','Duration'])
    
    print('Computing Station Lighting and Access intervals', flush=True)
    print('Stations: {}'.format(stations),flush=True)
    for gs in tqdm(stations):
        
        # Compute station lighting intervals
        # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
        gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-0.25)
    
        # Compute line-of-sight access intervals
        los_access = find_access(start_et,stop_et,station=gs)
        # Convert to dataframe
        dflos = window_to_dataframe(los_access,timefmt='ET') # Load as dataframe
        dflos.insert(0,'ID',dflos.index)
        dflos.insert(0,'Station',gs)
        dflos_global = dflos_global.append(dflos)
        
        # Compute visible (constrained) access intervals
        access = constrain_access_by_lighting(los_access,gslight,satdark)
        # Convert to dataframe
        dfvis = window_to_dataframe(access,timefmt='ET') # Load as dataframe
        dfvis.insert(0,'ID',dfvis.index)
        dfvis.insert(0,'Station',gs)
        dfvis_global = dfvis_global.append(dfvis)
        
        # Compute non-access intervals (complement of access periods)
        gap = spice.wncomd(start_et,stop_et,access)
        
        # Compute access metrics for each (avg_pass, avg_coverage, avg_interval)
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
        
    
    
    # Compute combined access intervals
    combinedaccess = access_list[0] # First station
    if len(access_list)>1:
        for win in access_list[1:]:
            combinedaccess = spice.wnunid(combinedaccess,win) # Union of intervals
    
    # Optical Trackability
    # Compute from combined list of visible access dfvis_global
    optical_results = compute_tracking_stats(dfvis_global,scenario_duration)
    optical_score = optical_results['Score'].mean()
    
    # Radar Trackability
    # Compute from combined list of line-of-sight access dflos_global
    radar_results = compute_tracking_stats(dflos_global,scenario_duration)
    radar_score = radar_results['Score'].mean() # Total radar score
    
    # Generate results for each station
    station_results = pd.DataFrame(columns=['Station',
                                    'num_access','total_access','shortest_access','longest_access','avg_pass',
                                    'avg_coverage','num_gaps','avg_interval1','avg_interval2'])
    station_results['Station'] = stations
    station_results['num_access'] = num_access_list
    station_results['total_access'] = total_access_list
    station_results['shortest_access'] = shortest_access_list
    station_results['longest_access'] = longest_access_list
    station_results['avg_pass'] = avg_pass_list
    station_results['avg_coverage'] = avg_coverage_list
    station_results['num_gaps'] = num_gaps_list
    station_results['avg_interval1'] = avg_interval1_list
    station_results['avg_interval2'] = avg_interval2_list
    
    # Exchange rows/columns
    station_results = station_results.T
    # station_results.columns = station_results.iloc[0].to_list()
    # print('')
    # print(station_results)
    
    # Plot the access periods
    plot_time_windows(access_list+[combinedaccess],stations+['Combined'],stations+['Combined'],colors+['red'])
    
    # Print radar results
    
    
    # Print Trackability Results
    print('\nRadar Trackability')
    print(radar_results)
    print("\nOverall T Radar Score: {}\n".format(radar_score))
    
    print('\nOptical Trackability')
    print(optical_results)
    print("\nOverall T Optical Score: {}\n".format(optical_score))
    
    
    return optical_results, radar_results, station_results

def compute_tracking_stats(df,scenario_duration):
    ''' 
    Compute the tracking statistics of access periods - either line-of-sight
    or visible. Return a dataframe sumarizing average pass duration, average
    coverage, and average interval, along with tiered scores for each.
    '''
    
    # Compute stats
    num_access = len(df) # Total number of access intervals
    total_access = df.Duration.sum() # Total duration of access (s)
    shortest_access = df.Duration.min() # Shortest access duration (s)
    longest_access = df.Duration.max() # Longest access duration (s)
    avg_pass = total_access / num_access # Average duration of access intervals (s)
    avg_coverage = total_access / scenario_duration # Fraction of coverage
    avg_interval = ( scenario_duration - total_access) / num_access # Definition 1
    
    # Average pass score
    if avg_pass < 120:
        pass_tier = 'Difficult to Track'
        pass_score = 0
    elif 120 <= avg_pass < 180:
        pass_tier = "Trackable"
        pass_score = 0.25
    elif 180 <= avg_pass < 400:
        pass_tier = 'More Trackable'
        pass_score = 0.5
    elif 400 <= avg_pass:
        pass_tier = 'Very Trackable'
        pass_score = 1.0
    
    # Coverage score
    if avg_coverage < 0.1:
        cover_tier = 'Difficult to Track'
        cover_score = 0
    elif 0.1 <= avg_coverage < .25:
        cover_tier = 'Trackable'
        cover_score = 0.25
    elif .25 <= avg_coverage < .60:
        cover_tier = 'More Trackable'
        cover_score = 0.5
    elif .60 < avg_coverage:
        cover_tier = 'Very Trackable'
        cover_score = 1.0
    
    # Interval score
    if avg_interval > 43200:
        int_tier = 'Difficult to Track'
        int_score = 0
    elif 43200 <= avg_interval < 14400:
        int_tier = 'Trackable'
        int_score = 0.25
    elif 14400 >= avg_interval:
        int_tier = 'More Trackable'
        int_score = 0.5
    
    # Create Results Dataframe
    results = pd.DataFrame(columns=['Metric','Value','Tier','Score'])
    pass_row = {'Metric': ' Avg Pass (s)', 'Value': avg_pass, 'Tier': pass_tier, 'Score': pass_score}
    results = results.append(pass_row, ignore_index=True)
    cover_row = {'Metric': ' Avg Coverage', 'Value': avg_coverage, 'Tier': cover_tier, 'Score': cover_score}
    results = results.append(cover_row, ignore_index=True)
    int_row = {'Metric': ' Avg Interval (s)', 'Value': avg_interval, 'Tier': int_tier, 'Score': int_score}
    results = results.append(int_row, ignore_index=True)
    
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




