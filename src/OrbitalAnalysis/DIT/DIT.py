# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:40:50 2022

@author: scott

DIT Module
----------

Main workflow for the DIT analysis
- Main function: run_analysis
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
import time

# Module imports
from OrbitalAnalysis.SatelliteData import query_norad
from OrbitalAnalysis.utils import get_data_home
from OrbitalAnalysis.DIT.Ephem import *
from OrbitalAnalysis.DIT.Events import *
from OrbitalAnalysis.Visualization import plot_time_windows, plot_visibility, plot_overpass_skyplot, plot_linkbudget
from OrbitalAnalysis.DIT.GroundstationData import get_groundstations
from OrbitalAnalysis.DIT.Communications import *
from OrbitalAnalysis.DIT.VisualMagnitude import *
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

def run_analysis(sat_dict,start_date,stop_date,step,save_folder='DITdata',prefilter='crossings'):
    ''' Main workflow for DIT score calculation. '''
    
    # 0. Check all generic kernels exist
    SPICEKernels.check_generic_kernels()
    
    # Convert start and stop dates to Ephemeris Time
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)
    
    # Define a working folder to output the data
    out_dir = get_data_home()/save_folder
    # Check if directory exists and create
    if not os.path.exists(str(out_dir)):
        os.makedirs(str(out_dir))
    
    
    # 1. Create Ephemeris files
    # - Satellite Ephemeris file (sat.bsp)
    # - SSR stations
    # - SSRD stations
    create_ephem_files(sat_dict,start_date,stop_date,step,method='two-body') # From elements
    # create_ephem_files(NORAD,start_date,stop_date,step,method='tle') # From TLE
    
    # 2. Compute satellite lighting intervals
    print('\nComputing Satellite Lighting intervals', flush=True)
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    satlight1 = spice.wnunid(satlight,satpartial) # Full or partial light
    
    # 3. Compute SSR and SSRD Station Lighting and access
    # SSR: use min_el = 30 deg (120 deg cone angle from zenith)
    # SSRD: use min_el = 5 deg (since targeted at satellite)
    dflos_ssr, dfvis_ssr, dfcomblos_ssr, dfcombvis_ssr = compute_station_access('SSR',start_et,stop_et,satlight1,30.,prefilter=prefilter,save_folder=save_folder,save=True)
    dflos_ssrd, dfvis_ssrd, dfcomblos_ssrd, dfcombvis_ssrd = compute_station_access('SSRD',start_et,stop_et,satlight1,5.,prefilter=prefilter,save_folder=save_folder,save=True)
    
    # 4. Optical Trackability
    # Compute from combined list of visible access dfvis_ssr
    # (Uses 49 SSR stations)
    optical_results = compute_tracking_stats(dfcombvis_ssr,scenario_duration)
    optical_score = optical_results['Score'].mean()
    
    # 5. Radar Trackability
    # Compute from combined list of line-of-sight access dfcomblos_ssr
    # (Uses 49 SSR stations)
    radar_results = compute_tracking_stats(dfcomblos_ssr,scenario_duration)
    radar_score = radar_results['Score'].mean() # Total radar score

    # 6. Optical Detectability
    optical_detectability_results, best_station_opt_det = compute_optical_detectability(dfvis_ssrd, save_folder=save_folder)
    opt_detect_score = optical_detectability_results['Score'].iloc[0]

    
    # 7. Radar Detectability
    rcs = sat_dict['rcs'] # Extract RCS (m^2)
    radar_detectability_results, best_station_radar_det = compute_radar_detectability(dflos_ssrd, rcs, save_folder=save_folder) # Use LOS 
    radar_detect_score = radar_detectability_results['Score'].iloc[0]
    
    # Save results to dict ---------------------------------
    # Results to return in python
    results = {'SSRD_los':dflos_ssrd, 'SSRD_vis':dfvis_ssrd,
                'SSR_los':dflos_ssr, 'SSR_vis':dfvis_ssr,
                'radar_track_results':radar_results,'optical_track_results':optical_results,
                'radar_track_score':radar_score,'optical_track_score':optical_score,
                'optical_detect_results':optical_detectability_results,
                'optical_detect_score': opt_detect_score}
    
    # JSON compatible version for saving
    results1 = {'sat_dict':sat_dict,
                'radar_track_results':radar_results.to_dict(),
                'optical_track_results':optical_results.to_dict(),
                'optical_detect_results':optical_detectability_results.to_dict(),
                'radar_track_score':radar_score,
                'optical_track_score':optical_score,
                'optical_detect_score': opt_detect_score}
    
    
    print("\n\n Results")
    print("---------")
    
    # Print Trackability Results
    print('\nRadar Trackability')
    print(radar_results)
    print("\nOverall T Radar Score: {}\n".format(radar_score))
    
    print('\nOptical Trackability')
    print(optical_results)
    print("\nOverall T Optical Score: {}\n".format(optical_score))
    
    # Print Detectability Results
    print('\nOptical Detectability')
    print(optical_detectability_results)
    print("\nOverall Optical Detectability Score: {}\n".format(opt_detect_score))
    
    # Print Detectability Results
    print('\nRadar Detectability')
    print(radar_detectability_results)
    print("\nOverall Radar Detectability Score: {}\n".format(radar_detect_score))
    
    # Save results to json
    # Save to file
    import json
    with open(out_dir/'Results.json', 'w+') as fp:
        json.dump(results1, fp, sort_keys=True)
    
    # Formated results to txt
    template = '''
####################
DIT Analysis Results
####################

Scenario
--------
Start Date: {start_date} UTGC ({start_et} ET)
Stop Date:  {stop_date} UTGC ({stop_et} ET)
Time Step: {step} s

Target Satellite
----------------
SMA: {a} km \t  Semi-major axis
ECC: {e} \t  Eccentricity
INC: {i} deg \t  Inclination
RAAN: {raan} deg \t Right Ascension of the Ascending Node
AOP: {aop} deg \t Argument of Periapsis
TA: {ta} deg \t True anomaly (at Epoch)
Epoch: {epoch} {epoch_fmt} \t Epoch
Propagator: Two-body

Radar Trackability
------------------
{radar_results}

Overall Radar Trackability Score: {radar_score}

Optical Trackability
--------------------
{optical_results}

Overall Optical Trackability Score: {optical_score}

Optical Detectability
---------------------
Best access station: {best_station_opt_det}

{opt_detect_results}

Overall Optical Detectability Score: {opt_detect_score}

Radar Detectability
---------------------
Best access station: {best_station_radar_det}

{radar_detect_results}

Overall Radar Detectability Score: {radar_detect_score}


'''.format(**{  "start_date":str(start_date),
                "start_et":str(start_et),
                "stop_date":str(stop_date),
                "stop_et":str(stop_et),
                "step":str(step),
                "a":str(sat_dict['SMA']),
                "e":str(sat_dict['ECC']),
                "i":str(sat_dict['INC']),
                "raan":str(sat_dict['RAAN']),
                "aop":str(sat_dict['AOP']),
                "ta":str(sat_dict['TA']),
                "epoch":str(sat_dict['Epoch']),
                "epoch_fmt":str(sat_dict['DateFormat']),
                "radar_results":radar_results.to_string(index=False),
                "radar_score":radar_score,
                "optical_results":optical_results.to_string(index=False),
                "optical_score":optical_score,
                "best_station_opt_det":best_station_opt_det,
                "opt_detect_results":optical_detectability_results.to_string(index=False),
                "opt_detect_score":opt_detect_score,
                
                "best_station_radar_det":best_station_radar_det,
                "radar_detect_results":radar_detectability_results.to_string(index=False),
                "radar_detect_score":radar_detect_score,
                } )
    
    # Print and save results
    print(template)
    filename = out_dir/'DIT_Results.txt'
    with  open(str(filename),'w+') as myfile:
            myfile.write(template)    
    
    
    return results

def run_analysis_optical(sat_dict,start_date,stop_date,step,prefilter='crossings',save_folder='DITdata'):
    ''' Main DIT workflow with optical detection only. '''
    
    # 0. Check all generic kernels exist
    SPICEKernels.check_generic_kernels()
    
    # Convert start and stop dates to Ephemeris Time
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)
    
    # Define a working folder to output the data
    out_dir = get_data_home()/save_folder
    # Check if directory exists and create
    if not os.path.exists(str(out_dir)):
        os.makedirs(str(out_dir))
    
    
    # 1. Create Ephemeris files
    # - Satellite Ephemeris file (sat.bsp)
    # - SSR stations
    # - SSRD stations
    create_ephem_files(sat_dict,start_date,stop_date,step,method='two-body') # From elements
    # create_ephem_files(NORAD,start_date,stop_date,step,method='tle') # From TLE
    
    # 2. Compute satellite lighting intervals
    print('\nComputing Satellite Lighting intervals', flush=True)
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    satlight1 = spice.wnunid(satlight,satpartial) # Full or partial light
    
    # 3. Compute SSR and SSRD Station Lighting and access
    # SSR: use min_el = 30 deg (120 deg cone angle from zenith)
    # SSRD: use min_el = 5 deg (since targeted at satellite)
    # dflos_ssr, dfvis_ssr, dfcomblos_ssr, dfcombvis_ssr = compute_station_access('SSR',start_et,stop_et,satlight1,30.,save_folder=save_folder,save=True)
    dflos_ssrd, dfvis_ssrd, dfcomblos_ssrd, dfcombvis_ssrd = compute_station_access('SSRD',start_et,stop_et,satlight1,5.,prefilter=prefilter,save_folder=save_folder,save=True)
    
    # # 4. Optical Trackability
    # # Compute from combined list of visible access dfvis_ssr
    # # (Uses 49 SSR stations)
    # optical_results = compute_tracking_stats(dfcombvis_ssr,scenario_duration)
    # optical_score = optical_results['Score'].mean()
    
    # # 5. Radar Trackability
    # # Compute from combined list of line-of-sight access dfcomblos_ssr
    # # (Uses 49 SSR stations)
    # radar_results = compute_tracking_stats(dfcomblos_ssr,scenario_duration)
    # radar_score = radar_results['Score'].mean() # Total radar score

    # 6. Optical Detectability
    optical_detectability_results, best_station_opt_det = compute_optical_detectability(dfvis_ssrd)
    opt_detect_score = optical_detectability_results['Score'].iloc[0]

    
    # # 7. Radar Detectability
    # rcs = sat_dict['rcs'] # Extract RCS (m^2)
    # radar_detectability_results, best_station_radar_det = compute_radar_detectability(dflos_ssrd, rcs) # Use LOS 
    # radar_detect_score = radar_detectability_results['Score'].iloc[0]
    
    # Save results to dict ---------------------------------
    # Results to return in python
    results = {'SSRD_los':dflos_ssrd, 'SSRD_vis':dfvis_ssrd,
                # 'SSR_los':dflos_ssr, 'SSR_vis':dfvis_ssr,
                # 'radar_track_results':radar_results,'optical_track_results':optical_results,
                # 'radar_track_score':radar_score,'optical_track_score':optical_score,
                'optical_detect_results':optical_detectability_results,
                'optical_detect_score': opt_detect_score}
    
    # JSON compatible version for saving
    results1 = {'sat_dict':sat_dict,
                # 'radar_track_results':radar_results.to_dict(),
                # 'optical_track_results':optical_results.to_dict(),
                'optical_detect_results':optical_detectability_results.to_dict(),
                # 'radar_track_score':radar_score,
                # 'optical_track_score':optical_score,
                'optical_detect_score': opt_detect_score}
    
    
    print("\n\n Results")
    print("---------")
    
    # # Print Trackability Results
    # print('\nRadar Trackability')
    # print(radar_results)
    # print("\nOverall T Radar Score: {}\n".format(radar_score))
    
    # print('\nOptical Trackability')
    # print(optical_results)
    # print("\nOverall T Optical Score: {}\n".format(optical_score))
    
    # Print Detectability Results
    print('\nOptical Detectability')
    print(optical_detectability_results)
    print("\nOverall Optical Detectability Score: {}\n".format(opt_detect_score))
    
    # # Print Detectability Results
    # print('\nRadar Detectability')
    # print(radar_detectability_results)
    # print("\nOverall Radar Detectability Score: {}\n".format(radar_detect_score))
    
    # Save results to json
    # Save to file
    import json
    with open(out_dir/'Results.json', 'w+') as fp:
        json.dump(results1, fp, sort_keys=True)
    
    # Formated results to txt
    template = '''
####################
DIT Analysis Results
####################

Scenario
--------
Start Date: {start_date} UTGC ({start_et} ET)
Stop Date:  {stop_date} UTGC ({stop_et} ET)
Time Step: {step} s

Target Satellite
----------------
SMA: {a} km \t  Semi-major axis
ECC: {e} \t  Eccentricity
INC: {i} deg \t  Inclination
RAAN: {raan} deg \t Right Ascension of the Ascending Node
AOP: {aop} deg \t Argument of Periapsis
TA: {ta} deg \t True anomaly (at Epoch)
Epoch: {epoch} {epoch_fmt} \t Epoch
Propagator: Two-body

Optical Detectability
---------------------
Best access station: {best_station_opt_det}

{opt_detect_results}



'''.format(**{  "start_date":str(start_date),
                "start_et":str(start_et),
                "stop_date":str(stop_date),
                "stop_et":str(stop_et),
                "step":str(step),
                "a":str(sat_dict['SMA']),
                "e":str(sat_dict['ECC']),
                "i":str(sat_dict['INC']),
                "raan":str(sat_dict['RAAN']),
                "aop":str(sat_dict['AOP']),
                "ta":str(sat_dict['TA']),
                "epoch":str(sat_dict['Epoch']),
                "epoch_fmt":str(sat_dict['DateFormat']),
                # "radar_results":radar_results.to_string(index=False),
                # "radar_score":radar_score,
                # "optical_results":optical_results.to_string(index=False),
                # "optical_score":optical_score,
                "best_station_opt_det":best_station_opt_det,
                "opt_detect_results":optical_detectability_results.to_string(index=False),
                # "opt_detect_score":opt_detect_score,
                
                # "best_station_radar_det":best_station_radar_det,
                # "radar_detect_results":radar_detectability_results.to_string(index=False),
                # "radar_detect_score":radar_detect_score,
                } )
    
    # Print and save results
    print(template)
    filename = out_dir/'DIT_Results.txt'
    with  open(str(filename),'w+') as myfile:
            myfile.write(template)    
    
    
    return results




#%% Subroutines

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

def compute_station_access(network,start_et,stop_et,satlight, min_el, prefilter='crossings', save_folder='DITdata', save=False):
    
    # Loop through all stations and compute line-of-sight and visible access 
    
    # Define station names
    if network == 'SSR':
        # SSR Network contains 49 stations
        stations = ['SSR-'+str(i+1) for i in range(49)]
        # stations = stations[:4] # FIXME: Shortened for testing.
        
    elif network == 'SSRD':
        # SSRD Network contains 7 stations
        stations = ['SSRD-'+str(i+1) for i in range(7)]
    
    # Define a working folder to output the data
    out_dir = get_data_home()/save_folder
    
    # Compute Sun's angular radius of sun for elevation constraints
    # Angle found from radius of Sun and average Earth-Sun distance.
    et = np.arange(start_et,stop_et,10); et = np.append(et,stop_et)
    dftopo = get_ephem_TOPO(et,groundstations=[stations[0]])[0] # Ephemeris of first stations
    rsun = 695700. # Radius of Sun (km)
    # ang_radius = np.mean(np.rad2deg(np.arctan2(rsun,dftopo['Sun.R'].to_numpy())))
    ang_radius = np.rad2deg(np.arctan2(rsun, np.mean(dftopo['Sun.R'].to_numpy()))) # 0.269
    # This angle can be used to find station lighting conditions
    del et, dftopo
    
    # Loop through stations
    # TODO: in paralellize this function
    # stations = stations[:1] # FIXME: shortened list for testing
    # stations = stations[29:] # FIXME: testing problem on station 29
    # stations = stations[24:] # FIXME: testing problem on station 24
    
    print('Computing {} Station Lighting and Access intervals'.format(network), flush=True)
    print('Stations: {}'.format(stations),flush=True)
    print('Prefilter: {}'.format(prefilter),flush=True)
    dflos = pd.DataFrame(columns=['Station','Access','Start','Stop','Duration'])
    dfvis = pd.DataFrame(columns=['Station','Access','Start','Stop','Duration'])
    los_access_list = [] # List of LOS access interval of stations
    vis_access_list = [] # List of visible access interval of stations
    
    for gs in tqdm(stations): 
        
        # Compute station lighting intervals (~0.15 s)
        # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs,method='eclipse')
        gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs,ref_el=ang_radius) # 0.268986 ref_el=-0.25
        
        # Compute line-of-sight access intervals (~22 s)
        # Use min_el = 30 deg (120 deg cone angle from zenith)
        t_start = time.time()
        los_access = find_access(start_et,stop_et,station=gs,min_el=min_el,prefilter=prefilter) # Change prefilter=None if error in pre-filtering algorithm
        los_access_list.append(los_access) # Append to list of station access intervals
        # Convert to dataframe
        dflos_i = window_to_dataframe(los_access,timefmt='ET') # Load as dataframe
        dflos_i.insert(0,'Access',dflos_i.index)
        dflos_i.insert(0,'Station',gs)
        dflos = dflos.append(dflos_i) # Append to global dataframe
        # dflos = pd.concat([dflos, dflos_i], ignore_index=True) # FIXME: replace with this new version
        # print('find_access: runtime {} s'.format(time.time() - t_start))
        
        # Compute visible (constrained) access intervals (~ 0.004 s)
        # access = constrain_access_by_lighting(los_access,gslight,satdark)
        access = constrain_access_by_lighting(los_access,gsdark,satlight)
        vis_access_list.append(access) # Append to list of station access intervals
        # Convert to dataframe
        dfvis_i = window_to_dataframe(access,timefmt='ET') # Load as dataframe
        dfvis_i.insert(0,'Access',dfvis_i.index)
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
            if len(dfvis_i)>0:
                # Add rows for global statistics (min,max,mean,total)
                # Minimum vals
                min_row = dflos_i.iloc[dflos_i['Duration'].idxmin()].to_dict()
                min_row['Station'] = 'Min Duration' # Change label
                dflos_i = dflos_i.append(min_row, ignore_index=True)
                # Maximum Duration vals
                max_row = dflos_i.iloc[dflos_i['Duration'].idxmax()].to_dict()
                max_row['Station'] = 'Max Duration' # Change label
                dflos_i = dflos_i.append(max_row, ignore_index=True)
                # Maximum Duration vals
                mean_row = {'Station':'Mean Duration','Duration':dflos_i['Duration'].mean()}
                dflos_i = dflos_i.append(mean_row, ignore_index=True)
                # Insert empty row
                ind = np.where(dflos_i['Station']=='Min Duration')[0]
                df_new = pd.DataFrame(index=ind -1. + 0.5) # New dataframe at half integer indices
                dflos_i = pd.concat([dflos_i, df_new]).sort_index().reset_index(drop=True)
                dflos_i['Access'] = pd.to_numeric(dflos_i['Access'], errors = 'coerce').astype(pd.Int32Dtype())
            # Save access data to file
            filename = gs + "_los_access_intervals.csv"
            dflos_i.to_csv(str(out_dir/filename),index=False)
            del dflos_i
            
            
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
            if len(dfvis_i)>0:
                # Add rows for global statistics (min,max,mean,total)
                # Minimum vals
                min_row = dfvis_i.iloc[dfvis_i['Duration'].idxmin()].to_dict()
                min_row['Station'] = 'Min Duration' # Change label
                dfvis_i = dfvis_i.append(min_row, ignore_index=True)
                # Maximum Duration vals
                max_row = dfvis_i.iloc[dfvis_i['Duration'].idxmax()].to_dict()
                max_row['Station'] = 'Max Duration' # Change label
                dfvis_i = dfvis_i.append(max_row, ignore_index=True)
                # Maximum Duration vals
                mean_row = {'Station':'Mean Duration','Duration':dfvis_i['Duration'].mean()}
                dfvis_i = dfvis_i.append(mean_row, ignore_index=True)
                # Insert empty row
                ind = np.where(dfvis_i['Station']=='Min Duration')[0]
                df_new = pd.DataFrame(index=ind -1. + 0.5) # New dataframe at half integer indices
                dfvis_i = pd.concat([dfvis_i, df_new]).sort_index().reset_index(drop=True)
                dfvis_i['Access'] = pd.to_numeric(dfvis_i['Access'], errors = 'coerce').astype(pd.Int32Dtype())
            # Save access data to file
            filename = gs + "_vis_access_intervals.csv"
            dfvis_i.to_csv(str(out_dir/filename),index=False)
            del dfvis_i
    
    # Compute combined LOS access intervals (union of all stations)
    combined_los_access = los_access_list[0] # First station
    if len(los_access_list)>1:
        for win in los_access_list[1:]:
            combined_los_access = spice.wnunid(combined_los_access,win) # Union of intervals
    dfcomblos = window_to_dataframe(combined_los_access,timefmt='ET') # Load as dataframe
    
    # Compute combined visible access intervals (union of all stations)
    combined_vis_access = vis_access_list[0] # First station
    if len(vis_access_list)>1:
        for win in vis_access_list[1:]:
            combined_vis_access = spice.wnunid(combined_vis_access,win) # Union of intervals
    dfcombvis = window_to_dataframe(combined_vis_access,timefmt='ET') # Load as dataframe
    
    # Save access lists to csv
    dflos.to_csv(str(out_dir/"All_LOS_Access_{}.csv".format(network)),index=False)
    dfvis.to_csv(str(out_dir/"All_Vis_Access_{}.csv".format(network)),index=False)
    dfcombvis.to_csv(str(out_dir/"Combined_Vis_Access_{}.csv".format(network)),index=False)
    dfcomblos.to_csv(str(out_dir/"Combined_LOS_Access_{}.csv".format(network)),index=False)

    # Plot the access periods
    filename = "Access_intervals_{}.html".format(network)
    title = network + " Network Visible Access Intervals"
    plot_time_windows(vis_access_list,stations,stations,#['blue']*len(stations), 
                      filename=str(out_dir/filename), 
                      group_label='Station',
                      title=title)
    
    return dflos, dfvis, dfcomblos, dfcombvis

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

def compute_optical_detectability(df,save_folder='DITdata'):
    '''
    Compute the optical detectability score

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe containing visible access intervals for the SSR network.

    Returns
    -------
    results : TYPE
        DESCRIPTION.
    best_station : str
        Station with the longest access used for metric computation.

    '''
    
    # Find station with largest total access.
    dfgroup = df[['Station','Duration']].groupby(['Station']).sum()
    best_station = dfgroup['Duration'].idxmax()
    # best_station = 'SSRD-' # Hardcoded
    print("\nComputing optical Detectability", flush=True)
    print("--------------------------------", flush=True)
    print("Station with best access: {}".format(best_station), flush=True)
    
    # Extract access for that station
    df = df[df['Station'] == best_station]
    
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
    # Get visible access for this station
    dfa = df[df.Station == best_station]
    
    # Compute visual magnitude
    Rsat = 1 # Radius of satellite (m)
    msat = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12) # Lambertian phase function
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,lambertian_phase_function=False) # Constant phase function v(alpha)=1
    # Add to dataframe
    dftopo.insert(len(dftopo.columns),'Vmag',list(msat))
    dftopo.insert(len(dftopo.columns),'Vmag2',list(msat2))
    # Save to file
    out_dir = get_data_home()/save_folder
    filename = out_dir/'BestAccess.csv'
    dftopo.to_csv(str(out_dir/filename),index=False)
    
    
    # Constraints
    cutoff_mag = 15. # Maximum magnitude for visibility
    
    # Compute contrained stats
    max_mag = np.nanmax(msat[msat<=cutoff_mag])  # Maximum (dimest) magnitude
    min_mag = np.nanmin(msat[msat<=cutoff_mag])  # Minimum (brightest) magnitude 
    avg_mag = np.nanmean(msat[msat<=cutoff_mag]) # Mean magnitude
    
    # Optical detectability Scoring Criteria
    # See Table 7 of R Steindl thesis.
    if avg_mag > 15:
        opt_detectability_tier = 'Difficult to Track'
        opt_detectability_score = 0.5
    elif avg_mag <= 15.:
        opt_detectability_tier = "Detectable"
        opt_detectability_score = 1.0
    
    results = pd.DataFrame(columns=['Metric','Value','Tier','Score'])
    opt_detect_row = {'Metric': 'Avg Vmag', 'Value': avg_mag, 'Tier': opt_detectability_tier, 'Score': opt_detectability_score}
    results = results.append(opt_detect_row, ignore_index=True)
    
    
    # # Generate plots
    # import matplotlib.pyplot as plt
    # fig, (ax1,ax2) = plt.subplots(2,1)
    # fig.suptitle('Visual Magnitude')
    # # Magnitude vs elevation
    # ax1.plot(np.rad2deg(dftopo['Sat.El']),msat,'.b')
    # ax1.plot(np.rad2deg(dftopo['Sat.El']),msat2,'.k')
    # ax1.plot([0,90],[max_mag,max_mag],'-r')
    # ax1.set_xlabel("Elevation (deg)")
    # ax1.set_ylabel("Visual Magnitude (mag)")
    # ax1.invert_yaxis() # Invert y axis
    # # Az/El
    # ax2.plot(dftopo['ET'],msat,'-b')
    # ax2.plot(dftopo['ET'],msat2,'-k')
    # # Max/min/mean
    # ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[max_mag,max_mag],'-r')
    # ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[min_mag,min_mag],'-r')
    # ax2.plot([dftopo['ET'].iloc[0],dftopo['ET'].iloc[-1]],[avg_mag,avg_mag],'-r')
    # ax2.invert_yaxis() # Invert y axis
    # ax2.set_xlabel("Epoch (ET)")
    # ax2.set_ylabel("Visual Magnitude (mag)")
    # fig.show()
    
    # Plot
    out_dir = get_data_home()/save_folder
    plot_visibility(dftopo, filename = str(out_dir/"OpticalDetectability.html"),
                    title="Optical Detectability Station {}".format(best_station)) # Plot of the satellite el,range,visible magnitude
    
    plot_overpass_skyplot(dftopo, dfa,
                          filename = str(out_dir/"OpticalDetectabilitySkyplot.html"),
                          title="Visible Overpasses Station {}".format(best_station)
                          ) # Sky plot of overpasses.
    
    
    return results, best_station

def compute_radar_detectability(dflos_ssrd, rcs, save_folder='DITdata'):
    
    # Find station with largest total access.
    dfgroup = dflos_ssrd[['Station','Duration']].groupby(['Station']).sum()
    best_station = dfgroup['Duration'].idxmax()
    print("\nComputing Radar Detectability", flush=True)
    print("-------------------------------", flush=True)
    print("Station with best access: {}".format(best_station), flush=True)
    
    # Extract access for that station
    df = dflos_ssrd[dflos_ssrd['Station'] == best_station]
    
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
    # Get visible access for this station
    dfa = dflos_ssrd[dflos_ssrd.Station == best_station]
    
    # Compute link budget to get SNR
    
    # Inputs
    Pt = 10*np.log10(10E6) # Transmit power (dBW) 70 dBW (computed from 10 MW) ref [1]
    Gt = 36.39 # Transmitter gain (dBi) [2] From MATLAB script
    Gr = 0 # Receiver gain (dBi) ref [3] LNAGain = 1 (== 0 dB)
    f = 0.45 # Carrier frequency (GHz) (450 MHz) ref [1]
    # rcs (m^2) (input variable)
    Ts = 290 # System temperature ref[3]  ConstantNoiseTemp = 290 K
    tp = 1E-07 # Pulse width (s) ref [3]  PulseWidth = 1e-07 sec
    R = dftopo['Sat.R'].to_numpy() # Groundstation to Sat Range (km) (from geometry)
    L = 0 # Additional losses (dBW) TODO 
    # References
    # [1] Riley's Thesis
    # [2] MATLAB script Radar_array.m uses Phased Array toolbox
    # [3] STK Radar1.rd file
    
    # Compute received power, noise, single-pulse SNR at time steps
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    
    # Use SNR1 to compute Probability of Detection
    pfa = 0.0001 # Probability of false alarm ref [3]
    PD = compute_probability_of_detection(SNR1,pfa=pfa)
    
    # Find max probability
    max_PD = np.nanmax(PD)
    
    
    # Add to dataframe
    dftopo.insert(len(dftopo.columns),'Pr',list(Pr))
    dftopo.insert(len(dftopo.columns),'Np',list(Np*np.ones(len(dftopo))))
    dftopo.insert(len(dftopo.columns),'SNR1',list(SNR1))
    dftopo.insert(len(dftopo.columns),'PD',list(PD))
    
    # Radar detectability Scoring Criteria
    # See Table 7 of R Steindl thesis.
    if max_PD < 0.5:
        radar_detectability_tier = 'Difficult to Track'
        radar_detectability_score = 0
    elif 0.5 <= max_PD < 0.75:
        radar_detectability_tier = "Detectable"
        radar_detectability_score = 0.5
    elif max_PD >= 0.75:
        radar_detectability_tier = "More Detectable"
        radar_detectability_score = 1.0
    
    results = pd.DataFrame(columns=['Metric','Value','Tier','Score'])
    radar_detect_row = {'Metric': 'Max Pd', 'Value': max_PD, 'Tier': radar_detectability_tier, 'Score': radar_detectability_score}
    results = results.append(radar_detect_row, ignore_index=True)
    
    # TODO: Generate plot
    out_dir = get_data_home()/save_folder
    plot_linkbudget(dftopo, filename = str(out_dir/"RadarDetectability.html"),
                    title="Radar Detectability Station {}".format(best_station)) # Plot of the satellite el,range,SNR1,Pd
    
    plot_overpass_skyplot(dftopo, dfa,
                          filename = str(out_dir/"RadarDetectabilitySkyplot.html"),
                          title="Overpasses Station {}".format(best_station)
                          ) # Sky plot of overpasses.

    return results, best_station


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




