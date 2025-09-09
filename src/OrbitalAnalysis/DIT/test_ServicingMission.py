# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 21:25:58 2025

@author: scott

Servicing Mission Analysis
--------------------------

Analyse cient and servicer bsp files

"""

from OrbitalAnalysis.DIT.Ephem import *
from OrbitalAnalysis.DIT.VisualMagnitude import *
from OrbitalAnalysis.DIT.Events import *
from OrbitalAnalysis.Visualization import plot_visibility
from OrbitalAnalysis.utils import get_data_home, get_root_dir
from OrbitalAnalysis.DIT.GroundstationData import get_groundstations

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import pdb


#%% Get Topocentric Ephemeris over entire time interval

def get_client_servicer_ephem_topo():
    
    # SPK file directory
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer.bsp')
        
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    
    
    
    # Client Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(client_ephem_file)
    sat_NAIF = list(results.keys())[0]
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    print('client_orbit.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))    
    print('')
    
    
    # Servicer Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(servicer_ephem_file)
    sat_NAIF = list(results.keys())[0]
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    print('servicer.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))    
    print('')
    
    # Create array of epochs
    # Add a buffer from the start and stop times
    step = 5. # Step size (s)
    et = np.arange(start_et+60., stop_et-60., step)
    station = 'DSS-43' # Select station
    out_dir = get_data_home()/'ServicingMission' # Output directory
    
    # Get ephemerides at DSS-43
    dfc = get_ephem_TOPO(et,groundstations=[station], sat_ephem=client_ephem_file)[0] # Client
    dfs = get_ephem_TOPO(et,groundstations=[station], sat_ephem=servicer_ephem_file)[0] # Servicer
    
    
    # Compute visual magnitude client
    Rsat = 0.713 # Satellite radius (m)
    p = 0.175 # Albedo (17.5%)
    msat = compute_visual_magnitude(dfc,Rsat,p=p,k=0.12) # Lambertian phase function
    msat2 = compute_visual_magnitude(dfc,Rsat,p=p,k=0.12,lambertian_phase_function=False) # Constant phase function v(alpha)=1
    # Add to dataframe
    dfc.insert(len(dfc.columns),'Vmag',list(msat))
    dfc.insert(len(dfc.columns),'Vmag2',list(msat2))
    # Save to file
    filename = out_dir/'ClientVisibility.csv'
    dfc.to_csv(str(out_dir/filename),index=False)
    # Plot results
    plot_visibility(dfc, title="Client Optical Detectability Station {}".format(station), filename='ClientVisibility.html')
    
    # Compute visual magnitude servicer
    Rsat = 0.713 # Satellite radius (m)
    p = 0.175 # Albedo (17.5%)
    msat = compute_visual_magnitude(dfs,Rsat,p=p,k=0.12) # Lambertian phase function
    msat2 = compute_visual_magnitude(dfs,Rsat,p=p,k=0.12,lambertian_phase_function=False) # Constant phase function v(alpha)=1
    # Add to dataframe
    dfs.insert(len(dfs.columns),'Vmag',list(msat))
    dfs.insert(len(dfs.columns),'Vmag2',list(msat2))
    # Save to file
    filename = out_dir/'ServicerVisibility.csv'
    dfs.to_csv(str(out_dir/filename),index=False)    
    # Plot results    
    plot_visibility(dfs, title="Servicer Optical Detectability Station {}".format(station), filename='ServicerVisibility.html')
    
    
    return dfc, dfs

#%% Get Access Times
    
def get_access_intervals(min_el = 30., network='LeoLabs', satellite='servicer'):
    
    # min_el = Minimum elevation (deg)
    # network = 'LeoLabs', 'SSR', 'SSRD'
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer.bsp')
    if satellite.lower() == 'servicer':
        sat_ephem_file = servicer_ephem_file
    elif satellite.lower() == 'client':
        sat_ephem_file = client_ephem_file
    
    
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.bsp") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.tf") )
    
    # Get groundstation data
    dfgs = get_groundstations(network=network) 
    stations = list(dfgs.Name) # List of stations
    
    # Define output folder
    out_dir = get_data_home()/'ServicingMission' # Output directory
    
    # Servicer Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(servicer_ephem_file)
    sat_NAIF = list(results.keys())[0]
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    
    # Create array of epochs
    # Add a buffer from the start and stop times
    step = 5. # Step size (s)
    # et = np.arange(start_et+60., stop_et-60., step)
    
    # Access settings
    start_et += 60. # Add buffer from ephemeris coverage
    stop_et -= 60.  # Add buffere from ephemeris coverage
    prefilter=None
    
    # Loop through ground stations
    print('Computing {} Station Lighting and Access intervals'.format(network), flush=True)
    print('Stations: {}'.format(stations),flush=True)
    print('Prefilter: {}'.format(prefilter),flush=True)
    dflos = pd.DataFrame(columns=['Station','Access','Start','Stop','Duration'])
    dfvis = pd.DataFrame(columns=['Station','Access','Start','Stop','Duration'])
    los_access_list = [] # List of LOS access interval of stations
    vis_access_list = [] # List of visible access interval of stations
    for gs in tqdm(stations): 
        
        # Compute line-of-sight access intervals (~22 s)
        # Use min_el = 30 deg (120 deg cone angle from zenith)
        t_start = time.time()
        station_name = gs
        los_access = find_access(start_et,stop_et,station=station_name,min_el=min_el,prefilter=prefilter,sat_ephem=sat_ephem_file) # Change prefilter=None if error in pre-filtering algorithm
        los_access_list.append(los_access) # Append to list of station access intervals
        # Convert to dataframe
        dflos_i = window_to_dataframe(los_access,timefmt='ET') # Load as dataframe
        dflos_i.insert(0,'Access',dflos_i.index)
        dflos_i.insert(0,'Station',gs)
        dflos = dflos.append(dflos_i) # Append to global dataframe
        # dflos = pd.concat([dflos, dflos_i], ignore_index=True) # FIXME: replace with this new version
        # print('find_access: runtime {} s'.format(time.time() - t_start))
    
    
    return dflos

