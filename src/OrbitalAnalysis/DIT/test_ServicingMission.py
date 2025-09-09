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

import numpy as np
import pandas as pd

import pdb


#%% 

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



