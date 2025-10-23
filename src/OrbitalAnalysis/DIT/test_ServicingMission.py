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
from OrbitalAnalysis.DIT.Communications import *
from OrbitalAnalysis.Functions import coe_from_sv
# from astrologistics.Functions import M_to_TA

from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.Density import *

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import pdb

#%% Create Groundstation ephemeris files

def create_groundstation_ephem_files():
    
    #TODO: Create details of network in GroundstationData.py
    
    # Load locations of groundstations
    df = get_groundstations(network='LeoLabs')
    # Select name of output files NAME_stations.bsp, NAME_stations.tf
    NAME= 'LEOLABS'
    create_station_ephem(df, network_name=NAME)
    
    return

#%% Get Topocentric Ephemeris over entire time interval

def get_client_servicer_ephem_topo():
    
    # SPK file directory
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit_through_prox_ops_plus_24hr.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer_through_prox_approach.bsp')
    
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    
    
    
    # Client Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(client_ephem_file)
    sat_NAIF = list(results.keys())[0]
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('client_orbit_through_prox_ops_plus_24hr.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))    
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
    print('')
    
    
    # Servicer Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(servicer_ephem_file)
    sat_NAIF = list(results.keys())[0]
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('servicer_through_prox_approach.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))    
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
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
    
    # Get ephem 
    
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
    
    
    # Compute link budget to get SNR
    
    # Inputs
    Pt = 10*np.log10(10E6) # Transmit power (dBW) 70 dBW (computed from 10 MW) ref [1]
    Gt = 36.39 # Transmitter gain (dBi) [2] From MATLAB script
    Gr = 0 # Receiver gain (dBi) ref [3] LNAGain = 1 (== 0 dB)
    f = 0.45 # Carrier frequency (GHz) (450 MHz) ref [1]
    # rcs (m^2) (input variable)
    
    Ts = 290 # System temperature ref[3]  ConstantNoiseTemp = 290 K
    tp = 1E-07 # Pulse width (s) ref [3]  PulseWidth = 1e-07 sec
    
    L = 0 # Additional losses (dBW) TODO 
    # References
    # [1] Riley's Thesis
    # [2] MATLAB script Radar_array.m uses Phased Array toolbox
    # [3] STK Radar1.rd file
    
    
    # Client 
    # Compute received power, noise, single-pulse SNR at time steps
    rcs = 1 # RCS of client satellite (m^2)
    R = dfc['Sat.R'].to_numpy() # Groundstation to Sat Range (km) (from geometry)
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    # Use SNR1 to compute Probability of Detection
    pfa = 0.0001 # Probability of false alarm ref [3]
    PD = compute_probability_of_detection(SNR1,pfa=pfa)
    # Find max probability
    max_PD = np.nanmax(PD)
    # Add to dataframe
    dfc.insert(len(dfc.columns),'Pr',list(Pr))
    dfc.insert(len(dfc.columns),'Np',list(Np*np.ones(len(dfc))))
    dfc.insert(len(dfc.columns),'SNR1',list(SNR1))
    dfc.insert(len(dfc.columns),'PD',list(PD))
    
    # Client 
    # Compute received power, noise, single-pulse SNR at time steps
    rcs = 1 # RCS of servicer satellite (m^2)
    R = dfs['Sat.R'].to_numpy() # Groundstation to Sat Range (km) (from geometry)
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    # Use SNR1 to compute Probability of Detection
    pfa = 0.0001 # Probability of false alarm ref [3]
    PD = compute_probability_of_detection(SNR1,pfa=pfa)
    # Find max probability
    max_PD = np.nanmax(PD)
    # Add to dataframe
    dfs.insert(len(dfs.columns),'Pr',list(Pr))
    dfs.insert(len(dfs.columns),'Np',list(Np*np.ones(len(dfs))))
    dfs.insert(len(dfs.columns),'SNR1',list(SNR1))
    dfs.insert(len(dfs.columns),'PD',list(PD))
    
    return dfc, dfs

#%% Get ITRF Ephemeris over entire time interval

def get_client_servicer_ephem_ITFR():
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit_through_prox_ops_plus_24hr.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer_through_prox_approach.bsp')
    
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.bsp") )
    
    
    # Client Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(client_ephem_file)
    sat_NAIF = list(results.keys())[0]
    client_NAIF = sat_NAIF
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('client_orbit_through_prox_ops_plus_24hr.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
    print('')
    
    # Servicer Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(servicer_ephem_file)
    sat_NAIF = list(results.keys())[0]
    servicer_NAIF = sat_NAIF
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('servicer_through_prox_approach.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
    print('')
    
    # Create array of epochs
    # Add a buffer from the start and stop times
    step = 5. # Step size (s)
    et = np.arange(start_et+60., stop_et-60., step)
    station = 'DSS-43' # Select station
    out_dir = get_data_home()/'ServicingMission' # Output directory
    
    # Get list of groundstations
    df = get_groundstations(network='LeoLabs')
    groundstations = df.Name.to_list()
    groundstations = ['DSS-43']
    
    # Get servicer ephem
    dfc = get_ephem_ITFR(et, groundstations=groundstations, sat_ephem=client_ephem_file)
    dfs = get_ephem_ITFR(et, groundstations=groundstations, sat_ephem=servicer_ephem_file)   
    
    # Get ephem coverage and orbital elements of satellites
    # Client
    # State vector at start
    mu = 398600 # Gravitational parameter of Earth 398600.4
    RE = 6378;   # Radius of Earth (km)
    # [sv0, ltime] = spice.spkezr( str(client_NAIF), et[0], 'ITRF93', 'lt+s', 'earth')
    [sv0, ltime] = spice.spkezr( str(client_NAIF), et[0], 'J2000', 'lt+s', 'earth')
    r_mag = np.linalg.norm(sv0[:3]) # Radius (km)
    v_mag = np.linalg.norm(sv0[3:]) # Velocity (km/s)
    coe = coe_from_sv(sv0[:3],sv0[3:],mu=398600.4,units='km')
    a,e,i,om,w,TA = coe
    # Altitude 419.3 km???
    
    # Plot orbits
    
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(dfc['Sat.X'], dfc['Sat.Y'], dfc['Sat.Z'], color='blue', marker='o')
    
    

    
    

    return dfc, dfs

def get_client_servicer_ephem_ECI():
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit_through_prox_ops_plus_24hr.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer_through_prox_approach.bsp')
    
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.bsp") )
    
    
    # Client Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(client_ephem_file)
    sat_NAIF = list(results.keys())[0]
    client_NAIF = sat_NAIF
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('client_orbit_through_prox_ops_plus_24hr.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
    print('')
    
    # Servicer Spacecraft
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    results = get_ephem_details(servicer_ephem_file)
    sat_NAIF = list(results.keys())[0]
    servicer_NAIF = sat_NAIF
    start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    start_dt = spice.et2datetime(start_et) # Datetime
    stop_dt = spice.et2datetime(stop_et) # Datetime
    print('servicer_through_prox_approach.bsp')
    print('NAIFID: ' + str(sat_NAIF))
    print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))
    print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))  
    print('')
    
    # Create array of epochs
    # Add a buffer from the start and stop times
    step = 5. # Step size (s)
    et = np.arange(start_et+60., stop_et-60., step)
    station = 'DSS-43' # Select station
    out_dir = get_data_home()/'ServicingMission' # Output directory
    
    # Get list of groundstations
    df = get_groundstations(network='LeoLabs')
    groundstations = df.Name.to_list()
    groundstations = ['DSS-43']
    
    # Get servicer ephem
    dfc = get_ephem_ECI(et, groundstations=groundstations, sat_ephem=client_ephem_file)
    dfs = get_ephem_ECI(et, groundstations=groundstations, sat_ephem=servicer_ephem_file)   
    
    # Get ephem coverage and orbital elements of satellites
    # Client
    # State vector at start
    mu = 398600 # Gravitational parameter of Earth 398600.4
    RE = 6378;   # Radius of Earth (km)
    # [sv0, ltime] = spice.spkezr( str(client_NAIF), et[0], 'ITRF93', 'lt+s', 'earth')
    [sv0, ltime] = spice.spkezr( str(client_NAIF), et[0], 'J2000', 'lt+s', 'earth')
    r_mag = np.linalg.norm(sv0[:3]) # Radius (km)
    v_mag = np.linalg.norm(sv0[3:]) # Velocity (km/s)
    coe = coe_from_sv(sv0[:3],sv0[3:],mu=398600.4,units='km')
    a,e,i,om,w,TA = coe
    # Altitude 419.3 km???
    
    # Plot orbits
    # pdb.set_trace()
    sun = dfc[['Sun.X','Sun.Y','Sun.Z']].to_numpy()
    sun = sun/np.linalg.norm(sun, axis=-1)[:, np.newaxis]
    
    # Earth sphere
    # Define the radius and center of the sphere
    radius = RE
    center_x, center_y, center_z = 0, 0, 0
    
    # Generate spherical coordinates
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    
    # Calculate Cartesian coordinates with the specified radius and center
    x = radius * np.cos(u) * np.sin(v) + center_x
    y = radius * np.sin(u) * np.sin(v) + center_y
    z = radius * np.cos(v) + center_z
    
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(dfc['Sat.X'], dfc['Sat.Y'], dfc['Sat.Z'], color='blue')
    ax.scatter3D(sun[:,0]*2*RE, sun[:,1]*2*RE, sun[:,2]*2*RE, color='yellow')
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.7)
    
    # Calculate the ranges for each axis
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    
    # Set the box aspect to match the data ranges
    # This makes a unit of data along each axis appear the same size
    ax.set_box_aspect([x_range, y_range, z_range])
    
    # pdb.set_trace()
    
    
    

    return dfc, dfs

#%% Get Access Times
    
def get_access_intervals(min_el = 30., network='LeoLabs', satellite='servicer'):
    
    # min_el = Minimum elevation (deg)
    # network = 'LeoLabs', 'SSR', 'SSRD'
    
    # Satellite kernel
    client_ephem_file = str(get_data_home()/'ServicingMission'/'client_orbit_through_prox_ops_plus_24hr.bsp')
    servicer_ephem_file = str(get_data_home()/'ServicingMission'/'servicer_through_prox_approach.bsp')
    if satellite.lower() == 'servicer':
        sat_ephem_file = servicer_ephem_file
    elif satellite.lower() == 'client':
        sat_ephem_file = client_ephem_file
    
    # Load kernels
    kernel_dir = get_data_home() / 'Kernels'
    # spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    spice.furnsh( str(kernel_dir/"earth_000101_220616_220323.bpc") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.bsp") )
    spice.furnsh( str(kernel_dir/"LEOLABS_stations.tf") )
    spice.furnsh( str(kernel_dir/'de440s.bsp') ) # Planetary ephemeris
    spice.furnsh( str(kernel_dir/'pck00010.tpc') )
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    
    # Get coverage of earth predict
    
    # spice_file = str(kernel_dir/"earth_200101_990825_predict.bpc")
    # results = get_ephem_details(spice_file)
    # sat_NAIF = list(results.keys())[0]
    # start_et, stop_et = results[sat_NAIF]['start_et'], results[sat_NAIF]['stop_et']
    # start_dt = spice.et2datetime(start_et) # Datetime
    # stop_dt = spice.et2datetime(stop_et) # Datetime
    # print("\n" + spice_file + " coverage")
    # print('Coverage (et): ' + str(start_et) + " to " + str(stop_et))    
    # print('Coverage (dt): ' + str(start_dt) + " to " + str(stop_dt))
    # print('')
    
    
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
    
    # Get satellite lighting -------
    # Use phase angle
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/phaseq_c.html
    #
    #                 illmn     obsrvr
    # illmn as seen      ^       /
    # from target at     |      /
    # et - lt.           |     /
    #                   >|..../< phase angle
    #                    |   /
    #                  . |  /
    #                .   | /
    #               .    |v        target as seen from obsrvr
    #         sep   .  target      at et
    #                .  /
    #                  /
    #                 v
    #  pi = sep + phase
    #  so
    #  phase = pi - sep
    
    # Iluminator = sun
    # Observer = Center of Earth 301
    # Target = satellite
    # Constraint for sunlight: phase angle > critical value
    
    # Create time window of interest
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfpa_c.html
    MAXWIN = 2000 # Maximum number of intervals
    cnfine = spice.cell_double(MAXWIN) # Initialize window of interest
    spice.wninsd(start_et + 60, stop_et - 60, cnfine ) # Insert time interval in window
    
    # TODO: Try find separation angle between sun and earth as viewed from sat
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/trgsep.html
    
    # Phase angle geometry search settings
    target = str(sat_NAIF) # Target
    illmn =  "SUN"          # Illuminator
    obsrvr = "EARTH" # Name of occulting body
    abcorr = "lt+s" # Aberration correction flag correct for one-way light time and stellar aberration using a Newtonian formulation.
    step = 10. # Step size (s)
    refval = 0.57598845 # Reference value for phase angle (rad)
    adjust = 0.0
    NINTVL = MAXWIN
    
    # Get phase angle at epochs
    step = 5. # Step size (s)
    et = np.arange(start_et, stop_et-60., step)
    phase = np.array([spice.phaseq(eti, target, illmn, obsrvr, abcorr ) for eti in et ])
    # phaseq = spice.phaseq(et, target, illmn, obsrvr, abcorr )
    # Minimum phase angle is 74.6 deg
    
    
    # Perform the search. The SPICE window `result' contains the set of times when the condition is met.
    result = spice.cell_double(2*MAXWIN) # Initialize result
    result = spice.gfpa( target, illmn, abcorr, obsrvr, ">", refval, 
                        adjust, step, NINTVL, cnfine, result );
    

    
    
    # Find occulations
    # Occultation geometry search settings
    occtyp = "ANY"  # Type of occultation (Full,Annular,Partial,Any)
    front = "EARTH" # Name of occulting body
    fshape = "ELLIPSOID" # Type of shape model for front body (POINT, ELLIPSOID, DSK/UNPRIORITIZED)
    fframe = "ITRF93" #"IAU_EARTH" # # Body-fixed frame of front body
    back =  "SUN" # Name of occulted body
    bshape = "ELLIPSOID" # Type of shape model for back body
    bframe = "IAU_SUN" # Body-fixed frame of back body (empty)
    # abcorr = "NONE" # Aberration correction flag
    abcorr = "lt"
    obsrvr = str(sat_NAIF[0])  # Observer
    step = 10. # Step size (s)
    
    # Full occulation (dark or umbra)
    dark = spice.cell_double(2*MAXWIN) # Initialize result
    dark = spice.gfoclt ( "ANY",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, dark          )
    
    
    df_dark = window_to_dataframe(dark,timefmt='ET',method='loop')
    
    
    
    
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

# get_access_intervals(min_el = 30., network='LeoLabs', satellite='servicer')

#%% Create spice file

def create_spice_file_tles():
    
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/"earth_200101_990825_predict.bpc") )
    spice.furnsh( str(kernel_dir/"earth_000101_220616_220323.bpc") )
    spice.furnsh( str(kernel_dir/"earth_2025_250826_2125_predict.bpc") )
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'de440s.bsp') ) # Planetary ephemeris
    spice.furnsh( str(kernel_dir/'pck00010.tpc') )
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topo

    
    # Servicing Mission Client
    save_folder = 'servicemission'
    sat_dict = {"DateFormat": "UTCGregorian", 
                "Epoch": '29 Jun 2025 00:00:00.000',
                # "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 6803.154120854776, "ECC": 0.0014176146268132896, "INC": np.rad2deg(0.9015521730288635),
                "RAAN": np.rad2deg(-2.9716551810327254), "AOP": np.rad2deg(0.8986224536838303), 
                "TA": np.rad2deg(M_to_TA(-0.8964062521791888, 0.0014176146268132896)),
                "rcs": 1.9888}
    
    
    # Jackie mission
    # start_et = 804427269.1841716 
    # stop_et = 804780214.9342455
    
    # 2020
    start_date = '2025-06-15 00:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2025-07-15 23:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    
    step = 10
    method='two-body'
    
    
    
    # create_satellite_ephem(sat_dict,start_et,stop_et,step,method=method)
    # #Saved as sat.bsp
    
    # Find lighting
    spice.furnsh( str(kernel_dir/'sat.bsp') ) # Leap second kernel
    
    sat_ephem = str(kernel_dir/"sat.bsp")
    satlight, satpartial, satdark = find_sat_lighting(start_et+60.,stop_et-60., sat_ephem = sat_ephem)
    light_df = window_to_dataframe(satlight,timefmt='ET',method='loop')
    
    
    
    return light_df

#%% Density Analysis

def test_kde_density():
    '''
    Test script to use KDE model of density distribution of satellite catalog
    in orbital angular momentum space p(hx,hy,hz).
    
    Fit a KDE model from the current object catalog TLE_latest.txt.
    Extract coords of a test object COSMOS 2251.
    Query the log-density at this location.
    
    '''

    # Load satellite catalog
    df = load_satellites()
    
    # Compute density
    result = compute_density(df)
    df['p_hxhyhz'] = result
    
    # Plot histogram of density
    # plt.hist(df.p_hxhyhz,bins=100)
    
    # Fit a kde model
    kde = fit_density_kde()
    
    # Extract an entry to test
    # Cosmos 2251
    Xq = df[['hx','hy','hz']][df.Name == 'COSMOS 2251'].to_numpy() # array([[ 39560.02609224, -32755.51934044,  14688.0599049 ]])
    pq = float(df['p_hxhyhz'][df.Name == 'COSMOS 2251']) # Density of this object p = 4.535157446596887
    # Query density at point
    Xqt = normalize_coords(Xq) # Normalize coords of query point (kde fit on normalized hx,hy,hz coords)
    p = float(kde.score_samples(Xqt)) # Interpolate log density at this location. Confirm that it does equal dq computed above

    return

