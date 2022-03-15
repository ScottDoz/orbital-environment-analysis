# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:40:50 2022

@author: scott

Overpass Module
-------------------

- compute access intervals between target satellite and groundstations using GMAT
- analyse statistics of the overpasses

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

import pdb

from astropy.time import Time
import spiceypy as spice

# Module imports
from SatelliteData import query_norad
from utils import get_data_home

# Load GMAT
from load_gmat import *



#%% Ground Stations

def get_groundstations():
    ''' Get the lat/long/alt coordinates of the custom groundstation network '''
    
    # Hardcoded facility positions
    facility_positions = [(-37.603, 140.388, 0.013851), (-45.639, 167.361, 0.344510), (-44.040, -176.375, 0.104582),
                          (-43.940, -72.450, 0.075715), (-51.655, -58.681, 0.127896), (-34.070, 19.703, 0.416372),
                          (-34.285, 115.934, 0.134033), (-49.530, 69.910, 0.199042), (18.872, -103.290, 0.735454),
                          (-15.096, -44.836, 0.697796), (-15.099, 15.875, 1.365027),
                          (-15.818, 45.893, -0.007232), (5.159, -53.637, 0.037340), (7.612, 134.631, 0.138556),
                          (-15.531, 134.143, 0.196179), (-22.500, 113.989, 0.068118), (-7.261, 72.376, -0.064980),
                          (-15.273, 166.878, 0.196300), (-13.890, -171.938, 0.392109), (18.532, -74.135, 0.291372),
                          (-9.798, -139.073, 0.845423), (-27.128, -109.355, 0.149995), (-7.947, -14.370, 0.216315),
                          (6.890, 158.216, 0.311603), (16.899, 102.561, 0.167567), (15.097, -15.726, 0.087358),
                          (14.846, 14.217, 0.359288), (14.846, 44.914, 2.071660), (17.396, 76.263, 0.382021),
                          (19.787, -155.658, 1.517667), (-15.450, -73.848, 4.202630), (44.676, -105.521, 1.249258),
                          (44.554, -75.459, 0.070607), (40.506, -124.123, 0.002242),
                          (43.040, -8.992, 0.411682), (47.014, -53.061, 0.191380), (45.481, 15.224, 0.252010),
                          (44.891, 44.590, 0.085764), (44.537, 75.371, 0.340541), (44.384, 104.729, 1.223731),
                          (45.271, 135.576, 0.399098), (53.312, 159.728, 0.536244), (55.395, -162.156, 0.673701),
                          (70.024, -162.191, 0.013845), (69.175, 18.258, 0.314617), (67.922, -103.469, -0.005155),
                          (74.757, -46.014, 2.651167), (72.423, 75.289, 0.011348), (71.372, 136.045, 0.010589)]
    
    # Load into dataframe
    df = pd.DataFrame()
    df['Lat'] = [x[0] for x in facility_positions]  # Latitude (deg)
    df['Long'] = [x[1] for x in facility_positions] # Longitude (deg)
    df['El'] = [x[2] for x in facility_positions]  # Elevation (km)
    
    # Ensure 0 < Long < 360
    df['Long'][df.Long < 0.] += 360.
    
    return df


#%% Compute access using GMAT

def compute_access_GMAT(sat_dict, gs_dict, duration=10., timestep=60., out_dir=None):
    '''
    Compute access between a satellite and groundstation using GMAT.
    
    This script configures a template GMAT script with custom satellite and
    groundstation parameters, runs the script and saves data into a defined
    output directory.

    Parameters
    ----------
    sat_dict : dict
        Dictionary containing key-value pairs of properties to change in the satellite.
        Keys include "SMA","ECC","INC","RAAN","AOP","TA","Epoch","DateFormat"
        see: https://documentation.help/GMAT/Spacecraft.html
    gs_dict : dict
        Dictionary containing key-value pairs of properties to change in the groundstation.
        Keys include: "Location1","Location2","Location3","MinimumElevationAngle"
        see: https://documentation.help/GMAT/GroundStation.html
    duration: float
        Duration of propagation (days).
    
    Returns
    -------
    None.

    '''
    
    # Set default output directory
    if out_dir is None:
        out_dir = Path.home()/'satellite_data'/'Data'/'GMATscripts'/'Access'
        # Check if directory exists and create
        if not os.path.exists(str(out_dir)):
            os.makedirs(str(out_dir))
    else:
        # Convert string to Path
        if type(out_dir)==str:
            out_dir = Path(out_dir)
    
    # Load a script into the GMAT API
    print('Loading template_mission.script', flush=True)
    gmat.LoadScript("template_mission.script")
    
    # Retrieve and display the results of the run
    sat = gmat.GetObject("Sat") # Satellite
    gs1 = gmat.GetObject("GS1") # GroundStation 1
    prop = gmat.GetObject("DefaultProp")
    fm = gmat.GetObject("DefaultProp_ForceModel")
    ecl = gmat.GetObject("EclipseLocator1") # Eclipse locator
    cl = gmat.GetObject("ContactLocator1") # Eclipse locator Sat-GS1
    eph = gmat.GetObject("EphemerisFile1") # Ephemeris of Sat
    rep = gmat.GetObject("ObservationReportFile1") # Report of Sat observations from gs
    
    # Variables
    var_prop_time_days = gmat.GetObject("prop_time_days") # Variable Propagation time (days) 
    
    # # Print the variables
    # # sat.Help()
    # print("Keplerian:\n ", sat.GetKeplerianState())
    # print("Cartesian:\n ", sat.GetCartesianState())
    # # gs1.Help()
    
    
    # Change the coordinates of the satellite
    
    # Satellite
    print('Setting Satellite properties', flush=True)
    for i, (key, value) in enumerate(sat_dict.items()):
        sat.SetField(key,value) # E.g. sat.SetField("SMA", 6963)
    start_date = sat.GetField("Epoch")
    
    # Ground station  
    print('Setting Groundstation properties', flush=True)
    for i, (key, value) in enumerate(gs_dict.items()):
        gs1.SetField(key,value)
    # gs1.SetField('Location1',72.03)     # Latitude (deg) = 72.03,
    # gs1.SetField('Location2',123.435)   # Longitude (deg) - 123.435 deg
    # gs1.SetField('Location3',0.0460127) # Altitude (km) = 0.0460127 km
    # gs1.SetField('MinimumElevationAngle',0) # Min elevation (deg)
    
    # Change location of output files
    print('Setting output file locations', flush=True)
    # out_dir = r'C:\Users\scott\Documents\Repos\python_rough_code\GMAT\Scripting'
    ecl.SetField('Filename', str(out_dir/'EclipseLocator1.txt'))
    cl.SetField('Filename', str(out_dir/'ContactLocator1.txt'))
    eph.SetField('Filename', str(out_dir/'EphemerisFile1.bsp'))
    rep.SetField('Filename', str(out_dir/'ObservationReportFile1.txt'))
    
    # Propagation time
    print('Setting propagation time', flush=True)
    var_prop_time_days.SetField('Value',duration)
    
    # Timestep
    print('Setting propagation time step', flush=True)
    prop.SetField('InitialStepSize',timestep)
    
    # Save the script and run the simulation
    
    # gmat.Initialize()
    print('Saving script', flush=True)
    gmat.SaveScript(str(out_dir/'configured_mission.script'))
    
    # Run the Script
    print('Running script ...', flush=True)
    loadstate = gmat.LoadScript(str(out_dir/'configured_mission.script'))
    runstate = gmat.RunScript()
    if runstate == True:
        print('Script ran successfully.')
        print('Output saved to {}'.format(str(out_dir)))
    else:
        print('Script failed to run.')
        
    # # Get scenario summary
    # sat = gmat.GetObject("Sat") # Satellite
    # gs1 = gmat.GetObject("GS1") # GroundStation 1
    # prop = gmat.GetObject("DefaultProp")
    # fm = gmat.GetObject("DefaultProp_ForceModel")
    
    # Get help summary of each object
    
    # List of Objects
    obj_list = gmat._gmat_py.ShowObjects() # List of objects
    
    
    # # Satellite
    # sat_help = gmat._gmat_py.GmatBase_Help(sat) # string
    # gs_help = gmat._gmat_py.GmatBase_Help(gs1)
    
    # Propagator properties
    # prop_dict = {field:prop.GetField(field) for field in ['Type','InitialStepSize',
    #                                                       'Accuracy','MinStep','MaxStep',
    #                                                       'MaxStepAttempts'] } 
    
    # TODO: Construct output file
    
    

    return

#%% Processing Access results

def load_access_results(DATA_DIR=None):
    '''
    Load access results from GMAT simulation. Returns a dataframe containint
    the Start Time, Stop Time, and Duration (s) of each access period.

    Parameters
    ----------
    DATA_DIR : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe containing access periods.

    '''
    
    # Set default data dir
    if DATA_DIR is None:
        DATA_DIR = get_data_home()/'GMATscripts'/'Access'
    else:
        if type(DATA_DIR)==str:
            DATA_DIR = Path(DATA_DIR)
    
    # Read the access file
    # (separated by at least two white spaces)
    df = pd.read_csv(str(DATA_DIR/'ContactLocator1.txt'), 
                     sep=r"\s\s+", engine='python',
                     skiprows=3)
    
    # Check for no access events
    if 'There are no contact events' in df.columns[0]:
        # No contact events. Return empty dataframe.
        df = pd.DataFrame(columns=['Access', 'Start', 
                                   'Stop', 'Duration'])
    else:
        # Rename columns
        df = df.rename(columns = {'Start Time (UTC)':'Start', 
                                  'Stop Time (UTC)':'Stop',
                                  'Duration (s)':'Duration'})
        
        # Remove text from final row
        if 'Number of events' in df[df.columns[0]].iloc[-1]:
            # Extract number of events
            num = int(df[df.columns[0]].iloc[-1][18:])
            df = df[:-1] # Remove final row
            # Confirm number of rows matches number of events
            assert(len(df) == num)
        
        # Insert an index column for numbering access periods
        df.insert(0, 'Access', df.index)
    
        # Convert times into iso
        # start = dfa['Start Time (UTC)'].to_list() # List of start times
        # start_et = spice.str2et(start) # Start times in ET
        # start_dt = spice.et2datetime(start_et) # Datetime
        # t = Time(start_dt, format='datetime', scale='utc') # Astropy Time object
        # start_iso = t.iso # Times in iso
        
        # Convert start time to iso
        df['Start'] = pd.to_datetime(df['Start'])
        df['Stop'] = pd.to_datetime(df['Stop'])
        # datetime.datetime.strptime(start, '%d %b %Y %H:%M:%S.%f')
    
    return df

def load_eclipse_results(DATA_DIR=None):
    
    
    # Set default data dir
    if DATA_DIR is None:
        DATA_DIR = get_data_home()/'GMATscripts'/'Access'
    else:
        if type(DATA_DIR)==str:
            DATA_DIR = Path(DATA_DIR)
    
    # Read the access file
    # (separated by at least two white spaces)
    df = pd.read_csv(str(DATA_DIR/'EclipseLocator1.txt'), 
                      sep=r"\s\s+", engine='python',skiprows=2)
    
    # Check for no eclipse events
    if 'There are no eclipse events' in df.columns[0]:
        # No eclipse events. Return empty dataframe.
        df = pd.DataFrame(columns=['Start', 'Stop', 
                                   'Duration', 'OccBody', 'Type', 
                                   'EventNumber', 'TotalDuration'])
    else:
        
        # Rename columns
        df = df.rename(columns = {'Start Time (UTC)':'Start', 
                                  'Stop Time (UTC)':'Stop', 
                                   'Duration (s)':'Duration',
                                   'Total Duration (s)':'TotalDuration',
                                   'Occ Body':'OccBody',
                                   'Event Number':'EventNumber',
                                   })
        
        # Remove summary values (bottom 4 lines)
        # num_events = int(df[df.columns[0]].iloc[-4][29:]) # Number of individual events
        # num_total_events = int(df[df.columns[0]].iloc[-3][29:]) # Number of individual events
        df = df[:-4]
        
        # Convert start time to iso
        df['Start'] = pd.to_datetime(df['Start'])
        df['Stop'] = pd.to_datetime(df['Stop'])
    
    
    return df



def load_observation_results(DATA_DIR=None):
    
    # Set default data dir
    if DATA_DIR is None:
        DATA_DIR = get_data_home()/'GMATscripts'/'Access'
    else:
        if type(DATA_DIR)==str:
            DATA_DIR = Path(DATA_DIR)
    
    # Read the access file
    # (separated by at least two white spaces)
    df = pd.read_csv(str(DATA_DIR/'ObservationReportFile1.txt'), 
                      sep=r"\s\s+", engine='python')
    
    # Create a column with datetime objects
    df.insert(0, 'Epoch', pd.to_datetime(df['Sat.UTCGregorian']))
    
    # Elevation
    # 'Sat.TopoGS1.DEC' is the declination or elevation of the spacecraft above
    # the local horizon
    # Confimed computation below
    df['El'] = np.rad2deg(np.arctan2(df['Sat.TopoGS1.Z'],
                        np.sqrt(df['Sat.TopoGS1.X']**2 + df['Sat.TopoGS1.Y']**2))) # Elevation (from xy plane)
    # Confirmed == Sat.TopoGS1.DEC
    # >> df[['Sat.TopoGS1.DEC','El']]
    
    
    # Azimuth
    # 'Sat.TopoGS1.RA' measures the right ascension of the satellite in local SEZ
    # coordinates, which gives the angle from south direction (x axis) measured
    # anti-clockwise.
    theta = np.rad2deg(np.arctan2(df['Sat.TopoGS1.Y'],df['Sat.TopoGS1.X'])) # From S measured anti-clockwise
    # Confirmed theta == Sat.TopoGS1.RA
    # >> df[['Sat.TopoGS1.RA','Az']]
    
    # Compute the Azimuth in an Alt/Az system (measured clock-wise from North)
    df['Az'] = 180 - theta
    
    
    # TODO: Compute proper motion in az/el
    
    
    # # Load access data
    # dfa = load_access_results(DATA_DIR=DATA_DIR)
    # Add access period label to each timestep
    
    
    return df

def compute_visual_magnitude(df):
    
    # Compute satellite apparent magnitude
    # https://www.eso.org/~ohainaut/satellites/22_ConAn_Bassa_aa42101-21.pdf
    # m_sat = m0 - 2.5*log10(p*Rsat^2) + 5*log10(dsat0*dsat)
    #         - 2.5*log10( v(alpha0) ) + k*X
    #
    # where:
    # m0 = -26.76 is the Solar V-band magnitude at Earth
    #
    # p*Rsat^2 is the photometric crossection
    # p = satellite geometric albedo
    # Rsat = radius of the (spherical) satellite
    #
    # dsat0 = distance from satellite to sun
    # dsat = distance from observer to satellite
    #
    # # alpha0 = solar phase angle
    # v(alpha0) = correction for solar phase angle (set at 1 to remove term)
    #
    # k = extinction coefficient (mag per unit airmass) = 0.12 in V-band
    # X = 1/sin(El) = airmass in the plane-parallel approximation
    # El = elevation above horizon
    
    
    # TODO: Compute airmass using astropy.coordinates.AltAz
    # https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html
    
    # Use airmass to compute atmospheric extinction
    # m(X) = m0 + k*X
    # where X = airmass, m=magnitude, k=extinction coefficient (mags/airmass)
    # see: https://warwick.ac.uk/fac/sci/physics/research/astro/teaching/mpags/observing.pdf
    
    
    return
