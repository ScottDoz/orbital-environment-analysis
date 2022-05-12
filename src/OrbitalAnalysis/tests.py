# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:59:04 2022

@author: scott

Tests
-----

Test functions

"""

from SatelliteData import *
# from Clustering import *
# from DistanceAnalysis import *
# from Visualization import *
from Overpass import *
from Ephem import *
from Events import *
# from GmatScenario import *



#%% Overpass
# Main module

def test_run_analysis():
    ''' Test of main DIT analysis workflow '''
    
    # INPUTS
    # NORAD = 25544 # NORAD ID of satellite e.g. 25544 for ISS
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    step = 10. # Time step (sec)
    sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 6963.0, "ECC": 0.0188, "INC": 97.60,
                "RAAN": 308.90, "AOP": 81.20, "TA": 0.00}
    
    # Run Analysis
    results = run_analysis(sat_dict,start_date,stop_date,step)
    
    
    return results


def test_analysis():
    '''
    Depreciated!
    
    Run an analysis. 
    Generating SPK files for a user-defined satellite.
    Run optical analysis to compute optical metrics for average duration, interval.
    '''
    
    # INPUTS
    # NORAD = 25544 # NORAD ID of satellite e.g. 25544 for ISS
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    step = 10. # Time step (sec)
    sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 6963.0, "ECC": 0.0188, "INC": 97.60,
                "RAAN": 308.90, "AOP": 81.20, "TA": 0.00}
    
    
    # Create Sat.bsp
    # create_ephem_files(NORAD,start_date,stop_date,step,method='tle') # From TLE
    create_ephem_files(sat_dict,start_date,stop_date,step,method='two-body') # From TLE
    
    # Trackability anayslis
    optical_results, radar_results, station_results = trackability_analysis(start_date,stop_date,step)
    
    
    return optical_results, radar_results, station_results




#%% Ephemerides

def test_create_satellite_ephem():
    
    # Define satellite properties in dictionary
    sat = 25544 # NORAD ID (ISS)
    
    # Generate ephemeris times
    cov = get_GMAT_coverage()
    step = 10.
    start_et = cov['start_et']
    stop_et = cov['stop_et']
    
    # Create ephem
    create_satellite_ephem(sat,start_et,stop_et,step,method='tle')
    
    return


def test_get_ephem_TOPO():
    ''' 
    Get the ephemerides of the satellite and sun in the ground station 
    Topocentric frame.
    '''
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et)[0]
    
    return dftopo

def test_compare_ephemerides():
    ''' Compare the ephemeris output from spice and GMAT reports '''
    
    # Load the GMAT ephemeris output
    dfobs = load_ephem_report_results()
    # Remove first and last timestep
    dfobs = dfobs[1:-1]
    
    # Extract ephemeris times
    et = dfobs.ET.to_numpy()
    
    # Compute ephemeris using Spice
    dfitfr = get_ephem_ITFR(et) # Earth fixed
    dftopo = get_ephem_TOPO(et)[0] # Topocentric frame
    
    # Rotate Spice Topocentric frame (NWU) to GMAT equivalent (SEZ)
    # (x -> x, y -> -y)
    dftopo['Sat.X'] = -dftopo['Sat.X']
    dftopo['Sat.Y'] = -dftopo['Sat.Y']
    # Convert Spice ephem angles to deg
    dftopo['Sat.Az'] = np.rad2deg(dftopo['Sat.Az'])
    dftopo['Sat.El'] = np.rad2deg(dftopo['Sat.El'])
    
    
    # Merge dataframes
    df = pd.merge(dfobs, dfitfr, how='left', left_on='ET', right_on='ET')
    
    # Plot Earth-Fixed Positions
    fig, (ax1,ax2) = plt.subplots(2, 1)
    fig.suptitle('Earth-Fixed Coordinates (GMAT-Spice)')
    plt.xlabel("Epoch (ET)")
    plt.ylabel("Earth-Fixed Position (km)")
    # Ground Station
    ax1.plot(df['ET'],df['GS1.EarthFixed.X']-df['DSS-43.X'],'-k',label='GS1 dX')
    ax1.plot(df['ET'],df['GS1.EarthFixed.Y']-df['DSS-43.Y'],'-b',label='GS1 dY')
    ax1.plot(df['ET'],df['GS1.EarthFixed.Z']-df['DSS-43.Z'],'-r',label='GS1 dZ')
    ax1.legend(loc="upper left")
    # Satellite
    ax2.plot(df['ET'],df['Sat.EarthFixed.X']-df['Sat.X'],'-k',label='Sat dX')
    ax2.plot(df['ET'],df['Sat.EarthFixed.Y']-df['Sat.Y'],'-b',label='Sat dY')
    ax2.plot(df['ET'],df['Sat.EarthFixed.Z']-df['Sat.Z'],'-r',label='Sat dZ')
    ax2.legend(loc="upper left")
    fig.show()
    
    # Topocentric
    df = pd.merge(dfobs, dftopo, how='left', left_on='ET', right_on='ET')
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('GS1 Topocentric Coordinates (GMAT-Spice)')
    plt.xlabel("Epoch (ET)")
    plt.ylabel("Topocentric Position (km)")
    # Satellite
    ax1.plot(df['ET'],df['Sat.TopoGS1.X']-df['Sat.X'],'-k',label='Sat dX')
    ax1.plot(df['ET'],df['Sat.TopoGS1.Y']-df['Sat.Y'],'-b',label='Sat dY')
    ax1.plot(df['ET'],df['Sat.TopoGS1.Z']-df['Sat.Z'],'-r',label='Sat dZ')
    ax1.set_ylabel("Topocentric Position (km)")
    ax1.legend(loc="upper left")
    # Az/El
    ax2.plot(df['ET'],df['Sat.TopoGS1.DEC']-df['Sat.El'],'-k',label='Sat dEl')
    ax2.plot(df['ET'],df['SatAz']-df['Sat.Az'],'-b',label='Sat dAz')
    ax2.set_ylabel("Angles (deg)")
    ax2.legend(loc="upper left")
    fig.show()
    
    
    return df






#%%

# TODO: Find all objects with associated debris.
# Find a clustering metric that minimizes the confusion between objects.
# i.e. groups of debris are clustered tight together, and well separated from
# other objects.
