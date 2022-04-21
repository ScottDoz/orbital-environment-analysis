# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:59:04 2022

@author: scott

Tests
-----

Test functions

"""

from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *
from Overpass import *
from Ephem import *
from Events import *
from GmatScenario import *

#%% Satellite Data Loading
# Methods from SatelliteData.py module to load orbital elements of satellites.

def test_query_norad():
    ''' Load the satellite for a set of Norad IDs '''
      
    IDs = [25544, 41335]
    df = query_norad(IDs,compute_params=True)
    
    return df

def test_load_satellite_data():
    ''' Load the satellite data and compute orbital parameters '''
    
    df = load_satellites(group='all',compute_params=True)
    
    return df

def test_load_vishnu_data_single_month():
    ''' Load Vishnu experiment data for a single month '''
    
    df = load_vishnu_experiment_data(1)
    
    return df

def test_load_vishnu_data_list_month():
    ''' Load Vishnu experiment data for a list of months '''
    
    df = load_vishnu_experiment_data([1,2,3])
    
    return df

def test_load_vishnu_data_all():
    ''' Load Vishnu experiment data for all months '''
    
    df = load_vishnu_experiment_data('all')
    
    return df


#%% Visualization h-space
# Methods from Clustering.py and Visualization.py modules for visualizing the
# satellites in orbital momentum space.

def test_h_space_visualization():
    ''' 
    Generate a 3D scatter plot of the satellite catalog in specific orbital 
    angular momentum space (hx,hy,hz).
    '''
    
    # Load the data
    # df = load_satellites(group='all',compute_params=True)
    df = load_vishnu_experiment_data(1)
    
    # Generate clusters in (h,hz) coordiantes
    label = 'test_clusters' # Field name holding clusters
    features = ['h','hz']   # Fields to use in clustering 
    df = generate_Kmeans_clusters(df,label,features,n_clusters=15,random_state=170)
    
    # Generate plotly figure and render in browser
    plot_h_space_cat(df,'test_clusters')
    
    return

def test_h_space_timeseries_visualization():
    '''
    Generate a 3D scatter plot showing the trajectories of the satellite
    catalog in specific orbital angular momentum space.

    '''
    
    # Load all data from Vishnu (~200,000 points)
    df = load_vishnu_experiment_data('all')
    obj_list = list(df.NoradId.unique()) # List of unique objects (17,415)
    
    # Generate clusters in (h,hz) coordiantes
    label = 'test_clusters' # Field name holding clusters
    features = ['h','hz']   # Fields to use in clustering 
    df = generate_Kmeans_clusters(df,label,features,n_clusters=15,random_state=170)
    
    
    # Randomly sample ~ 1000 objects
    import random
    objs = random.sample(obj_list,1000)
    df = df[df.NoradId.isin(objs)]
    
    # Plot the scatter plot
    plot_h_space_cat(df,cat='test_clusters')
    
    return

def test_2d_scatter_visualization():
    '''
    Generate a 2D scatter plot selecting x,y,color coordinates from available 
    data. This example plots (h,hz) and color = inclination

    '''
    
    # Load the data
    df = load_satellites(group='all',compute_params=True)
    
    # Compute distances
    # target = 13552 # COSMOS 1408
    # df = compute_distances(df, target)
    
    # Generate figure and render in browser
    plot_2d_scatter_numeric(df,'h','hz','i')
    
    return

#%% DensityAnalysis

def test_kde_visualization():
    '''
    Generate a 2D scatter plot of the positions, with a heat map showing the
    density computed using Kernel Density Estimation. This example plots (h,hz).
    '''
    
    # Load the data
    df = load_satellites(group='all',compute_params=True)
    
    plot_kde(df,'h','hz')
    
    return

#%% DistanceAnalysis

def test_distances():
    '''
    Test the computation of various distance metrics from a target satellite of
    interest. 
    
    '''
    
    # NORAD ID of target
    target = 13552 # COSMOS 1408
    
    # Load data
    df = load_satellites(group='all',compute_params=True)
    
    # Compute distances to target
    df = compute_distances(df, target)
    
    # Find the closest objects to the target (removing any related debris)
    df[['name','a','e','dH']][~df.name.str.contains('COSMOS 1408')].sort_values(by='dH')
    
    # We find that two objects are fairly close in terms of dH metric to the target
    # (18421) COSMOS 1892, and (15495) SL-14 R/B.
    # These objects are in the COSMOS 1408 debris cluster.
    
    # Both these objects will theoreticaly have a large number of objects with 
    # which they may be confused.
    
    
    return df

#%% GmatScenario
# Methods from the GmatScenario.py module for generating analyzing access between
# a target satellite and a series of ground stations.

def test_configure_run_GMAT():
    '''
    Configure and run GMAT access script with user-defined sat and groundstation
    '''
    
    # Define satellite properties in dictionary
    sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 6963.0, "ECC": 0.0188, "INC": 97.60,
                "RAAN": 308.90, "AOP": 81.20, "TA": 0.00}
    
    # sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
    #             "SMA": 6963.0, "ECC": 0.0188, "INC": 60.60,
    #             "RAAN": 308.90, "AOP": 81.20, "TA": 0.00}
    
    # # Define groundstation properties in dictionary
    # gs_dict = {"Location1": 72.03, "Location2": 123.435, "Location3": 0.0460127,
    #            "MinimumElevationAngle": 0.0}
    
    # # Define groundstation properties in dictionary
    # gs_dict = {"Location1": 0.00, "Location2": 123.435, "Location3": 0.0460127,
    #             "MinimumElevationAngle": 0.0}
    
    # DSS-43
    gs_dict = {"StateType":'Cartesian',"HorizonReference":'Ellipsoid',
                "Location1": -4460.894917, "Location2": 2682.361507, "Location3": -3674.748152,
                "MinimumElevationAngle": 0.0}
    # # DSS-43    399043  70m     -4460894.917    +2682361.507    -3674748.152
    # (35.402, 148.98, 0.6893)
    
    # Define propagation settings
    duration = 30. # Propagation duration (days)
    timestep = 30. # Propagation timestep (s)
    
    
    # Run
    configure_run_GMAT(sat_dict, gs_dict, duration, timestep)
    
    return

def test_load_GMAT_results():
    ''' Load the results of the GMAT simulation to Pandas dataframes '''
    
    # Load access data
    dfa = load_access_results()
    
    # Load satellite eclopse data
    dfec = load_sat_eclipse_results()
    
    # # Load observation data
    # dfobs = load_ephem_report_results()
    
    return dfa, dfec


#%% Overpass
# Main module

def test_analysis():
    '''
    Run an analysis. 
    Generating SPK files for a user-defined satellite.
    Run optical analysis to compute optical metrics for average duration, interval.
    '''
    
    # INPUTS
    NORAD = 25544 # NORAD ID of satellite e.g. 25544 for ISS
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    step = 10. # Time step (sec)
    sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 6963.0, "ECC": 0.0188, "INC": 97.60,
                "RAAN": 308.90, "AOP": 81.20, "TA": 0.00}
    
    
    # Create Sat.bsp
    # create_files(NORAD,start_date,stop_date,step,method='tle') # From TLE
    create_files(sat_dict,start_date,stop_date,step,method='two-body') # From TLE
    
    # Optical anayslis
    results = optical_analysis(start_date,stop_date,step)
    
    return results


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


#%% Events. Eclipses and Station Lighting

def test_station_lighting():
    
    # Select station
    # gs = 'DSS-43'
    gs = 'SSR-1'
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=False)
    start_et = et[0]
    stop_et = et[-1]
    # light, dark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-0.25)
    light, dark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-6.)
    
    result = dark
    
    # Print Dark Intervals
    print('Station Dark times')
    print('Start (UTC)                 Stop (UTC)                  Duration (s)')
    count = spice.wncard( result ) # Count of intervals
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    for i in range(count):
        beg,end = spice.wnfetd (result, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
    print('')
    
    
    # Print Sunlit Intervals
    result = light
    print('Station Light times')
    print('Start (UTC)                 Stop (UTC)                  Duration (s)')
    count = spice.wncard( result ) # Count of intervals
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    for i in range(count):
        beg,end = spice.wnfetd (result, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
    
    return dark, light

def test_sat_lighting():
    ''' Get the eclipse times of a satellite '''
    
    # Confirmed matches with output of GMAT EclipseLocator.
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=False)
    start_et = et[0]
    stop_et = et[-1]
    light, partial, dark = find_sat_lighting(start_et,stop_et)
    
    # Join full and partial results
    # resunion = spice.wnunid(light, partial)
    
    
    for lighting in ['Dark','Partial','Full']:
        
        # Select result
        if lighting == 'Full':
            result = light
        elif lighting == 'Partial':
            result = partial
        elif lighting == 'Dark':
            result = dark
        
        # Print results
        print('{} '.format(lighting))
        print('Start (UTC)                 Stop (UTC)                  Duration (s)')
        count = spice.wncard( result ) # Count of intervals
        # TIMFMT = "YYYY-MON-DD HR:MN:SC.###### (TDB) ::TDB ::RND" # Time format for printing
        TIMFMT = "DD-MON-YYYY HR:MN:SC.###### ::UTC ::RND" # Time format for printing
        
        TIMLEN = 41
        if count==0:
            print('No occultation was found.\n')
            continue
        
        for i in range(count):
            beg,end = spice.wnfetd (result, i)
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            
            # print( "Interval {}".format( i + 1));
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
            # print( " \n" );
        print('')
    
    return result

def test_access():
    
    # TODO: Check receive vs transmit results
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_et = et[0]
    stop_et = et[-1]
    
    # Select groundstation
    # gs = 'DSS-43'
    gs = 'SSR-1'
    # gs = 'SSR-2'    
    
    # Compute line-of-sight access intervals
    los_access = find_access(start_et,stop_et,station=gs)
    dflos = window_to_dataframe(los_access)
    
    # Compute station lighting intervals
    # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs, ref_el=-0.25)
    gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs, ref_el=-6.)
    
    # Compute satellite lighting intervals
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # Compute visible (constrained) access intervals
    access = constrain_access_by_lighting(los_access,gslight,satdark)
    dfaccess = window_to_dataframe(access)
    
    # Print results
    
    # Line-of-sight access (no constraints)
    count = spice.wncard( los_access ) # Count of intervals
    print('Line-of-Sight Access')
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    print('# Start (UTC)                 Stop (UTC)                  Duration (s)')
    for i in range(count):
        beg,end = spice.wnfetd (los_access, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {} {}".format(i+1, begstr, endstr, str(dur)) );
    print('')
    print('Min Duration {} s'.format(dflos.Duration.min()))
    print('Max Duration {} s'.format(dflos.Duration.max()))
    print('Mean Duration {} s'.format(dflos.Duration.mean()))
    print('Total Duration {} s'.format(dflos.Duration.sum()))
    print('')
    print('')
    
    # Visible access (no constraints)
    count = spice.wncard( access ) # Count of intervals
    print('Visible Access')
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    print('# Start (UTC)                 Stop (UTC)                  Duration (s)')
    for i in range(count):
        beg,end = spice.wnfetd (access, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {} {}".format(i+1, begstr, endstr, str(dur)) );
    print('')
    print('Min Duration {} s'.format(dfaccess.Duration.min()))
    print('Max Duration {} s'.format(dfaccess.Duration.max()))
    print('Mean Duration {} s'.format(dfaccess.Duration.mean()))
    print('Total Duration {} s'.format(dfaccess.Duration.sum()))
    print('')
    
    return access


#%% Plot Overpass

def test_plot_access():
    
    gs = 'DSS-43'
    # gs = 'SSR-30'
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_et = et[0]
    stop_et = et[-1]
    
    # Compute satellite lighting intervals
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # Compute station lighting intervals
    gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
    
    # Compute line-of-sight access intervals
    access = find_access(start_et,stop_et,station=gs)
    
    # Plot
    plot_access_times(access,gslight,gsdark,satlight, satpartial, satdark)
    
    
    return

def test_plot_overpass():
    
    # Load access
    dfa = load_access_results()
    
    # Generate ephemeris times
    # step = 10.
    step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et)[0]
    # dftopo['Sat.El'] = np.rad2deg(dftopo['Sat.El'])
    # dftopo['Sat.Az'] = np.rad2deg(dftopo['Sat.Az'])
    
    # Plot
    # plot_overpass(dftopo, dfa)
    plot_overpass_magnitudes(dftopo, dfa)
    
    return

def test_plot_visual_magnitude():
    
    
    # Generate ephemeris times
    et = generate_et_vectors_from_GMAT_coverage(30., exclude_ends=True)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et)[0]
    
    # Compute Visual magnitudes
    Rsat = 1 # Radius of satellite (m)
    msat = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12) # With airmass
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,include_airmass=False) # With airmass
    
    # Generate plots
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Visual Magnitude')
    # Magnitude vs elevation
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat,'.b')
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat2,'.k')
    ax1.set_xlabel("Elevation (deg)")
    ax1.set_ylabel("Visual Magnitude (mag)")
    ax1.invert_yaxis() # Invert y axis
    # Az/El
    ax2.plot(dftopo['ET'],msat,'-b')
    ax2.plot(dftopo['ET'],msat2,'-k')
    ax2.invert_yaxis() # Invert y axis
    ax2.set_xlabel("Epoch (ET)")
    ax2.set_ylabel("Visual Magnitude (mag)")
    fig.show()
    
    
    return




#%%

# TODO: Find all objects with associated debris.
# Find a clustering metric that minimizes the confusion between objects.
# i.e. groups of debris are clustered tight together, and well separated from
# other objects.
