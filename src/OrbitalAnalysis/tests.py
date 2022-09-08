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
from DIT import *
from Ephem import *
from Events import *
from VisualMagnitude import *
# from GmatScenario import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pdb

#%% DIT
# Main module

def test_run_analysis():
    ''' 
    Test of main DIT analysis workflow 
    1. Generating SPK files for a user-defined satellite.
    2. Comopute satellite lighting conditions.
    3. Compute lighting conditions for each station in SSR and SSRD networks.
    4. Compute optical trackability metrics.
    5. Compute radar trackability metrics.
    6. Compute optical detectability metrics.
    '''
    rE = 6371. # Radius of Earth (km)
    
    # INPUTS
    # NORAD = 25544 # NORAD ID of satellite e.g. 25544 for ISS
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    step = 10. # Time step (sec)
    # Satellite-Maya Example
    # sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
    #             "SMA": 6963.0, "ECC": 0.0188, "INC": 97.60,
    #             "RAAN": 308.90, "AOP": 81.20, "TA": 0.00,
    #             "rcs": 0.55}
    
    # # Steller Outer Shell Rating
    # sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
    #             "SMA": 1000+rE, "ECC": 0.0, "INC": 60.00,
    #             "RAAN": 0.0, "AOP": 0.0, "TA": 0.00,
    #             "rcs": 3.3385}
    
    # Steller Mid Shell Rating
    sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
                "SMA": 925.+rE, "ECC": 0.0, "INC": 80.00,
                "RAAN": 0.0, "AOP": 0.0, "TA": 0.00,
                "rcs": 1.9888}
    
    # # Steller Inner Shell Rating
    # sat_dict = {"DateFormat": "UTCGregorian", "Epoch": '26 Oct 2020 16:00:00.000',
    #             "SMA": 850.+rE, "ECC": 0.0, "INC": 60.00,
    #             "RAAN": 0.0, "AOP": 0.0, "TA": 0.00,
    #             "rcs": 1.9888}
    
    # Run Analysis
    results = run_analysis(sat_dict,start_date,stop_date,step)
    
    # Known errors:
    # SPICE(KERNELPOOLFULL)
    # Solution: Empty kernel pool. Or restart IDE.
    
    return results

#%% Plots of Optical Detectability

def test_plot_visual_magnitude(station):
    ''' 
    Get the ephemerides of the satellite and sun in the ground station 
    Topocentric frame.
    '''
    
    # Generate ephemeris times
    # step = 10.
    # step = 5.
    # et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    
    # Convert start and stop dates to Ephemeris Time
    step = 5.
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)
    
    # # Generate ephemeris times
    # et = np.arange(start_et,stop_et,10); et = np.append(et,stop_et)
    
    # 2. Compute satellite lighting intervals
    print('\nComputing Satellite Lighting intervals', flush=True)
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    satlight1 = spice.wnunid(satlight,satpartial) # Full or partial light
    
    # 3. Compute SSR and SSRD Station Lighting and access
    # SSR: use min_el = 30 deg (120 deg cone angle from zenith)
    # SSRD: use min_el = 5 deg (since targeted at satellite)
    # dflos_ssr, dfvis_ssr, dfcomblos_ssr, dfcombvis_ssr = compute_station_access('SSR',start_et,stop_et,satlight1,30.,save=True)
    dflos_ssrd, dfvis_ssrd, dfcomblos_ssrd, dfcombvis_ssrd = compute_station_access('SSRD',start_et,stop_et,satlight1,5.,save=True)
    
    # Extract access for station of interest
    df = dfvis_ssrd[dfvis_ssrd['Station'] == station]
    
    # Create time vector sampling all access periods
    step = 10 # Timestep (s)
    et = [] # Empty array
    for ind,row in df.iterrows():
        et_new = np.arange(row['Start']-2*step,row['Stop']+2*step,step)
        et += list(et_new)
    et = np.array(et) # Convert to numpy array
    et = np.sort(np.unique(et))  # Sort array and remove duplicates
    
    # Get Topocentric ephemeris relative to this station at these times
    dftopo = get_ephem_TOPO(et,groundstations=[station])
    dftopo = dftopo[0] # Select first station
    # Get visible access for this station
    dfa = df[df.Station == station]
    
    # Satellite radius
    # Outer: 1.6x1.0x0.7m = 1.12m^3 -> Rmean = (Vol*3/(4*pi))^(1/3) = 0.644
    # From area r = sqrt(A/pi) = 0.713 m
    Rsat = 0.713 # Outer shell sat
    
    # Compute visual magnitude
    p = 0.175 # Albedo (17.5%)
    msat = compute_visual_magnitude(dftopo,Rsat,p=p,k=0.12) # Lambertian phase function
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=p,k=0.12,lambertian_phase_function=False) # Constant phase function v(alpha)=1
    # Add to dataframe
    dftopo.insert(len(dftopo.columns),'Vmag',list(msat))
    dftopo.insert(len(dftopo.columns),'Vmag2',list(msat2))
    # Save to file
    out_dir = get_data_home()/'DITdata'
    filename = out_dir/'BestAccess.csv'
    dftopo.to_csv(str(out_dir/filename),index=False)
    
    # Plot results
    out_dir = get_data_home()/'DITdata'
    plot_visibility(dftopo,
                    title="Optical Detectability Station {}".format(station))
    
    
    return dftopo



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
    # step = 10.
    # step = 5.
    # et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    
    # Convert start and stop dates to Ephemeris Time
    step = 5.
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
    scenario_duration = stop_et - start_et # Length of scenario (s)
    
    # Generate ephemeris times
    et = np.arange(start_et,stop_et,10); et = np.append(et,stop_et)
    
    # Get Topocentric observations
    
    dftopo = get_ephem_TOPO(et,groundstations=['SSRD-5'])[0]
    
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


def validate_optical_access(case):
    '''
    Generate ephemeris data of particular passes from STK validation data.
    
    optical_chain_1_Access_Data.csv give the computed visible access times for
    the SSR network satellites (used for optical trackability metric). Check the
    sun elevation angle at the start and end of these access times to infer
    lighting constraints.
    
    Exampe pass to test: 10th pass of Station 1 (line 11).
    This pass does not match values computed using -0.5 elevation geometric
    constraint.
    
    '''
    
    # Load LSK
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    
    # Eclipse parameters
    front = "EARTH" # Name of occulting body
    fshape = "ELLIPSOID" # Type of shape model for front body (POINT, ELLIPSOID, DSK/UNPRIORITIZED)
    fframe = "ITRF93" #"IAU_EARTH" or "ITRF93" # # Body-fixed frame of front body
    back =  "SUN" # Name of occulted body
    bshape = "ELLIPSOID" # Type of shape model for back body
    bframe = "IAU_SUN" # Body-fixed frame of back body (empty)
    # abcorr = "NONE" # Aberration correction flag
    abcorr = "CN"
    obsrvr = "SSR-1"  # Observer
    step = 5. # Step size (s)

    
    # First test case ---------------------------------------------------------
    # Start and stop times of pass
    
    if case == 1:
        # Test case 1
        start_date = '2020-11-08 19:23:11.316'
        stop_date = '2020-11-08 19:27:40.092'
        tstep = 10 # Step size (s)
    elif case == 2:
        # Test case 2
        # Line #12 in csv value is a 14 second access that is not found.
        start_date = '2020-11-09 19:28:34.197'
        stop_date = '2020-11-09 19:28:48.888'
        tstep = 1 # Step size (s)
    elif case == 3:
        # Test case 3 (or 2b)
        # The equivalent access period for case 2, without lighting constraints
        start_date = '2020-11-09 19:28:34.212' # 2020-11-09 19:28:34.212
        stop_date = '2020-11-09 19:33:13.470' # 2020-11-09 19:33:13.470
        tstep = 1 # Step size (s)
        

    # Convert start and stop dates to Ephemeris Time
    start_et = spice.str2et(start_date)
    stop_et = spice.str2et(stop_date)
       
    # Generate vector of et
    et = np.arange(start_et,stop_et,tstep)
    et = np.append(et,stop_et)
    # Extended et
    et2 = np.arange(start_et-30*60,stop_et+30*60,tstep)
    et2 = np.append(et2,stop_et+30*60)
    
    # Get ephemeris in Topocentric frame
    dftopo = get_ephem_TOPO(et,groundstations=['SSR-1'])[0] # Access
    dftopo2 = get_ephem_TOPO(et2,groundstations=['SSR-1'])[0] # Extended time frame
    
    # Compute satellite lighting
    satlight, satpartial, satdark = find_sat_lighting(start_et-30*60,stop_et+30*60)
    satlight = window_to_dataframe(satlight)
    
    # Compute station lighting
    cnfine = spice.cell_double(2) # Initialize window of interest
    spice.wninsd(start_et-30*60, stop_et+30*60, cnfine ) # Insert time interval in window
    
    # Full occulation (dark or umbra)
    full = spice.cell_double(2*100) # Initialize result
    full = spice.gfoclt ( "FULL",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, full          )
    full = window_to_dataframe(full)
    
    # Full occulation (dark or umbra)
    partial = spice.cell_double(2*100) # Initialize result
    partial = spice.gfoclt ( "PARTIAL",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, partial          )
    partial = window_to_dataframe(partial)
    
    # light,dark = find_station_lighting(start_et-30*60,stop_et+30*60,station='SSR-1',method='eclipse')
    light, dark = find_station_lighting(start_et-30*60,stop_et+30*60,station='SSR-1',ref_el=0.268986)
    light = window_to_dataframe(light)
    dark = window_to_dataframe(dark)
    
    pdb.set_trace()
    
    # Plot satellite and sun elevations
    fig, ax1 = plt.subplots(1,1)
    fig.suptitle('SSR-1 Topocentric Coordinates')
    plt.xlabel("Epoch (ET)")
    plt.ylabel("Elevation (deg)")
    ax1.plot(dftopo2['ET'],np.rad2deg(dftopo2['Sun.El']),':k',label='Sun El') # Sun
    ax1.plot(dftopo['ET'],np.rad2deg(dftopo['Sun.El']),'-k',label='Sun El') # Sun
    ax1.plot(dftopo2['ET'],np.rad2deg(dftopo2['Sat.El']),':b',label='Sat El') # Sat
    ax1.plot(dftopo['ET'],np.rad2deg(dftopo['Sat.El']),'-b',label='Sat El') # Sat
    # try:
    #     ax1.add_patch(Rectangle((full['Start'].iloc[0], -10), full['Stop'].iloc[0] - full['Start'].iloc[0], 100,facecolor = 'grey',alpha=0.2))
    # except:
    #     pass
    # try:
    #     ax1.add_patch(Rectangle((partial['Start'].iloc[0], -10), partial['Stop'].iloc[0] - partial['Start'].iloc[0], 100,facecolor = 'blue',alpha=0.2))
    # except:
    #     pass
    try:
        ax1.add_patch(Rectangle((light['Start'].iloc[0], -10), light['Stop'].iloc[0] - light['Start'].iloc[0], 100,facecolor = 'yellow',alpha=0.2))
    except:
        pass
    ax1.add_patch(Rectangle((satlight['Start'].iloc[0], 100), satlight['Stop'].iloc[0] - satlight['Start'].iloc[0], 10,facecolor = 'yellow',alpha=0.2))
    # ax1.add_patch(Rectangle((satdark['Start'].iloc[0], 100), satdark['Stop'].iloc[0] - satdark['Start'].iloc[0], 10,facecolor = 'black',alpha=0.2))
    ax1.legend(loc="upper left")
    fig.show()
    
    # Analysis 1
    # For this access, the sun is rising at the end of the access period. The
    # period ends when the satellite elevation drops below the 30 deg threshold.
    # The sun elevation at this time is -0.1 deg.
    # From this, we can infer that STK cuts off sun elevations at 0 deg.

    # Analysis 2:
    # This access starts at sat el = 30 deg, and ends at Sun.El = +0.27226 deg.
    # From this, we infer that partial sunlight is included, and only full
    # sunlight is excluded from the visible access.
    
    # Solution:
    # Compute visible access from LOS access windows, and remove station full
    # daylight windows.
    
    return dftopo



#%%

# TODO: Find all objects with associated debris.
# Find a clustering metric that minimizes the confusion between objects.
# i.e. groups of debris are clustered tight together, and well separated from
# other objects.
