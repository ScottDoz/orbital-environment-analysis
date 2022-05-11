# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:23:46 2022

@author: scott

GMAT Scenario Module
--------------------

Contains code to interact with GMAT.
Edit template scenario scripts, run the scripts, and read in results.

"""

from astropy.time import Time
import spiceypy as spice

# Module imports
from utils import get_data_home

# Load GMAT
from load_gmat import *

#%% GMAT Scenario

def configure_run_GMAT(sat_dict, gs_dict, duration=10., timestep=60., out_dir=None):
    '''
    Configure and run a GMAT script to compute access between a satellite and 
    groundstation.
    
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

#%% Retreive GMAT Results

def load_access_results(DATA_DIR=None):
    '''
    Load access results from GMAT simulation. Returns a dataframe containing
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

def load_sat_eclipse_results(DATA_DIR=None):
    '''
    Load eclipse results from GMAT simulation. Returns a dataframe containing
    the Start Time, Stop Time, and Duration (s) of each eclipse period.

    Parameters
    ----------
    DATA_DIR : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe containing eclipse periods.

    '''
    
    
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

def load_ephem_report_results(DATA_DIR=None):
    '''
    Depreciated. Use Ephem.get_ephem_TOPO()
    
    Load ephemeris report results from GMAT simulation. Returns a dataframe 
    containing the position vectors of the satellite and ground station at
    timesteps generated by the propagator.
    
    Note: Problems with GMAT computing Sun position. Read from SPK file instead.

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
    df = pd.read_csv(str(DATA_DIR/'ObservationReportFile1.txt'), 
                      sep=r"\s\s+", engine='python')
    
    # Create a column with datetime objects
    df.insert(0, 'UTCG', pd.to_datetime(df['Sat.UTCGregorian']))
    
    # Add ephemeris time
    spice.furnsh( str(get_data_home()/'kernels'/'naif0012.tls') ) # Leap second kernel
    t = df['Sat.UTCGregorian'].astype(str).to_list() # Epochs in UTCGregorian
    et = spice.str2et(t) # Epochs in ET
    df.insert(0, 'ET', et)
    
    
    # Computations
    
    # Add Sun position
    # GMAT does not calculate sun position correctly. Generate from SPICE.
    # Load kernels
    kernel_dir = get_data_home()  / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Leap second kernel
    
    # Apparent position of Earth relative to the Moon's center in the IAU_MOON frame.
    targ = 'Sun'  # Target body
    ref =  'J2000'     # Reference frame 'J2000' or 'iau_earth'
    abcorr = 'lt+s'   # Aberration correction flag.
    obs = 'Earth'     # Observing body name
    # List of reference frames: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html
    
    # Earth Inertial Frame (J2000) 
    [sv_j2000, ltime] = spice.spkpos(targ, et, 'J2000', abcorr, obs) # Inertial state
    sunpos_j2000 = np.array(sv_j2000) # Position vector as numpy
    # Earth Body-fixed Frame (iau_earth)
    [sv_iau_earth, ltime] = spice.spkpos(targ, et, 'iau_earth', abcorr, obs) # Inertial state
    sunpos_iau_earth = np.array(sv_iau_earth) # Position vector as numpy
    
    # Add to dataframe
    df[['Sun.J2000.X','Sun.J2000.Y','Sun.J2000.Z']] = sunpos_j2000 
    
    
    
    # Compute Az/El angles relative to GS Topo frame
    # Elevation
    # 'Sat.TopoGS1.DEC' is the declination or elevation of the spacecraft above
    # the local horizon
    # Confimed computation below
    df['SatEl'] = np.rad2deg(np.arctan2(df['Sat.TopoGS1.Z'],
                        np.sqrt(df['Sat.TopoGS1.X']**2 + df['Sat.TopoGS1.Y']**2))) # Elevation (from xy plane)
    # Confirmed == Sat.TopoGS1.DEC
    # >> df[['Sat.TopoGS1.DEC','SatEl']]
    
    
    # Azimuth
    # 'Sat.TopoGS1.RA' measures the right ascension of the satellite in local SEZ
    # coordinates, which gives the angle from south direction (x axis) measured
    # anti-clockwise.
    theta = np.rad2deg(np.arctan2(df['Sat.TopoGS1.Y'],df['Sat.TopoGS1.X'])) # From S measured anti-clockwise
    # Confirmed theta == Sat.TopoGS1.RA
    # >> df[['Sat.TopoGS1.RA','SatAz']]
    
    # Compute the Azimuth in an Alt/Az system (measured clock-wise from North)
    df['SatAz'] = 180 - theta
    
    
    # TODO: Compute proper motion in az/el
    
    
    # # Load access data
    # dfa = load_access_results(DATA_DIR=DATA_DIR)
    # Add access period label to each timestep
    
    
    return df


#%% Scenario Coverage times

def get_GMAT_coverage(DATA_DIR=None):
    '''
    Get the date coverage of the GMAT scenario. Return a dictionary containing
    the start/stop dates in both ISO dates and ephemeris time (ET).

    '''
    
    # Set default data dir
    if DATA_DIR is None:
        DATA_DIR = get_data_home()/'GMATscripts'/'Access'
    else:
        if type(DATA_DIR)==str:
            DATA_DIR = Path(DATA_DIR)
    
    # Load the script of the configured mission
    gmat.LoadScript(str(DATA_DIR/"configured_mission.script"))
    
    # Load LSK for time conversions
    spice.furnsh( str(get_data_home()/'kernels'/'naif0012.tls') ) # Leap second kernel
    
    # Get GMAT objects
    sat = gmat.GetObject("Sat") # Satellite
    var_prop_time_days = gmat.GetObject("prop_time_days") # Variable Propagation time (days) 
    
    # Extract start time from satellite
    start_date = sat.GetField("Epoch")
    # Convert to astropy.Time object
    start_et = spice.str2et(start_date) # Epochs in ET
    
    # Extract propagation time (days)
    duration = float(var_prop_time_days.GetField('Value'))
    # Compute end date (1 day = 86400 s)
    stop_et = start_et + duration*86400.
    
    # Convert start
    t1 = Time(spice.et2datetime(start_et), scale='utc')
    start_date = t1.iso # ISO
    start_jd = t1.jd    # Julian date
    # Stop times
    t2 = Time(spice.et2datetime(stop_et), scale='utc')
    stop_date = t2.iso
    stop_jd = t2.jd
    
    # Format dictionary of results
    cov = {'start_date':start_date, 'start_et':start_et, 'start_jd':start_jd,
           'stop_date':stop_date, 'stop_et':stop_et, 'stop_jd':stop_jd}
    
    return cov

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