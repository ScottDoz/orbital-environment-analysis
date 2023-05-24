# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:51:33 2022

@author: scott

Events Module
-------------

Compute times of events such as access and eclipses.

Note: use fo the cspice command line tools requires downloading and installing
the cspice library from https://naif.jpl.nasa.gov/naif/utilities.html
Individual exe files can be downloaded from 
https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html

"""

import numpy as np
import pandas as pd
import spiceypy as spice

from utils import get_data_home
from Ephem import get_ephem_TOPO

from scipy.signal import chirp, find_peaks, peak_widths, welch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import pdb
import time


#%% Lighting Conditions
# Use https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfoclt_c.html
# To find eclipse times

def find_sat_lighting(start_et,stop_et):
    '''
    Find lighting conditions of the satellite. Time intervals when satellite
    is in sunlight, or in eclipse (when the Sun is occulted by the Earth as 
    seen from the Satellite).
    
    Utilizes SPICE gfoclt_c - Geometric occultation finder.
    "Determine time intervals when an observer sees one target occulted by, or 
    in transit across, another.The surfaces of the target bodies may be 
    represented by triaxial ellipsoids or by topographic data provided by DSK files."
    
    Occultation geometry:
    Find ocultations of the Sun by Earth as seen from the Satellite
    Target: Sun
    Observer: Satellite
    Occulting body: Earth
    
    This workflow is also used internally within GMATs EclipseLocator.
    
    See example 3 of gfoclt_c documentation
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfoclt_c.html
    
    Parameters
    ----------
    start_et, stop_et : float
        Start and stop times of the window of interest (Ephemeris Time).

    Returns
    -------
    light, partial, dark : SpiceCell
        Time intervals for light, partial, and dark lighting conditions.

    '''
    
    # Kernel file directory
    kernel_dir = get_data_home()  / 'Kernels'
    # sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    sat_ephem_file = str(get_data_home()/'Kernels'/'sat.bsp')
    
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    ids = spice.spkobj(sat_ephem_file) # SpiceCell object
    numobj = len(ids)
    sat_NAIF = [ids[i] for i in range(numobj) ] # -10002001 NAIF of the satellite
    
    # # Get the coverage of the spk file
    # # Coverage time is in et
    # cov = spice.spkcov(sat_ephem_file,ids[0]) # SpiceCell object
    
    # Create time window of interest
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    MAXWIN = 2000 # Maximum number of intervals
    cnfine = spice.cell_double(MAXWIN) # Initialize window of interest
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
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
    
    # Find occulations
    
    # Full occulation (dark or umbra)
    dark = spice.cell_double(2*MAXWIN) # Initialize result
    dark = spice.gfoclt ( "FULL",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, dark          )
    
    # Annular occulation
    annular = spice.cell_double(2*MAXWIN) # Initialize result
    annular = spice.gfoclt ( "ANNULAR",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, annular          )
    
    # Partial occulation (penumbra) 
    partial = spice.cell_double(2*MAXWIN) # Initialize result
    partial = spice.gfoclt ( "PARTIAL",
                          front,   fshape,  fframe,
                          back,    bshape,  bframe,
                          abcorr,  obsrvr,  step,
                          cnfine, partial          )
    
    # Join Annular and Partial occultations
    partial = spice.wnunid(partial, annular)
    
    # # Any occulation
    # anyocc = spice.cell_double(2*MAXWIN) # Initialize result
    # dark = spice.gfoclt ( "FULL",
    #                       front,   fshape,  fframe,
    #                       back,    bshape,  bframe,
    #                       abcorr,  obsrvr,  step,
    #                       cnfine, anyocc          )
    # # Note: Results for this do not match the union of the other types.
    # #       Use spice windows operations instead
    
    
    
    # Find sunlight times
    # Complement of time window when satellite is not in full or partial eclipse.
    
    # Join full and partial results to find times when not in full sunlight
    anyocc = spice.wnunid(dark, partial)
    
    # Take complement to find times when in full sunlight
    light = spice.wncomd(start_et,stop_et,anyocc)
    
    # TODO: Find occultations by the moon
    
    return light, partial, dark

def find_station_lighting(start_et,stop_et,station='DSS-43',method='ref_el', ref_el = -6.):
    '''
    Find time intervals when a ground station is in sunlight and darkness.
    Darkness is defined here using nautical twilight, when the local sun 
    elevation angle is below -6 deg.
    
    Utilizes SPICE gfposc_c - Geometry finder using observer-target position vectors.
    "Determine time intervals for which a coordinate of an observer-target 
    position vector satisfies a numerical constraint."
    
    This workflow is also used internally within GMATs EclipseLocator.
    
    See e.g. 5 from fgposc_c documentation.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfposc_c.html
    

    Parameters
    ----------
    start_et, stop_et : float
        Start and stop times of the window of interest (Ephemeris Time).
    station : str, optional
        Name of the ground station. The default is 'DSS-43'.
    method : str
        Method for comulting the lighting conditions.
    ref_el : float, optional
        Reference elevation of sun to distinguish between light and dark.

    Returns
    -------
    light, dark : SpiceCell
        Time intervals for light and dark lighting conditions of the station.

    '''
    
    # Find time intervals when the station is dark 
    # Nautical twilight: sun elevation < -6 deg and > -18 deg
    # Two ways to do this with SPICE

    
    # Method 1: gfposc_c
    # gfposc_c: Geometry finder using observer-target position vectors.
    
    # Alternative method
    # gfilum_c: Geometry finder using ilumination angles.
    # E.g. when solar incidence is below certain value
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gfilum_c.html
    
    
    
    # Kernel file directory
    kernel_dir = get_data_home()  / 'Kernels'
    station_ephem_file = str(kernel_dir/'earthstns_itrf93_201023.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    # DSN Stations
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides
    # SSR Stations
    spice.furnsh( str(kernel_dir/'SSR_stations.bsp') ) # SSR station Ephemerides
    spice.furnsh( str(kernel_dir/'SSR_stations.tf') )      # SSR topocentric frame text kernel
    # SSRD Stations
    spice.furnsh( str(kernel_dir/'SSRD_stations.bsp') ) # SSRD station Ephemerides
    spice.furnsh( str(kernel_dir/'SSRD_stations.tf') )  # SSRD topocentric frame text kernel
    
    
    # Get details of station kernel file
    ids = spice.spkobj(str(kernel_dir/'earthstns_itrf93_201023.bsp'))
    numobj = len(ids)
    gs_NAIF = [ids[i] for i in range(numobj) ]
    
    # Load the coverage window of station
    kernel_dir = get_data_home()  / 'Kernels'
    # ids = spice.spkobj(sat_ephem_file)
    cov = spice.spkcov(station_ephem_file,gs_NAIF[0]) # SpiceCell object
    
    # Create time window of interest
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    MAXWIN = 2000 # Maximum number of intervals
    cnfine = spice.cell_double(2) # Initialize window of interest
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
    if method == 'ref_el':
        # Find dark times based on reference elevaition
    
        # Settings for geometry search  
        # Find when the solar elevation is < -6 deg
        targ = "SUN" # Target
        frame  = station+"_TOPO" # Reference frame
        # abcorr = "NONE" # Aberration correction flag
        abcorr = "lt"
        obsrvr = station # Observer
        crdsys = "LATITUDINAL" # Coordinate system
        coord  = "LATITUDE" # Coordinate of interest
        refval = ref_el*spice.rpd() # Reference value
        relate = "<"             # Relational operator 
        adjust = 0. # Adjustment value for absolute extrema searches
        step = (1./24.)*spice.spd() # Step size (1 hrs)
        
        # Call the function to find eclipse times (station dark)
        dark = spice.cell_double(2*MAXWIN) # Initialize result
        dark = spice.gfposc(targ,frame,abcorr,obsrvr,crdsys,coord,relate,
                         refval,adjust,step,MAXWIN,cnfine,dark)
        
        # Find lit times
        # This is the complement of the dark time intervals, constrained by the 
        # original window. Use the SPICE wncomd_c function.
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/wncomd_c.html
        light = spice.wncomd(start_et,stop_et,dark)
        
    
    elif method == 'full eclipse':
        # Find dark times based on occultation/eclipse of Sun by Earth.
        # Dark = full eclipse
        
        # Find ocultations of the Sun by Earth as seen from the Groundstation
        # Target: Sun
        # Observer: GS
        # Occulting body: Earth
        
        # Occultation geometry search settings
        occtyp = "FULL"  # Type of occultation (Full,Annular,Partial,Any)
        front = "EARTH" # Name of occulting body
        fshape = "ELLIPSOID" # Type of shape model for front body (POINT, ELLIPSOID, DSK/UNPRIORITIZED)
        fframe = "ITRF93" #"IAU_EARTH" # # Body-fixed frame of front body
        back =  "SUN" # Name of occulted body
        bshape = "ELLIPSOID" # Type of shape model for back body
        bframe = "IAU_SUN" # Body-fixed frame of back body (empty)
        # abcorr = "NONE" # Aberration correction flag
        abcorr = "lt"
        obsrvr = station  # Observer
        step = 10. # Step size (s)
        
        # Find occulations
        
        # Full occulation (dark or umbra)
        dark = spice.cell_double(2*MAXWIN) # Initialize result
        dark = spice.gfoclt ( occtyp,
                              front,   fshape,  fframe,
                              back,    bshape,  bframe,
                              abcorr,  obsrvr,  step,
                              cnfine, dark          )
        
        # Find lit times
        # This is the complement of the dark time intervals, constrained by the 
        # original window. Use the SPICE wncomd_c function.
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/wncomd_c.html
        light = spice.wncomd(start_et,stop_et,dark)

    elif method == 'eclipse':
        # Find dark times based on occultation/eclipse of Sun by Earth.
        # Dark = full or partial eclipse
        
        # Find ocultations of the Sun by Earth as seen from the Groundstation
        # Target: Sun
        # Observer: GS
        # Occulting body: Earth
        
        # Occultation geometry search settings
        occtyp = "ANY"  # Type of occultation (Full,Annular,Partial,Any)
        front = "EARTH" # Name of occulting body
        fshape = "ELLIPSOID" # Type of shape model for front body (POINT, ELLIPSOID, DSK/UNPRIORITIZED)
        fframe = "ITRF93" #"IAU_EARTH" # # Body-fixed frame of front body
        back =  "SUN" # Name of occulted body
        bshape = "ELLIPSOID" # Type of shape model for back body
        bframe = "IAU_SUN" # Body-fixed frame of back body (empty)
        abcorr = "NONE" # Aberration correction flag
        # abcorr = "lt"
        obsrvr = station  # Observer
        step = 5. # Step size (s)
        
        # Find occulations
        
        # Full occulation (dark or umbra)
        full = spice.cell_double(2*MAXWIN) # Initialize result
        full = spice.gfoclt ( occtyp,
                              front,   fshape,  fframe,
                              back,    bshape,  bframe,
                              abcorr,  obsrvr,  step,
                              cnfine, full          )
        
        # Full occulation (dark or umbra)
        partial = spice.cell_double(2*MAXWIN) # Initialize result
        partial = spice.gfoclt ( "PARTIAL",
                              front,   fshape,  fframe,
                              back,    bshape,  bframe,
                              abcorr,  obsrvr,  step,
                              cnfine, partial          )
        
        
        # Full occulation (dark or umbra)
        dark = spice.cell_double(2*MAXWIN) # Initialize result
        dark = spice.gfoclt ( occtyp,
                              front,   fshape,  fframe,
                              back,    bshape,  bframe,
                              abcorr,  obsrvr,  step,
                              cnfine, dark          )
        
        
    
        # Find lit times
        # This is the complement of the dark time intervals, constrained by the 
        # original window. Use the SPICE wncomd_c function.
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/wncomd_c.html
        light = spice.wncomd(start_et,stop_et,full) # Complement of full eclipse times
        light = spice.wndifd(light,partial) # Remove partial eclipse times
    
    
    # TODO: Bin data based on Sun.El ranges
    # https://towardsdatascience.com/how-i-customarily-bin-data-with-pandas-9303c9e4d946
    
    
    return light, dark


#%% Access

def find_access(start_et,stop_et,station='DSS-43',min_el=0.):
    '''
    Find time intervals when a ground station has line-of-sight access to a
    satellite - when the satellite is above a minimum elevation angle in the
    local topocentric frame.
    
    Utilizes SPICE gfposc_c - Geometry finder using observer-target position vectors.
    "Determine time intervals for which a coordinate of an observer-target 
    position vector satisfies a numerical constraint."

    This workflow is also used internally within GMATs ContactLocator.

    Parameters
    ----------
    start_et : float
        Start time (Ephemeris time).
    stop_et : float
        Stop time (Ephemeris time).
    station : str, optional
        DESCRIPTION. The default is 'DSS-43'.
    min_el : Float, optional
        Minimum elevation (deg) to use for geometry search. The default is 0.

    Returns
    -------
    access : TYPE
        DESCRIPTION.

    '''
    
    # From documentation, GMAT uses gfposc to perform line-of-sight search above
    # a minimum elevation angle.
    
    # Kernel file directory
    t_start = time.time()
    kernel_dir = get_data_home()  / 'Kernels'
    station_ephem_file = str(kernel_dir/'earthstns_itrf93_201023.bsp')
    # sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    sat_ephem_file = str(get_data_home()/'Kernels'/'sat.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    # DSN Stations
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    # SSR Stations
    spice.furnsh( str(kernel_dir/'SSR_stations.bsp') ) # SSR station Ephemerides
    spice.furnsh( str(kernel_dir/'SSR_stations.tf') )      # SSR topocentric frame text kernel
    # SSRD Stations
    spice.furnsh( str(kernel_dir/'SSRD_stations.bsp') ) # SSRD station Ephemerides
    spice.furnsh( str(kernel_dir/'SSRD_stations.tf') )  # SSRD topocentric frame text kernel


    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    ids = spice.spkobj(sat_ephem_file) # SpiceCell object
    numobj = len(ids)
    sat_NAIF = [ids[i] for i in range(numobj) ] # -10002001 NAIF of the satellite
    print('Kernel loading time: {} s'.format(time.time() - t_start))
    
    # Load the coverage window of station
    t_start = time.time()
    kernel_dir = get_data_home()  / 'Kernels'
    # ids = spice.spkobj(sat_ephem_file)
    cov = spice.spkcov(sat_ephem_file,sat_NAIF[0]) # SpiceCell object
    
    # # Create time window of interest
    # # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    # MAXWIN = 5000 # Maximum number of intervals
    # cnfine = spice.cell_double(2) # Initialize window of interest
    # spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    # print('Coverage window creation time: {} s'.format(time.time() - t_start))
    
    
    # Pre-filter times --------------------------------------------------------
    # Use course ephemeris data to get rough estimates of time intervals for each
    # access time. Use these as initial intervals to constrain gfposc geometry
    # search - rather than running search over entire scenario time.
    # This should significantly speed up computation time.
    

    # Get satellite ephemeris    
    t_start = time.time() # Start time
    dt = 60 # Time step (s)
    et = np.arange(start_et,stop_et,60.) # List of epochs
    # Satellite ephemeris (targ, et, ref, abcorr, obs)
    [satv, ltime] = spice.spkpos( str(sat_NAIF[0]), et, station+'_TOPO', 'lt+s', station)
    # Convert to lat/long coords
    rlonlat_tups = [ spice.reclat( satv[i,:] ) for i in range(len(satv))] # List of (r,lon,lat) tupples
    r,lon,lat = np.array(list(zip(*rlonlat_tups)))
    el = lat
    print('Ephem load time: {} s'.format(time.time() - t_start))
    
    # Find peaks in elevation
    # Filter peaks above 0 deg
    # For each, find time
    # See: http://constans.pbworks.com/w/file/fetch/120908295/Simple_Algorithms_for_Peak_Detection_in_Time-Serie.pdf
    
    # Find peaks and zero crossings
    y = el
    
    # Find indices of peaks
    peaks, _ = find_peaks(el) # Find peaks in signal
    peaks_pos = peaks[el[peaks] > 0] # Peaks that are above 0 deg elevation
    peaks_neg = peaks[el[peaks] <= 0] # Peaks that are below 0 deg elevation
    t_peaks = et[peaks_pos] # Times of peaks
    
    # Compute widths of each peak at the x axis
    # Returns widths, with heights, interpolated positions of left and right 
    # widths, h_eval, left_ips, right_ips = peak_widths(y, peaks, rel_height=np.finfo(float).eps) # Widths of peaks at x-crossing
    
    # Find zero crossings (el==0)
    # Distinguish between upwards crossing (from -ve to +ve)
    # and downwards crossings (from +ve to -ve)
    crossings_up =   np.where((y[1:] >  0) * (y[:-1] <= 0))[0]  # Points where move from -ve to +ve
    crossings_down = np.where((y[1:] <= 0) * (y[:-1] >  0))[0] # Points where move from +ve to -ve
    crossings_down += 1 # Move to the next timestep
    t_cross_up = et[crossings_up] # Times of upward crossings
    t_cross_down = et[crossings_down] # Times of downward crossing
    
    
    
    # Find closest crossing either side of peaks
    # For each positive peak, find the nearest zero crossings either side.
    # Use np.searchsorted to find indices of closest zero crossings to the left and right
    # Closest updards crossing (left end of interval)
    ind_left = np.searchsorted(t_cross_up, t_peaks) # Index points should be inserted into crossssing_up
    ind_left -= 1 # Get the index of the point to the left of these
    t_left = t_cross_up[ind_left] # Times of these crossings (left end of each bracket)
    # Closest downward crossing (right end of interval)
    ind_right = np.searchsorted(t_cross_down, t_peaks) # Index points should be inserted into crossing_down
    t_right = t_cross_down[ind_right] # Times of these crossings (right end of each bracket)
    # Replace epochs where ind < 0 with stat epoch
    t_left[ind_left<0] = start_et
    # TODO: Check for right
    if max(ind_right)>len(t_cross_down):
        # Right end point beyond end of timeframe
        pdb.set_trace()
    
    
    # Check intervals
    if min(t_right > t_left) == False:
        print('Warning: Invervals are not ascending.')
        pdb.set_trace()
    
    # We now have a set of peaks in elevation at epochs t_peaks.
    # For each peak, we have the epochs of the zero crossings either size (t_left, t_right)
    # These form a set of intervals that should contain each access period
    # During each interval (t_left, t_right), the elevation will range from -ve
    # to a +ve maximum elevation, back to -ve.
    # We can use these intervals as inital guesses for the access intervals.
    
    # Create interval cnfine from these. 
    # Create time window of interest
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    MAXWIN = 5000 # Maximum number of intervals
    cnfine = spice.cell_double(2*len(t_peaks)) # Initialize window of interest
    [spice.wninsd(t_left[i], t_right[i], cnfine ) for i in range(len(t_peaks))] # Insert time intervals in window
    print('Coverage window creation time: {} s'.format(time.time() - t_start))

    # FIXME: Plot to confirm intervals are correct    
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(et,np.rad2deg(el),'-b') # Timeseries
    # ax.plot([et[0],et[-1]],[0,0],'-k') # el=0 line
    # ax.plot(et[peaks],np.rad2deg(y[peaks]),'or') # Peaks
    # ax.plot(et[crossings_up],np.rad2deg(y[crossings_up]),'ob')   # Up crossings
    # ax.plot(et[crossings_down],np.rad2deg(y[crossings_down]),'og') # Down crossings
    
    # for i in range(2):
    #     ax.plot([et[left_ips.astype(int)[i]], et[right_ips.astype(int)[i]] ],[0,0],'-r' )
    # # plt.hlines(h_eval, et[left_ips.astype(int)], et[right_ips.astype(int)], color="r",linewidth=2)
    # fig.show()
    
    # -------------------------------------------------------------------------
    
    # Settings for geometry search  
    # Find when the satellite elevation is > 0 deg
    targ = str(sat_NAIF[0]) # Target
    frame  = station+"_TOPO" # Reference frame
    # abcorr = "NONE" # Aberration correction flag
    abcorr = "lt+s" # Aberration correction flag
    obsrvr = station # Observer
    crdsys = "LATITUDINAL" # Coordinate system
    coord  = "LATITUDE" # Coordinate of interest
    refval = min_el*spice.rpd() # Reference value
    relate = ">"            # Relational operator 
    adjust = 0. # Adjustment value for absolute extrema searches
    step = 5. #10. # Step size (10 s)
    
    # Call the function to find full, anular and partial
    try:
        t_start = time.time()
        access = spice.cell_double(2*MAXWIN) # Initialize result
        access = spice.gfposc(targ,frame,abcorr,obsrvr,crdsys,coord,relate,
                         refval,adjust,step,MAXWIN,cnfine,access)
        print('gfposc time: {} s'.format(time.time() - t_start))
    except:
        pdb.set_trace()
    
    return access

def constrain_access_by_lighting(access,gsdark,satlight):
    '''
    Constrain the line-of-sight access intervals by the lighting conditions of
    the satellite and groundstation. Use SPICE window logical set functions to
    find intervals when there is access, the station is in darkness, and the 
    satellite is in sunlight.
    
    Visaccess = access ∩ (gsdark ∩ satlight)
    
    See pg 10 of SPICE Tutorial on Window Operations
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/29_geometry_finder.pdf
    

    Parameters
    ----------
    access : SpiceCell
        Time intervals for line-of-sight access from station to satellite.
    gsdark : TYPE
        Time intervals for darkness of the station.
    satlight : TYPE
        Time intervals for sunlight of the satellite.

    Returns
    -------
    visaccess : TYPE
        Constrained time intervals for visible access from station to satellite.

    '''
    
    # # Use spice window functions to compute the set differences
    # # visaccess = access - gslight -satdark
    # A1 = spice.wnintd(gsdark,satlight) # GS dark and sat light
    # visaccess = spice.wnintd(access,A1) # GS dark and sat light and LOS
    
    # Incremental test
    # LOS and sat light
    visaccess = spice.wnintd(access,satlight)  # LOS and Sat light
    visaccess = spice.wnintd(visaccess,gsdark) # LOS and Sat light and GS dark 
    
    
    return visaccess

def old_constrain_access_by_lighting(access,gslight,satdark):
    '''
    Constrain the line-of-sight access intervals by the lighting conditions of
    the satellite and groundstation. Use SPICE window logical set functions to
    remove intervals when the station is in sunlight and when the satellite is
    in darkness.
    
    Visaccess = access - gslight - satdark

    Parameters
    ----------
    access : SpiceCell
        Time intervals for line-of-sight access from station to satellite.
    gslight : TYPE
        Time intervals for sunlight of the station.
    satdark : TYPE
        Time intervals for darkness of the satellite.

    Returns
    -------
    visaccess : TYPE
        Constrained time intervals for visible access from station to satellite.

    '''
    
    # Use spice window functions to compute the set differences
    # visaccess = access - gslight -satdark
    visaccess = spice.wndifd(access,gslight) # Subtract station daylight
    visaccess = spice.wndifd(visaccess,satdark) # Subtract sat darkness
    
    return visaccess


#%% Utility functions

def window_to_dataframe(cnfine,timefmt='ET',method='loop'):
    '''
    Convert a SpiceCell window containing time intervals to a dataframe.

    Parameters
    ----------
    cnfine : SpiceCell
        Window containing time intervals.
    timefmt : str, optional
        Flag specifying the time format of the output times. The default is 'ET'.
    method : TYPE, optional
        DESCRIPTION. The default is 'loop'.

    Returns
    -------
    df : Pandas Dataframe
        Result dataframe.

    '''
    
    # Get number of intervals in window
    count = spice.wncard( cnfine ) # Count of intervals
    # print('{} intervals'.format(count))
    
    # Extract start and end times of each interval (2 methods)
    if method.lower() == 'loop':
        # Loop through each interval
        # 1.04 ms ± 16.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        start_et = np.zeros(count) # Initialize start times
        stop_et = np.zeros(count) # Initialize stop times
        for i in range(count):
            t0,t1 = spice.wnfetd (cnfine, i)
            start_et[i] = t0
            stop_et[i] = t1
    elif method.lower() == 'list comp':
        # List comprehension method (slightly faster)
        start_et,stop_et = zip(*[spice.wnfetd (cnfine, i) for i in range(count)])
        start_et = np.array(start_et)
        stop_et = np.array(stop_et)
    
    # Time results on window with 298 intervals
    # Loop:    1.04 ms ± 16.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    # Listcomp: 959 µs ± 13.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    # Around same time. Loop is faster for fewer intervals.
    
    # Construct empty dataframe
    cols = ['Start','Stop','Duration']
    df = pd.DataFrame(columns=cols)
    
    # Add Duration
    df['Duration'] = stop_et - start_et # Duration in seconds
    
    # Add start and stop times
    if timefmt.lower() == 'et':
        # Output times in Ephemeris Time
        df['Start'] = start_et
        df['Stop'] = stop_et
    elif timefmt.lower() in ['dt','datetime']:
        # Output times in datetime64 objects
        start_dt = spice.et2datetime(start_et)
        stop_dt = spice.et2datetime(stop_et)
        df['Start'] = start_dt
        df['Stop'] = stop_dt
    
    return df


# Time windows
# Methods to create and use time windows

def create_timewindow(start_et, stop_et, NINT):
    
    # N = max number of intervals in window
    
    
    # Time windows work with spice cells
    # See: https://pirlwww.lpl.arizona.edu/resources/guide/software/SPICE/windows.html
    
    # Create a double cell of size 2 to use as a window
    # Must be even number to use this cell as a window
    # N intervals requires 2*N cell values
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    cnfine = spice.cell_double(2*NINT) # Initialize window of interest
    
    # Insert a time interval from start and stop times
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
    
    # # Alternative method
    # from spiceypy.utils import support_types as stypes
    # cnfine = stypes.SPICEDOUBLE_CELL(2)
    # # cnfine =  spice.wnvald( 2, 2, cnfine )
    
    # Validate
    # spice.wnvald(8,2,WIN)

    return cnfine



#%% Notes on angles and reference frames

# Elevation
# Angle between position vector and x-y plane
# GMAT gives values in DEC (declination) e.g. Sat.TopoGS1.DEC

# Azimuth 
# Angle from +X axis rotated clockwise about Z axis (right-hand rule around -Z axis)
# GMAT gives the RA 

# 'Sat.TopoGS1.RA' measures the right ascension of the satellite in local SEZ
# coordinates, which gives the angle from south direction (x axis) measured
# anti-clockwise.


# Additional software tools that work with SPICE
# https://naif.jpl.nasa.gov/naif/SPICE_aware_Tools_List.pdf


