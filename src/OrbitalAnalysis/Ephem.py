# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:19:26 2022

@author: scott

Ephemeris module
---------------

Contains functions for extracting ephemerides of the satellite, sun, moon, and
groundstations, and computing lighting and access conditions.


Note: use fo the cspice command line tools requires downloading and installing
the cspice library from https://naif.jpl.nasa.gov/naif/utilities.html
Individual exe files can be downloaded from 
https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html

"""

import os
import requests
import numpy as np
import pandas as pd
from astropy.time import Time as Time
import spiceypy as spice

import time
import timeit

from utils import get_data_home

import pdb


#%% SPICE Data Access
# This class is copied from sr_tools package

class SPICEKernels:
    '''
    Dataset class to download SPK (Spacecraft Planetery Kernel) files from
    NAIF and the Horizons Asteroid & Comet SPK File Generation Request website
    https://ssd.jpl.nasa.gov/x/spk.html

    SPK files contain ephemerides of planetary objects.
    Data can be read useing the Spiceypy package.
    '''
    
    @classmethod
    def _check_data_directory(self):
        """
        Check the data directory DATA_DIR/Kernels exists. If not, create it.
        """
        
        # Set directory to save into
        DATA_DIR = get_data_home() # Data home directory
        _dir = DATA_DIR  / 'Kernels'
        
        # Check if directory exists and create
        if not os.path.exists(str(_dir)):
            os.makedirs(str(_dir))
        
        return

    # Helper function
    @classmethod
    def _getFilename_fromCd(self,cd):
        """
        Get filename from content-disposition
        Source: https://www.tutorialspoint.com/downloading-files-from-web-using-python
        """
        if not cd:
            return None

        fname = re.findall('filename=(.+)', cd)
        if len(fname) == 0:
            return None

        return fname[0]

    @classmethod
    def _get_spkid(self,query):
        '''
        Find the SPKID and ephemeris filename from an asteroid's PDES.

        Parameters
        ----------
        query : str
            A designation of number to query.

        Returns
        -------
        spkid : int
            The SPKID of the object.

        '''

        # Construct the url for the cgi request.
        url = "https://ssd.jpl.nasa.gov/x/smb_spk.cgi?OBJECT={}&OPT=Look+Up".format(str(query))

        # # Read query
        # from bs4 import BeautifulSoup
        # soup = BeautifulSoup(html_doc, 'html.parser')

        # Read query
        r = requests.get(url) # Read response
        # Extract each row of the response
        text = r.text.split('\n')
        # Extract text line containing 'Primary SPKID'
        spktxt = [s for s in text if 'Primary SPKID' in s][0]
        # Split spkid
        spkid = int(spktxt.split('=')[1].strip())

        return spkid

    @classmethod
    def download_spk_file(self,query):
        '''
        Download an asteroid SPK file using the website cgi form.

        '''

        # Check data directory
        self._check_data_directory()

        #TODO: In future could use command line tools to perform this query
        # ftp://ssd.jpl.nasa.gov/pub/ssd/SCRIPTS/smb_spk

        # Set directory to save into
        DATA_DIR = get_data_home() # Data home directory
        _dir = DATA_DIR  / 'Kernels'

        # Insert webform data into url request.
        # See: https://github.com/skyfielders/python-skyfield/issues/196
        url = "https://ssd.jpl.nasa.gov/x/smb_spk.cgi?OBJECT={}&OPT=Make+SPK&OPTION=Make+SPK&START=1000-JAN-01&STOP=2101-JAN-01&EMAIL=foo@bar.com&TYPE=-B".format(str(query))

        

        # Get the request
        print('Downloading SPK file.')
        r = requests.get(url, allow_redirects=True)

        # Get the filename from request
        # filename = self._getFilename_fromCd(r.headers.get('content-disposition'))

        # Get the spkid of the object to save
        spkid = self._get_spkid(query)
        filename = str(spkid) + '.bsp'

        # Write the results to file
        fullfilename = _dir / filename
        open(fullfilename, 'wb').write(r.content)
        print('File {} saved'.format(str(fullfilename)))

        return

    @classmethod
    def download_planet_spk(self):
        '''
        Download planetary SPK files from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

        '''
        
        # Check data directory
        self._check_data_directory()

        # DE440
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"

        # Get the request
        print('Downloading SPK file.')
        r = requests.get(url, allow_redirects=True)

        # Get the filename
        filename = url.split('/')[-1]
        # filename = self._getFilename_fromCd(r.headers.get('content-disposition'))

        # Set directory to save into
        DATA_DIR = get_data_home() # Data home directory
        _dir = DATA_DIR  / 'Kernels'

        # Write the results to file
        fullfilename = _dir / filename

        open(fullfilename, 'wb').write(r.content)
        print('File {} saved'.format(str(fullfilename)))

        return

    @classmethod
    def download_lsk(self):
        '''
        Download Leap Second Kernel files from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/

        '''
        
        # Check data directory
        self._check_data_directory()
        
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"

        # Get the request
        print('Downloading LSK file.')
        r = requests.get(url, allow_redirects=True)

        # Get the filename
        filename = url.split('/')[-1]
        # filename = self._getFilename_fromCd(r.headers.get('content-disposition'))

        # Set directory to save into
        DATA_DIR = get_data_home() # Data home directory
        _dir = DATA_DIR  / 'Kernels'

        # Write the results to file
        fullfilename = _dir / filename

        open(fullfilename, 'wb').write(r.content)
        print('File {} saved'.format(str(fullfilename)))

        return
    
    @classmethod
    def download_pck(self):
        '''
        Download Planetary Constants Kernel files from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

        '''
        
        # Check data directory
        self._check_data_directory()
        
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"

        # Get the request
        print('Downloading PCK file.')
        r = requests.get(url, allow_redirects=True)

        # Get the filename
        filename = url.split('/')[-1]
        # filename = self._getFilename_fromCd(r.headers.get('content-disposition'))

        # Set directory to save into
        DATA_DIR = get_data_home() # Data home directory
        _dir = DATA_DIR  / 'Kernels'

        # Write the results to file
        fullfilename = _dir / filename

        open(fullfilename, 'wb').write(r.content)
        print('File {} saved'.format(str(fullfilename)))
        
        return

#%% Ephemerides in Earth-fixed frame

def get_ephem_ITFR(et,groundstations=['DSS-43']):
    
    # Kernel file directory
    kernel_dir = get_data_home()  / 'Kernels'
    
    sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    ids = spice.spkobj(sat_ephem_file) # SpiceCell object
    numobj = len(ids)
    sat_NAIF = [ids[i] for i in range(numobj) ] # -10002001 NAIF of the satellite
    
    # Get the coverage of the spk file
    # Coverage time is in et
    cov = spice.spkcov(sat_ephem_file,ids[0]) # SpiceCell object
    start_et, stop_et = cov
    
    # Check satellite ephem
    if et.min() < start_et:
        raise ValueError("Warning: et outside spk coverage range")
    if et.max() > stop_et:
        raise ValueError("Warning: et outside spk coverage range")
    
    # Convert ET to datetimes
    dt = spice.et2datetime(et) # Datetime
    t = Time(dt, format='datetime', scale='utc') # Astropy Time object
    t_iso = t.iso # Times in iso
    
    
    # Create dataframe for output
    df = pd.DataFrame(columns=['ET','UTCG',
                               'Sat.X','Sat.Y','Sat.Z',
                               'Sun.X','Sun.Y','Sun.Z'])
    # Add epochs
    df.ET = et
    df.UTCG = t_iso
    df.UTCG = pd.to_datetime(df.UTCG) # Convert to Datetime object
    
    # Ephemeris settings
    # ref = 'ITRF93'  # High-precision Earth-fixed reference frame
    # abcorr = 'lt+s' # Aberration correction flag
    # obs = 'earth'   # Observing body name
    
    # Satellite ephemeris (targ, et, ref, abcorr, obs)
    [satv, ltime] = spice.spkpos( str(sat_NAIF[0]), et, 'ITRF93', 'lt+s', 'earth')
    df['Sat.X'] = satv[:,0]
    df['Sat.Y'] = satv[:,1]
    df['Sat.Z'] = satv[:,2]
    # Sun ephemeris  (targ, et, ref, abcorr, obs)
    [sunv, ltime] = spice.spkpos( 'sun', et, 'ITRF93', 'lt+s', 'earth')   
    df['Sun.X'] = sunv[:,0]
    df['Sun.Y'] = sunv[:,1]
    df['Sun.Z'] = sunv[:,2]
    
    # Groundstation ephemerides
    for gs in groundstations:
        # GS ephemeris  (targ, et, ref, abcorr, obs)
        [gsv, ltime] = spice.spkpos( gs, et, 'ITRF93', 'lt+s', 'earth')
        df[gs+'.X'] = gsv[:,0]
        df[gs+'.Y'] = gsv[:,1]
        df[gs+'.Z'] = gsv[:,2]
        
    
    return df


def get_ephem_TOPO(et,groundstations=['DSS-43']):
    '''
    Get the ephemerides of the satellite and sun in topocentric reference frames
    located at a set of groundstations. Returns additional computed properties
    inculding solar phase angle and visual magnitude of the satellite.

    Parameters
    ----------
    et : 1xN numpy array
        List of epochs in Ephemeris Time (ET).
    groundstations : List, optional
        List of groundstations. The default is ['DSS-43'].

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dfs : List
        List of dataframes containing ephemerides of each groundstation.

    '''
    
    # Kernel file directory
    kernel_dir = get_data_home()  / 'Kernels'
    
    # Filenames
    sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    station_ephem_file = str(kernel_dir/'earthstns_itrf93_201023.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    ids = spice.spkobj(sat_ephem_file) # SpiceCell object
    numobj = len(ids)
    sat_NAIF = [ids[i] for i in range(numobj) ] # -10002001 NAIF of the satellite
    
    # Get the coverage of the satellite spk file
    # Coverage time is in et
    cov = spice.spkcov(sat_ephem_file,ids[0]) # SpiceCell object
    start_et, stop_et = cov
    
    # Get details of station kernel file
    ids = spice.spkobj(str(kernel_dir/'earthstns_itrf93_201023.bsp'))
    numobj = len(ids)
    gs_NAIF = [ids[i] for i in range(numobj) ]
    # # Extract comments from the comment area of binary file
    # handle = spice.pcklof (station_ephem_file) # Handle to file
    # BUFFSZ = 100 # Buffer size (lines)
    # LENOUT = 1000
    # comments = spice.dafec ( handle, BUFFSZ, LENOUT)
    # # Alternatively, use the COMMNT command line tool
    
    # Check satellite ephem
    if et.min() < start_et:
        raise ValueError("Warning: et outside spk coverage range")
    if et.max() > stop_et:
        raise ValueError("Warning: et outside spk coverage range")
    
    # Convert ET to datetimes
    dt = spice.et2datetime(et) # Datetime
    t = Time(dt, format='datetime', scale='utc') # Astropy Time object
    t_iso = t.iso # Times in iso
    
    
    # Generate new dataframe for each of the ground stations
    dfs = []
    for gs in groundstations:
    
        # Create dataframe for output
        df = pd.DataFrame(columns=['ET','UTCG',
                                   'Sat.X','Sat.Y','Sat.Z','Sat.Az','Sat.El','Sat.R','Sat.alpha',
                                   'Sun.X','Sun.Y','Sun.Z','Sun.Az','Sun.El','Sun.R',])
        # Add epochs
        df.ET = et
        df.UTCG = t_iso
        df.UTCG = pd.to_datetime(df.UTCG) # Convert to Datetime object
        
        
        # Satellite ephemeris (targ, et, ref, abcorr, obs)
        [satv, ltime] = spice.spkpos( str(sat_NAIF[0]), et, gs+'_TOPO', 'lt+s', gs)
        # Convert to lat/long coords
        rlonlat_tups = [ spice.reclat( satv[i,:] ) for i in range(len(satv))] # List of (r,lon,lat) tupples
        r,lon,lat = np.array(list(zip(*rlonlat_tups)))
        # Convert to Az/El
        el = lat
        az = -lon
        az[az < 0.] += spice.twopi() # Wrap to 2po
        df['Sat.X'] = satv[:,0]
        df['Sat.Y'] = satv[:,1]
        df['Sat.Z'] = satv[:,2]
        df['Sat.Az'] = az
        df['Sat.El'] = el
        df['Sat.R'] = r
        # Sun ephemeris  (targ, et, ref, abcorr, obs)
        [sunv, ltime] = spice.spkpos( 'sun', et, gs+'_TOPO', 'lt+s', gs)   
        # Convert to lat/long coords
        rlonlat_tups = [ spice.reclat( sunv[i,:] ) for i in range(len(sunv))] # List of (r,lon,lat) tupples
        r,lon,lat = np.array(list(zip(*rlonlat_tups)))
        # Convert to Az/El
        el = lat
        az = -lon
        az[az < 0.] += spice.twopi() # Wrap to 2po
        
        df['Sun.X'] = sunv[:,0]
        df['Sun.Y'] = sunv[:,1]
        df['Sun.Z'] = sunv[:,2]
        df['Sun.Az'] = az
        df['Sun.El'] = el
        df['Sun.R'] = r
        
        # Compute the solar phase angle of the satellite (Obs-Sat-Sun angle)
        # This is the angle between the Sat->Obs and Sat->Sun vectors
        # See: https://amostech.com/TechnicalPapers/2010/NROC/Kervin.pdf
        #
        #              O Sun
        #            .
        #          .
        #    Sat o  Phase angle
        #          .
        #            .
        #             x Observer
        
        # Get the vectors
        r1 = -satv       # Satellite -> Groundstation vector (-ve of position vector)
        r2 = sunv - satv # Satellite -> Sun vector
        # Convert to unit vectors
        r1 = r1/np.linalg.norm(r1, axis=-1)[:, np.newaxis]
        r2 = r2/np.linalg.norm(r2, axis=-1)[:, np.newaxis]
        # Find angle between the vectors
        cos_alpha = np.einsum('ij,ij->i',r1,r2) # dot(r1,r2)
        alpha = np.arccos(cos_alpha)
        aplha = np.mod(alpha, 2*np.pi) # Wrap to 2pi
        # Alternative using spice
        # sep = spice.vsep( r1, r2 ) # Find the angular separation angle
        
        # Add to dataframe
        df['Sat.alpha'] = alpha
        
        # Visual magnitude
        
        
        
        
        
        
        # Append to dataframes
        dfs.append(df)

    return dfs

#%% Visual Mangitude

def compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,include_airmass=True):
    
    # Compute satellite apparent magnitude
    # https://www.eso.org/~ohainaut/satellites/22_ConAn_Bassa_aa42101-21.pdf
    # https://www.aanda.org/articles/aa/pdf/2020/04/aa37501-20.pdf
    #
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
    # X = 1/cos(90-El) = 1/sin(El) = airmass in the plane-parallel approximation
    # El = elevation above horizon
    
    # Note: distances should be in AU
    # see: https://www.aanda.org/articles/aa/pdf/2020/04/aa37501-20.pdf
    
    # TODO: Compute airmass using astropy.coordinates.AltAz
    # https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html
    # https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html
    
    # Use airmass to compute atmospheric extinction
    # m(X) = m0 + k*X
    # where X = airmass, m=magnitude, k=extinction coefficient (mags/airmass)
    # see: https://warwick.ac.uk/fac/sci/physics/research/astro/teaching/mpags/observing.pdf
    
    AU = 149597870.700 # Astronomical unit (km)
    
    # Extract Vectors and distances
    sunv = dftopo[['Sun.X','Sun.Y','Sun.Z']].to_numpy() # Sun position
    satv = dftopo[['Sat.X','Sat.Y','Sat.Z']].to_numpy() # Sun position
    
    # Compute distances
    dsat0 = np.linalg.norm(sunv - satv, axis=-1)/AU # Distance Sat to Sun (AU)
    dsat = dftopo['Sat.R'].to_numpy()/AU            # Distance Obs to Sat (AU)
    Rsat = (Rsat/1000)/AU # Radius of satellite in AU
    
    # Phase function for Lambertian sphere
    alpha = dftopo['Sat.alpha'].to_numpy() # Solar phase angle of satellite (rad)
    valpha = (1+np.cos(alpha))/2
    # Exclude airmass
    if include_airmass==False:
        valpha = 1. # To remove
    
    # Compute airmass
    el = dftopo['Sat.El'].to_numpy() # Elevation (rad)
    X = 1./np.sin(el) # airmass in the plane-parallel approximation
    
    # Compute phase
    m0 = -26.75 # Sun's apparent magnitude in V-band
    msat = m0 - 2.5*np.log10(p*(Rsat**2)) + 5*np.log10(dsat0*dsat) -2.5*np.log10(valpha) + k*X 
    
    # Remove magnitude when below horizon
    msat[el<np.deg2rad(0.1)] = np.nan
    
    return msat


#%% Time windows
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
    sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    # spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
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
    MAXWIN = 750 # Maximum number of intervals
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

def find_station_lighting(start_et,stop_et,station='DSS-43'):
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
    station : TYPE, optional
        DESCRIPTION. The default is 'DSS-43'.

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
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    
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
    MAXWIN = 750 # Maximum number of intervals
    cnfine = spice.cell_double(2) # Initialize window of interest
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
    # Settings for geometry search  
    # Find when the solar elevation is < -6 deg
    targ = "SUN" # Target
    frame  = station+"_TOPO" # Reference frame
    abcorr = "NONE" # Aberration correction flag
    obsrvr = station # Observer
    crdsys = "LATITUDINAL" # Coordinate system
    coord  = "LATITUDE" # Coordinate of interest
    refval = -6.*spice.rpd() # Reference value
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
    
    
    # TODO: Bin data based on Sun.El ranges
    # https://towardsdatascience.com/how-i-customarily-bin-data-with-pandas-9303c9e4d946
    
    
    return light, dark


#%% Access

def find_access(start_et,stop_et,station='DSS-43'):
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
    start_et : TYPE
        DESCRIPTION.
    stop_et : TYPE
        DESCRIPTION.
    station : TYPE, optional
        DESCRIPTION. The default is 'DSS-43'.

    Returns
    -------
    access : TYPE
        DESCRIPTION.

    '''
    
    # From documentation, GMAT uses gfposc to perform line-of-sight search above
    # a minimum elevation angle.
    
    # Kernel file directory
    kernel_dir = get_data_home()  / 'Kernels'
    station_ephem_file = str(kernel_dir/'earthstns_itrf93_201023.bsp')
    sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    
    # Get the NAIF IDs of spacecraft from satellite ephemeris file
    ids = spice.spkobj(sat_ephem_file) # SpiceCell object
    numobj = len(ids)
    sat_NAIF = [ids[i] for i in range(numobj) ] # -10002001 NAIF of the satellite
    
    # Load the coverage window of station
    kernel_dir = get_data_home()  / 'Kernels'
    # ids = spice.spkobj(sat_ephem_file)
    cov = spice.spkcov(sat_ephem_file,sat_NAIF[0]) # SpiceCell object
    
    # Create time window of interest
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/windows.html
    MAXWIN = 750 # Maximum number of intervals
    cnfine = spice.cell_double(2) # Initialize window of interest
    spice.wninsd(start_et, stop_et, cnfine ) # Insert time interval in window
    
    # Settings for geometry search  
    # Find when the satellite elevation is > 0 deg
    targ = str(sat_NAIF[0]) # Target
    frame  = station+"_TOPO" # Reference frame
    # abcorr = "NONE" # Aberration correction flag
    abcorr = "lt+s" # Aberration correction flag
    obsrvr = station # Observer
    crdsys = "LATITUDINAL" # Coordinate system
    coord  = "LATITUDE" # Coordinate of interest
    refval = 0.*spice.rpd() # Reference value
    relate = ">"             # Relational operator 
    adjust = 0. # Adjustment value for absolute extrema searches
    step = 10. # Step size (1 hrs)
    
    # Call the function to find full, anular and partial
    access = spice.cell_double(2*MAXWIN) # Initialize result
    access = spice.gfposc(targ,frame,abcorr,obsrvr,crdsys,coord,relate,
                     refval,adjust,step,MAXWIN,cnfine,access)
    
    return access

def constrain_access_by_lighting(access,gslight,satdark):
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


