# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:19:26 2022

@author: scott

Ephemeris module
---------------

Contains functions for extracting ephemerides of the satellite, sun, moon, and
groundstations.


Note: use fo the cspice command line tools requires downloading and installing
the cspice library from https://naif.jpl.nasa.gov/naif/utilities.html
Individual exe files can be downloaded from 
https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html

"""

import os
import requests
import numpy as np
import pandas as pd
from astropy.time import Time as astroTime
import spiceypy as spice

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
    
    # Create dataframe for output
    df = pd.DataFrame(columns=['ET',
                               'Sat.X','Sat.Y','Sat.Z',
                               'Sun.X','Sun.Y','Sun.Z'])
    df.ET = et
    
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
    
    
    # Generate new dataframe for each of the ground stations
    dfs = []
    for gs in groundstations:
    
        # Create dataframe for output
        df = pd.DataFrame(columns=['ET',
                                   'Sat.X','Sat.Y','Sat.Z','Sat.Az','Sat.El','Sat.R',
                                   'Sun.X','Sun.Y','Sun.Z','Sun.Az','Sun.El','Sun.R',])
        df.ET = et
        
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
        
        # Append to dataframes
        dfs.append(df)

    return dfs

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






