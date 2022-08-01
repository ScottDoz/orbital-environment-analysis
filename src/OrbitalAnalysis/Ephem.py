# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:19:26 2022

@author: scott

Ephemeris module
---------------

Contains functions for extracting ephemerides of the satellite, sun, moon, and
groundstations, and computing lighting and access conditions.

"""

import os
import requests
import numpy as np
import pandas as pd
from sys import platform
import subprocess
import shutil
import textwrap
from astropy.time import Time as Time
import spiceypy as spice
from sgp4.api import Satrec


import time
import timeit

from utils import get_data_home, get_root_dir

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
    
    
    
    # Generic kernel downloads ------------------------------------------------
    # Generic kernels needed for most SPICE programs
    #
    # Filename                        Description
    # naif0012.tls                    Leap second kernel
    # pck00010.tpc                    Planetary constants kernel
    # earth_000101_220616_220323.bpc  Earth binary PCK (Jan 2000 - Jun 2022)
    # earth_topo_201023.tf            Earth topocentric frame text kernel
    # geophysical.ker                 Geophysical constants kernel 
    # earthstns_itrf93_201023.bsp     Ephemerides of DSN earth stations
    
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
        
        Alternatively, copy saved file from the data folder.
        '''
        
        # Check data directory
        self._check_data_directory()
        
        if platform == 'win32':
            method = 'copy'
        else:
            method = 'download'
        
        
        if method == 'copy':
            
            print('Copying PCK file.')
            
            # Copy the pck00010.tpc file included in the repo
            root_dir = get_root_dir()
            _dir = root_dir.parent.parent/'data' # Data directory
            fullfilename = _dir/'pck00010.tpc'
            
            # Get kernel directory
            kernel_dir = get_data_home()/'Kernels'
            
            # Copy file using shutil
            shutil.copy(str(fullfilename), str(kernel_dir/'pck00010.tpc'))
            
            print('File {} saved'.format(str(kernel_dir/'pck00010.tpc')))
        
        elif method == 'download':
            # Download the pck00010.tpc file from online.
            # FIXME: Need to convert unix to dos.
            # No easy solution that is robust.
        
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
            
            # if platform == 'win32':
            #     # pck00010.tpc is formated for Linux. Need to convert if using windows.
            #     # Convert LF to CR-LF
                
            #     # TODO: 
                
            #     # # Convert using unix2dos
            #     # shdir = _dir # Directory to run from
            #     # # cmd = 'unix2dos pck00010.tpc' # Command to execute
            #     # cmd = 'TYPE pck00010.tpc | MORE /P >'
            #     # # cmd = 'TYPE pck00010.tpc | MORE /P > pck00010.tpc' # Command to execute'
            #     # result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, cwd=shdir)
            #     # msg = result.stdout.decode("utf-8") # Error message
            #     pass
                
                
            # TODO: Convert LF to CR-LF
            # unix2dos pck00010.tpc pck00010.tpc
        
        return
    
    @classmethod
    def download_earth_binary_pck(self):
        '''
        Download Binary Planetary Constants Kernel for Earth files from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

        '''
        
        # Check data directory
        self._check_data_directory()
        
        # TODO: This file should be updated depending on timeframe of 
        url = "https://naif.jpl.nasa.gov/pub/naif/EXOMARS2016/kernels/pck/earth_000101_220616_220323.bpc"

        # Get the request
        print('Downloading earth_000101_220616_220323.bpc file.')
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
    def download_earth_topo_tf(self):
        '''
        Download Text Kernel for Earth Topocentric frame from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/stations/

        '''
        
        # Check data directory
        self._check_data_directory()
        
        # TODO: This file should be updated depending on timeframe of 
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/stations/earth_topo_201023.tf"

        # Get the request
        print('Downloading earth_topo_201023.tf file.')
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
    def download_geophysical(self):
        '''
        Download Geophysical kernel from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/ock/

        '''
        
        # Check data directory
        self._check_data_directory()
        
        # TODO: This file should be updated depending on timeframe of 
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/geophysical.ker"

        # Get the request
        print('Downloading geophysical.kerfile.')
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
    def download_gravity_parameters_tpc(self):
        '''
        Download planetary gravitational data SPK files from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck

        '''
        
        # Check data directory
        self._check_data_directory()

        # DE440
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de431.tpc"

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
    def download_earthstations(self):
        '''
        Download ephemeris SPK files for DSN earthstations network data from:
        https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/

        '''
        
        # Check data directory
        self._check_data_directory()

        # URL
        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/earthstns_itrf93_201023.bsp"

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
    
    # -------------------------------------------------------------------------
    # Command line utilities
    #
    # SPICE command line exe files. Needed for the generation of spice kernels.
    #
    # Filename          Description
    # MKSPK.exe         Creates an SPK file from a text file containing trajectory information.
    # PINPOINT.exec     Create an SPK file and optionally, an FK file, for a set 
    #                   of objects having fixed locations or constant velocities 
    #                   relative to their respective centers of motion.
    
    @classmethod
    def install_mkspk(self):
        '''
        Download mkspk files from:
        https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/mkspk.exe

        '''
        
        # Check data directory
        self._check_data_directory()

        # DE440
        url = "https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/mkspk.exe"

        # Get the request
        print('Installing MKSPK.exe')
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
    def install_pinpoint(self):
         '''
         Installing pinpoint from:
         https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/pinpoint.exe
    
         '''
         
         # Check data directory
         self._check_data_directory()
    
         # DE440
         url = "https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/pinpoint.exe"
    
         # Get the request
         print('Installing pinpoint.exe')
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
    def check_generic_kernels(self):
        ''' Check to ensure all generic kernels needed for analysis are present. '''
        
        # Get kernel directory
        kernel_dir = get_data_home() / 'Kernels'
        
        # Check Leap second kernel
        filename = kernel_dir/'naif0012.tls'
        if filename.exists() == False:
            print("Missing Leap Second Kernel")
            SPICEKernels.download_lsk()
        
        # Check Planetary Constants Kernel
        filename = kernel_dir/'pck00010.tpc'
        if filename.exists() == False:
            print("Missing Planetary Constants Kernel")
            SPICEKernels.download_pck()
        
        # DE440 Solar System Ephemeris
        filename = kernel_dir/'de440s.bsp'
        if filename.exists() == False:
            print("Missing DE440s Solar System Ephemeris")
            SPICEKernels.download_planet_spk()
        
        # Earth binary PCK (Jan 2000 - Jun 2022)
        filename = kernel_dir/'earth_000101_220616_220323.bpc'
        if filename.exists() == False:
            print("Missing Earth Binary PCK")
            SPICEKernels.download_earth_binary_pck()
        
        # Earth topocentric frame text kernel
        filename = kernel_dir/'earth_topo_201023.tf'
        if filename.exists() == False:
            print("Missing Earth topocentric frame text kernel")
            SPICEKernels.download_earth_topo_tf()
        
        # Geophysical constants kernel
        filename = kernel_dir/'geophysical.ker'
        if filename.exists() == False:
            print("Missing Geophysical constants kernel")
            SPICEKernels.download_geophysical()
        
        # Gravity parameters
        filename = kernel_dir/'gm_de431.tpc'
        if filename.exists() == False:
            print("Missing Gravity parameters kernel")
            SPICEKernels.download_gravity_parameters_tpc()
           
        # Earth stations
        filename = kernel_dir/'earthstns_itrf93_201023.bsp'
        if filename.exists() == False:
            print("Missing Earthstations kernel")
            SPICEKernels.download_earthstations()
        
        return

# TODO: Convert pck00010.tpc to native. Need to include unix2dos in requirements: conda install -c conda-forge unix2dos

#%% Ephemeris generation

def create_satellite_ephem(sat,start_et,stop_et,step,method='tle'):
    
    # Check format of satellite data
    if type(sat)==int:
        # NORAD ID. Data from published TLEs
        data_type = 'NORAD'
    elif type(sat)==dict:
        # Elements provided by dictionary.
        data_type = 'dict'
    
    
    
    
    # Set directory to save into
    DATA_DIR = get_data_home() # Data home directory
    kernel_dir = DATA_DIR  / 'Kernels'
    
    # Get ephemeris
    if method.lower() in ['sgp4']:
        # SGP4 Propagator. From satellite TLE
        from SatelliteData import get_tle
        
        # Get TLEs for the object
        tle_lines = get_tle(sat,tle_type='tle')
        l1 = tle_lines[0] # Line 1
        l2 = tle_lines[1] # Line 2
        
        # Create object
        satellite = Satrec.twoline2rv(l1, l2)
        
        # Create array of epochs
        et = np.arange(start_et,stop_et,step)    # Ephemeris time
        jd = spice.timout(et,'JULIAND.#############').astype(float) # Julian date
        
        # Compute state vectors at epochs
        jd_whole = np.floor(jd).astype(int) # Whole part of jd
        jd_frac = jd % 1 # Fraction part of jd
        e, r, v = satellite.sgp4_array(jd_whole, jd_frac)
        # e is error flag - non-zero for errors
        
        # TODO: Finish
        
    elif method.lower() in ['tle']:
        # MKSPK TL_Elements option
        # Ephemeris from TLEs using SGP4 propagator.
        
        # Generate from TLEs
        # Get TLEs over epoch range and write to file
        from SatelliteData import get_tle
        tle_lines = get_tle(sat,epoch=[start_et,stop_et],tle_type='tle')
        infile = kernel_dir/'sat_TLEs.tle' # Filename
        print('Writing TLE file {}'.format(str(infile)), flush=True)
        with open(str(infile), 'w+') as fp:
            for line in tle_lines:
                fp.write(line + '\n')
        
        
        
        # Create setup file for mkspk
        template = """
                    \\begindata
                    ?INPUT_DATA_TYPE   = 'TL_ELEMENTS'
                    ?OUTPUT_SPK_TYPE   = 10
                    ?TLE_INPUT_OBJ_ID  = {ID}
                    ?TLE_SPK_OBJ_ID    = {ID}
                    ?CENTER_ID         = 399
                    ?REF_FRAME_NAME    = 'J2000'
                    ?TLE_START_PAD     = '2 days'
                    ?TLE_STOP_PAD      = '2 days'
                    ?LEAPSECONDS_FILE  = 'naif0012.tls'
                    ?INPUT_DATA_FILE   = 'sat_TLEs.tle'
                    ?OUTPUT_SPK_FILE   = 'sat.bsp'
                    ?PCK_FILE          = 'geophysical.ker'
                    ?SEGMENT_ID        = 'Satellite TLE-based Trajectory'
                    ?PRODUCER_ID       = 'Scott Dorrington, MIT'
                    \\begintext
                    """
        template = textwrap.dedent(template) # Remove formating indents
        template = template.replace('?', '   ') # Replace
        
        # Format the template file and write
        defs_filename = kernel_dir/'sat_setup.txt'
        print('Writing setup file {}'.format(str(defs_filename)), flush=True)
        with  open(defs_filename,'w') as myfile:
            myfile.write(template.format(**{ "ID":str(sat)} ))
    
        # Define output spk file
        outfile = kernel_dir/'sat.bsp'
        if outfile.exists():
            # If file exists, delete it.
            spice.kclear() # Clear all loaded kernels
            outfile.unlink()
    
        # Run the mkspk
        print('Making spk file with MKSPK.exe', flush=True)
        shdir = kernel_dir # Directory to run from
        cmd = 'mkspk -setup {setupfile} -input {infile} -output {outfile}'.format(**{'setupfile':defs_filename.name,'infile':str(infile), 'outfile':str(outfile) }) # Command to execute
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, cwd=shdir)
        msg = result.stdout.decode("utf-8") # Error message
        
        # Print any errors
        if msg != '':
            print(msg, flush=True)
    
    
    elif method.lower() in ['2 body','two-body']:
        # MKSPK ELEMENTS option
        # Ephemeris from orbital elements using two-body propagator
        
        # Extract data
        epoch = spice.timout(start_et,pictur='YYYY MON DD HR:MN:SC.#')
        start = spice.timout(start_et,pictur='YYYY MON DD')
        stop = spice.timout(stop_et,pictur='YYYY MON DD')
        a = sat['SMA']   # 
        ECC = sat['ECC'] # Eccentricty
        INC = sat['INC'] # Inclination (deg)
        LNODE = sat['RAAN'] # Longitude of ascending node
        ARGP = sat['AOP'] # Argument of periapsis (deg)
        M0 = 0. # Mean anomaly at epoch (deg)
        
        # Get gravitiational constant of Earth
        spice.furnsh( str(kernel_dir/'gm_de431.tpc') ) # Gravity constants for planets
        _, GM = spice.bodvrd ( "EARTH", "GM", 1); GM = float(GM) # km^3/s^2
        
        # Create array of epochs (extend 1 hour either side)
        et = np.arange(start_et-86400.,stop_et+86400.,step)    # Ephemeris time
        
        # Compute Period and mean motion
        T = 2*np.pi*np.sqrt(a**3/GM) # Orbital period (s)
        n = 2*np.pi/T # Mean motion (rad/s)
        
        # Propagate two-body
        # Get Mean anomaly at epochs
        # Using two-body propagation
        # M(t) = M0 + n*(t - t0)
        M = M0 + n*(et - start_et)
        M = M%(2*np.pi) # Wrap to [0, 2*pi]
        
        
        # Generate input data file
        data_template = """
                    Test: 99999 relative to 399 in frame J2000. GM= 398600.4354360959
                    JD, A, ECC, INC, LNODE, ARGP, M0
                    ---------------------------------
                    {epoch}
                    ?{A} {ECC} {INC} {LNODE} {ARGP} {M0}
                     """
        data_template = textwrap.dedent(data_template) # Remove formating indents
        data_template = data_template.replace('?', ' ') # Replace
        
        # Format the template file and write
        infile = kernel_dir/'sat_input.txt'
        print('Writing input file {}'.format(str(infile)), flush=True)
        with  open(infile,'w') as myfile:
            
            # Write header
            myfile.write('Test: 99999 relative to 399 in frame J2000. GM= 398600.4354360959' + '\n')
            myfile.write('JD, A, ECC, INC, LNODE, ARGP, M0' + '\n')
            myfile.write('---------------------------------' + '\n')
            
            # Write epoch data
            for i in range(len(et)):
                # myfile.write(spice.timout(et[i],pictur='YYYY MON DD HR:MN:SC.#') + '\n') # Epoch
                myfile.write(str(et[i]) + '\n') # Epoch
                myfile.write(' {A} {ECC} {INC} {LNODE} {ARGP} {M0} \n'.format(**{"A":str(a),
                                                                                 "ECC":str(ECC),"INC":str(INC),
                                                                                 "LNODE":str(LNODE),"ARGP":str(ARGP),
                                                                                 "M0":np.rad2deg(M[i])} ))
            # myfile.write(data_template.format(**{ "epoch":epoch,"A":str(a),
            #                                       "ECC":str(ECC),"INC":str(INC),
            #                                       "LNODE":str(LNODE),"ARGP":str(ARGP),
            #                                       "M0":M0} ))
        
        # Create setup file for mkspk
        template = """
                    \\begindata
                    ?INPUT_DATA_TYPE   = 'ELEMENTS'
                    ?OUTPUT_SPK_TYPE   = 5
                    ?OBJECT_ID         = 99999
                    ?OBJECT_NAME       = 'Target'
                    ?CENTER_ID         = 399
                    ?CENTER_NAME       = 'EARTH'
                    ?REF_FRAME_NAME    = 'J2000'
                    ?PRODUCER_ID       = 'Scott Dorrington, MIT'
                    ?DATA_ORDER        = 'EPOCH A E INC NOD PER MEAN'
                    ?TIME_WRAPPER      = '# ETSECONDS'
                    ?INPUT_DATA_UNITS  = ('ANGLES=DEGREES' 'DISTANCES=km')
                    ?DATA_DELIMITER    = ' '
                    ?LINES_PER_RECORD  = 2
                    ?IGNORE_FIRST_LINE = 3
                    ?LEAPSECONDS_FILE  = 'naif0012.tls'
                    ?PCK_FILE          = 'gm_de431.tpc'
                    ?SEGMENT_ID        = 'SPK_ELEMENTS_05'
                    \\begintext
                    """
        template = textwrap.dedent(template) # Remove formating indents
        template = template.replace('?', '   ') # Replace
        # Format the template file and write
        defs_filename = kernel_dir/'sat_setup.txt'
        print('Writing setup file {}'.format(str(defs_filename)), flush=True)
        with  open(defs_filename,'w') as myfile:
            myfile.write(template.format(**{ "start":start,"stop":stop} ))
        
        # Define output spk file
        outfile = kernel_dir/'sat.bsp'
        if outfile.exists():
            # If file exists, delete it.
            spice.kclear() # Clear all loaded kernels
            outfile.unlink()
        
        # Run the mkspk
        print('Making spk file with MKSPK.exe', flush=True)
        shdir = kernel_dir # Directory to run from
        cmd = 'mkspk -setup {setupfile} -input {infile} -output {outfile}'.format(**{'setupfile':defs_filename.name,'infile':str(infile), 'outfile':str(outfile) }) # Command to execute
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, cwd=shdir)
        msg = result.stdout.decode("utf-8") # Error message
        
        # Print any errors
        if msg != '':
            print(msg, flush=True)
        
    
    # Write SPK file
    
    # NAIF links for mkspk
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/mkspk.html
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/42_making_an_spk.pdf
    
    # SPK Data type
    # Type 3 - Chebyschev polynomial. Used for satellites
    # Type 5 - Descrete states (two body propagation)
    # Type 10 - Two Line Element Sets 
    
    
    
    return


def create_station_ephem(df, network_name='SSR'):
    '''
    Create a set of SPICE files containing ephemerides and topocentric reference
    frames for a set of ground stations.
    
    Station names and coordinates provided in the input dataframe are filled in
    to a templace Definitions file. The SPICE PINPOINT command line utility is
    then used to generate the ephemeris (.bsp file) and frame kernels (.tf file).
    
    See: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/pinpoint.ug
    
    Note: This function requires pinpoint.exe to be available. 
          It can be downloaded from:
          https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe containing the names, codes and coordinates of the stations
        for which the SPK and FK are to be created.

    '''
    # # Example format
    # \begindata
     
    #    SITES         = ( 'DSS-12',
    #                      'DSS-13' )
     
    #     DSS-12_CENTER = 399
    #     DSS-12_FRAME  = 'EARTH_FIXED'
    #     DSS-12_IDCODE = 399012
    #     DSS-12_XYZ    = ( -2350.443812, -4651.980837, +3665.630988 )
    #     DSS-12_UP     = 'Z'
    #     DSS-12_NORTH  = 'X'
     
    #     DSS-13_CENTER = 399
    #     DSS-13_FRAME  = 'EARTH_FIXED'
    #     DSS-13_IDCODE = 399013
    #     DSS-13_XYZ    = ( -2351.112452, -4655.530771, +3660.912823 )
    #     DSS-13_UP     = 'Z'
    #     DSS-13_NORTH  = 'X'
     
    # \begintext
    
    # Definitions file
    kernel_dir = get_data_home()  / 'Kernels'
    defs_filename = kernel_dir/str(network_name+'_stations.defs')
    
    # Get number of sights
    nSites = len(df)
    
    # Definitions file template
    template = """
                \\begindata
                
                \tSITES         = ( {sites} )
                
                {details}
                 
                \\begintext"""
    template = textwrap.dedent(template) # Remove formating indents
    template = template.replace('\t', '   ') # Replace tabs with spaces
    
    # Configure details blocks for each groundstation
    # Use ? as placefiller for 3 spaces
    details_template = """
                        ?{NAME}_CENTER = 399
                        ?{NAME}_FRAME  = 'EARTH_FIXED'
                        ?{NAME}_IDCODE = {CODE}
                        ?{NAME}_XYZ    = ( {x}, {y}, {z} )
                        ?{NAME}_UP     = 'Z'
                        ?{NAME}_NORTH  = 'X'
                        
                        """
    details_template = textwrap.dedent(details_template) # Remove formating indents
    details_template = details_template.replace('?', '   ') # Replace 
    
    # Configure SITES text in template
    sites_list = df.Name.to_list()
    # sites_list = ['DSS-12','DSS-13']
    sites_txt = "'{0}'".format("', \n                     '".join(sites_list))
    
    # Configure details text
    details_list = [details_template.format(**{'NAME':df['Name'].iloc[i], 
                                             'CODE':str(df['NAIF'].iloc[i]),
                                             'x':str(df['x'].iloc[i]),
                                             'y':str(df['y'].iloc[i]),
                                             'z':str(df['z'].iloc[i]),
                                             } ) for i in range(nSites) ]
    details_txt = '\n'.join(details_list)
    
    # Format the template file and write
    with  open(defs_filename,'w') as myfile:
        myfile.write(template.format(**{ "sites":sites_txt,"details":details_txt} ))
    
    
    # Output files
    outfile_bsp = kernel_dir/'{}_stations.bsp'.format(network_name) # Ephemeris file
    outfile_tf = kernel_dir/'{}_stations.tf'.format(network_name)  # Reference frame file

    # Check if files exist
    if outfile_bsp.exists():
        outfile_bsp.unlink()
    if outfile_tf.exists():
        outfile_tf.unlink()
    
    # Run the PINPOINT function
    cmd = 'pinpoint -def {file} -pck pck00010.tpc -spk {nn}_stations.bsp -fk  {nn}_stations.tf'.format(**{'file':defs_filename.name,'nn':network_name}) # Command
    
    if platform == 'linux':
        # Linux
        pass
    elif platform == 'win32':
        # Windows
        shdir = kernel_dir # Directory to run from
        # p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, cwd=shdir)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, cwd=shdir)
        msg = result.stdout.decode("utf-8") # Error message
        
        # Print any errors
        if msg != '':
            print(msg)
    
    return

# Download pinpoint.exe from
# https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html
# Useful links
# https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/pinpoint.ug
# https://space.stackexchange.com/questions/52207/spice-defining-a-topocentric-frame-on-the-moon
# https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Creating%20a%20Frame%20Kernel
# https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_64bit.html
# pinpoint -def pinpoint_ex1.defs -pck pck00010.tpc -spk dss_12_13.bsp -fk  dss_12_13.tf



#%% Ephemerides in Earth-fixed frame

def get_ephem_ITFR(et,groundstations=['DSS-43']):
    
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
    # DSN Stations
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    # SSR Stations
    spice.furnsh( str(kernel_dir/'SSR_stations.bsp') ) # SSR station Ephemerides
    spice.furnsh( str(kernel_dir/'SSR_stations.tf') )  # SSR topocentric frame text kernel
    # SSRD Stations
    spice.furnsh( str(kernel_dir/'SSRD_stations.bsp') ) # SSRD station Ephemerides
    spice.furnsh( str(kernel_dir/'SSRD_stations.tf') )  # SSRD topocentric frame text kernel
    
    
    
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
    # sat_ephem_file = str(get_data_home()/'GMATscripts'/'Access'/'EphemerisFile1.bsp')
    sat_ephem_file = str(get_data_home()/'Kernels'/'sat.bsp')
    station_ephem_file = str(kernel_dir/'earthstns_itrf93_201023.bsp')
    
    # Load Kernels
    
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    # Ephemerides
    spice.furnsh( str(kernel_dir/'de440s.bsp') )   # Solar System ephemeris
    spice.furnsh( sat_ephem_file ) # Satellite
    # Frame kernels
    spice.furnsh( str(kernel_dir/'earth_000101_220616_220323.bpc') ) # Earth binary PCK (Jan 2000 - Jun 2022)
    # DSN stations
    spice.furnsh( str(kernel_dir/'earthstns_itrf93_201023.bsp') ) # DSN station Ephemerides 
    spice.furnsh( str(kernel_dir/'earth_topo_201023.tf') ) # Earth topocentric frame text kernel
    # SSR stations
    spice.furnsh( str(kernel_dir/'SSR_stations.bsp') ) # SSR station Ephemerides
    spice.furnsh( str(kernel_dir/'SSR_stations.tf') )      # SSR topocentric frame text kernel
    # SSRD Stations
    spice.furnsh( str(kernel_dir/'SSRD_stations.bsp') ) # SSRD station Ephemerides
    spice.furnsh( str(kernel_dir/'SSRD_stations.tf') )  # SSRD topocentric frame text kernel
    
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
    
    # TODO: Constrain by max value
    
    return msat



