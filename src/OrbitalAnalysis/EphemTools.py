# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:19:26 2022

@author: scott

Ephemeris tools
---------------

"""

import os
import requests

from utils import get_data_home


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

