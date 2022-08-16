# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:33:03 2022

@author: scott

Satellite Data Module
---------------------

Methods to download and import TLE data from online sources.
Method to compute additional orbita parameters.

For description of TLEs see: https://kaitlyn.guru/projects/two-line-elements-tle/

"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import os
import configparser
import spiceypy as spice
from sklearn.decomposition import PCA

import pdb

# Relative imports
from utils import get_data_home


#%% ASTRIAGraph access

# TODO: Add functions to load/query data from ASTRIAGraph

#%% Load Catalog Data

# Celestrak has subsets of elements here
# https://celestrak.com/NORAD/elements/

# Spacetrack has all data, but requires login

# Qualitative RCS Sizes (m^2)
# Small  :       RCS < 0.1
# Medium : 0.1 < RCS < 1.0
# Large  : 1.0 < RCS

def load_satellites(group='all',compute_params=True,compute_pca=True):
    '''
    Load TLEs of special interest satellites from Celestrak or all objects from
    SpaceTrack. Return data as a dataframe.

    Parameters
    ----------
    group : str, optional
        String indicating subgroup of elements to load, including:
        ['active','analyst','stations','visual','last-30-days','active-geo',
        'gpz','gpz-plus','all']
        The default is 'all'.
    compute_params : Bool, optional
        Flag indicating if additional orbital parameters should be computed. 
        The default is True.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe containing TLEs and orbital parameters.

    '''

    import tletools
    import os, tempfile
    from urllib.request import urlopen
    
    # Get data directory
    DATA_DIR = get_data_home()
    
    # Select the url of the group of satellites
    
    # Special interest satellites ---------------------------------------------
    if group.lower() == 'active':
        # Active satellites (~5100)
        # url = 'https://celestrak.com/NORAD/elements/active.txt' 
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    elif group.lower() == 'analyst':
        # Analyst satellites (~400)
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=analyst' 
    elif group.lower() in ['space stations','stations']:
        # Space Stations (66)
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle'
    elif group.lower() == 'visual':
        # Brightest (~160)
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle'
    elif group.lower() in ['last-30-days']:
        # Last 30 Days
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle'
    
    # GEO region --------------------------------------------------------------
    elif group.lower() == 'active-geo':
        # Active Geosynchronous
        url = 'https://celestrak.com/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle'
    elif group.lower() in ['geo-protected','gpz']:
        # Geo protected zone (~800)
        url = 'https://celestrak.com/NORAD/elements/gp.php?SPECIAL=gpz&FORMAT=tle'
    elif group.lower() in ['geo-protected-plus','gpz-plus']:
        # Geo protected zone plus (~1780)
        url = 'https://celestrak.com/NORAD/elements/gp.php?SPECIAL=gpz-plus&FORMAT=tle'
    
    # TODO: add more groups
    
    # Space-Track -------------------------------------------------------------
    elif group.lower() == 'all':
        # All objects from Spacetrack (~24,000)
        
        # Check if file exists
        if os.path.isfile(DATA_DIR/'tle_latest.txt') == False:
            # Save data
            download_all_spacetrack()
        
        # Load data
        filename = DATA_DIR/'tle_latest.txt'
        df = tletools.load_dataframe(str(filename))
        df.norad = df.norad.astype(int) # Convert norad to int
        
        # Load satcat and merge
        dfs = pd.read_csv(DATA_DIR/'satcat.csv')
        df = pd.merge(df,dfs,how='left',left_on='norad',right_on='NORAD_CAT_ID')
        
        # Rename some columns to match ASTRIAGraph
        df = df.rename(columns = {'name':'Name','norad':'NoradId','epoch':'Epoch'})
        
        # Remove 0 from name
        df['Name'][df['Name'].str[:2]=='0 '] = df['Name'].str[2:]
        
        # Compute orbital parameters
        if compute_params:
            df = compute_orbital_params(df)
        
        # Compute principal components
        if compute_pca:
            df = compute_principal_components(df,n_components=10)
        
        return df
    
    # Saving data for Celestrack groups

    # Read in text from url
    urltxt = urlopen(url)
    
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    for line in urltxt:
        if line.find(b"Format invalid") != 0:
            tmp.write(line)
    tmp.seek(0)
    
    # Read in data to dataframe
    df = tletools.load_dataframe(tmp.name)
    
    # Close the file to remove it
    tmp.close()
    
    # Compute orbital parameters
    if compute_params:
        df = compute_orbital_params(df)
    
    # Compute principal components
    if compute_pca:
        df = compute_principal_components(df,n_components=10)
    
    # Convert norad to int
    df.norad = df.norad.astype(int)
    
    return df

def download_all_spacetrack():
    '''
    Download the full TLE and SATCAT catalogs from SpaceTrack to a txt file.
    
    '''
    
    # Get data directory
    DATA_DIR = get_data_home()
    
    # Read spacetrack email and password from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    email = config['Spacetrack']['email']
    pw = config['Spacetrack']['pw']
    # email = 's.dorrington@unswalumni.com'
    # pw = 'bgzWd4j9L8xr3Db'
    
    # Set up connection to client 
    from spacetrack import SpaceTrackClient
    st = SpaceTrackClient(email, pw)
    
    import spacetrack.operators as op
    
    # https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID,EPOCH/format/3le
    
    # Stream download line by line
    data = st.tle_latest(iter_lines=True, ordinal=1, epoch='>now-30',
                     # mean_motion=op.inclusive_range(0.0, 20.01), # (0.99,1.01)
                     # eccentricity=op.less_than(0.01), 
                     format='3le')
    
    # Write data to file
    with open(DATA_DIR/'tle_latest.txt', 'w') as fp:
        for line in data:
            fp.write(line + '\n')
    
    
    # Download SATCAT
    data = st.satcat(iter_lines=True, orderby='launch desc', format='csv')
    
    # Write data to file
    with open(DATA_DIR/'satcat.csv', 'w') as fp:
        for line in data:
            fp.write(line + '\n')
    
    
    return

#%% Query TLEs by NORAD

def get_tle(ID,epoch='latest',tle_type='3le'):
    ''' Query TLEs for a single NORAD ID return a list of the lines '''
    
    # Read spacetrack email and password from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    email = config['Spacetrack']['email']
    pw = config['Spacetrack']['pw']
    
    # Get data directory
    DATA_DIR = get_data_home()
    
    # Set up connection to client 
    from spacetrack import SpaceTrackClient
    import spacetrack.operators as op
    st = SpaceTrackClient(email, pw)
    
    # Query TLE string from Spacetrack
    if epoch=='latest':
        # Get the latest tle
        tle_string = st.tle_latest(norad_cat_id=ID, ordinal=1, format=tle_type)
        tle_lines = tle_string.strip().splitlines() # Separated lines
    else:
        
        # Create time range
        start_et = epoch[0]
        stop_et = epoch[1]
        d1 = spice.et2datetime(start_et)
        d2 = spice.et2datetime(stop_et)
        drange = op.inclusive_range(d1,d2)
        # Remove any extra time portions
        drange = drange.replace('+00:00','')
        # drange = '2020-10-26 16:00:00--2020-11-25 15:59:59.999495'
        lines = st.tle_publish(norad_cat_id=ID,iter_lines=True, publish_epoch=drange, orderby='TLE_LINE1', format='tle')
        
        # Extract lines
        tle_lines = [line for line in lines]
        
    # # Load data into TLE object
    # from tletools import TLE
    # tle = TLE.from_lines(*tle_lines)
    # data = tle.__dict__ # Extract data as dictionary
    
    return tle_lines


def query_norad(IDs,compute_params=True):
    '''
    Query TLEs for a list of objects by their NORAD IDs.
    Returns data in a pandas dataframe with the same field names as those above.
    '''
    
    # Input check
    if type(IDs)==int:
        IDs = [IDs]
    
    # Read spacetrack email and password from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    email = config['Spacetrack']['email']
    pw = config['Spacetrack']['pw']
    
    # Get data directory
    DATA_DIR = get_data_home()
    
    # Set up connection to client 
    from spacetrack import SpaceTrackClient
    st = SpaceTrackClient(email, pw)
    
    from tletools import TLE
    
    # Loop through IDs
    df = pd.DataFrame()
    for ID in IDs:
        
        # Query TLE string from Spacetrack
        tle_string = st.tle_latest(norad_cat_id=ID, ordinal=1, format='3le')
        tle_lines = tle_string.strip().splitlines() # Separated lines
        # Load data into TLE object
        tle = TLE.from_lines(*tle_lines)
        data = tle.__dict__ # Extract data as dictionary
        # Convert to dataframe
        row = pd.DataFrame.from_dict(data, orient='index').T
        df = df.append(row) # Append
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Fix data types
    df.norad = df.norad.astype(int) # Convert norad to int
    df.n = df.n.astype(float)
    df.ecc = df.ecc.astype(float)
    df.inc = df.inc.astype(float)
    df.raan = df.raan.astype(float)
    df.argp = df.argp.astype(float)
    
    # Remove 0 from name
    df['name'][df['name'].str[:2]=='0 '] = df['name'].str[2:]
    
    # Load satcat and merge
    dfs = pd.read_csv(DATA_DIR/'satcat.csv')
    df = pd.merge(df,dfs,how='left',left_on='norad',right_on='NORAD_CAT_ID')
    
    # Rename some columns to match ASTRIAGraph
    df = df.rename(columns = {'name':'Name','norad':'NoradId','epoch':'Epoch'})
    
    # Compute orbital parameters
    if compute_params:
        df = compute_orbital_params(df)
    
    
    return df


# Alternative methods for querying single objects
    
# # Satellite TLE
# from satellite_tle import fetch_tle_from_celestrak

# # Fetch TLEs for a single satellite from Celestrak
# norad_id_iss = 25544 # ISS (ZARYA)
# print(fetch_tle_from_celestrak(norad_id_iss))

# # Fetch a large set of TLEs
# norad_ids = [25544, # ISS (ZARYA)
#              42983, # QIKCOM-1
#              40379] # GRIFEX

# # Uses default sources and compares TLE set from each source and
# # returns the latest one for each satellite
# from satellite_tle import fetch_all_tles, fetch_latest_tles
# tles = fetch_latest_tles(norad_ids)


# TLE-tools

# https://federicostra.github.io/tletools/

# Small library to work with two-line element sets
# - Parse TLE sets into convenient TLE objects
# - Load entire TLE set files into pandas
# - convert TLE objects into poliastro.twobody.Orbits

# import tletools
# from tletools import TLE

# # Example from TLE text string
# tle_string = """
# ISS (ZARYA)
# 1 25544U 98067A   19249.04864348  .00001909  00000-0  40858-4 0  9990
# 2 25544  51.6464 320.1755 0007999  10.9066  53.2893 15.50437522187805
# """

# tle_lines = tle_string.strip().splitlines()
# tle = TLE.from_lines(*tle_lines)

#%% Load Experiment data ------------------------------------------------------

# Experimental data from Vishnu
# https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/LHX5KM
# Download and unzip files to data home
# ~/satellite_data/Data/dataverse_files

def load_vishnu_experiment_data(mm, compute_params=True):
    '''
    Load monthly satellite data from the year 2019.
    
    Experimental data used in:
    Nair, Vishnu, 2021, "Statistical Families of the Trackable Earth-Orbiting 
    Anthropogenic Space Object Population in Their Specific Orbital Angular 
    Moment Space", https://doi.org/10.18738/T8/LHX5KM, 
    Texas Data Repository, V2
    
    Parameters
    -------
    mm : int, list, or str
        Month(s) of year to load (1-12).
        int: Month of year to load (1-12).
        list: Load data for a list of months
        'all': Load data for all months
    
    '''
    
    if isinstance(mm, str):
        if mm.lower() == 'all':
            # Load all data
            mm = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    # Loading cases
    if isinstance(mm, int):
        # Load data for single month
        df = _load_vishnu_experiment_data_single_month(mm, compute_params=compute_params)
    
    elif isinstance(mm, list):
        # Load multiple datasets from a list of months
        df = df = pd.DataFrame()
        for m in mm:
            dfm = _load_vishnu_experiment_data_single_month(m, compute_params=compute_params)
            df = df.append(dfm) # Append
    
        # Sort data by epoch
        df = df.sort_values(by=['NoradId','Epoch']).reset_index(drop=True)
    
    return df


def _load_vishnu_experiment_data_single_month(mm: int, compute_params=True):
    '''
    Load satellite data for a single month
    
    Parameters
    -------
    mm : int
        Month of year to load (1-12).

    '''
    
    # Error checking
    if not isinstance(mm, int):
        raise ValueError('mm must be int')
    if (mm < 1) or (mm > 12):
        raise ValueError('mm must be month 1 - 12')
    
    
    # Get filename from month
    DATA_DIR = get_data_home()/'dataverse_files' # Data path
    mm = str(mm).zfill(2) # Zero padded string 01-12
    filename = DATA_DIR/'SpaceObjects-2019{}01.json'.format(mm)
    
    # Read json data
    with open(str(filename)) as f:
        result = json.load(f)
    
    # Read in each cluster
    df = pd.DataFrame()
    for k in result.keys():
        dfk = pd.json_normalize(result[k]['Objects'])
        dfk['vishnu_cluster'] = k  # Add cluster label
        df = df.append(dfk) # Append
    
    # Rename columns
    df = df.rename(columns={'SMA':'a','Ecc':'e','Inc':'i','RAAN':'om','ArgP':'w',
                       'MeanAnom':'M'})
    
    # Convert semimajor axis to km
    df.a = df.a/1000.
    
    # Convert angles to degrees (consistent with Spacktrack data)
    df.i = np.rad2deg(df.i)
    df.om = np.rad2deg(df.om)
    df.w = np.rad2deg(df.w)
    df.M = np.rad2deg(df.M)
    
    # Drop missing data
    df = df[pd.notnull(df.a)]
    
    # Convert NoradId to int
    # df.NoradId = df.NoradId.astype(int) # Convert norad to int
    df.NoradId = pd.to_numeric(df['NoradId'], errors='coerce').astype(pd.Int64Dtype())
    
    # Sort by NoradId
    df = df.sort_values(by=['NoradId','Name']).reset_index(drop=True)
    
    # # Remove duplicates
    # # Many objects contain two entries: one with a Name and one without.
    # # Keep the named object (first entry after sort) and drop the unnamed one.
    # df = df.drop_duplicates(subset='NoradId',keep='last')
    
    # For now, drop all duplicates
    df = df.drop_duplicates(subset='NoradId',keep=False)
    
    # Compute orbital parameters
    if compute_params:
        df = compute_orbital_params(df)
    
    return df



#%% Compute orbital Parameters ------------------------------------------------

def compute_orbital_params(df):
    '''
    Compute orbital parameters from the catalog of TLEs.
    Append results to original dataframe.
    
    Note: Some name changes to orbital element fields.

    '''
    
    # Constants
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Rename columns
    df = df.rename(columns={'ecc':'e','inc':'i','raan':'om','argp':'w'})
    
    
    # Compute a from mean motion
    # n = sqrt(mu/a^3)
    # a = (mu/n^2)^(1/3)
    # n = mean motion (revs/day)
    # n1 = n*2*pi/86400 (rad/s)
    if 'a' not in df.columns:
        df['a'] = ( mu/(df.n*2*np.pi/86400)**2  )**(1/3)
    
    
    # Compute semi-latus rectum, periapsis and apoapsis
    df['p'] = df.a*(1-df.e**2) # Semi-latus rectum (km)
    df['q'] = df.a*(1-df.e) # Periapsis (km)
    df['Q'] = df.a*(1+df.e) # Apoapsis (km^2/s)
    
    # Compute specific angular momentum
    df['h'] = np.sqrt(mu*df.p)
    
    # Compute x,y,z coordinates of angular momentum vector projected on the
    # equatorial plane.
    # In perifocal frame, h=[0,0,1]. Using transformation matrix from perifocal
    # to ECEF (see Curtis pg 174), we can find the components in ECEF
    # hx = sin(i)*sin(om), hy = -sin(i)*cos(om), hz = cos(i)
    # See also Alfriend, Lee & Creamer (2006) "Optimal Servicing of Geosynchronous Satellites" 
    df['hx'] = np.sin(np.deg2rad(df.i))*np.sin(np.deg2rad(df.om))
    df['hy'] = -np.sin(np.deg2rad(df.i))*np.cos(np.deg2rad(df.om))
    df['hz'] = np.cos(np.deg2rad(df.i))
    # Scale by angular momentum
    df['hx'] = df['h']*df['hx']
    df['hy'] = df['h']*df['hy']
    df['hz'] = df['h']*df['hz']
    
    # Angular momentum in spherical coordinates
    df['hphi'] = np.arctan2(np.sqrt(df.hx**2 + df.hy**2),df.hz) # Polar angle (from z axis)
    df['htheta'] = np.arctan2(df.hy,df.hx) # Azimuth angle
    
    
    # # Rearange dataframe
    # df = df[['name','norad','classification', 'int_desig',  # Designations
    #          'dn_o2', 'ddn_o6', 'bstar', 'set_num','rev_num', # Other
    #          'epoch_year','epoch_day','epoch', # Epochs
    #          'a','e','i','om','w','M', # Elements
    #          'n','p','q','Q','h','hx','hy','hz',
    #          ]]
    
    return df

def compute_principal_components(df,n_components=10):
    '''
    Perform Principal Component Analyis and return the PCs to the dataframe.

    Parameters
    ----------
    df : TYPE
        Input dataframe
    n_components : TYPE, optional
        Number of components to output. The default is 10.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    # Select features from dataset to include.
    # Exclude orbital parameters, since they are linear combinations of 
    # orbital elements
    features = ['a','e','i','om','w','n','h','hx','hy','hz']
    Xfull = df[features] # Extract feature data
    
    # Principal components
    # Run PCA on all numeric orbital parameters
    pca = PCA(n_composXfull)
    # Generate labels ['PC1','PC2', ...]
    labels = ['PC'+str(i+1) for i in range(n_components)]
    # Append data
    df[labels] = PC
    
    # Variance explained by each PC
    pca.explained_variance_ratio_
    # Feature importance
    # PC1,PC2,PC3: 4.27367824e-01, 2.86265363e-01, 2.60351753e-01, 
    # PC4,PC5,PC6: 2.21499988e-02, 3.86125358e-03, 2.54105655e-06, 
    # PC7,PC8,PC9: 1.26163922e-06, 4.55552048e-09, 6.39255498e-10, 
    # PC10: 5.17447310e-13
    
    # Most of variance explained by first 3 comonents
    # Drops off after that
    
    # Importance of each feature
    # Contribution of each input feature to each output PC
    # PC1: print(abs( pca.components_[0,:] ))
    # 
    # Main importance is contributed from
    # 'a','h','hx','hy','hz' 
    
    
    # See: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
    # For discussion on feature importances
    
    return df