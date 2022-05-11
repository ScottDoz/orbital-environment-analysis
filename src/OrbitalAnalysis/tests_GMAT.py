# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:00:01 2022

@author: scott

Tests for GMAT functions 

"""

from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *
from Overpass import *
from Ephem import *
from Events import *
from GmatScenario import *



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