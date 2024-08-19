# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:47:02 2024

@author: scott

Validation
----------

"""

import pandas as pd
import numpy as np
import spiceypy as spice

import pdb

from pathlib import Path
import matplotlib.pyplot as plt

from OrbitalAnalysis.utils import get_data_home


def compare_access_times(python_dir,stk_dir,test_case):
    
    # Load spice kernels
    kernel_dir = get_data_home() / 'Kernels'
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    
    # Get scenario start time
    start_date = '2020-10-26 16:00:00.000'
    start_et = spice.str2et(start_date)
    
    
    # Optical Trackability
    # Load combined visible access
    
    # Load Python results
    dfpvis = pd.read_csv(str(python_dir/test_case/'Combined_Vis_Access_SSR.csv'))
    
    # Load STK results
    dfsvis = pd.read_csv(str(stk_dir/test_case/'Access_Times_Optical.csv'))
    dfsvis.dropna(inplace=True)
    # Offset start and stop times
    dfsvis.Start += start_et
    dfsvis.Stop += start_et
    pdb.set_trace()
    
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    ax.plot(dfpvis.Start,dfpvis.Duration,'.b',label='Python',markersize=5)
    ax.plot(dfsvis.Start,dfsvis.Duration,'.r',label='STK',markersize=5)
    # ax.plot((dfpvis.Start+dfpvis.Stop)/2,dfpvis.Duration,'.b',label='Python',markersize=5)
    # ax.plot((dfsvis.Start+dfsvis.Stop)/2,dfsvis.Duration,'.r',label='STK',markersize=5)
    ax.set_xlabel('Start of Access (ET)',fontsize=16)
    ax.set_ylabel('Duration (s)',fontsize=16)
    plt.title(test_case+': Optical Trackability (Visible Accesses)')
    plt.legend()
    plt.show()    
    
    # Radar results
    # Load Python results
    dfplos = pd.read_csv(str(python_dir/test_case/'Combined_LOS_Access_SSR.csv'))
    
    # Load STK results
    dfslos = pd.read_csv(str(stk_dir/test_case/'Access_Times_Radar.csv'))
    dfslos.dropna(inplace=True)
    # Offset start and stop times
    dfslos.Start += start_et
    dfslos.Stop += start_et
    
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    ax.plot(dfplos.Start,dfplos.Duration,'.b',label='Python',markersize=5)
    ax.plot(dfslos.Start,dfslos.Duration,'.r',label='STK',markersize=5)
    # ax.plot((dfplos.Start+dfplos.Stop)/2,dfplos.Duration,'.b',label='Python',markersize=5)
    # ax.plot((dfslos.Start+dfslos.Stop)/2,dfslos.Duration,'.r',label='STK',markersize=5)
    ax.set_xlabel('Start of Access (ET)',fontsize=16)
    ax.set_ylabel('Duration (s)',fontsize=16)
    plt.title(test_case+': Radar Trackability (Line-of-Sight Access)')
    plt.legend()
    plt.show()
    
    
    return



if __name__ == "__main__":
    
    
    
    # Define paths to python and stk results
    python_dir = Path(r'C:\Users\scott\Dropbox\Asteroid Mining (PhD)\Personal\MIT Space Enabled\DIT Analysis\Validation\Python Results')
    stk_dir = Path(r'C:\Users\scott\Dropbox\Asteroid Mining (PhD)\Personal\MIT Space Enabled\DIT Analysis\Validation\STK Results')
    
    # test_case = 'BEIDOU_G7' # Error
    
    # test_case = 'Giove-A'
    test_case = 'Giove-B' # Matches well
    # test_case = 'NigeriaSat-2' # Lots of access times
    # test_case = 'THEOS_Sat' # Lots of access times
    # test_case = 'VESTA-1' # Lots of access times
    compare_access_times(python_dir,stk_dir,test_case)
    
    
    
    