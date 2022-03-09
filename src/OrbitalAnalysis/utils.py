# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:19:16 2022

@author: scott

Utilities Module

"""

from pathlib import Path
import os

#%% Dataset paths

def get_data_home():
    """
    Get the home data directory.

    By default the data dir is set to a folder named 'satellite_data'
    in the user's home folder (i.e. '~\satellite_tools\Data').

    Alternatively, this default location can be changed by setting an
    environment variable 'SATELLITE_DATA'.

    * Note!: The pathname must end in a folder name Data (i.e. '../Data' )

    If the folder does not already exist, it is automatically created.

    This data management system is inspired by the ASTROML python package.
    """

    # Look for Environment variable 'SR_TOOLS_DATA'
    envvar = os.environ.get('SATELLITE_DATA')

    if envvar is None:
        # Data path is not set in Environment variable.
        # Use a default data storage location at '~\sr_data\Data'
        data_home = Path.home()/'satellite_data'/'Data'
    else:
        # Use data location set in Environment variable
        data_home = Path(envvar)

    # Check if directory exists and create
    if not os.path.exists(str(data_home)):
        os.makedirs(str(data_home))

    return data_home