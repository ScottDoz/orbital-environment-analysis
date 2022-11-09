# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:58:18 2022

@author: scott

Experiment: Test the propagation of multiple objects using SGP4 propagator


"""



from sgp4.api import Satrec
from sgp4.api import SatrecArray
import numpy as np
import pandas as pd
import time

from SatelliteData import *
from utils import get_data_home


#%% Propagate single object

# Extract TLEs of single object
NORAD = 25544 # ISS
tle_lines = get_tle(NORAD,epoch='latest',tle_type='3le')

# Create satellite object
s = tle_lines[1] # 1st line
t = tle_lines[2] # 2nd line
satellite = Satrec.twoline2rv(s, t)

# Create a list of epochs
t0 = satellite.jdsatepoch # Epoch of TLEs (JD)
duration = 10 # Duration (days)
N = 1000000       # Number of time steps
t = np.linspace(t0,t0+duration,N)
# Split into whole number and fration
jd, fr = divmod(t, 1)

# Propagate
t_start = time.time()
e, r, v = satellite.sgp4_array(jd, fr)
print("Single object propagation: {} timesteps.".format(N))
print("Runtime: {} s".format(time.time() - t_start))
del e,r,v # Delete outputs to free up space

#%% Propagate multiple objects

# Load data
df = load_satellites(group='all',compute_params=True)

# Create array of satellites
satlist = [ Satrec.twoline2rv(row['line1'], row['line2']) for i,row in df.iterrows() ]
satarray = SatrecArray(satlist)

# Create a list of epochs
t0 = satlist[0].jdsatepoch # Epoch of 1st satellite
duration = 10 # Duration (days)
N = 1000       # Number of time steps
t = np.linspace(t0,t0+duration,N)
# Split into whole number and fration
jd, fr = divmod(t, 1)

# Propagate
t_start = time.time()
e, r, v = satarray.sgp4(jd, fr)
print("Catalog propagation: {} objects at {} timesteps.".format(len(df),N))
print("Runtime: {} s".format(time.time() - t_start))
del e,r,v # Delete outputs to free up space


# # Repeat for 3,000 epochs
# N = 3000       # Number of time steps
# t = np.linspace(t0,t0+duration,N)
# # Split into whole number and fration
# jd, fr = divmod(t, 1)

# # Propagate
# t_start = time.time()
# e, r, v = satarray.sgp4(jd, fr)
# print("Catalog propagation: {} objects at {} timesteps.".format(len(df),N))
# print("Runtime: {} s".format(time.time() - t_start))


#%% Results

# Approximate runtimes

# Single object at 1,000,000 timesteps: 0.296 s

# Catalog: 24,060 objects at
# 1000 timesteps: 8.38 s
# 3000 timesteps: 27.08 s

# Note on RAM contraints:
# (uncomment 2nd block of code to test)
# Test on 24060 objects
# 4,000 timesteps: Unable to allocate 2.15 GiB for an array with shape (24060, 4000, 3) and data type float64
# 5,000 timesteps: Unable to allocate 2.69 GiB for an array with shape (24060, 5000, 3) and data type float64
# 10,000 timesteps: Unable to allocate 5.38 GiB for an array with shape (24060, 10000, 3) and data type float64

# Limited to around 3000 timesteps for this size catalog


