# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:52:38 2022

@author: scott

Experiment. Compute MOID for satellites.



"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SatelliteData import load_satellites
from astrologistics.MOID import moid


#%% Load data

df = load_satellites(group='all',compute_params=True)

# Extract orbital elements a1,e1,w1,om1,inc1 (angles in deg)
X = df[['a','e','w','om','i']].to_numpy()


# Extract query object
target = 25544 # ISS
x = df[['a','e','w','om','i']][df.NoradId == target].iloc[0].to_numpy()
a1,e1,w1,om1,inc1 = x


# Compute MOID
dist = np.zeros(len(df))
for i in range(len(df)):
    a2,e2,w2,om2,inc2 = X[i,:] # Extract elements
    dist[i] = moid(a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2) # Compute distance

# Add to dataframe
df['moid'] = dist

# Plot
plt.hist(df['moid'][df.moid < 1000],bins=100)
