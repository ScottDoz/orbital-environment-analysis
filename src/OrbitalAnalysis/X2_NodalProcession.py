# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:00:35 2023

@author: mauri
"""
# Module imports
from SatelliteData import *
from utils import get_data_home
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = load_vishnu_experiment_data('all')
# TODO: Apply unwrapped to original dataframe (dtheta and dphi)
df_copy = df.copy()

dfstats = df.groupby(by='NoradId').agg({'h':['mean','std','min','max','median','mad','count'],
                                        'hz':['mean','std','min','max','median','mad','count'],
                                        'htheta':['mean','std','min','max','median','mad','count'],
                                        'hphi':['mean','std','min','max','median','mad','count']})

# Theoretical nodal precession
# Obtaining the first observations for each satellite
df1 = df.copy()
df1.sort_values(by=['NoradId','Epoch'],ascending=[True,True],inplace=True,ignore_index=True)
df1.drop_duplicates(subset=['NoradId'],keep='first',inplace=True,ignore_index=True)
# Calculate nodal precession
# See: https://en.wikipedia.org/wiki/Nodal_precession
Re = 6378.137; #Radius of Earth (km)
J2 = 1.08262668e-3; #J2 constant of Earth
mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
P = 2*np.pi*np.sqrt((df1['a']**3)/mu); # Period (s)
w = (np.pi*2)/P; # Angular velocity (rad/s)
wp = -(3/2)*((Re**2)/(df1.a*(1-df1.e**2))**2)*J2*w*np.cos(np.deg2rad(df1.i)) # Theoretical nodal precession (rad/s)

# Measured nodal precession from change in angular momentum
# Angles in radians, time in seconds
# Converting epoch to datetime format
df2 = df.copy()
EpochDT = pd.to_datetime(df2.Epoch)
df2.insert(8,("EpochDT"),EpochDT)
# Unwrapped
# TODO: Figure out how to unwrap multiple times to get the same range for all the objects
df2['htheta'] = df2[['NoradId','htheta']].groupby(['NoradId']).transform(lambda x: np.unwrap(x))
dfg = df2.groupby(by='NoradId').agg({'EpochDT':['max','min'],
                                     'htheta':['first','last']})
dfg[('htheta','dhtheta')] = dfg['htheta']['last']-dfg['htheta']['first']
dfg[('EpochDT','dEpochDT')] = abs(dfg['EpochDT']['max']-dfg['EpochDT']['min']).dt.total_seconds()
wh = dfg[('htheta','dhtheta')]/dfg[('EpochDT','dEpochDT')]
fig, ax = plt.subplots(1,1,figsize=(8, 8))                             
ax.plot(wh,wp,'.k')
ax.set_xlabel('wh')
ax.set_ylabel('wp')