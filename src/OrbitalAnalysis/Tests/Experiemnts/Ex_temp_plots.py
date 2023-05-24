# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:40:55 2022

@author: scott
"""

import matplotlib.pyplot as plt
import numpy as np

from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *

# Load data and sort
df = load_vishnu_experiment_data('all')
df.sort_values(by=['NoradId','Epoch'],ascending=[True,True],inplace=True)

# Group by NoradId
dfstats = df.groupby('NoradId').agg({'h': ['mean', 'min', 'max','count','first','last'],
                                     'htheta': ['mean', 'min', 'max','count','first','last'],
                                     'hphi': ['mean', 'min', 'max','count','first','last'],
                                     })

# Find objects that wrap around 2pi
obj_list = dfstats[ dfstats[('htheta','last')] < dfstats[('htheta','first')] ].index.to_list()


obj_list = dfstats[ ( dfstats[('h','max')] > dfstats[('h','first')] ) & ( dfstats[('h','max')] > dfstats[('h','last')] )  ].index.to_list()


obj_list = dfstats[ (dfstats[('h','max')] > dfstats[('h','first') ) & ( dfstats[('h','max')] > dfstats[('h','last')] ) ].index.to_list()



# Plot an example
fig, ax = plt.subplots()
ax.plot(df['Epoch'][df.NoradId == 11], df['htheta'][df.NoradId == 11],'-k',label='original')
ax.plot(df['Epoch'][df.NoradId == 11], np.unwrap(df['htheta'][df.NoradId == 11]),'-r',label='unwrapped')
ax.legend()

np.unwrap(phase_deg, period=360)


def myfunc(data):
    data['htheta'] = np.unwrap(data['htheta'])
    return data


# Apply unwrap to all
df1 = df.copy()
df1['htheta'] = df1[['NoradId','htheta']].groupby(['NoradId']).transform(lambda x: np.unwrap(x))




grouped = df1.groupby(lambda x: x.NoradId)


fig, ax = plt.subplots()
ax.plot(df['Epoch'][df.NoradId == 5], df['h'][df.NoradId == 5],'-k',label='original')
ax.legend()

