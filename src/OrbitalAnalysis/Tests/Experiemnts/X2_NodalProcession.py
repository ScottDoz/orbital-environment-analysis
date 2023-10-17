# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:00:35 2023

@author: mauri

Goal: Investigate the change in the theta component of angular momentum vectors
of the satellite catalog over a one year period. 

"""
# Module imports
from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.utils import get_data_home
import matplotlib.pyplot as plt
import numpy as np

#%% Helper functions

def compute_nodal_precession_rate(df):
    '''
    Compute the predicted precession of the nodes as a function of orbital parameters.
    
    OMdot = -(3/2)*J2*(Re/p)^2*n*cos(i)
    
    where
    p = a(1-e^2) is the semi-latus rectum
    n = sqrt(mu/a^3) is the mean motion
    i is the inclination
    Re = radius of the Earth
    J2 is the zonal harmonic coefficient
    

    Parameters
    ----------
    df : Input dataframe containing columns 'a' (km), 'e', 'i'(deg).

    Returns
    -------
    OMdot_p : Series containing predicted nodal precession rate.

    '''
    
    
    # Approach 1: From Wiki
    # https://en.wikipedia.org/wiki/Nodal_precession
    
    # Constants
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    
    
    # Precession rate OMdot_p
    # From: https://trs.jpl.nasa.gov/bitstream/handle/2014/37900/04-0263.pdf?sequence=1
    n = np.sqrt(mu/df.a**3) # Mean motion
    p = df.a*(1-df.e**2) # Semi-latus rectum
    OMdot_p = -(3./2.)*J2*((Re/p)**2)*n*np.cos(np.deg2rad(df.i))
    

    return OMdot_p

#%% Load data

# df = load_vishnu_experiment_data('all')
df = load_2019_experiment_data('all') # New dataset


Re = 6378.137; #Radius of Earth (km)
wE = 2*np.pi/(365.242199*86400) # Precession rate of the earth (rad/s)

# Sort by Epoch
df.sort_values(by=['NoradId','Epoch'],ascending=[True,True],inplace=True,ignore_index=True)

# # Unwrap multiple times to get the same range for all the objects
# df['htheta'] = df[['NoradId','htheta']].groupby(['NoradId']).transform(lambda x: np.unwrap(x))

# # Unwrap multiple times to get the same range for all the objects
# df['w'] = np.deg2rad(df.w) # convert to radians
# df['w'] = df[['NoradId','w']].groupby(['NoradId']).transform(lambda x: np.unwrap(x))
# df['w'] = np.rad2deg(df.w) # Back to deg

# # Unwrap multiple times to get the same range for all the objects
# df['om'] = np.deg2rad(df.om) # convert to radians
# df['om'] = df[['NoradId','om']].groupby(['NoradId']).transform(lambda x: np.unwrap(x))
# df['om'] = np.rad2deg(df.om) # Back to deg

# Alternative approach - single grouping
df['w'] = np.deg2rad(df.w) # convert to radians
df['om'] = np.deg2rad(df.om) # convert to radians
print('Grouping Data',flush=True)
dfg = df.groupby(['NoradId']) # Grouped dataframe
print('Unwrapping htheta',flush=True)
df['htheta'] = dfg['htheta'].transform(lambda x: np.unwrap(x))
print('Unwrapping w',flush=True)
df['w'] = dfg['w'].transform(lambda x: np.unwrap(x))
print('Unwrapping om',flush=True)
df['om'] = dfg['om'].transform(lambda x: np.unwrap(x))
df['w'] = np.rad2deg(df.w) # Back to deg
df['om'] = np.rad2deg(df.om) # Back to deg

# # Converting epoch to datetime format
# EpochDT = pd.to_datetime(df.Epoch)
# df.insert(8,("EpochDT"),EpochDT)


#%% Theoretical nodal precession

# The ascending nodes of orbits precess due to uneven gravity forces (J2 effect).
# This causes the RAAN (or om) to drift over time.
# There are analytical expressions for the rate of change in RAAN (OM dot or OMdot_p).
# (the _p suffix indicates "predicted")

# Note:
# The dependence on cos(inc) means that
# OMdot_p < 0 for i < 90 deg
# OMdot_p > 0 for i > 90 deg
# May be problematic close to 90 deg


# Compute expression at each epoch
OMdot_p = compute_nodal_precession_rate(df)
# df.insert(45,("OMdot_p"),OMdot_p)
df['OMdot_p'] = OMdot_p

#%%  Observed rate of change in angular momentum theta component

# Compute the observed rate of change in the htheta component between each epoch.
# hthetadot = d(htheta)/dt
# df['dhtheta'] = df['htheta'].diff() # htheta increment (rad)
# df['dt'] = df['EpochDT'].diff().dt.total_seconds() # Time step (s)
# df['hthetadot'] = df['htheta'].diff()/df['EpochDT'].diff().dt.total_seconds()

df['dhtheta'] = df.groupby(by='NoradId')['htheta'].diff() # htheta increment (rad)
df['dt'] = df.groupby(by='NoradId')['EpochDT'].diff().dt.total_seconds() # Time step (s)
df['hthetadot'] = df['dhtheta']/df['dt'] # Rate of change in htheta (rad/s)


#%% Extreme outliers (new dataset)

# Problem: very small timesteps dt< 1 second. Leads to extremely large htheta values.
# E.g. Norad = 89251 has htheta mean = 0.04816
#                           EpochDT     hthetadot            dt   dhtheta
# 660899 2019-02-16 15:35:52.422719 -2.399856e-07  1.048179e+07 -2.515480
# 660900 2019-02-19 14:55:30.127871 -2.424845e-07  2.567777e+05 -0.062265
# 660901 2019-03-07 12:12:36.184032 -2.405129e-07  1.372626e+06 -0.330134
# 660902 2019-03-14 11:30:55.545120 -2.408487e-07  6.022994e+05 -0.145063
# 660903 2019-03-14 11:30:55.547711  2.889796e-01  2.591000e-03  0.000749
# 660905 2019-03-14 11:30:57.080448  0.000000e+00  1.532737e+00  0.000000

# Solution: remove timesteps of < 1 hour (3600 seconds)

#%% Compute statistics
# Compute the average values of htheta, OMdot_p, hthetadot

# Remove points with zero or undefined timestep
df = df[df.dt > 0.] 

# Remove small timesteps (dt < say 3600 seconds)
df = df[df.dt > 3600.]

# Compute statitics
dfstats = df.groupby(by='NoradId').agg({'Name':['first'],
                                        'h':['mean','std','min','max','median','mad','count'],
                                        'hz':['mean','std','min','max','median','mad','count'],
                                        'htheta':['mean','std','min','max','median','mad','count'],
                                        'dhtheta':['mean','std','min','max','median','mad','count'],
                                        'hphi':['mean','std','min','max','median','mad','count'],
                                        'OMdot_p':['mean','std','min','max','median','mad','count'],
                                        'hthetadot':['mean','std','min','max','median','mad','count'],
                                        'a':['mean','std','min','max','median','mad','count'],
                                        'e':['mean','std','min','max','median','mad','count'],
                                        'i':['mean','std','min','max','median','mad','count'],
                                        'q':['mean','std','min','max','median','mad','count'],
                                        'p':['mean','std','min','max','median','mad','count'],
                                        })


#%% Outlier removal

# We want to remove any objects that have significant changes in their orbital elements
# These objects are undergoing active manouvering such as orbit raising and deorbiting.

# First, drop items with only one or two measurements (~578 objects)
dfstats = dfstats[dfstats['h']['count']>2]

# Next, remove any hyperbolic objects (those that have max(e) >= 1.)
# 4 objects:
# Norad  Name
# 29260 COSMOS 2422
# 35653 COSMOS 2251 DEB
# 40747 DELTA 4 R/B
# 44338 CZ-3B R/B
ind = dfstats['e']['max']>=1.
dfstats = dfstats[~ind]


# Next, we find all objects that have de-orbited over the period of interest.
# These are objects whose minimum perigee is less than or equal to the earth radius.
ind = dfstats['q']['min']<=Re
# 44 objects
dfstats = dfstats[~ind]

# Now, remove any objects transitioning from GTO to GEO.
# These objects have maximum ahove GEO (a = 42,167 km), but a minimum periapsis
# much lower. 
# E.g. NoradId = 44053
ind = (dfstats['a']['max'] > 40000) & (dfstats['q']['min'] < Re+5000)
# dfstats[[('a','min'),('a','max'),('q','min'),('q','max')]][ind]
dfstats = dfstats[~ind]

# # Plot histogram of std of h
# fig, ax = plt.subplots(1,1,figsize=(8, 4)) 
# plt.hist(dfstats_copy['h']['std']/dfstats_copy['h']['mean'],bins=100,label='orig')
# plt.hist(dfstats['h']['std']/dfstats['h']['mean'],bins=100,label='Filtered') 
# # plt.xscale('log')
# plt.yscale('log')
# ax.set_xlabel('std(h)/mean(h)')
# ax.set_ylabel('Frequency')


# fig, ax = plt.subplots(1,1,figsize=(8, 8)) 
# plt.errorbar(dfstats['h']['mean'], dfstats['htheta']['mean'], xerr=dfstats['h']['std'],yerr=dfstats['htheta']['std'], fmt="o")
# ax.set_xlabel('h')
# ax.set_ylabel('htheta')

#%% Trends in htheta motion

# Find satellites whose htheta values are monotonically increasing/decreasing
mon_inc_flag = df.groupby(by='NoradId')['htheta'].apply(lambda x: x.is_monotonic_increasing)
mon_dec_flag = df.groupby(by='NoradId')['htheta'].apply(lambda x: x.is_monotonic_decreasing)
dfstats[('htheta','Trend')] = '' # Column labelling trend
dfstats[('htheta','Trend')][mon_inc_flag] = 'Increasing'
dfstats[('htheta','Trend')][mon_dec_flag] = 'Decreasing'

# Indices of 
# Prograde: i<90
# Retrograde: i>90
# Oscillating: htheta is not in one direction

indPro = (dfstats[('i','mean')]<90) & (dfstats[('htheta','Trend')] != '')
indRetro = (dfstats[('i','mean')]>90) & (dfstats[('htheta','Trend')] != '')
indOsc = ~(indPro+indRetro)

norad_Pro = list(dfstats[indPro].index)
norad_Retro = list(dfstats[indRetro].index)
norad_Osc = list(dfstats[indOsc].index)


# Randomly sample 1000 objects of each type
import random
from random import sample
random.seed(10)
random_norad_Pro = sample(norad_Pro,200)
random_norad_Retro = sample(norad_Retro,200)
random_norad_Osc = sample(norad_Osc,200)

# Generate a plot of objects demonstrating the differnet types of motion

fig, ax = plt.subplots(3,1,figsize=(10, 8))
# Plot Prograde
for num in random_norad_Pro:
    ax[0].plot(df['EpochDT'][df.NoradId == num],df['htheta'][df.NoradId == num],'-r',label='Prograde')
for num in random_norad_Retro:
    ax[1].plot(df['EpochDT'][df.NoradId == num],df['htheta'][df.NoradId == num],'-b',label='Retrograde')
for num in random_norad_Osc:
    ax[2].plot(df['EpochDT'][df.NoradId == num],df['htheta'][df.NoradId == num],'-k',label='Oscillating')
# Axes [0] settings
# ax[0].set_xlabel(r'Epoch',fontsize=16)
ax[0].set_ylabel(r'$h_{\theta}$ (rad)',fontsize=16)
handles, labels = ax[0].get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax[0].legend(handles, labels, loc='best')
# Axes [1] settings
# ax[1].set_xlabel(r'Epoch',fontsize=16)
ax[1].set_ylabel(r'$h_{\theta}$ (rad)',fontsize=16)
handles, labels = ax[1].get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax[1].legend(handles, labels, loc='best')
# Axes [2] settings
ax[2].set_xlabel(r'Epoch',fontsize=16)
ax[2].set_ylabel(r'$h_{\theta}$ (rad)',fontsize=16)
handles, labels = ax[2].get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax[2].legend(handles, labels, loc='best')

#%% Plots

with_error_bars = False

# Plot predicted vs observed rates of change in htheta
fig, ax = plt.subplots(1,1,figsize=(8, 5))                             
# plt.plot(dfstats['OMdot_p']['mean'],dfstats['OMdot_p']['mean'],'.k')
plt.plot([-1.7e-6,1.3e-6],[-1.7e-6,1.3e-6],'-r')
if with_error_bars:
    # Plot with error bars
    plt.errorbar(dfstats['OMdot_p']['mean'][indPro], dfstats['hthetadot']['mean'][indPro], yerr=dfstats['OMdot_p']['std'][indPro], fmt=".r", label='Prograde') # Increasing
    plt.errorbar(dfstats['OMdot_p']['mean'][indRetro], dfstats['hthetadot']['mean'][indRetro], yerr=dfstats['OMdot_p']['std'][indRetro], fmt=".b", label='Retrograde') # Increasing
    plt.errorbar(dfstats['OMdot_p']['mean'][indOsc], dfstats['hthetadot']['mean'][indOsc], yerr=dfstats['OMdot_p']['std'][indOsc], fmt=".k", label='Oscillating') # Increasing
else:
    # Plot withput error bars
    plt.plot(dfstats['OMdot_p']['mean'][indPro], dfstats['hthetadot']['mean'][indPro], ".r", label='Prograde') # Increasing
    plt.plot(dfstats['OMdot_p']['mean'][indRetro], dfstats['hthetadot']['mean'][indRetro], ".b", label='Retrograde') # Increasing
    plt.plot(dfstats['OMdot_p']['mean'][indOsc], dfstats['hthetadot']['mean'][indOsc], ".k", label='Oscillating') # Increasing
   
ax.set_xlabel(r'$\dot \Omega$ (Predicted) (rad/s)',fontsize=16)
ax.set_ylabel(r'$ \dot h_{\theta} $ (Measured) (rad/s)',fontsize=16)
ax.set_xlim(-1.7e-6,1.3e-6)
ax.set_ylim(-1.7e-6,1.3e-6)
plt.legend(loc=2)
ax.set_aspect('equal', 'box')
# plt.xscale('log')
# plt.yscale('log')

# # Plot predicted vs inclination
# fig, ax = plt.subplots(1,1,figsize=(8, 4))                             
# # plt.plot(dfstats['OMdot_p']['mean'],dfstats['OMdot_p']['mean'],'.k')
# # plt.plot([-1.5e-6,1.2e-6],[-1.5e-6,1.2e-6],'-r')
# # plt.errorbar(dfstats['i']['mean'], dfstats['hthetadot']['mean'], yerr=dfstats['OMdot_p']['std'], fmt=".k") # All
# plt.errorbar(dfstats['i']['mean'][indPro], dfstats['OMdot_p']['mean'][indPro], yerr=dfstats['OMdot_p']['std'][indPro], fmt=".r", label='Prograde') # Increasing
# plt.errorbar(dfstats['i']['mean'][indRetro], dfstats['OMdot_p']['mean'][indRetro], yerr=dfstats['OMdot_p']['std'][indRetro], fmt=".b", label='Retrograde') # Increasing
# plt.errorbar(dfstats['i']['mean'][indOsc], dfstats['OMdot_p']['mean'][indOsc], yerr=dfstats['OMdot_p']['std'][indOsc], fmt=".k", label='Oscillating') # Increasing
# ax.set_xlabel('Incliantion (deg)')
# ax.set_ylabel('OMdot_p (Predicted)')

# # Plot predicted vs semi-latus rectum
# fig, ax = plt.subplots(1,1,figsize=(8, 4))                             
# # plt.plot(dfstats['OMdot_p']['mean'],dfstats['OMdot_p']['mean'],'.k')
# # plt.plot([-1.5e-6,1.2e-6],[-1.5e-6,1.2e-6],'-r')
# # plt.errorbar(dfstats['i']['mean'], dfstats['hthetadot']['mean'], yerr=dfstats['OMdot_p']['std'], fmt=".k") # All
# plt.errorbar(dfstats['p']['mean'][indPro], dfstats['OMdot_p']['mean'][indPro], yerr=dfstats['OMdot_p']['std'][indPro], fmt=".r", label='Prograde') # Increasing
# plt.errorbar(dfstats['p']['mean'][indRetro], dfstats['OMdot_p']['mean'][indRetro], yerr=dfstats['OMdot_p']['std'][indRetro], fmt=".b", label='Retrograde') # Increasing
# plt.errorbar(dfstats['p']['mean'][indOsc], dfstats['OMdot_p']['mean'][indOsc], yerr=dfstats['OMdot_p']['std'][indOsc], fmt=".k", label='Oscillating') # Increasing
# ax.set_xlabel('p')
# ax.set_ylabel('OMdot_p (Predicted)')

#%% 3D plots

# # # Extract the first epoch
df1 = df.groupby('NoradId').last()

from OrbitalAnalysis.Visualization import *
midval = abs(df1.OMdot_p.min()/(df1.OMdot_p.max()-df1.OMdot_p.min()))
colorscale = [[0, 'rgba(214, 39, 40, 0.85)'],   
              [midval, 'rgba(255, 255, 255, 0.85)'],  
              [1, 'rgba(6,54,21, 0.85)']],



colorscale = [[0, 'red'], [0.57, 'grey'], [1.0, 'blue']]

fig = plot_3d_scatter_numeric(df1,'hx','hy','hz',color='OMdot_p',
                            colorscale=colorscale,
                            color_label=u"\u03A9 (rad/s)",
                            # color_label = r'$\dot \Omega$',
                            # color_label = r'$\Delta t\textrm{(s)}$',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=True,
                            logColor=False,
                            )

# Note
# For prograde motion (OMdot_p < 0), we find hz > 0 for all
# Retrograde (OMdot_p > 0), we find hz < 0 for all

# So, all of the objects above the plane hz=0 rotate counter-clockwise
# all objects below the plane hz=0 rotate clock-wise



#%% Problem indices (Vishnu dataset)

# Update:
# The unwrapping issue is resolved using the newer dataset.

# New: Identify the extreme outliers. abs(hthetadot) > 0.001 rad/s 
ind_prob = abs(dfstats[('hthetadot','mean')])>0.001
norad_prob = list(dfstats[ind_prob].index)
# 19 objects


# # Vishnu Problem indices to investigate - top left of graph
# ind_prob = (dfstats[('OMdot_p','mean')]< -1.1e-6) & (dfstats[('hthetadot','mean')]>0.75e-6)
# norad_prob = list(dfstats[ind_prob].index)

# Plot histogram of ratio
# plt.hist(abs(dfstats[('hthetadot','mean')][ind_prob]/dfstats[('OMdot_p','mean')][ind_prob]),bins=100)

# Observations:
# For this set, the observed rate of change in htheta is around 70% of what we predict.
# Also: predict OMdot_p is -ve (prograde motion), but observe hthetadot is +ve

# CHECK UNWRAP
# E.g. objects to look at: [963, 1835, 1883, 1961, 3386, 3462, 10230, 20303, 20362, 20580]



fig, ax = plt.subplots(1,1,figsize=(8, 8))  
for num in [963, 1835, 1883, 1961, 3386, 3462, 10230, 20303, 20362, 20580]:
    plt.plot(df['EpochDT'][df.NoradId == num],df['htheta'][df.NoradId == num],'-k')
# for num in norad_Osc:
#     plt.plot(df['EpochDT'][df.NoradId == num],df['htheta'][df.NoradId == num],'-r')



# Circular orbits 6705 to 7250 km
# Inclinations 1.95 to 42.77 deg


# # Plot Ratio as fn of inclination
# fig, ax = plt.subplots(1,1,figsize=(8, 8))
# plt.title('Monitonically Increasing/Decreasing', fontweight ="bold")                            
# # ax.plot(dfstats['i']['mean'],dfstats['hthetadot']['mean']/dfstats['OMdot_p']['mean'],'.k') # All
# ax.plot(dfstats['i']['mean'][dfstats[('htheta','Trend')] == 'Increasing'],dfstats['hthetadot']['mean'][dfstats[('htheta','Trend')] == 'Increasing']/dfstats['OMdot_p']['mean'][dfstats[('htheta','Trend')] == 'Increasing'],'.r') # Mon Increasing
# ax.plot(dfstats['i']['mean'][dfstats[('htheta','Trend')] == 'Decreasing'],dfstats['hthetadot']['mean'][dfstats[('htheta','Trend')] == 'Decreasing']/dfstats['OMdot_p']['mean'][dfstats[('htheta','Trend')] == 'Decreasing'],'.b') # Mon Decreasing
# ax.plot(dfstats['i']['mean'][dfstats[('htheta','Trend')] == ''],dfstats['hthetadot']['mean'][dfstats[('htheta','Trend')] == '']/dfstats['OMdot_p']['mean'][dfstats[('htheta','Trend')] == ''],'.k') # Oscillating


# ax.plot(dfstats['i']['mean'][mon_inc_flag],dfstats['hthetadot']['mean'][mon_inc_flag]/dfstats['hthetadot']['mean'][mon_inc_flag],'.r',label='Inc')
# ax.plot(dfstats['i']['mean'][mon_dec_flag],dfstats['hthetadot']['mean'][mon_dec_flag]/dfstats['hthetadot']['mean'][mon_dec_flag],'.b',label='Dec')
# ax.set_xlabel('i')
# ax.set_ylabel('hthetadot/OMdot_p')
# plt.legend(loc=2, borderaxespad=0.)
# dfstats[dfstats[('htheta','Trend')] == 'Increasing']


# fig, ax = plt.subplots(1,1,figsize=(8, 8))                             
# # ax.plot(dfstats['OMdot_p']['mean'],dfstats['hthetadot']['mean'],'.k') # All points
# ax.plot(dfstats['OMdot_p']['mean'],dfstats['hthetadot']['mean']/dfstats['OMdot_p']['mean'],'.k')
# # ax.plot([dfstats['OMdot_p']['mean'].min(), dfstats['hthetadot']['mean'].max()],[1,1],'-r')
# ax.set_xlabel('OMdot_p')
# ax.set_ylabel('hthetadot/OMdot_p')




# # [mon_inc_flag | mon_dec_flag]

# fig, ax = plt.subplots(1,1,figsize=(8, 8))   
# plt.title('Problematic', fontweight ="bold")                              
# # ax.plot(dfstats['hthetadot']['mean'],dfstats['OMdot_p']['mean'],'.k')
# ax.plot(dfstats['i']['mean'][~mon_inc_flag & ~mon_dec_flag],dfstats['OMdot_p']['mean'][~mon_inc_flag & ~mon_dec_flag]/dfstats['hthetadot']['mean'][~mon_inc_flag & ~mon_dec_flag],'.r',label='Inc')
# ax.set_xlabel('i')
# ax.set_ylabel('OMdot_p/hthetadot')


#%% Investigate outliers

# Indices of objects where OMdot_p/hthetadot >> 1
dfout = dfstats[dfstats['OMdot_p']['mean']/dfstats['hthetadot']['mean'] > 1e6]

# Extreme outlier
# Norad Id = 43583
# This object has a significant change in semi-major axis.
# Drops from 
dfstats['h'][dfstats['OMdot_p']['mean'] < -0.007]





# List of extreme outliers
# Norad Id = 25272 IRIDIUM 55
#           a         e          i         om           w             h
# 7147.412447  0.001301  86.488194  38.757793   70.425436  53375.591627
#  423.975905  0.998820  86.391370  15.335830 -233.397748    631.300587

# Norad Id = 27375
#           a         e          i         om          w             h
# 7161.980707  0.001307  86.462738  38.925515  67.329515  53429.960096
#  659.141023  0.991963  86.373214  18.435118  -7.186256   2050.952442

# Norad Id = 43583
#                    a         e          i          om           w             h
# 195963  14859.039266  0.561044  54.947336  172.501280 -128.353542  63706.246015
# 195964  14469.997984  0.550072  54.987818  154.803818 -118.343022  63423.503068
# 195965  13757.249517  0.527541  55.006177  134.329108 -106.729181  62909.040143
# 195966  12559.688607  0.482977  55.030732  107.849709  -91.697859  61955.544814
# 195967  11337.041594  0.427382  55.041169   76.288630  -73.854776  60774.452561
# 195968   9953.916994  0.348088  54.991934   37.264731  -51.800235  59049.922378
# 195969    369.255618  0.640784  59.886396   85.749085   92.327898   9313.990807


# Norad Id = 2222.
# Issue with wraping w over 360.
#                  a         e         i         om           w              h
# 5103  40449.446907  0.015443  3.361581  62.765336 -261.656420  126961.821619
# 5104  40449.663367  0.015459  3.396086  62.997963 -261.227326  126962.130362
# 5105  40449.621372  0.015412  3.433363  62.946154 -260.544079  126962.157236
# 5106  40449.721976  0.015370  3.482211  62.655984 -259.690353  126962.397555
# 5107  40449.736797  0.015406  3.548868  62.432137  100.986926  126962.350030
# 5108  40449.724936  0.015407  3.620605  62.517631  100.983511  126962.330246
# 5109  40449.727086  0.015528  3.679172  62.875115  101.210732  126962.095898
# 5110  40449.726633  0.015603  3.713796  63.099942  102.012085  126961.945472
# 5111  40449.618939  0.015561  3.747641  62.927531  103.224192  126961.860020
# 5112  40449.626992  0.015463  3.804682  62.554659  104.447324  126962.066047
# 5113  40449.468797  0.015380  3.876793  62.390309  105.103173  126961.981186



# Norad Id = 2649
# Issue with wraping w over 360
#                  a         e         i         om           w              h
# 5795  40059.807984  0.003370  3.585236  58.481246   45.908751  126363.199297
# 5796  40060.089832  0.003390  3.614866  58.667239   44.902759  126363.635317
# 5797  40060.053777  0.003481  3.650212  58.607960   45.454736  126363.539014
# 5798  40059.911203  0.003484  3.696787  58.340747 -313.585969  126363.312747
# 5799  40059.807893  0.003578  3.759680  58.160571   47.969380  126363.107643
# 5800  40059.744091  0.003625  3.826615  58.255891   49.483087  126362.985520
# 5801  40059.770504  0.003667  3.880635  58.589205   51.007447  126363.007895
# 5802  40059.750190  0.003587  3.910429  58.777404   51.169689  126363.012490
# 5803  40059.917580  0.003538  3.941990  58.600053   53.438709  126363.298871
# 5804  40059.971214  0.003480  3.996882  58.263035   52.721023  126363.408917
# 5805  40060.121697  0.003425  4.065326  58.136526   53.999637  126363.670557

# Observation:
# Two main sources of errors
# 1. w wrapping over 360 deg. Can fix.
# 2. Significant change in orbit over time. Cannot fix.

# Interesting cases
# Norad 2222 and 2649
# Orbit is circular and almost equatorial, about 2000 km below GEO.
# Plotting htheta, or om it seems to oscillate.
# The node is oscillating about a point.
# This means that dhtheta is oscillating between +ve and -ve.
# Not measuring dhtheta correctly?


# # Drop this outlier
# # dfstats = dfstats.drop(dfstats[dfstats.index == 25272].index)
# dfstats = dfstats.drop(dfstats[dfstats['OMdot_p']['mean']/dfstats['hthetadot']['mean'] > 60000].index)





#%% Plot htheta vs time

# Mon increasing




#%% Testing

# # Extract a sequence of numbers that wraps over 2pi
# # E.g. NoradID = 5
# # df2[['EpochDT','htheta']][df2.NoradId==5]
# x = df2['EpochDT'][df2.NoradId==5].to_numpy()
# y = df2['htheta'][df2.NoradId==5].to_numpy()
# plt.plot(x,y,'.k')
# plt.plot(x,np.unwrap(y),'.k')

# Plotting htheta over time.
# We see that there is a linear trend. Observed for all satellites (??? check this)
# TODO: Check if each satellite has a linear trend.
# Observations: the slope of this line is approximately equal to the rate of nodal precession.
# We find this is true for the vast majority of the catalog. There are some satellites
# where the numbers don't match up (within 1 order of magnitude).
# TODO (future work): see if we can find a reason for this



# plt.hist(df2['hthetadot'][(pd.notnull(df2.hthetadot)) & (pd.notnull(df2.OMdot_p))]/df2['OMdot_p'][(pd.notnull(df2.hthetadot)) & (pd.notnull(df2.OMdot_p))])
# [pd.notnull(df2.OMdot_p)]

