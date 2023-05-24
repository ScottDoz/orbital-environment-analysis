# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:35:43 2022

@author: mauri

This is the original script. We have splitted up the script into two different sections.
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
#%% Differentials #############################################################
# Calculate differentials
# TODO: Check for wrapping around 2pi. Try signed values instead of absolute values
# Compute path-dependent distances rather than absolute distances. (Path differentials & path integrals)
dphi = abs(dfstats['hphi']['max']-dfstats['hphi']['min'])
dr = abs(dfstats['h']['max']-dfstats['h']['min'])
dtheta = abs(dfstats['htheta']['max']-dfstats['htheta']['min'])
dz = abs(dfstats['hz']['max']-dfstats['hz']['min'])
ds_sq_spherical = pow(dr,2)+pow((dfstats['h']['mean']),2)*pow(dtheta,2)+pow((dfstats['h']['mean']),2)*pow(np.sin(dfstats['htheta']['mean']),2)*pow(dphi,2)     
ds_sq_cylindrical = pow(dr,2)+pow((dfstats['h']['mean']),2)*pow(dtheta,2)+pow(dz,2)
# Insert these columns into dfstats
df = pd.merge(df,dfstats,how='left',on='NoradId') # Merge stats into original dataframe

#%% Outlier Removal ###########################################################
# TODO: Adjust analysis to look at wh rather than dhtheta (dh/dt)
# TODO: Same workflow for dphi (proving that change in dphi is minimal)

# Outlier removal
# NOTES: Remove any objects with changes in their orbit (GTO to GEO). 

# Looking at the plots h std vs. h mean, we want to remove outliers with large standard deviations.
# We want to apply it to h_std

# A more robust method is the modified Z-score method, using the median
# and the median absolute deviation (MAD).
# Modified z-score = 0.6745*(xi - x_med)/MAD
# Then remove any values with MZS > +/= 3.5
# See Iglewicz & Hoaglin

# Calculate the median and the mad of the h_std.
median_h_std = dfstats[('h','std')].median()
mad_h_std = dfstats[('h','std')].mad()
Z = 0.6745*(dfstats[('h', 'std')]-median_h_std)/mad_h_std
dfstats.insert(8,("h_std_MZS"),Z)
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.h_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['h']['mean']
y = dfstats['h']['std']/dfstats['h']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
ax.set_xlim([0,2e5])
ax.set_xlabel('h mean')
ax.set_ylabel('h std')
plt.show()

# Repeat second iteration
# Calculate the median and the mad of the h_std.
median_h_std = dfstats[('h','std')].median()
mad_h_std = dfstats[('h','std')].mad()
Z = 0.6745*(dfstats[('h', 'std')]-median_h_std)/mad_h_std
dfstats['h_std_MZS'] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.h_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['h']['mean']
y = dfstats['h']['std']/dfstats['h']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
ax.set_xlim([0,2e5])
ax.set_xlabel('h mean')
ax.set_ylabel('h std')
plt.show()

# Repeat third iteration
# Calculate the median and the mad of the h_std.
median_h_std = dfstats[('h','std')].median()
mad_h_std = dfstats[('h','std')].mad()
Z = 0.6745*(dfstats[('h', 'std')]-median_h_std)/mad_h_std
dfstats['h_std_MZS'] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.h_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['h']['mean']
y = dfstats['h']['std']/dfstats['h']['mean']

ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
ax.set_xlim([0,2e5])
ax.set_xlabel('h mean')
ax.set_ylabel('h std')
plt.show()

# After you remove the extreme outliers, you can run this again and get rid of the less extreme outliers.
# You can iteratively apply it to get rid of all the outliers. (Run and then plot to inspect if there are outliers.)
# Once there are no outliers, get the average value of the main cluster in the plot. Use it as a characteristic distance.
# This characteristic value would be a measure of the natural deviation from the mean h magnitude.
# Repeat with hphi, hz. We expect htheta to have the main motion of the objects.

# Investigate dependence of htheta in other parameters.
# No significance in the trends with the dh values, but we expect some dependence with dtheta values.
# Effect of J2 on precesion of the nodes (htheta)

# TODO: Save data
#outfile = get_data_homax.plot(x,y,'.k')e()/"DIT_Experiments"/"DistanceMetrics"/"ParameterStats_.csv"
#params_mean.to_csv(outfile,index=False)

# # Observations from the plot: most objects have a small deviation from the mean h magnitude (less than 1%, determine how much it is)
# # There are outliers that have large deviations. These may be object undergoing maneuvers. 
# # Getting the mean standard deviation of the objects in h, hphi, htheta

#%% Nodal precession ##########################################################
# TODO:  
# dhtheta/dt (angular rate of change in hphi), we expect this to be proportional to cos(i)
# See equation for nodal precession (dependent on h,cos(i))
# Plotting cos(i) vs. dhtheta/dt, should be a straight line
# Use data or equation to calculate average distance that h vector moves/precesses

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
# Observations:
# Linear relationship between theoretical nodal precession and the observed change \
# in theta component of the angular momentum. 
# Confirms that the motion of the theta component of the angular momentum is due
# to the precession of the nodes. This is the major movement that we see in the
# satellite's angular momentum over time. We only observe small variations in the
# radial and phi component of angular momentum. 
