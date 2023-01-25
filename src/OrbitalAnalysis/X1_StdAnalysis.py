# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:00:33 2023

@author: mauri

Goal: Find the average standard deviation of the entire population. What is the
magnitude of the fluctuations in hr, htheta, hphi?
Motivation is to see how stable the angular momentum coordinates are.

Method: 
    1. Plot std[h] vs. mean[h] (1 point per satellite)
    2. Compute median and MAD of entire catalog (single value for entire catalog)
    3. Compute MZS-scores for each satellite (one point per satellite)
    4. Remove any outliers (MZS>3.5)
    5. Update median and MAD of catalog and iterate
    
Results:
    * Most points form a straight line, but there are significant outliers
    * After the outlier removal, we obtain a consistent value of std
    
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
ax.plot(x,y,'.k')
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
# TODO: Getting the mean standard deviation of the objects in h, hphi, htheta

#%% hphi ###########
# Regenerate dfstats
dfstats = df.groupby(by='NoradId').agg({'h':['mean','std','min','max','median','mad','count'],
                                        'hz':['mean','std','min','max','median','mad','count'],
                                        'htheta':['mean','std','min','max','median','mad','count'],
                                        'hphi':['mean','std','min','max','median','mad','count']})

# Calculate the median and the mad of the hphi_std.
median_hphi_std = dfstats[('hphi','std')].median()
mad_hphi_std = dfstats[('hphi','std')].mad()
Z = 0.6745*(dfstats[('hphi', 'std')]-median_hphi_std)/mad_hphi_std
dfstats.insert(8,("hphi_std_MZS"),Z)
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hphi_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
print("{} points removed".format(sum(ind)))
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std']#/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('hphi mean')
ax.set_ylabel('hphi std')
plt.show()

# Repeat second iteration
# Calculate the median and the mad of the h_std.
median_hphi_std = dfstats[('hphi','std')].median()
mad_hphi_std = dfstats[('hphi','std')].mad()
Z = 0.6745*(dfstats[('hphi', 'std')]-median_hphi_std)/mad_hphi_std
dfstats['hphi_std_MZS'] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hphi_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
print("{} points removed".format(sum(ind)))
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std']#/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('hphi mean')
ax.set_ylabel('hphi std')
plt.show()

# Repeat third iteration
# Calculate the median and the mad of the h_std.
median_hphi_std = dfstats[('hphi','std')].median()
mad_hphi_std = dfstats[('hphi','std')].mad()
Z = 0.6745*(dfstats[('hphi', 'std')]-median_hphi_std)/mad_hphi_std
dfstats['hphi_std_MZS'] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hphi_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
print("{} points removed".format(sum(ind)))
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 8))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std']#/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('hphi mean')
ax.set_ylabel('hphi std')
plt.show()