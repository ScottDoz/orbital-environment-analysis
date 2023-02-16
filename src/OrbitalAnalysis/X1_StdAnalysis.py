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
import pdb

# Load data
# df = load_vishnu_experiment_data('all')
df = load_2019_experiment_data('all') # New dataset
# TODO: Apply unwrapped to original dataframe (dtheta and dphi)


#%% hr Outlier Removal ########################################################

# Compute stats
dfstats = df.groupby(by='NoradId').agg({'hr':['mean','std','min','max','median','mad','count'],
                                        # 'a':['mean','std','min','max','median','mad','count'],
                                        # 'e':['mean','std','min','max','median','mad','count'],
                                        # 'i':['mean','std','min','max','median','mad','count'],
                                        })

# TODO: Adjust analysis to look at wh rather than dhtheta (dh/dt)
# TODO: Same workflow for dphi (proving that change in dphi is minimal)

# Outlier removal
# NOTES: Remove any objects with changes in their orbit (GTO to GEO). 

# Looking at the plots h std vs. h mean, we want to remove outliers with large standard deviations.
# We want to apply it to h_std.

# A more robust method is the modified Z-score method, using the median
# and the median absolute deviation (MAD).
# Modified z-score = 0.6745*(xi - x_med)/MAD
# Then remove any values with MZS > +/= 3.5
# See Iglewicz & Hoaglin

# Plot original data 
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.15, 0.9, 'Raw data',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hr']['mean']
y = dfstats['hr']['std'] #/dfstats['hr']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
# ax.set_xlim([0,120000])
ax.set_xlabel('$h_{r}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{r}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()


# Calculate the median and the mad of the h_std.
median_hr_std = dfstats[('hr','std')].median()
mad_hr_std = dfstats[('hr','std')].mad()
Z = 0.6745*(dfstats[('hr', 'std')]-median_hr_std)/mad_hr_std
dfstats['hr_std_MZS'] = ''
dfstats['hr_std_MZS'] = Z
# dfstats.insert(8,("hr_std_MZS"),Z)
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hr_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.15, 0.9, '1st Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hr']['mean']
y = dfstats['hr']['std'] #/dfstats['hr']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
# ax.set_xlim([0,120000])
ax.set_xlabel('$h_{r}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{r}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()

# Repeat second iteration
# Calculate the median and the mad of the h_std.
median_hr_std = dfstats[('hr','std')].median()
mad_hr_std = dfstats[('hr','std')].mad()
Z = 0.6745*(dfstats[('hr', 'std')]-median_hr_std)/mad_hr_std
dfstats['hr_std_MZS'] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hr_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.99, 0.9, '2nd Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hr']['mean']
y = dfstats['hr']['std'] #/dfstats['hr']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
# ax.set_xlim([0,120000])
ax.set_xlabel('$h_{r}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{r}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()

# # Repeat third iteration
# # Calculate the median and the mad of the h_std.
# median_hr_std = dfstats[('hr','std')].median()
# mad_hr_std = dfstats[('hr','std')].mad()
# Z = 0.6745*(dfstats[('hr', 'std')]-median_hr_std)/mad_hr_std
# dfstats['hr_std_MZS'] = Z
# # Find indices with MZS>3.5 and remove them
# ind = (abs(dfstats.hr_std_MZS)>3.5) # Outlier indices to remove
# dfstats = dfstats[~ind] # Original dataframe with outliers removed
# # Plotting
# fig, ax = plt.subplots(1,1,figsize=(8, 5))
# plt.text(0.99, 0.9, '3rd Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
# x = dfstats['hr']['mean']
# y = dfstats['hr']['std'] #/dfstats['hr']['mean']
# ax.plot(x,y,'.k')
# # ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
# # ax.set_xlim([0,120000])
# ax.set_xlabel('$h_{r}$ mean (km$^{2}$/s)',fontsize=16)
# ax.set_ylabel('$h_{r}$ std (km$^{2}$/s)',fontsize=16)
# plt.xscale("log")
# plt.show()

# Print average values
print('\nAverage std(hr) = {}\n'.format(y.mean()))


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


#%% hz ###########
# Regenerate dfstats
dfstats = df.groupby(by='NoradId').agg({'hr':['mean','std','min','max','median','mad','count'],
                                        'hz':['mean','std','min','max','median','mad','count'],
                                        # 'htheta':['mean','std','min','max','median','mad','count'],
                                        # 'hphi':['mean','std','min','max','median','mad','count'],
                                        # 'a':['mean','std','min','max','median','mad','count'],
                                        # 'e':['mean','std','min','max','median','mad','count'],
                                        # 'i':['mean','std','min','max','median','mad','count'],
                                        # 'q':['mean','std','min','max','median','mad','count'],
                                        # 'p':['mean','std','min','max','median','mad','count'],
                                        })


# # Deliberate Outlier removal ------------------------------

# # We want to remove any objects that have significant changes in their orbital elements
# # These objects are undergoing active manouvering such as orbit raising and deorbiting.

# # First, drop items with only one or two measurements (~578 objects)
# dfstats = dfstats[dfstats['hz']['count']>2]


# # Next, remove any hyperbolic objects (those that have max(e) >= 1.)
# # 4 objects:
# # Norad  Name
# # 29260 COSMOS 2422
# # 35653 COSMOS 2251 DEB
# # 40747 DELTA 4 R/B
# # 44338 CZ-3B R/B
# ind = dfstats['e']['max']>=1.
# dfstats = dfstats[~ind]


# # Next, we find all objects that have de-orbited over the period of interest.
# # These are objects whose minimum perigee is less than or equal to the earth radius.
# ind = dfstats['q']['min']<=Re
# # 44 objects
# dfstats = dfstats[~ind]

# # Now, remove any objects transitioning from GTO to GEO.
# # These objects have maximum ahove GEO (a = 42,167 km), but a minimum periapsis
# # much lower. 
# # E.g. NoradId = 44053
# Re = 6378.137; #Radius of Earth (km)
# ind = (dfstats['a']['max'] > 40000) & (dfstats['q']['min'] < Re+5000)
# # dfstats[[('a','min'),('a','max'),('q','min'),('q','max')]][ind]
# dfstats = dfstats[~ind]
# # -----------------------------------------------------------------------------

# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.15, 0.9, 'Raw data',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hz']['mean']
y = dfstats['hz']['std']#/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{z}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{z}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()

# Calculate the median and the mad of the hphi_std.
median_hz_std = dfstats[('hz','std')].median()
mad_hz_std = dfstats[('hz','std')].mad()
Z = 0.6745*(dfstats[('hz', 'std')]-median_hz_std)/mad_hz_std
dfstats["hz_std_MZS"] = ''
dfstats["hz_std_MZS"] = Z
# dfstats.insert(8,("hz_std_MZS"),Z)
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hz_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
print("{} points removed".format(sum(ind)))
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.15, 0.9, '1st Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hz']['mean']
y = dfstats['hz']['std'] #/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{z}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{z}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()

# 2nd iteration
# Calculate the median and the mad of the hphi_std.
median_hz_std = dfstats[('hz','std')].median()
mad_hz_std = dfstats[('hz','std')].mad()
Z = 0.6745*(dfstats[('hz', 'std')]-median_hz_std)/mad_hz_std
dfstats["hz_std_MZS"] = Z
# Find indices with MZS>3.5 and remove them
ind = (abs(dfstats.hz_std_MZS)>3.5) # Outlier indices to remove
dfstats = dfstats[~ind] # Original dataframe with outliers removed
print("{} points removed".format(sum(ind)))
# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
plt.text(0.15, 0.9, '2nd Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
x = dfstats['hz']['mean']
y = dfstats['hz']['std'] #/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
# ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{z}$ mean (km$^{2}$/s)',fontsize=16)
ax.set_ylabel('$h_{z}$ std (km$^{2}$/s)',fontsize=16)
plt.xscale("log")
plt.show()

# # 3rd iteration
# # Calculate the median and the mad of the hphi_std.
# median_hz_std = dfstats[('hz','std')].median()
# mad_hz_std = dfstats[('hz','std')].mad()
# Z = 0.6745*(dfstats[('hz', 'std')]-median_hz_std)/mad_hz_std
# dfstats["hz_std_MZS"] = Z
# # Find indices with MZS>3.5 and remove them
# ind = (abs(dfstats.hz_std_MZS)>3.5) # Outlier indices to remove
# dfstats = dfstats[~ind] # Original dataframe with outliers removed
# print("{} points removed".format(sum(ind)))
# # Plotting
# fig, ax = plt.subplots(1,1,figsize=(8, 5))
# plt.text(0.15, 0.9, '3rd Iteration',ha='right', va='bottom',transform=plt.gca().transAxes)
# x = dfstats['hz']['mean']
# y = dfstats['hz']['std'] #/dfstats['hphi']['mean']
# ax.plot(x,y,'.k')
# # ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
# #ax.set_xlim([0,2e5])
# ax.set_xlabel('$h_{z}$ mean (km$^{2}$/s)',fontsize=16)
# ax.set_ylabel('$h_{z}$ std',fontsize=16)
# plt.xscale("log")
# plt.show()

# Print average values
print('\nAverage std(hz) = {}\n'.format(y.mean()))



#%% hphi ###########
# Regenerate dfstats
dfstats = df.groupby(by='NoradId').agg({'hr':['mean','std','min','max','median','mad','count'],
                                        'hz':['mean','std','min','max','median','mad','count'],
                                        'htheta':['mean','std','min','max','median','mad','count'],
                                        'hphi':['mean','std','min','max','median','mad','count']})

# Plotting
fig, ax = plt.subplots(1,1,figsize=(8, 5))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std']#/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{\phi}$ mean',fontsize=16)
ax.set_ylabel('$h_{\phi}$ std',fontsize=16)
plt.show()



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
fig, ax = plt.subplots(1,1,figsize=(8, 5))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std'] #/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{\phi}$ mean',fontsize=16)
ax.set_ylabel('$h_{\phi}$ std',fontsize=16)
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
fig, ax = plt.subplots(1,1,figsize=(8, 5))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std'] #/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{\phi}$ mean',fontsize=16)
ax.set_ylabel('$h_{\phi}$ std',fontsize=16)
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
fig, ax = plt.subplots(1,1,figsize=(8, 5))
x = dfstats['hphi']['mean']
y = dfstats['hphi']['std'] #/dfstats['hphi']['mean']
ax.plot(x,y,'.k')
ax.plot([x.min(),x.max()],[y.mean(),y.mean()],'-r')
#ax.set_xlim([0,2e5])
ax.set_xlabel('$h_{\phi}$ mean',fontsize=16)
ax.set_ylabel('$h_{\phi}$ std',fontsize=16)
plt.show()