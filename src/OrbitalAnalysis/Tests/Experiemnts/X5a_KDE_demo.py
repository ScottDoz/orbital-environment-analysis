# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:54:08 2023

@author: scott

Demonstrate the Principles of KDE
---------------------------------

"""

import matplotlib.pyplot as plt
import numpy as np
from astroML.plotting import hist
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from filterpy.stats import plot_covariance_ellipse, plot_covariance
from matplotlib.patches import Polygon

import plotly.graph_objects as go
import plotly
import plotly.express as px

# Module imports
from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.utils import get_data_home
from OrbitalAnalysis.Visualization import *
from OrbitalAnalysis.Density import *
from OrbitalAnalysis.Distances import *

#%% Load catalog

def process_data():

    # Load at 2nd epoch - full catalog
    df = load_2019_experiment_data(36) # New dataset
    df['Name'][pd.isna(df.Name)] = ''
    
    # Get scalling [0-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    features = ['hx','hy','hz']
    Xfull = df[features].to_numpy()
    Xfull = min_max_scaler.fit_transform(Xfull)
    # Get ranges of hx,hy,hz
    
    # Bandwidth is the standard deviation
    bandwidth = 0.008858667904100823 # Non-dimensional bandwidth
    sig_x = bandwidth*min_max_scaler.data_range_[0] # std in hx
    sig_y = bandwidth*min_max_scaler.data_range_[1] # std in hy
    sig_z = bandwidth*min_max_scaler.data_range_[2] # std in hz
    sig = np.mean([sig_x,sig_y]) # Mean
    
    # Add scalled coords to dataframe
    df[['hxs','hys','hzs']] = Xfull
    
    # Compute Density  
    df['log_p'] = compute_density(df) # Log density
    df['p'] = np.exp(df.log_p) # Compute density
    df['p'] = df['p']/df['p'].sum() # Normalize to unity
    
    return df, sig_x, sig_y, sig_z, sig


#%% Compute distances from target object

def select_tagets(df, case = 'Cosmos 2251', method='knn'):
    
    
    # Select target
    # target = 25544 # ISS
    
    if case == 'Cosmos 2251':
        target = 22675 # Cosmos 2251 *
        hz_range = [14000, 15250] # hz range of ring cluster
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    # target = 49445 # Atlas 5 Centaur DEB
    
    # Compute distances from target to other satellites
    # Compute distance metrics
    df = compute_distances(df,target,searchfield='NoradId')
    df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl','dHcyl_sign':'d_cylsign'}) # Rename

    # Subsets of objects
    if method == 'knn':
    
        # 1. Debris objects
        dfdeb = df[df['Name'].str.contains("DEB")]
        
        # Find peices of debris to remove. 10 closest peices of debris
        k = 10 # Number of samples
        dft = dfdeb.nsmallest(k, 'd_Euc')
        targets = dft.NoradId.to_list() # List of norads
        
    elif method == '1 sig':
        
        # 1. Debris objects
        dfdeb = df[df['Name'].str.contains("DEB")]
        
        # Find peices of debris to remove. 10 closest peices of debris
        k = 10 # Number of samples
        dft = dfdeb.nsmallest(k, 'd_Euc')
        targets = dft.NoradId.to_list() # List of norads
        
        # Find closest items to d_Euc = 0, 1*sig, 2*sig, ...
        targets = []
        for i in range(k):
            # Find norad closes to i*sigma
            row = dfdeb.iloc[(dfdeb['d_Euc']-i*sig).abs().argsort()[:1]]
            targets.append(row.NoradId.iloc[0]) # Add to list
        
        dft = dfdeb[dfdeb['NoradId'].isin(targets)]
    
    elif method == '2 sig':
        
        # 1. Debris objects
        dfdeb = df[df['Name'].str.contains("DEB")]
        
        # Find peices of debris to remove. 10 closest peices of debris
        k = 10 # Number of samples
        dft = dfdeb.nsmallest(k, 'd_Euc')
        targets = dft.NoradId.to_list() # List of norads
        
        # Find closest items to d_Euc = 0, 1*2sig, 2*2sig, ...
        targets = []
        for i in range(k):
            # Find norad closes to i*sigma
            row = dfdeb.iloc[(dfdeb['d_Euc']-i*2*sig).abs().argsort()[:1]]
            targets.append(row.NoradId.iloc[0]) # Add to list
        
        dft = dfdeb[dfdeb['NoradId'].isin(targets)]
    
    elif method == '3 sig':
        
        # 1. Debris objects
        dfdeb = df[df['Name'].str.contains("DEB")]
        
        # Find peices of debris to remove. 10 closest peices of debris
        k = 10 # Number of samples
        dft = dfdeb.nsmallest(k, 'd_Euc')
        targets = dft.NoradId.to_list() # List of norads
        
        # Find closest items to d_Euc = 0, 1*3sig, 2*3sig, ...
        targets = []
        for i in range(k):
            # Find norad closes to i*sigma
            row = dfdeb.iloc[(dfdeb['d_Euc']-i*3*sig).abs().argsort()[:1]]
            targets.append(row.NoradId.iloc[0]) # Add to list
        
        dft = dfdeb[dfdeb['NoradId'].isin(targets)]

    # Compute normalized density distributions 
    
    N = len(df) # Number of objects
    k = len(targets) # Number of targets
    
    # Compute log density contribution due to targets
    df['log_p_targ'] = compute_density_contributions(df,targets)     # All Target
    df['p_targ'] = np.exp(df['log_p_targ']) # Convert to density contribution due to targets
    df['p_targ'] = df['p_targ']/df['p_targ'].sum()
    # df['p_targ'] = df['p_targ'] / norm_const # Normalize subset
    
    # Compute log density contribution due to non-targets
    df['log_p_nontarg'] = compute_density_contributions_complement(df,targets)     # All Target
    df['p_nontarg'] = np.exp(df['log_p_nontarg']) # Convert to density contribution due to targets
    df['p_nontarg'] = df['p_nontarg']/df['p_nontarg'].sum()
    
    # All distributions now sum to 1
        
    # Multiply them by the lengths to get contribution
    df['p_targ'] = df['p_targ']*k/N
    df['p_nontarg'] = df['p_nontarg']*(N-k)/N
    
    # 2. Compute fractional change in density
    df['dp/p'] = df['p_targ']/df['p']
    

    return df, dft, target, targets, hz_range



#%% Compute normalized density distributions

# # Check log_p = log_p_targ + log_p_nontarg
# df[['log_p','log_p_targ','log_p_nontarg']]

# 10**df.log_p - 10**df.log_p_targ - 10**df.log_p_nontarg

# np.exp(df.log_p) - np.exp(10**df.log_p_targ) - np.exp(df.log_p_nontarg)


# # Compare regular density values
# df[['p','p_targ','p_nontarg']]
# df['p'] - df['p_targ'] - df['p_nontarg']


#%% Plot 3-sigma elipses of kernels of target objects

def plot_targets_3sig(df,dft,target, targets,hz_range):

    cmap = 'hot_r'
    
    # Vmin,vmax
    vmin=df['log_p'].min()
    vmax=df['log_p'].max()
    
    
    fig, axs = plt.subplots(1,3,figsize=(25, 8))
    # # hx vs hz
    # # axs[0,0].plot(df['hx'],df['hz'],'.k') # All objects
    # axs[0].plot(dft['hx'],dft['hz'],'.r') # Tagets
    # sc1 = axs[0].scatter(df['hx'],df['hz'],s=1,c=df['log_p'],cmap=cmap,vmin=vmin,vmax=vmax) # All objects
    # # axs[0,0].scatter(dft['hx'],dft['hz'],s=1,c='r') # Tagets
    # # Add 3-sigma ellipses
    # P = [[sig_x**2,0],[0,sig_z**2]]
    # plt.sca(axs[0]) # Set axis
    # for i in range(len(dft)):
    #     plot_covariance_ellipse((dft['hx'].iloc[i], dft['hz'].iloc[i]), P, 
    #                             fc='g', alpha=0.1, std=[1, 2, 3])
    # axs[0].set_ylabel("hx")
    # axs[0].set_ylabel("hz")
    # plt.colorbar(sc1, label='log density')
    
    # hx vs hz (zoomed)
    # axs[1,0].plot(df['hx'],df['hz'],'.k') # All objects
    axs[0].plot(dft['hx'],dft['hz'],'.r') # Tagets
    sc2 = axs[0].scatter(df['hx'],df['hz'],s=1,c=df['log_p'],cmap=cmap,vmin=vmin,vmax=vmax) # All objects
    # axs[1,0].scatter(dft['hx'],dft['hz'],s=1,c='r') # Tagets
    # Add 3-sigma ellipses
    P = [[sig_x**2,0],[0,sig_z**2]]
    plt.sca(axs[0]) # Set axis
    for i in range(len(dft)):
        plot_covariance_ellipse((dft['hx'].iloc[i], dft['hz'].iloc[i]), P, 
                                fc='g', alpha=0.1, std=[1, 2, 3])
    axs[0].set_ylabel("hx")
    axs[0].set_ylabel("hz")
    axs[0].set_aspect('equal')
    axs[0].set_aspect('equal', 'box')
    axs[0].set_xlim(-60000,60000)
    axs[0].set_ylim(-20000,140000)
    
    plt.colorbar(sc2, label='log density',fraction=0.030, orientation='horizontal')
    
    # hx vs hy for ring
    # axs[1].plot(df['hx'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],df['hy'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],'.k') # All objects
    axs[1].plot(dft['hx'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],dft['hy'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],'.r') # All objects
    sc3 = axs[1].scatter(df['hx'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],df['hy'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],s=1,c=df['log_p'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],cmap=cmap,vmin=vmin,vmax=vmax) # All objects
    # axs[0,1].scatter(dft['hx'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],dft['hy'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],s=1,c='r') # All objects
    # Add 3-sigma ellipses
    P = [[sig_x**2,0],[0,sig_y**2]]
    plt.sca(axs[1]) # Set axis
    for i in range(len(dft)):
        plot_covariance((dft['hx'].iloc[i], dft['hy'].iloc[i]), P, 
                        fc='g', alpha=0.1, std=[1, 2, 3])
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlabel("hx")
    axs[1].set_ylabel("hy")
    # plt.colorbar(sc3, label='log density')
    
    # hx vs hy (zoomed)
    # axs[1,1].plot(df['hx'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],df['hy'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],'.k') # All objects
    axs[2].plot(dft['hx'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],dft['hy'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],'.r') # All objects
    sc4 = axs[2].scatter(df['hx'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],df['hy'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],s=1,c=df['log_p'][(df.hz >= hz_range[0]) & (df.hz <= hz_range[1])],cmap=cmap,vmin=vmin,vmax=vmax) # All objects
    # axs[1,1].scatter(dft['hx'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],dft['hy'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])],s=1,c='r') # All objects
    # Add 3-sigma ellipses
    P = [[sig_x**2,0],[0,sig_y**2]]
    plt.sca(axs[2]) # Set axis
    for i in range(len(dft)):
        plot_covariance_ellipse((dft['hx'].iloc[i], dft['hy'].iloc[i]), P, 
                                fc='g', alpha=0.1, std=[1, 2, 3])
    axs[2].set_aspect('equal', 'box')
    axs[2].set_xlabel("hx")
    axs[2].set_ylabel("hy")
    axs[2].set_xlim(df['hx'][df.NoradId==target].iloc[0] - 20000,
                      df['hx'][df.NoradId==target].iloc[0] + 20000)
    axs[2].set_ylim(df['hy'][df.NoradId==target].iloc[0] - 20000,
                      df['hy'][df.NoradId==target].iloc[0] + 20000)
    # plt.colorbar(sc4, label='log density')
    
    return

#%% Plot of (log) density contribution from these targets to all others

def plot_density_contribution(df,dft,target, targets,hz_range):

    fig, axs = plt.subplots(1,2,figsize=(16, 8))
    
    # Get average radius of the ring
    ravg = dft['h'][(dft.hz >= hz_range[0]) & (dft.hz <= hz_range[1])].mean()
    theta0 = df['htheta'][df.NoradId == target].iloc[0]
    
    # Left plot
    axs[0].plot(ravg*(df['htheta'][~df['NoradId'].isin(targets)]-theta0),df['p_targ'][~df['NoradId'].isin(targets)],'.k')
    axs[0].set_xlabel("Distance from Center of Cluster")
    axs[0].set_ylabel("Density contribution due to targets")
    axs[0].grid(True, which="both")
    axs[0].set_xlim(-60000,60000)
    # axs[0].set_ylim(-258,20)
    axs[0].set_yscale('log')
    # Add lines for sigma values
    for i in range(len(dft)):
        # 1 sigma
        axs[0].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - sig, ravg*(dft['htheta'].iloc[i] -theta0) + sig, alpha=0.5, color='g')
        # 2 sigma
        axs[0].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) + sig, ravg*(dft['htheta'].iloc[i] -theta0) + 2*sig, alpha=0.2, color='g')
        axs[0].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - sig, ravg*(dft['htheta'].iloc[i] -theta0) - 2*sig, alpha=0.2, color='g')
        # 3 sigma
        axs[0].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) + 2*sig, ravg*(dft['htheta'].iloc[i] -theta0) + 3*sig, alpha=0.1, color='g')
        axs[0].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - 2*sig, ravg*(dft['htheta'].iloc[i] -theta0) - 3*sig, alpha=0.1, color='g')

    
    # Right plot
    axs[1].plot(ravg*(df['htheta'][~df['NoradId'].isin(targets)]-theta0),df['dp/p'][~df['NoradId'].isin(targets)],'.k')
    axs[1].set_xlabel("Distance from Center of Cluster")
    axs[1].set_ylabel("Fractional Density contribution due to targets")
    axs[1].grid(True, which="both")
    axs[1].set_xlim(-60000,60000)
    # axs[1].set_ylim(10**-258,10**20)
    axs[1].set_yscale('log')
    # axs[1].set_ylim(top=1000)
    # Add lines for sigma values
    for i in range(len(dft)):
        # 1 sigma
        axs[1].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - sig, ravg*(dft['htheta'].iloc[i] -theta0) + sig, alpha=0.5, color='g')
        # 2 sigma
        axs[1].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) + sig, ravg*(dft['htheta'].iloc[i] -theta0) + 2*sig, alpha=0.2, color='g')
        axs[1].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - sig, ravg*(dft['htheta'].iloc[i] -theta0) - 2*sig, alpha=0.2, color='g')
        # 3 sigma
        axs[1].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) + 2*sig, ravg*(dft['htheta'].iloc[i] -theta0) + 3*sig, alpha=0.1, color='g')
        axs[1].axvspan( ravg*(dft['htheta'].iloc[i] -theta0) - 2*sig, ravg*(dft['htheta'].iloc[i] -theta0) - 3*sig, alpha=0.1, color='g')


    # Compute total reductions
    tot_dp = df['p_targ'][~df['NoradId'].isin(targets)].sum()
    axs[0].text(0.01, 1.05, 'Total $\Delta$p = {}'.format(np.round(tot_dp,5)), fontsize=12,ha='left', va='top', transform=axs[0].transAxes)
    print('Total change in density: {}'.format(df['p_targ'][~df['NoradId'].isin(targets)].sum() ))

    return

#%% Main function

# Load data (do this once)
# df0, sig_x, sig_y, sig_z, sig = process_data()

# # Select targets Case 1
# df1, dft1, target1, targets1, hz_range1 = select_tagets(df0, case = 'Cosmos 2251', method='knn')
plot_targets_3sig(df1,dft1,target1, targets1,hz_range1)
plot_density_contribution(df1,dft1,target1,targets1,hz_range1)
    
# Select targets Case 2: 1 sigma spread
# df2, dft2, target2, targets2, hz_range2 = select_tagets(df0, case = 'Cosmos 2251', method='1 sig')
plot_targets_3sig(df2,dft2,target2, targets2,hz_range2)
plot_density_contribution(df2,dft2,target2,targets2,hz_range2)

# Case 3: 3-sigma spread
# df3, dft3, target3, targets3, hz_range3 = select_tagets(df0, case = 'Cosmos 2251', method='3 sig')
plot_targets_3sig(df3,dft3,target3, targets3,hz_range3)
plot_density_contribution(df3,dft3,target3,targets3,hz_range3)



