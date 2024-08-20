# -*- coding: utf-8 -*-
"""
Created on Sun May 28 12:14:50 2023

@author: scott

Density Module
--------------

Methods to generate density distributions.

"""

import matplotlib.pyplot as plt
import numpy as np

# from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import pdb

# Module imports
from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.utils import get_data_home
from OrbitalAnalysis.Visualization import *

#%% Define Catalog class

class Catalog():
    
    
    def __init__(self,dataset='spacetrack',period=1):
        
        # Load dataset
        if dataset == '2019':
            df = load_2019_experiment_data(period)
        elif dataset.lower() == 'vishnu':
            df = load_vishnu_experiment_data(period)
        else:
            df = load_satellites(group='all',compute_params=True)
        
        # Compute density
        df['p_hxhyhz'] = compute_density(df)

        # Store attributes
        self.df0 = df # Dataframe Reference dataframe
        self.norad_list = set(df.NoradId) # List of objects
        self.num_objects = len(self.norad_list)
    
    
    # Remove satellites -------------------------------------------------------
    
    def remove_satellites(sats):
        
        # Remove objects from Base
        df = self.df0.copy() # Copy base catalog
        
        
        return
    
    
    
    # Plotting methods
    def plot_density(self):
        
        # Initialize figure
        fig = plot_3d_scatter_numeric(self.df0,'hx','hy','hz',color='p_hxhyhz',
                                    logColor=False,colorscale='Blackbody_r',
                                    xrange=[-120000,120000],
                                    yrange=[-120000,120000],
                                    zrange=[-50000,150000],
                                    aspectmode='cube',
                                    render=False,
                                    )
        # fig.show()
        plotly.offline.plot(fig)
        
        return



#%% Compute density

def compute_density(df):
    '''
    Apply KDE using a number of different parameter sets, with the optimal
    kernel and bandwidth identified from cross validation steps. Evauate the
    log-likelihood density values at the associated coordinates of all satellites
    in the catalog. Return a dataframe containing normalized parameters and
    the log-likelihood density values.

    '''
    
    df1 = df.copy() # Copy original dataframe
    
    # Limit data fields
    df = df[['Name','NoradId','a','e','i','om','w','M','p','q','Q',
             'h','hx','hy','hz','hphi','htheta']]
    
    # Scale all data fields independently to 0-1
    min_max_scaler = preprocessing.MinMaxScaler()
    features = ['a','e','i','om','w','M','p','q','Q',
                'h','hx','hy','hz','hphi','htheta']
    Xfull = df[features].to_numpy()
    Xfull = min_max_scaler.fit_transform(Xfull)
    df[features] = Xfull
    
    # Determine optimal bandwidth
    features = ['hx','hy','hz']
    bandwidth = 0.008858667904100823
    
    
    # Apply KDE and evaluate at satellite points.
    # Uncomment/add different parameter sets
    
    X = df[features].to_numpy()
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    
    # Add to dataframe
    # df1['p_hxhyhz'] = log_satdens  
    
    return log_satdens


# See: evaluate_satellite_densities(save_output=True) in Ex_DensityAnalysis.py

#%%
    
def compute_density_contributions(df,targets):
    '''
    Apply KDE using a number of different parameter sets, with the optimal
    kernel and bandwidth identified from cross validation steps. Evauate the
    log-likelihood density values at the associated coordinates of all satellites
    in the catalog. Return a dataframe containing normalized parameters and
    the log-likelihood density values.

    '''
    
    df1 = df.copy() # Copy original dataframe
    
    # Limit data fields
    df = df[['Name','NoradId','a','e','i','om','w','M','p','q','Q',
             'h','hx','hy','hz','hphi','htheta']]
    
    # Scale all data fields independently to 0-1
    min_max_scaler = preprocessing.MinMaxScaler()
    features = ['a','e','i','om','w','M','p','q','Q',
                'h','hx','hy','hz','hphi','htheta']
    Xfull = df[features].to_numpy()
    Xfull = min_max_scaler.fit_transform(Xfull)
    df[features] = Xfull
    
    # Determine optimal bandwidth
    features = ['hx','hy','hz']
    bandwidth = 0.008858667904100823
    
    
    # Fit KDE with data points
    X = df[features][df['NoradId'].isin(targets)].to_numpy()
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde1 = kde1.fit(X)
    
    
    # Evaluating at satellite points
    X = df[features].to_numpy()
    log_satdens = kde1.score_samples(X)
    
    # Add to dataframe
    # df1['dp_hxhyhz'] = log_satdens  
    
    return log_satdens

def compute_density_contributions_complement(df,targets):
    '''
    Apply KDE using a number of different parameter sets, with the optimal
    kernel and bandwidth identified from cross validation steps. Evauate the
    log-likelihood density values at the associated coordinates of all satellites
    in the catalog. Return a dataframe containing normalized parameters and
    the log-likelihood density values.

    Complement - due to other than the target

    '''
    
    df1 = df.copy() # Copy original dataframe
    
    # Limit data fields
    df = df[['Name','NoradId','a','e','i','om','w','M','p','q','Q',
             'h','hx','hy','hz','hphi','htheta']]
    
    # Scale all data fields independently to 0-1
    min_max_scaler = preprocessing.MinMaxScaler()
    features = ['a','e','i','om','w','M','p','q','Q',
                'h','hx','hy','hz','hphi','htheta']
    Xfull = df[features].to_numpy()
    Xfull = min_max_scaler.fit_transform(Xfull)
    df[features] = Xfull
    
    # Determine optimal bandwidth
    features = ['hx','hy','hz']
    bandwidth = 0.008858667904100823
    
    
    # Fit KDE with data points
    X = df[features][~df['NoradId'].isin(targets)].to_numpy()
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde1 = kde1.fit(X)
    
    
    # Evaluating at satellite points
    X = df[features].to_numpy()
    log_satdens = kde1.score_samples(X)
    
    # Add to dataframe
    # df1['dp_hxhyhz'] = log_satdens  
    
    
    return log_satdens
