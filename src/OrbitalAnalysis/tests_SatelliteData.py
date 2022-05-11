# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:01:03 2022

@author: scott

Tests for satellite data and visualization

"""

from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *
# from Overpass import *
# from Ephem import *
# from Events import *
# from GmatScenario import *


#%% Satellite Data Loading
# Methods from SatelliteData.py module to load orbital elements of satellites.

def test_query_norad():
    ''' Load the satellite for a set of Norad IDs '''
      
    IDs = [25544, 41335]
    df = query_norad(IDs,compute_params=True)
    
    return df

def test_load_satellite_data():
    ''' Load the satellite data and compute orbital parameters '''
    
    df = load_satellites(group='all',compute_params=True)
    
    return df

def test_load_vishnu_data_single_month():
    ''' Load Vishnu experiment data for a single month '''
    
    df = load_vishnu_experiment_data(1)
    
    return df

def test_load_vishnu_data_list_month():
    ''' Load Vishnu experiment data for a list of months '''
    
    df = load_vishnu_experiment_data([1,2,3])
    
    return df

def test_load_vishnu_data_all():
    ''' Load Vishnu experiment data for all months '''
    
    df = load_vishnu_experiment_data('all')
    
    return df

#%% Visualization h-space
# Methods from Clustering.py and Visualization.py modules for visualizing the
# satellites in orbital momentum space.

def test_h_space_visualization():
    ''' 
    Generate a 3D scatter plot of the satellite catalog in specific orbital 
    angular momentum space (hx,hy,hz).
    '''
    
    # Load the data
    # df = load_satellites(group='all',compute_params=True)
    df = load_vishnu_experiment_data(1)
    
    # Generate clusters in (h,hz) coordiantes
    label = 'test_clusters' # Field name holding clusters
    features = ['h','hz']   # Fields to use in clustering 
    df = generate_Kmeans_clusters(df,label,features,n_clusters=15,random_state=170)
    
    # Generate plotly figure and render in browser
    plot_h_space_cat(df,'test_clusters')
    
    return

def test_h_space_timeseries_visualization():
    '''
    Generate a 3D scatter plot showing the trajectories of the satellite
    catalog in specific orbital angular momentum space.

    '''
    
    # Load all data from Vishnu (~200,000 points)
    df = load_vishnu_experiment_data('all')
    obj_list = list(df.NoradId.unique()) # List of unique objects (17,415)
    
    # Generate clusters in (h,hz) coordiantes
    label = 'test_clusters' # Field name holding clusters
    features = ['h','hz']   # Fields to use in clustering 
    df = generate_Kmeans_clusters(df,label,features,n_clusters=15,random_state=170)
    
    
    # Randomly sample ~ 1000 objects
    import random
    objs = random.sample(obj_list,1000)
    df = df[df.NoradId.isin(objs)]
    
    # Plot the scatter plot
    plot_h_space_cat(df,cat='test_clusters')
    
    return

def test_2d_scatter_visualization():
    '''
    Generate a 2D scatter plot selecting x,y,color coordinates from available 
    data. This example plots (h,hz) and color = inclination

    '''
    
    # Load the data
    df = load_satellites(group='all',compute_params=True)
    
    # Compute distances
    # target = 13552 # COSMOS 1408
    # df = compute_distances(df, target)
    
    # Generate figure and render in browser
    plot_2d_scatter_numeric(df,'h','hz','i')
    
    return

#%% DensityAnalysis

def test_kde_visualization():
    '''
    Generate a 2D scatter plot of the positions, with a heat map showing the
    density computed using Kernel Density Estimation. This example plots (h,hz).
    '''
    
    # Load the data
    df = load_satellites(group='all',compute_params=True)
    
    plot_kde(df,'h','hz')
    
    return

#%% DistanceAnalysis

def test_distances():
    '''
    Test the computation of various distance metrics from a target satellite of
    interest. 
    
    '''
    
    # NORAD ID of target
    target = 13552 # COSMOS 1408
    
    # Load data
    df = load_satellites(group='all',compute_params=True)
    
    # Compute distances to target
    df = compute_distances(df, target)
    
    # Find the closest objects to the target (removing any related debris)
    df[['name','a','e','dH']][~df.name.str.contains('COSMOS 1408')].sort_values(by='dH')
    
    # We find that two objects are fairly close in terms of dH metric to the target
    # (18421) COSMOS 1892, and (15495) SL-14 R/B.
    # These objects are in the COSMOS 1408 debris cluster.
    
    # Both these objects will theoreticaly have a large number of objects with 
    # which they may be confused.
    
    
    return df