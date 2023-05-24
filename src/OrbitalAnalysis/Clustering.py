# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:22:42 2022

@author: scott

Clustering Module
-----------------

Methods to visualize the density of catalog indifferent multi-dimensional spaces.

"""

import pandas as pd
import numpy as np
import os


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pdb

# Relative imports
from OrbitalAnalysis.SatelliteData import load_satellites
from OrbitalAnalysis.AsteroidData import load_asteroids


#%% Create clusters

def generate_Kmeans_clusters(df,label,features,n_clusters,random_state=170, plot=False):
    '''
    Apply Kmean clustering to generate a set of labels.

    Parameters
    ----------
    df : TYPE
        Dataframe
    label : TYPE
        Label of column in output dataframe
    features : TYPE
        List of features to use in clustering.
    n_clusters : TYPE
        Number of clusters
    random_state : TYPE, optional
        Seed value (for reproducibility). The default is 170.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    # Experiment with kMeans clustering on (h,hphi) space
    # Cluster in (h,hphi),(hphi,hz) 
    
    # See https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
    
    # Extract coordinates
    X = df[features].to_numpy()
    
    # Scale the data
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Predict
    km = KMeans(n_clusters, random_state=random_state).fit(X_scaled)
    
    y_pred = km.fit_predict(X_scaled)
    # TODO: Save the model, to apply to different datasets
    
    # Add clusters to dataframe
    df[label] = y_pred
    
    
    # Plot the clustering
    if plot==True:
        
        if len(features)==2:
            # 2D plot
            
            # For now, use matplotlib
            import matplotlib.pyplot as plt
            
            # Generate a voronoi diagram from the cluster centroids
            from scipy.spatial import voronoi_plot_2d, Voronoi
            points = km.cluster_centers_
            vor = Voronoi(points)
            
            fig,ax = plt.subplots(1,1)
            voronoi_plot_2d(vor,ax) # Plot voronoi
            ax.scatter(points[:,0],points[:,1],s=2,color='r') # Centroids
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)
            plt.show()
            
            
            
        elif len(features)==3:
            # TODO: 3D plot
            pass
        
        elif len(features)>3:
            # TODO: Multi-dimensional data.
            # Apply PCA to reduce dimenstions
            pass
        
    
    return df


