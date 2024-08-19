# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:32:11 2023

@author: scott

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import plotly.graph_objects as go
import plotly
import plotly.express as px

from astroML.plotting import hist

import pdb

from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.Clustering import *
from OrbitalAnalysis.Visualization import *
from OrbitalAnalysis.Density import *

#%% Network graph

def plot_socrates_network():

    # Load current database as points data
    data = load_satellites(group='all')
    data.set_index('NoradId',inplace=True)
    
    # Load edges from socrates database
    edges = load_socrates()
    source_list = edges['NORAD_CAT_ID_1'].to_list()
    target_list = edges['NORAD_CAT_ID_2'].to_list()
    # Find objects in edges list that are not in data
    missing_sources = set.difference(set(source_list),set(data.index))
    missing_targets = set.difference(set(target_list),set(data.index))
    # Remove 
    edges = edges[~edges.NORAD_CAT_ID_2.isin(missing_targets)]
    edges = edges[~edges.NORAD_CAT_ID_2.isin(missing_sources)]
    # Update
    source_list = edges['NORAD_CAT_ID_1'].to_list()
    target_list = edges['NORAD_CAT_ID_2'].to_list()
        
    # Create trace for nodes
    scatter_trace = go.Scatter3d(
        x=data['hx'],
        y=data['hy'],
        z=data['hz'],
        mode='markers',
        marker=dict(size=1, color='blue')
    )
    
    # Create edge trace
    edge_x = np.column_stack([ data['hx'][source_list].to_numpy(), data['hx'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()
    edge_y = np.column_stack([ data['hy'][source_list].to_numpy(), data['hy'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()
    edge_z = np.column_stack([ data['hz'][source_list].to_numpy(), data['hz'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(width=0.1, color='gray')
    )

           
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )
    
    fig = go.Figure(data=[scatter_trace, edge_trace], layout=layout)
    
    plotly.offline.plot(fig)
    
    return


def collision_statistics():
    
    # Load edges from socrates database
    edges = load_socrates()
    
    # Count occurances of each satellite
    source_stats = edges.groupby('NORAD_CAT_ID_1').size().reset_index(name='source_counts').rename({'NORAD_CAT_ID_1':"NoradId"})
    target_stats = edges.groupby('NORAD_CAT_ID_2').size().reset_index(name='target_counts').rename({'NORAD_CAT_ID_2':"NoradId"})
    
    # Rename columns
    source_stats.rename(columns={'NORAD_CAT_ID_1':"NoradId"},inplace=True)
    target_stats.rename(columns={'NORAD_CAT_ID_2':"NoradId"},inplace=True)
    
    # Merge into single dataframe
    stats = pd.merge(source_stats,target_stats,on='NoradId',how='outer')
    stats['COLA_num'] = stats['source_counts'] + stats['target_counts']    
    
    # Load satellite data and merge
    df = load_satellites(group='all')
    df = pd.merge(df,stats[['NoradId','COLA_num']],on='NoradId',how='left')
    
    # Compute Density
    log_satdens = compute_density(df)
    df['log_p'] = log_satdens # Log density
    df['p_hxhyhz'] = log_satdens
    # Compute density
    df['p'] = 10 ** df.log_p
    
    
    # # Plot
    # fig = plot_2d_scatter_numeric(df, 'hx','hz','COLA_num',size=3,equalscale=False,render=False)
    # plotly.offline.plot(fig)
    
    # No correlation between density and number of collision warnings
    
    return 

