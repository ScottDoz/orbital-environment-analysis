# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:52:47 2023

@author: scott

Orbital Zip Codes
-----------------

Experiments related to segmenting the Angular Momentum space into regions.
* First, define segmentations in hz direction
* Then, determine ways to segment the azimuthal direction


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

#%% Segmention in hz

def segment_catalog_hz(df):
    '''
    Analyze the 2019 dataset to determine distribution of hz values.
    Find thresholds to segment the catalog into distinct rings.

    Returns
    -------
    None.

    '''
    
    # # Load single data
    # df = load_2019_experiment_data([0]) # 
    
    # fig = plot_3d_scatter_numeric(df,'hx','hy','hz',color='hz',
    #                         logColor=False,colorscale='Blackbody_r',
    #                         xrange=[-120000,120000],
    #                         yrange=[-120000,120000],
    #                         zrange=[-50000,150000],
    #                         aspectmode='cube',
    #                         render=False,
    #                         )
    
    # fig = plot_2d_scatter_numeric(df, '1/h2','hz','cos_i',size=3,equalscale=False,render=False)
    # fig = plot_2d_scatter_numeric(df, 'cos_i','hz','1/h2',size=3,equalscale=False,render=False) # Interesting
    # fig = plot_2d_scatter_numeric(df, 'h2','cos_i','hz',size=3,equalscale=False,render=False)
    
    fig = plot_2d_scatter_numeric(df, 'h3','hz','hz/h3',size=3,equalscale=False,render=False)
    
    
    # fig
    # fig = plot_2d_scatter_numeric(df, 'abs_om_dot','hz','i',size=2,equalscale=False,render=False)
    # fig.update_yaxes(type="log") # log range: 10^0=1, 10^5=100000
    # fig.update_xaxes(type="log") # log range: 10^0=1, 10^5=100000
    
    # fig.show()
    plotly.offline.plot(fig)
    
    # plt.hist(df.hz)
    
    
    return