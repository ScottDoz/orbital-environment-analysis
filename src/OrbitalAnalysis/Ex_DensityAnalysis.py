# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:09:26 2022

@author: mauri
"""

from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *
from sklearn import preprocessing
# from Overpass import *
# from Ephem import *
# from Events import *
# from GmatScenario import *


def density_analysis(x,y):

    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    X = df[['a','e','i','om','w','h','hx','hy','hz']].to_numpy()
    print (df.columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    params_norm = min_max_scaler.fit_transform(X)
    df[['a','e','i','om','w','h','hx','hy','hz']] = params_norm
    plot_kde(df,x,y,0.025,normalized=False)
    #0.1 works well
    
    return df