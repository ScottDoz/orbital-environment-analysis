# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:36:37 2023

@author: scott

Investigate if Distance Metrics correlate with Delta-V

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import pdb

from SatelliteData import *
from DistanceAnalysis import *
from Visualization import *
from Clustering import *

# from sr_tools.Astrodynamics.OrbitToOrbit import *
from OrbitToOrbit import *

# Load data
df = load_satellites(group='all',compute_params=True,compute_pca=True)
# df = load_2019_experiment_data([36]) # New dataset


#%% Select Method

# 3 methods
# 1-to-N - From a single target object to all other objects.
# N-to-N - All pairs of N objects surrounding a single point.
# load - Load pre-computed values

mode = '1-to-N'
# mode = 'load'

#%% Density analysis and Sampling

# Generate clusters in hz

# # Generate clusters in (h,hz) coordiantes
# label = 'test_clusters' # Field name holding clusters
# features = ['hz']   # Fields to use in clustering 
# df = generate_Kmeans_clusters(df,label,features,n_clusters=20,random_state=170)

# # Generate plotly figure and render in browser
# #plot_h_space_cat(df,'test_clusters')
# #fig = plot_2d_scatter_numeric(dfnear,'d_cyl','dV',color='i',size=3,logColor=True)

# fig = plot_2d_scatter_cat(df,'htheta','hz','test_clusters', size=2,
#                         aspectmode='auto',
#                         filename='temp-plot.html')



#%% Compute Orbit-to-Orbit Delta-V
# Two methods - loop or parallel

# Compute deltaVs
mu = 398600 # Gravitational parameter of Earth (km^3/s^2)

# Define a function to apply to each row of the dataframe
def my_function(row):
    ''' 
    Main function executed by each processor for task 1. 
    '''
    
    # Extract values from row  
    orb1 = {'a':row.a1,'e':row.e1,'w':np.deg2rad(row.w1),
        'i':np.deg2rad(row.i1),'om':np.deg2rad(row.om1)}
    orb2 = {'a':row.a2,'e':row.e2,'w':np.deg2rad(row.w2),
        'i':np.deg2rad(row.i2),'om':np.deg2rad(row.om2)}
    # Solve Orbit to Orbit problem
    result = solve_OrbitToOrbit_mccue(orb1,orb2,mu,solver='Grid')
    dV = result.fun
    
    return dV

def compute_dVs_paralell():
    '''
    Compute the delta-Vs using Dask to parallelize the computations.
    '''
    
    import timeit
    start_time = timeit.default_timer()
    
    # Convert the Pandas DataFrame to a Dask DataFrame
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    
    # Create dask dataframe
    ddf = dd.from_pandas(dfnear,  npartitions=7*3)
    
    # Define the metadata for the output column 'dV'
    meta = pd.Series(dtype=np.float64)
    
    with ProgressBar():
        # Apply the sum_cols function to the Dask DataFrame using the apply() method
        ddf['dV'] = ddf.apply(my_function, axis=1, meta=meta)
        # Convert the Dask DataFrame back to a Pandas DataFrame
        result = ddf.compute()
    
    elapsed = timeit.default_timer() - start_time
    print('Runtime: {} s'.format(str(elapsed)))
    
    return result

def compute_dVs_loop():
    '''
    Compute the delta-Vs in a standard for-loop.
    '''
    dVs = np.zeros(len(dfnear)) # Instantiate array to hold
    for ind, row in tqdm(dfnear.iterrows(), total=dfnear.shape[0]):
        dV = my_function(row)
        dVs[ind] = dV

    # Add to dataframe
    dfnear['dV'] = dVs
    
    return dfnear
    


#%% Select Mode

if mode == '1-to-N':
    
    creat_arg_list()
    
    
    # 1-to-N
    # Compute delta-Vs from a single target to all other objects
    # Create dataframe that contains orb1 and orb1 for each row

    # Select target
    # target = 25544 # ISS
    # target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    target = 49445 # Atlas 5 Centaur DEB
    
    # Compute distance metrics
    df = compute_distances(df,target,searchfield='NoradId')
    df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    # 1. Closest N objects (using cylindrical distance)
    # N = 1000 # Number of objects
    # dfnear = df.nsmallest(N, 'd_cyl') # Nearest
    # dfnear = df.sample(n=N) # Random sample
    dfnear = df.copy() # All objects
    # dfnear = dfnear.head(100)
    
    # Add columns Point 1
    dfnear['from_norad'] = target
    dfnear['a1'] = df['a'][df.NoradId == target].iloc[0]
    dfnear['e1'] = df['e'][df.NoradId == target].iloc[0]
    dfnear['i1'] = df['i'][df.NoradId == target].iloc[0]
    dfnear['om1'] = df['om'][df.NoradId == target].iloc[0]
    dfnear['w1'] = df['w'][df.NoradId == target].iloc[0]
    
    
    # Rename Point 2
    dfnear['to_norad'] = dfnear.NoradId
    dfnear = dfnear.rename(columns = {'a':'a2','e':'e2','i':'i2','om':'om2','w':'w2'})
    dfnear['dV'] = np.nan # Empty array for outputs
    
    # Re-order columns
    dfnear = dfnear[['from_norad','to_norad',
                     'a1', 'e1', 'i1', 'om1', 'w1',
                     'a2', 'e2', 'i2', 'om2', 'w2','h','hx','hy','hz','Name',
                     'd_Euc', 'dphi', 'D1', 'p1', 'p2', 'p3', 'p4',
                     'p5', 'zappala', 'mnid', 'Edel', 'dHr', 'dHz', 'dHtheta', 'd_arc',
                     'd_cyl','dV']]
    
    pdb.set_trace()

    # Compute delta-Vs
    # dfnear = compute_dVs_paralell() # ~ 6:19 seconds for 1000 objects
    dfnear = compute_dVs_loop() # ~ 3:53 for 1000 objects
    print(dfnear)
    
    # Save data
    df1 = dfnear[['from_norad','to_norad','d_cyl','dV']]
    # df1.to_csv(str('deltaVs_{}.csv'.format(target)),index=False)


elif mode == 'N-to-N':
    # N-to-N
    # Compute N nearest neighbors to a target object
    # Compute delta-Vs between all pairs of objects
    
    # Select target
    # Select target
    # target = 25544 # ISS
    # target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    target = 49445 # Atlas 5 Centaur DEB
    
    
    # Compute distance metrics
    df = compute_distances(df,target,searchfield='NoradId')
    df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    # Closest N objects (using cylindrical distance)
    N = 200 # Number of objects
    dfnear = df.nsmallest(N, 'd_cyl') # Nearest
    
    # Repeat dataframe N times
    dfs = pd.concat([dfnear]*N)
    
elif mode == 'load':
    # Load pre-calculated data
    
    # Select target
    # Select target
    # target = 25544 # ISS
    # target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    target = 49445 # Atlas 5 Centaur DEB
    
    
    dfnear = pd.read_csv(str('deltaVs_{}.csv'.format(target)))
    
    # dfnear = compute_distances(dfnear,target,searchfield='NoradId')
    # dfnear = dfnear.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    

#%% Plot
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
ax.plot(dfnear.d_cyl,dfnear.dV,'.k')
ax.set_xlabel(r'$d_{cyl}$ (Cylindrical) (km$^{2}$/s)',fontsize=16)
ax.set_ylabel(r'${\Delta}V (km/s)$',fontsize=16)

# fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ax.plot(dfnear.p5,dfnear.dV,'.k')
# ax.set_xlabel(r'$d_{euc}$ (Euclidean) (km$^{2}$/s)',fontsize=16)
# ax.set_ylabel(r'${\Delta}V (km/s)$',fontsize=16)

#%% 2D scatter

dfnear = dfnear.rename(columns = {'a2':'a','e2':'e','i2':'i','om2':'om','w2':'w'})
fig = plot_2d_scatter_numeric(dfnear,'d_cyl','dV',color='i',size=3,logColor=True)
