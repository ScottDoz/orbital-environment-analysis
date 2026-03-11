# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:36:37 2023

@author: scott

Investigate if Distance Metrics correlate with Delta-V.

Use X4_Compute_DeltaVs.py to compute delta-Vs from certain satellites.



To look into:
    
Google searches:
    pairwise distances approximate distance from points using regression
    
    
Mantle test: https://en.wikipedia.org/wiki/Mantel_test
- compare two distance matrices

Embeddings
    Multidimensional scaling
    Simulated annealing

Regression:
    Quantile Regression for Distances
    Non-parametric regression




"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import pdb

from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.Distances import *
from OrbitalAnalysis.Visualization import *
from OrbitalAnalysis.Clustering import *

# from sr_tools.Astrodynamics.OrbitToOrbit import *
from OrbitalAnalysis.OrbitToOrbit import *




#%% Select Method

# 3 methods
# 1-to-N - From a single target object to all other objects.
# N-to-N - All pairs of N objects surrounding a single point.
# load - Load pre-computed values

# mode = '1-to-N'
# mode = 'N-to-N'
mode = 'load'

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

if mode == 'load':
    # Load pre-calculated data
    
    # Select target
    # Select target
    # target = 25544 # ISS
    target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    # target = 49445 # Atlas 5 Centaur DEB
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    
    # Compute distance metrics
    df1 = compute_distances(df,target,searchfield='NoradId')
    df1 = df1.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    
    # Get data directory
    DATA_DIR = get_data_home()
    _dir = DATA_DIR/'Delta-Vs' # Save directory
    
    # Load in delta-Vs
    # dfnear = pd.read_csv(str(_dir/'deltaVs_{}.csv'.format(target)))
    dfnear = pd.read_csv(str(_dir/'deltaVs_1-to-N_from_norad_{}.csv'.format(target)))
    # dfnear = pd.read_csv(str(_dir/'deltaVs_N-to-N_kNN_100_around_norad_22675.csv'.format(target)))
    
    # Add column for target
    # dfnear.insert(loc=0, column='from_norad', value=target)
    
    # Merge in delta-Vs
    df1 = pd.merge(df1,dfnear,how='left',left_on='NoradId',right_on='to_norad')
    
    # TODO:
    # # Extract list of norad_ids
    # norad_list = dfnear.NoradId.tolist()
    
    # # Create pair-wise combinations of these norads
    # # pairs = list(itertools.combinations(norad_list, 2))
    # pairs = list(itertools.permutations(norad_list, 2)) # Order is important
    # # Create new dataframe
    # df1 = pd.DataFrame(data=pairs,columns=['from_norad','to_norad'])
    # # Merge data for 'from_norad' node
    # df1 = pd.merge(df1,dfnear[['NoradId','a','e','i','om','w']],how='left',left_on='from_norad',right_on='NoradId')
    # df1 = df1.rename(columns = {'a':'a1','e':'e1','i':'i1','om':'om1','w':'w1'})
    # # Merge data for 'to_norad' node
    # df1 = pd.merge(df1,dfnear[['NoradId','a','e','i','om','w']],how='left',left_on='to_norad',right_on='NoradId')
    # df1 = df1.rename(columns = {'a':'a2','e':'e2','i':'i2','om':'om2','w':'w2'})
    
    
    
    
    # dfnear = compute_distances(dfnear,target,searchfield='NoradId')
    # dfnear = dfnear.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename


elif mode == '1-to-N':
    
    # create_arg_list()
    
    
    # 1-to-N
    # Compute delta-Vs from a single target to all other objects
    # Create dataframe that contains orb1 and orb1 for each row

    # Select target
    # target = 25544 # ISS
    target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    # target = 49445 # Atlas 5 Centaur DEB
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    # Compute distance metrics
    df = compute_distances(df,target,searchfield='NoradId')
    df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    # 1. Closest N objects (using cylindrical distance)
    # N = 1000 # Number of objects
    # dfnear = df.nsmallest(N, 'd_cyl') # Nearest
    # dfnear = df.sample(n=N) # Random sample
    dfnear = df.copy() # All objects
    dfnear = dfnear.head(1000) # Limit to 100
    
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
                     'd_cyl','dH_atx','dV']]
    
    # pdb.set_trace()

    # Compute delta-Vs
    # dfnear = compute_dVs_paralell() # ~ 6:19 seconds for 1000 objects
    dfnear = compute_dVs_loop() # ~ 3:53 for 1000 objects
    print(dfnear)
    
    # Save data
    df1 = dfnear[['from_norad','to_norad','d_cyl','dH_atx','dV']]
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
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    
    # Compute distance metrics
    df = compute_distances(df,target,searchfield='NoradId')
    df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    # Closest N objects (using cylindrical distance)
    N = 200 # Number of objects
    dfnear = df.nsmallest(N, 'd_cyl') # Nearest
    
    # Repeat dataframe N times
    dfs = pd.concat([dfnear]*N)
    
    pdb.set_trace()
    

    
    

#%% Plot

# # # dV vs d_cyl
# fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ax.plot(df1.d_cyl,df1.dV,'.k')
# ax.set_xlabel(r'$d_{cyl}$ (Cylindrical) (km$^{2}$/s)',fontsize=16)
# ax.set_ylabel(r'${\Delta}V (km/s)$',fontsize=16)

# fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ax.plot(dfnear.p5,dfnear.dV,'.k')
# ax.set_xlabel(r'$d_{euc}$ (Euclidean) (km$^{2}$/s)',fontsize=16)
# ax.set_ylabel(r'${\Delta}V (km/s)$',fontsize=16)

# dV vs dH_nodal
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ind = abs(df.i - iT) > 20
ax.plot(df1.dH_nodal,df1.dV,'.k')
# ax.plot(df1.dH_nodal[ind],df1.dV[ind],'.r') # large transfers
# ax.plot(df1.dH_nodal[ind],df1.dV[ind],'.r') # large transfers
ax.plot([0,df1.dV.max()],[0,df1.dV.max()],'-r')
ax.set_xlabel(r'$dH nodal (km/s)$',fontsize=16)
ax.set_ylabel(r'${\Delta}V (km/s)$ (Optimal delta-V)',fontsize=16)

# TODO: Check cases where dH_nodal is much less than dV
# For these orbits, the OrbitToOrbit problem did not find the optimal solution???
# df1[['dH_nodal','dV']][ df1.dH_nodal - df1.dV < -1.0 ]

# Plot dH/a - same shape as euclidean distance, but scaled to mag of dV
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ind = abs(df.i - iT) > 20
ax.plot(df1.dH_atx, df1.dV, '.k')
ax.plot([4,4],[0,12],'-r')
ax.plot([0,12],[4,4],'-r')
# ax.plot([0,df1.dV.max()],[0,df1.dV.max()],'-r')
ax.set_ylabel(r'${\Delta}V (km/s) $ (Optimal delta-V)',fontsize=16)
ax.set_xlabel(r'${\Delta}H/atx (km/s)$ (Euclidean distance in H scaled by orbit radius)',fontsize=16)

# Plot euclidean distance
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ind = abs(df.i - iT) > 20
ax.plot(df1.d_Euc, df1.dH_nodal, '.k')
# ax.plot([0,df1.dV.max()],[0,df1.dV.max()],'-r')
ax.set_ylabel(r'$dH nodal (km/s)/s)$',fontsize=16)
ax.set_xlabel(r'${\Delta}V (Euc) (km/s)$',fontsize=16)


#%% Plot error vs other properties

# Extract orbital elements of target
aT = df[df.NoradId == target]['a'].iloc[0]
eT = df[df.NoradId == target]['e'].iloc[0]
iT = df[df.NoradId == target]['i'].iloc[0]
omT = df[df.NoradId == target]['om'].iloc[0]
wT = df[df.NoradId == target]['w'].iloc[0]
qT = df[df.NoradId == target]['q'].iloc[0]
QT = df[df.NoradId == target]['Q'].iloc[0]
hT = df[df.NoradId == target]['h'].iloc[0]
hrT = df[df.NoradId == target]['hr'].iloc[0]
hthetaT = df[df.NoradId == target]['htheta'].iloc[0]
hzT = df[df.NoradId == target]['hz'].iloc[0]


# Identify outliers with large error
# ind = abs(df1.dH_atx - df1.dV)/df1.dV > 0.2 # Error > 1.5 km/s
ind = abs(df.i - iT) > 20
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ax.plot(df1['dH_atx'][~ind],df1['dV'][~ind],'.k') # Good values   abs(df1['dH_atx'] - df1['dV'])/df1['dV']
# ax.plot(df1['dH_atx'][ind],df1['dV'][ind],'.r')   # Outliers
# ax.plot(df1['i'][ind] - iT,100*abs(df1['dH_nodal'][ind] - df1['dV'][ind])/df1['dV'][ind],'.r')
# ax.plot(df1['i'][~ind] - iT,100*abs(df1['dH_nodal'][~ind] - df1['dV'][~ind])/df1['dV'][~ind],'.k')

ax.plot(df1['dphi'][ind] - iT,100*abs(df1['dH_nodal'][ind] - df1['dV'][ind])/df1['dV'][ind],'.r')
ax.plot(df1['dphi'][~ind] - iT,100*abs(df1['dH_nodal'][~ind] - df1['dV'][~ind])/df1['dV'][~ind],'.k')

ax.set_xlabel(r'$dphi$',fontsize=16)
ax.set_ylabel(r'Error (%)',fontsize=16)

# Compare the means and variances of the two groups
# h, hr, htheta, hphi, hz


#%%

# 1. Create dummy data for two levels (e.g., 'Group 1', 'Group 2') 
# and two categories within each (e.g., 'A', 'B')
data_A_group1 = df1['h'][ind]
data_B_group1 = df1['h'][~ind]
data_A_group2 = df1['q'][ind]
data_B_group2 = df1['q'][~ind]

# Combine the data for each level
level1_data = [data_A_group1, data_B_group1]
level2_data = [data_A_group2, data_B_group2]

# 2. Define the positions and width for the box plots
positions1 = [1, 2] # Positions for Group 1 data
positions2 = [4, 5] # Positions for Group 2 data
width = 0.35 # Adjust width for better visualization

fig, ax = plt.subplots()

# 3. Plot the data for each level, specifying positions and widths
ax.boxplot(level1_data, positions=[p - width/2 for p in positions1], widths=width, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax.boxplot(level2_data, positions=[p + width/2 for p in positions2], widths=width, patch_artist=True, boxprops=dict(facecolor='lightgreen'))

# 4. Set x-axis ticks and labels
ax.set_xticks([1.5, 4.5]) # Center the labels between the two boxes of each level
ax.set_xticklabels(['h', 'hz'])
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Box Plots at Each Level')

# 5. Add a legend (optional, requires handling artist handles)
# For simplicity, a basic approach is shown above with colors

plt.show()

#%% 2D scatter

# dfnear = dfnear.rename(columns = {'a2':'a','e2':'e','i2':'i','om2':'om','w2':'w'})
# fig = plot_2d_scatter_numeric(dfnear,'d_cyl','dV',color='i',size=3,logColor=True)

# dfnear = df1.rename(columns = {'a2':'a','e2':'e','i2':'i','om2':'om','w2':'w'})
# fig = plot_2d_scatter_numeric(df1,'dH_atx','dV',color='i',size=3,logColor=True)
