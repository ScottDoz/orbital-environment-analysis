# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 22:00:27 2026

@author: scott

Delta-V Network
---------------

Generate a network graph of the current space catalog, with edges connecting
all transfers with delta-V below a threshold value.
Use dV_nodal as a proxy for actual delta-V.

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
import itertools

import plotly.graph_objects as go
import plotly
import plotly.express as px

import pdb

from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.Distances import *
from OrbitalAnalysis.Visualization import *
from OrbitalAnalysis.Clustering import *

# from sr_tools.Astrodynamics.OrbitToOrbit import *
from OrbitalAnalysis.OrbitToOrbit import *
from OrbitalAnalysis.Distances import *

#%% Loat Data

# Load data
df = load_satellites(group='all',compute_params=True,compute_pca=True)
# df = load_2019_experiment_data([36]) # New dataset

# Limit size for testing
# df = df.head(10000)


#%% Ball Tree
# Use dist_dH_atx as iniital approx for distance.
# Build a BallTree and query to find all neighbours within threshold dV ()


# # Extract positions for dH_nodal metric
# X = df[['a','e','i','om','w']].to_numpy()
# X[:,2:] = np.deg2rad(X[:,2:]) # Convert angles to radians

# Extract positions for dH_atx
X = df[['hx','hy','hz','a']].to_numpy()

# Limit size
# N = 5000 
# X = X[:,:]

# Build the Ball tree
# leaf_size: Higher values make construction faster but query slower
t0 = time.perf_counter()
# tree = BallTree(X, leaf_size=40, metric=dist_dH_r_nodal)
tree = BallTree(X, leaf_size=10, metric=dist_dH_atx)
print("BallTree build time = {} s".format(str(time.perf_counter() - t0)) )

# #  Query the tree for 5 nearest neighbors of the first 3 points
# queries = X[:3]
# distances, indices = tree.query(queries, k=3)

# Query BallTree. Find all points within radius km/s
# Note: initial query is 2*radius to make sure we dont miss any
radius = 0.1 # 2.0
t0 = time.perf_counter()
# indices = tree.query_radius(X[:1], r=radius)
indices, dist = tree.query_radius(X, r=radius*2, return_distance=True)
print("BallTree query time = {} s".format(str(time.perf_counter() - t0)) )

# Count connections
# print(f"Indices within radius {radius}:", indices)
# Connections distributions
n_neigh = np.array([ len(sub_arr) for sub_arr in dist ]) # Number of neighbours for each object
n_edges = sum(n_neigh) # Total number of edges


# All-pairs query time
# leaf_size | r=2km/s | r=3km/s
#    5      |   308   | 
#   10      |   279   | 467.6 
#   15      |   322   |

#          dH_nodal                |
#     N | Build Time | Query time  | Build  | Query    
#  100  |    0.35    | 0.15        | 0.0035 | 0.00056
#  1000 |    8.74    | 1.15        | 0.017  | 0.00055
#  2000 |   21.95    | 2.99        | 0.001  | 0.00048
#  5000 |   64.44    | 6.95        | 0.14   | 0.003
#  all                             | 0.370  | 0.0075    

# Query all pairs
# indices = tree.query_radius(X, r=radius, return_distance=False)

#%% Process Edges

# Get source and target ids
target_list = [item for sublist in indices for item in sublist]
source_list = [item for i, sublist in enumerate(indices) for item in itertools.repeat(i, times=len(sublist))]
dfedges = pd.DataFrame(columns=['source','target','dist','dH_nodal'])
dfedges.source = source_list
dfedges.target = target_list
dfedges.dist = [item for sublist in dist for item in sublist]
# Drop self-edges and update
dfedges = dfedges[~(dfedges['source'] == dfedges['target'])] # Drop self-edges
source_list = dfedges.source.to_list()
target_list = dfedges.target.to_list()

# Compute Distance Metrics for each edge
# Nodal Transfer distance (better approx)
x1 = df[['a','e','i','om','w']].loc[source_list].to_numpy()
x1[:,2:] = np.deg2rad(x1[:,2:]) # Convert angles to radians
x2 = df[['a','e','i','om','w']].loc[target_list].to_numpy()
x2[:,2:] = np.deg2rad(x2[:,2:]) # Convert angles to radians
t0 = time.perf_counter()
dfedges['dH_nodal'] = dist_dH_r_nodal(x1,x2)
print("dH_nodal comp time = {} s".format(str(time.perf_counter() - t0)) )

# # Plot dH_atx vs dH_nodal to see approx
# fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
# ax.plot(dfedges.dist, dfedges.dH_nodal,'.k')
# ax.plot([0,radius],[0,radius],'-r')
# ax.set_ylabel(r'$dH_nodal (km/s)/s)$',fontsize=16)
# ax.set_xlabel(r'${\Delta}H/atx (km/s)$',fontsize=16)

# Limit edge list to dH_nodal<radius
dfedges = dfedges[dfedges.dH_nodal<=radius]
source_list = list(dfedges.source)
target_list = list(dfedges.target)
edge_list = list(zip(source_list, target_list))
# from_norad = df['NoradId'].loc[source_list].to_list()
# to_norad = df['NoradId'].loc[target_list].to_list()

#%% Add Actual Delta-V

dfres1 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.1_full.csv")
dfres2 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.2_full.csv")

# # Load dV from procesed results
dfres1a = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt1.csv")
dfres2a = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt2.csv")
# dfres3 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.2_pt3.csv")
# dfres4 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.2_pt4.csv")
# dfres5 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.1_pt5.csv")
# # Combine
dfres = pd.concat([dfres1, dfres2, dfres1a, dfres2a], ignore_index=True)
# dfres = dfres.drop(columns=['index'])
# dfres.to_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.2_full.csv",index=False)

# # Compute Distance Metrics for each edge
# # Nodal Transfer distance (better approx)
# x1 = dfres[['a1','e1','i1','om1','w1']].to_numpy()
# # x1[:,2:] = np.deg2rad(x1[:,2:]) # Convert angles to radians
# x2 = dfres[['a1','e1','i1','om1','w1']].to_numpy()
# # x2[:,2:] = np.deg2rad(x2[:,2:]) # Convert angles to radians
# t0 = time.perf_counter()
# dfres['dH_nodal'] = dist_dH_r_nodal(x1,x2)
# print("dH_nodal comp time = {} s".format(str(time.perf_counter() - t0)) )

# dfres = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_1-to-N_from_norad_25730.csv"); dfres.insert(0,'from_norad',25730)

# # # merge dV to dfedges
# dfres = pd.merge(dfedges, dfres[['from_norad','to_norad','dV']], left_on=['source','target'], right_on=['from_norad', 'to_norad'], how='left').drop(columns=['from_norad','to_norad'])
# # # Drop null dVs
# dfres = dfres[~pd.isna(dfres.dV)]

# Plot 
fig, ax = plt.subplots(1,1,figsize=(12, 8)) 
ax.plot(dfres.dV, dfres.dH_nodal,'.k', markersize=0.5)
ax.plot([0,radius],[0,radius],'-r')
ax.set_ylabel(r'$dH nodal (km/s)/s)$',fontsize=16)
ax.set_xlabel(r'${\Delta}V (km/s)$ (optimal)',fontsize=16)


#%% Create a graph
G = nx.Graph()
G.add_nodes_from(range(len(df)))
G.add_edges_from(edge_list)
print("Graph size")
print("N = {} nodes".format(len(df)))
print("E = {} edges with dH_atx <= {} km/s".format(len(dfedges), radius))

# Separate into connected components. This is a generator expression
component_subgraphs = (G.subgraph(c) for c in nx.connected_components(G))
# If you need a list of actual Graph objects with copied data:
# component_graphs_list = [G.subgraph(c).copy() for c in nx.connected_components(G)]

# # 3. Iterate and use the subgraphs
# for i, subgraph in enumerate(component_subgraphs):
#     n_sub_nodes = subgraph.number_of_nodes()
#     if n_sub_nodes > 10:
#         print(f"Subgraph {i+1} has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
#     #print(f"Nodes: {list(subgraph.nodes)}")
#     # You can also perform further analysis or drawing on each subgraph
#     # nx.draw(subgraph, with_labels=True)
#     # plt.show()


#%% Visualize the network (too big)

# Copy dataframe to apply filtering
dfedges_lim = dfedges.copy()

# Remove duplicate edges, only consider one per pair
# see: https://stackoverflow.com/questions/44792969/pandas-drop-duplicates-based-on-subset-where-order-doesnt-matter
df1 = pd.DataFrame(np.sort(dfedges_lim[['source','target']].values, axis=1), index=dfedges_lim[['source','target']].index, columns=dfedges_lim[['source','target']].columns)
dfedges_lim = dfedges_lim[~df1.duplicated()]
del df1

# Limit edge sizes
max_edges = 1000000 # Max number of edges to render in plotly
if len(dfedges)>max_edges:
    dfedges_lim = dfedges_lim.nlargest(max_edges, 'dH_nodal')
    
source_list_lim = list(dfedges_lim.source)
target_list_lim = list(dfedges_lim.target)
edge_list_lim = list(zip(source_list_lim, target_list_lim))


# Get edge coords
edge_x = np.column_stack([ df['hx'].loc[source_list_lim].to_numpy(), df['hx'].loc[target_list_lim].to_numpy(), np.full(len(source_list_lim), None)]).flatten()
edge_y = np.column_stack([ df['hy'].loc[source_list_lim].to_numpy(), df['hy'].loc[target_list_lim].to_numpy(), np.full(len(source_list_lim), None)]).flatten()
edge_z = np.column_stack([ df['hz'].loc[source_list_lim].to_numpy(), df['hz'].loc[target_list_lim].to_numpy(), np.full(len(source_list_lim), None)]).flatten()


# Create trace for nodes
scatter_trace = go.Scatter3d(
    x=df['hx'],
    y=df['hy'],
    z=df['hz'],
    hovertext= df['NoradId'].astype(str) + ' ' + df['Name'].astype(str), #  df['NoradId'] + df['Name'],
    mode='markers',
    marker=dict(size=0.5, color='blue'),
    name='Nodes',
)

# Add edges
edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    hoverinfo='none',
    line=dict(width=0.1, color='gray'),
    name='Edges',
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis')
    )
)

fig = go.Figure(data=[scatter_trace, edge_trace], layout=layout)

# Add legend title with graph details
fig.update_layout(
    legend=dict(
        title=dict(
            # text="<u>Delta-V Network</u><br>ΔV <= {} km/s <br>Nodes: {} <br>Edges: {} <br>Drawn edges: {} <br>".format(radius, len(G.nodes), len(G.edges), len(source_list_lim)) # Use <br> for a new line
            text=f"<u>Delta-V Network</u><br>ΔV <= {radius} km/s <br>Nodes: {len(G.nodes):,} <br>Edges: {len(G.edges):,} <br>Drawn edges: {len(source_list_lim):,} <br>"
        )
    )
)

plotly.offline.plot(fig)


#%% Other tests (long compute times)

#%% Pdist

# # Compute distance matrix
# # print()
# t0 = time.perf_counter()
# # Scipy
# # condensed_distances = pdist(X, metric=dist_dH_r_nodal, n_jobs=-1) # Condensed distance matrix
# # distance_matrix = squareform(condensed_distances) # Convert to NxN distance matrix
# # distance_matrix = pairwise_distances(X = X, metric = dist_dH_r_nodal, n_jobs = 1)
# distance_matrix = pairwise_distances(X = X, metric = dist_dH_atx, n_jobs = 1)

# np.fill_diagonal(distance_matrix, 0) # Set the diagonal elements to zero
# print("Runtime = {} s".format(str(time.perf_counter() - t0)) )

# # Runtime estimates                     dH_atx
# #     N | T(cores=1) | T(cores=-1)
# #   100 |    8.676   |  20.9            0.025
# #  1000 |            |                  2.28
# # 10000 |            |                  231.9


#%% Loop method
# ~0.2s each iteration -> 95 mins total

# for i, x in tqdm(enumerate(X[:1000,:])):
    
#     # Compute distances
#     X1 = np.tile(x,(len(df),1))
#     di = dist_dH_r_nodal(X1,X)
    
#     # End loop

#%% Stacked method

# Repeat the entire catalog 100, 1000 x and stack together
# X1    X2
# norad id
#    0      0  
#    0      1
#    ...    ...
# N = 100
# X1 = np.repeat(X[:N,:], len(X), axis=0)
# X2 = np.tile(X, (N, 1))
# t0 = time.perf_counter()
# D = dist_dH_r_nodal(X1,X2)
# print("Runtime = {} s".format(str(time.perf_counter() - t0)) )

# N   Time
# 1   0.4
# 10  4.8
# 100 62