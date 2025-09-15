# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:14:14 2024

@author: scott

X9 Topologicial Data Analysis
-----------------------------

Perform topological data analysis on the space object catalog
* Generate rips complex from distances

"""

import numpy as np
import networkx as nx
import time

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
import plotly.express as px

import gudhi as gd

import pdb

from OrbitalAnalysis.Catalog import Catalog
from OrbitalAnalysis.Distances import *

#%% Load catalog

# Space Object Catalog
cat = Catalog.from_spacetrack()

# Get eci data
df = cat.get_eci_states()

# Drop nan
df.dropna(subset='x',inplace=True)
df.reset_index(drop=True, inplace=True)

#%% Compute pair-wise euclidean distance in angular momentum space
distance_matrix = compute_pairwise_euclidean_distance(df)

#%% Create rips complex graph, where dist < t

threshold = 100 # Threshold  # 500 works
# threshold | Edges
# 100 | 91,868
# 200 | 

# 500 -> 213,584 edges
# 600 -> 234,997 edges
# 700 -> 257,325 edges
# 800 -> 281,573 edges
# 1000 -> 334421 edges
# 1200 -> 391428 edges
# 1500 -> 483772 edges
# 1700 -> 559006 edges # No display

# Create an adjacency matrix based on the threshold
adjacency_matrix = (distance_matrix < threshold).astype(int)
# Remove self-edges by setting the diagonal to 0
np.fill_diagonal(adjacency_matrix, 0)

# Extract the indices of non-zero elements (edges)
i, j = np.nonzero(adjacency_matrix)

# Create a list of edges, including duplicates
edges = np.array([i, j]).T

# Filter out duplicate edges (i.e., keep only one direction)
# Ensure each edge is represented in the form (min, max)
unique_edges = np.array([np.sort(edge) for edge in edges])
edges = np.unique(unique_edges, axis=0)


# # Create edge list
# edges = np.transpose(np.nonzero(adjacency_matrix))

# Get indexes of edges
source_list = edges[:,0]#.tolist()
target_list = edges[:,1]#.tolist()
print("Number of edges: {}".format(len(edges)))


#%% Alpha complex method
# Faster.


# # First create simplex tree of points
# st = gd.SimplexTree()
# [st.insert(p) for p in points]
# pts = np.array([st.get_point(i) for i in range(st.num_vertices())])

# Get unique points for complex.
# Some satellites share orbital locations
points = df[['hx','hy','hz']].drop_duplicates(keep='first')

# Alpha Complex
# Compute the alpha complex. This 
alpha_complex = gd.AlphaComplex(points = points.to_numpy()/1000000)
# Create simplex tree
st_alpha = alpha_complex.create_simplex_tree()


# Create barcode
# pairs(dimension, pair(birth, death))
BarCodes_Alpha = st_alpha.persistence()
df_barcode = pd.DataFrame(BarCodes_Alpha, columns = ['dim','range'])
df_barcode[['birth','death']] = pd.DataFrame(df_barcode['range'].tolist()) # Split tupple
df_barcode['length'] = df_barcode['death'] - df_barcode['birth']

# Plot persistence diagram
gd.plot_persistence_diagram(df_barcode[['dim','range']][(df_barcode.dim>1) & (df_barcode.length>0.000)].to_numpy());
# Plot persistence barcode
gd.plot_persistence_barcode(df_barcode[['dim','range']][(df_barcode.dim>1) & (df_barcode.length>0.0001)].to_numpy());



# # Get specific barcode elemnt
# # See: https://gudhi.inria.fr/python/latest/_downloads/9c981118f013a71f406ee5f80cfe57bb/plot_alpha_complex.py
# dim, rng = df_barcode[['dim','range']].iloc[0]
# points = np.array([alpha_complex.get_point(i) for i in range(st_alpha.num_vertices())])
# triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1] >= rng[0] and s[1] <= rng[1] ])
# st_gen = st_alpha.get_filtration()



#%% Plot Alpha complex triangles

# FIXME: indexing of points is wrong.
# Elements should be connected

# Get just dim 1
df_dim1 = df_barcode[df_barcode.dim==1].sort_values(by='length',ascending=False)
dim, rng = df_dim1[['dim','range']].iloc[3]
threshold = rng[1] # Threshold  # 500 works

# Get triangles
pts = np.array([alpha_complex.get_point(i) for i in range(st_alpha.num_vertices())])
# triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1] <= threshold]) # 0 < filt <  threshold
# triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1]>=rng[0] and s[1] <= rng[1]]) # 0 < filt <  threshold
triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1] <= (rng[0]+rng[1])/2]) # Mean filtration value
edges = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 2 and s[1]>=rng[0] and s[1] <= rng[1]]) # 0 < filt <  threshold

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(points[:, 0], points[:, 1], points[:, 2],'.k')
# l = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles = triangles[:20,:])
# # l = ax.triplot(points[:, 0], points[:, 1], points[:, 2], triangles = triangles[:20,:])
# plt.show()



# First possibility: plotly
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Create trace for nodes
scatter_trace = go.Scatter3d(
    x=pts[:,0]*1000000,y=pts[:,1]*1000000,z=pts[:,2]*1000000, name='satellites',
    mode='markers',
    marker=dict(size=0.2, color='blue')
)
fig.add_trace(scatter_trace)

# Filtration element dim 2
dim, rng = df_barcode[['dim','range']].iloc[8]
triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1]>=rng[0] and s[1] <= rng[1]]) # 0 < filt <  threshold
dim2_trace = go.Mesh3d(
    x=pts[:,0]*1000000,y=pts[:,1]*1000000,z=pts[:,2]*1000000,
    i = triangles[:,0],j = triangles[:,1],k = triangles[:,2],
    name= str(dim) + "( " + str(rng) + str(")"),
)
fig.add_trace(dim2_trace)

# # Filtration edges
# source_list = edges[:,0]; target_list = edges[:,1]
# edge_x = np.column_stack([ pts[source_list,0]*1000000, pts[target_list,0]*1000000, np.full(len(source_list), None)]).flatten()
# edge_y = np.column_stack([ pts[source_list,1]*1000000, pts[target_list,1]*1000000, np.full(len(source_list), None)]).flatten()
# edge_z = np.column_stack([ pts[source_list,2]*1000000, pts[target_list,2]*1000000, np.full(len(source_list), None)]).flatten()
# edge_trace = go.Scatter3d(
#     x=edge_x,y=edge_y,z=edge_z,name= str(dim) + " " + str(rng),
#     mode='lines',
#     line=dict(width=2.0, color='gray')
# )
# fig.add_trace(edge_trace)

# Update ranges
fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
        yaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
        zaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
        aspectmode = 'cube',
        ),
    # width=700,
    # margin=dict(r=20, l=10, b=10, t=10)
    )

plotly.offline.plot(fig, filename='filtration.html')



# # Second possibility: matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=triangles)
# plt.show()



#%% Persistance with ripser_parallel (still slow)

# from gph.python import ripser_parallel

# t0 = time.time()
# dgm = ripser_parallel(distance_matrix/1000, metric="precomputed", maxdim=2, n_threads=-1,collapse_edges=True)
# print("Ripser parallel time: {} s".format(time.time() - t0),flush=True)


#%% Persistace with GUDHI

# # See: https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb
# 

# scale = 1000000


# # Set maximum distance for filtration
# alpha = 1000/scale
# print("alpha = {}".format(alpha),flush=True)

# # 1-skeleton
# # Create a 1-skeleton of the dataset 
# # Collection of vertices and edges with distance < α
# t0 = time.time()
# skeleton = gd.RipsComplex(distance_matrix = distance_matrix/scale, max_edge_length = alpha) 
# print("1-skeleton creation time: {} s".format(time.time() - t0),flush=True)

# # Rips filtration
# # Create Rips simplex tree from the 1-skeleton
# # Use max dimension 2 (veritces, edges, and triangles)
# t0 = time.time()
# Rips_simplex_tree = skeleton.create_simplex_tree(max_dimension = 2)
# print("Rips simplex tree creating time: {} s".format(time.time() - t0),flush=True)

# # Compute persistance diagram
# # https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-persistence-diagrams.ipynb

# t0 = time.time()
# BarCodes_Rips = Rips_simplex_tree.persistence()
# print("Rips persistence time: {} s".format(time.time() - t0),flush=True)
# # # List the first 20
# # for i in range(20):
# #     print(BarCodes_Rips[i])
# # Plot persistance diagram
# gd.plot_persistence_diagram(BarCodes_Rips);



# alpha | Number of simplices
# ---------------------------
# 100   | 2,499,216
# 1000  | 12,377,337

# # Rips filtration (not needed for persistence diagram)
# t0 = time.time()
# rips_filtration = Rips_simplex_tree.get_filtration()
# rips_list = list(rips_filtration)
# print("Rips filtration time: {} s".format(time.time() - t0),flush=True)
# len(rips_list)





#%% Persistance Diagram scikit-tda (Takes too long)

# import ripser
# import persim

# def diagram_sizes(dgms):
#     return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])

# r = ripser.ripser(distance_matrix, distance_matrix=True, thresh = 1000)['dgms']
# dgms = r['dgms']
# persim.plot_diagrams(
#     dgm_noisy, show=True,
#     title=f"Satellite Catalog\n{diagram_sizes(dgms)}"
# )



#%% Plot graph

# # Create trace for nodes
# scatter_trace = go.Scatter3d(
#     x=df['hx'],
#     y=df['hy'],
#     z=df['hz'],
#     mode='markers',
#     marker=dict(size=0.2, color='blue')
# )

# # Create edge trace
# edge_x = np.column_stack([ df['hx'][source_list].to_numpy(), df['hx'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()
# edge_y = np.column_stack([ df['hy'][source_list].to_numpy(), df['hy'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()
# edge_z = np.column_stack([ df['hz'][source_list].to_numpy(), df['hz'][target_list].to_numpy(), np.full(len(source_list), None)]).flatten()

# edge_trace = go.Scatter3d(
#     x=edge_x,
#     y=edge_y,
#     z=edge_z,
#     mode='lines',
#     line=dict(width=2.0, color='gray')
# )

       
# layout = go.Layout(
#     scene=dict(
#         xaxis=dict(title='X-axis'),
#         yaxis=dict(title='Y-axis'),
#         zaxis=dict(title='Z-axis')
#     )
# )

# fig = go.Figure(data=[edge_trace], layout=layout)

# plotly.offline.plot(fig, filename='rips-complex.html')



# # #%% Graph 1
# # G = nx.from_numpy_array(adjacency_matrix)
# # G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops (diagonal elements in the adjacency matrix)
# # # del distance_matrix, adjacency_matrix # Delete to free memory
# # N_edges = len(G.edges)
# # N_nodes = len(G.nodes)

# # # nx.draw_shell(G, with_labels=False, node_size=1)
# # # plt.show()

#%% Plot alpha complex with slider for filtration
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-alpha-complex-visualization.ipynb

# import ipywidgets as widgets

# plotly.offline.init_notebook_mode()
# from plotly.offline import iplot

# import plotly.io as pio
# pio.renderers.default='browser'

# alpha = widgets.FloatSlider(
#     value = 0.05,
#     min = 0.0,
#     max = 16,
#     step = 0.5,
#     description = 'Alpha:', 
#     readout_format = '.4f'
# )

# # Initialize mesh
# triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1]>=rng[0] and s[1] <= 0.05 ]) # Mean filtration value
# mesh = go.Mesh3d(
#     x = pts[:, 0], 
#     y = pts[:, 1], 
#     z = pts[:, 2], 
#     i = triangles[:, 0], 
#     j = triangles[:, 1], 
#     k = triangles[:, 2]
# )

# fig = go.FigureWidget(
#     data = mesh, 
#     layout = go.Layout(
#         title = dict(
#             text = 'Alpha Complex Representation of the catalog'
#         ), 
#         # scene = dict(
#         #     xaxis = dict(nticks = 4, range = [-1.5, 1.5]), 
#         #     yaxis = dict(nticks = 4, range = [-1.5, 1.5]), 
#         #     zaxis = dict(nticks = 4, range = [-1.5, 1.5])
#         # )
#     )
# )


# def view_torus(alpha):
#     if alpha < 0.0015:
#         alpha = 0.0015
#     triangles = np.array([s[0] for s in st_alpha.get_skeleton(2) if len(s[0]) == 3 and s[1] <= alpha])
#     fig.data[0].i = triangles[:, 0]
#     fig.data[0].j = triangles[:, 1]
#     fig.data[0].k = triangles[:, 2]
#     iplot(fig)

# widgets.interact(view_torus, alpha = alpha);
# # plotly.offline.plot(fig, filename='filtration.html')

