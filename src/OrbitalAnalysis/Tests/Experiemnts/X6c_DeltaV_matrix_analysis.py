# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:30:54 2026

@author: scott

Delta-V Matrix Analysis
-----------------------

Analyse the results of Delta-Vs computed from the BallTree method.
Compute correlation between distance metrics


"""
import sys
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

from scipy.spatial.distance import squareform
from scipy.stats import f_oneway

import mantel

#FIXME: permanently add to python path
sys.path.append(r'C:\Users\scott\Documents\Repos\orbital-environment-analysis\src')
sys.path.append(r'C:\Users\scott\Documents\Repos\astrologistics')

from OrbitalAnalysis.Distances import *

#%% Load data

def load_dV_data(compute_distances=True):
    ''' Load delta-V data from saved csv files '''
    
    # Complet sets
    dfres1 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.1_full.csv")
    dfres2 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.2_full.csv")

    # Partial sets (ongoing)
    dfres1a = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt1.csv")
    dfres2a = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt2.csv")
    dfres3a = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt3.csv")
    # dfres4 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt4.csv")
    # dfres5 = pd.read_csv(r"C:\Users\scott\satellite_data\Data\Delta-Vs\deltaVs_BallTree_rad_0.3_pt5.csv")
    # # Combine
    df = pd.concat([dfres1, dfres2, dfres1a, dfres2a, dfres3a], ignore_index=True)
    
    
    # Compute Additional Parameters
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Code from compute_orbital_params() in SatelliteData
    
    # Semi-latus rectum, periapsis and apoapsis
    df['p1'] = df.a1*(1-df.e1**2) # Semi-latus rectum (km)
    df['p2'] = df.a2*(1-df.e2**2) # Semi-latus rectum (km)
    df['q1'] = df.a1*(1-df.e1) # Periapsis (km)
    df['q2'] = df.a2*(1-df.e2) # Periapsis (km)
    df['Q1'] = df.a1*(1+df.e1) # Apoapsis (km^2/s)
    df['Q2'] = df.a2*(1+df.e2) # Apoapsis (km^2/s)
    
    # Specific angular momentum
    df['h1'] = np.sqrt(mu*df.p1)
    df['h2'] = np.sqrt(mu*df.p2)
    
    # Angular momentum in spherical coordinates
    df['hr1'] = np.sqrt(df.hx1**2 + df.hy1**2)
    df['hr2'] = np.sqrt(df.hx2**2 + df.hy2**2)
    df['hphi1'] = np.arctan2(np.sqrt(df.hx1**2 + df.hy1**2),df.hz1) # Polar angle (from z axis)
    df['hphi2'] = np.arctan2(np.sqrt(df.hx2**2 + df.hy2**2),df.hz2) # Polar angle (from z axis)
    df['htheta1'] = np.arctan2(df.hy1,df.hx1) # Azimuth angle
    df['htheta2'] = np.arctan2(df.hy2,df.hx2) # Azimuth angle
    
    # Nodal Precession Rate due to J2 effect
    # OMdot = -(3/2)*J2*(Re/p)^2*n*cos(i)
    # where
    # p = a(1-e^2) is the semi-latus rectum
    # n = sqrt(mu/a^3) is the mean motion
    # i is the inclination
    # Re = radius of the Earth
    # J2 is the zonal harmonic coefficient
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    n1 = np.sqrt(mu/df.a1**3) # Mean motion
    n2 = np.sqrt(mu/df.a2**3) # Mean motion
    df['om_dot1'] = -(3./2.)*J2*((Re/df.p1)**2)*n1*np.cos(np.deg2rad(df.i1))
    df['om_dot2'] = -(3./2.)*J2*((Re/df.p2)**2)*n1*np.cos(np.deg2rad(df.i2))
    
    
    if compute_distances:
        # Compute additional distance metrics
        
        # dH_atx
        # Uses dH and velocities at node crossing
        x1 = df[['hx1','hy1','hz1','a1']].to_numpy()
        x2 = df[['hx2','hy2','hz2','a2']].to_numpy()
        df['dH_atx'] = dist_dH_atx(x1, x2)
        
        # dH Euclidean distance
        x1 = df[['hx1','hy1','hz1']].to_numpy()
        x2 = df[['hx2','hy2','hz2']].to_numpy()
        df['dH'] = dist_dH(x1,x2)
        
        # Plane change angle phi
        x1 = df[['hx1','hy1','hz1']].to_numpy()
        x2 = df[['hx2','hy2','hz2']].to_numpy()
        df['dphi'] = dist_planechange(x1,x2)
        
        # Edel
        x1 = df[['a1','e1','i1','om1','w1','hx1','hy1','hz1']].to_numpy()
        x1[:,2:5] = np.deg2rad(x1[:,2:5]) # Convert angles to radian
        x2 = df[['a2','e2','i2','om2','w2','hx2','hy2','hz2']].to_numpy()
        x2[:,2:5] = np.deg2rad(x2[:,2:5]) # Convert angles to radian
        df['Edel'] = dist_edel(x1,x2,astflag=False)
    
    
    return df

def create_graph(df):
    
    # Extract edge list
    sources = list(df.from_norad)
    targets = list(df.to_norad)
    weights = list(df.dV)
    weighted_edge_list = list(zip(sources, targets, weights))
    
    # # Create a list of 3-tuples: (u, v, attribute_dict)
    # edge_list_with_attrs = [
    #     (u, v, {'dV': w1, 'dH_nodal': w2, 'dist':w3})
    #     for u, v, w1, w2, w3 in zip(sources[:5], targets[:5], list(df.dV)[:5], list(df.dH_nodal)[:5], list(df.dist)[:5] )
    # ]
    
    edge_list_with_attributes = [
        (u, v, {'dV': w1, 'dH_nodal': w2, 'dH_atx':w3, 'dH':w4, 'dphi':w5, 'Edel':w6})
        for u, v, w1, w2, w3, w4, w5, w6 in zip(sources, targets, list(df.dV), list(df.dH_nodal), list(df.dH_atx), list(df.dH), list(df.dphi), list(df.Edel) )
    ]
    
    
    
    # Create Network graph
    G = nx.DiGraph()
    
    # Add the weighted edges to the graph
    # G.add_weighted_edges_from(weighted_edge_list)
    # G.add_weighted_edges_from(edge_list_with_attrs)
    G.add_edges_from(edge_list_with_attributes)

    # print("Graph size")
    # print("N = {} nodes".format(len(df)))
    # print("E = {} edges with dH_atx <= {} km/s".format(len(df), radius))
    
    
    return G

# Toy case for simple graph
# 1. Define the data
# edge_list_with_attributes = [
#     (1, 2, {'weight_1': 0.3, 'weight_2': 0.5}),
#     (2, 3, {'weight_1': 0.1, 'weight_2': 0.8}),
#     (3, 1, {'weight_1': 0.4, 'weight_2': 0.2}),
#     (1, 3, {'weight_1': 0.2, 'weight_2': 0.9}),
# ]

# edge_list_with_attributes = [
#     (u, v, {'dV': w1, 'dH_nodal': w2, 'dist':w3})
#     for u, v, w1, w2, w3 in zip(sources[:5], targets[:5], list(df.dV)[:5], list(df.dH_nodal)[:5], list(df.dist)[:5] )
# ]

# # 2. Create the Directed Graph (DiGraph)
# G = nx.DiGraph()
# G.add_edges_from(edge_list_with_attributes)

# adj_matrix_w1 = nx.to_pandas_adjacency(G, weight='dV')
# print("Adjacency Matrix (dV):")
# print(adj_matrix_w1)

# # Alternative: Using numpy (unlabeled)
# adj_matrix_np = nx.to_numpy_array(G, weight='dH_nodal')
# print("\nAdjacency Matrix (dH_nodal) NumPy:")
# print(adj_matrix_np)


#%% Analysis 1: Compare symetry in Distance Metrics

def ex1_distance_matrix_symetry(df,G):
    
    # Analysis: Compare symetry of delta-V distances 
    # Expect that entries D[i,j] == D[j,i]
    # Compute difference in values:
    # dD = Dij - Dij.T
    
    # Get adjacency matrix
    D = nx.to_pandas_adjacency(G, weight='dV', nonedge=np.nan).to_numpy()
    
    # Compute difference betweein Dij and Dji
    dD = D - D.T
    dD = np.triu(dD) # Only return upper triangle component
    dD[dD==0] = np.nan # Replace null entries with nan
    dD_vals = dD[~np.isnan(dD)] # Extract non null values to flat array
    # plt.hist(dD_vals,bins=100)
    # Result: fairly symetric. Mean = 4.651e-5 km/s, std =  0.0002, max = 0.06 km/s
    
    # Get stats
    print("\nSymetry of dV Distance Metric")
    print("-----------------------------")
    print(f"mean(|dVij-dVji|) = {np.mean(abs(dD_vals))} km/s, std(|dVij-dVji|) = {np.std(abs(dD_vals))} km/s")
    print("")
    
    
    return


def ex2_mandel_test_distances(df,G, weight_1 ='dH_atx', weight_2 = 'dV', perms=3):
    
    print("\nMandel Test comparing two distance matrices")
    print("-------------------------------------------")
    print(f"D1 = {weight_1}, D2 = {weight_2}")
    
    # Extract distance matricies
    D1 = nx.to_pandas_adjacency(G, weight=weight_1).to_numpy() # Leave missing distances as 0
    D2 = nx.to_pandas_adjacency(G, weight=weight_2, nonedge=np.nan).to_numpy() # Fill missing distances as nan
    ind = np.isnan(D1) # Check if any values in D1 are zero
    D1[ind] = 0.; D2[ind] = np.nan # Replace nan with 0, nan
    upper_tri_indices = np.triu_indices(D1.shape[0], k=1)
    D1c = squareform(D1[upper_tri_indices]) # Convert to condensed
    D2c = squareform(D2[upper_tri_indices])
    del D1, D2, upper_tri_indices # Free up memory
    
    # Perform the Mantel test and ignore NaNs in dm2
    # The 'ignore_nans' parameter defaults to False, so set it to True
    r, p, z = mantel.test(D1c, D2c, method='pearson', perms=perms, ignore_nans=True)
    print(f"Correlation (r): {r}")
    print(f"P-value (p): {p} (perms = {perms})")
    print(f"Z-score (z): {z}\n")
    
    # Comparisons to dV
    # dV-dH:           r = 0.51259
    # dV-dphi:         r = 0.65555
    # dV-Edel:         r = 0.887
    # dV-dH_atx:       r = 0.8737  *** dist used in BallTree
    # dV-dH_nodal:     r = 0.8843
    
    
    # Comparisons between metrics
    # dH_atx-dH_nodal: r = 0.99698
    
    del D1c, D2c # Free up memory
    
    return


def ex3_error_analysis(df,dist_1 = 'dV', dist_2 = 'dH_atx', eps=0.1):
    
    
    # Compute the error
    err = df[dist_2] - df[dist_1] #
    
    # Compute alignment of periapses
    l1 = df['om1'] + df['w1'] # Longitude of periapsis
    l1 = (l1 + np.pi) % (2.0 * np.pi) - np.pi # wrap to =pi,pi
    
    l2 = df['om2'] + df['w2'] # Longitude of periapsis
    l2 = (l2 + np.pi) % (2.0 * np.pi) - np.pi # wrap to =pi,pi
    dl = l2-l1 # Differene
    
    df['dl'] = dl
    
    
    # Divide into classes
    # 1. err > 0:             dist_1 over approximates dV (over-estimate)
    # 2. range < err < range: dist_1 within range of error
    # 3. err < range:         dist_1 under estimates dV (under-estimate)
    ind1 = err > 0
    ind2 = (err < 0) & (-eps < err)
    ind3 = err < -eps
    
    n1 = float(100*len(df[ind1])/len(df))
    n2 = float(100*len(df[ind2])/len(df))
    n3 = float(100*len(df[ind3])/len(df))
    
    
    # Perform ANOVA test
    
    # List of variables to consider
    var_list = ['dl']
    
    print("\nANOVA Tests")
    print("----------")
    for var in var_list:
        
        f_statistic, p_value = f_oneway(df[var][(ind1) & (~pd.isna(df[var][ind1]))], df[var][(ind2) & (~pd.isna(df[var][ind2]))], df[var][(ind3) & (~pd.isna(df[var][ind3]))])
        print(f"\nVariable: {var}")
        print(f"F-statistic: {f_statistic}")
        print(f"P-value: {p_value}")
    
    
    # Plot the results
    fig, ax = plt.subplots(1,2,figsize=(12, 8)) 
    
    # Scatter plot
    ax[0].plot(df[dist_1][ind1], df[dist_2][ind1],'.g', markersize=0.5, label = dist_2 + " > " + dist_1 + "(over-estimate) " + f"{n1:.2f} %" ) # dist > 0
    ax[0].plot(df[dist_1][ind2], df[dist_2][ind2],'.k', markersize=0.5, label = dist_2 + " - " + dist_1 + " < " + str(eps) + " (good-estimate) " + f"{n2:.2f} %"  ) # dist > 0
    ax[0].plot(df[dist_1][ind3], df[dist_2][ind3],'.r', markersize=0.5, label = dist_2 + " < " + dist_1 + "(under-estimate) " + f"{n3:.2f} %" ) # dist > 0
    ax[0].set_xlabel(dist_1,fontsize=16)
    ax[0].set_ylabel(dist_2,fontsize=16)
    ax[0].legend()
    
    # Histogram
    # ax[1].hist(err, bins=200, density=True) #, cumulative=-1)
    # ax[1].set_yscale("log")
    
    # Histogram of values
    var = var_list[0]
    ax[1].hist(df[var][(ind1) & (~pd.isna(df[var][ind1]))], bins=100, color='green')
    ax[1].hist(df[var][(ind2) & (~pd.isna(df[var][ind2]))], bins=100, color='black')
    ax[1].hist(df[var][(ind3) & (~pd.isna(df[var][ind3]))], bins=100, color='red')
    
    
    # # Error vs alignment of perioapsis
    # ax[1].plot(dl[ind1], err[ind1],'.g', markersize=0.5, label = dist_2 + " > " + dist_1 + "(over-estimate) " + f"{n1:.2f} %" ) # dist > 0
    # ax[1].plot(dl[ind2], err[ind2],'.k', markersize=0.5, label = dist_2 + " - " + dist_1 + " < " + str(eps) + " (good-estimate) " + f"{n2:.2f} %"  ) # dist > 0
    # ax[1].plot(dl[ind3], err[ind3],'.r', markersize=0.5, label = dist_2 + " < " + dist_1 + "(under-estimate) " + f"{n3:.2f} %" ) # dist > 0
    # ax[1].set_xlabel('Apse alignment',fontsize=16)
    # ax[1].set_ylabel("Error",fontsize=16)
    
    # Error vs variables
    # var = 'e1'
    # # ax[1].plot(df['om1'][ind1]+df['w1'][ind1], df['om2'][ind1]+df['w2'][ind1],'.g', markersize=0.5, label = dist_2 + " > " + dist_1 + "(over-estimate) " + f"{n1:.2f} %" ) # dist > 0
    # # ax[1].plot(df['om1'][ind2]+df['w1'][ind2], df['om2'][ind2]+df['w2'][ind2],'.k', markersize=0.5, label = dist_2 + " - " + dist_1 + " < " + str(eps) + " (good-estimate) " + f"{n2:.2f} %"  ) # dist > 0
    # # ax[1].plot(df['om1'][ind3]+df['w1'][ind3], df['om2'][ind3]+df['w2'][ind3],'.r', markersize=0.5, label = dist_2 + " < " + dist_1 + "(under-estimate) " + f"{n3:.2f} %" ) # dist > 0
    # # ax[1].set_xlabel(var,fontsize=16)
    # # ax[1].set_ylabel(f"Error ({dist_2} - {dist_1})",fontsize=16)
    # ax[1].legend()
    
    
    
    
    # # Box plot
    # var_list = ['e1','e2'] # Variables for x axis
    
    
    # # Create subplots (one row, three columns)
    # fig = make_subplots(rows=1, cols=len(var_list), subplot_titles=var_list)

    # for i, col in enumerate(var_list):
    #     col_idx = i + 1
        
    #     # 3. Add box for ind1 (Group 1)
    #     fig.add_trace(
    #         go.Box(y=df.loc[ind1, col], name=f'{col} Ind1', 
    #                x0=0, marker_color='green'),
    #         row=1, col=col_idx
    #     )
    #     # 4. Add box for ind2 (Group 2) side-by-side
    #     fig.add_trace(
    #         go.Box(y=df.loc[ind2, col], name=f'{col} Ind2', 
    #                x0=0.5, marker_color='black'),
    #         row=1, col=col_idx
    #     )
    #     # 4. Add box for ind3 (Group 3) side-by-side
    #     fig.add_trace(
    #         go.Box(y=df.loc[ind3, col], name=f'{col} Ind3', 
    #                x0=1.0, marker_color='red'),
    #         row=1, col=col_idx
    #     )
    
    # # 5. Configure layout to group them
    # fig.update_layout(boxmode='overlay', title="Grouped Box Plots")

    # plotly.offline.plot(fig)
    
    
    
    return

#

def test_box_plot():
    
    
    # 1. Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(8),
        'B': np.random.randn(8),
        'C': np.random.randn(8)
    })
    ind1 = [0, 1, 2]
    ind2 = [3, 5, 6]
    
    # 2. Initialize subplots (1 row, 3 cols)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Column A", "Column B", "Column C"))
    
    # Columns to iterate over
    cols = ['A', 'B', 'C']
    for i, col in enumerate(cols):
        col_idx = i + 1
        
        # 3. Add box for ind1 (Group 1)
        fig.add_trace(
            go.Box(y=df.loc[ind1, col], name=f'{col} Ind1', 
                   x0=0, marker_color='blue'),
            row=1, col=col_idx
        )
        # 4. Add box for ind2 (Group 2) side-by-side
        fig.add_trace(
            go.Box(y=df.loc[ind2, col], name=f'{col} Ind2', 
                   x0=0.5, marker_color='red'),
            row=1, col=col_idx
        )
    
    # 5. Configure layout to group them
    fig.update_layout(boxmode='overlay', title="Grouped Box Plots")
    
    # Show plot
    plotly.offline.plot(fig)
    
    return


#%%

if __name__ == "__main__":
    
    # Load data
    df = load_dV_data()
    # Create network graph
    # G = create_graph(df)
    # # Get adjacency matrix of distances
    # D = nx.to_pandas_adjacency(G, weight='dV', nonedge=np.nan).to_numpy()
    
    # # Analysis: Compare symetry of delta-V distances
    # ex1_distance_matrix_symetry(df,G)
    
    
    # # 2. Analysis: Mandel test on two adjacency matrices
    # ex2_mandel_test_distances(df,G, weight_1 ='dH', weight_2 = 'dV', perms=3)
    # ex2_mandel_test_distances(df,G, weight_1 ='dphi', weight_2 = 'dV', perms=3)
    # ex2_mandel_test_distances(df,G, weight_1 ='Edel', weight_2 = 'dV', perms=3)
    # ex2_mandel_test_distances(df,G, weight_1 ='dH_atx', weight_2 = 'dV', perms=3)
    # ex2_mandel_test_distances(df,G, weight_1 ='dH_nodal', weight_2 = 'dV', perms=3)
    
    # ex2_mandel_test_distances(df,G, weight_1 ='dH_atx', weight_2 = 'dH_nodal', perms=3)
    
    # 3. Error analysis
    # ex3_error_analysis(df,dist_1 = 'dV', dist_2 = 'dH_atx', eps=0.1)
    
    
    
    

#%% Test case


    