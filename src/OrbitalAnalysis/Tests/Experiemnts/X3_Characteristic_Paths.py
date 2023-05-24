# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:11:35 2023

@author: scott

Characteristic Paths
--------------------

Define a distance metric as the arc lenth of a characteristic curve connecting
two points in cylindrical coordinates.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pdb

from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.DistanceAnalysis import *
from OrbitalAnalysis.Visualization import *

#%% Distance Metrics

def d_const_rate(x1,x2):
    '''
    Distance metric defining the arc length of a path connecting two end points,
    where r and z increase linearly with theta.
    i.e. dr/dth = (r2-r1)/(th2-th1) = const and 
    dz/dth = (z2-z1)/(th2-th1)

    Parameters
    ----------
    x1 : TYPE
        DESCRIPTION.
    x2 : TYPE
        DESCRIPTION.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.

    '''
    
    # Extract elements
    r1,th1,z1 = x1.T
    r2,th2,z2 = x2.T
    
    # Compute gradients
    m1 = (r2-r1)/(th2-th1) # Gradient dr/dth
    m2 = (z2-z1)/(th2-th1) # Gradient dz/dth
    c1 = r1 - m1*th1 # Constant (intercept of r(theta) curve)
    
    # Compute arc length analytically
    L = (1/(2*m1))*(( r2*np.sqrt(r2**2+m1**2+m2**2) + (m1**2+m2**2)*np.log(r2 + np.sqrt(r2**2+m1**2+m2**2) )  ) - \
                    ( r1*np.sqrt(r1**2+m1**2+m2**2) + (m1**2+m2**2)*np.log(r1 + np.sqrt(r1**2+m1**2+m2**2) )  )
                    ) 
    
    dist = abs(L)
    
    return dist

#%% Plotting functions
def create_theta_array(th1,th2,step=0.1*np.pi/180.):
    
    if th2>th1:
        th_vec = np.arange(th1,th2,step)
    elif th2 < th1:
        th_vec = np.arange(th1,th2,-step)
    else:
        th_vec = np.array([th1,th2])
    
    return th_vec

def plot_paths(x1,x2):
    
    # Plot the paths between pairs of end points
    

    # Convert to pandas
    df = pd.DataFrame(np.concatenate([x1,x2],axis=1),columns=['hr1','htheta1','hz1','hr2','htheta2','hz2'])
    # Create indices
    df.insert(0,'PairNumber',[i for i in range(len(x1))])
    
    # Compute gradients
    df['m1'] = np.nan
    df['m2'] = np.nan
    df['c1'] = np.nan
    df.m1 = (df.hr2-df.hr1)/(df.htheta2-df.htheta1) # Gradient dr/dth
    df.m2 = (df.hz2-df.hz1)/(df.htheta2-df.htheta1) # Gradient dz/dth
    df.c1 = df.hr1 - df.m1*df.htheta1 # Constant (intercept of r(theta) curve)
    
    # Compute distances
    dist = d_const_rate(x1, x2)
    
    # Loop through lines
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    for i,row in df.iterrows():
        # th_vec = np.linspace(row.htheta1,row.htheta2,100)
        th_vec = create_theta_array(row.htheta1,row.htheta2)
        r = row.hr1 + row.m1*(th_vec - row.htheta1)
        x = r*np.cos(th_vec)
        y = r*np.sin(th_vec)
        z = row.hz1 + row.m2*(th_vec - row.htheta1) 
        L_num = sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
        
        # Add to plot
        ax.scatter3D(row.hr1*np.cos(row.htheta1),row.hr1*np.sin(row.htheta1),row.hz1, 'ok') # Start point
        ax.scatter3D(row.hr2*np.cos(row.htheta2),row.hr2*np.sin(row.htheta2),row.hz2, 'ok')  # End point
        ax.plot3D(row.hr1*np.cos(np.linspace(0,2*np.pi,100)),row.hr1*np.sin(np.linspace(0,2*np.pi,100)),row.hz1*np.ones(100),'--k')
        ax.plot3D(row.hr2*np.cos(np.linspace(0,2*np.pi,100)),row.hr2*np.sin(np.linspace(0,2*np.pi,100)),row.hz2*np.ones(100),'--k')
        ax.plot3D(x,y,z, '-r')
        
    
    # # Add theta vectors
    # df['th_vec'] = np.nan
    # df['th_vec'] = [np.linspace(df['htheta1'].iloc[i],df['htheta2'].iloc[i],100) for i in range(len(df)) ]

    # Explode array to new rows
    
    # # Compute r vectors
    # df['r_vec'] = np.nan
    # df['r_vec'] = [ df.hr1.iloc[0] + df.m1.iloc[0]*(df.th_vec.iloc[0] - df.htheta1.iloc[0])   for i in range(len(df)) ]
    
    
    # r = r0 + m1*(th_vec - th0) # c1 + m1*th_vec
    
    # x = r0*np.cos(th_vec) + m1*(th_vec - th0)*np.cos(th_vec)
    # y = r0*np.sin(th_vec) + m1*(th_vec - th0)*np.sin(th_vec)
    # z = z0 + m2*(th_vec - th0) 



    return


#%% Initial test of path between two points. Confirm analytical function

def test_mulitple_pairs_case1():
    
    # Case 1, starting from th1=0
    # Test -ve and +ve values for th2
    # Confirmed it works!!!
    
    N = 50 # Number of points
    
    # Initial point (fixed)
    r0,th0,z0 = 1., 0., 0.
    x1 = np.tile(np.array([r0,th0,z0]),(N,1))
    
    # End points. Distributed evenly in theta
    r2,z2 = 1.2,0.2
    th2min,th2max = -0.3*np.pi,1.5*np.pi # Range of th2 values
    x2 = np.zeros(x1.shape) # Initialize
    x2[:,0] = r2
    x2[:,2] = z2
    x2[:,1] = np.linspace(th2min,th2max,N)
    
    # Compute distances
    dist = d_const_rate(x1, x2)
    
    # Plot
    plot_paths(x1, x2)
    
    return

def test_mulitple_pairs_case2():
    
    # Case 2, starting from +ve th1
    # Test values of th2 < th1 and th2>th1
    # Confirmed it works!!!
    
    N = 50 # Number of points
    
    # Initial point (fixed)
    r0,th0,z0 = 1., 0.6*np.pi, 0.
    x1 = np.tile(np.array([r0,th0,z0]),(N,1))
    
    # End points. Distributed evenly in theta
    r2,z2 = 1.2,0.2
    th2min,th2max = 0.3*np.pi,1.5*np.pi # Range of th2 values
    x2 = np.zeros(x1.shape) # Initialize
    x2[:,0] = r2
    x2[:,2] = z2
    x2[:,1] = np.linspace(th2min,th2max,N)
    
    # Compute distances
    dist = d_const_rate(x1, x2)
    
    # Plot
    plot_paths(x1, x2)
    
    return

def test_mulitple_pairs_case3():
    
    # Case 3, starting from +ve th1
    # Test wrapping values around 2pi
    
    N = 50 # Number of points
    
    # Initial point (fixed)
    r0,th0,z0 = 1., 0., 0.
    x1 = np.tile(np.array([r0,th0,z0]),(N,1))
    
    # End points. Distributed evenly in theta
    r2,z2 = 1.2,0.2
    th2min,th2max = 0.9*np.pi,2.3*np.pi # Range of th2 values
    x2 = np.zeros(x1.shape) # Initialize
    x2[:,0] = r2
    x2[:,2] = z2
    x2[:,1] = np.linspace(th2min,th2max,N)
    
    # Compute distances
    dist = d_const_rate(x1, x2)
    
    # Plot
    plot_paths(x1, x2)
    
    return


def unit_test_sinle_pair():

    # Define start and end points
    r0,th0,z0 = 1., 0., 0.
    r1,th1,z1 = 1.2, 0.3*np.pi, 0.
    r1,th1,z1 = 1.2, 1.5*np.pi, 1.2
    
    # Compute x,y coods of ends
    x0, y0 = r0*np.cos(th0), r0*np.sin(th0)
    x1, y1 = r1*np.cos(th1), r1*np.sin(th1)
    
    # Compute gradients
    m1 = (r1-r0)/(th1-th0) # Gradient dr/dth
    m2 = (z1-z0)/(th1-th0) # Gradient dz/dth
    c1 = r0 - m1*th0 # Constant (intercept of r(theta) curve)
    
    # Define arc as fn of theta
    th_vec = np.linspace(th0,th1,1000)
    r = r0 + m1*(th_vec - th0) # c1 + m1*th_vec
    x = r0*np.cos(th_vec) + m1*(th_vec - th0)*np.cos(th_vec)
    y = r0*np.sin(th_vec) + m1*(th_vec - th0)*np.sin(th_vec)
    z = z0 + m2*(th_vec - th0) 
    
    # Compute arc length numerically
    L_num = sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    
    # Compute arc length analytically
    L = (1/(2*m1))*(( r1*np.sqrt(r1**2+m1**2+m2**2) + (m1**2+m2**2)*np.log(r1 + np.sqrt(r1**2+m1**2+m2**2) )  ) - \
                    ( r0*np.sqrt(r0**2+m1**2+m2**2) + (m1**2+m2**2)*np.log(r0 + np.sqrt(r0**2+m1**2+m2**2) )  )
                    ) 
    
    # Test function with single input
    x1 = np.array([r0,th0,z0])
    x2 = np.array([r1,th1,z1])
    dist = d_const_rate(x1,x2)    
    
    # Test multiple inputs
    x1 = np.tile(x1,(3,1))
    x2 = np.tile(x2,(3,1))
    dist = d_const_rate(x1,x2)   
        
    # Print results
    print('P0: (r0,th0,z0) = ({}, {}, {})'.format(r0,th0,z0))
    print('P1: (r1,th1,z1) = ({}, {}, {})'.format(r1,th1,z1))
    print('Arc length (numerical): {}'.format(L_num))
    print('Arc length (analytical): {}'.format(L))
    print('Difference: {}'.format(abs(L-L_num)))
    
    
    # Draw Arc
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    N = (max(z)-min(z))/(max(x)-min(x))
    ax.set_box_aspect((1, 1, N))
    ax.scatter3D(x0,y0,z0, 'ok') # Start point
    ax.scatter3D(x1,y1,z1, 'ok') # End point
    ax.plot3D(r0*np.cos(np.linspace(0,2*np.pi,100)),r0*np.sin(np.linspace(0,2*np.pi,100)),z0*np.ones(100),'--k')
    ax.plot3D(r1*np.cos(np.linspace(0,2*np.pi,100)),r1*np.sin(np.linspace(0,2*np.pi,100)),z1*np.ones(100),'--k')
    ax.plot3D(x,y,z, '-r')
    
    return

#%% Apply Distance Metric to Satellites
    
# Load data
# df = load_satellites(group='all',compute_params=True,compute_pca=True)
df = load_2019_experiment_data([36]) # New dataset
    
#%% Compute Distance metrics
# target = 25544 # ISS
target = 22675 # Cosmos 2251
df = compute_distances(df,target,searchfield='NoradId')

# Rename
df = df.rename(columns={'dH':'d_Euc',
                        'dHtheta_arc':'d_arc',
                        'dHcyl':'d_cyl'})


# Plot in 3d

# plot_h_space_numeric(df,color='dH',logColor=True,colorscale='Blackbody') # dH (Euclidean)

# plot_h_space_numeric(df,color='dHcyl',logColor=True,colorscale='Blackbody') # dH (Euclidean)


# Euclidean
fig = plot_3d_scatter_numeric(df,'hx','hy','hz',color='d_Euc',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=True,
                            logColor=True,
                            )

fig = plot_3d_scatter_numeric(df,'hx','hy','hz',color='d_arc',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=True,
                            logColor=True,
                            )

fig = plot_3d_scatter_numeric(df,'hx','hy','hz',color='d_cyl',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=True,
                            logColor=True,
                            )



#%% Comparison

max_dist = max([df.d_arc.max(),df.d_cyl.max(),df.d_Euc.max()])

# Euclidean distance is always smallest of the three
fig, ax = plt.subplots(2,2,figsize=(20, 15)) 
# plt.suptitle('Comparison', fontweight ="bold")  
ax[0,0].plot(df.d_Euc,df.d_arc,'.k')
ax[0,0].plot([0,max_dist],[0,max_dist],'-r')
ax[0,0].set_xlabel(r'$d_{Euc}$ (Euclidean) (km$^{2}$/s)',fontsize=16)
ax[0,0].set_ylabel(r'$d_{arc}$ (Mean arc length) (km$^{2}$/s)',fontsize=16)

ax[0,1].plot(df.d_Euc,df.d_cyl,'.k')
ax[0,1].plot([0,max_dist],[0,max_dist],'-r')
ax[0,1].set_xlabel(r'$d_{Euc}$ (Euclidean) (km$^{2}$/s)',fontsize=16)
ax[0,1].set_ylabel(r'$d_{cyl}$ (Cylindrical) (km$^{2}$/s)',fontsize=16)

ax[1,0].plot(df.d_arc,df.d_cyl,'.k')
ax[1,0].plot([0,max_dist],[0,max_dist],'-r')
ax[1,0].set_xlabel(r'$d_{arc}$ (Mean arc length) (km$^{2}$/s)',fontsize=16)
ax[1,0].set_ylabel(r'$d_{cyl}$ (Cylindrical) (km$^{2}$/s)',fontsize=16)


#%% Circular ring at

# Reference hz value
hz0 = df['hz'][df.NoradId==target].iloc[0]
hz_range = 2000 # Range in hz to consider

# Create a new dataset that just filters out the objects around the ring
df1 = df.copy()
df1 = df[abs(df.hz-hz0) <= hz_range]


fig = plot_2d_scatter_numeric(df1,'hx','hy',color='d_Euc',size=3,logColor=False)

fig = plot_2d_scatter_numeric(df1,'hx','hy',color='d_cyl',size=3,logColor=False)

#%% Plots


# # Plot dH (euclidean) vs dtheta
# fig, ax = plt.subplots(3,1,figsize=(8, 8)) 
# plt.suptitle('Cartesian Euclidean Distance', fontweight ="bold")  
# ax[0].plot(df.dHr,df.d_Euc,'.k')
# ax[0].set_xlabel(r'$h_{r} $',fontsize=16)
# ax[0].set_ylabel(r'dH',fontsize=16)

# ax[1].plot(df.dHz,df.d_Euc,'.k')
# ax[1].set_xlabel(r'$h_{z} $',fontsize=16)
# ax[1].set_ylabel(r'dH',fontsize=16)

# ax[2].plot(df.dHtheta,df.d_Euc,'.k')
# ax[2].set_xlabel(r'$h_{\theta} $',fontsize=16)
# ax[2].set_ylabel(r'dH',fontsize=16)

# # Cylindrical mean arc length
# fig, ax = plt.subplots(3,1,figsize=(8, 8))
# plt.suptitle('Cylindrical Mean Arc Length', fontweight ="bold")  
# ax[0].plot(df.dHr,df.d_arc,'.k')
# ax[0].set_xlabel(r'$h_{r} $',fontsize=16)
# ax[0].set_ylabel(r'dHcyl_mean',fontsize=16)

# ax[1].plot(df.dHz,df.d_arc,'.k')
# ax[1].set_xlabel(r'$h_{z} $',fontsize=16)
# ax[1].set_ylabel(r'dHcyl_mean',fontsize=16)

# ax[2].plot(df.dHtheta,df.d_arc,'.k')
# ax[2].set_xlabel(r'$h_{\theta} $',fontsize=16)
# ax[2].set_ylabel(r'dHcyl_mean',fontsize=16)

# # Cylindrical curvilinear
# fig, ax = plt.subplots(3,1,figsize=(8, 8)) 
# plt.suptitle('Cylindrical Curvilinear', fontweight ="bold")  
# ax[0].plot(df.dHr,df.d_cyl,'.k')
# ax[0].set_xlabel(r'$h_{r} $',fontsize=16)
# ax[0].set_ylabel(r'dHcyl_curv',fontsize=16)

# ax[1].plot(df.dHz,df.d_cyl,'.k')
# ax[1].set_xlabel(r'$h_{z} $',fontsize=16)
# ax[1].set_ylabel(r'dHcyl_curv',fontsize=16)

# ax[2].plot(df.dHtheta,df.d_cyl,'.k')
# ax[2].set_xlabel(r'$h_{\theta} $',fontsize=16)
# ax[2].set_ylabel(r'dHcyl_curv',fontsize=16)