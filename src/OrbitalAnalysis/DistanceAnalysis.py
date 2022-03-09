# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:49:34 2022

@author: scott

Distance Analysis
-----------------

Compute distances between sets of objects.

"""

import pdb

# Relavive imports
from DistanceMetrics import *

#%% Computing distances

def compute_distances(df,target,searchfield='norad'):
    '''
    Compute distance metrics between a target object and all other objects in
    the dataset.

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe containing satellite or asteorid orbital elements.
    target : str or int
        Identifier for the target object (e.g. norad id or spkid)
    searchfield : str, optional
        The data field for which to search for the target. 
        The default is 'norad'.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe containing the original elements and computed distances.

    '''
    
    # Error checking
    if target not in list(df[searchfield]):
        raise ValueError('Target object not found')
        
    
    
    # Get number of objects
    N = len(df)
    
    # Check if asteroid dataset
    astflag = False
    if 'pdes' in df.columns:
        astflag = True
    
    # H-space Euclidean = f(hx1,hy1,hz1)
    x1 = np.tile(df[['hx','hy','hz']][df[searchfield]==target].iloc[0].to_numpy(), (N,1))
    x2 = df[['hx','hy','hz']].to_numpy()
    df['dH'] = dist_dH(x1,x2)
    
    # Plane change = f(hx1,hy1,hz1)
    df['dphi'] = dist_planechange(x1, x2)
        
    # D1,p1,p2,p3,p4 = f(q1,e1,inc1,om1,w1)
    x1 = df[['q','e','i','om','w']][df[searchfield] == target].to_numpy()
    x1[:,2:] = np.deg2rad(x1[:,2:]) # Convert angles to radians
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df[['q','e','i','om','w']].to_numpy()
    x2[:,2:] = np.deg2rad(x2[:,2:]) # Convert angles to radians
    df['D1']  = dist_D1(x1,x2)
    df['p1'] = dist_p1(x1,x2)
    df['p2'] = dist_p2(x1,x2)
    df['p3'] = dist_p3(x1,x2)
    df['p4'] = dist_p4(x1,x2)
    
    # p5,Zapalla = f(p1,e1,inc1)
    df['p5'] = dist_p5(x1[:,:3],x2[:,:3]) 
    df['zappala'] = dist_zappala(x1[:,:3],x2[:,:3],astflag=astflag)
    
    # Minimum Nodal Intersection Distance = f(a,e,i,om,w)
    x1 = df[['a','e','i','om','w']][df[searchfield] == target].to_numpy()
    x1[:,2:] = np.deg2rad(x1[:,2:]) # Convert angles to radians
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df[['a','e','i','om','w']].to_numpy()
    x2[:,2:] = np.deg2rad(x2[:,2:]) # Convert angles to radians
    df['mnid'] = dist_mnid(x1,x2)
    
    
    # Edel = f(a1,e1,inc1,om1,w1,hx1,hy1,hz1)
    x1 = df[['a','e','i','om','w','hx','hy','hz']][df[searchfield] == target].to_numpy()
    x1[:,2:5] = np.deg2rad(x1[:,2:5]) # Convert angles to radians
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df[['a','e','i','om','w','hx','hy','hz']].to_numpy()
    x2[:,2:5] = np.deg2rad(x2[:,2:5]) # Convert angles to radians
    df['Edel'] = dist_edel(x1,x2,astflag=astflag)
    
    return df

