# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 00:26:09 2022

@author: scott
"""

def load_asteroids(compute_params=True):
    
    # Use SR-Tools package
    
    from sr_tools.Datasets.Datasets import Datasets
    
    # Load data from MPCORB
    df = Datasets.load_dataset('mpcorb')
    
    # Compute orbital parameters
    if compute_params:
        
        # Constants
        mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
        AU = 149597870.700 # Astronomical unit (km)
         
        # Compute semi-latus rectum, periapsis and apoapsis
        df['p'] = df.a*(1-df.e**2) # Semi-latus rectum (km)
        
        # Compute specific angular momentum
        df['h'] = np.sqrt(mu*df.p)
    
        # Compute x,y,z coordinates of angular momentum vector projected on the
        # equatorial plane.
        # In perifocal frame, h=[0,0,1]. Using transformation matrix from perifocal
        # to ECEF (see Curtis pg 174), we can find the components in ECEF
        # hx = sin(i)*sin(om), hy = -sin(i)*cos(om), hz = cos(i)
        # See also Alfriend, Lee & Creamer (2006) "Optimal Servicing of Geosynchronous Satellites" 
        df['hx'] = np.sin(np.deg2rad(df.i))*np.sin(np.deg2rad(df.om))
        df['hy'] = -np.sin(np.deg2rad(df.i))*np.cos(np.deg2rad(df.om))
        df['hz'] = np.cos(np.deg2rad(df.i))
        # Scale by angular momentum
        df['hx'] = df['h']*df['hx']
        df['hy'] = df['h']*df['hy']
        df['hz'] = df['h']*df['hz']
    
    return df
