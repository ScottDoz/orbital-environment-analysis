# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:34:12 2024

@author: scott

Catalog Module
--------------

* Catalog class defines a catalog of space objects from the TLEs of all tracked
  objects at a particular epoch.
* Methods for propagating orbits to common epochs using SGP4
* Methods for visualizing orbital parameters
* Methods for computing density and other analytics

"""



import numpy as np
import pandas as pd
from datetime import datetime

import cysgp4

import matplotlib
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
# for animations, you may need to install "ffmpeg" and/or "imagemagick"
from matplotlib import animation, rc
rc('animation', html='html5')

# Module imports
from SatelliteData import get_spacetrack_tles_as_string, compute_orbital_params
from OrbitalAnalysis.Visualization import *
from Functions import coe_from_sv, TA_to_M
from Density import compute_density

import pdb

#%% 
class Catalog:
    
    def __init__(self):
        
        # Define class attributes
        self.method = None
    
    
    # Constructors ------------------------------------------------------------
    @classmethod
    def from_spacetrack(cls):
        '''
        Instantiate a catalog from latest TLEs in the Spacetrack database.

        Returns
        -------
        cls    Class
        '''
        
        # Load TLEs from current Spacetrack TLEs
        tle_str = get_spacetrack_tles_as_string()
        
        # Convert into tle list
        tles = cysgp4.tles_from_text(tle_str)
        tles = np.array(tles) # Convert to numpy array to allow broadcasting

        
        # Extract other data
        norad_num = [ tle.catalog_number for tle in tles ] # Norad catalog numbers
        names =  [ tle.tle_strings()[0][2:] for tle in tles  ] # Satellite name
        epochs = [ tle.epoch for tle in tles  ] # Satellite name
        
        # Form dataframe
        df = pd.DataFrame(columns=['NoradId','Name','tle_epoch'])
        df['NoradId'] = norad_num
        df['Name'] = names
        df['tle_epoch'] = epochs
        # df['tle'] = tles.tolist()
        
        # Add to class attributes
        cls.tles = tles
        cls.df = df
        
        return cls()
    
    # Getters -----------------------------------------------------------------
    
    def get_eci_states(self,epoch='now',observers=None):
        
        if epoch == 'now':
            # Test. Define single epoch at current time
            dt = datetime.now()
            pydt = cysgp4.PyDateTime(dt)
            mjds = np.array([pydt.mjd])
        
        else:
            # FIXME: Array of epochs
        
            # Array of results
            start_mjd = epoch_dt.mjd
            td = np.arange(0, 600, 5) / 86400.  # 1 d in steps of 10 s
            mjds = start_mjd + td
        
        
        # Call propagation
        result  = self.propagate_to_epoch(mjds,observers=None)
        
        # Format for return
        eci_pos = result['eci_pos'][:,0,0,:] # ECI position (km)
        eci_vel = result['eci_vel'][:,0,0,:] # ECI velocity (km/s)
        R = eci_pos; V = eci_vel;
        
        # Angular momentum
        H = np.cross(R,V) # Vector
        h = np.linalg.norm(H,axis=-1) # Magnitude
        
        # Orbital Elements
        a,e,i,om,w,TA = coe_from_sv(R,V,mu=3.986004418e5,units='km')
        M = TA_to_M(TA,e)

        # Add to dataframe
        df = self.df[['NoradId','Name']].copy()
        df[['x','y','z']] = eci_pos
        df[['vx','vy','vz']] = eci_vel
        df[['Hx','Hy','Hz']] = H
        df['a'] = a; df['e'] = e; 
        df['i'] = np.rad2deg(i); df['om'] = np.rad2deg(om); 
        df['w'] = np.rad2deg(w); df['TA'] = np.rad2deg(TA);
        df['M'] = np.rad2deg(M);
        df['epoch_mjd'] = mjds[0] 
        
        # Compute orbital parameters
        df = compute_orbital_params(df)
        
        return df
    
    # Propagation -------------------------------------------------------------
    
    def propagate_to_epoch(self,mjds,observers= None):
        
        # # Effelsberg 100-m radio telescope
        # effbg_observer = cysgp4.PyObserver(6.88375, 50.525, 0.366)
        # # Parkes telescope ("The Dish")
        # parkes_observer = cysgp4.PyObserver(148.25738, -32.9933, 414.8)
        # observers = np.array([effbg_observer, parkes_observer])
        
        # # Full result with observers
        # result = cysgp4.propagate_many(
        #         mjds[np.newaxis, np.newaxis, :],
        #         self.tles[:, np.newaxis, np.newaxis],
        #         observers[np.newaxis, :, np.newaxis],
        #         on_error='coerce_to_nan',
        #         )       
        
        # Without observers
        result = cysgp4.propagate_many(
                mjds[np.newaxis, np.newaxis, :],
                self.tles[:, np.newaxis, np.newaxis],
                # observers[np.newaxis, :, np.newaxis],
                on_error='coerce_to_nan',
                )    
        
        return result
    
    
    # Plotting
    
    def plot_eci(self, epoch='now'):
        
        # Get states
        df = self.get_eci_states(epoch)

        # Initialize figure
        fig = plot_3d_scatter_numeric(df,'x','y','z',#color='p_hxhyhz',
                                    # logColor=False,colorscale='Blackbody_r',
                                    xrange=[-250000,250000],
                                    yrange=[-250000,250000],
                                    zrange=[-250000,250000],
                                    markersize=1.0,
                                    aspectmode='cube',
                                    render=False,
                                    )
        
        # Update camera focus
        name = 'looking up'
        camera = dict(center=dict(x=0, y=0, z=0))
        fig.update_layout(scene_camera=camera, title=name)
        # fig.update_layout( scene_camera={"up": {"x": 0.0, "y": 1.0, "z": 0.0}}, scene={ "dragmode": "turntable", "yaxis": {"title": "Z"}, "zaxis": {"title": "Y"}, }, )
        
        # fig.show()
        plotly.offline.plot(fig,filename='ECI-plot.html')
        
        return
    
    def plot_density(self,epoch='now'):
        
        # Get states
        df = self.get_eci_states(epoch)
        
        # Compute density
        df['p_hxhyhz'] = compute_density(df)
        
        # Initialize figure
        fig = plot_3d_scatter_numeric(df,'Hx','Hy','Hz',color='p_hxhyhz',
                                    logColor=False,colorscale='Blackbody_r',
                                    xrange=[-120000,120000],
                                    yrange=[-120000,120000],
                                    zrange=[-50000,150000],
                                    aspectmode='cube',
                                    render=False,
                                    )
        
        # Update camera focus
        name = 'looking up'
        camera = dict(
            center=dict(x=0, y=0, z=0))
        fig.update_layout(scene_camera=camera, title=name)
        
        # fig.show()
        plotly.offline.plot(fig,filename='H-density-plot.html')
        
        return
        
        
        
        
#%% Main fucntion

if __name__ == "__main__":
    
    cat = Catalog.from_spacetrack()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        