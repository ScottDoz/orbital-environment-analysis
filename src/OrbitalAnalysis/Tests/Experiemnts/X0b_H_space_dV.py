# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:06:34 2026

@author: scott

Goal: Explore fundamental properties of specific orbital angular momentum space.

* Consider reachability of H-space due to small fixed delta-V.
* Plot the angular momentum space with a unit sphere.Find locus of reachable points.

"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pdb

from OrbitalAnalysis.SatelliteData import *
from astrologistics.Functions import sv_from_coe

def plot_orbit_h_reachability(orb, dV_mag, fig=None, ax=None):
    
    
    # Extract orbtial elements
    # Orbit 1
    a = orb['a']
    e = orb['e']
    i = orb['i']
    om = orb['om']
    w = orb['w']
    M = orb['M']
    
    # Constants
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    
    
    
    
    # Get orbit
    M = np.linspace(0,2*np.pi, 100) 
    sv = sv_from_coe(a,e,i,om,w,M, mu=mu, units='km')
    R = sv[:,:3] # Position vector ECI (km)
    V = sv[:,3:] # Velocity vector ECI (km/s)
    
    
    # Angular momenetum vector
    h = np.sqrt(mu*a*(1 - e**2)) # Magnitude
    hx = h*np.sin(i)*np.sin(om) # x component
    hy = -np.sin(i)*np.cos(om)*h # y component
    hz = np.cos(i)*h # z component
    H = np.array([hx,hy,hz]) # Vector
    
    
    # Tangential Delta-V
    # At each location, apply fixed delta-V in tangential direction
    V_u = V/np.linalg.norm(V, axis=-1)[:, np.newaxis] # Unit vector in velocity
    T_u = np.cross(np.tile(H,(len(R),1)), R) # H x R
    T_u = T_u/np.linalg.norm(T_u, axis=-1)[:, np.newaxis] # Normalize
    V_plus = V + dV_mag*T_u # Velocity after tangential impulse
    
    # Normal Delta-V
    # h_u = np.cross(R,V); h_u = h_u/np.linalg.norm(h_u,axis=-1)[:, np.newaxis] 
    h_u = np.tile(H/np.linalg.norm(H), (len(R),1 ))
    V_plus = V + dV_mag*h_u #np.tile(h_u,(len(V),1))
    
    
    # # Radial Delta-V
    # R_u = R/np.linalg.norm(R, axis=-1)[:, np.newaxis] # Unit vector in R direction
    # V_plus = V + dV_mag*R_u # Velocity after impulse
    # # Confirmed: no change to H (as expected)
    
    # # Mixture
    # V_plus = V + 0.7*dV_mag*T_u + 0.3*dV_mag*h_u
    
    # # Re-compute H after delta-V
    # H_plus = np.cross(R,V_plus) # Angular momentum after impulse
    
    
    # Prepare 3D plot
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_box_aspect([1,1,1])
    
    # Plot H position
    ax.scatter(*H, color='r', s=40, alpha=0.3, label='H (Angular Momentum)')
    ax.plot([0, H[0]],[0, H[1]],[0, H[2]],'-r', label='H (Angular Momentum)')
    
    # Plot orbit
    ax.plot(R[:,0],R[:,1],R[:,2], color='b', alpha=0.3, label='Satellite Orbit (Position)') # At origin
    ax.plot(R[:,0] + hx, R[:,1] + hy, R[:,2] + hz, color='b', alpha=0.9, label='Satellite Orbit (Position)') # At H
    
    
    # Generate data for a sphere for plotting
    radius = h
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='r', alpha=0.05, antialiased=False)
    
    # Plot locii for arbitrary mixture of Tangential and Normal delta-V
    V_plus = V + 0.0*dV_mag*T_u + 1.0*dV_mag*h_u # Mixture
    H_plus = np.cross(R,V_plus) # Angular momentum after impulse
    # Plot H locii for Tangential delta-V 
    ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='orange', alpha=1.0, label='Tangential Impulse') # Locii of H after tangential impulse
    # Connections from position in orbit to final position in H space
    start_points = R + np.array([hx,hy,hz]) #np.tile(H,(len(H_plus),1))
    end_points = H_plus
    # segments = np.hstack([start_points, end_points]).reshape((-1, 2, 3))
    segments = np.hstack([start_points[0,:], end_points[0,:]]).reshape((-1, 2, 3)) # Only first point
    # Create the Line3DCollection object
    lc = Line3DCollection(segments, colors='orange', linewidths=1)
    # Add the collection to the axes
    ax.add_collection(lc)
    
    # Find angle between r and h_plus vectors
    cos_delta = np.einsum('ij,ij->i',R,H_plus)
    delta = np.arccos(cos_delta) 
    # delta=90 deg regardless of dV
    
    
    # Distance metric from original H
    dh_vec = H_plus - np.tile(H, (len(H_plus),1))
    dh = np.linalg.norm(dh_vec,axis=-1)
    dh_r = dh/np.linalg.norm(R,axis=-1) # == dV_mag
    
    
    # Plot reachable surface
    
    levels = np.linspace(0,1,100)
    
    # +ve T, +ve N
    array_list = []
    for c_T in levels:
        # c_T = % of delta-V in tangential direction
        c_N = 1.0 - c_T # % delta-V in Normal direction
        V_plus = V + c_T*dV_mag*T_u + c_N*dV_mag*h_u # Mixture
        H_plus = np.cross(R,V_plus) # Angular momentum after impulse
        array_list.append(H_plus)
        # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='blue', alpha=1.0, label='+ve T, +ve N') # Locii of H after tangential impulse
    points = np.concatenate(array_list, axis=0) # All points
    hull = ConvexHull(points) # convex hull
    s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                    color='blue', alpha=0.2, edgecolor=None, label='Access Region +ve T, +ve N')
    
    # Distance metric from original H
    dh_vec = H_plus - np.tile(H, (len(H_plus),1))
    dh = np.linalg.norm(dh_vec,axis=-1)
    dh_r = dh/np.linalg.norm(R,axis=-1) # == dV_mag
    
    # +ve T, -ve N
    array_list = []
    for c_T in levels:
        # c_T = % of delta-V in tangential direction
        c_N = 1.0 - c_T # % delta-V in Normal direction
        V_plus = V + c_T*dV_mag*T_u - c_N*dV_mag*h_u # Mixture
        H_plus = np.cross(R,V_plus) # Angular momentum after impulse
        array_list.append(H_plus)
        # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='green', alpha=1.0, label='+ve T, -ve N') # Locii of H after tangential impulse
    points = np.concatenate(array_list, axis=0) # All points
    hull = ConvexHull(points) # convex hull
    s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                    color='green', alpha=0.2, edgecolor=None, label='Access Region +ve T, -ve N')
    
    # -ve T, +ve N
    array_list = []
    for c_T in levels:
        # c_T = % of delta-V in tangential direction
        c_N = 1.0 - c_T # % delta-V in Normal direction
        V_plus = V - c_T*dV_mag*T_u + c_N*dV_mag*h_u # Mixture
        H_plus = np.cross(R,V_plus) # Angular momentum after impulse
        array_list.append(H_plus)
        # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='orange', alpha=1.0, label='-ve T, +ve N') # Locii of H after tangential impulse
    points = np.concatenate(array_list, axis=0) # All points
    hull = ConvexHull(points) # convex hull
    s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                    color='orange', alpha=0.2, edgecolor=None, label='Access Region -ve T, +ve N')
    
    # -ve T, -ve N
    array_list = []
    for c_T in levels:
        # c_T = % of delta-V in tangential direction
        c_N = 1.0 - c_T # % delta-V in Normal direction
        V_plus = V - c_T*dV_mag*T_u - c_N*dV_mag*h_u # Mixture
        H_plus = np.cross(R,V_plus) # Angular momentum after impulse
        array_list.append(H_plus)
        # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='red', alpha=1.0, label='-ve T, -ve N') # Locii of H after tangential impulse
    points = np.concatenate(array_list, axis=0) # All points
    hull = ConvexHull(points) # convex hull
    s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                    color='red', alpha=0.2, edgecolor=None, label='Access Region -ve T, -ve N')
    
    ax.legend()
    
    return fig,ax





if __name__ == "__main__":
    
    
    
    # Constants
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    
    # Define an orbit
    # State Vector. Circular orbit 
    orb = {'a':Re+200,
           'e':0.0,
           'i':np.deg2rad(40.),
           'om':np.deg2rad(60.),
           'w':np.deg2rad(30.),
           'M':np.deg2rad(0.),
           'units':'km'}
    
    # Delta-V mag
    dV_mag = 0.5 # Delta-V (km/s)
    
    
    # Plot orbit in H space
    fig,ax = plot_orbit_h_reachability(orb, dV_mag) # 200 km orbit
    
    
    # State Vector. Circular orbit 
    orb = {'a':Re+1000,
           'e':0.0,
           'i':np.deg2rad(40.),
           'om':np.deg2rad(60.),
           'w':np.deg2rad(30.),
           'M':np.deg2rad(0.),
           'units':'km'}
    
    # Delta-V mag
    dV_mag = 0.5 # Delta-V (km/s)
    
    
    # Plot orbit in H space
    fig,ax = plot_orbit_h_reachability(orb, dV_mag, fig,ax) # 200 km orbit
    
    # State Vector. Circular orbit 
    orb = {'a':Re+10000,
           'e':0.0,
           'i':np.deg2rad(40.),
           'om':np.deg2rad(60.),
           'w':np.deg2rad(30.),
           'M':np.deg2rad(0.),
           'units':'km'}
    
    # Delta-V mag
    dV_mag = 0.5 # Delta-V (km/s)
    
    
    # Plot orbit in H space
    fig,ax = plot_orbit_h_reachability(orb, dV_mag, fig,ax) # 200 km orbit
    
    
    # State Vector. Circular orbit 
    orb = {'a':Re+30000,
           'e':0.0,
           'i':np.deg2rad(40.),
           'om':np.deg2rad(60.),
           'w':np.deg2rad(30.),
           'M':np.deg2rad(0.),
           'units':'km'}
    
    # Delta-V mag
    dV_mag = 0.5 # Delta-V (km/s)
    
    
    # Plot orbit in H space
    fig,ax = plot_orbit_h_reachability(orb, dV_mag, fig,ax) # 200 km orbit

    # TODO: Things to test
    # Sanity check: pick a point outside of accessible area. What would it take to actually get to that orbit?
    # Sample locations close to satellite, compute delta-Vs and compare.
    # Plot accessibility envelope, when using combined dV_t and dV_n
    