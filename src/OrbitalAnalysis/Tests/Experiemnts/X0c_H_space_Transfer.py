# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:35:30 2026

@author: scott

Goal: Consider a two-impulse transfer between two arbitrary orbits.
      - plot the orbits in h space
      - draw locii of reachable h space with impulsive transfer
      - analyze overlapping region
      - plot optimal transfer, analyze geometry

"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import griddata
import pdb

from OrbitalAnalysis.SatelliteData import *
from astrologistics.Functions import sv_from_coe
from astrologistics.OrbitToOrbit import OrbitToOrbitProblem
from astrologistics.optimizers import chandrupatla

def plot_orbit_h_reachability(orb, dV_mag, fig=None, ax=None, show_sphere=False, plot_reachability=True, color=None):
    
    
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
        ax.set_aspect('equal')
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
    if show_sphere:
        ax.plot_surface(x, y, z, color='r', alpha=0.05, antialiased=False)
    
    # Plot locii for arbitrary mixture of Tangential and Normal delta-V
    # V_plus = V + 0.0*dV_mag*T_u + 1.0*dV_mag*h_u # Mixture
    V_plus = V + 0.7584838250184566*T_u + 0.7411197474551687*h_u # Mixture
    H_plus = np.cross(R,V_plus) # Angular momentum after impulse
    # Plot H locii for Tangential delta-V 
    # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='orange', alpha=1.0, label='Tangential Impulse') # Locii of H after tangential impulse
    
    # Connections from position in orbit to final position in H space
    start_points = R + np.array([hx,hy,hz]) #np.tile(H,(len(H_plus),1))
    end_points = H_plus
    # segments = np.hstack([start_points, end_points]).reshape((-1, 2, 3))
    segments = np.hstack([start_points[0,:], end_points[0,:]]).reshape((-1, 2, 3)) # Only first point
    # # Create the Line3DCollection object
    # lc = Line3DCollection(segments, colors='orange', linewidths=1)
    # # Add the collection to the axes
    # ax.add_collection(lc)
    
    # # Find angle between r and h_plus vectors
    # cos_delta = np.einsum('ij,ij->i',R,H_plus)
    # delta = np.arccos(cos_delta) 
    # # delta=90 deg regardless of dV
    
    
    # Distance metric from original H
    dh_vec = H_plus - np.tile(H, (len(H_plus),1))
    dh = np.linalg.norm(dh_vec,axis=-1)
    dh_r = dh/np.linalg.norm(R,axis=-1) # == dV_mag
    
    
    # Plot reachable surface
    if plot_reachability:
        levels = np.linspace(0,1,10)
        # levels = np.array([0.,0.7584838250184566/dV_mag, 1.0])
        
        # +ve T, +ve N
        array_list = []
        for c_T in levels:
            # c_T = % of delta-V in tangential direction
            # dV_t = c_T*dV_mag
            # dV_n = ???*dV_mag
            # dV_mag**2 = (c_T*dV_mag)**2 + (c_N*dV_mag)**2 
            # c_N*dV_mag = sqrt( dV_mag**2 - (c_T*dV_mag)**2  )
            # c_N = 1.0 - c_T # % delta-V in Normal direction
            c_N = np.sqrt( dV_mag**2 - (c_T*dV_mag)**2 )/dV_mag
            
            V_plus = V + c_T*dV_mag*T_u + c_N*dV_mag*h_u # Mixture
            H_plus = np.cross(R,V_plus) # Angular momentum after impulse
            array_list.append(H_plus)
            # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='blue', alpha=1.0, label='+ve T, +ve N') # Locii of H after tangential impulse
            
        points = np.concatenate(array_list, axis=0) # All points
        hull = ConvexHull(points) # convex hull
        if color is None:
            c = 'blue'
        else:
            c = color
        s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                        color=c, alpha=0.2, edgecolor=None, label='Access Region +ve T, +ve N')
        
        # Distance metric from original H
        dh_vec = H_plus - np.tile(H, (len(H_plus),1))
        dh = np.linalg.norm(dh_vec,axis=-1)
        dh_r = dh/np.linalg.norm(R,axis=-1) # == dV_mag
        
        # +ve T, -ve N
        array_list = []
        for c_T in levels:
            # c_T = % of delta-V in tangential direction
            c_N = np.sqrt( dV_mag**2 - (c_T*dV_mag)**2 )/dV_mag
            V_plus = V + c_T*dV_mag*T_u - c_N*dV_mag*h_u # Mixture
            H_plus = np.cross(R,V_plus) # Angular momentum after impulse
            array_list.append(H_plus)
            # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='green', alpha=1.0, label='+ve T, -ve N') # Locii of H after tangential impulse
        points = np.concatenate(array_list, axis=0) # All points
        hull = ConvexHull(points) # convex hull
        if color is None:
            c = 'green'
        else:
            c = color
        s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                        color=c, alpha=0.2, edgecolor=None, label='Access Region +ve T, -ve N')
        
        # -ve T, +ve N
        array_list = []
        for c_T in levels:
            # c_T = % of delta-V in tangential direction
            c_N = np.sqrt( dV_mag**2 - (c_T*dV_mag)**2 )/dV_mag
            V_plus = V - c_T*dV_mag*T_u + c_N*dV_mag*h_u # Mixture
            H_plus = np.cross(R,V_plus) # Angular momentum after impulse
            array_list.append(H_plus)
            # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='orange', alpha=1.0, label='-ve T, +ve N') # Locii of H after tangential impulse
        points = np.concatenate(array_list, axis=0) # All points
        hull = ConvexHull(points) # convex hull
        if color is None:
            c = 'orange'
        else:
            c = color
        s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                        color=c, alpha=0.2, edgecolor=None, label='Access Region -ve T, +ve N')
        
        # -ve T, -ve N
        array_list = []
        for c_T in levels:
            # c_T = % of delta-V in tangential direction
            c_N = np.sqrt( dV_mag**2 - (c_T*dV_mag)**2 )/dV_mag
            V_plus = V - c_T*dV_mag*T_u - c_N*dV_mag*h_u # Mixture
            H_plus = np.cross(R,V_plus) # Angular momentum after impulse
            array_list.append(H_plus)
            # ax.plot(H_plus[:,0],H_plus[:,1],H_plus[:,2], color='red', alpha=1.0, label='-ve T, -ve N') # Locii of H after tangential impulse
        points = np.concatenate(array_list, axis=0) # All points
        hull = ConvexHull(points) # convex hull
        if color is None:
            c = 'red'
        else:
            c = color
        s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                        color=c, alpha=0.2, edgecolor=None, label='Access Region -ve T, -ve N')
    
    
    
    
    # Fix axis aspect ratio
    # Get the current axis ranges
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    # Get max 
    max_val = max([max(xlim),max(ylim),max(zlim)])
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.set_box_aspect([1,1,1])
    
    # Add legend
    ax.legend()
    
    
    return fig,ax

def geometry_of_solution(H1,H2,Htx,r1_mag,r2_mag,dV,orb1,orb2):
    
    
    # Orthonormal Basis
    # Create basis aligned with H1,H2
    
    # Orthonormal Basis B1 (v1,v2)
    # Apply Gram-Schmidt Process to find orthonormal set
    # Given basis {x1,x2}
    v1 = H1 # First basis # Let v1 = x1 = H1
    v2 = H2 - (np.dot(H2,v1)/np.dot(v1,v1))*v1 # Second basis Let v2 = x2 - proj_v1_x2 = x2 - ((x2.v1)/(v1.v1))V1
    # Compute unit vectors
    v1 = v1/np.linalg.norm(v1,axis=-1)
    v2 = v2/np.linalg.norm(v2,axis=-1)
    v3 = np.cross(v1,v2) # Complete the orthogonal set
    
    # Transformation B1->B0
    # 
    # A = [H1, H2, H1xH2]

    
    # Basis Transformation B2 (u1,u2)
    # B1 = {v1, v2, v3} Orthonormal basis aligned with H1
    # B2 = { (1,0,0), (0,1,0), (0,0,1) } Algined basis
    
    # Transformation matrix A: B2->B1
    # A = [v1, v2, v3] 
    A = np.column_stack((v1,v2,v3))
    # np.matmul(A,np.array([1,0,0]).T) - v1 # A*[1,0,0] = u1
    # np.matmul(A,np.array([0,1,0]).T) - v2 # A*[0,1,0] = u2
    # np.matmul(A,np.array([0,0,1]).T) - v3 # A*[0,0,1] = u3
    
    # Inverse transformation Ainv: B1-B2
    Ainv = np.linalg.inv(A)
    
    # Characteristic parameters
    h1 = np.linalg.norm(H1,axis=-1) # Magnitude of H1
    h2 = np.linalg.norm(H2,axis=-1) # Magnitude of H2
    # phi = np.arccos( np.einsum('ij,ij->i',H1,H2) )
    cos_phi = np.dot(H1,H2)/(h1*h2); phi = np.arccos(np.clip(cos_phi, -1, 1)); phi = np.mod(phi, 2*np.pi) # Transfer angle (Wrap to 2pi)
    rp1 = orb1['a']*(1-orb1['e']); ra1 = orb1['a']*(1+orb1['e']) # Obrit 1
    rp2 = orb2['a']*(1-orb2['e']); ra2 = orb2['a']*(1+orb2['e']) # Obrit 2
    
    # Possible atx
    atx_min = (rp1+rp2)/2; etx_min = abs((rp1-rp2)/(rp1+rp2)); htx_min = np.sqrt(mu*atx_min*(1-etx_min**2)) # Peri-peri transfer
    atx_max = (ra1+ra2)/2; etx_max = abs((ra1-ra2)/(ra1+ra2)); htx_max = np.sqrt(mu*atx_max*(1-etx_max**2)) # Peri-peri transfer
    atx_hoh = (r1_mag+r2_mag)/2; etx_hoh = abs((r1_mag-r2_mag)/(r1_mag+r2_mag)); htx_hoh = np.sqrt(mu*atx_hoh*(1-etx_hoh**2)) # Hohmann transfer betweer r1 r2
    # TODO: a_min. Look at S6.1 and S11.1 of Battin
    # TODO: Compute bounds on parameter pmin, pmax for given geometry
    # See Line 1203 or OrbitToOrbit - references (eq 22,23 of McCue, 1963)
    # Also eq 6.12 of Batin: pm = 2*(r1*r2/c) for 180 deg transfer
    pm = 2*r1_mag*r2_mag/(r1_mag+r2_mag); htx_m = np.sqrt(pm*mu) # Min p for 180 deg transfer
    # Fundamental ellipse (min eccentricity)
    pF = pm*(r1_mag+r2_mag)/(r1_mag+r2_mag) # For 180 deg transfer pF = pm
    
    # atx_m = (r1_mag+r2_mag + (r_mag+r2_mag) )
    
    if ra1 < rp2:
        # Orb1 < Orb2
        # Transfer from peri1 tp apo2
        atx_max1 = (rp1 + ra2)/2; etx_max = abs((rp1-ra2)/(rp1+ra2)); htx_max = np.sqrt(mu*atx_max*(1-etx_max**2)) 
    elif ra2 < rp1:
        # Orb 2 < Orb 1
        # Transfer from peri2 to apo1
        atx_max1 = (rp2 + ra1)/2; etx_max = abs((rp2-ra1)/(rp2+ra1)); htx_max = np.sqrt(mu*atx_max*(1-etx_max**2))
    else:
        # Overlapping orbits.
        print("Overlapping orbits")
        # pdb.set_trace()
    
    
    # Optimal plane change angle
    # With htx = htx_hoh, find alpha that minimizes dV
    # Solve equation
    
    def f_optimal_pc(alpha, r1, h1, r2, h2, phi, htx):
        ''' Optimal plane change angle for nodal transfer '''
        
        # Delta-H magnitudes
        dH1 = np.sqrt(h1**2 + htx**2 - 2*h1*htx*np.cos(alpha) ) 
        dH2 = np.sqrt(h2**2 + htx**2 - 2*h2*htx*np.cos(phi-alpha) ) 
        
        # Compute f(alpha) - necessary condition for optimality
        # f = h1*np.sin(alpha)/(r1*dH1) - h2*np.sin(phi-alpha)/(r2*dH2) # = 0
        # xby dH1, dH2 to avoid singularities
        f = dH2*h1*np.sin(alpha)/(r1) - dH1*h2*np.sin(phi-alpha)/(r2) # = 0
        
        return f
    
    alpha_sol, iters = chandrupatla(f_optimal_pc,0,phi, 
                          args=(r1_mag, h1, r2_mag, h2, phi, htx_hoh),
                          return_iter=True,
                          )
    
    # Transform H1,H2,Htx into new basis
    # B1 (orthonomal)
    H1uv = np.matmul(Ainv,H1)
    H2uv = np.matmul(Ainv,H2)
    Htxuv = np.matmul(Ainv,Htx)
    
    # Create the figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Subplot 1: 3D plot (left)
    
    # Plot H1 position
    ax1.scatter(*H1, color='b', s=40, alpha=0.3, label='H1')
    ax1.plot([0, H1[0]],[0, H1[1]],[0, H1[2]],'-b')
    # Plot H2 position
    ax1.scatter(*H2, color='g', s=40, alpha=0.3, label='H2')
    ax1.plot([0, H2[0]],[0, H2[1]],[0, H2[2]],'-g')
    # Plot Htx position
    ax1.scatter(*Htx, color='r', s=40, alpha=0.3, label='Htx')
    ax1.plot([0, Htx[0]],[0, Htx[1]],[0, Htx[2]],'-r')
    # dH
    ax1.plot([H2[0], H1[0]],[H2[1], H1[1]],[H2[2], H1[2]],'-k', label='dH = H2-H1') # H2-H1
    ax1.plot([Htx[0], H1[0]],[Htx[1], H1[1]],[Htx[2], H1[2]],'-k', label='dH1 = Htx-H1') # Htx-H1
    ax1.plot([H2[0], Htx[0]],[H2[1], Htx[1]],[H2[2], Htx[2]],'-k', label='dH2 = H2-Htx') # H2-Htx
    
    
    ax1.set_title('3D Plot')
    
    # Fix axis aspect ratio
    # Get the current axis ranges
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    
    # Get max 
    max_val = max([max(xlim),max(ylim),max(zlim)])
    ax1.set_xlim(-max_val, max_val)
    ax1.set_ylim(-max_val, max_val)
    ax1.set_zlim(-max_val, max_val)
    ax1.set_box_aspect([1,1,1])
    # Add legend
    ax1.legend()
    
    # Subplot 2: 2D plot (right)
    
    
    # Plot H1 position
    ax2.plot(H1uv[1],H1uv[0], 'ob', markersize=10, alpha=0.3, label='H1')
    ax2.plot([0, H1uv[1]],[0, H1uv[0]],'-b')
    # Plot H2 position
    ax2.plot(H2uv[1],H2uv[0], 'og', markersize=10, alpha=0.3, label='H2')
    ax2.plot([0, H2uv[1]],[0, H2uv[0]],'-g')
    # Plot Htx position
    ax2.plot(Htxuv[1],Htxuv[0], 'or', markersize=10, alpha=0.3, label='Htx')
    ax2.plot([0, Htxuv[1]],[0, Htxuv[0]],'-r')
    
    # dH
    ax2.plot([H2uv[1], H1uv[1]],[H2uv[0], H1uv[0]],'-k', label='dH = H2-H1') # H2-H1
    ax2.plot([Htxuv[1], H1uv[1]],[Htxuv[0], H1uv[0]],'-k', label='dH1 = Htx-H1') # Htx-H1
    ax2.plot([H2uv[1], Htxuv[1]],[H2uv[0], Htxuv[0]],'-k', label='dH2 = H2-Htx') # H2-Htx
    
    # Create grid of uv to compute delta-V
    # 
    # Plot dV = dH1/r1 + dH2/r2
    # Note: if purely plane change, then this is equal to delta-V
    # Note: the value of dH1/r1 + dH2/r2 at the Htx solution does match the dV
    # There are some locations that result in lower values of dH1/r1 + dH2/r2.
    
    # TODO: for any arbitrary Htx, find R, V, dV
    
    # H = RxV
    
    #  y = v1
    #  |
    #  - x = v2
    v1_min = min(H1uv[0],H2uv[0])
    # v1_max = max(H1uv[0],H2uv[0])
    v1_max = max([h1,h2,htx_max]) # Vertical range
    v2_min = min(H1uv[1],H2uv[1])
    v2_max = max([H1uv[1],H2uv[1] ])
    x_vec = np.linspace(v2_min,v2_max, 300)
    y_vec = np.linspace(v1_min,v1_max, 300)
    X, Y = np.meshgrid(x_vec, y_vec)
    
    # Compute dH1 at each gridpoint
    dH1 = np.sqrt( (X-H1uv[1])**2 + (Y-H1uv[0])**2  )
    dH2 = np.sqrt( (X-H2uv[1])**2 + (Y-H2uv[0])**2  )
    Z = dH1/r1_mag + dH2/r2_mag
   
    
    # Interpolate value at Htx
    points = np.array([X.flatten(), Y.flatten()]).T
    values = Z.flatten()
    Ztx = float(griddata(points, values, np.array([Htxuv[1],Htxuv[0]]), method='linear'))
    
    Z[ X**2 + Y**2 < htx_min**2 ] = np.nan
    Z[ X**2 + Y**2 > htx_max**2 ] = np.nan
    ax2.pcolormesh(X, Y, Z, cmap='jet', 
                    # norm=colors.LogNorm(vmin=np.min(Z), vmax=np.max(Z))
                    )
    
    # Plot circles - max/min range of htx
    th2 = np.arctan2(H2uv[0],H2uv[1])
    theta = np.linspace(th2, np.pi/2, 100) # 100 points between 0 and 2*pi
    ax2.plot(htx_min*np.cos(theta), htx_min*np.sin(theta), color='blue')
    ax2.plot(htx_max*np.cos(theta), htx_max*np.sin(theta), color='blue')
    ax2.plot(h1*np.cos(theta), h1*np.sin(theta), color='k')
    ax2.plot(h2*np.cos(theta), h2*np.sin(theta), color='k')
    ax2.plot(htx_hoh*np.cos(theta), htx_hoh*np.sin(theta), color='r')
    ax2.plot(htx_m*np.cos(theta), htx_m*np.sin(theta), color='m')
    
    
    
    # Compute values at htx = htx_hoh, theta < th2
    # htx_vec = htx_hoh*np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta)) ])
    dH1_vec = np.sqrt( (htx_hoh*np.cos(theta)-H1uv[1])**2 + (htx_hoh*np.sin(theta)-H1uv[0])**2  )
    dH2_vec = np.sqrt( (htx_hoh*np.cos(theta)-H2uv[1])**2 + (htx_hoh*np.sin(theta)-H2uv[0])**2  )
    Z_vec = dH1_vec/r1_mag + dH2_vec/r2_mag
    ind = Z_vec.argmin() # Min value
    ax2.plot(htx_hoh*np.cos(theta[ind]), htx_hoh*np.sin(theta[ind]), '*r', markersize=10, label='Prediction')
    # TODO: find an algebraic equation to compute this minima
    ax2.plot(htx_hoh*np.cos(np.pi/2 - alpha_sol), htx_hoh*np.sin(np.pi/2 - alpha_sol), '*g', markersize=20, label='Optimum Geometry')
    
    # Plot the minimum
    # ind = Z.indmin()
    print("Min(Z) = {}".format(np.nanmin(Z)))
    print("Z(Htx) = {}".format(Ztx))
    print("Pred Z = {}".format(Z_vec.min()))
    print("Actual delta-V = {}".format(dV))
    print("Pred error = {:2f} %".format(100*(Z_vec.min()-dV)/dV))
    
    print("Htx_uv = {}", Htxuv)
    
    
    # Axes
    ax2.set_xlabel('v2')
    ax2.set_ylabel('v1')
    ax2.axis('equal')
    ax2.legend()
    
    
    return


#%%

def vector_projection(a, b):
    """
    Calculates the vector projection of vector a onto vector b.

    Args:
        a (list or np.array): The vector to be projected.
        b (list or np.array): The vector onto which 'a' is projected.

    Returns:
        np.array: The projection vector of 'a' onto 'b'.
    """
    a = np.array(a)
    b = np.array(b)
    if np.all(b == 0):
        raise ValueError("The vector onto which to project (b) cannot be a zero vector.")
        
    dot_product = np.dot(a, b)
    # The square of the magnitude of b is the dot product of b with itself
    magnitude_b_squared = np.dot(b, b) 
    
    # Calculate the scalar component and multiply by the unit vector of b
    projection = (dot_product / magnitude_b_squared) #* b
    
    return projection

def angle_between_vectors_3d(v1, v2):
    """
    Returns the angle in radians between 3D vectors 'v1' and 'v2'.
    The angle is in the range [0, pi].
    """
    cos_angle = np.dot(v1, v2)
    sin_angle = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sin_angle, cos_angle)

#%%

if __name__ == "__main__":
    
    
    
    # Constants
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    
    # Define an orbit 1
    orb1 = {'a':Re+200,
           'e':0.0,
           'i':np.deg2rad(0),
           'om':np.deg2rad(0.),
           'w':np.deg2rad(60.),
           'M':np.deg2rad(0.),
           'units':'km'}

    orb2 = {'a':Re+600,
           'e':0.2, # 0.2148
           'i':np.deg2rad(10),
           'om':np.deg2rad(0.),
           'w':np.deg2rad(0.),
           'M':np.deg2rad(0.),
           'units':'km'}
    
    # Get H vector of orbit 1
    h = np.sqrt(mu*orb1['a']*(1 - orb1['e']**2)) # Magnitude
    hx = h*np.sin(orb1['i'])*np.sin(orb1['om']) # x component
    hy = -np.sin(orb1['i'])*np.cos(orb1['om'])*h # y component
    hz = np.cos(orb1['i'])*h # z component
    H1 = np.array([hx,hy,hz]) # Vector
    
    
    # Get H vector of orbit 2
    h = np.sqrt(mu*orb2['a']*(1 - orb2['e']**2)) # Magnitude
    hx = h*np.sin(orb2['i'])*np.sin(orb2['om']) # x component
    hy = -np.sin(orb2['i'])*np.cos(orb2['om'])*h # y component
    hz = np.cos(orb2['i'])*h # z component
    H2 = np.array([hx,hy,hz]) # Vector
    
    
    
    
    # Optimal orbit-to-orbit transfer
    prob = OrbitToOrbitProblem(orb1, orb2, mu) # Initialize problem
    prob.solve(decode=True) # Solve problem and decode solution
    # prob.plot_porkchop() # Plot porkchop
    orb_tx = prob.result.txorb
    orb_tx['M'] = 0
    # Extract delta-Vs
    dV = prob.result.fun # Total delta-V
    dV1 = prob.result.dV1
    dV2 = prob.result.dV2
    Htx = np.cross(prob.result.r1, prob.result.vtx1) # H vector of transfer
    r1 = prob.result.r1 # Position 
    r2 = prob.result.r2 # Position 
    r1_mag = np.linalg.norm(r1 ,axis=-1)
    r2_mag = np.linalg.norm(r2 ,axis=-1)
    
    # Compute distance metrics for optimal solution
    dH1 = np.linalg.norm(Htx-H1 ,axis=-1) # H1 - Htx
    dH2 = np.linalg.norm(H2 - Htx ,axis=-1) # Htx - H2
    dist_H1 = dH1/np.linalg.norm(r1 ,axis=-1) # dV1 = dH1/r1
    dist_H2 = dH2/np.linalg.norm(r2 ,axis=-1) # dV2 = dH2/r2
    dist_tot = dist_H1 + dist_H2 # Total
    
    # Optimal solution
    # dV = dV1 + dV2 = dH1/r1 + dH2/r2
    # See if can predict min dV solution
    # Min dV = min dH1/r1 + dH2/r2
    #
    # If r1, r2 are fixed, then min dH1 and dH2
    # 
    #
    #                    * H2
    #           x       /
    # H1 *             /
    #     \           /
    #      \         /
    
    # TODO: Find midpoint between H1, H2
    # Why does this give larger dV???

    
    # Tangential and Normal components of dV1
    dV1_vec = prob.result.vtx1 - prob.result.v1 # Delta-V in ECI
    dV1_n = vector_projection(dV1_vec, H1/np.linalg.norm(H1,axis=-1)) # Projection of dV1 on H1
    dV1_t = np.sqrt( dV1**2 - dV1_n**2 ) # Tangential component
    
    # Tangential and Normal components of dV1
    dV2_vec = prob.result.vtx2 - prob.result.v2 # Delta-V in ECI
    dV2_n = vector_projection(dV2_vec, H2/np.linalg.norm(H2,axis=-1)) # Projection of dV1 on H1
    dV2_t = np.sqrt( dV2**2 - dV2_n**2 ) # Tangential component
    
    # FIXME: dHr and dHt components
    # Project h2 on h1
    # Find projection of h2 on h1, then subtract h1
    # proj_a_on_b = (a.b)/(b.b) 
    # b = h1
    # a = h2
    
    # proj_a_b = np.dot(dV1_vec,H1)/np.dot(H1,H1) #abs((dV1_vec[0]*H1[0] + hy1*H1[1] + hz1*H1[2])/(H1[0]**2 + H1[1]**2 + H1[2]**2)) # Scalar component a.b/b.b
    # # dh_n = np.sqrt((proj_a_b*H1[0] - hx1)**2 + (proj_a_b*hy1 - hy1)**2 + (proj_a_b*hz1 - hz1)**2)
    # # dh_t = np.sqrt( dH**2 - dh_n**2 )
    
    
    
    # Print results
    print("\nActual delta-V: {} km/s".format(dV))
    print("Predicted delta-V = dH1/r1 + dH2/r2 = {}".format(dist_tot))
    print("Error (%) = {}\n".format(100*(dist_tot-dV)/dV))
    
    
    print("dV1t = {}  dV1n = {}".format(dV1_t,dV1_n))
    print("dV2t = {}  dV2n = {}".format(dV2_t,dV2_n))
    
    print("dH1/dH = {}".format(dist_H1/dist_tot))
    print("dH2/dH = {}".format(dist_H2/dist_tot))
    print("dH1/dH2 = {}".format(dist_H1/dist_H2))
    
    
    # Angles between vectors
    print("angle(H1,H2) = {} deg".format(np.rad2deg(angle_between_vectors_3d(H1, H2))))
    print("angle(Htx,H1) = {} deg".format(np.rad2deg(angle_between_vectors_3d(Htx, H1))))
    print("angle(H2,Htx) = {} deg".format(np.rad2deg(angle_between_vectors_3d(H2, Htx))))
    print("angle(H1,Htx)/angle(H1,H2) = {}\n".format(np.rad2deg(angle_between_vectors_3d(Htx, H1)) / np.rad2deg(angle_between_vectors_3d(H1, H2))) )

    
    # Delta-V mag
    dV_mag = dV/2 # Delta-V (km/s)
    
    
    # # Plot orbit in H space
    # fig,ax = plot_orbit_h_reachability(orb1, dV1, color='blue') # Orbit 1
    # # Plot orbit in H space
    # fig,ax = plot_orbit_h_reachability(orb2, dV2, fig,ax, color='red') # Orbit 2
    # # Plot optimal transfer orbit
    # fig,ax = plot_orbit_h_reachability(orb_tx, dV_mag, fig,ax, plot_reachability=False, color='green') # Tx orbit
    # ax.plot([H1[0], H1[0] + r1[0]], [H1[1], H1[1] + r1[1]], [H1[2], H1[2] + r1[2]], '-b', alpha=1.0, label=' ') # At H
    
    
    # Analyze solution
    geometry_of_solution(H1,H2,Htx,r1_mag,r2_mag,dV,orb1,orb2)
    
    