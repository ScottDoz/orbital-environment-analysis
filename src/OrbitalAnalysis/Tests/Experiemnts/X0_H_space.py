# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:24:50 2025

@author: scott

Goal: Explore fundamental properties of specific orbital angular momentum space.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from OrbitalAnalysis.SatelliteData import *
from astrologistics.Functions import sv_from_coe

#%%

def test_covariance_transform():
    
    # Define a simple covariance in RTN coords and transform to ECI
    
    # Constants
    Re = 6378.137; #Radius of Earth (km)
    J2 = 1.08262668e-3; #J2 constant of Earth
    mu = 3.986004418e5; # Gravitational parameter of Earth (km3s-2)
    
    # State Vector. Circular orbit 
    a = Re + 200; e = 0.1; i = np.deg2rad(40);  # Orbit shape
    om = 0.; w=0.; M=np.deg2rad(90)            # Orbit orientation
    sv = sv_from_coe(a,e,i,om,w,M, mu=mu, units='km')
    R = sv[:3]
    V = sv[3:]
    H = np.cross(R,V)
    
    # Initialization of Covariance
    # See Battin pg 678
    h_vec = np.cross(sv[:3],sv[3:]) # Angular momentum vector
    h = np.linalg.norm(h_vec)
    E = 0.2*(np.linalg.norm(sv[3:])**2) - mu/np.linalg.norm(sv[:3]) # Total enegery E = 0.5v^2 - mu/r
    
    # RSW or RTN Coordinates
    # See Ch 2 or "Orbital Data Applications for Space Objects"
    # R = radial 
    # S = along track
    # W = normal
    Ru = sv[:3]/np.linalg.norm(sv[:3], axis=-1) # Radial
    Wu = np.cross(sv[:3],sv[3:])/np.linalg.norm(np.cross(sv[:3],sv[3:]) ) # Normal
    Su = np.cross(Wu,Ru) # Along track
    
    # Transformation matrix
    # See pg 257 of Tapley
    M_RSW_to_ECI = np.column_stack((Ru, Su, Wu)) # RSW to ECI [R S W]
    M_ECI_to_RSW = M_RSW_to_ECI.T # ECI to RSW
    # Block matrix for full transformation
    # [r]          [M 0][r]
    # [v]RTN     = [0 M][v]ECI
    A_RSW_to_ECI = np.block([[M_RSW_to_ECI, np.zeros((3,3))],[np.zeros((3,3)),M_RSW_to_ECI]]) # Block matrix
    # X_ECI = M_RSW_to_ECI*X_RSW
    # P_ECI = A_RSW_to_ECI*P_RSW*A_RSW_to_ECI.T
    
    # Define initial coviarance - diagonal in RSW
    P0_RSW = np.diag((10,100,10,0.1,0.1,0.1))
    # Transform to ECI
    # See pg 257 of Tapley
    
    # Transforming covariance
    P_ECI = A_RSW_to_ECI @ P0_RSW @ A_RSW_to_ECI.T
    
    
    # Covariance of H = R X V
    # Even if both R and V are gaussian, the resulting H will be non-gaussian.
    
    # Linearized approximation
    # If variances in R and V are small wrt R and V, we can estimate the 
    # covariance of H as a linear approximation using the jacobian
    # P_H_ECI = H_covariance(sv[:3], sv[3:], P_ECI) 
    H_mean_lin, P_H_lin = compute_H_covariance(R, V, P_ECI, method='linear')
    H_mean_mc, P_H_mc, samples = compute_H_covariance(R, V, P_ECI, method='mc', n_samples=100000, random_state=42)
    r_samples, v_samples = zip(*samples) # Unzip Monte Carlo samples
    r_samples = np.array(r_samples)
    v_samples = np.array(v_samples)
    # Compute H of samples
    H_samples = np.cross(r_samples, v_samples)

    
    
    # Plot
    # Prepare 3D plot
    fig = plt.figure(figsize=(8, 8))
    
    # --- Left axis: R, V, and position error ellipsoid ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_box_aspect([1,1,1])
    
    # Satellite
    ax1.scatter(*R, color='k', s=60, label='Satellite')
    # ax1.plot([0, R[0]],[0, R[1]],[0, R[2]],'-b', label='R (Position)') # Position vector R
    ax1.plot([R[0], R[0]+V[0]],[R[1], R[1]+V[1]],[R[2], R[2]+V[2]],'-g', label='V (Velocity)') # Velocity vector V
    # Draw monte carlo sample points
    ax1.scatter(r_samples[:,0],r_samples[:,1],r_samples[:,2], color='orange', s=1, label='MC samples')
    
    
    # 1-sigma position error ellipsoid
    draw_error_ellipsoid(ax1, R, P_ECI[:3,:3], color='orange', alpha=0.9)
    # Format axes
    ax1.set_title('Satellite Position and Velocity')
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    lim = np.linalg.norm(R)*1.5
    # ax1.set_xlim(-lim, lim)
    # ax1.set_ylim(-lim, lim)
    # ax1.set_zlim(-lim/2, lim/2)
    ax1.set_aspect('equal')
    ax1.legend()
    
    
    # --- Right axis: Angular momentum H ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_box_aspect([1,1,1])
    
    # Origin
    # ax2.scatter(0,0,0, color='k', s=40)
    # Angular momentum vector H
    # ax2.quiver(0,0,0,*H, color='r', arrow_length_ratio=0.05, linewidth=2, label='H (Angular Momentum)')
    ax2.plot([0, H[0]],[0, H[1]],[0, H[2]],'-r', label='H (Angular Momentum)')
    # Optional: show satellite position for reference
    ax2.scatter(*H, color='b', s=40, alpha=0.3, label='Satellite Position')
    
    # Draw monte carlo sample points
    ax2.scatter(H_samples[:,0],H_samples[:,1],H_samples[:,2], color='orange', s=1, label='MC samples')
    
    # 1-sigma position error ellipsoid
    draw_error_ellipsoid(ax2, H, P_H_lin, color='orange', alpha=0.5)
    draw_error_ellipsoid(ax2, H, P_H_mc, color='red', alpha=0.5)
    
    ax2.set_title('Angular Momentum Vector')
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_zlabel('Z [km]')
    lim = np.linalg.norm(H)*1.5
    # ax2.set_xlim(-lim, lim)
    # ax2.set_ylim(-lim, lim)
    # ax2.set_zlim(-lim/2, lim/2)
    ax2.set_aspect('equal')
    ax2.legend()
    
    plt.show()

    
    return

def S(a):
    a = np.asarray(a).flatten()
    
    Sa = np.array([[0,   -a[2],  a[1]],
                   [a[2],   0,  -a[0]],
                   [-a[1], a[0],   0]])
    
    return Sa 

def H_covariance(r, v, P):
    """
    r, v: 3-vectors (numpy)
    P6: 6x6 covariance of state [r; v] in same coords
    returns P_H (3x3)
    """
    
    # P_H ≈ J Px J.T,
    # where
    # J = [∂H/∂r ∂H/∂v] = [-S(v) S(r)]
    # where S(r) and S(v) are skew cross-product matrices
    #        [0   -rz   ry]           [0   -vz   vy]
    # S(r) = [rz    0  -rx]    S(v) = [vz    0  -vx]
    #        [-ry  rx    0]           [-vy  vx    0]
    #
    
    Sr = S(r)
    Sv = S(v)
    J = np.hstack((-Sv, Sr))   # 3x6
    P_H = J @ P @ J.T
    return P_H

def compute_H_covariance(r, v, P, method='linear', n_samples=100000, random_state=None):
    """
    Compute the covariance of the angular momentum H = r x v.

    Parameters
    ----------
    r : array_like, shape (3,)
        Nominal position vector.
    v : array_like, shape (3,)
        Nominal velocity vector.
    P6 : array_like, shape (6,6)
        6x6 covariance of the state [r; v].
    method : str, optional
        'linear' for Jacobian linearization (default), 'mc' for Monte Carlo.
    n_samples : int, optional
        Number of Monte Carlo samples if method='mc' (default 100000).
    random_state : int or None
        Seed for reproducibility in Monte Carlo.

    Returns
    -------
    H_mean : ndarray, shape (3,)
        Expected value of angular momentum (nominal cross product).
    P_H : ndarray, shape (3,3)
        Covariance of H.
    """
    r = np.asarray(r).flatten()
    v = np.asarray(v).flatten()
    P = np.asarray(P)
    
    # Nominal H
    H_mean = np.cross(r, v)
    
    if method == 'linear':
        # Skew-symmetric matrices
        def S(a):
            a = np.asarray(a).flatten()
            return np.array([[0, -a[2], a[1]],
                             [a[2], 0, -a[0]],
                             [-a[1], a[0], 0]])
        
        J = np.hstack((-S(v), S(r)))  # 3x6 Jacobian
        P_H = J @ P @ J.T
        return H_mean, P_H
    
    elif method == 'mc':
        rng = np.random.default_rng(random_state)
        # Draw samples from 6D Gaussian
        samples = rng.multivariate_normal(np.hstack([r,v]), P, size=n_samples)
        r_samples = samples[:, :3]
        v_samples = samples[:, 3:]
        H_samples = np.cross(r_samples, v_samples)
        P_H = np.cov(H_samples, rowvar=False)
        H_mean = np.mean(H_samples, axis=0)
        
        # Collect samples
        samples_set = zip(r_samples, v_samples)
        
        return H_mean, P_H, samples_set
    
    else:
        raise ValueError("Invalid method. Choose 'linear' or 'mc'.")

def draw_error_ellipsoid(ax, X, P, n_points=50, color='orange', alpha=0.5, rstride=2, cstride=2):
    """
    Draw a 1-sigma error ellipsoid at position X with covariance P on an existing Axes3D `ax`.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        The existing 3D axes to plot on.
    X : array_like, shape (3,)
        The center of the ellipsoid.
    P : array_like, shape (3,3)
        Covariance matrix for the position.
    n_points : int, optional
        Number of points for sphere parameterization (default 50).
    color : str, optional
        Color of the ellipsoid (default 'orange').
    alpha : float, optional
        Transparency (default 0.5).
    rstride, cstride : int, optional
        Row and column stride for wireframe.
    """
    X = np.asarray(X)
    P = np.asarray(P)
    
    # Eigen-decomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(P)
    radii = np.sqrt(eigvals)  # 1-sigma semi-axis lengths

    # Parameterize unit sphere
    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points//2)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)  # shape (n_points, n_points/2, 3)

    # Scale and rotate sphere to ellipsoid
    ellipsoid = np.tensordot(sphere, np.diag(radii), axes=(2,0))
    ellipsoid = ellipsoid @ eigvecs.T
    ellipsoid += X  # translate to center

    # Plot wireframe on existing axes
    ax.plot_wireframe(ellipsoid[:,:,0], ellipsoid[:,:,1], ellipsoid[:,:,2],
                      rstride=rstride, cstride=cstride, color=color, alpha=alpha)
    
    return