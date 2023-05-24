# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:19:59 2022

@author: mauri
"""

import numpy as np
import matplotlib.pyplot as plt
from SatelliteData import *
from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from numpy.random import multivariate_normal

# Load data
df = load_satellites(group='all',compute_params=True)

# Extract state vector for cetrain object
NORAD = 25544
elem = df[['a','e','i','om','w','M']][df.NoradId == NORAD].iloc[0]

# Create poliastro orbit object
a = elem[0] << u.km
ecc = elem[1] << u.one
inc = elem[2] << u.deg
raan = elem[3] << u.deg
argp = elem[4] << u.deg
nu = elem[5] << u.deg
orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)

# Extract state vectors
r = np.array(orb.r)
v = np.array(orb.v)

# Compute angular momentum
h = np.cross(r,v)

# Compute basis vectors
rhat = r/(np.linalg.norm(r))
what = h/(np.linalg.norm(h))
shat = np.cross(what,rhat)

# Form RSW transformation matrix and transform to RSW
M_ijktorsw = np.stack([rhat, shat, what], axis=0) 
M_rswtoijk = np.stack([rhat, shat, what], axis=1)
r_rsw = np.dot(M_ijktorsw,r)
v_rsw = np.dot(M_ijktorsw,v)

# Initial covariance matrix
# Note: estimate initial covariances from lookup table (Table 2 of
# Assessment and Categorization of TLE Orbit Errors for the US SSN Catalogue)
# For velocity components, add in made up values
P = np.diag([.107, .308, .169, .1, .1, .1]) #Covariance matrix
mean = np.concatenate([r_rsw,v_rsw]) #State vector

# Example from textbook:
#generate random points
A = multivariate_normal(mean=mean, cov=P, size=10000)
rs = A[:,:3]
vs = A[:,3:]
hs = np.cross(rs,vs)

# Plotting angular momentum in RSW frame
#TODO: Add labels to the plots, look at additional plots
# Look into statistical tests to see if covariance is gaussian
fig = plt.figure()
ax0 = fig.add_subplot(1,2,1,projection='3d')
ax1 = fig.add_subplot(1,2,2,projection='3d')
ax0.set_box_aspect((np.ptp(rs[:,0]), np.ptp(rs[:,1]), np.ptp(rs[:,2])))
ax1.set_box_aspect((np.ptp(hs[:,0]), np.ptp(hs[:,1]), np.ptp(hs[:,2])))
ax0.scatter(rs[:,0],rs[:,1],rs[:,2],c = 'b', marker='.')
ax1.scatter(hs[:,0],hs[:,1],hs[:,2],c = 'b', marker='.')

# Computing mean and covariance of the sample points
meanh = np.mean(hs,axis=0)
Ph = np.cov(hs.T)
eigen_values, eigen_vectors = np.linalg.eig(Ph)
projection_matrix = (eigen_vectors.T[:][:2]).T
X_pca = X.dot(projection_matrix)

# Transform covariance matrix
# Ph = M*P*M.T


