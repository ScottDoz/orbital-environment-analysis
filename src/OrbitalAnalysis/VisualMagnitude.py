# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:05:03 2022

@author: scott

Visual Magnitude Module
-----------------------

Methods to compute the visual magnitude of a satellite from a groundstation.

"""

import numpy as np
import pandas as pd
import trimesh

import pdb


# References:
# - Astronomical Algorithms ch 41

# See also: https://github.com/danielkucharski/SatLightPollution/blob/master/SatLightPollution/Source.cpp


#%% Visual Mangitude

def compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,lambertian_phase_function=True):
    
    # Compute satellite apparent magnitude
    # https://www.eso.org/~ohainaut/satellites/22_ConAn_Bassa_aa42101-21.pdf
    # https://www.aanda.org/articles/aa/pdf/2020/04/aa37501-20.pdf
    #
    # m_sat = m0 - 2.5*log10(p*Rsat^2) + 5*log10(dsat0*dsat)
    #         - 2.5*log10( v(alpha0) ) + k*X
    #
    # where:
    # m0 = -26.76 is the Solar V-band magnitude at Earth
    #
    # p*Rsat^2 is the photometric crossection
    # p = satellite geometric albedo
    # Rsat = radius of the (spherical) satellite
    #
    # dsat0 = distance from satellite to sun
    # dsat = distance from observer to satellite
    #
    # # alpha0 = solar phase angle
    # v(alpha0) = correction for solar phase angle (set at 1 to remove term)
    #
    # k = extinction coefficient (mag per unit airmass) = 0.12 in V-band
    # X = 1/cos(90-El) = 1/sin(El) = airmass in the plane-parallel approximation
    # El = elevation above horizon
    
    # Note: distances should be in AU
    # see: https://www.aanda.org/articles/aa/pdf/2020/04/aa37501-20.pdf
    
    # TODO: Compute airmass using astropy.coordinates.AltAz
    # https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html
    # https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html
    
    # Use airmass to compute atmospheric extinction
    # m(X) = m0 + k*X
    # where X = airmass, m=magnitude, k=extinction coefficient (mags/airmass)
    # see: https://warwick.ac.uk/fac/sci/physics/research/astro/teaching/mpags/observing.pdf
    
    AU = 149597870.700 # Astronomical unit (km)
    
    # Extract Vectors and distances
    sunv = dftopo[['Sun.X','Sun.Y','Sun.Z']].to_numpy() # Sun position
    satv = dftopo[['Sat.X','Sat.Y','Sat.Z']].to_numpy() # Sat position
    
    # Compute distances
    dsat0 = np.linalg.norm(sunv - satv, axis=-1)/AU # Distance Sat to Sun (AU)
    dsat = dftopo['Sat.R'].to_numpy()/AU            # Distance Obs to Sat (AU)
    Rsat = (Rsat/1000)/AU # Radius of satellite in AU
    
    # Phase function for Lambertian sphere
    alpha = dftopo['Sat.alpha'].to_numpy() # Solar phase angle of satellite (rad)
    valpha = (1+np.cos(alpha))/2
    # Exclude lambertian phase function
    if lambertian_phase_function==False:
        # Set constant phase function
        valpha = 1. # To remove
    
    # Compute airmass
    el = dftopo['Sat.El'].to_numpy() # Elevation (rad)
    X = 1./np.sin(el) # airmass in the plane-parallel approximation
    
    # Compute phase
    m0 = -26.75 # Sun's apparent magnitude in V-band
    msat = m0 - 2.5*np.log10(p*(Rsat**2)) + 5*np.log10(dsat0*dsat) -2.5*np.log10(valpha) + k*X 
    
    # Remove magnitude when below horizon
    msat[el<np.deg2rad(0.1)] = np.nan
    
    # TODO: Constrain by max value
    
    return msat


#%% Flat Facet Model

# Implementation of
# Linares et al. 2020 "Space Objects Classification via Light-Curve Measurements 
# Using Deep Convolutional Neural Networks

def compute_visual_magnitude_flatfacet(dftopo,Rsat,model='sphere',p=0.25,k=0.12):
    
    
    # Load satellite 3d model
    if model == 'sphere':
        # Spherical model
        c = (0,0,0) # Center (body fixed coords)
        mesh = trimesh.primitives.Sphere(Rsat,c,3)
    
    # Parameters of mesh
    N = len(mesh.face_normals)    # Number of faces
    u_n = mesh.face_normals       # Face normal vectors (unit)
    A = mesh.area_faces           # List of face areas
    
    AU = 149597870.700 # Astronomical unit (km)
    
    # Extract Vectors and distances
    sunv = dftopo[['Sun.X','Sun.Y','Sun.Z']].to_numpy() # Sun position
    satv = dftopo[['Sat.X','Sat.Y','Sat.Z']].to_numpy() # Sat position
    
    # Normal vectors to sun and to observer
    to_sun = (sunv - satv)/ np.linalg.norm(sunv - satv, axis=-1)[:, np.newaxis]
    to_obs = -satv/ np.linalg.norm(satv, axis=-1)[:, np.newaxis]
    
    
    # Compute distances
    dsat0 = np.linalg.norm(sunv - satv, axis=-1)/AU # Distance Sat to Sun (AU)
    dsat = dftopo['Sat.R'].to_numpy()/AU            # Distance Obs to Sat (AU)
    Rsat = (Rsat/1000)/AU # Radius of satellite in AU
    
    # Loop through time steps
    F = np.zeros(len(dftopo)) # Instantiate Flux
    Csunvis = 1062 # Power/area
    for i,row in dftopo.iterrows():
        
        # Tile unit vectors. Same length as number of facets in model.
        u_sun = np.tile(to_sun[i,:],(N,1)) # Sun direction (unit) (repeated for each facet)
        u_obs = np.tile(to_obs[i,:],(N,1)) # Observer direction (unit) (repeated for each facet)
        
        # Fsun. Fraction of incident light reflected from each facet.
        # Note: use einsum to compute dot product
        Fsun = Csunvis*np.einsum('ij,ij->i', u_n, u_sun) # Eq 6
        Fsun[Fsun<0] = 0. # If angle is < pi/2 no light reflected
        # This is the normal component of solar flux incident on the facet (W/m^2)
        
        # Fobs. Solar flux at the observer from each facet.
        ptotal = 1.
        Fobs = Fsun*ptotal*A*np.einsum('ij,ij->i', u_n, u_obs)/(row['Sat.R']*1000)**2
        Fobs[Fobs<0] = 0. # If angle is < pi/2 no light reflected
        
        # Total solar flux (sum of each facet)
        vCCD = 0 # Measurement noise from CCD
        F[i] = sum(Fobs) + vCCD
    
    # Compute airmass
    el = dftopo['Sat.El'].to_numpy() # Elevation (rad)
    X = 1./np.sin(el) # airmass in the plane-parallel approximation
    
    # Compute visual magnitude
    m0 = -26.7
    msat = m0 - 2.5*np.log10(abs(F/Csunvis))
    msat +=  + k*X # Add airmass factor
    
    # Remove magnitude when below horizon
    msat[el<np.deg2rad(0.1)] = np.nan
    
    return msat

