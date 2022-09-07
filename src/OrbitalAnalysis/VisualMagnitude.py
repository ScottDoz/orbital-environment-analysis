# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:05:03 2022

@author: scott

Visual Magnitude Module
-----------------------

Methods to compute the visual magnitude of a satellite from a groundstation.

"""

# References:
# - Astronomical Algorithms ch 41

# See also: https://github.com/danielkucharski/SatLightPollution/blob/master/SatLightPollution/Source.cpp


#%% Visual Mangitude

def compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,include_airmass=True):
    
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
    satv = dftopo[['Sat.X','Sat.Y','Sat.Z']].to_numpy() # Sun position
    
    # Compute distances
    dsat0 = np.linalg.norm(sunv - satv, axis=-1)/AU # Distance Sat to Sun (AU)
    dsat = dftopo['Sat.R'].to_numpy()/AU            # Distance Obs to Sat (AU)
    Rsat = (Rsat/1000)/AU # Radius of satellite in AU
    
    # Phase function for Lambertian sphere
    alpha = dftopo['Sat.alpha'].to_numpy() # Solar phase angle of satellite (rad)
    valpha = (1+np.cos(alpha))/2
    # Exclude airmass
    if include_airmass==False:
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



