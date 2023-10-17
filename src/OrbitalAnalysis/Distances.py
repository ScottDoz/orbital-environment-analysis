# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:27:39 2022

@author: scott

Distance Metrics
----------------

Define a number of metrics used to quantify the "distance" between pairs of orbits.
i.e. a measure of the closeness of two orbits.

"""

# Standard imports
import numpy as np
import pandas as pd

# Module imports
# from sr_tools.Astrodynamics.Functions import sv_from_coe

import pdb

#%% Euclidean Distance metrics ------------------------------------------------

def dist_dH(x1,x2):
    '''
    Euclidean distance in specific anglular momentum space (Hx,Hy,Hz)
    '''
    
    # Extract elements
    hx1,hy1,hz1 = x1.T
    hx2,hy2,hz2 = x2.T
    
    # Compute distance
    dist = np.linalg.norm(x2-x1,axis=-1)
    
    return dist

def dist_euclid3(x1,x2):
    
    # Simple 3d element space
    # d = sqrt( (ka*da)^2 + (ke*de)^2 + (ki*di)^2 )
    
    # Should select coefficients such that each component is approxamately equal.
    # Hence, step sizes are comparable in each dimension.
    # A change in da of 1 AU is expected to give much larger dV than inclination change of 1 deg.
    
    return

def dist_euclid5(x1,x2):
    
    # Simple 5d element space
    # d = sqrt( (ka*da)^2 + (ke*de)^2 + (ki*di)^2 + (kom*dom)^2 + (kw*dw)^2 )
    
    return

#%% Keplerian Quotient spaces -------------------------------------------------
def dist_D1(x1, x2):
    '''
    D criteria. Southworth & Hawkins (1963)
    "Statistics of Meteor Streams"
    
    Using formulation defined in Kholshevnikov
    '''
    # Extract elements
    q1,e1,inc1,om1,w1 = x1.T
    q2,e2,inc2,om2,w2 = x2.T
    
    # Compute cosi,sini,delta
    c1 = np.cos(inc1)
    s1 = np.sin(inc1)
    c2 = np.cos(inc2)
    s2 = np.sin(inc2)
    delta = om1-om2
    
    # Compute intermediate terms
    L = 1. # Scale factor (alternatively use L=4 for Kuiper belt region)
    cosI = c1*c2 + s1*s2*np.cos(delta)
    I = np.arccos(cosI)
    sin2I_2 = np.sin((inc1-inc2)/2)**2 + s1*s2*(np.sin(delta/2)**2) # sin^2(I/2)
    E = np.cos((inc1+inc2)/2)*np.sin(delta/2)/np.cos(I/2) # Greek letter ξ in Kholshevnikov
    # Quadrant check
    if hasattr(E, "__len__"):
        # Multiple values
        ind = delta <= np.pi
        PI = w1-w2 - 2*np.arcsin(E)
        PI[ind] = w1[ind]-w2[ind] + 2*np.arcsin(E[ind])
        
    else:
        # Single value
        if abs(delta) <= np.pi:
            PI = w1-w2 + 2*np.arcsin(E)
        else:
            PI = w1-w2 - 2*np.arcsin(E)
    
    # Compute metric D1^2
    D12 = (q1-q2)**2/(L**2) + 4*sin2I_2 + (e1-e2)**2 + ((e1+e2)*np.sin(PI/2))**2
    dist = np.sqrt(D12)
    
    return dist

def dist_p1(x1, x2):
    '''
    Kholshevnikov p1 distance metric.
    '''
    # Extract elements
    p1,e1,inc1,om1,w1 = x1.T
    p2,e2,inc2,om2,w2 = x2.T
    
    # Compute cosi,sini,delta
    c1 = np.cos(inc1)
    s1 = np.sin(inc1)
    c2 = np.cos(inc2)
    s2 = np.sin(inc2)
    delta = om1-om2
    
    # Compute intermediate terms cosI,cosP
    cosI = c1*c2 + s1*s2*np.cos(delta)
    cosP = s1*s2*np.sin(w1)*np.sin(w2) + (np.cos(w1)*np.cos(w2) + c1*c2*np.sin(w1)*np.sin(w2))*np.cos(delta) \
                + (c2*np.cos(w1)*np.sin(w2) - c1*np.sin(w1)*np.cos(w2))*np.sin(delta)
    L = 1. # Scale factor
    
    # Compute distance squared
    dist2 = (p1+p2 - 2*np.sqrt(p1*p2)*cosI)/L + (e1**2 + e2**2 - 2*e1*e2*cosP)
    dist = np.sqrt(dist2)
    
    return dist


def dist_p2(x1, x2):
    '''
    Kholshevnikov p2 distance metric.
    '''
    # Extract elements
    p1,e1,inc1,om1,w1 = x1.T
    p2,e2,inc2,om2,w2 = x2.T
    
    # Compute cosi,sini,delta
    c1 = np.cos(inc1)
    s1 = np.sin(inc1)
    c2 = np.cos(inc2)
    s2 = np.sin(inc2)
    delta = om1-om2
    
    # Compute intermediate terms cosI,cosP
    cosI = c1*c2 + s1*s2*np.cos(delta)
    cosP = s1*s2*np.sin(w1)*np.sin(w2) + (np.cos(w1)*np.cos(w2) + c1*c2*np.sin(w1)*np.sin(w2))*np.cos(delta) \
                + (c2*np.cos(w1)*np.sin(w2) - c1*np.sin(w1)*np.cos(w2))*np.sin(delta)
    
    # Compute distance squared
    dist2 = (1+e1**2)*p1 + (1+e2**2)*p2 - 2*np.sqrt(p1*p2)*(cosI + e1*e2*cosP)
    dist = np.sqrt(dist2)
    
    return dist

def dist_p3(x1, x2):
    '''
    Kholshevnikov p3 distance metric.
    '''
    # Extract elements
    p1,e1,inc1,om1,w1 = x1.T
    p2,e2,inc2,om2,w2 = x2.T
    
    # Compute cosi,sini,delta
    c1 = np.cos(inc1)
    s1 = np.sin(inc1)
    c2 = np.cos(inc2)
    s2 = np.sin(inc2)
    delta = om1-om2
    
    # Compute intermediate terms
    A4 = (e1**2)*(e2**2)*(1-s1**2*np.sin(w1)**2)*(1-s2**2*np.sin(w2)**2) \
            + 2*e1*e2*s1*s2*(np.cos(w1)*np.cos(w2) + c1*c2*np.sin(w1)*np.sin(w2) )
    A3 = c1*c2 + e1*e2*s1*s2*np.sin(w1)*np.sin(w2) + np.sqrt(s1**2*s2**2 + A4)
    
    
    # Compute distance squared
    dist2 = (1+e1**2)*p1 + (1+e2**2)*p2 - 2*np.sqrt(p1*p2)*A3
    dist = np.sqrt(dist2)
    
    return dist

def dist_p4(x1, x2):
    '''
    Kholshevnikov p4 distance metric.
    '''
    # Extract elements
    p1,e1,inc1,om1,w1 = x1.T
    p2,e2,inc2,om2,w2 = x2.T
    
    # Compute cosi,sini,delta
    c1 = np.cos(inc1)
    s1 = np.sin(inc1)
    c2 = np.cos(inc2)
    s2 = np.sin(inc2)
    delta = om1-om2
    
    # Compute intermediate terms cosI,cosP
    cosI = c1*c2 + s1*s2*np.cos(delta)
    
    # Compute distance squared
    dist2 = (1+e1**2)*p1 + (1+e2**2)*p2 - 2*np.sqrt(p1*p2)*(e1*e2 + cosI)
    dist = np.sqrt(dist2)
    
    return dist

def dist_p5(x1, x2):
    '''
    Kholshevnikov p5 distance metric.
    '''
    # Extract elements
    p1,e1,inc1 = x1.T
    p2,e2,inc2 = x2.T
    
    # Compute distance squared
    dist2 = (1+e1**2)*p1 + (1+e2**2)*p2 - 2*np.sqrt(p1*p2)*(e1*e2 + np.cos(inc1-inc2))
    dist = np.sqrt(dist2)
    
    return dist



def dist_planechange(x1,x2):
    '''
    Plane change angle between two orbits.
    
    A measure of combination of w and om.
    Alternatively computed from the angular momentum vector components.
    '''
    
    # Extract elements
    hx1,hy1,hz1 = x1.T
    hx2,hy2,hz2 = x2.T
    
    # Compute magnitude of H
    h1 = np.sqrt(hx1**2 + hy1**2 + hz1**2)
    h2 = np.sqrt(hx2**2 + hy2**2 + hz2**2)

    # Compute change in inclination (angle between orbit normals)
    dinc = np.arccos((hx1*hx2 + hy1*hy2 + hz1*hz2)/(h1*h2)) # Alternative direct method
    
    dist = dinc
    
    return dist

#%% Minimum Nodal Distance ----------------------------------------------------
def dist_mnid(x1,x2):
    '''
    Minimum Nodal Intersection Distance.
    
    Compute the distance between the orbits at the relative ascending and
    descending nodes of the two orbits. In some cases, this will give the 
    true MOID, but not always.
    '''
    
    # Formatulation is based on Eccentric anomaly approach
    # Lazovic (1993) "The Appoximate Values of Eccentric Anomalies of Proximity"
    
    # See also Hoots, et al. (1984) "An analytic method to determine future close
    # appoaches between satellites"
    
    # See also Murison & Munteanu (2006) "On the Distance Fucntion between Two
    # Confocal Keplerian Orbits" for an alternative approach
    
    # Extract elements
    a1,e1,inc1,om1,w1 = x1.T
    a2,e2,inc2,om2,w2 = x2.T

    # Find the eccentric anomalies of the relative nodes
    # See Lazovic (1993) "The Appoximate Values of Eccentric Anomalies of Proximity"

    # Compute the relative inclination of the two orbits (eq. 2)
    # see also Murison & Munteanu (2006) eq. 27
    I = np.arccos(np.cos(inc1)*np.cos(inc2) + np.sin(inc1)*np.sin(inc2)*np.cos(om2-om1) )
    
    # Compute basis vectors of the perifocal frames P,Q
    # Use transformation matrix from perifocal to ECEF (see Curtis pg 174) QxX
    # P = [QxX]*[1,0,0]^T (1st column of transformation matrix)
    P1 = np.column_stack([np.cos(om1)*np.cos(w1)-np.sin(om1)*np.sin(w1)*np.cos(inc1),
                          np.sin(om1)*np.cos(w1)+np.cos(om1)*np.cos(inc1)*np.sin(w1),
                          np.sin(inc1)*np.sin(w1),
                          ])
    P2 = np.column_stack([np.cos(om2)*np.cos(w2)-np.sin(om2)*np.sin(w2)*np.cos(inc2),
                          np.sin(om2)*np.cos(w2)+np.cos(om2)*np.cos(inc2)*np.sin(w2),
                          np.sin(inc2)*np.sin(w2),
                          ])
    
    # Q = [QxX]*[0,1,0]^T (2nd column of transformation matrix)
    Q1 = np.column_stack([-np.cos(om1)*np.sin(w1)-np.sin(om1)*np.cos(inc1)*np.cos(w1),
                          -np.sin(om1)*np.sin(w1)+np.cos(om1)*np.cos(inc1)*np.cos(w1),
                          np.sin(inc1)*np.cos(w1),
                          ])
    Q2 = np.column_stack([-np.cos(om2)*np.sin(w2)-np.sin(om2)*np.cos(inc2)*np.cos(w2),
                          -np.sin(om2)*np.sin(w2)+np.cos(om2)*np.cos(inc2)*np.cos(w2),
                          np.sin(inc2)*np.cos(w2),
                          ])
    
    # H = [QxX]*[0,0,1]^T (3rd column of transformation matrix)
    H1 = np.column_stack([np.sin(om1)*np.sin(inc1),
                          -np.cos(om1)*np.sin(inc1),
                          np.cos(inc1),
                          ])
    H2 = np.column_stack([np.sin(om2)*np.sin(inc2),
                          -np.cos(om2)*np.sin(inc2),
                          np.cos(inc2),
                          ])
    
    # Find the eccentric anomalies of the intersection
    # Want to find points where (r1.H2 = 0) & (r2.H1 = 0)
    
    # The position vectors can be written in the perifocal coordiantes
    # r1 = a1*(cos(E1) - e1)*P1 + b1*sinE1*Q1
    # r2 = a2*(cos(E2) - e2)*P2 + b2*sinE2*Q2
    # (using rmag = a(1-eCos(E)) )
    
    # Compute the semi-minor axis b
    b1 = a1*np.sqrt(1. - e1**2)
    b2 = a2*np.sqrt(1. - e2**2)
    
    # Compute the coefficients of the solution (eq 5)
    # Use einsum to compute dot products
    A1 = b1*np.einsum('ij,ij->i', Q1, H2 )
    B1 = a1*np.einsum('ij,ij->i', P1, H2 )
    C1 = -a1*e1*np.einsum('ij,ij->i', P1, H2 )
    A2 = b2*np.einsum('ij,ij->i', Q2, H1 )
    B2 = a2*np.einsum('ij,ij->i', P2, H1 )
    C2 = -a2*e2*np.einsum('ij,ij->i', P2, H1 )
    
    # Solve for eccentric anomalies (eq 6) (2 solutions)
    E11 = np.arctan2(-A1*B1 + C1*np.sqrt(A1**2 + B1**2 - C1**2), A1**2 - C1**2 )
    E12 = np.arctan2(-A1*B1 - C1*np.sqrt(A1**2 + B1**2 - C1**2), A1**2 - C1**2 )
    E21 = np.arctan2(-A2*B2 + C2*np.sqrt(A2**2 + B2**2 - C2**2), A2**2 - C2**2 )
    E22 = np.arctan2(-A2*B2 - C2*np.sqrt(A2**2 + B2**2 - C2**2), A2**2 - C2**2 )
    
    # Ensure each solution satisfies eq 6
    # The solution E11+pi is also valid. Only 1 will satisfy eq 6.
    # (Testing - valid within tolerance 1E-13)
    E11[abs(A1*np.sin(E11)+B1*np.cos(E11)+C1) > abs(A1*np.sin(E11+np.pi)+B1*np.cos(E11+np.pi)+C1)] += np.pi
    E12[abs(A1*np.sin(E12)+B1*np.cos(E12)+C1) > abs(A1*np.sin(E12+np.pi)+B1*np.cos(E12+np.pi)+C1)] += np.pi
    E21[abs(A2*np.sin(E21)+B2*np.cos(E21)+C2) > abs(A2*np.sin(E21+np.pi)+B2*np.cos(E21+np.pi)+C2)] += np.pi
    E22[abs(A2*np.sin(E22)+B2*np.cos(E22)+C2) > abs(A2*np.sin(E22+np.pi)+B2*np.cos(E22+np.pi)+C2)] += np.pi
    
    # Find position vectors at each of these points
    # r1 = a1*(cos(E1) - e1)*P1 + b1*sinE1*Q1
    r11 = (a1*(np.cos(E11) -e1))[:, np.newaxis]*P1 + (b1*np.sin(E11))[:, np.newaxis]*Q1
    r12 = (a1*(np.cos(E12) -e1))[:, np.newaxis]*P1 + (b1*np.sin(E12))[:, np.newaxis]*Q1
    r21 = (a2*(np.cos(E21) -e2))[:, np.newaxis]*P2 + (b2*np.sin(E21))[:, np.newaxis]*Q2
    r22 = (a2*(np.cos(E22) -e2))[:, np.newaxis]*P2 + (b2*np.sin(E22))[:, np.newaxis]*Q2
    
    # Compute minimum distances between pairs
    # (r11-r21),(r11,r22),(r12,r21),(r12,r22)
    dist = np.minimum.reduce([np.linalg.norm(r11-r21,axis=-1),
                              np.linalg.norm(r11-r22,axis=-1),
                              np.linalg.norm(r12-r21,axis=-1),
                              np.linalg.norm(r12-r22,axis=-1)
                              ])

    return dist

# Filtering via periapsis and apoapsis
# From http://adsabs.harvard.edu/pdf/1984CeMec..33..143H
# 1. Perigee-apogee test
# let q = max(q1,q2) and Q = min(Q1,Q2)
# if q - Q > D, then the orbits will not intersect.
# 2nd FIlter.
# Find the line of intersection between the two orbits.
# Find the distances d1, d2 between the orbits at the relative ascending 
# and descending nodes.
# if min(d1,d2) > D, then satellites cannot encounter closer than D.

# Alternative: Compute Line of relative nodes:
# see: https://www.sciencedirect.com/science/article/pii/S0094576518311019#fd12
# and see ref 12. ***

# TODO: ****** Look into relative state formulations
# https://arxiv.org/pdf/2003.02140.pdf
# See secition IV
# eq 54 and eq 58
# ζ(œ; η) = δp(1 +e1 cos λ1) −δξ cos(δφ)
# Collision safety margin.

#%% Delta-V based metric

def dist_zappala(x1, x2, astflag=False):
    '''
    Delta-V distance metric from Zappala et al. 1994.
    
    See also Masiero et al (2013)
    "Asteroid family identification using the hierarchical clustering method and
    WISE/ NEOWISE physical properties"
    '''
    # Extract elements
    a1,e1,inc1 = x1.T
    a2,e2,inc2 = x2.T
    
    AU = 149597870.700    # Astronomical unit (km)
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Correct for asteroids
    if astflag==True:
        # Asteroid dataset
        mu = 1.32712440018E11 # Gravitational parameter of the sun
        
        # Convert a1,a2 to km
        a1 *= AU
        a2 *= AU
    
    # Compute mean motion
    n1 = np.sqrt(mu/a1**3) # Mean motion
    
    # Compute differences in elements
    da = abs(a1-a2)
    de = abs(e1-e2)
    dsini = abs(np.sin(inc1)-np.sin(inc2))
    
    # Compute distance
    dist = n1*a1*np.sqrt( (5/4)*(da/a1)**2 + 2*(de**2) + 2*(dsini**2) )
    # dist = n1*a1*np.sqrt( 0.5*(da/a1)**2 + (3/4)*(de**2) + 4*(dsini**2) ) # Alternative
    
    return dist

#%% Low-thrust transfer distance metrics --------------------------------------

def dist_edel(x1,x2,astflag=False):
    '''
    Edelbaum delta-V distance metric.
    See Hasnain et al. (2012) "Capturing near-Earth asteroids around Earth"
    '''
    
    # Extract elements
    a1,e1,inc1,om1,w1,hx1,hy1,hz1 = x1.T
    a2,e2,inc2,om2,w2,hx2,hy2,hz2 = x2.T
    
    # Compute magnitude of H
    h1 = np.sqrt(hx1**2 + hy1**2 + hz1**2)
    h2 = np.sqrt(hx2**2 + hy2**2 + hz2**2)
    
    # Constants
    AU = 149597870.700    # Astronomical unit (km)
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Correct for asteroids
    if astflag==True:
        # Asteroid dataset
        mu = 1.32712440018E11 # Gravitational parameter of the sun
        
        # Convert a1,a2 to km
        a1 *= AU
        a2 *= AU
    
    # Compute average velocity
    V1 = np.sqrt(mu/(a1))*(1 - e1**2/4 -3*e1**4/64)
    V2 = np.sqrt(mu/(a2))*(1 - e2**2/4 -3*e2**4/64)
    
    
    # Compute change in inclination (angle between orbit normals)
    # h1 = np.array([hx1,hy1,hz1])
    # h2 = np.array([hx2,hy2,hz2])
    # dinc = np.arccos( np.dot(h1,h2) ) # Angle between orbits
    dinc = np.arccos((hx1*hx2 + hy1*hy2 + hz1*hz2)/(h1*h2)) # Alternative direct method
    
    # Compute Edelbaum delta-V
    dist = np.sqrt(V1**2 + V2**2 - 2*V1*V2*np.cos(dinc*np.pi/2 ))
    
    return dist

#%% Angular Momentum Based Distances

def dist_dhr(x1,x2):
    '''
    Absolute difference in hr radial component.

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
    hr1 = x1.T
    hr2 = x2.T
    
    # Dist ance
    dist = abs(hr2-hr1)
    
    return dist

def dist_dhz(x1,x2):
    '''
    Absolute difference in hz component.

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
    hz1 = x1.T
    hz2 = x2.T
    
    # Dist ance
    dist = abs(hz2-hz1)
    
    return dist

def dist_dhtheta(x1,x2,direction='min'):
    '''
    Relative difference in h theta component.

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
    hth1 = x1.T
    hth2 = x2.T
    
    # Wrap to [0,2*pi]
    hth1 = hth1%(2*np.pi)
    hth2 = hth2%(2*np.pi)
    
    # Compute difference accounting for direction
    dth_retro = np.maximum(hth1,hth2) - np.minimum(hth1,hth2) # Retrograd: measure counter-clockwise
    dth_pro = 2*np.pi - dth_retro # Prograde: measure clockwise
    
    # Select
    if direction.lower() in ['retro','retrograde']:
        dist = dth_retro
    elif direction.lower() in ['pro','prograde']:
        dist = dth_pro
    else:
        # Default - take min
        dist = np.minimum(dth_pro,dth_retro)
    
    return dist

def dist_htheta_arc(x1,x2,direction='min'):
    '''
    Arc length at mean radius spherical distance.
    Distance metric defined by the arc length along a reference circle 
    radius mean(r1,r2).
    
    dist = rmean*dtheta

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
    r1,th1 = x1.T
    r2,th2 = x2.T
    
    # Compute mean radius
    rmean = (r1+r2)/2
    dr = abs(r2-r1) # Difference in radius
    
    # Wrap to [0,2*pi]
    th1 = th1%(2*np.pi)
    th2 = th2%(2*np.pi)
    
    # Compute difference accounting for direction
    dth_retro = np.maximum(th1,th2) - np.minimum(th1,th2) # Retrograd: measure counter-clockwise
    dth_pro = 2*np.pi - dth_retro # Prograde: measure clockwise
    # Select
    if direction.lower() in ['retro','retrograde']:
        dth = dth_retro
    elif direction.lower() in ['pro','prograde']:
        dth = dth_pro
    else:
        # Default - take min
        dth = np.minimum(dth_pro,dth_retro)
    
    # Compute arc distance
    dist = rmean*abs(dth)
    
    return dist


def dist_h_cyl(x1,x2,direction='min'):
    '''
    Mean radius cylindrical distance.
    Distance metric defined by euclidean distance in cylindrical coordinates,
    where the theta component is computed as the arc length at the mean radius
    
    dist = sqrt( dr^2 + (rmean*dtheta)^2 + dz^2)

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
    
    # Compute mean radius
    rmean = (r1+r2)/2
    
    # Compute differences
    dr = r2-r1
    dz = z2-z1
    
    # Wrap to [0,2*pi]
    th1 = th1%(2*np.pi)
    th2 = th2%(2*np.pi)
    
    # Compute difference accounting for direction
    dth_retro = np.maximum(th1,th2) - np.minimum(th1,th2) # Retrograd: measure counter-clockwise
    dth_pro = 2*np.pi - dth_retro # Prograde: measure clockwise
    # Select
    if direction.lower() in ['retro','retrograde']:
        dth = dth_retro
    elif direction.lower() in ['pro','prograde']:
        dth = dth_pro
    elif direction.lower() in ['signed']:
        # Signed direction
        sign = np.sign(th2-th1) # Sign of direction
        dth = abs(th2-th1)%(2*np.pi) # Wrap to 2pi
        dth = dth*sign
        
    else:
        # Default - take min
        dth = np.minimum(dth_pro,dth_retro)
    
    # Compute distance
    dist = np.sqrt(dr**2 + (rmean*dth)**2 + dz**2)
    
    if direction.lower() in ['signed']:
        dist = dist*sign
    
    return dist

def dist_h_sph_mean(x1,x2):
    '''
    Mean radius spherical distance.
    Distance metric defined by the great circle distance along a reference
    sphere of radius mean(r1,r2). Extended in the radial direction.
    
    dist = sqrt( dr^2 + (rmean*Hav())^2 )

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
    r1,th1,phi1 = x1.T
    r2,th2,phi2 = x2.T
    
    # Compute mean radius
    rmean = (r1+r2)/2
    dr = r2-r1 # Difference in radius
    
    # Compute great circle arc
    hav = np.arccos( np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(th2-th1)  )
    
    # Compute distance
    dist = np.sqrt(dr**2 + (rmean*hav)**2)
    
    return dist

def dist_h_cyl_curv(x1,x2):
    '''
    Curvilinear cylindrical distance.
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


#%% Computing distances

def compute_distances(df,target,searchfield='NoradId'):
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
    
    # Angular Momentum parameters -------------------------------
    
    # dhr
    x1 = df['hr'][df[searchfield] == target].to_numpy()
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df['hr'].to_numpy()
    df['dHr']=dist_dhr(x1,x2)
    
    # dhz
    x1 = df['hz'][df[searchfield] == target].to_numpy()
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df['hz'].to_numpy()
    df['dHz']=dist_dhz(x1,x2)
    
    # dhtheta
    x1 = df['htheta'][df[searchfield] == target].to_numpy()
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df['htheta'].to_numpy()
    df['dHtheta']=dist_dhtheta(x1,x2)
    
    # Cylindrical Coords Distance
    
    # Mean arc length htheta arc length
    x1 = df[['hr','htheta']][df[searchfield] == target].to_numpy()
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df[['hr','htheta']].to_numpy()
    df['dHtheta_arc'] = dist_htheta_arc(x1,x2)
    
    # Cylindrical distance (extended mean radius arc length)
    x1 = df[['hr','htheta','hz']][df[searchfield] == target].to_numpy()
    x1 = np.repeat(x1,N,axis=0) # Duplicate
    x2 = df[['hr','htheta','hz']].to_numpy()
    df['dHcyl'] = dist_h_cyl(x1,x2)
    df['dHcyl_sign'] = dist_h_cyl(x1,x2,direction='signed')
    
    # # TODO: Cylindrical Curvilinear (dr/dtheta=const) f(hr,htheta,hz)
    # x1 = df[['hr','htheta','hz']][df[searchfield] == target].to_numpy()
    # x1 = np.repeat(x1,N,axis=0) # Duplicate
    # x2 = df[['hr','htheta','hz']].to_numpy()
    # df['dHcyl_curv']=dist_h_cyl_curv(x1,x2)
    
    # Spherical Coords
    
    # # Spherical mean radius great circle length
    # x1 = df[['hr','htheta','hphi']][df[searchfield] == target].to_numpy()
    # x1 = np.repeat(x1,N,axis=0) # Duplicate
    # x2 = df[['hr','htheta','hphi']].to_numpy()
    # df['dHsph_mean'] = dist_h_sph_mean(x1,x2)
    
    
    
    
    return df




