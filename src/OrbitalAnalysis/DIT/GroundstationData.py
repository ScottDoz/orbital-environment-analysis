# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:21:49 2022

@author: scott

Groundstation Data Module
-------------------------

Functions to define the locations of groundstations, and generate required
Spice files.

"""

import pandas as pd
import numpy as np
import spiceypy as spice

import pdb

from OrbitalAnalysis.utils import get_data_home


#%% Ground Stations

def get_groundstations(network='SSR'):
    ''' 
    Get the lat/long/alt coordinates of the custom groundstation networks. 
    
    The SSR analysis consists of two separate networks:
        1. SSR 
            - Contains 49 stations spread across the globe.
            - Used for the Trackability analysis
        2. SSRD
            - Contains 7 stations over a range of latitudes, centered at around 120 E.
            - Used for the Detectability analysis
    '''
    
    # Load kernels
    kernel_dir = get_data_home()  / 'Kernels'
    # Generic kernels
    spice.furnsh( str(kernel_dir/'naif0012.tls') ) # Leap second kernel
    spice.furnsh( str(kernel_dir/'pck00010.tpc') ) # Planetary constants kernel
    
    
    # Hardcoded facility positions (planetodetic lat,lon,alt coords)
    
    # SSR Network (49 Facilities)
    if network == 'SSR':
        station_pos = [(-37.603, 140.388, 0.013851), (-45.639, 167.361, 0.344510), (-44.040, -176.375, 0.104582),
                       (-43.940, -72.450, 0.075715), (-51.655, -58.681, 0.127896), (-34.070, 19.703, 0.416372),
                       (-34.285, 115.934, 0.134033), (-49.530, 69.910, 0.199042), (18.872, -103.290, 0.735454),
                       (-15.096, -44.836, 0.697796), (-15.099, 15.875, 1.365027),
                       (-15.818, 45.893, -0.007232), (5.159, -53.637, 0.037340), (7.612, 134.631, 0.138556),
                       (-15.531, 134.143, 0.196179), (-22.500, 113.989, 0.068118), (-7.261, 72.376, -0.064980),
                       (-15.273, 166.878, 0.196300), (-13.890, -171.938, 0.392109), (18.532, -74.135, 0.291372),
                       (-9.798, -139.073, 0.845423), (-27.128, -109.355, 0.149995), (-7.947, -14.370, 0.216315),
                       (6.890, 158.216, 0.311603), (16.899, 102.561, 0.167567), (15.097, -15.726, 0.087358),
                       (14.846, 14.217, 0.359288), (14.846, 44.914, 2.071660), (17.396, 76.263, 0.382021),
                       (19.787, -155.658, 1.517667), (-15.450, -73.848, 4.202630), (44.676, -105.521, 1.249258),
                       (44.554, -75.459, 0.070607), (40.506, -124.123, 0.002242),
                       (43.040, -8.992, 0.411682), (47.014, -53.061, 0.191380), (45.481, 15.224, 0.252010),
                       (44.891, 44.590, 0.085764), (44.537, 75.371, 0.340541), (44.384, 104.729, 1.223731),
                       (45.271, 135.576, 0.399098), (53.312, 159.728, 0.536244), (55.395, -162.156, 0.673701),
                       (70.024, -162.191, 0.013845), (69.175, 18.258, 0.314617), (67.922, -103.469, -0.005155),
                       (74.757, -46.014, 2.651167), (72.423, 75.289, 0.011348), (71.372, 136.045, 0.010589)]
            
        
        # Define site names
        sites = ['SSR-'+str(i+1) for i in range(len(station_pos)) ]
        # Define site NAIF ID codes
        # Follow similar naming conventions to DSS stations - contain the NAIF ID
        # of the central body, followed by 3 digits identifying the station
        # e.g. DSS-05 = 399005. DSS stations up to 66 are already defined.
        # To avoid using any defined stations, we will start the numbering from
        # 399100.
        # See: https://pirlwww.lpl.arizona.edu/resources/guide/software/SPICE/naif_ids.html
        codes = 399000 + 100 + np.arange(len(station_pos))+1
        
    elif network == 'SSRD':
        # Simplified network (7 Facilities)
        # (Extracted from STK scenario)
        station_pos = [(72.03,123.435,0.0460127),(50.7654,120.436,0.705911),
                       (32.4958,123.132,0.0153716),(16.6223,123.731,0.0394418),
                       (-8.83527,121.634,0.0379327),(-25.0083,121.035,0.579283),
                       (-41.3004,121.522,-0.0318248),
                       ]
        
        # Define site names
        sites = ['SSRD-'+str(i+1) for i in range(len(station_pos)) ]
        # Define site NAIF ID codes
        # Start numbering from 300200
        # See: https://pirlwww.lpl.arizona.edu/resources/guide/software/SPICE/naif_ids.html
        codes = 399000 + 200 + np.arange(len(station_pos))+1
    
    
    # Extract coords
    lat = np.array([x[0] for x in station_pos])
    lon = np.array([x[1] for x in station_pos])
    lon[lon < 0.] += 360. # Wrap to 0 < lon < 360
    alt = np.array([x[2] for x in station_pos])
    
    
    # Load the planetary radii from pck 
    n,radii = spice.bodvrd ( "EARTH", "RADII", 3)
    # Compute flattening coefficient
    re  =  radii[0];
    rp  =  radii[2];
    f   =  ( re - rp ) / re;
    
    # Compute xyz coordinates
    # Convert to planetodetic rectangular coords
    # See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/georec_c.html
    # xyz = spice.georec(lon,lat,alt,radii[0],f)
    
    xyz = np.array([ spice.georec(np.deg2rad(lon[i]),np.deg2rad(lat[i]),alt[i],re,f) for i in range(len(lon))])
    
    
    # Load into dataframe
    df = pd.DataFrame(columns=['Name','NAIF','Lat','Lon','Alt','x','y','z'])
    df['Name'] = sites
    df['NAIF'] = codes
    df['Lat'] = lat  # Latitude (deg)
    df['Lon'] = lon # Longitude (deg)
    df['Alt'] = alt   # Altitude/Elevation (km)
    df['x'] = xyz[:,0] # Planetodetic X coord (km)
    df['y'] = xyz[:,1] # Planetodetic Y coord (km)
    df['z'] = xyz[:,2] # Planetodetic Z coord (km)
    
    return df

# DSN Ground stations defined in the earthstns_itrf93_201023.bsp file
# Use COMMNT command line tool to view.
#
# Antenna   NAIF    Diameter   x (m)            y (m)           z (m)
# 
# DSS-13    399013  34m     -2351112.659    -4655530.636    +3660912.728
# DSS-14    399014  70m     -2353621.420    -4641341.472    +3677052.318
# DSS-15    399015  34m     -2353538.958    -4641649.429    +3676669.984 {3}
# DSS-24    399024  34m     -2354906.711    -4646840.095    +3669242.325
# DSS-25    399025  34m     -2355022.014    -4646953.204    +3669040.567
# DSS-26    399026  34m     -2354890.797    -4647166.328    +3668871.755
# DSS-34    399034  34m     -4461147.093    +2682439.239    -3674393.133 {1}
# DSS-35    399035  34m     -4461273.090    +2682568.925    -3674152.093 {1}
# DSS-36    399036  34m     -4461168.415    +2682814.657    -3674083.901 {1}
# DSS-43    399043  70m     -4460894.917    +2682361.507    -3674748.152
# DSS-45    399045  34m     -4460935.578    +2682765.661    -3674380.982 {3}
# DSS-53    399053  34m     +4849339.965     -360658.246    +4114747.290 {2}
# DSS-54    399054  34m     +4849434.488     -360723.8999   +4114618.835
# DSS-55    399055  34m     +4849525.256     -360606.0932   +4114495.084
# DSS-56    399056  34m     +4849421.679     -360549.659    +4114646.987
# DSS-63    399063  70m     +4849092.518     -360180.3480   +4115109.251
# DSS-65    399065  34m     +4849339.634     -360427.6637   +4114750.733



#%% Create Groundstation Kernels

