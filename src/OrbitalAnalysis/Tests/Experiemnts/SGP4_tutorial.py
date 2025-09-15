# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:41:19 2024

@author: scott
"""

# From https://github.com/bwinkel/cysgp4/blob/master/notebooks/02_starlink_constellation.ipynb

import numpy as np
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
# for animations, you may need to install "ffmpeg" and/or "imagemagick"
from matplotlib import animation, rc

import cysgp4

rc('animation', html='html5')

#%% Epochs

# want epoch for the following time
mjd_epoch = 58813.5
pydt = cysgp4.PyDateTime.from_mjd(mjd_epoch)
pydt

# the TLE uses a special format for the date/time:
pydt.tle_epoch

#%% Define single satellite

sat_name, sat_nr = 'MYSAT', 1
alt_km = 3000.  # satellite altitude
mean_motion = cysgp4.satellite_mean_motion(alt_km)
inclination = 10.  # deg
raan = 35.  # deg
eccentricity = 0.0001
argument_of_perigee = 0.  # deg
mean_anomaly = 112.  # deg

tle_tuple = cysgp4.tle_linestrings_from_orbital_parameters(
    sat_name,
    sat_nr,
    mjd_epoch,
    inclination,
    raan,
    eccentricity,
    argument_of_perigee,
    mean_anomaly,
    mean_motion
    )

tle_tuple

# Get position at epoch
my_tle = cysgp4.PyTle(*tle_tuple)
# want position at a given time, which can of course differ from the epoch
obs_dt = cysgp4.PyDateTime.from_mjd(58814.23)
my_sat = cysgp4.Satellite(my_tle)
my_sat.pydt = obs_dt
my_sat.eci_pos()  # in Geodetic frame

#%% Define constellation

altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9])
inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0])
nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])

def create_constellation(mjd_epoch, altitudes, inclinations, nplanes, sats_per_plane):
    
    my_sat_tles = []
    sat_nr = 1
    for alt, inc, n, s in zip(
            altitudes, inclinations, nplanes, sats_per_plane
            ):
        
        if s == 1:
            # random placement for lower orbits
            mas = np.random.uniform(0, 360, n)
            raans = np.random.uniform(0, 360, n)
        else:
            mas = np.linspace(0.0, 360.0, s, endpoint=False)
            mas += np.random.uniform(0, 360, 1)
            raans = np.linspace(0.0, 360.0, n, endpoint=False)
            mas, raans = np.meshgrid(mas, raans)
            mas, raans = mas.flatten(), raans.flatten()
        
        mm = cysgp4.satellite_mean_motion(alt)
        for ma, raan in zip(mas, raans):
            my_sat_tles.append(
                cysgp4.tle_linestrings_from_orbital_parameters(
                    'TEST {:d}'.format(sat_nr), sat_nr, mjd_epoch,
                    inc, raan, 0.001, 0., ma, mm
                    ))
                
            sat_nr += 1
    
    return my_sat_tles

# Create tle list
starlink_tle_tuples = create_constellation(
    mjd_epoch, altitudes, inclinations, nplanes, sats_per_plane
    )
len(starlink_tle_tuples)


# Create array of PyTLE objects
starlink_tles = np.array([
    cysgp4.PyTle(*tle)
    for tle in starlink_tle_tuples
    ])

#%% Propagate 

# Epochs
start_mjd = mjd_epoch
td = np.arange(0, 600, 5) / 86400.  # 1 d in steps of 10 s
mjds = start_mjd + td

# Observers
effbg_observer = cysgp4.PyObserver(6.88375, 50.525, 0.366) # Effelsberg 100-m radio telescope
parkes_observer = cysgp4.PyObserver(148.25738, -32.9933, 414.8) # Parkes telescope ("The Dish")
observers = np.array([effbg_observer, parkes_observer])

# Propagate
result = cysgp4.propagate_many(
    mjds[np.newaxis, np.newaxis, :],
    starlink_tles[:, np.newaxis, np.newaxis],
    observers[np.newaxis, :, np.newaxis]
    )

eci_pos = result['eci_pos']
topo_pos = result['topo']
len(mjds), len(starlink_tles), len(observers), eci_pos.shape, topo_pos.shape

eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
topo_pos_az, topo_pos_el, topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
topo_pos_az = (topo_pos_az + 180.) % 360. - 180.

# Get shape of data
num_sats, _, time_steps, _ = topo_pos.shape

#%% Plot

my_time = cysgp4.PyDateTime()
my_time.mjd = mjds[0]
plim = 8000

# The figure size should make such that one gets a nice pixel canvas
# that fits the standard movie sizes (at given dpi):
#    854 x  480  (480p) --> figsize=(8.54, 4.8), dpi=100
#   1280 x  720  (720p) --> figsize=(12.8, 7.2), dpi=100
#   1920 x 1080 (1080p) --> figsize=(12.8, 7.2), dpi=150
#   3840 x 2160    (4K) --> figsize=(12.8, 7.2), dpi=300
# so basically, divide desired width and height with dpi
# (beware, 4K videos get large and need a lot of RAM!)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# First observatory
for i in range(10):
    # ind  = (topo_pos_az[i][0] > 0.) & (topo_pos_az[i][0] > 0.)
    az = topo_pos_az[i][1]; el = topo_pos_el[i][1];
    ind  = (el > 0)
    
    ax.plot(np.deg2rad(el[ind]), az[ind],'-k')

# this takes a while!
plt.show()
# anim