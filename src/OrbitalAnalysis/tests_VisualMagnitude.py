# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:35:39 2022

@author: scott
"""

import numpy as np
import trimesh
import vedo
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import timeit
import pdb

# Module imports
from utils import get_data_home
from VisualMagnitude import *
from Ephem import *
from Epoch import *
from Events import *

#%% Simple Approximation

def test_plot_visual_magnitude(Rsat=1.,p=0.175,gs='SSRD-1',min_el=0.):
    '''
    Plot the predicted visual magnitude of a satellite as a function of epoch and
    elevation. Compare several different methods.
    
    Inputs:
        Rsat :  Radius of satellite (m)
        p :     Reflectivity of satellite (fraction)
        gs :    Name of ground stations
        min_el : Minimum elevation of ground station (deg)

    '''

    # Generate ephemeris times
    start_date = '2020-10-26 16:00:00.000' # Start Date e.g. '2020-10-26 16:00:00.000'
    stop_date =  '2020-11-25 15:59:59.999' # Stop Date e.g.  '2020-11-25 15:59:59.999'
    step = 10 # Step size (s)
    
    # Convert start and stop dates to Ephemeris Time
    et = et_from_date_range(start_date, stop_date, step)
    start_et = et[0]
    stop_et = et[-1]
    
    # Compute satellite lighting intervals
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # Compute station lighting intervals
    gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
    
    # Compute line-of-sight access intervals
    los_access = find_access(start_et,stop_et,station=gs,min_el=min_el)
    dflos = window_to_dataframe(los_access)
    
    # Compute visible (constrained) access intervals
    vis_access = constrain_access_by_lighting(los_access,gsdark,satlight)
    dfvis = window_to_dataframe(vis_access)
    
    # Generate ephemeris times covering access periods
    et = et_from_access_df(dfvis,step)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et,groundstations=[gs])
    dftopo = dftopo[0] # Select first station
    
    # # Plot overpass
    # from Visualization import plot_overpass_skyplot
    # plot_overpass_skyplot(dftopo, dfvis)
    
    
    # Compute Visual magnitudes
    msat = compute_visual_magnitude(dftopo,Rsat,p=p,k=0.12) # Lambertian phase function
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=p,k=0.12,lambertian_phase_function=False) # Constant phase function v(alpha)=1

    # Flatfacet model
    msat_ff = compute_visual_magnitude_flatfacet(dftopo,Rsat,model='sphere',p=0.25,k=0.12)
    
    # Add visual magnitudes to dataframe
    dftopo.insert(15,'m_lambertian',msat)
    dftopo.insert(16,'m_const',msat2)
    dftopo.insert(17,'m_ff',msat_ff)
    
    # Generate plots
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
    fig.suptitle('Visual Magnitude')
    # Magnitude vs elevation
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat,'.b',label='Lambertian phase function')
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat2,'.k',label='Constant phase function')
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat_ff,'.r',label='Flat Facet Model')
    ax1.set_xlabel("Elevation (deg)")
    ax1.set_ylabel("Visual Magnitude (mag)")
    ax1.legend(loc='lower right')
    ax1.invert_yaxis() # Invert y axis
    # Az/El
    ax2.plot(dftopo['ET'],msat,'-b',label='Lambertian phase function')
    ax2.plot(dftopo['ET'],msat2,'-k',label='Constant phase function')
    ax2.plot(dftopo['ET'],msat_ff,'-r',label='Flat Facet Model')
    ax2.invert_yaxis() # Invert y axis
    ax2.set_xlabel("Epoch (ET)")
    ax2.set_ylabel("Visual Magnitude (mag)")
    fig.show()
    
    
    return dftopo


#%% Flat Facet Model

def test_flat_facet_model():
    
    
    # Create icososphere mesh
    r = 1.0     # Radius (m)
    c = (0,0,0) # Center
    mesh = trimesh.primitives.Sphere(r,c,3)
    N = len(mesh.face_normals)    # Number of faces
        
    # Ephemeris at this epoch
    to_sun = np.array([1,0,0])
    to_obs = np.array([0,1,0])
    d_obs = 400*1000 # Distance to observer (m)
    
    # Face parameters
    u_n = mesh.face_normals       # Face normal vectors (unit)
    u_sun = np.tile(to_sun,(N,1)) # Sun direction (unit) (repeated for each facet)
    u_obs = np.tile(to_obs,(N,1)) # Observer direction (unit) (repeated for each facet)
    A = mesh.area_faces           # List of face areas
    cos_i = np.einsum('ij,ij->i', u_n, u_sun) # Cos of incidence angle (sun-normal)
    
    # Fsun. Fraction of incident light reflected from each facet.
    # Note: use einsum to compute dot product
    Csunvis = 1062 # Power/area
    Fsun = Csunvis*np.einsum('ij,ij->i', u_n, u_sun) # Eq 6
    Fsun[Fsun<0] = 0. # If angle is < pi/2 no light reflected
    # This is the normal component of solar flux incident on the facet (W/m^2)
    
    # Fobs. Solar flux at the observer from each facet.
    ptotal = 1.
    Fobs = Fsun*ptotal*A*np.einsum('ij,ij->i', u_n, u_obs)/d_obs**2
    Fobs[Fobs<0] = 0. # If angle is < pi/2 no light reflected
    
    # Total solar flux (sum of each facet)
    vCCD = 0 # Measurement noise from CCD
    F = sum(Fobs) + vCCD
    
    # Apparent magnitude
    m0 = -26.7
    mapp = m0 - 2.5*np.log10(abs(F/Csunvis))
    
    # Apparent magnitude from sphere approximation
    
    
    
    # Create vector actors
    # to_sun_arrow = vedo.shapes.Arrow(startPoint=(0,0,0),endPoint=to_sun*5,c='y',s=0.01)
    # to_obs_arrow = vedo.shapes.Arrow(startPoint=(0,0,0),endPoint=to_obs*5,c='c',s=0.01)
    
    to_sun_arrow = pv.Arrow(start=(0,0,0),direction=to_sun,scale=2,shaft_radius=0.01,tip_radius=0.02,tip_length=0.1)
    to_obs_arrow = pv.Arrow(start=(0,0,0),direction=to_obs,scale=2,shaft_radius=0.01,tip_radius=0.02,tip_length=0.1)
    
    # Create colormap
    from matplotlib import cm
    import matplotlib as mpl
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0., vmax=max(Fobs))
    scalarMap = cm.ScalarMappable(cmap='gray',norm=norm) # Colormap
    
    # Face colors rgba
    face_colors = scalarMap.to_rgba(Fobs)
    vert_colors =  trimesh.visual.color.face_to_vertex_color(mesh, face_colors)
    # mesh.face_colors = face_colors
    
    
    # Convert trimesh to pyvista mesh
    vertices = mesh.vertices
    tris = mesh.faces
    faces = np.hstack((np.full((len(tris), 1), 3), tris))
    pvmesh = pv.PolyData(vertices,faces)
    pvmesh1 = pv.PolyData(vertices,faces) # Copy of mesh
    
    # Append color
    pvmesh['F'] = vert_colors
    
    # pdb.set_trace()
    # # Plot with vedo
    # # Create plotter object
    # vp = vedo.Plotter(bg='black')
    # vedo_mesh = vedo.mesh.Mesh(mesh) # Vedo mesh
    # # vedo_mesh.cmap("jet", F)
    # pdb.set_trace()
    # vp.add(vedo_mesh)
    # vp.add(to_sun_arrow)
    # vp.add(to_obs_arrow)
    # vp.addGlobalAxes(axtype=4) # Add axes in corner
    # vp.show()
    
    
    # Plot with pyvista
    p = pv.Plotter(notebook=False,shape=(1, 2),lighting='none')
    # Left subplot (light only)
    p.add_mesh(pvmesh1,color='grey') # Space Object
    p.add_mesh(to_sun_arrow, color='y') # To sun
    p.add_mesh(to_obs_arrow, color='b') # To observer
    light = pv.Light(position=to_sun*10, light_type='scene light')
    p.add_light(light)
    # Right subplot
    p.subplot(0, 1)
    p.add_mesh(pvmesh,cmap='gray') # Space Object
    p.add_mesh(to_sun_arrow, color='y') # To sun
    p.add_mesh(to_obs_arrow, color='b') # To observer
    p.link_views() # link views
    p.camera_position = [(to_obs*10),(0,0,0),(0,0,1)] # Set camer position
    # pvmesh.plot_normals(mag=0.1, show_edges=True)
    # Add axes
    p.add_axes()
    p.show_grid()
    # p.add_legend()
    p.show()
    
    return