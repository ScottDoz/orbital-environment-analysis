# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:06:00 2023

@author: scott

Experiment on KDEs
------------------

Use KDE to compute change in density distribution as a metric for assessing
the impact of adding or removing satellites.


"""

import matplotlib.pyplot as plt
import numpy as np
from astroML.plotting import hist
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

import plotly.graph_objects as go
import plotly
import plotly.express as px

# Module imports
from OrbitalAnalysis.SatelliteData import *
from OrbitalAnalysis.utils import get_data_home
from OrbitalAnalysis.Visualization import *
from OrbitalAnalysis.Density import *
from OrbitalAnalysis.Distances import *


#%% Load catalog

# Load at 2nd epoch - full catalog
df = load_2019_experiment_data(36) # New dataset
df['Name'][pd.isna(df.Name)] = ''
# Compute Density
log_satdens = compute_density(df)
df['log_p'] = log_satdens # Log density
df['p_hxhyhz'] = log_satdens
# Compute density
df['p'] = 10 ** df.log_p

# Get scalling [0-1]
min_max_scaler = preprocessing.MinMaxScaler()
features = ['hx','hy','hz']
Xfull = df[features].to_numpy()
Xfull = min_max_scaler.fit_transform(Xfull)
# Get ranges of hx,hy,hz

# Add scalled coords to dataframe
df[['hxs','hys','hzs']] = Xfull


#%% Compute distances from target object

# Select target
# target = 25544 # ISS
target = 22675 # Cosmos 2251 *
# target = 13552 # Cosmos 1408 *
# target = 25730 # Fengyun 1C *
# target = 40271 # Intelsat 30 (GEO)
# target = 49445 # Atlas 5 Centaur DEB

# Compute distances from target to other satellites
# Compute distance metrics
df = compute_distances(df,target,searchfield='NoradId')
df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename


#%% Subsets of objects

# 1. Debris objects
dfdeb = df[df['Name'].str.contains("DEB")]

# 2. Active objects
dfact = df[~df['Name'].str.contains("DEB")]

# Find peices of debris to remove. 10 closest peices of debris
dft = dfdeb.nsmallest(10, 'd_Euc')
targets = dft.NoradId.to_list() # List of norads


#%% Plot the catalog

# Plot density
fig = plot_3d_scatter_numeric(df,'hx','hy','hz',color='log_p',
                            logColor=False,colorscale='Blackbody_r',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=False,
                            )

# Add second trace of Debris objects
fig.add_trace(go.Scatter3d(
                            x=dfdeb.hx,
                            y=dfdeb.hy,
                            z=dfdeb.hz,
                            name = 'DEB',
                            customdata=dfdeb[['Name','a','e','i','om','w']],
                            hovertext = dfdeb.Name,
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "x: %{x:.2f}<br>" +
                                "y: %{y:.2f}<br>" +
                                "z: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode='markers',
                            marker=dict(
                                size=1.,
                                opacity=0.8,
                            ),
                        )
    )

# Add second trace of Debris objects
fig.add_trace(go.Scatter3d(
                            x=dfact.hx,
                            y=dfact.hy,
                            z=dfact.hz,
                            name = 'Active',
                            customdata=dfdeb[['Name','a','e','i','om','w']],
                            hovertext = dfdeb.Name,
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "x: %{x:.2f}<br>" +
                                "y: %{y:.2f}<br>" +
                                "z: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode='markers',
                            marker=dict(
                                size=1.,
                                opacity=0.8,
                            ),
                        )
    )



# Targeted objects to remove
fig.add_trace(go.Scatter3d(
                            x=dft.hx,
                            y=dft.hy,
                            z=dft.hz,
                            name = 'Target',
                            customdata=dft[['Name','a','e','i','om','w']],
                            hovertext = dft.Name,
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "x: %{x:.2f}<br>" +
                                "y: %{y:.2f}<br>" +
                                "z: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode='markers',
                            marker=dict(
                                size=4.,
                                opacity=0.8,
                            ),
                        )
    )

# fig.show()
plotly.offline.plot(fig)


# # Plot histogram in hz
# hist(df.hz, bins='freedman')



#%% Computations


# Compute log density contribution due to all
df['dlogp_targALL'] = compute_density_contributions(df,targets)     # All Target

# Convert to density contribution
df['dp_targALL'] = 10 **(df['dlogp_targALL'])

# 2. Compute fractional change in density
df['dp/p'] = df['dp_targALL']/(10 ** (df['p_hxhyhz']))


#%% Check Addition

# 1. Compute density contributions due to these satellites
# df['dp_targ1'] = compute_density_contributions(df,[targets[0]])  # Target 1
# df['dp_targ2'] = compute_density_contributions(df,[targets[1]])  # Target 2
# df['dp_targ3'] = compute_density_contributions(df,[targets[2]])  # Target 3
# df['dp_targ4'] = compute_density_contributions(df,[targets[3]])  # Target 4
# df['dp_targ5'] = compute_density_contributions(df,[targets[4]])  # Target 5
# df['dp_targ6'] = compute_density_contributions(df,[targets[5]])  # Target 6
# df['dp_targ7'] = compute_density_contributions(df,[targets[6]])  # Target 7
# df['dp_targ8'] = compute_density_contributions(df,[targets[7]])  # Target 8
# df['dp_targ9'] = compute_density_contributions(df,[targets[8]])  # Target 9
# df['dp_targ10'] = compute_density_contributions(df,[targets[9]]) # Target 10
# df['dp_targALL'] = compute_density_contributions(df,targets)     # All Target


# # Check to see how these are added up.
# df['dp_checksum'] = np.exp(df['dp_targ1']) + np.exp(df['dp_targ2']) + np.exp(df['dp_targ3']) + np.exp(df['dp_targ4']) \
#                     + np.exp(df['dp_targ5']) + np.exp(df['dp_targ6']) + np.exp(df['dp_targ7']) + np.exp(df['dp_targ8']) \
#                     + np.exp(df['dp_targ9']) + np.exp(df['dp_targ10'])
# # plt.plot(np.exp(df.dp_targALL),df.dp_checksum,'.k')

# # Check to see how these are added up.
# df['dp_checksum'] = 10 ** (df['dp_targ1']) + 10 ** (df['dp_targ2']) + 10 ** (df['dp_targ3']) + 10 ** (df['dp_targ4']) \
#                     + 10 ** (df['dp_targ5']) + 10 ** (df['dp_targ6']) + 10 ** (df['dp_targ7']) + 10 ** (df['dp_targ8']) \
#                     + 10 ** (df['dp_targ9']) + 10 ** (df['dp_targ10'])
# # plt.plot(10 ** (df.dp_targALL),df.dp_checksum,'.k')


# Conclusion!!!
# exp(dp_all) = sum( exp(targ_i) )  = exp(targ1) + exp(targ2) + ...


#%% Plot of (log) density contribution from these targets to all others

fig, axs = plt.subplots(1,2,figsize=(12, 8))
# Left plot
axs[0].plot(df['d_Euc'][~df['NoradId'].isin(targets)].to_numpy(),df['dlogp_targALL'][~df['NoradId'].isin(targets)].to_numpy(),'.k')
axs[0].set_xlabel("Distance from Center of Cluster")
axs[0].set_ylabel("Log Density")
axs[0].grid(True, which="both")
axs[0].set_xlim(0,50000)
axs[0].set_ylim(-258,20)

# Right plot
axs[1].plot(df['d_Euc'][~df['NoradId'].isin(targets)].to_numpy(),(df['dp_targALL'][~df['NoradId'].isin(targets)].to_numpy()),'.k')
axs[1].set_xlabel("Distance from Center of Cluster")
axs[1].set_ylabel("Density")
axs[1].set_yscale('log')
axs[1].grid(True, which="both")
axs[1].set_xlim(0,50000)
# axs[1].set_ylim(10**-258,10**20)

#%% Plot difference in log density

fig, axs = plt.subplots(1,2,figsize=(12, 8))
# Left plot
axs[0].plot(df['d_Euc'][~df['NoradId'].isin(targets)].to_numpy(),df['dp_targALL'][~df['NoradId'].isin(targets)].to_numpy(),'.k')
axs[0].set_xlabel("Distance from Center of Cluster")
axs[0].set_ylabel("Change in Density")
axs[0].grid(True, which="both")
axs[0].set_xlim(0,50000)
# axs[0].set_ylim(-258,20)
axs[0].set_yscale('log')

# Right plot
axs[1].plot(df['d_Euc'][~df['NoradId'].isin(targets)].to_numpy(),df['dp/p'][~df['NoradId'].isin(targets)].to_numpy(),'.k')
axs[1].set_xlabel("Distance from Center of Cluster")
axs[1].set_ylabel("Fractional Change in Density")
axs[1].grid(True, which="both")
axs[1].set_xlim(0,50000)
# axs[1].set_ylim(-258,20)
axs[1].set_yscale('log')

# Plot the change

# # Initialize figure (difference)
# fig = plot_3d_scatter_numeric(df[(df['dp_targALL']>-50) & (~df['NoradId'].isin(targets))],'hx','hy','hz',color='dp_targALL',
#                             logColor=False,colorscale='Blackbody_r',
#                             xrange=[-120000,120000],
#                             yrange=[-120000,120000],
#                             zrange=[-50000,150000],
#                             aspectmode='cube',
#                             render=False,
#                             )

# 
fig = plot_3d_scatter_numeric(df[(df['dlogp_targALL']>-50) & (~df['NoradId'].isin(targets))],'hx','hy','hz',color='dp_targALL',
                            logColor=True,colorscale='Blackbody_r',
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            render=False,
                            )

# Targeted objects to remove
fig.add_trace(go.Scatter3d(
                            x=dft.hx,
                            y=dft.hy,
                            z=dft.hz,
                            name = 'Target',
                            customdata=dft[['Name','a','e','i','om','w']],
                            hovertext = dft.Name,
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "x: %{x:.2f}<br>" +
                                "y: %{y:.2f}<br>" +
                                "z: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode='markers',
                            marker=dict(
                                size=4.,
                                opacity=0.8,
                            ),
                        )
    )

# fig.show()
plotly.offline.plot(fig)