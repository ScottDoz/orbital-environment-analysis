# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:13:20 2022

@author: scott

Visualizations
--------------

Plotly-based interactive visualizations

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pdb

#%% Visualizing Orbital Angular Momentum Space

def plot_h_space_numeric(df,color='i',logColor=False,colorscale='Blackbody'):
    '''
    Plot the catalog of objects in angular momentum space.
    Color by a numeric parameter.
    
    '''
    
    method = 'plotly'
    
    if method == 'matplotlib':
        # Simple matplotlib scatter plot
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(df.hx,df.hy,df.hz,s=1)
        plt.show()
    
    elif method == 'plotly':
        # Plotly scatter
        
        import plotly.graph_objects as go
        import plotly
        import plotly.express as px
        
        # Select color data
        c = df[color]
        color_label = color
        if logColor == True:
            # Log of color
            c = np.log10(c)
            color_label = 'log('+color+')'
        
        
        fig = go.Figure(data=[go.Scatter3d(
                            x=df.hx,
                            y=df.hy,
                            z=df.hz,
                            customdata=df[['Name','a','e','i','om','w']],
                            hovertext = df.Name,
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "hx: %{x:.2f}<br>" +
                                "hy: %{y:.2f}<br>" +
                                "hz: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode='markers',
                            marker=dict(
                                size=1,
                                color=c,             # set color to an array/list of desired values
                                colorscale=colorscale,   # choose a colorscale 'Viridis'
                                opacity=0.8,
                                colorbar=dict(thickness=20,title=color_label)
                            ),
                        )])
        
        # Update figure title and layout
        fig.update_layout(
            # title='2D Scatter',
            title_x = 0.5,
            xaxis=dict(
                    title='hx',
                    gridcolor='white',
                    gridwidth=1,
                    # type="log",
                    # exponentformat = "power",
                    # range = [-1, 2],
                    ),
            yaxis=dict(
                    title='hy',
                    gridcolor='white',
                    gridwidth=1,
                    # autorange = True,
                    # type="log",
                    # exponentformat = "power",
                    # autorange='reversed',
                    # range=[0,1],
                    ),
            # paper_bgcolor='rgb(243, 243, 243)',
            # plot_bgcolor='rgb(243, 243, 243)',
            # paper_bgcolor='rgb(0, 0, 0)',
            # plot_bgcolor='rgb(0, 0, 0)',
            )
        
        
        # Render
        plotly.offline.plot(fig, validate=False, filename='AngMomentumScatter.html')
        
    
    return

def plot_h_space_cat(df,cat='vishnu_cluster'):
    '''
    Plot the catalog of objects in angular momentum space.
    Color by a categorical parameter
    
    '''
    
    import plotly.graph_objects as go
    import plotly
    
    # Check if data is timeseries (from multiple months)
    timeseries = False
    filename = 'AngMomentumScatter.html'
    mode = 'markers'
    if len(df[df.duplicated(subset='NoradId')]) > 0:
        # Timeseries plots need to add blank line of None values between lines
        # see: https://stackoverflow.com/questions/56723792/how-to-efficiently-plot-a-large-number-of-line-shapes-where-the-points-are-conne
        timeseries = True
        filename = 'AngMomentumScatterTimeseries.html'
        mode = 'lines+markers'
    
    
    # Create figure
    fig = go.Figure()
    
    # Extract region data
    from natsort import natsorted
    region_names = natsorted(list(df[cat].unique())) # Names of regions
    # Ensure region names are strings
    region_names = [str(x) for x in region_names]
    df[cat] = df[cat].astype(str)
    
    if timeseries == False:
    
        region_data = {region:df.query(cat+" == '%s'" %region)
                                  for region in region_names}

    else:
        # Timeseries data
        
        # Loop through regions
        region_data = {} # Instantiate region data dict
        for region in region_names:
            
            # Extract the data
            data = df.query(cat+" == '%s'" %region) # Get the data
            data = data.sort_values(by=['NoradId','Epoch']).reset_index(drop=True)
            
            # Add blank rows between groups of objects
            grouped = data.groupby('NoradId')
            data = pd.concat([i.append({'NoradId': None}, ignore_index=True) for _, i in grouped]).reset_index(drop=True)
            # Append to dict
            region_data.update({region : data})
        
    
    # Add traces
    for region_name, region in region_data.items():
        
        # Get the coordinates
        x = region['hx']
        y = region['hy']
        z = region['hz']
        
        fig.add_trace(go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            name = region_name,
                            customdata=region[['Name','a','e','i','om','w']],
                            hovertext = region['Name'],
                            hoverinfo = 'text+x+y+z',
                            hovertemplate=
                                "<b>%{customdata[0]}</b><br><br>" +
                                "hx: %{x:.2f}<br>" +
                                "hy: %{y:.2f}<br>" +
                                "hz: %{z:.2f}<br>" +
                                "a: %{customdata[1]:.2f} km<br>" +
                                "e: %{customdata[2]:.2f}<br>" +
                                "i: %{customdata[3]:.2f} deg<br>" +
                                "om: %{customdata[4]:.2f} deg<br>" +
                                "w: %{customdata[5]:.2f} deg<br>" +
                                "",
                            mode=mode,
                            marker=dict(
                                size=1,
                                # color = color_dict[region_name],
                                opacity=0.8,
                                # colorbar=dict(thickness=20,title=cat)
                            ),
                        )
            )
    
    if timeseries == True:
        # Do not connect timesereies
        fig.update_traces(connectgaps=False)
    
    # Update figure title and layout
    fig.update_layout(
        # title='2D Scatter',
        title_x = 0.5,
        xaxis=dict(
                title='hx',
                gridcolor='white',
                gridwidth=1,
                # type="log",
                # exponentformat = "power",
                # range = [-1, 2],
                ),
        yaxis=dict(
                title='hy',
                gridcolor='white',
                gridwidth=1,
                # autorange = True,
                # type="log",
                # exponentformat = "power",
                # autorange='reversed',
                # range=[0,1],
                ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        # paper_bgcolor='rgb(0, 0, 0)',
        # plot_bgcolor='rgb(0, 0, 0)',
        )
    
    # Update figure layout
    fig.update_layout(legend=dict(
                        title='Clusters: {}'.format(cat),
                        itemsizing='constant',
                        itemdoubleclick="toggleothers",
                        # yanchor="top",
                        # y=0.99,
                        # xanchor="right",
                        # x=0.01,
                    ))
    
    # Update ranges
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
            yaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
            zaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
            aspectmode = 'cube',
            ),
        # width=700,
        # margin=dict(r=20, l=10, b=10, t=10)
        )
        
    # Render
    plotly.offline.plot(fig, validate=False, filename=filename)


    return


#%% Scatter Plots

def plot_2d_scatter_numeric(df,xlabel,ylabel,color,logColor=False,size=1.):
    '''
    Generate a 2D scatter plot using any available numeric feilds as the x,y,
    and color coordinates. Returns an interactive scatter plot with hover data
    showing information on each satellite.
    
    Example: 
    >> plot_2d_scatter(df,'h','hz','i')
    '''
    
    import plotly.graph_objects as go
    import plotly
    import plotly.express as px
    
    # Error checking
    if xlabel not in list(df.columns):
        raise ValueError('xlabel not in dataset')
    if ylabel not in list(df.columns):
        raise ValueError('ylabel not in dataset')
    if color not in list(df.columns):
        raise ValueError('color not in dataset')
    
    X = df[[xlabel,ylabel]].to_numpy()
    
    # Create grid to evaluate
    Nx = 20
    Ny = 20
    xmin, xmax = (df[xlabel].min(), df[xlabel].max())
    ymin, ymax = (df[ylabel].min(), df[ylabel].max())
    # Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
    #                                         np.linspace(ymin, ymax, Ny)))).T
        
    # Evaluate density
    # from sklearn.neighbors import KernelDensity
    # kde1 = KernelDensity(bandwidth=5, kernel='gaussian')
    # log_dens1 = kde1.fit(X).score_samples(Xgrid)
    # dens1 = X.shape[0] * np.exp(log_dens1).reshape((Ny, Nx))
    
    # Select color data
    c = df[color]
    color_label = color
    if logColor == True:
        # Log of color
        c = np.log10(c)
        color_label = 'log('+color+')'
    
    
    # Construct figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(
        go.Scattergl(
                x = df[xlabel],
                y = df[ylabel],
                customdata=df[['Name','a','e','i','om','w','h','hx','hy','hz']],
                hovertext = df.Name,
                hoverinfo = 'text+x+y+z',
                hovertemplate=
                        "<b>%{customdata[0]}</b><br><br>" +
                        "x: %{x:.2f}<br>" +
                        "y: %{y:.2f}<br>" +
                        "a: %{customdata[1]:.2f} km<br>" +
                        "e: %{customdata[2]:.2f}<br>" +
                        "i: %{customdata[3]:.2f} deg<br>" +
                        "om: %{customdata[4]:.2f} deg<br>" +
                        "w: %{customdata[5]:.2f} deg<br>" +
                        "h: %{customdata[6]:.2f}<br>" +
                        "hx: %{customdata[7]:.2f}<br>" +
                        "hy: %{customdata[8]:.2f}<br>" +
                        "hz: %{customdata[9]:.2f}<br>" +
                        "",
                mode = 'markers',
                marker = dict(
                    color = c,
                    size = size,
                    colorscale='Blackbody',   # choose a colorscale 'Viridis'
                    opacity=0.99,
                    colorbar=dict(thickness=20,title=color_label)
                )
        )
    )
    
    # Add density trace
    # from skimage import data
    # img = data.camera()
    
    # fig.add_trace(go.Contour(
    #                 z=dens1,
    #                 x=np.linspace(xmin,xmax,Nx), # horizontal axis
    #                 y=np.linspace(ymin,ymax,Ny) # vertical axis
    #             )
    
    #     )
    
    
    # Update figure title and layout
    fig.update_layout(
        title='2D Scatter',
        title_x = 0.5,
        xaxis=dict(
                title=xlabel,
                gridcolor='white',
                gridwidth=1,
                # type="log",
                # exponentformat = "power",
                # range = [-1, 2],
                ),
        yaxis=dict(
                title=ylabel,
                gridcolor='white',
                gridwidth=1,
                # autorange = True,
                # type="log",
                # exponentformat = "power",
                # autorange='reversed',
                # range=[0,1],
                ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        # paper_bgcolor='rgb(0, 0, 0)',
        # plot_bgcolor='rgb(0, 0, 0)',
        
        )
        
    # Render
    plotly.offline.plot(fig, validate=False, filename='Scatter.html')
    
    
    return

def plot_kde(df,xlabel,ylabel):
    
    # Error checking
    if xlabel not in list(df.columns):
        raise ValueError('xlabel not in dataset')
    if ylabel not in list(df.columns):
        raise ValueError('ylabel not in dataset')
    # if color not in list(df.columns):
    #     raise ValueError('color not in dataset')
    
    # Extract data
    X = df[[xlabel,ylabel]].to_numpy()
    Nx = 50
    Ny = 50
    bandwidth = 10000
    xmin, xmax = (df[xlabel].min(), df[xlabel].max())
    ymin, ymax = (df[ylabel].min(), df[ylabel].max())
    Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
                                            np.linspace(ymin, ymax, Ny)))).T
    
    # # Create grid to evaluate
    # from astroML.datasets import fetch_great_wall
    # X = fetch_great_wall()
    # Nx = 50
    # Ny = 125
    # bandwidth = 5
    # xmin, xmax = (-375, -175)
    # ymin, ymax = (-300, 200)
    
    # Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
    #                                         np.linspace(ymin, ymax, Ny)))).T
        
    # Evaluate density
    from sklearn.neighbors import KernelDensity
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    log_dens1 = kde1.fit(X).score_samples(Xgrid)
    dens1 = X.shape[0] * np.exp(log_dens1).reshape((Ny, Nx))
    
    # Plot the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(dens1, origin='lower', 
              # norm=LogNorm(),
              # cmap=plt.cm.binary,
              cmap=plt.cm.hot_r,
              extent=(xmin, xmax, ymin, ymax), )
    plt.colorbar(label='density')
    ax.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='k') # Add points
    
    # Creat colorbar
    

    
    plt.show()
    
    return
