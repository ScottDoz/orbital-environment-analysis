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
import spiceypy as spice

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import plotly.graph_objects as go
import plotly
import plotly.express as px

import pdb

from Ephem import *
from Events import *

#%% Simple Histograms

def plot_hist(df,var,label=None,bins=100,xlog=False,ylog=False):
    '''
    Plot a histogram.

    Parameters
    ----------
    df : Dataframe
        Input dataset
    var : str
        Variable of interest in dataset.
    label : str, optional
        Label to place on the x axis.
        If left blank, will use variable name.
    bins : int, optional
        Number of bins in histogram. The default is 100.
    xlog : bool, optional
        Flag to specify log-scale on x axis. The default is False.
    ylog : bool, optional
        Flag to specify log-scale on y axis. The default is False.

    '''
    
    fig, ax = plt.subplots(1,1,figsize=(8, 8))
    plt.hist(df[var],bins=bins)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if label is None:
        label = var
    ax.set_xlabel(label)
    ax.set_ylabel('Frequnecy')
    plt.show()
    
    
    return

#%% 2D Hess Diagram

def plot_2d_hess(df,var1,var2,xlabel=None,ylabel=None,Nx=50,Ny=50,logscale=True):
    
    # Adapted from Fig 1.10 of AstroML
    # https://www.astroml.org/book_figures/chapter1/fig_S82_hess.html#book-fig-chapter1-fig-s82-hess
    
    # from astroML.plotting import setup_text_plots
    # setup_text_plots(fontsize=8, usetex=True)
    
    
    # Extract data
    x = df[var1].to_numpy()
    y = df[var2].to_numpy()
    
    # Define bins
    binsx = np.linspace(min(x),max(x),Nx)
    binsy = np.linspace(min(y),max(y),Ny)
    
    # Compute and plot 2D histogram
    H, xbins, ybins = np.histogram2d(x, y,bins=(binsx,binsy))
    # Get color of pixels
    if logscale:
        H[H == 0] = 1  # prevent warnings in log10
        c = np.log10(H).T
        clabel = 'log(Num per pixel)'
    else:
        c = H.T
        clabel = 'Num per pixel'
    
    
    # Create a black and white color map where bad data (NaNs) are white
    cmap = plt.cm.binary
    # cmap.set_bad('w', 1.)
    
    # Use the image display function imshow() to plot the result
    fig, ax = plt.subplots(figsize=(5, 3.75))
    
    im = ax.imshow(c, origin='lower',
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              # cmap=cmap, #interpolation='nearest',
              cmap=plt.cm.gist_heat_r,
              aspect='auto')
    
    if xlabel is None:
        xlabel = var1
    if ylabel is None:
        ylabel = var2
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    
    # cb = plt.colorbar(ticks=[0, 1, 2, 3], orientation='horizontal')
    plt.colorbar(im,ax=ax,label=clabel)
    # cb.set_label(r'$\mathrm{number\ in\ pixel}$')
    # plt.clim(0, 3)
    
    
    plt.show()
    
    return



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

#%% 3D Scatter Plots

def plot_3d_scatter_numeric(df,xlabel,ylabel,zlabel,color=None,
                            logColor=False,colorscale='Blackbody',
                            xrange=[None,None],yrange=[None,None],zrange=[None,None],
                            aspectmode='auto',
                            filename='temp-plot.html'):
    '''
    Plot the catalog of objects in angular 3D coordinates.
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
        if color is not None:
            c = df[color]
            color_label = color
            if logColor == True:
                # Log of color
                c = np.log10(c)
                color_label = 'log('+color+')'
            
        # Select x,y,z data
        x = df[xlabel]
        y = df[ylabel]
        z = df[zlabel]
        
        if color is None:
            # Plot without colorbar
            fig = go.Figure(data=[go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            customdata=df[['Name','a','e','i','om','w']],
                            hovertext = df.Name,
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
                                size=0.5,
                                opacity=0.8,
                            ),
                        )])
        
        else:
            # Plot with colorbar
        
            fig = go.Figure(data=[go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                customdata=df[['Name','a','e','i','om','w']],
                                hovertext = df.Name,
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
                                    size=1,
                                    color=c,             # set color to an array/list of desired values
                                    colorscale=colorscale,   # choose a colorscale 'Viridis'
                                    opacity=0.8,
                                    colorbar=dict(thickness=20,title=color_label)
                                ),
                            )])
        
        # Axes
        if xrange != [None,None]:
            fig.update_yaxes(range = xrange)
            xaxis=go.layout.scene.XAxis(title=xlabel,gridcolor='white',gridwidth=1,range=xrange)
        else:
            xaxis=go.layout.scene.XAxis(title=xlabel,gridcolor='white',gridwidth=1)
        if yrange != [None,None]:
            yaxis=go.layout.scene.YAxis(title=ylabel,gridcolor='white',gridwidth=1,range=yrange)
        else:
            yaxis=go.layout.scene.YAxis(title=ylabel,gridcolor='white',gridwidth=1)
        if zrange != [None,None]:
            zaxis=go.layout.scene.ZAxis(title=zlabel,gridcolor='white',gridwidth=1,range=zrange)
        else:
            zaxis=go.layout.scene.ZAxis(title=zlabel,gridcolor='white',gridwidth=1)

        
        # Update figure title and layout
        fig.update_layout(
            # title='2D Scatter',
            title_x = 0.5,
            scene=go.layout.Scene(
                xaxis=xaxis,
                yaxis=yaxis,
                zaxis=zaxis,
                aspectmode=aspectmode,
            ),
            # paper_bgcolor='rgb(243, 243, 243)',
            # plot_bgcolor='rgb(243, 243, 243)',
            # paper_bgcolor='rgb(0, 0, 0)',
            # plot_bgcolor='rgb(0, 0, 0)',
            margin=dict(l=20, r=20, t=20, b=20)
            )
        
        # Update axis ranges (optional)
        fig.update_xaxes(range = xrange)
        
        
        
        # Render
        plotly.offline.plot(fig, validate=False, filename=filename)
        
    
    return

def plot_3d_scatter_cat(df,xlabel,ylabel,zlabel, cat):
    '''
    Plot the catalog of objects in angular momentum space.
    Color by a categorical parameter
    
    '''
    
    import plotly.graph_objects as go
    import plotly
    
    # Check if data is timeseries (from multiple months)
    timeseries = False
    filename = '3DScatterCat.html'
    mode = 'markers'
    
    # Create figure
    fig = go.Figure()
    
    # Extract region data
    from natsort import natsorted
    region_names = natsorted(list(df[cat].unique())) # Names of regions
    # Ensure region names are strings
    region_names = [str(x) for x in region_names]
    df[cat] = df[cat].astype(str)
    
    region_data = {region:df.query(cat+" == '%s'" %region) for region in region_names}
    
    # Add traces
    for region_name, region in region_data.items():
        
        # Get the coordinates
        x = region[xlabel]
        y = region[ylabel]
        z = region[zlabel]
        
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
    
    # Update figure title and layout
    fig.update_layout(
        # title='2D Scatter',
        title_x = 0.5,
        scene=go.layout.Scene(
                xaxis=go.layout.scene.XAxis(title=xlabel,gridcolor='white',gridwidth=1),
                yaxis=go.layout.scene.YAxis(title=ylabel,gridcolor='white',gridwidth=1),
                zaxis=go.layout.scene.ZAxis(title=zlabel,gridcolor='white',gridwidth=1),
                # aspectmode='data',
            ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        # paper_bgcolor='rgb(0, 0, 0)',
        # plot_bgcolor='rgb(0, 0, 0)',
        )
    
    # Update figure layout
    fig.update_layout(legend=dict(
                        title='Category: {}'.format(cat),
                        itemsizing='constant',
                        itemdoubleclick="toggleothers",
                        # yanchor="top",
                        # y=0.99,
                        # xanchor="right",
                        # x=0.01,
                    ))
    
    # # Update ranges
    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
    #         yaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
    #         zaxis = dict(nticks=4, range=[-20*1E4,20*1E4],),
    #         aspectmode = 'cube',
    #         ),
    #     # width=700,
    #     # margin=dict(r=20, l=10, b=10, t=10)
    #     )
        
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


#%%

def plot_kde(df,xlabel,ylabel,bandwidth,normalized=False):
    
    # Error checking
    if xlabel not in list(df.columns):
        raise ValueError('xlabel not in dataset')
    if ylabel not in list(df.columns):
        raise ValueError('ylabel not in dataset')
    # if color not in list(df.columns):
    #     raise ValueError('color not in dataset')
    
    # Extract data
    X = df[[xlabel,ylabel]].to_numpy()
    Nx = 100
    Ny = 100
    # bandwidth = 10000
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
        
    # Create and fit the model
    from sklearn.neighbors import KernelDensity
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='tophat')
    kde1 = kde1.fit(X)
    # Evaluating at gridpoints
    log_dens1 = kde1.score_samples(Xgrid)
    dens1 = X.shape[0] * np.exp(log_dens1).reshape((Ny, Nx))
    
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    satdens = X.shape[0] * np.exp(log_satdens)
    print(satdens)
    # df['Density'] = satdens
    
    # Plot the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(dens1, origin='lower', 
              # norm=LogNorm(),
              # cmap=plt.cm.binary,
              cmap=plt.cm.gist_heat_r,
              extent=(xmin, xmax, ymin, ymax),aspect = 'auto' )
    plt.colorbar(label='density')
    ax.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='b') # Add points
    if normalized==False:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.xlabel(xlabel + " (normalized)")
        plt.ylabel(ylabel + " (normalized)")
    # Creat colorbar
    

    
    plt.show()
    
    return

#%% Main DIT Analysis Figures

def plot_time_windows(wins,groups,Types,
                      colors=None,filename=None,group_label='group',title="Time Windows"):
    '''
    Plot a Gantt chart displaying a set of time windows.
    '''
    
    df_list = []
    for i in range(len(wins)):
        
        # Convert window to dataframe
        win = wins[i] # Extract window
        dfi = window_to_dataframe(win,timefmt='datetime') # Access times (datetime)
        dfi[group_label] = groups[i] # y-labels
        dfi['Type'] = Types[i] # Types
        df_list.append(dfi) # Append to list
    
    # Concat all dataframes
    df = pd.concat(df_list)
    
    # Generate colors
    if colors is None:
        # colors = px.colors.qualitative.Plotly[:len(groups)]
        colors = px.colors.qualitative.Plotly
        
    
    # Create gant chart
    fig = px.timeline(df, x_start="Start", x_end="Stop", y=group_label, color="Type",
                      color_discrete_sequence=colors,
                      title=title,
                      )
    
    # Update bar height
    BARHEIGHT = .1
    fig.update_layout(
        yaxis={"domain": [max(1 - (BARHEIGHT * len(fig.data)), 0), 1]}, margin={"t": 0, "b": 0}
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    
    # # Add title to figure
    # fig.update_layout(
    #     title = {'text':title}
    # )
    
    # Render
    if filename is None:
        filename = 'temp-plot.html'
    plotly.offline.plot(fig, filename = str(filename), validate=False)

    
    return

def plot_linkbudget(dftopo,filename=None,title=None):
    ''' Plot the link budget data for a single ground station '''
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    start_et = dftopo.ET.min()
    stop_et = dftopo.ET.max()
    
    # Copy original dataframe
    dftopo1 = dftopo.copy()
    
    # Insert blank line between time gaps
    et = dftopo.ET.to_numpy() # Extract ephemeris time
    ind = np.where(np.diff(et)>100.)[0]
    df_new = pd.DataFrame(index=ind + 0.5) # New dataframe at half integer indices
    dftopo = pd.concat([dftopo, df_new]).sort_index()
    
    # Generate a subplot
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    
    # First trace. Solar and Sat Elevation.
    fig.add_trace(
        go.Scatter(x=dftopo.ET, y= np.rad2deg(dftopo['Sun.El']),
                   mode='lines',name='Sun.El',legendgroup = '1' ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dftopo.ET, y= np.rad2deg(dftopo['Sat.El']),
                   mode='lines',name='Sat.El',legendgroup = '1' ),
        row=1, col=1
    )
    
    # Second trace. Sat Range.
    fig.add_trace(
        go.Scatter(x=dftopo.ET, y=dftopo['Sat.R'],
                   mode='lines',name='Sat.Range',legendgroup = '2' ),
        row=2, col=1
    )
    
    # Third trace. SNR1.
    fig.add_trace(
        go.Scatter(x=dftopo.ET, y=dftopo['SNR1'],
                   mode='lines',name='SNR1',legendgroup = '3' ),
        row=3, col=1
    )
    
    # Fourth trace. Pd.
    fig.add_trace(
        go.Scatter(x=dftopo.ET, y=dftopo['PD'],
                   mode='lines',name='Pd',legendgroup = '4' ),
        row=4, col=1
    )
    
    # Update yaxis properties
    fig.update_xaxes(title_text="Epoch (ET)", row=4, col=1)
    # Update yaxis properties
    fig.update_yaxes(title_text="Elevation (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Range (km)", row=2, col=1)
    fig.update_yaxes(title_text="SNR1 (dB)", row=3, col=1)
    fig.update_yaxes(title_text="Pd", row=4, col=1)
    # Add gap in legend groups
    fig.update_layout(legend_tracegroupgap = 300)
    # Update title
    fig.update_layout(title_text=title)
    
    # Render
    if filename is None:
        filename = 'temp-plot.html'
    plotly.offline.plot(fig, filename = str(filename), validate=False)
    
    # Reset topo
    dftopo = dftopo1
    
    return
    
    


def plot_visibility(dftopo,filename=None,title=None):
    ''' Plot the visibility data for a single ground station '''
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    
    # Constraints
    cutoff_mag = 15. # Maximum magnitude for visibility
    # Compute contrained stats
    msat = dftopo.Vmag.to_numpy()
    max_mag = np.nanmax(msat[msat<=cutoff_mag])  # Maximum (dimest) magnitude
    min_mag = np.nanmin(msat[msat<=cutoff_mag])  # Minimum (brightest) magnitude 
    avg_mag = np.nanmean(msat[msat<=cutoff_mag]) # Mean magnitude
    
    start_et = dftopo.ET.min()
    stop_et = dftopo.ET.max()
    
    # Copy original dataframe
    dftopo1 = dftopo.copy()
    
    # Insert blank line between time gaps
    et = dftopo.ET.to_numpy() # Extract ephemeris time
    ind = np.where(np.diff(et)>100.)[0]
    df_new = pd.DataFrame(index=ind + 0.5) # New dataframe at half integer indices
    dftopo = pd.concat([dftopo, df_new]).sort_index()
    
    
    # Generate a subplot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    
    # First trace. Solar and Sat Elevation.
    fig.add_trace(
        go.Scatter(x=dftopo.UTCG, y= np.rad2deg(dftopo['Sun.El']),
                   mode='lines',name='Sun.El',legendgroup = '1' ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dftopo.UTCG, y= np.rad2deg(dftopo['Sat.El']),
                   mode='lines',name='Sat.El',legendgroup = '1' ),
        row=1, col=1
    )
    
    # Second trace. Sat Range.
    fig.add_trace(
        go.Scatter(x=dftopo.UTCG, y=dftopo['Sat.R'],
                   mode='lines',name='Sat.Range',legendgroup = '2' ),
        row=2, col=1
    )
    
    # Third trace. Visual Magnitude.
    fig.add_trace(
        go.Scatter(x=dftopo.UTCG, y=dftopo['Vmag'],
                   mode='lines',name='Vmag',legendgroup = '3' ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dftopo.UTCG, y=dftopo['Vmag2'],
                   mode='lines',name='Vmag2',legendgroup = '3' ),
        row=3, col=1
    )
    
    # Add shape regions
    fig.add_hrect(
        y0=min_mag, y1=max_mag,
        fillcolor="LightSalmon", opacity=0.3,
        layer="below", line_width=0,
        row=3, col=1
    ),
    
    
    
    
    
    # Update yaxis properties
    fig.update_xaxes(title_text="Epoch (ET)", row=3, col=1)
    # Update yaxis properties
    fig.update_yaxes(title_text="Elevation (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Range (km)", row=2, col=1)
    fig.update_yaxes(title_text="Visual Magnitude (mag)", row=3, col=1)
    # Reverse Vmag axes
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    # Add gap in legend groups
    fig.update_layout(legend_tracegroupgap = 300)
    # Update title
    fig.update_layout(title_text=title)
    
    # Render
    if filename is None:
        filename = 'temp-plot.html'
    plotly.offline.plot(fig, filename = str(filename), validate=False)
    
    # Reset topo
    dftopo = dftopo1
    
    return

def plot_overpass_skyplot(dftopo, dfa, filename=None,title=None):
    ''' Generate a skyplot of the visible passes for a single station '''
    
    # Bin data based on access time intervals
    # See: https://towardsdatascience.com/how-i-customarily-bin-data-with-pandas-9303c9e4d946
    
    dftopo1 = dftopo.copy()
    
    if 'Sat.Vmag' not in dftopo1.columns:
        # Compute visual magnitudes
        Rsat = 1 # Radius of satellite (m)
        msat = compute_visual_magnitude(dftopo1,Rsat,p=0.25,k=0.12) # With airmass
        dftopo1['Sat.Vmag'] = msat
    
    # Remove nan
    dftopo1 = dftopo1[pd.notnull(dftopo1['Sat.Vmag'])]
    
    # Create bins of ranges for each access interval
    ranges = pd.IntervalIndex.from_tuples(list(zip(dfa['Start'], dfa['Stop'])),closed='both')
    labels = dfa.Access.astype(str).to_list()
    # Apply cut to label access periods
    dftopo1['Access'] = pd.cut(dftopo1['ET'], bins=ranges, labels=labels).map(dict(zip(ranges,labels)))
    
    # Remove non-access
    dftopo1 = dftopo1[pd.notnull(dftopo1.Access)]
    
    
    # Add blank rows between groups of objects
    grouped = dftopo1.groupby('Access')
    dftopo1 = pd.concat([i.append({'Access': None}, ignore_index=True) for _, i in grouped]).reset_index(drop=True)
    # Forward fill na in Access 
    dftopo1.Access = dftopo1.Access.fillna(method="ffill")
    
    
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly
    
    # Convert angles to degrees
    dftopo1['Sat.El'] = np.rad2deg(dftopo1['Sat.El'])
    dftopo1['Sat.Az'] = np.rad2deg(dftopo1['Sat.Az'])
    
    # Plotly express (color by access)
    fig = px.line_polar(dftopo1, r="Sat.El", theta="Sat.Az",
                        color="Access",
                        color_discrete_sequence=px.colors.sequential.Plasma_r)
    
    # Multicolored lines
    # See: https://stackoverflow.com/questions/69705455/plotly-one-line-different-colors
    
    
    
    # Remove gaps
    fig.update_traces(connectgaps=False)
    
    # Reverse polar axis
    fig.update_layout(
        polar = dict(
          radialaxis = dict(range = [90,0]),
          angularaxis = dict(
                    tickfont_size=10,
                    rotation=90, # start position of angular axis
                    direction="clockwise",
                    showticklabels = True,
                    ticktext = ['0','1','2','3','4','5','6','7']
                  )
          ),
        )
    
    # # Add button to toggle traces on/off
    # button2 = dict(method='restyle',
    #             label='All',
    #             visible=True,
    #             args=[{'visible':True}],
    #             args2 = [{'visible': False}],
    #             )
    # # Create menu item    
    # um = [{'buttons':button2, 'label': 'Show', 'showactive':True,
    #         # 'x':0.3, 'y':0.99,
    #         }]
    
    # pdb.set_trace()
    
    # # add dropdown menus to the figure
    # fig.update_layout(showlegend=True, updatemenus=um)
    
    # Render
    if filename is None:
        filename = 'temp-plot.html'
    plotly.offline.plot(fig, filename = str(filename), validate=False)
    
    
    del dftopo1
    
    return

def plot_groundstation_network(filename=None):
    ''' Plot the locations of stations in the SSR or SSRD networks '''
    
    from plotly.subplots import make_subplots
    
    # Read data
    from GroundstationData import get_groundstations
    
    
    # Plotly go
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=2,column_widths=[0.35, 0.65],
                        specs=[[{"type": "scattergeo"},{"type": "scattergeo"}]],
                        )
    
    # SSR
    dfssr = get_groundstations(network='SSR')
    fig.add_trace(go.Scattergeo(
        name="SSR",
        lat=dfssr.Lat,
        lon=dfssr.Lon,
        mode='markers',
        marker_color='red',
        text=dfssr.Name,
        legendgroup='SSR',
        customdata=dfssr,
        hovertemplate=
                "<b>%{text}</b><br><br>" +
                "NAIF: %{customdata[1]}<br>" +
                "Lat: %{customdata[2]:.2f}<br>" +
                "Lon: %{customdata[3]:.2f}<br>" +
                "Alt: %{customdata[4]:.2f} km<br>" +
                "x: %{customdata[5]:.2f} km<br>" +
                "y: %{customdata[6]:.2f} km<br>" +
                "z: %{customdata[7]:.2f} km<br>",
        ),
        row=1, col=1,
    )
    # SSRD
    dfssrd = get_groundstations(network='SSRD')
    fig.add_trace(go.Scattergeo(
        name="SSRD",
        lat=dfssrd.Lat,
        lon=dfssrd.Lon,
        mode='markers',
        marker_color='blue',
        text=dfssrd.Name,
        legendgroup='SSRD',
        customdata=dfssrd,
        hovertemplate=
                "<b>%{text}</b><br><br>" +
                "NAIF: %{customdata[1]}<br>" +
                "Lat: %{customdata[2]:.2f}<br>" +
                "Lon: %{customdata[3]:.2f}<br>" +
                "Alt: %{customdata[4]:.2f} km<br>" +
                "x: %{customdata[5]:.2f} km<br>" +
                "y: %{customdata[6]:.2f} km<br>" +
                "z: %{customdata[7]:.2f} km<br>",
        ),
        row=1, col=1,
    )
    
    # SSR
    dfssr = get_groundstations(network='SSR')
    fig.add_trace(go.Scattergeo(
        name="SSR",
        lat=dfssr.Lat,
        lon=dfssr.Lon,
        mode='markers',
        marker_color='red',
        text=dfssr.Name,
        legendgroup='SSR', showlegend=False,
        customdata=dfssr,
        hovertemplate=
                "<b>%{text}</b><br><br>" +
                "NAIF: %{customdata[1]}<br>" +
                "Lat: %{customdata[2]:.2f}<br>" +
                "Lon: %{customdata[3]:.2f}<br>" +
                "Alt: %{customdata[4]:.2f} km<br>" +
                "x: %{customdata[5]:.2f} km<br>" +
                "y: %{customdata[6]:.2f} km<br>" +
                "z: %{customdata[7]:.2f} km<br>",
        ),
        row=1, col=2,
    )
    # SSRD
    dfssrd = get_groundstations(network='SSRD')
    fig.add_trace(go.Scattergeo(
        name="SSRD",
        lat=dfssrd.Lat,
        lon=dfssrd.Lon,
        mode='markers',
        marker_color='blue',
        text=dfssrd.Name,
        legendgroup='SSRD', showlegend=False,
        customdata=dfssrd,
        hovertemplate=
                "<b>%{text}</b><br><br>" +
                "NAIF: %{customdata[1]}<br>" +
                "Lat: %{customdata[2]:.2f}<br>" +
                "Lon: %{customdata[3]:.2f}<br>" +
                "Alt: %{customdata[4]:.2f} km<br>" +
                "x: %{customdata[5]:.2f} km<br>" +
                "y: %{customdata[6]:.2f} km<br>" +
                "z: %{customdata[7]:.2f} km<br>",
        ),
        row=1, col=2,
    )
    
    
    
    # Update layout
    # fig.update_layout(mapbox_style="open-street-map")
    # Update projection
    # See: https://plotly.com/python/reference/layout/geo/
    fig.update_geos(row=1, col=1,
                    projection_type="orthographic", # robinson
                    scope="world",showcountries=True, countrycolor="Black",
                    lataxis_showgrid=True, lonaxis_showgrid=True,
                    showland=True, landcolor="LightGreen",
                    showocean=True, oceancolor="LightBlue",
                    showlakes=True, lakecolor="Blue",
                    )
    
    fig.update_geos(row=1, col=2,
                    projection_type='robinson',
                    scope="world",showcountries=True, countrycolor="Black",
                    lataxis_showgrid=True, lonaxis_showgrid=True,
                    showland=True, landcolor="LightGreen",
                    showocean=True, oceancolor="LightBlue",
                    showlakes=True, lakecolor="Blue",
                    )
    
    # # Add title
    # fig.update_layout(
    #     title="Groundstation Networks",
    #     font=dict(
    #         family="Times New Roman",
    #         size=18,
    #         color="black"
    #     )
    # )
    
    fig.update_layout(
    title=go.layout.Title(
        text="<b>Groundstation Networks</b> <br><br><sup>SSR: 49 stations used for Trackability analysis. <br>SSRD: 7 stations used for Detectability analysis. </sup>",
        xref="paper",
        x=0
    ),
    )
    
    # Render
    if filename is None:
        filename = 'temp-plot.html'
    plotly.offline.plot(fig, filename = str(filename), validate=False)
    
    return



#%% Overpass plots

def plot_access_times(access,gslight,gsdark,satlight, satpartial, satdark):
    '''
    Generate a timeline plot showing the access intervals and lighting conditions
    of the satellite as seen from a groundstation.


    Parameters
    ----------
    access : SpiceCell
        Window containing line-of-sight access intervals.
    gsdark : SpiceCell
        Window containing time intervals of station darkness.
    satlight : SpiceCell
        Window containing time intervals of sat full sunlight.
    satpartial : SpiceCell
        Window containing time intervals of sat partial sunlight.

    '''
    
    # Process interval sets
    
    # Line-of-sight Access
    dfa = window_to_dataframe(access,timefmt='datetime') # Access times (datetime)
    dfa['trace'] = 'Viewing Geometry' # Trace label
    dfa['Type'] = 'Above horizon' # Access type
    
    # Visible Access
    # Compute set difference
    # visaccess = access - gslight -satdark
    vis = spice.wndifd(access,gslight) # Subtract station daylight
    vis = spice.wndifd(vis,satdark) # Subtract sat darkness
    dfvis = window_to_dataframe(vis,timefmt='datetime') # Access times (datetime)
    dfvis['trace'] = 'Visibility' # Trace label
    dfvis['Type'] = 'Visible Access' # Access type
    
    
    # Groundstation dark
    dfgs = window_to_dataframe(gsdark,timefmt='datetime') # Ground station dark times (datetime)
    dfgs['trace'] = 'Station Lighting' # Trace label
    dfgs['Type'] =  'GS Dark' # Trace label
    
    # Satellite Sunlight
    dfss = window_to_dataframe(satlight,timefmt='datetime') # Sat light times (datetime)
    dfss['trace'] = 'Sat Lighting' # Trace label
    dfss['Type'] =  'Sat Sun' # Trace label
    
    # Satellite Penumbra
    dfsp = window_to_dataframe(satpartial,timefmt='datetime') # Sat light times (datetime)
    dfsp['trace'] = 'Sat Lighting' # Trace label
    dfsp['Type'] =  'Sat Penumbra' # Trace label
    
    # Compine dataframes
    df = pd.concat( [dfgs[['Start', 'Stop', 'Duration','Type','trace']],
                     dfss[['Start', 'Stop', 'Duration','Type','trace']],
                     dfsp[['Start', 'Stop', 'Duration','Type','trace']],
                     dfa[['Start', 'Stop', 'Duration','Type','trace']],
                     dfvis[['Start', 'Stop', 'Duration','Type','trace']],
                     ])
    
    
    # Create gant chart
    fig = px.timeline(df, x_start="Start", x_end="Stop", y="trace", color="Type",
                      color_discrete_sequence=["black","goldenrod","grey","blue","red"],
                      )
    
    # Update bar height
    BARHEIGHT = .1
    fig.update_layout(
        yaxis={"domain": [max(1 - (BARHEIGHT * len(fig.data)), 0), 1]}, margin={"t": 0, "b": 0}
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    
    # Render
    filename = 'AccessPeriods.html'
    plotly.offline.plot(fig, validate=False, filename=filename)
    
    
    return





def plot_overpass_magnitudes(dftopo, dfa):
    
    
    # Bin data based on access time intervals
    # See: https://towardsdatascience.com/how-i-customarily-bin-data-with-pandas-9303c9e4d946
    
    dftopo1 = dftopo.copy()
    
    # Compute visual magnitudes
    Rsat = 1 # Radius of satellite (m)
    msat = compute_visual_magnitude(dftopo1,Rsat,p=0.25,k=0.12,include_airmass=True) # With airmass
    # msat = compute_visual_magnitude(dftopo1,Rsat,p=0.25,k=0.12,include_airmass=False) # Without airmass
    dftopo1['Sat.Vmag'] = msat
    
    # Remove nan
    dftopo1 = dftopo1[pd.notnull(dftopo1['Sat.Vmag'])]
    
    # Create bins of ranges for each access interval
    ranges = pd.IntervalIndex.from_tuples(list(zip(dfa['Start'], dfa['Stop'])),closed='both')
    labels = dfa.Access.astype(str).to_list()
    # Apply cut to label access periods
    dftopo1['Access'] = pd.cut(dftopo1['UTCG'], bins=ranges, labels=labels).map(dict(zip(ranges,labels)))
    
    # Remove non-access
    dftopo1 = dftopo1[pd.notnull(dftopo1.Access)]
    
    # Remove -ve elevations
    # dftopo1 = dftopo1[]
    
    # Add blank rows between groups of objects
    grouped = dftopo1.groupby('Access')
    dftopo1 = pd.concat([i.append({'Access': None}, ignore_index=True) for _, i in grouped]).reset_index(drop=True)
    # Forward fill na in Access 
    dftopo1.Access = dftopo1.Access.fillna(method="ffill")
    
    # Generate ticks for colorscale
    Vmin = dftopo1['Sat.Vmag'].min() # Min (brightest)
    Vmax = +30 # Limiting magnitude
    cticks = np.arange(int((Vmin//5)*5.),int(Vmax)+5, 5)
    
    # Assign markersize
    # Want to scale size of markers based on magnitude
    # Values range from 
    # (Brightest)    (Dimest)
    # -2   0   2   4   6  ... 30 ...   70  
    #                ^                    ^   
    # 10                      1         
    
    # Size range
    y1 = 5 # Max marker size
    y2 = 0.1  # Min marker size
    # Mag range
    x1 = 0 # Min mag (brightest)
    x2 = 30 # Max mag (dimmest)
    
    
    # Set size
    # See: https://github.com/eleanorlutz/western_constellations_atlas_of_space/blob/main/6_plot_maps.ipynb
    
    
    dftopo1['size'] = np.nan # Initialize
    dftopo1['size'] = y1 + ((y2-y1)/(x2-x1))*(dftopo1['Sat.Vmag'] - x1)
    dftopo1['size'][dftopo1['size']<1] = 1 # Limit minimum size
    dftopo1['size'][pd.isnull(dftopo1['size'])] = 1.
    
    # # Scaling markersize
    # # See "Scaling the size of bubble charts" https://plotly.com/python/bubble-charts/
    # # sizeref = 2.* max(array of size values) / (desired max size ** 2)
    # max_marker_size = 10 # Desired maximum marker size
    # max_val = (6.5 - 0)*5 # Magnitude 0 star
    # sizeref = 2*max_val/(max_marker_size**2)
    
    fig = go.Figure(data=
        go.Scatterpolar(
            r = np.rad2deg(dftopo1['Sat.El']),
            theta = np.rad2deg(dftopo1['Sat.Az']),
            # fillcolor='black',
            mode = 'markers+lines',
            line=dict(width=0.2, 
                       color='LightGrey',
                      # color=dftopo1['Sat.Vmag'],colorscale='Blackbody',
                      ),
            marker_size = dftopo1['size'],marker_sizemode='area',
            marker=dict(
                # size=5,
                # size=dftopo1['size'], #sizemode='area',sizeref=sizeref,sizemin=1,
                # color=np.log10(dftopo1['Sat.Vmag']), # set color to an array/list of desired values
                color=dftopo1['Sat.Vmag'],
                colorscale='Viridis',   # 'greys','Blackbody','Viridis'
                # autocolorscale='reversed',
                cmin = min(cticks),
                cmax = max(cticks),
                # cmax = dftopo1['Sat.Vmag'].max(),
                opacity=1.0,
                line=dict(width=0.1, color='DarkSlateGrey'),
                colorbar=dict(thickness=20,
                              title='Vmag',
                              # reversescale=True,
                              # tickmode="array",ticktext=[100,0,-100],tickvals=[-100,0,100],
                              tickmode="array",ticktext=np.flip(cticks),tickvals=cticks,
                              )
            ),
        ))
    
    # Reverse polar axis
    fig.update_layout(
        polar = dict(
          radialaxis = dict(range = [90,0]),
          angularaxis = dict(
                    tickfont_size=10,
                    rotation=90, # start position of angular axis
                    direction="clockwise",
                    showticklabels = True,
                    ticktext = ['0','1','2','3','4','5','6','7']
                  )
          ),
        )
    
    # # Reverse colorbar axis
    # fig.update_layout(
    #     color = dict(
    #       radialaxis = dict(range = [90,0]),
    #       angularaxis = dict(
    #                 tickfont_size=10,
    #                 rotation=90, # start position of angular axis
    #                 direction="clockwise",
    #                 showticklabels = True,
    #                 ticktext = ['0','1','2','3','4','5','6','7']
    #               )
    #       ),
    #     )
    
    
    
    # # Change background color
    fig.update_polars(bgcolor='black')
    # fig.update_polars(bgcolor='white')

    plotly.offline.plot(fig, validate=False)
        
    del dftopo1

    return
