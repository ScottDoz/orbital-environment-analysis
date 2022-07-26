# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:09:26 2022

@author: mauri
"""
# Module imports
from SatelliteData import *
from Clustering import *
from DistanceAnalysis import *
from Visualization import *
from sklearn import preprocessing
from utils import get_data_home
# from Overpass import *
# from Ephem import *
# from Events import *
# from GmatScenario import *

# Package imports
import pdb
from sklearn.neighbors import KernelDensity

def density_analysis(x,y):

    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    X = df[['a','e','i','om','w','h','hx','hy','hz']].to_numpy()
    print (df.columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    params_norm = min_max_scaler.fit_transform(X)
    df[['a','e','i','om','w','h','hx','hy','hz']] = params_norm
    plot_kde(df,x,y,0.1,normalized=True)
    #0.1 works well
    
    return df

def run_kde_experiment(filename,plot=True):
    
    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    X = df[['a','e','i','om','w','h','hx','hy','hz']].to_numpy()
    
    # Load experiments config file
    full_filename=get_data_home()/"DIT_Experiments"/"KDE"/filename
    df_config = pd.read_csv(full_filename)
    
    # Create figure
    if plot==True:
        num_row = df_config.plt_row.max()
        num_col = df_config.plt_col.max()
        # fig, axs = plt.subplots(num_row,num_col,figsize=(8, 8))
        fig = plt.figure()
    
    # Loop through rows and run experiment on each setting
    for index, row in df_config.iterrows():
        # Extract parameters
        ExpLabel = row['ExpLabel'].strip()
        kernel = row['kernel'].strip().lower()
        xlabel = row['xlabel'].strip()
        ylabel = row['ylabel'].strip()
        bandwidth = row['bandwidth']
        normalized = row['normalized']
        plt_row = row['plt_row']
        plt_col = row['plt_col']
        # Error checking
        if xlabel not in list(df.columns):
            raise ValueError('xlabel not in dataset')
        if ylabel not in list(df.columns):
            raise ValueError('ylabel not in dataset')
        # if color not in list(df.columns):
        #     raise ValueError('color not in dataset')
        
        # Extract features
        X = df[[xlabel,ylabel]].to_numpy()
        
        # Create grid
        Nx = 100
        Ny = 100
        # bandwidth = 10000
        xmin, xmax = (df[xlabel].min(), df[xlabel].max())
        ymin, ymax = (df[ylabel].min(), df[ylabel].max())
        Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
                                                np.linspace(ymin, ymax, Ny)))).T
        
        # Create and fit the model
        kde1 = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde1 = kde1.fit(X)
        # Evaluating at gridpoints
        log_dens1 = kde1.score_samples(Xgrid)
        dens1 = X.shape[0] * np.exp(log_dens1).reshape((Ny, Nx))
        # Evaluating at satellite points
        log_satdens = kde1.score_samples(X)
        satdens = X.shape[0] * np.exp(log_satdens)
        print(satdens)
        df[ExpLabel] = satdens
        
        # Plot the figure
        if plot==True:
            ind = rowcol_2_index(plt_row,plt_col,num_row,num_col)
            #ax = axs[ind]
            ax = fig.add_subplot(num_row,num_col,index+1)
            im = ax.imshow(dens1, origin='lower', 
                      # norm=LogNorm(),
                      # cmap=plt.cm.binary,
                      cmap=plt.cm.gist_heat_r,
                      extent=(xmin, xmax, ymin, ymax),aspect = 'auto' )
            plt.imshow(dens1, origin='lower', 
                      # norm=LogNorm(),
                      # cmap=plt.cm.binary,
                      cmap=plt.cm.gist_heat_r,
                      extent=(xmin, xmax, ymin, ymax),aspect = 'auto' )
            plt.colorbar(im,ax=ax,label='density')
            ax.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='b') # Add points
            if normalized==False:
                ax.set(xlabel=xlabel, ylabel=ylabel)
            else:
                ax.set(xlabel=xlabel, ylabel=ylabel)
                
            # Creat colorbar
        plt.show()      
    return df

def rowcol_2_index(row,col,nrows,ncols):
    ind = col + (row-1)*ncols-1
    return ind