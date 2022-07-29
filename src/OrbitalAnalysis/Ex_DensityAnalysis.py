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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
# from Overpass import *
# from Ephem import *
# from Events import *
# from GmatScenario import *

# Package imports
import pdb
from sklearn.neighbors import KernelDensity

def density_analysis(x,y):
    # This still works if you want to plot a single figure.
    # Depretiate in future???
    
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
    '''
    Run a batch of experiments to compute the Kernel Density Estimate (KDE) of
    the satellite catalog under differnt kernel and bandwidth parameters.
    
    Input coordinates, kernel function and bandwidth in a config csv file.
    See example example_kde_inputs.csv in main directory of repo. 
    Copy this file to ~/satellite_data/Data/DIT_Experiments/KDE.

    Parameters
    ----------
    filename : TYPE
        Name of the input config file 
        (located in ~/satellite_data/Data/DIT_Experiments/KDE).
        E.g. 'inputs.csv'
    plot : Bool, optional
        Flag to optionally plot the outputs. The default is True.

    Returns
    -------
    df : TYPE
        Orignial dataframe with attached density estimate results for each setting.

    '''
    
    # Load experiments config file
    full_filename=get_data_home()/"DIT_Experiments"/"KDE"/filename
    df_config = pd.read_csv(full_filename)
    
    # Extract unique sets of coordinates used in input file
    coord_list = list(set(list(zip(df_config.xlabel, df_config.ylabel, df_config.normalized))))
    
    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    
    # Pre-process data for each set of unique coords
    # We do this to avoid repeating data processing on settings that use the
    # same set of coordinates.
    # For each unique set, extract and (optionally) normalize the feature data
    # and save it to a dictionary for lookup.
    data_dict = {} # Dictionary to store data
    for i in range(len(coord_list)):
        # Extract feature data
        xlabel = coord_list[i][0]
        ylabel = coord_list[i][1]
        normalized = coord_list[i][2]
        X = df[[xlabel,ylabel]].to_numpy()
        
        # Normalize data
        if normalized:
            print('Normalizing data')
            min_max_scaler = preprocessing.MinMaxScaler()
            X = min_max_scaler.fit_transform(X)
        # Store in dictionary for easy lookup
        key = str(coord_list[i])
        data_dict[key] = X
        
    
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
        
        # Extract feature data from data_dict
        key = str((xlabel,ylabel,normalized)) # Key in data_dict lookup table
        X = data_dict[key] # Extract feature data
        
        # Create grid to display the density plot
        # (Hess diagram - 2d histogram)
        Nx = row['Nx']
        Ny = row['Ny']
        # bandwidth = 10000
        # Extract limits of the image
        xmin, xmax = (X[:,0].min(), X[:,0].max())
        ymin, ymax = (X[:,1].min(), X[:,1].max())
        # Create a grid of coordinates
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
        satdens = X.shape[0] * np.exp(log_satdens) # FIXME: Look into normalization!!!
        # print(satdens)
        # Append results to the satellite dataframe
        df[ExpLabel] = satdens
        
        # Plot the figure
        if plot==True:
            # Add subplot to figure and create new axes
            ind = rowcol_2_index(plt_row,plt_col,num_row,num_col)
            ax = fig.add_subplot(num_row,num_col,index+1)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            # Plot the density image with colorbar
            im = ax.imshow(dens1, origin='lower', 
                      # norm=LogNorm(),
                      # cmap=plt.cm.binary,
                      cmap=plt.cm.gist_heat_r,
                      extent=(xmin, xmax, ymin, ymax),aspect = 'auto' )
            plt.colorbar(im,ax=ax,label='density')
            
            # Plot the satellite points
            ax.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='b') # Add points
            # Add x and y labels
            if normalized==False:
                ax.set(xlabel=xlabel, ylabel=ylabel)
            else:
                ax.set(xlabel=xlabel, ylabel=ylabel)
             
        # Render figure
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.show()

    return df

def rowcol_2_index(row,col,nrows,ncols):
    ''' Convert row and columumn indices to a flat index '''
    ind = col + (row-1)*ncols-1
    return ind

def cross_validation(filename):
    # Load experiments config file
    full_filename=get_data_home()/"DIT_Experiments"/"KDE"/"CV"/filename
    df_config = pd.read_csv(full_filename)
    
    # Extract parameters
    kernel = df_config['Value'][df_config.Parameter == 'kernel'].iloc[0]
    spacing_type = df_config['Value'][df_config.Parameter == 'spacing_type'].iloc[0]
    bandwidth_max = float(df_config['Value'][df_config.Parameter == 'bandwidth_max'].iloc[0])
    bandwidth_min= float(df_config['Value'][df_config.Parameter == 'bandwidth_min'].iloc[0])
    N = int(df_config['Value'][df_config.Parameter == 'N'].iloc[0])
    xlabel = df_config['Value'][df_config.Parameter == 'xlabel'].iloc[0]
    ylabel = df_config['Value'][df_config.Parameter == 'ylabel'].iloc[0]
    normalized = df_config['Value'][df_config.Parameter == 'normalized'].iloc[0]
    
    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    
    # Extract data
    X = df[[xlabel,ylabel]].to_numpy()
    # Normalize data
    if normalized:
        print('Normalizing data')
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
    
    # Create array of binwidth values
    if spacing_type == 'linspace':
        # Linearly spaced points
        bandwidths = np.linspace(bandwidth_min,bandwidth_max,N)
    else:
        # TODO: Add logspace
        pass
    
    # Create CV grid
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut()) 
    grid.fit(X) # Fit the data
    # Extract results
    opt_bandwidth = grid.best_params_['bandwidth']
    pdb.set_trace()
    
    