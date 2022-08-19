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
from utils import get_data_home

# from Overpass import *
# from Ephem import *
# from Events import *
# from GmatScenario import *

# Package imports
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import pdb
import time

def density_analysis(x,y):
    # This still works if you want to plot a single figure.
    # Depretiate in future???
    
    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
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
    '''
    Cross validation of KDE

    Parameters
    ----------
    filename : str
        Name of the config file.

    Returns
    -------
    grid : TYPE
        DESCRIPTION.

    '''
    # Load experiments config file
    full_filename = get_data_home()/"DIT_Experiments"/"KDE"/"CV"/filename
    df_config = pd.read_csv(full_filename)
    
    # Load the satellite data and compute orbital parameters
    df = load_satellites(group='all',compute_params=True)
    
    # Extract parameters
    kernel = df_config['Value'][df_config.Parameter == 'kernel'].iloc[0]
    spacing_type = df_config['Value'][df_config.Parameter == 'spacing_type'].iloc[0]
    bandwidth_max = float(df_config['Value'][df_config.Parameter == 'bandwidth_max'].iloc[0])
    bandwidth_min= float(df_config['Value'][df_config.Parameter == 'bandwidth_min'].iloc[0])
    N = int(df_config['Value'][df_config.Parameter == 'N'].iloc[0])
    # xlabel = df_config['Value'][df_config.Parameter == 'xlabel'].iloc[0]
    # ylabel = df_config['Value'][df_config.Parameter == 'ylabel'].iloc[0]
    normalized = df_config['Value'][df_config.Parameter == 'normalized'].iloc[0]
    params = df_config['Value'][df_config.Parameter == 'parameters'].iloc[0]
    
    # Parse string list of params to list
    params_list = params.strip('][)(').split(',')
    params_list = [p.strip() for p in params_list] # Strip leading+trailing spaces
    

    
    
    # Extract data
    # X = df[[xlabel,ylabel]].to_numpy()
    X = df[params_list].to_numpy()
    
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
        bandwidths = np.logspace(bandwidth_min,bandwidth_max,N)
    
    # Solver settings
    cv = 5     # Cross validation method
    n_jobs = -1 # Number of jobs to compute in parallel # Use -1 to use every core
    
    
    # Print settings
    print('\n\nRunning cross validation')
    print('------------------------')
    print('KDE Settings:')
    print('Variables: {}'.format(params_list))
    print('Kernel: {}'.format(kernel))
    print('Parameter: bandwidth')
    
    print('\nSolver Settings:')
    print('Parameter values: {}'.format(bandwidths))
    print('K-folds: {}'.format(cv))
    print('N jobs: {}'.format(n_jobs))
    
    
    # Create CV grid
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                    {'bandwidth': bandwidths},
                    cv=cv, #LeaveOneOut(),
                    n_jobs=n_jobs,
                    )
    start = time.time()
    grid.fit(X) # Fit the data (time the function)
    print('Runtime: {}'.format(time.time() - start)) 
    
    # Extract results
    # opt_bandwidth = grid.best_params_['bandwidth']
    
    dfresults = pd.DataFrame(grid.cv_results_)
    
    outfile = filename.split('.csv')[0] + '_results.csv'
    full_outfilename = full_filename.parent/outfile
    dfresults.to_csv(str(full_outfilename))
    
    # TODO: Print results summary
    
    # Plot results
    fig, ax = plt.subplots(1,1,figsize=(8, 8))
    ax.plot(dfresults.param_bandwidth,dfresults.mean_test_score,'-k')
    ax.set_xlabel('bandwidth')
    ax.set_ylabel('score (total log-likelihood)')
    ax.set_yscale('log')
    if spacing_type == 'logspace':
        ax.set_xscale('log')
    plt.show()
    
    # Print output values
    print('\nResults')
    print('-------')
    print('Mean Scores: {}'.format(dfresults.mean_test_score.to_numpy()))
    
    # Print optimal bandwidth
    ind = np.argmax(dfresults.mean_test_score) # Index of peak
    opt_bandwidth = dfresults['param_bandwidth'].iloc[ind]
    print('Optimal binwidth: {}'.format(opt_bandwidth))
        
    return grid
    

#%% Density Values for Satellites

def evaluate_satellite_densities(save_output=True):
    '''
    Apply KDE using a number of different parameter sets, with the optimal
    kernel and bandwidth identified from cross validation steps. Evauate the
    log-likelihood density values at the associated coordinates of all satellites
    in the catalog. Return a dataframe containing normalized parameters and
    the log-likelihood density values.

    Returns
    -------
    df : Dataframe
        Dataframe with normalized parameters and log-likelihood measurements
        for a number of parameter combinations.

    '''
    
    
    # Import dataset
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # Limit data fields
    df = df[['Name','NoradId','a','e','i','om','w','M','p','q','Q',
             'h','hx','hy','hz','hphi','htheta',
             'PC1','PC2','PC3','PC4','PC5']]
    
    # Scale all data fields independently to 0-1
    min_max_scaler = preprocessing.MinMaxScaler()
    features = ['a','e','i','om','w','M','p','q','Q',
                'h','hx','hy','hz','hphi','htheta',
                'PC1','PC2','PC3','PC4','PC5',
                ]
    Xfull = df[features].to_numpy()
    Xfull = min_max_scaler.fit_transform(Xfull)
    df[features] = Xfull
    
    # Apply KDE and evaluate at satellite points.
    # Uncomment/add different parameter sets
    
    # 2D Combinations
    
    # 1. (a,e)
    X = df[['a','e']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.004125, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_ae=log_satdens)
    
    # 2. (a,i)
    X = df[['a','i']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.0034000000000000002, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_ai=log_satdens)
    
    # 3. (hx,hy)
    X = df[['hx','hy']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.00725, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hxhy=log_satdens)
    
    # 4. (hy,hz)
    X = df[['hy','hz']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.00765, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hyhz=log_satdens)
    
    # 5. (hx,hz)
    X = df[['hx','hz']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.006158482110660267, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hxhz=log_satdens)
    
    # 6. (h,hz)
    X = df[['h','hz']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.004281332398719396, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hhz=log_satdens)
    
    # 7. (h,htheta)
    X = df[['h','htheta']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.004281332398719396, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hhtheta=log_satdens)
    
    # 3D Combinations
    
    # Orbital Element Space (a,e,i)
    X = df[['a','e','i']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.006158482110660267, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_aei=log_satdens)
    
    # Angular momentum space (hx,hy,hz)
    X = df[['hx','hy','hz']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.008858667904100823, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_hxhyhz=log_satdens)
    
    # Principal Component Analysis
    
    # PC1,PC2
    X = df[['PC1','PC2']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.008858667904100823, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_PC1PC2=log_satdens)
    
    # PC1,PC2,PC3
    X = df[['PC1','PC2','PC3']].to_numpy()
    kde1 = KernelDensity(bandwidth=0.008858667904100823, kernel='gaussian')
    kde1 = kde1.fit(X)
    # Evaluating at satellite points
    log_satdens = kde1.score_samples(X)
    # Append to dataframe
    df = df.assign(p_PC1PC2PC3=log_satdens)
    
    
    # Plot Orbital Momentum
    # plot_h_space_numeric(df,color='p_hxhyhz',logColor=False,colorscale='Blackbody_r')
    
    
    # 
    
    # Plot
    # plt.scatter(df.h,df.hz,c=df.p_hhz.to_numpy(),s=0.1)
    
    # Save results
    if save_output:
        outfile = get_data_home()/"DIT_Experiments"/"KDE"/"Loglikelihood.csv"
        df.to_csv(outfile,index=False)
    
    
    return df

# Load results
def load_density_values():
    '''
    Load the density values from Loglikelihood.csv

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    filename = get_data_home()/"DIT_Experiments"/"KDE"/"Loglikelihood.csv"
    
    # Check if file exists
    if os.path.isfile(filename) == False:
        # Download it
        df = evaluate_satellite_densities(save_output=True)
    else:
        # Load saved file
        df = pd.read_csv(filename)
    
    return df
#%% Analysis for IAC Paper

# Coordinate systems
def explore_coordinates():
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # Load in density results
    dfd = load_density_values()
    # Merge
    df = pd.merge(df,dfd,how='left',on=['Name','NoradId'],suffixes=['','_norm'])
    
    # 2D Scatter Plots --------------------------------------------------------
    
    # Figure 1. a-e, a-i, om-i
    fig, axs = plt.subplots(1,3,figsize=(16, 5))
    # a,e
    axs[0].plot(df.a,df.e,'.k',markersize=0.2)
    axs[0].set_xlabel('Semi-major axis a (km)')
    axs[0].set_ylabel('Eccentricity e')
    axs[0].set_xlim([6000, 50000]) #
    # a,i
    axs[1].plot(df.a,df.i,'.k',markersize=0.2)
    axs[1].set_xlabel('Semi-major axis a (km)')
    axs[1].set_ylabel('Inclination i (deg)')
    axs[1].set_xlim([6000, 50000]) #
    # om,i
    axs[2].plot(df.om,df.i,'.k',markersize=0.2)
    axs[2].set_xlabel('Right ascension of Ascending Node om (deg)')
    axs[2].set_ylabel('Inclination i (deg)')
    plt.show()

    
    # Figure 2. 3D hx,hy,hz
    plot_3d_scatter_numeric(df,'hx','hy','hz',color=None,
                            xrange=[-120000,120000],
                            yrange=[-120000,120000],
                            zrange=[-50000,150000],
                            aspectmode='cube',
                            )
    
    
    
    # Figure 5. h-hz, PC1-PC2
    # Figure 1. a-e, a-i, om-i
    fig, axs = plt.subplots(1,2,figsize=(12, 5))
    # a,e
    axs[0].plot(df.h,df.hz,'.k',markersize=0.2)
    axs[0].set_xlabel('Angular momentum magntiude $h$ ($km^{2}/s$)')
    axs[0].set_ylabel('Angular momentum z-component $h_{z}$ ($km^{2}/s$)')
    axs[0].set_xlim([50000, 150000]) #
    axs[0].set_ylim([-50000, 150000]) #
    # PC1-PC2
    axs[1].plot(df.PC1,df.PC2,'.k',markersize=0.2)
    axs[1].set_xlabel('PC1')
    axs[1].set_ylabel('PC2')
    axs[1].set_xlim([-50000, 150000]) #
    axs[1].set_ylim([-100000, 100000]) #
    # fig.tight_layout()
    # plt.subplot_tool()
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    
    
    # PC 3D satter
    plot_3d_scatter_numeric(df,'PC1','PC2','PC3',color=None,
                            xrange=[-50000, 150000],
                            yrange=[-100000, 100000],
                            zrange=[-100000, 100000],
                            aspectmode='cube',
                        )
    
    return


# Principal Component Analysis
def analyze_principal_components():
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    
    # Select features from dataset to include.
    # Exclude orbital parameters, since they are linear combinations of 
    # orbital elements
    features = ['a','e','i','om','w','n','h','hx','hy','hz']
    Xfull = df[features] # Extract feature data
    
    # Principal components
    # Run PCA on all numeric orbital parameters
    n_components = 10
    pca = PCA(n_components)
    pca.fit(Xfull)
    PC = pca.transform(Xfull)
        
    
    # Variance explained by each PC
    pca.explained_variance_ratio_
    # Feature importance
    # PC1,PC2,PC3: 4.27367824e-01, 2.86265363e-01, 2.60351753e-01, 
    # PC4,PC5,PC6: 2.21499988e-02, 3.86125358e-03, 2.54105655e-06, 
    # PC7,PC8,PC9: 1.26163922e-06, 4.55552048e-09, 6.39255498e-10, 
    # PC10: 5.17447310e-13
    
    # Plot of feature explaination
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.arange(n_components)+1,np.cumsum(pca.explained_variance_ratio_)*100)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Explained variance (%)')
    ax.set_xticks(np.arange(n_components))
    plt.show()
    
    # Most of variance explained by first 3 comonents
    # Drops off after that
    
    # Importance of each feature
    # Contribution of each input feature to each output PC
    # PC1: print(abs( pca.components_[0,:] ))
    # 
    # Main importance is contributed from
    # 'a','h','hx','hy','hz' 
    
    # Format feature importance into a dataframe
    labels = ['PC'+str(i+1) for i in range(n_components)]    
    dffeatimp = pd.DataFrame(pca.components_.T,columns=labels)
    dffeatimp.insert(0,'Feature',features)
    dffeatimp.set_index('Feature',inplace=True)
    
    # Heatmap of feature importance
    import seaborn as sns
    sns.heatmap(dffeatimp.abs(), annot=True)
    
    return

# Histograms
def plot_histograms():
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    
    # Plot the histograms in section 2.
    
    fig, axs = plt.subplots(2,3,figsize=(12, 8))
    # Row 1: h,hz,a
    # h
    ax = axs[0,0]
    ax.hist(df.h, bins=50,label='$a$ (km)')
    ax.set_xlabel('$h$')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    # hz
    ax = axs[0,1]
    ax.hist(df.hz, bins=50,label='$a$ (km)')
    ax.set_xlabel('$h_{z}$')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    # a
    ax = axs[0,2]
    ax.hist(df.a, bins=50,label='$a$ (km)')
    ax.set_xlabel('$a$')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    
    # Row 2: PC1,PC2,PC3
    # PC1
    ax = axs[1,0]
    ax.hist(df.PC1, bins=50,label='$a$ (km)')
    ax.set_xlabel('$PC_{1}$')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    # PC2
    ax = axs[1,1]
    ax.hist(df.PC2, bins=50,label='$a$ (km)')
    ax.set_xlabel('$PC_{2}$')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    # PC3
    ax = axs[1,2]
    ax.hist(df.PC3, bins=50,label='$a$ (km)')
    ax.set_xlabel('$PC_{3}$')
    ax.set_yscale('log')
    ax.set_ylim([0.1,1E5])
    plt.show()
    
    
    # Plot 2d Histograms
    
    # Create a black and white color map where bad data (NaNs) are white
    cmap = plt.cm.binary
    
    # h vs hz
    Nx,Ny = 50,50
    
    # axs[0].set_xlim([50000, 150000]) #
    # axs[0].set_ylim([-50000, 150000]) #
    
    
    
 
    # Use the image display function imshow() to plot the result
    fig, axs = plt.subplots(1,2,figsize=(20, 8))
    
    # 1. h vs hz
    # Extract data
    x = df['h'].to_numpy()
    y = df['hz'].to_numpy()
    # Define bins
    # binsx = np.linspace(min(x),max(x),Nx)
    # binsy = np.linspace(min(y),max(y),Ny)
    binsx = np.linspace(50000, 150000,Nx)
    binsy = np.linspace(-50000, 150000,Ny)
    # Compute and plot 2D histogram
    H, xbins, ybins = np.histogram2d(x, y,bins=(binsx,binsy))
    ax = axs[0]
    im = ax.imshow(H.T, origin='lower',
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              # cmap=cmap, #interpolation='nearest',
              norm=matplotlib.colors.LogNorm(),
              cmap=plt.cm.gist_heat_r,
              aspect='auto')
    ax.set_xlabel('h')
    ax.set_ylabel('hz')
    ax.set_xlim(min(binsx), max(binsx))
    ax.set_ylim(min(binsy), max(binsy))
    plt.colorbar(im,ax=ax,label='Num per pixel')
    
    # 2. PC1 vs PV2
    # Extract data
    x = df['PC1'].to_numpy()
    y = df['PC2'].to_numpy()
    # Define bins
    # binsx = np.linspace(min(x),max(x),Nx)
    # binsy = np.linspace(min(y),max(y),Ny)
    binsx = np.linspace(-50000, 150000,Nx)
    binsy = np.linspace(-100000, 100000,Ny)
    # Compute and plot 2D histogram
    H, xbins, ybins = np.histogram2d(x, y,bins=(binsx,binsy))
    ax = axs[1]
    im = ax.imshow(H.T, origin='lower',
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              # cmap=cmap, #interpolation='nearest',
              norm=matplotlib.colors.LogNorm(),
              cmap=plt.cm.gist_heat_r,
              aspect='auto')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_xlim(min(binsx), max(binsx))
    ax.set_ylim(min(binsy), max(binsy))
    plt.colorbar(im,ax=ax,label='Num per pixel')
    plt.subplots_adjust(wspace=0.3)
    
    
    return

# Analysize density results
def analyze_density_results():
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # Load in density results
    dfd = load_density_values()
    # Merge
    df = pd.merge(df,dfd,how='left',on=['Name','NoradId'],suffixes=['','_norm'])
    
    # Histograms of values (plotly)
    # fig = go.Figure(data=[go.Histogram(x=df.p_PC1PC2PC3)]) # PC1
    # fig.add_trace(go.Histogram(x=df['p_hxhyhz'],))
    # fig.update_layout(barmode='overlay')
    # fig.add_trace(go.Histogram(x=df['p_aei'],))
    # plotly.offline.plot(fig)
    
    # 1. Investigate the distribution of values in each -----------------------
    # Also the correlation between the values 
    
    # Histograms matplotlib
    fig, axs = plt.subplots(3,1,figsize=(8, 8))
    axs[0].hist(df.p_aei, bins=50,label='Orbital Elements')
    axs[0].set_xlabel('Log-likelihood (a,e,i)')
    axs[1].hist(df.p_hxhyhz, bins=50,label='Angular Momentum')
    axs[1].set_xlabel('Log-likelihood (hx,hy,hz)')
    axs[2].hist(df.p_PC1PC2PC3, bins=50,label='Principal Components')
    axs[2].set_xlabel('Log-likelihood (PC1,PC2,PC3)')
    plt.show()
    
    # Show correlation between PCA and Angular Momentum
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df.p_PC1PC2PC3,df.p_hxhyhz,'.k',markersize=0.2)
    ax.set_xlabel('Log-likelihood (PC1,PC2,PC3)')
    ax.set_ylabel('Log-likelihood (hx,hy,hz)')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df.p_PC1PC2PC3,df.p_aei,'.k',markersize=0.2)
    ax.set_xlabel('Log-likelihood (PC1,PC2,PC3)')
    ax.set_ylabel('Log-likelihood (a,e,i)')
    plt.show()
    
    

    # plot_2d_scatter_numeric(df,'p_hxhyhz','p_PC1PC2PC3','OBJECT_TYPE',logColor=False,size=)
    
    #2. Plot the distribution of points and measures of their density ---------
    # Plot 3D scatter plots
    
    # Orbital Element Space
    plot_3d_scatter_numeric(df,'a_norm','e_norm','i_norm',color='p_aei',
                            logColor=False,colorscale='Blackbody_r',
                            xrange=[0,0.15],
                            filename = 'p_aei.html')

    
    
    # Angular Momentum Space
    plot_3d_scatter_numeric(df,'hx_norm','hy_norm','hz_norm',color='p_hxhyhz',
                            logColor=False,colorscale='Blackbody_r',
                            xrange=[0,0.8],
                            yrange=[0.2,1],
                            zrange=[0,0.8],
                            filename = 'p_hxhyhz.html')
    
    # plot_3d_scatter_numeric(df,'h','om','i',color='p_hxhyhz',
    #                             logColor=False,colorscale='Blackbody_r',
    #                             filename = 'p_hxhyhz.html')
    
    # # Principal Components (don't bother - already highly correlated to ang momentum)
    # plot_3d_scatter_numeric(df,'PC1','PC2','PC3',color='p_PC1PC2PC3',
    #                         logColor=False,colorscale='Blackbody_r',
    #                         filename = 'p_PCA.html')
    
    
    # 3D Scatter by Object type
    # plot_3d_scatter_cat(df,'hx','hy','hz', 'OBJECT_TYPE')
    
    
    
    return



