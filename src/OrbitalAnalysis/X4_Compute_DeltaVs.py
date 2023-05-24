# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:22:34 2023

@author: scott

Compute Delta-Vs
----------------

Compute Delta-Vs of optimal two-impulse transfers between pairs of satellites, 
and save data to csv. This data will be used for orbital logistics problems
related to active debris removal.

Two modes:
    1-to-N: Compute delta-Vs from the satellite to all other objects
    kNN:    Select the k nearest neighbors of the object, and compute pairwise
            delta-Vs between them (all permutations).

Inputs to change:
    num_processes: The number of CPU cores to use in parallel.
                   Suggest to keep at least 1 CPU free for other tasks.
    target: the NoradId of the object of interest
    mode: select either '1-to-N' or 'kNN'
    k: the number of neighbors to consider

Output data
Data is saved to csv files. The files have different name conventions for the two modes.

1-to-N:
    filename = deltaVs_1-to-N_from_norad_<NoradId of target>.csv 
    columns: 'to_norad': NoradId of the 2nd orbit. 
             (from_norad is understood to be the target norad in the filename)
             'dV' : the delta-V of the transfer from the taget to the to_norad (km/s)

kNN:
    filename = deltaVs_N-to-N_kNN_<k>_around_norad_<NoradId of target>.csv
    columns: 'from_norad': NoradId of the 1st orbit. 
             'to_norad': NoradId of the 2nd orbit. 
             'dV' : the delta-V of the transfer from the taget to the to_norad (km/s)

NOTE: This script uses multiprocessing
      It must be run from the command line!
      It causes problems when run in an IDE like Spyder
>> cd </path/to/X4_Compute_DeltaVs.py>
>> python X4_Compute_DeltaVs.py

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import itertools

import multiprocessing as mp
from multiprocessing import Pool, freeze_support, RLock
import signal
import time

import pdb

from SatelliteData import *
from DistanceAnalysis import *
# from Visualization import *
# from Clustering import *

# from sr_tools.Astrodynamics.OrbitToOrbit import *
from OrbitToOrbit import *

from utils import get_data_home

# Supress pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Supress numpy warnings
np.seterr(invalid='ignore')

#%% Main functions

def compute_deltaVs_1toN(target,num_processes,mp_method=1):
    '''
    Compute delta-Vs from a single target to all other objects.
    
    Saves results as a csv with columns 'to_norad' and 'dV'

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    # Convert angles to radians
    df['i'] = np.deg2rad(df['i'])
    df['om'] = np.deg2rad(df['om'])
    df['w'] = np.deg2rad(df['w'])
    
    if mp_method == 1:
        # Multiprocessing method 1.
        
        # 1. Create argument list
        argument_list, from_norad, to_norad = create_arg_list1(df,target,'1-to-N',k=None)
        del df # Free up memory
        
        # Print out
        print('\nRunning Combinatorial Delta-V Calculations \n\n')
        print('1-to-N mode: Compute delta-Vs from taget body to all others')
        print('Central body NoradID = {}'.format(target))
        print('Number of combinations: {}'.format(str(len(argument_list))))
        print('Number of cores: {} \n'.format(str(num_processes)))
        
        # Task 1: 
        # Runtime ~39 mins
        t0 = time.time() # Start timer
        result = control_task1(num_processes,argument_list)
        t1 = time.time()
        print('Runtime {} min\n\n'.format((t1-t0)/60.))
    
    # Construct results dataframe
    res = pd.DataFrame(columns=['from_norad','to_norad','dV'])
    # Fill in to and from NoradIds
    res['from_norad'] = from_norad
    res['to_norad'] = to_norad
    
    # Add result to dataframe
    res['dV'] = result
    
    # Get data directory
    DATA_DIR = get_data_home()
    _dir = DATA_DIR/'Delta-Vs' # Save directory
    _dir.mkdir(parents=True, exist_ok=True) # Create path if doesn't exist
    
    # Save data
    filename = str(_dir/'deltaVs_1-to-N_from_norad_{}.csv'.format(target))
    res = res[['to_norad','dV']]
    res.to_csv(filename,index=False)
    
    print('Data saved to {}'.format(filename))
    
    return

def compute_deltaVs_kNN(target,num_processes,mp_method=1,k=100,):
    '''
    Compute delta-Vs from a single target to k nearest neighbors

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    # Convert angles to radians
    df['i'] = np.deg2rad(df['i'])
    df['om'] = np.deg2rad(df['om'])
    df['w'] = np.deg2rad(df['w'])
    
    if mp_method == 1:
        # Multiprocessing method 1.
        
        # 1. Create argument list
        argument_list, from_norad, to_norad = create_arg_list1(df,target,'N-to-N',k=k)
        
        # Print out
        print('\nRunning Combinatorial Delta-V Calculations \n\n')
        print('kNN mode: Compute pairwise delta-Vs between k nearest neighbors of central body')
        print('Central body NoradID = {}'.format(target))
        print('k = {}'.format(k))
        print('Number of combinations: {}'.format(str(len(argument_list))))
        print('Number of cores: {} \n'.format(str(num_processes)))
        
        
        # Task 1: 
        # Runtime ~39 mins
        t0 = time.time() # Start timer
        result = control_task1(num_processes,argument_list)
        t1 = time.time()
        print('Runtime {} min\n\n'.format((t1-t0)/60.))
    
    # Construct results dataframe
    res = pd.DataFrame(columns=['from_norad','to_norad','dV'])
    # Fill in to and from NoradIds
    res['from_norad'] = from_norad
    res['to_norad'] = to_norad
    
    # Add result to dataframe
    res['dV'] = result
    
    # Get data directory
    DATA_DIR = get_data_home()
    _dir = DATA_DIR/'Delta-Vs' # Save directory
    _dir.mkdir(parents=True, exist_ok=True) # Create path if doesn't exist
    
    # Save data
    filename = str(_dir/'deltaVs_N-to-N_kNN_{k}_around_norad_{target}.csv'.format(k=k,target=target))
    res.to_csv(filename,index=False)
    
    print('Data saved to {}'.format(filename))
    
    return

#%% Multiprocessing implementation
# Method 1
# Create a single list of tasks to perform, and use n processors to do it.
# Tested: ~ 40 mins to run
    
# Function to allow handling of KeyboardInterupt
def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def control_task1(num_processes,argument_list):
    '''
    Main control script to run delta-V computations for approach 1.
    
    Single list of tasks that are computed by multiple processors.

    Parameters
    ----------
    num_processes : int
        Number of CPU processors to use in parallelization.

    Returns
    -------
    result_list : 1xN array
        List of computed delta-Vs

    '''
    
    # Define details of processes
    num_jobs = len(argument_list) # Number of jobs
    func = func_task1 # Name of function to apply
    
    # # Print out
    # print('\nRunning Combinatorial Delta-V Calculations \n\n')
    # print('Number of combinations: {}'.format(str(len(argument_list))))
    # print('Number of cores: {} \n'.format(str(num_processes)))
    
    # Apply function to process
    freeze_support() # For windows
    pool = Pool(processes=num_processes,initializer=initializer) # Create pool
    # try:
    jobs = [pool.apply_async(func=func, args=(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,)) for arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10 in argument_list]
    pool.close()
    # except KeyboardInterrupt:
    #     pool.terminate()
    #     pool.join()
    
    
    result_list = []
    for job in tqdm(jobs):
        result_list.append(job.get())
    
    
    
    return result_list


def func_task1(a1,e1,i1,om1,w1,a2,e2,i2,om2,w2, *args, **kwargs):
    '''
    Main function executed by each processor for method 1.
    Computes the delta-V of an optimal two-impulse transfer between orbits.

    Parameters
    ----------
    a1, e1, i1, om1, w1 : int
        Orbital elements of 1st orbit.
    a2, e2, i2, om2, w2 : int
        Orbital elements of nd orbit.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    dV : int
        Delta-V of two-impulse transfer between orbits 1 and 2.

    '''
    
    # 
    
    # arg1 = x1
    # arg2 = x2
    
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Extract values from row  
    # Note: angles already in radians
    orb1 = {'a':a1,'e':e1,'w':w1,'i':i1,'om':om1}
    orb2 = {'a':a2,'e':e2,'w':w2,'i':i2,'om':om2}
    # Solve Orbit to Orbit problem
    result = solve_OrbitToOrbit_mccue(orb1,orb2,mu,solver='Grid')
    dV = result.fun
    
    
    return dV
    
    

def create_arg_list1(df,target,prob_type,k=100):
    '''
    Create the list of arguments to be parsed into func_task1.

    Returns
    -------
    argument_list : list of tuples
        DESCRIPTION.

    '''

    if prob_type == '1-to-N':
        # Compute the delta-Vs from single target to all other satellites.
        
        dfnear = df.copy() # All objects
        
        # Add columns Point 1
        # dfnear['from_norad'] = target
        dfnear['a1'] = df['a'][df.NoradId == target].iloc[0]
        dfnear['e1'] = df['e'][df.NoradId == target].iloc[0]
        dfnear['i1'] = df['i'][df.NoradId == target].iloc[0]
        dfnear['om1'] = df['om'][df.NoradId == target].iloc[0]
        dfnear['w1'] = df['w'][df.NoradId == target].iloc[0]
    
        # Rename Point 2
        # dfnear['to_norad'] = dfnear.NoradId
        dfnear = dfnear.rename(columns = {'a':'a2','e':'e2','i':'i2','om':'om2','w':'w2'})
        
        # Extract aguments as list
        a1 = dfnear['a1'].tolist()
        e1 = dfnear['e1'].tolist()
        i1 = dfnear['i1'].tolist()
        om1 = dfnear['om1'].tolist()
        w1 = dfnear['w1'].tolist()
        
        a2 = dfnear['a2'].tolist()
        e2 = dfnear['e2'].tolist()
        i2 = dfnear['i2'].tolist()
        om2 = dfnear['om2'].tolist()
        w2 = dfnear['w2'].tolist()
        
        # # Extract from_norad and to_norad lists
        from_norad = [target]*len(a1)
        to_norad = dfnear['NoradId'].tolist()
    
    elif prob_type == 'N-to-N':
        # Compute pairwise delta-Vs between k nearest neighbors of target
        # TODO: Finish off
        
        # Compute distance metrics
        df = compute_distances(df,target,searchfield='NoradId')
        df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
        
        # Closest N objects (using cylindrical distance)
        N = k # Number of objects
        dfnear = df.nsmallest(N, 'd_cyl') # Nearest
        
        # Extract list of norad_ids
        norad_list = dfnear.NoradId.tolist()
        
        # Create pair-wise combinations of these norads
        # pairs = list(itertools.combinations(norad_list, 2))
        pairs = list(itertools.permutations(norad_list, 2)) # Order is important
        # Create new dataframe
        df1 = pd.DataFrame(data=pairs,columns=['from_norad','to_norad'])
        # Merge data for 'from_norad' node
        df1 = pd.merge(df1,dfnear[['NoradId','a','e','i','om','w']],how='left',left_on='from_norad',right_on='NoradId')
        df1 = df1.rename(columns = {'a':'a1','e':'e1','i':'i1','om':'om1','w':'w1'})
        # Merge data for 'to_norad' node
        df1 = pd.merge(df1,dfnear[['NoradId','a','e','i','om','w']],how='left',left_on='to_norad',right_on='NoradId')
        df1 = df1.rename(columns = {'a':'a2','e':'e2','i':'i2','om':'om2','w':'w2'})
        
        # Extract aguments as list
        a1 = df1['a1'].tolist()
        e1 = df1['e1'].tolist()
        i1 = df1['i1'].tolist()
        om1 = df1['om1'].tolist()
        w1 = df1['w1'].tolist()
        
        a2 = df1['a2'].tolist()
        e2 = df1['e2'].tolist()
        i2 = df1['i2'].tolist()
        om2 = df1['om2'].tolist()
        w2 = df1['w2'].tolist()
        
        # Extract from_norad and to_norad lists
        from_norad = df1['from_norad'].tolist()
        to_norad = df1['to_norad'].tolist()

    
    # Zip together into argument list
    argument_list = [(a1i,e1i,i1i,om1i,w1i,a2i,e2i,i2i,om2i,w2i) for a1i,e1i,i1i,om1i,w1i,a2i,e2i,i2i,om2i,w2i in zip(a1,e1,i1,om1,w1,a2,e2,i2,om2,w2) ]
    
    return argument_list, from_norad, to_norad

#%% Method 2
    
# FIXME: results are not returned yet
# Split the total list of computations into chuncks for n processors. 

def control_task2(num_processes):
    '''
    Main control script to run delta-V computations for approach 2.
    Splits the computations into chunks for n processors. Each processor
    then iterates through its list of tasks.

    Parameters
    ----------
    num_processes : int
        Number of CPU processors to use in parallelization.

    Returns
    -------
    result_list : 1xN array
        List of computed delta-Vs

    '''
    
    # 1. Create argument list
    argument_list = create_arg_list2(num_processes)
    
    # Printout
    print('\n\n\nTask2: Calculating Delta-Vs in chunks')
    # print('Number of calculations: {}'.format(str(len(pred_files))))
    print('Number of cores: {} \n'.format(str(num_processes)))
    
    # Apply function to process
    func = func_task2
    freeze_support() # For Windows support
    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    jobs = [pool.apply_async(func, args=(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,)) for arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11 in argument_list]
    # jobs = [tqdm( pool.apply_async(func, args=(arg1,arg2,)) ) for arg1,arg2 in argument_list]
    # jobs = list(tqdm.tqdm(pool.imap_unordered(myFunction, array))
    pool.close()
    result_list = [job.get() for job in jobs]
    
    print('\n\n\n\n')
    
    return result_list

def func_task2(a1_list,e1_list,i1_list,om1_list,w1_list,a2_list,e2_list,i2_list,om2_list,w2_list, pID, *args, **kwargs):
    
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Note: inputs are lists
    tqdm_text = "CPU #{}".format(pID).zfill(2) # Progress bar label
    with tqdm(total=len(a1_list), desc=tqdm_text, position=pID,leave=True) as pbar:
        for i in range(len(a1_list)):
            
            # Extract values from row  
            orb1 = {'a':a1_list[i],'e':e1_list[i],'w':np.deg2rad(w1_list[i]),
                'i':np.deg2rad(i1_list[i]),'om':np.deg2rad(om1_list[i])}
            orb2 = {'a':a2_list[i],'e':e2_list[i],'w':np.deg2rad(w2_list[i]),
                'i':np.deg2rad(i2_list[i]),'om':np.deg2rad(om2_list[i])}
            # Solve Orbit to Orbit problem
            result = solve_OrbitToOrbit_mccue(orb1,orb2,mu,solver='Grid')
            dV = result.fun
            
            # Update progress bar
            pbar.update(1)
    
    
    return {}

def create_arg_list2(num_processors):
    
    # Create list of arguments to be parsed into func_task1

    # Load data
    df = load_satellites(group='all',compute_params=True,compute_pca=True)
    # df = load_2019_experiment_data([36]) # New dataset
    
    
    # Select target
    # target = 25544 # ISS
    # target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    target = 49445 # Atlas 5 Centaur DEB
    
    # # Compute distance metrics
    # df = compute_distances(df,target,searchfield='NoradId')
    # df = df.rename(columns={'dH':'d_Euc','dHtheta_arc':'d_arc','dHcyl':'d_cyl'}) # Rename
    
    # 1. Closest N objects (using cylindrical distance)
    # N = 1000 # Number of objects
    # dfnear = df.nsmallest(N, 'd_cyl') # Nearest
    # dfnear = df.sample(n=N) # Random sample
    dfnear = df.copy() # All objects
    # dfnear = dfnear.head(100)
    
    # Add columns Point 1
    # dfnear['from_norad'] = target
    dfnear['a1'] = df['a'][df.NoradId == target].iloc[0]
    dfnear['e1'] = df['e'][df.NoradId == target].iloc[0]
    dfnear['i1'] = df['i'][df.NoradId == target].iloc[0]
    dfnear['om1'] = df['om'][df.NoradId == target].iloc[0]
    dfnear['w1'] = df['w'][df.NoradId == target].iloc[0]
    
    
    # Rename Point 2
    # dfnear['to_norad'] = dfnear.NoradId
    dfnear = dfnear.rename(columns = {'a':'a2','e':'e2','i':'i2','om':'om2','w':'w2'})
    
    # Extract aguments as list
    a1 = dfnear['a1'].tolist()
    e1 = dfnear['e1'].tolist()
    i1 = dfnear['i1'].tolist()
    om1 = dfnear['om1'].tolist()
    w1 = dfnear['w1'].tolist()
    
    a2 = dfnear['a2'].tolist()
    e2 = dfnear['e2'].tolist()
    i2 = dfnear['i2'].tolist()
    om2 = dfnear['om2'].tolist()
    w2 = dfnear['w2'].tolist()
    
    # Divide list into n groups
    n = num_processes
    k, m = divmod(len(a1), n) # Length of each group
    
    # Format arguments list for process function (vars_list,label,id)
    arg_1 = list(a1[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_2 = list(e1[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_3 = list(i1[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_4 = list(om1[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_5 = list(w1[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_6 = list(a2[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_7 = list(e2[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_8 = list(i2[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_9 = list(om2[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    arg_10 = list(w2[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    arg_11 = np.arange(n) # Processor id
    
    # Zip together into argument list
    argument_list = [(a1i,e1i,i1i,om1i,w1i,a2i,e2i,i2i,om2i,w2i,pIDi) for a1i,e1i,i1i,om1i,w1i,a2i,e2i,i2i,om2i,w2i,pIDi in zip(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11) ]
    
    return argument_list


#%% Function to implement

if __name__ == "__main__":
    
    # Select target
    # target = 25544 # ISS
    target = 22675 # Cosmos 2251 *
    # target = 13552 # Cosmos 1408 *
    # target = 25730 # Fengyun 1C *
    # target = 40271 # Intelsat 30 (GEO)
    # target = 49445 # Atlas 5 Centaur DEB
    
    # Mode inputs (Select only 1. Comment out the other.)
    # mode = '1-to-N'     # Caluculate delta-Vs from target to all other objects
    mode, k = 'kNN', 100 # Caluculate delta-Vs from between k nearest neighbors of target
    
    # Processor inputs
    num_processes = 6 # Number of parallel cores to use
    
    if mode == '1-to-N':
        # # Caluculate delta-Vs from target to all other objects
        compute_deltaVs_1toN(target,num_processes,mp_method=1)
    elif mode == 'kNN':
        # Caluculate delta-Vs from between k nearest neighbors of target
        compute_deltaVs_kNN(target,num_processes,k=k,mp_method=1)


