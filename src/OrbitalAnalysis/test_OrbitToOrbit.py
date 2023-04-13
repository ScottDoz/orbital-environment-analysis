# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:36:19 2021

@author: scott

Unit tests of Orbit-to-Orbit module
-----------------------------------

"""

import pandas as pd
import timeit
import time


from OrbitToOrbit import *

#%% Initial & Final orbits for Test Scenarios

def init_test_1():
    '''
    Example 1: Co-planar Circular to Elliptical orbit
    
    From Example 6.1 of Curtis (2005)
    Initial orbit: 480x800km altitude Earth orbit (a = 7018 km, e = 0.022799).
    Final orbit: 16000 km altitude circular orbit (a = 22378 km, e = 0).
    
    Known Solution
    dV = 1.7225 + 1.3297 = 3.0522 km/s
    '''
    
    # Instantiate the problem
    orb1 = {'a':7018.,'e':0.022799,'w':0.,'i':0.,'om':0.}
    orb2 = {'a':22378.,'e':0.,'w':0.,'i':0.,'om':0.}
    
    # Central body
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    
    # Known solution
    dV_soln = 3.0522
    
    return orb1,orb2,mu,dV_soln

def init_test_2():
    '''
    Example 2: LEO to GEO with inclination change
    
    From Example 6.11 of Curtis (2005)
    Initial orbit: 6678 km circular orbit inclined 28deg
    Final orbit: 42164 km circular orbit
    
    Solution (Note: Not optimal)
    dV = 3.8926 + 1.4877 = 5.3803 km/s 
    
    Equivalent coplanar problem (Hohmann transfer)
    dV = 3.8926
    
    '''
    
    # Initial and Final orbits
    orb1 = {'a':6678,'e':0,'i':np.deg2rad(0.01),'om':0.,'w':0.}
    orb2 = {'a':42164,'e':0,'i':0.,'om':0.,'w':0.}
    
    # Central body
    mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    # mu = 398616.683083029
    
    # Known solution
    dV_soln = 3.8926 
    # *** This is the equivalent coplanar problem.
    # Need to recalculate to include plane change.
    
    return orb1,orb2,mu,dV_soln

def init_test_3():
    '''
    Exampe 3: Non-coplanar Circular orbits
    
    Problem 3 in "Particle swarm optimization applied to impulsive orbital transfers"
    Pontani & Conway (2012)
    
    Knwon Solution:
    dV = 2.431770 + 1.509130 = 3.940899
    
    * This test case already has its orbits normalized to the final orbit.
    '''
    
    # Initial & Final orbits
    orb1 = {'a':6671.53,'e':0.,'w':0.,'om':0.,'i':np.deg2rad(10.)} # i=10
    orb2 = {'a':42163.95,'e':0.,'w':0.,'om':0.,'i':np.deg2rad(0.)} # i=0
    # Central body
    # mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    mu = 398616.683083029
    
    # Knwon solution
    dV_soln = 3.940899 
    
    return orb1,orb2,mu,dV_soln

def init_test_4():
    '''
    Exampe 4: Non-coplanar Circular to Elliptical orbit
    
    Problem 4 in "Particle swarm optimization applied to impulsive orbital transfers"
    Pontani & Conway (2012)
    Solution:
    dV = 1.392969 km/s
    alpha1 = TA1 = 163.71 deg = 2.8572 rad -> th1 = 1.2864 rad
    alpha2 = TA2 = 157.54 deg = 2.749 rad  -> th2 = 3.27319 rad
    '''
    
    # Initial & Final orbits
    orb1 = {'a':9645.833,'e':0.2,'i':np.deg2rad(5.),'om':0.,'w':np.deg2rad(270.)}
    orb2 = {'a':11575.,'e':0.2,'i':0.,'om':0.,'w':np.deg2rad(30.)}
    # Transformed parameters (wrt initial orbit) (table 7)
    # orb1 = {'a':9645.833,'e':0.2,'i':0.,'om':0.,'w':0.}
    # orb2 = {'a':11575.00,'e':0.2,'i':np.deg2rad(5.),'om':np.deg2rad(270.),'w':np.deg2rad(210)}

    # Central body
    # mu = 398600 # Gravitational parameter of Earth (km^3/s^2)
    mu = 398616.683083029
    
    # Known solution
    dV_soln = 1.392969
    
    return orb1,orb2,mu,dV_soln

#%% Test Objective Funtions
    
def test_mccue_function_eval():
    '''
    Test a single evaluation of the objective function for the McCue formulation.
    
    Uses test scenario 3, with a geometry from th1=0. to th2=60 deg.
    Solution generated from equivalent MATLAB program by David Eagle.

    '''
    
    # Load test case 3
    orb1,orb2,mu,dV_known = init_test_3()
    
    # Evaluate function at a single position (th1=0, th2=60 deg)
    # Using the Eagle MATLAB program, this evaluates to
    # dV = 10.5759 km/s
    x = [np.deg2rad(0.),np.deg2rad(60.)]
    
    # Start timer
    t0 = time.time()
    dV = f_mccue(x,
                  orb1['a'],orb1['e'],orb1['i'],orb1['w'],orb1['om'],
                  orb2['a'],orb2['e'],orb2['i'],orb2['w'],orb2['om'],
                  mu)
    # End timer and print
    print('Runtime {} s'.format(time.time()-t0))
    
    assert np.around(dV,decimals=4) == 10.5759
    print('Successful.')
    
    return
    

#%% Test solvers

def test_mccue_solvers(scenario=1):
    '''
    Solve the Orbit-to-Orbit scenario using the McCue formulation, with a 
    numer of different solvers. Print out the solution, runtime and convergence
    status of each solver.
    
    '''
    
    # Initialize the scenario
    if scenario==1:
        orb1,orb2,mu,dV_known = init_test_1()
    elif scenario==2:
        orb1,orb2,mu,dV_known = init_test_2()
    elif scenario==3:
        orb1,orb2,mu,dV_known = init_test_3()
    elif scenario==4:
        orb1,orb2,mu,dV_known = init_test_4()
    
    df = pd.DataFrame(columns=['method','x_sol','dV_sol','dV_known','converged','time'])
    
    
    # Solve the problem
    t0 = time.time()
    # solver='SHGO'
    # solver='Brute'
    solver='Grid'
    result = solve_OrbitToOrbit_mccue(orb1,orb2,mu,solver=solver)
    print('Runtime {} s'.format(time.time()-t0))
    
    # Check against the known solution
    print('           Known  | Achieved')
    print('Solution: {} | {}'.format(dV_known,result.fun))

    try:
        assert np.around(result.fun,decimals=4) == np.around(dV_known,decimals=4)
    except:
        assert np.around(result.fun,decimals=3) == np.around(dV_known,decimals=3)
    print('Converged successfully')
    
    print('')
    
    
    # ----
    # Run through different solvers
    methods = ['SHGO','Brute','Grid']
    rows_list = []
    for i,method in enumerate(methods):
        # Solve the problem
        t0 = time.time() # Start timer
        result = solve_OrbitToOrbit_mccue(orb1,orb2,mu,solver=method)
        runtime = time.time()-t0 # Computation time (s)
        # Check convergence
        if np.around(result.fun,decimals=4) == np.around(dV_known,decimals=4):
            converged = '4dp'
        elif np.around(result.fun,decimals=3) == np.around(dV_known,decimals=3):
            converged = '3dp'
        elif np.around(result.fun,decimals=2) == np.around(dV_known,decimals=2):
            converged = '2dp'
        elif np.around(result.fun,decimals=1) == np.around(dV_known,decimals=1):
            converged = '1dp'
        else:
            converged = 'False'
        
        # Append results to dataframe 
        dict1 = {'method':method,'x_sol':result.x,'f_sol':result.fun,'f_known':dV_known,
                 'converged':converged,'time':runtime}
        rows_list.append(dict1)
    
    # Form final dataframe
    df = pd.DataFrame(rows_list)
    
    # Print
    print(df)    
    
    # Decode solution
    atx,etx,itx,omtx,wtx,TAtx1,TAtx2 = decode_solution_mccue(orb1,orb2,mu,result)
    print('Transfer orbit')
    print('a = {} km'.format(atx))
    print('e = {}'.format(etx))
    print('i = {} deg'.format(np.rad2deg(itx)))
    print('om = {} deg'.format(np.rad2deg(omtx)))
    print('w = {} deg'.format(np.rad2deg(wtx)))
    print('TA1 = {} deg'.format(np.rad2deg(TAtx1)))
    print('TA2 = {} deg'.format(np.rad2deg(TAtx2)))
    
    
    return

#%% Test contour plots

def test_mccue_contour_plot(scenario=1,method='vectorized'):
    '''
    Generate a contour plot for the Orbit-to-Orbit scenario using the McCue 
    formulation, with the scipy SHGO solver.
    
    '''
    
    # Initialize the scenario
    if scenario==1:
        orb1,orb2,mu,dV_known = init_test_1()
    elif scenario==2:
        orb1,orb2,mu,dV_known = init_test_2()
    elif scenario==3:
        orb1,orb2,mu,dV_known = init_test_3()
    elif scenario==4:
        orb1,orb2,mu,dV_known = init_test_4()
    
    # Generate plot
    plot_porkchop_mccue(orb1,orb2,mu,method=method)
    
    return

#%% Vectorized implementation

def test_vec_mccue():
    
    # STATUS: Working, but some singularities in solutions.
    
    # Load test case 3
    orb1,orb2,mu,dV_known = init_test_3()
    
    # Extract elements
    ai = orb1['a']
    ei = orb1['e']
    inci = orb1['i']
    omi = orb1['om']
    wi = orb1['w']
    af = orb2['a']
    ef = orb2['e']
    incf = orb2['i']
    omf = orb2['om']
    wf = orb2['w']
    
    # Transform elements to re-reference to final orbit
    a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2 = transform_orbits(ai,ei,wi,omi,inci,af,ef,wf,omf,incf,reference='f')
    
    
    
    # Form a grid of th1,th2 points to evaluate during a single function call
    # (Same points as those generated for the contour plot)
    x0_ = np.linspace(0.,2*np.pi,100)
    x1_ = np.linspace(0.,2*np.pi,100)
    
    
    x0grid, x1grid = np.meshgrid(x0_, x1_, indexing='ij')
    x0 = x0grid.flatten()
    x1 = x1grid.flatten()
    x = np.column_stack((x0,x1))
    
    # Form elements into vectors matching this length
    a1 = a1*np.ones(len(x0))
    e1 = e1*np.ones(len(x0))
    inc1 = inc1*np.ones(len(x0))
    w1 = w1*np.ones(len(x0))
    om1 = om1*np.ones(len(x0))
    a2 = a2*np.ones(len(x0))
    e2 = e2*np.ones(len(x0))
    inc2 = inc2*np.ones(len(x0))
    w2 = w2*np.ones(len(x0))
    om2 = om2*np.ones(len(x0))
    mu = mu*np.ones(len(x0))
    
    # Evaluate the delta-V at these points
    t0 = time.time()
    dV = f_mccue_vec(x,a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu)
    print('Runtime {} s'.format(time.time()-t0))
    
    # Remove large delta-Vs
    dV[dV > 10.] = np.nan
    
    # Create porkchop plot
    # Contour plot
    fig1, ax = plt.subplots()
    CS = ax.contourf(x0grid, x1grid, dV.reshape(x0grid.shape), 
                      cmap='jet',
                      )
    
    
    cbar = plt.colorbar(CS)
    cbar.ax.set_title('dV (km/s)',fontsize=8)
    
    
    ax.set_xlabel('th1 (rad)')
    ax.set_ylabel('th2 (rad)')
    
    
    return

#%% Problem Class

def test_OrbitToOrbitProblem(scenario=1):
    
    # Initialize the scenario
    if scenario==1:
        orb1,orb2,mu,dV_known = init_test_1()
    elif scenario==2:
        orb1,orb2,mu,dV_known = init_test_2()
    elif scenario==3:
        orb1,orb2,mu,dV_known = init_test_3()
    elif scenario==4:
        orb1,orb2,mu,dV_known = init_test_4()
    
    # Initialize problem
    prob = OrbitToOrbitProblem(orb1, orb2, mu)
    
    # Solve problem
    prob.solve(decode=True)
    
    # Plot porkchop
    prob.plot_porkchop()
    
    
    return