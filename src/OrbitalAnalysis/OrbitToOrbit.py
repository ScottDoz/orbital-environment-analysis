# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:10:46 2021

@author: scott

Orbit-to-Orbit Module
---------------------

Contains methods for computing/optimizing orbit-to-orbit transfers.

This code is copied from the sr_tools project - under development.

"""

# Standard imports
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

# import pykep as pk
# import pygmo as pg

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Optimization
from scipy import optimize
from scipy import interpolate
from scipy.signal import find_peaks

# Module imports
from OrbitalAnalysis.Functions import sv_from_coe, coe_from_sv
from OrbitalAnalysis.optimizers import chandrupatla

import pdb

# Supress numpy warnings
np.seterr(invalid='ignore')

#%% Problem Class

class OrbitToOrbitProblem:
    '''
    Class defining an Orbit-to-Orbit transfer problem.
    This class simplifies the process of formulating and solving the problem.
    
    Workflow:
    prob = OrbitToOrbitProblem(orb1,orb2,mu) # Instantiate
    prob.solve(solver='Grid',decode=True) # Solve the problem
    prob.plot_porkchop() # Plot the porkchop plot
    
    '''
    
    def __init__(self,orb1,orb2,mu):
        '''
        Initialize the problem with the orbital elements of the two orbits.

        Parameters
        ----------
        orb1, orb2 : dict
            Dictionaries containing the orbital elements a,e,i,om,w,tp of the 
            initial and final orbits. Angles should be provided in radians.
        mu : float
            Gravitational parameter of central body (km^3/s^2)
        '''
        
        # Assign basic attributes
        self.mu = mu
        self.orb1 = orb1
        self.orb2 = orb2
        self.result = None
        
        # Extract initial orbit
        self.ai = orb1['a']
        self.ei = orb1['e']
        self.inci = orb1['i']
        self.omi = orb1['om']
        self.wi = orb1['w']
        
        # Extract final orbit
        self.af = orb2['a']
        self.ef = orb2['e']
        self.incf = orb2['i']
        self.omf = orb2['om']
        self.wf = orb2['w']
        
        # Transform elements to re-reference to final orbit
        a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2,B = transform_orbits(self.ai,self.ei,self.wi,self.omi,self.inci,
                                                                 self.af,self.ef,self.wf,self.omf,self.incf,
                                                                 reference='f',return_basis=True)
        
        # Append transformed elements and basis
        self._a1 = a1
        self._e1 = e1
        self._om1 = om1
        self._w1 = w1
        self._inc1 = inc1
        self._a2 = a2
        self._e2 = e2
        self._om2 = om2
        self._w2 = w2
        self._inc2 = inc2
        self._B = B
        
        return
        
    
    # Solver ------------------------------------------------------------------
    def solve(self,solver='Grid',decode=False):
        '''
        Solve the Orbit-to-Orbit problem. 
        The solution is appended as the 'result' class attribute.
        
        Optionally, decode the solution to extract the orbital elements of the 
        transfer, and state and impulse vectors at the termainal points.

        Parameters
        ----------
        solver : TYPE, optional
            DESCRIPTION. The default is 'Grid'.
        decode : Bool, optional
            Decode the solution to compute details of the transfer orbit. 
            The default is False.

        '''
        
        # Retrieve pre-computed transformed elements
        a1,e1,inc1,w1,om1 = self._a1, self._e1, self._inc1, self._w1, self._om1
        a2,e2,inc2,w2,om2 = self._a2, self._e2, self._inc2, self._w2, self._om2
        mu = self.mu
        
        # Call solver function
        if solver == 'SHGO':
            # SHGO - Simplicial Homology Global Optimization
            result = optimize.shgo(f_mccue, bounds=[(0.,2*np.pi), (0., 2*np.pi)],
                                   args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                                   n=10,iters=2,sampling_method='sobol')
    
        elif solver == 'Brute':
            # Brute force solver
            (x0,fval,grid,Jout) = optimize.brute(f_mccue, ranges=[(0.,2*np.pi), (0., 2*np.pi)],
                                    args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                                    Ns=5,full_output=True,
                                    )
            
            # Format result into dictionary
            result_dict = {'x':x0,'fun':fval,'x_grid':grid,'f_grid':Jout}
            # Convert to simple namespace to allow dot access to variables
            result = SimpleNamespace(**result_dict)
            
        elif solver == 'Grid':
            # Apply a grid-search method. 
            # Similar to scipy.optimize.brute, but we are specifiying the grid to use.
            # This alows use of the vectorized function to evaluate.
            # (Problems in passing vectorized function to Brute)
            
            # Generate grid of parameter space x=(th1,th2)
            Ns = 20 # Number of grid points in each direction
            x0_ = np.linspace(0.,2*np.pi,Ns)
            x1_ = np.linspace(0.,2*np.pi,Ns)
            x0grid, x1grid = np.meshgrid(x0_, x1_, indexing='ij')
            x0 = x0grid.flatten()
            x1 = x1grid.flatten()
            x = np.column_stack((x0,x1))
            # Evaluate at grid points
            dV = f_mccue_vec(x,
                             a1*np.ones(Ns**2),e1*np.ones(Ns**2),inc1*np.ones(Ns**2),
                             w1*np.ones(Ns**2),om1*np.ones(Ns**2),a2*np.ones(Ns**2),
                             e2*np.ones(Ns**2),inc2*np.ones(Ns**2),w2*np.ones(Ns**2),
                             om2*np.ones(Ns**2),mu*np.ones(Ns**2))
            
            # Find global solution to use as initial guess
            ind = np.nanargmin(dV)
            x0_guess = x[ind,:]
            
            # Finish with an optimization method ------------------------------
            
            # Using fmin (Downhill simplex algorithm)
            res = optimize.fmin(f_mccue, x0_guess, 
                          args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                          full_output=True)
            # Extract results res = (x0,fval,n_iter,funcals,warnflag,allvecs)
            # Format result into dictionary
            result_dict = {'x':res[0],'fun':res[1],'iter':x[2],'funcals':x[3],'warnflag':x[4]}
            # Convert to simple namespace to allow dot access to variables
            result = SimpleNamespace(**result_dict)
            
            # # Using minimize (slower than fmin)
            # # *** TODO: Provide jacobian for faster evaluation.
            # result = optimize.minimize(f_mccue, x0_guess, 
            #               args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
            #               )
            
        
        # elif solver =='BH':
            # Basin Hopping
        
        
        # Append result
        self.result = result
        
        # Decode solution (if requested)
        if decode:
            self.decode_solution()
        
        return
    
    def decode_solution(self):
        '''
        Decode the solution to extract the orbital elements of the transfer, 
        and state and impulse vectors at the termainal points.

        These details are added to the 'result' attribute.

        '''
        
        # Retrieve pre-computed transformed elements
        a1,e1,inc1,w1,om1 = self._a1, self._e1, self._inc1, self._w1, self._om1
        a2,e2,inc2,w2,om2 = self._a2, self._e2, self._inc2, self._w2, self._om2
        B = self._B
        mu = self.mu
        
        # Evaluate the function at the solution
        x = self.result.x # solution
        f,r1_ref,r2_ref,vtx1_ref,vtx2_ref,I1_ref,I2_ref,tx_type,pt = f_mccue(x,
                                                  a1,e1,inc1,w1,om1, # Initial orbit
                                                  a2,e2,inc2,w2,om2,  # Final orbit 
                                                  mu,full_output=True)
        
        
        # Transform to inertial frame
        
        # Compute transformation matrices
        A_inert_to_ref = np.column_stack((B[0],B[1],B[2])) # Inertial to reference
        A_ref_to_inert = np.linalg.inv(A_inert_to_ref) # # Reference to inertial (Inverse of A)
        # X_ref = A_inert_to_ref*X_inertial
        # X_inert = A_ref_to_inert*X_ref
        
        # Transform state vectors
        
        # Transform the position and velocity vetors to the original orbit frame
        # (need to convert each vector to column vectors)
        # First, stack the points together to perform the transform in one call.
        points_ref = np.row_stack((r1_ref,r2_ref,vtx1_ref,vtx2_ref,I1_ref,I2_ref))
        # Apply the transform 
        # (use transform to reshape the points into column vectors)
        points_inert = np.dot(A_ref_to_inert, points_ref.T).T
        # Extract vectors in inertial frame
        r1 = points_inert[0,:]
        r2 = points_inert[1,:]
        vtx1 = points_inert[2,:]
        vtx2 = points_inert[3,:]
        I1 = points_inert[4,:]
        I2 = points_inert[5,:]
        
        # Convert state vectors to elements
        atx,etx,itx,omtx,wtx,TAtx1 = coe_from_sv(r1,vtx1,mu=mu,units='km') # 1st impulse
        atx2,etx2,itx2,omtx2,wtx2,TAtx2 = coe_from_sv(r2,vtx2,mu=mu,units='km') # 2nd impulse
        
        # Append solution
        result = self.result # Get original solution
        result.txorb = {'a':atx, 'e':etx, 'w':wtx, 'om':omtx, 'TA1':TAtx1, 'TA2':TAtx2}
        result.r1 = r1
        result.r2 = r2
        result.vtx1 = vtx1
        result.vtx2 = vtx2
        result.I1 = I1
        result.I2 = I2
        
        # Override result attribute
        self.result = result
        
        return
    
    # Plotting ----------------------------------------------------------------
    def plot_porkchop(self,Nx=100,Ny=100,method='vectorized'):
        
        # Retrieve pre-computed transformed elements
        a1,e1,inc1,w1,om1 = self._a1, self._e1, self._inc1, self._w1, self._om1
        a2,e2,inc2,w2,om2 = self._a2, self._e2, self._inc2, self._w2, self._om2
        mu = self.mu
        
        # Generate grid of parameter space x=(th1,th2)
        x0_ = np.linspace(0.,2*np.pi,Nx)
        x1_ = np.linspace(0.,2*np.pi,Ny)
        x0grid, x1grid = np.meshgrid(x0_, x1_, indexing='ij')
        x0 = x0grid.flatten()
        x1 = x1grid.flatten()
        
        if method == 'loop':
            # Loop through grid points (x0,x1) and evaluate
            dV = np.zeros(len(x0))*np.nan # Initialize output    
            for n in tqdm(range(len(x0))):
                # Extract parameter
                x = [x0[n],x1[n]]
                # Compute objective
                dV[n] = f_mccue(x,a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu)
        elif method == 'vectorized':
            # Use vectorized function to evaluate in single function call
            
            # Form elements into vectors matching length of grid
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
            
            # Evaluate the delta-V at these points
            x = np.column_stack((x0,x1))
            dV = f_mccue_vec(x,a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu*np.ones(len(x0)))  
        
        # Extract result
        if self.result is None:
            self.solve()
            
        result = self.result
        
        # Create Contour plot
        fig1, ax = plt.subplots()
        CS = ax.contourf(x0grid, x1grid, dV.reshape(x0grid.shape), 
                          cmap='jet',
                          )
        
        cbar = plt.colorbar(CS)
        cbar.ax.set_title('dV (km/s)',fontsize=8)
        
        # Plot global solution
        ax.plot(result.x[0],result.x[1],'or',markersize=5)
        
        # # TODO: Plot local solutions
        # fl = result.funl # Values
        # xl = result.xl # Positions
        # if xl.shape == (2,):
        #     # Only a single solution
        #     pass
        # elif len(fl)>1:
        #     # Multiple solutions. Loop through each.
        #     for i in range(len(fl)):
        #         ax.plot(xl[i,0],xl[i,1],'og',markersize=10)
        
        ax.set_xlabel('th1 (rad)')
        ax.set_ylabel('th2 (rad)')
        
        return

    # TODO: Plot transfer orbit.
    # TODO: Plotting in Plotly

#%% Orbital Element Transformations

def transform_orbits(ai,ei,wi,omi,inci,af,ef,wf,omf,incf,reference='i',return_basis=False):
    '''
    Transform the orbital elements of an initial and final orbits to a new
    reference system defined by the node crossing of the two orbits, and the
    orbit normal of either the initial or final orbits. This transformation
    simplifies calculations of the orbit-to-orbit problem.

    Parameters
    ----------
    ai,ei,wi,omi,inci : Float
        Orbital elements of initial orbit
    af,ef,wf,omf,incf : Float
        Orbital elements of final orbit
    reference : str, optional
        Flag defining the reference orbit (initial or final). The default is 'i'.

    Returns
    -------
    a1,e1,w1,om1,inc1 : Float
        Orbital elements of initial orbit wrt reference orbit.
    a2,e2,w2,om2,inc2 : Float
        Orbital elements of final orbit wrt reference orbit.

    '''
    
    # Step 1: Geometry of initial & final orbits ------------------------------
    # Use periapsis state vectors of initial and final orbits to find the orbit
    # normals, and the line of nodes.
    
    # Initial orbit
    sv1 = sv_from_coe(ai,ei,inci,omi,wi,0.,mu=1.,units='km') 
    r1_vec = sv1[:3] # Periapsis direction of orbit 1
    v1_vec = sv1[3:]
    h1 = np.cross(r1_vec,v1_vec) # Normal vector
    h1 = h1/np.linalg.norm(h1) # Unit normal vector
    del sv1
    
    # Final orbit
    sv2 = sv_from_coe(af,ef,incf,omf,wf,0.,mu=1.,units='km')
    r2_vec = sv2[:3]
    v2_vec = sv2[3:]
    h2 = np.cross(r2_vec,v2_vec) # Normal vector
    h2 = h2/np.linalg.norm(h2) # Unit normal vector
    del sv2
    
    # Line of nodes (from cross product of orbit normals)
    if (inci==0.) & (incf==0.):
        # Coplanar problem.
        if reference=='i':
            # Reference to initial orbit 
            N = r1_vec # Direction of periapsis
        elif reference=='f':
            # Reference to final orbit.
            N = r2_vec # Direction of periapsis
        else:
            raise ValueError('Unrecognized reference. Use i or f.')
        
    else:
        # Non-coplanar problem
        if reference=='i':
            # Reference to initial orbit. Take h1xh2
            N = np.cross(h1,h2) # Direction of Node crossing
        elif reference=='f':
            # Reference to final orbit. Take h2xh1
            N = np.cross(h2,h1) # Direction of Node crossing
        else:
            raise ValueError('Unrecognized reference. Use i or f.')
    
    # Normalize the line of nodes
    if np.linalg.norm(N) != 0:
        N = N/np.linalg.norm(N) # Normalize
                         
    # Find the angle between initial and final orbit
    cos_phi = np.dot(h1,h2)
    phi = np.arccos(np.clip(cos_phi, -1, 1))
    phi = np.mod(phi, 2*np.pi) # Wrap to 2pi
    # This is the relative inclination of the final orbit wrt initial orbit.
    
    # Step 2: Transform orbits wrt line of nodes ------------------------------
    # We use this line of nodes as a reference direction for both orbits.
    # This sets the right ascension of ascending nodes of both orbits to zero.
    # We need to recompute the argument of periapsis of both orbits wrt the 
    # line of nodes. Call the new values w1, w2 to distinguish them from the 
    # original wi,wf.
    
    # w1: Signed angle from N to r1
    cross = np.cross(N,r1_vec)
    w1 = np.arctan2(  np.dot(cross,h1) ,  np.dot(N,r1_vec) )
    w1 = np.mod(w1, 2*np.pi) # Wrap to 2pi
    
    # w2: Signed angle from N to r2
    cross = np.cross(N,r2_vec)
    w2 = np.arctan2(  np.dot(cross,h2) ,  np.dot(N,r2_vec) )
    w2 = np.mod(w2, 2*np.pi) # Wrap to 2pi
    
    
    
    # Note: angle from Va to Vb = atan2((Va x Vb) . Vn, Va . Vb)
    
    # Step 3: Prepare orbital elements ----------------------------------------
    a1,e1 = ai,ei # a,e unchanged
    a2,e2 = af,ef # a,e unchanged
    om1,om2 = 0.,0. # Since referenced from common line of nodes
    
    if reference=='i':
        # Reference to initial orbit.
        inc1 = 0. # No inclination for reference orbit
        inc2 = phi
    else:
        # Reference to final orbit.
        inc1 = phi 
        inc2 = 0. # No inclination for reference orbit
    
    
    # Step 4: Compute the basis for the new coordiante system (if requested)
    # Basis: B ={b1,b2,b3}
    # b1 = N direction of node crossings.
    # b3 = w1 or w2 vector points in direction of reference orbital plane
    # b2 = b3 x b1 completes the right-hand coordinate system
    
    if return_basis==True:
    
        # x-direction
        b1 = N
        # z-direction
        if reference=='i':
            # Reference to initial orbit.
            b3 = h1 # Use normal of initial orbit
        else:
            # Reference to final orbit
            b3 = h2 # Use normal of final orbit
        # Compute y-direction
        b2 = np.cross(b3,b1)
        
        # Check they match
        assert np.all(np.cross(b1,b2) == b3)
        
        # Form basis as tuple
        B = (b1,b2,b3)
        
        # Return transformed elements and basis
        return a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2,B
    
    else:
        # Just return transformed elements
        return a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2


#%% ###########################################################################
#                        McCue Formulation
# 
# Two-impulse Orbit-to-Orbit computation using the formulation described in
# McCue & Bender (1961) 
# "Numerical Investigation of Minimum Impulsive Orbita Transfer"
# (plus related papers)
###############################################################################

# Main objective function -----------------------------------------------------
def f_mccue(x,
            a1,e1,inc1,w1,om1, # Initial orbit
            a2,e2,inc2,w2,om2,  # Final orbit 
            mu,
            full_output=False):
    '''
    Total delta-V of two-impulse transfer between two orbits.
    
    Following formulation in McCue & Bender (1965) 
    "Numerical Investigation of Minimum Impulse Orbital Transfer"

    Parameters
    ----------
    x = (th1,th2,pt) 
        Coordinates
    orb1 : Dict
        Orbital elements of first orbit (normalized to second orbit).
    orb2 : Dict
        Orbital elements of second orbit.
    mu : Float
        Gravitational parameter.

    Returns
    -------
    f : TYPE
        Delta-V objective

    '''
    
    # Extract coordinates
    th1, th2 = x[0],x[1]
    
    # Check normalization
    if inc2 !=0:
        raise ValueError('Orbits must be normalized to plane of final orbit.')
    
    # Semi-latus rectum
    p1 = abs(a1*(1. - e1**2)) # Semi-latus rectum
    p2 = abs(a2*(1. - e2**2)) # Semi-latus rectum
    
    # Step 1: Transfer Geometry -----------------------------------------------
    # Compute unit vectors of the terminal points and transfer angles.
    
    # Unit vectors
    U1 = np.array([np.cos(th1),np.sin(th1)*np.cos(inc1),np.sin(th1)*np.sin(inc1)])
    U2 = np.array([np.cos(th2),np.sin(th2),0.])
    
    # Position vectors
    r1m = p1/(1+e1*np.cos(th1-w1)) # Magnitude |r1|
    r2m = p2/(1+e2*np.cos(th2-w2)) # Magnitude |r2|
    r1 = r1m*U1
    r2 = r2m*U2
    
    # Unit normal vectors
    W1 = np.array([0.,-np.sin(inc1),np.cos(inc1)])
    W2 = np.array([0.,0.,1.])
    Wt = np.cross(U1,U2)
    Wt_norm = np.linalg.norm(Wt) # Norm of Wt
    if Wt_norm != 0:
        Wt = Wt/Wt_norm # Normalize
    else:
        # U1 and U2 are parallel. May lead to invalid solutions.
        pass
    
    # Eccentricity vectors
    ecc1 = e1*np.array([np.cos(w1),np.sin(w1)*np.cos(inc1),np.sin(w1)*np.sin(inc1)])
    ecc2 = e2*np.array([np.cos(w2),np.sin(w2)*np.cos(inc2),np.sin(w2)*np.sin(inc2)])
    
    # Transfer angle (0 < delta < 180 deg)
    delta = np.arccos(np.dot(U1,U2))
    # Check for parallel unit vectors
    if np.array_equal(U1,U2):
        delta = 0.
    
    # TODO:
    # Alternative formulation for U1.U2 = 0 from ref 6.
    if delta==0.:
        # For now, return large number
        return 1.0E99
    
    # Velocities
    V1 = np.sqrt(mu/p1)*np.cross(W1,(ecc1+U1))
    V2 = np.sqrt(mu/p2)*np.cross(W2,(ecc2+U2))
    
    # Find bounds on valid p values for transfer orbit
    # (eq 22,23 of McCue, 1963)
    r1_dot_r2 = np.dot(r1,r2)
    if abs(np.sin(delta)) < 1E-12:
        # 0/180 deg transfer.
        # Either pmin or pmax will be nan
        pmin = 0.1 # Set at small value
        pmax = a1*a2 # Set at large value
    else:
        pmin = (r1m*r2m - r1_dot_r2)/(r1m+r2m + np.sqrt(2*(r1m*r2m + r1_dot_r2)) )
        pmax = (r1m*r2m - r1_dot_r2)/(r1m+r2m - np.sqrt(2*(r1m*r2m + r1_dot_r2)) )
    
    
    
    # Evaluate dI/dp at these end points
    f1 = func_dVdp(pmin, r1,r2,U1,U2,V1,V2,delta,mu)
    f2 = func_dVdp(pmax, r1,r2,U1,U2,V1,V2,delta,mu)
    
    # Check conditions for root (need f(pmin) and f(pmax) to change signs)
    if f1*f2 >= 0.:
        # No root between pmin,pmax. Return a large value
        return 1.0E99
    
    # Find optimal pt from root of dVdp function
    # (dV/dp = 0 at optimal location)
    pt = optimize.brentq(func_dVdp,pmin,pmax,args=(r1,r2,U1,U2,V1,V2,delta,mu), rtol=1.0E-08)
    
    # Evaluate delta-V for this optimal value
    if full_output==True:
        
        # Compute details of the delta-V
        I,r1,r2,vtx1,vtx2,I1,I2,tx_type = func_dV(pt, r1,r2,U1,U2,V1,V2,delta,mu,full_output=True)
        
        # Return full details
        return I,r1,r2,vtx1,vtx2,I1,I2,tx_type,pt
        
        
    else:
        # Only return the delta-V value
        f = func_dV(pt, r1,r2,U1,U2,V1,V2,delta,mu)
    
    
    # Full details to return
    # r1,r2,pt
    
    return f

# Ojective functions for sub-problem at fixed transfer points -----------------
def func_dV(p, r1,r2,U1,U2,V1,V2,delta,mu,full_output=False):
    '''
    Delta-V for a particular parameter of a fixed transfer geometry (r1,r2).

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    '''
    
    # Compute v and z parameters eq 18 & 19.
    r1_cross_r2m = np.linalg.norm(np.cross(r1,r2))
    v = np.sqrt(mu*p)*(r2-r1)/r1_cross_r2m
    z = np.sqrt(mu/p)*np.tan(delta/2)
    
    # Short Transfer (upper sign)
    I1S = (v + z*U1) - V1 # 1st impulse vector
    I2S = V2 - (v - z*U2) # 2nd inpulse vector
    I1magS = np.linalg.norm(I1S) # 1st impulse magnitude
    I2magS = np.linalg.norm(I2S) # 2nd impulse magnitude
    ImagS = I1magS + I2magS # Total impulse magnitude
    
    # Long Transfer (lower sign)
    I1L = -(v + z*U1) - V1 # 1st impulse vector
    I2L = V2 + (v - z*U2) # 2nd inpulse vector
    I1magL = np.linalg.norm(I1L) # 1st impulse magnitude
    I2magL = np.linalg.norm(I2L) # 2nd impulse magnitude
    ImagL = I1magL + I2magL # Total impulse magnitude
    
    # Determine lower delta-V
    if ImagS < ImagL:
        # Short transfer is more efficient
        I1 = I1S
        I2 = I2S
        I1mag = I1magS
        I2mag = I2magS
        I = ImagS
        
    else:
        # Long transfer is more efficient
        I1 = I1L
        I2 = I2L
        I1mag = I1magL
        I2mag = I2magL
        I = ImagL
    
    # Full output
    if full_output==True:
        # Return full details of solution
        if ImagS < ImagL:
            tx_type = 'Short'
            Vtx1 = (v + z*U1) # Velocity of tx orbit at departure
            Vtx2 = (v - z*U2) # Veclocity of tx orbit at arrival
        else:
            tx_type = 'Long'
            Vtx1 = -(v + z*U1) # Velocity of tx orbit at departure
            Vtx2 = -(v - z*U2) # Veclocity of tx orbit at arrival
        
        # Return full results
        return I,r1,r2,Vtx1,Vtx2,I1,I2,tx_type
    else:
        # Return just magnitude 
        return I

def func_dVdp(p, r1,r2,U1,U2,V1,V2,delta,mu):
    '''
    Partial derivative of delta-V wrt parameter dV/dp.
    Needed to find optimal p for given transfer geometry.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    dIdp : TYPE
        DESCRIPTION.

    '''
    
    
    # Compute v and z parameters eq 18 & 19.
    dr = r2-r1
    r1_cross_r2m = np.linalg.norm(np.cross(r1,r2))
    v = np.sqrt(mu*p)*(r2-r1)/r1_cross_r2m
    z = np.sqrt(mu/p)*np.tan(delta/2)
    
    # Short Transfer (upper sign)
    I1S = (v + z*U1) - V1 # 1st impulse vector
    I2S = V2 - (v - z*U2) # 2nd inpulse vector
    I1magS = np.linalg.norm(I1S) # 1st impulse magnitude
    I2magS = np.linalg.norm(I2S) # 2nd impulse magnitude
    ImagS = I1magS + I2magS # Total impulse magnitude
    dIdpS = ( np.dot(I1S,(v-z*U1))/I1magS - np.dot(I2S,(v+z*U2))/I2magS  )/(2.*p) 
     
    # Long Transfer (lower sign)
    I1L = -(v + z*U1) - V1 # 1st impulse vector
    I2L = V2 + (v - z*U2) # 2nd inpulse vector
    I1magL = np.linalg.norm(I1L) # 1st impulse magnitude
    I2magL = np.linalg.norm(I2L) # 2nd impulse magnitude
    ImagL = I1magL + I2magL # Total impulse magnitude
    dIdpL = -( np.dot(I1L,(v-z*U1))/I1magL - np.dot(I2L,(v+z*U2))/I2magL  )/(2.*p) 
    
    # Determine lower delta-V
    if ImagS < ImagL:
        # Short transfer is more efficient
        I1 = I1S
        I2 = I2S
        I1mag = I1magS
        I2mag = I2magS
        I = ImagS
        dIdp = dIdpS
        
        
    else:
        # Long transfer is more efficient
        I1 = I1L
        I2 = I2L
        I1mag = I1magL
        I2mag = I2magL
        I = ImagL
        dIdp = dIdpL
    
    return dIdp


#%% Vectorized implementation

def f_mccue_vec(x,
            a1,e1,inc1,w1,om1, # Initial orbit
            a2,e2,inc2,w2,om2,  # Final orbit 
            mu):
    '''
    Total delta-V of two-impulse transfer between two orbits.
    
    Following formulation in McCue & Bender (1965) 
    "Numerical Investigation of Minimum Impulse Orbital Transfer"

    Parameters
    ----------
    x = (th1,th2,pt) 
        Coordinates
    orb1 : Dict
        Orbital elements of first orbit (normalized to second orbit).
    orb2 : Dict
        Orbital elements of second orbit.
    mu : Float
        Gravitational parameter.

    Returns
    -------
    f : TYPE
        Delta-V objective

    '''
    
    # Extract coordinates
    th1, th2 = x[:,0],x[:,1]
    
    # # Check normalization
    # if inc2 !=0:
    #     raise ValueError('Orbits must be normalized to plane of final orbit.')
    
    # Semi-latus rectum
    p1 = abs(a1*(1. - e1**2)) # Semi-latus rectum
    p2 = abs(a2*(1. - e2**2)) # Semi-latus rectum
    
    # Step 1: Transfer Geometry -----------------------------------------------
    # Compute unit vectors of the terminal points and transfer angles.
    
    # Unit vectors
    U1 = np.vstack(( np.cos(th1),np.sin(th1)*np.cos(inc1),np.sin(th1)*np.sin(inc1) )).T
    U2 = np.vstack(( np.cos(th2),np.sin(th2),np.zeros(len(th1)) )).T
    
    # Position vectors
    r1m = p1/(1+e1*np.cos(th1-w1)) # Magnitude |r1|
    r2m = p2/(1+e2*np.cos(th2-w2)) # Magnitude |r2|
    r1 = r1m[:, np.newaxis]*U1
    r2 = r2m[:, np.newaxis]*U2
    
    # Unit normal vectors
    W1 = np.vstack(( np.zeros(len(th1)),-np.sin(inc1),np.cos(inc1) )).T
    W2 = np.vstack(( np.zeros(len(th1)),np.zeros(len(th1)),np.ones(len(th1)) )).T
    Wt = np.cross(U1,U2)
    Wt_norm = np.linalg.norm(Wt, axis=-1) # Norm of Wt
    # Normalize
    prob_ind = Wt_norm == 0 # Indices where U1,U2 are parallel. Undefined transfer orbit.
    Wt[~prob_ind,:] = Wt[~prob_ind,:]/Wt_norm[~prob_ind][:, np.newaxis]
    
    # Eccentricity vectors
    ecc1 = e1[:, np.newaxis]*np.vstack(( np.cos(w1),np.sin(w1)*np.cos(inc1),np.sin(w1)*np.sin(inc1)  )).T
    ecc2 = e2[:, np.newaxis]*np.vstack(( np.cos(w2),np.sin(w2)*np.cos(inc2),np.sin(w2)*np.sin(inc2)  )).T
    
    # Transfer angle (0 < delta < 180 deg)
    cos_delta = np.einsum('ij,ij->i',U1,U2)
    delta = np.arccos(cos_delta)
    # # Check for parallel unit vectors
    # if np.array_equal(U1,U2):
    #     delta = 0.
    
    # # TODO:
    # # Alternative formulation for U1.U2 = 0 from ref 6.
    # if delta==0.:
    #     # For now, return large number
    #     return 1.0E99
    
    # Velocities
    V1 = np.sqrt(mu/p1)[:, np.newaxis]*np.cross(W1,(ecc1+U1))
    V2 = np.sqrt(mu/p2)[:, np.newaxis]*np.cross(W2,(ecc2+U2))
    
    # Find bounds on valid p values for transfer orbit
    # (eq 22,23 of McCue, 1963)
    pmin = np.zeros(len(th1))*np.nan # Initialize array
    pmax = np.zeros(len(th1))*np.nan # Initialize array
    # Compute pmin
    r1_dot_r2 = np.einsum('ij,ij->i',r1,r2) #np.dot(r1,r2)
    pmin = (r1m*r2m - r1_dot_r2)/(r1m+r2m + np.sqrt(2*(r1m*r2m + r1_dot_r2)) )
    pmax = (r1m*r2m - r1_dot_r2)/(r1m+r2m - np.sqrt(2*(r1m*r2m + r1_dot_r2)) )
    # Replace problem indices
    ind_prob = np.cos(delta) == 0. # Problem indices 0/180 deg transfer
    pmin[ind_prob] = 0.0001 # Set at small value
    pmax[ind_prob] = a1[ind_prob]*a2[ind_prob] # Set at large value
    # Compute midpoint
    p_mid = (pmin+pmax)/2 # Midpoint
    # f_mid = func_dVdp_vec(p_mid, r1,r2,U1,U2,V1,V2,delta,mu,pmin,pmax) # Function value at midpoint
    
    # Evaluate dI/dp at these end points
    f1 = func_dVdp_vec(pmin, r1,r2,U1,U2,V1,V2,delta,mu,pmin,pmax)
    f2 = func_dVdp_vec(pmax, r1,r2,U1,U2,V1,V2,delta,mu,pmin,pmax)
    
    # Check conditions for root (need f(pmin) and f(pmax) to change signs)
    ind_pass = f1*f2 < 0. # Values with root between pmin,pmax
    # The rest do not have a root - set delta-V as large value.
    
    # For the passable points, find the optimial p in the range pmin,pmax
    
    # Initial guesses from differnt methods -----------------------------------
    # init_guess = 'min'
    # init_guess = 'max' # pmax as initial guess - Fastest, but incorrect.
    # init_guess = '3quarter' # 3/4 point
    init_guess = 'mid' # Midpoint - BEST INITIAL GUESS
    # init_guess = 'linear_interp' # Secant 
    # init_guess = 'quadratic_interp' # Inverse quadratic - gives worse resutl than mid
    
    if init_guess == 'min':
        # Use pmin as initial guess
        x0 = pmin[ind_pass]
    elif init_guess == 'max':
        # Use pmax as initial guess
        x0 = pmax[ind_pass]
    elif init_guess == 'mid':
        # Use midpoint as initial guess
        x0 = p_mid[ind_pass] # Midpoint
    elif init_guess == '3quarter':
        # Use 3/4 point on the [pmin,pmax] interval
        x0 = (pmax[ind_pass]-pmin[ind_pass])*0.75 + pmin[ind_pass] # 3/4 point
    elif init_guess == 'linear_interp':
        # Secant (inverse linear interpolation)
        x0 = pmax[ind_pass] -f2[ind_pass]*(pmax[ind_pass]-pmin[ind_pass])/(f2[ind_pass]-f1[ind_pass])
    elif init_guess == 'quadratic_interp':
        # Inverse quadratic interpolation using a,b, mid
        # Using the points xx0=pmin, xx1 = pmid, xx2=pmax 
        # and their valuse fx0=f1, fx1=fmid=f(pmid), fx2=f2
        xx0 = pmin
        xx1 = p_mid
        xx2 = pmax
        fx0=f1
        fx1 = func_dVdp_vec(p_mid, r1,r2,U1,U2,V1,V2,delta,mu,pmin,pmax) # Function value at midpoint
        fx2 = f2
        # Find coeffs of Lagrange polynomial
        L0 = (xx0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
        L1 = (xx1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
        L2 = (xx2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
       
        # Now find new point from sum of these terms
        x0 = L0+L1+L2
        x0 = x0[ind_pass]
    
    elif init_guess == 'Interpolation':
        # Solve the problem over a small number of grid points covering the
        # the full range of delta values.
        # Use the solutions for these to fit a curve (e.g. spline).
        # Then interpolate new initial guesses from this fit.
        # E.g. use interpolate.splrep
        
        # 1. Pick the points
        
        # 2. Loop over points and solve
        # for pt in points:
        #   pt = optimize.brentq(func_dVdp,pmin[ind],pmax[ind],args=(r1,r2,U1,U2,V1,V2,delta,mu), rtol=1.0E-08)
        #   # Evaluate sample points xs=pt,ys=f(p)
        
        # 3. Interpolate
        # spl = interpolate.splrep(x, y) # Spline
        # x0 = interpolate.splev(delta[ind_pass], spl) # Interpolated values
        pdb.set_trace()
        
    
    # p-minimization method ---------------------------------------------------
    # Find the optimal p value for each grid point by either
    # a) solving dV(p)/dp = 0; or
    # b) minimizing dV(p)
    
    # Specify solver method to use
    min_method = 'chandrupatla' # Chandrupatla's method - much faster than Newton. Converges in <20 iterations
    # min_method = 'newton' # Use optimize.newton - not guaranteed to converge
    
    # Other methods to consider
    # root is not practical - long run time
    # fsolve is not practical - long run time
    # fmin - problems with argument shapes on iterations.
    
    # Brentq ------------------------------------------------------------------
    if min_method == 'brent':
        # Use Cythonized implementation of a loop calling scipy.optimize.brentq
        pdb.set_trace()
    
    elif min_method == 'chandrupatla':
        # Vectorized version of Chandrupatla's algorithm.
        # Obtained from https://github.com/scipy/scipy/issues/7242
        
        
        x_sol, iters = chandrupatla(func_dVdp_vec,pmin[ind_pass],pmax[ind_pass], 
                             args=(r1[ind_pass,:],r2[ind_pass,:],
                                      U1[ind_pass,:],U2[ind_pass,:],
                                      V1[ind_pass,:],V2[ind_pass,:],
                                      delta[ind_pass],mu[ind_pass],
                                      pmin[ind_pass],pmax[ind_pass]),
                             return_iter=True,
                             )
        
        # Find converged solutions
        # All converge in < 20 iterations
        converged = iters < 50
    
    
    # Using newton ------------------------------------------------------------
    elif min_method == 'newton': # Use scipy.optimize.newton method
        # Vectorized root finding (solves dV/dp=0)
        # Slow, problems with convergence of some points.
        
        (x_sol, converged, zero_der) = optimize.newton(func_dVdp_vec, x0, 
                                args=(r1[ind_pass,:],r2[ind_pass,:],
                                      U1[ind_pass,:],U2[ind_pass,:],
                                      V1[ind_pass,:],V2[ind_pass,:],
                                      delta[ind_pass],mu[ind_pass],
                                      pmin[ind_pass],pmax[ind_pass]),
                                full_output=True,
                                # x1=x0_sec,
                                maxiter=100,rtol=1.0E-04,tol=1.0E-5,
                                )
        
    
        
    
    # Fix problem p points ----------------------------------------------------
    if min_method == 'newton':
        # The solution p should be a smoothly varying function.
        # Experimentation shows that plotting p as a function of the transfer
        # angle delta gives an approximate sigmoid function.
        #
        # >> plt.plot(delta[ind_pass],x_sol)
        #
        #   ^                    x
        #   |               x
        #   |            x
        # p |         x
        #   |      x
        #   |    x
        #   |x_________________________>
        #               delta
        
        # The non-converged points are discontinuous on this line.
        # We can use the converges solutions to form a trend line, from which
        # we can interpolate to find new guesses for the non-converged solutions.
        
        # pdb.set_trace()
        
        # Create new x,y vectors from the converged solutions
        xfit = delta[ind_pass][converged] # Delta values
        yfit = x_sol[converged] # Converged p values
        # Now interpolate the non-converged solutions
        f_fit = interpolate.interp1d(xfit, yfit) # Fit function
        # Evaluate new values
        x_sol[~converged] = f_fit(delta[ind_pass][~converged])
        
        # Check how close each initial solution was
        # x0mid = p_mid[ind_pass] # Midpoint
    
    
    # Evaluate function at these solution values ------------------------------
    f_sol = func_dV_vec(x_sol, r1[ind_pass,:],r2[ind_pass,:],
                    U1[ind_pass,:],U2[ind_pass,:],
                    V1[ind_pass,:],V2[ind_pass,:],
                    delta[ind_pass],mu[ind_pass])
    
    df_sol = func_dVdp_vec(x_sol, r1[ind_pass,:],r2[ind_pass,:],
                    U1[ind_pass,:],U2[ind_pass,:],
                    V1[ind_pass,:],V2[ind_pass,:],
                    delta[ind_pass],mu[ind_pass],
                    pmin[ind_pass],pmax[ind_pass])
    
    # Return results
    dV = np.zeros(len(th1))*np.nan # Initialize
    dV[ind_pass] = f_sol
    
    
    # Debuging. Find peaks in map. Check their initial guess.
    # May be converging to a different solution.
    
    # By plotting positions of non-converged solutions
    # >> plt.plot(th1[ind_pass][~converged],th2[ind_pass][~converged],'*r')
    # we find that the discontinuities in the maps are at the same locations.
    
    return dV

def func_dV_vec(p, r1,r2,U1,U2,V1,V2,delta,mu):
    '''
    Delta-V for a particular parameter of a fixed transfer geometry (r1,r2).
    (Vectorized implementation)

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    dIdp : TYPE
        DESCRIPTION.

    '''
    
    # Compute v and z parameters eq 18 & 19.
    dr = r2-r1
    r1_cross_r2m = np.linalg.norm(np.cross(r1,r2), axis=-1)
    v = np.sqrt(mu*p)[:, np.newaxis]*(r2-r1)/r1_cross_r2m[:, np.newaxis]
    z = np.sqrt(mu/p)*np.tan(delta/2)
    
    # Short Transfer (upper sign)
    I1S = (v + z[:, np.newaxis]*U1) - V1 # 1st impulse vector
    I2S = V2 - (v - z[:, np.newaxis]*U2) # 2nd inpulse vector
    I1magS = np.linalg.norm(I1S, axis=-1) # 1st impulse magnitude
    I2magS = np.linalg.norm(I2S, axis=-1) # 2nd impulse magnitude
    ImagS = I1magS + I2magS # Total impulse magnitude
    # Derivative
    dot1S = np.einsum('ij,ij->i',I1S, (v-z[:, np.newaxis]*U1) ) # dot(I1S, v-z*U1)
    dot2S = np.einsum('ij,ij->i',I2S, (v+z[:, np.newaxis]*U2) ) # dot(I2S, v+z*U2)
    dIdpS = (dot1S/I1magS - dot2S/I2magS)/(2.*p)
    
    # Long Transfer (lower sign)
    I1L = -(v + z[:, np.newaxis]*U1) - V1 # 1st impulse vector
    I2L = V2 + (v - z[:, np.newaxis]*U2) # 2nd inpulse vector
    I1magL = np.linalg.norm(I1L, axis=-1) # 1st impulse magnitude
    I2magL = np.linalg.norm(I2L, axis=-1) # 2nd impulse magnitude
    ImagL = I1magL + I2magL # Total impulse magnitude
    # Derivative
    dot1L = np.einsum('ij,ij->i',I1L, (v-z[:, np.newaxis]*U1) ) # dot(I1S, v-z*U1)
    dot2L = np.einsum('ij,ij->i',I2L, (v+z[:, np.newaxis]*U2) ) # dot(I2S, v+z*U2)
    dIdpL = (dot1L/I1magL - dot2L/I2magL)/(2.*p)
    
    # Select optimal transfer for return
    I = np.zeros(len(p))*np.nan # Initialize output vector
    ind_S = ImagS < ImagL # Indices where Short transfer is more efficient
    I[ind_S] = ImagS[ind_S]
    I[~ind_S] = ImagL[~ind_S]
    
    return I


def func_dVdp_vec(p, r1,r2,U1,U2,V1,V2,delta,mu,pmin,pmax):
    '''
    Partial derivative of delta-V wrt parameter dV/dp.
    Needed to find optimal p for given transfer geometry.
    (Vectorized implementation)

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    dIdp : TYPE
        DESCRIPTION.

    '''
    
    # Ensure p is within bounds
    p[p<pmin] = pmin[p<pmin]
    p[p>pmax] = pmax[p>pmax]
    p[p<=0.] = 0.1
    
    
    # Compute v and z parameters eq 18 & 19.
    dr = r2-r1
    r1_cross_r2m = np.linalg.norm(np.cross(r1,r2), axis=-1)
    # Error handling for r1xr2 = 0
    # r1_cross_r2m[r1_cross_r2m==0.] = 1E-30 # Replace with small number
    
    v = np.sqrt(mu*p)[:, np.newaxis]*(r2-r1)/r1_cross_r2m[:, np.newaxis]
    z = np.sqrt(mu/p)*np.tan(delta/2)
    
    # Short Transfer (upper sign)
    I1S = (v + z[:, np.newaxis]*U1) - V1 # 1st impulse vector
    I2S = V2 - (v - z[:, np.newaxis]*U2) # 2nd inpulse vector
    I1magS = np.linalg.norm(I1S, axis=-1) # 1st impulse magnitude
    I2magS = np.linalg.norm(I2S, axis=-1) # 2nd impulse magnitude
    ImagS = I1magS + I2magS # Total impulse magnitude
    # Derivative
    dot1S = np.einsum('ij,ij->i',I1S, (v-z[:, np.newaxis]*U1) ) # dot(I1S, v-z*U1)
    dot2S = np.einsum('ij,ij->i',I2S, (v+z[:, np.newaxis]*U2) ) # dot(I2S, v+z*U2)
    dIdpS = (dot1S/I1magS - dot2S/I2magS)/(2.*p)
    
    # Long Transfer (lower sign)
    I1L = -(v + z[:, np.newaxis]*U1) - V1 # 1st impulse vector
    I2L = V2 + (v - z[:, np.newaxis]*U2) # 2nd inpulse vector
    I1magL = np.linalg.norm(I1L, axis=-1) # 1st impulse magnitude
    I2magL = np.linalg.norm(I2L, axis=-1) # 2nd impulse magnitude
    ImagL = I1magL + I2magL # Total impulse magnitude
    # Derivative
    dot1L = np.einsum('ij,ij->i',I1L, (v-z[:, np.newaxis]*U1) ) # dot(I1S, v-z*U1)
    dot2L = np.einsum('ij,ij->i',I2L, (v+z[:, np.newaxis]*U2) ) # dot(I2S, v+z*U2)
    dIdpL = (dot1L/I1magL - dot2L/I2magL)/(2.*p)
    
    # Select optimal transfer for return
    dIdp = np.zeros(len(p))*np.nan # Initialize output vector
    ind_S = ImagS < ImagL # Indices where Short transfer is more efficient
    dIdp[ind_S] = dIdpS[ind_S]
    dIdp[~ind_S] = dIdpL[~ind_S]
    
    return dIdp


#%% Problem Solving function

def solve_OrbitToOrbit_mccue(orb1,orb2,
                             mu,solver='SHGO'):
    '''
    Solve the Orbit-to-Orbit problem using the McCue formulation.

    Parameters
    ----------
    orb1 : (Dict) Orbital elements of initial orbit.
    orb2 : (Dict) Orbital elements of final orbit.
    mu : (float) Gravitational parameter of central body.
    solver : TYPE, optional
        Scipy solver method. The default is 'SHGO'.

    Returns
    -------
    result : TYPE
        Result of the scipy optimizer.

    '''
    # Extract elements
    ai = orb1['a']
    ei = orb1['e']
    inci = orb1['i']
    omi = orb1['om']
    wi = orb1['w']
    
    # Extract elements
    af = orb2['a']
    ef = orb2['e']
    incf = orb2['i']
    omf = orb2['om']
    wf = orb2['w']
    
    # Transform elements to re-reference to final orbit
    a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2 = transform_orbits(ai,ei,wi,omi,inci,af,ef,wf,omf,incf,reference='f')
    
    # Solver ------------------------------------------------------------------
    # Different options for Scipy.Optimize solvers.
    # We want to find a GLOBAL solution.
    # The problem is multivariate and non-convex
    # See: https://machinelearningmastery.com/function-optimization-with-scipy/
    # see: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    
    if solver == 'SHGO':
        # SHGO - Simplicial Homology Global Optimization
        result = optimize.shgo(f_mccue, bounds=[(0.,2*np.pi), (0., 2*np.pi)],
                               args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                               n=10,iters=2,sampling_method='sobol')
    
    elif solver == 'Brute':
        # Brute force solver
        (x0,fval,grid,Jout) = optimize.brute(f_mccue, ranges=[(0.,2*np.pi), (0., 2*np.pi)],
                                args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                                Ns=5,full_output=True,
                                )
        
        # Format result into dictionary
        result_dict = {'x':x0,'fun':fval,'x_grid':grid,'f_grid':Jout}
        # Convert to simple namespace to allow dot access to variables
        result = SimpleNamespace(**result_dict)
        
    elif solver == 'Grid':
        # Apply a grid-search method. 
        # Similar to scipy.optimize.brute, but we are specifiying the grid to use.
        # This alows use of the vectorized function to evaluate.
        # (Problems in passing vectorized function to Brute)
        
        # Generate grid of parameter space x=(th1,th2)
        Ns = 20 # Number of grid points in each direction
        x0_ = np.linspace(0.,2*np.pi,Ns)
        x1_ = np.linspace(0.,2*np.pi,Ns)
        x0grid, x1grid = np.meshgrid(x0_, x1_, indexing='ij')
        x0 = x0grid.flatten()
        x1 = x1grid.flatten()
        x = np.column_stack((x0,x1))
        # Evaluate at grid points
        dV = f_mccue_vec(x,
                         a1*np.ones(Ns**2),e1*np.ones(Ns**2),inc1*np.ones(Ns**2),
                         w1*np.ones(Ns**2),om1*np.ones(Ns**2),a2*np.ones(Ns**2),
                         e2*np.ones(Ns**2),inc2*np.ones(Ns**2),w2*np.ones(Ns**2),
                         om2*np.ones(Ns**2),mu*np.ones(Ns**2))
        
        # Find global solution to use as initial guess
        ind = np.nanargmin(dV)
        x0_guess = x[ind,:]
        
        # Finish with an optimization method ----------------------------------
        
        # Using fmin (Downhill simplex algorithm)
        res = optimize.fmin(f_mccue, x0_guess, 
                      args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
                      full_output=True,
                      disp=False)
        # Extract results res = (x0,fval,n_iter,funcals,warnflag,allvecs)
        # Format result into dictionary
        result_dict = {'x':res[0],'fun':res[1],'iter':x[2],'funcals':x[3],'warnflag':x[4]}
        # Convert to simple namespace to allow dot access to variables
        result = SimpleNamespace(**result_dict)
        
        # # Using minimize (slower than fmin)
        # # *** TODO: Provide jacobian for faster evaluation.
        # result = optimize.minimize(f_mccue, x0_guess, 
        #               args=(a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu),
        #               )
        
    
    # elif solver =='BH':
        # Basin Hopping
    
    
    return result

#%% Decoding solution

def decode_solution_mccue(orb1,orb2,mu,result):
    '''
    Find the characteristics of the optimal transfer orbit from the independent
    variables of the solution, using the McCue formulation.

    Parameters
    ----------
    result : Dict
        Optimization results, containing solution x.

    '''
    # Extract initial orbit
    ai = orb1['a']
    ei = orb1['e']
    inci = orb1['i']
    omi = orb1['om']
    wi = orb1['w']
    
    # Extract final orbit
    af = orb2['a']
    ef = orb2['e']
    incf = orb2['i']
    omf = orb2['om']
    wf = orb2['w']
    
    # Extract the solution
    x = result.x
    
    # Transform elements to re-reference to final orbit
    a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2,B = transform_orbits(ai,ei,wi,omi,inci,af,ef,wf,omf,incf,
                                                             reference='f',return_basis=True)
    
    # Find Transformation matrix
    # The above transforms the problem to a coordinate system wrt to the reference orbit.
    # The Basis B={b1,b2,b3} gives the transformation matrix from inertial space
    # to the new reference system. Take the inverse of this matrix to find the 
    # reverse transform back to inertial space.
    
    # Inertial to reference
    # X_ref = A_inert_to_ref*X_inertial
    A_inert_to_ref = np.column_stack((B[0],B[1],B[2]))
    
    # Reference to inertial
    # X_inert = A_ref_to_inert*X_ref
    A_ref_to_inert = np.linalg.inv(A_inert_to_ref) # Inverse of A
    
    # Evaluate the function at the solution
    f,r1_ref,r2_ref,vtx1_ref,vtx2_ref,I1_ref,I2_ref,tx_type,pt = f_mccue(x,
                                              a1,e1,inc1,w1,om1, # Initial orbit
                                              a2,e2,inc2,w2,om2,  # Final orbit 
                                              mu,full_output=True)
    
    # Transform the position and velocity vetors to the original orbit frame
    # (need to convert each vector to column vectors)
    # First, stack the points together to perform the transform in one call.
    points_ref = np.row_stack((r1_ref,r2_ref,vtx1_ref,vtx2_ref,I1_ref,I2_ref))
    # Apply the transform 
    # (use transform to reshape the points into column vectors)
    points_inert = np.dot(A_ref_to_inert, points_ref.T).T
    # Extract vectors in inertial frame
    r1 = points_inert[0,:]
    r2 = points_inert[1,:]
    vtx1 = points_inert[2,:]
    vtx2 = points_inert[3,:]
    I1 = points_inert[4,:]
    I2 = points_inert[5,:]
    
    # Convert state vectors to elements
    atx,etx,itx,omtx,wtx,TAtx1 = coe_from_sv(r1,vtx1,mu=mu,units='km') # 1st impulse
    atx2,etx2,itx2,omtx2,wtx2,TAtx2 = coe_from_sv(r2,vtx2,mu=mu,units='km') # 2nd impulse
    
    return atx,etx,itx,omtx,wtx,TAtx1,TAtx2

#%% Plotting functions

def plot_porkchop_mccue(orb1,orb2,mu,method='vectorized'):
    '''
    Plot a contour plot of the delta-V over a grid of departure and arrival
    positions (th1,th2).

    Parameters
    ----------
    orb1 : (Dict) Orbital elements of initial orbit.
    orb2 : (Dict) Orbital elements of final orbit.
    mu : (float) Gravitational parameter of central body.

    '''
    
    # Extract elements
    ai = orb1['a']
    ei = orb1['e']
    inci = orb1['i']
    omi = orb1['om']
    wi = orb1['w']
    
    # Extract elements
    af = orb2['a']
    ef = orb2['e']
    incf = orb2['i']
    omf = orb2['om']
    wf = orb2['w']
    
    # Transform elements to re-reference to final orbit
    a1,e1,w1,om1,inc1,a2,e2,w2,om2,inc2 = transform_orbits(ai,ei,wi,omi,inci,af,ef,wf,omf,incf,reference='f')
    
    
    # Generate grid of parameter space x=(th1,th2)
    x0_ = np.linspace(0.,2*np.pi,100)
    x1_ = np.linspace(0.,2*np.pi,100)
    x0grid, x1grid = np.meshgrid(x0_, x1_, indexing='ij')
    x0 = x0grid.flatten()
    x1 = x1grid.flatten()
    
    if method == 'loop':
        # Loop through grid points (x0,x1) and evaluate
        dV = np.zeros(len(x0))*np.nan # Initialize output    
        for n in tqdm(range(len(x0))):
            # Extract parameter
            x = [x0[n],x1[n]]
            # Compute objective
            dV[n] = f_mccue(x,a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu)
    elif method == 'vectorized':
        # Use vectorized function to evaluate in single function call
        
        # Form elements into vectors matching length of grid
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
        
        # Evaluate the delta-V at these points
        x = np.column_stack((x0,x1))
        dV = f_mccue_vec(x,a1,e1,inc1,w1,om1,a2,e2,inc2,w2,om2,mu*np.ones(len(x0)))  
    
    # Remove large delta-Vs
    dV[dV > 10.] = np.nan
    
    # Compute global and local solutions
    result = solve_OrbitToOrbit_mccue(orb1,orb2, mu,solver='SHGO')
    
    # Create porkchop plot
    # Contour plot
    fig1, ax = plt.subplots()
    CS = ax.contourf(x0grid, x1grid, dV.reshape(x0grid.shape), 
                      cmap='jet',
                      )
    
    cbar = plt.colorbar(CS)
    cbar.ax.set_title('dV (km/s)',fontsize=8)
    
    # Plot global solution
    ax.plot(result.x[0],result.x[1],'or',markersize=5)
    
    # Plot local solutions
    fl = result.funl # Values
    xl = result.xl # Positions
    if xl.shape == (2,):
        # Only a single solution
        pass
    elif len(fl)>1:
        # Multiple solutions. Loop through each.
        for i in range(len(fl)):
            ax.plot(xl[i,0],xl[i,1],'og',markersize=10)
    
    ax.set_xlabel('th1 (rad)')
    ax.set_ylabel('th2 (rad)')
    
    return


#%% ###########################################################################
#                        Eckel Formulation
# 
# Two-impulse Orbit-to-Orbit computation using the formulation described in
# Eckel & Vinh (1984) "Optimal Switching conditions for minimum fuel fixed time 
# transfer between non coplanar ellipctical orbits."
###############################################################################

# See OrbitToOrbit_old.py
