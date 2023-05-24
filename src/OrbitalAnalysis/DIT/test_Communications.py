# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:29:38 2022

@author: scott
"""

from Communications import *


#%% Test Two-way Radar equation

def test_two_way_radar_equation_1():
    
    # Example from: https://www.rfcafe.com/references/electrical/ew-radar-handbook/two-way-radar-equation.htm
    
    # "Assume that a 5 GHz radar has a 70 dBm (10 kilowatt) signal fed through 
    # a 5 dB loss transmission line to a transmit/receive antenna that has 45 dB 
    # gain. An aircraft that is flying 31 km from the radar has an RCS of 9 m2. 
    # What is the signal level at the input to the radar receiver? (There is an 
    # additional loss due to any antenna polarization mismatch but that loss 
    # will not be addressed in this problem)."
    
    
    
    # Constants
    c = 2.99792458E8 # Speed of light (m/s)
    
    # Extract parameters
    Pt = 10*np.log10(10E3) # Transmit Power = 40 dBW (from 10000 W) == 70 dBm
    Gt = 45 - 5 # Transmitter gain = 40 dBi (45 dB gain, 5 dB loss)
    rcs = 9 # Radar cross section (m^2)
    f = 5 # Frequency (GHz)
    R = 31 # Distance (km)
    lam = c/(f*1E9) # Wavelenth (m)
    tp = 1E-7 # Pulse width (s)
    Ts = 290 # System temperature (K)
    L = 0 # Additional losses (dB)
    
    
    # Received power
    EIRP = Pt + Gt 
    Ls =  20*np.log10(R) + 20*np.log10(f) + 20*np.log10(4*np.pi/c) + 2*30 + 2*90 # = 136.25 dB - correct
    
    # Gain
    # G = 10*np.log10(4*np.pi*rcs/lam**2) # = 44.9775 dB
    # G = 10*np.log10( 4*np.pi*rcs*((f*1E9)/c)**2 ) # = 44.9775
    G = 10*np.log10(rcs) + 20*np.log10(f) + 20*np.log10(1E9) + 10*np.log10(4*np.pi/c**2) # = 44.977 dB
    
    # Receiver power
    Pr1 = Pt + Gt - Ls + G - Ls + Gt # = -107.535 dBW == -77.535 dBm - correct
    
    # Test function
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    assert np.around(Pr+30,2) == -77.53 # Check value against known (adjusted by 0.01)
    print("Pr = {} dBW".format(str(Pr)))
    print("Np = {} dB".format(Np))
    print("SNR = {} dB".format(str(SNR1)))
    
    return

def test_two_way_radar_equation_2():
    
    # Another example from pg 50 of "Basic Radar Analysis" textbook
    # Known SNR = 14.66 dB
    # Achieved 14.65 dB (accounting for rounding errors)
    
    # Constants
    c = 2.99792458E8 # Speed of light (m/s)
    
    # Values
    Pt = 60 # dBW
    Gt = 38 # dB
    Gr = 38 # dB
    lam = 0.0375 # m
    f = (c/lam)*1E-9 # GHz
    rcs = 3.98 # m^2
    R = 60 # km
    tp = 0.4E-6 # s
    Ts = 3423 # K
    L = 4 # dB
    
    # Compute Pr
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    assert np.around(SNR1,1) == 14.7 # Check value against known (adjusted by 0.05 due to rounding errors)
    print("Pr = {} dBW".format(str(Pr)))
    print("Np = {} dB".format(Np))
    print("SNR = {} dB".format(str(SNR1)))
    
    return

def test_two_way_radar_equation_2b():
    
    # Another example from pg 50 of "Basic Radar Analysis" textbook
    # Known SNR = 14.66 dB
    # Achieved 14.65 dB (accounting for rounding errors)
    
    # This version tests an array of R values
    
    # Constants
    c = 2.99792458E8 # Speed of light (m/s)
    
    # Values
    Pt = 60 # dBW
    Gt = 38 # dB
    Gr = 38 # dB
    lam = 0.0375 # m
    f = (c/lam)*1E-9 # GHz
    rcs = 3.98 # m^2
    R = 60*np.ones(100) # km
    tp = 0.4E-6 # s
    Ts = 3423 # K
    L = 4 # dB
    
    # Compute Pr
    Pr, Np, SNR1 = compute_link_budget(Pt,Gt,Gt,f,R,rcs,Ts,tp,L)
    print("Pr = {} dBW".format(str(Pr)))
    print("Np = {} dB".format(Np))
    print("SNR = {} dB".format(str(SNR1)))
    
    return


def test_target_gain_factor():
    
    # Test the target gain factor computations
    c = 2.99792458E8 # Speed of light (m/s)
    G = lambda f,rcs: 10*np.log10(rcs) + 20*np.log10(f) + 20*np.log10(1E9) + 10*np.log10(4*np.pi/c**2)
    
    # Check values in conversion table
    # https://www.rfcafe.com/references/electrical/ew-radar-handbook/two-way-radar-equation.htm
    
    # Assert values                        Gain |    f (GHz)| rcs (m^2)
    assert np.around(G(0.5,9),decimals=2) == 24.98  #  1 GHz, 9 m^2
    assert np.around(G(1,9),decimals=2)   == 31.00  #  1 GHz, 9 m^2
    assert np.around(G(5,9),decimals=2)   == 44.98  #  5 GHz, 9 m^2
    assert np.around(G(7,9),decimals=2)   == 47.9   #  7 GHz, 9 m^2
    assert np.around(G(10,9),decimals=2)  == 51.00  # 10 GHz, 9 m^2
    assert np.around(G(20,9),decimals=2)  == 57.02  # 20 GHz, 9 m^2
    assert np.around(G(40,9),decimals=2)  == 63.04  # 20 GHz, 9 m^2
    
    assert np.around(G(0.5,100),decimals=2) == 35.44  #  1 GHz, 100 m^2
    assert np.around(G(1,100),decimals=2)   == 41.46  #  1 GHz, 100 m^2
    assert np.around(G(5,100),decimals=2)   == 55.44  #  5 GHz, 100 m^2
    assert np.around(G(7,100),decimals=2)   == 58.36  #  7 GHz, 100 m^2
    assert np.around(G(10,100),decimals=2)  == 61.46  # 10 GHz, 100 m^2
    assert np.around(G(20,100),decimals=2)  == 67.48  # 20 GHz, 100 m^2
    assert np.around(G(40,100),decimals=2)  == 73.50  # 20 GHz, 100 m^2
    
    
    return