# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:49:36 2022

@author: scott

Communications Module
---------------------

Compute the link budget between a ground station and satellite.
- estimate Signal to Noise ratio SNR
- estimate probability of detection of radar

"""

import numpy as np
import pandas as pd

import pdb

#%% Link Budget

def compute_link_budget(Pt,Gt,Gr,f,R,rcs,Ts,tp,L=0):
    '''
    Compute the signal to noise ratio (SNR1) for a communications link between 
    a satellite and ground station.
    
    A monostatic radar transmits a carrier signal from the ground station. The
    signal propagates towards the target statellite and is reflected back to
    a receiver at the same ground station.
    
    The peak power at the radar receiver is computed from the two-way(monostatic) 
    radar equation :
        
    Pr = Pt*Gt*Gr*λ^2*σ/( (4*pi)^3*R^4) = Pt*Gt*Gr*( σ*c^2/( (4*pi)^3*f^2*R^4) )
    where
    Pt(W) is the power of the transmitter
    Gt is the gain of the transmitter
    Gr is the gain of the receiver (Gr=Gt)
    λ(m) is the wavelenth of the carrier signal
    f(Hz) is the carrier frequency
    c = 2.99792458E8 (m/s) is the speed of light 
    R(m) is the distance between the ground station and target satellite
    σ(m^2) is the radar cross section of the target satellite.
    
    This can be converted into log form, expressing terms in decibels:
    
    Pr(dBW) = [ Pt(dBW) + Gt(dBi) ]                           // EIRP
              - [ (92.45) + 20*log(R(km)) + 20*log(f(GHz)) ]  // Free-space loss Tx->Sat
              + [ 10*log(σ(m^2)) + 20*log(f(GHz)) + 10*log(4*pi/c^2) + 180 ] // Gain of RCS reflected off Sat
              - [ (92.45) + 20*log(R(km)) + 20*log(f(GHz)) ]  // Free-space los Sat->Rx
              + [ Gr(dBi) ]                                   // Gain of receiver
    
    
    Components of equation
    
    1. Transmit Power
        Transmitter uses power to amplify and broadcast a carrier signal.
        Equivalent isotropic radiated power = measure of power radiated in the
        direction of the peak of the trasmit antenna directivity pattern.
        
        EIRP(dBW) = Pt(dBW) + Gt(dBi)
    
    2. Free-space and Atmospheric Losses
        Signal attenuates with distance and path through atmosphere.
        Free space loss (Ls) of path from transmitter to target satellite:
        
        Ls(dB) = 10*log( (4*pi*r/lam)^2 ) = 10*log( (4*pi*f/c)^2 )
               = (92.45) + 20*log(r) + 20*log(f)
        where
        r is the distance (km)
        f is carrier frequency (GHz)
        92.45 = 20*log(4*pi/c) + 2*30 + 2*90
        (conversion factor from m->km is 30, and from Hz->GHz is 90)

    3. Reflected power at Target
        The power received at the satellite will reflected and redirected back 
        towards the ground station. This is equivalent to a target gain factor 
        of 10*log(4*pi*σ(m^2)/λ^2). 
        
        G = 10*log(4*pi*σ(m^2)/λ^2) = 10*log(4*pi*σ(m^2)*f^2/c^2)
          = 10*log(σ(m^2)) + 20*log(f(Hz)) + 10*log(4*pi/c^2)
          = 10*log(σ(m^2)) + 20*log(f(GHz)) + 2*10*log(1E9) + 10*log(4*pi/c^2)
          = 10*log(σ(m^2)) + 20*log(f(GHz)) + 180 - 158.54
          

    4. Free-space Loss
        Sigmal is attenuated again through path from Sat to receiver.
        Same as term 2.
    
    5. Receiver Gain
        Receiver amplifies the received power. Same gain as transmitter.
    
    
    See: https://www.rfcafe.com/references/electrical/ew-radar-handbook/two-way-radar-equation.htm

    See S16.2 of SMAD (pg 466-)
    
    Parameters
    ----------
    Pt : float
        Transmission power (dBW).
    Gt : float
        Transmission Gain (dBi).
    Gr : float
        Receiver Gain (dBi).
    f : float
        Carrier frequency (GHz).
    R : float or 1D array
        Ground station to Sat range (km).
    rcs : float
        Radar Cross Section of target satellite (m^2).
    L : float
        Other losses (dBW).

    Returns
    -------
    light, partial, dark : SpiceCell
        Time intervals for light, partial, and dark lighting conditions.

    
    '''
    
    # Constants
    c = 2.99792458E8   # Speed of light (m/s)
    k = 1.380649E-23 # Boltzmann constant (W/Hz K)
    # k = 1.38E-23
    
    # Conversion wavelenth-frequency
    # lam = c/f
    # lam = wavelenght (m)
    # f = frequency (Hz)
    
    # Free-space loss (dB)
    Ls =  20*np.log10(R) + 20*np.log10(f) + 20*np.log10(4*np.pi/c) + 2*30 + 2*90
    
    # Target gain factor
    G = 10*np.log10(rcs) + 20*np.log10(f) + 20*np.log10(1E9) + 10*np.log10(4*np.pi/c**2)
    
    # Received power (dBW)
    Pr = Pt + Gt - Ls + G - Ls + Gr - L
    # print("Pr = {} dBW".format(str(Pr)))
    
    
    # Noise
    # Np = k*Ts/tp # Noise power spectral density (W/Hz)
    Np = 10*np.log10(k*Ts/tp) # Noise power spectral density dBW
    # N = -228.6 + Ts + Bn
    # -228.6 is Boltzmann's constant in dB
    # print("Np = {} dB".format(Np))
    
    # Signal to Noise ratio
    # SNR = Pt/Np or in decibel form SNR(dB) = Pt(dBW) - Np(dB)
    SNR1 = Pr - Np
    # print("SNR = {} dB".format(str(SNR1)))
    
    
    return Pr, Np, SNR1

#%% Probability of Detection

def compute_probability_of_detection(SNR1,pfa=1e-4):
    
    # The MATLAB script ROC_cruves.m uses the rocpfa function of the Phased
    # Array Systems toolbox to generate Receiver Operating Characteristic curves
    # based on a fixed false-alarm probability.
    # See: https://au.mathworks.com/help/phased/ref/rocpfa.html
    
    if pfa != 1e-4:
        raise ValueError("Warning: alternative pfa values not implemented. " +
                         "Defaulting to 1e-4")
    
    
    # Alternative method: Implement equation directly
    # PD=12erfc(erfc−1(2PFA)−√χ)
    # where X = SNR not in decibels
    # Defined in https://au.mathworks.com/help/phased/ref/rocpfa.html#bsy5w1g
    # Reference 
    
    
    # These values are used here to interpolate values
    # SNR1 values
    x = np.array([-5.000e+01,-4.950e+01,-4.900e+01,-4.850e+01,-4.800e+01,
                  -4.750e+01,-4.700e+01,-4.650e+01,-4.600e+01,-4.550e+01,
                  -4.500e+01,-4.450e+01,-4.400e+01,-4.350e+01,-4.300e+01,
                  -4.250e+01,-4.200e+01,-4.150e+01,-4.100e+01,-4.050e+01,
                  -4.000e+01,-3.950e+01,-3.900e+01,-3.850e+01,-3.800e+01,
                  -3.750e+01,-3.700e+01,-3.650e+01,-3.600e+01,-3.550e+01,
                  -3.500e+01,-3.450e+01,-3.400e+01,-3.350e+01,-3.300e+01,
                  -3.250e+01,-3.200e+01,-3.150e+01,-3.100e+01,-3.050e+01,
                  -3.000e+01,-2.950e+01,-2.900e+01,-2.850e+01,-2.800e+01,
                  -2.750e+01,-2.700e+01,-2.650e+01,-2.600e+01,-2.550e+01,
                  -2.500e+01,-2.450e+01,-2.400e+01,-2.350e+01,-2.300e+01,
                  -2.250e+01,-2.200e+01,-2.150e+01,-2.100e+01,-2.050e+01,
                  -2.000e+01,-1.950e+01,-1.900e+01,-1.850e+01,-1.800e+01,
                  -1.750e+01,-1.700e+01,-1.650e+01,-1.600e+01,-1.550e+01,
                  -1.500e+01,-1.450e+01,-1.400e+01,-1.350e+01,-1.300e+01,
                  -1.250e+01,-1.200e+01,-1.150e+01,-1.100e+01,-1.050e+01,
                  -1.000e+01,-9.500e+00,-9.000e+00,-8.500e+00,-8.000e+00,
                  -7.500e+00,-7.000e+00,-6.500e+00,-6.000e+00,-5.500e+00,
                  -5.000e+00,-4.500e+00,-4.000e+00,-3.500e+00,-3.000e+00,
                  -2.500e+00,-2.000e+00,-1.500e+00,-1.000e+00,-5.000e-01, 
                  0.000e+00, 5.000e-01, 1.000e+00, 1.500e+00, 2.000e+00, 
                  2.500e+00, 3.000e+00, 3.500e+00, 4.000e+00, 4.500e+00, 
                  5.000e+00, 5.500e+00, 6.000e+00, 6.500e+00, 7.000e+00, 
                  7.500e+00, 8.000e+00, 8.500e+00, 9.000e+00, 9.500e+00, 
                  1.000e+01, 1.050e+01, 1.100e+01, 1.150e+01, 1.200e+01, 
                  1.250e+01, 1.300e+01, 1.350e+01, 1.400e+01, 1.450e+01, 
                  1.500e+01, 1.550e+01, 1.600e+01, 1.650e+01, 1.700e+01, 
                  1.750e+01, 1.800e+01, 1.850e+01, 1.900e+01, 1.950e+01, 
                  2.000e+01, 2.050e+01, 2.100e+01, 2.150e+01, 2.200e+01, 
                  2.250e+01, 2.300e+01, 2.350e+01, 2.400e+01, 2.450e+01, 
                  2.500e+01, 2.550e+01, 2.600e+01, 2.650e+01, 2.700e+01, 
                  2.750e+01, 2.800e+01, 2.850e+01, 2.900e+01, 2.950e+01, 
                  3.000e+01, 3.050e+01, 3.100e+01, 3.150e+01, 3.200e+01, 
                  3.250e+01, 3.300e+01, 3.350e+01, 3.400e+01, 3.450e+01, 
                  3.500e+01, 3.550e+01, 3.600e+01, 3.650e+01, 3.700e+01, 
                  3.750e+01, 3.800e+01, 3.850e+01, 3.900e+01, 3.950e+01, 
                  4.000e+01, 4.050e+01, 4.100e+01, 4.150e+01, 4.200e+01, 
                  4.250e+01, 4.300e+01, 4.350e+01, 4.400e+01, 4.450e+01, 
                  4.500e+01, 4.550e+01, 4.600e+01, 4.650e+01, 4.700e+01, 
                  4.750e+01, 4.800e+01, 4.850e+01, 4.900e+01, 4.950e+01, 
                  5.000e+01])
    
    
    # Pd values for Pfa=1e-4
    y = np.array([1.0000921067e-04,1.0001033459e-04,1.0001159566e-04,
                  1.0001301061e-04,1.0001459824e-04,1.0001637961e-04,
                  1.0001837837e-04,1.0002062105e-04,1.0002313742e-04,
                  1.0002596090e-04,1.0002912897e-04,1.0003268370e-04,
                  1.0003667229e-04,1.0004114770e-04,1.0004616939e-04,
                  1.0005180405e-04,1.0005812654e-04,1.0006522086e-04,
                  1.0007318129e-04,1.0008211362e-04,1.0009213661e-04,
                  1.0010338353e-04,1.0011600395e-04,1.0013016578e-04,
                  1.0014605749e-04,1.0016389062e-04,1.0018390268e-04,
                  1.0020636032e-04,1.0023156289e-04,1.0025984654e-04,
                  1.0029158876e-04,1.0032721348e-04,1.0036719686e-04,
                  1.0041207380e-04,1.0046244526e-04,1.0051898650e-04,
                  1.0058245647e-04,1.0065370829e-04,1.0073370116e-04,
                  1.0082351387e-04,1.0092435998e-04,1.0103760515e-04,
                  1.0116478667e-04,1.0130763579e-04,1.0146810301e-04,
                  1.0164838689e-04,1.0185096700e-04,1.0207864154e-04,
                  1.0233457038e-04,1.0262232456e-04,1.0294594313e-04,
                  1.0330999882e-04,1.0371967386e-04,1.0418084799e-04,
                  1.0470020088e-04,1.0528533159e-04,1.0594489853e-04,
                  1.0668878413e-04,1.0752828902e-04,1.0847636232e-04,
                  1.0954787571e-04,1.1075995113e-04,1.1213235441e-04,
                  1.1368797040e-04,1.1545337925e-04,1.1745955901e-04,
                  1.1974274658e-04,1.2234549843e-04,1.2531800433e-04,
                  1.2871972359e-04,1.3262143419e-04,1.3710781352e-04,
                  1.4228070732e-04,1.4826329396e-04,1.5520542027e-04,
                  1.6329047817e-04,1.7274431847e-04,1.8384687210e-04,
                  1.9694738759e-04,2.1248452140e-04,2.3101297001e-04,
                  2.5323895541e-04,2.8006773499e-04,3.1266748784e-04,
                  3.5255554886e-04,4.0171516677e-04,4.6275393801e-04,
                  5.3911903153e-04,6.3538950285e-04,7.5767260357e-04,
                  9.1413911729e-04,1.1157422267e-03,1.3771745735e-03,
                  1.7181275048e-03,2.1649222388e-03,2.7525803219e-03,
                  3.5273838274e-03,4.5499359733e-03,5.8986613702e-03,
                  7.6735750397e-03,1.0000000000e-02,1.3031735466e-02,
                  1.6952998947e-02,2.1978331253e-02,2.8349622523e-02,
                  3.6329548390e-02,4.6191035507e-02,5.8202898049e-02,
                  7.2612435581e-02,8.9626436648e-02,1.0939254005e-01,
                  1.3198312771e-01,1.5738377728e-01,1.8548779426e-01,
                  2.1609756673e-01,2.4893259770e-01,2.8364324522e-01,
                  3.1982858432e-01,3.5705648136e-01,3.9488395392e-01,
                  4.3287612811e-01,4.7062251566e-01,5.0774981697e-01,
                  5.4393092935e-01,5.7889024013e-01,6.1240558004e-01,
                  6.4430739485e-01,6.7447577436e-01,7.0283597744e-01,
                  7.2935303562e-01,7.5402592810e-01,7.7688171795e-01,
                  7.9796993682e-01,8.1735741226e-01,8.3512365345e-01,
                  8.5135684866e-01,8.6615048127e-01,8.7960053863e-01,
                  8.9180326697e-01,9.0285341386e-01,9.1284289494e-01,
                  9.2185982146e-01,9.2998782843e-01,9.3730564852e-01,
                  9.4388688257e-01,9.4979992459e-01,9.5510800509e-01,
                  9.5986932268e-01,9.6413723917e-01,9.6796051824e-01,
                  9.7138359177e-01,9.7444684117e-01,9.7718688440e-01,
                  9.7963686124e-01,9.8182671153e-01,9.8378344272e-01,
                  9.8553138410e-01,9.8709242612e-01,9.8848624401e-01,
                  9.8973050535e-01,9.9084106172e-01,9.9183212487e-01,
                  9.9271642803e-01,9.9350537305e-01,9.9420916439e-01,
                  9.9483693062e-01,9.9539683455e-01,9.9589617277e-01,
                  9.9634146542e-01,9.9673853718e-01,9.9709259011e-01,
                  9.9740826914e-01,9.9768972090e-01,9.9794064652e-01,
                  9.9816434896e-01,9.9836377542e-01,9.9854155531e-01,
                  9.9870003428e-01,9.9884130461e-01,9.9896723242e-01,
                  9.9907948199e-01,9.9917953747e-01,9.9926872228e-01,
                  9.9934821650e-01,9.9941907228e-01,9.9948222771e-01,
                  9.9953851915e-01,9.9958869219e-01,9.9963341155e-01,
                  9.9967326977e-01,9.9970879508e-01,9.9974045834e-01,
                  9.9976867928e-01,9.9979383203e-01,9.9981625010e-01,
                  9.9983623074e-01,9.9985403891e-01,9.9986991078e-01,
                  9.9988405687e-01,9.9989666478e-01,9.9990790176e-01])
    
    import matplotlib.pyplot as plt
    
    # Interpolate input values
    PD = np.interp(SNR1,x,y,left=np.nan)
    
    # # Plot values
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot()
    # plt.plot(x,y,'-b',label='Pfa='+str(pfa))
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    return PD

#%% Phased Array Design

def get_array_geometry():
    '''
    Read the positions of the array elements from the STK Radar1.rd file.
    
    The contains *** elements, spaced over 32x32

    '''
    
    # Radar1.rd contains the X,Y positions of elements.
    # Each block is 14 lines long. Starting block at line 147. Example:
    # <SCOPE>
    #     <VAR name = "X">
    #         <REAL>-24.39187446136487</REAL>
    #     </VAR>
    #     <VAR name = "Y">
    #         <REAL>-24.39187446136487</REAL>
    #     </VAR>
    #     <VAR name = "Id">
    #         <INT>0</INT>
    #     </VAR>
    #     <VAR name = "Enabled">
    #         <BOOL>false</BOOL>
    #     </VAR>
    # </SCOPE>
    
    
    # Read file into dataframe
    df = pd.read_csv('Radar1.rd',header=None)
    n_rows = len(df)
    
    # Extract X values starting at line 149 to 61119 (indices 146)
    ind_list = np.arange(146,61116+14,14)
    dfx = df.iloc[ind_list]
    x = dfx[0].str.extract(r'([-+]?\d+.\d+)').astype('float').to_numpy().T[0] # Extract numbers
        
    # Extract Y values starting at line 152 to 61122
    ind_list = np.arange(149,61119+14,14)
    dfy = df.iloc[ind_list]
    y = dfy[0].str.extract(r'([-+]?\d+.\d+)').astype('float').to_numpy().T[0] # Extract numbers
      
        # Extract Ind values starting at line 155
    ind_list = np.arange(152,61122+14,14)
    dfind = df.iloc[ind_list]
    ind = dfind[0].str.extract(r'(\d+)').astype(int).to_numpy().T[0] # Extract numbers
    
    # Reproduce without reading 
    x1 = np.repeat(np.linspace(-24.39187446,24.39187446,66),66)
    y1 = np.tile(np.linspace(-24.39187446,24.39187446,66),66)

    # Create dataframe
    df = pd.DataFrame({'ID':ind,'x':list(x),'y':list(y)})
    
    return df

#%% Decibel Conversions
 
# P(dB) = 10*log(P/Pref)
# Using Pref = 1 W

# P(dBW) = 10*log[P(W)/1]
# converts power in W to power in dBW


#%% Antenna gain

# G = n*(4*pi/lam^2)*A
# where
# n = antenna efficiency
# lam = wavelenth (m)
# A = antenna area (m^2)

#%% Antenna Noise Temperature



#%% Atmospheric losses

# Look up Zeneth attenuation as fn of frequency [ITU 2011]
# For 5 < El < 90 deg,
# Attenuation increases by cosec of elevation angle.


# Rain losses
# Dependent on geographic location of ground station
# lookup from ITU 2002.


#%% Notes for STK Values

# STK Radar Object Default Values
# From Radar Tutorial: https://www.youtube.com/watch?v=CQytJB9eiIY

# Type: Monostatic
# PRF = 0.001 MHz (Pulse Repetition Frequency) = num pulses per second
# Pulse Width = 1e-07 sec
# Goal SNR = 16 dB, Maximum pulses 512

# System Temperature
# Ts = 290 K
# Based on STK documentation: https://help.agi.com/stk/11.0.1/Content/comm/CommRadar02-03.htm
# Also found in stk files Radar1.rd

# Noise Bandwidth/Pulse width
# tp = 1E-7 # Pulse width (s)

# Notes
# From documentation [1]: "SNR1 for CW radar is based upon a one second pulse width"
# which would mean Tint = 1 s, hence BN = 1/1 = 1 Hz.
# Although, further on it says that for PSN, the default pulse width is 1e-07 s
# which would give Bn = 1E7 Hz.
# [1] https://help.agi.com/stk/11.0.1/Content/comm/CommRadar05-02.htm
