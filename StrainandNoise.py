import numpy as np
import os
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

import IMRPhenomD as PhenomD

def calcPTAASD(sigma_rms,cadence,T_obs,ndetectors,nfreqs=int(1e3),A_stoch_back = 4e-16):
    #####################################################
    #PSD of a PTA taken from eqn 40 of Thrane & Romano 2013
    #Equation 5 from https://arxiv.org/pdf/1808.10071.pdf

    #Stochastic background amplitude from https://arxiv.org/pdf/1603.09348.pdf
    
    #frequency range of full PTA
    f_year = 1/u.yr
    P_w_tot = 0
    T_obs_tot = 0
    if ndetectors == 1:
        P_w = 2*sigma_rms**2/cadence #Avg white noise from pulsar array [s**2/Hz]
        f = np.logspace(np.log10(1/T_obs.value),np.log10(cadence.value/2),nfreqs)*u.Hz
    else:
        #Sum of pulsar noises in different time periods divided by total time
        for i in range(ndetectors):
            P_w_n = sigma_rms[i]**2*(T_obs[i]/cadence[i]) #Avg white noise from pulsar arrays [s**2/Hz]
            P_w_tot += P_w_n
            T_obs_tot += T_obs[i]
        P_w = 2*P_w_tot/T_obs_tot
        f = np.logspace(np.log10(1/T_obs_tot.value),np.log10(min(cadence).value/2),nfreqs)*u.Hz
        
    #P_red = A_red.*(f./f_year).**(-gamma) # red noise for some pulsar
    P_red = 0.0*u.s*u.s/u.Hz #Assume no pulsar red noise for simplicity
    
    #eqn 42 of T&R
    #h_inst = np.sqrt((12*np.pi**2*(P_w+P_red))*f**3)
    ##################################################
    #Stochastic background amplitude from https://arxiv.org/pdf/1603.09348.pdf
    P_sb = (A_stoch_back**2/12/np.pi**2)*f**(-3)*(f/f_year.to('Hz'))**(-4/3)
    #h_sb = A_stoch_back*(f/f_year)**(-2/3)
    ####################################################
    #PSD of the full PTA from Lam 2018
    P_n = P_w + P_red + P_sb
    
    #strain of the full PTA
    #h_f = np.sqrt((12*np.pi**2)*f**3*P_n)
    ASD = np.sqrt((12*np.pi**2)*f**2*P_n)
    return f,ASD

def calcASD(f,T,P_acc,P_ims,L):
    #Calculates amplitude spectral density
    PSD = 20/3/T**2*(4*P_acc+P_ims)/L**2 #Power Spectral Density
    return np.sqrt(PSD+Sgal4yr(f)) #Sqrt of PSD

def approxResponseFunction(f,L):
    #Response function approximation from Calculation described by Neil Cornish (unpublished)
    f_L = const.c/2/np.pi/L #Transfer frequency
    R_f = 3/10/(1+0.6*(f/f_L)**2) 
    return R_f

def Get_TransferFunction(L=2.5*u.Gm.to('m')*u.m,f_low=1e-5*u.Hz,f_high=1.0*u.Hz):
    LISA_Transfer_Function_filedirectory = os.getcwd() + '/LoadFiles/LISATransferFunction/'
    LISA_Transfer_Function_filename = 'transfer.dat' #np.loadtxting transfer function for Lisa noise curve
    LISA_Transfer_Function_filelocation = LISA_Transfer_Function_filedirectory + LISA_Transfer_Function_filename
    LISA_Transfer_Function_data = np.loadtxt(LISA_Transfer_Function_filelocation)
    fc = const.c/(2*L)  #light round trip freq
    LISA_Transfer_Function_f = fc*LISA_Transfer_Function_data[:,0]

    idx_f_5 = np.abs(LISA_Transfer_Function_f-f_low).argmin()
    idx_f_1 = np.abs(LISA_Transfer_Function_f-f_high).argmin()

    LISA_Transfer_Function = LISA_Transfer_Function_data[idx_f_5:idx_f_1,1]
    LISA_Transfer_Function_f = LISA_Transfer_Function_f[idx_f_5:idx_f_1]
    return [LISA_Transfer_Function_f,LISA_Transfer_Function]

def NeilSensitivity(f,T,S_acc=3e-15*u.m/u.s/u.s,S_oms=1.5e-11*u.m,L=2.5e9*u.m):   
    #Uses Calculation described by Neil Cornish (unpublished)
    f_L = const.c/2/np.pi/L #Transfer frequency
    P_acc = (S_acc)**2*(1+(0.4e-3*u.Hz/f)**2)*(1+(f/(8e-3*u.Hz))**4) #acceleration noise [Hz]**-1
    P_oms = (S_oms)**2*(1+(2e-3*u.Hz/f)**4) #Optical metrology noise [Hz]**-1
    
    P_n_f = P_oms/L**2 + 2*(1.0+(np.cos(f.value/f_L.value))**2)*P_acc/(2*np.pi*f)**4/L**2 #Total noise
    #R_f = approxResponseFunction(f,L) #Response function approximation
    #ASD = np.sqrt(P_n_f/R_f+Sgal4yr(f)) #Amplitude Spectral Density approx
    ASD = np.sqrt(10/3/T**2*P_n_f+Sgal4yr(f))
    return ASD

def LisaSensitivity(f,T,S_acc_low_knee=.4*u.mHz,S_acc_high_knee=8.*u.mHz,S_oms_knee=2.*u.mHz ,S_acc=3e-15*u.m/u.s/u.s,S_ims=10e-12*u.m,L=2.5e9*u.m):
    S_acc_low_knee.to('Hz')
    S_acc_high_knee.to('Hz')
    S_oms_knee.to('Hz')

    P_acc = S_acc**2*(1+(S_acc_low_knee/f)**2)*(1+(f/(S_acc_high_knee))**4)/(2*np.pi*f)**4 #Acceleration Noise 
    P_ims = S_ims**2*(1+(S_oms_knee/f)**4) #Displacement noise of the interferometric TM--to-TM 
    ASD = calcASD(f,T,P_acc,P_ims,L)
    return ASD

def Sgal4yr(f):
    #4 year Galactic confusions noise parameters
    A = 9e-45
    a = 0.138
    b = -221
    k = 521
    g = 1680
    f_k = 0.00113
    return A*np.exp(-(f.value**a)+(b*f.value*np.sin(k*f.value)))*(f.value**(-7/3))*(1 + np.tanh(g*(f_k-f.value))) #White Dwarf Background Noise

def Get_CharStrain(Vars,f,h):
    [f,h] = StrainConv(Vars,f,h)
    h_char = np.sqrt(4*f**2*h**2)
    return f,h_char

def Get_MonoStrain(Vars,T_obs,f_init,fT):
    [M,q,_,_,z] = Vars
    DL = cosmo.luminosity_distance(z)
    DL = DL.to('m')

    m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]

    eta = q/(1+q)**2
    M_redshifted_time = M*(1+z)*m_conv
    M_chirp = eta**(3/5)*M_redshifted_time
    #Source is emitting at one frequency (monochromatic)
    #strain of instrument at f_cw
    indxfgw = np.abs(fT-f_init).argmin()
    #Strain from Cornish et. al 2018 (eqn 27) https://arxiv.org/pdf/1803.01944.pdf
    #(ie. optimally oriented)
    h_gw = 8/np.sqrt(5)*np.sqrt(T_obs)*(const.c/DL)*(np.pi*fT[indxfgw])**(2./3.)*M_chirp**(5./3.)
    return [indxfgw,h_gw]

def StrainConv(Vars,f,h):
    [M,q,_,_,z] = Vars
    DL = cosmo.luminosity_distance(z)
    DL = DL.to('m')

    m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
    M_redshifted_time = M*(1+z)*m_conv
    
    freq_conv = 1/M_redshifted_time
    strain_conv = np.sqrt(5/16/np.pi)*(const.c/DL)*M_redshifted_time**2
    
    f = f*freq_conv
    h = h*strain_conv

    return [f,h]

def Get_Waveform(Vars,nfreqs=int(1e3),f_low=1e-9):
    fit_coeffs_filedirectory = os.getcwd() + '/LoadFiles/PhenomDFiles/'
    fit_coeffs_filename = 'fitcoeffsWEB.dat'
    fit_coeffs_file = fit_coeffs_filedirectory + fit_coeffs_filename
    fitcoeffs = np.loadtxt(fit_coeffs_file) #load QNM fitting files for speed later

    [phenomD_f,phenomD_h] = PhenomD.FunPhenomD(Vars,fitcoeffs,nfreqs,f_low=f_low)
    return [phenomD_f,phenomD_h]

