import numpy as np
import os
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

import IMRPhenomD as PhenomD

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('DetectorDesignSensitivities')
top_directory = "/".join(splt_path[0:top_path_idx+1])

def Get_PTAASD(inst_var_dict,nfreqs=int(1e3),A_stoch_back=4e-16):
    f,PSD = Get_PTAPSD(inst_var_dict,nfreqs=nfreqs,A_stoch_back=A_stoch_back)
    ASD = np.sqrt((12*np.pi**2)*f**2*PSD)
    return f,ASD

def Get_PTAASD_v2(inst_var_dict,nfreqs=int(1e3),A_stoch_back=4e-16):
    f,h_c = Get_PTAstrain(inst_var_dict,nfreqs=nfreqs)
    #from Jenet et al. 2006 https://arxiv.org/abs/astro-ph/0609013 (Only for GWB/broadband signals)
    P_w = h_c**2/12/np.pi**2*f**(-3)

    #Paragraph below eqn 4 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    #P_red = A_red.*(f./f_year).**(-gamma) # red noise for some pulsar
    P_red = 0.0*u.s*u.s/u.Hz #Assume no pulsar red noise for simplicity
    
    #eqn 42 of T&R 2013 https://arxiv.org/abs/1310.5300
    #h_inst = np.sqrt((12*np.pi**2*(P_w+P_red))*f**3)
    ##################################################
    #Stochastic background amplitude from Sesana et al. 2016 https://arxiv.org/pdf/1603.09348.pdf
    f_year = 1/u.yr
    P_sb = (A_stoch_back**2/12/np.pi**2)*f**(-3)*(f/f_year.to('Hz'))**(-4/3)
    #h_sb = A_stoch_back*(f/f_year)**(-2/3)

    ####################################################
    #PSD of the full PTA from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    P_n = P_w + P_red + P_sb
    
    #####################################################
    #PSD of a PTA taken from eqn 40 of Thrane & Romano 2013 https://arxiv.org/abs/1310.5300

    #strain of the full PTA
    #h_f = np.sqrt((12*np.pi**2)*f**3*P_n)
    ASD = np.sqrt((12*np.pi**2)*f**2*P_n)

    return f,ASD

def Get_PTAPSD(inst_var_dict,nfreqs=int(1e3),A_stoch_back=4e-16):
    T_obs = []
    sigma_rms = []
    N_p = []
    cadence = []
    ndetectors = 0
    #Unpacking dictionary
    for pta_name, pta_dict in inst_var_dict.items():
        ndetectors += 1
        for var_name,var_dict in pta_dict.items():
            if var_name == 'Tobs':
                T_obs.append(var_dict['val'])
            elif var_name == 'rms':
                sigma_rms.append(var_dict['val'])
            elif var_name == 'Np':
                N_p.append(var_dict['val'])
            elif var_name == 'cadence':
                cadence.append(var_dict['val'])

    #Equation 5 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    P_w_tot = 0
    T_obs_tot = 0
    #Sum of pulsar noises in different time periods divided by total time
    for i in range(ndetectors):
        P_w_n = sigma_rms[i]**2*(T_obs[i]/cadence[i])/N_p[i]/(N_p[i]-1) #Avg white noise from pulsar arrays [s**2/Hz]
        P_w_tot += P_w_n
        T_obs_tot += T_obs[i]
    P_w = 2*P_w_tot/T_obs_tot
    #frequency sampled from 1/observation time to nyquist frequency (c/2)
    f = np.logspace(np.log10(1/T_obs_tot.value),np.log10(min(cadence).value/2),nfreqs)*u.Hz
    
    f_year = 1/u.yr
    #Paragraph below eqn 4 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    #P_red = A_red.*(f./f_year).**(-gamma) # red noise for some pulsar
    P_red = 0.0*u.s*u.s/u.Hz #Assume no pulsar red noise for simplicity
    
    #eqn 42 of T&R 2013 https://arxiv.org/abs/1310.5300
    #h_inst = np.sqrt((12*np.pi**2*(P_w+P_red))*f**3)
    ##################################################
    #Stochastic background amplitude from Sesana et al. 2016 https://arxiv.org/pdf/1603.09348.pdf
    P_sb = (A_stoch_back**2/12/np.pi**2)*f**(-3)*(f/f_year.to('Hz'))**(-4/3)
    #h_sb = A_stoch_back*(f/f_year)**(-2/3)

    ####################################################
    #PSD of the full PTA from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    PSD = P_w + P_red + P_sb

    #strain of the full PTA
    #h_f = np.sqrt((12*np.pi**2)*f**3*P_n)
    #ASD = np.sqrt((12*np.pi**2)*f**2*P_n)
    return f,PSD


def Get_PTAstrain(inst_var_dict,nfreqs=int(1e3)):
    # Taken from Moore,Taylor, and Gair 2014 https://arxiv.org/abs/1406.5199

    T_obs = []
    sigma_rms = []
    N_p = []
    cadence = []
    ndetectors = 0
    #Unpacking dictionary
    for pta_name, pta_dict in inst_var_dict.items():
        ndetectors += 1
        for var_name,var_dict in pta_dict.items():
            if var_name == 'Tobs':
                T_obs.append(var_dict['val'])
            elif var_name == 'rms':
                sigma_rms.append(var_dict['val'])
            elif var_name == 'Np':
                N_p.append(var_dict['val'])
            elif var_name == 'cadence':
                cadence.append(var_dict['val'])

    #Equation 5 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
    P_w_tot = 0
    T_obs_tot = 0
    #Sum of pulsar noises in different time periods divided by total time
    for i in range(ndetectors):
        P_w_n = sigma_rms[i]**2*(T_obs[i]/cadence[i]) #Avg white noise from pulsar arrays [s**2/Hz]
        P_w_tot += P_w_n
        T_obs_tot += T_obs[i]
    P_w = 2*P_w_tot/T_obs_tot
    #frequency sampled from 1/observation time to nyquist frequency (c/2)
    f = np.logspace(np.log10(1/T_obs_tot.value),np.log10(min(cadence).value/2),nfreqs)*u.Hz

    SNR = 1.0 # Value of 3 is Used in paper
    chi_corr = 1/np.sqrt(3) #Sky averaged geometric factor eqn. 11
    overlap_freq = 2/T_obs_tot
    N_p = max(N_p) #Just use largest number of pulsars from all ptas

    #h_c_high_factor = (16*SNR**2/(3*chi_corr**4*N_p*(N_p-1)))**(1/4)*sigma_rms*np.sqrt(1/T_obs_tot/cadence)
    #h_c_low_factor = 3*np.sqrt(SNR)/(2**(7/4)*chi_corr*np.pi**3)*(13/N_p/(N_p-1))**(1/4)*sigma_rms*np.sqrt(1/T_obs_tot/cadence)*T_obs_tot**(-3)
    h_c_high_factor = (16*SNR**2/(3*chi_corr**4*N_p*(N_p-1)))**(1/4)*np.sqrt(P_w/2/T_obs_tot)
    h_c_low_factor = 3*np.sqrt(SNR)/(2**(7/4)*chi_corr*np.pi**3)*(13/N_p/(N_p-1))**(1/4)*np.sqrt(P_w/2/T_obs_tot)*T_obs_tot**(-3)

    phi = np.arccos(h_c_low_factor/h_c_high_factor*overlap_freq**(-3))

    h_c_high = h_c_high_factor*f
    h_c_low = h_c_low_factor*f**(-2)*(1/np.cos(phi))

    h_c = h_c_low + h_c_high #Below eqn 16, should it be added in quad?
    return f,h_c

def Get_ASD_from_PSD_LISA(f,T,P_acc,P_ims,L,Norm=20/3,Background=True):
    #Calculates amplitude spectral density
    PSD = Norm/T**2*(4*P_acc+P_ims)/L**2 #Strain Noise Power Spectral Density
    if Background:
        return np.sqrt(PSD+Sgal4yr(f)) #Sqrt of PSD
    else:
        return np.sqrt(PSD)

def Get_approxResponseFunction(f,L):
    #Response function approximation from Calculation described by Neil Cornish (unpublished)
    f_L = const.c/2/np.pi/L #Transfer frequency
    R_f = 3/10/(1+0.6*(f/f_L)**2) 
    return R_f

def Get_TransferFunction(L,f_low=1e-5*u.Hz,f_high=1.0*u.Hz):
    LISA_Transfer_Function_filedirectory = top_directory + '/LoadFiles/LISATransferFunction/'
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

def NeilSensitivity(inst_var_dict,Background=True):
    for var_name, var_dict in inst_var_dict.items():
        if var_name == 'L':
            L = var_dict['val']
        elif var_name == 'S_acc':
            S_acc = var_dict['val']
        elif var_name == 'S_oms':
            S_oms = var_dict['val']

    f,T = Get_TransferFunction(L)

    #Uses Calculation described by Neil Cornish (unpublished)
    f_L = const.c/2/np.pi/L #Transfer frequency
    P_acc = (S_acc)**2*(1+(0.4e-3*u.Hz/f)**2)*(1+(f/(8e-3*u.Hz))**4) #acceleration noise [Hz]**-1

    P_acc_term = 2*(1.0+(np.cos(f.value/f_L.value))**2)*P_acc/(2*np.pi*f)**4
    P_acc_term = P_acc_term/4 #adjust so we can use get ASD function

    P_oms = (S_oms)**2*(1+(2e-3*u.Hz/f)**4) #Optical metrology noise [Hz]**-1
    
    #P_n_f = P_oms/L**2 + 2*(1.0+(np.cos(f.value/f_L.value))**2)*P_acc/(2*np.pi*f)**4/L**2 #Total noise
    #R_f = approxResponseFunction(f,L) #Response function approximation
    #ASD = np.sqrt(P_n_f/R_f+Sgal4yr(f)) #Amplitude Spectral Density approx
    #ASD = np.sqrt(10/3/T**2*P_n_f+Sgal4yr(f))
    ASD = Get_ASD_from_PSD_LISA(f,T,P_acc_term,P_oms,L,Norm=10/3,Background=Background)
    return f,ASD

def LisaSensitivity(inst_var_dict,Background=True):
    for var_name, var_dict in inst_var_dict.items():
        if var_name == 'L':
            L = var_dict['val']
        elif var_name == 'S_oms_knee':
            S_oms_knee = var_dict['val']
        elif var_name == 'S_acc_low_knee':
            S_acc_low_knee = var_dict['val']
        elif var_name == 'S_acc_high_knee':
            S_acc_high_knee = var_dict['val']
        elif var_name == 'S_acc':
            S_acc = var_dict['val']
        elif var_name == 'S_ims':
            S_ims = var_dict['val']
    f,T = Get_TransferFunction(L)

    P_acc = S_acc**2*(1+(S_acc_low_knee/f)**2)*(1+(f/(S_acc_high_knee))**4)/(2*np.pi*f)**4 #Acceleration Noise 
    P_ims = S_ims**2*(1+(S_oms_knee/f)**4) #Displacement noise of the interferometric TM--to-TM 
    ASD = Get_ASD_from_PSD_LISA(f,T,P_acc,P_ims,L,Background=Background)
    return f,ASD

def MartinSensitivity(inst_var_dict,Background=True):
    for var_name, var_dict in inst_var_dict.items():
        if var_name == 'L':
            L = var_dict['val']
        elif var_name == 'S_sci':
            S_sci = var_dict['val']
        elif var_name == 'S_loc':
            S_loc = var_dict['val']
        elif var_name == 'S_other':
            S_other = var_dict['val']
        elif var_name == 'S_acc_low_knee':
            S_acc_low_knee = var_dict['val']
        elif var_name == 'S_acc_high_knee':
            S_acc_high_knee = var_dict['val']
        elif var_name == 'S_acc_low':
            S_acc_low = var_dict['val']
        elif var_name == 'S_acc_high':
            S_acc_high = var_dict['val']

    f,T = Get_TransferFunction(L)

    P_ims = S_sci**2+2*S_loc**2+S_other**2
    P_acc = ((S_acc_low)**2*((S_acc_low_knee/f)**10 + (S_acc_high_knee/f)**2) + (S_acc_high)**2)/(2*np.pi*f)**4   #red below 1e-4, white above
    ASD = Get_ASD_from_PSD_LISA(f,T,P_acc,P_ims,L,Background=Background)
    return f,ASD

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

def Get_MonoStrain_v2(Vars,T_obs,f_init,fT):
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
    #Strain from Rosado, Sesana, and Gair (2015) https://arxiv.org/abs/1503.04803
    #(ie. sky and inclination averaged)
    inc = 0 #optimally oriented
    a = 1+np.cos(inc)**2
    b = -2*np.cos(inc)
    A = 2*(const.c/DL)*(np.pi*fT[indxfgw])**(2./3.)*M_chirp**(5./3.)
    #h_gw = A*np.sqrt(.5*(a**2+b**2))
    h_gw = A*3
    return [indxfgw,h_gw]

def StrainConv(Vars,f,h):
    [M,q,_,_,z] = Vars
    DL = cosmo.luminosity_distance(z)
    DL = DL.to('m')

    m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
    M_redshifted_time = M*(1+z)*m_conv
    
    freq_conv = 1/M_redshifted_time
    #Normalized factor?
    #Changed from sqrt(5/16/pi)
    strain_conv = np.sqrt(1/4/np.pi)*(const.c/DL)*M_redshifted_time**2
    
    f = f*freq_conv
    h = h*strain_conv

    return [f,h]

def Get_Waveform(Vars,nfreqs=int(1e3),f_low=1e-9):
    fit_coeffs_filedirectory = top_directory + '/LoadFiles/PhenomDFiles/'
    fit_coeffs_filename = 'fitcoeffsWEB.dat'
    fit_coeffs_file = fit_coeffs_filedirectory + fit_coeffs_filename
    fitcoeffs = np.loadtxt(fit_coeffs_file) #load QNM fitting files for speed later

    [phenomD_f,phenomD_h] = PhenomD.FunPhenomD(Vars,fitcoeffs,nfreqs,f_low=f_low)
    return [phenomD_f,phenomD_h]

