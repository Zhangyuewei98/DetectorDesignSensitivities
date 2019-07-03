import os, time, sys, struct

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.image import NonUniformImage
import matplotlib.ticker as ticker
from matplotlib import cm

import scipy.interpolate as interp
import scipy.integrate as integrate

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

from fractions import Fraction

import StrainandNoise as SnN

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('DetectorDesignSensitivities')
top_directory = "/".join(splt_path[0:top_path_idx+1])
load_directory = top_directory + '/LoadFiles/InstrumentFiles/'

def checkFreqEvol(source_var_dict,T_obs,f_init):
    #####################################
    #If the initial observed time from merger is less than the time observed
    #(ie t_init-T_obs < 0 => f_evolve is complex),
    #the BBH will or has already merged during the observation

    #If the initial observed time from merger is greater than the time observed
    #(ie t_init-T_obs > 0 => f_evolve is real),
    #And if the frequency of the binary does evolve over more than one bin,
    #(ie f_T_obs-f_init < 1/T_obs), it is monochromatic, so we set the frequency
    #to the optimal frequency of the detector

    #Otherwise it is chirping and evolves over the observation and we
    #set the starting frequency we observe it at to f(Tobs), which is the 
    #frequency at an observation time before merger
    #####################################
    
    M = source_var_dict['M']['val']
    q = source_var_dict['q']['val']
    z = source_var_dict['z']['val']
    m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
    
    eta = q/(1+q)**2
    M_redshifted_time = M*(1+z)*m_conv
    M_chirp = eta**(3/5)*M_redshifted_time
    
    #f(t) from eqn 24 of Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    t_init = 5*(M_chirp)**(-5/3)*(8*np.pi*f_init)**(-8/3)
    f_evolve = 1./8./np.pi/M_chirp*(5*M_chirp/(t_init-T_obs))**(3./8.)
    f_T_obs = 1./8./np.pi/M_chirp*(5*M_chirp/(T_obs))**(3./8.)
    
    if (f_evolve-f_init) < (1/T_obs) and (t_init-T_obs) > 0:
        return f_init, True
    else:
        return f_T_obs, False

def getSNRMatrix(source_var_dict,inst_var_dict,var_x,sampleRate_x,var_y,sampleRate_y,diff_model,Background=False):
    # # Setting Up SNR Calculation
    # Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
    # selection of variables. Then it computes the SNR for each value.
    # Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample_yXsample_x
    # 

    #Get PhenomD waveform
    initVars = []
    for name in source_var_dict.keys():
        initVars.append(source_var_dict[name]['val'])
    [phenomD_f,phenomD_h] = SnN.Get_Waveform(initVars)

    for inst_name_tmp,inst_dict_tmp in inst_var_dict.items():
        inst_name = inst_name_tmp
        inst_dict = inst_dict_tmp

    #Get Samples for source variables (will return None if they arent var_x or var_y)
    [sample_x,sample_y,recalculate_strain] = Get_Samples(source_var_dict,var_x,sampleRate_x,var_y,sampleRate_y)
    #Get instrument noise and frequency
    [fT,S_n_f_sqrt] = Model_Selection(inst_var_dict,Background)

    if diff_model <= 4:
        if diff_model == 0:
            diff_name = 'diff0002'
        elif diff_model == 1:
            diff_name = 'diff0114'
        elif diff_model == 2:
            diff_name = 'diff0178'
        elif diff_model == 3:
            diff_name = 'diff0261'
        elif diff_model == 4:
            diff_name = 'diff0303'
        diff_filename = diff_name + '.dat'
        diff_filelocation = top_directory + '/LoadFiles/DiffStrain/EOBdiff/' + diff_filename
        diff_data = np.loadtxt(diff_filelocation)
        diff_t = diff_data[:,0]*u.s
        diff_hp = diff_data[:,1]
        diff_hc = diff_data[:,2] 
        [diff_f,diff_h_f] = SnN.Get_hf_from_hcross_hplus(diff_t,diff_hc,diff_hp)

    #Check if either sample is not a source variable, if not use instrument samples
    #Very sloppy way of doing it...
    recalculate_noise = 'neither' #Don't need to update inst_var_dict
    try: 
        if sample_x == None:
            recalculate_noise = 'x' #Need to update inst_var_dict on x-axis
            [sample_x,_,_] = Get_Samples(inst_dict,var_x,sampleRate_x,var_y,sampleRate_y)
    except:
        pass
    try:
        if sample_y == None:
            recalculate_noise = 'y' #Need to update inst_var_dict on y-axis
            [_,sample_y,_] = Get_Samples(inst_dict,var_x,sampleRate_x,var_y,sampleRate_y)
    except:
        pass
    try:
        if sample_x == None and sample_y == None:
            recalculate_noise = 'both' #Need to update inst_var_dict on x-axis and y-axis
            [sample_x,sample_y,_] = Get_Samples(inst_dict,var_x,sampleRate_x,var_y,sampleRate_y)
    except:
        pass   

    #Make sure samples have the correct astropy units
    [sample_x,sample_y] = Handle_Units(sample_x,var_x,sample_y,var_y)

    #Get optimal (highest sensitivity) frequency
    f_opt = fT[np.argmin(S_n_f_sqrt)]

    sampleSize_x = len(sample_x)
    sampleSize_y = len(sample_y)
    SNRMatrix = np.zeros((sampleSize_x,sampleSize_y))
    tmpSNRMatrix=np.zeros((sampleSize_x,sampleSize_y))
    
    for i in range(sampleSize_x):
        for j in range(sampleSize_y):

            if recalculate_noise == 'x':
                inst_var_dict[inst_name][var_x]['val'] = sample_x[i]
                source_var_dict[var_y]['val'] = sample_y[j]
            elif recalculate_noise == 'y':
                source_var_dict[var_x]['val'] = sample_x[i]
                inst_var_dict[inst_name][var_y]['val'] = sample_y[j]
            elif recalculate_noise == 'both':
                inst_var_dict[inst_name][var_x]['val'] = sample_x[i]
                inst_var_dict[inst_name][var_y]['val'] = sample_y[j]
            elif recalculate_noise == 'neither':
                source_var_dict[var_x]['val'] = sample_x[i]
                source_var_dict[var_y]['val'] = sample_y[j]
            
            if recalculate_noise != 'neither':
                #Recalculate noise curves if something is varied
                if inst_name.split('_')[0] == 'LISA' and (i==0 and j==0): #Don't reload Transfer Function every time
                    reload_data = True
                    Instrument_data = SnN.Load_TransferFunction()
                elif inst_name == 'ET' and (i==0 and j==0): #Load ET data once
                    load_name = 'ET_D_data.txt'
                    load_location = load_directory + 'EinsteinTelescope/StrainFiles/' + load_name
                    Instrument_data = np.loadtxt(load_location)
                    reload_data = False
                elif inst_name == 'aLIGO' and (i==0 and j==0): # Load aLIGO data once
                    load_name = 'aLIGODesign.txt'
                    load_location = load_directory + 'aLIGO/StrainFiles/' + load_name
                    Instrument_data = np.loadtxt(load_location)
                    reload_data = False
                elif inst_name.split('_')[0] == 'LISA' or inst_name == 'ET' or inst_name == 'aLIGO' and (i!=0 or j!=0): #Already loaded Transfer Function
                    reload_data = False
                else: # Doesn't have a transfer function
                    Instrument_data = None
                    reload_data = False
                [fT,S_n_f_sqrt] = Model_Selection(inst_var_dict,Background,reload_data=reload_data,I_data=Instrument_data)
            if recalculate_noise != 'both' or (recalculate_noise == 'neither' and i==0 and j==0) or var_x == 'Tobs' or var_y == 'Tobs':
                '''Only recalulate strain if necessary (otherwise leave it at the original values)
                    If it is the first iteration, calculate strain
                    If one of the varibles are Tobs, need to recalculate'''
                T_obs = inst_var_dict[inst_name]['Tobs']['val']
                #if ismono f_init=f_opt, else f_init=f_T_obs
                f_init, ismono = checkFreqEvol(source_var_dict,T_obs,f_opt)
                if ismono and model != 6:
                    if inst_name == 'NANOGrav' or inst_name == 'SKA': #Use PTA calculation
                        SNRMatrix[j,i] = calcPTAMonoSNR(source_var_dict,inst_var_dict,f_init)
                    else:
                        SNRMatrix[j,i] = calcMonoSNR(source_var_dict,fT,S_n_f_sqrt,T_obs,f_init)
                elif diff_model <= 4: # Model for the diff EOB waveform/SNR calculation
                    SNRMatrix[j,i] = calcDiffSNR(source_var_dict,fT,S_n_f_sqrt,f_init,diff_f,diff_h_f)
                else:
                    if recalculate_strain == True: #If we need to calculate the waveform everytime
                        newVars = []
                        for name in source_var_dict.keys():
                            newVars.append(source_var_dict[name]['val'])
                        [phenomD_f,phenomD_h] = SnN.Get_Waveform(newVars)
                    SNRMatrix[j,i] = calcChirpSNR(source_var_dict,fT,S_n_f_sqrt,f_init,phenomD_f,phenomD_h,recalculate_strain)

    return [sample_x,sample_y,SNRMatrix]

def Get_Samples(sup_dict,var_x,sampleRate_x,var_y,sampleRate_y):
    ''' Takes in a dictionary (either for the instrument or source), and the variables
        and sample rates desired in the SNR matrix. The function uses that to create a
        sample space for the variable either in linear space (for q,x1,or x2) or logspace
        for everything else.'''
    sample_x = None
    sample_y = None
    recalculate_strain = False

    for var_name,var_dict in sup_dict.items():
        if var_name == var_x:
            if len(var_dict) == 3: #If the variable has 'val','min',and 'max' dictionary attributes
                if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2' or var_x == 'Tobs':
                    #Sample in linear space for mass ratio and spins
                    sample_x = np.linspace(sup_dict[var_x]['min'],sup_dict[var_x]['max'],sampleRate_x)
                    recalculate_strain = True #Must recalculate the waveform at each point
                else:
                    #Sample in log space for any other variables
                    #Need exception for astropy variables
                    try:
                        sample_x = np.logspace(np.log10(sup_dict[var_x]['min'].value),np.log10(sup_dict[var_x]['max'].value),sampleRate_x)
                    except:
                        sample_x = np.logspace(np.log10(sup_dict[var_x]['min']),np.log10(sup_dict[var_x]['max']),sampleRate_x)
            print('x var: ',var_name)
        if var_name == var_y:
            if len(var_dict) == 3: #If the variable has 'val','min',and 'max' dictionary attributes
                if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
                    #Sample in linear space for mass ratio and spins
                    sample_y = np.linspace(sup_dict[var_y]['min'],sup_dict[var_y]['max'],sampleRate_y)
                    recalculate_strain = True #Must recalculate the waveform at each point
                elif var_y == 'Tobs':
                    #sample in linear space
                    try:
                        sample_y = np.linspace(sup_dict[var_y]['min'].value,sup_dict[var_y]['max'].value,sampleRate_y)
                    except:
                        sample_y = np.linspace(sup_dict[var_y]['min'],sup_dict[var_y]['max'],sampleRate_y)
                else:
                    #Sample in log space for any other variables 
                    #Need exception for astropy variables
                    try:
                        sample_y = np.logspace(np.log10(sup_dict[var_y]['min'].value),np.log10(sup_dict[var_y]['max'].value),sampleRate_y)
                    except:
                        sample_y = np.logspace(np.log10(sup_dict[var_y]['min']),np.log10(sup_dict[var_y]['max']),sampleRate_y)
            print('y var: ',var_name)
    return sample_x,sample_y,recalculate_strain

def Model_Selection(inst_var_dict,Background,reload_data=True,I_data=None):
    '''Uses inst_var_dict (the instrument dictionary) to calculate the frequency
        and amplitude spectral density corresponding to the detector's name in the
        dictionary
    '''

    #Get the name of the instrument
    if len(inst_var_dict) == 1:
        inst_name = list(inst_var_dict.keys())[0]
    elif len(inst_var_dict) == 2:
        #Do something else (mostly for PTAs)
        inst_name = 'SKA'
        print('Nothing Yet.')

    if inst_name == 'LISA_Neil': #Robson,Cornish,and Liu 2018, LISA (https://arxiv.org/pdf/1803.01944.pdf)
        fT,S_n_f_sqrt = SnN.NeilSensitivity(inst_var_dict,Background=Background,reload_T=reload_data,T_data=I_data)
        S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)
        
    elif inst_name == 'LISA_Martin': #Martin 2016: LISA Calculation without pathfinder correction (2016 model)
        fT,S_n_f_sqrt = SnN.MartinSensitivity(inst_var_dict,Background=Background,reload_T=reload_data,T_data=I_data)
        S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)
        
    elif inst_name == 'ET': #Einstein Telescope
        if reload_data == True:
            load_name = 'ET_D_data.txt'
            load_location = load_directory + 'EinsteinTelescope/StrainFiles/' + load_name
            Instrument_data = np.loadtxt(load_location)
        fT = I_data[:,0]*u.Hz
        S_n_f_sqrt = I_data[:,1]
        S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)
        
    elif inst_name == 'aLIGO': #aLIGO
        if reload_data == True:
            load_name = 'aLIGODesign.txt'
            load_location = load_directory + 'aLIGO/StrainFiles/' + load_name
            I_data = np.loadtxt(load_location)
        fT = I_data[:,0]*u.Hz
        S_n_f_sqrt = I_data[:,1]
        S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)
        
    elif inst_name == 'NANOGrav': #NANOGrav 15 yr
        ###############
        #NEED TO FIX BACKGROUND HARD CODING
        ###############
        fT,S_n_f_sqrt = SnN.Get_PTAASD_v2(inst_var_dict,A_stoch_back=0.0)
        
    elif inst_name == 'SKA': #SKA (2030s)
        ###############
        #NEED TO FIX BACKGROUND HARD CODING
        ###############
        fT,S_n_f_sqrt = SnN.Get_PTAASD_v2(inst_var_dict,A_stoch_back=0.0)
        
    elif inst_name == 'LISA_ESA': #L3 proposal
        ###############
        #NEED TO FIX BACKGROUND HARD CODING
        ###############
        fT,S_n_f_sqrt = SnN.LisaSensitivity(inst_var_dict,Background=Background,reload_T=reload_data,T_data=I_data)
        S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)
    else:
        print('Whoops, not the right name!')

    return fT,S_n_f_sqrt

def Handle_Units(sample_x,var_x,sample_y,var_y):
    '''Since I am using astropy units, I need to update units on the selected samples'''
    #Handle x variables
    if var_x == 'L' or var_x == 'S_oms' or var_x == 'S_sci' or var_x == 'S_loc' or var_x == 'S_other' or var_x == 'S_ims':
        sample_x = sample_x*u.m
    elif var_x == 'Tobs' or var_x == 'rms':
        sample_x = sample_x*u.s
    elif var_x == 'S_acc' or var_x == 'S_acc_low' or var_x == 'S_acc_high' or var_x == 'S_oms_knee':
        sample_x = sample_x*u.m/u.s/u.s
    elif var_x == 'S_acc_low_knee' or var_x == 'S_acc_high_knee' or var_x == 'cadence':
        sample_x = sample_x/u.s
    #Handle y variables
    if var_y == 'L' or var_y == 'S_oms' or var_y == 'S_sci' or var_y == 'S_loc' or var_y == 'S_other' or var_y == 'S_ims':
        sample_y = sample_y*u.m
    elif var_y == 'Tobs' or var_y == 'rms':
        sample_y = sample_y*u.s
    elif var_y == 'S_acc' or var_y == 'S_acc_low' or var_y == 'S_acc_high' or var_y == 'S_oms_knee':
        sample_y = sample_y*u.m/u.s/u.s
    elif var_y == 'S_acc_low_knee' or var_y == 'S_acc_high_knee' or var_y == 'cadence':
        sample_y = sample_y/u.s

    return sample_x,sample_y

def calcPTAMonoSNR(source_var_dict,inst_var_dict,f_init):
    #SNR for a monochromatic source in a PTA
    #From Moore,Taylor,and Gair 2015 https://arxiv.org/abs/1406.5199
    source_vars = []
    for name,sub_dict in source_var_dict.items():
        source_vars.append(source_var_dict[name]['val'])

    T_obs = 0.0
    for pta_name, pta_dict in inst_var_dict.items():
        for var_name,var_dict in pta_dict.items():
            if var_name == 'Tobs':
                T_obs += var_dict['val']

    f, h_c_inst = SnN.Get_PTAstrain(inst_var_dict)

    indxfgw,h_c_source = SnN.Get_MonoStrain(source_vars,T_obs,f_init,f)

    SNR = h_c_source/h_c_inst[indxfgw]
    return SNR

def calcMonoSNR(source_var_dict,fT,S_n_f_sqrt,T_obs,f_init):
    #Calculation of monochromatic source strain
    #See ~pg. 9 in Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    Vars = []
    for name,sub_dict in source_var_dict.items():
        Vars.append(source_var_dict[name]['val'])

    S_n_f = S_n_f_sqrt**2 #Amplitude Spectral Density
    indxfgw,h_gw = SnN.Get_MonoStrain(Vars,T_obs,f_init,fT,strain_const='Cornish')
    #CALCULATE SNR
    #Eqn. 26
    SNR = np.sqrt(h_gw**2/S_n_f[indxfgw])
    return SNR

def calcChirpSNR(source_var_dict,fT,S_n_f_sqrt,f_init,phenomD_f,phenomD_h,recalculate):
    #Calculates evolving source using non-precessing binary black hole waveform model IMRPhenomD
    #See Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253)
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
    Vars = []
    for name,sub_dict in source_var_dict.items():
        Vars.append(source_var_dict[name]['val'])

    S_n_f = S_n_f_sqrt**2 #Amplitude Spectral Density

    if recalculate:
        [phenomD_f,phenomD_h] = SnN.Get_Waveform(Vars)
    
    phenomD_f,phenomD_h = SnN.StrainConv(Vars,phenomD_f,phenomD_h)
    #Only want to integrate from observed frequency (f(T_obs_before_merger)) till merger
    indxfgw = np.abs(phenomD_f-f_init).argmin()
    if indxfgw >= len(phenomD_f)-1:
        #If the SMBH has already merged set the SNR to ~0
        return 1e-30  
    else:
        f_cut = phenomD_f[indxfgw:]
        h_cut = phenomD_h[indxfgw:]

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    S_n_f_interp_old = interp.interp1d(np.log10(fT.value),np.log10(S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = S_n_f_interp #Sky Averaged Noise Spectral Density
    numer = f_cut*h_cut**2

    integral_consts = 16/5 # 4 or(4*4/5) from sky/inclination/polarization averaging

    integrand = numer/denom
    SNRsqrd = integral_consts*np.trapz(integrand.value,np.log(f_cut.value),axis=0) #SNR**2
    SNR = np.sqrt(SNRsqrd)
    return SNR

def calcDiffSNR(source_var_dict,fT,S_n_f_sqrt,f_init,diff_f,diff_h):
    #Calculates the SNR loss from the difference in EOB waveforms and numerical relativity.
    # The strain is from Sean McWilliams in a private communication.
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
    Vars = []
    for name,sub_dict in source_var_dict.items():
        Vars.append(source_var_dict[name]['val'])

    S_n_f = S_n_f_sqrt**2 #Amplitude Spectral Density
    
    diff_f,diff_h = SnN.StrainConv(Vars,diff_f,diff_h)
    #Only want to integrate from observed frequency (f(T_obs_before_merger)) till merger
    indxfgw = np.abs(diff_f-f_init).argmin()
    if indxfgw >= len(diff_f)-1:
        #If the SMBH has already merged set the SNR to ~0
        return 1e-30  
    else:
        f_cut = diff_f[indxfgw:]
        h_cut = diff_h[indxfgw:]

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    S_n_f_interp_old = interp.interp1d(np.log10(fT.value),np.log10(S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = S_n_f_interp #Sky Averaged Noise Spectral Density
    numer = f_cut*h_cut**2

    integral_consts = 16/5 # 4 or(4*4/5) from sky/inclination/polarization averaging

    integrand = numer/denom
    SNRsqrd = integral_consts*np.trapz(integrand.value,np.log(f_cut.value),axis=0) #SNR**2
    SNR = np.sqrt(SNRsqrd)
    return SNR

def plotSNR(source_var_dict,inst_var_dict,var_x,sample_x,var_y,sample_y,SNRMatrix):
    '''Plots the SNR contours from calcSNR'''
    #Selects contour levels to separate sections into
    contLevels = np.array([5,10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    logLevels = np.log10(contLevels)
    axissize = 14
    labelsize = 16
    textsize = 11
    textcolor = 'k'
    linesize = 4
    figsize = (10,8)

    contourcolorPresent = 'plasma'
    transparencyPresent = 1.0
    contourcolorFuture = 'plasma'
    transparencyFuture = 1.0
    colornorm = colors.Normalize(vmin=0.0, vmax=5.0)
    colormap = 'viridis'
    logSNR = np.log10(SNRMatrix)

    inst_name = list(inst_var_dict.keys())[0]
    try:
        xlabel_min = source_var_dict[var_x]['min']
        xlabel_max = source_var_dict[var_x]['max']
    except:
        xlabel_min = inst_var_dict[inst_name][var_x]['min'].value
        xlabel_max = inst_var_dict[inst_name][var_x]['max'].value
    try:
        ylabel_min = source_var_dict[var_y]['min']
        ylabel_max = source_var_dict[var_y]['max']
    except:
        ylabel_min = inst_var_dict[inst_name][var_y]['min'].value
        ylabel_max = inst_var_dict[inst_name][var_y]['max'].value

    #Can't take log of astropy united variables
    try:
        tmp_sample_x = np.log10(sample_x)
    except:
        sample_x = sample_x.value
    try:
        tmp_sample_y = np.log10(sample_y)
    except:
        sample_y = sample_y.value

    #########################
    #Make the Contour Plots
    fig, ax = plt.subplots(figsize=figsize)

    #Set axis scales based on what data sampling we used 
    if (var_y == 'q' or var_y == 'chi1' or var_y == 'chi2') and (var_x != 'q' and var_x != 'chi1' and var_x != 'chi2'):
        CS1 = ax.contourf(np.log10(sample_x),sample_y,logSNR,logLevels,cmap = colormap)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.range(ylabel_min,ylabel_max,1)
        ax.set_yticks(y_labels)
        ax.set_yticklabels(y_labels,fontsize = axissize)
        ax.set_xticks(np.log10(x_labels))
        ax.set_xticklabels(np.log10(x_labels),fontsize = axissize)
    elif (var_y != 'q' or var_y != 'chi1'  or var_y != 'chi2' ) and (var_x == 'q' and var_x == 'chi1' and var_x == 'chi2'):
        CS1 = ax.contourf(sample_x,np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax.set_xlim(xlabel_min,xlabel_max)
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.range(xlabel_min,xlabel_max,1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels,fontsize = axissize)
        ax.set_yticks(np.log10(y_labels))
        ax.set_yticklabels(np.log10(y_labels),fontsize = axissize)
    else:
        CS1 = ax.contourf(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_yticks(np.log10(y_labels))
        ax.set_xticks(np.log10(x_labels))
        ax.set_yticklabels([x if int(x) < 1 else int(x) for x in y_labels],\
            fontsize = axissize)
        ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],\
            fontsize = axissize)

    ax.set_xlabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
    ax.set_ylabel(r'${\rm Redshift}$',fontsize = labelsize)
    ax.yaxis.set_label_coords(-.10,.5)
    #########################
    #Make colorbar
    cbar = fig.colorbar(CS1)
    cbar.set_label(r'$SNR$',fontsize = labelsize)
    cbar.ax.tick_params(labelsize = axissize)
    cbar.ax.set_yticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in logLevels])

    plt.show()

def saveSNR(sample_x,sample_y,SNRMatrix,save_location,SNR_filename,sample_filename):
    #Save SNR Matrix
    np.savetxt(save_location+SNR_filename,SNRMatrix)
    #Save Samples
    np.savetxt(save_location+sample_filename,[sample_x,sample_y])