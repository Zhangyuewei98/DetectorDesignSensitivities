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


def getSNRMatrix(source,instrument,var_x,sampleRate_x,var_y,sampleRate_y):
    # # Setting Up SNR Calculation
    # Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
    # selection of variables. Then it computes the SNR for each value.
    # Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample_yXsample_x
    # 

    #Get PhenomD waveform
    [phenomD_f,phenomD_h] = source.Get_PhenomD_Strain()
    source.Set_Instrument(instrument)

    #Get Samples for source variables (will return None if they arent var_x or var_y)
    [sample_x,sample_y,recalculate_strain] = Get_Samples(source,var_x,sampleRate_x,var_y,sampleRate_y)
    #Get instrument noise and frequency
    Model_Selection(instrument)

    #Check if either sample is not a source variable, if not use instrument samples
    #Very sloppy way of doing it...
    recalculate_noise = 'neither' #Don't need to update inst_var_dict
    if len(sample_x) == 0 and len(sample_y) == 0:
        recalculate_noise = 'both' #Need to update inst_var_dict on x-axis and y-axis
        [sample_x,sample_y,_] = Get_Samples(instrument,var_x,sampleRate_x,var_y,sampleRate_y)
    elif len(sample_x) == 0 and len(sample_y) != 0:
        recalculate_noise = 'x' #Need to update inst_var_dict on x-axis
        [sample_x,_,_] = Get_Samples(instrument,var_x,sampleRate_x,var_y,sampleRate_y)
    elif len(sample_x) != 0 and len(sample_y) == 0:
        recalculate_noise = 'y' #Need to update inst_var_dict on y-axis
        [_,sample_y,_] = Get_Samples(instrument,var_x,sampleRate_x,var_y,sampleRate_y)

    #Make sure samples have the correct astropy units
    [sample_x,sample_y] = Handle_Units(sample_x,var_x,sample_y,var_y)

    sampleSize_x = len(sample_x)
    sampleSize_y = len(sample_y)
    SNRMatrix = np.zeros((sampleSize_x,sampleSize_y))
    tmpSNRMatrix=np.zeros((sampleSize_x,sampleSize_y))
    
    for i in range(sampleSize_x):
        for j in range(sampleSize_y):

            if recalculate_noise == 'x':
                instrument.Update_Param_val(var_x,sample_x[i])
                source.Update_Param_val(var_y, sample_y[j])
            elif recalculate_noise == 'y':
                source.Update_Param_val(var_x,sample_x[j])
                instrument.Update_Param_val(var_y,sample_y[i])
            elif recalculate_noise == 'both':
                instrument.Update_Param_val(var_x,sample_x[i])
                instrument.Update_Param_val(var_y,sample_y[j])
            elif recalculate_noise == 'neither':
                source.Update_Param_val(var_x,sample_x[i])
                source.Update_Param_val(var_y,sample_y[j])
            
            if recalculate_noise != 'neither':
                #Recalculate noise curves if something is varied
                Model_Selection(instrument)
                source.Set_Instrument(instrument)

            source.checkFreqEvol()
            if source.ismono: #Monochromatic Source and not diff EOB SNR
                SNRMatrix[j,i] = calcMonoSNR(source,instrument)
            else: #Chirping Source
                if recalculate_strain == True: #If we need to calculate the waveform everytime
                    #Get PhenomD waveform
                    [phenomD_f,phenomD_h] = source.Get_PhenomD_Strain()
                source.StrainConv(phenomD_f,phenomD_h)
                SNRMatrix[j,i] = calcChirpSNR(source,instrument)

    return [sample_x,sample_y,SNRMatrix]

def Get_Samples(obj,var_x,sampleRate_x,var_y,sampleRate_y):
    ''' Takes in a object (either for the instrument or source), and the variables
        and sample rates desired in the SNR matrix. The function uses that to create a
        sample space for the variable either in linear space (for q,x1,or x2) or logspace
        for everything else.'''
    sample_x = []
    sample_y = []
    recalculate_strain = False

    var_x_dict = obj.Get_Param_Dict(var_x)
    var_y_dict = obj.Get_Param_Dict(var_y)

    if var_x_dict['min'] != None and var_x_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_x = np.linspace(var_x_dict['min'],var_x_dict['max'],sampleRate_x)
            recalculate_strain = True #Must recalculate the waveform at each point
        elif var_x == 'T_obs':
            #sample in linear space for instrument variables
            try:
                sample_x = np.linspace(var_x_dict['min'].value,var_x_dict['max'].value,sampleRate_x)
            except:
                sample_x = np.linspace(var_x_dict['min'],var_x_dict['max'],sampleRate_x)
        else:
            #Sample in log space for any other variables
            #Need exception for astropy variables
            try:
                sample_x = np.logspace(np.log10(var_x_dict['min'].value),np.log10(var_x_dict['max'].value),sampleRate_x)
            except:
                sample_x = np.logspace(np.log10(var_x_dict['min']),np.log10(var_x_dict['max']),sampleRate_x)
        
    if var_y_dict['min'] != None and var_y_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_y = np.linspace(var_y_dict['min'],var_y_dict['max'],sampleRate_y)
            recalculate_strain = True #Must recalculate the waveform at each point
        elif var_y == 'T_obs':
            #sample in linear space for instrument variables
            try:
                sample_y = np.linspace(var_y_dict['min'].value,var_y_dict['max'].value,sampleRate_y)
            except:
                sample_y = np.linspace(var_y_dict['min'],var_y_dict['max'],sampleRate_y)
        else:
            #Sample in log space for any other variables 
            #Need exception for astropy variables
            try:
                sample_y = np.logspace(np.log10(var_y_dict['min'].value),np.log10(var_y_dict['max'].value),sampleRate_y)
            except:
                sample_y = np.logspace(np.log10(var_y_dict['min']),np.log10(var_y_dict['max']),sampleRate_y)

    return sample_x,sample_y,recalculate_strain

def Model_Selection(instrument):
    '''Uses the instrument to calculate the frequency
        and amplitude spectral density corresponding to the detector's name in the
        dictionary
    '''
    inst_name = instrument.name

    if inst_name == 'ET': #Einstein Telescope
        load_name = 'ET_D_data.txt'
        load_location = load_directory + 'EinsteinTelescope/StrainFiles/' + load_name
        instrument.Get_ASD(load_location)

    elif inst_name == 'aLIGO': #aLIGO
        load_name = 'aLIGODesign.txt'
        load_location = load_directory + 'aLIGO/StrainFiles/' + load_name
        instrument.Get_ASD(load_location)

    else:
        instrument.Get_ASD()
    instrument.Set_f_opt()

def Handle_Units(sample_x,var_x,sample_y,var_y):
    '''Since I am using astropy units, I need to update units on the selected samples'''
    #Handle x variables
    if var_x == 'L' or var_x == 'A_IMS':
        sample_x = sample_x*u.m
    elif var_x == 'T_obs' or var_x == 'rms':
        sample_x = sample_x*u.s
    elif var_x == 'A_acc':
        sample_x = sample_x*u.m/u.s/u.s
    elif var_x == 'f_acc_break_high' or var_x == 'f_acc_break_low' or var_x == 'cadence':
        sample_x = sample_x/u.s
    #Handle y variables
    if var_y == 'L' or var_y == 'A_IMS':
        sample_y = sample_y*u.m
    elif var_y == 'T_obs' or var_y == 'rms':
        sample_y = sample_y*u.s
    elif var_y == 'A_acc':
        sample_y = sample_y*u.m/u.s/u.s
    elif var_y == 'f_acc_break_high' or var_y == 'f_acc_break_low' or var_y == 'cadence':
        sample_y = sample_y/u.s

    return sample_x,sample_y

def calcMonoSNR(source,instrument):
    #SNR for a monochromatic source in a PTA
    #From Moore,Taylor,and Gair 2015 https://arxiv.org/abs/1406.5199
    instrument.Get_Strain()

    if instrument.name == 'NANOGrav' or instrument.name == 'SKA':
        source.Get_MonoStrain(strain_const='Rosado')
    else:
        source.Get_MonoStrain(strain_const='Cornish')

    indxfgw = np.abs(instrument.fT-source.f_init).argmin()

    SNR = source.h_gw/instrument.h_n_f[indxfgw]
    return SNR

def calcChirpSNR(source,instrument):
    #Calculates evolving source using non-precessing binary black hole waveform model IMRPhenomD
    #See Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253)
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
    S_n_f = instrument.S_n_f_sqrt**2 #Amplitude Spectral Density

    #Only want to integrate from observed frequency (f(T_obs_before_merger)) till merger
    indxfgw = np.abs(source.f-source.f_T_obs).argmin()
    if indxfgw >= len(source.f)-1:
        #If the SMBH has already merged set the SNR to ~0
        return 1e-30  
    else:
        f_cut = source.f[indxfgw:]
        h_cut = source.h_f[indxfgw:]

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    S_n_f_interp_old = interp.interp1d(np.log10(instrument.fT.value),np.log10(S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = S_n_f_interp #Sky Averaged Noise Spectral Density
    numer = f_cut*h_cut**2

    integral_consts = 16/5 # 4 or(4*4/5) from sky/inclination/polarization averaging

    integrand = numer/denom
    SNRsqrd = integral_consts*np.trapz(integrand.value,np.log10(f_cut.value),axis=0) #SNR**2
    SNR = np.sqrt(SNRsqrd)
    return SNR

def calcDiffSNR(source_var_dict,fT,S_n_f_sqrt,diff_f,diff_h,f_init):
    #Calculates the SNR loss from the difference in EOB waveforms and numerical relativity.
    # The strain is from Sean McWilliams in a private communication.
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
    Vars = []
    for name,sub_dict in source_var_dict.items():
        Vars.append(source_var_dict[name]['val'])

    h_n_f = np.sqrt(fT)*S_n_f_sqrt #characteristic strain Amplitude

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
    h_n_f_interp_old = interp.interp1d(np.log10(fT.value),np.log10(h_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    h_n_f_interp_new = h_n_f_interp_old(np.log10(f_cut.value))
    h_n_f_interp = 10**h_n_f_interp_new

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = h_n_f_interp**2 #Sky Averaged Noise Spectral Density
    numer = h_cut**2

    integral_consts = 4 # 4or(4*4/5) from sky/inclination/polarization averaging

    integrand = numer/denom

    SNRsqrd = integral_consts*np.trapz(integrand.value,np.log10(f_cut.value),axis=0) #SNR**2
    SNR = np.sqrt(SNRsqrd)
    return SNR

def plotSNR(source_var_dict,inst_var_dict,var_x,sample_x,var_y,sample_y,SNRMatrix,display=True,dl_axis=False,figloc=None):
    '''Plots the SNR contours from calcSNR'''
    #Selects contour levels to separate sections into
    contLevels = np.array([5,10, 1e2, 1e3, 1e4, 1e5, 1e6])
    #contLevels = np.array([1,10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,1e8,1e9])
    #contLevels = np.array([1,10, 1e2, 1e3, 1e4])
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

    #Can't take log of astropy variables
    try:
        tmp_sample_x = np.log10(sample_x)
    except:
        sample_x = sample_x.value
    try:
        tmp_sample_y = np.log10(sample_y)
    except:
        sample_y = sample_y.value

    #Set whether log or linearly spaced axes
    if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
        xaxis_type = 'lin'
    else:
        xaxis_type = 'log'

    if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
        yaxis_type = 'lin'
    else:
        yaxis_type = 'log'

    #########################
    #Make the Contour Plots
    fig, ax = plt.subplots(figsize=figsize)

    #Set axis scales based on what data sampling we used 
    if yaxis_type == 'lin' and xaxis_type == 'log':
        CS1 = ax.contourf(np.log10(sample_x),sample_y,logSNR,logLevels,cmap = colormap)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.linspace(ylabel_min,ylabel_max,ylabel_max-ylabel_min+1)
        ax.set_yticks(y_labels)
        ax.set_xticks(np.log10(x_labels))
    elif yaxis_type == 'log' and xaxis_type == 'lin':
        CS1 = ax.contourf(sample_x,np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax.set_xlim(xlabel_min,xlabel_max)
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.linspace(xlabel_min,xlabel_max,xlabel_max-xlabel_min+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_xticks(x_labels)
        ax.set_yticks(np.log10(y_labels))
    else:
        CS1 = ax.contourf(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_yticks(np.log10(y_labels))
        ax.set_xticks(np.log10(x_labels))

    #Set axes labels and whether log or linearly spaced
    if var_x == 'M':
        ax.set_xlabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],fontsize = axissize)
    elif var_x == 'q':
        ax.set_xlabel(r'$q$',fontsize = labelsize)
        ax.set_xticklabels(x_labels,fontsize = axissize)
    elif var_x == 'z':
        ax.set_xlabel(r'${\rm Redshift}$',fontsize = labelsize)
        ax.set_xticklabels([x if int(x) < 1 else int(x) for x in x_labels],fontsize = axissize)
    elif var_x == 'chi1' or var_x == 'chi2':
        ax.set_xlabel(r'${\rm Spin}$',fontsize = labelsize)
        ax.set_xticklabels(x_labels,fontsize = axissize)
    elif var_x == 'L':
        ax.set_xlabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],fontsize = axissize)

    if var_y == 'M':
        ax.set_ylabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        ax.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)],fontsize = axissize)
    elif var_y == 'q':
        ax.set_ylabel(r'$q$',fontsize = labelsize)
        ax.set_yticklabels(y_labels,fontsize = axissize)
    elif var_y == 'z':
        ax.set_ylabel(r'${\rm Redshift}$',fontsize = labelsize)
        ax.set_yticklabels([y if int(y) < 1 else int(y) for y in y_labels],\
            fontsize = axissize)
    elif var_y == 'chi1' or var_x == 'chi2':
        ax.set_ylabel(r'${\rm Spin}$',fontsize = labelsize)
        ax.set_yticklabels(y_labels,fontsize = axissize)
    elif var_y == 'L':
        ax.set_ylabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        ax.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)],fontsize = axissize)

    ax.yaxis.set_label_coords(-.10,.5)

    #If true, display luminosity distance on right side of plot
    if dl_axis:
        dists = np.array([1e-1,1,10,1e2,1e3,1e4])*u.Gpc
        #dists = np.array([1e-1,1,10,1e2])*u.Gpc 
        distticks = [z_at_value(cosmo.luminosity_distance,dist) for dist in dists]
        #Set other side y-axis for lookback time scalings
        ax2 = ax.twinx()
        ax2.contour(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,colors = 'k',alpha=0.0)
        ax2.set_yticks(np.log10(distticks))
        #ax2.set_yticklabels(['%f' %dist for dist in distticks],fontsize = axissize)
        ax2.set_yticklabels([r'$10^{%i}$' %np.log10(dist) if np.abs(int(np.log10(dist))) > 1 else '{:g}'.format(dist) for dist in dists.value],fontsize = axissize)
        ax2.set_ylabel(r'$D_{L}$ [Gpc]',fontsize=labelsize)
        cbar = fig.colorbar(CS1,ax=(ax,ax2),pad=0.01)
    else:
        #########################
        #Make colorbar
        cbar = fig.colorbar(CS1)
    cbar.set_label(r'$SNR$',fontsize = labelsize)
    cbar.ax.tick_params(labelsize = axissize)
    cbar.ax.set_yticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in logLevels])

    if display:
        plt.show()

    #########################
    #Save Figure to File
    if figloc != None:
        fig.savefig(figloc,bbox_inches='tight')

def saveSNR(sample_x,sample_y,SNRMatrix,save_location,SNR_filename,sample_filename):
    #Save SNR Matrix
    np.savetxt(save_location+SNR_filename,SNRMatrix)
    #Save Samples
    np.savetxt(save_location+sample_filename,[sample_x,sample_y])