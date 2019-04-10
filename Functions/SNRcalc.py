import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.image import NonUniformImage
import matplotlib.ticker as ticker
from matplotlib import cm
import os, time, sys
import struct
import scipy.interpolate as interp
import scipy.integrate as integrate
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo
from fractions import Fraction

import StrainandNoise as SnN


def checkFreqEvol(var_dict,T_obs,f_init):
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
    
    M = var_dict['M']['val']
    q = var_dict['q']['val']
    z = var_dict['z']['val']
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

def getSNRMatrix(var_dict,fT,S_n_f_sqrt,T_obs,var_x,sampleRate_x,var_y,sampleRate_y):
    # # Setting Up SNR Calculation
    # Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
    # selection of variables. Then it computes the SNR for each value.
    # Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample_yXsample_x
    # 
    f_opt = fT[np.argmin(S_n_f_sqrt)]

    if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
        #Sample in linear space for mass ratio and spins
        sample_y = np.linspace(var_dict[var_y]['min'],var_dict[var_y]['max'],sampleRate_y)
        recalculate = True #Must recalculate the waveform at each point
    else:
        #Sample in log space for any other variables
        sample_y = np.logspace(np.log10(var_dict[var_y]['min']),np.log10(var_dict[var_y]['max']),sampleRate_y)

    if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
        #Sample in linear space for mass ratio and spins
        sample_x = np.linspace(var_dict[var_x]['min'],var_dict[var_x]['max'],sampleRate_x)
        recalculate = True #Must recalculate the waveform at each point
    else:
        #Sample in log space for any other variables
        sample_x = np.logspace(np.log10(var_dict[var_x]['min']),np.log10(var_dict[var_x]['max']),sampleRate_x)

    sampleSize_x = len(sample_x)
    sampleSize_y = len(sample_y)
    SNRMatrix = np.zeros((sampleSize_x,sampleSize_y))
    
    for i in range(sampleSize_x):
        for j in range(sampleSize_y):
            var_dict[var_x]['val'] = sample_x[i]
            var_dict[var_y]['val'] = sample_y[j]
            #if ismono f_init=f_opt, else f_init=f_T_obs
            f_init, ismono = checkFreqEvol(var_dict,T_obs,f_opt)
            if ismono:
                SNRMatrix[j,i] = calcMonoSNR(var_dict,fT,S_n_f_sqrt,T_obs,f_init)
            else:
                recalculate = False #Assume we only need to calculate the waveform once
                initVars = []
                for name in var_dict.keys():
                    initVars.append(var_dict[name]['val'])
                [phenomD_f,phenomD_h] = SnN.Get_Waveform(initVars)
                SNRMatrix[j,i] = calcChirpSNR(var_dict,fT,S_n_f_sqrt,T_obs,f_init,phenomD_f,phenomD_h,recalculate)
    return [sample_x,sample_y,SNRMatrix]


def calcMonoSNR(var_dict,fT,S_n_f_sqrt,T_obs,f_init):
    #Calculation of monochromatic source strain
    #See ~pg. 9 in Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    Vars = []
    for name,sub_dict in var_dict.items():
        Vars.append(var_dict[name]['val'])

    S_n_f = S_n_f_sqrt**2 #Amplitude Spectral Density
    indxfgw,h_gw = SnN.Get_MonoStrain(Vars,T_obs,f_init,fT)
    #CALCULATE SNR
    #Eqn. 26
    SNR = np.sqrt(h_gw**2/S_n_f[indxfgw])
    return SNR

def calcChirpSNR(var_dict,fT,S_n_f_sqrt,T_obs,f_init,phenomD_f,phenomD_h,recalculate):
    #Calculates evolving source using non-precessing binary black hole waveform model IMRPhenomD
    #See Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253)
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
    Vars = []
    for name,sub_dict in var_dict.items():
        Vars.append(var_dict[name]['val'])

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


def plotSNR(var_dict,var_x,sample_x,var_y,sample_y,SNRMatrix):
    '''Plots the SNR contours from calcSNR'''
    #Selects contour levels to separate sections into
    contLevels = np.array([5,10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
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
    logLevels = np.log10(contLevels)
    logSNR = np.log10(SNRMatrix)
    xlabel_min = var_dict[var_x]['min']
    xlabel_max = var_dict[var_x]['max']
    ylabel_min = var_dict[var_y]['min']
    ylabel_max = var_dict[var_y]['max']

    #########################
    #Make the Contour Plots
    fig1, ax1 = plt.subplots(figsize=figsize)

    #Set axis scales based on what data sampling we used 
    if (var_y == 'q' or var_y == 'chi1' or var_y == 'chi2') and (var_x != 'q' and var_x != 'chi1' and var_x != 'chi2'):
        CS1 = ax1.contourf(np.log10(sample_x),sample_y,logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.range(ylabel_min,ylabel_max,1)
        ax1.set_yticks(y_labels)
        ax1.set_yticklabels(y_labels,fontsize = axissize)
        ax1.set_xticks(np.log10(x_labels))
        ax1.set_xticklabels(np.log10(x_labels),fontsize = axissize)
    elif (var_y != 'q' or var_y != 'chi1'  or var_y != 'chi2' ) and (var_x == 'q' and var_x == 'chi1' and var_x == 'chi2'):
        CS1 = ax1.contourf(sample_x,np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(xlabel_min,xlabel_max)
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.range(xlabel_min,xlabel_max,1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(x_labels,fontsize = axissize)
        ax1.set_yticks(np.log10(y_labels))
        ax1.set_yticklabels(np.log10(y_labels),fontsize = axissize)
    elif (var_y != 'q' or var_y != 'chi1' or var_y != 'chi2') and (var_x != 'q' and var_x != 'chi1' and var_x != 'chi2'):
        CS1 = ax1.contourf(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_yticks(np.log10(y_labels))
        ax1.set_yticklabels(np.log10(y_labels),fontsize = axissize)
        ax1.set_xticks(np.log10(x_labels))
        ax1.set_xticklabels(np.log10(x_labels),fontsize = axissize)

    ax1.set_xlabel(r'${\rm log}(M_{\rm tot})$ $[M_{\odot}]$',fontsize = labelsize)
    ax1.set_ylabel(r'${\rm log}(z)$',fontsize = labelsize)
    ax1.yaxis.set_label_coords(-.10,.5)
    #########################
    #Make colorbar
    cbar1 = fig1.colorbar(CS1)
    cbar1.set_label(r'${\rm log}(SNR)$',fontsize = labelsize)
    cbar1.ax.tick_params(labelsize = axissize)

    plt.show()

def saveSNR(sample_x,sample_y,SNRMatrix,save_location,SNR_filename,sample_filename):
    #Save SNR Matrix
    np.savetxt(save_location+SNR_filename,SNRMatrix)
    #Save Samples
    np.savetxt(save_location+sample_filename,[sample_x,sample_y])