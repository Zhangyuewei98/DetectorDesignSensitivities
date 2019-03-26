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

# Adding Astropy units

# # Setting Up SNR Calculation
# Uses the index given and the data range to sample the space either logrithmically or linearly based on the 
# selection of variables. Then it computes the SNR for each value.
# Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample1Xsample2
# 

# # Check frequency evolution

def checkFreqEvol(Vars,T_obs,f_init):
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
    
    [M,q,_,_,z] = np.split(Vars,len(Vars))
    m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
    
    eta = q/(1+q)**2
    M_redshifted_time = M*(1+z)*m_conv
    M_chirp = eta**(3/5)*M_redshifted_time
    
    #f(t) from eqn 24 of Cornish et. al 2018 https://arxiv.org/pdf/1803.01944.pdf
    t_init = 5*(M_chirp)**(-5/3)*(8*np.pi*f_init)**(-8/3)
    f_evolve = 1./8./np.pi/M_chirp*(5*M_chirp/(t_init-T_obs))**(3./8.)
    f_T_obs = 1./8./np.pi/M_chirp*(5*M_chirp/(T_obs))**(3./8.)
    
    if (f_evolve-f_init) < (1/T_obs) and (t_init-T_obs) > 0:
        return f_init, True
    else:
        return f_T_obs, False


# ## getSNRMatrixVer8
# Mixed initial observing frequencies and time from merger


def getSNRMatrixVer8(Var1index,sampleRate1,Var2index,sampleRate2,Vars,VarLims,fT,S_n_f_sqrt,T_obs):
    recalculate = False #Assume we only need to calculate the waveform once

    [phenomD_f,phenomD_h] = SnN.Get_Waveform(Vars)
    
    f_opt = fT[np.argmin(S_n_f_sqrt)]

    if Var1index == 1 or Var1index == 2 or Var1index == 3:
        #Sample in linear space for mass ratio and spins
        sample1 = np.linspace(VarLims[Var1index,0],VarLims[Var1index,1],sampleRate1)
        recalculate = True #Must recalculate the waveform at each point
    else:
        #Sample in log space for any other variables
        sample1 = np.logspace(np.log10(VarLims[Var1index,0]),np.log10(VarLims[Var1index,1]),sampleRate1)

    if Var2index == 1 or Var2index == 2 or Var2index == 3:
        #Sample in linear space for mass ratio and spins
        sample2 = np.linspace(VarLims[Var2index,0],VarLims[Var2index,1],sampleRate2)
        recalculate = True #Must recalculate the waveform at each point
    else:
        #Sample in log space for any other variables
        sample2 = np.logspace(np.log10(VarLims[Var2index,0]),np.log10(VarLims[Var2index,1]),sampleRate2)

    sampleSize1 = len(sample1)
    sampleSize2 = len(sample2)
    SNRMatrix = np.zeros((sampleSize1,sampleSize2))

    for i in range(sampleSize1):
        for j in range(sampleSize2):
            Vars[Var1index] = sample1[i]
            Vars[Var2index] = sample2[j]
            #if ismono f_init=f_opt, else f_init=f_T_obs
            f_init, ismono = checkFreqEvol(Vars,T_obs,f_opt)
            SNRMatrix[i,j] = calcSNRVer11(Vars,phenomD_f,phenomD_h,fT,S_n_f_sqrt,T_obs,f_init,recalculate,ismono)
    return [sample1,sample2,SNRMatrix]


# # Calculation of SNR
# Attempts to replicate FuturePTA calculation with monochromatic vs
# chirping, but something is wrong

# # calcSNRver11


def calcSNRVer11(Vars,phenomD_f,phenomD_h,fT,S_n_f_sqrt,T_obs,f_init,recalculate,ismono):
    
    
    S_n_f = S_n_f_sqrt**2 #Amplitude Spectral Density
    
    if ismono:
        indxfgw,h_gw = SnN.Get_MonoStrain(Vars,fT,f_init,T_obs)
        #CALCULATE SNR
        SNR = np.sqrt(h_gw**2/S_n_f[indxfgw])[0]
    else:
        #Source is either evolving during the observation (f_T_obs-f_obs)>(1/T_obs),
        #or Completely Merges during the observation,
        #or already merged (Tobs > tinit => imaginary f_T_obs)  
        if recalculate:
            [phenomD_f,phenomD_h] = SnN.Get_Waveform(Vars)
        
        phenomD_f,phenomD_h = SnN.StrainConv(Vars,phenomD_f,phenomD_h)
        #Only want to integrate from observed frequency till merger
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


# # Plot SNR


def plotSNR(sample1,sample2,SNRMatrix,contLevels,VarLims,VarNames,index1,index2,figsize):
    '''Plots the SNR contours from calcSNR'''
    colormap = 'viridis'
    logLevels = np.log10(contLevels)
    logSNR = np.log10(SNRMatrix)
    xlabel_min = VarLims[index2,0]
    xlabel_max = VarLims[index2,1]
    ylabel_min = VarLims[index1,0]
    ylabel_max = VarLims[index1,1]

    #########################
    #Make the Contour Plots
    fig1, ax1 = plt.subplots(figsize=figsize)

    #Set axis scales based on what data sampling we used 
    if (index1 == 1 or index1 == 2 or index1 == 3) and (index2 != 1 and index2 != 2 and index2 != 3):
        CS1 = ax1.contourf(np.log10(sample2),sample1,logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.range(ylabel_min,ylabel_max,1)
        ax1.set_yticks(y_labels)
        ax1.set_yticklabels(y_labels,fontsize = axissize)
        ax1.set_xticks(np.log10(x_labels))
        ax1.set_xticklabels(np.log10(x_labels),fontsize = axissize)
    elif (index1 != 1 or index1 != 2 or index1 != 3) and (index2 == 1 and index2 == 2 and index2 == 3):
        CS1 = ax1.contourf(sample2,np.log10(sample1),logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(xlabel_min,xlabel_max)
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.range(xlabel_min,xlabel_max,1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(x_labels,fontsize = axissize)
        ax1.set_yticks(np.log10(y_labels))
        ax1.set_yticklabels(np.log10(y_labels),fontsize = axissize)
    elif (index1 != 1 or index1 != 2 or index1 != 3) and (index2 != 1 and index2 != 2 and index2 != 3):
        CS1 = ax1.contourf(np.log10(sample2),np.log10(sample1),logSNR,logLevels,cmap = colormap)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_yticks(np.log10(y_labels))
        ax1.set_yticklabels(np.log10(y_labels),fontsize = axissize)
        ax1.set_xticks(np.log10(x_labels))
        ax1.set_xticklabels(np.log10(x_labels),fontsize = axissize)

    ax1.set_xlabel(r'log $M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
    ax1.set_ylabel(r'$z$',fontsize = labelsize)
    ax1.yaxis.set_label_coords(-.10,.5)
    #########################
    #Make colorbar
    cbar1 = fig1.colorbar(CS1)
    cbar1.set_label(r'${\rm log}_ 10(SNR)$',fontsize = labelsize)
    cbar1.ax.tick_params(labelsize = axissize)

    plt.show()

