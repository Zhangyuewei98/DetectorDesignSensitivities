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

import StrainandNoise_v3 as SnN

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

    if not hasattr(source,'instrument'):
        source.instrument = instrument
    #Get Samples for variables
    [sample_x,sample_y,recalculate_strain,recalculate_noise] = Get_Samples(source,instrument,var_x,sampleRate_x,var_y,sampleRate_y)

    #Make sure samples have the correct astropy units
    [sample_x,sample_y] = Handle_Units(sample_x,var_x,sample_y,var_y)

    sampleSize_x = len(sample_x)
    sampleSize_y = len(sample_y)
    SNRMatrix = np.zeros((sampleSize_x,sampleSize_y))
    tmpSNRMatrix=np.zeros((sampleSize_x,sampleSize_y))
    
    for i in range(sampleSize_x):
        for j in range(sampleSize_y):

            if recalculate_noise == 'x':
                #Update Attribute (also updates dictionary)
                setattr(instrument,var_x,sample_x[i])
                setattr(source,var_y, sample_y[j])
            elif recalculate_noise == 'y':
                #Update Attribute (also updates dictionary)
                setattr(source,var_x,sample_x[i])
                setattr(instrument,var_y, sample_y[j])
            elif recalculate_noise == 'both':
                #Update Attribute (also updates dictionary)
                setattr(instrument,var_x,sample_x[i])
                setattr(instrument,var_y, sample_y[j])
            elif recalculate_noise == 'neither':
                #Update Attribute (also updates dictionary)
                setattr(source,var_x,sample_x[i])
                setattr(source,var_y, sample_y[j])
            
            if recalculate_noise != 'neither':
                #Recalculate noise curves if something is varied
                if hasattr(instrument,'fT'):
                    del instrument.fT
                if hasattr(instrument,'P_n_f'):
                    del instrument.P_n_f
                if hasattr(instrument,'S_n_f'):
                    del instrument.S_n_f
                if hasattr(instrument,'h_n_f'):
                    del instrument.h_n_f
                source.instrument = instrument

            source.checkFreqEvol()
            if source.ismono: #Monochromatic Source and not diff EOB SNR
                if hasattr(source,'h_gw'):
                    del source.h_gw
                SNRMatrix[j,i] = calcMonoSNR(source,instrument)
            else: #Chirping Source
                if recalculate_strain == True: #If we need to calculate the waveform everytime
                    #Delete old PhenomD waveform
                    if hasattr(source,'_phenomD_f'):
                        del source._phenomD_f
                    if hasattr(source,'_phenomD_h'):
                        del source._phenomD_h
                if hasattr(source,'f'):
                    del source.f
                if hasattr(source,'h_f'):
                    del source.h_f
                SNRMatrix[j,i] = calcChirpSNR(source,instrument)

    return [sample_x,sample_y,SNRMatrix]

def Get_Samples(source,instrument,var_x,sampleRate_x,var_y,sampleRate_y):
    ''' Takes in a object (either for the instrument or source), and the variables
        and sample rates desired in the SNR matrix. The function uses that to create a
        sample space for the variable either in linear space (for q,x1,or x2) or logspace
        for everything else.'''
    sample_x = []
    sample_y = []
    recalculate_strain = False
    recalculate_noise = 'neither'

    if var_x in source.var_dict.keys():
        var_x_dict = source.var_dict[var_x]
    elif var_x in instrument.var_dict.keys():
        recalculate_noise = 'x'
        var_x_dict = instrument.var_dict[var_x]
    else:
        raise ValueError(var_x + ' is not a variable in the source or the instrument.')
    
    if var_y in source.var_dict.keys():
        var_y_dict = source.var_dict[var_y]
    elif var_y in instrument.var_dict.keys():
        if recalculate_noise == 'x':
            recalculate_noise = 'both'
        else:
            recalculate_noise = 'y'
        var_y_dict = instrument.var_dict[var_y]
    else:
        raise ValueError(var_y + ' is not a variable in the source or the instrument.')

    if var_x_dict['min'] != None and var_x_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_x = np.linspace(var_x_dict['min'],var_x_dict['max'],sampleRate_x)
            recalculate_strain = True #Must recalculate the waveform at each point
        elif var_x == 'T_obs':
            #sample in linear space for instrument variables
            #Need exception for astropy variables
            if isinstance(var_x_dict['min'],u.Quantity) and isinstance(var_x_dict['max'],u.Quantity):
                T_obs_min = var_x_dict['min'].to('s')
                T_obs_max = var_x_dict['max'].to('s')
                sample_x = np.linspace(T_obs_min.value,T_obs_max.value,sampleRate_x)
            else:
                T_obs_min = var_x_dict['min']*u.yr.to('s')
                T_obs_max = var_x_dict['max']*u.yr.to('s')
                sample_x = np.linspace(T_obs_min,T_obs_max,sampleRate_x)
        else:
            #Sample in log space for any other variables
            #Need exception for astropy variables
            if isinstance(var_x_dict['min'],u.Quantity) and isinstance(var_x_dict['max'],u.Quantity):
                sample_x = np.logspace(np.log10(var_x_dict['min'].value),np.log10(var_x_dict['max'].value),sampleRate_x)
            else:
                sample_x = np.logspace(np.log10(var_x_dict['min']),np.log10(var_x_dict['max']),sampleRate_x)
        
    if var_y_dict['min'] != None and var_y_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_y = np.linspace(var_y_dict['min'],var_y_dict['max'],sampleRate_y)
            recalculate_strain = True #Must recalculate the waveform at each point
        elif var_y == 'T_obs':
            #sample in linear space for instrument variables
            #Need exception for astropy variables
            if isinstance(var_y_dict['min'],u.Quantity) and isinstance(var_y_dict['max'],u.Quantity):
                T_obs_min = var_y_dict['min'].to('s')
                T_obs_max = var_y_dict['max'].to('s')
                sample_y = np.linspace(T_obs_min.value,T_obs_max.value,sampleRate_y)
            else:
                T_obs_min = var_y_dict['min']*u.yr.to('s')
                T_obs_max = var_y_dict['max']*u.yr.to('s')
                sample_y = np.linspace(T_obs_min,T_obs_max,sampleRate_y)
        else:
            #Sample in log space for any other variables 
            #Need exception for astropy variables
            if isinstance(var_y_dict['min'],u.Quantity) and isinstance(var_y_dict['max'],u.Quantity):
                sample_y = np.logspace(np.log10(var_y_dict['min'].value),np.log10(var_y_dict['max'].value),sampleRate_y)
            else:
                sample_y = np.logspace(np.log10(var_y_dict['min']),np.log10(var_y_dict['max']),sampleRate_y)

    return sample_x,sample_y,recalculate_strain,recalculate_noise

def Handle_Units(sample_x,var_x,sample_y,var_y):
    '''Since I am using astropy units, I need to update units on the selected samples'''
    #Handle x variables
    if var_x == 'L' or var_x == 'A_IMS':
        sample_x = sample_x*u.m
    elif var_x == 'T_obs' or var_x == 'sigma':
        sample_x = sample_x*u.s
    elif var_x == 'A_acc':
        sample_x = sample_x*u.m/u.s/u.s
    elif var_x == 'f_acc_break_high' or var_x == 'f_acc_break_low' or var_x == 'cadence':
        sample_x = sample_x/u.s
    #Handle y variables
    if var_y == 'L' or var_y == 'A_IMS':
        sample_y = sample_y*u.m
    elif var_y == 'T_obs' or var_y == 'sigma':
        sample_y = sample_y*u.s
    elif var_y == 'A_acc':
        sample_y = sample_y*u.m/u.s/u.s
    elif var_y == 'f_acc_break_high' or var_y == 'f_acc_break_low' or var_y == 'cadence':
        sample_y = sample_y/u.s

    return sample_x,sample_y

def calcMonoSNR(source,instrument):
    #SNR for a monochromatic source in a PTA
    indxfgw = np.abs(instrument.fT-source.f_init).argmin()

    return source.h_gw/np.sqrt(instrument.S_n_f[indxfgw])

def calcChirpSNR(source,instrument):
    #Calculates evolving source using non-precessing binary black hole waveform model IMRPhenomD
    #See Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253)
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0

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
    S_n_f_interp_old = interp.interp1d(np.log10(instrument.fT.value),np.log10(instrument.S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    if isinstance(instrument,SnN.PTA):
        #CALCULATE SNR FOR BOTH NOISE CURVES
        denom = S_n_f_interp #Sky Averaged Noise Spectral Density
        numer = h_cut**2

        integral_consts = 4.

        integrand = numer/denom
        if isinstance(integrand,u.Quantity) and isinstance(f_cut,u.Quantity):
            SNRsqrd = integral_consts*np.trapz(integrand.value,f_cut.value,axis=0) #SNR**2
        else:
            SNRsqrd = integral_consts*np.trapz(integrand,f_cut,axis=0) #SNR**2

    elif isinstance(instrument,SnN.SpaceBased) or isinstance(instrument,SnN.GroundBased):
        #CALCULATE SNR FOR BOTH NOISE CURVES
        denom = S_n_f_interp #Sky Averaged Noise Spectral Density
        numer = h_cut**2

        integral_consts = 16./5.

        integrand = numer/denom
        if isinstance(integrand,u.Quantity) and isinstance(f_cut,u.Quantity):
            SNRsqrd = integral_consts*np.trapz(integrand.value,f_cut.value,axis=0) #SNR**2
        else:
            SNRsqrd = integral_consts*np.trapz(integrand,f_cut,axis=0) #SNR**2

    return np.sqrt(SNRsqrd)

def calcDiffSNR(source,instrument):
    #Calculates the SNR loss from the difference in EOB waveforms and numerical relativity.
    # The strain is from Sean McWilliams in a private communication.
    #Uses an interpolated method to align waveform and instrument noise, then integrates 
    # over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
    # Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    h_n_f_interp_old = interp.interp1d(np.log10(instrument.fT.value),np.log10(instrument.h_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    h_n_f_interp_new = h_n_f_interp_old(np.log10(source.f.value))
    h_n_f_interp = 10**h_n_f_interp_new

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = h_n_f_interp**2 #Sky Averaged Noise Spectral Density
    numer = self.h_f**2

    integral_consts = 4 # 4or(4*4/5) from sky/inclination/polarization averaging

    integrand = numer/denom

    SNRsqrd = integral_consts*np.trapz(integrand.value,np.log10(self.f.value),axis=0) #SNR**2
    SNR = np.sqrt(SNRsqrd)
    return SNR

def plotSNR(source,instrument,var_x,sample_x,var_y,sample_y,SNRMatrix,display=True,dl_axis=False,smooth_contours=False,figloc=None):
    '''Plots the SNR contours from calcSNR'''
    #Selects contour levels to separate sections into
    '''contLevels_max = round(max(SNRMatrix))
    contLevels = np.array([5,10, 1e2, 1e3, 1e4, 1e5, 1e6])
    #contLevels = np.array([1,10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,1e8,1e9])
    #contLevels = np.array([1,10, 1e2, 1e3, 1e4])
    logLevels = np.log10(contLevels)'''
    axissize = 14
    labelsize = 16
    textsize = 11
    textcolor = 'k'
    linesize = 4
    figsize = (10,8)

    colormap = 'viridis'
    logSNR = np.log10(SNRMatrix)

    logLevels_min = np.log10(np.array([5]))
    logLevels_max = np.ceil(np.amax(logSNR))
    print_logLevels = np.append(logLevels_min,np.arange(1,logLevels_max+1))
    if smooth_contours:
        logLevels = np.linspace(logLevels_min,logLevels_max,100)
    else:
        logLevels = print_logLevels


    if var_x in source.var_dict.keys():
        if isinstance(source.var_dict[var_x]['min'],u.Quantity):
            xlabel_min = source.var_dict[var_x]['min'].value
            xlabel_max = source.var_dict[var_x]['max'].value
        else:
            xlabel_min = source.var_dict[var_x]['min']
            xlabel_max = source.var_dict[var_x]['max']
    elif var_x in instrument.var_dict.keys():
        if isinstance(instrument.var_dict[var_x]['min'],u.Quantity):
            xlabel_min = instrument.var_dict[var_x]['min'].value
            xlabel_max = instrument.var_dict[var_x]['max'].value
        else:
            xlabel_min = instrument.var_dict[var_x]['min']
            xlabel_max = instrument.var_dict[var_x]['max']
        if var_x == 'T_obs':
            xlabel_min = xlabel_min*u.yr.to('s')
            xlabel_max = xlabel_max*u.yr.to('s')
    else:
        raise ValueError(var_x + ' is not a variable in the source or the instrument.')

    if var_y in source.var_dict.keys():
        if isinstance(source.var_dict[var_y]['min'],u.Quantity):
            ylabel_min = source.var_dict[var_y]['min'].value
            ylabel_max = source.var_dict[var_y]['max'].value
        else:
            ylabel_min = source.var_dict[var_y]['min']
            ylabel_max = source.var_dict[var_y]['max']
    elif var_y in instrument.var_dict.keys():
        if isinstance(instrument.var_dict[var_y]['min'],u.Quantity):
            ylabel_min = instrument.var_dict[var_y]['min'].value
            ylabel_max = instrument.var_dict[var_y]['max'].value
        else:
            ylabel_min = instrument.var_dict[var_y]['min']
            ylabel_max = instrument.var_dict[var_y]['max']
        if var_y == 'T_obs':
            ylabel_min = ylabel_min*u.yr.to('s')
            ylabel_max = ylabel_max*u.yr.to('s')
    else:
        raise ValueError(var_y + ' is not a variable in the source or the instrument.')

    #Can't take log of astropy variables
    if isinstance(sample_x,u.Quantity):
        sample_x = sample_x.value
    if isinstance(sample_y,u.Quantity):
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
    fig, ax1 = plt.subplots(figsize=figsize)
    #Set other side y-axis for lookback time scalings
    ax2 = ax1.twinx()

    #Set axis scales based on what data sampling we used 
    if yaxis_type == 'lin' and xaxis_type == 'log':
        CS1 = ax1.contourf(np.log10(sample_x),sample_y,logSNR,logLevels,cmap = colormap)
        ax2.contour(np.log10(sample_x),sample_y,logSNR,print_logLevels,colors = 'k',alpha=1.0)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.linspace(ylabel_min,ylabel_max,ylabel_max-ylabel_min+1)
        ax1.set_yticks(y_labels)
        ax1.set_xticks(np.log10(x_labels))
    elif yaxis_type == 'log' and xaxis_type == 'lin':
        CS1 = ax1.contourf(sample_x,np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax2.contour(sample_x,np.log10(sample_y),logSNR,print_logLevels,colors = 'k',alpha=1.0)
        ax1.set_xlim(xlabel_min,xlabel_max)
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.linspace(xlabel_min,xlabel_max,xlabel_max-xlabel_min+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_xticks(x_labels)
        ax1.set_yticks(np.log10(y_labels))
    else:
        CS1 = ax1.contourf(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,cmap = colormap)
        ax2.contour(np.log10(sample_x),np.log10(sample_y),logSNR,print_logLevels,colors = 'k',alpha=1.0)
        ax1.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax1.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax1.set_yticks(np.log10(y_labels))
        ax1.set_xticks(np.log10(x_labels))

    #Set axes labels and whether log or linearly spaced
    if var_x == 'M':
        ax1.set_xlabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        ax1.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],fontsize = axissize)
    elif var_x == 'q':
        ax1.set_xlabel(r'$q$',fontsize = labelsize)
        ax1.set_xticklabels(x_labels,fontsize = axissize)
    elif var_x == 'z':
        ax1.set_xlabel(r'${\rm Redshift}$',fontsize = labelsize)
        ax1.set_xticklabels([x if int(x) < 1 else int(x) for x in x_labels],fontsize = axissize)
    elif var_x == 'chi1' or var_x == 'chi2':
        ax1.set_xlabel(r'${\rm Spin}$',fontsize = labelsize)
        ax1.set_xticklabels(x_labels,fontsize = axissize)
    elif var_x == 'L':
        ax1.set_xlabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        ax1.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],fontsize = axissize)
    elif var_x == 'T_obs':
        ax1.set_xlabel(r'${\rm T_{obs}}$ $[yr]$',fontsize = labelsize)
        ax1.set_xticklabels([r'$%i$' %int(x) for x in x_labels/u.yr.to('s')],fontsize = axissize)

    if var_y == 'M':
        ax1.set_ylabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        ax1.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)],fontsize = axissize)
    elif var_y == 'q':
        ax1.set_ylabel(r'$q$',fontsize = labelsize)
        ax1.set_yticklabels(y_labels,fontsize = axissize)
    elif var_y == 'z':
        ax1.set_ylabel(r'${\rm Redshift}$',fontsize = labelsize)
        ax1.set_yticklabels([y if int(y) < 1 else int(y) for y in y_labels],\
            fontsize = axissize)
    elif var_y == 'chi1' or var_x == 'chi2':
        ax1.set_ylabel(r'${\rm Spin}$',fontsize = labelsize)
        ax1.set_yticklabels(y_labels,fontsize = axissize)
    elif var_y == 'L':
        ax1.set_ylabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        ax1.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)],fontsize = axissize)
    elif var_y == 'T_obs':
        ax1.set_ylabel(r'${\rm T_{obs}}$ $[yr]$',fontsize = labelsize)
        ax1.set_yticklabels([r'$%i$' %int(y) for y in y_labels/u.yr.to('s')],fontsize = axissize)

    ax1.yaxis.set_label_coords(-.10,.5)

    #If true, display luminosity distance on right side of plot
    if dl_axis:
        dists_min = cosmo.luminosity_distance(source.var_dict['z']['min']).to('Gpc')
        dists_min = np.ceil(np.log10(dists_min.value))
        dists_max = cosmo.luminosity_distance(source.var_dict['z']['max']).to('Gpc')
        dists_max = np.ceil(np.log10(dists_max.value))
        dists = np.arange(dists_min,dists_max)
        dists = 10**dists*u.Gpc

        distticks = [z_at_value(cosmo.luminosity_distance,dist) for dist in dists]
        #Set other side y-axis for lookback time scalings
        ax2.set_yticks(np.log10(distticks))
        #ax2.set_yticklabels(['%f' %dist for dist in distticks],fontsize = axissize)
        ax2.set_yticklabels([r'$10^{%i}$' %np.log10(dist) if np.abs(int(np.log10(dist))) > 1 else '{:g}'.format(dist) for dist in dists.value],fontsize = axissize)
        ax2.set_ylabel(r'$D_{L}$ [Gpc]',fontsize=labelsize)
        cbar = fig.colorbar(CS1,ax=(ax1,ax2),pad=0.01,ticks=print_logLevels)
    else:
        #########################
        #Make colorbar
        cbar = fig.colorbar(CS1,ticks=print_logLevels)
        #Remove y-axis labels
        ax2.tick_params(axis='y',right=False,labelright=False)
    cbar.set_label(r'$SNR$',fontsize = labelsize)
    cbar.ax.tick_params(labelsize = axissize)
    cbar.ax.set_yticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in print_logLevels])

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