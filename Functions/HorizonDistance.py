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

import StrainandNoise_v4 as SnN

def getHorizonDistance(source,instrument,var_x,sampleRate_x,rho_thresh,redshift_array=None):
    '''Setting Up Horizon Distance calculation
        Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
        selection of variables. 
        Then it computes the Horizon for each value at the particular SNR given by rho_thresh (aka the threshold SNR).
        It takes in a redshift_array corresponding to the output luminosity distances so the function can be used to
            iteratively converge to correct luminosity distance values.
        Returns the variable ranges used to calculate the horizon distance for each matrix and the horizon distance
    '''

    if not hasattr(source,'instrument'):
        source.instrument = instrument
    #Get Samples for variable
    [sample_x,recalculate_strain,recalculate_noise] = Get_Samples(source,instrument,var_x,sampleRate_x)

    sampleSize_x = len(sample_x)
    if not isinstance(redshift_array,np.ndarray):
        redshift_array = np.ones(sampleSize_x)*source.z
    DL_array = np.zeros(sampleSize_x)
    
    for i in range(sampleSize_x):

        if recalculate_noise == True:
            #Update Attribute (also updates dictionary)
            setattr(instrument,var_x,sample_x[i])
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
        elif recalculate_noise == False:
            #Update Attribute (also updates dictionary)
            setattr(source,var_x,sample_x[i])

        #Update particular source's redshift
        setattr(source,'z',redshift_array[i])

        source.checkFreqEvol()
        #print(source.ismono)
        if source.ismono: #Monochromatic Source and not diff EOB SNR
            DL_array[i] = calcMonoHD(source,instrument,rho_thresh)
        else: #Chirping Source
            if recalculate_strain == True: #If we need to calculate the waveform everytime
                #Delete old PhenomD waveform
                if hasattr(source,'_phenomD_f'):
                    del source._phenomD_f
                if hasattr(source,'_phenomD_h'):
                    del source._phenomD_h
            if hasattr(source,'f'):
                del source.f
            DL_array[i] = calcChirpHD(source,instrument,rho_thresh)

    DL_array = DL_array*u.m.to('Mpc')*u.Mpc

    new_redshift_array = np.zeros(np.shape(DL_array))
    for i,DL in enumerate(DL_array):
        if np.log10(DL.value) < -3:
            new_redshift_array[i] = -5
        else:
            new_redshift_array[i] = z_at_value(cosmo.luminosity_distance,DL)

    return [sample_x,DL_array,new_redshift_array]

def Get_Samples(source,instrument,var_x,sampleRate_x):
    ''' Takes in a object (either for the instrument or source), and the variables
        and sample rates desired to calculate the Horizon Distance. 
        The function uses that to create a
        sample space for the variable either in linear space (for q,x1,or x2) or logspace
        for everything else.'''
    sample_x = []
    recalculate_strain = False
    recalculate_noise = False

    if var_x in source.var_dict.keys():
        var_x_dict = source.var_dict[var_x]
    elif var_x in instrument.var_dict.keys():
        recalculate_noise = 'x'
        var_x_dict = instrument.var_dict[var_x]
    else:
        raise ValueError(var_x + ' is not a variable in the source or the instrument.')

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

    return sample_x,recalculate_strain,recalculate_noise

def calcMonoHD(source,instrument,rho_thresh):
    ''' Calculates the Horizon Distance for a monochromatic source at an SNR of rho_thresh'''

    m_conv = const.G/const.c**3 #Converts M = [M] to M = [sec]

    eta = source.q/(1+source.q)**2
    M_time = source.M.to('kg')*m_conv
    M_chirp = eta**(3/5)*M_time
    #Source is emitting at one frequency (monochromatic)
    #strain of instrument at f_cw
    if isinstance(instrument,SnN.PTA):
        const_val = 4.
    elif isinstance(instrument,SnN.SpaceBased) or isinstance(instrument,SnN.GroundBased):
        const_val = 8./np.sqrt(5)

    numer = const_val*np.sqrt(source.T_obs)*const.c*(np.pi*source.f_init)**(2./3.)*M_chirp**(5./3.)

    indxfgw = np.abs(instrument.fT-source.f_init).argmin()

    denom = rho_thresh*np.sqrt(instrument.S_n_f[indxfgw])

    DL = numer/denom

    return DL.value

def calcChirpHD(source,instrument,rho_thresh):
    '''Calculates the Horizon Distance for an evolving source using non-precessing binary black hole waveform model IMRPhenomD
                See Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253)
                Uses an interpolated method to align waveform and instrument noise, then integrates 
                over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 https://arxiv.org/abs/1803.01944
                Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0
                Uses a constant SNR of rho_thresh
    '''

    m_conv = const.G/const.c**3 #Converts M = [M] to M = [sec]
    M_time = source.M.to('kg')*m_conv
    M_redshifted_time = source.M.to('kg')*(1+source.z)*m_conv

    #Only want to integrate from observed frequency (f(T_obs_before_merger)) till merger
    source.f = source._phenomD_f/M_time
    indxfgw = np.abs(source.f-source.f_T_obs).argmin()
    if indxfgw >= len(source.f)-1:
        #If the SMBH has already merged set the SNR to ~0
        print('TOO LOW')
        return 1e-30  
    else:
        f_cut = source.f[indxfgw:]
        nat_h_cut = source._phenomD_h[indxfgw:]

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    S_n_f_interp_old = interp.interp1d(np.log10(instrument.fT.value),np.log10(instrument.S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False) 
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    if isinstance(instrument,SnN.PTA):
        integral_consts = 4.*1.5
    elif isinstance(instrument,SnN.SpaceBased) or isinstance(instrument,SnN.GroundBased):
        integral_consts = 16./5.

    integral_consts *= (const.c**2)/4./np.pi*M_time**4/rho_thresh**2
    integral_consts = integral_consts.value

    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = S_n_f_interp #Sky Averaged Noise Spectral Density
    numer = nat_h_cut**2

    integrand = numer/denom
    if isinstance(integrand,u.Quantity) and isinstance(f_cut,u.Quantity):
        DLsqrd = integral_consts*np.trapz(integrand.value,f_cut.value,axis=0) #DL**2
    elif not isinstance(integrand,u.Quantity) and isinstance(f_cut,u.Quantity):
        DLsqrd = integral_consts*np.trapz(integrand,f_cut.value,axis=0) #DL**2
    else:
        DLsqrd = integral_consts*np.trapz(integrand,f_cut,axis=0) #DL**2

    return np.sqrt(DLsqrd)


def plotHD(source,instrument,var_x,sample_x,DL_array,display=True,figloc=None):
    '''Plots the DL curves from calcHorizonDistance'''

    axissize = 16
    labelsize = 18
    textsize = 12
    textcolor = 'k'
    linesize = 4
    figsize = (10,8)


    #Can't take log of astropy variables
    if isinstance(sample_x,u.Quantity):
        sample_x = sample_x.value
    if isinstance(DL_array,u.Quantity):
        DL_array = DL_array.value

    #Set whether log or linearly spaced axes
    if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
        xaxis_type = 'lin'
    else:
        xaxis_type = 'log'

    #########################
    #Make the Contour Plots
    plt.figure(figsize=figsize)

    #Set axis scales based on what data sampling we used 
    if xaxis_type == 'log':
        x_label_min = np.min(np.floor(np.log10(sample_x)))
        x_label_max = np.max(np.ceil(np.log10(sample_x)))
        plt.loglog(sample_x,DL_array)
        plt.xlim([10**x_label_min,10**x_label_max])
        x_labels = np.arange(x_label_min,x_label_max+1)
    elif xaxis_type == 'lin':
        x_label_min = np.min(np.floor(sample_x))
        x_label_max = np.max(np.ceil(sample_x))
        plt.semilogy(sample_x,DL_array)
        plt.xlim([x_label_min,x_label_max])
        x_labels = np.linspace(x_label_min,x_label_max,x_label_max-x_label_min+1)

    #Set axes labels and whether log or linearly spaced
    if var_x == 'M':
        plt.xlabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        plt.xticks(10**x_labels,[r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in x_labels],fontsize = axissize)
    elif var_x == 'q':
        plt.xlabel(r'$q$',fontsize = labelsize)
        plt.xticks(x_labels,fontsize = axissize,rotation=45)
    elif var_x == 'chi1' or var_x == 'chi2':
        x_labels = np.arange(round(x_label_min*10),round(x_label_max*10)+1,1)/10
        plt.xticks(x_labels)
        plt.xlabel(r'${\rm Spin}$',fontsize = labelsize)
        plt.xticks(10**x_labels,fontsize = axissize,rotation=45)
        plt.legend(loc='lower right')
    elif var_x == 'L':
        plt.axvline(x=np.log10(2.5*u.Gm.to('m')),linestyle='--',color='k',label='Proposed arm length')
        plt.xlabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        plt.xticks(10**x_labels,[r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in x_labels],fontsize = axissize)
    elif var_x == 'T_obs':
        plt.xlabel(r'${\rm T_{obs}}$ $[yr]$',fontsize = labelsize)
        plt.xticks(x_labels,[r'$%i$' %int(x) for x in x_labels/u.yr.to('s')],fontsize = axissize)


    dists_min = -2
    dists_max = np.max(np.ceil(np.log10(DL_array)))
    dists = np.arange(dists_min,dists_max+1)
    plt.ylim([10**dists_min,10**dists_max])
    plt.yticks(10**dists,[r'$10^{%i}$' %dist if np.abs(int(dist)) > 1 else '{:g}'.format(10**dist) for dist in dists],fontsize = axissize)
    plt.ylabel(r'$D_{L}$ [Mpc]',fontsize=labelsize)

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