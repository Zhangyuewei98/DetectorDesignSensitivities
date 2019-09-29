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

from gwent.utils import make_quant
from gwent import detector,snr

def getHorizonDistance(source,instrument,var_x,sampleRate_x,rho_thresh):
    '''Setting Up Horizon Distance calculation
        Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
        selection of variables. 
        Then it computes the Horizon for each value at the particular SNR given by rho_thresh (aka the threshold SNR).
        It takes in a redshift_array corresponding to the output luminosity distances so the function can be used to
            iteratively converge to correct luminosity distance values.
        Returns the variable ranges used to calculate the horizon distance for each matrix and the horizon distance
    '''
    redshift_array = np.logspace(-2,3,100)

    source.instrument = instrument
    #Get Samples for variable
    [sample_x,recalculate_strain,recalculate_noise] = Get_Samples(source,instrument,var_x,sampleRate_x)

    sampleSize_x = len(sample_x)
    new_redshift_array = np.zeros(sampleSize_x)
    rho_z = np.zeros(len(redshift_array))
    
    for i in range(sampleSize_x):
        if recalculate_noise == True:
            #Update Attribute (also updates dictionary)
            setattr(instrument,var_x,sample_x[i])
            #Recalculate noise curves if something is varied
            if hasattr(instrument,'fT'):
                del instrument.fT
            if hasattr(instrument,'P_n_f'):
                del instrument.P_n_f
            if isinstance(instrument,detector.PTA) and hasattr(instrument,'_sensitivitycurve'):
                del instrument._sensitivitycurve
        else:
            #Update Attribute (also updates dictionary)
            setattr(source,var_x,sample_x[i])

        error_number = 1.0
        for j,z in enumerate(redshift_array):
            #Update particular source's redshift
            setattr(source,'z',z)

            source.Check_Freq_Evol()
            if source.ismono: #Monochromatic Source
                if hasattr(source,'h_gw'):
                    del source.h_gw
                rho_z[j] = snr.Calc_Mono_SNR(source,instrument)
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
                rho_z[j] = snr.Calc_Chirp_SNR(source,instrument)
            if rho_z[j] < 1.0:
                rho_z[j] = error_number
                error_number -= .01

        rho_interp = interp.InterpolatedUnivariateSpline(redshift_array,rho_z-rho_thresh)
        z_val = rho_interp.roots()
        if len(z_val) == 0:
            if min(rho_z) > rho_thresh:
                new_redshift_array[i] = 1e4
            else:
                new_redshift_array[i] = 1e-10 
        elif len(z_val) == 2:
            print('Multiple roots ','Mass: ',source.M,' z_vals: ',z_val)
            print('Taking the largest redshift.')
            new_redshift_array[i] = max(z_val)
        else:
            new_redshift_array[i] = z_val[0]
        """
        plt.figure()
        plt.plot(redshift_array,rho_z)
        plt.axhline(y=rho_thresh)
        plt.axvline(new_redshift_array[i])
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        """
    DL_array = cosmo.luminosity_distance(new_redshift_array)

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
            T_obs_min = make_quant(var_x_dict['min'],'s')
            T_obs_max = make_quant(var_x_dict['max'],'s')
            sample_x = np.linspace(T_obs_min.value,T_obs_max.value,sample_rate_x)
        else:
            #Sample in log space for any other variables
            #Need exception for astropy variables
            if isinstance(var_x_dict['min'],u.Quantity) and isinstance(var_x_dict['max'],u.Quantity):
                sample_x = np.logspace(np.log10(var_x_dict['min'].value),np.log10(var_x_dict['max'].value),sampleRate_x)
            else:
                sample_x = np.logspace(np.log10(var_x_dict['min']),np.log10(var_x_dict['max']),sampleRate_x)

    return sample_x,recalculate_strain,recalculate_noise


def plotHD(source,instrument,var_x,sample_x,DL_array,display=True,figloc=None,z_axis=False):
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
    fig, ax1 = plt.subplots(figsize=figsize)
    #Set other side y-axis for lookback time scalings
    ax2 = ax1.twinx()

    #Set axis scales based on what data sampling we used 
    if xaxis_type == 'log':
        x_label_min = np.min(np.floor(np.log10(sample_x)))
        x_label_max = np.max(np.ceil(np.log10(sample_x)))
        ax1.loglog(sample_x,DL_array,color='b')
        ax2.loglog(sample_x,DL_array,color='b')
        ax1.set_xlim([10**x_label_min,10**x_label_max])
        x_labels = np.arange(x_label_min,x_label_max+1)
    elif xaxis_type == 'lin':
        x_label_min = np.min(np.floor(sample_x))
        x_label_max = np.max(np.ceil(sample_x))
        ax1.semilogy(sample_x,DL_array,color='b')
        ax2.semilogy(sample_x,DL_array,color='b')
        ax1.set_xlim([x_label_min,x_label_max])
        x_labels = np.linspace(x_label_min,x_label_max,x_label_max-x_label_min+1)

    #Set axes labels and whether log or linearly spaced
    if var_x == 'M':
        ax1.set_xlabel(r'$M_{\rm tot}$ $[M_{\odot}]$',fontsize = labelsize)
        ax1.set_xticks(10**x_labels)
        ax1.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in x_labels],fontsize = axissize)
    elif var_x == 'q':
        ax1.set_xlabel(r'$q$',fontsize = labelsize)
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(fontsize = axissize,rotation=45)
    elif var_x == 'chi1' or var_x == 'chi2':
        x_labels = np.arange(round(x_label_min*10),round(x_label_max*10)+1,1)/10
        ax1.set_xticks(x_labels)
        ax1.set_xlabel(r'${\rm Spin}$',fontsize = labelsize)
        ax1.set_xticklabels(10**x_labels,fontsize = axissize,rotation=45)
        plt.legend(loc='lower right')
    elif var_x == 'L':
        ax1.axvline(x=np.log10(2.5*u.Gm.to('m')),linestyle='--',color='k',label='Proposed arm length')
        ax1.set_xlabel(r'${\rm Armlength}$ $[m]$',fontsize = labelsize)
        ax1.set_xticks(10**x_labels)
        ax1.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in x_labels],fontsize = axissize)
    elif var_x == 'T_obs':
        ax1.set_xlabel(r'${\rm T_{obs}}$ $[yr]$',fontsize = labelsize)
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels([r'$%i$' %int(x) for x in x_labels/u.yr.to('s')],fontsize = axissize)


    dists_min = -1
    dists_max = np.nanmax(np.ceil(np.log10(DL_array)))
    dists_max = min(dists_max,np.log10(cosmo.luminosity_distance(1e3).value))
    #print('dist max: ',np.ceil(np.log10(DL_array)))
    dists = np.arange(dists_min,dists_max)
    ax1.set_ylim([10**dists_min,10**dists_max])
    ax1.set_yticks(10**dists)
    ax1.set_yticklabels([r'$10^{%i}$' %dist if np.abs(int(dist)) > 1 else '{:g}'.format(10**dist) for dist in dists],fontsize = axissize)
    ax1.set_ylabel(r'$D_{L}$ [Mpc]',fontsize=labelsize)

    if z_axis:
        z_min = -3.0
        z_max = 3.0

        zees = np.arange(z_min,z_max+1)
        zticks = [cosmo.luminosity_distance(10**z).value for z in zees]
        print(zticks)
        #Set other side y-axis for lookback time scalings
        ax2.set_ylim([10**dists_min,10**dists_max])
        ax2.set_yticks(zticks)
        ax2.set_yticklabels([r'$10^{%i}$' %dist if np.abs(int(dist)) > 1 else '{:g}'.format(10**dist) for dist in np.log10(zticks)],fontsize = axissize)
        #ax2.set_yticklabels([r'$10^{%i}$' %z if np.abs(z) > 1 else '{:g}'.format(10**z) for z in zees],fontsize = axissize)
        ax2.set_ylabel(r'Redshift',fontsize=labelsize)

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