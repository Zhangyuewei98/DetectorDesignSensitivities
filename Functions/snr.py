import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate

import astropy.units as u

import os,sys
current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('DetectorDesignSensitivities')
top_directory = "/".join(splt_path[0:top_path_idx+1])

sys.path.insert(0,top_directory + '/Functions')
import detector
import binary
from utils import make_quant


def Get_SNR_Matrix(source,instrument,var_x,sample_rate_x,var_y,sample_rate_y):
    """Calculates SNR Matrix

    Parameters
    ----------
    source : object
        Instance of a gravitational wave source class
    instrument : object
        Instance of a gravitational wave detector class
    var_x : str
        x-axis variable
    sample_rate_x : int
        Number of samples at which SNRMatrix is calculated corresponding to the x-axis variable
    var_y : str
        y-axis variable
    sample_rate_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable

    Returns
    -------
    sample_x : array
        samples at which SNRMatrix was calculated corresponding to the x-axis variable
    sample_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable
    SNRMatrix : array-like
        the sample_rate_y X sample_rate_x matrix at which the SNR was calculated corresponding to the particular x and y-axis variable choices

    Notes
    -----
    Uses the variable given and the data range to sample the space either logrithmically or linearly based on the
    selection of variables. Then it computes the SNR for each value.
    Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample_yXsample_x

    """

    source.instrument = instrument
    #Get Samples for variables
    [sample_x,sample_y,recalculate_strain,recalculate_noise] = Get_Samples(source,instrument,var_x,sample_rate_x,var_y,sample_rate_y)

    sampleSize_x = len(sample_x)
    sampleSize_y = len(sample_y)
    SNRMatrix = np.zeros((sampleSize_x,sampleSize_y))

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
                if isinstance(instrument,detector.PTA) and hasattr(instrument,'_sensitivitycurve'):
                    del instrument._sensitivitycurve

            source.Check_Freq_Evol()
            if source.ismono: #Monochromatic Source and not diff EOB SNR
                if hasattr(source,'h_gw'):
                    del source.h_gw
                SNRMatrix[j,i] = Calc_Mono_SNR(source,instrument)
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
                SNRMatrix[j,i] = Calc_Chirp_SNR(source,instrument)

    return [sample_x,sample_y,SNRMatrix]

def Get_Samples(source,instrument,var_x,sample_rate_x,var_y,sample_rate_y):
    """Gets the x and y-axis samples

    Parameters
    ----------
    source : object
        Instance of a gravitational wave source class
    instrument : object
        Instance of a gravitational wave detector class
    var_x : str
        x-axis variable
    sample_rate_x : int
        Number of samples at which SNRMatrix is calculated corresponding to the x-axis variable
    var_y : str
        y-axis variable
    sample_rate_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable

    Returns
    -------
    sample_x : array
        samples at which SNRMatrix was calculated corresponding to the x-axis variable
    sample_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable

    Notes
    -----
        The function uses that to create a
        sample space for the variable either in linear space (for q,x1,or x2) or logspace
        for everything else.

    """
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
        raise ValueError(var_x + ' is not a variable in the source nor the instrument.')

    if var_y in source.var_dict.keys():
        var_y_dict = source.var_dict[var_y]
    elif var_y in instrument.var_dict.keys():
        if recalculate_noise == 'x':
            recalculate_noise = 'both'
        else:
            recalculate_noise = 'y'
        var_y_dict = instrument.var_dict[var_y]
    else:
        raise ValueError(var_y + ' is not a variable in the source nor the instrument.')

    if var_x_dict['min'] != None and var_x_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_x == 'q' or var_x == 'chi1' or var_x == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_x = np.linspace(var_x_dict['min'],var_x_dict['max'],sample_rate_x)
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
                sample_x = np.logspace(np.log10(var_x_dict['min'].value),np.log10(var_x_dict['max'].value),sample_rate_x)
            else:
                sample_x = np.logspace(np.log10(var_x_dict['min']),np.log10(var_x_dict['max']),sample_rate_x)

    if var_y_dict['min'] != None and var_y_dict['max'] != None: #If the variable has non-None 'min',and 'max' dictionary attributes
        if var_y == 'q' or var_y == 'chi1' or var_y == 'chi2':
            #Sample in linear space for mass ratio and spins
            sample_y = np.linspace(var_y_dict['min'],var_y_dict['max'],sample_rate_y)
            recalculate_strain = True #Must recalculate the waveform at each point
        elif var_y == 'T_obs':
            #sample in linear space for instrument variables
            T_obs_min = make_quant(var_y_dict['min'],'s')
            T_obs_max = make_quant(var_y_dict['max'],'s')
            sample_y = np.linspace(T_obs_min.value,T_obs_max.value,sample_rate_y)
        else:
            #Sample in log space for any other variables
            #Need exception for astropy variables
            if isinstance(var_y_dict['min'],u.Quantity) and isinstance(var_y_dict['max'],u.Quantity):
                sample_y = np.logspace(np.log10(var_y_dict['min'].value),np.log10(var_y_dict['max'].value),sample_rate_y)
            else:
                sample_y = np.logspace(np.log10(var_y_dict['min']),np.log10(var_y_dict['max']),sample_rate_y)

    return sample_x,sample_y,recalculate_strain,recalculate_noise

def Calc_Mono_SNR(source,instrument):
    """Calculates the SNR for a monochromatic source

    Parameters
    ----------
    source : object
        Instance of a gravitational wave source class
    instrument : object
        Instance of a gravitational wave detector class

    """
    if isinstance(instrument,detector.PTA):
    	source.h_gw = binary.Get_Mono_Strain(source,instrument.f_opt,strain_const='Optimal')
    else:
    	source.h_gw = binary.Get_Mono_Strain(source,instrument.f_opt,strain_const='Averaged')
    indxfgw = np.abs(instrument.fT-instrument.f_opt).argmin()

    return source.h_gw*np.sqrt(instrument.T_obs.to('s')/instrument.S_n_f[indxfgw])

def Calc_Chirp_SNR(source,instrument):
    """Calculates the SNR for an evolving source

    Parameters
    ----------
    source : object
        Instance of a gravitational wave source class
    instrument : object
        Instance of a gravitational wave detector class

    Notes
    -----
    Uses an interpolated method to align waveform and instrument noise, then integrates
    over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 <https://arxiv.org/abs/1803.01944>
    Values outside of the sensitivity curve are arbitrarily set to 1e30 so the SNR is effectively 0

    """

    #Use to integrate from initial observed frequency f(t_init) to f(t_init-T_obs)
    #Does not work unless t_init is randomly sampled, which we don't do
    #indxfgw_start = np.abs(source.f-source.f_init).argmin()
    #indxfgw_end = np.abs(source.f-source.f_T_obs).argmin()

    #Only want to integrate from observed frequency (f(T_obs_before_merger)) till merger
    indxfgw_start = np.abs(source.f-source.f_T_obs).argmin()
    indxfgw_end = len(source.f)
    if indxfgw_start >= len(source.f)-1:
        #If the SMBH has already merged set the SNR to ~0
        return 1e-30
    else:
        f_cut = source.f[indxfgw_start:indxfgw_end]
        h_cut = source.h_f[indxfgw_start:indxfgw_end]

    #################################
    #Interpolate the Strain Noise Spectral Density to only the frequencies the
    #strain runs over
    #Set Noise to 1e30 outside of signal frequencies
    S_n_f_interp_old = interp.interp1d(np.log10(instrument.fT.value),np.log10(instrument.S_n_f.value),kind='cubic',fill_value=30.0, bounds_error=False)
    S_n_f_interp_new = S_n_f_interp_old(np.log10(f_cut.value))
    S_n_f_interp = 10**S_n_f_interp_new

    if isinstance(instrument,detector.PTA):
        #Rescaled by 1.5 to make SNR plots match...
        integral_consts = 4.*1.5

    elif isinstance(instrument,detector.SpaceBased) or isinstance(instrument,detector.GroundBased):
        integral_consts = 16./5.


    #CALCULATE SNR FOR BOTH NOISE CURVES
    denom = S_n_f_interp #Sky Averaged Noise Spectral Density
    numer = h_cut**2
    integrand = numer/denom
    if isinstance(integrand,u.Quantity) and isinstance(f_cut,u.Quantity):
        SNRsqrd = integral_consts*np.trapz(integrand.value,f_cut.value,axis=0) #SNR**2
    else:
        SNRsqrd = integral_consts*np.trapz(integrand,f_cut,axis=0) #SNR**2

    return np.sqrt(SNRsqrd)


def saveSNR(sample_x,sample_y,SNRMatrix,save_location,SNR_filename,sample_filename):
    """Saves SNR Matrix

    Parameters
    ----------
    sample_x : array
        samples at which SNRMatrix was calculated corresponding to the x-axis variable
    sample_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable
    SNRMatrix : array-like
        the matrix at which the SNR was calculated corresponding to the particular x and y-axis variable choices
    save_location : str
        the directory to which the Samples and SNR are saved
    SNR_filename : str
        the name of the SNR file
    sample_filename : str
        the name of the sample file

    """
    np.savetxt(save_location+SNR_filename,SNRMatrix)
    np.savetxt(save_location+sample_filename,[sample_x,sample_y])
