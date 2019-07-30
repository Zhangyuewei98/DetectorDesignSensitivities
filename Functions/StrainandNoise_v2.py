import numpy as np
import os
import astropy.constants as const
import astropy.units as u
import scipy.interpolate as interp
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

import matplotlib.pyplot as plt

from fractions import Fraction

import IMRPhenomD as PhenomD

import hasasia.sensitivity as hassens
import hasasia.sim as hassim

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('DetectorDesignSensitivities')
top_directory = "/".join(splt_path[0:top_path_idx+1])

class PTA:
    def __init__(self,name):
        self.name = name
        self.inst_var_dict = {}
        self.fT = []
        self.h_n_f = []
        self.S_n_f_sqrt = []
        self.f_opt = []
        self.Background = False
        self.__nfreqs = int(1e3)
        self.__sensitivitycurve = []

    def Set_T_obs(self,T_obs,T_obs_min=None,T_obs_max=None):
        self.inst_var_dict['T_obs'] = {'val':T_obs,'min':T_obs_min,'max':T_obs_max}

    def Set_N_p(self,N_p,N_p_min=None,N_p_max=None):
        self.inst_var_dict['N_p'] = {'val':N_p,'min':N_p_min,'max':N_p_max}

    def Set_Cadence(self,cadence,cadence_min=None,cadence_max=None):
        self.inst_var_dict['cadence'] = {'val':cadence,'min':cadence_min,'max':cadence_max}

    def Set_Sigma(self,sigma,sigma_min=None,sigma_max=None):
        self.inst_var_dict['sigma'] = {'val':sigma,'min':sigma_min,'max':sigma_max}

    def Set_nfreqs(self,nfreqs):
        self.__nfreqs = nfreqs

    def Set_f_opt(self):
        if len(self.fT) == 0 or len(self.h_n_f) == 0:
            self.Get_Strain()
        #Get optimal (highest sensitivity) frequency
        self.f_opt = self.fT[np.argmin(self.h_n_f)]

    def Set_Freqs(self,f_low,f_high):
        self.fT = np.logspace(f_low,f_high,self.__nfreqs)

    def Get_Param_Dict(self,var_name):
        return self.inst_var_dict[var_name]

    def Load_Data(self,load_location):
        I_data = np.loadtxt(load_location)
        self.fT = I_data[:,0]*u.Hz
        self.h_n_f = I_data[:,1]

    def Add_Background(self,A_stoch_back=4e-16):
        #Stochastic background amplitude from Sesana et al. 2016 https://arxiv.org/pdf/1603.09348.pdf
        f_year = 1/u.yr
        P_sb = (A_stoch_back**2/12/np.pi**2)*self.fT**(-3)*(self.fT/f_year.to('Hz'))**(-4/3)
        #h_sb = A_stoch_back*(f/f_year)**(-2/3)
        return P_sb

    def Add_WhiteNoise(self):
        #Need to fix for multiple ptas
        #Equation 5 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
        sigma = self.Get_Param_Dict('sigma')['val']
        cadence = self.Get_Param_Dict('cadence')['val'].to('1/s')

        P_w = sigma**2/cadence #Avg white noise from pulsar arrays [s**2/Hz]
        return P_w

    def Add_RedNoise(self):
        #Paragraph below eqn 4 from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
        #P_red = A_red.*(f./f_year).**(-gamma) # red noise for some pulsar
        P_red = 0.0*u.s*u.s/u.Hz #Assume no pulsar red noise for simplicity
        return P_red

    def Get_ASD_Moore_2014(self):
        if len(self.h_n_f) == 0: 
            self.Get_Strain_Moore_2014()
        #from Jenet et al. 2006 https://arxiv.org/abs/astro-ph/0609013 (Only for GWB/broadband signals)
        P_w = self.h_n_f**2/12/np.pi**2*self.fT**(-3)

        P_red = self.Add_RedNoise()
        
        #eqn 42 of T&R 2013 https://arxiv.org/abs/1310.5300
        #h_inst = np.sqrt((12*np.pi**2*(P_w+P_red))*f**3)

        if self.Background:
            P_sb = self.Add_Background()
        else:
            P_sb = self.Add_Background(A_stoch_back=0.0)

        ####################################################
        #PSD of the full PTA from Lam,M.T. 2018 https://arxiv.org/abs/1808.10071
        P_n = P_w + P_red + P_sb
        
        #####################################################
        #PSD of a PTA taken from eqn 40 of Thrane & Romano 2013 https://arxiv.org/abs/1310.5300

        #strain of the full PTA
        #h_f = np.sqrt((12*np.pi**2)*f**3*P_n)
        ASD = np.sqrt((12*np.pi**2)*self.fT**2*P_n)
        self.S_n_f_sqrt = ASD

    def Get_Strain_Moore_2014(self):
        # Taken from Moore,Taylor, and Gair 2014 https://arxiv.org/abs/1406.5199
        sigma = self.Get_Param_Dict('sigma')['val']
        T_obs = self.Get_Param_Dict('T_obs')['val'].to('s')
        N_p = self.Get_Param_Dict('N_p')['val']
        cadence = self.Get_Param_Dict('cadence')['val'].to('1/s')

        P_w = self.Add_WhiteNoise()
        #frequency sampled from 1/observation time to nyquist frequency (c/2)
        self.fT = np.logspace(np.log10(1/T_obs.value),np.log10(cadence.value/2),self.__nfreqs)*u.Hz

        SNR = 1.0 # Value of 3 is Used in paper
        chi_corr = 1/np.sqrt(3) #Sky averaged geometric factor eqn. 11
        overlap_freq = 2/T_obs
        #N_p = max(N_p) #Just use largest number of pulsars from all ptas

        #h_c_high_factor = (16*SNR**2/(3*chi_corr**4*N_p*(N_p-1)))**(1/4)*sigma_rms*np.sqrt(1/T_obs_tot/cadence)
        #h_c_low_factor = 3*np.sqrt(SNR)/(2**(7/4)*chi_corr*np.pi**3)*(13/N_p/(N_p-1))**(1/4)*sigma_rms*np.sqrt(1/T_obs_tot/cadence)*T_obs_tot**(-3)
        h_c_high_factor = (16*SNR**2/(3*chi_corr**4*N_p*(N_p-1)))**(1/4)*np.sqrt(P_w/2/T_obs)
        h_c_low_factor = 3*np.sqrt(SNR)/(2**(7/4)*chi_corr*np.pi**3)*(13/N_p/(N_p-1))**(1/4)*np.sqrt(P_w/2/T_obs)*T_obs**(-3)

        phi = np.arccos(h_c_low_factor/h_c_high_factor*overlap_freq**(-3))

        h_c_high = h_c_high_factor*self.fT
        h_c_low = h_c_low_factor*self.fT**(-2)*(1/np.cos(phi))

        h_c = h_c_low + h_c_high #Below eqn 16, should it be added in quad?
        self.h_n_f = h_c

    def Init_PTA(self):
        #### Using Jeff's Methods/code https://arxiv.org/abs/1907.04341
        sigma = self.Get_Param_Dict('sigma')['val'].value
        T_obs = self.Get_Param_Dict('T_obs')['val'].value
        N_p = self.Get_Param_Dict('N_p')['val']
        cadence = self.Get_Param_Dict('cadence')['val'].value

        if len(self.fT) == 0:
            #5 is the default value?
            T_obs_sec = T_obs*u.yr.to('s')
            cadence_sec = cadence/u.yr.to('s')
            self.fT = np.logspace(np.log10(1/(5*T_obs_sec)),np.log10(cadence_sec/2),self.__nfreqs)*u.Hz

        #Random Sky Locations of Pulsars
        phi = np.random.uniform(0, 2*np.pi,size=N_p)
        theta = np.random.uniform(0, np.pi,size=N_p)

        #Make a set of psrs with the same parameters
        psrs = hassim.sim_pta(timespan=T_obs,cad=cadence,sigma=sigma,
            phi=phi, theta=theta, Npsrs=N_p)
        #Get Spectra of pulsars
        spectra= []
        for p in psrs:
             sp = hassens.Spectrum(p,freqs=self.fT.value)
             spectra.append(sp)

        self.__sensitivitycurve = hassens.DeterSensitivityCurve(spectra)

    def Get_ASD_Hazboun_2019(self):
        self.S_n_f_sqrt = np.sqrt(self.__sensitivitycurve.S_eff)/(u.Hz)**Fraction(1,2)

    def Get_Strain_Hazboun_2019(self):
        self.h_n_f = self.__sensitivitycurve.h_c

    def Default_Setup_Hazboun_2019(self,T_obs,N_p,sigma,cadence):
        self.Set_T_obs(T_obs)
        self.Set_N_p(N_p)
        self.Set_Sigma(sigma)
        self.Set_Cadence(cadence)
        self.Init_PTA()
        self.Get_Strain_Hazboun_2019()
        self.Get_ASD_Hazboun_2019()
        self.Set_f_opt()

    def Default_Setup_Moore_2014(self,T_obs,N_p,sigma,cadence):
        self.Set_T_obs(T_obs)
        self.Set_N_p(N_p)
        self.Set_Sigma(sigma)
        self.Set_Cadence(cadence)
        self.Get_Strain_Moore_2014()
        self.Get_ASD_Moore_2014()
        self.Set_f_opt()

class GroundBased:
    def __init__(self,name):
        self.name = name
        self.inst_var_dict = {}
        self.fT = []
        self.S_n_f_sqrt = []
        self.h_n_f = []
        self.f_opt = []
        self.__I_data = []

    def Set_T_obs(self,T_obs,T_obs_min=None,T_obs_max=None):
        self.inst_var_dict['T_obs'] = {'val':T_obs,'min':T_obs_min,'max':T_obs_max}

    def Set_f_opt(self):
        #Get optimal (highest sensitivity) frequency
        self.f_opt = self.fT[np.argmin(self.h_n_f)]

    def Get_Param_Dict(self,var_name):
        return self.inst_var_dict[var_name]

    def Load_Data(self,load_location):
        self.__I_data = np.loadtxt(load_location)

    def Get_ASD(self,load_location):
        if len(self.__I_data) == 0:
            self.Load_Data(load_location)
        self.fT = self.__I_data[:,0]*u.Hz
        self.S_n_f_sqrt = self.__I_data[:,1]/(u.Hz)**Fraction(1,2)

    def Get_Strain(self,load_location):
        if len(self.fT) == 0 or len(self.S_n_f_sqrt) == 0:
            self.Get_ASD(load_location)
        self.h_n_f = np.sqrt(self.fT)*self.S_n_f_sqrt

    def Default_Setup(self,load_location):
        self.Load_Data(load_location)
        self.Set_T_obs(4*u.yr.to('s')*u.s)
        self.Get_ASD(load_location)
        self.Get_Strain(load_location)
        self.Set_f_opt()

class SpaceBased:
    def __init__(self,name):
        self.name = name
        self.inst_var_dict = {}
        self.Background = False
        self.fT = []
        self.S_n_f_sqrt = []
        self.h_n_f = []
        self.transferfunction = []
        self.f_opt = []
        self.__nfreqs = int(1e3)
        self.__f_low = 1e-5*u.Hz
        self.__f_high = 1.0*u.Hz
        self.__transferfunctiondata = []

    def Set_T_obs(self,T_obs,T_obs_min=None,T_obs_max=None):
        self.inst_var_dict['T_obs'] = {'val':T_obs,'min':T_obs_min,'max':T_obs_max}

    def Set_L(self,L,L_min=None,L_max=None):
        self.inst_var_dict['L'] = {'val':L,'min':L_min,'max':L_max}

    def Set_A_acc(self,A_acc,A_acc_min=None,A_acc_max=None):
        self.inst_var_dict['A_acc'] = {'val':A_acc,'min':A_acc_min,'max':A_acc_max}

    def Set_f_acc_break_low(self,f_acc_break_low,f_acc_break_low_min=None,f_acc_break_low_max=None):
        self.inst_var_dict['f_acc_break_low'] = {'val':f_acc_break_low,'min':f_acc_break_low_min,'max':f_acc_break_low_max}

    def Set_f_acc_break_high(self,f_acc_break_high,f_acc_break_high_min=None,f_acc_break_high_max=None):
        self.inst_var_dict['f_acc_break_high'] = {'val':f_acc_break_high,'min':f_acc_break_high_min,'max':f_acc_break_high_max}

    def Set_A_IMS(self,A_IMS,A_IMS_min=None,A_IMS_max=None):
        self.inst_var_dict['A_IMS'] = {'val':A_IMS,'min':A_IMS_min,'max':A_IMS_max}

    def Set_f_IMS_break(self,f_IMS_break,f_IMS_break_min=None,f_IMS_break_max=None):
        self.inst_var_dict['f_IMS_break'] = {'val':f_IMS_break,'min':f_IMS_break_min,'max':f_IMS_break_max}

    def Set_f_low(self,f_low):
        self.__f_low = f_low

    def Set_f_high(self,f_high):
        self.__f_high = f_high

    def Set_nfreqs(self,nfreqs):
        self.__nfreqs = int(nfreqs)

    def Set_f_opt(self):
        if len(self.fT) == 0:
            self.Get_TransferFunction()
        if len(self.h_n_f) == 0:
            self.Get_Strain()
        #Get optimal (highest sensitivity) frequency
        self.f_opt = self.fT[np.argmin(self.h_n_f)]

    def Update_Param_val(self,valname,newval):
        self.inst_var_dict[valname]['val'] = newval

    def Get_Param_Dict(self,var_name):
        return self.inst_var_dict[var_name]

    def Load_Data(self,load_location):
        I_data = np.loadtxt(load_location)
        self.fT = I_data[:,0]*u.Hz
        S_n_f_sqrt = I_data[:,1]
        self.S_n_f_sqrt = S_n_f_sqrt/(u.Hz)**Fraction(1,2)

    def Load_TransferFunction(self):
        LISA_Transfer_Function_filedirectory = top_directory + '/LoadFiles/LISATransferFunction/'
        LISA_Transfer_Function_filename = 'transfer.dat' #np.loadtxting transfer function for Lisa noise curve
        LISA_Transfer_Function_filelocation = LISA_Transfer_Function_filedirectory + LISA_Transfer_Function_filename
        LISA_Transfer_Function_data = np.loadtxt(LISA_Transfer_Function_filelocation)
        self.__transferfunctiondata = LISA_Transfer_Function_data

    def Get_TransferFunction(self):
        if len(self.__transferfunctiondata) == 0:
            self.Load_TransferFunction()

        fc = const.c/(2*self.Get_Param_Dict('L')['val'])  #light round trip freq
        LISA_Transfer_Function_f = fc*self.__transferfunctiondata[:,0]

        idx_f_5 = np.abs(LISA_Transfer_Function_f-self.__f_low).argmin()
        idx_f_1 = np.abs(LISA_Transfer_Function_f-self.__f_high).argmin()

        self.transferfunction = self.__transferfunctiondata[idx_f_5:idx_f_1,1]
        self.fT = LISA_Transfer_Function_f[idx_f_5:idx_f_1]

    def Get_approxTransferFunction(self):
        #Response function approximation from Calculation described by Neil Cornish (unpublished)
        self.fT = np.logspace(np.log10(self.__f_low.value),np.log10(self.__f_high.value),self.__nfreqs)*u.Hz
        f_L = const.c/2/np.pi/self.Get_Param_Dict('L')['val'] #Transfer frequency
        R_f = 3/10/(1+0.6*(self.fT/f_L)**2) 
        self.transferfunction = np.sqrt(R_f)

    def Get_PSD(self):
        if len(self.fT) == 0:
            self.Get_TransferFunction()
        A_acc = self.Get_Param_Dict('A_acc')['val']

        f_acc_break_low = self.Get_Param_Dict('f_acc_break_low')['val']
        f_acc_break_high = self.Get_Param_Dict('f_acc_break_high')['val']

        A_IMS = self.Get_Param_Dict('A_IMS')['val']
        f_IMS_break = self.Get_Param_Dict('f_IMS_break')['val']

        L = self.Get_Param_Dict('L')['val']

        P_acc = A_acc**2*(1+(f_acc_break_low/self.fT)**2)*(1+(self.fT/(f_acc_break_high))**4)/(2*np.pi*self.fT)**4 #Acceleration Noise 
        P_IMS = A_IMS**2*(1+(f_IMS_break/self.fT)**4) #Displacement noise of the interferometric TM--to-TM 

        f_trans = const.c/2/np.pi/L #Transfer frequency
        PSD = (P_IMS + 2*(1+np.cos(self.fT.value/f_trans.value)**2)*P_acc)/L**2
        return PSD

    def Get_ASD(self,Norm=20/3):
        #Calculates amplitude spectral density
        PSD = self.Get_PSD()
        ASD = Norm/self.transferfunction**2*PSD #Strain Noise Power Spectral Density
        if self.Background:
            self.S_n_f_sqrt = np.sqrt(ASD+self.Add_Background()) #Sqrt of PSD
        else:
            self.S_n_f_sqrt = np.sqrt(ASD)

    def Get_Strain(self):
        if len(self.fT) == 0:
            self.Get_TransferFunction() 
        if len(self.S_n_f_sqrt) == 0:
            if self.name == 'Neil_LISA':
                self.Get_ASD(Norm=1.0)
            else:
                self.Get_ASD()
        self.h_n_f = np.sqrt(self.fT)*self.S_n_f_sqrt


    def Add_Background(self):
        #4 year Galactic confusions noise parameters
        A = 9e-45
        a = 0.138
        b = -221
        k = 521
        g = 1680
        f_k = 0.00113
        f = self.fT.value
        return A*np.exp(-(f**a)+(b*f*np.sin(k*f)))*(f**(-7/3))*(1 + np.tanh(g*(f_k-f))) #White Dwarf Background Noise

    def Default_Setup(self,T_obs,L,A_acc,f_acc_break_low,f_acc_break_high,A_IMS,f_IMS_break,Background):
        self.Set_T_obs(T_obs)
        self.Set_L(L)
        self.Set_A_acc(A_acc)
        self.Set_f_acc_break_low(f_acc_break_low)
        self.Set_f_acc_break_high(f_acc_break_high)
        self.Set_A_IMS(A_IMS)
        self.Set_f_IMS_break(f_IMS_break)
        self.Get_TransferFunction()
        self.Background = Background
        self.Get_ASD()
        self.Get_Strain()
        self.Set_f_opt()

class BlackHoleBinary:
    def __init__(self):
        self.source_var_dict = {}
        self.f = []
        self.h_f = []
        self.h_gw = []
        self.f_low = 1e-5
        self.nfreqs = int(1e3)
        self.T_obs = []
        self.f_init = []
        self.f_T_obs = []
        self.ismono = False
        self.__fitcoeffs = []
        self.__instrument = None

    def Set_Mass(self,M,M_min=None,M_max=None):
        self.source_var_dict['M'] = {'val':M,'min':M_min,'max':M_max}
    def Set_MassRatio(self,q,q_min=None,q_max=None):
        self.source_var_dict['q'] = {'val':q,'min':q_min,'max':q_max}
    def Set_Chi1(self,chi1,chi1_min=None,chi1_max=None):
        self.source_var_dict['chi1'] = {'val':chi1,'min':chi1_min,'max':chi1_max}
    def Set_Chi2(self,chi2,chi2_min=None,chi2_max=None):
        self.source_var_dict['chi2'] = {'val':chi2,'min':chi2_min,'max':chi2_max}
    def Set_Redshift(self,z,z_min=None,z_max=None):
        self.source_var_dict['z'] = {'val':z,'min':z_min,'max':z_max}
    def Set_Inclination(self,inc,inc_min=None,inc_max=None):
        self.source_var_dict['inc'] = {'val':inc,'min':inc_min,'max':inc_max}

    def Set_T_obs(self,T_obs):
        self.T_obs = T_obs
    def Set_f_init(self,f_init):
        self.f_init = f_init
    def Set_f_low(self,f_low):
        self.f_low = f_low
    def Set_nfreqs(self,nfreqs):
        self.nfreqs = nfreqs
    def Set_Instrument(self,instrument):
        self.__instrument = instrument
        self.Set_T_obs(instrument.Get_Param_Dict('T_obs')['val'])
        self.Set_f_init(instrument.f_opt)

    def Update_Param_val(self,valname,newval):
        self.source_var_dict[valname]['val'] = newval

    def Get_Param_Dict(self,var_name):
        return self.source_var_dict[var_name]

    def StrainConv(self,natural_f,natural_h):
        M = self.Get_Param_Dict('M')['val']
        q = self.Get_Param_Dict('q')['val']
        z = self.Get_Param_Dict('z')['val']

        DL = cosmo.luminosity_distance(z)
        DL = DL.to('m')

        m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
        M_redshifted_time = M*(1+z)*m_conv
        
        freq_conv = 1/M_redshifted_time
        #Normalized factor?
        #Changed from sqrt(5/16/pi)
        strain_conv = np.sqrt(1/4/np.pi)*(const.c/DL)*M_redshifted_time**2
        
        self.f = natural_f*freq_conv
        self.h_f = natural_h*strain_conv

    def Get_fitcoeffs(self):
        fit_coeffs_filedirectory = top_directory + '/LoadFiles/PhenomDFiles/'
        fit_coeffs_filename = 'fitcoeffsWEB.dat'
        fit_coeffs_file = fit_coeffs_filedirectory + fit_coeffs_filename
        self.__fitcoeffs = np.loadtxt(fit_coeffs_file) #load QNM fitting files for speed later

    def Get_PhenomD_Strain(self):
        if len(self.__fitcoeffs) == 0:
            self.Get_fitcoeffs()

        M = self.Get_Param_Dict('M')['val']
        q = self.Get_Param_Dict('q')['val']
        chi1 = self.Get_Param_Dict('chi1')['val']
        chi2 = self.Get_Param_Dict('chi2')['val']
        z = self.Get_Param_Dict('z')['val']
        Vars = [M,q,chi1,chi2,z]

        [phenomD_f,phenomD_h] = PhenomD.FunPhenomD(Vars,self.__fitcoeffs,self.nfreqs,f_low=self.f_low)
        return [phenomD_f,phenomD_h]

    def Get_CharStrain(self):
        if len(self.f) != 0 and len(self.h_f) != 0:
            h_char = np.sqrt(4*self.f**2*self.h_f**2)
            return h_char
        else:
            print('You need to get f and h_f first. \n')
            print('Call Get_Waveform, then StrainConv')
            return []

    def Get_MonoStrain(self,strain_const='Cornish'):
        M = self.Get_Param_Dict('M')['val']
        q = self.Get_Param_Dict('q')['val']
        z = self.Get_Param_Dict('z')['val']
        inc = self.Get_Param_Dict('inc')['val']

        DL = cosmo.luminosity_distance(z)
        DL = DL.to('m')

        m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]

        eta = q/(1+q)**2
        M_redshifted_time = M*(1+z)*m_conv
        M_chirp = eta**(3/5)*M_redshifted_time
        #Source is emitting at one frequency (monochromatic)
        #strain of instrument at f_cw
        if strain_const == 'Rosado':
            #Strain from Rosado, Sesana, and Gair (2015) https://arxiv.org/abs/1503.04803
            #(ie. sky and inclination averaged)
            #inc = 0.0 #optimally oriented
            a = 1+np.cos(inc)**2
            b = -2*np.cos(inc)
            A = 2*(const.c/DL)*(np.pi*self.f_init)**(2./3.)*M_chirp**(5./3.)
            h_gw = A*np.sqrt(.5*(a**2+b**2))
        elif strain_const == 'Cornish':
            #Strain from Cornish et. al 2018 (eqn 27) https://arxiv.org/pdf/1803.01944.pdf
            #(ie. optimally oriented)
            h_gw = 8/np.sqrt(5)*np.sqrt(self.T_obs)*(const.c/DL)*(np.pi*self.f_init)**(2./3.)*M_chirp**(5./3.)

        self.h_gw = h_gw

    def checkFreqEvol(self):
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
        M = self.source_var_dict['M']['val']
        q = self.source_var_dict['q']['val']
        z = self.source_var_dict['z']['val']
        m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
        
        eta = q/(1+q)**2
        M_redshifted_time = M*(1+z)*m_conv
        M_chirp = eta**(3/5)*M_redshifted_time

        #from eqn 41 from Hazboun,Romano, and Smith (2019) https://arxiv.org/abs/1907.04341
        t_init = 5*(M_chirp)**(-5/3)*(8*np.pi*self.f_init)**(-8/3)
        #f(t) from eqn 40
        f_evolve = 1./8./np.pi/M_chirp*(5*M_chirp/(t_init-self.T_obs))**(3./8.)
        self.f_T_obs = 1./8./np.pi/M_chirp*(5*M_chirp/self.T_obs)**(3./8.)
        #del(f) from eqn 42
        delf = 1./8./np.pi/M_chirp*(5*M_chirp/t_init)**(3./8.)*(3*self.T_obs/8/t_init)
        
        if delf < (1/self.T_obs):
            self.ismono = True
        else:
            self.ismono = False

    def Default_Setup(self,M,q,chi1,chi2,z,inc,instrument):
        self.Set_Mass(M)
        self.Set_MassRatio(q)
        self.Set_Chi1(chi1)
        self.Set_Chi2(chi2)
        self.Set_Redshift(z)
        self.nfreqs = int(1e3)   #Sample rate of strain/Transfer function frequencies
        self.Set_Inclination(inc)
        self.Set_Instrument(instrument)
        self.Get_MonoStrain()
        [phenomD_f,phenomD_h] = self.Get_PhenomD_Strain()
        self.StrainConv(phenomD_f,phenomD_h)

    

class TimeDomain:
    def __init__(self,name):
        self.name = name
        self.source_var_dict = {}
        self.natural_f = []
        self.natural_h_f = []
        self.f = []
        self.h_f = []
        self.t = []
        self.h_plus_t = []
        self.h_cross_t = []

    def Set_Mass(self,M,M_min=None,M_max=None):
        self.source_var_dict['M'] = {'val':M,'min':M_min,'max':M_max}
    def Set_MassRatio(self,q,q_min=None,q_max=None):
        self.source_var_dict['q'] = {'val':q,'min':q_min,'max':q_max}
    def Set_Redshift(self,z,z_min=None,z_max=None):
        self.source_var_dict['z'] = {'val':z,'min':z_min,'max':z_max}

    def Get_Param_Dict(self,var_name):
        return self.source_var_dict[var_name]

    def Load_Strain(self):
        diff_filename = self.name + '.dat'
        diff_filelocation = top_directory + '/LoadFiles/DiffStrain/EOBdiff/' + diff_filename
        diff_data = np.loadtxt(diff_filelocation)
        self.t = diff_data[:,0]*u.s
        self.h_plus_t = diff_data[:,1]
        self.h_cross_t = diff_data[:,2]

    def Get_hf_from_hcross_hplus(self,interp_res='coarse',windowing='left'):
        '''Converts dimensionless, time domain strain to frequency space'''

        #Interpolate time to evenly sampled data, can be fine or coarse
        diff_t = np.diff(self.t.value)
        if interp_res == 'fine':
            dt = min(diff_t)
        elif interp_res == 'coarse':
            dt = max(diff_t)

        interp_t = np.arange(self.t[0].value,self.t[-1].value,dt)
        #interpolate strain to evenly sampled data for FFT
        h_cross_t = interp.interp1d(self.t,self.h_cross_t,kind='cubic')
        h_plus_t = interp.interp1d(self.t,self.h_plus_t,kind='cubic')
        interp_h_cross_t = h_cross_t(interp_t)
        interp_h_plus_t = h_plus_t(interp_t)

        #Filter/Window
        hann_window = np.hanning(len(interp_t)) #Two sided
        if windowing == 'left':
            #########################
            '''Applies window to first (left) half'''
            first_half = hann_window[:int(len(interp_t)/2)] # Only need tapering on first half of waveform
            second_half = np.ones(len(interp_t)-len(first_half)) #no windowing on second half of waveform
            #########################
            window = np.append(first_half,second_half) # Only apply window to first half of waveform
        elif windowing == 'right':
            #########################
            '''Applies window to second (right) half'''
            second_half = hann_window[int(len(interp_t)/2):] # Only need tapering on second half of waveform
            first_half = np.ones(len(interp_t)-len(second_half)) #no windowing on first half of waveform
            #########################
            window = np.append(first_half,second_half)
        elif windowing == 'all':
            window = hann_window
        #Window!     
        win_h_cross_t = np.multiply(interp_h_cross_t,window)
        win_h_plus_t = np.multiply(interp_h_plus_t,window)

        #FFT the two polarizations
        h_cross_f = np.fft.fft(win_h_cross_t)
        h_plus_f = np.fft.fft(win_h_plus_t)
        freqs = np.fft.fftfreq(len(interp_t),d=dt)

        #cut = np.abs(freqs).argmax() #Cut off the negative frequencies
        f_cut_low = 3e-3 #Low Cutoff frequency
        f_cut_high = 1.5e-1 #High Cutoff frequency 
        cut_low = np.abs(freqs-f_cut_low).argmin() #Cut off frequencies lower than a frequency
        cut_high = np.abs(freqs-f_cut_high).argmin() #Cut off frequencies higher than a frequency
        #cut=int(len(freqs)*0.9) #Cut off percentage of frequencies
        h_cross_f = h_cross_f[cut_low:cut_high]
        h_plus_f = h_plus_f[cut_low:cut_high]
        self.natural_f = freqs[cut_low:cut_high]
        
        #Combine them for raw spectral power
        self.natural_h_f = np.sqrt((np.abs(h_cross_f))**2 + (np.abs(h_plus_f))**2)

    def Get_CharStrain(self):
        if len(self.f) != 0 and len(self.h_f) != 0:
            h_char = np.sqrt(4*self.f**2*self.h_f**2)
            return h_char
        else:
            print('You need to get f and h_f first. \n')
            return []

    def StrainConv(self):
        M = self.Get_Param_Dict('M')['val']
        q = self.Get_Param_Dict('q')['val']
        z = self.Get_Param_Dict('z')['val']

        DL = cosmo.luminosity_distance(z)
        DL = DL.to('m')

        m_conv = const.G*const.M_sun/const.c**3 #Converts M = [M] to M = [sec]
        M_redshifted_time = M*(1+z)*m_conv
        
        freq_conv = 1/M_redshifted_time
        #Normalized factor?
        #Changed from sqrt(5/16/pi)
        strain_conv = np.sqrt(1/4/np.pi)*(const.c/DL)*M_redshifted_time**2
        
        self.f = self.natural_f*freq_conv
        self.h_f = self.natural_h_f*strain_conv

    def Default_Setup(self,M,q,z):
        self.Get_hf_from_hcross_hplus()
        self.Set_Mass(M)
        self.Set_MassRatio(q)
        self.Set_Redshift(z)
        self.StrainConv()