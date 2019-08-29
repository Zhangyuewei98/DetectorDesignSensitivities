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
    '''
    Class to make a PTA instrument using the methods of Hazboun, Romano, Smith 2019
    '''
    def __init__(self,name,*args,**kwargs):
        '''
        name - name of the instrument
        args in order: 
        T_obs - the observation time of the PTA in [years]
        N_p - the number of pulsars in the PTA 
        sigma - the rms error on the pulsar TOAs in [sec]
        cadence - How often the pulsars are observed in [num/year]

        kwargs can be:
        load_location: If you want to load a PTA curve from a file, 
                        it's the file path
        A_GWB: Amplitude of the gravitational wave background added as red noise
        alpha_GWB: the GWB power law, if empty and A_GWB is set, it is assumed to be -2/3
        A_rn: Individual pulsar red noise amplitude, is a list of [min,max] values from
                which to uniformly sample
        alpha_rn: Individual pulsar red noise alpha (power law), is a list of [min,max] values from
                which to uniformly sample
        f_low: Assigned lowest frequency of PTA (default assigns 1/(5*T_obs))
        f_high: Assigned highest frequency of PTA (default is Nyquist freq cadence/2)
        nfreqs: Number of frequencies in logspace the sensitivity is calculated
        '''
        self.name = name
        for keys,value in kwargs.items():
            if keys == 'load_location':
                self.Load_Data(value)
            elif keys == 'A_GWB':
                self.A_GWB = value
            elif keys == 'alpha_GWB':
                self.alpha_GWB = value
            elif keys == 'A_rn':
                self.A_rn_min = value[0]
                self.A_rn_max = value[1]
            elif keys == 'alpha_rn':
                self.alpha_rn_min = value[0]
                self.alpha_rn_max = value[1]
            elif keys == 'f_low':
                self.f_low = value
            elif keys == 'f_high':
                self.f_high = value
            elif keys == 'nfreqs':
                self.nfreqs = value

        if not hasattr(self,'nfreqs'):
            self.nfreqs = int(1e3)
        if hasattr(self,'f_low') and hasattr(self,'f_high'):
            self.fT = np.logspace(self.f_low,self.f_high,self.nfreqs)

        if len(args) != 0:
            [T_obs,N_p,sigma,cadence] = args
            self.T_obs = T_obs
            self.N_p = N_p
            self.sigma = sigma
            self.cadence = cadence

    @property
    def T_obs(self):
        self._T_obs = make_quant(self._T_obs,'yr')
        return self._T_obs
    @T_obs.setter
    def T_obs(self,value):
        self.var_dict = ['T_obs',value]
        self._T_obs = self._return_value

    @property
    def N_p(self):
        return self._N_p
    @N_p.setter
    def N_p(self,value):
        self.var_dict = ['N_p',value]
        self._N_p = self._return_value

    @property
    def cadence(self):
        self._cadence = make_quant(self._cadence,'1/yr')
        return self._cadence
    @cadence.setter
    def cadence(self,value):
        self.var_dict = ['cadence',value]
        self._cadence = self._return_value

    @property
    def sigma(self):
        self._sigma = make_quant(self._sigma,'s')
        return self._sigma
    @sigma.setter
    def sigma(self,value):
        self.var_dict = ['sigma',value]
        self._sigma = self._return_value

    @property
    def var_dict(self):
        return self._var_dict
    @var_dict.setter
    def var_dict(self,value):
        Get_Var_Dict(self,value)

    @property
    def fT(self):
        if not hasattr(self,'_fT'):
            #frequency sampled from 1/observation time to nyquist frequency (c/2)
            #5 is the default value for now (from Hazboun et al. 2019)
            T_obs_sec = self.T_obs.to('s').value
            cadence_sec = self.cadence.to('1/s').value
            self._fT = np.logspace(np.log10(1/(5*T_obs_sec)),np.log10(cadence_sec/2),self.nfreqs)
        self._fT = make_quant(self._fT,'Hz')
        return self._fT
    @fT.setter
    def fT(self,value):
        self._fT = value
    @fT.deleter
    def fT(self):
        del self._fT

    @property
    def h_n_f(self):
        #Effective Strain Noise Amplitude
        if not hasattr(self,'_h_n_f'):
            if not hasattr(self,'_sensitivitycurve'):
                self.Init_PTA()
            self._h_n_f = self._sensitivitycurve.h_c
        return self._h_n_f
    @h_n_f.setter
    def h_n_f(self,value):
        self._h_n_f = value
    @h_n_f.deleter
    def h_n_f(self):
        del self._h_n_f
    
    @property
    def S_n_f(self):
        #Effective noise power amplitude
        if not hasattr(self,'_S_n_f'):
            if not hasattr(self,'_sensitivitycurve'):
                self.Init_PTA()
            self._S_n_f = self._sensitivitycurve.S_eff
        self._S_n_f = make_quant(self._S_n_f,'1/Hz')
        return self._S_n_f
    @S_n_f.setter
    def S_n_f(self,value):
        self._S_n_f = value
    @S_n_f.deleter
    def S_n_f(self):
        del self._S_n_f

    @property
    def f_opt(self):
        #The optimal frequency of the instrument ie. the frequecy at the lowest strain
        if not hasattr(self,'_f_opt'):
            self._f_opt = self.fT[np.argmin(self.h_n_f)]
        return self._f_opt

    def Load_Data(self,load_location):
        self._I_data = np.loadtxt(load_location)
        self.fT = self._I_data[:,0]
        self.h_n_f = self._I_data[:,1]

    def Init_PTA(self):
        #### Using Jeff's Methods/code https://arxiv.org/abs/1907.04341

        #Random Sky Locations of Pulsars
        phi = np.random.uniform(0, 2*np.pi,size=self.N_p)
        cos_theta = np.random.uniform(-1,1,size=self.N_p)
        theta = np.arccos(cos_theta)

        if hasattr(self,'A_GWB'):
            if not hasattr(self,'alpha_GWB'):
                self.alpha_GWB = -2/3.
            #Make a set of psrs with the same parameters with a GWB as red noise
            psrs = hassim.sim_pta(timespan=self.T_obs.value,cad=self.cadence.value,sigma=self.sigma.value,\
                phi=phi, theta=theta, Npsrs=self.N_p,A_rn=self.A_GWB,alpha=self.alpha_GWB,freqs=self.fT.value)
        elif hasattr(self,'A_rn_min') or hasattr(self,'alpha_rn_min'):
            if not hasattr(self,'A_rn_min'):
                A_rn = np.random.uniform(1e-16,1e-12,size=self.N_p)
            else:
                A_rn = np.random.uniform(self.A_rn_min,self.A_rn_max,size=self.N_p)
            if not hasattr(self,'alpha_rn_min'):
                alphas = np.random.uniform(-3/4,1,size=self.N_p)
            else:
                alphas = np.random.uniform(self.alpha_rn_min,self.alpha_rn_max,size=self.N_p)            
            #Make a set of psrs with uniformly sampled red noise
            psrs = hassim.sim_pta(timespan=self.T_obs.value,cad=self.cadence.value,sigma=self.sigma.value,\
                phi=phi, theta=theta, Npsrs=self.N_p,A_rn=A_rn,alpha=alphas,freqs=self.fT.value)
        else:
            #Make a set of psrs with the same parameters
            psrs = hassim.sim_pta(timespan=self.T_obs.value,cad=self.cadence.value,sigma=self.sigma.value,\
                phi=phi, theta=theta, Npsrs=self.N_p,freqs=self.fT.value)
        #Get Spectra of pulsars
        spectra= []
        for p in psrs:
             sp = hassens.Spectrum(p,freqs=self.fT.value)
             spectra.append(sp)

        self._sensitivitycurve = hassens.DeterSensitivityCurve(spectra)




class GroundBased:
    '''
    Class to make a Ground Based Instrument
    Can only be read in from a file at this point
    '''
    def __init__(self,name,load_location,T_obs):
        '''
        name - the name of the instrument
        T_obs - the observation time of the Ground Instrument in [years]
        load_location - If you want to load a PTA curve from a file, 
                        it's the file path
        '''
        self.name = name
        self.T_obs = T_obs
        self._I_data = np.loadtxt(load_location)

    @property
    def T_obs(self):
        self._T_obs = make_quant(self._T_obs,'yr')
        return self._T_obs
    @T_obs.setter
    def T_obs(self,value):
        self.var_dict = ['T_obs',value]
        self._T_obs = self._return_value

    @property
    def var_dict(self):
        return self._var_dict
    @var_dict.setter
    def var_dict(self,value):
        Get_Var_Dict(self,value)

    @property
    def S_n_f(self):
        #Effective Noise Power Specral Density
        if not hasattr(self,'_S_n_f'):
            S_n_f_sqrt = self._I_data[:,1]
            self._S_n_f = S_n_f_sqrt**2
        self._S_n_f = make_quant(self._S_n_f,'1/Hz')
        return self._S_n_f
    @S_n_f.deleter
    def S_n_f(self):
        del self._S_n_f

    @property
    def fT(self):
        if not hasattr(self,'_fT'):
            self._fT = self._I_data[:,0]*u.Hz
        self._fT = make_quant(self._fT,'Hz')
        return self._fT
    @fT.deleter
    def fT(self):
        del self._fT
    
    @property
    def h_n_f(self):
        #Characteristic Strain/effective strain noise amplitude
        self._h_n_f = np.sqrt(self.fT*self.S_n_f)
        return self._h_n_f
    @h_n_f.deleter
    def h_n_f(self):
        del self._h_n_f

    @property
    def f_opt(self):
        #The optimal frequency of the instrument ie. the frequecy at the lowest strain
        self._f_opt = self.fT[np.argmin(self.h_n_f)]
        return self._f_opt




class SpaceBased:
    def __init__(self,name,*args,**kwargs):
        '''arg order: T_obs,L,A_acc,f_acc_break_low,f_acc_break_high,A_IFO,f_IMS_break'''
        '''
        name - name of the instrument
        args in order: 
        T_obs - the observation time of the PTA in [years]
        L - the armlength the of detector in [meters]
        A_acc - the Amplitude of the Acceleration Noise in [meters/second^2]
        f_acc_break_low = the lower break frequency of the acceleration noise in [Hz]
        f_acc_break_high = the higher break frequency of the acceleration noise in [Hz]
        A_IFO = the amplitude of the interferometer 
        kwargs can be:
        load_location: If you want to load a PTA curve from a file, 
                        it's the file path
        Background: Add in a Galactic Binary Confusion Noise
        f_low: Assigned lowest frequency of instrument (default assigns 10^-5Hz)
        f_high: Assigned highest frequency of instrument (default is 1Hz)
        nfreqs: Number of frequencies in logspace the sensitivity is calculated (default is 1e3)
        '''
        self.name = name
        for keys,value in kwargs.items():
            if keys == 'load_location':
                self.load_location = value
            elif keys == 'Background':
                self.Background = value
            elif keys == 'f_low':
                self.f_low = value
            elif keys == 'f_high':
                self.f_high = value
            elif keys == 'nfreqs':
                self.nfreqs = value
            elif keys == 'Tfunction_Type':
                self.Set_Tfunction_Type(value)
            elif keys == 'I_type':
                self.I_type = value

        if not hasattr(self,'nfreqs'):
            self.nfreqs = int(1e3)
        if not hasattr(self,'f_low'):
            self.f_low = 1e-5*u.Hz
        if not hasattr(self,'f_high'):
            self.f_high = 1.0*u.Hz
        if not hasattr(self,'Background'): 
            self.Background = False
        if hasattr(self,'load_location'):
            self.Load_Data()

        if len(args) != 0:
            [T_obs,L,A_acc,f_acc_break_low,f_acc_break_high,A_IFO,f_IMS_break] = args
            self.T_obs = T_obs
            self.L = L
            self.A_acc = A_acc
            self.f_acc_break_low = f_acc_break_low
            self.f_acc_break_high = f_acc_break_high
            self.A_IFO = A_IFO
            self.f_IMS_break = f_IMS_break

        if not hasattr(self,'_Tfunction_Type') and not hasattr(self,'load_location'):
            self.Set_Tfunction_Type('N')

    @property
    def T_obs(self):
        self._T_obs = make_quant(self._T_obs,'yr')
        return self._T_obs
    @T_obs.setter
    def T_obs(self,value):
        self.var_dict = ['T_obs',value]
        self._T_obs = self._return_value

    @property
    def L(self):
        self._L = make_quant(self._L,'m')
        return self._L
    @L.setter
    def L(self,value):
        self.var_dict = ['L',value]
        self._L = self._return_value

    @property
    def A_acc(self):
        self._A_acc = make_quant(self._A_acc,'m/(s*s)')
        return self._A_acc
    @A_acc.setter
    def A_acc(self,value):
        self.var_dict = ['A_acc',value]
        self._A_acc = self._return_value

    @property
    def f_acc_break_low(self):
        self._f_acc_break_low = make_quant(self._f_acc_break_low,'Hz')
        return self._f_acc_break_low
    @f_acc_break_low.setter
    def f_acc_break_low(self,value):
        self.var_dict = ['f_acc_break_low',value]
        self._f_acc_break_low = self._return_value

    @property
    def f_acc_break_high(self):
        self._f_acc_break_high = make_quant(self._f_acc_break_high,'Hz')
        return self._f_acc_break_high
    @f_acc_break_high.setter
    def f_acc_break_high(self,value):
        self.var_dict = ['f_acc_break_high',value]
        self._f_acc_break_high = self._return_value

    @property
    def A_IFO(self):
        self._A_IFO = make_quant(self._A_IFO,'m')
        return self._A_IFO
    @A_IFO.setter
    def A_IFO(self,value):
        self.var_dict = ['A_IFO',value]
        self._A_IFO = self._return_value

    @property
    def f_IMS_break(self):
        self._f_IMS_break = make_quant(self._f_IMS_break,'Hz')
        return self._f_IMS_break
    @f_IMS_break.setter
    def f_IMS_break(self,value):
        self.var_dict = ['f_IMS_break',value]
        self._f_IMS_break = self._return_value

    @property
    def var_dict(self):
        return self._var_dict
    @var_dict.setter
    def var_dict(self,value):
        Get_Var_Dict(self,value)

    @property
    def fT(self):
        if not hasattr(self,'_fT'):
            if hasattr(self,'_Tfunction_Type'):
                if self._Tfunction_Type == 'numeric':
                    self.Get_Numeric_Transfer_Function()
                if self._Tfunction_Type == 'analytic':
                    self.Get_Analytic_Transfer_Function()
            else:
                self.Set_Tfunction_Type()

        self._fT = make_quant(self._fT,'Hz')
        return self._fT
    @fT.setter
    def fT(self,value):
        self._fT = value
    @fT.deleter
    def fT(self):
        del self._fT

    @property
    def f_opt(self):
        #The optimal frequency of the instrument ie. the frequecy at the lowest strain
        self._f_opt = self.fT[np.argmin(self.h_n_f)]
        return self._f_opt

    @property
    def P_n_f(self):
        #Power Spectral Density
        if not hasattr(self,'_P_n_f'):
            if not hasattr(self,'_Tfunction_Type'):
                self.Set_Tfunction_Type()

            P_acc = self.A_acc**2*(1+(self.f_acc_break_low/self.fT)**2)*(1+(self.fT/(self.f_acc_break_high))**4)/(2*np.pi*self.fT)**4 #Acceleration Noise 
            P_IMS = self.A_IFO**2*(1+(self.f_IMS_break/self.fT)**4) #Displacement noise of the interferometric TM--to-TM 

            f_trans = const.c/2/np.pi/self.L #Transfer frequency
            self._P_n_f = (P_IMS + 2*(1+np.cos(self.fT.value/f_trans.value)**2)*P_acc)/self.L**2
        return self._P_n_f
    @P_n_f.deleter
    def P_n_f(self):
        del self._P_n_f

    @property
    def S_n_f(self):
        #Effective Noise Power Specral Density
        if not hasattr(self,'_S_n_f'):
            if hasattr(self,'_I_data'):
                if self._I_Type == 'ASD':
                    S_n_f_sqrt = self._I_data[:,1]
                    self._S_n_f = S_n_f_sqrt**2
                elif self._I_Type == 'ENSD':
                    self._S_n_f = self._I_data[:,1]
                elif self._I_Type == 'h':
                    self._S_n_f = self.h_n_f**2/self.fT
            else:
                S_n_f = self.P_n_f/self.transferfunction**2 
                if self.Background:
                    self._S_n_f= S_n_f+self.Add_Background() 
                else:
                    self._S_n_f = S_n_f
        self._S_n_f = make_quant(self._S_n_f,'1/Hz')
        return self._S_n_f
    @S_n_f.deleter
    def S_n_f(self):
        del self._S_n_f

    @property
    def h_n_f(self):
        #Characteristic Strain/effective strain noise amplitude
        if not hasattr(self,'_h_n_f'):
            if hasattr(self,'_I_data') and self._I_Type == 'h':
                self._h_n_f = self._I_data[:,1]
            else:
                self._h_n_f = np.sqrt(self.fT*self.S_n_f)
        return self._h_n_f
    @h_n_f.deleter
    def h_n_f(self):
        del self._h_n_f

    def Load_Data(self):
        if not hasattr(self,'I_type'):
            print('Is the data:')
            print(' *Effective Noise Spectral Density - "E"')
            print(' *Amplitude Spectral Density- "A"')
            print(' *Effective Strain - "h"')
            self.I_type = input('Please enter one of the answers in quotations: ')
            self.Load_Data()

        if self.I_type == 'E' or self.I_type == 'e':
            self._I_Type = 'ENSD'
        elif self.I_type == 'A' or self.I_type == 'a':
            self._I_Type = 'ASD'
        elif self.I_type == 'h' or self.I_type == 'H':
            self._I_Type = 'h'
        else:
            print('Is the data:')
            print(' *Effective Noise Spectral Density - "E"')
            print(' *Amplitude Spectral Density- "A"')
            print(' *Effective Strain - "h"')
            self.I_type = input('Please enter one of the answers in quotations: ')
            self.Load_Data()
        self._I_data = np.loadtxt(self.load_location)
        self.fT = self._I_data[:,0]

    def Load_TransferFunction(self):
        LISA_Transfer_Function_filedirectory = top_directory + '/LoadFiles/LISATransferFunction/'
        LISA_Transfer_Function_filename = 'transfer.dat' #np.loadtxting transfer function for Lisa noise curve
        LISA_Transfer_Function_filelocation = LISA_Transfer_Function_filedirectory + LISA_Transfer_Function_filename
        LISA_Transfer_Function_data = np.loadtxt(LISA_Transfer_Function_filelocation)
        self._transferfunctiondata = LISA_Transfer_Function_data

    def Get_Numeric_Transfer_Function(self):
        if not hasattr(self,'_transferfunctiondata'):
            self.Load_TransferFunction()

        fc = const.c/(2*self.L)  #light round trip freq
        LISA_Transfer_Function_f = fc*self._transferfunctiondata[:,0]

        idx_f_5 = np.abs(LISA_Transfer_Function_f-self.f_low).argmin()
        idx_f_1 = np.abs(LISA_Transfer_Function_f-self.f_high).argmin()

        #3/10 is normalization 2/5sin(openingangle)
        #Some papers use 3/20, not summing over 2 independent low-freq data channels
        self.transferfunction = np.sqrt(3/10)*self._transferfunctiondata[idx_f_5:idx_f_1,1]
        self.fT = LISA_Transfer_Function_f[idx_f_5:idx_f_1]

    def Get_Analytic_Transfer_Function(self):
        #Response function approximation from Calculation described by Cornish, Robson, Liu 2019
        if isinstance(self.f_low,u.Quantity) and isinstance(self.f_low,u.Quantity):
            self.fT = np.logspace(np.log10(self.f_low.value),np.log10(self.f_high.value),self.nfreqs)
        else:
            self.fT = np.logspace(np.log10(self.f_low),np.log10(self.f_high),self.nfreqs)
        f_L = const.c/2/np.pi/self.L #Transfer frequency
        #3/10 is normalization 2/5sin(openingangle)
        R_f = 3/10/(1+0.6*(self.fT/f_L)**2) 
        self.transferfunction = np.sqrt(R_f)

    def Set_Tfunction_Type(self,calc_type):
        if calc_type == 'n' or calc_type == 'N':
            self._Tfunction_Type = 'numeric'
        elif calc_type == 'a' or calc_type == 'A':
            self._Tfunction_Type = 'analytic'
        else:
            print('\nYou can get the transfer function via 2 methods:')
            print(' *To use the numerically approximated method in Robson, Cornish, and Liu, 2019, input "N".')
            print(' *To use the analytic fit in Larson, Hiscock, and Hellings, 2000, input "A".')
            calc_type = input('Please select the calculation type: ')
            self.Set_Tfunction_Type(calc_type)
        if hasattr(self,'_Tfunction_Type'):
            if self._Tfunction_Type == 'numeric':
                self.Get_Numeric_Transfer_Function()
            if self._Tfunction_Type == 'analytic':
                self.Get_Analytic_Transfer_Function()


    def Add_Background(self):
        '''
        Galactic confusions noise parameters for 6months, 1yr, 2yr, and 4yr
            corresponding to array index 0,1,2,3 respectively
        '''
        A = 9e-45
        a = np.array([0.133,0.171,0.165,0.138])
        b = np.array([243,292,299,-221])
        k = np.array([482,1020,611,521])
        g = np.array([917,1680,1340,1680])
        f_k = np.array([0.00258,0.00215,0.00173,0.00113])

        if self.T_obs < 1.*u.yr:
            index = 0
        elif self.T_obs >= 1.*u.yr and self.T_obs < 2.*u.yr:
            index = 1
        elif self.T_obs >= 2.*u.yr and self.T_obs < 4.*u.yr:
            index = 2
        else:
            index = 3
        f = self.fT.value
        return A*np.exp(-(f**a[index])+(b[index]*f*np.sin(k[index]*f)))\
                *(f**(-7/3))*(1 + np.tanh(g[index]*(f_k[index]-f))) #White Dwarf Background Noise




class BlackHoleBinary:
    def __init__(self,*args,**kwargs):
        '''args order: M,q,chi1,chi2,z,inc
            kwargs: f_low=1e-5,nfreqs=int(1e3)'''

        [M,q,chi1,chi2,z,inc] = args
        self.M = M
        self.q = q
        self.z = z
        self.chi1 = chi1
        self.chi2 = chi2
        self.inc = inc

        for keys,value in kwargs.items():
            if keys == 'f_low':
                self.f_low = value
            elif keys == 'f_high':
                self.f_high = value
            elif keys == 'nfreqs':
                self.nfreqs = value
        if not hasattr(self,'nfreqs'):
            self.nfreqs = int(1e3)
        if not hasattr(self,'f_low'):
            self.f_low = 1e-5*u.Hz

        self.Get_fitcoeffs()

    @property
    def M(self):
        self._M = make_quant(self._M,'M_sun')
        return self._M
    @M.setter
    def M(self,value):
        self.var_dict = ['M',value]
        self._M = self._return_value

    @property
    def q(self):
        return self._q
    @q.setter
    def q(self,value):
        self.var_dict = ['q',value]
        self._q = self._return_value

    @property
    def chi1(self):
        return self._chi1
    @chi1.setter
    def chi1(self,value):
        self.var_dict = ['chi1',value]
        self._chi1 = self._return_value

    @property
    def chi2(self):
        return self._chi2
    @chi2.setter
    def chi2(self,value):
        self.var_dict = ['chi2',value]
        self._chi2 = self._return_value

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self,value):
        self.var_dict = ['z',value]
        self._z = self._return_value

    @property
    def inc(self):
        return self._inc
    @inc.setter
    def inc(self,value):
        self.var_dict = ['inc',value]
        self._inc = self._return_value

    @property
    def h_gw(self):
        if not hasattr(self,'_h_gw'):
            self.h_gw = 'Averaged'
        return self._h_gw
    @h_gw.setter
    def h_gw(self,strain_const):
        if isinstance(strain_const,str):
            DL = cosmo.luminosity_distance(self.z)
            DL = DL.to('m')

            m_conv = const.G/const.c**3 #Converts M = [M] to M = [sec]

            eta = self.q/(1+self.q)**2
            M_redshifted_time = self.M.to('kg')*(1+self.z)*m_conv
            M_chirp = eta**(3/5)*M_redshifted_time
            #Source is emitting at one frequency (monochromatic)
            #strain of instrument at f_cw
            if strain_const == 'UseInc':
                #Strain from Rosado, Sesana, and Gair (2015) https://arxiv.org/abs/1503.04803
                #inc = 0.0 #optimally oriented
                a = 1+np.cos(self.inc)**2
                b = -2*np.cos(self.inc)
                const_val = 2*np.sqrt(.5*(a**2+b**2))
            elif strain_const == 'Hazboun':
                const_val = 4.
            elif strain_const == 'Averaged':
                #Strain from Robson et al. 2019 (eqn 27) https://arxiv.org/pdf/1803.01944.pdf
                #(ie. #(ie. sky and inclination averaged 4 * sqrt(4/5))
                const_val = 8/np.sqrt(5)
            else:
                raise ValueError('Can only use "UseInc" or "Averaged" monochromatic strain calculation.')

            self._h_gw = const_val*(const.c/DL)*(np.pi*self.f_init)**(2./3.)*M_chirp**(5./3.)
        else:
            raise ValueError('Can only use "UseInc" or "Averaged" monochromatic strain calculation.')
    @h_gw.deleter
    def h_gw(self):
        del self._h_gw

    @property
    def h_f(self):
        if not hasattr(self,'_h_f'):
            if not (hasattr(self,'_phenomD_f') and hasattr(self,'_phenomD_h')):
                self.Get_PhenomD_Strain()
            [self.f,self._h_f] = StrainConv(self,self._phenomD_f,self._phenomD_h)
        return self._h_f
    @h_f.setter
    def h_f(self,value):
        self._h_f = value
    @h_f.deleter
    def h_f(self):
        del self._h_f

    @property
    def f(self):
        if not hasattr(self,'_f'):
            if not (hasattr(self,'_phenomD_f') and hasattr(self,'_phenomD_h')):
                self.Get_PhenomD_Strain()
            [self._f,self.h] = StrainConv(self,self._phenomD_f,self._phenomD_h)
        self._f = make_quant(self._f,'Hz')
        return self._f
    @f.setter
    def f(self,value):
        self._f = value
    @f.deleter
    def f(self):
        del self._f
    
    @property
    def var_dict(self):
        return self._var_dict
    @var_dict.setter
    def var_dict(self,value):
        Get_Var_Dict(self,value)

    def Get_fitcoeffs(self):
        fit_coeffs_filedirectory = top_directory + '/LoadFiles/PhenomDFiles/'
        fit_coeffs_filename = 'fitcoeffsWEB.dat'
        fit_coeffs_file = fit_coeffs_filedirectory + fit_coeffs_filename
        self._fitcoeffs = np.loadtxt(fit_coeffs_file) #load QNM fitting files for speed later

    def Get_PhenomD_Strain(self):
        if not hasattr(self,'_fitcoeffs'):
            self.Get_fitcoeffs()

        Vars = [self.M.value,self.q,self.chi1,self.chi2,self.z]

        [self._phenomD_f,self._phenomD_h] = PhenomD.FunPhenomD(Vars,self._fitcoeffs,self.nfreqs,f_low=self.f_low.value)

    def checkFreqEvol(self,T_obs):
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
        m_conv = const.G/const.c**3 #Converts M = [M] to M = [sec]
        eta = self.q/(1+self.q)**2

        M_time = self.M.to('kg')*m_conv
        M_chirp_source = eta**(3/5)*M_time

        T_obs = make_quant(T_obs,'s')

        T_obs_source = T_obs/(1+self.z)
        #print('T_obs_source: ',T_obs_source.to('yr'))

        #Assumes t_init is in source frame, can either be randomly drawn
        #   or set to T_obs
        #t_init_source = np.random.uniform(0,100)*u.yr
        t_init_source = T_obs_source

        f_init_source = 1./8./np.pi/M_chirp_source*\
                        (5*M_chirp_source/t_init_source)**(3./8.)
        #print('f_init_source: ',f_init_source)
        
        self.f_init = f_init_source/(1+self.z)
        #print('f_init_inst: ',self.f_init)
        
        
        f_T_obs_source = 1./8./np.pi/M_chirp_source*\
                        (5*M_chirp_source/(t_init_source-T_obs_source))**(3./8.)
        #print('f_end_source: ',f_T_obs_source)
        
        self.f_T_obs = f_T_obs_source/(1+self.z)
        #print('f_T_obs_inst: ',self.f_T_obs)
        
        delf_obs_source_exact = f_T_obs_source-f_init_source
        #print('delf_source: ',delf_obs_source_exact)
        
        #from eqn 41 from Hazboun,Romano, and Smith (2019) https://arxiv.org/abs/1907.04341
        #Uses binomial expansion of f_T_obs_inst - f_init_inst
        #Will not ever be imaginary, so probably better to use
        delf_obs_source_approx = 1./8./np.pi/M_chirp_source*(5*M_chirp_source/t_init_source)**(3./8.)*(3*T_obs_source/8/t_init_source)
        #print('delf_Jeff: ',delf_obs_source_approx)
        
        delf_obs =  delf_obs_source_approx/(1+self.z)

        '''
        Old way I was doing this....
        M_redshifted_time = self.M.to('kg')*(1+self.z)*m_conv
        M_chirp = eta**(3/5)*M_redshifted_time
        t_init = 5*(M_chirp)**(-5/3)*(8*np.pi*self.f_init/(1+self.z))**(-8/3)
        #f(t) from eqn 40
        self.f_evolve = 1./8./np.pi/M_chirp*(5*M_chirp/(t_init-self.T_obs))**(3./8.)
        self.f_T_obs = 1./8./np.pi/M_chirp*(5*M_chirp/self.T_obs)**(3./8.)
        #from eqn 41 from Hazboun,Romano, and Smith (2019) https://arxiv.org/abs/1907.04341
        delf = 1./8./np.pi/M_chirp*(5*M_chirp/t_init)**(3./8.)*(3*self.T_obs/8/t_init)
        '''
        
        if delf_obs < (1/T_obs):
            self.ismono = True
        else:
            self.ismono = False

    


class TimeDomain:
    def __init__(self,name,*args):
        '''args order: M,q,z'''
        self.name = name

        if len(args) != 0:
            [M,q,z] = args
            self.M = M
            self.q = q
            self.z = z
        self.Load_Data()

    @property
    def M(self):
        self._M = make_quant(self._M,'M_sun')
        return self._M
    @M.setter
    def M(self,value):
        self.var_dict = ['M',value]
        self._M = self._return_value

    @property
    def q(self):
        return self._q
    @q.setter
    def q(self,value):
        self.var_dict = ['q',value]
        self._q = self._return_value

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self,value):
        self.var_dict = ['z',value]
        self._z = self._return_value

    @property
    def t(self):
        if not hasattr(self,'_t'):
            self._t = self._diff_data[:,0]
        self._t = make_quant(self._t,'s')
        return self._t

    @property
    def h_plus_t(self):
        if not hasattr(self,'_h_plus_t'):
            self._h_plus_t = self._diff_data[:,1]
        return self._h_plus_t

    @property
    def h_cross_t(self):
        if not hasattr(self,'_h_cross_t'):
            self._h_cross_t = self._diff_data[:,1]
        return self._h_cross_t

    @property
    def h_f(self):
        if not hasattr(self,'_h_f'):
            [natural_f,natural_h] = self.Get_hf_from_hcross_hplus()
            [_,self._h_f] = StrainConv(self,natural_f,natural_h)
        return self._h_f
    @h_f.setter
    def h_f(self,value):
        self._h_f = value
    @h_f.deleter
    def h_f(self):
        del self._h_f

    @property
    def f(self):
        if not hasattr(self,'_f'):
            [natural_f,natural_h] = self.Get_hf_from_hcross_hplus()
            [self._f,_] = StrainConv(self,natural_f,natural_h)
        return self._f
    @f.setter
    def f(self,value):
        self._f = value
    @f.deleter
    def f(self):
        del self._f  

    @property
    def var_dict(self):
        return self._var_dict
    @var_dict.setter
    def var_dict(self,value):
        Get_Var_Dict(self,value)
    
    def Load_Data(self):
        diff_filename = self.name + '.dat'
        diff_filelocation = top_directory + '/LoadFiles/DiffStrain/EOBdiff/' + diff_filename
        self._diff_data = np.loadtxt(diff_filelocation)


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
        natural_f = freqs[cut_low:cut_high]
        
        #Combine them for raw spectral power
        natural_h_f = np.sqrt((np.abs(h_cross_f))**2 + (np.abs(h_plus_f))**2)
        return [natural_f,natural_h_f]

def StrainConv(source,natural_f,natural_h):
    DL = cosmo.luminosity_distance(source.z)
    DL = DL.to('m')

    m_conv = const.G/const.c**3 #Converts M = [M] to M = [sec]
    M_redshifted_time = source.M.to('kg')*(1+source.z)*m_conv
    
    #frequency and strain of source in detector frame
    freq_conv = 1/M_redshifted_time
    #Normalized factor to match Stationary phase approx at low frequencies?
    #Changed from sqrt(5/16/pi)
    strain_conv = np.sqrt(1/4/np.pi)*(const.c/DL)*M_redshifted_time**2
    
    f = natural_f*freq_conv
    h_f = natural_h*strain_conv
    return [f,h_f]

def Get_CharStrain(source):
    if hasattr(source,'f') and hasattr(source,'h_f'):
        h_char = np.sqrt(4*source.f**2*source.h_f**2)
        return h_char
    else:
        raise ValueError('You need to get f and h_f first. \n')

def make_quant(param, default_unit):
    """
    Taken from https://github.com/Hazboun6/hasasia/blob/master/hasasia/sensitivity.py#L834
    Convenience function to intialize a parameter as an astropy quantity.
    param == parameter to initialize.
    default_unit == string that matches an astropy unit, set as
                    default for this parameter.
    returns:
        an astropy quantity
    example:
        self.f0 = make_quant(f0,'MHz')
    """
    default_unit = u.core.Unit(default_unit)
    if hasattr(param, 'unit'):
        try:
            param.to(default_unit)
        except u.UnitConversionError:
            raise ValueError("Quantity {0} with incompatible unit {1}"
                             .format(param, default_unit))
        quantity = param.to(default_unit)
    else:
        quantity = param * default_unit

    return quantity


def Get_Var_Dict(obj,value):
    if not hasattr(obj,'var_dict'):
            obj._var_dict = {}
    if isinstance(value,list):
        if len(value) == 2 and isinstance(value[0],str):
            var_name = value[0]
            vals = value[1]
            if isinstance(vals,list) and len(vals) == 3:
                if isinstance(vals[0],(float,int,u.Quantity))\
                 and isinstance(vals[1],(float,int,u.Quantity))\
                  and isinstance(vals[2],(float,int,u.Quantity)):
                    obj._return_value = vals[0]
                    obj._var_dict[var_name] = {'val':vals[0],'min':vals[1],'max':vals[2]}
                else:
                    raise ValueError(DictError_3())
            elif isinstance(vals,(float,int,u.Quantity)):
                if isinstance(vals,(float,int,u.Quantity)):
                    if var_name in obj._var_dict.keys():
                        obj._var_dict[var_name]['val'] = vals
                    else:
                        obj.var_dict[var_name] = {'val':vals,'min':None,'max':None}
                    obj._return_value = vals
                else:
                    raise ValueError(DictError_2())
        else:
            raise ValueError(DictError_Full())
    else:
        raise ValueError(DictError_Full())

def DictError_Full():
    return 'Must assign either: \n\
    - A name and value in a list (ie. ["name",val]), or \n\
    - A name, a value, a minimum value, and maximum value in a list (ie. ["name",val,min,max]), \n\
    where where name is a string, and val,min,and max are either floats, ints, or an astropy Quantity.'
def DictError_3():
    return 'Must assign a name, a value, a minimum value, and maximum value in a list (ie. ["name",val,min,max]), \n\
    where name is a string, and val, min, and max are either floats, ints, or astropy Quantities.'
def DictError_2():
    return 'Must assign a name and value in a list (ie. ["name",val]) \n\
    where name is a string, and val is either a float, an int, or an astropy Quantity.'