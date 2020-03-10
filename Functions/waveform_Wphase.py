import numpy as np

def Get_Waveform(source,pct_of_peak=0.01):
    """Uses Mass Ratio (q <= 18), aligned spins (abs(a/m)~0.85 or when q=1 abs(a/m)<0.98),
    fitting coefficients for QNM type, and sampling rate
    Returns the frequency, the Phenom amplitude of the inspiral-merger-ringdown
    Uses methods found in <https://arxiv.org/abs/1508.07253> and <https://arxiv.org/abs/1508.07250>

    Parameters
    ----------
    source : object
            source object from StrainandNoise, contains all source parameters
    pct_of_peak : float, optional
            the percentange of the strain at merger that dictates the maximum frequency the waveform is calculated at in geometrized units (G=c=1)

    Returns
    -------
    Mf : numpy array of floats
        the waveform frequencies in geometrized units (G=c=1)
    fullwaveform : numpy array of floats
        the waveform strain in geometrized units (G=c=1)

    """
    f_low = source.f_low
    N = source.nfreqs
    q = source.q
    x1 = source.chi1
    x2 = source.chi2
    fitcoeffs = source._fitcoeffs

    #M = m1+m2 #Total Mass
    #q = m2/m1 #Mass Ratio: Paper tested up to 18
    #eta = m1*m2/M**2 reduced mass: Paper tested up to 0.05 (q=18)
    eta = q/(q+1)**2
    x_PN = chi_PN(eta,x1,x2) #PN reduced spin parameter
    a_f = a_final(x1,x2,q,eta) #dimensionless spin

    ##################
    #Finds f_ringdown and f_damp from fit taken from <https://arxiv.org/abs/gr-qc/0512160>
    n = 0      #QNM indices
    l = 2
    m = 2
    numn = 3   #number of n's included in the table

    index = (l-2)*(2*l+1)*numn + (l-m)*numn + n
    f_fit = fitcoeffs[index][3:6]
    q_fit = fitcoeffs[index][6:9]

    omega_RD = f_fit[0]+f_fit[1]*(1-a_f)**f_fit[2]          #M omega_{lmn}
    tau = 2*(q_fit[0]+q_fit[1]*(1-a_f)**q_fit[2])/omega_RD  #tau_{lmn}/M = 2 Q_{lmn}/(M omega_{lmn})
    ########################
    f_RD = omega_RD/2/np.pi
    f_damp = 1/tau/2/np.pi

    Gamma1 = Lambda(eta,x_PN,4)
    Gamma2 = Lambda(eta,x_PN,5)
    Gamma3 = Lambda(eta,x_PN,6)

    f_peak = Calc_f_peak(f_RD,f_damp,[Gamma1,Gamma2,Gamma3])

    f1 = 0.014
    f3 = f_peak
    f2 = (f1+f3)/2

    f1_phase = 0.018
    f2_phase = 0.5*f_RD

    cutoffFreq = Find_Cutoff_Freq(f_RD,f_damp,[Gamma1,Gamma2,Gamma3],pct_of_peak=pct_of_peak)

    #If lowest frequency is greater than cutoffFreq, then raise error.
    if f_low >= cutoffFreq:
        raise ValueError('Lower frequency bound (ie. f_low) must be lower than that of the merger ringdown.')
    
    Mf = np.logspace(np.log10(f_low),np.log10(cutoffFreq),N)
    #Mf_phase = np.logspace(log10(0.0035),log10(1.15*f_RD),N)
    #Mf_phase = np.logspace(log10(0.0035),log10(0.12),N)

    v1 = A_insp(f1,eta,x1,x2,x_PN)
    v2 = Lambda(eta,x_PN,3)
    v3 = A_MR(f3,f_RD,f_damp,[Gamma1,Gamma2,Gamma3])
    fund1 = DA_insp(f1,eta,x1,x2,x_PN)
    fund3 = DA_MR(f3,f_RD,f_damp,[Gamma1,Gamma2,Gamma3])

    #############################
    #Calculate Solutions to eqn 21 in intermediate region
    Del_solns = A_intermediate(f1,f2,f3,v1,v2,v3,fund1,fund3) # Solutions to eqn 21

    ##############################
    #Calculate all sections of waveform and Paste together
    indxf1 = np.argmin(np.abs(Mf-f1))
    indxfpeak = np.argmin(np.abs(Mf-f_peak))

    tmpinspiral = A_norm(Mf[0:indxf1+1],eta)*A_insp(Mf[0:indxf1+1],eta,x1,x2,x_PN)
    tmpintermediate = A_norm(Mf[indxf1+1:indxfpeak],eta)*A_int(Mf[indxf1+1:indxfpeak],Del_solns)
    tmpmergerringdown = A_norm(Mf[indxfpeak:],eta)*A_MR(Mf[indxfpeak:],f_RD,f_damp,[Gamma1,Gamma2,Gamma3])
    fullwaveform = np.hstack((tmpinspiral,tmpintermediate,tmpmergerringdown))

    ##############################
    #Calculate all section of waveform Phase
    indxf1_phase = np.argmin(np.abs(Mf-f1_phase))
    indxf2_phase = np.argmin(np.abs(Mf-f2_phase))

    tc=0.0

    ##############################
    #Calculate Phase connections alpha0 and Beta0:
    dphi_ins = Dphi_ins(f1_phase,eta,x1,x2,x_PN,tc)
    phi_ins = Phi_ins(f1_phase,eta,x1,x2,x_PN,tc)

    beta1 = eta*dphi_ins - Dphi_int(f1_phase,eta,x_PN,0.0)
    beta0 = eta*phi_ins - Phi_int(f1_phase,eta,x_PN,beta1,0.0)
    alpha1 = Dphi_int(f2_phase,eta,x_PN,beta1) - Dphi_MR(f2_phase,eta,x_PN,f_RD,f_damp,0.0)
    alpha0 = Phi_int(f2_phase,eta,x_PN,beta1,beta0) - Phi_MR(f2_phase,eta,x_PN,f_RD,f_damp,alpha1,0.0)


    dinspiral_phase = Dphi_ins(Mf[:indxf1_phase+1],eta,x1,x2,x_PN,tc) 
    dintermediate_phase = (1/eta)*Dphi_int(Mf[indxf1_phase+1:indxf2_phase],eta,x_PN,beta1) 
    dmerger_ringdown_phase = (1/eta)*Dphi_MR(Mf[indxf2_phase:],eta,x_PN,f_RD,f_damp,alpha1) 


    inspiral_phase = Phi_ins(Mf[0:indxf1_phase+1],eta,x1,x2,x_PN,tc) 
    intermediate_phase = (1/eta)*Phi_int(Mf[indxf1_phase+1:indxf2_phase],eta,x_PN,beta1,beta0) 
    merger_ringdown_phase = (1/eta)*Phi_MR(Mf[indxf2_phase:],eta,x_PN,f_RD,f_damp,alpha1,alpha0) 

    ############################
    #Join subsections of phase and amplitude 
    fullphase = np.hstack((inspiral_phase,intermediate_phase,merger_ringdown_phase))

    return [Mf,fullwaveform,fullphase]

def A_norm(freqs,eta):
    """Calculates the constant scaling factor A_0

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio

    """
    const = np.sqrt(2*eta/3/np.pi**(1/3))
    return const*freqs**-(7/6)


def A_insp(freqs,eta,x1,x2,X_PN):
    """Calculates the Inspiral Amplitude

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    x_PN : float
        The PN reduced spin parameter

    """
    A_PN = 0.0
    A_higher = 0.0
    for i in range(7):
        A_PN = A_PN + PN_coeffs(eta,x1,x2,i)*(np.pi*freqs)**(i/3)
        if i >= 1 and i <= 3:
            A_higher = A_higher + Lambda(eta,X_PN,i-1)*freqs**((6+i)/3)
    return (A_PN + A_higher)


def DA_insp(freqs,eta,x1,x2,X_PN):
    """Calculates Derivative of the inspiral amplitude.

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    x_PN : float
        The PN reduced spin parameter

    """
    DA_PN = 0.0
    DA_higher = 0.0
    for i in range(7):
        PN_const = np.pi**(i/3)*(i/3)*PN_coeffs(eta,x1,x2,i)
        DA_PN = DA_PN + PN_const*(freqs)**((i-3)/3)
        if i >= 1 and i <= 3:
            higher_const = ((6+i)/3)*Lambda(eta,X_PN,i-1)
            DA_higher = DA_higher + higher_const*freqs**((i+3)/3)
        
    return DA_PN + DA_higher


def A_MR(freqs,f_RD,f_damp,Gammas):
    """Calculates the Normalized Merger-Ringdown Amplitude
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    Gammas : array-like
        Normalizes lorentzian to correct shape

    """
    varf = freqs-f_RD
    fg_d = Gammas[2]*f_damp 
    return (Gammas[0]*fg_d)/(varf**2+fg_d**2)*np.exp(-(Gammas[1]/fg_d)*varf)


def DA_MR(freqs,f_RD,f_damp,Gammas):
    """Calculates Derivative of the Merger-Ringdown Amplitude
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    Gammas : array-like
        Normalizes lorentzian to correct shape

    """
    varf = freqs-f_RD
    fg_d = Gammas[2]*f_damp
    A_MR_0 = A_MR(freqs,f_RD,f_damp,Gammas)
    return -A_MR_0*(2*varf/(varf**2+fg_d**2)+Gammas[1]/fg_d)


def A_intermediate(f1,f2,f3,v1,v2,v3,d1,d3):
    """Solves system of equations for intermediate amplitude matching"""
    Mat = np.array([[1., f1, f1**2, f1**3, f1**4],[1., f2, f2**2, f2**3, f2**4],[1., f3, f3**2, f3**3, f3**4], \
            [0., 1., 2*f1, 3*f1**2, 4*f1**3],[0., 1., 2*f3, 3*f3**2, 4*f3**3]],dtype='float')
    a = np.array([v1,v2,v3,d1,d3],dtype='float')
    return np.linalg.solve(Mat,a)


def A_int(freqs,delt):
    """Calculates the Intermediate Amplitude

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    delt : array
        Coefficient solutions to match the inspiral to the merger-ringdown portion of the waveform

    """
    return (delt[0]+delt[1]*freqs+delt[2]*freqs**2+delt[3]*freqs**3+delt[4]*freqs**4)

###########################################################################
#Phase portion of waveform
###########################################################################

def Phi_ins(freqs,eta,x1,x2,x_PN,t_c,phi_c,sigma0):
    """Calculates the Inspiral Phase

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    x_PN : float
        The PN reduced spin parameter
    t_c : float
        Coalescence time??
    """

    #t_c = 0.0  #????
    #phi_c = 0.0  #????????
    #sigma0 = 0.0  #???
    sigma1 = Lambda(eta,x_PN,7) 
    sigma2 = Lambda(eta,x_PN,8) 
    sigma3 = Lambda(eta,x_PN,9) 
    sigma4 = Lambda(eta,x_PN,10) 
    TF2_expansion = 0.0 
    TF2_const = 3/(128*eta) 
    
    piMf = np.pi*freqs 
    
    for i in range(5):
        #First four of summation, others need to be separate for log(pi*Mf) factors.
        TF2_expansion = TF2_expansion + PN_coeffs_phase(eta,x1,x2,i)*(piMf)**((i-5)/3) 
    
    TF2_expansion = (TF2_expansion + (1+np.log10(piMf))*PN_coeffs_phase(eta,x1,x2,5) +
            (PN_coeffs_phase(eta,x1,x2,6) - (6848/63)*np.log10(64*piMf))*(piMf)**(1/3) +
            PN_coeffs_phase(eta,x1,x2,7)*(piMf)**(2/3))
    
    phi_TF2 = 2*t_c*piMf - phi_c - np.pi/4 + TF2_const*TF2_expansion 
    return (phi_TF2 + (1/eta)*(sigma0 + sigma1*freqs + (3/4*sigma2)*freqs**(4/3) + (3/5*sigma3)*freqs**(5/3) + (.5*sigma4)*freqs**2))

def Dphi_ins(freqs,eta,x1,x2,x_PN,t_c):
    """Calculates the Inspiral Phase Derivative

    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    x_PN : float
        The PN reduced spin parameter
    t_c : float
        Coalescence time??
    """

    #t_c = -200.0  #????
    sigma1 = Lambda(eta,x_PN,7) 
    sigma2 = Lambda(eta,x_PN,8) 
    sigma3 = Lambda(eta,x_PN,9) 
    sigma4 = Lambda(eta,x_PN,10) 
    TF2_const_deriv = 3/(128*eta) 
    TF2_expans_deriv = 0.0 
    
    for i in range(5):
        TF2_expans_deriv = TF2_expans_deriv + (PN_coeffs_phase(eta,x1,x2,i)*np.pi**((i-5)/3)*((i-5)/3))*freqs**((i-8)/3) 
    
    Dphi_TF2 = (2*np.pi*t_c + TF2_const_deriv*(TF2_expans_deriv +
        PN_coeffs_phase(eta,x1,x2,4)/freqs +
        (PN_coeffs_phase(eta,x1,x2,5)-(6848/63)*(3+np.log10((64*np.pi)*freqs)))*(np.pi**(1/3)/3)*freqs**(-2/3) +
        ((2/3)*np.pi**(2/3)*PN_coeffs_phase(eta,x1,x2,6))*freqs**(-1/3)))
    return(Dphi_TF2 + (1/eta)*(sigma1 + sigma2*freqs**(1/3) + sigma3*freqs**(2/3) + sigma4*freqs))

def Phi_MR(freqs,eta,x_PN,f_RD,f_damp,alpha1,alpha0):
    """Calculates the Normalized Merger-Ringdown Phase
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x_PN : float
        The PN reduced spin parameter
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    alpha1 : float
        ??
    alpha0 : float
        ??

    """
    ###########################
    #Calculates phase for merger ringdown (eqn 14)
    alpha2 = Lambda(eta,x_PN,15) 
    alpha3 = Lambda(eta,x_PN,16) 
    alpha4 = Lambda(eta,x_PN,17) 
    alpha5 = Lambda(eta,x_PN,18) 
    lorentzian = (freqs - alpha5*f_RD)/f_damp 
    
    return (alpha0 + alpha1*freqs - alpha2/freqs + (4./3.*alpha3)*freqs**(.75) + alpha4*np.arctan(lorentzian))

def Dphi_MR(freqs,eta,x_PN,f_RD,f_damp,alpha1):
    """Calculates Derivative of the Merger-Ringdown Phase Derivative
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x_PN : float
        The PN reduced spin parameter
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    alpha1 : float
        ??

    """
    ###########################
    #Calculates derivative of Phi_MR used to match regions should be eqn(13)/eta
    alpha2 = Lambda(eta,x_PN,15) 
    alpha3 = Lambda(eta,x_PN,16) 
    alpha4 = Lambda(eta,x_PN,17) 
    alpha5 = Lambda(eta,x_PN,18) 
    lorentzian = (alpha4*f_damp)/(f_damp**2 + (freqs - alpha5*f_RD)**2) 
    
    return (alpha1 + alpha2/freqs**2 + alpha3*freqs**(-.25) + lorentzian)

def Phi_int(freqs,eta,x_PN,beta1,beta0):
    """Calculates Derivative of the Intermediate Phase
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x_PN : float
        The PN reduced spin parameter
    beta1 : float
        ??
    beta0 : float
        ??

    """
    beta2 = Lambda(eta,x_PN,12) 
    beta3 = Lambda(eta,x_PN,13) 
    
    return (beta0 + beta1*freqs + beta2*np.log10(freqs) - (beta3/3.)*freqs**-3.)

def Dphi_int(freqs,eta,x_PN,beta1):
    """Calculates Derivative of the Intermediate Phase Derivative
    
    Parameters
    ----------
    freqs : array
        The frequencies in Natural units (Mf, G=c=1) of the waveform
    eta : float
        The reduced mass ratio
    x_PN : float
        The PN reduced spin parameter
    beta1 : float
        ??
    """
    beta2 = Lambda(eta,x_PN,12) 
    beta3 = Lambda(eta,x_PN,13) 
    
    return (beta1 + beta2/freqs + beta3*freqs**-4)


def Lambda(eta,x_PN,lmbda):
    """Gets the Lambdas from Eqn 31 in <https://arxiv.org/abs/1508.07253>

    Parameters
    ----------
    eta : float
        The reduced mass ratio
    x_PN : float
        The PN reduced spin parameter
    lmbda : int
        Iterator for different Lambda variables using the zeta function

    """
    xi = x_PN-1
    xi2 = xi*xi
    xi3 = xi2*xi
    eta2 = eta*eta
    if lmbda == 0:  #rho1
        coeffs = zeta(0)
    elif lmbda == 1: #rho2
        coeffs = zeta(1)
    elif lmbda == 2: #rho3
        coeffs = zeta(2)
    elif lmbda == 3: #v2
        coeffs = zeta(3)
    elif lmbda == 4: #gamma1
        coeffs = zeta(4)
    elif lmbda == 5: #gamma2
        coeffs = zeta(5)
    elif lmbda == 6: #gamma3
        coeffs = zeta(6)
    elif lmbda == 7: #sigma1
        coeffs = zeta(7) 
    elif lmbda == 8: #sigma2
        coeffs = zeta(8) 
    elif lmbda == 9: #sigma3
        coeffs = zeta(9) 
    elif lmbda == 10: #sigma4
        coeffs = zeta(10) 
    elif lmbda == 11: #Beta1
        coeffs = zeta(11) 
    elif lmbda == 12: #Beta2
        coeffs = zeta(12) 
    elif lmbda == 13: #Beta3
        coeffs = zeta(13) 
    elif lmbda == 14: #alpha1
        coeffs = zeta(14) 
    elif lmbda == 15: #alpha2
        coeffs = zeta(15) 
    elif lmbda == 16: #alpha3
        coeffs = zeta(16) 
    elif lmbda == 17: #alpha4
        coeffs = zeta(17) 
    elif lmbda == 18: #alpha5
        coeffs = zeta(18) 
    
    return coeffs[0] + coeffs[1]*eta + \
        (coeffs[2] + coeffs[3]*eta + coeffs[4]*eta2)*xi + \
        (coeffs[5] + coeffs[6]*eta + coeffs[7]*eta2)*xi2 + \
        (coeffs[8] + coeffs[9]*eta + coeffs[10]*eta2)*xi3


def zeta(k):
    """Coefficients in table 5 of <https://arxiv.org/abs/1508.07253>"""
    if k == 0: #rho 1
        coeffs = [3931.9, -17395.8, 3132.38, 343966.0, -1.21626e6, -70698.0, 1.38391e6, -3.96628e6, -60017.5, 803515.0, -2.09171e6]
    elif k == 1: #rho 2
        coeffs = [-40105.5, 112253.0, 23561.7, -3.47618e6, 1.13759e7, 754313.0, -1.30848e7, 3.64446e7, 596227.0, -7.42779e6, 1.8929e7]
    elif k == 2: #rho 3
        coeffs = [83208.4, -191238.0, -210916.0, 8.71798e6, -2.69149e7, -1.98898e6, 3.0888e7, -8.39087e7, -1.4535e6, 1.70635e7, -4.27487e7]
    elif k == 3: #v 2
        coeffs = [0.814984, 2.57476, 1.16102, -2.36278, 6.77104, 0.757078, -2.72569, 7.11404, 0.176693, -0.797869, 2.11624]
    elif k == 4: #gamma 1
        coeffs = [0.0069274, 0.0302047, 0.00630802, -0.120741, 0.262716, 0.00341518, -0.107793, 0.27099, 0.000737419, -0.0274962, 0.0733151]
    elif k == 5: #gamma 2
        coeffs = [1.01034, 0.000899312, 0.283949, -4.04975, 13.2078, 0.103963, -7.02506, 24.7849, 0.030932, -2.6924, 9.60937]
    elif k == 6: #gamma 3
        coeffs = [1.30816, -0.00553773, -0.0678292, -0.668983, 3.40315, -0.0529658, -0.992379, 4.82068, -0.00613414, -0.384293, 1.75618]
    elif k ==  7: #sigma1
        coeffs = [2096.55, 1463.75, 1312.55, 18307.3, -43534.1, -833.289, 32047.3, -108609.0, 452.251, 8353.44, -44531.3] 
    elif k ==  8: #sigma2
        coeffs = [-10114.1, -44631.0, -6541.31, -266959.0, 686328.0, 3405.64, -437508.0, 1.63182e6, -7462.65, -114585.0, 674402.0] 
    elif k ==  9: #sigma3
        coeffs = [22933.7, 230960.0, 14961.1, 1.19402e6, -3.10422e6, -3038.17, 1.87203e6, -7.30915e6, 42738.2, 467502.0, -3.06485e6] 
    elif k ==  10: #sigma4
        coeffs = [-14621.7, -377813, -9608.68, -1.71089e6, 4.33292e6, -22366.7, -2.50197e6, 1.02745e7, -85360.3, -570025.0, 4.39684e6] 
    elif k ==  11: #Beta1
        coeffs = [97.8975, -42.6597, 153.484, -1417.06, 2752.86, 138.741, -1433.66, 2857.74, 41.0251, -423.681, 850.359] 
    elif k ==  12: #Beta2
        coeffs = [-3.2827, -9.05138, -12.4154, 55.4716, -106.051, -11.953, 76.807, -155.332, -3.41293, 25.5724, -54.408] 
    elif k ==  13: #Beta3
        coeffs = [-2.51564e-5, 1.97503e-5, -1.83707e-5, 2.18863e-5, 8.25024e-5, 7.15737e-6, -5.578e-5, 1.91421e-4, 5.44717e-6, -3.22061e-5, 7.97402e-5] 
    elif k ==  14: #alpha1
        coeffs = [43.3151, 638.633, -32.8577, 2415.89, -5766.88, -61.8546, 2953.97, -8986.29, -21.5714, 981.216, -3239.57] 
    elif k ==  15: #alpha2
        coeffs = [-0.0702021, -0.162698, -0.187251, 1.13831, -2.83342, -0.17138, 1.71975, -4.53972, -0.0499834, 0.606207, -1.68277] 
    elif k ==  16: #alpha3
        coeffs = [9.59881, -397.054, 16.2021, -1574.83, 3600.34, 27.0924, -1786.48, 5152.92, 11.1757, -577.8, 1808.73] 
    elif k ==  17: #alpha4
        coeffs = [-0.0298949, 1.40221, -0.0735605, 0.833701, 0.224001, -0.0552029, 0.566719, 0.718693, -0.0155074, 0.157503, 0.210768] 
    elif k ==  18: #alpha5
        coeffs = [0.997441, -0.00788445, -0.0590469, 1.39587, -4.51663, -0.0558534, 1.75166, -5.99021, -0.0179453, 0.59651, -2.06089] 
    return coeffs


def PN_coeffs(eta,x1,x2,i):
    """Gets the PN Amplitude coefficients

    Parameters
    ----------
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    q : float
        The mass ratio m1/m2, m1<=m2
    i : int
        iterator to dictate which PN Amplitude to use

    Notes
    -----
    Coefficients in appendix B (eqns B14-B20) of <https://arxiv.org/abs/1508.07253>

    """
    delta = np.sqrt(1.0-4.0*eta)
    chi_s = (x1+x2)/2.0
    chi_a = (x1-x2)/2.0
    if i == 0:
        A_i = 1
    elif i == 1:
        A_i = 0
    elif i == 2:
        A_i = -323/224 + (451/168)*eta
    elif i == 3:
        A_i = (27/8)*delta*chi_a + (27/8 -(11/6)*eta)*chi_s
    elif i == 4:
        A_i = -27312085/8128512 -(1975055/338688)*eta + (105271/24192)*eta**2 + \
            (-81/32+8*eta)*chi_a**2 - 81/16*delta*chi_a*chi_s + (-81/32+17/8*eta)*chi_s**2
    elif i == 5:
        A_i = -85*np.pi/64 + 85*np.pi/16*eta + (285197/16128-1579/4032*eta)*delta*chi_a + \
            (285197/16128 - 15317/672*eta - 2227/1008*eta**2)*chi_s
    elif i == 6:
        A_i = -177520268561/8583708672 + (545384828789/5007163392 - 205*np.pi**2/48)*eta - \
            3248849057/178827264*eta**2 + 34473079/6386688*eta**3 + (1614569/64512-1873643/16128*eta+2167/42*eta**2)*chi_a**2 + \
            (31*np.pi/12 - 7*np.pi/3*eta)*chi_s + (1614569/64512-61391/1344*eta+57451/4032*eta**2)*chi_s**2 + \
            delta*chi_a*(31*np.pi/12+(1614569/32256-165961/2688*eta)*chi_s)
    return A_i

def PN_coeffs_phase(eta,x1,x2,i):
    """Gets the PN Phase coefficients

    Parameters
    ----------
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    q : float
        The mass ratio m1/m2, m1<=m2
    i : int
        iterator to dictate which PN Amplitude to use

    Notes
    -----
    Coefficients in appendix B (eqns B7-B13) of <https://arxiv.org/abs/1508.07253>

    """
    delta = np.sqrt(1.0-4.0*eta)
    chi_s = (x1+x2)/2.0
    chi_a = (x1-x2)/2.0
    eulergamma =  0.5772156649 #gamma_E = Euler-Mascheroni constant?
    if i == 0:
        phi_i = 1
    elif i == 1:
        phi_i = 0
    elif i == 2:
        phi_i = 3715/756 + eta*55/9
    elif i == 3:
        phi_i = -16*np.pi + (113/3)*delta*chi_a + (113/3-(76/3)*eta)*chi_s
    elif i == 4:
        phi_i = 15293365/508032 + (27145/504)*eta + (3085/72)*eta**2 + (-405/8+200*eta)*chi_a**2 - (405/4)*delta*chi_a*chi_s + (-405/8+(5/2)*eta)*chi_s**2
    elif i == 5:
        #Moved (1+log(pi*Mf) to the summation
        phi_i = 38645*np.pi/756 - (65*np.pi/9)*eta + delta*(-732985/2268 - (140/9)*eta)*chi_a + (-732985/2268 + (24260/81)*eta + (340/9)*eta**2)*chi_s
    elif i == 6:
        #Moved -(6848/63)*log(64*pi*Mf) to the summation
        phi_i = (11583231236531/4694215680 - 6848*eulergamma/21 - 640*np.pi**2/3 + (-15737765635/3048192 + 2255*np.pi**2/12)*eta + (76055/1728)*eta**2 - (127825/1296)*eta**3 +
        (2270/3*np.pi)*delta*chi_a + (2270*np.pi/3 - 520*np.pi*eta)*chi_s)
    elif i == 7:
        phi_i = (77096675*np.pi/254016 + (378515/1512*np.pi)*eta - (74045/756*np.pi)*eta**2 + delta*(-25150083775/3048192 + (26804935/6048)*eta - (1985/48)*eta**2)*chi_a +
            (-25150083775/3048192 + (10566655595/762048)*eta - (1042165/3024)*eta**2 + (5345/36)*eta**3)*chi_s)
    return phi_i

def Calc_f_peak(f_RD,f_damp,Gammas):
    """Calculates the frequency at the peak of the merger

    Parameters
    ----------
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    Gammas : array-like
        Normalizes lorentzian to correct shape
    
    Notes
    -----
    There is a problem with this expression from the paper becoming imaginary if gamma2 >= 1 
    so if gamma2 >= 1 then set the square root term to zero.

    """
    if Gammas[1] <= 1:
        f_max = np.abs(f_RD+f_damp*Gammas[2]*(np.sqrt(1-Gammas[1]**2)-1)/Gammas[1])
    else:
        f_max = np.abs(f_RD+(f_damp*Gammas[2]*-1)/Gammas[1])
    return f_max


def Find_Cutoff_Freq(f_RD,f_damp,Gammas,pct_of_peak=0.0001):
    """Cutoff signal when the amplitude is a factor of 10 below the value at f_RD

    Parameters
    ----------
    f_RD : float
        Frequency of the Ringdown transition
    f_damp : float
        Damping frequency
    Gammas : array-like
        Normalizes lorentzian to correct shape

    pct_of_peak : float, optional
        the percentange of the strain at merger that dictates the maximum 
        frequency the waveform is calculated at in geometrized units (G=c=1) 

    """
    tempfreqs = np.logspace(np.log10(f_RD),np.log10(10*f_RD),100)
    cutoffAmp = pct_of_peak*A_MR(f_RD,f_RD,f_damp,[Gammas[0],Gammas[1],Gammas[2]])
    merger_ringdown_Amp = A_MR(tempfreqs,f_RD,f_damp,[Gammas[0],Gammas[1],Gammas[2]])
    cutoffindex = np.argmin(np.abs(cutoffAmp-merger_ringdown_Amp))
    return tempfreqs[cutoffindex]

def a_final(x1,x2,q,eta):
    """The Final spin of the binary remnant black hole

    Parameters
    ----------
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.
    q : float
        The mass ratio m1/m2, m1<=m2
    eta : float
        The reduced mass ratio

    Notes
    -----
    Uses eq. 3 in <https://arxiv.org/abs/0904.2577>, changed to match our q convention
    a=J/M**2 where J = x1*m1**2 + x2*m2**2

    """
    a = (q**2*x1+x2)/(q**2+1)
    s4 = -0.1229
    s5 = 0.4537
    t0 = -2.8904
    t2 = -3.5171
    t3 = 2.5763
    return a + s4*a**2*eta + s5*a*eta**2 + t0*a*eta + 2*np.sqrt(3)*eta + t2*eta**2 + t3*eta**3


def chi_PN(eta,x1,x2):
    """Calculates the PN reduced spin parameter

    Parameters
    ----------
    eta : float
        The reduced mass ratio
    x1 : float
        The dimensionless spin parameter abs(a/m) for black hole m1.
    x2 : float
        The dimensionless spin parameter abs(a/m) for black hole m2.

    Notes
    -----
    See Eq 5.9 in <https://arxiv.org/abs/1107.1267v2>

    """
    delta = np.sqrt(1.0-4.0*eta)
    chi_s = (x1+x2)/2.0
    chi_a = (x1-x2)/2.0
    return chi_s*(1.0-eta*76.0/113.0) + delta*chi_a