# DetectorDesignSensitivities
Generates characteristic strain plots and signal-to-noise (SNR) waterfall plots for any gravitational wave (GW) detector sensitivity in the frequency domain and for various binary black hole (BBH) sources.

## `StrainandNoise.py`
Includes various classes that instantiate different GW detector sensitivities and Source strains: 
* Different GW detector sensitivities:
	* Pulsar Timing Arrays (PTAs)
		* name - name of the instrument
        	* args in order: 
			* T_obs - the observation time of the PTA in [years]
			* N_p - the number of pulsars in the PTA 
			* sigma - the rms error on the pulsar TOAs in [sec]
			* cadence - How often the pulsars are observed in [num/year]
		* kwargs can be:
			* load_location: If you want to load a PTA curve from a file, 
					it's the file path
			* A_GWB: Amplitude of the gravitational wave background added as red noise
			* alpha_GWB: the GWB power law, if empty and A_GWB is set, it is assumed to be -2/3
			* A_rn: Individual pulsar red noise amplitude, is a list of [min,max] values from
				which to uniformly sample
			* alpha_rn: Individual pulsar red noise alpha (power law), is a list of [min,max] values from
				which to uniformly sample
			* f_low: Assigned lowest frequency of PTA (default assigns 1/(5*T_obs))
			* f_high: Assigned highest frequency of PTA (default is Nyquist freq cadence/2)
			* nfreqs: Number of frequencies in logspace the sensitivity is calculated
		
	* Space Based Detectors (eg. LISA)
		* name - name of the instrument
		* args in order: 
			* T_obs - the observation time of the PTA in [years]
			* L - the armlength the of detector in [meters]
			* A_acc - the Amplitude of the Acceleration Noise in [meters/second^2]
			* f_acc_break_low = the lower break frequency of the acceleration noise in [Hz]
			* f_acc_break_high = the higher break frequency of the acceleration noise in [Hz]
			* A_IFO = the amplitude of the interferometer 
		* kwargs can be:
			* load_location: If you want to load a PTA curve from a file, 
					it's the file path
			* Background: Add in a Galactic Binary Confusion Noise
			* f_low: Assigned lowest frequency of instrument (default assigns 10^-5Hz)
			* f_high: Assigned highest frequency of instrument (default is 1Hz)
			* nfreqs: Number of frequencies in logspace the sensitivity is calculated (default is 1e3)
			* Tfunction_Type - The transfer function method. Can be either 'N', or 'A'. 
				To use the numerically approximated method in Robson, Cornish, and Liu, 2019, input "N".
				To use the analytic fit in Larson, Hiscock, and Hellings, 2000, input "A".
			* I_type - Used in selecting the type of curve given in the load location file
	* Ground Based Detectors (eg. LIGO,ET)
		* Can only be loaded in via file, currently...
		* name - the name of the instrument
        	* T_obs - the observation time of the Ground Instrument in [years]
        	* load_location - If you want to load a PTA curve from a file, 
                        it's the file path
* Black Hole Binary (BHB) Source Strains:
	* args in order: 
		* M - Total mass of binary in solar masses
		* q - mass ratio, assumed to be m2/m1 > 1
		* chi1 - spin of m1
		* chi2 - spin of m2
		* z - redshift of source
		* inc - source inclination (isn't implemented yet, assumed to be averaged)
	* kwargs can be: 
		* f_low - low frequency of signal, assumed to be 1e-5 in geometrized units (Mf)
		* nfreqs - number of frequencies at which the waveform is calculated, assumed to be int(1e3)
	* Deals with both cases in one place!
		* Frequency evolving "chirping" sources
		* Monochromatic, non-evolving sources
* Time Domain Strains that can be converted to frequency space for use in SNR calculations
	* args in order: 
		* M - Total mass of binary in solar masses
		* q - mass ratio, assumed to be m2/m1 > 1
		* z - redshift of source
* Various utility functions

## `HorizonDistance.py`
**Currently being worked on**
* Calculates the Horizon Distance using similar methods to `SNRcalc.py`

## `SNRcalc.py`
Includes various functions to calculate the SNR for any given GW detector sensitivity using different methods for particular sources.

* __`getSNRMatrix(source,instrument,var_x,sampleRate_x,var_y,sampleRate_y)`__
	* Uses the variables given (`var_x`,`var_y`) and the their data range from `Get_Samples` in to sample the space either logrithmically or linearly based on the selection of variables. Then it uses `checkFreqEvol` to determine whether to use `calcMonoSNR` for a monochromatic source, or `calcChirpSNR` for evolving/chirping sources. It uses the aformentioned functions to compute the SNR for each value in the sample space. It returns the variable ranges used to calculate the SNR for each matrix and the SNRs with size of the sample_xXsample_y.
* __`calcMonoSNR`__
	* Uses a similar method to Robson,Cornish,and Liu 2018 (https://arxiv.org/abs/1803.01944) ~pg.9 to calculate the SNR for an non-evolving/monochromatic source.
* __`calcChirpSNR`__
	* Calculates evolving source using non-precessing binary black hole waveform model in `IMRPhenomD.py`, see Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253). Uses an interpolated method to align waveform and instrument noise, then integrates over the overlapping region. See eqn 18 from Robson,Cornish,and Liu 2018 (https://arxiv.org/abs/1803.01944).
* __`calcDiffSNR`__:
	* Calculates the SNR loss from the difference in EOB waveforms and numerical relativity. The strain is from Sean McWilliams in a private communication. Uses an interpolated method to align waveform and instrument noise, then integrates over the overlapping region.
* __`plotSNR`__
	* Generates the waterfall plot for the variables given to calcSNR 
## `IMRPhenomD.py`
Calculates evolving source using non-precessing binary black hole waveform model using the methods of Husa et al. 2016 (https://arxiv.org/abs/1508.07250) and Khan et al. 2016 (https://arxiv.org/abs/1508.07253).
* __`FunPhenomD(Vars,fitcoeffs,N,f_low=1e-9,pct_of_peak=0.01)`__
	* Generates the frequency domain amplitude of the inspiral-merger-ringdown signal from BBHs with mass ratio (q <= 18) and spins (|a/m|~0.85 or when q=1 |a/m|<0.98). The waveform appoximation method utilizes PN-expansions for the inspiral, fits to numerical relativity simulations for the merger and ringdown sections, and a connection region to join the two.
	* Takes in `Vars` to get the source mass ratio and spins,
    fitting coefficients for QNM type (`fitcoeffs`) , sampling rate (`N`), and includes options for an initial starting frequency of the inspiral in Mf (`f_low`) and ending frequency as a percent of the amplitude height at merger.
    * Returns the frequency, the Phenom amplitude of the inspiral-merger-ringdown in natural units (Mf,strain in M units)

