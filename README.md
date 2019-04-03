# DetectorDesignSensitivities
Generates characteristic strain plots and signal-to-noise (SNR) waterfall plots for any gravitational wave detector sensitivity in the frequency domain and for various binary black hole sources.

# StrainandNoise.py
Includes various functions to calculate: 
* Different GW detector sensitivities:
	* Pulsar Timing Arrays
	* Multiple LISA configurations
	* LISA response/transfer functions
* Binary Black Hole Source Strains:
	* Frequency evolving "chirping" sources
	* Monochromatic, non-evolving sources
* Various strain conversions

# SNRcalc.py
Includes various functions to calculate the SNR for any given GW detector sensitivity.
Uses the variable given and the data range to sample the space either logrithmically or linearly based on the 
selection of variables. Then it computes the SNR for each value.
Returns the variable ranges used to calculate the SNR for each matrix, then returns the SNRs with size of the sample1Xsample2
