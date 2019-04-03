import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.image import NonUniformImage

import os

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

#Selects contour levels to separate sections into
contLevels = np.array([10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
#contLevels = np.array([10, 1e2, 1e3, 1e4, 1e5])
logLevels = np.log10(contLevels)

contourcolorPresent = 'plasma'
transparencyPresent = 1.0
contourcolorFuture = 'plasma'
transparencyFuture = 0.6
axissize = 10
labelsize = 14
textsize = 11
textcolor1 = 'k'
textcolor2 = 'w'
linesize = 4
figsize=(10,5)

isitsavetime = False

#Save Figure Name
figname = 'UpdatedWaterfallPlots_Sarah_v5.pdf'

current_path = os.path.dirname(os.path.abspath(__file__))
splt_path = current_path.split("/")
top_path_idx = splt_path.index('lisaparameterization')
top_directory = "/".join(splt_path[0:top_path_idx+1])

lisa_filedirectory = top_directory + '/SNR/SNRdatafiles/ESA_lisa/'
et_filedirectory = top_directory + '/SNR/SNRdatafiles/EinsteinTelescope/'
aLIGO_filedirectory = top_directory + '/SNR/SNRdatafiles/aLIGO/'
nanograv_filedirectory = top_directory + '/SNR/SNRdatafiles/NanoGrav/'
SKA_filedirectory = top_directory + '/SNR/SNRdatafiles/SKA/'
fig_filedirectory = top_directory + '/SNR/PlottedSNR/'


lisa_SNR_filename = 'ESALisaSNRMatrix_zvM_correct_1.dat'
lisa_Samples_filename = 'ESALisaSamples_zvM_correct_1.dat'
lisa_SNR_filelocation = lisa_filedirectory+lisa_SNR_filename
lisa_Samples_filelocation = lisa_filedirectory+lisa_Samples_filename

#log of SNR from file
lisa_SNR = np.loadtxt(lisa_SNR_filelocation)

#z and M sample space corresponding to SNR height
#First column is redshift, second is total mass
lisa_Samples = np.loadtxt(lisa_Samples_filelocation)

#Take log of z, M, and SNR for plotting
lisa_logSamples = np.log10(lisa_Samples)
lisa_logSNR = np.log10(lisa_SNR)

#Einstein Telescope SNR
et_SNR_filename = 'ETSNRMatrix_zvM_correct_1.dat'
et_Samples_filename = 'ETSamples_zvM_correct_1.dat'
et_SNR_filelocation = et_filedirectory+et_SNR_filename
et_Samples_filelocation = et_filedirectory+et_Samples_filename
et_SNR = np.loadtxt(et_SNR_filelocation)
et_Samples = np.loadtxt(et_Samples_filelocation)
et_logSamples = np.log10(et_Samples)
et_logSNR = np.log10(et_SNR)

#aLIGO SNR
aLIGO_SNR_filename = 'aLIGOSNRMatrix_zvM_correct_1.dat'
aLIGO_Samples_filename = 'aLIGOSamples_zvM_correct_1.dat'
aLIGO_SNR_filelocation = aLIGO_filedirectory+aLIGO_SNR_filename
aLIGO_Samples_filelocation = aLIGO_filedirectory+aLIGO_Samples_filename
aLIGO_SNR = np.loadtxt(aLIGO_SNR_filelocation)
aLIGO_Samples = np.loadtxt(aLIGO_Samples_filelocation)
aLIGO_logSNR = np.log10(aLIGO_SNR)
aLIGO_logSamples = np.log10(aLIGO_Samples)

#NanoGrav SNR
#nanograv_SNR_filename = 'Mingarelli25yrNanoGravCWSNR.dat'
#nanograv_Samples_filename = 'Mingarelli25yrNanoGravCWSamples.dat'
#nanograv_SNR_filename = 'Vigeland11yrNanoGravCWSNR.dat'
#nanograv_Samples_filename = 'Vigeland11yrNanoGravCWSamples.dat'
nanograv_SNR_filename = 'NanogravSNRMatrix_zvM_scaled_1.dat'
nanograv_Samples_filename = 'NanogravSamples_zvM_scaled_1.dat'
nanograv_SNR_filelocation = nanograv_filedirectory+nanograv_SNR_filename
nanograv_Samples_filelocation = nanograv_filedirectory+nanograv_Samples_filename
nanograv_SNR = np.loadtxt(nanograv_SNR_filelocation)
nanograv_Samples = np.loadtxt(nanograv_Samples_filelocation)
nanograv_logSamples = np.log10(nanograv_Samples)
nanograv_logSNR = np.log10(nanograv_SNR)

#SKA SNR
SKA_SNR_filename = 'SKASNRMatrix_zvM_scaled_2.dat'
SKA_Samples_filename = 'SKASamples_zvM_scaled_2.dat'
SKA_SNR_filelocation = SKA_filedirectory+SKA_SNR_filename
SKA_Samples_filelocation = SKA_filedirectory+SKA_Samples_filename
SKA_SNR = np.loadtxt(SKA_SNR_filelocation)
SKA_Samples = np.loadtxt(SKA_Samples_filelocation)
SKA_logSamples = np.log10(SKA_Samples)
SKA_logSNR = np.log10(SKA_SNR)

nanograv_scale_M = nanograv_logSamples[1] - 1.6

SKA_scale_M = SKA_logSamples[1] - 1.8
SKA_scale_z = SKA_logSamples[0] + 0.3

'''
fig1,ax1 = plt.subplots();
plt.contour(SKA_logSamples[:,1],SKA_logSamples[:,0],SKA_logSNR,logLevels,color = 'black')
im1 = NonUniformImage(ax1,interpolation = 'bilinear', cmap = 'viridis',
                     extent=(SKA_logSamples[0,1], SKA_logSamples[len(SKA_logSamples)-1,1], SKA_logSamples[0,0], SKA_logSamples[len(SKA_logSamples)-1,0]),
                     norm = colors.Normalize(vmin= contLevels[0],vmax = contLevels[len(logLevels)-1]))
im1.set_data(SKA_logSamples[:,1],SKA_logSamples[:,0],np.log10(SKA_SNR))
ax1.images.append(im1)
ax1.set_xlim(SKA_logSamples[0,1], SKA_logSamples[len(SKA_logSamples)-1,1])
ax1.set_ylim(SKA_logSamples[0,0], SKA_logSamples[len(SKA_logSamples)-1,0])
'''

#########################
#Make the Contour Plots
fig1, ax1 = plt.subplots()
#figsize=(10,5)
#CS1 = ax1.contourf(nanograv_logSamples[1],nanograv_logSamples[0],nanograv_logSNR,logLevels,cmap = contourcolorPresent, alpha = transparencyPresent)
CS1 = ax1.contourf(nanograv_scale_M,nanograv_logSamples[0],nanograv_logSNR,logLevels,cmap = contourcolorPresent, alpha = transparencyPresent)
#ax1.contour(nanograv_logSamples[1],nanograv_logSamples[0],nanograv_logSNR,logLevels,colors = 'k')
ax1.contour(nanograv_scale_M,nanograv_logSamples[0],nanograv_logSNR,logLevels,colors = 'k')
ax1.contourf(aLIGO_logSamples[1],aLIGO_logSamples[0],aLIGO_logSNR,logLevels,cmap = contourcolorPresent, alpha = transparencyPresent)
ax1.contour(aLIGO_logSamples[1],aLIGO_logSamples[0],aLIGO_logSNR,logLevels,colors = 'k')
#ax1.contourf(SKA_logSamples[1],SKA_logSamples[0],SKA_logSNR,logLevels,cmap = contourcolorFuture, alpha = transparencyFuture)
ax1.contourf(SKA_scale_M,SKA_scale_z,SKA_logSNR,logLevels,cmap = contourcolorFuture, alpha = transparencyFuture)
ax1.contourf(lisa_logSamples[1],lisa_logSamples[0],lisa_logSNR,logLevels, cmap=contourcolorFuture, alpha = transparencyFuture)
ax1.contourf(et_logSamples[1],et_logSamples[0],et_logSNR,logLevels,cmap = contourcolorFuture, alpha = transparencyFuture)
ax1.scatter(np.log10(1e9),np.log10(0.03))
ax1.scatter(np.log10(1e10),np.log10(2.17))


#########################
#Set axes limits 
#ax1.set_xlim(et_logSamples[1,1],SKA_logSamples[len(SKA_logSamples)-1,1])
#ax1.set_ylim(aLIGO_logSamples[1,0],nanograv_logSamples[len(nanograv_logSamples)-1,0])
#ax1.set_xlim(aLIGO_logSamples[1,1],aLIGO_logSamples[len(aLIGO_logSamples)-1,1])
ax1.set_xlim(et_logSamples[0][0],11)
ax1.set_ylim(SKA_logSamples[1][0],SKA_logSamples[1][-1])


#plt.xscale('log')
#ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


Mlabel_min = 1
Mlabel_max = 11
zlabel_min = -2.0
zlabel_max = 3.0
zlabels = np.logspace(zlabel_min,zlabel_max,zlabel_max-zlabel_min+1)
Mlabels = range(Mlabel_min,Mlabel_max+1,1)

ages1 = np.array([13,10,5,1])*u.Gyr 
ages2 = np.array([500,100,10,1])*u.Myr
ages2 = ages2.to('Gyr')
ages = np.hstack((ages1.value,ages2.value))
ages = ages*u.Gyr
ageticks = [z_at_value(cosmo.age,age) for age in ages]

#########################
#Set ticks and labels
ax1.set_yticks(np.log10(zlabels))
ax1.set_yticklabels(zlabels,fontsize = axissize)
ax1.set_xticks(Mlabels)
ax1.set_xticklabels(Mlabels,fontsize = axissize)

ax1.set_xlabel(r'${\rm log}(M_{\rm tot})$ $[M_{\odot}]$',fontsize = labelsize)
ax1.set_ylabel('Redshift',fontsize = labelsize)
ax1.yaxis.set_label_coords(-.12,.5)

'''
ax2 = ax1.twinx()
ax2.set_yticks(np.log10(ageticks))
ax2.set_yticklabels(['{:g}'.format(age) for age in ages.value],fontsize = axissize)
ax2.set_ylabel(r'$t_{\rm cosmic}$ [Gyr]',fontsize=labelsize)
'''
#########################
#Set text labels inside plot

labelaLIGO_text = 'aLIGO\n(2016)'
labelaLIGO_xpos = 0.15
labelaLIGO_ypos = 0.1
plt.text(labelaLIGO_xpos,labelaLIGO_ypos,labelaLIGO_text,fontsize = textsize, horizontalalignment='center', color = textcolor2,transform = ax1.transAxes)

labelnanograv_text = 'NANOGrav\n(2018)'
labelnanograv_xpos = 0.875
labelnanograv_ypos = 0.15
plt.text(labelnanograv_xpos,labelnanograv_ypos,labelnanograv_text,fontsize = textsize, horizontalalignment='center',verticalalignment='center', color = textcolor2,transform = ax1.transAxes)

labelet_text = 'ET\n(~2030s)'
labelet_xpos = 0.12
labelet_ypos = 0.5
plt.text(labelet_xpos,labelet_ypos,labelet_text,fontsize = textsize, horizontalalignment='center',verticalalignment='center', color = textcolor1,transform = ax1.transAxes)

labelLisa_text = 'LISA\n(~2030s)'
labelLisa_xpos = 0.55
labelLisa_ypos = 0.1
plt.text(labelLisa_xpos,labelLisa_ypos,labelLisa_text,fontsize = textsize, horizontalalignment='center',verticalalignment='center', color = textcolor1,transform = ax1.transAxes)

labelIpta_text = 'IPTA\n(~2030s)'
labelIpta_xpos = 0.91
labelIpta_ypos = 0.88
plt.text(labelIpta_xpos,labelIpta_ypos,labelIpta_text,fontsize = textsize, horizontalalignment='center',verticalalignment='center', color = textcolor1,transform = ax1.transAxes)

#########################
#Make colorbar
#cbar1 = fig1.colorbar(CS1,ax=(ax1,ax2))
cbar1 = fig1.colorbar(CS1)
cbar1.set_label(r'${\rm log}(SNR)$',fontsize = labelsize)
cbar1.ax.tick_params(labelsize = axissize)

plt.show()

#########################
#Save Figure to File
figloc = fig_filedirectory+figname
if isitsavetime:
	fig1.savefig(figloc,bbox_inches='tight')