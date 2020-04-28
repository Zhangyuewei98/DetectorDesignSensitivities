import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.constants import golden_ratio

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo

def Plot_SNR(fig,ax,var_x,sample_x,var_y,sample_y,SNRMatrix,
             display=True,dl_axis=False,smooth_contours=False,
             display_cbar=False,x_axis_label=True,y_axis_label=True,
             logLevels_min=-1.0,logLevels_max=0.0,
             hspace=0.15,wspace=0.1,y_label_loc = -.25,rotate=False,
             alpha=1.0,cf_alpha = 1.0,cline_color='k',cfill=True,
             cline_style='-'):
    """Plots the SNR contours from calcSNR

    Parameters
    ----------
    var_x : str
        x-axis variable
    sample_x : array
        samples at which SNRMatrix was calculated corresponding to the x-axis variable
    var_y : str
        y-axis variable
    sample_y : array
        samples at which SNRMatrix was calculated corresponding to the y-axis variable
    SNRMatrix : array-like
        the matrix at which the SNR was calculated corresponding to the particular x and y-axis variable choices

    display : bool, optional
        Option to turn off display if saving multiple plots to a file
    dl_axis : bool, optional
        Option to turn on the right hand side labels of luminosity distance
    smooth_contours : bool, optional
        Option to interpolate contours to a finer mesh size to appear smooth instead of tiered contours

    """
    """
    for key,value in kwargs.items():
        if key == 'display':
            display=value
        elif key == 'dl_axis':
            dl_axis = value
        elif key == 'smooth_contours':
            smooth_contours = value
        elif key == 'display_cbar':
            display_cbar = value
        elif key == 'x_axis_label':
            x_axis_label = value
        elif key == 'y_axis_label':
            y_axis_label = value
        elif key == 'logLevels_min':
            logLevels_min = value
        elif key == 'logLevels_max':
            logLevels_max = value
        elif key == 'hspace':
            hspace = value
        elif key == 'wspace':
            wspace = value
        elif key == 'y_label_loc':
            y_label_loc = value
        elif key == 'rotate':
            rotate = value
        elif key == 'alpha':
            alpha = value
        elif key == 'cf_alpha':
            cf_alpha = value
        elif key == 'cline_color':
            cline_color = value
        else:
            raise ValueError("%s, is not an accepted kwarg" %key)
        """
    textcolor = 'k'
    linewidth = 2.0
    colormap = 'viridis'

    logSNR = np.log10(SNRMatrix)
    if logLevels_min == -1.0:
        logLevels_min = np.log10(np.array([1.]))
    if logLevels_max == 0.0:
        logLevels_max = np.ceil(np.amax(logSNR))
    if logLevels_max < logLevels_min:
        raise ValueError('All SNRs are lower than 5.')
        
    logLevels_add = np.log10(np.array([3.,10.,31.]))
    print_logLevels = np.concatenate((logLevels_min,logLevels_add,np.arange(2.,logLevels_max+1.)))
    if smooth_contours:
        logLevels = np.linspace(logLevels_min,logLevels_max,100)[:,0]
    else:
        logLevels = print_logLevels
                
    ylabel_min = min(sample_y)
    ylabel_max = max(sample_y)
    xlabel_min = min(sample_x)
    xlabel_max = max(sample_x)

    #Set whether log or linearly spaced axes
    x_scale = np.log10(xlabel_max) - np.log10(xlabel_min)
    if x_scale >= 2.:
        xaxis_type = 'log'
    else:
        xaxis_type = 'lin'

    y_scale = np.log10(ylabel_max) - np.log10(ylabel_min)
    if y_scale >= 2.:
        yaxis_type = 'log'
    else:
        yaxis_type = 'lin'

    #Set axis scales based on what data sampling we used 
    if yaxis_type == 'lin' and xaxis_type == 'log':
        if cfill == False:
            CS1 = ax.contour(np.log10(sample_x),sample_y,logSNR,print_logLevels,alpha=alpha,cmap=colormap,linestyles=cline_style,linewidths=linewidth)
        else:
            CS1 = ax.contourf(np.log10(sample_x),sample_y,logSNR,logLevels,cmap = colormap,alpha=cf_alpha)
            ax.contour(np.log10(sample_x),sample_y,logSNR,print_logLevels,colors = cline_color,alpha=alpha)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.linspace(ylabel_min,ylabel_max,ylabel_max-ylabel_min+1)
        ax.set_yticks(y_labels)
        ax.set_xticks(np.log10(x_labels))
    elif yaxis_type == 'log' and xaxis_type == 'lin':
        if cfill == False:
            CS1 = ax.contour(sample_x,np.log10(sample_y),logSNR,print_logLevels,alpha=alpha,cmap=colormap,linestyles=cline_style,linewidths=linewidth)
        else:
            CS1 = ax.contourf(sample_x,np.log10(sample_y),logSNR,logLevels,cmap = colormap,alpha=cf_alpha)
            ax.contour(sample_x,np.log10(sample_y),logSNR,print_logLevels,colors = cline_color,alpha=alpha)
        ax.set_xlim(xlabel_min,xlabel_max)
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.linspace(xlabel_min,xlabel_max,xlabel_max-xlabel_min+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_xticks(x_labels)
        ax.set_yticks(np.log10(y_labels))
    elif yaxis_type == 'lin' and xaxis_type == 'lin':
        if cfill == False:
            CS1 = ax.contour(sample_x,sample_y,logSNR,print_logLevels,alpha=alpha,cmap=colormap,linestyles=cline_style,linewidths=linewidth)
        else:
            CS1 = ax.contourf(sample_x,sample_y,logSNR,logLevels,cmap = colormap,alpha=cf_alpha)
            ax.contour(sample_x,sample_y,logSNR,print_logLevels,colors = cline_color,alpha=alpha)
        ax.set_xlim(xlabel_min,xlabel_max)
        ax.set_ylim(ylabel_min,ylabel_max)
        x_labels = np.linspace(xlabel_min,xlabel_max,xlabel_max-xlabel_min+1)
        y_labels = np.linspace(ylabel_min,ylabel_max,ylabel_max-ylabel_min+1)
        ax.set_xticks(x_labels)
        ax.set_yticks(y_labels)
    else:
        if cfill == False:
            CS1 = ax.contour(np.log10(sample_x),np.log10(sample_y),logSNR,print_logLevels,alpha=alpha,cmap=colormap,linestyles=cline_style,linewidths=linewidth)
        else:
            CS1 = ax.contourf(np.log10(sample_x),np.log10(sample_y),logSNR,logLevels,cmap=colormap,alpha=cf_alpha)
            ax.contour(np.log10(sample_x),np.log10(sample_y),logSNR,print_logLevels,colors = cline_color,alpha=alpha)
        ax.set_xlim(np.log10(xlabel_min),np.log10(xlabel_max))
        ax.set_ylim(np.log10(ylabel_min),np.log10(ylabel_max))
        x_labels = np.logspace(np.log10(xlabel_min),np.log10(xlabel_max),np.log10(xlabel_max)-np.log10(xlabel_min)+1)
        y_labels = np.logspace(np.log10(ylabel_min),np.log10(ylabel_max),np.log10(ylabel_max)-np.log10(ylabel_min)+1)
        ax.set_yticks(np.log10(y_labels))
        ax.set_xticks(np.log10(x_labels))
                
    #Set axes labels and whether log or linearly spaced
    if var_x == 'M':
        ax.set_xlabel(r'$M_{\rm tot}$ $[\mathrm{M_{\odot}}]$')
        if rotate == True:
            ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)],
                               rotation=70,y=0.02)
        else:
            ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)])
    elif var_x == 'q':
        x_labels = x_labels[::2]
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$\mathrm{Mass~Ratio}$')
        ax.set_xticklabels([r'$%i$' %int(x) for x in x_labels])
    elif var_x == 'z':
        ax.set_xlabel(r'$\mathrm{Redshift}$',fontsize = labelsize)
        ax.set_xticklabels([x if int(x) < 1 else int(x) for x in x_labels])
    elif var_x in ['chi1','chi2']:
        x_labels = np.arange(round(xlabel_min*10),round(xlabel_max*10)+1,1)/10
        x_labels = x_labels[::2]
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$\mathrm{Spin}$')
        ax.set_xticklabels([r'$%.1f$' %x for x in x_labels])
    elif var_x == 'L':
        ax.axvline(x=np.log10(2.5*u.Gm.to('m')),linestyle='--',color='k',label='Proposed Value')
        ax.set_xlabel(r'Arm Length $[\mathrm{m}]$')
        ax.set_xticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in np.log10(x_labels)])
         
    elif var_x == 'A_acc':
        ax.axvline(x=np.log10(3e-15),linestyle='--',color='k',label='Proposed Value')
        #ax.set_xlabel(r'\centering Acceleration Noise \newline Amplitude $[\mathrm{m~s^{-2}}]$')
        ax.set_xlabel(r'$A_{\mathrm{acc}}[\mathrm{m~s^{-2}}]$')
        ax.set_xticklabels([r'$10^{%.0f}$' %x for x in np.log10(x_labels)],fontsize = axissize)
    elif var_x == 'A_IFO':
        ax.axvline(x=np.log10(10e-12),linestyle='--',color='k',label='Proposed Value')
        #ax.set_xlabel(r'\centering Optical Metrology \newline Noise Amplitude [m]')
        ax.set_xlabel(r'$A_{\mathrm{IFO}}$ [m]')
        #ax.set_xticklabels([r'$10^{%.0f}$' %x for x in np.log10(x_labels)])
    elif var_x == 'f_acc_break_low':
        ax.axvline(x=.4*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$f_{\mathrm{acc,low}}$ $[\mathrm{mHz}]$')
        ax.set_xticklabels([r'$%.1f$' %x for x in x_labels*1e3])
    elif var_x == 'f_acc_break_high':
        ax.axvline(x=8.*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$f_{\mathrm{acc,high}}$ $[\mathrm{mHz}]$')
        ax.set_xticklabels([r'$%.1f$' %x for x in x_labels*1e3])
    elif var_x == 'f_IFO_break':
        ax.axvline(x=2.*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$f_{\mathrm{IFO,break}}$ $[\mathrm{mHz}]$')
        ax.set_xticklabels([r'$%.1f$' %x for x in x_labels*1e3])
    elif var_x == 'N_p':
        sample_range = max(x_labels)-min(x_labels)
        sample_rate = max(2,int(sample_range/10))
        x_labels = x_labels[::sample_rate]
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$\mathrm{Number~of~Pulsars}$')
        ax.set_xticklabels([r'$%i$' %int(x) for x in x_labels])
    elif var_x == 'cadence': 
        x_labels = np.arange(round(xlabel_min),round(xlabel_max)+1,5)
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'$\mathrm{Observation~Cadence}$ $[\mathrm{yr}^{-1}]$')
        ax.set_xticklabels([r'$%i$' %int(x) for x in x_labels])
    elif var_x == 'sigma':
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'TOA Error RMS $[\mathrm{ns}]$')
        ax.set_xticklabels([r'$%.0f$' %x for x in x_labels*1e9])
    elif var_x == 'T_obs':
        x_labels = x_labels[::2]
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'${\rm T_{obs}}$ $[\mathrm{yr}]$')
        ax.set_xticklabels([r'$%i$' %int(x) for x in x_labels])
    elif var_x == 'Infrastructure Length':
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'Infrastructure Length [km]')
        ax.set_xticklabels([r'$%.0f$' %x for x in x_labels/1e3])
    elif var_x == 'Laser Power':
        ax.set_xticklabels([r'$10^{%.0f}$' %x if abs(int(x)) > 1 else r'$%.1f$' %(10**x) for x in np.log10(x_labels)])
        ax.set_xlabel(r'Laser Power [W]')
    elif var_x == 'Seismic Gamma':
        scale = 10**round(np.log10(xlabel_min))
        x_labels = np.arange(round(xlabel_min/scale),round(xlabel_max/scale)+1,1)*scale
        x_labels = x_labels[::2]
        ax.set_xticks(x_labels)
        ax.set_xlabel(r'Seismic Gamma')
        ax.set_xticklabels([r'$%.1f$' %x for x in x_labels])
    else:
        ax.set_xticks(x_labels)
        ax.set_xlabel(var_x)
        ax.set_xticklabels([r'$%.1f \times 10^{%i}$' %(x/10**int(np.log10(x)),np.log10(x)) if np.abs(int(np.log10(x))) > 1 else '{:g}'.format(x) for x in x_labels])
        
    if var_y == 'M':
        ax.set_ylabel(r'$M_{\rm tot}$ $[\mathrm{M_{\odot}}]$')
        ax.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)],
                           rotation=-45)
    elif var_y == 'q':
        y_labels = y_labels[::2]
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$\mathrm{Mass~Ratio}$')
        ax.set_yticklabels([r'$%i$' %int(y) for y in y_labels])
    elif var_y == 'z':
        ax.set_ylabel(r'$\mathrm{Redshift}$')
        ax.set_yticklabels([y if int(y) < 1 else int(y) for y in y_labels])
    elif var_y in ['chi1','chi2']:
        y_labels = np.arange(round(ylabel_min*10),round(ylabel_max*10)+1,1)/10
        y_labels = y_labels[::2]
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$\mathrm{Spin}$')
        ax.set_yticklabels([r'$%.1f$' %y for y in y_labels])
    elif var_y == 'L':
        ax.axhline(y=np.log10(2.5*u.Gm.to('m')),linestyle='--',color='k',label='Proposed Value')
        ax.set_ylabel(r'$Arm Length $[\mathrm{m}]$')
        ax.set_yticklabels([r'$10^{%i}$' %y if int(y) > 1 else r'$%i$' %(10**y) for y in np.log10(y_labels)])
    elif var_y == 'A_acc':
        ax.axhline(y=np.log10(3e-15),linestyle='--',color='k',label='Proposed Value')
        ax.set_ylabel(r'\centering Acceleration Noise \newline Amplitude $[\mathrm{m~s^{-2}}]$')
        ax.set_ylabel(r'$A_{\mathrm{acc}} [\mathrm{m~s^{-2}}]$')
        ax.set_yticklabels([r'$10^{%.0f}$' %y for y in np.log10(y_labels)])
    elif var_y == 'A_IFO':
        ax.axhline(y=np.log10(10e-12),linestyle='--',color='k', label='Proposed Value')
        ax.set_ylabel(r'\centering Optical Metrology \newline Noise Amplitude [m]')
        ax.set_ylabel(r'$A_{\mathrm{IFO}}$ [m]')
        ax.set_yticklabels([r'$10^{%.0f}$' %y for y in np.log10(y_labels)])
    elif var_y == 'f_acc_break_low':
        ax.axhline(y=.4*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(ylabel_min))
        y_labels = np.arange(round(ylabel_min/scale),round(ylabel_max/scale)+1,1)*scale
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$f_{\mathrm{acc,low}} [\mathrm{mHz}]$')
        ax.set_yticklabels([r'$%.1f$' %y for y in y_labels*1e3])
    elif var_y == 'f_acc_break_high':
        ax.axhline(y=8.*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(ylabel_min))
        y_labels = np.arange(round(ylabel_min/scale),round(ylabel_max/scale)+1,1)*scale
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$f_{\mathrm{acc,high}} [\mathrm{mHz}]$')
        ax.set_yticklabels([r'$%.1f$' %y for y in y_labels*1e3])
    elif var_y == 'f_IFO_break':
        ax.axhline(y=2.*u.mHz.to('Hz'),linestyle='--',color='k',label='Proposed Value')
        scale = 10**round(np.log10(ylabel_min))
        y_labels = np.arange(round(ylabel_min/scale),round(ylabel_max/scale)+1,1)*scale
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$f_{\mathrm{IFO,break}} [\mathrm{mHz}]$')
        ax.set_yticklabels([r'$%.1f$' %y for y in y_labels*1e3])
    elif var_y == 'N_p':
        sample_range = max(y_labels)-min(y_labels)
        sample_rate = max(2,int(sample_range/10))
        y_labels = y_labels[::sample_rate]
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$\mathrm{Number~of~Pulsars}$')
        ax.set_yticklabels([r'$%i$' %int(y) for y in y_labels])
    elif var_y == 'cadence': 
        y_labels = np.arange(round(ylabel_min),round(ylabel_max)+1,5)
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'$\mathrm{Observation~Cadence}$ $[\mathrm{yr}^{-1}]$')
        ax.set_yticklabels([r'$%i$' %int(y) for y in y_labels])
    elif var_y == 'sigma':
        scale = 10**round(np.log10(ylabel_min))
        y_labels = np.arange(round(ylabel_min/scale),round(ylabel_max/scale)+1,1)*scale
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'TOA Error RMS $[\mathrm{ns}]$')
        ax.set_yticklabels([r'$%.0f$' %y for y in y_labels*1e9])
    elif var_y == 'T_obs':
        y_labels = y_labels[::2]
        ax.set_yticks(y_labels)
        ax.set_ylabel(r'${\rm T_{obs}}$ $[\mathrm{yr}]$')
        ax.set_yticklabels([r'$%i$' %int(y) for y in y_labels])
    elif var_y == 'Infrastructure Length':
        ax.axhline(y=np.log10(3995),linestyle='--',color='k',label='Proposed Value')
        ax.set_ylabel(r'Arm Length [m]')
        ax.set_yticklabels([r'$10^{%.0f}$' %y if abs(int(y)) > 1 else r'$%.1f$' %(10**y) for y in np.log10(y_labels)])
    elif var_y == 'Laser Power':
        ax.axhline(y=np.log10(125),linestyle='--',color='k',label='Proposed Value')
        ax.set_yticklabels([r'$10^{%.0f}$' %y if abs(int(y)) > 1 else r'$%.1f$' %(10**y) for y in np.log10(y_labels)])
        ax.set_ylabel(r'Laser Power [W]')
    elif var_y == 'Seismic Gamma':
        ax.axhline(y=np.log10(0.8),linestyle='--',color='k',label='Proposed Value')
        ax.set_ylabel(r'Seismic Gamma')
        ax.set_yticklabels([r'$10^{%.0f}$' %y if abs(int(y)) > 1 else r'$%.1f$' %(10**y) for y in np.log10(y_labels)])
    else:
        ax.set_yticks(y_labels)
        ax.set_ylabel(var_y)
        ax.set_yticklabels([r'$%.1f \times 10^{%i}$' %(y/10**int(np.log10(y)),np.log10(y)) if np.abs(int(np.log10(y))) > 1 else '{:g}'.format(y) for y in y_labels])
        
    if not x_axis_label:
        #ax.get_xaxis().set_visible(False)
        ax.set_xticklabels('')
        ax.set_xlabel('')
    if not y_axis_label:
        #ax.get_yaxis().set_visible(False)
        ax.set_yticklabels('')
        ax.set_ylabel('')
    
    if var_y in ['A_acc','A_IFO']:
        ax.yaxis.set_label_coords(y_label_loc-.1,.5)
    elif var_y in ['chi1','chi2']:
        ax.yaxis.set_label_coords(y_label_loc-.05,.5)
    else:
        ax.yaxis.set_label_coords(y_label_loc,.5)

    if display_cbar:
        #If true, display luminosity distance on right side of plot
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.025, 0.7])
        if dl_axis:
            #Set other side y-axis for lookback time scalings
            ax2 = ax.twinx()
            #Set axis scales based on what data sampling we used 
            if yaxis_type == 'lin' and xaxis_type == 'log':
                ax2.contour(np.log10(sample_x),sample_y,logSNR,print_logLevels,colors = 'k',alpha=1.0)
            elif yaxis_type == 'log' and xaxis_type == 'lin':
                ax2.contour(sample_x,np.log10(sample_y),logSNR,print_logLevels,colors = 'k',alpha=1.0)
            else:
                ax2.contour(np.log10(sample_x),np.log10(sample_y),logSNR,print_logLevels,colors = 'k',alpha=1.0)

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
            ax2.set_yticklabels([r'$10^{%i}$' %np.log10(dist) if np.abs(int(np.log10(dist))) > 1 
                                 else '{:g}'.format(dist) for dist in dists.value])
            ax2.set_ylabel(r'$D_{L}$ [Gpc]')
            cbar = fig.colorbar(CS1,ax=(ax,ax2),pad=0.01,ticks=print_logLevels)
        else:
            if cfill == False:
                #Make colorbar
                #norm= colors.LinearSegmentedColormap.from_list('custom',print_logLevels,CS1.cmap.N)
                norm= colors.Normalize(vmin=logLevels_min, vmax=logLevels_max)
                #norm = mpl.colors.BoundaryNorm(print_logLevels, CS1.cmap.N)
                #sm = plt.cm.ScalarMappable(norm=norm, cmap = CS1.cmap)
                #sm.set_array([])
                tick_levels = np.linspace(float(logLevels_min),logLevels_max,len(print_logLevels))
                #cbar = fig.colorbar(sm, cax=cbar_ax, ticks=tick_levels)
                cbar = mpl.colorbar.ColorbarBase(cbar_ax,cmap=CS1.cmap,norm=norm,boundaries=tick_levels,
                    ticks=tick_levels,spacing='proportional')
                #cbar = fig.colorbar(CS1,cax=cbar_ax,ticks=print_logLevels)
            else:
                #Make colorbar
                cbar = fig.colorbar(CS1, cax=cbar_ax,ticks=print_logLevels)
            
        #cbar.set_label(r'$SNR$')
        cbar.ax.set_yticklabels([r'$10^{%i}$' %x if int(x) > 1 else r'$%i$' %(10**x) for x in print_logLevels])

    if display:
        #fig.tight_layout()
        fig.subplots_adjust(hspace=hspace,wspace=wspace)
        plt.show()