# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:53:12 2016

@author: JPC-Acquisition
"""

import numpy as np
#import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.colorbar as plt_colorbar
import matplotlib as mpl
import h5py
from scripts.flux_sweep import analysis_functions as af
from scripts.flux_sweep import FluxSweep as fs

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'svg'
mpl.rcParams['axes.linewidth'] = 2.0

mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 2.0
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 0.3

mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 2.0
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 0.3

mpl.rcParams['xtick.major.pad']='8'
mpl.rcParams['ytick.major.pad']='8'

# get all dataDirs in between the 2 given, sweep could span 2 days
def get_dataDirs(h5filename, dataDir1, dataDirL):
    dir1 = dataDir1.split('/')  # should be ['', '160223', '121212']
    dirL = dataDirL.split('/')
    dataDirs = []
    h5f = h5py.File(h5filename)
    try:
        grp = h5f[dir1[1]]
        times = grp.keys()
        i_1 = times.index(dir1[2])
        if dir1[1] != dirL[1]: # data take on 2 days
            times = times[i_1:]
            dataDirs += ['/' + dir1[1] + '/' + time for time in times]
            grp = h5f[dirL[1]]
            times = grp.keys()
            i_1 = 0
        i_L = times.index(dirL[2])
        times = times[i_1 : i_L+1] # only select the times between the two specified by dir1 and dirL
        dataDirs += ['/' + dirL[1] + '/' + time for time in times]  #adds the full dataDir to the dataDirs list
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    return dataDirs

# fit multiple power sweeps at dataDirs
def fitall_power_sweeps(h5filename, dataDirs, extParam_name=None, extParams=[], kc=40e6, ki=1e6, numFitPoints=0):
    if not extParam_name:
        extParams = np.zeros_like(dataDirs)
    for i_dir, dataDir in enumerate(dataDirs):
        fs.power_sweep_Qfit(h5filename, dataDir=dataDir, kc=40e6, ki=1e6, display=False, guessF=fs.guessF_maxReal, save=True, numFitPoints=0, extParam_name=extParam_name, extParam=extParams[i_dir])

# load all summary data from reflection fits of multiple sweeps, 'currents' or 'VNA_power'
def load_multiSweepQfitData(h5filename, dataDirs, sweep_var='currents', extParam_name=None):
    # h5filename: full path to h5file
    # dataDirs: list of relative directiories inside h5file (eg. ['/161111/001234', '/161111/002255'])
    # sweep_var: 'currents' for flux sweep, 'VNA_power' for vna power sweep
    dataList = []
    for dataDir in dataDirs:
        h5file = h5py.File(h5filename)
        meas = h5file[dataDir]
        xdata = np.array(meas[sweep_var])
        h5file.close()
        # all fit data for one sweep
        fitData = fs.load_Qfit(h5filename, dataDir, extParam_name=extParam_name)
        fitData[sweep_var] = xdata
        dataList.append(fitData)
    return dataList

# for certain ranges in xdata, remove all points within a range (x_lower, x_upper)
def remove_points(xdata, data_list, ranges=None):
    allData = data_list
    xdat = xdata
    if ranges:
        for i_range, pair in enumerate(ranges):
            i_0 = fs.find_nearest(xdat, pair[0])
            i_1 = fs.find_nearest(xdat, pair[1])
            if i_1 < i_0:
                tmp = i_0
                i_0 = i_1
                i_1 = tmp
            xdat = np.delete(xdat,np.s_[i_0:(i_1+1)])
            for i_dat, data in enumerate(allData):
                allData[i_dat] = np.delete(allData[i_dat],np.s_[i_0:(i_1+1)])
    return (xdat, allData)
            
 
# plot all summary data from reflection fits of multiple sweeps, 'currents' or 'VNA_power'
def plot_multipSweepQfitData(h5filename, dataDirs, labels, sweep_var='currents', atten=0, extParam_name=None, cutRanges=None, save=None, toFlux=False):
    dataList = load_multiSweepQfitData(h5filename, dataDirs, sweep_var=sweep_var, extParam_name=extParam_name)
    colormap = plt.get_cmap('rainbow')
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    fig.set_size_inches(10,8)
    fig.subplots_adjust(bottom=0.15)
    ylabels = (r'$f_0 \,\rm{(GHz)}$', r'$\kappa_c/2\pi \, \rm{(MHz)}$', r'$\kappa_i/2\pi \, \rm{(MHz)}$')
    for i_ax, ax in enumerate(axes):
        ax.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(dataList))])
        ax.set_ylabel(ylabels[i_ax])
    for i_data, data in enumerate(dataList):
        xdata = data[sweep_var]
        f0 = data['f0'] * 1e-9 # res freq in GHz
        kc = data['kc'] * 1e-6 # kc/2pi in MHz
        ki = data['ki'] * 1e-6 # ki/2pi in MHz
        if cutRanges:
            (x, allData) = remove_points(xdata, [f0,kc,ki], ranges=cutRanges)
            xdata = x
            f0 = allData[0]
            kc = allData[1]
            ki = allData[2]
        if sweep_var == 'currents':
            if toFlux:
                xlabel = r'$\Phi_{\rm{ext}}/\Phi_0$'
                xdata = curr2phi(xdata)
            else:
                xlabel = 'Currents (mA)'
                xdata = xdata * 1e3
        elif sweep_var == 'VNA_power':
            xlabel = 'VNA Power (dBm)'
            xdata = xdata + atten
        else:
            print('Sweep_var not supported in plot_multipSweepQfitData')
        axes[0].plot(xdata, f0, label=labels[i_data], marker='+',ls='None', fillstyle='none',mew = 1.0)
        axes[1].plot(xdata, kc, label=labels[i_data], marker='+',ls='None', fillstyle='none',mew = 1.0)
        axes[2].plot(xdata, ki, label=labels[i_data], marker='+',ls='None', fillstyle='none',mew = 1.0)
    axes[-1].set_xlabel(xlabel)
    axes[-1].legend(loc='upper right')
    if toFlux and sweep_var == 'currents':
        axes[-1].set_xlim([-1,1])
    axes[1].set_ylim([0,14])#48,14
    axes[1].set_yticks(np.arange(3)*5)#(5)10, (3)5
    axes[2].set_ylim([0,14])#23, 14
    axes[2].set_yticks(np.arange(3)*5)#(5)5 (3)5
    plt.rcParams.update({'font.size':18})
    if save:
        fig.savefig(save)
    
# plot frequency shifts, VNA sweeps
def plot_freqShifts(h5filename, dataDirs, labels, sweep_var='VNA_power', atten=0, extParam_name=None, cutRanges=None, save=None, toFlux=False):
    dataList = load_multiSweepQfitData(h5filename, dataDirs, sweep_var=sweep_var, extParam_name=extParam_name)
    colormap = plt.get_cmap('rainbow')
    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)
    fig.subplots_adjust(bottom=0.15)
    ax.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(dataList))])
    ax.set_ylabel(r'$\Delta f \,\rm{(MHz)}$')
    for i_data, data in enumerate(dataList):
        xdata = data[sweep_var]
        f0 = data['f0'] * 1e-6
        if cutRanges:
            (x, allData) = remove_points(xdata, [f0], ranges=cutRanges)
            xdata = x
            f0 = allData[0]
        if sweep_var == 'currents':
            if toFlux:
                xlabel = r'$\Phi_{\rm{ext}}/\Phi_0$'
                xdata = curr2phi(xdata)
            else:
                xlabel = 'Currents (mA)'
                xdata = xdata * 1e3
        elif sweep_var == 'VNA_power':
            xlabel = 'VNA Power (dBm)'
            xdata = xdata + atten
        else:
            print('Sweep_var not supported in plot_multipSweepQfitData')
        df0 = f0 - f0[0]            # freq shift in MHz
        ax.plot(xdata, df0, label=labels[i_data], marker='+')
    ax.set_xlabel(xlabel)
    ax.legend(loc='best')
    if toFlux:
        ax.set_xlim([-1,1])
    plt.rcParams.update({'font.size':18})
    if save:
        fig.savefig(save)

def get_fpump_2D(h5filename, dataDirs):
    fpump = np.zeros(len(dataDirs))
    for i_dir, dataDir in enumerate(dataDirs):
        h5f = h5py.File(h5filename)
        meas = h5f[dataDir]
        fpump[i_dir] = meas['powers'].attrs['pump_frequency']
        h5f.close()
    return fpump

# fit all gain curves and NVR of a 2D pump power sweep
def get_gain_NVR_2Dpump(h5filename, dataDirs, vna_states=('Signal','Idler'), nvr=True, recalc=False, displayFit=False,  memDataDirs=None, memvna_fmt='SCOM'):
    numDirs = len(dataDirs)
    pfreqs = get_fpump_2D(h5filename, dataDirs)
    data = {'powers': [],
            'gain_S': [], 
            'gain_I': [],
            'nvr_S': [],
            'nvr_I': [],
            'pfreqs': pfreqs}
    if not nvr:
        raise ValueError('Option nvr=False in get_gain_NVR_2Dpump not supported.')
    for i_dir, dataDir in enumerate(dataDirs):
        (pows, g_S, g_I, n_S, n_I) = fs.get_gain_NVR(h5filename=h5filename, dataDir=dataDir, vna_states=vna_states, nvr=True,
                                                    recalc=recalc, display=displayFit, memDataDirs=memDataDirs, memvna_fmt=memvna_fmt)
        if i_dir == 0:
            data['powers'] = pows
            data['gain_S'] = np.zeros((numDirs, g_S.shape[0]))
            data['gain_I'] = np.zeros((numDirs, g_I.shape[0]))
            data['nvr_S'] = np.zeros((numDirs, n_S.shape[0]))
            data['nvr_I'] = np.zeros((numDirs, n_I.shape[0]))
        data['gain_S'][i_dir,:] = g_S 
        data['gain_I'][i_dir,:] = g_I
        data['nvr_S'][i_dir,:] = n_S
        data['nvr_I'][i_dir,:] = n_I
    return data
    
# plot gain points as function of pump power, pump freq as suggested by MJH
def plot_gain_2Dplot_MJH(h5filename, dataDirs, vna_states=('Signal','Idler'), nvr=True, recalc=False, displayFit=False, 
                         memDataDirs=None, memvna_fmt='SCOM', save=None):
    data = get_gain_NVR_2Dpump(h5filename, dataDirs, vna_states=('Signal','Idler'), nvr=nvr,
                               recalc=recalc, displayFit=displayFit,  memDataDirs=memDataDirs, memvna_fmt='SCOM') 
    cm = plt.cm.get_cmap('rainbow')
    pfreqs = data['pfreqs']*1e-9
    ppows = data['powers']
    xlim = [pfreqs[0], pfreqs[-1]] 
    ylim = [ppows[0],ppows[-1]]
    labels = ['Pump frequency (GHz)', 'Pump power (dB)']
    figS, axS = plt.subplots()
    plt.title('Signal')
    pS = axS.pcolormesh(pfreqs, ppows, np.transpose(data['gain_S']), cmap=cm, vmin=10, vmax=30)
    cbS = figS.colorbar(pS, ax=axS, label='Gain (dB)')
    axS.set_xlim(xlim)
    axS.set_ylim(ylim)
    axS.set_xlabel(labels[0])
    axS.set_ylabel(labels[1])
    figI, axI = plt.subplots()
    plt.title('Idler')
    pI = axI.pcolormesh(pfreqs, ppows, np.transpose(data['gain_I']), cmap=cm, vmin=10, vmax=30)
    cbI = figI.colorbar(pI, ax=axI, label='Gain (dB)')
    axI.set_xlim(xlim)
    axI.set_ylim(ylim)
    axI.set_xlabel(labels[0])
    axI.set_ylabel(labels[1])
    if save:
        figS.savefig(save + r'\fig_2Dpump_S')
        figI.savefig(save + r'\fig_2Dpump_I')
#    for i_freq, pfreq in enumerate(data['pfreqs']*1e-9):
#        for i_pow, power in enumerate(data['powers']):
##            scS = plt.scatter(pfreq, power, c=data['gain_S'][i_freq,i_pow], vmin=0, vmax=30, cmap=cm)
#            scI = plt.scatter(pfreq, power, c=data['gain_I'][i_freq,i_pow], vmin=0, vmax=30, cmap=cm)
#

def plot_gain_trace(h5filename, dataDir, vna_fmt='PLOG', save=None):
#    data = fs.load_sweepData(h5filename, dataDir=dataDir, vna_fmt=vna_fmt, sweep_var=None)
#    freqs = data['frequencies'] * 1e-9
#    gain = 20 * np.log10(np.abs(data['a_complex']))
    h5file = h5py.File(h5filename)
    meas = h5file[dataDir]
    gain1 = np.array(meas['logmag'])
    freqs = np.array(meas['frequencies']) * 1e-9
    h5file.close()
    fig, ax = plt.subplots(1,1,figsize = (5,2.5))
#    plt.plot(freqs, gain, color='r')
    ax.plot(freqs, gain1, color='r')
    ax.set_xlim([7.6, 7.7])
    plt.setp(ax.get_xticklabels()[0::2], visible=False)
    plt.setp(ax.get_yticklabels()[0::2], visible=False)
    if save:
        fig.savefig(save)
    

def curr2phi(current, offset=-0.1988e-3, curr_1fq=4.4325e-3):
    return (current - offset) / curr_1fq
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    h5filename = r'Z:\Data\JPC_2016\2016-10-22 Cooldown\amj01.hdf5'
    fs.define_phaseColorMap()
    if False:
        h5filename = r'Z:\Data\JPC_2016\2016-05-06 Cooldown\jphf21.hdf5'
        dataDir = '/160606/175048'
        save = r'Z:\Talks\Nick\APS2017\figs\gain_intro'
        plot_gain_trace(h5filename, dataDir, vna_fmt='PLOG', save=save)
        
    if False:
        dataDirs = ['/161025/165032', '/161025/182602'] #Idler
        dataDirs = ['/161026/143005', '/161026/175546'] #Signal
        dataDirs = ['/161026/200856', '/161026/213726'] #spurious
        labels = [r'$S_{SS} \, \rm{forward}$', r'$S_{SS} \, \rm{reverse}$']
        cutRanges = [(-2.43e-3, -2.38e-3), (1.99e-3, 2.05e-3)]
#        cutRanges = [(-2.74e-3, -2.08e-3), (1.69e-3, 2.34e-3)]
        plot_multipSweepQfitData(h5filename, dataDirs, labels, sweep_var='currents', cutRanges=cutRanges,
                                 save=r'Z:\Talks\Nick\MLS2016\figs\fluxSweep_spur', toFlux=True)
     
    if False:
#        dataDirs = get_dataDirs(h5filename, '/161101/100605', '/161101/104547') #S_II 1.600-1.650
#         dataDirs = get_dataDirs(h5filename, '/161101/183936', '/161101/235351')
        dataDirs = get_dataDirs(h5filename, '/161102/105038', '/161102/121132') #S_SS 1.6-1.65
#        dataDirs = get_dataDirs(h5filename, '/161102/142501', '/161102/150911') #spurious
        curr_start = 1.600e-3 #1.600e-3
        curr_stop = 1.650e-3 #1.650e-3
        curr_step = 10e-6
        numCurr = round(abs(curr_stop - curr_start) / curr_step + 1)
        currents = np.linspace(curr_start, curr_stop, numCurr)
#        fitall_power_sweeps(h5filename, dataDirs, extParam_name='currents', extParams=currents, kc=1e6, ki=1e6, numFitPoints=0)
        labels = np.round(curr2phi(currents),3)
        cutRanges = None #[(-48, -40)]
        plot_multipSweepQfitData(h5filename, dataDirs, labels, sweep_var='VNA_power', atten=0, extParam_name='currents',
                                 save=r'Z:\Talks\Nick\MLS2016\figs\VNAppow_fit_SS')
        plot_freqShifts(h5filename, dataDirs, labels, sweep_var='VNA_power', atten=0, extParam_name='currents',
                        cutRanges=cutRanges, save=r'Z:\Talks\Nick\MLS2016\figs\VNAppow_shift_SS')
        
    if False:
        dataDirs = ['/161027/111646', '/161027/140859']
        data = fs.load_sweepData(h5filename, dataDirs[0], vna_fmt='SCOM')
        
    if False: # fit 2D pump power sweep gain/NVR
        memDataDirs = ['/161027/111646', '/161027/140859']
        dataDirs = get_dataDirs(h5filename, '/161027/152705', '/161027/200731')
        save = r'Z:\Talks\Nick\APS2017\figs'
#        data = get_gain_NVR_2Dpump(h5filename, dataDirs, vna_states=('Signal','Idler'), nvr=True,
#                                   recalc=False, displayFit=False,  memDataDirs=memDataDirs, memvna_fmt='SCOM')
        plot_gain_2Dplot_MJH(h5filename, dataDirs, vna_states=('Signal','Idler'), nvr=True,
                                   recalc=False, displayFit=False,  memDataDirs=memDataDirs, memvna_fmt='SCOM', save=None)