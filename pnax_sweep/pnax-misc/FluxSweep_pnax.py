# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 10:04:53 2017

@author: JPC-Acquisition
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.colorbar as plt_colorbar
from time import sleep
from time import strftime
from PyQt4 import QtGui
import h5py
from scipy.constants import Planck
from scripts.flux_sweep import analysis_functions as af
from scripts.flux_sweep import FluxSweep as fs


def set_center_span(pnax, center_freq, span, points=1601, averages=20, power=-65):
    pnax.set_format('Phase')
    pnax.set_points(points)
    pnax.set_sweep_type('LIN')    
    pnax.set_average_factor(averages)
    pnax.set_averaging_state(True)    
    pnax.set_power(power)
    start_freq =  center_freq - span/2.0
    stop_freq = center_freq + span/2.0
    pnax.set_mixerOUTP_freq_mode('SWEPT')
    pnax.set_mixerINP_freq_mode('SWEPT')
    pnax.set_mixerINP_freq_start(start_freq)
    pnax.set_mixerINP_freq_stop(stop_freq)
    pnax.set_trigger_source('IMM')
    pnax.set_averaging_mode('sweep')
    pnax.mixer_calc('OUTP')
    pnax.mixer_apply_settings() 


# runs a flux sweep on pnax, for each flux: takes data for each vna_state
def flux_sweep_pnax_state(pnax, yoko, currents=np.linspace(-2.0e-3,2.0e-3,401),
                    fridge=None, display=False, atten=0, vna_fmt='POL', printlog=True,
                    h5filename=None, vna_states=['Signal','Idler'], yoko_slew=10e-6):
    # pnax: must be valid PNAX object
    # yoko: must be valid yoko object
    # currents: numpy array of currents to sweep through [A]
    # fridge: valid fridge object to record temperature
    # display: True displays trace by trace of sweep
    # atten: total gain on line, so -70 is standard [dB]
    # vna_fmt: data format, 'POL' gives real/imag
    # printlog: True prints updates, False surpresses output
    # h5filename: full h5filename path to valid h5file
    # vna_states: list of valid vna_states, if vna_states is None then will save current vna state to 'python' and use that
    # yoko_slew: slew default 10e-6 [A/s]
    if h5filename:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        dataDir = _iteration_start+'//'
    if not vna_states:
        vna_states = ['python']
        pnax.save_state(vna_states[0]) #saves current pnax config to 'python' state
    # get all parameters for all vna states
    xs = {}
    ys = {}
    averages = {}
    num_points = {}
    if_bw = {}
    vna_pows = {}
    smoothing = {}
    elec_delay ={}
    sweep_times = {}
    tot_time = 0
    figs = {}
    axes = {}
    for state in vna_states:
        pnax.load_state(state)
#        sleep(0.1) # make sure it is loaded.
        pnax.opc()
        averages[state] = pnax.get_average_factor()
        num_points[state] = pnax.get_points()
        if_bw[state] = pnax.get_if_bandwidth()
        vna_pows[state] = float(pnax.get_power())
        smoothing[state] = pnax.get_smoothing()
        elec_delay[state] = pnax.get_electrical_delay()
        if pnax.get_sweep_type() == 'SEGM':
            sweep_times[state] = float(pnax.get_segment_sweep_time())
        else:
            sweep_times[state] = float(pnax.get_sweep_time())
        tot_time += sweep_times[state] * averages[state] + 0.5
        xs[state] = pnax.do_get_xaxis()
        ys[state] = np.zeros((len(currents), 2, num_points[state])) #matrix of ys resutls with dimensions [current, real/imag, freq]
        if display:
            figs[state] = plt.figure()
            axes[state] = figs[state].add_subplot(111)
    tot_time *= len(currents)
    if printlog:
        print("Getting timing information and preparing trigger...")
        print('Total duration of this expermient will be %.2f minutes.'%(tot_time/60.0))
    
    # sweep the flux
    for i_current, current in enumerate(currents):
        # tell yoko to go to current
        wait_time = yoko.set_current_ramp(current, slew=yoko_slew)
        if wait_time:
            sleep(wait_time + 0.5) #0.5 for GBIB overhead
        for state in vna_states:
            pnax.load_state(state)
#            sleep(0.1)
            pnax.opc()
            pnax.restart_averaging() # restarts averaging
            this_time = sweep_times[state] * averages[state] * 1250.0 #ms
            this_time = np.max(np.array([this_time, 5000])) # 5 s timeout minimum
            pnax.set_timeout(this_time)
            # get data
            ys[state][i_current,:,:] = pnax.do_get_data(fmt=vna_fmt, opc=True, timeout=this_time)
            # plotting
            try:
                if display:
                    axes[state].plot(xs[state], ys[state][i_current,0,:], label='%.3f mA'%(current*1e3))
                    figs[state].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
        # end state for loop
    # end current for loop
    
    # format plots and save data
    if h5filename:
        h5f = h5py.File(h5filename)
        g = h5f.require_group(cur_group) #set up group for day
    try:
        for state in vna_states:
            try:
                if display:
                    axes[state].set_title('VNA Trace State: ' + state)
                    axes[state].set_xlabel('Frequency (Hz)')
                    axes[state].set_ylabel(fs.DATA_LABEL[vna_fmt][0])
                    axes[state].legend(loc='best')
            except:
                pass
            if h5filename:
                thisDir = dataDir + state + '//'
                dg1 = g.create_dataset(thisDir + 'frequencies', data=xs[state])
                dg1.attrs.create('run_time', _iteration_start)
                dg = g.create_dataset(thisDir + fs.DATA_LABEL[vna_fmt][0], data=ys[state][:,0,:])
                dg.attrs.create('run_time', _iteration_start)
                dg.attrs.create('attenuation',atten)
                dg.attrs.create('IFbandwidth', if_bw[state])
                dg.attrs.create('avaerages', averages[state])
                dg.attrs.create('VNA_power', vna_pows[state])
                dg.attrs.create('smoothing', smoothing[state])
                dg.attrs.create('electrical_delay', elec_delay[state])
                dg.attrs.create('format', vna_fmt)
                dg2 = g.create_dataset(thisDir + fs.DATA_LABEL[vna_fmt][1], data= ys[state][:,1,:])
                dg2.attrs.create('run_time', _iteration_start)
                dg_currs = g.create_dataset(thisDir + 'currents', data=currents)
                dg_currs.attrs.create('run_time', _iteration_start)
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if h5filename:
            h5f.close()
            del(h5f)
    return (currents, xs, ys)

# takes flux sweep and gets Sparams for each meas_name at each flux in SMC mode    
def flux_sweep_SMC(pnax, yoko, meas_names, currents=np.linspace(-2.0e-3, 2.0e-3, 401),
                   fridge=None, display=False, atten=0, vna_fmt='POL', printlog=True,
                   yoko_slew = 10e-6):
    figs = {}
    axes = {}
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    if pnax.get_sweep_type() == 'SEGM':
        sweep_time = float(pnax.get_segment_sweep_time())
    else:
        sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 0.5) * len(meas_names) * len(currents)
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
#    xs = pnax.do_get_xaxis()
    ys = {}
    data = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(currents), 2, num_points)) #matrix of ys results
        if display:
            figs[meas] = plt.figure()
            axes[meas] = figs[meas].add_subplot(111)
    # sweep flux
    for i_current, current in enumerate(currents):
        wait_time = yoko.set_current_ramp(current, slew=yoko_slew)
        if wait_time:
            sleep(wait_time + 0.5) #0.5 for GBIP overhead
        data = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN', opc_arg=True)
        for meas in meas_names:
            ys[meas][i_current,:,:] = data['ys'][meas]
            try:
                if display:
                    axes[meas].plot(data['xs'], ys[meas][i_current,0,:],label='%.3f mA'%(current*1e3))
                    figs[meas].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
    # put full yoko sweep data in ys
    data['ys'] = ys
    data['currents'] = currents
    return data
    
# plot flux sweep data using color map
def plot_flux_sweep_SMC(data, savepath=None):
    if data['format'] != 'POL':
        print("Format not supported by plot_Sparams_SMC") #TODO - support other formats
        return
    fs.define_phaseColorMap()
    freqs = data['xs']
    freqLO = data['freqLO']
    meas_names = data['meas_names']
    for meas in meas_names:
        yc = data['ys'][meas][:,0,:] + 1j*data['ys'][meas][:,1,:]
        if 'S22' in meas:
            freqs = freqLO - freqs[::-1]
            yc = yc[:, ::-1]
        (ylog, yphase) = fs.complex_to_PLOG(yc)
        fs.colorMap_flux(data['currents'], freqs, ylog, yphase, colorMap=('hot','phase'), title=meas,savepath=savepath)

def fit_flux_sweep_SMC(data, display=False, kc=60e6, ki=1e6, numFitPoints=0,
                       guessF=fs.guessF_maxReal, updateKappaGuess=False):
    return fit_sweep_SMC(data, swept_var='currents', display=display, kc=kc, ki=ki,
                         numFitPoints=numFitPoints, guessF=guessF, updateKappaGuess=updateKappaGuess)

def fit_sweep_SMC(data, swept_var='currents', display=False, kc=60e6, ki=1e6, numFitPoints=0,
                  guessF=fs.guessF_maxReal, updateKappaGuess=False):
    results = {}
    meas_names = data['meas_names']
    freqLO = data['freqLO']
    currents = data[swept_var]
    xs = data['xs']
    ys = data['ys']
    if data['format'] != 'POL':
        raise ValueError('data format ' + data['format'] + 'is not supported by fit_flux_sweep_SMC')
    for meas in meas_names:
        freqs = xs
        yc = ys[meas][:,0,:] + 1j*ys[meas][:,1,:]
        if 'S22' in meas:
            freqs = freqLO - freqs[::-1]
            yc = yc[:, ::-1]
        (_, _, _, _, res) = fs.flux_sweep_Qfit_helper(freqs, currents, yc, numFitPoints=numFitPoints, kc=kc, ki=ki,
                                                        display=display, guessF=guessF, updateKappaGuess=updateKappaGuess)
        results[meas] = res
    return results

# saves flux sweep Qfit data in h5 under dataDir/'fits'/meas_name
def save_fit_sweep_SMC(fitdata, meas_names, h5filename, dataDir=None, extParam_name=None, extParam=0.0):
    h5file = h5py.File(h5filename)
    try:
        if not dataDir:
            print('Using most recent data')
            day = h5file.keys()[-1]
            grp = h5file[day]
            dataDir = day + '//' + grp.keys()[-1]
        folder = h5file[dataDir]
        g = folder.require_group('fits')
        for meas in meas_names:
            g_meas = g.require_group(meas)
            fs.save_Qfit_helper(fitdata[meas], g_meas, extParam_name=extParam_name, extParam=extParam)
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5file.close()
    return dataDir + '//fits'
 
def plot_fit_flux_sweep_SMC(fitdata, meas_names, currents):
    plot_fit_sweep_SMC(fitdata, meas_names, currents, swept_var='currents')
    
def plot_fit_sweep_SMC(fitdata, meas_names, xdat, swept_var='currents'):
    if swept_var == 'currents':
        xdata = xdat * 1e3
        xlabel = 'Current (mA)'
    elif swept_var == 'powers':
        xdata = xdat
        xlabel = 'Power (dBm)'
    else:
        raise ValueError('swept_var: ' + swept_var + ' not supported by plot_fit_sweep_SMC')
    fig, ax = plt.subplots(4, sharex=True)
    for meas in meas_names:
        f0 = fitdata[meas][:,0]
        kc = fitdata[meas][:,1]
        ki = fitdata[meas][:,2]
        ax[0].plot(xdata, f0*1e-9, label=meas)
        ax[1].plot(xdata, kc*1e-6, label=meas)
        ax[2].plot(xdata, ki*1e-6, label=meas)
        ax[3].plot(xdata, f0 / (kc+ki), label=meas)
    ax[0].set_ylabel('f0 (GHz)')
    ax[1].set_ylabel(r'$\kappa_c / 2\pi$ (MHz)')
    ax[2].set_ylabel(r'$\kappa_i / 2\pi$ (MHz)')
    ax[3].set_ylabel('Qtotal')
    ax[-1].set_xlabel(xlabel)
    ax[-1].legend()

# take S-parameters for SMC (Scalar Mixer Conversion) class
def get_Sparams_SMC(pnax, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True):
    # pnax: valid pnax object (SMC configured)
    # meas_names: list of valid measurement names on chan 1 (ie '"CH1_S11_3"' for an S11 measurement that is trace 1)
    # vna_fmt: valid data format (ie 'POL')
    # sweep_type: pnax sweep type 'LIN' for linear frequency sweep, 'POW' for power sweep
    if len(meas_names) == 0:
        return
    pnax.set_meas_select(meas_names[0])
    freqLO = pnax.get_mixerLO_freq_fixed()
    if sweep_type == 'LIN':
        power = pnax.get_power()
    elif sweep_type == 'POW':
        freqINP = pnax.get_mixerINP_freq_fixed()
        freqOUTP = pnax.get_mixerOUTP_freq_fixed()
    else:
        raise ValueError('sweep_type ' + sweep_type + ' is not supported for Sparams_SMC.')
    data = get_Sparams_helper(pnax, meas_names, vna_fmt=vna_fmt, sweep_type=sweep_type, opc_arg=opc_arg)
    data['freqLO'] = freqLO
    if sweep_type == 'LIN':
        data['power'] = power
    elif sweep_type == 'POW':
        data['freqINP'] = freqINP
        data['freqOUTP'] = freqOUTP
    return data

# take S-parameters for the standard class
def get_Sparams_standard(pnax, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True):
    assert meas_names, 'get_Sparams_standard needs a valid list of meas_names'
    pnax.set_meas_select(meas_names[0])
    assert sweep_type in ['LIN', 'POW'], 'sweep_type ' + sweep_type + ' not supported for Sparams_standard'
    constParam = pnax.get_power() if sweep_type == 'LIN' else pnax.get_cw_freq()
    data = get_Sparams_helper(pnax, meas_names, vna_fmt, sweep_type, opc_arg)
    if sweep_type == 'LIN':
        data['power'] = constParam
    else:
        data['cw_freq'] = constParam
    return data

def get_Sparams_helper(pnax, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True, fridge=None, time_mult=1):
    avg_steps = pnax.get_average_factor() if bool(pnax.get_averaging_state()) else 1
    this_time = np.max(np.array([float(pnax.get_sweep_time())*avg_steps*len(meas_names)*1500.0*time_mult, 5000]))
    if_bw = pnax.get_if_bandwidth()
    smoothing = pnax.get_smoothing()
    xs = pnax.do_get_xaxis()
    ys = {}
    opc =  opc_arg
    for meas in meas_names:
        pnax.set_meas_select(meas)
        pnax.set_trigger_mode('HOLD')
        tr_num = pnax.get_meas_select_trace()
#        pnax.set_averaging_mode('point')
        ys[meas] = pnax.do_get_data(fmt=vna_fmt, opc=opc, trace_num=tr_num, timeout=this_time)
        pnax.opc()
#        pnax.set_averaging_mode('sweep')
        opc=False
    data = {'meas_names': meas_names,
            'xs': xs,
            'ys': ys,
            'format': vna_fmt,
            'IFbandwidth': if_bw,
            'smoothing': smoothing,
            'averages': avg_steps,
            'sweep_type': sweep_type}
    if fridge:
        data['fridge_ch1_temp'] = fridge.get_ch1_temperature()
        data['fridge_ch2_temp'] = fridge.get_ch2_temperature()
    return data

#TODO - save settings in a pickle
# save S-parameters for SMC or SA class or Standard 
def save_Sparams(h5filename, data, memdata=None, meas_class='SMC', flux_sweep=False,
                 pump_pow_sweep=False, vna_pow_sweep=False, cw_VNApow_sweep=False, metadata=None):
    # h5filename: full path and name of h5file for data to be saved in
    # data: dictionary of data from Sparams_SMC or Sparams_SA measurement
    sweep_type = data['sweep_type']
    h5f = h5py.File(h5filename)
    try:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        dataDir = _iteration_start+'//' + sweep_type + '//'
        g = h5f.require_group(cur_group)
        if meas_class == 'SMC':
            save_Sparams_SMC_helper(g, _iteration_start, dataDir, data, meas_class=meas_class, flux_sweep=flux_sweep,
                                    pump_pow_sweep=pump_pow_sweep, vna_pow_sweep=vna_pow_sweep,
                                    cw_VNApow_sweep=cw_VNApow_sweep, metadata=metadata)
            if memdata:
                memDir = _iteration_start + '//memory//'
                save_Sparams_SMC_helper(g, _iteration_start, memDir, memdata, metadata=metadata)
        elif meas_class == 'SA':
            save_Sparams_SA_helper(g, _iteration_start, dataDir, data)
        elif meas_class == 'Standard':
            save_Sparams_Standard_helper(g, _iteration_start, dataDir, data, metadata)
        elif meas_class == 'SweptIMD':
            save_Sparams_sweptIMD_helper(g, _iteration_start, dataDir, data, pump_pow_sweep)
        elif meas_class == 'IMSpec':
            save_Sparams_IMSpec_helper(g, _iteration_start, dataDir, data)
        else:
            raise ValueError('meas_class: ' + meas_class + ' is not supported by save_Sparams()')
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    return '//' + cur_group + '//' + dataDir

# save S-parameters for Standard class in provided already open h5file in a particular dataDir
def save_Sparams_Standard_helper(g, _iteration_start, dataDir, data, metadata=None):    
    sweep_type = data['sweep_type']
    fmt = data['format']
    if sweep_type == 'LIN':
        dg = g.create_dataset(dataDir + 'frequencies', data=data['xs'])
    else:
        raise ValueError('sweep_type ' + sweep_type + ' is not yet supported for Standard h5 file saving.')  
    if fmt == 'POL':
        dg.attrs.create('format', fmt)
    else:
        raise ValueError('Data format ' + fmt + ' is not yet supported for Standard h5 file saving')    
    dg.attrs.create('sweep_type', sweep_type)
    dg.attrs.create('averages', data['averages'])
    dg.attrs.create('IFbandwidth', data['IFbandwidth'])    
    dg.attrs.create('smoothing', data['smoothing'])
    dg.attrs.create('run_time', _iteration_start)
    for meas in data['meas_names']:
        thisDir = dataDir + meas + '//'
        y0 =  data['ys'][meas][0,:]
        dgm0 = g.create_dataset(thisDir + fs.DATA_LABEL[fmt][0], data = y0)
        dgm0.attrs.create('run_time', _iteration_start)
        y1 = data['ys'][meas][1,:]
        dgm1 = g.create_dataset(thisDir + fs.DATA_LABEL[fmt][1], data = y1)
        dgm1.attrs.create('run_time', _iteration_start)
    if 'fridge_ch1_temp' in data.keys():
        dg.attrs.create('fridge_ch1_temp', data['fridge_ch1_temp'])
    if 'fridge_ch2_temp' in data.keys():
        dg.attrs.create('fridge_ch2_temp', data['fridge_ch2_temp'])
    if metadata:
        dg.attrs.create('metadata', metadata)
       
# save S-parameters for SMC class in provided already open h5file in a particular dataDir
def save_Sparams_SMC_helper(g, _iteration_start, dataDir, data, meas_class='SMC',
                            flux_sweep=False, pump_pow_sweep=False, vna_pow_sweep=False,
                            cw_VNApow_sweep=False, metadata=None):
    # g: valid data group in h5 pointing to day (ie use g=h5f.require_group(cur_group))
    # _iteration_start: seconds time tag in dataDir
    # dataDir: path from group g to desired folder for saving
    # data: dictionary of data from an Sparams_SMC measurement
    any_sweep = flux_sweep or pump_pow_sweep or vna_pow_sweep or cw_VNApow_sweep
    sweep_type = data['sweep_type']
    fmt = data['format']  
    if sweep_type == 'LIN':
        dg = g.create_dataset(dataDir + 'frequencies', data=data['xs'])
        if meas_class == 'SMC':
            dg.attrs.create('power', data['power'])
    elif sweep_type == 'POW':
        dg = g.create_dataset(dataDir + 'powers', data=data['xs'])
        if meas_class == 'SMC':
            dg.attrs.create('freqINP', data['freqINP'])
            dg.attrs.create('freqOUTP', data['freqOUTP'])
    else:
        raise ValueError('sweep_type ' + sweep_type + ' is not yet supported for h5 files.')
    dg.attrs.create('format', fmt)
    dg.attrs.create('IFbandwidth', data['IFbandwidth'])
    dg.attrs.create('sweep_type', sweep_type)
    dg.attrs.create('smoothing', data['smoothing'])
    dg.attrs.create('averages', data['averages'])
    dg.attrs.create('freqLO', data['freqLO'])
    dg.attrs.create('run_time', _iteration_start)
    for meas in data['meas_names']:
        thisDir = dataDir + meas + '//'
        y0 = data['ys'][meas][:,0,:] if any_sweep else data['ys'][meas][0,:]
        dgm0 = g.create_dataset(thisDir + fs.DATA_LABEL[fmt][0], data = y0)
        dgm0.attrs.create('run_time', _iteration_start)
        y1 = data['ys'][meas][:,1,:] if any_sweep else data['ys'][meas][1,:]
        dgm1 = g.create_dataset(thisDir + fs.DATA_LABEL[fmt][1], data = y1)
        dgm1.attrs.create('run_time', _iteration_start)
    if 'fridge_ch1_temp' in data.keys():
        dg.attrs.create('fridge_ch1_temp', data['fridge_ch1_temp'])
    if 'fridge_ch2_temp' in data.keys():
        dg.attrs.create('fridge_ch2_temp', data['fridge_ch2_temp'])
    if metadata:
        dg.attrs.create('metadata', metadata)
    if flux_sweep:
        dgcurr = g.create_dataset(dataDir + 'currents', data=data['currents'])
        dgcurr.attrs.create('run_time', _iteration_start)
    if pump_pow_sweep or vna_pow_sweep:
        dgpows = g.create_dataset(dataDir + 'powers_swept', data=data['powers'])
        dgpows.attrs.create('run_time', _iteration_start)
        if pump_pow_sweep:
            dgpows.attrs.create('fpump', data['fpump'])
    if cw_VNApow_sweep:
        dgcw = g.create_dataset(dataDir + 'cw_freqs', data=data['cw_freqs'])
        dgcw.attrs.create('run_time', _iteration_start)

# plot S-parameters for SMC class
def plot_Sparams_SMC(data, logPhase=True, memdata=None, save=None, metadata='', ymem=0, xatten=0):
    if data['format'] != 'POL':
        print("Format not supported by plot_Sparams_SMC") #TODO - support other formats
        return
    meas_names = data['meas_names']
    sweep_type = data['sweep_type']
    xlabel = 'Input frequency (GHz)'
    xs = data['xs'] + xatten
    if sweep_type == 'LIN':
        xs = xs*1e-9 # frequencies to GHz
    elif sweep_type == 'POW':
        xlabel = 'VNA Power (dBm)'
    else:
        raise ValueError('sweep_type ' + sweep_type + ' is not supported for Sparams_SMC.')
    ys = data['ys']
    ylabels = ['logmag (dB)', 'phase (deg)'] if logPhase else ['real', 'imag']
#    ylabels = ['Gain (dB)', 'phase (deg)']
    fig, ax = plt.subplots(2,1, sharex=True)
    if 'freqLO' in data.keys():
        ax[0].set_title('SMC: LO freq = %.4f GHz'%(data['freqLO']*1e-9))
        ax[1].set_title('SMC: LO freq = %.4f GHz'%(data['freqLO']*1e-9))
    else:
        ax[0].set_title('Standard Sparams ' + metadata)
        ax[1].set_title('Standard Sparams ' + metadata)
    for meas in meas_names:
        #TODO - implement display data/memdata
        (y0, y1) = (ys[meas][0,:], ys[meas][1,:])
        if logPhase:
            (y0, y1) = fs.complex_to_PLOG(y0 + 1j*y1, phase_shift=0, degrees=True)
            if memdata and ('11' in meas or '22' in meas):
                ymc = memdata['ys'][meas][0,:] + 1j*memdata['ys'][meas][1,:]
                (ym0, ym1) = fs.complex_to_PLOG(ymc , phase_shift=0, degrees=True)
                y0 = y0 - ym0 # divide by memory in log units
            y0 = y0 - ymem
        ax[0].plot(xs, y0, label=meas)
        ax[1].plot(xs, y1, label=meas)
    ax[1].set_xlabel(xlabel)
    ax[1].set_xlim([xs[0], xs[-1]])
#    ax[0].set_ylim([-2, 22])
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    if save:
        fig.savefig(save)

def guessCW_maxLog(xs, ys, fpump):
    i_freq = np.argmax(np.abs(ys[0,:] + 1j*ys[1,:]))
    return xs[i_freq]
    
def guessCW_half_fp(xs, ys, fpump, delta=100e3):
    return fpump / 2.0 + delta

def guessCW_near_fp(xs, ys, fpump, delta=100e3):
    return fpump + delta


# take gain point data, both LIN freq sweep and probe POW sweep Sparams_SMC
# PNAX must start with chan 1 in desired SMC LIN freq sweep, 
def take_gain_SMC(pnax, meas_names, vna_fmt='POL', pow_start=-80, pow_stop=-40, vna_pow=None,
                  numPoints=1601, averages=20, pow_numPoints=201, pow_aves=50, fpump=None,
                  guessCW=guessCW_maxLog, measure_P1dB=True):
    # pnax: valid pnax instrument
    # meas_names: list of valid measurement names on chan 1 (ie '"CH1_S11_3"' for an S11 measurement that is trace 1)
    # vna_fmt: valid data format (ie 'POL')
    data = {}
    set_up_Sparams_SMC(pnax, numPoints, averages, vna_pow)
    dataLIN = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
    data['LIN'] = dataLIN    
    
    if measure_P1dB:
        #TODO - need to do more to set up power sweep, need to set fixed freqs
        inp_meas_name = [meas for meas in meas_names if 'S11' in meas][0]
        xs = dataLIN['xs']
        freqLO = dataLIN['freqLO']
        # guessCW freq takes arguments xs, ys, fpump and returns the desired CW freq
        ys = dataLIN['ys'][inp_meas_name]        
        freqINP = guessCW(xs, ys, fpump)        
        set_up_P1dB_SMC(pnax, freqINP, freqLO, pow_numPoints, pow_aves, pow_start, pow_stop)
        pnax.opc()
        dataPOW = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='POW')
        data['POW'] = dataPOW

    set_up_Sparams_SMC(pnax, numPoints, averages, vna_pow)
    return data    

def set_up_P1dB_SMC(pnax, freqINP, freqLO, numPoints=201, averages=50,
                    pow_start=None, pow_stop=None):
    pnax.set_average_factor(averages)
    pnax.set_points(numPoints)
    pnax.set_mixerOUTP_freq_mode('FIXED')
    pnax.set_mixerINP_freq_mode('FIXED')
    pnax.set_mixerINP_freq_fixed(freqINP)
    pnax.set_mixerLO_freq_fixed(freqLO)
    pnax.mixer_calc('OUTP')
    pnax.set_sweep_type('POW')
    pnax.mixer_apply_settings()
    if pow_start is not None:
        pnax.set_start_pow(pow_start)
    if pow_stop is not None:
        pnax.set_stop_pow(pow_stop)
    pnax.set_trigger_mode('CONT')

    
def set_up_Sparams_SMC(pnax, numPoints=1601, averages=20, vna_pow=None):
    pnax.set_average_factor(averages)
    pnax.set_points(numPoints)
    pnax.set_mixerOUTP_freq_mode('SWEPT')
    pnax.set_mixerINP_freq_mode('SWEPT')
    pnax.set_sweep_type('LIN')
    pnax.mixer_apply_settings()
    if vna_pow is not None:
        pnax.set_power(vna_pow)

        
def set_up_sweptIMD_POW(pnax, fc, df=100e3, numPoints=201, averages=10,
                        pow_start=None, pow_stop=None, ifbw_im=None, ifbw_main=None):
    pnax.set_average_factor(averages)
    pnax.set_imd_sweep_type('POW')
    pnax.set_points(numPoints)
    pnax.set_imd_fc_cw(fc)
    pnax.set_imd_df_cw(df)
    if pow_start:
        pnax.set_imd_f1_pow_start(pow_start)
    if pow_stop:
        pnax.set_imd_f1_pow_stop(pow_stop)
    if ifbw_im:
        pnax.set_imd_IFBW_im(ifbw_im)
    if ifbw_main:
        pnax.set_imd_IFBW_main(ifbw_main)
    
# save gain point data in h5file under same dataDir
def save_gain_SMC(h5filename, data, memdata=None, specdata=None, noisedata=None, linfits=None, specfits=None, noisefits=None, powfits=None,
                  pump_pow_sweep=False):
    h5f = h5py.File(h5filename)
    try:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        g = h5f.require_group(cur_group)
        dataDirLIN = _iteration_start+'//LIN//'
        save_Sparams_SMC_helper(g, _iteration_start, dataDirLIN, data['LIN'], pump_pow_sweep=pump_pow_sweep)
        if memdata:
            memDir = _iteration_start + '//memory//'
            save_Sparams_SMC_helper(g, _iteration_start, memDir, memdata)
        if linfits:
            fitDir = _iteration_start + '//fits//LIN//'
            save_gain_fit_helper(g, _iteration_start, fitDir, linfits) #TODO - need to add sweep saving capabilities
        dataDirPOW = _iteration_start+'//POW//'
        if 'POW' in data.keys():
            save_Sparams_SMC_helper(g, _iteration_start, dataDirPOW, data['POW'], pump_pow_sweep=pump_pow_sweep)
        if powfits:
            fitDir = _iteration_start + '//fits//POW//'
            save_p1dB_fit_helper(g, _iteration_start, fitDir, powfits)
        if specdata:
            specDir = _iteration_start + '//spec//'
            specFitsDir = _iteration_start + '//fits//spec//'
            cfreqs = specdata.keys()
            for cfreq in cfreqs:
                this_dir = specDir + str(cfreq) + '//'
                save_Sparams_SA_helper(g, _iteration_start, this_dir, specdata[cfreq]['data'])
                if 'memory' in specdata[cfreq].keys():
                    memDir = this_dir + 'memory//'
                    save_Sparams_SA_helper(g, _iteration_start, memDir, specdata[cfreq]['memory'])
                if specfits:
                    currDir = specFitsDir + str(cfreq) + '//'
                    save_nvr_fit_helper(g, _iteration_start, currDir, specfits[cfreq])
        if noisedata:
            noiseDir = _iteration_start + '//noise//'
            noiseFitsDir = _iteration_start + '//fits//noise//'
            cfreqs = noisedata.keys()
            for cfreq in cfreqs:
                this_dir = noiseDir + str(cfreq) + '//'
                save_Sparams_NF_helper(g, _iteration_start, this_dir, noisedata[cfreq]['data'])
                if 'memory' in noisedata[cfreq].keys():
                    memDir = this_dir + 'memory//'
                    save_Sparams_NF_helper(g, _iteration_start, memDir, noisedata[cfreq]['memory'])
                if noisefits:
                    currDir = noiseFitsDir + str(cfreq) + '//'
                    save_nvr_fit_helper(g, _iteration_start, currDir, noisefits[cfreq])
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    return '//' + cur_group + '//' + _iteration_start
    
# save gain oint data, includes SMC measurements and SA measurements

    
# plot gain data (separtate figures)
def plot_gain_SMC(data, memdata=None, specdata=None, noisedata=None):
    plot_Sparams_SMC(data['LIN'])
    plot_Sparams_SMC(data['POW'])
    if specdata:
        cfreqs = specdata.keys()
        for cfreq in cfreqs:
            plot_Sparams_SA(specdata[cfreq])
    if noisedata:
        cfreqs = noisedata.keys()
        for cfreq in cfreqs:
            plot_Sparams_NF(noisedata[cfreq])


# measure noise spectrum using Noise Figure class
def take_Sparams_NF(pnax, meas_names, vna_fmt='MLOG'):
    # pnax: valid pnax instrument in Noise Figure configuration
    # meas_names: list of valid measurement names on the NF channel to be measured
    # vna_fmt: valid data format ('MLOG' for Noise Figure class)
    if len(meas_names) == 0:
        return
    pnax.set_meas_select(meas_names[0])
    avg_steps = pnax.get_average_factor() if bool(pnax.get_averaging_state()) else 1

    this_time = 1000000
    this_time = np.max(np.array([(float(pnax.get_sweep_time())+0.25)*avg_steps*len(meas_names)*1500.0, 5000]))

    rbw = pnax.get_receiver_bandwidth()
    xs = pnax.do_get_xaxis()
    ys = {}
    opc = True
    for meas in meas_names:
        pnax.set_meas_select(meas)
        pnax.set_trigger_mode('HOLD')
        tr_num = pnax.get_meas_select_trace()
        pnax.opc()
#        print(str(this_time) + ' ms')
        ys[meas] = pnax.do_get_data(fmt=vna_fmt, opc=opc, trace_num=tr_num, timeout=this_time)
        pnax.opc()
        opc = False
    data = {'meas_names': meas_names,
            'xs': xs,
            'ys': ys,
            'format': vna_fmt,
            'rbw': rbw,
            'averages': avg_steps,
            'sweep_type': 'noise'}
    return data

# saves NF measurment in h5file
def save_Sparams_NF_helper(g, _iteration_start, dataDir, data):
    sweep_type = data['sweep_type']
    fmt = data['format']
    if fmt != 'MLOG':
        raise ValueError('Data format ' + fmt + ' is not yet supported for NF h5 file saving')
    dg = g.create_dataset(dataDir + 'frequencies', data=data['xs'])
    dg.attrs.create('format', fmt)
    dg.attrs.create('sweep_type', sweep_type)
    dg.attrs.create('averages', data['averages'])
    dg.attrs.create('rbw', data['rbw'])
    dg.attrs.create('run_time', _iteration_start)
    for meas in data['meas_names']:
        thisDir = dataDir + meas + '//'
        dgm = g.create_dataset(thisDir + fs.DATA_LABEL[fmt], data=data['ys'][meas])
        if 'V_src' in data.keys():
            g.create_dataset(thisDir + 'V_src', data=data['V_src'])
            g.create_dataset(thisDir + 'V_snt', data=data['V_snt'])
        dgm.attrs.create('run_time', _iteration_start)


def plot_Sparams_NF(data, save=None):
    mem = 'memory' in data.keys()
    if mem:
        memdata = data['memory']
    if 'data' in data.keys():
        data = data['data']
    if data['format'] != 'MLOG':
        print('Format not supported by plot_Sparams_SMC') #TODO - support other formats MLIN
        return
    meas_names = data['meas_names']
    xlabel = 'Input frequency (GHz)'
    xs = data['xs'] * 1e-9 #freqs to GHz
    ys = data['ys']
    ylabel = 'Power (dB)'
    fig, ax = plt.subplots(1,1)
    ax.set_title('NF traces')
    for meas in meas_names:
        y0 = ys[meas]
        if mem:
            y0 = y0 - memdata['ys'][meas] # data/memory in dB
            print('noise: dividing data/memory')
        ax.plot(xs, y0, label=meas)
    ax.set_xlabel(xlabel)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    if save:
        fig.savefig(save)


# take spec analyzer data from pnax    
def take_Sparams_SA(pnax, meas_names, vna_fmt='MLOG'):
    # pnax: valid pnax instrument in Spec An configuration
    # meas_names: list of valid measurement names on the SA channel to be measured
    # vna_fmt: valid data format ('MLOG' for spec an)
    if len(meas_names) == 0:
        return
    pnax.set_meas_select(meas_names[0])
    avg_steps = pnax.get_average_factor() if bool(pnax.get_averaging_state()) else 1
    this_time = np.max(np.array([(float(pnax.get_sweep_time())+0.25)*avg_steps*len(meas_names)*1500.0, 5000]))
    rbw = pnax.get_spec_rbw()
    vbw = pnax.get_spec_vbw()
    xs = pnax.do_get_xaxis()
    ys = {}
    opc = True
    for meas in meas_names:
        pnax.set_meas_select(meas)
        pnax.set_trigger_mode('HOLD')
        tr_num = pnax.get_meas_select_trace()
        pnax.opc()
#        print(str(this_time) + ' ms')
        ys[meas] = pnax.do_get_data(fmt=vna_fmt, opc=opc, trace_num=tr_num, timeout=this_time)
        pnax.opc()
        opc = False
    data = {'meas_names': meas_names,
            'xs': xs,
            'ys': ys,
            'format': vna_fmt,
            'rbw': rbw,
            'vbw': vbw,
            'averages': avg_steps,
            'sweep_type': 'spec'}
    return data

# saves SA measurment in h5file
def save_Sparams_SA_helper(g, _iteration_start, dataDir, data):
    sweep_type = data['sweep_type']
    fmt = data['format']
    if fmt != 'MLOG':
        raise ValueError('Data format ' + fmt + ' is not yet supported for SA h5 file saving')
    dg = g.create_dataset(dataDir + 'frequencies', data=data['xs'])
    dg.attrs.create('format', fmt)
    dg.attrs.create('sweep_type', sweep_type)
    dg.attrs.create('averages', data['averages'])
    dg.attrs.create('rbw', data['rbw'])
    dg.attrs.create('vbw', data['vbw'])
    dg.attrs.create('run_time', _iteration_start)
    for meas in data['meas_names']:
        thisDir = dataDir + meas + '//'
        dgm = g.create_dataset(thisDir + fs.DATA_LABEL[fmt], data=data['ys'][meas])
        dgm.attrs.create('run_time', _iteration_start)
    
        
def plot_Sparams_SA(data, save=None):
    mem = 'memory' in data.keys()
    if mem:
        memdata = data['memory']
    if 'data' in data.keys():
        data = data['data']
    if data['format'] != 'MLOG':
        print('Format not supported by plot_Sparams_SMC') #TODO - support other formats MLIN
        return
    meas_names = data['meas_names']
    xlabel = 'Input frequency (GHz)'
    xs = data['xs'] * 1e-9 #freqs to GHz
    ys = data['ys']
    ylabel = 'Power (dB)'
    fig, ax = plt.subplots(1,1)
    ax.set_title('SA traces')
    for meas in meas_names:
        y0 = ys[meas]
        if mem:
            y0 = y0 - memdata['ys'][meas] # data/memory in dB
            print('spec: dividing data/memory')
        ax.plot(xs, y0, label=meas)
    ax.set_xlabel(xlabel)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    if save:
        fig.savefig(save)

def take_Sparams_sweptIMD(pnax, meas_names, vna_fmt='MLOG', opc_arg=True, fridge=None, num_im=1):
    assert len(meas_names) > 0, 'meas_names has length %d'%(len(meas_names))
    pnax.set_meas_select(meas_names[0])
    sweep_type = pnax.get_imd_sweep_type()
    if_main = pnax.get_imd_IFBW_main()
    if_im = pnax.get_imd_IFBW_im()
    time_mult = 1 + num_im * if_main / if_im    # multiplier to sweep time that accounts for the number of distinct IM products that are requested
    data = get_Sparams_helper(pnax, meas_names, vna_fmt, sweep_type, opc_arg, fridge, time_mult)
    data['meas_class'] = pnax.get_meas_class()
    data['imd_IFBW_main'] = if_main
    data['imd_IFBW_im'] = if_im
    data['imd_tpow_coupled'] = pnax.get_imd_tpow_coupled()
    if sweep_type == 'POW':
        data['imd_f1_cw'] = pnax.get_imd_f1_cw()
        data['imd_f2_cw'] = pnax.get_imd_f2_cw()
        data['imd_fc_cw'] = pnax.get_imd_fc_cw()
        data['imd_df_cw'] = pnax.get_imd_df_cw()
        data['imd_f1_pow_start'] = pnax.get_imd_f1_pow_start()
        data['imd_f1_pow_stop'] = pnax.get_imd_f1_pow_stop()
        data['imd_f2_pow_start'] = pnax.get_imd_f2_pow_start()
        data['imd_f2_pow_stop'] = pnax.get_imd_f2_pow_stop()
    elif sweep_type == 'FCEN':
        data['imd_df_cw'] = pnax.get_imd_df_cw()
        data['imd_fc_start'] = pnax.get_imd_fc_start()
        data['imd_fc_stop'] = pnax.get_imd_fc_stop()
        data['imd_fc_center'] = pnax.get_imd_fc_center()
        data['imd_fc_span'] = pnax.get_imd_fc_span()
        data['imd_f1_pow'] = pnax.get_imd_f1_pow()
        data['imd_f2_pow'] = pnax.get_imd_f2_pow()
    elif sweep_type == 'DFR':
        data['imd_df_start'] = pnax.get_imd_df_start()
        data['imd_df_stop'] = pnax.get_imd_df_stop()
        data['imd_fc_cw'] = pnax.get_imd_fc_cw()
        data['imd_f1_pow'] = pnax.get_imd_f1_pow()
        data['imd_f2_pow'] = pnax.get_imd_f2_pow()
    elif sweep_type == 'CW':
        data['imd_f1_cw'] = pnax.get_imd_f1_cw()
        data['imd_f2_cw'] = pnax.get_imd_f2_cw()
        data['imd_fc_cw'] = pnax.get_imd_fc_cw()
        data['imd_df_cw'] = pnax.get_imd_df_cw()
        data['imd_f1_pow'] = pnax.get_imd_f1_pow()
        data['imd_f2_pow'] = pnax.get_imd_f2_pow()
    return data

def save_Sparams_sweptIMD_helper(g, _iteration_start, dataDir, data, pump_pow_sweep=False):
    fmt = data['format']
    assert fmt == 'MLOG', 'Data format ' + fmt + ' is not yet supported for SweptIMD h5file saving'
    keys = data.keys()
    dg = g.create_dataset(dataDir + 'xs', data=data['xs'])
    dg.attrs.create('run_time', _iteration_start)
    keys.remove('xs')
    keys.remove('meas_names')
    for meas in data['meas_names']:
        thisDir = dataDir + meas + '//'
        dgm = g.create_dataset(thisDir + fs.DATA_LABEL[fmt], data=data['ys'][meas]) # should work for swept data or not swept since in MLOG
        dgm.attrs.create('run_time', _iteration_start)
    keys.remove('ys')
    if pump_pow_sweep:
        dgpows = g.create_dataset(dataDir + 'powers_swept', data=data['powers'])
        dgpows.attrs.create('run_time', _iteration_start)
        keys.remove('powers')
        dgpows.attrs.create('fpump', data['fpump'])
        keys.remove('fpump')
    for key in keys:
        val = data[key]
        if type(val) == unicode:
            val = str(val)
        dg.attrs.create(key, val)

def plot_Sparams_sweptIMD(data, attenInp=0, save=False):
    assert data['format'] == 'MLOG', 'Data format ' + data['format'] + ' is not yet supported for SweptIMD plotting'
    sweep_type = data['sweep_type']
    meas_names = data['meas_names']
    xs = data['xs']
    xlabel = 'Tone Power (dB)'
    if sweep_type == 'POW':
        xs = xs + attenInp
    elif sweep_type == 'FCEN':
        xs = xs*1e-9 #freqs to GHz
        xlabel = 'Carrier Frequency (GHz)'
    elif sweep_type == 'DFR':
        xs = xs*1e-6 #freqs to MHz
        xlabel = 'Delta Frequency (MHz)'
    elif sweep_type == 'CW':
        xlabel = 'Point Number'
    else:
        raise ValueError('sweep_type: ' + sweep_type + ' not supported by plot_Sparams_sweptIMD')
    ylabel = 'Power (dB)'
    fig, ax = plt.subplots(1,1)
    ax.set_title('SweptIMD ' + sweep_type + ' traces')
    for meas in meas_names:
        y0 = data['ys'][meas]
        ax.plot(xs, y0, label=meas)
    ax.set_xlabel(xlabel)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    if save:
        fig.savefig(save)
        
def fit_sweptIMD(data, x_point=None, fitTypes = ['IIP3', 'IM3'], fitTypeUnits = ['dBm', 'dB'], printOut=True):
    # extracts IIP3, IM3 at a particular point in the x axis sweep x_point (eg the pnax power for a POW sweep) from a sweptIMD data
    # x_point: power at the pnax where data should be extracted, if None then will use the middle of the x_axis span
    # fitTypes: list of valid measurement types from swept IMD that a point at x_point should be extracted from
    assert data['format'] == 'MLOG', 'Data format ' + data['format'] + ' is not yet supported for SweptIMD fitting'
    meas_names = data['meas_names']
    xs = data['xs']
    i_x = len(xs) / 2 if x_point is None else fs.find_nearest(xs, x_point)
    results = {}
    results['x'] = xs[i_x]
    for i_ft, ft in enumerate(fitTypes):
        meas = [m for m in meas_names if ft in m][0]
        results[ft] = data['ys'][meas][i_x]
        if printOut:
            print(ft +' @ %.2f dBm = %.2f '%(xs[i_x], results[ft]) + fitTypeUnits[i_ft])
    return results

def take_Sparams_IMSpec(pnax, meas_names, vna_fmt='MLOG', opc_arg=True, fridge=None):
    assert len(meas_names) > 0, 'meas_names has length %d'%(len(meas_names))
    pnax.set_meas_select(meas_names[0])
    sweep_type = pnax.get_ims_sweep_type()
    data = get_Sparams_helper(pnax, meas_names, vna_fmt, sweep_type, opc_arg, fridge)
    data['meas_class'] = pnax.get_meas_class()
    data['ims_rbw'] = pnax.get_ims_rbw()
    data['ims_resp_start'] = pnax.get_ims_resp_start()
    data['ims_resp_stop'] = pnax.get_ims_resp_stop()
    data['ims_resp_center'] = pnax.get_ims_resp_center()
    data['ims_resp_span'] = pnax.get_ims_resp_span()
    data['ims_stim_df'] = pnax.get_ims_stim_df()
    data['ims_stim_fc'] = pnax.get_ims_stim_fc()
    data['ims_stim_f1'] = pnax.get_ims_stim_f1()
    data['ims_stim_f2'] = pnax.get_ims_stim_f2()
    data['ims_sweep_order'] = pnax.get_ims_sweep_order()
    data['ims_tpow_coupled'] = pnax.get_ims_tpow_coupled()
    data['ims_stim_f1_pow'] = pnax.get_ims_stim_f1_pow()
    data['ims_stim_f2_pow'] = pnax.get_ims_stim_f2_pow()
    data['ims_tpow_level_mode'] = pnax.get_ims_tpow_level_mode()
    return data
    
def save_Sparams_IMSpec_helper(g, _iteration_start, dataDir, data):
    return save_Sparams_sweptIMD_helper(g, _iteration_start, dataDir, data)
    
def plot_Sparams_IMSpec(data, save=False):
    assert data['format'] == 'MLOG', 'Data format ' + data['format'] + ' is not yet supported for IMSpec plotting'
    sweep_type = data['sweep_type']
    sweep_order = data['ims_sweep_order'] if sweep_type == 'NTH' else 3
    xs = data['xs']*1e-9
    xlabel = 'Frequency (GHz)'
    ylabel = 'Power (dB)'
    fig, ax = plt.subplots(1,1)
    ax.set_title('IM Spectrum order ' + str(sweep_order))
    for meas in data['meas_names']:
        y0 = data['ys'][meas]
        ax.plot(xs, y0, label=meas)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([xs[0], xs[-1]])
    ax.legend(loc='best')
    if save:
        fig.savefig(save)
        
# fit a set of SMC gain curves, where data is the LINear sweep data, memdata is also from Sparams_SMC
# assumes that S22 is the converted frequency
def fit_gain_SMC(data, memdata=None, display=True):
    results = {}
    meas_names = data['meas_names']
    ref_meas_names = [meas for meas in meas_names if not 'SC' in meas]
    freqLO = data['freqLO']
    xs = data['xs']
    ys = data['ys']
    if data['format'] != 'POL':
        print('Format not supported by fit_gain_SMC') #TODO - support other formats
        raise ValueError('data format ' + data['format'] + ' is not supported by fit_gain_SMC')
    for meas in ref_meas_names:
        if len(ys[meas].shape) == 2:
            yc = ys[meas][0,:] + 1j * ys[meas][1,:]
        elif len(ys[meas].shape) == 3:
            yc = ys[meas][:,0,:] + 1j * ys[meas][:,1,:]
        else:
            raise ValueError('Dimension of ys not supported by fit_gain_SMC')
        if memdata:
            ymem = memdata['ys'][meas][0,:] + 1j * memdata['ys'][meas][1,:]
            yc = np.divide(yc, np.abs(ymem))
        if 'S22' in meas:
            xs = freqLO - xs[::-1]
            yc = yc[::-1] if len(yc.shape) == 1 else yc[:, ::-1]
        (ylog, yphase) = fs.complex_to_PLOG(yc, phase_shift=0)
        if len(ylog.shape) == 1:
            (f0, g_log, bw) = fs.fit_gain_curve(xs, ylog, display=display)
        else:
            f0 = np.zeros(ylog.shape[0])
            g_log = np.zeros_like(f0)
            bw = np.zeros_like(f0)
            for kk in np.arange(ylog.shape[0]):
                (f0[kk], g_log[kk], bw[kk]) = fs.fit_gain_curve(xs, ylog[kk,:], display=display)
        results[meas] = {'f0': f0, 'g_log':g_log, 'bw':bw}
    return results

# save gain fits
def save_gain_fit_helper(g, _iteration_start, dataDir, fit_results):
    meas_names = fit_results.keys()
    for meas in meas_names:
        thisDir = dataDir + meas + '//'
        dg0 = g.create_dataset(thisDir + 'glog', data = np.array(fit_results[meas]['g_log']))
        dg1 = g.create_dataset(thisDir + 'f0', data = fit_results[meas]['f0'])
        dg2 = g.create_dataset(thisDir + 'bw', data = fit_results[meas]['bw'])

# fit resonance in reflection using SMC, data is data form get_Sparams_SMC, assumes S22 is at converted freq
def fit_reflection_SMC(data, display=True, kc=60e6, ki=1e6, guessF=fs.guessF_middle):
    results = {}
    meas_names = data['meas_names']
    freqLO = data['freqLO']
    ref_meas_names = [meas for meas in meas_names if not 'SC' in meas]
    xs = data['xs']
    ys = data['ys']
    if data['format'] != 'POL':
        raise ValueError('data format ' + data['format'] + 'is not supported by fit_reflection_SMC')
    for meas in ref_meas_names:
        freqs = xs
        yc = ys[meas][0,:] + 1j*ys[meas][1,:]
        if 'S22' in meas:
            freqs = freqLO - freqs[::-1]
            yc = yc[::-1]
        y_back = (sum(yc[0:25]) + sum(yc[-25:])) / 50.0      
        f0guess = freqs[guessF(yc)]
        results[meas] = af.analyze_reflection(freqs, [data['power']], yc, f_0=f0guess,
                                                kc=kc, ki=ki, a_in=y_back, T=1e-12, display=display)
    return results

def save_fit_reflection(fit_results, h5filename, dataDir):
    for meas in fit_results.keys():
        thisDir = dataDir + meas + '//'
        fs.save_power_Qfit(fit_results[meas], h5filename, dataDir=thisDir)

# fits an nvr trace saved in specdata structure data[cfreq][meas_name] = dictionary of data xs,ys etc
def fit_nvr(specdata, display=True):
    results = {}
    cfreqs = specdata.keys()
    for cfreq in cfreqs:
        data = specdata[cfreq]['data']
        mem = specdata[cfreq]['memory']
        meas_names = data['meas_names']
        if data['format'] != 'MLOG':
            raise ValueError('data format ' + data['format'] + ' is not supported by fit_nvr')
        results[cfreq] = {}
        xs = data['xs']
        for meas in meas_names:
            results[cfreq][meas] = {}
            (nvr, _, f0) = fs.get_NVR(xs, data['ys'][meas], mem['ys'][meas], display=display)
            results[cfreq][meas]['nvr'] = nvr
            results[cfreq][meas]['f0'] = f0
    return results

def save_nvr_fit_helper(g, _iteration_start, dataDir, fit_results):
    for meas in fit_results.keys():
        thisDir = dataDir + meas + '//'
        dg = g.create_dataset(thisDir + 'nvr', data=np.array(fit_results[meas]['nvr']))
        dg.attrs.create('f0', fit_results[meas]['f0'])

def fit_P1dB_SMC(data, display=True):
    results = {}
    meas_names = data['meas_names']
    xs = data['xs']
    ys = data['ys']
    if data['format'] != 'POL':
        raise ValueError('data format ' + data['format'] + 'is not supported by fit_P1dB_SMC')
    for meas in meas_names:
        if len(ys[meas].shape) == 2:
            yc = ys[meas][0,:] + 1j*ys[meas][1,:]
        elif len(ys[meas].shape) == 3:
            yc = ys[meas][:,0,:] + 1j * ys[meas][:,1,:]
        else:
            raise ValueError('Dimension of ys not supported by fit_P1dB_SMC')
        (ylog, yphase) = fs.complex_to_PLOG(yc, phase_shift=0)
        if len(ylog.shape) == 1:
            (p1dB, p_low, i_cut) = fs.calc_P1dB(xs, ylog, display=display)
        else:
            p1dB = np.zeros(ylog.shape[0])
            p_low = np.zeros_like(p1dB)
            for kk in np.arange(ylog.shape[0]):
                (p1dB[kk], p_low[kk], i_cut) = fs.calc_P1dB(xs, ylog[kk,:], display=display)
        results[meas] = {'P_1dB': p1dB, 'P_low': p_low}
    return results

def save_p1dB_fit_helper(g, _iteration_start, dataDir, fit_results):
    meas_names = fit_results.keys()
    for meas in meas_names:
        thisDir = dataDir + meas + '//'
        dg0 = g.create_dataset(thisDir + 'P_1dB', data = np.array(fit_results[meas]['P_1dB']))
        dg1 = g.create_dataset(thisDir + 'P_low', data = fit_results[meas]['P_low'])
        
def pump_pow_sweep_SMC(pnax, ag, meas_names, spec_names=None, pows=np.linspace(-40, -20, 21),
                       display=True, vna_fmt='POL', printlog=True, takeDR=False,
                       powDR_start=-80, powDR_stop=-40, vna_pow=None, pow_numPoints=201, pow_aves=50,
                       fpump=None, guessCW=guessCW_maxLog, reCentering=False, kc_guess=None):
    figs = {}
    figsDR = {}
    axes = {}
    axesDR = {}
    if not meas_names:
        return
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    if vna_pow is not None:
        pnax.set_power(vna_pow)
    if pnax.get_sweep_type() == 'SEGM':
        sweep_time = float(pnax.get_segment_sweep_time())
    else:
        sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 0.5) * len(meas_names) * len(pows)
    if takeDR:
        tot_time = tot_time*2.0
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
    ys = {}
    ys_pow = {}
    data = {}
    results = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(pows), 2, num_points))
        ys_pow[meas] = np.zeros((len(pows), 2, pow_numPoints))
        if display:
            figs[meas] = plt.figure()
            axes[meas] = figs[meas].add_subplot(111)
            if takeDR:
                figsDR[meas] = plt.figure()
                axesDR[meas] = figsDR[meas].add_subplot(111)
    ag.set_rf_on(False)
    memdata = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
    if reCentering == True:
        fitdata = fit_reflection_SMC(memdata, kc=kc_guess, ki=1e6, guessF=fs.guessF_maxReal, display=False)           
        points = pnax.get_points()
        span = pnax.get_span()
        averages = pnax.get_average_factor()
        vna_pow = pnax.get_power(vna_pow)
        set_center_span(pnax, fitdata[meas_names[0]][0][0], span=span, points=points, averages=averages, power=vna_pow)
        pnax.opc() 
        memdata = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
    ag_freq = fpump if fpump else ag.get_frequency()
    ag.set_frequency(ag_freq)
    ag.set_rf_on(True)
    # sweep pump power
    for i_pow, power in enumerate(pows):
        ag.set_power(power)
        sleep(0.1) #allow generator to turn on and stabilize
        if takeDR:
            results = take_gain_SMC(pnax, meas_names, vna_fmt, powDR_start, powDR_stop, vna_pow,
                                    num_points, averages, pow_numPoints, pow_aves, ag_freq, guessCW)
            data = results['LIN']
        else:
            data = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
        if reCentering == True:        
            fitdata = fit_reflection_SMC(data, kc=kc_guess, ki=1e6, guessF=fs.guessF_maxReal, display=False)   
            set_center_span(pnax, fitdata[meas_names[0]][0][0], span=span, points=points, averages=averages, power=vna_pow)
            pnax.opc()         
            print('Re-centered to %f' % fitdata[meas_names[0]][0][0])
        for meas in meas_names:
            ys[meas][i_pow, :, :] = data['ys'][meas]
            if takeDR:
                ys_pow[meas][i_pow, :, :] = results['POW']['ys'][meas]
            try:
                if display:
                    yc = ys[meas][i_pow,0,:] + 1j*ys[meas][i_pow,1,:]
                    (ylog, yphase) = fs.complex_to_PLOG(yc)
                    axes[meas].plot(data['xs'], ylog, label='%.3f dBm'%(power))
                    figs[meas].canvas.draw()
                    if takeDR:
                        yc = ys_pow[meas][i_pow,0,:] + 1j*ys_pow[meas][i_pow,1,:]
                        (ylog, yphase) = fs.complex_to_PLOG(yc)
                        axesDR[meas].plot(results['POW']['xs'], ylog, label='%.3f dBm'%(power))
                        figsDR[meas].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
    ag.set_rf_on(False)
    # put full pow sweep data in ys
    data['ys'] = ys
    data['powers'] = pows
    data['fpump'] = ag_freq
    if takeDR:
        results['POW']['ys'] = ys_pow
        results['POW']['powers'] = pows
        results['POW']['fpump'] = ag_freq
    if display:
        try:
            for meas in meas_names:
                axes[meas].set_xlabel('Probe frequency (Hz)')
                axes[meas].set_ylabel('Sparams Logmag (dB)')
                axes[meas].legend()
                axes[meas].set_title(meas + ' log mag, fpump = %.5f'%(data['fpump']*1e-9))
        except:
            pass
    results['LIN'] = data
    results['memory'] = memdata
    return results

# plots a pump power sweep in colormap.
def plot_pump_pow_sweep_SMC(data, memdata=None, save=None, cmaps = ('hot','phase'), cw_VNApow_sweep=False):
    if data['format'] != 'POL':
        raise ValueError('Format not supported by plot_pump_pow_sweep_SMC')
    fs.define_phaseColorMap()
    figs = {}
    axes = {}
    xlabel = 'Probe Frequency (GHz)'
    check_invert_xs = True
    if cw_VNApow_sweep:
        pows = data['xs']
        xs = data['cw_freqs'] * 1e-9
    else: #TODO - fix swept powers naming
        pows = data['powers'] if 'powers' in  data.keys() else data['powers_swept']
        xs = data['xs']
        if data['sweep_type'] == 'LIN':
            xs = xs * 1e-9
        else:
            xlabel = 'Probe Power (dBm)'
            check_invert_xs = False
    freqLO = data['freqLO']
    meas_names = data['meas_names']
    for meas in meas_names:
        yc = data['ys'][meas][:,0,:] + 1j*data['ys'][meas][:,1,:]
        if cw_VNApow_sweep:
            yc = np.transpose(yc)
        if check_invert_xs and 'S22' in meas:
            xs = freqLO * 1e-9 - xs[::-1]
            yc = yc[:, ::-1]
        if memdata:
            ymem = memdata['ys'][meas][0,:] + 1j*memdata['ys'][meas][1,:]
            yc = np.divide(yc, ymem)
        (ylog, yphase) = fs.complex_to_PLOG(yc)
        (figs[meas], axes[meas]) = plt.subplots(2,1, sharex=True)
        p0 = (axes[meas][0]).pcolormesh(xs, pows, ylog, cmap=cmaps[0])
        cb0 = figs[meas].colorbar(p0, ax=axes[meas][0])
        cb0.set_label('Log mag (dB)', rotation=270)
        p1 = (axes[meas][1]).pcolormesh(xs, pows, yphase, cmap=cmaps[1])
        cb1 = figs[meas].colorbar(p1, ax=axes[meas][1], ticks=[-180, -90, 0, 90, 180])
        cb1.set_label('Phase (deg)', rotation=270)
#        axes[meas][1].set_xlim([6.1, 6.5])
        if 'fpump' in data.keys():
            axes[meas][0].set_title(meas + ', f_pump = %.5f GHz'%(data['fpump']*1e-9))
            axes[meas][0].set_ylabel('Pump Power (dBm)')
            axes[meas][1].set_ylabel('Pump Power (dBm)')
        else:
            axes[meas][0].set_title(meas + ' VNA pow sweep')
            axes[meas][0].set_ylabel('VNA power (dBm)')
            axes[meas][1].set_ylabel('VNA power (dBm)')
        axes[meas][-1].set_xlabel(xlabel)

# pump power and frequency sweep
def pump_powFreq_sweep_SMC(pnax, ag, meas_names, spec_names=None, pows=None, ag_freqs=None,
                           display=True, vna_fmt='POL', printlog=True, takeDR=False,
                           powDR_start=-80, powDR_stop=-40, vna_pow=None, pow_numPoints=201, pow_aves=50,
                           guessCW=guessCW_maxLog):
    if not meas_names or not pows or not ag_freqs:
        return
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    if not vna_pow:
        pnax.set_power(vna_pow)
    if pnax.get_sweep_type() == 'SEGM':
        sweep_time = float(pnax.get_segment_sweep_time())
    else:
        sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 0.5) * len(meas_names) * len(pows)
    if takeDR:
        tot_time = tot_time*2.0
    #TODO - must finish this...not functional

# for each vna power take a LIN SMC sweep     
def vna_pow_sweep_SMC(pnax, meas_names, pows=np.linspace(-60, -40, 21),
                      display=True, vna_fmt='POL', printlog=True):
    figs = {}
    axes = {}
    if not meas_names:
        return
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    if pnax.get_sweep_type() == 'SEGM':
        sweep_time = float(pnax.get_segment_sweep_time())
    else:
        sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 0.5) * len(meas_names) * len(pows)
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
    ys = {}
    data = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(pows), 2, num_points))
        if display:
            figs[meas] = plt.figure()
            axes[meas] = figs[meas].add_subplot(111)
    # sweep vna power
    for i_pow, power in enumerate(pows):
        pnax.set_power(power)
        data = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
        for meas in meas_names:
            ys[meas][i_pow, :, :] = data['ys'][meas]
            try:
                if display:
                    yc = ys[meas][i_pow, 0, :] + 1j*ys[meas][i_pow, 1, :]
                    (ylog, yphase) = fs.complex_to_PLOG(yc)
                    axes[meas].plot(data['xs'], ylog, label='%.3f dBm'%(power))
                    figs[meas].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
    pnax.set_power(np.min(pows))
    data['ys'] = ys
    data['powers'] = pows
    if display:
        try:
            for meas in meas_names:
                axes[meas].set_xlabel('Probe frequency (Hz)')
                axes[meas].set_ylabel('Sparams (dB)')
                axes[meas].legend()
                axes[meas].set_title(meas + ' log mag')
        except:
            pass
    return data

# fit vna/pupmp power sweep with reflection circle fit
def fit_pow_sweep_SMC(data, display=False, kc=60e6, ki=1e6, numFitPoints=0, guessF=fs.guessF_maxReal):
    return fit_sweep_SMC(data, swept_var='powers', display=display, kc=kc, ki=ki,
                         numFitPoints=numFitPoints, guessF=guessF)

def plot_fit_pow_sweep_SMC(fitdata, meas_names, powers):
    plot_fit_sweep_SMC(fitdata, meas_names, powers, swept_var='powers')

def plot_fitGain_pow_sweep_SMC(results, gainfits, powfits=None, atten=0):
    xlabel = 'Power (dBm)'
    numPlots = 4 if powfits else 3
    fig, ax = plt.subplots(numPlots, sharex=True)
    xs = results['LIN']['powers']
    meas_names = results['LIN']['ys'].keys()
    for meas in meas_names:
        f0 = gainfits[meas]['f0']*1e-9
        g_log = gainfits[meas]['g_log']
        bw = gainfits[meas]['bw']*1e-6
        ax[0].plot(xs, f0, label=meas)
        ax[1].plot(xs, g_log, label=meas)
        ax[2].plot(xs, bw, label=meas)
        if powfits:
            ax[3].plot(xs, powfits[meas]['P_1dB'] + atten, label=meas)
    ax[0].set_ylabel('f0 (GHz)')
    ax[1].set_ylabel('Gain (dB)')
    ax[2].set_ylabel('BW (MHz)')
    if powfits:
        ax[3].set_ylabel('P_-1dB (dBm)')
    ax[-1].set_xlabel(xlabel)
    ax[0].legend(loc='best')
    
# fits slope of P_-1dB vs. g_log, cut allows you to exceed early gain points
def fit_P1dB_gain(g_log, p_1dB, cut=0):
    if len(g_log.shape) == 1:
        pn = np.polyfit(g_log[cut:], p_1dB[cut:], 1)
        pn = np.array([pn])
    elif len(g_log.shape) == 2:
        pn = np.zeros([g_log.shape[0], 2])
        for kk in np.arange(g_log.shape[0]):
            pn[kk, :] = np.polyfit(g_log[kk, cut:], p_1dB[kk, cut:], 1)
    else:
        raise ValueError("g_log size not suppported by fit_P1dB_gain")
    return pn

def fit_P1dB_gain_dict(gainfits, powfits, atten=0):
    meas_names = gainfits.keys()
    pfits = {}
    for meas in meas_names:
        pfits[meas] = fit_P1dB_gain(gainfits[meas]['g_log'], powfits[meas]['P_1dB'] + atten)
    return pfits

# plot fit of slop of P_-1dB vs g_log
def plot_fit_P1dB_gain(gainfits, powfits, pfits=None, fpump = [], cmap='hot', atten=0):
    meas_names = gainfits.keys()
    if not pfits:
        pfits = fit_P1dB_gain_dict(gainfits, powfits)
    if not fpump:
        fpump_str = ['']*pfits[meas_names[0]].shape[0]
    else:
        fpump_str = [' %.5f GHz'%(x) for x in np.array(fpump)*1e-9]
    fig, ax = plt.subplots()
    cmap_handle = plt.get_cmap('hot')
    cmap_arr = np.linspace(0, 0.9, pfits[meas_names[0]].shape[0])
    colors = [cmap_handle(x) for x in cmap_arr]
    for meas in meas_names:
        for kk in np.arange(pfits[meas].shape[0]):
            label = meas + fpump_str[kk]
            g_log = gainfits[meas]['g_log']
            ax.plot(g_log, powfits[meas]['P_1dB']+atten, color=colors[kk], marker='D', fillstyle='none', ls='None', label=label)
            pfit = np.poly1d(pfits[meas][kk])
            ax.plot(g_log, pfit(g_log)+atten, color=colors[kk], label='slope %.2f dBm/dB'%(pfits[meas][kk][0]))
    ax.set_xlabel('Gain (dB)')
    ax.set_ylabel('P_-1dB (dBm)')
    ax.legend(loc='best')
    return pfits

# VNA power sweep at each flux
def flux_sweep_VNApow_sweep_SMC(pnax, yoko, meas_names, currents, pows,
                                display=False, vna_fmt='POL', printlog=True,
                                yoko_slew=10e-6):
    figs = {}
    axes = {}
    # TODO - implement displaying data as it is acquired
    if not meas_names:
        return
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    if pnax.get_sweep_type() == 'SEGM':
        sweep_time = float(pnax.get_segment_sweep_time())
    else:
        sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 0.5) * len(meas_names) * len(currents) * len(pows)
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
    ys = {}
    data = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(currents), len(pows), 2, num_points))
    # sweep yoko, then sweep powers
    for i_current, current in enumerate(currents):
        wait_time = yoko.set_current_ramp(current, slew=yoko_slew)
        if wait_time:
            sleep(wait_time + 0.5) #0.5 for GBIP overhead
        for i_pow, power in enumerate(pows):
            pnax.set_power(power)
            data = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='LIN')
            for meas in meas_names:
                ys[meas][i_current, i_pow, :, :] = data['ys'][meas]
    pnax.set_power(np.min(pows))
    data['ys'] = ys
    data['powers'] = pows
    data['currents'] = currents
    return data
    
# for each VNA CW frequency, take a POW SMC sweep to see bifurcation
def take_cw_VNApow_sweep_SMC(pnax, meas_names, cw_freqs, pow_start=-80, pow_stop=-30,
                             pow_step=1.0, display=True, vna_fmt='POL', printlog=True,
                             sweep_updown=False):
    figs = {}
    axes = {}
    if not meas_names:
        raise ValueError('Need to provide meas_names for cw_VNApow_sweep_SMC')
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    numPoints = round(abs(pow_stop - pow_start) / pow_step + 1)
    freqLO = pnax.get_mixerLO_freq_fixed()
    set_up_P1dB_SMC(pnax, cw_freqs[0], freqLO, numPoints=numPoints, averages=averages,
                    pow_start=pow_start, pow_stop=pow_stop)
    sweep_time = float(pnax.get_sweep_time())
    tot_time = (sweep_time * averages + 1.0) * len(meas_names) * len(cw_freqs)
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
    ys = {}
    data = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(cw_freqs), 2, numPoints))
        if display:
            figs[meas] = plt.figure()
            axes[meas] = figs[meas].add_subplot(111)
    # sweep cw freqs
    for i_freq, freq in enumerate(cw_freqs):
        if sweep_updown and i_freq % 2:
            set_up_P1dB_SMC(pnax, freq, freqLO, numPoints=numPoints, averages=averages,
                            pow_start=pow_stop, pow_stop=pow_start)
        else:
            set_up_P1dB_SMC(pnax, freq, freqLO, numPoints=numPoints, averages=averages,
                            pow_start=pow_start, pow_stop=pow_stop)
        data = get_Sparams_SMC(pnax, meas_names, vna_fmt=vna_fmt, sweep_type='POW')
        invert = -1 if sweep_updown and i_freq % 2 else 1
        data['xs'] = data['xs'][::invert]
        for meas in meas_names:
            ys[meas][i_freq, :, :] = data['ys'][meas][:, ::invert]
            try:
                if display:
                    yc = ys[meas][i_freq, 0, :] + 1j*ys[meas][i_freq, 1, :]
                    (ylog, yphase) = fs.complex_to_PLOG(yc)
                    axes[meas].plot(data['xs'], ylog, label='%.6f GHz'%(freq*1e-9))
                    figs[meas].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
    data['ys'] = ys
    data['cw_freqs'] = cw_freqs
    if display:
        try:
            for meas in meas_names:
                axes[meas].set_xlabel('VNA power (dBm)')
                axes[meas].set_ylabel('Sparams logmag (dB)')
                axes[meas].legend()
                axes[meas].set_title(meas + ' log mag')
        except:
            pass
    return data
    
def load_fits_LIN(h5filename, dataDir, sweep_var=None):
    # dataDir: just day and time
    # sweep_var: 'currents' or 'powers' if part of a larger sweep, None if idividual trace
    fitdata = {}
    h5f = h5py.File(h5filename)
    try:
        fitdata = load_fits_LIN_helper(h5f, dataDir, sweep_var)
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    return fitdata

def load_fits_LIN_helper(h5f, dataDir, sweep_var=None):
    fitdata = {}
    g = h5f[dataDir]
    if 'fits' not in g.keys():
        raise ValueError('Fits not available to be loaded in ' + dataDir)
    if sweep_var:
        fitdata[sweep_var] = np.array(g['LIN/' + sweep_var])
    g = g['fits']
    meas_names = g.keys()
    fitdata['fits'] = {}
    for meas in meas_names:
        fitdata['fits'][meas] = fs.load_Qfit_helper(g[meas])
    return fitdata
    
def fit_freqshift(linpows, freqshifts, porder=1):
    pcoeff = np.polyfit(linpows, freqshifts, porder)
    pfit = np.poly1d(pcoeff)
    results = {'linpows':linpows,
               'freqshifts': freqshifts,
               'pfit': pfit,
               'slope': pcoeff[-2],
                'pcoeff': pcoeff}
    return results

def fit_freqshift_sweep(fitdata, sweep_var='powers', atten=0, f_pump=0, porder=1):
    results = {}
    xdata = fitdata[sweep_var]
    detuning = 0
    if sweep_var == 'powers_swept':
        xdata = xdata + atten
        xdata = 1e-3*np.power(10, xdata / 10.0) #power in W
    meas_names = fitdata['fits'].keys()
    for meas in meas_names:
        mxdata = xdata
        f0 = fitdata['fits'][meas]['f0'][0]
        kc = fitdata['fits'][meas]['kc'][0]
        ki = fitdata['fits'][meas]['ki'][0]
        if sweep_var == 'powers_swept' and atten:
            if f_pump != 0:
                detuning = (f_pump - fitdata['fits'][meas]['f0']) #Hz, array of detunings
            mxdata = pow_to_nbar(mxdata, f0, kc, kc+ki, detuning)
        freqshifts = (fitdata['fits'][meas]['f0'] - f0)*1e-6 #MHz
        results[meas] = fit_freqshift(mxdata, freqshifts, porder)
        results[meas]['detuning'] = detuning
    return results
    
def plot_freqshift_helper(ax, fitshifts, label='trace', color='r', display=True):
    xdata = fitshifts['linpows']
    ax.plot(xdata, fitshifts['freqshifts'], color=color, marker='D', fillstyle='none', ls='none')
    ax.plot(xdata, fitshifts['pfit'](xdata), color=color, label = label + ' slope %.4f MHz/photon'%(fitshifts['slope']))
    print(label + ' slope %.4f MHz/photon'%(fitshifts['slope']))
    return fitshifts['slope']

# TODO - return slope for all measnames instead of just the last slope
def plot_freqshift_sweep(h5filename, dataDir, sweep_var='powers_swept', atten=0, f_pump=0, porder=1):
    fitdata = load_fits_LIN(h5filename, dataDir, sweep_var)
    results = fit_freqshift_sweep(fitdata, sweep_var, atten, f_pump=f_pump, porder=porder)
    meas_names = results.keys()
    fig, ax = plt.subplots()
    for meas in meas_names:
        slope = plot_freqshift_helper(ax, results[meas], label=meas)
    ax.set_xlabel('nbar')
    ax.set_ylabel('Frequency Shift (MHz)')
    ax.legend()
    return results
    
def pow_to_nbar(power, f0, kc, ktot, delta=0):
    numer = kc / (np.square(delta) + np.square(ktot / 2)) / (2*np.pi)
    return numer * power / (Planck * f0)
   
# load Sparams sweeps from h5file into same dictionary format, if no dataDir provided: use most recent
def load_h5_Sparams(h5filename, dataDir=None):
    results = {}
    h5f = h5py.File(h5filename)
    try:
        if not dataDir:
            cur_group = h5f.keys()[-1]
            cur_time = h5f[cur_group].keys()[-1]
            dataDir = '//' + cur_group + '//' + cur_time + '//'
        sweep_types = h5f[dataDir].keys()
        if 'fits' in sweep_types:
            # TODO - load fits into dictionary
            sweep_types.remove('fits')
        if 'spec' in sweep_types:
            #TODO - load spec data into dictionary from taking gain curve
            sweep_types.remove('spec')
        for sweep_type in sweep_types:
            curDir = dataDir + '/' + sweep_type
            results[sweep_type] = load_h5_Sparams_helper(h5f, curDir)
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    return results

#TODO - need to implement loading for sweeps, also test
def load_h5_Sparams_helper(h5f, dataDir):
    data ={}
    g = h5f[dataDir]
    keys = g.keys()
    sweep_names = ['frequencies', 'powers', 'currents', 'powers_swept']
    xs_name = 'frequencies' if 'frequencies' in keys else 'powers'
    for sweep_name in sweep_names:
        if sweep_name in keys:
            data_name = 'xs' if sweep_name in ['frequencies', 'powers'] else sweep_name
            data[data_name] = np.array(g[sweep_name])
            keys.remove(sweep_name)
#    xs_name = 'frequencies' if 'frequencies' in keys else 'powers'
#    data['xs'] = np.array(g[xs_name])
#    keys.remove(xs_name)
    meas_names = keys
#    print(xs_name)
#    print(dataDir)
#    print(meas_names)
    attrs_arr = g[xs_name].attrs.keys()
    for attr in attrs_arr:
        data[attr] = g[xs_name].attrs[attr]
    fmt = data['format'] if 'format' in data.keys() else 'POL'
    data['ys'] = {}
    for meas in meas_names:
        y0 = np.array(g[meas][fs.DATA_LABEL[fmt][0]])
        y1 = np.array(g[meas][fs.DATA_LABEL[fmt][1]])
        if len(y0.shape) == 1:
            ys = np.zeros([2, y0.size])
            ys[0, :] = y0
            ys[1, :] = y1
        else:
            ys = np.zeros((y0.shape[0], 2, y0.shape[1]))
            ys[:, 0, :] = y0
            ys[:, 1, :] = y1
        data['ys'][meas] = ys
    data['meas_names'] = meas_names
    return data

# take ag pump power sweep and record sweptIMD POW sweep data for each ag power
def pump_pow_sweep_sweptIMD(pnax, ag, meas_names, agPows, display=True,
                            vna_fmt='MLOG', printlog=True, num_im=2, fpump=None):
    figs = {}
    axes = {}
    assert meas_names, 'must provide meass_names for sweptIMD measurement'
    pnax.set_meas_select(meas_names[0])
    averages = pnax.get_average_factor()
    num_points = pnax.get_points()
    sweep_time = float(pnax.get_sweep_time())
    if_main = pnax.get_imd_IFBW_main()
    if_im = pnax.get_imd_IFBW_im()
    time_mult = 1 + num_im * if_main / if_im
    tot_time = (sweep_time * averages + 0.5) * time_mult * len(agPows)
    if printlog:
        print('Total duration of this experiment will be %.2f minutes.'%(tot_time/60.0))
    ys = {}
    data = {}
    for meas in meas_names:
        ys[meas] = np.zeros((len(agPows), num_points))
        if display:
            figs[meas] = plt.figure()
            axes[meas] = figs[meas].add_subplot(111)
    ag_freq = fpump if fpump else ag.get_frequency()
    ag.set_frequency(ag_freq)
    ag.set_rf_on(True)
    # sweep pump power
    for i_pow, power in enumerate(agPows):
        ag.set_power(power)
        sleep(0.1) # allow generator to turn on and stabilize
        data = take_Sparams_sweptIMD(pnax, meas_names, opc_arg=True, num_im=num_im)
        # add most recent sweep data to ys and display
        for meas in meas_names:
            ys[meas][i_pow, :] = data['ys'][meas]
            try:
                if display:
                    ylog = ys[meas][i_pow,:]
                    axes[meas].plot(data['xs'], ylog, label='%.3f dBm'%(power))
                    figs[meas].canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
    ag.set_rf_on(False)
    # put full pow sweep data in ys
    data['ys'] = ys
    data['powers'] = agPows
    data['fpump'] = ag_freq
    if display:
        try:
            for meas in meas_names:
                axes[meas].set_xlabel('Probe Power (dBm)')
                axes[meas].set_ylabel('Sparams Logmag (dB)')
                axes[meas].legend()
                axes[meas].set_title(meas + ' log mag, fpump = %.5f'%(data['fpump']*1e-9))
        except:
            pass
    return data
    
# plots a pump power sweep
def plot_pump_pow_sweep_sweptIMD(data, attenInp=0, save=None, cmap = 'hot'):
    assert data['format'] == 'MLOG', 'Data format ' + data['format'] + ' is not yet supported for SweptIMD plotting'
    sweep_type = data['sweep_type']
    meas_names = data['meas_names']
    agPows = data['powers']
    xs = data['xs']
    xlabel = 'Tone Power (dB)'
    if sweep_type == 'POW':
        xs = xs + attenInp
    elif sweep_type == 'FCEN':
        xs = xs*1e-9 #freqs to GHz
        xlabel = 'Carrier Frequency (GHz)'
    elif sweep_type == 'DFR':
        xs = xs*1e-6 #freqs to MHz
        xlabel = 'Delta Frequency (MHz)'
    elif sweep_type == 'CW':
        xlabel = 'Point Number'
    else:
        raise ValueError('sweep_type: ' + sweep_type + ' not supported by plot_Sparams_sweptIMD')
    extent = [xs[0], xs[-1], agPows[0], agPows[-1]]
    figs = {}
    axes = {}
    for meas in meas_names:
        y0 = data['ys'][meas]
        (figs[meas], axes[meas]) = plt.subplots(1,1)
        p0 = (axes[meas]).pcolormesh(xs,agPows, y0, cmap=cmap, extent=extent)
        cb0 = figs[meas].colorbar(p0, ax=axes[meas])
        cb0.set_label('Power (dB(m))', rotation=270)
        title = meas + ', f_pump = %.5f GHz'%(data['fpump']*1e-9)
        axes[meas].set_title(title)
        axes[meas].set_ylabel('Pump Power (dBm)')
        axes[meas].set_xlabel(xlabel)
        if save:
            figs[meas].savefig(save + '/' + title)