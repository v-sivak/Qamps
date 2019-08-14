# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:09:51 2019

@author: JPC-Acquisition

This code does the voltage sweep on the SNT and measures the noise power emmited by the SNT.
Noise measurement can be done either using the PNAX Noise Figure class or using the Spectrum 
Analyzer. In the latter case data taking is more caveman-style, because I haven't figured out
how to trigger the Spectrum Analyzer via GPIB, it doesn't seem to have the BUS trigger option.
Therefore, before taking data you need to measure the time of one complete averaging cycle
and store that in avg_time for dalaying the data taking.

"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.colorbar as plt_colorbar
import h5py
from time import sleep, strftime
from mclient import instruments
from scripts.flux_sweep import FluxSweep as fs
from scripts.flux_sweep import FluxSweep_pnax as fsp
    

def save_Sparams_spectrum_analyzer(g, _iteration_start, dataDir, data):
    fmt = 'MLOG'
    dg = g.create_dataset(dataDir + 'frequencies', data=data['xs'])
    dg.attrs.create('format', fmt)
    dg.attrs.create('averages', data['averages'])
    dg.attrs.create('rbw', data['rbw'])
    dg.attrs.create('run_time', _iteration_start)
    dgm = g.create_dataset(dataDir + fs.DATA_LABEL[fmt], data=data['ys'])
    if 'V_src' in data.keys():
        g.create_dataset(dataDir + 'V_src', data=data['V_src'])
        g.create_dataset(dataDir + 'V_snt', data=data['V_snt'])
    dgm.attrs.create('run_time', _iteration_start)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    h5filename = r'Z:\Data\JPC_2019\06-17-2019 Cooldown\SNT.hdf5'

    # choose how to measure: PNAX or Spectrum Analyzer
    inst = 'spec' # 'pnax'
    avg_time = 1000
        
    #instruments to use
    pnax_NF = instruments['pnax_NF']
    volt_source = instruments['yoko_vsr']  #use Yokogawa GS200 in voltage mode
    volt_meter = instruments['keith']  #use a Keithley 2400 as voltmeter
    spec = instruments['spec']

    noise_names = ['"CH6_SYSNPDI_14"'] #for PNAX

    vsr = np.linspace(-0.3, 0.3, 401)
    ramp_rate = 10e-3
    center_freq = 8.0e9
    span = 8.0e9
    step = 10e6
    averages = 10000
    points = int(span/step + 1)       
    
    # set up the Spectrum Analyzer of the PNAX Noise Figure class
    if inst == 'spec':
        spec.set_center_freq(center_freq)
        spec.set_span(span)
        spec.set_average_factor(averages)
        spec.set_points(points)
    elif inst == 'pnax':
        pnax_NF.set_center_freq(center_freq)
        pnax_NF.set_span(span)
        pnax_NF.set_average_factor(averages)
        pnax_NF.set_points(points)        
    
    # set up the voltmeter and source
    volt_source.set_source_range(np.max(vsr))
    volt_meter.set_status(1)
    volt_meter.set_source_current(0.0)    
    
    noise_arr = np.zeros((len(vsr),points))
    v_snt = np.zeros(vsr.shape)

    data = {}    
    for i, V in enumerate(vsr):
        wait_time = np.abs(V-volt_source.get_source_level())/ramp_rate + 0.5   
        volt_source.set_voltage_ramp(V, ramp_rate)
        sleep(wait_time)
        v_snt[i] = volt_meter.get_output_voltage()
        sleep(1)
        if inst == 'pnax':
            data = fsp.take_Sparams_NF(pnax_NF, noise_names)
            noise_arr[i] = data['ys'][noise_names[0]]
        elif inst == 'spec':
            spec.trig_restart()
            sleep(avg_time)
            noise_arr[i] = spec.do_get_yaxes()              
    
    if inst == 'pnax':
        data['ys'][noise_names[0]] = noise_arr
    elif inst == 'spec':
        data['ys'] = noise_arr
        data['xs'] = spec.do_get_xaxis() 
        data['averages'] = spec.get_average_factor()
        data['rbw'] = spec.get_resolution_bandwidth()
    data['V_src'] = vsr
    data['V_snt'] = v_snt


    # save data to hdf5 file
    h5f = h5py.File(h5filename)
    try:
        cur_group = strftime('%y%m%d')
        iteration_start = strftime('%H%M%S')
        g = h5f.require_group(cur_group)
        saveDir = iteration_start + '//noise//'
        if inst == 'pnax':
            fsp.save_Sparams_NF_helper(g, iteration_start, saveDir, data)
        elif inst == 'spec':
            save_Sparams_spectrum_analyzer(g, iteration_start, saveDir, data)
    finally:
        h5f.close()
    print('dataDir: '+cur_group+'/'+iteration_start)
