# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:59:26 2018

@author: Vladimir Sivak
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import h5py
from time import sleep, strftime
from mclient import instruments
from scripts.flux_sweep import FluxSweep as fs
from scripts.flux_sweep import FluxSweep_pnax as fsp
from SPA_spectroscopy_PNAX import PNAXspectroscopy 
from SPA_characterization_PNAX import tune_up_gain, set_center_span, figure_out_gains, get_response_at_fixed_freq

# --------------------------------------------------
# -------------------------------------------------- 
# ---------- gain stability measurement ------------
# -------------------------------------------------- 
# -------------------------------------------------- 



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)    
    path = r'Z:\Data\JPC_2019\04-26-2019 Cooldown\\'
    h5filename = path + 'JAMPA10.hdf5'
    meas_names = ['"CH1_S11_1"']
    meas = meas_names[0]

    #### instruments
    pnax_smc = instruments['pnax_smc']
    ag_gain = instruments['ag2']
    yoko = instruments['yoko']
    fridge = instruments['fridge']

    # Sweep parameters
    pow_start = -20.0
    pow_stop = 20.0
    averages = 50
    pow_step = 0.01
    count_max = 780
    time_step = 60

    
    # Guess parameters for the resonance fit
    kc_guess = 70e6
    ki_guess = 1e6

    # PNAX parameters
    span_gain = 0.2e9
    span_linear = 1.0e9
    vna_pow = -60.0
    averages_vna = 10
    num_points = 1601


    # fit the resonance
    data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
    dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
    fitdata = fsp.fit_reflection_SMC(data, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_maxReal)
    fsp.save_fit_reflection(fitdata, h5filename, dataDir)
    freqLO = data['freqLO']
    freq_pump = 2.0*fitdata[meas][0][0]
    print('Pump frequency %f GHz' %(freq_pump*1e-9))
    ag_gain.set_frequency(freq_pump)
    ag_gain.set_rf_on(False)       
    guess_CW = freq_pump/2.0 + 100e3
    
    
    #  Automatically tune up an original 20 dB gain point
#    Gain, logmag_memory, ag_pow, flag = tune_up_gain(pnax_smc, pow_start, pow_stop, guess_CW, averages, meas_names)

    want_gains = [20]
    possible_gains, ag_powers, gains = figure_out_gains(pnax_smc, ag_gain, guess_CW, pow_start, pow_stop, 
                                               pow_step, averages, vna_pow, freq_pump, 
                                               3e3, meas_names, want_gains)

    print('Possible gains:')
    print(possible_gains)
    print('At generatore powers:')
    print(ag_powers)

    if want_gains[0] in possible_gains:
#    if flag:

        ag_pow = ag_powers[0]
#
        # Loop to measure gain with old pump parameters every once in awhile, keep pump on
        start_date = strftime('%y%m%d')
        start_time = strftime('%H%M%S')

        ag_gain.set_rf_on(False)
        logmag_memory = get_response_at_fixed_freq(pnax_smc, guess_CW, meas_names, averages=averages, logmag=True)
        count = 0

        
        ag_gain.set_power(ag_pow)
        ag_gain.set_rf_on(True)
        while count < count_max:
            # take gain          
            Gain = get_response_at_fixed_freq(pnax_smc, guess_CW, meas_names, averages=averages, logmag=True) - logmag_memory
            
#            # measure the fridge temperature to correlate with the Gain jumps
#            fridge = instruments.reload('fridge')
#            T = fridge.get_ch2_temperature()

            
            f = open(path+'gain'+'.txt','a')
            f.write(str(float(Gain))+' ')
            f.close()
#            f = open(path+'temperature'+'.txt','a')
#            f.write(str(float(T))+' ')
#            f.close()

            
            count += 1
            sleep(time_step)
        array_gains = np.loadtxt(path+'gain.txt')
#        array_temperatures = np.loadtxt(path+'temperature.txt')
        
        # save data to hdf5 file
        h5f = h5py.File(h5filename)
        try:
            cur_group = strftime('%y%m%d')
            _iteration_start = strftime('%H%M%S')
            dataDir = _iteration_start+'//'
            g = h5f.require_group(cur_group)
            dg = g.create_dataset(dataDir + 'gains', data=array_gains)
#            dt = g.create_dataset(dataDir + 'temperatures', data=array_temperatures)
            dg.attrs.create('pump_power', ag_pow)
            dg.attrs.create('pump_freq', freq_pump)
            dg.attrs.create('CW_freq', guess_CW)
            dg.attrs.create('averages', averages)
            dg.attrs.create('logmag_memory', logmag_memory)
            dg.attrs.create('time_step', time_step)
            dg.attrs.create('start_time', start_time)
            dg.attrs.create('start_date', start_date)
            dg.attrs.create('stop_time', _iteration_start)
            dg.attrs.create('stop_date', cur_group)
        finally:
            h5f.close()      
    ag_gain.set_rf_on(False) 
    set_center_span(pnax_smc, freq_pump/2.0, span_linear, points=num_points, 
                    averages=averages_vna, power=vna_pow)


