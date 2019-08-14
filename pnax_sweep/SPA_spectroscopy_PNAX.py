"""
FOR THIS TO WORK :
    Need to enable the sweep option for the generators  in instrument server 
    plugin in qrlab (already done on the JPC computer)
    'C:\\code\\qrlab\\instrumentserver\\instrument_plugins'
        Agilent_E8257D
            kwargs['add_modulation'] = True
        Agilent_N5183A
            kwargs['add_modulation'] = True
    Then, you need to restart the instrument server to recreate these objects. 

    PNAX & Gen configuration:
        PNAX "aux trig 1 out" to Gen Trigg in.

    Set attenuators on the generator on hold, make sure this allows to sweep power 
    in the desired range without errors.

    Before running the script center the PNAX at the desired mode 
    whose reflection phase response will be monitored.

    If the kernel dies in the middle of the sweep you need to 
        * Update the curr_start variable 
        * Restart the kerel and console
        * Restart the instrument server
        * Run create_instruments.py 
        * Run this script again

@author: Vladimir Sivak
"""

import numpy as np
import logging

from time import sleep, time
from mclient import instruments
from scripts.flux_sweep import FluxSweep as fs
from scripts.flux_sweep import FluxSweep_pnax as fsp
import matplotlib.pyplot as plt
    
#from SPA_characterization_PNAX import set_center_span


# in the generator power sweep it goes too the attenuation hold mode, meaning that
# it doesn't change the attenuators during the sweep. Therefore, need to make sure that 
# you have the correct attenuator that will work for the whole range of swept powers

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


def PNAXspectroscopy(pnax, ag, center_freq, if_bandwidth, num_points, 
                     average_factor, vna_power, start_freq, stop_freq, 
                     start_power, stop_power, vna_fmt='Phase',report_time=True):

    pnax.set_mixerINP_freq_mode('FIXED')
    pnax.set_mixerOUTP_freq_mode('FIXED')
    pnax.set_mixerINP_freq_fixed(center_freq)
    pnax.mixer_calc('OUTP')
    pnax.set_sweep_type('CW')
    pnax.mixer_apply_settings()
    pnax.set_format(vna_fmt)
    pnax.set_if_bandwidth(if_bandwidth)
    pnax.set_points(num_points)
    pnax.set_average_factor(average_factor)
    pnax.set_power(vna_power)
    pnax.set_aux_trigger_1_out(True)
    pnax.set_aux_trigger_1_out_interval('point')
    pnax.set_averaging_state(False)
    pnax.set_averaging_mode('point')

    if stop_freq == None:
        ag.set_sweep_frequency_mode('CW')
        ag.set_frequency(start_freq)
    else:
        ag.set_sweep_frequency_mode('SWE')
        ag.set_sweep_start_frequency(start_freq)
        ag.set_sweep_stop_frequency(stop_freq)

    if stop_power == None:
        ag.set_sweep_amplitude_mode('FIX')
        ag.set_power(start_power)
    else:
        ag.set_sweep_amplitude_mode('SWE')
        ag.set_sweep_start_amplitude(start_power)
        ag.set_sweep_stop_amplitude(stop_power)
    ag.set_pulse_on(False)
    ag.set_sweep_n_points(num_points)
    ag.set_sweep_trigger('IMM')
    ag.set_sweep_point_trigger('EXT')
    ag.set_rf_on(True)

    pnax.set_trigger_source('MAN')
    pnax.set_averaging_state(True)
#    ag.reset_sweep()
    sleep(2) #5
    pnax.trigger()
    
    sweepTime = pnax.get_sweep_time()
    if report_time:    
        print "This will take %0.1f seconds to finish"%(sweepTime*1.1) #1.1 for two-tone #1.3 for characterization
    start_time = time()
    while True:
        time_elapsed = time()-start_time
        if time_elapsed>(sweepTime*1.1):
            break
        if report_time:        
            print "\r" + "Time Elapsed is %.F seconds"%time_elapsed,
        sleep(1)
    if report_time:
        print '\n'
    
    data = pnax.do_get_data(fmt ='POL')
    ag.set_rf_on(False)

    ag.set_sweep_amplitude_mode('FIX')
    ag.set_sweep_frequency_mode('CW')

    return data[0,:]+1j*data[1,:]




if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    path = r'Z:\Data\JPC_2019\05-15-2019 Cooldown\\'
    h5filename = path + 'JAMPA10.hdf5'

    #### instruments
    pnax_smc = instruments['pnax_smc']
    ag = instruments['ag2']
    yoko = instruments['yoko']

    ### measurements
    meas_names = ['"CH1_S11_1"']

    ### Yoko currents
    curr_start = 1.30e-3
    curr_stop = -1.00e-3
    curr_step = 0.02e-3
    currents = np.linspace(curr_start, curr_stop, round(abs((curr_stop - curr_start) / curr_step) + 1))
    
    ### pnax parameters
    if_bandwidth = 3e3 
    average_factor = 50 # for trigerring the generator
    averages_vna = 5
    points = 1601
    span_linear = 800e6
    vna_power = -60
    
    ### Fit parameters
    kc_guess = 100e6
    ki_guess = 1e6
    
    ### Generator sweep parameters
    start_freq = 0.1e9
    stop_freq = 36e9
    num_points = 901 
    ag_freqs = np.linspace(start_freq,stop_freq,num_points)
    np.savetxt(path+'ag_freqs.txt',ag_freqs)
    start_pow = -12
    stop_pow = 7
    ag_pows = np.linspace(start_pow,stop_pow,num_points)        
    np.savetxt(path+'ag_pows.txt',ag_pows)        

    ### Two-tone spectroscopy sweep
    res_freqs = np.zeros(len(currents))
    data = np.zeros(num_points)+1j*np.zeros(num_points)    
    for j, current in enumerate(currents):
        # change the current
        print('Current %f' %(current*1e3) )
        wait_time = yoko.set_current_ramp(current, slew=10e-6)
        if wait_time:
            sleep(wait_time + 0.5) #0.5 for GBIP overhead    
        # fit the resonance
        res_data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
#        dataDir = fsp.save_Sparams(h5filename, res_data, meas_class='SMC')
        fitdata = fsp.fit_reflection_SMC(res_data, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_maxReal)
#        fsp.save_fit_reflection(fitdata, h5filename, dataDir)
        freq_0 = fitdata[meas_names[0]][0][0]
        res_freqs[j] = freq_0
        plt.close('all')
        # sweep the generator (drive tone)
        data = PNAXspectroscopy(pnax_smc, ag, freq_0, if_bandwidth, num_points, average_factor, vna_power, start_freq, stop_freq, start_pow, stop_pow)
        real = np.real(data)
        imag = np.imag(data)
        # save stuff
        names = ['real','imag','res_freq','current']
        for index, array in enumerate((real,imag,[freq_0],[current])):
            f = open(path+names[index]+'.txt','a')
            for element in array:
                f.write(str(element)+' ')
            f.write('\n')
            f.close()
        # re-center PNAX and exit
        set_center_span(pnax_smc, freq_0, span_linear, points=points, averages=averages_vna, power=vna_power) 
        
        