"""
Created on Tue May 7 12:44:20 2019

FOR THIS TO WORK :
    Need to enable the sweep option for the generators  in instrument server plugin in qrlab.
    'C:\\code\\qrlab\\instrumentserver\\instrument_plugins'
        Agilent_E8257D
            kwargs['add_modulation'] = True
        Agilent_N5183A
            kwargs['add_modulation'] = True
    Then, you need to restart the instrument server to recreate these objects. 

    PNAX & Gen configuration:
        PNAX "aux trig 1 out" to Gen Trigg In.

    Set attenuators on the generator on hold, make sure this allows to sweep power 
    in the desired range without errors
    
@author: Vladimir Sivak
"""

import numpy as np
import logging

from time import sleep, time
from mclient import instruments
from scripts.flux_sweep import FluxSweep as fs
from scripts.flux_sweep import FluxSweep_pnax as fsp
import matplotlib.pyplot as plt
import h5py
import sys

from SPA_spectroscopy_PNAX import PNAXspectroscopy     
from SPA_characterization_PNAX import get_response_at_fixed_freq, set_center_span



# TODO: rewrite this with some sort of piece-wise resonance fitting
def locate_resonances(S11):
        S11 = np.vectorize(complex)(S11[0,:],S11[1,:])
        phase = 180.0*np.angle(S11)/np.pi
        # roughly locate the resonances by looking at where phase is close to zero. 
        # Need to cancel the electrical delay and choose a nice phase offset manually
        flag = True
        res_ind = []
        for i in range(len(S11)):
            if np.abs(phase[i])<10 and flag:
                flag=False
                res_ind = res_ind + [i+1]  # i+1 instead of i just because if I choose i, the absolute difference will be close to 10 instead of 0. 
            elif np.abs(phase[i])>60 and not flag:
                flag = True
        return res_ind

def load_array_modes(h5filename):    
    hdf5 = h5py.File(h5filename)
    try:
        if 'array_modes' in hdf5.keys():
            array = {}
            for x in hdf5['array_modes'].keys():
                currents = np.asarray(hdf5['array_modes'][x].get('currents'))
                freqs = np.asarray(hdf5['array_modes'][x].get('res_freqs'))
                array[x] = (currents, freqs)
            flag = True
        else:
            flag = False
    finally:
        hdf5.close()
    return array if flag else None 


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    path = r'Z:\Data\JPC_2019\05-15-2019 Cooldown\\'
    h5filename = path + 'JAMPA10.hdf5'

    #### instruments
    pnax_smc = instruments['pnax_smc']
    ag = instruments['ag2']
    yoko = instruments['yoko']
    pnax_NF = instruments['pnax_NF']

    #### define measurements
    meas_names = ['"CH1_S11_1"']
    noise_names = ['"CH6_SYSNPDI_14"']
    
    ### PNAX parameters
    if_bandwidth = 3e3
    average_factor = 15  
    vna_power = -60
    num_points = 1601   
    span_gain = 500e6
    span_linear = 1.2e9
    averages_vna = 10
    averages_nvr = 10

    ### Signal power sweep parameters
    pow_start_P1dB = -70.0
    pow_stop_P1dB = -30.0
    num_points_pow = 201
    averages_pow = 100
    
    #### Array of frequencies to try for gain
    freq_min = 7.1e9
    freq_max = 12.0e9
    freq_step = 100e6
    want_freqs = np.arange(freq_min,freq_max,freq_step)
#    want_freqs = np.array([6.4, 6.5, 7.5, 7.6, 7.7, 7.8, 7.9, 8.9, 9.0, 9.1, 9.2, 9.3, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.8, 10.9, 11.2, 11.3, 11.4, 11.5, 11.8, 11.9, 12.0])*1e9
    want_freqs = np.array([7.4e9, 8.9e9, 9.1e9, 10.7e9, 10.8e9, 10.2e9, 10.3e9, 11.8e9, 4.3e9, 5.2e9, 5.4e9])

    #### generator allowed frequencies
    ag_max_freq = 18e9 #20e9
    ag_min_freq = 12e9 #12e9
    ag_span = 600e6
    ag_spacing = 5e6
    
    #### generator powers to sweep
    start_pow = -20.0
    stop_pow = 20.0
    step_pow = 0.02
    ag_pows = np.arange(start_pow, stop_pow, step_pow)

    #### desired gain and nvr
    GAIN = 20.0
    nvr_min = -0.8
    nvr_max = 15.0    
    
    #### fitting guess
    kc_guess = 100e6
    ki_guess = 1e6

    #### Yoko parameters (flux sweep)
    curr_start = 4.50e-3
    curr_stop = 2.18e-3
    curr_step = 10e-6

    #### Yoko fine-tuning
    freq_precision = 10e6
    curr_fine_step = 0.01e-3
    yoko_count_max = 50
    Slope = +1

    #### Load the "array" dictionary of modes from the hdf5 file. Alternatively, use flux sweep
    array = None #load_array_modes(h5filename)
    if array is None:
        print('Couldn\'t find "array_modes" in the hdf5 file.\n\
        Will use a less efficient guess for the yoko current and resonance frequeencies.')
        flux_sweep_dataDir = '190520//132906/'
        results = fsp.load_h5_Sparams(h5filename, flux_sweep_dataDir)
        data = results['LIN']
        currents = np.array(data['currents'])

    #### Loop over desired operating frequencies and try to tune up gain at each
    for center_freq in want_freqs:
        print('--------------------------------------------------')
        print('Tune up gain at %.2f GHz' %(center_freq*1e-9))
        # For each center_frequency figure out the yoko current at which some
        # resonance is approximately at that frequency.
        if array is not None:
            possible_currents = []
            for n in array:
                if center_freq<max(array[n][1]) and center_freq>min(array[n][1]):
                    ind = np.argmin(np.abs(array[n][1]-center_freq))
                    possible_currents += [array[n][0][ind]]
            if len(possible_currents) == 0:
                print('No yoko currents found.')
        else:
            ind_freq = np.argmin(np.abs(data['xs']-center_freq))
            S11 = np.vectorize(complex)(np.transpose(data['ys'][meas_names[0]])[ind_freq][0],
                                 np.transpose(data['ys'][meas_names[0]])[ind_freq][1])
            real = np.real(S11/np.abs(S11))
            ind_curr = np.argmax(real)  
            possible_currents = [currents[ind_curr]]

        # Fine-tune the current to bring the mode resonance to center_freq
        for current in possible_currents:
            set_center_span(pnax_smc, center_freq, span_linear, points=num_points, averages=averages_vna, power=vna_power)        
            current_found_flag = False
            count = 0
            while not current_found_flag:
                wait_time = yoko.set_current_ramp(current, slew=100e-6)
                if wait_time: sleep(wait_time + 0.1)
                count += 1
                data_lin = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
                fitdata = fsp.fit_reflection_SMC(data_lin, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_middle,display=False)
                f_res = fitdata[meas_names[0]][0][0]
                print('Current %.2f mA, resonance at %.2f GHz' %(current*1e3,f_res*1e-9))
                if f_res < center_freq and np.abs(center_freq - f_res)>freq_precision and count<yoko_count_max:
                    current += curr_fine_step*Slope
                elif f_res > center_freq and np.abs(center_freq - f_res)>freq_precision and count<yoko_count_max:
                    current -= curr_fine_step*Slope
                else:
                    current_found_flag = True
                    if count == yoko_count_max:
                        print('Yoko current search didn\'t converge after %d iteratios.' %yoko_count_max)
            
            # Figure out approximately where all the other resonances are and sweep the pump
            # near the sum frequency for non-degenerate/degenerate gain.   
            if array is not None:
                resonances = []     
                for n in array:
                    if current>min(array[n][0]) and current<max(array[n][0]):
                        ind = np.argmin(array[n][0]-current)
                        resonances += [array[n][1][ind]]
                resonances = np.array(resonances)
            else:
                ind_curr = np.argmin(np.abs(current-currents))
                ind_res = locate_resonances(data['ys'][meas_names[0]][ind_curr])
                resonances = data['xs'][[min(s,len(data['xs'])-1) for s in ind_res]]
                
            if len(resonances) == 0:          
                ind_curr = np.argmin(np.abs(current-currents))
                ind_res = locate_resonances(data['ys'][meas_names[0]][ind_curr])
                resonances = data['xs'][[min(s,len(data['xs'])-1) for s in ind_res]]
            
            print('Modes guess:')
            print(resonances*1e-9)
#            for i, res in enumerate(resonances):
#                set_center_span(pnax_smc, res, 1.5e9, points=num_points, averages=averages_vna, power=vna_power)   
#                data_lin = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
#                fitdata = fsp.fit_reflection_SMC(data_lin, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_middle,display=False)
#                resonances[i] = fitdata[meas_names[0]][0][0]
#            print('Modes updated:')
#            print(resonances*1e-9)
            set_center_span(pnax_smc, center_freq, span_linear, points=num_points, averages=averages_vna, power=vna_power)
                  
            # Set up the Noise Figure class
            pnax_NF.set_center_freq(center_freq)
            pnax_NF.set_span(span_gain)
            pnax_NF.set_average_factor(averages_nvr)
            
            # Do the pump power and frequency sweep and monitor the gain at the desired 
            # frequency. Sweep small regions which correspond to the non-degenerate 
            # gain between different pairs of modes. When the 20 dB gain working point
            # is found, adjust the power for 20 dB, take the data and break the loop. 
            logmag_memory = get_response_at_fixed_freq(pnax_smc, center_freq, meas_names, averages=average_factor, logmag=True)
            gain_flag = False # shows if the working point is found
            for res in resonances:
                MAX = 0.0
                if res + center_freq > ag_min_freq and res + center_freq < ag_max_freq:
                    print('Try pumping around %.2f + %.2f = %.2f GHz' %(center_freq*1e-9,res*1e-9,center_freq*1e-9+res*1e-9))

#                    pump_detunings = np.arange(-ag_span/2.0, ag_span/2.0, ag_spacing)
#                    ag_freqs = center_freq + res + pump_detunings[np.argsort(np.abs(pump_detunings))]
                    
                    left_detunings = np.arange(0, -ag_span/2.0, ag_spacing)
                    right_detunings = np.arange(ag_spacing, ag_span/2.0, ag_spacing)     
                    i_left, max_gain_left, LEFT = 0, GAIN, True
                    i_right, max_gain_right, RIGHT = 0, GAIN, False
                    while i_left<len(left_detunings) or i_right<len(right_detunings):
                    
#                    for ag_freq in ag_freqs:
                        if i_left >= len(left_detunings) and LEFT:
                            LEFT, RIGHT = False, True
                            i_right += 3 if max_gain_right < GAIN/4.0 else 1
                            if i_right >= len(right_detunings):
                                break
                        elif i_right >= len(right_detunings) and RIGHT:
                            LEFT, RIGHT = True, False
                            i_left += 3 if max_gain_left < GAIN/4.0 else 1
                            if i_left >= len(left_detunings):
                                break
                            
                        ag_freq = center_freq + res + left_detunings[i_left] if LEFT else center_freq + res + right_detunings[i_right]         
                        
                        
                        pump_sweep = PNAXspectroscopy(pnax_smc, ag, center_freq, if_bandwidth, (stop_pow-start_pow)/step_pow, 
                                                      average_factor, vna_power, ag_freq, None, start_pow, stop_pow, vna_fmt='log mag',
                                                      report_time=False)
                        logmag = 20.0*np.log10(np.abs(pump_sweep))
                        gain = logmag - logmag_memory
                        
                        max_gain_left = max(gain) if LEFT else max_gain_left
                        max_gain_right = max(gain) if RIGHT else max_gain_right
                        
                        if max_gain_left > max_gain_right:
                            LEFT, RIGHT = True, False
                            i_left += 3 if max_gain_left < GAIN/4.0 else 1
                        else:
                            LEFT, RIGHT = False, True
                            i_right += 3 if max_gain_right < GAIN/4.0 else 1
                        
                        if max(gain)<GAIN and max(gain)>MAX:
                            MAX = max(gain)
                        
                        for j, g in enumerate(gain):
                            if g > GAIN - 0.07:
                                ind = np.argmin(np.abs(gain[:j+2]-GAIN))
                                ag_power = ag_pows[ind]
                                # Run this check function to fix a couple of situations: 
                                #   * Noisy gain curve above the bifurcation threshold, filter out by checking the NVR
                                #   * Max gain at some other frequency is much larger than GAIN, adjust the pump power to level it at GAIN
                                noise_mem = fsp.take_Sparams_NF(pnax_NF, noise_names)
                                ag.set_frequency(ag_freq)
                                ag.set_power(ag_power)
                                ag.set_rf_on(True)
                                noise_data = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                                ag.set_rf_on(False)
                                nvr = noise_data['ys'][noise_names[0]] - noise_mem['ys'][noise_names[0]]
                                pnax_smc.set_meas_select(meas_names[0]) # return to CH1
                                if np.all(nvr>nvr_min) and np.all(nvr<nvr_max):
                                    set_center_span(pnax_smc, center_freq, span_gain, points=num_points, averages=averages_vna, power=vna_power) 
                                    print('Pump at %.3f GHz passed NVR test. Adjusting pump power:' %(ag_freq*1e-9))
                                    reflection_mem = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')['ys'][meas_names[0]]
                                    ag.set_rf_on(True)
                                    unleveled = True
                                    while unleveled:
                                        reflection_data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')['ys'][meas_names[0]]
                                        gain = 10*np.log10(np.sum(reflection_data**2, axis=0)) - 10*np.log10(np.sum(reflection_mem**2, axis=0))
                                        max_gain = max(gain)                    
                                        print('Max gain %.2f' %max_gain)
                                        if max_gain<GAIN+0.1:
                                            unleveled = False
                                            gain_flag = True
                                            ag.set_rf_on(False)
                                        else:
                                            ag_power -= 0.02
                                            if ag_power<start_pow: # exit the loop in this case, even though it is potentially a good point. 
                                                unleveled = False
                                            else:
                                                ag.set_power(ag_power)
                                break
                        if gain_flag:
                            break
                    if gain_flag:
                        break
                    else:
                        print('Max gain was %.2f' %MAX)
                if gain_flag:
                    break
            if not gain_flag:
                print('Couldn\'t find %.1f dB gain at this flux.' %GAIN)
            else:
                print('Taking gain with pump at %.2f GHz, %.2f dBm' %(ag_freq*1e-9, ag_power))
                # Fit the resonance
                set_center_span(pnax_smc, center_freq, span_linear, points=num_points, averages=averages_vna, power=vna_power)
                data_lin = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
                fitdata = fsp.fit_reflection_SMC(data_lin, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_middle,display=False)
                dataDir = fsp.save_Sparams(h5filename, data_lin, meas_class='SMC')
                fsp.save_fit_reflection(fitdata, h5filename, dataDir)                
                # Take gain, signal power sweep and NVR trace
                set_center_span(pnax_smc, center_freq, span_gain, points=num_points, averages=averages_vna, power=vna_power)
                guessCW = fsp.guessCW_maxLog
                memdata = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
                ag.set_frequency(ag_freq)
                ag.set_power(ag_power)
                ag.set_rf_on(True)
                data_gain = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL',
                                 pow_start=pow_start_P1dB, pow_stop=pow_stop_P1dB, vna_pow=vna_power,
                                 averages=averages_vna, pow_numPoints=num_points_pow, pow_aves=averages_pow,
                                 fpump=ag_freq, guessCW=guessCW)
                ag.set_rf_on(False)
                cfreqs = np.array([data_gain['POW']['freqINP']]) #use for SPA
                noisedata = {} 
                for cfreq in cfreqs:
                    noisedata[cfreq] = {}
                    pnax_NF.set_center_freq(cfreq)
                    pnax_NF.set_span(span_gain)
                    noisedata[cfreq]['memory'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                    ag.set_rf_on(True)
                    noisedata[cfreq]['data'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                    ag.set_rf_on(False)
                data_gain['noise'] = noisedata
                gainfits = None #fsp.fit_gain_SMC(data_gain['LIN'], memdata=memdata)
                noisefits = None #fsp.fit_nvr(noisedata, display=True) 
                powfits = None #fsp.fit_P1dB_SMC(data_gain['POW'], display=True)
                dataDirGain = fsp.save_gain_SMC(h5filename, data_gain, memdata=memdata, noisedata=noisedata, 
                                                linfits=gainfits, noisefits=noisefits, powfits=powfits)
                print('Narrow span Gain dataDir:' + dataDirGain)
                
                # Take also a wide-span (4-12 GHz) gain  
                set_center_span(pnax_smc, 8e9, 8e9, points=4001, averages=averages_vna, power=vna_power)
                guessCW = fsp.guessCW_maxLog
                memdata = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
                ag.set_rf_on(True)
                data_gain = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL', vna_pow=vna_power,
                                 averages=averages_vna, measure_P1dB=False, numPoints=4001)
                ag.set_rf_on(False)
                dataDirGainWideSpan = fsp.save_gain_SMC(h5filename, data_gain, memdata=memdata)
                print('Wide span Gain dataDir:' + dataDirGainWideSpan)
    
                # re-center the screen, update pumping data files
                set_center_span(pnax_smc, center_freq, span_linear, points=num_points, averages=averages_vna, power=vna_power)
                file_names = ['curr', 'date_linear', 'time_linear', 'date', 'time', 'date_wide_span', 'time_wide_span', 'pump_freq', 'pump_power']                    
                data_to_write = [str(current*1e3), dataDir[2:8], dataDir[10:16], dataDirGain[2:8], dataDirGain[10:16], 
                                 dataDirGainWideSpan[2:8], dataDirGainWideSpan[10:16], str(ag_freq*1e-9), str(ag_power)]
                for index, element  in enumerate(data_to_write):
                    f = open(path+file_names[index]+'.txt','a')
                    f.write(element + ' ')
                    f.write('\n')
                    f.close()
            plt.close('all')
            if gain_flag:
                break


