#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:06:35 2018

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

def get_response_at_fixed_freq(pnax, freq, meas_names, averages = 300, logmag=True, phase=False,
                               vna_pow = False):
    if vna_pow: pnax.set_power(vna_pow)
    pnax.set_mixerOUTP_freq_mode('FIXED')
    pnax.set_mixerINP_freq_mode('FIXED')
    pnax.set_mixerINP_freq_fixed(freq)
    pnax.mixer_calc('OUTP')
    pnax.set_sweep_type('LIN')    
    pnax.mixer_apply_settings()
    pnax.set_trigger_mode('CONT')
    pnax.set_points(1)
    pnax.set_averaging_state(False)
    pnax.set_averaging_mode('sweep')
    pnax.set_average_factor(averages)
    pnax.set_averaging_state(True)
    memory = fsp.get_Sparams_helper(pnax, meas_names, vna_fmt='POL', sweep_type='LIN')
    meas = meas_names[0]
    Logmag = 10*np.log10( (memory['ys'][meas][0])**2 + (memory['ys'][meas][1])**2 )
    Phase = np.angle( memory['ys'][meas][0]+ memory['ys'][meas][1]*1j )/np.pi*180
    return Logmag if logmag else Phase
    
    
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



def tune_up_gain(pnax, pow_start, pow_stop, guess_CW, averages, meas_names, G=20):
#  sweep generator power and take logmag data at guess_CW until you reach 20 dB    
    logmag_memory = get_response_at_fixed_freq(pnax_smc, guess_CW, meas_names, averages=averages, logmag=True)
    ag_pow = pow_start
    ag_gain.set_rf_on(True)
    ag_gain.set_power(ag_pow)
    Gain = get_response_at_fixed_freq(pnax_smc, guess_CW, meas_names, averages=averages, logmag=True) - logmag_memory
    ag_gain.set_rf_on(False)
    print('Gain %f' %Gain)

#    while Gain < 19.95 and (ag_pow < pow_stop):
#        if Gain<4:
#            ag_pow +=2
#        elif Gain<15:
#            ag_pow += 1
#        elif Gain<18:
#            ag_pow += 0.02
#        else:
#            ag_pow += 0.01

    while Gain < G-0.05 and (ag_pow < pow_stop):
        if Gain<3:
            ag_pow +=1
        else:
            ag_pow += 0.02
            
        ag_gain.set_rf_on(True)
        ag_gain.set_power(ag_pow)
        Gain = get_response_at_fixed_freq(pnax_smc, guess_CW, meas_names, averages=averages, logmag=True) - logmag_memory
        ag_gain.set_rf_on(False)
        print('Gain %f' %Gain)

    flag = (ag_pow < pow_stop)
    return Gain, logmag_memory, ag_pow, flag




def figure_out_gains(pnax, ag, pnax_freq, pow_start, pow_stop, pow_step, averages, 
                     vna_pow, ag_freq, ifbw, meas_names, want_gains):

    logmag_memory = get_response_at_fixed_freq(pnax, pnax_freq, meas_names, averages=averages, logmag=True)
    data = PNAXspectroscopy(pnax, ag, pnax_freq, ifbw, (pow_stop-pow_start)/pow_step, 
                            averages, vna_pow, ag_freq, None, pow_start, pow_stop,vna_fmt='log mag') 
    logmag = 20*np.log10(np.abs(data))
    gain = logmag-logmag_memory
    ag_powers = np.linspace(pow_start,pow_stop,round((pow_stop-pow_start)/pow_step))
    gains_possible = []
    ag_powers_possible = []    
    i = 0
    for j, g in enumerate(gain):
        if g > want_gains[i]-0.07:
            ind = np.argmin(np.abs(gain[:j+2]-want_gains[i]))
            gains_possible.append(want_gains[i])
            ag_powers_possible.append(ag_powers[ind])
            i+=1
        if i == len(want_gains):
            break            
    return np.asarray(gains_possible), np.asarray(ag_powers_possible), np.asarray(gain)





if __name__ == '__main__':
    
    logging.getLogger().setLevel(logging.DEBUG)
    path = r'Z:\Data\JPC_2019\05-15-2019 Cooldown\\'
    h5filename = path + 'JAMPA10.hdf5'

    if 1: # instruments
        pnax_smc = instruments['pnax_smc']
        pnax_sweptIMD = instruments['pnax_sweptIMD']
        pnax_NF = instruments['pnax_NF']
        ag_gain = instruments['ag2']
        yoko = instruments['yoko']
        fridge = instruments['fridge']


# --------------------------------------------------
# -------------------------------------------------- 
# ---------- characterization sweep for SPA --------
# -------------------------------------------------- 
# -------------------------------------------------- 
    if 1:

        meas_names = ['"CH1_S11_1"']
        noise_names = ['"CH6_SYSNPDI_14"']
        meas = meas_names[0]

        # Current sweep
        curr_start = 1.30e-3
        curr_stop = -1.00e-3
        curr_step = 0.02e-3
        currents = np.linspace(curr_start, curr_stop, round(abs((curr_stop - curr_start) / curr_step) + 1))

        # PNAX parameters
        span_gain = 0.3e9
        span_linear = 1.0e9
        vna_pow = -60.0
        averages_vna = 10
        num_points = 1601
        
        # Generator powers
        pow_start = -20.0
        pow_stop = 20.0
        pow_step = 0.02
        averages = 20            # averages in the power sweep of the generator that tunes up gain


        # IMD measurement parameters
        pow_start_IMD = -60.0
        pow_stop_IMD = -30.0
        averages_IMD = 5.0
        delta_IMD = 500e3   # detuning from pump/2 of the centroid of the two main tones
        df_IMD = 100e3      # spacing between the two main tones
        ifbw_im = 50
        ifbw_main = 1e3
        
        # Power sweep
        num_points_pow = 201
        pow_start_P1dB = -70
        pow_stop_P1dB = -30
        averages_pow = 100 #200

        want_gains = [0, 20]
        
        # Guess parameters for the resonance fit
        kc_guess = 80e6
        ki_guess = 1e6


        for current in currents:
            
            # change the current
            wait_time = yoko.set_current_ramp(current, slew=10e-6)
            if wait_time:
                sleep(wait_time + 0.5) #0.5 for GBIP overhead    
            
            # fit the resonance
            data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
            dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
            fitdata = fsp.fit_reflection_SMC(data, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_maxReal)
            fsp.save_fit_reflection(fitdata, h5filename, dataDir)
            freq_c = fitdata[meas][0][0]
            freqLO = data['freqLO']
            freq_pump = 2.0*freq_c
            print('Pump frequency %f GHz' %(freq_pump*1e-9)  )
            guess_CW = fsp.guessCW_half_fp(None, None, freq_pump, delta=100e3)

            possible_gains, ag_powers, gains = figure_out_gains(pnax_smc, ag_gain, guess_CW, pow_start, pow_stop, 
                                                       pow_step, averages, vna_pow, freq_pump, 
                                                       3e3, meas_names, want_gains)
            
            print('Possible gains:')
            print(possible_gains)
            print('At generatore powers:')
            print(ag_powers)
                                                       
            f = open(path+'gain_sweep.txt','a')
            for element in gains:
                f.write(str(element)+' ')
            f.write('\n')
            f.close()                                                          
     
            for i, G in enumerate(possible_gains):
                new_path = path+r'Gain_'+str(G)+r'\\'
                if G==0:
                    # set up swept IMD measurement 
                    print('Taking IMD data with pump off' )
                    ag_gain.set_rf_on(False)
                    fsp.set_up_sweptIMD_POW(pnax_sweptIMD, 
                                            fsp.guessCW_half_fp(None,None, freq_pump, delta=delta_IMD),
                                            df=df_IMD, numPoints=num_points_pow, averages=averages_IMD, 
                                            pow_start=pow_start_IMD, pow_stop=pow_stop_IMD, 
                                            ifbw_im=ifbw_im, ifbw_main=ifbw_main)
#                    imd_names = ['"CH5_IM3_7"', '"CH5_PwrMain_5"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"', '"CH5_Pwr5_12"']
                    imd_names = ['"CH5_IM3_7"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"']
                    dataIMD = fsp.take_Sparams_sweptIMD(pnax_sweptIMD, imd_names, opc_arg=True, num_im=2)
                    dataDirIMD = fsp.save_Sparams(h5filename, dataIMD, meas_class='SweptIMD')
                    print('SweptIMD dataDir: ' + dataDirIMD)
                    
                    # update pumping data files
                    file_names = ['curr','date_linear','time_linear','IMD_time','IMD_date']                    
                    data_to_write = [str(current*1e3), dataDir[2:8], dataDir[10:16], dataDirIMD[10:16], dataDirIMD[2:8]]
                    for index, element  in enumerate(data_to_write):
                        f = open(new_path+file_names[index]+'.txt','a')
                        f.write(element + ' ')
                        f.write('\n')
                        f.close()                    
                else:
                    ag_pow = ag_powers[i]
                    print('%.1f dB gain at pump power %f' %(G,ag_pow) )       
                    set_center_span(pnax_smc, freq_pump/2.0, span_gain, points=num_points, averages=averages_vna, power=vna_pow)
    
                    # take full gain (Nick's code)
                    guessCW = lambda x, y, f: fsp.guessCW_half_fp(x, y, f, delta=100e3)
                    memdata = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
                    ag_gain.set_frequency(freq_pump)                    
                    ag_gain.set_power(ag_pow)                    
                    ag_gain.set_rf_on(True)
                    data = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL',
                                     pow_start=pow_start_P1dB, pow_stop=pow_stop_P1dB, vna_pow=vna_pow,
                                     averages=averages_vna, pow_numPoints=num_points_pow, pow_aves=averages_pow,
                                     fpump=freq_pump, guessCW=guessCW)      
                    ag_gain.set_rf_on(False)
                    cfreqs = np.array([data['POW']['freqINP']]) #use for SPA
                    noisedata = {} 
                    for cfreq in cfreqs:
                        noisedata[cfreq] = {} 
                        pnax_NF.set_center_freq(cfreq)
                        noisedata[cfreq]['memory'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                        ag_gain.set_rf_on(True)
                        noisedata[cfreq]['data'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                        ag_gain.set_rf_on(False)
                    data['noise'] = noisedata 
                    gainfits = fsp.fit_gain_SMC(data['LIN'], memdata=memdata)
                    noisefits = fsp.fit_nvr(noisedata, display=True) 
                    powfits = fsp.fit_P1dB_SMC(data['POW'], display=True)
                    p_1dB = powfits[meas_names[0]]['P_1dB']
                    dataDirGain = fsp.save_gain_SMC(h5filename, data, memdata=memdata, noisedata=noisedata, linfits=gainfits, noisefits=noisefits, powfits=powfits)
                    print(dataDirGain)                                     

                    # set up swept IMD measurement 
                    ag_gain.set_rf_on(True)
                    fsp.set_up_sweptIMD_POW(pnax_sweptIMD, 
                                            fsp.guessCW_half_fp(None,None, freq_pump, delta=delta_IMD),
                                            df=df_IMD, numPoints = num_points_pow, averages=averages_IMD, 
                                            pow_start=pow_start_IMD, pow_stop=pow_stop_IMD, 
                                            ifbw_im=ifbw_im, ifbw_main=ifbw_main)
#                    imd_names = ['"CH5_IM3_7"', '"CH5_PwrMain_5"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"', '"CH5_Pwr5_12"']
                    imd_names = ['"CH5_IM3_7"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"']
                    dataIMD = fsp.take_Sparams_sweptIMD(pnax_sweptIMD, imd_names, opc_arg=True, num_im=2)
                    dataDirIMD = fsp.save_Sparams(h5filename, dataIMD, meas_class='SweptIMD')
                    ag_gain.set_rf_on(False)
                    print('SweptIMD dataDir: ' + dataDirIMD)

                    # update pumping data files
                    file_names = ['curr','date_linear','time_linear','IMD_time',
                    'IMD_date','date','time','pump_freq','pump_power']                    
                    data_to_write = [str(current*1e3), dataDir[2:8], dataDir[10:16], dataDirIMD[10:16], 
                                     dataDirIMD[2:8], dataDirGain[2:8],dataDirGain[10:16],str(freq_pump*1e-9),str(ag_pow)]
                    for index, element  in enumerate(data_to_write):
                        f = open(new_path+file_names[index]+'.txt','a')
                        f.write(element + ' ')
                        f.write('\n')
                        f.close()

#           re center screen, prepare for next linear scattering
            set_center_span(pnax_smc, freq_pump/2.0, span_linear, points=num_points, 
                            averages=averages_vna, power=vna_pow)
            plt.close('all')
            


# --------------------------------------------------
# -------------------------------------------------- 
# ---------- Pump parameters sweep for SPA ---------
# -------------------------------------------------- 
# -------------------------------------------------- 
    if 0:

        meas_names = ['"CH1_S11_1"']
        noise_names = ['"CH6_SYSNPDI_14"']
        meas = meas_names[0]


        # PNAX parameters
        span_gain = 0.3e9
        span_linear = 0.7e9
        vna_pow = -65.0
        averages_vna = 10
        num_points = 1601
        
        # Generator powers
        pow_start = 0.0
        pow_stop = 20.0
        pow_step = 0.01
        averages = 40            # averages in the power sweep of the generator that tunes up gain


        # IMD measurement parameters
        pow_start_IMD = -70.0
        pow_stop_IMD = -40.0
        averages_IMD = 5.0
        delta_IMD = 500e3   # detuning from pump/2 of the centroid of the two main tones
        df_IMD = 100e3      # spacing between the two main tones
        ifbw_im = 50
        ifbw_main = 1e3
        
        # Power sweep
        num_points_pow = 201
        pow_start_P1dB = -70
        pow_stop_P1dB = -30
        averages_pow = 100 #200

        want_gains = [20]
        
        # Guess parameters for the resonance fit
        kc_guess = 100e6
        ki_guess = 1e6

        # Yoko current 
        current = 0.46e-3
        wait_time = yoko.set_current_ramp(current, slew=10e-6)
        if wait_time:
            sleep(wait_time + 0.5) #0.5 for GBIP overhead    

        # fit the resonance
        data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN', opc_arg=True)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
        fitdata = fsp.fit_reflection_SMC(data, kc=kc_guess, ki=ki_guess, guessF=fs.guessF_maxReal)
        fsp.save_fit_reflection(fitdata, h5filename, dataDir)
        
        
        #Pump frequency sweep
        freq_pump_0 = 2.0*fitdata[meas][0][0]
        freq_pump_start = freq_pump_0 + 0.3e9
        freq_pump_stop = freq_pump_0 - 0.3e9

#        freq_pump_start = 18.8e9
#        freq_pump_stop = 17.5e9


        freq_pump_step = 10e6
        pumps = np.linspace(freq_pump_start, freq_pump_stop, round(abs((freq_pump_stop- freq_pump_start) / freq_pump_step) + 1))
        np.savetxt(path+'pump_freqs.txt', pumps)
        
        
        for freq_pump in pumps:
            
            print('Pump frequency %f GHz' %(freq_pump*1e-9)  )
            guess_CW = fsp.guessCW_half_fp(None, None, freq_pump, delta=100e3)

            possible_gains, ag_powers, gains = figure_out_gains(pnax_smc, ag_gain, guess_CW, pow_start, pow_stop, 
                                                       pow_step, averages, vna_pow, freq_pump, 
                                                       3e3, meas_names, want_gains)
            flag_array = [True for g_ in possible_gains]

            print('Possible gains:')
            print(possible_gains)
            print('At generatore powers:')
            print(ag_powers)

            f = open(path+'gain_sweep.txt','a')
            for element in gains:
                f.write(str(element)+' ')
            f.write('\n')
            f.close()
     
            for i, G in enumerate(possible_gains):

                new_path = path+r'Gain_'+str(G)+r'\\'                
                if flag_array[i]:
                    np.savetxt(new_path+'current.txt',np.asarray([current])) 
                    f = open(new_path+'date_linear.txt','w')
                    f.write(dataDir[2:8])
                    f.close()                    
                    f = open(new_path+'time_linear.txt','w')
                    f.write(dataDir[10:16])
                    f.close()                     
                    flag_array[i] = False                                         
                    
                ag_pow = ag_powers[i]
                print('%.1f dB gain at pump power %f' %(G,ag_pow) )
                set_center_span(pnax_smc, freq_pump/2.0, span_gain, points=num_points, averages=averages_vna, power=vna_pow)

                # take full gain (Nick's code)
                guessCW = lambda x, y, f: fsp.guessCW_half_fp(x, y, f, delta=100e3)
                memdata = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
                ag_gain.set_frequency(freq_pump)                    
                ag_gain.set_power(ag_pow)                    
                ag_gain.set_rf_on(True)
                data = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL',
                                 pow_start=pow_start_P1dB, pow_stop=pow_stop_P1dB, vna_pow=vna_pow,
                                 averages=averages_vna, pow_numPoints=num_points_pow, pow_aves=averages_pow,
                                 fpump=freq_pump, guessCW=guessCW)
                ag_gain.set_rf_on(False)
                cfreqs = np.array([data['POW']['freqINP']]) #use for SPA
                noisedata = {} 
                for cfreq in cfreqs:
                    noisedata[cfreq] = {} 
                    pnax_NF.set_center_freq(cfreq)
                    noisedata[cfreq]['memory'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                    ag_gain.set_rf_on(True)
                    noisedata[cfreq]['data'] = fsp.take_Sparams_NF(pnax_NF, noise_names) 
                    ag_gain.set_rf_on(False)
                data['noise'] = noisedata 
                gainfits = fsp.fit_gain_SMC(data['LIN'], memdata=memdata)
                noisefits = fsp.fit_nvr(noisedata, display=True) 
                powfits = fsp.fit_P1dB_SMC(data['POW'], display=True)
                p_1dB = powfits[meas_names[0]]['P_1dB']
                dataDirGain = fsp.save_gain_SMC(h5filename, data, memdata=memdata, noisedata=noisedata, linfits=gainfits, noisefits=noisefits, powfits=powfits)
                print(dataDirGain)                                     

                
#                # set up swept IMD measurement 
#                ag_gain.set_rf_on(True)
#                fsp.set_up_sweptIMD_POW(pnax_sweptIMD, 
#                                        fsp.guessCW_half_fp(None,None, freq_pump, delta=delta_IMD),
#                                        df=df_IMD, numPoints = num_points_pow, averages=averages_IMD, 
#                                        pow_start=pow_start_IMD, pow_stop=pow_stop_IMD, 
#                                        ifbw_im=ifbw_im, ifbw_main=ifbw_main)
##                    imd_names = ['"CH5_IM3_7"', '"CH5_PwrMain_5"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"', '"CH5_Pwr5_12"']
#                imd_names = ['"CH5_IM3_7"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"','"CH5_PwrMain_5"']
#                dataIMD = fsp.take_Sparams_sweptIMD(pnax_sweptIMD, imd_names, opc_arg=True, num_im=2)
#                dataDirIMD = fsp.save_Sparams(h5filename, dataIMD, meas_class='SweptIMD')
#                ag_gain.set_rf_on(False)
#                print('SweptIMD dataDir: ' + dataDirIMD)
#
#                # update pumping data files
#                file_names = ['IMD_time','IMD_date','date','time','pump_freq','pump_power']
#                data_to_write = [dataDirIMD[10:16], dataDirIMD[2:8], dataDirGain[2:8],dataDirGain[10:16],str(freq_pump*1e-9),str(ag_pow)]
#                for index, element  in enumerate(data_to_write):
#                    f = open(new_path+file_names[index]+'.txt','a')
#                    f.write(element + ' ')
#                    f.write('\n')
#                    f.close()


                # update pumping data files
                file_names = ['date','time','pump_freq','pump_power']
                data_to_write = [dataDirGain[2:8],dataDirGain[10:16],str(freq_pump*1e-9),str(ag_pow)]
                for index, element  in enumerate(data_to_write):
                    f = open(new_path+file_names[index]+'.txt','a')
                    f.write(element + ' ')
                    f.write('\n')
                    f.close()
                        
#           re center screen, prepare for next linear scattering
            set_center_span(pnax_smc, freq_pump_0/2.0, span_linear, points=num_points, 
                            averages=averages_vna, power=vna_pow)
            plt.close('all')
            
        