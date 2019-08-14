# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 17:02:22 2017

@author: JPC-Acquisition

Main script to run JPC type experiments with vna/pnax, yoko, generator
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.colorbar as plt_colorbar
import h5py
from time import sleep
from mclient import instruments
from scripts.flux_sweep import analysis_functions as af
from scripts.flux_sweep import FluxSweep as fs
from scripts.flux_sweep import FluxSweep_pnax as fsp
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

#    h5filename = r'Z:\Data\JPC_2019\05-15-2019 Cooldown\JAMPA11.hdf5'
    h5filename = r'Z:\Data\JPC_2019\07-02-2019 Cooldown\PPFSPA02_v2.hdf5'

    if 1: # instruments
        pnax_smc = instruments['pnax_smc']
        pnax_sa = instruments['pnax_sa']
        pnax_NF = instruments['pnax_NF']
        pnax_standard = instruments['pnax_standard']
        pnax_sweptIMD = instruments['pnax_sweptIMD']
        pnax_IMSpec = instruments['pnax_IMSpec']
#        ag_gain = instruments['ag1']
        ag_gain = instruments['ag2']
        yoko = instruments['yoko_vsr']
        fridge = instruments['fridge']
#        fridge = instruments.reload('fridge')
#        keith = instruments['keith']
#        yoko_ctl = instruments['yoko_ctl']
#        spec = instruments['spec']
    
    if 0: #vna flux sweep, don't use this one
        data = fs.flux_sweep(pnax, yoko, curr_start = -2.000e-3, curr_stop = 2.000e-3, curr_step=10e-6, averages=4, num_points=1601,
                   fridge=None,folder=None,display=False, atten=0,vna_fmt='POL', vna_sweep_type='LIN',
                   printlog=True, h5file=h5filename)
    if 0: #plot vna flux sweep
#        dataDir = fs.get_last_dataDir(h5filename)
        dataDir = '//170704//140854'
        fs.define_phaseColorMap()
        fs.colorMap_h5file(h5filename, dataDir=dataDir, colorMap=('hot','phase'))
#        (currents,Qc,Qi,Qtot,results) = fs.flux_sweep_Qfit(h5filename, dataDir=dataDir, numFitPoints=400, kc=50e6, ki=1e6, display=False)
#        fs.plot_flux_Qfit(h5filename, dataDir=dataDir, vna_fmt='POL')
    
    if 0: #flux sweep by meas names
        meas_names = ['"CH1_S11_1"']
#        curr_start = -5e-3 #1.55e-3
        curr_start = 0        
#        curr_stop = 5e-3 #3.05e-3
        curr_stop = 1e-5
        curr_step = 10e-6
        currents = np.linspace(curr_start, curr_stop, round(abs((curr_stop - curr_start) / curr_step) + 1))
        data = fsp.flux_sweep_SMC(pnax_smc, yoko, meas_names, currents=currents, yoko_slew = 10e-6)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC', flux_sweep=True)
#        fit_res = fsp.fit_flux_sweep_SMC(data, display=False, kc=70e6, ki=1e6, numFitPoints=400)#400
#        fsp.save_fit_sweep_SMC(fit_res, meas_names, h5filename, dataDir=None)
#        fsp.plot_fit_flux_sweep_SMC(fit_res, meas_names, data['currents'])
        mix=pnax_smc.get_mixerLO_freq_fixed()
        print('Flux sweep dataDir: ' + dataDir)
        fsp.plot_flux_sweep_SMC(data)

    if 1: # measure SMC scattering parameters and fit reflection resonances
#        meas_names = ['"CH1_S11_3"', '"CH1_SC12_9"', '"CH1_SC21_1"', '"CH1_S22_4"']
#        meas_names = ['"CH1_S11_1"','"CH1_S22_12"']
        meas_names = ['"CH1_S11_1"']
        data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN',
                                   opc_arg=False)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
#        fitdata = fsp.fit_reflection_SMC(data, kc=400e6, ki=1e6, guessF=fs.guessF_maxReal)
#        fsp.save_fit_reflection(fitdata, h5filename, dataDir)
        fsp.plot_Sparams_SMC(data)
        print('Sparams dataDir: ' + dataDir)

    if 0: # take SMA scattering parameters
        meas_names = ['"CH1_S11_1"']
        data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN',
                                   opc_arg=False)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
        fsp.plot_Sparams_SMC(data)
        print('Sparams dataDir: ' + dataDir)
    
    if 0: # take SA trace
        meas_names = ['"CH2_B_2"', '"CH2_A_6"']
        data = fsp.take_Sparams_SA(pnax_sa, meas_names)
        fsp.save_Sparams(h5filename, data, meas_class='SA')
        fsp.plot_Sparams_SA(data)
    
    if 0: # take gain point just SMC (LIN and POW sweeps)
        #meas_names = ['"CH1_S11_3"', '"CH1_SC12_9"', '"CH1_SC21_1"', '"CH1_S22_4"']
        meas_names = ['"CH1_S11_1"']
        data = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL')
        fsp.save_gain_SMC(h5filename, data)
        fsp.plot_gain_SMC(data)

    if 0: # take standard class scattering
        meas_names = ['"CH3_S33_3"', '"CH3_S34_8"', '"CH3_S43_9"', '"CH3_S44_10"']
#        meas_names = ['"CH3_S44_10"']
#        meas_names = ['"CH3_S33_3"']
#        meas_names = ['"CH3_S11_8"']
#        meas_names = ['"CH3_S44_10"','"CH3_S43_9"']
        data = fsp.get_Sparams_standard(pnax_standard, meas_names, vna_fmt='POL',
                                      sweep_type='LIN', opc_arg=False)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='Standard')
        fsp.plot_Sparams_SMC(data)
        print('Sparams dataDir: ' + dataDir)  
       
       
    if 0: # take full gain point, SMC and NF/SA
        meas_names = ['"CH1_S11_1"']
        spec_names = ['"CH2_A_2"']
        noise_names = ['"CH6_SYSNPDI_14"']
#        guessCW = fsp.guessCW_maxLog    # use this for JPCs
        guessCW = lambda x, y, f: fsp.guessCW_half_fp(x, y, f, delta=100e3) #use this for SPA
#        guessCW = lambda x, y, f: fsp.guessCW_near_fp(x, y, f, delta=100e3) #use this for SPA 4 wave gain
        fpump = ag_gain.get_frequency()
        ag_gain.set_rf_on(False)
        pnax_smc.set_averaging_state(True)
        memdata = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
        ag_gain.set_rf_on(True)
        data = fsp.take_gain_SMC(pnax_smc, meas_names, vna_fmt='POL',
                                 pow_start=-70, pow_stop=-30, vna_pow=-60,
                                 averages=10, pow_numPoints=201, pow_aves=100,
                                 fpump=fpump, guessCW=guessCW)
        ag_gain.set_rf_on(False)
#        cfreqs = np.array([data['POW']['freqINP'], data['POW']['freqOUTP']]) #use for JPC
        cfreqs = np.array([data['POW']['freqINP']]) #use for SPA
        noisedata = {} # specdata = {} 
        for cfreq in cfreqs:
            noisedata[cfreq] = {} # specdata[cfreq] = {}
            pnax_NF.set_center_freq(cfreq)
            noisedata[cfreq]['memory'] = fsp.take_Sparams_NF(pnax_NF, noise_names) # specdata[cfreq]['memory'] = fsp.take_Sparams_SA(pnax_sa, spec_names)
            ag_gain.set_rf_on(True)
            noisedata[cfreq]['data'] = fsp.take_Sparams_NF(pnax_NF, noise_names) # specdata[cfreq]['data'] = fsp.take_Sparams_SA(pnax_sa, spec_names)
            ag_gain.set_rf_on(False)
        data['noise'] = noisedata # data['spec'] = specdata
        gainfits = fsp.fit_gain_SMC(data['LIN'], memdata=memdata)
        noisefits = fsp.fit_nvr(noisedata, display=True) # specfits = fsp.fit_nvr(specdata, display=True)
        powfits = fsp.fit_P1dB_SMC(data['POW'], display=True)
        p_1dB = powfits[meas_names[0]]['P_1dB']
#        dataDir = fsp.save_gain_SMC(h5filename, data, memdata, None, gainfits, powfits=powfits)
#        dataDir = fsp.save_gain_SMC(h5filename, data, memdata, specdata, gainfits, specfits=specfits, powfits=powfits)
        dataDir = fsp.save_gain_SMC(h5filename, data, memdata=memdata, noisedata=noisedata, linfits=gainfits, noisefits=noisefits, powfits=powfits)
        fsp.plot_gain_SMC(data, memdata, noisedata) # fsp.plot_gain_SMC(data, memdata, specdata)
        print(dataDir)
        if 0: # set up sweptIMD
            ag_gain.set_rf_on(True)
            fsp.set_up_sweptIMD_POW(pnax_sweptIMD, fsp.guessCW_half_fp(None,None, fpump, delta=500e3),
                                    df=100e3, averages=5, pow_start=-55, pow_stop=-25, ifbw_im=50, ifbw_main=1e3)
    #        pnax_sweptIMD.set_trigger_mode('CONT') #so that it starts averaging while I try to fix timing issues
    #        pnax_sweptIMD.set_meas_select_trace(3) # Selects trace on swept IMD channel
        if 0: # swept IMD measurement
#            imd_names = ['"CH5_IM3_7"', '"CH5_PwrMain_5"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"', '"CH5_Pwr5_12"']
            imd_names = ['"CH5_IM3_7"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"']
            dataIMD = fsp.take_Sparams_sweptIMD(pnax_sweptIMD, imd_names, opc_arg=True, num_im=2)
            dataDirIMD = fsp.save_Sparams(h5filename, dataIMD, meas_class='SweptIMD')
            ag_gain.set_rf_on(False)
            fsp.plot_Sparams_sweptIMD(dataIMD, attenInp=0)
            fit_im_50 = fsp.fit_sweptIMD(dataIMD, -50)
            fit_im_p1dB = fsp.fit_sweptIMD(dataIMD, p_1dB)
            print('SweptIMD dataDir: ' + dataDirIMD)
     
     
     
    if 0: # pump power sweep, gain
        meas_names = ['"CH1_S11_1"']
        pow_start = -15.0
        pow_stop = 20.0
        pow_step = 0.1
        pows = np.linspace(pow_start, pow_stop, round(abs(pow_stop - pow_start) / pow_step + 1))
        guessCW = lambda x, y, f: fsp.guessCW_half_fp(x, y, f, delta=100e3)
        ag_freqs = np.array([17.1868]) * 1e9
        dataDirs = {}
        for ag_freq in ag_freqs:
            ag_gain.set_frequency(ag_freq)
            results = fsp.pump_pow_sweep_SMC(pnax_smc, ag_gain, meas_names, pows=pows, display=True,
                                             takeDR=True, powDR_start=-60, powDR_stop=-20, vna_pow=-60,
                                             pow_numPoints=201, pow_aves=100, fpump=None,
                                             guessCW=guessCW)
    #        dataDir = fsp.save_Sparams(h5filename, results['LIN'], memdata=results['memory'], meas_class='SMC',
    #                                   pump_pow_sweep=True)
            gainfits = fsp.fit_gain_SMC(results['LIN'], memdata=results['memory'], display=False)
            powfits = fsp.fit_P1dB_SMC(results['POW'], display=False)
            dataDir = fsp.save_gain_SMC(h5filename, results, results['memory'], linfits=gainfits,
                                        powfits=powfits, pump_pow_sweep=True)
            fsp.plot_pump_pow_sweep_SMC(results['LIN'], memdata=results['memory'], cmaps = ('hot','phase'))
            fsp.plot_pump_pow_sweep_SMC(results['POW'], cmaps = ('hot','phase'))
            fsp.plot_fitGain_pow_sweep_SMC(results, gainfits, powfits, atten=-71.5)
            pfits = fsp.plot_fit_P1dB_gain(gainfits, powfits, fpump=[results['LIN']['fpump']],
                                           atten=-71.5)
            print('Pump power sweep dataDir: ' + dataDir)
            dataDirs[ag_freq] = dataDir
        print(dataDirs)




    if 0: # pump power sweep, sweptIMD
        imd_names = ['"CH5_IM3_7"', '"CH5_PwrMain_5"', '"CH5_Pwr3_6"', '"CH5_IIP3_11"', '"CH5_Pwr5_12"']
        pow_start = -15.0
        pow_stop = 20.0
        pow_step = 0.1
        agPows = np.linspace(pow_start, pow_stop, round(abs(pow_stop - pow_start) / pow_step + 1))
        ag_freqs = np.array([17.1868]) * 1e9
        dataDirs = {}
        for ag_freq in ag_freqs:
            ag_gain.set_frequency(ag_freq)
            ag_gain.set_rf_on(True)
            fsp.set_up_sweptIMD_POW(pnax_sweptIMD, fsp.guessCW_half_fp(None,None, ag_freq, delta=500e3),
                                    df=100e3, averages=10, pow_start=-60, pow_stop=-30, ifbw_im=50, ifbw_main=1e3)
            imdResults = fsp.pump_pow_sweep_sweptIMD(pnax_sweptIMD, ag_gain, imd_names, agPows,
                                                     display=True, num_im=2, fpump=None)
            dataDir = fsp.save_Sparams(h5filename, imdResults, meas_class='SweptIMD', pump_pow_sweep=True)
            #TODO - fit swept IMD sweeps
            fsp.plot_pump_pow_sweep_sweptIMD(imdResults, attenInp=0, cmap='hot')
            print('Pump power sweep dataDir: ' + dataDir)
            dataDirs[ag_freq] = dataDir
        print(dataDirs)
    
    if 0: # ag power sweep, Stark shift
        meas_names = ['"CH1_S11_1"']
        pow_start = -20.0
        pow_stop = 16.0

        pows = np.linspace(10**(pow_start/10.0), 10**(pow_stop/10.0), 81)
        pows = 10*np.log10(pows)

        pows = np.round(pows, 2)
#        pows = np.linspace(pow_start, pow_stop, round(abs(pow_stop - pow_start) / pow_step + 1))
        currents = np.asarray([1.14,1.15,1.16])*1e-3  #np.arange(-0.2, 1.1, 0.02)*1e-3
        ag_freqs = np.array([7.8])*1e9
        dataDirs = {}
        slopes = {}
        for ag_freq in ag_freqs:
            slopes[ag_freq] = {}
            dataDirs[ag_freq] = {}
            ag_gain.set_frequency(ag_freq)
            for current in currents:
                wait_time = yoko.set_current_ramp(current, slew=10e-6)
                if wait_time:
                    sleep(wait_time + 0.5)
                results = fsp.pump_pow_sweep_SMC(pnax_smc, ag_gain, meas_names, pows=pows)
                (data, memdata) = (results['LIN'], results['memory'])                
                dataDir = fsp.save_Sparams(h5filename, data, memdata=memdata, meas_class='SMC', pump_pow_sweep=True)
                fsp.plot_pump_pow_sweep_SMC(data)
                fit_res = fsp.fit_pow_sweep_SMC(data, display=False, kc=200e6, ki=1.0e6)
                fsp.save_fit_sweep_SMC(fit_res, meas_names, h5filename, dataDir=None,
                                       extParam_name='fpump', extParam=data['fpump'])
                fsp.plot_fit_pow_sweep_SMC(fit_res, meas_names, data['powers'])
                print('Pump power sweep dataDir: ' + dataDir)
                arr = dataDir.split('//')
                slopes[ag_freq][current] = fsp.plot_freqshift_sweep(h5filename, '//'+arr[1]+'//'+arr[2], sweep_var='powers_swept',
                                                            atten=-70.2, f_pump=data['fpump'], porder=2)
                dataDirs[ag_freq][current] = dataDir


        
        
    if 0: # vna power sweep, remember to set attenuators so you can achieve powers (-50 to -30 dB is good usually)
        meas_names = ['"CH1_S11_3"']
        pow_start = -80.0 #-60
        pow_stop = -65.0 #-35 also set source attenuation on pnax to -30
        pow_step = 0.5
        pows = np.linspace(pow_start, pow_stop, round(abs(pow_stop - pow_start) / pow_step + 1))
        currents = np.array([0.400])*1e-3
        dataDirs = {}
        slopes = {}
        for current in currents:
            wait_time = yoko.set_current_ramp(current, slew=10e-6)
            if wait_time:
                sleep(wait_time + 0.5)
            data = fsp.vna_pow_sweep_SMC(pnax_smc, meas_names, pows=pows)
            dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC', vna_pow_sweep=True)
            fsp.plot_pump_pow_sweep_SMC(data)
            fit_res = fsp.fit_pow_sweep_SMC(data, display=False, kc=60e6, ki=1e6, numFitPoints=0)
            fsp.save_fit_sweep_SMC(fit_res, meas_names, h5filename, dataDir=None)
            fsp.plot_fit_pow_sweep_SMC(fit_res, meas_names, data['powers'])
            print('VNA power sweep dataDir: ' + dataDir)
            arr = dataDir.split('//')
            slopes[current] = fsp.plot_freqshift_sweep(h5filename, '//'+arr[1]+'//'+arr[2], atten=-70.6)
            dataDirs[current] = dataDir
        
    if 0: # vna freq/power sweep, pow sweep at each cw_freq 
        meas_names = ['"CH1_S11_3"']
        pow_start = -40.0
        pow_stop = 10.0 
        pow_step = 0.1 # don't go higher than -5 dBm for line 4 (without add atten before receiver)
        cw_start = pnax_smc.get_mixerINP_freq_start()
        cw_stop = pnax_smc.get_mixerINP_freq_stop()
        cw_freqs = np.linspace(cw_start, cw_stop, 801)
        currents = np.array([2.1])*1e-3
        pnax_smc.set_power_on(True)
        for current in currents:
            wait_time = yoko.set_current_ramp(current, slew=10e-6)
            if wait_time:
                sleep(wait_time + 0.5)     
            data = fsp.take_cw_VNApow_sweep_SMC(pnax_smc, meas_names, cw_freqs, pow_start,
                                                pow_stop, pow_step, display=False,
                                                sweep_updown=True)
            dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC', cw_VNApow_sweep=True)
            fsp.plot_pump_pow_sweep_SMC(data, cw_VNApow_sweep=True) 
            print('current: ' + str(current*1e3) + ' mA')
            print('VNA cw freq and power sweep dataDir: ' + dataDir)
        fsp.set_up_Sparams_SMC(pnax_smc, averages=4)   
        pnax_smc.set_power_on(False)
#        yoko_cur=yoko_cur+0.2
        
    if 0: # vna power sweep at each flux
        meas_names = ['"CH1_S11_3"']
        pow_start = -60.0
        pow_stop = -20.0
        pow_step = 1.0
        pows = np.linspace(pow_start, pow_stop, round(abs(pow_stop - pow_start) / pow_step + 1))
        curr_start = -6.0e-3
        curr_stop = 6.0e-3
        curr_step = 10e-6
        currents = np.linspace(curr_start, curr_stop, round(abs(curr_stop - curr_start) / curr_step + 1))
        data = fsp.flux_sweep_VNApow_sweep_SMC(pnax_smc, yoko, meas_names, currents, pows)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC',
                                   flux_sweep=True, vna_pow_sweep=True)
        # TODO - plot the results
        print('VNA power sweep at each flux dataDir: ' + dataDir)
        
    if 0: #repeatedly measuring a static, standard set of SMC scattering parameters as a function of time 
        meas_names = ['"CH1_S11_1"','"CH1_S22_2"']
        wait_time=5 # wait time is in seconds. Check what happens with averaging. 
        total_time=7200 #total time you want to make repeated measurements for in seconds. Note this includes averaging.
        current_time=0        
        while(current_time<total_time):
            data = fsp.get_Sparams_SMC(pnax_smc, meas_names, vna_fmt='POL', sweep_type='LIN')
            dataDir = fsp.save_Sparams(h5filename, data, meas_class='SMC')
            print(current_time)
            print('repeated measurement data directory: ' + dataDir)
            sleep(wait_time)
            current_time=current_time+wait_time
            #end while
        print('experiment done')
        
    if 0: # IM spectrum
        meas_names = ['"CH4_Output_4"']
        data = fsp.take_Sparams_IMSpec(pnax_IMSpec, meas_names, opc_arg=False)
        dataDir = fsp.save_Sparams(h5filename, data, meas_class='IMSpec')
        fsp.plot_Sparams_IMSpec(data)
        print('IM Spectrum dataDir: ' + dataDir)

    if 0: #CMOS switch standard scattering over time
        meas_names = ['"CH3_S11_1"', '"CH3_S12_4"', '"CH3_S21_5"', '"CH3_S22_6"']
        switch_states = ['SUP_off_CTL_off', 'SUP_0.9_CTL_off', 'SUP_0.9_CTL_0.9']
        def bias_switch(sup_state=False, ctl_state=False):
            keith.set_output_state(sup_state)
            yoko_ctl.set_output_state(ctl_state)
        change_switch = [lambda: bias_switch(False, False),
                         lambda: bias_switch(True, False),
                        lambda: bias_switch(True, True)]
        dataDirs = {}
        temp_names = ['fridge_ch1_temp', 'fridge_ch2_temp'] #cernox, RuOx
        wait_time = 60*60
        times = np.arange(0, 48*60*60, wait_time) #times to take data in seconds
        temps_ch1 = {}
        temps_ch2 = {}
        for state in switch_states:
            dataDirs[state] = []
            temps_ch1[state] = np.zeros_like(times)
            temps_ch2[state] = np.zeros_like(times)
        for i_t, t in enumerate(times):
            print(t)
            fridge = instruments.reload('fridge')
            for kk, state in enumerate(switch_states):
                change_switch[kk]()
                data = fsp.get_Sparams_helper(pnax_standard, meas_names, vna_fmt='POL',
                                          sweep_type='LIN', opc_arg=True, fridge=fridge)
                dataDir = fsp.save_Sparams(h5filename, data, meas_class='Standard',
                                       metadata=state)
#                fsp.plot_Sparams_SMC(data)
                dataDirs[state].append(dataDir)
                temps_ch1[state][i_t] = data[temp_names[0]]
                temps_ch2[state][i_t] = data[temp_names[1]]
                print('Sparams dataDir: ' + dataDir)
            bias_switch(False, False)
            if i_t % 6 == 0:
                fsp.plot_Sparams_SMC(data, metadata='T_1 = %.3f K, T_2 = %.3f'%(data[temp_names[0]], data[temp_names[1]]))
            sleep(wait_time)
        print('experiment done')
        
    if 0: #plot flux sweep from file
       dataDir = '181111/124011/'
       results = fsp.load_h5_Sparams(h5filename, dataDir)
       data = results['LIN']
       meas_names = data['meas_names']
       fit_res = fsp.fit_flux_sweep_SMC(data, display=False, kc=1800e6, ki=1e6, numFitPoints=400, updateKappaGuess=False)#400
       fsp.plot_fit_flux_sweep_SMC(fit_res, meas_names, data['currents'])

       
        
        