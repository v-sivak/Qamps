from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:01:28 2018

@author: vs362
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from devices_functions import *
from Amplifier import Amplifier

from scipy.constants import Boltzmann, e, h

SLASH = '/' #'/' for Mac addresses or '\\' for PC



# will store the array modes as clustered from the two-tone spectroscopy sweep
class array_modes():
    def __init__(self):
        self.mode_num_arr = []
        self.mode_currents_arr = []
        self.mode_freqs_arr = []
    def add_mode(self, N, currents, freqs):
        self.mode_num_arr.append(N)
        self.mode_currents_arr.append(currents)
        self.mode_freqs_arr.append(freqs)
    def get_mode(self, n):
        m = self.mode_num_arr.index(n)
        return self.mode_currents_arr[m], self.mode_freqs_arr[m]


class JAMPA(Amplifier):
    
    def __init__(self, path, name, M):
        self.M = M
        super(JAMPA,self).__init__(path,name)
                
#   TODO: this is a bit overcomplicated, can simply implement this array as a dictionary           
    def add_array_modes(self, array):
        if array.mode_num_arr:
            hdf5 = h5py.File(self.data_file)
            try:
                if 'array_modes' in hdf5.keys():
                    del hdf5['array_modes']
                grp = hdf5.create_group('array_modes')
                for n in array.mode_num_arr:
                    i,f = array.get_mode(n)
                    mode_grp = grp.create_group(str(int(n)))
                    mode_grp.create_dataset('currents', data = i)
                    mode_grp.create_dataset('res_freqs', data = f)
            finally:
                hdf5.close()
    
    def load_array_modes(self):
        hdf5 = h5py.File(self.data_file)
        try:
            if 'array_modes' in hdf5.keys():
                array = array_modes()
                for x in hdf5['array_modes'].keys():
                    num_mode = int(x)
                    currents = np.asarray(hdf5['array_modes'][x].get('currents'))
                    freqs = np.asarray(hdf5['array_modes'][x].get('res_freqs'))
                    array.add_mode(num_mode,currents,freqs)
                flag = True
            else:
                flag = False
        finally:
            hdf5.close()
        return array if flag else None
        
                
    def plot_array_modes(self, fits = False):
        
        array = self.load_array_modes()
        attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a)) ]
        flux_calibrated = True if ('a' in attributes) and ('b' in attributes) else False
        if array:
            n_modes = len(array.mode_num_arr)
            colors = mpl.cm.rainbow(np.linspace(0,1,n_modes))
            fig4, ax4 = plt.subplots(1,dpi=150)
            for n in array.mode_num_arr:
                i,f = array.get_mode(n)
                X = Flux(np.asarray(i),self.a,self.b) if flux_calibrated else np.asarray(i)*1e3
                plt.plot(X,np.asarray(f)*1e-9, color=colors[n-1],marker='.',markersize=2,linestyle='None',label='mode %d'%n)
                ax4.set_ylabel(r'${\rm Frequency \,(GHz)}$')
                if not fits: 
                    ax4.set_xlabel(r'${\rm Current\,(mA)}$')
                else: 
                    hdf5 = h5py.File(self.data_file)
                    try:
                        currents_theor = np.asarray(hdf5['two_tone_spectroscopy_data'].get('currents'))
                    finally:
                        hdf5.close()
                    plt.plot(Flux(currents_theor,self.a,self.b),1e-9*freq_array_modes_SNAIL_with_cap(np.asarray((currents_theor,np.ones(len(currents_theor))*n)), 
                                  self.alpha, self.a, self.b, self.x_c, self.y_c, self.f_0), linestyle='-', linewidth=0.75, color=colors[n-1])
#                    plt.plot(Flux(currents_theor,self.a,self.b),1e-9*freq_array_modes_SNAIL_direct_coupl(np.asarray((currents_theor,np.ones(len(currents_theor))*n)), 
#                                  self.alpha, self.a, self.b, self.y_c, self.f_0), linestyle='-', linewidth=0.75, color=colors[n-1])
                    ax4.set_xlabel(r'${\rm Flux\,\Phi/\Phi_0}$')
                    plt.legend(loc='best')
            fig4.savefig(self.device_folder + 'array_modes_flux_sweep_fit.png',dpi=240)
                
                
    def fit_array_modes(self, a_guess, b_guess, x_c_guess, y_c_guess, f0_guess, alpha_guess):
        
        array = self.load_array_modes()
        # Fit the array modes!
        Ns, Is, Fs = ([],[],[])
        for n in array.mode_num_arr:
            i,f = array.get_mode(n)
            for ind, curr in enumerate(i):
                Ns.append(n)
                Is.append(curr)
                Fs.append(f[ind])
        self.alpha, self.a, self.b, self.x_c, self.y_c, self.f_0 = fit_array_modes_flux_sweep_v1(Is, Ns, Fs, a_guess, b_guess, x_c_guess, y_c_guess, f0_guess, alpha_guess)
#        self.alpha, self.a, self.b, self.y_c, self.f_0 = fit_array_modes_flux_sweep_v2(Is, Ns, Fs, a_guess, b_guess, y_c_guess, f0_guess, alpha_guess)
        self.save_attributes()

                
    def calculate_unit_cell_with_cap(self, L_j):
        self.L_j = L_j
        self.C_j = 1/L_j/(2*np.pi*self.f_0)**2
        self.C_0 = self.C_j*(self.y_c/self.M)**2
        self.C = self.M*self.C_0*self.x_c
        self.Z_j = np.sqrt(self.L_j/self.C_0)
        self.save_attributes()



    #   Loads the data about measured gain points into hdf5 file, then you can use it to plot all kinds of things
    def load_multiple_gains_sweep(self, meas_name, IMD_flag=False, NVR_flag=True):
        
        self.pumping_meas_name = meas_name
        nominal_gain_array = [20]
        
        hdf5 = h5py.File(self.data_file)  
        try:
            if 'multiple_gains_sweep' in hdf5.keys():
                del hdf5['multiple_gains_sweep']
            grp = hdf5.create_group('multiple_gains_sweep')        
            temp_folder = 'multiple_gains_sweep'+SLASH
            data_dict = {}
            data_dict['current'] = np.loadtxt(self.device_folder+temp_folder+'curr.txt')*1e-3
            data_dict['date_linear'] = np.loadtxt(self.device_folder+temp_folder+'date_linear.txt')
            data_dict['time_linear'] = np.loadtxt(self.device_folder+temp_folder+'time_linear.txt')
            if IMD_flag:
                data_dict['IMD_time'] = np.loadtxt(self.device_folder+temp_folder+'IMD_time.txt')
                data_dict['IMD_date'] = np.loadtxt(self.device_folder+temp_folder+'IMD_date.txt')
                IIP3 = np.zeros(len(data_dict['current']))
                g4_IMD = np.zeros(len(data_dict['current']))
            data_dict['date_wide_span'] = np.loadtxt(self.device_folder+temp_folder+'date_wide_span.txt')
            data_dict['time_wide_span'] = np.loadtxt(self.device_folder+temp_folder+'time_wide_span.txt')                
            data_dict['date'] = np.loadtxt(self.device_folder+temp_folder+'date.txt')
            data_dict['time'] = np.loadtxt(self.device_folder+temp_folder+'time.txt')
            data_dict['Pump_power'] = np.loadtxt(self.device_folder+temp_folder+'pump_power.txt')
            
            data_dict['Pump_freq'] = np.loadtxt(self.device_folder+temp_folder+'pump_freq.txt')*1e9
            CW_freqs = np.zeros(len(data_dict['current']))
            freq_c = np.zeros(len(data_dict['current']))
            kappa_c = np.zeros(len(data_dict['current']))
            kappa_t = np.zeros(len(data_dict['current']))
            P_1dB = np.zeros(len(data_dict['current']))
            max_Gain = np.zeros(len(data_dict['current']))
            bandwidth = np.zeros(len(data_dict['current']))
            if NVR_flag:
                NVR = np.zeros(len(data_dict['current']))
            Gains = np.zeros((len(data_dict['current']),201))
            Signal_Powers_dBm = np.zeros((len(data_dict['current']),201))
            # TODO change this number to the size of signal powers aray 

            for index, current in enumerate(data_dict['current']):
                # define all data folders (dates/times)
                time_linear = float_to_DateTime( data_dict['time_linear'][index] )
                date_linear = float_to_DateTime( data_dict['date_linear'][index] )
                date = float_to_DateTime( data_dict['date'][index] )
                time = float_to_DateTime( data_dict['time'][index] )
                if IMD_flag: 
                    IMD_time = float_to_DateTime( data_dict['IMD_time'][index] )
                    IMD_date = float_to_DateTime( data_dict['IMD_date'][index] )
                # load linear fits
                freq_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('f0')) )
                kappa_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc')) )
                kappa_i = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') ))
                kappa_t[index] = kappa_c[index] + kappa_i
                # continuous wave frequency for signal
                freq_CW = hdf5[date][time]['POW']['powers'].attrs.get('freqINP')
                CW_freqs[index] = freq_CW
                # signal powers
                Signal_Powers_dBm[index] = np.asarray( hdf5[date][time]['POW'].get('powers') )        
                if 'signal_attenuation' in hdf5.keys():
                    Signal_Powers_dBm[index] = Signal_Powers_dBm[index] + attenuation(freq_CW, self.data_file, 'signal_attenuation')
                    Signal_Powers_dBm[index] -= 0.3 if self.name == 'JAMPA10_v2' else 0.0 # to account for the 4-12 GHz circulator
                # Gains
                ind_CW = np.abs(np.asarray(hdf5[date][time]['memory'].get('frequencies'))-freq_CW).argmin() + 2           # index of approximately CW_frequency in the memory trace
                a_out = np.asarray( hdf5[date][time]['memory'][meas_name].get('real') )+ 1j*np.asarray( hdf5[date][time]['memory'][meas_name].get('imag') )
                logmag_pump_off = complex_to_PLOG(a_out)[0]
                logmag_pump_off_ref= logmag_pump_off[ind_CW]
                a_out = np.asarray( hdf5[date][time]['POW'][meas_name].get('real') )+ 1j*np.asarray( hdf5[date][time]['POW'][meas_name].get('imag') )
                logmag_pump_on = complex_to_PLOG(a_out)[0]  
                logmag_pump_on_ref = logmag_pump_on[0]      # logmag at the lowest CW power where amplifier is presumably not saturated yet 
                gain_ref = logmag_pump_on_ref-logmag_pump_off_ref
                Gains[index] = logmag_pump_on - logmag_pump_on_ref + gain_ref
                max_Gain[index] = gain_ref
                # Figuring out gain profile in frequency
                a_out = np.asarray( hdf5[date][time]['memory'][meas_name].get('real') )+ 1j*np.asarray( hdf5[date][time]['memory'][meas_name].get('imag') )
                logmag_pump_off = complex_to_PLOG(a_out)[0]
                a_out = np.asarray( hdf5[date][time]['LIN'][meas_name].get('real') )+ 1j*np.asarray( hdf5[date][time]['LIN'][meas_name].get('imag') )
                logmag_pump_on = complex_to_PLOG(a_out)[0]
                gain_profile = logmag_pump_on - logmag_pump_off
                max_Gain[index] = max_Gain[index] if max_Gain[index]<10*np.log10(200/10) else gain_profile[np.argmax(gain_profile)-1]
                
                # Finding the 3 dB bandwidth relative to CW frequency   
                ind_left, ind_right = find_3dB_bandwidth(gain_profile, ind_CW)
                bandwidth[index] = np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_right]-np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_left]                
                # P_1dB point
                Saturation_power_ind = find_1dB_compression_point(Gains[index], gain_ref)
                P_1dB[index] = Signal_Powers_dBm[index][Saturation_power_ind]
                # Pump power
                if 'pump_attenuation' in hdf5.keys():
                    data_dict['Pump_power'][index] += attenuation( data_dict['Pump_freq'][index], self.data_file, 'pump_attenuation')    
                # Noise       
                if NVR_flag:
                    key1 = list(hdf5[date][time]['noise'].keys())[0]
                    key2 = list(hdf5[date][time]['noise'][key1].keys())[0]
                    noise_raise = np.asarray(hdf5[date][time]['noise'][key1][key2].get('logmag')) - np.asarray(hdf5[date][time]['noise'][key1]['memory'][key2].get('logmag'))            
#                    imax = np.argmax(noise_raise)
#                    NVR[index] = min(noise_raise[imax-2],noise_raise[imax+2])
                    NVR[index] = max(noise_raise) #noise_raise[ind_CW]       
                # IMD 
                if IMD_flag:
                    data = np.asarray(hdf5[IMD_date][IMD_time]['POW']['"CH5_IIP3_11"'].get('logmag'))
                    derivative = derivative_smooth(data, N_smooth=15)
                    Ind = np.argmin( np.abs(derivative) )
                    IIP3[index] =np.mean(data[max(Ind-5,0):min(Ind+5,len(data)-1)])
                    if 'signal_attenuation' in hdf5.keys():
                        IIP3[index] = IIP3[index] + attenuation( float(hdf5[IMD_date][IMD_time]['POW']['xs'].attrs.get('imd_f1_cw')), self.data_file, 'signal_attenuation' ) 
                    IIP_3 = dBm_to_Watts( IIP3[index] )
                    G = np.power(10, max_Gain[index]/10 )
                    g4_IMD[index] = float(kappa_c[index]**2*freq_c[index]/12/IIP_3*(2*np.pi*h)*( (np.sqrt(G)-1)/(G-1) )**3 )
            grp = hdf5['multiple_gains_sweep']
            grp.create_dataset('freq_c', data = freq_c)
            grp.create_dataset('kappa_c', data = kappa_c)
            grp.create_dataset('kappa_t', data = kappa_t)
            grp.create_dataset('CW_freq', data = CW_freqs)              # CW frequencies  
            grp.create_dataset('current', data = data_dict['current'])
            grp.create_dataset('date_linear', data = data_dict['date_linear'])
            grp.create_dataset('time_linear', data = data_dict['time_linear'])
            if IMD_flag:
                grp.create_dataset('IIP3', data = IIP3)
                grp.create_dataset('IMD_time', data = data_dict['IMD_time'])   
                grp.create_dataset('IMD_date', data = data_dict['IMD_date'])   
                grp.create_dataset('g4_IMD', data = g4_IMD)
            if NVR_flag: 
                grp.create_dataset('NVR', data = NVR)
            grp.create_dataset('P_1dB', data = P_1dB)
            grp.create_dataset('Bandwidth', data = bandwidth)
            grp.create_dataset('Gain', data = max_Gain)                 # gain at CW frequency
            grp.create_dataset('Pump_power', data = data_dict['Pump_power'])
            grp.create_dataset('Gain_array', data = Gains)
            grp.create_dataset('Signal_pow_array', data = Signal_Powers_dBm)
            grp.create_dataset('Pump_freq', data = data_dict['Pump_freq'])
            grp.create_dataset('time', data = data_dict['time'])
            grp.create_dataset('date', data = data_dict['date'])
            grp.create_dataset('time_wide_span', data = data_dict['time_wide_span'])
            grp.create_dataset('date_wide_span', data = data_dict['date_wide_span'])
        finally:    
            hdf5.close()

    
    
    
    def calibrate_noise_and_gain_using_SNT(self, sweep_name='multiple_gains_sweep'):
        
    
        hdf5 = h5py.File(self.data_file, 'r')
        try:
            CW = np.asarray(hdf5[sweep_name].get('CW_freq'))
            NVR = np.asarray(hdf5[sweep_name].get('NVR'))
            Gain = np.asarray(hdf5[sweep_name].get('Gain'))
            
            dates = np.asarray(hdf5[sweep_name].get('date')) 
            times = np.asarray(hdf5[sweep_name].get('time')) 
            
            freqs_SNT = np.asarray(hdf5['SNT_calibration'].get('freqs'))
            T_sys = np.asarray(hdf5['SNT_calibration'].get('T_sys'))
            dT_sys = np.asarray(hdf5['SNT_calibration'].get('dT_sys'))            
            G_out = np.asarray(hdf5['SNT_calibration'].get('G'))
            recvA_pow_cal = np.asarray(hdf5['SNT_calibration'].get('recvA_pow_cal'))
            recv_frq_cal = np.asarray(hdf5['SNT_calibration'].get('recv_frq_cal'))

            # use the SNT calibraion to find the system noise temperature at the CW frequency
            T_sys_2 = np.zeros(len(CW))
            dT_sys_2 = np.zeros((2,len(CW)))
            G_out_2 = np.zeros(len(CW))
            P1dB = np.zeros(len(CW))

            for j, f in enumerate(CW):
                i = np.argmin(np.abs(f-freqs_SNT))
                T_sys_2[j] = T_sys[i]
                dT_sys_2[0,j] = dT_sys[0,i]
                dT_sys_2[1,j] = dT_sys[1,i]
                G_out_2[j] = G_out[i]
            # use gain and NVR measurements to calculate the system noise temperature with the SPA on
            Gain = 10**(Gain/10.0)    
            NVR = 10**(NVR/10.0)
            T_spa = T_sys_2/Gain*(NVR-1) # this definition includes half a photon from incident noise and another half from the idler
            dT_spa = dT_sys_2/T_sys_2*T_spa
            T_spa = T_spa - h*CW/2/Boltzmann # subtracting incident noise (half a photon) 

        
            for j, f in enumerate(CW):        
                date = float_to_DateTime(dates[j])
                time = float_to_DateTime(times[j])
            
                ind_CW = np.abs(np.asarray(hdf5[date][time]['memory'].get('frequencies'))-f).argmin() #+ 2           # index of approximately CW_frequency in the memory trace
                # with pump off, as a function of frequency
                a_out = np.asarray( hdf5[date][time]['memory']['"CH1_S11_1"'].get('real') )+ 1j*np.asarray( hdf5[date][time]['memory']['"CH1_S11_1"'].get('imag') )
                logmag_pump_off = complex_to_PLOG(a_out)[0]
                logmag_pump_off_ref= logmag_pump_off[ind_CW]
                # with pump on, as a function of signal power
                a_out = np.asarray( hdf5[date][time]['POW']['"CH1_S11_1"'].get('real') )+ 1j*np.asarray( hdf5[date][time]['POW']['"CH1_S11_1"'].get('imag') )
                logmag_pump_on = complex_to_PLOG(a_out)[0]  
                logmag_pump_on_ref = logmag_pump_on[0]      # logmag at the lowest CW power where amplifier is presumably not saturated yet 
                gain_ref = logmag_pump_on_ref-logmag_pump_off_ref
                
                # gains in the signal power sweep
                Gains = logmag_pump_on - logmag_pump_on_ref + gain_ref
                Signal_Powers_dBm = np.asarray( hdf5[date][time]['POW'].get('powers') ) 

                # find the receiver A power calibration
                ind_freq = np.argmin(np.abs(f-recv_frq_cal))
                power_offset = recvA_pow_cal[ind_freq]
        
                # P_1dB point
                P1dB_ind = find_1dB_compression_point(Gains, gain_ref)
                P1dB[j] = Signal_Powers_dBm[P1dB_ind] + (logmag_pump_on[P1dB_ind] - power_offset) - Gains[P1dB_ind] - G_out_2[j]
        
        finally:
            hdf5.close()


        hdf5 = h5py.File(self.data_file)
        try:
            del hdf5['SNT_calibration']
            grp = hdf5.create_group('SNT_calibration')
            grp.create_dataset('freqs', data = freqs_SNT)
            grp.create_dataset('T_sys', data = T_sys)
            grp.create_dataset('dT_sys', data = dT_sys)
            grp.create_dataset('G', data = G_out)
            grp.create_dataset('P_1dB', data = P1dB)
            grp.create_dataset('T_SPA', data = T_spa)
            grp.create_dataset('dT_SPA', data = dT_spa)
            grp.create_dataset('recvA_pow_cal', data = recvA_pow_cal)
            grp.create_dataset('recv_frq_cal', data = recv_frq_cal)            
        finally:
            hdf5.close()
            
            