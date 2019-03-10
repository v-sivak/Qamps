# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:43:03 2018

@author: vs362
"""

import h5py
import numpy as np
from devices_functions import *


SLASH = '/' #'/' for Mac addresses or '\\' for PC


class Amplifier(object):
    
    def __init__(self, path, name):
        self.name = name
        self.device_folder = path + self.name + SLASH
        self.data_file = self.device_folder + name + u'.hdf5'
        self.load_attributes()
    
        self.name = name
        self.device_folder = path + self.name + SLASH
        self.data_file = self.device_folder + name + u'.hdf5'
        self.save_attributes()
    
    
    #   saves all device data into 'results' group of the hdf5 file, overwrites existing data!    
    def save_attributes(self):
        hdf5_file = h5py.File(self.data_file,'r+')
        try:            
            if 'results' not in hdf5_file.keys():
                hdf5_file.create_group('results')
            attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a)) ]        
            for field in attributes:
                hdf5_file['results'].attrs[field] = getattr(self,field)
        finally:
            hdf5_file.close()
            
#   loads existing 'results' group from hdf5 file to python memory
    def load_attributes(self):
        hdf5_file = h5py.File(self.data_file,'r')
        try:
            if 'results' in hdf5_file.keys():
                for attr in hdf5_file['results'].attrs:
                    setattr(self, attr, hdf5_file['results'].attrs.get(attr) )  
        finally:
            hdf5_file.close()
            
            
            
#   Adds different components to attenuation of drive/signal/pump 
    def add_component_to_line_attenuation(self, component_name, date, time, meas_name, pump=False, signal=False, drive=False):
        hdf5_file = h5py.File(self.data_file,'r+')
        try:
            if pump and 'pump_attenuation' not in hdf5_file.keys():
                hdf5_file.create_group('pump_attenuation')
            elif signal and 'signal_attenuation' not in hdf5_file.keys():
                hdf5_file.create_group('signal_attenuation')  
            elif drive and 'drive_attenuation' not in hdf5_file.keys():
                hdf5_file.create_group('drive_attenuation')
            
            if pump:
                grp = hdf5_file['pump_attenuation'].create_group(component_name)
            elif signal:
                grp = hdf5_file['signal_attenuation'].create_group(component_name)
            elif drive:
                grp = hdf5_file['drive_attenuation'].create_group(component_name)
            grp.attrs['date'] = date
            grp.attrs['time'] = time
            grp.attrs['meas_name'] = meas_name
        finally:        
            hdf5_file.close()
            
    
    
#   Fits stark shift to get g4. Before using make sure you saved drive attenuation    
    def fit_stark_shift(self, meas_name, flux_calibrated=True):

        hdf5 = h5py.File(self.data_file)
        self.stark_meas_name = meas_name
        currents = np.loadtxt(self.device_folder+'stark'+SLASH+'curr.txt')*1e-3
        dates = np.loadtxt(self.device_folder+'stark'+SLASH+'date.txt')
        times = np.loadtxt(self.device_folder+'stark'+SLASH+'time.txt')
        drive_freqs = np.loadtxt(self.device_folder+'stark'+SLASH+'drive.txt')*1e9
        if flux_calibrated: fluxes = Flux(currents,self.a,self.b)
        Kerr_Data = np.empty(0)
        K_prime_Data = np.empty(0)
        try:
            for j, current in enumerate(currents):
                date = float_to_DateTime( dates[j] )
                time = float_to_DateTime( times[j] )
                f_d = drive_freqs[j]
                
                #linear scattering data
                f_0 = np.asarray(hdf5[date][time]['fits'][meas_name].get('f0'))
                k_c = np.asarray(hdf5[date][time]['fits'][meas_name].get('kc'))
                k_i = np.asarray(hdf5[date][time]['fits'][meas_name].get('ki'))
                k_t = k_c + k_i 
                
                line_attenuation = attenuation(f_d, self.data_file, 'drive_attenuation')
        
                if 'powers' in hdf5[date][time]['LIN'].keys():
                    log_powers = np.asarray(hdf5[date][time]['LIN'].get('powers'))
                elif 'powers_swept' in hdf5[date][time]['LIN'].keys():
                    log_powers = np.asarray(hdf5[date][time]['LIN'].get('powers_swept'))
                log_powers = log_powers + line_attenuation
                powers = dBm_to_Watts(log_powers)
                
                # fit the slope, this formula takes into account some non RWA corrections
                deg = 6
                Kerr = np.polyfit(24/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((2*f_d/(f_0[0]+f_d))**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) ),f_0,deg)[deg-1]    # in Hz
#                K_prime = np.polyfit(1/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((2*f_d/(f_0[0]+f_d))**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) ),f_0,2)[0]
                Kerr_Data = np.append(Kerr_Data, Kerr)
#                K_prime_Data = np.append(K_prime_Data,K_prime)
            if 'fit_stark_shift' in hdf5.keys():
                del hdf5['fit_stark_shift']
            print(len(currents))
            grp = hdf5.create_group('fit_stark_shift')
            grp.create_dataset('current', data = currents)
            grp.create_dataset('date', data = dates)
            grp.create_dataset('time', data = times)
            grp.create_dataset('drive_freq', data = drive_freqs)
            if flux_calibrated: grp.create_dataset('flux', data = fluxes)
            grp.create_dataset('g4', data = Kerr_Data)
#            grp.create_dataset('K_prime', data = K_prime_Data)
        finally:
            hdf5.close()
            
            
    #   Loads the data about measured gain points into hdf5 file, then you can use it to plot all kinds of things
    def load_nonlin_characterization(self, meas_name):
        
        self.pumping_meas_name = meas_name
        nominal_gain_array = np.loadtxt(self.device_folder+'nonlin_characterization'+SLASH+'gains.txt')
#        nominal_gain_array = [nominal_gain_array]
        
        
        hdf5 = h5py.File(self.data_file)  
        try:
            if 'nonlin_characterization' in hdf5.keys():
                del hdf5['nonlin_characterization']
            grp = hdf5.create_group('nonlin_characterization')
            grp.create_dataset('gains', data = nominal_gain_array)
        finally:
            hdf5.close()
        
        for log_g in nominal_gain_array:
            temp_folder = 'nonlin_characterization'+SLASH+'Gain_'+str(int(log_g))+SLASH
            data_dict = {}
            data_dict['current'] = np.loadtxt(self.device_folder+temp_folder+'curr.txt')*1e-3
            data_dict['date_linear'] = np.loadtxt(self.device_folder+temp_folder+'date_linear.txt')
            data_dict['time_linear'] = np.loadtxt(self.device_folder+temp_folder+'time_linear.txt')
            data_dict['IMD_time'] = np.loadtxt(self.device_folder+temp_folder+'IMD_time.txt')
            data_dict['IMD_date'] = np.loadtxt(self.device_folder+temp_folder+'IMD_date.txt')
            if log_g:
                data_dict['date'] = np.loadtxt(self.device_folder+temp_folder+'date.txt')
                data_dict['time'] = np.loadtxt(self.device_folder+temp_folder+'time.txt')
                data_dict['Pump_power'] = np.loadtxt(self.device_folder+temp_folder+'pump_power.txt')
                data_dict['Pump_freq'] = np.loadtxt(self.device_folder+temp_folder+'pump_freq.txt')*1e9
            g4_IMD = np.zeros(len(data_dict['current']))
            CW_freqs = np.zeros(len(data_dict['current']))
            freq_c = np.zeros(len(data_dict['current']))
            kappa_c = np.zeros(len(data_dict['current']))
            kappa_t = np.zeros(len(data_dict['current']))
            P_1dB = np.zeros(len(data_dict['current']))
            IIP3 = np.zeros(len(data_dict['current']))
            max_Gain = np.zeros(len(data_dict['current']))
            bandwidth = np.zeros(len(data_dict['current']))
            NVR = np.zeros(len(data_dict['current']))
            NVR_spike = np.zeros(len(data_dict['current']))
            Gains = np.zeros((len(data_dict['current']),201))
            Signal_Powers_dBm = np.zeros((len(data_dict['current']),201))
            # TODO change this number to the size of signal powers aray
            hdf5 = h5py.File(self.data_file)  
            try:
                for index, current in enumerate(data_dict['current']):
                    # define all data folders (dates/times)
                    IMD_time = float_to_DateTime( data_dict['IMD_time'][index] )
                    IMD_date = float_to_DateTime( data_dict['IMD_date'][index] )
                    time_linear = float_to_DateTime( data_dict['time_linear'][index] )
                    date_linear = float_to_DateTime( data_dict['date_linear'][index] )
                    if log_g:
                        date = float_to_DateTime( data_dict['date'][index] )
                        time = float_to_DateTime( data_dict['time'][index] )
                    
                    # load linear fits
                    freq_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('f0')) )
                    kappa_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc')) )
                    kappa_i = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') ))
                    kappa_t[index] = kappa_c[index] + kappa_i
                      
                    # continuous wave frequency for signal
                    if log_g:
                        freq_CW = hdf5[date][time]['POW']['powers'].attrs.get('freqINP')
                    else:
                        freq_CW = freq_c[index] + 100e6
                    CW_freqs[index] = freq_CW
                
                    
                    if log_g:
                        # signal powers
                        Signal_Powers_dBm[index] = np.asarray( hdf5[date][time]['POW'].get('powers') )        
                        Signal_Powers_dBm[index] = Signal_Powers_dBm[index] + attenuation(freq_CW, self.data_file, 'signal_attenuation')            
                          
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
    
                        # Finding the 3 dB bandwidth relative to CW frequency   
                        ind_left, ind_right = find_3dB_bandwidth(gain_profile, ind_CW)
                        bandwidth[index] = np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_right]-np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_left]

                    
                        # P_1dB and IIP3 points
                        Saturation_power_ind = find_1dB_compression_point(Gains[index], gain_ref)
                        P_1dB[index] = Signal_Powers_dBm[index][Saturation_power_ind]
                        
                        if 'noise' in hdf5[date][time]['fits'].keys():
                            noise_flag = True
                            key1 = list(hdf5[date][time]['fits']['noise'].keys())[0]
                            key2 = list(hdf5[date][time]['fits']['noise'][key1].keys())[0]
                            
                            noise_raise = np.asarray(hdf5[date][time]['noise'][key1][key2].get('logmag')) - np.asarray(hdf5[date][time]['noise'][key1]['memory'][key2].get('logmag'))
                            NVR[index] = noise_raise[ind_CW]
                            NVR_spike[index] = max(noise_raise) - NVR[index]
                        else:
                            noise_flag = False

                  
                    data = np.asarray(hdf5[IMD_date][IMD_time]['POW']['"CH5_IIP3_11"'].get('logmag'))

                    derivative = derivative_smooth(data, N_smooth=15)
                    Ind = np.argmin( np.abs(derivative) )
                    
                    IIP3[index] =np.mean(data[max(Ind-5,0):min(Ind+5,len(data)-1)])
                    IIP3[index] = IIP3[index] + attenuation( float(hdf5[IMD_date][IMD_time]['POW']['xs'].attrs.get('imd_f1_cw')), self.data_file, 'signal_attenuation' ) 
                    IIP_3 = dBm_to_Watts( IIP3[index] )
                    if log_g:
                        G = np.power(10, max_Gain[index]/10 )
                        g4_IMD[index] = float(kappa_c[index]**2*freq_c[index]/12/IIP_3*(2*np.pi*h)*( (np.sqrt(G)-1)/(G-1) )**3 )
                    else: 
                        G=1
                        g4_IMD[index] = float(kappa_c[index]**2*freq_c[index]/12/IIP_3*(2*np.pi*h)/8 )     
                         
                if 'pumping_data_'+str(int(log_g)) in hdf5.keys():
                    del hdf5['pumping_data_'+str(int(log_g))]
                grp = hdf5.create_group('pumping_data_'+str(int(log_g)))
                grp.create_dataset('freq_c', data = freq_c)
                grp.create_dataset('kappa_c', data = kappa_c)
                grp.create_dataset('kappa_t', data = kappa_t)
                grp.create_dataset('CW_freq', data = CW_freqs)              # CW frequencies  
                grp.create_dataset('current', data = data_dict['current'])
                grp.create_dataset('date_linear', data = data_dict['date_linear'])
                grp.create_dataset('time_linear', data = data_dict['time_linear'])
                grp.create_dataset('IIP3', data = IIP3)
                grp.create_dataset('IMD_time', data = data_dict['IMD_time'])   
                grp.create_dataset('IMD_date', data = data_dict['IMD_date'])   
                grp.create_dataset('g4_IMD', data = g4_IMD)
                if log_g:
                    if noise_flag: 
                        grp.create_dataset('NVR', data = NVR)
                        grp.create_dataset('NVR_spike', data = NVR_spike)
                    grp.create_dataset('P_1dB', data = P_1dB)
                    grp.create_dataset('Bandwidth', data = bandwidth)
                    grp.create_dataset('Gain', data = max_Gain)                 # gain at CW frequency
                    grp.create_dataset('Pump_power', data = data_dict['Pump_power'] + attenuation( data_dict['Pump_freq'], self.data_file, 'pump_attenuation') )
                    grp.create_dataset('Gain_array', data = Gains)
                    grp.create_dataset('Signal_pow_array', data = Signal_Powers_dBm)
                    grp.create_dataset('Pump_freq', data = data_dict['Pump_freq'])
                    grp.create_dataset('time', data = data_dict['time'])
                    grp.create_dataset('date', data = data_dict['date'])
            finally:    
                hdf5.close()
                
    
    
    def load_pump_power_sweep(self):

        new_folder = self.device_folder+'pump_power_sweep'+SLASH
        gains = np.loadtxt(new_folder+'gain_sweep.txt')
        currents = np.loadtxt(new_folder+'curr.txt')
        pump_powers = np.loadtxt(new_folder+'pump_powers.txt')

        attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a)) ]
        flux_calibrated = True if ('a' in attributes) and ('b' in attributes) else False
        
        hdf5 = h5py.File(self.data_file)  
        try:
            if 'pump_power_sweep' in hdf5.keys():
                del hdf5['pump_power_sweep']
            grp = hdf5.create_group('pump_power_sweep')
            grp.create_dataset('current', data = currents)
            grp.create_dataset('pump_powers', data = pump_powers)
            grp.create_dataset('gains', data = gains)
            if flux_calibrated:
                flux = Flux(currents,self.a,self.b)
                grp.create_dataset('flux', data = flux)
        finally:
            hdf5.close()        
        
        
    
    def add_two_tone_spectroscopy_data(self):

        new_path = self.device_folder + SLASH + 'two_tone_spectroscopy' + SLASH
        
        currents = np.loadtxt(new_path + 'current.txt')
        ag_freqs = np.loadtxt(new_path + 'ag_freqs.txt')
        real = np.loadtxt(new_path + 'real.txt')
        imag = np.loadtxt(new_path + 'imag.txt')
        ag_powers = np.loadtxt(new_path + 'ag_pows.txt')
        res_freqs = np.loadtxt(new_path + 'res_freq.txt')  
        # normalize the phase to have 0 for no response 
        data = np.vectorize(complex)(real, imag)
        z = data / np.abs(data)
        for ind,I in enumerate(currents):
            z0 = z[ind][0]
            z[ind] = z[ind]/z0
        z = np.angle(z,deg=True)
        
        hdf5 = h5py.File(self.data_file)  
        try:
            if 'two_tone_spectroscopy_data' in hdf5.keys():
                del hdf5['two_tone_spectroscopy_data']
            grp = hdf5.create_group('two_tone_spectroscopy_data')
            grp.create_dataset('currents', data = currents)
            grp.create_dataset('ag_freqs', data = ag_freqs)
            grp.create_dataset('real', data = real)
            grp.create_dataset('imag', data = imag)
            grp.create_dataset('ag_powers', data = ag_powers)
            grp.create_dataset('res_freqs', data = res_freqs)
            grp.create_dataset('phase_normalized', data = z)
        finally:
            hdf5.close()
            
        self.plot_two_tone_spectroscopy()
            
            
            
    def plot_two_tone_spectroscopy(self):
        
        hdf5 = h5py.File(self.data_file)
        try:
            z = np.asarray(hdf5['two_tone_spectroscopy_data'].get('phase_normalized'))
            ag_freqs = np.asarray(hdf5['two_tone_spectroscopy_data'].get('ag_freqs'))
            currents = np.asarray(hdf5['two_tone_spectroscopy_data'].get('currents'))
        finally:
            hdf5.close()
        # colorplot the two-tone spectroscopy data
        dpi = 150
        fig1, ax1 = plt.subplots(1,dpi=dpi)
        p1 = ax1.pcolormesh(currents*1e3, ag_freqs*1e-9, np.transpose(z), cmap='RdBu',vmin=-180,vmax=180)
        fig1.colorbar(p1,ax=ax1, label=r'$\rm Phase \,(deg)$')
        ax1.set_ylabel(r'${\rm Frequency \,(GHz)}$')
        ax1.set_xlabel(r'${\rm Current\,(mA)}$')
        fig1.savefig(self.device_folder + 'phase_colormap.png',dpi=dpi)
        
        
        
        
        
        
    def load_pump_params_sweep(self, meas_name, sweep_name='pump_params_sweep', IMD_flag=False, NVR_flag=True):
        
        
        self.pumping_meas_name = meas_name
        nominal_gain_array = np.loadtxt(self.device_folder+sweep_name+SLASH+'gains.txt')        
        
        nominal_gain_array = [nominal_gain_array]
        
        hdf5 = h5py.File(self.data_file)
        try:
            if sweep_name in hdf5.keys():
                del hdf5[sweep_name]
            grp = hdf5.create_group(sweep_name)
            grp.create_dataset('gains', data = nominal_gain_array)
        
            for log_g in nominal_gain_array:
                temp_folder = sweep_name+SLASH+'Gain_'+str(int(log_g))+SLASH
                data_dict = {}
                
                date_linear = float_to_DateTime(np.loadtxt(self.device_folder+temp_folder+'date_linear.txt'))
                time_linear = float_to_DateTime(np.loadtxt(self.device_folder+temp_folder+'time_linear.txt'))
                current = np.loadtxt(self.device_folder+temp_folder+'current.txt')

    
                # load linear fits
                freq_c = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('f0')) )
                kappa_c = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc')) )
                kappa_i = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') ))
                kappa_t = kappa_c + kappa_i
                
               
                data_dict['date'] = np.loadtxt(self.device_folder+temp_folder+'date.txt')
                data_dict['time'] = np.loadtxt(self.device_folder+temp_folder+'time.txt')
                if IMD_flag:
                    data_dict['IMD_time'] = np.loadtxt(self.device_folder+temp_folder+'IMD_time.txt')
                    data_dict['IMD_date'] = np.loadtxt(self.device_folder+temp_folder+'IMD_date.txt')

                data_dict['Pump_power'] = np.loadtxt(self.device_folder+temp_folder+'pump_power.txt')
                data_dict['Pump_freq'] = np.loadtxt(self.device_folder+temp_folder+'pump_freq.txt')*1e9
                CW_freqs = np.zeros(len(data_dict['Pump_freq']))
                P_1dB = np.zeros(len(data_dict['Pump_freq']))
                NVR = np.zeros(len(data_dict['Pump_freq']))
                period_doubling_peak = np.zeros(len(data_dict['Pump_freq']))
                IIP3 = np.zeros(len(data_dict['Pump_freq']))
                max_Gain = np.zeros(len(data_dict['Pump_freq']))
                bandwidth = np.zeros(len(data_dict['Pump_freq']))
                Gains = np.zeros((len(data_dict['Pump_freq']),201))
                Signal_Powers_dBm = np.zeros((len(data_dict['Pump_freq']),201))
                # TODO change this number to the size of signal powers array
    
                for index, pump_freq in enumerate(data_dict['Pump_freq']):       
                    
                    grp.attrs['freq_c'] = freq_c
                    grp.attrs['kappa_i'] = kappa_i
                    grp.attrs['kappa_c'] = kappa_c
                    grp.attrs['kappa_t'] = kappa_t
                    grp.attrs['current'] = current                    
                    
                    
                    
                    
                    # define all data folders (dates/times)
                    date = float_to_DateTime( data_dict['date'][index] )
                    time = float_to_DateTime( data_dict['time'][index] )
                    
                    if IMD_flag:
                        IMD_time = float_to_DateTime( data_dict['IMD_time'][index] )
                        IMD_date = float_to_DateTime( data_dict['IMD_date'][index] )                    
                      
                    # continuous wave frequency for signal
                    freq_CW = hdf5[date][time]['POW']['powers'].attrs.get('freqINP')
                    CW_freqs[index] = freq_CW
                
                    # signal powers
                    Signal_Powers_dBm[index] = np.asarray( hdf5[date][time]['POW'].get('powers') )        
                    Signal_Powers_dBm[index] = Signal_Powers_dBm[index] + attenuation(freq_CW, self.data_file, 'signal_attenuation')            
                      
                    # Gains
                    ind_CW = np.abs(np.asarray(hdf5[date][time]['memory'].get('frequencies'))-freq_CW).argmin() + 2            # index of approximately CW_frequency in the memory trace
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
    
                    # Finding the 3 dB bandwidth relative to CW frequency   
                    ind_left, ind_right = find_3dB_bandwidth(gain_profile, ind_CW)
                    bandwidth[index] = np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_right]-np.asarray(hdf5[date][time]['memory'].get('frequencies'))[ind_left]
    
                    # P_1dB and IIP3 points
                    Saturation_power_ind = find_1dB_compression_point(Gains[index], gain_ref)
                    P_1dB[index] = Signal_Powers_dBm[index][Saturation_power_ind]
                    if IMD_flag:
                        data = np.asarray(hdf5[IMD_date][IMD_time]['POW']['"CH5_IIP3_11"'].get('logmag'))
                        derivative = derivative_smooth(data, N_smooth=15)
                        Ind = np.argmin( np.abs(derivative) )
                        
                        IIP3[index] =np.mean(data[max(Ind-10,0):min(Ind+10,len(data)-1)])
                        IIP3[index] = IIP3[index] + attenuation( float(hdf5[IMD_date][IMD_time]['POW']['xs'].attrs.get('imd_f1_cw')), self.data_file, 'signal_attenuation' ) 
                        IIP_3 = dBm_to_Watts( IIP3[index] )

                    if NVR_flag:
                        # NVR
                        key1 = list(hdf5[date][time]['fits']['noise'].keys())[0]
                        key2 = list(hdf5[date][time]['fits']['noise'][key1].keys())[0]
    #                        NVR[index] = np.asarray(hdf5[date][time]['fits']['noise'][key1][key2].get('nvr'))
                        noise_raise = np.asarray(hdf5[date][time]['noise'][key1][key2].get('logmag')) - np.asarray(hdf5[date][time]['noise'][key1]['memory'][key2].get('logmag'))
                        ind_CW = np.abs(np.asarray(hdf5[date][time]['noise'][key1].get('frequencies'))-freq_CW).argmin() + 2
                        NVR[index] = noise_raise[ind_CW]
                        period_doubling_peak[index] = max(noise_raise) - NVR[index]
#                        period_doubling_peak[index] = noise_raise[ind_CW+3] - NVR[index]

                
#                if 'pump_params_sweep_'+str(int(log_g)) in grp.keys():
#                    del grp['pump_params_sweep_'+str(int(log_g))]
                grp1 = grp.create_group('pump_params_sweep_'+str(int(log_g)))
                
                grp1.create_dataset('CW_freq', data = CW_freqs)              # CW frequencies  
                grp1.create_dataset('P_1dB', data = P_1dB)
                grp1.create_dataset('Bandwidth', data = bandwidth)
                grp1.create_dataset('Gain', data = max_Gain)                 # gain at CW frequency
                grp1.create_dataset('Pump_power', data = data_dict['Pump_power'] + attenuation( data_dict['Pump_freq'], self.data_file, 'pump_attenuation') )
                grp1.create_dataset('Gain_array', data = Gains)
                grp1.create_dataset('Signal_pow_array', data = Signal_Powers_dBm)
                grp1.create_dataset('Pump_freq', data = data_dict['Pump_freq'])
                grp1.create_dataset('time', data = data_dict['time'])
                grp1.create_dataset('date', data = data_dict['date'])
                if NVR_flag:
                    grp1.create_dataset('NVR', data = NVR)
                    grp1.create_dataset('period_doubling_peak', data = period_doubling_peak)
                if IMD_flag:
                    grp1.create_dataset('IIP3', data = IIP3)
                    grp1.create_dataset('IMD_time', data = data_dict['IMD_time'])   
                    grp1.create_dataset('IMD_date', data = data_dict['IMD_date'])   
        finally:
            hdf5.close()
    



    def load_pump_params_sweep_dumb(self, meas_name, sweep_name='pump_params_sweep', IMD=False):
        
        
        self.pumping_meas_name = meas_name
        nominal_gain_array = np.loadtxt(self.device_folder+sweep_name+SLASH+'gains.txt')        
        
        nominal_gain_array = [nominal_gain_array]
        
        hdf5 = h5py.File(self.data_file)
        try:
            if sweep_name in hdf5.keys():
                del hdf5[sweep_name]
            grp = hdf5.create_group(sweep_name)
            grp.create_dataset('gains', data = nominal_gain_array)
        
            for log_g in nominal_gain_array:
                temp_folder = sweep_name+SLASH+'Gain_'+str(int(log_g))+SLASH
                data_dict = {}
                
                date_linear = float_to_DateTime(np.loadtxt(self.device_folder+temp_folder+'date_linear.txt'))
                time_linear = float_to_DateTime(np.loadtxt(self.device_folder+temp_folder+'time_linear.txt'))
                current = np.loadtxt(self.device_folder+temp_folder+'current.txt')

                # load linear fits
                grp.attrs['freq_c'] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('f0')) )
                grp.attrs['kappa_i'] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') ))
                grp.attrs['kappa_c'] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc')) )
                grp.attrs['kappa_t'] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') )) + float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc') ))
                grp.attrs['current'] = current
                

                data_dict['Pump_power'] = np.loadtxt(self.device_folder+temp_folder+'pump_power.txt')
                data_dict['Pump_freq'] = np.loadtxt(self.device_folder+temp_folder+'pump_freq.txt')*1e9                
                

                Signal_Powers_dBm = np.zeros((len(data_dict['Pump_freq']),201))                
                for index, pump_freq in enumerate(data_dict['Pump_freq']):  
                    Signal_Powers_dBm[index] = np.loadtxt(self.device_folder + sweep_name + SLASH + 'signal_powers.txt')
                    Signal_Powers_dBm[index] = Signal_Powers_dBm[index] + attenuation(pump_freq/2, self.data_file, 'signal_attenuation')            

                Gains = np.zeros((len(data_dict['Pump_freq']),201))
                logmag_on = np.loadtxt(self.device_folder+temp_folder+'logmag_on.txt')  
                logmag_off = np.loadtxt(self.device_folder+temp_folder+'logmag_off.txt')
                
                for s in range(len(logmag_off)):
                    Gains[s] = logmag_on[s] - logmag_off[s]
    
    
                grp1 = grp.create_group('pump_params_sweep_'+str(int(log_g)))
                grp1.create_dataset('Pump_power', data = data_dict['Pump_power'] + attenuation( data_dict['Pump_freq'], self.data_file, 'pump_attenuation') )
                grp1.create_dataset('Gain_array', data = Gains)
                grp1.create_dataset('Signal_pow_array', data = Signal_Powers_dBm)
                grp1.create_dataset('Pump_freq', data = data_dict['Pump_freq'])
        finally:
            hdf5.close()