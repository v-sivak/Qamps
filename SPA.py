# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:14:52 2018

@author: Vladimir
"""

from __future__ import division
import h5py
import numpy as np
from devices_functions import *
from Amplifier import Amplifier 


SLASH = '/' #'/' for Mac addresses or '\\' for Windows


class SPA(Amplifier):
    

#  fits flux sweep to LC model, saves the fitted parameters and arrays of data used for the fitting
    def fit_flux_sweep_LC_model(self, date, time, meas_name, L_j, M, a_guess, b_guess, alpha_guess, p_j_guess):
        self.L_j = L_j
        self.E_j =  1/self.L_j*(Phi_0)**2 / h   # in Hz   
        self.M = M
        self.flux_sweep_date = date
        self.flux_sweep_time = time
        self.flux_sweep_meas_name = meas_name
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file[date][time]['LIN'].get('currents'))
            f0_exp_fit = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('f0'))
            kc_exp_fit = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('kc'))
            #fixes divergent exp fits 
            frequencies = np.asarray(hdf5_file[date][time]['LIN'].get('frequencies'))
            freq_max = frequencies[ len(frequencies)-1 ]
            freq_min = frequencies[ 0 ]
            currents, f0_exp_fit, kc_exp_fit = fit_flux_sweep_helper(currents, f0_exp_fit, kc_exp_fit, freq_max, freq_min)            
            self.a, self.b, self.alpha, self.p_j, self.f0, self.C = fit_flux_sweep_LC(currents, f0_exp_fit, a=a_guess, b=b_guess, alpha=alpha_guess, p_j=p_j_guess, M=M, L_j=L_j)            
            if 'fit_flux_sweep' in hdf5_file.keys():
                del hdf5_file['fit_flux_sweep']
            grp = hdf5_file.create_group('fit_flux_sweep')
            grp.create_dataset('currents', data = currents)
            grp.create_dataset('f0_exp_fit', data = f0_exp_fit)
            grp.create_dataset('kc_exp_fit', data = kc_exp_fit)
        finally:
            hdf5_file.close()
        self.save_attributes()

#   fits the flux sweep to distributed model, fit alpha2 and f0_distributed
    def fit_flux_sweep_distributed_model(self, Z_c, mu, alpha2_guess, f0_distributed_guess, C_coupling_guess, N_iter=4):
        self.Z_c = Z_c
        self.mu = mu
        self.gamma = 2*self.Z_c/self.L_j
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file['fit_flux_sweep']['currents'])
            f0_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['f0_exp_fit'])
            kc_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['kc_exp_fit'])            
            self.alpha2, self.f0_distributed, self.C_coupling = fit_flux_sweep_distributed(currents, f0_exp_fit, kc_exp_fit, self.a, self.b, self.Z_c, self.gamma, self.M, alpha2_guess, f0_distributed_guess, C_coupling_guess, mu=self.mu, N_iter=N_iter)
        finally:
            hdf5_file.close()
        self.save_attributes()



#   fits the flux sweep to distributed model, fit alpha2, f0_distributed and Z_c
    def fit_flux_sweep_distributed_model_2(self, Z_c_guess, mu, alpha2_guess, f0_distributed_guess, C_coupling_guess, N_iter=4):
        self.mu = mu
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file['fit_flux_sweep']['currents'])
            f0_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['f0_exp_fit'])
            kc_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['kc_exp_fit'])
            self.alpha2, self.f0_distributed, self.Z_c, self.C_coupling = fit_flux_sweep_distributed_2(currents, f0_exp_fit, kc_exp_fit, self.a, self.b, Z_c_guess, self.gamma, self.M, self.L_j, alpha2_guess, f0_distributed_guess, C_coupling_guess, mu=self.mu, N_iter=N_iter)
            self.gamma = 2*self.Z_c/self.L_j
        finally:
            hdf5_file.close()
        self.save_attributes()

#   fits the flux sweep to distributed model for directly coupled resonators
    def fit_flux_sweep_distributed_model_3(self, Z_c, mu, alpha2_guess, f0_distributed_guess, Res_guess):
        self.direct_coupling = True
        self.C_coupling = 0
        self.mu = mu
        self.Z_c = Z_c
        self.gamma = 2*Z_c/self.L_j
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file['fit_flux_sweep']['currents'])
            f0_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['f0_exp_fit'])
            kc_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['kc_exp_fit'])
            self.alpha2, self.f0_distributed, self.Res  = fit_flux_sweep_distributed_3(currents, f0_exp_fit, kc_exp_fit, self.a, self.b, self.Z_c, self.gamma, self.M, self.L_j, alpha2_guess, f0_distributed_guess, Res_guess, mu=self.mu)
        finally:
            hdf5_file.close()
        self.save_attributes()

#   fits the flux sweep to distributed model, only fit f0_distributed
    def fit_flux_sweep_distributed_model_4(self, Z_c, mu, alpha2, f0_distributed_guess, C_coupling_guess, N_iter=4):
        self.alpha2 = alpha2
        self.Z_c = Z_c
        self.mu = mu
        self.gamma = 2*self.Z_c/self.L_j
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file['fit_flux_sweep']['currents'])
            f0_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['f0_exp_fit'])
            kc_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['kc_exp_fit'])            
            self.f0_distributed, self.C_coupling = fit_flux_sweep_distributed_4(currents, f0_exp_fit, kc_exp_fit, self.a, self.b, self.Z_c, self.gamma, self.M, self.alpha2, f0_distributed_guess, C_coupling_guess, mu=self.mu, N_iter=N_iter)
        finally:
            hdf5_file.close()
        self.save_attributes()


#   fits the flux sweep to distributed model, fit L_j and f0_distributed
    def fit_flux_sweep_distributed_model_5(self, Z_c, mu, alpha2, f0_distributed_guess, C_coupling_guess, N_iter=4):
        self.Z_c = Z_c
        self.alpha2 = alpha2
        self.mu = mu
        gamma_guess = 2*self.Z_c/self.L_j
        hdf5_file = h5py.File(self.data_file,'r+')
        try: 
            currents = np.asarray(hdf5_file['fit_flux_sweep']['currents'])
            f0_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['f0_exp_fit'])
            kc_exp_fit = np.asarray(hdf5_file['fit_flux_sweep']['kc_exp_fit'])            
            self.gamma, self.f0_distributed, self.C_coupling = fit_flux_sweep_distributed_5(currents, f0_exp_fit, kc_exp_fit, self.a, self.b, self.Z_c, gamma_guess, self.M, alpha2, f0_distributed_guess, C_coupling_guess, mu=self.mu, N_iter=N_iter)
            self.L_j = 2*self.Z_c/self.gamma
        finally:
            hdf5_file.close()
        self.save_attributes()

          
#   ------------------ Some functions from distributed model ---------            
    def g4_distributed(self, f, num_mode=1):
        return g4_distributed(f, self.a, self.b, self.alpha2, self.f0_distributed, self.M, self.L_j, self.Z_c, num_mode=num_mode, mu=self.mu)

    def g3_distributed(self, f, num_mode=1):
        return g3_distributed(f, self.a, self.b, self.alpha2, self.f0_distributed, self.M, self.L_j, self.Z_c, num_mode=num_mode, mu=self.mu)

    def participation(self, f, num_mode=1):
        return participation(f, self.a, self.b, self.alpha2, self.f0_distributed, self.Z_c, self.M, self.L_j, mode=num_mode, mu=self.mu)

    def Freq(self, f, num_mode=1):
        return freq_distributed_with_coupling(Current(f,self.a,self.b), self.a, self.b, self.alpha2, self.f0_distributed, self.Z_c, self.C_coupling, self.M, self.gamma, mode=num_mode, mu=self.mu)

    def capacitance(self, f, num_mode=1):
        return capacitance(f, self.a, self.b, self.alpha2, self.f0_distributed, self.Z_c, self.M, self.gamma, num_mode_cap=num_mode, mu=self.mu)

    def kappa(self, f, num_mode=1):
        return kappa(f, self.a, self.b, self.alpha2, self.f0_distributed, self.Z_c, self.C_coupling, self.M, self.gamma, mode=num_mode, mu=self.mu)
#        return kappa(f, self.a, self.b, self.alpha2, self.f0_distributed, self.Z_c, self.C_coupling, self.M, self.gamma, mode=num_mode, mu=self.mu, Res=self.Res, direct=self.direct_coupling)

#   -----------------------some functions from LC model --------------------

    def g4_LC(self, f):
        return c4_eff(f, self.p_j, self.alpha, self.M)/c2_eff(f, self.p_j, self.alpha, self.M)/4/24/R_Q/self.C

    def g4_corrected_LC(self, f):                                             # definition that people usually use
        return g4_corrected_LC(f, self.a, self.b, self.p_j, self.alpha, self.M, self.C, self.f0)
    
    def chi_LC(self, f):                                             # definition that people usually use
        return 12*g4_LC(f, self.p_j, self.alpha, self.M, self.C)
    
    def g3_LC(self, f):
        return c3_eff(f, self.p_j, self.alpha, self.M)/c2_eff(f, self.p_j, self.alpha, self.M)/12/np.sqrt(2)*np.sqrt(f_theory_LC(Current(f,self.a,self.b), self.a, self.b, self.alpha, self.p_j, self.f0, self.M)/self.C/R_Q)
    
    def g6_LC(self,f):
        return g6_LC(f, self.a, self.b, self.p_j, self.alpha, self.M, self.C, self.f0)

#    def g4_corrected_LC(self, f):
#        return g4_LC(f, self.p_j, self.alpha, self.M, self.C)-5*(g3_LC(f, self.p_j, self.alpha, self.M, self.C))**2/f_theory_LC(Current(f,self.a,self.b), self.a, self.b, self.alpha, self.p_j, self.f0, self.M)

    def f_theory_LC(self, f):
        return f_theory_LC(Current(f,self.a,self.b), self.a, self.b, self.alpha, self.p_j, self.f0, self.M)

    
    def g4_BBQ_LC(self, f):
        return c4_eff_BBQ(f, self.p_j, self.alpha, self.M)/c2_eff(f, self.p_j, self.alpha, self.M)/4/24/R_Q/self.C

    def c2_eff(self, f):
        return c2_eff(f, self.p_j, self.alpha, self.M)
    
    def c3_eff(self, f):
        return c3_eff(f, self.p_j, self.alpha, self.M)
    
    def c4_eff(self, f):
        return c4_eff(f, self.p_j, self.alpha, self.M)
        
    def c5_eff(self, f):
        return c5_eff(f, self.p_j, self.alpha, self.M)

    def c6_eff(self, f):
        return c6_eff(f, self.p_j, self.alpha, self.M)



#   Loads the data about measured gain points into hdf5 file, then you can use it to plot all kinds of things
    def load_pumping_data(self, meas_name, IMD = False):
        
        self.pumping_meas_name = meas_name
        data_dict = {}
        data_dict['current'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'curr.txt')*1e-3
        data_dict['date'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'date.txt')
        data_dict['date_linear'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'date_linear.txt')
        data_dict['time'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'time.txt')
        data_dict['time_linear'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'time_linear.txt')
        data_dict['Pump_power'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'pump_power.txt')
        data_dict['Pump_freq'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'pump_freq.txt')*1e9
        if IMD:
            data_dict['IMD_time'] = np.loadtxt(self.device_folder+'pumping'+SLASH+'IMD_time.txt')
            g4_IMD = np.zeros(len(data_dict['current']))
        CW_freqs = np.zeros(len(data_dict['current']))
        freq_c = np.zeros(len(data_dict['current']))
        kappa_c = np.zeros(len(data_dict['current']))
        kappa_t = np.zeros(len(data_dict['current']))
        P_1dB = np.zeros(len(data_dict['current']))
        IIP3 = np.zeros(len(data_dict['current']))
        max_Gain = np.zeros(len(data_dict['current']))
        bandwidth = np.zeros(len(data_dict['current']))
        Gains = np.zeros((len(data_dict['current']),201))
        Signal_Powers_dBm = np.zeros((len(data_dict['current']),201))
# TODO change this number to the size of signal powers aray
        hdf5 = h5py.File(self.data_file)  
        
        try:
            for index, current in enumerate(data_dict['current']):            
                # define all data folders (dates/times)
                date = float_to_DateTime( data_dict['date'][index] )
                time = float_to_DateTime( data_dict['time'][index] )
                if IMD:
                    IMD_time = float_to_DateTime( data_dict['IMD_time'][index] )
                time_linear = float_to_DateTime( data_dict['time_linear'][index] )
                date_linear = float_to_DateTime( data_dict['date_linear'][index] )
                
                # load linear fits
                freq_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('f0')) )
                kappa_c[index] = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('kc')) )
                kappa_i = float( np.asarray(hdf5[date_linear][time_linear]['LIN'][meas_name]['fits'].get('ki') ))
                kappa_t[index] = kappa_c[index] + kappa_i
                  
                # continuous wave frequency for signal
                print(current*1e3)
                freq_CW = hdf5[date][time]['POW']['powers'].attrs.get('freqINP')
                CW_freqs[index] = freq_CW
            
               # signal powers, there was some bug with SPA13, so this also fixes that 
                if self.name == 'SPA13':
                    if index>=14 and index<=27:
                        Signal_Powers_dBm[index] = (60-30)/(80-30)*np.asarray( hdf5[date][time]['POW'].get('powers') )+(80*(60-30)/(80-30)-60)
                    elif index>=28 and index<=36:
                        Signal_Powers_dBm[index] = (60-20)/(80-20)*np.asarray( hdf5[date][time]['POW'].get('powers') )+(80*(60-20)/(80-20)-60)
                    else: 
                        Signal_Powers_dBm[index] = np.asarray( hdf5[date][time]['POW'].get('powers') )
                else:
                    Signal_Powers_dBm[index] = np.asarray( hdf5[date][time]['POW'].get('powers') )        
                Signal_Powers_dBm[index] = Signal_Powers_dBm[index] + attenuation(freq_CW, self.data_file, 'signal_attenuation')            
                      
                # Gains
                ind_CW = np.abs(np.asarray(hdf5[date][time]['memory'].get('frequencies'))-freq_CW).argmin()            # index of approximately CW_frequency in the memory trace
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
                if IMD:
                    IIP3[index] = np.mean( np.asarray(hdf5[date][IMD_time]['POW']['"CH5_IIP3_11"'].get('logmag'))[33:66] )   #[0:33] on the ones with low dynamic range
                    IIP3[index] = IIP3[index] + attenuation( float(hdf5[date][IMD_time]['POW']['xs'].attrs.get('imd_f1_cw')), self.data_file, 'signal_attenuation' ) 
                    IIP_3 = dBm_to_Watts( IIP3[index] )
                    G = np.power(10, max_Gain[index]/10 )
                    g4_IMD[index] = float(kappa_c[index]**2*freq_c[index]/12/IIP_3*(2*np.pi*h)*( (np.sqrt(G)-1)/(G-1) )**3)     
                
                # TODO: change this 33:66 thing to somesthing more sensible
                # Are you sure you need same signal attenuation in IMD experiment???? 
            
            if 'pumping_data' in hdf5.keys():
                del hdf5['pumping_data']
            grp = hdf5.create_group('pumping_data')
            grp.create_dataset('freq_c', data = freq_c)
            grp.create_dataset('kappa_c', data = kappa_c)
            grp.create_dataset('kappa_t', data = kappa_t)
            grp.create_dataset('P_1dB', data = P_1dB)
            grp.create_dataset('Bandwidth', data = bandwidth)
            grp.create_dataset('Gain', data = max_Gain)                 # gain at CW frequency
            grp.create_dataset('CW_freq', data = CW_freqs)      # CW frequencies 
            grp.create_dataset('Pump_power', data = data_dict['Pump_power'] + attenuation( data_dict['Pump_freq'], self.data_file, 'pump_attenuation') )
            grp.create_dataset('Gain_array', data = Gains)
            grp.create_dataset('Signal_pow_array', data = Signal_Powers_dBm)
            grp.create_dataset('Pump_freq', data = data_dict['Pump_freq'])
            grp.create_dataset('current', data = data_dict['current'])
            grp.create_dataset('date', data = data_dict['date'])
            grp.create_dataset('time', data = data_dict['time'])
            grp.create_dataset('date_linear', data = data_dict['date_linear'])
            grp.create_dataset('time_linear', data = data_dict['time_linear'])
            if IMD:
                grp.create_dataset('IIP3', data = IIP3)
                grp.create_dataset('IMD_time', data = data_dict['IMD_time'])   
                grp.create_dataset('g4_IMD', data = g4_IMD)
        finally:    
            hdf5.close()
        
    

#   This is ridiculous function, but we have this problem.... 
    def fit_g3(self):
        hdf5_file = h5py.File(self.data_file)
        try:
            G = np.power(10,np.asarray(hdf5_file['pumping_data_20'].get('Gain'))/10)
            freq_c = np.asarray(hdf5_file['pumping_data_20'].get('freq_c'))
            Delta = freq_c - np.asarray(hdf5_file['pumping_data_20'].get('Pump_freq'))/2
            omega = np.asarray(hdf5_file['pumping_data_20'].get('CW_freq')) - np.asarray(hdf5_file['pumping_data_20'].get('Pump_freq'))/2
            kappa_c = np.asarray(hdf5_file['pumping_data_20'].get('kappa_c'))
            kappa_t = np.asarray(hdf5_file['pumping_data_20'].get('kappa_t'))
            P_pump = dBm_to_Watts(np.asarray(hdf5_file['pumping_data_20'].get('Pump_power')))
            Flux_Data = Flux(np.asarray(hdf5_file['pumping_data_20'].get('current')),self.a,self.b)

#            g4_=0
#            alpha_p = np.sqrt(4*P_pump/(2*np.pi*h)*kappa_c/freq_c) / freq_c
#            a = 16**2
#            b = -( 32*(Delta+g4_*32/3*alpha_p**2)**2 - 32*omega**2+8*kappa_t**2 +16*kappa_t**2/(G-1) )
#            c = (kappa_t*omega)**2+( kappa_t**2/4-omega**2+(Delta+32/3*g4_*alpha_p**2)**2 )**2    
#            g3_Data = np.sqrt((-b-np.sqrt(b**2-4*a*c))/2/a)/alpha_p
#            g3_Data_new = []
#            Flux_Data_new = []
#            for ind, g3_ in enumerate(g3_Data):
#                if not (np.isnan(g3_Data[ind]) or np.isinf(g3_Data[ind])):
#                    g3_Data_new += [g3_]
#                    Flux_Data_new += [Flux_Data[ind]]
#            g3_Data = np.asarray(g3_Data_new)
#            Flux_Data = np.asarray(Flux_Data_new)
#            
#            popt, pcov = curve_fit(lambda x,p : p*self.g3_distributed(x), Flux_Data, g3_Data, p0=[1])
#            fudge_factor = popt  
#            print(fudge_factor)
#            if 'g3_fit' in hdf5_file.keys():
#                del hdf5_file['g3_fit']
#            grp = hdf5_file.create_group('g3_fit')
#            grp.create_dataset('g3', data = g3_Data/fudge_factor)



            def g3_fitting(x,fudge_factor):
                P_pump = dBm_to_Watts(np.asarray(hdf5_file['pumping_data_20'].get('Pump_power'))) * fudge_factor
                Flux_Data = x
                g4_ =  self.g4_distributed( Flux_Data ) + (5-28*3/32)*self.g3_distributed( Flux_Data )**2/self.Freq( Flux_Data )
                alpha_p = np.sqrt(4*P_pump/(2*np.pi*h)*kappa_c/freq_c) / freq_c
                a = 16**2
                b = -( 32*(Delta+g4_*32/3*alpha_p**2)**2 - 32*omega**2+8*kappa_t**2 +16*kappa_t**2/(G-1) )
                c = (kappa_t*omega)**2+( kappa_t**2/4-omega**2+(Delta+32/3*g4_*alpha_p**2)**2 )**2    
                g3_Data = np.sqrt((-b-np.sqrt(b**2-4*a*c))/2/a)/alpha_p
                return g3_Data

            guess = [0.1]
            popt, pcov = curve_fit(lambda x,p: g3_fitting(x,p), Flux_Data, np.abs(self.g3_distributed(Flux_Data)), p0=guess)
            fudge_factor = popt
            print(fudge_factor)
            g3_Data = g3_fitting(Flux_Data,fudge_factor)

        
            if 'g3_fit' in hdf5_file.keys():
                del hdf5_file['g3_fit']
            grp = hdf5_file.create_group('g3_fit')
            grp.create_dataset('g3', data = g3_Data)
            grp.create_dataset('fluxes', data = Flux_Data)
            grp.attrs['fudge_factor'] = fudge_factor
        finally:
            hdf5_file.close()
            