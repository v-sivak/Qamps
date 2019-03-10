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


class JTLA(Amplifier):
    
    def __init__(self, path, name, M):
        self.M = M
        super(JTLA,self).__init__(path,name)
                
            
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
