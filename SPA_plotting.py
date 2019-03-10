# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:34:45 2018

@author: Vladimir
"""

from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib as mpl
from devices_functions import *
import h5py
import numpy as np



fontsize = 8 #1.5
fontsize_tick = 6#15
linewidth = 0.5 #0.25
spinewidth = 0.5 #0.1
markersize = linewidth*6
tick_size = 2.0 #0.5 #3
pad = 0.5 #0.05 #2

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.transparent'] = True
#mpl.rcParams['figure.subplot.bottom'] = 0.2
#mpl.rcParams['figure.subplot.right'] = 0.85
#mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['axes.linewidth'] = spinewidth #1.0 #2.0
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.labelpad'] = pad #4

mpl.rcParams['xtick.major.size'] = tick_size
mpl.rcParams['xtick.major.width'] = spinewidth#2.0
mpl.rcParams['xtick.minor.size'] = tick_size / 2.0
mpl.rcParams['xtick.minor.width'] = spinewidth / 2.0

mpl.rcParams['ytick.major.size'] = tick_size
mpl.rcParams['ytick.major.width'] = spinewidth #2.0
mpl.rcParams['ytick.minor.size'] = tick_size / 2.0 #3.0
mpl.rcParams['ytick.minor.width'] = spinewidth / 2.0

mpl.rcParams['xtick.major.pad']= pad #4
mpl.rcParams['ytick.major.pad']= pad #4
mpl.rcParams['xtick.minor.pad']= pad / 2.0 #4
mpl.rcParams['ytick.minor.pad']= pad / 2.0 #4

mpl.rcParams['xtick.labelsize'] = fontsize_tick
mpl.rcParams['ytick.labelsize'] = fontsize_tick

mpl.rcParams['legend.fontsize'] = fontsize_tick
mpl.rcParams['legend.frameon'] = True

mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = linewidth / 2
            
mpl.rcParams['legend.markerscale'] = 2.0 #relative to inherited marker




path = r'/Users/vs362/Qulab/SPA/DATA/'


    # saves phase Color Map for flux sweeps to name 'phase' in plt ColorMaps, standard Qulab color scheme for phase
def define_phaseColorMap():
    # all numbers from Igor wave 'phaseindex'
    # Igor colors take RGB from 0 to 65535
    rgb = np.zeros((360,3), dtype=np.float)
    rgb[0:90,0] = np.arange(0, 63000, 700)
    rgb[90:180, 0] = 63000 * np.ones(90)
    rgb[180:270, 0] = np.arange(63000, 0, -700)
    rgb[90:180, 1] = np.arange(0, 63000, 700)
    rgb[180:270, 1] = 63000 * np.ones(90)
    rgb[270:360, 1] = np.arange(63000, 0, -700)
    rgb = rgb  / 65535.0
    # ListedColormap takes an arry of RGB weights normalized to be in [0,1]
    phase_cmap = plt_colors.ListedColormap(rgb, name='phase')
    plt.register_cmap(name='phase', cmap=phase_cmap)


def plot_colorMAP(device, colorMap=('hot','phase'), title=None, ftype='.png'):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        date = hdf5_file['results'].attrs.get('flux_sweep_date')
        time = hdf5_file['results'].attrs.get('flux_sweep_time')
        meas_name = hdf5_file['results'].attrs.get('flux_sweep_meas_name') 
        
        
        frequencies = np.asarray(hdf5_file[date][time]['LIN'].get('frequencies'))
        fluxes = Flux(np.asarray(hdf5_file[date][time]['LIN'].get('currents')),device.a,device.b)
        real = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('real'))
        imag = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('imag'))
    finally:
        hdf5_file.close()

    define_phaseColorMap()
    (ylog, yphase) = complex_to_PLOG(np.vectorize(complex)(real, imag))

#    data = np.vectorize(complex)(real, imag)
#    z = data / np.abs(data)
#    for ind in range(np.shape(z)[0]):
#        z0 = z[ind][0]
#        z[ind] = z[ind]/z0
##    z = np.angle(z,deg=True)
#    (ylog, yphase) = complex_to_PLOG(z)
    
    freqs = 1e-9*frequencies
    a_abs = np.transpose(ylog)
    a_phase = np.transpose(yphase)
    

    fig, ax = plt.subplots(2,dpi=150)
    p0 = ax[0].pcolormesh(fluxes, freqs, a_abs, cmap=colorMap[0])
    fig.colorbar(p0,ax=ax[0], label=r'${\rm Log Mag \,(dB)}$')
    ax[0].set_ylabel(r'${\rm Frequency \,(GHz)}$')
    if title:
        ax[0].set_title(title)
    p1 = ax[1].pcolormesh(fluxes, freqs, a_phase, cmap=colorMap[1])
    fig.colorbar(p1, ax=ax[1], ticks=[-180, -90, 0, 90, 180], label=r'${\rm Phase \,(deg)}$')
    ax[1].set_ylabel(r'${\rm Frequency \,(GHz)}$')
#    ax[1].set_title('Phase of a_out (deg)')

    ax[1].set_xlabel(r'${\rm Flux}$, $\Phi/\Phi_0$')
#    ax[1].set_xlabel(r'${\rm Current}$, $(mA)$')
    fig.savefig(device.device_folder + 'phase_colormap' + ftype,dpi=150)





def plot_flux_sweep_fit(device, ftype='.png'):
        
    hdf5_file = h5py.File(device.data_file, 'r')
    try:
        I_data = hdf5_file['fit_flux_sweep']['currents']
        f0_data = hdf5_file['fit_flux_sweep']['f0_exp_fit']
        flux_data = Flux(I_data,device.a,device.b)
        flux_array = np.linspace(flux_data[0]-1/4,flux_data[len(I_data)-1]+1/4,2*len(I_data))
    
        fig1 = plt.figure(figsize=(9, 7),dpi=120)
        plt.subplot(1, 1, 1) 
        fig1.suptitle(device.name + ' flux sweep', fontsize=20)
        plt.xlabel('$\Phi/\Phi_0$', fontsize='x-large')
        plt.ylabel('$f_c(\Phi)$, GHz', fontsize='x-large')
#        plt.xlim(flux_data[0]-1/2, flux_data[len(I_data)-1]+1/2)
    
        plt.plot(flux_data, np.asarray(f0_data)*1e-9, color="red", linewidth=2.0, marker='.', linestyle='None', label='exp. data',zorder=1)        
        plt.plot(flux_array, device.Freq(flux_array)*1e-9, color="blue", linewidth=1, linestyle='-', label='theory fit',zorder=2)        
#        plt.plot(flux_array, device.f_theory_LC(flux_array)*1e-9, color="blue", linewidth=1, linestyle='-', label='theory fit',zorder=2)        

#        F = freq_distributed_without_coupling(Current(flux_array,device.a,device.b), device.a, device.b, device.alpha, device.f0, device.M, 2*45.8/device.L_j, num_mode=1, mu=0.5)
#        plt.plot(flux_array, F*1e-9, color="blue", linewidth=1, linestyle='-', label='theory fit',zorder=2)        

    
        plt.legend(loc='best',fontsize='large')
        fig1.savefig(device.device_folder + 'res_freq_vs_flux' + ftype,dpi=150)
    finally:
        hdf5_file.close()
        

     
def plot_kappa_fit(device, ftype='.png'):
    
    hdf5_file = h5py.File(device.data_file, 'r')
    try:    
        fig4=plt.figure(figsize=(9, 7),dpi=120)
        plt.subplot(1, 1, 1)
            
        fig4.suptitle(r'$\kappa_c$ vs flux', fontsize=20)
        plt.xlabel(r'$\Phi/\Phi_0$', fontsize='x-large')
        plt.ylabel(r'$\kappa_c,\;\rm{MHz}$', fontsize='x-large')
        
        fluxes = Flux(hdf5_file['fit_flux_sweep']['currents'], device.a, device.b)
        kc_data = hdf5_file['fit_flux_sweep']['kc_exp_fit']
        plt.plot(fluxes, np.asarray(kc_data)*1e-6, color="red", linewidth=1.0, marker='.', linestyle='None', label='exp.data',zorder=1)
        plt.plot(fluxes, device.kappa(fluxes)*1e-6, color="blue", linewidth=1.0, linestyle='-', label='theory fit', zorder=1)
        
        plt.legend(loc='best',fontsize='large')
        
        fig4.savefig(device.device_folder + 'kappa_fit' + ftype,dpi=150)
    finally:
        hdf5_file.close()



def plot_P1dB(device, ftype='.png',  sweep_20dB = True, large_sweep=False):

    if large_sweep: sweep_20dB = False
    fig = plt.figure(figsize=(9, 7),dpi=120)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel(r'$ \rm Power \;  (dBm)$', fontsize='xx-large')
#    plt.xticks(np.linspace(7.7,9,11))
    line_style = 'None'

    palette = plt.get_cmap('tab10')

    hdf5_file = h5py.File(device.data_file, 'r')    
    try:
        if large_sweep:
#            fig.suptitle(r'$\rm P_{-1 dB}\;  and/or \; IIP_3\; for \;different\; gains$', fontsize=20)
#            fig.suptitle(r'${\rm Input\;third\;order\;intercept\;point,\;} IIP_3$', fontsize=20)
            plt.xlabel(r'$\Phi/\Phi_0$', fontsize='xx-large')
            if 'nonlin_characterization' in hdf5_file.keys():
                gains = np.asarray(hdf5_file['nonlin_characterization'].get('gains'))
                colors = ['purple','blue','red','green','orange'] if len(gains)<=5 else mpl.cm.rainbow(np.linspace(0,1,len(gains)))
                for i, log_g in enumerate(gains):
                    print(log_g)
                    IIP3 = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('IIP3'))
                    P1dB = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('P_1dB'))
                    
                    flux_IMD = Flux(np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('current')),device.a, device.b)
#                    if log_g>0:
#                        plt.plot(np.abs(flux_IMD), P1dB, color=colors[i], markersize=5.0, alpha=0.5,linestyle='-', marker='.', label='IMD at %.0f dB gain' %log_g)
#                    plt.plot(flux_IMD, IIP3, color=colors[i], markersize=5.0, linestyle='None', marker='.', label='IIP3 at %.0f dB gain' %log_g)
                    plt.plot(flux_IMD, IIP3, color=palette(i), markersize=5.0, linestyle='None', marker='.', label='At %.0f dB gain' %log_g)

        if sweep_20dB:
            # plot only gain points between G_min and G_max
#            fig.suptitle(r'$\rm P_{-1 dB}\;  and/or \; IIP_3\; for \;20dB\; gains$', fontsize=20)
            plt.xlabel(r'$\omega_{\rm signal}/2\pi\rm \; (GHz)$', fontsize='xx-large')
            Happy_indices = []
            group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_20'
                
            for index, G in enumerate( hdf5_file[group]['Gain'] ):
                if G>19 and G<21 and all( g < G+1 for g in hdf5_file[group].get('Gain_array')[index] ) :
                    Happy_indices = Happy_indices + [index]            
            
            freqs = np.zeros(len(Happy_indices))
            current = np.zeros(len(Happy_indices))
            P_1dB_exp = np.zeros(len(Happy_indices))
            IIP3_exp = np.zeros(len(Happy_indices))
            
            for l, index in enumerate(Happy_indices):
                freqs[l] = hdf5_file[group].get('freq_c')[index]
                current[l] = hdf5_file[group].get('current')[index]                
                P_1dB_exp[l] =  hdf5_file[group].get('P_1dB')[index]
                if 'IIP3' in hdf5_file[group]:
                    IIP3_exp[l] = hdf5_file[group]['IIP3'][index]
                    line_style = '-' # when IMD data was taken we made detailed flux sweeps for gain measuremens, and in that case data points can be connected and it looks okay, otherwise it looks like scatter plot
    
#            plt.plot(freqs*1e-9, P_1dB_exp, linestyle = line_style, linewidth = 0.7, color='red', markersize=7,marker = '.', label=r'$P_{-1dB}$')        
            plt.plot(current*1e3, P_1dB_exp, linestyle = line_style, linewidth = 0.7, color='red', markersize=7,marker = '.', label=device.name + r', $P_{-1dB}$ data')        

            # if IMD data was taken, this will plot that too
            if 'IIP3' in hdf5_file[group]:
#                plt.plot(freqs*1e-9, IIP3_exp, linestyle = '-', color='grey', linewidth = 0.7, markersize=7,marker = '.', label=r'$IIP_3$')
                plt.plot(current*1e3, IIP3_exp, linestyle = '-', color='grey', linewidth = 0.7, markersize=7,marker = '.', label=device.name + r', $IIP_3$')

        
        plt.legend(loc = 'best', fontsize = 'large') 
        fig.savefig(device.device_folder + 'P_1dB_vs_freq' + ftype, dpi=600)
        
    finally:
        hdf5_file.close()



def plot_NVR(device, ftype='.png'):
    
    fig = plt.figure(figsize=(9, 7),dpi=120)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel(r'$ \rm NVR \;  (dB)$', fontsize='xx-large')
    plt.suptitle(r'$\Delta=0$', fontsize='xx-large')
    line_style = 'None'

    palette = plt.get_cmap('tab10')

    hdf5_file = h5py.File(device.data_file, 'r')    
    try:
        plt.xlabel(r'$\Phi/\Phi_0$', fontsize='xx-large')
        if 'nonlin_characterization' in hdf5_file.keys():
            gains = np.asarray(hdf5_file['nonlin_characterization'].get('gains'))
            colors = ['purple','blue','red','green','orange'] if len(gains)<=5 else mpl.cm.rainbow(np.linspace(0,1,len(gains)))
            for i, log_g in enumerate(gains[1:]):
                print(log_g)
                freq_c = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('freq_c'))                
                NVR = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('NVR'))
                NVR_spike = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('NVR_spike'))
                current = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('current'))
                flux_ = Flux(current,device.a, device.b)                
                Ind_bad_current = np.argmin(np.abs(current-1.455e-3))
#                plt.plot(freq_c*1e-9, NVR, color=palette(i), markersize=5.0, linestyle='None', marker='.', label='At %.0f dB gain' %log_g)
                NVR = np.zeros(len(flux_)) if log_g==0 else NVR
                NVR_spike = np.zeros(len(flux_)) if log_g==0 else NVR_spike
#                plt.plot(np.delete(flux_,Ind_bad_current), np.delete(NVR,Ind_bad_current), color=palette(i), markersize=5.0, linestyle='None', marker='.', label='At %.0f dB gain' %log_g)
#                plt.plot(np.delete(flux_,Ind_bad_current), np.delete(NVR+NVR_spike,Ind_bad_current), color=palette(i), markersize=5.0, linestyle='-', alpha =0.5, marker='.', label='At %.0f dB gain' %log_g)

                plt.plot(np.delete(freq_c*1e-9,Ind_bad_current), np.delete(NVR,Ind_bad_current), color=palette(i), markersize=5.0, linestyle='None', marker='.', label='At %.0f dB gain' %log_g)
                plt.plot(np.delete(freq_c*1e-9,Ind_bad_current), np.delete(NVR+NVR_spike,Ind_bad_current), color=palette(i), markersize=5.0, linestyle='-', alpha =0.5, marker='.', label='At %.0f dB gain' %log_g)

#        plt.ylim(-1,10)
        plt.legend(loc = 'best') 
        fig.savefig(device.device_folder + 'NVR_vs_flux' + ftype, dpi=150)
        
    finally:
        hdf5_file.close()

 
      
def plot_kappa_from_lin_and_bandwidth(device, ftype='.png',  sweep_20dB = True, large_sweep=False):
        
    if large_sweep: sweep_20dB=False
    fig = plt.figure(figsize=(9, 7),dpi=120)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    fig.suptitle(r'$Kappa, %s$' %device.name, fontsize=20)
    plt.ylabel(r'$kappa,\; (MHz)$', fontsize='xx-large')
    plt.xlabel(r'$f_{\rm signal}\rm \; (GHz)$', fontsize='xx-large')
#        plt.xticks(np.linspace(7.7,9,11))

    hdf5_file = h5py.File(device.data_file)
    try:
        if large_sweep:
            if 'nonlin_characterization' in hdf5_file.keys():
                gains = np.asarray(hdf5_file['nonlin_characterization'].get('gains'))
                colors = ['purple','blue','red','green','orange'] if len(gains)<=5 else mpl.cm.rainbow(np.linspace(0,1,len(gains)))
                for i, log_g in enumerate(gains):
                    freqs = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('freq_c'))
                    if log_g:
                        G = np.power(10,np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('Gain'))/10)
                        B = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('Bandwidth'))
                        kappa = B * np.sqrt((G-2)*(np.sqrt(G)+1)/2/np.sqrt(G)*(1+np.sqrt((G-1)/(G-2))))
                    else:
                        kappa = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('kappa_c'))
#                    if log_g ==0 or log_g==20:
                    plt.plot(freqs*1e-9, kappa*1e-6, color=colors[i], markersize=5.0, linestyle='-', marker='.', label='at %.0f dB gain' %log_g)
            
        if sweep_20dB:    
            group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_20'
            Happy_indices = []
            for index, G in enumerate( hdf5_file[group]['Gain'] ):
                if G>18 and G<22 and all( g < G+1 for g in hdf5_file[group].get('Gain_array')[index] ) :
                    Happy_indices = Happy_indices + [index] 
#            if G>18 and G<22:
#                Happy_indices = Happy_indices + [index]
            freqs = np.zeros(len(Happy_indices))
            kappa_bw_exp = np.zeros(len(Happy_indices))
            kappa_lin_exp = np.zeros(len(Happy_indices))        
            for l, index in enumerate(Happy_indices):
                freqs[l] = hdf5_file[group]['freq_c'][index]
                kappa_bw_exp[l] = hdf5_file[group]['Bandwidth'][index]*np.sqrt(np.power(10,hdf5_file[group]['Gain'][index]/10))
                kappa_lin_exp[l] = hdf5_file[group]['kappa_c'][index]
            plt.plot(freqs*1e-9, kappa_bw_exp*1e-6, linestyle = '-', linewidth = 0.7, color='red', markersize=7,marker = '.', label='from bandwidth',zorder=1)
            plt.plot(freqs*1e-9, kappa_lin_exp*1e-6, linestyle = '-', linewidth = 0.7, color='black', markersize=7,marker = '.', label='from linear fit',zorder=1)
        plt.legend(loc='best', fontsize='medium')
        label = '20dB' if sweep_20dB else 'large_sweep'
        fig.savefig(device.device_folder + 'kappa_from_bandwidth_' + label + ftype, dpi=150)
    finally:
        hdf5_file.close()


    

def plot_g3_from_pumping(device, ftype='.png'):
    
    hdf5_file = h5py.File(device.data_file, 'r')
    try:
        g3_Data = np.abs(np.asarray(hdf5_file['g3_fit'].get('g3')))
        Flux_Data = np.asarray(hdf5_file['g3_fit'].get('fluxes'))
        
        fig = plt.figure(figsize=(9, 7),dpi=120)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        fig.suptitle(r'$g_3\; \rm nonlinearity\; for\;$' + device.name, fontsize=20)
        plt.xlabel(r'$\Phi/\Phi_0$', fontsize='x-large')
        plt.ylabel(r'$g_3\;\rm (MHz)$', fontsize='x-large')
        plt.subplot(1,1,1).set_yscale('log')
        fluxes = np.linspace(-0.5,0.5,1001)
        plt.ylim(min(g3_Data*1e-6)/2,max(g3_Data*1e-6)*2)
        plt.xlim(max(min(Flux_Data)-0.05,-0.5), min(max(Flux_Data)+0.05,0.5))
        plt.yscale('log')
                
        plt.plot(fluxes, np.abs(device.g3_distributed(fluxes)*1e-6), color="black", linewidth=1.0, linestyle='-', label='$g_3$ theory')
        plt.plot(Flux_Data, np.abs(g3_Data*1e-6), color='red', linewidth=1.0, marker='.',linestyle='None',markersize=5, label='$g_3$ data')
        plt.legend(loc='best',fontsize='x-large')
        fig.savefig(device.device_folder+'g3_nonlinearity_exp' + ftype,dpi=150)
    finally:
        hdf5_file.close()

        

def plot_g4_exp_and_theory(device, IMD=False, large_sweep=False, ftype='.png'):
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        fig = plt.figure(figsize=(3.375, 2.0),dpi=240)
#        fig = plt.figure(figsize=(9, 7),dpi=120)
#        fig.suptitle(r'$\chi\equiv12g_4\;\rm nonlinearity \;in\;$' + device.name, fontsize=20)
        plt.subplot(1,1,1).set_yscale('log')
        plt.ylabel(r'$|K|/2\pi\;\rm (MHz)$') #fontsize='x-large'
        plt.xlabel(r'$\Phi/\Phi_0$') #fontsize='x-large'
#        plt.xlim(0.0,0.5)
#        plt.ylim(1e-3,1e-1)        
#        plt.ylim(1e-4,1e-3)

        fluxes = np.linspace(0.0,0.50,1001)
#        plt.plot(fluxes, np.abs(12*(device.g4_corrected_LC(fluxes))*1e-6), color="blue", linestyle='-', label='K, Lumped LC model',zorder=1)
#        plt.plot(fluxes, np.abs(12*device.g4_BBQ_LC(fluxes)*1e-6), color="red", linestyle='-', label='Linear participation ratio',zorder=1)
#        plt.plot(fluxes, np.abs(12*device.g4_distributed(fluxes)*1e-6), color='black', linestyle='-', label=r'$\rm Distributed \; model$',zorder=1)
#        plt.plot(fluxes, np.abs(180*device.g6_LC(fluxes)*1e-6), color="red", linestyle='-', label='K\', Lumped LC model',zorder=1)


        
        if 'fit_stark_shift' in hdf5_file.keys():
            if 'flux' in hdf5_file['fit_stark_shift'].keys():
                Flux_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('flux'))
                X = Flux_Data_Stark
                x_label = r'$\Phi/\Phi_0$'
            else:
                Current_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('current'))
                X = Current_Data_Stark
                x_label = r'Current (mA)'
                
            Kerr_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('g4'))
            K_prime_Data = np.asarray(hdf5_file['fit_stark_shift'].get('K_prime'))

#            ind = Flux_Data_Stark.argsort()            
#            plt.plot(np.abs(Flux_Data_Stark[ind]), 12*Kerr_Data_Stark[ind]*1e-6, color='blue', markersize=3, linestyle='None', marker='.', label='Extracted from Stark shift')   

            ind = X.argsort()
            plt.plot(X[ind], 12*np.abs(Kerr_Data_Stark[ind])*1e-6, color='blue', linestyle='None', marker='.', label='Extracted from Stark shift')   
            plt.xlabel(x_label)

#            plt.plot(X[ind], K_prime_Data*1e-6, color='red', linestyle='None', marker='.', label='Extracted from Stark shift')   

        
#            plt.plot(Current_Data_Stark*1e3, np.abs(12*Kerr_Data_Stark)*1e-6, color='blue', markersize=3, linestyle='None', marker='.', label='Extracted from Stark shift')   
#            plt.xlabel(r'Current (mA)')

        if IMD:
            group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_0'
            flux_IMD = Flux(np.asarray(hdf5_file[group].get('current')),device.a, device.b)
            g4_IMD = np.asarray(hdf5_file[group].get('g4_IMD'))
            plt.plot(flux_IMD, np.abs(12*g4_IMD*1e-6), color='black', markersize=4.0, linestyle='None', marker='.', label='K extracted from $IIP_3$')   
            print(np.abs(12*g4_IMD*1e-6))
        
        if large_sweep:
            if 'nonlin_characterization' in hdf5_file.keys():
                gains = np.asarray(hdf5_file['nonlin_characterization'].get('gains'))
                colors = ['purple','blue','red','green','orange'] if len(gains)<=5 else mpl.cm.rainbow(np.linspace(0,1,len(gains)))
                for i, log_g in enumerate(gains):
                    flux_IMD = Flux(np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('current')),device.a, device.b)
                    g4_IMD = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('g4_IMD'))
                    plt.plot(flux_IMD, np.abs(12*g4_IMD*1e-6), color=colors[i], markersize=5.0, linestyle='None', marker='.', label='IMD at %.0f dB gain' %log_g)
 
        plt.legend(loc='lower left') #fontsize='medium'
        fig.tight_layout()
        
        fig.savefig(device.device_folder + 'g4_nonlinearity_exp' + ftype,dpi=240)
    finally:
        hdf5_file.close()
    

def plot_filter_insertion_loss():
    ######################
#   PPF4
    ######################
    
    fig = plt.figure(figsize=(14, 7),dpi=120)
    plt.rc('xtick', labelsize=16)
    plt.yticks(np.linspace(-60,0,13))
    plt.rc('ytick', labelsize=16)
    plt.xticks(np.linspace(1,20,20))
    fig.suptitle(r'PPF4 insertion loss', fontsize=20)
    plt.ylabel(r'Power, dB', fontsize='xx-large')
    plt.xlabel(r'Frequency, GHz', fontsize='xx-large')
    plt.ylim(-60,2)

    # Fridge warm
    freq = np.linspace(3e9,12e9,1601)
    short_ = extract_attenuation(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.hdf5', '180325', '131538', '"CH1_S11_1"', freq)
    filter_ = extract_attenuation(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.hdf5', '180325', '135445', '"CH1_S11_1"', freq)
    insertion_loss = filter_ - short_
    plt.plot(freq*1e-9, insertion_loss, linestyle = '-', color='green', linewidth=1, label='Fridge warm')

    # Fridge cold
    short_ = extract_attenuation(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.hdf5', '180330', '093250', '"CH1_S11_1"', freq)
    filter_ = extract_attenuation(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.hdf5', '180330', '103406', '"CH1_S11_1"', freq)
    insertion_loss = filter_ - short_
    plt.plot(freq*1e-9, insertion_loss, linestyle = '-', color='blue', linewidth=2, label='Fridge cold')

    # Calibrated VNA
    freq = np.linspace(1e9,20e9,1601)
    insertion_loss = extract_attenuation(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.hdf5', '180320', '172415', '"CH3_S34_8"', freq)
    plt.plot(freq*1e-9, insertion_loss, linestyle = '-', color='red', linewidth=2, label='Calibrated VNA')

    # HFSS
    array = np.transpose( np.loadtxt(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ansys.txt') )
    insertion_loss = array[2]
    frequency = array[0]
    plt.plot(frequency, insertion_loss, linestyle = '-', color='green', linewidth=2, label='HFSS')


    plt.legend(loc='best',fontsize='large')
    fig.savefig(r'Y:\volodymyr_sivak\SPA\DATA\ppf4\ppf4.jpg',dpi=150)


 
def plot_resonator_population(device, ftype='.png'): 

    fig = plt.figure(figsize=(9, 7),dpi=120)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    fig.suptitle(r'Resonator population limitations ' + device.name, fontsize=20)
    plt.xlabel(r'$\Phi/\Phi_0$', fontsize='x-large')
    plt.ylabel(r'Number of photons', fontsize='x-large')
    plt.ylim(1,1e9)
    plt.subplot(1,1,1).set_yscale('log')
    fluxes = np.linspace(0.1,0.5,500)
              
    plt.plot(fluxes, (device.g3_distributed(fluxes)/device.g4_distributed(fluxes))**2, color="blue", linewidth=1.0, linestyle='-', label=r'$g_4$ vs $g_3$ estimate')
    plt.plot(fluxes, 2*device.E_j/device.Freq(fluxes)*device.c2_eff(fluxes)*( 5*device.c4_eff(fluxes)/device.c5_eff(fluxes) )**2, color="magenta", linewidth=1.0, linestyle='-', label=r'$g_4$ vs $g_5$ estimate')
    plt.plot(fluxes, np.abs(2*device.E_j/device.Freq(fluxes)*device.c2_eff(fluxes)*20*device.c3_eff(fluxes)/device.c5_eff(fluxes) ), color="orange", linewidth=1.0, linestyle='-', label=r'$g_3$ vs $g_5$ estimate')

    
    plt.plot(fluxes, np.abs(device.kappa(fluxes)/8/device.g3_distributed(fluxes))**2, color="red", linewidth=1.0, linestyle='-', label=r'$n_{\infty}$ for pumping')
    
    plt.legend(loc='best',fontsize='x-large')
#    fig.savefig(device.device_folder+'g3_nonlinearity_exp' + ftype,dpi=150)    
    

def plot_gain_stability(device, ftype='.png'):
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        fig = plt.figure(figsize=(9, 7),dpi=120)
        fig.suptitle(r'Gain stability in ' + device.name, fontsize=20)
        plt.xlabel(r'Time (min)', fontsize='x-large')
       
#        Gains = np.loadtxt(r'/Volumes/Shared/Data/JPC_2018/05-02-2018 Cooldown/gain.txt')
#        Ts = np.loadtxt(r'/Volumes/shared/Data/JPC_2018/05-02-2018 Cooldown/temperature.txt')

        Gains = np.loadtxt(r'/Volumes/shared/Data/JPC_2018/09-26-2018 Cooldown/gain.txt')
        Ts = np.loadtxt(r'/Volumes/shared/Data/JPC_2018/09-26-2018 Cooldown/temperature.txt')
        Times = [ i for i,g in enumerate(Gains) ]
#        Gains = np.asarray(hdf5_file['180417']['084530'].get('gains'))

        plt.subplot(2,1,1)
        plt.ylabel(r'Gain (dB)', fontsize='x-large')
        plt.plot(Times, Gains, color='red', markersize=5.0, linestyle='None', marker='.')   

        plt.subplot(2,1,2)
        plt.ylabel(r'Temperature (mK)', fontsize='x-large')
        plt.plot(Times, Ts*1000, color='red', markersize=5.0, linestyle='None', marker='.')    

        fig.savefig(device.device_folder + 'Gain_stability' + ftype, dpi=150)
    finally:
        hdf5_file.close()
        print(1)
    
    

def plot_flux_sweep_data_compare_devices(device_list, labels=None, colors=None, theory=False, fpath=None, ftype='.png'):
#    path = fpath if fpath else path
    fig1=plt.figure(figsize=(3.375, 2.0),dpi=240) #(9,7) #APS: (6, 3.5)
    plt.subplot(1, 1, 1)        
#    fig1.suptitle('Resonance frequency vs flux', fontsize=20)
    plt.xlabel(r'$\Phi/\Phi_0$')                                                #APS: fontsize='x-large'
    plt.ylabel(r'$\omega_a / 2\pi \;\rm (GHz)$')                                            #APS: fontsize='x-large'
    xlim = (-1.0, 1.0)
    plt.xticks(np.linspace(-1,1,21))
    plt.xlim(xlim)
    plt.ylim(4,8) #(4, 9.5)
    colors = colors if colors else mpl.cm.rainbow(np.linspace(0,1,len(device_list)))
    labels = labels if labels else [d.name for d in device_list]
    fluxes = np.linspace(xlim[0], xlim[-1], 2001)
    for index, device in enumerate(device_list):
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            I_data = hdf5_file['fit_flux_sweep']['currents']
            f0_data = hdf5_file['fit_flux_sweep']['f0_exp_fit']
            flux_data = Flux(I_data, device.a, device.b)
            if theory:
                plt.plot(fluxes, device.Freq(fluxes)*1e-9, color=colors[index], linestyle='-', label=labels[index])     #APS: linewidth=1.0
            plt.plot(flux_data, np.asarray(f0_data)*1e-9, color=colors[index], marker='.', linestyle='None')#, label=labels[index]), APS: markersize=2.0, 
        finally:
            hdf5_file.close()
    plt.legend(loc='best')                                                      #APS: fontsize='x-large'
    plt.tight_layout()
    fig1.savefig(path + 'flux_sweep_all_samples_paper' + ftype,dpi=240)
    
    
    
    
def plot_p_compare_devices(device_list, labels=None, colors=None, fpath=None, ftype='.png'):
    path = fpath if fpath else path
    fig = plt.figure(figsize=(1.75,1.5), dpi=240)                             #APS: (6,3.5), earlier fig (3.375, 2.0)
    plt.subplot(1, 1, 1)
    plt.xlabel(r'$\Phi/\Phi_0$')                                                #APS: fontsize='x-large'
    plt.ylabel(r'$p$')                                                          #APS: fontsize='x-large'
    xlim = [0.0, 0.5]                                                           #APS: (-1, 1)
    plt.xlim(xlim)
    plt.ylim(0, 1.0)
    colors = colors if colors else mpl.cm.rainbow(np.linspace(0,1,len(device_list)))
    labels = labels if labels else [d.name for d in device_list]
    fluxes = np.linspace(xlim[0], xlim[1], 1001) #APS: 2001
    
    for index, device in enumerate(device_list):
        plt.plot(fluxes, device.participation(fluxes), color=colors[index],
                 linestyle='-', label=labels[index])                            #APS: linewidth=1.0
#    plt.legend(loc='best', fontsize='x-large')
    plt.tight_layout()
    fig.savefig(path + 'p_all_samples_paper' + ftype, dpi=240)


def plot_g3_from_pumping_compare_devices(device_list, labels=None, colors=None, fpath=None, ftype='.png'):
    path = fpath if fpath else path
    fig = plt.figure(figsize=(3.375, 2.0), dpi=240) #(9,7) APS: (8,6)
#    fig.suptitle(r'$g_3\; \rm nonlinearity$', fontsize=20)
    plt.xlabel(r'$\Phi/\Phi_0$')                                                #APS: fontsize='xx-large'
    plt.ylabel(r'$|g_3|/2\pi\;\rm (MHz)$')                                      #APS: fontsize='xx-large'
    plt.subplot(1,1,1).set_yscale('log')
    plt.xlim([0.0, 0.505])
    plt.ylim([1e-1, 50])
    colors = colors if colors else mpl.cm.rainbow(np.linspace(0.0,1.0,len(device_list)))
    fluxes = np.linspace(-0.5,0.5,1001)
    leg_labels = labels if labels else [d.name for d in device_list]
    
    for index, device in enumerate(device_list):
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            g3_Data = np.abs(np.asarray(hdf5_file['g3_fit'].get('g3')))
            Flux_Data = np.asarray(hdf5_file['g3_fit'].get('fluxes'))
            plt.plot(fluxes, np.abs(device.g3_distributed(fluxes)*1e-6), color=colors[index],
                     linestyle='-', label=leg_labels[index])                    #APS: linewidth=1.0, 
            plt.plot(Flux_Data, np.abs(g3_Data*1e-6), color=colors[index],
                     marker='.', linestyle='None')                              #APS: markersize=6.0, 
        finally:
            hdf5_file.close()
        plt.legend(loc='best')                                                  #APS: fontsize='xx-large'
        plt.tight_layout()
        fig.savefig(path + 'g3_all_samples_paper' + ftype, dpi=240)
        
    
def plot_stark_shift_data_compare_devices(device_list, labels=None, colors=None, fpath=None, ftype='.png'):
    path = r'Y:\volodymyr_sivak\SPA\DATA\\'    
    path = fpath if fpath else path
    fig = plt.figure(figsize=(3.375, 2.0),dpi=240) #(9,7), APS: (8,6) #note: must replace/relink in illustrator if change figsize
#    fontsize = 8
#    fig.suptitle(r'$\chi\equiv12g_4$ for all devices', fontsize=20)
    axLeft = plt.gca()          
    axLeft.set_ylabel(r'$|g_4| / 2\pi\;\rm (MHz)$') #APS: 'xx-large'
    axLeft.set_xlabel('$\Phi/\Phi_0$') #APS: 'xx-large'
    axLeft.set_yscale('log')
    xlim = (0, 0.50)
    axLeft.set_xlim(xlim)
    ylimK = np.array((1e-4 * 12, 1e2))
    axLeft.set_ylim(ylimK / 12)
    colors = colors if colors else mpl.cm.rainbow(np.linspace(0.0,1.0,len(device_list)))
    fluxes = np.linspace(-0.5, 0.5, 1001)
    leg_labels = labels if labels else [d.name for d in device_list]
    axRight = axLeft.twinx()
    axRight.set_ylabel(r'$|K| / 2\pi\;\rm (MHz)$', rotation=270, labelpad=mpl.rcParams['axes.labelpad']+mpl.rcParams['axes.labelsize'])
    axRight.set_yscale('log')
    axRight.set_xlim(xlim)
    axRight.set_ylim(ylimK)
    
    for index, device in enumerate(device_list):
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            if 'fit_stark_shift' in hdf5_file.keys():
                Flux_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('flux'))
                Kerr_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('g4'))
                axLeft.plot(fluxes, np.abs(device.g4_distributed(fluxes)*1e-6), color=colors[index],
                         linestyle='-', label=leg_labels[index]) #linewidth=1.0, 
                axRight.plot(np.abs(Flux_Data_Stark), np.abs(12*Kerr_Data_Stark*1e-6), color=colors[index],
                         marker='.', linestyle='None') #markersize=4.0 (APS 6.0)
        finally:
            hdf5_file.close()
    axLeft.legend(loc='best') #APS: 'xx-large'
    plt.tight_layout()
    fig.savefig(path + 'g4_all_samples_paper' + ftype,dpi=240)

     
   
def plot_P1dB_data_compare_devices(device_list, fpath=None, ftype='.png'):
        path = fpath if fpath else  r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/DATA/'
#        device_list = args
        colors = mpl.cm.rainbow(np.linspace(0,1,len(device_list)))
        colors = ['green','orange','blue','red','black']
        palette = plt.get_cmap('tab10')
        labels = ['Device A','Device B','Device C','Device D','Device X']
        
        fig = plt.figure(figsize=(3.375,2.2), dpi=240)
        plt.ylabel(r'$ P_{1\rm dB} \; \rm (dBm)$')
        plt.xlabel(r'$f_{\rm signal}\rm \; (GHz)$')
        
        for i, device in enumerate(device_list):
            hdf5_file = h5py.File(device.data_file, 'r')    
            try:
                # plot only gain points between G_min and G_max
                group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_20'
                Happy_indices = []
                for index, G in enumerate( hdf5_file[group]['Gain'] ):
                    if G>19 and G<21:
                        Delta = float(hdf5_file[group]['freq_c'][index])-float(hdf5_file[group]['Pump_freq'][index])/2
                        if Delta < 3e6 and Delta>-3e6:
                            Happy_indices = Happy_indices + [index]

#                for index, G in enumerate( hdf5_file['pumping_data']['Gain'] ):
#                    if G>19 and G<21 and all( g < G+1 for g in hdf5_file['pumping_data'].get('Gain_array')[index] ):
#                        Happy_indices = Happy_indices + [index]
       
                freqs = np.zeros(len(Happy_indices))
                fluxes = np.zeros(len(Happy_indices))
                P_1dB_exp = np.zeros(len(Happy_indices))
                IIP3_exp = np.zeros(len(Happy_indices))
                line_style='-'
                print(device.name)
                
                for l, index in enumerate(Happy_indices):
                    freqs[l] = hdf5_file[group].get('freq_c')[index]
                    fluxes[l] = Flux(hdf5_file[group].get('current')[index],device.a,device.b) 
                    P_1dB_exp[l] =  hdf5_file[group].get('P_1dB')[index]
#                    if 'IIP3' in hdf5_file[group]:
#                        IIP3_exp[l] = hdf5_file[group]['IIP3'][index]
#                        line_style = '-' # when IMD data was taken we made detailed flux sweeps for gain measuremens, and in that case data points can be connected and it looks okay, otherwise it looks like scatter plot
                
                plt.xlabel(r'$\Phi/\Phi_0$')
                plt.plot(fluxes, P_1dB_exp, linestyle = line_style, color=palette(i), marker = '^', label=labels[i])        
                
#                plt.xlabel(r'$f_{\rm signal}\rm \; (GHz)$', fontsize='xx-large')
#                plt.plot(freqs*1e-9, P_1dB_exp, linestyle = line_style, linewidth = 0.7, color=colors[i], markersize=7,marker = '.', label=device.name + r', $P_{-1dB}$')        
#                
                # if IMD data was taken, this will plot that too
#                if 'IIP3' in hdf5_file['pumping_data']:
#                    plt.plot(freqs*1e-9, IIP3_exp, linestyle = '-', color=colors[i], linewidth = 0.7, markersize=7,marker = '.', label=device.name + r', $IIP_3$',alpha=0.3)
            finally:
                hdf5_file.close()            
        plt.legend(loc='best')
        fig.savefig(path + 'P_1db_vs_freq_all_devices' + ftype,dpi=240)


      
def plot_P1dB_and_pump_power_compare_devices(device_list, fpath=None, ftype='.png'):
    path = r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/DATA/'
    path = fpath if fpath else path
    colors = ['green','orange','blue','red','purple']
    palette = plt.get_cmap('tab10')
    
    labels = [x.name for x in device_list]
    labels = ['SPA04','SPA08','SPA13','SPA14','JTLA08_v2']
    labels = ['Device A','Device B','Device C','Device D','Device X']


    lower_lim = -120
    upper_lim = -50    

#    fig, axs = plt.subplots(1, 1, figsize=(3.375,2.0), dpi=240)
#    ax = axs

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(3.375,2.4), dpi=240)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax = plt.subplot(gs[0])


    plt.yticks(np.linspace(lower_lim,upper_lim, int((upper_lim-lower_lim)/10+1) ))
    plt.xticks(np.linspace(lower_lim,upper_lim, int((upper_lim-lower_lim)/10+1) ))

    plt.ylabel(r'$GP_{\rm 1dB}$ (dBm)')
    plt.xlabel(r'Pump power (dBm)')

    legend_handles_1 = []
    legend_handles_2 = []
        
    l = ax.plot(np.linspace(lower_lim,upper_lim), np.linspace(lower_lim,upper_lim)-3, color='black', linestyle='-', label= r'Efficiency $\eta_p=0.5$')
    legend_handles_1.append(l[0])
    l = ax.plot(np.linspace(-93,upper_lim+3), np.linspace(-93,upper_lim+3)-3-30, color='black', linestyle='--', label= r'Efficiency $\eta_p=10^{-3}$')
    legend_handles_1.append(l[0])

    
    for i, device in enumerate(device_list):
        hdf5_file = h5py.File(device.data_file, 'r')
        try:
            # plot only gain points between G_min and G_max
            group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_20'
            Happy_indices = []
            for index, G in enumerate( hdf5_file[group]['Gain'] ):
                if G>19 and G<21:
                    Delta = float(hdf5_file[group]['freq_c'][index])-float(hdf5_file[group]['Pump_freq'][index])/2
                    if Delta < 3e6 and Delta>-3e6:
                        Happy_indices = Happy_indices + [index]  
            P_1dB_exp = np.zeros(len(Happy_indices))
            Pump = np.zeros(len(Happy_indices))
            Gain = np.zeros(len(Happy_indices))
            print(device.name)            
            for l, index in enumerate(Happy_indices):
                P_1dB_exp[l] =  hdf5_file[group].get('P_1dB')[index]
                Pump[l] = hdf5_file[group].get('Pump_power')[index]
                Gain[l] = hdf5_file[group].get('Gain')[index]      
            l = ax.plot(Pump, P_1dB_exp + Gain, color=palette(i), linestyle='None', marker = '.', label=labels[i],alpha=1.0)                    
            legend_handles_1.append(l[0])
        finally:
            hdf5_file.close()


    l = ax.plot(-55.4, -123+20, color='black', marker='*', linestyle='none', markersize=5, label='Flux-pumped 8 SQUID JPA, Phys Rev B 89, 214517 (2014)')
    legend_handles_2.append(l[0])
    l = ax.plot(-64, -132+20, color='black', marker='X', linestyle='none', markersize=4, label='Wireless Josephson Paramp, Appl. Phys. Lett. 104, 232605 (2014)')
    legend_handles_2.append(l[0])
    l = ax.plot(-61, -100+20, color='black', marker='h', linestyle='none', markersize=4, label='Josephson Traveling-Wave Amplifier, Science 350, 307 (2015)')    
    legend_handles_2.append(l[0])
    l = ax.plot(-52, -133.5+20, color='black', marker='^', linestyle='none', markersize=4, label='Flux-pumped multimode paramp, J. Appl. Phys 118, 154501 (2015)')    
    legend_handles_2.append(l[0])
    l = ax.plot(-70, -110+20, color='black', marker='>', linestyle='none', markersize=4, label='Broadband JPA, Appl. Phys. Lett. 107, 262601 (2015)')    
    legend_handles_2.append(l[0])    
    l = ax.plot(-68, -120+20, color='black', marker='d', linestyle='none', markersize=4, label='State-of-the-art JPC, Appl. Phys. Lett. 111, 202603 (2017)')
    legend_handles_2.append(l[0])
    l = ax.plot(-80.5, -117+20, color='black', marker='P', linestyle='none', markersize=4, label='Current-pumped 80 SQUID JPA, arXiv:1809.08476 (2018)')    
    legend_handles_2.append(l[0])
    l = ax.plot(-62, -125+20, color='black', marker='<', linestyle='none', markersize=4, label='Flux-pumped lumped JPA, arXiv:1812.07621v1 (2018)')    
    legend_handles_2.append(l[0])



#    l = ax.plot(-33, -115+20, color='black', marker='>', linestyle='none', markersize=4, label='Flux-pumped 4 SQUID JPA, EPJ Web Conf., 198, 00008 (2019)')    
#    legend_handles_2.append(l[0])



    # For TWPA we estimated according to Hong-Tong (or whatever his name is) TWPA spread-sheet that pump power is about -71,
    # but based on Macklin's thesis I estimated -61 dBm. He gives the pump current as 0.9 of I_c which for their junctions is 4.6 uA
    # For broadband JPA I asked Tanay Roy to give me an estimate which he obtained from some simulation




    first_legend = ax.legend(handles=legend_handles_1, loc='upper left', frameon=False)
    ax.add_artist(first_legend)

    second_legend = ax.legend(handles=legend_handles_2, loc='upper left', bbox_to_anchor=(-0.15, -0.2))
    for i,handle in enumerate(legend_handles_2): second_legend.legendHandles[i]._legmarker.set_markersize(6) 
    ax.add_artist(second_legend)

    
#    plt.tight_layout()
    fig.savefig(path + 'power_plot_all_devices' + ftype,dpi=240)



def plot_stark_shift_vs_nbar(device, skip=1, fpath=None, ftype='.png'):
#    path = fpath if fpath else path
    hdf5_file = h5py.File(device.data_file,'r')
    try:    
        Kerr_Data_dict = {}
        Kerr_Data_dict['current'] = np.asarray(hdf5_file['fit_stark_shift'].get('current'))[::skip]
        Kerr_Data_dict['date'] = np.asarray(hdf5_file['fit_stark_shift'].get('date'))[::skip]
        Kerr_Data_dict['time'] = np.asarray(hdf5_file['fit_stark_shift'].get('time'))[::skip]
        Kerr_Data_dict['freq_drive'] = np.asarray(hdf5_file['fit_stark_shift'].get('drive_freq'))[::skip]

        temp = np.zeros(len(Kerr_Data_dict['current']))

        #now the figure
        fig = plt.figure(figsize=(3.375+0.25+1.0/16, 2.0), dpi=240) #APS: (6,3.5) #(9,7)
#        fig.suptitle(r'$\rm Stark\;shift\;vs\;\overline{n}\;for\;different\;flux\;points$', fontsize=20)    
        plt.ylabel(r'$\Delta \omega_a / 2\pi \;\rm (MHz)$')   #APS: 'x-large'   
        plt.xlabel(r'$\overline{n}\;\rm (photons)$')        #APS: 'x-large'
        
        nbar_max = 2700
        stark_shift_max = 60
#        plt.xlim(0,nbar_max)
#        plt.ylim(-stark_shift_max,stark_shift_max)
        norm = mpl.colors.Normalize(vmin = 0.0, vmax = 0.5)
#        colors = mpl.cm.viridis(np.linspace(0.0, 1.0, len(Kerr_Data_dict['current'])))
        
        for j, current in enumerate(Kerr_Data_dict['current']):
            
            date = float_to_DateTime( Kerr_Data_dict['date'][j] )
            time = float_to_DateTime( Kerr_Data_dict['time'][j] )
            f_d = Kerr_Data_dict['freq_drive'][j]
            
            #linear scattering data
            meas_name = hdf5_file['results'].attrs.get('flux_sweep_meas_name')
            f_0 = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('f0'))
            k_c = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('kc'))
            k_i = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('ki'))
            k_t = k_c + k_i 
            
            line_attenuation = attenuation(f_d, device.data_file, 'drive_attenuation')
            
            if 'powers' in list(hdf5_file[date][time]['LIN'].keys()):
                log_powers = np.asarray(hdf5_file[date][time]['LIN'].get('powers'))
            elif 'powers_swept' in list(hdf5_file[date][time]['LIN'].keys()):
                log_powers = np.asarray(hdf5_file[date][time]['LIN'].get('powers_swept'))
            log_powers = log_powers + line_attenuation
            powers = dBm_to_Watts(log_powers)
            
            x = np.linspace(0,nbar_max,100)
                        
            stark_shift = f_0 - f_0[0]
            nbar = 1/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((f_d/f_0[0])**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) )   


            p = np.polyfit(nbar,f_0,3)

            print('\n %.3f' % Flux(current,device.a,device.b) )
            print((p[0]*200**3+p[1]*200**2+p[2]*200)*1e-6)
            print(p[2]*200*1e-6)

            temp[j] = p[1]

            cmap = plt.get_cmap('rainbow_r')
            color = cmap(norm(Flux(current, device.a, device.b)))
#            color = mpl.cm.viridis(norm(Flux(current, device.a, device.b)))
            # don't plot all those points at the plato, because they will jam on the sides of the graph and it looks bad, so I select certain current ranges to plot
            if ( Flux(current,device.a,device.b)>0.08 and Flux(current,device.a,device.b)<0.11 ) or ( Flux(current,device.a,device.b)>0.31 and Flux(current,device.a,device.b)<0.48 ): 
                plt.plot( nbar, stark_shift*1e-6, color=color, linestyle='None', marker='.', ) #APS: linewidth=0.5, markersize=5.0
#                plt.plot( x, (p[0]*x**3+p[1]*x**2+p[2]*x)*1e-6, color = color, linestyle = '-') #APS: linewidth=0.5
#        plt.colorbar(ax=plt.gca())
    
        plt.tight_layout()
        cax, kw = mpl.colorbar.make_axes(plt.gca())
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, **kw)
        cb.set_label('$\Phi/\Phi_0$', rotation=90) #APS: fontsize='x-large', labelpad=2
        cb.ax.yaxis.set_label_position('left')
        fig.savefig(path+'Stark_shift_vs_nbar_paper' + ftype,dpi=240)    
    finally:
        hdf5_file.close()
    return temp 
        
      
def plot_gain_trace(device, h5date, h5time, ftype='.png', savepath = None):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[h5date][h5time]['LIN']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        if 'memory' in hdf5_file[h5date][h5time].keys():
            memGrp = hdf5_file[h5date][h5time]['memory']
            ymemr = np.array(memGrp[meas_name]['real'])
            ymemi = np.array(memGrp[meas_name]['imag'])
            ymemMag = np.abs(ymemr + 1j * ymemi)
        else: 
            ymemMag = 1
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        yr = np.array(grp[meas_name]['real'])
        yi = np.array(grp[meas_name]['imag'])
    finally:
        hdf5_file.close()
    
    yr = yr / ymemMag
    yi = yi / ymemMag
    glogmag = 20*np.log10(np.abs(yr + 1j*yi))
    # now for the figure
    fig, ax1 = plt.subplots(1, 1, figsize=(1.75,1.5), dpi=240)     #(1.75, 2.0)
    ax1.set_xlabel('Probe frequency (GHz)')
    plt.ylabel('Gain (dB)')
    plt.xlim([np.min(freqs), np.max(freqs)])
    plt.ylim([-2, 22])
    plt.plot(freqs, glogmag, color='k', linestyle='-')

    ax2 = ax1.twiny()
    ax2.set_xlim(-(max(freqs)-min(freqs))/2*1e3, (max(freqs)-min(freqs))/2*1e3)
    ax2.set_xlabel('Probe detuning (MHz)')    
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath + 'gain_' + h5date  + '_' + h5time + ftype, dpi=240)





def plot_NVR_trace(device, h5date, h5time, ftype='.png', savepath = None):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        key1 = list(hdf5_file[h5date][h5time]['noise'].keys())[0]
        grp = hdf5_file[h5date][h5time]['noise'][key1]
        memGrp = hdf5_file[h5date][h5time]['noise'][key1]['memory']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        noise_on = np.array(grp[meas_name]['logmag'])
        noise_off = np.array(memGrp[meas_name]['logmag'])
    finally:
        hdf5_file.close()
    # now for the figure
    NVR = noise_on - noise_off
    fig, ax1 = plt.subplots(1, 1, figsize=(1.75,1.5), dpi=240)     #(1.75, 2.0)
    ax1.set_xlabel('Probe frequency (GHz)')
    plt.ylabel('NVR (dB)')
    plt.xlim([np.min(freqs), np.max(freqs)])
    plt.ylim([-2, 20])
    plt.plot(freqs, NVR, color='k', linestyle='-')

    ax2 = ax1.twiny()
    ax2.set_xlim(-(max(freqs)-min(freqs))/2*1e3, (max(freqs)-min(freqs))/2*1e3)
    ax2.set_xlabel('Probe detuning (MHz)')    
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath + 'nvr_' + h5date  + '_' + h5time + ftype, dpi=240)


def plot_gain_p1dB_nvr(device, h5date, h5time, fpath=None, ftype='.png'):
    path = fpath if fpath else path
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[h5date][h5time]['LIN']
        memGrp = hdf5_file[h5date][h5time]['memory']
        powGrp = hdf5_file[h5date][h5time]['POW']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        yr = np.array(grp[meas_name]['real'])
        yi = np.array(grp[meas_name]['imag'])
        ymemr = np.array(memGrp[meas_name]['real'])
        ymemi = np.array(memGrp[meas_name]['imag'])
        pows_in = np.array(powGrp['powers'])
        powr = np.array(powGrp[meas_name]['real'])
        powi = np.array(powGrp[meas_name]['imag'])
        cwFreq = powGrp['powers'].attrs['freqINP'] *1e-9 #to GHz
        
        key2 = list(hdf5_file[h5date][h5time]['noise'].keys())[0]
        grp2 = hdf5_file[h5date][h5time]['noise'][key2]
        memGrp2 = hdf5_file[h5date][h5time]['noise'][key2]['memory']
        keys2 = grp2.keys()
        meas_name2 = [m for m in keys2 if 'CH' in m and '_' in m][0]
        freqs2 = np.array(grp2['frequencies']) * 1e-9 #to GHz
        noise_on = np.array(grp2[meas_name2]['logmag'])
        noise_off = np.array(memGrp2[meas_name2]['logmag'])
    finally:
        hdf5_file.close()
    ymemMag = np.abs(ymemr + 1j * ymemi)
    yr = yr / ymemMag
    yi = yi / ymemMag
    glogmag = 20*np.log10(np.abs(yr + 1j*yi))
    i_cw = np.argmin(np.abs(cwFreq - freqs))
    pows_in = pows_in + attenuation(freqs[i_cw]*1e9, device.data_file, 'signal_attenuation')
    pows_out = 20*np.log10(np.abs(powr + 1j*powi)) -  20*np.log10(ymemMag[i_cw]) #+0.4
    # now for the figure
    fig, ax = plt.subplots(1,3, figsize=(5,1.5), dpi=240)#(1.75, 2.0)
    ax[0].set_xlabel('Probe frequency (GHz)')
    ax[1].set_xlabel('Probe power (dBm)')
    ax[2].set_xlabel('Noise frequency (GHz)')
    ax[0].set_ylabel('Gain (dB)')
    ax[1].set_ylabel('Gain (dB)')
    ax[2].set_ylabel('NVR (dB)')
#    ax[0].set_xlim([np.min(freqs), np.max(freqs)])
#    ax[1].set_xlim([np.min(pows_in), np.max(pows_in)])
#    ax[0].set_ylim([-2, 22])
    ax[1].set_ylim([16, 21])
    ax[0].plot(freqs, glogmag, color='k', linestyle='-')
    ax[1].plot(pows_in, pows_out, color='k', linestyle='-')
    ax[0].set_xticks([6.4, 6.6, 6.8])

    NVR = noise_on - noise_off
#    ax[2].set_ylim([-2, 20])
    ax[2].set_xticks([6.4, 6.6, 6.8])
    ax[2].plot(freqs2, NVR, color='k', linestyle='-')
    
    fig.tight_layout() #w_pad=0.5
    fig.savefig(path + 'gain_p1dB_paper' + ftype, dpi=240)


def plot_gain_p1dB(device, h5date, h5time, fpath=None, ftype='.png'):
    path = fpath if fpath else path
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[h5date][h5time]['LIN']
        memGrp = hdf5_file[h5date][h5time]['memory']
        powGrp = hdf5_file[h5date][h5time]['POW']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        yr = np.array(grp[meas_name]['real'])
        yi = np.array(grp[meas_name]['imag'])
        ymemr = np.array(memGrp[meas_name]['real'])
        ymemi = np.array(memGrp[meas_name]['imag'])
        pows_in = np.array(powGrp['powers'])
        powr = np.array(powGrp[meas_name]['real'])
        powi = np.array(powGrp[meas_name]['imag'])
        cwFreq = powGrp['powers'].attrs['freqINP'] *1e-9 #to GHz
    finally:
        hdf5_file.close()
    ymemMag = np.abs(ymemr + 1j * ymemi)
    yr = yr / ymemMag
    yi = yi / ymemMag
    glogmag = 20*np.log10(np.abs(yr + 1j*yi))
    i_cw = np.argmin(np.abs(cwFreq - freqs))
    pows_in = pows_in + attenuation(freqs[i_cw]*1e9, device.data_file, 'signal_attenuation')
    pows_out = 20*np.log10(np.abs(powr + 1j*powi)) -  20*np.log10(ymemMag[i_cw]) #+0.4
    # now for the figure
    fig, ax = plt.subplots(1,2, figsize=(3.375 + 1.0/16,1.5), dpi=240)#(1.75, 2.0)
    ax[0].set_xlabel('Probe frequency (GHz)')
    ax[1].set_xlabel('Probe power (dBm)')
    ax[0].set_ylabel('Gain (dB)')
    ax[1].set_ylabel('Gain(dB)')
#    ax[0].set_xlim([np.min(freqs), np.max(freqs)])
#    ax[1].set_xlim([np.min(pows_in), np.max(pows_in)])
#    ax[0].set_ylim([-2, 22])
    ax[1].set_ylim([16, 21])
    ax[0].plot(freqs, glogmag, color='k', linestyle='-')
    ax[1].plot(pows_in, pows_out, color='k', linestyle='-')
    ax[0].set_xticks([6.4, 6.6, 6.8])
#    plt.setp( ax[0].get_xticklabels()[1::2], visible=False )
#    ax[0].locator_params(axis='x', nticks=4)
#    for a in ax:
#        plt.setp( a.get_xticklabels()[1::2], visible=False )
    fig.tight_layout() #w_pad=0.5
    fig.savefig(path + 'gain_p1dB_paper' + ftype, dpi=240)
    
   
def plot_saturation_curves(device, dates_times, fpath=None, ftype='.png'):
    
    # [('181116','230612'),('181113','232335'),('181114','094418')]
    path = fpath if fpath else path
    
    palette = plt.get_cmap('tab10')
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))    
    ax.set_xlabel('Signal power (dBm)',fontsize=10)
    ax.set_ylabel('Gain (dB)',fontsize=10)
    ax.set_ylim([12, 22])
    ax.set_xlim([-136.5, -97])

    for ind, (h5date, h5time) in enumerate(dates_times):
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[h5date][h5time]['LIN']
            memGrp = hdf5_file[h5date][h5time]['memory']
            powGrp = hdf5_file[h5date][h5time]['POW']
            keys = grp.keys()
            meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
            freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
            yr = np.array(grp[meas_name]['real'])
            yi = np.array(grp[meas_name]['imag'])
            ymemr = np.array(memGrp[meas_name]['real'])
            ymemi = np.array(memGrp[meas_name]['imag'])
            pows_in = np.array(powGrp['powers'])
            powr = np.array(powGrp[meas_name]['real'])
            powi = np.array(powGrp[meas_name]['imag'])
            cwFreq = powGrp['powers'].attrs['freqINP'] *1e-9 #to GHz
        finally:
            hdf5_file.close()
        ymemMag = np.abs(ymemr + 1j * ymemi)
        yr = yr / ymemMag
        yi = yi / ymemMag
        i_cw = np.argmin(np.abs(cwFreq - freqs))
        pows_in = pows_in + attenuation(freqs[i_cw]*1e9, device.data_file, 'signal_attenuation')
        pows_out = 20*np.log10(np.abs(powr + 1j*powi)) -  20*np.log10(ymemMag[i_cw]) #+0.4
        ax.plot(pows_in, pows_out, color=palette(ind), linestyle='-', linewidth=1)
        
        ind_p1db = np.argmin(np.abs(pows_out-pows_out[0]+1))
        ax.plot(pows_in[:ind_p1db], (pows_out[0]-1)*np.ones(ind_p1db), color=palette(ind), linewidth=0.8, linestyle='--', dashes=(5, 6))
        ax.plot(np.ones(2)*pows_in[ind_p1db], [0,pows_out[ind_p1db]], color=palette(ind), linewidth=0.8, linestyle='--', dashes=(5, 6))
        print(pows_in[ind_p1db])


    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    
    fig.tight_layout() #w_pad=0.5
    fig.savefig(path + 'gain_p1dB_paper' + ftype, dpi=240)    
    
    
def plot_sweptIMD(device, h5date, h5time, fpath=None, ftype='.png'):
    path = fpath if fpath else path
    hdf5_file = h5py.File(device.data_file,'r')
    meas_order = ['PwrMain', 'Pwr3', 'Pwr5']#, 'IM3', 'IIP3']
    labelmap = {'IIP3': r'$IIP_3$', 'IM3': '$IM_3$', 'PwrMain': '$P_1$', 'Pwr3': '$P_3$', 'Pwr5': '$P_5$'}
    try:
        grp = hdf5_file[h5date][h5time]['POW']
        keys = grp.keys()
        meas_names = [m for m in keys if 'CH' in m and '_' in m]
        xs = np.array(grp['xs']) #power sent at plane of PNAX
        ys = {}
#        labels = {}
        for meas_name in meas_names:
            meas_type = meas_name.split('_')[1]
            ys[meas_type] = np.array(grp[meas_name]['logmag'])
#            labels[meas_type] = labelmap[meas_type] if meas_type in labelmap.keys() else meas_type
        df = grp['xs'].attrs['imd_df_cw'] * 1
        fc = grp['xs'].attrs['imd_fc_cw'] * 1
    finally:
        hdf5_file.close()
    pows = xs + attenuation(fc, device.data_file, 'signal_attenuation')
    # now the figure
    fig, ax = plt.subplots(1,1, figsize=(3.375 - 0.5,3.0-0.5), dpi=240)#(1.75, 2.0)
    ax.set_xlabel('Input power (dBm)')
    ax.set_ylabel('Power (dBm)')
    for meas in meas_order:
        ax.plot(pows, ys[meas], linestyle='-', label=labelmap[meas])
    ax.legend(loc='lower right', ncol=1)
    fig.tight_layout() #w_pad=0.5
    fig.savefig(path + 'sweptIMD' + ftype, dpi=600)
    
def plot_IMDspec(device, h5date, h5time, fpath=None, ftype='.png', data_file=None):
    path = fpath if fpath else path
    hdf5_file = h5py.File(data_file if data_file else device.data_file, 'r')
    try:
        grp = hdf5_file[h5date][h5time]['NTH']
        keys = grp.keys()
        meas_names = [m for m in keys if 'CH' in m and '_' in m]
        xs = np.array(grp['xs']) #frequency 
        ys = np.array(grp[meas_names[0]]['logmag'])
    finally:
        hdf5_file.close()
    freqs = (xs - (14.9376e9/2.0))*1e-3 #freqs in kHz referenced to f_p/2
    # now the figure
    fig, ax = plt.subplots(1,1, figsize=(3.375, 2.0), dpi=240)
    ax.set_xlabel(r'$(\omega - \omega_p/2)/2\pi$ (kHz)')
    ax.set_ylabel('Power (dBm)')
    ax.plot(freqs, ys, linestyle='-', color='k')
    fig.tight_layout()
    fig.savefig(path + 'IMDspec' + ftype, dpi=600)
    
    
def plot_gain_divergence_figure():
    x = np.linspace(0,1,100)
    G = 1+4*x/(x-1)**2
    fig = plt.figure(figsize=(3.375, 2.0),dpi=240)
    plt.xlabel('Effective pump power $|4g|^2/\kappa^2$')
    plt.ylabel(r'$Gain$')
    plt.yscale('log')
    plt.ylim([1, 1e4-1])
    plt.plot(x, G, color='k', linestyle='-')
    plt.tight_layout()
#    fig.savefig('/Volumes/users/volodymyr_sivak/SPA/theory paper/Figures/fig-03.pdf', dpi=600)

    

def plot_pump_power_sweep(device, ftype='.png'):
    
    pump_powers = np.linspace(-10,20,int(30/0.01))
    currents = np.loadtxt(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JTLA/DATA/JTLA06_v3/pump_power_sweep/curr.txt')
    gains = np.loadtxt(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JTLA/DATA/JTLA06_v3/pump_power_sweep/gain_sweep.txt')    

        
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file['pump_power_sweep']
        currents = np.array(grp['current'])
        pump_powers = np.array(grp['pump_powers'])
        gains = np.array(grp['gains'])
    finally:
        hdf5_file.close()

    fig, ax = plt.subplots(1,dpi=150)
    p = ax.pcolormesh(Flux(currents*1e-3,device.a,device.b), pump_powers, np.transpose(gains), cmap='coolwarm')#,vmin=-20,vmax=20)

#    levels = [20]
#    CS = ax.contour(currents,pump_powers,np.transpose(gains),levels, colors='black')
#    ax.clabel(CS, inline=True, fontsize=4) 
#    plt.ylim(-10,20)

    ax.set_ylabel(r'${\rm Pump\, power (dBm)}$')
#    ax.set_xlabel(r'${\rm Current\,(mA)}$')
    ax.set_xlabel(r'${\Phi/\Phi_0}$')
    fig.colorbar(p, ax=ax, label=r'${\rm Gain }$')
    
    fig.savefig(device.device_folder + 'pump_power_sweep' + ftype, dpi=240)




def plot_all_gain_curves(device, sweep_name=None):
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        if sweep_name==None:
            savepath = device.device_folder + r'/nonlin_characterization/Gain_20/' + r'/all_gains/'
            grp = hdf5_file['pumping_data_20']
        else:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            savepath = device.device_folder + sweep_name + r'/all_gains/'
        date = [float_to_DateTime(x) for x in np.array(grp['date'])]
        time = [float_to_DateTime(x) for x in np.array(grp['time'])]

        for i, Date in enumerate(date):
            print(r'%d / %d' %(i,len(date)))
            plot_gain_trace(device, Date, time[i], ftype='.png', savepath = savepath)
            plt.close('all')
    finally:
        hdf5_file.close()    

def plot_all_nvr_curves(device, sweep_name=None):
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        if sweep_name==None:
            savepath = device.device_folder + r'/nonlin_characterization/Gain_20/' + r'/all_nvr/'
            grp = hdf5_file['pumping_data_20']
        else:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            savepath = device.device_folder + sweep_name + r'/all_nvr/'
        date = [float_to_DateTime(x) for x in np.array(grp['date'])]
        time = [float_to_DateTime(x) for x in np.array(grp['time'])]

        for i, Date in enumerate(date):
            print(r'%d / %d' %(i,len(date)))
            plot_NVR_trace(device, Date, time[i], ftype='.png', savepath = savepath)
            plt.close('all')
    finally:
        hdf5_file.close()

    



def plot_pump_params_sweep(device, sweep_name = None, ftype='.png', IMD = False):
    
    if sweep_name == None : sweep_name = 'pump_params_sweep'

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[sweep_name]['pump_params_sweep_20']
        signal_powers = np.array(grp['Signal_pow_array'])
        print(np.shape(signal_powers))
        gain_array = np.array(grp['Gain_array'])
        print(np.shape(gain_array))
        pump_freqs = np.array(grp['Pump_freq'])
        freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
        print(freq_c*1e-9)
        BW = np.array(grp['Bandwidth'])
        if 'IIP3' in grp.keys(): 
            print('IMD')
            IIP3 = np.array(grp['IIP3'])
            IMD_flag = True
        else:
            IMD_flag = False
        if 'NVR' in grp.keys(): 
            print('NVR')
            NVR = np.array(grp['NVR'])
            period_doubling_peak = np.array(grp['period_doubling_peak'])
            NVR_flag = True
        else:
            NVR_flag = False
    finally:
        hdf5_file.close()


    # First plot the pump frequency and power sweep, colormap of Gain

#    pump_powers = np.linspace(-15,20,int(35/0.01))
    pump_powers = np.loadtxt(device.device_folder + sweep_name + r'/pump_powers.txt')    
    pump_freqs1 = np.loadtxt(device.device_folder + sweep_name + r'/pump_freqs.txt')
    gains = np.loadtxt(device.device_folder + sweep_name  + r'/gain_sweep.txt')    

    fig, ax = plt.subplots(1,dpi=150)
    ax.set_ylabel(r'${\rm Pump\, power \,(dBm)}$')
    ax.set_xlabel(r'${\rm Pump\, detuning\, \omega_p/2-\omega_a\,(MHz)}$')
    pump_detuning1 = (pump_freqs1 - 2*freq_c)/2
    p = ax.pcolormesh(pump_detuning1*1e-6, pump_powers, np.transpose(gains), cmap='coolwarm',vmin=-40,vmax=40)
    levels = [20]
    CS = ax.contour(pump_detuning1*1e-6, pump_powers, np.transpose(gains),levels, colors='black')
    ax.clabel(CS, inline=True, fontsize=4) 
    fig.colorbar(p, ax=ax, label=r'${\rm Gain }$')
    fig.savefig(device.device_folder +  sweep_name + r'/pump_detuning_vs_pump_power_gain_colormap' + ftype, dpi=240)


    fig_s, ax_s = plt.subplots(1,dpi=150)
    ax_s.set_ylabel(r'${\rm Pump\, power \,(dBm)}$')
    ax_s.set_xlabel(r'${\rm Pump\, frequency\,(GHz)}$')
    p_s = ax_s.pcolormesh(pump_freqs1*1e-9, pump_powers, np.transpose(gains), cmap='coolwarm',vmin=-40,vmax=40)
    levels = [20]
    CS_s = ax_s.contour(pump_freqs1*1e-9, pump_powers, np.transpose(gains),levels, colors='black')
    ax_s.clabel(CS_s, inline=True, fontsize=4) 
    fig.colorbar(p_s, ax=ax_s, label=r'${\rm Gain }$')
    fig_s.savefig(device.device_folder +  sweep_name + r'/pump_freq_vs_pump_power_gain_colormap' + ftype, dpi=240)



    # Now plot the signal power sweep, colormap of Gain
    fig2, ax2 = plt.subplots(1,dpi=150)
    ax2.set_ylabel(r'${\rm Signal\, power \,(dBm)}$')
    ax2.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')
    plt.rc('xtick') #labelsize=16
    plt.rc('ytick') #labelsize=16)

    vmax = 35  #max([max(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #0
    vmin = 0 #min([min(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #25   

#    print(max( [max(gain_array[i]) for i in range(np.shape(gain_array)[0])] ))


    pump_detuning = (pump_freqs - 2*freq_c)/2
    p = ax2.pcolormesh(pump_detuning*1e-6, signal_powers[0], np.transpose(gain_array), cmap='gist_rainbow',vmin=vmin,vmax=vmax)
    fig2.colorbar(p, ax=ax2, label=r'${\rm Gain }$')
    levels = [19,21]
    CS = ax2.contour(pump_detuning*1e-6, signal_powers[0], np.transpose(gain_array),levels, colors='black')
    ax2.clabel(CS, inline=True, fontsize=4) 
    fig2.savefig(device.device_folder +  sweep_name + r'/pump_detuning_vs_signal_power_gain_colormap' + ftype, dpi=240)

    
#    signal_output_power = np.zeros(np.shape(gain_array))
#    signal_output_power = [signal_powers[0]]


    if IMD_flag:
        palette = plt.get_cmap('tab10')
        fig3, ax3 = plt.subplots(1, 1, figsize=(3.375, 2.0), dpi=240)
        ax3.set_ylabel(r'${IIP3 \,\rm (dBm)}$')
        ax3.set_xlabel(r'Pump detuning $\Delta/2\pi\rm\, (MHz)}$')
        ax3.plot(pump_detuning*1e-6, IIP3,linestyle='none',marker='.',color=palette(3))
        ax3.set_yticks([-110,-105,-100,-95])
        plt.tight_layout()
        fig3.savefig(device.device_folder + sweep_name + '/IIP3.pdf', dpi=240)

    fig4, ax4 = plt.subplots(1,dpi=150)
    ax4.set_ylabel(r'${\rm Bandwidth \,(MHz)}$')
    ax4.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')
    ax4.plot(pump_detuning*1e-6, BW*1e-6, marker='.',color='red')
    fig4.savefig(device.device_folder + sweep_name + r'/Bandwidth' + ftype, dpi=240)
    ax4_v2 = ax4.twiny()
    ax4_v2.set_xlim(pump_freqs[-1]/2*1e-9, pump_freqs[0]/2*1e-9)
    ax4_v2.set_xlabel('Center frequency (GHz)')
    

    if NVR_flag:
        fig5, ax5 = plt.subplots(1,dpi=150)
        ax5.set_ylabel(r'${\rm NVR \,(dB)}$')
        ax5.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')
        ax5.plot(pump_detuning*1e-6, NVR, marker='.',color='red')
        ax5.plot(pump_detuning*1e-6, period_doubling_peak+NVR, marker='.',color='blue')
        fig5.savefig(device.device_folder + sweep_name + '/NVR' + ftype, dpi=240)



def try_delta_effective(device,sweep_name=None):
    
    if sweep_name == None : sweep_name = 'pump_params_sweep'

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[sweep_name]['pump_params_sweep_20']
        gains = 10**(np.array(grp['Gain'])/10)
        pump_freqs = np.array(grp['Pump_freq'])
        freq_c = hdf5_file[sweep_name].attrs.get('freq_c')        
        pump_detuning = (pump_freqs - 2*freq_c)/2
        BW = np.array(grp['Bandwidth'])

        if 'nonlin_characterization' in hdf5_file.keys():
            log_g = 20
            freqs_for_calibration = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('freq_c'))
            G = np.power(10,np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('Gain'))/10)
            B = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('Bandwidth'))
            kappa_for_calibration = B * np.sqrt((G-2)*(np.sqrt(G)+1)/2/np.sqrt(G)*(1+np.sqrt((G-1)/(G-2)))) 
    finally:
        hdf5_file.close()
    
    kappa_of_omega = np.zeros(len(pump_freqs))
    for ind, f in enumerate(pump_freqs/2):
        indmin = np.argmin(np.abs(freqs_for_calibration-f))
        kappa_of_omega[ind] = kappa_for_calibration[indmin]
    
    Delta_squared = 1/4*(gains*BW**2-(kappa_of_omega)**2)
    plt.plot(pump_detuning*1e-6, np.sqrt(Delta_squared)*1e-6,label='Delta')
    plt.plot(pump_detuning*1e-6, np.abs(pump_detuning*1e-6),label='Delta')

#    plt.plot(pump_freqs/2*1e-9, np.sqrt(gains*BW**2)*1e-6,label='kappa_eff')        
#    plt.plot(pump_freqs/2*1e-9, np.sqrt(1/4*kappa_of_omega**2)*1e-6,label='kappa_calibration')
    plt.legend()
        
    
def try_all_nvr(device):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        sweep_names = [x for x in hdf5_file.keys() if 'pump_params_sweep' in x and 'curr' in x and 'dumb' not in x]
        for sweep in sweep_names:
            grp = hdf5_file[sweep]['pump_params_sweep_20']
            pump_freqs = np.array(grp['Pump_freq'])
            NVR = np.array(grp['NVR'])
            NVR_spike = np.array(grp['period_doubling_peak'])
            plt.plot(pump_freqs/2*1e-9,NVR)
            plt.plot(pump_freqs/2*1e-9,NVR+NVR_spike)
    finally:
        hdf5_file.close()
    



def plot_c_n_coeffs():
    fig, ax = plt.subplots(1,dpi=150)
    for n,m,alpha in [(3,1,0.1),(9,3,0.1)]:
        f = np.linspace(-3,3,10000)
        plt.plot( f, c2(offset(f,alpha,n=n,m=m), alpha, f, n=n, m=m) )
    

def plot_detuning_sweeps(device, sweeps, dpi=600, ftype='.png'):

    for j, sweep_name in enumerate(sweeps):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            signal_powers = np.array(grp['Signal_pow_array'])
            pump_freqs = np.array(grp['Pump_freq'])            
            gain_array = np.array(grp['Gain_array'])
            freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
            current = hdf5_file[sweep_name].attrs.get('current')
            if sweep_name == 'pump_params_sweep_curr_1290':
                pump_freqs = pump_freqs[:-2]
                gain_array = gain_array[:-2]
            if sweep_name == 'pump_params_sweep_curr_1340':
                pump_freqs = pump_freqs[:-1]
                gain_array = gain_array[:-1]
        finally:
            hdf5_file.close()

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        vmax = 35   
        vmin = 6  
    
        ax.set_ylim(-130,-97.5)    
        
        ax.set_yticks([-130,-120,-110,-100])    
        ax.set_yticklabels(['-130','-120','-110','-100'],fontsize=18)

        ax.set_xticks([-600,-500,-400,-300,-200,-100,0,100,200,300])    
        ax.set_xticklabels(['-600','-500','-400','-300','-200','-100','0','100','200','300'],fontsize=18)
        
        pump_detuning = (pump_freqs - 2*freq_c)/2

        print(np.shape(gain_array))
        print(len(pump_detuning))
        print(len(signal_powers[0]))

        p = ax.pcolormesh(pump_detuning*1e-6, signal_powers[0][35:], np.transpose(gain_array)[35:], cmap='RdYlGn_r',vmin=vmin,vmax=vmax)
        levels = [19,21]
        CS = ax.contour(pump_detuning*1e-6, signal_powers[0][45:], np.transpose(gain_array)[45:],levels, colors='black',linestyles=['solid','dashed'], linewidths=[1,1.5])
        plt.tight_layout()
        

#        f = Flux(current,device.a,device.b)
#        ax.annotate(r'$\Phi/\Phi_0=%.2f$' %f, xy=(0.03, 0.05), xycoords='axes fraction',fontsize=6)
        
 
#    cbar = fig.colorbar(p, ax=axes.ravel().tolist(),label=r'${\rm Gain\, (dB)}$',ticks=[10,15,20,25,30],location='right')
#    cbar.ax.set_yticklabels(['10', '15','20','25','30'])
    
#    plt.tight_layout()
    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig4_v3' + ftype, dpi=dpi)
 




##############################################
##############################################
##############################################
##############################################
# ---- Plot the figures for a Kerr-free paper 
##############################################
##############################################
##############################################
##############################################


def plot_stark_shift_vs_nbar_v2(device, skip=1, ftype='.png'):

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.0), dpi=240)
    ax.set_ylabel(r'$\Delta_{\rm Stark}/2\pi \;\rm (MHz)$')   #APS: 'x-large'   
    ax.set_xlabel(r'$\overline{n}\;\rm (\, 10^3\,photons)$')        #APS: 'x-large'
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:    
        currents = np.asarray(hdf5_file['fit_stark_shift'].get('current'))[::skip]
        dates = np.asarray(hdf5_file['fit_stark_shift'].get('date'))[::skip]
        times = np.asarray(hdf5_file['fit_stark_shift'].get('time'))[::skip]
        drive_freqs = np.asarray(hdf5_file['fit_stark_shift'].get('drive_freq'))[::skip]
        meas_name = hdf5_file['results'].attrs.get('flux_sweep_meas_name')        
        f0_arr = [np.asarray(hdf5_file[float_to_DateTime( dates[j] )][float_to_DateTime( times[j] )]['fits'][meas_name].get('f0')) for j in range(len(currents))]
        kc_arr = [np.asarray(hdf5_file[float_to_DateTime( dates[j] )][float_to_DateTime( times[j] )]['fits'][meas_name].get('kc')) for j in range(len(currents))]
        ki_arr = [np.asarray(hdf5_file[float_to_DateTime( dates[j] )][float_to_DateTime( times[j] )]['fits'][meas_name].get('ki')) for j in range(len(currents))]
        log_powers_arr = [np.asarray(hdf5_file[float_to_DateTime( dates[j] )][float_to_DateTime( times[j] )]['LIN'].get('powers_swept')) for j in range(len(currents))]
    finally:
        hdf5_file.close()

#        nbar_pump = np.zeros(len(currents))
#        Pump_dBm = np.zeros(len(currents))
#        Pump_stark_shift = np.zeros(len(currents))
        
        norm = mpl.colors.Normalize(vmin = min(Flux(currents,device.a,device.b)), vmax = max(Flux(currents,device.a,device.b)))
#        norm = mpl.colors.Normalize(vmin = 0, vmax = 0.5)

        cmap = plt.get_cmap('Spectral_r') #  'Dark2' 'jet'   #'tab10' 'rainbow_r' 'viridis'


    for j, current in enumerate(currents):
        
        f_d = drive_freqs[j]
        f_0 = f0_arr[j]
        k_c = kc_arr[j]
        k_i = ki_arr[j]
        k_t = k_c + k_i 
        
        line_attenuation = attenuation(f_d, device.data_file, 'drive_attenuation')     
        log_powers = log_powers_arr[j]
        log_powers = log_powers + line_attenuation
        powers = dBm_to_Watts(log_powers)
        
        
        stark_shift = f_0 - f_0[0]
        nbar = 1/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((f_d/f_0[0])**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) )


#        fudge_factor = 1#2.36
#        nbar_pump[j] = (k_c[0]/8/device.g3_distributed(Flux(current,device.a,device.b))/fudge_factor)**2    
#        P_p = float ((f_0[0])**3/k_c[0]*h*2*np.pi*nbar_pump[j]/4)
#        Pump_dBm[j] = Watts_to_dBm(P_p)
#        ind = np.argmin(np.abs(nbar-nbar_pump[j]))
#        Pump_stark_shift[j] = stark_shift[ind]

#       color = cmap(j)
        color = cmap(norm(Flux(current, device.a, device.b)))
        # don't plot all those points at the plato, because they will jam on the sides of the graph and it looks bad, so I select certain current ranges to plot
#            if ( Flux(current,device.a,device.b)>0.08 and Flux(current,device.a,device.b)<0.11 ) or ( Flux(current,device.a,device.b)>0.31 and Flux(current,device.a,device.b)<0.48 ): 
#                plt.plot( nbar, stark_shift*1e-6, color=color, linestyle='None', marker='.', ) #APS: linewidth=0.5, markersize=5.0
        plt.plot(nbar*1e-3, stark_shift*1e-6, linestyle='-', color=color, marker='.', markersize=2.5, markeredgewidth=0.01, markeredgecolor='grey') #APS: linewidth=0.5, markersize=5.0
#        plt.plot(nbar_pump*1e-3, Pump_stark_shift*1e-6,color='black',linestyle='None',marker='.',markersize=4)

    plt.xlim(0,34)
    plt.xticks(np.linspace(0,30,4))
    plt.tight_layout()


    
    cax, kw = mpl.colorbar.make_axes(plt.gca())
#        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, **kw)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=[0.25, 0.30, 0.35, 0.40, 0.45], **kw)
    cb.ax.set_yticklabels(['0.25','0.30','0.35','0.40','0.45'])
    cb.set_label('$\Phi/\Phi_0$', rotation=90) #APS: fontsize='x-large', labelpad=2
    cb.ax.yaxis.set_label_position('left')


    left, bottom, width, height = [0.55, 0.64, 0.225, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.xticks(np.linspace(0,20,3))

    for j, current in enumerate(currents):
        
        f_d = drive_freqs[j]
        f_0 = f0_arr[j]
        k_c = kc_arr[j]
        k_i = ki_arr[j]
        k_t = k_c + k_i 
        
        line_attenuation = attenuation(f_d, device.data_file, 'drive_attenuation')     
        log_powers = log_powers_arr[j]
        log_powers = log_powers + line_attenuation
        powers = dBm_to_Watts(log_powers)
        
        
        stark_shift = f_0 - f_0[0]
        nbar = 1/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((f_d/f_0[0])**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) )

#       color = cmap(j)
        color = cmap(norm(Flux(current, device.a, device.b)))
#         don't plot all those points at the plato, because they will jam on the sides of the graph and it looks bad, so I select certain current ranges to plot
        if  Flux(current,device.a,device.b)>0.35 and Flux(current,device.a,device.b)<0.41 : 
            plt.plot( nbar*1e-3, stark_shift*1e-6, linestyle='none', color=color, marker='.', markersize=2.5, markeredgewidth=0.01, markeredgecolor='grey') #APS: linewidth=0.5, markersize=5.0

    ax2.set_ylim(-50,50)
    ax2.set_xlim(0,20)

    plt.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/Stark-fig.pdf', dpi=240)





def plot_Kerr_free_figure(device, sweeps, dpi=600, ftype='.png'):

    fig, axes = plt.subplots(2, 2, sharey='row', figsize=(3.375, 4.2))
    
    topSubplot = fig.add_subplot(111, frameon=False)
    topSubplot.set_position([0.09,0.32,0.9,0.55])
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.ylabel("Input signal power (dBm)")
#    plt.xlabel(r'Pump detuning $(\omega_p/2-\omega_a)/2\pi$ (MHz)')
    plt.xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')

    for j, sweep_name in enumerate(sweeps):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            signal_powers = np.array(grp['Signal_pow_array'])
            pump_freqs = np.array(grp['Pump_freq'])            
            gain_array = np.array(grp['Gain_array'])
            freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
            current = hdf5_file[sweep_name].attrs.get('current')
            if sweep_name == 'pump_params_sweep_curr_1290':
                pump_freqs = pump_freqs[:-2]
                gain_array = gain_array[:-2]
            if sweep_name == 'pump_params_sweep_curr_1340':
                pump_freqs = pump_freqs[:-1]
                gain_array = gain_array[:-1]            
        finally:
            hdf5_file.close()
    
        f = Flux(current,device.a,device.b)
#        i_x = 0 if j<3 else 1
#        i_y = j if j<3 else j-3

        i_x = 0 if j<2 else 1
        i_y = j if j<2 else j-2

        ax = axes[i_x][i_y]
            
        vmax = 35   #30     #max([max(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #0
        vmin = 6    #10     #min([min(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #25   
    
#        ax.set_ylim(-136.7,-97.5)
        ax.set_ylim(-130,-97.5)        
        
        pump_detuning = (pump_freqs - 2*freq_c)/2

        print(np.shape(gain_array))
        print(len(pump_detuning))
        print(len(signal_powers[0]))

        p = ax.pcolormesh(pump_detuning*1e-6, signal_powers[0][35:], np.transpose(gain_array)[35:], cmap='RdYlGn_r',vmin=vmin,vmax=vmax)
        levels = [19,21]
        CS = ax.contour(pump_detuning*1e-6, signal_powers[0][45:], np.transpose(gain_array)[45:],levels, colors='black',linestyles=['solid','dashed'], linewidths=[0.4,0.5])

#        ax.text(0.5,0.5,'%.2f' %Flux(current,device.a,device.b),fontsize='x-small',transform=fig.transFigure)

        ax.annotate(r'$\Phi/\Phi_0=%.2f$' %f, xy=(0.55, 0.05), xycoords='axes fraction',fontsize=6)

#    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat],orientation='horisontal',location='top')
#    cbar = plt.colorbar(p, cax=cax, **kw,ticks=[10,15,20,25,30])
    
    cbar = fig.colorbar(p, ax=axes.ravel().tolist(),label=r'${\rm Gain\, (dB)}$',ticks=[10,15,20,25,30],location='bottom')
    cbar.ax.set_yticklabels(['10', '15','20','25','30'])

    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig4' + ftype, dpi=dpi)
    
    
    
def plot_Kerr_free_figure_v2(device, sweeps, dpi=600, ftype='.png'):

    fig, axes = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(6.6, 2.5), gridspec_kw = {'height_ratios':[3, 1],'width_ratios':[680000000, 650000000,585000000,535000000,290000000]})

    palette = plt.get_cmap('Set1')
    
#    topSubplot = fig.add_subplot(111, frameon=False)
#    topSubplot.set_position([0.09,0.32,0.9,0.55])
#    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#    plt.grid(False)
#    plt.ylabel("Input signal power (dBm)")
##    plt.xlabel(r'Pump detuning $(\omega_p/2-\omega_a)/2\pi$ (MHz)')
#    plt.xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')

    for j, sweep_name in enumerate(sweeps):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            signal_powers = np.array(grp['Signal_pow_array'])
            pump_freqs = np.array(grp['Pump_freq'])            
            gain_array = np.array(grp['Gain_array'])
            freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
            current = hdf5_file[sweep_name].attrs.get('current')
            NVR = np.array(grp['NVR'])
            period_doubling_peak = np.array(grp['period_doubling_peak'])
            if sweep_name == 'pump_params_sweep_curr_1290':
                pump_freqs = pump_freqs[:-2]
                gain_array = gain_array[:-2]
                NVR = NVR[:-2]
                period_doubling_peak = period_doubling_peak[:-2]
            if sweep_name == 'pump_params_sweep_curr_1340':
                pump_freqs = pump_freqs[:-1]
                gain_array = gain_array[:-1]            
                NVR = NVR[:-1]
                period_doubling_peak = period_doubling_peak[:-1]
        finally:
            hdf5_file.close()
    
        f = Flux(current,device.a,device.b)




        ax = axes[0][j]

        if not j:
            ax.set_ylabel("Input power (dBm)")
            ax.yaxis.set_label_coords(-0.25, 0.47)
            
        ax.set_yticks([-130,-120,-110,-100])    
#        ax.set_yticklabels(['$1.5\cdot10^8$','$2.5\cdot10^8$'])  
            
        vmax = 35   #30     #max([max(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #0
        vmin = 6    #10     #min([min(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #25   
    
#        ax.set_ylim(-136.7,-97.5)
        ax.set_ylim(-130,-97.5)        
        
        pump_detuning = (pump_freqs - 2*freq_c)/2

        print(np.shape(gain_array))
        print(len(pump_detuning))
        print(len(signal_powers[0]))

        p = ax.pcolormesh(pump_detuning*1e-6, signal_powers[0][35:], np.transpose(gain_array)[35:], cmap='RdYlGn_r',vmin=vmin,vmax=vmax)
        levels = [19,21]
        CS = ax.contour(pump_detuning*1e-6, signal_powers[0][45:], np.transpose(gain_array)[45:],levels, colors='black',linestyles=['solid','dashed'], linewidths=[0.4,0.5])

#        ax.text(0.5,0.5,'%.2f' %Flux(current,device.a,device.b),fontsize='x-small',transform=fig.transFigure)

        ax.annotate(r'$\Phi/\Phi_0=%.2f$' %f, xy=(0.03, 0.05), xycoords='axes fraction',fontsize=6)

        ax = axes[1][j]

        if not j:
            ax.set_ylabel("NVR (dB)")
            ax.yaxis.set_label_coords(-0.25, 0.47)

#        ax.set_yticks([0,10,20])
#        ax.set_ylim(0,27)
        ax.set_yticks([0,5,10,15])
        ax.set_ylim(0,15)
        
        ax.set_xlim(pump_detuning[-1]*1e-6,pump_detuning[0]*1e-6)

        ax.plot(pump_detuning*1e-6, NVR, marker='.',color=palette(2),linestyle='none', markersize=2, label=r'At $\omega = B_{\rm res}$') #label='Top of Lorenzian'
#        ax.plot(pump_detuning*1e-6, period_doubling_peak+NVR, marker='.',color=palette(4),linestyle='none',markersize=2, label=r'At $\omega=0$') #label='Spike above Lorenzian'
        ax.plot(pump_detuning*1e-6, 8.3*np.ones(len(pump_detuning)), color='black',linestyle='--') #label='Quantum limit'

        print(max(pump_detuning)-min(pump_detuning))

    plt.tight_layout()
    
#    cbar = fig.colorbar(p, ax=axes.ravel().tolist(),label=r'${\rm Gain\, (dB)}$',ticks=[10,15,20,25,30],location='bottom')
#    cbar.ax.set_yticklabels(['10', '15','20','25','30'])
    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig4_v2' + ftype, dpi=dpi)
     
    

def plot_Kerr_free_figure_v3(device, sweeps, dpi=600, ftype='.png'):

    n_sweeps = len(sweeps)
    fig, axes = plt.subplots(1, n_sweeps, sharex='col', sharey='row', figsize=(7, 1.8), gridspec_kw = {'width_ratios':[680000000, 650000000,585000000,535000000,290000000]})

    palette = plt.get_cmap('Set1')
    
#    topSubplot = fig.add_subplot(111, frameon=False)
#    topSubplot.set_position([0.5,0.5,1.0,1.0])
#    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#    plt.grid(False)

    for j, sweep_name in enumerate(sweeps):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[sweep_name]['pump_params_sweep_20']
            signal_powers = np.array(grp['Signal_pow_array'])
            pump_freqs = np.array(grp['Pump_freq'])            
            gain_array = np.array(grp['Gain_array'])
            freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
            current = hdf5_file[sweep_name].attrs.get('current')
            NVR = np.array(grp['NVR'])
            period_doubling_peak = np.array(grp['period_doubling_peak'])
            if sweep_name == 'pump_params_sweep_curr_1290':
                pump_freqs = pump_freqs[:-2]
                gain_array = gain_array[:-2]
            if sweep_name == 'pump_params_sweep_curr_1340':
                pump_freqs = pump_freqs[:-1]
                gain_array = gain_array[:-1]            
        finally:
            hdf5_file.close()
    
        f = Flux(current,device.a,device.b)


        ax = axes[j]
        
        if not j:
            ax.set_ylabel("Input power (dBm)")
            ax.set_xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')
            
        vmax = 35   #30     #max([max(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #0
        vmin = 6    #10     #min([min(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #25   
    
#        ax.set_ylim(-136.7,-97.5)
        ax.set_ylim(-130,-97.5)        
        
        pump_detuning = (pump_freqs - 2*freq_c)/2

        print(np.shape(gain_array))
        print(len(pump_detuning))
        print(len(signal_powers[0]))

        p = ax.pcolormesh(pump_detuning*1e-6, signal_powers[0][35:], np.transpose(gain_array)[35:], cmap='RdYlGn_r',vmin=vmin,vmax=vmax)
        levels = [19,21]
        CS = ax.contour(pump_detuning*1e-6, signal_powers[0][45:], np.transpose(gain_array)[45:],levels, colors='black',linestyles=['solid','dashed'], linewidths=[0.4,0.5])

#        ax.text(0.5,0.5,'%.2f' %Flux(current,device.a,device.b),fontsize='x-small',transform=fig.transFigure)

        ax.annotate(r'$\Phi/\Phi_0=%.2f$' %f, xy=(0.03, 0.05), xycoords='axes fraction',fontsize=6)
        
 
#    cbar = fig.colorbar(p, ax=axes.ravel().tolist(),label=r'${\rm Gain\, (dB)}$',ticks=[10,15,20,25,30],location='right')
#    cbar.ax.set_yticklabels(['10', '15','20','25','30'])
    
#    plt.tight_layout()
    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig4_v3' + ftype, dpi=dpi)
 
    
    
def plot_IMD_at_different_gains(device, dpi=240, ftype='.png'):

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.2)) 
    plt.xticks(np.linspace(0,0.5,6))
    plt.yticks(np.linspace(-110,-70,5))
    plt.ylabel(r'$ IIP_3 \;  \rm (dBm)$')
    plt.xlabel(r'$\Phi/\Phi_0$')
    line_style = 'None'

    palette = plt.get_cmap('tab10')  #'Set1'

    hdf5_file = h5py.File(device.data_file, 'r')    
    try:
        if 'nonlin_characterization' in hdf5_file.keys():
            gains = np.asarray(hdf5_file['nonlin_characterization'].get('gains'))
            colors = ['purple','blue','red','green','orange'] if len(gains)<=5 else mpl.cm.rainbow(np.linspace(0,1,len(gains)))
            for i, log_g in enumerate(gains):
                print(log_g)
                IIP3 = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('IIP3'))
                P1dB = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('P_1dB'))
                current = np.asarray(hdf5_file['pumping_data_'+str(int(log_g))].get('current'))
                Ind_bad_current = np.argmin(np.abs(current-1.455e-3))
                flux_IMD = Flux(current,device.a, device.b)
                plt.plot(np.delete(flux_IMD,Ind_bad_current), np.delete(IIP3,Ind_bad_current), color=palette(i), linestyle='None', marker='.', label='%.0f dB' %log_g)
                
                if log_g==0:
                    freq_c = device.Freq(flux_IMD)
                    K = 12*(device.g4_distributed(flux_IMD)) ## this already has the correction -5*g3**2/freq_c, because this g4 is from the distributed model !!!
                    kappa = device.kappa(flux_IMD)
                    G=1
                    IIP3_theor = Watts_to_dBm(kappa**2*freq_c/np.abs(K)*(2*np.pi*h)*( 1/(np.sqrt(G)+1) )**3)
                    plt.plot(flux_IMD,IIP3_theor)
                    
        plt.legend(loc = 'lower left',title='Gain')#,fontsize='x-small')
    finally:
        hdf5_file.close()
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'IMD' + ftype, dpi=dpi)




# TODO: add legends to all graphs
# TODO: Come up with the legend for g4 which is not actually g4 but rather K
# TODO: Compile everything together


def plot_Hamiltonian_parameters(device, dpi=240, ftype='.png'):

    palette = plt.get_cmap('Set1')  #'tab10'    
    
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 1.2))
    plt.xticks(np.linspace(-0.5,0.5,11))
    plt.ylabel(r'$\rm Frequency \;  \rm (Hz)$')
    plt.xlabel(r'$\Phi/\Phi_0$')
    plt.yscale('log')
    plt.ylim(5e1,2e7)
    plt.xlim(-0.75,0.75)
    ax.set_yticks([1e2,1e3,1e4,1e5,1e6])    
#    ax.set_yticklabels(['$1.5\cdot10^8$','$2.5\cdot10^8$'])  
    ax.set_xticks([-0.75,-0.5,-0.25,0,0.25,0.5,0.75])  

    fluxes = np.linspace(-0.75,0.75,2000)    
    plt.plot(fluxes, np.abs(device.g3_distributed(fluxes)),color=palette(3))
    plt.plot(fluxes, np.abs(device.g4_distributed(fluxes) + 0*(device.g3_distributed(fluxes))**2/device.Freq(fluxes)),color=palette(2))


    """ Now plot g3 """  
    hdf5_file = h5py.File(device.data_file, 'r')
    try:
        g3_Data = np.abs(np.asarray(hdf5_file['g3_fit'].get('g3')))
        Flux_Data = np.asarray(hdf5_file['g3_fit'].get('fluxes'))
        plt.plot(list(Flux_Data)[0:-1:4], list(np.abs(g3_Data))[0:-1:4], color=palette(3), marker='.',linestyle='None',label=r'$|g_3|/2\pi$')
    finally:
        hdf5_file.close()


    """ Now plot g4 """  
    hdf5_file = h5py.File(device.data_file, 'r')
    try: 
        if 'fit_stark_shift' in hdf5_file.keys():
            Flux_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('flux'))
            g4_Data_Stark = np.asarray(hdf5_file['fit_stark_shift'].get('g4'))
            g4_corrected = g4_Data_Stark + 0*(device.g3_distributed(Flux_Data_Stark))**2/device.Freq(Flux_Data_Stark)
            ind = Flux_Data_Stark.argsort()
            plt.plot(list(Flux_Data_Stark[ind])[0:-1:1], list(np.abs(g4_corrected[ind]))[0:-1:1], color=palette(4), linestyle='None', marker='.', label=r'$|g_4^*|/2\pi$, Stark shift')   
        group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_0'
        flux_IMD = Flux(np.asarray(hdf5_file[group].get('current')),device.a, device.b)
        g4_IMD = np.asarray(hdf5_file[group].get('g4_IMD'))
        g4_IMD = g4_IMD + 0*(device.g3_distributed(flux_IMD))**2/device.Freq(flux_IMD)
        plt.plot(list(flux_IMD)[0:-1:4], list(np.abs(g4_IMD))[0:-1:4], color=palette(2), linestyle='None', marker='.', label=r'$|g_4^*|/2\pi$, IMD')
    finally:
        hdf5_file.close()
    plt.legend(loc='best',frameon=False,ncol=3)
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'ham_param1' + ftype, dpi=dpi)

    
    """ Now plot Kappa """    
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 0.4))     
    plt.xlim(-0.75,0.75)
    hdf5_file = h5py.File(device.data_file, 'r')
    try:                        
        fluxes_kappa = Flux(hdf5_file['fit_flux_sweep']['currents'], device.a, device.b)
        kc_data = hdf5_file['fit_flux_sweep']['kc_exp_fit']
        plt.plot(list(fluxes_kappa)[0:-1:7], list(np.asarray(kc_data))[0:-1:7], color=palette(1), linewidth=1.0, marker='.', linestyle='None', label=r'$\kappa/2\pi$',zorder=1)
        plt.plot(fluxes, device.kappa(fluxes), color=palette(1), linestyle='-', zorder=1)        
    finally:
        hdf5_file.close()
    ax.set_yticks([1.5e8,2.5e8])    
    ax.set_yticklabels(['$1.5\cdot10^8$','$2.5\cdot10^8$'])    
    ax.set_xticks([])
    plt.legend(loc='best',frameon=False)
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'ham_param2' + ftype, dpi=dpi)
  
    
    """ Now plot Resonance Frequency """      
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 0.4))
    plt.xlim(-0.75,0.75)
    plt.ylim(6e9,7.6e9)
    hdf5_file = h5py.File(device.data_file, 'r')
    try:
        I_data = hdf5_file['fit_flux_sweep']['currents']
        f0_data = hdf5_file['fit_flux_sweep']['f0_exp_fit']
        flux_data = Flux(I_data,device.a,device.b)
        flux_array = np.linspace(flux_data[0]-1/4,flux_data[len(I_data)-1]+1/4,2*len(I_data))
        plt.plot(list(flux_data)[0:-1:7], list(np.asarray(f0_data))[0:-1:7], color=palette(0), marker='.', linestyle='none', label=r'$\omega_a/2\pi$',zorder=1)        
        plt.plot(fluxes, device.Freq(fluxes), color=palette(0), linestyle='-')        
    finally:
        hdf5_file.close()
    ax.set_yticks([6.0e9,7.0e9])
    ax.set_yticklabels(['$6.0\cdot10^9$','$7.0\cdot10^9$'])    
    ax.set_xticks([])
    plt.legend(loc='best',frameon=False)
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'ham_param3' + ftype, dpi=dpi)




def plot_pump_power_sweep_v2(device, dpi=240, ftype='.png'):
    
#    pump_powers = np.linspace(-10,20,int(30/0.01))
    
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file['pump_power_sweep']
        currents = np.array(grp['current'])
        pump_powers = np.array(grp['pump_powers'])
        gains = np.array(grp['gains'])
    finally:
        hdf5_file.close()

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.2)) 
    p = ax.pcolormesh(Flux(currents*1e-3,device.a,device.b), pump_powers, np.transpose(gains), cmap='coolwarm')#,vmin=-20,vmax=20)

#    levels = [20]
#    CS = ax.contour(currents,pump_powers,np.transpose(gains),levels, colors='black')
#    ax.clabel(CS, inline=True, fontsize=4) 
#    plt.ylim(-10,20)

    ax.set_ylabel(r'${\rm Pump\, power\, (dBm)}$')
    plt.xticks(np.linspace(0,0.5,6))
    ax.set_xlabel(r'${\Phi/\Phi_0}$')
    fig.colorbar(p, ax=ax, label=r'${\rm Gain }$')
    
    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/' + 'fig3' + ftype, dpi=dpi)
   
    
def plot_phase_diagram(device, sweep_name=None, dpi=240, ftype='.png'):
    

    fig, axes = plt.subplots(3, 1, figsize=(3.375, 4.6), sharex=True)


    """ Plot NVR and signal power sweep """
    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[sweep_name]['pump_params_sweep_20']
        
        dates = np.array(grp['date'])
        times = np.array(grp['time'])
        current = hdf5_file[sweep_name].attrs.get('current')       
        signal_powers = np.array(grp['Signal_pow_array'])
        print(np.shape(signal_powers))
        gain_array = np.array(grp['Gain_array'])
        print(np.shape(gain_array))
        pump_freqs = np.array(grp['Pump_freq'])
        freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
        NVR = np.array(grp['NVR'])
        period_doubling_peak = np.array(grp['period_doubling_peak'])
        pump_detuning = (pump_freqs - 2*freq_c)/2
    finally:
        hdf5_file.close()
    
    """ Plot NVR  """
    ax = axes[1]
    
    ax.set_xlim(pump_detuning[-1]*1e-6,pump_detuning[0]*1e-6)
    
    palette = plt.get_cmap('Set1') #Dark2
    ax.set_ylim(0,27)
    ax.set_ylabel(r'${\rm NVR \,(dB)}$')
    ax.yaxis.set_label_coords(-0.1, 0.47)
#    ax.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')
    ax.plot(pump_detuning*1e-6, NVR, marker='.',color=palette(2),linestyle='none', markersize=2, label=r'At $\omega = B_{\rm res}$') #label='Top of Lorenzian'
    ax.plot(pump_detuning*1e-6, period_doubling_peak+NVR, marker='.',color=palette(4),linestyle='none',markersize=2, label=r'At $\omega=0$') #label='Spike above Lorenzian'
    ax.plot(pump_detuning*1e-6, 8.5*np.ones(len(pump_detuning)), color='black',linestyle='--') #label='Quantum limit'
#    ax.legend(ncol=2,loc='lower left',frameon=False, bbox_to_anchor=(0.0, -0.045))
#    ax.legend(ncol=2,loc='lower left',frameon=False, bbox_to_anchor=(0.0, 0.05))
    ax.legend(loc='lower center',frameon=False,bbox_to_anchor=(0.5, -0.05))

    """ Plot NVR inset #1 """
    left, bottom, width, height = [0.48, 0.51, 0.16, 0.08]
    inset1 = fig.add_axes([left, bottom, width, height])

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        h5date = float_to_DateTime(dates[103])
        h5time = float_to_DateTime(times[103])
        key1 = list(hdf5_file[h5date][h5time]['noise'].keys())[0]
        grp = hdf5_file[h5date][h5time]['noise'][key1]
        memGrp = hdf5_file[h5date][h5time]['noise'][key1]['memory']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        noise_on = np.array(grp[meas_name]['logmag'])
        noise_off = np.array(memGrp[meas_name]['logmag'])
    finally:
        hdf5_file.close()
    # now for the figure
    NVR = noise_on - noise_off
#    plt.ylabel('NVR (dB)')
    this_detuning = (np.average(freqs)-freq_c*1e-9)*1e3
    new_detuning = (freqs-np.average(freqs))*1e3
    inset1.set_xlim([np.min(new_detuning), np.max(new_detuning)])
    inset1.set_ylim([-2, 20])
    inset1.set_yticks([0,10,20])
#    inset1.set_title(r'$\Delta=%.0f\,\rm MHz$'%this_detuning, fontsize=6)#, pad=0.1)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5)
    inset1.plot(new_detuning, NVR, color='k', linestyle='-')
#    inset1.set_xlabel('Noise frequency',fontsize='xx-small')
    inset1.xaxis.set_label_coords(1.2, -0.3)
    inset1.set_xlabel(r'$\omega/2\pi\,\rm (MHz)$',fontsize='xx-small')
    
    ax.scatter([this_detuning],[1.5],marker=11,color=palette(0))
    ax.scatter([this_detuning],[1.5],marker=2,color=palette(0))   
    inset1.scatter([-200],[17],marker='s',color=palette(0))
    
#    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    """ Plot NVR inset #2 """
    
    left, bottom, width, height = [0.48 + 0.16 + 0.07, 0.51, 0.16, 0.08]
    inset2 = fig.add_axes([left, bottom, width, height])

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        h5date = float_to_DateTime(dates[30])
        h5time = float_to_DateTime(times[30])
        key1 = list(hdf5_file[h5date][h5time]['noise'].keys())[0]
        grp = hdf5_file[h5date][h5time]['noise'][key1]
        memGrp = hdf5_file[h5date][h5time]['noise'][key1]['memory']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        noise_on = np.array(grp[meas_name]['logmag'])
        noise_off = np.array(memGrp[meas_name]['logmag'])
    finally:
        hdf5_file.close()
    # now for the figure
    NVR = noise_on - noise_off
    this_detuning = (np.average(freqs)-freq_c*1e-9)*1e3
#    ax1.set_xlabel('Probe frequency (GHz)')
#    plt.ylabel('NVR (dB)')
    new_detuning = (freqs-freq_c*1e-9)*1e3
    inset2.set_xlim([np.min(new_detuning), np.max(new_detuning)])
    plt.xticks(fontsize=5)
    inset2.set_ylim([-2, 20])
#    inset2.set_title(r'$\Delta=%.0f\,\rm MHz$'%this_detuning, fontsize=6)
    inset2.plot(new_detuning, NVR, color='k', linestyle='-')
#    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax.scatter([this_detuning],[1.5],marker=11,color=palette(1))
    ax.scatter([this_detuning],[1.5],marker=2,color=palette(1))   
    inset2.scatter([-200],[17],marker='s',color=palette(1))

    """ Plot signal power sweep """
    ax = axes[2]
    ax.set_ylabel(r'${\rm Signal\, power \,(dBm)}$')
    ax.yaxis.set_label_coords(-0.1, 0.47)
    signal_cutoff = 35
#    ax2.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')
    p = ax.pcolormesh(pump_detuning*1e-6, signal_powers[0][signal_cutoff:], np.transpose(gain_array)[signal_cutoff:], cmap='RdYlGn_r',vmin=6,vmax=35)

#    cb = plt.colorbar(p, ax=ax,label=r'${\rm Gain }$',shrink=0.45,aspect=5)
#    cb.set_ticks([10,20,30])
#    cb.set_label('Gain (dB)',fontsize='x-small')
    levels = [19,21]
    CS = ax.contour(pump_detuning*1e-6, signal_powers[0][45:], np.transpose(gain_array)[45:],levels, colors='black',linestyles=['solid','dashed'])

    ax.clabel(CS, inline=True, fontsize=4.5, fmt='%d',inline_spacing=8) 
    ax.set_xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)') 
    ax.xaxis.set_label_coords(0.5, -0.15)
    f = Flux(current, device.a, device.b)
    ax.annotate(r'$\Phi/\Phi_0=%.2f$' %f, xy=(0.78, 0.07), xycoords='axes fraction',fontsize=6)



    """ Plot the phase diagram """
    # parameters of the system
    f = Flux(1.27e-3,device.a,device.b)
    g3 = device.g3_distributed(f)
    g4 = device.g4_distributed(f) - (28*3/32-5)*g3**2/freq_c 
    ## !!!!!!! This g4 includes the correction from second order harmonic balance, do I need it ?
    kappa = device.kappa(f)
    Delta = np.linspace(-pump_detuning[-1],-pump_detuning[0],1e4)
    Delta = np.linspace(-175e6,-pump_detuning[-1],1e4)

    ax = axes[0]
    ax.set_yscale('log')
    palette = plt.get_cmap('Pastel2')
    lower_bound = 8e2 #8e2
    upper_bound = 2e4 #2e4
    ax.set_ylim(lower_bound,upper_bound)
#    ax.set_xlim(-600e6,200e6)
    lines_color = 'grey'
    ax.set_ylabel(r'$\rm Pump \; photons$')
    ax.yaxis.set_label_coords(-0.1, 0.47)


    # define the lower and upper infinite gain lines -- lines that bound the bistable region
    L1 = (-(Delta*g4*64/3-16*g3**2) - np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    L2 = (-(Delta*g4*64/3-16*g3**2) + np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    ax.plot(-Delta*1e-6,L1,color=lines_color)
    ax.plot(-Delta*1e-6,L2,color=lines_color)
    
    # define the range of detunings and the line that bounds the tristable region from below
    n_p_low_low = kappa**2/64/g3**2
    ind = np.where(Delta > -g4*32/3*n_p_low_low)[0]
    Delta_low_low = Delta[ind]
    L3 = n_p_low_low*np.ones(len(Delta_low_low))
    ax.plot(-Delta_low_low*1e-6, L3, color=lines_color)

    # fill different areas of phase diagram with different colors
    ax.fill_between(-Delta*1e-6, lower_bound*np.ones(len(Delta)), upper_bound*np.ones(len(Delta)), facecolor=palette(0), interpolate=True)  
    ax.fill_between(-Delta_low_low*1e-6, L1[ind], L3, where=L1[ind] >= L3, facecolor=palette(1), interpolate=True)
    ind2 = np.where(L1>0)
    Delta2 = Delta[ind2]
    print(len(Delta))
    print(len(Delta2))
    ax.fill_between(-Delta2*1e-6, L1[ind2], L2[ind2], where=L2[ind2] >= L1[ind2], facecolor=palette(2), interpolate=True)

    # Plot the 20 dB gain line
    def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
        g = 2*g3_*np.sqrt(n_p)   
        Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i )
        return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2

    nbar_pump_sweep = np.linspace(1,upper_bound,1000)
    n_p = np.empty(0)
    pump_detuning = np.empty(0)
    for i, delta in enumerate(Delta):
        gains = Gain(nbar_pump_sweep,0,0,g3, g4, kappa, -delta)
        gains = 10*np.log10(gains)
        count = 0
        while gains[count]<20 and count<len(gains)-1: 
            count +=1
        if count<len(gains)-1:
            n_p = np.append(n_p, nbar_pump_sweep[count])
            pump_detuning = np.append(pump_detuning, -delta)
    ax.plot(pump_detuning*1e-6, n_p, color='black')

    ax.annotate(r'$\rm (I)$', xy=(114, 3e3), fontsize=6)
    ax.annotate(r'$\rm (II)$', xy=(-100, 5e3), fontsize=6)
    ax.annotate(r'$\rm (III)$', xy=(-450, 2e3), fontsize=6)
    
#    ax.set_xlim(-600e6,pump_detuning[0])
#    ax.tight_layout()
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig3_new' + ftype, dpi=dpi)




def plot_phase_diagram_v2(device, sweep_name=None, dpi=240, ftype='.png'):
    
    fig, ax = plt.subplots(figsize=(3.375, 1.5))


    """ Plot the phase diagram """
    # parameters of the system
    f = Flux(1.27e-3,device.a,device.b)
    g3 = device.g3_distributed(f)
    g4 = device.g4_distributed(f) - (28*3/32-5)*g3**2/freq_c 
    ## !!!!!!! This g4 includes the correction from second order harmonic balance, do I need it ?
    kappa = device.kappa(f)
    Delta = np.linspace(-175e6,600e6,1e4)

    ax.set_yscale('log')
    palette = plt.get_cmap('Pastel2')
    lower_bound = 6e2 #8e2
    upper_bound = 2e4 #2e4
    ax.set_ylim(lower_bound,upper_bound)
    ax.set_xlim(-600,175)
    lines_color = 'grey'
    ax.set_ylabel(r'$\rm Pump \; photons$')
#    ax.yaxis.set_label_coords(-0.1, 0.47)
    ax.set_xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')


    # define the lower and upper infinite gain lines -- lines that bound the bistable region
    L1 = (-(Delta*g4*64/3-16*g3**2) - np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    L2 = (-(Delta*g4*64/3-16*g3**2) + np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    ax.plot(-Delta*1e-6,L1,color=lines_color)
    ax.plot(-Delta*1e-6,L2,color=lines_color)
    
    # define the range of detunings and the line that bounds the tristable region from below
    n_p_low_low = kappa**2/64/g3**2
    ind = np.where(Delta > -g4*32/3*n_p_low_low)[0]
    Delta_low_low = Delta[ind]
    L3 = n_p_low_low*np.ones(len(Delta_low_low))
    ax.plot(-Delta_low_low*1e-6, L3, color=lines_color)

    # fill different areas of phase diagram with different colors
    ax.fill_between(-Delta*1e-6, lower_bound*np.ones(len(Delta)), upper_bound*np.ones(len(Delta)), facecolor=palette(0), interpolate=True)  
    ax.fill_between(-Delta_low_low*1e-6, L1[ind], L3, where=L1[ind] >= L3, facecolor=palette(1), interpolate=True)
    ind2 = np.where(L1>0)
    Delta2 = Delta[ind2]
    print(len(Delta))
    print(len(Delta2))
    ax.fill_between(-Delta2*1e-6, L1[ind2], L2[ind2], where=L2[ind2] >= L1[ind2], facecolor=palette(2), interpolate=True)

    # Plot the 20 dB gain line
    def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
        g = 2*g3_*np.sqrt(n_p)   
        Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i )
        return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2

    nbar_pump_sweep = np.linspace(1,upper_bound,1e4)
    n_p = np.empty(0)
    pump_detuning = np.empty(0)
    for i, delta in enumerate(Delta):
        gains = Gain(nbar_pump_sweep,0,0,g3, g4, kappa, -delta)
        gains = 10*np.log10(gains)
        count = 0
        while gains[count]<20 and count<len(gains)-1: 
            count +=1
        if count<len(gains)-1:
            n_p = np.append(n_p, nbar_pump_sweep[count])
            pump_detuning = np.append(pump_detuning, -delta)
    ax.plot(pump_detuning*1e-6, n_p, color='black')

    ax.annotate(r'$\rm (I)$', xy=(114, 3e3), fontsize=6)
    ax.annotate(r'$\rm (II)$', xy=(-100, 5e3), fontsize=6)
    ax.annotate(r'$\rm (III)$', xy=(-450, 2e3), fontsize=6)
    plt.tight_layout()    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig3_v2_1' + ftype, dpi=dpi)
    

def plot_NVR_and_gain_compression(device, sweep_name, ftype='.pdf',dpi=240):
    
    fig, axes = plt.subplots(2, 3, sharey='row', figsize=(3.375, 2.0), gridspec_kw = {'height_ratios':[1, 1]})

    palette = plt.get_cmap('Set1')

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[sweep_name]['pump_params_sweep_20']
        
        dates = np.array(grp['date'])
        times = np.array(grp['time'])
        current = hdf5_file[sweep_name].attrs.get('current')       
        signal_powers = np.array(grp['Signal_pow_array'])
        print(np.shape(signal_powers))
        gain_array = np.array(grp['Gain_array'])
        print(np.shape(gain_array))
        pump_freqs = np.array(grp['Pump_freq'])
        freq_c = hdf5_file[sweep_name].attrs.get('freq_c')
        NVR = np.array(grp['NVR'])
        period_doubling_peak = np.array(grp['period_doubling_peak'])
        pump_detuning = (pump_freqs - 2*freq_c)/2
    finally:
        hdf5_file.close()



    ax = axes[0][0]
    ax.set_ylabel('NVR (dB)')
    
    ax = axes[1][0]
    ax.set_ylabel('Gain (dB)')
    
    ax = axes[1][1]    
    ax.set_xlabel(r'Input power (dBm)')

    ax = axes[0][1]    
    ax.set_xlabel(r'$\omega/2\pi$ (MHz)')

    nice_data = [130,60,20]

    """ Plot NVR """

    detuninglabels = []

    for i in range(3):
        ax = axes[0][i]    
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            h5date = float_to_DateTime(dates[nice_data[i]])
            h5time = float_to_DateTime(times[nice_data[i]])
            key1 = list(hdf5_file[h5date][h5time]['noise'].keys())[0]
            grp = hdf5_file[h5date][h5time]['noise'][key1]
            memGrp = hdf5_file[h5date][h5time]['noise'][key1]['memory']
            keys = grp.keys()
            meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
            freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
            noise_on = np.array(grp[meas_name]['logmag'])
            noise_off = np.array(memGrp[meas_name]['logmag'])
        finally:
            hdf5_file.close()
        # now for the figure
        NVR = noise_on - noise_off
        this_detuning = (np.average(freqs)-freq_c*1e-9)*1e3
        detuninglabels += [this_detuning]
        new_detuning = (freqs-np.average(freqs))*1e3
        ax.set_xlim([np.min(new_detuning), np.max(new_detuning)])
        ax.set_ylim([-2, 20])
#        ax.set_yticks([0,10,20])
        l = round(len(new_detuning)/2)
        # filtering out the spike:
        ax.plot(new_detuning[:l-2], NVR[:l-2], color='k', linestyle='-')
        ax.plot(new_detuning[l+2:], NVR[l+2:], color='k', linestyle='-')

#        ax.plot(new_detuning, NVR, color='k', linestyle='-')

        ax.plot(new_detuning, 8.3*np.ones(len(new_detuning)), color='black',linestyle='--')

        ax.set_xticks([-200,0,200])
        ax.scatter([-200],[17],marker='s',color=palette(2-i))
            

        """ Plot P1dB """

        ax = axes[1][i]  
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            h5date = float_to_DateTime(dates[nice_data[i]])
            h5time = float_to_DateTime(times[nice_data[i]])
            grp = hdf5_file[h5date][h5time]['LIN']
            memGrp = hdf5_file[h5date][h5time]['memory']
            powGrp = hdf5_file[h5date][h5time]['POW']
            keys = grp.keys()
            meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
            freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
            yr = np.array(grp[meas_name]['real'])
            yi = np.array(grp[meas_name]['imag'])
            ymemr = np.array(memGrp[meas_name]['real'])
            ymemi = np.array(memGrp[meas_name]['imag'])
            pows_in = np.array(powGrp['powers'])
            powr = np.array(powGrp[meas_name]['real'])
            powi = np.array(powGrp[meas_name]['imag'])
            cwFreq = powGrp['powers'].attrs['freqINP'] *1e-9 #to GHz
        finally:
            hdf5_file.close()
        ymemMag = np.abs(ymemr + 1j * ymemi)
        yr = yr / ymemMag
        yi = yi / ymemMag
        glogmag = 20*np.log10(np.abs(yr + 1j*yi))
        i_cw = np.argmin(np.abs(cwFreq - freqs))
        pows_in = pows_in + attenuation(freqs[i_cw]*1e9, device.data_file, 'signal_attenuation')
        pows_out = 20*np.log10(np.abs(powr + 1j*powi)) -  20*np.log10(ymemMag[i_cw]) #+0.4
    
    #    ax[0].plot(freqs, glogmag, color='k', linestyle='-')
        ax.plot(pows_in, pows_out, color='k', linestyle='-')
    #    plt.setp( ax[0].get_xticklabels()[1::2], visible=False )
        ax.scatter([-133],[32],marker='s',color=palette(2-i))
        ax.set_xticks([-130,-115,-100])
        ax.set_ylim([10, 38])
        ax.set_xlim([-140, -95])        

    plt.tight_layout()
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig3_v2_2' + ftype, dpi=dpi)



    fig, ax = plt.subplots(figsize=(3.375, 1.5))


    """ Plot the phase diagram """
    # parameters of the system
    f = Flux(1.27e-3,device.a,device.b)
    g3 = device.g3_distributed(f)
    g4 = device.g4_distributed(f) - (28*3/32-5)*g3**2/freq_c 
    ## !!!!!!! This g4 includes the correction from second order harmonic balance, do I need it ?
    kappa = device.kappa(f)
    Delta = np.linspace(-175e6,600e6,1e4)

    ax.set_yscale('log')
    palette = plt.get_cmap('Pastel2')
    lower_bound = 6e2 #8e2
    upper_bound = 2e4 #2e4
    ax.set_ylim(lower_bound,upper_bound)
    ax.set_xlim(-600,175)
    lines_color = 'grey'
    ax.set_ylabel(r'$\rm Pump \; photons$')
#    ax.yaxis.set_label_coords(-0.1, 0.47)
    ax.set_xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')


    # define the lower and upper infinite gain lines -- lines that bound the bistable region
    L1 = (-(Delta*g4*64/3-16*g3**2) - np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    L2 = (-(Delta*g4*64/3-16*g3**2) + np.sqrt( (Delta*g4*64/3-16*g3**2)**2 -4*g4**2*(32/3)**2*(Delta**2+kappa**2/4)))/g4**2/(32/3)**2/2
    ax.plot(-Delta*1e-6,L1,color=lines_color)
    ax.plot(-Delta*1e-6,L2,color=lines_color)
    
    # define the range of detunings and the line that bounds the tristable region from below
    n_p_low_low = kappa**2/64/g3**2
    ind = np.where(Delta > -g4*32/3*n_p_low_low)[0]
    Delta_low_low = Delta[ind]
    L3 = n_p_low_low*np.ones(len(Delta_low_low))
    ax.plot(-Delta_low_low*1e-6, L3, color=lines_color)

    # fill different areas of phase diagram with different colors
    ax.fill_between(-Delta*1e-6, lower_bound*np.ones(len(Delta)), upper_bound*np.ones(len(Delta)), facecolor=palette(0), interpolate=True)  
    ax.fill_between(-Delta_low_low*1e-6, L1[ind], L3, where=L1[ind] >= L3, facecolor=palette(1), interpolate=True)
    ind2 = np.where(L1>0)
    Delta2 = Delta[ind2]
    print(len(Delta))
    print(len(Delta2))
    ax.fill_between(-Delta2*1e-6, L1[ind2], L2[ind2], where=L2[ind2] >= L1[ind2], facecolor=palette(2), interpolate=True)

    # Plot the 20 dB gain line
    def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
        g = 2*g3_*np.sqrt(n_p)   
        Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i )
        return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2

    nbar_pump_sweep = np.linspace(1,upper_bound,1e4)
    n_p = np.empty(0)
    pump_detuning = np.empty(0)
    for i, delta in enumerate(Delta):
        gains = Gain(nbar_pump_sweep,0,0,g3, g4, kappa, -delta)
        gains = 10*np.log10(gains)
        count = 0
        while gains[count]<20 and count<len(gains)-1: 
            count +=1
        if count<len(gains)-1:
            n_p = np.append(n_p, nbar_pump_sweep[count])
            pump_detuning = np.append(pump_detuning, -delta)
    ax.plot(pump_detuning*1e-6, n_p, color='black')

    ax.annotate(r'$\rm (I)$', xy=(114, 3e3), fontsize=6)
    ax.annotate(r'$\rm (II)$', xy=(-100, 5e3), fontsize=6)
    ax.annotate(r'$\rm (III)$', xy=(-450, 2e3), fontsize=6)

    palette = plt.get_cmap('Set1')
    for i in range(3):
        ax.scatter([detuninglabels[i]],[2e3],marker=10,color=palette(2-i))
        ax.scatter([detuninglabels[i]],[2e3],marker=3,color=palette(2-i))

        
        



    plt.tight_layout()    
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'fig3_v2_1' + ftype, dpi=dpi)



##############################################
##############################################
##############################################
##############################################
# ---- Plot the figures for a JAMPA paper ---- 
##############################################
##############################################
##############################################
##############################################
    
def plot_two_tone_spectroscopy_compare_devices(device_list, dpi=240, ftype='.png'):

    fig, axes = plt.subplots(1, 3, sharey='row', figsize=(7, 3)) 

    for i, device in enumerate(device_list):
        
        hdf5 = h5py.File(device.data_file)
        try:
            z = np.asarray(hdf5['two_tone_spectroscopy_data'].get('phase_normalized'))
            ag_freqs = np.asarray(hdf5['two_tone_spectroscopy_data'].get('ag_freqs'))
            currents = np.asarray(hdf5['two_tone_spectroscopy_data'].get('currents'))
        finally:
            hdf5.close()
        ax = axes[i]
        p1 = ax.pcolormesh(currents*1e3, ag_freqs*1e-9, np.transpose(z), cmap='RdBu',vmin=-180,vmax=180)
        ax.set_ylim(0,35)
        if i==0: 
            ax.set_ylabel(r'${\rm Frequency \,(GHz)}$')
        if i==1:
            ax.set_xlabel(r'${\rm Current\, (mA)}$')
        ax.set_title('$M=%d$' %device.M, fontsize=8)
    cbar = fig.colorbar(p1, label=r'$\rm Phase \,(deg)$',ax=axes.ravel().tolist(), location='right')#,ticks=[10,15,20,25,30])
#    cbar.ax.set_yticklabels(['10', '15','20','25','30'])
    fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JTLA/paper/' + 'fig2' + ftype, dpi=dpi)     
    
    
def plot_gains_wide_span(device, h5date, h5time, inset_data=None, ftype='.png', savepath = None):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        grp = hdf5_file[h5date][h5time]['LIN']
        keys = grp.keys()
        meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
        if 'memory' in hdf5_file[h5date][h5time].keys():
            memGrp = hdf5_file[h5date][h5time]['memory']
            ymemr = np.array(memGrp[meas_name]['real'])
            ymemi = np.array(memGrp[meas_name]['imag'])
            ymemMag = np.abs(ymemr + 1j * ymemi)
        else: 
            ymemMag = 1
        freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
        yr = np.array(grp[meas_name]['real'])
        yi = np.array(grp[meas_name]['imag'])
    finally:
        hdf5_file.close()
    
    yr = yr / ymemMag
    yi = yi / ymemMag
    glogmag = 20*np.log10(np.abs(yr + 1j*yi))
    # now for the figure

    palette = plt.get_cmap('Set1')
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 1.8), dpi=240) 
    ax1.set_xlabel('Probe frequency (GHz)')
    plt.ylabel('Gain (dB)')
    plt.xlim([np.min(freqs), np.max(freqs)])
    plt.ylim([-2, 22])
    plt.plot(freqs, glogmag, color=palette(0), linestyle='-') 
    plt.xlim([4.4, 8.4])   


    """ Plot insets. inset_data is a list of tuples containing dates and times of the gain curve measurements.
    The current version of this plotting is very customized to a specific data for MLS, need to generalize it."""
    if inset_data:
        left, bottom, width, height = [0.57, 0.55, 0.15, 0.3]
        inset1 = fig.add_axes([left, bottom, width, height])
        h5date, h5time = inset_data[0]
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[h5date][h5time]['LIN']
            keys = grp.keys()
            meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
            if 'memory' in hdf5_file[h5date][h5time].keys():
                memGrp = hdf5_file[h5date][h5time]['memory']
                ymemr = np.array(memGrp[meas_name]['real'])
                ymemi = np.array(memGrp[meas_name]['imag'])
                ymemMag = np.abs(ymemr + 1j * ymemi)
            else: 
                ymemMag = 1
            freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
            yr = np.array(grp[meas_name]['real'])
            yi = np.array(grp[meas_name]['imag'])
        finally:
            hdf5_file.close()
        yr = yr / ymemMag
        yi = yi / ymemMag
        glogmag = 20*np.log10(np.abs(yr + 1j*yi))
        plt.xlim([np.min(freqs), np.max(freqs)])
        plt.ylim([-2, 24])
        plt.plot(freqs, glogmag, color=palette(1), linestyle='-')              


        left, bottom, width, height = [0.29, 0.55, 0.15, 0.3]
        inset1 = fig.add_axes([left, bottom, width, height])        
        h5date, h5time = inset_data[1]        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            grp = hdf5_file[h5date][h5time]['LIN']
            keys = grp.keys()
            meas_name = [m for m in keys if 'CH' in m and '_' in m][0]
            if 'memory' in hdf5_file[h5date][h5time].keys():
                memGrp = hdf5_file[h5date][h5time]['memory']
                ymemr = np.array(memGrp[meas_name]['real'])
                ymemi = np.array(memGrp[meas_name]['imag'])
                ymemMag = np.abs(ymemr + 1j * ymemi)
            else: 
                ymemMag = 1
            freqs = np.array(grp['frequencies']) * 1e-9 #to GHz
            yr = np.array(grp[meas_name]['real'])
            yi = np.array(grp[meas_name]['imag'])
        finally:
            hdf5_file.close()
        yr = yr / ymemMag
        yi = yi / ymemMag
        glogmag = 20*np.log10(np.abs(yr + 1j*yi))
        plt.xlim([np.min(freqs), np.max(freqs)])
        plt.ylim([-2, 24])
        plt.plot(freqs, glogmag, color=palette(1), linestyle='-') 
        
    plt.tight_layout()
    fig.savefig(device.device_folder + 'wide_span_gain'  + ftype, dpi=240)


def plot_phase_colorMAP(device, colorMap=('hot','phase'), title=None, ftype='.png'):

    hdf5_file = h5py.File(device.data_file,'r')
    try:
        date = hdf5_file['results'].attrs.get('flux_sweep_date')
        time = hdf5_file['results'].attrs.get('flux_sweep_time')
        meas_name = hdf5_file['results'].attrs.get('flux_sweep_meas_name') 
        frequencies = np.asarray(hdf5_file[date][time]['LIN'].get('frequencies'))
        fluxes = Flux(np.asarray(hdf5_file[date][time]['LIN'].get('currents')),device.a,device.b)
        real = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('real'))
        imag = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('imag'))
    finally:
        hdf5_file.close()

    define_phaseColorMap()
    (ylog, yphase) = complex_to_PLOG(np.vectorize(complex)(real, imag))

    freqs = 1e-9*frequencies
    a_abs = np.transpose(ylog)
    a_phase = np.transpose(yphase)
    
    fig, ax = plt.subplots(1,1,figsize=(3.375, 1.5),dpi=240)
    p1 = ax.pcolormesh(fluxes, freqs, a_phase, cmap=colorMap[1])
    fig.colorbar(p1, ax=ax, ticks=[-180, -90, 0, 90, 180], label=r'${\rm Phase \,(deg)}$')
    ax.set_ylabel(r'${\rm Frequency \,(GHz)}$')
#    ax.set_xlabel(r'${\rm Flux}$, $\Phi/\Phi_0$')
    ax.set_xlabel(r'$\rm Current \,(mA)$')
    plt.tight_layout()
    fig.savefig(device.device_folder + 'phase_colormap' + ftype,dpi=240)
    
    
    
    
