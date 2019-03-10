from __future__ import division
import numpy as np
from sympy import solve_poly_system, Symbol, solve
from scipy.optimize import root, fsolve
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl

from devices_functions import *
from SPA import *




if __name__ == '__main__':

    device_list = [spa34_v6]
    figIMD, axes = plt.subplots(2, 2, sharey='row', figsize=(7, 3.5))
    
    axes = axes.ravel()
    
    Freq_calibration = 25e6

    palette = plt.get_cmap('Set1')
        
    axIMD = axes[0]
    axIMD.set_xlabel(r'$\Phi/\Phi_0$')
    axIMD.xaxis.set_label_coords(0.5, -0.15)    
    axIMD.set_ylabel(r'$P_{-1 \rm  dB}$ (dBm)')
    IMD_lim = (-123, -95)
    


    for device_num, device in enumerate(device_list):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            group = 'pumping_data' if 'pumping_data'  in hdf5_file.keys() else 'pumping_data_20'
            # choose what gain points you want to process
            Happy_indices = []
            for index, G in enumerate(hdf5_file[group]['Gain']):
                if G>19 and G<21 and all(Gains<21 for Gains in hdf5_file[group]['Gain_array'][index]):
                    Delta = float(hdf5_file[group]['freq_c'][index])-float(hdf5_file[group]['Pump_freq'][index])/2
                    if Delta < 1e6 and Delta>-1e6:
                        Happy_indices = Happy_indices + [index]
        finally:
            hdf5_file.close()
    
        P_1dB_theor_Stark = np.zeros(len(Happy_indices))
        P_1dB_theor_Stark_IMD = np.zeros(len(Happy_indices))
        P_1dB_exp = np.zeros(len(Happy_indices))
        Fluxes_theor = np.zeros(len(Happy_indices))        
        IIP3 = np.zeros(len(Happy_indices))
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            kappa_c = np.asarray(hdf5_file[group].get('kappa_c'))
            freq_c = np.asarray(hdf5_file[group].get('freq_c'))
            Gain_sat = np.power(10,(hdf5_file[group]['Gain'][index]-1)/10)
            G = np.power(10,hdf5_file[group]['Gain'][index]/10)     

            IIP3 = np.asarray( hdf5_file[group].get('IIP3') )
            IIP3 = dBm_to_Watts( IIP3 )
            g4_IMD_arr= kappa_c**2*freq_c/12/IIP3*(2*np.pi*h)*( (np.sqrt(G)-1)/(G-1) )**3 
            
            for l, index in enumerate(Happy_indices):
                print('%d/%d' %(l, len(Happy_indices)))        
                f = Flux(hdf5_file[group]['current'][index], device.a, device.b)
                Fluxes_theor[l] = f                
                freq_c = np.asarray(hdf5_file[group]['freq_c'][index])                
                freq_p = hdf5_file[group]['Pump_freq'][index]
                Delta = freq_c - freq_p/2 - Freq_calibration
                kappa_c = hdf5_file[group]['kappa_c'][index]
                
                P_1dB_exp[l] = np.array(hdf5_file[group]['P_1dB'][index])                

                _g4_ = device.g4_distributed(f)
#                P_1dB_theor_Stark[l] = np.abs(kappa_c**2*freq_c/24/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h))
                P_1dB_theor_Stark[l] = kappa_c**2*freq_c/36/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h) if _g4_<0 else kappa_c**2*freq_c/24/_g4_/Gain_sat*( np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h)
                P_1dB_theor_Stark[l] = Watts_to_dBm( P_1dB_theor_Stark[l] )
                print(P_1dB_theor_Stark[l])

                _g4_ = -g4_IMD_arr[index]
#                P_1dB_theor_Stark[l] = np.abs(kappa_c**2*freq_c/24/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h))
                P_1dB_theor_Stark_IMD[l] = kappa_c**2*freq_c/36/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h) if _g4_<0 else kappa_c**2*freq_c/24/_g4_/Gain_sat*( np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h)

                
                
                
                


                P_1dB_theor_Stark_IMD[l] = Watts_to_dBm( P_1dB_theor_Stark_IMD[l] )
                print(P_1dB_theor_Stark_IMD[l])
        finally:
            hdf5_file.close()
               
            
            
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------


        axIMD.plot(Fluxes_theor, P_1dB_exp, color=palette(1), linestyle='none', markersize=1.5, marker='o', label=r'Data ') #color=colors[device_num] label=r'$P_{-1 \rm  dB}$
      
                
        indexes = [i for i,x in enumerate(P_1dB_theor_Stark_IMD) if x == -150 or x==-80 ]
        X = [x for i,x in enumerate(P_1dB_theor_Stark_IMD) if i not in indexes]
        Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]
        axIMD.plot(np.asarray(Y), X, color=palette(1), linestyle='--', linewidth=0.8, label=r'K from IMD')

        indexes = [i for i,x in enumerate(P_1dB_theor_Stark) if x == -150 or x==-80 ]
        X = [x for i,x in enumerate(P_1dB_theor_Stark) if i not in indexes]
        Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]
        axIMD.plot(np.asarray(Y), X, color=palette(1), linestyle='-', linewidth=0.8, label=r'K=$12g_4^*$')


#    center_freqs = pump_detuning + freq_c
#    ax3 = axIMD.twiny()
#    ax3.set_xlim(left_freq*1e-9, right_freq*1e-9)
#    ax3.set_xlabel(r'Center frequency (GHz)')
#    ax3.xaxis.set_label_coords(0.5, 1.15)


    axIMD.legend(loc='best',frameon=False)
    figIMD.tight_layout()



    sweep_name='pump_params_sweep_curr_1270'
#    sweep_name='dumb_pump_params_sweep_curr_1100'
    
#    sweep_name='pump_params_sweep_curr_1100'
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
        pump_detuning = (pump_freqs - 2*freq_c)/2
    finally:
        hdf5_file.close()

    ax = axes[1]

    """ Plot signal power sweep """


    print(np.shape(gain_array))
    print(np.shape(signal_powers[0]))

    line_19dB = [signal_powers[0][np.argmin(np.abs(gain_array[i]-19))] for i in range(np.shape(gain_array)[0])]
    ax.plot(pump_detuning*1e-6, line_19dB, linestyle='none', markersize=1.5, marker='o',color=palette(1), label = 'Data')

#    line_19dB = [signal_powers[0][np.argmin(np.abs(gain_array[i]-21))] for i in range(np.shape(gain_array)[0])]
#    ax.plot(pump_detuning*1e-6, line_19dB, linestyle='none', markersize=1.5, marker='o',color=palette(2))

    detunings_21, signals_21 = [np.empty(0),np.empty(0)]

    for i in range(np.shape(gain_array)[0]):
        flag = True
        for j in range(np.shape(gain_array)[1]):
            if 20.95<gain_array[i][j]<21.1 and flag:
                detunings_21 = np.append(detunings_21, pump_detuning[i]*1e-6)
                signals_21 = np.append(signals_21, signal_powers[0][j])
                flag = False
            if gain_array[i][j]>21.1:
                flag = True
    ax.plot(detunings_21,signals_21,markersize=1.5, marker='o',color=palette(4),linestyle='none',label = 'Theory')

    ax.set_xlabel(r'Pump detuning $\Delta/2\pi$ (MHz)')
    ax.xaxis.set_label_coords(0.5, -0.15)
    f = Flux(current, device.a, device.b)


    Gain_sat = 10**(19/10)
    G = 100
    g4 = device.g4_distributed(f)
    g3 = device.g3_distributed(f)
    kappa = device.kappa(f)
    K = g4

    ind = np.argmin(pump_detuning)

    pump_detuning2 = pump_detuning
    pump_detuning = np.linspace(pump_detuning[ind],max(pump_detuning),1000)
    Delta = -pump_detuning-Freq_calibration




    P_1dB_theor_Stark = kappa**2*freq_c/36/K/Gain_sat*( -np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa)*(2*np.pi*h)
    ind = np.where(P_1dB_theor_Stark>0)
    P_1dB_theor_Stark = Watts_to_dBm(P_1dB_theor_Stark[ind])
    ax.plot(pump_detuning[ind]*1e-6,P_1dB_theor_Stark, linewidth=0.8, color = palette(1))
    
    P_1dB_theor_Stark = kappa**2*freq_c/36/K/Gain_sat*( +np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa)*(2*np.pi*h)
    ind = np.where(P_1dB_theor_Stark>0)
    P_1dB_theor_Stark = Watts_to_dBm(P_1dB_theor_Stark[ind])
    ax.plot(pump_detuning[ind]*1e-6,P_1dB_theor_Stark, color = palette(1),  linewidth=0.8, label = r'$P_{-1\rm dB}$')
    
    Gain_sat = 10**(21/10)
    G = 100

    P_1dB_theor_Stark = kappa**2*freq_c/36/K/Gain_sat*( -np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa)*(2*np.pi*h)
    ind = np.where(P_1dB_theor_Stark>0)
    P_1dB_theor_Stark = Watts_to_dBm(P_1dB_theor_Stark[ind])
    ax.plot(pump_detuning[ind]*1e-6,P_1dB_theor_Stark,  linewidth=0.8, color = palette(4))
    
    P_1dB_theor_Stark = kappa**2*freq_c/36/K/Gain_sat*( +np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa)*(2*np.pi*h)
    ind = np.where(P_1dB_theor_Stark>0)
    P_1dB_theor_Stark = Watts_to_dBm(P_1dB_theor_Stark[ind])
    ax.plot(pump_detuning[ind]*1e-6,P_1dB_theor_Stark, color = palette(4), linewidth=0.8, label = r'$P_{+1\rm dB}$')


    plt.legend(loc='best',frameon=False)



    ax.set_xlim(min(pump_detuning)*1e-6-30,max(pump_detuning)*1e-6+30)
    ax.set_ylim(-123, -95)


#    left_freq = (min(pump_detuning)*1e-6-30)*1e6 + freq_c
#    right_freq = (max(pump_detuning)*1e-6+30)*1e6 + freq_c
#    
#    ax2 = ax.twiny()
#    ax2.set_xlim(left_freq*1e-9, right_freq*1e-9)
#    ax2.set_xlabel(r'Center frequency (GHz)')
#    ax2.xaxis.set_label_coords(0.5, 1.15)




    ##############################
    ##############################
    # Plot phase transition figure

#    fig2, axes = plt.subplots(1, 2, sharey='row', figsize=(7, 2))
    ax = axes[2]
    ax.set_xlabel(r'Input power (dBm)')
    ax.set_ylabel(r'Gain (dB)')

    cmap = plt.get_cmap('coolwarm_r')
    
    
    
    norm = mpl.colors.Normalize(vmin = -pump_detuning2[0]*1e-6, vmax = -pump_detuning2[56]*1e-6)
    

    for i in range(56):#range(np.shape(gain_array)[0]):
        Delta = -pump_detuning2[i]
        color = cmap(norm(Delta*1e-6))
        ax.plot(signal_powers[0],gain_array[i][:],color=color,zorder=0)
#        indmax = np.argmax(gain_array[i][:])
#        ax.plot([signal_powers[0][indmax]],[gain_array[i][indmax]],color='black',zorder=2,marker='.',markersize=3)
    
    cax, kw = mpl.colorbar.make_axes(plt.gca(), shrink=0.5, aspect=10, orientation='horizontal')
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, **kw)
    cb.set_label('Pump detuning $\Delta/2\pi$ (MHz)') #APS: fontsize='x-large', labelpad=2


#    cb = plt.colorbar(p, ax=ax,label=r'${\rm Gain }$',shrink=0.45,aspect=5)
#    cb.set_ticks([10,20,30])
#    cb.set_label('Gain (dB)',fontsize='x-small')


    ax.set_xlim(-135, -97.4)
    ax.set_ylim(11, 36)   

    ax = axes[3]

    for i, pump_detuning in enumerate(pump_detuning2[:56]):
        
        Delta = -pump_detuning-Freq_calibration
        
        linewidth = 0.5
#        linewidth = 1 if i==44 else 0.5
        
        
        color = cmap(norm(Delta*1e-6))
        G_0 = 100

        if Delta>0:
            G_max = G_0
        else:
            expr = 1/np.sqrt(G_0) - Delta**2/kappa**2/np.sqrt( Delta**2/kappa**2+1/4 )
            if expr>0.01:
                G_max = (expr)**(-2)
            else: G_max = 10**(30/10)
        print(G_max)

        G = 10**(np.linspace(12,10*np.log(G_max),5001)/10)
        P2 = kappa**2*freq_c/36/K/G*( +np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(G)-1/np.sqrt(G_0)) ) - Delta/kappa)*(2*np.pi*h)
        ind = np.where(P2>0)
        P2 = Watts_to_dBm(P2[ind])
        G = G[ind]
        ax.plot(P2, 10*np.log10(G), color = color,linewidth=linewidth)
#        if i==44:
#            indmax = np.argmax(P2)
#            ax.plot(P2[:indmax], 10*np.log10(G)[:indmax], color = 'blue',linestyle='--',linewidth=1)
#            P_crit = P2[indmax]
#            G_crit_min = 10*np.log10(G)[indmax]
        
        G = 10**(np.linspace(12,10*np.log(G_max),5001)/10)
        P1 = kappa**2*freq_c/36/K/G*( -np.sqrt( (Delta/kappa)**2+np.sqrt(1/4+(Delta/kappa)**2)*(1/np.sqrt(G)-1/np.sqrt(G_0)) ) - Delta/kappa)*(2*np.pi*h)
        ind = np.where(P1>0)
        P1 = Watts_to_dBm(P1[ind])
        G = G[ind]
        ax.plot(P1, 10*np.log10(G), color = color,linewidth=linewidth)
        
#        if i==44:
#            indmin = np.argmin(np.abs(P1-P_crit))
#            ax.plot(P1[:indmin], 10*np.log10(G)[:indmin], color = 'blue',linestyle='--',linewidth=1)
#            G_crit_max = 10*np.log10(G)[indmin]
#            ax.plot(P_crit*np.ones(2), [G_crit_min,G_crit_max], color = 'blue',linestyle=':',linewidth=1)


    
    ax.set_xlim(-135, -97.4)
    ax.set_ylim(11, 36)
    
    plt.tight_layout()
    
    figIMD.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/Kerr-free paper/Figures/' + 'saturation.pdf', dpi=240)
    