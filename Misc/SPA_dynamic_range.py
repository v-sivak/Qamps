from __future__ import division
import numpy as np
from sympy import solve_poly_system, Symbol, solve
from scipy.optimize import root, fsolve
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl

from devices_functions import *
from SPA import *

# Self consistent system of equations, obtained using harmonic balance method, that keeps track only
# of signl, idler and pump frequencies. 

def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))

def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))

def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))

def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))

def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))

def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
    
def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, g3_, p):
    Xp, Yp, Xs, Ys, Xi, Yi = p
    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )


# gain as a function of fourier components at signal, idler and pump frequencies

def Gain(freq_s, freq_i, freq_p, g4_, g3_, Delta, omega, kappa_t, freq_c, Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess):
  
    Xp, Yp, Xs, Ys, Xi, Yi = fsolve( lambda p: EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, g3_, p) , ( Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess ) )
    nbar_s = Xs**2+Ys**2
    nbar_p = (Xp**2+Yp**2)*10/9
    nbar_i = Xi**2+Yi**2    
    gg = 4*g3_**2*(Xp**2+Yp**2)
    g4_star = g4_ - 5*g3_**2/freq_c
    Delta_p = Delta + g4_*(32/3*(Xp**2+Yp**2)) - 28*g3_**2/freq_c*(Xp**2+Yp**2)
    Delta_s = Delta_p + 12*g4_star*((Xs**2+Ys**2)+2*(Xi**2+Yi**2))
    Delta_i = Delta_p + 12*g4_star*(2*(Xs**2+Ys**2)+(Xi**2+Yi**2))

    G = 1 + 4*kappa_t**2*gg/( ( (-Delta_s+omega)*(-Delta_i-omega) +kappa_t**2/4-4*gg)**2 + kappa_t**2*(-Delta_i+Delta_s-2*omega)**2*1/4 )
    G = 10*np.log10(float(G))
    return  G, nbar_p, nbar_s, nbar_i

def pump_strength_for_gain_helper(aa,g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    return (kappa_t*omega)**2+( 1/4*kappa_t**2-omega**2-16*g3_**2*aa+ (Delta+32/3*g4_*aa - 28*g3_**2/freq_c*aa )**2 )**2 -16*g3_**2*aa*kappa_t**2/(G-1)


def pump_strength_for_gain(g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    positive_sols = []
#    u = Symbol('u', positive = True) 
    a=(g4_*32/3)**2
    b=Delta*g4_*64/3-16*g3_**2
    c=Delta**2-omega**2+kappa_t**2/4
    aa_guess= (-b-np.sqrt(b**2-4*a*c))/2/a
#    for sol in solve_poly_system([pump_strength_for_gain_helper(u)], u ):
    for sol in [fsolve(lambda x: pump_strength_for_gain_helper(x,g4_,g3_,Delta,omega,kappa_t,freq_c,G), aa_guess )]:
#    for sol in np.roots( [-(32/3*g4_/freq_c**2)**2, 0, 4*(2*g3_/freq_c)**2-Delta*64/3*g4_/freq_c**2, 2*kappa_t*(2*np.abs(g3_)/freq_c)/np.sqrt(G-1), -(kappa_t**2)/4-Delta**2] ):
#    for sol in [brentq(pump_strength_for_gain_helper, u_p_guess/2, 2*u_p_guess) ]:
        if np.isreal(complex(sol[0])) :
            if sol[0]>0:
                positive_sols = positive_sols + [sol[0]]
    if positive_sols!=[]:
        return np.sqrt( float( min(positive_sols) ) )*freq_c 
    else:
        return 0




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------





#def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))
#
#def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) )  )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))
#
#def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))
#
#def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))
#
#def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))
#
#def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
#    
#def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, g3_, p):
#    Xp, Yp, Xs, Ys, Xi, Yi = p
#    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )
#
#
## gain as a function of fourier components at signal, idler and pump frequencies
#
#def Gain(freq_s, freq_i, freq_p, g4_, g3_, Delta, omega, kappa_t, freq_c, Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess):
#  
#    Xp, Yp, Xs, Ys, Xi, Yi = fsolve( lambda p: EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, g3_, p) , ( Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess ) )
#    nbar_s = Xs**2+Ys**2
#    nbar_p = (Xp**2+Yp**2)*10/9
#    nbar_i = Xi**2+Yi**2    
#    gg = 4*g3_**2*(Xp**2+Yp**2)
#    Delta_eff = Delta + g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) 
#    G = 1 + 4*kappa_t**2*gg/( (Delta_eff**2-omega**2+kappa_t**2/4-4*gg)**2 + omega**2*kappa_t**2 )
#    G = 10*np.log10(float(G))
#    return  G, nbar_p, nbar_s, nbar_i, Delta_eff 
#
#def pump_strength_for_gain_helper(aa,g4_,g3_,Delta,omega,kappa_t,freq_c,G):
#    return (kappa_t*omega)**2+( 1/4*kappa_t**2-omega**2-16*g3_**2*aa+ (Delta+32/3*g4_*aa )**2 )**2 -16*g3_**2*aa*kappa_t**2/(G-1)
#
#
#def pump_strength_for_gain(g4_,g3_,Delta,omega,kappa_t,freq_c,G):
#    positive_sols = []
##    u = Symbol('u', positive = True) 
#    a=(g4_*32/3)**2
#    b=Delta*g4_*64/3-16*g3_**2
#    c=Delta**2-omega**2+kappa_t**2/4
#    aa_guess= (-b-np.sqrt(b**2-4*a*c))/2/a
##    for sol in solve_poly_system([pump_strength_for_gain_helper(u)], u ):
#    for sol in [fsolve(lambda x: pump_strength_for_gain_helper(x,g4_,g3_,Delta,omega,kappa_t,freq_c,G), aa_guess )]:
##    for sol in np.roots( [-(32/3*g4_/freq_c**2)**2, 0, 4*(2*g3_/freq_c)**2-Delta*64/3*g4_/freq_c**2, 2*kappa_t*(2*np.abs(g3_)/freq_c)/np.sqrt(G-1), -(kappa_t**2)/4-Delta**2] ):
##    for sol in [brentq(pump_strength_for_gain_helper, u_p_guess/2, 2*u_p_guess) ]:
#        if np.isreal(complex(sol[0])) :
#            if sol[0]>0:
#                positive_sols = positive_sols + [sol[0]]
#    if positive_sols!=[]:
#        return np.sqrt( float( min(positive_sols) ) )*freq_c
#    else:
#        return 0





if __name__ == '__main__':
#    spa01 = SPA(path,'SPA01')
#    spa03 = SPA(path,'SPA03')
#    spa04 = SPA(path,'SPA04')
#    spa05 = SPA(path,'SPA05')
#    spa08 = SPA(path,'SPA08')
#    spa13_v2 = SPA(path,'SPA13_v2')
#    spa14_v2 = SPA(path,'SPA14_v2')
#    spa04_v2 = SPA(path,'SPA04_v2')
    
    

#    fig = plt.figure(figsize=(3.375, 2.0),dpi=240)               #(figsize=(9, 7),dpi=240)
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3.375+0.225,2.0))
#    fig.suptitle(r'Saturation power $P_{1dB}$', fontsize=20)
#    ax = plt.gca()
    ax = axs[1]
    ax2 = axs[0]
    ax.set_xlabel(r'Predicted $P_{-1 \rm  dB}$ (dBm)')
#    ax.set_ylabel(r'Measured $P_{-1 \rm  dB}$ (dBm)')
#    p1dB_lim = (-150, -90)    
    p1dB_lim = (-150, -90)
    ax.set_ylim(p1dB_lim)
    ax.set_xlim(p1dB_lim)
    ax.plot(np.linspace(p1dB_lim[0],p1dB_lim[1],50), np.linspace(p1dB_lim[0],p1dB_lim[1],50), linestyle = '-', color = 'black',zorder=1)
    
    ax2.set_xlabel(r'$\Phi/\Phi_0$')
    ax2.set_ylabel(r'Measured  $P_{-1 \rm  dB}$ (dBm)')
    ax2.set_xlim((0.07, 0.47))
    ax2.set_ylim(p1dB_lim)
    
    lines = []
    

    figIMD = plt.figure(figsize=(3.375, 2.0), dpi=240)
    axIMD = plt.gca()
    axIMD.set_xlabel(r'Center frequency (GHz)')
    axIMD.set_ylabel(r'Power (dBm)')
    IMD_lim = (-125, -95)
    axIMD.set_ylim(IMD_lim)
    
    
    

    device_list = [spa04_v2,spa08, spa13_v2, spa14_v2]    
    devices_IMD = [spa13_v2, spa14_v2]

    labels = ['Device B', 'Device C', 'Device D', 'Device E']
    colors = ['orange','green','blue','red']

    line_IMD = []
    line_P1dB = []


    for device_num, device in enumerate(device_list):
        
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            # choose what gain points you want to process
            Happy_indices = []
            for index, G in enumerate(hdf5_file['pumping_data']['Gain']):
                if G>19 and G<21 and all(Gains<21 for Gains in hdf5_file['pumping_data']['Gain_array'][index]):
                    Delta = float(hdf5_file['pumping_data']['freq_c'][index])-float(hdf5_file['pumping_data']['Pump_freq'][index])/2
                    if Delta < 1e6 and Delta>-1e6:
                        Happy_indices = Happy_indices + [index]
        finally:
            hdf5_file.close()
    
    
        
        Signal_Powers_dBm = np.linspace(-150,-80,201)
        Signal_Powers = dBm_to_Watts(Signal_Powers_dBm)
        Gains = np.zeros((len(Happy_indices),len(Signal_Powers)))
        nbar_signal = np.zeros_like(Gains)
        nbar_pump = np.zeros_like(Gains)
        nbar_idler = np.zeros_like(Gains)
        Delta_effective = np.zeros_like(Gains)
        crit_ratio = np.zeros((len(Happy_indices),len(Signal_Powers)))
        Pump_Powers_theor = np.zeros(len(Happy_indices))
    
        P_1dB_theor = np.zeros(len(Happy_indices))
        P_1dB_exp = np.zeros(len(Happy_indices))
        Fluxes_theor = np.zeros(len(Happy_indices))
        
        center_freq = np.zeros(len(Happy_indices))
        IIP3 = np.zeros(len(Happy_indices))
    
    
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            flux_IMD = Flux(np.asarray(hdf5_file['pumping_data'].get('current')),device.a, device.b)
            kappa_c = np.asarray(hdf5_file['pumping_data'].get('kappa_c'))
            freq_c = np.asarray(hdf5_file['pumping_data'].get('freq_c'))
            G = np.power(10, np.asarray( hdf5_file['pumping_data'].get('Gain'))/10 )
#            if device in devices_IMD:
#                IIP3 = np.asarray( hdf5_file['pumping_data'].get('IIP3') )
#                IIP3 = dBm_to_Watts( IIP3 )
#                g4_IMD = kappa_c**2*freq_c/12/IIP3*(2*np.pi*h)/G**(3/2)         
            
            for l, index in enumerate(Happy_indices):
                
                f = Flux(hdf5_file['pumping_data']['current'][index], device.a, device.b)
                Fluxes_theor[l] = f
    
                
                freq_p = hdf5_file['pumping_data']['Pump_freq'][index]
                freq_c = np.asarray(hdf5_file['pumping_data']['freq_c'][index])
                freq_s = hdf5_file['pumping_data']['CW_freq'][index]
                freq_i = freq_p - freq_s
                
                if device in devices_IMD:
                    center_freq[l] = freq_c
                    IIP3[l] = hdf5_file['pumping_data']['IIP3'][index]

                if device in [spa14_v2]:
                    fudge_factor = -2.36
                else:
                    fudge_factor = float(hdf5_file['g3_fit'].attrs.get('fudge_factor'))

                
                _g3_ =  device.g3_distributed(f) #* fudge_factor
#                _g3_ = hdf5_file['g3_fit'].get('g3')[index] * fudge_factor
                
                _g4_ =  device.g4_distributed(f) + 5*_g3_**2/freq_c
#                _g4_ = g4_IMD[l]
    
                G = np.power(10,hdf5_file['pumping_data']['Gain'][index]/10)
                Delta = freq_c - freq_p/2
                omega = freq_s - freq_p/2
                Bw = hdf5_file['pumping_data']['Bandwidth'][index]
                kappa_ripple = np.sqrt(G*Bw**2 - Delta**2)
                kappa_t = hdf5_file['pumping_data']['kappa_t'][index]
                kappa_c = hdf5_file['pumping_data']['kappa_c'][index]
                
    
                u_p = pump_strength_for_gain( _g4_, _g3_, Delta, omega, kappa_t, freq_c, G) 
                alpha_p = u_p/freq_c 
                
                P_p = float (freq_c/kappa_c*h*u_p**2*2*np.pi)
                Pump_dBm = Watts_to_dBm(P_p)
                Pump_Powers_theor[l] = Pump_dBm
            
            
                for i, P_s in enumerate(Signal_Powers):
    
                    u_s = np.sqrt(P_s/(2*np.pi*h)*kappa_c/freq_c )
                    
                    _Xp_guess = alpha_p
                    _Yp_guess = 0
                    
                    _Xs_guess = float( -u_s*( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )*(Delta+omega) )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
                    _Ys_guess = float( -kappa_t/2*( (Delta+omega)**2+kappa_t**2/4-16*alpha_p**2*_g3_**2  )*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
        
                    _Xi_guess = float( 4*_g3_*alpha_p*u_s*( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
                    _Yi_guess = -4*_g3_*alpha_p*kappa_t*omega*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 )
      
                    Gains[l,i], nbar_pump[l,i], nbar_signal[l,i], nbar_idler[l,i] = Gain(freq_s, freq_i, freq_p, _g4_, _g3_, Delta, omega, kappa_t, freq_c, _Xp_guess, _Yp_guess, _Xs_guess, _Ys_guess, _Xi_guess, _Yi_guess)              
                
                ind_compression = find_1dB_compression_point(Gains[l], Gains[l][0])
                P_1dB_theor[l] = Signal_Powers_dBm[ ind_compression ]

                print('Flux %.3f | Pump power %.1f | Gain %.2f' % (f,Pump_dBm, Gains[l][0]) )

                print('n_p, n_s, n_i INITIAL')
                print(nbar_pump[l,0], nbar_signal[l,0], nbar_idler[l,0] )
                
                print('n_p, n_s, n_i COMPRESSION')
                print(nbar_pump[l,ind_compression], nbar_signal[l,ind_compression], nbar_idler[l,ind_compression] )
                P_1dB_exp[l] = np.array(hdf5_file['pumping_data']['P_1dB'][index])
                print('\n')
                
                Gain_sat = np.power(10,(hdf5_file['pumping_data']['Gain'][index]-1)/10)
                G = np.power(10,hdf5_file['pumping_data']['Gain'][index]/10)
                _g4_ = _g4_ - 17/2*_g3_**2/freq_c
#                _g4_ = g4_IMD[l]
                P_1dB_theor[l] = np.abs(kappa_c**2*freq_c/48/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+4*np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h))
                P_1dB_theor[l] = Watts_to_dBm(P_1dB_theor[l])
        finally:
            hdf5_file.close()
               
            
            
#   ---------------------------- PLOTTING --------------------------------
            
        l = ax.plot(P_1dB_theor, P_1dB_exp, linestyle = 'None', color = colors[device_num],
                marker = '.', label=labels[device_num])
        lines += l
        ax2.plot(Fluxes_theor, P_1dB_exp, color=colors[device_num], linestyle='-',
                     marker='.', label=labels[device_num])
        
        
        if device in devices_IMD:
            l = axIMD.plot(center_freq*1e-9, P_1dB_exp, color=colors[device_num], linestyle='-',
                     marker='.', label=r'$P_{-1 \rm  dB}$ '+labels[device_num])
            line_P1dB.append(l[0])
            
            l = axIMD.plot(center_freq*1e-9, IIP3, color=colors[device_num], linestyle='-',
                     marker='s', fillstyle='none', label=r'$IIP_3$ '+labels[device_num])
            line_IMD.append(l[0])



#    ax.legend(loc='best')

#    ax.set(adjustable='box-forced',aspect='equal')
#    ax2.set(adjustable='box-forced')
    
    fig.legend(lines, labels, loc = 'upper center',  ncol=4, borderaxespad=0.2) #mode='expand',
#               bbox_to_anchor = (0, 1.02, 1, 0.102) )
    fig.tight_layout(w_pad=0.5)
    ax.set_xlim(p1dB_lim)
    ax.set_xticks(np.linspace(p1dB_lim[0],p1dB_lim[1],7))
    plt.setp( ax.get_xticklabels()[::2],visible=False )
    
#    saveDir = r'Y:\volodymyr_sivak\SPA\DATA\\'
#    fig.savefig(saveDir + 'P_1dB_combo.pdf', dpi=240)
    
#    axIMD.legend(loc='best')    
    
    first_legend = axIMD.legend(handles=line_IMD, loc=2)
    axIMD.add_artist(first_legend)
    second_legend = axIMD.legend(handles=line_P1dB, loc=4)
    
    saveDir=path
    
    figIMD.tight_layout()
    figIMD.savefig(saveDir + 'IMD.pdf', dpi=240)

    
    