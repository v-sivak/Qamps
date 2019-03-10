from __future__ import division
import numpy as np
from sympy import solve_poly_system, Symbol, solve
from scipy.optimize import root, fsolve
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl

from devices_functions import *
from SPA import *



### -------------------------------- Including the IMD stuff -------------------------------------
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------



# Self consistent system of equations, obtained using harmonic balance method, that keeps track only
# of signl, idler and pump frequencies. 

def Equation1_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )           -g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))

def Equation2_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )               -g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))

def Equation3_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )               -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))

def Equation4_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c  -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )                  -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))

def Equation5_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )                    -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))

def Equation6_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c  -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )          -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
    
def EquationS_IMD(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
    Xp, Yp, Xs, Ys, Xi, Yi = p
    return (Equation1_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation3_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation5_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )


# gain as a function of fourier components at signal, idler and pump frequencies

def Gain_IMD(freq_s, freq_i, freq_p, g4_, g3_, Delta, omega, kappa_t, freq_c, u_p, u_s, Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess):
  
    Xp, Yp, Xs, Ys, Xi, Yi = fsolve( lambda p: EquationS_IMD(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p) , ( Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess ) )
    nbar_s = Xs**2+Ys**2
    nbar_p = (Xp**2+Yp**2)*10/9
    nbar_i = Xi**2+Yi**2    
    gg = 4*g3_**2*(Xp**2+Yp**2)
    Delta_eff = Delta + g4_*(32/3*(Xp**2+Yp**2)+12*nbar_s+12*nbar_i ) - 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+25.5*(Xs**2+Ys**2)+25.5*(Xi**2+Yi**2))
    G = 1 + 4*kappa_t**2*gg/( (Delta_eff**2-omega**2+kappa_t**2/4-4*gg)**2 + omega**2*kappa_t**2 )
    G = 10*np.log10(float(G))
    return  G, nbar_p, nbar_s, nbar_i, Delta_eff 

def pump_strength_for_gain_helper_IMD(aa,g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    return (kappa_t*omega)**2+( 1/4*kappa_t**2-omega**2-16*g3_**2*aa+ (Delta+32/3*g4_*aa - 28*g3_**2/freq_c*aa )**2 )**2 -16*g3_**2*aa*kappa_t**2/(G-1)


def pump_strength_for_gain_IMD(g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    freq_p = (freq_c-Delta)*2
    positive_sols = []
    a=(g4_*32/3)**2
    b=Delta*g4_*64/3-16*g3_**2
    c=Delta**2-omega**2+kappa_t**2/4
    aa_guess= (-b-np.sqrt(b**2-4*a*c))/2/a
    for sol in [fsolve(lambda x: pump_strength_for_gain_helper_IMD(x,g4_,g3_,Delta,omega,kappa_t,freq_c,G), aa_guess )]:
        if np.isreal(complex(sol[0])) :
            if sol[0]>0:
                positive_sols = positive_sols + [sol[0]]
    if positive_sols!=[]:
        alpha_p = np.sqrt( float( min(positive_sols) ) )
        return  np.sqrt((freq_p-freq_c-32/9*g4_*alpha_p**2 +g3_**2/freq_c*928/45*alpha_p**2 )**2 +(2/3*kappa_t)**2)*alpha_p  #alpha_p*freq_c
    else:
        return 0




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

### -------------------------------- NOT Including the IMD stuff -------------------------------------
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------



def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))

def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) )  )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))

def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))

def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))

def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))

def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
    
def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
    Xp, Yp, Xs, Ys, Xi, Yi = p
    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )



# gain as a function of fourier components at signal, idler and pump frequencies

def Gain(freq_s, freq_i, freq_p, g4_, g3_, Delta, omega, kappa_t, freq_c, u_p, u_s, Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess):
  
    Xp, Yp, Xs, Ys, Xi, Yi = fsolve( lambda p: EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p) , ( Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess ) )
    nbar_s = Xs**2+Ys**2
    nbar_p = (Xp**2+Yp**2)*10/9
    nbar_i = Xi**2+Yi**2    
    gg = 4*g3_**2*(Xp**2+Yp**2)
    Delta_eff = Delta + g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )
    G = 1 + 4*kappa_t**2*gg/( (Delta_eff**2-omega**2+kappa_t**2/4-4*gg)**2 + omega**2*kappa_t**2 )
    G = 10*np.log10(float(G))
    return  G, nbar_p, nbar_s, nbar_i, Delta_eff

def pump_strength_for_gain_helper(aa,g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    return (kappa_t*omega)**2+( 1/4*kappa_t**2-omega**2-16*g3_**2*aa+ (Delta+32/3*g4_*aa )**2 )**2 -16*g3_**2*aa*kappa_t**2/(G-1)


def pump_strength_for_gain(g4_,g3_,Delta,omega,kappa_t,freq_c,G):
    freq_p = (freq_c-Delta)*2
    positive_sols = []
    a=(g4_*32/3)**2
    b=Delta*g4_*64/3-16*g3_**2
    c=Delta**2-omega**2+kappa_t**2/4
    aa_guess= (-b-np.sqrt(b**2-4*a*c))/2/a
    for sol in [fsolve(lambda x: pump_strength_for_gain_helper(x,g4_,g3_,Delta,omega,kappa_t,freq_c,G), aa_guess )]:
        if np.isreal(complex(sol[0])) :
            if sol[0]>0:
                positive_sols = positive_sols + [sol[0]]
    if positive_sols!=[]:
        alpha_p = np.sqrt( float( min(positive_sols) ) )
        return  np.sqrt((freq_p-freq_c-32/9*g4_*alpha_p**2)**2 +(2/3*kappa_t)**2)*alpha_p  #alpha_p*freq_c
    else:
        return 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#### -------------------------------- Including the g5 and g6 -------------------------------------
#### ----------------------------------------------------------------------------------------------
#### ----------------------------------------------------------------------------------------------
#
#
#
#def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))
#
#def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) )  )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))
#
#def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))
#
#def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))
#
#def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))
#
#def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) )  )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
#    
#def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
#    Xp, Yp, Xs, Ys, Xi, Yi = p
#    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )







if __name__ == '__main__':

    figIMD = plt.figure(figsize=(3.375, 2.0), dpi=240)
    axIMD = plt.gca()
    axIMD.set_xlabel(r'$\Phi/\Phi_0$')
    axIMD.set_ylabel(r'$P_{-1 \rm  dB}$ (dBm)')
    IMD_lim = (-123, -95)
    IMD_lim = (-150, -95)
#    axIMD.set_ylim(IMD_lim)
    
    

    device_list = [spa34_v6]
    devices_IMD = [spa34_v6]


#    device_list = [spa14_v4]
#    devices_IMD = [spa14_v4]

#    device_list = [spa04_v2,spa08,spa13_v2,spa14_v4]    
#    devices_IMD = [spa13_v2,spa14_v2]
    
    colors = ['orange','green','blue','red']


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
    
    
        
        Signal_Powers_dBm = np.linspace(-150,-80,201)
        Signal_Powers = dBm_to_Watts(Signal_Powers_dBm)
        Gains = np.zeros((len(Happy_indices),len(Signal_Powers)))
        Gains3 = np.zeros((len(Happy_indices),len(Signal_Powers)))
        Gains_IMD = np.zeros((len(Happy_indices),len(Signal_Powers)))
        nbar_signal = np.zeros_like(Gains)
        nbar_pump = np.zeros_like(Gains)
        nbar_idler = np.zeros_like(Gains)
        Delta_effective = np.zeros_like(Gains)
        crit_ratio = np.zeros((len(Happy_indices),len(Signal_Powers)))
        Pump_Powers_theor = np.zeros(len(Happy_indices))
    
        P_1dB_theor = np.zeros(len(Happy_indices))
        P_1dB_theor_IMD = np.zeros(len(Happy_indices))
        P_1dB_theor_3 = np.zeros(len(Happy_indices))
        P_1dB_theor_Stark = np.zeros(len(Happy_indices))
        P_1dB_exp = np.zeros(len(Happy_indices))
        Fluxes_theor = np.zeros(len(Happy_indices))
        
        center_freq = np.zeros(len(Happy_indices))
        IIP3 = np.zeros(len(Happy_indices))
    
    
        hdf5_file = h5py.File(device.data_file,'r')
        try:
            flux_IMD = Flux(np.asarray(hdf5_file[group].get('current')),device.a, device.b)
            kappa_c = np.asarray(hdf5_file[group].get('kappa_c'))
            freq_c = np.asarray(hdf5_file[group].get('freq_c'))
            G = np.power(10, np.asarray( hdf5_file[group].get('Gain'))/10 )
            if device in devices_IMD:
                IIP3 = np.asarray( hdf5_file[group].get('IIP3') )
                IIP3 = dBm_to_Watts( IIP3 )
                g4_IMD_arr = kappa_c**2*freq_c/12/IIP3*(2*np.pi*h)/G**(3/2)         
            
            for l, index in enumerate(Happy_indices):

                print('%d/%d' %(l, len(Happy_indices)))
                
                f = Flux(hdf5_file[group]['current'][index], device.a, device.b)
                Fluxes_theor[l] = f
    
                
                freq_p = hdf5_file[group]['Pump_freq'][index]
                freq_c = np.asarray(hdf5_file[group]['freq_c'][index])
                freq_s = hdf5_file[group]['CW_freq'][index]
                freq_i = freq_p - freq_s
                center_freq[l] = freq_c
                
                if device in devices_IMD:
                    IIP3[l] = hdf5_file[group]['IIP3'][index]

                if device in [spa14_v2,spa14_v4]:
                    fudge_factor = -2.36
                else:
                    fudge_factor = float(hdf5_file['g3_fit'].attrs.get('fudge_factor'))


                _g3_ =  device.g3_distributed(f) * fudge_factor
                
                _g4_ =  device.g4_distributed(f) 
                if device in devices_IMD:
                    g4_IMD = g4_IMD_arr[index]     
    
                G = np.power(10,hdf5_file[group]['Gain'][index]/10)
                Delta = freq_c - freq_p/2
                omega = freq_s - freq_p/2
                Bw = hdf5_file[group]['Bandwidth'][index]
                kappa_ripple = np.sqrt(G*Bw**2 - Delta**2)
                kappa_t = hdf5_file[group]['kappa_t'][index]
                kappa_c = hdf5_file[group]['kappa_c'][index]
                
    
                u_p = pump_strength_for_gain( _g4_, _g3_, Delta, omega, kappa_t, freq_c, G)
                alpha_p = u_p/freq_c 
                
                P_p = float (freq_c/kappa_c*h*u_p**2*2*np.pi)
                Pump_dBm = Watts_to_dBm(P_p)
                Pump_Powers_theor[l] = Pump_dBm
            
            
#                for i, P_s in enumerate(Signal_Powers):
#    
#                    u_s = np.sqrt(P_s/(2*np.pi*h)*kappa_c/freq_c )
#                    
#                    _Xp_guess = float(alpha_p)
#                    _Yp_guess = 0
#                    
#                    _Xs_guess = float( -u_s*( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )*(Delta+omega) )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
#                    _Ys_guess = float( -kappa_t/2*( (Delta+omega)**2+kappa_t**2/4-16*alpha_p**2*_g3_**2  )*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
#        
#                    _Xi_guess = float( 4*_g3_*alpha_p*u_s*( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 ) )
#                    _Yi_guess = -4*_g3_*alpha_p*kappa_t*omega*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*_g3_**2 )**2 + (omega*kappa_t)**2 )
#      
#                    Gains_IMD[l,i], nbar_pump[l,i], nbar_signal[l,i], nbar_idler[l,i], Delta_effective[l,i] = Gain_IMD(freq_s, freq_i, freq_p, _g4_, _g3_, Delta, omega, kappa_t, freq_c, u_p, u_s, _Xp_guess, _Yp_guess, _Xs_guess, _Ys_guess, _Xi_guess, _Yi_guess)          
#                    Gains[l,i] = Gain(freq_s, freq_i, freq_p, _g4_, _g3_, Delta, omega, kappa_t, freq_c, u_p, u_s, _Xp_guess, _Yp_guess, _Xs_guess, _Ys_guess, _Xi_guess, _Yi_guess)[0]     
#                    if device in devices_IMD:
#                        Gains3[l,i] = Gain(freq_s, freq_i, freq_p, g4_IMD, _g3_, Delta, omega, kappa_t, freq_c, u_p, u_s, _Xp_guess, _Yp_guess, _Xs_guess, _Ys_guess, _Xi_guess, _Yi_guess)[0]     
                
                
                ind_compression = find_1dB_compression_point(Gains[l], Gains[l][0])
                P_1dB_theor[l] = Signal_Powers_dBm[ ind_compression ]

                ind_compression = find_1dB_compression_point(Gains_IMD[l], Gains_IMD[l][0])
                P_1dB_theor_IMD[l] = Signal_Powers_dBm[ ind_compression ]

                ind_compression = find_1dB_compression_point(Gains3[l], Gains3[l][0])
                P_1dB_theor_3[l] = Signal_Powers_dBm[ ind_compression ]

#                print('Flux %.3f | Pump power %.1f | Gain %.2f' % (f,Pump_dBm, Gains[l][0]) )
#
#                print('n_p, n_s, n_i INITIAL')
#                print(nbar_pump[l,0], nbar_signal[l,0], nbar_idler[l,0] )
#                
#                print('n_p, n_s, n_i COMPRESSION')
#                print(nbar_pump[l,ind_compression], nbar_signal[l,ind_compression], nbar_idler[l,ind_compression] )
                P_1dB_exp[l] = np.array(hdf5_file[group]['P_1dB'][index])
                
                
                Gain_sat = np.power(10,(hdf5_file[group]['Gain'][index]-1)/10)
                G = np.power(10,hdf5_file[group]['Gain'][index]/10)
                _g4_ = _g4_
#                _g4_ = g4_IMD
#                P_1dB_theor_Stark[l] = np.abs(kappa_c**2*freq_c/24/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h))
                P_1dB_theor_Stark[l] = kappa_c**2*freq_c/24/_g4_/Gain_sat*( -np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h) if _g4_<0 else kappa_c**2*freq_c/24/_g4_/Gain_sat*( np.sqrt( (Delta/kappa_c)**2+np.sqrt(1/4+(Delta/kappa_c)**2)*(1/np.sqrt(Gain_sat)-1/np.sqrt(G)) ) - Delta/kappa_c)*(2*np.pi*h)

                print(P_1dB_theor_Stark[l])
                P_1dB_theor_Stark[l] = Watts_to_dBm( P_1dB_theor_Stark[l] )
                print(P_1dB_theor_Stark[l])
        finally:
            hdf5_file.close()
               
            
            
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------
#   ---------------------------- PLOTTING --------------------------------
      
#        x=-150 is a point where calculation doesn't converge, so I remove it from the array.
        indexes = [i for i,x in enumerate(P_1dB_theor) if x == -150 or x==-80 ]
        X = [x for i,x in enumerate(P_1dB_theor) if i not in indexes]
        Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]
        axIMD.plot(np.asarray(Y), X, color='blue', linestyle='-', label=r'$1^{\rm st}$  order harmonic balance')
                
        if device in devices_IMD:

            indexes = [i for i,x in enumerate(P_1dB_theor_3) if x == -150 or x==-80 ]
            X = [x for i,x in enumerate(P_1dB_theor_3) if i not in indexes]
            Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]            
            axIMD.plot(np.asarray(Y), X, color='red', linestyle='-', label=r'Using empirical $g_4$ from $IIP_3$')

        indexes = [i for i,x in enumerate(P_1dB_theor_IMD) if x == -150 or x==-80 ]
        X = [x for i,x in enumerate(P_1dB_theor_IMD) if i not in indexes]
        Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]
        axIMD.plot(np.asarray(Y), X, color='green', linestyle='-', label=r'$2^{\rm nd}$ order harmonic balance')

        indexes = [i for i,x in enumerate(P_1dB_theor_Stark) if x == -150 or x==-80 ]
        X = [x for i,x in enumerate(P_1dB_theor_Stark) if i not in indexes]
        Y = [y for i,y in enumerate(Fluxes_theor) if i not in indexes]
        axIMD.plot(np.asarray(Y), X, color='purple', linestyle='-', label=r'Stark shift theory')



        axIMD.plot(Fluxes_theor, P_1dB_exp, color=colors[device_num], linestyle='-', marker='o', markersize=1.5, label=r'Data ') #color=colors[device_num] label=r'$P_{-1 \rm  dB}$

    
#    figIMD.legend(loc='best')
    axIMD.legend(loc='best')
    figIMD.tight_layout()

    
    