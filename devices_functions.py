# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 --:--:-- 2018

@author: Vladimir
"""

from __future__ import division
import h5py
import numpy as np
from scipy.optimize import fminbound, curve_fit, fsolve, brentq
from time import strftime
import matplotlib.pyplot as plt
import matplotlib as mpl
import fractions
from scipy.special import factorial
import matplotlib.colors as plt_colors

def lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0



R_Q = 6.44026e3             # resistance quantum in Ohms, as in h/(2e)^2
h = 6.62607e-34             # Plack constant in SI units
hbar = h/(2*np.pi)
R = 50.0                    # transmission line characteristic impedance
Phi_0 = 3.29106e-16         # flux quantum (defined with hbar)



# --------------------------------------------------------------------------------------
# ----- SNAIL expansion coefficients ---------------------------------------------------

n_default = 3
m_default = 1

def H(phi, alpha, f, n=n_default, m=m_default): 
    return -m*alpha*np.cos(phi/m) - n*np.cos((2*np.pi*f-phi)/n)

def c1(phi, alpha, f, n=n_default, m=m_default):
    return alpha*np.sin(phi/m) - np.sin((2*np.pi*f-phi)/n)

def c2(phi, alpha, f, n=n_default, m=m_default):
    return (1.0/m)*alpha*np.cos(phi/m) + 1.0/n*np.cos((2*np.pi*f-phi)/n)

def c3(phi, alpha, f, n=n_default, m=m_default):
    return -(1.0/m**2)*alpha*np.sin(phi/m) + 1.0/n**2*np.sin((2*np.pi*f-phi)/n)
    
def c4(phi, alpha, f, n=n_default, m=m_default):
    return -(1.0/m**3)*alpha*np.cos(phi/m) - 1.0/n**3*np.cos((2*np.pi*f-phi)/n)

def c5(phi, alpha, f, n=n_default, m=m_default):
    return (1/m**4)*alpha*np.sin(phi/m) - 1.0/n**4*np.sin((2*np.pi*f-phi)/n)

def c6(phi, alpha, f, n=n_default, m=m_default):
    return (1/m**5)*alpha*np.cos(phi/m) + 1.0/n**5*np.cos((2*np.pi*f-phi)/n)


# returns the offset phase for a given flux or array of fluxes
def offset(f,alpha,n=n_default,m=m_default):
    f = [f] if isinstance(f, float) or isinstance(f, int) else f
    phi=[]
    for flux in f:
        phi = phi + [fminbound(H,-lcm(n,m)*np.pi,lcm(n,m)*np.pi,args=(alpha,flux,n,m))]
    return np.asarray(phi)



# --------------------------------------------------------------------------------------
# ----- RF SQUID expansion coefficients ------------------------------------------------

#def H(phi, alpha, n, f): 
#    return -np.cos(phi-2*np.pi*f)+1/(3*alpha)*phi**2/2
#
#def c1(phi, alpha, f, n=3):
#    return np.sin(phi-2*np.pi*f) + phi*1/(3*alpha)
#
#def c2(phi, alpha, f, n=3):
#    return np.cos(phi-2*np.pi*f) + 1/(3*alpha)
#
#def c3(phi, alpha, f, n=3):
#    return np.sin(2*np.pi*f-phi)
#    
#def c4(phi, alpha, f, n=3):
#    return -np.cos(2*np.pi*f-phi)
#
#def offset(f,alpha,n=3):
#    f = [f] if isinstance(f, float) or isinstance(f, int) else f
#    phi=[]
#    for flux in f:
#        phi = phi + [fminbound(H,-np.pi,np.pi,args=(alpha,n,flux))]
#    return np.asarray(phi)

# when using offset function with RF SQUID, n = 3 which currently stays as defualt will
# search over a 3 times wider flux range than necessary and will see additional minima,
# which is bad. Need to go over the code and make sure that it doesn't use 3 by default 



# ----- Different helpful functions ------------------------------------------------
# --------------------------------------------------------------------------------------

# looks for 3dB bandwidth in the given gain profile
def find_3dB_bandwidth(gain_profile, ind_CW):
    temp = True
    ind = ind_CW
    while temp:
        if (gain_profile[ind_CW]-gain_profile[ind]-3)>0 and np.abs((gain_profile[ind]-gain_profile[ind-1]))<1:
            temp = False
            ind_right = ind
        else:
            ind = ind + 1
            if ind==len(gain_profile):
                temp = False
                ind_right = ind-1
    temp = True
    ind = ind_CW
    while temp:
        if (gain_profile[ind_CW]-gain_profile[ind]-3)>0 and np.abs((gain_profile[ind]-gain_profile[ind-1]))<1:
            temp = False
            ind_left = ind
        else:
            if ind==0:
                temp = False
                ind_left = 0
            ind = ind - 1
    return ind_left, ind_right

def find_1dB_compression_point(gains, gain_ref):
    if all(G<(gain_ref+1) for G in gains):
        Gain_sat = gain_ref-1
    else:
        Gain_sat = gain_ref+1
    ind = np.abs(gains - Gain_sat).argmin()
    return ind



# converts a compex phase to angle and logmag of power
def complex_to_PLOG(a_complex):
    logmag=20*np.log10(np.abs(a_complex))
    phase=np.angle(a_complex, deg='degrees')
    return (logmag,phase)

# Converts between power units
def Watts_to_dBm(x):
    return 30+10*np.log10(x)

# Converts between power units
def dBm_to_Watts(x):
    return np.power(10,(x-30)/10.0)

# Flux in fractions of flux quantum, current in A
def Flux(I,a,b):
    return a*I+b

# Current in A, flux in fractions of flux quantum
def Current(f,a,b):
    return (f-b)/a 

# converts int numbers to string for date/time
def float_to_DateTime(x):
    t = str( int(x))
    if len(t)<6:
        t = '0'*(6-len(t))+t
    return t



def derivative_smooth(data, N_smooth=0):
    derivative = np.zeros(len(data))
    for i in range(0,len(data)):
        if i == 0:
            derivative[i] = data[i+1]-data[i]
        elif i==len(data)-1:
            derivative[i] = data[len(data)-1]-data[len(data)-2]
        else:
            derivative[i] = 1/2*data[i+1]-1/2*data[i-1]
    if N_smooth:
        temp = np.zeros(len(data))
        for j in range(0,N_smooth):
            for i in range(0,len(data)):
                temp[i]=np.mean( derivative[max(0,i-1):min(len(data),i+2)] )
#                if i==0:
#                    temp[i] = (derivative[i+1]+derivative[i])/2
#                elif i==len(data)-1:
#                    temp[i]=(derivative[i-1]+derivative[i-2])/2
#                else:
#                    temp[i]=(derivative[i-1]+derivative[i]+derivative[i+1])/3
            derivative = temp
    return np.asarray(derivative)
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------



# ------ Attenuation----------------------------------------------------------
# --------------------------------------------------------------------------------------


# extracts attenuation data from hdf5 file 
# linearly extrapolates to arbitrary inremediate frequency
def extract_attenuation(filename, date, time, meas_name, freq, txt=False):
    if txt:
        array = np.loadtxt(filename)
        freqs_atten = array[:,0]
        real = array[:,1]
        imag = array[:,2]
    else:
        hdf5 = h5py.File(filename)
        try:
            freqs_atten = np.asarray(hdf5[date][time]['LIN'].get('frequencies'))
            imag = np.asarray(hdf5[date][time]['LIN'][meas_name].get('imag'))
            real = np.asarray(hdf5[date][time]['LIN'][meas_name].get('real'))
        finally:
            hdf5.close()       
    atten = complex_to_PLOG(np.vectorize(complex)(real, imag))[0]
    if isinstance(freq, float):
        ind = np.abs(freqs_atten-freq).argmin()
        return atten[ind]
    else:
        attenuation = np.zeros(len(freq))
        for l, F in enumerate(freq):
            ind = np.abs(freqs_atten-F).argmin()
            attenuation[l] = atten[ind]
        return attenuation


# extracts attenuation data for a given device and frequency
def attenuation(freq, filename, group_name):
    
    hdf5_file = h5py.File(filename)
    attenuation = 0.0
    try:
        for component in hdf5_file[group_name].keys():
            date = hdf5_file[group_name][component].attrs.get('date')
            time = hdf5_file[group_name][component].attrs.get('time')
            meas_name = hdf5_file[group_name][component].attrs.get('meas_name')
            attenuation += extract_attenuation(filename, date, time, meas_name, freq)
    finally:
        hdf5_file.close()
    return attenuation
        
                
        
    

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# ------- Distributed Resonator Model for SPA------------------------------
# ---------------------------------------------------------------------------------


def helper_eq(freq, f, alpha, gamma, f0_distributed, M, mu=0.5):
    return gamma*c2(offset(f,alpha), alpha, f)/M/(2*np.pi*np.abs(freq)) -  2*np.sin(mu*np.pi*freq/f0_distributed)*np.sin((1-mu)*np.pi*freq/f0_distributed)/np.sin(np.pi*freq/f0_distributed)


# resonant frequency that the resonator would have without any couplings (distributed model)
def freq_distributed_without_coupling(I, a, b, alpha, f0_distributed, M, gamma, num_mode=1, mu=0.5):
    f = Flux(I,a,b)
    # special case when snail array is in the middle, even modes ideally don't couple
    if (mu==0.5) and (num_mode %2 == 0):
        if isinstance(f, float) or isinstance(f, int):
            return f0_distributed*num_mode
        else:
            return f0_distributed*num_mode*np.ones(len(f))
    else:
        if isinstance(f, float) or isinstance(f, int):
            return brentq(helper_eq,(num_mode-1)*f0_distributed+1e5,(num_mode)*f0_distributed-1e5,args=(f, alpha, gamma, f0_distributed, M, mu) )
        else:
            freqs=[]
            for flux in f:
                freqs = freqs+ [brentq(helper_eq, (num_mode-1)*f0_distributed+1e5,(num_mode)*f0_distributed-1e5,args=(flux, alpha, gamma, f0_distributed, M, mu) )]
            return np.asarray(freqs)


# internal capacitance in the effective flux-dependent LC model
def capacitance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_cap=1, mu=0.5):
    omega = 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=num_mode_cap, mu=mu)
    omega0 = 2*np.pi*f0_distributed
    return ((np.sin(np.pi*omega/omega0*(1-mu)))**2*( mu/2*np.pi/omega0/Z_c + 1.0/4/omega/Z_c*np.sin(2*np.pi*mu*omega/omega0))+(np.sin(np.pi*omega/omega0*mu))**2*( (1-mu)/2*np.pi/omega0/Z_c + 1/4/omega/Z_c*np.sin(2*np.pi*(1-mu)*omega/omega0)))/(np.sin(np.pi*omega/omega0*mu))**2


# internal inductance in the effective flux-dependent LC model
def inductance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_ind=1, mu=0.5):
    omega = 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=num_mode_ind, mu=mu)
    return 1/omega**2/capacitance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_cap=num_mode_ind, mu=mu)

# fit for resonant frequency in the presence of coupling capacitor
def freq_distributed_with_coupling(I, a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mode=1, mu=0.5):
    f = Flux(I,a,b)
    return 1/np.sqrt( inductance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_ind=mode, mu=mu)*(capacitance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_cap=mode, mu=mu) + C_coupling) ) / (2*np.pi)


# kappa_coupling calculated from the coupling capacitor (value in Hz)
# a good approximation only when it is small compared to omega_c
def kappa(f, a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mode=1, mu=0.5, Res=50, direct=False):
    if direct: 
        k_c = 1/capacitance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_cap=mode, mu=mu)/Res/(2*np.pi)
    else:
        omega = 2*np.pi*np.asarray( freq_distributed_with_coupling(Current(f,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mode=mode, mu=mu)  )
        k_c = C_coupling**2*omega**2*R/capacitance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_cap=mode, mu=mu)/(2*np.pi)/ (1+(omega*R*C_coupling)**2 )
    return k_c


# participation ratio of array based on distributed model
def participation(f, a, b, alpha, f0_distributed, Z_c, M, L_j, mode=1, mu=0.5):
    gamma = 2*Z_c/L_j
    L_array = M*L_j/c2(offset(f,alpha), alpha, f)
    return L_array/( L_array + inductance(f, a, b, alpha, f0_distributed, Z_c, M, gamma, num_mode_ind=mode, mu=mu) )


# This is g3 for the main mode
def g3_distributed(f, a, b, alpha, f0_distributed, M, L_j, Z_c, num_mode=1, mu=0.5):
    gamma = 2*Z_c/L_j
    omega = 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=num_mode, mu=mu)
    omega0 = 2*np.pi*f0_distributed
    c3_ = c3(offset(f,alpha), alpha, f)
    return c3_/M**2*(np.sin(np.pi*omega/omega0))**3 * np.sqrt( omega/2/np.pi/R_Q ) / ( 12*np.sqrt(2)*omega**2*L_j ) / ( (np.pi*mu/2/omega0/Z_c+1/4/omega/Z_c*np.sin(np.pi*2*mu*omega/omega0) )*(np.sin(np.pi*omega/omega0*(1-mu)) )**2 + (np.pi*(1-mu)/2/omega0/Z_c+1/4/omega/Z_c*np.sin(np.pi*2*(1-mu)*omega/omega0) )*(np.sin(np.pi*omega/omega0*mu) )**2 )**(3/2)
 

# This is g4 for the main mode
def g4_distributed(f, a, b, alpha, f0_distributed, M, L_j, Z_c, num_mode=1, mu=0.5, omega_optional = 0):
    gamma = 2*Z_c/L_j
    omega = omega_optional if omega_optional else 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=num_mode, mu=mu)
#    omega = 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=num_mode, mu=mu)
    omega0 = 2*np.pi*f0_distributed
    c4_ = c4(offset(f,alpha), alpha, f)
    c3_ = c3(offset(f,alpha), alpha, f)
    c2_ = c2(offset(f,alpha), alpha, f)
    # these huge formulas are based on the distributed resonator model for SPA
    monster1 = (np.pi*mu/2/omega0/Z_c+1/4/omega/Z_c*np.sin(np.pi*2*mu*omega/omega0) )*(np.sin(np.pi*omega/omega0*(1-mu)) )**2 + (np.pi*(1-mu)/2/omega0/Z_c+1/4/omega/Z_c*np.sin(np.pi*2*(1-mu)*omega/omega0) )*(np.sin(np.pi*omega/omega0*mu) )**2 
    monster2 = (np.sin(np.pi*omega/omega0))**2/48/c2_/R_Q*M
    monster3 = np.pi*c2_*Z_c*(1-mu)/omega0/L_j/M/( np.sin(np.pi*omega/omega0*(1-mu)) )**2 + np.pi*c2_*Z_c*mu/omega0/L_j/M/( np.sin(np.pi*omega/omega0*mu) )**2+1
    monster4 = c4_/M**3 - 2*c3_**2/c2_/M**3-c3_**2/( c2_/M - 2*omega/Z_c*L_j*np.tan(np.pi*2*mu*omega/omega0)*np.tan(np.pi*2*(1-mu)*omega/omega0)/( np.tan(np.pi*2*mu*omega/omega0) + np.tan(np.pi*2*(1-mu)*omega/omega0) ) )/M**4
    return monster2/monster1/monster3*monster4



# this gives third order couplings between all modes, see SPA notes
# <modes> should be array of size 3 containing mode indices
def g3_modes(f, a, b, alpha, L_j, f0_distributed, M, E_j, mu, Z_c, modes):
    c3_ = c3(offset(f,alpha), alpha, f)
    c2_ = c2(offset(f,alpha), alpha, f)
    L_s=L_j/c2_
    omega0=f0_distributed*2*np.pi
    g3 = c3_*M/6*E_j
    gamma = 2*Z_c/L_j
    # some combinatorial factors
    if (modes[0] != modes[1]) and (modes[1] != modes[2]) and (modes[0] != modes[2]):
        g3 *= 6
    elif (modes[0]==modes[1]) and (modes[0]==modes[2]):
        g3 *= 1
    else:
        g3 *= 3
    for k in modes:
        omega_k = 2*np.pi*freq_distributed_without_coupling(Current(f,a,b), a, b, alpha, f0_distributed, M, gamma, num_mode=k, mu=mu)
        g3 *= np.sqrt( omega_k*L_s/(R_Q/2/np.pi)/(M + np.pi*mu*Z_c/omega0/L_s/(np.sin(mu*np.pi*omega_k/omega0))**2+ np.pi*(1-mu)*Z_c/omega0/L_s/(np.sin((1-mu)*np.pi*omega_k/omega0))**2)) 
    return g3


# Determines f0_distributed for a given max frequency and other params
def f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M, mode=1):
    return brentq(helper_eq_opt, freq_max, 100*freq_max, args=(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mode, mu, M) )

def helper_eq_opt(f0_distributed, freq_max, a, b, alpha, Z_c, gamma, C_coupling, mode, mu, M):
    return freq_max-freq_distributed_with_coupling(Current(0,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mode=mode, mu=mu)


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------




# ------ Simple LC Model for SPA (it sucks) ------------------------------
# ---------------------------------------------------------------------------------



# ----- Expansion coefficients in LC model ---------------------------------------------------

def c2_eff(f, p_j, alpha, M):
    return p_j*c2(offset(f,alpha), alpha, f)/(M*p_j+c2(offset(f,alpha), alpha, f))

def c3_eff(f, p_j, alpha, M):
    return M*p_j**3*c3(offset(f,alpha), alpha, f)/(M*p_j+c2(offset(f,alpha), alpha, f))**3

def c4_eff(f, p_j, alpha, M):
    return M*c4(offset(f,alpha), alpha, f)*p_j**4/(c2(offset(f,alpha), alpha, f)+M*p_j)**4-3*M*(p_j**4)*(c3(offset(f,alpha), alpha, f))**2/(c2(offset(f,alpha), alpha, f)+M*p_j)**5

def c5_eff(f, p_j, alpha, M):
    p = p_j/(M*p_j+c2(offset(f,alpha), alpha, f))
    return p**5/M**4*( c5(offset(f,alpha), alpha, f) -10*c4(offset(f,alpha), alpha, f)*c3(offset(f,alpha), alpha, f)/c2(offset(f,alpha), alpha, f)*(1-p)+15*(c3(offset(f,alpha), alpha, f))**2/(c2(offset(f,alpha), alpha, f))**2*(1-p)**2 )

def c6_eff(f, p_j, alpha, M):
    p = M*p_j/(M*p_j+c2(offset(f,alpha), alpha, f))
    return p**6/M**5*( c6(offset(f,alpha), alpha, f) - (15*c5(offset(f,alpha), alpha, f)*c3(offset(f,alpha), alpha, f)+10*(c4(offset(f,alpha), alpha, f))**2)/c2(offset(f,alpha), alpha, f)*(1-p)+105*c4(offset(f,alpha), alpha, f)*(c3(offset(f,alpha), alpha, f))**2/(c2(offset(f,alpha), alpha, f))**2*(1-p)**2-105*(c3(offset(f,alpha), alpha, f))**4/(c2(offset(f,alpha), alpha, f))**3*(1-p)**3     ) 


def c4_eff_BBQ(f, p_j, alpha, M):
    return M*c4(offset(f,alpha), alpha, f)*p_j**4/(c2(offset(f,alpha), alpha, f)+M*p_j)**4

#def c5_eff(f, p_j, alpha, M):
#    return M*c5(offset(f,alpha), alpha, f)*p_j**5/(c2(offset(f,alpha), alpha, f)+M*p_j)**5 - 15*M*(c5(offset(f,alpha), alpha, f))**3*p_j**5/(c2(offset(f,alpha), alpha, f)+M*p_j)**7 - 10*M*c4(offset(f,alpha), alpha, f)*c3(offset(f,alpha), alpha, f)*p_j**5/(c2(offset(f,alpha), alpha, f)+M*p_j)**6

# theory formula for resonant frequency in LC model
def f_theory_LC(I, a, b, alpha, p_j, f0, M):
    if isinstance(I, float) or isinstance(I, int):
        I=np.asarray(I)
    f = a*I+b
    freq=np.zeros(len(I))
    offset_=offset(f,alpha)
    freq = f0/np.sqrt(1+M*p_j/c2(offset_,alpha,f))
    return freq

# ----- Nonlinearities in LC model, all values in Hz ----------------
def g4_LC(f, p_j, alpha, M, C):
    return c4_eff(f, p_j, alpha, M)/c2_eff(f, p_j, alpha, M)/4/24/R_Q/C

def chi_LC(f, p_j, alpha, M, C):                                             # definition that people usually use
    return 12*g4_LC(f, p_j, alpha, M, C)

def g3_LC(f, a, b, p_j, alpha, M, C, f0):
    return c3_eff(f, p_j, alpha, M)/c2_eff(f, p_j, alpha, M)/12/np.sqrt(2)*np.sqrt(f_theory_LC(Current(f,a,b), a, b, alpha, p_j, f0, M)/C/R_Q)

def g4_corrected_LC(f, a, b, p_j, alpha, M, C, f0):
    return g4_LC(f, p_j, alpha, M, C)-5*(g3_LC(f, a, b, p_j, alpha, M, C, f0))**2/f_theory_LC(Current(f,a,b), a, b, alpha, p_j, f0, M)

def g4_BBQ_LC(f, p_j, alpha, M, C):
    return c4_eff_BBQ(f, p_j, alpha, M)/c2_eff(f, p_j, alpha, M)/4/24/R_Q/C

def g5_LC(f, a, b, p_j, alpha, M, C, f0):
    return 1/h*2**(5-3)/factorial(5)/c5_eff(f, p_j, alpha, M)*c5_eff(f, p_j, alpha, M)*M*(h/8/R_Q/C)**(5/2-1)*(h*f_theory_LC(Current(f,a,b), a, b, alpha, p_j, f0, M))**(2-5/2)

def g6_LC(f, a, b, p_j, alpha, M, C, f0):
    return 1/h*2**(6-3)/factorial(6)/c2_eff(f, p_j, alpha, M)*c6_eff(f, p_j, alpha, M)*(h/8/R_Q/C)**(6/2-1)*(h*f_theory_LC(Current(f,a,b), a, b, alpha, p_j, f0, M))**(2-6/2)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


# ---- Flux sweep fits fr SPA (different models) --------------------------
# ------------------------------------------------------------------------


#    This is just a rough fit mainly to get a flux calibration and use it in later
#    fitting to distributed model or whatever. LC model just sucks
def fit_flux_sweep_LC(I_data, F_data, a, b, alpha, p_j, M, L_j):
    
    guess = [a,b,alpha,p_j,max(F_data)+1e8]
    popt, pcov = curve_fit(lambda x,p1,p2,p3,p4,p5 : f_theory_LC(x,p1,p2,p3,p4,p5,M), I_data, F_data, p0=guess)
    a, b, alpha, p_j, f0 = popt          
    C = p_j/(2*np.pi*f0)**2/L_j # mode capacitance estimate
    return a, b, alpha, p_j, f0, C




# Sometimes flux tuned resonance frequency goes beyound directional coupler 
# rolloff, and for that fluxes fit diverges, so this scetchy thing  
# removes such points. 
def fit_flux_sweep_helper(currents, f0_exp_fit, kc_exp_fit, freq_max, freq_min): 
    freq_array = np.empty(0)
    curr_array = np.empty(0)
    kc_array = np.empty(0)      
    for freq_ind, freq in enumerate(f0_exp_fit):
        if (freq < freq_max - 150e6) and (freq > freq_min + 50e6) and (kc_exp_fit[freq_ind]>6e6):
            freq_array = np.append(freq_array,f0_exp_fit[freq_ind])
            curr_array = np.append(curr_array,currents[freq_ind])
            kc_array = np.append(kc_array,kc_exp_fit[freq_ind])
    return curr_array, freq_array, kc_array
    

#   This fit should be done after the LC model fit, it relies on a & b being fitted already.
#   It iterates fitting of resonant frequency and kappa, not the smartest way to do it... need some kind of vector fitting
def fit_flux_sweep_distributed(I_data, F_data, kc_data, a, b, Z_c, gamma, M, alpha2_guess, f0_distributed_guess, C_coupling_guess, mu=0.5, N_iter=4):
    count = 0
    while count < N_iter:
        count+=1
        guess = [alpha2_guess,f0_distributed_guess]        
        popt, pcov = curve_fit(lambda x,p1,p2 : freq_distributed_with_coupling(x, a, b, p1, p2, Z_c, C_coupling_guess, M, gamma, mode=1, mu=mu), I_data, F_data, p0=guess)
        alpha2_guess, f0_distributed_guess = popt  
        
        guess= [C_coupling_guess]
        popt, pcov = curve_fit(lambda x,p1 : kappa(x, a, b, alpha2_guess, f0_distributed_guess, Z_c, p1, M, gamma, mu=mu), Flux(I_data,a,b), kc_data, p0=guess)
        C_coupling_guess = popt[0]
        print(' f0_distributed %f GHz \n alpha2 %f \n C_coupling %f pF \n' %(f0_distributed_guess*1e-9, alpha2_guess, C_coupling_guess*1e12) )
    return alpha2_guess, f0_distributed_guess, C_coupling_guess



#   This fit should be done after the LC model fit, it relies on a & b being fitted already.
#   It iterates fitting of resonant frequency and kappa. Z_c is a fit parameter, unlike in the previous version
def fit_flux_sweep_distributed_2(I_data, F_data, kc_data, a, b, Z_c_guess, gamma, M, L_j, alpha2_guess, f0_distributed_guess, C_coupling_guess, mu=0.5, N_iter=4):
    count = 0
    while count < N_iter:
        count+=1
        guess = [alpha2_guess, f0_distributed_guess, Z_c_guess]
        popt, pcov = curve_fit(lambda x,p1,p2,p3 : freq_distributed_with_coupling(x, a, b, p1, p2, p3, C_coupling_guess, M, 2*p3/L_j, mode=1, mu=mu), I_data, F_data, p0=guess)
        alpha2_guess, f0_distributed_guess, Z_c_guess = popt  
        
        guess = [C_coupling_guess]
        popt, pcov = curve_fit(lambda x,p1 : kappa(x, a, b, alpha2_guess, f0_distributed_guess, Z_c_guess, p1, M, 2*Z_c_guess/L_j, mu=mu), Flux(I_data,a,b), kc_data, p0=guess)
        C_coupling_guess = popt[0]
        print(' f0_distributed %f GHz \n alpha2 %f \n C_coupling %f pF \n Z_c %f \n' %(f0_distributed_guess*1e-9, alpha2_guess, C_coupling_guess*1e12, Z_c_guess) )
    return alpha2_guess, f0_distributed_guess, Z_c_guess, C_coupling_guess


#   This fit should be done after the LC model fit, it relies on a & b being fitted already.
#   It doesn't iterate fitting of res freq and kappa, because this is written for a directly coupled resonator, 
#   in which frequency is not loaded down by the coupling capacitor. This fits the damping resistance for the resonator.
def fit_flux_sweep_distributed_3(I_data, F_data, kc_data, a, b, Z_c, gamma, M, L_j, alpha2_guess, f0_distributed_guess, Res_guess, mu=0.5):

    
    guess = [alpha2_guess, f0_distributed_guess]
    popt, pcov = curve_fit(lambda x,p1,p2 : freq_distributed_without_coupling(x, a, b, p1, p2, M, gamma, num_mode=1, mu=0.5), I_data, F_data, p0=guess)
    alpha2_guess, f0_distributed_guess = popt  
        
    guess = [Res_guess]
    popt, pcov = curve_fit(lambda x,p1 : kappa(x, a, b, alpha2_guess, f0_distributed_guess, Z_c, 0, M, gamma, mode=1, mu=0.5, Res=p1, direct=True), Flux(I_data,a,b), kc_data, p0=guess)
    Res_guess = popt[0]

    return alpha2_guess, f0_distributed_guess, Res_guess



#   This fit should be done after the LC model fit, it relies on a & b being fitted already.
#   It iterates fitting of resonant frequency and kappa, not the smartest way to do it... need some kind of vector fitting
def fit_flux_sweep_distributed_4(I_data, F_data, kc_data, a, b, Z_c, gamma, M, alpha2, f0_distributed_guess, C_coupling_guess, mu=0.5, N_iter=4):
    count = 0
    while count < N_iter:
        count+=1
        guess = [f0_distributed_guess]        
        popt, pcov = curve_fit(lambda x,p2 : freq_distributed_with_coupling(x, a, b, alpha2, p2, Z_c, C_coupling_guess, M, gamma, mode=1, mu=mu), I_data, F_data, p0=guess)
        f0_distributed_guess = popt  
        
        guess= [C_coupling_guess]
        popt, pcov = curve_fit(lambda x,p1 : kappa(x, a, b, alpha2, f0_distributed_guess, Z_c, p1, M, gamma, mu=mu), Flux(I_data,a,b), kc_data, p0=guess)
        C_coupling_guess = popt[0]
        print(' f0_distributed %f GHz \n alpha2 %f \n C_coupling %f pF \n' %(f0_distributed_guess*1e-9, alpha2, C_coupling_guess*1e12) )
    return f0_distributed_guess, C_coupling_guess


#   This fit should be done after the LC model fit, it relies on a & b being fitted already.
#   It iterates fitting of resonant frequency and kappa, not the smartest way to do it... need some kind of vector fitting
def fit_flux_sweep_distributed_5(I_data, F_data, kc_data, a, b, Z_c, gamma_guess, M, alpha2, f0_distributed_guess, C_coupling_guess, mu=0.5, N_iter=4):
    count = 0
    while count < N_iter:
        count+=1
        guess = [gamma_guess,f0_distributed_guess]
        popt, pcov = curve_fit(lambda x,p1,p2 : freq_distributed_with_coupling(x, a, b, alpha2, p2, Z_c, C_coupling_guess, M, p1, mode=1, mu=mu), I_data, F_data, p0=guess)
        gamma_guess, f0_distributed_guess = popt  
        
        guess= [C_coupling_guess]
        popt, pcov = curve_fit(lambda x,p1 : kappa(x, a, b, alpha2, f0_distributed_guess, Z_c, p1, M, gamma_guess, mu=mu), Flux(I_data,a,b), kc_data, p0=guess)
        C_coupling_guess = popt[0]
        L_j_guess = 2*Z_c/gamma_guess
        print(' f0_distributed %f GHz \n L_j %f pH \n C_coupling %f pF \n' %(f0_distributed_guess*1e-9, L_j_guess*1e12, C_coupling_guess*1e12) )
    return gamma_guess, f0_distributed_guess, C_coupling_guess


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


# ------- Array modes ---------------------------------------------
# ------------------------------------------------------------------------

"""
Works for arbitrary transmission-line-like array WITH CAPACITANCE PADS.
Gives the frequency of the array mode N_mode, using dimensionless params x_c, y_c
and the plasma frequency f_p
"""
def freq_modes_with_cap(N_mode,x_c,y_c,f_p):
    eps = 1e-5
    N_mode = np.asarray([N_mode]) if isinstance(N_mode,int) or isinstance(N_mode,float) else np.asarray(N_mode) 
    freq=[]
    for ind, N in enumerate(N_mode):
        if x_c > 2/np.pi:
            lower_bound = max(0,np.pi*(N-2+1/2))
            upper_bound = np.pi*(N-1+1/2)
        else:
            if np.pi*(1/2+N)<1/x_c:
                lower_bound = np.pi*(N-1+1/2)
                upper_bound = np.pi*(N+1/2)
            elif np.pi*(1/2+N-2)>1/x_c:
                lower_bound = np.pi*(N-2+1/2)
                upper_bound = np.pi*(N-1+1/2)
            elif N == int( 1/np.pi/x_c-1/2 )+1:
                lower_bound = np.pi*(N-1+1/2)
                upper_bound = np.pi*(N-1+1)
            elif N == int( 1/np.pi/x_c-1/2 )+2:
                lower_bound = np.pi*(N-2+1)
                upper_bound = np.pi*(N-2+1+1/2)
        k = brentq(freq_modes_w_cap_helper,lower_bound+eps, upper_bound-eps, args=(x_c))
        f = float(f_p[ind]*np.sqrt( k**2/(k**2+y_c**2) ))
        freq.append(f)
    return np.asarray(freq)

def freq_modes_w_cap_helper(k,x_c):
    return (x_c**2*k**2-1)*np.tan(k)-2*x_c*k

"""
Same but for direct coupling
"""
def freq_modes_direct_coupl(N_mode,y_c,f_p):
    N_mode = np.asarray([N_mode]) if isinstance(N_mode,int) or isinstance(N_mode,float) else np.asarray(N_mode) 
    freq=[]
    for ind, N in enumerate(N_mode):
        f = float(f_p[ind]*np.sqrt( (N-1/2)**2/( (N-1/2)**2+y_c**2) ))
        freq.append(f)
    return np.asarray(freq)
    


"""
Works for arrays with the SNAIL unit cell.
Shape of X is (2,N): it has array of currents and array mode indices
Calls 'freq_modes_with_cap' after computing the SNAIL plasma frequency
"""
def freq_array_modes_SNAIL_with_cap(X, alpha, a, b, x_c, y_c, f_0):
    I = X[0]
    N_mode = X[1]
    f = Flux(I,a,b)
    f_p = np.asarray(f_0*np.sqrt( c2(offset(f,alpha), alpha, f) ))
    return freq_modes_with_cap(N_mode,x_c,y_c,f_p)


"""
Same but for direct coupling
"""
def freq_array_modes_SNAIL_direct_coupl(X, alpha, a, b, y_c, f_0):
    I = X[0]
    N_mode = X[1]
    f = Flux(I,a,b)
    f_p = np.asarray(f_0*np.sqrt( c2(offset(f,alpha), alpha, f) ))
    return freq_modes_direct_coupl(N_mode,y_c,f_p)

"""
Fit for the single flux point. 
Takes array of mode indices and resonant frequencies and fits x_c,y_c and f_p.
"""
def fit_array_modes_at_single_flux(N_data, F_data, x_c_guess, y_c_guess, f_p_guess):
    
    guess = [x_c_guess, y_c_guess, f_p_guess]
    bounds = (0, [1000, 1000, 1000])
    popt, pcov = curve_fit(lambda x,p1,p2,p3 : freq_modes_with_cap(x, p1, p2, p3), N_data, F_data, p0=guess, bounds=bounds)
    x_c, y_c, f_p = popt  
        
    return x_c, y_c, f_p

#"""
#Fit for the full two-tone spectroscopy flux sweep.
#Takes a list of currents, mode numbers and corresponding resonant frequencies.
#Does NOT fit alpha!!!
#"""   
#def fit_array_modes_flux_sweep(X_data,Y_data,a_guess, b_guess, x_c_guess, y_c_guess, f_0_guess):
#
#    alpha = 0.1
#    guess = [a_guess, b_guess, x_c_guess, y_c_guess, f_0_guess]
#    popt, pcov = curve_fit(lambda x,p1,p2,p3,p4,p5: freq_array_modes_SNAIL_with_cap(x,alpha,p1,p2,p3,p4,p5), X_data, Y_data, p0=guess)
#    a, b, x_c, y_c, f_0 = popt  
#    
#    return a, b, x_c, y_c, f_0 

"""
Fit for the full two-tone spectroscopy flux sweep.
Takes a list of currents, mode numbers and corresponding resonant frequencies.
Also fits alpha!!! USE FOR CAPACITIVELY COUPLED DEVICES !
"""    
def fit_array_modes_flux_sweep_v1(Currents, Mode_nums, Frequencies, a_guess, b_guess, x_c_guess, y_c_guess, f_0_guess, alpha_guess):
    
    X_data = np.asarray((Currents,Mode_nums))
    Y_data = np.asarray(Frequencies)
    guess = [alpha_guess, a_guess, b_guess, x_c_guess, y_c_guess, f_0_guess]
    popt, pcov = curve_fit(lambda x,p1,p2,p3,p4,p5,p6: freq_array_modes_SNAIL_with_cap(x,p1,p2,p3,p4,p5,p6), X_data, Y_data, p0=guess)
    alpha, a, b, x_c, y_c, f_0 = popt  
    
    return alpha, a, b, x_c, y_c, f_0 

"""
USE THIS FOR DIRECTLY COUPLED DEVICES !
"""
def fit_array_modes_flux_sweep_v2(Currents, Mode_nums, Frequencies, a_guess, b_guess, y_c_guess, f_0_guess, alpha_guess):
    
    X_data = np.asarray((Currents,Mode_nums))
    Y_data = np.asarray(Frequencies)
    guess = [alpha_guess, a_guess, b_guess, y_c_guess, f_0_guess]
    popt, pcov = curve_fit(lambda x,p1,p2,p3,p4,p5: freq_array_modes_SNAIL_direct_coupl(x,p1,p2,p3,p4,p5), X_data, Y_data, p0=guess)
    alpha, a, b, y_c, f_0 = popt  
    
    return alpha, a, b, y_c, f_0 




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

"""
Approximately (!!!) locates the resonances from the flux sweep data. 
Works even in the case when multiple modes are present.
"""

from sklearn.cluster import DBSCAN

def mode_detector_flux_sweep(h5filename, date, time, meas_name='"CH1_S11_1"', savepath = None, ftype='.png'):

    hdf5_file = h5py.File(h5filename,'r')
    try:
        frequencies = np.asarray(hdf5_file[date][time]['LIN'].get('frequencies'))
        currents = np.asarray(hdf5_file[date][time]['LIN'].get('currents'))
        real = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('real'))
        imag = np.asarray(hdf5_file[date][time]['LIN'][meas_name].get('imag'))
    finally:
        hdf5_file.close()
        
    # define the figure
    fig, axes = plt.subplots(2,2,figsize=(3.375*2, 1.5*2),dpi=240, sharex='col',sharey='row')
    define_phaseColorMap()


    A = real +1j*imag
    yphase = complex_to_PLOG(A)[1]
    a_phase = np.transpose(yphase)

    # first just plot the Phase data 
    ax = axes[0][0]
    p1 = ax.pcolormesh(currents*1e3, frequencies*1e-9, a_phase, cmap = 'phase')
#    fig.colorbar(p1, ax=ax, ticks=[-180, -90, 0, 90, 180], label=r'${\rm Phase \,(deg)}$')
    ax.set_ylabel(r'${\rm Frequency \,(GHz)}$')

    
    # Now convert to binary representation and plot that
    for i in range(a_phase.shape[0]-7):
        a_phase[i] = a_phase[i] - a_phase[i+7]
    a_phase[-1] = a_phase[-1]*0

    a_phase = np.abs(a_phase-90)
    threshold = 45
    for i in range(a_phase.shape[0]):
        for j in range(a_phase.shape[1]):
            if a_phase[i,j]>threshold:
                a_phase[i,j] = 0
            elif a_phase[i,j]<threshold:
                a_phase[i,j] = 1
    bin_rep = a_phase
    ax = axes[0][1]
    p1 = ax.pcolormesh(currents*1e3, frequencies*1e-9, bin_rep, cmap='binary')#colorMap[1])
#    fig.colorbar(p1, ax=ax, ticks=[-180, -90, 0, 90, 180], label=r'${\rm Contrast \,(0/1)}$')


    # convert the binary rep to the list of points on the 2D grid to which the clustering
    # algorithm will be applied
    points = []
    for i, bit_arr in enumerate(bin_rep):
        for j, bit in enumerate(bit_arr):
            if bit==1: points.append( [i,j] )

    # apply DBSCAN to separate points into clusters, use equal distance along both axes        
    db = DBSCAN(eps=20, min_samples=10).fit(points) 
    labels = db.labels_ #-1 labels the noise, i.e. points that don't have min_samples neighbors in their eps-vicinity 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Found %d clusters' %n_clusters)
    new_arr = np.ones(np.shape(bin_rep))*(-1)
    for index, point in enumerate(points):
        new_arr[point[0]][point[1]] = labels[index]

    # colorplot the clusters
    ax = axes[1][0]
    p3 = ax.pcolormesh(currents*1e3, frequencies*1e-9, new_arr, cmap='Set2_r') #tab20
#    fig.colorbar(p3, ax=ax, ticks=np.unique(labels), label=r'$\rm Cluster $')
    ax.set_ylabel(r'${\rm Frequency \,(GHz)}$')
    ax.set_xlabel(r'${\rm Current\,(mA)}$')        
    

    # select the average within each cluster as a proxy for the resonant frequency,
    # add clusters to the object array_modes()
    array_modes = {}
    for cluster_num in range(0,n_clusters):
        currs = []
        frqs = []
        for i, I in enumerate(currents):
            cluster_indices = [j for j, x in enumerate(new_arr.T[i]) if x == cluster_num]
            if cluster_indices:
                currs.append(I)
                ind = int(np.round(np.mean(np.asarray(cluster_indices))))
                frqs.append(frequencies[ind])
        array_modes[str(cluster_num)] = (np.array(currs),np.array(frqs))

    # Plot the extracted modes
    cmap = plt.get_cmap('Set2_r')
    ax = axes[1][1]
    for z in array_modes:
        ax.plot(array_modes[z][0]*1e3, array_modes[z][1]*1e-9, color=cmap(int(z)))
#    ax.set_ylabel(r'${\rm Frequency \,(GHz)}$')
    ax.set_xlabel(r'${\rm Current\,(mA)}$')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath + 'mode_detector_flux_sweep' + ftype, dpi=600)
        
    return array_modes
    