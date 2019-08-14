#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:08:27 2018

@author: Vladimir Sivak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound, brentq
from devices_functions import *
from SPA_plotting import *


"""Determinant equation for the resonance frequency"""
# EVEN modes
def eq1(omega, sign=1):
    return   sign*(  np.tan(d_ms*omega/2/v_ms)*np.tan(M*omega/2/omega_0/np.sqrt(1-omega**2/omega_p**2))-Z_ms/Z_j*np.sqrt(1-omega**2/omega_p**2)   )
# ODD modes
def eq2(omega, sign=1):
    return    sign*( np.tan(d_ms*omega/2/v_ms)/np.tan(M*omega/2/omega_0/np.sqrt(1-omega**2/omega_p**2))+Z_ms/Z_j*np.sqrt(1-omega**2/omega_p**2)   )


"""Defining the parameters"""

# Josephson Transmission Line
f = 0 # flux
alpha = 0.1
c_2 = c2(offset(f,alpha), alpha, f)
c_4 = c4(offset(f,alpha), alpha, f)

C_0 = 7.1e-16
C_j = 1.1e-13
L_j = 4.76e-11
L_s = L_j/c_2

Z_j = np.sqrt(L_j/C_0)
omega_0 = 1/np.sqrt(L_s*C_0)
omega_p = 1/np.sqrt(L_s*C_j)

a =  8e-7 #size of the unit cell (result should be independent of it)

# Microstrip
c = 3e8 #speed of light
Z_ms = 45.8
eps_ms = (11.45+1)/2  # TODO: check this!!!!
v_ms = c/np.sqrt(eps_ms)


# function that gives the resonant frequency of the JAMPA
def omega_arr_res(M,n):
    M = np.asarray(M)
    n = np.asarray(n)
    return omega_p/np.sqrt(1+(M*omega_p/np.pi/omega_0/n)**2)


# Desired maximal operating frequency

omega_max = 2*np.pi*8e9


""" Calculation of array modes and nonlinearities """

M_crit = int( np.round(np.pi*omega_0/omega_max*np.sqrt(1 - (omega_max/omega_p)**2)) )

M_arr = range(1, M_crit)
d_ms_list = np.zeros(len(M_arr))

eps = 1 #100 # stepping this much from the points where tan functions diverge


N_modes = 10            # interested in this many modes
N_modes_max = 100       # need to solve for this many


modes_arr_odd = np.zeros( (len(M_arr),N_modes) )
modes_arr_even = np.zeros( (len(M_arr),N_modes) )
g4_arr = np.zeros(len(M_arr))
g4_arr_odd = np.zeros( (len(M_arr),N_modes) )
g4_arr_even = np.zeros( (len(M_arr),N_modes) )
#g4_arr_v1 = np.zeros(len(M_arr))
participation_odd = np.zeros(g4_arr_odd.shape)
participation_even = np.zeros(g4_arr_even.shape)


for i, M in enumerate(M_arr):
    print('%d/%d' %(i,len(M_arr)))
 
    # figuring out the required microstrup length
    d_ms = 2*v_ms/omega_max*(np.arctan( Z_ms/Z_j*np.sqrt(1-(omega_max/omega_p)**2)/np.tan(M*omega_max/2/omega_0/np.sqrt(1-(omega_max/omega_p)**2) ) ))
    d_ms_list[i] = d_ms

    """ Calculation of Even array mode frequencies """
    # two lists of points where each tngent becomes infinite (see notes)
    list_1 = [2*np.pi*v_ms/d_ms*(1/2+n) for n in range( int(N_modes_max/2) )]
    list_2 = [omega_p/np.sqrt( 1 + (M*omega_p/2/np.pi/omega_0/(1/2+n))**2 ) for n in range( int(N_modes_max/2) ) ]
    list_ = [0] + sorted(list_1 + list_2) 
    list_ = np.asarray(list_)[:N_modes+1] # combined sorted list of all divergent points, consider only the first N_modes points
 
    sols = []
    for j in range(N_modes):
        xmin = list_[j]+eps
        xmax = list_[j+1]-eps
        # define the bounds to be slightly away from the divergent points
        if eq1(xmin)*eq1(xmax)<0:
            xopt = brentq(eq1, xmin, xmax)
            sols += [xopt]
        else:
            # separately treat the case when the divergencies have the same sign
            xnew, fval, ierr, numfunc = fminbound(lambda x: eq1(x,+1), xmin, xmax, full_output = True)
            if eq1(xmin,+1)*fval<0:
#                print('yes') # in this case there are two solutions! 
                xopt1 = brentq(eq1, xmin, xnew)
                sols += [xopt1]        
                xopt2 = brentq(eq1, xnew, xmax)
                sols += [xopt2]
    # define the array of even mode frequencies
    while len(sols) < N_modes: sols += [None]
    modes_arr_even[i] = sols[:N_modes]
    
    
    """ Calculation of Odd array mode frequencies """
    list_1 = [2*np.pi*v_ms/d_ms*(1/2+n) for n in range( int(N_modes_max/2) )]
    list_2 = [omega_p/np.sqrt( 1 + (M*omega_p/2/np.pi/omega_0/n)**2 ) for n in range(1, int(N_modes_max/2) ) ]
    list_ = [0] + sorted(list_1 + list_2)
    list_ = np.asarray(list_)[:N_modes+1]
 
    sols = []
    for j in range(N_modes):
        xmin = list_[j]+eps
        xmax = list_[j+1]-eps
        if eq2(xmin)*eq2(xmax)<0:
            xopt = brentq(eq2, xmin, xmax)
            sols += [xopt]
        else: # there is this trick with the minus sign, because the divergencies have the opposite sign now, so I need to maximize the function instead of minimizing it
            xnew, fval, ierr, numfunc = fminbound(lambda x: eq2(x,-1), xmin, xmax, full_output = True)
            if eq2(xmin,-1)*fval<0:
#                print('yes')
                xopt1 = brentq(eq2, xmin, xnew)
                sols += [xopt1]        
                xopt2 = brentq(eq2, xnew, xmax)
                sols += [xopt2]            

    while len(sols)<N_modes: sols += [None]
    modes_arr_odd[i] = sols[:N_modes]


    """ Calculation of g4 for the odd modes """
    for j in range(N_modes):    
        omega = modes_arr_even[i][j]
        # define a couple new variables to simplify the further matrix expressions
        K_arr = np.sqrt(omega**2/(omega_p**2-omega**2))*omega_p/omega_0
        k_ms = omega/v_ms
        d_0 = M*a
        d_1 = d_ms + d_0
        
        # see the notes for the meaning of these matrices
        Big1j = np.asarray( [[0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,K_arr/2/L_s*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,K_arr/2/L_s*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0]])

        Big1m = np.asarray( [[omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4,0,0,0,0],
                            [-omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4],
                            [0,0,0,0, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_1)-np.sin(k_ms*d_0)))/4]])
    
        Big1 = np.asarray( [[omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4,0,0,0,0],
                            [-omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4,0,0,0,0],
                            [0,0,K_arr/2/L_s*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,K_arr/2/L_s*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4],
                            [0,0,0,0, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_1)-np.sin(k_ms*d_0)))/4]])
    
    
        Big2 = np.asarray( [[1/8/omega/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))), 1/8/omega/Z_ms*(np.cos(k_ms*d_1)-np.cos(k_ms*d_0)),0,0,0,0],
                            [1/8/omega/Z_ms*(np.cos(k_ms*d_1)-np.cos(k_ms*d_0)), 1/8/omega/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))),0,0,0,0],
                            [0,0,C_0/2*(np.sin(K_arr*M)+K_arr*M)/2/K_arr+C_j*K_arr/2*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,C_0/2*(-np.sin(K_arr*M)+K_arr*M)/K_arr/2+C_j*K_arr/2*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0,1/8/omega/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))),1/8/omega/Z_ms*(np.cos(k_ms*d_0)-np.cos(k_ms*d_1))],
                            [0,0,0,0,1/8/omega/Z_ms*(np.cos(k_ms*d_0)-np.cos(k_ms*d_1)),1/8/omega/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))]])
        Tran = np.asarray([ [1,0,0,0,0], 
                           [0,1,0,0,0], 
                           [0,0,1,0,0], 
                           [-np.cos(k_ms*d_0/2)/np.sin(K_arr*M/2),np.sin(k_ms*d_0/2)/np.sin(K_arr*M/2),np.cos(K_arr*M/2)/np.sin(K_arr*M/2),0,0],
                          [0,0,0,1,0], 
                          [0,0,0,0,1] ])
        Inv = np.linalg.inv(np.asarray( [[np.sin(k_ms*d_1/2),np.cos(k_ms*d_1/2),0,0,0], 
                                         [0,0,0,-np.sin(k_ms*d_1/2), np.cos(k_ms*d_1/2)], 
                                         [np.cos(k_ms*d_0/2),-np.sin(k_ms*d_0/2),-np.cos(K_arr*M/2),0,0], 
                                         [0,0,-np.cos(K_arr*M/2), np.cos(k_ms*d_0/2), np.sin(k_ms*d_0/2)], 
                                         [-omega/Z_ms*np.sin(k_ms*d_0/2),-omega/Z_ms*np.cos(k_ms*d_0/2),-K_arr/L_s*(omega**2/omega_p**2-1)*np.sin(K_arr*M/2),0,0]],dtype='float'))
        vect = np.asarray(np.transpose([0,0,np.sin(K_arr*M/2),-np.sin(K_arr*M/2),-K_arr/L_s*(omega**2/omega_p**2-1)*np.cos(K_arr*M/2)]))
        z = -np.dot(Inv,vect)[2]
        L_a = 1/(2*np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1,np.dot(Tran,np.dot(Inv,vect)))))))
        C_a = 2*np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big2,np.dot(Tran,np.dot(Inv,vect))))))
    
        g_4 = (hbar/(2*np.pi))*(c_4/c_2)/24/L_s/Phi_0**2*((K_arr)**4)*(6*(1+z**2)**2*(K_arr*M)+8*(1-z**4)*np.sin(K_arr*M)+(1-6*z**2+z**4)*np.sin(2*K_arr*M))/(16*K_arr)*L_a/4/C_a
        g4_arr[i] = g_4
        g4_arr_odd[i][j] = g_4
        
 
        EnergyJunctions = np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1j,np.dot(Tran,np.dot(Inv,vect))))))
        EnergyMicrostrip = np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1m,np.dot(Tran,np.dot(Inv,vect))))))
        participation_odd[i][j] = EnergyJunctions/(EnergyJunctions + EnergyMicrostrip)
        
    g4_arr_odd[i] = g4_arr_odd[i][:N_modes]
    participation_odd[i] = participation_odd[i][:N_modes]

    """ Calculation of g4 for the even modes """ 
    for j in range(N_modes):    
        omega = modes_arr_odd[i][j]
        # define a couple new variables to simplify the further matrix expressions
        K_arr = np.sqrt(omega**2/(omega_p**2-omega**2))*omega_p/omega_0
        k_ms = omega/v_ms
        d_0 = M*a
        d_1 = d_ms + d_0
        
        # see the notes for the meaning of these matrices
        Big1j = np.asarray( [[0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,K_arr/2/L_s*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,K_arr/2/L_s*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0]])
        
        Big1m = np.asarray( [[omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4,0,0,0,0],
                            [-omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4],
                            [0,0,0,0, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_1)-np.sin(k_ms*d_0)))/4]])

        Big1 = np.asarray( [[omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4,0,0,0,0],
                            [-omega/2/Z_ms*((np.cos(k_ms*d_1)-np.cos(k_ms*d_0)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4,0,0,0,0],
                            [0,0,K_arr/2/L_s*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,K_arr/2/L_s*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))/4, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4],
                            [0,0,0,0, -omega/2/Z_ms*((np.cos(k_ms*d_0)-np.cos(k_ms*d_1)))/4, omega/2/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_1)-np.sin(k_ms*d_0)))/4]])

        Big2 = np.asarray( [[1/8/omega/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))), 1/8/omega/Z_ms*(np.cos(k_ms*d_1)-np.cos(k_ms*d_0)),0,0,0,0],
                            [1/8/omega/Z_ms*(np.cos(k_ms*d_1)-np.cos(k_ms*d_0)), 1/8/omega/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))),0,0,0,0],
                            [0,0,C_0/2*(np.sin(K_arr*M)+K_arr*M)/2/K_arr+C_j*K_arr/2*(-np.sin(K_arr*M)+K_arr*M)/2,0,0,0],
                            [0,0,0,C_0/2*(-np.sin(K_arr*M)+K_arr*M)/K_arr/2+C_j*K_arr/2*(np.sin(K_arr*M)+K_arr*M)/2,0,0],
                            [0,0,0,0,1/8/omega/Z_ms*(k_ms*(d_1-d_0)-(np.sin(k_ms*d_0)-np.sin(k_ms*d_1))),1/8/omega/Z_ms*(np.cos(k_ms*d_0)-np.cos(k_ms*d_1))],
                            [0,0,0,0,1/8/omega/Z_ms*(np.cos(k_ms*d_0)-np.cos(k_ms*d_1)),1/8/omega/Z_ms*(k_ms*(d_1-d_0)+(np.sin(k_ms*d_0)-np.sin(k_ms*d_1)))]])
        Tran = np.asarray([ [1,0,0,0,0], 
                           [0,1,0,0,0], 
                           [np.cos(k_ms*d_0/2)/np.cos(K_arr*M/2),-np.sin(k_ms*d_0/2)/np.cos(K_arr*M/2),np.sin(k_ms*d_0/2)/np.cos(K_arr*M/2),0,0], 
                           [0,0,1,0,0],
                          [0,0,0,1,0], 
                          [0,0,0,0,1] ])
        Inv = np.linalg.inv(np.asarray( [[np.sin(k_ms*d_1/2),np.cos(k_ms*d_1/2),0,0,0], 
                                         [0,0,0,-np.sin(k_ms*d_1/2), np.cos(k_ms*d_1/2)], 
                                         [np.cos(k_ms*d_0/2),-np.sin(k_ms*d_0/2),np.sin(K_arr*M/2),0,0], 
                                         [0,0,-np.sin(K_arr*M/2), np.cos(k_ms*d_0/2), np.sin(k_ms*d_0/2)], 
                                         [-omega/Z_ms*np.sin(k_ms*d_0/2),-omega/Z_ms*np.cos(k_ms*d_0/2),-K_arr/L_s*(omega**2/omega_p**2-1)*np.cos(K_arr*M/2),0,0]],dtype='float'))
        vect = np.asarray(np.transpose([0,0,-np.cos(K_arr*M/2),-np.cos(K_arr*M/2),-K_arr/L_s*(omega**2/omega_p**2-1)*np.sin(K_arr*M/2)]))
        z = -np.dot(Inv,vect)[2]
        L_a = 1/(2*np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1,np.dot(Tran,np.dot(Inv,vect)))))))
        C_a = 2*np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big2,np.dot(Tran,np.dot(Inv,vect))))))
    
        g_4 = (hbar/(2*np.pi))*(c_4/c_2)/24/L_s/Phi_0**2*((K_arr)**4)*(6*(1+z**2)**2*(K_arr*M)+8*(-1+z**4)*np.sin(K_arr*M)+(1-6*z**2+z**4)*np.sin(2*K_arr*M))/(16*K_arr)*L_a/4/C_a
        g4_arr[i] = g_4
        g4_arr_even[i][j] = g_4

        EnergyJunctions = np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1j,np.dot(Tran,np.dot(Inv,vect))))))
        EnergyMicrostrip = np.dot(np.transpose(vect),np.dot(np.transpose(Inv),np.dot(np.transpose(Tran),np.dot(Big1m,np.dot(Tran,np.dot(Inv,vect))))))
        participation_even[i][j] = EnergyJunctions/(EnergyJunctions + EnergyMicrostrip)


    g4_arr_even[i] = g4_arr_even[i][:N_modes]
    participation_even[i] = participation_even[i][:N_modes]
    
##   Looking at the lumped model to compare to this complete calculation
#    omega_ms = np.pi*v_ms/d_ms      
#    g4_arr_v1[i] = 1/12*(c_4/c_2)*omega*Z_ms*(np.sin(np.pi*omega/omega_ms))**2/R_Q/M**2/np.tan(np.pi/2*omega/omega_ms)/(np.pi*omega/omega_ms+np.sin(np.pi*omega/omega_ms))**2  # uses both the correct frequency and resonator length but plugs that int an incorrect formula
#    g4_arr_v2[i] = g4_distributed(0, 1, 0, 0.1, omega_ms/(2*np.pi), M, L_j, Z_ms) # uses the correct length of the resonator, but calcultes incorrect resonance frequency based on the lumped approximation
#    g4_arr_v1[i] = g4_distributed(0, 1, 0, 0.1, omega_ms/(2*np.pi), M, L_j, Z_ms, omega_optional = omega) # yet another option would be to plug into this omega and omega_ms into the formula for g4.
    




""" Plotting the results for array mode frequencies """
""" Paper-style formatting of figures """


#fig_paper, axes = plt.subplots(2, 1, sharex=True, figsize=(3.375, 4.2))  #figsize=
fig_paper, axes = plt.subplots(1, 2, sharex=True, figsize=(7, 2.1))

ax = axes[0]

palette = plt.get_cmap('tab10')

ax.axhspan(4, 12, alpha=0.1, color='green')


for mode_num in range(N_modes):
    ax.plot(M_arr, np.transpose(modes_arr_even)[mode_num]/(2*np.pi)*1e-9,color=palette(np.mod(2*mode_num,10)))
    ax.plot(M_arr, np.transpose(modes_arr_odd)[mode_num]/(2*np.pi)*1e-9,color=palette(np.mod(2*mode_num+1,10)))

ax.set_ylim(0, omega_p/(2*np.pi)*1e-9 + 4)

# Max number of SNAILs for the pure JTLA case
M_max = 1200
N_modes2 = 2*N_modes+1

M_arr_2 = [M_crit + k for k in range(M_max-M_crit)]
modes_arr_2 = np.zeros((N_modes2,len(M_arr_2)))
g4_arr_2 = np.zeros((N_modes2,len(M_arr_2)))

# calculating array mode frequencies and nonlinearities:
for mode_num in range(1,N_modes2):
    omega = omega_arr_res(M_arr_2,mode_num)
    modes_arr_2[mode_num-1] = omega
    g4_arr_2[mode_num-1] = (hbar/(2*np.pi))*(c_4/c_2)*3/8/24*L_s/M_arr_2/Phi_0**2*omega**2  #in Hz

for mode_num in range(N_modes2):
    ax.plot(M_arr_2, modes_arr_2[mode_num]/(2*np.pi)*1e-9, color=palette(np.mod(mode_num,10)))
ax.plot([1,M_max], np.ones(2)*omega_p/(2*np.pi)*1e-9,linestyle='--', color='black')


#ax.plot([20,20], (4,12),linestyle='--', color='black')
#ax.plot([200,200], (4,12),linestyle='--', color='black')
#ax.plot([1000,1000], (4,12),linestyle='--', color='black')
#ax.annotate('SPA08-\nSPA34', xy=(0, 0), xytext=(30, 4))
#ax.annotate('JTLA01', xy=(0, 0), xytext=(M_crit, 4))
#ax.annotate('JTLA02', xy=(0, 0), xytext=(M_max, 4))

ax.set_yticks(np.linspace(0,45,10))
#ax.set_xticks(np.linspace(0,M_max,int(M_max/100)+1))
#ax.set_xlabel(r'Number of SNAILs, $\rm M$')
ax.set_ylabel(r'Frequency $\rm (GHz)$')

#ax.axvspan(1, 105, alpha=0.1, color='red')
#ax.annotate('Lumped', xy=(0, 0), xytext=(5, 46.5),fontsize=6)

#ax.axvspan(20, 52, alpha=0.1, color='red')
#ax.axvspan(220, M_max, alpha=0.1, color='green')


#ax.axvspan(1, 15, alpha=0.1, color='yellow')
#ax.axvspan(15, 52, alpha=0.1, color='blue')
#ax.axvspan(52, 220, alpha=0.1, color='red')
#ax.axvspan(220, M_max, alpha=0.1, color='green')

ax.plot([15,15], [1,omega_p], linestyle=(0, (5, 10)), color='black')
ax.plot([70,70], [1,omega_p], linestyle=(0, (5, 10)), color='black')
ax.plot([220,220], [1,omega_p], linestyle=(0, (5, 10)), color='black')

ax.annotate(r'$\rm (I)$', xy=(3.5, 47), fontsize=6)
ax.annotate(r'$\rm (II)$', xy=(27, 47), fontsize=6)
ax.annotate(r'$\rm (III)$', xy=(100, 47), fontsize=6)
ax.annotate(r'$\rm (IV)$', xy=(400, 47), fontsize=6)

#ax.annotate('Lumped', xy=(0, 0), xytext=(25, 18),fontsize=6,rotation='vertical')
#ax.annotate('Josephson Transmission Line', xy=(0, 0), xytext=(280, 46.5),fontsize=6)
ax.plot([800,800,800],[38,40,42],color='black',linestyle='none', marker='.')



ax.plot([20],[7.2],color = palette(0),linestyle='none', marker='X',markersize=5)
ax.plot([20],[24.9],color = palette(1),linestyle='none', marker='X',markersize=5)
ax.plot([20],[29.1],color = palette(2),linestyle='none', marker='X',markersize=5)

ax.plot([200],[4.2],color = palette(0),linestyle='none', marker='s',markersize=3)
ax.plot([200],[10.4],color = palette(1),linestyle='none', marker='s',markersize=3)
ax.plot([200],[17.1],color = palette(2),linestyle='none', marker='s',markersize=3)
ax.plot([200],[22.8],color = palette(3),linestyle='none', marker='s',markersize=3)
ax.plot([200],[27.7],color = palette(4),linestyle='none', marker='s',markersize=3)
ax.plot([200],[31.5],color = palette(5),linestyle='none', marker='s',markersize=3)
ax.plot([200],[34.4],color = palette(6),linestyle='none', marker='s',markersize=3)



ax.plot([1000],[4.4],color = palette(2),linestyle='none', marker='^',markersize=4)
ax.plot([1000],[6.1],color = palette(3),linestyle='none', marker='^',markersize=4)
ax.plot([1000],[7.8],color = palette(4),linestyle='none', marker='^',markersize=4)
ax.plot([1000],[9.6],color = palette(5),linestyle='none', marker='^',markersize=4)
#ax.plot([1000],[11.1],color = palette(6),linestyle='none', marker='>',markersize=4)


ax.set_xscale('log')
ax.get_yaxis().set_label_coords(-0.1,0.5)



""" Plotting the results for array mode nonlinearities """

ax = axes[1]

#plt.tick_params(axis='both', labelsize=18)
ax.plot(M_arr, 12*np.abs(g4_arr)*1e-3)
ax.set_ylabel(r'Self-Kerr $\rm (kHz)$')
ax.set_xticks(np.linspace(0,M_max,int(M_max/100)+1))
ax.set_xlabel(r'Number of SNAILs, $\rm M$')



for mode_num in range(N_modes2-1):
    if mode_num %2 == 0:
        M_full_range = np.asarray(list(M_arr) + list(M_arr_2))
        g4_full_range = np.asarray(list(np.transpose(g4_arr_odd)[int(mode_num/2)]) + list(g4_arr_2[mode_num]))
#        p = np.polyfit(np.log(M_full_range[30:130]), np.log(np.abs(g4_full_range)[30:130]), 1)
#        print('gamma of mode %d is %.3f' %(mode_num,p[0]))
        ax.plot(M_full_range, 12*np.abs(g4_full_range)*1e-3,zorder=2,color=palette(np.mod(mode_num,10))) 
    else:
        M_full_range = np.asarray(list(M_arr) + list(M_arr_2))
        g4_full_range = np.asarray(list(np.transpose(g4_arr_even)[int(mode_num/2)]) + list(g4_arr_2[mode_num]))
#        p = np.polyfit(np.log(M_full_range[40:110]), np.log(np.abs(g4_full_range)[40:110]), 1)
#        print('gamma of mode %d is %.3f' %(mode_num,p[0]))
        ax.plot(M_full_range, 12*np.abs(g4_full_range)*1e-3,zorder=2,color=palette(np.mod(mode_num,10))) 


# For each mode n determine the number of SNAILs M at which it crosses the interesting frequency
mode_num_arr = range(1,6)
M_1 = np.zeros(len(mode_num_arr))
g4_final = np.zeros(len(mode_num_arr))
for i, mode_num in enumerate(mode_num_arr):
    M_1[i] = int(mode_num*np.pi*omega_0/omega_p*np.sqrt((omega_p**2-omega_max**2)/omega_max**2))
    g4_final[i] = 12*np.abs(g4_arr_2[mode_num-1][int(-M_crit+M_1[i]+1)])*1e-3
ax.plot(M_1,g4_final,zorder=3, color = palette(0),linestyle='none', marker='.',markersize=4)
#ax.plot(M_1,g4_final,zorder=3, color = palette(0),linestyle='--')

a = M_1[0]*g4_final[0]
ax.plot(np.asarray([100,M_max]),a/np.asarray([100,M_max]),zorder=3, color = palette(0),linestyle='--')


ax.errorbar([20],[25],yerr=10, color = palette(0),linestyle='none', marker='X',markersize=5, capsize=2,elinewidth=0.5)
#ax.errorbar([20],[24.6],yerr=6)
ax.errorbar([200],[27.5],yerr=12.5,color = palette(1),linestyle='none', marker='s',markersize=3, capsize=2,elinewidth=0.5)

#######
#ax.errorbar([1000],[1.0],yerr=0.25,color = palette(2),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)
#ax.errorbar([1000],[2],yerr=1,color = palette(3),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)
#ax.errorbar([1000],[3.75],yerr=1.25,color = palette(4),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)
#ax.errorbar([1000],[5],yerr=2,color = palette(5),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)
#ax.errorbar([1000],[5.5],yerr=2,color = palette(6),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)


ax.errorbar([1000],[1.0],yerr=0,color = palette(2),linestyle='none', marker='^',markersize=4, capsize=2,elinewidth=0.5)
ax.errorbar([1000],[2],yerr=0,color = palette(3),linestyle='none', marker='^',markersize=4, capsize=2,elinewidth=0.5)
ax.errorbar([1000],[3.75],yerr=0,color = palette(4),linestyle='none', marker='^',markersize=4, capsize=2,elinewidth=0.5)
ax.errorbar([1000],[5],yerr=0,color = palette(5),linestyle='none', marker='^',markersize=4, capsize=2,elinewidth=0.5)
#ax.errorbar([1000],[4.5],yerr=0,color = palette(6),linestyle='none', marker='>',markersize=4, capsize=2,elinewidth=0.5)



#ax.axvspan(20, 52, alpha=0.1, color='red')
#ax.axvspan(220, M_max, alpha=0.1, color='green')

#ax.axvspan(1, 15, alpha=0.1, color='yellow')
#ax.axvspan(15, 52, alpha=0.1, color='blue')
#ax.axvspan(52, 220, alpha=0.1, color='red')
#ax.axvspan(220, M_max, alpha=0.1, color='green')

ax.set_yscale('log')
ax.set_ylim(0.8,1e2)
ax.set_xscale('log')
ax.set_xlim(1,M_max+50)


ax.annotate(r'$\propto M$', xy=(1.4, 6), fontsize=7, color = palette(0))
#ax.annotate(r'$\propto 1/M$', xy=(70, 6e-3), fontsize=8)
ax.annotate(r'$\propto 1/M$', xy=(50, 3e1), fontsize=7, color = palette(0))
ax.annotate(r'$\propto 1/M^3$', xy=(100, 2), fontsize=7, color = palette(0))

ax.get_yaxis().set_label_coords(-0.1,0.5)


plt.tight_layout()

#fig_paper.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JAMPA/paper/Figs_aux/' + 'fig1.pdf', dpi=600)



# Plor the participation ratio figure

fig, axes = plt.subplots(1, 2, figsize=(7, 2.1))

axes[0].set_ylabel('EPR of the array')
axes[1].set_ylabel('EPR per unit cell')
axes[1].set_xlabel(r'Number of SNAILs, $\rm M$')
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlim(1,M_max+50)
    ax.get_yaxis().set_label_coords(-0.1,0.5)

participation = np.zeros((N_modes2,len(M_arr_2)))
for mode_num in range(1,N_modes2):
    participation[mode_num-1] = 1.0

M_full_range = np.asarray(list(M_arr) + list(M_arr_2))

for mode_num in range(4): #range(N_modes2-1):
    if mode_num %2 == 0:
        p_full_range = np.asarray(list(np.transpose(participation_odd)[int(mode_num/2)]) + list(participation[mode_num]))
        axes[0].plot(M_full_range, p_full_range, zorder=2,color=palette(np.mod(mode_num,10)),label='Mode %d' %(mode_num+1)) 
        axes[1].plot(M_full_range, p_full_range/M_full_range, zorder=2,color=palette(np.mod(mode_num,10))) 
    else:
        p_full_range = np.asarray(list(np.transpose(participation_even)[int(mode_num/2)]) + list(participation[mode_num]))
        axes[0].plot(M_full_range, p_full_range,zorder=2,color=palette(np.mod(mode_num,10)),label='Mode %d' %(mode_num+1))
        axes[1].plot(M_full_range, p_full_range/M_full_range,zorder=2,color=palette(np.mod(mode_num,10))) 


p_j = np.mean(participation_odd[:20,0]/np.arange(1,21,1))
axes[1].plot(np.asarray([1,200]),[p_j,p_j],zorder=3, color = palette(0),linestyle='--')
axes[1].annotate(r'$p_J=%.3f$'%p_j, xy=(40, 0.035), fontsize=7, color = palette(0))
axes[1].annotate(r'$\propto 1/M$', xy=(300, 0.007), fontsize=7, color = palette(0))

axes[0].legend(loc='best')
plt.tight_layout()
#fig.savefig(r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JAMPA/paper/Figs_aux/' + 'fig_participation.pdf', dpi=600)