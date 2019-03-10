#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:43:12 2018

@author: Vladimir Sivak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib as mpl
from devices_functions import *
import h5py
from scipy import interpolate
from scipy.optimize import fsolve


### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
### -------------------------------- Including period doubling -----------------------------------
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------



#def Gain(n_p,n_s,n_i,g3_,g4_,kappa,detuning):
#    g = 2*g3_*np.sqrt(n_p)
#    Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i )
# 
#    Ah2 = -1/(12*g4_)*( -detuning + g4_*(32/3*n_p+24*n_s+24*n_i ) +np.sqrt(4*g**2-kappa**2/4) )
#    
#    Ah2 = np.asarray([x if np.isreal(x) and x>0 else 0 for x in Ah2])
#    
#    
#    Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i + 24*Ah2)
#    g = 2*g3_*np.sqrt(n_p)*np.sqrt(( (-detuning + g4_*(32/3*n_p+24*n_s+24*n_i ))**2 +kappa**2/4 )/( (-detuning + g4_*(32/3*n_p+24*n_s+24*n_i+12*Ah2 ))**2 +kappa**2/4 ))
#    
#    return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2
    
#def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) +16*Ah2) )*Xp-2/3*kappa_t*Yp-u_p - 6*g3_*(Xi*Xs-Ys*Yi) -12*g3_**2*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*Xp +Yp*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )        )
#
#def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_p-freq_c-g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) +16*Ah2)  )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs)   -12*g3_**2*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*Yp  - Xp*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )            )
#
#def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) +24*Ah2) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp)   + 48*g3_*g4_*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*(Xp*Xi+Yp*Yi) + (Yp*Xi-Yi*Xp)*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )        )
#
#def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_s-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) +24*Ah2)  )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi)    + 48*g3_*g4_*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*(Yp*Xi-Xp*Yi) - (Xp*Xi+Yi*Yp)*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )        )
#
#def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) +24*Ah2)  )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp)  + 48*g3_*g4_*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*(Xp*Xs+Yp*Ys) + (Yp*Xs-Ys*Xp)*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )       )
#
#def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    Ah2 = -1/(12*g4_)*( freq_c-freq_p/2 + g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) ) +np.sqrt(16*g3_**2*(Xp**2+Yp**2)-kappa**2/4) )
#    Ah2 = Ah2 if np.isreal(Ah2) and Ah2>0 else 0
#    return float((freq_i-freq_c-g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) +24*Ah2)  )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs)  + 48*g3_*g4_*Ah2*( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )*(Yp*Xs-Xp*Ys) - (Xp*Xs+Ys*Yp)*kappa_t/2  )/ ( (freq_p/2-freq_c-g4_*(32/3*(Xp**2+Yp**2)+24*(Xs**2+Ys**2)+24*(Xi**2+Yi**2) +12*Ah2) )**2 + kappa_t**2/4  )                 )
#    
#def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
#    Xp, Yp, Xs, Ys, Xi, Yi = p
#    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )




### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
### -------------------------------- Using empirically measured Stark shift ----------------------
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------



#def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_a )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))
#
#def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c  )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))
#
#def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    stark_shift = 1/2*interpolate.splev(Xp**2+Yp**2+Xs**2+Ys**2+Xi**2+Yi**2, tck, der=0)
#    return float((freq_s-freq_c-stark_shift )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))
#
#def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    stark_shift = 1/2*interpolate.splev(Xp**2+Yp**2+Xs**2+Ys**2+Xi**2+Yi**2, tck, der=0)
#    return float((freq_s-freq_c-stark_shift  )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))
#
#def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    stark_shift = 1/2*interpolate.splev(Xp**2+Yp**2+Xs**2+Ys**2+Xi**2+Yi**2, tck, der=0)
#    return float((freq_i-freq_c-stark_shift  )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))
#
#def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    stark_shift = 1/2*interpolate.splev(Xp**2+Yp**2+Xs**2+Ys**2+Xi**2+Yi**2, tck, der=0)
#    return float((freq_i-freq_c-stark_shift  )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
#
#def EquationS(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
#    Xp, Yp, Xs, Ys, Xi, Yi = p
#    return (Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )
#
#def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
#    g = 2*g3_*np.sqrt(n_p)
#    stark_shift = 1/2*interpolate.splev(n_p+n_s+n_i, tck, der=0)
#    return 1 + 4*kappa**2*g**2/( (-detuning+stark_shift)**2+kappa**2/4-4*g**2 )**2


### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
### -------------------------------- First order g4, g3 harmonic balance -------------------------
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

def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
    g = 2*g3_*np.sqrt(n_p)   
    Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i )    
    return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2



### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
### -------------------------------- Second order g4, g3 harmonic balance (IMD) ------------------
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

#def Equation1(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )           -g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Xp-2/3*kappa_t*Yp-u_p -6*g3_*(Xi*Xs-Ys*Yi))
#
#def Equation2(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_p-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )               -g4_*(32/9*(Xp**2+Yp**2)+16*(Xs**2+Ys**2)+16*(Xi**2+Yi**2) ) + g3_**2/freq_c*(928/45*(Xp**2+Yp**2)+42*(Xs**2+Ys**2)+42*(Xi**2+Yi**2) ) )*Yp+2/3*kappa_t*Xp-6*g3_*(Xi*Ys+Yi*Xs))
#
#def Equation3(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )               -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Xs-1/2*kappa_t*Ys-u_s-4*g3_*(Xp*Xi+Yi*Yp))
#
#def Equation4(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_s-freq_c  -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )                  -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+15*(Xs**2+Ys**2)+36*(Xi**2+Yi**2) ) )*Ys+1/2*kappa_t*Xs-4*g3_*(-Yi*Xp+Yp*Xi))
#
#def Equation5(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )                    -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Xi-1/2*kappa_t*Yi-4*g3_*(Xp*Xs+Ys*Yp))
#
#def Equation6(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi):
#    return float((freq_i-freq_c  -g4_*12*(g3_/freq_c)**2*(36*(Xi**2+Yi**2)*(Xs**2+Ys**2)+4*(Xp**2+Yp**2)*(Xi**2+Yi**2+Xs**2+Ys**2)+9*( (Xi**2+Yi**2)**2 + (Xs**2+Ys**2) )+ 16/81*(Xp**2+Yp**2)**2+ (8/3*(Xp**2+Yp**2)+6*(Xi**2+Yi**2)+6*(Xs**2+Ys**2)   )**2  )          -g4_*(32/3*(Xp**2+Yp**2)+12*(Xs**2+Ys**2)+12*(Xi**2+Yi**2) ) + 4*g3_**2/freq_c*(7*(Xp**2+Yp**2)+36*(Xs**2+Ys**2)+15*(Xi**2+Yi**2) ) )*Yi+1/2*kappa_t*Xi-4*g3_*(-Ys*Xp+Yp*Xs))
#    
#def EquationS_IMD(freq_s, freq_i, freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, p):
#    Xp, Yp, Xs, Ys, Xi, Yi = p
#    return (Equation1_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation2_IMD(freq_p, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation3_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation4_IMD(freq_s, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), 
#            Equation5_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi), Equation6_IMD(freq_i, freq_c, g4_, kappa_t, u_p, u_s, g3_, Xp, Yp, Xs, Ys, Xi, Yi) )
#
#def Gain(n_p,n_s,n_i,g3_,g4_,kappa, detuning):
#    g = 2*g3_*np.sqrt(n_p)   
#    Delta_eff = -detuning + g4_*(32/3*n_p+12*n_s+12*n_i ) - 4*g3_**2/freq_a*(7*n_p+25.5*n_s+25.5*n_i)
#    return 1 + 4*kappa**2*g**2/( (Delta_eff)**2+kappa**2/4-4*g**2 )**2





device = spa34_v5


plt.close('all')


hdf5_file = h5py.File(device.data_file,'r')
try:    
    skip=1
    Kerr_Data_dict = {}
    Kerr_Data_dict['current'] = np.asarray(hdf5_file['fit_stark_shift'].get('current'))[::skip]
    Kerr_Data_dict['date'] = np.asarray(hdf5_file['fit_stark_shift'].get('date'))[::skip]
    Kerr_Data_dict['time'] = np.asarray(hdf5_file['fit_stark_shift'].get('time'))[::skip]
    Kerr_Data_dict['freq_drive'] = np.asarray(hdf5_file['fit_stark_shift'].get('drive_freq'))[::skip]

    for j, current in enumerate(Kerr_Data_dict['current']):
        
        date = float_to_DateTime( Kerr_Data_dict['date'][j] )
        time = float_to_DateTime( Kerr_Data_dict['time'][j] )
        f_d = Kerr_Data_dict['freq_drive'][j]
        meas_name = hdf5_file['results'].attrs.get('flux_sweep_meas_name')
        f_0 = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('f0'))
        k_c = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('kc'))
        k_i = np.asarray(hdf5_file[date][time]['fits'][meas_name].get('ki'))
        k_t = k_c + k_i             
        line_attenuation = attenuation(f_d, device.data_file, 'drive_attenuation') - 3
        if 'powers' in list(hdf5_file[date][time]['LIN'].keys()):
            log_powers = np.asarray(hdf5_file[date][time]['LIN'].get('powers'))
        elif 'powers_swept' in list(hdf5_file[date][time]['LIN'].keys()):
            log_powers = np.asarray(hdf5_file[date][time]['LIN'].get('powers_swept'))
        log_powers = log_powers + line_attenuation
        powers = dBm_to_Watts(log_powers)
        stark_shift = f_0 - f_0[0]
        nbar = 1/(2*np.pi*h)*powers*k_c[0]/f_0[0]*((f_d/f_0[0])**2)/( (f_d-f_0[0])**2 + (k_t[0]/2)**2*((2*f_d/(f_0[0]+f_d))**2) )   



        tck = interpolate.splrep(nbar, stark_shift, s=0)

        f = Flux(current,device.a,device.b)        
        fudge_factor = 2.3        
        g3_ = device.g3_distributed(f)*fudge_factor   
        g4_ = device.g4_distributed(f)  
        freq_a = f_0[0]
        kappa = k_c[0]
        kappa_t = k_t[0]
        kappa_c = k_c[0]

        n_crit = max(nbar)
        n_crit = 20000
        nbar_pump_sweep = np.linspace(0,n_crit,5000)
        pump_freqs = np.linspace(2*freq_a-1.2e9,2*freq_a+1.2e9,400)
        gains = np.zeros((len(pump_freqs),5000))

        for i, freq_p in enumerate(pump_freqs):
            Gain_ = Gain(nbar_pump_sweep,1/2,1/2,g3_, g4_, kappa,freq_p/2-freq_a)
            Gain_ = 10*np.log10(Gain_)
            gains[i] = Gain_
    
        fig, ax = plt.subplots(1,dpi=150)
        ax.set_title('Flux %.3f' %f)
        ax.set_ylabel(r'$nbar\;pump,\;(photons)}$')
        ax.set_xlabel(r'${\rm Pump\, detuning\,(MHz)}$')
        p = ax.pcolormesh((pump_freqs/2-freq_a)*1e-6, nbar_pump_sweep, np.transpose(gains), cmap='coolwarm',vmin=-40,vmax=40)
        fig.colorbar(p, ax=ax, label=r'${\rm Gain }$')


        n_p = []
        pump_detuning = []
        for i, f_p in enumerate(pump_freqs):
            count = 0
            while gains[i][count]<20 and count<len(gains[i])-1:
                count +=1
            if count<len(gains[i])-1:
                n_p += [nbar_pump_sweep[count]]
                pump_detuning += [f_p/2 - freq_a]
        pump_detuning = np.asarray(pump_detuning)
        n_p = np.asarray(n_p)

#        nbar_signal = np.linspace(0,500,501)
#        nbar_idler = nbar_signal

        Signal_Powers_dBm = np.linspace(-140,-100,51)
        Signal_Powers = dBm_to_Watts(Signal_Powers_dBm)
       


        fig2, ax2 = plt.subplots(1,dpi=150)
        ax2.set_title('Current %.3f' %(current*1e3))
        ax2.set_ylabel(r'${\rm Signal\, power \,(dBm)}$')
        ax2.set_xlabel(r'${\rm Pump\, detuning,}\; \frac{\omega_p}{2}-\omega_a,\rm (MHz)}$')  
        vmax = 35  #max([max(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #0
        vmin = 0 #min([min(gain_array[i])  for i in range(np.shape(gain_array)[0])]) #25   
        gain_array=np.zeros((len(pump_detuning),len(Signal_Powers_dBm)))
        for i, detuning in enumerate(pump_detuning):


            alpha_p = np.sqrt(n_p[i])            
            u_p = np.sqrt((freq_p-freq_a)**2 +(2/3*kappa_t)**2)*alpha_p

            freq_p = (freq_a + detuning)*2
            freq_s = freq_p/2 + 100e3
            freq_i = freq_p-freq_s
            omega = freq_s - freq_p/2
            
            
            Delta = -detuning
            g4_ = device.g4_distributed(f)

            
            for b, P_s in enumerate(Signal_Powers):
    

                u_s = np.sqrt(P_s/(2*np.pi*h)*kappa/freq_a )
                
                _Xp_guess = float(alpha_p)
                _Yp_guess = 0
                _Xs_guess = 0
                _Ys_guess = 0
                _Xi_guess = 0
                _Yi_guess = 0
#                for _ in range(2):
#                    stark_shift  = 1/2*interpolate.splev(_Xp_guess**2+_Yp_guess**2+_Xs_guess**2+_Ys_guess**2+_Xi_guess**2+_Yi_guess**2, tck, der=0)         
#                    Delta_eff = Delta + stark_shift
#                    _Xs_guess = float( -u_s*( ( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )*(Delta_eff+omega) )/( ( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
#                    _Ys_guess = float( -kappa_t/2*( (Delta_eff+omega)**2+kappa_t**2/4-16*alpha_p**2*g3_**2  )*u_s/( ( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
#                    _Xi_guess = float( 4*g3_*alpha_p*u_s*( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )/( ( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
#                    _Yi_guess = float(-4*g3_*alpha_p*kappa_t*omega*u_s/( ( Delta_eff**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ))
    
                Xp, Yp, Xs, Ys, Xi, Yi = fsolve( lambda p: EquationS(freq_s, freq_i, freq_p, freq_a, g4_, kappa, u_p, u_s, g3_, p) , ( _Xp_guess, _Yp_guess, _Xs_guess, _Ys_guess, _Xi_guess, _Yi_guess ), factor = 0.1, diag=np.ones(6)*0.1, epsfcn=10 )           

                
                nbar_pump = Xp**2 + Yp**2
                nbar_signal = Xs**2 + Ys**2
                nbar_idler = Xi**2 + Yi**2
                Gain_ = Gain(nbar_pump,nbar_signal,nbar_idler, g3_, g4_, kappa, detuning)
                Gain_ = 10*np.log10(Gain_)
                gain_array[i][b] = Gain_
                
            
            from scipy import integrate    
            plt.style.use('ggplot')
            np.seterr(divide='ignore', invalid='ignore')
            
            DELTA = float((-detuning + g4_*(32/3*alpha_p**2))*1e-6)
            _G_ = float(np.abs(2*g3_*alpha_p)*1e-6)
            KAPPA = float(kappa*1e-6)
            _K_ = float(12*g4_*1e-6)
            
            def ODE(A,t=0):
                X = A[0]
                Y = A[1]
                return ((DELTA+_K_*(X**2+Y**2))*Y-2*_G_*X-KAPPA*X/2,-(DELTA+_K_*(X**2+Y**2))*X+2*_G_*Y-KAPPA/2*Y)
                
            values = np.linspace(-200.0, 200.0, 100)
            vcolors = plt.cm.autumn_r(np.linspace(0.3, 1, len(values)))
            
            fig_nn = plt.figure()
            t = np.linspace(0, 1, 10000)
            Init1 = (1,0)
            
            for v, col in zip(values, vcolors):
                P0 = [I*v for I in Init1]
                # Integrate system of ODEs to get x and y values
                P = integrate.odeint(ODE, P0, t)
                # Plot each trajectory
                plt.plot( P[:,0], P[:,1], color=col) #, label='P0=(%.f, %.f)' % ( P0[0], P0[1]) )
            savepath = r'/Users/vs362/Downloads/%d/%d.png' %(j,i)
            fig_nn.savefig(savepath, dpi=240)
            plt.close(fig_nn)    
            
            
        if len(pump_detuning)>1:
            p = ax2.pcolormesh(pump_detuning*1e-6, Signal_Powers_dBm, np.transpose(gain_array), cmap='gist_rainbow',vmin=vmin,vmax=vmax)
            fig2.colorbar(p, ax=ax2, label=r'${\rm Gain }$')
            levels = [19,21]
            CS = ax2.contour(pump_detuning*1e-6, Signal_Powers_dBm, np.transpose(gain_array),levels, colors='black')
            ax2.clabel(CS, inline=True, fontsize=4) 

finally:
    hdf5_file.close()