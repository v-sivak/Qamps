import devices_functions 
import numpy as np




def Equation1(Delta_a_eff, Delta_b_eff):
    return float(-(Delta_b_eff - Delta_b) + g4_ab*( (Delta_a_eff+omega)**2+kappa_a**2/4+4*g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_p**2 )/((Delta_a_eff**2-omega**2+kappa_a**2/4-4*g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_p**2 )**2 + (kappa_a*omega)**2 )*u_s**2+12*g4_b/( Delta_b_eff**2+kappa_b**2/4 )*u_p**2 )

def Equation2(Delta_a_eff, Delta_b_eff):
    return float(-(Delta_a_eff - Delta_a) + 12*g4_a*( (Delta_a_eff+omega)**2+kappa_a**2/4+4*g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_p**2 )/((Delta_a_eff**2-omega**2+kappa_a**2/4-4*g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_p**2 )**2+(kappa_a*omega)**2)*u_s**2 + g4_ab/( Delta_b_eff**2+kappa_b**2/4 )*u_p**2) 

def EquationS(p):
    Delta_a_eff, Delta_b_eff = p
    return ( Equation1(Delta_a_eff, Delta_b_eff), Equation2(Delta_a_eff, Delta_b_eff) )


def Gain(Delta_a_guess, Delta_b_guess):
    Delta_a_eff, Delta_b_eff = fsolve( EquationS , (Delta_a_guess, Delta_b_guess))
    gg = g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_p**2  
    return  1 + 4*kappa_a**2*gg/( (Delta_a_eff**2-omega**2+kappa_a**2/4-4*gg)**2 + omega**2*kappa_a**2 )
    
def U_Eq1(Delta_a_eff, Delta_b_eff,u_pp):
    return float(-(Delta_b_eff - Delta_b) + 12*g4_b/( Delta_b_eff**2+kappa_b**2/4 )*u_pp**2 )

def U_Eq2(Delta_a_eff, Delta_b_eff,u_pp):
    return float(-(Delta_a_eff - Delta_a) + g4_ab/( Delta_b_eff**2+kappa_b**2/4 )*u_pp**2)

def U_Eq3(Delta_a_eff, Delta_b_eff,u_pp):
    gg = g3_ab**2/(Delta_b_eff**2+kappa_b**2/4)*u_pp**2 
    return float(1 + 4*kappa_a**2*gg/(Delta_a_eff**2+kappa_a**2/4-4*gg)**2 - G)

def U_EqS(p):
    Delta_a_eff, Delta_b_eff, u_pp = p
    return ( U_Eq1(Delta_a_eff, Delta_b_eff,u_pp), U_Eq2(Delta_a_eff, Delta_b_eff,u_pp), U_Eq3(Delta_a_eff, Delta_b_eff,u_pp))

def equation_for_up(Delta_a_guess, Delta_b_guess,u_p_guess):
    
    Delta_a_eff, Delta_b_eff, u_p = fsolve( U_EqS , (Delta_a_guess, Delta_b_guess,u_p_guess) )
    
    return u_p





def _Equation1(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (omega-Delta_a-g4_ab*(Xp**2+Yp**2)-12*g4_a*(Xs**2+Ys**2+Xi**2+Yi**2) )*Xs-kappa_a*Ys/2-2*g3_ab*(Xp*Xi+Yp*Yi) - u_s*np.cos(theta_s) )

def _Equation2(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (omega-Delta_a-g4_ab*(Xp**2+Yp**2)-12*g4_a*(Xs**2+Ys**2+Xi**2+Yi**2) )*Ys+kappa_a*Xs/2-2*g3_ab*(-Yi*Xp+Yp*Xi)-u_s*np.sin(theta_s) )

def _Equation3(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (-omega-Delta_a-g4_ab*(Xp**2+Yp**2)-12*g4_a*(Xs**2+Ys**2+Xi**2+Yi**2) )*Xi-kappa_a*Yi/2-2*g3_ab*(Xp*Xs+Yp*Ys) )

def _Equation4(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (-omega-Delta_a-g4_ab*(Xp**2+Yp**2)-12*g4_a*(Xs**2+Ys**2+Xi**2+Yi**2) )*Yi+kappa_a*Xi/2+2*g3_ab*(Xp*Ys-Yp*Xs) )

def _Equation5(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (-Delta_b-12*g4_b*(Xp**2+Yp**2)-g4_ab*(Xs**2+Ys**2+Xi**2+Yi**2) )*Xp-kappa_b*Yp/2-2*g3_ab*(Xi*Ys-Yi*Ys)-u_p*np.cos(theta_p) )

def _Equation6(Xp, Yp, Xs, Ys, Xi, Yi):
    return float( (-Delta_b-12*g4_b*(Xp**2+Yp**2)-g4_ab*(Xs**2+Ys**2+Xi**2+Yi**2) )*Yp+kappa_b*Xp/2-2*g3_ab*(Xi*Ys+Yi*Xs)-u_p*np.sin(theta_p) )

def _EquationS(p):
    Xp, Yp, Xs, Ys, Xi, Yi = p
    return (_Equation1(Xp, Yp, Xs, Ys, Xi, Yi), _Equation2(Xp, Yp, Xs, Ys, Xi, Yi), _Equation3(Xp, Yp, Xs, Ys, Xi, Yi), _Equation4(Xp, Yp, Xs, Ys, Xi, Yi), _Equation5(Xp, Yp, Xs, Ys, Xi, Yi), _Equation6(Xp, Yp, Xs, Ys, Xi, Yi) )

def _Gain(Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess):
  
    Xp, Yp, Xs, Ys, Xi, Yi = fsolve( _EquationS , ( Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess ) )

    nbar_s = Xs**2+Ys**2
    nbar_p = Xp**2+Yp**2
    nbar_i = Xi**2+Yi**2    

    gg = g3_ab**2*(Xp**2+Yp**2)
    Delta_eff = Delta_a + g4_ab*(Xp**2+Yp**2)+12*g4_a*(Xs**2+Ys**2+Xi**2+Yi**2)
    
    return  1 + 4*kappa_a**2*gg/( (Delta_eff**2-omega**2+kappa_a**2/4-4*gg)**2 + omega**2*kappa_a**2 ), nbar_p, nbar_s, nbar_i



freq_max = 8e9
mu=0.5
Z_c=45.8
L_j = 35*1.4e-12
E_j =  1/L_j*(3.3e-16)**2 / h
gamma = 2*Z_c/L_j
alpha = 0.1
C_coupling = 0.088e-12 #0.12e-12
a=1
b=0
#kappa = 450e6



if __name__=='__main__':
    
#   ---------------------- Tunability -------------------------------------

    Tunability = 1
    Coupling = 0
    Nonlinearities = 1
    Saturation = 0
    critical_current = 0
    Nonlinearities_v2 = 0

    array_of_Ms = [5]
    colors = ['red','blue','green','orange','purple']
    flux_th = np.linspace(-2.5,2.5,1000)

    f0_distributed = 15e9




    fig1 = plt.figure(figsize=(9, 7),dpi=120)
    ax1 = plt.subplot(1,1,1)
    ax1.set_title('Flux sweep', fontsize=20)
    ax1.set_xlabel('$\Phi/\Phi_0$', fontsize='x-large')
    ax1.set_ylabel('$f_c(\Phi)$, GHz', fontsize='x-large')

    fig2 = plt.figure(figsize=(9, 7),dpi=120)
    ax2 = plt.subplot(1,1,1)
    ax2.set_title(r'$g_4 \;\rm nonlinearity$', fontsize=20)
    ax2.set_ylabel(r'$4^{\rm th}\;\rm order\;\rm nonlin,\;\rm MHz$', fontsize='x-large')
    ax2.set_yscale('log')

    fig3 = plt.figure(figsize=(9, 7),dpi=120)
    ax3 = plt.subplot(1,1,1)
    ax3.set_title(r'$g_3 \;\rm nonlinearity$', fontsize=20)
    ax3.set_ylabel(r'$3^{\rm rd}\;\rm order\;\rm nonlin,\;\rm MHz$', fontsize='x-large')
    ax3.set_yscale('log')

    labels = ['1','3','5','7','11']
    labels = ['5','11']
    M = 10
    
#    for i, (n_x_default, n_y_default, m_x_default, m_y_default) in enumerate([(1,1,1,3), (3,3,3,9), (5,5,5,15), (7,7,7,21),(11,11,11,33)]):
    for i, (n_x_default, n_y_default, m_x_default, m_y_default) in enumerate([(5,5,5,15), (11,11,11,33)]):

        def H(phi, alpha, f, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            return -alpha*n_y*np.sin(2*np.pi*f*n_x/n_y/2)/np.sin(2*np.pi*f/2/n_y)*np.cos(phi/n_y-(n_x-1)/2/n_y*2*np.pi*f) - m_y*np.sin(2*np.pi*f*m_x/m_y/2)/np.sin(2*np.pi*f/2/m_y)*np.cos(phi/m_y-(m_x-1+2*n_x)/2/m_y*2*np.pi*f)
        
        def c1(phi, alpha, f, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            return alpha*np.sin(2*np.pi*f*n_x/n_y/2)/np.sin(2*np.pi*f/2/n_y)*np.sin(phi/n_y-(n_x-1)/2/n_y*2*np.pi*f) + np.sin(2*np.pi*f*m_x/m_y/2)/np.sin(2*np.pi*f/2/m_y)*np.sin(phi/m_y-(m_x-1+2*n_x)/2/m_y*2*np.pi*f)
        
        def c2(phi, alpha, f, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            return 1/n_y*alpha*np.sin(2*np.pi*f*n_x/n_y/2)/np.sin(2*np.pi*f/2/n_y)*np.cos(phi/n_y-(n_x-1)/2/n_y*2*np.pi*f) + 1/m_y*np.sin(2*np.pi*f*m_x/m_y/2)/np.sin(2*np.pi*f/2/m_y)*np.cos(phi/m_y-(m_x-1+2*n_x)/2/m_y*2*np.pi*f)
        
        def c3(phi, alpha, f, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            return -1/n_y**2*alpha*np.sin(2*np.pi*f*n_x/n_y/2)/np.sin(2*np.pi*f/2/n_y)*np.sin(phi/n_y-(n_x-1)/2/n_y*2*np.pi*f) - 1/m_y**2*np.sin(2*np.pi*f*m_x/m_y/2)/np.sin(2*np.pi*f/2/m_y)*np.sin(phi/m_y-(m_x-1+2*n_x)/2/m_y*2*np.pi*f)
        
        def c4(phi, alpha, f, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            return -1/n_y**3*alpha*np.sin(2*np.pi*f*n_x/n_y/2)/np.sin(2*np.pi*f/2/n_y)*np.cos(phi/n_y-(n_x-1)/2/n_y*2*np.pi*f) - 1/m_y**3*np.sin(2*np.pi*f*m_x/m_y/2)/np.sin(2*np.pi*f/2/m_y)*np.cos(phi/m_y-(m_x-1+2*n_x)/2/m_y*2*np.pi*f)
  
        def offset(f, alpha, n_x=n_x_default, m_x=m_x_default, n_y=n_y_default, m_y=m_y_default):
            f = [f] if isinstance(f, float) or isinstance(f, int) else f
            phi=[]
            for flux in f:
                phi = phi + [fminbound(H,-lcm(n_y,m_y)*np.pi,lcm(n_y,m_y)*np.pi,args=(alpha,flux,n_x,m_x,n_y,m_y))]
            return np.asarray(phi)
    
        # resonant frequency
        ax1.plot(flux_th, freq_distributed_with_coupling(Current(flux_th,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu, mode=1)*1e-9, color = colors[i], linewidth=1, linestyle='--', label=labels[i])
        # g4 nonlin    
        ax2.plot(flux_th, np.abs(12*g4_distributed(flux_th, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)*1e-6), color=colors[i], linewidth=1.0, linestyle='-', label=labels[i])                         
        #  g3 nonlin 
        ax3.plot(flux_th, np.abs(g3_distributed(flux_th, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)*1e-6), color=colors[i], linewidth=1.0, linestyle='-', label=labels[i])    

    ax1.legend(loc='best',fontsize='small')
    ax2.legend(loc='best',fontsize='small')
    ax3.legend(loc='best',fontsize='small')





#   --------------------------------------------------------------------------


#    if Tunability:
#        fig=plt.figure(figsize=(9, 7),dpi=120)
#        fig.suptitle('Flux sweep', fontsize=20)
#        plt.xlabel('$\Phi/\Phi_0$', fontsize='x-large')
#        plt.ylabel('$f_c(\Phi)$, GHz', fontsize='x-large')
#        for index, M in enumerate(array_of_Ms):
##            f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
#            plt.plot(flux_th, freq_distributed_with_coupling(Current(flux_th,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu, mode=1)*1e-9, color = colors[index], linewidth=1, linestyle='--', label=r'$M=%d$' % M,zorder=2)
##            plt.plot(flux_th, freq_distributed_with_coupling(Current(flux_th,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu, mode=2)*1e-9, color = colors[index], linewidth=1, linestyle='--', label=r'$M=%d$' % M,zorder=2)
#        plt.legend(loc='best',fontsize='small')
#
#     
#
#    if Nonlinearities:
#        # g4 nonlin    
#        fig = plt.figure(figsize=(9, 7),dpi=120)
#        fig.suptitle(r'$g_4 \;\rm nonlinearity$', fontsize=20)
#        ax = plt.subplot(1,1,1)
#        plt.ylabel(r'$4^{\rm th}\;\rm order\;\rm nonlin,\;\rm MHz$', fontsize='x-large')
#        ax.set_yscale('log')
##        plt.xlim(0.0, 0.51) 
#        for index, M in enumerate(array_of_Ms):
##            f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
#            plt.plot(flux_th, np.abs(12*g4_distributed(flux_th, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)*1e-6), color=colors[index], linewidth=1.0, linestyle='-', label='$M=%d$'% M,zorder=1)             
#        plt.legend(loc='upper left',fontsize='small')
#    
#        #  g3 nonlin 
#        fig = plt.figure(figsize=(9, 7),dpi=120)
#        fig.suptitle(r'$g_3 \;\rm nonlinearity$', fontsize=20)
#        plt.ylabel(r'$3^{\rm rd}\;\rm order\;\rm nonlin,\;\rm MHz$', fontsize='x-large')
#        ax = plt.subplot(1,1,1)
#        ax.set_yscale('log')
##        plt.xlim(0.05, 0.49)
##        plt.ylim(1e-2, 1e2)     
#        for index, M in enumerate(array_of_Ms):
##            f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
#            plt.plot(flux_th, np.abs(g3_distributed(flux_th, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)*1e-6), color=colors[index], linewidth=1.0, linestyle='-', label=r'$M=%d$'% M,zorder=1)    
#        plt.legend(loc='upper left',fontsize='small')



    if Coupling:
        fig=plt.figure(figsize=(9, 7),dpi=120)
        colors = mpl.cm.rainbow(np.linspace(0,1,len(array_of_Ms)))
        fig.suptitle('Coupling', fontsize=20)
        plt.xlabel('$\Phi/\Phi_0$', fontsize='x-large')
        plt.ylabel('$kappa(\Phi)$, MHz', fontsize='x-large')
        for index, M in enumerate(array_of_Ms):
#            f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
            plt.plot(flux_th, kappa(flux_th, a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu)*1e-6, color = colors[index], linewidth=1, linestyle='--', label=r'$M=%d$' % M,zorder=2)
        plt.legend(loc='best',fontsize='small')
#     ----------------------------------------------------------------------

   


    if critical_current:
        fig = plt.figure(figsize=(9, 7),dpi=120)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        fig.suptitle(r'$g_3\; \rm nonlinearity\; for\;$', fontsize=20)
        plt.xlabel(r'$\Phi/\Phi_0$', fontsize='x-large')
        plt.ylabel(r'$g_3\;\rm (MHz)$', fontsize='x-large')
        plt.ylim(1,1e9)
        plt.subplot(1,1,1).set_yscale('log')
        fluxes = np.linspace(0.1,0.5,500)
        f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
        
        p_j=(1/3+alpha)/M*( (f0_distributed/freq_distributed_with_coupling(Current(0,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu))**2-1 )
        print(p_j)
        plt.plot(fluxes, (g3_distributed(fluxes, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)/g4_distributed(fluxes, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu))**2, color="blue", linewidth=1.0, linestyle='-', label=r'$g_4$ vs $g_3$ estimate')
        plt.plot(fluxes, 2*E_j/freq_distributed_with_coupling(Current(fluxes,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu)*c2_eff(fluxes, p_j, alpha, M)*( 5*c4_eff(fluxes, p_j, alpha, M)/c5_eff(fluxes, p_j, alpha, M) )**2, color="magenta", linewidth=1.0, linestyle='-', label=r'$g_4$ vs $g_5$ estimate')
        plt.plot(fluxes, np.abs(2*E_j/freq_distributed_with_coupling(Current(fluxes,a,b), a, b, alpha, f0_distributed, Z_c, C_coupling, M, gamma, mu=mu)*c2_eff(fluxes, p_j, alpha, M)*20*c3_eff(fluxes, p_j, alpha, M)/c5_eff(fluxes, p_j, alpha, M) ), color="orange", linewidth=1.0, linestyle='-', label=r'$g_3$ vs $g_5$ estimate')
    
        
        plt.plot(fluxes, np.abs(kappa/8/g3_distributed(fluxes, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu))**2, color="red", linewidth=1.0, linestyle='-', label=r'$n_{\infty}$ for pumping')
        
        plt.legend(loc='best',fontsize='x-large')



# ---------------------------------------------------------------------------



    if Nonlinearities_v2:

        
        array_of_Ms = np.linspace(1,300,300)
        g4_zero_flux = []
        g4_half_flux = []
        g3_max = []
        particip_zero_flux = []
        particip_half_flux = []
        for index, M in enumerate(array_of_Ms):
            f0_distributed = f0_distributed_opt(freq_max, a, b, alpha, Z_c, gamma, C_coupling, mu, M)
            g4_zero_flux += [g4_distributed(0, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)]
            g4_half_flux += [g4_distributed(0.5, a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)]
            g3_max += [ max(np.abs(g3_distributed(np.linspace(0,0.5,100), a, b, alpha, f0_distributed, M, L_j, Z_c, mu=mu)))  ]
            particip_zero_flux += [participation(0, a, b, alpha, f0_distributed, Z_c, M, L_j, mu=mu)]
            particip_half_flux += [participation(0.5, a, b, alpha, f0_distributed, Z_c, M, L_j, mu=mu)]
            print(M)
        g3_max = np.asarray(g3_max)
        g4_half_flux = np.asarray(g4_half_flux)
        g4_zero_flux = np.asarray(g4_zero_flux)
        particip_half_flux = np.asarray(particip_half_flux)
        particip_zero_flux = np.asarray(particip_zero_flux)
        
        
        # g4 nonlin    
        fig = plt.figure(figsize=(6, 3),dpi=120)
        fig.suptitle(r'$g_4$ vs M', fontsize=20)
        ax = plt.subplot(1,1,1)
        plt.ylabel(r'$\chi=12g_4$ (MHz)', fontsize='x-large')
        plt.xlabel('M', fontsize='x-large')
        plt.plot(array_of_Ms, np.abs(12*g4_zero_flux*1e-6), color='blue', linewidth=3.0, linestyle='-', label='Zero flux',zorder=1)             
#        plt.plot(array_of_Ms, np.abs(12*g4_half_flux*1e-6), color='green', linewidth=1.0, linestyle='-', label='Half flux',zorder=1) 

    
#         g3 nonlin    
        fig = plt.figure(figsize=(6, 3),dpi=120)
        fig.suptitle(r'$g_3$ vs M', fontsize=20)
        ax = plt.subplot(1,1,1)
        plt.ylabel(r'$g_3$ (MHz)', fontsize='x-large')
        plt.xlabel('M', fontsize='x-large')
        plt.plot(array_of_Ms, np.abs(g3_max*1e-6), color='red', linewidth=3.0, linestyle='-', label='',zorder=1)             

        # participation 
        fig = plt.figure(figsize=(6, 3),dpi=120)
        fig.suptitle(r'Array participation ratio', fontsize=20)
        ax = plt.subplot(1,1,1)
        plt.ylabel(r'p', fontsize='x-large')
        plt.xlabel('M', fontsize='x-large')
        plt.plot(array_of_Ms, particip_zero_flux, color='green', linewidth=3.0, linestyle='-', label='Zero flux',zorder=1)   
#        plt.plot(array_of_Ms, particip_half_flux, color='blue', linewidth=3.0, linestyle='-', label='Half flux',zorder=1)   




#     --------------------- Dynamic Range --------------------
    
    if 0:
        
        flux_points_for_saturation = np.linspace(0.0,0.48,10)
        Signal_Powers_dBm = np.linspace(-160,-90,70)
        Signal_Powers = dBm_to_Watts(Signal_Powers_dBm)
        Pump_detunings = [0]
        
        Gains = np.zeros((len(flux_points_for_saturation),len(Pump_detunings),len(Signal_Powers)))
        Pump_Powers_theor = np.zeros(len(flux_points_for_saturation))
    
        array_of_Ms = [2,5,10,20]
        array_of_mus = [1/2,1/10]
        lineshapes = ['-','--',':']
        Happy_colors = mpl.cm.rainbow(np.random.permutation(np.linspace(0,1,len(flux_points_for_saturation))))
    
        n_idler, n_signal, n_pump = np.zeros_like(Gains), np.zeros_like(Gains), np.zeros_like(Gains)
            
            
    
        for index, M in enumerate(array_of_Ms):  
    
            fig1 = plt.figure(figsize=(9, 7),dpi=120)
            fig1.suptitle(r'Saturation curves for $M=%d$' %M, fontsize=20)                            
            ax1=plt.subplot(1,1,1)
            plt.ylabel(r'Gain, dB', fontsize='x-large')
            plt.xlabel(r'Signal power, dBm', fontsize='x-large')
            plt.ylim(5, 24)

            fig2 = plt.figure(figsize=(9, 7),dpi=120)
            fig2.suptitle(r'$\overline{n} \rm \;and \; n_{\rm crit} \; vs \; signal\; power\;for\; M=%d$'%M, fontsize=20)                            
            ax2=plt.subplot(1,1,1)
            ax2.set_yscale('log')
            plt.ylabel(r'$n, \rm photons$', fontsize='x-large')
            plt.xlabel(r'Signal power, dBm', fontsize='x-large')

            
            for mu_ind,mu in enumerate(array_of_mus):       
                
                f0_distributed = f0_distributed_opt(alpha2,gamma,freq_max)
            
                L_0 = 1/2*(  np.sqrt( (M*L_j/(alpha+1/n))**2 +(4/np.pi*Z_c/(2*np.pi*freq_max) )**2 ) - M*L_j/(alpha+1/n)  )
                p_j = L_j / L_0
                f0 = 1/(np.pi**2)*Z_c/L_0
                C = 1/L_0/(2*np.pi*f0)**2
    
                for l, f in enumerate(flux_points_for_saturation):
                    for s, Delta in enumerate(Pump_detunings):
    
    #                g3_ =  float( g3(f) )      
    #                g4_ = float( g4(f) )                                
                        g3_ =  float( g3_distributed(f) )      
                        g4_ = float( g4_distributed(f) )
    
                        freq_c = Freq_distributed(f)
                        freq_s = freq_c
                        freq_p = 2*(freq_c-Delta)                    
                        freq_i = freq_p-freq_s
                        omega = freq_s-freq_p/2
                        kappa_c = (freq_c**2)/(freq_max**2)*kappa_c_max
                        kappa_t = kappa_c + kappa_i
                        G = np.power(10,20.0/10)  # look at 20 dB gains
                        u_p = pump_strength_for_gain()   
                        print('marker')
                        alpha_p = u_p/freq_c   
                        P_p = float ( freq_c/kappa_c*h*u_p**2*2*np.pi )
                        Pump_dBm = Watts_to_dBm(P_p)
                        Pump_Powers_theor[l] = Pump_dBm
    
                        for i, P_s in enumerate(Signal_Powers):
    #            print( 'signal power %f dBm' % (Watts_to_dBm(P_s)) )
                            u_s = np.sqrt(P_s/(2*np.pi*h)*kappa_c/freq_c )        
                
                            Xp_guess = float( alpha_p )
                            Yp_guess = 0.0
                
                            Xs_guess = float( -u_s*( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )*(Delta+omega) )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
                            Ys_guess = float( -kappa_t/2*( (Delta+omega)**2+kappa_t**2/4-16*alpha_p**2*g3_**2  )*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
    
                            Xi_guess = float( 4*g3_*alpha_p*u_s*( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 ) )
                            Yi_guess = -4*g3_*alpha_p*kappa_t*omega*u_s/( ( Delta**2-omega**2+1/4*kappa_t**2 -16*alpha_p**2*g3_**2 )**2 + (omega*kappa_t)**2 )
                
    
                            G, n_pump[l,s,i], n_signal[l,s,i], n_idler[l,s,i] = Gain(Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess)                
                            Gain_dB = 10*np.log10(float(G))   
                            Gains[l,s,i] = Gain_dB
    
                       

                for l, f in enumerate(flux_points_for_saturation):
                    for s, Delta in enumerate(Pump_detunings): #linestyle=lineshapes[s]
                        ax1.plot(Signal_Powers_dBm, Gains[l,s], color = Happy_colors[l], linewidth=1.0, linestyle=lineshapes[mu_ind], label=r'$f=%.2f \; P_{\rm pump}=%.1f \; {\rm dBm}$' % (flux_points_for_saturation[l], Pump_Powers_theor[l]) ,zorder=1)  
#            

                for l, f in enumerate(flux_points_for_saturation):
                    for s, Delta in enumerate(Pump_detunings): #linestyle=lineshapes[s]
                        ax2.plot(Signal_Powers_dBm, n_pump[l,s]+n_signal[l,s]+n_idler[l,s], color = Happy_colors[l], linewidth=1.0, linestyle=lineshapes[mu_ind], label=r'$f=%.2f$' % flux_points_for_saturation[l] ,zorder=1)  
                        ax2.plot(Signal_Powers_dBm, np.ones(len(Signal_Powers_dBm))*n_crit(f), color = Happy_colors[l], linewidth=1.0, linestyle=':', label=r'$f=%.2f$' % flux_points_for_saturation[l] ,zorder=1)  
             
        
            
            ax1.legend(loc='lower left',fontsize='x-small')
            
            
            

#     --------------------- Dynamic Range Double mode --------------------
    
    if Saturation:
        
        flux_points_for_saturation = np.linspace(0.37,0.44,20)
        Happy_colors = mpl.cm.rainbow(np.linspace(0,1,len(flux_points_for_saturation)))
        Pump_powers = np.zeros(len(flux_points_for_saturation))
        
        f0_distributed = f0_distributed_opt(freq_max)

        
        Signal_Powers_dBm = np.linspace(-140,-70,800)
        Signal_Powers = dBm_to_Watts(Signal_Powers_dBm)
        Gains = np.zeros((len(flux_points_for_saturation),len(Signal_Powers)))
        
        
        for ind_f, f in  enumerate(flux_points_for_saturation):
        
            kappa_a = kappa(f,mode=1)
            kappa_b = kappa(f,mode=2)
            freq_a = Freq(f, k=1)       
            freq_b = Freq(f, k=2)
            freq_p = 2*freq_a
            freq_s = freq_a
            
            Delta_a = float( freq_a - freq_p/2 )
            Delta_b = float( freq_b - freq_p )
            omega = freq_s - freq_p/2
            
            g4_a = g4_distributed(f,mode=1)
            g4_b = g4_distributed(f,mode=2)
            g4_ab = -np.sqrt(np.abs(g4_a*g4_b))   # oh yeah
            g3_ab = g3_modes(f,[1,1,2])
            
            
            u_p_guess = np.sqrt((Delta_a**2+kappa_a**2/4)*(Delta_b**2+kappa_b**2)/4/g3_ab**2)
            G=100
            u_p = equation_for_up(Delta_a,Delta_b,u_p_guess)
            
            theta_p = 0
            
            
            P_p = (2*np.pi*h)*freq_b/kappa_b*u_p**2*(freq_b/freq_p)**2
            Pump_powers[ind_f] = Watts_to_dBm(P_p)
            
            for ind_P, P_s in enumerate(Signal_Powers):
                
                u_s = np.sqrt(kappa_a/freq_a*(freq_s/freq_a)**2*P_s/(2*np.pi*h))
                theta_s = 0
                # Kerr limited gain....
#                Delta_bguess = Delta_b + g4_ab*( (Delta_a+omega)**2+kappa_a**2/4+4*g3_ab**2/(Delta_b**2+kappa_b**2/4)*u_p**2 )/((Delta_a**2-omega**2+kappa_a**2/4-4*g3_ab**2/(Delta_b**2+kappa_b**2/4)*u_p**2 )**2 + (kappa_a*omega)**2 )*u_s**2+12*g4_b/( Delta_b**2+kappa_b**2/4 )*u_p**2 
#                Delta_aguess = Delta_a + 12*g4_a*( (Delta_a+omega)**2+kappa_a**2/4+4*g3_ab**2/(Delta_b**2+kappa_b**2/4)*u_p**2 )/((Delta_a**2-omega**2+kappa_a**2/4-4*g3_ab**2/(Delta_b**2+kappa_b**2/4)*u_p**2 )**2+(kappa_a*omega)**2)*u_s**2 + g4_ab/( Delta_b**2+kappa_b**2/4 )*u_p**2 
#                G = Gain(Delta_aguess, Delta_bguess)
#                Gains[ind_f,ind_P] = 10*np.log10(G)



                Xp_guess = float( -u_p/( Delta_b**2 + kappa_b**2/4 )*(Delta_b*np.cos(theta_p)-kappa_b/2*np.sin(theta_p) ) )
                Yp_guess = float( -u_p/( Delta_b**2 + kappa_b**2/4 )*(Delta_b*np.sin(theta_p)+kappa_b/2*np.cos(theta_p) ) )
                Xs_guess = float( -u_s/( Delta_a**2+kappa_a**2/4 - 4*g3_ab**2*(Xp_guess**2+Yp_guess**2) )*(Delta_a*np.cos(theta_s)-kappa_a/2*np.sin(theta_s) )    )        
                Ys_guess = float( -u_s/( Delta_a**2+kappa_a**2/4 - 4*g3_ab**2*(Xp_guess**2+Yp_guess**2) )*(Delta_a*np.sin(theta_s)+kappa_a/2*np.cos(theta_s) )    )
                Xi_guess = float( 2*g3_ab*u_s/( Delta_a**2+kappa_a**2/4 - 4*g3_ab**2*(Xp_guess**2+Yp_guess**2) )*( Xp_guess*np.cos(theta_s)-Yp_guess*np.sin(theta_s) )  )
                Yi_guess = float( 2*g3_ab*u_s/( Delta_a**2+kappa_a**2/4 - 4*g3_ab**2*(Xp_guess**2+Yp_guess**2) )*( Yp_guess*np.cos(theta_s)+Xp_guess*np.sin(theta_s) )  )
            
                G = _Gain(Xp_guess, Yp_guess, Xs_guess, Ys_guess, Xi_guess, Yi_guess)[0]
                Gains[ind_f,ind_P] = 10*np.log10(G)

            
            
        fig = plt.figure(figsize=(9, 7),dpi=120)
        fig.suptitle(r'Saturation curves', fontsize=20)
        ax=plt.subplot(1,1,1)
        plt.ylabel(r'Gain, dB', fontsize='x-large')
        plt.xlabel(r'Signal power, dBm', fontsize='x-large')

        for ind_f, ind_P in enumerate(flux_points_for_saturation):
                ax.plot(Signal_Powers_dBm, Gains[ind_f], color = Happy_colors[ind_f], linewidth=1.0, label=r'$f=%.3f \; P_{\rm pump}=%.1f \; {\rm dBm}$' % (flux_points_for_saturation[ind_f], Pump_powers[ind_f]) )  
        ax.legend(loc='lower left',fontsize='x-small')   
        ax.plot(Signal_Powers_dBm, 19*np.ones(len(Signal_Powers_dBm)), color = 'black', linewidth=1.0 )  
        ax.plot(Signal_Powers_dBm, 21*np.ones(len(Signal_Powers_dBm)), color = 'black', linewidth=1.0 )  
   
         
                     
        