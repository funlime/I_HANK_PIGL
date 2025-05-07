# solving the household problem

import numpy as np
import numba as nb
from GEModelTools import prev, next, lag, lead, isclose

from consav.linear_interp import interp_1d_vec
import math

@nb.njit
def price_index(P1,P2,eta,alpha):
    return (alpha*P1**(1-eta)+(1-alpha)*P2**(1-eta))**(1/(1-eta))



@nb.njit       
# def solve_hh_backwards(par,z_trans,beta,ra,vbeg_a_plus,vbeg_a,a,c, uc_TH,uc_NT, e, cnt, ct, cth, ctf,  PT, PNT, PF, PTH, u, n_NT,n_TH, WNT,WTH, tau, ce, cthf , PE, PTHF):
def solve_hh_backwards(par,z_trans,ra,vbeg_a_plus,vbeg_a,a,c, inc_NT, inc_TH, uc_TH,uc_NT, e, cnt, ct,   u, n_NT,n_TH,  p , v):

    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """



    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            # a. solve step

            # i. income
            if i_fix == 0:
                # inc = (n_NT*WNT*(1-tau))/PNT
                inc = inc_TH/par.sT
            else:
                # inc = (n_TH*WTH*(1-tau))/PNT
                inc = inc_NT/par.sNT #(1-par.sT)
         
            z = par.z_grid[i_z]

            # ii. EGM
            e_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/(1- par.epsilon ) )
            m_endo = e_endo + par.a_grid
            
            # iii. interpolation to fixed grid
            m = (1+ra)*par.a_grid + inc*z
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            e[i_fix,i_z] = m-a[i_fix,i_z]



        # b. expectation step
        # Marginal utility of expenditure 
        v_a = (1+ra)*e[i_fix]**(-( 1 - par.epsilon ) )
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

        # 
        # if par.run_u == True:
        #     print('Running u')
            # if math.isclose(par.epsilon,0):
            #     print(' epsilon is close to 0')
            #     v_ = (1/par.epsilon) * ( (e[0,:,:])**par.epsilon -1) - (par.nu/par.gamma)*( (p)**par.gamma -1)  -  par.varphiTH*(n_TH**(1+par.kappa))/ (1+par.kappa)
            #     v[i_fix] = z_trans[i_fix]@v_ # Skal det divideres med beta 



    
    # extra output
    uc_TH[:] = 0.0
    uc_NT[:] = 0.0

    # elif par.pref == 'PIGL':
    ct[:] = e * (p**(-1))  *  par.nu*  e**(-par.epsilon)  *p**(par.gamma)
    cnt[:] = e*(1-par.nu*e**(-par.epsilon)*p**(par.gamma))


    # if par.brute_force_C == False:
    #         ct[:] = e * (p**(-1))  *  par.nu*  e**(-par.epsilon)  *p**(par.gamma)
    #         cnt[:] = e*(1-par.nu*e**(-par.epsilon)*p**(par.gamma))


    # elif par.pref == 'PIGL_forces':

    #     epsilon_ = 0.18
    #     gamma_ = 0.29
    #     ct[:] = e * (p**(-1))  *  par.nu*  e**(-epsilon_)  *p**(gamma_)
    #     cnt[:] = e*(1-par.nu*e**(-epsilon_)*p**(gamma_))


    # elif par.pref == 'homothetic_force':

    #     ct[:] = e * (p**(-1))  *  par.nu *p**(par.gamma)
    #     cnt[:] = e*(1-par.nu*p**(par.gamma))



    # # Non homothetic consumption of tradables and non tradables

    # # Preferences 
    # elif par.pref == 'PIGL':
    #     ct[:] = e * (p**(-1))  *  par.nu*  e**(-par.epsilon)  *p**(par.gamma)
    #     cnt[:] = e*(1-par.nu*e**(-par.epsilon)*p**(par.gamma))




    # elif par.pref == 'CUBB_douglas': # Change to CD
    #     ct[:] = e*p**(-1)*par.nu
    #     cnt[:] = e*(1-par.nu)
    



        # ct[:] = e*p**(-1)*par.nu*p**par.gamma
        # cnt[:] = e*(1-par.nu*p**par.gamma)

    # else:
    #     raise NotImplementedError('Only PIGL and Cubb douglas preferences are implemented')
    # Cubb douglas
    # if par.PIGL == False:

    
    # else: 




    # Second term in consumption demand 
    # temp = par.nu*e**(-par.epsilon)*p**par.gamma

    # ct[:] = e/p*par.nu*e**(-par.epsilon)*p**(par.gamma)
    # cnt[:] = e*(1-par.nu*e**(-par.epsilon)*p**par.gamma)

    # Non-homothetic consumption of tradables and non tradables 
    # ct[:] = e/p_*par.nu*e**(-par.epsilon)*p_**(par.gamma) Helt forkert jo.....
    # cnt[:] = e*(1-par.nu*e**(-par.epsilon)*p_**par.gamma)
 
    # Tjeking budget constraint
    # print(e*PNT - cnt*PNT - ct*PT)

    # CES share of tradable good (THF) and energy consumption agregate called T 

    # ce[:] = par.alphaE*(PE/PT)**(-par.etaE)*ct
    # cthf[:] = (1-par.alphaE)*(PTHF/PT)**(-par.etaE)*ct

    # # CES shares og home and foreign tra
    # ctf[:] = par.alphaF*(PF/PTHF)**(-par.etaF)*cthf
    # cth[:] = (1-par.alphaF)*(PTH/PTHF)**(-par.etaF)*cthf

    # Calculating utility requires disutility of labor parpameter calculated in the steady state
    if par.run_u == False:
        u[:] = 0.0
    if par.run_u == True:
        try:
            if isclose(par.epsilon,0) or isclose(par.gamma,0):
                u[:] = 0.0
            else:
                u[0,:,:]=  (1/par.epsilon) * ( (e[0,:,:])**par.epsilon -1) - (par.nu/par.gamma)*( (p)**par.gamma -1)  -  par.varphiTH*(n_TH**(1+par.kappa))/ (1+par.kappa)
                u[1,:,:]=   (1/par.epsilon) * ( (e[1,:,:])**par.epsilon -1) - (par.nu/par.gamma)*( (p)**par.gamma -1)  - par.varphiNT*(n_NT**(1+par.kappa))/ (1+par.kappa) 

        except:
            pass



    for i_z in range(par.Nz):
        # Marginal utility of expenditure (not marginal utility of expenditure tilde)
        
        uc_TH[0,i_z,:] = e[0,i_z,:]**(-(1- par.epsilon ) )*par.z_grid[i_z]
        uc_NT[1,i_z,:] = e[1,i_z,:]**(-(1- par.epsilon ) )*par.z_grid[i_z]

        # uc_TH[0,i_z,:] = PNT**(-par.epsilon)*(PNT*e[0,i_z,:])**(-(1- par.epsilon ) )*par.z_grid[i_z]
        # uc_NT[1,i_z,:] = PNT**(-par.epsilon)*(PNT*e[1,i_z,:])**(-(1- par.epsilon ) )*par.z_grid[i_z]

        # c_TH[0,i_z,:] = c[0,i_z,:]
        # c_NT[1,i_z,:] = c[1,i_z,:]
    # c_TH[:] /= par.sT
    # c_NT[:] /= (1-par.sT)

    uc_TH[:] /= par.sT
    uc_NT[:] /= (1-par.sT)


    