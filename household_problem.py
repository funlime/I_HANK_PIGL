# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def price_index(P1,P2,eta,alpha):
    return (alpha*P1**(1-eta)+(1-alpha)*P2**(1-eta))**(1/(1-eta))



@nb.njit       
def solve_hh_backwards(par,z_trans,beta,ra,inc_TH,inc_NT,vbeg_a_plus,vbeg_a,a,c, uc_TH,uc_NT, e, cnt, ct, cth, ctf, p, PT, PF, PTH, u):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            # a. solve step

            # i. income
            if i_fix == 0:
                inc = inc_TH/par.sT
            else:
                inc = inc_NT/(1-par.sT)
         
            z = par.z_grid[i_z]

            # ii. EGM
            e_endo = (beta*vbeg_a_plus[i_fix,i_z])**(-1/(1- par.epsilon_ ) )
            m_endo = e_endo + par.a_grid
            
            # iii. interpolation to fixed grid
            m = (1+ra)*par.a_grid + inc*z
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            e[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        v_a = (1+ra)*e[i_fix]**(-( 1 - par.epsilon_ ) )
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

    # extra output
    uc_TH[:] = 0.0
    uc_NT[:] = 0.0




    # Non homothetic consumption of tradables and non tradables
    # Second term in consumption demand 
    # temp = par.nu_*e**(-par.epsilon_)*p**par.gamma_

    ct[:] = e/p*par.nu_*e**(-par.epsilon_)*p**(par.gamma_)
    cnt[:] = e*(1-par.nu_*e**(-par.epsilon_)*p**par.gamma_)


    # CES shares og home and foreign tra
    ctf[:] = par.alphaF*(PF/PT)**(-par.etaF)*ct
    cth[:] = (1-par.alphaF)*(PTH/PT)**(-par.etaF)*ct

    # Utility 
    u[:]=  (1/par.epsilon_) * ( (e)**par.epsilon_ -1) - (par.nu_/par.gamma_)*( (p)**par.gamma_ -1) 


    PNT = PT/p

    for i_z in range(par.Nz):
        
        uc_TH[0,i_z,:] = PNT**(-par.epsilon_)*e[0,i_z,:]**(-(1- par.epsilon_ ) )*par.z_grid[i_z]
        uc_NT[1,i_z,:] = PNT**(-par.epsilon_)*e[1,i_z,:]**(-(1- par.epsilon_ ) )*par.z_grid[i_z]

        # c_TH[0,i_z,:] = c[0,i_z,:]
        # c_NT[1,i_z,:] = c[1,i_z,:]
    # c_TH[:] /= par.sT
    # c_NT[:] /= (1-par.sT)

    uc_TH[:] /= par.sT
    uc_NT[:] /= (1-par.sT)


    