# find steady state
import time
import numpy as np
from scipy import optimize
from scipy.optimize import root_scalar


import blocks
from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[:,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        
        if i_fix == 0:
            ss.Dbeg[i_fix,:,0] = e_ergodic*par.sT
        elif i_fix == 1:
            ss.Dbeg[i_fix,:,0] = e_ergodic*(1-par.sT)
        else:
            raise NotImplementedError('i_fix must be 0 or 1')
        
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    # CHANGE TO NEW FUNCTIONIONAL FORM 
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]

            if i_fix == 0:
                inc = ss.inc_TH/par.sT*z
            elif i_fix == 1:
                inc = ss.inc_NT/(1-par.sT)*z

            c = (1+ss.ra)*par.a_grid + inc
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]

        
def obj_ss(x, model, do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    par.nu_ = x[0]
    par.sT = x[1]
    ss.beta = par.beta

    # a. prices

    # normalzied to 1
    # for varname in ['PF_s','E','PF','PTH','PT','PNT','P','PTH_s','Q']:
    for varname in ['PF_s','E','PF','PTH','PT','P','PTH_s', 'p', 'PNT','P']:
        ss.__dict__[varname] = 1.0

    
    
    # zero inflation
    for varname in ['pi_F_s','pi_F','pi_TH','pi_T','pi_NT','pi','pi_TH_s','piWTH','piWNT']:
        ss.__dict__[varname] = 0.0

    # real+nominal interest rates are equal to foreign interest rate
    ss.ra = ss.i = ss.iF_s = ss.r_real = ss.rF = par.rF_ss
    ss.UIP_res = 0.0


    # domestic interes rate shock:
    ss.i_shock = 0.0

    # b. production

    # normalize TFP and labor
    ss.ZTH = 1.0
    ss.ZNT = 1.0
    ss.NTH = 1.0*par.sT
    ss.NNT = 1.0*(1-par.sT)


    ss.n_NT = ss.NNT/par.sT
    ss.n_TH = ss.NTH/(1-par.sT)
    

    # production
    ss.YTH = ss.ZTH*ss.NTH
    ss.YNT = ss.ZNT*ss.NNT

    # real = nominal wages = value of mpl **** fixed 
    # ss.wTH = ss.WTH = ss.PTH*ss.ZTH
    # ss.wNT = ss.WNT = ss.PNT*ss.ZNT
    
    ss.WTH = ss.PTH*ss.ZTH
    ss.WNT = ss.PNT*ss.ZNT



    # c. household 
    ss.tau = par.tau_ss
    ss.inc_TH = (1-ss.tau)*ss.WTH*ss.NTH
    ss.inc_NT = (1-ss.tau)*ss.WNT*ss.NNT 



    par.run_u == False
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)


    # Nominal values
    ss.EX = ss.E_hh*ss.PNT # Expenditure
    ss.A = ss.A_hh*ss.PNT

    # d. government bonds (nominal)
    ss.B = ss.A

    # Real governmen consumption
    ss.G = (ss.tau*(ss.WTH*ss.NTH+ss.WNT*ss.NNT)-ss.i*ss.B)/ ss.PNT #  *** Why ex-ante interest?

    #Monetary policy
    if par.float == True:
        ss.CB = ss.E 
    else:
        ss.CB = ss.i
    
    # e. consumption

    ss.CT = ss.CT_hh # par.alphaT*ss.C_hh 
    ss.CNT = ss.CNT_hh #(1-par.alphaT)*ss.C_hh

    # home vs. foreign
    ss.CTH =  ss.CTH_hh #(1-par.alphaF)*ss.CT
    ss.CTF = ss.CTF_hh #par.alphaF*ss.CT

    # size of foreign market
    ss.CTH_s = ss.M_s = ss.YTH - ss.CTH # clearing_T

    # f. market clearing
    ss.clearing_YTH = ss.YTH - ss.CTH - ss.CTH_s 
    ss.clearing_YNT = ss.YNT - ss.CNT - ss.G

    # zero net foreign assets
    ss.NFA = ss.A - ss.B

    # zero net foreign assets
    ss.GDP = ss.PTH*ss.YTH + ss.PNT*ss.YNT
    ss.NX = ss.GDP - ss.EX - ss.PNT*ss.G 
    ss.NFA = ss.A - ss.B
    ss.CA = ss.NX + (1+ss.i)*ss.NFA
    ss.Walras = ss.CA

    # g. disutility of labor for NKWPCs

    ss.wTH = ss.WTH /1# w_tilde deflated with PNT
    ss.wNT = ss.WNT /1 # wage deflated with PIGL price index= 1 in initial steady state***.. Or is it

    par.varphiTH = 1/par.muw*(1-ss.tau)*ss.wTH*ss.UC_TH_hh / ((ss.NTH/par.sT)**par.nu)
    par.varphiNT = 1/par.muw*(1-ss.tau)*ss.wNT*ss.UC_NT_hh / ((ss.NNT/(1-par.sT))**par.nu)
    ss.NKWCT_res = 0.0
    ss.NKWCNT_res = 0.0

    # Utility 
    # ss.U = ss.U_hh - par.varphiTH *(ss.NTH/par.sT)**(1+par.nu)/(1+par.nu) - par.varphiNT *(ss.NNT/(1-par.sT))**(1+par.nu)/(1+par.nu)

    return [ss.clearing_YNT, ss.NX] #ss.NFA


def find_ss(model, do_print=False): 
    """ find the steady state """

    par = model.par
    ss = model.ss

    t0 = time.time()

    # a. Finding steady state  
    try:
        # Initial guess for solution
        x0 = [0.7, 0.5]  
        
        # Optimizer
        res = optimize.root(obj_ss, x0, args=(model,), method='hybr')  # or another method like 'lm', 'broyden1', etc.
        # obj_ss[res.x]
        print(f'Share of domestic workers in tradable sector = {res.x[1]:.2f}')
    except Exception as e:
        print(f"Failed: {e}")

    # b. Reruning model
    par.run_u = True
    obj_ss(res.x, model)

    # c. Initial average expenditure share on tradable goods, used for later calculating cost of living changes
    par.omega_T_ = ss.PNT * ss.CT_hh / ss.E_hh *ss.PNT



    # d. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.inc_TH = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.nu_ = :.3f}')
        print(f'{par.alphaF = :.3f}')
        print(f'{par.varphiTH = :.3f}')
        print(f'{par.varphiNT = :.3f}')
        print(f'{ss.M_s = :.3f}')
        print(f'{ss.clearing_YTH = :12.8f}')
        print(f'{ss.clearing_YNT = :12.8f}')
        print(f'{ss.G = :.3f}')
        print(f'{ss.NFA = :.3f}')





def find_ss_new(model, do_print=False):

    par = model.par
    ss = model.ss



    # # if do_print: print(f'starting at [{initial_guess[0]:.4f}]')


    # result = root_scalar(evaluate_ss, args=(model,), method='brentq', bracket=[0.1, 2.0])
    
    # # print(result)
    
    # evaluate_ss(result.root, model, do_print=True)


    if do_print:

        # print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.clearing_YTH = :12.8f}')
        print(f'{ss.clearing_YNT = :12.8f}')
        print(f'{ss.PNT = :12.8f}')
        print(f'{ss.inc_TH = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.alphaT = :.3f}')
        print(f'{par.alphaF = :.3f}')
        print(f'{par.varphiTH = :.3f}')
        print(f'{par.varphiNT = :.3f}')
        print(f'{ss.M_s = :.3f}')
        print(f'{ss.G = :.3f}')
        print(f'{ss.NFA = :.3f}')

    # # Call optimize.root with a scalar initial guess
    # try:
    #     res = optimize.root(evaluate_ss, float(initial_guess), args=(model,))
    # except ValueError as e:
    #     print("Error encountered:", e)

    # res = optimize.root(evaluate_ss, initial_guess, args=(model,))

    # if do_print: 
    #     print('')
    #     print(res)
    #     print('')