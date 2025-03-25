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
    
    # CHANGE TO NEW FUNCTIONIONAL FORM ****
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

    
    par.sT = x[0]

    # par.nu = x[0]
    # par.sT = x[1]

    # a. prices

    # normalzied to 1
    # for varname in ['PF_s','E','PF','PTH','PT','PNT','P','PTH_s','Q']:
    for varname in ['PF_s','E','PF','PTH','PT','P','PTH_s', 'p', 'PNT','P', 'PE', 'PTHF', 'PE_s', 'Q', 'DomP']:
        ss.__dict__[varname] = 1.0

    
    
    # zero inflation
    for varname in ['pi_F_s','pi_F','pi_TH','pi_T','pi_NT','pi','pi_TH_s','piWTH','piWNT', 'pi_DomP']:
        ss.__dict__[varname] = 0.0

    # real+nominal interest rates are equal to foreign interest rate
    ss.ra = ss.i = ss.iF_s = ss.r_real = ss.rF = par.rF_ss


    # domestic interes rate shock:
    ss.i_shock = 0.0

    # b. production

    # Productivity 
    ss.ZTH = 1.0
    ss.ZNT = 1.0
    # Aggregate labor supply 
    ss.NTH = 1.0*par.sT
    ss.NNT = par.sNT = 1.0*(1-par.sT)
    ss.N = ss.NTH + ss.NNT  
    # labor supply per houshold
    ss.n_TH = ss.NTH/par.sT
    ss.n_NT = ss.NNT/(1-par.sT)
    
    # production
    ss.YTH = ss.ZTH*ss.NTH
    ss.YNT = ss.ZNT*ss.NNT

    # real = nominal wages = value of mpl **** fixed 
    # ss.wTH = ss.WTH = ss.PTH*ss.ZTH
    # ss.wNT = ss.WNT = ss.PNT*ss.ZNT

    # c. NKPC
    # wages 
    ss.WTH = ss.ZTH/par.mu_p
    ss.WNT = ss.ZNT/par.mu_p
    
    # Adjustment cost
    ss.adj_TH = 0.0
    ss.adj_NT = 0.0

    # Dividend/profit
    ss.div_TH = ss.YTH*ss.PTH - ss.WTH*ss.NTH - ss.adj_TH*ss.YTH
    ss.div_NT = ss.YNT*ss.PNT - ss.WNT*ss.NNT - ss.adj_NT*ss.YNT
     
    
    # c. household 

    # Income
    ss.tau = par.tau_ss
    ss.inc_TH = (1-ss.tau)*ss.WTH*ss.NTH +ss.div_TH # Total income to households in the tradable sector
    ss.inc_NT = (1-ss.tau)*ss.WNT*ss.NNT +ss.div_NT# Total income to households in the non-tradable sector

    ss.INC = ss.inc =  ss.inc_TH + ss.inc_NT

    # Solving and simulating the household block
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
    # Tradable and non-tradable consumption
    ss.CT = ss.CT_hh # par.alphaT*ss.C_hh 
    ss.CNT = ss.CNT_hh #(1-par.alphaT)*ss.C_hh

    # Energy and non-energy tradable consumption
    ss.CE = par.alphaE*(ss.PE/ss.PT)**(-par.etaE)*ss.CT_hh
    ss.CTHF = (1-par.alphaE)*(ss.PTHF/ss.PT)**(-par.etaE)*ss.CT_hh

    # Home and foreign tradable consumption
    ss.CTF = par.alphaF*(ss.PF/ss.PTHF)**(-par.etaF)*ss.CTHF
    ss.CTH = (1-par.alphaF)*(ss.PTH/ss.PTHF)**(-par.etaF)*ss.CTHF

    # size of foreign market
    ss.CTH_s = ss.M_s = ss.YTH - ss.CTH # clearing_T

    # f. market clearing
    ss.clearing_YTH = ss.YTH - ss.CTH - ss.CTH_s  - ss.adj_TH*ss.YTH
    ss.clearing_YNT = ss.YNT - ss.CNT - ss.G - ss.adj_NT*ss.YNT

    # zero net foreign assets
    ss.NFA = ss.A - ss.B

    # Nominel GDP
    ss.GDP = ss.PTH*ss.YTH *(1-ss.adj_TH) + ss.PNT*ss.YNT*(1-ss.adj_NT) 

    # Net export 
    ss.NX = ss.GDP - ss.EX - ss.PNT*ss.G  

    # Current account
    ss.CA = ss.NX + (1+ss.i)*ss.NFA

    # Walras 
    ss.Walras = ss.CA

    # g. disutility of labor for NKWPCs
    par.varphiTH = 1/par.mu_w*(1-ss.tau)*ss.wTH*ss.UC_TH_hh / ((ss.NTH/par.sT)**par.kappa)
    par.varphiNT = 1/par.mu_w*(1-ss.tau)*ss.wNT*ss.UC_NT_hh / ((ss.NNT/(1-par.sT))**par.kappa)

    # Wage philp curve residuals
    ss.NKWCT_res = 0.0
    ss.NKWCNT_res = 0.0

    # Price Philips curve residuals
    ss.NKPCT_res = 0.0
    ss.NKPCNT_res = 0.0   

    # UIP  residuals
    ss.UIP_res = 0.0


    # Additional variables
    ss.wTH = ss.WTH /1# 
    ss.wNT = ss.WNT /1 # wage deflated with PIGL price index= 1 in initial steady state***.. Or is it
    ss.W = par.sT*ss.WTH + (1-par.sT)*ss.WNT # average wage
    ss.w = ss.W/ss.P
    ss.YH = ss.YTH + ss.YNT

    return [ss.clearing_YNT] 
    # return [ss.clearing_YNT, ss.NX] #ss.NFA


def find_ss(model, do_print=False): 
    """ find the steady state """

    par = model.par
    ss = model.ss

    t0 = time.time()

    # a. Finding steady state  
    try:
        # Initial guess for solution
        # x0 = [0.7, 0.5] 
        x0 = [ 0.5]  
        
        # Optimizer
        res = optimize.root(obj_ss, x0, args=(model,), method='hybr')  # or another method like 'lm', 'broyden1', etc.
        # obj_ss[res.x]
        print(f'Share of domestic workers in tradable sector = {res.x[0]:.2f}')
    
    except Exception as e:
        print(f"Failed: {e}")

    # b. Reruning model an calcualting the utility using parameters from the steady state
    par.run_u = True
    obj_ss(res.x, model)

    # c. Initial average expenditure share on tradable goods, used for later calculating cost of living changes
    par.omega_T = par.nu *ss.E**(-par.epsilon)*ss.p**(par.gamma) # *** Doublet tjek formel
    print(f'Average share of consumption of tradables{par.omega_T = :.3f}')

    # Average elicticity of substitution between tradable and non-tradable goods 
    par.eta_T_RA = 1 - par.gamma - (par.nu*(ss.PT/ss.PNT)**par.gamma) / ( (ss.EX/ss.PNT)**par.epsilon - par.nu*(ss.PT/ss.PNT)**par.gamma) * (par.gamma - par.epsilon)
    print(f'Average elasticity of substitution between tradable and non-tradable goods{par.eta_T_RA = :.3f}')


    # d. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.inc_TH = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.nu = :.3f}')
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