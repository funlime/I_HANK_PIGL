import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class IHANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        # self.inputs_hh = ['beta','ra', 'PT', 'PF', 'PTH', 'n_NT','n_TH', 'WNT','WTH', 'tau', 'PNT', 'PE','PTHF'] # direct inputs
        self.inputs_hh = ['ra',  'p', 'n_NT','n_TH',  'inc_NT', 'inc_TH',] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        # self.outputs_hh = ['a','c','uc_TH','uc_NT', 'e', 'cnt', 'ct', 'cth', 'ctf', 'u', 'ce','cthf'] # outputs
        self.outputs_hh = ['a','c','uc_TH','uc_NT', 'e', 'cnt', 'ct', 'u', 'v', 'q', 'x'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZTH','ZNT','M_s','rF','PF_s','G','i_shock', 'PE_s'] # exogenous inputs
        self.unknowns = ['CB','NNT','NTH','piWTH','piWNT', 'pi_NT', 'pi_TH'] # endogenous inputs
        self.targets = ['NKWCT_res','NKWCNT_res','clearing_YTH','clearing_YNT','UIP_res', 'NKPCT_res', 'NKPCNT_res' ] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.mon_pol',
            'blocks.production',
            'blocks.prices',
            # 'blocks.inflation', 
            'blocks.central_bank',
            'blocks.government',
            'blocks.intermediary_goods',
            'blocks.HH_pre',
            'hh',
            'blocks.HH_post',
            # 'blocks.cost_of_living_index',
            'blocks.NKWCs',
            'blocks.UIP',
            # 'blocks.consumption',
            'blocks.market_clearing',            
            'blocks.accounting',            
        ]        

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """

        par = self.par


        # For tejkking 
        # par.alt = False
        par.sticky_prices = True
        par.real_exchange_rate_PTH  = False

        # New 
        par.epsilon = 0.22 # controls the degree of non-homotheticity 
        par.gamma = 0.25 # controls the non-constant elicticity of substitution  between tradable and non-tradable goods
        par.gamma_homo = 0.12
        # par.epsilon = 0.18 # controls the degree of non-homotheticity 
        # par.gamma = 0.29 # controls the non-constant elicticity of substitution  between tradable and non-tradable goods
        par.nu = 0.55 # Scalling parameter
        par.omega_T = np.nan # agregate expenditure share on tradables in steady state
        par.run_u = False
        par.mon_policy = 'taylor'
        par.pf_fixed = True
        par.etaE = 0.4 # elasticity of substitution between tradable goods and energy 
        par.alphaE = 0.05 # share of energy in tradable + energy consumption
        par.eta_T_RA = np.nan
        # par.phi_inflation = 1.0
        par.sNT = np.nan # share of Workers in the non-tradable sector - determined in ss
        par.pref = 'PIGL' # 'PIGL' or 'Cuub douglas'
        par.brute_force_C = False
        par.CES_price_index = False
        par.real_wage_motive = 5.0

        # a. discrete states
        par.Nfix = 2 # number of sectors sectors
        par.Nz = 7 # idiosyncratic productivity
        par.sT = np.nan # share of workers in tradeable sector
        
        
        # Variables Monitary policy  
        par.rho_i = 0.9 # persistance of monetary policy
        par.phi_pi = 1.5 # inflation coefficient

        # b. preferences
        par.beta = 0.985 #0.975 # discount factor
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution

        # par.alphaT = np.nan # share of tradeable goods in home consumption (determined in ss)
        # par.etaT = 0.5 #2.0 # elasticity of substitution between tradeable and non-tradeable goods
        
        par.alphaF = 0.3 # share of foreign goods in home tradeable consumption
        par.etaF = 0.51 #*** # elasticity of substitution between home and foreign tradeable goods
          
        par.varphiTH = np.nan # disutility of labor in tradeable sector (determined in s)
        par.varphiNT = np.nan # disutility of labor in non-tradeable sector (determined in s)
        par.kappa = 2.0 # Frisch elasticity of labor supply
              
        # c. income parameters
        par.rho_z = 0.966 # AR(1) coefficient for idiosyncratic productivity Floden and Linde (2001) calibration
        par.sigma_psi = 0.13 # std. of psi - Floden and Linde (2001) calibration
        
        # d. price setting
        # NKWPC
        # par.kappa_w = 0.05 # slope of wage Phillips curve
        # par.mu_w = 1.2 # wage mark-up     
        par.kappa_w = 0.03 # slope of wage Phillips curve
        par.mu_w = 1.1 # wage mark-up       
 
        # NKPC
        # par.kappa_p = 0.1 # slope of price Phillips curve
        # par.mu_p = 1.2 # wage mark-down
        par.kappa_p = 0.15 # slope of price Phillips curve
        par.mu_p = 1.1 # wage mark-down

        # e. foreign Economy
        par.rF_ss = 0.005 # exogenous foreign interest rate
        par.eta_s =  0.51 # # Armington elasticity of foreign demand
        par.M_s_ss = np.nan # size of foreign market (determined in ss)

        # f. government
        par.tau_ss = 0.20 # tax rate on labor income
        par.omega = 0.10 # tax sensitivity to debt

        # central bank
        par.float = True # float or fix exchange rate
        par.phi = 1.5 # Taylor rule coefficient on inflation

        # g. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # h. shocks
        par.jump_M_s = 0.00 # initial jump
        par.rho_M_s = 0.00 # AR(1) coefficeint
        par.std_M_S = 0.00 # std.

        par.jump_rF = 0.00 # initial jump
        par.rho_rF = 0.00 # AR(1) coefficeint
        par.std_rF = 0.00 # std.

        par.jump_PF_s = 0.00 # initial jump
        par.rho_PF_s = 0.00 # AR(1) coefficeint
        par.std_PF_s = 0.00 # std.

        par.jump_beta = 0.00 # initial jump
        par.rho_beta = 0.00 # AR(1) coefficeint
        par.std_beta = 0.00 # std.

        par.jump_G = 0.00 # initial jump
        par.rho_G = 0.00 # AR(1) coefficeint
        par.std_G = 0.00 # std.

        par.jump_i_shock = 0.00 # initial jump
        par.rho_i_shock = 0.00 # AR(1) coefficeint
        par.std_i_shock = 0.00 # std.

        # i. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 500#100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        par.py_hh = False # use python in household problem
        par.py_blocks = False # use python in blocks

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss     
    find_ss_new = steady_state.find_ss_new


    def calc_additional(self):
        path = self.path
        par = self.par
        ss = self.ss

        # ------------ a. Aggregate variables ------------

        # i tradeable consumption share 
        path.T_share = path.CT / (path.CT + path.CNT) # share of tradeable consumption
        ss.T_share = ss.CT / (ss.CT + ss.CNT)



        #------------ b. Idiosyncratic variables ------------
    
        # i. inflation

         # base periode expenditure share on nontradables
        ct_exp_share = ss.ct/(ss.cnt + ss.ct) # prices are normalized to 1 so no need to multiply 

        # Step 1: Reshape arrays for broadcasting
        ct_exp_share = ct_exp_share[np.newaxis, :, :, :]         # (1, 2, 7, 500)
        PT = path.PT[:, 0].reshape(-1, 1, 1, 1)                      # (500, 1, 1, 1)
        PNT = path.PNT[:, 0].reshape(-1, 1, 1, 1)                    # (500, 1, 1, 1)

        # Step 2: Compute components
        term1 = (1 - (par.epsilon * ct_exp_share) / par.gamma) * PNT ** par.gamma
        term2 = ((par.epsilon * ct_exp_share) / par.gamma) * PT ** par.gamma

        # Step 3: Compute p_tilde and p
        p_tilde = (term1 + term2) ** (1 / par.gamma)
        path.q = p_tilde ** (par.gamma / par.epsilon) * PNT ** (1 - par.gamma / par.epsilon)


        # ii. MPC's 

        denom = (1 + par.rF_ss) * (par.a_grid[1:] - par.a_grid[:-1])  # shape (499,)

        # Step 2: List of consumption variables to compute MPC for
        variables = ['e', 'ct', 'cnt']

        # Step 3: Loop over variables and compute MPC with extrapolation
        for var in variables:
            cons = getattr(ss, var)  # e.g., model.ss.e, shape (2, 7, 500)

            # Compute finite differences over assets
            mpc = (cons[:, :, 1:] - cons[:, :, :-1]) / denom  # shape (2, 7, 499)

            # Extrapolate last point using linear extrapolation
            last_diff = mpc[:, :, -1] - mpc[:, :, -2]         # (2, 7)
            mpc_last = mpc[:, :, -1] + last_diff              # (2, 7)

            # Append extrapolated value
            mpc_full = np.concatenate([mpc, mpc_last[:, :, np.newaxis]], axis=2)  # shape (2, 7, 500)

            # Store result back to model.ss
            setattr(ss, f'MPC_{var}', mpc_full) 
        
        
        # iii. additional variables
        
        ss.omegaiT = ss.ct/(ss.ct+ss.cnt)
        ss.etaiT =  1- par.gamma- ss.omegaiT/(1-ss.omegaiT)   *(par.gamma-par.epsilon)# 1- par.gamma - ss.omegaiT/(1-ss.omegaiT)(par.gamma-par.gamma)



    def calc_MPCs(self):
        ss = self.ss
        par = self.par
        path = self.path
        denom = (1 + par.rF_ss) * (par.a_grid[1:] - par.a_grid[:-1])  # shape (499,)


            # Step 2: List of consumption variables to compute MPC for
        variables = ['e', 'ct', 'cnt']

        # Step 3: Loop over variables and compute MPC with extrapolation
        for var in variables:
            cons = getattr(ss, var)  # e.g., model.ss.e, shape (2, 7, 500)

            # Compute finite differences over assets
            mpc = (cons[:, :, 1:] - cons[:, :, :-1]) / denom  # shape (2, 7, 499)

            # Extrapolate last point using linear extrapolation
            last_diff = mpc[:, :, -1] - mpc[:, :, -2]         # (2, 7)
            mpc_last = mpc[:, :, -1] + last_diff              # (2, 7)

            # Append extrapolated value
            mpc_full = np.concatenate([mpc, mpc_last[:, :, np.newaxis]], axis=2)  # shape (2, 7, 500)

            # Store result back to model.ss
            setattr(ss, f'MPC_{var}', mpc_full) 


    
    def calc_additional_new(self):
        path = self.path
        par = self.par
        ss = self.ss
        ct_exp_share_ = ss.ct/(ss.cnt + ss.ct) # prices are normalized to 1 so no need to multiply 

        # Step 1: Reshape arrays for broadcasting
        ct_exp_share = ct_exp_share_[np.newaxis, :, :, :]         # (1, 2, 7, 500)
        PT = path.PT[:, 0].reshape(-1, 1, 1, 1)                      # (500, 1, 1, 1)
        PNT = path.PNT[:, 0].reshape(-1, 1, 1, 1)                    # (500, 1, 1, 1)


        # Step 3: Compute p_tilde and p
        if par.epsilon == 0.0 and par.gamma == 0.0:
            path.q = PT **  ct_exp_share * PNT ** (1-ct_exp_share )

        elif par.epsilon == 0.0 :
            path.q =  PNT * np.exp(ct_exp_share * ((PT / PNT) ** par.gamma) * (1 / par.gamma) * (((PT / PNT) ** par.gamma) - 1))

        else:
            path.q =PNT *(1+ct_exp_share*(par.epsilon/par.gamma)*((PT/PNT)**par.gamma-1))**(1/par.epsilon)
            # Step 2: Compute components
            # term1 = (1 - (par.epsilon * ct_exp_share) / par.gamma) * PNT ** par.gamma
            # term2 = ((par.epsilon * ct_exp_share) / par.gamma) * PT ** par.gamma
            # p_tilde = (term1 + term2) ** (1 / par.gamma)
            # path.q = p_tilde ** (par.gamma / par.epsilon) * PNT ** (1 - par.gamma / par.epsilon)

        path.x = (path.e*PNT)/path.q

        # c. aggregate
        for outputname in ['x', 'q']:

            Outputname_hh = f'{outputname.upper()}_hh'
            pathvalue = path.__dict__[Outputname_hh]

            pol = path.__dict__[outputname]
            pathvalue[:,0] = np.sum(pol*path.D,axis=tuple(range(1,pol.ndim)))


        # Correlation between e and ct_exp_share
        A = ss.e.ravel()
        B = ct_exp_share.ravel()
        w = ss.D.ravel()
        # print(np.max(A))

        # Normalize weights to sum to 1
        w /= np.sum(w)

        # Compute weighted means
        mean_A = np.sum(w * A)
        mean_B = np.sum(w * B)

        # Compute weighted covariance
        cov_AB = np.sum(w * (A - mean_A) * (B - mean_B))


        std_A = np.sqrt(np.sum(w * (A - mean_A) ** 2))
        std_B = np.sqrt(np.sum(w * (B - mean_B) ** 2))

        eps = 1e-10  # numerical tolerance

        if std_A < eps or std_B < eps:
            corr_AB = 0.0
        else:
            corr_AB = cov_AB / (std_A * std_B)

        self.cov_e_omegaT = cov_AB
        self.cor_e_omegaT = corr_AB


        # ii. MPC's 

        denom = (1 + par.rF_ss) * (par.a_grid[1:] - par.a_grid[:-1])  # shape (499,)

        # Step 2: List of consumption variables to compute MPC for
        variables = ['e', 'ct', 'cnt']

        # Step 3: Loop over variables and compute MPC with extrapolation
        for var in variables:
            cons = getattr(ss, var)  # e.g., model.ss.e, shape (2, 7, 500)

            # Compute finite differences over assets
            mpc = (cons[:, :, 1:] - cons[:, :, :-1]) / denom  # shape (2, 7, 499)

            # Extrapolate last point using linear extrapolation
            last_diff = mpc[:, :, -1] - mpc[:, :, -2]         # (2, 7)
            mpc_last = mpc[:, :, -1] + last_diff              # (2, 7)

            # Append extrapolated value
            mpc_full = np.concatenate([mpc, mpc_last[:, :, np.newaxis]], axis=2)  # shape (2, 7, 500)

            # Store result back to model.ss
            setattr(ss, f'MPC_{var}', mpc_full) 
        
        
        # iii. additional variables
        
        ss.omegaiT = ss.ct/(ss.ct+ss.cnt)
        ss.etaiT =  1- par.gamma- ss.omegaiT/(1-ss.omegaiT)   *(par.gamma-par.epsilon)# 1- par.gamma - ss.omegaiT/(1-ss.omegaiT)(par.gamma-par.gamma)
