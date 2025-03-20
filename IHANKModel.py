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
        self.outputs_hh = ['a','c','uc_TH','uc_NT', 'e', 'cnt', 'ct', 'u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZTH','ZNT','M_s','rF','PF_s','G','i_shock', 'PE_s', 'epsilon_i'] # exogenous inputs
        self.unknowns = ['CB','NNT','NTH','piWTH','piWNT'] # endogenous inputs
        self.targets = ['NKWCT_res','NKWCNT_res','clearing_YTH','clearing_YNT','UIP_res'] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.mon_pol',
            'blocks.production',
            'blocks.prices',
            # 'blocks.inflation', 
            'blocks.central_bank',
            'blocks.government',
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

        # New 
        par.epsilon_ = 0.18 # controls the degree of non-homotheticity 
        par.gamma_ = 0.29 # controls the non-constant elicticity of substitution  between tradable and non-tradable goods
        par.nu_ = 0.475 # Scalling parameter
        par.omega_T_ = np.nan # agregate expenditure share on tradables in steady state
        par.run_u = False
        par.mon_policy = 'real'
        par.pf_fixed = True
        par.etaE = 0.1 # elasticity of substitution between tradable goods and energy 
        par.alphaE = 0.05 # share of energy in tradable + energy consumption
        par.eta_T_RA = np.nan
        par.phi_inflation = 1.0
        par.sNT = np.nan # share of Workers in the non-tradable sector - determined in ss

        # a. discrete states
        par.Nfix = 2 # number of sectors sectors
        par.Nz = 7 # idiosyncratic productivity
        par.sT = np.nan # share of workers in tradeable sector

        # b. preferences
        par.beta = 0.985 #0.975 # discount factor
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution

        par.alphaT = np.nan # share of tradeable goods in home consumption (determined in ss)
        par.etaT = 0.5 #2.0 # elasticity of substitution between tradeable and non-tradeable goods
        
        par.alphaF = 1/3 # share of foreign goods in home tradeable consumption
        par.etaF = 2.0 #*** # elasticity of substitution between home and foreign tradeable goods
          
        par.varphiTH = np.nan # disutility of labor in tradeable sector (determined in s)
        par.varphiNT = np.nan # disutility of labor in non-tradeable sector (determined in s)
        par.nu = 2.0 # Frisch elasticity of labor supply
              
        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of psi
        
        # d. price setting
        par.kappa = 0.1 # slope of wage Phillips curve
        par.muw = 1.2 # wage mark-up       
 
        # e. foreign Economy
        par.rF_ss = 0.005 # exogenous foreign interest rate
        par.eta_s =  2.0 # # Armington elasticity of foreign demand
        par.M_s_ss = np.nan # size of foreign market (determined in ss)

        # f. government
        par.tau_ss = 0.30 # tax rate on labor income
        par.omega = 0.10 # tax sensitivity to debt

        # central bank
        par.float = True # float or fix exchange rate
        par.phi = 1.5 # Taylor rule coefficient on inflation

        # g. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
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
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
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