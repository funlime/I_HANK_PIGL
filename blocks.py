import numpy as np
import numba as nb
from GEModelTools import prev, next, lag, lead, isclose
from GEModelTools import lag, lead

##############
## Helpers ##
##############

@nb.njit
def price_index(P1,P2,eta,alpha):
    return (alpha*P1**(1-eta)+(1-alpha)*P2**(1-eta))**(1/(1-eta))

@nb.njit
def inflation_from_price(P,inival):

    P_lag = lag(inival,P) 
    pi = P/P_lag - 1

    return pi

@nb.njit
def price_from_inflation(P,pi,T,iniP):

    for t in range(T):
        if t == 0:
            P[t] = iniP*(1+pi[t]) 
        else:
            P[t] = P[t-1]*(1+pi[t]) 
   
############
## Blocks ##
############

@nb.njit
def mon_pol(par,ini,ss,E,CB):

    if par.float == True:
        E[:] = CB 
    else:
        E[:] = ss.E
    
@nb.njit
def production(par,ini,ss,
               ZTH,ZNT,NTH,NNT,piWTH,piWNT,
               YTH,YNT,WTH,WNT,PTH,PNT):
    
    # a. production
    YTH[:] = ZTH*NTH
    YNT[:] = ZNT*NNT
    
    # b. wages
    price_from_inflation(WTH,piWTH,par.T,ss.WTH) # piWTH, piWNT are unknowns
    price_from_inflation(WNT,piWNT,par.T,ss.WNT)

    # c. price = marginal cost
    PTH[:] = WTH/ZTH
    PNT[:] = WNT/ZNT

@nb.njit
def prices(par,ini,ss,
           PF_s,E,PTH,PNT,
           PF,PTH_s,PT,P, pi_F_s,pi_F,pi_NT,pi_TH,pi_T,pi,pi_TH_s, PE, PTHF, PE_s):
    
    # a. convert curency

    PF[:] = PF_s*E
    PTH_s[:] = PTH/E

    PE[:] = PE_s*E



    # for t in range(par.T): 
    #     # if np.isnan(PTHF[t]):
    #     #     print('PTHF is nan')
    #     if np.isnan(PTH[t]):
    #         print('PTH is nan')
    #     if np.isnan(PF[t]):
    #         print('PF is nan')

    # b. price indices

    PTHF[:] = price_index(PF,PTH,par.etaF,par.alphaF)

    # for t in range(par.T): 
    #     if np.isnan(PTHF[t]):
    #         print('PTHF is nan')
    #     if np.isnan(PTH[t]):
    #         print('PTH is nan')
    #     if np.isnan(PF[t]):
    #         print('PF is nan')

    # PT[:] = price_index(PTHF,PE,par.etaE,par.alphaE)

    # Testing PTHF and PE is a number


    PT[:] = price_index(PE,PTHF,par.etaE,par.alphaE)

    # c. PIGL Cost of living index for representative agents  (not used - look at first)
    # p_tilde = ((1-(par.epsilon_*par.omega_T_)/par.gamma_)*PNT**par.gamma_ + ((par.epsilon_*par.omega_T_)/par.gamma_)*PT**par.gamma_)**(1/par.gamma_)
    
    # P[:] = p_tilde**(par.gamma_/par.epsilon_)*PNT**(1-par.gamma_/par.epsilon_)


    # CES price index using average tradable share and  elasticity of substitution of average houshold from ss
    P[:] =   price_index(PT,PTHF,par.eta_T_RA, par.omega_T_)



    # d. inflation rates
    pi_F_s[:] = inflation_from_price(PF_s,ini.PF_s)
    pi_F[:] = inflation_from_price(PF,ini.PF)
    pi_NT[:] = inflation_from_price(PNT,ini.PNT)
    pi_TH[:] = inflation_from_price(PTH,ini.PTH)
    pi_T[:] = inflation_from_price(PT,ini.PT)
    pi[:] = inflation_from_price(P,ini.P)
    pi_TH_s[:] = inflation_from_price(PTH_s,ini.PTH_s)



@nb.njit
def central_bank(par,ini,ss,pi,i, i_shock,CB, pi_NT, epsilon_i):

    # TBD: Add choice of which inflation to target
    # Agregate inflaiton, how to weight tradable and non tradable inflation 
    # print(i_shock)
    # 1. setting interest rate
    if par.float == True: # taylor rule
        pi_plus = lead(pi,ss.pi)
        pi_plus_NT = lead(pi_NT,ss.pi_NT)


        if par.mon_policy == 'real':
#  i[:] = ss.i + par.phi_inflation*pi_plus + i_shock  # Real rule
            i[:] = ss.i + pi_plus # Real rule

        if par.mon_policy == 'real_PNT':

            i[:] = ss.i + pi_plus_NT + i_shock
        
        if par.mon_policy == 'taylor_OLD':

            i[:] = (1+ss.i) * ((1+pi_plus)/(1+ss.pi))**par.phi -1 + i_shock


    else: # fixed exchange rate
        i[:] = CB

    # TBD Add an terest rate in terms of general price level



@nb.njit
def government(par,ini,ss,
               PNT,NTH,NNT,G,B,tau, WTH, WNT, i):

    for t in range(par.T): 
    
    # a. nominal interest on last period bonds and last period nominal bonds
        lag_i = prev(i,t,ini.i)  
        B_lag = prev(B,t,ini.B)  

    # b. government budget
        
        # o. nomnial tax base
        tax_base =  WTH[t]*NTH[t]+WNT[t]*NNT[t]  
 
        # oo. tax rates following tax rule 
        tau[t] = ss.tau + par.omega*(B_lag/PNT[t]-ss.B/PNT[t])/(ss.YTH+ss.YNT) 

        # ooo. current nominal bonds from governmetn budget constraint
        B[t] = (1+lag_i)*B_lag + PNT[t]*G[t]-tau[t]*tax_base


@nb.njit
def HH_pre(par,ini,ss,
           PNT, WTH, WNT, pi_NT, i, tau, inc_TH, inc_NT, ra, p, PT, NNT, NTH,n_NT,n_TH , pi): # CHange inc_TH/inc_NT to w tilde
    

    # Housholds inputs

    # a. after tax real wage in terms of non-tradable goods (Also calculated inside HH block for eisire decomposition)
    inc_NT[:] = (NNT*WNT*(1-tau))/PNT
    inc_TH[:] = (NTH*WTH*(1-tau))/PNT

    # b. labor supply Wrong but works kinda
    n_TH[:] = NTH/par.sT
    n_NT[:] = NNT/(1-par.sT)

    # b. labor supply correct but does not work.... ***
    # n_NT[:] = NNT/(1-par.sT)
    # n_TH[:] = NTH/par.sT

    # b. Manual 
    # n_NT[:] = NNT/(1-0.4995882080832822)
    # n_TH[:] = NTH/0.4995882080832822

    # c. relative prices
    p[:] = PT/PNT

    #d. Interest rate
    # o. last periods interest rate 
    lag_i = lag(ini.i,i)

    # oo. deflated with inflation in non-tradable prices to get interest on assets_tilde (nominal assets divided by PNT)
    ra[:] = (1+lag_i)/(1+pi_NT)-1 

    # ooo. Ex ante real interest rate, defalted with the general price level: 
    pi_plus = lead(pi,ss.pi)
    # r_real[:] = (1+i)/(1+pi_plus)-1 # Real interest rate in terms of general price level

@nb.njit
def HH_post(par,ini,ss,
                C_hh,PT,PNT,P,PTH,PF,M_s,PTH_s,PF_s,PTHF,
                CT,CNT,CTF,CTH,CTH_s,NTH, NNT, CT_hh, CNT_hh, PE,  E_hh, E, A, A_hh, EX,  CTHF,  CE):

    # a bit redundant to change names from _hh

    # a. home - tradeable vs. non-tradeable 
    CT[:] =   CT_hh 
    CNT[:] = CNT_hh 
    

    # b. Energy and non-energy tradable consumption
    CE[:] = par.alphaE*(PE/PT)**(-par.etaE)*CT
    CTHF[:] = (1-par.alphaE)*(PTHF/PT)**(-par.etaE)*CT

    #. c Home and foreign tradeable consumption
    CTF[:] = par.alphaF*(PF/PTHF)**(-par.etaF)*CTHF
    CTH[:] = (1-par.alphaF)*(PTH/PTHF)**(-par.etaF)*CTHF

    # CTHF[:] = CTHF_hh
    # CTF[:] = CTF_hh
    # CTH[:] = CTH_hh
    # CE[:] = CE_hh

    # c. Nominal expnediture  
    EX[:] = E_hh * PNT
    A[:] = A_hh * PNT

    # c. foreign - home tradeable
    if par.pf_fixed == True:
        CTH_s[:] = (PTH_s)**(-par.eta_s)*M_s # Price level abroud not equal to price of imported goods but fixed at 1. 
    
    if par.pf_fixed == False:
        CTH_s[:] = (PTH_s/PF_s)**(-par.eta_s)*M_s



@nb.njit
def NKWCs(par,ini,ss,
          piWTH,piWNT,NTH,NNT,WTH, WNT, wTH,wNT,tau,UC_TH_hh,UC_NT_hh,NKWCT_res,NKWCNT_res, PNT):

    # SHOULD IT BE REAL WAGE? vi ser på marginal nytte af nominal expenditure **

    # a. Real wage 
    wTH[:] = WTH/PNT
    wNT[:] = WNT/PNT

    # b. phillips curve tradeable
    piWTH_plus = lead(piWTH,ss.piWTH)

    LHS = piWTH  

    RHS = par.kappa*(par.varphiTH*(NTH/par.sT)**par.nu-1/par.muw*(1-tau)*wTH*UC_TH_hh) + par.beta*piWTH_plus        
    NKWCT_res[:] = LHS-RHS # Target

    # c. phillips curve non-tradeable
    piWNT_plus = lead(piWNT,ss.piWNT)

    LHS = piWNT
    RHS = par.kappa*(par.varphiNT*(NNT/(1-par.sT))**par.nu-1/par.muw*(1-tau)*wNT*UC_NT_hh) + par.beta*piWNT_plus
    
    NKWCNT_res[:] = LHS-RHS # Target

@nb.njit
def UIP(par,ini,ss,rF,UIP_res, pi_F_s, E, i,iF_s):

    # a. expected future exchange rate
    E_plus = lead(E,ss.E)

    pi_F_s_plus = lead(pi_F_s,ss.pi_F_s)

    # b. nominal interest rate in foreign country
    iF_s[:] = (1+rF)*(1+pi_F_s_plus) - 1 # Nominal interest rate in foreign country, following the definition of pi_F_s

    # c. UIP
    LHS = (1+i) # Domestic nominal interest rate
    RHS = (1+iF_s)*E_plus/E # 

    UIP_res[:] = LHS-RHS # Target

    # Real in terms of the general pris level 



@nb.njit
def market_clearing(par,ini,ss,
             YTH,CTH,CTH_s,YNT,CNT,G,
             clearing_YTH,clearing_YNT):
    
    clearing_YTH[:] = YTH-CTH-CTH_s # Target
    clearing_YNT[:] = YNT-CNT-G # Target

@nb.njit
def accounting(par,ini,ss,
               PTH,YTH,PNT,YNT,P,C_hh,G,A,B,ra,
               GDP,NX,CA,NFA,Walras, E, iF_s, i,EX, YH,W, WNT, WTH,w, NNT, NTH, N, INC, inc, inc_NT, inc_TH):
    

    # not in use     
    E_plus = lead(E,ss.E) # Expected furture nominal exchange rate 
    E_lag = lag(ini.E,E)
    iF_s_lag = lag(ini.iF_s,iF_s)

    # a. last periods nominal interest rate
    lag_i = lag(ini.i,i)

    # b. Nominal GDP
    GDP[:] = PTH*YTH+PNT*YNT 


    # c. Net exports
    NX[:] = GDP-EX-PNT*G # Total production (nominal) - houhsolds private expenditure - Government expenditure

    # d. NFA
    # o. Nominal foreign assets
    NFA[:] = A-B # Nominal houshold savings - nominal government bonds
    
    # oo. lagged NFA
    NFA_lag = lag(ini.NFA,NFA)
    
    # ooo. Current account
    # CA[:] = NX + lag_i * NFA_lag# Current acount net exports this period should 
    CA[:] = NX + iF_s_lag * NFA_lag*(E/E_lag)# Current acount net exports this period 
    
    # oooo. Walras law check
    Walras[:] = (NFA-NFA_lag) - CA

    # For easier look at the results
    YH[:] = YNT+YTH
    W[:] = (par.sT*WTH + (1-par.sT)*WNT)
    w[:] = W/P # Tjeck what P is 
    N[:] = NNT+NTH
    INC[:] = (inc_NT + inc_TH)*PNT
    inc[:] = INC/P