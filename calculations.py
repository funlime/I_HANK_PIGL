
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def difine_shocks(model, scale=0.03, rho=0.8, plot_shocks=False):
        # Inflation shock 
    T_max = model.par.T//2 


    # Price shock 
    dPF_s = np.zeros(model.par.T)
    dPE_s = np.zeros(model.par.T)
    pi = np.zeros(model.par.T) # 
    PF_calc = np.zeros(model.par.T)
    pi_plus = np.zeros(model.par.T)
    iF_s = np.zeros(model.par.T)
    rF = np.zeros(model.par.T)
    drF = np.zeros(model.par.T)
    di_shock = np.zeros(model.par.T)
    depsilon_i = np.zeros(model.par.T)
    PE_calc = np.zeros(model.par.T)

    # Forigne inflation
    # inflation from period t to t+1
    for t in range(T_max):
        pi[t] = scale*rho**t


    # Prices
    for t in range(model.par.T):
        if t==0:
            PF_calc[t] = model.ss.PF_s*(1+pi[t])
        else:
            PF_calc[t] = PF_calc[t-1]*(1+pi[t])

    dPF_s = PF_calc- model.ss.PF_s
    dPE_s = dPF_s


    # Interst rate following taylor rule 
    for t in range(model.par.T):

        if t < model.par.T-1:
            pi_plus[t] = pi[t+1]

            iF_s[t] = (1+model.ss.i) * ((1+pi_plus[t])/(1+model.ss.pi))**model.par.phi -1
            
            rF[t] = (iF_s[t] + 1)/(1+pi_plus[t])-1

            drF[t] = model.ss.rF - rF[t]

        else:
            iF_s[t] = model.ss.i
            rF[t] = model.ss.rF



    if plot_shocks:
        fig = plt.figure(figsize=(20,7))

        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(pi[:T_max], label='$\pi_{t+1}$')
        ax0.set_title('Inflation')

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(PF_calc[:T_max], label='$Price$')
        ax1.set_title('Price')

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(rF)
        ax2.set_title('Forigne real interest rate')


    # Domestic interest rate shock      
    for t in range(T_max):
        di_shock[t] = -scale*rho**t
    
    # 3. Forigne energy price shock
        
    for t in range(T_max):
        pi[t] = scale*rho**t

    # Prices
    for t in range(model.par.T):
        if t==0:
            PE_calc[t] = model.ss.PE_s*(1+pi[t])
        else:
            PE_calc[t] = PE_calc[t-1]*(1+pi[t])

    dPE_s = PE_calc- model.ss.PE_s

    dPE_s[100:model.par.T] = 0 #***


    # Energy price shock - increasing then decreasing
    # for t in range(50):
    #     if t==0:
    #         PE_calc[t] = 0.0
    #     else:
    #         PE_calc[t] = -0.000012 * (t - 50)**2  + 0.000012 * (50)**2

    # for t in range(50, 300):
    #     PE_calc[t] = 0.03

    # for t in range(300, model.par.T):
    #     i = t-250
    #     PE_calc[t] = -0.000012 * (i - 50)**2  + 0.000012 * (50)**2

    # PE_calc = np.fmax(PE_calc, 0.0)

    # dPE_s = PE_calc


    # Energy price shock - AR(1) shock 
    dPE_s[:] = 0
    # inflation from period t to t+1
    for t in range(T_max):
        dPE_s[t] = scale*rho**t



    # Defining shocks 
    shock_forigne_interest = { 'drF':drF}
    shock_PF_s = {'dPF_s':dPF_s}
    shock_PF_s_taylor = {'dPF_s':dPF_s, 'drF':drF}
    shock_PE_s = {'dPE_s':dPE_s}
    shock_PE_PF = {'dPE_s':dPE_s, 'dPF_s':dPF_s}
    shock_PE_PF_taylor = {'dPE_s':dPE_s, 'dPF_s':dPF_s, 'drF':drF}
    # shock_i = {'depsilon_i':depsilon_i}
    shock_i = {'di_shock':di_shock}
    shock_PE_i = {'dPE_s':dPE_s, 'di_shock':di_shock}

    return [shock_PE_i, shock_PE_s, shock_forigne_interest, shock_PF_s, shock_PF_s_taylor, shock_PE_PF, shock_PE_PF_taylor, shock_i]



def test_model_properties(models):
    # Define the conditions to be tested
    conditions = [
        ('PF increase', lambda model: model.path.PF[0] > model.ss.PF),
        ('PT increases', lambda model: model.path.PT[0] > model.ss.PT),
        ('P increases', lambda model: model.path.P[0] > model.ss.P),
        ('CTF decreases', lambda model: model.path.CTF[0] < model.ss.CTF),
        ('i increases', lambda model: model.path.i[0] > model.ss.i),
        ('U decreases', lambda model: model.path.U_hh[0] < model.ss.U_hh)
    ]

    for model in models:
        print(f'\nTesting the aggregate properties of {model.name}')
        
        # Initialize lists to store fulfilled and unfulfilled conditions
        fulfilled = []
        unfulfilled = []

        # Test each condition
        for condition_name, condition_func in conditions:
            if condition_func(model):
                fulfilled.append(condition_name)
            else:
                unfulfilled.append(condition_name)

        # Print the results
        print('Fulfilled conditions:')
        for condition in fulfilled:
            print(f'  {condition}')

        print('Unfulfilled conditions:')
        for condition in unfulfilled:
            print(f'  {condition}')

