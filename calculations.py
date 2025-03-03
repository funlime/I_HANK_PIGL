
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def difine_shocks(model, scale=0.03, rho=0.8, plot_shocks=False):
        # Inflation shock 
    T_max = model.par.T//2 


    # Price shock 
    dPF_s = np.zeros(model.par.T)
    pi = np.zeros(model.par.T) # 
    PF_calc = np.zeros(model.par.T)
    pi_plus = np.zeros(model.par.T)
    iF_s = np.zeros(model.par.T)
    rF = np.zeros(model.par.T)
    drF = np.zeros(model.par.T)

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

    # Defining shocks 
        
    shock_forigne_interest = { 'drF':drF}
    shock_PF_s = {'dPF_s':dPF_s}
    shock_PF_s_taylor = {'dPF_s':dPF_s, 'drF':drF}

    return [shock_forigne_interest, shock_PF_s, shock_PF_s_taylor]