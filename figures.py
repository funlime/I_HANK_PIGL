import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from scipy.interpolate import CubicSpline
from scipy.stats import norm

import matplotlib.pyplot as plt   
from matplotlib.ticker import FormatStrFormatter

from seaborn import set_palette
from matplotlib import rc
plt.style.use('seaborn-v0_8-white')
set_palette("colorblind")
rc('font',**{'family':'serif','serif': ['Palatino']})
rc('text',usetex=True)


T_max = 20


abs_value = ['Walras']

pctp = ['iF_s','piF_s','piF',
        'pi','piNT','piH','ppi','r','i','r_ann','pi_ann',
        'Dompi','di','NFA','r_NFA','piH_ann','i_ann',
        'adjcost_T','adjcost_NT', 'tau', 'r_real', 'NX']


pathlabels = {

    'YH': 'Real GDP',
    'P': 'CPI, $P$',
    'W': 'Wages, $W$',
    'w': 'Real wages $w$',
    'r_real': 'Real initerest rate, $r$',
    'PE':'Price of energy $P_E$',
    'PT': '$P_T$', 
    'PNT': '$P_{NT}$',
    'p': '$P_T/P_{NT}$',
    'CNT_hh':'Cons. of non-tradeables ($C_{NT}$)',
    'CT_hh':'Cons. of tradeables ($C_{T}$)',
    'CNT':'Cons. of non-tradeables ($C_{NT}$)',
    'CT':'Cons. of tradeables ($C_{T}$)',
    'eps_beta':'Discount factor ($\\beta$)',
    'Exports':'Exports',
    'i_ann':'Nominal interest rate, annual ($i$)',
    'I':'Investment ($I$)',
    'iF_s':'Foreign interest rate ($i^*$)',
    'Imports':'Imports',
    'labor_comp':'Wage Income ($wN$)',
    'N':'Employment ($N$)',
    'NX':'Net exports',
    'pi_ann':'Inflation, annual ($\pi$)',
    'pi':'Inflation ($\pi$)',
    'piF_s':'Foreign inflation ($\pi^*_F$)',
    'piH_ann':'$P_H$ Inflation, annual ($\pi_H$)',
    'ppi':'PPI ($\pi^{PP}$)',
    'PT_PNT':r'$P_{T}/P_{NT}$',
    'Q':'Real exchange rate ($Q$)',
    'r_ann':'Real interest rate, annual ($r$)',
    'r':'Real interest rate ($r$)',
    'rel_C_hh':r'Sectoral consumption, $C_T^{hh} / C_{NT}^{hh}$',
    'rel_wn_hh':r'Sectoral income, $w_T N_T / (w_{NT}N_{NT}$)',
    'tau':'Tax rate ($\\tau$)',
    'ToT':'Terms of Trade',
    'Y_s':'Foreign output ($Y^*$)',
    'Y':'GDP ($Y$)',
    'YNT':'Non-Tradeable production ($Y_{NT}$)',
    'YTH':'Tradeable production ($Y_{TH}$)',
    # 'w' : 'Real wage rate ($w$)',
    'E_hh': 'Expenditure, EX', 
    'CTH_s': 'Foreign consumption ($C_{TH}$)',
}


hh_labels = {
    'e': 'Expenditure',
    'ct': 'Consumption, Tradeables',
    'cnt': 'Consumption Non-tradeables'
    }


paths_defaults = {

    'standard':
        ['Y','C','I','Exports','pi_ann',
         'YNT','CNT','N','Imports','r_ann',
         'YT','CT','labor_comp','ToT','Q'],

    'standard_vs_data':
        ['Y','C','labor_comp','Exports','pi_ann',
         'YNT','CNT','N','Imports','i_ann',
         'YT','CT','w','Q','r_ann'],

    'standard_vs_data_w_capital':
        ['Y','C','I','Exports','pi_ann',
         'YNT','CNT','N','Imports','i_ann',
         'YT','CT','labor_comp','Q','r_ann']
}

scaleval=1.0

#-------------------------------------------------------------------------------------------------------------
#                       Figures
#-------------------------------------------------------------------------------------------------------------



def plot_policy(model, varnames, ncol=3):
    """Plot the policy functions of the model for each variable in varnames list."""
    ss = model.ss
    par = model.par
    path = model.path

    i_fix = 0
    a_max = 100
    n_vars = len(varnames)
    
    # Calculate the number of rows needed based on ncol and n_vars
    nrow = int(np.ceil(n_vars / ncol))

    # Set the figure and subplots
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow), dpi=100)

    # Flatten axes for easy iteration if there are multiple rows
    axes = axes.flatten() if n_vars > 1 else [axes]

    # Add an overall title/heading
    fig.suptitle(f'Policy function model: {model.name}', fontsize=16)

    # Loop over each variable in varnames and corresponding subplot axis
    for idx, varname in enumerate(varnames):

        I = par.a_grid < a_max
        ax = axes[idx]
        # ax.set_title(f'{varname}')
        ax.set_title(f'{hh_labels[varname]}')

        # Plot policy functions for different states (i_z)
        for i_z in [0, par.Nz // 2, par.Nz - 1]:
            ax.plot(par.a_grid[I], ss.__dict__[varname][i_fix, i_z, I], label=f'z = {i_z}')

        ax.set_xlabel('savings, $a_{t-1}$')
        ax.set_ylabel(f'{varname}')

    ax.legend(frameon=True )
    # Remove any unused subplots if n_vars is not equal to nrow * ncol
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_cum(models, varnames, ncols= 3, xlim= [], print_gini = False, title= None): 
    

    """Plot the cumulative distribution function of a variable in the model
    Args:
    var (str): Variable name
    models: List of models to plot (eg. varying G or varying TFP)
    xlim (list): x-axis limits
    Returns:
    CDF plot and Gini coefficient
    """

    num = len(varnames)
    nrows = num//ncols+1
    if num%ncols == 0: nrows -= 1
    
    fig = plt.figure(figsize=(6*ncols,4*nrows),dpi=100)
    if title != None:
        fig.suptitle(title, fontsize=16)

    for i,varname in enumerate(varnames):
        var = varname
        
        ax = fig.add_subplot(nrows,ncols,i+1)
        title = varname
        ax.set_title(f'{varname}')
        # ax.set_title(f'{hh_labels[title]}',fontsize=14)
        # ax.set_title(f'{hh_labels[var]}')

    #var = 'a'
    # Flattening  data og weits

        for model in models:
            try:
                model = model
                var_ = model.ss.__dict__[var][:,:,:].flatten()
                weight = model.ss.D[:,:,:].flatten()

                # Sorting data and weits
                sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
                sorted_var = np.array(sorted_var)
                sorted_weights = np.array(sorted_weights)

                # Calculating the cumulative sum of sorted weights
                cumulative = np.cumsum(sorted_weights)

                ax.plot(sorted_var, cumulative, label=model.name ) #, color = model.c)
                ax.set_xscale('symlog')
                ax.set_xlabel(r'Begining of period assets, $a_{t-1}$')
                ax.set_ylabel('CDF')


                # # Normalized cumulative weights (needed for Lorenz curve)
                # normalized_cumulative_weights = cumulative / cumulative[-1]

                # # Normalized weighted cumulative values
                # normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

                # # Gini calculation
                # area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
                # weighted_gini = 1 - 2 * area_under_lorenz_curve

                # if print_gini:
                #     print(f'{model.name} {var}')
                #     print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')
            except:
                print(f'Could not plot {var} for {model.name}')


        if xlim != []:
            ax.set_xlim(xlim)

        # ax.set_xlabel(f'{var}')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #ax.set_xscale('symlog')

        # ax.set_ylabel('Cumulative Probability')
        # ax.set_title('Cumulative Distribution Function (CDF) of ' + var)
    ax.grid(True)
    ax.legend(loc= 'lower right')
    plt.tight_layout()

    return fig




# Plot shock
def plot_PE_s(model):
    PE_s =   (model.path.PE_s / model.ss.PE_s-1)*100

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(PE_s, label='PE_s')
    # ax.set_title(f'Golabal energy ($P_E^*$)')
    ax.set_title(r'Global energy price ($P_{E}^*$)', pad=10)
    ax.set_xlim(0, 20)
    ax.set_xlabel('Quarters')
    ax.set_ylabel(r'\% diff. to s.s.')
    plt.tight_layout()

    return fig

# Ploting jacobians wrt ptilde



def plot_jac_p(models):

   fig = plt.figure(figsize=(10, 10))
   # tittle
   fig.suptitle(r'Jacobians wrt. $\tilde p$')

   ax = fig.add_subplot(2,2,1)

   # Add right-hand side label to top-right plot
   ax.text(-0.35, 0.5, 'Tradeable consumption', va='center', rotation=90,
         transform=ax.transAxes, fontsize=14)

   # add a label on the rhs "Tradable C"
   ax.set_title(r'Non-homothetic')
   for s in [0, 50, 150, 250]:
      model = models[0]
      jac_hh_var = model.jac_hh[('CT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')
   ax.set_ylim(-0.45, 0.01)


   ax = fig.add_subplot(2,2,2)
   ax.set_title(r'Homothetic')
   for s in [0, 50, 150, 250]:
      model = models[1]
      jac_hh_var = model.jac_hh[('CT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')
   ax.set_ylim(-0.45, 0.01)


   # add a label on the rhs "Non-Tradable C"
   ax = fig.add_subplot(2,2,3)
   ax.text(-0.35, 0.5, 'Non-Tradeable Consumption', va='center', rotation=90,
        transform=ax.transAxes, fontsize=14)
   
   ax.set_title(r'Non-homothetic')
   model = models[0]
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')
   ax.set_ylim(-0.175, 0.01)

   ax = fig.add_subplot(2,2,4)

   ax.set_title(r'Homothetic')
   model = models[1]
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')
   ax.set_ylim(-0.175, 0.01)
   # legende outside box
   plt.legend(loc='lower right')
   # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   plt.tight_layout()

   return fig


def compare_IRFs_(models_list, ddd):
    
    labels = []
    for i in models_list:
        labels.append(i.name)
    print(labels)
    if ddd.filename == None:
        models_list[0].compare_IRFs(models=models_list, labels=labels, varnames=ddd.varnames,  T_max=ddd.T_max, ncols=ddd.ncols, lvl_value=ddd.lvl_value, do_shocks=ddd.do_shocks, do_targets=ddd.do_targets)
    else:
        models_list[0].compare_IRFs(models=models_list, labels=labels, varnames=ddd.varnames,  T_max=ddd.T_max, ncols=ddd.ncols, lvl_value=ddd.lvl_value, do_shocks=ddd.do_shocks, do_targets=ddd.do_targets, filename= ddd.filname)



def plot_dec(model):

    fig = plt.figure(figsize=(3*6,4),dpi=100)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    ax1.set_title(f"{pathlabels['E_hh']}")
    ax2.set_title(f"{pathlabels['CT_hh']}")
    ax3.set_title(f"{pathlabels['CNT_hh']}")
    fig.suptitle(f'{model.name}', fontsize=16)

    i_color = 0
    inputs_list = [ 'all', ['p'], ['inc_NT', 'inc_TH'], ['ra']]

    # for use_inputs in [[x] for x in model.inputs_hh]:
    for use_inputs in inputs_list:

        # a. compute
        path_alt = model.decompose_hh_path(do_print=False, use_inputs=use_inputs)

        # b. plot
        if use_inputs is None:
            label = 'No inputs'
            ls = '--'
            color = 'black' 
        elif use_inputs == 'all':
            label = 'All inputs'
            ls = '-'
            color = 'black'
        else:
            label = f'Effect from {use_inputs}'
            # label = f'Only effect from {use_inputs[0]}'
            ls = '-'
            color = f'C{i_color}' # use color index directly
            i_color += 1
        

        # ax_C_NT.plot((path_alt.U_hh[:T_max] - model.ss.U_hh ) , ls=ls, color=color, label=label)
        ax1.plot((path_alt.E_hh[:T_max]*model.path.PNT[:T_max] - model.ss.E_hh ) , ls=ls, color=color, label=label)
        ax2.plot((path_alt.CT_hh[:T_max] - model.ss.CT_hh ) , ls=ls, color=color, label=label)
        ax3.plot((path_alt.CNT_hh[:T_max] - model.ss.CNT_hh ) , ls=ls, color=color, label=label)



    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel('diff to s.s. abs')
        lgd = ax3.legend(frameon=True, ncol=1, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout() 

    return fig

def _plot_IRFs(ax,model,pathname,scale,lstyle,color,lwidth,label,T_max):

    global abs_value, pctp, pathlabels

    if pathname in pathlabels:
        pathlabel = pathlabels[pathname]
    else:
        pathlabel = pathname

    if scale:
        scaleval = getattr(model.par,'scale')
    else:
        scaleval = 1 

    # ssvalue and pathvalue
    ssvalue = getattr(model.ss,pathname)  
    pathvalue = getattr(model.path,pathname)
    dpathvalue = pathvalue - ssvalue

    T_max = np.fmin(pathvalue.size,T_max)
    
    # plot
    if pathname in abs_value:                     
    
        ax.plot(np.arange(T_max),(dpathvalue[:T_max])*scaleval,label=label,linestyle=lstyle,color=color,lw=lwidth)
        ax.set_ylabel('abs. diff. to s.s.')
        ax.set_title(pathlabel)
    
    elif pathname in pctp:
    
        ax.plot(np.arange(T_max),100*(dpathvalue[:T_max])*scaleval,label=label,linestyle=lstyle,color=color,lw=lwidth)
        ax.set_ylabel('\%-points diff. to s.s.')
        ax.set_title(pathlabel)
    
    # elif pathname == 'NX':   
        
    #     pathvalue_IM = getattr(model.path,'Imports')   
    #     pathvalue_EX = getattr(model.path,'Exports')  
    #     ssvalue_IM = getattr(model.ss,'Imports')
    #     ssvalue_EX = getattr(model.ss,'Exports')
    #     dIM = 100*(pathvalue_IM[:T_max]-ssvalue_IM)*scaleval / ssvalue_IM  
    #     dEX = 100*(pathvalue_EX[:T_max]-ssvalue_EX)*scaleval / ssvalue_EX      
    #     dNX = dEX-dIM  
    #     ax.plot(np.arange(T_max),dNX,label=label,linestyle=lstyle,color=color,lw=lwidth)
    #     ax.set_ylabel('\% diff. to s.s.')
    #     ax.set_title(pathlabel)

    else:

        if abs(ssvalue) > 0: 
        
            ax.plot(np.arange(T_max),((dpathvalue[:T_max])*scaleval/ssvalue)*100,label=label, linestyle=lstyle,color=color,lw=lwidth)
            ax.set_ylabel('\% diff. to s.s.')
            ax.set_title(pathlabel)
        
        else:
        
            ax.plot(np.arange(T_max),((dpathvalue[:T_max])*scaleval)*100,label=label, linestyle=lstyle, color=color,lw=lwidth)
            ax.set_ylabel('\% diff. to s.s.')
            ax.set_title(pathlabel)                           

def show_IRFs(models,paths,labels=None,
              T_max=20,scale=False,
              lwidth=1.3,lstyles=None,colors=None,
              maxcol=5,figsize=None,
              lfsize=12,legend_window=0,
              compare_LP=None,CI=False,do_stds=False,show=True):

    # a. models as list
    if type(models) is dict: models = [model for model in models.values()]
    model = models[0]
    par = model.par

    # b. inputs
    if paths is None: paths = paths_defaults['standard_vs_data']
    if type(paths) is str: paths = paths_defaults[paths]
    if labels is None: labels = [None]*len(models)
    assert len(labels) >= len(models), f'{len(labels) = } must be same as {len(models) = }'
    if lstyles is None: lstyles = ['-','--','-.',':',(0,(3, 1, 1, 1)),(0,(3, 5, 1, 5, 1, 5))]
    
    if T_max is None: T_max = par.T   
    T_max_LP = 17

    # c. figure
    num = len(paths)
    nrows = num//maxcol+1
    ncols = np.fmin(num,maxcol)
    if num%maxcol == 0: nrows -= 1 

#***** FiGURE SIZE
    if figsize is None:
        #fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
        x_size, y_size = 13.5/maxcol*ncols, 10/maxcol*nrows
        fig = plt.figure(figsize=(x_size, y_size), dpi=100)
    else:
        fig = plt.figure(figsize=(figsize[0]*ncols,figsize[1]*nrows))

    for i,pathname in enumerate(paths):

        ax = fig.add_subplot(nrows,ncols,i+1)

        # axis
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if T_max < 100:
            ax.set_xticks(np.arange(0,T_max,4))
        
        # models
        for j, model_ in enumerate(models):  
            
            if lstyles is None:
                lstyle = '-' 
            else:
                lstyle = lstyles[j]

            if not colors is None:
                color = colors[j]
            else:
                color = f'C{j}'

            label = labels[j]

            if j == 0: ax.plot(np.arange(T_max),np.zeros(T_max),'-', color='black')
            _plot_IRFs(ax,model_,pathname,scale,lstyle,color,lwidth,label,T_max)

            ax.set_xlim([0,T_max-1])
            if i >= ncols*(nrows-1): ax.set_xlabel('Quarters')
            
        if compare_LP is not None:

            try:

                ax.plot(np.arange(T_max_LP),compare_LP['IRF'][pathname][:T_max_LP]*100,label='LP',
                        linestyle='--',color='black',lw=lwidth)
                
                if do_stds:

                    if CI:

                        prob=0.05
                        sign = norm.ppf(1-prob/2)
                        SE = compare_LP['SE'][pathname]
                        LO = compare_LP['IRF'][pathname] - SE
                        HI = compare_LP['IRF'][pathname] + SE
                        ax.fill_between(np.arange(T_max_LP), LO[:T_max_LP], HI[:T_max_LP], alpha=0.15, color='C0')
                        LO = compare_LP['IRF'][pathname] - sign*SE
                        HI = compare_LP['IRF'][pathname] + sign*SE
                        ax.fill_between(np.arange(T_max_LP), LO[:T_max_LP], HI[:T_max_LP], alpha=0.08, color='C0')
                
                else:

                    if CI:

                        LO = compare_LP['IRF'][pathname] + compare_LP['LO'][pathname][:,0]*100  
                        HI = compare_LP['IRF'][pathname] + compare_LP['HI'][pathname][:,0]*100  
                        ax.fill_between(np.arange(T_max_LP), LO[:T_max_LP], HI[:T_max_LP], alpha=0.25, color='C0')
                        LO = compare_LP['IRF'][pathname] + compare_LP['LO'][pathname][:,1]*100  
                        HI = compare_LP['IRF'][pathname] + compare_LP['HI'][pathname][:,1]*100  
                        ax.fill_between(np.arange(T_max_LP), LO[:T_max_LP], HI[:T_max_LP], alpha=0.15, color='C0')
            
            except:
                
                pass

        if len(models) > 1:
            if i == legend_window: ax.legend(frameon=True)

    #fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.4)
    if show : plt.show()

    return fig



def show_price_IRFs(model):

    ncols = 3
    nrows = 1
    T_max = 17
    linewidth= 2.5 

    fig = plt.figure(figsize=(4.3*ncols/1.1,3.6*nrows/1.2),dpi=100)
    fig.suptitle(f'{model.name},  Price Response', fontsize=20)

    # Tradable and  non-tradable
    ax = fig.add_subplot(nrows,ncols,1)    
    # ax.plot((model.path.p-model.ss.p),label='$p$', linewidth=linewidth)
    ax.plot((model.path.P-model.ss.P),label='$P$', linewidth=linewidth)
    ax.plot((model.path.PT-model.ss.PT),ls='--',label='$P_T$', linewidth=linewidth)
    ax.plot((model.path.PNT-model.ss.PNT),ls=':',label='$P_{NT}$', linewidth=linewidth)
    ax.set_xlim([0,T_max])
    ax.legend()
    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Tradeable vs. non-tradeables')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))

    # Energy and non energy tradable 
    ax = fig.add_subplot(nrows,ncols,2)    
    ax.plot((model.path.PT-model.ss.PT),label='$P_T$', linewidth=linewidth)
    ax.plot((model.path.PTHF-model.ss.PTHF),ls='--',label='$P_{goods}$', linewidth=linewidth)
    ax.plot((model.path.PE-model.ss.PE),ls=':',label='$P_{Energy}$', linewidth=linewidth)
    ax.set_xlim([0,T_max])
    ax.legend()
    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Energy vs. Goods')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))

    # Forign and domestic tradable 
    ax = fig.add_subplot(nrows,ncols,3)    
    ax.plot((model.path.PTHF-model.ss.PTHF),label='$P_{goods}$', linewidth=linewidth)
    ax.plot((model.path.PF-model.ss.PF),ls='--',label='$P_F$', linewidth=linewidth)
    ax.plot((model.path.PTH-model.ss.PTH),ls=':',label='$P_{TH}$', linewidth=linewidth)
    ax.set_xlim([0,T_max])  
    ax.legend(loc='lower right')
    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Foreign to domestic price')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))


    fig.tight_layout()

    return fig


def show_c_IRFs(model):

    ncols = 3
    nrows = 1
    T_max = 17
    linewidth= 2.5 

    fig = plt.figure(figsize=(4.3*ncols/1.1,3.6*nrows/1.2),dpi=100)
    # Tradable and  non-tradable
    fig.suptitle(f'{model.name}, Consumption Response', fontsize=20)
    
    ax = fig.add_subplot(nrows,ncols,1)    
    # ax.plot((model.path.p-model.ss.p),label='$p$', linewidth=linewidth)
    ax.plot(((model.path.CT-model.ss.CT)/model.ss.CT),label='$C_T$', linewidth=linewidth)
    ax.plot(((model.path.CNT-model.ss.CNT)/model.ss.CNT),ls='--',label='$C_{NT}$', linewidth=linewidth)
    # ax.plot((model.path.PNT-model.ss.PNT),ls=':',label='$P_{NT}$', linewidth=linewidth)
    ax.set_xlim([0,T_max])
    ax.set_ylabel('\% diff. to s.s.')
    ax.legend()
    ax.set_title('Tradeable vs. non-tradeables')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))

    # Energy and non energy tradable 
    ax = fig.add_subplot(nrows,ncols,2)    
    ax.plot(((model.path.CE-model.ss.CE)/model.ss.CE),label='$C_E$', linewidth=linewidth)
    ax.plot(((model.path.CTHF-model.ss.CTHF)/model.ss.CTHF),ls='--',label='$C_{goods}$', linewidth=linewidth)
    # ax.plot((model.path.PE-model.ss.PE),ls=':',label='$P_{Energy}$', linewidth=linewidth)
    ax.set_ylabel('\% diff. to s.s.')
    ax.set_xlim([0,T_max])
    ax.legend()
    ax.set_title('Energy vs. Goods')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))

    # Forign and domestic tradable 
    ax = fig.add_subplot(nrows,ncols,3)    
    # ax.plot((model.path.PTHF-model.ss.PTHF),label='$P_{goods}$', linewidth=linewidth)
    ax.plot(((model.path.CTF-model.ss.CTF)/model.ss.CTF),ls='--',label='$C_F$', linewidth=linewidth)
    ax.plot(((model.path.CTH-model.ss.CTH)/model.ss.CTH),ls=':',label='$C_{TH}$', linewidth=linewidth)
    ax.set_ylabel('\% diff. to s.s.')
    ax.set_xlim([0,T_max])
    ax.legend(loc='lower right')
    ax.set_title('Forigne vs. Home tradable')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0,T_max,4))

    fig.tight_layout()

    return fig

# Ploting all jacs
def plot_jac(model):

   fig = plt.figure(figsize=(15, 15))
   # tittle
   fig.suptitle(model.name + ' Jacobians')

   ax = fig.add_subplot(3,3,3)
   ax.set_title(r'Expenditure wrt $\tilde p$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('E_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')

   ax = fig.add_subplot(3,3,1)
   ax.set_title(r'Consumption of tradables wrt $\tilde p$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')

   ax = fig.add_subplot(3,3,2)
   ax.set_title(r'Consumption of Non-tradables wrt $\tilde p$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')



   ax = fig.add_subplot(3,3,4)
   ax.set_title(r'Expenditure wrt $\tilde w_NT$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('E_hh', 'inc_TH')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')

   ax = fig.add_subplot(3,3,5)
   ax.set_title(r'Consumption of tradables wrt $w_NT$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CT_hh', 'inc_TH')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')

   ax = fig.add_subplot(3,3,6)
   ax.set_title(r'Consumption of Non-tradables wrt $w_NT$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'inc_TH')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')




   ax = fig.add_subplot(3,3,7)
   ax.set_title(r'Expenditure wrt $\tilde r$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('E_hh', 'ra')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')

   ax = fig.add_subplot(3,3,8)
   ax.set_title(r'Consumption of tradables wrt $\tilde r$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CT_hh', 'ra')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')

   ax = fig.add_subplot(3,3,9)
   ax.set_title(r'Consumption of Non-tradables wrt $\tilde r$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'ra')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')



   # legende outside box
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   plt.legend
   plt.tight_layout()

   return fig



def show_pc_IRFs(model):
    ncols = 3
    nrows = 2
    T_max = 17
    linewidth = 2.5

    fig = plt.figure(figsize=(4.3 * ncols / 1.05, 3.6 * nrows / 1.1), dpi=100)
    # fig.suptitle(f'{model.name}: Price and Cons. Responses', fontsize=22)

    # === Row 1: PRICE RESPONSES ===

    # (1,1) Tradable vs. Non-tradable Prices
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.plot((model.path.P - model.ss.P)*100, label='$P$', linewidth=linewidth)
    ax.plot((model.path.PT - model.ss.PT)*100, ls='--', label='$P_T$', linewidth=linewidth)
    ax.plot((model.path.PNT - model.ss.PNT)*100, ls=':', label='$P_{NT}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Tradabl and Non-tradable Prices')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend()

    # (1,2) Energy vs. Goods Prices
    ax = fig.add_subplot(nrows, ncols, 2)
    ax.plot((model.path.PT - model.ss.PT)*100, label='$P_T$', linewidth=linewidth)
    ax.plot((model.path.PTHF - model.ss.PTHF)*100, ls='--', label='$P_{Goods}$', linewidth=linewidth)
    ax.plot((model.path.PE - model.ss.PE)*100, ls=':', label='$P_{Energy}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Energy vs. Goods Prices')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend()

    # (1,3) Foreign vs. Domestic Tradables
    ax = fig.add_subplot(nrows, ncols, 3)
    ax.plot((model.path.PTHF - model.ss.PTHF)*100, label='$P_{Goods}$', linewidth=linewidth)
    ax.plot((model.path.PF - model.ss.PF)*100, ls='--', label='$P_F$', linewidth=linewidth)
    ax.plot((model.path.PTH - model.ss.PTH)*100, ls=':', label='$P_{TH}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Foreign vs. Domestic Tradables')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend(loc='lower right')

    # === Row 2: CONS. RESPONSES ===

    # (2,1) Tradable vs. Non-tradable Cons.
    ax = fig.add_subplot(nrows, ncols, 4)
    ax.plot(((model.path.CT - model.ss.CT) / model.ss.CT)*100, label='$C_T$', linewidth=linewidth)
    ax.plot(((model.path.CNT - model.ss.CNT) / model.ss.CNT)*100, ls='--', label='$C_{NT}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Tradable vs. Non-tradable Cons.')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend()

    # (2,2) Energy vs. Goods Cons.
    ax = fig.add_subplot(nrows, ncols, 5)
    ax.plot(((model.path.CE - model.ss.CE) / model.ss.CE)*100, label='$C_E$', linewidth=linewidth)
    ax.plot(((model.path.CTHF - model.ss.CTHF) / model.ss.CTHF)*100, ls='--', label='$C_{Goods}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Energy vs. Goods Cons.')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend()

    # (2,3) Foreign vs. Domestic Tradable Cons.
    ax = fig.add_subplot(nrows, ncols, 6)
    ax.plot(((model.path.CTF - model.ss.CTF) / model.ss.CTF)*100, ls='--', label='$C_F$', linewidth=linewidth)
    ax.plot(((model.path.CTH - model.ss.CTH) / model.ss.CTH)*100, ls=':', label='$C_{TH}$', linewidth=linewidth)
    ax.set_xlim([0, T_max])

    ax.set_ylabel('\% diff. to s.s.')
    ax.set_title('Foreign vs. Domestic Cons.')
    ax.set_xlabel('Quarters')
    ax.set_xticks(np.arange(0, T_max, 4))
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for suptitle

    return fig


def show_p_hh(model, linewidth =1.0, type = 0):

    ncols = 2
    nrows = 1
    T_max = 17


    fig = plt.figure(figsize=(4.3*ncols/1.1,3.6*nrows/1.2),dpi=100)
    # fig.suptitle(f'{model.name},  Individal Price indexes', fontsize=20)

    # period 0
    t = 0 
    ax = fig.add_subplot(nrows,ncols,1)    
    # For ever second income
    for inc in range(0, 7, 2):
        # For every period
        ax.plot((model.path.p[t,type,inc,:]-1)*100, label=f'z = {inc}', linewidth=linewidth)
    
    ax.set_title(f'Period {t}', fontsize=16)
    ax.set_xlabel(r' $a_{t-1}$', fontsize=16) # ****
    ax.set_ylabel('\% diff. to s.s.')
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    # ax.legend()

    # period 7 
    t = 20
    ax = fig.add_subplot(nrows,ncols,2)

    # For ever second income
    for inc in range(0, 7, 2):
        # For every period
        ax.plot((model.path.p[t,type,inc,:]-1)*100, label=f'z = {inc}', linewidth=linewidth)
    ax.set_title(f'Period {t}', fontsize=16)
    ax.set_xlabel(r' $a_{t-1}$', fontsize=16) # ****
    ax.set_ylabel('\% diff. to s.s.')
    # ax.set_ylim(-0.5, 0.5)

    fig.tight_layout()
    return fig
