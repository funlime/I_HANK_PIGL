import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



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
        ax.set_title(f'{varname}')

        # Plot policy functions for different states (i_z)
        for i_z in [0, par.Nz // 2, par.Nz - 1]:
            ax.plot(par.a_grid[I], ss.__dict__[varname][i_fix, i_z, I], label=f'i_z = {i_z}')

        ax.legend(frameon=True)
        ax.set_xlabel('savings, $a_{t-1}$')
        ax.set_ylabel(f'{varname}')

    # Remove any unused subplots if n_vars is not equal to nrow * ncol
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


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
        ax.set_title(title,fontsize=14)

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
        ax.legend(loc= 'lower right')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #ax.set_xscale('symlog')

        # ax.set_ylabel('Cumulative Probability')
        # ax.set_title('Cumulative Distribution Function (CDF) of ' + var)
        ax.grid(True)




# Plot shock
def plot_PE_s(model):
    PE_s = (model.path.PE_s / model.ss.PE_s-1)*100

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(PE_s, label='PE_s')
    ax.set_title(f'Golabal energy price shock $P_E^*$')
    ax.set_xlim(0, 40)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent increas')
    plt.tight_layout()
    return fig

# Ploting jacobians wrt ptilde
def plot_jac_p(model, title = None):

   fig = plt.figure(figsize=(15, 5))
   # tittle
   if title != None:
      fig.suptitle(title)

   ax = fig.add_subplot(1,2,1)
   ax.set_title(r'Consumption of tradables wrt $\tilde p$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_T$')

   ax = fig.add_subplot(1,2,2)
   ax.set_title(r'Consumption of Non-tradables wrt $\tilde p$')
   for s in [0, 50, 150, 250]:
      jac_hh_var = model.jac_hh[('CNT_hh', 'p')]
      ax.plot(np.arange(model.par.T), jac_hh_var[:, s],  label=f'shock at {s}')
   ax.set_ylabel(r'$dC_{NT}$')
   ax.set_xlabel('Quarters')

   # legende outside box
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   plt.tight_layout()

   return fig


def comp(model_baseline, model_alt_list, shock, paths_, lvl_path):
    labels = [model_baseline.name] + [model_alt.name for model_alt in model_alt_list]
    for model in model_alt_list:
        
        model.find_ss()
        model.compute_jacs()
        model.find_transition_path(shocks=shock, do_end_check=False)
    
    model_baseline.compare_IRFs(models=[model_baseline] + model_alt_list, labels=labels, varnames=paths_,  T_max=50, ncols=3, lvl_value=lvl_path, do_shocks=False, do_targets=False)
