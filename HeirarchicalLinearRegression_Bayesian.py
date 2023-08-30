#------------------------------------------------------------------------------------------------------------------------
#Author: Brandon S Coventry
# Date: 08/23/2022
# Purpose: This program is a modified port of my previous heirarchical linear regression programs, updated for pymc (v4)
# Revision History: Based on Code I used for my dissertation. This is the release version
# Dependencies: PyMC as well as all PyMC dependencies.
# References: Gelman et al, 2021: http://www.stat.columbia.edu/~gelman/book/BDA3.pdf
#             Kruske Doing Bayesian Data Analysis Text
#              Betancourt & Girolami (2013) https://arxiv.org/pdf/1312.0906.pdf
#------------------------------------------------------------------------------------------------------------------------
"""
To begin, let's import all of our dependencies, including our data and python packages
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import aesara
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle # python3
import seaborn as sns
if __name__ == '__main__':                                            #This statement is to allow for parallel sampling in windows. Linux distributions don't require this.
    print(f"Running on PyMC3 v{pm.__version__}")
    color = '#87ceeb'
    az.style.use("arviz-darkgrid")

    """
    Here we will load data in, and do necessary extraction. For the moment, we are interested in purely excitatory responses on pulse trains
    """
    data = pd.read_csv("C:\CodeRepos\Statistics\INS_statTable_MKIV.csv")           #Use pandas to read in data

    data1 = data.loc[data['Is_Responsive'] == 1]                    #Responses driven by INS
    data1.reset_index(drop=True, inplace = True)
    data2 = data1.loc[data1['Number_of_Pulses']>1]                  #Pulse trains
    data2.reset_index(drop=True, inplace = True)
    data3 = data2.loc[np.isnan(data2['Max_BARS_Z'])==0]             #Take care of any nans
    data3.reset_index(drop=True, inplace = True)
    data4 = data3.loc[data1['Max_BARS_Z']>0]                        #Purely excitatory, which is the vast majority of responses
    data4.reset_index(drop=True, inplace = True)

    data = data4                                                    #Don't need full dataframe
    """
    Convert power to energy based on laser power levels and pulse widths, then save this into the dataframe
    """
    Xenergy = data.Energy.values
    lenData = len(Xenergy)
    XenergyPerPulse = np.zeros((lenData,))
    XPW = data.Pulse_Width.values
    for ck in range(lenData):
        
        XenergyPerPulse[ck] = Xenergy[ck]/XPW[ck]
        if XenergyPerPulse[ck] < 0:
            XenergyPerPulse[ck] = 0
    data['XenergyPerPulse'] = XenergyPerPulse

    XenergyPerPulse = data.XenergyPerPulse
    #Grab response variable
    MaxZ = data["Max_BARS_Z"].astype(aesara.config.floatX)               #Convert to tensor
    MaxZ = np.log(MaxZ+0.1)
    #Plot distribution of data (log scale)
    sns.distplot(MaxZ, kde=True)
    

    """
    Now let's setup some meta data for our model analyses. We will set the number of burn in samples, which "primes" the markov chain Monte Carlo (MCMC) algorithm, and number of
    samples to draw from the posterior. In general, less "well behaved" data will require more samples to get MCMC to converge to steady state. 
    """
    numBurnIn = 4000
    numSamples = 5000
    RANDOM_SEED = 7
    """
    Since we are doing a within-subjects repeated measures design, we need to create a mapping between subjects and data. Each animal had a 16 channel recording array in A1 that
    is recording from different groups of neurons. So we sort by animal and electrode
    """
    animal_codes = np.unique(data.animal_code)
    animal_codes_map = np.arange(0,len(animal_codes))
    newCodes = np.zeros((len(data.animal_code)),dtype=int)
    for ck in range(len(data.animal_code)):
        curCode = data.animal_code[ck]
        newCode = np.where(curCode == animal_codes)
        newCode = newCode[0][0]
        newCodes[ck] = int(newCode)
    data['animal_code'] = newCodes
    animal_code_idx = data.animal_code.values
    n_channels = len(data.animal_code.unique())
    """
    Finally get our independent variables, ISI and energy per pulse
    """
    XDist = data.ISI.values
    XDist = np.log(XDist+0.1)
    
    XenergyPerPulse = data['XenergyPerPulse']#np.log(data['XenergyPerPulse']+0.1)
    XenergyPerPulse = np.asarray(np.log(XenergyPerPulse+0.1))
    # Plot data vs predictors
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(XenergyPerPulse,XDist,data['Max_Z_Score'])
    plt.xlabel('Energy')
    plt.ylabel('ISI')

    """
    Now define the model
    """
    pdb.set_trace()
    with pm.Model() as Heirarchical_Regression:
        # Hyperpriors for group nodes
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=1)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        mu_b = pm.Normal("mu_b", mu=0.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5)
        mu_b2 = pm.Normal("mu_b2",mu=0.0, sigma=1)
        sigma_b2 = pm.HalfNormal("sigma_b2",5)
        mu_b3 = pm.Normal("mu_b3", 1)
        sigma_b3 = pm.HalfNormal("sigma_b3",5)
        
        sigma_nu = pm.Exponential("sigma_nu",5.0)
        #Base layer
        nu = pm.HalfCauchy('nu', sigma_nu)          #Nu for robust regression
        a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=(n_channels))
        a = pm.Deterministic("a", mu_a + a_offset * sigma_a)

        b1_offset = pm.Normal('b1_offset', mu=0, sigma=1, shape=(n_channels))
        b1 = pm.Deterministic("b1", mu_b + b1_offset * sigma_b)
        
        b2_offset = pm.Normal("b2_offset",mu=0, sigma=1, shape=(n_channels))
        b2 = pm.Deterministic("b2", mu_b2 + b2_offset*sigma_b2)

        b3_offset = pm.Normal("b3_offset",mu=0, sigma=1, shape=(n_channels))
        b3 = pm.Deterministic("b3", mu_b3 + b3_offset*sigma_b3)

        eps = pm.HalfCauchy("eps", 5,shape=(n_channels))

        regression = a[animal_code_idx] + (b1[animal_code_idx] * XenergyPerPulse) + (b2[animal_code_idx] * XDist) +(b3[animal_code_idx]*XenergyPerPulse*XDist)

        likelihood = pm.StudentT("MaxZ_like",nu=nu,mu=regression,sigma=eps[animal_code_idx], observed= MaxZ) 

    """
    Now we run the model!
    """
    with Heirarchical_Regression:
        if __name__ == '__main__':
                step = pm.NUTS()
                rTrace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.95,chains = 4)
                #rTrace = pm.sampling_jax.sample_numpyro_nuts(numSamples, tune=numBurnIn, target_accept=0.95,chains = 4)

    """
    Now do model analytics
    """
    intercept = rTrace.posterior["a"]                #Grab the posterior distribution of a
    EnergySlope = rTrace.posterior["b1"]                    #Grab the posterior distribution of b1
    ISISlope = rTrace.posterior["b2"]                    #Grab the posterior distribution of B
    InteractionSlope = rTrace.posterior["b3"]                    #Grab the posterior distribution of B
    err = rTrace.posterior["eps"]                    #Grab the posterior distribution of model error
    f_dict = {'size':16}
    
    fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(2,3, figsize=(12,6))
    for ax, estimate, title, xlabel in zip(fig.axes,
                                [intercept, EnergySlope, ISISlope,InteractionSlope, err],
                                ['Intercept', 'Energy Slope','ISI Slope','Interaction Slope','Error Parameter'],
                                [r'$a$', r'$\beta1$', r'$\beta 2$', r'$\beta 3$' , r'$err$']):
        pm.plot_posterior(estimate, point_estimate='mode', ax=ax, color=color,hdi_prob=0.95)
        ax.set_title(title, fontdict=f_dict)
        ax.set_xlabel(xlabel, fontdict=f_dict)
    
    """
    Let's check out model with posterior predictive checks
    """
    with Heirarchical_Regression:
        if __name__ == '__main__':
            ppc = pm.sample_posterior_predictive(rTrace, random_seed=RANDOM_SEED)

    az.plot_bpv(ppc, hdi_prob=0.95,kind='p_value')
    az.plot_ppc(ppc)

    """
    Now let's plot our trace diagnostics
    """
    
    #pm.model_to_graphviz(Heirarchical_Regression)

    az.plot_trace(rTrace, var_names=["mu_a", "mu_b", "mu_b2", "mu_b3", "sigma_a", "sigma_b","sigma_b2","sigma_b3", "eps"])
    
    az.plot_trace(rTrace, var_names=["a"])
    
    az.plot_trace(rTrace, var_names=["b1"])

    az.plot_trace(rTrace, var_names=["b2"])

    az.plot_trace(rTrace, var_names=["b3"])

    az.plot_trace(rTrace, var_names=["nu"])
    plt.show()
    pdb.set_trace()
    az.to_netcdf(rTrace,filename='HierModel_Var5_StudentT_PerPulseScaled_SemilogT.netcdf')
    az.to_netcdf(ppc,filename='HierModel_Var5_StudentT_PerPulseScaled_Semilog_ppc.netcdf')
    