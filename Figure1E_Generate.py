#-----------------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry                    Purdue University, UW Madison Neurosurgery
# Date: 01/2024
# Purpose: This program simply plots figures of the illustrative example of Figure 1E. 
# Revision Hist: See Github
#-----------------------------------------------------------------------------------------------------------------------------------------------
"""
To begin, let's import all of our dependencies, including our data and python packages
"""
import numpy as np               #Numpy for numerical 'ala Matlab' computations
import pymc as pm                #pymc will be doing most of our heavy lifting for Bayesian calculations
import matplotlib.pyplot as plt  #This works as our plotting tool
import arviz as az               # arviz is a visualization package for plotting probabilistic variables, such as prior or posterior distributions
import aesara                    #Aesara is out tool for calculations involving tensors. PyMC will mostly work with this for us.
import pandas as pd              #Using pandas to read in CSV data files
import pickle                    #Pickle for saving data after completion of our model
import seaborn as sns            #We will use some of seaborn's distribution plotting tools
import pdb
from scipy.stats import norm 
# Let's load some data!
if __name__ == '__main__':       #This statement is to allow for parallel sampling in windows. Linux distributions don't require this.
    numBurnIn = 2000
    numSamples = 4000
    #These next two lines of code setup the color scheme used for plotting. Does not affect model at all
    color = '#87ceeb'
    az.style.use("arviz-darkgrid")
    Randomseed = 7
   

    nGroups = 2                                                              #Number of groups total                                     
    uniqueModDepths = np.array([0,1])                                  
    muData = 20
    kdata = 1
    firingRateStim = np.random.normal(muData,kdata,size=2000)
    firingRateSpon = np.random.normal(12,3,size=2000)
    sns.distplot(firingRateStim,kde=True)
    firingRate = np.concatenate((firingRateStim,firingRateSpon))
    modMap1 = np.zeros((2000,),dtype=int)
    modMap2 = np.ones((2000,),dtype=int)
    modMapping = np.concatenate((modMap1,modMap2),dtype=int)
    sns.distplot(firingRateSpon,kde=True)
    priorMean = 60
    priorStDv = 15
    
    with pm.Model() as CompareModel:
        mu = pm.Normal('mu', mu=priorMean, sigma=priorStDv, shape=nGroups)
        sigma = pm.HalfNormal('sigma', priorStDv, shape=nGroups)
        likelihood = pm.Normal('likelihood', mu[modMapping], sigma=sigma[modMapping], observed=firingRate)
    with CompareModel:
        traceCMP = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4)
    
    traceStim = traceCMP.posterior['mu'][:,:,0]
    traceSpon = traceCMP.posterior['mu'][:,:,1]
    traceStimS = traceCMP.posterior['sigma'][:,:,0]
    traceSponS = traceCMP.posterior['sigma'][:,:,1]
    pm.plot_trace(traceCMP)

    meanVars = (traceStim-traceSpon)
    
    effectVars = (traceStim-traceSpon)/np.sqrt((traceStimS**2+traceSponS**2)/2)
    counter = 0

    az.plot_posterior(meanVars, point_estimate='mode', color=color,hdi_prob=0.95)
    #axes[0,0].set_xlabel('$\mu_{24} - \mu_{30}$')
    # az.plot_posterior(meanVars[1], point_estimate='mode', ref_val=0, ax=axes[1,0], color=color,hdi_prob=0.95)
    # axes[1,0].set_xlabel('$\mu_{18} - \mu_{30}$')
    # az.plot_posterior(meanVars[2], point_estimate='mode', ref_val=0, ax=axes[2,0], color=color,hdi_prob=0.95)
    # axes[2,0].set_xlabel('$\mu_{12} - \mu_{30}$')
    # az.plot_posterior(meanVars[3], point_estimate='mode', ref_val=0, ax=axes[3,0], color=color,hdi_prob=0.95)
    # axes[3,0].set_xlabel('$\mu_6 - \mu_{30}$')
    # az.plot_posterior(meanVars[4], point_estimate='mode', ref_val=0, ax=axes[4,0], color=color,hdi_prob=0.95)
    # axes[4,0].set_xlabel('$\mu_2.5 - \mu_{30}$')
    # az.plot_posterior(meanVars[5], point_estimate='mode', ref_val=0, ax=axes[5,0], color=color,hdi_prob=0.95)
    # axes[5,0].set_xlabel('$\mu_0 - \mu_{30}$')

    # az.plot_posterior(stdvVars[0], point_estimate='mode', ref_val=0, ax=axes[0,1], color=color,hdi_prob=0.95)
    # axes[0,1].set_xlabel('$\sigma_{24} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[1], point_estimate='mode', ref_val=0, ax=axes[1,1], color=color,hdi_prob=0.95)
    # axes[1,1].set_xlabel('$\sigma_{18} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[2], point_estimate='mode', ref_val=0, ax=axes[2,1], color=color,hdi_prob=0.95)
    # axes[2,1].set_xlabel('$\sigma_{12} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[3], point_estimate='mode', ref_val=0, ax=axes[3,1], color=color,hdi_prob=0.95)
    # axes[3,1].set_xlabel('$\sigma_6 - \sigma_{30}$')
    # az.plot_posterior(stdvVars[4], point_estimate='mode', ref_val=0, ax=axes[4,1], color=color,hdi_prob=0.95)
    # axes[4,1].set_xlabel('$\sigma_2.5 - \sigma_{30}$')
    # az.plot_posterior(stdvVars[5], point_estimate='mode', ref_val=0, ax=axes[5,1], color=color,hdi_prob=0.95)
    # axes[5,1].set_xlabel('$\sigma_0 - \sigma_{30}$')
    
    
    
    az.plot_posterior(effectVars, point_estimate='mode', color=color,hdi_prob=0.95)
    #axes[0,2].set_xlabel('$EffectSize_{24} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[1], point_estimate='mode', ref_val=0, ax=axes[1,2], color=color,hdi_prob=0.95)
    # axes[1,2].set_xlabel('$EffectSize_{18} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[2], point_estimate='mode', ref_val=0, ax=axes[2,2], color=color,hdi_prob=0.95)
    # axes[2,2].set_xlabel('$EffectSize_{12} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[3], point_estimate='mode', ref_val=0, ax=axes[3,2], color=color,hdi_prob=0.95)
    # axes[3,2].set_xlabel('$EffectSize_6 - EffectSize_{30}$')
    # az.plot_posterior(effectVars[4], point_estimate='mode', ref_val=0, ax=axes[4,2], color=color,hdi_prob=0.95)
    # axes[4,2].set_xlabel('$EffectSize_2.5 - EffectSize_{30}$')
    # az.plot_posterior(effectVars[5], point_estimate='mode', ref_val=0, ax=axes[5,2], color=color,hdi_prob=0.95)
    # axes[5,2].set_xlabel('$EffectSize_0 - EffectSize_{30}$')
    


    plt.tight_layout()
    #plt.show()
    #pdb.set_trace()
    

    nGroups = 2                                                              #Number of groups total                                     
    
    priorMean = 22
    priorStDv = 1
    
    with pm.Model() as CompareModel:
        mu = pm.Normal('mu', mu=priorMean, sigma=priorStDv, shape=nGroups)
        sigma = pm.HalfNormal('sigma', priorStDv, shape=nGroups)
        likelihood = pm.Normal('likelihood', mu[modMapping], sigma=sigma[modMapping], observed=firingRate)
    with CompareModel:
        traceCMP = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4)
    
    traceStim = traceCMP.posterior['mu'][:,:,0]
    traceSpon = traceCMP.posterior['mu'][:,:,1]
    traceStimS = traceCMP.posterior['sigma'][:,:,0]
    traceSponS = traceCMP.posterior['sigma'][:,:,1]
    pm.plot_trace(traceCMP)

    meanVars = (traceStim-traceSpon)
    
    effectVars = (traceStim-traceSpon)/np.sqrt((traceStimS**2+traceSponS**2)/2)
    counter = 0

    az.plot_posterior(meanVars, point_estimate='mode', color=color,hdi_prob=0.95)
    #axes[0,0].set_xlabel('$\mu_{24} - \mu_{30}$')
    # az.plot_posterior(meanVars[1], point_estimate='mode', ref_val=0, ax=axes[1,0], color=color,hdi_prob=0.95)
    # axes[1,0].set_xlabel('$\mu_{18} - \mu_{30}$')
    # az.plot_posterior(meanVars[2], point_estimate='mode', ref_val=0, ax=axes[2,0], color=color,hdi_prob=0.95)
    # axes[2,0].set_xlabel('$\mu_{12} - \mu_{30}$')
    # az.plot_posterior(meanVars[3], point_estimate='mode', ref_val=0, ax=axes[3,0], color=color,hdi_prob=0.95)
    # axes[3,0].set_xlabel('$\mu_6 - \mu_{30}$')
    # az.plot_posterior(meanVars[4], point_estimate='mode', ref_val=0, ax=axes[4,0], color=color,hdi_prob=0.95)
    # axes[4,0].set_xlabel('$\mu_2.5 - \mu_{30}$')
    # az.plot_posterior(meanVars[5], point_estimate='mode', ref_val=0, ax=axes[5,0], color=color,hdi_prob=0.95)
    # axes[5,0].set_xlabel('$\mu_0 - \mu_{30}$')

    # az.plot_posterior(stdvVars[0], point_estimate='mode', ref_val=0, ax=axes[0,1], color=color,hdi_prob=0.95)
    # axes[0,1].set_xlabel('$\sigma_{24} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[1], point_estimate='mode', ref_val=0, ax=axes[1,1], color=color,hdi_prob=0.95)
    # axes[1,1].set_xlabel('$\sigma_{18} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[2], point_estimate='mode', ref_val=0, ax=axes[2,1], color=color,hdi_prob=0.95)
    # axes[2,1].set_xlabel('$\sigma_{12} - \sigma_{30}$')
    # az.plot_posterior(stdvVars[3], point_estimate='mode', ref_val=0, ax=axes[3,1], color=color,hdi_prob=0.95)
    # axes[3,1].set_xlabel('$\sigma_6 - \sigma_{30}$')
    # az.plot_posterior(stdvVars[4], point_estimate='mode', ref_val=0, ax=axes[4,1], color=color,hdi_prob=0.95)
    # axes[4,1].set_xlabel('$\sigma_2.5 - \sigma_{30}$')
    # az.plot_posterior(stdvVars[5], point_estimate='mode', ref_val=0, ax=axes[5,1], color=color,hdi_prob=0.95)
    # axes[5,1].set_xlabel('$\sigma_0 - \sigma_{30}$')
    
    
    
    az.plot_posterior(effectVars, point_estimate='mode', color=color,hdi_prob=0.95)
    #axes[0,2].set_xlabel('$EffectSize_{24} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[1], point_estimate='mode', ref_val=0, ax=axes[1,2], color=color,hdi_prob=0.95)
    # axes[1,2].set_xlabel('$EffectSize_{18} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[2], point_estimate='mode', ref_val=0, ax=axes[2,2], color=color,hdi_prob=0.95)
    # axes[2,2].set_xlabel('$EffectSize_{12} - EffectSize_{30}$')
    # az.plot_posterior(effectVars[3], point_estimate='mode', ref_val=0, ax=axes[3,2], color=color,hdi_prob=0.95)
    # axes[3,2].set_xlabel('$EffectSize_6 - EffectSize_{30}$')
    # az.plot_posterior(effectVars[4], point_estimate='mode', ref_val=0, ax=axes[4,2], color=color,hdi_prob=0.95)
    # axes[4,2].set_xlabel('$EffectSize_2.5 - EffectSize_{30}$')
    # az.plot_posterior(effectVars[5], point_estimate='mode', ref_val=0, ax=axes[5,2], color=color,hdi_prob=0.95)
    # axes[5,2].set_xlabel('$EffectSize_0 - EffectSize_{30}$')
    plt.figure()
    x_axis = np.arange(0, 100, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, 60, 15))  
    plt.figure()
    x_axis = np.arange(0, 40, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, 22, 1))  

    plt.tight_layout()
    plt.show()
    pdb.set_trace()