#-----------------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry                    Purdue University, UW Madison Neurosurgery
# Date: 08/2022
# Purpose: This program isn't discussed in the paper, but is a demo showing how we can simply compare groups without fancy ANOVAs. Care should
#          be taken though in posterior checks as these are subject to Type M and Type S errors due to lack of hierarchical structure offered 
#          in BANOVAs
# Revision Hist: See Github for rollback
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
# Let's load some data!
if __name__ == '__main__':       #This statement is to allow for parallel sampling in windows. Linux distributions don't require this.
    print(f"Running on PyMC v{pm.__version__}")      #Tells us which version of PyMC we are running
    AMDepthData = pd.read_csv("depthData.csv")      #Read in our data. Note this address may change based on where program files are downloaded.
    data1 = AMDepthData.loc[AMDepthData['Age'] == 1]                   #Let's consider young responses first.
    data1.reset_index(drop=True, inplace = True)
    AMDepthData = data1
    """
    Now let's setup some meta data for our model analyses. We will set the number of burn in samples, which "primes" the markov chain Monte Carlo (MCMC) algorithm, and number of
    samples to draw from the posterior. In general, less "well behaved" data will require more samples to get MCMC to converge to steady state. 
    """
    numBurnIn = 2000
    numSamples = 4000
    #These next two lines of code setup the color scheme used for plotting. Does not affect model at all
    color = '#87ceeb'
    az.style.use("arviz-darkgrid")
    Randomseed = 7
    """
    Now let's grab out relevant data. For AM Depth studies, our data file contains a tot_mean variable, which codes the mean firing rate elicited from a stimulus at a given 
    modulation depth. Modulation depth is metric data which exists across a continuum of values, meaning that a regression is a good choice. So the goal is to build a model
    of the form:
    firingRate = a + B*modDepth + modelError
    where a can be largely thought of as spontaneous, non evoked firing rates, B is slope describing the relative change of firing rate as a function of modulation depth, and
    modelError being the error term of the model describing misses of our fitting process. Because this is a Bayesian approach, a, B, and modelError are themselves distributions
    that we will make inference on.
    """
    # data3 = AMDepthData.loc[(AMDepthData['Age'])==1]
    # data3.reset_index(drop=True, inplace = True)
    # AMDepthData = data3
    modDepth = AMDepthData.ModDepth                     #This is our modulation depth vector
    firingRate = AMDepthData['TotMean']                 #This is our mean firing rate. Note, data can be accessed in a pandas array using dot notation (data.subsetData) or
    
    firingRate = np.log(firingRate+0.1)
                                                        #Index like data['subsetData']
    firingRate = firingRate.astype(aesara.config.floatX) #Make sure response variable is in a tensor like structure for computaiton. This is the only time we need to directly invoke aesara
    modDepth = np.asarray(modDepth)                     #Make sure Xs are not panda series, aesara is not a fan of those sometimes.
    """
    Now, let's see if there are differences between groups to determine threshold for Depth. We will do this by comparing differences between means and differences between variances on the posteriors.
    """
    # AMDepth_30 = AMDepthData.loc[(AMDepthData['ModDepth'])==-30]               #Grab Depth at -30
    # AMDepth_30.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_30 = AMDepth_30.Scaled
    
    # AMDepth_24 = AMDepthData.loc[(AMDepthData['ModDepth'])==-24]               #Grab Depth at -24
    # AMDepth_24.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_24 = AMDepth_24.Scaled

    # AMDepth_18 = AMDepthData.loc[(AMDepthData['ModDepth'])==-18]               #Grab Depth at -18
    # AMDepth_18.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_18 = AMDepth_18.Scaled

    # AMDepth_12 = AMDepthData.loc[(AMDepthData['ModDepth'])==-12]               #Grab Depth at -12
    # AMDepth_12.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_12 = AMDepth_12.Scaled

    # AMDepth_6 = AMDepthData.loc[(AMDepthData['ModDepth'])==-6]               #Grab Depth at -6
    # AMDepth_6.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_6 = AMDepth_6.Scaled

    # AMDepth_2_5 = AMDepthData.loc[(AMDepthData['ModDepth'])==-2.5]               #Grab Depth at -2.5
    # AMDepth_2_5.reset_index(drop=True, inplace = True)                          #This just cleans up the indexing for book keeping of data
    # firing_2_5 = AMDepth_2_5.Scaled

    # AMDepth_0 = AMDepthData.loc[(AMDepthData['ModDepth'])==0]                  #Grab Depth at -0
    # AMDepth_0.reset_index(drop=True, inplace = True)                           #This just cleans up the indexing for book keeping of data
    # firing_0 = AMDepth_0.Scaled

    nGroups = 7                                                               #Number of groups total
    modDepths = AMDepthData.ModDepth                                            #Grab the mod depths.
    uniqueModDepths = np.unique(modDepths)                                      #These are the Unique Mod Depths. Use this to index where they occur in data
    modMapping = np.zeros((len(modDepths,)), dtype=int)
    for ck in range(len(modDepths)):
        curMod = modDepths[ck]
        indx = np.argwhere(uniqueModDepths==curMod)
        modMapping[ck] = int(indx)

    priorMean = firingRate.mean()
    priorStDv = firingRate.std()
    
    with pm.Model() as CompareModel:
        mu = pm.Normal('mu', mu=priorMean, sigma=priorStDv, shape=nGroups)
        sigma = pm.HalfNormal('sigma', priorStDv*10, shape=nGroups)
        likelihood = pm.Normal('likelihood', mu[modMapping], sigma=sigma[modMapping], observed=firingRate)
    with CompareModel:
        traceCMP = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4)
    
    trace30M = traceCMP.posterior['mu'][:,:,0]
    trace24M = traceCMP.posterior['mu'][:,:,1]
    trace18M = traceCMP.posterior['mu'][:,:,2]
    trace12M = traceCMP.posterior['mu'][:,:,3]
    trace6M = traceCMP.posterior['mu'][:,:,4]
    trace2M = traceCMP.posterior['mu'][:,:,5]
    trace0M = traceCMP.posterior['mu'][:,:,6]

    trace30S = traceCMP.posterior['sigma'][:,:,0]
    trace24S = traceCMP.posterior['sigma'][:,:,1]
    trace18S = traceCMP.posterior['sigma'][:,:,2]
    trace12S = traceCMP.posterior['sigma'][:,:,3]
    trace6S = traceCMP.posterior['sigma'][:,:,4]
    trace2S = traceCMP.posterior['sigma'][:,:,5]
    trace0S = traceCMP.posterior['sigma'][:,:,6]
    pm.plot_trace(traceCMP)

    fig, axes = plt.subplots(6,3, figsize=(12, 12))
    meanVars = (trace24M-trace30M, trace18M-trace30M,trace12M-trace30M,trace6M-trace30M,trace2M-trace30M,trace0M-trace30M)
    stdvVars = (trace24S-trace30S, trace18S-trace30S,trace12S-trace30S,trace6S-trace30S,trace2S-trace30S,trace0S-trace30S)
    effectVars = ((trace24M-trace30M)/np.sqrt((trace24S**2+trace30S**2)/2),(trace18M-trace30M)/np.sqrt((trace18S**2+trace30S**2)/2),(trace12M-trace30M)/np.sqrt((trace12S**2+trace30S**2)/2),(trace6M-trace30M)/np.sqrt((trace6S**2+trace30S**2)/2),(trace2M-trace30M)/np.sqrt((trace2S**2+trace30S**2)/2),(trace0M-trace30M)/np.sqrt((trace0S**2+trace30S**2)/2))
    counter = 0

    az.plot_posterior(meanVars[0], point_estimate='mode', ref_val=0, ax=axes[0,0], color=color,hdi_prob=0.95)
    axes[0,0].set_xlabel('$\mu_{24} - \mu_{30}$')
    az.plot_posterior(meanVars[1], point_estimate='mode', ref_val=0, ax=axes[1,0], color=color,hdi_prob=0.95)
    axes[1,0].set_xlabel('$\mu_{18} - \mu_{30}$')
    az.plot_posterior(meanVars[2], point_estimate='mode', ref_val=0, ax=axes[2,0], color=color,hdi_prob=0.95)
    axes[2,0].set_xlabel('$\mu_{12} - \mu_{30}$')
    az.plot_posterior(meanVars[3], point_estimate='mode', ref_val=0, ax=axes[3,0], color=color,hdi_prob=0.95)
    axes[3,0].set_xlabel('$\mu_6 - \mu_{30}$')
    az.plot_posterior(meanVars[4], point_estimate='mode', ref_val=0, ax=axes[4,0], color=color,hdi_prob=0.95)
    axes[4,0].set_xlabel('$\mu_2.5 - \mu_{30}$')
    az.plot_posterior(meanVars[5], point_estimate='mode', ref_val=0, ax=axes[5,0], color=color,hdi_prob=0.95)
    axes[5,0].set_xlabel('$\mu_0 - \mu_{30}$')

    az.plot_posterior(stdvVars[0], point_estimate='mode', ref_val=0, ax=axes[0,1], color=color,hdi_prob=0.95)
    axes[0,1].set_xlabel('$\sigma_{24} - \sigma_{30}$')
    az.plot_posterior(stdvVars[1], point_estimate='mode', ref_val=0, ax=axes[1,1], color=color,hdi_prob=0.95)
    axes[1,1].set_xlabel('$\sigma_{18} - \sigma_{30}$')
    az.plot_posterior(stdvVars[2], point_estimate='mode', ref_val=0, ax=axes[2,1], color=color,hdi_prob=0.95)
    axes[2,1].set_xlabel('$\sigma_{12} - \sigma_{30}$')
    az.plot_posterior(stdvVars[3], point_estimate='mode', ref_val=0, ax=axes[3,1], color=color,hdi_prob=0.95)
    axes[3,1].set_xlabel('$\sigma_6 - \sigma_{30}$')
    az.plot_posterior(stdvVars[4], point_estimate='mode', ref_val=0, ax=axes[4,1], color=color,hdi_prob=0.95)
    axes[4,1].set_xlabel('$\sigma_2.5 - \sigma_{30}$')
    az.plot_posterior(stdvVars[5], point_estimate='mode', ref_val=0, ax=axes[5,1], color=color,hdi_prob=0.95)
    axes[5,1].set_xlabel('$\sigma_0 - \sigma_{30}$')
    
    
    
    az.plot_posterior(effectVars[0], point_estimate='mode', ref_val=0, ax=axes[0,2], color=color,hdi_prob=0.95)
    axes[0,2].set_xlabel('$EffectSize_{24} - EffectSize_{30}$')
    az.plot_posterior(effectVars[1], point_estimate='mode', ref_val=0, ax=axes[1,2], color=color,hdi_prob=0.95)
    axes[1,2].set_xlabel('$EffectSize_{18} - EffectSize_{30}$')
    az.plot_posterior(effectVars[2], point_estimate='mode', ref_val=0, ax=axes[2,2], color=color,hdi_prob=0.95)
    axes[2,2].set_xlabel('$EffectSize_{12} - EffectSize_{30}$')
    az.plot_posterior(effectVars[3], point_estimate='mode', ref_val=0, ax=axes[3,2], color=color,hdi_prob=0.95)
    axes[3,2].set_xlabel('$EffectSize_6 - EffectSize_{30}$')
    az.plot_posterior(effectVars[4], point_estimate='mode', ref_val=0, ax=axes[4,2], color=color,hdi_prob=0.95)
    axes[4,2].set_xlabel('$EffectSize_2.5 - EffectSize_{30}$')
    az.plot_posterior(effectVars[5], point_estimate='mode', ref_val=0, ax=axes[5,2], color=color,hdi_prob=0.95)
    axes[5,2].set_xlabel('$EffectSize_0 - EffectSize_{30}$')
    


    plt.tight_layout()
    plt.show()
    pdb.set_trace()
    """
    Prior Checks, flipping trial set data to see what priors give you.
    """