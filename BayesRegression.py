#-------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry     Wisconsin Institute for Translational Neuroengineering
# Date: 08-17-2022
# Purpose: This calculates a simple linear regression (Bayesian Formulation) on auditory evoked data with modulation of amplitude
#          modulated depth. Use this as a tutorial for simple linear regression for neural data
# Revision History: N/A
#-------------------------------------------------------------------------------------------------------------------------------------
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
    Let's do some data visualization first. Always good to look at the data first. We will plot the scatter plot of the response variable and predictor variable
    """
    plt.figure(1)
    plt.scatter(modDepth,firingRate)
    plt.show()
    #Plot the distribution of response variable, firing rate
    sns.displot(firingRate, kde=True)
    plt.show()
    """
    Okay great, the distribution looks somewhat normally distributed. Let's carry on
    """
    """
    Let's define our model. This is where the power of Bayesian responses really shows. We can easily define our own models to explicitly fit data. To do so, let's define a prior and
    a likelihood function. For regresison, our likelihood function takes the form of Y = a +B*predictor + modelError. Our prior can come from previous knowledge, or relatively uninformative,
    mostly letting data speak for itself. We will use a Normally distributed prior, which is preferred to a uniform prior. This is because if data falls outside of a uniform prior, the probability incorporated
    into the model is 0, effectively ignoring more extreme values. Intuitively, we want to incorporate more extreme data, but at a reduced level. Normal distributions do this for us.
    """

    with pm.Model() as regression:                    #Define a model that we call regression
        a = pm.Normal('a', mu=0, sigma = 5)           #Normally distributed prior on a
        B = pm.Normal('B', mu=0, sigma = 5)           #Normally distributed prior on B
        eps = pm.HalfCauchy("eps", 5)                 #Model error prior, half Cauchy distributed with variance 5
        # Now we define our likelihood function, which for regression is our regression function
        reg = pm.Deterministic('reg', a + (B*modDepth))      #Deterministic is for non probabilistic data. This is a modification to help sampling, inference is still probabilistic
        likelihood = pm.Normal('Y',mu = reg, sigma = eps, observed = firingRate)    
        """
        And that's it! We've quickly, explicitly, and easily defined our model. We set prior distributions on a, B, and modelError, and defined a likelihood function of a normal linear regression
        with our observed data being our firingRate, which is the variable we are trying to predict. Easy eh?
        """ 
    """
    Our last step is to run inference to get our posterior distribution which we do by MCMC sampling. In PyMC this is also easy.
    """
    with regression:                 #Access our defined model
        trace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4)       #4 parallel, independent MCMC chains.
    """
    Now we're sampling! We initialize the chain with numBurnIn samples, and then run for numSamples. Target_accept determines MCMC step size in order to get to a sampling acceptance rate. Generally,
    higher probability of target_accept helps with difficult to sample posteriors. For now, 0.9 is fine.

    You will see the sampling run in the command prompt. Once complete, we have our posterior and can now make inference!
    """
    
    intercept = trace.posterior["a"]                #Grab the posterior distribution of a
    Slope = trace.posterior["B"]                    #Grab the posterior distribution of B
    err = trace.posterior["eps"]                    #Grab the posterior distribution of model error
    
    """
    Let's plot our posteriors!
    """
    #az.plot_trace(trace, compact=True)
    az.plot_posterior(trace, var_names=['a', 'B','eps'], point_estimate='mode',hdi_prob=0.95)
    plt.show()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(
    111,
    xlabel=r"Modulation Depth",
    ylabel=r"Firing Rate",
    title="Posterior predictive regression lines",
    )
    sc = ax.scatter(modDepth, firingRate)

    az.plot_hdi(
        modDepth,
        trace.posterior.reg,
        color="k",
        hdi_prob=0.95,
        ax=ax,
        fill_kwargs={"alpha": 0.25},
        smooth=False,
    )
    plt.show()
    pdb.set_trace()
    """
    Now let's do some posterior predictive checks. PyMC has some nice functions that make this quite easy. We will also sample the posterior distribution for the
    standard 16,000 samples, which for this posterior should be more than enough.
    """
    with regression:
        ppcRegression = pm.sample_posterior_predictive(trace, random_seed=Randomseed)
    #The above code envokes the regression mode, then uses the posterior from the trace, pulling synthetic samples to compare to observed. Random seed is set so that each run can be perfectly replicated
    az.plot_bpv(ppcRegression, hdi_prob=0.95,kind='p_value')
    #Bayes p-values, similar to frequentist,can be used to assess if posterior predictive is sufficiently close to observed density. Should be centered around 0.50.
    az.plot_ppc(ppcRegression)
    plt.show()
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

    fig, axes = plt.subplots(6,2, figsize=(12, 12))
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
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(6,1, figsize=(12, 12))
    az.plot_posterior(effectVars[0], point_estimate='mode', ref_val=0, ax=axes[0], color=color,hdi_prob=0.95)
    axes[0].set_xlabel('$EffectSize_{24} - EffectSize_{30}$')
    az.plot_posterior(effectVars[1], point_estimate='mode', ref_val=0, ax=axes[1], color=color,hdi_prob=0.95)
    axes[1].set_xlabel('$EffectSize_{18} - EffectSize_{30}$')
    az.plot_posterior(effectVars[2], point_estimate='mode', ref_val=0, ax=axes[2], color=color,hdi_prob=0.95)
    axes[2].set_xlabel('$EffectSize_{12} - EffectSize_{30}$')
    az.plot_posterior(effectVars[3], point_estimate='mode', ref_val=0, ax=axes[3], color=color,hdi_prob=0.95)
    axes[3].set_xlabel('$EffectSize_6 - EffectSize_{30}$')
    az.plot_posterior(effectVars[4], point_estimate='mode', ref_val=0, ax=axes[4], color=color,hdi_prob=0.95)
    axes[4].set_xlabel('$EffectSize_2.5 - EffectSize_{30}$')
    az.plot_posterior(effectVars[5], point_estimate='mode', ref_val=0, ax=axes[5], color=color,hdi_prob=0.95)
    axes[5].set_xlabel('$EffectSize_0 - EffectSize_{30}$')
    


    plt.tight_layout()
    plt.show()
    pdb.set_trace()
    """
    Prior Checks, flipping trial set data to see what priors give you.
    """