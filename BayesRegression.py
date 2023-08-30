#-------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry     Wisconsin Institute for Translational Neuroengineering
# Date: 08-17-2022
# Purpose: This calculates a simple linear regression (Bayesian Formulation) on auditory evoked data with modulation of amplitude
#          modulated depth. Use this as a tutorial for simple linear regression for neural data
# Revision History: See Github for Rollback
# Notes: PDB is added in to debug code at your leasure. Add pdb.set_trace() anywhere in line to create a break point to explore 
# variables. We highly recommend doing this to explore the code to get a good feel for it.
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
if __name__ == '__main__':       #This statement is to allow for parallel sampling in windows. Linux distributions don't require this. Doesn't hurt Linux to have this either.
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
    prMean = np.mean(firingRate)
    with pm.Model() as regression:                    #Define a model that we call regression
        a = pm.Normal('a', mu=prMean, sigma = 5)           #Normally distributed prior on a
        B = pm.Normal('B', mu=prMean, sigma = 5)           #Normally distributed prior on B
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
    #Plot posterior HDIs
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
    #pdb.set_trace()
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
    az.plot_trace(trace,var_names=['a', 'B','eps'])
    plt.show()
   