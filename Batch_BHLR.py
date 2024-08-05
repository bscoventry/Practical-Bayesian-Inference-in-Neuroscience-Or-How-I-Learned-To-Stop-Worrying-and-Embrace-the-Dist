#--------------------------------------------------------------------------------------------------------------------------------------------
# Authors: 
#   Brandon S Coventry            Wisconsin Institute for Translational Neuroengineering
# Date: 05/4/2024                       Happy Easter! Wisconsin is finally warmed today, but snow later this week.
# Purpose: This is a function to perform robust  Bayesian hierarchical linear regression.
# Revision History: Will be tracked in Github.
# Notes: N/A
#--------------------------------------------------------------------------------------------------------------------------------------------

# Dependencies for fN if running code in this file
"""
Importing all dependencies
"""
import numpy as np
import pandas as pd
from scipy import stats as stats
# import statsmodels.formula.api as smf
import pymc as pm
# import aesara
import xarray as xr
import arviz as az
# import utils as utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import pdb
#Optional if you want to use speed up tools or other data reading formats
# import nutpie
# import pickle
# import json

"""
Importing main data
"""
#Import data here and do any preprocessing
dataFolder = '/Users/cluu/Documents/Research/INSxAuditory/Dataset/'
df = pd.read_pickle(dataFolder + 'LFPBandChange_Short.pkl')
#df = pd.read_pickle(dataFolder + 'LFPBandChange_Long.pkl')
#collected_var = df.columns.values.tolist()





# Define Bayesian hierarchical linear regression fN
def BHLR(dependentVar_og,indVar1_og,indVar2_og,animNames,electrodes, hyperpriorSigma, islogDep=1,islogIndep=1, model = 'StudentT', saveTrace=1, fileName='TestName', rootSavePath = 'TestName'):
    '''
    Inputs: dependentVar - Your dependent variable as a numpy array. Must be float or int
            indVar1,2 - Independent variable as numpy array. Must be float or int, float preferrable. Can easily add more.
            animNames - Must be a python list. For constructing, see comment in function below.
            electrodes - Must be a python list. For construction see comment in fucntion below.
            islog - This log transforms ind and dep variables. Generally good to start with LFP, and then test with posterior predictive checks
            saveTrace - Save posterior. Defaults to true.
    '''
    
    print(f"Running on PyMC v{pm.__version__}")
    
    # rootSavePath = './'
    
    # if __name__ == '__main__':      #Need to do this for windows. No idea why. Something to do with hyperthreading
        
    """
    Processing data for fitting the BHLR estimator
    """
    bandName = fileName.split('_')[0]
    
    #Transforming data into the log space
    translation_term = abs(min([indVar1_og.min(), indVar2_og.min()])) + 0.001   # Term to translate the input of the log transformation to avoid log(0) - taken as the min value that would bring the net up to 0, then add 0.001

    if islogDep:
        dependentVar = np.log(dependentVar_og + translation_term)
    else:
        dependentVar = dependentVar_og
        
    if islogIndep:
        indVar1 = np.log(indVar1_og + translation_term) 
        indVar2 = np.log(indVar2_og + translation_term)
    else:
        indVar1 = indVar1_og
        indVar2 = indVar2_og

    #Visualizing distributions and data first
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plt.sca(axs[0])
    sns.scatterplot(x=indVar1, y=dependentVar)
    plt.xlabel( 'Energy/ Pulse (mJ + ' + str(translation_term) + ', log)')
    plt.ylabel(bandName + '(log)')

    plt.sca(axs[1])
    sns.scatterplot(x=indVar2, y=dependentVar)
    plt.xlabel('ISI (ms + ' + str(translation_term) + ', log)')
    plt.ylabel(bandName + '(log)')

    plt.sca(axs[2])
    sns.distplot(dependentVar)
    plt.xlabel(bandName + '(log)')
    
    figTitle = 'Initial_Data'
    fig.suptitle(bandName + ': ' + figTitle)
    plt.savefig('/Users/cluu/Documents/Research/INSxAuditory/' + 'Figures/' + fileName + '_' + figTitle + '.jpg', format='jpg')
    plt.savefig(rootSavePath + 'Figures/' + fileName + '_' + figTitle + '.eps', format='eps')
    plt.show()

    
    
    #Organizing the data for hierarchal analysis: categorizing by by animal + electrode.
    dataLEN = len(indVar1)
    animalID = np.zeros((dataLEN,),dtype=int)
    IDXList = {}
    countIDX = 0

    for ck in range((dataLEN)):

        IDX = animNames[ck].split('//')
        curIDX = IDX[0] # Animal name (eg INS2102), extracted from the file str
        elecNum = electrodes[ck]
        keyIDX = curIDX+str(elecNum)
        
        if keyIDX in IDXList.keys():
            animalID[ck] = int(IDXList[keyIDX])     # Do we need int if the og Dict already encode the defn as an in?
        else:
            # animalID[ck] = int(countIDX)
            IDXList[keyIDX] = int(countIDX)
            animalID[ck] = int(IDXList[keyIDX]) 
            countIDX = countIDX+1
        
        
            
    """
    Building the BHLR estimator, then run it!
    """
    
    if model == 'StudentT':
        n_channels = len(np.unique(animalID))

        with pm.Model() as Hierarchical_Regression:
            # Hyperpriors for group nodes
            mu_a = pm.Normal("mu_a", mu=0.0, sigma= 1)
            sigma_a = pm.HalfNormal("sigma_a", hyperpriorSigma)
            mu_b = pm.Normal("mu_b", mu=0.0, sigma= 1)
            sigma_b = pm.HalfNormal("sigma_b", hyperpriorSigma)
            mu_b2 = pm.Normal("mu_b2",mu=0.0, sigma= 1)
            sigma_b2 = pm.HalfNormal("sigma_b2",hyperpriorSigma)
            mu_b3 = pm.Normal("mu_b3", mu=0.0, sigma= 1)
            sigma_b3 = pm.HalfNormal("sigma_b3", hyperpriorSigma)
            
            sigma_nu = pm.Exponential("sigma_nu", 5.0)
            
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

            epsil = pm.HalfCauchy("epsil", 5,shape=(n_channels))

            regression = a[animalID] + (b1[animalID] * indVar1) + (b2[animalID] * indVar2) +(b3[animalID]*indVar1*indVar2)

            likelihood = pm.StudentT("bandDiff_likeli",nu=nu,mu=regression,sigma=epsil[animalID], observed= dependentVar)
            
    elif model == 'Normal':
        
        n_channels = len(np.unique(animalID))

        with pm.Model() as Hierarchical_Regression:
            # Hyperpriors for group nodes
            mu_a = pm.Normal("mu_a", mu=0.0, sigma= 1)
            sigma_a = pm.HalfNormal("sigma_a", hyperpriorSigma)
            mu_b = pm.Normal("mu_b", mu=0.0, sigma= 1)
            sigma_b = pm.HalfNormal("sigma_b", hyperpriorSigma)
            mu_b2 = pm.Normal("mu_b2",mu=0.0, sigma= 1)
            sigma_b2 = pm.HalfNormal("sigma_b2",hyperpriorSigma)
            mu_b3 = pm.Normal("mu_b3", mu=0.0, sigma= 1)
            sigma_b3 = pm.HalfNormal("sigma_b3", hyperpriorSigma)
            
            
            #Base layer
            
            a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=(n_channels, n_components))
            a = pm.Deterministic("a", mu_a + a_offset * sigma_a)

            b1_offset = pm.Normal('b1_offset', mu=0, sigma=1, shape=(n_channels, n_components))
            b1 = pm.Deterministic("b1", mu_b + b1_offset * sigma_b)
            
            b2_offset = pm.Normal("b2_offset",mu=0, sigma=1, shape=(n_channels, n_components))
            b2 = pm.Deterministic("b2", mu_b2 + b2_offset*sigma_b2)

            b3_offset = pm.Normal("b3_offset",mu=0, sigma=1, shape=(n_channels, n_components))
            b3 = pm.Deterministic("b3", mu_b3 + b3_offset*sigma_b3)

            epsil = pm.HalfCauchy("epsil", 5,shape=(n_channels, n_components))

            regression = a[animalID] + (b1[animalID] * indVar1) + (b2[animalID] * indVar2) +(b3[animalID]*indVar1*indVar2)

            likelihood = pm.Normal("bandDiff_likeli", mu=regression, sigma=epsil[animalID], observed= dependentVar)
        


    # Fitting the estimator
    numBurnIn = 2000     #Burning in samples help increase convergence in mcmc of the samples we are keeping
    numSamples = 5000

    with Hierarchical_Regression:
        # if __name__ == '__main__':
        # step = pm.NUTS()
        rTrace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90, chains = 4, nuts_sampler="nutpie")
                
    
    
    """
    Now do model analytics. First plot posterior parameter distributions
    """
    # Extracting the posterior distribution of each parameter
    intercept = rTrace.posterior["a"]               
    EnergySlope = rTrace.posterior["b1"]                   
    ISISlope = rTrace.posterior["b2"]                   
    InteractionSlope = rTrace.posterior["b3"]                   
    err = rTrace.posterior["epsil"]                

    az.style.use("arviz-darkgrid")
    f_dict = {'size':16}
    color = '#87ceeb'

    fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(2,3, figsize=(18,10))
    for ax, estimate, title, xlabel in zip(fig.axes,
                                [intercept, EnergySlope, ISISlope,InteractionSlope, err],
                                ['Intercept', 'Energy Slope','ISI Slope','Interaction Slope','Error Parameter'],
                                [r'$a$', r'$\beta1$', r'$\beta 2$', r'$\beta 3$' , r'$err$']):
        az.rcParams["plot.max_subplots"] = n_channels       # default max is 40, but we have around 80 animal+electrode categories. Not sure if plotting only 40 affect numerical outputs, or just the lines drawn on the graph... so being cautious, I incr the max to 80 (at cost of slower run)
        pm.plot_posterior(estimate, point_estimate='mode', ax=ax, color=color,hdi_prob=0.95)
        ax.set_title(title, fontdict=f_dict)
        ax.set_xlabel(xlabel, fontdict=f_dict)
    
    
    figTitle = 'posteriorHDI'
    fig.suptitle(bandName + ': ' + figTitle)
    plt.savefig('/Users/cluu/Documents/Research/INSxAuditory/' + 'Figures/' + fileName + '_' + figTitle + '.jpg', format='jpg')
    plt.savefig(rootSavePath + 'Figures/' + fileName + '_' + figTitle + '.eps', format='eps')
    plt.show()

    
    """
    Now for the all important posterior predictive checks! 
    """
    # First calc model log-likelihood
    with Hierarchical_Regression:
        pm.compute_log_likelihood(rTrace)

    randomSeed = 777    #Seed for sampling the PPD
    with Hierarchical_Regression:
        # if __name__ == '__main__':
        ppc = pm.sample_posterior_predictive(rTrace, random_seed=randomSeed)

    # Plot PPC and Bayesian P-values
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    az.plot_ppc(ppc, ax=axs[0])
    
    # Setting a limit for the x axis based on the range of the dependent variable
    base = 5
    bound = round( max(abs(dependentVar))/ base) * base
    if bound > 10:
        bound_final = 10
    elif bound <= 10:
        bound_final = 5
    axs[0].set_xlim(-bound_final, bound_final)
    
    
    az.plot_bpv(ppc, hdi_prob=0.95,kind='p_value', ax=axs[1])
    
    
    figTitle = 'PPC_BPV'
    fig.suptitle(bandName + ': ' + figTitle)
    plt.savefig('/Users/cluu/Documents/Research/INSxAuditory/' + 'Figures/' + fileName + '_' + figTitle + '.jpg', format='jpg')
    plt.savefig(rootSavePath + 'Figures/' + fileName + '_'  + figTitle + '.eps', format='eps')


    """
    Saving posterior trace for model comparison/ sensitivity analysis later
    """
    if saveTrace==1:
        az.to_netcdf(rTrace, rootSavePath + 'Traces/' + fileName + '.netcdf')


#----------------------------------------------------------------------------------------------
# Use this section to load data and loop for batch
#----------------------------------------------------------------------------------------------

"""
Setting up variables of interest
"""
indVar1_og = df['EnergyPerPulse'].values   
indVar2_og = df['ISI'].values
animNames = df['DataID'].values
electrodes = df['Electrode'].values



"""
Setting the the set of parameters to loop and test
""" 
all_z = ['betaDiff'] #[var for var in collected_var if 'Diff' in var]  # Dependent variable set
hyperprior_sigma_range = [1] #, 0.5, 5, 10]     # Range of mu for generating the hyperprior sigma, variance of each prior

"""
Initial building and testing of the BHLR models
""" 
from BHLR import BHLR
#This is an example of doing sensitivity analysis by varying hyperprior distributions
model = 'Mixture'
for band_z  in all_z:
    for hyperprior_sigma in hyperprior_sigma_range:
        
        dependentVar_og = df[band_z].values
        fileName = band_z + '_' + model + '_Long_HyperPrior_' + str(hyperprior_sigma) + '_LogLogTransform'
        rootSavePath = '/Volumes/BlueRaven/Research/INSxAuditory/Band Diff - Long/'

        
        print('Fitting BHLR on: ' + band_z)
        print('Hyperprior sigma = ' + str(hyperprior_sigma))
        print('Saved files @ ' + fileName)
        
        BHLR(dependentVar_og, 
             indVar1_og, indVar2_og, 
             animNames, electrodes,
             hyperpriorSigma=hyperprior_sigma,
             islogDep=0,islogIndep=1,
             model = model,
             saveTrace=1, fileName=fileName, rootSavePath=rootSavePath)