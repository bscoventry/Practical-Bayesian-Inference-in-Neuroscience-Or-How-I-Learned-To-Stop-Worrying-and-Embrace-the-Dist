#--------------------------------------------------------------------
# Author: Brandon Coventry, PhD Wisconsin Institute for Translational Neuroengineering
# Date: 12/23/23
# Purpose: This is to demonstrate Bayesian Estimation Supercedes the T-Test in PyMC
# For group comparisons. See the original Kruscke paper: https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
# Data: BGTC circuit Dopamine depleted DBS vs no DBS.
# Note: Code adapted from Kruscke's original paper and the PyMC BEST tutorial
#--------------------------------------------------------------------
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import arviz as az
import pdb
import pickle as pl
from scipy.stats import mode

#Let's define the Bayesian Estimates Supersedes the T-Test (BEST) model.
if __name__ == '__main__':
    #To begin BEST estimation of mean difference between two groups, begin by getting total sample statistics
    with open('BetaPowerStimComparison.p','rb') as f:
        DBS = pl.load(f)
    with open('BetaPowerNoStimComparison.p','rb') as f:
        NoStim = pl.load(f)
    totalData = pd.DataFrame(
    dict(value=np.r_[DBS, NoStim], group=np.r_[["DBS"] * len(DBS), ["NoStim"] * len(NoStim)]))
    
    group_mu = totalData.value.mean()               
    group_sigma = totalData.value.std() * 2
    #Okay, let's define our model
    with pm.Model() as best:
        group1_mean = pm.Normal("group1_mean",mu=group_mu,sigma=group_sigma)
        group2_mean = pm.Normal("group2_mean",mu=group_mu,sigma=group_sigma)
        #This particular model does use uniform distributions on sigma, so need to be careful here. We know from the data that this encompases
        #All sigmas of interest, so we are not truncating data. Care should be taken when using one's own data. Or just use a normal distribution instead
        sig_low = 0.00001
        sig_high = 10
        group1_std = pm.Uniform("group1_std", lower=sig_low, upper=sig_high)
        group2_std = pm.Uniform("group2_std", lower=sig_low, upper=sig_high)
        #Following Kruscke's lead here
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        lambda1 = group1_std ** -2
        lambda2 = group2_std ** -2

        group1 = pm.StudentT("group1", nu=nu, mu=group1_mean, lam=lambda1, observed=DBS)
        group2 = pm.StudentT("group2", nu=nu, mu=group2_mean, lam=lambda2, observed=NoStim)
        #Get group statistics distributions of mean,stand dev, and effect size
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic("effect size", diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2))
        
    #Now that we have our model, let's start the sampling!
    with best:
        trace = pm.sample(2000, tune=500, target_accept=0.90,chains = 4)
    #Alright! Let's plot posteriors
    az.plot_posterior(trace,var_names=["group1_mean", "group2_mean", "group1_std", "group2_std"],hdi_prob=0.95,point_estimate = 'mode')
    az.plot_posterior(trace,var_names=["difference of means", "difference of stds", "effect size"],hdi_prob=0.95,point_estimate = 'mode')
    plt.show()