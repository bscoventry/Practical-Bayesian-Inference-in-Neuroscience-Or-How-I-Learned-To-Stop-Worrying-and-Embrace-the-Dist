#-----------------------------------------------------------------------------------------
#Author: Brandon S Coventry
#Date: 12/18/23
#Purpose: Demonstrate the role of sample size on two comparison t-tests and Bayes t-tests
#Requires: Data from DBS_LFP_GenData.py
#-----------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import arviz as az
import pdb
import pickle as pl
#Let's define the Bayesian Estimates Supersedes the T-Test (BEST) model.
def BEST(group1,group2):
    totalData = pd.DataFrame(
    dict(value=np.r_[group1, group2], group=np.r_[["group1"] * len(group1), ["group2"] * len(group2)]))
    totalData = np.concatenate(group1,group2)
    group_mu = totalData.value.mean()
    group_sigma = totalData.value.std() * 2
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

        group1 = pm.StudentT("group1", nu=nu, mu=group1_mean, lam=lambda1, observed=group1)
        group2 = pm.StudentT("group2", nu=nu, mu=group2_mean, lam=lambda2, observed=group2)
        #Get group statistics distributions of mean,stand dev, and effect size
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic("effect size", diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2))
    with best:
        trace = pm.sample(5000, tune=5000, target_accept=0.90,chains = 4)
    pdb.set_trace()
    diff_of_means_hdi =  az.hdi(trace, var_names=["diff_of_means"],hdi_prob=0.95) 
    effect_size_hdi =  az.hdi(trace, var_names=["effect_size"],hdi_prob=0.95)
    return diff_of_means_hdi
#Okay, lets load in our data
with open('BetaPowerStim.p','rb') as f:
    BetaStim = pl.load(f)
with open('BetaPowerNoStim.p','rb') as f:
    BetaNoStim = pl.load(f)
diffmeans = BEST(BetaStim,BetaNoStim)
