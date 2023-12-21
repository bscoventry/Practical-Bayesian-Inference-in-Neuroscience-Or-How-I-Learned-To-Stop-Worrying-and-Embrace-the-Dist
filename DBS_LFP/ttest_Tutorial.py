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
from scipy.stats import mode
#Let's define the Bayesian Estimates Supersedes the T-Test (BEST) model.
if __name__ == '__main__':
    def BEST(group_1,group_2):
        totalData = pd.DataFrame(
        dict(value=np.r_[group_1, group_2], group=np.r_[["group1"] * len(group_1), ["group2"] * len(group_2)]))
        #totalData = np.concatenate(group1,group2)
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

            group1 = pm.StudentT("group1", nu=nu, mu=group1_mean, lam=lambda1, observed=group_1)
            group2 = pm.StudentT("group2", nu=nu, mu=group2_mean, lam=lambda2, observed=group_2)
            #Get group statistics distributions of mean,stand dev, and effect size
            diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
            diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
            effect_size = pm.Deterministic("effect size", diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2))
        with best:
            trace = pm.sample(2000, tune=2000, target_accept=0.90,chains = 4)
        
        diff_of_means_hdi =  az.hdi(trace.posterior, var_names=["difference of means"],hdi_prob=0.95) 
        diff_of_means_hdi  =  diff_of_means_hdi['difference of means'].data
        dataSummary = az.summary(trace.posterior,var_names=["difference of means"],hdi_prob=0.95)
        gatherChains = np.squeeze(np.reshape(trace.posterior['difference of means'].data,(1,2000*4)))
        MAPest = mode(gatherChains)[0][0]
        
        if diff_of_means_hdi[0] < 0 <= diff_of_means_hdi[1]:
            acceptNull = True
        else:
            acceptNull = False
        effect_size_hdi =  az.hdi(trace.posterior, var_names=["effect size"],hdi_prob=0.95)
        return acceptNull,MAPest
    #Okay, lets load in our data
    with open('BetaPowerStim.p','rb') as f:
        BetaStim = pl.load(f)
    with open('BetaPowerNoStim.p','rb') as f:
        BetaNoStim = pl.load(f)
    #diffmeans = BEST(BetaStim,BetaNoStim)
    numComparisons = len(BetaStim)          #Equal group sizes, just want to shuffle number of n's
    bestMap = []
    bestResult = []
    pvalResult = []
    ttestcounter = 0
    BestCounter = 0
    ttestTotal = []
    BestTotal = []
    for ck in range(2,numComparisons,5):
        
        betaStimCur = BetaStim[0:ck]
        betaNoStimCur = BetaNoStim[0:ck]
        Result,MAPest = BEST(betaStimCur,betaNoStimCur)  
        bestMap.append(MAPest)
        if Result == False:
            BestCounter = BestCounter + 1
        bestResult.append(Result)
        BestTotal.append(BestCounter)
        t_test = ttest_ind(betaStimCur,betaNoStimCur)
        pval = t_test[1]
        pvalResult.append(pval)
        if pval < 0.05:
            correct = True
        else:
            ttestcounter = ttestcounter+1
        ttestTotal.append(ttestcounter)
        print(ck)
    pdb.set_trace()
    Ns = range(2,numComparisons)
    plt.plot(Ns,BestTotal)
    plt.plot(Ns,ttestTotal)
    plt.show()
    plt.plot(Ns,pvalResult)
    plt.plot(Ns,bestMap)
    plt.show()

        

