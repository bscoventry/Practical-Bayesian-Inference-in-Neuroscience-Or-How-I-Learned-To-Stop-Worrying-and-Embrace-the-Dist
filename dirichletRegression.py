#--------------------------------------------------------------------------------------------------------------------------------------------
# Authors: Brandon S Coventry            Wisconsin Institute for Translational Neuroengineering
# Date: 04/22/2024                       Wisconsin is springlike? Finally?
# Purpose: This performs dirichlet regression for wave decomposition
# Revision History: Will be tracked in Github.
# Notes: N/A
#--------------------------------------------------------------------------------------------------------------------------------------------
# Imports go here. Global use imports will be put here, we will upload specifics as needed.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import arviz as az
import pymc as pm
import aesara
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle # python3
import seaborn as sns
import matlab.engine
import jax

numSamples = 5000
numBurnIn = 2000
randomSeed = 77
def wavePreProcess(wave):
    waveVec = np.zeros((8,))
    [nSamps,nCols] = np.shape(wave)
    waveHold = np.zeros((nSamps,))
    for ck in range(nSamps):
        waveHold[ck] = float(wave[ck][0])*0.01
    waveVec[0:7] = waveHold[0:7]
    #Take care of non-wave components to ensure all sums to 1.
    waveVec[7] = np.sum(waveHold[7:nSamps])
    return waveVec

if __name__ == '__main__': 
    #Define the model
    df = pd.read_pickle('LFPVelocity.pkl')
    
    N = df.shape[0]
    epp = df['EnergyPerPulse'].values
    velocity = df.Velocity.values
    prct = df['prctSVD']
    PCAdims = 8
    x_expt_cond = np.zeros((N,PCAdims))
    for ck in range(len(epp)):
        curEpp = epp[ck]*np.ones((PCAdims,))
        x_expt_cond[ck,:] = curEpp

    waveDecomp = np.zeros((N,PCAdims))
    for ck in range(N):
        curWave = prct[ck]
        waveDecomp[ck,:] = wavePreProcess(curWave)
    WP = [f"wp-{i}" for i in range(PCAdims)]
    REPS = [f"{epp}-{i}" for i in range(N)]
    coords = {"wp": WP, "energy": REPS}
    intercept = np.ones_like((N,))        #Intercept the size of number of datapoints, N.
    
    with pm.Model(coords=coords) as dirichlet_reg:
        
        b0 = pm.Normal("b0", 0, 2.5)
        b1 = pm.Normal("b1", 0, 2.5, dims=("wp",))
        eta = pm.Deterministic("eta",b0 + b1[None, :] * x_expt_cond, dims=("energy", "wp"))
        mu = pm.Deterministic("mu", pm.math.exp(eta), dims=("energy", "wp"))
        y = pm.Dirichlet("y", mu, observed=waveDecomp, dims=("energy", "wp"))
    with dirichlet_reg:
        step = pm.NUTS()
        rTrace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4,nuts_sampler="numpyro")
        pm.compute_log_likelihood(rTrace)
        ppc = pm.sample_posterior_predictive(rTrace, random_seed=randomSeed)
    
    #az.summary(rTrace, var_names=["b0", "b1"], hdi_prob=0.95)
    intercept = rTrace.posterior["b0"]                #Grab the posterior distribution of a
    EnergySlope = rTrace.posterior["b1"] 
    pm.plot_posterior(intercept, point_estimate='mode',hdi_prob=0.95)
    pm.plot_posterior(EnergySlope, point_estimate='mode',hdi_prob=0.95)
    az.plot_bpv(ppc, hdi_prob=0.95,kind='p_value')
    az.plot_ppc(ppc)
    plt.show()
    pdb.set_trace()
    plt.scatter(epp,velocity)
    with pm.Model() as linear_reg:
        
        a = pm.Normal('a', mu=0, sigma = 5)           #Normally distributed prior on a
        B = pm.Normal('B', mu=0, sigma = 5)           #Normally distributed prior on B
        eps = pm.HalfCauchy("eps", 5)                 #Model error prior, half Cauchy distributed with variance 5
        nu = pm.Exponential("nu", 5) 
        # Now we define our likelihood function, which for regression is our regression function
        reg = pm.Deterministic('reg', a + B*epp)      #Deterministic is for non probabilistic data. This is a modification to help sampling, inference is still probabilistic
        likelihood = pm.StudentT('Y',mu = reg, sigma = eps, observed = velocity)    
    with linear_reg:
        step = pm.NUTS()
        rTrace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 4,nuts_sampler="numpyro")
        pm.compute_log_likelihood(rTrace)
        ppc = pm.sample_posterior_predictive(rTrace, random_seed=randomSeed)
    
    intercept = rTrace.posterior["a"]                #Grab the posterior distribution of a
    EnergySlope = rTrace.posterior["B"] 
    pm.plot_posterior(intercept, point_estimate='mode',hdi_prob=0.95)
    pm.plot_posterior(EnergySlope, point_estimate='mode',hdi_prob=0.95)
    az.plot_bpv(ppc, hdi_prob=0.95,kind='p_value')
    az.plot_ppc(ppc)
    plt.show()
    pdb.set_trace()