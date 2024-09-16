from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pandas as pd
from patsy import dmatrix
import pdb
import scipy.io as sio
from scipy import stats
if __name__ == "__main__":
    RANDOM_SEED = 777
    az.style.use("arviz-darkgrid")

    df = pd.read_pickle('SFC.pkl')
    EPP = df.EnergyPerPulse.values
    EPP = [float(i) for i in EPP]
    EPP = np.array(EPP)
    EPP[EPP<=0] = 0
    lepp = np.log(EPP+0.001)
    df['lepp'] = lepp
    df = df.loc[df.lepp>=-4.5]
    df.reset_index(drop=True, inplace = True)
    freqs = sio.loadmat('SFCFreqs.mat')
    freqs = freqs['freqs']
    freqs = np.squeeze(freqs)
    freqsWhere = np.where(freqs<=200)
    freqsWhere = freqsWhere[0]
    nRows = len(df)
    thetaCoh = np.nan*np.ones((nRows,))
    alphaCoh = np.nan*np.ones((nRows,))
    betaCoh = np.nan*np.ones((nRows,))
    lgCoh = np.nan*np.ones((nRows,))
    hgCoh = np.nan*np.ones((nRows,))
    freqVec = np.nan*np.ones((nRows,))
    uniqueISIs = np.unique(df.ISI.values)
    numISIs = len(uniqueISIs)
    for ck in range(nRows):
        curSFC = df.meanSFC[ck]
        curSFC = curSFC[freqsWhere]
        thetaCoh[ck] = curSFC[1]
        alphaCoh[ck] = curSFC[2]
        betaCoh[ck] = np.max(curSFC[3:6])
        #betaCoh[ck] = np.max(curSFC[3])
        lgCoh[ck] = np.max(curSFC[6:14])
        #lgCoh[ck] = np.max(curSFC[6])
        hgCoh[ck] = np.max(curSFC[14:34])
        #hgCoh[ck] = np.max(curSFC[14])
        maxSFC = np.nanmax(curSFC)
        
        freqLoc = np.where(curSFC==maxSFC)
        freqVec[ck] = freqsWhere[freqLoc]

    df['thetaCoh'] = thetaCoh
    df['alphaCoh'] = alphaCoh
    df['betaCoh'] = betaCoh
    df['lgCoh'] = lgCoh
    df['hgCoh'] = hgCoh
    df['freqVec'] = freqVec
    df['lISI'] = np.log(df.ISI+0.01)
    #Define knots
    num_knots = 1
    knot_list = [-1.4627]#[-1.4629]#np.quantile(df.lepp, np.linspace(0, 1, num_knots))
    knot_list
    #Setup Patsy B-Matrix for Splines
    B = dmatrix(
        "bs(lepp, knots=knots, degree=1, include_intercept=True) - 1",
        {"lepp": df.lepp.values, "knots": knot_list[1:-1]},
    )
    B

    spline_df = (
        pd.DataFrame(B)
        .assign(lepp=df.lepp.values)
        .melt("lepp", var_name="spline_i", value_name="value")
    )

    color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

    fig = plt.figure()
    for i, c in enumerate(color):
        subset = spline_df.query(f"spline_i == {i}")
        subset.plot("lepp", "value", c=c, ax=plt.gca(), label=i)
    plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6)

    #Setup regression model
    COORDS = {"splines": np.arange(B.shape[1])}

    with pm.Model(coords=COORDS) as spline_model:
        a = pm.Normal("a", 0, 1)
        w = pm.Normal("w", mu=0, sigma=1, size=B.shape[1], dims="splines")
        mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
        sigma = pm.Exponential("sigma", 1)
        nu = pm.HalfCauchy("nu",1)
        D = pm.StudentT("D", mu=mu,nu=nu, sigma=sigma, observed=df.lgCoh)

    with spline_model:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(draws=1500, tune=500, random_seed=RANDOM_SEED, chains=2,target_accept=0.95,nuts_sampler='numpyro'))
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    
    
    wp = idata.posterior["w"].mean(("chain", "draw")).values

    spline_df = (
        pd.DataFrame(B * wp.T)
        .assign(lepp=df.lepp.values)
        .melt("lepp", var_name="spline_i", value_name="value")
    )

    spline_df_merged = (
        pd.DataFrame(np.dot(B, wp.T))
        .assign(lepp=df.lepp.values)
        .melt("lepp", var_name="spline_i", value_name="value")
    )


    color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))
    fig = plt.figure()
    for i, c in enumerate(color):
        subset = spline_df.query(f"spline_i == {i}")
        subset.plot("lepp", "value", c=c, ax=plt.gca(), label=i)
    spline_df_merged.plot("lepp", "value", c="black", lw=2, ax=plt.gca())
    plt.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)

    for knot in knot_list:
        plt.gca().axvline(knot, color="grey", alpha=0.4)
    
    def median_sd(x):
        median = np.percentile(x, 50)
        sd = np.sqrt(np.mean((x-median)**2))
        return sd
    func_dict = {
        "std": np.std,
        "median_std": median_sd,
        "hdi_5%": lambda x: np.percentile(x, 5),
        "median": lambda x: np.percentile(x, 50),
        "hdi_95%": lambda x: np.percentile(x, 95),
        #"mode":    lambda x: stats.mode(x)
    }
    
    post_pred = az.summary(idata, var_names=["mu"],stat_funcs=func_dict).reset_index(drop=True)
    Band_data_post = df.copy().reset_index(drop=True)
    Band_data_post["pred_mean"] = post_pred["mean"]
    Band_data_post["pred_hdi_lower"] = post_pred["hdi_5%"]
    Band_data_post["pred_hdi_upper"] = post_pred["hdi_95%"]

    df.plot.scatter(
    "lepp",
    "lgCoh",
    color="cornflowerblue",
    s=10,
    title="lgCoherence data with posterior predictions",
    ylabel="Coherence",
    )
    
    for knot in knot_list:
        plt.gca().axvline(knot, color="grey", alpha=0.4)

    Band_data_post.plot("lepp", "pred_mean", ax=plt.gca(), lw=3, color="firebrick")
    plt.fill_between(
        Band_data_post.lepp,
        Band_data_post.pred_hdi_lower,
        Band_data_post.pred_hdi_upper,
        color="firebrick",
        alpha=0.4,
    )
    pm.plot_posterior(idata.posterior["w"], point_estimate='mode',hdi_prob=0.95)
    pm.plot_posterior(idata.posterior["a"], point_estimate='mode',hdi_prob=0.95)
    plt.show()
    pdb.set_trace()