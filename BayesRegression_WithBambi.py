#---------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Purpose: Simple Bayesian Regression with PyMC and BAMBI. Need to install Bambi for this to work.
# Note: We recommend going through the modeling process first with PyMC before this model. Bambi is very good but abstracts
#       much of the model building context away. Use ony if comfortable with Bayesian Models.
#---------------------------------------------------------------------------------------------------------------------------
"""
To begin, let's import all of our dependencies, including our data and python packages
"""
import numpy as np               #Numpy for numerical 'ala Matlab' computations
import pymc as pm                #pymc will be doing most of our heavy lifting for Bayesian calculations
import matplotlib.pyplot as plt  #This works as our plotting tool
import arviz as az               # arviz is a visualization package for plotting probabilistic variables, such as prior or posterior distributions
import bambi as bmb              #Model abstraction tool
import pandas as pd              #Database tool
import pdb
# Let's load some data!
if __name__ == '__main__':       #This statement is to allow for parallel sampling in windows. Linux distributions don't require this. Doesn't hurt Linux to have this either.
    print(f"Running on PyMC v{pm.__version__}")      #Tells us which version of PyMC we are running
    AMDepthData = pd.read_csv("depthData.csv")      #Read in our data. Note this address may change based on where program files are downloaded.
    data1 = AMDepthData.loc[AMDepthData['Age'] == 1]                   #Let's consider young responses first.
    data1.reset_index(drop=True, inplace = True)
    AMDepthData = data1
    AMDepthData['TotMean'] = np.log(AMDepthData['TotMean']+0.01)
    #Define the model
    studT = bmb.Model("TotMean ~ ModDepth",AMDepthData,family="t")              #Defaults to Gauss, use t for student t likelihood
    #Run the MCMC
    Robust_Reg  = studT.fit(draws=5000, idata_kwargs={"log_likelihood": True})
    studT.predict(Robust_Reg, kind="pps")
    
    #Now do some plotting
    plt.figure(figsize=(7, 5))
    # Plot Data
    plt.plot(AMDepthData['ModDepth'], AMDepthData['TotMean'], "x", label="data")
    # Plot recovered linear regression
    x_range = np.linspace(AMDepthData['ModDepth'], AMDepthData['ModDepth'], 2000)
    y_pred = Robust_Reg.posterior.ModDepth.mean().item() * x_range + Robust_Reg.posterior.Intercept.mean().item()
    plt.show(block=False)

    #So as we can see, quick and easy model running at the cost of interpretability. For example, priors aren't exactly the most transparent.
