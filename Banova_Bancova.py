import numpy as np               #Numpy for numerical 'ala Matlab' computations
import pymc as pm                #pymc will be doing most of our heavy lifting for Bayesian calculations
import matplotlib.pyplot as plt  #This works as our plotting tool
import arviz as az               # arviz is a visualization package for plotting probabilistic variables, such as prior or posterior distributions
import aesara                    #Aesara is out tool for calculations involving tensors. PyMC will mostly work with this for us.
import pandas as pd              #Using pandas to read in CSV data files
import pdb
import seaborn as sns
plt.style.use('seaborn-white')   #Set plot styles here
color = '#87ceeb'
if __name__ == '__main__':       #This statement is to allow for parallel sampling in windows. Linux distributions don't require this.
    print(f"Running on PyMC v{pm.__version__}")      #Tells us which version of PyMC we are running
    AMDepthData = pd.read_csv("depthData.csv")      #Read in our data. Note this address may change based on where program files are downloaded.
    """
    Now let's setup some meta data for our model analyses. We will set the number of burn in samples, which "primes" the markov chain Monte Carlo (MCMC) algorithm, and number of
    samples to draw from the posterior. In general, less "well behaved" data will require more samples to get MCMC to converge to steady state. 
    """
    numBurnIn = 2000
    numSamples = 5000
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
    
    modDepth = AMDepthData.ModDepth.values                    #This is our modulation depth vector
    
    firingRate = AMDepthData['TotMean']              #This is our mean firing rate. Note, data can be accessed in a pandas array using dot notation (data.subsetData) or
    firingRate = np.log(firingRate+0.01)             #Posterior predictive checks dictate that a log transformation on firing rates best fits a BANCOVA model
    AMDepthData['TotMean']  = firingRate             #Add firing rate back into pandas array for ease of data access. In certain cases, may want to keep these seperate
    #Get y variable (firing rates) mean and sd across groups for setting minimally informative prior parameters
    yMean = np.mean(firingRate) 
    yStDv = np.std(firingRate)
    modDepthMean = np.mean(modDepth)
    #Grab age class. In this data set, Age is already cast to encoded integers. 1 is aged, 2 is young.
    ageNum = AMDepthData['Age'].astype(aesara.config.floatX)
    
    age = ageNum - 1               #0 and 1s instead of 1 and 2 for categories. Because computer science and we like computer science
    numCategories = 2
    #The following two lines adds a category option of age to the dataframe. This is because we've built this to use Categories, and I want to demonstrate how to easily cast objects to this type
    #If the data being used already isn't in this form.
    AMDepthData['AgeClass'] = age
    AMDepthData.AgeClass = AMDepthData.AgeClass.astype('category')
    ClassAge = AMDepthData.AgeClass.cat.codes.values
    AMDepthData['LogModDepth'] = modDepth

    def plot_cred_lines(b0, bj, bcov, x, ax):
        #This function plots the covariance variable at the MAP beta values. Used later for plotting
        B = pd.DataFrame(np.c_[np.transpose(b0.values), np.transpose(bj.values), np.transpose(bcov.values)], columns=['beta0', 'betaj', 'betacov'])
        
        # Credible posterior prediction lines
        hpd_interval = az.hdi(B.values, hdi_prob=0.95)
        B_hpd = B[B.beta0.between(*hpd_interval[0,:]) & 
                B.betaj.between(*hpd_interval[1,:]) &
                B.betacov.between(*hpd_interval[2,:])] 
        xrange = np.linspace(x.min(), x.max())
        
        for i in np.random.randint(0, len(B_hpd), 10):
            ax.plot(xrange, B_hpd.iloc[i,0]+B_hpd.iloc[i,1]+B_hpd.iloc[i,2]*xrange, c=color, alpha=.6, zorder=0)
    """
    Define the ANCOVA Model
    """
    
    with pm.Model() as BANCOVA:
        #Define hyperprior on sigma
        bSigma = pm.HalfCauchy('bSigma',2.0)                  #Recommended by Gelman, this parameter doesn't overemphasize 0 on sigma.
        #Define Prior, likelihood distributions. Relatively noninformative 
        a = pm.Normal('a',yMean,sigma = np.sqrt(yStDv))
        B = pm.Normal('B',0,sigma=bSigma,shape=numCategories)
        #Define covariate, in our case this is mod depth, but we initially set mean and variance to that of observed firing rates. Will learn with data.
        Bcov = pm.Normal('Bcov',yMean,sigma = np.sqrt(yStDv))
        #variance on likelihood function is set to uniform to satisfy homogeneity of variance assumption of ANOVA like structures.
        sigmaLikelihood = pm.Uniform('sigmaLikelihood',yStDv/100,yStDv*10)
        #Set the BANCOVA model. Note that [ClassAge] effectively sets seperate Beta parameters per each class, which is what we want.
        BancovaModel = a + B[ClassAge] + (Bcov*(modDepth - modDepthMean))
        #Set likelihood on BANCOVA model
        y = pm.Normal('y',mu=BancovaModel,sigma = yStDv,observed=firingRate)
        #Now, make sure model coefficients sum to 0 to create an ANOVA-like structure
        aScaled = pm.Deterministic('aScaled',a+aesara.tensor.mean(B) + Bcov*(-modDepthMean))
        bScaled = pm.Deterministic('bScaled',B - aesara.tensor.mean(B))
    pm.model_to_graphviz(BANCOVA)                           #This creates a visualizer of our model. 
    plt.show()                                              #Go ahead and plot our model.

    """
    Our last step is to run inference to get our posterior distribution which we do by MCMC sampling. In PyMC this is also easy.
    """
    with BANCOVA:                 #Access our defined model
        trace = pm.sample(numSamples, tune=numBurnIn, target_accept=0.90,chains = 1)       
    
    #Now that we have our trace, time to plot raw data for analysis.
    fg = sns.FacetGrid(AMDepthData, col='AgeClass', despine=False)
    fg.map(plt.scatter, 'LogModDepth', 'TotMean', facecolor='none', edgecolor='b')
    #Grab our Beta posteriors for young and age classes respectively. 
    bScaledAge = trace.posterior['bScaled'][:,:,0]
    bScaledYoung = trace.posterior['bScaled'][:,:,1]
    #np.shape(trace)                             #Uncomment this line if you want to see the overall structure of the trace. Good for diagnostics
    aTrace = trace.posterior['aScaled']
    
    
    cTrace = trace.posterior['Bcov']
    scale = trace.posterior['sigmaLikelihood']
    
    #We perform inference over the contrasts between groups. In this case lets look at differences between aged and young classes.
    yVa = bScaledAge-bScaledYoung              #Difference calculation
    effectSize = yVa/scale      #Get the effect size

    #Plot differences
    fig, axes = plt.subplots(2,1)
    az.plot_posterior(yVa, hdi_prob=0.95, ref_val=0, ax=axes[0],point_estimate='mode')
    az.plot_posterior(effectSize, hdi_prob=0.95, ref_val=0, ax=axes[1],point_estimate='mode')

    
    dataY = AMDepthData.loc[AMDepthData['AgeClass'] == 'Young']                    #Get Young and Aged data
    dataY.reset_index(drop=True, inplace = True)
    dataA = AMDepthData.loc[AMDepthData['AgeClass'] == 'Aged']                   
    dataA.reset_index(drop=True, inplace = True)

    
    fg = sns.FacetGrid(AMDepthData, col='AgeClass', despine=False)
    fg.map(plt.scatter, 'LogModDepth', 'TotMean', facecolor='none', edgecolor='b')
    #Plot cred lines of 95% HDI of Beta class and Beta cov parameters
    for ck, ax in enumerate(fg.axes.flatten()):
        if ck == 0:
            plot_cred_lines(aTrace,
                            bScaledAge,
                            cTrace,
                            AMDepthData.ModDepth, ax)
        if ck == 1:
            plot_cred_lines(aTrace,
                            bScaledYoung,
                            cTrace,
                            AMDepthData.ModDepth, ax)
    ax.set_xticks(np.arange(.6, 1.1, .1))
    #Look at the data!
    plt.show(block=False)
    """
    Here we perform posterior predictive checks and get Bayesian P-Values.
    """
    with BANCOVA:
        ppcBANCOVA = pm.sample_posterior_predictive(trace, random_seed=Randomseed)
    #The above code envokes the regression mode, then uses the posterior from the trace, pulling synthetic samples to compare to observed. Random seed is set so that each run can be perfectly replicated
    az.plot_bpv(ppcBANCOVA, hdi_prob=0.95,kind='p_value')
    #Bayes p-values, similar to frequentist,can be used to assess if posterior predictive is sufficiently close to observed density. Should be centered around 0.50.
    az.plot_ppc(ppcBANCOVA)
    plt.show(block=False)
    #Pause the program in case we want to do inline exploration of data. type exit() in terminal to end program.
    pdb.set_trace()
    
