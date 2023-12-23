#----------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 12/16/2023
# Purpose: Generate data to address asymptotics of Bayesian vs Frequentist inference on a DBS-LFP model
#----------------------------------------------------------------------------------------------------------
from mfm import MFM            #This is the mean-field model of DBS-LFP. See mfm for details
import numpy as np
import matplotlib.pyplot as plt   #Helpful for plotting
import pdb
#Keywards to set up models
kwargsDBS = {                        #Dopamine depleted, continuous DBS, 100us pulsewidth
        'cDBS': True,
        'cDBS_amp': 0.01,   #1.8 if doing driven responses
        'cDBS_f': 130,
        'cDBS_width': 100,  #240 if doing driven responses 
        'DD' : True,
        'verbose' : False,
        'tstop'   : 15 
    }
kwargsStimless = {                    #Dopamine depleted, no stim
        'cDBS': False,
        'DD' : True,
        'verbose' : False,
        'tstop'   : 15
    }
# Let's set up the model!
DBS_LFP = MFM(**kwargsDBS)                        #Model that contains subthreshold DBS, ie should be identical in a PD model to no stim
Stimless_LFP = MFM(**kwargsStimless)                   #Model that recieves no stimulation whatsoever
# Let's setup some data generating parameters
numtrials = 2000                     #Number of trials to run the model.
BetaPower_NoStim = np.zeros((numtrials,))
BetaPower_Stim = np.zeros((numtrials,))
#Some helpful defs here to calculate beta band power
def calcBetaPower(PXX):
    betaband = [27,62]
    beta = PXX[betaband[0]:betaband[1]]
    deltabeta = 0.48828125
    linBeta = np.power(10,beta/10)
    betapower = 2*np.sum(linBeta*deltabeta)
    return betapower
#Get our data!
for ck in range(numtrials):
    DBS_LFP.run()
    newPXXStim = DBS_LFP.getPXX()
    BetaPower_Stim[ck] = calcBetaPower(newPXXStim)
    Stimless_LFP.run()
    newPXXStimless = Stimless_LFP.getPXX()
    BetaPower_NoStim[ck] = calcBetaPower(newPXXStimless)
    #Need to clear model vars
    DBS_LFP = MFM(**kwargsDBS)                        
    Stimless_LFP = MFM(**kwargsStimless)
pdb.set_trace()
import pickle as pl
pl.dump(BetaPower_Stim, open( "BetaPowerStim.p", "wb" ) )
pl.dump(BetaPower_NoStim, open( "BetaPowerNoStim.p", "wb" ) )
