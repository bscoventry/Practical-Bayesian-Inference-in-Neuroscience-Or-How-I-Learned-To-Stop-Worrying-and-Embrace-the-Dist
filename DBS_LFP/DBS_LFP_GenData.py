#----------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 12/16/2023
# Purpose: Generate data to address asymptotics of Bayesian vs Frequentist inference on a DBS-LFP model
#----------------------------------------------------------------------------------------------------------
from mfm import MFM            #This is the mean-field model of DBS-LFP. See mfm for details
import numpy as np
import matplotlib.pyplot as plt   #Helpful for plotting
# Let's set up the model!
DBS_LFP = MFM()                        #Model that contains subthreshold DBS, ie should be identical in a PD model to no stim
Stimless_LFP = MFM()                   #Model that recieves no stimulation whatsoever
# Let's setup some data generating parameters
numtrials = 1000                       #Number of trials to run the model.

