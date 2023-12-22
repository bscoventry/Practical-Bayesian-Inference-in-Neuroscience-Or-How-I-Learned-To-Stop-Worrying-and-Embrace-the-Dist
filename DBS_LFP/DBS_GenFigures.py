from mfm import MFM            #This is the mean-field model of DBS-LFP. See mfm for details
import numpy as np
import matplotlib.pyplot as plt   #Helpful for plotting
import pdb
import pickle as pl
with open('freqVals.p','rb') as f:
        freq = pl.load(f)
#Keywards to set up models
kwargsDBSSub = {                        #Dopamine depleted, continuous DBS, 100us pulsewidth
        'cDBS': True,
        'cDBS_amp': 0.01,
        'cDBS_f': 130,
        'cDBS_width': 100,
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
kwargsDBS = {                        #Dopamine depleted, continuous DBS, 100us pulsewidth
        'cDBS': True,
        'cDBS_amp': 1.8,
        'cDBS_f': 130,
        'cDBS_width': 240,
        'DD' : True,
        'verbose' : False,
        'tstop'   : 15 
    }
# Let's set up the model!
DBSSub_LFP = MFM(**kwargsDBSSub)                        #Model that contains subthreshold DBS, ie should be identical in a PD model to no stim
Stimless_LFP = MFM(**kwargsStimless)                   #Model that recieves no stimulation whatsoever
DBS_LFP = MFM(**kwargsDBS)                              #Model well driven DBS Stimulation
def calcBetaPower(PXX):
    betaband = [27,62]
    beta = PXX[betaband[0]:betaband[1]]
    deltabeta = 0.48828125
    linBeta = np.power(10,beta/10)
    betapower = 2*np.sum(linBeta*deltabeta)
    return betapower
DBS_LFP.run()
DBSSub_LFP.run()
Stimless_LFP.run()
pxxDBS= DBS_LFP.getPXX()
pxxDBSSub = DBSSub_LFP.getPXX()
pxxStimless = Stimless_LFP.getPXX()
plt.plot(freq[0:205],pxxDBS[0:205])
plt.plot(freq[0:205],pxxDBSSub[0:205],marker='o')
plt.plot(freq[0:205],pxxStimless[0:205],marker='+')
plt.axvline(x = 13, color = 'b', label = 'axvline - full height')
plt.axvline(x = 30, color = 'b', label = 'axvline - full height')
plt.show()
pdb.set_trace()