#-----------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry,           Wisconsin Institute for Translational Neuroengineering
# Date: 04/02/2024                      Wisconsin is literally having a snowstorm. Difficult to drive, about a foot of snow. INAPRIL
# Purpose: This demonstrates the stopping problem in frequentist approaches.
#-----------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.special import comb       #n choose k
#Let's define our experimental parameters
N = 1000
z = 430          #D2 count
theta = 0.459    #From previous literature
def binomialDist(N,z,theta):
    bdist = comb(N,z)*np.power(theta,z)*np.power((1-theta),N-z)
    return bdist
def negBinDist(N,z,theta):
    nbdist = (z/N)*(comb(N,z)*np.power(theta,z)*np.power((1-theta),N-z))
    return nbdist

#Calculate some p-values given our results. We need to calculate probability of getting results at least as extreme as observed. That means p of everything <= 430
pValBin = 0
pValNeg = 0
lessthanlist = np.arange(0,431,1)
for ck in range(len(lessthanlist)):
    pValBin = pValBin + binomialDist(N,lessthanlist[ck],theta)
    pValNeg = pValNeg + negBinDist(N,lessthanlist[ck],theta)

print('Pvalue Binomial'+ ' '+str(pValBin))
print('Pvalue Negative Binomial'+ ' '+str(pValNeg))
#Okay, so let's plot the distributions
zbyNBin = np.arange(0,1000,5)
zbyNBinNorm = np.arange(0,1,1/len(zbyNBin))

BinDist = np.zeros((len(zbyNBin,)))
for ck in range(len(zbyNBin)):
    BinDist[ck] = binomialDist(1000,zbyNBin[ck],theta)
plt.stem(zbyNBinNorm,BinDist)
plt.plot(zbyNBinNorm,BinDist)
NegDist = np.zeros((len(zbyNBin,)))
for ck in range(len(zbyNBin)):
    NegDist[ck] = binomialDist(zbyNBin[ck],z,theta)
plt.stem(zbyNBinNorm,NegDist,'r')
plt.plot(zbyNBinNorm,NegDist,'r')
plt.show()

#Hypothetical dists
plt.bar([0,1],[0.541,0.459])
plt.show()
pdb.set_trace()