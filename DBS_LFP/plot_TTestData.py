#---------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 12/23/23
# Purpose: Just to help plot t-test comparison data
#---------------------------------------------------------------------
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import pdb
datfile = open('ttestdata.p','rb')
ttestDict = pl.load(datfile)
datfile.close()
pval = ttestDict['pvalResult']
best = ttestDict['bestMap']

fig, ax1 = plt.subplots()
Ns = range(2,2000,5)
color = 'tab:blue'
ax1.set_xlabel('Trials')
ax1.set_ylabel('P-Value', color=color)
ax1.plot(Ns[0:201], pval[0:201], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=0.05, color='k', linestyle='--')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('BEST', color=color)  # we already handled the x-label with ax1
ax2.plot(Ns[0:201], best[0:201], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()