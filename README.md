# BayesianNeuralAnalysis
This repository is a supplement to Coventry and Bartlett's tutorial on using Bayesian methods in neuroscience studies using PyMC.

# OS recommendations
This code works for Windows and Linux distributions. We believe this should work for MacOS, but is untested.

# Setting up a PyMC environment
To use these scripts, we strongly recommend setting up an Anaconda environment dedicated to PyMC. Anaconda can be freely downloaded directly from \url{https://www.anaconda.com/}. To setup a PyMC environment, open an anaconda prompt and run:
\begin{code}
conda create -n PyMC python=3.8
\end{code}
which creates a conda environment named PyMC using Python version 3.8. We recommend at least this version of Python, but have tested down to Python 3.6.8 as working. Anaconda environments create self constrained python libraries such that Python dependencies from one environment are not altered by changed in other project environments. 

To activate an environment, in anaconda, run
\begin{code}
conda activate PyMC
\end{code}

