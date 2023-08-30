# Practical Bayesian Inference in Neuroscience: Or How I Learned To Stop Worrying and Embrace Probability
This repository is a supplement to Coventry and Bartlett's tutorial on using Bayesian methods in neuroscience studies using PyMC.
![alt text](https://github.com/bscoventry/BayesianNeuralAnalysis/blob/main/BayesSplanation.jpg?raw=true)

# OS recommendations
This code works for Windows and Linux distributions. We believe this should work for MacOS, but is untested.

# Setting up a PyMC environment
To use these scripts, we strongly recommend setting up an Anaconda environment dedicated to PyMC. Anaconda can be freely downloaded directly from https://www.anaconda.com/. To setup a PyMC environment, open an anaconda prompt and run:
```
conda create -n PyMC python=3.8
```
which creates a conda environment named PyMC using Python version 3.8. We recommend at least this version of Python, but have tested down to Python 3.6.8 as working. Anaconda environments create self constrained python libraries such that Python dependencies from one environment are not altered by changed in other project environments. 

To activate an environment, in anaconda, run
```python
conda activate PyMC
```

# Installing PyMC and required packages
To install PyMC, first activate a PyMC environment. Then run:
```python
conda install numpyro
conda install PyMC
pip install arviz
pip install aesara
pip install numpy
pip install pandas
pip install scipy
pip install seaborn
pip install matplotlib
```
This installs PyMC and all required packages individually. Alternatively, pip installations can be groups as "pip install scipy,numpy,..."

# Running sample programs
We recommend creating a new directory on your machine to download all programs and data to. To run programs, activate the PyMC environment and navigate to the code/data directory as:
```python
cd 'path to your directory'
```
An example of 'path to your directory might be 'C:\CodeRepos\PyMC'.

Programs are then run by simplying typing
```python
python BayesRegression.py
```
Programs will run and give diagnostics in the anaconda terminal and show plots found in the paper.

# Where do I find the data?
Data for Bayesian linear regressions, comparisons of models, and BANOVA/BANCOVAs are found at the following open science framework link:
Data for Bayesian Multilinear regressions can be found at its papers open science framework repository:
