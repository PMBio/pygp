"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) preprocessed by shiftCF(:py:class`covar.combinators.ShiftCF) and combined with noise
:py:class:`covar.noise.NoiseISOCF` by summing them up
(using :py:class:`covar.combinators.SumCF`).
"""

from pygp.covar import se, noise, combinators
from pygp.gp.basic_gp import GP
from pygp.priors import lnpriors
from pygp.optimize.optimize import opt_hyper
from pygp.plot import gpr_plot

import logging as LG
import numpy.random as random
#import pdb
import pylab as PL
import scipy as SP

def run_demo():
    LG.basicConfig(level=LG.INFO)
    
    random.seed(1)
    
    #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
    xmin = 1
    xmax = 2.5*SP.pi
    x1 = SP.arange(xmin,xmax,.7)
    x2 = SP.arange(xmin,xmax,.4)
    
    C = 2       #offset
    #b = 0.5
    sigma1 = 0.1
    sigma2 = 0.1
    n_noises = 1
    
    b = 0
    
    y1  = b*x1 + C + 1*SP.sin(x1)
#    dy1 = b   +     1*SP.cos(x1)
    y1 += sigma1*random.randn(y1.shape[0])
    y1-= y1.mean()
    
    y2  = b*x2 + C + 1*SP.sin(x2)
#    dy2 = b   +     1*SP.cos(x2)
    y2 += sigma2*random.randn(y2.shape[0])
    y2-= y2.mean()
    
    x1 = x1[:,SP.newaxis]
    x2 = (x2-1)[:,SP.newaxis]
    
    x = SP.concatenate((x1,x2),axis=0)
    y = SP.concatenate((y1,y2),axis=0)
    
    #predictions:
    X = SP.linspace(0,10,100)[:,SP.newaxis]
    
    #hyperparamters
    dim = 1
    replicate_indices = SP.concatenate([
        SP.repeat(i,len(xi)) for i,xi in enumerate((x1,x2))])
    n_replicates = len(SP.unique(replicate_indices))
    
    logthetaCOVAR = SP.log([1,1,SP.exp(0),SP.exp(0),sigma1])#,sigma2])
    hyperparams = {'covar':logthetaCOVAR}
    
    SECF = se.SEARDCF(dim)
    #noiseCF = noise.NoiseReplicateCF(replicate_indices)
    noiseCF = noise.NoiseISOCF()
    shiftCF = combinators.ShiftCF(SECF,replicate_indices)
    CovFun = combinators.SumCF((shiftCF,noiseCF))
    
    covar_priors = []
    #scale
    covar_priors.append([lnpriors.lngammapdf,[1,2]])
    for i in range(dim):
        covar_priors.append([lnpriors.lngammapdf,[1,1]])
    #shift
    for i in range(n_replicates):
        covar_priors.append([lnpriors.lngausspdf,[0,.5]])    
    #noise
    for i in range(n_noises):
        covar_priors.append([lnpriors.lngammapdf,[1,1]])
    
    priors = {'covar':covar_priors}
    Ifilter = {'covar': SP.array([1,1,1,1,1],dtype='int')}
    
    gpr = GP(CovFun,x=x,y=y) 
    [opt_model_params,opt_lml] = opt_hyper(gpr,hyperparams,priors=priors,gradcheck=True,Ifilter=Ifilter)
    
    #predict
    [M,S] = gpr.predict(opt_model_params,X)
    
    T = opt_model_params['covar'][2:4]
    gpr_plot.plot_sausage(X,M,SP.sqrt(S))
    gpr_plot.plot_training_data(x,y,shift=T,replicate_indices=replicate_indices)
    PL.show()
