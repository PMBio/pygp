"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) combined with noise
:py:class:`covar.noise.noiseCF` by summing them up
(using :py:class:`covar.combinators.sumCF`).
"""

import logging as LG
import numpy.random as random

from pygp.gp import GP
from pygp.covar import se, noise, combinators

import pygp.opt as opt
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors

import pylab as PL
import scipy as SP

def run_demo():
    LG.basicConfig(level=LG.INFO)
    
    random.seed(1)
    
    #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
    xmin = 1
    xmax = 2.5*SP.pi
    x = SP.arange(xmin,xmax,0.7)
    
    C = 2       #offset
    sigma = 0.01
    
    b = 0
    
    y  = b*x + C + 1*SP.sin(x)
#    dy = b   +     1*SP.cos(x)
    y += sigma*random.randn(y.shape[0])
    
    y-= y.mean()
    
    x = x[:,SP.newaxis]
    
    #predictions:
    X = SP.linspace(0,10,100)[:,SP.newaxis]
    
    
    #hyperparamters
    dim = 1
    
    logthetaCOVAR = SP.log([1,1,sigma])
    hyperparams = {'covar':logthetaCOVAR}
    
    SECF = se.SEARDCF(dim)
    noiseCF = noise.NoiseISOCF()
    covar = combinators.SumCF((SECF,noiseCF))
    covar_priors = []
    #scale
    covar_priors.append([lnpriors.lngammapdf,[1,2]])

    covar_priors.extend([[lnpriors.lngammapdf,[1,1]] for i in xrange(dim)])
    #noise
    covar_priors.append([lnpriors.lngammapdf,[1,1]])
    priors = {'covar':covar_priors}
    Ifilter = {'covar': SP.array([1,1,1],dtype='int')}
    
    gp = GP(covar,x=x,y=y)
    opt_model_params = opt.opt_hyper(gp,hyperparams,priors=priors,gradcheck=True,Ifilter=Ifilter)[0]
    
    #predict
    [M,S] = gp.predict(opt_model_params,X)
    
    gpr_plot.plot_sausage(X,M,SP.sqrt(S))
    gpr_plot.plot_training_data(x,y)
    PL.show()
    
if __name__ == '__main__':
    run_demo()
