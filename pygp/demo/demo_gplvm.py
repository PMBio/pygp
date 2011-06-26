"""
Demo Application for Gaussian process latent variable models
====================================

"""

import logging as LG
import numpy.random as random

from pygp.gp import gplvm,gplvm_ard
from pygp.covar import linear,se, noise, combinators

import pygp.optimize as opt
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors
import pygp.likelihood as lik

import pylab as PL
import scipy as SP
import pdb


if __name__ == '__main__':
    LG.basicConfig(level=LG.INFO)

    #1. simulate data from a linear PCA model
    N = 100
    K = 3
    D = 10

    SP.random.seed(1)
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T

    #factor analaysis nise, i.e. one distinct noise level per feature dimension?
    fa_noise = True        

    if fa_noise:
        #inerpolate noise levels
        noise_levels = SP.linspace(0.1,1.0,Y.shape[1])
        Ynoise =noise_levels*random.randn(N,D)
        Y+=Ynoise
    else:
        Y+= 0.5*SP.random.randn(N,D)

    #use "standard PCA"
    [Spca,Wpca] = gplvm.PCA(Y,K)

    #reconstruction
    Y_ = SP.dot(Spca,Wpca.T)

    if 1:
        covariance = linear.LinearCFISO(n_dimensions=K)
        hyperparams = {'covar': SP.log([1])}
    if 0:
        covariance = se.SqexpCFARD(n_dimensions=K)
        hyperparams = {'covar': SP.log([1]*(K+1)),'lik': SP.log([0.1])}
        
    if fa_noise:
        #factor analysis noise
        likelihood = lik.GaussLikARD(n_dimensions=D)
        hyperparams['lik'] = SP.log(SP.ones(Y.shape[1]))
    else:
        #standard Gaussian noise
        likelihood = lik.GaussLikISO()
        hyperparams['lik'] = SP.log([0.1])
        
    #initialization of X at arandom
    X0 = SP.random.randn(N,K)
    X0 = Spca
    hyperparams['x'] = X0

    if fa_noise:
        g = gplvm_ard.GPLVMARD(covar_func=covariance,likelihood=likelihood,x=X0,y=Y)
    else:
        g = gplvm.GPLVM(covar_func=covariance,likelihood=likelihood,x=X0,y=Y)

    #try evaluating marginal likelihood first
    g.LML(hyperparams)
        
    [opt_hyperparams,opt_lml] = opt.opt_hyper(g,hyperparams,gradcheck=True)
