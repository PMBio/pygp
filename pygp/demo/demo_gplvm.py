"""
Demo Application for Gaussian process latent variable models
====================================

"""

import logging as LG
import numpy.random as random

from pygp.gp import gplvm
from pygp.covar import linear, noise, combinators

import pygp.optimize as opt
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors

import pylab as PL
import scipy as SP


if __name__ == '__main__':
    LG.basicConfig(level=LG.INFO)

    #1. simulate data from a linear PCA model
    N = 100
    K = 3
    D = 10

    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T
    Y+= 0.5*SP.random.randn(N,D)

    #use "standard PCA"
    [Spca,Wpca] = gplvm.PCA(Y,K)

    #reconstruction
    Y_ = SP.dot(Spca,Wpca.T)

    #use GPLVM
    linear_cf = linear.LinearCFISO(n_dimensions=K)
    noise_cf = noise.NoiseCFISO()
    covariance = combinators.SumCF((linear_cf,noise_cf))

    #hyperparameters
    hyperparams = {'covar': SP.log([1,0.1])}
    #initialization of X at arandom
    X0 = SP.random.randn(N,K)
    X0 = Spca
    hyperparams['x'] = X0

    gplvm = gplvm.GPLVM(covar_func=covariance,x=X0,y=Y)
    [opt_hyperparams,opt_lml] = opt.opt_hyper(gplvm,hyperparams,gradcheck=True)
