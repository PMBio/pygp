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
import copy

import pylab as PL
import scipy as SP
import pdb

def run_demo():
    LG.basicConfig(level=LG.INFO)

    #1. simulate data from a linear PCA model
    N = 25
    K = 5
    D = 200

    SP.random.seed(1)
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T

    Y+= 0.5*SP.random.randn(N,D)

    #use "standard PCA"
    [Spca,Wpca] = gplvm.PCA(Y,K)

    #reconstruction
    Y_ = SP.dot(Spca,Wpca.T)

    if 1:
        #use linear kernel
        covariance = linear.LinearCFISO(n_dimensions=K)
        hyperparams = {'covar': SP.log([1.2])}
    if 0:
        #use ARD kernel
        covariance = se.SqexpCFARD(n_dimensions=K)
        hyperparams = {'covar': SP.log([1]*(K+1))}

    #initialization of X at arandom
    X0 = SP.random.randn(N,K)
    X0 = Spca
    hyperparams['x'] = X0
    
    #standard Gaussian noise
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    g = gplvm.GPLVM(covar_func=covariance,likelihood=likelihood,x=X0,y=Y,gplvm_dimensions=SP.arange(X0.shape[1]))

    #specify optimization bounds:
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*D)
    hyperparams['x'] = X0

    print "running standard gplvm"
    [opt_hyperparams,opt_lml2] = opt.opt_hyper(g,hyperparams,gradcheck=False)

    print "optimized latent X:"
    print opt_hyperparams['x']

if __name__ == '__main__':
    run_demo()
