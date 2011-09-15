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


if __name__ == '__main__':
    LG.basicConfig(level=LG.INFO)

    #1. simulate data from a linear PCA model
    N = 50
    K = 5
    D = 200

    SP.random.seed(1)
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T

    #factor analaysis nise, i.e. one distinct noise level per feature dimension?
    sim_fa_noise = True
    #use fa noise for analysis?
    fa_noise = True

    if sim_fa_noise:
        #inerpolate noise levels
        noise_levels = 0.1*SP.ones([D])
        #more noise level for first half of dimensions
        noise_levels[0:D/2] = 1.0
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
        hyperparams = {'covar': SP.log([1.2])}
    if 0:
        covariance = se.SqexpCFARD(n_dimensions=K)
        hyperparams = {'covar': SP.log([1]*(K+1))}
    import pdb;pdb.set_trace()
    #initialization of X at arandom
    X0 = SP.random.randn(N,K)
    X0 = Spca
    hyperparams['x'] = X0

    #copy for FA
    hyperparams_fa = copy.deepcopy(hyperparams)

    #factor analysis noise
    likelihood_fa = lik.GaussLikARD(n_dimensions=D)
    hyperparams_fa['lik'] = SP.log(SP.ones(Y.shape[1])+0.1*SP.random.randn(Y.shape[1]))
    g_fa = gplvm_ard.GPLVMARD(covar_func=covariance,likelihood=likelihood_fa,x=X0,y=Y)
    
    #standard Gaussian noise
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    g = gplvm.GPLVM(covar_func=covariance,likelihood=likelihood,x=X0,y=Y)
        
        
    #try evaluating marginal likelihood first

    #this works very well, without X
    del(hyperparams['x'])
    del(hyperparams_fa['x'])
    if 0:
        print "running standard gplvm"
        [opt_hyperparams,opt_lml] = opt.opt_hyper(g,hyperparams,gradcheck=True)
        print "running fa noise gplvm"
        [opt_hyperparams_fa,opt_lml_fa] = opt.opt_hyper(g_fa,hyperparams_fa,gradcheck=True)

    #now include x in optimization.
    #looks like this i more difficutl for the fa model:


    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*D)
    hyperparams['x'] = X0
    hyperparams_fa['x'] = X0
    print "running standard gplvm"
    [opt_hyperparams,opt_lml2] = opt.opt_hyper(g,hyperparams,gradcheck=False)
    import pdb;pdb.set_trace()
    if 0:
        print "running fa noise gplvm"
        [opt_hyperparams_fa,opt_lml_fa2] = opt.opt_hyper(g_fa,hyperparams_fa,gradcheck=True,bounds=bounds)
   
