import logging as LG
import numpy.random as random
import pylab as PL
from pygp.covar import se, noise, mu, combinators
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors
import pygp.likelihood as lik
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors

from pygp.gp import GP
from pygp.gp.warped_gp import WARPEDGP, TanhWarpingFunction, LinMeanFunction
from pygp.covar import se, noise, combinators
import pygp.likelihood as lik

import pygp.optimize as opt
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors

import pylab as PL
import scipy as SP

def trafo(y):
    return y**(float(3))
def Itrafo(y):
    return y**(1/float(3))

def create_toy_data():

    xmin, xmax = 1, 2.5*SP.pi
    
    x = SP.linspace(xmin,xmax,500)
    
    print len(x)
    X = SP.linspace(0,10,100)[:,SP.newaxis] # predictions
    
    b = 1
    C = 2
    SNR = 0.1
    y  = b*x + C + 1*SP.sin(x) 

    sigma = SNR * (y.max()-y.mean())

    y += sigma*SP.random.randn(len(x))    
    x = x[:,SP.newaxis]
    z = trafo(y)
    L = (z.max()-z.min())
    z /= L

    return x, y, z, sigma, X, Itrafo(z), L

def run_demo():
    LG.basicConfig(level=LG.DEBUG)
    SP.random.seed(10)

    #1. create toy data
    x,y,z,sigma,X,actual_inv,L = create_toy_data()
    n_dimensions = 1
    n_terms = 3
    # build GP
    likelihood = lik.GaussLikISO()
    covar_parms = SP.log([1,1,1E-5])
    hyperparams = {'covar':covar_parms,'lik':SP.log([sigma]), 'warping': (1E-2*SP.random.randn(n_terms,3))}

    SECF = se.SqexpCFARD(n_dimensions=n_dimensions)
    muCF = mu.MuCF(N=X.shape[0])
    covar = combinators.SumCF([SECF,muCF])
    warping_function = TanhWarpingFunction(n_terms=n_terms)
    mean_function    = LinMeanFunction(X= SP.ones([x.shape[0],1]))
    hyperparams['mean'] = 1E-2* SP.randn(1)
    bounds = {}
    bounds.update(warping_function.get_bounds())

    gp = WARPEDGP(warping_function = warping_function,
		  mean_function = mean_function,
		  covar_func=covar, likelihood=likelihood, x=x, y=z)
    opt_model_params = opt.opt_hyper(gp, hyperparams,
				 bounds = bounds,
				 gradcheck=True)[0]

    print "WARPED GP (neg) likelihood: ", gp.LML(hyperparams)

    #hyperparams['mean'] = SP.log(1)
    PL.figure()
    PL.plot(z)
    PL.plot(warping_function.f(y,hyperparams['warping']))
    PL.legend(["real function", "larnt function"])
    
    PL.figure()
    PL.plot(actual_inv)
    PL.plot(warping_function.f_inv(gp.y,hyperparams['warping']))
    PL.legend(['real inverse','learnt inverse'])

    hyperparams.pop("warping")
    hyperparams.pop("mean")    
    gp = GP(covar,likelihood=likelihood,x=x,y=y)
    opt_model_params = opt.opt_hyper(gp,hyperparams,
				     gradcheck=False)[0]
    print "GP (neg) likelihood: ", gp.LML(hyperparams)


if __name__ == '__main__':
    run_demo()
