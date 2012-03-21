"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) preprocessed by shiftCF(:py:class`covar.combinators.ShiftCF) and combined with noise
:py:class:`covar.noise.NoiseISOCF` by summing them up
(using :py:class:`covar.combinators.SumCF`).
"""

from pygp.covar import se, noise, combinators
from pygp.gp import GP
from pygp.priors import lnpriors
from pygp.optimize import opt_hyper
from pygp.plot import gpr_plot

import logging as LG
import numpy.random as random

#import pdb
import pylab as PL
import scipy as SP


def run_demo():
    LG.basicConfig(level=LG.INFO)
    PL.figure()
    
    random.seed(1)
    
    #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
    n_replicates = 4
    xmin = 1
    xmax = 2.5*SP.pi

    x1_time_steps = 10
    x2_time_steps = 20
    
    x1 = SP.zeros(x1_time_steps*n_replicates)
    x2 = SP.zeros(x2_time_steps*n_replicates)

    for i in xrange(n_replicates):
	x1[i*x1_time_steps:(i+1)*x1_time_steps] = SP.linspace(xmin,xmax,x1_time_steps)
	x2[i*x2_time_steps:(i+1)*x2_time_steps] = SP.linspace(xmin,xmax,x2_time_steps)

    C = 2       #offset
    #b = 0.5
    sigma1 = 0.15
    sigma2 = 0.15
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
    
    for i in xrange(n_replicates):
	x1[i*x1_time_steps:(i+1)*x1_time_steps] += .7 + (i/2.)
	x2[i*x2_time_steps:(i+1)*x2_time_steps] -= .7 + (i/2.)  

    x1 = x1[:,SP.newaxis]
    x2 = x2[:,SP.newaxis]
    
    x = SP.concatenate((x1,x2),axis=0)
    y = SP.concatenate((y1,y2),axis=0)
    
    #predictions:
    X = SP.linspace(xmin-n_replicates,xmax+n_replicates,100*n_replicates)[:,SP.newaxis]
    
    #hyperparamters
    dim = 1
    replicate_indices = []
    for i,xi in enumerate((x1,x2)):
        for rep in SP.arange(i*n_replicates, (i+1)*n_replicates):
            replicate_indices.extend(SP.repeat(rep,len(xi)/n_replicates))
    replicate_indices = SP.array(replicate_indices)
    n_replicates = len(SP.unique(replicate_indices))
    
    logthetaCOVAR = [1,1]
    logthetaCOVAR.extend(SP.repeat(SP.exp(1),n_replicates))
    logthetaCOVAR.extend([sigma1])
    logthetaCOVAR = SP.log(logthetaCOVAR)#,sigma2])
    hyperparams = {'covar':logthetaCOVAR}
    
    SECF = se.SqexpCFARD(dim)
    #noiseCF = noise.NoiseReplicateCF(replicate_indices)
    noiseCF = noise.NoiseCFISO()
    shiftCF = combinators.ShiftCF(SECF,replicate_indices)
    CovFun = combinators.SumCF((shiftCF,noiseCF))
    
    covar_priors = []
    #scale
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    for i in range(dim):
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    #shift
    for i in range(n_replicates):
        covar_priors.append([lnpriors.lnGauss,[0,.5]])    
    #noise
    for i in range(n_noises):
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    
    covar_priors = SP.array(covar_priors)
    priors = {'covar':covar_priors}
    Ifilter = {'covar': SP.ones(n_replicates+3)}
    
    gpr = GP(CovFun,x=x,y=y) 
    opt_model_params = opt_hyper(gpr,hyperparams,priors=priors,gradcheck=False,Ifilter=Ifilter)[0]
    
    #predict
    [M,S] = gpr.predict(opt_model_params,X)
    
    T = opt_model_params['covar'][2:2+n_replicates]
    
    PL.subplot(212)
    gpr_plot.plot_sausage(X,M,SP.sqrt(S),format_line=dict(alpha=1,color='g',lw=2, ls='-'))
    gpr_plot.plot_training_data(x,y,shift=T,replicate_indices=replicate_indices,draw_arrows=2)
    
    PL.suptitle("Example for GPTimeShift with simulated data", fontsize=23)
    
    PL.title("Regression including time shift")
    PL.xlabel("x")
    PL.ylabel("y")
    ylim = PL.ylim()
    
    gpr = GP(combinators.SumCF((SECF,noiseCF)),x=x,y=y)
    priors = {'covar':covar_priors[[0,1,-1]]}
    hyperparams = {'covar':logthetaCOVAR[[0,1,-1]]}
    opt_model_params = opt_hyper(gpr,hyperparams,priors=priors,gradcheck=False)[0]
    
    PL.subplot(211)
    #predict
    [M,S] = gpr.predict(opt_model_params,X)
    
    gpr_plot.plot_sausage(X,M,SP.sqrt(S),format_line=dict(alpha=1,color='g',lw=2, ls='-'))
    gpr_plot.plot_training_data(x,y,replicate_indices=replicate_indices)
    
    PL.title("Regression without time shift")
    PL.xlabel("x")
    PL.ylabel("y")
    PL.ylim(ylim)
    
    PL.subplots_adjust(left=.1, bottom=.1, 
    right=.96, top=.8,
    wspace=.4, hspace=.4)
    PL.show()
    
if __name__ == "__main__":
    run_demo()
