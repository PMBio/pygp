"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) combined with noise
:py:class:`covar.noise.noiseCF` by summing them up
(using :py:class:`covar.combinators.sumCF`).
"""

import sys
sys.path.append('./../')
sys.path.append('./')

#import sys
#sys.path.append('/kyb/agbs/stegle/work/ibg/lib')

import pdb
import pylab as PL
import scipy as SP
import numpy.random as random


from covar import se, noise, combinators
import gpr as GPR

import sys
import lnpriors
import logging as LG

LG.basicConfig(level=LG.INFO)

random.seed(1)

#0. generate Toy-Data; just samples from a superposition of a sin + linear trend
xmin = 1
xmax = 2.5*SP.pi
x = SP.arange(xmin,xmax,0.7)

C = 2       #offset
b = 0.5
sigma = 0.01

b = 0

y  = b*x + C + 1*SP.sin(x)
dy = b   +     1*SP.cos(x)
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
noise = noise.NoiseISOCF()
covar = combinators.SumCF((SECF,noise))
covar_priors = []
#scale
covar_priors.append([lnpriors.lngammapdf,[1,2]])
for i in range(dim):
    covar_priors.append([lnpriors.lngammapdf,[1,1]])
#noise
covar_priors.append([lnpriors.lngammapdf,[1,1]])
priors = {'covar':covar_priors}
Ifilter = {'covar': SP.array([1,1,1],dtype='int')}

gpr = GPR.GP(covar,x=x,y=y)
[opt_model_params,opt_lml]=GPR.optHyper(gpr,hyperparams,priors=priors,gradcheck=True,Ifilter=Ifilter)

#predict
[M,S] = gpr.predict(opt_model_params,X)

PL.plot(x[:,0], y, 'ro',
     X[:,0], M, 'g-',
     X[:,0], M+2*SP.sqrt(S), 'b-',
     X[:,0], M-2*SP.sqrt(S), 'b-')
#show()
