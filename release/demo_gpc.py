import sys
sys.path.append('./../')
sys.path.append('/home/os252/work/lib/python/mlib')

import pylab as PL
import scipy as S

from pygp.covar import *
from EPLikelihood import *

import pygp.gpcEP as GPCEP

import pydb
import sys
from stats.lnpriors import *
import logging as LG




random.seed(1)
#loglevel
LG.getLogger().setLevel(LG.DEBUG)


#simulate some data

x = S.linspace(-5,5,10).reshape([-1,1])
#target: +-1, dependin wether inside or outside a crice
Ip = x**2 < 9
y = S.zeros(x.shape[0])
y[Ip[:,0]] = 1
y[~Ip[:,0]] = -1



#covariance and GP

secf  = sederiv.SquaredExponentialCFnn()
noise = noiseCF.NoiseCovariance()
covar = sumCF.SumCovariance([secf,noise])

gpc   = GPCEP.GPCEP(covar=covar,x=x,y=y)
logtheta = S.log([5,1,1E-8])

X     = S.linspace(-7,7,100).reshape([-1,1])
[P,MU,S2]   = gpc.predict(logtheta,X)


if 1:
    PL.plot(x[Ip],ones_like(x[Ip]),'r+')
    PL.plot(x[~Ip],zeros_like(x[~Ip]),'k+')
    PL.plot(X,P,'k-')
    PL.ylim([-0.2,1.2])
if 0:
    PL.plot(X,MU,'k-')
    PL.plot(X,MU+S.sqrt(S2),'k-.')
    PL.plot(X,MU-S.sqrt(S2),'k-.')
    
PL.show()
