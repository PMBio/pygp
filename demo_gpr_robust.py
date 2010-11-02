import sys
sys.path.append('./..')

import pylab as PL
import scipy as S
import scipy.linalg as linalg
import numpy.random as random
from pygp.covar import *
import pygp.gpr as GPR
import pygp.gprEP as GPREP
from pygp.gpr_plot import *
from pygp.EPLikelihood import *

#import pydb
import sys
from stats.lnpriors import *
import logging as LG
import os


#import thesis-wide plot formats

random.seed(4)
#loglevel
LG.getLogger().setLevel(LG.DEBUG)
GPR.DEBUG = True

#covariance function
se_cf = sederiv.SquaredExponentialCFnn()
noise_cf = noiseCF.NoiseCovariance()
covar = sumCF.SumCovariance(covars=[se_cf,noise_cf])
#hyper parameters
logtheta = S.log([3,2,2E-1])
sigma = S.exp(logtheta[-1])
c = 0.9
#introduce extra hyperparameters
logthetaL = S.log([c,1-c,sigma,1E4])
#robust logtheta
logthetaR  = S.concatenate((logtheta,logthetaL))
logthetaR[2] = S.log(1E-1)

#range for plotting
X = S.linspace(-5,5,100)
X = X.reshape([-1,1])

#0. create random sample from covariance function
x = S.array([-3,-1,2,2.4,5]).reshape([-1,1])

random.seed(10)
x = random.rand(20)
x = x*10-5
x = x.reshape([-1,1])


#alpha
K     = covar.K(logtheta,x)
alpha = linalg.cholesky(K)
y     = S.dot(alpha,random.randn(x.shape[0]))

#add outliers 
Iout = zeros([x.shape[0]],dtype=bool)
if 1:
    Noutlier = 2
    perm     = random.permutation(x.shape[0])
    Iout[perm[0:Noutlier]] = True
    y[Iout] += 2*random.randn(Noutlier)
    pass

#initialize covariance,likelihood and gp...
gpr   = GPR.GP(covar,Smean=True,x=x,y=y)
if 1:
    gpr = GPR.GP(covar,Smean=True,x=x,y=y)
if 1:
    #use MOG likelihood function
    Nep = 4
    gprEP = GPREP.GPEP(covar=covar,Nep=Nep,likelihood=MOGLikelihood(),Smean=True,rescale=False,x=x,y=y)
    

    #plot raw data
    hdata=plot(x[~Iout,0], y[~Iout], 'ko',
         x[Iout,0], y[Iout], 'ro')


if 1:
    [M,V] = gpr.predict(logtheta,X)
    hold(True)
    hgp=plot_sausage(X,M,sqrt(V),{'alpha':0.1,'facecolor':'b'},{'linewidth':2,'color':'b'})

if 1:
    [MEP,VEP] = gprEP.predict(logthetaR,X)
    hgpep=plot_sausage(X,MEP,sqrt(VEP),{'alpha':0.1,'facecolor':'g'},{'linewidth':2,'color':'g'})

    PL.legend( (hgp,hgpep),('GP','GP robust'),'lower right')
else:
    PL.legend( [hgp] ,['GP'] ,loc='lower right')



PL.savefig('gp_robust2.png',dpi=200)

