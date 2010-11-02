"""demo_gpr_constrained
- demonstrate Gaussian Process regressio with a constrained (theta) likelihood
Constraints are created by ammending the input dimension by one.
These are indicators  0: normal data point
                     <0: constrainted < than datum (y)
                     >0: constrained  > than datum (y)

"""


import sys
sys.path.append('./../')


import sys
sys.path.append('/home/os252/work/lib/python/mlib')

from pylab import *
from numpy import *

from io.csv import *

from pygp.covar import *
from EPLikelihood import *

import pygp.gpr as GPR
import pygp.gprEP as GPREP

import pydb
import sys
from stats.lnpriors import *
import logging as LG

#seed
random.seed(1)
#loglevel
LG.getLogger().setLevel(LG.DEBUG)

GPR.DEBUG = True

#0. generate Toy-Data; just samples from a superposition of a sin + linear trend
xmin = 1
xmax = 2.5*pi
x = arange(xmin,xmax,0.3)

Nep = 3
C = 2       #offset
b = 0.5
sigma = 0.1

b = 0

y  = b*x + C + 1*sin(x)
dy = b   +     1*cos(x)
y += sigma*random.randn(size(y))

x = x.reshape(size(x),1)
#indicator for theta constraints; default 0, normal datapoint
It= zeros_like(x)
x = concatenate((x,It),axis=1)

#add constraints
Iout = zeros([x.shape[0]],dtype=bool)
if 1:
    Nc = 2
    perm     = random.permutation(x.shape[0])
    Iout[perm[0:Nc]] = True
    y[Iout] += 0.3*random.randn(Nc)
    x[Iout,1] = +1
    pass


#predictions:
X = linspace(0,10,100)
X = X.reshape(size(X),1)

logtheta = log([1,1,sigma])
dim = 1


#initialize covariance,likelihood and gp...

covar = sederiv.SquaredExponentialCF(dim)
gpr   = GPR.GP(covar,Smean=True,x=x,y=y)

if 0:
    gprEP = GPREP.GPEP(covar=covar,likelihood=GaussLikelihood(),Smean=True,rescale=False,x=x,y=y)
    logthetaEP = log([1,1,1E-6,sigma])
if 1:
    #use constrained likelihood
    likelihood = ConstrainedLikelihood(alt=GaussLikelihood())
    gprEP = GPREP.GPEP(covar=covar,Nep=Nep,likelihood=likelihood,Smean=True,rescale=False,x=x,y=y)
    logthetaEP = log([1,1,1E-6,sigma])
    
#predict

if 1:
    [MEP,SEP] = gprEP.predict(logthetaEP,X)
    figure(1)
    hold(True)
    plot(x[~Iout,0], y[~Iout], 'ko',
         x[Iout,0], y[Iout], 'ro',
         X[:,0], MEP, 'g-',
         X[:,0], MEP+2*sqrt(SEP), 'b-',
         X[:,0], MEP-2*sqrt(SEP), 'b-')
    title('EP')

if 1:
    [M,S] = gpr.predict(logtheta,X)
    figure(2)
    hold(True)
    plot(x[~Iout,0], y[~Iout], 'ko',
         x[Iout,0], y[Iout], 'ro',
         X[:,0], M, 'g-',
         X[:,0], M+2*sqrt(S), 'b-',
         X[:,0], M-2*sqrt(S), 'b-')
    title('non-EP')

    
show()


D = zeros([x.shape[0],x.shape[1]+1])
D[:,0:-1] = x
D[:,-1] = y

#write csv file
writeCSV("demo_gp.csv",D,",")
