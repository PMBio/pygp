import sys
sys.path.append('./../')
sys.path.append('./')

#import sys
#sys.path.append('/kyb/agbs/stegle/work/ibg/lib')

import pdb
from pylab import *
from numpy import *

from pyio.csvex import *

from covar import *
import gpr as GPR

import sys
from lnpriors import *
import logging as LG

LG.basicConfig(level=LG.INFO)


GPR.DEBUG = True

#0. generate Toy-Data; just samples from a superposition of a sin + linear trend
xmin = 1
xmax = 2.5*pi
x = arange(xmin,xmax,0.7)

C = 2       #offset
b = 0.5
sigma = 0.01

b = 0

y  = b*x + C + 1*sin(x)
dy = b   +     1*cos(x)
y += sigma*random.randn(size(y))

x = x.reshape(size(x),1)

#predictions:
X = linspace(0,10,100)
X = X.reshape(size(X),1)

logtheta = log([1,1,sigma])


dim = 1


#simulate fake 2d date
if 0:
    dim = 2
    x_ = zeros([x.shape[0],2])
    x_[:,0] = x[:,0]
    x_[:,1] = x[:,0]+3
    X_ = zeros([X.shape[0],2])
    X_[:,0] = X[:,0]
    X_[:,1] = X[:,0]+3

    x = x_
    X = X_
    logtheta = log([1,1,0.1,sigma])

SECF = se.SECF(dim)
SEnoise = noiseCF.NoiseCovariance()

covar = combinators.SumCovariance((SECF,SEnoise))

gpr = GPR.GP(covar,Smean=True,x=x,y=y)

if 1:
    GPR.DEBUG=2
    priors = []
    #scale
    priors.append([lngammapdf,[1,2]])
    for i in range(dim):
        priors.append([lngammapdf,[1,1]])
    #noise
    priors.append([lngammapdf,[1,1]])
      
    I_filter=array(ones_like(logtheta),dtype='bool')
    #maybe we should filter optimzing theta
    modelparameters = {'covar':logtheta}
    opt_model_params=GPR.optHyper(gpr,modelparameters,I_filter,priors=priors)
    print "optimized hyperparameters:" + str(exp(opt_model_params['covar']))
else:
    opt_model_params=modelparameters

#predict
[M,S] = gpr.predict(opt_model_params,X)


hold(True)
plot(x[:,0], y, 'ro',
     X[:,0], M, 'g-',
     X[:,0], M+2*sqrt(S), 'b-',
        X[:,0], M-2*sqrt(S), 'b-')
#show()


D = zeros([x.shape[0],x.shape[1]+1])
D[:,0:-1] = x
D[:,-1] = y

#write csv file
#writeCSV("demo_gp.csv",D,",")
