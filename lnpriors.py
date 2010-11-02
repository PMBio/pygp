# Priors for log likelihood calculation

from pylab import *
from numpy import *
from scipy.special import *


def lngammapdf(x,params):
    """ [ln gamma, d/dx lngamma] = log gammapdf (x,k,t)"""
    #explicitly convert to double to avoid int trouble :-)
    k=double(params[0])
    t=double(params[1])

    lng     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
    dlng    = (k-1)/x - 1/t

    return [lng,dlng]

def lnzeropdf(x,params):
    return [0,0]


def plotPrior(X,prior):
    Y = array(prior[0](X,prior[1]))
    hold(True)
    plot(X,exp(Y[0,:]))
    #plot(X,(Y[0,:]))
    #plot(X,Y[1,:],'r-')
    show()
    
if __name__ == "__main__":
    prior = [lngammapdf,[2,0.5]]
    X = arange(0.01,10,0.1)
    plotPrior(X,prior)
