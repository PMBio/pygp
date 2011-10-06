"""
Hyperpriors for log likelihood calculation
------------------------------------------
This module contains a set of commonly used priors for GP models.
Note some priors are available in a log transformed space and non-transformed space
"""

import scipy as SP
import scipy.special as SPs


def lnL1(x,params):
    """L1 type prior defined on the non-log weights
    params[0]: prior cost
    Note: this prior only works if the paramter is constraint to be strictly positive
    """
    l = SP.double(params[0])
    x_ = 1./x

    lng = -l * x_
    dlng = + l*x_**2
    return [lng,dlng]



def lnGamma(x,params):
    """
    Returns the ``log gamma (x,k,t)`` distribution and its derivation with::
    
        lngamma     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
        dlngamma    = (k-1)/x - 1/t
    
    
    **Parameters:**
    
    x : [double]
        the interval in which the distribution shall be computed.
    
    params : [k, t]
        the distribution parameters k and t.
    
    """
    #explicitly convert to double to avoid int trouble :-)
    k=SP.double(params[0])
    t=SP.double(params[1])

    lng     = (k-1)*SP.log(x) - x/t -SPs.gammaln(k) - k*SP.log(t)
    dlng    = (k-1)/x - 1/t
    return [lng,dlng]


def lnGammaExp(x,params):
    """
    
    Returns the ``log gamma (exp(x),k,t)`` distribution and its derivation with::
    
        lngamma     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
        dlngamma    = (k-1)/x - 1/t
   
    
    **Parameters:**
    
    x : [double]
        the interval in which the distribution shall be computed.
    
    params : [k, t]
        the distribution parameters k and t.
    
    """
    #explicitly convert to double to avoid int trouble :-)
    ex = SP.exp(x)
    rv = lnGamma(ex,params)
    rv[1]*= ex
    return rv


def lnGauss(x,params):
    """
    Returns the ``log normal distribution`` and its derivation in interval x,
    given mean mu and variance sigma::

        [N(params), d/dx N(params)] = N(mu,sigma|x).

    **Note**: Give mu and sigma as mean and variance, the result will be logarithmic!

    **Parameters:**

    x : [double]
        the interval in which the distribution shall be computed.

    params : [k, t]
        the distribution parameters k and t.
        
    """
    mu = SP.double(params[0])
    sigma = SP.double(params[1])
    halfLog2Pi = 0.91893853320467267 # =.5*(log(2*pi))
    N = SP.log(SP.exp((-((x-mu)**2)/(2*(sigma**2))))/sigma)- halfLog2Pi
    dN = -(x-mu)/(sigma**2)
    return [N,dN]

def lnuniformpdf(x,params):
    """
    Implementation of ``lnzeropdf`` for development purpose only. This
    pdf returns always ``[0,0]``.  
    """
    return [0,0]

def _plotPrior(X,prior):
    import pylab as PL
    Y = SP.array(prior[0](X,prior[1]))
    PL.hold(True)
    PL.plot(X,SP.exp(Y[0,:]))
    #plot(X,(Y[0,:]))
    #plot(X,Y[1,:],'r-')
    PL.show()
    
if __name__ == "__main__":
    prior = [lnGammaExp,[4,2]]
    X = SP.arange(0.01,10,0.1)
    _plotPrior(X,prior)
