"""
Hyperpriors for log likelihood calculation
------------------------------------------
"""

import scipy as SP
import scipy.special as SPs


def lngammapdf(x,params):
    """
    Returns the ``log gamma (x,k,t)`` distribution and its derivation with::
        lngamma     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
        dlngamma    = (k-1)/x - 1/t

    **Parameters:**

    x : vector of ints
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

def lngauss(x,params):
    """
    Returns the ``log normal distribution`` in interval x,
    given mean mu and variance sigma.

    [N(params), d/dx N(params)] = N(mu,sigma|x).
    Note: Give mu and sigma as mean and variance, the result will be logarithmic!

    **Parameters:**

    x : vector of ints
        the interval in which the distribution shall be computed.

    params : [k, t]
        the distribution parameters k and t.
    """
    mu = SP.double(params[0])
    sigma = SP.double(params[1])

    # selfCalculated = True
    halfLog2Pi = 0.91893853320467267 # =.5*(log(2*pi))
    #N = -(((x-mu)**2)/(2*(sigma**2))) - log(sigma) - halfLog2Pi
    N = log(exp((-((x-mu)**2)/(2*(sigma**2))))/sigma)- halfLog2Pi
    # else:
    #     N = array(log(normpdf(x,mu,sigma)))
    dN = -(x-mu)/(sigma**2)

    # if N.shape != ():
    #     N[exp(N) < .1] = -1e4
    #     dN[exp(N) < .1] = 0
    # else:
    #     if exp(N) < .1:
    #         N = -1e4
    #         dN = 0

    # if __name__ == '__main__':
    #     pdb.set_trace()

    return [N,dN]

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
