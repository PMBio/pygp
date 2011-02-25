"""
Sample from Gaussian Processes
==============================

Helper functions to sample from a Gaussian process"""
import scipy as SP

def GP_sample_prior(covar,X,logtheta,ns=1):
    """
    Create samples form a GP prior

    **Parameters:**
    
    X : [double]
        inputs to sample from.

    covar : :py:class:`covar.CovarianceFunction`
        Covariance function to sample from.

    logtheta : [double]
        Hyperparameters

    ns : int
        Number of samples
    """
    K = covar.K(logtheta,X) 

    L = SP.linalg.cholesky(K).T
    Y = SP.dot(L,SP.random.randn(X.shape[0],ns))   
    return Y

def GP_sample_posterior(covar,X,logtheta,x,y,ns=1):
    """
    Sample from the posterior distribution of a GP
    
    x : [double]
        training inputs

    y : [double]
        training targets

    other :
        See :py:func:`gp_sample.GP_sample_prior`
    """

    KXx = covar.K(logtheta,x,X)
    KXX = covar.K(logtheta,X)
    Kxx = covar.K(logtheta,x)

    iKxx = SP.linalg.inv(Kxx+SP.eye(Kxx.shape[0])*0.01)

    mu = SP.dot(KXx.T,SP.dot(iKxx,y)).reshape([-1,1])
    cov = KXX - SP.dot(KXx.T,SP.dot(iKxx,KXx))   
    L  = SP.linalg.cholesky(cov).T
    Y  = mu + SP.dot(L,SP.random.randn(X.shape[0],ns))
    return Y
