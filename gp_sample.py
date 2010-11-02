"""helper function to sample from a Gaussian process"""
import scipy as SP


def GP_sample_prior(covar,X,logtheta,ns=1):
    """create samples form a GP prior
    X: inputs to sample from
    covar: covariance function
    logtheta: hyper parameters
    ns: number of samples
    """
    K = covar.K(logtheta,X) 

    L = SP.linalg.cholesky(K).T
    Y = SP.dot(L,random.randn(X.shape[0],ns))   
    return Y

def GP_sample_posterior(covar,X,logtheta,x,y,ns=1):
    """
    x: training inputs
    y: trainint targets
    else like GP_sample_prior
    """

    KXx = covar.K(logtheta,x,X)
    KXX = covar.K(logtheta,X)
    Kxx = covar.K(logtheta,x)

    iKxx = SP.linalg.inv(Kxx+eye(Kxx.shape[0])*0.01)

    mu = SP.dot(KXx.T,SP.dot(iKxx,y)).reshape([-1,1])
    cov = KXX - SP.dot(KXx.T,SP.dot(iKxx,KXx))   
    L  = SP.linalg.cholesky(cov).T
    Y  = mu + SP.dot(L,random.randn(X.shape[0],ns))
    return Y
