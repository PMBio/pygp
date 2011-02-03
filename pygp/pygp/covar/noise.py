"""
Noise covariance function
-------------------------

adding Gaussian observation noise to given CovarianceFunction.
"""

import sys
sys.path.append("../")


# import python / numpy:
import scipy as SP

from pygp.covar import CovarianceFunction,CF_Kd_dx


class NoiseReplicateCF(CovarianceFunction):
    """Covariance function for replicate-wise Gaussian observation noise"""

    def __init__(self, replicate_indices,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.replicate_indices = replicate_indices
        self.n_hyperparameters = len(SP.unique(replicate_indices))

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = ["Sigma %i" % (i) for i in range(self.n_hyperparameters)]
        return names

    def K(self,logtheta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters logtheta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameters!

        **Parameters:**
        See :py:class:`covar.CovarianceFunction` 
        """
        assert len(logtheta)==self.n_hyperparameters,'Too many hyperparameters'
        #noise is only present if have a single argument
        if(x2 is None):
            noise = SP.eye(x1.shape[0])
            for i_,n in enumerate(logtheta):
                noise[self.replicate_indices==i_] *= SP.exp(2*n)
        else:
            noise = 0 
        return noise

    def Kd(self,logtheta,x1,i):
        '''
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. calculate kernel
        #no noise
        assert i<self.n_hyperparameters, 'unknown hyperparameter'
        K = SP.eye(x1.shape[0])
        K[self.replicate_indices==i] *= SP.exp(
            2*logtheta[i])
        K[self.replicate_indices!=i] *= 0
        return 2*K  

class NoiseISOCF(CF_Kd_dx):
    """
    Covariance function for Gaussian observation noise for
    all datapoints as a whole.
    """

    def __init__(self,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.n_hyperparameters = 1

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = []
        names.append('Sigma')
        return names

    def K(self,logtheta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters logtheta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameter!

        **Parameters:**
        See :py:class:`covar.CovarianceFunction` 
        """
        
        #noise is only presenet if have a single argument
        if(x2 is None):
            noise = SP.eye(x1.shape[0])*SP.exp(2*logtheta[0])
        else:
            noise = 0 

        return noise

    def Kd(self,logtheta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        #1. calculate kernel
        #no noise
        K = self.K(logtheta,x1)
        assert i==0, 'unknown hyperparameter'
        return 2*K        

    def Kd_dx(self,logtheta,x1,d):
        RV = SP.zeros([x1.shape[0],x1.shape[0]])
        return RV

    
