"""
Noise covariance function
-------------------------

adding Gaussian observation noise to an arbitrary CovarianceFunction chosen.
"""

import sys
sys.path.append("../")


# import python / numpy:
import scipy as SP

from covar import CovarianceFunction


class NoiseISOCF(CovarianceFunction):
    """Covariance function for Gaussian observation noise"""

    def __init__(self):
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
        '''The derivatives of the covariance matrix for each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. calculate kernel
        #no noise
        K = self.K(logtheta,x1)
        assert i==0, 'unknown hyperparameter'
        return 2*K        


        

    
