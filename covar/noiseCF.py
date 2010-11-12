"""
Noise covariance function
-------------------------

adding Gaussian observation noise to an arbitrary CovarianceFunction chosen.
"""

import sys
sys.path.append("../")


# import python / numpy:
from pylab import *
from numpy import *

from covar import CovarianceFunction


class NoiseCovariance(CovarianceFunction):
    """Covariance function for Gaussian observation noise"""

    def __init__(self):
        self.n_hyperparameters = 1

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = []
        names.append('Sigma')
        return names

    def K(self,modelparameters,*args):
        """
        Get Covariance matrix K with given hyperparameters logtheta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameter!

        **Parameters:**
        See :py:class:`covar.CovarianceFunction` 
        """
        x1 = args[0]
        logtheta = modelparameters['covar']

        #noise is only presenet if have a single argument
        if(len(args)==1):
            noise = eye(x1.shape[0])*exp(2*logtheta[0])
            x2    = x1
        else:
            noise = 0 
            x2 = args[1]

        return noise

    def Kd(self,modelparameters,*args):
        '''The derivatives of the covariance matrix for each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. calculate kernel
        #no noise
        _K = self.K(modelparameters,*args)

        rv = zeros([self.n_hyperparameters,_K.shape[0],_K.shape[1]])
        rv[:] = _K
        #from exp.
        rv[0]*= 2
        return rv
        

    
