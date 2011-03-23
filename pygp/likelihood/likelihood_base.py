import scipy as SP
from pygp.linalg import *
import copy 



class ALik(object):
    """abstract class for arbitrary likelihood model"""
    pass




class GaussLikISO(ALik):
    """Gaussian isotropic likelihood model
    This may serve as a blueprint for other more general likelihood models
    _get_Knoise serves as an effective component of the covariance funciton and may be adapted as needed.
    """

    def __init__(self):
        self.n_hyperparameters = 1
        pass

    def get_number_of_parameters(self):
        return self.n_hyperparameters

    def K(self,theta,x1):
        sigma = SP.exp(2*theta[0])
        Knoise = sigma*SP.eye(x1.shape[0])
        return Knoise

    def Kdiag(self,theta,x1):
        sigma = SP.exp(2*theta[0])
        return sigma*SP.ones(x1.shape[0])

    def Kgrad_theta(self,theta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. calculate kernel
        #no noise
        K = self.K(theta,x1)
        assert i==0, 'unknown hyperparameter'
        return 2*K
