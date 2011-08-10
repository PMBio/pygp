"""fixed covariance functions
Classes for fixed covarinace functions
======================================
Linear covariance functions

FixedCF

"""

#TODO: fix ARD covariance 

import scipy as SP
from pygp.covar import CovarianceFunction


class FixedCF(CovarianceFunction):
    """
    Fixed covariance function with learnt scaling parameter
    """

    def __init__(self,K,**kw_args):
        """K: the fixed covariance structure"""
        super(CovarianceFunction, self).__init__(**kw_args)
        self._K = K 
        self.n_hyperparameters = 1

    def K(self,theta,x1,x2=None):
        #get input data:
        if x2 is None:
            x2 = x1
        #scale parameter
        A  = SP.exp(2*theta[0])
        #if dimensions match return K
        if (x1.shape[0]==self._K.shape[0]) & (x2.shape[0]==self._K.shape[1]):
            return A*self._K
        else:
            return SP.zeros([x1.shape[0],x2.shape[0]])

    def Kdiag(self,theta,x1,i):
        """self covariance"""
        if (x1.shape[0]==self._K.shape[0]):
            return self._K.diagonal()
        else:
            return SP.zeros([x1.shape[0]])

    def Kgrad_theta(self,theta,x1,i):
        RV = self.K(theta,x1)
        #derivative w.r.t. to amplitude
        RV*=2
        return RV


    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    
    def Kgrad_xdiag(self,theta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = SP.zeros([x1.shape[0]])
        return RV



