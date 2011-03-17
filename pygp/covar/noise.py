"""
Noise covariance function
-------------------------

NoiseCFISO
NoiseCFReplicates
"""

import sys
sys.path.append("../")


# import python / numpy:
import scipy as SP

from pygp.covar import CovarianceFunction



class NoiseCFISO(CovarianceFunction):
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

    def K(self,theta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters theta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameter!

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        
        #noise is only presenet if have a single argument
        if(x2 is None):
            noise = SP.eye(x1.shape[0])*SP.exp(2*theta[0])
        else:
            noise = 0 

        return noise

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

    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    def Kgrad_xdiag(self,theta,x1,d):
        RV = SP.zeros([x1.shape[0]])
        return RV


    

class NoiseCFReplicates(CovarianceFunction):
    """Covariance function for replicate-wise Gaussian observation noise"""

    def __init__(self, replicate_indices,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.replicate_indices = replicate_indices
        self.n_hyperparameters = len(SP.unique(replicate_indices))

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = ["Sigma %i" % (i) for i in range(self.n_hyperparameters)]
        return names

    def K(self,theta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters theta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameters!

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        assert len(theta)==self.n_hyperparameters,'Too many hyperparameters'
        #noise is only present if have a single argument
        if(x2 is None):
            noise = SP.eye(x1.shape[0])
            for i_,n in enumerate(theta):
                noise[self.replicate_indices==i_] *= SP.exp(2*n)
        else:
            noise = 0 
        return noise

    def Kgrad_theta(self,theta,x1,i):
        '''
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. calculate kernel
        #no noise
        assert i<self.n_hyperparameters, 'unknown hyperparameters'
        K = SP.eye(x1.shape[0])
        K[self.replicate_indices==i] *= SP.exp(
            2*theta[i])
        K[self.replicate_indices!=i] *= 0
        return 2*K  

    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    def Kgrad_xdiag(self,theta,x1,d):
        RV = SP.zeros([x1.shape[0]])
        return RV
