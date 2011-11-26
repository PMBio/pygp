import scipy as SP
from pygp.linalg import *
import pdb
import copy 



class ALik(object):
    """abstract class for arbitrary likelihood model"""
    pass



class GaussGroupLikISO(ALik):
    """Gaussian isotropic noise with a specific noise parameter per group
    """

    def __init__(self,n_groups=2,column=0):
        self.n_hyperparameters = n_groups
        self.column = column
        pass

    def get_number_of_parameters(self):
        return self.n_hyperparameters

    def K(self,theta,x1):
        return SP.diag(self.Kdiag(theta,x1))

    def Kdiag(self,theta,x1):
        sigma = SP.exp(2*theta)
        Knoise = SP.zeros([x1.shape[0]])
        for i in xrange(len(theta)):
            Knoise += sigma[i]*(1.0*(x1[:,self.column]==i))
        return Knoise

    def Kgrad_theta(self,theta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        sigma = SP.exp(2*theta[i])
        Knoise = SP.diag(2*sigma*(1.0*(x1[:,self.column]==i)))
        return Knoise


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
        sigma = SP.exp(2*theta)
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



class GaussLikARD(ALik):
    """Gaussian likelihood model with one noise level per dimension
    Note: currently this is only supported in gplvm module
    """

    def __init__(self,n_dimensions=1):
        self.n_dimensions      = n_dimensions
        self.n_hyperparameters = n_dimensions
        pass

    def get_number_of_parameters(self):
        return self.n_hyperparameters

    def K(self,theta,x1):
        sigma = SP.exp(2*theta[0])
        Knoise = sigma*SP.eye(x1.shape[0])
        return Knoise

    def Kdiag(self,theta,x1):
        ### in ARD noise, the diagonal is a matrix with one diagonal per dimension
        sigma = SP.exp(2*theta)
        RV = SP.tile(sigma[SP.newaxis,:],[x1.shape[0],1])
        return RV

    def Kgrad_theta(self,theta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. calculate kernel
        #no noise
        K = self.Kdiag(theta,x1)[:,i]
        return 2*K



def n2mode(x):
    """convert from natural parameter to mode and back"""
    return SP.array([x[0]/x[1],1/x[1]])

def sigmoid(x):
    """sigmoid function int_-inf^+inf Normal(x,1)"""
    return ( 1+SP.special.erf(x/SP.sqrt(2.0)) )/2.0
def gos(x):
    """Gaussian over sigmoid"""
    return ( SP.sqrt(2.0/SP.pi)*SP.exp(-0.5*x**2)/(1+SP.special.erf(x/SP.sqrt(2.0))) )

class ProbitLik(ALik):
    """Probit likelihood for GP classification"""

    def get_number_of_parameters(self):
        return 0

    def K(self,theta,x1):
        zi      = x1*theta[0]/SP.sqrt(theta[1]*(1+theta[1]))
        K      = theta[0]/theta[1] + x1*gos(zi)/SP.sqrt(theta[1]*(1+theta[1]))
        #import pdb;pdb.set_trace()
        return SP.diag(K)

    def Kdiag(self,theta,x1):
        zi      = x1*theta[0]/SP.sqrt(theta[1]*(1+theta[1]))
        K      = theta[0]/theta[1] + x1*gos(zi)/SP.sqrt(theta[1]*(1+theta[1]))
        #import pdb;pdb.set_trace()
        return SP.squeeze(K)

    def Kgrad_theta(self,theta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. calculate kernel
        #no noise
        zi      = x1*theta[0]/SP.sqrt(theta[1]*(1+theta[1]))
        Kgrad     = (1-gos(zi)*(zi+gos(zi))/(1+theta[1]))/theta[1]
        return Kgrad
    
#    def calcExpectations(self,t,cav_np,x=None):
#        """calc expectation values (moments) for EP udpates
#        t: the target
#        cav_np: (nu,tau) of cavity (natural params)
#        x: (optional) input (not used in this likelihood)
#        """
#
#
#        #the derivation here follows Rasmussen & Williams
#        zi      = t*cav_np[0]/SP.sqrt(cav_np[1]*(1+cav_np[1]))
#        #0. moment
#        Z       = sigmoid(zi)
#        #1. moment
#        Fu      = cav_np[0]/cav_np[1] + t*gos(zi)/SP.sqrt(cav_np[1]*(1+cav_np[1]))
#        Fs2     = (1-gos(zi)*(zi+gos(zi))/(1+cav_np[1]))/cav_np[1]
#        
#        return S.array([Fu,Fs2,Z])
#        pass
