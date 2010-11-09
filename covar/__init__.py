"""
Covariance Functions
====================

We implemented several different covariance functions (CFs) you can work with. To use GP-regression with these covariance functions it is highly recommended to model the noise of the data in one extra covariance function (:py:class:`noiseCF.NoiseCovariance`) and add this noise CF to the CF you are calculating by putting them all together in one :py:class:`sumCF.SumCovariance`.

For example to use the squared exponential CF with noise this should work for you::

 from noiseCF import *
 from sederiv import *

 covariance = SumCovariance((SquaredExponentialCFnn(1),
 NoiseCovariance()))

**Abstract super class for all implementations of covariance functions:**

"""

__all__ = ["sederiv","sq_dist","sumCF","productCF","noiseCF"]

# import python / numpy:
from pylab import *
from numpy import * 

class CovarianceFunction(object):
    """
    Abstract Covariance Function.
    
    **Important:** *All Covariance Functions have
    to inherit from this class in order to work
    properly with this GP framework.*

    """
    __slots__= ["n_params","dimension","index","Iactive"]

    
    def __init__(self):
        self.n_params = nan
        self.dimension = 1
        self.Iactive = arange(self.dimension)
        pass

    def K(self,logtheta,*args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**

        logtheta : [amplitude,length-scale(s)
        [,time-parameter(s)], noise]

           The hyperparameters for which the covariance
           matrix shall be computed.

        args : X[, X']
        
            The interpolation inputs, which shall be
            used as covariance inputs.
        """
        print "implement K"
        pass
        
    def Kd(self, logtheta, *args):
        """
        Get Derivatives of Covariance matrix K for each given
        hyperparameter resepctively. Output matrix with
        derivatives will have the same order, as the
        hyperparameters *logtheta* have.

        **Parameters:**

        logtheta : [amplitude,length-scale(s)
        [,time-parameter(s)], noise]

           The hyperparameters for which the derivative
           covariance matrix shall be computed.

        args : X[, X']

            The interpolation inputs, which shall be
            used as covariance inputs.
        """
        print "please implement Kd"
    	pass
    
    def _dist(self,x1,x2,L):
        """
        Pointwise distance between vector x1 and x2,
        normalized (divided) by L.
        """
        x1 = array(x1,dtype='float64')/L
        x2 = array(x2,dtype='float64')/L
        return sq_dist.dist(x1,x2)

    def getNparams(self):
        return self.n_params;

    def getParamNames(self):
        """return the names of hyperparameters to make
        identification easier"""
        return []
    
    def getDefaultParams(self,x=None,y=None):
        #"Default implementation: no parameters for CV"
        return array([])
    
    def setActiveDimensions(self,Iactive = None,**kwargin):
        #"set active subset dimensions"
        self.Iactive = Iactive
        pass
