"""
Covariance function module abstract base class.
"""

__all__ = ["sederiv","sq_dist","sumCF","productCF","noiseCF"]

# import python / numpy:
from pylab import *
from numpy import * 

class CovarianceFunction(object):
#    __slots__= ["n_params","dimension","index","Iactive"]

    
    def __init__(self):
        self.n_params = nan
        self.dimension = 1
        self.Iactive = arange(self.dimension)
        pass

    def K(self,logtheta,*args):
        "covariance"
        print "implement K"
        pass
        
    def Kd(self, logtheta, *args):
        print "please implement Kd"
    	pass

    def getNparams(self):
        return self.n_params;

    def getParamNames(self):
        """return the names of hyperparameters to make identificatio neasier"""
        return []
    
    def getDefaultParams(self,x=None,y=None):
        "Default implementation: no parameters for CV"
        return array([])
    
    def setActiveDimensions(self,Iactive = None,**kwargin):
        """est active subset dimensions"""
        self.Iactive = Iactive
        pass
