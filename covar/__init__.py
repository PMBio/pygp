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


__all__ = ["se","sq_dist","combinators","noise","linear"]

# import python / numpy:
from pylab import *
from numpy import * 

import sq_dist

class CovarianceFunction(object):
    """
    Abstract Covariance Function.
    
    **Important:** *All Covariance Functions have
    to inherit from this class in order to work
    properly with this GP framework.*
    """
    __slots__= ["n_hyperparameters",
                "dimensions",
                "dimension_indices",
                "active_dimension_indices"]

    def __init__(self):
        self.n_hyperparameters = nan
        self.n_dimensions = 1
        self.active_dimension_indices = arange(self.dimensions)
        pass

    def K(self, modelparameters, *args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**

        modelparameters : dict = {'covar': hyperparameters, ...}

            The hyperparameters for which the covariance
            matrix shall be computed. :py:func:`hyperparameters` are the
            hyperparameters for the respective covariance function.
            For instance, the :py:class:`covar.se.SECF` holds hyperparameters
            like :py:func`[Amplitude, Length-Scale(s)]`.

        args : X[, X']
        
            The (interpolation) inputs, which shall be
            the pointwise covariance calculated for.
        """
        print "implement K"
        pass
        
    def Kd(self, modelparameters, *args):
        """
        Get Derivatives of Covariance matrix K for each given
        hyperparameter resepctively. Output matrix with
        derivatives will have the same order, as the
        hyperparameters have.

        **Parameters:**

        modelparameters : dict = {'covar': hyperparameters, ...}

            The hyperparameters for which the derivative
            covariance matrix shall be computed.

        args : X[, X']

            The (interpolation) inputs, which shall be
            the pointwise covariance calculated for.
        """
        print "please implement Kd"
    	pass

    def _pointwise_distance(self,x1,x2,L=None):
        """
        Pointwise distance between vector x1 and x2. Optionally normalized (divided) by L
        """
        if L != None:
            x1 = array(x1,dtype='float64')/L
            x2 = array(x2,dtype='float64')/L
        return sq_dist.dist(x1,x2)

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to make
        identification easier
        """
        return []

    def get_number_of_parameters(self):
        return self.n_hyperparameters
    
    def get_default_hyperparameters(self,x=None,y=None):
        return array([])

    def get_n_dimensions(self):
        return self.n_dimensions
    
    def set_active_dimensions(self,active_dimension_indices = None):
        self.active_dimension_indices = active_dimension_indices
        pass
