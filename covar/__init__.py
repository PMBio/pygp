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
import scipy as SP

import sq_dist

class CovarianceFunction(object):
    """
    Abstract Covariance Function.
    
    **Important:** *All Covariance Functions have
    to inherit from this class in order to work
    properly with this GP framework.*

    **Parameters:**

    n_dimensions : int

        standard: n_dimension = 1. The number of dimensions this CF
        computes.

    dimension_indices : [int]

        The indices of feature dimensions this CF takes into account.

    """
    __slots__= ["n_hyperparameters",
                "dimensions",
                "dimension_indices",
                "active_dimension_indices"]

    def __init__(self,n_dimensions=1,dimension_indices=None):
        self.n_hyperparameters = SP.nan
        self.n_dimensions = n_dimensions
        if dimension_indices != None:
            self.dimension_indices = SP.array(dimension_indices,dtype='int32')
        elif n_dimensions:
            self.dimension_indices = SP.arange(0,n_dimensions)
        self.n_dimensions = self.dimension_indices.max()+1-self.dimension_indices.min()
        pass

    def _filter_x(self, x):
        """
        Filter out the dimensions, which not correspond to a feature.
        (Only self.dimension_inices are needed)
        """
        return x[:,self.dimension_indices]

    def _filter_input_dimensions(self, x1, x2):
        """
        Filter out all dimensions, which not correspond to a feature.
        Filter is self.dimension_indices.
        
        Returns : filtered x1, filtered x2
        """
        if x2 == None:
            return self._filter_x(x1), self._filter_x(x1)
        return self._filter_x(x1), self._filter_x(x2)

    def K(self, logtheta, x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and optional x2. If only x1 is given the covariance
        matrix is computed with x1 against x1.

        **Parameters:**

        logtheta : [double]

            The hyperparameters for which the covariance
            matrix shall be computed. :py:func:`logtheta` are the
            hyperparameters for the respective covariance function.
            For instance, the :py:class:`covar.se.SEARDCF` holds hyperparameters
            like :py:func`[Amplitude, Length-Scale(s)]`.

        x1 : [double]
        
            The training input X, which shall be
            the pointwise covariance calculated for.

        x2 : [double]
        
            The interpolation input X\`*`, which shall be
            the pointwise covariance calculated for.

        """
        print "implement K"
        pass

    def Kdiag(self,logtheta, x1):
        """
        Get diagonal of a (squared) covaraince matrix
        """
        #default: naive implementation
        return self.K(logtheta,x1).diagonal()
        
        
    def Kd(self, logtheta, x1,i):
        """
        Get Derivatives of Covariance matrix K for each given
        hyperparameter resepctively. Output matrix with
        derivatives will have the same order, as the
        hyperparameters have.

        **Parameters:**

        logtheta : [double]

            The hyperparameters for which the derivative
            covariance matrix shall be computed.

        x1 : [double]
        
            The training input X, which shall be
            the covariance derivative calculated for.

        i : int

            The index of the theta, which's partial derivative shall be
            returned.

        """
        print "please implement Kd"
    	pass

    def _pointwise_distance(self,x1,x2,L=None):
        """
        Pointwise distance between vector x1 and x2. Optionally normalized (divided) by L
        """
        if L is not None:
            x1 = x1.copy()/L
            x2 = x2.copy()/L
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
        """
        Get the active_dimensions for this covariance function, i.e.
        the indices of the feature dimensions of the training inputs, which shall
        be used for the covariance.
        """
        self.active_dimension_indices = active_dimension_indices
        pass
