"""
Covariance Functions
====================

We implemented several different covariance functions (CFs) you can work with. To use GP-regression with these covariance functions it is highly recommended to model the noise of the data in one extra covariance function (:py:class:`covar.noise.NoiseISOCF`) and add this noise CF to the CF you are calculating by putting them all together in one :py:class:`covar.combinators.SumCF`.

For example to use the squared exponential CF with noise::

    from covar import se, noise, combinators
    
    #Feature dimension of the covariance: 
    dimensions = 1
    
    SECF = se.SEARDCF(dim)
    noise = noise.NoiseISOCF()
    covariance = combinators.SumCF((SECF,noise))

**Abstract super class for all implementations of covariance functions:**

"""


__all__ = ["se","sq_dist","combinators","noise","linear"]

# import python / numpy:
import scipy as SP

import sq_dist

import logging as LG

class CovarianceFunction(object):
    """
    **Important:** *All Covariance Functions have
    to inherit from this class in order to work
    properly with this GP framework.*

    **Parameters:**

    n_dimensions : int

        standard: n_dimension = 1. The number of
        dimensions (i.e. features) this CF holds.

    dimension_indices : [int]

        The indices of dimensions (features) this CF takes into account.

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

    def K(self, logtheta, x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and optional x2.
        If only x1 is given the covariance
        matrix is computed with x1 against x1.

        **Parameters:**

        logtheta : [double]

            The hyperparameters for which the covariance
            matrix shall be computed. *logtheta* are the
            hyperparameters for the respective covariance function.
            For instance :py:class:`covar.se.SEARDCF`
            holds hyperparameters as follows::

                `[Amplitude, Length-Scale(s)]`.

        x1 : [double]
        
            The training input X, for which the
            pointwise covariance shall be calculated.

        x2 : [double]
        
            The interpolation input X\`*`, for which the
            pointwise covariance shall be calculated.

        """
        LG.critical("implement K")
        print "implement K"
        pass

    def Kdiag(self,logtheta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        *Default*: Return the diagonal of the fully
        calculated Covariance Matrix. This may be overwritten
        more efficiently.
        """
        LG.debug("No Kdiag specified; Return default naive computation.")
        return self.K(logtheta,x1).diagonal()
        
        
    def Kd(self, logtheta, x1, i):
        """
        Get partial derivative of covariance matrix K
        with respect to the i-th given
        hyperparameter `logtheta[i]`.

        **Parameters:**

        logtheta : [double]

            The hyperparameters for covariance.

        x1 : [double]
        
            The training input X.

        i : int

            The index of the hyperparameter, which's
            partial derivative shall be returned.

        """
        LG.critical("implement Kd")
        print "please implement Kd"
    	pass

    def get_hyperparameter_names(self):
        """
        Return names of hyperparameters to make
        identification easier
        """
        return []

    def get_number_of_parameters(self):
        """
        Return number of hyperparameters, specified by user.
        """
        return self.n_hyperparameters
    
    def get_default_hyperparameters(self,x=None,y=None):
        """
        Return default hyperpameters.
        
        *Default:*: No hyperparameters; Returns an empty array.
        """
        return array([])

    def get_n_dimensions(self):
        """
        Returns the number of dimensions, specified by user.
        """
        return self.n_dimensions
    
    def set_active_dimensions(self,active_dimension_indices = None):
        """
        Get the active_dimensions for this covariance function, i.e.
        the indices of the feature dimensions of the training inputs, which shall
        be used for the covariance.
        """
        self.dimension_indices = active_dimension_indices
        pass

    def get_Iexp(self, logtheta):
        """
        Return the indices of which hyperparameters of logtheta
        are to be exponentiated before optimizing them.

        *Default:* Exponentiate all hyperparameters!

        **Parameters:**

        logtheta : [double]

            Hyperparameters of CF.
        """
        LG.debug("%s: No explicit Iexp given; default exponentiate all hyperparameters." % (self.__repr__))
        return SP.array(SP.ones_like(logtheta),dtype='bool')

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

    def _pointwise_distance(self,x1,x2,L=None):
        """
        Pointwise distance between vector x1 and x2.
        Optionally normalized (divided) by L.
        """
        if L is not None:
            x1 = x1.copy()/L
            x2 = x2.copy()/L
        return sq_dist.dist(x1,x2)


class CF_Kd_dx(CovarianceFunction):
    """
    Covariance function, which provides partial derivative with
    respect to input x.
    
    **Parameters:**
    See :py:class:`covar.CovarianceFunction
    """
    
    def Kd_dx(self,logtheta,x,d):
        """
        Matrix derivatives of the self covariance with respect to dimension d
        RV = d/dx_n,d K(X,X)_i,j = k(x_i,x_j)
        i.e. RV[n,:] = d/dx_n,d k(x_n,:)
        Note: all covariance functions always return an nXn matrix. If d is not in active dimensions the matrix is zeros

        #TODO: update description properly
        Partial derivative of covariance matrix K with respect
        to training input x.

        *Default*: return K(logtheta,x). **!This might be wrong!**

        **Parameters:**
        logtheta : [double]

            Hyperparameters of CF.

        x : [double]
        
            The training input X. The return value is
            the partial derivative of the covariance
            matrix with respect to given training input X.

        """
        LG.critical("implement Kd_dx")
        print "implement Kd_dx"
        pass
