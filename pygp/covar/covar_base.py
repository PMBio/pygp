"""
Covariance Functions
====================

We implemented several different covariance functions (CFs) you can work with. To use GP-regression with these covariance functions it is highly recommended to model the noise of the data in one extra covariance function (:py:class:`pygp.covar.noise.NoiseISOCF`) and add this noise CF to the CF you are calculating by putting them all together in one :py:class:`pygp.covar.combinators.SumCF`.

For example to use the squared exponential CF with noise::

    from pygp.covar import se, noise, combinators
    
    #Feature dimension of the covariance: 
    dimensions = 1
    
    SECF = se.SEARDCF(dim)
    noise = noise.NoiseISOCF()
    covariance = combinators.SumCF((SECF,noise))

"""

# import scipy:
import scipy as SP
import logging as LG


class CovarianceFunction(object):
    """
    *Abstract super class for all implementations of covariance functions:*
    
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
                "n_dimensions",
                "dimension_indices"]

    def __init__(self,n_dimensions=1,dimension_indices=None):
        self.n_hyperparameters = SP.nan
        self.n_dimensions = n_dimensions
        #set relevant dimensions for coveriance function
        #either explicit index
        if dimension_indices is not None:
            self.dimension_indices = SP.array(dimension_indices,dtype='int32')
        #or via the number of used dimensions
        elif n_dimensions:
            self.dimension_indices = SP.arange(0,n_dimensions)
        self.n_dimensions = self.dimension_indices.max()+1-self.dimension_indices.min()
        pass

    def K(self, theta, x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and optional x2.
        If only x1 is given the covariance
        matrix is computed with x1 against x1.

        **Parameters:**

        theta : [double]

            The hyperparameters for which the covariance
            matrix shall be computed. *theta* are the
            hyperparameters for the respective covariance function.
            For instance :py:class:`pygp.covar.se.SEARDCF`
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
        print("%s: Function K not yet implemented"%(self.__class__))
        return None
        pass

    def Kdiag(self,theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        *Default*: Return the diagonal of the fully
        calculated Covariance Matrix. This may be overwritten
        more efficiently.
        """
        LG.debug("No Kdiag specified; Return default naive computation.")
        #print("%s: Function Kdiag not yet implemented"%(self.__class__))
        return self.K(theta,x1).diagonal()
        
        
    def Kgrad_theta(self, theta, x1, i):
        """
        Get partial derivative of covariance matrix K
        with respect to the i-th given
        hyperparameter `theta[i]`.

        **Parameters:**

        theta : [double]

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

    def Kgrad_x(self,theta,x1,x2,d):
        """
        Partial derivatives of K[X1,X2] with respect to x1(:)^d
        RV: matrix of size [x1,x2] containin all values of
        d/dx1^{i,d} K(X1,X2)
        """
        LG.critical("implement Kgrad_x")
        #print("%s: Function Kgrad_x not yet implemented"%(self.__class__))
        return None

    def Kgrad_xdiag(self,theta,x1,d):
        """
        Diagonal of partial derivatives of K[X1,X1] w.r.t. x1(:)^d
        RV: vector of size [x1] cotaining all partial derivatives
        d/dx1^{i,d} diag(K(X1,X2))
        """
        LG.critical("implement the partial derivative w.r.t x")
        #print("%s: Function Kgrad_xdiag not yet implemented"%(self.__class__))
        return None

    def get_hyperparameter_names(self):
        """
        Return names of hyperparameters to make
        identification easier
        """
        print "%s: WARNING: hyperparamter name not specified yet!" % (self.__class__)
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
        return {'covar':SP.array([])}

    def get_n_dimensions(self):
        """
        Returns the number of dimensions, specified by user.
        """
        return self.n_dimensions
    
    def set_dimension_indices(self,active_dimension_indices = None):
        """
        Get the active_dimensions for this covariance function, i.e.
        the indices of the feature dimensions of the training inputs, which shall
        be used for the covariance.
        """
        self.dimension_indices = active_dimension_indices
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
