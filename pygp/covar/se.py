"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a :py:class:`pygp.covar.combinators.SumCF`
or :py:class:`pygp.covar.combinators.ProductCF` with the :py:class:`pygp.covar.noise.NoiseISOCF`, if you want noise to be modelled by this GP.
"""

import scipy as SP
import logging as LG
from pygp.covar import CovarianceFunction
import dist
import pdb

class SqexpCFARD(CovarianceFunction):
    """
    Standart Squared Exponential Covariance function.

    **Parameters:**
    
    - dimension : int
        The dimension of this SE. For instance a 2D SE has
        hyperparameters like::
        
          covar_hyper = [Amplitude,1stD Length-Scale, 2ndD Length-Scale]

    - dimension_indices : [int]
        Optional: The indices of the n_dimensions in the input.
        For instance the n_dimensions of inputs are in 2nd and
        4th dimension dimension_indices would have to be [1,3].

    """   
    def __init__(self,*args,**kwargs):
        super(SqexpCFARD, self).__init__(*args,**kwargs)
        self.n_hyperparameters = self.n_dimensions+1
        pass

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = []
        names.append('SECF Amplitude')
        for dim in self.dimension_indices:
            names.append('%d.D Length-Scale' % dim)
        return names
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.n_dimensions+1;

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        and inputs X=x1 and X\`*`=x2.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. get inputs
        x1, x2 = self._filter_input_dimensions(x1,x2)
        #2. exponentialte parameters
        V0 = SP.exp(2*theta[0])
        L  = SP.exp(theta[1:1+self.n_dimensions])
        sqd = dist.sq_dist(x1/L,x2/L)
        #3. calculate the whole covariance matrix:
        rv = V0*SP.exp(-0.5*sqd)
        return rv

    def Kdiag(self,theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #diagonal is independent of data
        x1 = self._filter_x(x1)
        V0 = SP.exp(2*theta[0])
        return V0*SP.exp(0)*SP.ones([x1.shape[0]])
    
    def Kgrad_theta(self, theta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = self._filter_x(x1)
        # 2. exponentiate params:
        V0 = SP.exp(2*theta[0])
        L  = SP.exp(theta[1:1+self.n_dimensions])
        # calculate squared distance manually as we need to dissect this below
        x1_ = x1/L
        d  = dist.dist(x1_,x1_)
        sqd = (d*d)
        sqdd = sqd.sum(axis=2)
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*SP.exp(-0.5*sqdd)
        if i==0:
            return 2*rv0
        else:
            return rv0*sqd[:,:,i-1]

    
    def Kgrad_x(self,theta,x1,x2,d):
        """
        The partial derivative of the covariance matrix with
        respect to x, given hyperparameters `theta`.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        # if we are not meant return zeros:
        if(d not in self.dimension_indices):
            return SP.zeros([x1.shape[0],x2.shape[0]])
        rv = self.K(theta,x1,x2)
#        #1. get inputs and dimension
        x1, x2 = self._filter_input_dimensions(x1,x2)
        d -= self.dimension_indices.min()
#        #2. exponentialte parameters
#        V0 = SP.exp(2*theta[0])
#        L  = SP.exp(theta[1:1+self.n_dimensions])[d]
        L2 = SP.exp(2*theta[1:1+self.n_dimensions])
#        # get squared distance in right dimension:
#        sqd = dist.sq_dist(x1[:,d]/L,x2[:,d]/L)
#        #3. calculate the whole covariance matrix:
#        rv = V0*SP.exp(-0.5*sqd)
        #4. get non-squared distance in right dimesnion:
        nsdist = -dist.dist(x1,x2)[:,:,d]/L2[d]
        
        return rv * nsdist
    
    def Kgrad_xdiag(self,theta,x1,d):
        """"""
        #digaonal derivative is zero because d/dx1 (x1-x2)^2 = 0
        #because (x1^d-x1^d) = 0
        RV = SP.zeros([x1.shape[0]])
        return RV
