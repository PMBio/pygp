"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a :py:class:`pygp.covar.combinators.SumCF`
or :py:class:`pygp.covar.combinators.ProductCF` with the :py:class:`pygp.covar.noise.NoiseISOCF`, if you want noise to be modelled by this GP.
"""

import logging as LG
from pygp.covar import CovarianceFunction
import dist
import pdb
import numpy

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
        V0 = numpy.exp(2*theta[0])
        L  = numpy.exp(theta[1:1+self.n_dimensions])
        sqd = dist.sq_dist(x1/L,x2/L)
        #3. calculate the whole covariance matrix:
        rv = V0*numpy.exp(-0.5*sqd)
        return rv

    def Kdiag(self,theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #diagonal is independent of data
        x1 = self._filter_x(x1)
        V0 = numpy.exp(2*theta[0])
        return V0*numpy.exp(0)*numpy.ones([x1.shape[0]])
    
    def Kgrad_theta(self, theta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = self._filter_x(x1)
        # 2. exponentiate params:
        V0 = numpy.exp(2*theta[0])
        L  = numpy.exp(theta[1:1+self.n_dimensions])
        # calculate squared distance manually as we need to dissect this below
        x1_ = x1/L
        d  = dist.dist(x1_,x1_)
        sqd = (d*d)
        sqdd = sqd.sum(axis=2)
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*numpy.exp(-0.5*sqdd)
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
            return numpy.zeros([x1.shape[0],x2.shape[0]])
        rv = self.K(theta,x1,x2)
#        #1. get inputs and dimension
        x1, x2 = self._filter_input_dimensions(x1,x2)
        d -= self.dimension_indices.min()
#        #2. exponentialte parameters
#        V0 = numpy.exp(2*theta[0])
#        L  = numpy.exp(theta[1:1+self.n_dimensions])[d]
        L2 = numpy.exp(2*theta[1:1+self.n_dimensions])
#        # get squared distance in right dimension:
#        sqd = dist.sq_dist(x1[:,d]/L,x2[:,d]/L)
#        #3. calculate the whole covariance matrix:
#        rv = V0*numpy.exp(-0.5*sqd)
        #4. get non-squared distance in right dimesnion:
        nsdist = -dist.dist(x1,x2)[:,:,d]/L2[d]
        
        return rv * nsdist
    
    def Kgrad_xdiag(self,theta,x1,d):
        """"""
        #digaonal derivative is zero because d/dx1 (x1-x2)^2 = 0
        #because (x1^d-x1^d) = 0
        RV = numpy.zeros([x1.shape[0]])
        return RV


class SqexpCFARDwPsyStat(SqexpCFARD):
    def psi_0(self, theta, mean, variance, inducing_points):
        return mean.shape[0] * theta[0]

    def psi_0grad_theta(self, theta, mean, variance, inducing_points, i):
        """
        Gradients with respect to each hyperparameter, respectively. 
        """
        # no gradients except of A
        if i == 0:
            return mean.shape[0]
        return 0
    
    def psi_1(self, theta, mean, variance, inducing_points):
        alpha = theta[1:1+self.get_n_dimensions()]
        return theta[0] * self._exp_psi_1(alpha, mean, variance, inducing_points)
        
    def _inner_sum_psi_1(self, alpha, mean, variance, inducing_points):
        distances = alpha * (mean.T - inducing_points)**2
        normalizing_factor = (alpha * variance) + 1
        summand = -.5 * ((distances / normalizing_factor.T) + numpy.log(normalizing_factor.T))
        #import pdb;pdb.set_trace()
        return summand
    
    def psi_1grad_theta(self, theta, mean, variance, inducing_points, i):
        if i==0:
            return self._exp_psi_1(theta[1:1+self.get_n_dimensions()], mean, variance, inducing_points)
        q = i-1
        distances = (numpy.atleast_2d(mean[:,q]).T - numpy.atleast_2d(inducing_points[:,q]))**2
        alpha_q = theta[i]
        S_q = numpy.atleast_2d(variance[:,q]).T
        normalizing_factor = (alpha_q * S_q) + 1
        dexp_psi_1 = -.5 * ( (distances / normalizing_factor**2) + (S_q / normalizing_factor))  
        return self.psi_1(theta, mean, variance, inducing_points) * dexp_psi_1

    def _exp_psi_1(self, alpha, mean, variance, inducing_points):
        return numpy.exp(numpy.add.reduce(\
                                [self._inner_sum_psi_1(alpha[q], 
                                                       numpy.atleast_2d(mean[:,q]), 
                                                       numpy.atleast_2d(variance[:,q]), 
                                                       numpy.atleast_2d(inducing_points[:,q]))\
                                 for q in xrange(self.get_n_dimensions())], 
                                axis=0))

    
    def psi_2(self, theta, mean, variance, inducing_points):
        alpha = theta[1:1+self.get_n_dimensions()]
        summ = numpy.add.reduce([
                              numpy.exp(numpy.add.reduce( [
                                                        self._inner_sum_wrapper_psi_2(q, n, alpha, mean, variance, inducing_points) 
                                                        for q in xrange(self.get_n_dimensions())
                                                        ], 
                                                      axis = 0))
                              for n in xrange(mean.shape[0])
                              ], axis = 0)
#        summ = numpy.zeros((inducing_points.shape[0], inducing_points.shape[0]))
#        for n in xrange(mean.shape[0]):
#            inner_inner = numpy.zeros((inducing_points.shape[0], inducing_points.shape[0]))
#            for q in xrange(self.n_dimensions):
#                inner_inner += self._inner_sum_wrapper_psi_2(q, n, alpha, mean, variance, inducing_points)
#            summ += numpy.exp(inner_inner)
            
        return mean.shape[0] * theta[0]**2 * summ

    def _inner_sum_psi_2(self, alpha, mean, variance, inducing_points):
        inducing_points_distances = ((inducing_points.T - inducing_points)**2 ) / 2.
        normalizing_factor = (alpha * variance) + .5
        
        inducing_points_cross_means = (inducing_points.T + inducing_points) / 2.
        inducing_points_cross_distances = ((mean - inducing_points_cross_means)**2) / normalizing_factor
        
        summand = -.5 * ( alpha * (inducing_points_distances + inducing_points_cross_distances) + numpy.log(2 * normalizing_factor.T))
        if 0 or numpy.iscomplex(summand).any():
            import pdb;pdb.set_trace()
        return summand
    
    def _inner_sum_wrapper_psi_2(self, q, n, alpha, mean, variance, inducing_points):
        return self._inner_sum_psi_2(alpha[q],
                                     numpy.atleast_2d(mean[n,q]), 
                                     numpy.atleast_2d(variance[n,q]), 
                                     numpy.atleast_2d(inducing_points[:,q]))
        