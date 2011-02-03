"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a :py:class:`covar.combinators.SumCF`
or :py:class:`covar.combinators.ProductCF` with the :py:class:`covar.noise.NoiseISOCF`, if you want noise to be modelled by this GP.
"""

import scipy as SP

# import super class CovarianceFunction
#from covar import CovarianceFunction
# import super class CF_Kd_dx

from pygp.covar import CF_Kd_dx

class SEARDCF(CF_Kd_dx):
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
    #__slots__= ["n_hyperparameters",
    #            "n_dimensions",
    #            "dimension_indices",
    #            "active_dimension_indices"]
    
    def __init__(self,*args,**kw_args):
        CF_Kd_dx.__init__(self,*args,**kw_args)
        self.n_hyperparameters = self.n_dimensions+1
        pass

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = []
        names.append('Amplitude')
        for dim in self.dimension_indices:
            names.append('%d.D Length-Scale' % dim)
        return names
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.n_dimensions+1;

    def K(self, logtheta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        and inputs X=x1 and X\`*`=x2.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        x1, x2 = self._filter_input_dimensions(x1,x2)
        # 2. exponentiate params:
        V0 = SP.exp(2*logtheta[0])
        L  = SP.exp(logtheta[1:1+self.n_dimensions])#[self.Iactive])
        # calculate the distance betwen x1,x2 for each dimension separately, reweighted by L. 
        dd = self._pointwise_distance(x1,x2,L)
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
        #3. calculate the whole covariance matrix:
        rv = V0*SP.exp(-0.5*sqd)
        return rv

    def Kd(self, logtheta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        x1 = self._filter_x(x1)
        # 2. exponentiate params:
        V0 = SP.exp(2*logtheta[0])
        L  = SP.exp(logtheta[1:1+self.n_dimensions])
        # calculate the distance between
        # x1,x2 for each dimension separately.
        dd = self._pointwise_distance(x1,x1,L)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqdd = sqd
        sqd = sqd.sum(axis=2)
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*SP.exp(-0.5*sqd)

        if i==0:
            return 2*rv0
        else:
            return rv0*sqdd[:,:,i-1]

    def Kdiag(self,logtheta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        #default: naive implementation
        LG.debug("SEARDCF: Kdiag: Default unefficient implementation!")
        return self.K(logtheta,x1).diagonal()
    

    def Kd_dx(self,logtheta,x):
        """
        The partial derivative of the covariance matrix with
        respect to x, given hyperparameters `logtheta`.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """

        #TODO: I am pretty sure this only works for a single dimension, right?
        L = SP.exp(logtheta[1:1+self.n_dimensions])
        dd = self._pointwise_distance(x,x,-(L**2))
        return self.K(logtheta,x) * dd.transpose(2,0,1)

    def get_default_hyperparameters(self,x=None,y=None):
        """
        Return default parameters for a particular
        dataset (optional).
        """
        #start with data independent default
        rv = ones(self.n_hyperparameters)
        #start with a smallish variance
        rv[-1] = 0.1
        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)

