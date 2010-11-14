"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a py:class:`sumCF`
or :py:class:`produtCF` with the py:class:`noiseCF`, if you want noise to be modelled by this GP.
"""

import sys
sys.path.append("../")
import scipy as SP

# import super class CovarianceFunction
from covar import CovarianceFunction

class SECF(CovarianceFunction):
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
    
    def __init__(self,n_dimensions=1,dimension_indices=None):
        if dimension_indices != None:
            self.dimension_indices = array(dimension_indices,dtype='int32')
        elif n_dimensions:
            self.dimension_indices = SP.arange(0,n_dimensions)
        self.n_dimensions = self.dimension_indices.max()+1-self.dimension_indices.min()
        self.n_hyperparameters = self.n_dimensions+1
        pass

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identification easier"""
        names = []
        names.append('Amplitude')
        for dim in self.dimension_indices:
            names.append('%d.D Length-Scale' % dim)
        return names
   
    def get_number_of_parameters(self):
        return self.n_dimensions+1;

    def K(self, logtheta, *args):
        """
        Get Covariance matrix K with given hyperparameters
        and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        x1 = args[0][:,self.dimension_indices]#[:,self.Iactive]
        if(len(args)==1):
            x2 = x1
        else:
           x2 = args[1][:,self.dimension_indices]#[:,self.Iactive]
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

    def Kd(self, logtheta, x1,i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        x1 = x1[:,self.dimension_indices]#[:,self.Iactive]
        x2 = x1
        
        # 2. exponentiate params:
        V0 = SP.exp(2*logtheta[0])
        L  = SP.exp(logtheta[1:1+self.n_dimensions])#[:,self.Iactive])
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = self._pointwise_distance(x1,x2,L)
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


    def get_default_hyperparameters(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
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

if __name__ == "__main__":
    # tests for SECF:
    covar = SECF(n_dimensions = 2, dimension_indices = [1,2])
    X = arange(0,9).reshape(-1,3)
    Xprime = arange(0,12).reshape(-1,3)
    A = 2
    L1 = 2
    L2 = .5
    logtheta = SP.log([A,L1,L2])

    # number of n_dimensions
    assert covar.get_n_dimensions() == 2
    # parameter names
    assert covar.get_hyperparameter_names() == ['Amplitude', '1.D Length-Scale', '2.D Length-Scale']
    # Kovariance matrices
    K = covar.K(logtheta,X)
    assert ((K.diagonal() == A**2).all())
    K = covar.K(logtheta,X,Xprime)
    assert ((K.diagonal() == A**2).all())
    Kd = covar.Kd(logtheta,X)
    assert Kd.shape == (covar.get_number_of_parameters(),X.shape[0],X.shape[0])
    Kd = covar.Kd(logtheta,X,Xprime)
    assert Kd.shape == (covar.get_number_of_parameters(),X.shape[0],Xprime.shape[0])
    
    # tests for SETP
    covar = SETP(n_dimensions = 2, dimension_indices = [1,2], n_replicates = 2)
    X = array([[1,2,3,0],[4,5,6,0],[2,3,4,1]])
    Xprime = linspace(0,6,20).reshape(-1,2)
    A = 2
    L1 = 2
    L2 = .5
    T1 = 1
    T2 = -1
    logtheta = SP.array[log(A),log(L1),log(L2),T1,T2]

    # number of n_dimensions
    assert covar.get_n_dimensions() == 2
    # parameter names
    assert covar.get_hyperparameter_names() == ['Amplitude',
                                                '1.D Length-Scale',
                                                '2.D Length-Scale',
                                                'Time-Parameter rep0',
                                                'Time-Parameter rep1']
    # Kovariance matrices \\TODO
    K = covar.K(logtheta,X)
    assert ((K.diagonal() == A**2).all())
    K = covar.K(logtheta,X,Xprime)
    Kd = covar.Kd(logtheta,X)
    
