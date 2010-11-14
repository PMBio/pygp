"""simple class for a linear covariance function
K = \sum_d alpha_d^2 * x_d *x_d^{\t}
"""

import scipy as SP
import pdb

from covar import CovarianceFunction


class LinearCovariance(CovarianceFunction):

    def __init__(self,n_dimensions=1,dimension_indices=None):
        if dimension_indices != None:
            self.dimension_indices = SP.array(dimension_indices,dtype='int32')
        elif n_dimensions:
            self.dimension_indices = SP.arange(0,n_dimensions)
        self.n_dimensions = self.dimension_indices.max()+1-self.dimension_indices.min()
        self.n_hyperparameters = self.n_dimensions

    def get_hyperparameter_names(self):
        names = []
        names.append('Amplitude')
        return names

    def K(self,logtheta,x1,x2=None):
        if x2 is None:
            x2 = x1
        # 2. exponentiate params:
        L  = SP.exp(2*logtheta[0:self.n_dimensions])
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        for i in xrange(self.n_dimensions):
            iid = self.dimension_indices[i]
            RV+=L[i]*SP.dot(x1[:,iid:iid+1],x2[:,iid:iid+1].T)
        return RV

    def Kd(self,logtheta,x1,i):
        iid = self.dimension_indices[i]
        Li = SP.exp(2*logtheta[i])
        RV = 2*Li*SP.dot(x1[:,iid:iid+1],x1[:,iid:iid+1].T)
        return RV
    

