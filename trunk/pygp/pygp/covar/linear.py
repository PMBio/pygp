"""
Classes for linear covariance function
======================================

::
    
    K = \sum_d alpha_d^2 * x_d *x_d^{\t}

"""

import scipy as SP

from pygp.covar import CovarianceFunction,CF_Kd_dx


class LinearCFISO(CF_Kd_dx):
    """
    isotropic linear covariance function with a single hyperparameter
    """

    def __init__(self,*args,**kw_args):
        CF_Kd_dx.__init__(self,*args,**kw_args)
        self.n_hyperparameters = 1

        
    def K(self,logtheta,x1,x2=None):
        #get input data:
        x1, x2 = self._filter_input_dimensions(x1,x2)

        # 2. exponentiate params:
        A  = SP.exp(2*logtheta[0])
        RV = A*SP.dot(x1,x2.T)
        return RV

    def Kd(self,logtheta,x1,i):
        RV = self.K(logtheta,x1)
        #derivative w.r.t. to amplitude
        RV*=2
        return RV

    def Kdiag(self,logtheta,x1,i):
        x1 = self._filter_x(x1)
        RV = SP.dot(x1,x1).sum(axis=1)
        RV*=2
        return RV

    def Kd_dx(self,logtheta,x1,d):
        RV = SP.zeros([x1.shape[0],x1.shape[0]])
        if d not in self.dimension_indices:
            return RV
        #no filtering here, as we have dimension argument anyway
        #get going:
        RV[:,:] = 2*x1[:,d]
        return RV



class LinearCF(CovarianceFunction):

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
    

