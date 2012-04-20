"""
Classes for linear covariance function
======================================
Linear covariance functions

LinearCFISO
LinearCFARD

"""

#TODO: fix ARD covariance 

import numpy
from pygp.covar import CovarianceFunction
import pdb
from pdb import Pdb

class LinearCFISO(CovarianceFunction):
    """
    isotropic linear covariance function with a single hyperparameter
    """

    def __init__(self,*args,**kw_args):
        super(LinearCFISO, self).__init__(*args,**kw_args)
        self.n_hyperparameters = 1

    def K(self,theta,x1,x2=None):
        #get input data:
        x1, x2 = self._filter_input_dimensions(x1,x2)
        # 2. exponentiate params:
        A  = numpy.exp(2*theta[0])
        RV = A*numpy.dot(x1,x2.T)
        return RV

    def Kdiag(self,theta,x1):
        x1 = self._filter_x(x1)
        RV = numpy.dot(x1,x1.T).sum(axis=1)
        RV*=2
        return RV


    def Kgrad_theta(self,theta,x1,i):
        assert i==0, 'LinearCF: Kgrad_theta: only one hyperparameter for linear covariance'
        RV = self.K(theta,x1)
        #derivative w.r.t. to amplitude
        RV*=2
        return RV


    def Kgrad_x(self,theta,x1,x2,d):
        x1, x2 = self._filter_input_dimensions(x1,x2)
        RV = numpy.zeros([x1.shape[0],x2.shape[0]])
        if d not in self.dimension_indices:
            return RV
        d-=self.dimension_indices.min()
        A = numpy.exp(2*theta[0])
        RV[:,:] = A*x2[:,d]
        return RV

    
    def Kgrad_xdiag(self,theta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        x1 = self._filter_x(x1)
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.dimension_indices:
            return RV
        d-=self.dimension_indices.min()
        A = numpy.exp(2*theta[0])
        RV[:] = 2*A*x1[:,d]
        return RV

    def get_hyperparameter_names(self):
        names = []
        names.append('LinearCFISO Amplitude')
        return names
    
class LinearCF(CovarianceFunction):

    def __init__(self,n_dimensions=1,dimension_indices=None):
        if dimension_indices != None:
            self.dimension_indices = numpy.array(dimension_indices,dtype='int32')
        elif n_dimensions:
            self.dimension_indices = numpy.arange(0,n_dimensions)
        if (len(self.dimension_indices)>0):
            self.n_dimensions = len(self.dimension_indices)
            self.n_hyperparameters = self.n_dimensions
        else:
            self.n_dimensions = 0
            self.n_hyperparameters = 0
        
        
    def get_hyperparameter_names(self):
        names = []
        names.append('Amplitude')
        return names

    def K(self,logtheta,x1,x2=None):
        if x2 is None:
            x2 = x1
        # 2. exponentiate params:
        # L  = SP.exp(2*logtheta[0:self.n_dimensions])
        # RV = SP.zeros([x1.shape[0],x2.shape[0]])
        # for i in xrange(self.n_dimensions):
        #     iid = self.dimension_indices[i]
        #     RV+=L[i]*SP.dot(x1[:,iid:iid+1],x2[:,iid:iid+1].T)

        if self.n_dimensions > 0:
            M = numpy.diag(numpy.exp(2*logtheta[0:self.n_dimensions]))
            RV = numpy.dot(numpy.dot(x1[:, self.dimension_indices], M), x2[:, self.dimension_indices].T)
        else:
            RV = numpy.zeros([x1.shape[0],x2.shape[0]])
            
        return RV

    def Kgrad_theta(self,logtheta,x1,i):
        iid = self.dimension_indices[i]
        Li = numpy.exp(2*logtheta[i])
        RV = 2*Li*numpy.dot(x1[:,iid:iid+1],x1[:,iid:iid+1].T)
        return RV
    

    def Kgrad_x(self,logtheta,x1,x2,d):
        RV = numpy.zeros([x1.shape[0],x2.shape[0]])
        if d not in self.dimension_indices:
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.dimension_indices==d)[0][0]
        A = numpy.exp(2*logtheta[i])
        RV[:,:] = A*x2[:,d]
        return RV

    
    def Kgrad_xdiag(self,logtheta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.dimension_indices:
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.dimension_indices==d)[0][0]
        A = numpy.exp(2*logtheta[i])
        RV = numpy.zeros([x1.shape[0]])
        RV[:] = 2*A*x1[:,d]
        return RV

class LinearCFARD(CovarianceFunction):
    """identical to LinearCF, however alternative paramerterisation of the ard parameters"""

    def __init__(self,n_dimensions=1,dimension_indices=None):
        if dimension_indices != None:
            self.dimension_indices = numpy.array(dimension_indices,dtype='int32')
        elif n_dimensions:
            self.dimension_indices = numpy.arange(0,n_dimensions)
        if (len(self.dimension_indices)>0):
            self.n_dimensions = len(self.dimension_indices)
            self.n_hyperparameters = self.n_dimensions
        else:
            self.n_dimensions = 0
            self.n_hyperparameters = 0
        
       
    def get_hyperparameter_names(self):
        names = []
        names.append('Amplitude')
        return names

    def K(self,theta,x1,x2=None):
        if x2 is None:
            x2 = x1
        # 2. exponentiate params:
        #L  = SP.exp(-2*theta[0:self.n_dimensions])
        # L  = 1./theta[0:self.n_dimensions]

        # RV = SP.zeros([x1.shape[0],x2.shape[0]])
        # for i in xrange(self.n_dimensions):
        #     iid = self.dimension_indices[i]
        #     RV+=L[i]*SP.dot(x1[:,iid:iid+1],x2[:,iid:iid+1].T)
        RV = numpy.dot(numpy.dot(x1[:, self.dimension_indices], self._A(theta)), x2[:, self.dimension_indices].T)
        return RV

    def Kgrad_theta(self,theta,x1,i):
        iid = self.dimension_indices[i]
        #Li = SP.exp(-2*theta[i])
        Li = 1./theta[i]
        RV = -1*Li**2*numpy.dot(x1[:,iid:iid+1],x1[:,iid:iid+1].T)
        return RV
    

    def Kgrad_x(self,theta,x1,x2,d):
        RV = numpy.zeros([x1.shape[0],x2.shape[0]])
        if d not in self.dimension_indices:
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.dimension_indices==d)[0][0]
        #A = SP.exp(-2*theta[i])
        RV[:,:] = self._A(theta,i)*x2[:,d]
        return RV

    
    def Kgrad_xdiag(self,theta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.dimension_indices:
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.dimension_indices==d)[0][0]
        #A = SP.exp(-2*theta[i])
        RV = numpy.zeros([x1.shape[0]])
        RV[:] = 2*self._A(theta,i)*x1[:,d]
        return RV
    
    def _A(self, theta, i=None):
        if i is None:
            return numpy.diagflat(1./theta[0:self.n_dimensions])
        return 1./theta[i]
        # return numpy.diag(theta[0:self.n_dimensions])

class LinearCFPsiStat(LinearCF):
    def psi_0(self, theta, mean, variance, inducing_variables):
        # old = numpy.sum([numpy.trace(numpy.dot(self._A(theta), numpy.dot(mean[n:n+1,:].T, mean[n:n+1,:]) + numpy.diagflat(variance[n,:]))) for n in xrange(mean.shape[0])])
        # As you would read it:
        #        new_n = 0
        #        A = numpy.diag(theta)
        #        for i in range(mean.shape[0]):
        #            mu = mean[i:i+1].T
        #            M = numpy.dot(mu, mu.T)
        #            S = numpy.diag(variance[i])
        #            new = numpy.dot(A, M + S)
        #            new_n += numpy.trace(new)
        #        return new_n
        # as you would implement:
        new_new = numpy.sum(numpy.exp(2*theta)*(mean**2+variance))
        return new_new

    def psi_0grad_theta(self, theta, mean, variance, inducing_variables):
        # diag_sum = 0
        # As you would read it:
        #        for i in range(mean.shape[0]):
        #            mu = mean[i:i+1].T
        #            M = numpy.dot(mu, mu.T)
        #            S = numpy.diag(variance[i])
        #            new = M + S
        #            diag_sum += numpy.diag(new)
        #            
        #        full_sum = numpy.zeros_like(theta)
        #        
        #        for i in range(mean.shape[0]):
        #            mu = mean[i:i+1].T
        #            M = numpy.dot(mu, mu.T)
        #            S = numpy.diag(variance[i])
        #            new = M + S
        #            for q in range(mean.shape[1]):
        #                full_sum[q] += new[q,q]
        # as you could implement:
        new = 2*numpy.sum(numpy.exp(2*theta)*(mean**2+variance),0)
        return new

    def psi_1(self, theta, mean, variance, inducing_variables):
        A = numpy.diagflat(numpy.exp(2*theta))
        new = numpy.dot(mean, numpy.dot(A, inducing_variables.T))
        #assert numpy.allclose(new, numpy.array([[numpy.dot(numpy.atleast_2d(mean[n,:]), numpy.dot(A, numpy.atleast_2d(inducing_variables[m,:]).T)) for m in xrange(inducing_variables.shape[0])] for n in xrange(mean.shape[0])])[:,:,0,0])
        return new 
            
    def psi_1grad_theta(self, theta, mean, variance, inducing_variables):
        new = mean[:,None,:] * inducing_variables * 2*numpy.exp(2*theta)
#        old = numpy.zeros((mean.shape[0],inducing_variables.shape[0],len(theta)))
#        for i in xrange(self.get_n_dimensions()):
#            iid = self.dimension_indices[i]
#            Li = numpy.exp(2*theta[i])
#            old[:,:,i] = 2*Li*numpy.dot(mean[:,iid:iid+1],inducing_variables[:,iid:iid+1].T)
#        assert numpy.allclose(new, old)
        return new
    
    def psi_1grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        A = numpy.diagflat(numpy.exp(2*theta))
#        old = numpy.dot(mean, A)#        import pdb;pdb.set_trace()
#        #import pdb;pdb.set_trace()
        #M = inducing_variables.shape[0]
        #N = mean.shape[0] 
        #Q = mean.shape[1]
        #A = numpy.ones_like(inducing_variables) * numpy.exp(2*theta)
        #new = numpy.dot(mean, A.T)[:,:,None] * numpy.ones(mean.shape[1])
        #new = mean[:,None] * A
        prod = numpy.dot(mean,numpy.diagflat(numpy.exp(2*theta)))
        new = prod[:,None,None,:] * numpy.eye(inducing_variables.shape[0])[None,:,:,None]
#        ret = numpy.zeros((N,M,M,Q))
#        for m in xrange(M):
#            for q in xrange(Q):
#                ret[:,m,:,q] = numpy.dot(prod, self._J(M,Q,m,q).T)
        return new
    
    def psi_2(self, theta, mean, variance, inducing_variables):
#        ZA = numpy.dot(inducing_variables, numpy.diagflat(numpy.exp(2*theta)))
#        inner = numpy.dot(mean.T, mean) + numpy.diag(variance.sum(0))
#        new = numpy.dot(ZA, numpy.dot(inner, ZA.T))
#        A = numpy.diagflat(numpy.exp(2*theta))
#        old = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0]))
#        for n in xrange(mean.shape[0]):
#            for m in xrange(inducing_variables.shape[0]):
#                for mprime in xrange(inducing_variables.shape[0]):
#                    mu = mean[n:n+1,:].T
#                    M = numpy.dot(mu,mu.T)
#                    S = numpy.diagflat(variance[n:n+1,:])
#                    MS = M+S
#                    AMSA = numpy.dot(A,numpy.dot(MS,A))
#                    zAMSA = numpy.dot(inducing_variables[m:m+1,:],AMSA)
#                    zAMSAz = numpy.dot(zAMSA, inducing_variables[mprime:mprime+1,:].T)
#                    old[m,mprime] += zAMSAz
#        
        outer = inducing_variables * numpy.exp(2*theta)
        inner = numpy.dot(mean.T,mean) + numpy.diag(numpy.sum(variance,0))
        new = numpy.dot(outer, numpy.dot(inner, outer.T))
#        assert numpy.allclose(old, new)
        
        return new
            
    def psi_2grad_theta(self, theta, mean, variance, inducing_variables):
#        A = numpy.diagflat(numpy.exp(2*theta))
#        old = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0]))
#        for n in xrange(mean.shape[0]):
#            for m in xrange(inducing_variables.shape[0]):
#                for mprime in xrange(inducing_variables.shape[0]):
#                    mu = mean[n:n+1,:].T
#                    M = numpy.dot(mu,mu.T)
#                    S = numpy.diagflat(variance[n:n+1,:])
#                    MS = M+S
#                    AMSA = numpy.dot(A,numpy.dot(MS,A))
#                    zAMSA = numpy.dot(inducing_variables[m:m+1,:],AMSA)
#                    zAMSAz = numpy.dot(zAMSA, inducing_variables[mprime:mprime+1,:].T)
#                    old[m,mprime] += zAMSAz
        #theta_ = numpy.zeros_like(theta)
        outer = inducing_variables * 2*numpy.exp(2*theta)
        inner = numpy.dot(mean.T,mean) + numpy.diagflat(numpy.sum(variance,0))
        outin = numpy.dot(outer, inner)
        grad = outer[:,None] * outin
        return grad
    
    def psi_2grad_inducing_variables(self, theta, mean, variance, inducing_variables):
        M = inducing_variables.shape[0]
#        outer = inducing_variables * numpy.exp(2*theta)
#        outer_ = numpy.ones_like(inducing_variables) * numpy.exp(2*theta)
#        inner = numpy.dot(mean.T,mean) + numpy.diagflat(numpy.sum(variance,0))
#        outin = numpy.dot(outer, inner)
#        new = numpy.dot(outin, outer_.T)[:,:,None] * numpy.eye(M,M)
#        grad = numpy.rollaxis(new,0) + numpy.rollaxis(new,1)
        outer = inducing_variables * numpy.exp(2*theta)
        outer_ = numpy.ones_like(inducing_variables) * numpy.exp(2*theta)
        inner = numpy.dot(mean.T,mean) + numpy.diagflat(numpy.sum(variance,0))
        outin = numpy.dot(outer, inner)
#        new = numpy.dot(outin, numpy.exp(2*theta))
        #new = outin[:,None] * outer_
        K = (outin * outer_)
        new = numpy.eye(M,M)[:,:,None,None] * K
        grad = new.swapaxes(1,2) + new.swapaxes(0,2)
#        grad = numpy.rollaxis(grad,3).T.swapaxes
#        grad = numpy.rollaxis(new,0) + numpy.rollaxis(new,1)
#        A = numpy.diagflat(numpy.exp(2*theta))
#        K = lambda n: numpy.dot(mean[n:n+1,:], mean[n:n+1,:].T) + numpy.diagflat(variance[n:n+1,:])
#        M = inducing_variables.shape[0]
#        old = numpy.zeros((M,M,M))
#        for m in xrange(M):
#            for mprime in xrange(M):
#                old[m, mprime] = numpy.add.reduce([numpy.dot(A,numpy.dot(K(n),numpy.dot(A,inducing_variables[mprime:mprime+1,:].T))) for n in xrange(mean.shape[0])]).sum()
        return grad
    
    def _J(self, m, n, i, j):
        z = numpy.zeros((m,n))
        z[i,j] = 1.
        return z