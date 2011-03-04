"""
Base class for Gaussian process latent variable models
This is really not ready for release yet but is used by the gpasso model
"""
import sys
sys.path.append('./..')

from pygp.gp import GP
from pygp.opt import opt_hyper

import scipy as SP
import numpy.linalg as linalg

def PCA(Y,components):
    """run PCA, retrieving the first (components) principle components
    return [s0,w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices = 0);
    [s0,w0] = [sv[0][:,0:components], SP.dot(SP.diag(sv[1]),sv[2]).T[:,0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0,w0]

    
class GPLVM(GP):
    """
    derived class form GP offering GPLVM specific functionality
    
    
    """
    __slots__ = ["gplvm_dimensions"]
    
    def __init__(self,gplvm_dimensions=None,**kw_args):
        """gplvm_dimensions: dimensions to learn using gplvm, default -1; i.e. all"""
        self.gplvm_dimensions = gplvm_dimensions
        GP.__init__(self,**kw_args)



    def setData(self,*args,**kw_args):
        GP.setData(self,*args,**kw_args)
        #handle non-informative gplvm_dimensions vector
        if self.gplvm_dimensions is None:
            self.gplvm_dimensions = SP.arange(self.x.shape[1])
        
    def _update_inputs(self,hyperparams):
        """update the inputs from gplvm models if supplied as hyperparms"""

        #TODO: x_gplvm does not update the right thing
        #self.x = hyperparams['x']
        #get gplvm index view
        x_gplvm = self._getXGPLVM()
        #check everything is consistent
        assert self.x[:,self.gplvm_dimensions].shape==x_gplvm.shape, 'shape error for latent x'
        #update
        self.x[:,self.gplvm_dimensions] = hyperparams['x']
        pass

    def _getXGPLVM(self):
        """get the GPLVM part of the inputs"""
        return self.x[:,self.gplvm_dimensions]        
   
    def lMl(self,hyperparams,priors=None,**kw_args):
        """
        Calculate the log Marginal likelihood
        for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ... }
            The hyperparameters for the log marginal likelihood.

        priors : [:py:class:`lnpriors`]
            the prior beliefs for the hyperparameter values

        Ifilter : [bool]
            Denotes which hyperparameters shall be optimized.
            Thus ::

                Ifilter = [0,1,0]

            has the meaning that only the second
            hyperparameter shall be optimized.

        kw_args :
            All other arguments, explicitly annotated
            when necessary.
            
        """
        if 'x' in hyperparams:
            self._update_inputs(hyperparams)

        #covariance hyper
        lMl = self._lMl_covar(hyperparams)

        
        #account for prior
        if priors is not None:
            plml = self._lml_prior(hyperparams,priors=priors,**kw_args)
            lMl -= SP.array([p[:,0].sum() for p in plml.values()]).sum()
        return lMl
        

    def dlMl(self,hyperparams,priors=None,**kw_args):
        """
        Returns the log Marginal likelihood for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ...}
            The hyperparameters which shall be optimized and derived

        priors : [:py:class:`lnpriors`]
            The hyperparameters which shall be optimized and derived

        """
        # Ideriv : 
        #      indicator which derivativse to calculate (default: all)

        if 'x' in hyperparams:
            self._update_inputs(hyperparams)
            
        RV = self._dlMl_covar(hyperparams)
        #
        if 'x' in hyperparams:
            RV_ = self.dlMl_x(hyperparams)
            #update RV
            RV.update(RV_)

        #prior
        if priors is not None:
            plml = self._lml_prior(hyperparams,priors=priors,**kw_args)
            for key in RV.keys():
                RV[key]-=plml[key][:,1]                       
        return RV


    ####PRIVATE####

    def dlMl_x(self,hyperparams):
        """GPLVM derivative w.r.t. to latent variables
        """
        dlMl = SP.zeros_like(self.x)
        W = self._covar_cache['W']

        #the standard procedure would be
        #dlMl[n,i] = 0.5*SP.odt(W,dKx_n,i).trace()
        #we can calcualte all the derivatives efficiently; see also interface of Kd_dx of covar
        for i in xrange(len(self.gplvm_dimensions)):
            d = self.gplvm_dimensions[i]
            dKx = self.covar.Kd_dx(hyperparams['covar'],self.x,d)
            dlMl[:,i] = 0.5*(W*dKx).sum(axis=1)
            pass
        RV = {'x':dlMl}
        return RV
        

if __name__ =='__main__':
    from pygp.covar import linear, noise, combinators
    
    import logging as LG
    LG.basicConfig(level=LG.DEBUG)
    
    #1. simulate data
    N = 100
    K = 3
    D = 10

    
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)
    
    Y = SP.dot(W,S.T).T
    Y+= 0.5*SP.random.randn(N,D)
  
    [Spca,Wpca] = PCA(Y,K)
    
    #reconstruction
    Y_ = SP.dot(Spca,Wpca.T)
    
    #construct GPLVM model
    linear_cf = linear.LinearCFISO(n_dimensions=K)
    noise_cf = noise.NoiseISOCF()
    covariance = combinators.SumCF((linear_cf,noise_cf))


    #no inputs here (later SNPs)
    X = Spca.copy()
    #X = SP.random.randn(N,K)
    gplvm = GPLVM(covar_func=covariance,x=X,y=Y)

    gpr = GP(covar_func=covariance,x=X,y=Y[:,0])
    
    #construct hyperparams
    covar = SP.log([1.0,0.1])

    #X are hyperparameters, i.e. we optimize over them also

    #1. this is jointly with the latent X
    X_ = X.copy()
    hyperparams = {'covar': covar, 'x': X_}
    

    #for testing just covar params alone:
    #hyperparams = {'covar': covar}
    
    #evaluate log marginal likelihood
    lml = gplvm.lMl(hyperparams=hyperparams)
    [opt_model_params,opt_lml]= opt_hyper(gplvm,hyperparams,gradcheck=False)
    Xo = opt_model_params['x']
    

    for k in xrange(K):
        print SP.corrcoef(Spca[:,k],S[:,k])

    for k in xrange(K):
        print SP.corrcoef(Xo[:,k],S[:,k])
