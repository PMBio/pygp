"""
Base class for Gaussian process latent variable models
This is really not ready for release yet but is used by the gpasso model
"""
import sys
sys.path.append('./..')

from pygp.gpr import GP,_solve_chol,optHyper
import scipy as SP
import numpy.linalg as linalg
import pdb
import copy


def grad_checkK(K,logtheta,x1,dimensions):
    """perform grad check"""
    from pyplot import *
    L=0;
    relchange = 1E-5;
    x1_0 = x1.copy()
    n = x1.shape[0]
    nd = len(dimensions)
    diff = SP.zeros([n,nd,n,n])
    for i in xrange(n):
        for iid in xrange(nd):
            d = dimensions[iid] 
            change = relchange*x1_0[i,d]
            change = max(change,1E-5)
            x1[i,d] = x1_0[i,d] + change
            Lplus = K.K(logtheta,x1,x1)
            x1[i,d] = x1_0[i,d] - change
            Lminus = K.K(logtheta,x1,x1)
            x1[i,d] = x1_0[i,d]

            diff[i,iid,:,:] = (Lplus-Lminus)/(2*change)
    #anal
    anal = SP.zeros([n,nd,n,n])
    for iid in xrange(nd):
        d = dimensions[iid]
        dKx = K.Kd_dx(logtheta,x1,x1,d)
        dKx_diag = K.Kd_dx_diag(logtheta,x1,d)
        for i in xrange(n):
            anal[i,iid,i,:] = dKx[i,:]
            anal[i,iid,:,i] = dKx[i,:]
            anal[i,iid,i,i] = dKx_diag[i]
            
    delta = anal -diff
    print "delta"
    pdb.set_trace()
    pass


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
    
    def __init__(self,**kw_args):
        """gplvm_dimensions: dimensions to learn using gplvm, default -1; i.e. all"""
        GP.__init__(self,**kw_args)
        self.gplvm_dimensions = None



    def setData(self,gplvm_dimensions=None,**kw_args):
        GP.setData(self,**kw_args)
        #handle non-informative gplvm_dimensions vector
        if gplvm_dimensions is None:
            self.gplvm_dimensions = SP.arange(self.x.shape[1])
        else:
            self.gplvm_dimensions = gplvm_dimensions
        
    def _update_inputs(self,hyperparams):
        """update the inputs from gplvm models if supplied as hyperparms"""
        #update
        self.x[:,self.gplvm_dimensions] = hyperparams['x']
        pass
   
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
        #grad check of the covariance ? 
        #grad_checkK(self.covar,hyperparams['covar'],self.x,self.gplvm_dimensions)
        
        dlMl = SP.zeros([self.n,len(self.gplvm_dimensions)])

        #K^-1 * y from cache:
        W = self._covar_cache['W']

        #the standard procedure would be
        #dlMl[n,i] = 0.5*SP.dot(W,dKx_n,i).trace()
        #we can calcualte all the derivatives efficiently; see also interface of Kd_dx of covar
                   
        for i in xrange(len(self.gplvm_dimensions)):
            d = self.gplvm_dimensions[i]
            #dKx is general, not knowing that we are computing the diagonal:
            dKx = self.covar.Kd_dx(hyperparams['covar'],self.x,self.x,d)
            dKx_diag = self.covar.Kd_dx_diag(hyperparams['covar'],self.x,d)
            #set diagonal
            dKx.flat[::(dKx.shape[1]+1)]=dKx_diag
            #precalc elementwise product of W and K
            WK = W*dKx
            if 0:
                #explicit calculation, slow!
                #this is only in here to see what is done
                for n in xrange(self.n):
                    dKxn = SP.zeros([self.n,self.n])
                    dKxn[n,:] = dKx[n,:]
                    dKxn[:,n] = dKx[n,:]
                    dlMl[n,i] = 0.5*SP.dot(W,dKxn).trace()
                    pass
            if 1:
                #fast calculation
                #we need twice the sum WK because of the matrix structure above, WK.diagonal() accounts for the double counting
                dlMl[:,i] = 0.5*( 2*WK.sum(axis=1) - WK.diagonal() )
            pass
        RV = {'x':dlMl}
        return RV
        

if __name__ =='__main__':
    import sys
    from pygp.covar import linear, noise, combinators
    import pygp.gpr as GPR
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

    gpr = GPR.GP(covar_func=covariance,x=X,y=Y[:,0])
    
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
    [opt_model_params,opt_lml]=GPR.optHyper(gplvm,hyperparams,gradcheck=False)
    Xo = opt_model_params['x']
    

    for k in xrange(K):
        print SP.corrcoef(Spca[:,k],S[:,k])

    for k in xrange(K):
        print SP.corrcoef(Xo[:,k],S[:,k])
