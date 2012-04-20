'''
Created on 4 Apr 2012

@author: maxz
'''
from pygp.gp.gplvm import GPLVM
import numpy
from pygp.linalg.linalg_matrix import jitChol
from copy import deepcopy
from numpy.linalg import linalg

variational_gplvm_hyperparam_means_id = 'X'
variational_gplvm_hyperparam_vars_id = 'S'
variational_gplvm_inducing_variables_id = 'Xm'

class BayesianGPLVM(GPLVM):
    "Bayesian GPLVM"
    
    def __init__(self, gplvm_dimensions=None, n_inducing_variables=10, **kw_args):
        """gplvm_dimensions: dimensions to learn using gplvm, default -1; i.e. all"""
        super(GPLVM, self).__init__(**kw_args)
        self.setData(gplvm_dimensions=gplvm_dimensions, n_inducing_variables=n_inducing_variables, **kw_args)
        
    def setData(self, y, gplvm_dimensions=None, n_inducing_variables=10, **kw_args):
        self.y = y
        #for GPLVM models:
        self.n = self.y.shape[0]
        self.d = self.y.shape[1]
        
        #invalidate cache
        self._invalidate_cache()
        self.gplvm_dimensions = gplvm_dimensions
        self.m = n_inducing_variables
        
        # precalc some data for faster computation:
        self.TrYY = numpy.trace(numpy.dot(y,y.T))  
        
    def LML(self, hyperparams, priors=None, **kw_args):
        """
        For Bayesian GPLVM we introduce a lower bound on 
        the marginal likelihood approximating the true 
        marginal likelihood through a variational distribution q
        """
        if variational_gplvm_hyperparam_means_id not in hyperparams.keys():
            raise "gplvm hyperparameters not found"
        
        if self._is_cached(hyperparams) and not self._active_set_indices_changed:
            pass
        else:
            self._update_stats(hyperparams)
            bound  = self._compute_variational_bound(hyperparams)
            bound += self._compute_kl_divergence(hyperparams)
            
            self._covar_cache['bound'] =  -bound
            
        bound = self._covar_cache['bound']   
#        #account for prior
#        if priors is not None:
#            plml = self._LML_prior(hyperparams, priors=priors)
#            bound -= numpy.array([p[:, 0].sum() for p in plml.values()]).su
        return bound
    

    def LMLgrad(self, hyperparams, priors=None, **kw_args):
        """
        gradients w.r.t hyperparams.keys() = ['covar', 'X', 'beta', 'S', 'Xm']:
        """
        if self._is_cached(hyperparams) and not self._active_set_indices_changed:
            pass
        else:
            theta = hyperparams['covar']
            mean = hyperparams[variational_gplvm_hyperparam_means_id]
            variance = hyperparams[variational_gplvm_hyperparam_vars_id]
            inducing_variables = hyperparams[variational_gplvm_inducing_variables_id]

            LMLgrad = {}
            # Xm (inducing variables)
            #LMLgrad[variational_gplvm_inducing_variables_id] = numpy.zeros_like(inducing_variables)
            Kmmgrad_inducing_variables = numpy.array([self.covar.Kgrad_x(hyperparams['covar'], 
                                                                         inducing_variables,
                                                                         inducing_variables,
                                                                         i)
                                         for i in xrange(inducing_variables.shape[1])])
            #Kmmgrad_inducing_variables = numpy.add.reduce(Kmmgrad_inducing_variables)

            LMLgrad[variational_gplvm_inducing_variables_id] = self._LMLgrad_wrt(hyperparams['beta'], self._get_y(),
                                                                                 numpy.zeros_like(inducing_variables),
                                                                                 self.covar.psi_1grad_inducing_variables(theta,mean,variance,inducing_variables), 
                                                                                 self.covar.psi_2grad_inducing_variables(theta,mean,variance,inducing_variables),
                                                                                 Kmmgrad_inducing_variables.T)
            
            # covar (theta):
            Kmmgrad_theta = numpy.array([self.covar.Kgrad_theta(hyperparams['covar'], 
                                                               inducing_variables,
                                                                i) 
                                         for i in xrange(len(theta))]).T
            
            LMLgrad['covar'] = self._LMLgrad_wrt(hyperparams['beta'], self._get_y(), 
                                                 self.covar.psi_0grad_theta(theta,mean,variance,inducing_variables), 
                                                 self.covar.psi_1grad_theta(theta,mean,variance,inducing_variables), 
                                                 self.covar.psi_2grad_theta(theta,mean,variance,inducing_variables), 
                                                 Kmmgrad_theta.T)
            # beta:
            LMLgrad['beta'] = self._LMLgrad_beta(hyperparams)
            # X: (stat means)
            LMLgrad[variational_gplvm_hyperparam_means_id] = self._LMLgrad_means(hyperparams)
            # S: (stat variances)
            LMLgrad[variational_gplvm_hyperparam_vars_id] = self._LMLgrad_variances(hyperparams)
            # cache
            self._covar_cache['grad'] = LMLgrad
            
        return self._covar_cache['grad']
    
    def _LMLgrad_wrt(self, beta, y, psi_0grad, psi_1grad, psi_2grad, Kgrad):
        # const
#        const  = numpy.ones(psi_2grad.shape[-1]) * (self.d * (self.n - self.m) / 2.) * numpy.log(beta)
#        const += .5 * beta * numpy.sum(numpy.dot(self._get_y().T, self._get_y()))
        # gKern0
        gKern0 = -(beta / 2.) * self.d * psi_0grad 
        # gKern1
        gKern1 = beta * numpy.trace(numpy.dot(numpy.dot(y,self.B.T), psi_1grad), axis1=0, axis2=1)
        # gKern2
        gKern2 = (beta / 2.) * numpy.trace(numpy.dot(self.T1, psi_2grad), axis1=0, axis2=1)
        
#        glml_theta-= (beta / 2.) * (self.TrYY#numpy.dot(y.T, y) 
#                                    - self.d * psi_0grad
#                                    - numpy.trace(numpy.dot(psi_2grad.T, self.T1),axis1=2, axis2=1))
        # gKern3
        T2 = beta * self.d * numpy.dot(self.LmInv.T, 
                                       numpy.dot(self.C, 
                                                 self.LmInv))
        gKern3 = .5 * numpy.trace(numpy.dot((self.T1-T2).T, Kgrad), axis1=0, axis2=1)
        import pdb;pdb.set_trace()
        return - (
                  gKern0 
                  + gKern1 
                  + gKern2 
                  + gKern3
                  )

    def _LMLgrad_beta(self, hyperparams):
        beta_inv = 1. / (1.*hyperparams['beta'][0])
        LATildeInfTP = numpy.dot(self.LAtildeInv.T, self.P)
        gBeta = .5 * (self.d * (numpy.trace(self.C) + (self.n - self.m) * beta_inv - self.psi_0)
                      - self.TrYY + self.TrPP 
                      + beta_inv**2 * self.d * numpy.sum(self.LAtildeInv * self.LAtildeInv) 
                      + beta_inv * numpy.sum(LATildeInfTP ** 2))
        return - gBeta # negative because gradient is w.r.t loglikelihood

    def _LMLgrad_means(self, hyperparams):
        return -hyperparams[variational_gplvm_hyperparam_means_id].T

    def _LMLgrad_variances(self, hyperparams):
        return .5 - .5 * hyperparams[variational_gplvm_hyperparam_vars_id].T
    
    def _compute_variational_bound(self, hyperparams):
        logDAtilde = 2 * numpy.sum(numpy.log(numpy.diag(self.LAtilde)))            
        beta = hyperparams['beta'][0]
        bound  = self.d * ( - (self.n - self.m) * numpy.log(beta) + logDAtilde)
        bound -= beta * ( self.TrPP - self.TrYY )
        bound += self.d * beta * ( self.psi_0 - numpy.trace(self.C) )
        bound *= -.5

        bound -= (self.n * self.d / 2.) * numpy.log(2 * (numpy.pi))
        return bound
    
    def _compute_kl_divergence(self, hyperparams):
        mean = hyperparams[variational_gplvm_hyperparam_means_id]
        variance = hyperparams[variational_gplvm_hyperparam_vars_id]
        variational_mean = numpy.sum(mean * mean)
        variational_variance = numpy.sum(variance - numpy.log(variance))
        return -.5 * (variational_mean + variational_variance) + .5 * self.m * self.n
    
    def _update_stats(self, hyperparams):
        self.psi_0 = self._compute_psi_zero(hyperparams)     # scalar
        self.psi_1 = self._compute_psi_one(hyperparams)      # N x M
        self.psi_2 = self._compute_psi_two(hyperparams)      # M x M
        
        self.Kmm = self.covar.K(hyperparams['covar'], hyperparams[variational_gplvm_inducing_variables_id])
        self.Lm = jitChol(self.Kmm)[0].T # lower triangular
        #self.LmInv = scipy.lib.lapack.flapack.dpotri(self.Lm)[0]
        self.LmInv = linalg.inv(self.Lm)
        self.KmmInv = numpy.dot(self.LmInv.T, self.LmInv)
        self.C = numpy.dot(numpy.dot(self.LmInv, 
                                     self.psi_2),
                           self.LmInv.T)                     # M x M
        self.Atilde = (1./hyperparams['beta'][0]) * numpy.eye(self.m) + self.C  # M x M
        self.LAtilde = jitChol(self.Atilde)[0].T
        #self.LAtildeInv = scipy.lib.lapack.flapack.dpotri(self.LAtilde)[0]
        self.LAtildeInv = linalg.inv(self.LAtilde)
        self.P1 = numpy.dot(self.LAtildeInv, self.LmInv)    # M x M
        self.P = numpy.dot(self.P1, numpy.dot(self.psi_1.T, self._get_y()))     # M x D
        self.B = numpy.dot(self.P1.T, self.P)      # M x D
#        import pdb;pdb.set_trace()
        Tb = (1./hyperparams['beta'][0]) * self.d * numpy.dot(self.P1.T,self.P1) 
        Tb += numpy.dot(self.B,self.B.T)     # M x M
        self.T1 = self.d * self.KmmInv - Tb  # M x M
        self.TrPP = numpy.sum(self.P * self.P) # scalar
        
        #import pdb;pdb.set_trace()
        #store hyperparameters for cachine
        self._covar_cache = {}
        self._covar_cache['hyperparams'] = deepcopy(hyperparams)
        self._active_set_indices_changed = False

    def _compute_psi_zero(self, hyperparams):
        return self.covar.psi_0(hyperparams['covar'], 
                                hyperparams[variational_gplvm_hyperparam_means_id], 
                                hyperparams[variational_gplvm_hyperparam_vars_id], 
                                hyperparams[variational_gplvm_inducing_variables_id])

    def _compute_psi_one(self, hyperparams):
        return self.covar.psi_1(hyperparams['covar'], 
                                hyperparams[variational_gplvm_hyperparam_means_id], 
                                hyperparams[variational_gplvm_hyperparam_vars_id], 
                                hyperparams[variational_gplvm_inducing_variables_id])

    def _compute_psi_two(self, hyperparams):
        return self.covar.psi_2(hyperparams['covar'], 
                                hyperparams[variational_gplvm_hyperparam_means_id], 
                                hyperparams[variational_gplvm_hyperparam_vars_id], 
                                hyperparams[variational_gplvm_inducing_variables_id])
    
    def _LMLgrad_x(self, hyperparams):
        
        return super(BayesianGPLVM, self)._LMLgrad_x(hyperparams)
    
    
