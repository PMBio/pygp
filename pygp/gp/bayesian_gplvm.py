'''
Created on 4 Apr 2012

@author: maxz
'''
from pygp.gp.gplvm import GPLVM
import numpy
from pygp.linalg.linalg_matrix import jitChol
from copy import deepcopy
import scipy
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
        self.n = len(y)
        self.d = self.y.shape[1]
        
        #invalidate cache
        self._invalidate_cache()
        self.gplvm_dimensions = gplvm_dimensions
        self.m = n_inducing_variables
        self.jitter = 1E-6
        
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
        
            self._covar_cache['bound'] = bound
            
        bound = self._covar_cache['bound']   
#        #account for prior
#        if priors is not None:
#            plml = self._LML_prior(hyperparams, priors=priors)
#            bound -= numpy.array([p[:, 0].sum() for p in plml.values()]).su
        return - bound
    
    def LMLgrad(self, hyperparams, priors=None, **kw_args):
        """
        gradients w.r.t hyperparams.keys() = ['covar', 'X', 'beta', 'S', 'Xm']:
        """
        LMLgrad = {}
        # {covar, Xm}: 
        LMLgrad[variational_gplvm_inducing_variables_id] = 0
        import pdb;pdb.set_trace()
        glml_theta = self._LMLgrad_wrt(hyperparams['beta'][0], 
                                       self._get_y(), 
                                       psi_0, 
                                       psi_1, 
                                       psi_2, 
                                       numpy.array([self.covar.Kgrad_theta(hyperparams['covar'], hyperparams[variational_gplvm_inducing_variables_id], i) for i in xrange(len(hyperparams['covar']))]))
                                         
        LMLgrad['covar'] = glml_theta
        # beta:
        LMLgrad['beta'] = self._LMLgrad_beta(hyperparams)
        # X:
        LMLgrad[variational_gplvm_hyperparam_means_id] = - hyperparams[variational_gplvm_hyperparam_means_id].T
        # S:
        LMLgrad[variational_gplvm_hyperparam_vars_id] = .5 - .5 * hyperparams[variational_gplvm_hyperparam_vars_id].T
        return LMLgrad
    
    def _LMLgrad_wrt(self, beta, y, psi_0, psi_1, psi_2, K):
        glml_theta = ( (self.d(self.n - self.m) / 2.) * numpy.log(beta) 
                       - (beta / 2.) * (numpy.dot(y.T, y) 
                                        - self.d * psi_0
                                        - numpy.trace(numpy.dot(psi_2, self.T1)))
                       + beta * numpy.trace(numpy.dot(psi_1.T, numpy.dot(y, self.B.T))) 
                       + .5 * numpy.trace( numpy.dot(K, 
                                                     (self.T1 - beta * self.d * numpy.dot(self.LmInv.T, 
                                                                                          numpy.dot(self.C, 
                                                                                                    self.LmInv)))))
                       )
        return glml_theta

    def _LMLgrad_beta(self, hyperparams):
        beta_inv = 1. / hyperparams['beta'][0]
        LATildeInfTP = numpy.dot(self.LAtildeInv.T, self.P)
        gBeta = .5 * (self.d * (numpy.trace(self.C) + (self.n - self.m) * beta_inv - self.psi_0)
                      - self.TrYY + self.TrPP 
                      + beta_inv**2 * self.d * numpy.trace(self.LAtildeInv * self.LAtildeInv) 
                      + beta_inv * numpy.trace(LATildeInfTP ** 2))
        return gBeta
    
    def _compute_variational_bound(self, hyperparams):
        logDAtilde = 2 * numpy.sum(numpy.log(numpy.diag(self.LAtilde)))            
        beta = hyperparams['beta'][0]
        
        bound  = -.5 * ( self.d * ( - (self.n - self.m) * numpy.log(beta) + logDAtilde)
                         - beta * ( self.TrPP - self.TrYY )
                         + self.d * beta * ( self.psi_0 - numpy.trace(self.C) )
                         )

        bound -= self.n * self.d / (2. * numpy.log(2 * (numpy.pi)))
        return bound
    
    def _compute_kl_divergence(self, hyperparams):
        mean = hyperparams[variational_gplvm_hyperparam_means_id]
        variance = hyperparams[variational_gplvm_hyperparam_vars_id]
        variational_mean = numpy.sum(mean * mean)
        variational_variance = numpy.sum(variance - numpy.log(variance))
        return -.5 * (variational_mean + variational_variance) + .5 * self.m * self.n
    
    def _update_stats(self, hyperparams):
        self.Kmm = self.covar.K(hyperparams['covar'], hyperparams[variational_gplvm_inducing_variables_id])
        self.Lm = jitChol(self.Kmm)[0].T # lower triangular
        #self.LmInv = scipy.lib.lapack.flapack.dpotri(self.Lm)[0]
        self.LmInv = linalg.inv(self.Lm)
        self.KmmInf = numpy.dot(self.LmInv.T, self.LmInv)
        self.C = numpy.dot(numpy.dot(self.LmInv, 
                                     self._compute_psi_two(hyperparams)),
                           self.LmInv.T)
        self.Atilde = (1./hyperparams['beta'][0]) * numpy.eye(self.m) + self.C
        self.LAtilde = jitChol(self.Atilde)[0].T
        #self.LAtildeInv = scipy.lib.lapack.flapack.dpotri(self.LAtilde)[0]
        self.LAtildeInv = linalg.inv(self.LAtilde)
        self.P1 = numpy.dot(self.LAtildeInv, self.LmInv)
        self.P = numpy.dot(self.P1, numpy.dot(self._compute_psi_one(hyperparams).T, self._get_y()))
        self.B = numpy.dot(self.P1.T, self.P)
        self.T1 = self.d * (self.KmmInf - (1./hyperparams['beta'][0]) * numpy.dot(self.P1.T,self.P1)) 
        self.TrPP = numpy.trace(self.P * self.P) # sum?
        self.psi_0 = self._compute_psi_zero(hyperparams)
        
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
