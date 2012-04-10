'''
Created on 4 Apr 2012

@author: maxz
'''
from pygp.gp.gplvm import GPLVM
import numpy
from pygp.linalg.linalg_matrix import jitChol
from copy import deepcopy
import scipy

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
        
#        #account for prior
#        if priors is not None:
#            plml = self._LML_prior(hyperparams, priors=priors)
#            bound -= numpy.array([p[:, 0].sum() for p in plml.values()]).sum()

        return - bound
    
    def _compute_variational_bound(self, hyperparams):
        logDAtilde = 2 * numpy.sum(numpy.log(numpy.diag(self.LAtilde)))            
        beta = hyperparams['beta'][0]
        
        bound  = -.5 * ( ( self.d * ( - (self.n - self.m) * numpy.log(beta) + logDAtilde) )
                         - beta * ( numpy.trace(numpy.dot(self.P, self.P.T) ) - self.TrYY )
                         - self.d * beta * (self._compute_psi_zero(hyperparams) - numpy.trace(self.C) )
                         )

        bound -= self.n * self.d / 2. * numpy.log(2 * (numpy.pi))
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
        self.LmInv = scipy.lib.lapack.flapack.dpotri(self.Lm)[0]
        self.C = numpy.dot(numpy.dot(self.LmInv, 
                                     self._compute_psi_two(hyperparams)),
                           self.LmInv.T)
        self.Atilde = self.jitter * numpy.eye(self.m) + self.C
        self.LAtilde = jitChol(self.Atilde)[0].T
        self.LAtildeInv = scipy.lib.lapack.flapack.dpotri(self.LAtilde)[0]
        self.P1 = numpy.dot(self.LAtildeInv, self.LmInv)
        self.P = numpy.dot(self.P1, numpy.dot(self._compute_psi_one(hyperparams).T, self._get_y()))
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
        pass
