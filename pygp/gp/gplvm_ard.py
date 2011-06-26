"""
ARD gplvm with one covaraince structure per dimension (at least implicitly)
"""
import sys
sys.path.append('./../..')
from pygp.gp import GP
import pygp.gp.gplvm as GPLVM
import pdb
from pygp.optimize.optimize_base import opt_hyper
import scipy as SP
import scipy.linalg as linalg


VERBOSE=True

def PCA(Y, components):
    """run PCA, retrieving the first (components) principle components
    return [s0,w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices=0);
    [s0, w0] = [sv[0][:, 0:components], SP.dot(SP.diag(sv[1]), sv[2]).T[:, 0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0, w0]

class GPLVMARD(GPLVM.GPLVM):
    """
    derived class form GP offering GPLVM specific functionality
    """

    def __init__(self, *args, **kw_args):
        """gplvm_dimensions: dimensions to learn using gplvm, default -1; i.e. all"""
        super(GPLVM.GPLVM, self).__init__(*args,**kw_args)


    def get_covariances(self,hyperparams):
        if not self._is_cached(hyperparams) or self._active_set_indices_changed:
            #update covariance structure
            K = self.covar.K(hyperparams['covar'],self.x)
            #calc eigenvalue decomposition
            [S,U] = SP.linalg.eigh(K)
            #noise diagonal
            #depending on noise model this may be either a vector or a matrix 
            Knoise = self.likelihood.Kdiag(hyperparams['lik'],self.x)
            #noise version of S
            Sn = Knoise + SP.tile(S[:,SP.newaxis],[1,10])
            #inverse
            Si = 1./Sn
            #rotate data
            y_rot = SP.dot(U.T,self.y)
            #also store version of data rotated and Si applied
            y_roti = (y_rot*Si)
            self._covar_cache = {'hyperparams':hyperparams,'S':S,'U':U,'K':K,'Knoise':Knoise,'Sn':Sn,'Si':Si,'y_rot':y_rot,'y_roti':y_roti}
            pass
        #return update covar cache
        return self._covar_cache


    ####PRIVATE####

    def _LML_covar(self, hyperparams):
        """

	log marginal likelihood contributions from covariance hyperparameters

	"""
        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6

        #all in one go
        #negative log marginal likelihood, see derivations
        lquad = 0.5* (KV['y_rot']*KV['Si']*KV['y_rot']).sum()
        ldet  = 0.5*-SP.log(KV['Si'][:,:]).sum()
        LML   = 0.5*self.d * SP.log(2*SP.pi) + lquad + ldet
        if VERBOSE:
            #1. slow and explicit way
            lmls_ = SP.zeros([self.d])
            for i in xrange(self.d):
                _y = self.y[:,i]
                sigma2 = SP.exp(2*hyperparams['lik'])
                _K = KV['K'] + sigma2[i] * SP.eye(self.n)
                _Ki = SP.linalg.inv(_K)
                lquad_ = 0.5 * SP.dot(_y,SP.dot(_Ki,_y))
                ldet_ = 0.5 * SP.log(SP.linalg.det(_K))
                lmls_[i] = 0.5 * SP.log(2*SP.pi) + lquad_ + ldet_
            assert SP.absolute(lmls_.sum()-LML)<1E-5, 'outch'
        return LML



    def _LMLgrad_covar(self, hyperparams):
        #1. get inggredients for computations
        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c']))}
        pdb.set_trace()
        pass


    def _LMLgrad_lik(self,hyperparams):
        """derivative of the likelihood parameters"""
        logtheta = hyperparams['lik']
        #note: we assume hard codede that this is called AFTER LMLgrad_covar has been called
        KV = self._covar_cache
        pdb.set_trace()

        RV = {'lik': LMLgrad}
        return RV


    def _LMLgrad_x(self, hyperparams):
        """GPLVM derivative w.r.t. to latent variables
        """
        if not 'x' in hyperparams:
            return {}

        pdb.set_trace()
        pass

	dlMl = SP.zeros([self.n,len(self.gplvm_dimensions)])
        W = self._covar_cache['W']

        for i in xrange(len(self.gplvm_dimensions)):
            d = self.gplvm_dimensions[i]
            #dKx is general, not knowing that we are computing the diagonal:
            dKx = self.covar.Kgrad_x(hyperparams['covar'], self.x, self.x, d)
            dKx_diag = self.covar.Kgrad_xdiag(hyperparams['covar'], self.x, d)
            #set diagonal
            dKx.flat[::(dKx.shape[1] + 1)] = dKx_diag
            #precalc elementwise product of W and K
            WK = W * dKx
            if 0:
                for n in xrange(self.n):
                    dKxn = SP.zeros([self.n, self.n])
                    dKxn[n, :] = dKx[n, :]
                    dKxn[:, n] = dKx[n, :]
                    dlMl[n, i] = 0.5 * SP.dot(W, dKxn).trace()
                    pass
            if 1:
                #fast calculation
                #we need twice the sum WK because of the matrix structure above, WK.diagonal() accounts for the double counting
                dlMl[:, i] = 0.5 * (2 * WK.sum(axis=1) - WK.diagonal())
            pass
        RV = {'x':dlMl}
        return RV
        

if __name__ == '__main__':
    from pygp.covar import linear, noise, fixed, combinators
    import logging as LG
    LG.basicConfig(level=LG.DEBUG)
    SP.random.seed(1)
    #1. simulate data
    N = 100
    K = 3
    D = 10

    
    S = SP.random.randn(N, K)
    W = SP.random.randn(D, K)
    
    Y = SP.dot(W, S.T).T
    Y += 0.5 * SP.random.randn(N, D)
  
    [Spca, Wpca] = PCA(Y, K)
    
    #reconstruction
    Y_ = SP.dot(Spca, Wpca.T)
    
    #construct GPLVM model
    linear_cf = linear.LinearCFISO(n_dimensions=K)
    noise_cf = noise.NoiseCFISO()
    mu_cf = fixed.FixedCF(SP.ones([N,N]))
    covariance = combinators.SumCF((mu_cf, linear_cf, noise_cf))
    # covariance = combinators.SumCF((linear_cf, noise_cf))


    #no inputs here (later SNPs)
    X = Spca.copy()
    #X = SP.random.randn(N,K)
    gplvm = GPLVM(covar_func=covariance, x=X, y=Y)
   
    gpr = GP(covar_func=covariance, x=X, y=Y[:, 0])
    
    #construct hyperparams
    covar = SP.log([0.1, 1.0, 0.1])

    #X are hyperparameters, i.e. we optimize over them also

    #1. this is jointly with the latent X
    X_ = X.copy()
    hyperparams = {'covar': covar, 'x': X_}
    

    #for testing just covar params alone:
    #hyperparams = {'covar': covar}
    
    #evaluate log marginal likelihood
    lml = gplvm.LML(hyperparams=hyperparams)
    [opt_model_params, opt_lml] = opt_hyper(gplvm, hyperparams, gradcheck=False)
    Xo = opt_model_params['x']
    

    for k in xrange(K):
        print SP.corrcoef(Spca[:, k], S[:, k])
    print "=================="
    for k in xrange(K):
        print SP.corrcoef(Xo[:, k], S[:, k])
