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
import copy

#Verbose implements all gradients and evaluations using naive methods and inlcudes debug asserts
VERBOSE=False

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
            Sn = Knoise + SP.tile(S[:,SP.newaxis],[1,self.d])
            #inverse
            Si = 1./Sn 
            #rotate data
            y_rot = SP.dot(U.T,self.y)
            #also store version of data rotated and Si applied
            y_roti = (y_rot*Si)
            self._covar_cache = {'S':S,'U':U,'K':K,'Knoise':Knoise,'Sn':Sn,'Si':Si,'y_rot':y_rot,'y_roti':y_roti}
            self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams)
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
        LML   = 0.5*self.n*self.d * SP.log(2*SP.pi) + lquad + ldet
        if VERBOSE:
            #1. slow and explicit way
            lmls_ = SP.zeros([self.d])
            for i in xrange(self.d):
                _y = self.y[:,i]
                sigma2 = SP.exp(2*hyperparams['lik'])
                _K = KV['K'] + SP.diag(KV['Knoise'][:,i])
                _Ki = SP.linalg.inv(_K)
                lquad_ = 0.5 * SP.dot(_y,SP.dot(_Ki,_y))
                ldet_ = 0.5 * SP.log(SP.linalg.det(_K))
                lmls_[i] = 0.5 * self.n* SP.log(2*SP.pi) + lquad_ + ldet_
            assert SP.absolute(lmls_.sum()-LML)<1E-3, 'outch'
        return LML



    def _LMLgrad_covar(self, hyperparams):
	logtheta = hyperparams['covar']
	try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6

        LMLgrad = SP.zeros(len(logtheta))
        for i in xrange(len(logtheta)):
            #1. derivative of the log det term
            Kd = self.covar.Kgrad_theta(hyperparams['covar'], self._get_x(), i)
            #rotate Kd with U, U.T
            Kd_rot = SP.dot(SP.dot(KV['U'].T,Kd),KV['U'])
            #now loop over the various different noise levels which is efficient at this point
            dldet = 0.5*(Kd_rot.diagonal()[:,SP.newaxis]*KV['Si']).sum()            
            #2. deriative of the quadratic term
            y_roti = KV['y_roti']
            DKy_roti = SP.dot(Kd_rot,KV['y_roti'])
            dlquad  = -0.5*(y_roti*DKy_roti).sum()

            if VERBOSE:
                dldet_ = SP.zeros([self.d])
                dlquad_ = SP.zeros([self.d])
                for d in xrange(self.d):
                    _K = KV['K'] + SP.diag(KV['Knoise'][:,d])
                    _Ki = SP.linalg.inv(_K)
                    dldet_[d] = 0.5*SP.dot(_Ki,Kd).trace()
                    dKq = SP.dot(SP.dot(_Ki,Kd),_Ki)
                    dlquad_[d] = -0.5*SP.dot(SP.dot(self.y[:,d],dKq),self.y[:,d])

                    
                assert SP.absolute(dldet-dldet_.sum())<1E-3, 'outch'
                assert SP.absolute(dlquad-dlquad_.sum())<1E-3, 'outch'

            #set results
            LMLgrad[i] = dldet + dlquad
        RV = {'covar': LMLgrad}
        return RV


    def _LMLgrad_lik(self,hyperparams):
        """derivative of the likelihood parameters"""

	logtheta = hyperparams['covar']
        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6
	
        #loop through all dimensions
        #logdet term:
        Kd = 2*KV['Knoise']
        dldet = 0.5*(Kd*KV['Si']).sum(axis=0)
        #quadratic term
        y_roti = KV['y_roti']
        dlquad = -0.5 * (y_roti * Kd * y_roti).sum(axis=0)
        if VERBOSE:
            dldet_  = SP.zeros([self.d])
            dlquad_ = SP.zeros([self.d])
            for d in xrange(self.d):
                _K = KV['K'] + SP.diag(KV['Knoise'][:,d])
                _Ki = SP.linalg.inv(_K)
                dldet_[d] = 0.5* SP.dot(_Ki,SP.diag(Kd[:,d])).trace()
                dlquad_[d] = -0.5*SP.dot(self.y[:,d],SP.dot(_Ki,SP.dot(SP.diag(Kd[:,d]),SP.dot(_Ki,self.y[:,d]))))

            assert (SP.absolute(dldet-dldet_)<1E-3).all(), 'outch'
            assert (SP.absolute(dlquad-dlquad_)<1E-3).all(), 'outch'


        LMLgrad = dldet + dlquad
        RV = {'lik': LMLgrad}
    
        return RV


    def _LMLgrad_x(self, hyperparams):
        """GPLVM derivative w.r.t. to latent variables
        """
        if not 'x' in hyperparams:
            return {}

        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6


        pass

	dlMl = SP.zeros([self.n,len(self.gplvm_dimensions)])
        #dlMl_det  = SP.zeros([self.n,len(self.gplvm_dimensions)])
        #dlMl_quad = SP.zeros([self.n,len(self.gplvm_dimensions)])

        #U*Si*y
        UYi=SP.dot(KV['U'],KV['y_roti'])
        
        for i in xrange(len(self.gplvm_dimensions)):
            d = self.gplvm_dimensions[i]
            #dKx is general, not knowing that we are computing the diagonal:
            dKx = self.covar.Kgrad_x(hyperparams['covar'], self.x, self.x, d)
            #vector with all diagonals of SP.dot(SP.dot(KV['U'].T,dKxn),KV['U']) for n=1..N
            dKx_rot = 2*KV['U']*SP.dot(dKx,KV['U'])
            #caching for easier construction below
            dKx_U   = SP.dot(dKx,UYi)
            if 0:
                # an attept to vectorize this but I think we should use pyrex and dont make this code completely unreadable.
                #log det
                dKx_rot_tile  = SP.tile(dKx_rot[:,:,SP.newaxis],[1,1,self.d])
                Si_tile       = SP.tile(KV['Si'][SP.newaxis,:,:],[self.n,1,1])
                dlMl_det[:,i] = 0.5*( dKx_rot_tile*Si_tile).sum(axis=2).sum(axis=1)
                #quad
                UYi_tile = SP.tile(UYi[SP.newaxis,:,:],[self.n,1,1])
                dxU_tile = SP.zeros([self.n,self.n,self.d])
                dxU_tile[:,:,:] = SP.tile(dKx[:,:,SP.newaxis],[1,1,self.d])
                dxU_tile[:,:,:] *= SP.tile(UYi[:,SP.newaxis,:],[1,self.n,1])
                #dxU_tile[:,:,:] += SP.tile(dKx_U[:,SP.newaxis,:],[1,self.n,1])
                dlMl_quad[:,i]  = -0.5*dxU_tile.sum(axis=2).sum(axis=1)            
            for n in xrange(self.n):
                dldet  = 0.5* (dKx_rot[n,:][:,SP.newaxis]*KV['Si']).sum()
                #create SP.dot(dKxn,Uyi) using precaclulated dKx_U vectors
                dxU = SP.zeros([self.n, self.d])
                dxU[n,:] = dKx_U[n,:]
                
                dxU[:,:] += SP.outer(dKx[n,:],UYi[n,:])
                #the res ist now UYi[:,d] * dxU[:,i], pointwise multiplication to do this for all d at the same time and some over them:
                dlquad = -0.5*(UYi*dxU).sum()
                dlMl[n,i] = dldet + dlquad
                if VERBOSE:
                    #naive way
                    dKxn = SP.zeros([self.n, self.n])
                    dKxn[n, :]  = dKx[n, :]
                    dKxn[:, n] += dKx[n, :]
                    Kd_rot = SP.dot(SP.dot(KV['U'].T,dKxn),KV['U'])
                    y_roti = KV['y_roti']
                    DKy_roti = SP.dot(Kd_rot,KV['y_roti'])
                    dldet = 0.5*(Kd_rot.diagonal()[:,SP.newaxis]*KV['Si']).sum()            
                    dlquad  = -0.5*(y_roti*DKy_roti).sum()
                    assert SP.absolute(dlMl[n,i]-(dldet+dlquad)).max()<1E-5 , 'outch'
        pass
        RV = {'x':dlMl}
        return RV
        

if __name__ == '__main__':
    from pygp.gp import gplvm
    from pygp.covar import linear,se, noise, combinators

    import pygp.optimize as opt
    import pygp.plot.gpr_plot as gpr_plot
    import pygp.priors.lnpriors as lnpriors
    import pygp.likelihood as lik
    import optimize_test
    import copy
    import logging as LG

    LG.basicConfig(level=LG.INFO)
    
    #1. simulate data from a linear PCA model
    N = 200
    K = 3
    D = 10

    SP.random.seed(1)
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T


    sim_fa_noise = False
    if sim_fa_noise:
        #inerpolate noise levels
        noise_levels = SP.linspace(0.1,1.0,Y.shape[1])
        Ynoise =noise_levels*random.randn(N,D)
        Y+=Ynoise
    else:
        Y+= 0.1*SP.random.randn(N,D)

    #use "standard PCA"
    [Spca,Wpca] = gplvm.PCA(Y,K)

    #reconstruction
    Y_ = SP.dot(Spca,Wpca.T)

    covariance = linear.LinearCFISO(n_dimensions=K)
    hyperparams = {'covar': SP.log([1.2])}
    hyperparams_fa = {'covar': SP.log([1.2])}

        
    #factor analysis noise
    likelihood_fa = lik.GaussLikARD(n_dimensions=D)
    hyperparams_fa['lik'] = SP.log(0.1*SP.ones(Y.shape[1]))
    
    #standard Gaussian noise
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
        
    #initialization of X at arandom
    X0 = SP.random.randn(N,K)
    X0 = Spca
    hyperparams['x'] = X0
    hyperparams_fa['x'] = X0

    #try evaluating marginal likelihood first
    #del(hyperparams['x'])
    #del(hyperparams_fa['x'])


    g_fa = GPLVMARD(covar_func=covariance,likelihood=likelihood_fa,x=X0,y=Y)
    g = gplvm.GPLVM(covar_func=covariance,likelihood=likelihood,x=X0,y=Y)


    if 1:
        lml=g.LML(hyperparams)
        lml_fa = g_fa.LML(hyperparams_fa)
        dg = g.LMLgrad(hyperparams)
        dg_fa = g_fa.LMLgrad(hyperparams_fa)

        

    #hyperparams['covar'] = SP.array([-0.02438411])
    if 0:
        #manual gradcheck
        relchange = 1E-5;
        change = hyperparams['covar'][0]*relchange
        hyperparams_ = copy.deepcopy(hyperparams)
        xp = hyperparams['covar'][0] + change
        pdb.set_trace()
        hyperparams_['covar'][0] = xp
        Lp = g.LML(hyperparams_)
        xm = hyperparams['covar'][0] - change
        hyperparams_['covar'][0] = xm
        Lm = g.LML(hyperparams_)
        diff = (Lp-Lm)/(2.*change)

        anal = g.LMLgrad(hyperparams)
        
    

    Ifilter_fa = {}
    for key in hyperparams_fa:
        Ifilter_fa[key] = SP.ones(hyperparams_fa[key].shape,dtype='bool')
    #Ifilter_fa['lik'][:] = False

    Ifilter = {}
    for key in hyperparams:
        Ifilter[key] = SP.ones(hyperparams[key].shape,dtype='bool')
    #Ifilter['lik'][:] = False


    #[opt_hyperparams,opt_lml] = opt.opt_hyper(g,hyperparams,gradcheck=True,Ifilter=Ifilter)
#    hyperparams['covar'] = opt_hyperparams['covar']
#    hyperparams['x']     = opt_hyperparams['x']
#    hyperparams_fa['covar'] = opt_hyperparams['covar']
#    hyperparams_fa['x']     = opt_hyperparams['x']

    [opt_hyperparams_fa,opt_lml_fa] = opt.opt_hyper(g_fa,hyperparams_fa,gradcheck=True,Ifilter=Ifilter)
    #[opt_hyperparams_fa,opt_lml_fa] = optimize_test.opt_hyper(g_fa,g,hyperparams_fa,hyperparams,Ifilter=Ifilter_fa,Ifilter2=Ifilter,gradcheck=True)
    
    if 0:
    

        lml=g.LML(opt_hyperparams)
        lml_fa = g_fa.LML(hyperparams_fa)
        dg = g.LMLgrad(opt_hyperparams)
        dg_fa = g_fa.LMLgrad(hyperparams_fa)

        
        #[opt_hyperparams,opt_lml] = opt.opt_hyper(g_fa,hyperparams_fa,gradcheck=True,Ifilter=Ifilter_fa)



    

