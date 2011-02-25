"""
Basic Gaussian Process class
============================


"""
import copy
import numpy.linalg as linalg
import scipy as SP

import logging as LG

class GP(object):
    """
    Gaussian Process regression class. Holds all information
    for the GP regression to take place.

    **Parameters:**

    covar_func : :py:class:`pygp.covar`
        The covariance function, which calculates the covariance
        of the outputs

    x : [double]
        training inputs (might be high dimensional,
        depending on which covariance function is chosen)
        Note: x must be of dimension `(-1,1)`

    y : [double]
        training targets

    Detailed descriptions of the fields of this class:
    
    ================================ ============ ===========================================
    Data                             Type/Default Explanation
    ================================ ============ ===========================================
    x                                array([])    inputs
    t                                array([])    targets
    n                                0            size of training data
    mean                             0            mean of the data

    **Settings:**

    **Covariance:**
    covar                            None         Covariance function

    **caching of covariance-stuff:** 
    alpha                            None         cached alpha
    L                                None         chol(K)
    Nlogtheta                        0            total number of hyperparameters
                                                  for set kernel etc.
                                                  which if av. will be used
                                                  for predictions
    ================================ ============ ===========================================
    """
    # Smean : boolean
    # Subtract mean of Data
    # TODO: added d
    __slots__ = ["x","y","n","d","covar", \
                 "_covar_cache"]
    
    def __init__(self, covar_func=None, x=None,y=None):
        '''GP(covar_func,Smean=True,x=None,y=None)
        covar_func: Covariance
        x/y:        training input/targets
        '''       
        if not (x is None):
            self.setData(x,y)
        # Store the constructor parameters
        self.covar   = covar_func
        self._invalidate_cache()
        pass

    
        
       
    def getData(self):
        """ Returns the data, currently set for this GP"""
        return [self.x,self.y]

    
    def setData(self,x,y):
        """
        setData(x,t) with **Parameters:**

        x : inputs: [N x D]

        y : targets/outputs [N x d]
        #note d dimensional data structure only make sense for GPLVM
        """
        if(len(x.shape) <= 1):
            x=x.reshape(-1,1)
        self.x = x
        #squeeeze targets; this should only be a vector
        self.y = y.squeeze()
        #assert shapes
        if len(self.y.shape)==1:
            self.y = self.y[:,SP.newaxis]
        assert self.x.shape[0]==self.y.shape[0], 'input/target shape missmatch'
        self.n = len(self.x)
        #for GPLVM models:
        self.d = self.y.shape[1]
        
        #invalidate cache
        self._invalidate_cache()
        pass


    

    def lMl(self,hyperparams,priors=None,**kw_args):
        """
        Calculate the log Marginal likelihood
        for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ... }
            The hyperparameters for the log marginal likelihood.

        priors : [:py:class:`pygp.priors`]
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

        priors : [:py:class:`pygp.priors`]
            The hyperparameters which shall be optimized and derived

        """
        # Ideriv : 
        #      indicator which derivativse to calculate (default: all)

        RV=self._dlMl_covar(hyperparams)
        
        #prior
        if priors is not None:
            plml = self._lml_prior(hyperparams,priors=priors,**kw_args)
            for key in RV.keys():
                RV[key]-=plml[key][:,1]                       
        return RV

    def getCovariances(self,hyperparams,x=None,y=None):
        """
        Return the Cholesky decompositions L and alpha::

            K 
            L     = chol(K)
            alpha = solve(L,t)
            return [covar_struct] = getCovariances(hyperparam)
        """
        if(x is None or y is None):
            x=self.x
            y=self.y
        
        if self._is_cached(hyperparams):
            pass
        else:
            #update cache
            K = self.covar.K(hyperparams['covar'],x)
            L = linalg.cholesky(K)               
            alpha = _solve_chol(L.T,y)
            self._covar_cache = {'K': K,'L':L,'alpha':alpha,'hyperparams':copy.deepcopy(hyperparams)}
        return self._covar_cache 
       
        
    def predict(self,hyperparams,xstar,output=0,var=True,interval_indices=None):
        '''
        Predict mean and variance for given **Parameters:**

        hyperparams : {}
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        var      : boolean
            return predicted variance
        
        interval_indices : [ int || bool ]
            Either scipy array-like of boolean indicators, 
            or scipy array-like of integer indices, denoting 
            which x indices to predict from data. 
        
        output   : output dimension for prediction (0)
        '''
        if(interval_indices is None):
            x=self.x
            y=self.y
        else:
            x=self.x[interval_indices]
            y=self.y[interval_indices]

        KV = self.getCovariances(hyperparams,x,y)
        #cross covariance:
        Kstar       = self.covar.K(hyperparams['covar'],x,xstar)
        mu = SP.dot(Kstar.transpose(),KV['alpha'][:,output])
        if(var):            
            Kss_diag         = self.covar.Kdiag(hyperparams['covar'],xstar)
            v    = linalg.solve(KV['L'],Kstar)
            S2   = Kss_diag - sum(v*v,0).transpose()
            S2   = abs(S2)
            return [mu,S2]
        else:
            return mu


    ########PRIVATE FUNCTIONS########

    #log marginal likelihood contributions from covaraince hyperparameters:

    def _lMl_covar(self,hyperparams):
        
        try:   
            KV = self.getCovariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6

        #Change: no supports multi dimensional stuff for GPLVM
        lMl = 0.5*(KV['alpha']*self.y).sum() + self.d*(sum(SP.log(KV['L'].diagonal())) + 0.5*self.n*SP.log(2*SP.pi))
        return lMl


    def _dlMl_covar(self,hyperparams):
        #currently only support derivatives of covar params
        logtheta = hyperparams['covar']
        try:   
            KV = self.getCovariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'covar':SP.zeros(len(logtheta))}
        logtheta = hyperparams['covar']
        n = self.n
        L = KV['L']

        alpha = KV['alpha']
        W  =  self.d*linalg.solve(L.transpose(),linalg.solve(L,SP.eye(n))) - SP.dot(alpha,alpha.transpose())
        self._covar_cache['W'] = W
        

        dlMl = SP.zeros(len(logtheta))
        for i in xrange(len(logtheta)):
            Kd = self.covar.Kd(hyperparams['covar'],self.x,i)
            dlMl[i] = 0.5*(W*Kd).sum()
        RV = {'covar': dlMl}
        return RV

                   
    def _invalidate_cache(self):
        """reset cache structure"""
        self._covar_cache = None
        pass

    def _lml_prior(self,hyperparams,priors={}):
        """calculate the prior contribution to the log marginal likelihood"""
        if priors is None:
            priors = {}
        RV = {}
        for key,value in hyperparams.iteritems():
            pvalues = SP.zeros([len(value),2])
            if key in priors:
                plist = priors[key]
                theta = copy.deepcopy(hyperparams[key])
                Iexp = self.covar.get_Iexp(theta)
                theta[Iexp] = SP.exp(theta[Iexp])
                for i in xrange(len(theta)):
                    pvalues[i,:] = plist[i][0](theta[i],plist[i][1])
                #chain rule
                pvalues[Iexp,1]*=theta[Iexp]
            RV[key] = pvalues
        return RV

    def _is_cached(self,hyperparams):
        """check whether model parameters are cached"""
        if self._covar_cache is None:
            return False
        else:
            #compare
            for key in hyperparams.keys():
                if not (self._covar_cache['hyperparams'][key]==hyperparams[key]).all():
                    return False
            #otherwise they are cached:
            return True


    
def _solve_chol(A,B):
    """
    Solve cholesky decomposition::
    
        return A\(A'\B)

    """
    X = linalg.solve(A,linalg.solve(A.transpose(),B))
    return X

