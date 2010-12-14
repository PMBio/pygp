"""
Module for Gaussian process Regression
--------------------------------------

This module is a lot modelled after Karl Rasmussen Gaussian process
package for Matlab (http://www.gaussianprocess.org/gpml/).

Methods and Classes

.. func *optHyper*:
   use a gradient based optimiser to optimise
   GP hyperparameters subject to prior parameters


.. class **GP**: basic class for GP regression:
   * claculation of log marginal likelihood
   * prediction
   * data rescaling
   * transformation into log space

"""


# import python / numpy:
#from pylab import *
import scipy as SP
import numpy.linalg as linalg
import scipy.optimize.optimize as OPT
import logging as LG
import copy
import pdb

def optHyper(gpr,hyperparams,Ifilter=None,maxiter=100,gradcheck=False,**kw_args):
    """
    Optimize hyperparemters of gp gpr starting from gpr

    **Parameters:**
    
    gpr : :py:class:`gpr.GP`
        GP regression class

    hyperparams : {'covar':logtheta, ...}
        Dictionary filled with starting hyperparameters
        for optimization. logtheta are the CF hyperparameters.

    Ifilter : [boolean]
        Index vector, indicating which hyperparameters shall
        be optimized. For instance::

            logtheta = [1,2,3]
            Ifilter = [0,1,0]

        means that only the second entry (which equals 2 in
        this example) of logtheta will be optimized
        and the others remain untouched.

    priors : [:py:class:`lnpriors`]
        non-default prior, otherwise assume
        first index amplitude, last noise, rest:lengthscales
    """
  
    #1. convert the dictionaries to parameter lists
    X0 = gpr._param_dict_to_list(hyperparams)
    if Ifilter is not None:
        Ifilter_x = SP.array(gpr._param_dict_to_list(Ifilter),dtype='bool')
    else:
        Ifilter_x = SP.ones(len(X0),dtype='bool')

    def f(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.lMl(x_,**kw_args)
        LG.debug("L("+str(x_)+")=="+str(rv))
        if SP.isnan(rv):
            return 1E6
        return rv
    
    def df(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.dlMl(x_,**kw_args)
        #convert to list
        rv = gpr._param_dict_to_list(rv)
        LG.debug("dL("+str(x_)+")=="+str(rv))
        if SP.isnan(rv).any():
            In = SP.isnan(rv)
            rv[In] = 1E6
        return rv[Ifilter_x]
        
    #2. set stating point of optimization, truncate the non-used dimensions
    x  = X0.copy()[Ifilter_x]
        
    LG.info("startparameters for opt:"+str(x))
    LG.info("start optimization")

    opt_RV=OPT.fmin_bfgs(f, x, fprime=df, args=(), gtol=1.0000000000000001e-04, norm=SP.inf, epsilon=1.4901161193847656e-08, maxiter=maxiter, full_output=1, disp=(0), retall=0)

    #get optimized parameters out
    opt_x = X0.copy()
    opt_x[Ifilter_x] = opt_RV[0]
    opt_hyperparams = gpr._param_list_to_dict(opt_x)
    #get the log marginal likelihood at the optimum:
    opt_lml = opt_RV[1]

    if gradcheck:
        LG.info("check_grad:" + str(OPT.check_grad(f,df,opt_x[Ifilter_x])))
        raw_input()

    LG.info("old parameters:")
    LG.info(str(hyperparams))
    LG.info("optimized parameters:")
    LG.info(str(opt_hyperparams))
    LG.info("grad:"+str(df(opt_x)))
    return [opt_hyperparams,opt_lml]
    
def _solve_chol(A,B):
    """
    Solve cholesky decomposition::
    
        return A\(A'\B)

    """
    X = linalg.solve(A,linalg.solve(A.transpose(),B))
    return X

class GP(object):
    """
    Gaussian Process regression class. Holds all information
    for the GP regression to take place.

    **Parameters:**

    covar_func : :py:class:`covar`
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
    __slots__ = ["x","y","n","covar", \
                 "_covar_cache","_param_struct"]
    
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
        # create a prototype of the parameter dictionary
        # additional fields will follow
        self._param_struct = {'covar': self.covar.get_number_of_parameters()}
        pass

    
        
       
    def getData(self):
        """ Returns the data, currently set for this GP"""
        return [self.x,self.y]

    
    def setData(self,x,y):
        """
        setData(x,t) with **Parameters:**

        x : inputs: [N x D]

        y : targets/outputs [ N x 1]
        """

        self.x = x
        #squeeeze targets; this should only be a vector
        self.y = y.squeeze()
        #assert shapes 
        assert len(self.y.shape)==1, 'target shape eror'
        assert self.x.shape[0]==self.y.shape[0], 'input/target shape missmatch'
        self.n = len(self.x)
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
        if not isinstance(hyperparams,dict):
            hyperparams = self._param_list_to_dict(hyperparams)
                    
        try:   
            KV = self.getCovariances(hyperparams)
        except linalg.LinAlgError,e:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6

        #calc:
        lMl = 0.5*SP.dot(KV['alpha'],self.y) + sum(SP.log(KV['L'].diagonal())) + 0.5*self.n*SP.log(2*SP.pi)

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

        if not isinstance(hyperparams,dict):
            hyperparams = self._param_list_to_dict(hyperparams)

        RV = {}
        #currently only support derivatives of covar params
        logtheta = hyperparams['covar']
        try:   
            KV = self.getCovariances(hyperparams)
        except linalg.LinAlgError,e:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'covar':SP.zeros(len(logtheta))}
        logtheta = hyperparams['covar']
        n = self.n
        L = KV['L']
        alpha = KV['alpha'][:,SP.newaxis]
        W  = linalg.solve(L.transpose(),linalg.solve(L,SP.eye(n))) - SP.dot(alpha,alpha.transpose())

        dlMl = SP.zeros(len(logtheta))
        for i in xrange(len(logtheta)):
            Kd = self.covar.Kd(hyperparams['covar'],self.x,i)
            dlMl[i] = 0.5*(W*Kd).sum()
        RV = {'covar': dlMl}
        
        #prior
        if priors is not None:
            plml = self._lml_prior(hyperparams,priors=priors,**kw_args)
            for key in RV.keys():
                RV[key]-=plml[key][:,1]                       
        return RV

    def getCovariances(self,hyperparams):
        """
        Return the Cholesky decompositions L and alpha::

            K 
            L     = chol(K)
            alpha = solve(L,t)
            return [covar_struct] = getCovariances(hyperpam)
        """
        if self._is_cached(hyperparams):
            pass
        else:
            #update cache
            K = self.covar.K(hyperparams['covar'],self.x)
            L = linalg.cholesky(K)               
            alpha = _solve_chol(L.transpose(),self.y)
            self._covar_cache = {'K': K,'L':L,'alpha':alpha,'hyperparams':copy.deepcopy(hyperparams)}
        return self._covar_cache 
       
        
    def predict(self,hyperparams,xstar,var=True):
        '''
        Predict mean and variance for given **Parameters:**

        hyperparams : {}
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        var      : boolean
            return predicted variance
        '''
        KV = self.getCovariances(hyperparams)
        #cross covariance:
        Kstar       = self.covar.K(hyperparams['covar'],self.x,xstar)
        mu = SP.dot(Kstar.transpose(),KV['alpha'])
        if(var):            
            Kss_diag         = self.covar.Kdiag(hyperparams['covar'],xstar)
            v    = linalg.solve(KV['L'],Kstar)
            S2   = Kss_diag - sum(v*v,0).transpose()
            S2   = abs(S2)
            return [mu,S2]
        else:
            return mu


    ########PRIVATE FUNCTIONS########

    def _param_dict_to_list(self,dict):
        """convert from param dictionary to list"""
        RV = SP.concatenate([val for val in dict.values()])
        return RV
        pass

    def _param_list_to_dict(self,list):
        """convert from param dictionary to list"""
        RV = []
        i0= 0
        for key in self._param_struct.keys():
            np = self._param_struct[key]
            i1 = i0+np
            RV.append((key,list[i0:i1]))
            i0 = i1
        return dict(RV)

    def _get_number_of_parameters(self):
        """calculate the number of parameters of the gp object"""
        #currently: only covariance function has parameters
        return SP.array([len(val) for val in self._param_dict.values()]).sum()

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
