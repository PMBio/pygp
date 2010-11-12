"""
Module for Gaussian Process Regression 
--------------------------------------

This module is a lot modelled after Karl Rasmussen Gaussian process
package for Matlab (http://TOOD).

Methods and Classes

func *optHyper*:
    use a gradient based optimiser to optimise
    GP hyperparameters subject to prior parameters

class **GroupGP**:
    group multiple GP objects for joint optimisation of hyperparameters

class **GP**: basic class for GP regression:
    * claculation of log marginal likelihood
    * prediction
    * data rescaling
    * transformation into log space

"""


# import python / numpy:
from pylab import *
from numpy import * 
import scipy.optimize.optimize as OPT
import logging as LG
#from mlib.stats.lnpriors import *
from lnpriors import *
import pdb


def _solve_chol(A,B):
    """
    Solve cholesky decomposition::
    
        return A\(A'\B)

    """
    X = linalg.solve(A,linalg.solve(A.transpose(),B))
    return X



def _sampleHyper(gpr,modelparameters,Ifilter=None,priors=None,Nsamples=100,eps=1E-2,Nleap=10):
    """
    sample from the posterior distribution of
    GP hyperparmeters (Hyrbid Monte Carlo)
    """
    def fE(x):
        _logtheta[Ifilter] = x
        rv = gpr.lMl(logtheta=_logtheta,lml=True,dlml=False,priors=priors)
        return rv
    def fdE(x):
        _logtheta[Ifilter] = x
        rv = gpr.lMl(logtheta=_logtheta,lml=False,dlml=True,priors=priors)
        return rv[Ifilter]

    #make sure there is a prior
    if priors is None:
        priors = defaultPriors(gpr,logtheta)
    if Ifilter is None:
        Ifilter = ones_like(logtheta)
    #convert I_filter to bools
    Ifilter = Ifilter==1
    _logtheta = logtheta.copy()
    #initialize HMC
    x = logtheta[Ifilter]
    
    g = fdE(x)
    E = fE(x)

    Rtau = xrange(Nleap)                    #leapfrog steps
    #samples
    X    = zeros([Nsamples,logtheta.shape[0]])
    #initialize with logtheta due to filtering
    X[:,:] = logtheta
    naccept = 0
    try:
        for ns in xrange(Nsamples):
            p = random.standard_normal(x.shape)
            H = 0.5*dot(p,p) + E

            xnew = x; gnew = g
            #leapfrogs
            for tau in Rtau:
                p-= 0.5*eps*gnew
                xnew+=eps*p
                gnew = fdE(xnew)
                p-=0.5*eps*gnew
            Enew = fE(xnew)
            Hnew = 0.5*dot(p,p) + Enew
            dH   = Hnew-H
            if (dH<0):
                accept = 1
            elif rand()<exp(-dH):
                accept = 1
                print "AA:"+str(dH)
                pass
            else:
                accept = 0

            if(accept):
                LG.debug("accept: %d" % (ns)+str(exp(xnew))+ "E()="+str(Enew))
                g=gnew; x=xnew; E=Enew
                naccept+=1
            else:
                LG.debug("reject: %d" % (ns)+str(exp(xnew))+ "E()="+str(Enew))
            #store sample
            X[ns,Ifilter] = xnew
    except KeyboardInterrupt:
        sys.stderr.write('Keyboard interrupt')
        sys.stderr.write('returning what we have so far')
    LG.info("samples %d samples, accept/reject = %.2f" % (Nsamples,double(naccept)/Nsamples))
    return X

    

def optHyper(gpr,modelparameters,Ifilter=None,priors=None,maxiter=100,gradcheck=False,**kw_args):
    """
    Optimize hyperparemters of gp gpr starting from gpr
    optHyper(gpr,logtheta,filter=None,prior=None)

    **Parameters:**
    
    gpr : :py:class:`gpr.GP`
        GP regression class

    modelparameters : [double]
        starting hyperparameters for optimization

    Ifilter : [boolean]
        Index vector, indicating which hyperparameters shall
        be optimized. For instance::

            logtheta = [1,2,3]
            Ifilter = [0,1,0]

        means that only the second entry (which equals 2 in this example) of
        logtheta will be optimized and the others remain untouched.

    prior : [:py:class:`lnpriors`]
        non-default prior, otherwise assume
        first index amplitude, last noise, rest:lengthscales
    """
    logtheta = modelparameters['covar']

    modelparameters_filtered = modelparameters.copy()
    modelparameters_filtered['covar'] = modelparameters['covar'][Ifilter]

    curr_index = 0
    curr_indices = []
    for key in modelparameters_filtered.keys():
        curr_element = modelparameters_filtered[key]
        curr_indices.append([key,[arange(curr_index,len(curr_element))]])
        curr_index+=len(curr_element)

    modelparameter_indices = dict(curr_indices)

    if priors is None:        # use a very crude default prior if we don't get anything else:
        priors = defaultPriors(gpr,logtheta)

    def fixlogtheta(logtheta,limit=1E3):
        """make a valid logtheta which is non-infinite and non-0"""
        rv      = logtheta.copy()
        I_upper = logtheta>limit
        I_lower = logtheta<-limit
        rv[I_upper] = +limit
        rv[I_lower] = -limit
        return rv

    def checklogtheta(logtheta,limit=1E3):
        """make a valid logtheta which is non-infinite and non-0"""
        I_upper = logtheta>limit
        I_lower = logtheta<-limit
        return not (I_upper.any() or I_lower.any())
        
    #TODO: mean-function
    def f(x):
        #logtheta_ = fixlogtheta(logtheta)
#        pdb.set_trace()
        logtheta_ = logtheta
        logtheta_[Ifilter] = x
        if not checklogtheta(logtheta):
            print logtheta
            #make optimzier/sampler search somewhere else
            return 1E6

        for key in modelparameter_indices:
            modelparameters_filtered[key] = x[modelparameter_indices[key]]

        rv =  gpr.lMl(modelparameters_filtered,lml=True,dlml=False,priors=priors,**kw_args)
        LG.debug("L("+str(logtheta_)+")=="+str(rv))
        if isnan(rv):
            return 1E6
        # logtheta zu dict \TODO
        return rv
    
    def df(x):
        
        #logtheta_ = fixlogtheta(logtheta)
#        pdb.set_trace()
        logtheta_ = logtheta
        logtheta_[Ifilter] = x
        if not checklogtheta(logtheta):
            #make optimzier/sampler search somewhere else
            print logtheta
            return zeros_like(logtheta_)

        for key in modelparameter_indices:
            modelparameters_filtered[key] = x[modelparameter_indices[key]]

        rv =  gpr.lMl(modelparameters_filtered,lml=False,dlml=True,priors=priors,**kw_args)
        LG.debug("dL("+str(logtheta_)+")=="+str(rv))
        #mask out filtered dimensions
#        if not Ifilter is None:
#            rv = rv*Ifilter
        if isnan(rv).any():
            In = isnan(rv)
            rv[In] = 1E6
        return rv[Ifilter]

    plotit = False
    if(plotit):
        X = arange(0.001,0.05,0.001)
        Y = zeros(size(X))
        dY = zeros(size(X))
        k=2
        theta = logtheta
        for i in range(len(X)):
            theta[k] = log(X[i])
            Y[i] = f(theta)
            dY[i] = df(theta)[k]
        plot(X,Y)
        hold(True);
        plot(X,dY)
        show()

    
    if Ifilter is None:
        Ifilter = ones(logtheta.shape[0],dtype='bool')
    else:
        Ifilter = array(Ifilter,dtype='bool')

    #start-parameters
    x0 = concatenate(modelparameters_filtered.values())
    
    LG.info("startparameters for opt:"+str(x0))
    if gradcheck:
        LG.info("check_grad:" + str(OPT.check_grad(f,df,x0)))
        raw_input()
    LG.info("start optimization")

    opt_result=OPT.fmin_bfgs(f, x0, fprime=df, args=(), gtol=1.0000000000000001e-04, norm=inf, epsilon=1.4901161193847656e-08, maxiter=maxiter, full_output=1, disp=(0), retall=0)
    
    opt_params = modelparameters
    opt_params['covar'][Ifilter] = opt_result[0][modelparameter_indices['covar']]

    for key in opt_params:
        if key == 'covar':
            continue
        opt_params[key] = opt_result[0][modelparameter_indices[key]]
        
    rv = opt_params

    LG.info("old parameters:")
    LG.info(str(exp(logtheta)))
    LG.info("optimized parameters:")
    LG.info(str(exp(rv['covar'])))
    LG.info("grad:"+str(df(rv['covar'])))
    return rv

def defaultPriors(gpr,logtheta):
    """create some crude default priros based on a logtheta and gpr(we might query the covariance function)"""
    priors = []
    for i in range(size(logtheta)):
        #priors.append([lngammapdf,[10,0.1]])
        priors.append([lnzeropdf,[10,0.1]])
    #priors[0][0] = lngammapdf
    priors[0][1] = [2,10]
    #priors[-1][0] = lngammapdf
    priors[-1][1] = [1,0.4]
    return priors
    

class GroupGP(object):
    __slots__ = ["N","GPs"]

    """
    Class to bundle one or more GPs for joint
    optimization of hyperparameters.

    **Parameters:**

    GPs : [:py:func:`GP.lMl`]
        Array, holding al GP classes to be optimized together
    """

    def __init__(self,GPs=None):
        if GPs is None:
            print "you need to specify a list of Gaussian Processes to bundle"
            return None
        self.N = len(GPs)
        self.GPs = GPs
        

    def lMl(self,modelparameters,**lml_kwargs):
        """
        Returns the log Marginal likelyhood for the given logtheta
        and the lMl_kwargs:

        logtheta : [double]
            Array of hyperparameters, which define the covariance function

        lMl_kwargs : lml, dlml, clml, sdlml, priors, Ifilter
            See :py:class:`gpr.GP.lMl`
    
        """
        #lMl(logtehtas,lml=Ture,dlml=true,clml=True,cdlml=True,priors=None)
        #just call them all and add up:
        R = []
        #calculate them for all N
        R = 0
        for n in range(self.N):
            L = self.GPs[n].lMl(modelparameters,**lml_kwargs)
            R = R+ L
        return R

    def setData(self,x,t):
        """
        set inputs x and targets t with **Parameters:**

        x : [double]
            trainging input

        t : [double]
            training targets
            
        rescale_dim : int
            dimensions to be rescaled (default all real)

        process : boolean
            subtract mean and rescale inputs

        """
        for n in range(self.N):
            xn = x[n]
            tn = t[n]
            self.GPs[n].setData(xn,tn)
            
        


class GP(object):
    """
    Gaussian Process regression class. Holds all information
    for the GP regression to take place.

    **Parameters:**

    covar_func : :py:class:`covar`
        The covariance function, which calculates the covariance
        of the outputs

    Smean : boolean
        Subtract mean of Data

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
    Smean                            True         subtract mean from ther data
    rescale                          True         rescale the dataset to standard dimensions

    **Covariance:**
    covar                            None         Covariance function

    **caching of covariance-stuff:** 
    alpha                            None         cached alpha
    L                                None         chol(K)
    Nlogtheta                        0            total number of hyperparameters
                                                  for set kernel etc.
    logtheta                         NAN          logtheta from cached problem.
    logtheta_samples                 None         samples from the hyperparameters
                                                  which if av. will be used
                                                  for predictions
    IlogthetaK                                    index of logtheta for kernel parameters
    ================================ ============ ===========================================
    """

    __slots__ = ["x","t","n","mean","Smean","rescale","covar", \
                 "cached_alpha","cached_L","cached_lMl","cached_dlMl","modelparameters","Nlogtheta","priors","nrX","minX","scaleX","logtheta_samples","IlogthetaK"]

    
    def __init__(self, covar=None, Smean=True,rescale=True,
                 x=None,y=None,modelparameters={'covar':[]},rescale_dim=None):
        '''GP(covar_func,Smean=True,x=None,y=None)
        covar_func: Covariance
        Smean:      subtract mean of Data
        x/y:        training input/targets
        '''
        
        self.Smean=Smean
        self.rescale=rescale
        if not (x is None):
            self.setData(x,y,rescale_dim=rescale_dim)
        # Store the constructor parameters
        self.covar   = covar
        self.modelparameters = modelparameters
        #default object: IlogthetaK is just all of logtheta
        self.IlogthetaK = ones([self.covar.get_number_of_parameters()],dtype='bool')
        if covar is not None:
            self.Nlogtheta  = self.covar.get_number_of_parameters()
        self.logtheta_samples = None
        pass

    def getData(self):
        """ Returns the data, currently set for this GP"""
        return [self.x,self.t]

    def invalidate_cache(self):
        self.cached_lMl = None
        self.cached_dlMl = None
        self.cached_L = None
        pass
    
    def setData(self,x,t,process=True, rescale_dim=None):
        """
        setData(x,t) with **Parameters:**

        x : inputs

        t : targets

        rescale_dim: dimensions to be rescaled (default all real)

        process: subtract mean and rescale input
        """
        if rescale_dim is None:
            rescale_dim = arange(x.shape[1])
        self.x = x
        #squeeeze targets; this should only be a vector
        self.t = t.squeeze()
        #assert shapes 
        assert len(self.t.shape)==1, 'target shape eror'
        assert self.x.shape[0]==self.t.shape[0], 'input/target shape missmatch'

        self.n = len(self.x)
        #invalidate cache
        self.invalidate_cache()
        #process
        if not process:
            return
        
        if(self.Smean):
            self.mean = self.t.mean()
            self.t = self.t -self.mean
        else:
            self.mean = 0

        if(self.rescale):
            self.nrX = array([isreal(x[0,i]) for i in rescale_dim])
            #create a  copy before rescaling
            self.x = self.x.copy()
            x_   = array(self.x[:,self.nrX],dtype='float')
            minX = x_.min(axis=0)
            maxX = x_.max(axis=0)
            scaleX = 10/(maxX-minX)
            #where there is no variation set scaling to 1
            Iinf = isinf(scaleX)
            scaleX[Iinf] = 1
            # x_ is from -5..5 for each dimension
            x_   = (x_-minX)* scaleX -5 
            #store scaling parameters
            self.minX = minX
            self.scaleX = scaleX
            self.x[:,self.nrX] = x_
            pass

        pass
        
        
    def lMl(self,modelparameters,lml=True,dlml=True,clml=True,cdlml=True,priors=None,
            Ifilter_dlml=None):
        """
        Returns the log Marginal likelyhood for the given logtheta.

        **Parameters:**

        logtheta : [double]
            Array of hyperparameters, which define the covariance function

        lml : boolean
            return the log marginal likelihood

        dlml : boolean
            return the derivative of the log marginal likelihood

        clml : boolean
            calculate the quantities for caching

        sdlml : boolean
            calculate the quantities for caching

        priors : [:py:class:`lnpriors`]
            the priors for each hyperparameter, respectively
            
        Ifilter_dlml : [boolean]
            Filter of which parameters will be the derivative calculated

        these are passed onto the lower level covariance functions,
        reducing the computationally complexity
        """

        #is the lml/dlml cached ?

        logtheta = modelparameters['covar']
        
        if array((logtheta==self.modelparameters['covar'])).all() and (self.cached_lMl is not None):
            lMl = self.cached_lMl
            dlMl= self.cached_dlMl
        else:
            
            #else calculate
            pvalues = zeros([size(logtheta),2])
            # exponentiate parameters
            theta = exp(logtheta)
            if(priors):
                for i in range(size(logtheta)):
                    pvalues[i,:] = priors[i][0](theta[i],priors[i][1])
                #*exp(loghteta) to get chainrule right
                pvalues[:,1]*=theta
                
            try:   
                [L,alpha] = self.getCovariances(logtheta)
            except Exception,e:
                LG.error("exception caught (%s)" % (str(exp(logtheta))))
                lMl = self.cached_lMl+5000
                dlMl = self.cached_dlMl
                
                self.modelparamaters['covar'] = logtheta
                self.cached_lMl = lMl
                self.cached_dlMl = dlMl
                clml=False
                cdlml=False
                print e 

            if(clml):
                lMl = 0.5*dot(alpha,self.t) + sum(log(L.diagonal())) + 0.5*self.n*log(2*pi)
                lMl = lMl - sum(pvalues[:,0])
                self.cached_lMl = lMl
            if(cdlml):
                modelparameters_copy = modelparameters.copy()
                modelparameters_copy['covar'] = logtheta[self.IlogthetaK]
                Kd = self.covar.Kd(modelparameters_copy,self.x)

    #            dlMl= zeros(self.covar.getNparams())
    #            W   = linalg.solve(L.transpose(),(linalg.solve(L,eye(self.n)))) - dot(alpha,alpha)
    #            #dlMl[:]= 0.5*((Kd[:,:,:]*W).sum(axis=1)).sum(axis=1)
    #            for i in range(size(logtheta)):
    #                dlMl[i]= 0.5*((Kd[i,:,:]*W).sum()).sum()

                K = self.covar.K(modelparameters_copy,self.x)    
                D = Kd
                try:
                    iC = linalg.inv(K)
                    #rv0 = -0.5*trace(dot(D,iC),axis1=1,axis2=2)
                    #much faster:
                    rv0 = -0.5*(D*iC).sum(axis=1).sum(axis=1)

                    R = dot(D,dot(iC,self.t))
                    L = dot(self.t,iC)
                    rv1 = 0.5*dot(R,L)
                    rv = rv1 + rv0
                    dlMl = -rv
                    dlMl = dlMl - pvalues[:,1]
                    self.cached_dlMl=dlMl
                except Exception, e:
                    lMl+=5000
                    dlMl = self.cached_dlMl
        #return appropriate stuff:
        if lml and not dlml:
            return lMl
        elif dlml and not lml:
            return dlMl
        else:
            return [lMl,dlMl]

    def getCovariances(self,logtheta):
        """
        Return the Cholesky decompositions L and alpha::

            L     = chol(K)
            alpha = solve(L,t)
            return [L,Alpha] = getCovariances()
        """
#        print str(logtheta)
        if array((logtheta==self.modelparameters['covar'])).all() and (self.cached_L is not None):
            return [self.cached_L,self.cached_alpha]
        else:
            #recalculate it:
            if size(logtheta)!=self.covar.get_number_of_parameters():
                LG.error("wrong number of parameters for covariance Function")
            #this is all along the lines of Karl rasmussen's code:
            self.modelparameters['covar'] = logtheta.copy()
            modelparameters_copy = self.modelparameters.copy()
            modelparameters_copy['covar'] = logtheta[self.IlogthetaK]
                
            K = self.covar.K(modelparameters_copy,self.x)
            self.cached_L = linalg.cholesky(K)
            self.cached_alpha = _solve_chol(self.cached_L.transpose(),self.t)
            return [self.cached_L,self.cached_alpha]
        #::
        
    def rescaleInputs(self,xstar):
        """
        rescale prediction targets, i.e. invert
        the scaling transformation of inputs
        """
        x_ = xstar[:,self.nrX]
        XS = xstar.copy()
        XS[:,self.nrX] = (x_-self.minX)*self.scaleX - 5
        return XS
        
        
    def predict(self,modelparameters,xstar,mean=True,rescale=True,var=True):
        '''
        Predict mean and variance for given **Parameters:**

        modelparameters : [double]
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        mean     : boolean
            add mean to the prediction(True)

        rescale  : boolean
            rescale in the same way as training data has(True)

        var      : boolean
            return predicted variance
        '''
        #NOTE: this not quite optimal if we are only interested in the variance
        logtheta = modelparameters['covar']
        
        #1. rescale xsta
        if(self.rescale & rescale):
            #1. rescale range
            xstar = self.rescaleInputs(xstar)
            
        [L,alpha] = self.getCovariances(logtheta)
        
        modelparameters_copy = modelparameters.copy()
        modelparameters_copy['covar'] = logtheta[self.IlogthetaK]

        Kstar       = self.covar.K(modelparameters_copy,self.x,xstar)
        
        mu = dot(Kstar.transpose(),alpha)
        S2 = 0
        if(mean):
            mu = mu + self.mean
        if(var):            
            Kss         = self.covar.K(modelparameters_copy,xstar)
            v    = linalg.solve(L,Kstar)
            S2   = Kss.diagonal() - sum(v*v,0).transpose()
            S2   = abs(S2)
            #test
            #K = self.covar.K(logtheta[self.IlogthetaK],self.x)
            #Ki = linalg.inv(K)
            #v_    = dot(Ki,Kstar)
            #S2_ = Kss.diagonal() - dot(Kstar.T,dot(Ki,Kstar))
        return [mu,S2]
        
    def predictM(self,modelparameters,xstar,mean=True):
        '''
        same as predict but only does mean prediction and ignores variance

        **Parameters:**

        See :py:func:`gpr.GP.predict`
        '''
        logtheta = modelparameters['covar']
        
        [L,alpha] = self.getCovariances(logtheta)

        modelparameters_copy = modelparameters.copy()
        modelparameters_copy['covar'] = logtheta[self.IlogthetaK]
        
        Kstar       = self.covar.K(modelparameters_copy,self.x,xstar)
        
        mu = dot(Kstar.transpose(),alpha)
        if(mean):
            mu = mu + self.mean
        v    = linalg.solve(L,Kstar)
        return mu
    
class GPex(GP):
    """
    Gaussian Process regression class. Holds all information
    for the GP regression to take place. Additionally it provides
    an indicator Iexp, indicating which hyperpriors shall be exponentiated
    for the optimizer.

    **Parameters:**

    See :py:class:`gpr.GP`
    """
    
    def lMl(self,modelparameters,lml=True,dlml=True,clml=True,cdlml=True,priors=None,Ifilter_dlml=None,Iexp=None):
        """
        **Parameters:**

        Iexp : [boolean]
            indicator which hyperparmeters are exponentiated

        others :
            See :py:func:`gpr.GP.lMl`

        """

        if Iexp is None:
            Iexp = ones(logtheta.shape,dtype='bool')

        logtheta = modelparameters['covar']
        
        if (logtheta==self.logtheta).all() and (self.cached_lMl is not None):
            lMl = self.cached_lMl
            dlMl= self.cached_dlMl
        else:
            
            #else calculate
            pvalues = zeros([size(logtheta),2])
            # exponentiate parameters, those we need
            theta = logtheta.copy()
            theta[Iexp] = exp(logtheta[Iexp])
            if priors is not None:
                for i in range(size(logtheta)):
                    pvalues[i,:] = priors[i][0](theta[i],priors[i][1])
                #*exp(loghteta) to get chainrule right
                pvalues[Iexp,1]*=theta[Iexp]
                
            try:   
                [L,alpha] = self.getCovariances(logtheta)
            except Exception,e:
                LG.error("exception caught (%s)" % (str(exp(logtheta))))
                #if self.cached_lMl is not None:
                lMl = self.cached_lMl+5000
                # else:
                #     lMl = 5000
                dlMl = self.cached_dlMl
                
                self.logtheta = logtheta
                self.cached_lMl = lMl
                self.cached_dlMl = dlMl
                clml=False
                cdlml=False
                print e 

            if(clml):
                lMl = 0.5*dot(alpha,self.t) + sum(log(L.diagonal())) + 0.5*self.n*log(2*pi)
                lMl = lMl - sum(pvalues[:,0])
                self.cached_lMl = lMl
            if(cdlml):
                modelparameters_copy = modelparameters.copy()
                modelparameters_copy['covar'] = logtheta[self.IlogthetaK]
        
                Kd = self.covar.Kd(modelparameters_copy,self.x)
                K = self.covar.K(modelparameters_copy,self.x)    
                D = Kd
                try:
                    iC = linalg.inv(K)
                    #rv0 = -0.5*trace(dot(D,iC),axis1=1,axis2=2)
                    #much faster:
                    rv0 = -0.5*(D*iC).sum(axis=1).sum(axis=1)

                    R = dot(D,dot(iC,self.t))
                    L = dot(self.t,iC)
                    rv1 = 0.5*dot(R,L)
                    rv = rv1 + rv0
                    dlMl = -rv
                    dlMl = dlMl - pvalues[:,1]
                    self.cached_dlMl=dlMl
                except Exception, e:
                    lMl+=5000
                    dlMl = self.cached_dlMl
        #return appropriate stuff:
        if lml and not dlml:
            return lMl
        elif dlml and not lml:
            return dlMl
        else:
            return [lMl,dlMl]


