"""
Warped Gaussian processes base class, overriding gp_base
"""
import sys
import logging as LG
sys.path.append('./../..')
from pygp.gp import GP
import pdb
from pygp.optimize.optimize_base import opt_hyper
import scipy as SP
import scipy.linalg as linalg

class MeanFunction(object):
    """
    abstract base clase for mean functions
    """

    def __init__(self):
        pass

    def f(self,psi):
        pass

    def fgrad_psi(self,psi):
        pass


class LinMeanFunction(MeanFunction):

    def __init__(self,X):
        self.X = X

    def f(self,psi):
        return SP.dot(self.X,psi)[:,SP.newaxis]

    def fgrad_psi(self,psi):
        return self.X

class WarpingFunction(object):
    """
    abstract function for warping
    z = f(y) 
    """

    def __init__(self):
        pass

    def f(self,y,psi):
        """function transformation
        y is a list of values (GP training data) of shpape [N,1]
        """
        pass

    def fgrad_y(self,y,psi):
        """gradient of f w.r.t to y"""
        pass

    def fgrad_y_psi(self,y,psi):
        """gradient of f w.r.t to y"""
        pass

    def f_inv(self,z,psi):
        """inverse function transformation"""
        pass

    def get_bounds(self, bounds_dict):
	""" returns the optimization bounds for the warping function """
	pass

    
class TanhWarpingFunction(WarpingFunction):
    """implementaiton of the tanh warping fuction thing from Ed Snelson"""

    def __init__(self,n_terms=3):
        """n_terms specifies the number of tanh terms to be used"""
        self.n_terms = n_terms
        pass

    def f(self,y,psi):
        """transform y with f using parameter vector psi
        psi = [[a,b,c]]
        f = \sum_{terms} a * tanh(b*(y+c))
        """

        #1. check that number of params is consistent
        assert psi.shape[0]==self.n_terms, 'inconsistent parameter dimensions'
        assert psi.shape[1]==3, 'inconsistent parameter dimensions'

        #2. exponentiate the a and b (positive!)
        mpsi = psi.copy()
        mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

        #3. transform data
	z = y.copy()
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    z += a*SP.tanh(b*(y+c))	    
        return z

    def f_inv(self, y, psi, iterations = 10):
        """
	calculate the numerical inverse of f

	== input ==
	iterations: number of N.R. iterations
	
	"""
	y = y.copy()
	z = self.f(y, psi)

	for i in range(iterations):
	    y -= (self.f(y, psi) - z)/self.fgrad_y(y,psi)
	
        return y
   
    def fgrad_y(self, y, psi, return_precalc = False):
        """
	gradient of f w.r.t to y ([N x 1])
	returns: Nx1 vector of derivatives, unless return_precalc is true,
	then it also returns the precomputed stuff
	"""

	mpsi = psi.copy()
	mpsi[:,0:2] = SP.exp(mpsi[:,0:2])
	s = SP.zeros((len(psi), y.shape[0], y.shape[1]))
	r = SP.zeros((len(psi), y.shape[0], y.shape[1]))	
	d = SP.zeros((len(psi), y.shape[0], y.shape[1]))
	
	grad = 1
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    s[i] = b*(y+c)
	    r[i] = SP.tanh(s[i])
	    d[i] = 1 - r[i]**2    
	    grad += a*b*d[i]

        #vectorized version
        S = (mpsi[:,1]*(y + mpsi[:,2])).T
        R = SP.tanh(S)
        D = 1-R**2
        GRAD = (1+(mpsi[:,0:1]*mpsi[:,1:2]*D).sum(axis=0))[:,SP.newaxis]

        if return_precalc:
            return GRAD,S,R,D
	    #return grad, s, r, d
	
	return grad


    def fgrad_y_psi(self, y, psi, return_covar_chain = False):
        """
	gradient of f w.r.t to y and psi

	returns: NxIx3 tensor of partial derivatives

	"""

	# 1. exponentiate the a and b (positive!)
        mpsi = psi.copy()
        mpsi[:,0:2] = SP.exp(mpsi[:,0:2])
	w, s, r, d = self.fgrad_y(y, psi, return_precalc = True)

	gradients = SP.zeros((y.shape[0], len(mpsi), 3))
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    gradients[:,i,0] = a*(b*(1.0/SP.cosh(s[i]))**2).flatten()
	    gradients[:,i,1] = b*(a*(1-2*s[i]*r[i])*(1.0/SP.cosh(s[i]))**2).flatten()
	    gradients[:,i,2] = (-2*a*(b**2)*r[i]*((1.0/SP.cosh(s[i]))**2)).flatten()

	covar_grad_chain = SP.zeros((y.shape[0], len(mpsi), 3))
	import numpy as NP
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    covar_grad_chain[:, i, 0] = a*(r[i])
            covar_grad_chain[:, i, 1] = b*(a*(c+y[:,0])*(1.0/SP.cosh(s[i]))**2)
	    covar_grad_chain[:, i, 2] = a*b*((1.0/SP.cosh(s[i]))**2).flatten()
    
	if return_covar_chain:
	    return gradients, covar_grad_chain
	return gradients

    def get_bounds(self, bounds_dict = None):
	"""
	Optimization bounds for the warping function. Returns a dictionary
	that contains (n_terms*3, 2) bounds (flattened)

	Input:

	bounds_dict -> dictionary containing existing bounds (default None)
	
	"""

	if bounds_dict == None:
	    bounds_dict = {}

	bounds = SP.zeros((self.n_terms, 3, 2))

	# no bounds for all parametrs
        bounds[:,:,0] = -SP.inf
        bounds[:,:,1] = +SP.inf
        # but the second one
        bounds[:,1,0] = -SP.inf
        bounds[:,1,1] = SP.log(20)	

	# flatten the bounds matrix across the first dimension
	bounds = bounds.reshape((bounds.shape[0]*bounds.shape[1], 2))

	bounds_dict["warping"] = bounds.tolist()
	
	return bounds_dict
	   

class WARPEDGP(GP):
    __slots__ = ["warping_function","mean_function"]

    def __init__(self, warping_function = None, mean_function = None, **kw_args):
        """warping_function: warping function of type WarpingFunction"""
        self.warping_function = warping_function
        self.mean_function    = mean_function
        super(WARPEDGP, self).__init__(**kw_args)
	
    def _get_y(self,hyperparams):
        """get_y return the effect y being used"""
        #transform data using warping hyperparameters
        y_ = self._get_active_set(self.y)
        if self.warping_function is not None:
            y_  = self.warping_function.f(y_,hyperparams['warping'])
        if self.mean_function is not None:
            y_ = y_ - self.mean_function.f(hyperparams['mean']) 
        return y_
            
    def LML(self,hyperparams, *args, **kw_args):
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
        # 1. calculate standard LML
        # (note: transformation implemented in _get_y)
        LML = super(WARPEDGP, self).LML(hyperparams,*args,**kw_args)
        if self.warping_function is not None:
            # 2. add jacobian from transoformation
            warping_grad_y = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])
            LML -= SP.log(warping_grad_y).sum()
        return LML

    def LMLgrad(self, hyperparams, *args, **kw_args):
        #1. call old code
        RV = super(WARPEDGP, self).LMLgrad(hyperparams,*args,**kw_args)

        #2. add warping if in hyperparameter object
        if self.warping_function is not None:
            RV.update(self._LMLgrad_warping(hyperparams))

        #3. add mean funciton derivative
        if self.mean_function is not None:
            RV.update(self._LMLgrad_mean(hyperparams))
	    
        return RV


    def predict(self, hyperparams, xstar, output=0, var=True):
        R = super(WARPEDGP, self).predict(hyperparams,xstar,output,var)
        #add mean prediction
        if self.mean_function is not None:
            mean = self.mean_function.f(hyperparams['mean'])

            if var:
                R[0] += mean[:,0]
            else:
                R += mean[:,0]
        return R
        
        

    def _LMLgrad_mean(self,hyperparams):
        # 2. derivative of quadtratic term in LML
        grad_f_psi = self.mean_function.fgrad_psi(hyperparams['mean'])
        #scale up K^{-1}*y (Kiy) for matrix operations with grad_psi
        Kiy = super(WARPEDGP, self).get_covariances(hyperparams)['alpha']

        mean_grad_quad = -1.0*SP.dot(grad_f_psi.T,Kiy[:,0]).T

        RV ={'mean': mean_grad_quad}
	
        return RV

    def _LMLgrad_warping(self,hyperparams):
        """gradient with respect to warping function parameters"""
        #1. get gradients of warping function with respect to y and params
        grad_y = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self._get_active_set(self.y),
								 hyperparams['warping'],
								 return_covar_chain = True)

        # 1. derivartive of log jacobian term of LML
        #scale up the inerse of grad_y
        Igrad_y_psi = SP.tile((1./grad_y)[:,:,SP.newaxis],(1,grad_psi.shape[1],grad_psi.shape[2]))
        #calculate chain rule of the log term with inner deritivae w.r.t. psi and sum over datapoitns
        warp_grad_det = -(Igrad_y_psi*grad_y_psi).sum(axis=0)

        # 2. derivative of quadtratic term in LML
        #scale up K^{-1}*y (Kiy) for matrix operations with grad_psi
        Kiy = super(WARPEDGP, self).get_covariances(hyperparams)['alpha']
        warp_grad_quad = SP.dot(grad_psi.T,Kiy[:,0]).T	
        #create result structure
        RV = {'warping':warp_grad_quad+warp_grad_det}
        return RV
            
        
                


if __name__ == '__main__':
    import pylab as PL
    from pygp.covar import se, noise, mu, combinators
    import pygp.plot.gpr_plot as gpr_plot
    import pygp.priors.lnpriors as lnpriors
    import pygp.likelihood as lik
    import pygp.plot.gpr_plot as gpr_plot
    import pygp.priors.lnpriors as lnpriors

    LG.basicConfig(level=LG.DEBUG)
    SP.random.seed(10)

    n_dimensions = 1
    xmin, xmax = 1, 2.5*SP.pi
    
    x = SP.linspace(xmin,xmax,500)
    
    print len(x)
    X = SP.linspace(0,10,100)[:,SP.newaxis] # predictions
    
    b = 1
    C = 0
    SNR = 0.1
    y  = b*x + C + 1*SP.sin(x) 

    sigma = SNR * (y.max()-y.mean())

    y += sigma*SP.random.randn(len(x))
    # warp the data using a simple function
    #y -= y.mean()
    # transform to -1..1
    
    x = x[:,SP.newaxis]

    def trafo(y):
        return y**(float(3))
    def Itrafo(y):
        return y**(1/float(3))

    z = trafo(y)
    L = (z.max()-z.min())
    z /= L
        
    n_terms = 3
    # build GP
    likelihood = lik.GaussLikISO()
    # covar_parms = SP.log([1,1,1E-5])
    covar_parms = SP.log([1,1])
    hyperparams = {'covar':covar_parms,'lik':SP.log([sigma])}

    
    SECF = se.SqexpCFARD(n_dimensions=n_dimensions)
    muCF = mu.MuCF(N=X.shape[0])
    #covar = combinators.SumCF([SECF,muCF])
    covar = SECF
    warping_function = None
    mean_function = None
    bounds = {}
    if 1:
        warping_function = TanhWarpingFunction(n_terms=n_terms)
        hyperparams['warping'] = 1E-2*SP.random.randn(n_terms,3)
        bounds.update(warping_function.get_bounds())
        
    if 0:
        mean_function    = LinMeanFunction(X= SP.ones([x.shape[0],1]))
        hyperparams['mean'] = 1E-2* SP.randn(1)

    gp = WARPEDGP(warping_function = warping_function, mean_function = mean_function, covar_func=covar, likelihood=likelihood, x=x, y=z)



    if warping_function is not None:
        PL.figure(1)
        z_values = SP.linspace(z.min(),z.max(),100)
        PL.plot(z_values,Itrafo(L*z_values))
        PL.plot(z_values,warping_function.f(z_values,hyperparams['warping']))
        PL.legend('real inverse','learnt inverse')


    if 0:
        #check gradients of warping function
        from pygp.optimize.optimize_base import checkgrad,OPT
        
        # derivative w.r.t. y
        # derivative w.r.t y psi
        def f1(x):
            return warping_function.f(x,hyperparams['warping'])
        def df1(x):
            return warping_function.fgrad_y(x,hyperparams['warping'])
        def f2(x):
	    return warping_function.fgrad_y(gp.y[10:11],x)
        def df2(x):
	    return warping_function.fgrad_y_psi(gp.y[10:11],x)

        C = SP.linalg.inv(gp.get_covariances(hyperparams)['K'])
        Cs = C.copy()
	def f3(x):
	    return SP.double(warping_function.pLML(x,C,gp.y))
        def df3(x):
	    return warping_function.pLMLgrad(x,C,gp.y)

        def f4(x):
            hyperparams['warping'][:] = x
            return gp.LML(hyperparams,)
        def df4(x):
            hyperparams['warping'][:] = x
            return gp.LMLgrad(hyperparams)['warping']

        x = hyperparams['warping'].copy()       
        checkgrad(f4,df4,x)

    lmld= gp.LMLgrad(hyperparams)
    print lmld


    
    #gp = GP(covar,likelihood=likelihood,x=x,y=y)    
    opt_model_params = opt_hyper(gp, hyperparams,
				 bounds = bounds,
				 maxiter=10000,
				 gradcheck=True)[0]

    
    PL.figure(2)
    z_values = SP.linspace(z.min(),z.max(),100)
    PL.plot(Itrafo(gp.y))
    #opt_hyperparamsPL.plot(z_values,Itrafo(L*z_values))
    pred_inverse = warping_function.f_inv(gp.y,
					  opt_model_params['warping'],
					  iterations = 10)
    PL.plot(pred_inverse)
    #PL.plot(z,y,'r.')
    #PL.legend(['real inverse','learnt inverse','data'])


