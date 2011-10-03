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


class TanhWarpingFunction(WarpingFunction):
    """implementaiton of the tanh warping fuction thing from Ed Snelson"""

    def sech(self, x, power=1):
	return (1.0/SP.cosh(x))**power

    def __init__(self,n_terms=3):
        """n_terms specifies the number of tanh terms to be used"""
        self.n_terms = n_terms
        pass

    def f(self,y,psi):
        #1. check that number of params is consistent
        assert psi.shape[0]==self.n_terms, 'inconsistent parameter dimensions'
        assert psi.shape[1]==3, 'inconsistent parameter dimensions'

        #2. exponentiate the a and b (positive!)
        mpsi = psi.copy()
        #mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

	z = y
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    z += a*SP.tanh(b*(y+c))
	    
        return z

    def f_inv(self,z,psi):
        pass
   
    def fgrad_y(self,y,psi):
        """gradient of f w.r.t to y"""

	mpsi = psi.copy()
	#mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

	grad = 1
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    grad += a*b*(1-SP.tanh(b*(y+c))**2)

        return grad


    def fgrad_y_psi(self,y,psi):
        """gradient of f w.r.t to y"""

	# 1. exponentiate the a and b (positive!)
        mpsi = psi.copy()
        #mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

	gradients = []
	
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    grad_a = b*self.sech(b*(c+y), 2)
	    grad_b = a*(1-2*b*(c+y)*SP.tanh(b*(c+y)))*self.sech(b*(c+y), 2)
	    grad_c = -2*a*(b**2)*SP.tanh(b*(c+y))* self.sech(b*(c+y),2)
	    
	    gradients.append([grad_a.flatten(), grad_b.flatten(), grad_c.flatten()])

	gradients = SP.asarray(gradients)
	
	return gradients
	   


class WARPEDGP(GP):
    __slots__ = ["warping_function"]

    def __init__(self, warping_function = None, n_terms = None, **kw_args):
        """warping_function: warping function of type WarpingFunction"""
        self.warping_function = warping_function(n_terms = n_terms)
        super(WARPEDGP, self).__init__(**kw_args)
    
    def _get_y(self,hyperparams):
        """get_y return the effect y being used"""
        #transform data using warping hyperparameters
        return self.warping_function.f(self._get_active_set(self.y),hyperparams['warping'])
    
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
        LML = super(WARPEDGP, self).LML(hyperparams,*args,**kw_args)

        # 2. add jacobian from transformation
        # 2.1 get grad y values from transformation
        warping_grad_y = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])

        LML -= SP.log(warping_grad_y).sum()

        return LML

    def LMLgrad(self, hyperparams, *args, **kw_args):
        #1. call old code%
        RV = super(WARPEDGP, self).LMLgrad(hyperparams,*args,**kw_args)

        #2. add warping if in hyperparameter object
        if self.warping_function is not None:
            RV.update(self._LMLgrad_warping(hyperparams))
	    
        return RV


    def _LMLgrad_warping(self,hyperparams):
        """gradient with respect to warping function parameters"""
        #1. get gradients of warping function with respect to y and params
        dfdt     = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])
        dfdtdpsi = self.warping_function.fgrad_y_psi(self._get_active_set(self.y),hyperparams['warping'])

	warp_grad = SP.zeros_like(dfdtdpsi)
	for i in range(warp_grad.shape[0]):
	    for j in range(warp_grad.shape[1]):	    
		warp_grad[i,j,:] = (-1./dfdt).flatten()*dfdtdpsi[i,j,:]

	grad = warp_grad.sum(axis=2)
        #create result structure
        RV = {'warping':grad}
	
        return RV
            
        
        
        


if __name__ == '__main__':
    import pylab as PL
    from pygp.covar import se, noise, combinators
    import pygp.plot.gpr_plot as gpr_plot
    import pygp.priors.lnpriors as lnpriors
    import pygp.likelihood as lik
    import pygp.plot.gpr_plot as gpr_plot
    import pygp.priors.lnpriors as lnpriors

    LG.basicConfig(level=LG.INFO)
    SP.random.seed(1)

    n_dimensions = 1
    xmin, xmax = 1, 2.5*SP.pi
    
    x = SP.arange(xmin,xmax,0.05)
    X = SP.linspace(0,10,100)[:,SP.newaxis] # predictions
    
    b = 0
    C = 2
    sigma = 0.01

    noise = sigma*SP.random.randn(len(x))
    y  = b*x + C + 1*SP.sin(x) + noise
    # warp the data using a simple function
    print "Y before warping: ", y
    y = y**(1/float(3))
    print "Y after warping: ", y
    
    y-= y.mean()
    x = x[:,SP.newaxis]
    

    
    # build GP
    likelihood = lik.GaussLikISO()
    covar_parms = SP.log([1,1])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1]), 'warping': SP.log(SP.random.randn(2,3))}
    #hyperparams = {'covar':covar_parms,'lik':SP.log([1])}    
    SECF = se.SqexpCFARD(n_dimensions=n_dimensions)
    covar = SECF
    covar_priors = []
    # scale
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors.extend([[lnpriors.lnGammaExp,[1,1]] for i in xrange(n_dimensions)])
    lik_priors = []
    # noise
    lik_priors.append([lnpriors.lnGammaExp,[1,1]])
    priors = {'covar':covar_priors,'lik':lik_priors}

    gp = WARPEDGP(warping_function = TanhWarpingFunction, n_terms = 2, covar_func=covar, likelihood=likelihood, x=x, y=y)
    #gp = GP(covar,likelihood=likelihood,x=x,y=y)    
    opt_model_params = opt_hyper(gp,hyperparams,priors=priors,gradcheck=True)[0]
    
    #predict
    [M,S] = gp.predict(opt_model_params,X)

    #create plots
    gpr_plot.plot_sausage(X,M,SP.sqrt(S))
    gpr_plot.plot_training_data(x,y)
    PL.show()
