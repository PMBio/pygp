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
        mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

	z = y.copy()
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    z += a*SP.tanh(b*(y+c))
	    
        return z

    def plot_f(self, psi):
	Y = SP.arange(-10, 10, 0.1)

	f_y = []

	for y in Y:
	    f_y.append(self.f(y, psi).sum())

	PL.figure()
	PL.plot(Y, f_y)
	

    def f_inv(self,z,psi):
        pass
   
    def fgrad_y(self, y, psi, return_precalc = False):
        """
	gradient of f w.r.t to y

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

	if return_precalc:
	    return grad, s, r, d
	
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

	# TODO: precompute (1/cosh(s[i]))^2 for efficiency
	gradients = SP.zeros((y.shape[0], len(mpsi), 3))
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]

	    gradients[:,i,0] = a*(b*(1.0/SP.cosh(s[i]))**2).flatten()
	    gradients[:,i,1] = b*(a*(1-2*s[i]*r[i])*(1.0/SP.cosh(s[i]))**2).flatten()
	    gradients[:,i,2] = (-2*a*(b**2)*r[i]*((1.0/SP.cosh(s[i]))**2)).flatten()

	covar_grad_chain = SP.zeros((y.shape[0], len(mpsi), 3))
	for i in range(len(mpsi)):
	    a,b,c = mpsi[i]
	    covar_grad_chain[:, i, 0] = a*(r[i]).flatten()
	    covar_grad_chain[:, i, 1] = b*(a*(c+y)*(1.0/SP.cosh(s[i]))**2).flatten()
	    covar_grad_chain[:, i, 2] = a*b*((1.0/SP.cosh(s[i]))**2).flatten()

	    
	if return_covar_chain:
	    return gradients, covar_grad_chain
	
	return gradients
	   
    def horrible_shit(self, crap, crap_covar):
	def f(x):
	    f_y = self.f(y, x)
	    grad_y = self.fgrad_y(y, x)
	    ll1 = 0.5*SP.dot(SP.dot(f_y.T, C), f_y)
	    ll2 = - SP.log(grad_y).sum()	    
	    return ll1 + ll2

	def df(x):
	    f_y = self.f(y, x)
	    grad_y = self.fgrad_y(y, x)
	    grad_y_psi, grad_psi = self.fgrad_y_psi(y, x, return_covar_chain = True)


	    warp_grad = SP.zeros_like(grad_psi)
	    for i in range(warp_grad.shape[1]):
		for j in range(warp_grad.shape[2]):	    
		    warp_grad[:,i,j] = (-1./grad_y).flatten()*grad_y_psi[:,i,j]
		    # warp_grad[:,i,j] += SP.dot(grad_psi[:,i,j][:,SP.newaxis].T, SP.dot(C, f_y)).squeeze()

	    warp_grad = warp_grad.sum(axis=0)
	    for i in range(warp_grad.shape[0]):
		for j in range(warp_grad.shape[1]):	    
		    warp_grad[i,j]  += SP.dot(grad_psi[:,i,j][:,SP.newaxis].T, SP.dot(C, f_y)).squeeze()
	    return warp_grad#.sum(axis=0)

	y = crap
	C = SP.linalg.inv(crap_covar)
	psi = SP.random.randn(self.n_terms, 3)

	from pygp.optimize.optimize_base import checkgrad
	checkgrad(f,df,psi)

	

class WARPEDGP(GP):
    __slots__ = ["warping_function"]

    def __init__(self, warping_function = None, **kw_args):
        """warping_function: warping function of type WarpingFunction"""
        self.warping_function = warping_function
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

        LML += SP.log(warping_grad_y).sum()

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
        grad_y = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self._get_active_set(self.y),
								 hyperparams['warping'],
								 return_covar_chain = True)


	C = super(WARPEDGP, self).get_covariances(hyperparams)['alpha'] # returns Cinv*y
	
	warp_grad = SP.zeros_like(grad_psi)
	for i in range(warp_grad.shape[1]):
	    for j in range(warp_grad.shape[2]):	    
		warp_grad[:,i,j] = (-1./grad_y).flatten()*grad_y_psi[:,i,j]


	warp_grad = warp_grad.sum(axis=0)

	for i in range(warp_grad.shape[0]):
	    for j in range(warp_grad.shape[1]):	    
		warp_grad[i,j]  += SP.dot(grad_psi[:,i,j][:,SP.newaxis].T, C).squeeze()
		
	grad = warp_grad
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

    LG.basicConfig(level=LG.DEBUG)
    SP.random.seed(10)

    n_dimensions = 1
    xmin, xmax = 1, 2.5*SP.pi
    
    x = SP.arange(xmin,xmax,0.03)
    print len(x)
    X = SP.linspace(0,10,100)[:,SP.newaxis] # predictions
    
    b = 1
    C = 2
    sigma = 0.01

    noise = sigma*SP.random.randn(len(x))
    y  = b*x + C + 1*SP.sin(x) + noise
    # warp the data using a simple function
    y = y**(1/float(3))
    y-= y.mean()
    x = x[:,SP.newaxis]
    

    n_terms = 2
    # build GP
    likelihood = lik.GaussLikISO()
    covar_parms = SP.log([1,1])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1]), 'warping': (SP.random.randn(n_terms,3))}
    #hyperparams = {'covar':covar_parms,'lik':SP.log([1]), 'warping': SP.ones((n_terms,3))}
    hyperparams["warping"][:,0] += 2
    hyperparams['warping'][:,1] = 2
    hyperparams["warping"][0,2] += 5
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
    warping_function = TanhWarpingFunction(n_terms=n_terms)

    warping_function.plot_f(hyperparams["warping"])

    gp = WARPEDGP(warping_function = warping_function, covar_func=covar, likelihood=likelihood, x=x, y=y)
    
    if 1:
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
	def f3(x):
	    return warping_function.f(gp.y[10:11], x)
        def df3(x):
	    return warping_function.fgrad_y_psi(gp.y[10:11], x, return_covar_chain=True)[1]
	print "=== Gradients df/dy ==="
        checkgrad(f1,df1,gp.y[0:1,:])
	print "=== Gradients df/dy dpsi ==="
        checkgrad(f2,df2,hyperparams['warping'])
	print "=== Gradients df/dpsi ==="	
	checkgrad(f3,df3,hyperparams['warping'])
	warping_function.horrible_shit(gp.y, gp.get_covariances(hyperparams)['K'])


	pdb.set_trace()
    lmld= gp.LMLgrad(hyperparams)
    print lmld
    
    #gp = GP(covar,likelihood=likelihood,x=x,y=y)    
    opt_model_params = opt_hyper(gp,hyperparams, gradcheck=True)[0]
    
    #predict
    [M,S] = gp.predict(opt_model_params,X)
    warping_function.plot_f(opt_model_params["warping"])
#     #create plots
#     gpr_plot.plot_sausage(X,M,SP.sqrt(S))
#     gpr_plot.plot_training_data(x,y)
#     PL.show()
