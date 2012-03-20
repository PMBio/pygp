"""
Package for Gaussian Process Optimization
=========================================

This package provides optimization functionality
for hyperparameters of covariance functions
:py:class:`pygp.covar` given. 

"""


# import scipy:
import scipy as SP
import scipy.optimize as OPT
import logging as LG
import pdb

# LG.basicConfig(level=LG.INFO)

def param_dict_to_list(dict,skeys=None):
    """convert from param dictionary to list"""
    #sort keys
    RV = SP.concatenate([dict[key].flatten() for key in skeys])
    return RV
    pass

def param_list_to_dict(list,param_struct,skeys):
    """convert from param dictionary to list
    param_struct: structure of parameter array
    """
    RV = []
    i0= 0
    for key in skeys:
        val = param_struct[key]
        shape = SP.array(val) 
        np = shape.prod()
        i1 = i0+np
        params = list[i0:i1].reshape(shape)
        RV.append((key,params))
        i0 = i1
    return dict(RV)

def checkgrad(f, fprime, x, *args,**kw_args):
    """
    Analytical gradient calculation using a 3-point method
    
    """
    
    import numpy as np
    
    # using machine precision to choose h
    eps = np.finfo(float).eps
    step = np.sqrt(eps)*(x.min())
    # shake things up a bit by taking random steps for each x dimension
    h = step*np.sign(np.random.uniform(-1, 1, x.size))
    
    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*h)
    analytical_gradient = fprime(x, *args, **kw_args)
    ratio = (f_ph - f_mh)/(2*np.dot(h, analytical_gradient))

    if True:
	h = np.zeros_like(x)
	
	for i in range(len(x)):
	    h[i] = step
	    f_ph = f(x+h, *args, **kw_args)
	    f_mh = f(x-h, *args, **kw_args)

	    numerical_gradient = (f_ph - f_mh)/(2*step)
	    analytical_gradient = fprime(x, *args, **kw_args)[i]
	    ratio = (f_ph - f_mh)/(2*step*analytical_gradient)
	    
	    h[i] = 0

	    print "[%d] numerical: %f, analytical: %f, ratio: %f" % (i, numerical_gradient,
								     analytical_gradient,
								     ratio)
	    


def opt_hyper(gpr,hyperparams,Ifilter=None,maxiter=1000,gradcheck=False,bounds = None,optimizer=OPT.fmin_tnc,gradient_tolerance=1E-4,*args,**kw_args):
    """
    Optimize hyperparemters of :py:class:`pygp.gp.basic_gp.GP` ``gpr`` starting from given hyperparameters ``hyperparams``.

    **Parameters:**

    gpr : :py:class:`pygp.gp.basic_gp`
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

    bounds : [[min,max]]
        Array with min and max value that can be attained for any hyperparameter

    maxiter: int
        maximum number of function evaluations
    gradcheck: boolean 
        check gradients comparing the analytical gradients to their approximations
    optimizer: :py:class:`scipy.optimize`
        which scipy optimizer to use? (standard lbfgsb)

    ** argument passed onto LML**

    priors : [:py:class:`pygp.priors`]
        non-default prior, otherwise assume
        first index amplitude, last noise, rest:lengthscales
    """

    def f(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.LML(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        #LG.debug("L("+str(x_)+")=="+str(rv))
        if SP.isnan(rv):
            return 1E6
        return rv
    
    def df(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.LMLgrad(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        rv = param_dict_to_list(rv,skeys)
        #LG.debug("dL("+str(x_)+")=="+str(rv))
        if not SP.isfinite(rv).all(): #SP.isnan(rv).any():
            In = SP.isnan(rv)
            rv[In] = 1E6
        return rv[Ifilter_x]

    #0. store parameter structure
    skeys = SP.sort(hyperparams.keys())
    param_struct = dict([(name,hyperparams[name].shape) for name in skeys])

    
    #1. convert the dictionaries to parameter lists
    X0 = param_dict_to_list(hyperparams,skeys)
    if Ifilter is not None:
        Ifilter_x = SP.array(param_dict_to_list(Ifilter,skeys),dtype='bool')
    else:
        Ifilter_x = SP.ones(len(X0),dtype='bool')

    #2. bounds
    if bounds is not None:
        #go through all hyperparams and build bound array (flattened)
        _b = []
        for key in skeys:
            if key in bounds.keys():
                _b.extend(bounds[key])
            else:
                _b.extend([(-SP.inf,+SP.inf)]*hyperparams[key].size)
        bounds = SP.array(_b)
        bounds = bounds[Ifilter_x]
        pass
       
        
    #2. set stating point of optimization, truncate the non-used dimensions
    x  = X0.copy()[Ifilter_x]
        
    LG.debug("startparameters for opt:"+str(x))
    
    if gradcheck:
	checkgrad(f, df, x)
        LG.info("check_grad (pre) (Enter to continue):" + str(OPT.check_grad(f,df,x)))
        raw_input()
	
    LG.debug("start optimization")

    #general optimizer interface
    #note: x is a subset of X, indexing the parameters that are optimized over
    # Ifilter_x pickes the subest of X, yielding x
    opt_RV=optimizer(f, x, fprime=df, maxfun=int(maxiter),pgtol=gradient_tolerance, messages=False, bounds=bounds)
    # optimizer = OPT.fmin_l_bfgs_b
    # opt_RV=optimizer(f, x, fprime=df, maxfun=int(maxiter),iprint =1, bounds=bounds, factr=10.0, pgtol=1e-10)
    opt_x = opt_RV[0]
    
    #relate back to X
    Xopt = X0.copy()
    Xopt[Ifilter_x] = opt_x
    #convert into dictionary
    opt_hyperparams = param_list_to_dict(Xopt,param_struct,skeys)
    #get the log marginal likelihood at the optimum:
    opt_lml = gpr.LML(opt_hyperparams,**kw_args)

    if gradcheck:
	checkgrad(f, df, opt_RV[0])
        LG.info("check_grad (post) (Enter to continue):" + str(OPT.check_grad(f,df,opt_RV[0])))

	pdb.set_trace()
        # raw_input()

    LG.debug("old parameters:")
    LG.debug(str(hyperparams))
    LG.debug("optimized parameters:")
    LG.debug(str(opt_hyperparams))
    LG.debug("grad:"+str(df(opt_x)))
    
    return [opt_hyperparams,opt_lml]
