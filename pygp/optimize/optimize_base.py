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


def checkgrad(f,fprime,x,step=1e-3, tolerance = 1e-4, *args,**kw_args):
	"""check the gradient function fprime by comparing it to a numerical estiamte from the function f"""
	import numpy as np
	import scipy as sp

        if 1:
            numerical_gradient = SP.zeros_like(x)
            for i in xrange(x.shape[0]):
                for j in xrange(x.shape[1]):
                    #choose a random direction to step in:
                    dx = step*SP.absolute(x[i,j])
                    x_ = SP.zeros_like(x)
                    x_[i,j]=dx
                    f1 = f(x+x_,*args,**kw_args)
                    f2 = f(x-x_,*args,**kw_args)
                    ng = (f1-f2)/(2*dx)
                    numerical_gradient[i,j] = ng
            gradient = SP.squeeze(fprime(x,*args,**kw_args))
            ratio = (gradient - numerical_gradient)/(gradient+1E-10)

        if 0:
            dx = step*np.sign(np.random.uniform(-1,1,x.shape))

            #evaulate around the point x
            f1 = f(x+dx,*args,**kw_args)
            f2 = f(x-dx,*args,**kw_args)

            numerical_gradient = (f1-f2)/(2*dx)
            gradient = fprime(x,*args,**kw_args)
            ratio = (gradient - numerical_gradient)
        #ratio = (f1-f2)/(2*np.dot(dx,gradient))
	print "gradient = ",gradient
	print "numerical gradient = ",numerical_gradient
	print "Delta = ", ratio, '\n'

	if (np.abs(ratio)>tolerance).any():
            print "outch"
            pdb.set_trace()
            pass
		## print "Ratio far from unity. Testing individual gradients"
		## for i in range(len(x)):
		## 	dx = np.zeros(x.shape)
		## 	dx[i] = step*np.sign(np.random.uniform(-1,1,x[i].shape))

		## 	f1 = f(x+dx,*args,**kw_args)
		## 	f2 = f(x-dx,*args,**kw_args)

		## 	numerical_gradient = (f1-f2)/(2*dx)
		## 	gradient = fprime(x,*args,**kw_args)
		## 	print i,"th element"
		## 	#print "gradient = ",gradient
		## 	#print "numerical gradient = ",numerical_gradient
		## 	ratio = (f1-f2)/(2*np.dot(dx,gradient))
		## 	print "ratio = ",ratio,'\n'


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
        if SP.isnan(rv).any():
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
        LG.info("check_grad (pre) (Enter to continue):" + str(OPT.check_grad(f,df,x)))
        raw_input()
	
    LG.debug("start optimization")

    #general optimizer interface
    #note: x is a subset of X, indexing the parameters that are optimized over
    #Ifilter_x pickes the subest of X, yielding x
    opt_RV=optimizer(f, x, fprime=df, maxfun=int(maxiter),pgtol=gradient_tolerance,messages=True,bounds=bounds)
#     optimizer = OPT.fmin_l_bfgs_b
#     opt_RV=optimizer(f, x, fprime=df, maxfun=int(maxiter),iprint = 1,bounds=bounds)
    opt_x = opt_RV[0]
    
    #relate back to X
    Xopt = X0.copy()
    Xopt[Ifilter_x] = opt_x
    #convert into dictionary
    opt_hyperparams = param_list_to_dict(Xopt,param_struct,skeys)
    #get the log marginal likelihood at the optimum:
    opt_lml = gpr.LML(opt_hyperparams,**kw_args)

    if gradcheck:
        LG.info("check_grad (post) (Enter to continue):" + str(OPT.check_grad(f,df,opt_RV[0])))
        raw_input()

    LG.debug("old parameters:")
    LG.debug(str(hyperparams))
    LG.debug("optimized parameters:")
    LG.debug(str(opt_hyperparams))
    LG.debug("grad:"+str(df(opt_x)))
    
    return [opt_hyperparams,opt_lml]
