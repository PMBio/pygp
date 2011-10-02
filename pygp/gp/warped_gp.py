"""
Warped Gaussian processes base class, overriding gp_base
"""
import sys
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
        assert psi.shape[0]==self.n_temrs, 'inconsistent parameter dimensions'
        assert psi.shape[1]==3, 'inconsistent parameter dimensions'

        #2. exponentiate the a and b (positive!)
        mpsi = psi.copy()
        mpsi[:,0:2] = SP.exp(mpsi[:,0:2])

        #3. evaluate the sum of factors in one go and sum
        # we need to do one tanh transformation per datum and n_term
        z = SP.asarray([y + m[0]*SP.tanh(m[1]*(y+m[2])) for m in mspi])
        #sum accross transformation terms
        z = z.sum(axis=1)
        return z

    def f_inv(self,z,psi):
        pass
   
    def fgrad_y(self,y,psi):
        """gradient of f w.r.t to y"""
        pass

    def fgrad_y_psi(self,y,psi):
        """gradient of f w.r.t to y"""
        pass
        
        
    
    
    


class WARPEDGP(GP):
    __slots__ = ["warping_function"]

    def __init__(self,warping_function,**kw_args):
        """warping_function: warping function of type WarpingFunction"""
        self.warping_function = warping_function
        super(GPLVM, self).__init__(**kw_args)
    
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
        #1. calculate standard LML
        LML = GP.LML(hperparams,*args,**,kw_args)
        #2. add jacobian from transformation
        #2.1 get grad y values from transformation
        warping_grad_y = self.warping_function.fgrad_y(self._get_active_set(self.y),hyperparams['warping'])
        LML += - SP.log(warping_grad_y).sum()
        

    def LMLgrad(self, hyperparams, *args, **kw_args):
        #1. call old code
        RV = GP.LMLgrad(hyperparams,*args,**kw_args)

        #2. add warping if in hyperparameter object
        if self.warping_function isnot None:
            RV.update(self._LMLgrad_warping(hyperparams))
        pass


    def _LMLgrad_warping(self,hyperparams):
        """gradient with respect to warping function parameters"""
        #1. get gradients of warping function with respect to y and params
        dfdt     = self.warping_function.f_grad_y(self._get_active_set(self.y),hyperparams['warping'])
        dfdtdpsi = self.warping_function.f_grad_y_psi(self._get_active_set(self.y),hyperparams['warping'])

        grad = - 1./dfdt * dfdtdpsi
        #sum over data
        grad.sum(axis=0)
        #create result structure

        RV = {'warping':grad}
        return RV
            
        
        
        


if __name__ == '__main__':
    #DEBUG
    
