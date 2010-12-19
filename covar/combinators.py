"""
Covariance Function Combinators
-------------------------------

Each combinator is a covariance function (CF) itself. It combines one or several covariance function(s) into another. For instance, :py:class:`combinators.SumCF` combines all given CFs into one sum; use this class to add noise.

"""

import sys
sys.path.append('../')

from pygp.covar import CovarianceFunction
from pygp.covar import CF_Kd_dx

import scipy as SP
import pdb



class SumCF(CF_Kd_dx):
    """
    Sum Covariance function. This function adds
    up the given CFs and returns the resulting sum.

    *covars* : [:py:class:`covar.CovarianceFunction`]
    
        Covariance functions to sum up.
    """

#    __slots__ = ["n_params_list","covars","covars_logtheta_I"]

    def __init__(self,covars,*args,**kw_args):
        #1. check that all covars are covariance functions
        #2. get number of params
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.n_params_list = []
        self.covars = []
        self.covars_logtheta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            assert isinstance(covar,CovarianceFunction), 'SumCF: SumCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_logtheta_I.append(SP.arange(i,i+covar.get_number_of_parameters()))
            for ip in xrange(Nparam):
                self.covars_covar_I.append(nc)
            i+=covar.get_number_of_parameters()
            
        self.n_params_list = SP.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identification easier"""
        names = []
        for covar in self.covars:
            names = SP.concatenate((names,covar.get_hyperparameter_names()))
        return names

    def K(self,logtheta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and x2. The result
        will be the sum covariance of all covariance
        functions combined in this sum covariance.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction` 
        """
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        #2. create sum of covarainces..
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            K_ = covar.K(_logtheta,x1,x2)
            if (nc==0):
                K = K_
            else:
                K+= K_
        return K

    def Kd(self,logtheta, x1, i):
        '''
        The partial derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d  = self.covars_logtheta_I[nc].min()
        j  = i-d
        return covar.Kd(logtheta[self.covars_logtheta_I[nc]],x1,j)
        
    def get_Iexp(self,logtheta):
        Iexp = []
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            Iexp = SP.concatenate((Iexp,covar.get_Iexp(_logtheta)))
        return SP.array(Iexp,dtype='bool')

    #derivative with respect to inputs
    def Kd_dx(self,logtheta,x1,d):
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        RV = SP.zeros([x1.shape[0],x1.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            RV += covar.Kd_dx(_logtheta,x1,d)

        return RV

class ProductCF(CovarianceFunction):
    """
    Product Covariance function. This function multiplies
    the given CFs and returns the resulting product.
    
    **Parameters:**
    
    covars : [CFs of type :py:class:`covar.CovarianceFunction`]
    
        Covariance functions to be multiplied.
        
    """
    #    __slots__=["n_params_list","covars","covars_logtheta_I"]
    
    def __init__(self,covars,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.n_params_list = []
        self.covars = []
        self.covars_logtheta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            assert isinstance(covar,CovarianceFunction), 'ProductCF: ProductCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_logtheta_I.append(SP.arange(i,i+covar.get_number_of_parameters()))
            for ip in xrange(Nparam):
                self.covars_covar_I.append(nc)
            i+=covar.get_number_of_parameters()
            
        self.n_params_list = SP.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        for covar in self.covars:
            names = SP.concatenate((names,covar.get_hyperparameter_names()))
        return names


    def K(self,logtheta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and x2. The result
        will be the product covariance of all covariance
        functions combined in this product covariance.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction` 
        """
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'ProductCF: K: logtheta has wrong shape'
        #2. create sum of covarainces..
        if x2 is None:
            K = SP.ones([x1.shape[0],x1.shape[0]])
        else:
            K = SP.ones([x1.shape[0],x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            K     *=  covar.K(_logtheta,x1,x2)
        return K


    def Kd(self,logtheta, x1, i):
        '''The derivatives of the covariance matrix for
        the i-th hyperparameter.
        
        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters,'ProductCF: K: logtheta has wrong shape'
        # nc = self.covars_covar_I[i]
        # covar = self.covars[nc]
        # d  = self.covars_logtheta_I[nc].min()
        # j  = i-d
        # return covar.Kd(logtheta[self.covars_logtheta_I[nc]],x1,j)
        # rv = SP.ones([self.n_hyperparameters,x1.shape[0],x2.shape[0]])
        # for nc in xrange(len(self.covars)):
        #     covar = self.covars[nc]
        #     #get kernel and derivative
        #     K_ = covar.K(logtheta[self.covars_logtheta_I[nc]],*args)
        #     Kd_= covar.Kd(logtheta[self.covars_logtheta_I[nc]],*args)
        #     #for the parmeters of this covariance multiply derivative
        #     rv[self.covars_logtheta_I[nc]] *= Kd_
        #     #for all remaining ones kernel
        #     rv[~self.covars_logtheta_I[nc]] *= K_
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d  = self.covars_logtheta_I[nc].min()
        j  = i-d
        Kd = covar.Kd(logtheta[self.covars_logtheta_I[nc]],x1,j)
        for ind in xrange(len(self.covars)):
            if(ind is not nc):
                _logtheta = logtheta[self.covars_logtheta_I[ind]]
                Kd *= self.covars[ind].K(_logtheta,x1)
        return Kd

    def get_Iexp(self,logtheta):
        Iexp = []
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            Iexp = SP.concatenate((Iexp,covar.get_Iexp(_logtheta)))
        return SP.array(Iexp,dtype='bool')

class ShiftCF(CovarianceFunction):
    """
    Time Shift Covariance function. This covariance function depicts
    the time shifts induced by the data and covariance function given
    and passes the shifted inputs to the covariance function given.
    To calculate the shifts of the inputs make shure the covariance
    function passed implements the derivative after the input
    Kd_dx(logtheta, x).
    
    covar : CF of type :py:class:`covar.CovarianceFunction`
    
        Covariance function to be used to depict the time shifts.
    
    replicate_indices : [int]

        The indices of the respective replicates, corresponding to
        the inputs. For instance: An input with three replicates:

        ===================== ========= ========= =========
        /                     rep1      rep2      rep3
        ===================== ========= ========= =========
        input = [             -1,0,1,2, -1,0,1,2, -1,0,1,2]
        replicate_indices = [ 0,0,0,0,  1,1,1,1,  2,2,2,2]
        ===================== ========= ========= =========

            
        Thus, the replicate indices represent
        which inputs correspond to which replicate.
        
    """
#    __slots__=["n_params_list","covars","covars_logtheta_I"]

    def __init__(self,covar,replicate_indices,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        #1. check that covar is covariance function
        assert isinstance(covar,CF_Kd_dx),'ShiftCF: ShiftCF is constructed from a CF_Kd_dx, which provides the partial derivative for the covariance matrix K with respect to input X'
        #2. get number of params
        self.replicate_indices = replicate_indices
        self.n_replicates = len(SP.unique(replicate_indices))
        self.n_hyperparameters = covar.get_number_of_parameters() + self.n_replicates
        self.covar = covar

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names=SP.concatenate(covar.get_hyperparameter_names(),["Time-Shift rep%i" % (i) for i in SP.unique(self.replicate(indices))])
        return names

    def K(self,logtheta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        logtheta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        logtheta : [double]
            the hyperparameters of this CF. Its structure is as follows:
            [logtheta of covar, time-shift-parameters]
        
        Others see :py:class:`covar.CovarianceFunction` 
        """
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'ShiftCF: K: logtheta has wrong shape'
        #2. shift inputs of covarainces..
        # get time shift parameter
        covar_n_hyper = self.covar.get_number_of_parameters()
        # shift inputs
        T  = logtheta[covar_n_hyper:covar_n_hyper+self.n_replicates]
        shift_x1 = self._shift_x(x1.copy(),T)
        K = self.covar.K(logtheta[:covar_n_hyper],shift_x1,x2)
        return K


    def Kd(self,logtheta, x, i):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        logtheta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        logtheta : [double]
            the hyperparameters of this CF. Its structure is as follows::
            [logtheta of covar, time-shift-parameters]

        i : int
            the partial derivative of the i-th
            hyperparameter shal be returned. 
            
        """
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'ShiftCF: K: logtheta has wrong shape'
        covar_n_hyper = self.covar.get_number_of_parameters()
        T  = logtheta[covar_n_hyper:covar_n_hyper+self.n_replicates]
        shift_x = self._shift_x(x.copy(), T)        
        if i >= covar_n_hyper:
            Kd_dx = self.covar.Kd_dx(logtheta[:covar_n_hyper],shift_x)
            c = SP.array(self.replicate_indices==(i-covar_n_hyper),
                         dtype='int')[:,SP.newaxis]
            cdist = self._pointwise_distance(-c,-c)
            cdist = cdist.transpose(2,0,1)
            return Kd_dx * cdist
        else:
            return self.covar.Kd(logtheta[:covar_n_hyper],shift_x,i)

    def get_Iexp(self, logtheta):
        """
        Return indices of which hyperparameters are to be exponentiated
        for optimization. Here we do not want
        
        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        covar_n_hyper = self.covar.get_number_of_parameters()
        Iexp = SP.concatenate((self.covar.get_Iexp(logtheta[:covar_n_hyper]),
                               SP.zeros(self.n_replicates)))
        Iexp = SP.array(Iexp,dtype='bool')
        return Iexp
        
    def _shift_x(self, x, T):
        # subtract T, respectively
        for i in SP.unique(self.replicate_indices):
            x[self.replicate_indices==i] -= T[i]
        return x
