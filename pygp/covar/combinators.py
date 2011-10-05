"""
Covariance Function Combinators
-------------------------------

Each combinator is a covariance function (CF) itself. It combines one or several covariance function(s) into another. For instance, :py:class:`pygp.covar.combinators.SumCF` combines all given CFs into one sum; use this class to add noise.

"""

from pygp.covar import CovarianceFunction
from pygp.covar.dist import dist
import scipy as sp
import sys
sys.path.append('../')
import pdb


class SumCF(CovarianceFunction):
    """
    Sum Covariance function. This function adds
    up the given CFs and returns the resulting sum.

    *covars* : [:py:class:`pygp.covar.CovarianceFunction`]
    
        Covariance functions to sum up.
    """

#    __slots__ = ["n_params_list","covars","covars_theta_I"]

    def __init__(self, covars, *args, **kw_args):
        #1. check that all covars are covariance functions
        #2. get number of params
        super(SumCF, self).__init__()
        self.n_params_list = []
        self.covars = []
        self.covars_theta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            assert isinstance(covar, CovarianceFunction), 'SumCF: SumCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_theta_I.append(sp.arange(i, i + covar.get_number_of_parameters()))
            self.covars_covar_I.extend(sp.repeat(nc, Nparam))
            i += covar.get_number_of_parameters()
        self.n_params_list = sp.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identification easier"""
        names = []
        for covar in self.covars:
            names = sp.concatenate((names, covar.get_hyperparameter_names()))
        return names

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the sum covariance of all covariance
        functions combined in this sum covariance.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        #2. create sum of covarainces..
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            K_ = covar.K(_theta, x1, x2)
            if (nc == 0):
                K = K_
            else:
                K += K_
        return K

    def Kgrad_theta(self, theta, x1, i):
        '''
        The partial derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d = self.covars_theta_I[nc].min()
        j = i - d
        return covar.Kgrad_theta(theta[self.covars_theta_I[nc]], x1, j)
        

    #derivative with respect to inputs
    def Kgrad_x(self, theta, x1, x2, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        RV = sp.zeros([x1.shape[0], x1.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            RV += covar.Kgrad_x(_theta, x1, x2, d)
        return RV

    #derivative with respect to inputs
    def Kgrad_xdiag(self, theta, x1, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        RV = sp.zeros([x1.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            RV += covar.Kgrad_xdiag(_theta, x1, d)
        return RV


class ProductCF(CovarianceFunction):
    """
    Product Covariance function. This function multiplies
    the given CFs and returns the resulting product.
    
    **Parameters:**
    
    covars : [CFs of type :py:class:`pygp.covar.CovarianceFunction`]
    
        Covariance functions to be multiplied.
        
    """
    #    __slots__=["n_params_list","covars","covars_theta_I"]
    
    def __init__(self, covars, *args, **kw_args):
        super(ProductCF, self).__init__()
        self.n_params_list = []
        self.covars = []
        self.covars_theta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            #assert isinstance(covar, CovarianceFunction), 'ProductCF: ProductCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_theta_I.append(sp.arange(i, i + covar.get_number_of_parameters()))
            self.covars_covar_I.extend(sp.repeat(nc, Nparam))
#            for ip in xrange(Nparam):
#                self.covars_covar_I.append(nc)
            
            i += Nparam
            
        self.n_params_list = sp.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        for covar in self.covars:
            names.extend(covar.get_hyperparameter_names())
        return names


    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the product covariance of all covariance
        functions combined in this product covariance.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ProductCF: K: theta has wrong shape'
        #2. create sum of covarainces..
        if x2 is None:
            K = sp.ones([x1.shape[0], x1.shape[0]])
        else:
            K = sp.ones([x1.shape[0], x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            K *= covar.K(_theta, x1, x2)
        return K


    def Kgrad_theta(self, theta, x, i):
        '''The derivatives of the covariance matrix for
        the i-th hyperparameter.
        
        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ProductCF: K: theta has wrong shape'
        # nc = self.covars_covar_I[i]
        # covar = self.covars[nc]
        # d  = self.covars_theta_I[nc].min()
        # j  = i-d
        # return covar.Kd(theta[self.covars_theta_I[nc]],x1,j)
        # rv = sp.ones([self.n_hyperparameters,x1.shape[0],x2.shape[0]])
        # for nc in xrange(len(self.covars)):
        #     covar = self.covars[nc]
        #     #get kernel and derivative
        #     K_ = covar.K(theta[self.covars_theta_I[nc]],*args)
        #     Kd_= covar.Kd(theta[self.covars_theta_I[nc]],*args)
        #     #for the parmeters of this covariance multiply derivative
        #     rv[self.covars_theta_I[nc]] *= Kd_
        #     #for all remaining ones kernel
        #     rv[~self.covars_theta_I[nc]] *= K_
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d = i - self.covars_theta_I[nc].min()
        Kd = covar.Kgrad_theta(theta[self.covars_theta_I[nc]],x,d)
        for ind in xrange(len(self.covars)):
            if(ind != nc):
                _theta = theta[self.covars_theta_I[ind]]
                Kd *= self.covars[ind].K(_theta, x)
        return Kd

    #derivative with respect to inputs
    def Kgrad_x(self, theta, x1, x2, d):
        assert theta.shape[0] == self.n_hyperparameters, 'Product CF: K: theta has wrong shape'
        RV_sum = sp.zeros([x1.shape[0], x1.shape[0]])
        RV_prod = sp.ones([x1.shape[0], x1.shape[0]])
        for nc in xrange(len(self.covars)):
            RV_prod = sp.ones([x1.shape[0], x1.shape[0]])
            for j in xrange(len(self.covars)):
                _theta = theta[self.covars_theta_I[j]]
                covar = self.covars[j]
                if(j==nc):
                    RV_prod*=covar.Kgrad_x(_theta,x1,x2,d)
                else:
                    RV_prod*=covar.K(_theta, x1, x2)
            RV_sum += RV_prod
        return RV_sum
#            covar = self.covars[nc]
#            if(d in covar.dimension_indices):
#                dims = covar.dimension_indices.copy()
#                #covar.set_dimension_indices([d])
#                _theta = theta[self.covars_theta_I[nc]]
#                K = covar.K(_theta,x1,x2)
#                RV_sum += covar.Kgrad_x(_theta, x1, x2, d)/K
#                #import pdb;pdb.set_trace()
#                RV_prod *= K
#                #covar.set_dimension_indices(dims)
#        return RV_sum*RV_prod
    
    def Kgrad_xdiag(self, theta, x1, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
#        RV_sum = sp.zeros([x1.shape[0], x1.shape[0]])
#        RV_prod = sp.ones([x1.shape[0], x1.shape[0]])
#        for nc in xrange(len(self.covars)):
#            covar = self.covars[nc]
#            _theta = theta[self.covars_theta_I[nc]]
#            if(d in covar.dimension_indices):
#                dims = covar.dimension_indices.copy()
#                #covar.set_dimension_indices([d])
#                _theta = theta[self.covars_theta_I[nc]]
#                K = covar.Kdiag(_theta,x1)
#                RV_sum += covar.Kgrad_xdiag(_theta, x1, d)/K
#                RV_prod *= K
#                #covar.set_dimension_indices(dims)
#        return RV_sum * RV_prod

#        pdb.set_trace()
        RV_sum = sp.zeros([x1.shape[0]])
        for nc in xrange(len(self.covars)):
            RV_prod = sp.ones([x1.shape[0]])
            for j in xrange(len(self.covars)):
                _theta = theta[self.covars_theta_I[j]]
                covar = self.covars[j]
                if(j==nc):
                    RV_prod*=covar.Kgrad_xdiag(_theta,x1,d)
                else:
                    RV_prod*=covar.Kdiag(_theta, x1)
            RV_sum += RV_prod
        return RV_sum

#    def get_Iexp(self, theta):
#        Iexp = []
#        for nc in xrange(len(self.covars)):
#            covar = self.covars[nc]
#            _theta = theta[self.covars_theta_I[nc]]
#            Iexp = sp.concatenate((Iexp, covar.get_Iexp(_theta)))
#        return sp.array(Iexp, dtype='bool')

class ShiftCF(CovarianceFunction):
    """
    Time Shift Covariance function. This covariance function depicts
    the time shifts induced by the data and covariance function given
    and passes the shifted inputs to the covariance function given.
    To calculate the shifts of the inputs make shure the covariance
    function passed implements the derivative after the input
    Kd_dx(theta, x).
    
    covar : CF of type :py:class:`pygp.covar.CovarianceFunction`
    
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
#    __slots__=["n_params_list","covars","covars_theta_I"]

    def __init__(self, covar, replicate_indices, *args, **kw_args):
        super(ShiftCF, self).__init__()
        #1. check that covar is covariance function
        assert isinstance(covar, CovarianceFunction), 'ShiftCF: ShiftCF is constructed from a CovarianceFunction, which provides the partial derivative for the covariance matrix K with respect to input X'
        #2. get number of params
        self.replicate_indices = replicate_indices
        self.n_replicates = len(sp.unique(replicate_indices))
        self.n_hyperparameters = covar.get_number_of_parameters() + self.n_replicates
        self.covar = covar

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        return sp.concatenate((self.covar.get_hyperparameter_names(),["Time-Shift rep%i" % (i) for i in sp.unique(self.replicate_indices)]))

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        theta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        theta : [double]
            the hyperparameters of this CF. Its structure is as follows:
            [theta of covar, time-shift-parameters]
        
        Others see :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ShiftCF: K: theta has wrong shape'
        #2. shift inputs of covarainces..
        # get time shift parameter
        covar_n_hyper = self.covar.get_number_of_parameters()
        # shift inputs
        T = theta[covar_n_hyper:covar_n_hyper + self.n_replicates]
        shift_x1 = self._shift_x(x1.copy(), T)
        K = self.covar.K(theta[:covar_n_hyper], shift_x1, x2)
        return K


    def Kgrad_theta(self, theta, x, i):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        theta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        theta : [double]
            the hyperparameters of this CF. Its structure is as follows::
            [theta of covar, time-shift-parameters]

        i : int
            the partial derivative of the i-th
            hyperparameter shal be returned. 
            
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ShiftCF: K: theta has wrong shape'
        covar_n_hyper = self.covar.get_number_of_parameters()
        T = theta[covar_n_hyper:covar_n_hyper + self.n_replicates]
        shift_x = self._shift_x(x.copy(), T)        
        if i >= covar_n_hyper:
            Kdx = self.covar.Kgrad_x(theta[:covar_n_hyper], shift_x, shift_x, 0)
            c = sp.array(self.replicate_indices == (i - covar_n_hyper),
                         dtype='int').reshape(-1,1)
            cdist = dist(-c, -c)
            cdist = cdist.transpose(2, 0, 1)
            try:
                return Kdx * cdist
            except ValueError:
                return Kdx

        else:
            return self.covar.Kgrad_theta(theta[:covar_n_hyper], shift_x, i)

#    def get_Iexp(self, theta):
#        """
#        Return indices of which hyperparameters are to be exponentiated
#        for optimization. Here we do not want
#        
#        **Parameters:**
#        See :py:class:`pygp.covar.CovarianceFunction`
#        """
#        covar_n_hyper = self.covar.get_number_of_parameters()
#        Iexp = sp.concatenate((self.covar.get_Iexp(theta[:covar_n_hyper]),
#                               sp.zeros(self.n_replicates)))
#        Iexp = sp.array(Iexp,dtype='bool')
#        return Iexp
        
    def _shift_x(self, x, T):
        # subtract T, respectively
        if(x.shape[0]==self.replicate_indices.shape[0]):
            for i in sp.unique(self.replicate_indices):
                x[self.replicate_indices == i] -= T[i]
        return x
