"""
Covariance Function Combinators
-------------------------------
"""

import sys
sys.path.append('../')

from covar import CovarianceFunction
import scipy as S
import pdb



class SumCF(CovarianceFunction):
    """
    Sum Covariance function. This function adds
    up the given CFs and returns the resulting sum.

    *covars* : set of CFs of type :py:class:`covar.CovarianceFunction`
    
        Covariance functions to be summed up.
    """

#    __slots__ = ["n_params_list","covars","covars_logtheta_I"]

    def __init__(self,covars):
        #1. check that all covars are covariance functions
        #2. get number of params

        self.n_params_list = []
        self.covars = []
        self.covars_logtheta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        for nc in xrange(len(covars)):
            covar = covars[nc]
            assert isinstance(covar,CovarianceFunction), 'SumCovariance is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_logtheta_I.append(S.arange(i,i+covar.get_number_of_parameters()))
            for ip in xrange(Nparam):
                self.covars_covar_I.append(nc)
            i+=covar.get_number_of_parameters()
        self.n_params_list = S.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()



    def _parse_args(self,*args):
        x1 = args[0]
        if(len(args)==1):
            x2 = x1
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = args[1]
        return [x1,x2]

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        for covar in self.covars:
            names = S.concatenate((names,covar.get_hyperparameter_names()))
        return names

    def set_active_dimensions(self,**kwargin):
        """set active data dimension subset"""
        #this information is just passed on to the downstream covariance functions
        for covar in self.covars:
            covar.set_active_dimensions(**kwargin)

    def K(self,logtheta,*args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X']. The result
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
            K_ = covar.K(_logtheta,*args)
            if (nc==0):
                K = K_
            else:
                K+= K_
        return K




    def Kd(self,logtheta, x1,i):
        '''The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        '''
        #1. check logtheta has correct length
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d  = self.covars_logtheta_I[nc].min()
        j  = i-d
        return covar.Kd(logtheta[self.covars_logtheta_I[nc]],x1,j)
        

class ProductCF(CovarianceFunction):
#    __slots__=["n_params_list","covars","covars_logtheta_I"]

    def __init__(self,covars):
        #1. check that all covars are covariance functions
        #2. get number of params

        self.n_params_list = []
        self.covars = []
        self.covars_logtheta_I = []
        i = 0
        for covar in covars:
            assert isinstance(covar,CovarianceFunction), 'SumCovariance is constructed from a list of covaraince functions'
            self.n_params_list.append(covar.get_number_of_parameters())
            self.covars_logtheta_I.append(S.arange(i,i+covar.get_number_of_parameters()))
            i+=covar.get_number_of_parameters()
        self.n_params_list = S.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()
        self.covars = covars
        #convert the param lists to indicator vector to mak them easily invertable
        for n in xrange(len(covars)):
            _ilogtheta = S.zeros((self.n_hyperparameters),dtype='bool')
            _ilogtheta[self.covars_logtheta_I[n]]=True
            self.covars_logtheta_I[n] = _ilogtheta


    def _parse_args(self,*args):
        x1 = args[0]
        if(len(args)==1):
            x2 = x1
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = args[1]
        return [x1,x2]

    def setActiveDimensions(self,**kwargin):
        """set active data dimension subset"""
        #this information is just passed on to the downstream covariance functions
        for covar in self.covars:
            covar.set_active_dimensions(**kwargin)

    def getParamNames(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        for covar in self.covars:
            names = S.concatenate((names,covar.get_hyperparameter_names()))
        return names


    def K(self,logtheta,*args):
        "kernel"
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        #2. create sum of covarainces..
        [x1,x2] = self._parse_args(*args)
        K = S.ones([x1.shape[0],x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _logtheta = logtheta[self.covars_logtheta_I[nc]]
            K     *=  covar.K(_loghteta,*args)
        return K


    def Kd(self,logtheta, *args):
        "derivative kernel"
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        [x1,x2] = self._parse_args(*args)
        rv      = S.ones([self.n_hyperparameters,x1.shape[0],x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            #get kernel and derivative
            K_ = covar.K(logtheta[self.covars_logtheta_I[nc]],*args)
            Kd_= covar.Kd(logtheta[self.covars_logtheta_I[nc]],*args)
            #for the parmeters of this covariance multiply derivative
            rv[self.covars_logtheta_I[nc]] *= Kd_
            #for all remaining ones kernel
            rv[~self.covars_logtheta_I[nc]] *= K_
        return rv

