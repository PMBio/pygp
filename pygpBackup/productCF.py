"""
Covariance function: product of covariances
"""


from pygp.covar import CovarianceFunction
import scipy as S


class ProductCovariance(CovarianceFunction):
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


    def parse_args(self,*args):
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


    def K(self,modelparameters,*args):
        "kernel"
        logtheta = modelparameters['covar']
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        #2. create sum of covarainces..
        [x1,x2] = self.parse_args(*args)
        K = S.ones([x1.shape[0],x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            K     *=  covar.K(logtheta[self.covars_logtheta_I[nc]],*args)
        return K


    def Kd(self,modelparameters, *args):
        "derivative kernel"
        logtheta = modelparameters['covar']
        #1. check logtheta has correct length
        assert logtheta.shape[0]==self.n_hyperparameters, 'K: logtheta has wrong shape'
        [x1,x2] = self.parse_args(*args)
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
