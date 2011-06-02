import scipy as SP
from pygp.covar import CovarianceFunction



class MuCF(CovarianceFunction):
    """isotropic mean parameter which is integrated out
    """

    def __init__(self,*args,**kw_args):
        super(CovarianceFunction, self).__init__(**kw_args)
        self.n_hyperparameters = 1

        
    def K(self,theta,x1,x2=None):
        #get input data:
        # 2. exponentiate params:
        A  = SP.exp(2*theta[0])
        n = x1.shape[0]
        RV = A*SP.ones([n,n])
        return RV

    def Kgrad_theta(self,theta,x1,i):
        RV = self.K(theta,x1)
        #derivative w.r.t. to amplitude
        RV*=2
        return RV


    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    
    def Kgrad_xdiag(self,theta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = SP.zeros([x1.shape[0]])
        return RV
