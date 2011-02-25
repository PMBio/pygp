"""simple covariance function modelling a mean variable which is integrated out
K = \alpha^2 1
"""
import scipy as SP
import pdb

from pygp.covar import CovarianceFunction,CF_Kd_dx


class MuCF(CF_Kd_dx):
    """isotropic mean parameter which is integrated out
    """

    def __init__(self,*args,**kw_args):
        CF_Kd_dx.__init__(self,*args,**kw_args)
        self.n_hyperparameters = 1

        
    def K(self,logtheta,x1,x2=None):
        #get input data:
        # 2. exponentiate params:
        A  = SP.exp(2*logtheta[0])
        n = x1.shape[0]
        RV = A*SP.ones([n,n])
        return RV

    def Kd(self,logtheta,x1,i):
        RV = self.K(logtheta,x1)
        #derivative w.r.t. to amplitude - trivial
        RV*=2
        return RV

    def Kd_dx(self,logtheta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV
    
    def Kd_dx_diag(self,logtheta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = SP.zeros([x1.shape[0]])
        return RV
