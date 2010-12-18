import combinators
import se
import unittest
import scipy as SP
import scipy.optimize.optimize as OPT
import sys
sys.path.append('../')
import gpr as GPR

class TestProductCF(unittest.TestCase):
    
    def setUp(self):
        self.SECF = se.SEARDCF(1)
        self.covar = combinators.ProductCF((self.SECF,self.SECF))
        self.logtheta = SP.log(SP.array([1,3,3,1]))
        x = SP.arange(0,10,1)
        self.x = x[:,SP.newaxis]
        self.y = SP.sin(self.x)
        self.gpr = GPR.GP(self.covar,self.x,self.y)

    def test_K(self):
        K_Prod = self.covar.K(self.logtheta,self.x)
        K_SECF1 = self.SECF.K(self.logtheta[:2], self.x)
        K_SECF2 = self.SECF.K(self.logtheta[2:], self.x)
        K_SECF = (K_SECF1 * K_SECF2)
        for i in range(K_Prod.shape[0]):
            for j in range(K_Prod.shape[1]):
                self.assertAlmostEqual(K_Prod[i,j], K_SECF[i,j],7)

    def test_Kd(self):
        Kd_Prod1 = self.covar.Kd(self.logtheta,self.x,0)
        Kd_Prod2 = self.covar.Kd(self.logtheta,self.x,1)
        Kd_Prod3 = self.covar.Kd(self.logtheta,self.x,2)
        Kd_Prod4 = self.covar.Kd(self.logtheta,self.x,3)

        K_SECF1 = self.SECF.K(self.logtheta[:2], self.x)
        K_SECF2 = self.SECF.K(self.logtheta[2:], self.x)

        Kd_SECF11 = self.SECF.Kd(self.logtheta[:2], self.x, 0)
        Kd_SECF12 = self.SECF.Kd(self.logtheta[:2], self.x, 1)

        Kd_SECF21 = self.SECF.Kd(self.logtheta[2:], self.x, 0)
        Kd_SECF22 = self.SECF.Kd(self.logtheta[2:], self.x, 1)

        Kd_SECF11 = (Kd_SECF11 * K_SECF2)
        Kd_SECF12 = (Kd_SECF12 * K_SECF2)
        Kd_SECF21 = (K_SECF1 * Kd_SECF21)
        Kd_SECF22 = (K_SECF1 * Kd_SECF22)

        for i in range(Kd_Prod1.shape[0]):
            for j in range(Kd_Prod1.shape[1]):
                self.assertAlmostEqual(Kd_Prod1[i,j], Kd_SECF11[i,j],4)
                self.assertAlmostEqual(Kd_Prod2[i,j], Kd_SECF12[i,j],4)
                self.assertAlmostEqual(Kd_Prod3[i,j], Kd_SECF21[i,j],4)
                self.assertAlmostEqual(Kd_Prod4[i,j], Kd_SECF22[i,j],4)

    def test_grad(self):
        hyperparams = dict(covar=self.logtheta)
        X0 = self.gpr._param_dict_to_list(hyperparams)
        Ifilter_x = SP.ones(len(X0),dtype='bool')

        def f(x):
            x_ = X0
            x_[Ifilter_x] = x
            rv =  self.gpr.lMl(x_)
            #LG.debug("L("+str(x_)+")=="+str(rv))
            if SP.isnan(rv):
                return 1E6
            return rv
        
        def df(x):
            x_ = X0
            x_[Ifilter_x] = x
            rv =  self.gpr.dlMl(x_)
            #convert to list
            rv = self.gpr._param_dict_to_list(rv)
            #LG.debug("dL("+str(x_)+")=="+str(rv))
            if SP.isnan(rv).any():
                In = isnan(rv)
                rv[In] = 1E6
            return rv[Ifilter_x]
    
        self.assertAlmostEqual(OPT.check_grad(f,df,self.logtheta),10**-4,3)

        
if __name__ == '__main__':
    unittest.main()
