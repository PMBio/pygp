"""simple class for a linear covariance function"""

import scipy as SP


from covar import CovarianceFunction


class LinearCovariance(CovarianceFunction):

    def __init__(self):
        self.n_params = 1


    def K(self,logtheta,*args):


        x1 =args[0]
        if(len(args)==1):
            x2 = x1
        else:
            x2 = args[1]

        A = SP.exp(logtheta[0])
            
        K = A*SP.dot(x1,x2)
        return K
