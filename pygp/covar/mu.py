import scipy as SP
from pygp.covar import CovarianceFunction
from pygp.covar.fixed import FixedCF



class MuCF(FixedCF):
    """isotropic mean parameter which is integrated out
    """

    def __init__(self, N, **kw_args):
	"""
	Constructor for the MuCF covariance function. It inherits everything from
	FixedCF, as it is a special case of FixedCF with a matrix of ones.
	
	Input:
	N = number of samples
	"""
	
	super(MuCF, self).__init__(SP.ones((N,N)), **kw_args)
