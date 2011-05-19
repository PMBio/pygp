"""
Covariance Functions
====================

We implemented several different covariance functions (CFs) you can work with. To use GP-regression with these covariance functions it is highly recommended to model the noise of the data in one extra covariance function (:py:class:`pygp.covar.noise.NoiseISOCF`) and add this noise CF to the CF you are calculating by putting them all together in one :py:class:`pygp.covar.combinators.SumCF`.

For example to use the squared exponential CF with noise::

    from pygp.covar import se, noise, combinators
    
    #Feature dimension of the covariance: 
    dimensions = 1
    
    SECF = se.SEARDCF(dim)
    noise = noise.NoiseISOCF()
    covariance = combinators.SumCF((SECF,noise))

"""


try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


#import covar_base
from covar_base import *
