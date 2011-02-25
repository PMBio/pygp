"""
Gaussian Process Package
========================

Holds all Gaussian Process classes, which hold all informations for a Gaussian Process to work porperly.

.. class **GP**: basic class for GP regression:
   * claculation of log marginal likelihood
   * prediction
   * data rescaling
   * transformation into log space

   
"""
import pkgutil
__all__ = ['composite','gpcEP','gpEP']

try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
    del pkg_resources
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
    del pkgutil

from .basic_gp import GP