"""
Gaussian process likelihood models
========================
"""

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)

#import covar_base
from likelihood_base import *
