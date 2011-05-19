"""
Linear algebra tools needed for GP models
"""

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


#Default: import linalg_base
from linalg_matrix import *
