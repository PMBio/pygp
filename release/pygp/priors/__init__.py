"""
Package for Priors of Gaussian Processes
========================================

This package provides priors for gaussian processes in which
you can declare your prior beliefs of the hyperparameter
distribution.

"""

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
