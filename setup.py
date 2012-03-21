#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

__description__ = """Python package for Gaussian process regression in python 

========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions"""

setup(name='pygp',
      version = '1.0.0',
      description = __description__,
      author = "Oliver Stegle, Max Zwiessele, Nicolo Fusi",
      author_email='EMAIL HERE',
      url='https://github.com/PMBio/pygp',
      packages = ['pygp', 'pygp.covar', 'pygp.gp', 'pygp.likelihood', 'pygp.linalg',
      		  'pygp.optimize', 'pygp.plot', 'pygp.priors', 'pygp.demo', 'pygp.doc'],
      package_dir = {'pygp': 'pygp'},
      package_data = {'pygp.doc': ['html/*.html',
				   'html/_static/*',
				   'html/_sources/*',
				   'html/_images/*'],
		      '' : ['*.txt']},
      install_requires = ['numpy','scipy'],
      include_package_data = True,
      cmdclass = {'build_sphinx': BuildDoc},
      license = 'GPLv2',
      )
