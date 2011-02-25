#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup


__description__ = """Python package for Gaussian process regression in python 

========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions"""

setup(name='pygp',
      #namespace_packages=['covar'],
      version = '0.7.0',
      description = __description__,
      #summary = __description__.split("\n")[0],
      #platform = "Linux/MaxOSX/Windows"
      #author = 'Oliver Stegle, Max Zwießele',
      #author_email = 'email_not_yet@support.ed',
      #url = 'no.url.given'
      install_requires = ['numpy','scipy']
      )
