.. GPTwoSample documentation master file, created by
   sphinx-quickstart on Fri Oct 29 11:59:50 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyGP
===============

This package provides Gaussian Process based regression. 

The different GP Regression classes provide following computations:

   - :py:class:`pygp.gpr` Basic gp regression package
   - :py:class:`pygp.gpr_ep` GP regression with EP likelihood models. (Not yet implemented)


Contents:

.. toctree:: 
   :maxdepth: 3

   Gaussian Process Regression <gp>
   Gaussian Process Hyperparameter optimization <opt_hyper>
   Covariance Functions <covars>

     Example demonstration of gpr <demo_gpr>
     Example demonstration of gpr with input shift <demo_gpr_shiftx>   

..   Grouping GP regression classes <composite>
    Hyperprior Distributions <priors>
     Plotting gpr output <plot_gpr>
     Sampling from a GP <gp_sample>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

