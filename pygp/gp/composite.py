"""
Grouping GP regression classes
==============================

Module for composite Gaussian processes models that combine multiple GPs into one model
"""

from pygp.gp import GP
import scipy as SP

class GroupGP(GP):
    """
    Class to bundle one or more GPs for joint
    optimization of hyperparameters.

    **Parameters:**

    GPs : [:py:class:`gpr.GP`]
        Array, holding al GP classes to be optimized together
    """
    __slots__ = ["N","GPs"]


    def __init__(self,GPs=None):
        # create a prototype of the parameter dictionary
        # additional fields will follow
        self._invalidate_cache()
        if GPs is None:
            print "you need to specify a list of Gaussian Processes to bundle"
            return None
        self.N = len(GPs)
        self.GPs = GPs

    def LML(self,hyperparams,**LML_kwargs):
        """
        Returns the log Marginal likelyhood for the given logtheta
        and the LML_kwargs:

        logtheta : [double]
            Array of hyperparameters, which define the covariance function

        LML_kwargs : lml, dlml, clml, sdlml, priors, Ifilter
            See :py:class:`gpr.GP.LML`
    
        """
        R = 0
        for n in range(self.N):
            L = self.GPs[n].LML(hyperparams,**LML_kwargs)
            R = R+L
        return R

    def LMLgrad(self,hyperparams,**lml_kwargs):
        """
        Returns the log Marginal likelihood for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ...}
            The hyperparameters which shall be optimized and derived

        priors : [:py:class:`pygp.priors`]
            The hyperparameters which shall be optimized and derived

        """
        #just call them all and add up:
        R = 0
        #calculate them for all N
        for n in range(self.N):
            L = self.GPs[n].LMLgrad(hyperparams,**lml_kwargs)
            for key in L.keys():
                R += (L[key])
        return {'covar':R}

    def setData(self,x,y):
        """
        set inputs x and outputs y with **Parameters:**

        x : [double]
            training input

        y : [double]
            training targets
            
        rescale_dim : int
            dimensions to be rescaled (default all real)

        process : boolean
            subtract mean and rescale inputs

        """
        for n in range(self.N):
            xn = x[n]
            yn = y[n]
            self.GPs[n].setData(xn,yn)

    def getData(self):
        data = []
        for n in range(self.N):
            data.append(self.GPs[n].getData())
        return data
            
    def predict(self,*args,**kwargs):
        '''
        Predict mean and variance for each GP and given Parameters.
        
        **Parameters:**
        
        hyperparams : {}
            hyperparameters in logSpace.
        xstar    : [double]
            prediction inputs.
        var      : boolean
            return predicted variance.
        output   : int
            output dimension for prediction (0)

        **Return:**
        Array as follows::
    
            [[1st_predictions_mean, 2nd, ..., nth_predictions_mean],
             [1st_predictions_var, 2nd, ..., nth_predictions_var]]
            
        See :py:class:`pygp.gp.basic_gp.GP` for individual prediction outputs.
        '''
        means = []
        var = []
        for n in range(self.N):
            prediction = self.GPs[n].predict(*args,**kwargs)
            means.append(prediction[0])
            var.append(prediction[1])
            
        return SP.array([means,var])
