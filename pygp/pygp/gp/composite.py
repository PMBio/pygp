"""
Grouping GP regression classes
==============================

Module for composite Gaussian processes models that combine multiple GPs into one model
"""

from gpr import GP
import scipy as SP

class GroupGP(GP):
    __slots__ = ["N","GPs"]

    """
    Class to bundle one or more GPs for joint
    optimization of hyperparameters.

    **Parameters:**

    GPs : [:py:class:`gpr.GP`]
        Array, holding al GP classes to be optimized together
    """

    def __init__(self,GPs=None,*args,**kw_args):
        # create a prototype of the parameter dictionary
        # additional fields will follow
        self._invalidate_cache()
        if GPs is None:
            print "you need to specify a list of Gaussian Processes to bundle"
            return None
        self.N = len(GPs)
        self.GPs = GPs

    def lMl(self,hyperparams,**lml_kwargs):
        """
        Returns the log Marginal likelyhood for the given logtheta
        and the lMl_kwargs:

        logtheta : [double]
            Array of hyperparameters, which define the covariance function

        lMl_kwargs : lml, dlml, clml, sdlml, priors, Ifilter
            See :py:class:`gpr.GP.lMl`
    
        """
        #lMl(logtehtas,lml=Ture,dlml=true,clml=True,cdlml=True,priors=None)
        #calculate them for all N
        R = 0
        for n in range(self.N):
            L = self.GPs[n].lMl(hyperparams,**lml_kwargs)
            R = R+L
        return R

    def dlMl(self,hyperparams,priors=None,**lml_kwargs):
        """
        Returns the log Marginal likelihood for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ...}
            The hyperparameters which shall be optimized and derived

        priors : [:py:class:`lnpriors`]
            The hyperparameters which shall be optimized and derived

        """
        #just call them all and add up:
        R = 0
        #calculate them for all N
        for n in range(self.N):
            L = self.GPs[n].dlMl(hyperparams,**lml_kwargs)
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
        Predict mean and variance for each GP and given **Parameters:**

        hyperparams : {}
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        var      : boolean
            return predicted variance
            
        output   : output dimension for prediction (0)

        **returns:** [[1st_predictions_mean, 2nd, ..., nth_predictions_mean],
                      [1st_predictions_var, 2nd, ..., nth_predictions_var]]
            See :py:class:`gpr.GP` for individual prediction outputs.

        '''
        means = []
        var = []
        for n in range(self.N):
            means.append(self.GPs[n].predict(*args,**kwargs)[0])
            var.append(self.GPs[n].predict(*args,**kwargs)[1])
            
        return SP.array([means,var])
