"""
Modulde for composite Gaussian processes models that bombine multiple GPs into one model


class **GroupGP**:
    group multiple GP objects for joint optimisation of hyperparameters


"""

from gpr import *


class GroupGP(object):
    __slots__ = ["N","GPs"]

    """
    Class to bundle one or more GPs for joint
    optimization of hyperparameters.

    **Parameters:**

    GPs : [:py:func:`GP.lMl`]
        Array, holding al GP classes to be optimized together
    """

    def __init__(self,GPs=None):
        if GPs is None:
            print "you need to specify a list of Gaussian Processes to bundle"
            return None
        self.N = len(GPs)
        self.GPs = GPs
        

        

    def lMl(self,modelparameters,**lml_kwargs):
        """
        Returns the log Marginal likelyhood for the given logtheta
        and the lMl_kwargs:

        logtheta : [double]
            Array of hyperparameters, which define the covariance function

        lMl_kwargs : lml, dlml, clml, sdlml, priors, Ifilter
            See :py:class:`gpr.GP.lMl`
    
        """
        #lMl(logtehtas,lml=Ture,dlml=true,clml=True,cdlml=True,priors=None)
        #just call them all and add up:
        R = []
        #calculate them for all N
        R = 0
        for n in range(self.N):
            L = self.GPs[n].lMl(modelparameters,**lml_kwargs)
            R = R+ L
        return R

    def setData(self,x,t):
        """
        set inputs x and targets t with **Parameters:**

        x : [double]
            trainging input

        t : [double]
            training targets
            
        rescale_dim : int
            dimensions to be rescaled (default all real)

        process : boolean
            subtract mean and rescale inputs

        """
        for n in range(self.N):
            xn = x[n]
            tn = t[n]
            self.GPs[n].setData(xn,tn)
            
        
