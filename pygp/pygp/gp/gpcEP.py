"""
Class for Gaussian process classification using EP
==================================================

"""

import scipy as S
import pygp.likelihood.EPLikelihood as EPL
import pygp.gp.gpEP.GPEP as GPEP

class GPCEP(GPEP):
    __slots__ = []


    def __init__(self,*argin,**kwargin):
        likelihood = EPL.ProbitLikelihood()
        #call constructor, pass on likelihood function and switch off subtraction of mean
        super(GPCEP, self).__init__(Smean=False,likelihood=likelihood,*argin,**kwargin)
        self.Nep = 3
        

        
        

    def predict(self,*argin,**kwargin):
        """Binary classification prediction"""

        #1. get Gaussian prediction
        [MU,S2] = GPEP.predict(self,*argin,**kwargin)
        #2. push thorugh sigmoid
        #predictive distribution is int_-inf^+inf normal(f|mu,s2)sigmoid(f)
        Pt = EPL.sigmoid ( MU / S.sqrt(1+S2))
        return [Pt,MU,S2]
        pass
    


    def setData(self,x,t,**kwargin):
        """set Data
        x: inputs [N,D]
        t: targets [N]
        - targets are either -1,+1 or False/True
        """
        assert isinstance(t,S.ndarray), 'setData requires numpy arrays'
        #check whether t is bool
        if t.dtype=='bool':
            t_ = S.ones([t.shape[0]])
            t_[t] = +1
            t_[~t] = -1
            t = t_
        GPEP.setData(self,x,t,**kwargin)
