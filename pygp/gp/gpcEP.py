"""
Class for Gaussian process classification using EP
==================================================

"""

import scipy as S

from pygp.likelihood.EP import sigmoid

from pygp.gp.gprEP import GPEP
from pygp.likelihood.likelihood_base import GaussLikISO, ProbitLik

class GPCEP(GPEP):
    __slots__ = []


    def __init__(self,*argin,**kwargin):
        likelihood = ProbitLik()#GaussLikISO()#ProbitLikelihood()
        #call constructor, pass on likelihood function and switch off subtraction of mean
        super(GPCEP, self).__init__(likelihood=likelihood,*argin,**kwargin)#Smean=False,likelihood=likelihood,*argin,**kwargin)
        self.Nep = 3
        
        
        

    def predict(self,*argin,**kwargin):
        """Binary classification prediction"""

        #1. get Gaussian prediction
        [MU,S2] = GPEP.predict(self,*argin,**kwargin)
        #2. push thorugh sigmoid
        #predictive distribution is int_-inf^+inf normal(f|mu,s2)sigmoid(f)
        Pt = sigmoid ( MU / S.sqrt(1+S2))
        return [Pt,MU,S2]
        pass
    


    def setData(self,x,y,*args,**kwargin):
        """set Data
        x: inputs [N,D]
        t: targets [N]
        - targets are either -1,+1 or False/True
        """
        assert isinstance(y,S.ndarray), 'setData requires numpy arrays'
        #check whether t is bool
        if y.dtype=='bool':
            y_ = S.ones([y.shape[0]])
            y_[y] = +1
            y_[~y] = -1
            y = y_
        else:
            assert len(SP.unique(y))==2, 'need either binary inputs or inputs of length 2 for classification'
        GPEP.setData(self,x=x,y=y,*args,**kwargin)
