"""
Gaussian process plotting tools
===============================

Tools to plot gpr output.
"""

# import python / numpy:
import pylab as PL
import scipy as S

def plot_training_data(x,y,args={'alpha':1,
                                 'color':'r',
                                 'linestyle':'circles',
                                 'markersize':28}):
    PL.plot(x,y,**args)

def plot_sausage(X,mean,std,format_fill={'alpha':0.2,'facecolor':'k'},format_line={'alpha':1, 'color':'g'}):
    """plot saussage plot of GP

    **Parameters:**

    X : [double]
        Interval X for which the saussage shall be plottet.
        
    mean : [double]
        The mean of to be plottet.
        
    std : [double]
        Pointwise standard deviation.

    """
    Xp = S.concatenate((X,X[::-1]))
    Yp = S.concatenate(((mean+2*std),(mean-2*std)[::-1]))
    hf=PL.fill(Xp,Yp,**format_fill)
    hp=PL.plot(X,mean,**format_line)
    return hp
    
