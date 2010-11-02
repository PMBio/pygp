"""
Gaussian process plotting tools
"""

# import python / numpy:
import pylab as PL
import scipy as S


def plot_sausage(X,mean,std,format_fill={'alpha':0.1,'facecolor':'k'},format_line={}):
    """plot saussage plot of GP
    X: inputs
    mean: mean
    std: standard deviation"""

    Xp = S.concatenate((X,X[::-1]))
    Yp = S.concatenate(((mean+2*std),(mean-2*std)[::-1]))
    hf=PL.fill(Xp,Yp,**format_fill)
    hp=PL.plot(X,mean,**format_line)
    return hp
    
