"""
Gaussian process plotting tools
===============================

Tools to plot gpr output.
"""

# import python / numpy:
import pylab as PL
import scipy as S

def plot_training_data(x,y,
                       format_data={'alpha':1,
                                    'color':'r',
                                    'marker':'.',
                                    'linestyle':'',
                                    'markersize':12}):
    PL.plot(S.array(x).reshape(-1),
            S.array(y).reshape(-1),**format_data)

def plot_training_data_with_shiftx(x,y,
                               shift=[],
                               replicate_indices=[],
                               format_data={'alpha':1,
                                    'color':'r',
                                    'marker':'.',
                                    'linestyle':'',
                                    'markersize':12}):
    x = S.array(x).reshape(-1)
    y = S.array(y).reshape(-1)

    if shift is None:
        shift=[]
        replicate_indices=[]

    assert len(shift)==len(S.unique(replicate_indices)), 'We need one shift per replicate to plot properly'

    _format_data = format_data.copy()
    _format_data['alpha'] = .2
    PL.plot(x,y,**_format_data)

    x_shift = x.copy()

    for i in S.unique(replicate_indices):
        x_shift[replicate_indices==i] -= shift[i]

    for i in xrange(len(x)):
        PL.annotate("",xy=(x_shift[i],y[i]),
                    xytext=(x[i],y[i]),
                    arrowprops=dict(facecolor=format_data['color'], alpha=.3,shrink=.2,frac=.3))


    PL.plot(x_shift,y,
            **format_data)

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
    
