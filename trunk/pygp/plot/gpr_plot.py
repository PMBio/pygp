"""
pygp plotting tools
===============================

Tools to plot Gaussian process :py:class:`pygp.gp` regression output.

"""

# import python / numpy:
import pylab as PL
import scipy as S
import matplotlib

def plot_training_data(x,y,
                       shift=None,
                       replicate_indices=None,
                       format_data={'alpha':1,
                                    'color':'r',
                                    'marker':'.',
                                    'linestyle':'',
                                    'markersize':12}):
    """
    Plot training data input x and output y into the
    active figure (See http://matplotlib.sourceforge.net/ for details of figure).

    Instance plot without replicate groups:
    
    .. image:: ../images/plotTraining.png
       :height: 8cm
       
    Instance plot with two replicate groups and a shift in x-koords:
    
    .. image:: ../images/plotTrainingShiftX.png
       :height: 8cm

    **Parameters:**

    x : [double]
        Input x (e.g. time).

    y : [double]
        Output y (e.g. expression).

    shift : [double]
        The shift of each replicate group.
        
    replicate_indices : [int]
        Indices of replicates for each x, rexpectively

    format_data : {format}
        Format of the data points. See http://matplotlib.sourceforge.net/ for details. 
        
    """
    x = S.array(x).reshape(-1)
    y = S.array(y).reshape(-1)

    x_shift = x.copy()

    if shift is not None and replicate_indices is not None:
        assert len(shift)==len(S.unique(replicate_indices)), 'Need one shift per replicate to plot properly'

        _format_data = format_data.copy()
        _format_data['alpha'] = .2
        PL.plot(x,y,**_format_data)

        for i in S.unique(replicate_indices):
            x_shift[replicate_indices==i] -= shift[i]

        for i in xrange(len(x)):
            PL.annotate("",xy=(x_shift[i],y[i]),
                        xytext=(x[i],y[i]),
                        arrowprops=dict(facecolor
                                        =format_data['color'],
                                        alpha=.3,
                                        shrink=.2,
                                        frac=.3))


    return PL.plot(x_shift,y,**format_data)

def plot_sausage(X,mean,std,format_fill={'alpha':0.2,'facecolor':'k'},format_line={'alpha':1, 'color':'g'}):
    """
    plot saussage plot of GP. I.e:

    .. image:: ../images/sausage.png
      :height: 8cm

    **returns:** : [fill_plot, line_plot]
        The fill and the line of the sausage plot. (i.e. green line and gray fill of the example above)
        
    **Parameters:**

    X : [double]
        Interval X for which the saussage shall be plottet.
        
    mean : [double]
        The mean of to be plottet.
        
    std : [double]
        Pointwise standard deviation.

    format_fill : {format}
        The format of the fill. See http://matplotlib.sourceforge.net/ for details.

    format_line : {format}
        The format of the mean line. See http://matplotlib.sourceforge.net/ for details.
        
    """
    Xp = S.concatenate((X,X[::-1]))
    Yp = S.concatenate(((mean+2*std),(mean-2*std)[::-1]))
    hf=PL.fill(Xp,Yp,**format_fill)
    hp=PL.plot(X,mean,**format_line)
    return [hf,hp]
    
class CrossRect(matplotlib.patches.Rectangle):
    def __init__(self, *args, **kwargs):
        matplotlib.patches.Rectangle.__init__(self, *args, **kwargs)
        
        #self.ax = ax

    # def get_verts(self):
    #     rectverts = matplotlib.patches.Rectangle.get_verts(self)
        
    #     return verts

    def get_path(self, *args, **kwargs):
        old_path = matplotlib.patches.Rectangle.get_path(self)
        verts = []
        codes = []
        for vert,code in old_path.iter_segments():
            verts.append(vert)
            codes.append(code)
        verts.append([1,1])
        codes.append(old_path.LINETO)
        new_path = matplotlib.artist.Path(verts,codes) 
        return new_path
