""" cheeck for covarince matrices"""
import scipy as SP
import pdb
import pylab
from matplotlib.pyplot import get_current_fig_manager

relchange = 1E-5;


def grad_check_logtheta(K,logtheta,x0,dimensions=None):
    """perform grad check with respect to hyperparameters logtheta"""
    L=0;
    x1 = logtheta.copy()
    n = x1.shape[0]
    nx = x0.shape[0]
    diff = SP.zeros([n,nx,nx])
    for i in xrange(n):
        change = relchange*x1[i]
        change = max(change,1E-5)
        x1[i] = logtheta[i] + change
        Lplus = K.K(x1,x0,x0)
        x1[i] = logtheta[i] - change
        Lminus = K.K(x1,x0,x0)
        x1[i] = logtheta[i]
        diff[i,:,:] = (Lplus-Lminus)/(2.*change)
    #ana
    ana = SP.zeros([n,nx,nx])
    for iid in xrange(n):
        ana[iid,:,:] = K.Kgrad_theta(x1,x0,iid)
    delta = (ana -diff)/(diff+1E-10)
    print "delta %.2f" % SP.absolute(delta).max()
    pdb.set_trace()
    pass

def grad_check_Kx(K,logtheta,x0,dimensions=None):
    """perform grad check with respect to input x"""
    L=0;
    x1 = x0.copy()
    n = x1.shape[0]
    if dimensions is None:
        dimensions = SP.arange(x0.shape[1])
    nd = len(dimensions)
    diff = SP.zeros([n,nd,n,n])
    for i in xrange(n):
        for iid in xrange(nd):
            d = dimensions[iid] 
            change = relchange*x0[i,d]
            change = max(change,1E-5)
            x1[i,d] = x0[i,d] + change
            Lplus = K.K(logtheta,x1,x1)
            x1[i,d] = x0[i,d] - change
            Lminus = K.K(logtheta,x1,x1)
            x1[i,d] = x0[i,d]

            diff[i,iid,:,:] = (Lplus-Lminus)/(2.*change)
    #ana
    ana = SP.zeros([n,nd,n,n])
    ana2 = SP.zeros([n,nd,n,n])
    for iid in xrange(nd):
        d = dimensions[iid]
        dKx = K.Kgrad_x(logtheta,x1,x1,d)
        for iin in xrange(n):
            dKxn = SP.zeros([n, n])
            dKxn[iin, :] = 1.*dKx[iin, :]
            dKxn[:, iin] += 1.*dKx[iin, :]
            ana[iin,iid,:,:] = dKxn
            
    delta = ((ana -diff)**2).sum()
    print "delta %.2f" % SP.absolute(delta).max()
    pdb.set_trace()
    pass
