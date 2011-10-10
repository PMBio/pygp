""" cheeck for covarince matrices"""
import scipy as SP
import pdb
import pylab

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
        #dKx_diag = K.Kgrad_xdiag(logtheta,x1,d)
        #dKx.flat[::(dKx.shape[1] + 1)] = dKx_diag
        for iin in xrange(n):
            dKxn = SP.zeros([n, n])
            dKxn[iin, :] = dKx[iin, :]
            dKxn[:, iin] += dKx[iin, :]
            ana[iin,iid,:,:] = dKxn
            
    delta = (ana -diff)/(diff+1E-10)
    print "delta %.2f" % SP.absolute(delta).max()
    for d in xrange(nd):
        pylab.close("all")
        pylab.figure()
        pylab.pcolor(ana[:,d,:,:].sum(0))
        pylab.title("analytical")
        pylab.colorbar()
        
        pylab.figure()
        pylab.pcolor(diff[:,d,:,:].sum(0))
        pylab.title("numerical")
        pylab.colorbar()
        import pdb;pdb.set_trace()
    pass
