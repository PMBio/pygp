""" MCMC samling module for GPR
"""
import scipy as SP
import logging as LG

def sampleHyper(gpr,hyperparams,Ifilter=None,priors=None,Nsamples=100,eps=1E-2,Nleap=10):
    """
    sample from the posterior distribution of
    GP hyperparmeters (Hyrbid Monte Carlo)
    """
    def fE(x):
        _logtheta[Ifilter] = x
        rv = gpr.lMl(logtheta=_logtheta,lml=True,dlml=False,priors=priors)
        return rv
    def fdE(x):
        _logtheta[Ifilter] = x
        rv = gpr.lMl(logtheta=_logtheta,lml=False,dlml=True,priors=priors)
        return rv[Ifilter]

    logtheta = hyperparams['covar']

    if Ifilter is None:
        Ifilter = SP.ones_like(logtheta)
    #convert I_filter to bools
    Ifilter = Ifilter==1
    _logtheta = logtheta.copy()
    #initialize HMC
    x = logtheta[Ifilter]
    
    g = fdE(x)
    E = fE(x)

    Rtau = xrange(Nleap)                    #leapfrog steps
    #samples
    X    = SP.zeros([Nsamples,logtheta.shape[0]])
    #initialize with logtheta due to filtering
    X[:,:] = logtheta
    naccept = 0
    try:
        for ns in xrange(Nsamples):
            p = SP.random.standard_normal(x.shape)
            H = 0.5*SP.dot(p,p) + E

            xnew = x; gnew = g
            #leapfrogs
            i = len(Rtau)
            while(i>0):
                p-= 0.5*eps*gnew
                xnew+=eps*p
                gnew = fdE(xnew)
                p-=0.5*eps*gnew
                i-=1
            Enew = fE(xnew)
            Hnew = 0.5*SP.dot(p,p) + Enew
            dH   = Hnew-H
            if (dH<0):
                accept = 1
            elif SP.rand()<SP.exp(-dH):
                accept = 1
                print "AA:"+str(dH)
                pass
            else:
                accept = 0

            if(accept):
                LG.debug("accept: %d" % (ns)+str(SP.exp(xnew))+ "E()="+str(Enew))
                g=gnew; x=xnew; E=Enew
                naccept+=1
            else:
                LG.debug("reject: %d" % (ns)+str(SP.exp(xnew))+ "E()="+str(Enew))
            #store sample
            X[ns,Ifilter] = xnew
    except KeyboardInterrupt:
        import sys
        sys.stderr.write('Keyboard interrupt')
        sys.stderr.write('returning what we have so far')
    LG.info("samples %d samples, accept/reject = %.2f" % (Nsamples,SP.double(naccept)/Nsamples))
    return X
