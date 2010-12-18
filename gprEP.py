"""
Class for Gaussian Process Regression with arbitrary likelihoods
commonly we will use EP to obtain a Gaussian approximation to the likelihood function
"""

from gpr import *
#TMP
from numpy import *
from numpy.linalg.linalg import *
import scipy.stats 


class GPEP(GP):
    __slots__ = ["likelihood","muEP","zEP","vEP","Nep","IlogthetaL"]
    #zEP are stored 0th moments as an additional outcome of the EP calculations
    #muEP: Ep site parameter mean
    #vEP:  EP site parameter variance
    #Nep: number of Ep iterations
    #IlogthetaL: index vector for likelihood kernel parameters

    """Gaussian Process class with an arbitrary likelihood (likelihood) which will be approximiated
    using an EP approximation"""

    #def __init__(self,*argin,likelihood=None,**kwargin):
    def __init__(self,likelihood=None,Nep=3,*argin,**kwargin):
        #call constructor from GP module
        GP.__init__(self,*argin,**kwargin)
        #initialize a few extra parmeters
        self.likelihood=likelihood
        #number of EP sweeps
        self.Nep = Nep
        #split logtheta in likelihood and kernel logtheta
        Nk = self.covar.getNparams()
        Nl = self.likelihood.getNparams()
        Nt = Nk+Nl
        self.Nlogtheta = Nt
        self.IlogthetaK = zeros([Nt],dtype='bool')
        self.IlogthetaL = zeros([Nt],dtype='bool')
        self.IlogthetaK[0:Nk] = True
        self.IlogthetaL[Nk:Nk+Nl] = True
        

    def updateEP(self,K,logthetaL=None):
        """update a kernel matrix K using Ep approximation
        [K,t,C0] = updateEP(K,logthetaL)
        logthetaL: likelihood hyperparameters
        t: new means of training targets
        K: new effective kernel matrix
        C0:0th moments
        """
        assert K.shape[0]==K.shape[1], "Kernel matrix must be square"
        assert K.shape[0]==self.n,     "Kernel matrix has wrong dimension"
        #approximate site parmeters; 3 moments
        # note g is in natural parameter representation (1,2)
        g = zeros([self.n,2])
        # a copy for damping
        g2= zeros([self.n,2])
        # 0. moment is just catptured in z 
        z = zeros([self.n])
        # damping factors
        damp = ones([self.n])
        #approx is
        #p(f) = N(f|mu,Sigma)
        # where Sigma = (K^{-1} + PI^{-1})^{-1}; PI is created from teh diaginal
        # entries in g; PI = diag(Var(g))
        # mu = Sigma*PI^{-1}Mean(g)
        # where \mu is form the site parameters in g also

        #add some gitter to make it invertible
        K+=eye(K.shape[0])*1E-6
        #initialize current approx. of full covariance
        Sigma = K.copy()
        #invert Kernel matrix; which is used later on
        #TODO: replace by chol
        KI    = linalg.inv(K)
        #current approx. mean
        mu    = zeros([self.n])
        
        #conversion nat. parameter/moment representation
        n2mode = lambda x: array([x[0]/x[1],1/x[1]])
        #set hyperparameter of likelihood object
        self.likelihood.setLogtheta(logthetaL)

        for nep in range(self.Nep):
            #get order of site function update
            perm = random.permutation(self.n)
            perm = arange(self.n)
            for ni in perm:
                #cavity as natural parameter representation
                cav_np = n2mode([mu[ni],Sigma[ni,ni]])-g[ni]
                #ensure we don't have negative variances. good idea?
                cav_np[1] = abs(cav_np[1])
                #calculate expectation values (int_, int_y,int_y^2)
                ML = self.likelihood.calcExpectations(self.t[ni],cav_np,x=self.x[ni])
                #1. and 2. moment can be back-calculated to enw site parameters
                #update the site parameters;
                #in natural parameters this is just deviding out the site function; v. convenient
                gn = n2mode(ML[0:2])-cav_np
                #delta gn in nat. parameters
                dg = gn-g[ni]
                #difference of second moment (old-new)
                ds2 = gn[1]-g[ni,1]
                #update with damping factor damp[ni]
                g[ni] = g[ni]+damp[ni]*dg
                if(g[ni,1]<0):
                    g[ni,1] = 1E-10
                z[ni] = ML[2]
                if 1:
                    #rank one updates
                    Sigma2= Sigma
                    Sigma = Sigma - ds2/(1+ds2*Sigma[ni,ni])*outer(Sigma[:,ni],Sigma[ni,:])
                    if 1:
                        #check that Sigma is still pos. definite, otherweise we need to to do some damping...
                        try:
                            Csigma = linalg.cholesky(Sigma)
                        #except linalg.linalg.LinAlgError:
                        except LinAlgError:
                            LG.debug('damping')
                            Sigma=Sigma2
                            g[ni] = g2[ni]
                            #increase damping factor
                            damp[ni]*=0.9
                            pass
                    #update mu; mu[i] = Sigma[i,i]*(1/Var(g[i]))*Mean(g[i])
                    #as go is in nat. parameter this is always like this
                    mu    = dot(Sigma,g[:,0])
                else:
                    #slow updates
                    Sigma = linalg.inv(KI + diag(g[:,1]));
                    mu    = dot(Sigma,g[:,0])
                pass
            #after every sweep recalculate entire covariance structure
            [Sigma,mu,lml] = self.epComputeParams(K,KI,g)
            
            #create a copy for damping
            g2 = g.copy()
            pass
            
        if nep == (self.Nep-1):
            #LG.warn('maximum number of EP iterations reached')
            pass
        #update site parameters
        self.muEP = g[:,0]/g[:,1]
        self.vEP  = 1/g[:,1]
                

    def epComputeParams(self,K,KI,g):
        """calculate the ep Parameters
        K: plain kernel matrix
        g: [0,1]: natural parameter rep. [2]: 0. moment for lml
        """
        #inverse of EP kernel matrix
        KepI  = diag(g[:,1])
        Sigma = linalg.inv(KI + KepI)
        #however g[:,0] = mu/var... so that's all we need
        mu    = dot(Sigma,g[:,0])
        #TODO: implement lml
        lml   = 0
        return [Sigma,mu,lml]
        
    def getCovariances(self,logtheta):
        """[L,Alpha] = getCovariances()
        - special overwritten version of getCovariance (gpr.py)
        - here: EP updates are employed"""


        if (logtheta==self.logtheta).all() and (self.cached_L is not None):
            return [self.cached_L,self.cached_alpha]

        #1. copy logtheta
        self.logtheta = logtheta.copy()
        
        assert (self.Nlogtheta)==logtheta.shape[0], "incorrect shape of kernel parameter matrix"

        #2. vanilla Kernel matrix
        K = self.covar.K(logtheta[self.IlogthetaK],self.x)

        #3. run EP updates
        #EP effectively creates a new Kernel matrix (with input dependent noise) and new effective training means
        #in addition we store a 0th moment which is used for the lMl calculation
        self.updateEP(K,logtheta[self.IlogthetaL])
        #updateEP computes the site parameters which we use here to calcualte the full covarince for test predictions
        Keff = (K + diag(self.vEP))

        self.cached_L = linalg.cholesky(Keff)
        self.cached_alpha = solve_chol(self.cached_L.transpose(),self.muEP)
        return [self.cached_L,self.cached_alpha]



