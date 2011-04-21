"""
Class with a collection of likelihood functions for EP
- as executation time of this class is an issue we expect the hyperparameters to be set upfront
"""

from numpy import *
import scipy as S
import scipy.stats as stats
import scipy.special as special

def n2mode(x):
    """convert from natural parameter to mode and back"""
    return array([x[0]/x[1],1/x[1]])

def sigmoid(x):
    """sigmoid function int_-inf^+inf Normal(x,1)"""
    return ( 1+special.erf(x/S.sqrt(2.0)) )/2.0
def gos(x):
    """Gaussian over sigmoid"""
    return ( sqrt(2.0/S.pi)*S.exp(-0.5*x**2)/(1+special.erf(x/S.sqrt(2.0))) )



class ALikelihood(object):
    __slots__=["logtheta"]
    #logtheta:local copy of likelihood hyperparamters
    """Abstract baseclass for likelihood classes"""

    def get_number_of_parameters(self):
        return 0

    def setLogtheta(self,logthetaL):
        assert logthetaL.shape[0]==self.get_number_of_parameters(), "hyperparameters have wrong shape"
        self.logtheta = logthetaL

    def calcExpectations(self,t,cav_np,x=None):
        """calculate expectation values for EP updates
        t: target
        cav_np: cavitiy distribution (natural parameters)
        x: optional: input
        """
        return None


class ProbitLikelihood(ALikelihood):
    """Probit likelihood for GP classification"""

    def get_number_of_parameters(self):
        return 0

    def calcExpectations(self,t,cav_np,x=None):
        """calc expectation values (moments) for EP udpates
        t: the target
        cav_np: (nu,tau) of cavity (natural params)
        x: (optional) input (not used in this likelihood)
        """


        #the derivation here follows Rasmussen & Williams
        zi      = t*cav_np[0]/S.sqrt(cav_np[1]*(1+cav_np[1]))
        #0. moment
        Z       = sigmoid(zi)
        #1. moment
        Fu      = cav_np[0]/cav_np[1] + t*gos(zi)/S.sqrt(cav_np[1]*(1+cav_np[1]))
        Fs2     = (1-gos(zi)*(zi+gos(zi))/(1+cav_np[1]))/cav_np[1]
        
        return S.array([Fu,Fs2,Z])
        pass


class GaussLikelihood(ALikelihood):
    __slots__ = ["sl"]
    #sl: variance
    """GaussianLikelihood
    - a dummy Class for debg purposes; in fact the likelihood is equivalent to a standard GP and hence should also yield
    identical results"""

    def get_number_of_parameters(self):
        return 1

    def setLogtheta(self,logthetaL):
        ALikelihood.setLogtheta(self,logthetaL)
        #exp. variance parameter upfront
        self.sl = exp(2*logthetaL[0])

    def calcExpectations(self,t,cav_np,x=None):
        """calculate expectation values for EP updates
        t: the target
        cav_np: (nu,tau) of cavity (natural params)
        logthetaL: log hyperparameter for likelihood(std. dev)"""

        cav_mp = n2mode(cav_np)

        # Z = int_{f} cav(f|cm,cv) N(f|y,sl)
        # Z = N(y|0+cm,cv+sl)
        stot = self.sl + cav_mp[1]
        Z  = stats.norm(cav_mp[0],S.sqrt(stot)).pdf(t)

        #calc. product
        C = (1.0/self.sl + 1.0/cav_mp[1])**(-1)

        #1. moment
        # Fu = 1/Z * int_{fi} fi * cav(f|cm,cv) * N(f|y,sl)
        Fu = C*(cav_mp[0]/cav_mp[1] + t/self.sl)

        #2. variance (2. moment-Fu**2)
        Fs2 = C
        return S.array([Fu,Fs2,Z])
        


class MOGLikelihood(ALikelihood):
    """Mixture of Gaussian likelihood
    robust likelihood with nm mixture components"""
    __slots__ = ["Ncomponents","phi","sl"]
    # phi: mixing ratio
    # sl:  variances


    def __init__(self,Ncomponents=2):
        """MOGLikelihood(Ncomponents=2)
        - mog likelihood with Ncomponents mixture components"""
        self.Ncomponents = Ncomponents

    def get_number_of_parameters(self):
        """get_number_of_parameters; for every mixture components 2 parametrs"""
        return self.Ncomponents*2

    def setLogtheta(self,logthetaL):
        """set hyperparameter of likelihood :
        logthetaL=log([pi_1,...pi_n,Var_1,..Var_n]
        where pi_i are mixing ratios (sum_pi = 1); and Var_i are the corresponding
        variances of the mixing components"""
        #set likelihood which asserts shape
        ALikelihood.setLogtheta(self,logthetaL)
        #exp. variance parameter upfront
        np_2 = self.get_number_of_parameters()/2
        #split in phi/sl
        self.phi = exp(logthetaL[0:np_2])
        self.sl = exp(2*logthetaL[np_2::])

    def calcExpectations(self,t,cav_np,x=None):
        """calculate expectation values for EP updates
        t: the target
        cav_np: (nu,tau) of cavity (natural params)
        x: (optional) input (not used in this likelihood)
        """

        cav_mp = n2mode(cav_np)
        ##this is a on2one copy from matlab code.. maybe it does not work :-(
        Z  = S.zeros([3])
        dc = S.zeros([3])
        for c in range(self.Ncomponents):
            #total variance for this mixture component
            Sc = self.sl[c]+cav_mp[1]
            #0. moment
            dc[0] = self.phi[c]*stats.norm(cav_mp[0],S.sqrt(Sc)).pdf(t)
            dc[1] = dc[0]*(t-cav_mp[0])*(1.0/Sc)
            dc[2] = dc[0]*( ((t-cav_mp[0])*(1.0/Sc))**2  - (1.0/Sc))
            Z+=dc

        alpha = 1.0/Z[0]*Z[1]
        nu    = - (1.0/Z[0]*Z[2] - alpha**2)

        Fu = cav_mp[1]*alpha+cav_mp[0]
        Fs2  = (1-cav_mp[1]*nu)*cav_mp[1]
        Z    = Z[0]
        return S.array([Fu,Fs2,Z])



#TODO: think about a general switching class managing between alternative likelihoods...
class ConstrainedLikelihood(ALikelihood):
    """ConstrainedLikelihood:
     - likelihood which allows datapoitns to be constraints rather than full
     datum
     - this likelihood allows to include an alternative ''default'' likelihood, for instance a standard
     Gaussian for non-constrained data entries
     """
    __slots__ = ["alt","index"]
    #alt: likelihood function to use for ''standard'' points
    #index: index which is used to determint the type of datapoint(default -1)

    def __init__(self,alt=GaussLikelihood(),index=-1):
        """ConstarinedLikelihood
        - alt specifies the likelihood function which is used for ''normal'' datums"""
        assert isinstance(alt,ALikelihood), "alt likelihood needs to be of type ALikelihood"
        self.alt = alt
        self.index = index

    def get_number_of_parameters(self):
        """get_number_of_parameters() returns the number of parameters of the alt likelihood"""
        return self.alt.get_number_of_parameters()

    def setLogtheta(self,logthetaL):
        """set hyperparameter of likelihood :
        - sets hyperparemter in standard likelihood"""
        self.alt.setLogtheta(logthetaL)

    def calcExpectations(self,t,cav_np,x=None):
        """calculate expectation values for EP updates
        t: the target
        cav_np: natural parameter of cavity distribution (nu,tau)
        x: (optional) input (not used in this likelihood)
        """
        cav_mp = n2mode(cav_np)
        
        Fu = 0
        Fs2 = 0.1
        Z = 0

        ##TODO: this does not really work yet

        #cavity gaussian
        mu    = cav_mp[0]
        sigma2= cav_mp[1]
        sigma = S.sqrt(sigma2)

        cav = stats.norm(mu,sigma)
        norm = stats.norm()
        
        #1. check whether datum is handeled by alt or this likelihood:
        if x[self.index]==0:
            return self.alt.calcExpectations(t,cav_mp,x)
        #this is from wikipedia:
        #(http://en.wikipedia.org/wiki/Truncated_normal_distribution)        
        elif x[self.index]<0:
            Z = cav.cdf(t)
            alpha = (t-mu)/sigma
            #make sure CDF is never 1
            calpha = min(norm.cdf(alpha),1-1E-10)
            lmb   = norm.pdf(alpha)/(1-calpha)
            delta = lmb*(lmb-alpha)
            Fu = mu - sigma*lmb
            Fs2= sigma2*(1-delta)
            pass
        elif x[self.index]>0:
            Z = cav.cdf(t+2*(mu-t))
            alpha = (t-mu)/sigma
            #make sure CDF is never 1
            calpha = min(norm.cdf(alpha),1-1E-10)
            lmb   = norm.pdf(alpha)/(1-calpha)
            delta = lmb*(lmb-alpha)
            Fu = mu + sigma*lmb
            Fs2= sigma2*(1-delta)

        print [mu,sigma,t]
        print [Fu,S.sqrt(Fs2)]
        return S.array([Fu,Fs2,Z])
        

        
        
