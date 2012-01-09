"""
Squared Exponential CF with breakpoint detection
================================================

Detects breakpoint T where two timeseries diverge.
"""

import scipy as SP

from pygp.covar import CovarianceFunction

import logging as LG
from scipy import special

class DivergeCF(CovarianceFunction):

    """
    Squared Exponential Covariance function, detecting breakpoint
    where two timeseries diverge.

    **Parameters:**
    
    replicate_indices : [int]

        The indices of the respective replicates, corresponding to
        the inputs. For instance: An input with three replicates:

        ===================== ========= ========= =========
        /                     rep1      rep2      rep3
        ===================== ========= ========= =========
        input = [             -1,0,1,2, -1,0,1,2, -1,0,1,2]
        replicate_indices = [ 0,0,0,0,  1,1,1,1,  2,2,2,2]
        ===================== ========= ========= =========

            
        Thus, the replicate indices represent
        which inputs correspond to which replicate.

    group_indices : [bool]

        Indices of group of each x. Thus this array depicts the
        group which each x belongs to. Only for two groups
        implemented yet!
        
    """

    def __init__(self,*args,**kw_args):
        super(DivergeCF, self).__init__(*args,**kw_args)
        #2. get number of params
        # self.replicate_indices = replicate_indices
        #self.group_indices = group_indices
        #self.n_replicates = len(SP.unique(replicate_indices))
        self.n_hyperparameters = 2
        assert self.n_hyperparameters == 2, "Not implemented yet for %i groups" % (self.n_hyperparameters)
        
    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = ['Breakpoint', "Breakpoint Length-Scale"]
        return names
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.n_hyperparameters;

    def K(self, logtheta, x1, x2=None, k=10):
        """
        Get Covariance matrix K with given hyperparameter BP
        and inputs X=x1 and X\`*`=x2.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1_f,x2_f = self._filter_input_dimensions(x1,x2)
        BP = logtheta[0]
        L = logtheta[1]

#        if(x1.shape[1] > 1):
#            grouping_1 = x1[:,1]
#        else:
#            grouping_1 = SP.repeat(-1,x1_f.shape[0])
#
#        if(x2 is not None and x2.shape[1] > 1):
#            grouping_2 = x2[:,1]
#        elif(x2 is None):
#            grouping_2 = grouping_1
#        else:
#            grouping_2 = SP.repeat(-1,x2_f.shape[0])
#
#        BP  = logtheta[0]
#        BPs1 = SP.array((x1_f<BP),dtype="int8")
#        BPs2 = SP.array((x2_f<BP),dtype="int8")
#        BPs = BPs1 * BPs2.reshape(-1)
#        grouping = grouping_1.reshape(-1) == grouping_2.reshape(1,-1)
#        if (False):
#            import pdb
#            pdb.set_trace()
#        return (SP.exp(k*BPs)+SP.exp(k*grouping)) / SP.exp(k)
        return -.5 * special.erf(((1./L) * (SP.dot(x1_f,x2_f.T))) - BP) + .5

    def Kgrad_theta(self, theta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        return self.K(theta, x1)[i]

    def Kdiag(self,logtheta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #default: naive implementation
        LG.debug("SEARDCF: Kdiag: Default unefficient implementation!")
        return self.K(logtheta,x1).diagonal()
    

    def Kgrad_x(self, logtheta, x, d):
        """
        The partial derivative of the covariance matrix with
        respect to x, given hyperparameters `logtheta`.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        L = SP.exp(logtheta[1:1+self.n_dimensions])
        dd = self._pointwise_distance(x,x,-(L**2))
        return self.K(logtheta,x) * dd.transpose(2,0,1)

    def get_default_hyperparameters(self,x=None,y=None):
        """
        Return default parameters for a particular
        dataset (optional).
        """
        #start with data independent default
        rv = SP.ones(self.n_hyperparameters)
        #start with a smallish variance
        rv[-1] = 0.1
        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return SP.log(rv)
    
    def get_Iexp(self, logtheta):
        """
        Return indices of which hyperparameters are to be exponentiated
        for optimization. Here we do not want
        
        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        return [0]

def softmax(x,y):
    ma = max(x,y)
    mi = min(x,y)
    return ma+SP.log(1+SP.exp(mi-ma))

def pointwise_softmax(x,y):
    return SP.array([[softmax(xi,yi) for xi in x] for yi in y])