"""
String kernel with derivatives
"""

import sys
sys.path.append("../../")
sys.path.append('/home/os252/work/lib/python')


# import python / numpy:
from pylab import *
from numpy import *
import cPickle

from pygp.covar import CovarianceFunction
import pygp.covar.sq_dist as sqdist
#from elefant.kernels.stringkernel.basicstringkernel import *
from shogun.Kernel import *
from shogun.Features import *
from shogun.PreProc import SortWordString, SortUlongString


class StringCovariance(CovarianceFunction):
    """String kernel adapted from elephant toolbox"""
    __slots__=["kernel","K_","x1_","x2_"]

    def __init__(self,index=0,swfGiven="constant", swfParamGiven=0.0):
        """string kernel on index index"""
        self.index = index
        self.n_params = 1


        self.kernel=None
        self.x1_ = self.x2_ = array([])

        #initialise degree string kernel

    def getParamNames(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        names.append('Astr')
        return names


    def K_old(self,logtheta,*args):
        x1 = args[0][:,self.index].copy()
        if(len(args)==1):
            x2 = x1.copy()
        else:
           x2 = args[1][:,self.index].copy()
        V0 = exp(2*logtheta[0])

        swf = self.swf
        swfParam = self.swfParam

        l1 = x1.shape[0]
        l2 = x2.shape[0]
        K = zeros([l1,l2])

        for i in xrange(l1):
            localSK = CBasicStringKernel(x1[i], None, swf, swfParam)
            K[i,:] = localSK.klist(x2.tolist())
        localSK = CBasicStringKernel(None, None, swf, swfParam)

        self.K_ = K
        #scale factor
        K*=V0
        

        return K


    def K(self,logtheta,*args):
        """new version using shogun"""
        x1 = args[0][:,self.index].copy()
        if(len(args)==1):
            x2 = x1.copy()
        else:
           x2 = args[1][:,self.index].copy()
        V0 = exp(2*logtheta[0])

        #try to get K from cache
        if (self.x1_.shape[0]==x1.shape[0]) and (self.x2_.shape[0]==x2.shape[0]):
            if ((self.x1_==x1).all() and (self.x2_==x2).all()):
                K = self.K_
                K = K*V0
                return K
            

        DNA = 0
        order = 3
        gap =0
        reverse = False
        charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(x1.tolist())
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(x2.tolist())
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()
	use_sign=False

        kernel=CommWordStringKernel(feats_train, feats_train, use_sign)
        kernel.init(feats_train,feats_test)
	K=kernel.get_kernel_matrix()
        self.K_ = K

        self.x1_ = x1
        self.x2_ = x2


        #scale factor
        K=K*V0
        return K
    

    def Kd(self,logtheta,*args):
        #1. calculate kernel
        #no noise
        _K = self.K(logtheta,*args)

        V0 = exp(2*logtheta[0])
        rv = zeros([self.n_params,_K.shape[0],_K.shape[1]])
        rv[:] = _K
        #amplitude
        rv[0]*= 2
        return rv
        

if __name__ == "__main__":
    import sys
    import cPickle

    x1 = array(["ACTGAA","ATT"])
    x2 = array(["ACTGAA", "CCCT"])

    
    
    x1 = x1.reshape([-1,1])
    x2 = x2.reshape([-1,1])
    sk = StringCovariance(swfParamGiven=0)

#    [x1,x2] = cPickle.load(open('debug.pickle','rb'))
    x1 = x1.reshape([-1,1])
    x2 = x2.reshape([-1,1])
    logtheta = log([1,0.1])
    for i in range(100):
        K12=sk.K(logtheta,x1,x2)

    print K12

    if 0:
        K=sk.K(logtheta,x1)
        Kd=sk.Kd(logtheta,x1)

        print K
        print Kd

    
    

    
