"""
Different Squared Exponential Covariance Function classes
---------------------------------------------------------
"""

import sys
sys.path.append("../")


# import python / numpy:
from pylab import *
from numpy import *

from covar import CovarianceFunction
import covar.sq_dist as sqdist
import pdb

class SquaredExponentialDCF(CovarianceFunction):

    dimension = 1                    #dimension of the data
    prior_params = []                #parameter of priors; we use gamme-priors

    def __init__(self,dimension=1):
        self.dimension = dimension
        self.n_params = self.dimension+2
        pass
   
    def weave_inline_blitz(self,X, Y):
        """
        Uses weave.inline
        """
        #print "Weave inline:"
    
   
        n, m = X.shape
        Z = zeros((n, m), 'd')
    
        cpp_code = (
                    """
            int i = 0, j = 0;
            for (i = 0; i < n; ++i)
                for (j = 0; j < m; ++j)
                    Z(i, j) = sin(X(i, j)) * cos(Y(i, j))+1;
            """)
        
        #print "Weave starting"
        scipy.weave.inline(cpp_code, ['X', 'Y', 'Z', 'n', 'm'], 
                     type_converters = converters.blitz)
        
        #print "Weave returning"
        return Z
   


    def K(self, logtheta, *args):
        '''K(params,x1) - params = [factor,length-scale(s)]'''
        # additional index: one index after dimension: derivative index!
        x1 = args[0][:,0:self.dimension+1]
        if(len(args)==1):
            x2 = x1
        else:
           x2 = args[1][:,0:self.dimension+1]
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv = V0*exp(-0.5*sqd)
                          
        #4. go through all other combinations and calc the corresponding terms:
        def k_(d1):
            rv = dd[:,:,d1]/L[d1]
            return rv
        def k__(d1,d2):
            f1 = dd[:,:,d1]/L[d1]
            f2 = dd[:,:,d2]/L[d2]
            return ((d1==d2)/(L[d1]**2) - f1*f2)
        for i in arange(-1,self.dimension):
            for j in arange(-1,self.dimension):
                if i==j==-1:
                    continue
                index = Bdist(d1==i,d2==j)
                if not any(index):
                    continue
                if((i==-1) & (j!=-1)):
                    rv[index] *= -1*k_(j)[index]
                elif((i!=-1) &(j==-1)):
                    rv[index] *= k_(i)[index]
                else:
                    rv[index] *= k__(i,j)[index]
        
        if(len(args)==1):                                     # add noise if we do not have two independen input data-sets
            rv+=eye(len(x1))*exp(2*logtheta[-1])
        return rv


    def Kd(self, logtheta, *args):
        '''Kd(params,x1) - params = [factor,length-scale(s)]'''
        # additional index: one index after dimension: derivative index!
        x1 = args[0][:,0:self.dimension+1]
        if(len(args)==1):
            x2 = x1
            noise=eye(len(x1))*exp(2*logtheta[-1])
        else:
           x2 = args[1][:,0:self.dimension+1]
           noise =eye(len(x1))*0
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L

        rv = zeros((self.n_params,len(x1),len(x2)))
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqdd = dd*dd
        sqd = sqdd.sum(axis=2)

        sqdd = sqdd.transpose(2,0,1)
       
        #3. calcualte without derivatives, need this anyway:
        rv[:] = V0*exp(-0.5*sqd)
        #amplitude:
        rv[0] = rv[0]*2
        #lengthscales:
        rv[1:1+self.dimension] *= sqdd
        #noise:
        rv[-1] = 2*noise
        
        return rv

   
    
    def Kb(self, logtheta, *args):
        '''K(params,x1) - params = [factor,length-scale(s)]'''
        # additional index: one index after dimension: derivative index!
        dim=self.dimension
        X1 = args[0][:,0:dim+1]
        if(len(args)==1):
            X2 = X1
        else:
           X2 = args[1][:,0:dim+1]
       
        rv = zeros((len(X1),len(X2)),'double')
        theta = exp(logtheta)
        n, m = rv.shape
        support_code = (
                        """
                        #include <blitz/array.h>

                        using namespace blitz;
                        
                        inline double k(Array<double,1> theta,Array<double,1> x1,Array<double,1> x2)
                        {
                            double sum=0;
                            for(int i=0;i<x1.shape().length();i++)
                            {
                                sum+=pow((x1(i)-x2(i))/theta(i+1),2);
                            }
                            return theta(0)*exp(-0.5*sum);
                        }
                        inline double k_(Array<double,1> theta,Array<double,1> x1,Array<double,1> x2,int d1)
                        {
                            //printf("k_");
                            double rv = k(theta,x1,x2);
                            
                            rv *= (x1(d1)-x2(d1))/pow(theta(d1+1),2);
                            return rv;
                        }
                        inline double k__(Array<double,1> theta,Array<double,1> x1,Array<double,1> x2,int d1,int d2)
                        {
                            //printf("k__");
                            double rv = k(theta,x1,x2);
                            double f1 = (x1(d1)-x2(d1))/pow(theta(d1+1),2);
                            double f2 = (x1(d2)-x2(d2))/pow(theta(d2+1),2);
                            rv*=( (d1==d2)*1.0/pow(theta(d1+1),2) - f1*f2 );
                            return rv;
                        }
                        """)
        cpp_code = (
                    """
            int TT;
            int i = 0, j = 0;
            int d1;
            int d2;
            Array<double,1> x1;
            Array<double,1> x2;
            //Array<double,1> test;
            for (i = 0; i < n; ++i)
                for (j = 0; j < m; ++j)
                    {
                       d1=(int) X1(i,dim);
                       d2=(int) X2(i,dim);
                       Array<double,1> x1= X1(i,Range(0,dim-1));
                       Array<double,1> x2= X2(j,Range(0,dim-1));

                    
                       //switch between different cases:
                       //printf("d1:%d,d2:%d\\n",d1,d2);
                       if( (d1!=-1) & (d2!=-1) )
                           rv(i,j)=k__(theta,x1,x2,d1,d2);
                       else if( (d1==-1) & (d2!=-1) )
                            rv(i,j)=k_(theta,x1,x2,d2);
                       else if( (d1!=-1) & (d2==-1) )
                            rv(i,j)=k_(theta,x2,x1,d1);
                       else
                            rv(i,j)=k(theta,x1,x2);

                            
                    }
            """)

        scipy.weave.inline(cpp_code, ['X1', 'X2', 'rv', 'n', 'm','dim','theta'], 
               type_converters = converters.blitz,support_code=support_code)
        
        

        if(len(args)==1):                                     # add noise if we do not have two independen input data-sets
            rv+=eye(len(X1))*exp(logtheta[-1])
        return rv
    








class SquaredExponentialCF(CovarianceFunction):
    """
    Squared Exponential Covariance Function with noise parameter.
    """
#    __slots__ = ["prior_params"]
    
    def __init__(self,dimension=None,index=None):
        if (index is not None):
            self.index = array(index,dtype='int32')
        elif dimension is not None:
            self.index = arange(0,dimension)

        self.dimension = self.index.max()+1-self.index.min()
        self.n_params = self.dimension+2
        pass

    # def dist(self,x1,x2,L):
    #     """Distance function"""
    #     x1 = array(x1,dtype='float64')/L
    #     x2 = array(x2,dtype='float64')/L
    #     return sqdist.dist(x1,x2)
        
   
    def K(self, logtheta, *args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        # additional index: one index after dimension: derivative index!
        #x1 = array(args[0][:,0:self.dimension],dtype='float64')
        x1 = args[0][:,self.index]
        if(len(args)==1):
            x2 = x1
            noise=eye(len(x1))*(exp(2*logtheta[-1])+1E-6)
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = args[1][:,self.index]
           noise =0
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        # calculate the distance betwen x1,x2 for each dimension separately, reweighted by L.
        dd = self._dist(x1,x2,L)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv = V0*exp(-0.5*sqd) + noise
                          
        
        return rv


    def Kd(self, logtheta, *args):
        """The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
                # additional index: one index after dimension: derivative index!
        #x1 = array(args[0][:,0:self.dimension],dtype='float64')
        x1 = args[0][:,self.index]
        if(len(args)==1):
            x2 = x1
            noise=eye(len(x1))*exp(2*logtheta[-1])
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = args[1][:,0:self.index]
           noise =eye(len(x1))*0
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = self._dist(x1,x2,L)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqdd = sqd.transpose(2,0,1)
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*exp(-0.5*sqd)
        
       
        rv = zeros((self.n_params,len(x1),len(x2)))
        
        #3. calcualte without derivatives, need this anyway:
        rv[:] = V0*exp(-0.5*sqd)
        #amplitude:
        rv[0] = rv[0]*2
        #lengthscales:
        rv[1:1+self.dimension] *= sqdd
        #noise:
        rv[-1] = 2*noise
        
        return rv

    def getDefaultParams(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
        #start with data independent default
        rv = ones(self.n_params)
        #start with a smallish variance
        rv[-1] = 0.1

        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
            #noise
            rv[-1] = 1E-2*rv[0]
            rv[-1] = 1E-1*rv[0]
        
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)

    def getDimension(self):
        return self.dimension






class SquaredExponentialCFnn(CovarianceFunction):
    """
    Squared Exponential covariance function without noise.
    """
    #__slots__ = ["prior_params","Iactive"]
    

    def __init__(self,dimension=None,index=None):
        if (index is not None):
            self.index = array(index,dtype='int32')
        elif dimension is not None:
            self.index = arange(0,dimension)
        else:
            self.index = arange(0,1)

        self.dimension = self.index.max()+1-self.index.min()
        self.Iactive = arange(self.dimension)
        self.n_params = self.dimension+1
        pass

    def getParamNames(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = []
        names.append('Ase')
        for dim in range(self.dimension):
            names.append('L%d' % dim)
        return names

    # def dist(self,x1,x2,L):
    #     """Distance function"""
    #     x1 = array(x1,dtype='float64')/L
    #     x2 = array(x2,dtype='float64')/L
    #     return sqdist.dist(x1,x2)
        
   
    def K(self, logtheta, *args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        # additional index: one index after dimension: derivative index!
        x1 = args[0][:,self.index][:,self.Iactive]
        if(len(args)==1):
            x2 = x1
        else:
           x2 = args[1][:,self.index][:,self.Iactive]
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension][self.Iactive])
        
        # calculate the distance betwen x1,x2 for each dimension separately, reweighted by L. 
        dd = self._dist(x1,x2,L)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv = V0*exp(-0.5*sqd)
                          
        
        return rv


    def Kd(self, logtheta, *args):
        """The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
                # additional index: one index after dimension: derivative index!
        #x1 = array(args[0][:,0:self.dimension],dtype='float64')
        x1 = args[0][:,self.index][:,self.Iactive]
        if(len(args)==1):
            x2 = x1
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = args[1][:,0:self.index][:,self.Iactive]
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension][:,self.Iactive])
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = self._dist(x1,x2,L)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqdd = sqd.transpose(2,0,1)
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*exp(-0.5*sqd)
        
       
        rv = zeros((self.n_params,len(x1),len(x2)))
        
        #3. calcualte without derivatives, need this anyway:
        rv[:] = V0*exp(-0.5*sqd)
        #amplitude:
        rv[0] = rv[0]*2
        #lengthscales:
        rv[1:1+self.dimension][self.Iactive] *= sqdd
        
        return rv

    def getDefaultParams(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
        #start with data independent default
        rv = ones(self.n_params)
        #start with a smallish variance
        rv[-1] = 0.1

        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
        
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)

    def getDimension(self):
        return self.dimension




class SquaredExponentialCFinoise(CovarianceFunction):
    """individual noise per datum"""
    
    dimension = 1                    #dimension of the data
    prior_params = []                #parameter of priors; we use gamme-priors

    def __init__(self,dimension=1):
        self.dimension = dimension
        self.n_params = self.dimension+2
        pass
   
    def K(self, logtheta, *args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        # additional index: one index after dimension: derivative index!
        x1 = args[0][:,0:self.dimension]
        #noise_diag comes from the last input dimension:
        noise_diag = args[0][:,-1]
        if(len(args)==1):
            x2 = x1
            noise=diag(noise_diag)*(exp(2*logtheta[-1])+1E-6)
        else:
           x2 = args[1][:,0:self.dimension]
           noise =0
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv = V0*exp(-0.5*sqd) + noise
                          
        
        return rv


    def Kd(self, logtheta, *args):
        """The derivatives of the covariance matrix for
        each hyperparameter, respectively.
        
        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        # additional index:
        # one index after dimension: derivative index!
        x1 = args[0][:,0:self.dimension]
        noise_diag = args[0][:,-1]
        if(len(args)==1):
            x2 = x1
            noise=diag(noise_diag)*exp(2*logtheta[-1])
        else:
            x2 = args[1][:,0:self.dimension]
            noise =eye(len(x1))*0
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqdd = sqd.transpose(2,0,1)
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*exp(-0.5*sqd)
        
       
        
        
        
        rv = zeros((self.n_params,len(x1),len(x2)))
        
        #3. calcualte without derivatives, need this anyway:
        rv[:] = V0*exp(-0.5*sqd)
        #amplitude:
        rv[0] = rv[0]*2
        #lengthscales:
        rv[1:1+self.dimension] *= sqdd
        #noise:
        rv[-1] = 2*noise
        
        return rv

    def getDefaultParams(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
        #start with data independent default
        rv = ones(self.n_params)
        #start with a smallish variance
        rv[-1] = 0.1

        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
            #noise
            rv[-1] = 1E-2*rv[0]
            rv[-1] = 1E-1*rv[0]
        
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)





class SquaredExponentialClusterCF(CovarianceFunction):
    """
    same as SquaredExponentialCF but here we assume
    multiple groups with different noise levels which
    are learnt separately
    """
        
    dimension = 1                    #dimension of the data
    prior_params = []                #parameter of priors; we use gamme-priors
    clusters  = 2 

    def __init__(self,dimension=1,clusters=2):
        """__init__(self,dimensio=1,clusters=2)"""
        self.dimension = dimension
        self.clusters  = clusters
        self.n_params = self.dimension+clusters+1
        pass
   
    def K(self, logtheta, *args):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
        # additional index: one index after dimension: derivative index!
        x1      = args[0][:,0:self.dimension]
        #cluster index is in the next index, convert to int to allow indexing
        cluster = array(args[0][:,self.dimension],dtype='int')
        if(len(args)==1):
            x2 = x1
            #noise depends on the cluster memebership:
            noise=(exp(2*logtheta[-self.clusters+cluster])+1E-2)
        else:
            x2 = args[1][:,0:self.dimension]
            noise = [0]
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv = V0*exp(-0.5*sqd) + diag(noise)
        return rv


    def Kd(self, logtheta, *args):
        """The derivatives of the covariance matrix for
        each hyperparameter, respectively.
        
        **Parameters:**
        See :py:class:`covar.CovarianceFunction`
        """
                # additional index: one index after dimension: derivative index!
        x1 = args[0][:,0:self.dimension]
        cluster = array(args[0][:,self.dimension],dtype='int')
        if(len(args)==1):
            x2 = x1
            noise=ones(len(x1))*exp(2*logtheta[-self.clusters+cluster]+1E-2)
        else:
            x2 = args[1][:,0:self.dimension]
            noise =zeros(len(x1))

        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension])
        
        d1 = x1[:,-1::]
        d2 = x2[:,-1::]
        x1 = x1[:,0:self.dimension]/L
        x2 = x2[:,0:self.dimension]/L
        
        # calculate the distance betwen x1,x2 for each dimension separately.
        dd = sqdist.dist(x1,x2)
        # sq. distance is neede anyway:
        sqd = dd*dd
        sqdd = sqd.transpose(2,0,1)
        sqd = sqd.sum(axis=2)
       
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0*exp(-0.5*sqd)
        
       
        
        
        
        rv = zeros((self.n_params,len(x1),len(x2)))
        
        #3. calcualte without derivatives, need this anyway:
        rv[:] = V0*exp(-0.5*sqd)
        #amplitude:
        rv[0] = rv[0]*2
        #lengthscales:
        rv[1:1+self.dimension] *= sqdd
        #noise:
        for ic in range(self.clusters):
            Ic = (cluster==ic)
            diagonal = zeros(x1.shape[0])
            diagonal[Ic] = 2*noise[Ic]
            rv[-self.clusters+ic][:,:] = diag(diagonal)
        return rv

    def getDefaultParams(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
        #start with data independent default
        rv = ones(self.n_params)
        #start with a smallish variance
        rv[-self.clusters::] = 0.1

        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
            #noise
            rv[-1] = 1E-2*rv[0]
            rv[-1] = 1E-1*rv[0]
        
        if x is not None:
            rv[1:-self.clusters] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)

class SquaredExponentialCFTPnn(CovarianceFunction):
    """
    Provides computation of the SETP
    (Squared Exponential with Time Parameter) without
    noise.

    **Parameters:**
    
    n_replicates : int
        Number of replicates which have an own time parameter.

    dimension : int
        number of dimensions for calculation
        (self.Iactive = arange(dimension)).

    index : == dimension????   TODO

    """

    __slots__ = ["prior_params","Iactive","n_replicates"]
    
    def __init__(self,dimension=None,index=None,n_replicates=1):
        """Get an instance of SETP.
         """

        if (index is not None):
            self.index = array(index,dtype='int32')
        elif dimension is not None:
            self.index = arange(0,dimension)
        else:
            self.index = arange(0,1)

        self.dimension = self.index.max()+1-self.index.min()
        self.Iactive = arange(self.dimension)
        self.n_replicates = n_replicates
        
        self.n_params = self.dimension+n_replicates+1
        pass

    def getParamNames(self):
        #"return the names of hyperparameters to make identification easier"
        names = []
        names.append('Amplitude')
        for dim in range(self.dimension):
            names.append('Length-Scale %d' % dim)
        for dim in range(n_replicates):
            names.append('Time-Parameter rep%d' % dim) 
        return names        
   
    def K(self, logtheta, *xs):
        """
        Get Covariance matrix K with given hyperparameters
        logtheta and inputs *args* = X[, X'].

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = xs[0][:,self.index][:,self.Iactive]
        
        if(xs[0].shape[1]==2):
            # control is the index of replicate of the respective x. 
            control = xs[0][:,self.index+1][:,self.Iactive]
        else:
            # there is no control
            control = zeros_like(x1)

        # get the time parameter(s) T
        T  = logtheta[1+self.dimension:1+self.dimension+self.n_replicates]

        # subtract respective time parameters from the input x.
        for i in unique(control):
            x1[control==i] -= T[i]

        if(len(xs)==1 and x1.shape[1]>0):
            # there is only one input
            x2 = x1
        else:
            # compare X=x1 to X*=x2
            x2 = xs[1][:,self.index][:,self.Iactive]
        
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension][self.Iactive])
        
        # calculate the distance between x1,x2 for each dimension separately, reweighted by L. 
        dd = self._dist(x1,x2,L)
        sqd = dd*dd
        sqd = sqd.sum(axis=2)
       
        #calculate without derivatives
        rv = V0*exp(-0.5*sqd)
        return rv
    
    def Kd(self, logtheta, *xs):
        """The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = xs[0][:,self.index][:,self.Iactive]
        x1_ = x1.copy()

        if (xs[0].shape[1]==2):
            # control is the index of replicate of the respective x. 
            control = xs[0][:,self.index+1][:,self.Iactive]
        else:
            # there is no control
            control = zeros_like(x1)

        # get the time parameter(s) T
        T  = logtheta[1+self.dimension:1+self.dimension+self.n_replicates]

        # # subtract respective time parameters from the input x. 
        for i in unique(control):
            x1[control==i] -= T[i]#n_replicate_deltaT

        if(len(xs)==1):
            x2 = x1
        else:
           #x2 = array(args[1][:,0:self.dimension],dtype='float64')
           x2 = xs[1][:,self.index][:,self.Iactive]
        
        # 2. exponentiate params:
        V0 = exp(2*logtheta[0])
        L  = exp(logtheta[1:1+self.dimension][self.Iactive])
        
        # calculate the distance betwen x1,x2 for each dimension separately, reweighted by L.
        dd = self._dist(x1,x2,L)
        sqd = dd*dd
        sqdd = sqd.transpose(2,0,1)
        sqd = sqd.sum(axis=2)

        rv = zeros((self.n_params,len(x1),len(x2)))
        
        # calculate derivates
        rv[:] = V0*exp(-0.5*sqd)

        #amplitude:
        rv[0] = rv[0]*2

        #lengthscales:
        rv[1:1+self.dimension][self.Iactive] *= sqdd

        #time-shift
        absdd = -dd / L
        # each replicate has its own time shift
        for rep in range(self.n_replicates):
            # get control for replicate rep
            c_ = array(control==rep,dtype='int')
            # for each replicate rep get its own distance changes
            cdist = sqdist.dist(-c_,-c_)
            absdd_ = absdd * cdist
            dds = absdd_.transpose(2,0,1)
            # multiply to K (see derivatives of SETP).
            rv[1+rep+self.dimension:2+rep+self.dimension][self.Iactive] *= dds
        return rv

    def getDefaultParams(self,x=None,y=None):
        #"""getDefaultParams(x=None,y=None)
        #- return default parameters for a particular dataset (optional)
        #"""
        #start with data independent default
        rv = ones(self.n_params)
        #start with a smallish variance
        rv[-1] = 0.1

        if y is not None:
            #adjust amplitude
            rv[0] = (y.max()-y.min())/2
        
        if x is not None:
            rv[1:-1] = (x.max(axis=0)-x.min(axis=0))/4
        return log(rv)

    def getDimension(self):
        return self.dimension



if __name__ == "__main__":
    import sys
    sys.path.append("../../")
    cf=SquaredExponentialDCF(1) 
    x = arange(0,10,dtype=double)
    X = zeros((size(x),2))
    Y = zeros((size(x),2))
    X[:,0] = x
    Y[:,0] = x
    X[:,1] = -1
    Y[:,1] = 0
    print "before"
    rv=cf.K(log([1,1]),X,Y)
    print "after"
    #print rv
    
    
    

