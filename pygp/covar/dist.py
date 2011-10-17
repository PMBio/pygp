"""helper module for distances"""

import scipy as SP

def dist(X,Y=None):
    '''calcualte disntance of all inputs:
    dist(X)     : Matrix of all combinations of distances of Xi with Xj
    dist(X1,X2) : Matrix of all combinations of distances of X1i with X2j'''
    if(len(X.shape)<=1):
        X=X.reshape(-1,1)
    if(Y is None):
        Y=X
    if(X.shape[1]>1):
        #save length and dimension
        lx = X.shape[0]
        ly = Y.shape[0]
        dim = X.shape[1]
        #reshape stuff and copy it.
        Xr = X.reshape((1,lx,dim))
        Yr = Y.reshape((ly,1,dim))
        #create repeats:
        A =  Xr.repeat(ly,0)
        B =  Yr.repeat(lx,1)
        rv = B-A
    else:
        rv = _dist_1_dimension(X, Y)
    return rv

def sq_dist(*args):
    '''calcualte square-distance of all inputs:
    sq_dist(X)     : Matrix of all combinations of distances of Xi with Xj
    sq_dist(X1,X2) : Matrix of all combinations of distances of X1i with X2j'''
    rv = dist(*args)
    #elementwise product does the tric:
    rv = rv*rv
    #now sum up the last dimension:
    rv = rv.sum(axis=2)
    return rv


def Bdist(*args):
    '''binary distance matrix:
    dist(X)    -  return matrix of size (len(X),len(X)) all True!
    dist(X1,X2)-  return matrix of size (len(X1),len(X2)) with (xi==xj)'''
    
    
    if(len(args)==1):
        #return true
        X  = args[0]
        Y  = args[0]
    elif(len(args)>=2):
        X  = args[0]
        Y  = args[1]
    A = SP.repeat(X,1,len(Y))
    B = SP.repeat(Y.T,len(X),1)    
    rv = (A&B)
   
    return rv



def _dist_1_dimension(X,Y=None):
    if(Y is None):
        Y=X
    #reshape stuff and copy it.
    rv = X.reshape(-1,1,1) - Y.reshape(-1,1)   
    return rv
   
    
if __name__ == "__main__":
    X = SP.array([[1,2],[5,6]],dtype='double')
    Y = SP.array([[1,2],[8,7],[0,0]],dtype='double')
    print sq_dist(X,Y)
