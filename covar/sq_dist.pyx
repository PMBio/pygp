import numpy as np
import cython

# def Bdist(*args):
#     '''binary distance matrix:
#     dist(X)    -  return matrix of size (len(X),len(X)) all True!
#     dist(X1,X2)-  return matrix of size (len(X1),len(X2)) with (xi==xj)'''
    
    
#     if(len(args)==1):
#         #return true
#         X  = args[0]
#         Y  = args[0]
#         #X = N.matrix(args[0])
#         #Y = N.matrix(args[0])
#     elif(len(args)>=2):
#         X  = args[0]
#         Y  = args[1]
#         #X = N.matrix(args[0])
#         #Y = N.matrix(args[1])
        
        
#     rv = N.zeros((len(X),len(Y)),'double')
    
#     A = N.repmat(X,1,len(Y))
#     B = N.repmat(Y.T,len(X),1)
    
#     rv = (A&B)
    
#     return rv

cpdef dist(x1, x2):
    '''calcualte disntance of all inputs:
    dist(X)     : Matrix of all combinations of distances of Xi with Xj
    dist(X1,X2) : Matrix of all combinations of distances of X1i with X2j'''
    cdef int i,j
    cdef float x1val,x2val
    cdef ret = np.ones((len(x1),len(x2)))
    for i,x1val in enumerate(x1):
        for j,x2val in enumerate(x2):
            ret[i][j] = x1val - x2val
    return np.array(ret)


# def sq_dist(*args):
#     '''calcualte square-distance of all inputs:
#     sq_dist(X)     : Matrix of all combinations of distances of Xi with Xj
#     sq_dist(X1,X2) : Matrix of all combinations of distances of X1i with X2j'''
#     rv = dist(*args)
#     #elementwise product does the tric:
#     rv = rv*rv
#     #now sum up the last dimension:
#     rv = rv.sum(axis=2)
#     return rv
    
if __name__ == "__main__":
    X = np.array([[1,2],[5,6]],dtype='double')
    Y = np.array([[1,2],[8,7],[0,0]],dtype='double')
    print dist(X,Y)
