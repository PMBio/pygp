def dist(*args):
    '''calcualte disntance of all inputs:
    dist(X)     : Matrix of all combinations of distances of Xi with Xj
    dist(X1,X2) : Matrix of all combinations of distances of X1i with X2j'''
    
    
    Y = args[0]
    if(len(args)==1):
        X = args[0]
    elif(len(args)>=2):
        X = args[1]
    
    #reshape stuff and copy it.
    rv = Y.reshape(-1,1) - X.reshape(-1,1,1)
   
    return rv
