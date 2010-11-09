import pylab as PL
import scipy as SP
import scipy.linalg as linalg

import covar.sederiv as sederiv

import covar.noiseCF as noiseCF
import covar.sumCF as sumCF
import lnpriors as LNP

import gpr as GPR

import pdb

def raiseGPon(x,y,interpolInterval,logtheta,covar,priors=None,opt=True):
    """ raise GP Regression on input input, output y on the interpolation interval interpolInterval with hyperparemters logtheta log(amplitude,lengthscale,timeshift,noise)"""
    # append replicate param
    xrep = SP.concatenate([SP.repeat(i,len(x)/2) for i in range(2)]).reshape(-1,1)
    x = SP.concatenate((x,xrep),axis=1)
    # create GP regression
    gprtp = GPR.GPex(covar,Smean=True,x=x,y=y.reshape(-1,1),
                     rescale_dim=SP.arange(1))
    pdb.set_trace()
    
    if opt:
        logtheta0 = GPR.optHyper(gprtp,logtheta,SP.ones_like(logtheta),
                                 priors=priors,Iexp=SP.array([1,1,0,0,1]))
    else:
        logtheta0 = logtheta

    [M,S] = GPR.predict(logtheta0,interpolInterval)
    
    return [M,S,logtheta0]

def plotData(input,output,shape='g-'):
    pl = PL.plot(input,output,shape)
    return pl

def plotMeanAndSDV(M,S,color='green',alpha=0.1,shape='g-',deltaT=None):
    if(deltaT is None):
        plot = PL.plot(X[:,0],M,shape)
        PL.fill_between(X[:,0], M+2*SP.sqrt(S), M-2*SP.sqrt(S), 
                        facecolor=color, alpha=alpha)
    else:
        plot = PL.plot(X[:,0]-deltaT,M,shape)
        PL.fill_between(X[:,0]-deltaT, M+2*SP.sqrt(S), M-2*SP.sqrt(S), 
                        facecolor=color, alpha=alpha)
    return plot

if __name__ == '__main__':
    # plot random gprs (3) (with deltaT)

    # create inputs (Note: x is to have shape (-1,1))
    x = SP.arange(0,20,1)
    x = x.reshape(-1,1)

    # create covariance function (noise + setp)
    nCF = noiseCF.NoiseCovariance()
    tpCF = sederiv.SquaredExponentialCFTPnn(1,n_replicates=2)
    covar = sumCF.SumCovariance((tpCF,nCF))
       
    # x*, interpolation interval
    X = SP.linspace(x[0][0],x[-1][0],x.shape[0]*10)
    X = X.reshape(-1,1)

    # define priors to be used
    priors = []
    priors.append([LNP.lngammapdf,[3,1]])
    priors.append([LNP.lngammapdf,[15,.1]])
    priors.append([LNP.lngauss,[0,1]])
    priors.append([LNP.lngauss,[0,1]])
    priors.append([LNP.lngammapdf,[1,1]])

    # design
    shapesData = ['g:+','b:+','r:+']
    shapesMean = ['g-','b-','r-']
    colors = ['green','blue','red']

    # logthetas: mean of each prior
    logtheta = SP.array([SP.log(p[1][0]*p[1][1]) for p in priors])
    # Time hyper parameter is nonlog
    logtheta[2:4] = SP.exp(logtheta[2:4])
    
    # plots and logthetas for legend
    plots=[]
    logthetaOs = []

    # new figure
    PL.figure()

    # 3 examples
    for i in range(3):
        # get logtheta
        # create training data set y
        y = (10*(i+1))*SP.random.randn(x.shape[0])
        # plot training data set
        PL.plot(x,y,marker='+',color=colors[i],markersize=12)
        # raise GP regression on training data x,y
        [M,S,logtheta0] = raiseGPon(x,y,X,logtheta,covar,priors=priors,opt=True)
        # append to plots for legend
        plots.append(plotMeanAndSDV(M,S,color=colors[i],shape=shapesMean[i]))
        logthetaOs.append(logtheta0)

    # label x/y
    PL.xlabel('input')
    PL.ylabel('output')

    legend = [r'$A=%.1f, L=%.1f$'%(l[0],l[1]) for l in SP.exp(logthetaOs)]
    PL.legend(plots,legend)
