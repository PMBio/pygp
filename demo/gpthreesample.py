'''
Created on Aug 10, 2011

@author: maxz
'''
import scipy as SP, pylab as PL
from pygp.covar.fixed import FixedCF
from pygp.covar.combinators import SumCF
from pygp.covar.se import SqexpCFARD
from pygp.priors import lnpriors
from pygp import likelihood as lik
from pygp.gp.gp_base import GP
from pygp.optimize.optimize_base import opt_hyper
from pygp.plot import gpr_plot

def run_gpthreesample(xg, y, sigma):
    X = SP.linspace(xg.min(), xg.max(), 100)
    
    print "creating covariance"
    K=[]
    for s in sigma:
        K.extend(SP.repeat(s, xg.shape[0]))
    K = SP.diag(K)
    covar_fixed_sigma = FixedCF(K)
    covar_squared_exponential = SumCF((SqexpCFARD(),covar_fixed_sigma))
    
    likelihood = lik.GaussLikISO()

    covar_parms = SP.log([1])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1])}       
    covar_priors = []
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    lik_priors = []
    lik_priors.append([lnpriors.lnGammaExp,[1,1]])
    priors = {'covar':covar_priors,'lik':lik_priors}
    
    print "constant model"
    gp_fixed = GP(covar_fixed_sigma,likelihood=likelihood,x=SP.tile(xg,3),y=SP.concatenate(y))
    print "optimzing"
    opt_model_params_fixed = opt_hyper(gp_fixed,hyperparams,priors=priors,gradcheck=True)[0]

    covar_parms = SP.log([1,1])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1])}       
    covar_priors = []
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    lik_priors = []
    lik_priors.append([lnpriors.lnGammaExp,[1,1]])
    priors = {'covar':covar_priors,'lik':lik_priors}
    
    print "standard model"
    gp_standard = GP(covar_squared_exponential,likelihood=likelihood,x=SP.tile(xg,3),y=SP.concatenate(y))
    print "optimzing"
    opt_model_params_standard = opt_hyper(gp_standard,hyperparams,priors=priors,gradcheck=True)[0]

    print "plotting"
    [M_fixed,S_fixed] = gp_fixed.predict(opt_model_params_fixed,X)
    [M_standard,S_standard] = gp_standard.predict(opt_model_params_standard,X)
    gpr_plot.plot_sausage(X,M_fixed,SP.sqrt(S_fixed))
    gpr_plot.plot_training_data(xg,y)
    PL.figure()

    gpr_plot.plot_sausage(X,M_standard,SP.sqrt(S_standard))
    gpr_plot.plot_training_data(xg,y)
    PL.show()

    
if __name__ == '__main__':
    
    xg = SP.linspace(0,SP.pi,5)
    
    y1 = SP.sin(xg+.5*SP.pi) * SP.cos(2*xg)
    y2 = y1+.1*SP.exp(xg)-.4
    y3 = SP.sin(xg+.5*SP.pi)
    
    sigma1 = SP.var(y1)
    sigma2 = SP.var(y2)
    sigma3 = SP.var(y3)
    
    run_gpthreesample(SP.arange(5,10), [y1, y2, y3], [sigma1, sigma2, sigma3])
