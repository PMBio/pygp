"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) preprocessed by shiftCF(:py:class`covar.combinators.ShiftCF) and combined with noise
:py:class:`covar.noise.NoiseISOCF` by summing them up
(using :py:class:`covar.combinators.SumCF`).
"""

import cPickle

#import sys
#sys.path.append('/kyb/agbs/stegle/work/ibg/lib')

import pylab as PL
import scipy as SP
import numpy.random as random

from pygp.covar import se, noise, combinators, breakpoint
import pygp.priors.lnpriors as lnpriors

import logging
import copy
import os

from pygp.gp import GP
from pygp.gp.composite import GroupGP
from pygp.optimize import opt_hyper

logging.basicConfig(level=logging.INFO)

random.seed(1)

# #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
# xmin = 1
# xmax = 5
# x1 = SP.arange(xmin,xmax,.7)
# x2 = SP.arange(xmin,xmax,.4)

# C = 2       #offset
# b = 0.5
# sigma1 = 0.01
# sigma2 = 0.01
# n_noises = 1

# b = 0

# y1  = b*x1 + C + 1*SP.sin(x1)
# dy1 = b   +     1*SP.cos(x1)
# y1 += sigma1*random.randn(y1.shape[0])
# y1-= y1.mean()

# y2  = b*x2 + C + 1*SP.sin(x2)
# br = x2 >= 3
# dy2 = b   +     1*SP.cos(x2)
# y2 -= y2.mean()
# y2 -= br * y2
# y2 += sigma2*random.randn(y2.shape[0])

# x1 = x1[:,SP.newaxis]
# x2 = x2[:,SP.newaxis]

# x = SP.concatenate((x1,x2),axis=0)
# y = SP.concatenate((y1,y2),axis=0)

""" Get the warwick data """
# Data of warwick.pickle (depends on which system were running)

data_path = '/kyb/agbs/maxz/Documents/MPG/GP/diff_expression/time_delay/data'
data_file = os.path.join(data_path, 'data_warwick.pickle')
data_file_f = open(data_file, 'rb')
data = cPickle.load(data_file_f)

#for name, probe in data.iteritems():
for name in ['CATMA3A12810', 'CATMA3A22550' , 'CATMA4A36120', 'CATMA3A19900']:    
#    if not name == 'CATMA3A12810': #CATMA3A22550 , CATMA4A36120, CATMA3A19900
#        continue
    probe = data[name]
    
    C = probe['C']
    T = probe['T']

    x1 = C[0][0].reshape(-1, 1)
    x1_rep = SP.repeat(0, len(x1)).reshape(-1, 1)
    x1 = SP.concatenate((x1, x1_rep), axis=1)
    x2 = T[0][0].reshape(-1, 1)
    x2_rep = SP.repeat(1, len(x2)).reshape(-1, 1)
    x2 = SP.concatenate((x2, x2_rep), axis=1)

    x = SP.concatenate((x1, x2), axis=0)
    y = SP.concatenate((C[1][0], T[1][0]), axis=0).reshape(-1, 1)
    
    #predictions:
    X = SP.linspace(2, x2.max(), 100)[:, SP.newaxis]
    X_g1 = SP.repeat(0, len(X)).reshape(-1, 1)
    X_g2 = SP.repeat(1, len(X)).reshape(-1, 1)

    #hyperparamters
    dim = 1
    group_indices = SP.concatenate([SP.repeat(i, len(xi)) for i, xi in enumerate((C[0].reshape(-1, 1),
                                                                                T[0].reshape(-1, 1)))])
    SECF = se.SqexpCFARD(dim, dimension_indices=[0])
    breakpointCF = breakpoint.DivergeCF(dimension_indices=[0])
    #noiseCF = noise.NoiseReplicateCF(replicate_indices)
    noiseCF = noise.NoiseCFISO(dimension_indices=[0])
    SECF_noise = combinators.SumCF((SECF, noiseCF), dimension_indices=[0])
    CovFun = combinators.ProductCF((SECF_noise, breakpointCF), dimension_indices=[0])

    covar_priors = []
    #scale
    covar_priors.append([lnpriors.lnGammaExp, [6, .3]])
    for i in range(dim):
        covar_priors.append([lnpriors.lnGammaExp, [30, .1]])
    #noise
    for i in range(1):
        covar_priors.append([lnpriors.lnGammaExp, [1, .3]])
    #breakpoint, no knowledge
    for i in range(1):
        covar_priors.append([lnpriors.lnuniformpdf, [0, 0]])    

    logthetaCOVAR = SP.log([.4, 3.2, 0.3])#,sigma2])
    hyperparams = {'covar':logthetaCOVAR}

    covar_priors = SP.array(covar_priors)
    priors = {'covar' : covar_priors[[0, 1, 2]]}
    priors_BP = {'covar' : covar_priors}
    Ifilter = {'covar' : SP.array([1, 1, 1], dtype='int')}
    Ifilter_BP = {'covar' : SP.array([1, 1, 1, 0], dtype='int')}

    #gpr_BP = GPR.GP(CovFun,x=x,y=y)
    gpr_BP = GP(CovFun, x=x, y=y)
#    gpr_opt_hyper = GP(combinators.SumCF((SECF,noiseCF)),x=x,y=y)
    gpr_opt_hyper = GroupGP((GP(combinators.SumCF((SECF, noiseCF)), x=x1, y=C[1][0]),
                             GP(combinators.SumCF((SECF, noiseCF)), x=x2, y=T[1][0])))

    [opt_model_params, opt_lml] = opt_hyper(gpr_opt_hyper, hyperparams, priors=priors, gradcheck=False, Ifilter=Ifilter)
#    opt_model_params = hyperparams
    print SP.exp(opt_model_params['covar'])

#    import copy
#    _hyperparams = copy.deepcopy(opt_model_params)
    # _logtheta = SP.array([0,0,0,0],dtype='double')
    # _logtheta[:2] = _hyperparams['covar'][:2]
    # _logtheta[3] = _hyperparams['covar'][2]
    # _hyperparams['covar'] = _logtheta

    #[opt_model_params,opt_lml]=GPR.optHyper(gpr_BP,hyperparams,priors=priors,gradcheck=True,Ifilter=Ifilter)

    import pygp.plot.gpr_plot as gpr_plot
    first = True
    norm = PL.Normalize()

    break_lml = []
    plots = SP.int_(SP.sqrt(24) + 1)
    for i, BP in enumerate(x[:24, 0]):
        #PL.subplot(plots,plots,i+1)
        _hyper = copy.deepcopy(opt_model_params)
        _logtheta = _hyper['covar']
        _logtheta = SP.concatenate((_logtheta, [BP]))
        _hyper['covar'] = _logtheta
        #[opt_model_params,opt_lml] = opt_hyper(gpr_BP,_hyper,priors=priors_BP,gradcheck=False,Ifilter=Ifilter_BP)
        #break_lml.append(opt_lml)
        
        priors_BP[3] = [lnpriors.lngausspdf, [BP, 1]]
        
        break_lml.append(gpr_BP.LML(_hyper, priors_BP))

        # PL.plot(C[0].transpose(),C[1].transpose(),'+b',markersize=10)
        # PL.plot(T[0].transpose(),T[1].transpose(),'+r',markersize=10)

        # [M,S] = gpr_BP.predict(opt_model_params,X)

        # gpr_plot.plot_sausage(X,M,SP.sqrt(S),format_fill={'alpha':0.1,'facecolor':'k'})

        # gpr_BP_1 = GPR.GP(CovFun,x=x,y=y)
        # [M_1,S_1] = gpr_BP_1.predict(opt_model_params,SP.concatenate((X,X_g1),axis=1))
        # gpr_plot.plot_sausage(X,M_1,SP.sqrt(S_1),format_fill={'alpha':0.2,'facecolor':'b'})

        # x_filter = (x2<BP)[:,0]
        # gpr_BP_1_st = GPR.GP(combinators.SumCF((SECF,noiseCF),
        #                                        dimension_indices=[0]),
        #                      x=SP.concatenate((x1,x2[x_filter]),axis=0),
        #                      y=SP.concatenate((C[1].reshape(-1),
        #                                        T[1].reshape(-1)[x_filter]),axis=0).reshape(-1,1))
        # priors_st = {'covar' : SP.array(covar_priors)[[0,1,2]]}
        # Ifilter_st = {'covar' : SP.array([1,1,1],dtype='int')}
        # hyperparams_st = copy.deepcopy(hyperparams)
        # hyperparams_st['covar'] = hyperparams['covar'][[0,1,2]]
        # [opt_model_params_st,opt_lml_st]=GPR.optHyper(gpr_BP_1_st,hyperparams_st,priors=priors_st,
        #                                         gradcheck=False,Ifilter=Ifilter_st)
        # [M_1,S_1] = gpr_BP_1_st.predict(opt_model_params_st,X)
        # gpr_plot.plot_sausage(X,M_1,SP.sqrt(S_1),format_fill={'alpha':0.2,'facecolor':'g'})

        
        # PL.figure()
#        if(first):
#            first=False
#            K = CovFun.K(opt_model_params['covar'], x)
#            norm.autoscale(2*K)
#            PL.pcolor(K, norm=norm)
#        else:
#            PL.pcolor(CovFun.K(opt_model_params['covar'], x), norm=norm)
#
#        PL.title("BP = %i" % (BP))
        
        #gpr_BP_2 = GPR.GP(CovFun,x=x,y=y)
        #[M_2,S_2] = gpr_BP_2.predict(opt_model_params,SP.concatenate((X,X_g2),axis=1))
        #gpr_plot.plot_sausage(X,M_2,SP.sqrt(S_2),format_fill={'alpha':0.2,'facecolor':'r'})

        # predict
        # PL.subplot(plots,plots,i+1)
        
        if(BP == 20):
            [M, S] = gpr_BP.predict(_hyper, X)
            gpr_plot.plot_sausage(X, M, SP.sqrt(S))
            PL.plot(C[0].transpose(), C[1].transpose(), '-+b')
            PL.plot(T[0].transpose(), T[1].transpose(), '-+r')
        
    PL.figure()
    PL.plot(x[:24], break_lml)

    PL.show()

    #pdb.set_trace() 

    #PL.close()
    #PL.clf()
