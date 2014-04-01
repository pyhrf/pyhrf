
import os
import unittest
import numpy as _np

from pyhrf.jde.beta import *
from pyhrf.boldsynth.field import *
from pyhrf.graph import *
from pyhrf.tools import montecarlo

from pyhrf.validation import config
from pyhrf.validation.config import figfn


class field_energy_calculator:
    def __init__(self, graph):
        self.graph = graph
    def __call__(self, labels):
        hc = count_homo_cliques(self.graph, labels)
        return -float(hc)/len(self.graph)

class PottsTest(unittest.TestCase):

    def setUp(self):
        self.plot = config.savePlots
        self.outDir = os.path.join(config.plotSaveDir, 
                                   './PottsPrior')
        if self.plot and  not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        self.verbose = False

    
    def test_sw_nrj(self):
        size = 100
        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        nc = 2
        betas = _np.arange(0, 1.4, .2)
        mU = _np.zeros(len(betas))
        vU = _np.zeros(len(betas))

        nrjCalc = field_energy_calculator(g)
        
        for ib, b in enumerate(betas):
            #print 'MC for beta ', b
            pottsGen = potts_generator(graph=g, beta=b, nbLabels=nc,
                                       method='SW')
            mU[ib], vU[ib] = montecarlo(pottsGen, nrjCalc, nbit=40)
            
        if config.savePlots:
            import matplotlib.pyplot as plt
            plt.plot(betas, mU)
            plt.errorbar(betas, mU, vU**.5)
            plt.xlabel('beta')
            plt.ylabel('mean U per site')
            plt.show()

        #print mU
        #print vU
        # assert max (d U(beta)) == 0.88

    def test_SW_nrj(self):
        size = 100
        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        nc = 2
        betas = _np.arange(0, 2.5, .2)
        mU = _np.zeros(len(betas))
        vU = _np.zeros(len(betas))

        nrjCalc = field_energy_calculator(g)
        
        for ib, b in enumerate(betas):
            #print 'MC for beta ', b
            pottsGen = potts_generator(graph=g, beta=b, nbLabels=nc,
                                       method='SW')
            mU[ib], vU[ib] = montecarlo(pottsGen, nrjCalc, nbit=5)


        import matplotlib as plt
        plt.plot(betas, mU,'b-')
        plt.errorbar(betas, mU, vU**.5,fmt=None,ecolor='b')

        for ib, b in enumerate(betas):
            #print 'MC for beta ', b
            pottsGen = potts_generator(graph=g, beta=b, nbLabels=3,
                                       method='SW')
            mU[ib], vU[ib] = montecarlo(pottsGen, nrjCalc, nbit=5)
            
        if config.savePlots:
            plt.plot(betas, mU,'r-')
            plt.errorbar(betas, mU, vU**.5,fmt=None,ecolor='r')
            plt.xlabel('beta')
            plt.ylabel('mean U per site')
            plt.xlim(betas[0]-.1,betas[-1]*1.05)
            plt.show()

        #print mU
        #print vU
        # assert max (d U(beta)) == 0.88


    def test_SW_nrj_2C_3C(self):
        size = 400
        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        betas = _np.arange(0, 2.7, .2)
        nitMC = 100
        mU2C = _np.zeros(len(betas))
        vU2C = _np.zeros(len(betas))
        mU3C = _np.zeros(len(betas))
        vU3C = _np.zeros(len(betas))

        nrjCalc = field_energy_calculator(g)
        
        #print "nbClasses = 2"
        for ib, b in enumerate(betas):
            #print '    MC for beta ', b
            pottsGen = potts_generator(graph=g, beta=b, nbLabels=2,
                                       method='SW')
            mU2C[ib], vU2C[ib] = montecarlo(pottsGen, nrjCalc, nbit=nitMC)

        #print '  mu2C=',mU2C
        #print '  vU2C=',vU2C


        #print "nbClasses = 3"
        for ib, b in enumerate(betas):
            #print '    MC for beta ', b
            pottsGen = potts_generator(graph=g, beta=b, nbLabels=3,
                                       method='SW')
            mU3C[ib], vU3C[ib] = montecarlo(pottsGen, nrjCalc, nbit=nitMC)

        #print '  mu3C=',mU3C
        #print '  vU3C=',vU3C
            
        if config.savePlots:
            import matplotlib.pyplot as plt
            plt.plot(betas, mU2C,'b-',label="2C")
            plt.errorbar(betas, mU2C, vU2C**.5,fmt=None,ecolor='b')
            plt.plot(betas, mU3C,'r-',label="3C")
            plt.errorbar(betas, mU3C, vU3C**.5,fmt=None,ecolor='r')
            plt.legend(loc='upper right')
            plt.title('Mean energy in terms of beta \n for 2-color and 3-color Potts (SW sampling)')
            plt.xlabel('beta')
            plt.ylabel('mean U per site')
            plt.xlim(betas[0]-.1,betas[-1]*1.05)
            figFn = os.path.join(self.outDir, figfn('potts_energy_2C_3C'))
            #print figFn
            plt.savefig(figFn)
            #plt.show()

        # assert max (d U2C(beta)) == 0.88

    def test_sw_sampling(self):
        # assert proba(site) = 1/2
        pass

    def test_gibbs(self):
        # plot nrj(beta)
        # assert max (d U(beta)) == 0.88
        pass

class PartitionFunctionTest(unittest.TestCase):

    def setUp(self):
        self.plot = config.savePlots
        self.outDir = os.path.join(config.plotSaveDir, 
                                   './PottsPartitionFunction')
        if self.plot and  not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        self.verbose = True
        
    def test_onsager1(self):
        size = 10000
        beta = .3
        pf = logpf_ising_onsager(size, beta)
        assert _np.allclose(logpf_ising_onsager(size, 0.), _np.log(2)*size)

    def test_onsager(self):
        size = 900
        dbeta = 0.001
        beta = _np.arange(0., 2., dbeta)
        pf = logpf_ising_onsager(size, beta)
        dpf = _np.diff(pf)/dbeta
        d1beta = beta[1:]  
        d2pf = _np.diff(dpf)/dbeta
        d2beta = beta[2:]

        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(beta, pf/size, label='logZ')
            plt.plot(beta[1:], dpf/size, label='dlogZ')
            plt.plot(beta[2:], d2pf/size, label='d2logZ')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('Log partition function per site and its derivatives' \
                            '.\nObtained with Onsager equations')
            figFn = os.path.join(self.outDir, figfn('logPF_onsager'))
            plt.savefig(figFn)
            #plt.show()

        #critical value:
        if self.verbose:
            print 'critical beta:', d2beta[_np.argmax(d2pf)]
            print 'beta grid precision:', dbeta

        assert _np.abs(d2beta[_np.argmax(d2pf)] - 0.88) <= 0.005
        
    def test_path_sampling(self):
        size = 900
        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        pf, beta = Cpt_Vec_Estim_lnZ_Graph(g,2)
        dbeta = beta[1]-beta[0]
        dpf = _np.diff(pf)/dbeta
        d1beta = beta[1:]  
        d2pf = _np.diff(dpf)/dbeta
        d2beta = beta[2:]
        
        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(beta, pf/size, label='logZ')
            plt.plot(beta[1:], dpf/size, label='dlogZ')
            plt.plot(beta[2:], d2pf/size, label='d2logZ')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('Log partition function per site and its derivatives' \
                            '.\nDiscretized using Path Sampling')
            #print '##### ', 
            figFn = os.path.join(self.outDir, figfn('logPF_PS'))
            plt.savefig(figFn)
            #plt.show()

        #critical value:
        if self.verbose:
            print 'critical beta:', d2beta[_np.argmax(d2pf)]
            print 'beta grid precision:', dbeta
        assert _np.abs(d2beta[_np.argmax(d2pf)] - 0.88) <= dbeta

    def test_extrapolation(self):
        size = 900
        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        pf, beta = Cpt_Vec_Estim_lnZ_Graph_fast(g,2)
        dbeta = beta[1]-beta[0]
        dpf = _np.diff(pf)/dbeta
        d1beta = beta[1:]  
        d2pf = _np.diff(dpf)/dbeta
        d2beta = beta[2:]
        
        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(beta, pf/size, label='logZ')
            plt.plot(beta[1:], dpf/size, label='dlogZ')
            plt.plot(beta[2:], d2pf/size, label='d2logZ')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('Log partition function per site and its derivatives' \
                            '.\nDiscretized using the Extrapolation Scheme.')
            figFn = os.path.join(self.outDir, figfn('logPF_ES'))
            plt.savefig(figFn)
            #plt.show()

        #critical value:
        if self.verbose:
            print 'critical beta:', d2beta[_np.argmax(d2pf)]
            print 'beta grid precision:', dbeta
        assert _np.abs(d2beta[_np.argmax(d2pf)] - 0.88) <= 2 * dbeta

    
    def test_comparison(self):
        size = 1000

        shape = (int(size**.5), int(size**.5))
        mask = _np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n,toroidal=True)

        dbeta = 0.05

        # ES
        pfES, betaES = Cpt_Vec_Estim_lnZ_Graph_fast3(g,2,BetaStep=dbeta)
        if self.verbose:
            print 'betaES:', betaES
        pfES = pfES[:-1]
        if self.verbose:
            print 'pfES:', len(pfES)
        #print pfES
        dpfES = _np.diff(pfES)/dbeta
        #print 'dpfES:'
        #print _np.diff(pfES)
        d2pfES = _np.diff(dpfES)/dbeta


        # Path Sampling
        pfPS, beta = Cpt_Vec_Estim_lnZ_Graph(g,2,BetaStep=dbeta,SamplesNb=30)
        if self.verbose:
            print 'beta grid from PS:', beta
        dpfPS = _np.diff(pfPS)/dbeta
        d1beta = beta[1:]  
        d2pfPS = _np.diff(dpfPS)/dbeta
        d2beta = beta[2:]
        

        # Onsager
        if self.verbose: print 'Onsager ...'
        pfOns = logpf_ising_onsager(size, beta)*.96
        dpfOns = _np.diff(pfOns)/dbeta
        d2pfOns = _np.diff(dpfOns)/dbeta

        
        if self.plot:
            if self.verbose: print 'Plots ...'
            import matplotlib.pyplot as plt
            # PF plots
            plt.figure()
            plt.plot(beta, pfES, 'r-+',  label='logZ-ES')
            plt.plot(beta, pfPS, 'b', label='logZ-PS')
            plt.plot(beta, pfOns,'g', label='logZ-Onsager')
            #plt.xlabel('beta')
            #plt.legend(loc='upper left')
            #plt.title('Log partition function per site - comparison')
            figFn = os.path.join(self.outDir, figfn('logPF_ES_PS_Ons'))
            print  'saved:', figFn
            plt.savefig(figFn)

            plt.figure()
            plt.plot(d1beta, dpfES/size, 'r-+',  label='dlogZ-ES')
            plt.plot(d1beta, dpfPS/size, 'b', label='dlogZ-PS')
            plt.plot(d1beta, dpfOns/size,'g', label='dlogZ-Onsager')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('dLog partition function per site - comparison')
            figFn = os.path.join(self.outDir, figfn('dlogPF_ES_PS_Ons'))
            print  'saved:', figFn
            plt.savefig(figFn)

            plt.figure()
            plt.plot(d2beta, d2pfES/size, 'r-+',  label='d2logZ-ES')
            plt.plot(d2beta, d2pfPS/size, 'b', label='d2logZ-PS')
            plt.plot(d2beta, d2pfOns/size,'g', label='d2logZ-Onsager')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('d2Log partition function per site - comparison')
            figFn = os.path.join(self.outDir, figfn('d2logPF_ES_PS_Ons'))
            print  'saved:', figFn
            plt.savefig(figFn)

            plt.figure()
            plt.plot(beta, _np.abs(pfES-pfOns)/size, 'r-+',  
                       label='|logZ_ES-logZ-Ons|')
            plt.plot(beta, _np.abs(pfPS-pfOns)/size, 'b', 
                       label='|logZ_PS-logZ-Ons|')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            plt.title('Error of Log partition function per site')
            figFn = os.path.join(self.outDir, figfn('logPF_error_ES_PS'))
            print  'saved:', figFn
            plt.savefig(figFn)

            #plt.show()



            
