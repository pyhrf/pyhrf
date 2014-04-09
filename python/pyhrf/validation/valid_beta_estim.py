# -*- coding: utf-8 -*-
import os, sys
import unittest
import numpy as np

from pyhrf.jde.beta import *
from pyhrf.boldsynth.field import genPotts #, count_homo_cliques
from pyhrf.graph import *
from pyhrf.boldsynth.scenarios import *

from pyhrf.tools import cached_eval

from pyhrf.validation import config
from pyhrf.validation.config import figfn

from pyhrf.boldsynth.pottsfield.swendsenwang import *
from pyhrf.ndarray import stack_cuboids

def dist(x,y):
    return ((x-y)**2).sum()**.5


def beta_estim_obs_field_mc(graph, nbClasses, beta, gridLnz, mcit=1, 
                            cachePotts=False):

    cumulBetaEstim = 0.
    cumul2BetaEstim = 0.

    for mci in xrange(mcit):
        labels = cached_eval(genPotts, (graph, beta, nbClasses), 
                             save=cachePotts, new=(not cachePotts))
        print 'labels:',labels
        betaEstim, _ = beta_estim_obs_field(graph, labels, gridLnz,
                                            method='ML')
        #betaEstim, pb = beta_estim_obs_field(graph, labels, gridLnz,
        #                                     method='MAP')
        print ' -mc- be=%f' %betaEstim
        cumulBetaEstim += betaEstim
        cumul2BetaEstim += betaEstim**2

    meanBetaEstim = cumulBetaEstim / mcit
    stdBetaEstim = ((cumul2BetaEstim / mcit) - meanBetaEstim**2)**.5

    return meanBetaEstim, stdBetaEstim

def test_beta_estim_obs_fields(graphs, betas, nbLabels, pfmethod, mcit=1):
    
    nbGraphs = len(graphs)
    meanBetaEstims = np.zeros(len(betas))
    stdBetaEstims = np.zeros(len(betas))
    stdMCBetaEstims = np.zeros(len(betas))
    gridLnz = range(nbGraphs)
    if mcit == 1:
        cachePotts = True
    else:
        cachePotts = False
        
    # compute partition functions:
    print 'compute partition functions ... nbgraphs:', nbGraphs
    print 'Method =', pfmethod
    print 'nbLabels =', nbLabels
    if pfmethod == 'ES':
        for ig, g in enumerate(graphs):
            grid = cached_eval(Cpt_Vec_Estim_lnZ_Graph_fast3, (g,nbLabels),
                               path=config.cacheDir, prefix='pyhrf_')
            gridLnz[ig] = (grid[0][:-1], grid[1][:-1])    
    elif pfmethod == 'PS':
        for ig, g in enumerate(graphs):
            gridLnz[ig] = cached_eval(Cpt_Vec_Estim_lnZ_Graph, (g,nbLabels),
                                      path=config.cacheDir, new=False,
                                      prefix='pyhrf_')
    elif pfmethod == 'ONSAGER':
        for ig, g in enumerate(graphs):
            betaGrid = cached_eval(Cpt_Vec_Estim_lnZ_Graph,([[]],nbLabels),
                                   path=config.cacheDir)[1]
            lpf = cached_eval(logpf_ising_onsager, (len(g),betaGrid),
                              path=config.cacheDir, prefix='pyhrf_')
            gridLnz[ig] = (lpf, betaGrid)
    else:
        print 'unknown pf method!'
        sys.exit(1)

    print 'gridLnz:'
    print gridLnz

    # Beta estimation:
    print 'beta estimation test loop ...'
    for ib, b in enumerate(betas):
        cumulBetaEstim = 0.
        cumul2BetaEstim = 0.
        cumulStdMC = 0.
        print 'testing with beta =', b, '...'
        for ig, graph in enumerate(graphs):
            mbe, stdmc = beta_estim_obs_field_mc(graph, nbLabels, b,
                                                 gridLnz[ig], mcit, 
                                                 cachePotts=False)
            print '%f(%f)' %(mbe,stdmc)
            cumulBetaEstim += mbe
            cumulStdMC += stdmc
            cumul2BetaEstim += mbe**2
            #print betaEstim, cumulBetaEstim, cumul2BetaEstim

        meanBetaEstims[ib] = cumulBetaEstim / nbGraphs
        stdBetaEstims[ib] = ((cumul2BetaEstim / nbGraphs) - \
                                 meanBetaEstims[ib]**2)**.5
        stdMCBetaEstims[ib] = cumulStdMC / nbGraphs
        #print '-> ', meanBetaEstims[ib], stdBetaEstims[ib]
    return meanBetaEstims, stdBetaEstims, stdMCBetaEstims



class ObsField2DTest(unittest.TestCase):
    """ Test estimation of beta with on observed 2D fields
    """

    def setUp(self):
        self.outDir = os.path.join(config.plotSaveDir, 
                                   './BetaEstimation')
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        self.plot = config.savePlots

        shape = (15,15)
        mask = np.ones(shape, dtype=int) #full mask
        self.g = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)
        
    def test_single_PFES_MAP(self):
        """PF estimation method : extrapolation scheme. MAP on p(beta|label).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
	nbTests = 30
	#print 'g:', self.g
        
	# partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph_fast3(self.g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
	#print 'generating potts ..., beta =', beta
	bes = np.zeros(nbTests)
        for t in xrange(nbTests):
	    labels = genPotts(self.g, beta, nbClasses)
        
	    # beta estimation
	    be, pb = beta_estim_obs_field(self.g, labels, gridLnz, 'MAP')
	    print 'betaMAP:', be
	    bes[t] = be
	print 'mean betaMAP:', bes.mean(), bes.std()
        assert abs(bes.mean()-beta) <= gridPace


    def test_single_PFES_ML(self):
        """PF estimation method : extrapolation scheme. ML on p(label|beta).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta
        labels = genPotts(self.g, beta, nbClasses)
        # partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph_fast3(self.g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
        # beta estimation
        be, pb = beta_estim_obs_field(self.g, labels, gridLnz, 'ML')
        print 'betaML:', be
        assert abs(be-beta) <= gridPace


    def test_single_PFPS_MAP(self):
        """PF estimation method : path sampling. MAP on p(beta|label).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta
        labels = genPotts(self.g, beta, nbClasses)
        # partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(self.g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
        # beta estimation
        be, pb = beta_estim_obs_field(self.g, labels, gridLnz, 'MAP')
        print 'betaMAP:', be
        assert abs(be-beta) <= gridPace


    def test_single_PFPS_ML(self):
        """PF estimation method : path sampling. ML on p(label|beta).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta
        labels = genPotts(self.g, beta, nbClasses)
        # partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(self.g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
        # beta estimation
        be, pb = beta_estim_obs_field(self.g, labels, gridLnz, 'ML')
        print 'betaML:', be
        assert abs(be-beta) <= gridPace


    def test_single_surface_PFPS_ML(self):
        """PF estimation method : path sampling. ML on p(label|beta).
        topology from a surfacic RDI
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta

        
        # grab surfacic data:
        from pyhrf.graph import graph_from_mesh, sub_graph, graph_is_sane
        from pyhrf.tools.io.tio import Texture
        from soma import aims
        print 'import done'
        roiId = 20
        mfn = pyhrf.get_data_file_name('right_hemisphere.mesh')
        print 'mesh file:', mfn
        mesh = aims.read(mfn)
        print 'mesh read'
        triangles = [t.arraydata() for t in mesh.polygon().list()]
        print 'building graph ... '
        wholeGraph = graph_from_mesh(triangles)
        
        roiMaskFile = pyhrf.get_data_file_name('roimask_gyrii_tuned.tex')
        roiMask = Texture.read(roiMaskFile).data.astype(int)
        mroi = np.where(roiMask==roiId)
        g, nm = sub_graph(wholeGraph, mroi[0])
        print "g:", len(g), len(g[0])

        nnodes = len(g)
        points = np.vstack([v.arraydata() for v in mesh.vertex().list()])
        weights = [[1./dist(points[j],points[k]) for k in g[j]]
                   for j in xrange(nnodes)]
        print "weights:", len(weights), len(weights[0])
        
        if 1:
            for j in xrange(nnodes):
                s = sum(weights[j]) * 1.
                for k in xrange(len(weights[j])):
                    weights[j][k] = weights[j][k]/s * len(weights[j])

        labels = genPotts(g, beta, nbClasses, weights=weights)
        print labels
        # partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses, 
                                          GraphWeight=weights)

        print 'gridLnz with weights:'
        print gridLnz

        # beta estimation
        be, pb = beta_estim_obs_field(g, labels, gridLnz, 'ML', weights)
        print 'betaML:', be

        weights = None
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses, 
                                          GraphWeight=weights)

        
        print 'gridLnz without weights:'
        print gridLnz

        # beta estimation
        be, pb = beta_estim_obs_field(g, labels, gridLnz, 'ML', weights)
        print 'betaML:', be

        gridPace = gridLnz[1][1] - gridLnz[1][0]
        assert abs(be-beta) <= gridPace


    def test_single_Onsager_ML(self):
        """PF method: Onsager. ML on p(beta|label).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta
        labels = genPotts(self.g, beta, nbClasses)
        # partition function estimation
        dbeta = 0.05
        betaGrid = np.arange(0, 1.5, dbeta)
        lpf = logpf_ising_onsager(labels.size, betaGrid)
        #lpf, betaGrid = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses)
        print 'lpf:'
        print lpf
        betaML = beta_estim_obs_field(self.g, labels, (lpf, betaGrid), 'ML')
        print 'betaML:', betaML

    def test_single_Onsager_MAP(self):
        """PF method: Onsager. MAP on p(label|beta).
        """
        # generate a field:
        beta = 0.4
        nbClasses = 2
        print 'generating potts ..., beta =', beta
        labels = genPotts(self.g, beta, nbClasses)
        # partition function estimation
        dbeta = 0.05
        betaGrid = np.arange(0, 1.5, dbeta)
        lpf = logpf_ising_onsager(labels.size, betaGrid)
        #lpf, betaGrid = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses)
        print 'lpf:'
        print lpf
        betaMAP = beta_estim_obs_field(self.g, labels, (lpf, betaGrid), 'MAP')
        print 'betaMAP:', betaMAP


    def test_MC_comp_pfmethods_ML_10x10(self):
        self.MC_comp_pfmethods_ML((10,10))

    def test_MC_comp_pfmethods_ML_30x30(self):
        self.MC_comp_pfmethods_ML((30,30))

    def test_MC_comp_pfmethods_ML_3C_50x50(self):
        self.MC_comp_pfmethods_ML_3C((50,50))

    def test_MC_comp_pfmethods_ML_3C_10x10(self):
        self.MC_comp_pfmethods_ML_3C((10,10))

    def test_MC_comp_pfmethods_ML_3C_20x20(self):
        self.MC_comp_pfmethods_ML_3C((20,20))

    def test_MC_comp_pfmethods_ML_100x100(self):
        self.MC_comp_pfmethods_ML((100, 100))
        
    def test_MC_comp_pfmethods_ML_3C_30x30(self):
        self.MC_comp_pfmethods_ML_3C((30, 30))        

    def MC_comp_pfmethods_ML(self, shape):

        mask = np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)

        # Betas to iterate over ...
        betaMax = 1.4
        dbeta = .1
        betas = np.arange(0., betaMax, dbeta)

        # nb of monte carlo iterations
        mcit = 40#100
        
        # nb of classes
        nbc = 2

        # Path sampling:
        mBePS, stdBePS, stdBeMCPS = cached_eval(test_beta_estim_obs_fields,
                                                ([g], betas, nbc, 'PS',
                                                 mcit), path=config.cacheDir,
                                                prefix='pyhrf_',
                                                new=False)
        
        # Extrapolation scheme:
        mBeES, stdBeES, stdBeMCES = cached_eval(test_beta_estim_obs_fields,
                                                ([g], betas, nbc, 'ES',
                                                 mcit), path=config.cacheDir,
                                                prefix='pyhrf_',
                                                new=False)

        # Onsager:
        #mBeON, stdBeON, stdBeMCON = cached_eval(test_beta_estim_obs_fields,
        #                                        ([g], betas, nbc, 'ONSAGER',
        #                                         mcit), path=config.cacheDir,
        #                                        prefix='pyhrf_',
        #                                        new=True)
        
        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure()

            plt.errorbar(betas, mBeES, stdBeMCES, fmt='r', label='ES')
            plt.errorbar(betas, mBePS, stdBeMCPS, fmt='b', label='PS')
            #plt.errorbar(betas, mBeON, stdBeMCON, fmt='g', label='Onsager')
            plt.plot(betas, betas, 'k', label='true')

            #plt.xlabel('simulated beta')
            #plt.ylabel('beta MAP')
            plt.legend(loc='upper left')
            plt.ylim((0.0,1.45))
            #plt.title('MC validation (%d it) for beta estimation on '\
            #                ' observed fields.' %mcit)
            #fn = 'comp_obs_field_MC_PS_ES_Onsager_%dx%d' %shape 
            fn = 'comp_obs_field_MC_PS_%dx%d' %shape 
            figFn = os.path.join(self.outDir, figfn(fn))
            plt.savefig(figFn)



    def MC_comp_pfmethods_ML_3C(self, shape):

        mask = np.ones(shape, dtype=int) #full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)

        # Betas to iterate over ...
        betaMax = 1.4
        dbeta = .1
        betas = np.arange(0., betaMax, dbeta)

        # nb of monte carlo iterations
        mcit = 100#100
        
        # nb of classes
        nbc = 3

        # Path sampling:
        mBePS, stdBePS, stdBeMCPS = cached_eval(test_beta_estim_obs_fields,
                                                ([g], betas, nbc, 'PS',
                                                 mcit), path=config.cacheDir,
                                                prefix='pyhrf_',
                                                new=False)
        
        # Extrapolation scheme:
        mBeES, stdBeES, stdBeMCES = cached_eval(test_beta_estim_obs_fields,
                                                ([g], betas, nbc, 'ES',
                                                 mcit), path=config.cacheDir,
                                                prefix='pyhrf_',
                                                new=False)

        # Onsager:
        #mBeON, stdBeON, stdBeMCON = cached_eval(test_beta_estim_obs_fields,
        #                                        ([g], betas, nbc, 'ONSAGER',
        #                                         mcit), path=config.cacheDir,
        #                                        prefix='pyhrf_',
        #                                        new=True)
        
        if self.plot:
            import plt
            plt.figure()

            plt.errorbar(betas, mBeES, stdBeMCES, fmt='r', label='ES')
            plt.errorbar(betas, mBePS, stdBeMCPS, fmt='b', label='PS')
            #plt.errorbar(betas, mBeON, stdBeMCON, fmt='g', label='Onsager')
            plt.plot(betas, betas, 'k', label='true')
            #plt.xlabel('simulated beta')
            #plt.ylabel('beta MAP')
            plt.legend(loc='upper left')
            #plt.title('MC validation (%d it) for beta estimation on '\
            #                ' observed fields.' %mcit)
            fn = 'comp_obs_field_3class_MC_PS_ES_%dx%d' %shape 
            figFn = os.path.join(self.outDir, figfn(fn))
            plt.savefig(figFn)
            
            
            plt.figure()
            grid = cached_eval(Cpt_Vec_Estim_lnZ_Graph_fast3, (g,nbc),
                               path=config.cacheDir, prefix='pyhrf_')
            lnzES, betas = (grid[0][:-1], grid[1][:-1])    
            lnzPS, betas = cached_eval(Cpt_Vec_Estim_lnZ_Graph, (g,nbc),
                                       path=config.cacheDir, new=False,
                                       prefix='pyhrf_')

            plt.plot(betas, lnzES, 'r', label='ES')
            plt.plot(betas, lnzPS, 'b', label='PS')
            fn = 'comp_PF_3class_PS_ES_%dx%d' %shape 
            figFn = os.path.join(self.outDir, figfn(fn))
            plt.savefig(figFn)
