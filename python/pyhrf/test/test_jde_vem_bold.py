# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tempfile
import shutil
import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.jde.models import simulate_bold
from pyhrf.ui.vb_jde_analyser_compMod import JDEVEMAnalyser
from pyhrf.vbjde.vem_bold_constrained import Main_vbjde_Extension_constrained
import pyhrf.vbjde.vem_tools as vt
from pyhrf.boldsynth.hrf import getCanoHRF
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict
    
class VEMBOLDTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir
        simu = simulate_bold(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

        
    def tearDown(self):
        if 1:
            pyhrf.verbose(1, 'Remove tmp dir %s' %self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            pyhrf.verbose(1, 'Keep tmp dir %s' %self.tmp_dir)
       
        
    def test_jdevemanalyser(self):
        """ Test BOLD VEM sampler on small simulation with small 
        nb of iterations. Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)
        jde_vem_analyser = JDEVEMAnalyser(beta=.8, dt=.5, hrfDuration=25.,
                                        nItMax=2,nItMin=2, fast = True, 
                                        computeContrast = False, PLOT=False,
                                        constrained = True)
        tjde_vem = FMRITreatment(fmri_data=self.data_simu, 
                                analyser=jde_vem_analyser, 
                                output_dir=None)
        tjde_vem.run()  
        
        
    def test_vem_bold_constrained(self):
        """ Test BOLD VEM constraint function.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()
        
        NbIter, nrls, estimated_hrf, \
        labels, noiseVar, mu_k, sigma_k, \
        Beta, L, PL, CONTRAST, CONTRASTVAR, \
        cA,cH,cZ,cAH,cTime,cTimeMean, \
        Sigma_nrls, StimuIndSignal,\
        FreeEnergy = Main_vbjde_Extension_constrained(graph,data.bold,Onsets,
                                                      Thrf=25.,K=2,TR=1.,
                                                      beta=1.0,dt=.5, 
                                                      NitMax=2,NitMin=2)

    def test_minivem(self):
        """ Test BOLD VEM constraint function.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)

        data = self.data_simu
        Y = data.bold
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()
        S = 100
        Thrf=25.
        dt=.5
        TR=1.
        D = int(np.ceil(Thrf/dt)) + 1 #D = int(numpy.ceil(Thrf/dt)) 
        M = len(Onsets)
        N = Y.shape[0]
        J = Y.shape[1]
        K = 2
        maxNeighbours = max([len(nl) for nl in graph])
        neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
        XX = np.zeros((M,N,D),dtype=np.int32)
        X = OrderedDict([])
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        invR = np.linalg.inv(R)
        Det_invR = np.linalg.det(invR)
        Gamma = np.identity(N)
        Det_Gamma = np.linalg.det(Gamma)
        Q_barnCond = np.zeros((M,M,D,D),dtype=np.float64)
        XGamma = np.zeros((M,D,N),dtype=np.float64)  
        p_Wtilde = np.zeros((M,K),dtype=np.float64)
        p_Wtilde1 = np.zeros((M,K),dtype=np.float64)
        p_Wtilde[:,1] = 1
    
        vt.MiniVEM_CompMod(Thrf,TR,dt,1.0,data.bold,2,7.5,0.003,200,
                        D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,
                        X,R,Det_invR,Gamma,Det_Gamma,p_Wtilde,1,Q_barnCond,
                        XGamma,0.0,0.0,K,0.05,True)
        """vt.MiniVEM_CompMod(Thrf,TR,dt,beta,Y,K,gamma,gradientStep,
                           MaxItGrad,D,M,N,J,S,maxNeighbours,
                           neighboursIndexes,XX,X,R,Det_invR,Gamma,
                           Det_Gamma,p_Wtilde,scale,Q_barnCond,XGamma,
                           tau1,tau2,Nit,sigmaH,True)"""
    
    
    def test_free_energy(self):
        """ Test of vem tool to compute free energy
        """
        M = 51
        D = 3
        N = 325
        J = 25
        K = 2
        TR = 1.
        Thrf=25.
        dt=.5
        
        data = self.data_simu
        Y = data.bold
        graph = data.get_graph()
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        y_tilde = Y - np.dot(P,L)
        
        TT,m_h = getCanoHRF(Thrf,dt) 
        
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        invR = np.linalg.inv(R)
        Det_invR = np.linalg.det(invR)
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        maxNeighbours = max([len(nl) for nl in graph])
        neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
        Beta = np.ones((M),dtype=np.float64)
        sigma_epsilone = np.ones(J)
        XX = np.zeros((M,N,D),dtype=np.int32)
        Gamma = np.identity(N)
        Det_Gamma = np.linalg.det(Gamma)
        XGamma = np.zeros((M,D,N),dtype=np.float64)

        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        mu_M = np.zeros((M,K),dtype=np.float64)
        sigma_M = np.ones((M,K),dtype=np.float64)
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        p_Wtilde = np.zeros((M,K),dtype=np.float64)
        p_Wtilde1 = np.zeros((M,K),dtype=np.float64)
        p_Wtilde[:,1] = 1
        
        FreeEnergy = vt.Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,
                                           m_H,Sigma_H,R,Det_invR,0.0,p_Wtilde,
                                           0.0,0.0,q_Z,neighboursIndexes,
                                           maxNeighbours,Beta,sigma_epsilone,
                                           XX,Gamma,Det_Gamma,XGamma,
                                           J,M,D,N,2,100,"CompMod")
        
        