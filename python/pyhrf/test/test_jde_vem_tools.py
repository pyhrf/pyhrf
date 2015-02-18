# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tempfile
import shutil
import pyhrf
import pyhrf.tools as tools
from pyhrf import FmriData
from pyhrf.jde.models import simulate_bold
import pyhrf.vbjde.vem_tools as vt
from pyhrf.boldsynth.hrf import getCanoHRF
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict
    
class VEMToolsTest(unittest.TestCase):

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
                        X,R,Det_invR,Gamma,Det_Gamma,p_Wtilde,1,
                        Q_barnCond,XGamma,0.0,0.0,K,0.05,True)
    
    
    def test_free_energy(self):
        """ Test of vem tool to compute free energy """
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

       
    def test_computeFit(self):
        X = OrderedDict([])
        #for condition,Ons in Onsets.iteritems():
        #    X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        M = 51
        N = 325
        J = 25
        Thrf=25.
        dt=.5
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_A = np.zeros((J,M),dtype=np.float64)
        m_H = np.array(m_h).astype(np.float64)
        stimIndSignal = vt.computeFit(m_H, m_A, X, J, N)
        
        
    def test_entropyA(self):
        M = 51
        J = 25
        Sigma_A = np.zeros((M,M,J),np.float64)
        entropy = vt.A_Entropy(Sigma_A, M, J)
        
        
    def test_entropyH(self):
        D = 3
        Sigma_H = np.ones((D,D),dtype=np.float64)
        entropy = vt.H_Entropy(Sigma_H, D)
        
        
    def test_entropyZ(self):
        M = 51
        K = 2
        J = 25
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        entropy = vt.Z_Entropy(q_Z, M, J)
        
        
    def test_max_mu_sigma(self):
        M = 51
        K = 2
        J = 25
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        sigma_M = np.ones((M,K),dtype=np.float64)
        sigma_M[:,0] = 0.1
        sigma_M[:,1] = 1.0
        mu_M = np.zeros((M,K),dtype=np.float64)
        Mu,Sigma = vt.maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        
        
    def test_max_L(self):
        M = 51
        N = 325
        J = 25
        TR = 1.
        Thrf=25.
        dt=.5
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_A = np.zeros((J,M),dtype=np.float64)
        m_H = np.array(m_h).astype(np.float64)
        data = self.data_simu
        Y = data.bold
        X = OrderedDict([])
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        zerosP = np.zeros((P.shape[0]),dtype=np.float64)
        L = vt.maximization_L(Y,m_A,X,m_H,L,P,zerosP)
        
        
    def test_max_sigmaH(self):
        D = 3
        Thrf=25.
        dt=.5
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_h = m_h[:D]
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        sigmaH = vt.maximization_sigmaH(D,Sigma_H,R,m_H)
    

    def test_max_sigmaH_prior(self):
        D = 3
        Thrf=25.
        dt=.5
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_h = m_h[:D]
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        gamma_h = 1000
        sigmaH = vt.maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
    
            
    def test_max_sigma_noise(self):
        M = 51
        D = 3
        N = 325
        J = 25
        TR = 1.
        Thrf=25.
        dt=.5
        data = self.data_simu
        X = OrderedDict([])
        Y = data.bold
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        PL = np.dot(P,L)
        TT,m_h = getCanoHRF(Thrf,dt) 
        sigma_epsilone = np.ones(J)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        zerosMM = np.zeros((M,M),dtype=np.float64)
        sigma_eps = vt.maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,
                                                PL,sigma_epsilone,M,zerosMM)
    
    
    def test_expectZ(self):
        M = 51
        J = 25
        K = 2
        data = self.data_simu
        graph = data.get_graph()
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        mu_M = np.zeros((M,K),dtype=np.float64)
        sigma_M = np.ones((M,K),dtype=np.float64)
        beta=.8
        Beta = beta * np.ones((M),dtype=np.float64)
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        Z_tilde = q_Z.copy()
        zerosK = np.zeros(K)
        q_Z, Z_tilde = vt.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,
                                        mu_M,q_Z,graph,M,J,K,zerosK)


    def test_expectH(self):
        M = 51
        K = 2
        J = 25
        N = 325
        D = 3
        TR = 1.
        Thrf=25.
        dt=.5
        data = self.data_simu
        Gamma = np.identity(N)
        X = OrderedDict([])
        Y = data.bold
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        PL = np.dot(P,L)
        y_tilde = Y - np.dot(P,L)
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_h = m_h[:D]
        m_H = np.array(m_h)
        sigma_epsilone = np.ones(J)
        Sigma_H = np.ones((D,D),dtype=float)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        scale=1
        zerosDD = np.zeros((D,D),dtype=np.float64)
        zerosD = np.zeros((D),dtype=np.float64)
        zerosND = np.zeros((N,D),dtype=np.float64)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        sigmaH = 0.1
        Sigma_H, m_H = vt.expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,
                                        sigmaH,J,N,y_tilde,zerosND,
                                        sigma_epsilone,scale,zerosDD,
                                        zerosD)
    
    
    def test_expectA(self):
        M = 51
        K = 2
        J = 25
        N = 325
        D = 3
        TR = 1.
        Thrf=25.
        dt=.5
        data = self.data_simu
        Y = data.bold
        Onsets = data.get_joined_onsets()
        Gamma = np.identity(N)
        X = OrderedDict([])
        for condition,Ons in Onsets.iteritems():
            X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)
        PL = np.dot(P,L)
        y_tilde = Y - np.dot(P,L)
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_h = m_h[:D]
        sigma_epsilone = np.ones(J)
        m_H = np.array(m_h)
        Sigma_H = np.ones((D,D),dtype=float)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.ones((M,M,J),np.float64)
        for j in xrange(0,J):
            Sigma_A[:,:,j] = 0.01*np.identity(M)
        mu_M = np.zeros((M,K),dtype=np.float64)
        sigma_M = np.ones((M,K),dtype=np.float64)
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        zerosJMD = np.zeros((J,M,D),dtype=np.float64)
        Sigma_A, m_A = vt.expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,
                                        sigma_M,q_Z,mu_M,D,N,J,M,K,
                                        y_tilde,Sigma_A,sigma_epsilone,
                                        zerosJMD)
       
      
    def test_matrix(self):
        Thrf=25.
        dt=.5
        TT,m_h = getCanoHRF(Thrf,dt) 
        m_H = np.array(m_h)
        m = vt.mult(m_H,m_H) 
        #matrix = mult(v1,v2)
    
    
    def test_maximum(self):
        M = 51
        K = 2
        vector = np.ones((K,1),dtype=np.float64)
        vector[0] = 2
        maxs, maxs_ind = vt.maximum(vector)
    
    
    def test_normpdf(self):
        y = vt.normpdf(1, 1, 1)
        #y = normpdf(x, mu, sigma)
    
    
    def test_polyFit(self):
        N = 325
        TR = 1.
        data = self.data_simu
        Y = data.bold
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
    
    
    def test_PolyMat(self):  
        N = 325
        TR = 1.
        P = vt.PolyMat( N , 4 , TR)
        
    
    def test_compute_mat_X2(self):  
        N = 325
        TR = 1.
        D = 3
        dt = .5
        data = self.data_simu
        X = OrderedDict([])
        Onsets = data.get_joined_onsets()
        for condition,Ons in Onsets.iteritems():
            X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
     
    
    def test_buildFiniteDiffMatrix(self):
        order = 2
        D = 3
        diffMat = vt.buildFiniteDiffMatrix(order, D)
            
           
    def test_gradient(self):
        beta=.8
        gamma = 7.5
        m = 0
        J = 25
        K = 2
        M = 51
        data = self.data_simu
        graph = data.get_graph()
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        Z_tilde = q_Z.copy()
        Gr = vt.gradient(q_Z,Z_tilde,J,m,K,graph,beta,gamma)


    def test_max_beta(self):
        beta=.8
        gamma = 7.5
        m = 0
        J = 25
        K = 2
        M = 51
        data = self.data_simu
        graph = data.get_graph()
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        Z_tilde = q_Z.copy()
        maxNeighbours = max([len(nl) for nl in graph])
        neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
        neighboursIndexes -= 1
        for i in xrange(J):
            neighboursIndexes[i,:len(graph[i])] = graph[i]
        beta = vt.maximization_beta(beta,q_Z,Z_tilde,J,K,m,graph,gamma,
                                    neighboursIndexes,maxNeighbours)

