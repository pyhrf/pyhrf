# -*- coding: utf-8 -*-

import unittest
import tempfile
import shutil
import logging

import numpy as np

import pyhrf
import pyhrf.tools as tools
import pyhrf.vbjde.vem_tools as vt

from pyhrf import FmriData
from pyhrf.jde.models import simulate_bold
from pyhrf.boldsynth.hrf import getCanoHRF
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


logger = logging.getLogger(__name__)


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
            logger.info('Remove tmp dir %s', self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            logger.info('Keep tmp dir %s', self.tmp_dir)


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
        gamma_h = 1000
        data = self.data_simu
        Y = data.bold
        graph = data.get_graph()
        onsets = data.paradigm.get_joined_onsets()
        durations = data.paradigm.stimDurations
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        y_tilde = Y - np.dot(P,L)
        TT,m_h = getCanoHRF(Thrf,dt)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order,D)
        R = np.dot(D2,D2) / pow(dt,2*order)
        invR = np.linalg.inv(R)
        Det_invR = np.linalg.det(invR)
        q_Z = 0.5 * np.ones((M,K,J),dtype=np.float64)
        neighbours_indexes = vt.create_neighbours(graph)
        Beta = np.ones((M),dtype=np.float64)
        sigma_epsilone = np.ones(J)
        _, occurence_matrix, _ = vt.create_conditions(onsets, durations, M, N, D, TR, dt)
        Gamma = np.identity(N)
        Det_Gamma = np.linalg.det(Gamma)
        XGamma = np.zeros((M,D,N),dtype=np.float64)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        mu_M = np.zeros((M,K),dtype=np.float64)
        sigma_M = np.ones((M,K),dtype=np.float64)
        m_H = np.array(m_h[:D]).astype(np.float64)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        free_energy = vt.free_energy_computation(m_A, Sigma_A, m_H, Sigma_H, D,
                                                 q_Z, y_tilde, occurence_matrix,
                                                 sigma_epsilone, Gamma, M, J, N,
                                                 K, mu_M, sigma_M, neighbours_indexes,
                                                 Beta, Sigma_H, np.linalg.inv(R),
                                                 R, Det_Gamma, gamma_h)


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
        entropy = vt.nrls_entropy(Sigma_A, M)


    def test_entropyH(self):
        D = 3
        Sigma_H = np.ones((D,D),dtype=np.float64)
        entropy = vt.hrf_entropy(Sigma_H, D)


    def test_entropyZ(self):
        M = 51
        K = 2
        J = 25
        q_Z = np.zeros((M,K,J),dtype=np.float64)
        entropy = vt.labels_entropy(q_Z)


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
        Mu,Sigma = vt.maximization_class_proba(q_Z, m_A, Sigma_A)


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
        onsets = data.paradigm.get_joined_onsets()
        durations = data.paradigm.get_joined_durations()
        Y = data.bold
        X = OrderedDict([])
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        zerosP = np.zeros((P.shape[0]),dtype=np.float64)
        _, occurence_matrix, _ = vt.create_conditions(onsets, durations, M, N,
                                                      len(m_H), TR, dt)
        L = vt.maximization_drift_coeffs(Y, m_A, occurence_matrix, m_H, np.identity(N), P)


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
        N = 325
        J = 25
        TR = 1.
        Thrf=25.
        dt=.5
        data = self.data_simu
        X = OrderedDict([])
        Y = data.bold
        onsets = data.get_joined_onsets()
        durations = data.paradigm.stimDurations
        P = vt.PolyMat( N , 4 , TR)
        L = vt.polyFit(Y, TR, 4,P)
        PL = np.dot(P,L)
        TT,m_h = getCanoHRF(Thrf,dt)
        sigma_epsilone = np.ones(J)
        Gamma = np.identity(N)
        m_A = np.zeros((J,M),dtype=np.float64)
        Sigma_A = np.zeros((M,M,J),np.float64)
        m_H = np.array(m_h).astype(np.float64)
        D = len(m_H)
        Sigma_H = np.ones((D,D),dtype=np.float64)
        zerosMM = np.zeros((M,M),dtype=np.float64)
        _, occurence_matrix, _ = vt.create_conditions(onsets, durations, M, N,
                                                      D, TR, dt)
        sigma_eps = vt.maximization_noise_var(occurence_matrix, m_H, Sigma_H, m_A, Sigma_A,
                                              Gamma, Y, N)


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
        neighbours_indexes = vt.create_neighbours(graph)
        q_Z = vt.labels_expectation(Sigma_A, m_A, sigma_M, mu_M, Beta, q_Z,
                                    neighbours_indexes, M, K)


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
        onsets = data.get_joined_onsets()
        durations = data.paradigm.stimDurations
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
        _, occurence_matrix, _ = vt.create_conditions(onsets, durations, M, N,
                                                      D, TR, dt)
        m_H, Sigma_H = vt.hrf_expectation(Sigma_A, m_A, occurence_matrix, Gamma, R,
                                          sigmaH, J, y_tilde, sigma_epsilone)


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
        durations = data.paradigm.stimDurations
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
        q_Z = 0.5 * np.ones((M,K,J),dtype=np.float64)
        zerosJMD = np.zeros((J,M,D),dtype=np.float64)
        _, occurence_matrix, _ = vt.create_conditions(Onsets, durations, M, N,
                                                      D, TR, dt)
        m_A, Sigma_A = vt.nrls_expectation(m_H, m_A, occurence_matrix, Gamma,
                                           q_Z, mu_M, sigma_M, M, y_tilde, Sigma_A,
                                           Sigma_H, sigma_epsilone)


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


    def test_create_neighbours(self):
        graph = self.data_simu.get_graph()
        neighbours_indexes = vt.create_neighbours(graph)


    def test_create_conditions(self):
        nb_conditions = self.data_simu.nbConditions
        nb_scans = 325
        hrf_len = 3
        tr = 1.
        dt = .5
        onsets = self.data_simu.get_joined_onsets()
        durations = self.data_simu.paradigm.stimDurations
        X, occurence_matrix, condition_names = vt.create_conditions(
            onsets, durations, nb_conditions, nb_scans, hrf_len, tr, dt
        )

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
        neighbours_indexes = vt.create_neighbours(graph)
        labels_neigh = vt.sum_over_neighbours(neighbours_indexes, q_Z)
        Gr = vt.beta_gradient(beta, q_Z, labels_neigh, neighbours_indexes, gamma)


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
        neighbours_indexes = vt.create_neighbours(graph)
        beta = vt.beta_maximization(beta, q_Z, neighbours_indexes, gamma)

