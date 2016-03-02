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
from pyhrf.vbjde import UtilsC
from pyhrf.ui.treatment import FMRITreatment
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
        self.clean_tmp = True
        simu = simulate_bold(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

    def tearDown(self):
        if self.clean_tmp:
            logger.info('Remove tmp dir %s', self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            logger.info('Keep tmp dir %s', self.tmp_dir)

    def test_max_L(self):
        M = 51
        N = 325
        J = 25
        D = 3
        TR = 1.
        Thrf = 25.
        dt = .5
        TT, m_h = getCanoHRF(Thrf, dt)
        m_A = np.zeros((J, M), dtype=np.float64)
        m_H = np.array(m_h).astype(np.float64)
        data = self.data_simu
        Y = data.bold
        XX = np.zeros((M, N, D), dtype=np.int32)
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)
        Ndrift = L.shape[0]
        zerosP = np.zeros((P.shape[0]), dtype=np.float64)
        UtilsC.maximization_L(
            Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M, Ndrift, N)

    def test_max_sigma_noise(self):
        M = 51
        D = 3
        N = 325
        J = 25
        TR = 1.
        Thrf = 25.
        dt = .5
        Gamma = np.identity(N)
        data = self.data_simu
        XX = np.zeros((M, N, D), dtype=np.int32)
        Y = data.bold
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)
        PL = np.dot(P, L)
        TT, m_h = getCanoHRF(Thrf, dt)
        sigma_epsilone = np.ones(J)
        m_A = np.zeros((J, M), dtype=np.float64)
        Sigma_A = np.zeros((M, M, J), np.float64)
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D, D), dtype=np.float64)
        UtilsC.maximization_sigma_noise(Gamma, PL, sigma_epsilone,
                                        Sigma_H, Y, m_A, m_H, Sigma_A,
                                        XX.astype(np.int32), J, D, M, N)

    def test_expectZ(self):
        M = 51
        J = 25
        K = 2
        data = self.data_simu
        graph = data.get_graph()
        m_A = np.zeros((J, M), dtype=np.float64)
        Sigma_A = np.zeros((M, M, J), np.float64)
        mu_M = np.zeros((M, K), dtype=np.float64)
        sigma_M = np.ones((M, K), dtype=np.float64)
        beta = .8
        Beta = beta * np.ones((M), dtype=np.float64)
        q_Z = np.zeros((M, K, J), dtype=np.float64)
        Z_tilde = q_Z.copy()
        maxNeighbours = max([len(nl) for nl in graph])
        neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
        neighboursIndexes -= 1
        for i in xrange(J):
            neighboursIndexes[i, :len(graph[i])] = graph[i]
        UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z,
                             neighboursIndexes.astype(np.int32), M, J, K,
                             maxNeighbours)

    def test_expectH(self):
        M = 51
        J = 25
        N = 325
        D = 3
        TR = 1.
        Thrf = 25.
        dt = .5
        data = self.data_simu
        Gamma = np.identity(N)
        Q_barnCond = np.zeros((M, M, D, D), dtype=np.float64)
        XGamma = np.zeros((M, D, N), dtype=np.float64)
        XX = np.zeros((M, N, D), dtype=np.int32)
        Y = data.bold
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)
        PL = np.dot(P, L)
        y_tilde = Y - np.dot(P, L)
        TT, m_h = getCanoHRF(Thrf, dt)
        m_h = m_h[:D]
        m_H = np.array(m_h)
        sigma_epsilone = np.ones(J)
        Sigma_H = np.ones((D, D), dtype=float)
        m_A = np.zeros((J, M), dtype=np.float64)
        Sigma_A = np.zeros((M, M, J), np.float64)
        scale = 1
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order, D)
        R = np.dot(D2, D2) / pow(dt, 2 * order)
        sigmaH = 0.1
        UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma,
                             R, Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), J, D, M, N, scale, sigmaH)

    def test_expectA(self):
        M = 51
        K = 2
        J = 25
        N = 325
        D = 3
        TR = 1.
        Thrf = 25.
        dt = .5
        data = self.data_simu
        Y = data.bold
        Onsets = data.get_joined_onsets()
        Gamma = np.identity(N)
        XX = np.zeros((M, N, D), dtype=np.int32)
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)
        PL = np.dot(P, L)
        y_tilde = Y - np.dot(P, L)
        TT, m_h = getCanoHRF(Thrf, dt)
        m_h = m_h[:D]
        sigma_epsilone = np.ones(J)
        m_H = np.array(m_h)
        Sigma_H = np.ones((D, D), dtype=float)
        m_A = np.zeros((J, M), dtype=np.float64)
        Sigma_A = np.zeros((M, M, J), np.float64)
        for j in xrange(0, J):
            Sigma_A[:, :, j] = 0.01 * np.identity(M)
        mu_M = np.zeros((M, K), dtype=np.float64)
        sigma_M = np.ones((M, K), dtype=np.float64)
        q_Z = np.zeros((M, K, J), dtype=np.float64)
        UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                             Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), J, D, M, N, K)
