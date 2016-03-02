# -*- coding: utf-8 -*-

import unittest
import tempfile
import shutil
import logging

import numpy as np

import pyhrf
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
        self.clean_tmp = True
        simu = simulate_bold(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

    def tearDown(self):
        if self.clean_tmp:
            logger.info('Remove tmp dir %s', self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            logger.info('Keep tmp dir %s' % self.tmp_dir)

    def test_computeFit(self):
        X = OrderedDict([])
        M = 51
        N = 325
        J = 25
        Thrf = 25.
        dt = .5
        TT, m_h = getCanoHRF(Thrf, dt)
        m_A = np.zeros((J, M), dtype=np.float64)
        m_H = np.array(m_h).astype(np.float64)
        stimIndSignal = vt.computeFit(m_H, m_A, X, J, N)

    def test_entropyA(self):
        M = 51
        J = 25
        Sigma_A = np.zeros((M, M, J), np.float64)
        entropy = vt.A_Entropy(Sigma_A, M, J)

    def test_entropyH(self):
        D = 3
        Sigma_H = np.ones((D, D), dtype=np.float64)
        entropy = vt.H_Entropy(Sigma_H, D)

    def test_entropyZ(self):
        M = 51
        K = 2
        J = 25
        q_Z = np.zeros((M, K, J), dtype=np.float64)
        entropy = vt.Z_Entropy(q_Z, M, J)

    def test_max_sigmaH(self):
        D = 3
        Thrf = 25.
        dt = .5
        TT, m_h = getCanoHRF(Thrf, dt)
        m_h = m_h[:D]
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D, D), dtype=np.float64)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order, D)
        R = np.dot(D2, D2) / pow(dt, 2*order)
        sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)

    def test_max_sigmaH_prior(self):
        D = 3
        Thrf = 25.
        dt = .5
        TT, m_h = getCanoHRF(Thrf, dt)
        m_h = m_h[:D]
        m_H = np.array(m_h).astype(np.float64)
        Sigma_H = np.ones((D, D), dtype=np.float64)
        order = 2
        D2 = vt.buildFiniteDiffMatrix(order, D)
        R = np.dot(D2, D2) / pow(dt, 2*order)
        gamma_h = 1000
        sigmaH = vt.maximization_sigmaH_prior(D, Sigma_H, R, m_H, gamma_h)

    def test_matrix(self):
        Thrf = 25.
        dt = .5
        TT, m_h = getCanoHRF(Thrf, dt)
        m_H = np.array(m_h)
        m = vt.mult(m_H, m_H)

    def test_maximum(self):
        M = 51
        K = 2
        vector = np.ones((K, 1), dtype=np.float64)
        vector[0] = 2
        maxs, maxs_ind = vt.maximum(vector)

    def test_normpdf(self):
        y = vt.normpdf(1, 1, 1)

    def test_polyFit(self):
        N = 325
        TR = 1.
        data = self.data_simu
        Y = data.bold
        P = vt.PolyMat(N, 4, TR)
        L = vt.polyFit(Y, TR, 4, P)

    def test_PolyMat(self):
        N = 325
        TR = 1.
        P = vt.PolyMat(N, 4, TR)

    def test_compute_mat_X2(self):
        N = 325
        TR = 1.
        D = 3
        dt = .5
        data = self.data_simu
        X = OrderedDict([])
        Onsets = data.get_joined_onsets()
        for condition, Ons in Onsets.iteritems():
            X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)

    def test_buildFiniteDiffMatrix(self):
        order = 2
        D = 3
        diffMat = vt.buildFiniteDiffMatrix(order, D)
