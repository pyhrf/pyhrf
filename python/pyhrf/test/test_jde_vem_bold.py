# -*- coding: utf-8 -*-

import unittest
import tempfile
import shutil
import logging

import numpy as np

import pyhrf
import pyhrf.tools as tools

from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.jde.models import simulate_bold
from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
from pyhrf.vbjde.vem_bold_constrained import (Main_vbjde_Extension_constrained,
                                              Main_vbjde_Python_constrained)


logger = logging.getLogger(__name__)


class VEMBOLDTest(unittest.TestCase):

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

    def test_jdevemanalyser(self):
        """ Test BOLD VEM sampler on small simulation with small
        nb of iterations. Estimation accuracy is not tested.
        """
        jde_vem_analyser = JDEVEMAnalyser(beta=.8, dt=.5, hrfDuration=25.,
                                          nItMax=2, nItMin=2, fast=True,
                                          computeContrast=False, PLOT=False,
                                          constrained=True)
        tjde_vem = FMRITreatment(fmri_data=self.data_simu,
                                 analyser=jde_vem_analyser,
                                 output_dir=None)
        tjde_vem.run()

    @unittest.skipIf(not tools.is_importable('cvxpy'),
                     'cvxpy (optional dep) is N/A')
    def test_vem_bold_constrained(self):
        """ Test BOLD VEM constraint function.
        Estimation accuracy is not tested.
        """
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()

        NbIter, nrls, estimated_hrf, \
            labels, noiseVar, mu_k, sigma_k, \
            Beta, L, PL, CONTRAST, CONTRASTVAR, \
            cA, cH, cZ, cAH, cTime, cTimeMean, \
            Sigma_nrls, StimuIndSignal,\
            FreeEnergy = Main_vbjde_Extension_constrained(graph, data.bold, Onsets,
                                                          Thrf=25., K=2, TR=1.,
                                                          beta=1.0, dt=.5,
                                                          NitMax=2, NitMin=2)

    @unittest.skipIf(not tools.is_importable('cvxpy'),
                     'cvxpy (optional dep) is N/A')
    def test_vem_bold_constrained_python(self):
        """ Test BOLD VEM constraint function.
        Estimation accuracy is not tested.
        """
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()

        m_A, m_H, q_Z, sigma_epsilone, \
            mu_M, sigma_M, Beta, L, \
            PL = Main_vbjde_Python_constrained(graph, data.bold, Onsets,
                                               25., 2, 1., 1.0, .5,
                                               NitMax=2, NitMin=2)
