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
from pyhrf.jde.asl import simulate_asl
from pyhrf.ui.vb_jde_analyser_asl_fast import JDEVEMAnalyser


logger = logging.getLogger(__name__)


class VEMASLTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir
        simu = simulate_asl(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

    def tearDown(self):
        if 1:
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
                                          PLOT=False,
                                          constrained=True)
        tjde_vem = FMRITreatment(fmri_data=self.data_simu,
                                 analyser=jde_vem_analyser,
                                 output_dir=None)
        tjde_vem.run()

