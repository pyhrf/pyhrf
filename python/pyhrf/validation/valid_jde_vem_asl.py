# -*- coding: utf-8 -*-

#import os
import unittest
import numpy as np
import shutil
#from copy import deepcopy

import pyhrf
from pyhrf.core import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.vb_jde_analyser_asl import JDEVEMAnalyser


class ASLTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        self.tmp_dir = pyhrf.get_tmp_path()
        self.clean_tmp = False  # HACK True

    def tearDown(self):
        if self.clean_tmp:
            pyhrf.verbose(1, 'Remove tmp dir %s' % self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            pyhrf.verbose(1, 'Keep tmp dir %s' % self.tmp_dir)

    def test_prf(self):
        """ Validate estimation of PRF """
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['prf'], fdata, simu, nItMax=20,
                                       estimateG=True)
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def test_brf(self):
        """ Validate estimation of BRF at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['brf'], fdata, simu, nItMax=100, \
                                       estimateH=True)
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_brls(self):
        """ Validate estimation of BRLs at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['bold_response_levels'], fdata, simu,
                                       nItMax=100, estimateA=True)
        print 'pyhrf_view_qt3 %s/*brl*nii' % self.tmp_dir

    def test_prls(self):
        """ Validate estimation of PRLs at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['perf_response_levels'], fdata, simu,
                                       nItMax=100, estimateC=True)
        print 'pyhrf_view_qt3 %s/*prl*nii' % self.tmp_dir

    def test_labels(self):
        """ Validate estimation of labels at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['labels'], fdata, simu, nItMax=100,
                                       estimateZ=True)
        print 'pyhrf_view_qt3 %s/*label*nii' % self.tmp_dir

    def test_noise_var(self):
        """ Validate estimation of noise variances at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['noise_var'], fdata, simu, nItMax=100,
                                       estimateNoise=True)
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def test_la(self):
        """ Validate estimation of drift at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_parameters(['drift_perf_baseline'], fdata, simu,
                                       nItMax=100, estimateLA=True)
        print 'pyhrf_view_qt3 %s/*drift*nii' % self.tmp_dir

    def test_all(self):
        """ Validate estimation of full ASL model at high SNR"""
        pyhrf.verbose.set_verbosity(2)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        np.random.seed(25430)
        v = ['bold_response_levels', 'perf_response_levels',
             'brf', 'brf_var', 'prf', 'labels', 'bold_mixt_params',
             'perf_mixt_params', 'drift_perf_baseline']

        self._test_specific_parameters(v, fdata, simu,
                                       estimateSigmaH=False, nItMax=100,
                                       nItMin=10, estimateBeta=True,
                                       estimateSigmaG=True, PLOT=True,
                                       constrained=True, fast=False,
                                       estimateH=True, estimateG=True,
                                       estimateA=True, estimateC=True,
                                       estimateZ=True, estimateLA=True)
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def _test_specific_parameters(self, parameter_name, fdata, simu,
                                  beta=.8, dt=.5, nItMax=100, nItMin=10,
                                  hrfDuration=25., estimateSigmaH=False,
                                  estimateBeta=False, estimateSigmaG=False,
                                  PLOT=False, constrained=True, fast=False,
                                  estimateH=False, estimateG=False,
                                  estimateA=False, estimateC=False,
                                  estimateZ=False, estimateLA=False,
                                  estimateNoise=False):
        """
        Test specific samplers.
        """
        pyhrf.verbose(1, '_test_specific_parameters %s' % str(parameter_name))
        output_dir = self.tmp_dir
        #JDE analysis
        jde_vem_analyser = JDEVEMAnalyser(beta=beta, dt=dt,
                                          hrfDuration=hrfDuration,
                                          estimateSigmaH=estimateSigmaH,
                                          nItMax=nItMax, nItMin=nItMin,
                                          estimateBeta=estimateBeta,
                                          estimateSigmaG=estimateSigmaG,
                                          PLOT=PLOT,
                                          constrained=constrained, fast=fast,
                                          fmri_data=simu,
                                          estimateH=estimateH,
                                          estimateG=estimateG,
                                          estimateA=estimateA,
                                          estimateC=estimateC,
                                          estimateLabels=estimateZ,
                                          estimateLA=estimateLA,
                                          estimateNoise=estimateNoise)
        tjde_vem = FMRITreatment(fmri_data=fdata, analyser=jde_vem_analyser,
                                 output_dir=output_dir)
        outputs = tjde_vem.run()
        print 'out_dir:', output_dir
        return outputs
