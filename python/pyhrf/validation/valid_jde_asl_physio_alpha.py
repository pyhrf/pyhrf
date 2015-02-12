# -*- coding: utf-8 -*-

import unittest
import numpy as np
import shutil
import logging

from copy import deepcopy

import pyhrf
import pyhrf.jde.asl_physio_alpha as jasl  # FIXME: no asl_physio_alpha
#import pyhrf.jde.asl_physio as jasl

from pyhrf.core import FmriData
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.treatment import FMRITreatment


logger = logging.getLogger(__name__)


class ASLTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        self.tmp_dir = pyhrf.get_tmp_path()
        self.clean_tmp = False  # HACK True

        self.sampler_params_for_single_test = {
            'nb_iterations': 40,
            'smpl_hist_pace': 1,
            'obs_hist_pace': 1,
            'brf': jasl.PhysioBOLDResponseSampler(do_sampling=False,
                                                  normalise=1.,
                                                  use_true_value=True,
                                                  zero_constraint=False),
            'brf_var':
            jasl.PhysioBOLDResponseVarianceSampler(do_sampling=False,
                                                   val_ini=np.array([0.1]),
                                                   use_true_value=False),
            'prf': jasl.PhysioPerfResponseSampler(do_sampling=False,
                                                  normalise=1.,
                                                  use_true_value=True,
                                                  zero_constraint=False,
                                                  prior_type='physio_stochastic_regularized'),
            'prf_var':
            jasl.PhysioPerfResponseVarianceSampler(do_sampling=False,
                                                   val_ini=np.array(
                                                       [.001]),
                                                   use_true_value=False),
            'noise_var': jasl.NoiseVarianceSampler(do_sampling=False,
                                                   use_true_value=True),
            'drift_var': jasl.DriftVarianceSampler(do_sampling=False,
                                                   use_true_value=True),
            'drift': jasl.DriftCoeffSampler(do_sampling=False,
                                            use_true_value=True),
            'bold_response_levels':
            jasl.BOLDResponseLevelSampler(do_sampling=False,
                                          use_true_value=True),
            'perf_response_levels':
            jasl.PerfResponseLevelSampler(do_sampling=False,
                                          use_true_value=True),
            'labels': jasl.LabelSampler(do_sampling=False,
                                        use_true_value=True),
            'bold_mixt_params': jasl.BOLDMixtureSampler(do_sampling=False,
                                                        use_true_value=True),
            'perf_mixt_params': jasl.PerfMixtureSampler(do_sampling=False,
                                                        use_true_value=True),
            'perf_baseline': jasl.PerfBaselineSampler(do_sampling=False,
                                                      use_true_value=True),
            'perf_baseline_var':
            jasl.PerfBaselineVarianceSampler(do_sampling=False,
                                             use_true_value=True),
            'check_final_value': 'raise',  # print or raise
        }

    def tearDown(self):
        if self.clean_tmp:
            logger.info('Remove tmp dir %s', self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            logger.info('Keep tmp dir %s', self.tmp_dir)

    def test_prf_physio_reg(self):
        """ Validate estimation of PRF """
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['prf'], fdata, nb_its=20,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def test_brf_physio_reg(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['brf'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_brf_basic_reg(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['brf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='basic_regularized')
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_prf_basic_reg(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['prf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='basic_regularized')
        print 'pyhrf_view_qt3 %s/*prf*nii' % self.tmp_dir

    def test_brf_physio_nonreg(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['brf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='physio_stochastic_not_regularized')
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_prf_physio_nonreg(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        print simu['prf'].shape
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['prf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='physio_stochastic_not_regularized')
        print 'pyhrf_view_qt3 %s/*prf*nii' % self.tmp_dir

    def test_brf_physio_det(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['brf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='physio_deterministic')
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_prf_physio_det(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        print simu['prf'].shape
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['prf'], fdata, nb_its=100,
                                     check_fv='raise',
                                     rf_prior_type='physio_deterministic')
        print 'pyhrf_view_qt3 %s/*prf*nii' % self.tmp_dir

    def test_brf_var(self):
        """ Validate estimation of BRF at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['brf_var'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*brf*nii' % self.tmp_dir

    def test_prf_var(self):
        """ Validate estimation of PRF """
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['prf_var'], fdata, nb_its=20,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*prf*nii' % self.tmp_dir

    def test_brls(self):
        """ Validate estimation of BRLs at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['bold_response_levels'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*brl*nii' % self.tmp_dir

    def test_prls(self):
        """ Validate estimation of PRLs at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['perf_response_levels'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*prl*nii' % self.tmp_dir

    def test_labels(self):
        """ Validate estimation of labels at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['labels'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*label*nii' % self.tmp_dir

    def test_noise_var(self):
        """ Validate estimation of noise variances at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['noise_var'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def test_drift(self):
        """ Validate estimation of drift at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['drift'], fdata, nb_its=200,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*drift*nii' % self.tmp_dir

    def test_drift_var(self):
        """ Validate estimation of drift at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['drift_var'], fdata, nb_its=100,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*drift*nii' % self.tmp_dir

    def test_perf_baseline(self):
        """ Validate estimation of drift at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        print simu['perf_baseline']
        # print 'pyhrf_view_qt3 %s/*perf*nii' %self.tmp_dir
        self._test_specific_samplers(['perf_baseline'], fdata, nb_its=100,
                                     check_fv='print')
        print 'pyhrf_view_qt3 %s/*perf*nii' % self.tmp_dir

    def test_perf_baseline_var(self):
        """ Validate estimation of drift at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        perf_baseline = simu['perf_baseline']
        perf_baseline_mean = simu['perf_baseline_mean']
        print 'perf_baseline_mean = ', perf_baseline_mean
        print 'perf_baseline_mean emp = ', np.mean(perf_baseline)
        perf_baseline_var = simu['perf_baseline_var']
        print 'perf_baseline_var = ', perf_baseline_var
        print 'perf_baseline_var emp = ', np.var(perf_baseline)
        fdata = FmriData.from_simulation_dict(simu)
        self._test_specific_samplers(['perf_baseline_var'], fdata, nb_its=15,
                                     check_fv='raise')
        print 'pyhrf_view_qt3 %s/*perf*nii' % self.tmp_dir

    def test_all(self):
        """ Validate estimation of full ASL model at high SNR"""
        # pyhrf.verbose.set_verbosity(2)
        pyhrf.logger.setLevel(logging.INFO)
        from pyhrf.jde.asl import simulate_asl
        simu = simulate_asl(self.tmp_dir, spatial_size='normal')
        fdata = FmriData.from_simulation_dict(simu)
        np.random.seed(25430)
        v = ['bold_response_levels', 'perf_response_levels', 'drift', 'drift_var',
             'brf', 'brf_var', 'prf', 'labels', 'bold_mixt_params',
             'perf_mixt_params', 'perf_baseline', 'perf_baseline_var']

        self._test_specific_samplers(v, fdata, nb_its=500, check_fv='print')
        print 'pyhrf_view_qt3 %s/*nii' % self.tmp_dir

    def _test_specific_samplers(self, sampler_names, fdata,
                                nb_its=None, use_true_val=None,
                                save_history=False, check_fv=None,
                                normalize_brf=1., normalize_prf=1.,
                                normalize_mu=1.,
                                rf_prior_type='physio_stochastic_regularized'):
        """
        Test specific samplers.
        """
        if use_true_val is None:
            use_true_val = dict((n, False) for n in sampler_names)

        logger.info('_test_specific_samplers %s ...', str(sampler_names))

        params = deepcopy(self.sampler_params_for_single_test)

        # Loop over given samplers to enable them
        for var_name in sampler_names:
            var_class = params[var_name].__class__
            use_tval = use_true_val[var_name]

            # special case for HRF -> normalization and prior type
            if var_class == jasl.PhysioBOLDResponseSampler:
                params[var_name] = \
                    jasl.PhysioBOLDResponseSampler(do_sampling=True,
                                                   use_true_value=use_tval,
                                                   normalise=normalize_brf,
                                                   zero_constraint=False)
            elif var_class == jasl.PhysioPerfResponseSampler:
                params[var_name] = \
                    jasl.PhysioPerfResponseSampler(do_sampling=True,
                                                   use_true_value=use_tval,
                                                   normalise=normalize_brf,
                                                   zero_constraint=False,
                                                   prior_type=rf_prior_type)

            else:
                params[var_name] = var_class(do_sampling=True,
                                             use_true_value=use_tval)

        if nb_its is not None:
            params['nb_iterations'] = nb_its

        if save_history:
            params['smpl_hist_pace'] = 1
            params['obs_hist_pace'] = 1

        if check_fv is not None:
            params['check_final_value'] = check_fv

        sampler = jasl.ASLPhysioSampler(**params)

        output_dir = self.tmp_dir

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=fdata.simulation[0]['dt'], driftParam=4,
                                   driftType='polynomial',
                                   outputPrefix='jde_mcmc_',
                                   pass_error=False)

        treatment = FMRITreatment(fmri_data=fdata, analyser=analyser,
                                  output_dir=output_dir)

        outputs = treatment.run()
        print 'out_dir:', output_dir
        return outputs
