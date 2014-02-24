# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np

import tempfile
from copy import deepcopy
import shutil

import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import jde_vol_from_files, jde_surf_from_files
from pyhrf.parcellation import parcellation_report, parcellation_for_jde

from pyhrf.jde.models import BOLDGibbsSampler as BG
from pyhrf.jde.beta import BetaSampler as BS
from pyhrf.jde.nrl.bigaussian import NRLSampler as NS
from pyhrf.jde.nrl.bigaussian import BiGaussMixtureParamsSampler as BGMS
from pyhrf.jde.hrf import RHSampler as HVS
from pyhrf.jde.hrf import HRFSampler as HS
from pyhrf.jde.models import simulate_bold
from pyhrf.jde.noise import NoiseVarianceSampler


from pyhrf.jde.samplerbase import GibbsSamplerVariable

class JDETest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

        bf = 'subj0_bold_session0.nii.gz'
        self.boldFiles = [pyhrf.get_data_file_name(bf)]
        pf = 'subj0_parcellation.nii.gz'
        self.parcelFile = pyhrf.get_data_file_name(pf)
        self.tr = 2.4
        self.dt = .6
        self.onsets = pyhrf.onsets_loc_av
        #self.durations = pyhrf.durations_loc_av
        self.durations = None
        self.nbIt = 3
        self.pfMethod = 'es'

        simu = simulate_bold(self.tmp_dir, spatial_size='normal')
        self.data_simu = FmriData.from_simulation_dict(simu)


        self.tmp_dir_small = tempfile.mkdtemp(prefix='small_simu',
                                              dir=self.tmp_dir)
        simu = simulate_bold(self.tmp_dir_small, spatial_size='small')
        self.data_small_simu = FmriData.from_simulation_dict(simu)

        print 'Create sampler_params_for_single_test ...'
        self.sampler_params_for_single_test = {
            BG.P_NB_ITERATIONS : 100,
            BG.P_SMPL_HIST_PACE : 1,
            BG.P_OBS_HIST_PACE : 1,
            # level of spatial correlation = beta
            BG.P_BETA : BS({
                    BS.P_SAMPLE_FLAG : False,
                    BS.P_USE_TRUE_VALUE : False,
                    BS.P_VAL_INI : np.array([0.6]),
                    }),
            # HRF
            BG.P_HRF : HS({
                    HS.P_SAMPLE_FLAG : False,
                    HS.P_USE_TRUE_VALUE : True,
                    HS.P_PRIOR_TYPE : 'singleHRF',
                    }),
            # HRF variance
            BG.P_RH : HVS({
                    HVS.P_USE_TRUE_VALUE : True,
                    HVS.P_SAMPLE_FLAG : False,
                    }),
            # neural response levels (stimulus-induced effects)
            BG.P_NRLS : NS({
                    NS.P_USE_TRUE_NRLS : True,
                    NS.P_USE_TRUE_LABELS : True,
                    NS.P_SAMPLE_FLAG : False,
                    NS.P_SAMPLE_LABELS : False,
                    }),
            BG.P_MIXT_PARAM : BGMS({
                    BGMS.P_SAMPLE_FLAG : False,
                    BGMS.P_USE_TRUE_VALUE : True,
                    }),
            BG.P_NOISE_VAR : NoiseVarianceSampler({
                    NoiseVarianceSampler.P_SAMPLE_FLAG : False,
                    NoiseVarianceSampler.P_USE_TRUE_VALUE : True,
                    }),
            BG.P_CHECK_FINAL_VALUE : 'raise', #print or raise
            }


        print 'Create sampler_params_for_full_test ...'
        self.sampler_params_for_full_test = {
            BG.P_NB_ITERATIONS : 500,
            BG.P_SMPL_HIST_PACE : 1,
            BG.P_OBS_HIST_PACE : 1,
            # level of spatial correlation = beta
            BG.P_BETA : BS({
                    BS.P_SAMPLE_FLAG : True,
                    BS.P_USE_TRUE_VALUE : False,
                    BS.P_VAL_INI : np.array([0.6]),
                    }),
            # HRF
            BG.P_HRF : HS({
                    HS.P_SAMPLE_FLAG : True,
                    HS.P_USE_TRUE_VALUE : False,
                    HS.P_NORMALISE : 1.,
                    }),
            # HRF variance
            BG.P_RH : HVS({
                    HVS.P_USE_TRUE_VALUE : False,
                    HVS.P_SAMPLE_FLAG : True,
                    }),
            # neural response levels (stimulus-induced effects)
            BG.P_NRLS : NS({
                    NS.P_USE_TRUE_NRLS : False,
                    NS.P_USE_TRUE_LABELS : False,
                    NS.P_SAMPLE_FLAG : True,
                    NS.P_SAMPLE_LABELS : True,
                    }),
            BG.P_MIXT_PARAM : BGMS({
                    BGMS.P_SAMPLE_FLAG : True,
                    BGMS.P_USE_TRUE_VALUE : False,
                    }),
            BG.P_NOISE_VAR : NoiseVarianceSampler({
                    NoiseVarianceSampler.P_SAMPLE_FLAG : True,
                    NoiseVarianceSampler.P_USE_TRUE_VALUE : False,
                    }),
            BG.P_CHECK_FINAL_VALUE : 'print', #print or raise
            }

    def tearDown(self):
        if 0:
            pyhrf.verbose(1, 'Remove tmp dir %s' %self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            pyhrf.verbose(1, 'Keep tmp dir %s' %self.tmp_dir)


    def _test_specific_samplers(self, sampler_names, simu_scenario='small',
                                nb_its=None, use_true_val=None,
                                use_true_nrlsbar=False, use_true_labels=False,
                                sample_labels=True, sample_nrlsbar=True,
                                save_history=False, check_fv=None,
                                normalize_hrf=1., hrf_prior_type='singleHRF'):

        if use_true_val is None:
            use_true_val = dict( (n,False) for n in sampler_names )

        pyhrf.verbose(1, '_test_specific_samplers %s ...' %str(sampler_names))

        params = deepcopy(self.sampler_params_for_single_test)

        for var_name in sampler_names:
            var_class = params[var_name].__class__

            if var_class == HS:
                dict_hrf =  {
                    HS.P_SAMPLE_FLAG : True,
                    HS.P_USE_TRUE_VALUE : \
                        use_true_val[var_name],
                    HS.P_NORMALISE : normalize_hrf,
                    HS.P_PRIOR_TYPE : hrf_prior_type,
                    }
                params[var_name] = var_class(dict_hrf)
            elif var_class == NS:
                dict_nrl =  {
                    var_class.P_SAMPLE_FLAG : sample_nrlsbar,
                    var_class.P_USE_TRUE_NRLS : use_true_nrlsbar,
                    var_class.P_SAMPLE_LABELS : sample_labels,
                    var_class.P_USE_TRUE_LABELS : use_true_labels,
                    }
                params[var_name] = var_class(dict_nrl)
            else:
                dict_var = {var_class.P_SAMPLE_FLAG : True,
                            var_class.P_USE_TRUE_VALUE : use_true_val[var_name],
                            }
                params[var_name] = var_class(dict_var)

        for var_name, param in params.iteritems():
            if isinstance(param, GibbsSamplerVariable):
                print 'param[%s] -> use_true_val: %s' \
                  %(var_name, param.useTrueValue)

        if nb_its is not None:
            params[BG.P_NB_ITERATIONS] = nb_its

        if save_history:
            params[BG.P_SMPL_HIST_PACE] = 1
            params[BG.P_OBS_HIST_PACE] = 1

        if check_fv is not None:
            params[BG.P_CHECK_FINAL_VALUE] = check_fv


        if BG.P_HRF not in sampler_names:
            dict_hrf =  {
                HS.P_SAMPLE_FLAG : False,
                HS.P_USE_TRUE_VALUE : True,
                HS.P_NORMALISE : normalize_hrf,
                HS.P_PRIOR_TYPE : hrf_prior_type,
                }
            params[BG.P_HRF] = HS(dict_hrf)

        sampler = BG(params)

        if simu_scenario == 'small':
            simu = self.data_small_simu
            output_dir = self.tmp_dir_small
        elif simu_scenario == 'normal':
            simu = self.data_simu
            output_dir = self.tmp_dir
        else:
            simu = simu_scenario
            output_dir = self.tmp_dir

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_mcmc_',
                                   randomSeed=5421087, pass_error=False)

        treatment = FMRITreatment(fmri_data=simu, analyser=analyser,
                                  output_dir=output_dir)

        treatment.run()
        print 'out_dir:', output_dir




    def test_noise_var_sampler(self):
        pyhrf.verbose.set_verbosity(2)
        self._test_specific_samplers(['noiseVariance'],
                                     check_fv='raise')


    def test_hrf_var_sampler(self):

        # estimation of HRF variance tested in the following situation:
        #    -  simulated gaussian smooth HRF is not normalized
        #       -> else the simulated HRF variance is not consistent

        pyhrf.verbose.set_verbosity(2)

        simu = simulate_bold(self.tmp_dir, spatial_size='small',
                             normalize_hrf=False)
        simu = FmriData.from_simulation_dict(simu)

        self._test_specific_samplers(['HRFVariance'], simu_scenario=simu,
                                     check_fv='raise', nb_its=100,
                                     hrf_prior_type='singleHRF')


    def test_hrf_var_sampler_2(self):

        # estimation of HRF variance tested in the following situation:
        #    -  simulated gaussian smooth HRF is not normalized
        #       -> else the simulated HRF variance is not consistent

        pyhrf.verbose.set_verbosity(2)

        simu = simulate_bold(self.tmp_dir, spatial_size='small',
                             normalize_hrf=False)
        simu = FmriData.from_simulation_dict(simu)

        self._test_specific_samplers(['HRFVariance'], simu_scenario=simu,
                                     check_fv='raise', nb_its=100,
                                     hrf_prior_type='voxelwiseIID')



    def test_hrf_with_var_sampler(self):
        # estimation of HRF and its variance tested in the following situation:
        #    -  simulated gaussian smooth HRF is not normalized

        pyhrf.verbose.set_verbosity(2)

        simu = simulate_bold(self.tmp_dir, spatial_size='small',
                             normalize_hrf=False)
        simu = FmriData.from_simulation_dict(simu)

        self._test_specific_samplers(['HRFVariance','HRF'], simu_scenario=simu,
                                     check_fv='print', nb_its=100,
                                     hrf_prior_type='singleHRF')

    def test_hrf_with_var_sampler_2(self):
        # estimation of HRF and its variance tested in the following situation:
        #    -  simulated gaussian smooth HRF is not normalized

        pyhrf.verbose.set_verbosity(2)

        simu = simulate_bold(self.tmp_dir, spatial_size='small',
                             normalize_hrf=False)
        simu = FmriData.from_simulation_dict(simu)

        self._test_specific_samplers(['HRFVariance','HRF'], simu_scenario=simu,
                                     check_fv='print', nb_its=100,
                                     hrf_prior_type='voxelwiseIID')



    def test_full_sampler(self):
        """ Test JDE on simulation with normal size.
        Estimation accuracy is tested.
        """
        pyhrf.verbose.set_verbosity(2)

        simu = simulate_bold(self.tmp_dir, spatial_size='normal',
                             normalize_hrf=False)
        simu = FmriData.from_simulation_dict(simu)


        sampler = BG(self.sampler_params_for_full_test)

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_mcmc_',
                                   randomSeed=None)


        treatment = FMRITreatment(fmri_data=simu,
                                  analyser=analyser, output_dir=self.tmp_dir)

        treatment.run()
        print 'output_dir:', self.tmp_dir
