import os
import os.path as op
import numpy as np
import unittest

import pyhrf
from pyhrf.core import merge_fmri_sessions

class MultiSessTest(unittest.TestCase):
    #@debug_on()

    def setUp(self):

        #pyhrf.verbose.set_verbosity(2)

        np.random.seed(8652761)

        # tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
        #     dir=pyhrf.cfg['global']['tmp_path'])

        self.tmp_outputs = True #save outputs in tmp dir
                                #if False then save in current dir

        if not self.tmp_outputs:
            self.tmp_dir_small = './JDE_MS_test_small_simu'
            if not op.exists(self.tmp_dir_small): os.makedirs(self.tmp_dir_small)
            self.tmp_dir_big = './JDE_MS_test_big_simu'
            if not op.exists(self.tmp_dir_big): os.makedirs(self.tmp_dir_big)
        else:
            self.tmp_dir_small = pyhrf.get_tmp_path()
            self.tmp_dir_big = pyhrf.get_tmp_path()

        simu = simulate_sessions(output_dir = self.tmp_dir_small,
                                 snr_scenario='high_snr', spatial_size='tiny')
        self.data_small_simu = merge_fmri_sessions(simu)

        simu = simulate_sessions(output_dir=self.tmp_dir_big,
                                 snr_scenario='low_snr', spatial_size='normal')
        self.data_simu = merge_fmri_sessions(simu)

        # Parameters for multi-sessions sampler
        dict_beta_single = {
                    BetaSampler.P_VAL_INI : np.array([0.5]),
                    BetaSampler.P_SAMPLE_FLAG : False,
                    BetaSampler.P_PARTITION_FUNCTION_METH : 'es',
                    BetaSampler.P_USE_TRUE_VALUE : False,
                    }

        dict_hrf_single = {
                    HRF_MultiSess_Sampler.P_SAMPLE_FLAG : False,
                    HRF_MultiSess_Sampler.P_NORMALISE : 1., # normalise samples
                    HRF_MultiSess_Sampler.P_USE_TRUE_VALUE :  True,
                    HRF_MultiSess_Sampler.P_ZERO_CONSTR :  True,
                    #HRF_MultiSess_Sampler.P_PRIOR_TYPE : 'singleHRF',
                    }

        dict_var_hrf_single = {
                        RHSampler.P_SAMPLE_FLAG : False,
                        RHSampler.P_VAL_INI : np.array([0.001]),
                    }

        dict_nrl_sess_single =   {
                        NRL_Multi_Sess_Sampler.P_SAMPLE_FLAG : False,
                        NRL_Multi_Sess_Sampler.P_USE_TRUE_VALUE :  True,
                        }

        dict_nrl_sess_var_single = {
                            Variance_GaussianNRL_Multi_Sess.P_SAMPLE_FLAG : False,
                            Variance_GaussianNRL_Multi_Sess.P_USE_TRUE_VALUE :  True,
                            }

        dict_nrl_bar_single =  {
                        NRLsBar_Drift_Multi_Sess_Sampler.P_SAMPLE_FLAG : False,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_USE_TRUE_NRLS : True,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_SAMPLE_LABELS : False,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_USE_TRUE_LABELS : True,
                        }

        dict_drift_single = {
                    Drift_MultiSess_Sampler.P_SAMPLE_FLAG : False,
                    Drift_MultiSess_Sampler.P_USE_TRUE_VALUE : True,
                    }

        dict_drift_var_single = {
                        ETASampler_MultiSess.P_SAMPLE_FLAG : False,
                        ETASampler_MultiSess.P_USE_TRUE_VALUE : True,
                        }

        dict_noise_var_single = {
                        NoiseVariance_Drift_Multi_Sess_Sampler.P_SAMPLE_FLAG : False,
                        NoiseVariance_Drift_Multi_Sess_Sampler.P_USE_TRUE_VALUE :  True,
                        }

        dict_mixt_param_single =  {
                            BiGaussMixtureParamsSampler.P_SAMPLE_FLAG : False,
                            BiGaussMixtureParamsSampler.P_USE_TRUE_VALUE : True,
                            BiGaussMixtureParamsSampler.P_HYPER_PRIOR : 'Jeffrey',
                            }


        self.sampler_params_for_single_test = {
            BMSS.P_NB_ITERATIONS : 100,
            BMSS.P_SMPL_HIST_PACE : -1,
            BMSS.P_OBS_HIST_PACE : -1,
            # level of spatial correlation = beta
            BMSS.P_BETA : BetaSampler(dict_beta_single),
            # HRF
            BMSS.P_HRF : HRF_MultiSess_Sampler(dict_hrf_single),
            # HRF variance
            BMSS.P_RH : RHSampler(dict_var_hrf_single),
            # neural response levels (stimulus-induced effects) by session
            BMSS.P_NRLS_SESS : NRL_Multi_Sess_Sampler(dict_nrl_sess_single),
            # neural response levels by session --> variance
            BMSS.P_NRLS_SESS_VAR : Variance_GaussianNRL_Multi_Sess(dict_nrl_sess_var_single),
            # neural response levels mean: over sessions
            BMSS.P_NRLS_BAR : NRLsBar_Drift_Multi_Sess_Sampler(dict_nrl_bar_single),
            # drift
            BMSS.P_DRIFT : Drift_MultiSess_Sampler(dict_drift_single),
            #drift variance
            BMSS.P_ETA : ETASampler_MultiSess(dict_drift_var_single),
            #noise variance
            BMSS.P_NOISE_VAR_SESS : NoiseVariance_Drift_Multi_Sess_Sampler(dict_noise_var_single),
            #weights o fthe mixture
            #parameters of the mixture
            BMSS.P_MIXT_PARAM_NRLS_BAR : BiGaussMixtureParamsSampler(dict_mixt_param_single),
            BMSS.P_CHECK_FINAL_VALUE : 'raise', #print or raise
        }


        # Parameters for multi-sessions sampler - full test
        dict_beta_full = {
                    BetaSampler.P_VAL_INI : np.array([0.5]),
                    BetaSampler.P_SAMPLE_FLAG : True,
                    BetaSampler.P_PARTITION_FUNCTION_METH : 'es',
                    }

        dict_hrf_full = {
                    HRF_MultiSess_Sampler.P_SAMPLE_FLAG : True,
                    HRF_MultiSess_Sampler.P_NORMALISE : 1., # normalise samples
                    HRF_MultiSess_Sampler.P_USE_TRUE_VALUE :  False,
                    HRF_MultiSess_Sampler.P_ZERO_CONSTR : True,
                    #HRF_MultiSess_Sampler.P_PRIOR_TYPE : 'singleHRF',
                    }

        dict_var_hrf_full = {
                        RHSampler.P_SAMPLE_FLAG : False,
                        RHSampler.P_VAL_INI : np.array([0.001]),
                    }

        dict_nrl_sess_full =   {
                        NRL_Multi_Sess_Sampler.P_SAMPLE_FLAG : True,
                        NRL_Multi_Sess_Sampler.P_USE_TRUE_VALUE :  False,
                        }

        dict_nrl_sess_var_full = {
                            Variance_GaussianNRL_Multi_Sess.P_SAMPLE_FLAG : True,
                            Variance_GaussianNRL_Multi_Sess.P_USE_TRUE_VALUE : False,
                            }

        dict_nrl_bar_full =  {
                        NRLsBar_Drift_Multi_Sess_Sampler.P_SAMPLE_FLAG : True,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_USE_TRUE_NRLS : False,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_SAMPLE_LABELS : True,
                        NRLsBar_Drift_Multi_Sess_Sampler.P_USE_TRUE_LABELS : False,
                        }

        dict_drift_full = {
                    Drift_MultiSess_Sampler.P_SAMPLE_FLAG : True,
                    Drift_MultiSess_Sampler.P_USE_TRUE_VALUE : False,
                    }

        dict_drift_var_full = {
                        ETASampler_MultiSess.P_SAMPLE_FLAG : True,
                        ETASampler_MultiSess.P_USE_TRUE_VALUE : False,
                        }

        dict_noise_var_full = {
                        NoiseVariance_Drift_Multi_Sess_Sampler.P_SAMPLE_FLAG : True,
                        NoiseVariance_Drift_Multi_Sess_Sampler.P_USE_TRUE_VALUE :  False,
                        }

        dict_mixt_param_full =  {
                            BiGaussMixtureParamsSampler.P_SAMPLE_FLAG : True,
                            BiGaussMixtureParamsSampler.P_USE_TRUE_VALUE : False,
                            BiGaussMixtureParamsSampler.P_HYPER_PRIOR : 'Jeffrey',
                            }


        self.sampler_params_for_full_test = {

            BMSS.P_NB_ITERATIONS : 400,
            BMSS.P_SMPL_HIST_PACE : -1,
            BMSS.P_OBS_HIST_PACE : -1,
            # level of spatial correlation = beta
            BMSS.P_BETA : BetaSampler(dict_beta_full),
            # HRF
            BMSS.P_HRF : HRF_MultiSess_Sampler(dict_hrf_full),
            # HRF variance
            BMSS.P_RH : RHSampler(dict_var_hrf_full),
            # neural response levels (stimulus-induced effects) by session
            BMSS.P_NRLS_SESS : NRL_Multi_Sess_Sampler(dict_nrl_sess_full),
            # neural response levels by session --> variance
            BMSS.P_NRLS_SESS_VAR : Variance_GaussianNRL_Multi_Sess(dict_nrl_sess_var_full),
            # neural response levels mean: over sessions
            BMSS.P_NRLS_BAR : NRLsBar_Drift_Multi_Sess_Sampler(dict_nrl_bar_full),
            # drift
            BMSS.P_DRIFT : Drift_MultiSess_Sampler(dict_drift_full),
            #drift variance
            BMSS.P_ETA : ETASampler_MultiSess(dict_drift_var_full),
            #noise variance
            BMSS.P_NOISE_VAR_SESS : NoiseVariance_Drift_Multi_Sess_Sampler(dict_noise_var_full),
            #weights o fthe mixture
            #parameters of the mixture
            BMSS.P_MIXT_PARAM_NRLS_BAR : BiGaussMixtureParamsSampler(dict_mixt_param_full),
            BMSS.P_CHECK_FINAL_VALUE : 'raise', #print or raise
        }


    def tearDown(self):
        if self.tmp_outputs:
            shutil.rmtree(self.tmp_dir_big)
            shutil.rmtree(self.tmp_dir_small)

    def test_simulation(self):

        pyhrf.verbose.set_verbosity(0)
        simulate_sessions(output_dir=self.tmp_dir_small,
                          snr_scenario='high_snr', spatial_size='tiny')

    def test_default_jde_small_simulation(self):
        """ Test JDE multi-sessions sampler on small
        simulation with small nb of iterations.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)

        sampler = BMSS()

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_MS_mcmc_',
                                   randomSeed=9778946)

        treatment = FMRITreatment(fmri_data=self.data_simu,
                                  analyser=analyser)

        treatment.run()

    def _test_specific_samplers(self, sampler_names, simu_scenario='small',
                                nb_its=None, use_true_val=None,
                                use_true_nrlsbar=False, use_true_labels=False,
                                sample_labels=True, sample_nrlsbar=True,
                                save_history=False, check_fv=None):

        if use_true_val is None:
            use_true_val = dict( (n,False) for n in sampler_names )

        pyhrf.verbose(1, '_test_specific_samplers %s ...' %str(sampler_names))

        params = deepcopy(self.sampler_params_for_single_test)

        for var_name in sampler_names:
            var_class = params[var_name].__class__

            if var_class == HRF_MultiSess_Sampler:
                dict_hrf =  {
                    HRF_MultiSess_Sampler.P_SAMPLE_FLAG : True,
                    HRF_MultiSess_Sampler.P_USE_TRUE_VALUE : \
                        use_true_val[var_name],
                    #HRF_MultiSess_Sampler.P_ZERO_CONSTR :  False,
                    }
                params[var_name] = var_class(dict_hrf)
            elif var_class == NRLsBar_Drift_Multi_Sess_Sampler:
                dict_nrl_bar =  {
                    var_class.P_SAMPLE_FLAG : sample_nrlsbar,
                    var_class.P_USE_TRUE_NRLS : use_true_nrlsbar,
                    var_class.P_SAMPLE_LABELS : sample_labels,
                    var_class.P_USE_TRUE_LABELS : use_true_labels,
                    }
                params[var_name] = var_class(dict_nrl_bar)
            else:
                dict_var = {var_class.P_SAMPLE_FLAG : True,
                            var_class.P_USE_TRUE_VALUE : use_true_val[var_name],
                            }
                params[var_name] = var_class(dict_var)

        if nb_its is not None:
            params[BMSS.P_NB_ITERATIONS] = nb_its

        if save_history:
            params[BMSS.P_SMPL_HIST_PACE] = 1
            params[BMSS.P_OBS_HIST_PACE] = 1

        if check_fv is not None:
            params[BMSS.P_CHECK_FINAL_VALUE] = check_fv

        sampler = BMSS(params)

        if simu_scenario == 'small':
            simu = self.data_small_simu
            output_dir = self.tmp_dir_small
        else:
            simu = self.data_simu
            output_dir = self.tmp_dir_big

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_MS_mcmc_',
                                   randomSeed=5421087)

        treatment = FMRITreatment(fmri_data=simu, analyser=analyser,
                                  output_dir=output_dir)

        treatment.run()
        #print 'out_dir:', output_dir


    def test_nrl_by_session_hrf(self):
        pyhrf.verbose.set_verbosity(0)
        self._test_specific_samplers(['responseLevels_by_session',BMSS.P_HRF],
                                     simu_scenario='big', nb_its=50,
                                     save_history=True)

    def test_noise_var_sampler(self):
        self._test_specific_samplers(['noiseVariance'])

    def test_nrl_by_session_sampler(self):
        self._test_specific_samplers(['responseLevels_by_session'],
                                     simu_scenario='big', check_fv='print')

    def test_nrl_by_session_var_sampler(self):
        self._test_specific_samplers(['responseLevels_by_sessionVariance'],
                                     simu_scenario='big')

    def test_nrl_bar_only_sampler(self):
        self._test_specific_samplers(['MeanresponseLevels'],
                                     use_true_labels=True, sample_labels=False,
                                     simu_scenario='big')

    def test_label_sampler(self):
        self._test_specific_samplers(['MeanresponseLevels'],
                                     use_true_nrlsbar=True, sample_nrlsbar=False,
                                     simu_scenario='big')

    def test_nrl_bar_sampler(self):
        self._test_specific_samplers(['MeanresponseLevels'],
                                     simu_scenario='big')

    def test_hrf_sampler(self):
        self._test_specific_samplers(['HRF'])

    def test_hrf_var_sampler(self):
        self._test_specific_samplers(['HRFVariance'])

    def test_bmixt_sampler(self):
        n = 'mixtureParameters_for_responseLevels_by_session'
        self._test_specific_samplers([n], simu_scenario='big')

    def test_drift_and_var_sampler(self):
        self._test_specific_samplers(['drift', 'driftVar'],
                                     simu_scenario='big', check_fv='print')

    def test_drift_sampler(self):
        self._test_specific_samplers(['drift'])

    def test_drift_var_sampler(self):
        self._test_specific_samplers(['driftVar'])

    def test_full_sampler(self):
        """ Test JDE Multi-sessions sampler on simulation with normal size.
        Estimation accuracy is tested.
        """
        pyhrf.verbose.set_verbosity(0)
        params = self.sampler_params_for_full_test.copy()

        params[BMSS.P_CHECK_FINAL_VALUE] = 'print' #print or raise
        params[BMSS.P_NB_ITERATIONS] = 170

        if 0: #save history
            params[BMSS.P_SMPL_HIST_PACE] = 1
            params[BMSS.P_OBS_HIST_PACE] = 1


        sampler = BMSS(params)

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_MS_mcmc_',
                                   randomSeed=None)

        treatment = FMRITreatment(fmri_data=self.data_simu,
                                  analyser=analyser, output_dir=self.tmp_dir_big)

        treatment.run()
