import os
import os.path as op
import numpy as np
import pyhrf

from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser

def jde_analyse_2steps(output_dir, fmri_data, dt, nb_iterations, brf_var=None,
                       do_sampling_brf_var=False):

    """
    Return:
        dict of outputs
    """

    nb_iterations_m1 = nb_iterations_m2 = nb_iterations

    jde_output_dir = op.join(output_dir, 'bold_only')
    if not op.exists(jde_output_dir):
        os.makedirs(jde_output_dir)

    dummy_sampler = dummy_jde(fmri_data, dt)
    prf_ini_m1 = np.zeros_like(dummy_sampler.get_variable('prf').finalValue)
    prls_ini_m1 = np.zeros_like(dummy_sampler.get_variable('prl').finalValue)

    jde_mcmc_sampler_m1 = \
      physio_build_jde_mcmc_sampler(nb_iterations_m1, 'basic_regularized',
                                    prf_ini=prf_ini_m1,
                                    do_sampling_prf=False,
                                    prls_ini=prls_ini_m1,
                                    do_sampling_prls=False,
                                    brf_var_ini=brf_var,
                                    do_sampling_brf_var=do_sampling_brf_var)

    pyhrf.verbose(2, 'JDE first pass -> BOLD fit')
    analyser_m1 = JDEMCMCAnalyser(jde_mcmc_sampler_m1, copy_sampler=False,
                                  dt=dt)
    analyser_m1.set_pass_errors(False)
    tjde_mcmc_m1 = FMRITreatment(fmri_data, analyser_m1,
                                 output_dir=jde_output_dir)
    outputs_m1, fns_m1 = tjde_mcmc_m1.run()


    jde_output_dir = op.join(output_dir, 'perf_only_from_res')
    if not op.exists(jde_output_dir):
        os.makedirs(jde_output_dir)


    brf_m1 = jde_mcmc_sampler_m1.get_variable('brf').finalValue
    prf_m2 = np.dot(jde_mcmc_sampler_m1.get_variable('prf').omega_operator,
                    brf_m1)

    #force begin & end to be zero
    #TODO: rather fix omega to avoid this kind of thing or
    # use zero-contrainst again
    prf_m2[0] = 0.
    prf_m2[-5:] = 0.

    brls_m1 = jde_mcmc_sampler_m1.get_variable('brl').finalValue
    labels_m1 = jde_mcmc_sampler_m1.get_variable('label').finalValue >= 0.5
    labels_m1 = labels_m1.astype(np.int32)

    perf_bl_m1 = jde_mcmc_sampler_m1.get_variable('perf_baseline').finalValue
    drift_m1 = jde_mcmc_sampler_m1.get_variable('drift_coeff').finalValue
    noise_var_m1 = jde_mcmc_sampler_m1.get_variable('noise_var').finalValue

    pyhrf.verbose(2, 'ASL JDE second pass -> Perfusion fit')
    jde_mcmc_sampler_m2 = \
      physio_build_jde_mcmc_sampler(nb_iterations_m2, 'basic_regularized',
                                    prf_ini=prf_m2,
                                    do_sampling_prf=False,
                                    brls_ini=brls_m1,
                                    do_sampling_brls=False,
                                    brf_ini=brf_m1,
                                    do_sampling_brf=False,
                                    brf_var_ini=brf_var,
                                    do_sampling_brf_var=False,
                                    perf_bl_ini=perf_bl_m1,
                                    drift_ini=drift_m1,
                                    noise_var_ini=noise_var_m1,
                                    labels_ini=labels_m1,
                                    do_sampling_labels=True)

    analyser_m2 = JDEMCMCAnalyser(jde_mcmc_sampler_m2, dt=dt)
    analyser_m2.set_pass_errors(False)
    tjde_mcmc_m2 = FMRITreatment(fmri_data, analyser_m2,
                                 output_dir=jde_output_dir)
    outputs_m2, fns_m2 = tjde_mcmc_m2.run()

    # combine outputs
    outputs = {}
    for o in ['brf_pm', 'brl_pm']:
        outputs[o] = outputs_m1[o]

    outputs['noise_var_pm_m1'] = outputs_m1['noise_var_pm']

    for o in ['prf_pm', 'prl_pm', 'label_pm']:
        outputs[o] = outputs_m2[o]
    outputs['noise_var_pm_m2'] = outputs_m2['noise_var_pm']

    return outputs


def dummy_jde(fmri_data, dt):
    print 'run dummy_jde ...'
    output_dir = pyhrf.get_tmp_path()
    jde_mcmc_sampler = \
      physio_build_jde_mcmc_sampler(3, 'basic_regularized', do_sampling_prf=False,
                                    do_sampling_brf=False,
                                    do_sampling_prls=False,
                                    do_sampling_labels=False,
                                    brf_var_ini=np.array([0.1]))

    analyser = JDEMCMCAnalyser(jde_mcmc_sampler, copy_sampler=False, dt=dt)
    analyser.set_pass_errors(False)
    print 'sampler:', jde_mcmc_sampler
    tjde_mcmc = FMRITreatment(fmri_data, analyser, output_dir=output_dir)
    outputs, fns = tjde_mcmc.run()
    print' tjde_mcmc.analyser.sampler:',  tjde_mcmc.analyser.sampler
    print 'dummy_jde done!'
    return tjde_mcmc.analyser.sampler



def physio_build_jde_mcmc_sampler(nb_iterations,
                                  rf_prior,
                                  brf_var_ini=None,
                                  prf_var_ini=None,
                                  do_sampling_brf_var=False,
                                  do_sampling_prf_var=False,
                                  prf_ini=None,
                                  do_sampling_prf=True,
                                  prls_ini=None,
                                  do_sampling_prls=True,
                                  brf_ini=None,
                                  do_sampling_brf=True,
                                  brls_ini=None,
                                  do_sampling_brls=True,
                                  perf_bl_ini=None,
                                  drift_ini=None,
                                  noise_var_ini=None,
                                  labels_ini=None,
                                  do_sampling_labels=True,
                                  ):
    """
    """
    import pyhrf.jde.asl_physio as jap
    sampler_params = {
            'nb_iterations' : nb_iterations,
            'smpl_hist_pace' : -1,
            'obs_hist_pace' : -1,
            'brf' : \
            jap.PhysioBOLDResponseSampler(val_ini=brf_ini,
                                          zero_constraint=False,
                                          do_sampling=do_sampling_brf,
                                          use_true_value=False),
            'brf_var' : \
                jap.PhysioBOLDResponseVarianceSampler(\
                    val_ini=np.array([brf_var_ini]),
                    do_sampling=do_sampling_brf_var),
            'prf' : \
                jap.PhysioPerfResponseSampler(val_ini=prf_ini,
                                              zero_constraint=False,
                                              do_sampling=do_sampling_prf,
                                              use_true_value=False,
                                              prior_type=rf_prior),
            'prf_var' : \
            jap.PhysioPerfResponseVarianceSampler(val_ini=np.array([brf_var_ini]),
                                                  do_sampling=False),
            'noise_var' : \
                jap.NoiseVarianceSampler(val_ini=noise_var_ini,
                                         use_true_value=False,
                                         do_sampling=True),
            'drift_var' : \
                jap.DriftVarianceSampler(use_true_value=False,
                                         do_sampling=True),
            'drift' : \
                jap.DriftCoeffSampler(val_ini=drift_ini,
                                      use_true_value=False,
                                      do_sampling=True),
            'bold_response_levels' : \
                jap.BOLDResponseLevelSampler(val_ini=brls_ini,
                                             use_true_value=False,
                                             do_sampling=do_sampling_brls),
            'perf_response_levels' : \
                jap.PerfResponseLevelSampler(val_ini=prls_ini,
                                             use_true_value=False,
                                             do_sampling=do_sampling_prls),
            'bold_mixt_params' : \
                jap.BOLDMixtureSampler(use_true_value=False,
                                       do_sampling=do_sampling_brls),
            'perf_mixt_params' : \
                jap.PerfMixtureSampler(use_true_value=False,
                                       do_sampling=do_sampling_prls),
            'labels' : \
                jap.LabelSampler(val_ini=labels_ini,
                                 use_true_value=False,
                                 do_sampling=do_sampling_labels),
            'perf_baseline' : \
                jap.PerfBaselineSampler(val_ini=perf_bl_ini,
                                        use_true_value=False,
                                        do_sampling=True),
            'perf_baseline_var' : \
                jap.PerfBaselineVarianceSampler(use_true_value=False,
                                                do_sampling=True),
            'check_final_value' : 'none',
        }
    sampler = jap.ASLPhysioSampler(**sampler_params)
    return sampler
