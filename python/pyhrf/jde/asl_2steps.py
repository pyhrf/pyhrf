import os
import os.path as op
import numpy as np
import pyhrf
import copy

from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser


def jde_analyse_2steps_v1(output_dir, fmri_data, dt, nb_iterations, brf_var=None,
                 do_sampling_brf_var=False, prf_var=None, do_sampling_prf_var=False):
    """
    #Return:
    #    dict of outputs
    """    
    nb_iterations_m1 = nb_iterations_m2 = nb_iterations

    jde_output_dir = op.join(output_dir, 'bold_only')
    if not op.exists(jde_output_dir):
        os.makedirs(jde_output_dir)

    dummy_sampler = dummy_jde(fmri_data, dt)
    
    prf_ini_m1 = np.zeros_like(dummy_sampler.get_variable('prf').finalValue)
    prls_ini_m1 = np.zeros_like(dummy_sampler.get_variable('prl').finalValue)
    #perf_bl_ini_m1 = np.zeros_like(dummy_sampler.get_variable('perf_baseline').finalValue)

    jde_mcmc_sampler_m1 = \
      physio_build_jde_mcmc_sampler(nb_iterations_m1, 'basic_regularized',
                                    prf_ini=prf_ini_m1,
                                    do_sampling_prf=False,
                                    prls_ini=prls_ini_m1,
                                    do_sampling_prls=False,
                                    brf_var_ini=brf_var,
                                    do_sampling_brf_var=do_sampling_brf_var,
                                    prf_var_ini=brf_var,
                                    do_sampling_prf_var=False,
                                    flag_zc = False)

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
    omega = jde_mcmc_sampler_m1.get_variable('prf').omega_operator
    print omega.shape
    print np.concatenate(([0],[0],[0],brf_m1,[0],[0],[0])).shape
    #prf_m2 = np.dot(omega, brf_m1)
    prf_m2 = np.dot(omega, np.concatenate(([0],[0],[0],brf_m1,[0],[0],[0])))[3:-3]
    
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(brf_m1)
        plt.plot(prf_m2)
        plt.show()
        
    if 0:
        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].matshow(omega)
        axarr[0, 1].plot(prf_m2)
        import copy
        omega2 = copy.deepcopy(omega)
        omega2[-5:,:5] = 0.
        omega2[:5,-5:] = 0.
        prf_m2 = np.dot(omega2, brf_m1)
        axarr[1, 0].matshow(omega2)
        axarr[1, 1].plot(prf_m2)
        plt.show()
    
    #print 'PRF M2 shape = ', prf_m2.shape
    #force begin & end to be zero
    #XXX TODO: rather fix omega to avoid this kind of thing or use zero-contrainst again
    #prf_m2[0] = 0.
    #prf_m2[-5:] = 0.
    brls_m1 = jde_mcmc_sampler_m1.get_variable('brl').finalValue
    labels_m1 = jde_mcmc_sampler_m1.get_variable('label').finalValue >= 0.5
    labels_m1 = labels_m1.astype(np.int32)
    drift_m1 = jde_mcmc_sampler_m1.get_variable('drift_coeff').finalValue
    drift_var_m1 = jde_mcmc_sampler_m1.get_variable('drift_var').finalValue
    noise_var_m1 = jde_mcmc_sampler_m1.get_variable('noise_var').finalValue

    pyhrf.verbose(2, 'ASL JDE second pass -> Perfusion fit')
    pyhrf.verbose(1, 'Physiological prior stochastic regularized')
    
    jde_mcmc_sampler_m2 = \
      physio_build_jde_mcmc_sampler(nb_iterations_m2, 'physio_stochastic_not_regularized',
                                    #prf_ini=prf_m2,
                                    do_sampling_prf=True,
                                    flag_zc=True,
                                    prf_var_ini = prf_var,
                                    do_sampling_prf_var = do_sampling_prf_var,
                                    brls_ini=brls_m1, #if prls_ini_m1, it should be the same
                                    do_sampling_brls=False,
                                    brf_ini=brf_m1, #if prf_ini_m1, PRF sampler 'basic_regularized'
                                    do_sampling_brf=False,
                                    brf_var_ini=brf_var,
                                    do_sampling_brf_var=False,
                                    #perf_bl_ini=perf_bl_m1,
                                    drift_ini=drift_m1,
                                    do_sampling_drift = False,
                                    drift_var_ini = drift_var_m1,
                                    do_sampling_drift_var = False,
                                    noise_var_ini=noise_var_m1,
                                    labels_ini=labels_m1,
                                    do_sampling_labels=True)
                                    
    """ available_priors = ['physio_stochastic_regularized',
                            'physio_stochastic_not_regularized',
                            'physio_deterministic',
                            'physio_deterministic_hack',
                            'basic_regularized']                """                        
                            
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
    
    return outputs_m1
    
    
"""    
def jde_analyse_2steps(output_dir, fmri_data, dt, nb_iterations, brf_var=None,
                       do_sampling_brf_var=False):

    
    #Return:
    #    dict of outputs
    

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
                                    do_sampling_brf_var=do_sampling_brf_var,
                                    prf_var_ini=brf_var,
                                    do_sampling_prf_var=False)
                                    
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
    #prf_m2[0] = 0.
    #prf_m2[-5:] = 0.

    brls_m1 = jde_mcmc_sampler_m1.get_variable('brl').finalValue
    labels_m1 = jde_mcmc_sampler_m1.get_variable('label').finalValue >= 0.5
    labels_m1 = labels_m1.astype(np.int32)

    perf_bl_m1 = jde_mcmc_sampler_m1.get_variable('perf_baseline').finalValue
    drift_m1 = jde_mcmc_sampler_m1.get_variable('drift_coeff').finalValue
    drift_var_m1 = jde_mcmc_sampler_m1.get_variable('drift_var').finalValue
    noise_var_m1 = jde_mcmc_sampler_m1.get_variable('noise_var').finalValue

    pyhrf.verbose(2, 'ASL JDE second pass -> Perfusion fit')
    jde_mcmc_sampler_m2 = \
      physio_build_jde_mcmc_sampler(nb_iterations_m2, 'basic_regularized',
                                    flag_zc = True,
                                    prf_ini=prf_m2,
                                    do_sampling_prf=False,
                                    prf_var_ini=brf_var,
                                    do_sampling_prf_var=False,
                                    brls_ini=brls_m1,
                                    do_sampling_brls=False,
                                    brf_ini=brf_m1,
                                    do_sampling_brf=False,
                                    brf_var_ini=brf_var,
                                    do_sampling_brf_var=False,
                                    perf_bl_ini=perf_bl_m1,
                                    drift_ini=drift_m1,
                                    do_sampling_drift = False,
                                    drift_var_ini = drift_var_m1,
                                    do_sampling_drift_var = False,
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
"""

def dummy_jde(fmri_data, dt):
    print 'run dummy_jde ...'
    jde_mcmc_sampler = \
        physio_build_jde_mcmc_sampler(3, 'basic_regularized', 
                                        do_sampling_prf=False,
                                        do_sampling_brf=False,
                                        do_sampling_prls=False,
                                        do_sampling_labels=False,
                                        do_sampling_prf_var=False,
                                        do_sampling_brf_var=False,
                                        brf_var_ini=np.array([0.1]),
                                        prf_var_ini=np.array([0.1]))

    analyser = JDEMCMCAnalyser(jde_mcmc_sampler, copy_sampler=False, dt=dt)
    analyser.set_pass_errors(False)
    tjde_mcmc = FMRITreatment(fmri_data, analyser, output_dir=None)
    outputs, fns = tjde_mcmc.run()
    print 'dummy_jde done!'
    return tjde_mcmc.analyser.sampler



def physio_build_jde_mcmc_sampler(nb_iterations,
                                  rf_prior,
                                  flag_zc = False,
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
                                  do_sampling_perf_bl=True,
                                  do_sampling_perf_var=True,
                                  drift_ini=None,
                                  do_sampling_drift = True,
                                  drift_var_ini=None,
                                  do_sampling_drift_var = True,
                                  noise_var_ini=None,
                                  labels_ini=None,
                                  do_sampling_labels=True,
                                  ):
    """
    """
    #import pyhrf.jde.asl_physio as jap
    import pyhrf.jde.asl_physio_1step as jap
    #print 'brf_var_ini', brf_var_ini
    #print 'prf_var_ini', prf_var_ini
    sampler_params = {
            'nb_iterations' : nb_iterations,
            'smpl_hist_pace' : -1,
            'obs_hist_pace' : -1,
            'brf' : \
            jap.PhysioBOLDResponseSampler(val_ini=brf_ini,
                                          zero_constraint=flag_zc,
                                          do_sampling=do_sampling_brf,
                                          normalise = 0.,
                                          use_true_value=False),
            'brf_var' : \
                jap.PhysioBOLDResponseVarianceSampler(\
                    val_ini=np.array([brf_var_ini]),
                    do_sampling=do_sampling_brf_var),
            'prf' : \
                jap.PhysioPerfResponseSampler(val_ini=prf_ini,
                                              zero_constraint=flag_zc,
                                              do_sampling=do_sampling_prf,
                                              normalise = 0.,
                                              smooth_order = 2,
                                              use_true_value=False,
                                              prior_type=rf_prior),
            'prf_var' : \
                jap.PhysioPerfResponseVarianceSampler(\
                    val_ini=np.array([prf_var_ini]),
                    do_sampling=do_sampling_prf_var),
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
                                        do_sampling=do_sampling_perf_bl),
            'perf_baseline_var' : \
                jap.PerfBaselineVarianceSampler(use_true_value=False,
                                                do_sampling=do_sampling_perf_var),
            'check_final_value' : 'none',
            'output_fit': True,
        }
    sampler = jap.ASLPhysioSampler(**sampler_params)
    return sampler

