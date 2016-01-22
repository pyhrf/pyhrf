#import sys
import os
import os.path as op
import numpy as np

import pyhrf
import pyhrf.paradigm
from pyhrf.core import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.vb_jde_analyser_bold_fast import JDEVEMAnalyser
from pyhrf.ndarray import xndarray
from pyhrf import logging

from pyhrf.jde.asl_2steps import jde_analyse_2steps_v1
from pyhrf.sandbox.physio_params import PHY_PARAMS_FRISTON00, PHY_PARAMS_DONNET06
from pyhrf.sandbox.physio_params import PHY_PARAMS_DENEUX06, PHY_PARAMS_HAVLICEK11
from pyhrf.sandbox.physio_params import PHY_PARAMS_KHALIDOV11, PHY_PARAMS_DUARTE12


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():

    #np.random.seed(np.random.randint(0, 1000000))
    np.random.seed(48258)

    te = 0.011
    tr = 2.5
    dt = tr / 2.
    n_scans = 164
    frametimes = np.arange(0, n_scans * tr, tr)

    phy_params = PHY_PARAMS_KHALIDOV11
    phy_params['TE'] = te
    vem = True

    archives = './archives'
    #subjects = ['CD110147', 'RG130377']
    subjects = ['RG130377']

    for subject in subjects:

        print 'Subject ', subject        
        
        # Load data for condition cond and subject subject
        data_dir = op.join(archives, subject)
        data_analysed_dir = op.join('/home/aina/Data/ASL/Data', 'HEROES_analysis', subject)
        fdata = load_data(archives, data_dir, data_analysed_dir, subject, tr)
        
        output_dir = op.join('./', subject, 'jde_results')
        if not op.exists(output_dir): os.makedirs(output_dir)
        
        if vem:
            print 'JDE VEM analysis on real data ...'
            vem_output_dir = op.join(output_dir, 'vem_complete_' + prior)
            if not op.exists(vem_output_dir): os.makedirs(vem_output_dir)
            jde_analyse_vem(vem_output_dir, fdata, dt=dt, nItMin=2,
                            use_hyperprior=True)
            print 'JDE VEM analysis on real data done!'
        else:
            prf_var = 0.00000001
            brf_var = 0.01     
            nbit = 3000       
            mcmc_output_dir = op.join(output_dir, 'mcmc')
            if not op.exists(mcmc_output_dir): os.makedirs(mcmc_output_dir)            
            jde_mcmc_sampler = jde_analyse(mcmc_output_dir, fdata, dt,
                                    nb_iterations=nbit,
                                    rf_prior='physio_stochastic_regularized',
                                    brf_var=brf_var, prf_var=prf_var, 
                                    phy_params=phy_params,
                                    do_sampling_brf_var=False,
                                    do_sampling_prf_var=False)

        del fdata


def bold_mean_and_range(bold_fn, gm_fn):
    """
    Returns signal mean and range
    """
    #gm = xndarray.load(gm_fn).data
    #print gm.shape
    bold = xndarray.load(bold_fn).data
    print 'BOLD shape ', bold.shape
    bold_mean = np.mean(bold) #[np.where(gm > 0)])
    bold_range = (np.max(bold) - np.min(bold))
    del bold #, gm
    print 'BOLD mean ', bold_mean
    print 'BOLD range ', bold_range
    return bold_mean, bold_range


def load_data(archives, data_dir, data_analysed_dir, subject, tr):
    """
    Load data and create 
    """
    # Folder and file names
    data_fn = op.join(data_dir, 'fMRI', 
              'wrvismot2_BOLDepi1_' + subject.lower() + '_20141112_acq1_08.nii')
    gm_fn = op.join(data_dir, 'anat', 'c1' + subject + '_anat-0001.nii')
    paradigm_fn = op.join('./paradigm_data', 'paradigm_bilateral_v2_no_final_rest.csv')
    parcel_dir = op.join(data_analysed_dir, 'parcellation')
    roi_mask_fn = op.join(parcel_dir, 'parcellation_func.nii')
    #roi_mask_fn = op.join(parcel_dir, 'parcels_to_analyse.nii')
    mask = xndarray.load(roi_mask_fn).data
    print 'Mask shape ', mask.shape
    mask_mean = np.mean(mask) #[np.where(gm > 0)])
    mask_range = (np.max(mask) - np.min(mask))
    print 'Mask mean ', mask_mean
    print 'Mask range ', mask_range

    # Loading and scaling data
    data_mean, data_range = bold_mean_and_range(data_fn, gm_fn)            
    fdata = FmriData.from_vol_files(roi_mask_fn, paradigm_fn, [data_fn], tr)
    fdata.bold = (fdata.bold - data_mean) * 100 / data_range    
    print 'mean bold = ', np.mean(fdata.bold)
    print 'shape bold = ', fdata.bold.shape

    return fdata


def jde_analyse_vem(output_dir, fmri_data, dt=0.5, physio=True, nItMin=2,
                    use_hyperprior=False):
    """
    Runs JDE VEM sampler
    """
    if not op.exists(output_dir): os.makedirs(output_dir)
    contrasts = {"checkerboard_motor_d5000_left-checkerboard_motor_d5000_right": 
                  "checkerboard_motor_d5000_left-checkerboard_motor_d5000_right",
                 "checkerboard_motor_d2500_left-checkerboard_motor_d2500_right": 
                  "checkerboard_motor_d2500_left-checkerboard_motor_d2500_right", 
                 "checkerboard_motor_d2500_left-checkerboard_motor_d5000_left": 
                  "checkerboard_motor_d2500_left-checkerboard_motor_d5000_left", 
                 "checkerboard_motor_d2500_right-checkerboard_motor_d5000_right": 
                  "checkerboard_motor_d2500_right-checkerboard_motor_d5000_right"}
    vh = 0.0001 #0.0001
    gamma_h = 1000  # 10000000000  # 7.5 #100000
    jde_vem_analyser = JDEVEMAnalyser(beta=1., dt=dt, hrfDuration=25.,
                            nItMax=100, nItMin=nItMin, PLOT=True,
                            sigmaH=vh, gammaH=gamma_h, constrained=True,
                            contrasts=contrasts, computeContrast=True)
    tjde_vem = FMRITreatment(fmri_data=fmri_data, analyser=jde_vem_analyser,
                             output_dir=output_dir)
    tjde_vem.run() #parallel='local', n_jobs=16)
    return


def jde_analyse(output_dir, fmri_data, dt, nb_iterations, rf_prior,
                brf_var, prf_var, phy_params = PHY_PARAMS_FRISTON00, 
                do_sampling_brf_var=False, do_sampling_prf_var=False, 
                do_basic_nN=False):
    """
    Return:
        result of FMRITreatment.run(), that is: (dict of outputs, output fns)
    """
    jde_mcmc_sampler = physio_build_jde_mcmc_sampler(nb_iterations, 
                                rf_prior, phy_params, brf_var, prf_var,
                                do_sampling_brf_var, do_sampling_prf_var,
                                do_basic_nN=do_basic_nN)

    analyser = JDEMCMCAnalyser(jde_mcmc_sampler, dt=dt)
    analyser.set_pass_errors(False) #do not bypass errors during sampling
                                    #default initialization sets this true
    tjde_mcmc = FMRITreatment(fmri_data, analyser, output_dir=output_dir)
    tjde_mcmc.run()
    
    return jde_mcmc_sampler



def physio_build_jde_mcmc_sampler(nb_iterations, rf_prior, phy_params,
                                  brf_var_ini=None, prf_var_ini=None,
                                  do_sampling_brf_var=False,
                                  do_sampling_prf_var=False,
                                  prf_ini=None, do_sampling_prf=True,
                                  prls_ini=None, do_sampling_prls=True,
                                  brf_ini=None, do_sampling_brf=True,
                                  brls_ini=None, do_sampling_brls=True,
                                  perf_bl_ini=None, drift_ini=None,
                                  noise_var_ini=None, labels_ini=None,
                                  do_sampling_labels=True, 
                                  do_basic_nN=False):
    """
    """
    #import pyhrf.jde.asl_physio as jap
    if rf_prior=='physio_stochastic_regularized' or do_basic_nN:
        import pyhrf.jde.asl_physio_1step_params as jap
        norm = 0.
    else:
        import pyhrf.jde.asl_physio as jap
        norm = 1.
    
    zc = False

    sampler_params = {
            'nb_iterations' : nb_iterations,
            'smpl_hist_pace' : -1,
            'obs_hist_pace' : -1,
            'brf' : \
                jap.PhysioBOLDResponseSampler(phy_params=phy_params,
                                          val_ini=brf_ini,
                                          zero_constraint=zc,
                                          normalise = norm,
                                          do_sampling=do_sampling_brf,
                                          use_true_value=False),
            'brf_var' : \
                jap.PhysioBOLDResponseVarianceSampler(\
                    val_ini=np.array([brf_var_ini]),
                    do_sampling=do_sampling_brf_var),
            'prf' : \
                jap.PhysioPerfResponseSampler(phy_params=phy_params,
                                              val_ini=prf_ini,
                                              zero_constraint=zc,
                                              normalise = norm,
                                              do_sampling=do_sampling_prf,
                                              use_true_value=False,
                                              prior_type=rf_prior),
            'prf_var' : \
                jap.PhysioPerfResponseVarianceSampler(\
                    val_ini=np.array([prf_var_ini]), do_sampling=False),
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


if __name__ == '__main__':
    main()
