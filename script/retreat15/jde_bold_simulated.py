
"""
Apply the BOLD JDE analysis to artificial BOLD signal (generated from model):

* Directory where to run this script

The script will store data in the current folder (from where it is run).
The current folder *must not* be located in the source directory of pyhrf.

A convenient way to run a script located in the source directory from another
location is to create a shortcut. Say we run it in /my/data/folder, then one
can use the following in shell:

$ cd /my/data/folder
$ pyhrf_script_shortcut ./runme.py -f testing_asl_physio.py

Then launch ./runme.py
"""

import sys
import os
import os.path as op
import numpy as np

import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
from pyhrf.ndarray import xndarray
import pyhrf.jde.models as jdem
import pyhrf.sandbox.physio as phym

pyhrf.logger.setLevel("INFO")


############################
##### JDE BOLD Set Up  #####
############################

def main():

    np.random.seed(48258)

    simulate = True
    do_jde_asl = True
    analyse_vem = True
    analyse_mcmc = False
    scen = 'high_snr'
    tr = 2.5
    dt = tr/2.
    v_noise = 2.0
    
    # Folder names
    fig_prefix = 'bold_TR25_dt' + str(np.round(dt*10).astype(np.int32))\
                    + '_dur15_' + scen + '_'
    simulation_dir = fig_prefix + 'simulated'
    fig_dir = fig_prefix + 'figs'

    # create output folder
    np.random.seed(48258)
    if not op.exists(fig_dir):
        os.makedirs(fig_dir)
    if not op.exists(simulation_dir):
        os.makedirs(simulation_dir)

    if simulate:
        bold_items, conds = simulate_bold(output_dir=simulation_dir,
                                          noise_scenario=scen,
                                          spatial_size='normal', dt=dt)  
    if do_jde_asl:
        vem_output_dir = op.join(simulation_dir, 'jde_vem_analysis')
        if not op.exists(vem_output_dir):
            os.makedirs(vem_output_dir)
        mcmc_output_dir = op.join(simulation_dir, 'jde_mcmc_analysis')
        if not op.exists(mcmc_output_dir):
            os.makedirs(mcmc_output_dir)

        if analyse_vem:
            print 'JDE analysis VEM on simulation ...'
            jde_analyse_vem(simulation_dir, vem_output_dir,
                            bold_items, dt=dt)
        if analyse_mcmc:
            print 'JDE analysis MCMC on simulation ...'
            jde_analyse_mcmc(simulation_dir, mcmc_output_dir,
                             bold_items, dt=dt, nb_its=1000)



def jde_analyse_vem(simulation_dir,output_dir,simulation,constrained=False,
                fast=True, dt=2.):
    # Create an FmriData object directly from the simulation dictionary:
    fmri_data = FmriData.from_simulation_dict(simulation, mask=None)
    print 'dt = ', dt

    #JDE analysis
    jde_vem_analyser = JDEVEMAnalyser(beta=.8, dt=dt, hrfDuration=25.,
                                    nItMax=50, nItMin=9, estimateBeta=True,
                                    computeContrast=False, PLOT=True, 
                                    fast=fast)
                                                     
    tjde_vem = FMRITreatment(fmri_data=fmri_data, analyser=jde_vem_analyser, 
                             output_dir=output_dir)
    tjde_vem.run()
    return 


def build_jde_mcmc_sampler(nbIterations, estimHrf=True, estimBeta=True,
                           beta=.8, hrfVar=.01, use_true_nrls=False,
                           use_true_labels=False):
    """ Build a JDE MCMC sampler object with the given parameters

    Args:
        - nbIterations (int): number of iterations for the Gibbs Sampling
                              (recommanded: 1000 to 3000)
        - estimHrf (bool): flag to estimate the HRF or not. If not, then it
                           is set to the canonical HRF.
        - estimBeta (bool): flag to estimate beta (spatial regularization factor).
                            If not estimated, then it is fixed to the value
                            of the argument *beta*.
        - beta (float): initial value of the spatial regularization factor.
        - hrfVar (float): variance of the HRF (default is fine on simulation)
        - use_true_nrls (bool): flag to initializ the NRLs to their simulated
                                values
        - use_true_labels (bool): flag to initialize the labels (= state of
                                  voxels -> activated or nont) to their
                                  simulated values.

    Return:
         instance of pyhrf.jde.models.BOLDGibbsSampler

    """
    from pyhrf.jde.models import BOLDGibbsSampler as BG
    from pyhrf.jde.beta import BetaSampler as BS
    from pyhrf.jde.nrl import NRLSampler as NS
    from pyhrf.jde.hrf import RHSampler as HVS
    from pyhrf.jde.hrf import HRFSampler as HS

    sampler = BG(nb_iterations=nbIterations, 
                 beta=BS(do_sampling=estimBeta, val_ini=np.array([beta])),
                 hrf=HS(do_sampling=estimHrf), 
                 hrf_var=HVS(do_sampling=False, val_ini=np.array([hrfVar])),
                 response_levels=NS(use_true_nrls=use_true_nrls,
                                    use_true_labels=use_true_labels))

    return sampler



def jde_analyse_mcmc(simulation_dir, output_dir, simulation, dt=2., nb_its=1000):
    # Pack simulation into a pyhrf.core.FmriData object
    #fmri_data = FmriData.from_vol_files(mask_file, paradigm_file, [bold_file], tr)
    fmri_data = FmriData.from_simulation_dict(simulation, mask=None)
    print 'dt = ', dt

    # JDE
    jde_mcmc_sampler = build_jde_mcmc_sampler(nb_its, 
                                              use_true_nrls=False,
                                              use_true_labels=True)
    analyser = JDEMCMCAnalyser(jde_mcmc_sampler, dt=dt)
    tjde_mcmc = FMRITreatment(fmri_data, analyser, output_dir=output_dir,
                              make_outputs=True)
    tjde_mcmc.run()


##################
### Simulation ###
##################

#from pyhrf.boldsynth.scenarios import *

import pyhrf.boldsynth.scenarios as sim
from pyhrf import Condition
from pyhrf.tools import Pipeline

def simulate_bold(output_dir=None, noise_scenario='high_snr', v_noise=None,
                  spatial_size='tiny', normalize_hrf=True, dt=0.5):
    
    drift_var = 10.
    tr = 2.5
    dsf = tr/dt

    import pyhrf.paradigm as mpar
    paradigm_csv_file = './../paradigm_data/paradigm_bilateral_v2_no_final_rest.csv'
    paradigm_csv_delim = ' '
    
    paradigm = mpar.Paradigm.from_csv(paradigm_csv_file,
                                      delim=paradigm_csv_delim)
    print 'Paradigm information: '
    print paradigm.get_info()
    condition_names = paradigm.get_stimulus_names()
    #stop
    lmap1, lmap2, lmap3, lmap4 = 'ghost', 'icassp13', 'stretched_1', 'pacman'

    print 'creating condition response levels...'
    if noise_scenario == 'high_snr':
        v_noise = v_noise or 0.05
        conditions = [
            Condition(name=condition_names[0], m_act=15., v_act=.1,
                      v_inact=.2, label_map=lmap1),
            Condition(name=condition_names[1], m_act=14., v_act=.11,
                      v_inact=.21, label_map=lmap2),
            Condition(name=condition_names[2], m_act=15., v_act=.1,
                      v_inact=.2, label_map=lmap3),
            Condition(name=condition_names[3], m_act=14., v_act=.11, 
                      v_inact=.21, label_map=lmap4),
        ]
    elif noise_scenario == 'low_snr_low_prl':
        v_noise = v_noise or 7.
        scale = .3
        conditions = [
            Condition(name=condition_names[0], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name=condition_names[1], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap2),
            Condition(name=condition_names[2], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap3),
            Condition(name=condition_names[3], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap4),
                      ]
    else:  # low_snr
        v_noise = v_noise or 2.
        conditions = [
            Condition(name=condition_names[0], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name=condition_names[1], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap2),
            Condition(name=condition_names[2], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap3),
            Condition(name=condition_names[3], m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap4),
        ]

    print 'creating simulation steps...'
    #from pyhrf.sandbox.physio_params import create_omega_prf, PHY_PARAMS_KHALIDOV11
    #brf = create_canonical_hrf(dt=dt)
    simulation_steps = {
        'dt': dt,
        'dsf': dsf,
        'tr': dt * dsf,
        'condition_defs': conditions,
        # Paradigm
        'paradigm': paradigm, #sim.create_localizer_paradigm_avd,
        'rastered_paradigm': sim.rasterize_paradigm,
        # Labels
        'labels_vol': sim.create_labels_vol,
        'labels': sim.flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # NRLs
        'nrls': sim.create_time_invariant_gaussian_nrls,
        # HRF
        'hrf_var': 0.1,
        'primary_hrf': sim.create_canonical_hrf(dt=dt),
        'normalize_hrf': normalize_hrf,
        'hrf': sim.duplicate_hrf,
        # Stim induced
        'stim_induced_signal': sim.create_stim_induced_signal,
        # Noise
        'v_noise': v_noise,
        'noise': sim.create_gaussian_noise,
        # Drift
        'drift_order': 4,
        'drift_coeff_var': drift_var,
        'drift_coeffs': sim.create_drift_coeffs,
        'drift': sim.create_polynomial_drift_from_coeffs,
        # Final BOLD signal
        'bold_shape': sim.get_bold_shape,
        'bold': sim.create_bold_from_stim_induced,
    }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    simulation = simu_graph.get_values()

    if output_dir is not None:
        try:
            simu_graph.save_graph_plot(op.join(output_dir,
                                               'simulation_graph.png'))
        except ImportError:  # if pygraphviz not available
            pass

        sim.simulation_save_vol_outputs(simulation, output_dir)

    return simulation, condition_names


#############
#### run ####
#############

if __name__ == '__main__':
    main()
