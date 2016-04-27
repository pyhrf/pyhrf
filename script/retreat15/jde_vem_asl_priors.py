
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
import time

import pyhrf
import pyhrf.jde.asl as jdem
import pyhrf.sandbox.physio as phym
from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.vb_jde_analyser_asl_fast import JDEVEMAnalyser
from pyhrf.ndarray import xndarray
from pyhrf.sandbox.physio_params import PHY_PARAMS_FRISTON00, PHY_PARAMS_KHALIDOV11

import matplotlib.pyplot as plt

# Let's use TeX to render text
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='sans serif', size=23)


############################
##### JDE BOLD Set Up  #####
############################

def main():

    np.random.seed(48258)

    prior_types = np.array(['balloon'])
    #noise_scenarios = np.array(['low_snr', 'high_snr'])
    noise_scenarios = np.array(['low_snr_scale'])
    #noise_scenarios = np.array(['high_snr'])


    # Initialisation values
    simulate = True
    do_jde_asl = True
    tr = 2.5
    dt = tr / 2.
    s = 0.5

    v_noise_range = np.arange(10.0, 10.3, 1.)
    snr_range = np.zeros_like(v_noise_range)
    hyp_opt = np.array([True])
    pos_opt = np.array([False])
    cons_opt = np.array([False])

    n_method = len(hyp_opt) * len(pos_opt)
    print v_noise_range
    error = np.zeros((len(v_noise_range), n_method, 4))
    error2 = np.zeros((len(v_noise_range), n_method, 4))
    block_opt = np.array([True])

    for prior in prior_types:

        for noise_scen in noise_scenarios:

            for block in block_opt:

                for hyp in hyp_opt:

                    for cons in cons_opt:

                        fig_prefix = 'prior' + prior + '_very' + noise_scen + '_block' * block + '_hyp' * hyp + '_cons' * cons
                        simulation_dir = fig_prefix + '_simulated'
                        fig_dir = fig_prefix + '_figs'
                        if not op.exists(fig_dir):
                            os.makedirs(fig_dir)
                        if not op.exists(simulation_dir):
                            os.makedirs(simulation_dir)

                        for pos in pos_opt:
                            for ivn, v_noise in enumerate(v_noise_range):
                                print 'Generating BOLD data ...'
                                print 'index v_noise = ', ivn
                                print 'v_noise = ', v_noise
                                if simulate:
                                    if block:
                                        # HEROES
                                        asl_items, conds = simulate_asl(output_dir=simulation_dir,
                                                                      noise_scenario=noise_scen,
                                                                      v_noise=v_noise, dt=dt, scale=s)
                                    else:
                                        # AINSI
                                        conds = ['audio', 'video']
                                        asl_items = jdem.simulate_asl(output_dir=simulation_dir,
                                                                      noise_scenario='low_snr',
                                                                      spatial_size='normal',
                                                                      v_noise=v_noise, dt=dt)

                                    Y = asl_items['bold']
                                    n = asl_items['noise']
                                    print 'noise mean = ', np.mean(n)
                                    print 'noise var = ', np.var(n)
                                    p = asl_items['perf_baseline']
                                    print 'perfusion baseline mean = ', np.mean(p)
                                    print 'perfusion baseline var = ', np.var(p)
                                    l = asl_items['drift']
                                    print 'drifts mean = ', np.mean(l)
                                    print 'drifts var = ', np.var(l)
                                    snr_range[ivn] = 20 * (np.log(np.linalg.norm(Y) / \
                                        np.linalg.norm(n))) / np.log(10.)
                                norm = plot_jde_inputs(simulation_dir, fig_dir,
                                                       'simu_vn' + str(np.round(v_noise*10).astype(np.int32)) + '_', conds)
                                print 'Finished generation of ASL data.'

                                old_output_dir2 = op.join(simulation_dir, 'jde_analysis_vem')
                                if do_jde_asl:
                                    print 'JDE VEM analysis'
                                    if not op.exists(old_output_dir2):
                                        os.makedirs(old_output_dir2)
                                    print 'JDE analysis VEM on simulation ...'
                                    t2 = time.time()
                                    jde_vem_sampler = jde_analyse_vem(simulation_dir, old_output_dir2, asl_items,
                                                                      do_physio=True, positivity=pos,
                                                                      use_hyperprior=hyp, dt=(dt), nItMin=6,
                                                                      constrained=cons, prior=prior)
                                    print time.time() - t2
                                plot_jde_outputs(old_output_dir2, fig_dir, 'vem_', norm, conds, asl_items=asl_items)#, dt_est=dt)
                                print 'printing HRF results together...'
                                plot_jde_rfs(simulation_dir, old_output_dir2, fig_dir,
                                             'vn' + str(v_noise.astype(np.int32)) + '_', asl_items)



def jde_analyse_vem(simulation_dir, output_dir, simulation, constrained=False,
                fast=False, do_physio=True, positivity = False,
                use_hyperprior=False, prior='omega', dt=0.5, nItMin=10):
    # Create an FmriData object directly from the simulation dictionary:
    fmri_data = FmriData.from_simulation_dict(simulation, mask=None)
    vh = 0.00001 #0.0001
    vg = 0.00001 #0.0001
    gamma_h = 1000000000  # 10000000000
    gamma_g = 1000000000
    do = True
    do2 = False
    phy_params=PHY_PARAMS_KHALIDOV11.copy()
    """
    PHY_PARAMS_KHALIDOV11 = { 'model_name': 'Khalidov11',
    'tau_s': 1.54, 'tau_m': 0.98, 'tau_f': 2.46,
    'eps': .54, 'alpha_w': .33, 'E0': .34, 'V0': 1, 'e': 1.43,
    'TE': 0.018,
    'model': 'RBM', 'linear': False, 'obata': False, 'buxton': False } """
    slightly_different = False
    very_different_brf = False
    very_different_prf = False
    if slightly_different:
        phy_params['tau_s'] = 1.74 #2.74
        phy_params['tau_m'] = 1.1 #1.8
        phy_params['tau_f'] = 2.66 #3.1
        phy_params['eps'] = 0.74#1.74
        phy_params['alpha_w'] = 0.53#1.3
        phy_params['E0'] = 0.54#0.94
        phy_params['V0'] = 1.2#2.
        phy_params['e'] = 1. #0.3
    if very_different_brf:
        phy_params['tau_s'] = 2.74
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 3.1
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.3
    if very_different_prf:
        phy_params['tau_s'] = 6.
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 2.
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.3

    jde_vem_analyser = JDEVEMAnalyser(beta=0.8, dt=dt, hrfDuration=25.,
                                      nItMax=100, nItMin=nItMin, PLOT=False,
                                      estimateA=do, estimateH=do, estimateC=do, estimateG=do,
                                      estimateSigmaH=do, sigmaH=vh, gammaH=gamma_h,
                                      estimateSigmaG=do, sigmaG=vg, gammaG=gamma_g,
                                      estimateLabels=do, physio=do_physio, sigmaMu=100.,
                                      estimateBeta=do, estimateMixtParam=do, estimateLA=do,
                                      estimateNoise=do, fast=fast, constrained=constrained,
                                      fmri_data=simulation, positivity=positivity,
                                      use_hyperprior=use_hyperprior, prior=prior,
                                      phy_params=phy_params, zero_constrained=False)
    tjde_vem = FMRITreatment(fmri_data=fmri_data, analyser=jde_vem_analyser,
                             output_dir=output_dir)
    tjde_vem.run()
    return 'asl_'



##################
### Simulation ###
##################

from pyhrf.boldsynth.scenarios import *

def simulate_asl(output_dir=None, noise_scenario='high_snr', v_noise=None,
                 dt=2.5, scale=1.):
    from pyhrf import Condition
    from pyhrf.tools import Pipeline

    #dt = 2.5
    drift_var = 10.
    tr = 2.5
    dsf = tr/dt
    print 'simulated dt = ', dt
    #dt = 2.5
    #dsf = 1  # down sampling factor

    import pyhrf.paradigm as mpar
    #paradigm_csv_file = './../paradigm_data/paradigm_bilateral_vjoint.csv'
    #paradigm_csv_file = './../paradigm_data/paradigm_bilateral_v1_no_final_rest.csv'
    #paradigm_csv_file = './paradigm_data/paradigm_bilateral_v2_no_final_rest_1.csv'
    paradigm_csv_file = './../HEROES_DB/paradigm_data/paradigm_bilateral_v2_no_final_rest.csv'
    paradigm_csv_delim = ' '
    #onsets, durations = load_paradigm(paradigm_csv_file)
    #paradigm2 = mpar.Paradigm(onsets, durations)
    paradigm = mpar.Paradigm.from_csv(paradigm_csv_file, delim=paradigm_csv_delim)
    print 'Paradigm information: '
    print paradigm.get_info()
    condition_names = paradigm.get_stimulus_names()
    lmap1, lmap2, lmap3, lmap4 = 'ghost', 'icassp13', 'stretched_1', 'pacman'

    print 'creating condition response levels...'
    if noise_scenario == 'high_snr':
        v_noise = v_noise or 0.05
        conditions = [
            Condition(name=condition_names[0], perf_m_act=10., perf_v_act=.1, perf_v_inact=.2,
                      bold_m_act=15., bold_v_act=.1, bold_v_inact=.2, label_map=lmap1),
            Condition(name=condition_names[1], perf_m_act=11., perf_v_act=.11, perf_v_inact=.21,
                      bold_m_act=14., bold_v_act=.11, bold_v_inact=.21, label_map=lmap2),
            Condition(name=condition_names[2], perf_m_act=10., perf_v_act=.1, perf_v_inact=.2,
                      bold_m_act=15., bold_v_act=.1, bold_v_inact=.2, label_map=lmap3),
            Condition(name=condition_names[3], perf_m_act=11., perf_v_act=.11, perf_v_inact=.21,
                      bold_m_act=14., bold_v_act=.11, bold_v_inact=.21, label_map=lmap4),
        ]
    elif noise_scenario == 'low_snr_scale':
        v_noise = v_noise or 7.
        #scale = .3
        conditions = [
            Condition(name=condition_names[0], perf_m_act=1.6*scale, perf_v_act=.1, perf_v_inact=.1,
                      bold_m_act=2.2*scale, bold_v_act=.3, bold_v_inact=.3, label_map=lmap1),
            Condition(name=condition_names[1], perf_m_act=1.6*scale, perf_v_act=.1, perf_v_inact=.1,
                      bold_m_act=2.2*scale, bold_v_act=.3, bold_v_inact=.3, label_map=lmap2),
            Condition(name=condition_names[2], perf_m_act=1.6*scale, perf_v_act=.1, perf_v_inact=.1,
                      bold_m_act=2.2*scale, bold_v_act=.3, bold_v_inact=.3, label_map=lmap3),
            Condition(name=condition_names[3], perf_m_act=1.6*scale, perf_v_act=.1, perf_v_inact=.1,
                      bold_m_act=2.2*scale, bold_v_act=.3, bold_v_inact=.3, label_map=lmap4),
                      ]
    else:  # low_snr
        v_noise = v_noise or 2.
        conditions = [
            Condition(name=condition_names[0], perf_m_act=1.6, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3, label_map=lmap1),
            Condition(name=condition_names[1], perf_m_act=1.6, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3, label_map=lmap2),
            Condition(name=condition_names[2], perf_m_act=1.6, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3, label_map=lmap3),
            Condition(name=condition_names[3], perf_m_act=1.6, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3, label_map=lmap4),
        ]

    print 'creating simulation steps...'
    from pyhrf.sandbox.physio_params import create_omega_prf, PHY_PARAMS_KHALIDOV11, \
                                            create_physio_brf, create_physio_prf

    #brf = create_canonical_hrf(dt=dt)
    #prf = create_omega_prf(brf, dt, PHY_PARAMS_KHALIDOV11)
    Thrf = 25.
    phy_params = PHY_PARAMS_KHALIDOV11.copy()
    slightly_different = False
    very_different_brf = False
    very_different_prf = True
    if slightly_different:
        phy_params['tau_s'] = 1.74 #2.74
        phy_params['tau_m'] = 1.1 #1.8
        phy_params['tau_f'] = 2.66 #3.1
        phy_params['eps'] = 0.74#1.74
        phy_params['alpha_w'] = 0.53#1.3
        phy_params['E0'] = 0.54#0.94
        phy_params['V0'] = 1.2#2.
        phy_params['e'] = 1. #0.3
    if very_different_brf:
        phy_params['tau_s'] = 2.74
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 3.1
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.3
    if very_different_prf:
        phy_params['tau_s'] = 6.
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 2.
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.3
    brf = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
    brf /= np.linalg.norm(brf)
    prf = create_physio_prf(phy_params, response_dt=dt, response_duration=Thrf)
    prf /= np.linalg.norm(prf)

    simulation_steps = {
        'dt': dt,
        'dsf': dsf,
        'tr': tr,
        'condition_defs': conditions,
        # Paradigm
        'paradigm': paradigm,  #create_localizer_paradigm_avd,
        'rastered_paradigm': rasterize_paradigm,
        # Labels
        'labels_vol': create_labels_vol,
        'labels': flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # Brls
        'brls': create_time_invariant_gaussian_brls,
        # Prls
        'prls': create_time_invariant_gaussian_prls,
        # BRF
        'primary_brf': brf,
        'brf': duplicate_brf,
        # PRF
        #'primary_prf': create_prf,  # canonical HRF for testing
        'primary_prf': prf,
        'prf': duplicate_prf,
        # Perf baseline
        'perf_baseline': create_perf_baseline,
        'perf_baseline_mean': 0.,
        'perf_baseline_var': .4,
        # Stim induced
        'bold_stim_induced': create_bold_stim_induced_signal,
        'perf_stim_induced': create_perf_stim_induced_signal,
        # Noise
        'v_gnoise': v_noise,
        'noise': create_gaussian_noise_asl,
        # Drift
        'drift_order': 4,
        'drift_var': drift_var,
        'drift_coeffs': create_drift_coeffs_asl,
        'drift': create_polynomial_drift_from_coeffs_asl,
        # Bold # maybe rename as ASL (should be handled afterwards ...
        'ctrl_tag_mat': build_ctrl_tag_matrix,
        'asl_shape': calc_asl_shape,
        'bold': create_asl_from_stim_induced,
    }
    print 'simu_graph'
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    print 'resolve'
    simu_graph.resolve()
    print 'graph'
    simulation = simu_graph.get_values()

    print output_dir
    if output_dir is not None:
        simulation_save_vol_outputs(simulation, output_dir)
    return simulation, condition_names



##################
#### Plotting ####
##################

from pyhrf.plot import autocrop, set_ticks_fontsize, plot_palette


def save_and_crop(fn):
    plt.savefig(fn)
    autocrop(fn)
    plt.close()


def plot_cub_as_curve(c, orientation=None, colors=None, plot_kwargs=None,
                      legend_prefix=''):
    """
    Plot a cuboid (ndims <= 2) as curve(s).
    If the input is 1D: one single curve.
    If the input is 2D :
       * multiple curves are plotted: one for each domain value on the 1st axis
       * legends are shown to display which domain value is associated
         to which curve.

    Args:
        - orientation (list of str|None): list of axis names defining the
            orientation for 2D data:
                - orientation[0] each domain value of this axis corresponds
                                 to one curve.
                - orientation[1] corresponds to the x axis
            If None: orientation is the current axes of the cuboid
        - colors (dict <domain value>: <matplotlib color>):
            associate domain values of axis orientation[0] to colors to display
            curves
        - plot_kwargs (dict <arg name>:<arg value>):
            dictionary of named argument passed to the plot function
        - legend_prefix (str): prefix to prepend to legend labels.

    Return:
        None
    """

    ori = orientation or c.axes_names
    colors = colors or {}
    plot_kwargs = plot_kwargs or {}
    if c.get_ndims() == 1:
        #c.data[-5:] = 0
        print c.axes_domains[ori[0]]
        print c.axes_domains
        plt.plot(c.axes_domains[ori[0]], c.data, **plot_kwargs)
        #plt.plot(c.data, **plot_kwargs)
        #plt.plot(np.arange(0, len(c.data)/2, .5), c.data, **plot_kwargs)
    elif c.get_ndims() == 2:
        for val, sub_c in c.split(ori[0]).iteritems():
            pkwargs = plot_kwargs.copy()
            pkwargs['color'] = colors.get(val, None)
            pkwargs['label'] = legend_prefix + str(val)
            plot_cub_as_curve(sub_c, plot_kwargs=pkwargs)

        plt.legend()
    else:
        raise Exception('xndarray has too many dims (%d), expected at most 2' \
                        % c.get_ndims())


def plot_error(fig_dir, v_noise_range, error):
    fs = 23 #23 #fontsize
    ls = 10 #legend fontsize
    lw = 1  #linewtidth -> better bigger since image is often small
    #label0 = ['hyperprior', 'no hyperprior', 'hyperprior, positive', 'no hyperprior, positive']
    label0 = ['hyperprior', 'no hyperprior', 'hyperprior in PRF', 'hyperprior in BRF']
    lst = ['solid', 'dashed', ':', '-.']
    plt.figure(1)
    plt.hold('on')
    for i in np.arange(0,error.shape[1]):
        plt.plot(v_noise_range, error[:,i,0], color='b', linewidth = lw, linestyle = lst[i], label = 'BRF, '+label0[i])
        plt.plot(v_noise_range, error[:,i,1], color='r', linewidth = lw, linestyle = lst[i], label = 'PRF, '+label0[i])
    #plt.xlabel('SNR(dB)')
    plt.ylabel('RMSE')
    plt.legend(loc = 2 ,prop={'size':ls})
    #plt.axis([v_noise_range[0],v_noise_range[-1],0,1])
    plt.axis('tight')
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, 'error_prf_and_brf.png'))
    plt.close()
    plt.figure(2)
    plt.hold('on')
    for i in np.arange(0,error.shape[1]):
        plt.plot(v_noise_range, error[:,i,2], color='g', linewidth = lw, linestyle = lst[i], label = 'BRL, '+label0[i])
        plt.plot(v_noise_range, error[:,i,3], color='m', linewidth = lw, linestyle = lst[i], label = 'PRL, '+label0[i])
    plt.xlabel('noise variance')
    plt.ylabel('RMSE')
    plt.legend(loc = 2 ,prop={'size':ls})
    plt.axis('tight')
    #plt.axis([v_noise_range[0],v_noise_range[-1],0,1])
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, 'error_prl_and_brl.png'))
    plt.close()


def plot_jde_rfs(simu_dir, jde_dir, fig_dir, fig_prefix, asl_items=None):
    fs = 23            # fontsize
    lw = 2             # linewtidth -> better bigger since image is often small
    enorm = plt.normalize(0., 1.)

    from pyhrf.sandbox.physio_params import create_omega_prf, PHY_PARAMS_KHALIDOV11, \
                                            create_physio_brf, create_physio_prf
    phy_params=PHY_PARAMS_KHALIDOV11.copy()
    """
    PHY_PARAMS_KHALIDOV11 = { 'model_name': 'Khalidov11',
    'tau_s': 1.54, 'tau_m': 0.98, 'tau_f': 2.46,
    'eps': .54, 'alpha_w': .33, 'E0': .34, 'V0': 1, 'e': 1.43,
    'TE': 0.018,
    'model': 'RBM', 'linear': False, 'obata': False, 'buxton': False } """
    slightly_different = False
    very_different_brf = False
    very_different_prf = False
    if slightly_different:
        phy_params['tau_s'] = 1.74 #2.74
        phy_params['tau_m'] = 1.1 #1.8
        phy_params['tau_f'] = 2.66 #3.1
        phy_params['eps'] = 0.74#1.74
        phy_params['alpha_w'] = 0.53#1.3
        phy_params['E0'] = 0.54#0.94
        phy_params['V0'] = 1.2#2.
        phy_params['e'] = 1. #0.3
    if very_different_brf:
        phy_params['tau_s'] = 2.74
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 3.1
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.3
    if very_different_prf:
        phy_params['tau_s'] = 8.
        phy_params['tau_m'] = 1.8
        phy_params['tau_f'] = 2.
        phy_params['eps'] = 1.74
        phy_params['alpha_w'] = 1.3
        phy_params['E0'] = 0.94
        phy_params['V0'] = 2.
        phy_params['e'] = 0.1
    Thrf = 25.
    dt = 1.25
    brf = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
    brf /= np.linalg.norm(brf)
    prf = create_physio_prf(phy_params, response_dt=dt, response_duration=Thrf)
    prf /= np.linalg.norm(prf)
    time_array = np.arange(0, Thrf + dt, dt)

    #BRF
    plt.figure()
    ch = xndarray.load(op.join(simu_dir, 'brf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(sagittal=0, coronal=0,
                                    axial=0).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='simulated BRF ',
                      plot_kwargs={'linewidth': lw, 'linestyle': '--',
                                   'color': 'k'})
    plt.hold('on')
    ch = xndarray.load(op.join(jde_dir+'', 'jde_vem_asl_brf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix=' estimated BRF ',
                      plot_kwargs={'linewidth': lw, 'color': 'b'})
    plt.plot(time_array, brf, 'g', label='prior mean')
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'brf_est.png'))
    plt.close()

    #PRF
    plt.figure()
    ch = xndarray.load(op.join(simu_dir, 'prf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(sagittal=0, coronal=0,
                                    axial=0).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='PRF ',
                      plot_kwargs={'linewidth': lw, 'linestyle': '--',
                                   'color': 'k'})
    plt.hold('on')
    ch = xndarray.load(op.join(jde_dir, 'jde_vem_asl_prf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='PRF ',
                      plot_kwargs={'linewidth': lw, 'color': 'r'})
    plt.plot(time_array, prf, 'g', label='prior mean')
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'prf_est.png'))
    plt.close()
    plt.close('all')


def plot_jde_outputs(jde_dir, fig_dir, fig_prefix, norm, conds,
                     asl_items=None):
    fs = 23            # fontsize
    lw = 2             # linewtidth -> better bigger since image is often small
    enorm = plt.normalize(0., 1.)

    #FE
    plt.figure()
    name = 'jde_vem_asl_convergence_FE.nii'
    ch = xndarray.load(op.join(jde_dir, name))
    plt.plot(ch.sub_cuboid(ROI=1).data, label='free energy', linewidth=lw, marker='o')
    plt.legend(loc=4)
    plt.grid()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'free_energy.png'))

    #FE
    plt.figure()
    name = 'jde_vem_asl_convergence_FE.nii'
    ch = xndarray.load(op.join(jde_dir, name))
    plt.plot(ch.sub_cuboid(ROI=1).data[1:], label='free energy', linewidth=lw, marker='o')
    plt.legend(loc=4)
    plt.grid()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'free_energy_no0.png'))

    #BRF
    plt.figure()
    if fig_prefix=='mcmc_':
        name = 'jde_mcmc_brf_pm.nii'
    else:
        name = 'jde_vem_asl_brf.nii'
    ch = xndarray.load(op.join(jde_dir, name))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                colors={'estim': 'b', 'true': 'r'}, legend_prefix='BRF ',
                plot_kwargs={'linewidth': lw})
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'brf_est.png'))
    hrf1 = ch.sub_cuboid(ROI=1).roll('time').data
    hrf2 = asl_items['primary_brf']
    if 0:
        error_hrf_abs = np.sqrt(np.sum((hrf1 - hrf2) ** 2))
        error_hrf_rel = np.sqrt(np.sum((hrf1 - hrf2) ** 2) / np.sum(hrf2 ** 2))
        print 'BRF:'
        print ' - Mean Absolute Error = ', np.mean(error_hrf_abs)
        print ' - Mean Relative Error = ', np.mean(error_hrf_rel)

    #BRLS
    if fig_prefix=='mcmc_':
        name = 'jde_mcmc_brl_pm.nii'
    else:
        name = 'jde_vem_asl_brls.nii'
    b_nrls = xndarray.load(op.join(jde_dir, name))
    for icond, cond in enumerate(conds):
        cmap = plt.cm.jet
        plt.matshow(b_nrls.data[0, :, :, icond], cmap=cmap, norm=norm)  # there are 2
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_pm_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + 'brls_pm_' + cond + '_palette_est.png'))
        nrls1 = b_nrls.data[0, :, :, icond].flatten()
        nrls2 = asl_items['brls'][icond]
        error_nrls_abs = np.abs(nrls1 - nrls2)
        error_nrls_rel = np.abs((nrls1 - nrls2) / (nrls2))
        print error_nrls_rel.shape
        print error_nrls_abs.shape
        #BRLS absolute error
        print 'BRLS:'
        print ' - Mean Absolute Error = ', np.mean(error_nrls_abs)
        cmap = plt.cm.jet
        plt.matshow(error_nrls_abs.reshape(20, 20), cmap=cmap, norm=norm)
        plt.title('error = '+str(np.mean(error_nrls_abs)))
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_abs_err_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, \
                              fig_prefix + 'brls_abs_err_' + cond + '_palette_est.png'))
        #BRLS relative error
        print ' - Mean Relative Error = ', np.mean(error_nrls_rel)
        cmap = plt.cm.jet
        plt.matshow(error_nrls_rel.reshape(20, 20), cmap=cmap, norm=enorm)
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_rel_err_' + cond + '.png'))
        plot_palette(cmap, norm=enorm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir,
                              fig_prefix + 'brls_rel_err_' + cond + '_palette_est.png'))

    #PRF
    if fig_prefix=='mcmc_':
        name = 'jde_mcmc_prf_pm.nii'
    else:
        name = 'jde_vem_asl_prf.nii'
    plt.figure()
    ch = xndarray.load(op.join(jde_dir, name))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                    colors={'estim': 'b', 'true': 'r'},
                    legend_prefix='PRF ',
                    plot_kwargs={'linewidth': lw})
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'prf_est.png'))
    hrf1 = ch.sub_cuboid(ROI=1).roll('time').data
    hrf2 = asl_items['primary_prf']
    if 0:
        error_prf_abs = np.sqrt(np.sum((hrf1 - hrf2) ** 2))
        error_prf_rel = np.sqrt(np.sum((hrf1 - hrf2) ** 2) / np.sum(hrf2 ** 2))
        print 'PRF:'
        print ' - Mean Absolute Error = ', np.mean(error_prf_abs)
        print ' - Mean Relative Error = ', np.mean(error_prf_rel)

    #PRLS
    if fig_prefix=='mcmc_':
        name = 'jde_mcmc_prl_pm.nii'
    else:
        name = 'jde_vem_asl_prls.nii'
    b_nrls = xndarray.load(op.join(jde_dir, name))
    for icond, cond in enumerate(conds):
        cmap = plt.cm.jet
        plt.matshow(b_nrls.data[0, :, :, icond], cmap=cmap, norm=norm)  # there are 2
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'prl_pm_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + 'prls_pm_' + cond + '_palette_est.png'))
        nrls1 = b_nrls.data[0, :, :, icond].flatten()
        nrls2 = asl_items['prls'][icond]
        error_prls_abs = np.abs(nrls1 - nrls2)
        error_prls_rel = np.abs((nrls1 - nrls2) / (nrls2))
        #PRLS absolute error
        print 'PRLS:'
        print ' - Mean Absolute Error = ', np.mean(error_prls_abs)
        cmap = plt.cm.jet
        plt.matshow(error_prls_abs.reshape(20, 20), cmap=cmap, norm=norm)
        plt.title('error = '+str(np.mean(error_prls_abs)))
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'prl_abs_err_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, \
                              fig_prefix + 'prls_abs_err_' + cond + '_palette_est.png'))
        #PRLS relative error
        print ' - Mean Relative Error = ', np.mean(error_prls_rel)
        cmap = plt.cm.jet
        plt.matshow(error_prls_rel.reshape(20, 20), cmap=cmap, norm=enorm)
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'prl_rel_err_' + cond + '.png'))
        plot_palette(cmap, norm=enorm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir,
                              fig_prefix + 'prls_rel_err_' + cond + '_palette_est.png'))

    #Labels
    if fig_prefix=='mcmc_':
        name = 'jde_mcmc_label_pm.nii'
        labels = xndarray.load(op.join(jde_dir, name))
        labels = labels.sub_cuboid()
    else:
        name = 'jde_vem_asl_labels.nii'
        labels = xndarray.load(op.join(jde_dir, name))
        labels = labels.sub_cuboid(Act_class='activ')
    cmap = plt.cm.jet
    for icond, cond in enumerate(conds):
        plt.matshow(labels.data[0, :, :, icond], cmap=cmap)  # there are 2
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'labels_pm_' + cond + '.png'))
        plot_palette(cmap, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + 'labels_pm_' + cond + '_palette_est.png'))
    plt.close('all')

    return #np.mean(error_hrf_rel), np.mean(error_prf_rel), \
           #np.mean(error_nrls_rel), np.mean(error_prls_rel)


def plot_jde_inputs(jde_dir, fig_dir, fig_prefix, conds):

    fs = 23         # fontsize
    lw = 2          # linewtidth -> better bigger since image is often small

    #BRLs
    for cond in conds:
        b_nrls = xndarray.load(op.join(jde_dir, 'brls_' + cond + '.nii'))
        b_nrls = b_nrls.sub_cuboid(sagittal=0)
        plt.matshow(b_nrls.data)
        plt.gca().set_axis_off()
        plt.title('brls')
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_brl.png'))
        norm0 = plt.normalize(b_nrls.data.min(), b_nrls.data.max())
        plot_palette(plt.cm.jet, norm=norm0, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_brls_palette.png'))

    #BRF
    plt.figure()
    ch = xndarray.load(op.join(jde_dir, 'brf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(sagittal=0, coronal=0,
                                    axial=0).roll('time'),
                      colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='BRF ',
                      plot_kwargs={'linewidth': lw})
    plt.legend()
    #plt.xlabel('sec.')
    #plt.ylabel('BRF')
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'brf_sim.png'))

    #PRLs
    for cond in conds:
        p_nrls = xndarray.load(op.join(jde_dir, 'prls_' + cond + '.nii'))
        p_nrls = p_nrls.sub_cuboid(sagittal=0)
        plt.matshow(p_nrls.data, norm=norm0)
        plt.gca().set_axis_off()
        plt.title('prls')
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_prl.png'))
        norm1 = plt.normalize(p_nrls.data.min(), p_nrls.data.max())
        plot_palette(plt.cm.jet, norm=norm0, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_prls_palette.png'))

    #PRF
    plt.figure()
    ch = xndarray.load(op.join(jde_dir, 'prf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(sagittal=0, coronal=0,
                                    axial=0).roll('time'),
                      colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='PRF ',
                      plot_kwargs={'linewidth': lw})
    plt.legend()
    #plt.xlabel('sec.')
    #plt.ylabel('BRF')
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'prf_sim.png'))

    #noise variance
    noise_var = xndarray.load(op.join(jde_dir, 'noise_emp_var.nii'))
    noise_var = noise_var.sub_cuboid(sagittal=0)
    if noise_var.has_axis('type'):
        noise_var = noise_var.sub_cuboid(type='estim')
    plt.matshow(noise_var.data)
    plt.gca().set_axis_off()
    plt.title('noise variance')
    save_and_crop(op.join(fig_dir, fig_prefix + 'noise_var_pm.png'))
    plot_palette(plt.cm.jet, norm=plt.normalize(noise_var.data.min(),
                                                noise_var.data.max()),
                                                fontsize=fs * 2)
    save_and_crop(op.join(fig_dir, fig_prefix + 'noise_var_palette.png'))
    plt.close('all')

    #labels
    for cond in conds:
        labels = xndarray.load(op.join(jde_dir, 'labels_' + cond + '.nii'))
        labels = labels.sub_cuboid(sagittal=0)
        plt.matshow(labels.data)
        plt.gca().set_axis_off()
        plt.title('audio')
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_labels.png'))
        norm4 = plt.normalize(labels.data.min(), labels.data.max())
        plot_palette(plt.cm.jet, norm=norm4, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + cond + '_labels_palette.png'))

    plt.close('all')

    return norm0



##########################################
#### Main: Simulation specifications #####
##########################################

def get_nb_voxels_from_labels(labels_vol):
    return labels_vol.size

#############
#### run ####
#############

if __name__ == '__main__':
    #np.seterr('raise') #HACK  (shouldn't be commented)
    np.seterr(all='ignore')
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        main()
