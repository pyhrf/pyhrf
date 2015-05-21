
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
#import shutil

import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser
#from pyhrf.ui.vb_jde_analyser_asl import JDEVEMAnalyser
#import pyhrf.jde.asl as jdem
from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
from pyhrf.ndarray import xndarray
import pyhrf.jde.models as jdem
import pyhrf.sandbox.physio as phym
#from pyhrf import Condition
#import pyhrf.boldsynth.scenarios as simu
#from pyhrf.tools.misc import Pipeline

import matplotlib.pyplot as plt

# Let's use TeX to render text
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='sans serif', size=23)

pyhrf.logger.setLevel("INFO")


############################
##### JDE BOLD Set Up  #####
############################

def main():

    np.random.seed(48258)

    # Tags
    simulate = True
    do_jde_asl = True
    analyse_vem = False
    analyse_mcmc = True
    plot_results_jde = False

    for dt in np.array([ 1.]):   # 2.5, 1.25, 0.5

        # Folder names
        fig_prefix = 'vem_asl_TR25_dt' + str(np.round(dt*10).astype(np.int32))\
                        + '_dur15_lowSNR_'
        simulation_dir = fig_prefix + 'simulated_bold'
        fig_dir = fig_prefix + 'figs'
    
        # create output folder
        np.random.seed(48258)
        if not op.exists(fig_dir):
            os.makedirs(fig_dir)
        if not op.exists(simulation_dir):
            os.makedirs(simulation_dir)
    
        v_noise_range = np.arange(2.0, 2.3, 1.)
        hyp = False
        pos = False
        n_method = 1
        error = np.zeros((len(v_noise_range), n_method, 4))
        m = 0
        
        for ivn, v_noise in enumerate(v_noise_range):
            print 'Generating BOLD data ...'
            print 'index v_noise = ', ivn
            print 'v_noise = ', v_noise
            if simulate:
                bold_items, conds = simulate_bold(output_dir=simulation_dir,
                                                  noise_scenario='low_snr',
                                                  spatial_size='normal', dt=dt)  
                print bold_items
                print conds
            norm = plot_jde_inputs(simulation_dir, fig_dir,
                                   'simu_vn' + str(v_noise) + '_', conds)
            print 'Finished generation of BOLD data.'

            if do_jde_asl:
                np.random.seed(48258)
                vem_output_dir = op.join(simulation_dir,
                                         'jde_vem_analysis_vn' + str(v_noise))
                if not op.exists(vem_output_dir):
                    os.makedirs(vem_output_dir)
                mcmc_output_dir = op.join(simulation_dir,
                                          'jde_mcmc_analysis_vn' + str(v_noise))
                if not op.exists(mcmc_output_dir):
                    os.makedirs(mcmc_output_dir)

                if analyse_vem:
                    print 'JDE analysis VEM on simulation ...'
                    jde_analyse_vem(simulation_dir, vem_output_dir,
                                    bold_items, dt=dt)
                if analyse_mcmc:
                    print 'JDE analysis MCMC on simulation ...'
                    jde_analyse_mcmc(simulation_dir, mcmc_output_dir,
                                     bold_items, dt=dt)

                if plot_results_jde:
                    print 'JDE analysis (old) on simulation done!'
                    error[ivn, m, :] = plot_jde_outputs(old_output_dir, fig_dir,
                                                        'vn' + str(v_noise) + '_',
                                                        norm, conds, bold_items)
                    plot_jde_rfs(simulation_dir, old_output_dir, fig_dir,
                                 'vn' + str(v_noise) + '_',
                                 bold_items)
        m += 1 
        #plot_error(fig_dir, v_noise_range, error)



def jde_analyse_vem(simulation_dir,output_dir,simulation,constrained=False,
                fast=True, dt=2.):
    # Create an FmriData object directly from the simulation dictionary:
    fmri_data = FmriData.from_simulation_dict(simulation, mask=None)
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
                           beta=.8, hrfVar=.001, use_true_nrls=False,
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



def jde_analyse_mcmc(simulation_dir, output_dir, simulation, dt=2., nb_its=1500):
    # Pack simulation into a pyhrf.core.FmriData object
    #fmri_data = FmriData.from_vol_files(mask_file, paradigm_file, [bold_file], tr)
    fmri_data = FmriData.from_simulation_dict(simulation, mask=None)

    # JDE
    jde_mcmc_sampler = build_jde_mcmc_sampler(nb_its, 
                                              use_true_nrls=False,
                                              use_true_labels=True)
    analyser = JDEMCMCAnalyser(jde_mcmc_sampler, dt=dt)
    tjde_mcmc = FMRITreatment(fmri_data, analyser, output_dir=output_dir)
    tjde_mcmc.execute()


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
    tr = 1.
    dsf = tr/dt
    
    #dt = 2.5
    #dsf = 1  # down sampling factor

    import pyhrf.paradigm as mpar
    #paradigm_csv_file = './../paradigm_data/paradigm_bilateral_vjoint.csv'
    #paradigm_csv_file = './../paradigm_data/paradigm_bilateral_v1_no_final_rest.csv'
    #paradigm_csv_file = './paradigm_data/paradigm_bilateral_v2_no_final_rest_1.csv'
    paradigm_csv_file = './../paradigm_data/paradigm_bilateral_v2_no_final_rest.csv'
    paradigm_csv_delim = ' '

    #onsets, durations = load_paradigm(paradigm_csv_file)
    #paradigm2 = mpar.Paradigm(onsets, durations)
    
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

        # f = open(op.join(output_dir, 'simulation.pck'), 'w')
        # cPickle.dump(simulation, f)
        # f.close()

    return simulation, condition_names[0:3]
    
    
def load_paradigm(fn):
    from collections import defaultdict

    fn_content = open(fn).readlines()
    onsets = defaultdict(list)
    durations = defaultdict(list)
    for line in fn_content:
        sline = line.split(' ')
        print 'sline:', sline
        if len(sline) < 4:
            cond, onset, _ = sline
        else:
            sess, cond, onset, duration, amplitude = sline
            #sess, cond, onset, duration = sline
            duration = duration[:-1]
            if 1:            
                print 'sess = ', sess
                print 'cond = ', cond
                print 'onset = ', onset
                print 'duration = ', duration
                #0 "clicGaudio" 355.9 0
        onsets[cond.strip('"')].append(float(onset))
        durations[cond.strip('"')].append(float(duration))

    ons_to_return = {}
    dur_to_return = {}
    for cn, ons in onsets.iteritems():
        sorting = np.argsort(ons)
        ons_to_return[cn] = np.array(ons)[sorting]
        dur_to_return[cn] = np.array(durations[cn])[sorting]

    return ons_to_return, dur_to_return


def print_descrip_onsets(onsets):
    onsets = dict((n, o) for n, o in onsets.iteritems() \
                       if n not in ['blanc', 'blank'])
    all_onsets = np.hstack(onsets.values())
    diffs = np.diff(np.sort(all_onsets))
    #pprint(onsets)
    print 'mean ISI:', format_duration(diffs.mean())
    print 'max ISI:', format_duration(diffs.max())
    print 'min ISI:', format_duration(diffs.min())
    print 'first event:', format_duration(all_onsets.min())
    print 'last event:', format_duration(all_onsets.max())


def format_duration(dt):
    s = ''
    if dt / 3600 >= 1:
        s += '%dH' % int(dt / 3600)
        dt = dt % 3600
    if dt / 60 >= 1:
        s += '%dmin' % int(dt / 60)
        dt = int(dt % 60)
    s += '%1.3fsec' % dt
    return s


##################
#### Plotting ####
##################

from pyhrf.plot import autocrop, set_ticks_fontsize, plot_palette

def save_and_crop(fn):
    plt.savefig(fn)
    autocrop(fn)


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


def plot_jde_outputs(jde_dir, fig_dir, fig_prefix, norm, conds,
                     asl_items=None):
    fs = 23            # fontsize
    lw = 2             # linewtidth -> better bigger since image is often small
    enorm = plt.normalize(0., 1.)
    print conds
    
    #BRF
    plt.figure()
    ch = xndarray.load(op.join(jde_dir, 'jde_vem_hrf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                colors={'estim': 'b', 'true': 'r'}, legend_prefix='BRF ',
                plot_kwargs={'linewidth': lw})
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'brf_est.png'))
    hrf1 = ch.sub_cuboid(ROI=1).roll('time').data
    #print asl_items
    hrf2 = asl_items['primary_hrf']
    if 0:
        error_hrf_abs = np.sqrt(np.sum((hrf1 - hrf2) ** 2))
        error_hrf_rel = np.sqrt(np.sum((hrf1 - hrf2) ** 2) / np.sum(hrf2 ** 2))
        print 'BRF:'
        print ' - Mean Absolute Error = ', np.mean(error_hrf_abs)
        print ' - Mean Relative Error = ', np.mean(error_hrf_rel)

    #BRLS
    b_nrls = xndarray.load(op.join(jde_dir, 'jde_vem_nrls.nii'))
    for icond, cond in enumerate(conds):
        print cond
        cmap = plt.cm.jet
        plt.matshow(b_nrls.data[0, :, :, icond], cmap=cmap, norm=norm)  # there are 2
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_pm_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, fig_prefix + 'brls_pm_' + cond + '_palette_est.png'))
        nrls1 = b_nrls.data[0, :, :, icond].flatten()
        nrls2 = asl_items['nrls'][icond]
        error_nrls_abs = np.abs(nrls1 - nrls2)
        error_nrls_rel = np.abs((nrls1 - nrls2) / (nrls2))
        #BRLS absolute error
        print 'BRLS:'
        print ' - Mean Absolute Error = ', np.mean(error_nrls_abs)
        cmap = plt.cm.jet
        plt.matshow(error_nrls_abs.reshape(20, 20), cmap=cmap, norm=norm)
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_abs_err_' + cond + '.png'))
        plot_palette(cmap, norm=norm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir, \
                              fig_prefix + 'brls_abs_err_' + cond + '_palette_est.png'))
        plt.close('all')
        #BRLS relative error
        print ' - Mean Relative Error = ', np.mean(error_nrls_rel)
        cmap = plt.cm.jet
        plt.matshow(error_nrls_rel.reshape(20, 20), cmap=cmap, norm=enorm)
        plt.gca().set_axis_off()
        save_and_crop(op.join(fig_dir, fig_prefix + 'brl_rel_err_' + cond + '.png'))
        plot_palette(cmap, norm=enorm, fontsize=fs * 2)
        save_and_crop(op.join(fig_dir,
                              fig_prefix + 'brls_rel_err_' + cond + '_palette_est.png'))
        plt.close('all')

    #Labels
    labels = xndarray.load(op.join(jde_dir, 'jde_vem_labels.nii'))
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
    

def plot_jde_rfs(simu_dir, jde_dir, fig_dir, fig_prefix, asl_items=None):
    fs = 23            # fontsize
    lw = 2             # linewtidth -> better bigger since image is often small
    enorm = plt.normalize(0., 1.)

    #BRF
    plt.figure()
    ch = xndarray.load(op.join(simu_dir, 'hrf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(sagittal=0, coronal=0,
                                    axial=0).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix='simulated BRF ',
                      plot_kwargs={'linewidth': lw, 'linestyle': '--',
                                   'color': 'k'})
    plt.hold('on')
    ch = xndarray.load(op.join(jde_dir, 'jde_vem_hrf.nii'))
    plot_cub_as_curve(ch.sub_cuboid(ROI=1).roll('time'),
                      #colors={'estim': 'b', 'true': 'r'},
                      legend_prefix=' estimated BRF ',
                      plot_kwargs={'linewidth': lw, 'color': 'b'})
    plt.legend()
    set_ticks_fontsize(fs)
    save_and_crop(op.join(fig_dir, fig_prefix + 'brf_est.png'))
    plt.close()


def plot_jde_inputs(jde_dir, fig_dir, fig_prefix, conds):

    fs = 23         # fontsize
    lw = 2          # linewtidth -> better bigger since image is often small

    #BRLs
    for cond in conds:
        b_nrls = xndarray.load(op.join(jde_dir, 'nrls_' + cond + '.nii'))
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
    ch = xndarray.load(op.join(jde_dir, 'hrf.nii'))
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
