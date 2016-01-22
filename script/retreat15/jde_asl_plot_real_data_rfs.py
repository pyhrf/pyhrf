"""
Plot paper-ready figures for real data results of the MICCAI14 paper.

IMPORTANT:
 - the script 'jde_asl_real_data.py' must have been run prior to
   running this script**
 - The script assumes that the data dir is the current directory
   so it must be invoked where the data and results are stored

For the 2-steps and basic JDE approaches, the script produces:
  - plot of PRF and BRF curves with the canonical HRF
  - plot PRL and BRL maps superimposed with anatomy, with their colorbars
    in separate figs. Figs are cropped to zoom on the temporal region of the
    considered ROI

Some rescaling of RFs and RLs is performed so that RFs are comparable across
methods (adjust peak height).

"""
import numpy as np
import os.path as op
from pyhrf.ndarray import xndarray
from pyhrf.plot import plot_func_slice, autocrop, plot_palette
from pyhrf.plot import set_ticks_fontsize
import matplotlib.pyplot as plt
from pyhrf.boldsynth.hrf import getCanoHRF

import matplotlib
from matplotlib.colors import normalize, LinearSegmentedColormap
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

import os
from pyhrf.tools import add_suffix

from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='sans serif')


def cmstring_to_mpl_cmap(s):
    lrgb = s.split('#')
    r = [float(v) for v in lrgb[0].split(';')]
    g = [float(v) for v in lrgb[1].split(';')]
    b = [float(v) for v in lrgb[2].split(';')]

    cdict = {'red': (), 'green': (), 'blue': ()}
    for iv in xrange(0, len(r), 2):
        cdict['red'] += ((r[iv], r[iv + 1], r[iv + 1]), )
    for iv in xrange(0, len(b), 2):
        cdict['blue'] += ((b[iv], b[iv + 1], b[iv + 1]), )
    for iv in xrange(0, len(g), 2):
        cdict['green'] += ((g[iv], g[iv + 1], g[iv + 1]), )

    return LinearSegmentedColormap('mpl_colmap', cdict, 256)

fs = 35  # fontsize
# color map used for plots of RLs:
cmap_string = '0.; 0.; 0.5 ; 0.0; 0.75; 1.0; 1.; 1.0#' \
              '0.; 0.; 0.5 ; 1. ; 0.75; 1. ; 1.; 0. #' \
              '0.; 0.; 0.25; 1. ; 0.5 ; 0. ; 1.; 0. '

"""cmap_string = '0.; 0.; 0.5 ; 0.0; 0.75; 1.0; 1.; 1.0#' \
              '0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. #' \
              '0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. '"""
cmap = cmstring_to_mpl_cmap(cmap_string)
#cmap = 'hot'

# Path where to store figures
#method = 'mcmc'
#fig_dir = './Results_cluster_rfs'
fig_dir = './Results_HEROES_plot'

def main():

    if not op.exists(fig_dir):
        os.makedirs(fig_dir)

    subjects = ['CD110147', 'RG130377']
    prior_types = ['omega', 'balloon', 'no']

    for cond in prior_types:
        print 'condition ', cond

        for subject in subjects:
            print 'subject ', subject

            #data_dir = op.join('./archives', subject)
            #data_dir = op.join('./', subject)
            prev_prf_scale = None
            n_roi = 0

            output_dir = op.join('./', subject, 'jde_results')

            roi_hrf = 11
            for roi_hrf in xrange(1,210):
                #if 1:
                print roi_hrf
                
                try:

                    #"""
                    method = 'vem'
                    analysis_dir = op.join(output_dir, 'vem_' + cond)
                    fname_brf = 'jde_vem_asl_brf.nii'
                    fname_prf = 'jde_vem_asl_prf.nii'                
                    print analysis_dir 
                    print 'existe? ', op.isfile(op.join(analysis_dir, fname_brf)) 

                    ### Plot of RFs ###
                    brf_v = xndarray.load(op.join(analysis_dir, fname_brf))
                    brf_v = brf_v.sub_cuboid(ROI=roi_hrf)
                    prf_v = xndarray.load(op.join(analysis_dir, fname_prf))
                    prf_v = prf_v.sub_cuboid(ROI=roi_hrf)
                    taxis_prf = prf_v.axes_domains['time']
                    #"""

                    method = 'mcmc'
                    analysis_dir = op.join(output_dir, 'mcmc')            
                    fname_brf = 'jde_mcmc_brf_pm.nii'
                    fname_prf = 'jde_mcmc_prf_pm.nii'
                    print analysis_dir 
                    print 'existe? ', op.isfile(op.join(analysis_dir, fname_brf)) 

                    ### Plot of RFs ###
                    brf_m = xndarray.load(op.join(analysis_dir, fname_brf))
                    brf_m = brf_m.sub_cuboid(ROI=roi_hrf)
                    prf_m = xndarray.load(op.join(analysis_dir, fname_prf))
                    prf_m = prf_m.sub_cuboid(ROI=roi_hrf)
                    taxis_prf = prf_m.axes_domains['time']
                                        
                    
                    dt = .5
                    dur = 25.
                    taxis_physio = np.arange(0., dur + dt, dt)
                    print taxis_physio

                    from pyhrf.sandbox.physio_params import PHY_PARAMS_KHALIDOV11, PHY_PARAMS_FRISTON00, \
                                                            create_physio_brf, create_physio_prf
                    physiological_params = PHY_PARAMS_KHALIDOV11
                    physio_brf = create_physio_brf(physiological_params, response_dt=dt,
                                                   response_duration=dur)
                    physio_prf = create_physio_prf(physiological_params, response_dt=dt,
                                                   response_duration=dur)
                    print physio_prf.shape
                    print taxis_physio.shape
                    
    
                    #taxis_cano_hrf, cano_hrf = getCanoHRF(taxis_prf.max(),
                    #                                taxis_prf[1] - taxis_prf[0])
                    #print 'perfusion norm = ', np.linalg.norm(prf_v.data)
                    
                    lw = 3.
                    plt.figure()
                    plt.plot(prf_m.axes_domains['time'], prf_m.data, 'r', linewidth=lw, label='PRF, JDE')
                    plt.plot(prf_v.axes_domains['time'], prf_v.data, 'm', linewidth=lw, label='PRF, JDE')
                    plt.plot(brf_m.axes_domains['time'], brf_m.data, 'b', linewidth=lw, label='HRF, JDE')
                    plt.plot(brf_v.axes_domains['time'], brf_v.data, 'g', linewidth=lw, label='HRF, JDE')
                    plt.plot(taxis_physio, physio_prf, 'k--', linewidth=lw, label='PRF, Balloon')
                    plt.plot(taxis_physio, physio_brf, 'k', linewidth=lw, label='HRF, Balloon')
                    plt.xlim(0, 25.5)
                    set_ticks_fontsize(fs)
                    plt.legend(prop={'size':20})
                    fig_fn = op.join(fig_dir, '%s_real_data_brf_prf_%s_%s_roi%d.png' \
                                              % (method, cond, subject, roi_hrf))
                    #print 'Save BRFs to:', fig_fn
                    plt.savefig(fig_fn)
                    autocrop(fig_fn)
                    plt.close()

                    fname_brf = 'jde_vem_asl_Sigma_brf.nii'
                    fname_prf = 'jde_vem_asl_Sigma_prf.nii'                
                    brf_v = xndarray.load(op.join(analysis_dir, fname_brf))
                    brf_v = brf_v.sub_cuboid(ROI=roi_hrf)
                    prf_v = xndarray.load(op.join(analysis_dir, fname_prf))
                    prf_v = prf_v.sub_cuboid(ROI=roi_hrf)
                    
                    plt.figure()
                    plt.matshow(brf_v.data)
                    plt.colorbar()
                    fig_fn = op.join(fig_dir, '%s_real_data_sigma_brf_%s_%s_roi%d.png' \
                                              % (method, cond, subject, roi_hrf))
                    plt.savefig(fig_fn)
                    autocrop(fig_fn)
                    plt.close()

                    plt.figure()
                    plt.matshow(prf_v.data)
                    plt.colorbar()
                    fig_fn = op.join(fig_dir, '%s_real_data_sigma_prf_%s_%s_roi%d.png' \
                                              % (method, cond, subject, roi_hrf))
                    plt.savefig(fig_fn)
                    autocrop(fig_fn)
                    plt.close()


                except:
                    fake_variable = 0
                    print 'ROI ' + str(roi_hrf) + ' does not exist in map ' \
                            + cond + ' on subject ' + subject 


if __name__ == '__main__':
    main()
