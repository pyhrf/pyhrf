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
#version = 'vem'
fig_dir = './Results_HEROES_plot' #+ version
if not op.exists(fig_dir): os.makedirs(fig_dir)

def main():

    if not op.exists(fig_dir):
        os.makedirs(fig_dir)

    methods = ['vem'] #['mcmc']#, 'vem']    
    subjects = ['RG130377']
    prior_types = ['omega', 'balloon', 'no']

    # If I put it here I have the same range between conditions
    brl_norm = None
    prl_norm = None
            

    for subject in subjects:
        print 'subject ', subject

        for prior in prior_types:
            print 'prior ', prior

            for method in methods:
                print 'method ', method

                data_dir = op.join('./archives', subject)
                anat_dir = op.join(data_dir, 'anat')
                data_scenario = 'prior' + prior                 

                brls_plots_params = []
                prls_plots_params = []
                prev_prf_scale = None

                n_roi = 0

                if method=='vem':
                    jde_folder = op.join('./', subject, 'jde_results_' + prior + '_hyplower')
                    analysis_dir = op.join(data_dir, jde_folder)
                    fname_brf = 'jde_vem_asl_brf.nii'
                    fname_prf = 'jde_vem_asl_prf.nii'
                    fname_brl = 'jde_vem_asl_brls.nii'
                    fname_prl = 'jde_vem_asl_prls.nii'
                    fname_base = 'jde_vem_asl_alpha.nii'
                    fname_brlvar = 'jde_vem_asl_Sigma_brls.nii'
                    fname_prlvar = 'jde_vem_asl_Sigma_prls.nii'
                else:
                    jde_folder = op.join('jde_results', method + '_jde_analysis_' + cond)
                    analysis_dir = op.join(data_dir, jde_folder)
                    fname_brf = 'jde_mcmc_brf_pm.nii'
                    fname_prf = 'jde_mcmc_prf_pm.nii'
                    fname_brl = 'jde_mcmc_brl_pm.nii'
                    fname_prl = 'jde_mcmc_prl_pm.nii'
                    fname_base = 'jde_mcmc_perf_baseline_pm.nii'
                    fname_brlvar = 'jde_mcmc_brl_mcmc_var.nii'
                    fname_prlvar = 'jde_mcmc_prl_mcmc_var.nii'


                #for roi_hrf in xrange(0,210):#np.array([200, 198, 8, 5, 9, 6, 18, 7, 2, \
                    #    #          156, 38, 152, 32, 31, 166, 159, 49, \
                    #    #          65, 165, 72, 79, 122, 186, 190, 183]):
                if 0:
                    #roi_hrf = 1
                    print 'ROI ', roi_hrf
                    try:
                        ### Plot of RFs ###
                        brf = xndarray.load(op.join(analysis_dir, fname_brf))
                        brf = brf.sub_cuboid(ROI=roi_hrf)
                        n_roi += 1
                        #print 'sign_corr_brf:', sign_corr_brf
                        prf = xndarray.load(op.join(analysis_dir, fname_prf))
                        prf = prf.sub_cuboid(ROI=roi_hrf)
                        
                        taxis_prf = prf.axes_domains['time']
                        taxis_cano_hrf, cano_hrf = getCanoHRF(taxis_prf.max(),
                                                        taxis_prf[1] - taxis_prf[0])
                        lw = 3.
                        plt.figure()
                        plt.plot(prf.axes_domains['time'], prf.data, 'r', linewidth=lw)
                        plt.plot(brf.axes_domains['time'], brf.data, 'b', linewidth=lw)
                        plt.plot(taxis_cano_hrf, cano_hrf, 'k--', linewidth=lw)
                        plt.xlim(0, 25.5)
                        set_ticks_fontsize(fs)
                        #plt.legend(prop={'size':32})
                        fig_fn = op.join(fig_dir, '%s_real_data_brf_prf_%s_%s_roi%d.png' \
                                                  % (method, prior, subject, roi_hrf))
                        #print 'Save BRFs to:', fig_fn
                        plt.savefig(fig_fn)
                        autocrop(fig_fn)
                        plt.close()
                    except:
                        fake_variable = 0
                        #print 'ROI ' + str(roi_hrf) + ' does not exist in map ' \
                        #        + prior + ' on subject ' + subject 
                    
                
                #anat_fn = op.join(data_dir, 'wmAINSI_010_TV_t1mri.nii')
                #anat_fn = op.join(anat_dir, 'w' + subject + '_anat-0001.nii')
                #anat_fn = op.join(anat_dir, 'm' + subject + '_anat-0001.nii')
                anat_fn = op.join(anat_dir, subject + '_anat-0001_coreg.nii')
                #AINSI_010_TV_anat-0001_coreg
                #print 'Number of ROI found: ', n_roi
                if 1: # or n_roi > 0:
                    parcel_dir = op.join(data_dir, 'ASLf', 'parcellation')
                    #prepare plots of RL maps:
                    """if cond == 'video':
                        #print cond
                        ax_slice = 22
                        crop_def = "140x181+170+0"
                        #roi_mask = xndarray.load(op.join('./archives', 'roi_mask_visual_2.nii'))
                        roi_mask_fn = op.join(parcel_dir, 'parcellation_video_func_masked.nii')  
                        #roi_mask = xndarray.load(roi_mask_fn)
                    else:
                        ax_slice = 21
                        crop_def = "140x181+0+174"
                        #roi_mask = xndarray.load(op.join('./archives', 'roi_mask_audio.nii'))
                        roi_mask_fn = op.join(parcel_dir, 'parcellation_audio_func_masked.nii')"""
                    roi_mask_fn = op.join(parcel_dir, 'parcellation_func.nii')
                    fn = op.join(analysis_dir, fname_brl)
                    brl_var_fn = op.join(analysis_dir, fname_brlvar)
                    slice_def = {'axial': ax_slice}
                    fig_fn = op.join(fig_dir,
                                     '%s_jde_vem_brl_pm_%s_%s.png' \
                                     % (method, prior, subject))
                    brls_plots_params.append({'fn': fn, 'slice_def': slice_def,
                                              'mask': roi_mask_fn,
                                              'output_fig_fn': fig_fn,
                                              'var': brl_var_fn})
                    fn = op.join(analysis_dir, fname_prl)
                    perf_fn = op.join(analysis_dir, fname_base)
                    prl_var_fn = op.join(analysis_dir, fname_prlvar)
                    slice_def = {'axial': ax_slice}
                    fig_fn = op.join(fig_dir,
                                     '%s_jde_vem_prl_pm_%s_%s.png' \
                                     % (method, prior, subject))
                    prl_scale = 1.
                    prls_plots_params.append({'fn': fn, 'slice_def': slice_def,
                                              'mask': roi_mask_fn,
                                              'output_fig_fn': fig_fn,
                                              'scale_factor': prl_scale,
                                              'perf': perf_fn,
                                              'var': prl_var_fn})   
                                                  
                    #print 'plotting response level maps...'
                    # Plot all maps with the same color range superimposed with anatomy
                    # and cropped on temporal region:
                    if brl_norm == None:
                        brl_norm, prl_norm = plot_maps(brls_plots_params, prls_plots_params, anat_fn,
                                             roi_mask_fn, {"axial": ax_slice * 3},
                                             crop_def=crop_def, 
                                             subject=subject)
                    else:
                        plot_maps(brls_plots_params, prls_plots_params, anat_fn, roi_mask_fn,
                                  {"axial": ax_slice * 3}, crop_def=crop_def,
                                  norm=brl_norm, norm2=prl_norm, subject=subject)
                    #print 'brl_norm = ', brl_norm
                    cmap = 'seismic'
                    plot_palette(cmap, brl_norm, 45)
                    palette_fig_fn = op.join(fig_dir,
                                             'real_data_brls_palette_%s_%s.png' \
                                             % (prior, subject))
                    plt.savefig(palette_fig_fn)
                    autocrop(palette_fig_fn)
                    
                    plot_palette(cmap, prl_norm, 45)
                    palette_fig_fn = op.join(fig_dir,
                                             'real_data_prls_palette_%s_%s.png' \
                                             % (prior, subject))
                    plt.savefig(palette_fig_fn)
                    autocrop(palette_fig_fn)
                    
                else:
                    print 'NOTHING TO BE PLOTTED! '


def plot_maps(plot_params, plot_params2, anat_fn, roi_mask_fn, anat_slice_def, flip_sign=False,
              crop_def=None, norm=None, norm2=None, cond='_', subject=None):
    print 'entro en plot_maps'
    ldata = []
    ldata2 = []
    for ip, p in enumerate(plot_params):
        #print 'load:', p['fn']
        prl = 'prl' in p['fn']
        c = xndarray.load(p['fn']).sub_cuboid(condition=cond)
        c.set_orientation(['coronal', 'axial', 'sagittal'])
        c.data *= p.get('scale_factor', 1.)
        #print c.data.shape
        if flip_sign:
            ldata = c.data * -1.
        else:
            ldata = c.data
        p2 = plot_params2[ip]
        c2 = xndarray.load(p2['fn']).sub_cuboid(condition=cond)
        c2.set_orientation(['coronal', 'axial', 'sagittal'])
        c2.data *= p2.get('scale_factor', 1.)
        #print c.data.shape
        if flip_sign:
            ldata2 = c2.data * -1.
        else:
            ldata2 = c2.data
        

    all_data = np.array(ldata)
    #all_data_left = np.array(ldata)
    all_data_mean = np.array(ldata)
    all_data_right = np.array(ldata)
    all_data2 = np.array(ldata2)
    """
    roi_max = np.argmax(np.max(all_data, axis=1))
    print roi_max
    print 'sum = ', np.sum((np.max(all_data, axis=0)==np.mean(all_data, axis=0))*1.)
    print all_data.shape
    print all_data[np.where(all_data>2.)]
    print 'maximum of all data:'
    print np.max(all_data)
    print np.max(all_data, axis=0)
    print np.max(all_data, axis=1)
    print np.max(all_data, axis=0).shape
    print np.max(all_data, axis=1).shape
    print np.argmax(all_data)
    print np.argmax(all_data)
    indices = np.where( all_data == np.max(all_data) )
    print indices
    x_y_coords =  zip(indices[0], indices[1])
    print x_y_coords
    sum_data = np.sum(all_data, 0)
    print 'maximum of sumed data:'
    print np.max(sum_data)
    print np.argmax(sum_data)
    """
    #print all_data.shap[:,:,sum_data.shape[2]/2:]e
    #all_data_left[:,:,all_data_left.shape[2]/2:] = 0
    #ind_left = np.unravel_index(all_data_left.argmax(), all_data.shape)
    all_data_right[:,:,:all_data_right.shape[2]/2] = 0
    ind_right = np.unravel_index(all_data_right.argmax(), all_data.shape)
    #print 'Index of the maximum left: ', ind_left
    print 'Index of the maximum right: ', ind_right
    roi_mask = xndarray.load(roi_mask_fn)
    roi_mask.set_orientation(['coronal', 'axial', 'sagittal'])
    #print roi_mask.data.sum()
    mask = roi_mask.data
    #print 'ROI of the maximum left: ', mask[ind_left]
    print 'ROI of the maximum right: ', mask[ind_right]
    
    #sum_data[:,:,sum_data.shape[2]/2:] = 0
    #inds = np.unravel_index(np.argmax(sum_data), sum_data.shape)
    for mind in xrange(0, 220):
        all_data_mean[np.where(mask==mind)] = np.mean(all_data_mean[np.where(mask==mind)])
    ind_mean = np.unravel_index(all_data_mean.argmax(), all_data.shape)
    print 'Index of the mean max: ', ind_mean
    print 'ROI of the mean max: ', mask[ind_mean]

    """max_roi = 0
    for mind in xrange(0, 220):
        roi_m = all_data[np.where(mask==mind)]
        roi_max = roi_m.max()
        if roi_max>max_rois:
            roi = mind
            max_rois = roi_max
        #print roi_m
        
        voxs = np.unravel_index(roi_m.argmax(), all_data.shape)
    """        
    #m = np.where(mask > 0)
    #all_data_masked = all_data[:, m[0], m[1]]
    #print mask.shape
    if cond == 'video':
        #voxel2 = ind_left #[12, 22, 42]
        voxel = ind_right #[12, 25, 20]
        roi_voxel = mask[voxel[0], voxel[1], voxel[2]]
        m = np.where(mask == roi_voxel)
        print 'voxel = ', voxel
        print 'voxel ROI = ', roi_voxel    
        #roi_voxel = mask[voxel2[0], voxel2[1], voxel2[2]]
        #print 'voxel2 = ', voxel2
        #print 'voxel2 ROI = ', roi_voxel    
        #m2 = np.where(mask == roi_voxel)
    else:
        #voxel2 = ind_left #[38, 25, 51]
        voxel = ind_right #[40, 24, 11]
        roi_voxel = mask[voxel[0], voxel[1], voxel[2]]
        m = np.where(mask == roi_voxel)
        print 'voxel = ', voxel
        print 'voxel ROI = ', roi_voxel    
        #roi_voxel = mask[voxel2[0], voxel2[1], voxel2[2]]
        #print 'voxel2 = ', voxel2
        #print 'voxel2 ROI = ', roi_voxel    
        #m2 = np.where(mask == roi_voxel)
    #print m
    all_data_masked1 = all_data[:, voxel[1], :]
    all_data_masked2 = all_data2[:, voxel[1], :]
    #all_data_masked3 = all_data[:, voxel[1], :]
    #all_data_masked4 = all_data2[:, voxel[1], :]
    mask_data = np.zeros_like(all_data)
    

    mask_data[m] = 1.
    #mask_data[m2] = 1.

    if norm == None:
        
        print all_data_masked1.min()
        print all_data_masked1.max()
        print all_data[m].max()
        norm = normalize(-all_data_masked1.max(), all_data_masked1.max())
        #norm2 = norm
        norm2 = normalize(-all_data_masked2.max(), all_data_masked2.max())
        #norm3 = normalize(all_data_masked3.min(), all_data_masked3.max())
        #norm4 = normalize(all_data_masked4.min(), all_data_masked4.max())

    #c_anat = xndarray.load(anat_fn).sub_cuboid(**anat_slice_def)
    #c_anat.set_orientation(['coronal', 'sagittal'])
    data = np.reshape(all_data, c.data.shape)
    data2 = np.reshape(all_data2, c.data.shape)
    c_anat = xndarray.load(anat_fn)
    c_anat.set_orientation(['coronal', 'axial', 'sagittal'])
    try:
        perf = xndarray.load(p2['perf'])
        perf.set_orientation(['coronal', 'axial', 'sagittal'])
        norm_perf = normalize(-(perf.data).max(), perf.data.max())
    except:
        print 'no perfusion'
    try:
        brl_var = xndarray.load(p['var'])
        print 'brl_var loaded'
        #print brl_var
        #brl_var.set_orientation(['coronal', 'axial', 'sagittal'])
        #print 'brl_var orientation'
        prl_var = xndarray.load(p2['var'])
        print 'prl_var loaded'
        #prl_var.set_orientation(['coronal', 'axial', 'sagittal'])
        #print 'prl_var orientation'
    except:
        print 'no variance'

    if 1:
        print 'data shape = ', data.shape
        for isl in xrange(0, data.shape[1]):
            print 'slice ', isl
            
            plt.figure()
            plot_func_slice(data[:, isl, :], anatomy=c_anat.data[:, isl*3, :],
                            parcellation=mask[:, isl, :], #func_cmap=cmap,
                            parcels_line_width=1., func_norm=norm)
            set_ticks_fontsize(fs)
            fn = plot_params[0]['fn']
            output_fig_fn = op.join(fig_dir, '%s_sum_%s_%s_slice%d.png' \
                                    % (op.splitext(op.basename(fn))[0],
                                       subject, cond, isl))
            #print 'Save to: %s' % output_fig_fn
            plt.savefig(output_fig_fn)
            autocrop(output_fig_fn)
            plt.close()

            plt.figure()
            plot_func_slice(data2[:, isl, :], anatomy=c_anat.data[:, isl*3, :],
                            parcellation=mask[:, isl, :], #func_cmap=cmap,
                            parcels_line_width=1., func_norm=norm2)
            set_ticks_fontsize(fs)
            fn = plot_params2[0]['fn']
            output_fig_fn = op.join(fig_dir, '%s_sum_%s_%s_slice%d.png' \
                                    % (op.splitext(op.basename(fn))[0],
                                       subject, cond, isl))
            #print 'Save to: %s' % output_fig_fn
            plt.savefig(output_fig_fn)
            autocrop(output_fig_fn)
            plt.close()
            
            
            plt.figure()
            plot_func_slice(mask_data[:, isl, :], anatomy=c_anat.data[:, isl*3, :],
                            parcellation=mask[:, isl, :], #func_cmap=cmap,
                            parcels_line_width=1.)#, func_norm=norm2)
            set_ticks_fontsize(fs)
            fn = plot_params2[0]['fn']
            output_fig_fn = op.join(fig_dir, '%s_mask_%s_%s_slice%d.png' \
                                    % (op.splitext(op.basename(fn))[0],
                                       subject, cond, isl))
            #print 'Save to: %s' % output_fig_fn
            plt.savefig(output_fig_fn)
            autocrop(output_fig_fn)
            plt.close()

            try:
                plt.figure()
                plot_func_slice(perf.data[:, isl, :]+data2[:, isl, :],
                                anatomy=c_anat.data[:, isl*3, :],
                                parcellation=mask[:, isl, :], #func_cmap=cmap,
                                parcels_line_width=1., func_norm=norm_perf)
                set_ticks_fontsize(fs)
                fn = plot_params2[0]['fn']
                output_fig_fn = op.join(fig_dir, '%s_perfsum_%s_%s_slice%d.png' \
                                        % (op.splitext(op.basename(fn))[0],
                                           subject, cond, isl))
                #print 'Save to: %s' % output_fig_fn
                plt.savefig(output_fig_fn)
                autocrop(output_fig_fn)
                plt.close()
            except:
                print 'no perfusion'

            try:
                plt.figure()
                plot_func_slice(perf.data[:, isl, :], anatomy=c_anat.data[:, isl*3, :],
                                parcellation=mask[:, isl, :], #func_cmap=cmap,
                                parcels_line_width=1., func_norm=norm_perf)
                set_ticks_fontsize(fs)
                fn = plot_params2[0]['fn']
                output_fig_fn = op.join(fig_dir, '%s_perf_%s_%s_slice%d.png' \
                                        % (op.splitext(op.basename(fn))[0],
                                           subject, cond, isl))
                #print 'Save to: %s' % output_fig_fn
                plt.savefig(output_fig_fn)
                autocrop(output_fig_fn)
                plt.close()
            except:
                print 'no perfusion'

            try:
                print brl_var.data.shape
                plt.figure()
                plot_func_slice(brl_var.data[:, isl, :, 0], anatomy=c_anat.data[:, isl*3, :],
                                parcellation=mask[:, isl, :], #func_cmap=cmap,
                                parcels_line_width=1., func_norm=norm)
                set_ticks_fontsize(fs)
                fn = plot_params[0]['fn']
                output_fig_fn = op.join(fig_dir, '%s_var_%s_%s_slice%d.png' \
                                        % (op.splitext(op.basename(fn))[0],
                                           subject, cond, isl))
                #print 'Save to: %s' % output_fig_fn
                plt.savefig(output_fig_fn)
                autocrop(output_fig_fn)
                plt.close()
            except:
                print 'no HRF var'

            try:
                print prl_var.data.shape
                plt.figure()
                plot_func_slice(prl_var.data[:, isl, :, 0], anatomy=c_anat.data[:, isl*3, :],
                                parcellation=mask[:, isl, :], #func_cmap=cmap,
                                parcels_line_width=1., func_norm=norm2)
                set_ticks_fontsize(fs)
                fn = plot_params[0]['fn']
                output_fig_fn = op.join(fig_dir, '%s_var_%s_%s_slice%d.png' \
                                        % (op.splitext(op.basename(fn))[0],
                                           subject, cond, isl))
                #print 'Save to: %s' % output_fig_fn
                plt.savefig(output_fig_fn)
                autocrop(output_fig_fn)
                plt.close()
            except:
                print 'no PRF var'
            
        try:
            #plot_palette('seismic', norm_perf, 45)
            plot_palette('jet', norm_level, 45)
            palette_fig_fn = op.join(fig_dir,
                                     'real_data_perf_palette_%s_%s.png' \
                                     % (cond, subject))
            plt.savefig(palette_fig_fn)
            autocrop(palette_fig_fn)
            plt.close()
        except:
            print 'no perfusion'

            
        plt.close('all')
        del c_anat, data, mask, all_data_masked1, all_data_masked2
    
    #print 'norm:', (all_data_masked.min(), all_data_masked.max())
    return norm, norm2
    
    
def plot_masks(plot_params, plot_params2, anat_fn, roi_mask_fn, anat_slice_def,
               flip_sign=False, crop_def=None, norm=None, cond='video',
               subject=None):
    roi_mask = xndarray.load(roi_mask_fn)
    roi_mask.set_orientation(['coronal', 'axial', 'sagittal'])
    mask = roi_mask.data
    
    if cond == 'video':
        rois_voxel = [100]
        
    else:
        rois_voxel = [100]
        for roi_voxel in rois_voxel:
            m = np.where(mask == roi_voxel)

    mask_data = np.zeros_like(mask)
    m = np.where(mask == roi_voxel)
    mask_data[m] = 1.

    if norm == None:
        
        print all_data_masked1.min()
        print all_data_masked1.max()
        print all_data[m].max()
        norm = normalize(all_data_masked1.min(), all_data_masked1.max())
        norm2 = normalize(all_data_masked2.min(), all_data_masked2.max())

    data = np.reshape(all_data, c.data.shape)
    data2 = np.reshape(all_data2, c.data.shape)
    c_anat = xndarray.load(anat_fn)
    c_anat.set_orientation(['coronal', 'axial', 'sagittal'])
    
    if 1:
        for isl in xrange(0, data.shape[1]):
            plt.figure()
            plot_func_slice(mask_data[:, isl, :], anatomy=c_anat.data[:, isl, :],
                            parcellation=mask[:, isl, :], func_cmap=cmap,
                            parcels_line_width=1.)#, func_norm=norm2)
            set_ticks_fontsize(fs)
            fn = plot_params2[0]['fn']
            output_fig_fn = op.join(fig_dir, '%s_mask_%s_%s_slice%d.png' \
                                    % (op.splitext(op.basename(fn))[0],
                                       subject, cond, isl))
            #print 'Save to: %s' % output_fig_fn
            plt.savefig(output_fig_fn)
            autocrop(output_fig_fn)
            plt.close()
        
        del c_anat, data, mask, all_data_masked1, all_data_masked2
    
    return norm, norm2



def plot_maps2(plot_params, anat_fn, anat_slice_def, flip_sign=False,
              crop_def=None, norm=None, cond='video'):
    ldata = []
    for p in plot_params:
        print 'load:', p['fn']
        prl = 'prl' in p['fn']
        c = xndarray.load(p['fn']).sub_cuboid(**p['slice_def'])
        c.set_orientation(['coronal', 'sagittal'])
        c.data *= p.get('scale_factor', 1.)
        if flip_sign:
            ldata.append(c.data * -1.)
        else:
            ldata.append(c.data)

    c_anat = xndarray.load(anat_fn).sub_cuboid(**anat_slice_def)
    c_anat.set_orientation(['coronal', 'sagittal'])
    
    all_data = np.array(ldata)
    mask = plot_params[0].get('mask')
    m = np.where(mask > 0)
    all_data_masked = all_data[:, m[0], m[1]]
    if norm == None:
        norm = normalize(all_data_masked.min(), all_data_masked.max())

    print 'norm:', (all_data_masked.min(), all_data_masked.max())
    
    for data, plot_param in zip(all_data, plot_params):
        #total_map = 
        fn = plot_param['fn']

        plt.figure()
        print 'fn:', fn
        print '->', (data.min(), data.max())
        plot_func_slice(data, anatomy=c_anat.data,
                        parcellation=mask,
                        func_cmap=cmap,
                        parcels_line_width=1., func_norm=norm)
        set_ticks_fontsize(fs)

        fig_fn = op.join(fig_dir, '%s.png' % op.splitext(op.basename(fn))[0])
        output_fig_fn = plot_param.get('output_fig_fn', fig_fn)

        print 'Save to: %s' % output_fig_fn
        plt.savefig(output_fig_fn)
        autocrop(output_fig_fn)

        if crop_def is not None:
            # convert to jpg (avoid page size issues):
            output_fig_fn_jpg = op.splitext(output_fig_fn)[0] + '.jpg'
            os.system('convert %s %s' % (output_fig_fn, output_fig_fn_jpg))
            # crop and convert back to png:
            output_fig_fn_cropped = add_suffix(output_fig_fn, '_cropped')
            print 'output_fig_fn_cropped:', output_fig_fn_cropped
            os.system('convert %s -crop %s +repage %s' \
                      % (output_fig_fn_jpg, crop_def, output_fig_fn_cropped))
    
    return norm


if __name__ == '__main__':
    main()
