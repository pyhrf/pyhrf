import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import pyhrf
from pyhrf.core import FmriData
from pyhrf import get_data_file_name
from pyhrf.ndarray import xndarray, MRI4Daxes

from matplotlib.colors import normalize, LinearSegmentedColormap
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='sans serif')
from pyhrf.plot import plot_func_slice, autocrop, plot_palette
from pyhrf.plot import set_ticks_fontsize, plot_cub_as_curve

from pyhrf.ui.glm_ui import GLMAnalyser
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.rfir import RFIREstim
from pyhrf.ui.rfir_ui import RFIRAnalyser
from pyhrf.ui.treatment import FMRITreatment

from pyhrf.parcellation import make_parcellation_from_files

def main():

    pyhrf.verbose.set_verbosity(1)

    output_dir = './'
    bold_file = get_data_file_name('real_data_vol_4_regions_BOLD.nii.gz')
    mask_file = get_data_file_name('real_data_vol_4_regions_mask.nii.gz')
    paradigm_file = get_data_file_name('paradigm_loc.csv')

    contrasts = pyhrf.paradigm.default_contrasts_loc

    condition_of_interest = 'calculaudio'
    contrast_of_interest = 'computation-sentences'
    point_of_interest = {'axial':33, 'coronal':25, 'sagittal':53}
    parcel_of_interest = 11
    plot_label =  'parietal'

    parcellation_dir = get_dir(output_dir, 'parcellation')

    # Load data
    fdata = FmriData.from_vol_files(mask_file=mask_file,
                                    bold_files=[bold_file],
                                    paradigm_csv_file=paradigm_file)

    if 0:
        # GLM with canonical HRF
        print 'GLM with canonical HRF'
        glm_hcano_output_dir = get_dir(output_dir, 'glm_cano')
        glm_analyse(fdata, contrasts, output_dir=glm_hcano_output_dir,
                    output_prefix='glm_hcano_')

    if 1:
        # GLM with basis set
        print 'GLM with basis set'
        glm_hderiv_output_dir = get_dir(output_dir, 'glm_hderivs')
        glm_analyse(fdata, contrasts, hrf_model="Canonical with Derivative",
                    output_dir=glm_hderiv_output_dir,
                    output_prefix='glm_hderivs_')

    if 0:
        # GLM FIR
        print 'GLM FIR'
        glm_fir_output_dir = get_dir(output_dir, 'glm_fir')
        glm_analyse(fdata, contrasts={}, hrf_model="FIR",
                    output_dir=glm_fir_output_dir, output_prefix='glm_fir_',
                    fir_delays=range(11))

    # Regularized FIR
    print 'RFIR'
    rfir_mask = op.join(parcellation_dir, 'mask_single_voxel_for_rfir.nii')
    make_single_voxel_mask(point_of_interest, mask_file, rfir_mask)
    fdata_rfir = FmriData.from_vol_files(mask_file=rfir_mask,
                                         bold_files=[bold_file],
                                         paradigm_csv_file=paradigm_file)
    rfir_output_dir = get_dir(output_dir, 'rfir')
    rfir_analyse(fdata_rfir, output_dir=rfir_output_dir)

    # parcellation from results of GLM basis set

    parcellation_file =  op.join(parcellation_dir,  'parcellation_func.nii')
    mask_file = op.join(glm_hderiv_output_dir, 'mask.nii')
    beta_files = [op.join(glm_hderiv_output_dir, '*%s*.nii'%c) \
                          for c in fdata.get_condition_names()]
    make_parcellation_from_files(beta_files, mask_file, parcellation_file,
                                 nparcels=20, method='ward_and_gkm')

    # JDE
    print 'JDE'
    jde_output_dir = get_dir(output_dir, 'jde')
    fdata_parc = FmriData.from_vol_files(mask_file=parcellation_file,
                                         bold_files=[bold_file],
                                         paradigm_csv_file=paradigm_file)
    jde_analyse(fdata_parc, jde_output_dir)

    # GLM hcano rescaled onto JDE (provide the same results as normal GLM hcano
    # but with effects resized so that there is a consistency btw
    # X^m.h in JDE and the corresponding column of the design matrix in GLM
    print 'GLM rescaled'
    rescale_factor_file = op.join(jde_output_dir, 'scale_factor_for_glm.nii')
    glm_hcano_rs_output_dir = get_dir(output_dir, 'glm_hcano_rescaled_on_jde')
    glm_analyse(fdata, contrasts, output_dir=glm_hcano_rs_output_dir,
                output_prefix='glm_hcano_rs_',
                rescale_factor_file=rescale_factor_file)

    ## Outputs

    fig_dir = get_dir(output_dir, 'figs')
    plot_detection_results(fig_dir, point_of_interest, condition_of_interest,
                           contrast_of_interest, jde_output_dir,
                           glm_hcano_rs_output_dir)

    plot_estimation_results(fig_dir, point_of_interest, parcel_of_interest,
                            condition_of_interest, plot_label,
                            glm_fir_output_dir, rfir_output_dir, jde_output_dir)

def glm_analyse(fdata, contrasts, output_dir, output_prefix,
                hrf_model="Canonical", fir_delays=None):
    glm_analyser = GLMAnalyser(hrf_model=hrf_model, contrasts=contrasts,
                               outputPrefix=output_prefix,
                               fir_delays=fir_delays)
    glm_analyser.set_pass_errors(False)
    tt = FMRITreatment(fdata, glm_analyser, output_dir=output_dir)
    tt.run()

def jde_analyse(fdata, contrasts, output_dir):
    from pyhrf.jde.models import BOLDGibbsSampler as BG
    from pyhrf.jde.hrf import RHSampler as HVS
    from pyhrf.jde.nrl.bigaussian import NRLSampler as NS

    sampler = BG({
        BG.P_NB_ITERATIONS : 2, # HACK
        # HRF variance
        BG.P_RH : HVS({
            HVS.P_SAMPLE_FLAG : False,
            HVS.P_VAL_INI : np.array([0.05]),
            }),
        BG.P_NRLS : NS({
            NS.P_CONTRASTS : contrasts,
            })
        })

    analyser = JDEMCMCAnalyser(sampler=sampler)
    tt = FMRITreatment(fdata, analyser, output_dir=output_dir)
    tt.run(parallel='local')

def rfir_analyse(fdata, output_dir):
    analyser = RFIRAnalyser(RFIREstim(nb_its_max=3)) #HACK
    tt = FMRITreatment(fdata, analyser, output_dir=output_dir)
    tt.run(parallel='local')

def make_single_voxel_mask(poi, mask_file, new_mask_file):

    m = xndarray.load(mask_file)
    i,j,k = poi['sagittal'], poi['coronal'], poi['axial']
    new_m = xndarray.xndarray_like(m)
    new_m.data[i,j,k] = 1
    new_m.save(new_mask_file)

def plot_detection_results(fig_dir, poi, condition, coi, parcellation_file,
                           jde_output_dir, glm_hcano_rs_output_dir,
                           fig_dpi=100):
    """
    coi (str): contrast of interest
    poi (dict): defines the point of interest for plots of HRFs and maps
    """
    orientation = ['coronal',  'sagittal']
    axial_slice =  poi['axial']

    anat_file = get_data_file_name('real_data_vol_4_regions_anatomy.nii.gz')

    parcellation = xndarray.load(parcellation_file)
    parcellation = parcellation.sub_cuboid(axial=axial_slice)
    parcellation = parcellation.reorient(orientation)

    ## Detection maps
    detection_plots_params = []

    #JDE NRLs
    fn = op.join(jde_output_dir, 'jde_mcmc_nrl_pm.nii')

    slice_def = {'axial':axial_slice, 'condition':condition}
    fig_fn = op.join(fig_dir, 'real_data_jde_mcmc_nrls_%s.png' %condition)
    detection_plots_params.append({'fn':fn, 'slice_def':slice_def,
                                   'mask': parcellation.data,
                                   'output_fig_fn':fig_fn})

    #GLM hcano
    fn = op.join(glm_hcano_rs_output_dir, 'glm_hcano_rs_beta_%s.nii' %condition)

    slice_def = {'axial':axial_slice}

    fig_fn = op.join(fig_dir, 'real_data_glm_hcano_rs_%s.png'%condition)

    detection_plots_params.append({'fn':fn, 'slice_def':slice_def,
                                   'mask': (parcellation.data != 0),
                                   'output_fig_fn':fig_fn})


    perf_norm = plot_maps(detection_plots_params, anat_file,
                          {'axial':axial_slice*3},
                          fig_dir, orientation=orientation,
                          crop_extension=None, plot_anat=True)

    palette_fig_fn = op.join(fig_dir, 'real_data_detection_%s_palette.png' \
                             %condition)
    plot_palette(cmap, perf_norm, 45)
    plt.savefig(palette_fig_fn, dpi=fig_dpi)
    autocrop(palette_fig_fn)


    #JDE Contrast
    fn = op.join(jde_output_dir, 'jde_mcmc_nrl_contrasts.nii')

    slice_def = {'axial':axial_slice, 'contrast':coi}
    fig_fn = op.join(fig_dir, 'real_data_jde_mcmc_con_%s.png' %coi)
    detection_plots_params.append({'fn':fn, 'slice_def':slice_def,
                                   'mask': parcellation.data,
                                   'output_fig_fn':fig_fn})

    #GLM hcano
    fn = op.join(glm_hcano_rs_output_dir, 'glm_hcano_rs_con_effect_%s.nii' %coi)

    slice_def = {'axial':axial_slice}

    fig_fn = op.join(fig_dir, 'real_data_glm_hcano_rs_con_%s.png' %coi)

    detection_plots_params.append({'fn':fn, 'slice_def':slice_def,
                                   'mask': (parcellation.data != 0),
                                   'output_fig_fn':fig_fn})


    perf_norm = plot_maps(detection_plots_params, anat_file,
                          {'axial':axial_slice*3},
                          fig_dir, orientation=orientation,
                          crop_extension=None, plot_anat=True)

    palette_fig_fn = op.join(fig_dir, 'real_data_detection_con_%s_palette.png'
                             %coi)
    plot_palette(cmap, perf_norm, 45)
    plt.savefig(palette_fig_fn, dpi=fig_dpi)
    autocrop(palette_fig_fn)


def plot_estimation_results(fig_dir, poi, jde_roi, cond, plot_label,
                            glm_fir_output_dir, rfir_output_dir,
                            jde_output_dir, plot_fontsize=25,
                            ymin=-1.55, ymax=1.05):

    ## HRF plots

    fn = op.join(glm_fir_output_dir, 'glm_fir_hrf.nii')
    fir = xndarray.load(fn).sub_cuboid(condition=cond, **poi)
    #fir /= (fir**2).sum()**.5
    fir /= fir.max()

    fn = op.join(rfir_output_dir, 'rfir_ehrf.nii')
    rfir = xndarray.load(fn).sub_cuboid(condition=cond, **poi)
    #rfir /= (rfir**2).sum()**.5
    rfir /= rfir.max()

    fn = op.join(jde_output_dir, 'jde_mcmc_hrf_pm.nii')
    jde = xndarray.load(fn).sub_cuboid(ROI=jde_roi)
    jde /= jde.max()

    plt.figure()
    pargs = {'linewidth' : 2.7}
    plot_cub_as_curve(fir, show_axis_labels=False, plot_kwargs=pargs)
    plot_cub_as_curve(rfir, show_axis_labels=False, plot_kwargs=pargs)
    plot_cub_as_curve(jde, show_axis_labels=False, plot_kwargs=pargs)

    from pyhrf.boldsynth.hrf import getCanoHRF
    time_points, hcano = getCanoHRF()
    hcano /= hcano.max()
    plt.plot(time_points, hcano, 'k.-',linewidth=1.5)

    set_ticks_fontsize(plot_fontsize)
    plt.xlim(0,25)
    plt.ylim(ymin, ymax)

    plt.gca().xaxis.grid(True, 'major', linestyle='--', linewidth=1.2,
                         color='gray')

    hrf_fig_fn = op.join(fig_dir, 'real_data_hrfs_%s.png' %plot_label)
    print 'hrf_fig_fn:', hrf_fig_fn
    plt.savefig(hrf_fig_fn)
    autocrop(hrf_fig_fn)


def get_dir(*dirs):
    d = op.join(*dirs)
    if not op.exists(d):
        os.makedirs(d)
    return d




def cmstring_to_mpl_cmap(s):
    lrgb = s.split('#')
    r = [float(v) for v in lrgb[0].split(';')]
    g = [float(v) for v in lrgb[1].split(';')]
    b = [float(v) for v in lrgb[2].split(';')]

    cdict = {'red':(),'green':(),'blue':()}
    for iv in xrange(0,len(r),2):
        cdict['red'] += ((r[iv],r[iv+1],r[iv+1]),)
    for iv in xrange(0,len(b),2):
        cdict['blue'] += ((b[iv],b[iv+1],b[iv+1]),)
    for iv in xrange(0,len(g),2):
        cdict['green'] += ((g[iv],g[iv+1],g[iv+1]),)

    return LinearSegmentedColormap('mpl_colmap',cdict,256)

cmap_string = '0;0;0.5;0.0;0.75;1.0;1.;1.0#' \
              '0;0;0.5;1;0.75;1;1;0.#'       \
              '0;0;0.25;1;0.5;0;1;0.'
cmap = cmstring_to_mpl_cmap(cmap_string)


def plot_maps(plot_params, anat_fn, anat_slice_def, fig_dir,
              orientation=['axial','sagittal'], crop_extension=None,
              plot_anat=True, plot_fontsize=25, fig_dpi=75):

    ldata = []
    for p in plot_params:
        c = xndarray.load(p['fn']).sub_cuboid(**p['slice_def'])
        c.set_orientation(orientation)
        ldata.append(c.data)

    c_anat = xndarray.load(anat_fn).sub_cuboid(**anat_slice_def)
    c_anat.set_orientation(orientation)

    resolution = c_anat.meta_data[1]['pixdim'][1:4]
    slice_resolution = resolution[MRI4Daxes.index(orientation[0])], \
      resolution[MRI4Daxes.index(orientation[1])]

    all_data = np.array(ldata)

    if 'prl' in plot_params[0]['fn']:
        norm = normalize(all_data.min(), all_data.max()*1.05)
        print 'norm:', (all_data.min(), all_data.max())
    else:
        norm = normalize(all_data.min(), all_data.max())

    print 'norm:', (all_data.min(), all_data.max())
    for data, plot_param in zip(all_data, plot_params):
        fn = plot_param['fn']
        plt.figure()
        print 'fn:', fn
        print '->', (data.min(), data.max())
        if plot_anat:
            anat_data = c_anat.data
        else:
            anat_data = None
        plot_func_slice(data, anatomy=anat_data,
                        parcellation=plot_param.get('mask'),
                        func_cmap=cmap,
                        parcels_line_width=1., func_norm=norm,
                        resolution=slice_resolution,
                        crop_extension=crop_extension)
        set_ticks_fontsize(plot_fontsize)

        fig_fn = op.join(fig_dir, '%s.png' %op.splitext(op.basename(fn))[0])
        output_fig_fn = plot_param.get('output_fig_fn', fig_fn)

        print 'Save to: %s' %output_fig_fn
        plt.savefig(output_fig_fn, dpi=fig_dpi)
        autocrop(output_fig_fn)
    return norm


if __name__ == '__main__':
    main()


