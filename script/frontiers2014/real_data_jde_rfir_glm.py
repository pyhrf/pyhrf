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

    experiments = [ ('audio', 'audio-video',
                     {'axial':17, 'coronal':42, 'sagittal':9}, 1,
                     'temporal', -1., 1.05),
                     ('calculaudio', 'computation-sentences',
                     {'axial':30, 'coronal':25, 'sagittal':53}, 11,
                     'parietal', -1.55, 1.05),
                   ]


    parcellation_dir = get_dir(output_dir, 'parcellation')

    # Load data
    fdata = FmriData.from_vol_files(mask_file=mask_file,
                                    bold_files=[bold_file],
                                    paradigm_csv_file=paradigm_file)

    glm_hcano_output_dir = get_dir(output_dir, 'glm_cano')
    if 0:
        # GLM with canonical HRF
        print 'GLM with canonical HRF'
        glm_analyse(fdata, contrasts, output_dir=glm_hcano_output_dir,
                    output_prefix='glm_hcano_')

    if 0:
        # GLM with basis set
        print 'GLM with basis set'
        glm_hderiv_output_dir = get_dir(output_dir, 'glm_hderivs')
        glm_analyse(fdata, contrasts, hrf_model="Canonical with Derivative",
                    output_dir=glm_hderiv_output_dir,
                    output_prefix='glm_hderivs_')

    parcellation_file =  op.join(parcellation_dir,  'parcellation_func.nii')
    if 0:
        # parcellation from results of GLM basis set

        mask_file = op.join(glm_hderiv_output_dir, 'glm_hderivs_mask.nii')
        beta_files = [op.join(glm_hderiv_output_dir,'glm_hderivs_beta_%s.nii'%c)\
                              for c in fdata.get_condition_names()]
        make_parcellation_from_files(beta_files, mask_file, parcellation_file,
                                     nparcels=20, method='ward_and_gkm')
    jde_output_dir = get_dir(output_dir, 'jde')
    if 0:
        # JDE
        print 'JDE'
        fdata_parc = FmriData.from_vol_files(mask_file=parcellation_file,
                                             bold_files=[bold_file],
                                             paradigm_csv_file=paradigm_file)
        jde_analyse(fdata_parc, contrasts, jde_output_dir)

    glm_hcano_rs_output_dir = get_dir(output_dir,'glm_hcano_rescaled_on_jde')
    if 0:
        # GLM hcano rescaled onto JDE (provide the same results as normal
        # GLM hcano but with effects resized so that there is a consistency btw
        # X^m.h in JDE and the corresponding column of the design matrix in GLM
        print 'GLM rescaled'
        rescale_factor_file = op.join(jde_output_dir, 'scale_factor_for_glm.nii')
        compute_jde_glm_rescaling(jde_output_dir, glm_hcano_output_dir,
                                  rescale_factor_file)

        glm_analyse(fdata, contrasts, output_dir=glm_hcano_rs_output_dir,
                    output_prefix='glm_hcano_rs_',
                    rescale_factor_file=rescale_factor_file)

    ## Outputs
    for condition_of_interest, contrast_of_interest, point_of_interest, \
        parcel_of_interest, plot_label, ymin, ymax  in experiments:

        if 'temporal' in plot_label:
            paradigm_tag = 'loc_av'
        else:
            paradigm_tag = 'loc'

        paradigm_file = get_data_file_name('paradigm_%s.csv' %paradigm_tag)
        fir_mask = op.join(parcellation_dir, 'mask_single_voxel_for_fir_%s.nii'\
                           %paradigm_tag)
        make_mask_from_points([point_of_interest], mask_file, fir_mask)
        fdata_fir = FmriData.from_vol_files(mask_file=fir_mask,
                                            bold_files=[bold_file],
                                            paradigm_csv_file=paradigm_file)

        glm_fir_output_dir = get_dir(output_dir, 'glm_fir_%s' %paradigm_tag)
        if 1:
            # GLM FIR
            print 'GLM FIR'
            glm_analyse(fdata_fir, contrasts={}, hrf_model="FIR",
                        output_dir=glm_fir_output_dir, output_prefix='glm_fir_',
                        fir_delays=range(11))



        rfir_output_dir = get_dir(output_dir, 'rfir_%s' %paradigm_tag)
        if 1:
            # Regularized FIR
            print 'RFIR'
            rfir_analyse(fdata_fir, output_dir=rfir_output_dir)


        fig_dir = get_dir(output_dir, 'figs')
        plot_detection_results(fig_dir, point_of_interest, condition_of_interest,
                               contrast_of_interest, parcellation_file,
                               plot_label, jde_output_dir,
                               glm_hcano_rs_output_dir)

        plot_estimation_results(fig_dir, point_of_interest, parcel_of_interest,
                                condition_of_interest, plot_label,
                                glm_fir_output_dir, rfir_output_dir,
                                jde_output_dir, ymin, ymax)

def glm_analyse(fdata, contrasts, output_dir, output_prefix,
                hrf_model="Canonical", fir_delays=None,
                rescale_factor_file=None):
    glm_analyser = GLMAnalyser(hrf_model=hrf_model, contrasts=contrasts,
                               outputPrefix=output_prefix, fir_delays=fir_delays,
                               rescale_factor_file=rescale_factor_file)
    glm_analyser.set_pass_errors(False)
    tt = FMRITreatment(fdata, glm_analyser, output_dir=output_dir)
    tt.run()

def jde_analyse(fdata, contrasts, output_dir):
    from pyhrf.jde.models import BOLDGibbsSampler as BG
    from pyhrf.jde.hrf import RHSampler
    from pyhrf.jde.nrl.bigaussian import NRLSampler

    sampler = BG(nb_iterations=2, # HACK
                 hrf_var=RHSampler(do_sampling=False, val_ini=np.array([0.05])),
                 response_levels=NRLSampler(contrasts=contrasts))

    analyser = JDEMCMCAnalyser(sampler=sampler)
    tt = FMRITreatment(fdata, analyser, output_dir=output_dir)
    tt.run(parallel='local')

def rfir_analyse(fdata, output_dir):
    analyser = RFIRAnalyser(RFIREstim(nb_its_max=3)) #HACK
    tt = FMRITreatment(fdata, analyser, output_dir=output_dir)
    tt.run(parallel='local')


def compute_jde_glm_rescaling(jde_path, glm_path, output_file):

    #TODO: check consistency of condition order btwn GLM & JDE !!!

    # load matX from JDE results (same for all parcels)
    # matX is a matrix of shape (time x nb_hrf_coeff)
    # matX = sum_m(X^m * m) where X is the matrix defined in the fwd JDE model
    # jde_dm_fn = op.join(jde_path,'jde_mcmc_matX.nii')
    # jde_dm = xndarray.load(jde_dm_fn).sub_cuboid(ROI=12).reorient(('time','P'))

    # print 'jde_dm:'
    # print jde_dm.descrip()

    # ny,lgCI = jde_dm.data.shape
    # nbConditions = len(np.unique(jde_dm.data)) - 1

    # # reconstruct all X^m from matX
    # varX = np.zeros((nbConditions,ny,lgCI))
    # for j in xrange(nbConditions):
    #     varX[j,:,:] = (jde_dm.data == j).astype(int)


    jde_varX_fn = op.join(jde_path,'jde_mcmc_varX.nii')
    jde_varX = xndarray.load(jde_varX_fn).sub_cuboid(ROI=1)
    jde_varX = jde_varX.reorient(('condition', 'time','P'))

    print 'jde_dm varX:'
    print jde_varX.descrip()

    nbConditions,ny,lgCI = jde_varX.data.shape
    varX = jde_varX.data

    # More convenient matrix structure to perfom product with HRF afterwards
    stackX = np.zeros((ny*nbConditions,lgCI), dtype=int)

    for j in xrange(nbConditions):
        stackX[ny*j:ny*(j+1), :] = varX[j,:,:]

    print 'stackX:', stackX.shape

    # Load HRFs from JDE results
    jde_hrf_fn =  op.join(jde_path, 'jde_mcmc_hrf_pm.nii')
    jde_hrf = xndarray.load(jde_hrf_fn).reorient(('ROI','time'))

    print 'jde_hrf:'
    print jde_hrf.descrip()

    roi_ids = jde_hrf.axes_domains['ROI']
    jde_xh = np.zeros((len(roi_ids),ny,nbConditions))

    for iroi,roi in enumerate(roi_ids):
        h = jde_hrf.sub_cuboid(ROI=roi).data[1:-1]
        # make sure that the HRF is normalized:
        h /= (h**2).sum()**.5
        stackXh = np.dot(stackX, h)
        jde_xh[iroi,:,:] = np.reshape(stackXh, (nbConditions,ny)).transpose()

    jde_roi_mask_fn =  op.join(jde_path, 'jde_mcmc_roi_mapping.nii')
    print 'jde_roi_mask_fn:', jde_roi_mask_fn
    jde_roi_mask = xndarray.load(jde_roi_mask_fn)

    glm_dm_fn =  op.join(glm_path, 'glm_hcano_design_matrix.nii')
    glm_dm = xndarray.load(glm_dm_fn).sub_cuboid(ROI=1).reorient(('time',
                                                                  'regressor'))

    print 'glm_dm:'
    print glm_dm.descrip()

    # align condition axis of GLM design matrix onto condition axis of JDE
    # design matrix:
    jde_xh_tmp = jde_xh.copy()
    cond_domain = []
    for cidx,cn in enumerate(glm_dm.axes_domains['regressor']):
        if cn in jde_varX.axes_domains['condition']:
            jde_cidx = np.where(jde_varX.axes_domains['condition']==cn)[0][0]
            print 'cidx:', cidx
            print 'jde_cidx:', jde_cidx
            jde_xh_tmp[:,:,cidx] = jde_xh[:,:,jde_cidx]
            cond_domain.append(cn)

    #glm_norm_weights = (glm_dm.data**2).sum(0)**.5
    glm_norm_weights = glm_dm.data.ptp(0)

    jde_norm_weights = np.zeros(jde_roi_mask.data.shape + (nbConditions,))
    scale_factor = np.zeros_like(jde_norm_weights)
    for iroi,roi in enumerate(roi_ids):
        m = np.where(jde_roi_mask.data == roi)
        #jde_w = (jde_xh[iroi]**2).sum(0)**.5
        jde_w = jde_xh[iroi].ptp(0)
        jde_norm_weights[m[0],m[1],m[2],:] = jde_w
        scale_factor[m[0],m[1],m[2],:] =  glm_norm_weights[:len(jde_w)] / jde_w

    from pyhrf.ndarray import MRI3Daxes
    csf = xndarray(scale_factor, axes_names=MRI3Daxes+['condition'],
                 axes_domains={'condition':np.array(cond_domain)},
                 meta_data=jde_hrf.meta_data)

    print 'rescale factor:'
    print csf.descrip()

    print 'scale_factor file:', output_file
    csf.reorient(['condition']+MRI3Daxes).save(output_file)


def make_mask_from_points(pois, mask_file, new_mask_file):

    m = xndarray.load(mask_file)
    new_m = xndarray.xndarray_like(m)
    for poi in pois:
        i,j,k = poi['sagittal'], poi['coronal'], poi['axial']
        new_m.data[i,j,k] = 1

    new_m.save(new_mask_file)

def plot_detection_results(fig_dir, poi, condition, coi, parcellation_file,
                           plot_label, jde_output_dir, glm_hcano_rs_output_dir,
                           fig_dpi=100):
    """
    coi (str): contrast of interest
    poi (dict): defines the point of interest for plots of HRFs and maps
    """
    if condition == 'audio':
        condition = 'phraseaudio'
        
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
                            jde_output_dir, ymin=-1.55, ymax=1.05,
                            plot_fontsize=25):


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


