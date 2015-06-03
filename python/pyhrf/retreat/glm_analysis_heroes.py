import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import nibabel
from nipy.modalities.fmri import experimental_paradigm, glm, design_matrix
from nipy.labs import viz
from nipy.labs.statistical_mapping import (get_3d_peaks, cluster_stats)
from nipy.labs.utils.mask import compute_mask_files

from locator import meta_auditory, meta_motor, get_maxima
from glm_tools import combine_masks, fix_paradigm, compute_prf_regressor


#######################################
# Data and analysis parameters
#######################################

subjects = ['RG130377', 'SC120530', 'CD110147']
subjects = np.genfromtxt(
    '/volatile/new/salma/asl/data/HEROES_ASL1/subjects.txt' ,dtype=str)
subject = subjects[0]
data_dir = os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                        'gin_struct/archives', subject, 'standard')

# load functionals, anatomical, motion parameters and paradigm
func_file = os.path.join(data_dir, 'swr' + subject + '_ASLf_correctionT1.nii')
anat_file = glob.glob(os.path.join(data_dir, 'wmanat_*brain*'))[0]
mvt_file = glob.glob(os.path.join(data_dir, 'rp_*.txt'))[0]
paradigm_file = os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                             'gin_struct/archives',
                             'paradigm_bilateral_v2_no_final_rest.csv')

# load needed images to create the mask
unsmoothed_file = os.path.join(data_dir,
                               'wr' + subject + '_ASLf_correctionT1.nii')
gm_file = glob.glob(os.path.join(data_dir, 'wc1anat_*.nii'))[0]
wm_file = glob.glob(os.path.join(data_dir, 'wc2anat_*.nii'))[0]

# Compute binary mask beyonf the neck
# TODO: use modulated masks
func_mask_file = os.path.join(data_dir, 'func_mask_' + subject + '.nii')
img = nibabel.load(unsmoothed_file)
func_mask = compute_mask_files(unsmoothed_file, func_mask_file,  m=0.3, M=.4,
                               cc=0, opening=0)

# Compute probabilistic tissue mask
img = nibabel.load(gm_file)
gm_data = img.get_data()
img = nibabel.load(wm_file)
wm_data = img.get_data()
tissue_data = (gm_data + wm_data) / 2.
tissue_img = nibabel.Nifti1Image(tissue_data, img.get_affine(),
                                 img.get_header())
tissue_mask_file = os.path.join(data_dir, 'tissue_mask.nii')
nibabel.save(tissue_img, tissue_mask_file)

# Combine neck mask and tissue mask
mask_file = os.path.join(data_dir, 'cut_tissue_mask.nii')
mask_img = combine_masks(func_mask_file, tissue_mask_file, mask_file,
                         threshold=.25)

# timing  # TODO: load timing parameters
n_scans = 164
tr = 2.5
frametimes = np.arange(0, n_scans * tr, tr)

# HRF and PRF
hrf_model = 'canonical'
prf_model = 'physio'
hrf_length = 32.
oversampling = 16  # default oversampling in nipy
dt = tr / oversampling
from pyhrf.sandbox.physio_params import (linear_rf_operator,
                                         PHY_PARAMS_KHALIDOV11)
prf_matrix = linear_rf_operator(hrf_length / dt,  PHY_PARAMS_KHALIDOV11, dt,
                                calculating_brf=False)

# drift and GLM models
drift_model = 'polynomial'
drift_order = 4
model = 'ar1'  # other possible choice: 'ols'

# write directory
write_dir = os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                         'outputs/tissue_mask/motion_regs/physio', subject)
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

print('Computation will be performed in directory: %s' % write_dir)

#####################################
# Regressors
#####################################

paradigm = experimental_paradigm.load_paradigm_from_csv_file(
    paradigm_file)['0']
fix_paradigm(paradigm)

# Baseline ASL regressor
reg = np.ones(n_scans)
reg[1::2] *= 0.5
reg[::2] *= -0.5
add_regs = [reg]
add_reg_names = ['perfusion_baseline']

# Activation BOLD and ASL regressors
bold_regs = []
normalize = True  # normalize the HRF and PRF
for condition_name in np.unique(paradigm.con_id):
    onsets = paradigm.onset[paradigm.con_id == condition_name]
    values = paradigm.amplitude[paradigm.con_id == condition_name]
    duration = paradigm.duration[paradigm.con_id == condition_name]
    exp_condition = (onsets, duration, values)
    bold_regs.append(compute_prf_regressor(exp_condition, hrf_model,
                                           frametimes, prf_model=prf_model,
                                           prf_matrix=None,
                                           con_id=condition_name,
                                           oversampling=16,
                                           normalize=normalize)[0])
    reg, reg_name = compute_prf_regressor(exp_condition, hrf_model,
                                          frametimes, prf_model=prf_model,
                                          prf_matrix=prf_matrix,
                                          con_id=condition_name,
                                          oversampling=16, normalize=normalize)
    reg[1::2] *= 0.5
    reg[::2] *= -0.5
    add_regs.append(reg)
    add_reg_names.append(reg_name[0] + '_' + condition_name)

bold_regs = np.array(bold_regs).squeeze(axis=-1)
bold_regs = bold_regs.transpose()
add_regs = np.array(add_regs).transpose()

# Motion regressors
add_regs = np.hstack((add_regs, np.genfromtxt(mvt_file, skip_header=1)))
add_reg_names += ['translation x', 'translation y', 'translation z',
                  'pitch', 'roll', 'yaw']

# Plot the regressors
f, axes = plt.subplots(3)
axes[0].plot(bold_regs)
axes[0].set_ylabel('task-related BOLD')
axes[1].plot(add_regs[:, 0])
axes[1].set_ylabel('perfusion baseline')
axes[2].plot(add_regs[:, 1:5])
axes[2].set_ylabel('task-related perfusion')
plt.savefig(os.path.join(write_dir, 'asl_regressors.png'))

f, axes = plt.subplots(2)
axes[0].plot(add_regs[:, 5:8])
axes[0].set_ylabel('translation (mm)')
axes[1].plot(add_regs[:, 8:])
axes[1].set_ylabel('rotation (deg)')
plt.savefig(os.path.join(write_dir, 'mvt_regressors.png'))

# TODO: include mean WM and CSF signals
########################################
# Design matrix
########################################

print('Loading design matrix...')

# Create the design matrix
dmtx = design_matrix.make_dmtx(frametimes, paradigm=paradigm,
                               hrf_model=hrf_model, drift_model=drift_model,
                               drift_order=drift_order, add_regs=add_regs,
                               add_reg_names=add_reg_names)
reg_names = []
for name in dmtx.names:
    if 'perfusion' in name:
        name = name.replace('_checkerboard_motor', '')
    else:
        name = name.replace('checkerboard_motor', 'BOLD')
    name = name.replace('_', ' ')
    print name
    reg_names.append(name)
dmtx.names = reg_names

# Use normalized BOLD regressors
if normalize:
    dmtx.matrix[:, :4] = bold_regs

ax = dmtx.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

plt.savefig(os.path.join(write_dir, 'design_matrix.png'))
#########################################
# Specify the contrasts
#########################################
# TODO plot the contrasts

# simplest ones
contrasts = {}
n_columns = len(dmtx.names)
for n, name in enumerate(dmtx.names):
    contrasts[name] = np.zeros((n_columns,))
    contrasts[name][n] = 1

import copy
indiv_contrasts = copy.copy(contrasts)

# t_tests
contrasts['[BOLD d2500] left - right'] = \
    contrasts['BOLD d2500 left'] - contrasts['BOLD d2500 right']
contrasts['[BOLD d5000] left - right'] = \
    contrasts['BOLD d5000 left'] - contrasts['BOLD d5000 right']
contrasts['[perfusion d2500] left - right'] = \
    contrasts['perfusion d2500 left'] - contrasts['perfusion d2500 right']
contrasts['[perfusion d5000] left - right'] = \
    contrasts['perfusion d5000 left'] - contrasts['perfusion d5000 right']

# pooled t-tests
contrasts['BOLD left'] = contrasts['BOLD d2500 left'] + \
    contrasts['BOLD d5000 left']
contrasts['BOLD right'] = contrasts['BOLD d2500 right'] + \
    contrasts['BOLD d5000 right']
contrasts['perfusion left'] = contrasts['perfusion d2500 left'] + \
    contrasts['perfusion d5000 left']
contrasts['perfusion right'] = contrasts['perfusion d2500 right'] + \
    contrasts['perfusion d5000 right']
contrasts['[BOLD] left - right'] = \
    contrasts['[BOLD d2500] left - right'] + \
    contrasts['[BOLD d5000] left - right']
contrasts['[perfusion] left - right'] = \
    contrasts['[perfusion d2500] left - right'] + \
    contrasts['[perfusion d5000] left - right']
contrasts['movements'] = contrasts['translation x'] + \
    contrasts['translation y'] + contrasts['translation z'] + \
    contrasts['roll'] + contrasts['pitch'] + contrasts['yaw']

########################################
# Perform a GLM analysis
########################################

print('Fitting a GLM (this takes time)...')
fmri_glm = glm.FMRILinearModel(func_file, dmtx.matrix,
                               mask=mask_file)
fmri_glm.fit(do_scaling=True, model=model)

#########################################
# Estimate the contrasts
#########################################

print('Computing contrasts...')
# Save the t-maps
for index, (contrast_id, contrast_val) in enumerate(indiv_contrasts.items()):
    print(' Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # save the z_image
    image_path = os.path.join(write_dir, 'to_aina',
                              '%s_t_map.nii' % contrast_id)
    t_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id,
                               output_stat=True, output_z=False)
    nibabel.save(t_map, image_path)


# WIP: Save the z-maps and clusters labels in harvard-oxford
from scipy.stats import norm
n_voxels = np.sum(mask_img.get_data())
print n_voxels
print np.sum(gm_data)
threshold = norm.isf(0.05 / n_voxels)  # Z-score threshold, Bonferroni
unc_threshold = norm.isf(1e-3)  # Z-score threshold, no correction
to_write = []
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(' Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # save the z_image
    image_path = os.path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
    nibabel.save(z_map, image_path)

    # Create snapshots of the contrasts
    z_data = z_map.get_data()
    signif_mask = np.abs(z_data) > threshold
    vmax = z_data[np.isfinite(z_data)].max()
    vmin = z_data[np.isfinite(z_data)].min()
    vmax = max(-vmin, vmax)
    anat_img = nibabel.load(anat_file)

    if np.max(np.abs(z_data)) > threshold:
        viz.plot_map(z_map.get_data(), z_map.get_affine(),
                     cmap=viz.cm.cold_hot, vmin=-vmax, vmax=vmax,
                     slicer='z', black_bg=True, threshold=unc_threshold,
                     title=contrast_id, anat=anat_img.get_data(),
                     anat_affine=anat_img.get_affine())
        plt.savefig(os.path.join(write_dir, '%s_z_map.png' % contrast_id))

    reg_type = 'BOLD'
    if reg_type in contrast_id and contrast_id != 'perfusion baseline':
        # Get motor and primary auditory regions from the literature
        auditory_names, auditory_coords = meta_auditory()
        motor_names, motor_coords = meta_motor()
        regions = auditory_names[:2] + motor_names
        cuts = auditory_coords[:2] + motor_coords
        regions = []
        cuts = []

        # Find the clusters
        # TODO: debug nipy.labs.viz_tools.coord_tools.find_cut_coords
        # TODO: use the same cuts over contrasts ?
        if reg_type == 'perfusion':
            cluster_th = 0
        else:
            cluster_th = 0
        clusters, info = cluster_stats(z_map, mask=mask_img, height_th=.05,
                                       height_control='bonferroni',
                                       cluster_th=cluster_th, nulls={})
        if clusters:
            print('  {} clusters'.format(len(clusters)))
            to_write.append('Contrast {0}: {1} clusters\n'.format(
                contrast_id, len(clusters)))
            for n, cluster in enumerate(clusters):
                maxima_regions, maxima_coords = get_maxima(cluster,
                                                           min_distance=20.)
                print '   cluster of size {0}: {1}'.format(cluster['size'],
                                                           maxima_regions)
                to_write.append(' cluster of size {0}: {1}\n'.format(
                    cluster['size'], maxima_regions))
                if n < 6 or 'Heschl' in maxima_regions:
                    n_regions = min(3, len(maxima_regions))
                    regions += maxima_regions[:n_regions]
                    cuts += maxima_coords[:n_regions]

        # TODO: move to locator.py
        # Find the peaks
        plot_peaks = False
        peaks = get_3d_peaks(z_map, mask=None, threshold=threshold, nn=18,
                             order_th=0)
        if peaks and plot_peaks:
            n_peaks = min(len(peaks), 0)
            for n, peak in enumerate(peaks[:n_peaks]):
                regions.append(' peak {}'.format(n))
                cuts.append(tuple(peak['pos']))
                print peak['val']

        for (region, cut_coords) in zip(regions, cuts):
            title = contrast_id + ', ' + region
            if np.max(np.abs(z_data)) > unc_threshold:
                viz.plot_map(z_map.get_data(), z_map.get_affine(),
                             cmap=viz.cm.cold_hot, vmin=-vmax, vmax=vmax,
                             slicer='ortho', black_bg=True,
                             threshold=unc_threshold,
                             title=title, anat=anat_img.get_data(),
                             anat_affine=anat_img.get_affine(),
                             cut_coords=cut_coords)
                region = region.replace(' ', '_')
                if len(region) > 10:
                    region = region[:10]
                folder = os.path.join(write_dir, region)
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                plt.savefig(os.path.join(folder,
                                         '{0}_z_map.png'.format(contrast_id)))
        plt.show()
# Write clusters sizes and regions to file
output_txt = os.path.join(write_dir, 'clusters.txt')
with open(output_txt, "w") as text_file:
    for line in to_write:
        text_file.writelines(line)

print("All the  results were witten in %s" % write_dir)
