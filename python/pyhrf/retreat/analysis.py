import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import nibabel
from nipy.modalities.fmri.hemodynamic_models import _hrf_kernel
try:
    from nipy.modalities.fmri.hemodynamic_models import (sample_condition,
                                                         resample_regressor)
except ImportError:
    from nipy.modalities.fmri.hemodynamic_models import _sample_condition as \
        sample_condition
    from nipy.modalities.fmri.hemodynamic_models import _resample_regressor as\
        resample_regressor
from nipy.modalities.fmri.experimental_paradigm import \
    load_paradigm_from_csv_file
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm


def fix_paradigm(paradigm):
    """Fix a paradigm. Force its amplitude to be an array of floats.

    Parameters
    ----------
    paradigm : nipy.modalities.fmri.experimental_paradigm.Paradigm instance.
        The paradigm to fix.
    """
    if not paradigm.amplitude.dtype is float:
        paradigm.amplitude = np.array(paradigm.amplitude, dtype=float)


def compute_prf_regressor(exp_condition, hrf_model, frametimes,
                          prf_model='physio', prf_matrix=None, con_id='cond',
                          oversampling=16, fir_delays=None):
    """ Convolve regressors with perfusion response function (PRF).

    Parameters
    ----------
    exp_condition : tuple of three 1D arrays of the same shape
        Descriptor of an experimental condition (onsets, duration, values).

    hrf_model : string
        The hrf model to be used. Can be chosen among: 'spm', 'spm_time',
        'spm_time_dispersion', 'canonical', 'canonical_derivative', 'fir'.

    prf_model : string, optional
        The perfusion response function model to be used. Can be chosen
        among: 'physio', 'hrf'.

    prf_matrix : array, optional
        The physiological matrix that links the PRF to the HRF.

    frametimes : array of shape (n)
        The sought.

    con_id : string, optional
        Identifier of the condition.

    oversampling : int, optional
        Oversampling factor to perform the convolution.

    fir_delays : array-like of int
        Onsets corresponding to the fir basis.

    Returns
    -------
    creg : array of shape (n_scans, )
        The computed regressor sampled at frametimes.

    reg_names : string
        The regressor name.

    Notes
    -----
    The different hemodynamic models can be understood as follows:
    'spm': this is the hrf model used in spm
    'spm_time': this is the spm model plus its time derivative (2 regressors)
    'spm_time_dispersion': idem, plus dispersion derivative (3 regressors)
    'canonical': this one corresponds to the Glover hrf
    'canonical with derivative': the Glover hrf + time derivative (2 
    regressors)
    'fir': finite impulse response basis, a set of delayed dirac models
    with arbitrary length. This one currently assumes regularly spaced
    frametimes (i.e. fixed time of repetition).
    It is expected that spm standard and Glover model would not yield
    large differences in most cases.
    """
    # this is the average tr in this session, not necessarily the true tr
    tr = float(frametimes.max()) / (np.size(frametimes) - 1)

    # create the high temporal resolution regressor
    hr_regressor, hr_frametimes = sample_condition(
        exp_condition, frametimes, oversampling)

    # get hrf model and keep only the hrf (no time derivatives)
    hkernel = _hrf_kernel(hrf_model, tr, oversampling, fir_delays)
    hkernel = hkernel[0]

    # compute the prf
    if prf_matrix is None:
        prf_matrix = np.eye(hkernel.shape[-1])

    if prf_model == 'physio':
        # TODO: resample the matrix
        pkernel = prf_matrix.dot(hkernel)
    else:
        pkernel = hkernel

    # convolve the regressor and hrf, and downsample the regressor
    conv_reg = np.array([np.convolve(hr_regressor, pkernel)[
                        :hr_regressor.size]])

    # temporally resample the regressor
    creg = resample_regressor(conv_reg, hr_frametimes, frametimes)

    # generate regressor name
    reg_names = ['perfusion']
    return creg, reg_names


#######################################
# Data and analysis parameters
#######################################

# functionals, grey mask and paradigm
subjects = ['RG130377', 'SC120530', 'CD110147']
subject = 'RG130377'
func_file = os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                         'gin_struct/archives', subject, 'standard',
                         'swr' + subject + '_ASLf_correctionT1.nii')
gm_file = glob.glob(os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                                 'gin_struct/archives', subject, 'standard',
                                 'wc1anat_*.nii'))[0]
anat_file = glob.glob(os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                                   'gin_struct/archives', subject, 'standard',
                                   'wmanat_*.nii'))[0]
paradigm_file = os.path.join('/volatile/new/salma/asl/data/HEROES_ASL1/',
                             'gin_struct/archives',
                             'paradigm_bilateral_v2_no_final_rest.csv')

# Compute binary grey matter mask
img = nibabel.load(gm_file)
mask = img.get_data()
mask[mask <= 0.1] = 0
mask[mask > 0.1] = 1
mask_img = nibabel.Nifti1Image(mask, img.get_affine(), img.get_header())
mask_file = '/tmp/gm_mask.nii'
nibabel.save(mask_img, mask_file)

# timing  # TODO: load timing parameters
n_scans = 164
tr = 2.5
frametimes = np.arange(0, n_scans * tr, tr)

# HRF, PRF, drift and GLM models
hrf_model = 'canonical with derivative'
prf_model = 'physio'
prf_matrix = None
drift_model = 'polynomial'
drift_order = 4
model = 'ar1'  # other possible choice: 'ols'

# write directory
write_dir = os.path.join(os.getcwd(), 'results')
write_dir = os.path.join('/tmp/glm_mask', subject)
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

print('Computation will be performed in directory: %s' % write_dir)

#####################################
# ASL additionnal regressors
#####################################

paradigm = load_paradigm_from_csv_file(paradigm_file)['0']
fix_paradigm(paradigm)

# Baseline ASL regressor
reg = np.ones(n_scans)
reg[1::2] *= 0.5
reg[::2] *= -0.5
add_regs = [reg]
add_reg_names = ['perfusion_baseline']


# Activation ASL regressors
for condition_name in np.unique(paradigm.con_id):
    onsets = paradigm.onset[paradigm.con_id == condition_name]
    values = paradigm.amplitude[paradigm.con_id == condition_name]
    duration = paradigm.duration[paradigm.con_id == condition_name]
    exp_condition = (onsets, duration, values)
    reg, reg_name = compute_prf_regressor(exp_condition, hrf_model,
                                          frametimes, prf_model=prf_model,
                                          prf_matrix=prf_matrix,
                                          con_id=condition_name,
                                          oversampling=16)
    reg[1::2] *= 0.5
    reg[::2] *= -0.5
    add_regs.append(reg)
    add_reg_names.append(reg_name[0] + '_' + condition_name)

add_regs = np.array(add_regs).transpose()
f, axes = plt.subplots(2)
axes[0].plot(add_regs[:, 0])
axes[0].set_ylabel('perfusion baseline')
axes[1].plot(add_regs[:, 1:])
axes[1].set_ylabel('task-related perfusion')
plt.savefig(os.path.join(write_dir, 'asl_regressors.png'))
# TODO: add motion regressors

########################################
# Design matrix
########################################

print('Loading design matrix...')


# Create the design matrix
design_matrix = make_dmtx(frametimes, paradigm=paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, drift_order=drift_order,
                          add_regs=add_regs, add_reg_names=add_reg_names)
reg_names = []
for name in design_matrix.names:
    if 'perfusion' in name:
        name = name.replace('_checkerboard_motor', '')
    else:
        name = name.replace('checkerboard_motor', 'BOLD')
    name = name.replace('_', ' ')
    print name
    reg_names.append(name)
design_matrix.names = reg_names
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

plt.savefig(os.path.join(write_dir, 'design_matrix.png'))

#########################################
# Specify the contrasts
#########################################
# TODO plot the contrasts

# simplest ones
contrasts = {}
n_columns = len(design_matrix.names)
for n, name in enumerate(design_matrix.names):
    contrasts[name] = np.zeros((n_columns,))
    contrasts[name][n] = 1

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

# F-tests
do_Ftests = False
if do_Ftests:
    contrasts['H0: [BOLD] d2500 = d5000'] = np.zeros((2, n_columns))
    contrasts['H0: [BOLD] d2500 = d5000'][0, 0] = 1
    contrasts['H0: [BOLD] d2500 = d5000'][0, 2] = -1
    contrasts['H0: [BOLD] d2500 = d5000'][1, 1] = 1
    contrasts['H0: [BOLD] d2500 = d5000'][1, 3] = -1
    contrasts['H0: [perfusion] left = right'] = np.zeros((2, n_columns))
    contrasts['H0: [perfusion] left = right'][0, 5] = 1
    contrasts['H0: [perfusion] left = right'][0, 7] = -1
    contrasts['H0: [perfusion] left = right'][1, 6] = 1
    contrasts['H0: [perfusion] left = right'][1, 8] = -1


########################################
# Perform a GLM analysis
########################################

print('Fitting a GLM (this takes time)...')
fmri_glm = FMRILinearModel(func_file, design_matrix.matrix,
                           mask=mask_file)
fmri_glm.fit(do_scaling=True, model=model)

#########################################
# Estimate the contrasts
#########################################

print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # save the z_image
    image_path = os.path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
    nibabel.save(z_map, image_path)

    # Create snapshots of the contrasts
    # TODO: cut at motor regions e.g. (-39, -27, 48)
    data = z_map.get_data()
    vmax = data[np.isfinite(data)].max()
    vmin = data[np.isfinite(data)].min()
    vmax = max(-vmin, vmax)
    # TODO from nipy.labs.viz_tools.coord_tools import find_cut_coords 
    # and use cut_coords = find_cut_coords(z_map.get_data()),
    # mask=mask_img.get_data()) with slicer='ortho' to produce cuts

    anat_img = nibabel.load(anat_file)
    plot_map(z_map.get_data(), z_map.get_affine(),
             cmap=cm.cold_hot, vmin=-vmax, vmax=vmax,
             slicer='z', black_bg=True, threshold=3.1,  # 3.1 for pval<1e-3
             title=contrast_id, anat=anat_img.get_data(),
             anat_affine=anat_img.get_affine())
    plt.savefig(os.path.join(write_dir, '%s_z_map.png' % contrast_id))

print("All the  results were witten in %s" % write_dir)

plt.show()

