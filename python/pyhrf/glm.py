# -*- coding: utf-8 -*-
import os
import os.path as op
import numpy as np
import glob

import pyhrf
from pyhrf import FmriData
from pyhrf.ndarray import expand_array_in_mask
from pyhrf.tools.io import read_volume, write_volume
from pyhrf.ndarray import xndarray

import scipy as sp

import tempfile
import shutil
#from nipy.labs import compute_mask_files

from nibabel import load, save, Nifti1Image
import nipy.labs.glm
from nipy.modalities.fmri import design_matrix as dm
try:
    from nipy.modalities.fmri.experimental_paradigm import \
     load_paradigm_from_csv_file
except ImportError:
    from nipy.modalities.fmri.experimental_paradigm import \
     load_protocol_from_csv_file as load_paradigm_from_csv_file

#from nipy.labs import compute_mask_files

#from nipy.labs.viz import plot_map, cm

def glm_nipy(fmri_data, contrasts=None, hrf_model='Canonical',
             drift_model='Cosine', hfcut=128,
             residuals_model='spherical', fit_method='ols',
             fir_delays=[0],
             rescale_results=False, rescale_factor=None):

    """
    Perform a GLM analysis on fMRI data using the implementation of Nipy.

    Args:
        fmri_data (pyhrf.core.FmriData): the input fMRI data defining the
            paradigm and the measured 3D+time signal.
        contrasts (dict): keys are contrast labels and values are arithmetic
            expressions involving regressor names. Valid names are:
            * names of experimental conditions as defined in fmri_data
            * constant
        hrf_model: "Canonical", "Canonical with Derivative", "FIR"
        residuals_model: "spherical", "ar1"
        fit_method: "ols", "kalman" (If residuals_model is "ar1" then method
            is set to "kalman" and this argument is ignored)
        fir_delays: list of integers indicating the delay of each FIR coefficient
                    (in terms of scans). Eg if TR = 2s. and we want a FIR
                    duration of 20s.: fir_delays=range(10)
    Returns:
        (glm instance, design matrix, dict of contrasts of objects)

    Examples:
    >>> from pyhrf.core import FmriData
    >>> from pyhrf.glm import glm_nipy
    >>> g,dmtx,con = glm_nipy(FmriData.from_vol_ui())
    >>> g,dmtx,con = glm_nipy(FmriData.from_vol_ui(), \
                              contrasts={'A-V':'audio-video'})
    """

    paradigm = fmri_data.paradigm.to_nipy_paradigm()


    # BOLD data
    Y = fmri_data.bold.T
    n_scans = Y.shape[1]
    # pyhrf.verbose(1, 'Input BOLD: nvox=%d, nscans=%d' %Y.shape)

    # Design matrix
    frametimes = np.linspace(0, (n_scans-1)*fmri_data.tr, n_scans)
    design_matrix = dm.make_dmtx(frametimes, paradigm,
                                 hrf_model=hrf_model,
                                 drift_model=drift_model, hfcut=hfcut,
                                 fir_delays=fir_delays)

    ns, nr = design_matrix.matrix.shape
    pyhrf.verbose(2, 'Design matrix built with %d regressors:' %nr)
    for rn in design_matrix.names:
        pyhrf.verbose(2, '    - %s' %rn)

    # ax = design_matrix.show()
    # ax.set_position([.05, .25, .9, .65])
    # ax.set_title('Design matrix')
    # plt.savefig(op.join(output_dir, 'design_matrix.png'))

    # GLM fit
    my_glm = nipy.labs.glm.glm.glm()
    pyhrf.verbose(2, 'Fit GLM - method: %s, residual model: %s' \
                      %(fit_method,residuals_model))
    my_glm.fit(Y.T, design_matrix.matrix, method=fit_method,
               model=residuals_model)

    from pyhrf.tools import map_dict
    from pyhrf.paradigm import contrasts_to_spm_vec

    if rescale_results:

        # Rescale by the norm of the HRF:
        # from nipy.modalities.fmri.hemodynamic_models import _hrf_kernel, \
        #     sample_condition
        # oversampling = 16
        # hrfs = _hrf_kernel(hrf_model, fmri_data.tr, oversampling,
        #                    fir_delays=fir_delays)
        # hframetimes = np.linspace(0, 32., int(32./fmri_data.tr))
        # hr_regressor, hr_frametimes = sample_condition(
        #     (np.array([0]),np.array([0]),np.array([1])),
        #     hframetimes, oversampling)
        # from scipy.interpolate import interp1d
        # for i in xrange(len(hrfs)):
        #     f = interp1d(hr_frametimes, hrfs[i])
        #     hrfs[i] = f(hframetimes).T

        # n_conds = len(fmri_data.paradigm.stimOnsets)
        # for i in xrange(n_conds * len(hrfs)):
        #     h = hrfs[i%len(hrfs)]
        #     my_glm.beta[i] = my_glm.beta[i] * (h**2).sum()**.5

        #my_glm.variance = np.zeros_like(my_glm.beta)
        if 1:
            if rescale_results and rescale_factor is None:
                #Rescale by the norm of each regressor in the design matrix
                dm_reg_norms = (design_matrix.matrix**2).sum(0)**.5
                pyhrf.verbose(2,'GLM results (beta and con effects) are '\
                                  'rescaled by reg norm. Weights: %s ' \
                                  %str(dm_reg_norms))
    
                for ib in xrange(my_glm.beta.shape[0]):
                    my_glm.beta[ib] = my_glm.beta[ib] * dm_reg_norms[ib]
                    #my_glm.nvbeta[ib,:] = my_glm.nvbeta[ib,:] * dm_reg_norms[ib]**2
    
            else:
                pyhrf.verbose(2,'GLM results (beta and con effects) are '\
                                  'rescaled by input scale factor.')
    
                # Use input rescale factors:
                for ib in xrange(rescale_factor.shape[0]):
                    my_glm.beta[ib] = my_glm.beta[ib] * rescale_factor[ib]
                    #TOCHECK: nvbeta seems to be a covar matrix between reg
                    # -> we dont get position-specific variances ...
                    #my_glm.nvbeta[ib,:] = my_glm.nvbeta[ib,:] * rescale_factor[ib]**2
    
    if contrasts is not None:
        con_vectors = contrasts_to_spm_vec(design_matrix.names, contrasts)
        # if rescale_results:
        #     for con_vec in con_vectors.itervalues():
        #         con_vec *= dm_reg_norms
        contrast_result = map_dict(my_glm.contrast, con_vectors)
    else:
        contrast_result = None


    return my_glm, design_matrix, contrast_result

#actually: not possible to compute PPM from glm results
#Should relaunch estimation with propoer model under SPM
#def PPMcalculus_glmWN(beta, var_beta, dm, threshold_value):
    '''
    #Function to compute PPM after glm estimation with white noise
    #Inputs:
    #-beta: beta values
    #-var_beta: normalized variance of beta
    #-dm: design matrix
    #-threshol value: phyisiologically relevant size of effect
            #--> take a percentage of mean signal value
    #GLM estimation must be performed with nipy
    #Compute the pp for one voxel!
    #'''
    #from scipy.integrate import quad

    #X = dm.matrix
    #mean_apost = 1./(np.dot(dm.transpose(), dm)*1./(var_beta) + 1./beta)
    #var_apost = mean_apost*np.dot(dm.transpose(), dm)*1./(var_beta)

    #f_distrib = lambda t: 1/np.sqrt(2*np.pi*var_apost**2)*np.exp(- (t - mean_apost)**2 / (2*var_apost**2) )
    #Proba = quad(f_distrib, threshold_value, float('inf'))[0]

    #return Proba




def glm_nipy_from_files(bold_file, tr,  paradigm_csv_file, output_dir,
                        mask_file, session=0, contrasts=None,
                        con_test_baseline=0.0,
                        hrf_model='Canonical',
                        drift_model='Cosine', hfcut=128,
                        residuals_model='spherical', fit_method='ols',
                        fir_delays=[0]):
    """
    #TODO: handle surface data
    hrf_model : Canonical | Canonical with Derivative | FIR

    """

    fdata = FmriData.from_vol_files(mask_file, paradigm_csv_file,
                                    [bold_file], tr)
    g, dm, cons = glm_nipy(fdata, contrasts=contrasts, hrf_model=hrf_model,
                            hfcut=hfcut, drift_model=drift_model,
                            residuals_model=residuals_model,
                            fit_method=fit_method, fir_delays=fir_delays)

    ns, nr = dm.matrix.shape
    cdesign_matrix = xndarray(dm.matrix, axes_names=['time','regressor'],
                            axes_domains={'time':np.arange(ns)*tr,
                                          'regressor':dm.names})

    cdesign_matrix.save(op.join(output_dir, 'design_matrix.nii'))

    beta_files = []
    beta_values = dict.fromkeys(dm.names)
    beta_vars = dict.fromkeys(dm.names)
    beta_vars_voxels = dict.fromkeys(dm.names)
    for ib, bname in enumerate(dm.names):
        #beta values
        beta_vol = expand_array_in_mask(g.beta[ib], fdata.roiMask>0)
        beta_fn = op.join(output_dir, 'beta_%s.nii' %bname)
        write_volume(beta_vol, beta_fn, fdata.meta_obj)
        beta_files.append(beta_fn)
        beta_values[bname] = beta_vol

        #normalized variance of betas
        beta_vars[bname] = sp.diag(g.nvbeta)[ib] #variance: diag of cov matrix
        sig2 = g.s2 #ResMS
        var_cond = sp.diag(g.nvbeta)[ib]*g.s2 #variance for all voxels, condition ib
        beta_vars_voxels[bname] = var_cond
        #beta_var_fn = op.join(output_dir, 'var_beta_%s.nii' %bname)
        #write_volume(beta_var, beta_var_fn, fdata.meta_obj)
        #beta_var_files.append(beta_var_fn)

    if cons is not None:
        con_files = []
        pval_files = []
        for cname, con in cons.iteritems():
            con_vol = expand_array_in_mask(con.effect, fdata.roiMask>0)
            con_fn = op.join(output_dir, 'con_effect_%s.nii' %cname)
            write_volume(con_vol, con_fn, fdata.meta_obj)
            con_files.append(con_fn)

            pval_vol = expand_array_in_mask(con.pvalue(con_test_baseline),
                                            fdata.roiMask>0)
            pval_fn = op.join(output_dir, 'con_pvalue_%s.nii' %cname)
            write_volume(pval_vol, pval_fn, fdata.meta_obj)
            pval_files.append(pval_fn)
    else:
        con_files = None
        pval_files = None

    dof = g.dof
    #if do_ppm:
        #for

    #TODO: FIR stuffs
    return beta_files, beta_values, beta_vars_voxels, dof#, con_files, pval_files



    # Paradigm

    # Fix bug in nipy/modalities/fmri/experimental_paradigm.py
    # -> amplitudes are not converted to floats
    # @@ -187,7 +187,7 @@ def load_paradigm_from_csv_file(path, session=None):
    #          if len(row) > 3:
    #              duration.append(float(row[3]))
    #          if len(row) > 4:
    # -            amplitude.append(row[4])
    # +            amplitude.append(float(row[4]))

if 0:
    paradigm = load_paradigm_from_csv_file(paradigm_csv_file,
                                           session=str(session))
    if paradigm is None:
        raise Exception('Failed to load paradigm data from %s (session=%d)' \
                            %(paradigm_csv_file, session))
    pyhrf.verbose(1, 'Loaded paradigm: condition=%s, nb events=%d'
                  %(str(list(set(paradigm.con_id))),paradigm.n_events))

    assert op.exists(mask_file)
    # # Functional mask
    # if not op.exists(mask_file):
    #     pyhrf.verbose(1, 'Mask file does not exist. Computing mask from '\
    #                       'BOLD data')
    #     compute_mask_files(data_path, mask_file, False, 0.4, 0.9)
    #     mask_array = compute_mask_files(bold_file, mask_file,
    #                                     False, 0.4, 0.9)

    mask_array = load(mask_file).get_data()
    m = np.where(mask_array > 0)
    pyhrf.verbose(1, 'Mask: shape=%s, nb vox in mask=%d'
                  %(str(mask_array.shape), mask_array.sum()))

    # BOLD data
    fmri_image = load(bold_file)
    #pyhrf.verbose(1, 'BOLD shape : %s' %str(fmri_image.get_shape()))
    Y = fmri_image.get_data()[m[0],m[1],m[2],:]
    n_scans = Y.shape[1]
    pyhrf.verbose(1, 'Loaded BOLD: nvox=%d, nscans=%d' %Y.shape)

    # Design matrix
    frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
    design_matrix = dm.make_dmtx(frametimes, paradigm,
                                 hrf_model=hrf_model,
                                 drift_model=drift_model, hfcut=hfcut,
                                 fir_delays=fir_delays)
    ns, nr = design_matrix.matrix.shape
    pyhrf.verbose(2, 'Design matrix built with %d regressors:' %nr)
    for rn in design_matrix.names:
        pyhrf.verbose(2, '    - %s' %rn)


    cdesign_matrix = xndarray(design_matrix.matrix,
                            axes_names=['time','regressor'],
                            axes_domains={'time':np.arange(ns)*tr,
                                          'regressor':design_matrix.names})

    cdesign_matrix.save(op.join(output_dir, 'design_matrix.nii'))
    import cPickle
    f = open(op.join(output_dir, 'design_matrix.pck'), 'w')
    cPickle.dump(design_matrix, f)
    f.close()

    # import matplotlib.plt as mp
    # design_matrix.show()
    # mp.show()

    # ax = design_matrix.show()
    # ax.set_position([.05, .25, .9, .65])
    # ax.set_title('Design matrix')
    # plt.savefig(op.join(output_dir, 'design_matrix.png'))
    # design_matrix.save(...)

    # GLM fit
    my_glm = nipy.labs.glm.glm()
    pyhrf.verbose(1, 'Fit GLM - method: %s, residual model: %s' \
                      %(fit_method,residuals_model))
    glm = my_glm.fit(Y.T, design_matrix.matrix, method=fit_method,
                     model=residuals_model)

    # Beta outputs
    beta_files = []
    affine = fmri_image.get_affine()
    for ib, bname in enumerate(design_matrix.names):
        beta_vol = expand_array_in_mask(my_glm.beta[ib], mask_array)
        beta_image = Nifti1Image(beta_vol, affine)
        beta_file = op.join(output_dir, 'beta_%s.nii' %bname)
        save(beta_image, beta_file)
        beta_files.append(beta_file)


    from nipy.modalities.fmri.hemodynamic_models import _regressor_names
    from pyhrf.ndarray import MRI3Daxes
    if hrf_model == 'FIR':
        drnames = design_matrix.names
        nvox = mask_array.sum()
        print 'nvox:', nvox
        for cn in set(paradigm.con_id):
            fir_rnames = _regressor_names(cn, 'FIR', fir_delays)
            #lfir = len(fir_rnames)
            beta_fir = np.array([my_glm.beta[drnames.index(n)] \
                                     for n in fir_rnames])

            norm_fir = (beta_fir**2).sum(axis=0)**.5

            print 'np.diff(beta_fir,0):', np.diff(beta_fir,0).shape
            print 'np.zeros(1, nvox):', np.zeros((1, nvox)).shape
            beta_fir_diff1 = np.concatenate((np.zeros((1, nvox)),
                                             np.diff(beta_fir,axis=0)))

            beta_fir_diff2 = np.concatenate((np.zeros((1, nvox)),
                                             np.diff(beta_fir_diff1,axis=0)))

            print 'beta_fir:', beta_fir.shape
            print 'beta_fir_diff1:', beta_fir_diff1.shape
            print 'beta_fir_diff2:', beta_fir_diff2.shape
            beta_fir_d = np.array([beta_fir, beta_fir_diff1,
                                   beta_fir_diff2])
            cbeta_fir = xndarray(beta_fir_d,
                               axes_names=['diff', 'time', 'voxel'],
                               axes_domains={'time':np.array(fir_delays)*tr,
                                             'diff':['d0','d1','d2']})
            cbeta_fir = cbeta_fir.expand(mask_array, 'voxel', MRI3Daxes)
            cbeta_fir.save(op.join(output_dir, 'FIR_%s.nii' %cn))

            norm_fir_d = (beta_fir_d**2).sum(axis=1)**.5
            cnorm_fir_d = xndarray(norm_fir_d,
                                 axes_names=['diff', 'voxel'],
                                 axes_domains={'diff':['d0','d1','d2']})
            cnorm_fir_d = cnorm_fir_d.expand(mask_array, 'voxel', MRI3Daxes)
            cnorm_fir_d.save(op.join(output_dir, 'FIR_norm_%s.nii' %cn))

            fn = op.join(output_dir, 'FIR_norm_01_%s.nii' %cn)
            cnorm_fir_d.sub_cuboid(diff='d0').rescale_values().save(fn)

            ttp = beta_fir.argmax(0)
            cttp = xndarray(ttp, axes_names=['voxel'])
            cttp = cttp.expand(mask_array, 'voxel', MRI3Daxes)
            cttp.save(op.join(output_dir, 'TTP_%s.nii' %cn))

            cttp.rescale_values().save(op.join(output_dir, \
                                                   'FIR_TTP_01_%s.nii' %cn))

            def array_summary(a):
                return '%s -- %1.3f (%1.3f) [%1.3f %1.3f]' \
                    %(str(a.shape), a.mean(), a.std(), a.min(), a.max())

            print 'norm_fir: %s' %array_summary(norm_fir)
            print 'beta_fir:', array_summary(beta_fir)
            print 'beta_fir/norm_fir:', array_summary(beta_fir/norm_fir)
            nrj = np.sum(np.diff(np.diff(beta_fir/norm_fir,0),0)**2, 0)**.5
            print 'nrj:', array_summary(nrj)
            cnrj = xndarray(nrj, axes_names=['voxel'])
            cnrj = cnrj.expand(mask_array, 'voxel', MRI3Daxes)
            cnrj.save(op.join(output_dir, 'fir_nrj_%s.nii' %cn))

    cresiduals = xndarray(my_glm.s2, axes_names=['voxel'],
                        meta_data=(affine, fmri_image.get_header()))
    fn = op.join(output_dir, 'residuals.nii')
    cresiduals.expand(mask_array, 'voxel', MRI3Daxes).save(fn)


    pyhrf.verbose(1, 'Saved %d beta files in %s' \
                      %(len(beta_files),output_dir))

    return beta_files, beta_vars

    # import numpy as np
    # import os.path as op
    # import matplotlib.pyplot as plt
    # import tempfile

    # #from nipy.labs import compute_mask_files
    # from nibabel import load, save, Nifti1Image
    # #import get_data_light
    # import nipy.labs.glm
    # import nipy.labs.utils.design_matrix as dm
    # from nipy.labs.viz import plot_map, cm

    # data_path = op.join(output_path, 'bold.nii')
    # paradigm_file = op.join(output_path, 'paradigm.csv')

    # tr = 1.
    # n_scans = bold.shape[0]
    # frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
    # conditions = [c.name for c in condition_defs]

    # # confounds
    # hrf_model = 'Canonical' #'Canonical With Derivative'
    # drift_model = "Blank" #"Cosine"
    # hfcut = 0

    # # write directory
    # swd = op.join(output_path, 'GLM_nipy')
    # if not op.exists(swd): os.makedirs(swd)

    # print 'Computation will be performed in temporary directory: %s' % swd

    # ########################################
    # # Design matrix
    # ########################################

    # print 'Loading design matrix...'
    # paradigm = load_paradigm_from_csv_file(paradigm_file, session=0)

    # design_matrix = dm.DesignMatrix(frametimes, paradigm, hrf_model=hrf_model,
    #                                 drift_model=drift_model, hfcut=hfcut)

    # ax = design_matrix.show()
    # ax.set_position([.05, .25, .9, .65])
    # ax.set_title('Design matrix')
    # plt.savefig(op.join(swd, 'design_matrix.png'))
    # # design_matrix.save(...)

    # ########################################
    # # Mask the data
    # ########################################

    # #print 'Computing a brain mask...'
    # #mask_path = op.join(swd, 'mask.nii')
    # #mask_array = compute_mask_files( data_path, mask_path, False, 0.4, 0.9)
    # #mask_array = np.ones(tuple(reversed(bscan.shape)),dtype=int)
    # mask_array = np.ones_like(simulation_items['labels_vol'][0])

    # ########################################
    # # Perform a GLM analysis
    # ########################################

    # print 'Fitting a GLM (this takes time)...'
    # #fmri_image = load(data_path)
    # #m = np.where(mask_array)
    # #Y = fmri_image.get_data()[m[0],m[1],m[2],:]
    # model = "spherical" #"ar1"
    # method = "ols" #"kalman"
    # my_glm = nipy.labs.glm.glm()
    # glm = my_glm.fit(bold, design_matrix.matrix,
    #                  method=method, model=model)


    # #########################################
    # # Beta outputs
    # #########################################

    # for ib, bname in enumerate(design_matrix.names):
    #     writeImageWithPynifti(expand_array_in_mask(my_glm.beta[ib], mask_array),
    #                           op.join(swd, 'beta_%s.nii' %bname))

    # #########################################
    # # Specify the contrasts
    # #########################################

    # # simplest ones
    # contrasts = {}
    # contrast_id = conditions
    # for i in range(len(conditions)):
    #     contrasts['%s' % conditions[i]]= np.eye(len(design_matrix.names))[2*i]


    # #########################################
    # # Estimate the contrasts
    # #########################################

    # print 'Computing contrasts...'
    # for index, contrast_id in enumerate(contrasts):
    #     print '  Contrast % 2i out of %i: %s' % (index+1,
    #                                              len(contrasts), contrast_id)
    #     lcontrast = my_glm.contrast(contrasts[contrast_id])
    #     #
    #     contrast_path = op.join(swd, '%s_z_map.nii'% contrast_id)
    #     write_array = mask_array.astype(np.float)
    #     write_array[np.where(mask_array)] = lcontrast.zscore()
    #     #contrast_image = Nifti1Image(write_array) #, fmri_image.get_affine() )
    #     #save(contrast_image, contrast_path)
    #     writeImageWithPynifti(write_array, contrast_path)
    #     #affine = fmri_image.get_affine()


    #     # vmax = max(-write_array.min(), write_array.max())
    #     # plot_map(write_array, affine,
    #     #          cmap=cm.cold_hot,
    #     #          vmin=-vmax,
    #     #          vmax=vmax,
    #     #          anat=None,
    #     #          figure=10,
    #     #          threshold=2.5)
    #     # plt.savefig(op.join(swd, '%s_z_map.png' % contrast_id))
    #     # plt.clf()



    # #########################################
    # # End
    # #########################################

    # print "All the  results were witten in %s" %swd

    # # plot_map(write_array, affine,
    # #                 cmap=cm.cold_hot,
    # #                 vmin=-vmax,
    # #                 vmax=vmax,
    # #                 anat=None,
    # #                 figure=10,
    # #                 threshold=3)

    # """
    # plot_map(write_array, affine,
    #                 cmap=cm.cold_hot,
    #                 vmin=-vmax,
    #                 vmax=vmax,
    #                 anat=None,
    #                 figure=10,
    #                 threshold=3, do3d=True)

    # from nipy.labs import viz3d
    # viz3d.plot_map_3d(write_array, affine,
    #                 cmap=cm.cold_hot,
    #                 vmin=-vmax,
    #                 vmax=vmax,
    #                 anat=None,
    #                 threshold=3)
    # """
    # #plt.show()


if pyhrf.cfg['global']['spm_path'] is not None:
    def glm_matlab_from_files(bold_file, tr, paradigm_csv_file, output_dir,
                              mask_file, hf_cut=128, hack_mask=False):
        """
        Only mono-session
        #TODO: compute mask if mask_file does not exist
        #TODO: handle contrasts
        """

        # Functional mask
        # if not op.exists(mask_file):
        #     pyhrf.verbose(1, 'Mask file does not exist. Computing mask from '\
        #                       'BOLD data')
        #     compute_mask_files(bold_files, mask_file, False, 0.4, 0.9)


        #split BOLD into 3D vols:
        bold, hbold = read_volume(bold_file)
        bold_files = []
        tmp_path = tempfile.mkdtemp(dir=pyhrf.cfg['global']['tmp_path'])
        for iscan, bscan in enumerate(bold):
            f = op.join(tmp_path, 'bold_%06d.nii' %iscan)
            write_volume(bscan, f, hbold)
            bold_files.append(f)
        bold_files = ';'.join(bold_files)

        script_path = op.join(op.dirname(pyhrf.__file__),'../../script/SPM')
        spm_path = pyhrf.cfg['global']['spm_path']
        matlab_code = "cd %s;paradigm_file='%s';TR=%f;mask_file='%s';" \
            "bold_files='%s';output_path='%s';" \
            "HF_cut=%f;spm_path='%s';api=1;hack_mask=%d;glm_intra_subj;exit" \
            %(script_path,paradigm_csv_file,tr,mask_file,bold_files,output_dir,
              hf_cut,spm_path,hack_mask)

        matlab_cmd = 'matlab -nosplash -nodesktop -r "%s"'%matlab_code

        if op.exists(op.join(output_dir,'SPM.mat')):
            #remove SPM.mat so that SPM won't ask over ask overwriting
            os.remove(op.join(output_dir,'SPM.mat'))

        #print 'matlab cmd:'
        #print matlab_cmd
        os.system(matlab_cmd)

        # Fix shape of outputs if necessary
        # eg if input data has shape (n,m,1) then SPM will write outputs of
        # shape (n,m) so that they are not consistent with their QForm
        input_shape = bscan.shape
        for foutput in glob.glob(op.join(output_dir, '*.img')):
            data, h = read_volume(foutput)
            if data.ndim < 3:
                sm = ','.join([ [':','np.newaxis'][d==1] \
                                    for d in input_shape ] )
                exec('data = data[%s]' %sm)
                assert data.shape == input_shape
                write_volume(data, foutput, h)

        shutil.rmtree(tmp_path)
        #TODO: maybe find a better way to grab beta file names
        beta_files = sorted(glob.glob(op.join(output_dir,'beta_*.img')))
        return beta_files
