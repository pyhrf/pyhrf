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

from nipy.labs.glm import glm
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
    my_glm = glm.glm()
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
        #sig2 = g.s2 #ResMS
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
