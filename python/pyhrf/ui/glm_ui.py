# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import numpy as np


import pyhrf
from pyhrf import xmlio
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.backport import OrderedDict
from pyhrf.ndarray import xndarray, tree_to_cuboid, stack_cuboids

from pyhrf.ui.analyser_ui import FMRIAnalyser
from pyhrf.glm import glm_nipy

class GLMAnalyser(FMRIAnalyser):

    parametersToShow = []

    parametersComments = {
        'fit_method' : 'Either "ols" or "kalman"',
        'residuals_model' : 'Either "spherical" or "ar1". If "ar1" then '\
            'the kalman fit method is used'
        }

    def __init__(self, contrasts={'dummy_contrast_example':'3*audio-video/3'},
                 contrast_test_baseline=0.0,
                 hrf_model='Canonical', drift_model='Cosine', hfcut=128.,
                 residuals_model='spherical',fit_method='ols',
                 outputPrefix='glm_', rescale_results=False,
                 rescale_factor_file=None, fir_delays=[0]):

        xmlio.XmlInitable.__init__(self)
        FMRIAnalyser.__init__(self, outputPrefix)

        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.fir_delays = fir_delays
        self.hfcut = hfcut
        self.residuals_model = residuals_model
        self.fit_method = fit_method
        self.contrasts = contrasts
        self.contrasts.pop('dummy_contrast_example',None)
        self.con_bl = contrast_test_baseline
        self.rescale_results = rescale_results

        if rescale_factor_file is not None:
            self.rescale_factor = xndarray.load(rescale_factor_file).data
        else:
            self.rescale_factor = None

    def get_label(self):
        return 'pyhrf_GLM_analysis'

    def analyse_roi(self, fdata):
        pyhrf.verbose(1, 'Run GLM analysis (ROI %d) ...' %fdata.get_roi_id())

        if self.rescale_factor is not None:
            m = np.where(fdata.roiMask)
            rescale_factor = self.rescale_factor[:,m[0],m[1],m[2]]
        else:
            rescale_factor = None

        glm, dm, cons = glm_nipy(fdata, contrasts=self.contrasts,
                                 hrf_model=self.hrf_model,
                                 drift_model=self.drift_model,
                                 hfcut=self.hfcut,
                                 residuals_model=self.residuals_model,
                                 fit_method=self.fit_method,
                                 fir_delays=self.fir_delays,
                                 rescale_results=self.rescale_results,
                                 rescale_factor=rescale_factor)

        outputs = {}

        ns, nr = dm.matrix.shape
        tr = fdata.tr
        if rescale_factor is not None:
            #same sf for all voxels
            dm.matrix[:,:rescale_factor.shape[0]] /= rescale_factor[:,0]

        cdesign_matrix = xndarray(dm.matrix,
                                axes_names=['time','regressor'],
                                axes_domains={'time':np.arange(ns)*tr,
                                              'regressor':dm.names})
        outputs['design_matrix'] = cdesign_matrix

        axes_names = ['time', 'voxel']
        axes_domains = {'time' : np.arange(ns)*tr}
        bold = xndarray(fdata.bold.astype(np.float32),
                      axes_names=axes_names,
                      axes_domains=axes_domains,
                      value_label='BOLD')

        fit = np.dot(dm.matrix, glm.beta)
        cfit = xndarray(fit, axes_names=['time','voxel'],
                      axes_domains={'time':np.arange(ns)*tr})

        outputs['bold_fit'] = stack_cuboids([bold,cfit], 'stype', ['bold', 'fit'])


        nb_cond = fdata.nbConditions
        fit_cond = np.dot(dm.matrix[:,:nb_cond], glm.beta[:nb_cond,:])
        fit_cond -= fit_cond.mean(0)
        fit_cond += fdata.bold.mean(0)

        outputs['fit_cond'] = xndarray(fit_cond, axes_names=['time','voxel'],
                                     axes_domains={'time':np.arange(ns)*tr})


        outputs['s2'] = xndarray(glm.s2, axes_names=['voxel'])


        if 0:
            cbeta = xndarray(glm.beta, axes_names=['reg_name','voxel'],
                           axes_domains={'reg_name':dm.names})

            outputs['beta'] = cbeta
        else:
            if self.hrf_model == 'FIR':
                fir = dict((d * fdata.tr, OrderedDict()) for d in self.fir_delays)
            for ib, bname in enumerate(dm.names):
                outputs['beta_' + bname] = xndarray(glm.beta[ib],
                                                  axes_names=['voxel'])
                if self.hrf_model == 'FIR' and 'delay' in bname:
                    #reconstruct filter:
                    cond, delay = bname.split('_delay_')
                    delay = int(delay) * fdata.tr
                    fir[delay][cond] = xndarray(glm.beta[ib], axes_names=['voxel'])

            if self.hrf_model == 'FIR':
                chrf = tree_to_cuboid(fir, ['time', 'condition'])
                outputs['hrf'] = chrf
                outputs['hrf_norm'] = (chrf**2).sum('time')**.5

            for cname, con in cons.iteritems():
                #print 'con:'
                #print dir(con)
                outputs['con_effect_'+cname] = xndarray(con.effect,
                                                      axes_names=['voxel'])

                #print '%%%%%%% con.variance:', con.variance.shape
                ncon = con.effect / con.variance.std()
                outputs['ncon_effect_'+cname] = xndarray(ncon, axes_names=['voxel'])

                outputs['con_pvalue_'+cname] = xndarray(con.pvalue(self.con_bl),
                                                      axes_names=['voxel'])


        roi_lab_vol = np.zeros(fdata.get_nb_vox_in_mask(), dtype=np.int32) + \
            fdata.get_roi_id()

        outputs['mask'] = xndarray(roi_lab_vol, axes_names=['voxel'])

        # for ib, bname in enumerate(design_matrix.names):
        #     beta_vol = expand_array_in_mask(my_glm.beta[ib], mask_array)
        #     beta_image = Nifti1Image(beta_vol, affine)
        #     beta_file = op.join(output_dir, 'beta_%s.nii' %bname)
        #     save(beta_image, beta_file)
        #     beta_files.append(beta_file)


        return outputs
