# -*- coding: utf-8 -*-
import numpy as np
from pyhrf.ndarray import xndarray

from pyhrf.ui.analyser_ui import FMRIAnalyser
from pyhrf.glm import glm_nipy

class GLMAnalyser(FMRIAnalyser):

    def __init__(self, outputPrefix='glm_'):
        FMRIAnalyser.__init__(self, outputPrefix)


    def get_label(self):
        return 'pyhrf_GLM_analysis'

    def analyse_roi(self, fdata):
        glm = glm_nipy(fdata, contrasts=None, hrf_model='Canonical',
                        drift_model='Cosine', hfcut=128,
                        residuals_model='spherical', fit_method='ols',
                        fir_duration=None, fir_delays=None)

        outputs = {}
        ns = fdata.shape[0]
        dm = glm.design_matrix.matrix
        tr = fdata.tr
        cdesign_matrix = xndarray(dm,
                                axes_names=['time','regressor'],
                                axes_domains={'time':np.arange(ns)*tr,
                                              'regressor':dm.names})

        outputs['design_matrix'] = cdesign_matrix

        return outputs
