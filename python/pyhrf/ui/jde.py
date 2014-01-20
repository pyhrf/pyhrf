

# -*- coding: utf-8 -*-
import os
import os.path as op
import numpy as np
import copy as copyModule
import time

import pyhrf
from analyser_ui import FMRIAnalyser
from pyhrf.jde.beta import BetaSampler
from pyhrf.jde.nrl.bigaussian import NRLSampler #, NRLSamplerWithRelVar
from pyhrf.jde.models import BOLDGibbsSampler
from pyhrf.xmlio import XmlInitable
from pyhrf.tools.io import read_volume
#from pyhrf.parcellation import parcellation_for_jde

DEFAULT_CFG_FILE = 'detectestim.xml'
DEFAULT_OUTPUT_FILE = 'jde_output.xml'

distribName = 'pyhrf'

class JDEAnalyser(FMRIAnalyser):

    def __init__(self, outputPrefix='jde_', pass_error=True):
        FMRIAnalyser.__init__(self, outputPrefix, pass_error=pass_error)

    # def check_mask(self, roi_mask, bg_label):
    #     roi_ids = np.unique(roi_mask)
    #     # if nb rois == 1 -> simulation
    #     # if nb rois >= 3 -> n-ary mask -> assume it's OK
    #     if len(roi_ids) == 1 or len(roi_ids) > 2:
    #         return True

    #     # if nb rois == 2 -> binary mask -> should be tested
    #     counts = np.bincount(roi_mask[np.where(roi_mask!=bg_label)])
    #     pyhrf.verbose(1, 'Check mask ... biggest parcel is %d. ' \
    #                       'Max parcel size is %d' %(counts.max(),
    #                                                 self.max_parcel_size))
    #     return counts.max() <= self.max_parcel_size

    def get_label(self):
        return 'pyhrf_JDE_analysis'

    # def split_data(self, fmri_data, output_dir=None):

    #     pyhrf.verbose(1, "Handle parcellation ...")
    #     if self.force_input_parcellation:
    #         if fmri_data.data_type == 'volume' and \
    #                 not fmri_data.mask_loaded_from_file:
    #             raise Exception('Use of input parcellation is forced but ' \
    #                                 'none provided')
    #         else:
    #             pyhrf.verbose(1, "Use input parcellation without checking it "\
    #                           "(forced)")
    #             if len(self.roi_ids) > 0:
    #                 pyhrf.verbose(1, 'Analysis limited to some ROIs: %s' \
    #                                   %str(self.roi_ids))

    #                 m0 = fmri_data.roiMask.copy()
    #                 m = np.zeros_like(m0) + fmri_data.backgroundLabel
    #                 for roi_id in self.roi_ids:
    #                     m[np.where(m0==roi_id)] = roi_id
    #             else:
    #                 m = None
    #             return fmri_data.roi_split(m)
    #     else:
    #         if self.check_mask(fmri_data.roiMask, fmri_data.backgroundLabel):
    #             pyhrf.verbose(1, "Input parcellation is OK")
    #             return fmri_data.roi_split()
    #         else:
    #             pyhrf.verbose(1, "Input parcellation is not OK, " \
    #                               "computing it ...")
    #             parcellation,_ = parcellation_for_jde(fmri_data,
    #                                                   self.avg_parcel_size,
    #                                                   output_dir)
    #             if parcellation.min() == -1:
    #                 parcellation += 1

    #             return fmri_data.roi_split(parcellation)


class JDEMCMCAnalyser(JDEAnalyser):

    P_SAMPLER = 'sampler'
    P_OSFMAX = 'osfMax'
    P_DTMIN = 'dtMin'
    P_DT = 'dt'
    P_DRIFT_LFD_PARAM = 'driftParam'
    P_DRIFT_LFD_TYPE = 'driftType'
    P_RANDOM_SEED = 'randomSeed'

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        parametersToShow = [P_DT, P_DTMIN, P_DRIFT_LFD_TYPE, P_DRIFT_LFD_PARAM,
                            P_RANDOM_SEED, P_SAMPLER]
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = [ P_DTMIN, P_DT, P_DRIFT_LFD_TYPE,
                             P_DRIFT_LFD_PARAM, P_SAMPLER]

    parametersComments = {
        P_DTMIN : 'Minimum time resolution for the oversampled estimated signal',
        P_DT : 'If different from 0 or None:\nactual time resolution for the '\
            'oversampled estimated signal (%s is ignored).\n Better when '\
            'it\'s a multiple of the time of repetition' %P_DTMIN,
        P_DRIFT_LFD_PARAM : 'Parameter of the drift modelling.\nIf drift is '\
                             '"polynomial" then this is the order of the '\
                             'polynom.\nIf drift is "cosine" then this is the '\
                             'cut-off period in second.',
        P_DRIFT_LFD_TYPE : 'Either "cosine" or "polynomial" or "None"',
        P_SAMPLER : 'Set of parameters for the sampling scheme',
        }

    def __init__(self, sampler=BOLDGibbsSampler(), osfMax=4, dtMin=.4,
                 dt=.6, driftParam=4, driftType='polynomial',
                 outputFile=DEFAULT_OUTPUT_FILE,outputPrefix='jde_mcmc_',
                 randomSeed=None, pass_error=True):

        XmlInitable.__init__(self)
        JDEAnalyser.__init__(self, outputPrefix, pass_error=pass_error)
        

        self.sampler = copyModule.copy(sampler)
        self.osfMax = osfMax
        self.dtMin = dtMin
        self.dt = dt
        self.driftLfdParam = driftParam
        self.driftLfdType = driftType

    def enable_draft_testing(self):
        self.sampler.set_nb_iterations(3)

    def analyse_roi(self, atomData):
        #print 'atomData:', atomData

        sampler = copyModule.deepcopy(self.sampler)
        sInput = self.packSamplerInput(atomData)
        sampler.linkToData(sInput)
        #if self.parameters[self.P_RANDOM_SEED] is not None:
        #    np.random.seed(self.parameters[self.P_RANDOM_SEED])
        # #HACK:
        # if len(self.roi_ids) > 0:
        #     if atomData.get_roi_id() not in self.roi_ids:
        #         return None

        pyhrf.verbose(1, 'Treating region %d' %(atomData.get_roi_id()))
        tStart = time.time()
        sampler.runSampling()
        pyhrf.verbose(1, 'Cleaning memory ...')
        sampler.dataInput.cleanMem()
        return sampler

    def packSamplerInput(self, roiData):

        try:
            shrf = self.sampler.get_variable('hrf')
        except KeyError:
            shrf = self.sampler.get_variable('brf')
            
        hrfDuration = shrf.duration
        zc = shrf.zc

        simu = None

        if simu != None and shrf.sampleFlag==0:
            hrfDuration = (len(simu.hrf.get_hrf(0,0))-1)*simu.hrf.dt
            pyhrf.verbose(6,'Found simulation data and hrf is '\
                          'not sampled, setting hrfDuration to:' \
                          +str(hrfDuration))

        pyhrf.verbose(2,'building BOLDSamplerInput ...')

        if simu == None or shrf.sampleFlag:
            dt = self.dt if (self.dt!=None and self.dt!=0.) else -self.dtMin
        elif simu != None and shrf.sampleFlag == 0:
            dt = simu.hrf.dt

        samplerInput = self.sampler.inputClass(roiData, dt=dt,
                                               typeLFD=self.driftLfdType,
                                               paramLFD=self.driftLfdParam,
                                               hrfZc=zc,
                                               hrfDuration=hrfDuration)
        return samplerInput


    if 0 :
        def handle_mask(self, mask_file, bold_files, tr,
                        onsets, durations, output_dir, mesh_file=None):

            if mesh_file is None: #Volumic
                if self.force_input_parcellation:
                    if not op.exists(mask_file):
                        raise IOError("Input parcellation is forced but " \
                                          "mask file %s not found" %mask_file)
                    else:
                        #TODO: check if n-ary
                        return mask_file

                FMRIAnalyser.handle_mask(self, mask_file, bold_files, onsets,
                                         durations, mesh_file)

                mask, mask_obj = read_volume(mask_file)
                roi_ids = np.unique(mask)
                if len(roi_ids) <= 2:
                    glm_output_dir = op.join(output_dir, 'GLM')
                    if not op.exists(glm_output_dir):
                        os.makedirs(glm_output_dir)
                    return glm_parcellation(bold_file, tr)

#######################################
# Some functions used in scripts      #
# -> used to parallelize calculation  #
#######################################

from pyhrf.jde.models import BOLDGibbsSampler
from pyhrf.jde.hrf import HRFSampler, RHSampler #, HRFSamplerWithRelVar

def runEstimationBetaEstim(params):

    sessData = params['data']
    nbIterations = params['nbIterations']
    histPace = params.get('histPace',-1)
    sigma = params.get('sigma', 1.5)
    prBetaCut = params.get('prBetaCut', 2.5)
    sampleBetaFlag = params.get('sampleBetaFlag',True)
    pfmethod = params.get('PFMethod','ps')
    gridlnz = params.get('gridLnZ',None)
    sampleHRF = params.get('sampleHRF', True)
    pyhrf.verbose.set_verbosity(params.get('verbose', 1))
    #print 'pfmethod :', pfmethod


    betaSampler = BetaSampler({
            BetaSampler.P_VAL_INI : [0.5],
            BetaSampler.P_SAMPLE_FLAG : 1,
            BetaSampler.P_PARTITION_FUNCTION_METH : pfmethod,
            BetaSampler.P_PARTITION_FUNCTION : gridlnz,
            })

    hrfSampler = HRFSampler({
            HRFSampler.P_SAMPLE_FLAG : sampleHRF,
            })

    #hrfSampler = HRFSamplerWithRelVar({
            #HRFSamplerWithRelVar.P_SAMPLE_FLAG : sampleHRF,
            #})


    hrfVarSampler = RHSampler({
            RHSampler.P_SAMPLE_FLAG : 1,
            })


    #sampler = GGG_BOLDGibbsSampler({
    sampler = BOLDGibbsSampler({
            BOLDGibbsSampler.P_NB_ITERATIONS : nbIterations,
            BOLDGibbsSampler.P_BETA : betaSampler,
            BOLDGibbsSampler.P_HRF : hrfSampler,
            BOLDGibbsSampler.P_RH : hrfVarSampler,
            })

    analyser = JDEAnalyser(sampler=sampler, dt=0.5,
                            outputFile=None)

    result = analyser.analyse(sessData)
    output = analyser.outputResults(result)

    return output


def runEstimationSupervised(params):

    data = params.get('data')
    nbIterations = params.get('nbIterations', 100)
    histPace = params.get('histPace', -1)
    beta = params['betaSpv']

    sampleHRF = params.get('sampleHRF', True)

    pyhrf.verbose.set_verbosity(params.get('verbose', 1))

    prmCAm = params.get('prmCAm',10.)
    prmCAv = params.get('prmCAv',10.)
    prvCIa = params.get('prvCIa',3.)
    prvCIb = params.get('prvCIb',20.)
    prvCAa = params.get('prvCAa',2.4)
    prvCAb = params.get('prvCAb',2.8)

    if histPace == None:
        histPace = nbIterations/10


    betaSampler = BetaSampler({
            BetaSampler.P_SAMPLE_FLAG : 0,
            BetaSampler.P_VAL_INI : np.array([beta]),
            })

    hrfSampler = HRFSampler({
            HRFSampler.P_SAMPLE_FLAG : sampleHRF,
            })

    #hrfSampler = HRFSamplerWithRelVar({
            #HRFSamplerWithRelVar.P_SAMPLE_FLAG : sampleHRF,
            #})

    hrfVarSampler = RHSampler({
            RHSampler.P_SAMPLE_FLAG : 1,
            })



    sampler = BOLDGibbsSampler({
        BOLDGibbsSampler.P_NB_ITERATIONS : nbIterations,
        BOLDGibbsSampler.P_BETA : betaSampler,
        BOLDGibbsSampler.P_RH : hrfVarSampler,
        BOLDGibbsSampler.P_HRF : hrfSampler,
        })

    analyser = JDEAnalyser({
        JDEAnalyser.P_SAMPLER:sampler,
        JDEAnalyser.P_DT:0.5,
        JDEAnalyser.P_OUTPUT_FORMAT:'none',
        })

    result = analyser.analyse(data)
    output = analyser.outputResults(result)
    return output

def jde_analyse(data=None, nbIterations=3,
                hrfModel='estimated', hrfNorm=1., hrfTrick=False,
                sampleHrfVar=True,
                hrfVar=1e-5,keepSamples=False,samplesHistPace=1):
    """
    """
    if data is None:
        data = pyhrf.data.get_default_data()
        data.keep_only_rois([2])

    if hrfModel == 'estimated':
        sampleHRF = True
    elif hrfModel == 'canonical':
        sampleHRF = False
    else:
        raise Exception('Unknown hrf model %s' %hrfModel)

    hrfSampler = HRFSampler({
            HRFSampler.P_SAMPLE_FLAG : sampleHRF,
            HRFSampler.P_NORMALISE : hrfNorm,
            HRFSampler.P_TRICK : hrfTrick,
            #HRFSampler.P_KEEP_SAMPLES : keepSamples,
            #HRFSampler.P_SAMPLE_HIST_PACE : samplesHistPace,
            })

    #hrfSampler = HRFSamplerWithRelVar({
            #HRFSamplerWithRelVar.P_SAMPLE_FLAG : sampleHRF,
            #HRFSamplerWithRelVar.P_NORMALISE : hrfNorm,
            #HRFSamplerWithRelVar.P_TRICK : hrfTrick,
            ##HRFSamplerWithRelVar.P_KEEP_SAMPLES : keepSamples,
            ##HRFSamplerWithRelVar.P_SAMPLE_HIST_PACE : samplesHistPace,
            #})

    hrfVarSampler = RHSampler({
            RHSampler.P_SAMPLE_FLAG : sampleHrfVar,
            #RHSampler.P_KEEP_SAMPLES : keepSamples,
            #RHSampler.P_SAMPLE_HIST_PACE : samplesHistPace,
            })

    nrlSampler = NRLSampler({
            NRLSampler.P_KEEP_SAMPLES : keepSamples,
            NRLSampler.P_SAMPLE_HIST_PACE : samplesHistPace,
            })

    #nrlSampler = NRLSamplerWithRelVar({
            #NRLSamplerWithRelVar.P_KEEP_SAMPLES : keepSamples,
            #NRLSamplerWithRelVar.P_SAMPLE_HIST_PACE : samplesHistPace,
            #})


    sampler = BOLDGibbsSampler({
        BOLDGibbsSampler.P_NB_ITERATIONS : nbIterations,
        #BOLDGibbsSampler.P_BETA : betaSampler,
        BOLDGibbsSampler.P_RH : hrfVarSampler,
        BOLDGibbsSampler.P_HRF : hrfSampler,
        BOLDGibbsSampler.P_NRLS : nrlSampler,
        })

    analyser = JDEAnalyser({
            JDEAnalyser.P_SAMPLER:sampler,
            JDEAnalyser.P_OUTPUT_FILE:None,})
    result = analyser.analyse(data)
    output = analyser.outputResults(result)

    return output


