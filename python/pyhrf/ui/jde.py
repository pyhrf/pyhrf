# -*- coding: utf-8 -*-

import os
import os.path as op
import copy as copyModule
import logging

import numpy as np

import pyhrf

from pyhrf.ui.analyser_ui import FMRIAnalyser
from pyhrf.jde.beta import BetaSampler
from pyhrf.jde.nrl.bigaussian import NRLSampler  # , NRLSamplerWithRelVar
from pyhrf.jde.models import BOLDGibbsSampler
from pyhrf.xmlio import XmlInitable
from pyhrf.tools._io import read_volume


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_CFG_FILE = 'detectestim.xml'

distribName = 'pyhrf'


class JDEAnalyser(FMRIAnalyser):

    def __init__(self, outputPrefix='jde_', pass_error=True):
        FMRIAnalyser.__init__(self, outputPrefix, pass_error=pass_error)

    def get_label(self):
        return 'pyhrf_JDE_analysis'


class JDEMCMCAnalyser(JDEAnalyser):
    """
    Class that wraps a JDE Gibbs Sampler to launch an fMRI analysis
    TODO: remove parameters about dt and osf (should go in HRF Sampler class),
    drift (should go in Drift Sampler class)
    """
    P_SAMPLER = 'sampler'
    P_OSFMAX = 'osfMax'         # over-sampling factor
    P_DTMIN = 'dtMin'
    P_DT = 'dt'
    P_DRIFT_LFD_PARAM = 'driftParam'
    P_DRIFT_LFD_TYPE = 'driftType'
    P_RANDOM_SEED = 'randomSeed'

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        parametersToShow = [P_DT, P_DTMIN, P_DRIFT_LFD_TYPE, P_DRIFT_LFD_PARAM,
                            P_RANDOM_SEED, P_SAMPLER]
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = [P_DTMIN, P_DT, P_DRIFT_LFD_TYPE,
                            P_DRIFT_LFD_PARAM, P_SAMPLER]

    parametersComments = {
        P_DTMIN: 'Minimum time resolution for the oversampled estimated signal',
        P_DT: 'If different from 0 or None:\nactual time resolution for the '
        'oversampled estimated signal (%s is ignored).\n Better when '
        'it\'s a multiple of the time of repetition' % P_DTMIN,
        P_DRIFT_LFD_PARAM: 'Parameter of the drift modelling.\nIf drift is '
        '"polynomial" then this is the order of the '
        'polynom.\nIf drift is "cosine" then this is the '
        'cut-off period in second.',
        P_DRIFT_LFD_TYPE: 'Either "cosine" or "polynomial" or "None"',
        P_SAMPLER: 'Set of parameters for the sampling scheme',
    }

    def __init__(self, sampler=BOLDGibbsSampler(), osfMax=4, dtMin=.4,
                 dt=.6, driftParam=4, driftType='polynomial',
                 outputPrefix='jde_mcmc_', randomSeed=None, pass_error=True,
                 copy_sampler=True):

        XmlInitable.__init__(self)
        JDEAnalyser.__init__(self, outputPrefix, pass_error=pass_error)

        self.sampler = sampler
        self.osfMax = osfMax
        self.dtMin = dtMin
        self.dt = dt
        self.driftLfdParam = driftParam
        self.driftLfdType = driftType
        self.copy_sampler = copy_sampler

    def enable_draft_testing(self):
        self.sampler.set_nb_iterations(3)

    def analyse_roi(self, atomData):
        """
        Launch the JDE Gibbs Sampler on a parcel-specific data set *atomData*
        Args:
            - atomData (pyhrf.core.FmriData): parcel-specific data
        Returns:
            JDE sampler object
        """

        if self.copy_sampler:
            sampler = copyModule.deepcopy(self.sampler)
        else:
            sampler = self.sampler
        sInput = self.packSamplerInput(atomData)
        sampler.linkToData(sInput)

        logger.info('Treating region %d', atomData.get_roi_id())
        sampler.runSampling(atomData)
        logger.info('Cleaning memory ...')
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

        if simu != None and shrf.sampleFlag == 0:
            hrfDuration = (len(simu.hrf.get_hrf(0, 0)) - 1) * simu.hrf.dt
            logger.debug('Found simulation data and hrf is not sampled, '
                         'setting hrfDuration to: %s', str(hrfDuration))

        logger.info('building BOLDSamplerInput ...')

        if simu == None or shrf.sampleFlag:
            dt = self.dt if (
                self.dt != None and self.dt != 0.) else -self.dtMin
        elif simu != None and shrf.sampleFlag == 0:
            dt = simu.hrf.dt

        samplerInput = self.sampler.inputClass(roiData, dt=dt,
                                               typeLFD=self.driftLfdType,
                                               paramLFD=self.driftLfdParam,
                                               hrfZc=zc,
                                               hrfDuration=hrfDuration)

        return samplerInput

    if 0:
        def handle_mask(self, mask_file, bold_files, tr,
                        onsets, durations, output_dir, mesh_file=None):

            if mesh_file is None:  # Volumic
                if self.force_input_parcellation:
                    if not op.exists(mask_file):
                        raise IOError("Input parcellation is forced but "
                                      "mask file %s not found" % mask_file)
                    else:
                        # TODO: check if n-ary
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
from pyhrf.jde.hrf import HRFSampler, RHSampler  # , HRFSamplerWithRelVar


def runEstimationBetaEstim(params):

    sessData = params['data']
    nbIterations = params['nbIterations']
    histPace = params.get('histPace', -1)
    sigma = params.get('sigma', 1.5)
    prBetaCut = params.get('prBetaCut', 2.5)
    sampleBetaFlag = params.get('sampleBetaFlag', True)
    pfmethod = params.get('PFMethod', 'ps')
    gridlnz = params.get('gridLnZ', None)
    sampleHRF = params.get('sampleHRF', True)
    # pyhrf.verbose.set_verbosity(params.get('verbose', 1))
    # pyhrf.logger.setLevel(params.get('verbose', logging.WARNING))

    betaSampler = BetaSampler({
        BetaSampler.P_VAL_INI: [0.5],
        BetaSampler.P_SAMPLE_FLAG: 1,
        BetaSampler.P_PARTITION_FUNCTION_METH: pfmethod,
        BetaSampler.P_PARTITION_FUNCTION: gridlnz,
    })

    hrfSampler = HRFSampler({
        HRFSampler.P_SAMPLE_FLAG: sampleHRF,
    })

    hrfVarSampler = RHSampler({
        RHSampler.P_SAMPLE_FLAG: 1,
    })

    sampler = BOLDGibbsSampler({
        BOLDGibbsSampler.P_NB_ITERATIONS: nbIterations,
        BOLDGibbsSampler.P_BETA: betaSampler,
        BOLDGibbsSampler.P_HRF: hrfSampler,
        BOLDGibbsSampler.P_RH: hrfVarSampler,
    })

    analyser = JDEAnalyser(sampler=sampler, dt=0.5)

    result = analyser.analyse(sessData)
    output = analyser.outputResults(result, outputResults)

    return output


def runEstimationSupervised(params):

    data = params.get('data')
    nbIterations = params.get('nbIterations', 100)
    histPace = params.get('histPace', -1)
    beta = params['betaSpv']

    sampleHRF = params.get('sampleHRF', True)

    # pyhrf.verbose.set_verbosity(params.get('verbose', 1))
    # pyhrf.logger.setLevel(params.get('verbose', logging.WARNING))

    prmCAm = params.get('prmCAm', 10.)
    prmCAv = params.get('prmCAv', 10.)
    prvCIa = params.get('prvCIa', 3.)
    prvCIb = params.get('prvCIb', 20.)
    prvCAa = params.get('prvCAa', 2.4)
    prvCAb = params.get('prvCAb', 2.8)

    if histPace == None:
        histPace = nbIterations / 10

    betaSampler = BetaSampler({
        BetaSampler.P_SAMPLE_FLAG: 0,
        BetaSampler.P_VAL_INI: np.array([beta]),
    })

    hrfSampler = HRFSampler({
        HRFSampler.P_SAMPLE_FLAG: sampleHRF,
    })

    hrfVarSampler = RHSampler({
        RHSampler.P_SAMPLE_FLAG: 1,
    })

    sampler = BOLDGibbsSampler({
        BOLDGibbsSampler.P_NB_ITERATIONS: nbIterations,
        BOLDGibbsSampler.P_BETA: betaSampler,
        BOLDGibbsSampler.P_RH: hrfVarSampler,
        BOLDGibbsSampler.P_HRF: hrfSampler,
    })

    analyser = JDEAnalyser({
        JDEAnalyser.P_SAMPLER: sampler,
        JDEAnalyser.P_DT: 0.5,
        JDEAnalyser.P_OUTPUT_FORMAT: 'none',
    })

    result = analyser.analyse(data)
    output = analyser.outputResults(result)
    return output


def jde_analyse(data=None, nbIterations=3,
                hrfModel='estimated', hrfNorm=1., hrfTrick=False,
                sampleHrfVar=True,
                hrfVar=1e-5, keepSamples=False, samplesHistPace=1):
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
        raise Exception('Unknown hrf model %s' % hrfModel)

    hrfSampler = HRFSampler({
        HRFSampler.P_SAMPLE_FLAG: sampleHRF,
        HRFSampler.P_NORMALISE: hrfNorm,
        HRFSampler.P_TRICK: hrfTrick,
    })

    hrfVarSampler = RHSampler({
        RHSampler.P_SAMPLE_FLAG: sampleHrfVar,
    })

    nrlSampler = NRLSampler({
        NRLSampler.P_KEEP_SAMPLES: keepSamples,
        NRLSampler.P_SAMPLE_HIST_PACE: samplesHistPace,
    })

    sampler = BOLDGibbsSampler({
        BOLDGibbsSampler.P_NB_ITERATIONS: nbIterations,
        BOLDGibbsSampler.P_RH: hrfVarSampler,
        BOLDGibbsSampler.P_HRF: hrfSampler,
        BOLDGibbsSampler.P_NRLS: nrlSampler,
    })

    analyser = JDEAnalyser({
        JDEAnalyser.P_SAMPLER: sampler,
        JDEAnalyser.P_OUTPUT_FILE: None, })
    result = analyser.analyse(data)
    output = analyser.outputResults(result)

    return output
