# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import randn, rand, permutation
#from numpy import *
import copy as copyModule

from pyhrf.jde.samplerbase import *
import pyhrf
from pyhrf import xmlio
from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
from pyhrf.jde.intensivecalc import sampleSmmNrl2, computeYtilde, sampleSmmNrlBar
from bigaussian import NRLSampler, BiGaussMixtureParamsSampler, NRLSamplerWithRelVar
from pyhrf.jde.intensivecalc import sampleSmmNrl2WithRelVar
#, sampleSmmNrl2WithRelVar_NEW


class NRL_Drift_Sampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling.
    """
    defaultParameters = copyModule.copy(NRLSampler.defaultParameters)

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        NRLSampler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)


    def computeVarYTildeOpt(self, varXh):
        NRLSampler.computeVarYTildeOpt(self, varXh)
        matPl = self.samplerEngine.getVariable('drift').matPl
        self.varYbar = self.varYtilde - matPl


    def sampleNextInternal(self, variables):
        #TODO : comment
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        sHrf = variables[self.samplerEngine.I_HRF]
        h = sHrf.currentValue
        sDrift = variables[self.samplerEngine.I_DRIFT]
        varXh = sHrf.varXh
        varLambda = variables[self.samplerEngine.I_WEIGHTING_PROBA].currentValue

        pyhrf.verbose(5,'varXh %s :' %str(varXh.shape))
        pyhrf.verbose.printNdarray(5, varXh)

        self.computeVarYTildeOpt(varXh)

        self.labelsSamples = rand(self.nbConditions, self.nbVox)
        self.nrlsSamples   = randn(self.nbConditions, self.nbVox)

        gTg = np.diag(np.dot(varXh.transpose(),varXh))

        if self.imm:
            #self.sampleNrlsParallel(varXh, rb, h, varLambda, varCI,
            #                        varCA, meanCA, gTQg, variables)
            raise NotImplementedError("IMM with drift sampling is not available")
        else: #smm
            self.sampleNrlsSerial(rb, h, varCI, varCA, meanCA, gTg, variables)
            self.computeVarYTildeOpt(varXh)

        if (self.currentValue >= 1000).any() and \
                pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )
            #pyhrf.verbose.setVerbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.computeAA(self.currentValue, self.aa)

        self.printState(4)
        self.iteration += 1 #TODO : factorize !!

    def sampleNrlsSerial(self, rb, h, varCI, varCA, meanCA ,
                         gTg, variables):

        pyhrf.verbose(3, 'Sampling Nrls (serial, spatial prior) ...')
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue

        # Add one dimension to be consistent with habituation model
        varXh = variables[self.samplerEngine.I_HRF].varXh
        varXht = varXh.transpose()
        nrls = self.currentValue

        neighbours = self.dataInput.neighboursIndexes

        beta = self.samplerEngine.getVariable('beta').currentValue
        voxOrder = permutation(self.nbVox)

        sampleSmmNrl2(voxOrder.astype(np.int32), rb.astype(np.float64),
                      neighbours.astype(np.int32), self.varYbar,
                      self.labels, np.array([varXh], dtype=np.float64),
                      self.currentValue,
                      self.nrlsSamples.astype(np.float64),
                      self.labelsSamples.astype(np.float64),
                      np.array([varXht], dtype=np.float64),
                      gTg.astype(np.float64),
                      beta.astype(np.float64), mean.astype(np.float64),
                      var.astype(np.float64), self.meanClassApost,
                      self.varClassApost, self.nbClasses,
                      self.sampleLabelsFlag+0, self.iteration,
                      self.nbConditions)

        self.countLabels(self.labels, self.voxIdx, self.cardClass)

class NRL_Drift_SamplerWithRelVar(NRLSamplerWithRelVar):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling and relevant variable.
    """
    defaultParameters = copyModule.copy(NRLSamplerWithRelVar.defaultParameters)

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        NRLSamplerWithRelVar.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)


    def computeVarYTildeOptWithRelVar(self, varXh, w):
        NRLSamplerWithRelVar.computeVarYTildeOptWithRelVar(self, varXh, w)
        matPl = self.samplerEngine.getVariable('drift').matPl
        self.varYbar = self.varYtilde - matPl


    def sampleNextInternal(self, variables):
        
        #print 'Step 1 : NRLs and Labels Sampling *****RelVar*****'
        
        #TODO : comment
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        sHrf = variables[self.samplerEngine.I_HRF]
        h = sHrf.currentValue
        sDrift = variables[self.samplerEngine.I_DRIFT]
        varXh = sHrf.varXh
        varLambda = variables[self.samplerEngine.I_WEIGHTING_PROBA].currentValue
        w = variables[self.samplerEngine.I_W].currentValue

        t1 = variables[self.samplerEngine.I_W].t1
        t2 = variables[self.samplerEngine.I_W].t2
        
        pyhrf.verbose(5,'varXh %s :' %str(varXh.shape))
        pyhrf.verbose.printNdarray(5, varXh)

        self.computeVarYTildeOptWithRelVar(varXh, w)

        self.labelsSamples = rand(self.nbConditions, self.nbVox)
        self.nrlsSamples   = randn(self.nbConditions, self.nbVox)

        gTg = np.diag(np.dot(varXh.transpose(),varXh))
        
        if self.imm:
            #self.sampleNrlsParallel(varXh, rb, h, varLambda, varCI,
            #                        varCA, meanCA, gTQg, variables)
            raise NotImplementedError("IMM with drift sampling is not available")
        else: #smm
            self.sampleNrlsSerialWithRelVar(rb, h, gTg, variables, w, t1, t2)
            self.computeVarYTildeOptWithRelVar(varXh, w)

        if (self.currentValue >= 1000).any() and \
                pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )
            #pyhrf.verbose.setVerbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.computeAA(self.currentValue, self.aa)

        wa = np.zeros((self.nbConditions, self.nbVox))
        self.computeWA(self.currentValue, w, wa)
        self.computeSumWAxh(wa, varXh)

        self.printState(4)
        self.iteration += 1 #TODO : factorize !!

    def sampleNrlsSerialWithRelVar(self, rb, h, gTg, variables, w, t1, t2):

        pyhrf.verbose(3, 'Sampling Nrls (serial, spatial prior) ...')
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        y = self.dataInput.varMBY
        matPl = self.samplerEngine.getVariable('drift').matPl
        sampleWFlag = variables[self.samplerEngine.I_W].sampleFlag

        # Add one dimension to be consistent with habituation model
        varXh = variables[self.samplerEngine.I_HRF].varXh
        varXht = varXh.transpose()
        nrls = self.currentValue

        neighbours = self.dataInput.neighboursIndexes

        beta = self.samplerEngine.getVariable('beta').currentValue
        voxOrder = permutation(self.nbVox)


        cardClassCA = np.zeros(self.nbConditions, dtype=int)
        for i in range(self.nbConditions):
            cardClassCA[i] = self.cardClass[self.L_CA,i]
        
        #sampleSmmNrl2WithRelVar(voxOrder.astype(np.int32), rb.astype(np.float64),
                      #neighbours.astype(np.int32), self.varYbar,
                      #self.labels, np.array([varXh], dtype=np.float64),
                      #self.currentValue,
                      #self.nrlsSamples.astype(np.float64),
                      #self.labelsSamples.astype(np.float64),
                      #np.array([varXht], dtype=np.float64),
                      #gTg.astype(np.float64),
                      #beta.astype(np.float64), mean.astype(np.float64),
                      #var.astype(np.float64), self.meanClassApost,
                      #self.varClassApost, w.astype(np.int32), t1, t2,
                      #cardClassCA.astype(np.int32),
                      #self.nbClasses,
                      #self.sampleLabelsFlag+0, self.iteration,
                      #self.nbConditions)
        
        sampleSmmNrl2WithRelVar_NEW(voxOrder.astype(np.int32), rb.astype(np.float64),
                      neighbours.astype(np.int32), self.varYbar , y.astype(np.float64), matPl.astype(np.float64),
                      self.labels, np.array([varXh], dtype=np.float64),
                      self.currentValue,
                      self.nrlsSamples.astype(np.float64),
                      self.labelsSamples.astype(np.float64),
                      np.array([varXht], dtype=np.float64),
                      gTg.astype(np.float64),
                      beta.astype(np.float64), mean.astype(np.float64),
                      var.astype(np.float64), self.meanClassApost,
                      self.varClassApost, w.astype(np.int32), t1, t2,
                      cardClassCA.astype(np.int32),
                      self.nbClasses,
                      self.sampleLabelsFlag+0, self.iteration,
                      self.nbConditions,sampleWFlag)

        self.countLabels(self.labels, self.voxIdx, self.cardClass)


class NRLsBar_Drift_Multi_Sess_Sampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling.
    """
    defaultParameters = copyModule.copy(NRLSampler.defaultParameters)

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        NRLSampler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbSessions = self.dataInput.nbSessions
        self.cardClass = np.zeros((self.nbClasses, self.nbConditions), dtype=int)
        self.voxIdx = [range(self.nbConditions) for c in xrange(self.nbClasses)]

        #print dataInput.simulData
        #TODO handle condition matching
        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, dict):
                if dataInput.simulData.has_key('nrls'):
                    nrls = dataInput.simulData['nrls']
                    if isinstance(nrls, xndarray):
                        self.trueValue = nrls.reorient(['condition','voxel']).data
                    else:
                        self.trueValue = nrls
                if dataInput.simulData.has_key('labels'):
                    labels = dataInput.simulData['labels']
                    if isinstance(labels, xndarray):
                        self.trueLabels = labels.reorient(['condition','voxel']).data
                    else:
                        self.trueLabels = labels
            else:
                sd = dataInput.simulData[0]
                self.trueValue = sd['nrls'].astype(np.float64)
                self.trueLabels = sd['labels'].astype(np.int32)

            self.trueLabels = self.trueLabels[:self.nbConditions,:].astype(np.int32)
            self.trueValue = self.trueValue[:self.nbConditions,:].astype(np.float64)
        else:
            self.trueLabels = None

            
    def checkAndSetInitValue(self, variables):
        NRLSampler.checkAndSetInitLabels(self,variables)
        NRLSampler.checkAndSetInitNRL(self,variables)
        mixt_par = variables[self.samplerEngine.I_MIXT_PARAM_NRLS_BAR]
        mixt_par.checkAndSetInitValue(variables)
        weights_par = variables[self.samplerEngine.I_WEIGHTING_PROBA_NRLS_BAR]
        weights_par.checkAndSetInitValue(variables)

    def sampleNextAlt(self, variables):
        pass

    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """

        # Precalculations and allocations :


        self.imm = self.samplerEngine.getVariable('beta').currentValue[0] < 0

        self.varClassApost = np.zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                dtype=np.float64)
        self.sigClassApost = np.zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                dtype=float)
        self.meanClassApost = np.zeros((self.nbClasses,self.nbConditions,
                                    self.nbVox), dtype=np.float64)
        self.meanApost = np.zeros((self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = np.zeros((self.nbConditions, self.nbVox), dtype=float)

        self.iteration = 0

        self.countLabels(self.labels, self.voxIdx, self.cardClass)


    def sampleNextInternal(self, variables):
        #TODO : comment
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM_NRLS_BAR]
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]

        self.labelsSamples = rand(self.nbConditions, self.nbVox)
        self.nrlsSamples = randn(self.nbConditions, self.nbVox)

        if self.imm:
            #self.sampleNrlsParallel(varXh, rb, h, varLambda, varCI,
            #                        varCA, meanCA, gTQg, variables)
            raise NotImplementedError("IMM with drift sampling is not available")
        else: #smm
            self.sampleNrlsSerial(varCI, varCA, meanCA, variables)
            #self.computeVarYTildeOpt(varXh)
            #matPl = self.samplerEngine.getVariable('drift').matPl
            #self.varYbar = self.varYtilde - matPl

        if (self.currentValue >= 1000).any() and \
                pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )
            #pyhrf.verbose.setVerbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.printState(4)
        self.iteration += 1 #TODO : factorize !!

    def sampleNrlsSerial(self, varCI, varCA, meanCA ,
                         variables):

        pyhrf.verbose(3, 'Sampling Nrls (serial, spatial prior) ...')
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM_NRLS_BAR ]
        nrl_var_sess = self.samplerEngine.getVariable('variance_nrls_by_session').currentValue
        sum_nrl_sess = self.samplerEngine.getVariable('nrl_by_session').currentValue.sum(0)

        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()

        neighbours = self.dataInput.neighboursIndexes

        beta = self.samplerEngine.getVariable('beta').currentValue
        voxOrder = permutation(self.nbVox)
        #print 'voxorder:', voxOrder
        #print 'badam!!'
        #print 'type var sess:', type(nrl_var_sess)
        #print nrl_var_sess

        pyhrf.verbose(3, 'Labels sampling: %s' %str(self.sampleLabelsFlag))

        sampleSmmNrlBar(voxOrder.astype(np.int32),
                        neighbours.astype(np.int32),
                        self.labels, self.currentValue,
                        self.nrlsSamples.astype(np.float64),
                        self.labelsSamples.astype(np.float64),
                        beta.astype(np.float64), mean.astype(np.float64),
                        var.astype(np.float64), self.meanClassApost,
                        self.varClassApost, nrl_var_sess, sum_nrl_sess,
                        self.nbClasses,
                        self.sampleLabelsFlag+0, self.iteration,
                        self.nbConditions, self.nbSessions)



        self.countLabels(self.labels, self.voxIdx, self.cardClass)


## Multi-sessions case

class BiGaussMixtureParams_Multi_Sess_NRLsBar_Sampler(xmlio.XMLParamDrivenClass,
                                  GibbsSamplerVariable):
    """
    #TODO : comment

    """

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_USE_TRUE_VALUE = 'useTrueValue'
    #P_ACT_MEAN_TRUE_VALUE = 'ActMeanTrueValue'
    #P_ACT_VAR_TRUE_VALUE = 'ActVarTrueValue'
    #P_INACT_VAR_TRUE_VALUE = 'InactVarTrueValue'

    P_MEAN_CA_PR_MEAN = 'meanCAPrMean'
    P_MEAN_CA_PR_VAR = 'meanCAPrVar'

    P_VAR_CI_PR_ALPHA = 'varCIPrAlpha'
    P_VAR_CI_PR_BETA = 'varCIPrBeta'

    P_VAR_CA_PR_ALPHA = 'varCAPrAlpha'
    P_VAR_CA_PR_BETA = 'varCAPrBeta'

    P_HYPER_PRIOR = 'hyperPriorType'

    P_ACTIV_THRESH = 'mean_activation_threshold'

    #"peaked" priors
    defaultParameters = {
        P_VAL_INI : None,
        P_SAMPLE_FLAG : True,
        P_USE_TRUE_VALUE : False,
        P_HYPER_PRIOR : 'Jeffrey',
        #P_HYPER_PRIOR : 'proper',
        P_MEAN_CA_PR_MEAN : 5.,
        P_MEAN_CA_PR_VAR : 20.0,
        P_VAR_CI_PR_ALPHA : 2.04,
        P_VAR_CI_PR_BETA : 2.08,
        P_VAR_CA_PR_ALPHA : 2.01,
        P_VAR_CA_PR_BETA : .5,
        P_ACTIV_THRESH : 4.,
        #P_ACT_MEAN_TRUE_VALUE : { 'audio': 0.0, 'video': 0.0 },
        #P_ACT_VAR_TRUE_VALUE : { 'audio': 1.0, 'video': 1.0 },
        #P_INACT_VAR_TRUE_VALUE : { 'audio': 1.0, 'video': 1.0 },
        }



    ##"flat" priors
    #defaultParameters = {
        #P_VAL_INI : None,
        #P_SAMPLE_FLAG : True,
        #P_USE_TRUE_VALUE : False,
        ##P_HYPER_PRIOR : 'Jeffrey',
        #P_HYPER_PRIOR : 'proper',
        #P_SAMPLE_FLAG : 1,
        #P_MEAN_CA_PR_MEAN : 10.,
        #P_MEAN_CA_PR_VAR : 100.0,
        #P_VAR_CI_PR_ALPHA : 2.04,
        #P_VAR_CI_PR_BETA : 2.08,
        #P_VAR_CA_PR_ALPHA : 2.001,
        #P_VAR_CA_PR_BETA : 1.01,
        #}


    # a=2.5, b=0.5 => m=0.5/(2.5-1)=1/3 & v=0.5**2/((2.5-1)**2*(2.5-2))=0.2
    # m=b/(a-1) , v=b**2/((a-1)**2*(a-2)
    # a=m**2/v +2 , b=m**3/v + m

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = []


    L_CA = NRLSampler.L_CA
    L_CI = NRLSampler.L_CI

    parametersToShow = [ P_VAL_INI, P_SAMPLE_FLAG, P_ACTIV_THRESH,
                         P_USE_TRUE_VALUE,
                         #P_ACT_MEAN_TRUE_VALUE, P_ACT_VAR_TRUE_VALUE, P_INACT_VAR_TRUE_VALUE,
                         P_HYPER_PRIOR,
                         P_MEAN_CA_PR_MEAN, P_MEAN_CA_PR_VAR, P_VAR_CI_PR_ALPHA,
                         P_VAR_CI_PR_BETA, P_VAR_CA_PR_ALPHA, P_VAR_CA_PR_BETA]

    parametersComments = {
        P_HYPER_PRIOR : "Either 'proper' or 'Jeffrey'",
        P_ACTIV_THRESH : "Threshold for the max activ mean above which the "\
            "region is considered activating",
        #P_ACT_MEAN_TRUE_VALUE : \
            #"Define the simulated values of activated class means."\
            #"It is taken into account when mixture parameters are not sampled.",
        #P_ACT_VAR_TRUE_VALUE : \
            #"Define the simulated values of activated class variances."\
            #"It is taken into account when mixture parameters are not sampled.",
        #P_INACT_VAR_TRUE_VALUE : \
            #"Define the simulated values of inactivated class variances."\
            #"It is taken into account when mixture parameters are not sampled.",
        }

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment
        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]
        useTrueVal = self.parameters[self.P_USE_TRUE_VALUE]

        # get values for priors :
        self.varCIPrAlpha = self.parameters[self.P_VAR_CI_PR_ALPHA]
        self.varCIPrBeta = self.parameters[self.P_VAR_CI_PR_BETA]
        self.varCAPrAlpha = self.parameters[self.P_VAR_CA_PR_ALPHA]
        self.varCAPrBeta = self.parameters[self.P_VAR_CA_PR_BETA]

        self.meanCAPrMean = self.parameters[self.P_MEAN_CA_PR_MEAN]
        self.meanCAPrVar = self.parameters[self.P_MEAN_CA_PR_VAR]

        #self.ActMeanTrueValue = self.parameters[self.P_ACT_MEAN_TRUE_VALUE]
        #self.ActVarTrueValue = self.parameters[self.P_ACT_VAR_TRUE_VALUE]
        #self.InactVarTrueValue = self.parameters[self.P_INACT_VAR_TRUE_VALUE]

        an = ['component','condition']
        ad = {'component' : self.PARAMS_NAMES}
        GibbsSamplerVariable.__init__(self, 'mixt_params', valIni=valIni,
                                      useTrueValue=useTrueVal,
                                      sampleFlag=sampleFlag, axes_names=an,
                                      axes_domains=ad)

        php = self.parameters[self.P_HYPER_PRIOR]
        self.hyperPriorFlag = False if php=='Jeffrey' else True

        self.activ_thresh = self.parameters[self.P_ACTIV_THRESH]

    def linkToData(self, dataInput):
        self.dataInput =  dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX

        self.nrlCI = range(self.nbConditions)
        self.nrlCA = range(self.nbConditions)

        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%self.dataInput.simulData : ', self.dataInput.simulData.__class__
        #print 'self.dataInput : ', self.dataInput
        #print 'isinstance(self.dataInput, BOLDModel) : ', isinstance(self.dataInput, BOLDModel)


        if self.dataInput.simulData is not None and \
            isinstance(self.dataInput.simulData, list):
            print '%%%%%%%%%%:',  self.dataInput.simulData[0].has_key('condition_defs')
            if self.dataInput.simulData[0].has_key('condition_defs'):

                #take only 1st session -> same is assumed for others
                cdefs = self.dataInput.simulData[0]['condition_defs']
                self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                                          dtype=float)
                self.trueValue[self.I_MEAN_CA] = np.array([c.m_act for c in cdefs])
                self.trueValue[self.I_VAR_CA] = np.array([c.v_act for c in cdefs])
                self.trueValue[self.I_VAR_CI] = np.array([c.v_inact for c in cdefs])


        if self.dataInput.simulData is not None :
            if (isinstance(self.dataInput.simulData, BOLDModel) or \
                      isinstance(self.dataInput.simulData, BOLDModel2)):
                mixtures = self.dataInput.simulData.nrls.getMixture()
                itemsCond = mixtures.items()
                nbCondInSimu = len(itemsCond)
                self.trueValue = np.zeros((self.NB_PARAMS, nbCondInSimu ),
                                        dtype=float)

                #assert len(itemsCond) == self.nbConditions
                meanCA = np.zeros(nbCondInSimu, dtype=float)
                varCA = np.zeros(nbCondInSimu, dtype=float)
                varCI = np.zeros(nbCondInSimu, dtype=float)
                for cn,mixt in mixtures.iteritems():
                    genActiv = mixt.generators['activ']
                    genInactiv = mixt.generators['inactiv']
                    indCond = self.dataInput.simulData.nrls.condIds[cn]
                    varCI[indCond]= genInactiv.std**2
                    pyhrf.verbose(4, 'genActiv type = %s'
                                    %(genActiv.type))
                    pyhrf.verbose(4,  'genInactiv type = %s'
                                    %(genInactiv.type))
                    if genActiv.type=='gaussian' or genActiv.type=='log-normal':
                        meanCA[indCond]= genActiv.mean
                        varCA[indCond]= genActiv.std**2
                    elif genActiv.type =='gamma':
                        meanCA[indCond]= genActiv.a *genActiv.b
                        varCA[indCond]= genActiv.a *genActiv.b**2
                    if 0:
                        #TODO: debug -> why rewritting on meanCA and varCA ?
                        if genInactiv.type=='gaussian' or genInactiv.type=='log-normal':
                            meanCA[indCond]= genInactiv.mean
                            varCA[indCond]= genInactiv.std**2
                        elif genInactiv.type == 'uniform':
                            meanCA[indCond]= (genInactiv.minV+genInactiv.maxV)/2.
                            varCA[indCond]= (genInactiv.maxV-genInactiv.minV)/12.
                        elif genInactiv.type == 'beta':
                            meanCA[indCond]= genInactiv.a/                   \
                                                (genInactiv.a+genInactiv.b)
                            varCA[indCond]= genInactiv.a *                   \
                                            genInactiv.b/((genInactiv.a+     \
                                                            genInactiv.b)**2 *\
                                                            (genInactiv.a+     \
                                                            genInactiv.b+1.))
                self.trueValue[self.I_MEAN_CA] = meanCA
                self.trueValue[self.I_VAR_CA] = varCA
                self.trueValue[self.I_VAR_CI] = varCI

            ## To handle multi-sessions case
            #self.dataInput.simulData[0] or [1] has same nrls bar values
            if hasattr(self.dataInput, 'nbSessions') and \
                            ((isinstance(self.dataInput.simulData[0], BOLDModel) or \
                            isinstance(self.dataInput.simulData[0], BOLDModel2))):
                mixtures = self.dataInput.simulData[0].nrls.getMixture()
                itemsCond = mixtures.items()
                nbCondInSimu = len(itemsCond)
                self.trueValue = np.zeros((self.NB_PARAMS, nbCondInSimu ),
                                        dtype=float)

                #assert len(itemsCond) == self.nbConditions
                meanCA = np.zeros(nbCondInSimu, dtype=float)
                varCA = np.zeros(nbCondInSimu, dtype=float)
                varCI = np.zeros(nbCondInSimu, dtype=float)
                for cn,mixt in mixtures.iteritems():
                    genActiv = mixt.generators['activ']
                    genInactiv = mixt.generators['inactiv']
                    indCond = self.dataInput.simulData[0].nrls.condIds[cn]
                    varCI[indCond]= genInactiv.std**2
                    pyhrf.verbose(4, 'genActiv type = %s'
                                    %(genActiv.type))
                    pyhrf.verbose(4,  'genInactiv type = %s'
                                    %(genInactiv.type))
                    if genActiv.type=='gaussian' or genActiv.type=='log-normal':
                        meanCA[indCond]= genActiv.mean
                        varCA[indCond]= genActiv.std**2
                    elif genActiv.type =='gamma':
                        meanCA[indCond]= genActiv.a *genActiv.b
                        varCA[indCond]= genActiv.a *genActiv.b**2
                    if 0:
                        #TODO: debug -> why rewritting on meanCA and varCA ?
                        if genInactiv.type=='gaussian' or genInactiv.type=='log-normal':
                            meanCA[indCond]= genInactiv.mean
                            varCA[indCond]= genInactiv.std**2
                        elif genInactiv.type == 'uniform':
                            meanCA[indCond]= (genInactiv.minV+genInactiv.maxV)/2.
                            varCA[indCond]= (genInactiv.maxV-genInactiv.minV)/12.
                        elif genInactiv.type == 'beta':
                            meanCA[indCond]= genInactiv.a/                   \
                                                (genInactiv.a+genInactiv.b)
                            varCA[indCond]= genInactiv.a *                   \
                                            genInactiv.b/((genInactiv.a+     \
                                                            genInactiv.b)**2 *\
                                                            (genInactiv.a+     \
                                                            genInactiv.b+1.))
                self.trueValue[self.I_MEAN_CA] = meanCA
                self.trueValue[self.I_VAR_CA] = varCA
                self.trueValue[self.I_VAR_CI] = varCI



        elif self.dataInput.simulData is not None and \
                isinstance(self.dataInput.simulData, dict):

            self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                dtype=float)
            simulation = self.dataInput.simulData
            if simulation.has_key('condition_defs'):
                cdefs = simulation['condition_defs']
                self.trueValue[self.I_MEAN_CA] = np.array([c.m_act for c in cdefs])
                self.trueValue[self.I_VAR_CA] = np.array([c.v_act for c in cdefs])
                self.trueValue[self.I_VAR_CI] = np.array([c.v_inact for c in cdefs])



        #print 'meanCA linkToData : ', self.trueValue[self.I_MEAN_CA]
        #print 'varCA linkToData : ', self.trueValue[self.I_VAR_CA]
        #print 'varCI linkToData : ', self.trueValue[self.I_VAR_CI]


    def checkAndSetInitValue(self, variables):
        if self.currentValue is None:
            if self.useTrueValue:
                if self.trueValue is not None:
                    #TODO fix condition matching
                    self.currentValue = self.trueValue.copy()[:,:self.nbConditions]
                else:
                    raise Exception('Needed a true value but none defined')

            elif 0 and self.useTrueValue:
                self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions ), dtype=float)
                self.currentValue = np.zeros((self.NB_PARAMS, self.nbConditions), dtype=float)
                self.trueValue[self.I_MEAN_CA] = self.ActMeanTrueValue.values()
                self.trueValue[self.I_VAR_CA] = self.ActVarTrueValue.values()
                self.trueValue[self.I_VAR_CI] = self.InactVarTrueValue.values()
                self.currentValue = self.trueValue.copy()[:,:self.nbConditions]

            else:
                nc = self.nbConditions
                self.currentValue = np.zeros((self.NB_PARAMS, self.nbConditions), dtype=float)
                self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30.
                self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.
                self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.
                #self.currentValue[self.I_MEAN_CA] = self.trueValue[self.I_MEAN_CA]
                #self.currentValue[self.I_VAR_CA] = self.trueValue[self.I_VAR_CA]
                #self.currentValue[self.I_VAR_CI] = self.trueValue[self.I_VAR_CI]


    def getCurrentVars(self):
        return np.array([self.currentValue[self.I_VAR_CI],
                      self.currentValue[self.I_VAR_CA]])

    def getCurrentMeans(self):
        return np.array([np.zeros(self.nbConditions),
                      self.currentValue[self.I_MEAN_CA]])

    def computeWithProperPriors(self, j, cardCIj, cardCAj):
        #print 'sample hyper parameters with proper priors ...'
        if cardCIj > 1:
            nu0j = .5*np.dot(self.nrlCI[j], self.nrlCI[j])
            varCIj = 1.0/np.random.gamma(.5*cardCIj + self.varCIPrAlpha,
                                      1/(nu0j + self.varCIPrBeta))
        else :
            pyhrf.verbose(6,'using only hyper priors for CI (empty class) ...')
            varCIj = 1.0/np.random.gamma(self.varCIPrAlpha, 1/self.varCIPrBeta)


        if cardCAj > 1:
            #print 'cardCAj', cardCAj
            eta1j = np.mean(self.nrlCA[j])
            nrlCACentered = self.nrlCA[j] - self.currentValue[self.I_MEAN_CA,j]#eta1j
            nu1j = .5 * np.dot(nrlCACentered, nrlCACentered)
            #r = np.random.gamma(0.5*(cardCAj-1),2/nu1j)
            varCAj = 1.0/np.random.gamma(0.5*cardCAj + self.varCAPrAlpha,
                                      1/(nu1j + self.varCAPrBeta))
        else :
            pyhrf.verbose(6,'using only hyper priors for CA (empty class) ...')
            eta1j = 0.0
            varCAj = 1.0/np.random.gamma(self.varCAPrAlpha, 1/self.varCAPrBeta)

        invVarLikelihood = (cardCAj+0.)/varCAj
##            print 'self.meanCAPrVar :', self.meanCAPrVar

        meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar)

##            print 'meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar) :'
##            print '%f = 1/(%f + 1/%f)' %(meanCAVarAPost,invVarLikelihood,self.meanCAPrVar)
        #print 'meanCAVarAPost :', meanCAVarAPost
        rPrMV = self.meanCAPrMean/self.meanCAPrVar
        meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood+rPrMV)
        #print 'meanCAMeanAPost :', meanCAMeanAPost
##            print 'meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood + rPrMV) :'
##            print '%f = %f *(%f*%f + %f)' %(meanCAMeanAPost,meanCAVarAPost,eta1j,invVarLikelihood,rPrMV)
##            print 'meanCAMeanAPost :', meanCAMeanAPost
        meanCAj = np.random.normal(meanCAMeanAPost, meanCAVarAPost**0.5)

        return varCIj,meanCAj,varCAj

    def computeWithJeffreyPriors(self, j, cardCIj, cardCAj):

        #print 'sample hyper parameters with improper Jeffrey\'s priors ...'
        if pyhrf.verbose.verbosity >= 3:
            print 'cond %d - card CI = %d' %(j,cardCIj)
            print 'cond %d - card CA = %d' %(j,cardCAj)
            print 'cond %d - cur mean CA = %f' %(j,self.currentValue[self.I_MEAN_CA,j])
            if cardCAj > 0:
                print 'cond %d - nrl CA: %f(v%f)[%f,%f]' %(j,self.nrlCA[j].mean(),
                                                           self.nrlCA[j].var(),
                                                           self.nrlCA[j].min(),
                                                           self.nrlCA[j].max())
            if cardCIj > 0:
                print 'cond %d - nrl CI: %f(v%f)[%f,%f]' %(j,self.nrlCI[j].mean(),
                                                           self.nrlCI[j].var(),
                                                           self.nrlCI[j].min(),
                                                           self.nrlCI[j].max())

        if cardCIj > 1:
            nu0j = np.dot(self.nrlCI[j], self.nrlCI[j])
            varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj + 1) - 1, 2. / nu0j)
            #varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj - 1), 2. / nu0j)
        else :
            varCIj = 1.0 / np.random.gamma(0.5, 0.2)

        #HACK
        #varCIj = .5

        if cardCAj > 1:
            nrlC1Centered = self.nrlCA[j] - self.currentValue[self.I_MEAN_CA,j]
            ##print 'nrlC1Centered :', nrlC1Centered
            nu1j = np.dot(nrlC1Centered, nrlC1Centered)
            #r = np.random.gamma(0.5 * (cardCAj + 1) - 1, 2 / nu1j)
            #print 'nu1j / 2. :', nu1j / 2.
            #print '0.5 * (cardCAj + 1) - 1 =', 0.5 * (cardCAj + 1) - 1
            if pyhrf.verbose.verbosity >= 3:
                print 'varCA ~ InvGamma(%f, nu1j/2=%f)' %(0.5*(cardCAj+1)-1,
                                                          nu1j/2.)
                print ' -> mean =', (nu1j/2.)/(0.5*(cardCAj+1)-1)
            varCAj = 1.0 / np.random.gamma(0.5 * (cardCAj + 1) - 1, 2. / nu1j)
            #varCAj = 1.0 / np.random.gamma(0.5 * (cardCAj - 1), 2. / nu1j)
            pyhrf.verbose(3,'varCAj (j=%d) : %f' %(j,varCAj))
            if varCAj <= 0.:
                print 'variance for class activ and condition %s '\
                    'is negative or null: %f' %(self.dataInput.cNames[j],varCAj)
                print 'nu1j:', nu1j, '2. / nu1j', 2. / nu1j
                print 'cardCAj:', cardCAj, '0.5 * (cardCAj + 1) - 1:', \
                    0.5 * (cardCAj + 1) - 1
                print '-> setting it to almost 0.'
                varCAj = 0.0001
            #print '(varC1j/cardC1[j])**0.5 :', (varCAj/cardCAj)**0.5
            eta1j = np.mean(self.nrlCA[j])
            #print 'eta1j :', eta1j
            meanCAj = np.random.normal(eta1j, (varCAj / cardCAj)**0.5)

            # variance for class activ and condition video is negative or null:
            # 0.000000
            # nu1j: 2.92816412349e-306 2. / nu1j 6.83021823796e+305
            # cardCAj: 501 0.5 * (cardCAj + 1) - 1: 250.0
            # -> setting it to almost 0.

        else :
            #print 'Warning : cardCA <= 1!'
            varCAj = 1.0 / np.random.gamma(.5, 2.)
            if cardCAj == 0 :
                meanCAj = np.random.normal(5.0, varCAj**0.5)
            else:
                meanCAj = np.random.normal(self.nrlCA[j], varCAj**0.5)

        if pyhrf.verbose.verbosity >= 3:
            print 'Sampled components - cond', j
            print 'var CI =', varCIj
            print 'mean CA =', meanCAj, 'var CA =', varCAj

        return varCIj, meanCAj, varCAj


    def sampleNextInternal(self, variables):
        #TODO : comment

##        print '- Sampling Mixt params ...'

        nrlsSmpl = self.samplerEngine.getVariable('nrl')

        cardCA = nrlsSmpl.cardClass[self.L_CA,:]
        cardCI = nrlsSmpl.cardClass[self.L_CI,:]

        for j in xrange(self.nbConditions):
            vICI = nrlsSmpl.voxIdx[nrlsSmpl.L_CI][j]
            vICA = nrlsSmpl.voxIdx[nrlsSmpl.L_CA][j]
            self.nrlCI[j] = nrlsSmpl.currentValue[j, vICI]
            self.nrlCA[j] = nrlsSmpl.currentValue[j, vICA]

        for j in xrange(self.nbConditions):
        #for j in np.random.permutation(self.nbConditions):
            if self.hyperPriorFlag:
                varCIj,meanCAj,varCAj = self.computeWithProperPriors(j,
                                                                     cardCI[j],
                                                                     cardCA[j])
            else:
                varCIj,meanCAj,varCAj = self.computeWithJeffreyPriors(j,
                                                                      cardCI[j],
                                                                      cardCA[j])

            self.currentValue[self.I_VAR_CI, j] = varCIj
            self.currentValue[self.I_MEAN_CA, j] = meanCAj #absolute(meanCAj)
            self.currentValue[self.I_VAR_CA, j] = varCAj


            pyhrf.verbose(5, 'varCI,%d=%f' \
                              %(j,self.currentValue[self.I_VAR_CI,j]))
            pyhrf.verbose(5, 'meanCA,%d=%f' \
                              %(j,self.currentValue[self.I_MEAN_CA,j]))
            pyhrf.verbose(5, 'varCA,%d = %f' \
                              %(j,self.currentValue[self.I_VAR_CA,j]))

    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)
        sHrf = self.samplerEngine.getVariable('hrf')
        sScale = self.samplerEngine.getVariable('scale')

        if sHrf.sampleFlag and np.allclose(sHrf.normalise,0.) and \
                not sScale.sampleFlag and self.sampleFlag:
            pyhrf.verbose(6, 'Normalizing Posterior mean of Mixture Parameters at each iteration ...')
            #print '%%%% scaling NRL PME %%% - hnorm = ', sHrf.norm
            # Undo previous calculation:
            self.cumul -= self.currentValue
            #self.cumul2 -= self.currentValue**2
            self.cumul3 -=  (self.currentValue - self.mean)**2

            # Use scaled quantities instead:
            cur_m_CA = self.currentValue[self.I_MEAN_CA]
            cur_v_CA = self.currentValue[self.I_VAR_CA]
            cur_v_CI = self.currentValue[self.I_VAR_CI]
            self.cumul[self.I_MEAN_CA] +=  cur_m_CA * sHrf.norm
            #self.cumul2[self.I_MEAN_CA] += (cur_m_CA * sHrf.norm)**2
            self.cumul[self.I_VAR_CA] +=  cur_v_CA * sHrf.norm**2
            #self.cumul2[self.I_VAR_CA] += (cur_v_CA * sHrf.norm**2)**2
            self.cumul[self.I_VAR_CI] +=  cur_v_CI * sHrf.norm**2
            #self.cumul2[self.I_VAR_CI] += (cur_v_CI * sHrf.norm**2)**2

            self.mean = self.cumul / self.nbItObservables

            self.cumul3[self.I_MEAN_CA] +=  (cur_m_CA * sHrf.norm - self.mean[self.I_MEAN_CA])**2
            self.cumul3[self.I_VAR_CA] +=  (cur_v_CA * sHrf.norm**2 - self.mean[self.I_VAR_CA])**2
            self.cumul3[self.I_VAR_CI] +=  (cur_v_CI * sHrf.norm**2 - self.mean[self.I_VAR_CI])**2

            #self.error = self.cumul2 / self.nbItObservables - \
                #self.mean**2
            self.error = self.cumul3 / self.nbItObservables

    def get_string_value(self, v):
        v = v.transpose()
        if 0:
            print 'get_string_value for mixt_params ...'
            print v.shape, self.dataInput.cNames
            print '->', v[:,:len(self.dataInput.cNames)].shape
        return get_2Dtable_string(v[:len(self.dataInput.cNames),:],
                                  self.dataInput.cNames,
                                  self.PARAMS_NAMES,)

    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
            mixtp = np.zeros((2, self.nbConditions, 2))
            mixtp[self.L_CA, :, 0] = self.finalValue[self.I_MEAN_CA,:]
            mixtp[self.L_CA, :, 1] = self.finalValue[self.I_VAR_CA,:]
            mixtp[self.L_CI, :, 0] = 0.
            mixtp[self.L_CI, :, 1] = self.finalValue[self.I_VAR_CI,:]

            an = ['class','condition','component']
            ad = {'class':['inactiv','activ'],'condition':self.dataInput.cNames,
                  'component':['mean','var']}
            outputs['pm_'+self.name] = xndarray(mixtp, axes_names=an,
                                              axes_domains=ad)


            mixtp_mapped = np.tile(mixtp, (self.nbVox, 1, 1, 1))
            outputs['pm_'+self.name+'_mapped'] = xndarray(mixtp_mapped,
                                                        axes_names=['voxel']+an,
                                                        axes_domains=ad)

            region_is_active = self.finalValue[self.I_MEAN_CA,:].max() > \
                self.activ_thresh
            region_is_active = region_is_active.astype(np.int16)
            region_is_active = np.tile(region_is_active, self.nbVox)

            an = ['voxel']
            outputs['active_regions_from_mean_activ'] = xndarray(region_is_active,
                                                               axes_names=an)

        return outputs

    def finalizeSampling(self):
        GibbsSamplerVariable.finalizeSampling(self)
        del self.nrlCA
        del self.nrlCI






