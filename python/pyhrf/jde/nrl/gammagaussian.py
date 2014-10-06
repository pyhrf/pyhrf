# -*- coding: utf-8 -*-


import numpy as np
from pyhrf import xmlio
from pyhrf.jde.samplerbase import *
from pyhrf.jde.beta import *
#import sys


##########################################################
# Gamma/Gaussian mixture models
##########################################################
class InhomogeneousNRLSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Class handling the Gibbs sampling of Neural Response Levels according to
    Salima Makni's algorithm (IEEE SP 2005). Inherits the abstract class
    C{GibbsSamplerVariable}.
    #TODO : comment attributes
    """

    # parameters specifications :
    P_SAMPLE_LABELS = 'sampleLabels'
    P_LABELS_INI = 'labelsIni'
    P_BETA = 'beta'
    P_LABELS_COLORS = 'labelsColors'
    P_TRUE_LABELS = 'trueLabels'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_VAL_INI = 'initialValue'


    # parameters definitions and default values :
    defaultParameters = {
        P_SAMPLE_FLAG : 1,
        P_VAL_INI : None,
        P_BETA : 0.4,
        P_SAMPLE_LABELS : 1,
        P_LABELS_INI : None,
        P_LABELS_COLORS : np.array([0.0,0.0], dtype=float),
        }

    # other class attributes
    L_CI = 0
    L_CA = 1


    def __init__(self, parameters=None, xmlHandler=None, xmlLabel=None,
                 xmlComment=None):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]
        GibbsSamplerVariable.__init__(self,'nrl', valIni=valIni,
                                      sampleFlag=sampleFlag)


        # instance variables affectation from parameters :
        self.sampleLabelsFlag = self.parameters[self.P_SAMPLE_LABELS]
        self.labels = self.parameters[self.P_LABELS_INI]
        if self.parameters.has_key(self.P_TRUE_LABELS):
            self.trueLabels = params[self.P_TRUE_LABELS]
        else :
            self.trueLabels = None
        self.beta = self.parameters[self.P_BETA]
        self.constJ = 1
        #self.labelsColors = self.parameters[self.P_LABELS_COLORS]


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.cardCI = np.zeros(self.nbConditions, dtype=int)
        self.cardCA = np.zeros(self.nbConditions, dtype=int)
        self.voxIdxCI = range(self.nbConditions)
        self.voxIdxCA = range(self.nbConditions)

        self.cumulLabels = np.zeros((self.nbConditions,self.nbVox), dtype=int)

        self.meanClassAApost = range(self.nbConditions)
        self.meanClassIApost = range(self.nbConditions)
        self.varXjhtQjeji = range(self.nbConditions)

        # some allocations :
        for j in xrange(self.nbConditions):
            self.meanClassAApost[j] = numpy.empty((self.nbVox))
            self.meanClassIApost[j] = numpy.empty((self.nbVox))
            self.varXjhtQjeji[j] = numpy.empty((self.nbVox))


    def checkAndSetInitValue(self, variables):
        # Generating default labels if necessary :
        if self.labels == None : # if no initial labels specified
            if not self.sampleLabelsFlag and self.dataInput.simulData != None :
                self.labels = self.dataInput.simulData.nrls.labels
            else:
                self.labels = np.zeros((self.nbConditions, self.nbVox), dtype=int)
                nbVoxInClassI = self.nbVox/2
                # Uniform dispatching :
                for j in xrange(self.nbConditions) :
                    self.labels[j,0:nbVoxInClassI] = self.L_CI
                    self.labels[j,nbVoxInClassI:self.nbVox] = self.L_CA
                    self.labels[j,:] = np.random.permutation(self.labels[j,:])

        self.countLabels()

        if self.currentValue == None :
            if not self.sampleFlag and self.dataInput.simulData != None :
                self.currentValue = self.dataInput.simulData.nrls.data
            else:
                nrlsIni = np.zeros((self.nbConditions, self.nbVox), dtype=float)
                # Init Nrls according to classes definitions :
                smplMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
                # ensure that mixture parameters are correctly set
                smplMixtP.checkAndSetInitValue(variables)

                var_classI = smplMixtP.currentValue[GamGaussMixtureParamsSampler.I_VAR_CI]
                shape_classA = smplMixtP.currentValue[GamGaussMixtureParamsSampler.I_SHAPE_CA]
                scale_classA = smplMixtP.currentValue[GamGaussMixtureParamsSampler.I_SCALE_CA]
                for j in xrange(self.nbConditions):
                    nrlsIni[j, self.voxIdxCI[j]] = np.random.randn(self.cardCI[j])*var_classI[j]**0.5
                    nrlsIni[j, self.voxIdxCA[j]] = np.random.gamma(shape_classA[j], scale_classA[j], self.cardCA[j])
                self.currentValue = nrlsIni

    def countLabels(self):
        for j in xrange(self.nbConditions) :
            self.voxIdxCI[j] = where(self.labels[j,:]==self.L_CI)[0]
            self.voxIdxCA[j] = where(self.labels[j,:]==self.L_CA)[0]
            self.cardCI[j] = len(self.voxIdxCI[j])
            self.cardCA[j] = len(self.voxIdxCA[j])
            pyhrf.verbose(5,
                          'Nb vox in CI for cond %d : %d' %(j,self.cardCI[j]))
            pyhrf.verbose(5,
                          'Nb vox in CI for cond %d : %d' %(j,self.cardCA[j]))


    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """
        ##print 'NRLs warming up ...'

        # Compute precalculations :
        smplHRF = self.get_variable('hrf')
        smplHRF.checkAndSetInitValue(variables)
        self.normHRF = smplHRF.norm
        self.varYtilde = np.zeros((self.ny, self.nbVox), dtype=float)
        self.computeVarYTilde(smplHRF.varXh)
        self.varXhtQXh_allvox = empty((self.nbConditions,self.nbVox),dtype=float)
        self.sumRmatXhtQXh = empty((self.nbConditions,self.nbVox),dtype=float)
        self.varClassIApost = empty((self.nbConditions,self.nbVox),dtype=float)
        self.varClassAApost = empty((self.nbConditions,self.nbVox),dtype=float)
        self.sigClassIApost = empty((self.nbConditions,self.nbVox),dtype=float)
        self.sigClassAApost = empty((self.nbConditions,self.nbVox),dtype=float)

    def computeVarYTilde(self, varXh):
        for i in xrange(self.nbVox):
            repNRLi = repmat(self.currentValue[:,i], self.ny, 1)
            aijXjh = repNRLi * varXh
            self.varYtilde[:,i] = self.dataInput.varMBY[:,i] - aijXjh.sum(axis=1)

    def sampleNextAlt(self, variables):
        varXh = self.get_variable('hrf').varXh
        self.computeVarYTilde(varXh)

    def computeVariablesApost(self, varCI, shapeCA, scaleCA, rb, varXh,
                              varLambda):

        self.varXhtQ = dot(varXh.transpose(),self.dataInput.delta)
        varXhtQXh = diag(dot(self.varXhtQ,varXh))
        rmatVarXhtQXh = repeat(varXhtQXh,self.nbVox).reshape(varXhtQXh.size,
                                                             self.nbVox)

        numpy.divide(rmatVarXhtQXh, repmat(rb,self.nbConditions,1),
                     self.varXhtQXh_allvox)

        rmatInvVarCI = repeat(1.0/varCI,self.nbVox).reshape(self.nbConditions,
                                                            self.nbVox)

        numpy.power(numpy.add(rmatInvVarCI, self.varXhtQXh_allvox,
                              self.sumRmatXhtQXh),-1,self.varClassIApost)
        numpy.power(self.varXhtQXh_allvox, -1,self.varClassAApost)

        numpy.square(self.varClassIApost,self.sigClassIApost)
        numpy.square(self.varClassAApost,self.sigClassAApost)

#        rmatMCAVCA = repeat(meanCA/varCA, self.nbVox).reshape(self.nbConditions,
#                                                              self.nbVox)
        numpy.multiply( - scaleCA,self.varClassAApost,self.interMeanClassA)

        if self.beta <= 0:
            self.ratioLambdaI_A = ( (1-varLambda)* numpy.gamma()(varCA**0.5) ) \
                                 / ( varLambda  *(varCI**0.5) )
        else:
            self.ratioLambdaI_A = (varCA/varCI)**0.5

        pyhrf.verbose(5, 'ratioLambdaI_A :')
        pyhrf.verbose.printNdarray(5, self.ratioLambdaI_A)

    def computeMeanClassApost(self, j, nrls, varXhj, rb):

        pyhrf.verbose(5,'self.varXhtQ[%d,:] :'%j)
        pyhrf.verbose.printNdarray(5, self.varXhtQ[j,:])


        for i in xrange(self.nbVox):
            eji = self.varYtilde[:,i] + nrls[j,i]*varXhj
            self.varXjhtQjeji[i] = dot(self.varXhtQ[j,:],eji)/rb[i]
        numpy.add(self.interMeanClassA[j,:],
                  multiply(self.varClassAApost[j,:], self.varXjhtQjeji),
                  self.meanClassAApost[j])
        pyhrf.verbose(5,'meanClassAApost cond %d :'%j)
        pyhrf.verbose.printNdarray(5,
                                   self.meanClassAApost[j])

        multiply(self.varClassIApost[j,:],
                 self.varXjhtQjeji,
                 self.meanClassIApost[j])
        pyhrf.verbose(5,'meanClassIApost cond %d :'%j)
        pyhrf.verbose.printNdarray(5,
                                   self.meanClassIApost[j])


    def sampleNextInternal(self, variables):
        #TODO : comment

        varCI = variables[self.samplerEngine.I_MIXT_PARAM].currentValue[BiGaussMixtureParamsSampler.I_VAR_CI]
        varCA = variables[self.samplerEngine.I_MIXT_PARAM].currentValue[BiGaussMixtureParamsSampler.I_VAR_CA]
        meanCA = variables[self.samplerEngine.I_MIXT_PARAM].currentValue[BiGaussMixtureParamsSampler.I_MEAN_CA]
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        varXh = self.get_variable('hrf').varXh
        varLambda = variables[self.samplerEngine.I_WEIGHTING_PROBA].currentValue

        self.computeVarYTilde(varXh)
        self.computeVariablesApost(varCI, varCA, meanCA, rb, varXh, varLambda)

        self.labelsSamples = np.random.rand(self.nbConditions, self.nbVox)
        nrlsSamples = np.random.randn(self.nbConditions, self.nbVox)


        for j in xrange(self.nbConditions):
            self.computeMeanClassApost(j, self.currentValue, varXh[:,j], rb)
            if self.sampleLabelsFlag:
                pyhrf.verbose(3,'Sampling labels ...')
                self.sampleLabels(j, varCI, varCA, meanCA)
                pyhrf.verbose(3,'Sampling labels done!')

            sigApost = np.zeros(self.nbVox, dtype=float)

            putmask(sigApost, self.labels[j,:], self.sigClassAApost[j,:])
            putmask(sigApost, 1-self.labels[j,:], self.sigClassIApost[j,:])

            meanApost = np.zeros(self.nbVox, dtype=float)
            putmask(meanApost,self.labels[j,:], self.meanClassAApost[j] )
            putmask(meanApost,1-self.labels[j,:], self.meanClassIApost[j] )

            self.currentValue[j,:] = multiply(nrlsSamples[j,:], sigApost) + \
                                     meanApost
            pyhrf.verbose(6, 'All nrl cond %d:'%j)
            pyhrf.verbose.printNdarray(6,self.currentValue[j,:])
            pyhrf.verbose(4,
                          'nrl cond %d = %1.3f(%1.3f)'
                          %(j,self.currentValue[j,:].mean(),
                            self.currentValue[j,:].std()))


            pyhrf.verbose(6, 'All nrl activ cond %d:'%j)
            pyhrf.verbose.printNdarray(6,
                                       self.currentValue[j,self.voxIdxCA[j]])

            pyhrf.verbose(4,
                          'nrl activ cond %d = %1.3f(%1.3f)'
                          %(j,self.currentValue[j,self.voxIdxCA[j]].mean(),
                            self.currentValue[j,self.voxIdxCA[j]].std()))

            pyhrf.verbose(6, 'All nrl inactiv cond %d:'%j)
            pyhrf.verbose.printNdarray(6,
                                       self.currentValue[j,self.voxIdxCI[j]])

            pyhrf.verbose(4,
                          'nrl inactiv cond %d = %1.3f(%1.3f)'
                          %(j,self.currentValue[j,self.voxIdxCI[j]].mean(),
                            self.currentValue[j,self.voxIdxCI[j]].std()))

            #TODO : may be improved by not completely updating varYtilde ...
            self.computeVarYTilde(varXh)

        self.countLabels()
        self.normHRF = self.get_variable('hrf').norm


    def sampleLabels(self, cond, varCI, varCA, meanCA):
        #print 'sampling labels ...'

        fracLambdaTilde = self.ratioLambdaI_A[cond]*                     \
                          ( self.sigClassIApost[cond,:]                 \
                            /self.sigClassAApost[cond,:] ) *            \
                            np.exp(0.5*(self.meanClassIApost[cond]**2      \
                                     /self.varClassIApost[cond,:]       \
                                     -self.meanClassAApost[cond]**2     \
                                     /self.varClassAApost[cond,:]       \
                                     +meanCA[cond]**2/varCA[cond] ) )
        if self.beta > 0:
            #TODO : a paralleliser : possible ??
            for i in xrange(self.nbVox):
                fracLambdaTilde[i] *= np.exp( -self.beta*(self.calcEnergy(i, self.L_CI, cond)-
                                                          self.calcEnergy(i, self.L_CA, cond)) )

        varLambdaApost = 1/(1+fracLambdaTilde)

        self.labels[cond,:] = np.array( self.labelsSamples[cond,:] <= varLambdaApost, dtype=int )


    def calcEnergy(self, voxIdx, label, cond):

        neighboursMask = self.dataInput.voxelMapping.getNeighboursIndexes(voxIdx)
        spCorr = np.sum( 2*(self.labels[cond, neighboursMask]==label)-1.0 )

        return -self.constJ*spCorr #-self.labelsColors[label]


    def finalizeSampling(self):
        self.finalLabels = np.zeros((self.nbConditions,self.nbVox), dtype=int)
        for j in xrange(self.nbConditions):
            self.finalLabels[j,where(self.meanLabels[j,:]>=0.5)] = 1

        if self.trueLabels != None:
            for j in xrange(self.nbConditions):
                deltaLab = self.finalLabels[j,:] - self.trueLabels[j,:]
                self.finalLabels[j,where(deltaLab==-1)] = -1 # false negative
                self.finalLabels[j,where(deltaLab==1)] = 2 # false positive

        smplHRF = self.samplerEngine.get_variable('hrf')

        # Correct sign ambiguity :
        if smplHRF.detectSignError():
            self.finalValue *= -1

        # Correct hrf*nrl scale ambiguity :
        scaleF = smplHRF.getScaleFactor()
        # Use HRF amplitude :
        self.finalValueScaleCorr = self.finalValue*scaleF

        # clean memory of temporary variables :
        del self.varXhtQXh_allvox
        del self.sumRmatXhtQXh
        del self.varClassIApost
        del self.varClassAApost
        del self.sigClassIApost
        del self.sigClassAApost
        del self.interMeanClassA
        del self.meanClassAApost
        del self.meanClassIApost
        del self.varXjhtQjeji

    def computeMean(self):
##        print 'computing nrls mean ...'

        self.cumul += self.currentValue#*self.normHRF
        self.cumulLabels += self.labels

        self.nbItMean += 1
        self.mean = self.cumul/self.nbItMean
        self.meanLabels = self.cumulLabels/(self.nbItMean+0.0)




class GamGaussMixtureParamsSampler(GibbsSamplerVariable):
    """
    #TODO : comment

    """

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Shape_Activ', 'Scale_Activ', 'Var_Inactiv']

    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'

    P_SHAPE_CA_PR_MEAN = 'shapeCAPrMean'
    P_SCALE_CA_PR_ALPHA = 'scaleCAPrAlpha'
    P_SCALE_CA_PR_BETA = 'scaleCAPrBeta'

    P_VAR_CI_PR_ALPHA = 'varCIPrAlpha'
    P_VAR_CI_PR_BETA = 'varCIPrBeta'



    defaultParameters = {
        P_VAL_INI : None,
        P_SAMPLE_FLAG : 1,
        P_SHAPE_CA_PR_MEAN : 10.,
        P_SCALE_CA_PR_ALPHA : 2.5,
        P_SCALE_CA_PR_BETA : 1.5,
        P_VAR_CI_PR_ALPHA : 2.5,
        P_VAR_CI_PR_BETA : .5,
        }

    def __init__(self, parameters=None, xmlHandler=None,
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment
        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]

        # get values for priors :
        self.varCIPrAlpha = self.parameters[self.P_VAR_CI_PR_ALPHA]
        self.varCIPrBeta = self.parameters[self.P_VAR_CI_PR_BETA]
        self.scaleCAPrAlpha = self.parameters[self.P_SCALE_CA_PR_ALPHA]
        self.scaleCAPrBeta = self.parameters[self.P_SCALE_CA_PR_BETA]

        self.shapeCAPrMean = self.parameters[self.P_SHAPE_CA_PR_MEAN]

        GibbsSamplerVariable.__init__(self, 'mixtParams', valIni=valIni, sampleFlag=sampleFlag)


    def linkToData(self, dataInput):
        self.dataInput =  dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX

        self.nrlCI = range(self.nbConditions)
        self.nrlCA = range(self.nbConditions)

    def checkAndSetInitValue(self, variables):

        if self.currentValue == None:
            self.currentValue = np.zeros((self.NB_PARAMS, self.nbConditions), dtype=float)
            if not self.sampleFlag and self.dataInput.simulData != None :
                mixtures = self.dataInput.simulData.nrls.getMixture()
                itemsCond = mixtures.items()
                assert len(itemsCond) == self.nbConditions
                shapeCA = np.zeros(self.nbConditions, dtype=float)
                scaleCA = np.zeros(self.nbConditions, dtype=float)
                varCI = np.zeros(self.nbConditions, dtype=float)
                for cn,mixt in mixtures.iteritems():
                    genActiv = mixt.generators['activ']
                    genInactiv = mixt.generators.generators['inactiv']
                    indCond = self.dataInput.simulData.nrls.condIds[cn]
                    meanCA= genActiv.mean
                    varCA= genActiv.std**2
                    shapeCA[indCond] = meanCA**2/(varCA+0.)
                    scaleCA[indCond] = varCA/(meanCA+0.)
                    varCI[indCond]= genInactiv.std**2
                self.currentValue[self.I_SHAPE_CA] = shapeCA
                self.currentValue[self.I_SCALE_CA] = scaleCA
                self.currentValue[self.I_VAR_CI] = varCI
        else:
            #TODO : put constants in default init values ...
            self.currentValue[self.I_SHAPE_CA] = np.zeros(self.nbConditions) + 10.0
            self.currentValue[self.I_SCALE_CA] = np.zeros(self.nbConditions) + 5.0
            self.currentValue[self.I_VAR_CI] = np.zeros(self.nbConditions) + 1.


    def sampleNextInternal(self, variables):
        #TODO : comment

##        print '- Sampling Mixt params ...'

        nrlsSmpl = variables[self.samplerEngine.I_NRLS]
        cardCA = nrlsSmpl.cardCA
        cardCI = nrlsSmpl.cardCI


        for j in xrange(self.nbConditions):
            self.nrlCI[j] = nrlsSmpl.currentValue[j, nrlsSmpl.voxIdxCI[j]]
            self.nrlCA[j] = nrlsSmpl.currentValue[j, nrlsSmpl.voxIdxCA[j]]

        self.currentValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                                  dtype=float)

        for j in xrange(self.nbConditions):
            if cardCI[j] > 0:
                nu0j = .5*dot(self.nrlCI[j], self.nrlCI[j])
                varCIj = 1.0/np.random.gamma(.5*cardCI[j] + self.varCIPrAlpha,
                                          1/(nu0j + self.varCIPrBeta))
            else :
                varCIj = 1.0/np.random.gamma(self.varCIPrAlpha, 1/self.varCIPrBeta)

            self.currentValue[self.I_VAR_CI, j] = varCIj
#            self.currentValue[self.I_VAR_CI, j] = 0.0000001

            #sampling of shapeCA and scaleCA to be completed

#            invVarLikelihood = cardCA[j]/varCAj

            meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar)
##            print 'meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar) :'
##            print '%f = 1/(%f + 1/%f)' %(meanCAVarAPost,invVarLikelihood,self.meanCAPrVar)
            rPrMV = self.meanCAPrMean/self.meanCAPrVar
            meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood + rPrMV)

##            print 'meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood + rPrMV) :'
##            print '%f = %f *(%f*%f + %f)' %(meanCAMeanAPost,meanCAVarAPost,eta1j,invVarLikelihood,rPrMV)
##            print 'meanCAMeanAPost :', meanCAMeanAPost
            shapeCAj = np.random.normal(meanCAMeanAPost, meanCAVarAPost**0.5)

            self.currentValue[self.I_SHAPE_CA, j] = shapeCAj #absolute(meanCAj)
            self.currentValue[self.I_SCALE_CA, j] = scaleCAj


            pyhrf.verbose(4, 'varCI,%d = %f' \
                          %(j,self.currentValue[self.I_VAR_CI,j]))

            pyhrf.verbose(4, 'meanCA,%d = %f' \
                          %(j,self.currentValue[self.I_SHAPE_CA,j]))

            pyhrf.verbose(4, 'varCA,%d = %f' \
                          %(j,self.currentValue[self.I_SCALE_CA,j]))

