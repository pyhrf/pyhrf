# -*- coding: utf-8 -*-

import logging

import numpy.matlib

from numpy import *

from pyhrf.jde.samplerbase import *
from pyhrf.jde.beta import *
from pyhrf.jde.nrl.bigaussian import NRLSampler


logger = logging.getLogger(__name__)


class NRLARSampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels according
    to Salima Makni (ISBI 2006).
    Inherits the abstract class C{GibbsSamplerVariable}.

    #TODO : comment attributes
    """

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.cardClass = zeros((self.nbClasses, self.nbConditions), dtype=int)
        self.voxIdx = [range(self.nbConditions)
                       for _ in xrange(self.nbClasses)]

        self.cumulLabels = zeros((self.nbConditions, self.nbVox), dtype=int)
        self.meanClassAApost = empty((self.nbConditions, self.nbVox),
                                     dtype=float)
        self.meanClassIApost = empty((self.nbConditions, self.nbVox),
                                     dtype=float)
        self.varXjhtLambdaXjh = empty((self.nbVox), dtype=float)
        self.varXjhtLambdaeji = empty((self.nbVox), dtype=float)

    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """

        # Compute precalculations :
        smplHRF = self.get_variable('hrf')
        smplHRF.checkAndSetInitValue(variables)
        smpldrift = self.get_variable('drift')
        smpldrift.checkAndSetInitValue(variables)
        self.varYtilde = zeros((self.ny, self.nbVox), dtype=float)
        self.computeVarYTilde(smplHRF.varXh, smpldrift.varMBYPl)

        self.normHRF = smplHRF.norm
        shape = (self.nbClasses, self.nbConditions, self.nbVox)
        self.varClassApost = empty(shape, dtype=float)
        self.sigClassApost = empty(shape, dtype=float)
        self.meanClassApost = empty(shape, dtype=float)
        self.meanApost = empty((self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = zeros((self.nbConditions, self.nbVox), dtype=float)

        self.corrEnergies = zeros((self.nbConditions, self.nbVox), dtype=float)

    def computeVarYTilde(self, varXh, varMBYPl):
        for i in xrange(self.nbVox):
            repNRLi = numpy.matlib.repmat(self.currentValue[:, i], self.ny, 1)
            aijXjh = repNRLi * varXh
            self.varYtilde[:, i] = varMBYPl[:, i] - aijXjh.sum(axis=1)

    def sampleNextAlt(self, variables):
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        self.computeVarYTilde(varXh, varMBYPl)

    def computeMeanVarClassApost(self, j, variables):
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()

        reps = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        varInvAutoCorrNoise = (variables[self.samplerEngine.I_NOISE_ARP].
                               InvAutoCorrNoise)

        nrls = self.currentValue
        varXhj = varXh[:, j]

        logger.debug('varInvAutoCorrNoise[:,:,0] :')
        logger.debug(varInvAutoCorrNoise[:, :, 0])
        for i in xrange(self.nbVox):
            eji = self.varYtilde[:, i] + nrls[j, i] * varXhj
            varXjhtLambda = dot(varXhj.transpose(),
                                varInvAutoCorrNoise[:, :, i]) / reps[i]
            logger.debug('varXjhtLambda :')
            logger.debug(varXjhtLambda)

            self.varXjhtLambdaXjh = dot(varXjhtLambda, varXhj)
            self.varXjhtLambdaeji = dot(varXjhtLambda, eji)
            for c in xrange(self.nbClasses):
                self.varClassApost[c, j, i] = 1. / (1. / var[c, j] +
                                                    self.varXjhtLambdaXjh)
                self.varClassApost[
                    c, j, i] = self.sigClassApost[c, j, i] ** 0.5

                if c > 0:  # assume 0 stands for inactivating class
                    self.meanClassApost[c, j, i] = ((mean[c, j] / var[c, j] +
                                                     self.varXjhtLambdaeji) *
                                                    self.varClassApost[c, j, i])
                else:
                    self.meanClassApost[c, j, i] = (self.varClassApost[c, j, i] *
                                                    self.varXjhtLambdaeji)
            logger.debug('meanClassApost %d cond %d :', j, i)
            logger.debug(self.meanClassApost[:, j, i])

    def sampleNextInternal(self, variables):
        # TODO : comment

        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        varInvAutoCorrNoise = (variables[self.samplerEngine.I_NOISE_ARP].
                               InvAutoCorrNoise)

        self.computeVarYTilde(varXh, varMBYPl)

        self.labelsSamples = random.rand(self.nbConditions, self.nbVox)
        nrlsSamples = random.randn(self.nbConditions, self.nbVox)

        for j in xrange(self.nbConditions):
            self.computeMeanVarClassApost(j, variables)
            logger.info('Sampling labels - cond %d ...', j)
            if self.sampleLabelsFlag:
                self.sampleLabels(j, variables)

            for c in xrange(self.nbClasses):
                putmask(self.sigApost[j, :], self.labels[j, :] == c,
                        self.sigClassApost[c, j, :])
                putmask(self.meanApost[j, :], self.labels[j, :] == c,
                        self.meanClassApost[c, j, :])

            add(multiply(nrlsSamples[j, :], self.sigApost[j, :]),
                self.meanApost[j, :],
                self.currentValue[j, :])

            logger.debug('All nrl cond %d:', j)
            logger.debug(self.currentValue[j, :])
            logger.info('nrl cond %d = %1.3f(%1.3f)',
                        j, self.currentValue[j, :].mean(),
                        self.currentValue[j, :].std())
            for c in xrange(self.nbClasses):
                logger.debug('All nrl %s cond %d:', self.CLASS_NAMES[c], j)
                ivc = self.voxIdx[c][j]
                logger.debug(self.currentValue[j, ivc])

                logger.info('nrl %s cond %d = %1.3f(%1.3f)',
                            self.CLASS_NAMES[c], j,
                            self.currentValue[j, ivc].mean(),
                            self.currentValue[j, ivc].std())

            self.computeVarYTilde(varXh, varMBYPl)

        self.countLabels(self.labels, self.voxIdx, self.cardClass)

    def cleanMemory(self):
        # clean memory of temporary variables :
        del self.varClassApost
        del self.sigClassApost
        del self.varXjhtLambdaeji
        del self.varXjhtLambdaXjh
        del self.sigApost
        del self.meanApost
        del self.varYtilde
        if hasattr(self, 'labelsSamples'):
            del self.labelsSamples
        del self.corrEnergies
        del self.labels
        del self.voxIdx
