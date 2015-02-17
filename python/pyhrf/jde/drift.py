# -*- coding: utf-8 -*-

import logging

from numpy import *
import numpy as np

from numpy.linalg import cholesky
from numpy.matlib import repmat

import pyhrf
import intensivecalc
from pyhrf import xmlio
from samplerbase import *
from numpy.matlib import *


logger = logging.getLogger(__name__)


class DriftSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Gibbs sampler of the parameters modelling the low frequency drift in
    the fMRI time course, in the case of white noise.
    """

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        # TODO : comment
        xmlio.XmlInitable.__init__(self)

        an = ['order', 'voxel']
        GibbsSamplerVariable.__init__(self, 'drift', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='PM LFD')

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbSess = self.dataInput.nbSessions
        self.dimDrift = self.dataInput.colP
        self.nbVox = self.dataInput.nbVoxels
        self.P = self.dataInput.lfdMat[0]  # 0 for 1st session

        if dataInput.simulData is not None and \
                isinstance(dataInput.simulData, BOLDModel):
            self.trueValue = dataInput.simulData.rdrift.lfd

    def checkAndSetInitValue(self, variables):
        smplVarDrift = self.get_variable('drift_var')
        smplVarDrift.checkAndSetInitValue(variables)
        varDrift = smplVarDrift.currentValue

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue
            else:
                raise Exception('Needed a true value for drift init but '
                                'None defined')

        if 0 and self.currentValue is None:
            self.currentValue = np.sqrt(varDrift) * \
                np.random.randn(self.dimDrift, self.nbVox)

        if self.currentValue is None:
            logger.info("Initialisation of Drift from the data")
            ptp = numpy.dot(self.P.transpose(), self.P)
            invptp = numpy.linalg.inv(ptp)
            invptppt = numpy.dot(invptp, self.P.transpose())
            self.currentValue = numpy.dot(invptppt, self.dataInput.varMBY)

        self.updateNorm()
        self.matPl = dot(self.P, self.currentValue)
        self.ones_Q_J = np.ones((self.dimDrift, self.nbVox))
        self.ones_Q = np.ones((self.dimDrift))

    def updateNorm(self):
        self.norm = (self.currentValue * self.currentValue).sum()

        # if self.trueValue is not None:
        # print 'cur drift norm:', self.norm
        # print 'true drift norm:', (self.trueValue * self.trueValue).sum()

        #n2 = sum( diag( dot( self.currentValue.transpose(), self.currentValue ) ) )
        # if not np.allclose(self.norm,n2):
        #     print 'norm != n2'
        #     print self.norm
        #     print n2

    def sampleNextInternal(self, variables):
        reps = self.get_variable('noise_var').currentValue
        snrls = self.get_variable('nrl')
        sHrf = self.get_variable('hrf')
        eta = self.get_variable('drift_var').currentValue

        for i in xrange(self.nbVox):
            # inv_sigma = np.eye(self.dimDrift) * (1/eta + 1/reps[i])
            # pty = np.dot(self.P.transpose(), snrls.varYtilde[:,i])
            # self.currentValue[:,i] = sampleDrift(inv_sigma, pty, self.dimDrift)

            v_lj = reps[i] * eta / (reps[i] + eta)
            #v_lj = ( 1/reps[i] + 1/eta )
            mu_lj = v_lj / reps[i] * \
                np.dot(self.P.transpose(), snrls.varYtilde[:, i])
            # print 'ivox=%d, v_lj=%f, std_lj=%f mu_lj=' %(i,v_lj,v_lj**.5),
            # mu_lj
            self.currentValue[:, i] = np.random.randn(
                self.dimDrift) * v_lj ** .5 + mu_lj

        logger.debug('eta : %f', eta)
        logger.debug('reps :')
        logger.debug(reps)

        inv_vars_l = (1 / reps + 1 / eta) * self.ones_Q_J
        mu_l = 1 / inv_vars_l * np.dot(self.P.transpose(), snrls.varYtilde)

        logger.debug('vars_l :')
        logger.debug(1 / inv_vars_l)

        logger.debug('mu_l :')
        logger.debug(mu_l)

        cur_val = np.random.normal(mu_l, 1 / inv_vars_l)

        logger.debug('drift params :')
        logger.debug(self.currentValue)

        logger.debug('drift params (alt) :')
        logger.debug(cur_val)

        #assert np.allclose(cur_val, self.currentValue)

        self.updateNorm()
        self.matPl = dot(self.P, self.currentValue)

        # updating VarYTilde and VarYbar
        varXh = sHrf.varXh
        snrls.computeVarYTildeOpt(varXh)

    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)
        drifts = np.dot(self.P, self.finalValue)
        an = ['time', 'voxel']
        ad = {'time': arange(self.dataInput.ny) * self.dataInput.tr}
        outputs['drift_signal'] = xndarray(drifts, axes_names=an, axes_domains=ad,
                                           value_label='Delta BOLD')

        return outputs


class DriftSamplerWithRelVar(DriftSampler):
    """
    Gibbs sampler of the parameters modelling the low frequency drift in
    the fMRI time course, in the case of white noise.
    """

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbSess = self.dataInput.nbSessions
        self.dimDrift = self.dataInput.colP
        self.nbVox = self.dataInput.nbVoxels
        self.P = self.dataInput.lfdMat[0]  # 0 for 1st session

        if dataInput.simulData is not None and \
                isinstance(dataInput.simulData, BOLDModel):
            self.trueValue = dataInput.simulData.rdrift.lfd

    def checkAndSetInitValue(self, variables):
        smplVarDrift = self.get_variable('drift_var')
        smplVarDrift.checkAndSetInitValue(variables)
        varDrift = smplVarDrift.currentValue

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue
            else:
                raise Exception('Needed a true value for drift init but '
                                'None defined')

        if 0 and self.currentValue is None:
            self.currentValue = np.sqrt(varDrift) * \
                np.random.randn(self.dimDrift, self.nbVox)

        if self.currentValue is None:
            logger.info("Initialisation of Drift from the data")
            n = len(self.dataInput.varMBY)
            ptp = numpy.dot(self.P.transpose(), self.P)
            invptp = numpy.linalg.inv(ptp)
            invptppt = numpy.dot(invptp, self.P.transpose())
            self.currentValue = numpy.dot(invptppt, self.dataInput.varMBY)

        self.updateNorm()
        self.matPl = dot(self.P, self.currentValue)
        self.ones_Q_J = np.ones((self.dimDrift, self.nbVox))
        self.ones_Q = np.ones((self.dimDrift))

    def updateNorm(self):
        self.norm = (self.currentValue * self.currentValue).sum()

    def sampleNextInternal(self, variables):
        reps = self.get_variable('noise_var').currentValue
        snrls = self.get_variable('nrl')
        sHrf = self.get_variable('hrf')
        eta = self.get_variable('drift_var').currentValue
        w = self.get_variable('W').currentValue

        for i in xrange(self.nbVox):
            v_lj = reps[i] * eta / (reps[i] + eta)
            mu_lj = v_lj / reps[i] * \
                np.dot(self.P.transpose(), snrls.varYtilde[:, i])
            self.currentValue[:, i] = np.random.randn(
                self.dimDrift) * v_lj ** .5 + mu_lj

        logger.debug('eta : %f' % eta)
        logger.debug('reps :')
        logger.debug(reps)

        inv_vars_l = (1 / reps + 1 / eta) * self.ones_Q_J
        mu_l = 1 / inv_vars_l * np.dot(self.P.transpose(), snrls.varYtilde)

        logger.debug('vars_l :')
        logger.debug(1 / inv_vars_l)

        logger.debug('mu_l :')
        logger.debug(mu_l)

        cur_val = np.random.normal(mu_l, 1 / inv_vars_l)

        logger.debug('drift params :')
        logger.debug(self.currentValue)

        logger.debug('drift params (alt) :')
        logger.debug(cur_val)

        #assert np.allclose(cur_val, self.currentValue)

        self.updateNorm()
        self.matPl = dot(self.P, self.currentValue)

        # updating VarYTilde and VarYbar
        varXh = sHrf.varXh
        snrls.computeVarYTildeOptWithRelVar(varXh, w)
        # print '         varYbar end =',snrls.varYbar.sum()
        # print '         varYtilde end =',snrls.varYtilde.sum()

    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)
        drifts = np.dot(self.P, self.finalValue)
        an = ['time', 'voxel']
        ad = {'time': arange(self.dataInput.ny) * self.dataInput.tr}
        outputs['drift_signal'] = xndarray(drifts, axes_names=an, axes_domains=ad,
                                           value_label='Delta BOLD')

        return outputs


def sampleDrift(varInvSigma_drift, ptLambdaY, dim):

    mean_drift = np.linalg.solve(varInvSigma_drift, ptLambdaY)
    choleskyInvSigma_drift = cholesky(varInvSigma_drift).transpose()
    drift = np.linalg.solve(choleskyInvSigma_drift, random.randn(dim))
    drift += mean_drift
    return drift


class DriftARSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Gibbs sampler of the parameters modelling the low frequency drift in the
    fMRI time course, in the case of AR noise
    """

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        # TODO : comment
        xmlio.XmlInitable.__init__(self)

        an = ['order', 'voxel']
        GibbsSamplerVariable.__init__(self, 'drift', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='PM LFD')

#        self.functionBasis = self.parameters[self.P_FUNCTION_BASIS]
#        self.dimDrift = self.parameters[self.P_POLYORDER] +1

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.dimDrift = self.dataInput.colP
        self.P = self.dataInput.lfdMat[0]  # for 1st session
        self.varPtLambdaP = zeros((self.dimDrift, self.dimDrift, self.nbVox),
                                  dtype=float)
        self.varPtLambdaYmP = zeros((self.dimDrift, self.nbVox), dtype=float)

    def updateNorm(self):
        self.norm = sum(
            diag(dot(self.currentValue.transpose(), self.currentValue)))

    def updateVarYmDrift(self):
        self.matPl = dot(self.P, self.currentValue)
        # print matPl.shape, self.dataInput.varMBY.shape
        for v in range(self.nbVox):
            self.varMBYPl[:, v] = self.dataInput.varMBY[
                :, v] - self.matPl[:, v]

    def computeVarYTilde(self, varNrls, varXh):
        for v in xrange(self.nbVox):
            repNRLv = repmat(varNrls[:, v], self.ny, 1)
            avjXjh = repNRLv * varXh
            self.varYTilde[:, v] = self.varMBYPl[:, v] - avjXjh.sum(axis=1)

    def checkAndSetInitValue(self, variables):
        smplVarDrift = self.get_variable('drift_var')
        smplVarDrift.checkAndSetInitValue(variables)
        VarDrift = smplVarDrift.currentValue
        # print 'nbscans=', self.ny, 'nbvox=', self.nbVox
        if self.currentValue == None:
            if not self.sampleFlag and self.dataInput.simulData != None:
                self.currentValue = self.dataInput.simulData.drift.lfd
                logger.debug('drift dimensions : %s',
                             str(self.currentValue.shape))
                logger.debug('self.dimDrift : %s', str(self.dimDrift))
                assert self.dimDrift == self.currentValue.shape[0]
            else:
                self.currentValue = sqrt(
                    VarDrift) * random.randn(self.dimDrift, self.nbVox)

        self.updateNorm()
        self.varMBYPl = zeros((self.ny, self.nbVox), dtype=float)
        self.updateVarYmDrift()

    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """
        # Precalculations and allocations :
        smplHRF = self.get_variable('hrf')
        smplHRF.checkAndSetInitValue(variables)
        smplNRLs = self.get_variable('nrl')
        smplNRLs.checkAndSetInitValue(variables)
        self.varMBYPl = zeros((self.ny, self.nbVox), dtype=float)
        self.varYTilde = zeros((self.ny, self.nbVox), dtype=float)
        self.updateVarYmDrift()
        self.computeVarYTilde(smplNRLs.currentValue, smplHRF.varXh)

    def sampleNextAlt(self, variables):
        varXh = self.get_variable('hrf').varXh
        varNRLs = self.get_variable('nrl').currentValue
        self.updateVarYmDrift()
        self.computeVarYTilde(varNRLs, varXh)

    def sampleNextInternal(self, variables):
        reps = self.get_variable('noise_var').currentValue
        smplVarARp = self.get_variable('noise_var')
        invAutoCorrNoise = smplVarARp.InvAutoCorrNoise
        varNrls = self.get_variable('nrl').currentValue
        smplVarh = self.get_variable('hrf')
        varXh = smplVarh.varXh
        eta = self.get_variable('drift_var').currentValue
        invSigma = empty((self.dimDrift, self.dimDrift), dtype=float)
        datamPredict = empty((self.ny), dtype=float)
        self.updateVarYmDrift()
        self.computeVarYTilde(varNrls, varXh)

        if 1:
            logger.debug('Computing PtDeltaP and PtDeltaY in C fashion')
            tSQSOptimIni = time.time()
            intensivecalc.computePtLambdaARModel(self.P,
                                                 invAutoCorrNoise,
                                                 varNrls,
                                                 varXh,
                                                 self.dataInput.varMBY,
                                                 reps,
                                                 self.varPtLambdaP,
                                                 self.varPtLambdaYmP)
            logger.debug('Computing PtDeltaP and PtDeltaY in C fashion'
                         ' done in %1.3f sec', time.time() - tSQSOptimIni)
            for v in xrange(self.nbVox):
                invSigma = eye(self.dimDrift, dtype=float) / \
                    eta + self.varPtLambdaP[:, :, v]
                self.currentValue[:, v] = sampleDrift(invSigma,
                                                      self.varPtLambdaYmP[
                                                          :, v],
                                                      self.dimDrift)
            logger.debug('Sampling drift in C fashion done in %1.3f sec',
                         time.time() - tSQSOptimIni)
        if 0:
            logger.debug('Computing PtDeltaP and PtDeltaY in Numpy fashion')
            tSQSOptimIni = time.time()
            for v in xrange(self.nbVox):
                projCovNoise = dot(
                    self.P.transpose(), invAutoCorrNoise[:, :, v]) / reps[v]
                invSigma = dot(projCovNoise, self.P)
                assert numpy.allclose(invSigma, self.varPtLambdaP[:, :, v])
                invSigma += eye(self.dimDrift, dtype=float) / eta

                repNRLv = repmat(varNrls[:, v], self.ny, 1)
                avjXjh = repNRLv * varXh
                datamPredict = self.dataInput.varMBY[:, v] - avjXjh.sum(axis=1)
                datamPredict = dot(projCovNoise, datamPredict)
                assert numpy.allclose(datamPredict, self.varPtLambdaYmP[:, v])
                self.currentValue[:, v] = sampleDrift(
                    invSigma, datamPredict, self.dimDrift)
            logger.debug('Sampling drift in Numpy fashion done in %1.3f sec',
                         time.time() - tSQSOptimIni)

        logger.debug('drift params :')
        logger.debug(
            numpy.array2string(self.currentValue, precision=3))
        self.updateNorm()
        self.updateVarYmDrift()
        self.computeVarYTilde(varNrls, varXh)

    def initOutputs2(self, outputs, nbROI=-1):
        self.initOutputObservables(outputs, nbROI)
        nbd = self.dimDrift
        voxShape = self.dataInput.finalShape
        shape = (nbd,) + voxShape
        axes_names = ['LFDorder', 'axial', 'coronal', 'sagital']
        outputs['pmLFD'] = xndarray(zeros(shape, dtype=float),
                                    axes_names=axes_names,
                                    value_label="pm LFD")

    def fillOutputs2(self, outputs, iROI=-1):
        self.fillOutputObservables(outputs, iROI)
        d = outputs['pmLFD'].data
        vm = self.dataInput.voxelMapping
        m = vm.getNdArrayMask()
        d[:, m[0], m[1], m[2]] = self.mean
        dl[:, :, m[0], m[1], m[2]] = self.meanLabels.swapaxes(0, 1)

    def finalizeSampling(self):
        # clean memory of temporary variables :
        del self.varPtLambdaP
        del self.varPtLambdaYmP
        del self.varMBYPl
        del self.varYTilde


class ETASampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Gibbs sampler of the variance of the Inverse Gamma prior used to
    regularise the estimation of the low frequency drift embedded
    in the fMRI time course
    """

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=np.array([1.0])):

        # TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self, 'driftVar', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVox = self.dataInput.nbVoxels

        if dataInput.simulData is not None and \
                isinstance(dataInput.simulData, BOLDModel):
            self.trueValue = np.array([dataInput.simulData.rdrift.var])
        if dataInput.simulData is not None and \
                isinstance(dataInput.simulData, list):  # multisession
            #self.trueValue = np.array([dataInput.simulData[0]['drift_var']])
            sd = dataInput.simulData
            # self.trueValue = np.array([np.array([ssd['drift_coeffs'] \
            #                                      for ssd in sd]).var()])

            # Better to recompute drift coeffs from drift signals
            # -> take into account amplitude factor
            P = self.get_variable('drift').P
            v = np.var([np.dot(P[s].T, sd[s]['drift'])
                        for s in xrange(len(sd))])
            self.trueValue = np.array([v])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

    def sampleNextInternal(self, variables):
        # TODO : comment
        smpldrift = self.get_variable('drift')
        alpha = .5 * (smpldrift.dimDrift * self.nbVox)
        beta = 2.0 / smpldrift.norm
        logger.info('eta ~ Ga(%1.3f,%1.3f)', alpha, beta)
        self.currentValue[0] = 1.0 / random.gamma(alpha, beta)


class ETASampler_MultiSess(ETASampler):

    def linkToData(self, dataInput):
        ETASampler.linkToData(self, dataInput)
        self.nbSessions = dataInput.nbSessions

    def sampleNextInternal(self, variables):

        smpldrift = self.get_variable('drift')
        alpha = .5 * (self.nbSessions * smpldrift.dimDrift * self.nbVox - 1)
        beta_d = 0.5 * smpldrift.norm
        logger.info('eta ~ Ga(%1.3f,%1.3f)', alpha, beta_d)
        self.currentValue[0] = 1.0 / random.gamma(alpha, 1 / beta_d)

        samples = 1.0 / random.gamma(alpha, 1 / beta_d, 1000)
        logger.debug('true var drift : %f\n'
                     'm_theo=%f, v_theo=%f\n'
                     'm_empir=%f, v_empir=%f',
                     self.trueValue,
                     beta_d / (alpha - 1), beta_d ** 2 /
                     ((alpha - 1) ** 2 * (alpha - 2)),
                     samples.mean(), samples.var())
