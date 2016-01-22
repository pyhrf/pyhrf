# -*- coding: utf-8 -*-

import logging

import pyhrf
import numpy as np
from numpy.testing import assert_almost_equal

from pyhrf import xmlio
from pyhrf.ndarray import xndarray, stack_cuboids

from pyhrf.jde.models import (WN_BiG_Drift_BOLDSamplerInput,
                              GSDefaultCallbackHandler)
from pyhrf.jde.samplerbase import GibbsSampler, GibbsSamplerVariable

from pyhrf.boldsynth.hrf import genGaussianSmoothHRF, getCanoHRF
from pyhrf.jde.intensivecalc import asl_compute_y_tilde
from pyhrf.jde.intensivecalc import sample_potts

logger = logging.getLogger(__name__)


def b():
    raise Exception()


class NoiseVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'],
                                      value_label='PM Noise Var')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny

        # Do some allocations :

        if self.dataInput.simulData is not None:
            assert isinstance(self.dataInput.simulData[0], dict)
            sd = dataInput.simulData[0]
            if sd.has_key('noise'):
                self.trueValue = sd['noise'].var(0)

        if self.trueValue is not None and self.trueValue.size == 1:
            self.trueValue = self.trueValue * np.ones(self.nbVoxels)

    def checkAndSetInitValue(self, variables):
        if self.useTrueValue:
            if self.trueValue is not None:
                logger.info('Use true noise variance value ...')
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('True noise variance have to be used but '
                                'none defined.')

        if self.currentValue is None:
            self.currentValue = 0.1 * self.dataInput.varData

    def compute_y_tilde(self):
        logger.info('NoiseVarianceSampler.compute_y_tilde ...')

        sumaXh = self.samplerEngine.get_variable('brl').sumBXResp
        sumcXg = self.samplerEngine.get_variable('prl').sumBXResp
        Pl = self.samplerEngine.get_variable('drift_coeff').Pl
        wa = self.samplerEngine.get_variable('perf_baseline').wa

        y = self.dataInput.varMBY

        return y - sumaXh - sumcXg - Pl - wa

    def sampleNextInternal(self, variables):
        y_tilde = self.compute_y_tilde()

        beta = (y_tilde * y_tilde).sum(0) / 2

        # gammaSamples = np.random.gamma(0.5*(self.ny - self.dataInput.colP +1)-1, 1,
        #                            self.nbVoxels)

        gammaSamples = np.random.gamma((self.ny - 1.) / 2, 1, self.nbVoxels)

        np.divide(beta, gammaSamples, self.currentValue)


class DriftVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([1.0]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'drift_var', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      sampleFlag=do_sampling)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, list):  # multisession
                self.trueValue = np.array(
                    [dataInput.simulData[0]['drift_var']])
            # one session (old case)
            elif isinstance(dataInput.simulData, dict):
                self.trueValue = np.array([dataInput.simulData['drift_var']])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

    def sampleNextInternal(self, variables):

        smpldrift = self.samplerEngine.get_variable('drift_coeff')
#        print 'shape dimDrift...', smpldrift.dimDrift
#        print 'norm Drift', smpldrift.norm
        alpha = .5 * (smpldrift.dimDrift * self.nbVoxels - 1)
        beta = 2.0 / smpldrift.norm
        logger.info('eta ~ Ga(%1.3f,%1.3f)' % (alpha, beta))
        self.currentValue[0] = 1.0 / np.random.gamma(alpha, beta)

        if 1:
            beta = 1 / beta
            if self.trueValue is not None:
                logger.info('true var drift : %f', self.trueValue)
            logger.info('m_theo=%f, v_theo=%f', beta / (alpha - 1),
                        beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2)))
            samples = 1.0 / np.random.gamma(alpha, 1 / beta, 1000)
            logger.info('m_empir=%f, v_empir=%f',
                        samples.mean(), samples.var())
            logger.info('current sample: %f', self.currentValue[0])


class DriftCoeffSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'drift_coeff', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      axes_names=['lfd_order', 'voxel'],
                                      sampleFlag=do_sampling)

        self.final_signal = None
        self.true_value_signal = None

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.P = self.dataInput.lfdMat[0]
        self.dimDrift = self.P.shape[1]

        self.y_bar = np.zeros((self.ny, self.nbVoxels), dtype=np.float64)
        self.ones_Q_J = np.ones((self.dimDrift, self.nbVoxels))

        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, list):  # multisession
                self.trueValue = dataInput.simulData[0]['drift_coeffs']
            # one session (old case)
            elif isinstance(dataInput.simulData, dict):
                self.trueValue = dataInput.simulData['drift_coeffs']

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:

            # self.currentValue = np.random.randn(self.dimDrift,
            #                                    self.nbVoxels).astype(np.float64)
            self.currentValue = np.dot(self.P.T, self.dataInput.varMBY)

        self.Pl = np.dot(self.P, self.currentValue)
        self.updateNorm()

    def compute_y_tilde(self):

        sumaXh = self.samplerEngine.get_variable('brl').sumBXResp
        sumcXg = self.samplerEngine.get_variable('prl').sumBXResp
        wa = self.samplerEngine.get_variable('perf_baseline').wa

        y = self.dataInput.varMBY

        return y - sumaXh - sumcXg - wa

    def sampleNextInternal(self, variables):

        ytilde = self.compute_y_tilde()

        v_l = self.samplerEngine.get_variable('drift_var').currentValue
        v_b = self.samplerEngine.get_variable('noise_var').currentValue

        logger.debug('Noise vars :')
        logger.debug(v_b)

        for i in xrange(self.nbVoxels):

            v_lj = v_b[i] * v_l / (v_b[i] + v_l)
            mu_lj = v_lj / v_b[i] * np.dot(self.P.transpose(), ytilde[:, i])
            logger.debug('ivox=%d, v_lj=%f, std_lj=%f mu_lj=%s',
                         i, v_lj, v_lj ** .5, mu_lj)

            self.currentValue[:, i] = (np.random.randn(self.dimDrift) *
                                       v_lj ** .5) + mu_lj

            logger.debug('v_l : %f', v_l)

            # if i == 15:
            #    raise Exception()
        logger.debug('drift params :')
        logger.debug(self.currentValue)

        if 0:
            inv_vars_l = (1 / v_b + 1 / v_l) * self.ones_Q_J
            mu_l = 1 / inv_vars_l * np.dot(self.P.transpose(), ytilde)

            logger.debug('vars_l :')
            logger.debug(1 / inv_vars_l)

            logger.debug('mu_l :')
            logger.debug(mu_l)

            cur_val = np.random.normal(mu_l, 1 / inv_vars_l)

            logger.debug('drift params (alt) :')
            logger.debug(cur_val)

            #assert np.allclose(cur_val, self.currentValue)

        self.updateNorm()
        self.Pl = np.dot(self.P, self.currentValue)

    def updateNorm(self):

        self.norm = (self.currentValue * self.currentValue).sum()

        if self.trueValue is not None:
            logger.info('cur drift norm: %f', self.norm)
            logger.info('true drift norm: %f', (self.trueValue *
                                                self.trueValue).sum())

    def get_final_value(self):
        if self.final_signal is None:
            self.final_signal = np.dot(self.P, self.finalValue)

        return self.final_signal

    def get_true_value(self):
        if self.true_value_signal is None:
            self.true_value_signal = np.dot(self.P, self.trueValue)

        return self.true_value_signal

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        tol = atol
        tv = self.get_true_value()
        fv = self.get_final_value()
        err = (((fv - tv) ** 2).sum(0) / (tv ** 2).sum(0)) ** .5
        # logger.info('Drift estimation error wrt ground truth:\n%s',
        #              str(err))
        return ['voxel'], err < tol

    # def getOutputs(self):
    #     outputs = GibbsSamplerVariable.getOutputs(self)
    #     drift_signal = np.dot(self.P, self.finalValue)
    #     an = ['time','voxel']
    #     c = xndarray(drift_signal, axes_names=an, value_label='Delta ASL')

    #     if self.trueValue is not None:
    #         tv = self.get_true_value()
    #         c_true = xndarray(tv, axes_names=an)
    #         c = stack_cuboids([c, c_true], axis='type', domain=['estim', 'true'],
    #                           axis_pos='last')

    #     outputs['drift_signal_pm'] = c

    #     return outputs


def compute_StS_StY(rls, v_b, mx, mxtx, ybar, rlrl, yaj, ajak_vb):
    """ yaj and ajak_vb are only used to store intermediate quantities, they're
    not inputs.
    """
    nb_col_X = mx.shape[2]
    nb_conditions = mxtx.shape[0]
    varDeltaS = np.zeros((nb_col_X, nb_col_X), dtype=float)
    varDeltaY = np.zeros((nb_col_X), dtype=float)

    for j in xrange(nb_conditions):
        np.divide(ybar, v_b, yaj)
        yaj *= rls[j, :]
        varDeltaY += np.dot(mx[j, :, :].T, yaj.sum(1))

        for k in xrange(nb_conditions):
            np.divide(rlrl[j, k, :], v_b, ajak_vb)
            logger.debug('ajak/rb :')
            logger.debug(ajak_vb)
            varDeltaS += ajak_vb.sum() * mxtx[j, k, :, :]

        return (varDeltaS, varDeltaY)


class ResponseSampler(GibbsSamplerVariable):
    """
    Generic parent class to perfusion response & BOLD response samplers
    """

    def __init__(self, name, response_level_name, variance_name, smooth_order=2,
                 zero_constraint=True, duration=25., normalise=1., val_ini=None,
                 do_sampling=True,
                 use_true_value=False):

        self.response_level_name = response_level_name
        self.var_name = variance_name
        an = ['time']
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='Delta signal')

        self.normalise = normalise
        self.zc = zero_constraint
        self.duration = duration
        self.varR = None
        self.derivOrder = smooth_order

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt

        # print dataInput.simulData
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            # 1st voxel:
            self.trueValue = dataInput.simulData[0][self.name][:, 0]

        self.yBj = np.zeros((self.ny, self.nbVoxels), dtype=float)
        self.BjBk_vb = np.zeros((self.nbVoxels), dtype=float)

        self.ytilde = np.zeros((self.ny, self.nbVoxels), dtype=float)

        # self.track_sampled_quantity(self.ytilde, self.name + '_ytilde',
        #                             axes_names=['time', 'voxel'])

    def checkAndSetInitValue(self, variables):

        _, self.varR = genGaussianSmoothHRF(self.zc,
                                            self.hrfLength,
                                            self.eventdt, 1.,
                                            order=self.derivOrder)
        hrfValIni = None
        if self.useTrueValue:
            if self.trueValue is not None:
                hrfValIni = self.trueValue[:]
            else:
                raise Exception('Needed a true value for hrf init but '
                                'None defined')

        if hrfValIni is None:
            logger.debug('self.duration=%d, self.eventdt=%1.2f',
                         self.duration, self.eventdt)

            logger.debug('genCanoHRF -> dur=%f, dt=%f',
                         self.duration, self.eventdt)
            dt = self.eventdt
            hIni = getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

            hrfValIni = np.array(hIni)
            logger.debug('genCanoHRF -> shape h: %s',
                         str(hrfValIni.shape))

        if self.zc:
            logger.info('hrf zero constraint On')
            hrfValIni = hrfValIni[1:(self.hrfLength - 1)]

        logger.info('hrfValIni:' + str(hrfValIni.shape))
        logger.debug(hrfValIni)
        logger.info('self.hrfLength:' + str(self.hrfLength))

        normHRF = (sum(hrfValIni ** 2)) ** (0.5)
        hrfValIni /= normHRF

        self.currentValue = hrfValIni[:]

        if self.zc:
            self.axes_domains['time'] = np.arange(len(self.currentValue) + 2) \
                * self.eventdt
        else:
            self.axes_domains['time'] = np.arange(len(self.currentValue)) \
                * self.eventdt

        logger.info('hrfValIni after ZC:' + str(self.currentValue.shape))
        logger.debug(self.currentValue)

        self.updateNorm()
        self.updateXResp()

    def calcXResp(self, resp, stackX=None):
        if stackX is None:
            stackX = self.get_stackX()
        stackXResp = np.dot(stackX, resp)
        return np.reshape(stackXResp, (self.nbConditions, self.ny)).transpose()

    def updateXResp(self):
        self.varXResp = self.calcXResp(self.currentValue)

    def updateNorm(self):
        self.norm = sum(self.currentValue ** 2.0) ** 0.5

    def get_stackX():
        raise NotImplementedError()

    def get_mat_X(self):
        raise NotImplementedError()

    def get_rlrl(self):
        raise NotImplementedError()

    def get_mat_XtX(self):
        raise NotImplementedError()

    def get_ybar(self):
        raise NotImplementedError()

    def computeYTilde(self):
        raise NotImplementedError()

    def sampleNextInternal(self, variables):
        rl_sampler = self.samplerEngine.get_variable(self.response_level_name)
        rl = rl_sampler.currentValue
        rlrl = rl_sampler.rr

        noise_var = self.samplerEngine.get_variable('noise_var').currentValue

        mx = self.get_mat_X()
        mxtx = self.get_mat_XtX()

        self.ytilde[:] = self.computeYTilde()

        StS, StY = compute_StS_StY(rl, noise_var, mx, mxtx, self.ytilde, rlrl,
                                   self.yBj, self.BjBk_vb)

        v_resp = self.samplerEngine.get_variable(self.var_name).currentValue

        varInvSigma = StS + self.nbVoxels * self.varR / v_resp
        mean_h = np.linalg.solve(varInvSigma, StY)
        resp = np.random.multivariate_normal(
            mean_h, np.linalg.inv(varInvSigma))
        if self.normalise:
            norm = (resp ** 2).sum() ** .5
            resp /= norm
            #rl_sampler.currentValue *= norm
        self.currentValue = resp

        self.updateXResp()
        self.updateNorm()

        rl_sampler.computeVarYTildeOpt()

    def setFinalValue(self):

        fv = self.mean  # /self.normalise
        if self.zc:
            # Append and prepend zeros
            self.finalValue = np.concatenate(([0], fv, [0]))
            self.error = np.concatenate(([0], self.error, [0]))
            if self.meanHistory is not None:
                nbIt = len(self.obsHistoryIts)

                self.meanHistory = np.hstack((np.hstack((np.zeros((nbIt, 1)),
                                                         self.meanHistory)),
                                              np.zeros((nbIt, 1))))

            if self.smplHistory is not None:
                nbIt = len(self.smplHistoryIts)
                self.smplHistory = np.hstack((np.hstack((np.zeros((nbIt, 1)),
                                                         self.smplHistory)),
                                              np.zeros((nbIt, 1))))
        else:
            self.finalValue = fv

        logger.info('%s finalValue :', self.name)
        logger.info(self.finalValue)


class BOLDResponseSampler(ResponseSampler, xmlio.XmlInitable):

    def __init__(self, smooth_order=2, zero_constraint=True, duration=25.,
                 normalise=1., val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseSampler.__init__(self, 'brf', 'brl', 'brf_var', smooth_order,
                                 zero_constraint, duration, normalise, val_ini,
                                 do_sampling, use_true_value)

    def get_stackX(self):
        return self.dataInput.stackX

    def get_mat_X(self):
        return self.dataInput.varX

    def get_mat_XtX(self):
        return self.dataInput.matXtX

    def computeYTilde(self):
        """ y - \sum cWXg - Pl - wa """

        sumcXg = self.samplerEngine.get_variable('prl').sumBXResp
        Pl = self.samplerEngine.get_variable('drift_coeff').Pl
        wa = self.samplerEngine.get_variable('perf_baseline').wa
        y = self.dataInput.varMBY

        return y - sumcXg - Pl - wa


class PerfResponseSampler(ResponseSampler, xmlio.XmlInitable):

    def __init__(self, smooth_order=2, zero_constraint=True, duration=25.,
                 normalise=1., val_ini=None, do_sampling=True,
                 use_true_value=False, diff_res=True):
        """
        *diff_res*: if True then partial residuals (ytilde values) are differenced
        so that sampling is the same as for BRF.
        It avoids bad tail estimation, because of bad condionning of WtXtXW ?
        """
        xmlio.XmlInitable.__init__(self)
        self.diff_res = diff_res
        ResponseSampler.__init__(self, 'prf', 'prl', 'prf_var', smooth_order,
                                 zero_constraint, duration, normalise, val_ini,
                                 do_sampling, use_true_value)

    def get_stackX(self):
        return self.dataInput.stackWX

    def get_mat_X(self):
        if not self.diff_res:
            return self.dataInput.WX
        else:
            return self.dataInput.varX

    def get_mat_XtX(self):
        if not self.diff_res:
            return self.dataInput.WXtWX
        else:
            return self.dataInput.matXtX

    def computeYTilde(self):
        """ y - \sum aXh - Pl - wa """

        brf_sampler = self.get_variable('brf')
        brl_sampler = self.get_variable('brl')
        sumaXh = brl_sampler.sumBXResp

        drift_sampler = self.samplerEngine.get_variable('drift_coeff')
        Pl = drift_sampler.Pl
        perf_baseline_sampler = self.samplerEngine.get_variable(
            'perf_baseline')
        wa = perf_baseline_sampler.wa
        y = self.dataInput.varMBY

        res = y - sumaXh - Pl - wa

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
                    brl_sampler.useTrueValue and brf_sampler.useTrueValue:
                logger.debug('check sum_m a_j^m X^m h = '
                             'true bold stim induced component')
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])

            if not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                logger.debug('check Pl = true drift')
                assert_almost_equal(Pl, sd['drift'])

            if not perf_baseline_sampler.sampleFlag and \
                    perf_baseline_sampler.useTrueValue:
                logger.debug('check wa = true perf baseline')
                assert_almost_equal(wa, np.dot(self.dataInput.W,
                                               sd['perf_baseline']))

            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
                    brl_sampler.useTrueValue and brf_sampler.useTrueValue and \
                    not drift_sampler.sampleFlag and drift_sampler.useTrueValue and\
                    not perf_baseline_sampler.sampleFlag and \
                    perf_baseline_sampler.useTrueValue:
                logger.debug('check residuals = true perf sig + noise')
                perf = np.dot(
                    self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(res, perf + sd['noise'])

        if not self.diff_res:
            return res
        else:
            return np.dot(self.dataInput.W, res)


class ResponseLevelSampler(GibbsSamplerVariable):

    def __init__(self, name, response_name, mixture_name,
                 val_ini=None, do_sampling=True,
                 use_true_value=False):

        self.response_name = response_name
        self.mixture_name = mixture_name
        an = ['condition', 'voxel']
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='amplitude')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = dataInput.simulData[0].get(self.name + 's', None)

        # Precalculations and allocations :
        self.varYtilde = np.zeros((self.ny, self.nbVoxels), dtype=np.float64)
        self.BXResp = np.empty((self.nbVoxels, self.ny,
                                self.nbConditions), dtype=float)
        self.sumBXResp = np.zeros((self.ny, self.nbVoxels), dtype=float)

        self.rr = np.zeros((self.nbConditions, self.nbConditions, self.nbVoxels),
                           dtype=float)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:
            #rnd = np.random.rand(self.nbConditions, self.nbVoxels)
            #self.currentValue = (rnd.astype(np.float64) - .5 ) * 10
            self.currentValue = np.zeros((self.nbConditions,
                                          self.nbVoxels), dtype=np.float64)
            yptp = self.dataInput.varMBY.ptp(0)
            for j in xrange(self.nbConditions):
                self.currentValue[j, :] = yptp * .1

    def samplingWarmUp(self, variables):
        """
        """

        self.response_sampler = self.samplerEngine.get_variable(
            self.response_name)
        self.mixture_sampler = self.samplerEngine.get_variable(
            self.mixture_name)

        self.meanApost = np.zeros(
            (self.nbConditions, self.nbVoxels), dtype=float)
        self.varApost = np.zeros(
            (self.nbConditions, self.nbVoxels), dtype=float)

        self.labeled_vars = np.zeros((self.nbConditions, self.nbVoxels))
        self.labeled_means = np.zeros((self.nbConditions, self.nbVoxels))

        # Intermediate quantities to compute online variance
        # conditional to the MAP label
        self.nbClasses = self.get_variable('label').nbClasses
        self.final_labeled_vars_cumul = np.zeros((self.nbClasses,
                                                  self.nbConditions,
                                                  self.nbVoxels))

        self.final_labeled_vars_cumul2 = np.zeros((self.nbClasses,
                                                   self.nbConditions,
                                                   self.nbVoxels),
                                                  dtype=np.float64)

        self.final_labeled_vars_it_count = np.zeros((self.nbClasses,
                                                     self.nbConditions,
                                                     self.nbVoxels),
                                                    dtype=np.int32)
        self.iteration = 0

        self.computeRR()

    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)
        labels = self.get_variable('label').currentValue
        for j in xrange(self.nbConditions):
            for iclass in xrange(self.nbClasses):
                class_mask = np.where(labels[j] == iclass)
                self.final_labeled_vars_it_count[iclass, j, class_mask] += 1
                self.final_labeled_vars_cumul[iclass, j, class_mask] += \
                    self.currentValue[j, class_mask]
                self.final_labeled_vars_cumul2[iclass, j, class_mask] += \
                    (self.currentValue[j, class_mask] -
                     self.final_labeled_vars_cumul[iclass, j, class_mask] /
                     self.final_labeled_vars_it_count[iclass, j, class_mask]) ** 2

    def sampleNextInternal(self, variables):

        labels = self.samplerEngine.get_variable('label').currentValue
        v_b = self.samplerEngine.get_variable('noise_var').currentValue

        Xresp = self.response_sampler.varXResp

        gTg = np.diag(np.dot(Xresp.transpose(), Xresp))

        mixt_vars = self.mixture_sampler.get_current_vars()
        mixt_means = self.mixture_sampler.get_current_means()

        ytilde = self.computeVarYTildeOpt()

        for iclass in xrange(len(mixt_vars)):
            v = mixt_vars[iclass]
            m = mixt_means[iclass]
            for j in xrange(self.nbConditions):
                class_mask = np.where(labels[j] == iclass)
                self.labeled_vars[j, class_mask[0]] = v[j]
                self.labeled_means[j, class_mask[0]] = m[j]

        for j in xrange(self.nbConditions):
            Xresp_m = Xresp[:, j]
            ytilde_m = ytilde + (self.currentValue[np.newaxis, j, :] *
                                 Xresp_m[:, np.newaxis])
            v_q_j = self.labeled_vars[j]
            m_q_j = self.labeled_means[j]
            self.varApost[j, :] = (v_b * v_q_j) / (gTg[j] * v_q_j + v_b)
            self.meanApost[j, :] = self.varApost[j, :] * \
                (np.dot(Xresp_m.T, ytilde_m) / v_b + m_q_j / v_q_j)

            rnd = np.random.randn(self.nbVoxels)
            self.currentValue[j, :] = rnd * self.varApost[j, :] ** .5 + \
                self.meanApost[j, :]
            ytilde = self.computeVarYTildeOpt()

            # b()

        self.computeRR()

    def computeVarYTildeOpt(self):
        raise NotImplementedError()

    def computeRR(self):
        # aa[m,n,:] == aa[n,m,:] -> nb ops can be /2
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                np.multiply(self.currentValue[j, :], self.currentValue[k, :],
                            self.rr[j, k, :])

    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)

        # TODO: put this in finalizeSampling?
        final_labels = self.get_variable('label').get_MAP_labels()
        final_labeled_vars = np.zeros((self.nbConditions,
                                       self.nbVoxels))
        for j in xrange(self.nbConditions):
            for iclass in xrange(self.nbClasses):
                class_mask = np.where(final_labels[j] == iclass)
                final_labeled_vars[j, class_mask] = \
                    self.final_labeled_vars_cumul2[iclass, j, class_mask] /   \
                    self.final_labeled_vars_it_count[iclass, j, class_mask]

        outputs[self.name + '_mcmc_MAPlab_var'] = \
            xndarray(final_labeled_vars.astype(np.float32),
                     axes_names=['condition', 'voxel'])

        return outputs


class BOLDResponseLevelSampler(ResponseLevelSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseLevelSampler.__init__(self, 'brl', 'brf', 'bold_mixt_params',
                                      val_ini, do_sampling, use_true_value)

    def samplingWarmUp(self, v):
        ResponseLevelSampler.samplingWarmUp(self, v)
        # BOLD response sampler is in charge of initialising ytilde of prls:
        # -> update_perf=True
        self.computeVarYTildeOpt(update_perf=True)

    def computeVarYTildeOpt(self, update_perf=False):
        """
        if update_perf is True then also update sumcXg and prl.ytilde
        update_perf should only be used at init of variable values.
        """

        logger.info(' BOLDResp.computeVarYTildeOpt(update_perf=%s) ...',
                    str(update_perf))

        brf_sampler = self.get_variable('brf')
        Xh = brf_sampler.varXResp
        sumaXh = self.sumBXResp

        prl_sampler = self.samplerEngine.get_variable('prl')
        prls = prl_sampler.currentValue
        sumcXg = prl_sampler.sumBXResp
        prf_sampler = self.samplerEngine.get_variable('prf')
        WXg = prf_sampler.varXResp

        compute_bold = 1

        if update_perf:
            compute_perf = 1
        else:
            compute_perf = 0

        asl_compute_y_tilde(Xh, WXg, self.currentValue, prls,
                            self.dataInput.varMBY, self.varYtilde,
                            sumaXh, sumcXg, compute_bold, compute_perf)

        if update_perf:
            ytilde_perf = prl_sampler.varYtilde
            asl_compute_y_tilde(Xh, WXg, self.currentValue, prls,
                                self.dataInput.varMBY, ytilde_perf,
                                sumaXh, sumcXg, 0, 0)

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not self.sampleFlag and not brf_sampler.sampleFlag and\
                    self.useTrueValue and brf_sampler.useTrueValue:
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])
            if not prl_sampler.sampleFlag and not prf_sampler.sampleFlag and\
                    prl_sampler.useTrueValue and not prf_sampler.useTrueValue:
                perf = np.dot(
                    self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(sumcXg, perf)

        logger.debug('varYtilde %s', str(self.varYtilde.shape))
        logger.debug(self.varYtilde)

        Pl = self.samplerEngine.get_variable('drift_coeff').Pl
        wa = self.samplerEngine.get_variable('perf_baseline').wa

        return self.varYtilde - Pl - wa

    def getOutputs(self):
        outputs = ResponseLevelSampler.getOutputs(self)

        axes_names = ['voxel']
        roi_lab_vol = np.zeros(self.nbVoxels, dtype=np.int32) + \
            self.dataInput.roiId
        outputs['roi_mapping'] = xndarray(roi_lab_vol, axes_names=axes_names,
                                          value_label='ROI')
        return outputs


class PerfResponseLevelSampler(ResponseLevelSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseLevelSampler.__init__(self, 'prl', 'prf', 'perf_mixt_params',
                                      val_ini, do_sampling, use_true_value)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:
            #rnd = np.random.rand(self.nbConditions, self.nbVoxels)
            #self.currentValue = (rnd.astype(np.float64) - .5 ) * 10
            self.currentValue = np.zeros((self.nbConditions,
                                          self.nbVoxels), dtype=np.float64)

            perf_baseline = (self.dataInput.varMBY *
                             self.dataInput.w[:, np.newaxis]).mean(0)

            for j in xrange(self.nbConditions):
                self.currentValue[j, :] = perf_baseline * .1

    def computeVarYTildeOpt(self):
        """
        """

        logger.info(' PerfRespLevel.computeVarYTildeOpt() ...')

        brf_sampler = self.samplerEngine.get_variable('brf')
        Xh = brf_sampler.varXResp
        brl_sampler = self.samplerEngine.get_variable('brl')
        sumaXh = brl_sampler.sumBXResp
        brls = brl_sampler.currentValue

        prf_sampler = self.samplerEngine.get_variable('prf')
        WXg = prf_sampler.varXResp
        sumcXg = self.sumBXResp

        compute_bold = 0
        compute_perf = 1

        asl_compute_y_tilde(Xh, WXg, brls, self.currentValue,
                            self.dataInput.varMBY, self.varYtilde,
                            sumaXh, sumcXg, compute_bold, compute_perf)

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
                    brl_sampler.useTrueValue and brf_sampler.useTrueValue:
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])
            if not self.sampleFlag and not prf_sampler.sampleFlag and\
                    self.useTrueValue and not prf_sampler.useTrueValue:
                perf = np.dot(
                    self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(sumcXg, perf)

        logger.debug('varYtilde %s', str(self.varYtilde.shape))
        logger.debug(self.varYtilde)

        Pl = self.samplerEngine.get_variable('drift_coeff').Pl
        wa = self.samplerEngine.get_variable('perf_baseline').wa

        return self.varYtilde - Pl - wa


class ResponseVarianceSampler(GibbsSamplerVariable):

    def __init__(self, name, response_name,
                 val_ini=np.array([0.001]), do_sampling=True,
                 use_true_value=False):
        self.response_name = response_name
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      value_label='Var ' + self.response_name)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            if dataInput.simulData[0].has_key(self.name):
                self.trueValue = np.array([dataInput.simulData[0][self.name]])

    def checkAndSetInitValue(self, v):
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '
                                'None defined' % ResponseVarianceSampler)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

    def sampleNextInternal(self, v):
        resp_sampler = self.samplerEngine.get_variable(self.response_name)
        R = resp_sampler.varR
        resp = resp_sampler.currentValue

        alpha = (len(resp) * self.nbVoxels - 1) / 2
        beta = np.dot(np.dot(resp.T, R), resp) / 2

        self.currentValue[0] = 1 / np.random.gamma(alpha, 1 / beta)


class BOLDResponseVarianceSampler(ResponseVarianceSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([0.001]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)

        ResponseVarianceSampler.__init__(self, 'brf_var', 'brf',
                                         val_ini, do_sampling, use_true_value)


class PerfResponseVarianceSampler(ResponseVarianceSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([0.001]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)

        ResponseVarianceSampler.__init__(self, 'prf_var', 'prf',
                                         val_ini, do_sampling, use_true_value)


class LabelSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    L_CI = 0
    L_CA = 1

    CLASSES = np.array([L_CI, L_CA], dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)

        an = ['condition', 'voxel']
        GibbsSamplerVariable.__init__(self, 'label', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an)

        self.nbClasses = len(self.CLASSES)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        self.cardClass = np.zeros(
            (self.nbClasses, self.nbConditions), dtype=int)
        self.voxIdx = [range(self.nbConditions)
                       for c in xrange(self.nbClasses)]

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = dataInput.simulData[0]['labels'].astype(np.int32)

    def checkAndSetInitValue(self, variables):
        logger.info('LabelSampler.checkAndSetInitLabels ...')

        # Generate default labels if necessary :
        if self.useTrueValue:
            if self.trueValue is not None:
                logger.info('Use true label values ...')
                self.currentValue = self.trueValue[:]
            else:
                raise Exception(
                    'True labels have to be used but none defined.')

        if self.currentValue is None:

            self.currentValue = np.zeros((self.nbConditions, self.nbVoxels),
                                         dtype=np.int32)

            for j in xrange(self.nbConditions):
                self.currentValue[j, :] = np.random.binomial(
                    1, .9, self.nbVoxels)

        self.beta = np.zeros((self.nbConditions), dtype=np.float64) + .7

        self.countLabels()

    def countLabels(self):
        logger.info('LabelSampler.countLabels ...')
        labs = self.currentValue
        for j in xrange(self.nbConditions):
            for c in xrange(self.nbClasses):
                self.voxIdx[c][j] = np.where(labs[j, :] == self.CLASSES[c])[0]
                self.cardClass[c, j] = len(self.voxIdx[c][j])
                logger.debug('Nb vox in C%d for cond %d : %d',
                             c, j, self.cardClass[c, j])

            if self.cardClass[:, j].sum() != self.nbVoxels:
                raise Exception('cardClass[cond=%d]=%d != nbVox=%d'
                                % (j, self.cardClass[:, j].sum(), self.nbVoxels))

    def samplingWarmUp(self, v):
        self.iteration = 0
        self.current_ext_field = np.zeros((self.nbClasses, self.nbConditions,
                                           self.nbVoxels), dtype=np.float64)

    def compute_ext_field(self):
        bold_mixtp_sampler = self.samplerEngine.get_variable(
            'bold_mixt_params')
        asl_mixtp_sampler = self.samplerEngine.get_variable('perf_mixt_params')

        v = bold_mixtp_sampler.get_current_vars()
        rho = asl_mixtp_sampler.get_current_vars()

        mu = bold_mixtp_sampler.get_current_means()
        eta = asl_mixtp_sampler.get_current_means()

        a = self.samplerEngine.get_variable('brl').currentValue
        c = self.samplerEngine.get_variable('prl').currentValue

        for k in xrange(self.nbClasses):
            for j in xrange(self.nbConditions):
                e = .5 * (-np.log2(v[k, j] * rho[k, j]) -
                          (a[j, :] - mu[k, j]) ** 2 / v[k, j] -
                          (c[j, :] - eta[k, j]) ** 2 / rho[k, j])
                self.current_ext_field[k, j, :] = e

    def sampleNextInternal(self, v):

        neighbours = self.dataInput.neighboursIndexes

        beta = self.beta

        voxOrder = np.random.permutation(self.nbVoxels)

        self.compute_ext_field()

        rnd = np.random.rand(*self.currentValue.shape).astype(np.float64)

        sample_potts(voxOrder.astype(np.int32), neighbours.astype(np.int32),
                     self.current_ext_field, beta, rnd, self.currentValue,
                     self.iteration)

        self.countLabels()
        self.iteration += 1

    def get_MAP_labels(self):

        flabels = self.get_final_value()
        return flabels > .5  # TODO: use other threshold?


class MixtureParamsSampler(GibbsSamplerVariable):

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    L_CA = LabelSampler.L_CA
    L_CI = LabelSampler.L_CI

    def __init__(self, name, response_level_name,
                 val_ini=None, do_sampling=True,
                 use_true_value=False):

        self.response_level_name = response_level_name
        an = ['component', 'condition']
        ad = {'component': self.PARAMS_NAMES}

        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      axes_domains=ad)

    def get_true_values_from_simulation_dict(self):
        raise NotImplementedError()

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            cdefs = self.dataInput.simulData[0]['condition_defs']
            if hasattr(cdefs[0], 'bold_m_act'):
                tmca, tvca, tvci = \
                    self.get_true_values_from_simulation_cdefs(cdefs)
                self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                                          dtype=float)
                self.trueValue[self.I_MEAN_CA] = tmca
                self.trueValue[self.I_VAR_CA] = tvca
                self.trueValue[self.I_VAR_CI] = tvci

        self.rlCI = range(self.nbConditions)
        self.rlCA = range(self.nbConditions)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()[
                    :, :self.nbConditions]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            nc = self.nbConditions
            self.currentValue = np.zeros((self.NB_PARAMS, nc), dtype=float)
            # self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30.
            # self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.
            # self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.

            y = self.dataInput.varMBY
            self.currentValue[self.I_MEAN_CA, :] = y.ptp(0).mean() * .1
            self.currentValue[self.I_VAR_CA, :] = y.var(0).mean() * .1
            self.currentValue[self.I_VAR_CI, :] = y.var(0).mean() * .05

    def get_current_vars(self):
        return np.array([self.currentValue[self.I_VAR_CI],
                         self.currentValue[self.I_VAR_CA]])

    def get_current_means(self):
        return np.array([np.zeros(self.nbConditions),
                         self.currentValue[self.I_MEAN_CA]])

    def computeWithJeffreyPriors(self, j, cardCIj, cardCAj):

        # print 'sample hyper parameters with improper Jeffrey\'s priors ...'
        logger.info('cond %d - card CI = %d\n'
                    'cond %d - card CA = %d\n'
                    'cond %d - cur mean CA = %f\n',
                    j, cardCIj, j, cardCAj,
                    j, self.currentValue[self.I_MEAN_CA, j])
        if cardCAj > 0:
            logger.info('cond %d - rl CA: %f(v%f)[%f,%f]',
                        j, self.rlCA[j].mean(), self.rlCA[j].var(),
                        self.rlCA[j].min(), self.rlCA[j].max())
        if cardCIj > 0:
            logger.info('cond %d - rl CA: %f(v%f)[%f,%f]',
                        j, self.rlCA[j].mean(), self.rlCA[j].var(),
                        self.rlCA[j].min(), self.rlCA[j].max())

        if cardCIj > 1:
            nu0j = np.dot(self.rlCI[j], self.rlCI[j])
            varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj + 1) - 1, 2. / nu0j)
        else:
            varCIj = 1.0 / np.random.gamma(0.5, 0.2)

        if cardCAj > 1:
            rlC1Centered = self.rlCA[j] - self.currentValue[self.I_MEAN_CA, j]
            nu1j = np.dot(rlC1Centered, rlC1Centered)
            logger.info('varCA ~ InvGamma(%f, nu1j/2=%f)\n'
                        ' -> mean = %f',
                        0.5 * (cardCAj + 1) - 1, nu1j / 2.,
                        (nu1j / 2.) / (0.5 * (cardCAj + 1) - 1))
            varCAj = 1.0 / np.random.gamma(0.5 * (cardCAj + 1) - 1, 2. / nu1j)
            logger.info('varCAj (j=%d) : %f', j, varCAj)
            if varCAj <= 0.:
                logger.info('variance for class activ and condition %s '
                            'is negative or null: %f\n'
                            'nu1j = %f, 2. / nu1j = %f\n'
                            'cardCAj = %f, 0.5 * (cardCAj+1) - 1 = %f\n'
                            '-> setting it to almost 0.',
                            self.dataInput.cNames[j], varCAj,
                            nu1j, 2. / nu1j,
                            cardCAj, 0.5 * (cardCAj + 1) - 1)
                varCAj = 0.0001
            eta1j = np.mean(self.rlCA[j])
            meanCAj = np.random.normal(eta1j, (varCAj / cardCAj) ** 0.5)

        else:
            varCAj = 1.0 / np.random.gamma(.5, 2.)
            if cardCAj == 0:
                meanCAj = np.random.normal(5.0, varCAj ** 0.5)
            else:
                meanCAj = np.random.normal(self.rlCA[j], varCAj ** 0.5)

        logger.info('Sampled components - cond: %d\n'
                    'var CI = %f\n'
                    'mean CA = %f, var CA = %f', j, varCIj, meanCAj, varCAj)

        return varCIj, meanCAj, varCAj

    def sampleNextInternal(self, variables):

        rl_sampler = self.samplerEngine.get_variable(self.response_level_name)
        label_sampler = self.samplerEngine.get_variable('label')

        cardCA = label_sampler.cardClass[self.L_CA, :]
        cardCI = label_sampler.cardClass[self.L_CI, :]

        for j in xrange(self.nbConditions):
            vICI = label_sampler.voxIdx[self.L_CI][j]
            vICA = label_sampler.voxIdx[self.L_CA][j]
            self.rlCI[j] = rl_sampler.currentValue[j, vICI]
            self.rlCA[j] = rl_sampler.currentValue[j, vICA]

            varCIj, meanCAj, varCAj = self.computeWithJeffreyPriors(j,
                                                                    cardCI[j],
                                                                    cardCA[j])

            self.currentValue[self.I_VAR_CI, j] = varCIj
            self.currentValue[self.I_MEAN_CA, j] = meanCAj  # absolute(meanCAj)
            self.currentValue[self.I_VAR_CA, j] = varCAj

            logger.debug('varCI,%d=%f',
                         j, self.currentValue[self.I_VAR_CI, j])
            logger.debug('meanCA,%d=%f',
                         j, self.currentValue[self.I_MEAN_CA, j])
            logger.debug('varCA,%d = %f',
                         j, self.currentValue[self.I_VAR_CA, j])


class PerfMixtureSampler(MixtureParamsSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        MixtureParamsSampler.__init__(self, 'perf_mixt_params', 'prl',
                                      val_ini, do_sampling, use_true_value)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()[
                    :, :self.nbConditions]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            nc = self.nbConditions
            self.currentValue = np.zeros((self.NB_PARAMS, nc), dtype=float)

            perf_baseline = (self.dataInput.varMBY *
                             self.dataInput.w[:, np.newaxis]).mean(0)

            self.currentValue[self.I_MEAN_CA, :] = perf_baseline.mean() * .1
            self.currentValue[self.I_VAR_CA, :] = perf_baseline.var() * .1
            self.currentValue[self.I_VAR_CI, :] = perf_baseline.var() * .05

    def get_true_values_from_simulation_cdefs(self, cdefs):
        return np.array([c.perf_m_act for c in cdefs]), \
            np.array([c.perf_v_act for c in cdefs]), \
            np.array([c.perf_v_inact for c in cdefs])


class BOLDMixtureSampler(MixtureParamsSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        MixtureParamsSampler.__init__(self, 'bold_mixt_params', 'brl',
                                      val_ini, do_sampling, use_true_value)

    def get_true_values_from_simulation_cdefs(self, cdefs):
        return np.array([c.bold_m_act for c in cdefs]), \
            np.array([c.bold_v_act for c in cdefs]), \
            np.array([c.bold_v_inact for c in cdefs])


class PerfBaselineSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'perf_baseline', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'])

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.w = self.dataInput.w
        self.W = self.dataInput.W

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = self.dataInput.simulData[0][self.name][0, :]

        if self.trueValue is not None and np.isscalar(self.trueValue):
            self.trueValue = self.trueValue * np.ones(self.nbVoxels)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            self.currentValue = (self.dataInput.varMBY *
                                 self.dataInput.w[:, np.newaxis]).mean(0)

        self.wa = self.compute_wa()

    def compute_wa(self, a=None):
        if a is None:
            a = self.currentValue
        return self.w[:, np.newaxis] * a[np.newaxis, :]

    def compute_residuals(self):

        brf_sampler = self.samplerEngine.get_variable('brf')
        prf_sampler = self.samplerEngine.get_variable('prf')

        prl_sampler = self.samplerEngine.get_variable('prl')
        brl_sampler = self.samplerEngine.get_variable('brl')

        drift_sampler = self.samplerEngine.get_variable('drift_coeff')

        sumcXg = prl_sampler.sumBXResp
        sumaXh = brl_sampler.sumBXResp
        Pl = drift_sampler.Pl

        y = self.dataInput.varMBY

        res = y - sumcXg - sumaXh - Pl

        if self.dataInput.simulData is not None:
            # only for debugging when using artificial data
            if not brf_sampler.sampleFlag and brf_sampler.useTrueValue and\
                    not brl_sampler.sampleFlag and brl_sampler.useTrueValue and \
                    not prf_sampler.sampleFlag and prf_sampler.useTrueValue and \
                    not brf_sampler.sampleFlag and brf_sampler.useTrueValue and \
                    not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                sd = self.dataInput.simulData[0]
                osf = int(sd['tr'] / sd['dt'])
                perf = np.dot(
                    self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                true_res = sd['bold'] - sd['bold_stim_induced'][0:-1:osf] - \
                    perf - sd['drift']
                true_res = sd['noise'] + np.dot(self.dataInput.W,
                                                sd['perf_baseline'])
                assert_almost_equal(res, true_res)

        return res

    def sampleNextInternal(self, v):

        v_alpha = self.get_variable('perf_baseline_var').currentValue
        w = np.diag(self.dataInput.W)

        residuals = self.compute_residuals()
        v_b = self.samplerEngine.get_variable('noise_var').currentValue

        for i in xrange(self.nbVoxels):
            m_apost = ( np.dot(w.T, residuals[:, i]) ) /  \
                      (self.ny + v_b[i] / v_alpha)
            v_apost = (v_alpha * v_b[i]) / (self.ny * v_alpha + v_b[i])

            a = np.random.randn() * v_apost ** .5 + m_apost
            self.currentValue[i] = a
            self.wa[:, i] = self.w * a


class PerfBaselineVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'perf_baseline_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            assert isinstance(sd[0], dict)
            if sd[0].has_key(self.name):
                self.trueValue = np.array([sd[0][self.name]])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            self.currentValue = np.array([(self.dataInput.varMBY *
                                           self.dataInput.w[:, np.newaxis]).mean(0).var()])

    def sampleNextInternal(self, v):

        alpha = self.samplerEngine.get_variable('perf_baseline').currentValue

        a = (self.nbVoxels - 1) / 2.
        b = (alpha ** 2).sum() / 2

        self.currentValue[0] = 1 / np.random.gamma(a, 1 / b)


class WN_BiG_ASLSamplerInput(WN_BiG_Drift_BOLDSamplerInput):

    def __init__(self, data, dt, typeLFD, paramLFD, hrfZc, hrfDuration):
        self.idx_first_tag = data.get_extra_data('asl_first_tag_scan_idx', 0)
        WN_BiG_Drift_BOLDSamplerInput.__init__(self, data, dt, typeLFD,
                                               paramLFD, hrfZc, hrfDuration)

    def makePrecalculations(self):
        WN_BiG_Drift_BOLDSamplerInput.makePrecalculations(self)

        self.w = np.ones(self.ny)
        self.w[self.idx_first_tag::2] = -1
        self.W = np.diag(self.w)

        self.WX = np.zeros_like(self.varX)
        self.WXtWX = np.zeros_like(self.matXtX)
        self.stackWX = np.zeros_like(self.stackX)

        for j in xrange(self.nbConditions):
            self.WX[j, :, :] = np.dot(self.W, self.varX[j, :, :])
            self.stackWX[self.ny * j:self.ny * (j + 1), :] = self.WX[j, :, :]
            for k in xrange(self.nbConditions):
                self.WXtWX[j, k, :, :] = np.dot(self.WX[j, :, :].transpose(),
                                                self.WX[k, :, :])

    def cleanPrecalculations(self):
        WN_BiG_Drift_BOLDSamplerInput.cleanPrecalculations(self)
        del self.WXtWX


class ASLSampler(xmlio.XmlInitable, GibbsSampler):

    inputClass = WN_BiG_ASLSamplerInput

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        default_nb_its = 3
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        default_nb_its = 3000
        parametersToShow = ['nb_its', 'bold_response_levels',
                            'brf', 'brf_var', 'prf', 'prf_var']

    def __init__(self, nb_iterations=default_nb_its,
                 obs_hist_pace=-1., glob_obs_hist_pace=-1,
                 smpl_hist_pace=-1., burnin=.3,
                 callback=GSDefaultCallbackHandler(),
                 bold_response_levels=BOLDResponseLevelSampler(),
                 perf_response_levels=PerfResponseLevelSampler(),
                 labels=LabelSampler(), noise_var=NoiseVarianceSampler(),
                 brf=BOLDResponseSampler(), brf_var=BOLDResponseVarianceSampler(),
                 prf=PerfResponseSampler(), prf_var=PerfResponseVarianceSampler(),
                 bold_mixt_params=BOLDMixtureSampler(),
                 perf_mixt_params=PerfMixtureSampler(),
                 drift=DriftCoeffSampler(), drift_var=DriftVarianceSampler(),
                 perf_baseline=PerfBaselineSampler(),
                 perf_baseline_var=PerfBaselineVarianceSampler(),
                 check_final_value=None, output_fit=False):

        variables = [noise_var, brf, brf_var, prf, prf_var,
                     drift_var, drift, perf_response_levels,
                     bold_response_levels, perf_baseline, perf_baseline_var,
                     bold_mixt_params, perf_mixt_params, labels]

        nbIt = nb_iterations
        obsHistPace = obs_hist_pace
        globalObsHistPace = glob_obs_hist_pace
        smplHistPace = smpl_hist_pace
        nbSweeps = burnin

        check_ftval = check_final_value

        if obsHistPace > 0. and obsHistPace < 1:
            obsHistPace = max(1, int(round(nbIt * obsHistPace)))

        if globalObsHistPace > 0. and globalObsHistPace < 1:
            globalObsHistPace = max(1, int(round(nbIt * globalObsHistPace)))

        if smplHistPace > 0. and smplHistPace < 1.:
            smplHistPace = max(1, int(round(nbIt * smplHistPace)))

        if nbSweeps > 0. and nbSweeps < 1.:
            nbSweeps = int(round(nbIt * nbSweeps))

        callbackObj = GSDefaultCallbackHandler()
        self.cmp_ftval = False
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps,
                              callbackObj,
                              globalObsHistoryPace=globalObsHistPace,
                              check_ftval=check_ftval, output_fit=output_fit)

    def finalizeSampling(self):

        if self.cmp_ftval:
            msg = []
            for v in self.variables:

                if v.trueValue is None:
                    print 'Warning; no true val for %s' % v.name
                else:
                    fv = v.finalValue
                    tv = v.trueValue
                    # tol = .7
                    # if v.name == 'drift_coeff':
                    #     delta = np.abs(np.dot(v.P,
                    #                           v.finalValue - v.trueValue)).mean()
                    #     crit = detla > tol
                    # else:
                    #     delta = np.abs(v.finalValue - v.trueValue).mean()
                    #     crit = delta > tol
                    tol = .1
                    if self.dataInput.nbVoxels < 10:
                        if 'var' in v.name:
                            tol = 1.

                    if v.name == 'drift_coeff':
                        fv = np.dot(v.P, v.finalValue)
                        tv = np.dot(v.P, v.trueValue)
                        delta = np.abs((fv - tv) / np.maximum(tv, fv))
                    elif v.name == 'prf' or v.name == 'brf':
                        delta = (((v.finalValue - v.trueValue) ** 2).sum() /
                                 (v.trueValue ** 2).sum()) ** .5
                        tol = 0.05
                    elif v.name == 'label':
                        delta = (v.finalValue != v.trueValue).sum() * \
                            1. / v.nbVoxels
                    else:
                        # delta = (((v.finalValue - v.trueValue)**2).sum() / \
                        #         (v.trueValue**2).sum())**.5
                        delta = np.abs((v.finalValue - v.trueValue) /
                                       np.maximum(v.trueValue, v.finalValue))
                    crit = (delta > tol).any()

                    if crit:
                        m = "Final value of %s is not close to " \
                            "true value (mean delta=%f).\n" \
                            " Final value:\n %s\n True value:\n %s\n" \
                            % (v.name, delta.mean(), str(fv), str(tv))
                        msg.append(m)
                        #raise Exception(m)

            if len(msg) > 0:
                if 0:
                    raise Exception("\n".join(msg))
                else:
                    print "\n".join(msg)

    def computeFit(self):
        brf_sampler = self.get_variable('brf')
        prf_sampler = self.get_variable('prf')

        brl_sampler = self.get_variable('brl')
        prl_sampler = self.get_variable('prl')

        drift_sampler = self.get_variable('drift_coeff')
        perf_baseline_sampler = self.get_variable('perf_baseline')

        brf = brf_sampler.finalValue
        if brf is None:
            brf = brf_sampler.currentValue
        elif brf_sampler.zc:
            brf = brf[1:-1]
        vXh = brf_sampler.calcXResp(brf)  # base convolution

        prf = prf_sampler.finalValue
        if prf is None:
            prf = prf_sampler.currentValue
        elif prf_sampler.zc:
            prf = prf[1:-1]
        vXg = prf_sampler.calcXResp(prf)  # base convolution

        brl = brl_sampler.finalValue
        if brl is None:
            brl = brl_sampler.currentValue

        prl = prl_sampler.finalValue
        if prl is None:
            prl = prl_sampler.currentValue

        l = drift_sampler.finalValue
        p = drift_sampler.P
        if l is None:
            l = drift_sampler.currentValue

        perf_baseline = perf_baseline_sampler.finalValue
        if perf_baseline is None:
            perf_baseline = perf_baseline_sampler.currentValue
        wa = perf_baseline_sampler.compute_wa(perf_baseline)

        fit = np.dot(vXh, brl) + np.dot(vXg, prl) + np.dot(p, l) + wa

        return fit

    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)

        bf = outputs.pop('bold_fit', None)
        if bf is not None:
            cdict = bf.split('stype')
            signal = cdict['bold']
            fit = cdict['fit']

            # Grab fitted components
            brf_sampler = self.get_variable('brf')
            prf_sampler = self.get_variable('prf')

            brl_sampler = self.get_variable('brl')
            prl_sampler = self.get_variable('prl')

            drift_sampler = self.get_variable('drift_coeff')
            perf_baseline_sampler = self.get_variable('perf_baseline')

            brf = brf_sampler.finalValue
            if brf_sampler.zc:
                brf = brf[1:-1]
            vXh = brf_sampler.calcXResp(brf)  # base convolution
            #demod_stackX = brf_sampler.get_stackX()

            prf = prf_sampler.finalValue
            if prf_sampler.zc:
                prf = prf[1:-1]

            # base convolution:
            vXg = prf_sampler.calcXResp(prf)

            brl = brl_sampler.finalValue
            prl = prl_sampler.finalValue

            l = drift_sampler.finalValue
            p = drift_sampler.P

            perf_baseline = perf_baseline_sampler.finalValue
            #wa = perf_baseline_sampler.compute_wa(perf_baseline)

            #fit = np.dot(vXh, brl) + np.dot(vXg, prl) + np.dot(p, l) + wa

            an = fit.axes_names
            ad = fit.axes_domains
            fitted_drift = xndarray(
                np.dot(p, l), axes_names=an, axes_domains=ad)
            w = self.dataInput.w
            fitted_perf = xndarray(w[:, np.newaxis] * np.dot(vXg, prl) +
                                   fitted_drift.data +
                                   perf_baseline, axes_names=an, axes_domains=ad)
            fitted_bold = xndarray(np.dot(vXh, brl) + fitted_drift.data,
                                   axes_names=an, axes_domains=ad)
            fitted_perf_baseline = xndarray(perf_baseline + fitted_drift.data,
                                            axes_names=an, axes_domains=ad)

            rp = self.dataInput.paradigm.get_rastered(self.dataInput.tr,
                                                      tMax=ad['time'].max())
            p = np.array([rp[n][0] for n in self.dataInput.cNames])

            p_adjusted = p[:, :, np.newaxis] * .15 * signal.ptp('time').data + \
                signal.min('time').data

            ad = {'time': fit.get_domain('time'),
                  'condition': self.dataInput.cNames}

            c_paradigm = xndarray(p_adjusted,
                                  axes_names=['condition', 'time', 'voxel'],
                                  axes_domains=ad)
            # stack everything

            outputs['fits'] = stack_cuboids([signal, fit, fitted_perf,
                                             fitted_bold,
                                             c_paradigm.sum('condition'),
                                             fitted_perf_baseline],
                                            'stype',
                                            ['signal', 'fit', 'perf',
                                                       'bold', 'paradigm',
                                                       'perf_baseline'])

        return outputs

# TODO : move this import to the beginning of the module
from pyhrf.boldsynth.scenarios import *


def simulate_asl(output_dir=None, noise_scenario='high_snr',
                 spatial_size='tiny', v_noise=None, dt=0.5, tr=2.5):
    from pyhrf import Condition
    from pyhrf.tools import Pipeline

    drift_var = 10.
    
    #tr = 2.5
    #dt = 0.5
    dsf = tr/dt  # down sampling factor

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    elif spatial_size == 'random_small':
        lmap1, lmap2, lmap3 = 'random_small', 'random_small', 'random_small'
    else:
        lmap1, lmap2, lmap3 = 'ghost', 'icassp13', 'house_sun'

    if noise_scenario == 'high_snr':
        v_noise = v_noise or 0.05
        conditions = [
            Condition(name='audio', perf_m_act=10., perf_v_act=.1, perf_v_inact=.2,
                      bold_m_act=15., bold_v_act=.1, bold_v_inact=.2,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=11., perf_v_act=.11, perf_v_inact=.21,
                      bold_m_act=14., bold_v_act=.11, bold_v_inact=.21,
                      label_map=lmap2),
            #Condition(name='damier', perf_m_act=12.,
            #          perf_v_act=.12, perf_v_inact=.22,
            #          bold_m_act=20., bold_v_act=.12, bold_v_inact=.22,
            #          label_map=lmap3),
        ]
    elif noise_scenario == 'low_snr_low_prl':
        v_noise = v_noise or 7.
        scale = .3
        conditions = [
            Condition(name='audio', perf_m_act=1.6*scale, perf_v_act=.1,
                      perf_v_inact=.1,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=1.6*scale, perf_v_act=.1,
                      perf_v_inact=.1,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap2),
                      ]
    else:  # low_snr
        v_noise = v_noise or 2.
        
        conditions = [
            Condition(name='audio', perf_m_act=1.7, perf_v_act=.31, perf_v_inact=.32,
                      bold_m_act=2.3, bold_v_act=.31, bold_v_inact=.32,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=1.2, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=2., bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap2),
            # 2.2 .3  1.6 .3
        ]
        """
        conditions = [
            Condition(name='audio', perf_m_act=4.7, perf_v_act=.31, perf_v_inact=.32,
                      bold_m_act=8., bold_v_act=.31, bold_v_inact=.32,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=4.9, perf_v_act=.3, perf_v_inact=.3,
                      bold_m_act=8.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap2),
            # 2.2 .3  1.6 .3
        ]"""

    from pyhrf.sandbox.physio_params import create_omega_prf, PHY_PARAMS_KHALIDOV11,\
                                            create_physio_brf, create_physio_prf
    brf = create_canonical_hrf(dt=dt)
    physiological_params = PHY_PARAMS_KHALIDOV11
    Thrf = 25.
    prf = create_omega_prf(brf, dt, PHY_PARAMS_KHALIDOV11)
    #brf = create_physio_brf(PHY_PARAMS_KHALIDOV11, response_dt=dt, response_duration=Thrf)
    brf /= np.linalg.norm(brf)
    #prf = create_physio_prf(PHY_PARAMS_KHALIDOV11, response_dt=dt, response_duration=Thrf)
    prf /= np.linalg.norm(prf)
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(brf)
        plt.plot(prf)
        plt.show()
    simulation_steps = {
        'dt': dt,
        'dsf': dsf,
        'tr': tr,
        'condition_defs': conditions,
        # Paradigm
        'paradigm': create_localizer_paradigm_avd,
        'rastered_paradigm': rasterize_paradigm,
        # Labels
        'labels_vol': create_labels_vol,
        'labels': flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # Brls
        'brls': create_time_invariant_gaussian_brls,
        # Prls
        'prls': create_time_invariant_gaussian_prls,
        # BRF
        'primary_brf': brf,
        'brf': duplicate_brf,
        # PRF
        #'primary_prf': create_prf,  # canonical HRF for testing
        'primary_prf': prf,
        #'primary_prf': prf,
        'prf': duplicate_prf,
        # Perf baseline
        'perf_baseline': create_perf_baseline,
        'perf_baseline_mean': 0.,
        'perf_baseline_var': .4,
        # Stim induced
        'bold_stim_induced': create_bold_stim_induced_signal,
        'perf_stim_induced': create_perf_stim_induced_signal,
        # Noise
        'v_gnoise': v_noise,
        'noise': create_gaussian_noise_asl,
        # Drift
        'drift_order': 4,
        'drift_var': drift_var,
        'drift_coeffs': create_drift_coeffs_asl,
        'drift': create_polynomial_drift_from_coeffs_asl,
        # Bold # maybe rename as ASL (should be handled afterwards ...
        'ctrl_tag_mat': build_ctrl_tag_matrix,
        'asl_shape': calc_asl_shape,
        'bold': create_asl_from_stim_induced,
    }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    simulation = simu_graph.get_values()

    if output_dir is not None:
        # simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))

        simulation_save_vol_outputs(simulation, output_dir)

        # f = open(op.join(output_dir, 'simulation.pck'), 'w')
        # cPickle.dump(simulation, f)
        # f.close()

    return simulation

import pyhrf.jde.models
pyhrf.jde.models.allModels['ASL0'] = {'class': ASLSampler,
                                      'doc': 'BOLD and perfusion component, BiGaussian on stationary '
                                      'response levels, iid white noise, explicit drift'
                                      }
