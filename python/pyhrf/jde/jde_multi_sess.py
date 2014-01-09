# -*- coding: utf-8 -*-
import os
import os.path as op
import pyhrf
import numpy as np
from numpy.random import randn, rand, permutation
import time

from numpy.testing import assert_almost_equal
import scipy
import scipy.linalg
from pyhrf.jde.noise import NoiseVariance_Drift_Sampler
from pyhrf.jde.drift import ETASampler
from pyhrf.jde.hrf import RHSampler, ScaleSampler
from pyhrf.jde.nrl.bigaussian import NRLSampler, MixtureWeightsSampler, BiGaussMixtureParamsSampler
from pyhrf.jde.beta import BetaSampler
from pyhrf.jde.samplerbase import GSDefaultCallbackHandler
from pyhrf import xmlio
from pyhrf.graph import graph_nb_cliques

from pyhrf.jde.intensivecalc import computeYtilde, sampleSmmNrlBar
from pyhrf.ndarray import xndarray, stack_cuboids, expand_array_in_mask

from pyhrf.tools import get_2Dtable_string
from pyhrf.tools.io import write_volume
from collections import defaultdict

from samplerbase import GibbsSampler, GibbsSamplerVariable

from pyhrf.tools import array_summary

def b():
    raise Exception

##################################################
# Noise Sampler #
##################################################
class NoiseVariance_Drift_Multi_Sess_Sampler(xmlio.XmlInitable,
                                             GibbsSamplerVariable):

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['session', 'voxel'],
                                      value_label='PM Noise Var')

    def linkToData(self, dataInput):
        NoiseVariance_Drift_Sampler.linkToData(self, dataInput)

        self.nbSess = self.dataInput.nbSessions
        sd = self.dataInput.simulData
        if sd is not None:
            if hasattr(self.dataInput.simulData, 'noise'):
                self.trueValue = np.array([ssd.noise.variance for ssd in sd])
            else:
                # self.trueValue = np.array([sd['v_noise'] \
                #                            for sd in self.dataInput.simulData])
                self.trueValue = np.array([ssd['noise'] for ssd in sd]).var(1)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)
        if self.currentValue is None:
            self.currentValue = 0.5 * self.dataInput.varData

    def sampleNextInternal(self, variables):

        self.varYbar = self.samplerEngine.getVariable('nrl_by_session').varYbar

        for s in xrange(self.nbSess):
            for j in xrange(self.nbVox):
                var_y_bar = self.varYbar[s,:,j]
                #if j==1 and s==0:
                    #print "s=%d, j=%d" %(s,j)
                    #print 'var_y_bar:', var_y_bar[:10]
                beta_g    = np.dot(var_y_bar.transpose(), var_y_bar)/2
                #print 'beta_g:', beta_g

                gammaSample = np.random.gamma((self.ny-1)/2, 1)
                #print 'gammasamples:', gammaSamples

                self.currentValue[s,j] = np.divide(beta_g, gammaSample)
        #print 'value:', self.currentValue[0,1]
        #print 'test:', (self.varYbar[0,:,185]*self.varYbar[0,:,185]).sum()
        #print 'beta_g_185:', np.dot(self.varYbar[0,:,1].transpose(), self.varYbar[0,:,1])/2
        #print 'self.currentValue:', self.currentValue
        #print 'nrl:',self.samplerEngine.getVariable('nrl_by_session').currentValue[0,0,1]



##################################################
# Drift Sampler and var drift sampler#
##################################################
class Drift_MultiSess_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):


    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)

        an = ['session', 'order','voxel']
        GibbsSamplerVariable.__init__(self,'drift', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='PM LFD')

        self.final_signal = None
        self.true_value_signal = None

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.ny = self.dataInput.ny
        self.dimDrift = self.dataInput.colP
        self.nbVox = self.dataInput.nbVoxels
        self.P = self.dataInput.lfdMat # : for all sessions
        self.nbSess = self.dataInput.nbSessions

        if dataInput.simulData is not None and \
          isinstance(dataInput.simulData, list):
            if 0: #theoretical value
                self.trueValue = np.array([sd['drift_coeffs'] \
                                           for sd in dataInput.simulData])
            else: #empiracl value -> better if a multiplicative factor used to
                # generate drifts
                sd = dataInput.simulData
                self.trueValue = np.array([np.dot(self.P[s].T, sd[s]['drift']) \
                                           for s in xrange(len(sd))])

    def checkAndSetInitValue(self, variables):
        smplVarDrift = variables[self.samplerEngine.I_ETA]
        smplVarDrift.checkAndSetInitValue(variables)

        if self.useTrueValue :
            if self.trueValue is not None:
                self.currentValue = self.trueValue
            else:
                raise Exception('Needed a true value for drift init but '\
                                    'None defined')


        if self.currentValue is None:
            #if not self.sampleFlag and self.dataInput.simulData is None :
                #self.currentValue = self.dataInput.simulData.drift.lfd
                #pyhrf.verbose(6,'drift dimensions :' \
                              #+str(self.currentValue.shape))
                #pyhrf.verbose(6,'self.dimDrift :' \
                              #+str(self.dimDrift))
                #assert self.dimDrift == self.currentValue.shape[0]
            #else:
            #self.currentValue = np.sqrt(varDrift) * \
                #np.random.randn(self.nbSess, self.dimDrift, self.nbVox)

            self.currentValue = np.array([np.dot(self.P[s].T, self.dataInput.varMBY[s]) for s in range(self.nbSess)])

        self.updateNorm()
        self.matPl = np.zeros((self.nbSess, self.ny, self.nbVox))

        for s in xrange(self.nbSess):
            self.matPl[s] = np.dot(self.P[s], self.currentValue[s])

        self.ones_Q   = np.ones((self.dimDrift))


    def updateNorm(self):
        #for s in xrange(self.nbSess):
            #norm = np.dot(self.currentValue[s].T, self.currentValue[s])

        cv = self.currentValue
        self.norm = np.array([(cv[s] * cv[s]).sum() \
                              for s in xrange(self.nbSess)]).sum()


        #if self.trueValue is not None:
            #print 'cur drift norm:', self.norm
            #print 'true drift norm:', (self.trueValue * self.trueValue).sum()

    def sampleNextInternal(self, variables):
        eta =  variables[self.samplerEngine.I_ETA].currentValue
        snrls = self.samplerEngine.getVariable('nrl_by_session')
        noise_vars = self.samplerEngine.getVariable('noise_var').currentValue

        for j in xrange(self.nbVox):
            for s in xrange(self.nbSess):
                reps = noise_vars[s,j]

                pyhrf.verbose(5, 'eta : %f' %eta)
                pyhrf.verbose(5, 'reps :' )
                pyhrf.verbose.printNdarray(5, reps)

                ##inv_vars_l = 1/eta * self.ones_Q + 1/reps * np.dot(self.P[s].transpose(), self.P[s])
                ##print 'PtP:', np.dot(self.P[s].transpose(), self.P[s])
                #inv_vars_l = (1./eta +1./reps)* self.ones_Q
                #mu_l = np.dot(1./inv_vars_l, np.dot(self.P[s].transpose(), snrls.varYtilde[s,:,j])) * 1./reps

                #pyhrf.verbose(5, 'vars_l_j_s :')
                #pyhrf.verbose.printNdarray(5, 1/inv_vars_l)
                #pyhrf.verbose(5, 'mu_l_j_s :')
                #pyhrf.verbose.printNdarray(5, mu_l)
                #print 'mu et invl:', mu_l, inv_vars_l
                #self.currentValue[s][:,j] = randn(self.dimDrift) * 1./inv_vars_l**.5 + mu_l
                ##self.currentValue[s][:,j] = np.random.multivariate_normal(mu_l, 1/inv_vars_l)
                #print j,s

                v_lj = reps*eta / (reps + eta)
                mu_lj = v_lj/reps * np.dot(self.P[s].transpose(),
                                           snrls.varYtilde[s,:,j])

                self.currentValue[s,:,j] = randn(self.dimDrift) * \
                    v_lj**.5 + mu_lj
                #print 'drifts coeffs:', self.currentValue[s][:,j]


        self.updateNorm()

        #sHrf = variables[self.samplrEngine.I_HRF]
        #self.varXh = sHrf.varXh

        for s in xrange(self.nbSess):
            self.matPl[s] = np.dot(self.P[s], self.currentValue[s])
            #snrls.computeVarYTildeSessionOpt(self.varXh[s], s)

        pyhrf.verbose(5, 'drift params :')
        pyhrf.verbose.printNdarray(5, self.currentValue)


    def sampleNextAlt(self, variables):
        self.updateNorm()

    def get_final_value(self):
        if self.final_signal is None:
            self.final_signal = np.array([np.dot(self.P[s], self.finalValue[s]) \
                                         for s in xrange(self.nbSess)])

        return self.final_signal

    def get_true_value(self):
        if self.true_value_signal is None:
            self.true_value_signal = np.array([np.dot(self.P[s],
                                                      self.trueValue[s]) \
                                              for s in xrange(self.nbSess)])
        return self.true_value_signal

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        tol = 0.025
        tv = self.get_true_value()
        fv = self.get_final_value()
        err = np.zeros((self.nbSess, self.nbVox), dtype=np.int8)
        for s in xrange(self.nbSess):
            err[s] = (((fv[s] - tv[s])**2).sum(0) / (tv[s]**2).sum(0))**.5
        return ['session','voxel'], err < tol

    def getOutputs(self):

        #sn = self.dataInput.sNames
        outputs = GibbsSamplerVariable.getOutputs(self)
        drifts = self.get_final_value()
        an = ['session', 'time','voxel']
        #ad = {'time' : arange(self.ny)*self.dataInput.tr, 'session':sn}
        c = xndarray(drifts, axes_names=an)

        if self.trueValue is not None:
            tv = self.get_true_value()
            c_true = xndarray(tv, axes_names=an)
            c = stack_cuboids([c, c_true], axis='type', domain=['estim', 'true'],
                              axis_pos='last')

        outputs['drift_signal'] = c

        return outputs


class ETASampler_MultiSess(ETASampler):


    def linkToData(self, dataInput):
        ETASampler.linkToData(self, dataInput)
        self.nbSessions = dataInput.nbSessions
        # self.dataInput = dataInput
        # self.nbVox = self.dataInput.nbVoxels

        # if dataInput.simulData is not None and \
        #         isinstance(dataInput.simulData[0], BOLDModel):
        #     self.trueValue = np.array([dataInput.simulData[0].rdrift.var])



    def sampleNextInternal(self, variables):

        smpldrift = variables[self.samplerEngine.I_DRIFT]
        alpha = .5 * (self.nbSessions * smpldrift.dimDrift * self.nbVox -1)
        beta_d = 0.5*smpldrift.norm
        pyhrf.verbose(4, 'eta ~ Ga(%1.3f,%1.3f)'%(alpha,beta_d))
        self.currentValue[0] = 1.0/np.random.gamma(alpha,1/beta_d)

        if pyhrf.verbose.verbosity > 3:
            print 'true var drift :', self.trueValue
            print 'm_theo=%f, v_theo=%f' %(beta_d/(alpha-1),
                                           beta_d**2/((alpha-1)**2 * (alpha-2)))
            samples = 1.0/np.random.gamma(alpha,1/beta_d,1000)
            print 'm_empir=%f, v_empir=%f' %(samples.mean(), samples.var())




##################################################
# HRF Sampler #
##################################################

def sampleHRF_voxelwise_iid( stLambdaS, stLambdaY, varR, rh, nbColX, nbVox,
                             nbSess):

    pyhrf.verbose(4,'stLambdaS:')
    pyhrf.verbose.printNdarray(4,stLambdaS)
    pyhrf.verbose(4,'varR:')
    pyhrf.verbose.printNdarray(4,varR)
    pyhrf.verbose(4,'rh: %f' %rh)
    pyhrf.verbose(4,'varR/rh:')
    pyhrf.verbose.printNdarray(4,varR/rh)

    varInvSigma_h = stLambdaS + nbSess * nbVox * varR/rh

    #sv = scipy.linalg.svdvals(varInvSigma_h)
    #pyhrf.verbose(5,
    #              "Conditioning of varInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
    mean_h = np.linalg.solve(varInvSigma_h, stLambdaY)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()
        #choleskyInvSigma_h = msqrt(varInvSigma_h)
        #sv = scipy.linalg.svdvals(choleskyInvSigma_h)
        #pyhrf.verbose(5,
        #              "Conditioning of choleskyInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf


def sampleHRF_single_hrf_hack(stLambdaS, stLambdaY, varR, rh, nbColX, nbVox):
    varInvSigma_h = stLambdaS/nbVox

    pyhrf.verbose(4,'stLambdaS:')
    pyhrf.verbose.printNdarray(4,stLambdaS)
    pyhrf.verbose(4,'varR:')
    pyhrf.verbose.printNdarray(4,varR)
    pyhrf.verbose(4,'rh: %f' %rh)
    pyhrf.verbose(4,'varR/rh:')
    pyhrf.verbose.printNdarray(4,varR/rh)

#     if trick == True and nbVox is not None:
#         varInvSigma_h += varR/rh*nbVox
#     else:
    varInvSigma_h += varR/rh

    #sv = scipy.linalg.svdvals(varInvSigma_h)
    #pyhrf.verbose(5,
    #              "Conditioning of varInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
    mean_h = np.linalg.solve(varInvSigma_h, stLambdaY/nbVox)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()
        #choleskyInvSigma_h = msqrt(varInvSigma_h)
        #sv = scipy.linalg.svdvals(choleskyInvSigma_h)
        #pyhrf.verbose(5,
        #              "Conditioning of choleskyInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf


def sampleHRF_single_hrf(stLambdaS, stLambdaY, varR, rh, nbColX, nbVox):

    varInvSigma_h = stLambdaS
    pyhrf.verbose(4,'stLambdaS:')
    pyhrf.verbose.printNdarray(4,stLambdaS)
    pyhrf.verbose(4,'varR:')
    pyhrf.verbose.printNdarray(4,varR)
    pyhrf.verbose(4,'rh: %f' %rh)
    pyhrf.verbose(4,'varR/rh:')
    pyhrf.verbose.printNdarray(4,varR/rh)

#     if trick == True and nbVox is not None:
#         varInvSigma_h += varR/rh*nbVox
#     else:
    varInvSigma_h += varR/rh

    #sv = scipy.linalg.svdvals(varInvSigma_h)
    #pyhrf.verbose(5,
    #              "Conditioning of varInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
    mean_h = np.linalg.solve(varInvSigma_h, stLambdaY)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()
        #choleskyInvSigma_h = msqrt(varInvSigma_h)
        #sv = scipy.linalg.svdvals(choleskyInvSigma_h)
        #pyhrf.verbose(5,
        #              "Conditioning of choleskyInvSigma_h: %1.3g" %(sv[0]/sv[-1]))
        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf



class HRF_MultiSess_Sampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    HRF sampler for multisession model
    """

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = ['do_sampling', 'duration', 'zero_constraint']

    parametersComments = {
        'duration' : 'HRF length in seconds',
        'zero_constraint' : 'If True: impose first and last value = 0.\n'\
                        'If False: no constraint.',
        'do_sampling' : 'Flag for the HRF estimation (True or False).\n'\
                        'If set to False then the HRF is fixed to a canonical '\
                        'form.',
        'prior_type' : 'Type of prior:\n - "singleHRF": one HRF modelled '\
            'for the whole parcel ~N(0,v_h*R).\n' \
            ' - "voxelwiseIID": one HRF per voxel, '\
            'all HRFs are iid ~N(0,v_h*R).',
        'covar_hack' : 'Divide the term coming from the likelihood by the nb '\
            'of voxels\n when computing the posterior covariance. The aim is '\
            ' to balance\n the contribution coming from the prior with that '\
            ' coming from the likelihood.\n Note: this hack is only taken into '\
            ' account when "singleHRf" is used for "prior_type"',
        'normalise' : 'If 1. : Normalise samples of Hrf and NRLs when they are sampled.\n'\
                      'If 0. : Normalise posterior means of Hrf and NRLs when they are sampled.\n'\
                      'else : Do not normalise.'

        }

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None, duration=25., zero_constraint=True,
                 normalise=1., deriv_order=2, covar_hack=False,
                 prior_type='voxelwiseIID', do_voxelwise_outputs=False,
                 compute_ah_online=False):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)

        self.compute_ah_online = compute_ah_online
        GibbsSamplerVariable.__init__(self, 'hrf', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['time'],
                                      value_label='delta BOLD')

        self.duration = duration
        self.zc = zero_constraint
        self.normalise = normalise
        #print 'normalise', self.normalise
        self.derivOrder = deriv_order
        self.varR = None
        self.covarHack = covar_hack
        self.priorType = prior_type
        self.voxelwise_outputs = do_voxelwise_outputs

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt
        self.nbSess = self.dataInput.nbSessions

        if dataInput.simulData is not None and \
            isinstance(dataInput.simulData, dict):
            simu_hrf = dataInput.simulData['hrf']
            if isinstance(simu_hrf, xndarray):
                self.trueValue = simu_hrf.data
            else:
                self.trueValue = simu_hrf
        elif dataInput.simulData is not None and \
            isinstance(dataInput.simulData, list):
            #mutlisession
            # take only HRF of 1st session
            simu_hrf = dataInput.simulData[0]['hrf']
            if isinstance(simu_hrf, xndarray):
                self.trueValue = simu_hrf.data
            else:
                self.trueValue = simu_hrf[:,0] #HACK retrieve only one hrf

        # Allocations :
        self.ajak_rb =  np.zeros((self.nbVox), dtype=float)
        #self.ajak_rb =  np.zeros((self.nbVox), dtype=float)

        if 0:
            self.varStLambdaS = np.zeros((self.nbColX, self.nbColX, self.nbVox),
                                      dtype=float)
            self.varStLambdaY = np.zeros((self.nbColX, self.nbVox), dtype=float)
        self.varYaj = np.zeros((self.ny,self.nbVox), dtype=float)

        self.varXh = np.zeros((self.nbSess, self.ny, self.nbConditions),
                              dtype=np.float64)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5


    def checkAndSetInitValue(self, variables):
        smplRH = self.samplerEngine.getVariable('hrf_var')
        smplRH.checkAndSetInitValue(variables)
        rh = smplRH.currentValue
        pyhrf.verbose(4, 'Hrf variance is :%1.3f' %rh)
        pyhrf.verbose(4, 'hrfValIni is None -> setting it ...')


        #HACK
        #if not self.sampleFlag and self.dataInput.simulData != None :
        #if 0:
        #HACK
        hrfValIni = None
        if self.useTrueValue :
            if self.trueValue is not None:
                hrfValIni = self.trueValue
            else:
                raise Exception('Needed a true value for hrf init but '\
                                    'None defined')

        if hrfValIni is None:
            pyhrf.verbose(6, 'self.duration=%d, self.eventdt=%1.2f' \
                              %(self.duration,self.eventdt))

            #hIni = genCanoBezierHRF(self.duration, self.eventdt)[1]
            pyhrf.verbose(5,'genCanoHRF -> dur=%f, dt=%f' \
                              %(self.duration, self.eventdt))
            dt = self.eventdt
            hIni = msimu.getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

            #if len(hIni) > self.hrfLength:
            #    hIni = getCanoHRF((self.hrfLength-1)*dt,self.eventdt)[1]
            #elif len(hIni) < self.hrfLength:
            #    hIni = getCanoHRF((self.hrfLength+1)*dt, self.eventdt)[1]

            hrfValIni = np.array(hIni)
            pyhrf.verbose(5,'genCanoHRF -> shape h: %s' \
                          %str(hrfValIni.shape))

        if self.zc :
            pyhrf.verbose(4,'hrf zero constraint On' )
            hrfValIni = hrfValIni[1:(self.hrfLength-1)]

        pyhrf.verbose(4,'hrfValIni:' +\
                          str(hrfValIni.shape))
        pyhrf.verbose.printNdarray(6, hrfValIni)
        pyhrf.verbose(4, 'self.hrfLength:' \
                          +str(self.hrfLength))

        normHRF = (sum(hrfValIni**2))**(0.5)
        hrfValIni /= normHRF

        self.currentValue = hrfValIni[:]

        if self.zc :
            self.axes_domains['time'] = np.arange(len(self.currentValue)+2) \
                                       *self.eventdt
        else:
            self.axes_domains['time'] = np.arange(len(self.currentValue)) \
                                       *self.eventdt


        pyhrf.verbose(4,'hrfValIni after ZC:' +\
                      str(self.currentValue.shape))
        pyhrf.verbose.printNdarray(6, self.currentValue )

        self.updateNorm()
        self.updateXh()

        self.track_sampled_quantity(self.varXh, 'varXh', axes_names=['session',
                                                                     'time',
                                                                     'condition',
                                                                     ])

        #else -> #TODO : check consistency between given init value
        # and self.hrfLength ...

        #Update Ytilde



    def getCurrentVar(self):
        smplRH = self.samplerEngine.getVariable('hrf_var')
        rh = smplRH.currentValue
        (useless, varR) = msimu.genGaussianSmoothHRF(self.zc,
                                                     self.hrfLength,
                                                     self.eventdt, rh)
        return varR/rh

    def getFinalVar(self):
        smplRH = self.samplerEngine.getVariable('hrf_var')
        rh = smplRH.finalValue
        (useless, varR) = msimu.genGaussianSmoothHRF(self.zc,
                                                     self.hrfLength,
                                                     self.eventdt, rh)
        return varR/rh



    def samplingWarmUp(self, variables):
        if self.varR == None :
            smplRH = variables[self.samplerEngine.I_RH]
            rh = smplRH.currentValue
            (useless, self.varR) = msimu.genGaussianSmoothHRF(self.zc,
                                                              self.hrfLength,
                                                              self.eventdt, rh,
                                                              order=self.derivOrder)
            #self.varR = buildDiagGaussianMat(self.hrfLength-self.zc*2,4)
            # HACK
            #self.varR = ones_like(self.varR)


    def computeStDS_StDY_one_session(self, rb, nrls, aa, sess):
        #case drift sampling

        varX = self.dataInput.varX[sess][:,:,:]
        matXtX = self.dataInput.matXtX[sess]
        drift_sampler = self.getVariable('drift')
        matPl = drift_sampler.matPl[sess]
        y = self.dataInput.varMBY[sess] - matPl

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[sess]
            osf = int(sd['tr'] / sd['dt'])
            if not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                assert_almost_equal(self.dataInput.varMBY[sess], sd['bold'])
                assert_almost_equal(matPl, sd['drift'])
                assert_almost_equal(y, sd['stim_induced'][::osf] + sd['noise'])

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )

        for j in xrange(self.nbConditions):
            np.divide(y, rb, self.varYaj)
            self.varYaj *= nrls[j,:]
            varDeltaY +=  np.dot(varX[j,:,:].transpose(),self.varYaj.sum(1))

            for k in xrange(self.nbConditions):
                    np.divide(aa[j,k,:], rb, self.ajak_rb[:])
##                    pyhrf.verbose(6, 'ajak/rb :')
##                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                    varDeltaS += self.ajak_rb.sum()*matXtX[j,k,:,:]

        return (varDeltaS, varDeltaY)

    def computeStDS_StDY_from_HRFSampler(self, rb, nrls, aa):
        """ just for comparison purpose. Should be removed in the end.
        """
        matXQ = self.dataInput.matXQ
        matXQX = self.dataInput.matXQX
        y = self.dataInput.varMBY

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )

        for j in xrange(self.nbConditions):
            np.divide(y, rb, self.varYaj)
            self.varYaj *= nrls[j,:]
            varDeltaY +=  np.dot(matXQ[j,:,:],self.varYaj.sum(1))

            for k in xrange(self.nbConditions):
                    np.divide(aa[j,k,:], rb, self.ajak_rb)
##                    pyhrf.verbose(6, 'ajak/rb :')
##                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                    varDeltaS += self.ajak_rb.sum()*matXQX[j,k,:,:]

        return (varDeltaS, varDeltaY)

    def computeStDS_StDY(self, rb_allSess, nrls_allSess, aa_allSess):

        varDeltaS_Sess = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY_Sess = np.zeros((self.nbColX), dtype=float )

        for s in xrange(self.nbSess):
            ds, dy = self.computeStDS_StDY_one_session(rb_allSess[s,:],
                            nrls_allSess[s,:,:], aa_allSess[s,:,:,:], s)
            varDeltaS_Sess += ds
            varDeltaY_Sess += dy

        return (varDeltaS_Sess, varDeltaY_Sess)



    def sampleNextAlt(self, variables):
        self.reportCurrentVal()

    def sampleNextInternal(self, variables):
        #TODO : comment

        snrl = self.samplerEngine.getVariable('nrl_by_session')
        nrls = snrl.currentValue
        rb   = self.samplerEngine.getVariable('noise_var').currentValue

        pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
        tSQSOptimIni = time.time()
        (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY(rb, nrls,
                                                                 snrl.aa)
        pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                      ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        rh = variables[self.samplerEngine.I_RH].currentValue


        if self.priorType == 'voxelwiseIID':
            h = sampleHRF_voxelwise_iid(self.varDeltaS, self.varDeltaY,
                                        self.varR,
                                        rh, self.nbColX, self.nbVox, self.nbSess)
        elif self.priorType == 'singleHRF':
            if self.covarHack:
                h = sampleHRF_single_hrf_hack(self.varDeltaS, self.varDeltaY,
                                              self.varR,
                                              rh, self.nbColX, self.nbVox)
            else:
                h = sampleHRF_single_hrf(self.varDeltaS, self.varDeltaY,
                                         self.varR,
                                         rh, self.nbColX, self.nbVox)

        self.currentValue = h


        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, '\
                          'Nrls and mixture parameters at each iteration ...')
            f = self.norm
            #HACK
            #f = self.currentValue.max()
            self.currentValue = self.currentValue / f #/(self.normalise+0.)
            if 0 and self.getVariable('nrl_by_session').sampleFlag:
                self.getVariable('nrl_by_session').currentValue *= f

                # Normalizing Mixture components
                smixt_params = self.samplerEngine.getVariable('mixt_params')
                if 0 and smixt_params.sampleFlag:
                    # Normalizing Mean's activation class
                    smixt_params.currentValue[smixt_params.I_MEAN_CA] *= f
                    # Normalizing Variance's activation class
                    smixt_params.currentValue[smixt_params.I_VAR_CI] *= f**2
                    # Normalizing Variance's in-activation class
                    smixt_params.currentValue[smixt_params.I_VAR_CA] *= f**2


        #HACK:
        # self.currentValue = self.trueValue[1:-1]

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        self.updateXh()
        self.reportCurrentVal()


        # update ytilde for nrls
        nrlsmpl = variables[self.samplerEngine.I_NRLS_SESS]
        for s in xrange(self.nbSess):
            nrlsmpl.computeVarYTildeSessionOpt(self.varXh[s], s)



    def reportCurrentVal(self):
        if pyhrf.verbose.verbosity >= 3:
            maxHRF = self.currentValue.max()
            tMaxHRF = np.where(self.currentValue==maxHRF)[0]*self.dt
            pyhrf.verbose(1, 'sampled HRF = %1.3f(%1.3f)'
                          %(self.currentValue.mean(),self.currentValue.std()))
            pyhrf.verbose(1,'sampled HRF max = ' + \
                              '(tMax=%1.3f,vMax=%1.3f)' %(tMaxHRF, maxHRF))

    def calcXh(self, hrf):
        pyhrf.verbose(4,'CalcXh got stackX ' + \
                      str(self.dataInput.stackX.shape))
        all_stackXh = np.zeros((self.nbSess, self.ny, self.nbConditions),
                               dtype=np.float64)
        for s in xrange(self.nbSess):
            stackXh = np.dot(self.dataInput.stackX[s], hrf)
            all_stackXh[s] = np.reshape(stackXh,
                                        (self.nbConditions,
                                         self.ny)).transpose()
        return all_stackXh

    def updateXh(self):
        self.varXh[:] = self.calcXh(self.currentValue)

    def setFinalValue(self):
        fv = self.mean#/self.normalise
        if self.zc:
            # Append and prepend zeros
            self.finalValue = np.concatenate(([0], fv, [0]))
            #self.error = np.concatenate(([0], self.error, [0]))
            if self.meanHistory is not None:
                nbIt = len(self.obsHistoryIts)

                self.meanHistory =  np.hstack((np.hstack((np.zeros((nbIt,1)),
                                                    self.meanHistory)),
                                            np.zeros((nbIt,1))))

            if self.smplHistory is not None:
                nbIt = len(self.smplHistoryIts)
                self.smplHistory = np.hstack((np.hstack((np.zeros((nbIt,1)),
                                                    self.smplHistory)),
                                           np.zeros((nbIt,1))))
        else:
            self.finalValue = fv

        pyhrf.verbose(4, 'HRF finalValue :\n')
        pyhrf.verbose.printNdarray(4, self.finalValue)


    def getScaleFactor(self):
        if self.finalValue == None:
            self.setFinalValue()
        # Use amplitude :
        #scaleF = self.finalValue.max()-self.finalValue.min()
        scaleF = 1.
        # Use norm :
        #scaleF = ((self.finalValue**2.0).sum())**0.5
        # Use max :
        #scaleF = self.finalValue.max()
        return scaleF


    def initObservables(self):
        GibbsSamplerVariable.initObservables(self)
        if self.compute_ah_online:
            self.current_ah = np.zeros((self.currentValue.shape[0],self.nbVox,
                                        self.nbConditions))
            self.cumul_ah = np.zeros_like(self.current_ah)
            self.cumul2_ah = np.zeros_like(self.current_ah)


    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)
        sScale = self.samplerEngine.getVariable('scale')
        if self.sampleFlag and np.allclose(self.normalise,0.) and \
                not sScale.sampleFlag:
            pyhrf.verbose(6, 'Normalizing posterior mean of HRF '\
                          'each iteration ...')
            #print '%%%%%% scaling PME (hrf) %%%%%%%'
            # Undo previous mean calculation:
            self.cumul -= self.currentValue
            self.cumul3 -= (self.currentValue - self.mean)**2
            # Use normalised quantities instead:
            self.cumul += self.currentValue/self.norm
            self.mean = self.cumul / self.nbItObservables
            self.cumul3 += (self.currentValue/self.norm - self.mean)**2

            self.error = self.cumul3 / self.nbItObservables

        if self.compute_ah_online:
            for j in xrange(self.nbConditions):
                hrep = np.repeat(self.currentValue,self.nbVox)
                ncoeffs = self.currentValue.shape[0]
                nrls = self.samplerEngine.getVariable('nrl').currentValue
                self.current_ah[:,:,j] = hrep.reshape(ncoeffs,self.nbVox) * \
                    nrls[j,:]

            self.cumul_ah += self.current_ah
            self.cumul2_ah += self.current_ah**2
            self.mean_ah = self.cumul_ah / self.nbItObservables
            self.var_ah = self.cumul2_ah / self.nbItObservables - self.mean_ah**2

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        crit_norm = np.array([((fv - tv)**2).sum() / (tv**2).sum()**.5 < 0.02 ])
        return ['crit_norm'], crit_norm


    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag: #TODO chech for NaNs ...  and not _np.isnan(rh):
            # store errors:
            rh = self.samplerEngine.getVariable('hrf_var').finalValue

            rb = self.samplerEngine.getVariable('noise_var').finalValue
            snrls = self.samplerEngine.getVariable('nrl_by_session')
            nrls = snrls.finalValue
            aa = np.zeros_like(snrls.aa)
            snrls.computeAA(nrls, aa)
            stDS, useless = self.computeStDS_StDY(rb, nrls, aa)
            varSigma_h = np.asmatrix(stDS + self.varR/rh).I
            if self.zc:
                self.error[1:-1] = np.diag(varSigma_h)**.5
            else:
                self.error = np.diag(varSigma_h)**.5

        if hasattr(self, 'varDeltaY'):
            del self.varDeltaY
            del self.varDeltaS

        # Temporary variables :
        del self.varYaj
        del self.ajak_rb

        # may be needed afterwards :
        #del self.varXh
        #del self.varR


    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)

        h = self.finalValue
        nrls = self.samplerEngine.getVariable('nrl').finalValue
        ah = np.zeros((h.shape[0],self.nbVox, self.nbConditions))
        for j in xrange(self.nbConditions):
            ah[:,:,j] = np.repeat(h,self.nbVox).reshape(h.shape[0],self.nbVox) * \
                nrls[j,:]
        ad = self.axes_domains.copy()
        ad['condition'] = self.dataInput.cNames
        outputs['ah'] = xndarray(ah, axes_names=['time','voxel','condition'],
                               axes_domains=ad,
                               value_label='Delta BOLD')

        if self.zc:
            dm = self.calcXh(self.finalValue[1:-1])
        else:
            dm = self.calcXh(self.finalValue)
        xh_ad = {
            'time' : np.arange(self.dataInput.ny)*self.dataInput.tr,
            'condition':self.dataInput.cNames
            }

        outputs['Xh'] = xndarray(np.array(dm),
                               axes_names=['session','time','condition'],
                               axes_domains=xh_ad)

        if getattr(self, 'compute_ah_online', False):
            z = np.zeros((1,)+self.mean_ah.shape[1:], dtype=np.float32)

            outputs['ah_online'] = xndarray(np.concatenate((z,self.mean_ah,z)),
                                          axes_names=['time','voxel',
                                                      'condition'],
                                          axes_domains=ad,
                                          value_label='Delta BOLD')

            outputs['ah_online_var'] = xndarray(np.concatenate((z,self.var_ah,z)),
                                              axes_names=['time','voxel',
                                                          'condition'],
                                              axes_domains=ad,
                                              value_label='var')

        if hasattr(self, 'finalValue_sign_corr'):
            outputs['hrf_sign_corr'] = xndarray(self.finalValue_sign_corr,
                                              axes_names=self.axes_names,
                                              axes_domains=self.axes_domains,
                                              value_label='Delta BOLD')

        #print 'hrf - finalValue:'
        #print self.finalValue
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            nrls = self.samplerEngine.getVariable('nrl').finalValue
            #print np.argmax(abs(nrls))
            #print nrls.shape
            nrlmax = nrls[np.unravel_index(np.argmax(abs(nrls)),nrls.shape)]
            # ttp:
            h = self.finalValue * nrlmax
            ttp = np.where(np.diff(h)<0)[0][0]
            if ttp == 0:
                ttp = np.where(np.diff(np.diff(h))>0)[0][0]
                if ttp == 0:
                    ttp = h.argmax()
                #ttp = -1
            #print '->ttp:', ttp
            outputs['ttp'] = xndarray(np.zeros(self.nbVox) + ttp * self.dt,
                                    axes_names=['voxel'], value_label='TTP')

            #ttu = self.finalValue[self.finalValue.argmax():].argmin() * self.dt
            #outputs['ttu'] = xndarray(zeros(self.nbVox) + ttp,
            #                        axes_names=['voxel'], value_label='TTU')
            try:
                hInterp = scipy.interpolate.interp1d(np.arange(0,len(self.finalValue)),
                                                     self.finalValue)
                r = 0.01
                nh = hInterp(np.arange(0,len(self.finalValue)-1, r))
                #print 'nh.shape', nh.shape
                #print 'ttp*r=', ttp/r
                #print 'nh[ttp/r]/2:', nh[ttp/r]/2
                #print 'nh[ttp/r:0]:', nh[ttp/r:0]
                #print 'np.where(nh[ttp/r]/2-nh[ttp/r:0]>=0):'
                #print np.where(nh[ttp/r]/2-nh[ttp/r:-1]>=0)
                whM = (np.where(nh[ttp/r]/2-nh[ttp/r:-1]>=0)[0][0] - \
                           np.where(nh[ttp/r]/2-nh[0:ttp/r]>=0)[0][0]) * r * self.dt

                outputs['whM'] = xndarray(np.zeros(self.nbVox) + whM,
                                          axes_names=['voxel'], value_label='whM')
            except Exception, e:
                pyhrf.verbose(3, "could not compute whm, exception:")
                pyhrf.verbose(3, str(e))
                outputs['whM'] = xndarray(np.zeros(self.nbVox) - 1,
                                          axes_names=['voxel'], value_label='whM')

        for k,v in outputs.iteritems():
            yield k,v



###########################################################################
# NRL by session Sampler and variance of session-specific nrls sampler#
###########################################################################

class NRL_Multi_Sess_Sampler(xmlio.XMLParamDrivenClass, GibbsSamplerVariable):


    def __init__(self, do_sampling=True, val_ini=None, use_true_value=False):

        an = ['session', 'condition', 'voxel']
        GibbsSamplerVariable.__init__(self,'nrl_by_session', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='PM NRL')


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbSessions = self.dataInput.nbSessions


        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, dict):
                raise Exception('TO FIX')
                if dataInput.simulData.has_key('nrls'):
                    nrls = dataInput.simulData['nrls']
                    if isinstance(nrls, xndarray):
                        self.trueValue = nrls.reorient(['condition','voxel']).data
                    else:
                        self.trueValue = nrls
            elif isinstance(dataInput.simulData, list):
                v = np.array([sd['nrls_session'] \
                              for sd in dataInput.simulData])
                self.trueValue = v.astype(np.float64)
            else:
                raise Exception('No true value found for nrl_by_session')
        else:
            self.trueValue = None

    def checkAndSetInitValue(self, variables):
        pyhrf.verbose(3, 'NRL_Multi_Sess_Sampler.checkAndSetInitNRLs ...')
        smplNrlBar = variables[self.samplerEngine.I_NRLS_BAR]
        smplNrlBar.checkAndSetInitValue(variables)

        self.smplHRF = variables[self.samplerEngine.I_HRF]

        self.smplDrift = variables[self.samplerEngine.I_DRIFT]
        self.smplDrift.checkAndSetInitValue(variables)

        self.varYtilde = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        self.sumaXh = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        self.varYbar = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for nrls init but '\
                                    'None defined')
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None :

            #nrlsIni = np.zeros((self.nbSessions,self.nbConditions, self.nbVox), dtype=np.float64)
            ## Init Nrls according to classes definitions :
            #smplGaussP = variables[self.samplerEngine.I_NRLs_Gauss_P]
            ## ensure that mixture parameters are correctly set
            #smplGaussP.checkAndSetInitValue(variables)
            #var_nrls = smplMixtP.getCurrentVars()
            #means = smplMixtP.getCurrentMeans()

            #for s in xrange(self.nbSessions):
                #for m in xrange(self.nbConditions):
                    #for j in xrange(self.nbVox):
                        #nrlsIni[s,m,j] = np.random.randn() \
                        #* var_nrls**0.5 + means[s,m]
            #self.currentValue = nrlsIni
            ##HACK (?)
            if 0:
                self.currentValue = np.zeros((self.nbSessions, self.nbConditions, self.nbVox),
                                            dtype=np.float64)
                nrl_bar = self.samplerEngine.getVariable('nrl').currentValue
                var_sess = self.samplerEngine.getVariable('variance_nrls_by_session').currentValue
                labels = self.samplerEngine.getVariable('nrl').labels
                for m in xrange(self.nbConditions):
                    Ac_pos = np.where(labels[m])
                    for s in xrange(self.nbSessions):
                        Nrls_sess = np.random.randn((self.nbVox))*var_sess**0.5 #+ nrl_bar[s,m]
                        Nrls_sess[Ac_pos[0]] = np.random.randn((Ac_pos[0].size))*var_sess**0.5 + 30
                        self.currentValue[s,m] = Nrls_sess.astype(np.float64)
            #self.currentValue[s]
            self.currentValue = np.zeros((self.nbSessions, self.nbConditions,
                                          self.nbVox), dtype=np.float64)+20


    def saveCurrentValue(self, it):
        GibbsSamplerVariable.saveCurrentValue(self, it)

    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """

        # Precalculations and allocations :
        smplHRF = self.samplerEngine.getVariable('hrf')

        #self.egsurrb = np.empty(( self.nbConditions, self.nbVox), dtype=float)
        #self.varYtilde = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        #self.varYbar = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        #self.sumaXh = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        self.aa = np.zeros((self.nbSessions, self.nbConditions,
                            self.nbConditions, self.nbVox), dtype=float)
        self.meanApost = np.zeros((self.nbSessions, self.nbConditions,
                                   self.nbVox), dtype=float)
        self.varApost = np.zeros((self.nbSessions,self.nbConditions,
                                  self.nbVox), dtype=float)

        self.track_sampled_quantity(self.meanApost, 'nrl_sess_mean_apost',
                                    ['session','condition','voxel'])

        self.track_sampled_quantity(self.varApost, 'nrl_sess_var_apost',
                                    ['session','condition','voxel'])

        self.imm = self.samplerEngine.getVariable('beta').currentValue[0] < 0

        for s in xrange(self.nbSessions):
            self.computeVarYTildeSessionOpt(smplHRF.varXh[s], s)

        self.track_sampled_quantity(self.sumaXh, 'sum_aXh',
                                    ['session','time','voxel'])

        self.track_sampled_quantity(self.varYtilde, 'var_ytilde',
                                    ['session','time','voxel'])

        self.track_sampled_quantity(self.varYbar, 'var_ybar',
                                    ['session','time','voxel'])

        self.computeAA(self.currentValue, self.aa)
        self.iteration = 0
        pyhrf.verbose(5,'varYtilde at end of warm up %s' \
          %str(self.varYtilde.shape))

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):

        snrl_bar = self.getVariable('nrl')
        labs = snrl_bar.finalLabels
        # same criterion as np.allclose:
        acc = abs_error <= (atol + rtol * np.maximum(np.abs(tv),
                                                     np.abs(fv)))

        # take only absolute error for inactivated voxels
        m = np.where(labs==0)
        atol = .31
        for s in xrange(self.nbSessions):
            acc[s,m[0],m[1]] = abs_error[s,m[0],m[1]] <= atol


        return self.axes_names, acc

    def computeAA(self, nrls, destaa):
        for s in xrange(self.nbSessions):
            for j in xrange(self.nbConditions):
                for k in xrange(self.nbConditions):
                    np.multiply(nrls[s,j,:], nrls[s,k,:],
                                destaa[s,j,k])


    def computeVarYTildeSessionOpt(self, varXh, s):
        #print 'shapes:', varXh.shape, self.currentValue[s].shape, self.dataInput.varMBY[s].shape, self.varYtilde[s].shape, self.sumaXh[s].shape
        computeYtilde(varXh.astype(np.float64),
                      self.currentValue[s], self.dataInput.varMBY[s],
                      self.varYtilde[s], self.sumaXh[s])

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde[s].shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde[s])
        matPl = self.getVariable('drift').matPl
        self.varYbar[s] = self.varYtilde[s] - matPl[s]

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[s]
            osf = int(sd['tr'] / sd['dt'])
            if not self.sampleFlag and  not self.smplHRF.sampleFlag and\
              self.useTrueValue and self.smplHRF.useTrueValue:
              assert_almost_equal(self.sumaXh[s], sd['stim_induced_signal'][::osf])
              assert_almost_equal(self.varYtilde[s], sd['bold'] - \
                                  sd['stim_induced_signal'][::osf])
              if not self.smplDrift.sampleFlag and \
                self.smplDrift.useTrueValue:
                assert_almost_equal(self.varYbar[s], sd['bold'] - \
                                    sd['stim_induced_signal'][::osf] - sd['drift'])


    def sampleNextAlt(self, variables):
        pass
        #used when sampling if OFF !
        # if self.getVariable('hrf').sampleFlag:
        #     varXh = variables[self.samplerEngine.I_HRF].varXh
        #     for s in xrange(self.nbSessions):
        #         self.computeVarYTildeSessionOpt(varXh[s], s)
        # -> already done at the end of HRF sampling

    def computeComponentsApost(self, s, m, varXh):
        var_a = self.getVariable('variance_nrls_by_session').currentValue
        rb = self.getVariable('noise_var').currentValue
        nrls = self.currentValue
        nrl_bar = self.getVariable('nrl').currentValue
        pyhrf.verbose(6, 'rb %s :'%str(rb.shape))
        pyhrf.verbose.printNdarray(6, rb)

        gTg = np.diag(np.dot(varXh[s].transpose(),varXh[s]))
        varXh_m = varXh[s,:,m]
        ejsm = self.varYbar[s] + (nrls[np.newaxis,s,m,:] * \
                                  varXh_m[:,np.newaxis])

        #pyhrf.verbose(6, 'varYtilde %s :'%str((self.varYtilde.shape)))
        #pyhrf.verbose.printNdarray(6, self.varYtilde)

        #pyhrf.verbose(6, 'nrls[%d,:] %s :'%(j,nrls[j,:]))
        #pyhrf.verbose.printNdarray(6, nrls[j,:])

        #pyhrf.verbose(6, 'varXh[:,%d] %s :'%(j,str(varXh[:,j].shape)))
        #pyhrf.verbose.printNdarray(6, varXh[:,j])

        #pyhrf.verbose(6, 'repmat(varXh[:,%d],self.nbVox, 1).transpose()%s:' \
                          #%(j,str((repmat(varXh[:,j],self.nbVox, 1).transpose().shape))))
        #pyhrf.verbose.printNdarray(6, repmat(varXh[:,j],self.nbVox, 1).transpose())

            # self.egsurrb = np.divide(np.dot(ejsm.transpose(), varXh[s][:,m]),
            #                          rb[s,:])

            #print 'varYbar:', self.varYbar[s][150]
            #print 'nrls*g:', nrls[s,m,:] * repmat(varXh[s][:,m],self.nbVox, 1).transpose()
            #print 'ejsm:', ejsm[:,150]
            #print 'g:', varXh[s][:,m]
            #print 'ejsm*g:', np.dot(ejsm.transpose(), varXh[s][:,m])[150]
            #print 'nrlbar:', nrl_bar[m,150]
            #print 'varXh:', varXh[s][150,m]
            #print 'egsurrb', self.egsurrb[150]

            #self.varApost[s,m,:] = np.sqrt(1./(1./var_a + gTg[m]*1./rb[s,:]))
        self.varApost[s,m,:] = (rb[s,:] * var_a) / (gTg[m] * var_a + rb[s,:])
            # np.multiply(self.sigApost[s,m,:]**2,
            #             np.add(nrl_bar[m,:]/var_a, self.egsurrb),
            #             self.meanApost[s,m,:])
        self.meanApost[s,m,:] = self.varApost[s,m,:] * \
          (np.dot(varXh_m.T, ejsm)/rb[s,:] + nrl_bar[m,:] / var_a)


    def sampleNextInternal(self, variables):
        pyhrf.verbose(3, 'NRL_Multi_Sess_Sampler.sampleNextInternal ...')
        varXh = self.getVariable('hrf').varXh

        for s in xrange(self.nbSessions):
            self.computeVarYTildeSessionOpt(varXh[s], s)
            for m in xrange(self.nbConditions):
                self.computeComponentsApost(s, m, varXh)
                smpl = self.meanApost[s,m,:] + np.random.randn(self.nbVox)*\
                  self.varApost[s,m,:]**.5
                self.currentValue[s,m,:] = smpl
                self.computeVarYTildeSessionOpt(varXh[s], s)
                #b()
        # pyhrf.verbose(4, 'meanPost and sigPost [2,2,15]: %f, %f' \
        #               %(self.meanApost[2,2,15],  self.sigApost[2,2,15]))
        # pyhrf.verbose(4,'currVal[2,2,15]: %f' %self.currentValue[2,2,15])

        if (self.currentValue >= 1000).any() and pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )


        self.computeAA(self.currentValue, self.aa)

        self.iteration += 1 #TODO : factorize !!


    if 0:
        def updateObsersables(self):
            GibbsSamplerVariable.updateObsersables(self)

            self.nbItObservables += 1
            pyhrf.verbose(6, 'Generic update Observables for var %s, it=%d ...' \
                              %(self.name,self.nbItObservables))
            pyhrf.verbose(6, 'CurrentValue:')
            pyhrf.verbose.printNdarray(6, self.currentValue)

            pyhrf.verbose(6, 'Cumul:')
            pyhrf.verbose.printNdarray(6, self.cumul)

            self.mean = self.cumul / self.nbItObservables

            # Another Computing of error to avoid negative value when cumul is < 1
            self.cumul3 += (self.currentValue - self.mean)**2
            pyhrf.verbose(6, 'Cumul3:')
            pyhrf.verbose.printNdarray(6, self.cumul3)
            self.error = self.cumul3 / self.nbItObservables

            pyhrf.verbose(6, 'Mean')
            pyhrf.verbose.printNdarray(6, self.mean)

            if self.nbItObservables < 2:
                self.error = np.zeros_like(self.cumul3) + 1e-6

            tol = 1e-10
            neg_close_to_zero = np.where(np.bitwise_and(self.error<0,
                                                        np.abs(self.error)<tol))
            self.error[neg_close_to_zero] = tol
            #if len(neg_close_to_zero[0])>0:
                #self.error[neg_close_to_zero] = tol
            #else:
                #self.error = np.array((0))
            #print 'Case where error is empty and thus put to 0...hack...'

            pyhrf.verbose(6, 'Error:')
            pyhrf.verbose.printNdarray(6, self.error)

            if (self.error <0.).any():
                    raise Exception('neg error on variable %s' %self.name)


    def cleanMemory(self):

        # clean memory of temporary variables :

        del self.varApost
        del self.meanApost
        del self.aa
        del self.varYtilde
        del self.varXhtQ
        del self.sumaXh

        if hasattr(self,'nrlsSamples'):
            del self.nrlsSamples

        del self.voxIdx

    #def setFinalValue(self):
        #for s in xrange(self.nbSessions):
            #self.finalValue[s,:,:] = self.getMean(2)
            ##Mean over iterations ==> give many iterations to ensure convergence!

    def finalizeSampling(self):

        #print 'currentValue shape:', self.currentValue.shape
        #print 'finalValue.shape:', self.finalValue.shape
        #print 'finalVal:', self.finalValue[2,2,15]
        GibbsSamplerVariable.finalizeSampling(self)

        smplHRF = self.samplerEngine.getVariable('hrf')


        # Correct sign ambiguity :
        if hasattr(smplHRF,'detectSignError'):
            sign_error = smplHRF.detectSignError()
            pyhrf.verbose(2, 'sign error - Flipping nrls')
            self.finalValue_sign_corr = self.finalValue * (1-2*sign_error)

        # Correct hrf*nrl scale ambiguity :
        scaleF = smplHRF.getScaleFactor()
        # Use HRF amplitude :
        pyhrf.verbose(3, 'scaleF=%1.2g' %scaleF)
        pyhrf.verbose(3, 'self.finalValue : %1.2g - %1.2g' \
                      %(self.finalValue.min(), self.finalValue.max()))
        self.finalValueScaleCorr = self.finalValue * scaleF


    def is_accurate(self):
        tv = self.trueValue
        fv = self.finalValue
        return (((fv - tv)**2).sum() / (tv**2).sum())**.5 < 0.02


    def getOutputs(self):

        outputs = GibbsSamplerVariable.getOutputs(self)

        cn = self.dataInput.cNames
        sn = self.dataInput.sNames

        an = ['session', 'condition', 'voxel']

        if self.meanHistory is not None:
            outName = self.name+'_pm_history'
            if hasattr(self,'obsHistoryIts'):
                axes_domains = {'iteration': self.obsHistoryIts}
            else:
                axes_domains = {}
            axes_domains.update(self.axes_domains)

            axes_names = ['iteration'] + an
            outputs[outName] = xndarray(self.meanHistory,
                                      axes_names=axes_names,
                                      axes_domains=axes_domains,
                                      value_label=self.value_label)
        if hasattr(self, 'smplHistory') and self.smplHistory is not None:
            axes_names = ['iteration'] + an
            outName = self.name+'_smpl_history'
            if hasattr(self,'smplHistoryIts'):
                axes_domains = {'iteration': self.smplHistoryIts}
            else:
                axes_domains = {}
            axes_domains.update(self.axes_domains)
            outputs[outName] = xndarray(self.smplHistory,
                                      axes_names=axes_names,
                                      axes_domains=axes_domains,
                                      value_label=self.value_label)


        pyhrf.verbose(4, '%s final value:' %self.name)
        pyhrf.verbose.printNdarray(4, self.finalValue)
        # if 1 and hasattr(self, 'error'):
        #     err = self.error**.5
        # else:
        #     err = None

        # c = xndarray(self.finalValue,
        #            axes_names=self.axes_names,
        #            axes_domains=self.axes_domains,
        #            value_label=self.value_label)

        # outputs[self.name+'_pm'] = c


        axes_names = ['voxel']
        roi_lab_vol = np.zeros(self.nbVox, dtype=np.int32) + \
            self.dataInput.roiId
        outputs['roi_mapping'] = xndarray(roi_lab_vol, axes_names=axes_names,
                                        value_label='ROI')
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            if hasattr(self, 'finalValue_sign_corr'):
                outputs['nrl_sign_corr'] = xndarray(self.finalValue_sign_corr,
                                                  axes_names=self.axes_names,
                                                  axes_domains=self.axes_domains,
                                                  value_label=self.value_label)


            axes_names = ['session', 'condition', 'voxel']
            axes_domains = {'condition' : cn, 'session' : sn}


        if self.dataInput.simulData is not None:
            #trueNrls = self.dataInput.simulData.nrls.data
            trueNrls = self.trueValue
            if trueNrls.shape == self.finalValue.shape:
                axes_names = ['session', 'condition', 'voxel']
                ad = {'condition':cn, 'session' : sn}
                relErrorNrls = abs(trueNrls - self.finalValue)
                outputs['nrl_pm_error'] = xndarray(relErrorNrls,
                                                  axes_names=axes_names,
                                                  axes_domains=ad)

                axes_names = ['session', 'condition']
                nt = (trueNrls.astype(np.float32) - \
                                self.finalValue.astype(np.float32))**2
                outputs['nrl_pm_rmse'] = xndarray(nt.mean(2),
                                                  axes_names=axes_names,
                                                  axes_domains=ad)

        drift = self.getVariable('drift').get_final_value()

        axes_names = ['type','session', 'time', 'voxel']
        outputs['ysignals'] = xndarray(np.array([self.dataInput.varMBY,
                                               self.varYbar,self.sumaXh,
                                               drift]),
                                             axes_names=axes_names,
                                             axes_domains={'type':['Y','Ybar',
                                                                   'sumaXh',
                                                                   'drift']})

        axes_names = ['session', 'time', 'voxel']
        # outputs['ytilde'] = xndarray(self.varYtilde,
        #                          axes_names=axes_names,)

        # outputs['ybar'] = xndarray(self.varYbar,
        #                          axes_names=axes_names,)

        # outputs['sumaXh'] = xndarray(self.sumaXh,
        #                            axes_names=axes_names,)

        outputs['mby'] = xndarray(np.array(self.dataInput.varMBY),
                                   axes_names=axes_names,)

        return outputs


class Variance_GaussianNRL_Multi_Sess(xmlio.XmlInitable, GibbsSamplerVariable):

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=np.array([1.])):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self,'variance_nrls_by_session',
                                      valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        self.nbSessions = self.dataInput.nbSessions

        if dataInput.simulData is not None:
            #self.trueValue = np.array(np.array([dataInput.simulData[s]['nrls_session'] for s in xrange(self.nbSessions)]).var(0))
            sd = dataInput.simulData
            if 0: #theoretical true value
                self.trueValue = np.array([sd[0]['var_sess']])
            else: #empirical true value
                nrl_mean = np.array([ssd['nrls_session'] for ssd in sd]).mean(0)
                v = np.array([ssd['nrls_session'] - nrl_mean \
                              for ssd in sd]).var()
                self.trueValue = np.array([v])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)


    def sampleNextInternal(self, variables):

        nrls = variables[self.samplerEngine.I_NRLS_SESS].currentValue
        nrlsBAR = variables[self.samplerEngine.I_NRLS_BAR].currentValue

        #sum_s_j_m=0
        #for s in xrange(self.nbSessions):
            #for m in xrange(self.nbConditions):
                #for j in xrange(self.nbVoxels):
                    #sum_s_j_m += (nrls[s][m][j] - nrlsBAR[m][j])**2
        sum_s_j_m = ((nrls - nrlsBAR)**2).sum()
        alpha = (self.nbSessions*self.nbConditions*self.nbVoxels-1)/2.
        beta_g  = 0.5*sum_s_j_m
        self.currentValue[0] = 1.0/np.random.gamma(alpha, 1/beta_g)
        #self.currentValue.astype(np.float64)
    #def sampleNextAlt(self, variables):




##################################################
# Mean nrl over sessions Sampler #
##################################################
class NRLsBar_Drift_Multi_Sess_Sampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling.
    """

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
                sd = dataInput.simulData
                self.trueLabels = sd[0]['labels'].astype(np.int32)
                if 0:
                    #theoretical true value:
                    self.trueValue = sd[0]['nrls'].astype(np.float64)
                else:
                    #empirical true value
                    self.trueValue = np.array([ssd['nrls_session']
                                               for ssd in sd]).mean(0)

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


    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):


        labs = self.getFinalLabels()
        # same criterion as np.allclose:
        acc = abs_error <= (atol + rtol * np.maximum(np.abs(tv),
                                                     np.abs(fv)))

        # take only absolute error for inactivated voxels
        atol = .3
        m = np.where(labs==0)
        acc[m] = abs_error[m] <= atol

        return self.axes_names, acc


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
            #pyhrf.verbose.set_verbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.printState(4)
        self.iteration += 1 #TODO : factorize !!

    def setFinalValue(self):
        self.finalValue = self.getMean()

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


    def is_accurate(self):
        tv = self.trueValue
        fv = self.finalValue
        return (((fv - tv)**2).sum() / (tv**2).sum())**.5 < 0.02

##################################################
# Bigaussian parameters Sampler #
##################################################
class BiGaussMixtureParams_Multi_Sess_NRLsBar_Sampler(xmlio.XmlInitable,
                                                      GibbsSamplerVariable):

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    # #"peaked" priors
    # defaultParameters = {
    #     P_VAL_INI : None,
    #     P_SAMPLE_FLAG : True,
    #     P_USE_TRUE_VALUE : False,
    #     #P_HYPER_PRIOR : 'Jeffrey',
    #     P_HYPER_PRIOR : 'proper',
    #     P_MEAN_CA_PR_MEAN : 5.,
    #     P_MEAN_CA_PR_VAR : 20.0,
    #     P_VAR_CI_PR_ALPHA : 2.04,
    #     P_VAR_CI_PR_BETA : .5,#2.08,
    #     P_VAR_CA_PR_ALPHA : 2.01,
    #     P_VAR_CA_PR_BETA : .5,
    #     P_ACTIV_THRESH : 4.,
    #     }



    ##"flat" priors
    #defaultParameters = {
        #P_VAL_INI : None,
        #P_SAMPLE_FLAG : True,
        #P_USE_TRUE_VALUE : False,
        ##P_HYPER_PRIOR : 'Jeffreys',
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

    parametersComments = {
        'hyper_prior_type' : "Either 'proper' or 'Jeffreys'",
        'activ_thresh' : "Threshold for the max activ mean above which the "\
            "region is considered activating",
        }

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None, hyper_prior_type='Jeffreys', activ_thresh=4.,
                 var_ci_pr_alpha=2.04, var_ci_pr_beta=.5,
                 var_ca_pr_alpha=2.01, var_ca_pr_beta=.5,
                 mean_ca_pr_mean=5., mean_ca_pr_var=20.):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)

        # get values for priors :
        self.varCIPrAlpha = var_ci_pr_alpha
        self.varCIPrBeta = var_ci_pr_beta
        self.varCAPrAlpha = var_ca_pr_alpha
        self.varCAPrBeta = var_ca_pr_beta

        self.meanCAPrMean = mean_ca_pr_mean
        self.meanCAPrVar = mean_ca_pr_var

        an = ['component','condition']
        ad = {'component' : self.PARAMS_NAMES}
        GibbsSamplerVariable.__init__(self, 'mixt_params', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      sampleFlag=do_sampling, axes_names=an,
                                      axes_domains=ad)

        self.hyperPriorFlag = (hyper_prior_type == 'Jeffreys')

        self.activ_thresh = activ_thresh

    def linkToData(self, dataInput):
        self.dataInput =  dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX

        self.nrlCI = range(self.nbConditions)
        self.nrlCA = range(self.nbConditions)

        #print 'self.dataInput : ', self.dataInput
        #print 'isinstance(self.dataInput, BOLDModel) : ', isinstance(self.dataInput, BOLDModel)


        if self.dataInput.simulData is not None and \
            isinstance(self.dataInput.simulData, list):
            if self.dataInput.simulData[0].has_key('condition_defs'):

                #take only 1st session -> same is assumed for others
                sd = self.dataInput.simulData
                cdefs = sd[0]['condition_defs']
                self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                                          dtype=float)
                if 0:
                    #Theorethical true values:
                    mean_act = np.array([c.m_act for c in cdefs])
                    var_act = np.array([c.v_act for c in cdefs])
                    var_inact = np.array([c.v_inact for c in cdefs])
                else:
                    #Empirical true values:
                    m_act = np.where(sd[0]['labels'] == self.L_CA)
                    m_inact = np.where(sd[0]['labels'] == self.L_CI)
                    all_nrls = np.array([ssd['nrls'] for ssd in sd])
                    nbc = self.nbConditions
                    mean_act = np.array([all_nrls[:,j,m_act[0]].mean() \
                                         for j in xrange(nbc)])
                    var_act = np.array([all_nrls[:,j,m_act[0]].var() \
                                         for j in xrange(nbc)])
                    var_inact = np.array([all_nrls[:,j,m_inact[0]].var() \
                                         for j in xrange(nbc)])

                self.trueValue[self.I_MEAN_CA] = mean_act
                self.trueValue[self.I_VAR_CA] = var_act
                self.trueValue[self.I_VAR_CI] = var_inact


        if self.dataInput.simulData is not None and \
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
        outputs = GibbsSamplerVariable.getOutputs(self)

        if pyhrf.__usemode__ == pyhrf.DEVEL:

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
        NRLSampler.finalizeSampling(self)
        del self.nrlCA
        del self.nrlCI



##########################################################
# BOLD sampler input and BOLDSampler for multisessions
##########################################################

class BOLDSampler_Multi_SessInput :

    """
    Class holding data needed by the sampler : BOLD time courses for each voxel,
    onsets and voxel topology.
    It also perform some precalculation such as the convolution matrix based on
    the onsests (L{stackX})
    ----
    Multi-sessions version
    """

    def __init__(self, data, dt, typeLFD, paramLFD, hrfZc, hrfDuration) :
        """
        Initialize a BOLDSamplerInput object. Mainly extract data from boldData.
        """
        pyhrf.verbose(3, 'BOLDSamplerInput init ...')
        pyhrf.verbose(3, 'Received data:')
        pyhrf.verbose(3, data.getSummary())

        self.roiId = data.get_roi_id()
        self.nbSessions = len(data.sessionsScans)
        #names of sessions:
        self.sNames = ['Session%s' %str(i+1) for i in xrange(self.nbSessions)]
        self.nys = [len(ss) for ss in data.sessionsScans]
        # Scan_nb=[data.sessionsScans[0]]
        # for s in xrange(self.nbSessions-1):
        #     s=s+1
        #     Scan_nb.append(np.array(data.sessionsScans[s]) + len(data.sessionsScans[s-1]))
        # Scan_nb = np.array(Scan_nb)

        self.varMBY = np.array([data.bold[ss,:] for ss in data.sessionsScans])

        self.ny = self.nys[0]
        self.varData = np.array([self.varMBY[ss].var(0) for ss in xrange(self.nbSessions)])
        self.tr = data.tr

        # if any ... would be used to compare results:
        self.simulData = data.simulation

        graph = data.get_graph()
        self.nbVoxels = len(graph)

        nmax = max([len(nl) for nl in graph])

        self.neighboursIndexes = np.zeros((self.nbVoxels, nmax), dtype=int)
        self.neighboursIndexes -= 1
        for i in xrange(self.nbVoxels):
            self.neighboursIndexes[i,:len(graph[i])] = graph[i]
        self.nbCliques = graph_nb_cliques(self.neighboursIndexes)
        # Store some parameters usefull for analysis :
        self.typeLFD = typeLFD
        self.paramLFD = paramLFD

        # Treat onset to be consistent with BOLD signal
        # build convol matrices according to osfMax and hrf parameters
        pyhrf.verbose(3, 'Chewing up onsets ...')
        self.paradigm = data.paradigm
        self.cNames = data.paradigm.get_stimulus_names()
        self.nbConditions = len(self.cNames)
        onsets = data.paradigm.stimOnsets
        self.onsets = [onsets[cn] for cn in self.cNames]
        durations = data.paradigm.stimDurations
        self.durations = [durations[cn] for cn in self.cNames]
        #print 'durations are :', self.durations
        self.chewUpOnsets(dt, hrfZc, hrfDuration)

        # Build matrices related to low frequency drift
        pyhrf.verbose(3, 'Building LFD mats %% ...')
        self.setLFDMat(paramLFD,typeLFD)

        pyhrf.verbose(3, 'Making precalculcations ...')
        self.makePrecalculations()


    def makePrecalculations(self):
        # XQX & XQ:
        XQX=[]
        XQ=[]
        Qy=[]
        yTQ=[]
        yTQy=[]
        XtX=[]
        for iSess in xrange(self.nbSessions):
            self.matXQX = np.zeros( (self.nbConditions, self.nbConditions,
                                        self.nbColX, self.nbColX),dtype=float)
            self.matXQ = np.zeros((self.nbConditions, self.nbColX, self.ny),
                                    dtype=float)

            for j in xrange(self.nbConditions):
                self.matXQ[j,:,:] = np.dot(self.varX[iSess,j,:,:].transpose(),
                                            self.delta[iSess] )
                for k in xrange(self.nbConditions):
                    self.matXQX[j,k,:,:] = np.dot( self.matXQ[j,:,:],
                                                    self.varX[iSess,k,:,:] )
            XQX.append(self.matXQX)
            XQ.append(self.matXQ)
            # Qy, yTQ & yTQy  :
            self.matQy = np.zeros((self.ny,self.nbVoxels), dtype=float)
            self.yTQ = np.zeros((self.ny,self.nbVoxels), dtype=float)
            self.yTQy = np.zeros(self.nbVoxels, dtype=float)

            for i in xrange(self.nbVoxels):
                self.matQy[:,i] = np.dot(self.delta[iSess],self.varMBY[iSess][:,i])
                self.yTQ[:,i] = np.dot(self.varMBY[iSess][:,i],self.delta[iSess])
                self.yTQy[i] = np.dot(self.varMBY[iSess][:,i],self.matQy[:,i])


            self.matXtX = np.zeros( (self.nbConditions, self.nbConditions,
                        self.nbColX, self.nbColX),dtype=float)
            for j in xrange(self.nbConditions):
                for k in xrange(self.nbConditions):
                    self.matXtX[j,k,:,:] = np.dot( self.varX[iSess,j,:,:].transpose(),
                    self.varX[iSess,k,:,:] )

            Qy.append(self.matQy)
            yTQ.append(self.yTQ)
            yTQy.append(self.yTQy)
            XtX.append(self.matXtX)

        self.matXQX = np.array(XQX)
        self.matXQ = np.array(XQ)
        self.matQy = np.array(Qy)
        self.yTQ   = np.array(yTQ)
        self.yTQy  = np.array(yTQy)
        self.matXtX = np.array(XtX)

    def cleanPrecalculations(self):

        del self.yTQ
        del self.yTQy
        del self.matQy
        del self.matXQ
        del self.matXQX


    def chewUpOnsets(self, dt, hrfZc, hrfDuration):

        #print 'onsets:', self.onsets
        pyhrf.verbose(1, 'Chew up onsets ...')
        if dt > 0.:
            self.dt = dt
        else:
            self.dt = self.calcDt(-dt)

        pyhrf.verbose(1, 'dt = %1.3f' %self.dt)

        nbc = self.nbConditions

        self.stimRepetitions = [len(self.onsets[ind]) for ind in xrange(nbc)]

        pyhrf.verbose(5, 'nb of Trials :')
        pyhrf.verbose.printNdarray(5, self.stimRepetitions)

        pyhrf.verbose(3, 'computing sampled binary onset sequences ...')

        rp = self.paradigm.get_rastered(self.dt)
        self.paradigmData = [rp[n] for n in self.cNames]

        pyhrf.verbose(3, 'building paradigm convol matrix ...')
        availIdx = [np.arange(0,n, dtype=int) for n in self.nys]
        self.buildParadigmConvolMatrix(hrfZc, hrfDuration, availIdx,
                                       self.paradigmData)
        pyhrf.verbose(5, 'matrix X : %s' %str(self.varX.shape))


    def setLFDMat(self, paramLFD, typeLFD): #TODO : factoriser eventuellement
                                            # avec fonction deja presente dans
                                            # boldsynth ...
        """
        Build the low frequency basis from polynomial basis functions.

        """
        #ppt = []
        self.lfdMat = []
        self.delta = []
        self.varPtP = []
        pyhrf.verbose(3, 'LFD type :' + typeLFD)
        for iSess in xrange(self.nbSessions):
            if typeLFD == 'polynomial':
                lfdMat = self.buildPolyMat( paramLFD , self.nys[iSess])
            elif typeLFD == 'cosine':
                lfdMat = self.buildCosMat( paramLFD , self.nys[iSess])
            elif typeLFD == 'None':
                lfdMat = np.zeros((self.nys[iSess],2))

            pyhrf.verbose(3, 'LFD Matrix :')
            pyhrf.verbose.printNdarray(3, lfdMat)
            #print lfdMat
            self.lfdMat.append(lfdMat)
            varPPt = np.dot(lfdMat, lfdMat.transpose())
            if typeLFD is not 'None':
                self.colP = np.shape(lfdMat)[1]
            else:
                self.colP = 0

            self.delta.append(np.eye(self.nys[iSess], dtype=float) - varPPt)
            self.varPtP.append(np.dot(lfdMat.transpose(), lfdMat))

            pyhrf.verbose(6, 'varPtP :')
            pyhrf.verbose.printNdarray(6, self.varPtP[-1])
            if typeLFD != 'None':
                assert np.allclose(self.varPtP[-1],
                                      np.eye(self.colP, dtype=float),
                                      rtol=1e-5 )


    def buildPolyMat( self, paramLFD , n ):

        regressors = self.tr * np.arange(0, n)
        timePower = np.arange(0,paramLFD+1, dtype=int)
        regMat = np.zeros((len(regressors),paramLFD+1),dtype=float)
        for v in xrange(paramLFD+1):
            regMat[:,v] = regressors[:]

        tPowerMat = np.matlib.repmat(timePower, n, 1)
        lfdMat = np.power(regMat,tPowerMat)
        lfdMat = np.array(scipy.linalg.orth(lfdMat))
        return lfdMat

    def buildCosMat( self, paramLFD , ny):
        n = np.arange(0,ny)
        fctNb = np.fix(2*(ny*self.tr)/paramLFD + 1.);# +1 stands for the
                                                        # mean/cst regressor
        lfdMat = np.zeros( (ny, fctNb), dtype=float)
        lfdMat[:,0] = np.ones( ny, dtype= float) / np.sqrt(ny)
        samples = 1. + np.arange(fctNb-2)
        for k in samples:
          lfdMat[:,k] = np.sqrt(2/ny) \
                        * np.cos( np.pi*(2.*n+1.)*k / (2*ny) )
        return lfdMat

    def buildParadigmConvolMatrix(self, zc, estimDuration, availableDataIndex,
                                  parData) :
        osf = self.tr/self.dt

        pyhrf.verbose(2, 'osf = %1.2f' %osf)
        pyhrf.verbose(6, 'availableDataIndex :')
        pyhrf.verbose.printNdarray(6, availableDataIndex)

        lgt = (self.ny+2)*osf
        allMatH = []
        for iSess in xrange(self.nbSessions):
            matH = np.zeros( (lgt, self.nbConditions), dtype=int)
            for j in xrange(self.nbConditions) :
                matH[:len(parData[j][iSess]), j] = parData[j][iSess][:]
            pyhrf.verbose(6, 'matH for Sess %d :' %iSess)
            if pyhrf.verbose.verbosity >= 6:
                for a in xrange(matH.shape[0]):
                    print ' [',
                    for b in xrange(matH.shape[1]):
                        print matH[a,b],
                    print ']'

            allMatH.append(matH)

        self.hrfLength = len(np.arange(0, estimDuration+self.dt, self.dt))
        pyhrf.verbose(5, 'hrfLength = int(round(%d/%1.2g))=%d' \
                          % (estimDuration,self.dt,self.hrfLength))
        if zc :
            self.hrfColIndex = np.arange(1, self.hrfLength-1, dtype=int)
            self.colIndex = np.arange(0, self.hrfLength-2, dtype=int)
        else :
            self.hrfColIndex = np.arange(0, self.hrfLength, dtype=int)
            self.colIndex = np.arange(0, self.hrfLength, dtype=int)
        self.lgCI = len(self.colIndex)
        pyhrf.verbose(5, 'lgCI = %d'% self.lgCI)

        self.varOSAvailDataIdx = [np.array(ai*osf, dtype=int)
                                  for ai in availableDataIndex]
        vX = []
        pyhrf.verbose(2, 'Build pseudo teoplitz matrices')
        for iSess in xrange(self.nbSessions):
            self.lenData = len(self.varOSAvailDataIdx[iSess])
            varX = np.zeros( (self.nbConditions, self.lenData, self.lgCI),
                          dtype=int )
            pyhrf.verbose(6, 'iSess : %d' %iSess)
            for j in xrange(self.nbConditions):
                pyhrf.verbose(6, ' cond : %d' %j)
                col = np.concatenate(([allMatH[iSess][0,j]],
                                   np.zeros(self.hrfLength-1, dtype=int)))
                pyhrf.verbose(6, ' col :')
                if pyhrf.verbose.verbosity >= 6:
                    print ' [',
                    for b in xrange(col.shape[0]):
                        print col[b],
                    print ']'


                matTmp = np.array(scipy.linalg.toeplitz( allMatH[iSess][:,j], col), dtype=int)
                pyhrf.verbose(6, ' matTmp :')
                if pyhrf.verbose.verbosity >= 6:
                    for b in xrange(matTmp.shape[0]):
                        print ' [',
                        for a in xrange(matTmp.shape[1]):
                            print matTmp[b,a],
                        print ']'
                d0 = matTmp[self.varOSAvailDataIdx[iSess],:]
                d1 = d0[:,self.hrfColIndex]
                varX[j,:,:] = d1

            vX.append(varX)
        #self.varX = hstack(vX)
        self.varX = np.array(vX)
        pyhrf.verbose(4, 'varX : ' + str(self.varX.shape))
        self.buildOtherMatX()

        self.nbColX = np.shape(self.varX[0])[2]

    def buildOtherMatX(self):
        varMBX=[]
        stackX=[]
        Id=[]

        for s in xrange(self.nbSessions):
            self.varMBX = np.zeros( (self.ny, self.nbConditions*self.lgCI),
                             dtype=int)
            self.stackX = np.zeros( (self.ny*self.nbConditions, self.lgCI),
                             dtype=int)

            for j in xrange(self.nbConditions):
                self.varMBX[:, j*self.lgCI+self.colIndex] = self.varX[s, j,:,:]

                self.stackX[self.ny*j:self.ny*(j+1), :] = self.varX[s, j,:,:]


            self.notNullIdxStack = np.dstack(np.where(self.stackX != 0)).ravel()

            varMBX.append(self.varMBX)
            stackX.append(self.stackX)
            Id.append(self.notNullIdxStack)

        self.varMBX = np.array(varMBX)
        self.stackX = np.array(stackX)
        self.notNullIdxStack = np.array(Id, dtype=object)




    def calcDt(self, dtMin) :

        pyhrf.verbose(2, 'dtMin = %1.3f' %dtMin)
        pyhrf.verbose(2, 'Trying to set dt automatically from data')

        tr = self.tr
        vectSOA = np.array([], dtype=float)
        for ons in self.onsets:
            vectSOA = np.concatenate((vectSOA, ons))

        vectSOA.sort()
        pyhrf.verbose(5, 'vectSOA %s:' %str(vectSOA.shape))
        pyhrf.verbose.printNdarray(5,vectSOA)

        momRT = np.arange(0, vectSOA[-1]+tr, tr)
        pyhrf.verbose(5,'momRT %s:' %str(momRT.shape))
        pyhrf.verbose.printNdarray(5, momRT)

        momVect = np.concatenate((vectSOA, momRT))
        momVect.sort()

        varSOA = np.diff(momVect)
        pyhrf.verbose(5, 'vectSOA diff:')
        pyhrf.verbose.printNdarray(5, vectSOA)

        nonZeroSOA = varSOA[np.where(varSOA > 0.0)]
        pyhrf.verbose(5, 'nonZeroSOA :')
        pyhrf.verbose.printNdarray(5, nonZeroSOA)

        delta = nonZeroSOA.min()
        pyhrf.verbose(5, 'delta : %1.3f' %delta)
        pyhrf.verbose(5, 'dtMin : %1.3f' %dtMin)
        dt = max(dtMin, delta)
        pyhrf.verbose(2, 'dt from onsets: %1.2f, dtMin=%1.3f => '\
                          'dt=%1.2f (osf=%1.3f)'
                      %(delta, dtMin, dt, (tr+0.)/dt))
        return dt



    def cleanMem(self):

        pyhrf.verbose(3, 'cleaning Memory BOLD Sampler Input')

        del self.varMBX

        del self.varOSAvailDataIdx
        del self.notNullIdxStack
        #del self.onsets
        del self.colP
        del self.delta
        del self.lfdMat
        del self.varPtP
        #del self.varMBY
        del self.paradigmData
        del self.neighboursIndexes
        #del self.varSingleCondXtrials
        del self.stimRepetitions
        del self.hrfColIndex
        del self.colIndex
        self.cleanPrecalculations()


class BOLDGibbs_Multi_SessSampler(xmlio.XmlInitable, GibbsSampler):

    #TODO : comment


    inputClass = BOLDSampler_Multi_SessInput

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        default_nb_its = 3
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        default_nb_its = 3000
        parametersToShow = ['nb_its', 'response_levels', 'hrf', 'hrf_var']

    parametersComments = {
        'smpl_hist_pace': 'To save the samples at each iteration\n'\
            'If x<0: no save\n ' \
            'If 0<x<1: define the fraction of iterations for which samples are '\
            'saved\n'\
            'If x>=1: define the step in iterations number between saved '\
            ' samples.\n'\
            'If x=1: save samples at each iteration.',
        'obs_hist_pace' : 'See comment for samplesHistoryPaceSave.'
        }


    def __init__(self, nb_its=default_nb_its,
                 obs_hist_pace=-1., glob_obs_hist_pace=-1,
                 smpl_hist_pace=-1., burnin=.3,
                 callback=GSDefaultCallbackHandler(),
                 response_levels_sess=NRL_Multi_Sess_Sampler(),
                 response_levels_mean=NRLsBar_Drift_Multi_Sess_Sampler(),
                 beta=BetaSampler(),
                 noise_var=NoiseVariance_Drift_Multi_Sess_Sampler(),
                 hrf=HRF_MultiSess_Sampler(),
                 hrf_var=RHSampler(), mixt_weights=MixtureWeightsSampler(),
                 mixt_params=BiGaussMixtureParamsSampler(), scale=ScaleSampler(),
                 drift=Drift_MultiSess_Sampler(),
                 drift_var=ETASampler_MultiSess(), stop_crit_threshold=-1,
                 stop_crit_from_start=False, check_final_value=None):
        """
        check_final_value: None, 'print' or 'raise'
        """
        #print 'param:', parameters
        xmlio.XmlInitable.__init__(self)

        variables = [response_levels_sess, response_levels_mean, hrf, hrf_var,
                     mixt_weights, mixt_params, beta, scale, noise_var, drift,
                     drift_var]

        nbIt = nb_its
        obsHistPace = obs_hist_pace
        globalObsHistPace = glob_obs_hist_pace
        smplHistPace = smpl_hist_pace
        nbSweeps = burnin

        check_ftval = check_final_value

        if obsHistPace > 0. and obsHistPace < 1:
            obsHistPace = max(1,int(round(nbIt * obsHistPace)))

        if globalObsHistPace > 0. and globalObsHistPace < 1:
            globalObsHistPace = max(1,int(round(nbIt * globalObsHistPace)))

        if smplHistPace > 0. and smplHistPace < 1.:
            smplHistPace = max(1,int(round(nbIt * smplHistPace)))

        if nbSweeps > 0. and nbSweeps < 1.:
            nbSweeps = int(round(nbIt * nbSweeps))

        #pyhrf.verbose(2,'smplHistPace: %d'%smplHistPace)
        #pyhrf.verbose(2,'obsHistPace: %d'%obsHistPace)

        self.stop_threshold = stop_crit_threshold
        self.crit_diff_from_start = stop_crit_from_start
        #callbackObj = self.parameters[self.P_CALLBACK]
        self.full_crit_diff_trajectory = defaultdict(list)
        self.full_crit_diff_trajectory_timing = []
        self.crit_diff0 = {}
        #print 'self.crit_diff_from_start:', self.crit_diff_from_start
        callbackObj = GSDefaultCallbackHandler()
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps, callbackObj,
                              globalObsHistoryPace=globalObsHistPace,
                              check_ftval=check_ftval)



    def stop_criterion(self, it):
        #return False
        if it < self.nbSweeps+1 or self.stop_threshold < 0.:
            return False
        epsilon = self.stop_threshold
        diffs = np.array([d for d in self.crit_diff.values() ])
        pyhrf.verbose(3, "Stop criterion (it=%d):" %it)
        for k,v in self.crit_diff.iteritems():
            pyhrf.verbose(3, " - %s : %f < %f -> %s" \
                              %(k,v,epsilon,str(v<epsilon)))
        return (diffs < epsilon).all()

    def compute_crit_diff(self, old_vals, means=None):
        crit_diff = {}
        for vn, v in self.variablesMapping.iteritems():
            if means is None:
                new_val = v.mean.flatten()
            else:
                new_val = means[vn]
            old_val = old_vals[vn]
            diff = ((new_val - old_val)**2).sum() / (old_val**2).sum()

            old_vals[vn] = new_val
            crit_diff[vn] = diff

        return crit_diff

    def initGlobalObservables(self):

        if self.stop_threshold >= 0.:
            self.crit_diff = {}
            self.conv_crit_diff = defaultdict(list)
            self.variables_old_val = {}

            self.old_values_for_full_crit_diff = {}
            self.cumul_for_full_crit_diff = {}
            self.means_for_full_crit_diff = {}
            for vn, v in self.variablesMapping.iteritems():
                val = v.currentValue.flatten()
                self.variables_old_val[vn] = val
                self.old_values_for_full_crit_diff[vn] = val
                self.cumul_for_full_crit_diff[vn] = val.copy()

    def updateGlobalObservables(self):
        if self.stop_threshold >= 0.:
            self.crit_diff.update(self.compute_crit_diff(self.variables_old_val))

    def cleanObservables(self):
        #TODO: more to clean ?
        del self.variables_old_val

    def saveGlobalObservables(self, it):
        #print 'saveGlobalObservables ...'
        GibbsSampler.saveGlobalObservables(self, it)
        if self.stop_threshold >= 0.:
            for vn, d in self.crit_diff.iteritems():
                self.conv_crit_diff[vn].append(d)

    def finalizeSampling(self):
        return

        class DummyVariable(): pass
        msg = []
        report = defaultdict(dict)
        if self.check_ftval is not None:

            for v in self.variables+['labels']:

                if v == 'labels':
                    v = DummyVariable()
                    v.name = 'labels'
                    v.finalValue = snrls.finalLabels
                    v.trueValue = snrls.trueLabels

                if v.trueValue is None:
                    print 'Warning: no true val for %s' %v.name
                elif not v.sampleFlag and v.useTrueValue:
                    continue
                else:
                    fv = v.finalValue
                    tv = v.trueValue
                    rtol = 0.1
                    atol = 0.1

                    if v.name == 'drift':
                        fv = np.array([np.dot(v.P[s], v.finalValue[s]) \
                                        for s in xrange(v.nbSess)])
                        tv = np.array([np.dot(v.P[s], v.trueValue[s]) \
                                        for s in xrange(v.nbSess)])
                    elif v.name == 'hrf':
                        delta = (((fv - tv)**2).sum() / (tv**2).sum())**.5
                        report['hrf']['is_accurate'] = delta < 0.05
                    elif v.name == 'labels':
                        delta = (fv != tv).sum()*1. / fv.shape[1]
                        report['labels']['is_accurate'] = delta < 0.05

                    abs_error = np.abs(tv - fv)
                    report[v.name]['abs_error'] = abs_error
                    report[v.name]['rel_error'] = abs_error / np.maximum(tv,fv)

                    # same criterion as np.allclose:
                    nac = abs_error >= (atol + rtol * np.maximum(np.abs(tv),
                                                                 np.abs(fv)))
                    report[v.name]['not_close'] = nac

                    if report[v.name].get('is_accurate') is None:
                        report[v.name]['is_accurate'] = not nac.any()

                    if not report[v.name]['is_accurate']:
                        m = "Final value of %s is not close to " \
                        "true value.\n -> aerror: %s\n -> rerror: %s\n" \
                        " Final value:\n %s\n True value:\n %s\n" \
                        %(v.name, array_summary(report[v.name]['abs_error']),
                          array_summary(report[v.name]['rel_error']),
                          str(fv), str(tv))
                        msg.append(m)
                        if self.check_ftval == 'raise':
                            raise Exception(m)

            if self.check_ftval == 'print':
                print "\n".join(msg)

        self.check_ftval_report = report

    def computeFit(self):
        nbVox = self.dataInput.nbVoxels
        nbSess = self.dataInput.nbSessions

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if hrf is None:
            hrf = shrf.currentValue
        elif shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl_by_session').finalValue
        if nrl is None:
            nrl = self.getVariable('nrl_by_session').currentValue

        stimIndSignal = np.zeros((nbSess, nbVals, nbVox), dtype=np.float32)
        drift = self.getVariable('drift').get_final_value()
        for s in xrange(nbSess):
            stimIndSignal[s] = np.dot(vXh[s], nrl[s]) + drift[s]

        return stimIndSignal


    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)
        axes_domains = {'time' : np.arange(self.dataInput.ny)*self.dataInput.tr}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            # output of design matrix:
            dMat = np.zeros_like(self.dataInput.varX[0,0,:,:])
            for ic,vx in enumerate(self.dataInput.varX[0]):
                dMat += vx * (ic+1)

            outputs['matX_first_sess'] = xndarray(dMat, axes_names=['time','P'],
                                                axes_domains=axes_domains,
                                                value_label='value')
        return outputs


        if self.globalObsHistoryIts is not None:
            if hasattr(self,'conv_crit_diff'):
                it_axis = self.globalObsHistoryIts
                print 'it_axis:'
                print it_axis
                if len(it_axis) > 1:
                    it_axis = np.arange(it_axis[0],self.nbIterations,
                                        it_axis[1]-it_axis[0])
                axes_domains = {'iteration':it_axis}
                print 'it_axis filled:'
                print it_axis

                it_durations = np.array(self.globalObsHistoryTiming)
                print 'it_durations:', len(it_durations)
                print it_durations
                if len(it_durations) > 1:
                    c = np.arange(it_axis[0],
                                  self.nbIterations-len(it_durations)) * \
                        np.diff(it_durations).mean() + it_durations[-1]
                    it_durations = np.concatenate((it_durations,c))

                print 'it_durations filled:', len(it_durations)
                print it_durations

                axes_domains = {'duration':it_durations}
                for k,v in self.conv_crit_diff.iteritems():
                    conv_crit = np.zeros(len(it_axis)) - .001
                    conv_crit[:len(v)] = v
                    c = xndarray(conv_crit,
                               axes_names=['duration'],
                               axes_domains=axes_domains,
                               value_label='conv_criterion')

                    outputs['conv_crit_diff_timing_from_burnin_%s'%k] = c

        if hasattr(self, 'full_crit_diff_trajectory'):
            try:
                #print 'full_crit_diff_trajectory:'
                #print self.full_crit_diff_trajectory
                it_axis = np.arange(self.nbIterations)
                axes_domains = {'iteration':it_axis}

                it_durations = self.full_crit_diff_trajectory_timing
                axes_domains = {'duration':it_durations}
                for k,v in self.full_crit_diff_trajectory.iteritems():
                    conv_crit = np.zeros(len(it_axis)) - .001
                    conv_crit[:len(v)] = v
                    c = xndarray(conv_crit,
                               axes_names=['duration'],
                               axes_domains=axes_domains,
                               value_label='conv_criterion')

                    outputs['conv_crit_diff_timing_from_start_%s'%k] = c
            except Exception, e:
                print 'Could not save output of convergence crit'
                print e

        return outputs

    def computePMStimInducedSignal(self):

        nbVox = self.dataInput.nbVoxels
        nbSess = self.dataInput.nbSessions

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl_by_session').finalValue

        self.stimIndSignal = np.zeros((nbSess, nbVals, nbVox))
        meanBold = np.zeros((nbSess, nbVox))

        for s in xrange(nbSess):
            meanBold[s] = self.dataInput.varMBY[s].mean(axis=0)
            for i in xrange(nbVox):
                # Multiply by corresponding NRLs and sum over conditions :
                si = (vXh[s]*nrl[s,:,i]).sum(1)
                # Adjust mean to original Bold (because drift is not explicit):
                self.stimIndSignal[s,:,i] = si-si.mean() +  meanBold[s,i]




##########################################
## for simulations #######################
##########################################
from pyhrf import Condition
import pyhrf.boldsynth.scenarios as msimu
from pyhrf.tools import Pipeline
from pyhrf.core import FmriData

def simulate_single_session(output_dir, var_sessions_nrls, cdefs, nrls_bar,
                            labels, labels_vol, v_noise, drift_coeff_var,
                            drift_amplitude):
    dt = .5
    dsf = 2 #down sampling factor dt/TR TR=2.3s for language paradigm

    simulation_steps = {
        'dt' : dt,
        'dsf' : dsf,
        'tr' : dt * dsf,
        'var_sess' : var_sessions_nrls,
        'condition_defs' : cdefs,
        # Paradigm
        'paradigm' : msimu.create_localizer_paradigm_avd,
        'rastered_paradigm' : msimu.rasterize_paradigm,
        # Labels
        'labels_vol' : labels_vol,
        'labels' : labels,
        'nb_voxels': labels.shape[1],
        # Nrls
        'nrls' : nrls_bar,#create_time_invariant_gaussian_nrls,
        'nrls_session' : msimu.create_gaussian_nrls_sessions_and_mean,
        # HRF
        'primary_hrf' : msimu.create_canonical_hrf,
        'hrf' : msimu.duplicate_hrf,
        # Stim induced
        'stim_induced_signal' : msimu.create_multisess_stim_induced_signal,
        # Noise
        'v_gnoise' : v_noise,
        'v_noise' : msimu.duplicate_noise_var,
        'noise' : msimu.create_gaussian_noise,
        # Drift
        'drift_order' : 4,
        'drift_coeff_var' : drift_coeff_var,
        'drift_amplitude' : drift_amplitude,
        'drift_mean' : 0.,
        'drift_coeffs': msimu.create_drift_coeffs,
        'drift' : msimu.create_polynomial_drift_from_coeffs,
        # Bold
        'bold_shape' : msimu.get_bold_shape,
        'bold' : msimu.create_bold_from_stim_induced,
        }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    if 0 and output_dir is not None:
        simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))

    # Retrieve all results
    simulation = simu_graph.get_values()

    # Save outputs o simulation: nii volumes:
    if output_dir is not None:
        msimu.simulation_save_vol_outputs(simulation, output_dir)

    # print 'Simulation done.'
    # print ''

    return simulation


def simulate_sessions(output_dir, snr_scenario='high_snr', spatial_size='tiny'):

    drift_coeff_var = 1.
    drift_amplitude = 10.

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    elif spatial_size == 'random_small':
        lmap1, lmap2, lmap3 = 'random_small', 'random_small', 'random_small'
    else:
        lmap1, lmap2, lmap3 = 'pacman', 'cat2', 'house_sun'

    var_sessions_nrls = .1

    if snr_scenario == 'low_snr': #low snr
        vars_noise=[.6, .4, 15., 17.]
        conditions = [
            Condition(name='audio', m_act=3., v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', m_act=2.5, v_act=.3, v_inact=.3,
                      label_map=lmap2),
            Condition(name='damier', m_act=2, v_act=.3, v_inact=.3,
                      label_map=lmap3),
            ]
    else: #high snr
        vars_noise=[.01, .02, .04, .05]
        conditions = [
            Condition(name='audio', m_act=13., v_act=.2, v_inact=.1,
                      label_map=lmap1),
            Condition(name='video', m_act=11.5, v_act=.2, v_inact=.1,
                      label_map=lmap2),
            Condition(name='damier', m_act=10, v_act=.2, v_inact=.1,
                      label_map=lmap3),
            ]

    nb_sessions = len(vars_noise)

    # Common variable across sessions:
    labels_vol = msimu.create_labels_vol(conditions)
    labels     = msimu.flatten_labels_vol(labels_vol)
    # nrls_bar is the same for all sessions, must be computed before loop over sessions
    nrls_bar = msimu.create_time_invariant_gaussian_nrls(conditions, labels)

    simu_sessions = []
    simus = []
    for i_session in xrange(nb_sessions):
        if output_dir is not None:
            out_dir = op.join(output_dir, 'session_%d' %i_session)
            if not op.exists(out_dir): os.makedirs(out_dir)
        else:
            out_dir = None
        s = simulate_single_session(out_dir, var_sessions_nrls, conditions,
                                    nrls_bar,
                                    labels, labels_vol, vars_noise[i_session],
                                    drift_coeff_var, drift_amplitude)
        simus.append(s)
        simu_sessions.append(FmriData.from_simulation_dict(s))

    if output_dir is not None:
        mask_vol = np.ones_like(simus[0]['labels_vol'][0])
        for ic,c in enumerate(simus[0]['condition_defs']):
            mean_nrls_session = np.array([s['nrls_session'] \
                                          for s in simus]).mean(0)

            fn = op.join(output_dir, 'nrls_session_mean_%s.nii' \
                                    %(c.name))
            write_volume(expand_array_in_mask(mean_nrls_session[ic,:],mask_vol),
                         fn)


    return simu_sessions


