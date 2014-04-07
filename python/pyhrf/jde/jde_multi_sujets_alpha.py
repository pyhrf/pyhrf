# -*- coding: utf-8 -*-
import pyhrf
import numpy as np
import scipy
from numpy.random import randn
import time

import os
import os.path as op

from numpy.testing import assert_almost_equal #assert_array_equal,

from pyhrf.jde.noise import NoiseVariance_Drift_Sampler
#from pyhrf.jde.beta import BetaSampler
from pyhrf.jde.samplerbase import GSDefaultCallbackHandler
from pyhrf import xmlio
from pyhrf.graph import graph_nb_cliques
from pyhrf.boldsynth.hrf import getCanoHRF, genGaussianSmoothHRF
from pyhrf.core import FmriData, FmriGroupData
from pyhrf.jde.intensivecalc import computeYtilde
from pyhrf.jde.intensivecalc import sample_potts
from pyhrf.ndarray import xndarray, stack_cuboids

from collections import defaultdict

from pyhrf.jde.samplerbase import GibbsSampler, GibbsSamplerVariable

from pyhrf.tools import get_2Dtable_string

def b():
    raise Exception

##################################################
# Noise Sampler #
##################################################
class NoiseVariance_Drift_MultiSubj_Sampler(NoiseVariance_Drift_Sampler):

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = []

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=val_ini,
                                        useTrueValue=use_true_value,
                                        sampleFlag=do_sampling,
                                        axes_names=['subject','voxel'],
                                        value_label='PM Noise Var')

    def linkToData(self, dataInput):
        NoiseVariance_Drift_Sampler.linkToData(self, dataInput)
        self.nySubj = self.dataInput.nySubj
        self.nbSubj = self.dataInput.nbSubj
        self.nbVox = self.dataInput.nbVoxels
        sd = self.dataInput.simulData
        if sd is not None:
            self.trueValue = np.array([ssd['noise'].var(0) for ssd in sd])

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
        snrl = self.samplerEngine.get_variable('nrl')
        matPl =  self.samplerEngine.get_variable('drift').matPl
        
        varYbar = snrl.varYtilde - matPl

        for s in xrange(self.nbSubj):
            for j in xrange(self.nbVox):
                var_y_bar = varYbar[s,:,j]
                beta_g    = np.dot(var_y_bar.transpose(), var_y_bar)/2
                gammaSample = np.random.gamma((self.ny-1)/2, 1)

                self.currentValue[s,j] = np.divide(beta_g, gammaSample)
        



##################################################
# Drift Sampler and var drift sampler #
##################################################
class Drift_MultiSubj_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Gibbs sampler of the parameters modelling the low frequency drift in
    the fMRI time course, in the case of white noise.
    """
    # parameter labels definitions :
    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_USE_TRUE_VALUE = 'useTrueValue'

    # parameters definitions and default values :
    defaultParameters = {
        P_VAL_INI : None,
        P_SAMPLE_FLAG : True,
        P_USE_TRUE_VALUE : False,
        }
    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        an = ['subject','order','voxel']
        GibbsSamplerVariable.__init__(self, 'drift', valIni=val_ini,
                                      sampleFlag=do_sampling, axes_names=an,
                                      useTrueValue=use_true_value,
                                      value_label='PM LFD')

        self.final_signal = None
        self.true_value_signal = None

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nySubj = self.dataInput.nySubj
        self.dimDrift = self.dataInput.colP
        self.nbVox = self.dataInput.nbVoxels
        self.P = self.dataInput.lfdMat # : for all sessions
        self.nbSubj = self.dataInput.nbSubj
        self.varMBY = self.dataInput.varMBY

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            if 0: #theoretical value
                self.trueValue = np.array([ssd['drift_coeffs'] for ssd in sd])
            else: #empiracl value -> better if a multiplicative factor used to
                # generate drifts
                self.trueValue = np.array([np.dot(self.P.T, ssd['drift']) \
                                           for ssd in sd])

    def checkAndSetInitValue(self, variables):
        smplVarDrift = self.samplerEngine.get_variable('driftVar')
        smplVarDrift.checkAndSetInitValue(variables)

        if self.useTrueValue :
            if self.trueValue is not None:
                self.currentValue = self.trueValue
            else:
                raise Exception('Needed a true value for drift init but '\
                                    'None defined')

        if self.currentValue is None:
            self.currentValue = np.array([np.dot(self.P.T, ys) \
                                          for ys in self.varMBY])

        self.updateNorm()
        self.matPl = np.zeros((self.nbSubj, self.nySubj[0], self.nbVox))
        #suppose same nb of scans for all sujects

        for s in xrange(self.nbSubj):
            self.matPl[s] = np.dot(self.P, self.currentValue[s])

        self.ones_Q   = np.ones((self.dimDrift))


    def updateNorm(self):
        cv = self.currentValue
        self.norm = np.array([(cv[s] * cv[s]).sum() \
                              for s in xrange(self.nbSubj)])

    def sampleNextInternal(self, variables):
        eta =  self.samplerEngine.get_variable('driftVar').currentValue
        snrls = self.samplerEngine.get_variable('nrl')
        noise_vars = self.samplerEngine.get_variable('noise_var').currentValue

        for j in xrange(self.nbVox):
            for s in xrange(self.nbSubj):
                reps = noise_vars[s,j]
                pyhrf.verbose(5, 'eta :')
                pyhrf.verbose.printNdarray(5,eta[s])
                pyhrf.verbose(5, 'reps :' )
                pyhrf.verbose.printNdarray(5, reps)
                v_lj = reps*eta[s] / (reps + eta[s])
                mu_lj = v_lj/reps * np.dot(self.P.transpose(),
                                           snrls.varYtilde[s,:,j])

                self.currentValue[s,:,j] = randn(self.dimDrift) * \
                    v_lj**.5 + mu_lj

        self.updateNorm()

        for s in xrange(self.nbSubj):
            self.matPl[s] = np.dot(self.P, self.currentValue[s])

        pyhrf.verbose(5, 'drift params :')
        pyhrf.verbose.printNdarray(5, self.currentValue)


    def sampleNextAlt(self, variables):
        self.updateNorm()

    def get_final_value(self):
        if self.final_signal is None:
            self.final_signal = np.array([np.dot(self.P, self.finalValue[s]) \
                                         for s in xrange(self.nbSubj)])

        return self.final_signal

    def get_true_value(self):
        if self.true_value_signal is None:
            self.true_value_signal = np.array([np.dot(self.P,
                                                      self.trueValue[s]) \
                                              for s in xrange(self.nbSubj)])
        return self.true_value_signal

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        tol = 0.025
        tv = self.get_true_value()
        fv = self.get_final_value()
        err = np.zeros((self.nbSubj, self.nbVox), dtype=np.int8)
        for s in xrange(self.nbSubj):
            err[s] = (((fv[s] - tv[s])**2).sum(0) / (tv[s]**2).sum(0))**.5
        return ['session','voxel'], err < tol

    def getOutputs(self):

        #sn = self.dataInput.sNames
        outputs = GibbsSamplerVariable.getOutputs(self)
        drifts = self.get_final_value()
        an = ['subject', 'time','voxel']
        c = xndarray(drifts, axes_names=an)

        if self.trueValue is not None:
            tv = self.get_true_value()
            c_true = xndarray(tv, axes_names=an)
            c = stack_cuboids([c, c_true], axis='type', domain=['estim', 'true'],
                              axis_pos='last')

        outputs['drift_signal'] = c

        return outputs



class ETASampler_MultiSubj(xmlio.XmlInitable, GibbsSamplerVariable):
    """
        Gibbs sampler of the variance of the Inverse Gamma prior used to
        regularise the estimation of the low frequency drift embedded
        in the fMRI time course
    """

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        an = ['subject']
        GibbsSamplerVariable.__init__(self,'driftVar', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      axes_names=an, sampleFlag=do_sampling)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbSubj = self.dataInput.nbSubj
        self.nbVox = self.dataInput.nbVoxels
        self.varMBY = self.dataInput.varMBY
        self.P = self.dataInput.lfdMat
        
        if dataInput.simulData is not None:
            sd = dataInput.simulData

            self.P = self.get_variable('drift').P
            self.trueValue = np.array([np.dot(self.P.T, ssd['drift']).var() \
                                       for ssd in sd])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:
            self.currentValue = np.array([np.dot(self.P.T, ys).var() \
                                          for ys in self.varMBY])

    def sampleNextInternal(self, variables):
        #TODO : comment
        smpldrift = self.samplerEngine.get_variable('drift')
        for s in xrange(self.nbSubj):
            beta_d    = 0.5*smpldrift.norm[s]
            gammaSample = np.random.gamma( (smpldrift.dimDrift*self.nbVox - 1)/2, 1)
            
            self.currentValue[s] = np.divide(beta_d, gammaSample)



##################################################
# HRF Sampler #
##################################################

def sampleHRF_voxelwise_iid( stLambdaS, stLambdaY, varR, rh, nbColX, nbVox,
                             hgroup, only_hrf_subj, reg, nbsubj):
    varInvSigma_h = stLambdaS
    pyhrf.verbose(4,'stLambdaS:')
    pyhrf.verbose.printNdarray(4,stLambdaS)
    pyhrf.verbose(4,'varR:')
    pyhrf.verbose.printNdarray(4,varR)
    pyhrf.verbose(4,'rh over subjects:')
    pyhrf.verbose.printNdarray(4,rh)
    pyhrf.verbose(4,'varR/rh:')
    pyhrf.verbose.printNdarray(4,varR/rh)

    if reg:
        varInvSigma_h += nbVox*varR/rh
    else:
        varInvSigma_h += np.eye(varR.shape[0])/rh #if hrf subject NOT regularized
    
    if only_hrf_subj:   
    #do not ponderate hrfgroup because it is at zero
        mean_h = np.linalg.solve(varInvSigma_h, stLambdaY )
    else:
        mean_h = np.linalg.solve(varInvSigma_h, stLambdaY + nbsubj*hgroup/rh)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()
        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf


def sampleHRF_single_hrf_hack(stLambdaS, stLambdaY, varR, rh, nbColX, nbVox, hgroup):
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
    #varInvSigma_h += varR/rh

    varInvSigma_h += np.eye(varR.shape[0])/rh #if hrf subject NOT regularized
    mean_h = np.linalg.solve(varInvSigma_h, stLambdaY/nbVox + hgroup/rh)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()
        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf


def sampleHRF_single_hrf(stLambdaS, stLambdaY, varR, rh, nbColX, nbVox, hgroup, reg):

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
    #varInvSigma_h += varR/rh
    if reg:
        varInvSigma_h += varR/rh
    else:
        varInvSigma_h += np.eye(varR.shape[0])/rh #if hrf subject NOT regularized

    mean_h = np.linalg.solve(varInvSigma_h, stLambdaY+ hgroup/rh)

    if 0:
        choleskyInvSigma_h = np.linalg.cholesky(varInvSigma_h).transpose()

        hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX))
        hrf += mean_h
    else:
        hrf = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
    return hrf


######################
## HRF by subject #
######################
class HRF_Sampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    HRF sampler for multisession model
    """

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = ['duration', 'zero_constraint', 'do_sampling',
                            'output_hrf_pm',]



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
        'hack_covar_apost' : 'Divide the term coming from the likelihood by the nb '\
            'of voxels\n when computing the posterior covariance. The aim is '\
            ' to balance\n the contribution coming from the prior with that '\
            ' coming from the likelihood.\n Note: this hack is only taken into '\
            ' account when "singleHRf" is used for "prior_type"',
        'normalise' : 'If 1. : Normalise samples of Hrf, NRLs and Mixture Parameters when they are sampled.\n'\
                      'If 0. : Normalise posterior means of Hrf, NRLs and Mixture Parameters when they are sampled.\n'\
                      'else : Do not normalise.'

        }

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False,
                 duration=25., zero_constraint=True, normalise=1., derivOrder=2,
                 output_hrf_pm=True, hack_covar_apost=False, 
                 prior_type='voxelwiseIID', compute_ah_online=False, 
                 regularize_hrf=True, model_subjects_only=False, 
                 voxelwise_outputs=False):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)

        self.only_hrf_subj = model_subjects_only
        self.regularise = regularize_hrf
        self.compute_ah_online = compute_ah_online

        GibbsSamplerVariable.__init__(self, 'hrf', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['subject','time'],
                                      value_label='delta BOLD')

        self.duration = duration
        self.zc = zero_constraint
        self.normalise = normalise

        self.derivOrder = derivOrder
        self.varR = None
        self.outputHrf = output_hrf_pm
        self.covarHack = hack_covar_apost
        self.priorType = prior_type
        self.signErrorDetected = None
        self.voxelwise_outputs = voxelwise_outputs

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.nySubj = self.dataInput.nySubj
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt
        self.nbSubj = self.dataInput.nbSubj

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            self.trueValue = np.array([ssd['primary_hrf'] for ssd in sd])
        

        # Allocations :
        self.ajak_rb =  np.zeros((self.nbSubj, self.nbVox), dtype=float)

        self.varYaj = np.zeros((self.nbSubj, self.nySubj[0], self.nbVox), dtype=float)

        self.varXh = np.zeros((self.nbSubj, self.nySubj[0], self.nbConditions),
                              dtype=np.float64)

        self.norm = np.zeros(self.nbSubj)

    def updateNorm(self):
        for s in xrange(self.nbSubj):
            self.norm[s] = (self.currentValue[s]**2.0).sum()**0.5


    def checkAndSetInitValue(self, variables):
        smplRH = self.samplerEngine.get_variable('hrf_subj_var')
        smplRH.checkAndSetInitValue(variables)
        rh = smplRH.currentValue
        pyhrf.verbose(4, 'Hrf variance is :%s' %str(rh))
        pyhrf.verbose(4, 'hrfValIni is None -> setting it ...')

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
            hIni = getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

            hrfValIni = np.array([hIni]*self.nbSubj)
            pyhrf.verbose(5,'genCanoHRF -> shape h: %s' \
                          %str(hrfValIni.shape))
        
        if self.zc :
            pyhrf.verbose(4,'hrf zero constraint On' )
            hrfValIni = hrfValIni[:,1:(self.hrfLength-1)]

        pyhrf.verbose(4,'hrfValIni:' +\
                          str(hrfValIni.shape))
        pyhrf.verbose.printNdarray(6, hrfValIni)
        pyhrf.verbose(4, 'self.hrfLength:' \
                          +str(self.hrfLength))

        for s in xrange(self.nbSubj):
            normHRF = (hrfValIni[s]**2).sum()**(0.5)
            hrfValIni[s] /= normHRF
          

        self.currentValue = hrfValIni[:]

        if self.zc :
            self.axes_domains['time'] = np.arange(self.currentValue.shape[1]+2) \
                                       *self.eventdt
        else:
            self.axes_domains['time'] = np.arange(self.currentValue.shape[1]) \
                                       *self.eventdt


        pyhrf.verbose(4,'hrfValIni after ZC:' +\
                      str(self.currentValue.shape))
        pyhrf.verbose.printNdarray(6, self.currentValue )

        self.updateNorm()
        self.updateXh()

        self.track_sampled_quantity(self.varXh, 'varXh', axes_names=['subject',
                                                                     'time',
                                                                     'condition',
                                                                     ])

    def getCurrentVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_subj_var')
        rh = smplRH.currentValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh

    def getFinalVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_subj_var')
        rh = smplRH.finalValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh



    def samplingWarmUp(self, variables):
        if self.varR == None :
            smplRH = self.samplerEngine.get_variable('hrf_subj_var')
            rh = smplRH.currentValue
            (useless, self.varR) = genGaussianSmoothHRF(self.zc,
                                                        self.hrfLength,
                                                        self.eventdt, rh[0],
                                                        order=self.derivOrder)
            #self.varR = buildDiagGaussianMat(self.hrfLength-self.zc*2,4)
            # HACK
            #self.varR = ones_like(self.varR)


    def computeStDS_StDY_one_subject(self, rb, nrls, aa, subj):
        #case drift sampling

        varX = self.dataInput.varX[:,:,:]
        matXtX = self.dataInput.matXtX
        drift_sampler = self.get_variable('drift')
        matPl = drift_sampler.matPl[subj]
        y = self.dataInput.varMBY[subj] - matPl

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[subj]
            osf = int(sd['tr'] / sd['dt'])
            if not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                assert_almost_equal(self.dataInput.varMBY[subj], sd['bold'])
                assert_almost_equal(matPl, sd['drift'])
                assert_almost_equal(y, sd['stim_induced_signal'][::osf] + sd['noise'])

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )

        for j in xrange(self.nbConditions):
            self.varYaj = y/rb[subj]
            self.varYaj *= nrls[subj, j,:]
            varDeltaY +=  np.dot(varX[j,:,:].transpose(),self.varYaj.sum(1))

            for k in xrange(self.nbConditions):
                    np.divide(aa[subj,j,k,:], rb[subj], self.ajak_rb[subj, :])
                    varDeltaS += self.ajak_rb[subj].sum()*matXtX[j,k,:,:]

        return (varDeltaS, varDeltaY)


    def computeStDS_StDY(self, rb_allSubj, nrls_allSubj, aa_allSubj):

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )

        for s in xrange(self.nbSubj):
            ds, dy = self.computeStDS_StDY_one_subject(rb_allSubj[s,:],
                            nrls_allSubj[s,:,:], aa_allSubj[s,:,:,:], s)
            varDeltaS += ds
            varDeltaY += dy

        return (varDeltaS, varDeltaY)



    def sampleNextAlt(self, variables):
        self.reportCurrentVal()

    def sampleNextInternal(self, variables):
        #TODO : comment

        snrl = self.samplerEngine.get_variable('nrl')
        nrls = snrl.currentValue
        rb   = self.samplerEngine.get_variable('noise_var').currentValue       
        
        hg = self.samplerEngine.get_variable('hrf_group').currentValue
        D1 = np.eye(self.hrfLength, k=1) - np.eye(self.hrfLength, k=-1)
        alphas = self.samplerEngine.get_variable('alpha_hs').currentValue
        
        
        pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
        tSQSOptimIni = time.time()
        
        pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                      ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        rh = self.samplerEngine.get_variable('hrf_subj_var').currentValue

        h = self.currentValue
        for s in xrange(self.nbSubj):
            hgroup = hg + alphas[s]*np.dot(D1, hg)
            if self.priorType == 'voxelwiseIID':
            
                (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY_one_subject(rb, nrls,
                                                        snrl.aa, s)
                h[s] = sampleHRF_voxelwise_iid(self.varDeltaS, 
                                               self.varDeltaY,
                                               self.varR, rh[s], self.nbColX, 
                                               self.nbVox, hgroup, self.only_hrf_subj, 
                                               self.regularise, self.nbSubj)
            elif self.priorType == 'singleHRF':
               
                (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY_one_subject(rb, nrls,
                    snrl.aa, s)
                h[s] = sampleHRF_single_hrf(self.varDeltaS, self.varDeltaY,
                                        self.varR,
                                        rh[s], self.nbColX, self.nbVox, hgroup, self.regularise)

        
        self.updateNorm()

        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, '\
                          'Nrls and mixture parameters at each iteration ...')
            for s in xrange(self.nbSubj):
                f = self.norm[s]
                #HACK retrieve normalized hrf
                self.currentValue[s] = self.currentValue[s] / f #/(self.normalise+0.)
            if 0 and self.get_variable('nrls_by_subject').sampleFlag:
                self.get_variable('nrls_by_subject').currentValue *= f

                # Normalizing Mixture components
                smixt_params = self.samplerEngine.get_variable('mixt_params')
                if 0 and smixt_params.sampleFlag:
                    # Normalizing Mean's activation class
                    smixt_params.currentValue[smixt_params.I_MEAN_CA] *= f
                    # Normalizing Variance's activation class
                    smixt_params.currentValue[smixt_params.I_VAR_CI] *= f**2
                    # Normalizing Variance's in-activation class
                    smixt_params.currentValue[smixt_params.I_VAR_CA] *= f**2

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        self.updateXh()
        self.reportCurrentVal()


        # update ytilde for nrls
        nrlsmpl = self.samplerEngine.get_variable('nrl')
        for s in xrange(self.nbSubj):
            nrlsmpl.computeVarYTildeOpt(self.varXh[s], s)
        


    def reportCurrentVal(self):
        if pyhrf.verbose.verbosity >= 3:
            maxHRF = self.currentValue.max()
            tMaxHRF = np.where(self.currentValue==maxHRF)[0]*self.dt
            pyhrf.verbose(1, 'sampled HRF = %1.3f(%1.3f)'
                          %(self.currentValue.mean(),self.currentValue.std()))
            pyhrf.verbose(1,'sampled HRF max = ' + \
                              '(tMax=%1.3f,vMax=%1.3f)' %(tMaxHRF, maxHRF))

    def calcXh(self, hrfs):
        pyhrf.verbose(4,'CalcXh got stackX ' + \
                      str(self.dataInput.stackX.shape))
        all_stackXh = np.zeros((self.nbSubj, self.nySubj[0], self.nbConditions),
                               dtype=np.float64)

        for s in xrange(self.nbSubj):
            stackXh = np.dot(self.dataInput.stackX, hrfs[s])
            all_stackXh[s] = np.reshape(stackXh,
                                        (self.nbConditions,
                                         self.nySubj[0])).transpose()
        return all_stackXh

    def updateXh(self):
        self.varXh[:] = self.calcXh(self.currentValue)

    def setFinalValue(self):
        fv = self.mean
        if self.zc:
            # Append and prepend zeros
            self.finalValue = np.array([np.concatenate(([0], fv[s], [0]))\
                                        for s in xrange(self.nbSubj)])
            #self.error = np.concatenate(([0], self.error, [0]))
            if self.meanHistory is not None:
                nbIt = len(self.obsHistoryIts)

                z = np.zeros((nbIt, self.nbSubj,1))
                self.meanHistory =  np.concatenate((z,self.meanHistory,z),
                                                   axis=2)

            if self.smplHistory is not None:
                nbIt = len(self.smplHistoryIts)
                z = np.zeros((nbIt,self.nbSubj,1))
                self.smplHistory =  np.concatenate((z,self.smplHistory,z),
                                                    axis=2)
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
        if self.sampleFlag and np.allclose(self.normalise,0.):
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
                nrls = self.samplerEngine.get_variable('nrl').currentValue
                self.current_ah[:,:,j] = hrep.reshape(ncoeffs,self.nbVox) * \
                    nrls[j,:]

            self.cumul_ah += self.current_ah
            self.cumul2_ah += self.current_ah**2
            self.mean_ah = self.cumul_ah / self.nbItObservables
            self.var_ah = self.cumul2_ah / self.nbItObservables - self.mean_ah**2

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        tv = self.trueValue
        fv = self.finalValue
        crit_norm = np.array([((fv - tv)**2).sum() / (tv**2).sum()**.5 < 0.02 ])
        return ['crit_norm'], crit_norm


    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        # normalise hrf at the end of the sampling:
        for s in xrange(self.nbSubj):
            norm = (self.finalValue[s]**2.0).sum()**0.5
            self.finalValue[s] = self.finalValue[s] / norm

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if 0 and self.sampleFlag: #TODO adapt for multi subject
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue

            rb = self.samplerEngine.get_variable('noise_var').finalValue
            nrls = self.samplerEngine.get_variable('nrl')
            nrls = nrls.finalValue
            aa = np.zeros_like(nrls.aa)
            nrls.computeAA(nrls, aa)
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

        nrls = self.samplerEngine.get_variable('nrl').finalValue

        # hrfs = self.finalValue
        # ah = np.zeros((hrfs.shape[0], hrfs[0].shape,
        #                self.nbVox, self.nbConditions))
        # for s in xrange(self.nbSubj):
        #     for j in xrange(self.nbConditions):
        #         ah[s,:,:,j] = np.repeat(hrfs[s],self.nbVox).reshape(hrfs[s].shape[0],self.nbVox) * \
        #             nrls[j,:]
        ad = self.axes_domains.copy()
        ad['condition'] = self.dataInput.cNames
        # outputs['ah'] = xndarray(ah, axes_names=['subject', 'time','voxel','condition'],
        #                        axes_domains=ad,
        #                        value_label='Delta BOLD')

        if self.zc:
            dm = self.calcXh(self.finalValue[:, 1:-1])
        else:
            dm = self.calcXh(self.finalValue)

        xh_ad = {
            'time' : np.arange(self.dataInput.ny)*self.dataInput.tr,
            'condition':self.dataInput.cNames
            }

        outputs['Xh'] = xndarray(np.array(dm),
                               axes_names=['subject','time','condition'],
                               axes_domains=xh_ad)

        if getattr(self, 'compute_ah_online', False):
            z = np.zeros((1,)+self.mean_ah.shape[1:], dtype=np.float32)

            outputs['ah_online'] = xndarray(np.concatenate((z,self.mean_ah,z)),
                                          axes_names=['subject', 'time','voxel',
                                                      'condition'],
                                          axes_domains=ad,
                                          value_label='Delta BOLD')

            outputs['ah_online_var'] = xndarray(np.concatenate((z,self.var_ah,z)),
                                              axes_names=['subject','time','voxel',
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
        if 0 and pyhrf.__usemode__ == pyhrf.DEVEL:
            nrls = self.samplerEngine.get_variable('nrl').finalValue
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

        return outputs


##################################################
# Alpha for hrf group Sampler #
##################################################
class Alpha_hgroup_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = []

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'alpha_hs', valIni=val_ini,
                                        useTrueValue=use_true_value,
                                        sampleFlag=do_sampling,
                                        axes_names=['subject'],
                                        value_label='PM Noise Var')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nySubj = self.dataInput.nySubj
        self.nbSubj = self.dataInput.nbSubj
        self.nbVox = self.dataInput.nbVoxels
        self.hg = self.samplerEngine.get_variable('hrf_group')
        self.rhSubj = self.samplerEngine.get_variable('hrf_subj_var')
        self.varAlpha = self.samplerEngine.get_variable('alpha_var')
   
        sd = self.dataInput.simulData
        if sd is not None:
            self.trueValue = np.array([ssd['alpha'] for ssd in sd])

    def checkAndSetInitValue(self, variables):
        #self.hg.checkAndSetInitValue(variables)
        #self.varAlpha.checkAndSetInitValue(variables)
        #self.rhSubj.checkAndSetInitValue(variables)
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)
        
        if self.currentValue is None:
            self.currentValue = self.trueValue.astype(np.float64)

        
    def sampleNextInternal(self, variables):
        hs = self.samplerEngine.get_variable('hrf').currentValue
        D1 = np.eye(self.hg.currentValue.shape[0], k=1) - np.eye(self.hg.currentValue.shape[0], k=-1)
        #print D1.shape
        #print self.hg.currentValue.shape
        hgD = np.dot(D1, self.hg.currentValue)
        
        hgThgD = np.dot(self.hg.currentValue.transpose(), hgD)
        hgDThgD = np.dot(hgD.transpose(), hgD)
        
        varAlpha = self.varAlpha.currentValue[0]
        for s in xrange(self.nbSubj):
            hsThgD = np.dot(hs[s].transpose(), hgD)
            
            rhSubj = self.rhSubj.currentValue[s]
            rhSubj_varAlpha = rhSubj/varAlpha
            
            mean_Alpha = (hgThgD - hsThgD)/(hgDThgD + rhSubj_varAlpha)
            var_Alpha_apost  = rhSubj*varAlpha/(rhSubj + varAlpha*hgDThgD)
            #print s
            #print 'varalpha a prior:', varAlpha
            #print 'mean_Alpha a post:', mean_Alpha
            #print 'varalpha a post:', var_Alpha_apost
            self.currentValue[s] = randn(1) * var_Alpha_apost**.5 + mean_Alpha
            #print 'alpha in Alphaclass:', self.currentValue[s]
        

################################################
### Variance of alpha ##########################
################################################
class AlphaVar_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
        Gibbs sampler of the variance of the Inverse Gamma prior used to
        regularise the estimation of the low frequency drift embedded
        in the fMRI time course
    """

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        an = ['subject']
        GibbsSamplerVariable.__init__(self,'alpha_var', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      axes_names=an, sampleFlag=do_sampling)

    def linkToData(self, dataInput):
        self.alpha_hs = self.samplerEngine.get_variable('alpha_hs')
        self.dataInput = dataInput
        self.nbSubj = self.dataInput.nbSubj
        sd = self.dataInput.simulData
        if sd is not None:
            self.trueValue = np.array([np.array([ssd['alpha'] for ssd in sd]).var()])
            
            
    def checkAndSetInitValue(self, variables):
        self.alpha_hs.checkAndSetInitValue(variables)
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:
            self.currentValue = self.trueValue.astype(np.float64)
            
                                    
    def sampleNextInternal(self, variables):   
        #for s in xrange(self.nbSubj):
            #alpha_hs = self.alpha_hs.currentValue[s]
            #beta    = 0.5*alpha_hs**2
            #gammaSample = np.random.gamma(1, 1)
            
            #self.currentValue[s] = np.divide(beta, gammaSample)
            #print 'var alpha prior in VARAlph class:', self.currentValue[s]
        
        alpha_hs = self.alpha_hs.currentValue  
        #print alpha_hs
        
        beta    = np.dot(alpha_hs.transpose(), alpha_hs)/2
        #print beta
        
        gammaSample = np.random.gamma((self.nbSubj-1)/2, 1)
        
        self.currentValue[0] = np.divide(beta, gammaSample)
        #print self.currentValue
        

########################################
# Variance of HRF subject ##############
########################################
class HRFVarianceSubjectSampler(xmlio.XmlInitable, GibbsSamplerVariable) :

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        VAL_INI = 0.15
        PR_MEAN = 0.001
        PR_VAR = 10. #1000
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        VAL_INI = 0.05
        PR_MEAN = 0.001
        PR_VAR = 10.

    def __init__(self, val_ini=np.array([VAL_INI]), do_sampling=False, 
                 use_true_value=False, pr_mean=PR_MEAN, pr_var=PR_VAR):

        xmlio.XmlInitable.__init__(self)
        m = pr_mean
        v = pr_var
        #a=m**2/v +2 , b=m**3/v + m
        if 0:
            self.alpha0 = m**2/v
            self.beta0 = m**3/v + m
        else: #Jeffrey
            self.alpha0 = 2.
            self.beta0 = 0.

        GibbsSamplerVariable.__init__(self,'hrf_subj_var', valIni=val_ini,
                                      axes_names=['subject'], 
                                      sampleFlag=do_sampling, 
                                      useTrueValue=use_true_value)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbColX = self.dataInput.nbColX
        self.nbVox = self.dataInput.nbVoxels
        self.nbSubj = self.dataInput.nbSubj

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData       
            self.trueValue = np.array([ssd['var_subject_hrf'] for ssd in sd])
            
            
    def checkAndSetInitValue(self, variables):  
        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue=self.trueValue
            else:
                raise Exception('Needed a true value for %s init but '\
            'None defined' %self.name)
                
                
        if self.currentValue == None:
            if not self.sampleFlag and self.dataInput.simulData != None \
                   and self.dataInput.simulData.hrf.hrfParams.has_key('rh'):
                simulRh = self.dataInput.simulData.hrf.hrfParams['rh']
                self.currentValue = np.array([simulRh])
            else:
                self.currentValue = np.array([0.0001])

        if self.currentValue.size == 1:
            self.currentValue = np.tile(self.currentValue, self.nbSubj)


    def sampleNextInternal(self, variables):
        #TODO : comment
        shrf = self.samplerEngine.get_variable('hrf')
        hrf = shrf.currentValue
        varR = shrf.varR
        hgroup = self.samplerEngine.get_variable('hrf_group').currentValue
        
        for s in xrange(self.nbSubj):
            hrfT_R_hrf = np.dot(np.dot( (hrf[s]-hgroup), varR), (hrf[s]-hgroup) )
            pyhrf.verbose(5, 'hrfT_R^-1_hrf = ' + str(hrfT_R_hrf))
            if shrf.priorType == 'voxelwiseIID':
                alpha = 0.5*(self.nbColX*self.nbVox +  self.alpha0 - 1)
                beta = .5*hrfT_R_hrf*self.nbVox + self.beta0
                
            else:
                alpha = 0.5*(self.nbColX + self.alpha0 - 1)
                beta = .5*hrfT_R_hrf + self.beta0

    #         if shrf.trick:
    #             alpha = 0.5*(self.nbColX) * self.nbVox + self.alpha0 - 1
    #             beta = 1./(.5*self.nbVox*hrfT_R_hrf + self.beta0)
    #         else:
    #             alpha = 0.5*(self.nbColX) +  self.alpha0 - 1
    #             beta = 1./(.5*hrfT_R_hrf + self.beta0)

            #alpha = 0.5*self.nbColX + 1 + self.alpha0 #check !!
            #beta = 1/(.5*gJ*hrfT_R_hrf + self.beta0)
            pyhrf.verbose(5, 'varHRF apost ~1/Ga(%f,%f)'%(alpha, beta))
            if pyhrf.verbose.verbosity >= 5:
                ig_samples =  1.0/np.random.gamma(alpha, 1./beta,1000)
                pyhrf.verbose(5, 'empirical -> meanApost=%f, varApost=%f'
                            %(ig_samples.mean(),ig_samples.var()))
                pyhrf.verbose(5, 'theorical -> meanApost=%f, varApost=%f'
                            %(beta/(alpha-1), beta**2/((alpha-1)**2*(alpha-2))))

            #self.currentValue = 1.0/np.random.gamma(0.5*(self.nbColX+1)-1+self.alpha0,
            #                                     1/(.5*gJ*hrfT_R_hrf + self.beta0))
            self.currentValue[s] = np.array([1.0/np.random.gamma(alpha, 1./beta)])

            pyhrf.verbose(4,'varHRf[subj=%s] curval = '%s + \
                             str(self.currentValue[s]))
        


    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
        return outputs


######################
## HRF by subject #
######################
class HRF_Group_Sampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    HRF sampler for multisubjects model
    """

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = ['duration', 'zero_constraint', 'do_sampling',
                            'output_hrf_pm',] #P_USE_TRUE_VALUE,


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
        'hack_covar_apost' : 'Divide the term coming from the likelihood by the nb '\
            'of voxels\n when computing the posterior covariance. The aim is '\
            ' to balance\n the contribution coming from the prior with that '\
            ' coming from the likelihood.\n Note: this hack is only taken into '\
            ' account when "singleHRf" is used for "prior_type"',
        'normalise' : 'If 1. : Normalise samples of Hrf, NRLs and Mixture Parameters when they are sampled.\n'\
                      'If 0. : Normalise posterior means of Hrf, NRLs and Mixture Parameters when they are sampled.\n'\
                      'else : Do not normalise.',
        'model_subjects_only' : 'If 1: Put hrf group at zero and only estimate hrf by subjects.'\
                           'If 0: Perform group hemodynamic estimation, hrf group sampled',
                                        

        }

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False,
                 duration=25., zero_constraint=True, normalise=1., derivOrder=2,
                 output_hrf_pm=True, hack_covar_apost=False, 
                 prior_type='voxelwiseIID', compute_ah_online=False, 
                 regularize_hrf=True, model_subjects_only=False, 
                 voxelwise_outputs=False):

        xmlio.XmlInitable.__init__(self)

        self.regularise_subj_hrfs = regularize_hrf
        self.compute_ah_online = compute_ah_online

        
        if model_subjects_only:
            do_sampling = False
            use_true_value = False
            val_ini = np.zeros((43)) #case of 43 coeff for the hrf HACK
            if self.ValIni==None:
                raise Exception('Needed an initial value at zeros for group '
                                'hrf but None defined')

        GibbsSamplerVariable.__init__(self, 'hrf_group', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['time'],
                                      value_label='delta BOLD')

        self.duration = duration
        self.zc = zero_constraint
        self.normalise = normalise

        self.derivOrder = derivOrder
        self.varR = None
        self.outputHrf = output_hrf_pm
        self.covarHack = hack_covar_apost
        self.priorType = prior_type
        self.signErrorDetected = None
        self.voxelwise_outputs = voxelwise_outputs

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.nySubj = self.dataInput.nySubj
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt
        self.nbSess = self.dataInput.nbSessions
        self.nbSubj= self.dataInput.nbSubj

        self.D1 = np.eye(self.hrfLength, k=1) - np.eye(self.hrfLength, k=-1)
        if dataInput.simulData is not None : 
            sd = dataInput.simulData
            self.trueValue = dataInput.simulData[0]['hrf_group']
            if 0:
                hg = dataInput.simulData[0]['hrf_group']
                hrfs_subj = np.array([ssd['primary_hrf'] for ssd in sd])
                alphas_subj = np.array([ssd['alpha'] for ssd in sd])
                
                h=[]
                for s in xrange(self.nbSubj):
                    hrf = hrfs_subj[s] - alphas_subj[s]*np.dot(self.D1, hg)
                    h.append(hrf)
                self.trueValue = np.array(h).mean(0)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5

    def checkAndSetInitValue(self, variables):
        smplRH = self.samplerEngine.get_variable('var_hrf_group')
        smplRH.checkAndSetInitValue(variables)
        rh = smplRH.currentValue
        pyhrf.verbose(4, 'Hrf variance is :%1.3f' %rh)
        pyhrf.verbose(4, 'hrfValIni is None -> setting it ...')

        if self.parameters[self.P_ONLY_HRFS_SUBJ] is False:
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
                hIni = getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

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

            #normHRF = (sum(hrfValIni**2))**(0.5)
            #hrfValIni /= normHRF
            
        else:
            hrfValIni = self.ValIni

        self.currentValue = hrfValIni[:]
        #print self.currentValue
        
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


    def getCurrentVar(self):
        smplRH = self.samplerEngine.get_variable('var_hrf_group')
        rh = smplRH.currentValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh

    def getFinalVar(self):
        smplRH = self.samplerEngine.get_variable('var_hrf_group')
        rh = smplRH.finalValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh



    def samplingWarmUp(self, variables):
        if self.varR == None :
            smplRH = self.samplerEngine.get_variable('var_hrf_group')
            rh = smplRH.currentValue
            (useless, self.varR) = genGaussianSmoothHRF(self.zc,
                                                        self.hrfLength,
                                                        self.eventdt, rh, order=self.derivOrder)



    def sampleNextAlt(self, variables):
        self.reportCurrentVal()

    def sampleNextInternal(self, variables):
        #TODO : comment
        alphas = self.samplerEngine.get_variable('alpha_hs').currentValue
        
        rh = self.samplerEngine.get_variable('var_hrf_group').currentValue
        rhSubj = self.samplerEngine.get_variable('hrf_subj_var').currentValue
        hrf_subj = self.samplerEngine.get_variable('hrf').currentValue
        #self.currentValue = h
        hs_1alph_on_rh = np.zeros((hrf_subj.shape))
        s1alph_on_rh = np.zeros((self.nbSubj, hrf_subj.shape[1], hrf_subj.shape[1]))
        for s in xrange(self.nbSubj):
            hs_1alph_on_rh[s,:] = np.dot( hrf_subj[s], np.eye(self.hrfLength)+alphas[s]*self.D1.transpose() )/rhSubj[s]
            s1alph_on_rh[s] = (np.eye(self.hrfLength)+alphas[s]*self.D1.transpose())/rhSubj[s]
            
        if self.priorType == 'voxelwiseIID':
            if self.regularise_subj_hrfs:
                varInvSigma_h = self.varR*(s1alph_on_rh.sum(0) + self.nbSubj * np.eye(self.hrfLength)/rh )
            else:
                varInvSigma_h = self.varR*1/rh* self.nbSubj + np.eye(self.varR.shape[0])*s1alph_on_rh.sum(0)
                #if hrf subject NOT regularized
        else:
            if self.regularise_subj_hrfs:
                varInvSigma_h = self.varR*( s1alph_on_rh.sum(0) + np.eye(self.hrfLength)/rh )
            else:
                varInvSigma_h = self.varR*1/rh + np.eye(self.varR.shape[0])*s1alph_on_rh.sum(0)
                #if hrf subject NOT regularized
            
        # h_on_rh = np.zeros((hrf_subj.shape))
        # h_on_rhxalph = np.zeros((hrf_subj.shape))
        
        #mean_h = np.linalg.solve(varInvSigma_h, np.dot(self.varR,h_on_rh.sum(0)))
        #print 'hrhalp:', h_on_rhxalph
        #print h_on_rhxalph.sum(0)
        #print varInvSigma_h
        #print hs_1alph_on_rh.sum(0)
        mean_h = np.linalg.solve(varInvSigma_h, hs_1alph_on_rh.sum(0))
        self.currentValue = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma_h))
        
        self.updateNorm()

        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, '\
                          'Nrls and mixture parameters at each iteration ...')
            f = self.norm
            #HACK retrieve only HRF without normalizing
          
            #self.currentValue = self.currentValue
            self.currentValue = self.currentValue / f #/(self.normalise+0.)

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        self.reportCurrentVal()
        


    def reportCurrentVal(self):
        if pyhrf.verbose.verbosity >= 3:
            maxHRF = self.currentValue.max()
            tMaxHRF = np.where(self.currentValue==maxHRF)[0]*self.dt
            pyhrf.verbose(1, 'sampled HRF = %1.3f(%1.3f)'
                          %(self.currentValue.mean(),self.currentValue.std()))
            pyhrf.verbose(1,'sampled HRF max = ' + \
                              '(tMax=%1.3f,vMax=%1.3f)' %(tMaxHRF, maxHRF))


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
        
        norm = (self.finalValue**2.0).sum()**0.5
        self.finalValue = self.finalValue / norm

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


    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)
        if self.sampleFlag and np.allclose(self.normalise,0.):
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


    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        tv = self.trueValue
        fv = self.finalValue
        crit_norm = np.array([((fv - tv)**2).sum() / (tv**2).sum()**.5 < 0.02 ])
        return ['crit_norm'], crit_norm


    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        ## Correct hrf*nrl scale ambiguity :
        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()



    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)

        h = self.finalValue

        if hasattr(self, 'finalValue_sign_corr'):
            outputs['hrf_sign_corr'] = xndarray(self.finalValue_sign_corr,
                                              axes_names=self.axes_names,
                                              axes_domains=self.axes_domains,
                                              value_label='Delta BOLD')

        #print 'hrf - finalValue:'
        #print self.finalValue
        if 0  and pyhrf.__usemode__ == pyhrf.DEVEL:
            nrls = self.samplerEngine.get_variable('nrl').finalValue
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

        return outputs


###########################################
## Variance of Group HRF Sampler ##########
###########################################
class RHGroupSampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    #TODO : comment
    """

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        VAL_INI = 0.15
        PR_MEAN = 0.001
        PR_VAR = 10. #1000
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        VAL_INI = 0.05
        PR_MEAN = 0.001
        PR_VAR = 10.


    def __init__(self, val_ini=np.array([VAL_INI]), do_sampling=False, 
                 use_true_value=False, pr_mean=PR_MEAN, pr_var=PR_VAR):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        m = pr_mean
        v = pr_var
        #a=m**2/v +2 , b=m**3/v + m
        if 0:
            self.alpha0 = m**2/v
            self.beta0 = m**3/v + m
        else: #Jeffrey
            self.alpha0 = 2.
            self.beta0 = 0.

        GibbsSamplerVariable.__init__(self,'var_hrf_group', valIni=val_ini,
                                      sampleFlag=do_sampling, 
                                      useTrueValue=use_true_value)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbColX = self.dataInput.nbColX
        self.nbVox = self.dataInput.nbVoxels

    def checkAndSetInitValue(self, variables):
        if self.currentValue == None:
            if not self.sampleFlag and self.dataInput.simulData != None \
                   and self.dataInput.simulData.hrf.hrfParams.has_key('rh'):
                simulRh = self.dataInput.simulData.hrf.hrfParams['rh']
                self.currentValue = np.array([simulRh])
            else:
                self.currentValue = np.array([0.0001])

    def sampleNextInternal(self, variables):
        #TODO : comment
        shrf = self.samplerEngine.get_variable('hrf_group')
        hrf = shrf.currentValue
        varR = shrf.varR
        hrfT_R_hrf = np.dot(np.dot(hrf, varR), hrf)
        pyhrf.verbose(5, 'hrfT_R^-1_hrf = ' + str(hrfT_R_hrf))


        if shrf.priorType == 'voxelwiseIID':
            alpha = 0.5*(self.nbColX * self.nbSubj +  self.alpha0 - 1)
            beta = .5*hrfT_R_hrf * self.nbSubj + self.beta0
        else:
            alpha = 0.5*(self.nbColX + self.alpha0 - 1)
            beta = .5*hrfT_R_hrf + self.beta0

#         if shrf.trick:
#             alpha = 0.5*(self.nbColX) * self.nbVox + self.alpha0 - 1
#             beta = 1./(.5*self.nbVox*hrfT_R_hrf + self.beta0)
#         else:
#             alpha = 0.5*(self.nbColX) +  self.alpha0 - 1
#             beta = 1./(.5*hrfT_R_hrf + self.beta0)

        #alpha = 0.5*self.nbColX + 1 + self.alpha0 #check !!
        #beta = 1/(.5*gJ*hrfT_R_hrf + self.beta0)
        pyhrf.verbose(5, 'varHRF apost ~1/Ga(%f,%f)'%(alpha, beta))
        if pyhrf.verbose.verbosity >= 5:
            ig_samples =  1.0/np.random.gamma(alpha, 1./beta,1000)
            pyhrf.verbose(5, 'empirical -> meanApost=%f, varApost=%f'
                          %(ig_samples.mean(),ig_samples.var()))
            pyhrf.verbose(5, 'theorical -> meanApost=%f, varApost=%f'
                          %(beta/(alpha-1), beta**2/((alpha-1)**2*(alpha-2))))

        #self.currentValue = 1.0/np.random.gamma(0.5*(self.nbColX+1)-1+self.alpha0,
        #                                     1/(.5*gJ*hrfT_R_hrf + self.beta0))
        self.currentValue = np.array([1.0/np.random.gamma(alpha, 1./beta)])

        pyhrf.verbose(4,'varHRf curval = ' + str(self.currentValue))



    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
        return outputs



############################################
# Labels Sampler #
############################################
class LabelSampler(xmlio.XmlInitable, GibbsSamplerVariable):

    L_CI = 0
    L_CA = 1

    CLASSES = np.array([L_CI, L_CA],dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        an = ['subject', 'condition', 'voxel']
        GibbsSamplerVariable.__init__(self, 'label', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an)

        self.nbClasses = len(self.CLASSES)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.nbSubj   = self.dataInput.nbSubj

        self.cardClass = np.zeros((self.nbSubj, self.nbClasses, self.nbConditions), dtype=int)
        self.voxIdx = [[range(self.nbConditions) for c in xrange(self.nbClasses)] \
                        for s in xrange(self.nbSubj)]

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            sd = dataInput.simulData
            self.trueValue = np.array([ssd['labels'].astype(np.int32) \
                                       for ssd in sd])


    def checkAndSetInitValue(self, variables):
        pyhrf.verbose(1, 'LabelSampler.checkAndSetInitLabels ...')

        # Generate default labels if necessary :
        if self.useTrueValue:
            if self.trueValue is not None:
                pyhrf.verbose(1, 'Use true label values ...')
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('True labels have to be used but none defined.')

        if self.currentValue is None:

            self.currentValue = np.zeros((self.nbSubj, self.nbConditions, self.nbVoxels),
                                         dtype=np.int32)

            for s in xrange(self.nbSubj):
                for j in xrange(self.nbConditions):
                    self.currentValue[s,j,:] = 0#np.random.binomial(1, .3, self.nbVoxels)

        self.beta = np.zeros((self.nbSubj, self.nbConditions), dtype=np.float64) + .5

        self.currentValue = self.currentValue.astype(np.int32)
        self.countLabels()

    def countLabels(self):
        pyhrf.verbose(3, 'LabelSampler.countLabels ...')
        labs = self.currentValue
        for s in xrange(self.nbSubj):
            for j in xrange(self.nbConditions):
                for c in xrange(self.nbClasses):
                    self.voxIdx[s][c][j] = np.where(labs[s,j,:]==self.CLASSES[c])[0]
                    self.cardClass[s,c,j] = len(self.voxIdx[s][c][j])
                    pyhrf.verbose(5, 'Nb vox in C%d for cond %d, subj %d : %d' \
                                  %(c,j,s,self.cardClass[s,c,j]))

                if self.cardClass[s,:,j].sum() != self.nbVoxels:
                    raise Exception('cardClass[subj=%d,cond=%d]=%d != nbVox=%d' \
                                %(s,j,self.cardClass[s,:,j].sum(), self.nbVoxels))

    def samplingWarmUp(self, v):
        self.iteration = 0
        self.current_ext_field = np.zeros((self.nbSubj, self.nbClasses, self.nbConditions,
                                           self.nbVoxels), dtype=np.float64)

    def compute_ext_field(self):
        smixtp_sampler = self.samplerEngine.get_variable('mixt_params')

        v = smixtp_sampler.get_current_vars()

        mu = smixtp_sampler.get_current_means()

        nrls = self.samplerEngine.get_variable('nrl').currentValue

        for s in xrange(self.nbSubj):
            for k in xrange(self.nbClasses):
                for j in xrange(self.nbConditions):
                    e = .5 * (-np.log2(v[k,s,j]) - \
                            (nrls[s,j,:] - mu[k,s,j])**2 / v[k,s,j])
                    self.current_ext_field[s,k,j,:] = e

    def sampleNextInternal(self, v):

        neighbours = self.dataInput.neighboursIndexes
        beta = self.beta
        voxOrder = np.random.permutation(self.nbVoxels)
        self.compute_ext_field()
        
        for s in xrange(self.nbSubj):
            rnd = np.random.rand(*self.currentValue[s].shape).astype(np.float64)
            
            sample_potts(voxOrder.astype(np.int32), neighbours.astype(np.int32),
                self.current_ext_field[s].astype(np.float64), beta[s].astype(np.float64), 
                rnd, self.currentValue[s], self.iteration)

        self.countLabels()
        self.iteration += 1
        
        #if self.iteration==30:
            



############################################
#### Variance of subject's NRLS (not used now)##
############################################
class Variance_GaussianNRL_Multi_Subj(xmlio.XmlInitable, GibbsSamplerVariable):

    def __init__(self, val_ini=np.array([1.]), do_sampling=True, 
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'variance_nrls_by_subject', 
                                      valIni=val_ini, useTrueValue=use_true_value,
                                      sampleFlag=do_sampling)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.nbSubj = self.dataInput.nbSubj
        self.nbSessions = self.dataInput.nbSessions

        if dataInput.simulData is not None:
            #self.trueValue = np.array(np.array([dataInput.simulData[s]['nrls_session'] for s in xrange(self.nbSessions)]).var(0))
            sd = dataInput.simulData
            if 0: #theoretical true value
                self.trueValue = np.array([sd[0]['var_subject']])
            else: #empirical true value
                vv=[]
                for s in xrange(self.nbSubj):
                    nrl_mean = np.array([ssd[0]['nrls'] for ssd in sd]).mean(0)
                    v = np.array([ssd[0]['nrls'] - nrl_mean \
                              for ssd in sd]).var()
                    vv.append(v)
                self.trueValue = np.array(vv)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)


    def sampleNextInternal(self, variables):

        nrls = self.get_variable('nrl_by_session').currentValue
        nrlsBAR = self.get_variable('nrl_bar').currentValue

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


####################################################
## NRLs sampler #
####################################################
class NRLs_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):

    # other class attributes
    L_CI = 0
    L_CA = 1
    CLASSES = np.array([L_CI, L_CA],dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    FALSE_POS = 2
    FALSE_NEG = 3

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        an = ['subject', 'condition', 'voxel']
        #TODO: adapt to account for different nb of voxels across subjects
        GibbsSamplerVariable.__init__(self, 'nrl', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='nrls')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.nySubj = self.dataInput.nySubj
        self.nbSubj = self.dataInput.nbSubj
        self.nbClasses = len(self.CLASSES)

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            self.trueValue = np.array([ssd['nrls'] for ssd in sd])

        


    def checkAndSetInitValue(self, variables):
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None :
            self.currentValue = np.zeros((self.nbSubj, self.nbConditions,
                                          self.nbVoxels), dtype=np.float64)
            for s in xrange(self.nbSubj):
                for j in xrange(self.nbConditions):
                    #10% of the signal range
                    self.currentValue[s,j,:] = self.dataInput.varMBY[s].ptp(0) * .1

        # Precalculations and allocations :
        self.varYtilde = np.zeros((self.nbSubj, self.nySubj[0], self.nbVoxels), dtype=np.float64)
        self.sumaXh = np.zeros((self.nbSubj, self.nySubj[0], self.nbVoxels), dtype=np.float64)
        #self.varYbar = np.zeros((self.nbSubj, self.nySubj[0], self.nbVoxels), dtype=np.float64)
        self.aa = np.zeros((self.nbSubj, self.nbConditions,
                            self.nbConditions, self.nbVoxels),
                            dtype=np.float64)


    def samplingWarmUp(self, variables):
        """
        """
        varXh = self.samplerEngine.get_variable('hrf').varXh
        self.response_sampler = self.samplerEngine.get_variable('hrf')
        self.mixture_sampler = self.samplerEngine.get_variable('mixt_params')


        self.meanApost = np.zeros((self.nbSubj, self.nbConditions,
                                   self.nbVoxels), dtype=float)
        self.varApost = np.zeros((self.nbSubj, self.nbConditions,
                                  self.nbVoxels), dtype=float)

        self.labeled_vars = np.zeros((self.nbSubj, self.nbConditions,
                                      self.nbVoxels))
        self.labeled_means = np.zeros((self.nbSubj, self.nbConditions,
                                       self.nbVoxels))

        self.iteration = 0

        self.computeAA()
        for s in xrange(self.nbSubj):
            self.computeVarYTildeOpt(varXh[s], s)


    def sampleNextInternal(self, variables):

        labels = self.samplerEngine.get_variable('label').currentValue
        v_b = self.samplerEngine.get_variable('noise_var').currentValue

        varXh = self.samplerEngine.get_variable('hrf').varXh
        
        mixt_vars = self.mixture_sampler.get_current_vars()
        mixt_means = self.mixture_sampler.get_current_means()
        
        matPl = self.samplerEngine.get_variable('drift').matPl
        
        for s in xrange(self.nbSubj):
            ytilde = self.computeVarYTildeOpt(varXh[s], s) - matPl[s]
            
            gTg = np.diag(np.dot(varXh[s].transpose(),varXh[s]))
         
            for iclass in xrange(len(mixt_vars)):
                v = mixt_vars[iclass,s]
                m = mixt_means[iclass,s]
                for j in xrange(self.nbConditions):
                    class_mask = np.where(labels[s,j]==iclass)
                    self.labeled_vars[s,j,class_mask[0]] = v[j]
                    self.labeled_means[s,j,class_mask[0]] = m[j]

            for j in xrange(self.nbConditions):
                varXh_m = varXh[s,:,j]
                cv = self.currentValue[s]
                ytilde_m = ytilde[s] + (cv[np.newaxis,j,:] * \
                                        varXh_m[:,np.newaxis])
                v_q_j = self.labeled_vars[s,j]
                m_q_j = self.labeled_means[s,j]
                self.varApost = (v_b[s] * v_q_j) / (gTg[j] * v_q_j + v_b[s])
                self.meanApost = self.varApost * (np.dot(varXh_m.T, ytilde_m)/v_b[s] +\
                                                m_q_j / v_q_j )

                rnd = np.random.randn(self.nbVoxels)
                self.currentValue[s,j,:] = rnd * self.varApost**.5 + self.meanApost
                ytilde = self.computeVarYTildeOpt(varXh[s], s) - matPl[s]

                #b()

        self.computeAA()
        
        
    def computeVarYTildeOpt(self, varXh, s):
        #print 'shapes:', varXh.shape, self.currentValue[s].shape, self.dataInput.varMBY[s].shape, self.varYtilde[s].shape, self.sumaXh[s].shape
        computeYtilde(varXh.astype(np.float64),
                      self.currentValue[s], self.dataInput.varMBY[s],
                      self.varYtilde[s], self.sumaXh[s])

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde[s].shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde[s])
        matPl = self.get_variable('drift').matPl
        #self.varYbar[s] = self.varYtilde[s] - matPl[s]

        if self.dataInput.simulData is not None:
            sd = self.dataInput.simulData[s]
            smplHRF = self.samplerEngine.get_variable('hrf')
            smplDrift = self.samplerEngine.get_variable('drift')
            
            osf = int(sd['tr'] / sd['dt'])
            if not self.sampleFlag and  not smplHRF.sampleFlag and\
              self.useTrueValue and smplHRF.useTrueValue:
              assert_almost_equal(self.sumaXh[s], sd['stim_induced_signal'][::osf])
              assert_almost_equal(self.varYtilde[s], sd['bold'] - \
                                  sd['stim_induced_signal'][::osf])
              if not smplDrift.sampleFlag and \
                smplDrift.useTrueValue:
                varYbar = self.varYtilde[s] - matPl[s]
                assert_almost_equal(varYbar, sd['bold'] - \
                                    sd['stim_induced_signal'][::osf] - sd['drift'])
        return self.varYtilde


    def computeAA(self):
        for s in xrange(self.nbSubj):
            for j in xrange(self.nbConditions):
                for k in xrange(self.nbConditions):
                    np.multiply(self.currentValue[s,j,:],
                                self.currentValue[s,k,:],
                                self.aa[s,j,k,:])


##################################################
# Gaussian parameteres Sampler #
##################################################
class MixtureParamsSampler(xmlio.XmlInitable, GibbsSamplerVariable):

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    L_CA = LabelSampler.L_CA
    L_CI = LabelSampler.L_CI

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        an = ['subject', 'component','condition']
        ad = {'component' : self.PARAMS_NAMES}

        GibbsSamplerVariable.__init__(self, 'mixt_params', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      axes_domains=ad)

    def get_true_values_from_simulation_cdefs(self, cdefs):
        return np.array([[c.m_act for c in subj_cdefs] \
                          for subj_cdefs in cdefs]), \
               np.array([[c.v_act for c in subj_cdefs] \
                          for subj_cdefs in cdefs]), \
               np.array([[c.v_inact for c in subj_cdefs] \
                          for subj_cdefs in cdefs])

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.nbSubj = self.dataInput.nbSubj

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            cdefs = [ssd['condition_defs'] for ssd in sd]
            tmca, tvca, tvci = self.get_true_values_from_simulation_cdefs(cdefs)
            self.trueValue = np.zeros((self.nbSubj, self.NB_PARAMS, self.nbConditions),
                                      dtype=float)
            self.trueValue[:,self.I_MEAN_CA,:] = tmca
            self.trueValue[:,self.I_VAR_CA,:] = tvca
            self.trueValue[:,self.I_VAR_CI,:] = tvci

        self.nrlsCI = [range(self.nbConditions) for s in range(self.nbSubj)]
        self.nrlsCA = [range(self.nbConditions) for s in range(self.nbSubj)]

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            nc = self.nbConditions
            nbSubj = self.nbSubj
            self.currentValue = np.zeros((nbSubj,self.NB_PARAMS, nc), dtype=float)
            # self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30.
            # self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.
            # self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.
            y = self.dataInput.varMBY
            for s in xrange(nbSubj):
                self.currentValue[s,self.I_MEAN_CA,:] = y[s].ptp(0).mean() * .1
                self.currentValue[s,self.I_VAR_CA,:] = y[s].var(0).mean() * .1
                self.currentValue[s,self.I_VAR_CI,:] = y[s].var(0).mean() * .05

    def get_current_vars(self):
        """ return array of shape (class, subject, condition)
        """
        return np.array([self.currentValue[:,self.I_VAR_CI,:],
                         self.currentValue[:,self.I_VAR_CA,:]])

    def get_current_means(self):
        """ return array of shape (class, subject, condition)
        """
        return np.array([np.zeros((self.nbSubj,self.nbConditions)),
                         self.currentValue[:,self.I_MEAN_CA,:]])


    def computeWithJeffreyPriors(self, j, s, cardCIj, cardCAj):

        #print 'sample hyper parameters with improper Jeffrey\'s priors ...'
        if pyhrf.verbose.verbosity >= 3:
            print 'cond %d - card CI = %d' %(j,cardCIj)
            print 'cond %d - card CA = %d' %(j,cardCAj)
            print 'cond %d - cur mean CA = %f' %(j,self.currentValue[s, self.I_MEAN_CA,j])
            if cardCAj > 0:
                print 'cond %d - rl CA: %f(v%f)[%f,%f]' %(j,self.nrlsCA[s][j].mean(),
                                                          self.nrlsCA[s][j].var(),
                                                          self.nrlsCA[s][j].min(),
                                                          self.nrlsCA[s][j].max())
            if cardCIj > 0:
                print 'cond %d - rl CI: %f(v%f)[%f,%f]' %(j,self.nrlsCI[s][j].mean(),
                                                            self.nrlsCI[s][j].var(),
                                                            self.nrlsCI[s][j].min(),
                                                            self.nrlsCI[s][j].max())

        if cardCIj > 1:
            nu0j = np.dot(self.nrlsCI[s][j], self.nrlsCI[s][j])
            varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj + 1) - 1, 2. / nu0j)
            #varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj - 1), 2. / nu0j)
        else :
            varCIj = 1.0 / np.random.gamma(0.5, 0.2)

        #HACK
        #varCIj = .5

        if cardCAj > 1:
            rlC1Centered = self.nrlsCA[s][j] - self.currentValue[s,self.I_MEAN_CA,j]
            ##print 'rlC1Centered :', rlC1Centered
            nu1j = np.dot(rlC1Centered, rlC1Centered)
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
            eta1j = np.mean(self.nrlsCA[s][j])
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
                meanCAj = np.random.normal(self.nrlsCA[s][j], varCAj**0.5)

        if pyhrf.verbose.verbosity >= 3:
            print 'Sampled components - cond', j
            print 'var CI =', varCIj
            print 'mean CA =', meanCAj, 'var CA =', varCAj


            #b()

        return varCIj, meanCAj, varCAj

    def sampleNextInternal(self, variables):

        nrls_sampler = self.samplerEngine.get_variable('nrl')
        label_sampler = self.samplerEngine.get_variable('label')

        for s in xrange(self.nbSubj):
            cardCA = label_sampler.cardClass[s,self.L_CA,:]
            cardCI = label_sampler.cardClass[s,self.L_CI,:]
            for j in xrange(self.nbConditions):
            #for j in np.random.permutation(self.nbConditions):
                vICI = label_sampler.voxIdx[s][self.L_CI][j]
                vICA = label_sampler.voxIdx[s][self.L_CA][j]
                self.nrlsCI[s][j] = nrls_sampler.currentValue[s, j, vICI]
                self.nrlsCA[s][j] = nrls_sampler.currentValue[s, j, vICA]

                
                varCIj,meanCAj,varCAj = self.computeWithJeffreyPriors(j,s, cardCI[j],
                                                                      cardCA[j])

                self.currentValue[s, self.I_VAR_CI, j] = varCIj
                self.currentValue[s, self.I_MEAN_CA, j] = meanCAj #absolute(meanCAj)
                self.currentValue[s, self.I_VAR_CA, j] = varCAj

                pyhrf.verbose(5, 'Subject: %d' %s)
                pyhrf.verbose(5, '   varCI,%d=%f' \
                                %(j,self.currentValue[s,self.I_VAR_CI,j]))
                pyhrf.verbose(5, '   meanCA,%d=%f' \
                                %(j,self.currentValue[s,self.I_MEAN_CA,j]))
                pyhrf.verbose(5, '   varCA,%d = %f' \
                                %(j,self.currentValue[s,self.I_VAR_CA,j]))


# Not used !!-> version where labels are sampled within NRLs
class BiGaussMixtureParamsSampler(xmlio.XmlInitable, GibbsSamplerVariable):

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

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


    L_CA = NRLs_Sampler.L_CA
    L_CI = NRLs_Sampler.L_CI

    parametersComments = {
        'prior_type' : "Either 'proper' or 'Jeffrey'",
        'mean_activation_threshold' : "Threshold for the max activ mean above which the "\
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

    def __init__(self, val_ini=None, do_sampling=True, use_true_value=False,
                 prior_type='Jeffrey', var_ci_pr_alpha=2.04, var_ci_pr_beta=2.08,
                 var_ca_pr_alpha=2.01, var_ca_pr_beta=.5,
                 mean_ca_pr_mean=5., mean_ca_pr_var=20., 
                 mean_activation_threshold=4.):
        xmlio.XmlInitable.__init__(self)

        # get values for priors :
        self.varCIPrAlpha = var_ci_pr_alpha
        self.varCIPrBeta = var_ci_pr_beta
        self.varCAPrAlpha = var_ca_pr_alpha
        self.varCAPrBeta = var_ca_pr_beta

        self.meanCAPrMean = mean_ca_pr_mean
        self.meanCAPrVar = mean_ca_pr_var

        #self.ActMeanTrueValue = self.parameters[self.P_ACT_MEAN_TRUE_VALUE]
        #self.ActVarTrueValue = self.parameters[self.P_ACT_VAR_TRUE_VALUE]
        #self.InactVarTrueValue = self.parameters[self.P_INACT_VAR_TRUE_VALUE]

        an = ['component','condition']
        ad = {'component' : self.PARAMS_NAMES}
        GibbsSamplerVariable.__init__(self, 'mixt_params', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      sampleFlag=do_sampling, axes_names=an,
                                      axes_domains=ad)

        php = prior_type
        self.hyperPriorFlag = False if php=='Jeffrey' else True

        self.activ_thresh = mean_activation_threshold

    def linkToData(self, dataInput):
        self.dataInput =  dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.nbSubj = self.dataInput.nbSubj

        self.nrlCI = range(self.nbConditions)
        self.nrlCA = range(self.nbConditions)

        if self.dataInput.simulData is not None :
            if self.dataInput.simulData[0][0].has_key('condition_defs'):
                for s in xrange(self.nbSubj):
                    self.trueValue = np.zeros((self.nbSubj, self.NB_PARAMS, self.nbConditions),
                    dtype=float)
                    cdefs = self.dataInput.simulData[s][0]['condition_defs']
                    self.trueValue[s,self.I_MEAN_CA] = np.array([c.m_act for c in cdefs])
                    self.trueValue[s,self.I_VAR_CA] = np.array([c.v_act for c in cdefs])
                    self.trueValue[s,self.I_VAR_CI] = np.array([c.v_inact for c in cdefs])

                #if 0:
                    ##Theorethical true values:
                    #mean_act = np.array([c.m_act for c in cdefs])
                    #var_act = np.array([c.v_act for c in cdefs])
                    #var_inact = np.array([c.v_inact for c in cdefs])
                #else:
                    ##Empirical true values:
                    #nbc = self.nbConditions
                    #m_act = [np.where(sd[0]['labels'][j,:] == self.L_CA) \
                             #for j in xrange(nbc) ]
                    #m_inact = [np.where(sd[0]['labels'][j,:] == self.L_CI)\
                               #for j in xrange(nbc) ]
                    #all_nrls = np.array([ssd['nrls'] for ssd in sd])

                    #mean_act = np.array([all_nrls[:,j,m_act[j][0]].mean() \
                                         #for j in xrange(nbc)])
                    #var_act = np.array([all_nrls[:,j,m_act[j][0]].var() \
                                         #for j in xrange(nbc)])
                    #var_inact = np.array([all_nrls[:,j,m_inact[j][0]].var() \
                                         #for j in xrange(nbc)])
                    ##raise Exception()
                #self.trueValue[self.I_MEAN_CA] = mean_act
                #self.trueValue[self.I_VAR_CA] = var_act
                #self.trueValue[self.I_VAR_CI] = var_inact


        if self.dataInput.simulData is not None :

            for s in xrange(self.nbSubj):
                self.trueValue = np.zeros((self.nbSubj, self.NB_PARAMS, self.nbConditions),
                    dtype=float)
                simulation = self.dataInput.simulData[s][0]
                if simulation.has_key('condition_defs'):
                    cdefs = simulation['condition_defs']
                    self.trueValue[s,self.I_MEAN_CA] = np.array([c.m_act for c in cdefs])
                    self.trueValue[s,self.I_VAR_CA] = np.array([c.v_act for c in cdefs])
                    self.trueValue[s,self.I_VAR_CI] = np.array([c.v_inact for c in cdefs])



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

        nrlsSmpl = self.samplerEngine.get_variable('nrl')

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
        sHrf = self.samplerEngine.get_variable('hrf')

        if sHrf.sampleFlag and np.allclose(sHrf.normalise,0.):
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



##########################################################
# BOLD sampler input and BOLDSampler for multisessions
##########################################################

class BOLDSampler_MultiSujInput :

    """
    Class holding data needed by the sampler : BOLD time courses for each voxel,
    onsets and voxel topology.
    It also perform some precalculation such as the convolution matrix based on
    the onsests (L{stackX})
    ----
    Multi-subjects version
    (cf. merge_fmri_subjects in core.py)
    """

    def __init__(self, GroupData, dt, typeLFD, paramLFD, hrfZc, hrfDuration) :
        """
        Initialize a BOLDSamplerInput object.
        Groupdata: FmriGroupData object (list of FmriData data from all subjects),
        with attributes roiMask and roiIds (infos on multisubject parcellation)

        """
        pyhrf.verbose(3, 'BOLDSamplerInput init ...')

        data = GroupData.data_subjects
        self.roiId = GroupData.get_roi_id()
        self.nbSubj = GroupData.nbSubj
        self.nbSessions = len(data[0].sessionsScans) #suppose same nb of sessions for all subjects

        #names of subjects:
        self.subjNames = ['Subject%s' %str(i+1) for i in xrange(self.nbSubj)]
        #TODO change following line to adapt for all subjects
        self.nySubj = [len(subj) for subj in data[0].sessionsScans]
        self.ny = self.nySubj[0] #assume same nb of scans for all subjects
        self.varMBY = np.array([data[subj].bold[:] for subj in xrange(self.nbSubj)])

        self.varData = self.varMBY.var(1)
        self.tr = data[0].tr #suppose same paradigm for all subjects

        # if any ... would be used to compare results:
        # sd.simulation[0] -> assume only one session
        self.simulData = [sd.simulation[0] for sd in data]

        #as the mask is taken for all subjects:
        graph = data[0].get_graph()
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
        self.paradigm = data[0].paradigm #suppose same paradigm over all subjects
        self.cNames = data[0].paradigm.get_stimulus_names()
        self.nbConditions = len(self.cNames)
        onsets = data[0].paradigm.stimOnsets
        self.onsets = [onsets[cn] for cn in self.cNames]
        durations = data[0].paradigm.stimDurations
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
        self.matXQX = np.zeros( (self.nbConditions, self.nbConditions,
                                    self.nbColX, self.nbColX),dtype=float)
        self.matXQ = np.zeros((self.nbConditions, self.nbColX, self.nySubj[0]),
                                 dtype=float)

        for j in xrange(self.nbConditions):
            #print 'self.varX :', self.varX[j,:,:].transpose().shape
            #print 'self.delta :', self.delta.shape
            self.matXQ[j,:,:] = np.dot(self.varX[j,:,:].transpose(),
                                          self.delta )
            for k in xrange(self.nbConditions):
                self.matXQX[j,k,:,:] = np.dot( self.matXQ[j,:,:],
                                                  self.varX[k,:,:] )
        # Qy, yTQ & yTQy  :
        self.matQy = np.zeros((self.nySubj[0],self.nbVoxels), dtype=float)
        self.yTQ = np.zeros((self.nySubj[0],self.nbVoxels), dtype=float)
        self.yTQy = np.zeros(self.nbVoxels, dtype=float)

        for s in xrange(self.nbSubj):
            for i in xrange(self.nbVoxels):
                self.matQy[:,i] = np.dot(self.delta,self.varMBY[s,:,i])
                self.yTQ[:,i] = np.dot(self.varMBY[s,:,i],self.delta)
                self.yTQy[i] = np.dot(self.varMBY[s,:,i],self.matQy[:,i])

        self.matXtX = np.zeros( (self.nbConditions, self.nbConditions,
                                    self.nbColX, self.nbColX),dtype=float)
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                self.matXtX[j,k,:,:] = np.dot( self.varX[j,:,:].transpose(),
                self.varX[k,:,:] )

    #def makePrecalculations(self):
        ## XQX & XQ:
        #XQX=[]
        #XQ=[]
        #Qy=[]
        #yTQ=[]
        #yTQy=[]
        #XtX=[]
        #for isubj in xrange(self.nbSubj):
            #self.matXQX = np.zeros( (self.nbConditions, self.nbConditions,
                                        #self.nbColX, self.nbColX),dtype=float)
            #self.matXQ = np.zeros((self.nbConditions, self.nbColX, self.nySubj[isubj]),
                                    #dtype=float)

            #for j in xrange(self.nbConditions):
                #self.matXQ[j,:,:] = np.dot(self.varX[j,:,:].transpose(),
                                            #self.delta[isubj] )
                #for k in xrange(self.nbConditions):
                    #self.matXQX[j,k,:,:] = np.dot( self.matXQ[j,:,:],
                                                    #self.varX[k,:,:] )
            #XQX.append(self.matXQX)
            #XQ.append(self.matXQ)
            ## Qy, yTQ & yTQy  :
            #self.matQy = np.zeros((self.nySubj[isubj],self.nbVoxels), dtype=float)
            #self.yTQ = np.zeros((self.nySubj[isubj],self.nbVoxels), dtype=float)
            #self.yTQy = np.zeros(self.nbVoxels, dtype=float)

            #for i in xrange(self.nbVoxels):
                #self.matQy[:,i] = np.dot(self.delta[isubj],self.varMBY[isubj][:,i])
                #self.yTQ[:,i] = np.dot(self.varMBY[isubj][:,i],self.delta[isubj])
                #self.yTQy[i] = np.dot(self.varMBY[isubj][:,i],self.matQy[:,i])


            #self.matXtX = np.zeros( (self.nbConditions, self.nbConditions,
                        #self.nbColX, self.nbColX),dtype=float)
            #for j in xrange(self.nbConditions):
                #for k in xrange(self.nbConditions):
                    #self.matXtX[j,k,:,:] = np.dot( self.varX[isubj,j,:,:].transpose(),
                    #self.varX[isubj,k,:,:] )

            #Qy.append(self.matQy)
            #yTQ.append(self.yTQ)
            #yTQy.append(self.yTQy)
            #XtX.append(self.matXtX)

        #self.matXQX = np.array(XQX)
        #self.matXQ = np.array(XQ)
        #self.matQy = np.array(Qy)
        #self.yTQ   = np.array(yTQ)
        #self.yTQy  = np.array(yTQy)
        #self.matXtX = np.array(XtX)


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
        self.paradigmData = [rp[n][0] for n in self.cNames]

        pyhrf.verbose(3, 'building paradigm convol matrix ...')
        availIdx = [np.arange(0,n, dtype=int) for n in self.nySubj]
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

        pyhrf.verbose(3, 'LFD type :' + typeLFD)

        if typeLFD == 'polynomial':
            self.lfdMat = self.buildPolyMat( paramLFD , self.nySubj[0])
        elif typeLFD == 'cosine':
            self.lfdMat = self.buildCosMat( paramLFD , self.nySubj[0])
        elif typeLFD == 'None':
            self.lfdMat = np.zeros((self.nySubj[0],2))

        pyhrf.verbose(3, 'LFD Matrix :')
        pyhrf.verbose.printNdarray(3, self.lfdMat)

        self.varPPt = np.dot(self.lfdMat, self.lfdMat.transpose())
        if typeLFD is not 'None':
            self.colP = np.shape(self.lfdMat)[1]
        else:
            self.colP = 0

        self.delta = np.eye(self.nySubj[0], dtype=float) - self.varPPt
        self.varPtP = np.dot(self.lfdMat.transpose(), self.lfdMat)

        pyhrf.verbose(6, 'varPtP :')
        pyhrf.verbose.printNdarray(6, self.varPtP)
        if typeLFD != 'None':
            assert np.allclose(self.varPtP,
                                    np.eye(self.colP, dtype=float),
                                    rtol=1e-5 )


    def buildPolyMat( self, paramLFD , n ):

        regressors = self.tr*np.arange(0, n)
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
        lfdMat[:,0] = np.ones( ny, dtype= float)/np.sqrt(ny)
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

        lgt = (self.nySubj[0]+2)*osf #suppose same nb of scans for all subjects
        matH = np.zeros( (lgt, self.nbConditions), dtype=int)
        for j in xrange(self.nbConditions) :
            matH[:len(parData[j]), j] = parData[j][:]
            if pyhrf.verbose.verbosity >= 6:
                for a in xrange(matH.shape[0]):
                    print ' [',
                    for b in xrange(matH.shape[1]):
                        print matH[a,b],
                    print ']'


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
                                  for ai in availableDataIndex][0]
        pyhrf.verbose(2, 'Build pseudo teoplitz matrices')
        self.lenData = len(self.varOSAvailDataIdx)
        varX = np.zeros( (self.nbConditions, self.lenData, self.lgCI),
                        dtype=int )
        for j in xrange(self.nbConditions):
            pyhrf.verbose(6, ' cond : %d' %j)
            col = np.concatenate(([matH[0,j]],
                                np.zeros(self.hrfLength-1, dtype=int)))
            pyhrf.verbose(6, ' col :')
            if pyhrf.verbose.verbosity >= 6:
                print ' [',
                for b in xrange(col.shape[0]):
                    print col[b],
                print ']'


            matTmp = np.array(scipy.linalg.toeplitz( matH[:,j], col), dtype=int)
            pyhrf.verbose(6, ' matTmp :')
            if pyhrf.verbose.verbosity >= 6:
                for b in xrange(matTmp.shape[0]):
                    print ' [',
                    for a in xrange(matTmp.shape[1]):
                        print matTmp[b,a],
                    print ']'
            d0 = matTmp[self.varOSAvailDataIdx,:]
            d1 = d0[:,self.hrfColIndex]
            varX[j,:,:] = d1


            #vX.append(varX)
        #self.varX = hstack(vX)
        #self.varX = np.array(vX)
        self.varX = varX
        pyhrf.verbose(4, 'varX : ' + str(self.varX.shape))
        self.buildOtherMatX()

        self.nbColX = np.shape(self.varX)[2]

    def buildOtherMatX(self):
        varMBX=[]
        stackX=[]
        Id=[]

        self.varMBX = np.zeros( (self.nySubj[0], self.nbConditions*self.lgCI),
                            dtype=int)
        self.stackX = np.zeros( (self.nySubj[0]*self.nbConditions, self.lgCI),
                            dtype=int)

        for j in xrange(self.nbConditions):
            self.varMBX[:, j*self.lgCI+self.colIndex] = self.varX[j,:,:]

            self.stackX[self.nySubj[0]*j:self.nySubj[0]*(j+1), :] = self.varX[j,:,:]


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
        #self.cleanPrecalculations()


class BOLDGibbs_Multi_SubjSampler(xmlio.XmlInitable, GibbsSampler):

    inputClass = BOLDSampler_MultiSujInput

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        default_nb_its = 3
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        default_nb_its = 3000

    def __init__(self, nb_iterations=default_nb_its,
                 obs_hist_pace=-1., glob_obs_hist_pace=-1,
                 smpl_hist_pace=-1., burnin=.3,
                 callback=GSDefaultCallbackHandler(),
                 bold_response_levels_subj=NRLs_Sampler(),
                 labels=LabelSampler(), 
                 noise_var=NoiseVariance_Drift_MultiSubj_Sampler(),
                 hrf_subj=HRF_Sampler(), hrf_subj_var=HRFVarianceSubjectSampler(),
                 hrf_group=HRF_Group_Sampler(), hrf_group_var=RHGroupSampler(),
                 mixt_params=MixtureParamsSampler(),
                 drift=Drift_MultiSubj_Sampler(), 
                 drift_var=ETASampler_MultiSubj(),
                 alpha=Alpha_hgroup_Sampler(),
                 alpha_var=AlphaVar_Sampler(),
                 check_final_value=None):

        xmlio.XmlInitable.__init__(self)
        variables = [bold_response_levels_subj, labels, noise_var, hrf_subj, 
                     hrf_group, hrf_group_var, mixt_params, drift, drift_var,
                     alpha, alpha_var]

        nbIt = nb_iterations
        obsHistPace = obs_hist_pace
        globalObsHistPace = glob_obs_hist_pace
        smplHistPace = smpl_hist_pace
        nbSweeps = burnin

        if obsHistPace > 0. and obsHistPace < 1:
            obsHistPace = max(1,int(round(nbIt * obsHistPace)))

        if globalObsHistPace > 0. and globalObsHistPace < 1:
            globalObsHistPace = max(1,int(round(nbIt * globalObsHistPace)))

        if smplHistPace > 0. and smplHistPace < 1.:
            smplHistPace = max(1,int(round(nbIt * smplHistPace)))

        if nbSweeps > 0. and nbSweeps < 1.:
            nbSweeps = int(round(nbIt * nbSweeps))

        callbackObj = callback

        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps, callbackObj,
                              globalObsHistoryPace=globalObsHistPace,
                              check_ftval=check_final_value)



    def stop_criterion(self, it):
        return False

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
        del self.variables_old_val

    def saveGlobalObservables(self, it):
        #print 'saveGlobalObservables ...'
        GibbsSampler.saveGlobalObservables(self, it)
        if self.stop_threshold >= 0.:
            for vn, d in self.crit_diff.iteritems():
                self.conv_crit_diff[vn].append(d)

    # def buildSharedDataTree(self):

    #     self.sharedData = Pipeline()
    #     self.regVarsInPipeline()

    #     computeRules = []
    #     # Some more roots :
    #     computeRules.append({'label' : 'matXQX', 'ref' : self.matXQX})
    #     computeRules.append({'label' : 'varX', 'ref' : self.varX})
    #     computeRules.append({'label' : 'varMBY', 'ref' : self.varMBY})

    #     # Add shared quantities to update during sampling :
    #     computeRules.append({'label' : 'hXQXh', 'dep' : ['hrf','matXQX'],
    #                          'computeFunc' : computehXQXh})
    #     computeRules.append({'label' : 'matXh' , 'dep' : ['varX', 'hrf'],
    #                          'computeFunc' : computeXh})

    #     computeRules.append({'label' : 'sumj_aXh', 'dep' : ['matXh','nrls'],
    #                          'computeFunc' : computeSumjaXh})

    #     computeRules.append({'label' : 'yTilde', 'dep' : ['sumj_aXh','varMBY'],
    #                          'computeFunc' : computeYTilde})


    def finalizeSampling(self):
        return

        # class DummyVariable(): pass
        # msg = []
        # report = defaultdict(dict)
        # if self.check_ftval is not None:

        #     for v in self.variables+['labels']:

        #         if v == 'labels':
        #             v = DummyVariable()
        #             v.name = 'labels'
        #             v.finalValue = nrls.finalLabels
        #             v.trueValue = nrls.trueLabels

        #         if v.trueValue is None:
        #             print 'Warning: no true val for %s' %v.name
        #         elif not v.sampleFlag and v.useTrueValue:
        #             continue
        #         else:
        #             fv = v.finalValue
        #             tv = v.trueValue
        #             rtol = 0.1
        #             atol = 0.1

        #             if v.name == 'drift':
        #                 fv = np.array([np.dot(v.P[s], v.finalValue[s]) \
        #                                 for s in xrange(v.nbSubj)])
        #                 tv = np.array([np.dot(v.P[s], v.trueValue[s]) \
        #                                 for s in xrange(v.nbSubj)])
        #             elif v.name == 'hrf':
        #                 delta = (((fv - tv)**2).sum() / (tv**2).sum())**.5
        #                 report['hrf']['is_accurate'] = delta < 0.05
        #             elif v.name == 'labels':
        #                 delta = (fv != tv).sum()*1. / fv.shape[1]
        #                 report['labels']['is_accurate'] = delta < 0.05

        #             abs_error = np.abs(tv - fv)
        #             report[v.name]['abs_error'] = abs_error
        #             report[v.name]['rel_error'] = abs_error / np.maximum(tv,fv)

        #             # same criterion as np.allclose:
        #             nac = abs_error >= (atol + rtol * np.maximum(np.abs(tv),
        #                                                          np.abs(fv)))
        #             report[v.name]['not_close'] = nac

        #             if report[v.name].get('is_accurate') is None:
        #                 report[v.name]['is_accurate'] = not nac.any()

        #             if not report[v.name]['is_accurate']:
        #                 m = "Final value of %s is not close to " \
        #                 "true value.\n -> aerror: %s\n -> rerror: %s\n" \
        #                 " Final value:\n %s\n True value:\n %s\n" \
        #                 %(v.name, array_summary(report[v.name]['abs_error']),
        #                   array_summary(report[v.name]['rel_error']),
        #                   str(fv), str(tv))
        #                 msg.append(m)
        #                 if self.check_ftval == 'raise':
        #                     raise Exception(m)

        #     if self.check_ftval == 'print':
        #         print "\n".join(msg)

        # self.check_ftval_report = report

    
    def computeFit(self):
        #TODO: adapt for multisubject
        nbVox = self.dataInput.nbVoxels
        nbSubj = self.dataInput.nbSubj

        nbVals = self.dataInput.ny
        
        #hrfs for all subjects
        shrf = self.get_variable('hrf')
        hrf = shrf.finalValue
        if hrf is None:
            hrf = shrf.currentValue
        elif shrf.zc:
            hrf = hrf[1:-1]
        
        varXh = shrf.varXh
        
        #nrls for all subjects
        nrls = self.get_variable('nrl').finalValue
        if nrls is None:
            nrls = self.get_variable('nrl').currentValue

        stimIndSignal = np.zeros((nbSubj, nbVals, nbVox), dtype=np.float32)
        
        #drifts for all subjects
        drift = self.get_variable('drift').get_final_value()
        
        
        for s in xrange(nbSubj):
            stimIndSignal[s] = np.dot(varXh[s], nrls[s]) + drift[s]

        return stimIndSignal


    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)
        axes_domains = {'time' : np.arange(self.dataInput.ny)*self.dataInput.tr}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            # output of design matrix:
            dMat = np.zeros_like(self.dataInput.varX[0,:,:])
            for ic,vx in enumerate(self.dataInput.varX[0]):
                dMat += vx * (ic+1)

            outputs['matX_first_sess'] = xndarray(dMat, axes_names=['time','P'],
                                                axes_domains=axes_domains,
                                                value_label='value')
        


        if 0 and self.globalObsHistoryIts is not None:
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

        if 0 and hasattr(self, 'full_crit_diff_trajectory'):
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

        fit = self.computeFit()
        if self.dataInput.varMBY.ndim == 2:
            axes_names = ['time', 'voxel']
        else: #multisubjects
            axes_names = ['subject', 'time', 'voxel']
        bold = xndarray(self.dataInput.varMBY.astype(np.float32),
                        axes_names=axes_names,
                        axes_domains=axes_domains,
                        value_label='BOLD')

        if pyhrf.__usemode__ == pyhrf.DEVEL:
            cfit = xndarray(fit.astype(np.float32),
                        axes_names=axes_names,
                        axes_domains=axes_domains,
                        value_label='BOLD')

            outputs['bold_fit'] = stack_cuboids([bold,cfit], 'stype', ['bold', 'fit'])
            
        return outputs


    def computePMStimInducedSignal(self):

        nbVox = self.dataInput.nbVoxels
        nbSubj = self.dataInput.nbSubj

        nbVals = self.dataInput.ny
        shrf = self.get_variable('hrf')
        hrf = shrf.finalValue
        if shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.get_variable('nrl').finalValue

        self.stimIndSignal = np.zeros((nbSubj, nbVals, nbVox))
        meanBold = np.zeros((nbSubj, nbVox))

        for s in xrange(nbSubj):
            meanBold[s] = self.dataInput.varMBY[s].mean(axis=0)
            for i in xrange(nbVox):
                # Multiply by corresponding NRLs and sum over conditions :
                si = (vXh*nrl[s,:,i]).sum(1)
                # Adjust mean to original Bold (because drift is not explicit):
                self.stimIndSignal[s,:,i] = si-si.mean() +  meanBold[s,i]




##########################################
## for simulations #######################
##########################################
from pyhrf import Condition
from pyhrf.boldsynth.scenarios import create_alpha_for_hrfgroup, create_localizer_paradigm_avd, rasterize_paradigm, create_time_invariant_gaussian_nrls, create_gaussian_hrf_subject, create_stim_induced_signal, duplicate_hrf, duplicate_noise_var, create_gaussian_noise, create_drift_coeffs, create_polynomial_drift_from_coeffs, get_bold_shape, create_bold_from_stim_induced, simulation_save_vol_outputs, create_canonical_hrf, create_labels_vol, flatten_labels_vol

from pyhrf.tools import Pipeline

def simulate_single_subject(output_dir, cdefs, var_subject_hrf,
                            labels, labels_vol, v_noise, drift_coeff_var,
                            drift_amplitude, hrf_group_level, alpha_var, dt=0.6, dsf=4):

    simulation_steps = {
        'dt' : dt,
        'dsf' : dsf,
        'tr' : dt * dsf,
        'condition_defs' : cdefs,
        # Paradigm
        'paradigm' : create_localizer_paradigm_avd,
        'rastered_paradigm' : rasterize_paradigm,
        # Labels
        'labels_vol' : labels_vol,
        'labels' : labels,
        'nb_voxels': labels.shape[1],
        # Nrls
        'nrls' : create_time_invariant_gaussian_nrls,
        # HRF
        'alpha_var' : alpha_var,
        'alpha' : create_alpha_for_hrfgroup,
        'hrf_group' : hrf_group_level,
        'var_subject_hrf' : var_subject_hrf,
        'primary_hrf' : create_gaussian_hrf_subject,
        'hrf' : duplicate_hrf, #hrf for the subject
        # Stim induced
        'stim_induced_signal' : create_stim_induced_signal,
        # Noise
        'v_gnoise' : v_noise,
        'v_noise' : duplicate_noise_var,
        'noise' : create_gaussian_noise,
        # Drift
        'drift_order' : 4,
        'drift_coeff_var' : drift_coeff_var,
        'drift_amplitude' : drift_amplitude,
        'drift_mean' : 0.,
        'drift_coeffs': create_drift_coeffs,
        'drift' : create_polynomial_drift_from_coeffs,
        # Bold
        'bold_shape' : get_bold_shape,
        'bold' : create_bold_from_stim_induced,
        }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    if output_dir is not None:
        simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))

    # Retrieve all results
    simulation = simu_graph.get_values()

    # Save outputs o simulation: nii volumes:
    if output_dir is not None:
        simulation_save_vol_outputs(simulation, output_dir)

    return simulation


Vhrfs = [.0006, .0004, .00009, .00002, .000015,
.00002, .0001,  .00003, .000075, .000032   ] #by subject           
Vnoise = [0.2, 0.5, 0.4, 0.8, 0.6,
2.1, 2.5, 3.1, 2.75, 7.3] #by subject
VAlph = 0.6

def simulate_subjects(output_dir, snr_scenario='high_snr', spatial_size='tiny', \
                      hrf_group=create_canonical_hrf(dt=0.6), nbSubj=10, vars_hrfs=Vhrfs,\
                      vars_noise=Vnoise, alpha_var=VAlph):
    '''
    Simulate data for multiple subjects (5 subjects by default)
    '''

    drift_coeff_var = 1.
    drift_amplitude = 10.

    #if spatial_size == 'tiny':
        #lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    #else:
        #lmap1, lmap2, lmap3 = 'pacman', 'cat2', 'house_sun'

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    else:
        lmap1, lmap2, lmap3 = 'pacman', 'ghost', 'house_sun'

    if snr_scenario == 'low_snr': #low snr
        vars_hrfs = [.06, .04, 0.09, 0.2, 0.015]
        vars_noise = [.1, .3, .4, 15, 6]
        conditions = [
            Condition(name='audio', m_act=3., v_act=.3, v_inact=.3,
                        label_map=lmap1),
            Condition(name='video', m_act=2.5, v_act=.3, v_inact=.3,
                        label_map=lmap2),
            Condition(name='damier', m_act=2, v_act=.3, v_inact=.3,
                        label_map=lmap3),
            ]
    else: #high snr
        
        conditions = [
            Condition(name='audio', m_act=13., v_act=.2, v_inact=.1,
                        label_map=lmap1),
            #Condition(name='video', m_act=11.5, v_act=.2, v_inact=.1,
                        #label_map=lmap2),
            #Condition(name='damier', m_act=10, v_act=.2, v_inact=.1,
                        #label_map=lmap3),
            ]

    nb_subjects = nbSubj
    #if nbSubj!=len(vars_hrfs):
        #raise Exception('Error: not the same number of subjects and number of variance given!')

    # Common variable across subjects:
    labels_vol = create_labels_vol(conditions)
    labels     = flatten_labels_vol(labels_vol)

    simu_subjects = []
    simus = []
    for isubj in xrange(nb_subjects):
        if output_dir is not None:
            out_dir = op.join(output_dir, 'subject_%d' %isubj)
            if not op.exists(out_dir): os.makedirs(out_dir)
        else:
            out_dir = None
        s = simulate_single_subject(out_dir, conditions, vars_hrfs[isubj],
                                    labels, labels_vol, vars_noise[isubj],
                                    drift_coeff_var,
                                    drift_amplitude, hrf_group, alpha_var, dt=0.6, dsf=4)
        simus.append(s)
        simu_subjects.append(FmriData.from_simulation_dict(s))
    simu_subjects = FmriGroupData(simu_subjects)


    return simu_subjects


