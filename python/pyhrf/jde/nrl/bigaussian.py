# -*- coding: utf-8 -*-



# -*- coding: utf-8 -*-
import os
#from numpy import *
import numpy as np
import numpy.matlib
from numpy.matlib import repmat
from scipy.integrate import trapz
import copy as copyModule
#from libc.stdio import printf

from pyhrf import xmlio
from pyhrf.tools import resampleToGrid, get_2Dtable_string
from pyhrf.ndarray import xndarray
from pyhrf.jde.intensivecalc import calcCorrEnergies, sampleSmmNrl, sampleSmmNrl2,computeYtilde
from pyhrf.jde.intensivecalc import sampleSmmNrlWithRelVar, sampleSmmNrl2WithRelVar, computeYtildeWithRelVar
from pyhrf.jde.samplerbase import *
from pyhrf.jde.beta import *
from pyhrf.boldsynth.spatialconfig import hashMask

from base import *

from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.tools.aexpression import ArithmeticExpressionNameError, \
    ArithmeticExpressionSyntaxError

from pyhrf.stats import compute_roc_labels_scikit, threshold_labels, \
    mark_wrong_labels, compute_roc_labels, cpt_ppm_a_mcmc

from scipy.integrate import quad

from pyhrf.tools.io import read_volume

#class NrlChecker:
    #def __init__(self):
        #self.called = False
    #def __call__(self):

class NRLSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    Class handling the Gibbs sampling of Neural Response Levels with a prior
    bi-gaussian mixture model. It handles independent and spatial versions.
    Refs : Vincent 2010 IEEE TMI, Makni 2008 Neuroimage, Sockel 2009 ICASSP
    #TODO : comment attributes
    """

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = ['contrasts']

    parametersComments = {
        'contrasts' : 'Define contrasts as arithmetic expressions.\n'\
            'Condition names used in expressions must be consistent with ' \
            'those specified in session data above',
        }

    # other class attributes
    L_CI = 0
    L_CA = 1
    CLASSES = np.array([L_CI, L_CA],dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    FALSE_POS = 2
    FALSE_NEG = 3

    def __init__(self, do_sampling=True, val_ini=None,
                 contrasts={'dummy_contrast_example' :
                            '0.5 * audio - 0.5 * video'},
                 do_label_sampling=True, use_true_nrls=False,
                 use_true_labels=False, labels_ini=None,
                 ppm_proba_threshold=0.05, ppm_value_threshold=0,
                 ppm_value_multi_threshold=np.arange(0.,4.1,0.1),
                 mean_activation_threshold=4, rescale_results=False,
                 wip_variance_computation=False):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        self.sampleLabelsFlag = do_label_sampling
        sampleFlag = do_sampling
        valIni = val_ini
        useTrueVal = use_true_nrls
        self.useTrueLabels = use_true_labels
        an = ['condition', 'voxel']
        GibbsSamplerVariable.__init__(self,'nrl', valIni=valIni,
                                      sampleFlag=sampleFlag,
                                      useTrueValue=useTrueVal,
                                      axes_names=an,
                                      value_label='PM NRL')

        self.labels = labels_ini
        self.contrasts_expr = contrasts
        self.contrasts_expr.pop('dummy_contrast_example', None)
        self.computeContrastsFlag = ( len(self.contrasts_expr) > 0 )
        self.activ_thresh = mean_activation_threshold
        #print 'computeContrastsFlag :', self.computeContrastsFlag
        #self.parseContrasts(contrasts)

        self.nbClasses = len(self.CLASSES)
        pyhrf.verbose(6, 'NRLSampler - classes: %s (%d)' \
                          %(str(self.CLASS_NAMES), self.nbClasses))

        self.labelsMeanHistory = None
        self.labelsSmplHistory = None

        self.wip_variance_computation = wip_variance_computation
        self.ppm_proba_thresh = ppm_proba_threshold
        self.ppm_value_thresh = ppm_value_threshold
        self.ppm_value_multi_thresh = ppm_value_multi_threshold
        self.rescale_results = rescale_results

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
            elif isinstance(dataInput.simulData, list):
                sd = dataInput.simulData[0]
                if isinstance(sd, dict):
                    self.trueValue = sd['nrls'].astype(np.float64)
                    self.trueLabels = sd['labels'].astype(np.int32)
                else:
                    self.trueValue = sd.nrls.data.astype(np.float64)
                    self.trueLabels = sd.nrls.labels
            else:
                self.trueValue = dataInput.simulData.nrls.data.astype(np.float64)
                self.trueLabels = dataInput.simulData.nrls.labels

            self.trueLabels = self.trueLabels[:self.nbConditions,:].astype(np.int32)
            self.trueValue = self.trueValue[:self.nbConditions,:].astype(np.float64)
        else:
            self.trueLabels = None
        #print self.trueLables

    def init_contrasts(self):

        pyhrf.verbose(3, 'Init of contrasts ...')
        pyhrf.verbose(3, 'self.dataInput.cNames: %s'
                      %str(self.dataInput.cNames))
        cnames = self.dataInput.cNames
        #print 'cnames', cnames
        self.nrls_conds = dict([(str(cond), self.currentValue[icond,:]) \
                               for icond,cond in enumerate(cnames)] )



        if isinstance(self.contrasts_expr, str):
            cexpr = dict([('contrast%d'%i,s) \
                         for i,s in enumerate(self.contrasts_expr.split(";")) \
                         if len(s) > 0])
        else:
            cexpr = self.contrasts_expr

        cexpr.pop('dummy_contrast_example', None)
        #print 'testeeeuu', cexpr.items
        self.conds_in_contrasts = dict([(str(cn), \
                                             filter(lambda x: x in cv, cnames)) \
                                            for cn,cv in cexpr.items()] )
        #print 'balh:', self.conds_in_contrasts
        #print 'self.contrasts_expr', self.contrasts_expr
        self.contrasts_calc = dict([ (str(cn),AExpr(str(e), **self.nrls_conds)) \
                                         for cn,e in cexpr.iteritems() ])
        #print 'self.contrasts_calc', self.contrasts_calc.values()

        #for cn,e in cexpr.iteritems():
            #print 'AExpr(str(e), **nrls_conds):', (AExpr(str(e), **nrls_conds))

        for cn,cc in self.contrasts_calc.iteritems():

            try:
                cc.check()
            except ArithmeticExpressionNameError, err:
                msg = 'Error in definition of contrast "%s":' %cn
                pyhrf.verbose(1,msg)
                pyhrf.verbose(1,'Unknown conditions: ' + ', '.join(err.args[2]))
                pyhrf.verbose(1,'Expression was: "%s"' %err.args[1])
                raise err
            except ArithmeticExpressionSyntaxError, err:
                msg = 'Syntax error in definition of contrast %s:' %cn
                pyhrf.verbose(1,msg)
                pyhrf.verbose(1, 'expression was: ' + err.args[1])
                raise err

        self.cumulContrast = dict([ (cn,np.zeros(self.nbVox)) \
                                        for cn,e in cexpr.iteritems() ])
        #print 'cexpr.iteritems: ', cexpr.iteritems()
        self.cumul2Contrast = dict([ (cn,np.zeros(self.nbVox)) \
                                         for cn,e in cexpr.iteritems() ])

    def checkAndSetInitValue(self, variables):
        self.checkAndSetInitLabels(variables)
        self.checkAndSetInitNRL(variables)

    def checkAndSetInitLabels(self, variables):
        pyhrf.verbose(1, 'NRLSampler.checkAndSetInitLabels ...')
        # Generate default labels if necessary :
        #print 'blab', self.useTrueLabels
        if self.useTrueLabels:
            if self.trueLabels is not None:
                pyhrf.verbose(3, 'Use true label values ...')
                #TODO : take only common conditions
                self.labels = self.trueLabels.copy()
                #print 'True labels : ', self.labels.shape
                #HACK
                #tmpl = np.zeros_like(self.labels)
                #tmpl[:,:self.nbVox/30] = 1
                #tmpl = np.array([np.random.permutation(t) for t in tmpl])
                #self.labels = np.bitwise_or(self.labels, tmpl)

                # print '~~~~~~~'
                # print 'tmpl', tmpl
                # print np.unique(self.labels)
                # print 'labels:',self.labels
            else:
                raise Exception('True labels have to be used but none defined.')


        if self.labels is None : # if no initial labels specified
            pyhrf.verbose(1, 'Labels are not initialized -> random init')
            if 0:
                self.labels = np.zeros((self.nbConditions, self.nbVox),
                                    dtype=np.int32)
                nbVoxInClass = np.zeros(self.nbClasses, dtype=int)   \
                    + self.nbVox/self.nbClasses
                nbVoxInClass[0] = self.nbVox-nbVoxInClass[1:].sum()
                #nbVoxInClass0 = self.nbVox/2
                # Uniform dispatching :
                for j in xrange(self.nbConditions) :
                    l = []
                    for c in xrange(self.nbClasses) :
                        l += [self.CLASSES[c]] * nbVoxInClass[c]
                    self.labels[j,:] = np.random.permutation(l)
            else:
                # sometimes it's better to put all voxels in the activating class
                # -> mixture components are less likely to degenerate
                #self.labels = np.ones((self.nbConditions, self.nbVox), dtype=np.int32)

                sh = (self.nbConditions, self.nbVox)
                self.labels = np.random.randint(0, 2, np.prod(sh)).reshape(sh).astype(np.int32)

                #self.labels = np.zeros((self.nbConditions, self.nbVox), dtype=np.int32)
                #print 'here'
                #nlabs = self.nbConditions * self.nbVox
                if 0:
                    self.labels = np.random.binomial(1,0.9,nlabs).reshape(self.nbConditions, self.nbVox).astype(np.int32)

        #print 'self.labels to see', self.labels, self.labels.shape
        pyhrf.verbose(5, 'init labels :')
        pyhrf.verbose.printNdarray(6, self.labels)
        self.countLabels(self.labels, self.voxIdx, self.cardClass)

    def checkAndSetInitNRL(self, variables):
        pyhrf.verbose(3, 'NRLSampler.checkAndSetInitNRLs ...')

        if self.currentValue is None :
            if 0 and self.useTrueValue:
                if self.trueValue is None:
                    raise Exception('Needed a true value for nrls init but '\
                                        'None defined')
                else:
                    self.currentValue = self.trueValue.astype(np.float64)

            elif 1 and self.useTrueValue:
                if self.TrueNrlsFilename is not None:
                    pyhrf.verbose(3, 'Use true Nrls values ...')
                    self.currentValue = np.zeros((self.nbConditions, self.nbVox),
                                                dtype=np.float64)
                    TrueNrls = read_volume(self.TrueNrlsFilename)
                    for cond in np.arange(TrueNrls[0].shape[3]):
                        count=0
                        for i in np.arange(TrueNrls[0].shape[0]):
                            for j in np.arange(TrueNrls[0].shape[1]):
                                for k in np.arange(TrueNrls[0].shape[1]):
                                    self.currentValue[cond,count] = TrueNrls[0][i,j,k,cond]
                                    count += 1
                    self.trueValue = self.currentValue.copy()
                else:
                    raise Exception('Needed a true value for nrls init but '\
                                        'None defined')

            else:
                #nrlsIni = np.zeros((self.nbConditions, self.nbVox), dtype=np.float64)
                ## Init Nrls according to classes definitions :
                #smplMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
                ## ensure that mixture parameters are correctly set
                #smplMixtP.checkAndSetInitValue(variables)
                #var = smplMixtP.getCurrentVars()
                #means = smplMixtP.getCurrentMeans()

                #for j in xrange(self.nbConditions):
                    #for c in xrange(self.nbClasses):
                        #iv = self.voxIdx[c][j]
                        #nrlsIni[j,iv] = np.random.randn(self.cardClass[c,j]) \
                            #* var[c,j]**0.5 + means[c,j]
                #self.currentValue = nrlsIni
                ##HACK (?)
                #self.currentValue = np.zeros((self.nbConditions, self.nbVox),
                                          #dtype=np.float64) + 20
                #self.currentValue = (np.random.rand(self.nbConditions, self.nbVox).astype(np.float64) - .5 ) * 10
                #self.currentValue = (np.random.rand(self.nbConditions, self.nbVox).astype(np.float64) - .5 ) * 10
                #print 'nrlsIni : ', self.currentValue

                #Initialise nrls using initial labels
                if 1:
                    self.currentValue = np.zeros((self.nbConditions, self.nbVox),dtype=np.float64)
                    #Mixt_par = self.samplerEngine.get_variable('mixt_params')
                    #Mean_CA = Mixt_par.currentValue[Mixt_par.I_MEAN_CA,:]
                    #Var_CA = Mixt_par.currentValue[Mixt_par.I_VAR_CA,:]
                    #Var_CI = Mixt_par.currentValue[Mixt_par.I_VAR_CI,:]
                    Mean_CA = 30. * np.ones(self.nbConditions)
                    Var_CA = 1. * np.ones(self.nbConditions)
                    Var_CI = 1. * np.ones(self.nbConditions)
                    #Mean_CA = 2. * np.ones(self.nbConditions)
                    #Var_CA = 0.5 * np.ones(self.nbConditions)
                    #Var_CI = 0.5 * np.ones(self.nbConditions)
                    for m in xrange(self.nbConditions):
                        Ac_pos = np.where(self.labels[m])
                        Nrls = np.random.randn((self.nbVox))*Var_CI[m]**0.5 + 0
                        Nrls[Ac_pos[0]] = np.random.randn((Ac_pos[0]).size)*Var_CA[m]**0.5 + Mean_CA[m]
                        self.currentValue[m] = Nrls.astype(np.float64)


    def countLabels(self, labels, voxIdx, cardClass):
        pyhrf.verbose(3, 'NRLSampler.countLabels ...')
        # print 'countLabels .......'
        # print labels.shape
        # print len(voxIdx), len(voxIdx[0])
        # print len(cardClass)
        for j in xrange(self.nbConditions):
            for c in xrange(self.nbClasses):
                if len(labels.shape) == 2:
                    labs_c = labels
                else:
                    labs_c = labs_c
                try:
                    voxIdx[c][j] = np.where(labels[j,:]==self.CLASSES[c])[0]
                except Exception, e:
                    print e
                    print '~~~~~~~~~~~~~~~~~~'
                    print j,c
                    print labels.shape
                    print labels[j,:].shape
                    print len(voxIdx), len(voxIdx[0])
                    print self.CLASSES[c]
                    print np.where(labels[j,:]==self.CLASSES[c])[0]
                    raise e

                cardClass[c,j] = len(voxIdx[c][j])
                pyhrf.verbose(5, 'Nb vox in C%d for cond %d : %d' \
                                  %(c,j,cardClass[c,j]))
            #assert self.cardClass[:,j].sum() == self.nbVox

    def initObservables(self):
        pyhrf.verbose(3, 'NRLSampler.initObservables ...')
        GibbsSamplerVariable.initObservables(self)
        self.meanLabels = None
        shape_utile = np.zeros((3))
        self.cumulLabels = np.zeros((self.nbClasses,)+np.shape(self.currentValue),
                                 dtype=np.float32) #insert the final nb of iterations

        self.count_above_thresh = np.zeros_like(self.currentValue).astype(int)
        self.count_above_Multi_thresh = np.zeros((len(self.ppm_value_multi_thresh),self.currentValue.shape[0],self.currentValue.shape[1])).astype(int)
        #print 'self.count_above_Multi_thresh shape =',self.count_above_Multi_thresh.shape
        #print 'self.currentValue', self.currentValue, self.currentValue.shape

        #self.cumulLabels_all_iterations = np.zeros((self.nbClasses,)+np.shape(self.currentValue,)+shape_utile.shape,
                                 #dtype=float)

        #print 'Concerning shape of labels', self.cumulLabels_all_iterations.shape

        #print 'Concerning labels', self.cumulLabels.shape, self.nbClasses, self.currentValue.shape, self.nbConditions, self.nbVox
        #print self.nbItObservables
        #print 'Pour voir sur iterations', self.nbIterations, 'test'

        if self.computeContrastsFlag:
            self.init_contrasts()

        if 0 and self.computeContrastsFlag:
            #for c1 in self.cumulContrast.iterkeys():
                #for c2 in self.cumulContrast[c1].iterkeys():
                    #self.cumulContrast[c1][c2] = np.zeros(self.nbVox, dtype=float) #sum of Linear Combination of nrls
                    #self.cumul2Contrast[c1][c2] = np.zeros(self.nbVox, dtype=float) #sum square of Linear Combination of nrls

            self.cumulContrast_Lc_Rc  = np.zeros(self.nbVox, dtype=float) #Lc-Rc
            self.cumul2Contrast_Lc_Rc = np.zeros(self.nbVox, dtype=float)
            self.cumulContrast_V_A    = np.zeros(self.nbVox, dtype=float) #V-A
            self.cumul2Contrast_V_A   = np.zeros(self.nbVox, dtype=float)
            self.cumulContrast_C_S    = np.zeros(self.nbVox, dtype=float) #C-S
            self.cumul2Contrast_C_S   = np.zeros(self.nbVox, dtype=float)
            self.cumulContrast_C_S_A  = np.zeros(self.nbVox, dtype=float) #C-S_A
            self.cumul2Contrast_C_S_A = np.zeros(self.nbVox, dtype=float)


            ################# CHANGE HERE DEPENDING ON THE NUMBER OF ITERATIONS !!
            if self.wip_variance_computation:
                self.saveNRL = np.zeros((2000,self.nbConditions, self.nbVox),
                                        dtype=float)

            sh = (self.nbClasses,) + np.shape(self.currentValue)
            self.diff_nrl_mean_masked = np.zeros(sh,dtype=float)
            self.diff_nrl_mean_non_masked = np.zeros(np.shape(self.currentValue),
                                                  dtype=float)



            self.Covar_masked = np.zeros((4,self.nbVox), dtype=float)
            self.Covar_non_masked = np.zeros((self.nbVox), dtype=float)

            self.varCon_2cond_corr_masked   = np.zeros((4,self.nbVox),
                                                       dtype=float)
            self.varCon_2cond_corr_apost    = np.zeros((4,self.nbVox),
                                                       dtype=float)


        #return self.cumulLabels_all_iterations
        self.cumul_mean_apost = np.zeros_like(self.meanClassApost).astype(np.float32)
        self.cumul_var_apost = np.zeros_like(self.varClassApost).astype(np.float32)


        #self.sum_nrls_carr_tot_cond           = np.zeros((self.nbClasses,)+np.shape(self.currentValue),dtype=float)
        self.sum_nrls_carr_both_classes_cond   = np.zeros((self.nbClasses,)+np.shape(self.currentValue),dtype=float)
        #self.sum_nrls_carr_class_inactiv_cond = np.zeros((self.nbClasses,)+np.shape(self.currentValue),dtype=float)
        #self.sum_nrls_tot_cond           = np.zeros(np.shape(self.currentValue),dtype=float)
        self.sum_nrls_both_classes_cond   = np.zeros((self.nbClasses,)+np.shape(self.currentValue),dtype=float)
        #self.sum_nrls_class_inactiv_cond = np.zeros(np.shape(self.currentValue),dtype=float)


        self.finalVariances = np.zeros(np.shape(self.currentValue),
                                    dtype=np.float32)
        self.final_mean_var_a_post = np.zeros(np.shape(self.currentValue),
                                           dtype=np.float32)
        self.varCon_2cond_indep_masked  = np.zeros((4,self.nbVox),
                                                   dtype=np.float32)
        self.varCon_2cond_indep_apost   = np.zeros((4,self.nbVox),
                                                   dtype=np.float32)

    def updateObsersables(self):
        pyhrf.verbose(4, 'NRLSampler.updateObsersables ...')
        GibbsSamplerVariable.updateObsersables(self)
        sHrf = self.samplerEngine.get_variable('hrf')
        sScale = self.samplerEngine.get_variable('scale')

        if sHrf.sampleFlag and np.allclose(sHrf.normalise,0.) and \
                not sScale.sampleFlag and self.sampleFlag:
            pyhrf.verbose(6, 'Normalizing Posterior mean of NRLs at each iteration ...')
            #print '%%%% scaling NRL PME %%% - hnorm = ', sHrf.norm
            # Undo previous mean calculation:
            self.cumul -= self.currentValue
            self.cumul3 -=  (self.currentValue - self.mean)**2
            #self.cumul2 -= self.currentValue**2

            # Use scaled quantities instead:
            self.cumul += self.currentValue * sHrf.norm
            #self.cumul2 += (self.currentValue * sHrf.norm)**2
            self.mean = self.cumul / self.nbItObservables

            self.cumul3 +=  (self.currentValue * sHrf.norm - self.mean)**2
            self.error = self.cumul3 / self.nbItObservables

            #self.error = self.cumul2 / self.nbItObservables - \
                #self.mean**2

        for c in xrange(self.nbClasses):
            self.cumulLabels[c,:,:] += (self.labels==c)
            #self.cumulLabels_all_iterations[c,:,:,
            #print 'labels at iteration'
            #print self.cumulLabels[c]
            #print self.labels
        self.meanLabels = self.cumulLabels / self.nbItObservables

        self.count_above_thresh += self.currentValue > self.ppm_value_thresh
        self.freq_above_thresh = self.count_above_thresh / self.nbItObservables

        #print 'self.currentValue.shape =',self.currentValue.shape
        #print 'self.currentValue[2,:].sum() =',self.currentValue[2,:].sum()

        for i in xrange(len(self.ppm_value_multi_thresh)):
            self.count_above_Multi_thresh[i] += abs(self.currentValue) >= self.ppm_value_multi_thresh[i]
        self.freq_above_Multi_thresh = self.count_above_Multi_thresh / self.nbItObservables

        #print 'Mean labels at each iteration', self.meanLabels, self.meanLabels.shape
        #print 'Cumul labels at each iteration', self.cumulLabels[c], self.cumulLabels.shape, self.cumulLabels[c].shape
        #print 'Concerning classes', self.nbClasses
        #print 'cumulLabels_all_iterations', self.cumulLabels_all_iterations.shape, self.cumulLabels_all_iterations[:,:,:,1].shape

        #print 'Current value:', self.currentValue, self.currentValue.shape
        #print 'labels at iteration:'
        #print self.labels, self.labels.shape

        if self.wip_variance_computation:
            #To save value at each iteration
            for cond in xrange(self.nbConditions):
                #self.sum_nrls_tot_cond[cond] += (self.currentValue[cond,:])
                #self.sum_nrls_carr_tot_cond[cond] += (self.currentValue[cond,:])**2

                for c in xrange(self.nbClasses):

                    #self.sum_nrls_class_activ_cond[cond] += (self.currentValue[cond,:])*self.labels[cond,:]
                    #self.sum_nrls_class_inactiv_cond[cond] += self.sum_nrls_tot_cond[cond] - self.sum_nrls_class_activ_cond[cond]

                    #self.sum_nrls_carr_tot_cond[cond] += (self.currentValue[cond,:])**2
                    #self.sum_nrls_carr_class_activ_cond[cond] += ((self.currentValue[cond,:])**2)*self.labels[cond,:]
                    #self.sum_nrls_carr_class_inactiv_cond[cond] += self.sum_nrls_carr_tot_cond[cond] - self.sum_nrls_carr_class_activ_cond[cond]

                    #print 'TO TEST: ',  self.sum_nrls_both_classes_cond[c,cond,:], self.currentValue[cond,:], self.labels[cond,:]

                    if c==1:
                        self.sum_nrls_both_classes_cond[c,cond,:] += \
                          (self.currentValue[cond,:])* self.labels[cond,:]
                        self.sum_nrls_carr_both_classes_cond[c,cond] += \
                          ((self.currentValue[cond,:])**2)*self.labels[cond,:]
                    elif c==0:
                        self.sum_nrls_both_classes_cond[c,cond,:] += \
                          (self.currentValue[cond,:])*(1-self.labels[cond,:])
                        self.sum_nrls_carr_both_classes_cond[c,cond] += \
                          ((self.currentValue[cond,:])**2) * \
                          (1-self.labels[cond,:])

                    #print 'TO TEST after attribution: ',  self.sum_nrls_both_classes_cond[c,cond,:], self.currentValue[cond,:], self.labels[cond,:]

            if 0:
                print 'Verification about self.sum_nrls_both_classes_cond on voxels 0 to 3:',
                print self.sum_nrls_both_classes_cond[:,:,310:316]

            #print 'prod tot carr:'
            #print self.sum_nrls_carr_tot_cond, self.sum_nrls_carr_tot_cond.shape
            #print 'prod activ carr:'
            #print self.sum_nrls_carr_class_activ_cond, self.sum_nrls_carr_class_activ_cond.shape
            #print 'prod inactiv carr:'
            #print self.sum_nrls_carr_class_inactiv_cond, self.sum_nrls_carr_class_inactiv_cond.shape

            #print 'prod tot:'
            #print self.sum_nrls_tot_cond, self.sum_nrls_tot_cond.shape
            #print 'prod activ:'
            #print self.sum_nrls_class_activ_cond, self.sum_nrls_class_activ_cond.shape
            #print 'prod inactiv:'
            #print self.sum_nrls_class_inactiv_cond, self.sum_nrls_class_inactiv_cond.shape
            if pyhrf.verbose.verbosity > 4:
                print 'Non zeros positions for cumulLabels:'
                print np.where(self.cumulLabels[1,1,:]>0)
                print 'Non zeros for self.sum_nrls_carr_class_activ_cond and self.sum_nrls_class_activ_cond :'
                #print np.where(self.sum_nrls_carr_class_activ_cond>0)
                #print np.where(self.sum_nrls_class_activ_cond>0)
                print 'Non zeros for labels :',
                print np.where(self.labels[1,:]>0)

        #print 'Test:', self.sum_nrls_carr_class_activ_cond.shape, self.sum_nrls_carr_class_activ_cond[298], self.sum_nrls_class_activ_cond[298]

        pyhrf.verbose(4,'nb of iterations: %d' %self.nbItObservables)

        pyhrf.verbose(4,'computeContrastsFlag: %s',
                      str(self.computeContrastsFlag))


        #print 'self.computeContrastsFlag:', self.computeContrastsFlag

        #To save value at each iteration
        if self.wip_variance_computation:
            self.saveNRL[self.nbItObservables-1,:,:] = self.currentValue
        #print 'self.saveNRL.shape',  self.saveNRL.shape

        #print 'self.saveNRL', self.saveNRL[self.nbItObservables-1,0,:], self.saveNRL[self.nbItObservables-1,0,:].shape
        #print 'current value to compare:', self.currentValue[0,:]

        if self.computeContrastsFlag:

            cv = self.currentValue
            if 0:
                print "Current value used for con:", cv.shape
                print "nrl values:", cv[0,310:316], cv[1,310:316],
                print "con values?:", 2*cv[1,310:316]- cv[0,310:316]

            #print 'blob', self.cumulContrast
            #for cname, cumul  in self.cumulContrast.iteritems():
                #print 'self.cumulContrast[cname] debut:', cname, self.cumulContrast[cname]
                #print 'self.cumul2Contrast[cname] debut:', cname, self.cumul2Contrast[cname]
                ##print 'cname:', cname
                #contrast = self.contrasts_calc[cname].evaluate()
                #print 'self.contrasts_calc[cname].evaluate(): ', self.contrasts_calc[cname].evaluate()
                #self.cumulContrast[cname] += contrast
                #self.cumul2Contrast[cname] += contrast**2

            #print  'self.cumulContrast.iteritems() :', self.cumulContrast.values()
            for cname, cumul  in self.cumulContrast.iteritems():
                if 0:
                    print 'cname:', cname
                    print 'nbIt', self.nbItObservables
                #print 'self.cumulContrast[cname]:', cname, self.cumulContrast[cname]
                contrast = self.contrasts_calc[cname].evaluate()
                #print cname, 'self.contrasts_calc[cname].evaluate(): ', self.contrasts_calc[cname].evaluate()
                self.cumulContrast[cname] += contrast
                #print cname, 'cumul:', cumul

            for cname, cumul2  in self.cumul2Contrast.iteritems():
                if 0:
                    print 'cname:', cname
                    print 'nbIt', self.nbItObservables
                #print 'self.cumul2Contrast[cname]:', cname, self.cumul2Contrast[cname]
                contrast2 = (self.contrasts_calc[cname].evaluate())**2
                #print cname, 'self.contrasts_calc[cname].evaluate() carre: ', (self.contrasts_calc[cname].evaluate())**2
                self.cumul2Contrast[cname] += contrast2
                #print cname, 'cumul2:', cumul2


                #Contrast = sum of two conditions --> variances estimate study
                #contrast_V-A = cv




            #B/II 1/
            if self.wip_variance_computation:
                for cond in xrange(self.nbConditions):
                    self.diff_nrl_mean_non_masked[cond,:] += self.saveNRL[self.nbItObservables, cond,:] - self.saveNRL[self.nbItObservables, cond,:].mean()
                    self.NRL_activ_masked   = (self.saveNRL[self.nbItObservables, cond,:])*self.labels[cond,:]
                    self.NRL_inactiv_masked = (self.saveNRL[self.nbItObservables, cond,:])*(1-self.labels[cond,:])
                    for c in xrange(self.nbClasses):
                        if c==1:
                            self.diff_nrl_mean_masked[c,cond,:] += ( self.NRL_activ_masked - self.NRL_activ_masked.mean() )

                        elif c==0:
                            self.diff_nrl_mean_masked[c,cond,:] += ( self.NRL_inactiv_masked - self.NRL_inactiv_masked.mean() )

                    # ic1 = self.dataInput.cNames.index(c1)
                    # for c2 in self.cumulContrast[c1].iterkeys():
                    #     ic2 = self.dataInput.cNames.index(c2)
                    #     diff = self.currentValue[ic1,:]-self.currentValue[ic2,:]
                    #     self.cumulContrast[c1][c2] += diff
                    #     self.cumul2Contrast[c1][c2] += diff**2

            #To get infos on object dataInput : print its fields (line1) or print its classe (line2)
            #print dir(self.dataInput) #.cNames
            #print self.dataInput.__class__
            #print self.dataInput.cNames
            #print cumulContrast_Lc_Rc
            if 0 and ('calculaudio' in self.dataInput.cNames):
                ic1  = self.dataInput.cNames.index('calculaudio')
                ic2  = self.dataInput.cNames.index('calculvideo')
                ic3  = self.dataInput.cNames.index('clicDaudio')
                ic4  = self.dataInput.cNames.index('clicDvideo')
                ic5  = self.dataInput.cNames.index('clicGaudio')
                ic6  = self.dataInput.cNames.index('clicGvideo')
                ic7  = self.dataInput.cNames.index('damier_H')
                ic8  = self.dataInput.cNames.index('damier_V')
                ic9  = self.dataInput.cNames.index('phraseaudio')
                ic10 = self.dataInput.cNames.index('phrasevideo')
                cv = self.currentValue

                #print 'CURRENT VALUE:', self.currentValue, self.currentValue.shape
                contrast_Lc_Rc = cv[ic5,:] + cv[ic6,:] - \
                cv[ic3,:] - cv[ic4,:] #Lc-Rc
                self.cumulContrast_Lc_Rc += contrast_Lc_Rc
                self.cumul2Contrast_Lc_Rc += contrast_Lc_Rc**2

                contrast_V_A = cv[ic2,:] + cv[ic4,:] + cv[ic6,:] +\
                    cv[ic10,:] - cv[ic1,:] - cv[ic3,:] - cv[ic5,:] - cv[ic9,:] #V-A
                self.cumulContrast_V_A += contrast_V_A
                self.cumul2Contrast_V_A += contrast_V_A**2

                contrast_C_S = cv[ic1,:] + cv[ic2,:] - cv[ic9,:] - cv[ic10,:] #C-S
                self.cumulContrast_C_S += contrast_C_S
                self.cumul2Contrast_C_S += contrast_C_S**2

                contrast_C_S_A = cv[ic1,:] - cv[ic9,:]  #C-S_A
                self.cumulContrast_C_S_A += contrast_C_S_A
                self.cumul2Contrast_C_S_A += contrast_C_S_A**2


            # mean of posterior components:
            self.cumul_mean_apost += self.meanClassApost
            self.cumul_var_apost += self.varClassApost
            self.mean_mean_apost = self.cumul_mean_apost / self.nbItObservables
            self.mean_var_apost = self.cumul_var_apost / self.nbItObservables


            if 0:
                print 'nb of iterations', self.nbItObservables
                print 'tests for voxels 0 to 3 - condition 1:'
                print 'labels for the current iteration:',
                print self.labels[0,310:316]
                print 'nrls for the current iteration:',
                print self.currentValue[0,310:316]
                print 'Sum nrls for class inactiv: self.sum_nrls_both_classes_cond[c=0,cond=0,310:316]:'
                print self.sum_nrls_both_classes_cond[0,0,310:316]
                print 'Sum nrls for class activ: self.sum_nrls_both_classes_cond[c=1,cond=0,310:316]:'
                print self.sum_nrls_both_classes_cond[1,0,310:316]
                print 'Sum carr nrls for class inactiv: self.sum_nrls_carr_both_classes_cond[c=0,cond=0,310:316]:'
                print self.sum_nrls_carr_both_classes_cond[0,0,310:316]
                print 'Sum carr nrls for class activ: self.sum_nrls_carr_both_classes_cond[c=1,cond=0,310:316]:'
                print self.sum_nrls_carr_both_classes_cond[1,0,310:316]
                print '--'
                print 'self.mean_var_apost[0,0,310:316] - 1ere condition, classe inactive:',
                print self.mean_var_apost[0,0,310:316]
                print 'self.mean_var_apost[1,0,310:316] - 1ere condition, classe active:',
                print self.mean_var_apost[1,0,310:316]


                print '#########################################'
                print 'tests for voxels 0 to 3 - condition 2:'
                print 'labels for the current iteration:',
                print self.labels[1,310:316]
                print 'nrls for the current iteration:',
                print self.currentValue[1,310:316]
                print 'Sum nrls for class inactiv: self.sum_nrls_both_classes_cond[c=0,cond=1,310:316]:'
                print self.sum_nrls_both_classes_cond[0,1,310:316]
                print 'Sum nrls for class activ: self.sum_nrls_both_classes_cond[c=1,cond=1,310:316]:'
                print self.sum_nrls_both_classes_cond[1,1,310:316]
                print 'Sum carr nrls for class inactiv: self.sum_nrls_carr_both_classes_cond[c=0,cond=1,310:316]:'
                print self.sum_nrls_carr_both_classes_cond[0,1,310:316]
                print 'Sum carr nrls for class activ: self.sum_nrls_carr_both_classes_cond[c=1,cond=1,310:316]:'
                print self.sum_nrls_carr_both_classes_cond[1,1,310:316]
                print '--'
                print 'self.mean_var_apost[0,1,310:316] - 2eme condition, classe inactive:',
                print self.mean_var_apost[0,1,310:316]
                print 'self.mean_var_apost[1,1,310:316] - 2eme condition, classe active:',
                print self.mean_var_apost[1,1,310:316]


    def saveObservables(self, it):
        GibbsSamplerVariable.saveObservables(self, it)
        if self.labelsMeanHistory is not None :
            self.labelsMeanHistory = np.concatenate((self.labelsMeanHistory,
                                                  [self.meanLabels]))
        else :
            self.labelsMeanHistory = np.array([self.meanLabels.copy()])

        #print 'save trucs'

    def saveCurrentValue(self, it):
        #print 'self.labels', self.labels
        GibbsSamplerVariable.saveCurrentValue(self, it)
        if self.labelsSmplHistory is not None :
            self.labelsSmplHistory = np.concatenate((self.labelsSmplHistory,
                                                  [self.labels]))
        else :
            self.labelsSmplHistory = np.array([self.labels.copy()])

    def cleanObservables(self):
        GibbsSamplerVariable.cleanObservables(self)
        if 0: #hack to save cumulLabels if necessary
            del self.cumulLabels
        self.cleanMemory()





    def PPMcalculus(threshold_value, apost_mean_activ, apost_var_activ,  \
                apost_mean_inactiv, apost_var_inactiv, labels_activ, labels_inactiv):
        '''
        Function to calculate the probability that the nrl in voxel j,
        condition m, is superior to a given hreshold_value
        '''

        m1 = apost_mean_activ
        sig1 = apost_var_activ
        m2 = apost_mean_inactiv
        sig2 = apost_var_inactiv
        perc1 = labels_activ #proportion of samples drawn from the activ class
        perc2 = labels_inactiv #proportion of samples drawn from the inactiv class

        #posterior probability distribution
        fmix = lambda t: perc1 * 1/np.sqrt(2*np.pi*sig1**2)*np.exp(- (t - m1)**2 / (2*sig1**2) )  +  perc2 * 1/np.sqrt(2*np.pi*sig2**2)*np.exp(- (t - m2)**2 / (2*sig2**2) )
        Proba = quad(fmix, threshold_value, float('inf'))[0]

        return Proba



    def ThresholdPPM(proba_voxel, threshold_pval):
        if proba_voxel > threshold_pval:
            Proba = proba_voxel
        elif proba_voxel > threshold_pval:
            Proba = None

        return Proba


    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """

        # Precalculations and allocations :
        smplHRF = self.get_variable('hrf')
        self.imm = self.get_variable('beta').currentValue[0] < 0
        self.varYtilde = np.zeros((self.ny, self.nbVox), dtype=np.float64)
        self.aXh = np.empty((self.nbVox, self.ny, self.nbConditions), dtype=float)
        self.vycArray = np.zeros((self.nbVox, self.ny, self.nbConditions))
        self.sumaXh = np.zeros((self.ny, self.nbVox), dtype=float)
        self.computeVarYTildeOpt(smplHRF.varXh)
        self.varXhtQ = np.empty((self.nbConditions,self.ny),dtype=float)

        self.varClassApost = np.zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                   dtype=np.float64)
        self.sigClassApost = np.zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                   dtype=float)
        self.meanClassApost = np.zeros((self.nbClasses,self.nbConditions,
                                     self.nbVox), dtype=np.float64)
        self.meanApost = np.zeros((self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = np.zeros((self.nbConditions, self.nbVox), dtype=float)

        self.aa = np.zeros((self.nbConditions, self.nbConditions, self.nbVox),
                        dtype=float)

        if self.imm:
            self.sumRmatXhtQXh = np.zeros((self.nbConditions,self.nbVox),dtype=float)
            self.varXjhtQjeji = np.empty((self.nbVox), dtype=float)

        self.computeAA(self.currentValue, self.aa)

        self.iteration = 0



    def computeAA(self, nrls, destaa):
        # aa[m,n,:] == aa[n,m,:] -> nb ops can be /2
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                np.multiply(nrls[j,:], nrls[k,:],
                         destaa[j,k,:])


    def computeVarYTildeOpt(self, varXh):
        # C function:
        pyhrf.verbose(6, 'Calling C function computeYtilde ...')
        computeYtilde(varXh, self.currentValue, self.dataInput.varMBY,
                      self.varYtilde, self.sumaXh)
        #print 'sumaXh = ', self.sumaXh
        #print 'varYtilde = ', self.varYtilde
        #print 'Ytilde computing is finished ...'

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde.shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde)

    def sampleNextAlt(self, variables):
        varXh = self.samplerEngine.get_variable('hrf').varXh
        self.computeVarYTildeOpt(varXh)

    def computeComponentsApost(self, variables, j, gTQg):
        sIMixtP = self.samplerEngine.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = self.samplerEngine.get_variable('noise_var').currentValue
        varXh = self.samplerEngine.get_variable('hrf').varXh
        nrls = self.currentValue

        gTQgjrb = gTQg[j]/rb

        if pyhrf.verbose > 4:
            print 'Current components:'
            print 'mean CI = %f, var CI = %f' %(mean[self.L_CI,j], var[self.L_CI,j])
            print 'mean CA = %f, var CA = %f' %(mean[self.L_CA,j], var[self.L_CA,j])
            print 'gTQg =', gTQg[j]

        pyhrf.verbose(6, 'gTQg[%d] %s:'%(j,str(gTQg[j].shape)))
        pyhrf.verbose.printNdarray(6, gTQg[j])

        pyhrf.verbose(6, 'rb %s :'%str(rb.shape))
        pyhrf.verbose.printNdarray(6, rb)

        pyhrf.verbose(6, 'gTQgjrb %s :'%str(gTQgjrb.shape))
        pyhrf.verbose.printNdarray(6, gTQgjrb)

        ej = self.varYtilde + nrls[j,:] \
             * repmat(varXh[:,j],self.nbVox, 1).transpose()

        pyhrf.verbose(6, 'varYtilde %s :'%str((self.varYtilde.shape)))
        pyhrf.verbose.printNdarray(6, self.varYtilde)

        pyhrf.verbose(6, 'nrls[%d,:] %s :'%(j,nrls[j,:]))
        pyhrf.verbose.printNdarray(6, nrls[j,:])

        pyhrf.verbose(6, 'varXh[:,%d] %s :'%(j,str(varXh[:,j].shape)))
        pyhrf.verbose.printNdarray(6, varXh[:,j])

        pyhrf.verbose(6, 'repmat(varXh[:,%d],self.nbVox, 1).transpose()%s:' \
                          %(j,str((repmat(varXh[:,j],self.nbVox, 1).transpose().shape))))
        pyhrf.verbose.printNdarray(6, repmat(varXh[:,j],self.nbVox, 1).transpose())

        pyhrf.verbose(6, 'ej %s :'%str((ej.shape)))
        pyhrf.verbose.printNdarray(6, ej)

        np.divide(np.dot(self.varXhtQ[j,:],ej), rb, self.varXjhtQjeji)

        if pyhrf.verbose.verbosity > 5:
            pyhrf.verbose(5, 'np.dot(self.varXhtQ[j,:],ej) %s :' \
                              %str(np.dot(self.varXhtQ[j,:],ej).shape))
            pyhrf.verbose.printNdarray(5, np.dot(self.varXhtQ[j,:],ej))

            pyhrf.verbose(5, 'self.varXjhtQjeji %s :' \
                              %str(self.varXjhtQjeji.shape))
            pyhrf.verbose.printNdarray(5, self.varXjhtQjeji)

        for c in xrange(self.nbClasses):
            #print 'var[%d,%d] :' %(c,j), var[c,j]
            #print 'mean[%d,%d] :' %(c,j), mean[c,j]
            self.varClassApost[c,j,:] = 1./(1./var[c,j] + gTQgjrb)
            if 0:
                print 'shape of self.varClassApost[c,j,:] :', \
                    self.varClassApost.shape
            #print 'varClassApost[%d,%d,:]:' %(c,j), self.varClassApost[c,j,:]
            np.sqrt(self.varClassApost[c,j,:], self.sigClassApost[c,j,:])
            if c > 0: # assume 0 stands for inactivating class
                np.multiply(self.varClassApost[c,j,:],
                               add(mean[c,j]/var[c,j], self.varXjhtQjeji),
                               self.meanClassApost[c,j,:])
            else:
                np.multiply(self.varClassApost[c,j,:], self.varXjhtQjeji,
                         self.meanClassApost[c,j,:])

            pyhrf.verbose(5, 'meanClassApost %d cond %d :'%(c,j))
            pyhrf.verbose.printNdarray(5, self.meanClassApost[c,j,:])
            pyhrf.verbose(5, 'varClassApost %d cond %d :'%(c,j))
            pyhrf.verbose.printNdarray(5, self.varClassApost[c,j,:])
            pyhrf.verbose(5, 'shape of self.varClassApost[c,j,:] : %s' \
                              %str(self.varClassApost.shape))

    def computeVarXhtQ(self, h, varXQ):
        for j in xrange(self.nbConditions):
            self.varXhtQ[j,:] = np.dot(h,varXQ[j,:,:])

    def sampleNrlsSerial(self, rb, h, varCI, varCA, meanCA ,
                         gTQg, variables):

        pyhrf.verbose(3, 'Sampling Nrls (serial, spatial prior) ...')
        pyhrf.verbose(3, 'Label sampling: ' + str(self.sampleLabelsFlag))
        sIMixtP = self.samplerEngine.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = self.samplerEngine.get_variable('noise_var').currentValue

        # Add one dimension to be consistent with habituation model
        varXh = np.array([self.samplerEngine.get_variable('hrf').varXh],
                         dtype=np.float64)

        neighbours = self.dataInput.neighboursIndexes

        beta = self.samplerEngine.get_variable('beta').currentValue
        voxOrder = np.random.permutation(self.nbVox)

        sampleSmmNrl2(voxOrder.astype(np.int32), rb.astype(np.float64),
                      neighbours.astype(np.int32), self.varYtilde,
                      self.labels, varXh, self.currentValue,
                      self.nrlsSamples.astype(np.float64),
                      self.labelsSamples.astype(np.float64),
                      np.array([self.varXhtQ]).astype(np.float64),
                      gTQg.astype(np.float64),
                      beta.astype(np.float64), mean.astype(np.float64),
                      var.astype(np.float64), self.meanClassApost,
                      self.varClassApost, self.nbClasses,
                      self.sampleLabelsFlag+0, self.iteration,
                      self.nbConditions)
        if (self.varClassApost<=0).any():
            raise Exception('Negative posterior variances!')

        self.countLabels(self.labels, self.voxIdx, self.cardClass)

    def printState(self, verboseLevel):
        if pyhrf.verbose.verbosity >= verboseLevel:
            for j in xrange(self.nbConditions):
                #pyhrf.verbose(verboseLevel, 'All nrl cond %d:'%j)
                #pyhrf.verbose.printNdarray(verboseLevel, self.currentValue[j,:])
                pyhrf.verbose(verboseLevel, 'nrl cond %d = %1.3f(%1.3f)' \
                                  %(j,self.currentValue[j,:].mean(),
                                    self.currentValue[j,:].std()))
                for c in xrange(self.nbClasses):
                    #pyhrf.verbose(verboseLevel, 'All nrl %s cond %d:' \
                    #                  %(self.CLASS_NAMES[c],j))
                    ivc = self.voxIdx[c][j]
                    #pyhrf.verbose.printNdarray(verboseLevel,
                    #                           self.currentValue[j,ivc])

                    pyhrf.verbose(verboseLevel, 'nrl %s cond %d = %1.3f(%1.3f)' \
                                      %(self.CLASS_NAMES[c],j,
                                        self.currentValue[j,ivc].mean(),
                                        self.currentValue[j,ivc].std()))


    def sampleNrlsParallel(self, varXh, rb, h, varLambda, varCI, varCA,
                           meanCA, gTQg, variables):
        pyhrf.verbose(3, 'Sampling Nrls (parallel, no spatial prior) ...')
        for j in xrange(self.nbConditions):
            self.computeComponentsApost(variables, j, gTQg)
            if self.sampleLabelsFlag:
                pyhrf.verbose(3, 'Sampling labels - cond %d ...'%j)

                self.sampleLabels(j, variables)
                self.countLabels(self.labels, self.voxIdx, self.cardClass)

                pyhrf.verbose(3,'Sampling labels done!')
                pyhrf.verbose(6, 'All labels cond %d:'%j)
                pyhrf.verbose.printNdarray(6, self.labels[j,:])
                if self.trueLabels is not None:
                    pyhrf.verbose(6, 'All true labels cond %d:'%j)
                    pyhrf.verbose.printNdarray(6, self.trueLabels[j,:])

            for c in xrange(self.nbClasses):
                putmask(self.sigApost[j,:], self.labels[j,:]==c,
                        self.sigClassApost[c,j,:])
                putmask(self.meanApost[j,:],self.labels[j,:]==c,
                        self.meanClassApost[c,j,:])

            oldVal = self.currentValue[j,:]
            add(np.multiply(self.nrlsSamples[j,:], self.sigApost[j,:]),
                self.meanApost[j,:], self.currentValue[j,:])

            self.computeVarYTildeOpt(varXh)


    def sampleNextInternal(self, variables):
        #TODO : comment

        sIMixtP = self.get_variable('mixt_params')
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        rb = self.get_variable('noise_var').currentValue
        sHrf = self.get_variable('hrf')
        varXh = sHrf.varXh
        h = sHrf.currentValue
        self.nh = np.size(h)
        varLambda = self.get_variable('mixt_weights').currentValue

        #Ytilde(:,i) = Ytilde(:,i) + ( CptStruct.nrl_old(j,i) - ...
        #                              CptStruct.nrl(j,i)) * Xh(:,j);

        pyhrf.verbose(5,'varXh %s :' %str(varXh.shape))
        pyhrf.verbose.printNdarray(5, varXh)

        self.computeVarYTildeOpt(varXh)

        self.computeVarXhtQ(h, self.dataInput.matXQ)
        pyhrf.verbose(6,'varXhtQ %s :' %str(self.varXhtQ.shape))
        pyhrf.verbose.printNdarray(5, self.varXhtQ)

        self.labelsSamples = np.random.rand(self.nbConditions, self.nbVox)
        #print 'labelsSamples = ', self.labelsSamples
        self.nrlsSamples = np.random.randn(self.nbConditions, self.nbVox)

        gTQg = np.diag(np.dot(self.varXhtQ,varXh))

        if self.imm:
            self.sampleNrlsParallel(varXh, rb, h, varLambda, varCI,
                                    varCA, meanCA, gTQg, variables)
        else: #MMS
            self.sampleNrlsSerial(rb, h, varCI, varCA, meanCA, gTQg, variables)
            self.computeVarYTildeOpt(varXh)

        if (self.currentValue >= 1000).any() and pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )
            #pyhrf.verbose.set_verbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.computeAA(self.currentValue, self.aa)

        self.printState(4)
        self.iteration += 1 #TODO : factorize !!

        #print 'nrl = ', self.currentValue

    def reportDetection(self):
        if self.trueLabels is not None:
            try:
                for j in xrange(self.nbConditions):
                    wrong = np.where(self.trueLabels[j,:] != self.labels[j,:])
                    print 'Nb of wrongly detected :', len(wrong[0])
                    if len(wrong[0]) > 0:
                        print 'False inactivating:'
                        for w in wrong[0]:
                            if self.trueLabels[j,w] != 0 and self.labels[j,w] == 0:
                                print 'it%04d-cond%02d-vox%03d : nrl = %f' \
                                    %(self.iteration,j,w,self.currentValue[j,w])
                        print 'False activating:'
                        for w in wrong[0]:
                            if self.trueLabels[j,w] != 1 and self.labels[j,w] == 1:
                                print 'it%04d-cond%02d-vox%03d : nrl = %f' \
                                    %(self.iteration,j,w,self.currentValue[j,w])
                        if self.nbClasses == 3:
                            print 'False deactivating:'
                            for w in wrong[0]:
                                if self.trueLabels[j,w] != 2 and self.labels[j,w] == 2:
                                    print 'it%04d-cond%02d-vox%03d : nrl = %f' \
                                        %(self.iteration,j,w,self.currentValue[j,w])
            except Exception:
                # may happen if nb conditions in simulation != nb conditions
                # when estimating
                pass



    def calcFracLambdaTilde(self, cond, c1, c2, variables):
        sMixtP = self.get_variable('mixt_params')
        sWeightP = self.get_variable('mixt_weights')
        varLambda = sWeightP.currentValue
        var = sMixtP.getCurrentVars()
        means = sMixtP.getCurrentMeans()
        if self.samplerEngine.get_variable('beta').currentValue[cond] <= 0:
            ratio = ( varLambda[c1] * var[c2]**0.5 ) \
                /(varLambda[c2] * var[c1]**0.5 )
        else:
            ratio = (var[c2]/var[c1])**0.5

        return ratio[cond] * ( self.sigClassApost[c1,cond,:]              \
                               /self.sigClassApost[c2,cond,:] ) *         \
                               np.exp(0.5*(self.meanClassApost[c1,cond,:]**2 \
                                        /self.varClassApost[c1,cond,:]    \
                                        -self.meanClassApost[c2,cond,:]**2\
                                        /self.varClassApost[c2,cond,:]    \
                                        - means[c1, cond]**2              \
                                        / var[c1, cond]                   \
                                        + means[c2, cond]**2              \
                                        / var[c2, cond]                   \
                                            )\
                                       )

    # def calcFracLambdaTilde


    def sampleLabels(self, cond, variables):

        fracLambdaTilde = self.calcFracLambdaTilde(cond, self.L_CI, self.L_CA,
                                                   variables)
        varLambdaApost = 1./(1.+fracLambdaTilde)

        self.labels[cond,:] = self.labelsSamples[cond,:]<=varLambdaApost

        if pyhrf.verbose > 6:
            for i in xrange(self.nbVox):
                print 'it%04d-cond%02d-Vox%03d ...' %(self.iteration,cond,i)
                print 'mApostCA =', self.meanClassApost[self.L_CA,cond,i],
                print 'mApostCI =', self.meanClassApost[self.L_CI,cond,i]
                print 'sApostCA =', self.sigClassApost[self.L_CA,cond,i],
                print 'sApostCI =', self.sigClassApost[self.L_CI,cond,i]
                print 'rl_I_A =', fracLambdaTilde[i]
                print 'lambda Apost CA =', varLambdaApost[i]
                print 'random =', self.labelsSamples[cond,i]
                print '-> labels = ', self.labels[cond,i]


    def getFinalLabels(self, thres=None):
    #def getFinalLabels(self, thres):
        # take the argmax over classes
        return threshold_labels(self.meanLabels)
        #return threshold_labels(self.meanLabels, thres)
    #print self.cumulContrast

    def computeContrasts(self):
        #print self.contrasts_calc.iterkeys()
        pyhrf.verbose(2, 'computeContrasts ...')
        #print '~~~~~~~~~ self.cumulContrast :'
        #print self.cumulContrast_Lc_Rc
        # compute final contrasts:

        self.contrasts = {}
        self.contrastsVar = {}
        nit = self.nbItObservables
        pyhrf.verbose(5,'pour les contrastes: dict: %s'
                      %str(self.contrasts_calc))
        for name in self.contrasts_calc.iterkeys():
            #print 'name contraste: ', name
            self.contrasts[name] = self.cumulContrast[name]/nit
            self.contrastsVar[name] = self.cumul2Contrast[name]/nit - \
                self.contrasts[name]**2
        #print 'Les contrastes', self.contrasts
        #print 'Les variances', self.contrastsVar
        #print self.contrasts_calc.iterkeys
        if self.wip_variance_computation and \
                ('calculaudio' in self.dataInput.cNames):
            #for c1 in self.cumulContrast.iterkeys():
                #self.contrasts[c1] = {}
                #self.contrastsVar[c1] = {}
                #for c2 in self.cumulContrast[c1].iterkeys():
                    #self.contrasts[c1][c2] = self.cumulContrast[c1][c2]       \
                        #/ self.nbItObservables
                    #self.contrastsVar[c1][c2] = self.cumul2Contrast[c1][c2]   \
                                              #/ self.nbItObservables \
                                              #- self.contrasts[c1][c2]**2

            #print "COntrastsss!!"
            #print self.cumulContrast_Lc_Rc
            #Lc-Rc
            self.contrast_Lc_Rc = self.cumulContrast_Lc_Rc / self.nbItObservables
            self.contrast_var_Lc_Rc = self.cumul2Contrast_Lc_Rc / self.nbItObservables - self.contrast_Lc_Rc**2

            #V-A
            self.contrast_V_A = self.cumulContrast_V_A / self.nbItObservables
            self.contrast_var_V_A = self.cumul2Contrast_V_A / self.nbItObservables - self.contrast_V_A**2

            #C-S
            self.contrast_C_S = self.cumulContrast_C_S / self.nbItObservables
            self.contrast_var_C_S = self.cumul2Contrast_C_S / self.nbItObservables - self.contrast_C_S**2

            #C-S_A
            self.contrast_C_S_A = self.cumulContrast_C_S_A / self.nbItObservables
            self.contrast_var_C_S_A = self.cumul2Contrast_C_S_A / self.nbItObservables - self.contrast_C_S_A**2
            #print '%%%%%% contrast :',
            #print self.contrasts



            #For the variances estimates study: two conditions case
            #A/II 1/ Independant conditions
            #A/II 2/ Recuperation of the variance a posteriori for the good runs
            if 0:
                print 'self.varcontrast_cond_both_classes shape:', self.varcontrast_cond_both_classes.shape
                print 'self.varCon_2cond_indep_masked shape:', self.varCon_2cond_indep_masked.shape

                print 'shapes Covar and diff_nrl_mean:', self.Covar_non_masked.shape, self.diff_nrl_mean_non_masked.shape, (self.diff_nrl_mean_non_masked[0,:]*self.diff_nrl_mean_non_masked[1,:]).shape
                print 'shape Covar_masked', self.Covar_masked.shape

                #print 'Concerning contrasts: self.contrasts_calc.iteritems(), self.contrasts_calc[0]: ', self.contrasts_calc.keys()[0]
                print 'self.nbConditions:', self.nbConditions
                #print 'shapes self.contrastsVar, self.Covar_non_masked:', self.contrastsVar.shape,
                self.Covar_non_masked[:] = self.diff_nrl_mean_non_masked[0,:]*self.diff_nrl_mean_non_masked[1,:]

                #print 'self.contrastsVar[self.contrasts_calc.keys()[0]]', self.contrastsVar[self.contrasts_calc.keys()[0]]
                print 'self.contrasts_calc:'
                print self.contrasts_calc

            if 0:
                self.varCon_2cond_corr_non_masked = self.contrastsVar[self.contrasts_calc.keys()[0]] + self.contrastsVar[self.contrasts_calc.keys()[1]] -2*self.Covar_non_masked #works only in this case of 2cond and contrast = Cond1-Cond2



            if 0:
                print '##################'
                print 'self.finalLabels[0,310:316] - 1ere condition:',
                print self.finalLabels[0,310:316]
                print 'self.varcontrast_cond_both_classes[0,0,310:316] - 1ere condition, classe inactive:',
                print self.varcontrast_cond_both_classes[0,0,310:316]
                print 'self.varcontrast_cond_both_classes[0,1,310:316] - 1ere condition, classe active::',
                print self.varcontrast_cond_both_classes[1,0,310:316]
                print '--'
                print 'self.finalVariances[0,310:316]:',
                print self.finalVariances[0,310:316]
                print '--'

                print 'self.final_mean_var_a_post[0,310:316] - 1ere condition:',
                print self.final_mean_var_a_post[0,310:316]

                print ' ######'
                print 'self.finalLabels[0,310:316] - 2eme condition:',
                print self.finalLabels[1,310:316]
                print 'self.varcontrast_cond_both_classes[1,0,310:316] - 2eme condition, classe inactive:',
                print self.varcontrast_cond_both_classes[0,1,310:316]
                print 'self.varcontrast_cond_both_classes[1,1,310:316] - 2eme condition, classe active::',
                print self.varcontrast_cond_both_classes[1,1,310:316]
                print '--'
                print 'self.finalVariances[1,310:316]:',
                print self.finalVariances[1,310:316]
                print '--'

                print 'self.final_mean_var_a_post[1,310:316] - 2eme condition:',
                print self.final_mean_var_a_post[1,310:316]

            if (self.nbConditions > 1):

                pyhrf.verbose(1, 'Computing Contrasts ...')

                for j in xrange(self.nbVox):

                    #First case: labels at 0 for both conditions
                    if (self.finalLabels[0,j]==0) & (self.finalLabels[1,j]==0):
                        #case independant
                        self.varCon_2cond_indep_masked[0,j] = \
                            self.varcontrast_cond_both_classes[0,0,j] + \
                            self.varcontrast_cond_both_classes[0,1,j]
                        self.varCon_2cond_indep_apost[0,j]  = \
                            self.mean_var_apost[0,0,j] + self.mean_var_apost[0,1,j]

                        #case correlation
                        self.Covar_masked[0,j] = \
                            self.diff_nrl_mean_masked[0,0,j] * \
                            self.diff_nrl_mean_masked[0,1,j]
                        self.varCon_2cond_corr_masked[0,j] = \
                            self.varCon_2cond_indep_masked[0,j] - \
                            2*self.Covar_masked[0,j]
                        self.varCon_2cond_corr_apost[0,j] = \
                            self.mean_var_apost[0,0,j] + \
                            self.mean_var_apost[0,1,j] - 2*self.Covar_masked[0,j]

                    #Second case: labels at 0 for cond1 and at 1 for cond2
                    elif (self.finalLabels[0,j]==0) & (self.finalLabels[1,j]==1):
                        #case independant
                        self.varCon_2cond_indep_masked[1,j] = \
                            self.varcontrast_cond_both_classes[0,0,j] + \
                            self.varcontrast_cond_both_classes[1,1,j]
                        self.varCon_2cond_indep_apost[1,j] = \
                            self.mean_var_apost[0,0,j] + self.mean_var_apost[1,1,j]

                        #case correlation
                        self.Covar_masked[1,j] = self.diff_nrl_mean_masked[0,0,j] * \
                            self.diff_nrl_mean_masked[1,1,j]
                        self.varCon_2cond_corr_masked[1,j] = \
                            self.varCon_2cond_indep_masked[1,j] - \
                            2*self.Covar_masked[1,j]
                        self.varCon_2cond_corr_apost[1,j] = \
                            self.mean_var_apost[0,0,j] + \
                            self.mean_var_apost[1,1,j] - 2*self.Covar_masked[1,j]

                    #Third case: labels at 1 for cond1 and at 0 for cond2
                    elif (self.finalLabels[0,j]==1) & (self.finalLabels[1,j]==0):
                        #case independant
                        self.varCon_2cond_indep_masked[2,j] = \
                            self.varcontrast_cond_both_classes[1,0,j] + \
                            self.varcontrast_cond_both_classes[0,1,j]
                        self.varCon_2cond_indep_apost[2,j]  = \
                            self.mean_var_apost[1,0,j] + self.mean_var_apost[0,1,j]

                        #case correlation
                        self.Covar_masked[2,j] = self.diff_nrl_mean_masked[1,0,j] * \
                            self.diff_nrl_mean_masked[0,1,j]
                        self.varCon_2cond_corr_masked[2,j] = \
                            self.varCon_2cond_indep_masked[2,j] - \
                            2*self.Covar_masked[2,j]
                        self.varCon_2cond_corr_apost[2,j] = \
                            self.mean_var_apost[1,0,j] +  \
                            self.mean_var_apost[0,1,j] - \
                            2*self.Covar_masked[2,j]

                    #Fourth case: labels at 1 for both conditions
                    elif (self.finalLabels[0,j]==1) & (self.finalLabels[1,j]==1):
                        #case independant
                        self.varCon_2cond_indep_masked[3,j] = \
                            self.varcontrast_cond_both_classes[1,0,j] + \
                            self.varcontrast_cond_both_classes[1,1,j]
                        self.varCon_2cond_indep_apost[3,j] = \
                            self.mean_var_apost[1,0,j] + self.mean_var_apost[1,1,j]

                        #case correlation
                        self.Covar_masked[3,j] = self.diff_nrl_mean_masked[1,0,j] * \
                            self.diff_nrl_mean_masked[1,1,j]
                        self.varCon_2cond_corr_masked[3,j] = \
                            self.varCon_2cond_indep_masked[3,j] - \
                            2*self.Covar_masked[3,j]
                        self.varCon_2cond_corr_apost[3,j] = \
                            self.mean_var_apost[1,0,j] + \
                            self.mean_var_apost[1,1,j] - 2*self.Covar_masked[3,j]

                #print 'finalVariances: ', self.finalVariances, \
                #    self.finalVariances.shape

                #TOFIX !!
                # for contrast,conds in self.conds_in_contrasts.iteritems():
                #     iconds = [cnames.index(c) for c in conds]
                #     cov = np.vstack([self.Covar[cpl[0],cpl[1],:] \
                #                          for cpl in couples(iconds)]).sum(0)
                #     self.contrast_var[contrast] = \
                #         self.varcontrast_singleton[mask_class, icond, :].sum(1) + cov

            else:
                pyhrf.verbose(1, 'We have one condition only, no '\
                                  'contrast computing ...')


            if 0:
                print '##################'
                print 'self.mean_var_apost', self.mean_var_apost[1,0,:]
                print 'self.mean_var_apost', self.mean_var_apost[1,1,:]
                print '--'
                print 'self.varCon_2cond_indep_masked[0,310:316]: - 1ere cas',
                print self.varCon_2cond_indep_masked[0,310:316]
                print 'self.varCon_2cond_indep_masked[1,310:316]: - 2eme cas',
                print self.varCon_2cond_indep_masked[1,310:316]
                print 'self.varCon_2cond_indep_masked[2,310:316]: - 3eme cas',
                print self.varCon_2cond_indep_masked[2,310:316]
                print 'self.varCon_2cond_indep_masked[3,310:316]: - 4eme cas',
                print self.varCon_2cond_indep_masked[3,310:316]
                print '--'
                print 'self.varCon_2cond_indep_apost[0,310:316]: - 1ere cas',
                print self.varCon_2cond_indep_apost[0,310:316]
                print 'self.varCon_2cond_indep_apost[1,310:316]: - 2eme cas',
                print self.varCon_2cond_indep_apost[1,310:316]
                print 'self.varCon_2cond_indep_apost[2,310:316]: - 3eme cas',
                print self.varCon_2cond_indep_apost[2,310:316]
                print 'self.varCon_2cond_indep_apost[3,310:316]: - 4eme cas',
                print self.varCon_2cond_indep_apost[3,310:316]

                print '-----'
                print 'Positions different from zeros:'
                print 'self.mean_var_apost >0:', np.where(self.mean_var_apost>0)



    def get_final_summary(self):
        s = GibbsSamplerVariable.get_final_summary(self)
        vi = [range(self.nbConditions) for c in xrange(self.nbClasses)]
        cc = np.zeros((self.nbClasses, self.nbConditions), dtype=int)
        s += ' labels sampling report: \n'
        self.countLabels(self.finalLabels, vi, cc)
        sv = get_2Dtable_string(cc.T, self.dataInput.cNames, self.CLASS_NAMES,
                                precision=0)
        if '\n' in sv:
            s += ' - final labels:\n' + sv
        else:
            s += ' - final labels: ' + sv + '\n'
        if self.trueLabels is not None:
            #nlabs = len(unique(self.trueLabels))
            nlabs = self.nbClasses
            vi = [range(self.nbConditions) for c in xrange(nlabs)]
            cc = np.zeros((nlabs, self.nbConditions), dtype=int)
            self.countLabels(self.trueLabels, vi, cc)
            if nlabs <= self.nbClasses:
                sv = get_2Dtable_string(cc.T, self.dataInput.cNames,
                                        self.CLASS_NAMES[:nlabs], precision=0)
            else:
                cn = self.CLASS_NAMES+['C%d'%c for c in xrange(self.nbClasses,
                                                               nlabs+1)]
                sv = get_2Dtable_string(cc.T, self.dataInput.cNames, cn, precision=0)
            if '\n' in sv:
                s += ' - true labels:\n' + sv
            else:
                s += ' - true labels: ' + sv + '\n'
            if self.trueLabels.shape == self.finalLabels.shape:
                #TODO: adapt to cases where nbClasses differ btw true and estim
                errorRate = np.zeros((self.nbConditions, self.nbClasses))
                for j in xrange(self.nbConditions):
                    for c in xrange(self.nbClasses):
                        # select which true labels are not in the
                        # considered class:
                        tlnotc = (self.trueLabels[j,:] != c)
                        # select which estimated labels are in the
                        # considered class:
                        flc = (self.finalLabels[j,:] == c)
                        # select which labels are classified in the
                        # considered class and were not truely in this class:
                        diffs = np.bitwise_and(flc,tlnotc)
                        errorRate[j,c] = diffs.sum()*100. / cc[c,j]
                sv = get_2Dtable_string(errorRate, self.dataInput.cNames,
                                        self.CLASS_NAMES,precision=1)
                if '\n' in sv:
                    s += ' - error (percent):\n' + sv
                else:
                    s += ' - error (percent): ' + sv + '\n'
        return s

    def cleanMemory(self):

        self.meanClassApost = self.meanClassApost.astype(np.float32)
        self.varClassApost = self.varClassApost.astype(np.float32)
        self.meanLabels = self.meanLabels.astype(np.float32)
        self.freq_above_thresh = self.freq_above_thresh.astype(np.float32)
        self.freq_above_Multi_thresh = self.freq_above_Multi_thresh.astype(np.float32)
        # clean memory of temporary variables :
        if self.imm:
            del self.sumRmatXhtQXh
            del self.varXjhtQjeji

        # del self.varClassApost
        # del self.meanClassApost

        del self.sigClassApost
        del self.sigApost
        del self.meanApost
        if hasattr(self, 'aa'):
            del self.aa
        if hasattr(self, 'aXh'):
            del self.aXh
        if hasattr(self, 'varYtilde'):
            del self.varYtilde
        if hasattr(self, 'varXhtQ'):
            del self.varXhtQ
        if hasattr(self, 'sumaXh'):
            del self.sumaXh
        if hasattr(self, 'vycArray'):
            del self.vycArray
        if hasattr(self,'labelsSamples'):
            del self.labelsSamples
        if hasattr(self,'nrlsSamples'):
            del self.nrlsSamples
        #del self.corrEnergies
        del self.labels
        del self.voxIdx

        if self.wip_variance_computation:
            del self.saveNRL

        if not self.wip_variance_computation: #and self.computeContrastsFlag:
            del self.cumulLabels
            #del self.mean_both_classes_cond
            #del self.mean_mean_apost
            #del self.mean_var_apost
            # del self.cumul2Contrast_C_S_A
            # del self.cumul2Contrast_C_S
            # del self.cumul2Contrast_Lc_Rc
            # del self.cumul2Contrast_V_A
            # del self.cumulContrast_C_S_A
            # del self.cumulContrast_Lc_Rc
            # del self.cumulContrast_V_A
            # del self.cumulContrast_C_S
            del self.cumul_mean_apost
            del self.cumul_var_apost
            #del self.diff_nrl_mean_masked
            #del self.diff_nrl_mean_non_masked
            del self.final_mean_var_a_post
            # del self.varCon_2cond_corr_apost
            # del self.varCon_2cond_corr_masked
            del self.varCon_2cond_indep_apost
            del self.varCon_2cond_indep_masked
            # del self.varcontrast_cond_both_classes
            del self.finalVariances
            del self.sum_nrls_carr_both_classes_cond
            del self.sum_nrls_both_classes_cond
            # del self.Covar_non_masked
            # del self.Covar_masked

    def markWrongLabels(self, labels):
        if self.trueLabels != None:
            for j in xrange(self.nbConditions):
                #print 'labels :'
                #print labels[j,:]
                #print 'trueLabels '
                #print self.trueLabels[j,:]
                el = (labels[j,:] == self.L_CA)
                tl = (self.trueLabels[j,:] == self.L_CA)
                nel = np.bitwise_not(el)
                ntl = np.bitwise_not(tl)
                labels[j, np.bitwise_and(el,tl)] = self.L_CA
                labels[j, np.bitwise_and(nel,ntl)] = self.L_CI
                labels[j, np.bitwise_and(el,ntl)] = self.FALSE_POS
                labels[j, np.bitwise_and(nel,tl)] = self.FALSE_NEG
                #print '-> marked :'
                #print labels[j,:]

    def finalizeSampling(self):
        GibbsSamplerVariable.finalizeSampling(self)

        self.finalLabels = self.getFinalLabels()
        #self.finalLabels = self.getFinalLabels(0.8722)
        #self.markWrongLabels(self.finalLabels)
        #print 'finalLabels.shape', self.finalLabels.shape

        smplHRF = self.samplerEngine.get_variable('hrf')

        # Correct sign ambiguity :
        if hasattr(smplHRF, 'detectSignError'):
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


        if self.computeContrastsFlag and self.wip_variance_computation:
            #Work on varainces estimates
            #self.masked_var_cond

            ########
            #np.seterr(all='ignore') # to ignore warning when nan is generated or when there's divide by zero, here some parts of cummulLabels are equal
                                    # to zero so we have nan in "self.mean_both_classes_cond"
            ########

            #print 'sum_nrls_both_classes_cond :',self.sum_nrls_both_classes_cond
            #print 'cumulLabels :', self.cumulLabels
            self.mean_both_classes_cond           = self.sum_nrls_both_classes_cond / self.cumulLabels
            #print 'mean_both_classes_cond :', self.mean_both_classes_cond
            #self.mean_class_inactiv_cond         = self.sum_nrls_class_inactiv_cond / self.cumulLabels[1,0,:]
            self.varcontrast_cond_both_classes    = self.sum_nrls_carr_both_classes_cond / self.cumulLabels - (self.mean_both_classes_cond)**2
            #self.varcontrast_cond_class_inactiv  = self.sum_nrls_carr_class_inactiv_cond / self.cumulLabels[1,0,:] - (self.mean_class_inactiv_cond)**2

            #print 'self.mean_class_activ_cond', self.mean_class_activ_cond, self.mean_class_activ_cond.shape
            #print 'self.varcontrast_cond_class_activ', self.varcontrast_cond_class_activ, self.varcontrast_cond_class_activ.shape

            #print 'shapes:', self.sum_nrls_carr_both_classes_cond.shape, self.cumulLabels[1,0,:].shape, self.mean_both_classes_cond.shape, self.currentValue.shape
            #print 'self.finalVariances.shape:', self.finalVariances.shape
            #print 'self.varcontrast_cond_class_activ.shape', self.varcontrast_cond_both_classes.shape

            for icond in xrange(self.nbConditions):
                for j in xrange(self.nbVox):
                    if self.finalLabels[icond,j]==1:
                        self.finalVariances[icond,j] = self.varcontrast_cond_both_classes[1,icond,j]
                        self.final_mean_var_a_post[icond,j] = self.mean_var_apost[1,icond,j]
                    elif self.finalLabels[icond,j]==0:
                        self.finalVariances[icond,j] = self.varcontrast_cond_both_classes[0,icond,j]
                        self.final_mean_var_a_post[icond,j] = self.mean_var_apost[0,icond,j]

            #print 'Comparisons between self.varcontrast_cond_both_classes and self.finalVariances:'
            #print self.varcontrast_cond_both_classes, self.finalVariances
            #print 'self.finalLabels', self.finalLabels

            #print 'tests end for condition1: '
            #print 'Variances for class activ: self.varcontrast_cond_both_classes[c=1, cond=0, 310:316]:'
            #print self.varcontrast_cond_both_classes[1, 0, 310:316]
            #print 'Variances for class inactiv: self.varcontrast_cond_both_classes[c=0, cond=0, 310:316]:'
            #print self.varcontrast_cond_both_classes[0, 0, 310:316]
            #print 'Final variances for condition1: self.finalVariances[cond=0, 310:316]:'
            #print self.finalVariances[0, 310:316]

            #print '##################'
            #print 'tests for condition 2:'
            #print 'Variances for class activ: self.varcontrast_cond_both_classes[c=1, cond=1, 310:316]:'
            #print self.varcontrast_cond_both_classes[1, 1, 310:316]
            #print 'Variances for class inactiv: self.varcontrast_cond_both_classes[c=0, cond=1, 310:316]:'
            #print self.varcontrast_cond_both_classes[0, 1, 310:316]
            #print 'Final variances for condition1: self.finalVariances[cond=1, 310:316]:'
            #print self.finalVariances[1, 310:316]

            #print 'Non zeros positions for self.mean_both_classes_cond :'
            #print np.where(self.mean_both_classes_cond>0)
            #print 'Non zeros for self.sum_nrls_carr_both_classes_cond :'
            #print np.where(self.sum_nrls_carr_both_classes_cond>0)
            #print 'Non zeros for self.sum_nrls_both_classes_cond :'
            #print np.where(self.sum_nrls_both_classes_cond>0)
            #print 'Non zeros positions for self.varcontrast_cond_both_classes :'
            #print np.where(self.varcontrast_cond_both_classes>0)

            #print 'To compare:', (self.sum_nrls_carr_class_activ_cond / self.cumulLabels[1,0,:])[169], (self.mean_class_activ_cond**2)[169]

            # -------------- For contrast variance analysis -------------------------------

            #print 'self.finalLabels infos :', np.where(self.finalLabels==1), np.where(self.finalLabels==0)

            if 0:
                print '###############################################'
                print 'Verification concerning the division of 3D numpy array in python:'
                print 'shapes:', self.mean_both_classes_cond.shape, self.sum_nrls_both_classes_cond.shape, self.cumulLabels.shape, self.varcontrast_cond_both_classes.shape
                print 'self.mean_both_classes_cond[0,1,300:306]:', self.mean_both_classes_cond[0,1,300:306]
                print 'self.sum_nrls_both_classes_cond[0,1,300:306]:', self.sum_nrls_both_classes_cond[0,1,300:306]
                print 'self.cumulLabels[0,1,300:306]:', self.cumulLabels[0,1,300:306]
                print 'self.sum_nrls_carr_both_classes_cond[0,1,300:306]:', self.sum_nrls_carr_both_classes_cond[0,1,300:306]
                print 'self.varcontrast_cond_both_classes[0,1,300:306]:', self.varcontrast_cond_both_classes[0,1,300:306]


        if self.computeContrastsFlag:
            self.computeContrasts()
            #self.cleanMemory()


        if self.samplerEngine.check_ftval is not None:
            if self.trueLabels is None:
                pyhrf.verbose(4, 'Warning: no true labels to check against')
            elif self.sampleLabels:
                fv = self.finalLabels
                tv = self.trueLabels
                diffs = (fv != tv)
                delta = diffs.sum()*1. / fv.shape[1]
                if delta > 0.05:
                    m = "Final value of labels is not close to " \
                    "true value.\n -> %%diffs: %1.2f\n" \
                    " Final value:\n %s\n True value:\n %s\n" \
                    %(delta, str(fv), str(tv))
                    if self.samplerEngine.check_ftval == 'raise':
                        raise Exception(m)
                    elif self.samplerEngine.check_ftval == 'print':
                        print '\n'.join(['!! '+ s for s in m.split('\n')])

        self.compute_summary_stats()

    def getRocData(self, dthres=0.005):

        if self.trueLabels is not None:
            thresholds = arange(0,1/dthres) * dthres
            oneMinusSpecificity = np.zeros((self.nbConditions, len(thresholds)))
            sensitivity = np.zeros((self.nbConditions, len(thresholds)))

            for it,thres in enumerate(thresholds):
                labs = threshold_labels(self.meanLabels,thres)
                self.markWrongLabels(labs)
                for cond in xrange(self.nbConditions):
                    if 1 and self.dataInput.cNames[cond] == 'audio':
                        print "**cond %d **" %cond
                        print 'marked labels:'
                        print labs[cond,:]
                        print 'simulated labels:'
                        print self.dataInput.simulData.nrls.labels[cond,:]
                    counts = bincount(labs[cond,:])
                    nbTrueNeg = counts[0]
                    nbTruePos = counts[1] if len(counts)>1 else 0
                    fp = self.FALSE_POS
                    nbFalsePos = counts[fp] if len(counts)>fp else 0
                    fn = self.FALSE_NEG
                    nbFalseNeg = counts[fn] if len(counts)>fn else 0
                    if 1 and self.dataInput.cNames[cond] == 'audio':
                        print 'TN :', nbTrueNeg
                        print 'TP :', nbTruePos
                        print 'FP :', nbFalsePos
                        print 'FN :', nbFalseNeg
                    if nbTruePos == 0:
                        sensitivity[cond,it] = 0
                    else:
                        sensitivity[cond,it] = nbTruePos /                    \
                                               (nbTruePos+nbFalseNeg+0.0)
                    spec = 1-nbTrueNeg/(nbTrueNeg+nbFalsePos+0.0)
                    oneMinusSpecificity[cond,it] = spec
                    if 1 and self.dataInput.cNames[cond] == 'audio':
                        print '-> se = ', sensitivity[cond, it]
                        print '-> 1-sp = ', oneMinusSpecificity[cond,it]
            spGrid = arange(0.,1.,0.01)
            omspec = np.zeros((self.nbConditions, len(spGrid)))
            sens = np.zeros((self.nbConditions, len(spGrid)))
            for cond in xrange(self.nbConditions):
                order = argsort(oneMinusSpecificity[cond,:])
                if oneMinusSpecificity[cond,order][0] != 0.:
                    osp = np.concatenate(([0.],oneMinusSpecificity[cond,order]))
                    se = np.concatenate(([0.],sensitivity[cond,order]))
                else:
                    osp = oneMinusSpecificity[cond,order]
                    se = sensitivity[cond,order]

                if osp[-1] != 1.:
                    osp = np.concatenate((osp,[1.]))
                    se = np.concatenate((se,[1.]))

                sens[cond,:] = resampleToGrid(osp, se, spGrid)
                omspec[cond, :] = spGrid
                if 1 and self.dataInput.cNames[cond] == 'audio':
                    print '-> se :'
                    print sens[cond,:]
                    print 'spec grid :'
                    print spGrid
            return sens, omspec
        else:
            return None, None


    def compute_summary_stats(self):
        self.stats = {}

        pyhrf.verbose(4, 'Compute PPM outputs ...')
        vthresh = getattr(self, 'ppm_value_thresh', 0)
        if hasattr(self, 'count_above_thresh'):
            ppm_mcmc = self.freq_above_thresh
            ppm_tag = 'PPM_g_MCMC'
            self.stats[ppm_tag] = ppm_mcmc

            ppm_mcmc = self.freq_above_Multi_thresh
            ppm_tag = 'PPM_g_MCMC_MultiThresh'
            self.stats[ppm_tag] = ppm_mcmc

            #if self.smplHistory is not None:

                ##quant = [1-self.ppm_proba_thresh]
                #ppm_mcmc = np.zeros_like(self.finalValue)
                #for j in range(self.nbConditions):
                    ## ppm_mcmc[j,:] = mquantiles(self.smplHistory[its,j,:],
                    ##                            prob=quant, axis=0)
                    #ppm_mcmc[j,:] = cpt_ppm_a_mcmc(self.smplHistory[self.samplerEngine.nbSweeps:,j,:],
                                                       #self.ppm_proba_thresh)
                #ppm_tag = 'PPM_a_MCMC'
                #self.stats[ppm_tag] = ppm_mcmc


            #if hasattr(self, 'meanClassApost'):
                #from pyhrf.stats import gm_cdf

                #mci = self.meanClassApost[self.L_CI,:,:]
                #vci = self.varClassApost[self.L_CI,:,:]
                #pci = self.meanLabels[self.L_CI,:,:]
                #mca = self.meanClassApost[self.L_CA,:,:]
                #vca = self.varClassApost[self.L_CA,:,:]
                #pca = self.meanLabels[self.L_CA,:,:]

                ## PPM as sf(thresh) of \sum_i \Nc(m_apost_i, v_apost_i)
                #ppm_tag = 'PPM_g_apost'
                #ppm_nrls = np.zeros_like(self.finalValue)
                #for i in xrange(self.nbConditions):
                    ##avoid underflow errors in pdf computation:
                    #v = self.varClassApost[:,i,:].astype(np.float64)
                    #ppm_nrls[i,:] = 1 - gm_cdf(vthresh,
                                               #self.meanClassApost[:,i,:],
                                               #v,
                                               #self.meanLabels[:,i,:])
                #self.stats[ppm_tag] = ppm_nrls

                ## PPM as inv_cdf(1-ppm_proba) of \sum_c \Nc(m_apost_c, v_apost_c)
                ## TODO ! -> need numerical computation (-> loop over voxels...)


            #from scipy.stats import norm

            ##PPM as isf(proba) of \Nc(mean_MCMC, var_MCMC)
            #output_name = 'PPM_a_norm_online'
            #pthresh = getattr(self, 'ppm_proba_thresh', 0.05)
            #ppm_empirical_pt = norm.isf(pthresh, self.finalValue,
                                        #self.error**.5)

            #self.stats[output_name] = ppm_empirical_pt

            ##PPM as sf(thresh) of \Nc(mean_MCMC, var_MCMC)
            #output_name = 'PPM_g_norm_online'
            #if self.error.size != 1:
                #self.error[np.where(self.error==0.)] = 1e-6
                #x = ((vthresh-self.finalValue)/self.error**.5).clip(-20,10)
            ## for ls,es in zip(self.finalValue, self.error**.5):
            ##     for l,e in zip(ls,es):
                #ppm_empirical_vt = norm.sf(x)
            #else:
                #pyhrf.verbose(1, 'error is empty and thus put to 0')
            #ppm_empirical_vt=0
            #pyhrf.verbose(1, 'Warning ppm_empiricall_vt put to 0')
            #self.stats[output_name] = ppm_empirical_vt


            ##PPM as isf(proba) of \Nc(mean_c, var_c) with c=argmax(labels)
            #output_name = 'PPM_a_norm_max_q'

            #argmax_labels = np.argmax(self.meanLabels,0)
            #mu_q_max = np.zeros_like(self.finalValue)
            #var_q_max = np.zeros_like(self.finalValue)
            #for i in range(self.nbClasses):
                #m = np.where(argmax_labels==i)
                #mu_q_max[m] = self.meanClassApost[i,m[0],m[1]]
                #var_q_max[m] = self.varClassApost[i,m[0],m[1]]

            #var_q_max[np.where(var_q_max==0)] = 1e-10
            #ppm_napprox_pt = norm.isf(pthresh, mu_q_max, var_q_max**.5)

            #self.stats[output_name] = ppm_napprox_pt

            ##PPM as sf(thresh) of \Nc(mean_c, var_c) with c=argmax(labels)
            #output_name = 'PPM_g_norm_max_q'
            #ppm_napprox_vt = norm.sf(vthresh, loc=mu_q_max,
                                     #scale=var_q_max**.5)

            #self.stats[output_name] = ppm_napprox_vt

            ## pvalue for H_0: A=0 and A ~ \Nc(0, var_c) with c=argmax(labels)
            #output_name = 'pval_max_q'
            #x = (self.finalValue/var_q_max**.5).clip(-20,10)
            #pval = norm.sf(x)
            #self.stats[output_name] = pval

            ## pvalue for H_0: A=0 and A ~ \Nc(0, var_MCMC)
            #output_name = 'pval_online'
            #x = (self.finalValue/self.error**.5).clip(-20,10)
            #pval = norm.sf(x)
            #self.stats[output_name] = pval



    def getClassifRate(self):
        r = np.zeros((self.nbClasses, self.nbConditions))
        for j in xrange(self.nbConditions):
            for ic in xrange(self.nbClasses):
                idx = np.where(self.finalLabels[j,:] == ic)
                r[ic,j] = (self.trueLabels[j,idx] == ic).sum(dtype=float) / \
                    (self.trueLabels[j,:] == ic).sum(dtype=float)
        return r

    def getOutputs(self):

        outputs = GibbsSamplerVariable.getOutputs(self)
        cn = self.dataInput.cNames

        axes_names = ['voxel']
        roi_lab_vol = np.zeros(self.nbVox, dtype=np.int32) + \
            self.dataInput.roiId
        outputs['roi_mapping'] = xndarray(roi_lab_vol, axes_names=axes_names,
                                        value_label='ROI')

        if self.rescale_results:
            shrf = self.samplerEngine.get_variable('hrf')
            xh = shrf.calcXh(shrf.finalValue[1:-1])
            nrl_rescaled = np.zeros_like(self.finalValue)
            for c in xrange(xh.shape[1]):
                nrl_rescaled[c,:] = self.finalValue[c,:] * \
                    (xh[:,c]**2).sum()**.5
            outputs['nrl_rescaled'] = xndarray(nrl_rescaled,
                                             axes_names=self.axes_names,
                                             axes_domains=self.axes_domains,
                                             value_label=self.value_label)
            ad = {'condition':cn,
                  'time' : np.arange(self.dataInput.ny)*self.dataInput.tr
                  }
            outputs['design_matrix'] = xndarray(xh,
                                              axes_names=['time','condition'],
                                              axes_domains=ad)

        if pyhrf.__usemode__ == pyhrf.DEVEL:
            if hasattr(self, 'finalValue_sign_corr'):
                outputs['nrl_sign_corr'] = xndarray(self.finalValue_sign_corr,
                                                  axes_names=self.axes_names,
                                                  axes_domains=self.axes_domains,
                                                  value_label=self.value_label)


            axes_names = ['class','condition', 'voxel']
            axes_domains = {'condition' : cn, 'class' : self.CLASS_NAMES}

            t = self.activ_thresh
            from scipy.stats.mstats import mquantiles
            region_is_active = mquantiles(self.finalValue.max(0), prob=[.9]) > \
                self.activ_thresh
            region_is_active = region_is_active.astype(np.int16)
            pyhrf.verbose(5, 'mquantiles(self.finalValue.max(0), prob=[.9]):')
            pyhrf.verbose.printNdarray(5, mquantiles(self.finalValue.max(0),
                                                  prob=[.9]))
            pyhrf.verbose(5, 'self.finalValue.mean(1).max(): %f' \
                              %self.finalValue.mean(1).max())
            pyhrf.verbose(5, '(self.finalValue.max(0) > t).sum(): %d' \
                              %(self.finalValue.max(0) > t).sum())
            region_is_active = np.tile(region_is_active, self.nbVox)

            outputs['active_regions_from_nrls'] = xndarray(region_is_active,
                                                         axes_names=['voxel'])

            if hasattr(self, 'cumulLabels'):
                outputs['pm_cumulLabels'] = xndarray(self.cumulLabels,
                                                   axes_names=axes_names,
                                                   axes_domains=axes_domains)


            outputs['labels_pm'] = xndarray(self.meanLabels,
                                          axes_names=axes_names,
                                          axes_domains=axes_domains,
                                          value_label="pm Labels")


            #if self.trueLabels is not None:
            #    outputs['pmLabels'].applyMask(self.trueLabelsMask)


            axes_names = ['condition', 'voxel']
            axes_domains = {'condition' : cn}
            l = self.finalLabels.astype(np.int32)
            outputs['labels_pm_thresh'] = xndarray(l, axes_names=axes_names,
                                                 axes_domains=axes_domains,
                                                 value_label="pm Labels Thres")



#             if hasattr(self, 'meanBeta'):
#                 #print 'output beta mapped !!!!'
#                 axes_names = ['condition', 'voxel']
#                 nbv, nbc = self.nbVox, self.nbConditions
#                 repeatedBeta = repeat(self.meanBeta, nbv).reshape(nbc, nbv)
#                 outputs['pm_BetaMapped'] = xndarray(repeatedBeta,
#                                                   axes_names=axes_names,
#                                                   axes_domains=axes_domains,
#                                                   value_label="pm Beta")

            if 0:
                axes_names = ['gamma', 'condition', 'voxel']
                axes_domains = {'gamma': self.ppm_value_multi_thresh, 'condition' : cn}
                outputs['PPM_g_MCMC_MultiThresh'] = xndarray(self.stats['PPM_g_MCMC_MultiThresh'], axes_names=axes_names,
                                        axes_domains=axes_domains)

                for stat_name, stat in self.stats.iteritems():
                    if stat_name != 'PPM_g_MCMC_MultiThresh':
                        axes_names = ['condition', 'voxel']
                        axes_domains = {'condition' : cn}
                        outputs[stat_name] = xndarray(stat, axes_names=axes_names,
                                                axes_domains=axes_domains)


            if hasattr(self, 'meanClassApost'):
                mci = self.meanClassApost[self.L_CI,:,:]
                vci = self.varClassApost[self.L_CI,:,:]
                pci = self.meanLabels[self.L_CI,:,:]
                mca = self.meanClassApost[self.L_CA,:,:]
                vca = self.varClassApost[self.L_CA,:,:]
                pca = self.meanLabels[self.L_CA,:,:]


                outputs['mean_CA_apost'] = xndarray(mca,
                                                  axes_names=axes_names,
                                                  axes_domains=axes_domains)

                outputs['var_CA_apost'] = xndarray(vca,
                                                 axes_names=axes_names,
                                                 axes_domains=axes_domains)

                outputs['proba_CA_apost'] = xndarray(pca,
                                                   axes_names=axes_names,
                                                   axes_domains=axes_domains)


                outputs['mean_CI_apost'] = xndarray(mci,
                                                  axes_names=axes_names,
                                                  axes_domains=axes_domains)

                outputs['var_CI_apost'] = xndarray(vci,
                                                 axes_names=axes_names,
                                                 axes_domains=axes_domains)

                outputs['proba_CI_apost'] = xndarray(pci,
                                                   axes_names=axes_names,
                                                   axes_domains=axes_domains)



            if hasattr(self, 'labelsMeanHistory') and \
                    self.labelsMeanHistory is not None:
                axes_names = ['iteration', 'class', 'condition', 'voxel']
                axes_domains = {'condition' : cn,
                               'class': self.CLASS_NAMES,
                               'iteration': self.obsHistoryIts}
                outputs['labels_pm_hist'] = xndarray(self.labelsMeanHistory,
                                                  axes_names=axes_names,
                                                  axes_domains=axes_domains,
                                                  value_label="label")

            if hasattr(self, 'labelsSmplHistory') and \
                    self.labelsSmplHistory is not None:
                axes_names = ['iteration', 'condition', 'voxel']
                axes_domains = {'condition' : cn,
                               'iteration':self.smplHistoryIts}
                outputs['labels_smpl_hist'] = xndarray(self.labelsSmplHistory,
                                                  axes_names=axes_names,
                                                  axes_domains=axes_domains,
                                                  value_label="label")


            if self.trueLabels is not None:
                if 0:
                    mlabels = self.meanLabels[self.L_CA,:,:]
                    #easy_install --prefix=$USRLOCAL -U scikits.learn
                    se,sp,auc = compute_roc_labels_scikit(mlabels,
                                                          self.trueLabels)
                    sensData, specData = se, sp
                else:
                    sensData,specData,auc = compute_roc_labels(self.meanLabels,
                                                               self.trueLabels,
                                                               0.005,
                                                               self.L_CA,
                                                               self.L_CI,
                                                               self.FALSE_POS,
                                                               self.FALSE_NEG)

                    pyhrf.verbose(2, 'Areas under ROC curves are : %s' \
                                      %str(auc))
                    #auc = np.array([trapz(sensData[j,:], specData[j,:])
                    #             for j in xrange(self.nbConditions)])
                    #print auc

                # axes_names = ['condition']
                # outName = 'Area under ROC curve'
                # outputs[outName] = xndarray(area, axes_names=axes_names,
                #                           axes_domains={'condition' : cn})

                axes_names = ['condition','1-specificity']
                outName = 'ROC'
                ad = {'1-specificity':specData[0],'condition':cn}
                outputs[outName] = xndarray(sensData, axes_names=axes_names,
                                          axes_domains=ad,
                                          value_label='sensitivity')

                axes_names = ['condition']
                outputs['AUROC'] = xndarray(auc, axes_names=axes_names,
                                          axes_domains={'condition':cn})

                cRate = self.getClassifRate()
                axes_names = ['class', 'condition']
                ad = {'condition':cn, 'class':self.CLASS_NAMES}
                outputs['labels_classif_rate'] = xndarray(cRate,
                                                        axes_names=axes_names,
                                                        axes_domains=ad)

            if self.trueLabels is not None:
                markedLabels = self.getFinalLabels().copy()
                #markedLabels = self.getFinalLabels(0.8722).copy()
                self.markWrongLabels(markedLabels)
                axes_names = ['condition', 'voxel']
                ad = {'condition':cn}
                outputs['labels_thresh_marked'] = xndarray(markedLabels,
                                                         axes_names=axes_names,
                                                         axes_domains=ad)


        #axes_names = ['condition', 'ny', 'nh']
            #axes_domains = {'condition' : cn,
                           #'ny': arange(self.ny),
               #'nh': arange(self.nh)}
            #outputs['varX'] = xndarray(self.dataInput.varX,
                                     #axes_names=axes_names,
                                     #axes_domains=axes_domains,
                                     #value_label="varX")
        #outputs['varXCond'] = xndarray(self.dataInput.varSingleCondXtrials,
                                         #axes_names=axes_names,
                                         #axes_domains=axes_domains,
                                         #value_label="varCondXtrials")
        #outputs['multXXcond'] = xndarray(self.dataInput.varSingleCondXtrials*self.dataInput.varX,
                                           #axes_names=axes_names,
                                           #axes_domains=axes_domains,
                                           #value_label="multX-CondX")



        pyhrf.verbose(3,'computeContrastsFlag: %s' \
                          %str(self.computeContrastsFlag))

        if self.dataInput.simulData is not None:
            #trueNrls = self.dataInput.simulData.nrls.data
            trueNrls = self.trueValue
            if trueNrls.shape == self.finalValue.shape:
                axes_names = ['condition', 'voxel']
                ad = {'condition':cn}
                relErrorNrls = abs(trueNrls - self.finalValue)
                outputs['nrl_pm_error'] = xndarray(relErrorNrls,
                                                  axes_names=axes_names,
                                                  axes_domains=ad)

                axes_names = ['condition', 'voxel']
                ad = {'condition':cn}
                marked_labels = mark_wrong_labels(self.finalLabels,
                                                  self.trueLabels)
                outputs['pm_Labels_marked'] = xndarray(marked_labels,
                                                     axes_names=axes_names,
                                                     axes_domains=ad)

                n = (trueNrls.astype(np.float32) - \
                                self.finalValue.astype(np.float32))**2
                outputs['nrl_pm_rmse'] = xndarray(n.mean(1),
                                                  axes_names=['condition'],
                                                  axes_domains=ad)

                # Computing Nrl RMSE in Activated Voxels only, by multipying estimated NRLs with estimated Labels
                #nl = n * l
                #outputs['nrl_labels_pm_rmse'] = xndarray(nl.mean(1),
                                                  #axes_names=['condition'],
                                                  #axes_domains=ad)



        if self.computeContrastsFlag:

            pyhrf.verbose(3,'self.outputConVars:')
            pyhrf.verbose.printNdarray(3, self.outputConVars)

            cons = np.array(self.contrasts.values())
            con_names = self.contrasts.keys()
            con_doms = axes_domains={'contrast':con_names}
            outputs['nrl_contrasts'] = xndarray(cons,
                                              axes_names=['contrast','voxel'],
                                              axes_domains=con_doms,
                                              value_label='contrast')

            con_vars = np.array([self.contrastsVar[c] for c in con_names])
            outputs['nrl_contrasts_var'] = xndarray(con_vars,
                                                  axes_names=['contrast','voxel'],
                                                  axes_domains=con_doms,
                                                  value_label='contrast_var')

            outputs['nrl_ncontrasts'] = xndarray(cons/con_vars**.5,
                                               axes_names=['contrast','voxel'],
                                               axes_domains=con_doms,
                                               value_label='contrast')

            # axes_names = ['voxel']
            # for con_name, con_val in self.contrasts.iteritems():
            #     outputName = 'nrl_con_' + con_name
            #     print 'contrastes:', outputName
            #     outputs[outputName] = xndarray(con_val, axes_names=axes_names,
            #                                  value_label="contrast")
            #     if self.outputConVars:
            #         con_var = self.contrastsVar[con_name]
            #         outputName = 'nrl_con_var_' + con_name
            #         outputs[outputName] = xndarray(con_var, axes_names=axes_names,
            #                                      value_label="contrastVar")


        if self.computeContrastsFlag and self.wip_variance_computation:

            axes_names = ['class','condition','voxel']
            axes_domains = {'condition' : cn,'class': self.CLASS_NAMES}
            outputs['nrl_mean_mean_apost'] = xndarray(self.mean_mean_apost,
                                                     axes_names=axes_names,
                                                     axes_domains=axes_domains)

            outputs['nrl_mean_var_apost'] = xndarray(self.mean_var_apost,
                                                    axes_names=axes_names,
                                                    axes_domains=axes_domains)


            #Work after simulation step
            #AII1/ Contraste = CL of conditions - independant samples
            #Variances masked for contrast
            outputName = 'pm_var_con_2cond_indep_masked'#+cond+'-'+cond2
            var = self.varCon_2cond_indep_masked
            outputs[outputName] = xndarray(var, axes_names=['cases', 'voxel'])

            outputName = 'pm_var_con_2cond_indep_apost'#+cond+'-'+cond2
            var = self.varCon_2cond_indep_apost
            outputs[outputName] = xndarray(var, axes_names=['cases', 'voxel'])

            #B II/Contraste = CL of conditions - correlated samples
            outputName = 'pm_var_con_2cond_corr_masked'#+cond+'-'+cond2
            var = self.varCon_2cond_corr_masked
            outputs[outputName] = xndarray(var, axes_names=['cases', 'voxel'])


            outputName = 'pm_var_con_2cond_corr_non_masked'#+cond+'-'+cond2
            var = self.varCon_2cond_corr_non_masked
            outputs[outputName] = xndarray(var, axes_names=['voxel'])

            #if 0:
                #outputName = 'pm_nrl_con_2cond_corr_non_masked'#+cond+'-'+cond2
                #var = self.varCon_2cond_corr_non_masked
                #outputs[outputName] = xndarray(var, axes_names=['voxel'])


            outputName = 'pm_var_con_2cond_corr_apost'#+cond+'-'+cond2
            var = self.varCon_2cond_corr_apost
            outputs[outputName] = xndarray(var, axes_names=['cases', 'voxel'])


            if self.wip_variance_computation \
                    and ('calculaudio' in self.dataInput.cNames):
                #for cond in self.contrasts.iterkeys():
                    #for cond2 in self.contrasts[cond].iterkeys():
                        #if self.outputCons:
                            ##print 'outputCons ...'
                            #outputName = 'nrl_con_'+cond+'-'+cond2
                            #con = self.contrasts[cond][cond2]
                            #outputs[outputName] = xndarray(con,
                                                         #axes_names=axes_names,
                                                         #value_label="contrast")
                            #outputName = 'nrl_ncon_'+cond+'-'+cond2
                            #con = self.contrasts[cond][cond2]
                            #conVar = self.contrastsVar[cond][cond2]
                            #outputs[outputName] = xndarray(con/conVar**.5,
                                                         #axes_names=axes_names,
                                                         #value_label="contrast")
                        #if self.outputConVars:
                            ##print 'outputConVars ...'
                            #outName = 'nrl_convar_'+cond+'_'+cond2
                            #conVar = self.contrastsVar[cond][cond2]
                            #outputs[outName] = xndarray(conVar,
                                                      #axes_names=axes_names,
                                                      #value_label="contrastVar")

                #Save variances of contrasts
                #print "Contrasts"
                outputName = 'pm_nrl_contrast_Lc-Rc'#+cond+'-'+cond2
                con = self.contrast_Lc_Rc
                outputs[outputName] = xndarray(con,
                                                axes_names=['voxel'],
                                                value_label="contrast")

                outputName = 'pm_nrl_contrast_Lc-Rc_variance'#+cond+'-'+cond2
                var = self.contrast_var_Lc_Rc
                outputs[outputName] = xndarray(var,
                                        axes_names=['voxel'],
                                        value_label="contrastVar")


                outputName = 'pm_nrl_contrast_V-A'#+cond+'-'+cond2
                con = self.contrast_V_A
                outputs[outputName] = xndarray(con,
                                                axes_names=['voxel'],
                                                value_label="contrast")

                outputName = 'pm_nrl_contrast_V-A_variance'#+cond+'-'+cond2
                var = self.contrast_var_V_A
                outputs[outputName] = xndarray(var,
                                        axes_names=['voxel'],
                                        value_label="contrastVar")


                outputName = 'pm_nrl_contrast_C-S'#+cond+'-'+cond2
                con = self.contrast_C_S
                outputs[outputName] = xndarray(con,
                                                axes_names=['voxel'],
                                                value_label="contrast")

                outputName = 'pm_nrl_contrast_C-S_variance'#+cond+'-'+cond2
                var = self.contrast_var_C_S
                outputs[outputName] = xndarray(var,
                                        axes_names=['voxel'],
                                        value_label="contrastVar")

                outputName = 'pm_nrl_contrast_C-S_A'#+cond+'-'+cond2
                con = self.contrast_C_S_A
                outputs[outputName] = xndarray(con,
                                                axes_names=['voxel'],
                                                value_label="contrast")

                outputName = 'pm_nrl_contrast_C-S_A_variance'#+cond+'-'+cond2
                var = self.contrast_var_C_S_A
                outputs[outputName] = xndarray(var,
                                        axes_names=['voxel'],
                                        value_label="contrastVar")






            #Variance masked --> for both classes
            axes_names = ['class','condition', 'voxel']
            ad = {'condition' : cn, 'class' : self.CLASS_NAMES}
            outputs['pm_VarContrast_both_classes'] = xndarray(self.varcontrast_cond_both_classes,
                                                axes_names=axes_names,axes_domains=ad)

            #Variance masked --> final values with only values corresponding to the final class for each voxel
            axes_names = ['condition', 'voxel']
            ad = {'condition' : cn}
            outputs['pm_finalVariances'] = xndarray(self.finalVariances,
            axes_names=axes_names,axes_domains=ad)

            #Variance mean a post --> final values with only values corresponding to the final class for each voxel
            axes_names = ['condition', 'voxel']
            ad = {'condition' : cn}
            outputs['pm_final_mean_var_a_post'] = xndarray(self.final_mean_var_a_post,
            axes_names=axes_names,axes_domains=ad)



            ##Variance masked for inactiv class
            #axes_names = ['condition', 'voxel']
            #ad = {'condition':cn}
            #outputs['pm_VarContrast_class_inactiv'] = xndarray(self.varcontrast_cond_both_classes,
                                                            #axes_names=axes_names,axes_domains=ad)




        return outputs



class NRLSamplerWithRelVar(NRLSampler):

    def createWAxh(self,aXh, w):
        np.multiply(w, aXh, self.WaXh)

    def computeWA(self, a, w, wa):
        for j in np.arange(self.nbConditions):
            wa[j,:] = w[j] * a[j,:]

    def computeSumWAxh(self, wa, varXh):
        self.sumWaXh = np.dot(varXh, wa)

    def subtractYtildeWithRelVar(self):
        np.subtract(self.dataInput.varMBY, self.sumWaXh, self.varYtilde)

    def computeVarYTildeOptWithRelVar(self, varXh, w):
        if 0:
            # yTilde_j = y_j - sum_m(a_j^m w^m X^m h)
            pyhrf.verbose(5,'computeVarYTildeOpt...')
            pyhrf.verbose(5,'varXh:' +str(varXh.shape))

            wa = np.zeros((self.nbConditions, self.nbVox))
            self.computeWA(self.currentValue, w, wa)
            self.computeSumWAxh(wa, varXh)

            pyhrf.verbose(5,'sumWaXh %s' %str(self.sumWaXh.shape))
            pyhrf.verbose.printNdarray(6, self.sumWaXh)

            #np.subtract(self.dataInput.varMBY, self.sumaXh, self.varYtilde)
            self.subtractYtildeWithRelVar()
        else:
            wa = np.zeros((self.nbConditions, self.nbVox), dtype=float)

            computeYtildeWithRelVar(varXh,
                      self.currentValue,
                      self.dataInput.varMBY,
                      self.varYtilde,
                      self.sumWaXh,
                      w.astype(np.int32),
                      wa.astype(np.float64))

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde.shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde)

    def computeComponentsApostWithRelVar(self, variables, j, gTQg, w):

        sIMixtP = self.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = self.get_variable('noise_var')
        varXh = self.get_variable('hrf').varXh
        nrls = self.currentValue

        if(w):
            # If wj = 1 The condition is relevant, we compute the posterior components as we did
            # without introducing the relevant variable w

            gTQgjrb = gTQg[j]/rb

            if pyhrf.verbose > 4:
                print 'Current components:'
                print 'mean CI = %f, var CI = %f' %(mean[self.L_CI,j], var[self.L_CI,j])
                print 'mean CA = %f, var CA = %f' %(mean[self.L_CA,j], var[self.L_CA,j])
                print 'gTQg =', gTQg[j]

            pyhrf.verbose(6, 'gTQg[%d] %s:'%(j,str(gTQg[j].shape)))
            pyhrf.verbose.printNdarray(6, gTQg[j])

            pyhrf.verbose(6, 'rb %s :'%str(rb.shape))
            pyhrf.verbose.printNdarray(6, rb)

            pyhrf.verbose(6, 'gTQgjrb %s :'%str(gTQgjrb.shape))
            pyhrf.verbose.printNdarray(6, gTQgjrb)

            ej = self.varYtilde + nrls[j,:] \
                * repmat(varXh[:,j],self.nbVox, 1).transpose()

            pyhrf.verbose(6, 'varYtilde %s :'%str((self.varYtilde.shape)))
            pyhrf.verbose.printNdarray(6, self.varYtilde)

            pyhrf.verbose(6, 'nrls[%d,:] %s :'%(j,nrls[j,:]))
            pyhrf.verbose.printNdarray(6, nrls[j,:])

            pyhrf.verbose(6, 'varXh[:,%d] %s :'%(j,str(varXh[:,j].shape)))
            pyhrf.verbose.printNdarray(6, varXh[:,j])

            pyhrf.verbose(6, 'repmat(varXh[:,%d],self.nbVox, 1).transpose()%s:' \
                          %(j,str((repmat(varXh[:,j],self.nbVox, 1).transpose().shape))))
            pyhrf.verbose.printNdarray(6, repmat(varXh[:,j],self.nbVox, 1).transpose())

            pyhrf.verbose(6, 'ej %s :'%str((ej.shape)))
            pyhrf.verbose.printNdarray(6, ej)

            np.divide(np.dot(self.varXhtQ[j,:],ej), rb, self.varXjhtQjeji)

            if pyhrf.verbose.verbosity > 5:
                pyhrf.verbose(5, 'np.dot(self.varXhtQ[j,:],ej) %s :' \
                              %str(np.dot(self.varXhtQ[j,:],ej).shape))
                pyhrf.verbose.printNdarray(5, np.dot(self.varXhtQ[j,:],ej))

                pyhrf.verbose(5, 'self.varXjhtQjeji %s :' \
                              %str(self.varXjhtQjeji.shape))
                pyhrf.verbose.printNdarray(5, self.varXjhtQjeji)

            for c in xrange(self.nbClasses):
                self.varClassApost[c,j,:] = 1./(1./var[c,j] + gTQgjrb)
                np.sqrt(self.varClassApost[c,j,:], self.sigClassApost[c,j,:])
                if c > 0: # assume 0 stands for inactivating class
                    np.multiply(self.varClassApost[c,j,:],
                               add(mean[c,j]/var[c,j], self.varXjhtQjeji),
                               self.meanClassApost[c,j,:])
                else:
                    np.multiply(self.varClassApost[c,j,:], self.varXjhtQjeji,
                         self.meanClassApost[c,j,:])

                pyhrf.verbose(5, 'meanClassApost %d cond %d :'%(c,j))
                pyhrf.verbose.printNdarray(5, self.meanClassApost[c,j,:])
                pyhrf.verbose(5, 'varClassApost %d cond %d :'%(c,j))
                pyhrf.verbose.printNdarray(5, self.varClassApost[c,j,:])

        else:

            for c in xrange(self.nbClasses):
                self.varClassApost[c,j,:] = var[0,j]
                np.sqrt(self.varClassApost[c,j,:], self.sigClassApost[c,j,:])
                self.meanClassApost[c,j,:] = mean[0,j]

                pyhrf.verbose(5, 'meanClassApost %d cond %d :'%(c,j))
                pyhrf.verbose.printNdarray(5, self.meanClassApost[c,j,:])
                pyhrf.verbose(5, 'varClassApost %d cond %d :'%(c,j))
                pyhrf.verbose.printNdarray(5, self.varClassApost[c,j,:])


    def deltaWCorr0 (self, nbVox, moyqvoxj, t1, t2):

        result = np.zeros(self.nbVox, dtype=float)
        for i in xrange(self.nbVox):
            num = np.exp( t1 * ((1/nbVox)*(moyqvoxj[i] + 1) - t2)) + 1
            denom = np.exp( t1 * ((1/nbVox)*(moyqvoxj[i]) - t2)) + 1
            result[i] = num[i]/denum[i]

        return result

    def deltaWCorr1 (self, nbVox, moyqvoxj, t1, t2):

        result = np.zeros(self.nbVox, dtype=float)
        for i in xrange(self.nbVox):
            num = np.exp( - t1 * ((1/nbVox)*(moyqvoxj[i] + 1) - t2)) + 1
            denom = np.exp( - t1 * ((1/nbVox)*(moyqvoxj[i]) - t2)) + 1
            result[i] = num[i]/denum[i]

        return result

    def calcFracLambdaTildeWithRelCond(self, l, nbVox, moyqvoxj, t1, t2):

        dWCorr1 = deltaWCorr1(nbVox, moyqvoxj, t1, t2)
        return l*dWCorr1


    def calcFracLambdaTildeWithIRRelCond(self, cond, c1, c2, variables, nbVox, moyqvoxj, t1, t2):

        sWeightP = self.get_variable('mixt_weights')
        varLambda = sWeightP.currentValue

        if self.samplerEngine.get_variable('beta').currentValue[cond] <= 0:
            ratio = varLambda[c1]/varLambda[c2]
        else:
            ratio = 1

        dWcorr0 = deltaWCorr0(nbVox, moyqvoxj, t1, t2)

        return ratio*dWCorr0

    def computemoyqvox(self, cardClass, nbVox):
        '''
        Compute mean of labels in ROI (without the label of voxel i)
        '''
        moyqvox = np.zeros((self.nbConditions, self.nbVox), dtype=float)
        for i in xrange(self.nbVox):
            moyqvox[:,i] = np.divide( cardClass[L_CA,:] - self.labels[:, i].transpose(),  nbVox)

        return moyqvox

    def samplingWarmUp(self, variables):

        NRLSampler.samplingWarmUp(self, variables)
        self.sumWaXh = np.zeros((self.ny, self.nbVox), dtype=float)

    def sampleLabelsWithRelVar(self, cond, variables):

        # Parameters of Sigmoid function
        t1 = 50
        t2 = 0.25

        moyqvox =  self.computemoyqvox(CardClass, nbVox)
        moyqvoxj = moyqvox[cond,:] # Vecteur of length nbVox

        w = self.samplerEngine.get_variable('W').currentValue[cond]

        if w:
            fracLambdaTilde = self.calcFracLambdaTilde(cond, self.L_CI, self.L_CA,
                                                   variables)

            fracLambdaTildeWithRelVar = self.calcFracLambdaTildeWithRelCond(self.fracLambdaTilde, nbVox, moyqvoxj, t1, t2)

        else :
            fracLambdaTildeWithRelVar = self.calcFracLambdaTildeWithIRRelCond(cond, self.L_CI, self.L_CA,
                                                   variables, nbVox, moyqvoxj, t1, t2)

        beta = self.samplerEngine.get_variable('beta').currentValue[cond]
        if self.samplerEngine.get_variable('beta').currentValue[cond] > 0:

            #corrEnergiesC = np.zeros_like(self.corrEnergies)
            if 1:
                deltaCol = 0.
                #TODO generalize ...
                calcCorrEnergies(cond, self.labels, self.corrEnergies,
                             self.dataInput.neighboursIndexes,
                             deltaCol, self.nbClasses, self.L_CI, self.L_CA)
                fracLambdaTilde *= np.exp(beta * self.corrEnergies[cond,:])
                #print 'self.corrEnergies :'
                #print self.corrEnergies
            else :
                for i in xrange(self.nbVox):
                    deltaE = self.calcDeltaEnergy(i, cond)
                    self.corrEnergies[cond,i] = deltaE
                    fracLambdaTilde[i] *= np.exp(beta * deltaE)

            #assert np.allclose(corrEnergiesC[cond,:], self.corrEnergies[cond,:])

        varLambdaApost = 1./(1.+fracLambdaTildeWithRelVar)

        self.labels[cond,:] = np.array(self.labelsSamples[cond,:]<=varLambdaApost,
                                    dtype=int )
        if pyhrf.verbose > 6:
            for i in xrange(self.nbVox):
                print 'it%04d-cond%02d-Vox%03d ...' %(self.iteration,cond,i)
                print 'mApostCA =', self.meanClassApost[self.L_CA,cond,i],
                print 'mApostCI =', self.meanClassApost[self.L_CI,cond,i]
                print 'sApostCA =', self.sigClassApost[self.L_CA,cond,i],
                print 'sApostCI =', self.sigClassApost[self.L_CI,cond,i]
                print 'rl_I_A =', fracLambdaTilde[i]
                print 'lambda Apost CA =', varLambdaApost[i]
                print 'random =', self.labelsSamples[cond,i]
                print '-> labels = ', self.labels[cond,i]


    def sampleNrlsParallelWithRelVar(self, varXh, rb, h, varLambda, varCI, varCA,
                           meanCA, gTQg, variables, w):
        pyhrf.verbose(3, 'Sampling Nrls (parallel, no spatial prior) ...')

        for j in xrange(self.nbConditions):
            self.computeComponentsApostWithRelVar(variables, j, gTQg, w[j])
            if self.sampleLabelsFlag:
                pyhrf.verbose(3, 'Sampling labels - cond %d ...'%j)

                self.sampleLabelsWithRelVar(j, variables)
                self.countLabels(self.labels, self.voxIdx, self.cardClass)

                pyhrf.verbose(3,'Sampling labels done!')
                pyhrf.verbose(6, 'All labels cond %d:'%j)
                pyhrf.verbose.printNdarray(6, self.labels[j,:])
                if self.trueLabels is not None:
                    pyhrf.verbose(6, 'All true labels cond %d:'%j)
                    pyhrf.verbose.printNdarray(6, self.trueLabels[j,:])

            for c in xrange(self.nbClasses):
                putmask(self.sigApost[j,:], self.labels[j,:]==c,
                        self.sigClassApost[c,j,:])
                putmask(self.meanApost[j,:],self.labels[j,:]==c,
                        self.meanClassApost[c,j,:])

            oldVal = self.currentValue[j,:]
            add(np.multiply(self.nrlsSamples[j,:], self.sigApost[j,:]),
                self.meanApost[j,:], self.currentValue[j,:])

            self.computeVarYTildeOptWithRelVar(varXh, w)
            #self.computeVarYTilde(varXh)

    def sampleNrlsSerialWithRelVar(self, rb, h,
                         gTQg, variables, w, t1, t2):

        pyhrf.verbose(3, 'Sampling Nrls (serial, spatial prior) ...')
        sIMixtP = self.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        varXh = np.array([self.get_variable('hrf').varXh])
        neighbours = self.dataInput.neighboursIndexes
        beta = self.samplerEngine.get_variable('beta').currentValue
        voxOrder = np.random.permutation(self.nbVox)

        if 0:
            for j in xrange(self.nbConditions):
                betaj = beta[j]
                sampleSmmNrlWithRelVar(voxOrder, rb, neighbours, self.varYtilde,
                             self.labels[j,:], varXh[:,:,j],
                             self.currentValue[j,:],
                             self.nrlsSamples[j,:], self.labelsSamples[j,:],
                             np.array([self.varXhtQ[j,:]]), gTQg[j], betaj,
                             mean[:,j], var[:,j], self.nbClasses,
                             self.sampleLabelsFlag+0, self.iteration, j, w[j], self.cardClass[:,j])
                #sys.exit(1)
            self.countLabels(self.labels, self.voxIdx, self.cardClass)

        else:
            cardClassCA = np.zeros(self.nbConditions, dtype=int)
            for i in range(self.nbConditions):
                cardClassCA[i] = self.cardClass[self.L_CA,i]

            sampleSmmNrl2WithRelVar(voxOrder.astype(np.int32), rb.astype(np.float64),
                          neighbours.astype(np.int32),
                          self.varYtilde,
                          self.labels, varXh.astype(np.float64),
                          self.currentValue,
                          self.nrlsSamples.astype(np.float64),
                          self.labelsSamples.astype(np.float64),
                          np.array([self.varXhtQ]).astype(np.float64),
                          gTQg.astype(np.float64),
                          beta.astype(np.float64), mean.astype(np.float64),
                          var.astype(np.float64), self.meanClassApost,
                          self.varClassApost, w.astype(np.int32), t1, t2,
                          cardClassCA.astype(np.int32), #self.cardClass,
                          self.nbClasses, self.sampleLabelsFlag+0, self.iteration,
                          self.nbConditions)

            if (self.varClassApost<=0).any():
                raise Exception('Negative posterior variances!')

            self.countLabels(self.labels, self.voxIdx, self.cardClass)


    def sampleNextInternal(self, variables):
        #TODO : comment
        #print 'iteration :', self.iteration
        sIMixtP = self.get_variable('mixt_params')
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI] # Varaince of in-activated class for all conditions
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA] # Varaince of activated class for all conditions
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA] # Mean of activated class for all conditions
        rb = self.get_variable('noise_var').currentValue
        sHrf = self.get_variable('hrf')
        varXh = sHrf.varXh
        h = sHrf.currentValue
        sw = self.get_variable('w')
        w = sw.currentValue
        t1 = sw.t1
        t2 = sw.t2

        self.nh = np.size(h)
        varLambda = self.get_variable('mixt_weights').currentValue

        pyhrf.verbose(5,'varXh %s :' %str(varXh.shape))
        pyhrf.verbose.printNdarray(5, varXh)

        self.computeVarYTildeOptWithRelVar(varXh, w)

        self.computeVarXhtQ(h, self.dataInput.matXQ)

        pyhrf.verbose(6,'varXhtQ %s :' %str(self.varXhtQ.shape))
        pyhrf.verbose.printNdarray(5, self.varXhtQ)

        self.labelsSamples = np.random.rand(self.nbConditions, self.nbVox)
        self.nrlsSamples = np.random.randn(self.nbConditions, self.nbVox)

        gTQg = np.diag(np.dot(self.varXhtQ,varXh))

        if self.samplerEngine.get_variable('beta').currentValue[0] < 0:
            self.sampleNrlsParallelWithRelVar(varXh, rb, h, varLambda, varCI,
                                    varCA, meanCA, gTQg, variables, w)
        else:
            self.sampleNrlsSerialWithRelVar(rb, h, gTQg, variables, w, t1, t2)
            self.computeVarYTildeOptWithRelVar(varXh, w)

        if (self.currentValue >= 1000).any() and pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )
            #pyhrf.verbose.set_verbosity(6)

        if pyhrf.verbose.verbosity >= 4:
            self.reportDetection()

        self.computeAA(self.currentValue, self.aa)

        wa = np.zeros((self.nbConditions, self.nbVox))
        self.computeWA(self.currentValue, w, wa)
        self.computeSumWAxh(wa, varXh)

        self.printState(4)
        print 'iteration  ',self.iteration
        self.iteration += 1 #TODO : factorize !!

class BiGaussMixtureParamsSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    #TODO : comment

    """

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

        if self.dataInput.simulData is not None and \
            isinstance(self.dataInput.simulData, list):
            if isinstance(self.dataInput.simulData[0], dict) and \
              self.dataInput.simulData[0].has_key('condition_defs'):

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
                    nbc = self.nbConditions
                    m_act = [np.where(sd[0]['labels'][j,:] == self.L_CA) \
                             for j in xrange(nbc) ]
                    m_inact = [np.where(sd[0]['labels'][j,:] == self.L_CI)\
                               for j in xrange(nbc) ]
                    all_nrls = np.array([ssd['nrls'] for ssd in sd])

                    mean_act = np.array([all_nrls[:,j,m_act[j][0]].mean() \
                                         for j in xrange(nbc)])
                    var_act = np.array([all_nrls[:,j,m_act[j][0]].var() \
                                         for j in xrange(nbc)])
                    var_inact = np.array([all_nrls[:,j,m_inact[j][0]].var() \
                                         for j in xrange(nbc)])
                    #raise Exception()
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
                self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30. #np.array([2.5,-3.4,0.,0.])
                self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.0
                self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.0
                #self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 2. #np.array([2.5,-3.4,0.,0.])
                #self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 0.5
                #self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 0.5
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

        #print 'v0 =',varCIj,',  v1 =',varCAj,',     m1 =',meanCAj

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
        sScale = self.samplerEngine.get_variable('scale')

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



#class NRL_Multi_Sess_NRLsBar_Sampler(xmlio.XMLParamDrivenClass, GibbsSamplerVariable):

    ## parameters specifications :
    #P_SAMPLE_FLAG = 'sampleFlag'
    #P_VAL_INI = 'initialValue'
    #P_USE_TRUE_NRLS = 'useTrueNrls'
    #P_TrueNrlFilename = 'TrueNrlFilename'

    #P_OUTPUT_NRL = 'writeResponsesOutput'

    ## parameters definitions and default values :
    #defaultParameters = {
        #P_SAMPLE_FLAG : True,
        #P_VAL_INI : None,
        #P_USE_TRUE_NRLS : False, #False,
        #P_OUTPUT_NRL : True,
        #P_TrueNrlFilename : None, #'./nrls.nii',
        #}

    #if pyhrf.__usemode__ == pyhrf.DEVEL:
        #parametersToShow = [P_SAMPLE_FLAG, P_VAL_INI, P_USE_TRUE_NRLS,                        P_TrueNrlFilename,
                            #P_OUTPUT_NRL]

    #elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        #parametersToShow = [P_OUTPUT_NRL]

    #parametersComments = {
        ## P_CONTRASTS : 'Define contrasts as a string with the following format:'\
        ##     '\n condition1-condition2;condition1-condition3\n' \
        ##     'Must be consistent with condition names specified in session data' \
        ##     'above',
        #P_TrueNrlFilename :'Define the filename of simulated NRLs.\n'\
            #'It is taken into account when NRLs is not sampled.',
        #}

    #def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 #xmlLabel=None, xmlComment=None):

        ##TODO : comment
        #xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           #xmlLabel, xmlComment)
        #sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        #valIni = self.parameters[self.P_VAL_INI]
        #useTrueVal = self.parameters[self.P_USE_TRUE_NRLS]
        #self.TrueNrlsFilename = self.parameters[self.P_TrueNrlFilename]
        #an = ['condition', 'voxel']
        #GibbsSamplerVariable.__init__(self,'nrl', valIni=valIni,
                                      #sampleFlag=sampleFlag,
                                      #useTrueValue=useTrueVal,
                                      #axes_names=an,
                                      #value_label='PM NRL')

        #self.outputNrls = self.parameters[self.P_OUTPUT_NRL]


    #def computeComponentsApost(self, variables, j, gTQg):
        #sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        #var = sIMixtP.getCurrentVars()
        ##var_nrlSess = #TODO
        #mean = sIMixtP.getCurrentMeans()
        #rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        #varXh = self.get_variable('hrf').varXh
        #nrls = self.currentValue

        #gTQgjrb = gTQg[j]/rb

        #if pyhrf.verbose > 4:
            #print 'Current components:'
            #print 'mean CI = %f, var CI = %f' %(mean[self.L_CI,j], var[self.L_CI,j])
            #print 'mean CA = %f, var CA = %f' %(mean[self.L_CA,j], var[self.L_CA,j])
            #print 'gTQg =', gTQg[j]

        #pyhrf.verbose(6, 'gTQg[%d] %s:'%(j,str(gTQg[j].shape)))
        #pyhrf.verbose.printNdarray(6, gTQg[j])

        #pyhrf.verbose(6, 'rb %s :'%str(rb.shape))
        #pyhrf.verbose.printNdarray(6, rb)

        #pyhrf.verbose(6, 'gTQgjrb %s :'%str(gTQgjrb.shape))
        #pyhrf.verbose.printNdarray(6, gTQgjrb)

        #ej = self.varYtilde + nrls[j,:] \
             #* repmat(varXh[:,j],self.nbVox, 1).transpose()

        #pyhrf.verbose(6, 'varYtilde %s :'%str((self.varYtilde.shape)))
        #pyhrf.verbose.printNdarray(6, self.varYtilde)

        #pyhrf.verbose(6, 'nrls[%d,:] %s :'%(j,nrls[j,:]))
        #pyhrf.verbose.printNdarray(6, nrls[j,:])

        #pyhrf.verbose(6, 'varXh[:,%d] %s :'%(j,str(varXh[:,j].shape)))
        #pyhrf.verbose.printNdarray(6, varXh[:,j])

        #pyhrf.verbose(6, 'repmat(varXh[:,%d],self.nbVox, 1).transpose()%s:' \
                          #%(j,str((repmat(varXh[:,j],self.nbVox, 1).transpose().shape))))
        #pyhrf.verbose.printNdarray(6, repmat(varXh[:,j],self.nbVox, 1).transpose())

        #pyhrf.verbose(6, 'ej %s :'%str((ej.shape)))
        #pyhrf.verbose.printNdarray(6, ej)

        #np.divide(np.dot(self.varXhtQ[j,:],ej), rb, self.varXjhtQjeji)

        #if pyhrf.verbose.verbosity > 5:
            #pyhrf.verbose(5, 'np.dot(self.varXhtQ[j,:],ej) %s :' \
                              #%str(np.dot(self.varXhtQ[j,:],ej).shape))
            #pyhrf.verbose.printNdarray(5, np.dot(self.varXhtQ[j,:],ej))

            #pyhrf.verbose(5, 'self.varXjhtQjeji %s :' \
                              #%str(self.varXjhtQjeji.shape))
            #pyhrf.verbose.printNdarray(5, self.varXjhtQjeji)

        #for c in xrange(self.nbClasses):
            ##print 'var[%d,%d] :' %(c,j), var[c,j]
            ##print 'mean[%d,%d] :' %(c,j), mean[c,j]
            #self.varClassApost[c,j,:] = 1./(1./var[c,j] + 1/var_nrlSess)
            #if 0:
                #print 'shape of self.varClassApost[c,j,:] :', \
                    #self.varClassApost.shape
            ##print 'varClassApost[%d,%d,:]:' %(c,j), self.varClassApost[c,j,:]
            #np.sqrt(self.varClassApost[c,j,:], self.sigClassApost[c,j,:])
            #if c > 0: # assume 0 stands for inactivating class
                #np.multiply(self.varClassApost[c,j,:],
                               #add(mean[c,j]/var[c,j], nrls[:,j,c].sum()/var_nrlSess),
                               #self.meanClassApost[c,j,:])
                               ##nrls[:,j,c].sum() = sum on sessions of aj,m,s
            #else:
                #np.multiply(self.varClassApost[c,j,:], nrls[:,j,c].sum()/var_nrlSess,
                                #self.meanClassApost[c,j,:])

            #pyhrf.verbose(5, 'meanClassApost %d cond %d :'%(c,j))
            #pyhrf.verbose.printNdarray(5, self.meanClassApost[c,j,:])
            #pyhrf.verbose(5, 'varClassApost %d cond %d :'%(c,j))
            #pyhrf.verbose.printNdarray(5, self.varClassApost[c,j,:])
            #pyhrf.verbose(5, 'shape of self.varClassApost[c,j,:] : %s' \
                              #%str(self.varClassApost.shape))






class NRL_Multi_Sess_Sampler(GibbsSamplerVariable):
# parameters specifications :
    P_SAMPLE_FLAG = 'sampleFlag'
    P_VAL_INI = 'initialValue'
    P_USE_TRUE_NRLS = 'useTrueNrls'
    P_TrueNrlFilename = 'TrueNrlFilename'
    P_OUTPUT_NRL = 'writeResponsesOutput'

    # parameters definitions and default values :
    defaultParameters = {
        P_SAMPLE_FLAG : True,
        P_VAL_INI : None,
        P_USE_TRUE_NRLS : False, #False,
        P_OUTPUT_NRL : True,
        P_TrueNrlFilename : None, #'./nrls.nii',
        }

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        parametersToShow = [P_SAMPLE_FLAG, P_VAL_INI, P_USE_TRUE_NRLS,
                            P_TrueNrlFilename, P_OUTPUT_NRL
                            ]

    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = [P_OUTPUT_NRL]

    parametersComments = {
        P_TrueNrlFilename :'Define the filename of simulated NRLs.\n'\
            'It is taken into account when NRLs is not sampled.',
        }

    def __init__(self, parameters=None, xmlHandler=None,
                 xmlLabel=None, xmlComment=None):

        #TODO : comment
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]
        useTrueVal = self.parameters[self.P_USE_TRUE_NRLS]
        self.TrueNrlsFilename = self.parameters[self.P_TrueNrlFilename]
        an = ['session', 'condition', 'voxel']
        GibbsSamplerVariable.__init__(self,'nrl_by_session', valIni=valIni,
                                      sampleFlag=sampleFlag,
                                      useTrueValue=useTrueVal,
                                      axes_names=an,
                                      value_label='PM NRL')


        self.outputNrls = self.parameters[self.P_OUTPUT_NRL]

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbSessions = self.dataInput.nbSessions


        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, dict):
                if dataInput.simulData.has_key('nrls'):
                    nrls = dataInput.simulData['nrls']
                    if isinstance(nrls, xndarray):
                        self.trueValue = nrls.reorient(['condition','voxel']).data
                    else:
                        self.trueValue = nrls
            elif isinstance(dataInput.simulData, list):
                v = np.array([sd['nrls_session'].astype(np.float64)\
                              for sd in dataInput.simulData])
                self.trueValue = v
            else:
                if hasattr(dataInput.simulData[0], 'nrls_session'):
                    self.trueValue = np.array([dataInput.simulData[s]['nrls_session'].data.astype(np.float64)\
                                            for s in xrange(self.nbSessions)])


        else:
            self.trueValue = None

    def checkAndSetInitValue(self, variables):
        pyhrf.verbose(3, 'NRL_Multi_Sess_Sampler.checkAndSetInitNRLs ...')
        smplNrlBar = self.get_variable('nrl_bar')
        smplNrlBar.checkAndSetInitValue(variables)

        smplDrift = self.get_variable('drift')
        smplDrift.checkAndSetInitValue(variables)

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
                nrl_bar = self.samplerEngine.get_variable('nrl').currentValue
                var_sess = self.samplerEngine.get_variable('variance_nrls_by_session').currentValue
                labels = self.samplerEngine.get_variable('nrl').labels
                for m in xrange(self.nbConditions):
                    Ac_pos = np.where(labels[m])
                    for s in xrange(self.nbSessions):
                        Nrls_sess = np.random.randn((self.nbVox))*var_sess**0.5 #+ nrl_bar[s,m]
                        Nrls_sess[Ac_pos[0]] = np.random.randn((Ac_pos[0].size))*var_sess**0.5 + 30
                        self.currentValue[s,m] = Nrls_sess.astype(np.float64)
            #self.currentValue[s]
            self.currentValue = np.zeros((self.nbSessions, self.nbConditions, self.nbVox),
            dtype=np.float64)+20


    def saveCurrentValue(self, it):
        GibbsSamplerVariable.saveCurrentValue(self, it)

    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """

        # Precalculations and allocations :
        smplHRF = self.samplerEngine.get_variable('hrf')
        imm=[]
        aXh=[]
        sumaXh=[]
        computeVarYtildeOpt=[]
        #self.egsurrb = np.empty(( self.nbConditions, self.nbVox), dtype=float)
        #self.varYtilde = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        #self.varYbar = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        #self.sumaXh = np.zeros((self.nbSessions, self.ny, self.nbVox), dtype=np.float64)
        self.aa = np.zeros((self.nbSessions, self.nbConditions, self.nbConditions, self.nbVox), dtype=float)
        self.meanApost = np.zeros((self.nbSessions, self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = np.zeros((self.nbSessions,self.nbConditions, self.nbVox), dtype=float)

        for s in xrange(self.nbSessions):
            self.imm = self.samplerEngine.get_variable('beta').currentValue[0] < 0
            imm.append(self.imm)

            self.computeVarYTildeSessionOpt(smplHRF.varXh[s], s)
            self.aXh = np.empty((self.nbVox, self.ny, self.nbConditions), dtype=float)
            aXh.append(self.aXh)

        self.computeAA(self.currentValue, self.aa)
        imm = np.array(self.imm)
        aXh = np.array(self.aXh)
        self.iteration = 0
        pyhrf.verbose(5,'varYtilde at end of warm up %s' \
          %str(self.varYtilde.shape))

    def computeAA(self, nrls, destaa):
        for s in xrange(self.nbSessions):
            for j in xrange(self.nbConditions):
                for k in xrange(self.nbConditions):
                    np.multiply(nrls[s,j,:], nrls[s,k,:],
                                destaa[s,j,k])


    def computeVarYTildeSessionOpt(self, varXh, s):
        #print 'shapes:', varXh.shape, self.currentValue[s].shape, self.dataInput.varMBY[s].shape, self.varYtilde[s].shape, self.sumaXh[s].shape
        computeYtilde(varXh, self.currentValue[s], self.dataInput.varMBY[s],
                      self.varYtilde[s], self.sumaXh[s])

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde[s].shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde[s])
        matPl = self.samplerEngine.get_variable('drift').matPl
        self.varYbar[s] = self.varYtilde[s] - matPl[s]


    def sampleNextAlt(self, variables):
        #used in case of trueValue choice !
        varXh = self.get_variable('hrf').varXh
        for s in xrange(self.nbSessions):
            self.computeVarYTildeSessionOpt(varXh[s], s)


    def computeComponentsApost(self, variables, m, varXh, s):
        self.var_a = self.samplerEngine.get_variable('variance_nrls_by_session').currentValue
        rb    = self.samplerEngine.get_variable('noise_var').currentValue
        nrls = self.currentValue
        nrl_bar = self.samplerEngine.get_variable('nrl').currentValue
        pyhrf.verbose(6, 'rb %s :'%str(rb.shape))
        pyhrf.verbose.printNdarray(6, rb)
        pyhrf.verbose(6, 'var_a %f :'%self.var_a[0])

        gTg = np.diag(np.dot(varXh[s].transpose(),varXh[s]))

        ejsm = self.varYbar[s] + nrls[s,m,:] \
                            * repmat(varXh[s][:,m],self.nbVox, 1).transpose()

        #pyhrf.verbose(6, 'varYtilde %s :'%str((self.varYtilde.shape)))
        #pyhrf.verbose.printNdarray(6, self.varYtilde)

        #pyhrf.verbose(6, 'nrls[%d,:] %s :'%(j,nrls[j,:]))
        #pyhrf.verbose.printNdarray(6, nrls[j,:])

        #pyhrf.verbose(6, 'varXh[:,%d] %s :'%(j,str(varXh[:,j].shape)))
        #pyhrf.verbose.printNdarray(6, varXh[:,j])

        #pyhrf.verbose(6, 'repmat(varXh[:,%d],self.nbVox, 1).transpose()%s:' \
        #%(j,str((repmat(varXh[:,j],self.nbVox, 1).transpose().shape))))
        #pyhrf.verbose.printNdarray(6, repmat(varXh[:,j],self.nbVox, 1).transpose())

        self.egsurrb = np.divide(np.dot(ejsm.transpose(), varXh[s][:,m]), rb[s,:])

        #print 'varYbar:', self.varYbar[s][150]
        #print 'nrls*g:', nrls[s,m,:] * repmat(varXh[s][:,m],self.nbVox, 1).transpose()
        #print 'ejsm:', ejsm[:,150]
        #print 'g:', varXh[s][:,m]
        #print 'ejsm*g:', np.dot(ejsm.transpose(), varXh[s][:,m])[150]
        #print 'nrlbar:', nrl_bar[m,150]
        #print 'varXh:', varXh[s][150,m]
        #print 'egsurrb', self.egsurrb[150]
        self.sigApost[s,m,:] = np.sqrt(1./(1./self.var_a + gTg[m]*1./rb[s,:]))
        np.multiply(self.sigApost[s,m,:]**2,
                           np.add(nrl_bar[m,:]/self.var_a, self.egsurrb),
                           self.meanApost[s,m,:])

        pyhrf.verbose(6, "sigApost[s=%d,m=%d,:2]" %(s,m))
        pyhrf.verbose.printNdarray(6, self.sigApost[s,m,:2])
        pyhrf.verbose(6, "nrl_bar[m=%d,:2]/var_a" %(m))
        pyhrf.verbose.printNdarray(6, (nrl_bar[m,:]/self.var_a)[:2])
        pyhrf.verbose(6, "ejsm[:2]")
        pyhrf.verbose.printNdarray(6, ejsm[:,:2])
        pyhrf.verbose(6, "ejsm.(Xh)t[:2]")
        pyhrf.verbose.printNdarray(6,np.dot(ejsm.transpose(), varXh[s][:,m])[:2])
        pyhrf.verbose(6, "egsurrb[:2]")
        pyhrf.verbose.printNdarray(6, self.egsurrb[:2])
        pyhrf.verbose(6, "meanApost[s=%d,m=%d,:2]" %(s,m))
        pyhrf.verbose.printNdarray(6, self.meanApost[s,m,:2])


    def sampleNextInternal(self, variables):
        pyhrf.verbose(3, 'NRL_Multi_Sess_Sampler.sampleNextInternal ...')
        varXh = self.samplerEngine.get_variable('hrf').varXh

        for s in xrange(self.nbSessions):
            self.computeVarYTildeSessionOpt(varXh[s], s)
            for m in xrange(self.nbConditions):
                self.computeComponentsApost(variables, m, varXh, s)
                for j in xrange(self.nbVox):
                    self.currentValue[s][m,j] = np.random.normal(self.meanApost[s,m,j], self.sigApost[s,m,j])
                self.computeVarYTildeSessionOpt(varXh[s], s)
        #print '""""', self.currentValue
        #print 'mean apost:', self.meanApost[s,m,j]
        #print 'sig apost:', self.sigApost[s,m,j]
        #print 'ééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééé'
        if (self.currentValue >= 1000).any() and pyhrf.__usemode__ == pyhrf.DEVEL:
            pyhrf.verbose(2, "Weird NRL values detected ! %d/%d" \
                              %((self.currentValue >= 1000).sum(),
                                self.nbVox*self.nbConditions) )


        self.computeAA(self.currentValue, self.aa)

        self.iteration += 1 #TODO : factorize !!


    def cleanMemory(self):

        # clean memory of temporary variables :

        del self.sigApost
        del self.meanApost
        del self.aa
        del self.aXh
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

        GibbsSamplerVariable.finalizeSampling(self)

        smplHRF = self.samplerEngine.get_variable('hrf')

        # Correct sign ambiguity :
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


    def getOutputs(self):

        #outputs = GibbsSamplerVariable.getOutputs(self)
        cn = self.dataInput.cNames
        sn = self.dataInput.sNames

        outputs = {}
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
        if 1 and hasattr(self, 'error'):
            err = self.error**.5
        else:
            err = None

        c = xndarray(self.finalValue,
                   axes_names=self.axes_names,
                   axes_domains=self.axes_domains,
                   value_label=self.value_label)

        outputs[self.name+'_pm'] = c


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

        if 0:
            axes_names = ['type','session', 'time', 'voxel']
            outputs['ysignals'] = xndarray(np.array([self.dataInput.varMBY,self.varYbar,self.sumaXh]),
                                         axes_names=axes_names,
                                         axes_domains={'type':['Y','Ybar','sumaXh']})

            axes_names = ['session', 'time', 'voxel']
            outputs['ytilde'] = xndarray(self.varYtilde,
                                     axes_names=axes_names,)

            outputs['ybar'] = xndarray(self.varYbar,
                                     axes_names=axes_names,)

            outputs['sumaXh'] = xndarray(self.sumaXh,
                                       axes_names=axes_names,)

            outputs['mby'] = xndarray(np.array(self.dataInput.varMBY),
                                       axes_names=axes_names,)

        return outputs


class Variance_GaussianNRL_Multi_Sess(GibbsSamplerVariable):
    '''
    '''
    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_USE_TRUE_VALUE = 'useTrueValue'

    defaultParameters = {
        P_USE_TRUE_VALUE : False,
        P_VAL_INI : np.array([1.]),
        P_SAMPLE_FLAG : False, #By default, beta>0 -> SMM
        }

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = [P_USE_TRUE_VALUE]


    def __init__(self, parameters=None, xmlHandler=None,
                 xmlLabel=None, xmlComment=None):
        #TODO : comment
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]
        useTrueVal = self.parameters[self.P_USE_TRUE_VALUE]
        GibbsSamplerVariable.__init__(self, 'variance_nrls_by_session', valIni=valIni,
                                            useTrueValue=useTrueVal,
                                            sampleFlag=sampleFlag)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        self.nbSessions = self.dataInput.nbSessions

        if dataInput.simulData is not None:
            #self.trueValue = np.array(np.array([dataInput.simulData[s]['nrls_session'] for s in xrange(self.nbSessions)]).var(0))
            self.trueValue = np.array([dataInput.simulData[0]['var_sess']])

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

        sum_s_j_m=0
        for s in xrange(self.nbSessions):
            for m in xrange(self.nbConditions):
                for j in xrange(self.nbVoxels):
                    sum_s_j_m += (nrls[s][m][j] - nrlsBAR[m][j])**2

        alpha = (self.nbSessions*self.nbConditions*self.nbVoxels-1)/2.
        beta_g  = 0.5*sum_s_j_m
        self.currentValue[0] = 1.0/np.random.gamma(alpha, 1/beta_g)


        #self.currentValue.astype(np.float64)
    #def sampleNextAlt(self, variables):

class BiGaussMixtureParamsSamplerWithRelVar_OLD(BiGaussMixtureParamsSampler):


    def computeWithProperPriorsWithRelVar(self, nrlsj,  j, cardCIj, cardCAj, wj):

        if(wj):
            if cardCIj > 1: # If we have only one voxel inactive we can't compute inactive variance
                nu0j = .5*np.dot(self.nrlCI[j], self.nrlCI[j])
                varCIj = 1.0/np.random.gamma(.5*cardCIj + self.varCIPrAlpha,
                                      1/(nu0j + self.varCIPrBeta))
            else :
                pyhrf.verbose(6,'using only hyper priors for CI (empty class) ...')
                varCIj = 1.0/np.random.gamma(self.varCIPrAlpha, 1/self.varCIPrBeta)


            if cardCAj > 1:
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

            meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar)

            rPrMV = self.meanCAPrMean/self.meanCAPrVar

            meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood+rPrMV)

            meanCAj = np.random.normal(meanCAMeanAPost, meanCAVarAPost**0.5)

        else:
            nu0j = .5*np.dot(nrlsj, nrlsj)

            varCIj = 1.0/np.random.gamma(.5*self.nbVox + self.varCIPrAlpha,
                                      1/(nu0j + self.varCIPrBeta))

            varCAj = 1.0/np.random.gamma(self.varCAPrAlpha, 1/self.varCAPrBeta)

            meanCAj = np.random.normal(self.meanCAPrMean, self.meanCAPrVar**0.5)

        return varCIj,meanCAj,varCAj


    def sampleNextInternal(self, variables):
        #TODO : comment

        nrlsSmpl = self.get_variable('nrls')

        cardCA = nrlsSmpl.cardClass[self.L_CA,:]
        cardCI = nrlsSmpl.cardClass[self.L_CI,:]

        w = self.get_variable('W').currentValue

        for j in xrange(self.nbConditions):
            vICI = nrlsSmpl.voxIdx[nrlsSmpl.L_CI][j]
            vICA = nrlsSmpl.voxIdx[nrlsSmpl.L_CA][j]
            self.nrlCI[j] = nrlsSmpl.currentValue[j, vICI]
            self.nrlCA[j] = nrlsSmpl.currentValue[j, vICA]

        for j in xrange(self.nbConditions):
        #for j in np.random.permutation(self.nbConditions):
            if self.hyperPriorFlag:
                varCIj,meanCAj,varCAj = self.computeWithProperPriorsWithRelVar(nrlsSmpl.currentValue[j,:], j, cardCI[j],
                                                                     cardCA[j], w[j])
            else:
                raise Exception('Prior distributions of mixture parameters should be Proper NOT Jeffrey')
                # No Jeffrey Prior, it's complicated with relevance variable

            self.currentValue[self.I_VAR_CI, j] = varCIj
            self.currentValue[self.I_MEAN_CA, j] = meanCAj #absolute(meanCAj)
            self.currentValue[self.I_VAR_CA, j] = varCAj


            pyhrf.verbose(5, 'varCI,%d=%f'%(j,self.currentValue[self.I_VAR_CI,j]))
            pyhrf.verbose(5, 'meanCA,%d=%f'%(j,self.currentValue[self.I_MEAN_CA,j]))
            pyhrf.verbose(5, 'varCA,%d = %f'%(j,self.currentValue[self.I_VAR_CA,j]))

class BiGaussMixtureParamsSamplerWithRelVar(BiGaussMixtureParamsSampler):


    def computeWithProperPriorsWithRelVar(self, nrlsj,  j, cardCIj, cardCAj, wj):

        #if j ==1:
            #print 'NBInactvox =',cardCIj,',     NBActVox =',cardCAj

        if cardCIj > 1: # If we have only one voxel inactive we can't compute inactive variance
            A0 = self.varCIPrAlpha + 0.5*self.nbVox
            A1 = self.varCIPrAlpha + 0.5*cardCIj
            B0 = self.varCIPrBeta + 0.5*np.dot(nrlsj, nrlsj)
            B1 = self.varCIPrBeta + 0.5*np.dot(self.nrlCI[j], self.nrlCI[j])

            varCIj = (1 - wj) * (1.0/np.random.gamma(A0,1/B0)) + wj*(1.0/np.random.gamma(A1,1/B1))

            #if j==1:
                #print 'A1 =',A1,',  B1 =',B1,',     v0 =',varCIj

        else :
            pyhrf.verbose(6,'using only hyper priors for CI (empty class) ...')
            varCIj = 1.0/np.random.gamma(self.varCIPrAlpha, 1/self.varCIPrBeta)


        if cardCAj > 1:
            eta1j = np.mean(self.nrlCA[j])
            nrlCACentered = self.nrlCA[j] - self.currentValue[self.I_MEAN_CA,j]#eta1j
            nu1j = .5 * np.dot(nrlCACentered, nrlCACentered)
            A0 = self.varCAPrAlpha
            A1 = self.varCAPrAlpha + 0.5*cardCAj
            B0 = self.varCAPrBeta
            B1 = self.varCAPrBeta + nu1j
            #r = np.random.gamma(0.5*(cardCAj-1),2/nu1j)
            varCAj = (1 - wj) * (1.0/np.random.gamma(A0,1/B0)) + wj*(1.0/np.random.gamma(A1,1/B1))
            #if j==1:
                #print 'A1 =',A1,',  B1 =',B1,',     v1 =',varCAj
        else :
            pyhrf.verbose(6,'using only hyper priors for CA (empty class) ...')
            eta1j = 0.0
            varCAj = 1.0/np.random.gamma(self.varCAPrAlpha, 1/self.varCAPrBeta)


        invVarLikelihood = (cardCAj+0.)/varCAj
        meanCAVarAPost = 1/(invVarLikelihood + 1/self.meanCAPrVar)
        rPrMV = self.meanCAPrMean/self.meanCAPrVar
        meanCAMeanAPost = meanCAVarAPost * (eta1j*invVarLikelihood+rPrMV)
        meanCAj = (1 - wj) * np.random.normal(self.meanCAPrMean,self.meanCAPrVar**0.5) + wj * np.random.normal(meanCAMeanAPost,meanCAVarAPost**0.5)

        #print 'Cond =',j,',     v0 =',varCIj,',  v1 =',varCAj,',     m1 =',meanCAj

        return varCIj,meanCAj,varCAj


    def sampleNextInternal(self, variables):
        #TODO : comment

        nrlsSmpl = self.get_variable('nrl')

        cardCA = nrlsSmpl.cardClass[self.L_CA,:]
        cardCI = nrlsSmpl.cardClass[self.L_CI,:]

        w = self.get_variable('W').currentValue

        for j in xrange(self.nbConditions):
            vICI = nrlsSmpl.voxIdx[nrlsSmpl.L_CI][j]
            vICA = nrlsSmpl.voxIdx[nrlsSmpl.L_CA][j]
            self.nrlCI[j] = nrlsSmpl.currentValue[j, vICI]
            self.nrlCA[j] = nrlsSmpl.currentValue[j, vICA]

        for j in xrange(self.nbConditions):
        #for j in np.random.permutation(self.nbConditions):
            if self.hyperPriorFlag:
                varCIj,meanCAj,varCAj = self.computeWithProperPriorsWithRelVar(nrlsSmpl.currentValue[j,:], j, cardCI[j],
                                                                     cardCA[j], w[j])
            else:
                raise Exception('Prior distributions of mixture parameters should be Proper NOT Jeffrey')
                # No Jeffrey Prior, it's complicated with relevance variable

            self.currentValue[self.I_VAR_CI, j] = varCIj
            self.currentValue[self.I_MEAN_CA, j] = meanCAj #absolute(meanCAj)
            self.currentValue[self.I_VAR_CA, j] = varCAj


            pyhrf.verbose(5, 'varCI,%d=%f'%(j,self.currentValue[self.I_VAR_CI,j]))
            pyhrf.verbose(5, 'meanCA,%d=%f'%(j,self.currentValue[self.I_MEAN_CA,j]))
            pyhrf.verbose(5, 'varCA,%d = %f'%(j,self.currentValue[self.I_VAR_CA,j]))

class MixtureWeightsSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """

    #TODO : comment

    """


    def __init__(self, do_sampling=True, use_true_value=False, val_ini=None):
        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'mixt_weights', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

    def checkAndSetInitValue(self, variables):
        self.nbClasses = self.get_variable('nrl').nbClasses
        if self.currentValue == None :
            self.currentValue = np.zeros( (self.nbClasses, self.nbConditions),
                                       dtype = float)+0.5
            if 0 and not self.sampleFlag and self.dataInput.simulData != None :
                sn = self.dataInput.simulData.nrls
                for c in xrange(self.nbClasses):
                    #print 'self.currentValue[c,:]:', self.currentValue[c,:].shape
                    #print '(sn.labels==c).sum(axis=1,dtype=float):', (sn.labels==c).sum(axis=1,dtype=float).shape
                    #print 'sn.labels :', sn.labels
                    self.currentValue[c,:] = (sn.labels==c).sum(axis=1,dtype=float) \
                                             /self.nbVoxels

    def sampleNextInternal(self, variables):
        #TODO : comment
        ##print '- Sampling MixtWeights ...'
        #self.currentValue = np.zeros(self.nbConditions, dtype=float)

        nrlsSmpl = self.get_variable('nrl')

        lca = nrlsSmpl.L_CA
        lci = nrlsSmpl.L_CI
        card = nrlsSmpl.cardClass
        nbv = self.nbVoxels
        for j in xrange(self.nbConditions):
            if self.nbClasses == 2:
                self.currentValue[lca,j] = np.random.beta(card[lca,j]+1.5,
                                                       nbv-card[lca,j]+1.5)
                self.currentValue[lci,j] = 1 - self.currentValue[lca,j]
            elif self.nbClasses == 3:
                #TODO : sampling with dirichlet process
                raise NotImplementedError()

        ##print '- Done sampling MixtWeights ...\n'

        assert (self.currentValue.sum(0) == 1.).all()

    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
        return outputs
