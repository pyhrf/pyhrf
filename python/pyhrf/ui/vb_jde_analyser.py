

# -*- coding: utf-8 -*-
import numpy as np
from time import time
from pyhrf.ui.analyser_ui import FMRIAnalyser
from pyhrf.ndarray import xndarray
from pyhrf.vbjde.Utils import Main_vbjde_Extension,Main_vbjde_Extension_NoDrifts,Main_vbjde_Python,roc_curve,Main_vbjde_NoDrifts_ParsiMod_Python
from pyhrf.vbjde.Utils import Main_vbjde_NoDrifts_ParsiMod_C_1,Main_vbjde_Extension_ParsiMod_C_1,Main_vbjde_Extension_ParsiMod_C_1_MeanLabels,Main_vbjde_NoDrifts_ParsiMod_C_2,Main_vbjde_NoDrifts_ParsiMod_C_3
from pyhrf.vbjde.Utils import Main_vbjde_Extension_ParsiMod_C_3,Main_vbjde_Extension_ParsiMod_C_3_tau2,Main_vbjde_Extension_ParsiMod_C_3_tau2_FixedTau1
from pyhrf.vbjde.Utils import Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond,Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond_FixedTau1,Main_vbjde_Extension_ParsiMod_C_4
from pyhrf.vbjde.Utils import Main_vbjde_Extension_ParsiMod_C_4_tau2, Main_vbjde_Extension_ParsiMod_C_4_tau2_FixedTau1
from pyhrf.vbjde.Utils import Main_vbjde_Extension_ParsiMod_C_RVM,classify,Main_vbjpde


#from pyhrf.vbjde.Utils import Main_vbjde, Main_vbjde_Fast, Main_vbjde_Extension,Main_vbjde_Extension_NoDrifts
from scipy.linalg import norm
from pyhrf.tools.io import read_volume
#######################
#from pylab import *
#from matplotlib import pyplot
#######################
from pylab import *
import pyhrf
from pyhrf.xmlio import XmlInitable
from pyhrf.tools import format_duration
import os.path as op
def change_dim(labels):
    '''
    Change labels dimension from
    (ncond, nclass, nvox)
    to
    (nclass, ncond, nvox)
    '''
    ncond = labels.shape[0]
    nclass = labels.shape[1]
    nvox = labels.shape[2]
    newlabels = np.zeros((nclass, ncond, nvox))
    for cond in xrange(ncond):
       for clas in xrange(nclass):
           for vox in xrange(nvox):
               newlabels[clas][cond][vox] = labels[cond][clas][vox]

    return newlabels

DEFAULT_INIT_PARCELLATION = pyhrf.get_data_file_name('subj0_parcellation_for_jpde.nii.gz')

from pyhrf.ui.jde import JDEAnalyser
class JDEVEMAnalyser(JDEAnalyser):

    parametersComments = {
        'CompMod' : 'running Complet Model (Without Condition selection)',
        'ParsiMod' : 'running Parsimonious Model with variable selection',
        'definition1' : 'running the first version of parsimonious model (Q -> W)',
        'definition2' : 'running the second version of parsimonious model (W -> Q)',
        'definition3' : 'running the third version of parsimonious model (\mu_1 -> W)',
        'definition4' : 'running the fourth version of parsimonious model (dKL -> W)',
        'FixedThresh' : 'running the parsimonious model with a predefined threshold value (tau2)',
        'EstimThresh' : 'running the parsimonious model with threshold estimation (tau2)',
        'OneThresh' : 'Estimation of one threshold value for all experimental conditions',
        'ThreshPerCond' : 'Estimation of a threshold value for each experimental conditions (only for definition3)',
        'FixedTau1' : 'Tau1 is fixed during analysis and does not change with tau2',
        'RVM' : 'running the parsimonious model using the Relevant Vector Machine technique',
        'dt' : 'time resolution of the estimated HRF in seconds',
        'hrfDuration': 'duration of the HRF in seconds',
        'sigmaH': 'variance of the HRF',
        'fast': 'running fast VEM with C extensions',
        'nbClasses': 'number of classes for the response levels',
        'PLOT': 'plotting flag for convergence curves',
        'nItMax': 'maximum iteration number',
        'nItMin': 'minimum iteration number',
        'scale': 'flag for the scaling factor applied to the data fidelity '\
            'term during m_h step.\n'
            'If scale=False then do nothing, else divide ' \
            'the data fidelity term by the number of voxels',
        'beta': 'initial value of spatial Potts regularization parameter',
        'simulation' : 'indicates whether the run corresponds to a simulation example or not',
        'estimateSigmaH': 'estimate or not the HRF variance',
        'estimateHRF': 'estimate or not the HRF',
        'TrueHrfFlag' : 'If True, HRF will be fixed to the simulated value',
        'HrfFilename' : 'True HRF Filename',
        'estimateBeta': 'estimate or not the Potts spatial regularization '\
            'parameter',
        'estimateDrifts': 'Explicit drift estimation (if False then drifts' \
            ' are marginalized',
        #'driftType': 'type of the drift basis (default=``polynomlial``)', (not used in VEM)',
        'outputFile': 'output xml file',
        'sigmaH': 'Initial HRF variance',
        'contrasts': 'Contrasts to be evaluated' ,
        'hyper_prior_sigma_H': 'Parameter of the hyper-prior on sigma_H (if zero, no prior is applied)',
        'jpde': "Jointly estimate the parcellation",
        'init_parcellation_file': "Parcellation mask to init JPDE",
        'estimateW':'estimate or not the relevance variable W',
        'tau1' : "Slot of sigmoid function",
        'tau2' : "Threshold of sigmoid function",
        'alpha_tau2' : "first parameter of gamma prior on tau2",
        'lambda_tau2' : "second parameter of gamma prior on tau2",
        'S': "Number of MC step Iterations",
        #'alpha' : "Prior Bernoulli probability of w=1",
        #'alpha_0' : "External field",
        'alpha' : "Confidence level in Posterior Probability Map (PPM)",
        'gamma' : "activation Threshold in Posterior Probability Map (PPM)",
        'estimateLabels' : 'estimate or not the Labels',
        'LabelsFilename' : 'True Labels Filename',
        'MFapprox' : 'Using of the Mean Field approximation in labels estimation',
        'estimateMixtParam' : 'estimate or not the mixture parameters',
        'InitVar' : 'Initiale value of active and inactive gaussian variances',
        'InitMean' : 'Initiale value of active gaussian means',
        'MiniVemFlag' : 'Choosing, if True, the best initialisation of MixtParam and gamma_h',
        'NbItMiniVem' : 'The number of iterations in Mini VEM algorithme',
        }

    parametersToShow = ['CompMod','ParsiMod', 'definition1', 'definition2', 'definition3', 'definition4',
                        'FixedThresh', 'EstimThresh', 'OneThresh', 'ThreshPerCond', 
                        'FixedTau1', 'RVM', 'dt', 'hrfDuration', 'nItMax', 'nItMin',
                        'estimateSigmaH', 'estimateHRF', 'TrueHrfFlag', 'HrfFilename', 'estimateBeta',
                        'estimateLabels','LabelsFilename','MFapprox','estimateW',
                        'estimateDrifts','estimateMixtParam', 'InitVar', 'InitMean', 'outputFile',
                        'scale', 'nbClasses', 'fast', 'PLOT','sigmaH',
                        'contrasts','hyper_prior_sigma_H', 'jpde',
                        'init_parcellation_file', 'tau1', 'tau2', 'alpha_tau2', 'lambda_tau2', 'alpha', 'gamma', 'S',#'alpha','alpha_0',
                        'simulation','MiniVemFlag','NbItMiniVem']
                        
    def __init__(self, hrfDuration=25., sigmaH=0.1, fast=True, CompMod=True, ParsiMod=False, 
                 definition1=False, definition2=False, definition3=False, definition4=False,
                 FixedThresh=False, EstimThresh=True, OneThresh=True, ThreshPerCond=False, 
                 FixedTau1=True, RVM=False, computeContrast=True, nbClasses=2,
                 PLOT=False, nItMax=1, nItMin=1, scale=False, beta=1.0 ,
                 estimateSigmaH=True, estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateDrifts=True,
                 hyper_prior_sigma_H=1000,
                 estimateSigmaEpsilone=True, dt=.6, estimateBeta=True,
                 contrasts={'1':'rel1'},
                 simulation=False,
                 outputFile='./jde_vem_outputs.xml',
                 jpde=False,
                 init_parcellation_file=DEFAULT_INIT_PARCELLATION, estimateW=True, tau1=1.,
                 tau2=0.1, alpha_tau2=3.0, lambda_tau2=4.0, alpha=0.95, gamma=0.0, S=100,# alpha=0.5, alpha_0=0.5, 
                 estimateLabels=True, LabelsFilename='labels.nii', MFapprox=False,
                 estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVemFlag=False,NbItMiniVem=5):

        XmlInitable.__init__(self)
        JDEAnalyser.__init__(self, outputPrefix='jde_vem_')


        # Important thing : all parameters must have default values
        self.dt = dt
        #self.driftType = driftType
        self.hrfDuration = hrfDuration
        self.nbClasses = nbClasses
        self.nItMax = nItMax
        self.estimateSigmaH = estimateSigmaH
        self.scale = scale
        self.estimateDrifts = estimateDrifts
        self.PLOT = PLOT
        self.fast = fast
        self.ParsiMod = ParsiMod
        self.CompMod = CompMod
        self.definition1 = definition1
        self.definition2 = definition2
        self.definition3 = definition3
        self.definition4 = definition4
        self.FixedThresh = FixedThresh
        self.EstimThresh = EstimThresh
        self.OneThresh = OneThresh
        self.ThreshPerCond = ThreshPerCond
        self.FixedTau1 = FixedTau1
        self.RVM = RVM
        self.simulation = simulation
        self.beta = beta
        self.sigmaH = sigmaH
        self.estimateHRF = estimateHRF
        self.TrueHrfFlag = TrueHrfFlag
        self.HrfFilename = HrfFilename
        self.estimateSigmaEpsilone = estimateSigmaEpsilone
        self.nItMin = nItMin
        self.estimateBeta = estimateBeta
        self.estimateW = estimateW
        self.estimateMixtParam = estimateMixtParam
        self.tau1 = tau1
        self.tau2 = tau2
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_tau2 = alpha_tau2
        self.lambda_tau2 = lambda_tau2
        self.S = S
        #self.alpha = alpha
        #self.alpha_0 = alpha_0
        self.estimateLabels = estimateLabels
        self.LabelsFilename = LabelsFilename
        self.MFapprox = MFapprox
        self.InitVar = InitVar
        self.InitMean = InitMean
        self.MiniVemFlag = MiniVemFlag
        self.NbItMiniVem = NbItMiniVem
        if contrasts is None:
            contrasts = {}
        self.contrasts = contrasts
        self.computeContrast = computeContrast
        self.hyper_prior_sigma_H  = hyper_prior_sigma_H

        self.jpde = jpde
        self.init_parcellation_file = init_parcellation_file

        pyhrf.verbose(2, "VEM analyzer:")
        pyhrf.verbose(2, " - JPDE: %s" %str(self.jpde))
        pyhrf.verbose(2, " - estimate sigma H: %s" %str(self.estimateSigmaH))
        pyhrf.verbose(2, " - init sigma H: %f" %self.sigmaH)
        pyhrf.verbose(2, " - hyper_prior_sigma_H: %f" %self.hyper_prior_sigma_H)
        pyhrf.verbose(2, " - estimate drift: %s" %str(self.estimateDrifts))


        #self.contrasts.pop('dummy_example', None)	
        
    def analyse_roi(self, roiData):
        #roiData is of type FmriRoiData, see pyhrf.core.FmriRoiData
        # roiData.bold : numpy array of shape
        #print '!! JDEVEMAnalyser !!'
        ## BOLD has shape (nscans, nvoxels)
        #print 'roiData.bold:', roiData.bold.shape
        #print roiData.bold
        #print 'roiData.tr:'
        #print roiData.tr
        #nbVoxels = roiData.nbVoxels
        #print 'nbVoxels:', nbVoxels
        #print '** paradigm **'
        #print 'onsets:', roiData.get_joined_onsets()
        #print 'bin seq of sampled onsets:'
        #print roiData.get_rastered_onset(self.dt)

        #roiData.graph #list of neighbours
        data = roiData.bold
        #print data.shape,roiData.get_nb_vox_in_mask()
        #raw_input('')
        #noise = roiData.rnoise.data
        #snr = 20*log(norm(data-noise) / norm(noise))
        #print roiData.onsets
        #print "----------------------------"
        Onsets = roiData.get_joined_onsets()
        #print Onsets
        TR = roiData.tr
        #K = 2 #number of classes
        beta = self.beta
        scale = 1#roiData.nbVoxels
        #print dir(roiData)
        #print roiData.get_nb_vox_in_mask()
        nvox = roiData.get_nb_vox_in_mask()
        if self.scale:
            scale = nvox
            #scale = roiData.nbVoxels
        #print self.sigmaH
        rid = roiData.get_roi_id()
        pyhrf.verbose(1,"JDE VEM - roi %d, nvox=%d, nconds=%d, nItMax=%d" \
                          %(rid, nvox, len(Onsets),self.nItMax))

        self.contrasts.pop('dummy_example', None)
        cNames = roiData.paradigm.get_stimulus_names()
        graph = roiData.get_graph()

        t_start = time()

        if self.jpde:
            #print 'Do the wonderful joint detection estimation !'
            #print 'fix subsequent if / else (if needed) ...'
            init_parcellation = read_volume(self.init_parcellation_file)[0]
            #flatten to align with BOLD data:
            init_parcellation = init_parcellation[np.where(roiData.roiMask)]

            #print 'init parcellation:'
            from pyhrf.parcellation import parcellation_report
            pyhrf.verbose(2, parcellation_report(init_parcellation))
            #nbParcels = len(np.unique(init_parcellation))

            init_parcellation -= init_parcellation.min()
            init_parcellation = np.array(init_parcellation) + 1
            J = init_parcellation.shape[0]
            Pmask0 = init_parcellation
            #Pmask0 = np.zeros(J)
            #for j in xrange(0,J):
                #if ((init_parcellation[j] == 0) or (init_parcellation[j] == 1)):
                    #Pmask0[j] = 1
                #if ((init_parcellation[j] == 2) or (init_parcellation[j] == 3)):
                    #Pmask0[j] = 2

            #print init_parcellation.shape,Pmask0.shape
            #raw_input('')
            #for j in xrange(0,J):
                #if ((init_parcellation[j] == 0) or (init_parcellation[j] == 1)):
                    #Pmask0[j] = 1
                #if ((init_parcellation[j] == 2) or (init_parcellation[j] == 3)):
                    #Pmask0[j] = 2
                #if ((init_parcellation[j] == 4) or (init_parcellation[j] == 5)):
                    #Pmask0[j] = 3
                #if ((init_parcellation[j] == 6) or (init_parcellation[j] == 7)):
                    #Pmask0[j] = 4

            Pmask0 = Pmask0.astype(int)
            nbParcels = Pmask0.max()+1
            #print nbParcels
            #raw_input('')
            #print range(nbParcels)
            #raw_input('')
            #print Pmask0.max(),Pmask0.min()
            sigmaH_prior = 0.5*self.sigmaH
            beta_par = 0.5
            #print self.sigmaH
            #print nbClasses
            #print init_parcellation
            #print init_parcellation.shape
            #nrls, estimated_hrf, labels, parcels, EstPmask, EstHRFDict, noiseVar, mu_k, sigma_k, Beta, L, PL,cA,cH,cZ = Main_vbjpde(graph,data,Onsets,self.hrfDuration,init_parcellation,TR,self.dt,self.nbClasses,nbParcels,self.sigmaH,sigmaH_prior,beta,beta_par,self.nItMax,self.nItMin,self.estimateBeta)
            #nrls, estimated_hrf, labels, parcels, EstPmask, EstHRFDict, noiseVar, mu_k, sigma_k, Beta, L, PL,cA,cH,cZ,cQ = Main_vbjpde(graph,data,Onsets,self.hrfDuration,Pmask0,TR,self.dt,self.nbClasses,sigmaH_prior,self.sigmaH,beta,beta_par,self.nItMax,self.nItMin,self.estimateBeta)
	    nrls, estimated_hrf, labels, parcels, EstPmask, EstHRFDict, noiseVar, mu_k, sigma_k, Beta, L, PL,cA,cH,cZ,cQ = Main_vbjpde(graph,data,Onsets,self.hrfDuration,Pmask0,TR,self.dt,self.nbClasses,sigmaH_prior,beta,beta_par,self.nItMax,self.nItMin,self.estimateBeta)
	    #m_A , m_H, q_Z, q_Q, EstPmask,EstHRFDict, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, cA,cH,cZ,cQ =	 Main_vbjpde(graph,Y,Onsets,Thrf,Pmask0,TR,dt,K,v_h,beta,beta_Q,nItMax,nItMin,outDir='/home/chaari/Boulot/Data/JPDE/simuls') 

            #EstPmask = classify(parcels,EstPmask)
            #EstPmask += 1


        else:
            if self.fast:
                if self.CompMod:
                    if self.estimateDrifts:
                        pyhrf.verbose(2, "fast VEM with drift estimation")
                        NbIter, nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cAH,cTime,cTimeMean,Sigma_nrls, StimuIndSignal,FreeEnergy = Main_vbjde_Extension(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag,self.HrfFilename,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)
                        #print 'cTimeMean=',cTimeMean
                    else:
                        pyhrf.verbose(2, "fast VEM without drift estimation")
                        #print 'self.contrasts=',self.contrasts
                        nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, CONTRAST, CONTRASTVAR, cA,cH,cZ,cTime,cTimeMean,Sigma_nrls, StimuIndSignal  = Main_vbjde_Extension_NoDrifts(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF)
 
                if self.ParsiMod:
                    if self.estimateDrifts:
                        if self.definition1:
                            Tau2,NbIter,nrls,estimated_hrf,labels,noiseVar,mu_k,sigma_k,Beta,L,PL,CONTRAST,CONTRASTVAR,cA,cH,cZ,cW,cAH,w,cTime,cTimeMean,Sigma_nrls,MCMean,StimuIndSignal,FreeEnergy = Main_vbjde_Extension_ParsiMod_C_1(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.estimateW,self.tau1,self.tau2,self.S,self.estimateLabels,self.LabelsFilename,self.InitVar,self.InitMean)
                            #Tau2,NbIter,nrls,estimated_hrf,labels,noiseVar,mu_k,sigma_k,Beta,L,PL,CONTRAST,CONTRASTVAR,cA,cH,cZ,cW,cAH,w,cTime,cTimeMean,Sigma_nrls,MCMean,StimuIndSignal,FreeEnergy = Main_vbjde_Extension_ParsiMod_C_1_MeanLabels(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.estimateW,self.tau1,self.tau2,self.S,self.estimateLabels,self.LabelsFilename,self.InitVar,self.InitMean)
                        
                        #if self.definition2:    ##### To Do
                        
                        if self.definition3:
                            
                            if self.FixedThresh:
                                pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 3)) with drift estimation")
                                NbIter, nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2  = Main_vbjde_Extension_ParsiMod_C_3(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag,self.HrfFilename,self.estimateW, self.tau1, self.tau2,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)

                            if self.EstimThresh:
                                if self.OneThresh:
                                    pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 3 / Estimation of tau2)) with drift estimation")
                                    if not self.FixedTau1:
                                        nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2 = Main_vbjde_Extension_ParsiMod_C_3_tau2(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag, self.HrfFilename,self.estimateW,self.tau1, self.tau2,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)
                                    if self.FixedTau1:
                                        NbIter, nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2 = Main_vbjde_Extension_ParsiMod_C_3_tau2_FixedTau1(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag,self.HrfFilename,self.estimateW, self.tau1, self.tau2,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)
                                    
                                if self.ThreshPerCond:
                                    pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 3 / Estimation of tau2 per cond)) with drift estimation")
                                    if not self.FixedTau1:
                                        nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2 = Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag,self.HrfFilename, self.estimateW,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)
                                    if self.FixedTau1:
                                        NbIter, nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2 = Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond_FixedTau1(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.TrueHrfFlag,self.HrfFilename, self.estimateW,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean,self.MiniVemFlag,self.NbItMiniVem)
                                    
                                if(self.OneThresh==True and self.ThreshPerCond==True):
                                    raise Exception('YOU CHOOSE ONE AND MULTIPLE THRESHOLD PER COND AT THE SAME TIME :-(' )
                                if(self.OneThresh==False and self.ThreshPerCond==False):
                                    raise Exception('DO YOU WANT ONE OR MULTIPLE THRESHOLD PER COND ?' )
                                
                            if(self.FixedThresh==True and self.EstimThresh==True):
                                raise Exception('YOU CHOOSED A FIXED OR ESTIMATED THRESHOLD AT THE SAME TIME :-(' )
                            
                            if(self.FixedThresh==False and self.EstimThresh==False):
                                raise Exception('DO YOU WANT A FIXED OR ESTIMATED THRESHOLD ?' )
                        
                        if self.definition4:
                            
                            if self.FixedThresh:
                                pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 4)) with drift estimation")
                                nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy  = Main_vbjde_Extension_ParsiMod_C_4(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.estimateW, self.tau1, self.tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean)
                           
                            if self.EstimThresh:
                                pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 4 / Estimation of tau2)) with drift estimation")
                                #nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2  = Main_vbjde_Extension_ParsiMod_C_4_tau2(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.estimateW, self.tau1, self.tau2,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean)  
                                nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls, StimuIndSignal, FreeEnergy, Tau2  = Main_vbjde_Extension_ParsiMod_C_4_tau2_FixedTau1(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.estimateW, self.tau1, self.tau2,self.alpha_tau2,self.lambda_tau2,self.S,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean)  
                                
                            if(self.FixedThresh==True and self.EstimThresh==True):
                                raise Exception('YOU CHOOSED A FIXED OR ESTIMATED THRESHOLD AT THE SAME TIME :-(' )
                            
                            if(self.FixedThresh==False and self.EstimThresh==False):
                                raise Exception('DO YOU WANT A FIXED OR ESTIMATED THRESHOLD ?' )
                                
                            
                        if self.RVM:
                            pyhrf.verbose(2, "fast Parsimonious Model VEM ((Relevant Vector Machine)) with drift estimation")
                            NbIter,nrls,estimated_hrf,labels,noiseVar,mu_k,sigma_k,Beta,L,PL,CONTRAST,CONTRASTVAR,cA,cH,cZ,cW,w,Sigma_w,alpha_RVM,cTime,cTimeMean,Sigma_nrls = Main_vbjde_Extension_ParsiMod_C_RVM(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF,self.estimateW,self.estimateLabels,self.LabelsFilename,self.MFapprox,self.estimateMixtParam,self.InitVar,self.InitMean)                                                                                                                                                                             

                        if(self.definition1==False and self.definition3==False and self.definition4==False and self.RVM==False):
                            raise Exception('YOU DID NOT CHOOSE ANY DEFINITION FOR THE PARSIMONIOUS MODEL :-(' )
                            
                        if((self.definition3==True and self.definition4==True) or (self.definition3==True and self.RVM==True) or (self.RVM==True and self.definition4==True) or (self.definition3==True and self.definition4==True and self.RVM==True)):
                            raise Exception('YOU CHOOSED MANY DIFFERENT DEFINITIONS FOR THE PARSIMONIOUS MODEL :-( ')
                            

                    else:   
                        if self.definition1:
                            pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 1)) without drift estimation")
                            NbIter, nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW,cAH, w,cTime,cTimeMean,Sigma_nrls,MCMean, StimuIndSignal, FreeEnergy  = Main_vbjde_NoDrifts_ParsiMod_C_1(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.tau1, self.tau2,self.S,self.InitVar,self.InitMean)
                        if self.definition2:
                            pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 2)) without drift estimation")
                            nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW, w,cTime,cTimeMean,Sigma_nrls,MCMean, StimuIndSignal  = Main_vbjde_NoDrifts_ParsiMod_C_2(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.tau1, self.tau2,self.S, self.alpha, self.alpha_0)
                        if self.definition3:
                            pyhrf.verbose(2, "fast Parsimonious Model VEM ((Definition 3)) without drift estimation")
                            nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, CONTRAST, CONTRASTVAR, cA,cH,cZ,cW, w,cTime,cTimeMean,Sigma_nrls,MCMean, StimuIndSignal,FreeEnergy  = Main_vbjde_NoDrifts_ParsiMod_C_3(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT,self.contrasts,self.computeContrast,self.hyper_prior_sigma_H,self.estimateHRF, self.tau1, self.tau2,self.S)
                        if(self.definition1==False and self.definition2==False and self.definition3==False):
                            raise Exception('YOU DID NOT CHOOSE ANY DEFINITION FOR THE PARSIMONIOUS MODEL :-( ')
                        if((self.definition1==True and self.definition2==True) or (self.definition1==True and self.definition3==True) or (self.definition2==True and self.definition3==True) or (self.definition1==True and self.definition2==True and self.definition3==True)):
                            raise Exception('YOU CHOOSED MANY DIFFERENT DEFINITIONS FOR THE PARSIMONIOUS MODEL :-( ')
                        
            else:
                if self.CompMod:
                    if self.estimateDrifts:
                        pyhrf.verbose(2, "not fast VEM")
                        nrls, estimated_hrf, labels, noiseVar, mu_k, sigma_k, Beta, L, PL = Main_vbjde_Python(graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.nItMax,self.nItMin,self.estimateBeta,self.PLOT)     
                        
        self.analysis_duration = time() - t_start
        pyhrf.verbose(1, 'JDE VEM analysis took: %s' \
                          %format_duration(self.analysis_duration))

            #nrls, estimated_hrf, labels, noiseVar,Hist_sigmaH = Main_vbjde(roiData.graph,data,Onsets,self.hrfDuration,self.nbClasses,TR,beta,self.dt,scale,self.estimateSigmaH,self.sigmaH,self.PLOT,self.nItMax,self.nItMin)
        #figure(1)
        #plot(estimated_hrf)
        #show()
        # Pack all outputs within a dict
        outputs = {}
        hrf_time = np.arange(len(estimated_hrf)) * self.dt
        #print hrf_time.shape

        #Xit = np.arange(len(Hist_sigmaH))

        #print estimated_hrf[0:3].shape
        #print Xit.shape
        #print hrf_time
        #print Xit
        #outputs['sigmaH'] = xndarray(Hist_sigmaH, axes_names=['iteration'],
                #axes_domains={'iteration'it},value_label="sigmaH")

        #outputs['sigmaH'] = xndarray(Hist_sigmaH, axes_names=['iteration'])

        if not self.jpde:
            #axes_names = ['time', 'voxel']
            #axes_domains = {'time' : np.arange(data.shape[0])*TR}
            #outputs['StimuIndSignal'] = xndarray(StimuIndSignal.astype(np.float64),
                                               #axes_names=axes_names,
                                               #axes_domains=axes_domains)

            #e = np.sqrt((StimuIndSignal.astype(np.float64) - data.astype(np.float64))**2)
            #outputs['rmse'] = xndarray(e.mean(0), axes_names=['voxel'], value_label='Rmse')
            
            if ((self.ParsiMod and (self.definition1 or self.definition3 or self.definition4)) or self.CompMod):
                axes_names = ['iteration']
                axes_domains = {'iteration':np.arange(FreeEnergy.shape[0])}
                outputs['FreeEnergy'] = xndarray(FreeEnergy,
                                            axes_names=axes_names,
                                            axes_domains=axes_domains)

        if self.jpde:
            outputs['hrf'] = xndarray(estimated_hrf, axes_names=['time','voxel'],
                                axes_domains={'time':hrf_time,'voxel':range(estimated_hrf.shape[1])},
                                value_label="HRF")
            tmp = np.array(EstHRFDict)
            D = EstHRFDict[0].shape[0]
            tmp2 = np.zeros((nbParcels,D))
            for i in xrange(0,nbParcels):
                #print EstHRFDict[i]
                #print EstHRFDict[i].shape,tmp2[i,:].shape
		#raw_input('')
		tmp2[i,:] = EstHRFDict[i]

            #print tmp
            #print tmp.shape
            #tmp[3,:] = 0
            #tmp[6,:] = 1
            
            
            outputs['EstHRFDict'] = xndarray(tmp2, axes_names=['class','time'],
                        axes_domains={'class':range(nbParcels),'time':hrf_time},
                        value_label="EstHRFDict")

            domParcel = {'parcel':range(nbParcels),'voxel':range(parcels.shape[1])}
            outputs['parcels'] = xndarray(parcels,value_label="Parcels",
                        axes_names=['parcel','voxel'],
                        axes_domains=domParcel)

            outputs['Pmask'] = xndarray(EstPmask,value_label="ROI",
                        axes_names=['voxel'])

            outputs['Pmask0'] = xndarray(init_parcellation,value_label="ROI",
                        axes_names=['voxel'])

        else:
            outputs['hrf'] = xndarray(estimated_hrf, axes_names=['time'],
                                axes_domains={'time':hrf_time},
                                value_label="HRF")

        domCondition = {'condition':cNames}
        outputs['nrls'] = xndarray(nrls.transpose(),value_label="NRLs",
                                 axes_names=['condition','voxel'],
                                 axes_domains=domCondition)
        
        ad = {'condition':cNames,'condition2':Onsets.keys()}
        if not self.jpde:
            outputs['Sigma_nrls'] = xndarray(Sigma_nrls,value_label="Sigma_NRLs",
                                           axes_names=['condition','condition2','voxel'],
                                           axes_domains=ad)
            
            outputs['NbIter'] = xndarray(np.array([NbIter]),value_label="NbIter")
            ### Computing PPM
            #from scipy.stats import norm
            #NbVox = nrls.shape[0]
            #NbCond = nrls.shape[1]
            #PPM = np.zeros((NbVox,NbCond),dtype=float)
            #PPM_Tresh = np.zeros((NbVox,NbCond),dtype=int)
            #Thresh = np.zeros((NbVox),dtype=float)
            #NbVoxPPMThresh = np.zeros((NbCond),dtype=int)
            #gammaPPM = self.gamma * np.ones(NbVox,dtype=float)
            #for m in xrange(NbCond):
                #mPPM = nrls[:,m]
                #sPPM = np.sqrt(Sigma_nrls[m,m,:])
                #Thresh = ( gammaPPM - mPPM ) / sPPM
                #PPM[:,m] = 1.0 - norm.cdf(Thresh)
                #PPM_Tresh[np.where(PPM[:,m] > self.alpha),m] = 1
                #NbVoxPPMThresh[m] = sum(PPM_Tresh[:,m])
                
            #outputs['PPM'] = xndarray(PPM.transpose(),value_label="PPM",
                #axes_names=['condition','voxel'],
                #axes_domains=domCondition)
                
            #outputs['PPM_Tresh'] = xndarray(PPM_Tresh.transpose(),value_label="PPM_Tresh",
                #axes_names=['condition','voxel'],
                #axes_domains=domCondition)
             
            #outputs['NbVoxPPMThresh'] = xndarray(NbVoxPPMThresh,value_label="NbVoxPPMThresh",
                #axes_names=['condition'],
                #axes_domains=domCondition) 
    
        outputs['beta'] = xndarray(Beta,value_label="beta",
                                 axes_names=['condition'],
                                 axes_domains=domCondition)

        nbc, nbv = len(cNames), nrls.shape[0]
        repeatedBeta = np.repeat(Beta, nbv).reshape(nbc, nbv)
        outputs['beta_mapped'] = xndarray(repeatedBeta,value_label="beta",
                                        axes_names=['condition','voxel'],
                                        axes_domains=domCondition)

        outputs['roi_mask'] = xndarray(np.zeros(nbv)+roiData.get_roi_id(),
                                     value_label="ROI",
                                     axes_names=['voxel'])

        if self.ParsiMod:
            if not self.RVM:
                an = ['condition','class']
                ad = {'class':['inactiv','activ'],
                            'condition': cNames}
                outputs['w'] = xndarray(w,value_label="w",
                                    axes_names=an,
                                    axes_domains=ad)
                
                #if (self.EstimThresh==True and self.OneThresh==True):
                if (self.definition1==True or self.definition3==True):
                    outputs['tau2'] = xndarray(np.array([Tau2]),value_label="tau2")
                
                if (self.EstimThresh==True and self.ThreshPerCond==True):
                    an = ['condition']
                    ad = {'condition': cNames}
                    outputs['tau2'] = xndarray(Tau2,value_label="tau2", 
                                        axes_names=an,
                                        axes_domains=ad)
            if self.RVM:
                an = ['condition']
                ad = {'condition': cNames}
                outputs['w'] = xndarray(w,value_label="w", 
                                      axes_names=an,
                                      axes_domains=ad)
                
                outputs['alpha_RVM'] = xndarray(alpha_RVM,value_label="alpha_RVM", 
                                              axes_names=an,
                                              axes_domains=ad)
                
                ad = {'condition':cNames,'condition2':Onsets.keys()}
                outputs['Sigma_w'] = xndarray(Sigma_w,value_label="Sigma_w",
                                            axes_names=['condition','condition2'],
                                            axes_domains=ad)
                
                
            #an = ['condition','voxel','S','class']
            #ad = {'condition': Onsets.keys(),
            #'S': np.arange(MCMean.shape[2]),
            #'class':['inactiv','activ']}
            #outputs['MCMean'] = xndarray(MCMean,value_label="MCMean",
                                   #axes_names=an,
                                   #axes_domains=ad)

        h = estimated_hrf
        nrls = nrls.transpose()

        nvox = nrls.shape[1]
        nbconds = nrls.shape[0]
        ah = zeros((h.shape[0], nvox, nbconds))
        #for j in xrange(nbconds):
            #ah[:,:,j] = repeat(h,nvox).reshape(h.shape[0],nvox) * \
                #nrls[j,:]
        #ad = {'time':hrf_time, 'condition':roiData.paradigm.get_stimulus_names()}
        #outputs['ah'] = xndarray(ah, axes_names=['time','voxel','condition'],
                               #axes_domains=ad,
                               #value_label='Delta BOLD')

        if 0:
            # let's look for label switching
            # assume mean closest to 0 corresponds to inactivating class
            for m in xrange(roiData.nbConditions):
                i_inact = np.argmin(np.abs(mu_k[m,:]))
                mu_k[m,i_inact],mu_k[m,0] = mu_k[m,0],mu_k[m,i_inact]
                sigma_k[m,i_inact],sigma_k[m,0] = sigma_k[m,0],sigma_k[m,i_inact]
                labels[m,i_inact,:],labels[m,0,:] = labels[m,0,:],labels[m,i_inact,:]

        mixtp = np.zeros((roiData.nbConditions, self.nbClasses, 2))
        mixtp[:, :, 0] = mu_k
        mixtp[:, :, 1] = sigma_k**2
       
        an = ['condition','Act_class','component']
        ad = {'Act_class':['inactiv','activ'],
              'condition': cNames,
              'component':['mean','var']}
        outputs['mixt_p'] = xndarray(mixtp, axes_names=an, axes_domains=ad)

        ad = {'class' : ['inactiv','activ'],
              'condition': cNames,
              }
        outputs['labels'] = xndarray(labels,value_label="Labels",
                                   axes_names=['condition','class','voxel'],
                                   axes_domains=ad)
        outputs['noiseVar'] = xndarray(noiseVar,value_label="noiseVar",
                                     axes_names=['voxel'])
        if self.estimateDrifts:
            outputs['drift_coeff'] = xndarray(L,value_label="Drift",
                            axes_names=['coeff','voxel'])
            outputs['drift'] = xndarray(PL,value_label="Delta BOLD",
                        axes_names=['time','voxel'])
        if not self.jpde and (len(self.contrasts) >0) and self.computeContrast:
            #keys = list((self.contrasts[nc]) for nc in self.contrasts)
            domContrast = {'contrast':self.contrasts.keys()}
            outputs['contrasts'] = xndarray(CONTRAST, value_label="Contrast",
                                          axes_names=['voxel','contrast'],
                                          axes_domains=domContrast)
            #print 'contrast output:'
            #print outputs['contrasts'].descrip()

            c = xndarray(CONTRASTVAR, value_label="Contrasts_Variance",
                       axes_names=['voxel','contrast'],
                       axes_domains=domContrast)
            outputs['contrasts_variance'] = c

            outputs['ncontrasts'] = xndarray(CONTRAST/CONTRASTVAR**.5,
                                           value_label="Normalized Contrast",
                                           axes_names=['voxel','contrast'],
                                           axes_domains=domContrast)

        # use 'voxel' to specify the axis where positions are encoded
        # -> it will be mapped according to the ROI shape aftewards
        #outputs['voxel_stuff'] = xndarray(np.random.randn(nbVoxels),
                                        #axes_names=['voxel'])
        #print "Input SNR = " + str(snr)
        #print "22211212121"
        #print roiData.simulation
        #print dir(roiData.simulation)
        #axes_names = ['iteration']


        ################################################################################
        axes_names = ['duration']

        if not self.jpde:
            #           Convergence         #
            #print cZ
            #print cH
            #print len(cZ),len(cH)
            outName = 'Convergence_Labels'
            #ad = {'Conv_Criterion':np.arange(len(cZ))}
            ax = np.arange(self.nItMax)*cTimeMean
            #print cTimeMean
            #print '------ check -------------'
            #print len(cZ)
            #print len(cTime)
            #print '------ END check -------------'


            ax[:len(cTime)] = cTime
            ad = {'duration':ax}
            #ad = {'iteration':np.arange(self.nItMax)}
            #ad = {'iteration':cTime}
            c = np.zeros(self.nItMax) #-.001 #
            c[:len(cZ)] = cZ
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                      axes_domains=ad,
                                      value_label='Conv_Criterion_Z')

            outName = 'Convergence_HRF'
            #ad = {'Conv_Criterion':np.arange(len(cH))}
            c = np.zeros(self.nItMax) #-.001 #
            c[:len(cH)] = cH
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                      axes_domains=ad,
                                      value_label='Conv_Criterion_H')
            #outName = 'Convergence_HRF'
            #axes_names = ['Conv_Criterion']
            #ad = {'Conv_Criterion_H':np.arange(len(cH))}
            #outputs[outName] = xndarray(np.array(cH),value_label='Conv_Criterion_H')

            outName = 'Convergence_NRL'
            c = np.zeros(self.nItMax)# -.001 #
            c[:len(cA)] = cA
            #ad = {'Conv_Criterion':np.arange(len(cA))}
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                      axes_domains=ad,
                                      value_label='Conv_Criterion_A')
            
            if self.ParsiMod:
                outName = 'Convergence_W'
                c = np.zeros(self.nItMax) -.001 #
                c[:len(cW)] = cW
                #ad = {'Conv_Criterion':np.arange(len(cA))}
            
                outputs[outName] = xndarray(c, axes_names=axes_names,
                                          axes_domains=ad,
                                          value_label='Conv_Criterion_W')                          

        ################################################################################

        #outputs['labels'] = xndarray(labels,value_label="Labels",
                                   #axes_names=['condition','class','voxel'],
                                   #axes_domains=domCondition)
        #raw_input('')
        #print dir(roiData)
        #print roiData.get_data_files()[0]
        #raw_input('')
        if self.simulation:
            from pyhrf.stats import compute_roc_labels
            #print dir(roiData)
            #print dir(roiData.simulation)
            #print roiData.simulation['labels'][0]
            #print roiData.get_data_files()
            #raw_input('')
            #fn = roiData.get_data_files()[0]
            #idx = fn.index('bold')
            #fn = fn[0:idx]
            #labels_file = fn + 'labels_video.nii'
            #labels_file = op.realpath( labels_file )
            #labels_vem_video,_ = read_volume( labels_file )
            labels_vem_audio = roiData.simulation['labels'][0]
            labels_vem_video = roiData.simulation['labels'][1]
            #labels_file = fn + 'labels_audio.nii'
            #labels_vem_audio,_ = read_volume( labels_file )

            #print labels.shape
            #raw_input('')
            M = labels.shape[0]
            K = labels.shape[1]
            J = labels.shape[2]
            true_labels = np.zeros((K,J))
            #print true_labels.shape,labels_vem_audio.shape
            true_labels[0,:] = reshape(labels_vem_audio,(J))
            true_labels[1,:] = reshape(labels_vem_video,(J))
            newlabels = np.reshape(labels[:,1,:],(M,J))
            #print newlabels.shape,true_labels.shape
            se = []
            sp = []
            size = prod(labels.shape)

            for i in xrange(0,M):
                se0,sp0, auc = roc_curve(newlabels[i,:].tolist(),
                                         true_labels[i,:].tolist())
                se.append(se0)
                sp.append(sp0)
                size = min(size,len(sp0))
            SE = np.zeros((M,size),dtype=float)
            SP = np.zeros((M,size),dtype=float)
            for i in xrange(0,M):
                tmp = np.array(se[i])
                SE[i,:] = tmp[0:size]
                tmp = np.array(sp[i])
                SP[i,:] = tmp[0:size]

            sensData, specData = SE, SP
            axes_names = ['condition','1-specificity']
            outName = 'ROC_audio'
            ad = {'1-specificity':specData[0],'condition':cNames}
            outputs[outName] = xndarray(sensData, axes_names=axes_names,
                                      axes_domains=ad,
                                      value_label='sensitivity')



            #axes_names = ['1-specificity','condition']
            #outName = 'ROC'
            #ad = {'1-specificity':specData.transpose(),'condition':Onsets.keys()}
            #print ad
            ##print specData.transpose().shape
            ##print nrls.transpose().shape
            ##print Onsets.keys()
            #outputs[outName] = xndarray(sensData.transpose(), axes_names=axes_names,
                                      #axes_domains=ad,
                                      #value_label='sensitivity')
            #raw_input('')
            #domCondition = {'condition':Onsets.keys()}
        #outputs['nrls'] = xndarray(nrls.transpose(),value_label="NRLs",
                                 #axes_names=['condition','voxel'],
                                 #axes_domains=domCondition)

            #ad = {'Conv_Criterion':np.arange(len(cH))}
        #outputs[outName] = xndarray(np.array(cH), axes_names=axes_names,
                                      #axes_domains=ad,
                                      #value_label='Conv_Criterion_H')

            m = specData[0].min()
            #m2 = specData[0].min()
            #print m,m2
            #print min(m,m2)
            import matplotlib.font_manager as fm
            figure(200)
            #plot(se[0],sp[0],'--',color='k',linewidth=2.0)
            #hold(True)
            #plot(se[1],sp[1],color='k',linewidth=2.0)
            #legend(('audio','video'))
            plot(sensData[0],specData[0],'--',color='k',linewidth=2.0,label='m=1')
            hold(True)
            plot(sensData[1],specData[1],color='k',linewidth=2.0,label='m=2')
            #legend(('audio','video'))
            xticks(color = 'k', size = 14,fontweight='bold')
            yticks(color = 'k', size = 14,fontweight='bold')
            #xlabel('1 - Specificity',fontsize=16,fontweight='bold')
            #ylabel('Sensitivity',fontsize=16,fontweight='bold')
            prop = fm.FontProperties(size=14,weight='bold')
            legend(loc=1,prop=prop)
            axis([0., 1., m, 1.02])
            #grid(True)
            #show()
            #savefig('ROC.png')

            #raw_input('')

            #print true_labels.shape
            #print op.realpath(nrl_file)
            #nrl_vem_audio = dat[0,:,:,0]
            #nrl_vem_video = dat[0,:,:,1]
            #raw_input('')

        #if roiData.simulation is not None:
            #print "999211212121"
            #easy_install --prefix=$USRLOCAL -U scikits.learn
            #from pyhrf.stats import compute_roc_labels_scikit
            from pyhrf.stats import compute_roc_labels
            if hasattr(roiData.simulation, 'nrls'):
                true_labels = roiData.simulation.nrls.labels
                true_nrls = roiData.simulation.nrls.data
            elif isinstance(roiData.simulation, dict) and \
                    roiData.simulation.has_key('labels') and \
                    roiData.simulation.has_key('nrls') :
                true_labels = roiData.simulation['labels']
                true_nrls = roiData.simulation['nrls']
            else:
                raise Exception('Simulation can not be retrieved from %s' \
                                    %str(roiData.simulation))
            #se,sp,auc = compute_roc_labels_scikit(labels[:,1,:], true_labels
            #print 'labels dimension : ', labels.shape
            #print 'true_labels dimension : ', true_labels.shape
            #newlabels = change_dim(labels) # Christine
            #newlabels = labels
            domCondition = {'condition':cNames}
            outputs['Truenrls'] = xndarray(true_nrls,value_label="True_nrls",
                                         axes_names=['condition','voxel'],
                                         axes_domains=domCondition)
            M = labels.shape[0]
            K = labels.shape[1]
            J = labels.shape[2]
            #
            newlabels = np.reshape(labels[:,1,:],(M,J))

            #print 'newlabels dimension : ', newlabels.shape
            #se,sp, auc = compute_roc_labels(newlabels, true_labels)
            #print se.shape,sp.shape,auc
            #print newlabels.shape
            #print true_labels.shape
            #print type(np.array(se))
            #raw_input('')
            #se = np.array(se)
            #sp = np.array(sp)

            #from ROC import roc_curve

            for i in xrange(0,M):
                se0,sp0, auc = roc_curve(newlabels[i,:].tolist(),
                                         true_labels[i,:].tolist())
                se.append(se0)
                sp.append(sp0)
                size = min(size,len(sp0))
            SE = np.zeros((M,size),dtype=float)
            SP = np.zeros((M,size),dtype=float)
            for i in xrange(0,M):
                tmp = np.array(se[i])
                SE[i,:] = tmp[0:size]
                tmp = np.array(sp[i])
                SP[i,:] = tmp[0:size]


            #########
            # noise #
            #########
            #se,sp, auc = roc_curve(newlabels[0,:].tolist(), true_labels[0,:].tolist())

            #SE[0,:] = np.array(se)
            #SP[0,:] = np.array(sp)
            #print SE.shape, (np.array(se)).shape
            #for i in xrange(1,M):
                #se,sp, auc = roc_curve(newlabels[i,:].tolist(), true_labels[i,:].tolist())
                #print SE.shape, (np.array(se)).shape
                #print SP.shape, (np.array(sp)).shape
                #SE[i,:] = np.array(se)
                #SP[i,:] = np.array(sp)

            #raw_input('')

            #se,sp, auc = roc_curve(newlabels[0,1,:].tolist(), true_labels[0,:].tolist())
            #se = np.array(se)
            #sp = np.array(sp)
            #se,sp, auc = compute_roc_labels(newlabels, true_labels)
            #print se.shape
            #print sp.shape
            #print auc
            #sensData, specData = se, sp






            #print sensData.shape,specData[0].shape
            #print Onsets.keys()
            #raw_input('')
            #outputs[outName] = xndarray(sensData, axes_names=axes_names,
                                      #axes_domains=ad,
                                      #value_label='sensitivity')

            #axes_names = ['condition']
            #outputs['AUROC'] = xndarray(auc, axes_names=axes_names,
                                      #axes_domains={'condition':Onsets.keys()})



            #axes_names = ['iteration','Conv_Criterion']
            #axes_names = ['Conv_Criterion']
            #outName = 'Convergence_NRL'
            #ad = {'Conv_Criterion':np.arange(len(cA))}
            #outputs[outName] = xndarray(np.array(cA), axes_names=axes_names,
                                      #axes_domains=ad,
                                      #value_label='Conv_Criterion_A')

            #outName = 'Convergence_Labels'
            #ad = {'Conv_Criterion_Z':np.arange(len(cZ))}
            #outputs[outName] = xndarray(np.array(cZ), axes_names=axes_names,
                                      #axes_domains=ad,
                                      #value_label='Conv_Criterion_Z')


            #outName = 'Convergence_HRF'
            ##print "---------------------------------------"
            ##print outName
            ##print "---------------------------------------"
            #ad = {'Conv_Criterion_H':np.arange(len(cH))}
            #outputs[outName] = xndarray(np.array(cH),value_label='Conv_Criterion_H', axes_names=axes_names,
                                      #axes_domains=ad)


        d = {'parcel_size':np.array([nvox])}
        outputs['analysis_duration'] = xndarray(np.array([self.analysis_duration]),
                                              axes_names=['parcel_size'],
                                              axes_domains=d)

        return outputs



# Function to use directly in parallel computation
def run_analysis(**params):
    # from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
    # import pyhrf
    pyhrf.verbose.set_verbosity(1)
    fdata = params.pop('roi_data')
    # print 'doing params:'
    # print params
    vem_analyser = JDEVEMAnalyser(**params)
    return (dict([('ROI',fdata.get_roi_id())] + params.items()), \
                vem_analyser.analyse_roi(fdata))

