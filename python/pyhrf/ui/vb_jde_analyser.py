# -*- coding: utf-8 -*-

import logging

import numpy as np
from time import time
from pyhrf.ui.analyser_ui import FMRIAnalyser
from pyhrf.ndarray import xndarray
from pyhrf.vbjde.vem_tools import roc_curve
from pyhrf.vbjde.vem_bold_old import Main_vbjde_Extension, Main_vbjde_Extension_stable, Main_vbjde_Python
from pyhrf.vbjde.vem_bold import jde_vem_bold
from pyhrf.vbjde.vem_bold_constrained import Main_vbjde_Extension_constrained, Main_vbjde_Python_constrained
from scipy.linalg import norm
from pyhrf.tools._io import read_volume
import pyhrf
from pyhrf.xmlio import XmlInitable
from pyhrf.tools import format_duration
import os.path as op


logger = logging.getLogger(__name__)


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


from pyhrf.ui.jde import JDEAnalyser
class JDEVEMAnalyser(JDEAnalyser):

    parametersComments = {
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
        'sigmaH': 'Initial HRF variance',
        'contrasts': 'Contrasts to be evaluated' ,
        'hyper_prior_sigma_H': 'Parameter of the hyper-prior on sigma_H (if zero, no prior is applied)',
        'estimateLabels' : 'estimate or not the Labels',
        'LabelsFilename' : 'True Labels Filename',
        'MFapprox' : 'Using of the Mean Field approximation in labels estimation',
        'estimateMixtParam' : 'estimate or not the mixture parameters',
        'InitVar' : 'Initiale value of active and inactive gaussian variances',
        'InitMean' : 'Initiale value of active gaussian means',
        'MiniVemFlag' : 'Choosing, if True, the best initialisation of MixtParam and gamma_h',
        'NbItMiniVem' : 'The number of iterations in Mini VEM algorithme',
        'constrained' : 'adding constrains: positivity and norm = 1 ',
        }

    parametersToShow = ['dt', 'hrfDuration', 'nItMax', 'nItMin',
                        'estimateSigmaH', 'estimateHRF', 'TrueHrfFlag', 'HrfFilename', 'estimateBeta',
                        'estimateLabels','LabelsFilename','MFapprox',
                        'estimateDrifts','estimateMixtParam', 'InitVar', 'InitMean',
                        'scale', 'nbClasses', 'fast', 'PLOT','sigmaH',
                        'contrasts','hyper_prior_sigma_H', 'constrained',
                        'simulation','MiniVemFlag','NbItMiniVem']

    def __init__(self, hrfDuration=25., sigmaH=0.1, fast=True,
                 computeContrast=True, nbClasses=2, PLOT=False, nItMax=100,
                 nItMin=1, scale=False, beta=1.0, estimateSigmaH=True,
                 estimateHRF=True, TrueHrfFlag=False,HrfFilename='hrf.nii',
                 estimateDrifts=True, hyper_prior_sigma_H=1000, dt=.6,
                 estimateBeta=True, contrasts=None, simulation=False,
                 estimateLabels=True, LabelsFilename='labels.nii',
                 MFapprox=False, estimateMixtParam=True, constrained = False,
                 InitVar=0.5,InitMean=2.0,MiniVemFlag=False,NbItMiniVem=5):

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
        self.simulation = simulation
        self.beta = beta
        self.sigmaH = sigmaH
        self.estimateHRF = estimateHRF
        self.TrueHrfFlag = TrueHrfFlag
        self.HrfFilename = HrfFilename
        self.nItMin = nItMin
        self.estimateBeta = estimateBeta
        self.estimateMixtParam = estimateMixtParam
        self.estimateLabels = estimateLabels
        self.LabelsFilename = LabelsFilename
        self.MFapprox = MFapprox
        self.InitVar = InitVar
        self.InitMean = InitMean
        self.MiniVemFlag = MiniVemFlag
        self.NbItMiniVem = NbItMiniVem
        if contrasts is None:
            contrasts = OrderedDict()
        self.contrasts = contrasts
        self.computeContrast = computeContrast
        self.hyper_prior_sigma_H = hyper_prior_sigma_H
        self.constrained = constrained


        logger.info("VEM analyzer:")
        logger.info(" - estimate sigma H: %s", str(self.estimateSigmaH))
        logger.info(" - init sigma H: %f", self.sigmaH)
        logger.info(" - hyper_prior_sigma_H: %f", self.hyper_prior_sigma_H)
        logger.info(" - estimate drift: %s", str(self.estimateDrifts))

    def analyse_roi(self, roiData):
        #roiData is of type FmriRoiData, see pyhrf.core.FmriRoiData
        # roiData.bold : numpy array of shape
        ## BOLD has shape (nscans, nvoxels)

        #roiData.graph #list of neighbours
        data = roiData.bold
        Onsets = roiData.get_joined_onsets()
        TR = roiData.tr
        #K = 2 #number of classes
        beta = self.beta
        scale = 1#roiData.nbVoxels
        nvox = roiData.get_nb_vox_in_mask()
        if self.scale:
            scale = nvox
        rid = roiData.get_roi_id()
        logger.info("JDE VEM - roi %d, nvox=%d, nconds=%d, nItMax=%d", rid,
                    nvox, len(Onsets), self.nItMax)

        self.contrasts.pop('dummy_example', None)
        cNames = roiData.paradigm.get_stimulus_names()
        graph = roiData.get_graph()

        t_start = time()

        if self.fast:
            logger.info("fast VEM with drift estimation"+
                        ("and a constraint"*self.constrained))
            (NbIter, nrls, estimated_hrf, hrf_covariance, labels, noiseVar, mu_k,
             sigma_k, Beta, L, PL, CONTRAST, CONTRASTVAR, cA, cH, cZ, cAH, cTime,
             cTimeMean, Sigma_nrls, StimuIndSignal, density_ratio, ppm_a_nrl,
             ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts) = jde_vem_bold(
                 graph, data, Onsets, self.hrfDuration, self.nbClasses, TR,
                 beta, self.dt, scale, self.estimateSigmaH, self.sigmaH,
                 self.nItMax, self.nItMin, self.estimateBeta, self.PLOT,
                 self.contrasts, self.computeContrast,
                 self.hyper_prior_sigma_H, self.estimateHRF,
                 self.TrueHrfFlag, self.HrfFilename, self.estimateLabels,
                 self.LabelsFilename, self.constrained)
        else:
            # if not self.fast
            if self.estimateDrifts:
                logger.info("not fast VEM")
                logger.info("NOT WORKING")
                nrls, estimated_hrf, \
                labels, noiseVar, mu_k, \
                sigma_k, Beta, L, \
                PL = Main_vbjde_Python_constrained(graph,data,Onsets,
                                       self.hrfDuration,self.nbClasses,
                                       TR,beta,self.dt,scale,
                                       self.estimateSigmaH,self.sigmaH,
                                       self.nItMax,self.nItMin,
                                       self.estimateBeta,self.PLOT)

        # Plot analysis duration
        self.analysis_duration = time() - t_start
        logger.info('JDE VEM analysis took: %s',
                    format_duration(self.analysis_duration))


        if self.fast:
            ### OUTPUTS: Pack all outputs within a dict
            outputs = {}
            hrf_time = np.arange(len(estimated_hrf)) * self.dt

            axes_names = ['iteration']
            """axes_domains = {'iteration':np.arange(FreeEnergy.shape[0])}
            outputs['FreeEnergy'] = xndarray(FreeEnergy,
                                        axes_names=axes_names,
                                        axes_domains=axes_domains)
            """
            outputs['hrf'] = xndarray(estimated_hrf, axes_names=['time'],
                                axes_domains={'time':hrf_time},
                                value_label="HRF")

            domCondition = {'condition':cNames}
            outputs['nrls'] = xndarray(nrls.transpose(),value_label="NRLs",
                                    axes_names=['condition','voxel'],
                                    axes_domains=domCondition)

            ad = {'condition':cNames,'condition2':Onsets.keys()}

            outputs['Sigma_nrls'] = xndarray(Sigma_nrls,value_label="Sigma_NRLs",
                                            axes_names=['condition','condition2','voxel'],
                                            axes_domains=ad)

            outputs['NbIter'] = xndarray(np.array([NbIter]),value_label="NbIter")

            outputs['beta'] = xndarray(Beta,value_label="beta",
                                    axes_names=['condition'],
                                    axes_domains=domCondition)

            nbc, nbv = len(cNames), nrls.shape[0]
            repeatedBeta = np.repeat(Beta, nbv).reshape(nbc, nbv)
            outputs['beta_mapped'] = xndarray(repeatedBeta,value_label="beta",
                                            axes_names=['condition','voxel'],
                                            axes_domains=domCondition)

            repeated_hrf = np.repeat(estimated_hrf, nbv).reshape(-1, nbv)
            outputs["hrf_mapped"] = xndarray(repeated_hrf, value_label="HRFs",
                                             axes_names=["time", "voxel"],
                                             axes_domains={"time": hrf_time})

            repeated_hrf_covar = np.repeat(np.diag(hrf_covariance), nbv).reshape(-1, nbv)
            outputs["hrf_variance_mapped"] = xndarray(repeated_hrf_covar,
                                                        value_label="HRFs covariance",
                                                        axes_names=["time", "voxel"],
                                                        axes_domains={"time": hrf_time})

            outputs['roi_mask'] = xndarray(np.zeros(nbv)+roiData.get_roi_id(),
                                        value_label="ROI",
                                        axes_names=['voxel'])

            outputs["density_ratio"] = xndarray(np.zeros(nbv)+density_ratio,
                                                value_label="Density Ratio",
                                                axes_names=["voxel"])

            outputs["ppm_a_nrl"] = xndarray(ppm_a_nrl, value_label="PPM NRL alpha fixed",
                                            axes_names=["voxel", "condition"],
                                            axes_domains=domCondition)

            outputs["ppm_g_nrl"] = xndarray(ppm_g_nrl, value_label="PPM NRL gamma fixed",
                                            axes_names=["voxel", "condition"],
                                            axes_domains=domCondition)

            h = estimated_hrf
            nrls = nrls.transpose()

            nvox = nrls.shape[1]
            nbconds = nrls.shape[0]
            ah = np.zeros((h.shape[0], nvox, nbconds))

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
            if (len(self.contrasts) >0) and self.computeContrast:
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

                outputs["ppm_a_contrasts"] = xndarray(ppm_a_contrasts,
                                                     value_label="PPM Contrasts alpha fixed",
                                                     axes_names=["voxel", "contrast"],
                                                     axes_domains=domContrast)

                outputs["ppm_g_contrasts"] = xndarray(ppm_g_contrasts,
                                                     value_label="PPM Contrasts alpha fixed",
                                                     axes_names=["voxel", "contrast"],
                                                     axes_domains=domContrast)


            ################################################################################
            # CONVERGENCE

            axes_names = ['duration']
            outName = 'Convergence_Labels'
            ax = np.arange(self.nItMax)*cTimeMean
            ax[:len(cTime)-1] = cTime[:-1]
            ad = {'duration':ax}
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
            outName = 'Convergence_NRL'
            c = np.zeros(self.nItMax)# -.001 #
            c[:len(cA)] = cA
            #ad = {'Conv_Criterion':np.arange(len(cA))}
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_A')

        ################################################################################
        # SIMULATION

        if self.simulation and self.fast:
            from pyhrf.stats import compute_roc_labels
            labels_vem_audio = roiData.simulation['labels'][0]
            labels_vem_video = roiData.simulation['labels'][1]

            M = labels.shape[0]
            K = labels.shape[1]
            J = labels.shape[2]
            true_labels = np.zeros((K,J))
            true_labels[0,:] = reshape(labels_vem_audio,(J))
            true_labels[1,:] = reshape(labels_vem_video,(J))
            newlabels = np.reshape(labels[:,1,:],(M,J))
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

            m = specData[0].min()
            import matplotlib.font_manager as fm
            figure(200)
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

            domCondition = {'condition':cNames}
            outputs['Truenrls'] = xndarray(true_nrls,value_label="True_nrls",
                                         axes_names=['condition','voxel'],
                                         axes_domains=domCondition)
            M = labels.shape[0]
            K = labels.shape[1]
            J = labels.shape[2]

            newlabels = np.reshape(labels[:,1,:],(M,J))

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

        # END SIMULATION
        ##########################################################################
        if self.fast:
            d = {'parcel_size':np.array([nvox])}
            outputs['analysis_duration'] = xndarray(np.array([self.analysis_duration]),
                                                axes_names=['parcel_size'],
                                                axes_domains=d)

        return outputs



# Function to use directly in parallel computation
def run_analysis(**params):
    # pyhrf.verbose.set_verbosity(1)
    # pyhrf.logger.setLevel(logging.INFO)
    fdata = params.pop('roi_data')
    # print 'doing params:'
    # print params
    vem_analyser = JDEVEMAnalyser(**params)
    return (dict([('ROI', fdata.get_roi_id())] + params.items()),
            vem_analyser.analyse_roi(fdata))
