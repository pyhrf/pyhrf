

from numpy import *
import numpy.matlib

import copy as copyModule

from pyhrf.jde.samplerbase import *
from pyhrf.jde.beta import *

from bigaussian import NRLSampler


class NRLARSampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels according
    to Salima Makni (ISBI 2006). Inherits the abstract class C{GibbsSamplerVariable}.
    #TODO : comment attributes
    """

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.cardClass = zeros((self.nbClasses, self.nbConditions), dtype=int)
        #self.cardCI = zeros(self.nbConditions, dtype=int)
        #self.cardCA = zeros(self.nbConditions, dtype=int)
        #self.voxIdxCI = range(self.nbConditions)
        #self.voxIdxCA = range(self.nbConditions)
        self.voxIdx = [range(self.nbConditions) for c in xrange(self.nbClasses)]

        self.cumulLabels = zeros((self.nbConditions,self.nbVox), dtype=int)
        self.meanClassAApost = empty((self.nbConditions,self.nbVox), dtype = float)
        self.meanClassIApost = empty((self.nbConditions,self.nbVox), dtype = float)
        self.varXjhtLambdaXjh = empty((self.nbVox), dtype=float)
        self.varXjhtLambdaeji = empty((self.nbVox), dtype=float)


    def samplingWarmUp(self, variables):
        """
        #TODO : comment
        """
        ##print 'NRLs warming up ...'

        # Compute precalculations :
        smplHRF = self.get_variable('hrf')
        smplHRF.checkAndSetInitValue(variables)
        smpldrift =  self.get_variable('drift')
        smpldrift.checkAndSetInitValue(variables)
        self.varYtilde = zeros((self.ny, self.nbVox), dtype=float)
        self.computeVarYTilde( smplHRF.varXh, smpldrift.varMBYPl )

        self.normHRF = smplHRF.norm
        #self.varClassIApost = empty((self.nbConditions,self.nbVox),dtype=float)
        #self.varClassAApost = empty((self.nbConditions,self.nbVox),dtype=float)
        #self.sigClassIApost = empty((self.nbConditions,self.nbVox),dtype=float)
        #self.sigClassAApost = empty((self.nbConditions,self.nbVox),dtype=float)
        #self.interMeanClassA = empty((self.nbConditions),dtype=float)
        shape = (self.nbClasses,self.nbConditions,self.nbVox)
        self.varClassApost = empty(shape, dtype=float)
        self.sigClassApost = empty(shape, dtype=float)
        self.meanClassApost = empty(shape, dtype=float)
        self.meanApost = empty((self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = zeros((self.nbConditions, self.nbVox), dtype=float)
        
        self.corrEnergies = zeros((self.nbConditions, self.nbVox), dtype=float)
 

    def computeVarYTilde(self, varXh,varMBYPl):
        for i in xrange(self.nbVox):
            repNRLi = numpy.matlib.repmat(self.currentValue[:,i], self.ny, 1)
            aijXjh = repNRLi * varXh
            self.varYtilde[:,i] = varMBYPl[:,i] - aijXjh.sum(axis=1)

    def sampleNextAlt(self, variables):
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        self.computeVarYTilde(varXh,varMBYPl)

    def computeMeanVarClassApost(self, j, variables):
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        #varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        #varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        #meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        
        reps = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        varInvAutoCorrNoise = variables[self.samplerEngine.I_NOISE_ARP].InvAutoCorrNoise
    
        nrls = self.currentValue
        varXhj = varXh[:,j]

        pyhrf.verbose(6, 'varInvAutoCorrNoise[:,:,0] :')
        pyhrf.verbose.printNdarray(6, varInvAutoCorrNoise[:,:,0])
        for i in xrange(self.nbVox):
            eji = self.varYtilde[:,i] + nrls[j,i]*varXhj
#            print 'size eji = ', shape(eji), 'size varXjh=', shape(varXhj)
            varXjhtLambda = dot(varXhj.transpose(),varInvAutoCorrNoise[:,:,i])/reps[i]
            pyhrf.verbose(6, 'varXjhtLambda :')
            pyhrf.verbose.printNdarray(6, varXjhtLambda)
            
            self.varXjhtLambdaXjh =  dot( varXjhtLambda, varXhj )
            self.varXjhtLambdaeji = dot( varXjhtLambda,eji )
            for c in xrange(self.nbClasses):
                self.varClassApost[c,j,i] = 1./(1./var[c,j] + self.varXjhtLambdaXjh)
                self.varClassApost[c,j,i] = self.sigClassApost[c,j,i]**0.5
            
                if c > 0: # assume 0 stands for inactivating class
                    self.meanClassApost[c,j,i] = ( mean[c,j]/var[c,j] + self.varXjhtLambdaeji) * self.varClassApost[c,j,i]
                else:
                    self.meanClassApost[c,j,i] = self.varClassApost[c,j,i] * self.varXjhtLambdaeji
            pyhrf.verbose(5,
                              'meanClassApost %d cond %d :'%(j,i))
            pyhrf.verbose.printNdarray(5,
                                       self.meanClassApost[:,j,i])
            
        #self.varClassAApost[j,i] = 1/( 1.0/varCA[j] + varXjhtLambdaXjh)
            #self.varClassIApost[j,i] = 1/( 1.0/varCI[j] + varXjhtLambdaXjh)
            #self.sigClassAApost[j,i] = self.varClassAApost[j,i]**0.5
            #self.sigClassIApost[j,i] = self.varClassIApost[j,i]**0.5
        #self.meanClassAApost[j,i] = self.interMeanClassA[j] + \
                                        #self.varClassAApost[j,i]*varXjhtLambdaeji
            #self.meanClassIApost[j,i] = self.varClassIApost[j,i]*varXjhtLambdaeji

    def sampleNextInternal(self, variables):
        #TODO : comment

        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        #varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        #varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        #meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        
        #varReps = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        varXh = self.get_variable('hrf').varXh
        varMBYPl = self.get_variable('drift').varMBYPl
        varInvAutoCorrNoise = variables[self.samplerEngine.I_NOISE_ARP].InvAutoCorrNoise
        varLambda = variables[self.samplerEngine.I_WEIGHTING_PROBA].currentValue # (note : lambda = python reserved keyword)

        self.computeVarYTilde(varXh,varMBYPl)
        #self.interMeanClassA = numpy.divide(meanCA, varCA)
#        self.computeComponentsApost(variables)
        
        self.labelsSamples = random.rand(self.nbConditions, self.nbVox)
        nrlsSamples = random.randn(self.nbConditions, self.nbVox)
        
        for j in xrange(self.nbConditions):
            self.computeMeanVarClassApost(j, variables)
            pyhrf.verbose(3,
                              'Sampling labels - cond %d ...'%j)
            if self.sampleLabelsFlag:
                #self.sampleLabels(j, varCI, varCA, meanCA)
                self.sampleLabels(j, variables)

            for c in xrange(self.nbClasses):
                putmask(self.sigApost[j,:], self.labels[j,:]==c,
                        self.sigClassApost[c,j,:])
                putmask(self.meanApost[j,:],self.labels[j,:]==c,
                        self.meanClassApost[c,j,:])

            oldVal = self.currentValue[j,:]
            add(multiply(nrlsSamples[j,:], self.sigApost[j,:]),
                self.meanApost[j,:],
                self.currentValue[j,:])
            
            pyhrf.verbose(6, 'All nrl cond %d:'%j)
            pyhrf.verbose.printNdarray(6, self.currentValue[j,:])
            pyhrf.verbose(4,
                          'nrl cond %d = %1.3f(%1.3f)'
                          %(j,self.currentValue[j,:].mean(),
                            self.currentValue[j,:].std()))
            for c in xrange(self.nbClasses):
                pyhrf.verbose(6, 'All nrl %s cond %d:' \
                              %(self.CLASS_NAMES[c],j))
                ivc = self.voxIdx[c][j]
                pyhrf.verbose.printNdarray(6,
                                           self.currentValue[j,ivc])
            
                pyhrf.verbose(4,
                              'nrl %s cond %d = %1.3f(%1.3f)'
                              %(self.CLASS_NAMES[c],j,
                                self.currentValue[j,ivc].mean(),
                                self.currentValue[j,ivc].std()))
        
        #sigApost = zeros(self.nbVox, dtype=float)            
            #putmask(sigApost, self.labels[j,:], self.sigClassAApost[j,:])
            #putmask(sigApost, 1-self.labels[j,:], self.sigClassIApost[j,:])
            
            #meanApost = zeros(self.nbVox, dtype=float)
            #putmask(meanApost,self.labels[j,:], self.meanClassAApost[j] )
            #putmask(meanApost,1-self.labels[j,:], self.meanClassIApost[j] )

            #pyhrf.verbose(6, 'meanApost :')
            #pyhrf.verbose.printNdarray(6, meanApost)

            #pyhrf.verbose(6, 'sigApost :')
            #pyhrf.verbose.printNdarray(6, sigApost)
            #pyhrf.verbose(6, 'All nrl cond %d:'%j)
            #pyhrf.verbose.printNdarray(6,
                                       #self.currentValue[j,:])
            #pyhrf.verbose(4,
                          #'nrl cond %d = %1.3f(%1.3f)'
                          #%(j,self.currentValue[j,:].mean(),
                            #self.currentValue[j,:].std()))

            
            #pyhrf.verbose(6, 'All nrl activ cond %d:'%j)
            #pyhrf.verbose.printNdarray(6,
                                       #self.currentValue[j,self.voxIdxCA[j]])
            
            #pyhrf.verbose(4,
                          #'nrl activ cond %d = %1.3f(%1.3f)'
                          #%(j,self.currentValue[j,self.voxIdxCA[j]].mean(),
                            #self.currentValue[j,self.voxIdxCA[j]].std()))

            #pyhrf.verbose(6, 'All nrl inactiv cond %d:'%j)
            #pyhrf.verbose.printNdarray(6,
                                       #self.currentValue[j,self.voxIdxCI[j]])

            #pyhrf.verbose(4,
                          #'nrl inactiv cond %d = %1.3f(%1.3f)'
                          #%(j,self.currentValue[j,self.voxIdxCI[j]].mean(),
                            #self.currentValue[j,self.voxIdxCI[j]].std()))
            
            
            #self.currentValue[j,:] = multiply(nrlsSamples[j,:], sigApost) + meanApost
            self.computeVarYTilde(varXh,varMBYPl)

        self.countLabels(self.labels, self.voxIdx, self.cardClass)
#        self.normHRF = self.get_variable('hrf').norm

    def cleanMemory(self):
        # clean memory of temporary variables :
        del self.varClassApost
        del self.sigClassApost
        del self.varXjhtLambdaeji
        del self.varXjhtLambdaXjh
        del self.sigApost
        del self.meanApost
        del self.varYtilde
        if hasattr(self,'labelsSamples'):
            del self.labelsSamples
        del self.corrEnergies
        del self.labels
        del self.voxIdx
        #del self.gridLnZ
        # clean memory of temporary variables :
        #del self.varClassIApost
        #del self.varClassAApost
        #del self.sigClassIApost
        #del self.sigClassAApost
        #del self.interMeanClassA
        #del self.meanClassAApost 
        #del self.meanClassIApost 



