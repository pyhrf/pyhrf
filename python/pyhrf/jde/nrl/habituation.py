

import copy as copyModule
from numpy import *
import numpy.matlib
from numpy.matlib import repmat

from pyhrf.ndarray import xndarray

import pyhrf
#from pyhrf import xmlio
from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
from pyhrf.jde.samplerbase import *
from bigaussian import NRLSampler
from pyhrf.jde.intensivecalc import sampleSmmNrl

class NRLwithHabSampler(NRLSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels in combination
    with habituation speed factor sampling. The underlying model is exponential decaying
    #TODO : comment attributes
    """

    P_TRUE_HABITS = 'trueHabits'
    P_SAMPLE_HABITS = 'sampleHabit'## flag for habituation speed sampling
    P_HABITS_INI = 'habitIni'## flag for definition of initial habituation speed value
    P_HAB_ALGO_PARAM = 'paramLexp' ##lambda-like parameter of the Laplacian distribution for function habitSample()
    P_OUTPUT_RATIO = 'outputRatio'
    
    
    defaultParameters = copyModule.copy(NRLSampler.defaultParameters)
    defaultParameters.update( {
                    #parameters for habituation sampling
                    P_HABITS_INI : None,
                    P_SAMPLE_HABITS : 1,
                    P_HAB_ALGO_PARAM : 5.,
                    P_OUTPUT_RATIO : 0,
                })
    
    parametersToShow = NRLSampler.parametersToShow + [P_SAMPLE_HABITS , P_HABITS_INI , P_HAB_ALGO_PARAM , P_OUTPUT_RATIO]
    parametersComments = NRLSampler.parametersComments
    parametersComments[P_HAB_ALGO_PARAM] = 'lambda-like parameter of the Laplacian distribution in habit sampling\n recommended between 1. and 10.'
    # other class attributes
#    nbClasses = 2

    
    
    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(), xmlLabel=None, xmlComment=None):
        
        NRLSampler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)
     
# Comment setting of beta parameter since it is properly handled in the base class
#        self.beta = self.parameters[self.P_BETA]

        ## habituation speed sampling parameters
        self.sampleHabitFlag = self.parameters[self.P_SAMPLE_HABITS]        
        self.habits = self.parameters[self.P_HABITS_INI] #Habituation initialization
        if self.parameters.has_key(self.P_TRUE_HABITS): 
            self.trueHabits = params[self.P_TRUE_HABITS] #load true habits if exist
        else :
            self.trueHabits = None
        self.Lexp = self.parameters[self.P_HAB_ALGO_PARAM] # parametre pour la Laplacienne tronquee

        self.habitsHistory = None                           # pour la sauvegarde des donnees
        self.outputRatio = self.parameters[self.P_OUTPUT_RATIO] #affiche ou pas les ratio
        if self.outputRatio :
            self.ratioHistory = None                #pour voir les ratio
            self.ratiocourbeHistory = None            #pour voir la courbe des ratio

        self.labelsColors = self.parameters[self.P_LABELS_COLORS]


    def linkToData(self, dataInput):
    
        NRLSampler.linkToData(self,dataInput)

        self.varSingleCondXtrials=self.dataInput.varSingleCondXtrials ## recuperation de X (matrice (nbCond * ny * nh) ) 
    
        #pour les onsets
        #self.nbTrials=zeros((self.nbSessions,self.nbConditions), dtype=int)
        #for isess in xrange(self.nbSessions):

        self.onsets = self.dataInput.onsets
        self.lastonset = 0.
        self.deltaOns = {}
        self.nbTrials=zeros(self.nbConditions, dtype=int)
        for nc in xrange(self.nbConditions):
            self.deltaOns[nc] = numpy.diff(self.onsets[nc])
            self.nbTrials[nc]=len(self.onsets[nc])
            self.lastonset= max(self.lastonset, self.onsets[nc][self.nbTrials[nc]-1])
            
        
        # astuce pour les donnees reelles -> pour remettre 'a zero' tout les 4 occurences
            for trial in xrange(self.nbTrials[nc]-1):
                if (((trial + 1)%4) == 0):
                    self.deltaOns[nc][trial] = 100.
                else:
                    # on divise les deltaOns par 4. pour prendre plus en compte les valeurs anterieures d'habituation
                    self.deltaOns[nc][trial] = self.deltaOns[nc][trial]/4.   
        
        # calcul du dernier onset -> pour afficher les timesNRLs
        self.lastonset = int(self.lastonset+1)
        #print self.lastonset
    
        # determination du nbGammas utile dans spExtract -> mais il y a une erreur pour Xmask (dans sparsedot
        self.nbGammas = range(self.nbConditions) 
        self.nnulls = range(self.nbConditions)
        self.Xmask= range(self.nbConditions)
        for nc in xrange(self.nbConditions) :
            self.Xmask[nc] = range(self.nbTrials[nc])    
            self.nbGammas[nc] = zeros(self.nbTrials[nc], dtype = int)
            self.nnulls[nc] = zeros((self.varSingleCondXtrials[nc,:,:]==0).sum())
            for i in xrange(self.nbTrials[nc]):
                #self.nbGammas[j][i] = (self.varSingleCondXtrials[j,:,:] == (i + 1)).sum()
                self.Xmask[nc][i]=transpose(where(self.varSingleCondXtrials[nc,:,:]==(i+1)))    
                self.nbGammas[nc][i] = shape(self.Xmask[nc][i])[0]
    
        pyhrf.verbose(4, 'deltaOns :' )
        pyhrf.verbose.printNdarray(4, self.deltaOns )
     


    def checkAndSetInitValue(self, variables):
        
        # init NRL and Labels
        NRLSampler.checkAndSetInitValue(self,variables)
        self.checkAndSetInitHabit(variables)
    
    def checkAndSetInitHabit(self, variables) :
     
        # init habituation speed factors and time-varying NRLs
        if self.habits == None :  # if no habituation speed specified 
            if not self.sampleHabitFlag :
                if self.dataInput.simulData != None : # Attention: on a un probleme si on fait tourner ce modele sur des donnees simulees par le modele stationnaire. Dans ce cas, il faut forcer ici a passer et prendre des habits nulles
                   ## using simulated Data for HABITUATION sampling
                   print "load Habituation from simulData", self.dataInput.simulData.habitspeed.data
                   self.habits = self.dataInput.simulData.habitspeed.data
                   self.timeNrls = self.dataInput.simulData.nrls.timeNrls
                   self.Gamma = self.setupGamma()
                else : # sinon, on prend que des zeros (modele stationnaire)
                    self.habits = numpy.zeros((self.nbConditions, self.nbVox), dtype=float)
                    self.setupTimeNrls()
            else:
                pyhrf.verbose(2, "Uniform set up of habituation factors")
                habitCurrent = numpy.zeros((self.nbConditions, self.nbVox), dtype=float)
                for nc in xrange(self.nbConditions):
                    habitCurrent[nc,self.voxIdx[1][nc]] = numpy.random.rand(self.cardClass[1][nc])
                self.habits = habitCurrent

        if self.outputRatio :
            self.ratio = zeros((self.nbConditions, self.nbVox, 2), dtype = float)
            self.ratiocourbe = zeros((self.nbConditions, self.nbVox, 100, 5), dtype = float)
            self.compteur = numpy.zeros((self.nbConditions, self.nbVox), dtype=float)
         
        self.setupTimeNrls()
        pyhrf.verbose(4, 'habituation initiale')
        pyhrf.verbose.printNdarray(5, self.habits)
        
    
    def initObservables(self):
        NRLSampler.initObservables(self)
        #sauvegarde des habits et nrls
        self.nrlsHistory = None
        self.meanHabits = None
        self.cumulHabits = zeros((self.nbConditions,self.nbVox), dtype=float) # Toutes les habituations par voxels
        self.meanHabitsCond = zeros((self.nbClasses, self.nbConditions,self.nbVox), dtype=float)
        self.cumulHabitsCond = zeros((self.nbClasses, self.nbConditions,self.nbVox), dtype=float)
        self.voxelActivity = zeros((self.nbClasses, self.nbConditions,self.nbVox), dtype=int)
        self.obs = 0


    def updateObsersables(self):
        NRLSampler.updateObsersables(self)
        self.obs += 1
        self.cumulHabits += self.habits
       #self.meanHabits = self.cumulHabits / self.nbItObservables
        self.meanHabits = self.cumulHabits / self.obs
        temp = zeros((self.nbConditions,self.nbVox), dtype=float)
        for c in xrange(self.nbClasses):
            putmask(temp, self.labels==c, self.habits)
            self.cumulHabitsCond[c,:,:] += temp 
            self.voxelActivity[c,:,:] += self.labels==c
        self.meanHabitsCond = self.cumulHabitsCond / self.voxelActivity


    def cleanObservables(self):
        NRLSampler.cleanObservables(self)
        del self.cumulHabits
    #self.cleanMemory()    
    
    
    def saveCurrentValue(self):
        
        NRLSampler.saveCurrentValue(self)
    
        if self.keepSamples:
            if (self.iteration % self.sampleHistoryPace) == 0:
                #if self.habitsHistory != None:
                    #self.habitsHistory = concatenate((self.habitsHistory,[self.habits]))
                #else:
                    #self.habitsHistory = [self.habits]
                if self.habitsHistory != None:
                    self.habitsHistory = concatenate((self.habitsHistory,[self.habits]))
                else:
                    self.habitsHistory = array([self.habits])
                if self.nrlsHistory != None:
                    self.nrlsHistory = concatenate((self.nrlsHistory,[self.currentValue]))
                else:
                    self.nrlsHistory = array([self.currentValue])
        if self.outputRatio :
            if self.ratioHistory != None:
                self.ratioHistory = concatenate((self.ratioHistory,[self.ratio]))
            else:
                self.ratioHistory = array([self.ratio])
            if self.ratiocourbeHistory != None:
                self.ratiocourbeHistory = concatenate((self.ratiocourbeHistory,[self.ratiocourbe]))
            else:
                self.ratiocourbeHistory = array([self.ratiocourbe])
    
    
    def samplingWarmUp(self, variables):
        
        pyhrf.verbose(2, 'Habit Sampling Warm up')
    
        # issu de NRLSampler
        #---------------------------------------------------------------
        
        smplHRF = variables[self.samplerEngine.I_HRF]
        varHRF = smplHRF.currentValue
        self.nh = len(varHRF)            # taille de la HRF
    
        self.varXh = zeros((self.nbVox, self.ny, self.nbConditions), dtype = float)
        self.aXh = zeros((self.ny , self.nbVox , self.nbConditions), dtype = float)
        #self.vycArray = zeros((self.nbVox, self.ny, self.nbConditions))
        self.sumaX = zeros((self.ny,self.nh , self.nbVox), dtype = float)

        #self.sumaXQ = zeros((self.ny,self.nh , self.nbVox), dtype = float)
    
        self.updateXh(varHRF)
    
        self.varYtilde = zeros((self.ny, self.nbVox), dtype=float)
        self.sumaXh=zeros((self.ny, self.nbVox), dtype = float)
        self.updateYtilde()    
            
        self.varXhtQ = zeros((self.nbVox , self.nbConditions , self.ny),dtype=float)
    
        # utilise par noise.py
        #self.sumaXtQ = zeros((self.nbVox, self.nh, self.ny) , dtype=float)
        #self.sumaXtQaX = zeros((self.nbVox, self.nh, self.nh), dtype=float)
        self.sumaXhtQaXh = zeros(self.nbVox , dtype= float)

        self.varClassApost = zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                   dtype=float)
        self.sigClassApost = zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                   dtype=float)
        self.meanClassApost = zeros((self.nbClasses,self.nbConditions,self.nbVox),
                                   dtype=float)
        self.meanApost = zeros((self.nbConditions, self.nbVox), dtype=float)
        self.sigApost = zeros((self.nbConditions, self.nbVox), dtype=float)
        
        self.interMeanClassA = zeros((self.nbConditions,self.nbVox),dtype=float)

        self.corrEnergies = zeros((self.nbConditions, self.nbVox), dtype=float)
        
        self.aa = zeros((self.nbConditions, self.nbConditions, self.nbVox),
                        dtype=float)
        self.computeAA(self.currentValue, self.aa)

        self.iteration = 0

    
    def updateXh(self, varHRF):
        temp = empty((self.nbVox, self.nbConditions), dtype = float)
        self.sumaX = zeros((self.ny,self.nh , self.nbVox), dtype = float)
        #self.sumaXQ = zeros((self.ny,self.nh , self.nbVox), dtype = float)    
        for nv in xrange(self.nbVox):
            for nc in xrange(self.nbConditions):
                self.updateGammaTimeNRLs(nc,nv)
                Xtilde = self.spExtract(self.Gamma[nc][nv,:],self.varSingleCondXtrials[nc,:,:], nc)
                aXtilde = self.spExtract(self.timeNrls[nc][nv,:], self.varSingleCondXtrials[nc,:,:], nc)
                self.sumaX[:,:,nv] += Xtilde
                self.varXh[nv,:,nc] = dot(Xtilde , varHRF)
        
                # ici, j ai fait un essai avec sparsedot, mais ca ne marche pas...
                #Xhtemp = zeros(self.ny, dtype = float)
                #aXhtemp = zeros(self.ny, dtype = float)
                #for t in xrange(self.nbTrials[nc]):
                    #Xhtemp += sparsedotdimun(self.Gamma[nc][nv,t], varHRF, self.Xmask[nc][t], self.ny)
                    #aXhtemp += sparsedotdimun(self.timeNrls[nc][nv,t], varHRF, self.Xmask[nc][t], self.ny)            
                    ##self.sumaXQ[:,:,nv] += sparsedot(self.timeNrls[nc][nv,t], self.dataInput.delta, self.Xmask[nc][t], (self.ny, self.nh))
                #self.varXh[nv,:,nc]= Xhtemp    
            
                self.aXh[:,nv,nc] = dot(aXtilde, varHRF)
                #self.aXh[:,nv,nc] = aXhtemp

                temp[nv,nc] = dot(dot(self.aXh[:,nv,nc], self.dataInput.delta), self.aXh[:,nv,nc])
                
                #aXQaX = zeros((self.nh, self.nh), dtype=float)
                ##aXQaX = self.bilinearsparsedot(self.timeNrls[nc][nv,:], self.dataInput.delta, self.nh, nc)
                #temp[nv,nc] = dot(dot(varHRF, aXQaX), varHRF)
        
        #utile pour noise.py
        self.sumaXh = sum(self.aXh,axis=2)
        self.sumaXhtQaXh = sum(temp, axis=1)

    
    def updateYtilde(self): # doit etre precede d un updateXh pour avoir sumaXh
        self.varYtilde = self.dataInput.varMBY - self.sumaXh
    
    
    def setupTimeNrls(self):## initialisation of temporal-varying  NRLs (tout a 0)
        self.timeNrls = range(self.nbConditions)
        self.Gamma = range(self.nbConditions)
        for nc in xrange(self.nbConditions):
            self.timeNrls[nc] =  numpy.zeros((self.nbVox,self.nbTrials[nc]), dtype=float) 
            self.Gamma[nc] =numpy.zeros((self.nbVox,self.nbTrials[nc]), dtype=float)
            for nv in xrange(self.nbVox) :
                self.updateGammaTimeNRLs(nc,nv)

    def updateGammaTimeNRLs(self,nc,nv) :
    
        self.Gamma[nc][nv,0] = 1.
        self.timeNrls[nc][nv,0] = self.currentValue[nc,nv]
        skk=0.
        for k in xrange(1,self.nbTrials[nc]):
            skk=(self.habits[nc,nv]**self.deltaOns[nc][k-1])*(skk+self.timeNrls[nc][nv,k-1])
            self.Gamma[nc][nv,k]=1./(1.+skk)
            self.timeNrls[nc][nv,k]=self.Gamma[nc][nv,k]*self.currentValue[nc,nv]
        
    def setupGamma(self) :
        # lorsqu on connait les timeNRLs
        Gamma = range(self.nbConditions)
        for nc in xrange(self.nbConditions):
            Gamma[nc] =numpy.zeros((self.nbVox,self.nbTrials[nc]), dtype=float)
            for k in xrange(self.nbTrials[nc]):
                Gamma[nc][:,k] = self.timeNrls[nc][:,k]/self.currentValue[nc,:]
        return Gamma
        
    
    
    def computeVarXhtQ(self, Q):
        for nc in xrange(self.nbConditions):
            self.varXhtQ[:,nc,:] = dot(self.varXh[:,:,nc].swapaxes(0,1).transpose(), Q)
            #self.varXhtQ[:,nc,:] = dot(self.varXh[:,:,nc], Q)
            #self.varXhtQXh[:, nc] = diag(dot(self.varXhtQ[:,nc,:],self.varXh[:,:,nc].swapaxes(0,1)))
            
    
     
    def computeComponentsApost(self, variables, j, XhtQXh):
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        nrls = self.currentValue
        
        #for j in xrange(self.nbConditions):
        gTQgjrb = XhtQXh[:,j]/rb   # de taille nbVox
    
        ej = self.varYtilde + repmat(nrls[j,:], self.ny, 1) * self.varXh[:,:,j].swapaxes(0,1)
        numpy.divide(diag(dot(self.varXhtQ[:,j,:],ej)), rb, self.varXjhtQjeji)

        for c in xrange(self.nbClasses):            #ici classe: 0 (inactif) ou 1 (actif)
            self.varClassApost[c,j,:] = 1./(1./var[c,j] + gTQgjrb)
            numpy.sqrt(self.varClassApost[c,j,:], self.sigClassApost[c,j,:])
            if c > 0: # assume 0 stands for inactivating class
                numpy.multiply(self.varClassApost[c,j,:],
                               add(mean[c,j]/var[c,j], self.varXjhtQjeji),
                               self.meanClassApost[c,j,:])
            else:
                multiply(self.varClassApost[c,j,:], self.varXjhtQjeji,
                         self.meanClassApost[c,j,:])
                
            pyhrf.verbose(5, 'meanClassApost %d cond %d :'%(c,j))
            pyhrf.verbose.printNdarray(5, self.meanClassApost[c,j,:])

    def habitCondSampler(self, j, rb, varHRF):
        
        ##Samples habituation factors
        for i in xrange(self.nbVox):## loops over voxels
            ##compute new voxel-dependant design matrix
            #for j in xrange(self.nbConditions):## loops over conditions
            pyhrf.verbose(5, 'Condition %i :'%j)
            if (self.labels[j,i]==1):## sampling the habituation factors   
                eji = self.varYtilde[:,i] + self.aXh[:,i,j]
                    
            ## Generate a candidate for the habitutation parameter in voxel i
            newHabit,KhabNew=LaplacianPdf(self.Lexp, self.habits[j,i],0.,1.,1)
            
            if self.keepSamples and self.outputRatio : # pour afficher les courbes de ratio
                if (self.iteration % self.sampleHistoryPace) == 0:
                    Dif2 = eji - self.aXh[:,i,j]
                    for iii in range(100) : # on calcule les ratio tous les 1/100
                        ii=iii/100.
                        tirage=0.
                        vrai=0.
                        Gamma_ii,timeNrls_ii=subcptGamma(self.currentValue[j,i],ii,self.nbTrials[j],self.deltaOns[j][:])
                        XsumWT_ii = self.spExtract(timeNrls_ii,self.varSingleCondXtrials[j], j)
                        varG_ii = numpy.dot(XsumWT_ii,varHRF)
                        XsumWTb_ii = self.spExtract(Gamma_ii,self.varSingleCondXtrials[j], j) 
                        varGb_ii = numpy.dot(XsumWTb_ii,varHRF)
                        
                        Dif1 = eji - varG_ii
                        
                        Dif12QDif12=dot(dot(Dif1.transpose(),self.dataInput.delta),Dif1) - dot(dot(Dif2.transpose(),self.dataInput.delta),Dif2)
                        #Dif12QDif12=dot(Dif1.transpose(),Dif1) - dot(Dif2.transpose(),Dif2)
                            
                        # probleme observe lorsque le ratio est trop grand
                        if Dif12QDif12/rb[i]>-1400. :
                            Ratio1 = math.exp( -.5*Dif12QDif12/rb[i])
                        else :
                            Ratio1 = 1000000.
                        Ratio2 = (2. - math.exp(-self.Lexp * self.habits[j,i]) - math.exp(self.Lexp *(self.habits[j,i]-1.)))/(2. - math.exp(-self.Lexp*ii) - math.exp(self.Lexp * (ii-1.)))
                        if abs(ii-newHabit)<0.005 :
                            tirage = min(Ratio1*Ratio2, 1)
                        
                        if abs(ii-self.trueHabits[j,i])<0.005 :
                            vrai = 1.
                        if abs(ii-self.habits[j,i])<0.005 :
                            vrai = 0.7                                    
                                        
                        self.ratiocourbe[j,i,iii,:]=[min(Ratio1,10.), Ratio2, min(Ratio1*Ratio2, 1), tirage, vrai]
            #calcul gamma - habituation factor
            GammaNew,timeNrlsNew=subcptGamma(self.currentValue[j,i],newHabit,self.nbTrials[j],self.deltaOns[j][:])
            ## compute key quantities for the new candidate
            XsumWT_x = self.spExtract(timeNrlsNew,self.varSingleCondXtrials[j], j)
            varG_x = numpy.dot(XsumWT_x,varHRF)
            XsumWTb_x = self.spExtract(GammaNew,self.varSingleCondXtrials[j], j) 
            varGb_x = numpy.dot(XsumWTb_x,varHRF)

            Dif1 = eji - varG_x
            Dif2 = eji - self.aXh[:,i,j]
#           Dif2 = self.varYtilde[:,i]
            
            #Dif12QDif12=dot(Dif1.transpose(),Dif1) - dot(Dif2.transpose(),Dif2)
            Dif12QDif12=dot(dot(Dif1.transpose(),self.dataInput.delta),Dif1) - dot(dot(Dif2.transpose(),self.dataInput.delta),Dif2)
            
            
            # probleme observe lorsque le ratio est trop grand -> on tronque dans ce cas
            if Dif12QDif12/rb[i]>-1400. :
                Ratio1 = math.exp( -.5*Dif12QDif12/rb[i])
            else :
                Ratio1 = 1000000. # assure que Alpha = 1 
                    
            Ratio2 = (2. - math.exp(-self.Lexp * self.habits[j,i]) - math.exp(self.Lexp * (self.habits[j,i]-1.)))/(2. - math.exp(-self.Lexp*newHabit) - math.exp(self.Lexp * (newHabit-1.)))
                    
            # TODO pour gagner du temps -> prendre KhabNew issu de LaplacianPdf
            # Ratio2 = self.Khab[j,i] / KhabNew   # probleme car KhabNew est en fait l'ancienne valeur
            
            # sauvegarde du ratio
            if self.outputRatio :
                self.ratio[j,i,:]=[Ratio1, Ratio2]
            
            ## Compute the acceptation rate of the MH algo
            Alpha = min(Ratio1*Ratio2, 1)
            u=numpy.random.rand()
            if (u<=Alpha):
                self.habits[j,i]=newHabit
            
                # sauvegarde le nombre de fois que l'habituation a change
                if self.keepSamples and self.outputRatio :
                    self.compteur[j,i] += 1
            
            self.updateGammaTimeNRLs(j,i)            
        
        else:  # Inactiv voxel
            self.habits[j,i]=0.
            self.updateGammaTimeNRLs(j,i)            

    def sampleNrlsSerial_bak(self, rb, h, varLambda, varCI, varCA,
                 meanCA, varXhtQXh, variables):

        for j in xrange(self.nbConditions):
            for i in random.permutation(self.nbVox):
                if self.iteration > 200:
                    pyhrf.verbose(4,'Doing vox %d, cond %d' %(i,j))
                    pyhrf.verbose(4,'-----------------------------')
                self.computeComponentsApostInVox(variables, j, i, gTQg)
                if self.sampleLabelsFlag:
                    self.sampleLabel(j, i, variables)
                label = self.labels[j,i]
                oldVal = self.currentValue[j,i]
    
    def sampleNrlsSerial(self, varXh, rb, h, varCI,
                         varCA, meanCA, variables):

        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        #varXh = variables[self.samplerEngine.I_HRF].varXh
        nrls = self.currentValue
        #gTQgjrb = gTQg[j]/rb[i]
        neighbours = self.dataInput.neighboursIndexes

        for j in xrange(self.nbConditions):
        #for j in random.permutation(self.nbConditions):
##             print 'self.nbVox : ', self.nbVox
            voxOrder = random.permutation(self.nbVox)
##             print 'voxOrder :', voxOrder
##                 print 'varXh : ', varXh.shape, varXh.dtype
##                 print 'neighbours :', neighbours
##                 print 'Components :'
##                 print 'mCA =', mean[self.L_CA],'vCA =', var[self.L_CA]
##                 print 'mCI =', mean[self.L_CI],'vCI =', var[self.L_CI]
##                 print 'beta = ', 
            #print 'labelsSamples, cond %d -> mean=%f' %(j,self.labelsSamples[j,:].mean())
            betaj = self.samplerEngine.getVariable('beta').currentValue[j]
            #print 'self.varYtilde:', self.varYtilde.shape
            sampleSmmNrl(voxOrder, rb, neighbours, self.varYtilde,
                         self.labels[j,:], varXh[:,:,j],
                         self.currentValue[j,:],
                         self.nrlsSamples[j,:], self.labelsSamples[j,:],
                         self.varXhtQ[:,j,:], array([0.]), betaj,
                         mean[:,j], var[:,j], self.nbClasses, 
                         self.sampleLabelsFlag+0, self.iteration, j)
            #print 'self.currentValue[j,:] :'
            #print self.currentValue[j,:]
            #print 'self.labels[j,:]:'
            #print self.labels[j,:]
            #sys.exit(1)
            self.countLabels(self.labels, self.voxIdx, self.cardClass)



    def sampleNrlsParallel(self, rb, h, varLambda, varCI, varCA,
                           meanCA, varXhtQXh, variables):
        
        pass
    
    def habitCondSamplerParallel(self, rb, h):
        pass
    
    def habitCondSamplerSerial(self, rb, h):
        pass
    

    def computeVarYTildeHab(self, varXh):
        #print 'varXh.shape:', varXh.shape
        #print 'self.varYtilde:', self.varYtilde.shape
        for i in xrange(self.nbVox):
            repNRLi = repmat(self.currentValue[:,i], self.ny, 1)
            aijXjh = repNRLi * varXh[i,:,:]
            #print 'aijXjh:', aijXjh.shape
            self.varYtilde[:,i] = self.dataInput.varMBY[:,i] - aijXjh.sum(axis=1)


    def computeVarYTildeHabOld(self, varXh):
        # yTilde_j = y_j - sum_m(a_j^m X^m h)
        pyhrf.verbose(5,'computeVarYTildeOpt...')
        pyhrf.verbose(5,'varXh:' +str(varXh.shape))
        self.aXh = repmat(varXh,self.nbVox,1).reshape(self.nbVox,
                                                      self.ny,
                                                      self.nbConditions)
        
        self.aXh = self.aXh.swapaxes(0,1).swapaxes(1,2)
        self.aXh *= self.currentValue
        self.aXh.sum(1, out=self.sumaXh)
        pyhrf.verbose(5,'sumaXh %s' %str(self.sumaXh.shape))
        pyhrf.verbose.printNdarray(5, self.sumaXh)
        
        numpy.subtract(self.dataInput.varMBY, self.sumaXh, self.varYtilde)

        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde.shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde)

    def sampleNextInternal(self, variables):
        #TODO : comment
    
        # load mixture params et mixture weight
        sIMixtP = variables[self.samplerEngine.I_MIXT_PARAM]
        varCI = sIMixtP.currentValue[sIMixtP.I_VAR_CI]
        varCA = sIMixtP.currentValue[sIMixtP.I_VAR_CA]
        meanCA = sIMixtP.currentValue[sIMixtP.I_MEAN_CA]
        varLambda = variables[self.samplerEngine.I_WEIGHTING_PROBA].currentValue
        # load noise
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        # load hrf
        sHrf = variables[self.samplerEngine.I_HRF]
        #varXh = sHrf.varXh
        h = sHrf.currentValue
        #varHRF = variables[self.samplerEngine.I_HRF].currentValue
        pyhrf.verbose(2, 'iteration %i' %self.iteration)
            
        self.updateXh(h)#-> update self.varXh
        self.updateYtilde()
        self.computeVarXhtQ(self.dataInput.delta)
            
        self.labelsSamples = random.rand(self.nbConditions, self.nbVox)
        self.nrlsSamples = random.randn(self.nbConditions, self.nbVox)
        
        varXhtQXh = zeros((self.nbVox, self.nbConditions), dtype=float)
        for i in xrange(self.nbVox):
            varXhtQXh[i,:] = diag(dot(self.varXhtQ[i,:,:], self.varXh[i,:,:]))
            
        #Calcul des variables a post regroupe dans self.computeComponentsApost
        #self.computeLambdaAPost(varCI, varCA, varLambda)
        
        if self.samplerEngine.getVariable('beta').currentValue[0] <= 0:
            self.sampleNrlsParallel(rb, h, varLambda, varCI,
                                    varCA, meanCA, varXhtQXh, variables)
            self.computeVarYTildeHab(self.varXh)
            if self.sampleHabitFlag:
                pyhrf.verbose(3, '(it %i) Habituation conditional sampling parallel mode ...')
                self.habitCondSamplerParallel(rb, h)
                self.computeVarYTildeHab(self.varXh)
                #self.computeVarYTildeHab(self.varXh)
        else:
            #print 'beta >0 ...'
            #gTQg = diag(dot(self.varXhtQ,self.varXh))
            #self.sampleNrlsSerial(rb, h, varLambda, varCI,
            #                      varCA, meanCA, varXhtQXh, variables)
            self.sampleNrlsSerial(self.varXh, rb, h, varCI,
                                  varCA, meanCA, variables)
            self.computeVarYTildeHab(self.varXh)
            #self.computeVarYTildeHab(self.varXh)
            if self.sampleHabitFlag:
                pyhrf.verbose(3, '(it %i) Habituation conditional sampling serial mode ...')
                self.habitCondSamplerSerial(rb, h)
                #self.computeVarYTildeOpt(self.varXh)
            self.saveSamples()
        self.updateXh(h)
        self.updateYtilde()
        self.computeAA(self.currentValue, self.aa)        
            
        pyhrf.verbose(4, 'NRLS current Value: ')
        pyhrf.verbose.printNdarray(4, self.currentValue)
        
        for j in xrange(self.nbConditions):
            pyhrf.verbose(4, 'All nrl cond %d:'%j)
            pyhrf.verbose.printNdarray(4, self.currentValue[j,:])
            pyhrf.verbose(4, 'nrl cond %d = %1.3f(%1.3f)'
                          %(j,self.currentValue[j,:].mean(),
                            self.currentValue[j,:].std()))
            for c in xrange(self.nbClasses):
                pyhrf.verbose(4, 'All nrl %s cond %d:' \
                                  %(self.CLASS_NAMES[c],j))
                ivc = self.voxIdx[c][j]
                pyhrf.verbose.printNdarray(4, self.currentValue[j,ivc])
                
                pyhrf.verbose(4, 'nrl %s cond %d = %1.3f(%1.3f)'
                              %(self.CLASS_NAMES[c],j,
                                self.currentValue[j,ivc].mean(),
                                self.currentValue[j,ivc].std()))
    
        self.iteration += 1 #TODO : factorize !!
        
        
        #for j in random.permutation(self.nbConditions):
                ##self.computeVarApostSimple(j, varCI, varCA, meanCA, rb, varLambda)
            #self.computeComponentsApost(variables, j, varXhtQXh)
            
            #pyhrf.verbose(6, 'self.varClassIApost %s :' %str(self.varClassApost[0].shape))
                #pyhrf.verbose.printNdarray(6,self.varClassApost[0])
                #pyhrf.verbose(6, 'self.varClassAApost %s :' %str(self.varClassApost[1].shape))
            #pyhrf.verbose.printNdarray(6,self.varClassApost[1])
    
            
            ## sampling labels
            #if self.sampleLabelsFlag:
                    #pyhrf.verbose(3,'(It %i) Sampling labels - cond %d ...'%(self.iteration,j))
                    #self.sampleLabels(j, variables)
                    #pyhrf.verbose(3,'(It %i) Sampling labels done!' %self.iteration)
                    #self.countLabels(self.labels, self.voxIdx, self.cardClass)        
    
                
            #for c in xrange(self.nbClasses):
                    #putmask(self.sigApost[j,:], self.labels[j,:]==c,
                            #self.sigClassApost[c,j,:])
                    #putmask(self.meanApost[j,:],self.labels[j,:]==c,
                            #self.meanClassApost[c,j,:])
            
            ##SAMPLING NRLs
                #add(multiply(nrlsSamples[j,:], self.sigApost[j,:]), self.meanApost[j,:], self.currentValue[j,:])
                
            #pyhrf.verbose(6, 'All nrl cond %d:'%j)
                #pyhrf.verbose.printNdarray(6, self.currentValue[j,:])
                #pyhrf.verbose(5,
                              #'nrl cond %d = %1.3f(%1.3f)'
                              #%(j,self.currentValue[j,:].mean(),
                                #self.currentValue[j,:].std()))
    
                #pyhrf.verbose(6, 'All nrl activ cond %d:'%j)
                #pyhrf.verbose.printNdarray(6,
                                           #self.currentValue[j,self.voxIdx[1][j]])
                
                #pyhrf.verbose(5,
                              #'nrl activ cond %d = %1.3f(%1.3f)'
                              #%(j,self.currentValue[j,self.voxIdx[1][j]].mean(),
                                #self.currentValue[j,self.voxIdx[1][j]].std()))
    
                #pyhrf.verbose(6, 'All nrl inactiv cond %d:'%j)
                #pyhrf.verbose.printNdarray(6,
                                           #self.currentValue[j,self.voxIdx[0][j]])
    
                #pyhrf.verbose(5,
                              #'nrl inactiv cond %d = %1.3f(%1.3f)'
                              #%(j,self.currentValue[j,self.voxIdx[0][j]].mean(),
                                #self.currentValue[j,self.voxIdx[0][j]].std()))
            
            
            ## sampling habits
            #if self.sampleHabitFlag:
                    #pyhrf.verbose(3, '(it %i) Habituation conditional sampling (condition %d) ...' %(self.iteration, j))
                    #self.habitCondSampler(j, rb, varHRF)
                #pyhrf.verbose(4, '(it %i) Habituation conditional sampling done ...' %self.iteration)
            
        #self.updateXh(h)
        #self.updateYtilde()
            
            #pyhrf.verbose(4, 'NRLS current Value: ')
        #pyhrf.verbose.printNdarray(4, self.currentValue)
            
            #self.saveSamples()
            ## for the hrf
        #self.computeAA(self.currentValue, self.aa)        
        #self.iteration += 1    
    
    def sampleNextAlt(self, variables): ## if we don't want to sample Nrls
        
        rb = variables[self.samplerEngine.I_NOISE_VAR].currentValue
        varHRF = variables[self.samplerEngine.I_HRF].currentValue
            
        pyhrf.verbose(2, 'iteration %i' %self.iteration)

        self.updateXh(varHRF)
        self.updateYtilde()
            
    
        #sampling labels
        if self.sampleLabelsFlag:
            self.computeVarXhtQ(self.dataInput.delta)
            self.labelsSamples = random.rand(self.nbConditions, self.nbVox)
            varXhtQXh = zeros((self.nbVox, self.nbConditions), dtype=float)
            for i in xrange(self.nbVox):
                varXhtQXh[i,:] = diag(dot(self.varXhtQ[i,:,:], self.varXh[i,:,:]))
        
            for j in random.permutation(self.nbConditions):
                self.computeComponentsApost(variables, j, varXhtQXh)        
                pyhrf.verbose(3,'(It %i) Sampling labels - cond %d ...'%(self.iteration,j))
                self.sampleLabels(j, variables)
                pyhrf.verbose(3,'(It %i) Sampling labels done!' %self.iteration)
                self.countLabels(self.labels, self.voxIdx, self.cardClass)        
    
    
        # sampling habits     
        if self.sampleHabitFlag:
            pyhrf.verbose(3, '(it %i) Habituation conditional sampling ...' %self.iteration)
            for j in random.permutation(self.nbConditions):
                self.habitCondSampler(j, rb,varHRF)
            pyhrf.verbose(3, '(it %i) update done ...' %self.iteration)    

        self.saveSamples()
        self.computeAA(self.currentValue, self.aa)
        self.iteration += 1    


    def finalizeSampling(self):
    
        self.finalLabels = self.getFinalLabels()
        
        # on n'afiche que les habituations du label final
        for c in xrange(self.nbClasses):
            putmask(self.meanHabits, self.finalLabels==c, self.meanHabitsCond[c,:,:])

    
        
        # affichage de timeNRLs -> 10 voxels tires au hasard (car on ne peux tout sauvegarder -> trop grand (nc * nVox * lastonset)
        self.outputTimeNRLs = zeros((self.nbConditions, 10, self.lastonset), dtype = float)
        for nc in xrange(self.nbConditions):
            # tirage des voxels observes
            a=[int(self.nbVox*random.rand()) for i in range(10)]
            # si on veut choisir precisement les voxels, il faut passer par ipython (comme plus loin) avec getIndex([axial Coronal Sagittal])
            # il faut prendre exactement 10 voxels ou alors changer la def de self.outputTimeNRLs et le axesDomain de getOutput
            # attention, on observe les memes voxels pour toutes les regions, il faut donc il ne faut pas depasser le nbVox de la plus petite region
            # par ex: 
            # a = [ min( 23, self.nbVox), min(36, self.nbVox), etc...]
            
            
            # important: print des voxels observes
            #print 'voxels observes', a
            for nt in xrange(self.nbTrials[nc]):
                # on sauvegarde les timeNRLs qui nous interessent
                self.outputTimeNRLs[nc,:,int(self.onsets[nc][nt])] = self.timeNrls[nc][a,nt]
        self.voxelobs = a
        # sauvegarde dans result des voxels observes
        # pour trouver les coordonnes correspondant, il faut sous ipython:
        # charger les donnees: import cPickle
        #                      s = cPickle.load(open('result.pyd'))
        # pour voir les coor.: s[numeroROI]['sampler'].dataInput.voxelMapping.getCoord(voxelobs)
        #     
        
        smplHRF = self.samplerEngine.getVariable('hrf')
    
            
        
        # Correct hrf*nrl scale ambiguity :
        scaleF = smplHRF.getScaleFactor()
        
        # Use HRF amplitude :
        pyhrf.verbose(5,'scaleF=%1.2g' %scaleF)
        pyhrf.verbose(5,
                      'self.finalValue : %1.2g - %1.2g' \
                      %(self.finalValue.min(), self.finalValue.max()))
        self.finalValueScaleCorr = self.finalValue*scaleF

        if self.computeContrastsFlag:
            self.computeContrasts()




    def getOutputs(self):

        outputs = NRLSampler.getOutputs(self)
        
        axes_names = ['condition', 'voxel']
        axes_domains = {'condition' : self.dataInput.cNames}
        outputs['pm_Habits'] = xndarray(self.meanHabits,
                                      axes_names=axes_names,
                                      axes_domains=axes_domains,
                                      value_label="Habituation")
                       
        axes_names = ['condition', 'numeroVox', 'time']
        axes_domains = {'condition' : self.dataInput.cNames, 
                       'numeroVox' : [0,1,2,3,4,5,6,7,8,9], 
                       'time' : range(self.lastonset)}
        outputs['pm_timeNrls'] = xndarray(self.outputTimeNRLs,
                                        axes_names=axes_names,
                                        axes_domains=axes_domains,
                                        value_label="TimeNRLs")
        
        dt = self.samplerEngine.getVariable('hrf').dt
        rpar = self.dataInput.paradigm.get_joined_and_rastered(dt)
        parmask = [where(rpar[c]==1) for c in self.dataInput.cNames]
        tnrl = zeros((self.nbVox, self.nbConditions, len(rpar[rpar.keys()[0]])), 
                     dtype=float)
        for i in xrange(self.nbVox):
            for j in xrange(self.nbConditions):
                tnrl[i,j,parmask[j]] = self.timeNrls[j][i,:]
                
        #TODO: dirac outputs to use less space
        axes_domains = {'condition' : self.dataInput.cNames,
                       'time' : arange(0,self.dataInput.paradigm.get_t_max(), dt)}
        outputs['pm_tNrls'] = xndarray(tnrl,
                                     axes_names=['voxel','condition','time'],
                                     axes_domains=axes_domains,
                                     value_label="nrl")
        if self.keepSamples:
            axes_names = ['iteration', 'condition', 'voxel']
            axes_domains = {'classe' : ['inactif','actif'] , 
                           'condition' : self.dataInput.cNames, 
                           'iteration' : self.sampleHistoryIterations}
            outputs['pm_Habits_hist'] = xndarray(self.habitsHistory,
                                               axes_names=axes_names,
                                               axes_domains=axes_domains,
                                               value_label="Habituation")
                           
            if self.outputRatio:
                axes_names = ['iteration', 'condition', 'voxel', 'ratio']
                axes_domains = {'condition' : self.dataInput.cNames, 
                               'iteration' : self.sampleHistoryIterations, 
                               'ratio' : [1,2]}
                outputs['pm_ratio_hist'] = xndarray(self.ratioHistory,
                                                  axes_names=axes_names,
                                                  axes_domains=axes_domains,
                                                  value_label="Ratio")
                axes_names = ['iteration', 'condition', 'voxel', 'courbe', 'ratio']
                axes_domains = {'condition' : self.dataInput.cNames,
                               'iteration' : self.sampleHistoryIterations,
                               'ratio' : [1,2,3,4,5], 
                               'courbe' : arange(0.,1.,0.01)
                               }
                outputs['pm_ratiocourbe_hist'] = xndarray(self.ratiocourbeHistory,
                                                        axes_names=axes_names,
                                                        axes_domains=axes_domains,
                                                        value_label="Courbe Ratio")
                                    
                axes_names = ['condition', 'voxel']
                axes_domains = {'condition' : self.dataInput.cNames}
                outputs['pm_compteur'] = xndarray(self.compteur,
                                                axes_names=axes_names,
                                                axes_domains=axes_domains,
                                                value_label="compteur")
    
        
        return outputs
    
    
    
    

    def cleanMemory(self):
        
        #NRLSampler.cleanMemory(self)
        
        del self.varClassApost
        del self.sigClassApost
        del self.sigApost
        del self.meanApost
        del self.aa
        del self.aXh
        del self.varYtilde
        del self.varXhtQ
        del self.sumaXh
        if hasattr(self,'labelsSamples'):
            del self.labelsSamples
        del self.corrEnergies
        del self.labels
        del self.voxIdx
        
        del self.sumaX
        #del self.sumaXQ  
        del self.Xmask
        del self.nnulls
        del self.varSingleCondXtrials
        del self.voxelActivity
  
        del self.sumaXhtQaXh     
    
    
    
        # clean memory of temporary variables :
        #del self.timeNrls
        del self.Gamma
        del self.habits
    
        del self.onsets
        del self.deltaOns
        del self.nbTrials

        #del self.outputRatio
        #del self.trueHabits
        del self.nh
        #del self.sumaXtQ
        #del self.sumaXtQaX

        del self.varXh
        #del self.varXhtQXh
        #del self.varsumXh
    
    def spExtract(self, spInd, mtrx, cond):
        X = mtrx + 0.
        xr = X.ravel()
        xr[argsort(xr)] = concatenate((self.nnulls[cond],spInd.repeat(self.nbGammas[cond])), axis = 0)
        return X       
    
        #def bilinearsparsedot(self, gamma, Q, taille, cond):
        #Y = zeros((taille, taille), dtype= float)
            
        #for t1 in xrange(self.nbTrials[cond]):
                #for (k,i) in self.Xmask[cond][t1]:
                    #for t2 in xrange(self.nbTrials[cond]):
                        #for (l,j) in  self.Xmask[cond][t2]:
                            #Y[i,j] += gamma[t1] * gamma[t2] * Q[k,l]
        #return Y 
    
# il y a un probleme sur cette fonction    
def sparsedot(X,A, mask, taille):
    Y = zeros(taille, dtype=float)
    #print shape(Y)
    #print shape(A)
    #raw_input()
    for (i,k) in mask:
    #print i,k
        Y[:,k] += X * A[:,i]
    return Y                            
                
# il y a un probleme aussi sur cette fonction
def sparsedotdimun(X,A, mask, lenght):
    Y = zeros(lenght, dtype=float)
    for (i,k) in mask:
        Y[i] += X * A[k]
    return Y


##----- functions for habitSamper() ------##

def LaplacianPdf(beta,r0Hab,a,b,N=1):## instrumental law for MH algo
    if size(r0Hab)>1:
        N=size(r0Hab)
        u=numpy.random.rand(1,N)
    elif size(r0Hab)==1:
        u = numpy.random.rand()
    la=beta*a
    lb=beta*b
    lr0Hab=beta*r0Hab

    if all(r0Hab<a): 
#        print "r0Hab<a"
        abConst=math.exp(-la)-math.exp(-lb)
        x=-log(math.exp(-la)-u*abConst)/beta
        r0HabConst=1.
    elif all(r0Hab > b):
#        print "r0Hab>b"
        abConst = math.exp(lb) -math.exp(la)
        x = log( math.exp(la) + u * abConst ) /beta
        r0HabConst=1.        
    elif size(r0Hab)==1:  ##  r0Hab value in [a,b] interval
#        print "size(r0Hab)==1"
        e_lamr0Hab = math.exp(beta*(a-r0Hab))
        e_lr0Habmb = math.exp(beta*(r0Hab-b))
        r0HabConst =( 2. - e_lamr0Hab - e_lr0Habmb)/beta
        F_r0Hab = ( 1. - e_lamr0Hab )/(2. - e_lamr0Hab - e_lr0Habmb)
        if (u<F_r0Hab):
            abConst = r0HabConst * beta * math.exp(lr0Hab)
            x = log( math.exp(la) + u * abConst )/beta
        elif (u >= F_r0Hab):
            abConst = r0HabConst * beta
            x = r0Hab - log( 2. - e_lamr0Hab - u * abConst)/beta
        else:
            abConst0 = r0HabConst * beta * math.exp(lr0Hab)
            abConst1 = r0HabConst * beta
#            for k in xrange(0,N):
            if (u < F_r0Hab ):
                x = log(math.exp(la) + u*abConst0)/beta
            else:
                x = r0Hab - log(2. - e_lamr0Hab - u*abConst1)/beta
    else:
        x=zeros((N), dtype=float)
        for i in xrange(N):
            x[i] = LaplacianPdf(beta,r0Hab[i],a,b,1.0)
            r0HabConst.append(r0HabConst)
    return x, r0HabConst
        
#def spExtract(spInd, mtrx):
    #X = mtrx + 0.
    #for j in xrange(spInd.size):
        #X[mtrx == (j+1)] = spInd[j]
    #return X       
        
#def spExtract(spInd, mtrx, cond):
    #X = mtrx + 0.
    #xr = X.ravel()
    #xr[argsort(xr)] = concatenate(self.nnulls,spInd.repeat(self.nbGammas[cond]))
    #return X       
    
# spExtract a bouge dans la class Habit


def subcptGamma(nrl,habit,nbTrials,deltaOns):
    ##print 'function subcptGamma: start...',
    gamma=zeros(nbTrials,dtype=float)
    timeNrls=zeros(nbTrials,dtype=float)
    gamma[0]=1.
    timeNrls[0]=nrl
    skk=0.
    for k in xrange(1,nbTrials):
        skk=(habit**deltaOns[k-1])*(skk+timeNrls[k-1])
        gamma[k]=1./(1.+skk)
        timeNrls[k]=gamma[k]*nrl      
    ##print 'OK'
    ##print 'skk',skk, 'gamma',gamma, 'timeNrls',timeNrls     
#    return gamma
    return gamma, timeNrls

#def spExtract(spInd, mtrx):
    
    #X = array(mtrx, dtype=float) +0.
    
    #for j in xrange(spInd.size):
        #X[mtrx == (j+1)] = spInd[j]
    #return X       
    
    
    
##def spExtract(vctr, mtrxX):
    ##x = mtrxX
    ##v=(numpy.where(mtrxX!=0))
    ##a=x[v]
    ##kk=zeros(len(a))
    ##for i in xrange(len(a)):
        ##kk[i]=vctr[(a[i]-1)]
    ##x[v]=kk
    ##return x
