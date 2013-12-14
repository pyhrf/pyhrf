# -*- coding: utf-8 -*-
from scipy.linalg import toeplitz
from pyhrf import xmlio
from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler

from pyhrf.graph import graph_nb_cliques

from samplerbase import *
from hrf import *
from nrl import *
from noise import *
from drift import *
from beta import *
from wsampler import *
#from likelihood import LikelihoodSampler

from nrl.bigaussian import *
from nrl.bigaussian_drift import *

from pyhrf.tools.misc import Pipeline, diagBlock

import numpy as np

from collections import defaultdict

def computehXQXh(hrf, matXQX, dest=None):

    h = hrf.currentValue
    res = np.dot(np.dot(h,matXQX),h)
    if dest == None:
        return res
    else:
        dest[:] = res[:]

def computeXh(hrf, varX, dest=None):

    h = hrf.currentValue
    res = np.dot(np.dot(h,varX),h)
    if dest == None:
        return res
    else:
        dest[:] = res[:]

def computePl(drift, varP, dest=None):

    l = drift.currentValue
    res = np.dot(varP, l)
    if dest == None:
        return res
    else:
        dest[:] = res[:]

def computeSumjaXh(nrl, matXh, dest=None):

    aXh = repmat(varXh,nrl.nbVox,1).reshape(nrl.nbVox, nrl.ny, nrl.nbConditions)
    aXh = aXh.swapaxes(0,1).swapaxes(1,2)
    aXh *= nrl.currentValue
    if dest == None:
        return aXh.sum(1)
    else:
        dest[:] = aXh.sum(1)


def computeYTilde(sumj_aXh, varMBY, dest=None):
    if dest == None:
        return varMBY - sumj_aXh
    else:
        np.subtract(varMBY, sumj_aXh, dest)

def computeYBar(varMBY, varPl, dest=None):
    if dest == None:
        return varMBY - varPl
    else:
        np.subtract(varMBY, varPl, dest)

def computeYTilde_Pl(sumj_aXh, yBar, dest=None):
    if dest == None:
        return yBar - sumj_aXh
    else:
        np.subtract(yBar, sumj_aXh, dest)


class BOLDSamplerInput :

    """
    Class holding data needed by the sampler : BOLD time courses for each voxel,
    onsets and voxel topology.
    It also perform some precalculation such as the convolution matrix based on
    the onsests (L{stackX})
    """
    #TODO : comment attributes

    def __init__(self, data, dt, typeLFD, paramLFD, hrfZc, hrfDuration) :
        """
        Initialize a BOLDSamplerInput object. Mainly extract data from boldData.
        """
        pyhrf.verbose(3, 'BOLDSamplerInput init ...')
        pyhrf.verbose(3, 'Recieved data:')
        pyhrf.verbose(3, data.getSummary())

        self.roiId = data.get_roi_id()
        self.nbSessions = len(data.sessionsScans)
        self.varMBY = data.bold
        #raise Exception()
        self.nys = [len(ss) for ss in data.sessionsScans]
        self.ny = shape(self.varMBY)[0]
        self.varData = self.varMBY.var(0)
        self.tr = data.tr


        # if any ... would be used to compare results:
        self.simulData = data.simulation

        graph = data.get_graph()
        #self.nbVoxels = len(graph)
        self.nbVoxels = shape(self.varMBY)[1]

#         self.roiMapper = xndarrayMapper1D(spatialConfig.getMapping(),
#                                         self.finalShape,
#                                         spatialConfig.getTargetAxesNames(),
#                                         'voxel')

        # put neighbour lists into a 2D numpy array so that it will be easily
        # passed to C-code
        #print 'nl in graph:', graph
        #for nl in graph:
            #print 'nl:', nl
        #print 'graph:', graph[0]
        #print 'graph:'
        #print graph

        nmax = max([len(nl) for nl in graph])
        #print 'nmax:', nmax

        #hack
        #nmax=4
        #for nb in graph:
            #print 'nb:', nb
            #nl = graph[nb]
            #print 'nl:', nl[0]
            #nmax = max(len(nl))

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
        onsets = data.paradigm.get_joined_onsets()
        self.onsets = [onsets[cn] for cn in self.cNames]
        #print 'onsets are :', self.onsets
        durations = data.paradigm.get_joined_durations()
        self.durations = [durations[cn] for cn in self.cNames]
        #print 'durations are :', self.durations
        self.chewUpOnsets(dt, hrfZc, hrfDuration)

        # Build matrices related to low frequency drift
        pyhrf.verbose(3, 'Building LFD mats ...')
        self.setLFDMat(paramLFD,typeLFD)

        pyhrf.verbose(3, 'Making precalculcations ...')
        self.makePrecalculations()

    def makePrecalculations(self):
        pass

    def cleanPrecalculations(self):
        pass

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
        #print 'binOnsets:'
        #print self.paradigmData

        pyhrf.verbose(5,'paradigm :')
        for iSess in xrange(self.nbSessions):
            pyhrf.verbose(5,' session: %d'%iSess)
            for iCond in xrange(self.nbConditions):
                m = 'cond:%d -> l=%d'  %(iCond,len(self.paradigmData[iCond][iSess]))
                pyhrf.verbose.printNdarray(5 ,m)
                pyhrf.verbose.printNdarray(6, self.paradigmData[iCond][iSess])
        # paradigm should be zero-expanded to match length of Bold data

        pyhrf.verbose(3, 'building paradigm convol matrix ...')
        availIdx = [arange(0,n, dtype=int) for n in self.nys]
        self.buildParadigmConvolMatrix(hrfZc, hrfDuration, availIdx,
                                       self.paradigmData)
        pyhrf.verbose(5, 'matrix X : %s' %str(self.varX.shape))
        #pyhrf.verbose.printNdarray(5, self.varX)
        if pyhrf.verbose.verbosity >= 5:
            for i in xrange(self.varX.shape[0]):
                print ''
                for j in xrange(self.varX.shape[1]):
                    print '',
                    for k in xrange(self.varX.shape[2]):
                        print self.varX[i,j,k],
                    print ''
                print ''

        ###############################################
        #TODO : adapt to make it multi sessions ...     #
        ###############################################
        pyhrf.verbose(3,
                      'building Single Cond paradigm convol matrix for'+  \
                      'Habituation model...')
        for iSess in xrange(self.nbSessions):
            pyhrf.verbose(5,' session: %d'%iSess)
            #self.buildParadigmSingleCondMatrix(hrfZc, hrfDuration,
                                               #arange(0,self.ny, dtype=int),
                                               #self.paradigmData[iSess][:])
            #self.buildParadigmSingleCondMatrix(self.onsets[iSess], self.stimDurations[iSess])
            self.buildParadigmSingleCondMatrix(hrfZc, hrfDuration, availIdx, self.paradigmData)

            pyhrf.verbose(5,
                        'single cond X matrix with incremental trials:')
            pyhrf.verbose.printNdarray(5,
                                    self.varSingleCondXtrials)



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
        pyhrf.verbose(3, 'LFD type:' + typeLFD)
        for iSess in xrange(self.nbSessions):
            if typeLFD == 'polynomial':
                lfdMat = self.buildPolyMat( paramLFD , self.nys[iSess])
            elif typeLFD == 'cosine':
                lfdMat = self.buildCosMat( paramLFD , self.nys[iSess])
            elif typeLFD == 'None':
                lfdMat = np.zeros((self.nys[iSess],2))
                #        print 'shape LFDMat : ', self.lfdMat.shape
            #(Drift, varLFDcoeff, varLFDMat) = genDrift( .1, typeLFD, paramLFD,
            #                                            self.tr, self.nys[iSess],
            #                                            self.nbVoxels )


            self.lfdMat.append(lfdMat)
            varPPt = np.dot(lfdMat, lfdMat.transpose())
            if typeLFD is not 'None':
                self.colP = shape(lfdMat)[1]
            else:
                self.colP = 0

            #print 'varPPt :', varPPt.shape
            #print 'np.eye(self.nys[iSess], dtype=float):', np.eye(self.nys[iSess], dtype=float).shape
            self.delta.append(np.eye(self.nys[iSess], dtype=float) - varPPt)
            self.varPtP.append(np.dot(lfdMat.transpose(), lfdMat))

            pyhrf.verbose(6, 'varPtP :')
            pyhrf.verbose.printNdarray(6, self.varPtP[-1])
            if typeLFD != 'None':
                assert np.allclose(self.varPtP[-1],
                                      np.eye(self.colP, dtype=float),
                                      rtol=1e-5 )

        self.delta = diagBlock(self.delta)
        pyhrf.verbose(6, 'delta %s :' %str(self.delta.shape))
        pyhrf.verbose.printNdarray(6, self.delta)


    def buildPolyMat( self, paramLFD , n ):

        regressors = self.tr*arange(0, n)
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
        lfdMat[:,0] = np.ones( ny, dtype= float)/sqrt(ny)
        samples = 1. + np.arange(fctNb-2)
        for k in samples:
          lfdMat[:,k] = np.sqrt(2/ny) \
                        * np.cos( np.pi*(2.*n+1.)*k / (2*ny) )
        return lfdMat

    def buildParadigmConvolMatrix(self, zc, estimDuration, availableDataIndex,
                                  parData) :
#        print 'buildParadigmConvolMatrix ...'
        osf = self.tr/self.dt

        pyhrf.verbose(2, 'osf = %1.2f' %osf)
        pyhrf.verbose(6, 'availableDataIndex :')
        pyhrf.verbose.printNdarray(6, availableDataIndex)

        lgt = (self.ny+2)*osf
        allMatH = []
        for iSess in xrange(self.nbSessions):
            matH = np.zeros((lgt, self.nbConditions), dtype=int)
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

        self.hrfLength = len(np.arange(0,estimDuration+self.dt, self.dt))
        pyhrf.verbose(5, 'hrfLength = range(0,%f+%f,%f)=%d' \
                          % (estimDuration,self.dt,self.dt,self.hrfLength))
        if zc :
            self.hrfColIndex = arange(1, self.hrfLength-1, dtype=int)
            self.colIndex = arange(0, self.hrfLength-2, dtype=int)
        else :
            self.hrfColIndex = arange(0, self.hrfLength, dtype=int)
            self.colIndex = arange(0, self.hrfLength, dtype=int)
        self.lgCI = len(self.colIndex)
        pyhrf.verbose(5, 'lgCI = %d'% self.lgCI)

        #vDiv = arange(0,self.tr, self.dt, dtype=float)

        self.varOSAvailDataIdx = [array(ai*osf, dtype=int)
                                  for ai in availableDataIndex]
        #+ len(vDiv)
        vX = []
        pyhrf.verbose(2, 'Build pseudo teoplitz matrices')
        for iSess in xrange(self.nbSessions):
            self.lenData = len(self.varOSAvailDataIdx[iSess])
            varX = np.zeros( (self.nbConditions, self.lenData, self.lgCI),
                          dtype=int )
            pyhrf.verbose(6, 'iSess : %d' %iSess)
            for j in xrange(self.nbConditions):
                pyhrf.verbose(6, ' cond : %d' %j)
                col = concatenate(([allMatH[iSess][0,j]],
                                   np.zeros(self.hrfLength-1, dtype=int)))
                pyhrf.verbose(6, ' col :')
                if pyhrf.verbose.verbosity >= 6:
                    print ' [',
                    for b in xrange(col.shape[0]):
                        print col[b],
                    print ']'


                matTmp = array(toeplitz( allMatH[iSess][:,j], col), dtype=int)
                pyhrf.verbose(6, ' matTmp :')
                if pyhrf.verbose.verbosity >= 6:
                    for b in xrange(matTmp.shape[0]):
                        print ' [',
                        for a in xrange(matTmp.shape[1]):
                            print matTmp[b,a],
                        print ']'
##                 print 'self.varOSAvailDataIdx :'
##                 print self.varOSAvailDataIdx[iSess]
                d0 = matTmp[self.varOSAvailDataIdx[iSess],:]
##                 print 'd0 :', d0.shape
##                 for b in xrange(d0.shape[0]):
##                     print ' [',
##                     for a in xrange(d0.shape[1]):
##                         print d0[b,a],
##                     print ']'

                d1 = d0[:,self.hrfColIndex]
##                 print 'd1 :', d1.shape
##                 for b in xrange(d1.shape[0]):
##                     print ' [',
##                     for a in xrange(d1.shape[1]):
##                         print d1[b,a],
##                     print ']'

                varX[j,:,:] = d1

            vX.append(varX)
        self.varX = hstack(vX)
        pyhrf.verbose(4, 'varX : ' + str(self.varX.shape))
        self.buildOtherMatX()

        self.nbColX = shape(self.varX)[2]

#        self.notNullIdxStack = dstack(where(self.stackX != 0)).ravel()
##        print 'self.notNullIdxStack :', self.notNullIdxStack.shape , ' - ', self.notNullIdxStack.dtype
##        print array2string(self.notNullIdxStack, 80, 0)


    def buildOtherMatX(self):
        self.varMBX = np.zeros( (self.ny, self.nbConditions*self.lgCI),
                             dtype=int)
        self.stackX = np.zeros( (self.ny*self.nbConditions, self.lgCI),
                             dtype=int)

        for j in xrange(self.nbConditions):
            self.varMBX[:, j*self.lgCI+self.colIndex] = self.varX[j,:,:]

            self.stackX[self.ny*j:self.ny*(j+1), :] = self.varX[j,:,:]

        self.notNullIdxStack = dstack(where(self.stackX != 0)).ravel()

    #def buildParadigmSingleCondMatrix(self, zc, estimDuration,
                                      #availableDataIndex, parData) :
        #osf = self.tr/self.dt
        #lgt = (self.ny+2)*osf
        #matH = zeros( (lgt, self.nbConditions), dtype=float)


        ##for isess in xrange(self.nbSessions):

        #for j in xrange(self.nbConditions) :
            #trials = int32(parData[j][:] >0.)
            #idx_trials = where(trials)
            #nbtrials = trials.sum()
            #matH[idx_trials, j] = arange(1,nbtrials+1, dtype=int)

            ##self.hrfLength = int(round(estimDuration/self.dt)) +1
            ##already computed in the previous function
                ##if zc :
            ##hrfColIndex = arange(1, self.hrfLength-1, dtype=int)
            ##self.colIndex = arange(0,self.hrfLength-2, dtype=int)
            ##else :
            ##hrfColIndex = arange(0, self.hrfLength, dtype=int)
            ##self.colIndex = arange(0,self.hrfLength, dtype=int)
            ##self.lgCI = len(self.colIndex)
            ##varOSAvailDataIdx = array(availableDataIndex*osf, dtype=int) #+ len(vDiv)
            ##self.lenData = len(varOSAvailDataIdx)

            ##self.varSingleCondXtrials = zeros( (self.nbConditions, self.lenData,
                            ##self.lgCI), dtype=int)
            #self.varSingleCondXtrials = zeros( (self.nbConditions, self.ny,
                            #self.lgCI), dtype=int)
            #self.varSingleCondXtrials = parData

            ###for j in xrange(self.nbConditions):
            ###    col = concatenate(([matH[0,j]], zeros(self.hrfLength-1, dtype=int)))
            ###    matTmp = array(toeplitz( matH[:,j], col), dtype=int)
            ###    d = matTmp[self.varOSAvailDataIdx,:][:,self.hrfColIndex]
            ###    self.varSingleCondXtrials[j,:,:] = d[0][0]


    #def buildParadigmSingleCondMatrix(self, onsets, stiml) :


        ##for isess in xrange(self.nbSessions):

        #self.varSingleCondXtrials = zeros( (self.nbConditions, self.ny, self.hrfLength), dtype=int)


        #for nc in xrange(self.nbConditions) :
            #for h in range(self.hrfLength) :
                #for ons in range(len(onsets[nc])) :
                    #for l in range(int(round(stiml[nc][ons])+1)) :
                        #if int(round(onsets[nc][ons])) + l + h < self.ny :
                        #self.varSingleCondXtrials[nc][int(round(onsets[nc][ons])) + l + h][h] = ons
        #print shape(self.varSingleCondXtrials)
        #raw_input()

    def buildParadigmSingleCondMatrix(self, zc, estimDuration, availableDataIndex, parData) :

        pyhrf.verbose(2, 'Build Paradigm Matrix (Single, Conditional)')
        osf = self.tr/self.dt

        parData_trials = copyModule.deepcopy(parData)
        #parData[nonzero(parData)] = arange(size(nonzero(parData)))+ones(size(nonzero(parData)))


        lgt = (self.ny+2)*osf
        allMatH = []
        for iSess in xrange(self.nbSessions):
            nbTrials = np.zeros(self.nbConditions)

            matH = np.zeros( (lgt, self.nbConditions), dtype=int)
            for j in xrange(self.nbConditions) :
                nbTrials[j] = size(nonzero(parData[j][iSess]))
                parData_trials[j][iSess][nonzero(parData[j][iSess])] = arange(1,size(nonzero(parData[j][iSess]))+1)
                #parData_trials[iSess][j][nonzero(parData[iSess][j])] = arange(1,nbTrials[j]+1)
                #matH[:len(parData[iSess][j]), j] = range(len(parData[iSess][j])+1)[1:]
                #matH[:len(parData[iSess][j]), j] = parData[iSess][j]
                matH[:len(parData_trials[j][iSess]), j] = parData_trials[j][iSess][:]

                allMatH.append(matH)

        if zc :
            self.hrfColIndex = arange(1, self.hrfLength-1, dtype=int)
            self.colIndex = arange(0,self.hrfLength-2, dtype=int)
        else :
            self.hrfColIndex = arange(0, self.hrfLength, dtype=int)
            self.colIndex = arange(0,self.hrfLength, dtype=int)
        self.lgCI = len(self.colIndex)
        pyhrf.verbose(6, 'lgCI = %d'% self.lgCI)

        #vDiv = arange(0, self.tr, self.dt, dtype=float)
        #self.varOSAvailDataIdx = [array(ai*osf + len(vDiv), dtype=int)
                                  #for ai in availableDataIndex]
        self.varOSAvailDataIdx = [array(ai*osf, dtype=int)
                                  for ai in availableDataIndex]



        vSCXt = []
        pyhrf.verbose(2, 'Build pseudo teoplitz matrices')
        for iSess in xrange(self.nbSessions):
            self.lenData = len(self.varOSAvailDataIdx[iSess])
            varSCXt = np.zeros( (self.nbConditions, self.lenData, self.lgCI),
                          dtype=int)
            for j in xrange(self.nbConditions):
                col = concatenate(([allMatH[iSess][0,j]],
                                   np.zeros(self.hrfLength-1, dtype=int)))

                matTmp = array(toeplitz( allMatH[iSess][:,j], col), dtype=int)
                pyhrf.verbose(6, ' matTmp :')

                d0 = matTmp[self.varOSAvailDataIdx[iSess],:]

                d1 = d0[:,self.hrfColIndex]

                varSCXt[j,:,:] = d1
            vSCXt.append(varSCXt)
        self.varSingleCondXtrials = hstack(vSCXt)

    def calcDt(self, dtMin) :

        pyhrf.verbose(2, 'dtMin = %1.3f' %dtMin)
        pyhrf.verbose(2, 'Trying to set dt automatically from data')

        tr = self.tr
        vectSOA = array([], dtype=float)
        for ons in self.onsets:
            vectSOA = concatenate((vectSOA, ons))

        vectSOA.sort()
        pyhrf.verbose(5, 'vectSOA %s:' %str(vectSOA.shape))
        pyhrf.verbose.printNdarray(5,vectSOA)

        momRT = arange(0, vectSOA[-1]+tr, tr)
        pyhrf.verbose(5,'momRT %s:' %str(momRT.shape))
        pyhrf.verbose.printNdarray(5, momRT)

        momVect = concatenate((vectSOA, momRT))
        momVect.sort()

        varSOA = diff(momVect)
        pyhrf.verbose(5, 'vectSOA diff:')
        pyhrf.verbose.printNdarray(5, vectSOA)

        nonZeroSOA = varSOA[where(varSOA > 0.0)]
        pyhrf.verbose(5, 'nonZeroSOA :')
        pyhrf.verbose.printNdarray(5, nonZeroSOA)

        delta = nonZeroSOA.min()
        pyhrf.verbose(5, 'delta : %1.3f' %delta)
        pyhrf.verbose(5, 'dtMin : %1.3f' %dtMin)
        #osfi = osf = round_(tr/delta)
        #osf = min(osf, osfMax)
        dt = max(dtMin, delta)
        pyhrf.verbose(2, 'dt from onsets: %1.2f, dtMin=%1.3f => '\
                          'dt=%1.2f (osf=%1.3f)'
                      %(delta, dtMin, dt, (tr+0.)/dt))
        return dt



    def cleanMem(self):
 #       del self.varX

        pyhrf.verbose(3, 'cleaning Memory BOLD Sampler Input')

        del self.varMBX

        del self.varOSAvailDataIdx
        #del self.voxelMapping.neighboursIndexLists
        #if hasattr(self.voxelMapping, 'neighboursCoordLists'):
        #    del self.voxelMapping.neighboursCoordLists
        #del self.stackX
        del self.notNullIdxStack
        #del self.onsets
        del self.colP
        del self.delta
        #del self.lfdMat
        del self.varPtP
        #del self.varMBY
        del self.paradigmData
        del self.neighboursIndexes
        del self.varSingleCondXtrials
        del self.stimRepetitions
        del self.hrfColIndex
        del self.colIndex
        self.cleanPrecalculations()


class WN_BiG_BOLDSamplerInput(BOLDSamplerInput):

    def makePrecalculations(self):
        # XQX & XQ:
        self.matXQX = np.zeros( (self.nbConditions, self.nbConditions,
                                    self.nbColX, self.nbColX),dtype=float)
        self.matXQ = np.zeros((self.nbConditions, self.nbColX, self.ny),
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
        self.matQy = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.yTQ = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.yTQy = np.zeros(self.nbVoxels, dtype=float)

        for i in xrange(self.nbVoxels):
            self.matQy[:,i] = dot(self.delta,self.varMBY[:,i])
            self.yTQ[:,i] = dot(self.varMBY[:,i],self.delta)
            self.yTQy[i] = dot(self.varMBY[:,i],self.matQy[:,i])


    def cleanPrecalculations(self):

        del self.yTQ
        del self.yTQy
        del self.matQy
        del self.matXQ
        del self.matXQX


class WN_BiG_Drift_BOLDSamplerInput(BOLDSamplerInput):

    def makePrecalculations(self):
        # XQX & XQ:
        self.matXtX = np.zeros( (self.nbConditions, self.nbConditions,
                                    self.nbColX, self.nbColX),dtype=float)

        for j in xrange(self.nbConditions):
            #print 'self.varX :', self.varX[j,:,:].transpose().shape
            #print 'self.delta :', self.delta.shape
            for k in xrange(self.nbConditions):
                self.matXtX[j,k,:,:] = np.dot( self.varX[j,:,:].transpose(),
                                                  self.varX[k,:,:] )
        # Qy, yTQ & yTQy  :
        # self.matQy = np.zeros((self.ny,self.nbVoxels), dtype=float)
        # self.yTQ = np.zeros((self.ny,self.nbVoxels), dtype=float)
        # self.yTQy = np.zeros(self.nbVoxels, dtype=float)

        # for i in xrange(self.nbVoxels):
        #     self.matQy[:,i] = dot(self.delta,self.varMBY[:,i])
        #     self.yTQ[:,i] = dot(self.varMBY[:,i],self.delta)
        #     self.yTQy[i] = dot(self.varMBY[:,i],self.matQy[:,i])


    def cleanPrecalculations(self):

        #del self.yTQ
        #del self.yTQy
        #del self.matQy
        del self.matXtX
        #del self.matXQX



class ARN_BiG_BOLDSamplerInput(BOLDSamplerInput):

    def makePrecalculations(self):
        pass

    def cleanPrecalculations(self):
        pass

class Hab_WN_BiG_BOLDSamplerInput(WN_BiG_BOLDSamplerInput):

    def makePrecalculations(self):
        self.matXQX = np.zeros( (self.nbConditions, self.nbConditions,
                                    self.nbColX, self.nbColX),dtype=float)
        self.matXQ = np.zeros((self.nbConditions, self.nbColX, self.ny),
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
        self.matQy = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.yTQ = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.yTQy = np.zeros(self.nbVoxels, dtype=float)

        for i in xrange(self.nbVoxels):
            self.matQy[:,i] = dot(self.delta,self.varMBY[:,i])
            self.yTQ[:,i] = dot(self.varMBY[:,i],self.delta)
            self.yTQy[i] = dot(self.varMBY[:,i],self.matQy[:,i])


    def cleanPrecalculations(self):
        #pass

        pyhrf.verbose(3, 'cleaning Precalculations')

        del self.yTQ
        del self.yTQy
        del self.matQy
        del self.matXQ
        del self.matXQX



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

        self.varMBY = [data.bold[ss,:].astype(np.float64) \
                       for ss in data.sessionsScans]

        self.ny = self.nys[0]
        self.varData = np.array([self.varMBY[ss].var(0) \
                                 for ss in xrange(self.nbSessions)])
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
                self.matQy[:,i] = dot(self.delta[iSess],self.varMBY[iSess][:,i])
                self.yTQ[:,i] = dot(self.varMBY[iSess][:,i],self.delta[iSess])
                self.yTQy[i] = dot(self.varMBY[iSess][:,i],self.matQy[:,i])


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

        self.matXQ = np.array(XQX)
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

        self.stimRepetitions = [[len(self.onsets[ind][s]) for ind in xrange(nbc)] \
                                for s in xrange(self.nbSessions)]

        pyhrf.verbose(5, 'nb of Trials :')
        for s in xrange(self.nbSessions):
            pyhrf.verbose(5, '- session %d :' %s)
            pyhrf.verbose.printNdarray(5, self.stimRepetitions[s])

        pyhrf.verbose(3, 'computing sampled binary onset sequences ...')

        rp = self.paradigm.get_rastered(self.dt)
        self.paradigmData = [rp[n] for n in self.cNames]

        pyhrf.verbose(5,'paradigm :')
        for iSess in xrange(self.nbSessions):
            pyhrf.verbose(5,' session: %d'%iSess)
            for iCond in xrange(self.nbConditions):
                m = 'cond:%d -> l=%d'  %(iCond,len(self.paradigmData[iCond][iSess]))
                pyhrf.verbose.printNdarray(5 ,m)
                pyhrf.verbose.printNdarray(6, self.paradigmData[iCond][iSess])
        # paradigm should be zero-expanded to match length of Bold data


        pyhrf.verbose(3, 'building paradigm convol matrix ...')
        availIdx = [arange(0,n, dtype=int) for n in self.nys] #availableDataIndex='
        self.buildParadigmConvolMatrix(hrfZc, hrfDuration, availIdx,
                                       self.paradigmData)
        pyhrf.verbose(5, 'matrix X : %s' %str(self.varX.shape))

        if pyhrf.verbose.verbosity >= 5:
            for s in xrange(self.varX.shape[0]):
                print '------ session %d' %s
                for i in xrange(self.varX.shape[1]):
                    print ''
                    for j in xrange(self.varX.shape[2]):
                        print '',
                        for k in xrange(self.varX.shape[3]):
                            print self.varX[s,i,j,k],
                        print ''
                    print ''
                print ''


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
            print lfdMat
            self.lfdMat.append(lfdMat)
            varPPt = np.dot(lfdMat, lfdMat.transpose())
            if typeLFD is not 'None':
                self.colP = shape(lfdMat)[1]
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

        regressors = self.tr*arange(0, n)
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
        lfdMat[:,0] = np.ones( ny, dtype= float)/sqrt(ny)
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
            matH = zeros( (lgt, self.nbConditions), dtype=int)
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

        self.hrfLength = len(np.arange(0,estimDuration+self.dt, self.dt))

        pyhrf.verbose(5, 'hrfLength = range(0,%f+%f,%f)=%d' \
                          % (estimDuration,self.dt,self.dt,self.hrfLength))
        if zc :
            self.hrfColIndex = arange(1, self.hrfLength-1, dtype=int)
            self.colIndex = arange(0, self.hrfLength-2, dtype=int)
        else :
            self.hrfColIndex = arange(0, self.hrfLength, dtype=int)
            self.colIndex = arange(0, self.hrfLength, dtype=int)
        self.lgCI = len(self.colIndex)
        pyhrf.verbose(5, 'lgCI = %d'% self.lgCI)

        self.varOSAvailDataIdx = [array(ai*osf, dtype=int)
                                  for ai in availableDataIndex]
        vX = []
        pyhrf.verbose(2, 'Build pseudo teoplitz matrices')
        for iSess in xrange(self.nbSessions):
            self.lenData = len(self.varOSAvailDataIdx[iSess])
            varX = zeros( (self.nbConditions, self.lenData, self.lgCI),
                          dtype=int )
            pyhrf.verbose(6, 'iSess : %d' %iSess)
            for j in xrange(self.nbConditions):
                pyhrf.verbose(6, ' cond : %d' %j)
                col = concatenate(([allMatH[iSess][0,j]],
                                   zeros(self.hrfLength-1, dtype=int)))
                pyhrf.verbose(6, ' col :')
                if pyhrf.verbose.verbosity >= 6:
                    print ' [',
                    for b in xrange(col.shape[0]):
                        print col[b],
                    print ']'


                matTmp = array(toeplitz( allMatH[iSess][:,j], col), dtype=int)
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

        self.nbColX = shape(self.varX[0])[2]

    def buildOtherMatX(self):
        self.varMBX=[]
        self.stackX=[]
        Id=[]

        for s in xrange(self.nbSessions):
            varMBX = zeros( (self.ny, self.nbConditions*self.lgCI),
                             dtype=int)
            stackX = zeros( (self.ny*self.nbConditions, self.lgCI),
                             dtype=int)

            for j in xrange(self.nbConditions):
                varMBX[:, j*self.lgCI+self.colIndex] = self.varX[s, j,:,:]

                stackX[self.ny*j:self.ny*(j+1), :] = self.varX[s, j,:,:]

            notNullIdxStack = dstack(where(stackX != 0)).ravel()

            self.varMBX.append(varMBX)
            self.stackX.append(stackX)
            Id.append(notNullIdxStack)

        self.varMBX = np.array(self.varMBX)
        self.stackX = np.array(self.stackX)
        self.notNullIdxStack = np.array(Id)




    def calcDt(self, dtMin) :

        pyhrf.verbose(2, 'dtMin = %1.3f' %dtMin)
        pyhrf.verbose(2, 'Trying to set dt automatically from data')

        tr = self.tr
        vectSOA = array([], dtype=float)
        for ons in self.onsets:
            vectSOA = concatenate((vectSOA, ons))

        vectSOA.sort()
        pyhrf.verbose(5, 'vectSOA %s:' %str(vectSOA.shape))
        pyhrf.verbose.printNdarray(5,vectSOA)

        momRT = arange(0, vectSOA[-1]+tr, tr)
        pyhrf.verbose(5,'momRT %s:' %str(momRT.shape))
        pyhrf.verbose.printNdarray(5, momRT)

        momVect = concatenate((vectSOA, momRT))
        momVect.sort()

        varSOA = diff(momVect)
        pyhrf.verbose(5, 'vectSOA diff:')
        pyhrf.verbose.printNdarray(5, vectSOA)

        nonZeroSOA = varSOA[where(varSOA > 0.0)]
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
        #del self.lfdMat
        del self.varPtP
        #del self.varMBY
        del self.paradigmData
        del self.neighboursIndexes
        #del self.varSingleCondXtrials
        del self.stimRepetitions
        del self.hrfColIndex
        del self.colIndex
        self.cleanPrecalculations()



class CallbackCritDiff(GSDefaultCallbackHandler):

    def __init__(self, parameters=None, xmlHandler=xmlio.TypedXMLHandler(), xmlLabel=None, xmlComment=None):
        GSDefaultCallbackHandler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)


    def callback(self, it, variables, samplerEngine):

        se = samplerEngine
        nbIterations = it+1
        for k,v in se.variablesMapping.iteritems():
            se.cumul_for_full_crit_diff[k] += v.currentValue.flatten()
            se.means_for_full_crit_diff[k] = se.cumul_for_full_crit_diff[k] / nbIterations
        old_vals = se.old_values_for_full_crit_diff
        se.crit_diff0.update(se.compute_crit_diff(old_vals, se.means_for_full_crit_diff))
        #print 'crit_diff0:', se.crit_diff0
        for k,v in se.crit_diff0.iteritems():
            se.full_crit_diff_trajectory[k].append(v)
        se.full_crit_diff_trajectory_timing.append(se.loop_timing)

class BOLDGibbsSampler(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment

    # Indices and labels of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'responseLevels'

    I_NOISE_VAR = 3
    P_NOISE_VAR = 'noiseVariance'

    I_RH = 4
    P_RH = 'HRFVariance'

    I_WEIGHTING_PROBA = 5
    P_WEIGHTING_PROBA = 'mixtureWeights'

    I_MIXT_PARAM = 2
    P_MIXT_PARAM = 'mixtureParameters'

    I_SCALE = 6
    P_SCALE = 'scale'

    I_BETA = 7
    P_BETA = 'beta'

    I_HRF = 1
    P_HRF = 'HRF'

    #I_LIK = 8
    #P_LIK = 'LIKELIHOOD'

    inputClass = WN_BiG_BOLDSamplerInput

    #variablesToSample = [P_NRLS, P_HRF, P_MIXT_PARAM, P_NOISE_VAR, P_RH, P_WEIGHTING_PROBA, P_SCALE, P_BETA, P_LIK]
    variablesToSample = [P_NRLS, P_HRF, P_MIXT_PARAM, P_NOISE_VAR, P_RH, P_WEIGHTING_PROBA, P_SCALE, P_BETA]

    P_NB_ITERATIONS = 'nbIterations'
    P_OBS_HIST_PACE = 'observablesHistoryPaceSave'
    P_GLOB_OBS_HIST_PACE = 'globalObservablesHistoryPaceSave'
    P_SMPL_HIST_PACE = 'samplesHistoryPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'
    P_STOP_THRESHOLD = 'stop_criterion_threshold'
    P_CRIT_DIFF_FROM_START = 'crit_diff_from_start'
    P_CHECK_FINAL_VALUE = 'check_final_value_with_truth'

    defaultParameters = {
        P_OBS_HIST_PACE : -1.,
        P_GLOB_OBS_HIST_PACE : -1,
        P_SMPL_HIST_PACE : -1.,
        P_NB_SWEEPS : .3,
        P_RANDOM_SEED : 193843200,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRLSampler(),
        P_BETA : BetaSampler(),
        P_NOISE_VAR : NoiseVarianceSampler(),
        P_HRF : HRFSampler(),
        P_RH : RHSampler(),
        P_WEIGHTING_PROBA : MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        P_STOP_THRESHOLD : -1,
        P_CRIT_DIFF_FROM_START : False,#False,
        P_CHECK_FINAL_VALUE : None, # 'print', 'raise'
        #P_LIK : LikelihoodSampler(),
        }


    if pyhrf.__usemode__ == pyhrf.DEVEL:
        defaultParameters[P_NB_ITERATIONS] = 3
        parametersToShow = [P_NB_ITERATIONS, P_SMPL_HIST_PACE, P_OBS_HIST_PACE,
                            P_GLOB_OBS_HIST_PACE,
                            P_NB_SWEEPS,
                            P_RANDOM_SEED,
                            P_CALLBACK, P_NRLS, P_BETA, P_NOISE_VAR,
                            P_HRF, P_RH,
                            P_WEIGHTING_PROBA,  P_MIXT_PARAM, P_SCALE,
                            P_STOP_THRESHOLD, P_CRIT_DIFF_FROM_START
                            #,
                            #P_LIK
                            ]

    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        defaultParameters[P_NB_ITERATIONS] = 3000
        parametersToShow = [P_NB_ITERATIONS, P_NRLS, P_HRF, P_RH, P_RANDOM_SEED]

    parametersComments = {
        P_SMPL_HIST_PACE: 'To save the samples at each iteration\n'\
            'If x<0: no save\n ' \
            'If 0<x<1: define the fraction of iterations for which samples are saved\n'\
            'If x>=1: define the step in iterations number between backup copies.\n'\
            'If x=1: save samples at each iteration.',
        P_OBS_HIST_PACE: 'See comment for samplesHistoryPaceSave.'
        }

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        #print 'param:', parameters
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        variables = [self.parameters[vLab] for vLab in self.variablesToSample]
        #print self.variablesToSample
        #for vLab in self.variablesToSample:
             #print vLab

        nbIt = self.parameters[self.P_NB_ITERATIONS]
        obsHistPace = self.parameters[self.P_OBS_HIST_PACE]
        globalObsHistPace = self.parameters[self.P_GLOB_OBS_HIST_PACE]
        smplHistPace = self.parameters[self.P_SMPL_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]

        check_ftval = self.parameters[self.P_CHECK_FINAL_VALUE]

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

        self.stop_threshold = self.parameters[self.P_STOP_THRESHOLD]
        self.crit_diff_from_start = self.parameters[self.P_CRIT_DIFF_FROM_START]
        seed = self.parameters[self.P_RANDOM_SEED]
        #callbackObj = self.parameters[self.P_CALLBACK]
        self.full_crit_diff_trajectory = defaultdict(list)
        self.full_crit_diff_trajectory_timing = []
        self.crit_diff0 = {}
        #print 'self.crit_diff_from_start:', self.crit_diff_from_start
        if self.crit_diff_from_start:
            callbackObj = CallbackCritDiff()
        else:
            callbackObj = GSDefaultCallbackHandler()
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps,
                              callbackObj, randomSeed=seed,
                              globalObsHistoryPace=globalObsHistPace,
                              check_ftval=check_ftval)


        ##self.buildSharedDataTree()

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
        BOLDGibbsSampler.cleanObservables(self)

        # del self.h_old
        # del self.labels_old
        # del self.nrls_old
        del self.variables_old_val

    def saveGlobalObservables(self, it):
        #print 'saveGlobalObservables ...'
        GibbsSampler.saveGlobalObservables(self, it)
        if self.stop_threshold >= 0.:
            for vn, d in self.crit_diff.iteritems():
                self.conv_crit_diff[vn].append(d)

        # self.conv_crit_diff_h.append(self.crit_diff_h)
        # self.conv_crit_diff_nrls.append(self.crit_diff_nrls)
        # self.conv_crit_diff_labels.append(self.crit_diff_labels)

    def buildSharedDataTree(self):

        self.sharedData = Pipeline()
        self.regVarsInPipeline()

        computeRules = []
        # Some more roots :
        computeRules.append({'label' : 'matXQX', 'ref' : self.matXQX})
        computeRules.append({'label' : 'varX', 'ref' : self.varX})
        computeRules.append({'label' : 'varMBY', 'ref' : self.varMBY})

        # Add shared quantities to update during sampling :
        computeRules.append({'label' : 'hXQXh', 'dep' : ['hrf','matXQX'],
                             'computeFunc' : computehXQXh})
        computeRules.append({'label' : 'matXh' , 'dep' : ['varX', 'hrf'],
                             'computeFunc' : computeXh})

        computeRules.append({'label' : 'sumj_aXh', 'dep' : ['matXh','nrl'],
                             'computeFunc' : computeSumjaXh})

        computeRules.append({'label' : 'yTilde', 'dep' : ['sumj_aXh','varMBY'],
                             'computeFunc' : computeYTilde})


    def computeFit(self):
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if hrf is None:
            hrf = shrf.currentValue
        elif shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl').finalValue
        if nrl is None:
            nrl = self.getVariable('nrl').currentValue

        stimIndSignal = np.zeros((nbVals, nbVox), dtype=np.float32)

        stimIndSignal = np.dot(vXh, nrl)


        # add lsq fit of drift:
        p = self.dataInput.lfdMat[0]
        stimIndSignal += np.dot(p, np.dot(p.T, self.dataInput.varMBY))

        # for i in xrange(nbVox):
        #     # Multiply by corresponding NRLs and sum over conditions :
        #     si = (vXh*nrl[:,i]).sum(1)
        #     # Adjust mean to original Bold (because drift is not explicit):
        #     stimIndSignal[:,i] = si-si.mean() +  meanBold[i]

        #raise Exception()

        return stimIndSignal


    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)
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

            # if hasattr(self,'conv_crit_diff_h'):
            #     it_axis = self.globalObsHistoryIts
            #     axes_domains = {'iteration':it_axis}
            #     a_conv_crit = np.array(self.conv_crit_diff_h)
            #     outputs['conv_crit_diff_h'] = xndarray(a_conv_crit,
            #                                          axes_names=['iteration'],
            #                                          axes_domains=axes_domains,
            #                                          value_label='conv_criterion')
            # if hasattr(self,'conv_crit_diff_nrls'):
            #     it_axis = self.globalObsHistoryIts
            #     axes_domains = {'iteration':it_axis}
            #     a_conv_crit = np.array(self.conv_crit_diff_nrls)
            #     outputs['conv_crit_diff_nrls'] = xndarray(a_conv_crit,
            #                                          axes_names=['iteration'],
            #                                          axes_domains=axes_domains,
            #                                          value_label='conv_criterion')

            # if hasattr(self,'conv_crit_diff_labels'):
            #     it_axis = self.globalObsHistoryIts
            #     axes_domains = {'iteration':it_axis}
            #     a_conv_crit = np.array(self.conv_crit_diff_labels)
            #     outputs['conv_crit_diff_labels'] = xndarray(a_conv_crit,
            #                                               axes_names=['iteration'],
            #                                               axes_domains=axes_domains,
            #                                               value_label='conv_criterion')

        return outputs

    def computePMStimInducedSignal(self):

        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl').finalValue

        self.stimIndSignal = np.zeros((nbVals, nbVox))
        meanBold = self.dataInput.varMBY.mean(axis=0)

        for i in xrange(nbVox):
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.stimIndSignal[:,i] = si-si.mean() +  meanBold[i]


class BOLDGibbsSampler_AR(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment

    # Indices of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_NOISE_ARP = 2
    P_NOISE_ARP = 'noiseARParam'

    I_HRF = 3
    P_HRF = 'hrf'

    I_RH = 4
    P_RH = 'hrfVar'

    I_DRIFT = 5
    P_DRIFT = 'drift'

    I_ETA = 6
    P_ETA = 'driftVar'

    I_WEIGHTING_PROBA = 7
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 8
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 9
    P_SCALE = 'scale'

    I_BETA = 10
    P_BETA = 'beta'

    inputClass = ARN_BiG_BOLDSamplerInput

    variablesToSample = [ P_NRLS, P_NOISE_VAR, P_NOISE_ARP,  P_HRF,
                          P_RH, P_DRIFT, P_ETA, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA ]

    P_NB_ITERATIONS = 'nbIterations'
    P_OBS_HIST_PACE = 'observablesHistoryPaceSave'
    P_GLOB_OBS_HIST_PACE = 'globalObservablesHistoryPaceSave'
    P_SMPL_HIST_PACE = 'samplesHistoryPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'
    P_STOP_THRESHOLD = 'stop_criterion_threshold'
    P_CRIT_DIFF_FROM_START = 'crit_diff_from_start'

    defaultParameters = {
        P_OBS_HIST_PACE : -1.,
        P_SMPL_HIST_PACE : -1.,
        P_GLOB_OBS_HIST_PACE : -1,
        P_NB_SWEEPS : .3,
        P_NB_ITERATIONS : 3,
        P_RANDOM_SEED : 193843200,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRLARSampler(),
        P_NOISE_VAR : NoiseVarianceARSampler(),
        P_NOISE_ARP : NoiseARParamsSampler(),
        P_HRF : HRFARSampler(),
        P_RH : RHSampler(),
        P_DRIFT : DriftARSampler(),
        P_ETA : ETASampler(),
        P_WEIGHTING_PROBA: MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        P_BETA : BetaSampler(),
        P_STOP_THRESHOLD : -1,
        P_CRIT_DIFF_FROM_START : False,
        }

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        defaultParameters[P_NB_ITERATIONS] = 3
        parametersToShow = [P_NB_ITERATIONS, P_SMPL_HIST_PACE, P_OBS_HIST_PACE,
                            P_GLOB_OBS_HIST_PACE,
                            P_NB_SWEEPS,
                            P_RANDOM_SEED,
                            P_CALLBACK, P_NRLS, P_BETA, P_NOISE_VAR,
                            P_HRF, P_RH,
                            P_WEIGHTING_PROBA,  P_MIXT_PARAM, P_SCALE,
                            P_STOP_THRESHOLD, P_CRIT_DIFF_FROM_START]

    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        defaultParameters[P_NB_ITERATIONS] = 3000
        parametersToShow = [P_NB_ITERATIONS, P_NRLS, P_HRF, P_RH, P_RANDOM_SEED]

    parametersComments = {
        P_SMPL_HIST_PACE: 'To save the samples at each iteration\n'\
            'If x<0: no save\n ' \
            'If 0<x<1: define the fraction of iterations for which samples are saved\n'\
            'If x>=1: define the step in iterations number between backup copies.\n'\
            'If x=1: save samples at each iteration.',
        P_OBS_HIST_PACE: 'See comment for samplesHistoryPaceSave.'
        }

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        #print 'param:', parameters
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        variables = [self.parameters[vLab] for vLab in self.variablesToSample]
        #print self.variablesToSample
        #for vLab in self.variablesToSample:
             #print vLab

        nbIt = self.parameters[self.P_NB_ITERATIONS]
        obsHistPace = self.parameters[self.P_OBS_HIST_PACE]
        globalObsHistPace = self.parameters[self.P_GLOB_OBS_HIST_PACE]
        smplHistPace = self.parameters[self.P_SMPL_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]

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

        self.stop_threshold = self.parameters[self.P_STOP_THRESHOLD]
        self.crit_diff_from_start = self.parameters[self.P_CRIT_DIFF_FROM_START]
        seed = self.parameters[self.P_RANDOM_SEED]
        #callbackObj = self.parameters[self.P_CALLBACK]
        self.full_crit_diff_trajectory = defaultdict(list)
        self.full_crit_diff_trajectory_timing = []
        self.crit_diff0 = {}
        #print 'self.crit_diff_from_start:', self.crit_diff_from_start
        if self.crit_diff_from_start:
            callbackObj = CallbackCritDiff()
        else:
            callbackObj = GSDefaultCallbackHandler()
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps,
                              callbackObj, randomSeed=seed,
                              globalObsHistoryPace=globalObsHistPace)


        ##self.buildSharedDataTree()

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
        BOLDGibbsSampler.cleanObservables(self)

        # del self.h_old
        # del self.labels_old
        # del self.nrls_old
        del self.variables_old_val

    def saveGlobalObservables(self, it):
        #print 'saveGlobalObservables ...'
        GibbsSampler.saveGlobalObservables(self, it)
        if self.stop_threshold >= 0.:
            for vn, d in self.crit_diff.iteritems():
                self.conv_crit_diff[vn].append(d)

        # self.conv_crit_diff_h.append(self.crit_diff_h)
        # self.conv_crit_diff_nrls.append(self.crit_diff_nrls)
        # self.conv_crit_diff_labels.append(self.crit_diff_labels)

    def buildSharedDataTree(self):

        self.sharedData = Pipeline()
        self.regVarsInPipeline()

        computeRules = []
        # Some more roots :
        computeRules.append({'label' : 'matXQX', 'ref' : self.matXQX})
        computeRules.append({'label' : 'varX', 'ref' : self.varX})
        computeRules.append({'label' : 'varMBY', 'ref' : self.varMBY})

        # Add shared quantities to update during sampling :
        computeRules.append({'label' : 'hXQXh', 'dep' : ['hrf','matXQX'],
                             'computeFunc' : computehXQXh})
        computeRules.append({'label' : 'matXh' , 'dep' : ['varX', 'hrf'],
                             'computeFunc' : computeXh})

        computeRules.append({'label' : 'sumj_aXh', 'dep' : ['matXh','nrl'],
                             'computeFunc' : computeSumjaXh})

        computeRules.append({'label' : 'yTilde', 'dep' : ['sumj_aXh','varMBY'],
                             'computeFunc' : computeYTilde})


    def computeFit(self):
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if hrf is None:
            hrf = shrf.currentValue
        elif shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl').finalValue
        if nrl is None:
            nrl = self.getVariable('nrl').currentValue

        stimIndSignal = np.zeros((nbVals, nbVox), dtype=np.float32)

        stimIndSignal = np.dot(vXh, nrl)


        # add lsq fit of drift:
        p = self.dataInput.lfdMat[0]
        stimIndSignal += np.dot(p, np.dot(p.T, self.dataInput.varMBY))

        # for i in xrange(nbVox):
        #     # Multiply by corresponding NRLs and sum over conditions :
        #     si = (vXh*nrl[:,i]).sum(1)
        #     # Adjust mean to original Bold (because drift is not explicit):
        #     stimIndSignal[:,i] = si-si.mean() +  meanBold[i]

        #raise Exception()

        return stimIndSignal


    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)
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

        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl').finalValue

        self.stimIndSignal = np.zeros((nbVals, nbVox))
        meanBold = self.dataInput.varMBY.mean(axis=0)

        for i in xrange(nbVox):
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.stimIndSignal[:,i] = si-si.mean() +  meanBold[i]



class W_BOLDGibbsSampler(BOLDGibbsSampler):

    defaultParameters = copyModule.deepcopy(BOLDGibbsSampler.defaultParameters)

    # Indices and labels of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'responseLevels'

    I_HRF = 1
    P_HRF = 'HRF'

    I_NOISE_VAR = 2
    P_NOISE_VAR = 'noiseVariance'

    I_RH = 3
    P_RH = 'HRFVariance'

    I_WEIGHTING_PROBA = 4
    P_WEIGHTING_PROBA = 'mixtureWeights'

    I_MIXT_PARAM = 5
    P_MIXT_PARAM = 'mixtureParameters'

    I_SCALE = 6
    P_SCALE = 'scale'

    I_BETA = 7
    P_BETA = 'beta'

    I_W = 8
    P_W = 'relevantVariable'

    inputClass = WN_BiG_BOLDSamplerInput

    defaultParameters.update({
        P_NRLS : NRLSamplerWithRelVar(),
        P_NOISE_VAR : NoiseVarianceSamplerWithRelVar(),
        P_HRF : HRFSamplerWithRelVar(),
        P_MIXT_PARAM : BiGaussMixtureParamsSamplerWithRelVar(),
        P_W : WSampler()
        })

    variablesToSample = [P_NRLS,  P_HRF, P_NOISE_VAR,
                          P_RH, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA, P_W]

    parametersToShow = copyModule.deepcopy(BOLDGibbsSampler.parametersToShow)
    parametersToShow += [P_W]

    #print variablesToSample
    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                    xmlLabel=None, xmlComment=None):

        BOLDGibbsSampler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)

    #def buildSharedDataTree(self):
        #BOLDGibbsSampler.buildSharedDataTree(self)
        #computeRules.append({'label' : 'varQg', 'dep' : ['delta','matXh'],
                             #'computeFunc' : computeQXh})


class GGG_BOLDGibbsSampler(BOLDGibbsSampler):


    defaultParameters = copyModule.deepcopy(BOLDGibbsSampler.defaultParameters)
    #P_BF_UPDATER = 'BayesFactorUpdater'
    #I_BF_UPDATER = 7
    defaultParameters.update({
        BOLDGibbsSampler.P_NRLS : GGGNRLSampler(),
        BOLDGibbsSampler.P_MIXT_PARAM : TriGaussMixtureParamsSampler(),
        })


    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                    xmlLabel=None, xmlComment=None):

        BOLDGibbsSampler.__init__(self, parameters, xmlHandler, xmlLabel,
                                  xmlComment)

class BOLDGamGaussGibbsSampler(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment

    # Indices and labels of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_HRF = 2
    P_HRF = 'hrf'

    I_RH = 3
    P_RH = 'hrfVar'

    I_WEIGHTING_PROBA = 4
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 5
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 6
    P_SCALE = 'scale'

    variablesMapping = {
        P_NRLS : I_NRLS,
        P_NOISE_VAR : I_NOISE_VAR,
        P_HRF : I_HRF,
        P_RH : I_RH,
        P_WEIGHTING_PROBA : I_WEIGHTING_PROBA,
        P_MIXT_PARAM : I_MIXT_PARAM,
        P_SCALE : I_SCALE,
        }


    P_NB_ITERATIONS = 'nbIterations'
    P_HIST_PACE = 'historyPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'

    defaultParameters = {
        P_NB_ITERATIONS : 3,
        P_HIST_PACE : 2,
        P_NB_SWEEPS : None,
        P_RANDOM_SEED : None,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : InhomogeneousNRLSampler(),
        P_NOISE_VAR : NoiseVarianceSampler(),
        P_HRF : HRFSampler(),
        P_RH : RHSampler(),
        P_WEIGHTING_PROBA : MixtureWeightsSampler(),
        P_MIXT_PARAM : GamGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        }

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(), xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)

        variables = range(len(self.variablesMapping))
        for vLab, vIndex in self.variablesMapping.items() :
            variables[vIndex] = self.parameters[vLab]
        nbIt = self.parameters[self.P_NB_ITERATIONS]
        histPace = self.parameters[self.P_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]
        seed = self.parameters[self.P_RANDOM_SEED]
        if nbSweeps == None :
            nbSweeps = nbIt/3
        callbackObj = self.parameters[self.P_CALLBACK]
        GibbsSampler.__init__(self, variables, nbIt, histPace, nbSweeps, callbackObj, randomSeed=seed)

    def computePMStimInducedSignal(self):

        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        vXh = shrf.varXh # base convolution
        nrl = self.getVariable('nrl').finalValue


        self.stimIndSignal = np.zeros((nbVals, nbVox))
        meanBold = self.dataInput.varMBY.mean(axis=0)

        for i in xrange(nbVox):
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.stimIndSignal[:,i] = si-si.mean() +  meanBold[i]


class Drift_BOLDGibbsSampler(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment Class for Gibbs sampling algorithm using a AR(1) noise model

    # Indices of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_HRF = 2
    P_HRF = 'hrf'

    I_RH = 3
    P_RH = 'hrfVar'

    I_DRIFT = 4
    P_DRIFT = 'drift'

    I_ETA = 5
    P_ETA = 'driftVar'

    I_WEIGHTING_PROBA = 6
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 7
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 8
    P_SCALE = 'scale'

    I_BETA = 9
    P_BETA = 'beta'

    inputClass = WN_BiG_Drift_BOLDSamplerInput

    variablesToSample = [ P_NRLS, P_NOISE_VAR,  P_HRF,
                          P_RH, P_DRIFT, P_ETA, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA ]

    P_NB_ITERATIONS = 'nbIterations'
    P_OBS_HIST_PACE = 'observablesHistoryPaceSave'
    P_SMPL_HIST_PACE = 'samplesHistoryPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'

    defaultParameters = {
        P_OBS_HIST_PACE : -1.,
        P_SMPL_HIST_PACE : -1.,
        P_NB_SWEEPS : .3,
        P_NB_ITERATIONS : 3,
        P_RANDOM_SEED : 193843200,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRL_Drift_Sampler(),
        P_NOISE_VAR : NoiseVariance_Drift_Sampler(),
        P_HRF : HRF_Drift_Sampler(),
        P_RH : RHSampler(),
        P_DRIFT : DriftSampler(),
        P_ETA : ETASampler(),
        P_WEIGHTING_PROBA: MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        P_BETA : BetaSampler(),
        }




    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)

        variables = [self.parameters[vLab] for vLab in self.variablesToSample]

        nbIt = self.parameters[self.P_NB_ITERATIONS]

        obsHistPace = self.parameters[self.P_OBS_HIST_PACE]
        smplHistPace = self.parameters[self.P_SMPL_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]

        if obsHistPace > 0. and obsHistPace < 1:
            obsHistPace = max(1,int(round(nbIt * obsHistPace)))

        if smplHistPace > 0. and smplHistPace < 1.:
            smplHistPace = max(1,int(round(nbIt * smplHistPace)))

        if nbSweeps > 0. and nbSweeps < 1.:
            nbSweeps = int(round(nbIt * nbSweeps))


        seed = self.parameters[self.P_RANDOM_SEED]
        callbackObj = self.parameters[self.P_CALLBACK]
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace,
                              nbSweeps,
                              callbackObj, randomSeed=seed)



    def computeFit(self):
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        if hrf is None:
            hrf = shrf.currentValue
        elif shrf.zc:
            hrf = hrf[1:-1]
        vXh = shrf.calcXh(hrf) # base convolution
        nrl = self.getVariable('nrl').finalValue
        if nrl is None:
            nrl = self.getVariable('nrl').currentValue

        stimIndSignal = np.dot(vXh, nrl)

        sdrift = self.getVariable('drift')
        stimIndSignal += np.dot(sdrift.P, sdrift.finalValue)

        return stimIndSignal


class W_Drift_BOLDGibbsSampler(xmlio.XMLParamDrivenClass, Drift_BOLDGibbsSampler):

    #TODO : comment Class for Gibbs sampling algorithm using a AR(1) noise model

    # Indices of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_HRF = 2
    P_HRF = 'hrf'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_RH = 3
    P_RH = 'hrfVar'

    I_DRIFT = 4
    P_DRIFT = 'drift'

    I_ETA = 5
    P_ETA = 'driftVar'

    I_WEIGHTING_PROBA = 6
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 7
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 8
    P_SCALE = 'scale'

    I_BETA = 9
    P_BETA = 'beta'

    I_W = 10
    P_W = 'relevantVariable'

    inputClass = WN_BiG_Drift_BOLDSamplerInput

    variablesToSample = [ P_NRLS, P_NOISE_VAR, P_HRF,
                          P_RH, P_DRIFT, P_ETA, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA, P_W ]

    P_NB_ITERATIONS = 'nbIterations'
    P_OBS_HIST_PACE = 'observablesHistoryPaceSave'
    P_SMPL_HIST_PACE = 'samplesHistoryPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'

    defaultParameters = {
        P_OBS_HIST_PACE : -1.,
        P_SMPL_HIST_PACE : -1.,
        P_NB_SWEEPS : .3,
        P_NB_ITERATIONS : 3,
        P_RANDOM_SEED : 193843200,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRL_Drift_SamplerWithRelVar(),
        P_NOISE_VAR : NoiseVariance_Drift_Sampler(),
        P_HRF : HRF_Drift_SamplerWithRelVar(),
        P_RH : RHSampler(),
        P_DRIFT : DriftSamplerWithRelVar(),
        P_ETA : ETASampler(),
        P_WEIGHTING_PROBA: MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSamplerWithRelVar(),
        P_SCALE : ScaleSampler(),
        P_BETA : BetaSampler(),
        P_W : W_Drift_Sampler(),
        }

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        Drift_BOLDGibbsSampler.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)


class ARN_BOLDGibbsSampler(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment Class for Gibbs sampling algorithm using a AR(1) noise model

    # Indices of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_NOISE_ARP = 2
    P_NOISE_ARP = 'noiseARParam'

    I_HRF = 3
    P_HRF = 'hrf'

    I_RH = 4
    P_RH = 'hrfVar'

    I_DRIFT = 5
    P_DRIFT = 'drift'

    I_ETA = 6
    P_ETA = 'driftVar'

    I_WEIGHTING_PROBA = 7
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 8
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 9
    P_SCALE = 'scale'

    I_BETA = 10
    P_BETA = 'beta'

    inputClass = ARN_BiG_BOLDSamplerInput

    variablesToSample = [ P_NRLS, P_NOISE_VAR, P_NOISE_ARP,  P_HRF,
                          P_RH, P_DRIFT, P_ETA, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA ]

    P_NB_ITERATIONS = 'nbIterations'
    P_OBS_HIST_PACE = 'observablesHistoryPaceSave'
    P_SMPL_HIST_PACE = 'samplesHistoryPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'

    defaultParameters = {
        P_OBS_HIST_PACE : -1.,
        P_SMPL_HIST_PACE : -1.,
        P_NB_SWEEPS : .3,
        P_NB_ITERATIONS : 3,
        P_RANDOM_SEED : 193843200,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRLARSampler(),
        P_NOISE_VAR : NoiseVarianceARSampler(),
        P_NOISE_ARP : NoiseARParamsSampler(),
        P_HRF : HRFARSampler(),
        P_RH : RHSampler(),
        P_DRIFT : DriftARSampler(),
        P_ETA : ETASampler(),
        P_WEIGHTING_PROBA: MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        P_BETA : BetaSampler(),
        }


    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)

        variables = [self.parameters[vLab] \
                         for vLab in self.variablesToSample]
        nbIt = self.parameters[self.P_NB_ITERATIONS]
        obsHistPace = self.parameters[self.P_OBS_HIST_PACE]
        smplHistPace = self.parameters[self.P_SMPL_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]

        if obsHistPace > 0. and obsHistPace < 1:
            obsHistPace = max(1,int(round(nbIt * obsHistPace)))

        if smplHistPace > 0. and smplHistPace < 1.:
            smplHistPace = max(1,int(round(nbIt * smplHistPace)))

        if nbSweeps > 0. and nbSweeps < 1.:
            nbSweeps = int(round(nbIt * nbSweeps))


        seed = self.parameters[self.P_RANDOM_SEED]
        callbackObj = self.parameters[self.P_CALLBACK]
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace,
                              nbSweeps,
                              callbackObj, randomSeed=seed)

        ##self.buildSharedDataTree()

    # def buildSharedDataTree(self):

    #     self.sharedData = Pipeline()
    #     self.regVarsInPipeline()

    #     computeRules = []
    #     # Some more roots :
    #     computeRules.append({'label' : 'varX', 'ref' : self.varX})
    #     computeRules.append({'label' : 'varP', 'ref' : self.varP})
    #     computeRules.append({'label' : 'varMBY', 'ref' : self.varMBY})

    #     # Add shared quantities to update during sampling :
    #     computeRules.append({'label' : 'matXh' , 'dep' : ['varX', 'hrf'],
    #                          'computeFunc' : computeXh})
    #     computeRules.append({'label' : 'varPl' , 'dep' : ['varP', 'drift'],
    #                          'computeFunc' : computePl})
    #     computeRules.append({'label' : 'sumj_aXh', 'dep' : ['matXh','nrl'],
    #                          'computeFunc' : computeSumjaXh})

    #     computeRules.append({'label' : 'yBar', 'dep' : ['varMBY', 'varPl'],
    #                          'computeFunc' : computeYBar})

    #     computeRules.append({'label' : 'yTilde', 'dep' : ['sumj_aXh','yBar'],
    #                          'computeFunc' : computeYTilde_Pl})

    def computePMStimInducedSignal(self):

        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        vXh = shrf.varXh # base convolution
        nrl = self.getVariable('nrl').finalValue

        self.stimIndSignal = np.zeros((nbVals, nbVox))
        meanBold = self.dataInput.varMBY.mean(axis=0)

        for i in xrange(nbVox):
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.stimIndSignal[:,i] = si-si.mean() +  meanBold[i]

    def computeFittedSignal(self):
        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        vXh = shrf.varXh # base convolution
        nrl = self.getVariable('nrl').finalValue
        sdrift = self.getVariable('drift')
        vPl = sdrift.Pl

        self.fittedSignal = np.zeros((nbVals, nbVox))

        for i in xrange(nbVox):
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.fittedSignal[:,i] = si + vPl[:,i]


##class Hab_WN_BiG_BOLDSamplerInput(WN_BiG_BOLDSamplerInput):

    ##def makePrecalculations(self):
        ### Qy, yTQ & yTQy  :
        ##self.matQy = np.zeros((self.ny,self.nbVoxels), dtype=float)
        ##self.yTQy = np.zeros(self.nbVoxels, dtype=float)

        ##for i in xrange(self.nbVoxels):
            ##self.matQy[:,i] = dot(self.delta,self.varMBY[:,i])
###            self.yTQ[:,i] = dot(self.varMBY[:,i],self.delta)
            ##self.yTQy[i] = dot(self.varMBY[:,i],self.matQy[:,i])

class HAB_BOLDGibbsSampler(xmlio.XMLParamDrivenClass, GibbsSampler):

    #TODO : comment

    # Indices and labels of each variable registered in sampler :
    I_NRLS = 0
    P_NRLS = 'nrl'

    I_NOISE_VAR = 1
    P_NOISE_VAR = 'noiseVar'

    I_HRF = 2
    P_HRF = 'hrf'

    I_RH = 3
    P_RH = 'hrfVar'

    I_WEIGHTING_PROBA = 4
    P_WEIGHTING_PROBA = 'mixtureWeight'

    I_MIXT_PARAM = 5
    P_MIXT_PARAM = 'mixtureParams'

    I_SCALE = 6
    P_SCALE = 'scale'

#     I_HAB = 7
#     P_HAB = 'habituationSpeed'

    I_BETA = 7
    P_BETA = 'beta'

    inputClass = Hab_WN_BiG_BOLDSamplerInput

    variablesToSample = [ P_NRLS, P_NOISE_VAR, P_HRF, P_RH, P_WEIGHTING_PROBA,
                          P_MIXT_PARAM, P_SCALE, P_BETA ]

    P_NB_ITERATIONS = 'nbIterations'
    P_HIST_PACE = 'historyPaceSave'
    P_NB_SWEEPS = 'nbSweeps'
    P_CALLBACK = 'callBackHandler'
    P_RANDOM_SEED = 'numpyRandomSeed'

    defaultParameters = {
        P_NB_ITERATIONS : 3,
        P_HIST_PACE : -1,
        P_NB_SWEEPS : None,
        P_RANDOM_SEED : None,
        P_CALLBACK : GSDefaultCallbackHandler(),
        P_NRLS : NRLwithHabSampler(), ## nrl trial-dependent sampling
        P_BETA : BetaSampler(),
        P_NOISE_VAR : NoiseVariancewithHabSampler(),
        P_HRF : HRFwithHabSampler(),
        P_RH : RHSampler(),
        P_WEIGHTING_PROBA : MixtureWeightsSampler(),
        P_MIXT_PARAM : BiGaussMixtureParamsSampler(),
        P_SCALE : ScaleSampler(),
        }
#        P_HRF : HRFSampler(),

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        parametersToShow = [P_NB_ITERATIONS, P_HIST_PACE , P_NB_SWEEPS,
                            P_RANDOM_SEED,
                            P_CALLBACK, P_NRLS, P_BETA, P_NOISE_VAR,
                            P_HRF, P_RH,
                            P_WEIGHTING_PROBA,  P_MIXT_PARAM, P_SCALE]
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        #defaultParameters[P_NB_ITERATIONS] = 2000
        parametersToShow = [P_NB_ITERATIONS, P_NRLS, P_HRF, P_RANDOM_SEED]


    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        #TODO : comment

        """
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)
        variables = [self.parameters[vLab] for vLab in self.variablesToSample]
        nbIt = self.parameters[self.P_NB_ITERATIONS]
        histPace = self.parameters[self.P_HIST_PACE]
        nbSweeps = self.parameters[self.P_NB_SWEEPS]
        seed = self.parameters[self.P_RANDOM_SEED]
        if nbSweeps == None :
            nbSweeps = nbIt/3
        callbackObj = self.parameters[self.P_CALLBACK]
        GibbsSampler.__init__(self, variables, nbIt, histPace,
                              nbSweeps=nbSweeps,
                              callbackObj=callbackObj, randomSeed=seed)

    def computePMStimInducedSignal(self):

        nbCond = self.dataInput.nbConditions
        nbVox = self.dataInput.nbVoxels

        nbVals = self.dataInput.ny
        shrf = self.getVariable('hrf')
        hrf = shrf.finalValue
        vXh = shrf.varXh # base convolution
        snrl = self.getVariable('nrl')
        nrl = snrl.finalValue
#        habit = self.getVariable('habituationSpeed').finalValue


        self.stimIndSignal = np.zeros((nbVals, nbVox))
        meanBold = self.dataInput.varMBY.mean(axis=0)

        for i in xrange(nbVox):
            # MORE TO COME HERE ABOUT COMPUTATION OF STIM INDUCED SIGNAL
            # Multiply by corresponding NRLs and sum over conditions :
            si = (vXh*nrl[:,i]).sum(1)
            # Adjust mean to original Bold (because drift is not explicit):
            self.stimIndSignal[:,i] = si-si.mean() +  meanBold[i]





import pyhrf.boldsynth.scenarios as sim
from pyhrf import Condition

def simulate_bold(output_dir=None, noise_scenario='high_snr',
                  spatial_size='tiny', normalize_hrf=True):

    drift_var = 10.

    dt = .5
    dsf = 2 #down sampling factor

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    else:
        lmap1, lmap2, lmap3 = 'icassp13', 'ghost', 'house_sun'

    if noise_scenario == 'high_snr':
        v_noise = 0.1
        conditions = [
            Condition(name='audio',
                      m_act=15., v_act=.1, v_inact=.2,
                      label_map=lmap1),
            Condition(name='video',
                      m_act=14., v_act=.11, v_inact=.21,
                      label_map=lmap2),
            Condition(name='damier',
                      m_act=20., v_act=.12, v_inact=.22,
                      label_map=lmap3),
                      ]
    else: #low_snr
        v_noise = 2.
        conditions = [
            Condition(name='audio',
                      m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name='video',
                      m_act=2.2, v_act=.3, v_inact=.3,
                      label_map=lmap2),
                      ]

    simulation_steps = {
        'dt' : dt,
        'dsf' : dsf,
        'tr' : dt * dsf,
        'condition_defs' : conditions,
        # Paradigm
        'paradigm' : sim.create_localizer_paradigm_avd,
        'rastered_paradigm' : sim.rasterize_paradigm,
        # Labels
        'labels_vol' : sim.create_labels_vol,
        'labels' : sim.flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # NRLs
        'nrls' : sim.create_time_invariant_gaussian_nrls,
        # HRF
        'hrf_var' : 0.1,
        'primary_hrf' : sim.create_gsmooth_hrf,
        'normalize_hrf' : normalize_hrf,
        'hrf' : sim.duplicate_hrf,
        # Stim induced
        'stim_induced' : sim.create_stim_induced_signal,
        # Noise
        'v_gnoise' : v_noise,
        'noise' : sim.create_gaussian_noise,
        # Drift
        'drift_order' : 4,
        'drift_coeff_var' : drift_var,
        'drift_coeffs': sim.create_drift_coeffs,
        'drift' : sim.create_polynomial_drift_from_coeffs,
        # Final BOLD signal
        'bold_shape' : sim.calc_bold_shape,
        'bold' : sim.create_bold_from_stim_induced,
        }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.update_all()
    simulation = simu_graph.get_values()

    if output_dir is not None:
        try:
            simu_graph.save_graph_plot(op.join(output_dir,
                                               'simulation_graph.png'))
        except ImportError: #if pygraphviz not available
            pass

        sim.simulation_save_vol_outputs(simulation, output_dir)

        # f = open(op.join(output_dir, 'simulation.pck'), 'w')
        # cPickle.dump(simulation, f)
        # f.close()

    return simulation



allModels = {
    'WNSGGMS' : { 'class' : BOLDGibbsSampler,
                  'doc' : 'iid White Noise, BiGaussian Spatial Mixture NRL,'\
                  ' stationary temporal model on NRL, marginalized drifts'
                  },
    'WNSGGMSD' : { 'class' : Drift_BOLDGibbsSampler,
                  'doc' : 'iid White Noise, BiGaussian Spatial Mixture NRL,'\
                  ' stationary temporal model on NRL, explicit drifts'
                  },

    'WNSGGGMS' : { 'class' : GGG_BOLDGibbsSampler,
                   'doc' : 'iid White Noise, TriGaussian Spatial Mixture NRL, ' \
                   'stationary temporal model on NRL'
                   },
    'WNSGGMH' : { 'class' : HAB_BOLDGibbsSampler,
                'doc' : 'iid White Noise, BiGaussian Spatial Mixture NRL, ' \
                  'temporal model of habituation on NRL'
                  },
    'ANSGGMS' : { 'class' : ARN_BOLDGibbsSampler,
                'doc' : 'AR(1) Noise, BiGaussian Spatial Mixture NRL, ' \
                  'stationnary temporal model on NRL'
                  },
    'WNSGGMSW' : { 'class' : W_BOLDGibbsSampler,
                  'doc' : 'iid White Noise, BiGaussian Spatial Mixture NRL,'\
                  ' stationary temporal model on NRL, marginalized drifts'\
                  ' and condition relevance variable'
                  },
    'WNSGGMSWD' : { 'class' : W_Drift_BOLDGibbsSampler,
                  'doc' : 'iid White Noise, BiGaussian Spatial Mixture NRL,'\
                  ' stationary temporal model on NRL, explicit drifts'\
                  ' and condition relevance variable'
                  },
    #    'WNSGGMS_BF' : { 'class' : BayesFactorBOLDGibbsSampler,
#                    'doc' : 'Same as WNSGGMS with computation of bayes factor'
#                     },
    }

defaultModel = 'WNSGGMS'
if pyhrf.__usemode__ == pyhrf.DEVEL:
    availableModels = allModels
elif pyhrf.__usemode__ == pyhrf.ENDUSER:
    availableModels = {'WNSGGMS' : allModels['WNSGGMS']}







