# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
#from numpy import *
import numpy as np
import scipy.linalg
import scipy.interpolate
import copy as copyModule # avoids conflict with copy function from numpy

from numpy.linalg import cholesky, det, inv, eigh
import pyhrf
from pyhrf import xmlio

from samplerbase import *
#import intensivecalc
#from pyhrf import boldsynth
from pyhrf.boldsynth.hrf import genGaussianSmoothHRF, buildFiniteDiffMatrix
from pyhrf.boldsynth.hrf import genCanoBezierHRF, getCanoHRF

from pyhrf.ndarray import xndarray

try:
    from pyhrf.stats import cRandom
except ImportError ,e :
    cRandom = None
    if 0 and pyhrf.__usemode__ == pyhrf.DEVEL:
        print 'cRandom could not be imported ... check installation of unuran'\
            ' and prng and recompile pyhrf'

from nrl import BiGaussMixtureParamsSampler

def msqrt(cov):
    """
    sig = msqrt(cov)

    Return a matrix square root of a covariance matrix. Tries Cholesky
    factorization first, and factorizes by diagonalization if that fails.
    """
    # Try Cholesky factorization
    if 0 :
        try:
            sig = asmatrix(cholesky(cov))
            # If there's a small eigenvalue, diagonalize
        except LinAlgError:
            val, vec = eigh(cov)
            sig = np.zeros(vec.shape)
            for i in range(len(val)):
                if val[i]<0.:
                    val[i]=0.
                sig[:,i] = vec[:,i]*sqrt(val[i])
        return asmatrix(sig).T
    else:
        val, vec = eigh(cov)
        sig = np.zeros(vec.shape)
        for i in xrange(len(val)):
            if val[i]<0.:
                val[i]=0.
            sig[:,i] = vec[:,i]*sqrt(val[i])
        return asmatrix(sig).T

def buildDiagGaussianMat(size,width):
    a = scipy.stats.distributions.norm.pdf(range(size),size/2,width)
    b = a[len(a)/2:]
    print b.shape
    print np.zeros(size-len(b)).shape
    return toeplitz(np.concatenate((b, np.zeros(size-len(b)))))



def sampleHRF_voxelwise_iid( stLambdaS, stLambdaY, varR, rh, nbColX, nbVox):

    pyhrf.verbose(4,'stLambdaS:')
    pyhrf.verbose.printNdarray(4,stLambdaS)
    pyhrf.verbose(4,'varR:')
    pyhrf.verbose.printNdarray(4,varR)
    pyhrf.verbose(4,'rh: %f' %rh)
    pyhrf.verbose(4,'varR/rh:')
    pyhrf.verbose.printNdarray(4,varR/rh)

    varInvSigma_h = stLambdaS + nbVox * varR/rh

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



##def sampleHRF_optim( stDeltaS, stDeltaY, varR, rh, nbColX):

##    varInvSigma_h = stDeltaS + varR/rh
##    mean_h = np.linalg.solve(varInvSigma_h, stDeltaY )
##    choleskyInvSigma_h = cholesky(varInvSigma_h).transpose()
##    hrf = np.linalg.solve(choleskyInvSigma_h, np.random.randn(nbColX)) + mean_h
##    return hrf


class HRFSampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    #TODO : HRF sampler for BiGaussian NLR mixture
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
                 compute_ah_online=False, output_ah=False):
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
        
        self.Ini = val_ini
        self.duration = duration
        self.zc = zero_constraint
        self.normalise = normalise
        #print 'normalise', self.normalise
        self.derivOrder = deriv_order
        self.varR = None
        self.covarHack = covar_hack
        self.priorType = prior_type
        self.output_ah = output_ah
        self.signErrorDetected = None
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

        if dataInput.simulData is not None and \
            isinstance(dataInput.simulData[0], dict):
            simu_hrf = dataInput.simulData[0]['hrf'].mean(1)
            if isinstance(simu_hrf, xndarray):
                self.trueValue = simu_hrf.data
            else:
                self.trueValue = simu_hrf


        # Allocations :
        self.ajak_rb =  np.zeros((self.nbVox), dtype=float)

        if 0:
            self.varStLambdaS = np.zeros((self.nbColX, self.nbColX, self.nbVox),
                                      dtype=float)
            self.varStLambdaY = np.zeros((self.nbColX, self.nbVox), dtype=float)
        self.varYaj = np.zeros((self.ny,self.nbVox), dtype=float)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5
	#print 'HRF Norm =', self.norm


    def checkAndSetInitValue(self, variables):
        smplRH = self.samplerEngine.get_variable('hrf_var')
        smplRH.checkAndSetInitValue(variables)
        rh = smplRH.currentValue
        pyhrf.verbose(4, 'Hrf variance is :%1.3f' %rh)

        if self.useTrueValue :
            if self.trueValue is not None:
                hrfValIni = self.trueValue.copy()
            else:
                raise Exception('Needed a true value for hrf init but '\
                                    'None defined')

        if self.Ini is None:
            hrfValIni=None
        else:
            hrfValIni=self.Ini

        if hrfValIni is None:
            pyhrf.verbose(4, 'hrfValIni is None -> setting it ...')
            pyhrf.verbose(6, 'self.duration=%d, self.eventdt=%1.2f' \
                              %(self.duration,self.eventdt))
            tAxis = np.arange(0, self.duration+self.eventdt,
                               self.eventdt)

            #hIni = genCanoBezierHRF(self.duration, self.eventdt)[1]
            pyhrf.verbose(5,'genCanoHRF -> dur=%f, dt=%f' \
                              %(self.duration, self.eventdt))
            dt = self.eventdt
            hIni = getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

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

        if self.normalise == 1.:
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

        #else -> #TODO : check consistency between given init value
        # and self.hrfLength ...

        #Update Ytilde



    def getCurrentVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_var')
        rh = smplRH.currentValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh

    def getFinalVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_var')
        rh = smplRH.finalValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh



    def samplingWarmUp(self, variables):
        if self.varR == None :
            smplRH = self.get_variable('hrf_var')
            rh = smplRH.currentValue
            (useless, self.varR) = genGaussianSmoothHRF(self.zc,
                                                        self.hrfLength,
                                                        self.eventdt, rh, order=self.derivOrder)
            #self.varR = buildDiagGaussianMat(self.hrfLength-self.zc*2,4)
            # HACK
            #self.varR = ones_like(self.varR)

    def computeStDS_StDY(self, rb, nrls, aa):

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

    def sampleNextAlt(self, variables):
        self.reportCurrentVal()

    def sampleNextInternal(self, variables):

        #TODO : comment

        try:
            snrl = self.samplerEngine.get_variable('nrl_by_session')
        except KeyError:
            snrl = self.samplerEngine.get_variable('nrl')

        nrls = snrl.currentValue
        rb   = self.samplerEngine.get_variable('noise_var').currentValue

        if 1:
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
            tSQSOptimIni = time.time()
            (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY(rb, nrls,
                                                                     snrl.aa)
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        if 0:
            nnIdx = self.dataInput.notNullIdxStack
            pyhrf.verbose(6, 'Computing StQS StQY C fashion')
            tSQSOptimIni = time.time()
            import intensivecalc # conditionnal import of C module
                                 # won't work with Pyro Mobile code
            intensivecalc.computeStLambdaSparse(nrls,
                                                self.dataInput.stackX,
                                                nnIdx,
                                                self.dataInput.delta,
                                                self.dataInput.varMBY,
                                                self.varStLambdaS,
                                                self.varStLambdaY, rb);
            pyhrf.verbose(6, 'Computing StQS StQY C fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

            assert np.allclose(self.varDeltaS, self.varStLambdaS.sum(2))
            assert np.allclose(self.varDeltaY, self.varStLambdaY.sum(1))

            self.varDeltaS = self.varStLambdaS.sum(2)
            self.varDeltaY = self.varStLambdaY.sum(1)

        if 0 :
            pyhrf.verbose(6, 'deltaS :' + \
                          str(self.varDeltaS.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaS)
            pyhrf.verbose(6, 'sum StLambdaS :')
            pyhrf.verbose.printNdarray(6, self.varStLambdaS.sum(2))

            pyhrf.verbose(6, 'deltaY :' + str(self.varDeltaY.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaY)
            pyhrf.verbose(6, 'StLambdaY :' )
            pyhrf.verbose.printNdarray(6, self.varStLambdaY.sum(1))
##            iDiffs = (self.varDeltaS != self.varStLambdaS.sum(2))
##            print 'self.varDeltaS[iDiffs] : ',self.varDeltaS[iDiffs].shape
##            print self.varDeltaS[iDiffs]
##            print 'self.varStLambdaS.sum(2)[iDiffs] :', \
##                  self.varStLambdaS.sum(2)[iDiffs].shape
##            print self.varStLambdaS.sum(2)[iDiffs]

        rh = self.get_variable('hrf_var').currentValue


        if self.priorType == 'voxelwiseIID':
            h = sampleHRF_voxelwise_iid(self.varDeltaS, self.varDeltaY,
                                        self.varR,
                                        rh, self.nbColX, self.nbVox)
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

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, '\
                          'Nrls and mixture parameters at each iteration ...')
            f = self.norm
            #HACK
            #f = self.currentValue.max()
            self.currentValue = self.currentValue / f #/(self.normalise+0.)
            if 1:
                try:
                    if self.samplerEngine.get_variable('nrl_by_session').sampleFlag:
                        self.samplerEngine.get_variable('nrl_by_session').currentValue *= f
                except KeyError:
                    if self.samplerEngine.get_variable('nrl').sampleFlag:
                        self.samplerEngine.get_variable('nrl').currentValue *= f

                    ## Normalizing Mixture components
                    #smixt_params = self.samplerEngine.get_variable('mixt_params')
                    #if 0 and smixt_params.sampleFlag:
                        #smixt_params.currentValue[smixt_params.I_MEAN_CA] *= f # Normalizing Mean's activation class
                        #smixt_params.currentValue[smixt_params.I_VAR_CI] *= f**2 # Normalizing Variance's activation class
                        #smixt_params.currentValue[smixt_params.I_VAR_CA] *= f**2 # Normalizing Variance's in-activation class

            self.updateNorm()

        self.updateXh()
        self.reportCurrentVal()

        # update ytilde for nrls
        try:
            self.samplerEngine.get_variable('nrl_by_session')
        except KeyError: # only if not in multisession case, else ytilde update
                         # is handled by HRF_MultiSess_Sampler.sampleNextAlt
            nrlsmpl = self.samplerEngine.get_variable('nrl')
            nrlsmpl.computeVarYTildeOpt(self.varXh)


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
        #print 'GLOB:', self.dataInput.stackX.shape,  hrf.shape
        stackXh = np.dot(self.dataInput.stackX, hrf)
        return np.reshape(stackXh, (self.nbConditions,self.ny)).transpose()

    def updateXh(self):
        self.varXh = self.calcXh(self.currentValue)

    def detectSignError(self):
        #return False
        pyhrf.verbose(5,'detectSignError...')
        if self.signErrorDetected is not None:
            return self.signErrorDetected
        else:
            pyhrf.verbose(5, 'HRF PM:')
            pyhrf.verbose.printNdarray(3, self.mean)
            pyhrf.verbose(5, 'NRLs PM:')
            pyhrf.verbose.printNdarray(3, self.samplerEngine.get_variable('nrl').mean)
            pyhrf.verbose(5, 'nb negs: %d' %(self.mean <= 0.).sum(dtype=float))
            if (self.mean <= 0.).sum(dtype=float)/len(self.mean) > .9 or \
                    self.mean[np.argmax(abs(self.mean))] < -.2:
                self.signErrorDetected = True
            else:
                #hRef = np.array(genCanoBezierHRF(self.duration, self.eventdt)[1],
                #                   dtype=float)
                #hRef /= ((hRef**2.).sum())**.5
                #h = self.finalValue/((self.finalValue**2.).sum())**.5

                #if (hRef*h).sum() < 0: # if d(hRef,hPM) > d(hRef,-hPM) => sign error

                #np.abs(self.mean.min()) > self.mean.max()

                self.signErrorDetected = False

        return self.signErrorDetected


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

        # print '~~~~~~~~~~~~~~~~~~~~~~~'
        # print 'self.finalValue.shape:', self.finalValue.shape
        # print 'self.trueValue.shape:', self.trueValue.shape

        # Correct sign ambiguity error :
        sign_error = self.detectSignError()
        self.finalValue_sign_corr = self.finalValue * (1 - 2 * sign_error) #sign_error at 0 or 1
        if sign_error:
            pyhrf.verbose(1,'Warning : sign error on HRF')

        pyhrf.verbose(4, 'HRF finalValue :\n')
        pyhrf.verbose.printNdarray(4, self.finalValue)



    def get_final_value(self):
        """ Used to compare with simulated value """
        scale_f = 1.

        hestim = self.finalValue
        htrue = self.trueValue

        if htrue is not None:
            scale_f = ((htrue**2).sum() / (hestim**2).sum())**.5

        return self.finalValue * scale_f

    def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
        crit_norm = np.array([((fv - tv)**2).sum() / (tv**2).sum()**.5 < 0.025 ])
        return ['crit_norm'], crit_norm



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
        sScale = self.samplerEngine.get_variable('scale')
        if self.sampleFlag and np.allclose(self.normalise,0.) and \
                not sScale.sampleFlag:
            pyhrf.verbose(6, 'Normalizing posterior mean of HRF each iteration ...')
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


    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag: #TODO chech for NaNs ...  and not _np.isnan(rh):
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue

            rb = self.samplerEngine.get_variable('noise_var').finalValue
            snrls = self.samplerEngine.get_variable('nrl')
            nrls = snrls.finalValue
            aa = np.zeros((self.nbConditions, self.nbConditions, self.nbVox),
                       dtype=float)
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

        #if self.voxelwise_outputs or pyhrf.__usemode__ == pyhrf.ENDUSER:
        #to obtain hrf.nii ie hrf on each voxel
        if 0:
            #TODO: fix to work with generator ?
            # -> ah output might be enough to browse results voxelwise ...
            for on,o in outputs.iteritems():
                pyhrf.verbose(4, "Treating hrf output %s :\n%s" %(on,o.descrip()))
                newOutput = o.repeat(self.nbVox, 'voxel')
                pyhrf.verbose(4, "New hrf output :\n%s" %(newOutput.descrip()))
                outputs[on] = newOutput


        if self.output_ah:
            h = self.finalValue
            nrls = self.samplerEngine.get_variable('nrl').finalValue
            ah = np.zeros((h.shape[0],self.nbVox, self.nbConditions))
            for j in xrange(self.nbConditions):
                ah[:,:,j] = np.repeat(h,self.nbVox).reshape(h.shape[0],
                                                            self.nbVox) * \
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

        outputs['Xh'] = xndarray(dm, axes_names=['time','condition'],
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

        for k,v in outputs.iteritems():
            yield k,v



class HRF_Drift_Sampler(HRFSampler):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling.
    """

    def computeStDS_StDY(self, rb, nrls, aa):

        varX = self.dataInput.varX
        matXtX = self.dataInput.matXtX
        matPl = self.samplerEngine.get_variable('drift').matPl
        y = self.dataInput.varMBY - matPl

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )


        # aX = np.zeros((self.nbConditions,self.ny,self.nbColX,self.nbVox),
        # dtype=float )
        # for j in xrange(self.nbConditions):
        #     aX[j,:,:,:] = (nrls[j,:] * varX[j,:,:,newaxis])
        # sumaX = aX.sum(0)

        for j in xrange(self.nbConditions):
            np.divide(y, rb, self.varYaj)
            self.varYaj *= nrls[j,:]
            varDeltaY +=  np.dot(varX[j,:,:].transpose(),self.varYaj.sum(1))

            for k in xrange(self.nbConditions):
                    np.divide(aa[j,k,:], rb, self.ajak_rb)
##                    pyhrf.verbose(6, 'ajak/rb :')
##                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                    varDeltaS += self.ajak_rb.sum()*matXtX[j,k,:,:]

        return (varDeltaS, varDeltaY)


############################################################################
# 		HRF sampling for dealing with AR noise
############################################################################

class HRFARSampler(HRFSampler) :
    """
    #THis class implements the sampling of the HRF
    when modelling a serially AR(1) noise process in the
    data. The structure of this noise is spatially varying in the
    sense that there is one AR parameter in combination with one
    noise variance per voxel.
    """


    def linkToData(self, dataInput):

        HRFSampler.linkToData(self, dataInput)
        self.varStLambdaS = np.zeros((self.nbColX, self.nbColX, self.nbVox),
                                  dtype=float)
        self.varStLambdaY = np.zeros((self.nbColX, self.nbVox), dtype=float)


    def computeStDS_StDY(self, reps, noiseInvCov, nrls, varMBYPl):
        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )
        for v in  range(self.nbVox):
            S = np.zeros( (self.ny,self.nbColX), dtype=float )
            for m in range(self.nbConditions):
                VectIdx = m*self.ny + np.arange(self.ny)
                S += nrls[m,v]*self.dataInput.stackX[VectIdx,:]
                StLambda = np.dot( S.transpose(), noiseInvCov[:,:,v] )
                StLambda = np.multiply( StLambda, 1./reps[v] )
                self.varStLambdaS[:,:,v] = np.dot( StLambda, S )
                self.varStLambdaY[:,v] = np.dot( StLambda, varMBYPl[:,v])
        varDeltaS = self.varStLambdaS.sum(2)
        varDeltaY = self.varStLambdaY.sum(1)
        return (varDeltaS, varDeltaY)


    def sampleNextInternal(self, variables):
        #TODO : comment
        ##print '- Sampling HRF ...'
        nrls = self.get_variable('nrl').currentValue
        smpldrift =  self.get_variable('drift')
        varMBYPl = smpldrift.varMBYPl
        varReps = self.get_variable('noise_var').currentValue
        smplnoise = self.get_variable('noiseARParam')
        noiseARp= smplnoise.currentValue
        noiseInvCov= smplnoise.InvAutoCorrNoise

##        matXQ = self.dataInput.matXQ
        if 0:
            print "taille variances bruit", varReps.shape
            print "taille donnees", varMBYPl.shape
            print "taille cov bruit", noiseInvCov.shape

        if 0:
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
            tSQSOptimIni = time.time()
#            self.varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
            self.varXtLambdaY = np.zeros((self.nbColX), dtype=float )
            self.varLambdaY = np.zeros((self.ny,self.nbVox), dtype=float)
            self.varLambdaX = np.zeros((self.ny,self.nbColX), dtype=float)
            self.varXtLambdaX = np.zeros((self.nbColX,self.nbColX), dtype=float)
            np.divide(varMBYPl, varReps, self.varYaj)
            for v in xrange(self.nbVox):
                self.varLambdaY[:,v] = np.dot(noiseInvCov[:,:,v], self.varYaj[:,v])
            tempY = np.zeros((self.ny,self.nbVox), dtype=float)
            for j in xrange(self.nbConditions):
                np.multiply( self.varLambdaY, nrls[j,:], tempY)
                self.varXtLambdaY +=  np.dot(self.dataInput.varX[j,:,:].transpose(),tempY.sum(axis=1))

                for k in xrange(self.nbConditions):
                    np.multiply(nrls[j,:], nrls[k,:], self.ajak_rb)
                    self.ajak_rb /= varReps
                    tempX = np.zeros((self.ny,self.ny), dtype=float)
                    for v in xrange(self.nbVox):
                        tempX += self.ajak_rb[v] * noiseInvCov[:,:,v]
                    self.varLambdaX += np.dot(tempX, self.dataInput.varX[k,:,:])
                    pyhrf.verbose(6, 'ajak :')
                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                self.varXtLambdaX += np.dot(self.dataInput.varX[j,:,:].transpose(), self.varLambdaX)
                    #self.varDeltaS +=sum(np.divide(np.multiply( nrls[j,:], nrls[k,:]),
                    #                            rb ) ) * matXQX[j,k,:,:]
            pyhrf.verbose(6, 'Computing StLambdaS StLambdaY optim fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

#        for v in range(self.nbVox):
##            DatamDrift[:,v] = self.dataInput.varMBY[:,v] - drift.varPl[:,v]

        if 1:
            nnIdx = self.dataInput.notNullIdxStack
            pyhrf.verbose(6, 'Computing StLambdaS StLambdaY: C fashion')
            tSQSOptimIni = time.time()
            self.varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
            self.varDeltaY = np.zeros((self.nbColX), dtype=float )
            import intensivecalc
            intensivecalc.computeStLambdaARModelSparse(nrls, self.dataInput.stackX,
                                                    nnIdx , noiseInvCov, varMBYPl,
                                                    self.varStLambdaS,
                                                    self.varStLambdaY,
                                                    varReps);
            pyhrf.verbose(6, 'Computing StLambdaS StLambdaY: C '+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

#            assert np.allclose(self.varDeltaS, self.varStLambdaS.sum(2))
#            assert np.allclose(self.varDeltaY, self.varStLambdaY.sum(1))
            self.varDeltaS = self.varStLambdaS.sum(2)
            self.varDeltaY = self.varStLambdaY.sum(1)
                #intensivecalc.computeStLambdaARModel( nrls, self.dataInput.stackX,
                                                    #noiseInvCov, varMBYPl,
                                                    #self.varStLambdaS, self.varStLambdaY,
                                                    #reps)
        if 0:
            pyhrf.verbose(6, 'Computing StLambdaS StLambdaY: Numpy fashion')
            tSQSOptimIni = time.time()
            (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY(varReps, noiseInvCov, nrls, varMBYPl)
            pyhrf.verbose(6, 'Computing StLambdaS StLambdaY: NumPy '+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        if 0 :
            pyhrf.verbose(6, 'deltaS :' + \
                        str(self.varDeltaS.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaS)
            pyhrf.verbose(6, 'sum StLambdaS :')
            pyhrf.verbose.printNdarray(6,
                                    self.varStLambdaS.sum(2))

            pyhrf.verbose(6, 'deltaY :' + \
                        str(self.varDeltaY.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaY)
            pyhrf.verbose(6, 'StLambdaY :' )
            pyhrf.verbose.printNdarray(6,
                                    self.varStLambdaY.sum(1))
            iDiffs = (self.varDeltaS != self.varStLambdaS.sum(2))
##            print 'self.varDeltaS[iDiffs] : ',self.varDeltaS[iDiffs].shape
##            print self.varDeltaS[iDiffs]
##            print 'self.varStLambdaS.sum(2)[iDiffs] :', \
##                  self.varStLambdaS.sum(2)[iDiffs].shape
##            print self.varStLambdaS.sum(2)[iDiffs]

        rh = self.get_variable('hrf_var').currentValue

#        self.currentValue = sampleHRF(self.varXtLambdaX, self.varXtLambdaY,
#                                      self.varR, rh, self.nbColX)
        self.currentValue = sampleHRF_single_hrf(self.varDeltaS,
                                      self.varDeltaY,
                                      self.varR, rh, self.nbColX, self.nbVox)

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        if self.normalise:
          self.currentValue /= self.norm
          self.updateNorm()

        self.updateXh()
        maxHRF = self.currentValue.max()
        tMaxHRF = np.where(self.currentValue==maxHRF)[0]*self.dt
        pyhrf.verbose(4, 'sampled HRF mean = %1.3f(%1.3f)'
                      %(self.currentValue.mean(),self.currentValue.std()))
        pyhrf.verbose(4, 'sampled HRF max = ' + \
                          '(tMax=%1.3f,vMax=%1.3f)' %(tMaxHRF, maxHRF))

#    def updateXh(self):
#        stackXh = np.dot(self.dataInput.stackX, self.currentValue)
#        self.varXh = np.reshape(stackXh,( self.nbConditions,self.ny)).transpose()


    def finalizeSampling(self):
        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.errors = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag:
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue
            snrls = self.samplerEngine.get_variable('nrl')
            nrls = snrls.finalValue
            sreps = self.samplerEngine.get_variable('noise_var')
            varReps = sreps.currentValue
            sARp = self.samplerEngine.get_variable('noiseARParam')
            noiseInvCov= sARp.InvAutoCorrNoise
            sdrift = self.samplerEngine.get_variable('drift')
            varMBYPl = sdrift.varMBYPl
            stDS, useless = self.computeStDS_StDY(varReps,
                                                noiseInvCov, nrls, varMBYPl)
            varSigma_h = np.asmatrix(stDS + self.varR/rh).I
            if self.zc:
                self.errors[1:-1] = np.diag(varSigma_h)**.5
            else:
                self.errors = np.diag(varSigma_h)**.5
            del self.varDeltaY
            del self.varDeltaS

        ## Correct hrf*nrl scale ambiguity :
        del self.varStLambdaS
        del self.varStLambdaY
        del self.varXh
        del self.varR


############################################################################
# 		HRF sampling accounting for habituation phenomena
#       Need to update X onset matrices depending on habituation speeds
############################################################################

class HRFwithHabSampler(HRFSampler) :


    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt

        self.varYrb = np.zeros((self.ny,self.nbVox), dtype=float)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5


    def computeStDS_StDY(self, rb, sumaX, Q):

        # c'est la partie qui prend enormement de temps (car ny est grand ainsi que nbColX)

        #matXQ = self.dataInput.matXQ
        #matXQX = self.dataInput.matXQX
        y = self.dataInput.varMBY
        #Q = self.dataInput.delta

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )
        np.divide(y, rb, self.varYrb)


        for nv in range(self.nbVox) :
            #print shape(sumaX[:,:,nv]), shape(Q)
            temp = np.dot(sumaX[:,:,nv].transpose(),Q)
            varDeltaY +=  np.dot(temp, self.varYrb[:,nv])
            varDeltaS +=  np.dot(temp, sumaX[:,:,nv]) / rb[nv]

        return (varDeltaS, varDeltaY)

    def sampleNextInternal(self, variables):
        #TODO : comment

        snrl = self.get_variable('nrl')
        nrls = self.get_variable('nrl').currentValue
        rb = self.get_variable('noise_var').currentValue

        sumaX = snrl.sumaX
        #sumaXQ = snrl.sumaXQ


        matXQX = self.dataInput.matXQX
        matXQ = self.dataInput.matXQ
        pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
        tSQSOptimIni = time.time()
        (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY(rb, sumaX,
                                                                 self.dataInput.delta)
        pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                      ' done in %1.3f sec' %(time.time()-tSQSOptimIni))



        rh = self.get_variable('hrf_var').currentValue


        self.currentValue = sampleHRF(self.varDeltaS, self.varDeltaY,
                                      self.varR, rh, self.nbColX)

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        if self.normalise:
            self.currentValue /= self.norm /(self.normalise+0.)
            #if self.detectSignError():
            #print '2'
            #self.currentValue *= -1
            self.updateNorm()


        self.varXh = self.get_variable('nrl').varXh

        maxHRF = self.currentValue.max()
        tMaxHRF = np.where(self.currentValue==maxHRF)[0]*self.dt

    def getScaleFactor(self):
        if self.finalValue is None:
            self.setFinalValue()
        # Use amplitude :
        #scaleF = self.finalValue.max()-self.finalValue.min()
        # Use norm :
        scaleF = ((self.finalValue**2.0).sum())**0.5
        # Use max :
        #scaleF = self.finalValue.max()
        return scaleF

    def finalizeSampling(self):

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag:
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue
            rb = self.samplerEngine.get_variable('noise_var').finalValue
            snrls = self.samplerEngine.get_variable('nrl')
            nrls = snrls.finalValue
            #aa = np.zeros((self.nbConditions, self.nbConditions, self.nbVox),
                       #dtype=float)
            #snrls.computeAA(nrls, aa)
            if 0 and hasattr(snrls,'sumaX'):
                stDS, useless = self.computeStDS_StDY(rb, snrls.sumaX, self.dataInput.delta )
                varSigma_h = np.asmatrix(stDS + self.varR/rh).I
                if self.zc:
                    self.error[1:-1] = np.diag(varSigma_h)**.5
                else:
                    self.error = np.diag(varSigma_h)**.5
            del self.varDeltaY
            del self.varDeltaS


        # Temporary variables :
        del self.varYrb
        #del self.ajak_rb

        # may be needed afterwards :
        del self.varXh
        del self.varR


############################################################################
#
#    HRF Sampling with relevance variable
#
############################################################################

class HRFSamplerWithRelVar(HRFSampler) :

    """
    This class introduce a new variable w (Relevant Variable)
    w take its value in {0, 1} with :
            w = 1  condition m is relevant in the studied parcel
            w = 1  otherwise
    """


    def linkToData(self, dataInput):

        HRFSampler.linkToData(self, dataInput)

    def computeStDS_StDY_WithRelVar(self, rb, nrls, aa, w):

        matXQ = self.dataInput.matXQ
        matXQX = self.dataInput.matXQX
        y = self.dataInput.varMBY

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )

        for j in xrange(self.nbConditions):
            np.divide(y, rb, self.varYaj)
            self.varYaj *= nrls[j,:]
            varDeltaY +=  np.dot(matXQ[j,:,:],self.varYaj.sum(1)*w[j])

            for k in xrange(self.nbConditions):
                np.divide(aa[j,k,:], rb, self.ajak_rb)
##                    pyhrf.verbose(6, 'ajak/rb :')
##                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                varDeltaS += self.ajak_rb.sum()*w[j]*w[k]*matXQX[j,k,:,:]

        return (varDeltaS, varDeltaY)

    def sampleNextInternal(self, variables):

        #print 'Step 2 : HRF Sampling *****RelVar*****'


        #TODO : comment

        snrl = self.get_variable('nrl')
        #print '         varYbar begin =',snrl.varYbar.sum()
        #print '         varYtilde begin =',snrl.varYtilde.sum()
        nrls = snrl.currentValue
        rb = self.get_variable('noise_var').currentValue
        w = self.get_variable('W').currentValue

        if 1:
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
            tSQSOptimIni = time.time()
            (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY_WithRelVar(rb, nrls,
                                                                     snrl.aa, w)

##            for j in xrange(self.nbConditions):
##                divide(self.dataInput.varMBY, rb, self.varYaj)
##                self.varYaj *= nrls[j,:]
##                self.varDeltaY +=  dot(matXQ[j,:,:],self.varYaj.sum(1))

##                for k in xrange(self.nbConditions):
##                    numpy.divide(snrl.aa[j,k,:], rb, self.ajak_rb)
##                    self.varDeltaS += self.ajak_rb.sum()*matXQX[j,k,:,:]


            pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        if 0:
            nnIdx = self.dataInput.notNullIdxStack
            pyhrf.verbose(6, 'Computing StQS StQY C fashion')
            tSQSOptimIni = time.time()
            import intensivecalc # conditionnal import of C module
                                 # won't work with Pyro Mobile code
            intensivecalc.computeStLambdaSparse(nrls,
                                                self.dataInput.stackX,
                                                nnIdx,
                                                self.dataInput.delta,
                                                self.dataInput.varMBY,
                                                self.varStLambdaS,
                                                self.varStLambdaY, rb);
            pyhrf.verbose(6, 'Computing StQS StQY C fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

            assert numpy.allclose(self.varDeltaS, self.varStLambdaS.sum(2))
            assert numpy.allclose(self.varDeltaY, self.varStLambdaY.sum(1))

            self.varDeltaS = self.varStLambdaS.sum(2)
            self.varDeltaY = self.varStLambdaY.sum(1)

        if 0 :
            pyhrf.verbose(6, 'deltaS :' + \
                          str(self.varDeltaS.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaS)
            pyhrf.verbose(6, 'sum StLambdaS :')
            pyhrf.verbose.printNdarray(6, self.varStLambdaS.sum(2))

            pyhrf.verbose(6, 'deltaY :' + str(self.varDeltaY.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaY)
            pyhrf.verbose(6, 'StLambdaY :' )
            pyhrf.verbose.printNdarray(6, self.varStLambdaY.sum(1))
            iDiffs = (self.varDeltaS != self.varStLambdaS.sum(2))

        rh = self.get_variable('hrf_var').currentValue


        if self.priorType == 'voxelwiseIID':
            h = sampleHRF_voxelwise_iid(self.varDeltaS, self.varDeltaY,
                                        self.varR,
                                        rh, self.nbColX, self.nbVox)
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

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, Nrls and mixture parameters at each iteration ...')
            f = self.norm
            #HACK
            #f = self.currentValue.max()
            self.currentValue = self.currentValue / f #/(self.normalise+0.)
            if self.samplerEngine.get_variable('nrl').sampleFlag:
                self.samplerEngine.get_variable('nrl').currentValue *= f

            ## Normalizing Mixture components
            #smixt_params = self.samplerEngine.get_variable('mixt_params')
            #if 0 and smixt_params.sampleFlag:
                #smixt_params.currentValue[smixt_params.I_MEAN_CA] *= f # Normalizing Mean's activation class
                #smixt_params.currentValue[smixt_params.I_VAR_CI] *= f**2 # Normalizing Variance's activation class
                #smixt_params.currentValue[smixt_params.I_VAR_CA] *= f**2 # Normalizing Variance's in-activation class

            self.updateNorm()

        self.updateXh()
        self.reportCurrentVal()

        snrl.computeVarYTildeOptWithRelVar(self.varXh, w)
        #print '         varYbar end =',snrl.varYbar.sum()
        #print '         varYtilde end =',snrl.varYtilde.sum()

    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag: #TODO chech for NaNs ...  and not _np.isnan(rh):
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue
            w = self.samplerEngine.get_variable('W').finalValue
            rb = self.samplerEngine.get_variable('noise_var').finalValue
            snrls = self.samplerEngine.get_variable('nrl')
            nrls = snrls.finalValue
            aa = np.zeros((self.nbConditions, self.nbConditions, self.nbVox),
                       dtype=float)
            snrls.computeAA(nrls, aa)
            stDS, useless = self.computeStDS_StDY_WithRelVar(rb, nrls, aa, w)
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

class HRF_Drift_SamplerWithRelVar(HRFSamplerWithRelVar):
    """
    Class handling the Gibbs sampling of Neural Response Levels in the case of
    joint drift sampling.
    """

    def computeStDS_StDY_WithRelVar(self, rb, nrls, aa, w):

        #print '         New Function for drift estimaton in step 3 ...'

        varX = self.dataInput.varX
        matXtX = self.dataInput.matXtX
        matPl = self.samplerEngine.get_variable('drift').matPl
        y = self.dataInput.varMBY - matPl

        varDeltaS = np.zeros((self.nbColX,self.nbColX), dtype=float )
        varDeltaY = np.zeros((self.nbColX), dtype=float )


        # aX = np.zeros((self.nbConditions,self.ny,self.nbColX,self.nbVox),
        # dtype=float )
        # for j in xrange(self.nbConditions):
        #     aX[j,:,:,:] = (nrls[j,:] * varX[j,:,:,newaxis])
        # sumaX = aX.sum(0)

        for j in xrange(self.nbConditions):
            np.divide(y, rb, self.varYaj)
            self.varYaj *= nrls[j,:]
            varDeltaY +=  np.dot(varX[j,:,:].transpose(),self.varYaj.sum(1)*w[j])

            for k in xrange(self.nbConditions):
                    np.divide(aa[j,k,:], rb, self.ajak_rb)
##                    pyhrf.verbose(6, 'ajak/rb :')
##                    pyhrf.verbose.printNdarray(6,self.ajak_rb)
                    varDeltaS += self.ajak_rb.sum()*w[j]*w[k]*matXtX[j,k,:,:]

        return (varDeltaS, varDeltaY)


####################################################
###### estimation in two parts of the HRF ##########
####################################################
class HRF_two_parts_Sampler(HRFSampler) :

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt

        if dataInput.simulData is not None and \
            isinstance(dataInput.simulData, dict):
            simu_hrf = dataInput.simulData['hrf']
            if isinstance(simu_hrf, xndarray):
                self.trueValue = simu_hrf.data
            else:
                self.trueValue = simu_hrf

        # Allocations :
        self.ajak_rb =  np.zeros((self.nbVox), dtype=float)

        if 0:
            self.varStLambdaS = np.zeros((self.nbColX, self.nbColX, self.nbVox),
                                      dtype=float)
            self.varStLambdaY = np.zeros((self.nbColX, self.nbVox), dtype=float)
        self.varYaj = np.zeros((self.ny,self.nbVox), dtype=float)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5
    #print 'HRF Norm =', self.norm


    def checkAndSetInitValue(self, variables):
        smplRH = self.samplerEngine.get_variable('hrf_var')
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
            tAxis = np.arange(0, self.duration+self.eventdt,
                               self.eventdt)

            #hIni = genCanoBezierHRF(self.duration, self.eventdt)[1]
            pyhrf.verbose(5,'genCanoHRF -> dur=%f, dt=%f' \
                              %(self.duration, self.eventdt))
            dt = self.eventdt
            hIni = getCanoHRF(self.hrfLength * dt, dt)[1][:self.hrfLength]

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
        #else -> #TODO : check consistency between given init value
        # and self.hrfLength ...

        #Update Ytilde



    def getCurrentVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_var')
        rh = smplRH.currentValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh

    def getFinalVar(self):
        smplRH = self.samplerEngine.get_variable('hrf_var')
        rh = smplRH.finalValue
        (useless, varR) = genGaussianSmoothHRF(self.zc,
                                               self.hrfLength,
                                               self.eventdt, rh)
        return varR/rh



    def samplingWarmUp(self, variables):
        if self.varR == None :
            smplRH = self.get_variable('hrf_var')
            rh = smplRH.currentValue
            (useless, self.varR) = genGaussianSmoothHRF(self.zc,
                                                        self.hrfLength,
                                                        self.eventdt, rh, order=self.derivOrder)
            #self.varR = buildDiagGaussianMat(self.hrfLength-self.zc*2,4)
            # HACK
            #self.varR = ones_like(self.varR)

    def computeStDS_StDY(self, rb, nrls, aa):

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

    def sampleNextAlt(self, variables):
        self.reportCurrentVal()

    def sampleNextInternal(self, variables):
        #TODO : comment

        try:
            snrl = self.samplerEngine.get_variable('nrl_by_session')
        except KeyError:
            snrl = self.samplerEngine.get_variable('nrl')

        nrls = snrl.currentValue
        rb   = self.samplerEngine.get_variable('noise_var').currentValue

        if 1:
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion')
            tSQSOptimIni = time.time()
            (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY(rb, nrls,
                                                                     snrl.aa)
            pyhrf.verbose(6, 'Computing StQS StQY optim fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

        if 0:
            nnIdx = self.dataInput.notNullIdxStack
            pyhrf.verbose(6, 'Computing StQS StQY C fashion')
            tSQSOptimIni = time.time()
            import intensivecalc # conditionnal import of C module
                                 # won't work with Pyro Mobile code
            intensivecalc.computeStLambdaSparse(nrls,
                                                self.dataInput.stackX,
                                                nnIdx,
                                                self.dataInput.delta,
                                                self.dataInput.varMBY,
                                                self.varStLambdaS,
                                                self.varStLambdaY, rb);
            pyhrf.verbose(6, 'Computing StQS StQY C fashion'+\
                          ' done in %1.3f sec' %(time.time()-tSQSOptimIni))

            assert np.allclose(self.varDeltaS, self.varStLambdaS.sum(2))
            assert np.allclose(self.varDeltaY, self.varStLambdaY.sum(1))

            self.varDeltaS = self.varStLambdaS.sum(2)
            self.varDeltaY = self.varStLambdaY.sum(1)

        if 0 :
            pyhrf.verbose(6, 'deltaS :' + \
                          str(self.varDeltaS.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaS)
            pyhrf.verbose(6, 'sum StLambdaS :')
            pyhrf.verbose.printNdarray(6, self.varStLambdaS.sum(2))

            pyhrf.verbose(6, 'deltaY :' + str(self.varDeltaY.shape))
            pyhrf.verbose.printNdarray(6, self.varDeltaY)
            pyhrf.verbose(6, 'StLambdaY :' )
            pyhrf.verbose.printNdarray(6, self.varStLambdaY.sum(1))
##            iDiffs = (self.varDeltaS != self.varStLambdaS.sum(2))
##            print 'self.varDeltaS[iDiffs] : ',self.varDeltaS[iDiffs].shape
##            print self.varDeltaS[iDiffs]
##            print 'self.varStLambdaS.sum(2)[iDiffs] :', \
##                  self.varStLambdaS.sum(2)[iDiffs].shape
##            print self.varStLambdaS.sum(2)[iDiffs]

        rh = self.get_variable('hrf_var').currentValue


        if self.priorType == 'voxelwiseIID':
            h = sampleHRF_voxelwise_iid(self.varDeltaS, self.varDeltaY,
                                        self.varR,
                                        rh, self.nbColX, self.nbVox)
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

        pyhrf.verbose(6,'All HRF coeffs :')
        pyhrf.verbose.printNdarray(6, self.currentValue)

        self.updateNorm()

        if np.allclose(self.normalise,1.):
            pyhrf.verbose(6, 'Normalizing samples of HRF, '\
                          'Nrls and mixture parameters at each iteration ...')
            f = self.norm
            #HACK
            #f = self.currentValue.max()
            self.currentValue = self.currentValue / f #/(self.normalise+0.)
            if 1:
                try:
                    if self.samplerEngine.get_variable('nrl_by_session').sampleFlag:
                        self.samplerEngine.get_variable('nrl_by_session').currentValue *= f
                except KeyError:
                    if self.samplerEngine.get_variable('nrl').sampleFlag:
                        self.samplerEngine.get_variable('nrl').currentValue *= f

                    # Normalizing Mixture components
                    smixt_params = self.samplerEngine.get_variable('mixt_params')
                    if 0 and smixt_params.sampleFlag:
                        smixt_params.currentValue[smixt_params.I_MEAN_CA] *= f # Normalizing Mean's activation class
                        smixt_params.currentValue[smixt_params.I_VAR_CI] *= f**2 # Normalizing Variance's activation class
                        smixt_params.currentValue[smixt_params.I_VAR_CA] *= f**2 # Normalizing Variance's in-activation class

            self.updateNorm()

        self.updateXh()
        self.reportCurrentVal()

        # update ytilde for nrls
        try:
            self.samplerEngine.get_variable('nrl_by_session')
        except KeyError: # only if not in multisession case, else ytilde update
                         # is handled by HRF_MultiSess_Sampler.sampleNextAlt
            nrlsmpl = self.samplerEngine.get_variable('nrl')
            nrlsmpl.computeVarYTildeOpt(self.varXh)


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
        stackXh = np.dot(self.dataInput.stackX, hrf)
        return np.reshape(stackXh, (self.nbConditions,self.ny)).transpose()

    def updateXh(self):
        self.varXh = self.calcXh(self.currentValue)

    def detectSignError(self):
        #return False
        pyhrf.verbose(5,'detectSignError...')
        if self.signErrorDetected is not None:
            return self.signErrorDetected
        else:
            pyhrf.verbose(5, 'HRF PM:')
            pyhrf.verbose.printNdarray(3, self.mean)
            pyhrf.verbose(5, 'NRLs PM:')
            pyhrf.verbose.printNdarray(3, self.samplerEngine.get_variable('nrl').mean)
            pyhrf.verbose(5, 'nb negs: %d' %(self.mean <= 0.).sum(dtype=float))
            if (self.mean <= 0.).sum(dtype=float)/len(self.mean) > .9 or \
                    self.mean[np.argmax(abs(self.mean))] < -.2:
                self.signErrorDetected = True
            else:
                #hRef = np.array(genCanoBezierHRF(self.duration, self.eventdt)[1],
                #                   dtype=float)
                #hRef /= ((hRef**2.).sum())**.5
                #h = self.finalValue/((self.finalValue**2.).sum())**.5

                #if (hRef*h).sum() < 0: # if d(hRef,hPM) > d(hRef,-hPM) => sign error

                #np.abs(self.mean.min()) > self.mean.max()

                self.signErrorDetected = False

        return self.signErrorDetected


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

        # print '~~~~~~~~~~~~~~~~~~~~~~~'
        # print 'self.finalValue.shape:', self.finalValue.shape
        # print 'self.trueValue.shape:', self.trueValue.shape

        # Correct sign ambiguity error :
        sign_error = self.detectSignError()
        self.finalValue_sign_corr = self.finalValue * (1 - 2 * sign_error) #sign_error at 0 or 1
        if sign_error:
            pyhrf.verbose(1,'Warning : sign error on HRF')

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
        sScale = self.samplerEngine.get_variable('scale')
        if self.sampleFlag and np.allclose(self.normalise,0.) and \
                not sScale.sampleFlag:
            pyhrf.verbose(6, 'Normalizing posterior mean of HRF each iteration ...')
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


    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)

        ## Correct hrf*nrl scale ambiguity :

        self.finalValueScaleCorr = self.finalValue/self.getScaleFactor()
        self.error = np.zeros(self.hrfLength, dtype=float)
        if self.sampleFlag: #TODO chech for NaNs ...  and not _np.isnan(rh):
            # store errors:
            rh = self.samplerEngine.get_variable('hrf_var').finalValue

            rb = self.samplerEngine.get_variable('noise_var').finalValue
            snrls = self.samplerEngine.get_variable('nrl')
            nrls = snrls.finalValue
            aa = np.zeros((self.nbConditions, self.nbConditions, self.nbVox),
                       dtype=float)
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

        #if self.voxelwise_outputs or pyhrf.__usemode__ == pyhrf.ENDUSER:
        #to obtain hrf.nii ie hrf on each voxel
        if 0:
            #TODO: fix to work with generator ?
            # -> ah output might be enough to browse results voxelwise ...
            for on,o in outputs.iteritems():
                pyhrf.verbose(4, "Treating hrf output %s :\n%s" %(on,o.descrip()))
                newOutput = o.repeat(self.nbVox, 'voxel')
                pyhrf.verbose(4, "New hrf output :\n%s" %(newOutput.descrip()))
                outputs[on] = newOutput


        h = self.finalValue
        nrls = self.samplerEngine.get_variable('nrl').finalValue
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

        outputs['Xh'] = xndarray(dm, axes_names=['time','condition'],
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

        for k,v in outputs.iteritems():
            yield k,v













####################################################
#		Sampling of the HRF prior variance         #
####################################################

class RHSampler(xmlio.XmlInitable, GibbsSamplerVariable) :
    """
    #TODO : comment
    """

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = ['do_sampling', 'val_ini']

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=np.array([0.1]), prior_mean=0.001, prior_var=10):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)
        m = prior_mean
        v = prior_var
        #a=m**2/v +2 , b=m**3/v + m
        if 0:
            self.alpha0 = m**2/v
            self.beta0 = m**3/v + m
        else: #Jeffrey
            self.alpha0 = 1.
            self.beta0 = 0.

        GibbsSamplerVariable.__init__(self,'hrf_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbColX = self.dataInput.nbColX
        self.nbVox = self.dataInput.nbVoxels


        if self.dataInput.simulData is not None:
            if isinstance(self.dataInput.simulData[0], dict):
                self.trueValue = dataInput.simulData[0].get('hrf_var', None)

        if self.trueValue is not None and np.isscalar(self.trueValue):
            self.trueValue = np.array([self.trueValue])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                pyhrf.verbose(3, 'Use true HRF variance value ...')
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('True HRF variance have to be used but '\
                                'none defined.')

        if self.currentValue == None:
            self.currentValue = np.array([0.0001])

    def sampleNextInternal(self, variables):
        #TODO : comment
        shrf = self.get_variable('hrf')
        hrf = shrf.currentValue
        varR = self.get_variable('hrf').varR
        hrfT_R_hrf = np.dot(np.dot(hrf, varR), hrf)
        pyhrf.verbose(5, 'hrfT_R^-1_hrf = ' + str(hrfT_R_hrf))

        # With Jeffrey prior:
        if shrf.priorType == 'voxelwiseIID':
            alpha = 0.5 * (self.nbColX * self.nbVox)
            beta = .5 * hrfT_R_hrf * self.nbVox + self.beta0
        else:
            alpha = 0.5 * (self.nbColX)
            beta = .5 * hrfT_R_hrf

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

    def get_final_value(self):

        scale_f = 1.

        shrf = self.samplerEngine.get_variable('hrf')
        hestim = shrf.finalValue
        htrue = shrf.trueValue

        if htrue is not None:
            scale_f = (htrue**2).sum() / (hestim**2).sum()

        return self.finalValue * scale_f



    # def get_accuracy(self, abs_error, rel_error, fv, tv, atol, rtol):
    #     """ Return the accuray of the estimate *fv*, compared to the true
    #     value *tv*
    #     """
    #     scale_f = 1.

    #     shrf = self.samplerEngine.get_variable('hrf')
    #     hestim = shrf.finalValue
    #     htrue = shrf.trueValue
    #     print 'htrue:', (htrue**2).sum()
    #     if htrue is not None:
    #         scale_f = (htrue**2).sum() / (hestim**2).sum()

    #     fv_scale_corr = fv * scale_f

    #     print 'scale_f:', scale_f
    #     pyhrf.verbose(2, 'HRF variance scale corrected: %f' %fv_scale_corr)
    #     abs_error = np.abs(tv - fv_scale_corr)
    #     rel_error = abs_error/np.maximum(np.abs(tv),np.abs(fv_scale_corr))

    #     acc = rel_error <= rtol

    #     return self.axes_names, acc



    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
        return outputs

if 0: #not maintained
    #######################################################################
    ##### Cas where the hrf is estimated in two parts #####################
    #######################################################################
    class RH_two_parts_Sampler(GibbsSamplerVariable) :
        """
        #TODO : comment
        """

        P_SAMPLE_FLAG = 'sampleFlag'
        P_VAL_INI1 = 'initialValue1'
        P_VAL_INI2 = 'initialValue2'
        P_PR_MEAN = 'priorMean'
        P_PR_VAR = 'priorVar'
        #P_HYPER_PRIOR = 'Jeffrey' #'Proper'

        if pyhrf.__usemode__ == pyhrf.DEVEL:
            defaultParameters = {
                P_VAL_INI1 : np.array([0.001]),
                P_VAL_INI2 : np.array([0.01]),
                P_SAMPLE_FLAG : False,
                P_PR_MEAN : 0.0001,
                P_PR_VAR : 1000.,
                }

        elif pyhrf.__usemode__ == pyhrf.ENDUSER:
            defaultParameters = {
                P_VAL_INI1 : np.array([0.005]),
                P_VAL_INI2 : np.array([0.05]),
                P_SAMPLE_FLAG : False,
                P_PR_MEAN : 0.001,
                P_PR_VAR : 10.,
                }

        parametersToShow = [P_VAL_INI1, P_SAMPLE_FLAG, P_PR_MEAN, P_PR_VAR]

        def __init__(self, parameters=None, xmlHandler=None,
                     xmlLabel=None, xmlComment=None):

            #TODO : comment
            xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel, xmlComment)
            sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
            valIni = self.parameters[self.P_VAL_INI]
            m = self.parameters[self.P_PR_MEAN]
            v = self.parameters[self.P_PR_VAR]
            #a=m**2/v +2 , b=m**3/v + m
            if 0:
                self.alpha0 = m**2/v
                self.beta0 = m**3/v + m
            else: #Jeffrey
                self.alpha0 = -1.
                self.beta0 = 0.

            GibbsSamplerVariable.__init__(self,'hrf_var', valIni=valIni,
                                          sampleFlag=sampleFlag)


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
            shrf = self.get_variable('hrf')
            hrf = shrf.currentValue
            varR = self.get_variable('hrf').varR
            hrfT_R_hrf = np.dot(np.dot(hrf, varR), hrf)
            pyhrf.verbose(5, 'hrfT_R^-1_hrf = ' + str(hrfT_R_hrf))


            if shrf.priorType == 'voxelwiseIID':
                alpha = 0.5*self.nbColX*self.nbVox +  self.alpha0 - 1
                beta = .5*hrfT_R_hrf*self.nbVox + self.beta0
            else:
                alpha = 0.5*self.nbColX + self.alpha0 - 1
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


class ScaleSampler(xmlio.XmlInitable, GibbsSamplerVariable):

    ########################################################
    #TODO : IMPORTANT ! : make it handle 3 class NRL model #
    ########################################################
    ## Sampling not enabled by default -> link to GIG sampler not maintained

    def __init__(self, do_sampling=False, use_true_value=False,
                 val_ini=np.array([1.])):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self,'scale', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)
        self.scaleArrayTmp = np.zeros(1, dtype=float)

    def linkToData(self, dataInput):
        #self.dataInput = dataInput
        self.nbConditions = dataInput.nbConditions
        lhrf = dataInput.hrfLength
        nbc = dataInput.nbConditions
        nbv = dataInput.nbVoxels

        self.theta = .5*(lhrf -1 - nbc*(nbv+1) )

    def sampleNextInternal(self, variables):

        #print 'sampling scale ...'
        # Retrieve dependencies :
        sHrf = self.get_variable('hrf')
        hrf = sHrf.currentValue
        varR = sHrf.varR

        varHRF = self.get_variable('hrf_var').currentValue
        sNrls = self.get_variable('nrl')
        nrl = sNrls.currentValue
        labels = sNrls.labels
        cardC1 = sNrls.cardClass[sNrls.L_CA]

        sMixtP = self.get_variable('mixt_params')
        meanC1 = sMixtP.currentValue[BiGaussMixtureParamsSampler.I_MEAN_CA]
        m1PrVar = sMixtP.meanCAPrVar
        varC1 = sMixtP.currentValue[BiGaussMixtureParamsSampler.I_VAR_CA]
        varC0 = sMixtP.currentValue[BiGaussMixtureParamsSampler.I_VAR_CI]

        # Scale as max of hrf coefficients :
        scale = np.abs(hrf.max())

        # Compute scaled quantities :
        hrfTilde = hrf/scale
        meanC1Tilde = meanC1*scale
        pyhrf.verbose(5, 'meanC1 = '+str(meanC1))
        nrlTilde = nrl*scale
        varC0Tilde = varC0*scale
        varC1Tilde = varC1*scale

        # Compute ditribution parameters :
##        print "varHRF : ", varHRF
        alpha = np.dot(hrfTilde, np.dot(varR, hrfTilde))/varHRF
##        print 'alpha : ', alpha
        beta0 = np.dot(meanC1Tilde, meanC1Tilde)/m1PrVar
        beta1 = np.zeros(self.nbConditions, dtype=float)

        for j in xrange(self.nbConditions):
            pyhrf.verbose(5,'j = '+str(j))
            diff = nrlTilde[j,:] \
                   - (labels[j,:]==sNrls.L_CA).astype(int)*meanC1Tilde[j]
            #print 'nrlTilde[j,:] :'
            #print nrlTilde[j,:]
            #print 'meanC1Tilde[j]:'
            #print meanC1Tilde[j]
            pyhrf.verbose(5,'labels[j,:]:' + \
                          np.array2string(labels[j,:]))
#            print 'diff=',diff
            invSig = np.diag( labels[j,:]/varC1[j] ) + \
                     np.diag((1-labels[j,:])/varC0[j])

            pyhrf.verbose(5,'invSig = '+str(invSig))
            beta1[j] = np.dot(diff, np.dot(invSig,diff))
            pyhrf.verbose(5,'beta1 = '+str(beta1[j]))
            if cardC1[j] > 0 :
                beta0 += meanC1Tilde[j]**2/m1PrVar

        pyhrf.verbose(5,
                      'beta1.sum() ='+str(beta1.sum()))
        pyhrf.verbose(5,'beta0 ='+str(beta0))

        beta = beta1.sum() + beta0

        ####################################################
        ### Note on GIG parametrisation                    #
        ### unuran : GIG(theta, omage, eta) (C library)    #
        ###   VS                                           #
        ### randraw : GIG(lam, chi, psi) (matlab)          #
        ### theta = lam                                    #
        ### omega = sqrt(chi*psi)                          #
        ### eta = sqrt(chi/psi)                            #
        ### ----                                           #
        ### lam = theta                                    #
        ### chi = omega*eta                                #
        ### psi = omega/eta                                #
        ####################################################

        # Adapt parameters :
        omega = (beta*alpha)**0.5
        eta = (beta/alpha)**0.5
        pyhrf.verbose(5,'beta='+str(beta) + \
                      ', alpha='+str(alpha))
        # Draw scale**2 from GIG :
#        if self.theta <1:
#	    self.theta = -1 - self.theta
#	    omegab=omega
#	    omega = eta
#	    eta=omegab
        pyhrf.verbose(5, 'GIG(t=%f, o=%f, e=%f)' %(self.theta, omega, eta))
        if cRandom is not None:
            cRandom.gig(self.theta, omega, eta, 1, self.scaleArrayTmp)
        else:
            raise Exception('pyhrf.stat.cRandom is not available.')
        self.currentValue = self.scaleArrayTmp**0.5
        pyhrf.verbose(5, 'current scale = '+str(self.currentValue))
        assert not np.isnan(self.currentValue)

        # Update HRF scale :
        shrf = self.samplerEngine.get_variable('hrf')
        shrf.currentValue = hrfTilde*self.currentValue
        shrf.updateXh()
        shrf.updateNorm()

        # Update mixture component :
        sMixtP = self.samplerEngine.get_variable('mixt_params')
        sMixtP.currentValue[sMixtP.I_MEAN_CA] = meanC1Tilde/self.currentValue

        # Update NRLs :
        sNRLs = self.samplerEngine.get_variable('nrl')
        sNRLs.currentValue = nrlTilde/scale

#    def finalizeSampling(self):
#        del self.scaleArrayTmp

    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
        return outputs
