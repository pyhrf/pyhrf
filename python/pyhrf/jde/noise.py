# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import numpy as np
from samplerbase import *
import pyhrf
from pyhrf import xmlio
import copy as copyModule


class NoiseVariance_Drift_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):
        """
        #TODO : comment
        """
        xmlio.XmlInitable.__init__(self)
        sampleFlag = do_sampling
        valIni = val_ini
        useTrueVal = use_true_value

        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=valIni,
                                      useTrueValue=useTrueVal,
                                      sampleFlag=sampleFlag, axes_names=['voxel'],
                                      value_label='PM Noise Var')


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny


    def checkAndSetInitValue(self, variables):
        if self.currentValue is None :
            if not self.sampleFlag and self.dataInput.simulData != None :
                self.currentValue = self.dataInput.simulData.noise.variance
            else:
                self.currentValue = 0.5 * self.dataInput.varData

    def sampleNextInternal(self, variables):

        #print 'Step 3 : Noise variance Sampling *****RelVar*****'
        #print '         varYbar begin =',self.get_variable('nrl').varYbar.sum()
        #print '         varYtilde begin =',self.get_variable('nrl').varYtilde.sum()

        var_y_bar = self.get_variable('nrl').varYbar

        beta = (var_y_bar * var_y_bar).sum(0)/2.

        #gammaSamples = np.random.gamma(0.5*(self.ny - self.dataInput.colP +1)-1, 1,
        #                            self.nbVox)

        #gammaSamples = np.random.gamma((self.ny + 1.)/2, 1, self.nbVox)
        gammaSamples = np.random.gamma((self.ny)/2., 1, self.nbVox) # Jeffrey's prior

        np.divide(beta, gammaSamples, self.currentValue)


class NoiseVarianceSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """
    #TODO : comment

    """
    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'],
                                      value_label='PM Noise Var')


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX

        # Do some allocations :
        self.beta = np.zeros(self.nbVox, dtype=float)
        self.mXhQXh = np.zeros((self.nbConditions,self.nbConditions),
                               dtype=float)

        if self.dataInput.simulData is not None:
            if isinstance(self.dataInput.simulData, dict):
                if dataInput.simulData.has_key('v_gnoise'):
                    self.trueValue = self.dataInput.simulData['v_noise']
                    pyhrf.verbose(3, 'True noise variance = %1.3f' \
                                  %self.trueValue)

            elif isinstance(dataInput.simulData, list):
                    sd = dataInput.simulData[0]
                    if isinstance(sd, dict):
                        self.trueValue = sd['noise'].var(0).astype(np.float64)
                    else:
                        self.trueValue = sd.noise.variance
            else:
                self.trueValue = self.dataInput.simulData.noise.variance


    def checkAndSetInitValue(self, variables):
        if self.trueValue is not None and np.isscalar(self.trueValue):
            self.trueValue = self.trueValue * np.ones(self.nbVox)
        if self.currentValue == None :
            if self.useTrueValue:
                if self.trueValue is not None:
                    pyhrf.verbose(3, 'Use true noise variance value ...')
                    self.currentValue = self.trueValue
                else:
                    raise Exception('True noise varaince have to be used but none defined.')
            else:
                self.currentValue = 0.5 * self.dataInput.varData


        #if self.currentValue == None :
            #if not self.sampleFlag and self.dataInput.simulData != None :
                #self.currentValue = self.dataInput.simulData.noise.variance
            #else:
                #self.currentValue = 0.1 * self.dataInput.varData

    def computeMXhQXh(self ,h, varXQX):
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                self.mXhQXh[j,k] = np.dot(np.dot(h,varXQX[j,k]),h)

    def compute_aaXhQXhi(self, aa, i):
        aaXhQXhi = 0.0
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                aaXhQXhi += aa[j,k,i] * self.mXhQXh[j,k]
                #print 'aa[%d,%d,%d] * self.mXhQXh[%d,%d] = %f * %f = %f' \
                #    %(j,k,i,j,k, aa[j,k,i], self.mXhQXh[j,k],
                #      aa[j,k,i] * self.mXhQXh[j,k])
                #print aa[j,k,i] * self.mXhQXh[j,k],', ',
                #print ''
                #aaXhQXhiTab[i] += snrl.aa[j,k,i] * self.mXhQXh[j,k]
        return aaXhQXhi

    def sampleNextInternal_bak(self, variables):

        y_tilde = self.samplerEngine.get_variable('nrl').varYtilde
        beta = (y_tilde * y_tilde).sum(0)/2
        gammaSamples = np.random.gamma((self.ny - 1.)/2, 1, self.nbVox)

        np.divide(beta, gammaSamples, self.currentValue)



    def sampleNextInternal(self, variables):
        #TODO : comment

        h = self.get_variable('hrf').currentValue
        varXQX = self.dataInput.matXQX
        snrl = self.get_variable('nrl')

        saXh = snrl.sumaXh
        ## Example to retrieve variable from sharedData :
        # saXh = self.samplerEngine.sharedData.get('varXh')

        yTQy = self.dataInput.yTQy
        matQy = self.dataInput.matQy
##        aaXhQXh = np.zeros((self.nbConditions,self.nbConditions,self.nbVox),
##                        dtype=float)

##        aa = snrl.aa.reshape(self.nbConditions**2,self.nbVox)
##        mXhQXh = self.mXhQXh[j,k]
##        multiply(aa, , aaXhQXh[j,k,:])

        self.computeMXhQXh(h, varXQX)

        #aaXhQXhiTab = np.zeros(self.nbVox)
        aa = snrl.aa.swapaxes(2,0)
        aaXhQXh = (aa * self.mXhQXh).sum(1).sum(1)
        #print '(aa * self.mXhQXh)[0,:,:] ', (aa * self.mXhQXh)[0,:,:].shape
        #print np.array2string( (aa * self.mXhQXh)[0,:,:], max_line_width=200,
        #                         separator=',')
        if 0:
            for i in xrange(self.nbVox):
                #aaXhQXhi = self.compute_aaXhQXhi(snrl.aa, i)
                #print 'aaXhQXhi :', aaXhQXhi
                #print 'aaXhQXh[i] :', aaXhQXh[i]
                #assert np.allclose(aaXhQXhi, aaXhQXh[i])
                self.beta[i] = .5*(yTQy[i] - 2*np.dot(saXh[:,i], matQy[:,i]) \
                                  + aaXhQXh[i])


    ##            pyhrf.verbose(6,'aXhi %s'%str(aXhi.shape))
    ##            pyhrf.verbose(6,
    ##                          np.array2string(aXhi,precision=3))

    ##            pyhrf.verbose(6,'mQyi %s'%str(aXhi.shape))
    ##            pyhrf.verbose(6,
    ##                          np.array2string(aXhi,precision=3))


    ##        pyhrf.verbose(6,'All betas apost opt calc:')
    ##        pyhrf.verbose(6,np.array2string(betaOpt,precision=3))

        else:
            y_tilde = self.samplerEngine.get_variable('nrl').varYtilde
            for i in xrange(self.nbVox):
                varYtildeTdelta = np.dot(y_tilde[:,i],self.dataInput.delta)
                self.beta[i] = 0.5*np.dot(varYtildeTdelta,y_tilde[:,i])
        pyhrf.verbose(6, 'All betas apost :')
        pyhrf.verbose.printNdarray(6,self.beta)
        pyhrf.verbose(4, 'betas apost = %1.3f(%1.3f)'
                      %(self.beta.mean(),self.beta.std()))

##        assert allclose(self.beta, betaOpt)

        gammaSamples = np.random.gamma(0.5*(self.ny - self.dataInput.colP - 1),
                                       1, self.nbVox)

        pyhrf.verbose(5,'sigma2 ~betas/Ga(%1.3f,1)'
                      %(0.5*(self.ny - self.dataInput.colP +1)))
        self.currentValue = np.divide(self.beta, gammaSamples)

        #pyhrf.verbose(4, 'All noise vars :')
        #pyhrf.verbose.printNdarray(4,self.currentValue)
        pyhrf.verbose(4, 'noise vars = %1.3f(%1.3f)'
                      %(self.currentValue.mean(), self.currentValue.std()))
        if (self.currentValue < 0).any():
            print 'negative variances !!'
            print 'at vox : ', np.where(self.currentValue<0)
            print '-> betas = ', self.beta[np.where(self.currentValue<0)]
            print '-> gamma samples = ', gammaSamples[np.where(self.currentValue<0)]
            print '-> yTQy[i] = ', yTQy[np.where(self.currentValue<0)]
            i = np.where(self.currentValue<0)[0]
            print '-> setting to almost zero'
            self.currentValue[np.where(self.currentValue<0)] = 0.0001
            #print '-> np.dot(saXh[:,i], matQy[:,i]) =', np.dot(saXh[:,i], matQy[:,i])
            #print '-> aaXhQXhiTab[i] =', aaXhQXhiTab[np.where(self.currentValue<0)]

    # def initOutputs(self, outputs, nbROI=-1):
    #     if pyhrf.__usemode__ == pyhrf.DEVEL:
    #         GibbsSamplerVariable.initOutputs(self, outputs, nbROI)

    # def fillOutputs(self, outputs, iROI=-1):
    #     if pyhrf.__usemode__ == pyhrf.DEVEL:
    #         GibbsSamplerVariable.fillOutputs(self, outputs, iROI)

        #print 'noise variance = ', self.currentValue


    def finalizeSampling(self):
        GibbsSamplerVariable.finalizeSampling(self)
        del self.mXhQXh
        del self.beta

#######################################################
# Noise Variance sampler with Relevant Variable
#######################################################

class NoiseVarianceSamplerWithRelVar(NoiseVarianceSampler):


    def compute_aawwXhQXhi(self, ww, aa, i):
        aawwXhQXhi = 0.0
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                aawwXhQXhi += ww[j,k] * aa[j,k,i] * self.mXhQXh[j,k]
        return aawwXhQXhi

    def computeWW(self, w, destww):
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                destww[j,k] = w[j]*w[k]

    def sampleNextInternal(self, variables):
        #TODO : comment

        h = self.get_variable('hrf').currentValue
        varXQX = self.dataInput.matXQX
        snrl = self.get_variable('nrl')
        w = self.get_variable('W').currentValue

        swaXh = snrl.sumWaXh

        nrl = snrl.currentValue
        varYtilde = snrl.varYtilde
        yTQy = self.dataInput.yTQy
        matQy = self.dataInput.matQy

        self.computeMXhQXh(h, varXQX)

        ww = np.zeros((self.nbConditions, self.nbConditions), dtype=int)
        self.computeWW(w, ww)

        aa = snrl.aa.swapaxes(2,0)
        aaww = aa * ww
        aawwXhQXh = (aaww * self.mXhQXh).sum(1).sum(1)

        for i in xrange(self.nbVox):
            self.beta[i] = .5*(yTQy[i] - 2*np.dot(swaXh[:,i], matQy[:,i]) \
                              + aawwXhQXh[i])

        pyhrf.verbose(6, 'All betas apost :')
        pyhrf.verbose.printNdarray(6,self.beta)
        pyhrf.verbose(4, 'betas apost = %1.3f(%1.3f)'
                      %(self.beta.mean(),self.beta.std()))

        gammaSamples = np.random.gamma(0.5*(self.ny - self.dataInput.colP +1)-1, 1,
                                    self.nbVox)

        pyhrf.verbose(5,'sigma2 ~betas/Ga(%1.3f,1)'
                      %(0.5*(self.ny - self.dataInput.colP +1)))
        self.currentValue = np.divide(self.beta, gammaSamples)

        pyhrf.verbose(4, 'noise vars = %1.3f(%1.3f)'
                      %(self.currentValue.mean(), self.currentValue.std()))
        if (self.currentValue < 0).any():
            print 'negative variances !!'
            print 'at vox : ', np.where(self.currentValue<0)
            print '-> betas = ', self.beta[np.where(self.currentValue<0)]
            print '-> gamma samples = ', gammaSamples[np.where(self.currentValue<0)]
            print '-> yTQy[i] = ', yTQy[np.where(self.currentValue<0)]
            i = np.where(self.currentValue<0)[0]
            print '-> np.dot(swaXh[:,i].transpose(), matQy[:,i]) =', np.dot(swaXh[:,i].transpose(), matQy[:,i])

    # def initOutputs(self, outputs, nbROI=-1):
    #     if pyhrf.__usemode__ == pyhrf.DEVEL:
    #         GibbsSamplerVariable.initOutputs(self, outputs, nbROI)

    # def fillOutputs(self, outputs, iROI=-1):
    #     if pyhrf.__usemode__ == pyhrf.DEVEL:
    #         GibbsSamplerVariable.fillOutputs(self, outputs, iROI)

    def finalizeSampling(self):
        GibbsSamplerVariable.finalizeSampling(self)
        del self.mXhQXh
        del self.beta


#######################################################
# Noise Variance sampler with Habituation
#######################################################



#class NoiseVariancewithHabSampler(NoiseVarianceSampler):
    #pass



    #"""--------------------

class NoiseVariancewithHabSampler(NoiseVarianceSampler):
    """
    #TODO : Sampling procedure for noise variance parameters (white noise)
    #in case of habituation modeling wrt magnitude

    """

    def sampleNextInternal(self, variables):
        #TODO : comment
        h = self.get_variable('hrf').currentValue
        smplNRL = self.get_variable('nrl')
        aXh = smplNRL.aXh
        saXh = smplNRL.sumaXh
        #aaXhtQXh = smplNRL.sumaXhtQaXh
        #saXtQ = smplNRL.sumaXtQ
        varYtilde = smplNRL.varYtilde
        ## Example to retrieve variable from sharedData :
        # saXh = self.samplerEngine.sharedData.get('varXh')

        yTQy = self.dataInput.yTQy
        matQy = self.dataInput.matQy

        for i in xrange(self.nbVox):
            #aaXhtQXhi = np.dot(h, np.dot(saXtQaX[i,:,:], h) )
            aaXhQXhi = 0.0
	    for j in xrange(self.nbConditions):
                for k in xrange(self.nbConditions):
                    aaXhQXhi += np.dot(np.dot(aXh[:,i,j], self.dataInput.delta) , aXh[:,i,k])


	    self.beta[i] = .5*(yTQy[i] - 2*np.dot(saXh[:,i], matQy[:,i]) \
                              + aaXhQXhi)

        pyhrf.verbose(6,'All betas apost :')
        pyhrf.verbose.printNdarray(6,self.beta)
        pyhrf.verbose(5, 'betas apost = %1.3f(%1.3f)'
                      %(self.beta.mean(),self.beta.std()))

        a = 0.5*(self.ny - self.dataInput.colP +1)
        gammaSamples = np.random.gamma(a, 1, self.nbVox)

        pyhrf.verbose(5,'sigma2 ~betas/Ga(%1.3f,1)' %a)
        self.currentValue = np.divide(self.beta, gammaSamples)

        pyhrf.verbose(6, 'All noise vars :')
        pyhrf.verbose.printNdarray(6,self.currentValue)
        pyhrf.verbose(4, 'noise vars = %1.3f(%1.3f)'
                      %(self.currentValue.mean(), self.currentValue.std()))

    def finalizeSampling(self):

        del self.beta

    #"""

#######################################################
# Noise Variance AR sampler
#######################################################
class NoiseVarianceARSampler(NoiseVarianceSampler):


    def checkAndSetInitValue(self, variables):

        NoiseVarianceSampler.checkAndSetInitValue(self, variables)
	self.varYTilde = np.empty((self.ny, self.nbVox), dtype = float)

    def computeVarYTilde(self, varNrls, varXh, varMBYPl):
        for v in xrange(self.nbVox):
            repNRLv = np.tile(varNrls[:,v], (self.ny, 1))
            avjXjh = repNRLv * varXh
            self.varYTilde[:,v] = varMBYPl[:,v] - avjXjh.sum(axis=1)

    def sampleNextInternal(self, variables):
        #TODO : comment

        smplARp = variables[self.samplerEngine.I_NOISE_ARP]
        InvAutoCorrNoise = smplARp.InvAutoCorrNoise
        smplHRF = self.get_variable('hrf')
        varXh = smplHRF.varXh
        smplDrift =  self.get_variable('drift')
        varMBYPl = smplDrift.varMBYPl
        smplNRL = self.get_variable('nrl')
        varNRLs = smplNRL.currentValue
        self.computeVarYTilde(varNRLs, varXh, varMBYPl)
#        self.varYtilde = self.get_variable('nrl').varYtilde

        for i in xrange(self.nbVox):
            varYtildeTdelta = np.dot(self.varYTilde[:,i],InvAutoCorrNoise[:,:,i])
            self.beta[i] = 0.5*np.dot(varYtildeTdelta,self.varYTilde[:,i])
        pyhrf.verbose(6,'betas apost :')
        pyhrf.verbose(6,np.array2string(self.beta,precision=3))
        pyhrf.verbose(6,'sigma2 ~betas/Ga(%1.3f,1)'
                      %(0.5*(self.ny + 1)))
        gammaSamples = np.random.gamma(0.5*(self.ny + 1), 1, self.nbVox)
        self.currentValue = np.divide(self.beta, gammaSamples)
        pyhrf.verbose(6, 'All noise vars :')
        pyhrf.verbose(6,
                      np.array2string(self.currentValue,precision=3))
        pyhrf.verbose(4, 'noise vars = %1.3f(%1.3f)'
                      %(self.currentValue.mean(), self.currentValue.std()))

    def finalizeSampling(self):
        del self.beta
        del self.mXhQXh
        del self.varYTilde


class NoiseARParamsSampler(xmlio.XmlInitable, GibbsSamplerVariable):
    """

    """

    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_USE_TRUE_VALUE = 'useTrueValue'

    # parameters definitions and default values :
    defaultParameters = {
        P_VAL_INI : None,
        P_USE_TRUE_VALUE : False,
        P_SAMPLE_FLAG : True,
        }


    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None):

        #TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self, 'noiseARParam', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'],
                                      value_label='PM AR Params')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVox = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.voxIdx = np.arange(self.nbVox, dtype=int)

        # Do some allocations :
        varARp = np.zeros((self.nbVox), dtype=float)

    def checkAndSetInitValue(self, variables):
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for init but None defined')
            else:
                self.currentValue = self.trueValue

        if self.currentValue is None :
            if not self.sampleFlag and self.dataInput.simulData != None \
                   and hasattr(self.dataInput.simulData.noise,'AR'):
                if self.dataInput.simulData.noise.AR.shape[0] == 1 :# SI
                    self.currentValue = np.zeros(self.nbVox) + \
                        self.dataInput.simulData.noise.AR[0]
                else: # SV
                    self.currentValue = self.dataInput.simulData.noise.AR[:,0]
            else:
                self.currentValue = 0.4 * np.random.rand( self.nbVox ) + .1
        #print "checkAndSetInitValue NOISE ARP"
        self.InvAutoCorrNoise = np.zeros( (self.ny, self.ny, self.nbVox) ,
                                       dtype=float)
        self.computeInvAutoCorrNoise(self.currentValue)


    def sampleNextInternal(self, variables):
        reps = self.get_variable('noise_var').currentValue

        #TODO : precomputations for parallel AR sampling
        self.varYtilde = self.get_variable('nrl').varYtilde
        Ytilde_truncated = self.varYtilde[1:self.ny-1,:]    #
        # compute the sequence of A_i = sum_{n=2}^{nscans -1} y_{n,i}^2, i is indexing voxels
        A = np.diag(np.dot( Ytilde_truncated.transpose(), Ytilde_truncated ) )

        # compute the sequence of B_i = sum_{n=1}^{nscans -1} y_{n,i}y_{i,n+1}, i is indexing voxels
        Ytilde_shifted= self.varYtilde[:self.ny-1,:]
        B = np.diag( np.dot(Ytilde_shifted.transpose(), self.varYtilde[1:,:]) )
        M = np.divide(B,A)
        sig2 = np.divide( reps, A)
#        CommonPart = .5 * np.divide((1. - M**2),sig2)
#        CommonPartB = .5 * np.divide(2.* M - 1,log(2.)*sig2 )
#        idxM = where( M<1. )
#        Minf1 = idxM[1]
#        Msup1 = setdiff1d( self.voxIdx, Minf1 )
        #Computation of a_i = A_i/(2*reps_i)*(1-M_i^2)*(1+M_i)
#        a = zeros(self.nbVox, dtype=float)
#        a[Minf1] = multiply( CommonPart[Minf1], 1 + M[Minf1] )
#        a[Msup1] = CommonPartB[Msup1]
        #Computation of b_i = A_i/(2*reps_i)*(1-M_i^2)*(1-M_i)
#        b = zeros(self.nbVox, dtype=float)
#        b[Minf1] =  multiply( CommonPart[Minf1], 1 - M[Minf1] )
#        b[Msup1] = zeros(size(Msup1) )
	varARp = self.MH_ARsampling_gauss_proposal(sig2, M)
#	ARp = self.MH_ARsampling_optim(A,reps,M)

        self.currentValue = varARp
        pyhrf.verbose(6, 'All AR params :')
        pyhrf.verbose(6, np.array2string(self.currentValue,precision=3))
        pyhrf.verbose(4, 'ARp = %1.3f(%1.3f)'
                      %(self.currentValue.mean(), self.currentValue.std()))
        self.computeInvAutoCorrNoise(varARp)


    def computeInvAutoCorrNoise(self, ARp):
        pyhrf.verbose(6, 'ARp :')
        pyhrf.verbose.printNdarray(6, ARp)

        for v in xrange(self.nbVox):
            DiagAutoCorr = np.repeat( 1+ARp[v]**2,self.ny-2 )
            DiagAutoCorr = np.concatenate( ([1],DiagAutoCorr, [1]) )
##            pyhrf.verbose(6, 'DiagAutoCorr :')
##            pyhrf.verbose.printNdarray(6, DiagAutoCorr)
            InvAutoCorrMatrix = np.diag( DiagAutoCorr, k=0 )
##            pyhrf.verbose(6, 'InvAutoCorrMatrix 1st diag :')
##            pyhrf.verbose.printNdarray(6, InvAutoCorrMatrix)

            DiagInfAutoCorr = np.repeat( -ARp[v], self.ny -1 )
            InvAutoCorrMatrix += np.diag( DiagInfAutoCorr, k=1 )
            InvAutoCorrMatrix += np.diag( DiagInfAutoCorr, k=-1 )
##            pyhrf.verbose(6, 'InvAutoCorrMatrix all diags :')
##            pyhrf.verbose.printNdarray(6, InvAutoCorrMatrix)
            self.InvAutoCorrNoise[:,:,v] = InvAutoCorrMatrix

        pyhrf.verbose(6, 'InvAutoCorrNoise :')
        pyhrf.verbose.printNdarray(6, self.InvAutoCorrNoise[:,:,0])

    def MH_ARsampling_gauss_proposal(self, sig2, M):
        ARp = np.zeros(self.nbVox, dtype=float)
        for i in xrange(self.nbVox):
            rho0=self.currentValue[i]
	    #frho0 = np.sqrt(1. -rho0**2 )*np.exp(-.5/sig2[i]*(rho0-M[i])**2)
	    #grho0=np.exp(-.5/sig2[i]*(rho0-M[i])**2)
	    #grho0=(1+rho0)**(d[i]+.5)*(1-rho0)**(e[i]+.5)
	    Frac_rho0= np.sqrt(1. -rho0**2)
	    y=10
	    while abs(y)>1:
		#y = np.random.normal( M[i], np.sqrt(sig2[i]) )
		y = np.sqrt(sig2[i]) * np.random.randn(1) + M[i]
	    #frho=np.sqrt(1-y**2)*np.exp(-.5/sig2[i]*(y-M[i])**2)
	    #grho= np.exp(-.5/sig2[i]*(y-M[i])**2)
	    Frac_rho=np.sqrt(1. - y**2)
	    accept_ratio = min( np.array((1.,Frac_rho/Frac_rho0), dtype = float) )
	    u=np.random.rand(1)
	    if u<accept_ratio:
	      ARp[i] =y
	    else:
	      ARp[i] = self.currentValue[i]
            #print "Voxel ",i, ": acceptation rate = ", accept_rate[i]
        return ARp


    def MH_ARsampling_optim(self,A,reps,M):
        ARp = np.zeros(self.nbVox, dtype=float)
##        accept_rate =  zeros(self.nbVox, dtype=float)
        R = []
        for i in xrange(self.nbVox):
            rho0=self.currentValue[i]
            Lam =  A[i]/reps[i]/2
            LamM = Lam * M[i]
            polyCoeffs=[-Lam, LamM, Lam+.5,- LamM]
            R = scipy.roots(polyCoeffs)
            x = R[abs(R)<=1]
            #Evaluate the density function at rho0
            d = .5 + Lam * (1 + M[i] - 2*x) * (1+x)^2
            e = .5 + Lam * (1 - M[i] + 2*x) * (1-x)^2
            #frho0 = np.sqrt(1. -rho0**2)*np.exp(-.5/sig2[i] *( rho0 - M[i])**2)
 	    #grho0 = (1+rho0)**(a[i]+.5)*(1-rho0)**(b[i]+.5)

            #Sampling from the right Beta distribution
	    x = np.random.beta( d+1., e+1. )
	    #change of variables
            rho=2*x-1
            #frho =  np.sqrt(1.-rho**2)*np.exp(-.5/sig2[i] *( rho - M[i])**2)
	    #grho = (1+rho)**(a[i]+.5)*(1-rho)**(b[i]+.5)
	    #Perform the Metropolis step
	    #Frac_rho = np.sqrt(1.-rho**2)
            LogFrac = (d-.5)*log((1+rho0)/(1+rho)) + (e-.5)*log((1-rho0)/(1-rho))
            LogFrac-= Lam*(rho-rho0)*(rho+rho0-2*M[i])
            if LogFrac>=0:
                accept_ratio = 1
                ARp[i] =rho
                ACCEPT=1
            else:
               accept_ratio = np.min( array((1,np.exp(LogFrac)), dtype = float) )
	       u=np.random.rand(1)
	       if u<accept_ratio:
	           ARp[i] =rho
	       else:
	           ARp[i] = self.currentValue[i]
            #print "Voxel ",i, ": acceptation rate = ", accept_rate[i]
        return ARp

    def finalizeSampling(self):
	 pass
#        del self.InvAutoCorrNoise
