import pyhrf
import numpy as np
from numpy.testing import assert_almost_equal
from samplerbase import GibbsSampler, GibbsSamplerVariable

from pyhrf import xmlio
from pyhrf.ndarray import xndarray, stack_cuboids

from pyhrf.jde.models import WN_BiG_Drift_BOLDSamplerInput, GSDefaultCallbackHandler

from pyhrf.boldsynth.hrf import genGaussianSmoothHRF, getCanoHRF
from pyhrf.boldsynth.scenarios import build_ctrl_tag_matrix
from pyhrf.jde.intensivecalc import asl_compute_y_tilde
from pyhrf.jde.intensivecalc import sample_potts

def b():
    raise Exception()

def compute_StS_StY(rls, v_b, mx, mxtx, ybar, rlrl, yaj, ajak_vb):
    """ yaj and ajak_vb are only used to store intermediate quantities, they're
    not inputs.
    """
    nb_col_X = mx.shape[2]
    nb_conditions = mxtx.shape[0]
    varDeltaS = np.zeros((nb_col_X,nb_col_X), dtype=float )
    varDeltaY = np.zeros((nb_col_X), dtype=float )

    for j in xrange(nb_conditions):
        np.divide(ybar, v_b, yaj)
        yaj *= rls[j,:]
        varDeltaY +=  np.dot(mx[j,:,:].T, yaj.sum(1))

        for k in xrange(nb_conditions):
            np.divide(rlrl[j,k,:], v_b, ajak_vb)
            pyhrf.verbose(6, 'ajak/rb :')
            pyhrf.verbose.printNdarray(6, ajak_vb)
            varDeltaS += ajak_vb.sum() * mxtx[j,k,:,:]

    return (varDeltaS, varDeltaY)


def compute_bRpR(brl, prl, nbConditions, nbVoxels):
    # aa[m,n,:] == aa[n,m,:] -> nb ops can be /2
    rr = np.zeros((nbConditions, nbConditions, nbVoxels), dtype=float)
    for j in xrange(nbConditions):
        for k in xrange(nbConditions):
            np.multiply(brl[j,:], prl[k,:], rr[j,k,:])
    return rr


class ResponseSampler(GibbsSamplerVariable):
    """
    Generic parent class to perfusion response & BOLD response samplers
    """

    def __init__(self, name, response_level_name, variance_name, smooth_order=2,
                 zero_constraint=True, duration=25., normalise=1., val_ini=None,
                 do_sampling=True, use_true_value=False):

        self.response_level_name = response_level_name
        self.var_name = variance_name
        an = ['time']
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='Delta signal')

        self.normalise = normalise
        self.zc = zero_constraint
        self.duration = duration
        self.varR = None
        self.derivOrder = smooth_order

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.nbColX = self.dataInput.nbColX
        self.hrfLength = self.dataInput.hrfLength
        self.dt = self.dataInput.dt
        self.eventdt = self.dataInput.dt

        #print dataInput.simulData
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            #1st voxel:
            self.trueValue = dataInput.simulData[0][self.name][:,0]

        self.yBj = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.BjBk_vb =  np.zeros((self.nbVoxels), dtype=float)

        self.ytilde = np.zeros((self.ny,self.nbVoxels), dtype=float)

        self.track_sampled_quantity(self.ytilde, self.name + '_ytilde',
                                    axes_names=['time', 'voxel'])

    def checkAndSetInitValue(self, variables):

        _,  self.varR = genGaussianSmoothHRF(self.zc,
                                             self.hrfLength,
                                             self.eventdt, 1.,
                                             order=self.derivOrder)
        hrfValIni = None
        if self.useTrueValue :
            if self.trueValue is not None:
                hrfValIni = self.trueValue[:]
            else:
                raise Exception('Needed a true value for hrf init but '\
                                    'None defined')

        if hrfValIni is None:
            pyhrf.verbose(6, 'self.duration=%d, self.eventdt=%1.2f' \
                              %(self.duration,self.eventdt))

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

        pyhrf.verbose(4,'hrfValIni:' + str(hrfValIni.shape))
        pyhrf.verbose.printNdarray(6, hrfValIni)
        pyhrf.verbose(4, 'self.hrfLength:' \
                          +str(self.hrfLength))

        normHRF = (sum(hrfValIni**2))**(0.5)
        hrfValIni /= normHRF

        self.currentValue = hrfValIni[:]

        if self.zc :
            self.axes_domains['time'] = np.arange(len(self.currentValue)+2) \
                                         * self.eventdt
        else:
            self.axes_domains['time'] = np.arange(len(self.currentValue)) \
                                         * self.eventdt


        pyhrf.verbose(4,'hrfValIni after ZC:' +\
                      str(self.currentValue.shape))
        pyhrf.verbose.printNdarray(6, self.currentValue )

        self.updateNorm()
        self.updateXResp()


    def calcXResp(self, resp, stackX=None):
        stackX = stackX or self.get_stackX()
        stackXResp = np.dot(stackX, resp)
        return np.reshape(stackXResp, (self.nbConditions,self.ny)).transpose()

    def updateXResp(self):
        self.varXResp = self.calcXResp(self.currentValue)

    def updateNorm(self):
        self.norm = sum(self.currentValue**2.0)**0.5

    def get_stackX():
        raise NotImplementedError()

    def get_mat_X(self):
        raise NotImplementedError()

    def get_rlrl(self):
        raise NotImplementedError()

    def get_mat_XtX(self):
        raise NotImplementedError()

    def get_ybar(self):
        raise NotImplementedError()

    def computeYTilde(self):
        raise NotImplementedError()

    def sampleNextInternal(self, variables):
        raise NotImplementedError

    def setFinalValue(self):

        fv = self.mean #/self.normalise
        if self.zc:
            # Append and prepend zeros
            self.finalValue = np.concatenate(([0], fv, [0]))
            self.error = np.concatenate(([0], self.error, [0]))
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

        pyhrf.verbose(4, '%s finalValue :' %self.name)
        pyhrf.verbose.printNdarray(4, self.finalValue)



class PhysioBOLDResponseSampler(ResponseSampler, xmlio.XmlInitable):

    def __init__(self, smooth_order=2, zero_constraint=True, duration=25.,
                 normalise=1., val_ini=None, do_sampling=True,
                 use_true_value=False):

        """
        """
        xmlio.XmlInitable.__init__(self)
        ResponseSampler.__init__(self, 'brf', 'brl', 'brf_var', smooth_order,
                                 zero_constraint, duration, normalise, val_ini,
                                 do_sampling, use_true_value)

    def get_stackX(self):
        return self.dataInput.stackX

    def get_mat_X(self):
        return self.dataInput.varX

    def get_mat_XtX(self):
        return self.dataInput.matXtX

    def get_mat_XtWX(self):
        return self.dataInput.XtWX

    def samplingWarmUp(self, v):
        self.new_factor_mean = np.zeros_like(self.currentValue)
        self.track_sampled_quantity(self.new_factor_mean, self.name + '_new_factor_mean',
                                    axes_names=['time'])
        
            
    def computeYTilde(self):
        """ y - \sum cWXg - Pl - wa """

        sumcXg = self.get_variable('prl').sumBXResp
        drift_sampler = self.get_variable('drift_coeff')
        Pl = drift_sampler.Pl
        bl_sampler = self.get_variable('perf_baseline')
        wa = bl_sampler.wa
        y = self.dataInput.varMBY

        ytilde = y - sumcXg - Pl - wa

        if 0 and self.dataInput.simulData is not None: #hack
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            brl_sampler = self.get_variable('brl')
            prl_sampler = self.get_variable('prl')
            prf_sampler = self.get_variable('prf')

            if not prl_sampler.sampleFlag and not prf_sampler.sampleFlag and\
                    prl_sampler.useTrueValue and prf_sampler.useTrueValue:
                perf = np.dot(self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(sumcXg, perf)

            if not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                assert_almost_equal(Pl, sd['drift'])

            if not bl_sampler.sampleFlag and bl_sampler.useTrueValue:
                assert_almost_equal(wa, np.dot(self.dataInput.W,
                                               sd['perf_baseline']))

            if not brl_sampler.sampleFlag and brl_sampler.useTrueValue and \
                    not drift_sampler.sampleFlag and drift_sampler.useTrueValue and\
                    not prf_sampler.sampleFlag and prf_sampler.useTrueValue and\
                    not prl_sampler.sampleFlag and prl_sampler.useTrueValue and\
                    not bl_sampler.sampleFlag and bl_sampler.useTrueValue:
                assert_almost_equal(ytilde, sd['bold_stim_induced'][0:-1:osf] +\
                                        sd['noise'])
        return ytilde

    def sampleNextInternal(self, variables):
        """
        Sample BRF

        changes to mean:
        changes to var:
        """

        rl_sampler = self.get_variable(self.response_level_name)
        rl = rl_sampler.currentValue
        rlrl = rl_sampler.rr

        noise_var = self.get_variable('noise_var').currentValue

        mx = self.get_mat_X()
        mxtx = self.get_mat_XtX()

        self.ytilde[:] = self.computeYTilde()

        StS, StY = compute_StS_StY(rl, noise_var, mx, mxtx, self.ytilde, rlrl,
                                   self.yBj, self.BjBk_vb)

        v_resp = self.get_variable(self.var_name).currentValue

        omega = self.get_variable('prf').omega_value

        prf = self.get_variable('prf').currentValue
        if 'deterministic' in self.get_variable('prf').prior_type:
            v_prf = 1.
        else:
            v_prf =  self.get_variable('prf_var').currentValue

        sigma_g_inv = self.get_variable('prf').varR

        self.new_factor_mean[:] = np.dot(np.dot(omega.transpose(),sigma_g_inv),prf)\
                                  /v_prf
        new_factor_var = np.dot(np.dot(omega.transpose(), sigma_g_inv),omega)\
                         /v_prf

        varInvSigma = StS + self.nbVoxels * self.varR / v_resp + new_factor_var
        mean_h = np.linalg.solve(varInvSigma,StY + self.new_factor_mean)
        resp = np.random.multivariate_normal(mean_h,np.linalg.inv(varInvSigma))
        if self.normalise:
            norm = (resp**2).sum()**.5
            resp /= norm
            #rl_sampler.currentValue *= norm
        self.currentValue = resp

        self.updateXResp()
        self.updateNorm()

        rl_sampler.computeVarYTildeOpt()


class PhysioPerfResponseSampler(ResponseSampler, xmlio.XmlInitable):

    def __init__(self, smooth_order=2, zero_constraint=True, duration=25.,
                 normalise=1., val_ini=None, do_sampling=True,
                 use_true_value=False, diff_res=True,
                 prior_type='physio_stochastic_regularized'):
        """
        *diff_res*: if True then residuals (ytilde values) are differenced
        so that sampling is the same as for BRF.
        It avoids bad tail estimation, because of bad condionning of WtXtXW ?

        *prior_type*:
            - 'physio_stochastic_regularized'
            - 'physio_stochastic_not_regularized'
            - 'physio_deterministic'
            - 'basic_regularized'

        """
        available_priors = ['physio_stochastic_regularized',
                              'physio_stochastic_not_regularized',
                              'physio_deterministic',
                              'basic_regularized']
        if prior_type not in available_priors:
            raise Exception('Wrong prior type %s. Available choices: %s'\
                            %(prior_type, available_priors))
        xmlio.XmlInitable.__init__(self)
        self.diff_res = diff_res
        ResponseSampler.__init__(self, 'prf', 'prl', 'prf_var', smooth_order,
                                 zero_constraint, duration, normalise, val_ini,
                                 do_sampling, use_true_value)
        self.prior_type = prior_type

    def get_stackX(self):
        return self.dataInput.stackWX

    def get_mat_X(self):
        if not self.diff_res:
            return self.dataInput.WX
        else:
            return self.dataInput.varX

    def get_mat_XtX(self):
        if not self.diff_res:
            return self.dataInput.WXtWX
        else:
            return self.dataInput.matXtX

    def samplingWarmUp(self, variables):

        from pyhrf.sandbox.physio import PHY_PARAMS_FRISTON00 as phy_params
        from pyhrf.sandbox.physio import linear_rf_operator

        hrf_length = self.currentValue.shape[0]

        self.omega_operator =  linear_rf_operator(hrf_length, phy_params, self.dt,
                                    calculating_brf=False)

        if 'physio' in self.prior_type:
            self.omega_value = self.omega_operator
        else: # basic
            self.omega_value = np.zeros_like(self.omega_operator)

        if 'not_regularized' in self.prior_type:
            self.varR = np.eye(self.varR.shape[0])

    def computeYTilde(self):
        """ y - \sum aXh - Pl - wa """

        brf_sampler = self.get_variable('brf')
        brl_sampler = self.get_variable('brl')
        sumaXh = brl_sampler.sumBXResp

        drift_sampler = self.get_variable('drift_coeff')
        Pl = drift_sampler.Pl
        perf_baseline_sampler = self.get_variable('perf_baseline')
        wa = perf_baseline_sampler.wa
        y = self.dataInput.varMBY

        res = y - sumaXh - Pl - wa

        if 0 and self.dataInput.simulData is not None: #hack
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
              brl_sampler.useTrueValue and brf_sampler.useTrueValue:
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])

            if not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                assert_almost_equal(Pl, sd['drift'])
            if not perf_baseline_sampler.sampleFlag and \
              perf_baseline_sampler.useTrueValue:
                assert_almost_equal(wa, np.dot(self.dataInput.W,
                                               sd['perf_baseline']))


            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
              brl_sampler.useTrueValue and brf_sampler.useTrueValue and \
              not drift_sampler.sampleFlag and drift_sampler.useTrueValue and\
              not perf_baseline_sampler.sampleFlag and \
              perf_baseline_sampler.useTrueValue:

                perf = np.dot(self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(res, perf + sd['noise'])



        if not self.diff_res:
            return res
        else:
            return np.dot(self.dataInput.W, res)


    def sampleNextInternal(self, variables):
        """
        Sample PRF with physio prior

        changes to mean: add a factor of Omega h Sigma_g^-1 v_g^-1
        """

        rl_sampler = self.get_variable(self.response_level_name)
        rl = rl_sampler.currentValue
        rlrl = rl_sampler.rr
        smpl_brf = self.get_variable('brf')
        omega = self.omega_value
        brf = smpl_brf.currentValue

        if 'deterministic' in self.prior_type:
            resp = np.dot(omega,brf)
        else: #stochastic
            noise_var = self.get_variable('noise_var').currentValue

            mx = self.get_mat_X()
            mxtx = self.get_mat_XtX()

            self.ytilde[:] = self.computeYTilde()

            StS, StY = compute_StS_StY(rl, noise_var, mx, mxtx, self.ytilde, rlrl,
                                       self.yBj, self.BjBk_vb)

            v_resp = self.get_variable(self.var_name).currentValue

            sigma_g_inv = self.varR


            new_factor = np.dot(sigma_g_inv, np.dot(omega,brf))/v_resp

            varInvSigma = StS + self.nbVoxels * self.varR / v_resp
            mean_h = np.linalg.solve(varInvSigma, StY+new_factor)
            resp = np.random.multivariate_normal(mean_h,
                                                 np.linalg.inv(varInvSigma))

        if self.normalise:
            norm = (resp**2).sum()**.5
            resp /= norm
            #rl_sampler.currentValue *= norm
        self.currentValue = resp

        self.updateXResp()
        self.updateNorm()

        rl_sampler.computeVarYTildeOpt()



class ResponseVarianceSampler(GibbsSamplerVariable):

    def __init__(self, name, response_name, val_ini=None, do_sampling=True,
                 use_true_value=False):
        self.response_name = response_name
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      value_label='Var ' + self.response_name)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            if dataInput.simulData[0].has_key(self.name):
                self.trueValue = np.array([dataInput.simulData[0][self.name]])

    def checkAndSetInitValue(self, v):
        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %ResponseVarianceSampler)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

    def sampleNextInternal(self, v):
        """
        Sample variance of BRF or PRF

        TODO: change code below --> no changes necessary so far
        """
        resp_sampler = self.get_variable(self.response_name)
        R = resp_sampler.varR
        resp = resp_sampler.currentValue

        alpha = (len(resp) * self.nbVoxels - 1)/2.  
        #HACK! self.nbVoxels = size(parcel)  --> remove maybe?
        beta = np.dot(np.dot(resp.T, R), resp)/2.

        self.currentValue[0] = 1/np.random.gamma(alpha, 1/beta)

        
class PhysioBOLDResponseVarianceSampler(ResponseVarianceSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([0.001]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)

        ResponseVarianceSampler.__init__(self, 'brf_var', 'brf',
                                         val_ini, do_sampling, use_true_value)

                                         
class PhysioPerfResponseVarianceSampler(ResponseVarianceSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([0.001]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseVarianceSampler.__init__(self, 'prf_var', 'prf',
                                         val_ini, do_sampling, use_true_value)


class NoiseVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'noise_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'],
                                      value_label='PM Noise Var')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny

        # Do some allocations :

        if self.dataInput.simulData is not None:
            assert isinstance(self.dataInput.simulData[0], dict)
            sd = dataInput.simulData[0]
            if sd.has_key('noise'):
                # self.trueValue = np.array([sd['v_gnoise']])
                # pyhrf.verbose(3, 'True noise variance = %1.3f' \
                #               %self.trueValue)

                self.trueValue = sd['noise'].var(0)

        if self.trueValue is not None and self.trueValue.size==1:
            self.trueValue = self.trueValue * np.ones(self.nbVoxels)


    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                pyhrf.verbose(3, 'Use true noise variance value ...')
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('True noise variance have to be used but '\
                                'none defined.')

        if self.currentValue is None :
            self.currentValue = 0.1 * self.dataInput.varData

    def compute_y_tilde(self):
        pyhrf.verbose(4, 'NoiseVarianceSampler.compute_y_tilde ...')

        sumaXh = self.get_variable('brl').sumBXResp
        sumcXg = self.get_variable('prl').sumBXResp
        Pl = self.get_variable('drift_coeff').Pl
        wa = self.get_variable('perf_baseline').wa

        y = self.dataInput.varMBY

        return y - sumaXh - sumcXg - Pl - wa

    def sampleNextInternal(self, variables):
        y_tilde = self.compute_y_tilde()
        
        alpha = (self.ny - 1.)/2.
        beta = (y_tilde * y_tilde).sum(0)/2.

        gammaSamples = np.random.gamma(alpha, 1./beta)
        self.currentValue = 1.0/gammaSamples


class DriftVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=np.array([1.0]), do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self,'drift_var', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      sampleFlag=do_sampling)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, list): #multisession
                self.trueValue = np.array([dataInput.simulData[0]['drift_var']])
            elif isinstance(dataInput.simulData, dict): #one session (old case)
                self.trueValue = np.array([dataInput.simulData['drift_var']])

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)


    def sampleNextInternal(self, variables):

        smpldrift = self.get_variable('drift_coeff')
#        print 'shape dimDrift...', smpldrift.dimDrift
#        print 'norm Drift', smpldrift.norm
        alpha = .5 * (smpldrift.dimDrift * self.nbVoxels - 1)
        beta = 2.0 / smpldrift.norm
        pyhrf.verbose(4, 'eta ~ Ga(%1.3f,%1.3f)'%(alpha,beta))
        self.currentValue[0] = 1.0/np.random.gamma(alpha,beta)

        if 0:
            beta = 1/beta
            if self.trueValue is not None:
                pyhrf.verbose(4, 'true var drift : %f' %self.trueValue)
            pyhrf.verbose(4, 'm_theo=%f, v_theo=%f' \
                          %(beta/(alpha-1), beta**2/((alpha-1)**2 * (alpha-2))))
            samples = 1.0/np.random.gamma(alpha,1/beta,1000)
            pyhrf.verbose(4, 'm_empir=%f, v_empir=%f' \
                          %(samples.mean(), samples.var()))

            pyhrf.verbose(4, 'current sample: %f' %self.currentValue[0])


class DriftCoeffSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self,'drift_coeff', valIni=val_ini,
                                      useTrueValue=use_true_value,
                                      axes_names=['lfd_order','voxel'],
                                      sampleFlag=do_sampling)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.P = self.dataInput.lfdMat[0]
        self.dimDrift = self.P.shape[1]

        self.y_bar = np.zeros((self.ny, self.nbVoxels), dtype=np.float64)
        self.ones_Q_J = np.ones((self.dimDrift, self.nbVoxels))

        if dataInput.simulData is not None:
            if isinstance(dataInput.simulData, list):   # multisession
                self.trueValue = dataInput.simulData[0]['drift_coeffs']
            elif isinstance(dataInput.simulData, dict): # one session (old case)
                self.trueValue = dataInput.simulData['drift_coeffs']

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None:

            #self.currentValue = np.random.randn(self.dimDrift,
            #                                    self.nbVoxels).astype(np.float64)
            self.currentValue = np.dot(self.P.T, self.dataInput.varMBY)
            # Projection of data y on P.T

        self.Pl = np.dot(self.P, self.currentValue)
        self.updateNorm()

    def samplingWarmUp(self, v):
        self.ytilde = np.zeros((self.ny,self.nbVoxels), dtype=float)
        self.track_sampled_quantity(self.ytilde, self.name + '_ytilde',
                                    axes_names=['time', 'voxel'])


    def compute_y_tilde(self):

        sumaXh = self.get_variable('brl').sumBXResp
        sumcXg = self.get_variable('prl').sumBXResp
        wa = self.get_variable('perf_baseline').wa

        y = self.dataInput.varMBY

        return y - sumaXh - sumcXg - wa


    def sampleNextInternal(self, variables):

        self.ytilde[:] = self.compute_y_tilde()

        v_l = self.get_variable('drift_var').currentValue
        v_b = self.get_variable('noise_var').currentValue

        pyhrf.verbose(5, 'Noise vars :' )
        pyhrf.verbose.printNdarray(5, v_b)
        
        PtP = np.dot(self.P.transpose(),self.P)
        I_F = np.eye((self.P.shape[1]))
        assert_almost_equal( PtP, I_F)
        
        for i in xrange(self.nbVoxels):
            
            v_lj = v_b[i] * v_l / (v_b[i] + v_l)
            S_lj = v_b[i] * v_l / (v_b[i] + v_l) * I_F
            S_lj2 = v_b[i] * v_l * np.linalg.inv(v_b[i] * I_F + v_l * PtP) 
            mu_lj = v_lj * np.dot(self.P.transpose(), self.ytilde[:,i]) / v_b[i]
            mu_lj1 = np.dot(S_lj, np.dot(self.P.transpose(), self.ytilde[:,i])) / v_b[i]
            mu_lj2 = np.dot(S_lj2, np.dot(self.P.transpose(), self.ytilde[:,i])) / v_b[i]
            #pyhrf.verbose(5, 'ivox=%d, v_lj=%f, std_lj=%f mu_lj=%s' \
            #              %(i,S_lj,S_lj**.5, str(mu_lj)))
            self.currentValue[:,i] = (np.random.randn(self.dimDrift) * \
                                      v_lj**.5) + mu_lj
            #print 'res1 = ',(np.random.randn(self.dimDrift) * v_lj**.5) + mu_lj
            #print 'res2 = ',np.random.multivariate_normal(mu_lj1, S_lj)
            #print 'res3 = ',np.random.multivariate_normal(mu_lj2, S_lj2)
            #self.currentValue[:,i] = np.random.multivariate_normal(mu_lj1, S_lj)

            pyhrf.verbose(5, 'v_l : %f' %v_l)

        pyhrf.verbose(5, 'drift params :')
        pyhrf.verbose.printNdarray(5, self.currentValue)

        if 1: # some tests
            inv_vars_l = (1/v_b + 1/v_l) * self.ones_Q_J
            mu_l = 1/inv_vars_l * np.dot(self.P.transpose(), self.ytilde)

            pyhrf.verbose(5, 'vars_l :')
            pyhrf.verbose.printNdarray(5, 1/inv_vars_l)

            pyhrf.verbose(5, 'mu_l :')
            pyhrf.verbose.printNdarray(5, mu_l)

            cur_val = np.random.normal(mu_l, 1/inv_vars_l)

            pyhrf.verbose(5, 'drift params (alt) :')
            pyhrf.verbose.printNdarray(5, cur_val)

            #assert np.allclose(cur_val, self.currentValue)

        self.updateNorm()
        self.Pl = np.dot(self.P, self.currentValue)
        
    def updateNorm(self):

        self.norm = (self.currentValue * self.currentValue).sum()

        if self.trueValue is not None:
            pyhrf.verbose(4, 'cur drift norm: %f' %self.norm)
            pyhrf.verbose(4, 'true drift norm:' %(self.trueValue * \
                                                  self.trueValue).sum())

    def getOutputs(self):
        outputs = GibbsSamplerVariable.getOutputs(self)
        drift_signal = np.dot(self.P, self.finalValue)
        print drift_signal
        an = ['time','voxel']
        a = xndarray(drift_signal, axes_names=an, value_label='Delta ASL')
        if self.trueValue is not None:
            ta = xndarray(np.dot(self.P, self.trueValue),
                          axes_names=an, value_label='Delta ASL')
            a = stack_cuboids([a, ta], 'type', ['estim', 'true'])

        outputs['drift_signal_pm'] = a
        return outputs



class ResponseLevelSampler(GibbsSamplerVariable):


    def __init__(self, name, response_name, mixture_name,
                 val_ini=None, do_sampling=True,
                 use_true_value=False):

        self.response_name = response_name
        self.mixture_name = mixture_name
        an = ['condition', 'voxel']
        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      value_label='amplitude')

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = dataInput.simulData[0].get(self.name + 's', None)

        # Precalculations and allocations :
        self.varYtilde = np.zeros((self.ny, self.nbVoxels), dtype=np.float64)
        self.BXResp = np.empty((self.nbVoxels, self.ny,
                                self.nbConditions), dtype=float)
        self.sumBXResp = np.zeros((self.ny, self.nbVoxels), dtype=float)

        self.rr = np.zeros((self.nbConditions, self.nbConditions, self.nbVoxels),
                           dtype=float)


    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None :
            #rnd = np.random.rand(self.nbConditions, self.nbVoxels)
            #self.currentValue = (rnd.astype(np.float64) - .5 ) * 10
            self.currentValue = np.zeros((self.nbConditions,
                                          self.nbVoxels), dtype=np.float64)
            yptp = self.dataInput.varMBY.ptp(0)
            for j in xrange(self.nbConditions):
                self.currentValue[j,:] = yptp * .1

    def samplingWarmUp(self, variables):
        """
        """

        self.response_sampler = self.get_variable(self.response_name)
        self.mixture_sampler = self.get_variable(self.mixture_name)


        self.meanApost = np.zeros((self.nbConditions, self.nbVoxels), dtype=float)
        self.varApost = np.zeros((self.nbConditions, self.nbVoxels), dtype=float)

        self.labeled_vars = np.zeros((self.nbConditions, self.nbVoxels))
        self.labeled_means = np.zeros((self.nbConditions, self.nbVoxels))

        self.iteration = 0

        self.computeRR()



    def sampleNextInternal(self, variables):

        labels = self.get_variable('label').currentValue
        v_b = self.get_variable('noise_var').currentValue

        Xresp = self.response_sampler.varXResp

        gTg = np.diag(np.dot(Xresp.transpose(),Xresp))

        mixt_vars = self.mixture_sampler.get_current_vars()
        mixt_means = self.mixture_sampler.get_current_means()

        ytilde = self.computeVarYTildeOpt()

        for iclass in xrange(len(mixt_vars)):
            v = mixt_vars[iclass]
            m = mixt_means[iclass]
            for j in xrange(self.nbConditions):
                class_mask = np.where(labels[j]==iclass)
                self.labeled_vars[j,class_mask[0]] = v[j]
                self.labeled_means[j,class_mask[0]] = m[j]

        for j in xrange(self.nbConditions):
            Xresp_m = Xresp[:,j]
            ytilde_m = ytilde + (self.currentValue[np.newaxis,j,:] * \
                                 Xresp_m[:,np.newaxis])
            v_q_j = self.labeled_vars[j]
            m_q_j = self.labeled_means[j]
            self.varApost[j,:] = (v_b * v_q_j) / (gTg[j] * v_q_j + v_b)
            self.meanApost[j,:] = self.varApost[j,:] * \
                (np.dot(Xresp_m.T, ytilde_m)/v_b + m_q_j / v_q_j )

            rnd = np.random.randn(self.nbVoxels)
            self.currentValue[j,:] = rnd * self.varApost[j,:]**.5 + \
              self.meanApost[j,:]
            ytilde = self.computeVarYTildeOpt()

            #b()

        self.computeRR()

    def computeVarYTildeOpt(self):
        raise NotImplementedError()

    def computeRR(self):
        # aa[m,n,:] == aa[n,m,:] -> nb ops can be /2
        for j in xrange(self.nbConditions):
            for k in xrange(self.nbConditions):
                np.multiply(self.currentValue[j,:], self.currentValue[k,:],
                         self.rr[j,k,:])

class BOLDResponseLevelSampler(ResponseLevelSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseLevelSampler.__init__(self, 'brl', 'brf', 'bold_mixt_params',
                                      val_ini, do_sampling, use_true_value)


    def samplingWarmUp(self, v):
        ResponseLevelSampler.samplingWarmUp(self, v)
        # BOLD response sampler is in charge of initialising ytilde of prls:
        # -> update_perf=True
        self.computeVarYTildeOpt(update_perf=True)


    def computeVarYTildeOpt(self, update_perf=False):
        """
        if update_perf is True then also update sumcXg and prl.ytilde
        update_perf should only be used at init of variable values.
        """

        pyhrf.verbose(4, ' BOLDResp.computeVarYTildeOpt(update_perf=%s) ...' \
                      %str(update_perf))

        brf_sampler = self.get_variable('brf') 
        Xh = brf_sampler.varXResp
        sumaXh = self.sumBXResp

        prl_sampler = self.get_variable('prl')
        prls = prl_sampler.currentValue
        sumcXg = prl_sampler.sumBXResp
        prf_sampler = self.get_variable('prf')
        WXg = prf_sampler.varXResp

        compute_bold = 1
        if update_perf:
            compute_perf = 1
        else:
            compute_perf = 0
        asl_compute_y_tilde(Xh, WXg, self.currentValue, prls,
                            self.dataInput.varMBY, self.varYtilde,
                            sumaXh, sumcXg, compute_bold, compute_perf)
        if update_perf:
            ytilde_perf = prl_sampler.varYtilde
            asl_compute_y_tilde(Xh, WXg, self.currentValue, prls,
                                self.dataInput.varMBY, ytilde_perf,
                                sumaXh, sumcXg, 0, 0)

        if self.dataInput.simulData is not None: 
            # Some tests
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not self.sampleFlag and not brf_sampler.sampleFlag and\
              self.useTrueValue and brf_sampler.useTrueValue:
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])
            if not prl_sampler.sampleFlag and not prf_sampler.sampleFlag and\
              prl_sampler.useTrueValue and prf_sampler.useTrueValue:
                perf = np.dot(self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(sumcXg, perf)

        #print 'sumaXh = ', self.sumaXh
        #print 'varYtilde = ', self.varYtilde
        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde.shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde)

        Pl = self.get_variable('drift_coeff').Pl
        wa = self.get_variable('perf_baseline').wa

        return self.varYtilde - Pl - wa


    def getOutputs(self):

        outputs = GibbsSamplerVariable.getOutputs(self)

        axes_names = ['voxel']
        roi_lab_vol = np.zeros(self.nbVoxels, dtype=np.int32) + \
          self.dataInput.roiId
        outputs['roi_mapping'] = xndarray(roi_lab_vol, axes_names=axes_names,
                                        value_label='ROI')

        return outputs



class PerfResponseLevelSampler(ResponseLevelSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        ResponseLevelSampler.__init__(self, 'prl', 'prf', 'perf_mixt_params',
                                      val_ini, do_sampling, use_true_value)


    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is None:
                raise Exception('Needed a true value for %s init but '\
                                    'None defined' %self.name)
            else:
                self.currentValue = self.trueValue.astype(np.float64)

        if self.currentValue is None :
            #rnd = np.random.rand(self.nbConditions, self.nbVoxels)
            #self.currentValue = (rnd.astype(np.float64) - .5 ) * 10
            self.currentValue = np.zeros((self.nbConditions,
                                          self.nbVoxels), dtype=np.float64)

            perf_baseline = (self.dataInput.varMBY * \
                             self.dataInput.w[:,np.newaxis]).mean(0)

            for j in xrange(self.nbConditions):
                self.currentValue[j,:] = perf_baseline * .1


    def computeVarYTildeOpt(self):
        """
        """

        pyhrf.verbose(4, ' PerfRespLevel.computeVarYTildeOpt() ...')

        brf_sampler = self.get_variable('brf')
        Xh = brf_sampler.varXResp
        brl_sampler = self.get_variable('brl')
        sumaXh = brl_sampler.sumBXResp
        brls = brl_sampler.currentValue

        prf_sampler = self.get_variable('prf')
        WXg = prf_sampler.varXResp
        sumcXg = self.sumBXResp

        compute_bold = 0
        compute_perf = 1
        asl_compute_y_tilde(Xh, WXg, brls, self.currentValue,
                            self.dataInput.varMBY, self.varYtilde,
                            sumaXh, sumcXg, compute_bold, compute_perf)

        if self.dataInput.simulData is not None:
            # Some tests
            sd = self.dataInput.simulData[0]
            osf = int(sd['tr'] / sd['dt'])
            if not brl_sampler.sampleFlag and not brf_sampler.sampleFlag and\
              brl_sampler.useTrueValue and brf_sampler.useTrueValue:
                assert_almost_equal(sumaXh, sd['bold_stim_induced'][0:-1:osf])
            if not self.sampleFlag and not prf_sampler.sampleFlag and\
              self.useTrueValue and not prf_sampler.useTrueValue:
                perf = np.dot(self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                assert_almost_equal(sumcXg, perf)

        #print 'sumaXh = ', self.sumaXh
        #print 'varYtilde = ', self.varYtilde
        pyhrf.verbose(5,'varYtilde %s' %str(self.varYtilde.shape))
        pyhrf.verbose.printNdarray(5, self.varYtilde)

        if np.isnan(self.varYtilde).any():
            raise Exception('Nan values in ytilde of prf')

        Pl = self.get_variable('drift_coeff').Pl
        wa = self.get_variable('perf_baseline').wa

        return self.varYtilde - Pl - wa





class LabelSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    L_CI = 0
    L_CA = 1

    CLASSES = np.array([L_CI, L_CA],dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)

        an = ['condition', 'voxel']
        GibbsSamplerVariable.__init__(self, 'label', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an)

        self.nbClasses = len(self.CLASSES)


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        self.cardClass = np.zeros((self.nbClasses, self.nbConditions), dtype=int)
        self.voxIdx = [range(self.nbConditions) for c in xrange(self.nbClasses)]



        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = dataInput.simulData[0]['labels'].astype(np.int32)


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

            self.currentValue = np.zeros((self.nbConditions, self.nbVoxels),
                                         dtype=np.int32)

            for j in xrange(self.nbConditions):
                self.currentValue[j,:] = np.random.binomial(1, .9, self.nbVoxels)

        self.beta = np.zeros((self.nbConditions), dtype=np.float64) + .7

        self.countLabels()

    def countLabels(self):
        pyhrf.verbose(3, 'LabelSampler.countLabels ...')
        labs = self.currentValue
        for j in xrange(self.nbConditions):
            for c in xrange(self.nbClasses):
                self.voxIdx[c][j] = np.where(labs[j,:]==self.CLASSES[c])[0]
                self.cardClass[c,j] = len(self.voxIdx[c][j])
                pyhrf.verbose(5, 'Nb vox in C%d for cond %d : %d' \
                                  %(c,j,self.cardClass[c,j]))

            if self.cardClass[:,j].sum() != self.nbVoxels:
                raise Exception('cardClass[cond=%d]=%d != nbVox=%d' \
                                %(j,self.cardClass[:,j].sum(), self.nbVoxels))

    def samplingWarmUp(self, v):
        self.iteration = 0
        self.current_ext_field = np.zeros((self.nbClasses, self.nbConditions,
                                           self.nbVoxels), dtype=np.float64)

    def compute_ext_field(self):
        bold_mixtp_sampler = self.get_variable('bold_mixt_params')
        asl_mixtp_sampler = self.get_variable('perf_mixt_params')

        v = bold_mixtp_sampler.get_current_vars()
        rho = asl_mixtp_sampler.get_current_vars()

        mu = bold_mixtp_sampler.get_current_means()
        eta = asl_mixtp_sampler.get_current_means()

        a = self.get_variable('brl').currentValue
        c = self.get_variable('prl').currentValue

        for k in xrange(self.nbClasses):
            for j in xrange(self.nbConditions):
                # WARNING!! log base changed from 2 to e
                e = .5 * (-np.log(v[k,j] * rho[k,j]) - \
                          (a[j,:] - mu[k,j])**2 / v[k,j]  - \
                          (c[j,:] - eta[k,j])**2 / rho[k,j])
                self.current_ext_field[k,j,:] = e

    def sampleNextInternal(self, v):

        neighbours = self.dataInput.neighboursIndexes

        beta = self.beta

        voxOrder = np.random.permutation(self.nbVoxels)

        self.compute_ext_field()

        rnd = np.random.rand(*self.currentValue.shape).astype(np.float64)

        sample_potts(voxOrder.astype(np.int32), neighbours.astype(np.int32),
                     self.current_ext_field, beta, rnd, self.currentValue,
                     self.iteration)

        self.countLabels()
        self.iteration += 1

class MixtureParamsSampler(GibbsSamplerVariable):

    I_MEAN_CA = 0
    I_VAR_CA = 1
    I_VAR_CI = 2
    NB_PARAMS = 3
    PARAMS_NAMES = ['Mean_Activ', 'Var_Activ', 'Var_Inactiv']

    L_CA = LabelSampler.L_CA
    L_CI = LabelSampler.L_CI

    def __init__(self, name, response_level_name,
                 val_ini=None, do_sampling=True,
                 use_true_value=False):

        self.response_level_name = response_level_name
        an = ['component','condition']
        ad = {'component' : self.PARAMS_NAMES}

        GibbsSamplerVariable.__init__(self, name, valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=an,
                                      axes_domains=ad)

    def get_true_values_from_simulation_dict(self):
        raise NotImplementedError()

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            cdefs = self.dataInput.simulData[0]['condition_defs']
            if hasattr(cdefs[0], 'bold_m_act'):
                tmca,tvca,tvci = self.get_true_values_from_simulation_cdefs(cdefs)
                self.trueValue = np.zeros((self.NB_PARAMS, self.nbConditions),
                                          dtype=float)
                self.trueValue[self.I_MEAN_CA] = tmca
                self.trueValue[self.I_VAR_CA] = tvca
                self.trueValue[self.I_VAR_CI] = tvci

        self.rlCI = range(self.nbConditions)
        self.rlCA = range(self.nbConditions)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()[:,:self.nbConditions]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            nc = self.nbConditions
            self.currentValue = np.zeros((self.NB_PARAMS, nc), dtype=float)
            # self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30.
            # self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.
            # self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.

            y = self.dataInput.varMBY
            self.currentValue[self.I_MEAN_CA,:] = y.ptp(0).mean() * .1
            self.currentValue[self.I_VAR_CA,:] = y.var(0).mean() * .1
            self.currentValue[self.I_VAR_CI,:] = y.var(0).mean() * .05

    def get_current_vars(self):
        return np.array([self.currentValue[self.I_VAR_CI],
                         self.currentValue[self.I_VAR_CA]])

    def get_current_means(self):
        return np.array([np.zeros(self.nbConditions),
                         self.currentValue[self.I_MEAN_CA]])


    def computeWithJeffreyPriors(self, j, cardCIj, cardCAj):

        #print 'sample hyper parameters with improper Jeffrey\'s priors ...'
        if pyhrf.verbose.verbosity >= 3:
            print 'cond %d - card CI = %d' %(j,cardCIj)
            print 'cond %d - card CA = %d' %(j,cardCAj)
            print 'cond %d - cur mean CA = %f' %(j,self.currentValue[self.I_MEAN_CA,j])
            if cardCAj > 0:
                print 'cond %d - rl CA: %f(v%f)[%f,%f]' %(j,self.rlCA[j].mean(),
                                                          self.rlCA[j].var(),
                                                          self.rlCA[j].min(),
                                                          self.rlCA[j].max())
            if cardCIj > 0:
                print 'cond %d - rl CI: %f(v%f)[%f,%f]' %(j,self.rlCI[j].mean(),
                                                          self.rlCI[j].var(),
                                                          self.rlCI[j].min(),
                                                          self.rlCI[j].max())

        if cardCIj > 1:
            nu0j = np.dot(self.rlCI[j], self.rlCI[j])
            varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj + 1) - 1, 2. / nu0j)
            #varCIj = 1.0 / np.random.gamma(0.5 * (cardCIj - 1), 2. / nu0j)
        else :
            varCIj = 1.0 / np.random.gamma(0.5, 0.2)

        #HACK
        #varCIj = .5

        if cardCAj > 1:
            rlC1Centered = self.rlCA[j] - self.currentValue[self.I_MEAN_CA,j]
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
            eta1j = np.mean(self.rlCA[j])
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
                meanCAj = np.random.normal(self.rlCA[j], varCAj**0.5)

        if pyhrf.verbose.verbosity >= 3:
            print 'Sampled components - cond', j
            print 'var CI =', varCIj
            print 'mean CA =', meanCAj, 'var CA =', varCAj


            #b()

        return varCIj, meanCAj, varCAj

    def sampleNextInternal(self, variables):

        rl_sampler = self.get_variable(self.response_level_name)
        label_sampler = self.get_variable('label')

        cardCA = label_sampler.cardClass[self.L_CA,:]
        cardCI = label_sampler.cardClass[self.L_CI,:]

        for j in xrange(self.nbConditions):
        #for j in np.random.permutation(self.nbConditions):
            vICI = label_sampler.voxIdx[self.L_CI][j]
            vICA = label_sampler.voxIdx[self.L_CA][j]
            self.rlCI[j] = rl_sampler.currentValue[j, vICI]
            self.rlCA[j] = rl_sampler.currentValue[j, vICA]

            # if self.hyperPriorFlag:
            #     varCIj,meanCAj,varCAj = self.computeWithProperPriors(j,
            #                                                          cardCI[j],
            #                                                          cardCA[j])
            # else:
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

class PerfMixtureSampler(MixtureParamsSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        MixtureParamsSampler.__init__(self, 'perf_mixt_params', 'prl',
                                      val_ini, do_sampling, use_true_value)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()[:,:self.nbConditions]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            nc = self.nbConditions
            self.currentValue = np.zeros((self.NB_PARAMS, nc), dtype=float)
            # self.currentValue[self.I_MEAN_CA] = np.zeros(nc) + 30.
            # self.currentValue[self.I_VAR_CA] = np.zeros(nc) + 1.
            # self.currentValue[self.I_VAR_CI] = np.zeros(nc) + 1.

            perf_baseline = (self.dataInput.varMBY * \
                             self.dataInput.w[:,np.newaxis]).mean(0)

            self.currentValue[self.I_MEAN_CA,:] = perf_baseline.mean() * .1
            self.currentValue[self.I_VAR_CA,:] = perf_baseline.var() * .1
            self.currentValue[self.I_VAR_CI,:] = perf_baseline.var() * .05


    def get_true_values_from_simulation_cdefs(self, cdefs):
        return np.array([c.perf_m_act for c in cdefs]), \
          np.array([c.perf_v_act for c in cdefs]), \
          np.array([c.perf_v_inact for c in cdefs])


class BOLDMixtureSampler(MixtureParamsSampler, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        MixtureParamsSampler.__init__(self, 'bold_mixt_params', 'brl',
                                      val_ini, do_sampling, use_true_value)

    def get_true_values_from_simulation_cdefs(self, cdefs):
        return np.array([c.bold_m_act for c in cdefs]), \
          np.array([c.bold_v_act for c in cdefs]), \
          np.array([c.bold_v_inact for c in cdefs])


class PerfBaselineSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):

        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'perf_baseline', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value,
                                      axes_names=['voxel'])

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels
        self.ny = self.dataInput.ny
        self.w = self.dataInput.w
        self.W = self.dataInput.W

        if dataInput.simulData is not None:
            assert isinstance(dataInput.simulData[0], dict)
            self.trueValue = self.dataInput.simulData[0][self.name][0,:]

        if self.trueValue is not None and np.isscalar(self.trueValue):
            self.trueValue = self.trueValue * np.ones(self.nbVoxels)

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue[:]
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            #self.currentValue = np.zeros(self.nbVoxels) + 1.
            self.currentValue = (self.dataInput.varMBY * \
              self.dataInput.w[:,np.newaxis]).mean(0)

        self.wa = self.compute_wa()


    def compute_wa(self, a=None):
        if a is None:
            a = self.currentValue
        return self.w[:,np.newaxis] * a[np.newaxis,:]


    def compute_residuals(self):

        brf_sampler = self.get_variable('brf')
        prf_sampler = self.get_variable('prf')

        prl_sampler = self.get_variable('prl')
        brl_sampler = self.get_variable('brl')

        drift_sampler = self.get_variable('drift_coeff')

        sumcXg = prl_sampler.sumBXResp
        sumaXh = brl_sampler.sumBXResp
        Pl = drift_sampler.Pl

        y = self.dataInput.varMBY

        res = y - sumcXg - sumaXh - Pl

        if self.dataInput.simulData is not None:
            # only for debugging when using artificial data
            if not brf_sampler.sampleFlag and brf_sampler.useTrueValue and\
              not brl_sampler.sampleFlag and brl_sampler.useTrueValue and \
              not prf_sampler.sampleFlag and prf_sampler.useTrueValue and \
              not brf_sampler.sampleFlag and brf_sampler.useTrueValue and \
              not drift_sampler.sampleFlag and drift_sampler.useTrueValue:
                sd = self.dataInput.simulData[0]
                osf = int(sd['tr'] / sd['dt'])
                perf = np.dot(self.dataInput.W, sd['perf_stim_induced'][0:-1:osf])
                true_res = sd['bold'] - sd['bold_stim_induced'][0:-1:osf] - \
                  perf - sd['drift']
                true_res = sd['noise'] + np.dot(self.dataInput.W,
                                                sd['perf_baseline'])
                assert_almost_equal(res, true_res)

        return res

    def sampleNextInternal(self, v):

        v_alpha = self.get_variable('perf_baseline_var').currentValue
        w = np.diag(self.dataInput.W)

        residuals = self.compute_residuals()
        v_b = self.get_variable('noise_var').currentValue

        for i in xrange(self.nbVoxels):
            #m_apost = ( np.dot(w.T, residuals[:,i]) ) /  \
            #          ( self.ny + v_b[i] / v_alpha)
            m_apost = ( np.dot(w.T, residuals[:,i]) * v_alpha) /  \
                      ( self.ny * v_alpha + v_b[i])
            v_apost = ( v_alpha * v_b[i] ) / ( self.ny * v_alpha + v_b[i])

            a = np.random.randn() * v_apost**.5 + m_apost
            self.currentValue[i] = a
            self.wa[:,i] = self.w * a

class PerfBaselineVarianceSampler(GibbsSamplerVariable, xmlio.XmlInitable):

    def __init__(self, val_ini=None, do_sampling=True,
                 use_true_value=False):
        xmlio.XmlInitable.__init__(self)
        GibbsSamplerVariable.__init__(self, 'perf_baseline_var', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)

    def linkToData(self, dataInput):
        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.nbVoxels = self.dataInput.nbVoxels

        if dataInput.simulData is not None:
            sd = dataInput.simulData
            assert isinstance(sd[0], dict)
            if sd[0].has_key(self.name):
                self.trueValue = np.array([sd[0][self.name]])


    def checkAndSetInitValue(self, variables):

        if self.useTrueValue:
            if self.trueValue is not None:
                self.currentValue = self.trueValue.copy()
            else:
                raise Exception('Needed a true value but none defined')

        if self.currentValue is None:
            self.currentValue = np.array([(self.dataInput.varMBY * \
              self.dataInput.w[:,np.newaxis]).mean(0).var()])


    def sampleNextInternal(self, v):

        alpha = self.get_variable('perf_baseline').currentValue

        a = (self.nbVoxels - 1) / 2.
        b = (alpha**2).sum() / 2.

        self.currentValue[0] = 1 / np.random.gamma(a, 1/b)


class WN_BiG_ASLSamplerInput(WN_BiG_Drift_BOLDSamplerInput):

    def makePrecalculations(self):
        WN_BiG_Drift_BOLDSamplerInput.makePrecalculations(self)

        self.W = build_ctrl_tag_matrix((self.ny,))
        self.w = np.diag(self.W)

        self.WX = np.zeros_like(self.varX)
        self.WXtWX = np.zeros_like(self.matXtX)
        self.XtWX = np.zeros_like(self.matXtX)
        self.stackWX = np.zeros_like(self.stackX)

        for j in xrange(self.nbConditions):
            #print 'self.varX :', self.varX[j,:,:].transpose().shape
            #print 'self.delta :', self.delta.shape
            self.WX[j,:,:] = np.dot(self.W, self.varX[j,:,:])
            self.stackWX[self.ny*j:self.ny*(j+1), :] = self.WX[j,:,:]
            for k in xrange(self.nbConditions):
                self.WXtWX[j,k,:,:] = np.dot(self.WX[j,:,:].transpose(),
                                             self.WX[k,:,:] )
                self.XtWX[j,k,:,:] = np.dot(self.varX[j,:,:].transpose(),
                                             self.WX[k,:,:] )


    def cleanPrecalculations(self):
        WN_BiG_Drift_BOLDSamplerInput.cleanPrecalculations(self)
        del self.WXtWX
        #del self.XtWX


class ASLPhysioSampler(xmlio.XmlInitable, GibbsSampler):

    inputClass = WN_BiG_ASLSamplerInput

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        default_nb_its = 3
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        default_nb_its = 3000
        parametersToShow = ['nb_its', 'response_levels', 'hrf', 'hrf_var']

    def __init__(self, nb_iterations=default_nb_its,
                 obs_hist_pace=-1., glob_obs_hist_pace=-1,
                 smpl_hist_pace=-1., burnin=.3,
                 callback=GSDefaultCallbackHandler(),
                 bold_response_levels=BOLDResponseLevelSampler(),
                 perf_response_levels=PerfResponseLevelSampler(),
                 labels=LabelSampler(), noise_var=NoiseVarianceSampler(),
                 brf=PhysioBOLDResponseSampler(),
                 brf_var=PhysioBOLDResponseSampler(),
                 prf=PhysioPerfResponseSampler(),
                 prf_var=PhysioPerfResponseSampler(),
                 bold_mixt_params=BOLDMixtureSampler(),
                 perf_mixt_params=PerfMixtureSampler(),
                 drift=DriftCoeffSampler(), drift_var=DriftVarianceSampler(), 
                 perf_baseline=PerfBaselineSampler(),               
                 perf_baseline_var=PerfBaselineVarianceSampler(),   
                 check_final_value=None):

        variables = [noise_var, brf, brf_var, prf, prf_var,
                     drift_var, drift, perf_response_levels,
                     bold_response_levels, perf_baseline, perf_baseline_var,
                     bold_mixt_params, perf_mixt_params, labels]

        nbIt = nb_iterations
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

        callbackObj = GSDefaultCallbackHandler()
        self.cmp_ftval = False #TODO: remove this, check final value has been
                               # factored in GibbsSamplerVariable
        GibbsSampler.__init__(self, variables, nbIt, smplHistPace,
                              obsHistPace, nbSweeps,
                              callbackObj,
                              globalObsHistoryPace=globalObsHistPace,
                              check_ftval=check_ftval)

    def finalizeSampling(self):
        if self.cmp_ftval:

            msg = []
            for v in self.variables:

                if v.trueValue is None:
                    print 'Warning; no true val for %s' %v.name
                else:
                    fv = v.finalValue
                    tv = v.trueValue
                    # tol = .7
                    # if v.name == 'drift_coeff':
                    #     delta = np.abs(np.dot(v.P,
                    #                           v.finalValue - \
                    #                           v.trueValue)).mean()
                    #     crit = detla > tol
                    # else:
                    #     delta = np.abs(v.finalValue - v.trueValue).mean()
                    #     crit = delta > tol
                    tol = .1
                    if self.dataInput.nbVoxels < 10:
                        if 'var' in v.name:
                            tol = 1.

                    if v.name == 'drift_coeff':
                        fv = np.dot(v.P, v.finalValue)
                        tv = np.dot(v.P, v.trueValue)
                        delta = np.abs( (fv - tv) / np.maximum(tv,fv))
                    elif v.name == 'prf' or v.name == 'brf':
                        delta = (((v.finalValue - v.trueValue)**2).sum() / \
                                 (v.trueValue**2).sum())**.5
                        tol = 0.05
                    elif v.name == 'label':
                        delta = (v.finalValue!=v.trueValue).sum()*1. / v.nbVoxels
                    else:
                        #delta = (((v.finalValue - v.trueValue)**2).sum() / \
                        #         (v.trueValue**2).sum())**.5
                        delta = np.abs((v.finalValue-v.trueValue) / \
                                       np.maximum(v.trueValue,v.finalValue))
                    crit = (delta > tol).any()

                    if  crit:
                        m = "Final value of %s is not close to " \
                            "true value (mean delta=%f).\n" \
                            " Final value:\n %s\n True value:\n %s\n" \
                            %(v.name, delta.mean(), str(fv), str(tv))
                        msg.append(m)
                        #raise Exception(m)

            if len(msg) > 0:
                if 0:
                    raise Exception("\n".join(msg))
                else:
                    print "\n".join(msg)

    def computeFit(self):
        brf_sampler = self.get_variable('brf')
        prf_sampler = self.get_variable('prf')

        brl_sampler = self.get_variable('brl')
        prl_sampler = self.get_variable('prl')

        drift_sampler = self.get_variable('drift_coeff')
        perf_baseline_sampler = self.get_variable('perf_baseline')


        brf = brf_sampler.finalValue
        if brf is None:
            brf = brf_sampler.currentValue
        elif brf_sampler.zc:
            brf = brf[1:-1]
        vXh = brf_sampler.calcXResp(brf) # base convolution


        prf = prf_sampler.finalValue
        if prf is None:
            prf = prf_sampler.currentValue
        elif prf_sampler.zc:
            prf = prf[1:-1]
        vXg = prf_sampler.calcXResp(prf) # base convolution

        brl = brl_sampler.finalValue
        if brl is None:
            brl = brl_sampler.currentValue

        prl = prl_sampler.finalValue
        if prl is None:
            prl = prl_sampler.currentValue

        l = drift_sampler.finalValue
        p = drift_sampler.P
        if l is None:
            l = drift_sampler.currentValue

        perf_baseline = perf_baseline_sampler.finalValue
        if perf_baseline is None:
            perf_baseline = perf_baseline_sampler.currentValue
        wa = perf_baseline_sampler.compute_wa(perf_baseline)

        fit = np.dot(vXh, brl) + np.dot(vXg, prl) + np.dot(p, l) + wa

        return fit


    def getGlobalOutputs(self):
        outputs = GibbsSampler.getGlobalOutputs(self)

        bf = outputs.pop('bold_fit', None)
        if bf is not None:
            cdict = bf.split('stype')
            signal = cdict['bold']
            fit = cdict['fit']

            # Grab fitted components
            brf_sampler = self.get_variable('brf')
            prf_sampler = self.get_variable('prf')

            brl_sampler = self.get_variable('brl')
            prl_sampler = self.get_variable('prl')

            drift_sampler = self.get_variable('drift_coeff')
            perf_baseline_sampler = self.get_variable('perf_baseline')


            brf = brf_sampler.finalValue
            if brf_sampler.zc:
                brf = brf[1:-1]
            vXh = brf_sampler.calcXResp(brf) # base convolution
            #demod_stackX = brf_sampler.get_stackX()

            prf = prf_sampler.finalValue
            if prf_sampler.zc:
                prf = prf[1:-1]

            # base convolution:
            vXg = prf_sampler.calcXResp(prf)

            brl = brl_sampler.finalValue
            prl = prl_sampler.finalValue

            l = drift_sampler.finalValue
            p = drift_sampler.P

            perf_baseline = perf_baseline_sampler.finalValue
            #wa = perf_baseline_sampler.compute_wa(perf_baseline)

            #fit = np.dot(vXh, brl) + np.dot(vXg, prl) + np.dot(p, l) + wa

            an = fit.axes_names
            ad = fit.axes_domains
            fitted_drift = xndarray(np.dot(p, l), axes_names=an, axes_domains=ad)
            w = self.dataInput.w
            fitted_perf = xndarray(w[:,np.newaxis] * np.dot(vXg, prl) + \
                                 fitted_drift.data + \
                                 perf_baseline, axes_names=an, axes_domains=ad)
            fitted_bold = xndarray(np.dot(vXh, brl) + fitted_drift.data,
                                 axes_names=an, axes_domains=ad)
            fitted_perf_baseline = xndarray(perf_baseline + fitted_drift.data,
                                          axes_names=an, axes_domains=ad)

            rp = self.dataInput.paradigm.get_rastered(self.dataInput.tr,
                                                      tMax=ad['time'].max())
            p = np.array([rp[n][0] for n in self.dataInput.cNames])

            p_adjusted = p[:,:,np.newaxis] * .15 * signal.ptp('time').data + \
              signal.min('time').data

            ad = {'time':fit.get_domain('time'),
                  'condition':self.dataInput.cNames}

            c_paradigm = xndarray(p_adjusted,
                                axes_names=['condition', 'time', 'voxel'],
                                axes_domains=ad)
            #stack everything

            outputs['fits'] = stack_cuboids([signal, fit, fitted_perf,
                                             fitted_bold,
                                             c_paradigm.sum('condition'),
                                             fitted_perf_baseline], 'stype',
                                             ['signal','fit','perf',
                                              'bold','paradigm',
                                              'perf_baseline'])
        return outputs


import pyhrf.jde.models
pyhrf.jde.models.allModels['ASL_PHYSIO0'] = {'class' : ASLPhysioSampler,
    'doc' : 'BOLD and perfusion component, physiological prior on responses,'
    'BiGaussian prior on stationary response levels, iid white noise, '\
    'explicit drift'
    }
