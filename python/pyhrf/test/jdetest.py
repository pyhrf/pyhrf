# -*- coding: utf-8 -*-
import unittest
import numpy as np

import tempfile
import shutil

import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import jde_vol_from_files, jde_surf_from_files
from pyhrf.parcellation import parcellation_report, parcellation_for_jde

from pyhrf.jde.models import BOLDGibbsSampler as BG
from pyhrf.jde.beta import BetaSampler as BS
from pyhrf.jde.nrl.bigaussian import NRLSampler as NS
from pyhrf.jde.nrl.bigaussian import BiGaussMixtureParamsSampler as BGMS
from pyhrf.jde.hrf import RHSampler as HVS
from pyhrf.jde.hrf import HRFSampler as HS
from pyhrf.jde.models import simulate_bold
from pyhrf.jde.noise import NoiseVarianceSampler

import pyhrf.sandbox.physio as phym
from pyhrf import tools


class JDETest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

        bf = 'subj0_bold_session0.nii.gz'
        self.boldFiles = [pyhrf.get_data_file_name(bf)]
        pf = 'subj0_parcellation.nii.gz'
        self.parcelFile = pyhrf.get_data_file_name(pf)
        self.tr = 2.4
        self.dt = .6
        self.onsets = pyhrf.onsets_loc_av
        #self.durations = pyhrf.durations_loc_av
        self.durations = None
        self.nbIt = 3
        self.pfMethod = 'es'

        simu = simulate_bold(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

        #print 'Create sampler_params_for_single_test ...'
        self.sampler_params_for_single_test = {
            BG.P_NB_ITERATIONS : 100,
            BG.P_SMPL_HIST_PACE : 1,
            BG.P_OBS_HIST_PACE : 1,
            # level of spatial correlation = beta
            BG.P_BETA : BS({
                    BS.P_SAMPLE_FLAG : False,
                    BS.P_USE_TRUE_VALUE : False,
                    BS.P_VAL_INI : np.array([0.6]),
                    }),
            # HRF
            BG.P_HRF : HS({
                    HS.P_SAMPLE_FLAG : False,
                    HS.P_USE_TRUE_VALUE : True,
                    HS.P_PRIOR_TYPE : 'singleHRF',
                    }),
            # HRF variance
            BG.P_RH : HVS({
                    HVS.P_USE_TRUE_VALUE : True,
                    HVS.P_SAMPLE_FLAG : False,
                    }),
            # neural response levels (stimulus-induced effects)
            BG.P_NRLS : NS({
                    NS.P_USE_TRUE_NRLS : True,
                    NS.P_USE_TRUE_LABELS : True,
                    NS.P_SAMPLE_FLAG : False,
                    NS.P_SAMPLE_LABELS : False,
                    }),
            BG.P_MIXT_PARAM : BGMS({
                    BGMS.P_SAMPLE_FLAG : False,
                    BGMS.P_USE_TRUE_VALUE : True,
                    }),
            BG.P_NOISE_VAR : NoiseVarianceSampler({
                    NoiseVarianceSampler.P_SAMPLE_FLAG : False,
                    NoiseVarianceSampler.P_USE_TRUE_VALUE : True,
                    }),
            BG.P_CHECK_FINAL_VALUE : 'raise', #print or raise
            }


        #print 'Create sampler_params_for_full_test ...'
        self.sampler_params_for_full_test = {
            BG.P_NB_ITERATIONS : 500,
            BG.P_SMPL_HIST_PACE : 1,
            BG.P_OBS_HIST_PACE : 1,
            # level of spatial correlation = beta
            BG.P_BETA : BS({
                    BS.P_SAMPLE_FLAG : True,
                    BS.P_USE_TRUE_VALUE : False,
                    BS.P_VAL_INI : np.array([0.6]),
                    }),
            # HRF
            BG.P_HRF : HS({
                    HS.P_SAMPLE_FLAG : True,
                    HS.P_USE_TRUE_VALUE : False,
                    HS.P_NORMALISE : 1.,
                    }),
            # HRF variance
            BG.P_RH : HVS({
                    HVS.P_USE_TRUE_VALUE : False,
                    HVS.P_SAMPLE_FLAG : True,
                    }),
            # neural response levels (stimulus-induced effects)
            BG.P_NRLS : NS({
                    NS.P_USE_TRUE_NRLS : False,
                    NS.P_USE_TRUE_LABELS : False,
                    NS.P_SAMPLE_FLAG : True,
                    NS.P_SAMPLE_LABELS : True,
                    }),
            BG.P_MIXT_PARAM : BGMS({
                    BGMS.P_SAMPLE_FLAG : True,
                    BGMS.P_USE_TRUE_VALUE : False,
                    }),
            BG.P_NOISE_VAR : NoiseVarianceSampler({
                    NoiseVarianceSampler.P_SAMPLE_FLAG : True,
                    NoiseVarianceSampler.P_USE_TRUE_VALUE : False,
                    }),
            BG.P_CHECK_FINAL_VALUE : 'print', #print or raise
            }

    def tearDown(self):
        if 1:
            pyhrf.verbose(1, 'Remove tmp dir %s' %self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            pyhrf.verbose(1, 'Keep tmp dir %s' %self.tmp_dir)


    def testDefaultWithOutputs(self):
        #pyhrf.verbose.setVerbosity(1)
        treatment, xml_file = jde_vol_from_files(self.boldFiles,
                                                 self.parcelFile,
                                                 self.dt, self.tr,
                                                 nbIterations=self.nbIt,
                                                 pfMethod=self.pfMethod,
                                                 outputDir=self.tmp_dir)
        #treatment.clean_output_files()



    def test_parcellation(self):

        p_size = 300
        np.random.seed(125437)
        parcellation,_ = parcellation_for_jde(FmriData.from_vol_ui(), p_size,
                                              output_dir=self.tmp_dir)
        ms = np.mean([(parcellation==i).sum() \
                          for i in np.unique(parcellation) if i != 0])
        size_tol = 50
        if np.abs(ms-p_size) > size_tol:
            raise Exception('Mean size of parcellation seems too ' \
                                'large: %1.2f >%d+-%d ' %(ms,p_size,size_tol))
        #print 'parcel ids:', np.unique(parcellation)
        if 0:
            print parcellation_report(parcellation)

    def test_surface_treatment(self):
        #pyhrf.verbose.setVerbosity(2)
        treatment, xml_file, result = jde_surf_from_files(nbIterations=2,
                                                          outputDir=self.tmp_dir)
        #treatment.clean_output_files()
        #if xml_file is not None:
        #    os.remove(xml_file)





from pyhrf.tools.io import read_volume
from pyhrf.graph import parcels_to_graphs, kerMask3D_6n
from pyhrf.jde.beta import Cpt_Vec_Estim_lnZ_Graph, Cpt_Vec_Estim_lnZ_Graph_fast3

class PartitionFunctionTest(unittest.TestCase):

    def setUp(self):
        pf = 'subj0_parcellation.nii.gz'
        fnm = pyhrf.get_data_file_name(pf)
        m,mh = read_volume(fnm)
        self.graph = parcels_to_graphs(m.astype(int), kerMask3D_6n,
                                       toDiscard=[0])[1]


    def testExtrapolation2C(self):
        nbclasses = 2
        lnz_ps = Cpt_Vec_Estim_lnZ_Graph(self.graph, nbclasses)
        lnz_es = Cpt_Vec_Estim_lnZ_Graph_fast3(self.graph, nbclasses)
        assert ((lnz_ps[0]-lnz_es[0][:len(lnz_ps[0])])/lnz_ps[0]).max() < 0.02



from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.jde.asl import simulate_asl
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.jde import asl as jde_asl
from pyhrf.jde import asl_physio as jde_asl_physio

class ASLTest(unittest.TestCase):

    def setUp(self):

        pyhrf.verbose.setVerbosity(0)

        np.random.seed(8652761)

        self.tmp_dir = pyhrf.get_tmp_path()
        #self.tmp_dir = './'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_simulation(self):

        pyhrf.verbose.setVerbosity(0)
        simulate_asl(spatial_size='random_small')

    def test_default_jde_small_simulation(self):
        """ Test ASL sampler on small simulation with small nb of iterations.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.setVerbosity(0)


        simu = simulate_asl(spatial_size='random_small')
        fdata = FmriData.from_simulation_dict(simu)

        sampler = jde_asl.ASLSampler()

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputFile=None,outputPrefix='jde_mcmc_',
                                   randomSeed=None)

        treatment = FMRITreatment(fmri_data=fdata, analyser=analyser)

        treatment.run()

class ASLPhysioTest(unittest.TestCase):

    def setUp(self):

        pyhrf.verbose.setVerbosity(0)

        np.random.seed(8652761)

        self.tmp_dir = pyhrf.get_tmp_path()
        #self.tmp_dir = './'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_default_jde_small_simulation(self):
        """ Test ASL Physio sampler on small simulation with small nb of
        iterations. Estimation accuracy is not tested.
        """
        pyhrf.verbose.setVerbosity(0)

        sampler_params = {
            jde_asl_physio.ASLPhysioSampler.P_NB_ITERATIONS : 100,
            jde_asl_physio.ASLPhysioSampler.P_SMPL_HIST_PACE : 1,
            jde_asl_physio.ASLPhysioSampler.P_OBS_HIST_PACE : 1,
            'brf' : jde_asl_physio.PhysioBOLDResponseSampler(zero_constraint=False),
            'brf_var' : jde_asl_physio.PhysioBOLDResponseVarianceSampler(val_ini=\
                                                                         np.array([1e-3])),
            'prf' : jde_asl_physio.PhysioPerfResponseSampler(zero_constraint=False),
            'prf_var' : jde_asl_physio.PhysioPerfResponseVarianceSampler(val_ini=\
                                                                         np.array([1e-3])),
            'noise_var' : jde_asl_physio.NoiseVarianceSampler(),
            'drift_var' : jde_asl_physio.DriftVarianceSampler(),
            'drift_coeff' : jde_asl_physio.DriftCoeffSampler(),
            'brl' : jde_asl_physio.BOLDResponseLevelSampler(),
            'prl' : jde_asl_physio.PerfResponseLevelSampler(),
            'bold_mixt_params' : jde_asl_physio.BOLDMixtureSampler(),
            'perf_mixt_params' : jde_asl_physio.PerfMixtureSampler(),
            'label' : jde_asl_physio.LabelSampler(),
            'perf_baseline' : jde_asl_physio.PerfBaselineSampler(),
            'perf_baseline_var' : jde_asl_physio.PerfBaselineVarianceSampler(),
            'assert_final_value_close_to_true' : False,
        }


        sampler = jde_asl_physio.ASLPhysioSampler(sampler_params)

        simu_items = phym.simulate_asl_physio_rfs(spatial_size='random_small')
        simu_fdata = FmriData.from_simulation_dict(simu_items)

        dt = simu_items['dt']
        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=dt, driftParam=4, driftType='polynomial',
                                   outputFile=None,outputPrefix='jde_mcmc_',
                                   randomSeed=None)

        treatment = FMRITreatment(fmri_data=simu_fdata, analyser=analyser)
        treatment.run()


def test_suite():
    tests = [unittest.makeSuite(ASLTest, 'test_jde')]
    return unittest.TestSuite(tests)

if __name__== '__main__':
    #unittest.main(argv=['pyhrf.test_glm'])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite())



from pyhrf.jde.jde_multi_sess import BOLDGibbs_Multi_SessSampler as BMSS
from pyhrf.jde.jde_multi_sess import simulate_sessions
from pyhrf.core import merge_fmri_sessions

#import functools
#To automatically enter the debugger in case of bug
#def debug_on(*exceptions):
    #if not exceptions:
        #exceptions = (AssertionError, )
    #def decorator(f):
        #@functools.wraps(f)
        #def wrapper(*args, **kwargs):
            #try:
                #return f(*args, **kwargs)
            #except exceptions:
                #pdb.post_mortem(sys.exc_info()[2])
        #return wrapper
    #return decorator

class MultiSessTest(unittest.TestCase):
    #@debug_on()

    def setUp(self):

        self.tmp_dir = pyhrf.get_tmp_path()

        simu = simulate_sessions(output_dir = self.tmp_dir,
                                 snr_scenario='high_snr',
                                 spatial_size='random_small')
        self.data_simu = merge_fmri_sessions(simu)


    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_default_jde_small_simulation(self):
        """ Test JDE multi-sessions sampler on small
        simulation with small nb of iterations.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.setVerbosity(0)

        sampler = BMSS()

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.5, driftParam=4, driftType='polynomial',
                                   outputFile=None,outputPrefix='jde_MS_mcmc_',
                                   randomSeed=9778946)

        treatment = FMRITreatment(fmri_data=self.data_simu,
                                  analyser=analyser)

        treatment.run()


# def test_suite():
#     tests = [unittest.makeSuite(MultiSessTest, 'test_jde_multi_sess')]
#     return unittest.TestSuite(tests)

#Method Matthieu to debug directly:
#MultiSessTest.runTest=None
#then we can instantiate the class directly

#Plot histos to verify values
#(z.varClassApost[1,0,np.where(z.finalLabels[0,:])]).mean()


