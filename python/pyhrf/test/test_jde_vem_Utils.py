# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tempfile
import shutil
import pyhrf
import pyhrf.tools as tools
from pyhrf import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.jde.models import simulate_bold
from pyhrf.ui.vb_jde_analyser_compMod import JDEVEMAnalyser
from pyhrf.vbjde.Utils import Main_vbjde_Extension, Main_vbjde_Python, Main_vbjde
from pyhrf.boldsynth.hrf import getCanoHRF
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict
    
class VEMBOLDTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8652761)

        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir
        simu = simulate_bold(self.tmp_dir, spatial_size='random_small')
        self.data_simu = FmriData.from_simulation_dict(simu)

        
    def tearDown(self):
        if 1:
            pyhrf.verbose(1, 'Remove tmp dir %s' %self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        else:
            pyhrf.verbose(1, 'Keep tmp dir %s' %self.tmp_dir)
       
        
    def test_jdevemanalyser(self):
        """ Test BOLD VEM sampler on small simulation with small 
        nb of iterations. Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)
        jde_vem_analyser = JDEVEMAnalyser(beta=.8, dt=.5, hrfDuration=25.,
                                        nItMax=2,nItMin=2, fast = True, 
                                        computeContrast = False, PLOT=False,
                                        constrained = True)
        tjde_vem = FMRITreatment(fmri_data=self.data_simu, 
                                analyser=jde_vem_analyser, 
                                output_dir=None)
        tjde_vem.run()  


    def test_vem_bold(self):
        """ Test BOLD VEM function.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()
        
        NbIter, nrls, estimated_hrf, \
        labels, noiseVar, mu_k, sigma_k, \
        Beta, L, PL, CONTRAST, CONTRASTVAR, \
        cA,cH,cZ,cAH,cTime,cTimeMean, \
        Sigma_nrls, StimuIndSignal,\
        FreeEnergy = Main_vbjde_Extension(graph,data.bold,Onsets,
                                            Thrf=25.,K=2,TR=1.,
                                            beta=1.0,dt=.5, 
                                            NitMax=2,NitMin=2)

    
    
    def test_vem_bold_python(self):
        """ 
        Test BOLD VEM function.
        Estimation accuracy is not tested
        """
        pyhrf.verbose.set_verbosity(0)
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()
        
        m_A,m_H, q_Z, sigma_epsilone, \
        mu_M , sigma_M, Beta, L, PL = Main_vbjde_Python(graph,data.bold,Onsets,
                                                        Thrf=25.,K=2,TR=1.,
                                                        beta=1.0,dt=.5, 
                                                        NitMax=2,NitMin=2) 
                                                        


    def test_vem_bold_v1(self):
        """ Test BOLD VEM function.
        Estimation accuracy is not tested.
        """
        pyhrf.verbose.set_verbosity(0)
        data = self.data_simu
        graph = data.get_graph()
        Onsets = data.get_joined_onsets()
        
        m_A, m_H, q_Z, sigma_epsilone, \
        Hist_sigmaH = Main_vbjde(graph,data.bold,Onsets,
                                Thrf=25.,K=2,TR=1.,
                                beta=1.0,dt=.5, 
                                NitMax=2,NitMin=2)  
        
        # TODO: check this test. Fails because of arguments 
        #       of the functions expectation and maximization
        
        
        
        