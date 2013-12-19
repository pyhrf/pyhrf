import os.path as op
import shutil
import unittest
import numpy as np
import numpy.testing as npt
from scipy.stats import truncnorm

import pyhrf
from pyhrf import Condition
import pyhrf.sandbox.physio as phy



class SimulationTest(unittest.TestCase):

    def setUp(self):
        # called before any unit test of the class
        #self.tmp_path = pyhrf.get_tmp_path() #create a temporary folder
        #self.clean_tmp = False

        #HACK:
        self.tmp_path = '/home/tom/Data/Pyhrf/Test/Unittest'
        self.clean_tmp = False

    def tearDown(self):
        # called after any unit test of the class
        if self.clean_tmp:
            pyhrf.verbose(1, 'cleaning temporary folder ...')
            shutil.rmtree(self.tmp_path)


    def test_simulate_asl_full_physio(self):

        pyhrf.verbose.set_verbosity(0)

        r = phy.simulate_asl_full_physio()
        # let's just test the shapes of objects and the presence of some
        # physio-specific simulation items
        item_names = r.keys()
        self.assertIn('perf_stim_induced', item_names)
        self.assertIn('flow_induction', item_names)
        self.assertIn('perf_stim_induced', item_names)
        self.assertEqual(r['labels_vol'].shape, (3,1,2,2)) #cond, spatial axes
        self.assertEqual(r['bold'].shape, (297, 4)) #nb scans, flat spatial axis

    def test_simulate_asl_full_physio_outputs(self):

        pyhrf.verbose.set_verbosity(1)

        phy.simulate_asl_full_physio(self.tmp_path)

        def makefn(fn):
            return op.join(self.tmp_path, fn)

        self.assertTrue(op.exists(makefn('flow_induction.nii')))
        self.assertTrue(op.exists(makefn('neural_efficacies_audio.nii')))



    def test_simulate_asl_physio_rfs(self):

        pyhrf.verbose.set_verbosity(0)

        r = phy.simulate_asl_physio_rfs()
        # let's just test the shapes of objects and the presence of some
        # physio-specific simulation items
        item_names = r.keys()
        self.assertIn('perf_stim_induced', item_names)
        self.assertIn('primary_brf', item_names)
        self.assertIn('perf_stim_induced', item_names)
        self.assertEqual(r['labels_vol'].shape, (3,1,2,2)) #cond, spatial axes
        self.assertEqual(r['bold'].shape, (321, 4)) #nb scans, flat spatial axis
        #TODO: nb scans in final BOLD should be defined by the session length
        #      from the paradigm -> 297 instead of 321



    def test_create_tbg_neural_efficacies(self):
        """ Test the generation of neural efficacies from a truncated
        bi-Gaussian mixture
        """
        m_act = 5.
        v_act = .05
        v_inact = .05
        cdef = [Condition(m_act=m_act, v_act=v_act, v_inact=v_inact)]
        npos = 5000
        labels = np.zeros((1,npos), dtype=int)
        labels[0, :npos/2] = 1
        phy_params = phy.PHY_PARAMS_FRISTON00
        ne = phy.create_tbg_neural_efficacies(phy_params, cdef, labels)

        #check shape consistency:
        self.assertEqual(ne.shape, labels.shape)

        #check that moments are close to theoretical ones
        ne_act = ne[0, np.where(labels[0])]
        ne_inact = ne[0, np.where(labels[0]==0)]
        m_act_theo = truncnorm.mean(0, phy_params['eps_max'], loc=m_act,
                                    scale=v_act**.5)
        v_act_theo = truncnorm.var(0, phy_params['eps_max'], loc=m_act,
                                   scale=v_act**.5)
        (ne_act.mean(), m_act_theo)
        npt.assert_approx_equal(ne_act.var(), v_act_theo, significant=2)

        m_inact_theo = truncnorm.mean(0, phy_params['eps_max'], loc=0.,
                                      scale=v_inact**.5)
        v_inact_theo = truncnorm.var(0, phy_params['eps_max'], loc=0.,
                                     scale=v_inact**.5)
        npt.assert_approx_equal(ne_inact.mean(), m_inact_theo, significant=2)
        npt.assert_approx_equal(ne_inact.var(), v_inact_theo, significant=2)
        npt.assert_array_less(ne, phy_params)
        npt.assert_array_less(0., ne)


    def test_create_physio_brf(self):
        phy_params = phy.PHY_PARAMS_FRISTON00
        dt = .5
        duration = 25.
        brf = phy.create_physio_brf(phy_params, response_dt=dt,
                                    response_duration=duration)

        if 0:
            import matplotlib.pyplot as plt
            t = np.arange(brf.size) * dt
            plt.plot(t, brf)
            plt.title('BRF')
            plt.show()

        npt.assert_approx_equal(brf[0], 0., significant=4)
        npt.assert_array_almost_equal(brf[-1], 0., decimal=4)

        npt.assert_approx_equal(np.argmax(brf)*dt, 3.5, significant=5)

    def test_create_physio_prf(self):


        phy_params = phy.PHY_PARAMS_FRISTON00
        dt = .5
        duration = 25.
        prf = phy.create_physio_prf(phy_params, response_dt=dt,
                                    response_duration=duration)

        if 0:
            import matplotlib.pyplot as plt
            t = np.arange(prf.size) * dt
            plt.plot(t, prf)
            plt.title('PRF')
            plt.show()

        npt.assert_approx_equal(prf[0], 0., significant=4)
        npt.assert_array_almost_equal(prf[-1], 0., decimal=4)

        npt.assert_approx_equal(np.argmax(prf)*dt, 2.5, significant=5)


    def test_create_evoked_physio_signal(self):
        import pyhrf.paradigm

        phy_params = phy.PHY_PARAMS_FRISTON00
        tr = 1.
        duration = 20.
        ne = np.array([[10., 5.]])
        nb_conds, nb_vox = ne.shape
        # one single stimulation at the begining
        paradigm = pyhrf.paradigm.Paradigm({'c':[np.array([0.])]}, [duration],
                                           {'c':[np.array([1.])]})
        s, f, hbr, cbv = phy.create_evoked_physio_signals(phy_params, paradigm,
                                                          ne, tr)
        #shape of a signal: (nb_vox, nb_scans)

        if 0:
            import matplotlib.pyplot as plt
            t = np.arange(f[0].size) * tr
            plt.plot(t, f[0])
            plt.title('inflow')
            plt.show()

        self.assertEqual(s.shape, (int(duration/tr), nb_vox))

        # check signal causality:
        self.assertEqual(f[0,0], 1.)
        npt.assert_approx_equal(f[-1,0], 1., significant=3)

        # non-regression test:
        self.assertEqual(np.argmax(f[:,0])*tr, 2)


    def test_phy_integrate_euler(self):
        phy_params = phy.PHY_PARAMS_FRISTON00
        tstep = .05
        nb_steps = 400
        stim_duration = int(1/tstep)
        stim = np.array([1.]*stim_duration + [0.]*(nb_steps-stim_duration))
        epsilon = .5

        s,f,q,v = phy.phy_integrate_euler(phy_params, tstep, stim, epsilon)

        # signal must be causal:
        self.assertEqual(f[0], 1.)
        npt.assert_approx_equal(f[-1], 1., significant=3)

        # non-regression checks:
        npt.assert_approx_equal(np.argmax(f)*tstep, 2.3)
        npt.assert_approx_equal(f.max(), 1.384, significant=4)

        if 0:
            import matplotlib.pyplot as plt
            t = np.arange(nb_steps) * tstep
            plt.plot(t,f)
            plt.title('inflow')
            plt.show()



####
    def test_finite_dif_matrix(self):
        phy.buildOrder1FiniteDiffMatrix(10)
        return None
