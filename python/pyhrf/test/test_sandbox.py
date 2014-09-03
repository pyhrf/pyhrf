import unittest
import numpy as np

import pyhrf
from pyhrf.sandbox.design_and_ui import Initable, UiNode
from numpy.testing import assert_array_equal, assert_almost_equal
from optparse import OptionParser

class Dummy(Initable):

    def __init__(self, string_p, float_p, int_p, array_p, obj_p):
        Initable.__init__(self)
        self.string_p = string_p
        self.float_p = float_p
        self.int_p = int_p
        self.array_p = array_p
        self.obj_p = obj_p

class Dummy2(Initable):
    def __init__(self, p=3):
        Initable.__init__(self)
        self.p = p

class Dummy3(Initable):

    def __init__(self, d):
        Initable.__init__(self)
        self.d = d

class FmriData(Initable):
    def __init__(self, paradigm=np.arange(5).astype(int),
                 func_file='./bold.nii'):
        Initable.__init__(self)
        self.paradigm = paradigm
        self.func_file = func_file

class Analyser(Initable):
    def __init__(self, nb_iterations=1000, make_outputs=True):
        Initable.__init__(self)
        self.nb_iterations = nb_iterations
        self.make_outputs = make_outputs


class Treatment(Initable):
    def __init__(self, data=FmriData(), analyser=Analyser(), info={},
                 l=[5,6]):
        Initable.__init__(self)

        self.data = data
        self.analyser = analyser
        self.info = info
        self.list = l

class InitableTest(unittest.TestCase):

    def test_new_obj(self):

        d = Dummy('a', 1.3, 5, np.arange(5), Dummy2('rr'))
        self.assertEqual(d.string_p, 'a')
        self.assertEqual(d.float_p, 1.3)
        self.assertEqual(d.int_p, 5)

        d2 = d.init_new_obj()
        self.assertEqual(d2.string_p, 'a')
        self.assertEqual(d2.float_p, 1.3)
        self.assertEqual(d2.int_p, 5)

    def test_to_ui_node_basic(self):

        d = Dummy('a', 1.3, 5, np.arange(5), Dummy2('rr'))
        root_node = d.to_ui_node('dummy')

        #                dummy.string_p.string_p_value
        self.assertEqual(root_node.child(0).child(0).label(), 'a')
        self.assertEqual(root_node.child(0).get_attribute('type'), str)
        #                dummy.float_p.string_p_value
        self.assertEqual(root_node.child(1).child(0).label(), '1.3')
        self.assertEqual(root_node.child(1).get_attribute('type'), type(1.3))

    def test_from_ui_node_basic(self):
        d = Dummy('a', 1.3, 5, np.arange(5), Dummy2('rr'))
        root_node = d.to_ui_node('dummy')
        d2 = Initable.from_ui_node(root_node)

        self.assertEqual(d.string_p, d2.string_p)
        self.assertEqual(d.float_p, d2.float_p)
        self.assertEqual(d.int_p, d2.int_p)
        assert_array_equal(d.array_p, d2.array_p)

    def test_to_ui_node_dict(self):
        pyhrf.verbose.set_verbosity(0)
        subd = Dummy('a', 1.3, 5, np.arange(5), Dummy2('rr'))
        d = Dummy3(d={'subdummy':subd})
        root_node = d.to_ui_node('dummy')

        if pyhrf.verbose.verbosity > 0:
            print 'root_node:'
            print root_node.log()

        #                dummy.d.subdummy.string_p.string_p_value
        node_subdummy = root_node.child(0).child(0)
        self.assertEqual(node_subdummy.child(0).child(0).label(), 'a')
        self.assertEqual(node_subdummy.child(0).get_attribute('type'), str)
        #                dummy.subdummy.float_p.string_p_value
        self.assertEqual(node_subdummy.child(1).child(0).label(), '1.3')
        self.assertEqual(node_subdummy.child(1).get_attribute('type'),
                         type(1.3))



    def test_to_ui_node_complex(self):
        pyhrf.verbose.set_verbosity(0)
        info = {'test':53}
        treatment = Treatment(FmriData(func_file='blah.nii'),
                              Analyser(nb_iterations=345),
                              info=info)

        root_node = treatment.to_ui_node('treatment')

        if pyhrf.verbose.verbosity > 0:
            print 'root_node:'
            print root_node.log()

        #                treatment.fdata.funcfile.funcfile_value
        self.assertEqual(root_node.child(0).child(1).child(0).label(), 'blah.nii')
        self.assertEqual(root_node.child(0).child(1).get_attribute('type'), str)
        self.assertEqual(root_node.childCount(), 4)
        self.assertEqual(root_node.child(0).childCount(), 2)
        #                treatment.fdata.paradigm
        paradigm_node = root_node.child(0).child(0)
        a = treatment.data.paradigm
        pval0 = ' '.join( str(e) for e in a.ravel() )
        ptype0 = str(a.dtype.name)
        self.assertEqual(paradigm_node.child(0).label(), pval0)
        self.assertEqual(paradigm_node.get_attribute('type'), 'ndarray')
        self.assertEqual(paradigm_node.get_attribute('dtype'), ptype0)
        self.assertEqual(paradigm_node.childCount(), 1)

        self.assertEqual(root_node.child(0).child(1).get_attribute('type'), str)
        #                treatment.analyzer.nb_iterations.nb_iterations_value
        self.assertEqual(root_node.child(1).child(0).child(0).label(), '345')
        self.assertEqual(root_node.child(1).child(0).get_attribute('type'), int)
        #                treatment.analyzer.make_outputs
        self.assertEqual(root_node.child(1).child(1).get_attribute('type'), bool)
        #                treatment.info
        self.assertEqual(root_node.child(2).get_attribute('type'), 'dict')
        self.assertEqual(root_node.child(2).child(0).label(), 'test')
        self.assertEqual(root_node.child(2).child(0).child(0).label(), str(53))

        #                treatment.list
        self.assertEqual(root_node.child(3).get_attribute('type'), 'list')
        self.assertEqual(root_node.child(3).child(0).label(), 'item0')

        self.assertEqual(root_node.child(3).child(0).get_attribute('type'), int)
        self.assertEqual(root_node.child(3).child(0).child(0).label(), '5')

    def test_xml_io(self):
        pyhrf.verbose.set_verbosity(1)

        info = {'test':53}
        treatment = Treatment(FmriData(func_file='blah.nii'),
                              Analyser(nb_iterations=345),
                              info=info)

        root_node = treatment.to_ui_node('treatment')

        if pyhrf.verbose.verbosity > 0:
            print 'root_node:'
            print root_node.log()

        sxml = root_node.to_xml()

        if pyhrf.verbose.verbosity > 0:

            print 'sxml:'
            print sxml
            print ''


            print 'from_xml ...'
        root_node2 = UiNode.from_xml(sxml)

        if pyhrf.verbose.verbosity > 0:

            print 'root_node2:'
            print root_node2.log()
            print ''

        treatment2 = Treatment.from_ui_node(root_node2)

        if pyhrf.verbose.verbosity > 0:

            print 'treatment2:'
            print treatment2



class UiNodeTest(unittest.TestCase):

    def test_from_numpy1D(self):

        n = UiNode.from_py_object('n', np.arange(5))
        if 0:
            print 'n:'
            print n.log()

        o = Initable.from_ui_node(n)
        assert_array_equal(o, np.arange(5))


    def test_from_numpyND(self):

        sh = (2,3,4,5)
        a = np.arange(np.prod(sh)).reshape(sh)
        n = UiNode.from_py_object('n', a)
        if 0:
            print 'n:'
            print n.log()

        o = Initable.from_ui_node(n)
        assert_array_equal(o, a)



    def test_from_numpy_scalar(self):

        a = np.float128(1.2)
        n = UiNode.from_py_object('a', a)
        if 0:
            print 'n:'
            print n.log()

        o = Initable.from_ui_node(n)
        self.assertEqual(o, a)
        self.assertEqual(o.dtype, a.dtype)


    def test_serialize_init_obj(self):
        d = Dummy2()
        s = UiNode._serialize_init_obj(d.__init__)
        f = UiNode._unserialize_init_obj(s)

        self.assertEqual(d.__class__, f)

    def test_to_xml(self):
        pyhrf.verbose.set_verbosity(0)
        d = Dummy2()
        n = UiNode.from_py_object('dummy', d)

        xml = n.to_xml()

        if pyhrf.verbose.verbosity > 0:
            print 'xml:'
            print xml


    def test_from_xml(self):
        pyhrf.verbose.set_verbosity(0)

        d = Dummy2(p=56)
        n = UiNode.from_py_object('dummy', d)

        if pyhrf.verbose.verbosity > 0:
            print 'n:'
            print n.log()
            print ''

        xml = n.to_xml()

        if pyhrf.verbose.verbosity > 0:
            print 'xml:'
            print xml

        n2 = UiNode.from_xml(xml)

        if pyhrf.verbose.verbosity > 0:
            print 'n2:'
            print n2.log()
            print ''

#####################
# Core data objects #
#####################

import pyhrf.sandbox.core as xcore
from pyhrf.ndarray import xndarray
from pyhrf.tools._io import load_paradigm_from_csv
from numpy.testing import assert_almost_equal

class CoreTest(unittest.TestCase):

    def setUp(self):
        cmask = xndarray.load(xcore.DEFAULT_MASK)
        self.mask_vol = cmask.data
        self.vol_meta_data = cmask.meta_data

        self.bold_vol = xndarray.load(xcore.DEFAULT_BOLD).data
        m = np.where(self.mask_vol)
        self.bold_flat = self.bold_vol[m[0], m[1], m[2],:].T

        self.onsets = xcore.DEFAULT_ONSETS
        self.durations = xcore.DEFAULT_STIM_DURATIONS

    def test_fmridata_vol(self):

        fd = xcore.FmriData(self.bold_vol, self.onsets, self.durations,
                            xcore.DEFAULT_TR, 'volume', self.mask_vol,
                            meta_data=self.vol_meta_data)

        assert_almost_equal(self.bold_flat, fd.fdata)
        assert_almost_equal(self.mask_vol, fd.mask)

    def test_fmridata_ui_vol(self):

        pyhrf.verbose.set_verbosity(0)
        fdui = xcore.FmriDataUI()
        fd = fdui.get_fmri_data()
        assert_almost_equal(self.bold_flat, fd.fdata)
        assert_almost_equal(self.mask_vol, fd.mask)

    def test_fmridata_ui_vol_paradigm_csv(self):
        # TODO: check onsets !
        pyhrf.verbose.set_verbosity(0)
        fdui = xcore.FmriDataUI.from_paradigm_csv()
        fd = fdui.get_fmri_data()
        assert_almost_equal(self.bold_flat, fd.fdata)
        assert_almost_equal(self.mask_vol, fd.mask)

    def test_fmridata_ui_from_cmd_options_default(self):
        parser = OptionParser(usage='test', description='for test')
        xcore.FmriDataUI.append_cmd_options(parser)
        options, args = parser.parse_args([])

        fdui = xcore.FmriDataUI.from_cmd_options(options)
        fd = fdui.get_fmri_data()

        assert_almost_equal(self.bold_flat, fd.fdata)
        assert_almost_equal(self.mask_vol, fd.mask)

    def test_to_xml(self):

        fdui = xcore.FmriDataUI()
        xml = fdui.to_xml()
        if 0:
            print 'xml:'
            print xml

        pyhrf.verbose.set_verbosity(0)
        fdui2 = xcore.FmriDataUI.from_xml(xml)

        self.assertEqual(fdui.mask_ui.data_type, fdui2.mask_ui.data_type)
        self.assertEqual(fdui.tr, fdui2.tr)






########################
# Spatial parcellation #
########################

import numpy as np


class SpatialParcellationTest(unittest.TestCase):

    def setUp(self):
        self.territories_2D_2parcels = np.array([[1,1,1],
                                                 [1,1,2],
                                                 [2,2,2],
                                                 [2,2,2]])

    # def test_on_true_territories(self):
    #     from pyhrf.sandbox.parcellation import parcellate_spatial
    #     import matplotlib.pyplot as plt
    #     parcellation = parcellate_spatial(self.territories_2D_2parcels)

    #     if 0:
    #         plt.matshow(self.territories_2D_2parcels)
    #         plt.title('True territories')

    #         plt.matshow(parcellation)
    #         plt.title('parcellation')

    #         plt.show()




#################
# Gibbs Sampler #
#################


from pyhrf.sandbox.stats import GSVariable, GibbsSampler

class GibbsSamplerTest(unittest.TestCase):

    def test_inference(self):

        class GSVar_X(GSVariable):

            def __init__(self):
                GSVariable.__init__(self, 'x', initialization=0.)
                                    # required_variables=['y', 'noise_var',
                                    #                     'x_prior_var'])

            def sample(self):
                y = self.get_variable_value('y')
                s2 = self.get_variable_value('noise_var')
                x_prior_var = self.get_variable_value('x_prior_var')

                # Do the sampling:
                post_var = 1 / (y.size * (1/s2 + 1/x_prior_var))
                post_mean = y.sum() * post_var / s2
                return np.random.randn() * post_var + post_mean

        class MyGS(GibbsSampler):

            def __init__(self):
                GibbsSampler.__init__(self, [GSVar_X()], nb_its_max=100)

        x_true = 1.
        n = 500
        noise = np.random.randn(n) * .2
        y = x_true + noise

        gs = MyGS()
        gs.set_variables({'y':y, 'noise_var':noise.var(), 'x_prior_var':1000.})
        gs.set_true_values({'x' : x_true})

        gs.run()
        outputs = gs.get_outputs(output_type='cuboid')
        x_pm = outputs['x_obs_mean'].data[0]
        #x_pv = outputs['x_obs_var'].data[0]
        #print 'x_pm:', x_pm, 'x_pv:', x_pv

        #let's allow 5% error:
        #TODO: should be tested at the end of GS
        if np.abs(x_pm - x_true) / x_true > 0.05:
            raise Exception('Inaccurate post mean value: %f. '\
                            'Excepted: %f' %(x_pm, x_true))


    def test_trajectories(self):
        class GSVar_X(GSVariable):

            def __init__(self):
                GSVariable.__init__(self, 'x', initialization=np.zeros(2))

            def sample(self):
                return self.current_value + 1

        class MyGS(GibbsSampler):

            def __init__(self, nb_its, burnin, obs_hist_pace, smpl_hist_pace):
                GibbsSampler.__init__(self, [GSVar_X()], nb_its_max=nb_its,
                                      obs_hist_pace=obs_hist_pace,
                                      sample_hist_pace=smpl_hist_pace,
                                      burnin=burnin)
        burnin = 2
        obs_hist_pace = 3
        smpl_hist_pace = 3
        nb_its = 10
        gs = MyGS(nb_its, burnin, obs_hist_pace, smpl_hist_pace)
        gs.run()
        outputs = gs.get_outputs(output_type='cuboid')

        x_samples = outputs['x_hist_smpl'].data
        x_pm_hist = outputs['x_hist_obs_mean']
        self.assertEqual(x_samples.shape[0], np.ceil(nb_its*1./smpl_hist_pace))
        self.assertEqual(x_pm_hist.data.shape[0],
                         np.ceil((nb_its*1.-burnin)/obs_hist_pace))
        self.assertEqual(x_samples.shape,
                         (int(np.ceil(nb_its*1./smpl_hist_pace)), 2))
        self.assertEqual(x_pm_hist.axes_names[0], 'iteration')
        assert_array_equal(x_pm_hist.axes_domains['iteration'],
                           np.arange(burnin, nb_its, obs_hist_pace, dtype=int))



    def test_initialization(self):

        class GSVar_X(GSVariable):

            def __init__(self, init):
                GSVariable.__init__(self, 'x', initialization=init)

            def sample(self):
                return self.current_value + 1

            def get_custom_init(self):
                y = self.get_variable('y')
                return y.mean()

            def get_random_init(self):
                return np.pi

        class MyGS(GibbsSampler):

            def __init__(self, nb_its, burnin, x_init):
                GibbsSampler.__init__(self, [GSVar_X(x_init)], nb_its_max=nb_its,
                                      burnin=burnin)

        burnin = 2
        nb_its = 10
        x_init = 0.
        gs = MyGS(nb_its, burnin, x_init)
        y = np.random.randn(20)

        #TODO: split into specific unit tests

        #default initialization for x (to zero):
        gs.run()
        self.assertEqual(gs.get_variable_value('x'), x_init + nb_its)

        # custom initialization for x -> require definition of y:
        gs.reset()
        gs.set_variable('y', y)
        gs.set_initialization('x', 'custom')
        gs.run()
        assert_almost_equal(gs.get_variable_value('x'), y.mean() + nb_its)

        # random initialization for x -> set to np.pi:
        gs.reset()
        gs.set_initialization('x', 'random')
        gs.run()
        assert_almost_equal(gs.get_variable_value('x'), np.pi + nb_its)

        # initialization to true value for x:
        x_true = 2.
        gs.reset()
        gs.set_initialization('x', 'truth')
        gs.set_true_value('x', x_true)
        gs.run()
        assert_almost_equal(gs.get_variable_value('x'), x_true + nb_its)


