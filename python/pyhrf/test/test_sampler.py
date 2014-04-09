import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from pyhrf.jde.samplerbase import Trajectory

class TrajectoryTest(unittest.TestCase):

    def test_basic(self):

        v = np.array([1, 2])
        nb_its = 10
        pace = 1
        start = 0
        t = Trajectory(v, ['time'], {'time':np.array([1,2])},
                       pace, start, nb_its)

        for i in xrange(nb_its):
            v *= 2
            t.record(i)

        assert_array_equal(t.get_last(), np.array([1, 2]) * 2**nb_its)
        self.assertEqual(t.saved_iterations, [-1] + range(start,nb_its))

    def test_burnin(self):

        v = np.array([1, 2])
        nb_its = 10
        pace = 1
        start = 5
        t = Trajectory(v, ['time'], {'time':np.array([1,2])},
                       pace, start, nb_its, first_saved_iteration=start)

        for i in xrange(nb_its):
            v *= 2
            t.record(i)

        assert_array_equal(t.get_last(), np.array([1, 2]) * 2**nb_its)
        self.assertEqual(t.saved_iterations, range(start,nb_its))

        

class GibbsTest(unittest.TestCase):

    # class DummyGSVar(GibbsSamplerVariable):
    #     def __init__(self, do_sampling):
    #         GibbsSamplerVariable.__init__('dummy', np.array([0.]),
    #                                       sampleFlag=do_sampling)


    def test_var_tracking(self):
        pass
