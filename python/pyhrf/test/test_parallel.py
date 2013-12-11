# -*- coding: utf-8 -*-

import unittest
import pyhrf
from pyhrf.parallel import remote_map, RemoteException
from pyhrf.configuration import cfg

def foo(a, b):
    return a+b

def foo_raise(a,b):
    raise Exception('raised by a foo')

class ParallelTest(unittest.TestCase):

    def _test_remote_map_basic(self, mode):

        r = remote_map(foo, [(1,),(2,)], [{'b':5},{'b':10}], mode=mode)
        if 0:
            print 'result:'
            print r

    def test_remote_map_serial(self):
        self._test_remote_map_basic(mode='serial')

    def test_remote_map_local(self):
        self._test_remote_map_basic(mode='local')



    def test_remote_map_local_cartesian_args(self):
        from pyhrf.tools import cartesian_combine_args

        varying_args = { 'b' : range(10) }
        fixed_args = {'a' : 3 }
        args_comb = cartesian_combine_args(varying_args, fixed_args)

        r = remote_map(foo, lkwargs=args_comb, mode='local')

        if 0:
            print 'result:'
            print r

    if cfg['parallel-cluster']['enable_unit_test'] == 1:

        def test_remote_map_cluster_many_jobs(self):
            print 'cfg:', cfg['parallel-cluster']['enable_unit_test']
            remote_map(foo, [(5,6)]*10, mode='remote_cluster')


        def test_remote_map_cluster_exception(self):
            self.assertRaises(RemoteException, remote_map,
                              foo_raise, [(1,),(2,)], [{'b':5},{'b':10}],
                              mode='remote_cluster')


        def test_remote_map_cluster_basic(self):
            self._test_remote_map_basic(mode='remote_cluster')


        def test_remote_map_cluster_cartesian_args(self):
            from pyhrf.tools import cartesian_combine_args

            varying_args = { 'b' : range(3) }
            fixed_args = {'a' : 3 }
            args_comb = cartesian_combine_args(varying_args, fixed_args)

            r = remote_map(foo, lkwargs=args_comb, mode='remote_cluster')

            if 0:
                print 'result:'
                print r
