import unittest
import pyhrf
import os
import cPickle
import os.path as op

from pyhrf.ui.treatment import FMRITreatment
from pyhrf.configuration import cfg
from pyhrf import tools

import shutil

class TreatmentTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path() #'./'

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def test_default_treatment(self):

        #pyhrf.verbose.set_verbosity(4)
        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run()
    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_parallel_local(self):

        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run(parallel='local', n_jobs=2)

    def test_pickle_treatment(self):
        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        cPickle.loads(cPickle.dumps(t))

    def test_sub_treatment(self):

        t = FMRITreatment(output_dir=self.tmp_dir)
        t.enable_draft_testing()
        sub_ts = t.split()
        for sub_t in sub_ts:
            sub_t.run()

    def test_jde_estim_from_treatment_pck(self):

        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        sub_ts = t.split()
        sub_t_fn = op.join(self.tmp_dir, 'treatment.pck')
        fout = open(sub_t_fn, 'w')
        cPickle.dump(sub_ts[0], fout)
        fout.close()
        cmd = 'pyhrf_jde_estim  -t %s' %sub_t_fn
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')

    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_default_treatment_parallel_local(self):
        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run(parallel='local')

    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_default_jde_cmd_parallel_local(self):
        t = FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t_fn = op.join(self.tmp_dir, 'treatment.pck')
        fout = open(t_fn, 'w')
        cPickle.dump(t, fout)
        fout.close()
        cmd = 'pyhrf_jde_estim -t %s -x local' %t_fn
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')


    def test_default_treatment_parallel_LAN(self):
        #pyhrf.verbose.set_verbosity(1)
        if cfg['parallel-LAN']['enable_unit_test'] == 1:
            t = FMRITreatment(make_outputs=False, result_dump_file=None,
                              output_dir=self.tmp_dir)
            t.enable_draft_testing()
            t.run(parallel='LAN')
        else:
            print 'LAN testing is off '\
              '([parallel-LAN][enable_unit_test] = 0 in ~/.pyhrf/config.cfg'

    def test_remote_dir_writable(self):
        if cfg['parallel-LAN']['enable_unit_test'] == 1:
            from pyhrf import grid
            lhosts = cfg['parallel-LAN']['hosts'].split(',')
            res = grid.remote_dir_is_writable(cfg['parallel-LAN']['user'],
                                              lhosts,
                                              cfg['parallel-LAN']['remote_path'])
            bad_hosts = [h for r,h in zip(res, cfg['parallel-LAN']['hosts']) \
                         if r=='no']

            if len(bad_hosts) > 0:
                raise Exception('Remote dir %s is not writable from the '\
                                'following hosts:\n %s' %'\n'.join(bad_hosts))

        else:
            print 'LAN testing is off '\
              '([parallel-LAN][enable_unit_test] = 0 in config.cfg'


    def test_default_treatment_parallel_cluster(self):
        #pyhrf.verbose.set_verbosity(1)
        if cfg['parallel-cluster']['enable_unit_test'] == 1:
            t = FMRITreatment(make_outputs=False, result_dump_file=None,
                              output_dir=self.tmp_dir)
            t.enable_draft_testing()
            t.run(parallel='cluster')
        else:
            print 'Cluster testing is off '\
            '([cluster-LAN][enable_unit_test] = 0 in config.cfg'
