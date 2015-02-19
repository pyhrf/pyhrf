# -*- coding: utf-8 -*-

import unittest
import os
import cPickle
import os.path as op
import shutil

import numpy.testing as npt

import pyhrf
import pyhrf.ui.treatment as ptr
import pyhrf.paradigm as ppar

from pyhrf.configuration import cfg
from pyhrf import tools


class CmdInputTest(unittest.TestCase):
    """
    Test extraction of information from the command line to create an
    FmriTreatment
    """

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()  # './'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _test_spm_option_parse(self, spm_ver):
        """
        Test parsing of option "-s SPM.mat" with given SPM version (int)
        """
        spm_file = op.join(pyhrf.get_tmp_path(), 'SPM.mat')
        tools._io._zip.gunzip(pyhrf.get_data_file_name('SPM_v%d.mat.gz' % spm_ver),
                              outFileName=spm_file)

        options = ['-s', spm_file]
        from optparse import OptionParser
        parser = OptionParser()
        ptr.append_common_treatment_options(parser)

        fd = ptr.parse_data_options(parser.parse_args(options)[0])

        self.assertEqual(fd.tr, 2.4)  # nb sessions
        p = fd.paradigm
        # nb sessions
        self.assertEqual(len(p.stimOnsets[p.stimOnsets.keys()[0]]), 2)
        npt.assert_almost_equal(p.stimOnsets['audio'][0],
                                ppar.onsets_loc_av['audio'][0])
        npt.assert_almost_equal(p.stimOnsets['audio'][1],
                                ppar.onsets_loc_av['audio'][0])
        npt.assert_almost_equal(p.stimOnsets['video'][1],
                                ppar.onsets_loc_av['video'][0])

    def test_spm5_option_parse(self):
        """
        Test parsing of option "-s SPM.mat" (SPM5)
        """
        self._test_spm_option_parse(5)

    def test_spm8_option_parse(self):
        """
        Test parsing of option "-s SPM.mat" (SPM8)
        """
        self._test_spm_option_parse(8)

    def test_spm12_option_parse(self):
        """
        Test parsing of option "-s SPM.mat" (SPM12)
        """
        self._test_spm_option_parse(12)


class TreatmentTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()  # './'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_default_treatment(self):

        # pyhrf.verbose.set_verbosity(4)
        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run()

    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_parallel_local(self):

        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run(parallel='local', n_jobs=2)

    def test_pickle_treatment(self):
        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        cPickle.loads(cPickle.dumps(t))

    def test_sub_treatment(self):

        t = ptr.FMRITreatment(output_dir=self.tmp_dir)
        t.enable_draft_testing()
        sub_ts = t.split()
        for sub_t in sub_ts:
            sub_t.run()

    def test_jde_estim_from_treatment_pck(self):

        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        sub_ts = t.split()
        sub_t_fn = op.join(self.tmp_dir, 'treatment.pck')
        fout = open(sub_t_fn, 'w')
        cPickle.dump(sub_ts[0], fout)
        fout.close()
        cmd = 'pyhrf_jde_estim  -t %s' % sub_t_fn
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')

    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_default_treatment_parallel_local(self):
        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t.run(parallel='local')

    @unittest.skipIf(not tools.is_importable('joblib'),
                     'joblib (optional dep) is N/A')
    def test_default_jde_cmd_parallel_local(self):
        t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None)
        t.enable_draft_testing()
        t_fn = op.join(self.tmp_dir, 'treatment.pck')
        fout = open(t_fn, 'w')
        cPickle.dump(t, fout)
        fout.close()
        cmd = 'pyhrf_jde_estim -t %s -x local' % t_fn
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')

    def test_default_treatment_parallel_LAN(self):
        # pyhrf.verbose.set_verbosity(1)
        if cfg['parallel-LAN']['enable_unit_test'] == 1:
            t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None,
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
            bad_hosts = [h for r, h in zip(res, cfg['parallel-LAN']['hosts'])
                         if r == 'no']

            if len(bad_hosts) > 0:
                raise Exception('Remote dir %s is not writable from the '
                                'following hosts:\n %s' % '\n'.join(bad_hosts))

        else:
            print 'LAN testing is off '\
                '([parallel-LAN][enable_unit_test] = 0 in config.cfg'

    def test_default_treatment_parallel_cluster(self):
        # pyhrf.verbose.set_verbosity(1)
        if cfg['parallel-cluster']['enable_unit_test'] == 1:
            t = ptr.FMRITreatment(make_outputs=False, result_dump_file=None,
                                  output_dir=self.tmp_dir)
            t.enable_draft_testing()
            t.run(parallel='cluster')
        else:
            print 'Cluster testing is off '\
                '([cluster-LAN][enable_unit_test] = 0 in config.cfg'
