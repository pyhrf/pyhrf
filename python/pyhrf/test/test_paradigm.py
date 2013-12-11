import os.path as op
import numpy as np
import tempfile
import unittest
import shutil

import pyhrf
import pyhrf.paradigm as pdgm

from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm, BlockParadigm


class ParadigmTest(unittest.TestCase):


    def setUp(self):
        self.onsets_1sess = {'c1' : [np.array([1.2, 5.6, 7.4])],
                             'c2' : [np.array([2.3, 3.6, 6.4, 9.8])]}
        self.durations_1sess = {'c1' : [np.array([3., 3., 3.])],
                                'c2' : [np.array([4., 4., 4., 4.])],}

        self.onsets_2sess = {'c1' : [np.array([1.2, 5.6, 7.4]),
                                     np.array([1.2, 5.6, 7.4,11.2])],
                             'c2' : [np.array([2.3, 3.6, 6.4, 9.8]),
                                     np.array([2.3, 3.6, 6.4, 9.8, 20.7])]}
        self.durations_2sess = {'c1' : [np.array([3., 3., 3.]),
                                        np.array([3., 3., 3., 3.])],
                                'c2' : [np.array([4., 4., 4., 4.]),
                                        np.array([4., 4., 4., 4.,4.])],}


        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


    def test_to_nipy_ER(self):
        """
        Test event-related paradigm
        """

        p = pdgm.Paradigm(self.onsets_1sess)
        p_nipy = p.to_nipy_paradigm()
        assert isinstance(p_nipy, EventRelatedParadigm)

    def test_to_nipy_ER_2sess(self):
        """
        Test event-related paradigm
        """

        p = pdgm.Paradigm(self.onsets_2sess)
        p_nipy = p.to_nipy_paradigm()
        #assert isinstance(p_nipy, dict)
        #assert isinstance(p_nipy[p_nipy.keys()[0]], EventRelatedParadigm)
        assert isinstance(p_nipy, EventRelatedParadigm)

    def test_to_nipy_Block(self):
        """
        Test event-related paradigm
        """

        p = pdgm.Paradigm(self.onsets_1sess, stimDurations=self.durations_1sess)
        p_nipy = p.to_nipy_paradigm()
        assert isinstance(p_nipy, BlockParadigm)

    def test_to_nipy_Block_2sess(self):
        """
        Test event-related paradigm
        """

        p = pdgm.Paradigm(self.onsets_2sess, stimDurations=self.durations_2sess)
        p_nipy = p.to_nipy_paradigm()
        assert isinstance(p_nipy, BlockParadigm)


    def test_to_spm_mat_1st_level(self):
        p = pdgm.Paradigm(self.onsets_2sess, stimDurations=self.durations_2sess)
        p.save_spm_mat_for_1st_level_glm(op.join(self.tmp_dir,
                                                 'conditions.mat'))


    def test_merge_onsets(self):

        o,d = pdgm.merge_onsets(pdgm.onsets_loc, 'audio',
                                durations=pdgm.durations_loc)

        self.assertIsInstance(o['audio'], list)
        self.assertIsInstance(d['audio'], list)
        self.assertEqual(len(o['audio']), 1)
        self.assertIsInstance(o['audio'][0], np.ndarray)
        self.assertEqual(o['audio'][0].ndim, 1)
        self.assertEqual(o['audio'][0].size, 30)

    def test_onsets_loc_av(self):

        o = pdgm.onsets_loc_av

        self.assertIsInstance(o['audio'], list)
        self.assertEqual(len(o['audio']), 1)
        self.assertIsInstance(o['audio'][0], np.ndarray)
        self.assertEqual(o['audio'][0].ndim, 1)
        self.assertEqual(o['audio'][0].size, 30)
