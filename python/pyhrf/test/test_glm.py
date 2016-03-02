# -*- coding: utf-8 -*-

import os.path as op
import unittest
import shutil
import logging

import pyhrf


logger = logging.getLogger(__name__)


class NipyGLMTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_glm_default_real_data(self):

        from pyhrf import FmriData
        from pyhrf.glm import glm_nipy

        fdata = FmriData.from_vol_ui()
        glm_nipy(fdata)

    def test_glm_contrasts(self):

        from pyhrf import FmriData
        from pyhrf.glm import glm_nipy
        cons = {'audio-video': 'audio - video',
                'video-audio': 'video - audio',
                }
        fdata = FmriData.from_vol_ui()
        g, dm, cons = glm_nipy(fdata, contrasts=cons)

    def test_glm_with_files(self):

        output_dir = self.tmp_dir

        bold_name = 'subj0_bold_session0.nii.gz'
        bold_file = pyhrf.get_data_file_name(bold_name)
        tr = 2.4

        paradigm_name = 'paradigm_loc_av.csv'
        paradigm_file = pyhrf.get_data_file_name(paradigm_name)

        mask_name = 'subj0_parcellation.nii.gz'
        mask_file = pyhrf.get_data_file_name(mask_name)

        from pyhrf.glm import glm_nipy_from_files
        glm_nipy_from_files(bold_file, tr, paradigm_file, output_dir,
                            mask_file)

        self.assertTrue(op.exists(output_dir))

    def test_fir_glm(self):

        from pyhrf import FmriData
        from pyhrf.glm import glm_nipy

        fdata = FmriData.from_vol_ui()
        glm_nipy(fdata, hrf_model='FIR', fir_delays=range(10))

    def makeQuietOutputs(self, xmlFile):

        from pyhrf import xmlio
        t = xmlio.from_xml(file(xmlFile).read())
        t.set_init_param('output_dir', None)
        f = open(xmlFile, 'w')
        f.write(xmlio.to_xml(t))
        f.close()

    def test_command_line(self):
        cfg_file = op.join(self.tmp_dir, 'glm.xml')
        cmd = 'pyhrf_glm_buildcfg -o %s -v %d' % (cfg_file,
                                                  logger.getEffectiveLevel())
        logger.debug('cfg file: %s', cfg_file)
        import os
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')
        self.makeQuietOutputs(cfg_file)

        cmd = 'pyhrf_glm_estim -c %s' % cfg_file
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')


def test_suite():
    tests = [unittest.makeSuite(NipyGLMTest)]
    return unittest.TestSuite(tests)


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite())
