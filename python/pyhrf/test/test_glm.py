import unittest
import pyhrf
import os.path as op
import shutil


class NipyGLMTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


    # def _simulate_bold(self):
    #     boldf, tr, paradigmf, maskf = simulate_bold(output_dir=self.tmp_dir)
    #     glm_nipy_from_files(boldf, tr, paradigmf, output_dir=output_dir,
    #                         hcut=0, drift_model='Blank', mask_file=maskf)


    def test_glm_default_real_data(self):

        from pyhrf import FmriData
        from pyhrf.glm import glm_nipy

        #pyhrf.verbose.setVerbosity(3)
        fdata = FmriData.from_vol_ui()
        # print 'fdata:'
        # print fdata.getSummary()
        glm_nipy(fdata)


    def test_glm_contrasts(self):

        from pyhrf import FmriData
        from pyhrf.glm import glm_nipy
        cons = {'audio-video': 'audio - video',
                'video-audio': 'video - audio',
                }
        #pyhrf.verbose.setVerbosity(3)
        fdata = FmriData.from_vol_ui()
        # print 'fdata:'
        # print fdata.getSummary()
        g, dm, cons = glm_nipy(fdata, contrasts=cons)


    def test_glm_with_files(self):

        #pyhrf.verbose.setVerbosity(1)
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
        
        #pyhrf.verbose.setVerbosity(3)
        fdata = FmriData.from_vol_ui()
        # print 'fdata:'
        # print fdata.getSummary()
        glm_nipy(fdata, hrf_model='FIR', fir_delays=range(10))



    def test_command_line(self):
        cfg_file = op.join(self.tmp_dir, 'glm.xml')
        cmd = 'pyhrf_glm_buildcfg -o %s' %(cfg_file)
        import os
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')

        cmd = 'pyhrf_glm_estim -c %s' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


def test_suite():
    tests = [unittest.makeSuite(NipyGLMTest)]
    return unittest.TestSuite(tests)


if __name__== '__main__':
    #unittest.main(argv=['pyhrf.test_glm'])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite())
