# -*- coding: utf-8 -*-


import unittest
import os
import os.path as op
import tempfile
import string
import shutil

import pyhrf
from pyhrf import FMRISessionVolumicData, FmriData, FMRISessionSimulationData, \
    FMRISessionSurfacicData
from pyhrf.jde.models import availableModels


from pyhrf import xmlio
from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
from pyhrf.xmlio import read_xml #, write_xml

from pyhrf.ui.jde import JDEMCMCAnalyser

class TreatmentCommandTest(unittest.TestCase):

    def setUp(self):
        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _test_buildcfg(self, cmd, paradigm, data_type, data_scenario,
                       **other_options):
        cfg_file = op.join(self.tmp_dir, 'pyhrf_cfg.xml')
        scmd = '%s -p %s -d %s -t %s -o %s' \
            %(cmd, paradigm, data_type, data_scenario, cfg_file)
        for k,v in other_options.iteritems():
            scmd += ' -%s %s' %(k,v)
        if os.system(scmd) != 0:
            raise Exception('"' + scmd + '" did not execute correctly')
        if not op.exists(cfg_file):
            raise Exception('XML cfg file not created by cmd %s' %scmd)

        return cfg_file


    def test_buildcfg_jde_loc_vol_default(self):
        cfg_file = self._test_buildcfg(cmd='pyhrf_jde_buildcfg',
                                       paradigm='loc',
                                       data_type='volume',
                                       data_scenario='default')

        stim_names = sorted(pyhrf.paradigm.onsets_loc.keys())
        t = self._check_treatment_data(cfg_file, mask_shape=(53, 63, 46),
                                       bold_shape=(125, 1272),
                                       stim_names=stim_names)

        self.assertTrue( isinstance(t.analyser, JDEMCMCAnalyser) )


    def test_buildcfg_jde_locav_vol_default(self):
        cfg_file = self._test_buildcfg(cmd='pyhrf_jde_buildcfg',
                                       paradigm='loc_av',
                                       data_type='volume',
                                       data_scenario='default')

        stim_names = sorted(pyhrf.paradigm.onsets_loc_av.keys())
        t = self._check_treatment_data(cfg_file, mask_shape=(53, 63, 46),
                                       bold_shape=(125, 1272),
                                       stim_names=stim_names)

        self.assertTrue( isinstance(t.analyser, JDEMCMCAnalyser) )


    def test_buildcfg_jde_locav_surf_default(self):
        cfg_file = self._test_buildcfg(cmd='pyhrf_jde_buildcfg',
                                       paradigm='loc_av',
                                       data_type='surface',
                                       data_scenario='default')

        stim_names = sorted(pyhrf.paradigm.onsets_loc_av.keys())
        t = self._check_treatment_data(cfg_file, mask_shape=(63,),
                                       bold_shape=(128, 63),
                                       stim_names=stim_names)

        self.assertTrue( isinstance(t.analyser, JDEMCMCAnalyser) )


    def test_buildcfg_contrasts(self):
        if 0 and pyhrf.__usemode__ == pyhrf.DEVEL:
            raise NotImplementedError("TODO: buildcfg with different default "\
                                          " scenarios for contrasts")


    def _check_treatment_data(self, cfg_file, mask_shape, bold_shape, stim_names):
        treatment = read_xml(cfg_file)
        self.assertEqual(treatment.data.roiMask.shape, mask_shape)
        self.assertEqual(treatment.data.bold.shape, bold_shape )
        loaded_stim_names = sorted(treatment.data.paradigm.get_stimulus_names())
        self.assertEqual(loaded_stim_names, stim_names)
        return treatment

    def setDummyInputData(self, xmlFile):

        f = open(xmlFile,'r')
        xml = f.read()
        t = xmlio.fromXML(xml)
        if t.data.data_type == 'volume':
            dataFn = pyhrf.get_data_file_name('dummySmallBOLD.nii.gz')
            maskFn = pyhrf.get_data_file_name('dummySmallMask.nii.gz')
            sd = FMRISessionVolumicData(bold_file=dataFn)
            t.set_init_param('fmri_data',
                             FmriData.from_vol_ui(mask_file=maskFn,
                                                  sessions_data=[sd]))

        elif t.data.data_type == 'surface':
            fn = 'real_data_surf_tiny_bold.gii'
            dataFn = pyhrf.get_data_file_name(fn)
            fn = 'real_data_surf_tiny_parcellation.gii'
            maskFn = pyhrf.get_data_file_name(fn)
            fn = 'real_data_surf_tiny_mesh.gii'
            meshFn = pyhrf.get_data_file_name(fn)
            sd = FMRISessionSurfacicData(bold_file=dataFn)
            t.set_init_param('fmri_data',
                             FmriData.from_surf_ui(mask_file=maskFn,
                                                   mesh_file=meshFn,
                                                   sessions_data=[sd]))
        else:
            raise Exception('Unsupported class ... todo')

        f = open(xmlFile, 'w')
        f.write(xmlio.toXML(t, handler=NumpyXMLHandler()))
        f.close()


    def setSimulationData(self, xmlFile, simu_file):

        f = open(xmlFile,'r')
        xml = f.read()
        t = xmlio.fromXML(xml)
        sd = FMRISessionSimulationData(simulation_file=simu_file)
        t.set_init_param('fmri_data', FmriData.from_simu_ui(sessions_data=[sd]))

        f = open(xmlFile, 'w')
        sxml = xmlio.toXML(t, handler=NumpyXMLHandler())
        # print 'sxml:'
        # print sxml
        f.write(sxml)
        f.close()


    def makeQuietOutputs(self, xmlFile):

        #print 'makeQuietOutputs ...'
        #print 'xmlFile:', xmlFile
        t = xmlio.fromXML(file(xmlFile).read())
        t.set_init_param('output_dir', None)
        f = open(xmlFile, 'w')
        f.write(xmlio.toXML(t, handler=NumpyXMLHandler()))
        f.close()

    def testDetectEstimDefault(self):
        from pyhrf.ui.jde import DEFAULT_CFG_FILE
        cfg_file = op.join(self.tmp_dir, DEFAULT_CFG_FILE)
        cmd = 'pyhrf_jde_buildcfg -o %s -n 3' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')
        if not os.path.exists(cfg_file):
            raise Exception('Default cfg file was not created')

        self.makeQuietOutputs(cfg_file)
        # if pyhrf.__usemode__ == 'enduser':
        #     self.makeQuickJDE(cfg_file)

        cmd = 'pyhrf_jde_estim -s -c %s' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


    # In-dev models:
    # if pyhrf.cfg['global']['use_mode'] == 'devel':
    #     def test_ANSGGMS_vol_cmd(self):
    #         self._testJDEModelCmd('ANSGGMS')

    #     def test_WNSGGMH_vol_cmd(self):
    #         self._testJDEModelCmd('WNSGGMH')

    #     def test_WNSGGGMS_vol_cmd(self):
    #         self._testJDEModelCmd('WNSGGGMS')

    #     def test_WNSGGGMS_surf_cmd(self):
    #         self._testJDEModelCmd('WNSGGGMS', datatype='surface')

    #     def test_WNSGGMH_surf_cmd(self):
    #         self._testJDEModelCmd('WNSGGMH', datatype='surface')

    #     def test_ANSGGMS_surf_cmd(self):
    #         self._testJDEModelCmd('ANSGGMS', datatype='surface')

    def test_WNSGGMS_surf_cmd(self):
        self._testJDEModelCmd('WNSGGMS', datatype='surface')

    def test_WNSGGMS(self):
        self._testJDEModelCmd('WNSGGMS')



    def _testDetectEstimAllModels(self):
        from pyhrf.ui.jde import DEFAULT_CFG_FILE
        #print 'testing all models ...'
        cfg_file = op.join(self.tmp_dir, DEFAULT_CFG_FILE)

        errors = []
        for modelLabel in availableModels.keys():
            pyhrf.verbose(1,'Trying model :', modelLabel)
            cmd = 'pyhrf_jde_buildcfg -l %s -n 3' %modelLabel
            if os.system(cmd) != 0 :
                errors.append('"' + cmd + '" did not execute correctly')
                continue
            if not os.path.exists(cfg_file):
                errors.append('Model %s - from cmd "%s" -> default cfg file'\
                              'was not created' %(modelLabel,cmd))
                continue

            self.setDummyInputData(cfg_file)
            # if pyhrf.__usemode__ == 'enduser':
            #     self.makeQuickJDE(cfg_file)
            self.makeQuietOutputs(cfg_file)
            cmd = 'pyhrf_jde_estim -s -c %s' %cfg_file
            if os.system(cmd) != 0 :
                errors.append('Model %s - "%s" did not execute correctly' \
                              %(modelLabel, cmd))
            # if not os.path.exists(DEFAULT_DUMP_FILE):
            #     os.remove(cfg_file)
            #     errors.append('Model %s - from cmd %s -> %s was not created'\
            #                   %(modelLabel, cmd, DEFAULT_DUMP_FILE))
            #     continue
            # else:
            #     r = self.checkAnalyserResults(DEFAULT_DUMP_FILE)
            #     if r is not None:
            #         err = 'Model %s - from "%s" \n'\
            #               ' -> error within treatment:\n%s'\
            #               %(modelLabel, cmd, r)
            #         errors.append(modelLabel+' ->\n'+r)
            #         os.remove(DEFAULT_DUMP_FILE)
            #         os.remove(cfg_file)
            #         continue
            # os.remove(DEFAULT_DUMP_FILE)
            os.remove(cfg_file)
            pyhrf.verbose(1, 'Model %s OK' %modelLabel)

        if errors != []:
            pyhrf.verbose(1, 'Model %s NOT ok' %modelLabel)
            e = Exception(string.join(['']+errors,'\n'))
            raise e


    def _testJDEModelCmd(self, modelLabel, datatype='volume'):

        from pyhrf.ui.jde import DEFAULT_CFG_FILE
        cfg_file = op.join(self.tmp_dir, DEFAULT_CFG_FILE)
        pyhrf.verbose(1,'Trying model: %s, datatype: %s' \
                          %(modelLabel, datatype))
        cmd = 'pyhrf_jde_buildcfg -l %s -d %s -o %s' \
            %(modelLabel, datatype, cfg_file)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')

        if not os.path.exists(cfg_file):
            raise Exception('Model %s - from cmd "%s" -> default cfg file'\
                                'was not created' %(modelLabel,cmd))

        self.setDummyInputData(cfg_file)
        # if pyhrf.__usemode__ == 'enduser':
        #     self.makeQuickJDE(cfg_file)
        self.makeQuietOutputs(cfg_file)
        cmd = 'pyhrf_jde_estim -s -c %s' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('Model %s, datatype %s - "%s" did not execute correctly' \
                                %(modelLabel, datatype, cmd))

        # if not os.path.exists(DEFAULT_DUMP_FILE):
        #     os.remove(cfg_file)
        #     raise Exception('Model %s - from cmd %s -> %s was not created'\
        #                         %(modelLabel, cmd, DEFAULT_DUMP_FILE))
        # else:
        #     r = self.checkAnalyserResults(DEFAULT_DUMP_FILE)
        #     if r is not None:
        #         err = 'Model %s - from "%s" \n'\
        #             ' -> error within treatment:\n%s'\
        #             %(modelLabel, cmd, r)
        #         raise Exception(modelLabel+' ->\n'+r)
        #         os.remove(DEFAULT_DUMP_FILE)
        #         os.remove(cfg_file)

        # os.remove(DEFAULT_DUMP_FILE)
        #print 'Model %s OK' %modelLabel


    def testHrfEstim(self):
        from pyhrf.ui.rfir_ui import DEFAULT_CFG_FILE
        cfg_file = op.join(self.tmp_dir, DEFAULT_CFG_FILE)
        cmd = 'pyhrf_rfir_buildcfg -o %s -n 3' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')
        if not os.path.exists(cfg_file):
            raise Exception('cmd "%s" -> default cfg file'\
                                'was not created' %(cmd))

        self.setDummyInputData(cfg_file)
        if pyhrf.__usemode__ == 'enduser':
            self.makeQuickHrfEstim(cfg_file)
        self.makeQuietOutputs(cfg_file)
        cmd = 'pyhrf_rfir_estim -s -c %s' %cfg_file
        if os.system(cmd) != 0 :
            raise Exception('"%s" did not execute correctly' \
                                %(cmd))
        # if not os.path.exists(DEFAULT_DUMP_FILE):
        #     os.remove(cfg_file)
        #     raise Exception('cmd %s -> %s was not created'\
        #                         %(cmd, DEFAULT_DUMP_FILE))
        # else:
        #     r = self.checkAnalyserResults(DEFAULT_DUMP_FILE)
        #     if r is not None:
        #         os.remove(DEFAULT_DUMP_FILE)
        #         os.remove(cfg_file)
        #         raise Exception('"%s" -> error within treatment:\n%s' %(cmd, r))
        # os.remove(DEFAULT_DUMP_FILE)
        #os.remove(DEFAULT_CFG_FILE)

        # def _testBayesFactorCalc(self):
        #     print 'testBF !'
        #     cmd = 'pyhrf_jde_buildcfg -l WNSGGMS_BF'
        #     if os.system(cmd) != 0:
        #         raise 'raise "%s" did not execute correctly' %cmd

        #     self.setDummyInputData(DEFAULT_CFG_FILE)
        #     cmd = 'pyhrf_jde_estim -s'
        #     if os.system(cmd) != 0:
        #         raise 'raise "%s" did not execute correctly' %cmd

        #     cmd = 'pyhrf_bayesfactor %s %s'%(DEFAULT_DUMP_FILE,DEFAULT_DUMP_FILE)
        #     if os.system(cmd) != 0:
        #         if os.path.exists(DEFAULT_DUMP_FILE):
        #             os.remove(DEFAULT_DUMP_FILE)
        #         raise 'raise "%s" did not execute correctly' %cmd

        #     if os.path.exists(DEFAULT_DUMP_FILE):
        #         os.remove(DEFAULT_DUMP_FILE)


try:
    from subprocess import check_output
except:
    pass

class MiscCommandTest(unittest.TestCase):


    def setUp(self):
        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

        self.files = [
            (op.join(self.tmp_dir,'subject1','fmri'),
             ('paradigm.csv',)
            ),
            (op.join(self.tmp_dir,'subject1','fmri','run1'),
             ('bold_scan_0001.nii', 'bold_scan_0002.nii', 'bold_scan_0003.nii',)
            ),
            (op.join(self.tmp_dir,'subject1','fmri','run2'),
             ('bold_scan_0001.nii', 'bold_scan_0002.nii', 'bold_scan_0003.nii',)
            ),
            (op.join(self.tmp_dir,'subject1','t1mri'),
             ('anatomy.img', 'anatomy.hdr')
            ),
            (op.join(self.tmp_dir,'subject1','fmri','analysis'),
             ('analysis_result_1.nii','analysis_result_2.csv',
              'analysis_summary.txt')
            ),
            ]

        for d,fns in self.files:
            os.makedirs(d)
            for fn in fns:
                open(op.join(d,fn),'w').close()


    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_gls_default(self):

        output = check_output(['pyhrf_gls', self.tmp_dir])

        expected_ouput="""%s:
%s/subject1
""" %(self.tmp_dir, self.tmp_dir)

        self.assertEqual(output,expected_ouput)


    def test_gls_recursive(self):
        cmd = ['pyhrf_gls', '-r', self.tmp_dir]
        output = check_output(cmd)

        pyhrf.verbose(1, 'output:')
        pyhrf.verbose(1, output)
        expected_ouput="""%s:
%s/subject1:
%s/subject1/fmri:
paradigm.csv

%s/subject1/fmri/analysis:
analysis_result_1.nii
analysis_result_2.csv
analysis_summary.txt

%s/subject1/fmri/run1:
bold_scan_[1...3].nii

%s/subject1/fmri/run2:
bold_scan_[1...3].nii

%s/subject1/t1mri:
anatomy.{hdr,img}

""" %((self.tmp_dir,)*7)
        if output != expected_ouput:
            raise Exception('Output of command %s is not as expected.\n'\
                            'Output is:\n%sExcepted:\n%s'
                            %(' '.join(cmd), output, expected_ouput))


    def test_gls_recursive_group(self):
        """ test pyhrf_gls command in recursive mode with file groups specified
        by a regular expression """
        group_re = '(?P<group_name>analysis)_.*'
        cmd = ['pyhrf_gls', '-r', '--colors=off', '-g', group_re,
               self.tmp_dir]
        output = check_output(cmd)
        #print 'output:'
        #print output
        expected_ouput="""%s:
%s/subject1:
%s/subject1/fmri:
paradigm.csv

%s/subject1/fmri/analysis:
analysis...

%s/subject1/fmri/run1:
bold_scan_[1...3].nii

%s/subject1/fmri/run2:
bold_scan_[1...3].nii

%s/subject1/t1mri:
anatomy.{hdr,img}

""" %((self.tmp_dir,)*7)
        if output != expected_ouput:
            raise Exception('Output of command %s is not as expected.\n'\
                            'Output is:\n%sExcepted:\n%s'
                            %(' '.join(cmd), output, expected_ouput))





