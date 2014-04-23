

import unittest
import numpy as _np
import tempfile

import pyhrf
from pyhrf import get_data_file_name
from pyhrf.tools.io import *
from pyhrf.ndarray import MRI3Daxes, MRI4Daxes

import os
import numpy as np

import shutil

from pyhrf.tools.io import *

class RxCopyTest(unittest.TestCase):

    def setUp(self,):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def _create_tmp_files(self, fns):
        for fn in [op.join(self.tmp_dir,fn) for fn in fns]:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
            open(fn, 'a').close()

    def assert_file_exists(self, fn, test_exists=True):
        if test_exists and not op.exists(fn):
                raise Exception('File %s does not exist' %fn)
        elif not test_exists and op.exists(fn):
            raise Exception('File %s exists' %fn)

    def test_basic(self):
        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['AC0832_anat.nii', 'AC0832_asl.nii', 
                                          'AC0832_bold.nii', 'PK0612_asl.nii',
                                          'PK0612_bold.nii', 'dummy.nii']])
        src = '(?P<subject>[A-Z]{2}[0-9]{4})_(?P<modality>[a-zA-Z]+).nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = 'data.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder)

        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['AC0832/bold/data.nii',
                                 'AC0832/anat/data.nii',
                                 'AC0832/asl/data.nii', 
                                 'PK0612/bold/data.nii', 
                                 'PK0612/asl/data.nii']]:
            self.assert_file_exists(fn)

    def test_advanced(self):
        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['ASL mt_TG_PASL_s004a001.nii', 
                                          'ASL mt_TG_PASL_s008a001.nii', 
                                          'ASL mt_PK_PASL_s064a001.nii', 
                                          'ASL mt_PK_PASL_s003a001.nii']])
        src = 'ASL mt_(?P<subject>[A-Z]{2})_(?P<modality>[a-zA-Z]+)_'\
              's(?P<session>[0-9]{3})a[0-9]{3}.nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = 'ASL_session_{session}.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder)

        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['TG/PASL/ASL_session_004.nii',
                                 'TG/PASL/ASL_session_008.nii',
                                 'PK/PASL/ASL_session_064.nii', 
                                 'PK/PASL/ASL_session_003.nii']]:
            self.assert_file_exists(fn)
        

    def test_missing_tags_dest_folder(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '(?P<subject>[A-Z]{2}[0-9]{2})_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{study}', '{modality}', 
                       '{session}')
        dest_basename = '{subject}.nii'
        self.assertRaisesRegexp(MissingTagError, 
                                "Tags in dest_folder not defined in src: "\
                                "study, session",  rx_copy, 
                                src, src_folder, dest_basename, dest_folder)

    def test_missing_tags_dest_basename(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = '{subject}_{session}.nii'
        self.assertRaisesRegexp(MissingTagError, 
                                "Tags in dest_basename not defined in src: "\
                                "(subject, session)|(session, subject)",  
                                rx_copy, src, src_folder, 
                                dest_basename, dest_folder)

    def test_dry(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = 'data.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder, dry=True)
        fn = op.join(self.tmp_dir, 'export', 'T1', 'data.nii')
        if op.exists(fn):
            raise Exception('File %s should not exist' %fn)


    def test_duplicates_targets(self):
        self._create_tmp_files(['AK98_T1_s01.nii', 'AK98_T1_s02.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+).*nii'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = 'data.nii'
        error_msg = r'Copy is not injective, the following copy ' \
                    'operations have the same destination:\n'   \
                    '.*AK98_T1_s01\.nii\n.*AK98_T1_s02\.nii\n'      \
                    '-> .*export/T1/data.nii'
        self.assertRaisesRegexp(DuplicateTargetError, error_msg,
                                rx_copy, src, src_folder, 
                                dest_basename, dest_folder)

    def test_replacement(self):
        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['ASL mt_TG_PASL_s004a001.nii', 
                                          'ASL mt_TG_T1_s008a001.nii', 
                                          'ASL mt_PK_PASL_s064a001.nii', 
                                          'ASL mt_PK_T1_s003a001.nii']])
        src = 'ASL mt_(?P<subject>[A-Z]{2})_(?P<modality>[a-zA-Z0-9]+)_'\
              's(?P<session>[0-9]{3})a[0-9]{3}.nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = '{modality}_session_{session}.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder,
                replacements=[('T1','anat'),('PASL','aslf')])
        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['TG/aslf/aslf_session_004.nii',
                                 'TG/anat/anat_session_008.nii',
                                 'PK/aslf/aslf_session_064.nii', 
                                 'PK/anat/anat_session_003.nii']]:
            self.assert_file_exists(fn)


    def test_callback(self):
        def filter_odd_session(s,d):
            if (int(d[-5]) % 2) != 0:
                return None
            else:
                return d

        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['ASL mt_TG_PASL_s004a001.nii', 
                                          'ASL mt_TG_T1_s008a001.nii', 
                                          'ASL mt_PK_PASL_s064a001.nii', 
                                          'ASL mt_PK_T1_s003a001.nii']])
        src = 'ASL mt_(?P<subject>[A-Z]{2})_(?P<modality>[a-zA-Z0-9]+)_'\
              's(?P<session>[0-9]{3})a[0-9]{3}.nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = '{modality}_session_{session}.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder,
                replacements=[('T1','anat'),('PASL','aslf')],
                callback=filter_odd_session)
        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['TG/aslf/aslf_session_004.nii',
                                 'TG/anat/anat_session_008.nii',
                                 'PK/aslf/aslf_session_064.nii']]:
            self.assert_file_exists(fn)
        
        self.assert_file_exists('PK/anat/anat_session_003.nii', False)


class NiftiTest(unittest.TestCase):

    def setUp(self):
        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.tmp_dir = tmpDir

    def tearDown(self):
        if 1:
            shutil.rmtree(self.tmp_dir)


    def test_process_history_extension(self):
        nii_fn = pyhrf.get_data_file_name('real_data_vol_4_regions_mask.nii.gz')

        nii_fn_out = op.join(self.tmp_dir, 'proc_ext_test.nii')
        #print 'nii_fn_out:', nii_fn_out
        input_pname = 'dummy_proc_test'
        input_pparams = {'my_param' : 5.5, 'input_file':'/home/blh'}

        append_process_info(nii_fn, input_pname, input_pparams,
                            img_output_fn=nii_fn_out)

        i2,(aff,header) = read_volume(nii_fn_out)
        #print 'Loaded extensions:', header.extensions

        reloaded_pinfo = get_process_info(nii_fn_out)
        self.assertNotEqual(reloaded_pinfo, None)
        self.assertEqual(reloaded_pinfo[0]['process_name'], input_pname)
        self.assertEqual(reloaded_pinfo[0]['process_inputs'], input_pparams)
        self.assertEqual(reloaded_pinfo[0]['process_version'], None)
        self.assertEqual(reloaded_pinfo[0]['process_id'], None)

class DataLoadTest(unittest.TestCase):

    def test_paradigm_csv(self):
        pfn = get_data_file_name('paradigm_loc_av.csv')
        o,d = load_paradigm_from_csv(pfn)
        if 0:
            print 'onsets:'
            print o
            print 'durations:'
            print d


    def test_paradigm_csv2(self):
        pfn = get_data_file_name('paradigm_loc_av.csv')
        o,d = load_paradigm_from_csv(pfn)
        if 0:
            print 'onsets:'
            print o
            print 'durations:'
            print d


    def test_frmi_vol(self):
        """ Test volumic data loading
        """
        boldFn = get_data_file_name('subj0_bold_session0.nii.gz')
        roiMaskFn = get_data_file_name('subj0_parcellation.nii.gz')
        g, b, ss, m, h = load_fmri_vol_data([boldFn, boldFn], roiMaskFn)
        if 0:
            print len(g), g[1]
            print b[1].shape
            print ss
            print m.shape, _np.unique(m)
            print h




class xndarrayIOTest(unittest.TestCase):

    def setUp(self):
        self.cub0 = xndarray(_np.random.rand(10,10))
        self.cub3DVol = xndarray(_np.random.rand(10,10,10),
                               axes_names=MRI3Daxes)
        d4D = _np.zeros((2,2,2,3))
        for c in xrange(3):
            d4D[:,:,:,c] = _np.ones((2,2,2))*(c-2)

        self.cub4DVol = xndarray(d4D, axes_names=['condition']+MRI3Daxes)

        self.cub4DTimeVol = xndarray(_np.random.rand(100,10,10,10),
                               axes_names=['time']+MRI3Daxes)
        self.cubNDVol = xndarray(_np.random.rand(10,2,2,2,3),
                               axes_names=['time']+MRI3Daxes+['condition'],
                               axes_domains={'condition':['audio','video','na']})

        self.tmp_dir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                        dir=pyhrf.cfg['global']['tmp_path'])



    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


    def test_save_nii_3D(self):
        fn = op.join(self.tmp_dir, 'test3D.nii')
        self.cub3DVol.save(fn)

    def test_save_nii_4D(self):
        fn = op.join(self.tmp_dir, 'test4D.nii')
        self.cub4DTimeVol.save(fn)

    def test_save_nii_multi(self):
        c = self.cubNDVol.reorient(MRI4Daxes + ['condition'])
        c.save(op.join(self.tmp_dir, './testND.nii'))

    # def test_cuboid_save_xml(self):
    #     cxml = xndarrayXml.fromxndarray(self.cubNDVol, 'mydata1')
    #     #print pyhrf.xmlio.to_xml(cxml,NumpyXMLHandler(),pretty=True)
    #     cxml.cleanFiles()

    # def test_cuboid_load_xml(self):
    #     cxml = xndarrayXml.fromxndarray(self.cub4DVol, 'mydata2')
    #     sxml = pyhrf.xmlio.to_xml(cxml, NumpyXMLHandler(),pretty=True)
    #     #print sxml
    #     #print 'loading from xml ...'
    #     c = pyhrf.xmlio.from_xml(sxml, NumpyXMLHandler())
    #     #print 'loaded cuboid:'
    #     #print c.cuboid.descrip()
    #     #print c.cuboid.data
    #     #print 'original cuboid:'
    #     #print self.cub4DVol.descrip()
    #     #print self.cub4DVol.data
    #     cxml.cleanFiles()

    # def test_cuboidND_load_xml(self):
    #     #print 'Creating xmlable cuboid ...'
    #     cxml = xndarrayXml.fromxndarray(self.cubNDVol, 'mydata3')
    #     import cPickle
    #     #cPickle.dump(self.cubNDVol, open('./cubNDVol.pck','w'))
    #     #print 'Converting to XML & dumping data ...'
    #     sxml = pyhrf.xmlio.to_xml(cxml, NumpyXMLHandler(),pretty=True)
    #     #print sxml
    #     #print 'loading from xml ...'
    #     c = pyhrf.xmlio.from_xml(sxml, NumpyXMLHandler())
    #     #print 'loaded cuboid:'
    #     #print c.cuboid.descrip()
    #     #cPickle.dump(c.cuboid,open('./cubfromxml.pck','w'))
    #     #print c.cuboid.data
    #     #print 'original cuboid:'
    #     #print self.cubNDVol.descrip()
    #     c.cleanFiles()
    #     #print self.cubNDVol.data

    #     #from ndview import stack_cuboids
    #     #comparxndarray = stack_cuboids([self.cubNDVol,c.cuboid],'case',['orig','xml'])
    #     #cPickle.dump(comparxndarray,open('./cubFromXml.pck','w'))


class FileHandlingTest(unittest.TestCase):

    def setUp(self):

        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_split_ext(self):
        bfn = pyhrf.get_data_file_name('subj0_bold_session0.nii.gz')
        split_ext_safe(bfn)
        #print 's:', s

    def test_split4DVol(self):
        s = 'subj0_bold_session0.nii.gz'
        bfn = pyhrf.get_data_file_name(s)
        bold_files = split4DVol(bfn, output_dir=self.tmp_dir)
        #print bold_files
        i,meta = read_volume(bold_files[0])

        if 0:
            affine, header = meta
            print ''
            print 'one vol shape:'
            print i.shape
            print 'header:'
            pprint.pprint(dict(header))

        for bf in bold_files:
            os.remove(bf)


class GiftiTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                        dir=pyhrf.cfg['global']['tmp_path'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_read_tex_gii_label(self):
        tex_fn = 'real_data_surf_tiny_parcellation.gii'
        tex_fn = pyhrf.get_data_file_name(tex_fn)
        t,tgii = read_texture(tex_fn)

    def test_read_default_real_data_tiny(self):
        mesh_file = pyhrf.get_data_file_name('real_data_surf_tiny_mesh.gii')
        bold_file = pyhrf.get_data_file_name('real_data_surf_tiny_bold.gii')
        fn = 'real_data_surf_tiny_parcellation.gii'
        parcel_file = pyhrf.get_data_file_name(fn)

        cor, tri, mesh_gii = read_mesh(mesh_file)
        bold, bold_gii = read_texture(bold_file)
        parcellation, parcel_gii = read_texture(parcel_file)

        if 0:
            print 'cor:', cor.shape, cor.dtype
            print 'tri:', tri.shape, tri.dtype
            print 'bold:', bold.shape, bold.dtype
            print 'parcellation:', parcellation.shape, parcellation.dtype


    def test_load_fmri_surf_data(self):
        """ Test surfacic data loading
        """
        mesh_file = pyhrf.get_data_file_name('real_data_surf_tiny_mesh.gii')
        bold_file = pyhrf.get_data_file_name('real_data_surf_tiny_bold.gii')
        fn = 'real_data_surf_tiny_parcellation.gii'
        parcel_file = pyhrf.get_data_file_name(fn)

        # boldFn = pyhrf.get_data_file_name('localizer_surface_bold.tex')
        # roiMaskFn = pyhrf.get_data_file_name('roimask_gyrii.tex')
        # meshFn = pyhrf.get_data_file_name('right_hemisphere.mesh')
        #g, b, ss, m, h = load_fmri_surf_data([boldFn, boldFn],  meshFn,
        #                                     roiMaskFn)


        # graph, bold, session_scans, mask, edge lengthes
        #pyhrf.verbose.set_verbosity(3)
        g, b, ss, m, el = load_fmri_surf_data([bold_file, bold_file],
                                              mesh_file,
                                              parcel_file)
        assert len(g) == len(np.unique(m))

        if 0:
            first_parcel = g.keys()[0]
            print len(g), 'g[%d]:'%first_parcel, len(g[first_parcel])
            print 'edge lengthes of roi %d:' %first_parcel
            print el[first_parcel]
            print b[first_parcel].shape
            print ss[0][0],'-',ss[0][-1],',',ss[1][0],'-',ss[1][-1]
            print m.shape, _np.unique(m)


    def test_write_tex_gii_labels(self):
        labels = np.random.randint(0,2,10)
        # print 'labels:', labels.dtype
        # print labels
        tex_fn = op.join(self.tmp_dir, 'labels.gii')
        write_texture(labels, tex_fn)
        t,tgii = read_texture(tex_fn)
        assert t.dtype == labels.dtype
        assert (t == labels).all()
        # print 'labels loaded:', labels.dtype
        # print t

    def test_write_tex_gii_float(self):
        values = np.random.randn(10)
        # print 'values:', values.dtype
        # print values
        tex_fn = op.join(self.tmp_dir, 'float_values.gii')
        write_texture(values, tex_fn)
        t,tgii = read_texture(tex_fn)
        assert t.dtype == values.dtype
        assert np.allclose(t,values)
        # print 'loaded values:', t.dtype
        # print t


    def test_write_tex_gii_time_series(self):
        values = np.random.randn(120,10).astype(np.float32)
        # print 'values:', values.dtype
        # print values
        tex_fn = op.join(self.tmp_dir, 'time_series.gii')
        write_texture(values, tex_fn, intent='time series')
        t,tgii = read_texture(tex_fn)
        assert t.dtype == values.dtype
        assert np.allclose(t,values)
        # print 'loaded values:', t.dtype
        # print t


    def test_write_tex_gii_2D_float(self):
        values = np.random.randn(2,10).astype(np.float32)
        # print 'values:', values.dtype
        # print values
        tex_fn = op.join(self.tmp_dir, 'floats_2d.gii')
        write_texture(values, tex_fn)
        t,tgii = read_texture(tex_fn)
        assert t.dtype == values.dtype
        assert np.allclose(t,values)
        # print 'loaded values:', t.dtype
        # print t


class SPMIOTest(unittest.TestCase):

    def setUp(self):
        pass

    def _test_load_regnames(self, spm_ver):
        spm_file = op.join(pyhrf.get_tmp_path(), 'SPM.mat')
        gunzip(pyhrf.get_data_file_name('SPM_v%d.mat.gz' %spm_ver),
               outFileName=spm_file)
        expected = ['Sn(1) audio*bf(1)', 'Sn(1) video*bf(1)', 
                    'Sn(2) audio*bf(1)', 'Sn(2) video*bf(1)',
                    'Sn(1) constant', 'Sn(2) constant']
        self.assertEqual(spmio.load_regnames(spm_file), expected)
        
    def test_load_regnames_SPM8(self):
        self._test_load_regnames(8)

    def test_load_regnames_SPM5(self):
        self._test_load_regnames(5)

    def test_load_regnames_SPM12(self):
        self._test_load_regnames(12)

    # def test_set_contrasts(self):

    #     spm_mat_fn = "/home/tom/Projects/PyHRF/ShippedData/SPM.mat"
    #     if not op.exists(spm_mat_fn):
    #         return

    #     try:
    #         from scipy.io.mio import loadmat
    #     except:
    #         from scipy.io.matlab import loadmat
    #     from pprint import pprint

    #     d = loadmat(spm_mat_fn)
    #     if 0:
    #         print 'd:'
    #         pprint(d)
    #         print ''

    #     spm = d['SPM']


    #     if isinstance(spm, np.ndarray):
    #         spm = spm[0]
    #     if isinstance(spm, np.ndarray):
    #         spm = spm[0]

    #     if isinstance(spm, np.void):
    #         if len(spm['xCon']) > 0:
    #             cons = spm['xCon'][0]
    #             print [ (c['name'], c['c'].squeeze(), c['STAT']) for c in cons ]
    #         else:
    #             print []
    #     elif isinstance(spm, np.ndarray):
    #         cons = spm.xCon[0]
    #         return [ (c.name, c.c.squeeze(), c.STAT) for c in cons ]
    #     else:
    #         raise Exception("Type of input (%s) is unsupported" \
    #                             %str(spm.__class__))




