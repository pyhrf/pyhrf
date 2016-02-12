# -*- coding: utf-8 -*-

import unittest
import os
import os.path as op
import shutil
import logging

import numpy as np
import numpy.testing as npt

import pyhrf
import pyhrf.parcellation as pm

from pyhrf.ndarray import xndarray
from pyhrf.graph import graph_from_lattice, kerMask3D_6n
from pyhrf import tools


logger = logging.getLogger(__name__)
debug = False


class SpatialTest(unittest.TestCase):

    def test_split_parcel(self):
        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=int)
        mask[1:-1, 1:-1, 1:-1] = 1
        mask_labels = mask[np.where(mask)]
        g = graph_from_lattice(mask, kerMask3D_6n)

        nparcels = 2
        pm.split_parcel(mask_labels, {1: g}, 1, nparcels, inplace=True,
                        verbosity=0)

    def test_balanced_parcellation(self):
        """
        Test if balanced partitioning returns parcels with almost equal
        sizes (tolerance=1) on a 3D rectangular mask
        """
        np.random.seed(56437)
        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=int)
        mask[1:-1, 1:-1, 1:-1] = 1

        nb_parcels = 3
        p = pm.parcellate_balanced_vol(mask, nb_parcels)

        expected_psize = mask.sum() / nb_parcels
        for iparcel in xrange(1, nb_parcels + 1):
            npt.assert_allclose((p == iparcel).sum(), expected_psize, atol=1)

    def test_voronoi_parcellation(self):

        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=int)
        mask[1:-1, 1:-1, 1:-1] = 1

        nb_parcels = 3
        p = pm.parcellate_voronoi_vol(mask, nb_parcels)
        self.assertEqual(len(np.unique(p)), nb_parcels + 1)


from numpy.testing import assert_array_equal


class MeasureTest(unittest.TestCase):

    def setUp(self):
        self.p1 = np.array([[1, 1, 0, 3],
                            [1, 1, 3, 3],
                            [0, 1, 2, 2],
                            [0, 2, 2, 2],
                            [0, 0, 2, 0]], dtype=np.int32)
        self.mask = np.where(self.p1 != 0)
        self.fp1 = self.p1[self.mask]

        self.p2 = np.array([[1, 1, 0, 3],
                            [3, 3, 3, 3],
                            [0, 2, 2, 3],
                            [0, 2, 2, 2],
                            [0, 0, 2, 0]], dtype=np.int32)
        self.fp2 = self.p2[self.mask]

    def test_intersection_matrix(self):

        from pyhrf.cparcellation import compute_intersection_matrix

        im = np.zeros((self.fp1.max() + 1, self.fp2.max() + 1),
                      dtype=np.int32)

        compute_intersection_matrix(self.fp1, self.fp2, im)

        assert_array_equal(im, np.array([[0, 0, 0, 0],
                                         [0, 2, 1, 2],
                                         [0, 0, 5, 1],
                                         [0, 0, 0, 3]], dtype=np.int32),
                           "Intersection graph not OK", 1)

    @unittest.skipIf(not tools.is_importable('munkres'),
                     'munkres (optional dep) is N/A')
    def test_parcellation_distance(self):

        from pyhrf.parcellation import parcellation_dist

        dist, cano_parcellation = parcellation_dist(self.p1, self.p2)
        self.assertEqual(dist, 4)


class ParcellationMethodTest(unittest.TestCase):

    def setUp(self):
        self.p1 = np.array([[1, 1, 3, 3],
                            [1, 0, 0, 0],
                            [0, 0, 2, 2],
                            [0, 2, 2, 2],
                            [0, 0, 2, 4]], dtype=np.int32)

    @unittest.skipIf(not tools.is_importable('sklearn') or
                     not tools.is_importable('munkres'),
                     'scikit-learn or munkres (optional deps) is N/A')
    def test_ward_spatial_scikit(self):
        from pyhrf.parcellation import parcellation_dist, \
            parcellation_ward_spatial
        from pyhrf.graph import graph_from_lattice, kerMask2D_4n

        X = np.reshape(self.p1, (-1, 1))
        graph = graph_from_lattice(np.ones(self.p1.shape), kerMask2D_4n)

        labels = parcellation_ward_spatial(X, n_clusters=5, graph=graph)

        labels = np.reshape(labels, self.p1.shape)
        # +1 because parcellation_dist sees 0 as background
        dist = parcellation_dist(self.p1 + 1, labels + 1)[0]
        self.assertEqual(dist, 0)

    @unittest.skipIf(not tools.is_importable('sklearn') or
                     not tools.is_importable('munkres'),
                     'scikit-learn or munkres (optional deps) is N/A')
    def test_ward_spatial_scikit_with_mask(self):
        from pyhrf.parcellation import parcellation_dist, parcellation_ward_spatial
        from pyhrf.graph import graph_from_lattice, kerMask2D_4n
        from pyhrf.ndarray import expand_array_in_mask

        if debug:
            print 'data:'
            print self.p1
            print ''

        mask = self.p1 != 0
        graph = graph_from_lattice(mask, kerMask2D_4n)

        X = self.p1[np.where(mask)].reshape(-1, 1)

        labels = parcellation_ward_spatial(X, n_clusters=4, graph=graph)

        labels = expand_array_in_mask(labels, mask)
        #+1 because parcellation_dist sees 0 as background:
        dist = parcellation_dist(self.p1 + 1, labels + 1)[0]
        self.assertEqual(dist, 0)


class CmdParcellationTest(unittest.TestCase):

    def setUp(self):
        from pyhrf.ndarray import MRI3Daxes
        self.tmp_dir = pyhrf.get_tmp_path()

        self.p1 = np.array([[[1, 1, 1, 3],
                             [1, 1, 3, 3],
                             [0, 1, 2, 2],
                             [0, 2, 2, 2],
                             [0, 0, 2, 4]]], dtype=np.int32)

        self.p1_fn = op.join(self.tmp_dir, 'p1.nii')
        xndarray(self.p1, axes_names=MRI3Daxes).save(self.p1_fn)

        self.p2 = self.p1 * 4.5
        self.p2_fn = op.join(self.tmp_dir, 'p2.nii')
        xndarray(self.p2, axes_names=MRI3Daxes).save(self.p2_fn)

        self.mask = (self.p1 > 0).astype(np.int32)
        self.mask_fn = op.join(self.tmp_dir, 'mask.nii')
        xndarray(self.mask, axes_names=MRI3Daxes).save(self.mask_fn)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @unittest.skipIf(not tools.is_importable('sklearn') or
                     not tools.is_importable('munkres'),
                     'scikit-learn or munkres (optional deps) is N/A')
    def test_ward_spatial_cmd(self):
        from pyhrf.parcellation import parcellation_dist

        output_file = op.join(self.tmp_dir, 'parcellation_output_test.nii')

        nparcels = 4
        cmd = 'pyhrf_parcellate_glm -m %s %s %s -o %s -v %d ' \
            '-n %d -t ward_spatial ' \
            % (self.mask_fn, self.p1_fn, self.p2_fn, output_file,
               logger.getEffectiveLevel(), nparcels)
        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')
        logger.info('cmd: %s', cmd)

        labels = xndarray.load(output_file).data
        logger.info('labels.dtype:%s', str(labels.dtype))
        dist = parcellation_dist(self.p1, labels)[0]
        logger.info('dist:%d', dist)
        self.assertEqual(dist, 0)

    @unittest.skipIf(not tools.is_importable('sklearn') or
                     not tools.is_importable('munkres'),
                     'scikit-learn or munkres (optional deps) is N/A')
    def test_ward_spatial_real_data(self):
        from pyhrf.glm import glm_nipy_from_files

        fn = 'subj0_parcellation.nii.gz'
        mask_file = pyhrf.get_data_file_name(fn)

        bold = 'subj0_bold_session0.nii.gz'
        bold_file = pyhrf.get_data_file_name(bold)

        paradigm_csv_file = pyhrf.get_data_file_name('paradigm_loc_av.csv')
        output_dir = self.tmp_dir
        output_file = op.join(output_dir,
                              'parcellation_output_test_real_data.nii')

        tr = 2.4
        bet = glm_nipy_from_files(bold_file, tr,
                                  paradigm_csv_file, output_dir,
                                  mask_file, session=0,
                                  contrasts=None,
                                  hrf_model='Canonical',
                                  drift_model='Cosine', hfcut=128,
                                  residuals_model='spherical',
                                  fit_method='ols', fir_delays=[0])[0]

        logger.info('betas_files: %s', ' '.join(bet))

        cmd = 'pyhrf_parcellate_glm -m %s %s -o %s -v %d -n %d '\
            '-t ward_spatial ' \
            % (mask_file, ' '.join(bet), output_file,
               logger.getEffectiveLevel(), 10)

        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')
        logger.info('cmd: %s', cmd)

    def test_voronoi_with_seeds(self):

        import os.path as op
        from pyhrf.ndarray import xndarray
        import pyhrf
        fn = 'subj0_parcellation.nii.gz'
        mask_file = pyhrf.get_data_file_name(fn)

        orientation = ['axial', 'coronal', 'sagittal']
        seeds = xndarray.xndarray_like(
            xndarray.load(mask_file)).reorient(orientation)

        seed_coords = np.array([[24, 35, 8],  # axial, coronal, sagittal
                                [27, 35, 5],
                                [27, 29, 46],
                                [31, 28, 46]])

        seeds.data[:] = 0
        seeds.data[tuple(seed_coords.T)] = 1

        seed_file = op.join(self.tmp_dir, 'voronoi_seeds.nii')
        seeds.save(seed_file, set_MRI_orientation=True)

        output_file = op.join(self.tmp_dir, 'voronoi_parcellation.nii')
        cmd = 'pyhrf_parcellate_spatial %s -m voronoi -c %s -o %s -v %d' \
            % (mask_file, seed_file, output_file, logger.getEffectiveLevel())

        if os.system(cmd) != 0:
            raise Exception('"' + cmd + '" did not execute correctly')

        logger.info('cmd: %s', cmd)

        assert op.exists(output_file)
        parcellation = xndarray.load(output_file)

        n_parcels = len(np.unique(parcellation.data)) - 1

        self.assertEqual(n_parcels, len(seed_coords))
