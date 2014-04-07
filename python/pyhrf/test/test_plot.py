import unittest
import os
import numpy as np
import shutil


import pyhrf
from pyhrf.plot import plot_cub_as_curve, plot_cub_as_image

import matplotlib.pyplot as plt

from pyhrf.ndarray import xndarray

class PlotCommandTest(unittest.TestCase):

    def setUp(self):
        tag = 'subj0_%s.nii.gz'
        self.func_file = pyhrf.get_data_file_name(tag%'bold_session0')
        self.anatomy_file = pyhrf.get_data_file_name(tag%'anatomy')
        self.roi_mask_file = pyhrf.get_data_file_name(tag%'parcellation')

        self.ax_slice = 24
        self.sag_slice = 7
        self.cor_slice = 34

        self.tmp_dir = pyhrf.get_tmp_path() #'./'

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def test_plot_func_slice_func_only(self):
        cmd = 'pyhrf_plot_slice %s -a %d -d %s' \
            %(self.func_file, self.ax_slice, self.tmp_dir)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')

    def test_plot_func_slice_func_only_multiple_slices(self):
        cmd = 'pyhrf_plot_slice %s -a %d -d %s -s %d -c %d' \
            %(self.func_file, self.ax_slice, self.tmp_dir, self.sag_slice,
              self.cor_slice)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


    def test_plot_func_slice_func_roi(self):
        cmd = 'pyhrf_plot_slice %s -a %d -d %s -m %s' \
            %(self.func_file, self.ax_slice, self.tmp_dir, self.roi_mask_file)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


    def test_plot_func_slice_func_roi_anat(self):
        cmd = 'pyhrf_plot_slice %s -a %d -d %s -m %s -y %s' \
            %(self.func_file, self.ax_slice, self.tmp_dir, self.roi_mask_file,
              self.anatomy_file)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


    def test_plot_func_slice_func_roi_anat_multiple_slices(self):
        cmd = 'pyhrf_plot_slice %s -a %d -d %s -m %s -y %s -s %d -c %d' \
            %(self.func_file, self.ax_slice, self.tmp_dir, self.roi_mask_file,
              self.anatomy_file, self.sag_slice, self.cor_slice)
        if os.system(cmd) != 0 :
            raise Exception('"' + cmd + '" did not execute correctly')


    # def test_plot_func_slice_func_highlighted_roi_anat(self):
    #     plot_func_slice(self.func_data, parcellation=self.roi_data,
    #                     anatomy=self.anat_data,
    #                     highlighted_parcels_col={1:'red'})
    #     plt.show()





class PlotFunctionsTest(unittest.TestCase):


    def setUp(self):
        tag = 'subj0_%s.nii.gz'
        func_file = pyhrf.get_data_file_name(tag%'bold_session0')
        anatomy_file = pyhrf.get_data_file_name(tag%'anatomy')
        roi_mask_file = pyhrf.get_data_file_name(tag%'parcellation')

        islice = 24
        cfunc = xndarray.load(func_file).sub_cuboid(time=0,axial=islice)
        cfunc.set_orientation(['coronal', 'sagittal'])
        self.func_data = cfunc.data

        canat = xndarray.load(anatomy_file).sub_cuboid(axial=islice*3)
        canat.set_orientation(['coronal', 'sagittal'])
        self.anat_data = canat.data

        croi_mask = xndarray.load(roi_mask_file).sub_cuboid(axial=islice)
        croi_mask.set_orientation(['coronal', 'sagittal'])
        self.roi_data = croi_mask.data

    #     self.tmp_dir = pyhrf.get_tmp_path()

    # def tearDown(self):
    #     shutil.rmtree(self.tmp_dir)

    def test_plot_cuboid_as_curve(self):
        from pyhrf.ndarray import xndarray
        sh = (10,10,5,3)
        data = np.zeros(sh)
        data[:,:,:,0] = 1.
        data[:,:,:,1] = 2.
        data[:,:,:,2] = 3.
        c1 = xndarray(data, axes_names=['sagittal','coronal','axial','condition'],
                    axes_domains={'condition':['audio1','audio2', 'video']})

        f = plt.figure()
        ax = f.add_subplot(111)

        ori = ['condition', 'sagittal']
        plot_cub_as_curve(c1.sub_cuboid(axial=0, coronal=0).reorient(ori),
                          colors={'audio1':'red', 'audio2':'orange',
                                  'video': 'blue'}, axes=ax)
        if 0:
            plt.show()


    def test_plot_cuboid2d_as_image(self):
        from pyhrf.ndarray import xndarray
        import matplotlib
        sh = (10,3)
        c1 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['sagittal','condition'],
                    axes_domains={'condition':['audio1','audio2', 'video']})

        f = plt.figure()
        ax = f.add_subplot(111)

        ori = ['condition', 'sagittal']
        cm = matplotlib.cm.get_cmap('winter')
        norm = matplotlib.colors.Normalize(vmin=5, vmax=20)
        plot_cub_as_image(c1.reorient(ori), cmap=cm, norm=norm, axes=ax,
                          show_axes=True, show_axis_labels=True,
                          show_colorbar=True,
                          show_tick_labels=False)
        if 0:
            plt.show()


    def test_plot_cuboid1d_as_image(self):
        from pyhrf.ndarray import xndarray
        import matplotlib
        sh = (3,)
        c2 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['condition'],
                    axes_domains={'condition':['audio1','audio2', 'video']})

        f = plt.figure()
        ax = f.add_subplot(111)

        cm = matplotlib.cm.get_cmap('winter')
        norm = matplotlib.colors.Normalize(vmin=0., vmax=3.)
        plot_cub_as_image(c2, cmap=cm, norm=norm, axes=ax,
                          show_axes=True, show_axis_labels=True,
                          show_tick_labels=True)
        if 0:
            plt.show()


    def test_plot_cuboid1d_as_curve(self):
        from pyhrf.ndarray import xndarray
        sh = (3,)
        conds = np.array(['audio1','audio2', 'video'])
        c2 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['condition'],
                    axes_domains={'condition': conds})

        f = plt.figure()
        ax = f.add_subplot(111)

        plot_cub_as_curve(c2, axes=ax, show_axis_labels=True)
        if 0:
            plt.show()


