

import unittest
import numpy as np
import numpy.testing as npt
import os.path as op

from pyhrf.ndarray import *
import shutil
from pyhrf.tools import add_suffix

debug = False

class xndarrayTest(unittest.TestCase):

    def setUp(self):
        self.arr1d = np.array([5,6,10])
        self.arr1dNames = ['time']
        self.arr1dDom = {'time' : np.array(['jan','feb','mar'])}
        self.arr1dLabel = 'growth'

        self.sh3d = (4,5,6)
        self.arr3d = np.arange(np.prod(self.sh3d)).reshape(self.sh3d)

        self.arr3dNames = ['axial','coronal','sagittal']
        self.arr3dDom = {'axial' : np.arange(self.sh3d[0])*3,
                          'coronal' : np.arange(self.sh3d[1])*3,
                         'sagittal' : np.arange(self.sh3d[2])*3,
                         }
        self.arr3dLabel = 'intensity_value'

        self.mask2d = np.array([ [0,0,0,0,0],
                                 [0,0,0,0,0],
                                 [0,0,1,1,0],
                                 [0,1,1,0,0],
                                 [0,0,1,1,0],
                                 [0,1,1,1,0],
                                 [0,0,0,0,0], ], dtype=int)

        self.mask2d_axes = ['x','y']

        nb_positions = self.mask2d.sum()
        conditions = np.array(['audio','video','computation','sentence'])
        nb_conditions = len(conditions)
        self.arr_flat_sh = (nb_positions, nb_conditions)

        sh = self.arr_flat_sh
        self.arr_flat = np.arange(np.prod(sh)).reshape(sh)
        self.arr_flat_names = ['position', 'condition']
        self.arr_flat_domains = {'condition':conditions}

        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init(self):
        c = xndarray(self.arr1d, self.arr1dNames, self.arr1dDom,
                   self.arr1dLabel)
        if debug:
            print 'c:'
            print c.descrip()

    def test_squeeze(self):

        c = xndarray(np.arange(10).reshape(1,5,1,2,1),
                   ['a','b','c','d','e'], {'a':[2], 'b':np.arange(5)*3, 'e':[5]})

        cs = c.squeeze()

        self.assertEqual(cs.axes_names, ['b','d'])

        cs2 = c.squeeze_all_but(['a','b','c'])

        self.assertEqual(cs2.axes_names, ['a','b','c','d'])


    def test_to_latex_1d(self):
        c = xndarray(self.arr1d, self.arr1dNames, self.arr1dDom,
                   self.arr1dLabel)

        if debug:
            print 'c:'
            print c.descrip()
            print self.arr1d

        result = c.to_latex(col_axes=['time'], val_fmt='%d')

        good_result = \
          '\\begin{tabular}{c c c}\n' \
          '\multicolumn{3}{c}{time}\\\\\n' \
          'jan & feb & mar\\\\\n' \
          '5 & 6 & 10\\\\\n' \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)


    def test_to_latex_3d(self):

        sh = (2,3,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        if debug:
            print 'c:'
            print c.descrip()

        result = c.to_latex(col_axes=['mod','time','subj'],
                            val_fmt='%d')

        good_result = \
          '\\begin{tabular}{' + ' '.join(['c']*24) + '}\n' \
          '\multicolumn{24}{c}{mod}\\\\\n' \
          '\multicolumn{12}{c}{m1} & \multicolumn{12}{c}{m2}\\\\\n' \
          '\multicolumn{24}{c}{time}\\\\\n' \
          '\multicolumn{4}{c}{0.0} & \multicolumn{4}{c}{0.5} & '\
          '\multicolumn{4}{c}{1.0} & \multicolumn{4}{c}{0.0} & '\
          '\multicolumn{4}{c}{0.5} & \multicolumn{4}{c}{1.0}\\\\\n' \
          '\multicolumn{24}{c}{subj}\\\\\n' \
          's1 & s2 & s3 & s4 & s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4 '\
          '& s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4\\\\\n' + \
          ' & '.join([str(i) for i in np.arange(np.prod(sh), dtype=int)]) + \
           '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)


        result = c.to_latex(col_axes=['mod','subj'], row_axes=['time'],
                            hval_fmt={'time':'%1.1f'}, val_fmt='%d')

        l1 = range(4) + range(12, 16)
        l2 = range(4,8) + range(16, 20)
        l3 = range(8,12) + range(20, 24)
        good_result = \
          '\\begin{tabular}{c c | ' + ' '.join(['c']*8) + '}\n' \
          '&&\multicolumn{8}{c}{mod}\\\\\n' \
          '&&\multicolumn{4}{c}{m1} & \multicolumn{4}{c}{m2}\\\\\n' \
          '&&\multicolumn{8}{c}{subj}\\\\\n' \
          '&&s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4\\\\\n' + \
          '\multirow{3}{*}{time} & 0.0 & ' + \
          ' & '.join([str(i) for i in l1]) + '\\\\\n' + \
          ' & 0.5 & ' + ' & '.join([str(i) for i in l2]) + '\\\\\n' + \
          ' & 1.0 & ' + ' & '.join([str(i) for i in l3]) + '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)


        result = c.to_latex(col_axes=['subj'], row_axes=['mod','time'],
                            hval_fmt={'time':'%1.1f'}, val_fmt='%d')

        lines = [range(i,i+4) for i in range(0,24,4)]
        #print 'lines:', lines
        good_result = \
          '\\begin{tabular}{c c c c | ' + ' '.join(['c']*4) + '}\n' \
          '&&&&\multicolumn{4}{c}{subj}\\\\\n' \
          '&&&&s1 & s2 & s3 & s4\\\\\n' + \
          '\multirow{6}{*}{mod} & \multirow{3}{*}{m1} & '\
           '\multirow{6}{*}{time} & 0.0 & ' + \
           ' & '.join([str(i) for i in lines[0]]) + '\\\\\n' + \
          ' &  &  & 0.5 & ' + ' & '.join([str(i) for i in lines[1]]) + '\\\\\n' + \
          ' &  &  & 1.0 & ' + ' & '.join([str(i) for i in lines[2]]) + '\\\\\n' + \
          ' & \multirow{3}{*}{m2} &  & 0.0 & ' + \
           ' & '.join([str(i) for i in lines[3]]) + '\\\\\n' + \
           ' &  &  & 0.5 & ' + ' & '.join([str(i) for i in lines[4]]) + '\\\\\n'+ \
           ' &  &  & 1.0 & ' + ' & '.join([str(i) for i in lines[5]]) + '\\\\\n'+ \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)


    def test_to_latex_3d_join_style(self):

        sh = (2,3,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        if debug:
            print 'c:'
            print c.descrip()


        result = c.to_latex(col_axes=['mod','subj'], row_axes=['time'],
                            header_styles={'mod':'join','time':'join'},
                            val_fmt='%d',
                            hval_fmt={'time':'%1.1f'})

        l1 = range(4) + range(12, 16)
        l2 = range(4,8) + range(16, 20)
        l3 = range(8,12) + range(20, 24)
        good_result = \
          '\\begin{tabular}{c | ' + ' '.join(['c']*8) + '}\n' \
          '&\multicolumn{4}{c}{mod=m1} & \multicolumn{4}{c}{mod=m2}\\\\\n' \
          '&\multicolumn{8}{c}{subj}\\\\\n' \
          '&s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4\\\\\n' + \
          'time=0.0 & ' + \
          ' & '.join([str(i) for i in l1]) + '\\\\\n' + \
          'time=0.5 & ' + ' & '.join([str(i) for i in l2]) + '\\\\\n' + \
          'time=1.0 & ' + ' & '.join([str(i) for i in l3]) + '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)

    def test_to_latex_3d_inner_axes(self):

        sh = (2,3,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        if debug:
            print 'c:'
            print c.descrip()


        result = c.to_latex(col_axes=['subj'], row_axes=['time'],
                            inner_axes=['mod'],
                            header_styles={'mod':'join','time':'join'},
                            val_fmt="%d",
                            hval_fmt={'time':"%1.1f"})

        l1 = ['%d | %d' %(a,b) for a,b in zip(range(4),range(12, 16))]
        l2 = ['%d | %d' %(a,b) for a,b in zip(range(4,8),range(16, 20))]
        l3 = ['%d | %d' %(a,b) for a,b in zip(range(8,12),range(20, 24))]
        good_result = \
          '\\begin{tabular}{c | ' + ' '.join(['c']*4) + '}\n' \
          '&\multicolumn{4}{c}{subj}\\\\\n' \
          '&s1 & s2 & s3 & s4\\\\\n' + \
          'time=0.0 & ' + \
          ' & '.join([str(i) for i in l1]) + '\\\\\n' + \
          'time=0.5 & ' + ' & '.join([str(i) for i in l2]) + '\\\\\n' + \
          'time=1.0 & ' + ' & '.join([str(i) for i in l3]) + '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)


    def test_to_latex_3d_col_align(self):
        sh = (2,3,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        if debug:
            print 'c:'
            print c.descrip()


        result = c.to_latex(col_axes=['mod','subj'], row_axes=['time'],
                            header_styles={'mod':'join','time':'join'},
                            val_fmt='%d', col_align={'mod':'c@{}'},
                            hval_fmt={'time':'%1.1f'})

        l1 = range(4) + range(12, 16)
        l2 = range(4,8) + range(16, 20)
        l3 = range(8,12) + range(20, 24)
        good_result = \
          '\\begin{tabular}{c | c c c c@{} c c c c@{}}\n' \
          '&\multicolumn{4}{c}{mod=m1} & \multicolumn{4}{c}{mod=m2}\\\\\n' \
          '&\multicolumn{8}{c}{subj}\\\\\n' \
          '&s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4\\\\\n' + \
          'time=0.0 & ' + \
          ' & '.join([str(i) for i in l1]) + '\\\\\n' + \
          'time=0.5 & ' + ' & '.join([str(i) for i in l2]) + '\\\\\n' + \
          'time=1.0 & ' + ' & '.join([str(i) for i in l3]) + '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)



    def test_to_latex_3d_hide_name_style(self):

        sh = (2,3,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        if debug:
            print 'c:'
            print c.descrip()


        result = c.to_latex(col_axes=['mod','subj'], row_axes=['time'],
                            header_styles={'mod':'join','time':'hide_name'},
                            hval_fmt={'time':'%1.1f'}, val_fmt='%d')

        l1 = range(4) + range(12, 16)
        l2 = range(4,8) + range(16, 20)
        l3 = range(8,12) + range(20, 24)
        good_result = \
          '\\begin{tabular}{c | ' + ' '.join(['c']*8) + '}\n' \
          '&\multicolumn{4}{c}{mod=m1} & \multicolumn{4}{c}{mod=m2}\\\\\n' \
          '&\multicolumn{8}{c}{subj}\\\\\n' \
          '&s1 & s2 & s3 & s4 & s1 & s2 & s3 & s4\\\\\n' + \
          '0.0 & ' + \
          ' & '.join([str(i) for i in l1]) + '\\\\\n' + \
          '0.5 & ' + ' & '.join([str(i) for i in l2]) + '\\\\\n' + \
          '1.0 & ' + ' & '.join([str(i) for i in l3]) + '\\\\\n' + \
          '\\end{tabular}'

        self.assertEqual(result, good_result,
            'to latex returned:\n' + result + '\n' + 'expected:\n' + good_result)






    def test_combine_domains(self):
        c = xndarray(np.zeros((2,3,4)), ['mod','time','subj'],
                   {'mod': ['m1', 'm2'],
                    'time': np.arange(3)*.5,
                    'subj' : ['s1', 's2', 's3', 's4']})

        cdom = c._combine_domains(['subj','time','mod'])
        expectation = [['s1', 's2', 's3', 's4'],
                       list(np.arange(3)*.5) * 4,
                       ['m1', 'm2'] * 4 * 3
                       ]
        self.assertEqual(cdom, expectation)


        cdom = c._combine_domains(['subj','time'])
        expectation = [['s1', 's2', 's3', 's4'],
                       list(np.arange(3)*.5) * 4,
                       ]
        self.assertEqual(cdom, expectation)


    def test_fill(self):
        """
        TODO
        """

        data = xndarray(np.array([[[1,1,2,2],
                                   [2,3,3,4],
                                   [4,4,4,4]],
                                   [[2,2,3,3],
                                    [3,4,4,5],
                                    [5,5,5,5]]]),
                                   axes_names=['time', 'axial','sagital'])

        fill_data = xndarray(np.array([[10,10,10,10],
                                       [20,20,20,20],
                                       [30,30,30,30]]),
                                       axes_names=['axial','sagital'])
        data.fill(fill_data)

        npt.assert_array_equal(data.data[0], fill_data.data)
        npt.assert_array_equal(data.data[1], fill_data.data)

    def test_explode(self):
        mask = xndarray(np.array([[0,0,1,1],
                                  [1,7,7,3],
                                  [3,3,3,3]]), axes_names=['axial','sagital'])

        data = xndarray(np.array([[[1,1,2,2],
                                   [2,3,3,4],
                                   [4,4,4,4]],
                                  [[2,2,3,3],
                                   [3,4,4,5],
                                   [5,5,5,5]]]),
                                   axes_names=['time', 'axial','sagital'])

        exploded_data = data.explode(mask, new_axis='position')

        self.assertEqual(len(exploded_data), len(np.unique(mask.data)))
        self.assertEqual(exploded_data[0].axes_names, ['time', 'position'])
        npt.assert_array_equal(exploded_data[0].data, np.array([[1,1],[2,2]]))
        npt.assert_array_equal(exploded_data[7].data, np.array([[3,3],[4,4]]))
        npt.assert_array_equal(exploded_data[3].data, np.array([[4,4,4,4,4],
                                                                [5,5,5,5,5]]))


    def test_merge(self):
        """
        TODO !!!
        """
        mask = xndarray(np.array([[0,0,1,1],
                                  [1,7,7,3],
                                  [3,3,3,3]]), axes_names=['axial','sagital'])

        data = xndarray(np.array([[[1,1,2,2],
                                   [2,3,3,4],
                                   [4,4,4,4]],
                                  [[2,2,3,3],
                                   [3,4,4,5],
                                   [5,5,5,5]]]),
                                   axes_names=['time', 'axial','sagital'])

        exploded_data = data.explode(mask, new_axis='position')

        new_data = merge(exploded_data, mask, axis='position')

        npt.assert_array_equal(new_data.data, data.data)
        npt.assert_array_equal(new_data.axes_names, data.axes_names)


    def test_set_orientation(self):
        c = xndarray(self.arr3d, self.arr3dNames, self.arr3dDom, self.arr3dLabel)
        if debug:
            print 'Original cuboid:'
            print c.descrip()

        c.set_orientation(['sagittal', 'coronal', 'axial'])
        if debug:
            print 'After changing orientation to ', \
                ['sagittal', 'coronal', 'axial']
            print c.descrip()
        assert c.data.shape == (self.sh3d[2],self.sh3d[1],self.sh3d[0])

        c.set_orientation(['sagittal', 'axial', 'coronal'])
        if debug:
            print 'After changing orientation to ', \
                ['sagittal', 'axial', 'coronal']
            print c.descrip()
        assert c.data.shape == (self.sh3d[2],self.sh3d[0],self.sh3d[1])


    def test_expansion(self):

        c = xndarray(self.arr_flat, self.arr_flat_names, self.arr_flat_domains)

        if debug:
            print 'Original cuboid:'
            print c.descrip()

        expanded_c = c.expand(self.mask2d, axis='position',
                              target_axes=self.mask2d_axes)
        if debug:
            print 'Expanded cuboid:'
            print expanded_c.descrip()

        assert expanded_c.data.shape == self.mask2d.shape + \
            self.arr_flat.shape[1:]


    def test_flatten_and_expand(self):

        sh = (2,100,50)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['c','voxel', 't'])
        # print 'c:'
        # print c.descrip()
        m = np.zeros((10,10,10), dtype=int)
        m.flat[:100] = 1
        c_expanded = c.expand(m, axis='voxel', target_axes=['x','y','z'])

        c_flat = c_expanded.flatten(m, ['x','y','z'], 'voxel')
        # print 'c_flat:'
        # print c_flat.descrip()
        assert c == c_flat

    def test_sub_cuboid(self):
        pyhrf.verbose.set_verbosity(0)

        c = xndarray(self.arr_flat, self.arr_flat_names, self.arr_flat_domains)

        expanded_c = c.expand(self.mask2d, axis='position',
                              target_axes=self.mask2d_axes)
        if debug:
            print 'Original cuboid:'
            print expanded_c.descrip()

        sub_c = expanded_c.sub_cuboid(condition='audio', orientation=['y','x'])
        if debug:
            print 'Sub cuboid:'
            print sub_c.descrip()

        assert sub_c.data.shape == (self.mask2d.shape[1],self.mask2d.shape[0])

    def test_operations(self):

        data1 = np.arange(4*5, dtype=float).reshape(4,5) + 1
        # print 'data1:', data1.dtype
        data1_copy = data1.copy()
        data2 = np.arange(4*5).reshape(4,5) * 4. + 1
        # print 'data2:', data2.dtype
        data2_copy = data2.copy()

        c = xndarray(data1, ['x','time'], {'time':np.arange(5)+2/3.})
        c_bak = c
        # print 'c:', id(c)
        # print c.descrip()
        # print c.data
        # print ''

        c2 = xndarray(data2, ['x','time'], {'time':np.arange(5)+2/3.})
        # print 'c2:', id(c2)
        # print c2.descrip()
        # print c2.data
        # print ''

        c_add = c + c2
        # print 'c_add:', id(c_add)
        # print c_add.descrip()
        # print c_add.data
        assert (c_add.data == data1_copy + data2_copy).all()

        c += c2
        # print 'c after inplace add:', id(c)
        # print c.descrip()
        # print c.data
        assert c.data is data1
        assert (c.data == data1_copy + data2_copy).all()
        assert c is c_bak

        c_sub = c - c2
        assert (c_sub.data == data1 - data2).all()

        c -= c2
        assert (c_sub.data == data1_copy).all()
        assert c is c_bak

        c_mul = c * c2
        assert (c_mul.data == data1_copy * data2_copy).all()

        c_div = c / c2
        # print 'c_div:'
        # print c_div.data
        # print data1_copy / data2_copy
        #assert np.isnan(c_div.data[0,0])
        assert (c_div.data.flat[1:] == (data1_copy / data2_copy).flat[1:]).all()


        r = 2 / c
        npt.assert_array_equal(r.data, 2 / c.data)

        r = 4 * c
        npt.assert_array_equal(r.data, 4 * c.data)

        r = 4 + c
        npt.assert_array_equal(r.data, 4 + c.data)



        r = 5 - c
        npt.assert_array_equal(r.data, 5 - c.data)

    def test_sub_cuboid_with_float_domain(self):

        c = xndarray(np.arange(4*2).reshape(4,2), ['x','time'],
                   {'time':np.array([2/3.,2.3])})
        if debug:
            print 'Original cuboid:'
            print c.descrip()

        sub_c = c.sub_cuboid(time=2/3.)
        if debug:
            print 'sub_cuboid at time=2/3.:'
            print sub_c.descrip()


    def test_equality(self):
        c1 = xndarray(self.arr3d, self.arr3dNames, self.arr3dDom,
                    self.arr3dLabel)

        c2 = xndarray(self.arr3d, self.arr3dNames, self.arr3dDom,
                    self.arr3dLabel)

        assert c1 == c2

        c3 = xndarray(self.arr3d, self.arr3dNames)

        assert c1 != c3

    def test_save_as_nii(self):

        pyhrf.verbose.set_verbosity(0)
        c = xndarray(self.arr3d, ['x','y','z'],
                     {'x' : np.arange(self.sh3d[0])*3,
                      'y' : np.arange(self.sh3d[1])*3,
                      'z' : np.arange(self.sh3d[2])*3,
                      },
                   self.arr3dLabel)

        if debug:
            print 'Original cuboid:'
            print c.descrip()

        fn = op.join(self.tmp_dir, 'cuboid.nii')
        if debug:
            print 'fn:', fn


        c.save(fn)
        assert op.exists(fn)

        c_loaded = xndarray.load(fn)

        if debug:
            print 'Loaded cuboid:'
            print c_loaded.descrip()

        assert c == c_loaded



    def test_save_as_gii(self):

        c = xndarray(np.arange(4*2).reshape(4,2), ['x','time'],
                   {'time':np.array([2/3.,2.3])})

        if debug:
            print 'Original cuboid:'
            print c.descrip()
            print c.data

        fn = op.join(self.tmp_dir, 'cuboid.gii')
        if debug:
            print 'fn:', fn


        c.save(fn)
        assert op.exists(fn)

        c_loaded = xndarray.load(fn)

        if debug:
            print 'Loaded cuboid:'
            print c_loaded.descrip()
            print c_loaded.data

        assert c == c_loaded





    def test_split(self):

        pyhrf.verbose.set_verbosity(0)
        sh = (2,4,4,4)
        c = xndarray(np.arange(np.prod(sh)).reshape(sh), ['condition']+MRI3Daxes,
                   {'condition':['audio','video']})
        if debug:
            print 'Original cub:'
            print c.descrip()

        fn = op.join(self.tmp_dir, 'cub.nii')
        if debug:
            print 'Save and load original cub'
        c.save(fn)
        c = xndarray.load(fn)

        fn = op.join(self.tmp_dir, 'cub2.nii')
        if debug:
            print 'Save and load new cub with meta data from original cuboid'
        sh = (4,4,4)
        c2 = xndarray(np.arange(np.prod(sh)).reshape(sh), MRI3Daxes,
                    meta_data=c.meta_data)
        c2.save(fn)
        c2 = xndarray.load(fn)

        fns = []
        sub_cuboids = []
        if debug:
            print 'Split and save sub cuboids'
        for dvalue,sub_c in c.split('condition').iteritems():
            fn = op.join(self.tmp_dir, add_suffix('sub_c.nii',
                                                  '_%s' %str(dvalue)))
            if debug and dvalue == 'audio':

                print 'fn_out:', fn
                print 'sub_c:'
                print sub_c.descrip()
                sub_cuboids.append(sub_c)
                sub_c.save(fn)
                fns.append(fn)
        if debug:
            print ''
            print 'Load sub c again ...'
        for fn, sub_c in zip(fns, sub_cuboids):
            if debug:
                print 'fn:', fn
            c_loaded = xndarray.load(fn)
            if debug:
                print 'c_loaded:'
                print c_loaded.descrip()
            self.assertEqual(c_loaded, sub_c)



    def test_stack(self):

        d1 = np.arange(4*5).reshape(4,5)
        c1 = xndarray(d1, ['x','y'],
                    {'x' : np.arange(4)*3,
                     'y' : np.arange(5)*3,
                     })
        if debug:
            print 'c1:'
            print c1.descrip()

        d2 = np.arange(4*5).reshape(4,5)*2
        c2 = xndarray(d2, ['x','y'],
                    {'x' : np.arange(4)*3,
                     'y' : np.arange(5)*3,
                     })
        if debug:
            print 'c2:'
            print c2.descrip()

        c_stacked = stack_cuboids([c1,c2], 'stack_axis', ['c1','c2'])

        if debug:
            print 'c_stacked:'
            print c_stacked.descrip()

        a_stacked = np.array([d1,d2])
        assert (c_stacked.data == a_stacked).all()


    def test_tree_to_xndarray(self):
        from pyhrf.ndarray import xndarray, tree_to_xndarray
        from pyhrf.tools import set_leaf
        d1 = {}
        set_leaf(d1, ['1','2.1','3.1'], xndarray(np.array([1])))
        set_leaf(d1, ['1','2.1','3.2'], xndarray(np.array([2])))
        set_leaf(d1, ['1','2.2','3.1'], xndarray(np.array([3])))
        set_leaf(d1, ['1','2.2','3.2'], xndarray(np.array([3.1])))
        d2 = {}
        set_leaf(d2, ['1','2.1','3.1'], xndarray(np.array([10])))
        set_leaf(d2, ['1','2.1','3.2'], xndarray(np.array([11])))
        set_leaf(d2, ['1','2.2','3.1'], xndarray(np.array([12])))
        set_leaf(d2, ['1','2.2','3.2'], xndarray(np.array([13])))

        d = {'d1':d1, 'd2':d2}
        labels = ['case', 'p1', 'p2', 'p3']

        c = tree_to_xndarray(d, labels)
        self.assertEqual(c.data.shape, (2,1,2,2,1))

        npt.assert_array_equal(c.axes_domains['case'], ['d1', 'd2'])
        npt.assert_array_equal(c.axes_domains['p1'], ['1'])
        npt.assert_array_equal(c.axes_domains['p2'], ['2.1', '2.2'])
        npt.assert_array_equal(c.axes_domains['p3'], ['3.1', '3.2'])

    def test_cartesian_eval(self):
        """
        Test the multiple evaluations of a function that returns
        a xndarray, over the cartesian products of given arguments.
        """

        def foo(a, b, size, aname):
            return xndarray(np.ones(size) * a + b, axes_names=[aname])

        from pyhrf.tools import cartesian_apply
        from pyhrf.tools.backports import OrderedDict

        varying_args = OrderedDict([ ('a', [1,2]), ('b', [.5, 1.5]) ])
        fixed_args = {'size' : 2, 'aname': 'my_axis'}

        res = tree_to_xndarray(cartesian_apply(varying_args, foo, fixed_args))

        self.assertEqual(res.data.shape, (2, 2, 2))
        npt.assert_array_equal(res.data, np.array([[[1.5, 1.5],
                                                    [2.5, 2.5]],
                                                   [[2.5, 2.5],
                                                    [3.5, 3.5]]]))
