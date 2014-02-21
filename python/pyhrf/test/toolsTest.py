

# -*- coding: utf-8 -*-
import unittest, os
import numpy as np

import tempfile
import shutil

from pyhrf.tools import *
import pyhrf.tools as mtools
import time
from pyhrf.verbose import dictToString
import pyhrf

import numpy.testing as npt

class GeometryTest(unittest.TestCase):

    def test_distance(self):
        pass

    def test_convex_hull(self):

        mask = np.array([[0,1,1,1,1],
                         [0,0,0,1,1],
                         [1,0,0,1,1],
                         [1,1,1,1,1]])

        expected_concex_hull = np.array([[0,1,1,1,1],
                                         [0,1,1,1,1],
                                         [1,1,1,1,1],
                                         [1,1,1,1,1]])


        chull = mtools.convex_hull_mask(mask)

        npt.assert_equal(chull, expected_concex_hull)


class CropTest(unittest.TestCase):

    def testBasic(self):
        a = np.zeros((4,4))
        a[1:3,1:3] = 1
##        print 'a :'
##        print a
##        print 'cropped a:'
##        print cropArray(a)


class MiscTest(unittest.TestCase):

    def test_decorator_do_if_file_exist(self):
        from pyhrf.tools import do_if_nonexistent_file

        @do_if_nonexistent_file('fn','other_file')
        def perform_task(a, fn, blah, other_file='coco'):
            return True

        f = './my_file.txt'
        # f does not exist yet, expect perform_task not to be executed
        assert perform_task(1, f, 'yoyo')

        # create f -> perform_task should not be executed
        os.system('touch %s' %f)
        os.system('touch coco')
        if perform_task(0, f, 'yoyo') is not None:
            os.system('rm %s coco' %f)
            raise Exception('perform_task should return None when file exists')
        os.system('rm %s coco' %f)

    def test_decorator_do_if_file_exist2(self):

        def foo(a, b, c='4'):
            return True

        r = do_if_nonexistent_file('a')(foo)("1","2")
        assert r


    def test_decorator_do_if_file_exist_force(self):
        from pyhrf.tools import do_if_nonexistent_file

        @do_if_nonexistent_file('fn','other_file', force=True)
        def perform_task(a, fn, blah, other_file='coco'):
            return True

        f = './my_file.txt'
        os.system('touch %s' %f)
        os.system('touch coco')
        # files exist, but force=True -> expect perform_task to be executed
        if not perform_task(0, f, 'yoyo'):
            os.system('rm %s coco' %f)
            raise Exception('perform_task should return None when file exists')
        os.system('rm %s coco' %f)



class PeelVolumeTest(unittest.TestCase):

    def testPeel(self):

        a = np.ones((5,5,5), dtype=int)
        peeledA = peelVolume3D(a)
        correctPeeling = np.zeros_like(a)
        correctPeeling[1:-1, 1:-1, 1:-1] = 1
##        print "peeledA:"
##        print peeledA
##        print "correctPeeling:"
##        print correctPeeling
        assert (peeledA == correctPeeling).all()

        m = np.zeros((6,6,6), dtype=int)
        m[1:-1,1:-1,1:-1] = 1
        #m[3,3,3] = 0
##        print 'm:'
##        print m
        peeledM = peelVolume3D(m)
##        print 'peeledM:'
##        print peeledM

class CartesianTest(unittest.TestCase):

    def testCartesianBasic(self):

        domains = [(0,1,2),('a','b')]
        cartProd = list(cartesian(*domains))
        assert cartProd == [[0, 'a'], [0, 'b'], [1, 'a'], [1, 'b'], [2, 'a'],
                            [2, 'b']]



    def test_cartesian_apply(self):
        from pyhrf.tools import cartesian_apply
        from pyhrf.tools.backports import OrderedDict
        
        def foo(a, b, c=1, d=2):
            return a + b + c + d

        # OrderDict to keep track of parameter orders in the result
        # arg values must be hashable
        varying_args = OrderedDict([('a' , range(2)),
                                    ('b' , range(2)),
                                    ('c' , range(2))])

        fixed_args = {'d' : 10}

        result_tree = cartesian_apply(varying_args, foo,
                               fixed_args=fixed_args)

        self.assertEqual(result_tree, {0 : { 0 : { 0 : 10, 1 : 11},
                                             1 : { 0 : 11, 1 : 12}},
                                       1 : { 0 : { 0 : 11, 1 : 12},
                                             1 : { 0 : 12, 1 : 13}}})


class DictToStringTest(unittest.TestCase):

    def testBasic(self):
        d = { 'k1' : 'v1',
              65480 : 'v2',
              'k3' : 90,
              600 : 143,
              }

##        print dictToString(d)

    def testOnHierachicDict(self):

        d = { 'k1' : 'v1',
              'd2' : {'d2k1':0, 1:190, 'd2-3' : {'plop':'plip',435:'fr'}},
              'k3' : 1098,
              }

##        print dictToString(d)

    def testOnNumpyArray(self):

        na = np.random.randn(4,4)
        d = { 'k1' : 'v1',
              'd2' : {'d2k1':0, 1:190, 'na':na,
                      'd2-3' : {'plop':'plip',435:'fr'}},
              'k3' : 1098,
              }

        sd1 = dictToString(d)
        #print sd1
        naf = {'precision':2, 'max_line_width':100}
        sd2 = dictToString(d, numpyArrayFormat=naf)
##        print sd2
        #print np.array2string(na,**naf)

    def testOnSpmMat(self):
        spmFile = '/export/TMP/thomas/Data/Localizer/bru2698/functional/fMRI/'\
                  'spm_analysis/SPM.mat'
        if os.path.exists(spmFile):
            from scipy.io.mio import loadmat
            spm = loadmat(spmFile)
##            print dictToString(spm, exclude=['private'])



class TableStringTest(unittest.TestCase):

    def setUp(self):
        nrows = 10
        ncols = 2
        self.rownames = ['row-%d'%i for i in xrange(nrows)]
        self.colnames = ['col-%d'%i for i in xrange(ncols)]
        self.data1D = np.random.randn(nrows)
        self.data2D = np.random.randn(nrows, ncols)
        self.data3D = np.random.randn(nrows, ncols, 7)
        self.data4D = np.random.randn(nrows, ncols, 7, 20)

    def test1Darray(self):
        s = get_2Dtable_string(self.data1D, self.rownames, ['Value'])
        #print s

    def test2Darray(self):
        s = get_2Dtable_string(self.data2D, self.rownames, self.colnames)
        #print s

    def test3Darray(self):
        s = get_2Dtable_string(self.data3D, self.rownames, self.colnames)
        if 0:
            print s

    def test4Darray(self):
        s = get_2Dtable_string(self.data3D, self.rownames, self.colnames,
                               precision=2, outline_char='-', line_end='|',
                               line_start='|')
        if 0:
            print ''
            print s

    def test2Darray_latex(self):
        s = get_2Dtable_string(self.data2D, self.rownames, self.colnames, line_end='\\\\', col_sep='&')
        if 0:
            print 's:'
            print s



class DiagBlockTest(unittest.TestCase):

    def testAll2D(self):

        m1 = np.arange(2*3).reshape(2,3)
        m2 = np.arange(2*4).reshape(2,4)

##        print 'm1 :'
##        print m1
##        print 'm2 :'
##        print m2
        bm = diagBlock([m1,m2])

##        print 'bm:'
##        print bm

    def testFrom1D(self):

        m1 = np.arange(2*3)

##        print 'm1 :'
##        print m1
        bm = diagBlock([m1])

##        print 'bm:'
##        print bm

    def testFromNdarray(self):
        m1 = np.arange(2*3)

##        print 'm1 :'
##        print m1
        bm = diagBlock([m1])

##        print 'bm:'
##        print bm


    def testRepFrom1D(self):

        m1 = np.arange(2*3)

##        print 'm1 :'
##        print m1
        bm = diagBlock(m1, 5)

##        print 'bm:'
##        print bm

    def testRepFrom2D(self):

        m1 = np.arange(2*3).reshape(2,3)

##        print 'm1 :'
##        print m1
        bm = diagBlock(m1, 2)

##        print 'bm:'
##        print bm


    def testRepFromBlocks(self):

        m1 = np.arange(2*3).reshape(2,3)
        m2 = np.arange(2*4).reshape(2,4)

##        print 'm1 :'
##        print m1
##        print 'm2 :'
##        print m2

        bm = diagBlock([m1,m2], 2)

##        print 'bm:'
##        print bm


class ResampleTest(unittest.TestCase):

    def testResampleToGrid(self):

        import numpy
        size = 10

        x = numpy.concatenate(([0.],numpy.sort(numpy.random.rand(size)),[1.]))
        y = numpy.concatenate(([0.],numpy.sort(numpy.random.rand(size)),[1.]))
        grid = numpy.arange(0, 1., 0.01)
        ny = resampleToGrid(x,y,grid)

        #import matplotlib.pyplot as plt
        #plt.plot(x,y,'o-')
        #plt.plot(grid,ny,'x-')
        #plt.show()

    def testLargerTargetGrid(self):


        import numpy
        size = 10

        x = numpy.concatenate(([0.],numpy.sort(numpy.random.rand(size)),[1.]))
        y = numpy.concatenate(([0.],numpy.sort(numpy.random.rand(size)),[1.]))
        grid = numpy.arange(-0.2, 1.1, 0.01)
        ny = resampleToGrid(x,y,grid)

##         import matplotlib.pyplot as plt
##         plt.plot(x,y,'o-')
##         plt.plot(grid,ny,'x-')
##         plt.show()



class treeToolsTest(unittest.TestCase):

    def test_set_leaf(self):
        d = {}
        set_leaf(d, ['b1','b2','b3'], 'theLeaf')
#        print d

    def test_get_leaf(self):
        d = {}
        set_leaf(d, ['b1','b2','b3.1'], 'theLeaf1')
        set_leaf(d, ['b1','b2','b3.2'], 'theLeaf2')
        l1 = get_leaf(d, ['b1','b2','b3.1'])
        l2 = get_leaf(d, ['b1','b2','b3.2'])
        return l2

    def test_walk_branches(self):
        d = {}
        set_leaf(d, ['b1','b2.1','b3.1'], 'theLeaf1')
        set_leaf(d, ['b1','b2.1','b3.2'], 'theLeaf2')
        set_leaf(d, ['b1','b2.2','b3.3'], 'theLeaf3')
#         print 'd:'
#         print d
#         for b in treeBranches(d):
#             print 'b', b

    def test_stack_trees(self):
        d1 = {}
        set_leaf(d1, ['b1','b2.1','b3.1'], 'd1Leaf1')
        set_leaf(d1, ['b1','b2.1','b3.2'], 'd1Leaf2')
        set_leaf(d1, ['b1','b2.2','b3.3'], 'd1Leaf3')

        d2 = {}
        set_leaf(d2, ['b1','b2.1','b3.1'], 'd2Leaf1')
        set_leaf(d2, ['b1','b2.1','b3.2'], 'd2Leaf2')
        set_leaf(d2, ['b1','b2.2','b3.3'], 'd2Leaf3')

#        print stack_trees([d1,d2])

    def test_rearrange(self):

        d1 = {}
        set_leaf(d1, ['1','2.1','o1'], 'c1')
        set_leaf(d1, ['1','2.1','o2'], 'c2')
        set_leaf(d1, ['1','2.2','o1'], 'c3')
        set_leaf(d1, ['1','2.2','o2'], 'c4')

        blabels = ['p1', 'p2', 'outname']

        nblabels = ['outname', 'p1', 'p2']
        tree_rearrange(d1, blabels, nblabels)
        #from pprint import pprint
        #pprint(d1)
        #print ''
        #print '->'
        #print ''
        #pprint(tree_rearrange(d1, blabels, nblabels))



# class VerboseTest(unittest.TestCase):

#     def test_new_line(self):
#         pyhrf.verbose.set_verbosity(1)
#         pyhrf.verbose(1,'some text to be continued ...', new_line=False)
#         pyhrf.verbose(1,'cont\'d!')
#         pyhrf.verbose(1,'Some other text')
#         pyhrf.verbose.set_verbosity(0)


def foo_func(a,b):
    return a+b

def slow_func(a,b):
    time.sleep(1)
    return a+b


class CachedEvalTest(unittest.TestCase):

    def setUp(self):
        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.cache_dir = tmpDir


    def test_simple(self):
        cached_eval(foo_func,(1,2), path=self.cache_dir)

    def test_simple_args(self):
        cached_eval(foo_func, {'a':4,'b':6}, path=self.cache_dir)

    def test_slow_func(self):
        #print '1'
        t0 = time.time()
        cached_eval(slow_func, {'a':4,'b':6}, path=self.cache_dir)
        delta = time.time() - t0
        #print delta
        #print '2'
        t0 = time.time()
        cached_eval(slow_func, {'a':4,'b':6}, path=self.cache_dir)
        delta = time.time() - t0
        #print delta
        assert delta < .1
        #print '3'
        t0 = time.time()
        cached_eval(slow_func, {'a':4,'b':8}, path=self.cache_dir)
        delta = time.time() - t0
        #print delta

    def test_code_digest(self):
        t0 = time.time()
        cached_eval(slow_func, {'a':4,'b':8}, digest_code=True,
                    path=self.cache_dir)
        delta = time.time() - t0
        #print 'took ', delta, 'sec'

    def tearDown(self):
        shutil.rmtree(self.cache_dir)

def computeB(a,e):
    return a+e

def computeC(a):
    return a**2

def computeF(g,e):
    return g/e

def computeD(f,b,c):
    return (f+b)*c

def computeJ(i,l):
    return i**3+l

def computeK(j):
    return j/2.

def computeL(k):
    return k/3.


def foo_default_arg(a,d=1):
    return a+d

def foo_a(c=1):
    return c


def foo_multiple_returns(e):
    return e, e/2

class PipelineTest(unittest.TestCase):

    def setUp(self):


        tmpDir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                  dir=pyhrf.cfg['global']['tmp_path'])
        self.cache_dir = tmpDir

        #pyhrf.verbose.set_verbosity(0)

        # Be sure to put scalars in arrays so that they can be referenced
        self.a = np.array([3])
        self.e = np.array([10])
        self.g = np.array([9.3])


        # Good acyclic dependence tree :
        #   g-\.
        #   e--.f
        #    \.  \
        #     .b  \.
        #   a/ \___.d
        #    \.c__/
        self.depTree = { 'a' : self.a,
                         'e' : self.e,
                         'g' : self.g,
                         'b' : computeB,
                         'c' : computeC,
                         'f' : computeF,
                         'd' : computeD,
                         }


        # Bad cyclic dependence tree :
        #    j.
        #    | \
        # i  . .l
        #  \.k/

        self.i = np.array([34.5])

        self.badDepTree = { 'i' : self.i,
                            'j' : computeJ,
                            'k' : computeK,
                            'l' : computeL
                            }
    def tearDown(self):
        shutil.rmtree(self.cache_dir)


    def testGoodDepTreeInit(self):

        data = Pipeline(self.depTree)
        # print 'data.values:',
        # print data.values
        assert (data.get_value('b') == self.a+self.e).all()
        assert (data.get_value('c') == self.a**2).all()
        assert (data.get_value('d') == (self.g/self.e+self.a+self.e)*self.a**2).all()
        assert (data.get_value('f') == self.g/self.e).all()

    def testBadDepTreeInit(self):

        try:
            data = Pipeline(self.badDepTree)
        except Exception , e:
            pass
            #print 'Exception normally raised :'+str(e)
        else:
            raise Exception()

    def testRepr(self):
        data = Pipeline(self.depTree)
        #print 'printing all deps :'
        #print data.reprAllDeps()

    def test_func_default_args(self):
        data = Pipeline({'a' : foo_a,
                            'b' : foo_default_arg,
                            'd' : 7})
        data.resolve()

    def test_cached(self):
        try:
            from joblib import Memory
            mem = Memory(self.cache_dir)
            dep_tree = {
                'a' : 5,
                'b' : 6,
                'c' : mem.cache(slow_func),
                }
            data = Pipeline(dep_tree)
            t0 = time.time()
            data.resolve()
            delta = time.time() - t0
            #print 'delta 1:', delta

            t0 = time.time()
            data.resolve()
            delta = time.time() - t0
            assert delta < .1
            #print 'delta 2:', delta
        except:
            pass

    def test_multiple_output_values(self):

        pyhrf.verbose.set_verbosity(0)
        data = Pipeline({'e' : foo_a,
                            'b' : foo_default_arg,
                            ('a','d') : foo_multiple_returns})
        data.resolve()

        #data.save_graph_plot('./g.png')

        # if data['e'] != :
        #     raise Exception('Wrong value for quantity "e". Got %s, expected %s'\
        #                     %(str(data[e]), str())


