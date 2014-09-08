# -*- coding: utf-8 -*-


import os
import os.path as op
import unittest
import tempfile
import shutil

import numpy as _np

from pyhrf.graph import *
from pyhrf.boldsynth.spatialconfig import lattice_indexes

from pyhrf.tools._io import read_volume, write_volume

class GraphTest(unittest.TestCase):

    def setUp(self):
        self.lattice2D = _np.array([[1]]*4+[[2]]*4+[[0]], dtype=int).reshape(3,3)
        self.fullMask2D = _np.ones_like(self.lattice2D)
        self.indexes2D = lattice_indexes(_np.ones_like(self.lattice2D))

        flat3DLabels = [[1]]*5 + [[2]]*15 + [[0]]*20 + [[2]]*10 + [[0]]*20
        self.lattice3D = _np.array(flat3DLabels).reshape(2,5,7)
        self.indexes3D = lattice_indexes(_np.ones_like(self.lattice3D))
        self.fullMask3D = _np.ones_like(self.lattice3D)

        self.tmp_dir = tempfile.mkdtemp(prefix='pyhrf_tests',
                                        dir=pyhrf.cfg['global']['tmp_path'])


    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


    def test_graph_is_sane(self):
        #012
        #345
        #678
        good = _np.array( [_np.array([1,3]), _np.array([0,2,4]), _np.array([1,5]),
                           _np.array([0,4,6]), _np.array([1,5,3,7]), 
                           _np.array([2,4,8]), _np.array([3,7]), 
                           _np.array([6,4,8]), _np.array([7,5])],
                          dtype=object)
        assert graph_is_sane(good)
        bad1 = _np.array( [_np.array([1,3]), _np.array([0,2,4]), _np.array([1,5]),
                           _np.array([0,4,6]), _np.array([1,5,3,7]), 
                           _np.array([2,8]), _np.array([3,7]), 
                           _np.array([6,4,8]), _np.array([7,5])],
                          dtype=object)
        assert not graph_is_sane(bad1)
        bad2 = _np.array( [_np.array([1,3,3]), _np.array([0,2,4]), _np.array([1,5]),
                           _np.array([0,4,6]), _np.array([1,5,3,7]), 
                           _np.array([2,8]), _np.array([3,7]), 
                           _np.array([6,4,8]), _np.array([7,5])],
                          dtype=object)
        assert not graph_is_sane(bad2)

    def test_from_mesh(self):
        m = [ [1,3,2], [0,1,3], [0,3,4], [2,3,4] ]
        graph_from_mesh(m)
        
    def test_from_lattice1(self):
        """ Test default behaviour of graph_from_lattice, in 2D
        """
        #print 'lattice:'
        #print self.lattice2D
        g = graph_from_lattice(self.fullMask2D)
        assert graph_is_sane(g)
        #print 'g:'
        #print g

    def test_from_lattice_toro(self):
        """ Test graph_from_lattice, 2D toroidal case
        """
        #print 'lattice:'
        #print self.lattice2D
        #print 'indexes:'
        #print lattice_indexes(self.fullMask2D)
        g = graph_from_lattice(self.fullMask2D, toroidal=True, 
                               kerMask=kerMask2D_4n)
        assert graph_is_sane(g)
#         print 'g:'
#         print g
        assert graph_is_sane(g)
        
    def test_from_lattice_toro_huge(self):
        """ Test graph_from_lattice, 2D toroidal case
        """
        g = graph_from_lattice(_np.ones((100,100),dtype=int), toroidal=True, 
                               kerMask=kerMask2D_4n)
#         print 'g:'
#         print g
        assert graph_is_sane(g, toroidal=True)

    def test_from_lattice2(self):
        """ Test graph_from_lattice in 2D with another kernel mask
        """
#         print 'lattice:'
#         print self.lattice2D
        g = graph_from_lattice(self.fullMask2D, kerMask=kerMask2D_4n)
#         print 'g:'
#         print g

    def test_from_lattice2(self):
        """ Test graph_from_lattice in 3D with another kernel mask
        """
#         print 'lattice:'
#         print self.lattice3D
        g = graph_from_lattice(self.fullMask3D, kerMask=kerMask3D_6n)
#         print 'g:'
#         print g

    def test_sub_graph(self):
#         print 'lattice:'
#         print self.lattice2D
        fullGraph = graph_from_lattice(self.fullMask2D, kerMask=kerMask2D_4n)

        nodesWhere1 = self.indexes2D[self.lattice2D==1]
#         print 'nodesWhere1 :'
#         print nodesWhere1
        graphWhere1, idxMap1 = sub_graph(fullGraph, nodesWhere1)
#         print 'subgraph for label 1:'
#         print graphWhere1

        nodesWhere2 = self.indexes2D[self.lattice2D==2]
#         print 'nodesWhere2 :'
#         print nodesWhere2
        graphWhere2, idxMap2 = sub_graph(fullGraph, nodesWhere2)
#         print 'subgraph for label 2:'
#         print graphWhere2
        assert graph_is_sane(graphWhere2)

    def test_parcels_to_graphs(self):
#         print 'lattice:'
#         print self.lattice3D
        pgs = parcels_to_graphs(self.fullMask3D, kerMask=kerMask3D_6n)
        for ip, pg in pgs.iteritems():
            assert graph_is_sane(pg)


    def test_bfs(self):

        vol = np.array( [[1,1,0,1,1],
                         [1,1,0,1,1],
                         [0,1,0,0,0],
                         [1,1,1,1,0],
                         [0,0,1,1,0]], dtype=int )
        # print 'vol:'
        # print vol
        # print 'build from lattice ...'
        g = graph_from_lattice(vol, kerMask2D_4n)
        # print 'g:'
        # print g
        _,order = breadth_first_search(g)
        # print 'bfs order:'
        # print order
        visited_vol = np.zeros(len(g), dtype=int)
        visited_vol[order] = 1
        visited_vol = expand_array_in_mask(visited_vol, vol)
        # print 'visited positions:'
        # print visited_vol
        
        true_visited_vol = np.array( [[1,1,0,0,0],
                                      [1,1,0,0,0],
                                      [0,1,0,0,0],
                                      [1,1,1,1,0],
                                      [0,0,1,1,0]], dtype=int )

        assert (true_visited_vol == visited_vol).all()

    def test_split_vol_cc_2D(self):
        
        vol = np.array( [[1,1,0,1,1],
                         [1,1,0,1,1],
                         [0,0,0,0,0],
                         [1,0,1,1,0],
                         [0,0,1,1,0]], dtype=int )
        #print 'vol:'
        #print vol
        #print ''
        #print '______________'
        for i,mcc in enumerate(split_mask_into_cc_iter(vol, 4, kerMask2D_4n)):
            #print mcc
            #print ''
            pass

        assert i==2
            

    def test_split_vol_cc_3D(self):
        
        fn = 'subj0_parcellation.nii.gz'
        mask_file = pyhrf.get_data_file_name(fn)
        #print mask_file
        mask = read_volume(mask_file)[0].astype(int)
        #print np.unique(mask), mask.dtype
        mask[np.where(mask==2)] = 1
        #print np.unique(mask), mask.dtype
        km = kerMask3D_6n
        for i,mcc in enumerate(split_mask_into_cc_iter(mask, kerMask=km)):
            #print mcc
            #print ''
            pass
        #print i
        assert i==1 #2 rois

    def test_pyhrf_extract_cc_vol(self):
        test_mask = np.array( [[[1,1,0,1,1],
                                [1,1,0,1,1],
                                [0,0,0,0,0],
                                [1,0,1,1,0],
                                [0,0,1,1,0]],
                               [[1,1,0,1,1],
                                [1,1,0,1,1],
                                [0,0,0,0,0],
                                [0,0,1,1,0],
                                [0,0,1,1,0]]],
                              dtype=int )
        mask_file = op.join(self.tmp_dir, 'test_mask.nii')
        write_volume(test_mask, mask_file)

        cmd = 'pyhrf_extract_cc_vol -v0 -m 2 %s' %mask_file
        if os.system(cmd) != 0:
            raise Exception('Command %s failed' %cmd)
        #print os.listdir(self.tmp_dir)
        assert len(os.listdir(self.tmp_dir)) == 4
