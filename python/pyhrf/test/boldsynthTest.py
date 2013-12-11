

import unittest
import numpy as np

from pyhrf.boldsynth.spatialconfig import *
from pyhrf.boldsynth.field import *
from pyhrf.boldsynth.pottsfield.pottsfield_c import genPottsField
from pyhrf.graph import *
from pyhrf.stats.random import *
from pyhrf.boldsynth.scenarios import *
from pyhrf.verbose import dictToString

class Mapper1DTest(unittest.TestCase):

    def test3D(self):

        a = np.arange(2*2*2).reshape(2,2,2)
        #print 'a :'
        #print a
        mapping = np.array([ [0,0], [0,1], [1,0], [1,1] ])
        #print 'mapping :'
        #print mapping
        mapper = Mapper1D(mapping, a.shape[1:])
        fa = mapper.flattenArray(a, 1)
        #print 'fa :'
        #print fa

        efa = mapper.expandArray(fa, 1)
        #print 'efa :'
        #print efa

        assert (efa == a).all()

    def testIrregularMapping(self):
        a = np.arange(2*2*2).reshape(2,2,2)
        #print 'a :'
        #print a
        mapping = np.array([ [1,1], [0,1], [1,0], [0,0] ])
        #print 'mapping :'
        #print mapping
        mapper = Mapper1D(mapping, a.shape[1:])
        fa = mapper.flattenArray(a, 1)
        #print 'fa :'
        #print fa

        efa = mapper.expandArray(fa, 1)
        #print 'efa :'
        #print efa

        assert (efa == a).all()
        
    def testIncompleteMapping(self):
        a = np.arange(2*2*2).reshape(2,2,2)
#         print 'a :'
#         print a
        mapping = np.array([ [1,1], [0,1], [0,0] ])
#         print 'mapping :'
#         print mapping
        mapper = Mapper1D(mapping, a.shape[1:])
        fa = mapper.flattenArray(a, 1)
#         print 'fa :'
#         print fa

        efa = mapper.expandArray(fa, 1, fillValue=-1)
#         print 'efa :'
#         print efa

        assert (efa[:, [1,0,0], [1,1,0] ] == a [:, [1,0,0], [1,1,0] ]).all()


class FieldFuncsTest(unittest.TestCase):
    
    def test_count_homo_cliques(self):
        labels = np.array( [[ 1, 1, 1],
                             [ 1, 1, 1],
                             [ 1, 1, 1]], dtype=int)
        mask = np.ones_like(labels)
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)
#         print 'labels:'
#         print labels
        hc = count_homo_cliques(g, labels[np.where(mask)])
#         print '-> homo cliques:'
#         print hc
        assert hc == 2*labels.size

    def test_count_homo_cliques1(self):
        labels = np.array( [[ 0, 0, 1],
                             [ 0, 1, 1]], dtype=int)
        mask = np.ones_like(labels)
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
#         print 'labels:'
#         print labels
        hc = count_homo_cliques(g, labels[np.where(mask)])
#         print '-> homo cliques:'
#         print hc
        assert hc == 4

    def test_count_homo_cliques2(self):
        labels = np.array( [[ 0, 0, 1],
                             [ 1, 1, 1],
                             [ 1, 1, 1]], dtype=int)
        mask = np.ones_like(labels)
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
#         print 'labels:'
#         print labels
        hc = count_homo_cliques(g, labels[np.where(mask)])
#         print '-> homo cliques:'
#         print hc
        assert hc == 9
        
    def test_swendsenwang(self):    
        
        nbLabels = 2
        shape = (5,5)
        mask = np.ones(shape, dtype=int)
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        labels = genPepperSaltField(mask.size, nbLabels)
        SwendsenWangSampler_graph(g, labels, 50.9, 2)
        mapPotts = np.ones_like(mask)
        mapPotts[np.where(mask)] = labels
#         print 'potts map:'
#         print mapPotts

    def test_potts_gibbs(self):
        nbLabels = 2
        shape = (15,15)
        mask = np.ones(shape, dtype=int)
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        labels = genPepperSaltField(mask.size, nbLabels).astype(np.int32)
        toroidal = 1
        beta = .1
        genPottsField(shape[0], shape[1], toroidal, beta, 100000, labels,
                      nbLabels)
        mapPotts = np.ones_like(mask)
        mapPotts[np.where(mask)] = labels
        #print 'potts map:'
        #print mapPotts



