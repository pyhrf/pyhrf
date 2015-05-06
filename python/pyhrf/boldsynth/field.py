# -*- coding: utf-8 -*-


import numpy as _np

from pyhrf.boldsynth.spatialconfig import *
#from pyhrf.boldsynth.pottsfield import pottsfield_c
from pyhrf.boldsynth.pottsfield.swendsenwang import *

def genPepperSaltField(size, nbLabels, initProps=None):
    if initProps is None:
        # equirepartition:
        initProps = [1./nbLabels] * nbLabels
    else:
        assert nbLabels==len(initProps) or nbLabels==len(initProps)-1

    data = _np.array([], dtype=int)
    for ic in xrange(nbLabels-1):
        cs = _np.round(size * initProps[ic])
        data = _np.hstack((data, _np.zeros(cs) + ic))
    # Calculate last class size from the sum of the others :
    cs = size - len(data)
    data = _np.hstack((data, _np.zeros(cs) + nbLabels - 1))    
    return _np.random.permutation(data).astype(int)

class random_field_generator:
    def __init__(self, size, nbClasses):
        self.size = size
        self.nbClasses = nbClasses
    def __call__(self):
        return _np.random.randint(0, self.nbClasses, self.size)
    def __iter__(self):
        yield self.__call__()

def genPotts(graph, beta, nbLabels=2, labelsIni=None, method='SW',
             weights=None):
    """
    Simulate a realisation of a Potts Field with spatial correlation amount 
    'beta'.
    'graph' is list of lists, ie a neighbors for each node index.
    'nbLabels' is the number of labels
    'method' can be either 'SW' (swensdsen-wang) or 'gibbs'
    """
    
    if method == 'SW':
        nbIt = 30
        if labelsIni is None:
            # start from a pepper n salt configuration
            labels = genPepperSaltField(len(graph), nbLabels)
        else:
            labels = labelsIni
        #print 'g:',graph
        #print 'labels:', labels
        GraphBetaMix(graph, labels, beta, nbLabels, nbIt, weights)
        return labels
    else:
        raise NotImplementedError('genPotts for method '+method)

def potts_generator(**args):
    while True:
        yield genPotts(**args)


def genPottsMap(mask, beta, nbLabels, method='SW'):
    # build a neighbourhoodSystem over mask
    if mask.ndim == 3:
        kmask = NeighbourhoodSystem.kerMask3D_6n
        ns = NeighbourhoodSystem.fromLattice(mask, kerMask=kmask)
    elif mask.ndim == 2:
        kmask = NeighbourhoodSystem.kerMask2D_2n
        ns = NeighbourhoodSystem.fromLattice(mask, kerMask=kmask)
    else:
        raise Exception("Unsupported mask shape")
    # get the corresponding graph:
    g = ns.getNeighboursLists()

    ## generate a Potts realisation:
    labels = genPotts(g, beta, nbLabels)
    labelVolume = _np.zeros(mask.shape, dtype=int)
    labelVolume[_np.where(mask)] = labels    

    return labelVolume

def count_homo_cliques(graph, labels, weights=None):
    sum = 0.
    for i,nl in enumerate(graph):
        if weights is None:
            sum += (labels[i] == labels[nl]).sum()
        else:
            sum += ((labels[i] == labels[nl])*weights[i]).sum()
    return sum/2 #we parsed twice the number of cliques

