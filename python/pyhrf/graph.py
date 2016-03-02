# -*- coding: utf-8 -*-
"""
Module to handle graphs.
Base structures :
- undirected, unweighted graph: a list of neighbours index (list of numpy array).
"""

import logging

import numpy as np

import pyhrf

from pyhrf.tools import cartesian
from pyhrf.boldsynth.spatialconfig import lattice_indexes
from pyhrf.boldsynth.spatialconfig import mask_to_coords
from pyhrf.ndarray import expand_array_in_mask


logger = logging.getLogger(__name__)


try:
    from pygraph.algorithms.searching import breadth_first_search as bfs_pygraph
    # raise ImportError() # for test
except ImportError, e:
    if 0 and pyhrf.__usemode__ == pyhrf.DEVEL:
        logger.warning('---------------------------------------------------')
        logger.warning('Warning: pygraph is not available. Maybe it is not installed.')
        logger.warning('You may try "easy_install python-graph-core" or the package'
                       '"python-pygraph" may be available from ubuntu reps.')
        logger.warning('---------------------------------------------------')
    pygraph_available = False
else:
    pygraph_available = True


def graph_is_sane(g, toroidal=False):
    """ Check the structure of graph 'g', which is a list of neighbours index.
    Return True if check is ok, else False.
    -> every nid in g[id] should verify id in g[nid]
    -> any neighbours list must have unique elements
    -> no isolated node
    """
    nn = len(g[0])
    for i, nl in enumerate(g):
        # if len(nl) == 0:
        #    print 'orphan node'
        #    return False
        if toroidal and len(nl) != nn:
            logger.info('not consistent with toroidal geometry')
            return False
        if np.unique(nl).size != nl.size:
            logger.info('duplicate neighbours, nl: %s', str(nl))
            return False
        for k in nl:
            # print 'k:', k
            # print 'nl:', nl
            if i not in g[k]:
                logger.info(
                    '%d is neighbour of %d but converse is not true', i, k)
                return False
    return True


def graph_from_mesh(polygonList):
    """Return the list of neighbours indexes for each position, from a list of
    polygons. Each polygon is a triplet.

    """

    indexSet = set()
    for l in polygonList:
        indexSet.update(l)
    nbPositions = len(indexSet)

    logger.debug('indexSet:')
    logger.debug(indexSet)

    neighbourSets = {}
    for triangle in polygonList:
        for idx in triangle:
            if not neighbourSets.has_key(idx):
                neighbourSets[idx] = set()
            neighbourSets[idx].update(triangle)

    # idx<->set --> idx<->list
    neighbourLists = np.empty(nbPositions, dtype=object)
    for idx in indexSet:
        neighbourSets[idx].remove(idx)
        neighbourLists[idx] = np.array(list(neighbourSets[idx]), dtype=int)

    return neighbourLists


def graph_nb_cliques(graph):
    return sum([len(x) for x in graph]) / 2.


def center_mask_at_v01(mask, pos, shape):
    new = range(len(mask))
    toKeep = set(range(len(mask[0])))
    for i, (d, p, s) in enumerate(zip(mask, pos, shape)):
        comp = d + p
        new[i] = comp
        toKeep.difference_update(np.where(comp < 0)[0])
        toKeep.difference_update(np.where(comp >= s)[0])
    toKeep = list(toKeep)
    # print 'toKeep:', toKeep
    # print 'new:', new
    return tuple([n[toKeep] for n in new])


def center_mask_at(mask, pos, indexes, toroidal=False):
    ndim = len(mask)
    # print 'ndim:', ndim
    new = range(ndim)
    # print 'new:', new
    toKeep = np.ones(len(mask[0]), dtype=bool)
    # print 'toKeep:', toKeep
    for ic in xrange(ndim):
        # print 'ic:', ic
        # print 'pos[ic]:', pos[ic]
        comp = mask[ic] + pos[ic]
        # print 'comp:', comp
        if not toroidal:
            insiders = np.bitwise_and(comp >= 0, comp < indexes.shape[ic])
            np.bitwise_and(toKeep, insiders, toKeep)
        else:
            comp[np.where(comp < 0)] = indexes.shape[ic] - 1
            comp[np.where(comp >= indexes.shape[ic])] = 0
            # print 'indexes.shape[ic]:', indexes.shape[ic]
            # print 'comp after modif:', comp
        new[ic] = comp
        # print 'new[ic]:', new[ic]
    #m = tuple( [n[toKeep] for n in new] )
    #np.bitwise_and(toKeep, indexes[m]!=-1, toKeep)
    # print 'toKeep:', toKeep
    # print 'new:', new
    return tuple([n[toKeep] for n in new])


def center_mask_at_v02(mask, pos, shape):
    new = [[] for n in xrange(len(mask))]
    # print ' mask:', mask
    # print ' pos:', pos
    # print 'new:', new
    ndim = len(mask)
    for i in xrange(len(mask[0])):
        ncoords = range(ndim)
        valid = True
        for ic in xrange(ndim):
            ncoord = mask[ic][i] + pos[ic]
            if ncoord < 0 or ncoord >= shape[ic]:
                valid = False
                break
            else:
                ncoords[ic] = ncoord
        if valid:
            for ic in xrange(ndim):
                new[ic].append(ncoords[ic])
    return tuple(new)


# kerMask3D_6n = np.array( [ [1,0,0]  ,  [0,1,0] , [0,0,1],
#                           [-1,0,0] , [0,-1,0] , [0,0,-1] ], dtype=int )
kerMask3D_6n = (np.array([1, 0, 0, -1, 0, 0], dtype=int),
                np.array([0, 1, 0, 0, -1, 0], dtype=int),
                np.array([0, 0, 1, 0, 0, -1], dtype=int))

#kerMask2D_4n = np.array( [ [-1,0], [1,0], [0,1], [0,-1] ], dtype=int )
kerMask2D_4n = (np.array([-1, 1, 0, 0], dtype=int),
                np.array([0, 0, 1, -1], dtype=int))

kerMask2D_8n = (np.array([-1, 1, 0, 0, -1, 1, -1, 1], dtype=int),
                np.array([0, 0, 1, -1, -1, 1, 1, -1], dtype=int))


def graph_pool_indexes(g):
    nodeMap = dict([(v, iv) for iv, v in enumerate(nodes)])
    for nl in subg:
        for i, n in enumerate(nl):
            nl[i] = nodeMap[n]


def flatten_and_graph(data, mask=None, kerMask=None, depth=1,
                      toroidal=False):

    if mask is None:
        mask = data > 0

    fdata = data[np.where(mask)]
    g = graph_from_lattice(mask, kerMask, depth, toroidal)
    return fdata, g

#---------------------------   Lotfi   -----------------------------#


def graph_from_lattice3D(mask, kerMask=None, depth=1, toroidal=False):
    """
    Creates a graph from a n-dimensional lattice
    'mask' define valid positions to build the graph over.
    'kerMask' is numpy array mask (tuple of arrays) which defines the
    neighbourhood system, ie the relative positions of neighbours for a given
    position in the lattice.
    """
    # print 'size:', latticeIndexes.size
    if kerMask is not None:
        assert mask.ndim == len(kerMask)
    else:
        # Full neighbourhood, ie 8 in 2D, 26 in 3D ...
        ndim = mask.ndim
        neighbourCoords = list(cartesian(*[[0, -1, 1]] * ndim))[1:]
        kerMask = tuple(np.array(neighbourCoords, dtype=int).transpose())

    # loop over valid positions:

    positions = np.array(np.where(mask >= 0)).transpose()
    # print "----------------------------"
    # print np.prod(mask.shape)
    # print positions.shape
    # print "----------------------------"
    # raw_input('')
    #positions = mask_to_coords(mask)
    latticeIndexes = lattice_indexes(mask)
    # print positions
    # print positions.shape
    # Build lists of closest neighbours:
    closestNeighbours = np.empty(len(positions), dtype=object)
    for idx, pos in enumerate(positions):
        # print 'idx :', idx, '- pos:', pos
        # O(n) :
        m = center_mask_at(kerMask, pos, latticeIndexes, toroidal)
        closestNeighbours[idx] = latticeIndexes[m][latticeIndexes[m] >= 0]

    if depth == 1:
        return closestNeighbours

    neighboursSets = copyModule.deepcopy(closestNeighbours)

    for idx in xrange(nbPositions):
        # Starting positions = closest neighbours of current position
        neighboursToTreat = closestNeighbours[idx]
        visited = set([idx])
        # O(max( sum(lattice shape), kerMask size))*O(kerMask size)*O(d)
        for d in xrange(depth - 1):
            newNeighboursToTreat = set()
            # In the worst case, neighbours to treat are on the perimeter
            # of the lattice, say O(sum(lattice shape))
            # OR
            # If the lattice is kind of flat, the worst case is
            # O(size of kerMask)
            # So complexity is :
            # O(max( sum(lattice shape), kerMask size)) * O(kerMask size)
            for n in neighboursToTreat:
                # O(kerMask size)
                newNeighboursToTreat.update(closestNeighbours[n])
            # Remember what we just saw:
            visited.update(neighboursToTreat)
            # Update neighbours of current position and remove those
            # already seen:
            newNeighboursToTreat.difference_update(visited)
            neighboursSets[idx].update(newNeighboursToTreat)
            # Define new starting positions for next loop:
            neighboursToTreat = newNeighboursToTreat

    return neighboursSets

#--------------------------------   Lotfi   -------------------------#


def graph_to_sparse_matrix(graph):
    """
    Creates a connectivity sparse matrix from the adjacency graph
    (list of neighbors list)
    """
    from scipy.sparse.coo import coo_matrix
    n_vox = len(graph)

    ij = [[], []]
    for i in xrange(n_vox):
        ij[0] += [i] * len(graph[i])
        ij[1] += list(graph[i])

    return coo_matrix((np.ones(len(ij[0]), dtype=int), ij))


def graph_from_lattice(mask, kerMask=None, depth=1, toroidal=False):
    """
    Creates a graph from a n-dimensional lattice
    'mask' define valid positions to build the graph over.
    'kerMask' is numpy array mask (tuple of arrays) which defines the
    neighbourhood system, ie the relative positions of neighbours for a given
    position in the lattice.
    """
    # print 'size:', latticeIndexes.size
    if kerMask is not None:
        assert mask.ndim == len(kerMask)
    else:
        # Full neighbourhood, ie 8 in 2D, 26 in 3D ...
        ndim = mask.ndim
        neighbourCoords = list(cartesian(*[[0, -1, 1]] * ndim))[1:]
        kerMask = tuple(np.array(neighbourCoords, dtype=int).transpose())

    # loop over valid positions:
    # print mask.shape
    positions = mask_to_coords(mask)
    latticeIndexes = lattice_indexes(mask)
    # print positions
    # print positions.shape
    # Build lists of closest neighbours:
    closestNeighbours = np.empty(len(positions), dtype=object)
    for idx, pos in enumerate(positions):
        # print 'idx :', idx, '- pos:', pos
        # O(n) :
        m = center_mask_at(kerMask, pos, latticeIndexes, toroidal)
        # print 'm:', m
        closestNeighbours[idx] = latticeIndexes[m][latticeIndexes[m] >= 0]

    if depth == 1:
        return closestNeighbours

    neighboursSets = copyModule.deepcopy(closestNeighbours)

    for idx in xrange(nbPositions):
        # Starting positions = closest neighbours of current position
        neighboursToTreat = closestNeighbours[idx]
        visited = set([idx])
        # O(max( sum(lattice shape), kerMask size))*O(kerMask size)*O(d)
        for d in xrange(depth - 1):
            newNeighboursToTreat = set()
            # In the worst case, neighbours to treat are on the perimeter
            # of the lattice, say O(sum(lattice shape))
            # OR
            # If the lattice is kind of flat, the worst case is
            # O(size of kerMask)
            # So complexity is :
            # O(max( sum(lattice shape), kerMask size)) * O(kerMask size)
            for n in neighboursToTreat:
                # O(kerMask size)
                newNeighboursToTreat.update(closestNeighbours[n])
            # Remember what we just saw:
            visited.update(neighboursToTreat)
            # Update neighbours of current position and remove those
            # already seen:
            newNeighboursToTreat.difference_update(visited)
            neighboursSets[idx].update(newNeighboursToTreat)
            # Define new starting positions for next loop:
            neighboursToTreat = newNeighboursToTreat

    return neighboursSets


def breadth_first_search(graph, root=0, visitable=None):
    """Traverses a graph in breadth-first order.

    The first argument should be the tree root; visitable should be an
    iterable with all searchable nodes;
    """

    if visitable is None:
        visitable = range(len(graph))  # all nodes are visitable

    # makes a shallow copy, makes it a collection, removes duplicates
    unvisited = list(set(visitable))

    queue = []
    if root in unvisited:
        unvisited.remove(root)
        queue.append(root)

    order = []
    while len(queue) > 0:
        node = queue.pop(0)
        order.append(node)

        for child in graph[node]:
            if child in unvisited:
                unvisited.remove(child)
                queue.append(child)

    return None, order  # return None to be compliant with pygraph which also
    # return the spanning tree


def graph_pygraph(g):
    from pygraph.classes.graph import graph
    gr = graph()
    gr.add_nodes(range(len(g)))
    for inode, nl in enumerate(g):
        for ineighbour in nl:
            if ineighbour > inode:
                gr.add_edge((inode, ineighbour))
    return gr


def connected_components_iter(g):
    if pygraph_available:
        g = graph_pygraph(g)
        bfs = bfs_pygraph
    else:
        logger.warning('Warning: pygraph not available ... fall back to slow BFS function')
        bfs = breadth_first_search

    visited = np.zeros(len(g), dtype=bool)
    root = 0
    while not visited.all():
        root = np.where(visited == False)[0][0]
        # print 'root:', root
        st, order = bfs(g, root=root)
        visited[order] = True
        yield sorted(order)


def connected_components(g):

    if pygraph_available:
        g = graph_pygraph(g)
        bfs = bfs_pygraph
    else:
        logger.warning('Warning: pygraph not available ... fall back to slow BFS function')
        bfs = breadth_first_search

    if 1:
        visited = np.zeros(len(g), dtype=bool)
        components = []
        root = 0
        while not visited.all():
            root = np.where(visited == False)[0][0]
            st, order = bfs(g, root=root)
            visited[order] = True
            components.append(sorted(order))
        return components
    else:
        from pygraph.algorithms.accessibility import connected_components as cc
        return cc(g)

if pygraph_available:
    def connected_components_labeled(g, labels):

        gr = graph_pygraph(g)

        counts = np.bincount(labels)
        to_visit = set(range(len(gr)))

        class label_filter(object):
            def __init__(self, l, labels):
                self.l = l
                self.labels = labels

            def configure(self, gr, s):
                pass

            def __call__(self, node, parent):
                return self.labels[node] == self.l

        components = dict([(i, []) for i in np.unique(labels)])
        while len(to_visit) > 0:
            root = to_visit.pop()
            label = labels[root]
            st, o = bfs_pygraph(gr, root, label_filter(label, labels))
            components[label].append(o)
            to_visit.difference_update(o)

        return components


def split_mask_into_cc_iter(mask, min_size=0, kerMask=None):
    """ Return an iterator over all connected components (CC) within input mask.
    CC which are smaller than min_size are discarded. 'kerMask' defines the connexity,
    eg kerMask3D_6n for 6-neighbours in 3D.
    Example:
    vol = np.array( [[1,1,0,1,1],
                     [1,1,0,1,1],
                     [0,0,0,0,0],
                     [1,0,1,1,0],
                     [0,0,1,1,0]], dtype=int )
    for cc in split_mask_into_cc_iter(vol):
        print cc

    Should output:
    np.array( [[1,1,0,0,0],
               [1,1,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]]
    np.array( [[0,0,0,1,1],
               [0,0,0,1,1],
               [0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]]
    ...
    """
    assert isinstance(min_size, int)
    flat_mask = mask[np.where(mask)]
    # print 'flat_mask:', flat_mask.shape
    g = graph_from_lattice(mask, kerMask)
    for cc in connected_components_iter(g):
        # print 'cc:'
        # print cc
        if len(cc) >= min_size:
            flat_mask[:] = 0
            flat_mask[cc] = 1
            # print expand_array_in_mask(flat_mask, mask)
            yield expand_array_in_mask(flat_mask, mask)


def bfs_set_label(g, root, data, value, radius):
    from pygraph.algorithms.filters.radius import radius as Radius
    from pygraph.algorithms.searching import breadth_first_search
    gr = graph_pygraph(g)
    filt_radius = Radius(radius)
    _, o = breadth_first_search(gr, root, filt_radius)
    data[o] = value


def bfs_sub_graph(g, root, radius):
    from pygraph.algorithms.filters.radius import radius as Radius
    from pygraph.algorithms.searching import breadth_first_search
    gr = graph_pygraph(g)
    filt_radius = Radius(radius)
    _, o = breadth_first_search(gr, root, filt_radius)
    to_keep = sorted(o)
    new_indexes = np.zeros(len(g))
    new_indexes[to_keep] = range(len(to_keep))
    subg = g[to_keep]
    for nl in subg:
        for i in xrange(len(nl)):
            nl[i] = new_indexes[nl[i]]
    return subg, to_keep


def sub_graph(graph, nodes):
    sNodeIds = set(nodes)
    # Filter the graph to only keep given nodes:
    subg = np.array([np.array(list(sNodeIds.intersection(nl)))
                     for nl in graph[nodes]], dtype=object)
    # Fix indexes to be in the new referential:
    nodeMap = dict([(v, iv) for iv, v in enumerate(nodes)])
    for nl in subg:
        for i, n in enumerate(nl):
            nl[i] = nodeMap[n]
    return subg, nodeMap


def parcels_to_graphs(parcellation, kerMask, toKeep=None,
                      toDiscard=None):
    """
    Compute graphs for each parcel in parcels. A graph is simply defined as
    a list of neihbour indexes.
    'parcellation' is a n-ary numpy array.
    'kerMask' defines the connectivity
    Return :
     - a dictionnary mapping a roi ID to its graph
    """
    # determine the set of parcels to work on:
    parcelIds = np.unique(parcellation)  # everything
    if toKeep is not None:
        assert set(toKeep).issubset(set(parcelIds))
        parcelIds = toKeep
    if toDiscard is not None:
        #assert set(toDiscard).issubset(set(parcelIds))
        parcelIds = set(parcelIds).difference(toDiscard)
    # print 'parcelIds:', parcelIdsx

    parcelGraphs = {}
    parcelNodeMap = {}
    # loop over parcels
    for pid in parcelIds:
        parcelGraphs[pid] = graph_from_lattice(parcellation == pid,
                                               kerMask=kerMask)
        # print 'pg:', parcelGraphs[pid]
        assert graph_is_sane(parcelGraphs[pid])

    return parcelGraphs
