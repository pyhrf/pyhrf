

# -*- coding: utf-8 -*-
import numpy as NP
import copy as copyModule
import hashlib
from numpy import cos, sin
import pyhrf
from pyhrf.tools import cartesian

from pyhrf.ndarray import *


def hashMask(m):
    return hashlib.sha512(m.astype(int).tostring()).hexdigest()

def getRotationMatrix(axis, angle):
    """
    Compute the  3x3 matrix for the 3D rotation defined by 'angle' and the
    direction 'axis'.
    """
    ca = cos(angle)
    sa = sin(angle)
    mc = 1-ca

    v = NP.array(axis)
    n = NP.sqrt(NP.sum(v**2))
    # Ensure that axis is a unit vector :
    x = v[0]/n
    y = v[1]/n
    z = v[2]/n
    return NP.array( [ [ ca+mc*x*x  , mc*x*y-sa*z , mc*x*z+sa*y ],
                       [ mc*y*x+sa*z , ca+mc*y*y  , mc*y*z-sa*x ],
                       [ mc*z*x-sa*y , mc*z*y+sa*x , ca+mc*z*z  ] ] )

####################
# Spatial Mappings #
####################

#TODO: maybe make it recursive to manage different flatten level?
# -> better to use chain iterator
def flattenElements(l):
    dest = []
    for le in l:
        dle = []
        for e in le:
            if np.iterable(e):
                dle.extend(e)
            else:
                dle.append(e)
        dest.append(dle)
    return dest


class Mapper1D:
    """
    Handles a mapping between a nD coordinate space (expanded) and a 1D
    coordinate space (flatten). Can be applied to numpy.ndarray objects
    """

    def __init__(self, mapping, expandedShape):
        """
        Initialise a Mapper1D:
        'mapping' is a list of coordinates in the expanded space.
        'expandedShape' is the shape of the expanded space.
        """
        assert len(mapping[0]) == len(expandedShape)
        self.mapping = mapping
        self.expandedShape = expandedShape

    def flattenArray(self, array, firstMappedAxis=0):
        """
        Reduce dimensions of 'array'. 'firstMappedAxis' is index of the axis to
        be reduced (other mapped axes are assumed to follow this one).
        """
        nd = len(self.expandedShape)
        ia = firstMappedAxis
        assert len(self.expandedShape) <= array.ndim
        #print 'self.expandedShape :', self.expandedShape
        #print 'array.shape :' , array.shape
        #print 'array.shape[ia:ia+nd]:', array.shape[ia:ia+nd]
        assert self.expandedShape == array.shape[ia:ia+nd]
        ranges = [range(d) for d in array.shape]
        
        ranges = ranges[:ia] + [self.mapping] + ranges[ia+nd:]
        coords = cartesian(*ranges)
        coords = flattenElements(coords)
        #TODO: remove explicit loop!
        mask = [NP.array(a) for a in NP.array(coords).transpose()]
        destshape = array.shape[:ia]+(len(self.mapping),)+array.shape[ia+nd:]
        destData = NP.empty(destshape, dtype=array.dtype)
        destData.flat = array[mask]
        return destData

    def createExpandedArray(self, flatShape, type, mappedAxis=0, fillValue=0):
        ia = mappedAxis
        destshape = flatShape[:ia] + self.expandedShape + flatShape[ia+1:]
        pyhrf.verbose(6,'dest shape: %s' %str(destshape))
        return NP.zeros(destshape, dtype=type) + fillValue
        
    def expandArray(self, a, mappedAxis=0, dest=None, fillValue=0):
        """
        Expand dimensions of 'a' following predefined mapping. 'mappedAxis'
        is the axis index to be expanded in 'a'.
        If dest is not None the map values from 'a' to dest.
        If dest is None then return a new array and fill positions not involved
        in mapping with 'fillValue'.
        """
        ia = mappedAxis
        m = self.mapping.transpose()
        lmask = [':' for i in xrange(a.ndim+len(m)-1)]
        for i in xrange(len(m)):
            lmask[ia+i] = 'm[%d]' %i

        if dest is None:
            dest = self.createExpandedArray(a.shape, a.dtype, mappedAxis,
                                            fillValue)

        # print 'a b4', a.shape
        # print a
        # print 'mappedAxis:', mappedAxis
        # print 'mapping:'
        # print len(self.mapping)

        if a.shape[mappedAxis] == 0 and len(self.mapping) > 1: 
            #assume diplucation
            #print 'diplucation'
            a = NP.repeat(a,len(self.mapping[0]), mappedAxis)
            
        sexec = "dest[%s] = a" %','.join(lmask)
        # print 'dest:'
        # print dest.shape
        # print 'a'
        # print a.shape
        # print 'sexec:'
        # print sexec
        exec(sexec)

        return dest

class xndarrayMapper1D(Mapper1D):

    def __init__(self, mapping, expandedShape, expandedAxesNames, flatAxisName):
        Mapper1D.__init__(self, mapping, expandedShape)
        self.expandedAxesNames = expandedAxesNames
        self.flatAxisName = flatAxisName

    def isExpendable(self, c):
        return self.flatAxisName in c.getAxesNames()

    def isFlattenable(self, c):
        return set(self.expandedAxesNames).issubset(c.getAxesNames())

    def flattenxndarray(self, c):

        expDim = len(self.expandedAxesNames)
        fia = c.getAxisId(self.expandedAxesNames[0])
        can = c.getAxesNames()
        assert (can[fia:fia+expDim] == self.expandedAxesNames)

        destAN = can[:fia] + [self.flatAxisName] + can[fia+expDim:]
        destAD = c.getAxesDomainsDict()
        for a in self.expandedAxesNames:
            destAD.pop(a, None)
        flatData = self.flattenArray(c.data, fia)
        flatErrors = None
        if c.errors is not None:
            flatErrors = self.flattenArray(c.errors, fia)
        return xndarray(flatData, flatErrors, axes_names=destAN, axes_domains=destAD,
                      value_label=c.value_label)
        
        
    def expandxndarray(self, c, dest=None, fillValue=0):
        pyhrf.verbose(4,'expandxndarray ...')
        ia = c.getAxisId(self.flatAxisName)
        can = c.getAxesNames()
        pyhrf.verbose(4,'can : %s' %str(can))
        nd = len(self.expandedAxesNames)
        destAN = can[:ia] + self.expandedAxesNames + can[ia+1:]
        pyhrf.verbose(4,'destAN : %s' %str(destAN))
        destAD = c.getAxesDomainsDict()
        destAD.pop(self.flatAxisName, None)
        if dest is None:
            destData = self.createExpandedArray(c.data.shape, c.data.dtype,
                                                ia, fillValue)
            destErrors = None
            if c.errors is not None:
                destErrors = self.createExpandedArray(c.data.shape,
                                                      c.errors.dtype, ia,
                                                      fillValue)
            dest = xndarray(destData, destErrors, axes_names=destAN,
                          axes_domains=destAD, value_label=c.value_label)
        
        pyhrf.verbose(4,'src shape : %s, -> des shape: %s' \
                          %(str(c.data.shape),str(dest.data.shape)))
        Mapper1D.expandArray(self, c.data, ia, dest.data, fillValue)
        if c.errors is not None:
            Mapper1D.expandArray(self, c.errors, ia, dest.errors, fillValue)

        return dest

def lattice_indexes(mask):
    mpos = NP.where(mask)
    #print 'lattice_indexes'
    #print mask.size
    indexes = NP.zeros_like(mask).astype(int) -1
    #print indexes.size
    indexes[mpos] = range(len(mpos[0]))
    #print len(mpos[0])
    return indexes

# class LatticeIndexMapper:
#     """
#     Handle mapping between the position in a Lattice and the index node.
#     """

#     def __init__(self, indexes, torusFlag=False):
#         self.indexes = indexes        
#         self.torusFlag = torusFlag
#         #print 'indexes :', indexes
#         self.positions = mask_to_coords(indexes != -1)
#         #print 'positions:', self.positions

#     def indexFromCoord(self, coords):
#         #print "LatticeIndexMapper.__call__"
#         #print ' got coords :', coords
#         sh = self.indexes.shape
#         if self.torusFlag: #handle coords that exceed shape
#             ccoords = coords - .5*(coords/NP.abs(coords)-1) \
#                 * NP.ceil(coords/sh) * sh
#             coords = ccoords - NP.floor(ccoords/sh)*sh

#         if (coords>=0).all() and (coords<sh).all():
#             #print 'returning :', self.indexes[NP.split(coords.astype(int),len(sh))][0]
#             return self.indexes[NP.split(coords.astype(int),len(sh))][0]
#         else:
#             #print 'returning none'
#             return None

#     def coordFromIndex(self, index):
#         return self.positions[index]

#     def indexesFromMask(self, mask):
#         assert mask.shape == self.indexes.shape
#         return self.indexes[mask]

class NeighbourhoodSystem:

    kerMask3D_6n = NP.array( [ [1,0,0]  ,  [0,1,0] , [0,0,1],
                               [-1,0,0] , [0,-1,0] , [0,0,-1] ], dtype=int )

    kerMask2D_4n = NP.array( [ [-1,0], [1,0], [0,1], [0,-1] ], dtype=int )
    
    
    def __init__(self, neighboursSets):
        self.neighboursSets = neighboursSets
        #self.nbPositions = len(neighboursSets)
        self.maxNeighbours = None
        self.neighboursArray = None
        
    def getNeighboursLists(self):
        return NP.asarray([list(ns) for ns in self.neighboursSets], dtype=object)
    
    def getNeighboursSets(self):
        return self.neighboursSets
    
    def getMaxNeighbours(self):
        if self.maxNeighbours == None:
            m = 0
            #for nrl in self.neighboursSets.itervalues():
            for nl in self.neighboursSets:
                m = len(nl) if len(ln)>m else m
            self.maxNeighbours = m
        return self.maxNeighbours

    def getMaxIndex(self):
        if self.maxIndex == None:
            #self.maxIndex = max(self.neighboursSets.keys())
            self.maxIndex = len(self.neighboursSets)-1
        return self.maxIndex

    def getNeighboursArrays(self):
        if self.neighboursArray == None:
            mn = self.getMaxNeighbours()
            mi = self.getMaxIndex()
            self.neighboursArray = NP.zeros((mi+1,mn),dtype=int)-1
            idx = 0
            #for idx,nl in self.neighboursSets.iteritems():
            for nl in self.neighboursSets:
                self.neighboursArray[idx,:len(nl)] = nl
                idx += 1
        return self.neighboursArray
    
    def getNeighbours(self, nodeId):
        return self.neighboursSets[nodeId]


    def sub(self, nodeIds):

        # Filter neighbourslists to only keep given nodes:
        sNodeIds = set(nodeIds)
##        print 'sub with nodeIds :', nodeIds
        newNL = NP.array([sNodeIds.intersection(nl)
                          for nl in self.neighboursSets[nodeIds]])
        #newNL = dict(zip(nodeIds), newNL)

        # Make nodeIds = [0,1,2, ...]
        #newIds = [0]*max(nodeIds)
        #newIds[nodeIds] = range(len(nodeIds))
        #for ni,nl in newNL.items():
        #    self.neighboursSets[newIds]

        return NeighbourhoodSystem(newNL)
        

    @staticmethod
    def fromMesh(polygonList):
        nls = graphFromMesh(polygonList)
        ns = NP.empty(max(nl.iterkeys()), dtype=object)
        for idx, nl in nls:
            ns[idx] = nl
        return NeighbourhoodSystem(ns)

    @staticmethod
    def fromLattice(latticeIndexes, kerMask=None, depth=1, torusFlag=False):
        """
        Creates a NeighbourhoodSystem instance from a n-dimensional lattice
        """
        mpos = NP.where(latticeIndexes != -1)
        positions = NP.vstack(mpos).transpose()


        if kerMask == None:
            # Full neighbourhood, ie 8 in 2D, 26 in 3D ...
            ndim = latticeIndexes.ndim
            kerMask = NP.array(list(cartesian(*[[0,-1,1]]*ndim))[1:], dtype=int)

        #indexLattice = NP.zeros_like(maskLattice)
        #indexLattice[mpos] = range(nbPositions)
        
        mapper = LatticeIndexMapper(latticeIndexes, torusFlag)
        nbPositions = latticeIndexes.size

        # Build lists of closest neighbours:
        closestNeighbours = NP.empty(nbPositions, dtype=object)
        for idx, pos in enumerate(positions):
            #print 'idx :', idx, '- pos:', pos
            # O(n) :
            #print 'kerMask+pos =', kerMask+pos 
            ni = map(mapper.indexFromCoord, kerMask+pos)
            closestNeighbours[idx] = set(filter(lambda x: x is not None, ni))
            #print '->closestNeighbours[%d]=%s'  %(idx,closestNeighbours[idx])

        if depth == 1:
            return NeighbourhoodSystem(closestNeighbours)

        neighboursSets = copyModule.deepcopy(closestNeighbours)
        
        for idx in xrange(nbPositions):
##            print 'Treating idx :', idx
            # Starting positions = closest neighbours of current position
            neighboursToTreat = closestNeighbours[idx]
##            print ' neighboursToTreat =', neighboursToTreat
            visited = set([idx])
##            print ' already visited =', visited
            # O(max( sum(lattice shape), kerMask size))*O(kerMask size)*O(d)
            for d in xrange(depth-1):
##                print '  d=', d
##                print '  We have to treat:', neighboursToTreat
                newNeighboursToTreat = set()
                # In the worst case, neighbours to treat are on the perimeter
                # of the lattice, say O(sum(lattice shape))
                # OR
                # If the lattice is kind of flat, the worst case is
                # O(size of kerMask)
                # So complexity is :
                # O(max( sum(lattice shape), kerMask size)) * O(kerMask size)
                for n in neighboursToTreat:
##                    print '   processing neighbour:', n
##                    print '   updating to treat with;', closestNeighbours[n]
                    # O(kerMask size)
                    newNeighboursToTreat.update(closestNeighbours[n])
                # Remember what we just saw:
##                print '  We have just visited:', neighboursToTreat
                visited.update(neighboursToTreat)
##                print '  So all visited are;', visited
                # Update neighbours of current position and remove those
                # already seen:
                newNeighboursToTreat.difference_update(visited)
##                print '  neighbours which are left to treat:', \
##                      newNeighboursToTreat
                neighboursSets[idx].update(newNeighboursToTreat)
##                print '  list of neighbours for %d is now: %s' \
##                      %(idx, str(neighboursSets[idx]))
                # Define new starting positions for next loop:
                neighboursToTreat = newNeighboursToTreat
        
        return NeighbourhoodSystem(neighboursSets)
    

class SpatialMapping2:
    
    
    def __init__(self, positions, ns, parentMapping=None, parentIndex=None):
        self.ns = ns
        self.positions = positions
        self.parent = parentMapping
        self.parentIndex = parentIndex

    def getPositions(self):
        return self.positions

    def sub(self, nodeIds):
        #subPos = dict(zip(nodeIds,[self.positions[n] for n in nodeIds]))
        subPos = self.positions[nodeIds]
        return SpatialMapping2(subPos, self.ns.sub(nodeIds), self)
    
    @staticmethod
    def fromMesh(triangles, positions):
        return GraphMapping(NeighbourhoodSystem.fromMesh(triangles),positions)

    @staticmethod
    def fromLattice(lattice, kerMask=None, nsDepth=1, torusFlag=False):
        mpos = NP.where(lattice)
        #positions = dict(zip(range(len(mpos[0])),
        #                     NP.vstack(mpos).transpose()))
        positions = NP.vstack(mpos).transpose()
        ns = NeighbourhoodSystem.fromLattice(lattice, kerMask=kerMask,
                                             depth=nsDepth,  torusFlag=torusFlag)
        return SpatialMapping2(ns, positions)


class RegularLatticeMapping2(SpatialMapping2):

    def __init__(self, maskLattice, kerMask=None, nsDepth=1, parentMapping=None,
                 torusFlag=False):
        self.nbDims = maskLattice.ndim
        self.targetShape = maskLattice.shape
        indexLattice = NP.zeros_like(maskLattice)
        m = NP.where(maskLattice)
        ids = range(len(m[0]))
        indexLattice[m] = ids
        #positions = dict(zip(ids, NP.vstack(m).transpose()))
        positions = NP.vstack(m).transpose()
        ns = NeighbourhoodSystem.fromLattice(indexLattice, kerMask=kerMask,
                                             depth=nsDepth, torusFlag=torusFlag)
        SpatialMapping2.__init__(self, positions, ns)

    def flattenData(self, data, firstMappedAxis=0):
        #Note: very similar to pyhrf.ndarray.buildMappingMask ...
        #TODO: factorize axis expansion stuffs !!!
        ranges = [range(d) for d in data.shape]
        ia = firstMappedAxis
        nd = self.nbDims
        ranges = ranges[:ia] + [self.positions] + ranges[ia+nd:]
        coords = cartesian(*ranges)
        coords = flattenElements(coords)
        #TODO: remove explicit loop!
        mask = [NP.array(a) for a in NP.array(coords).transpose()]
        destshape = data.shape[:ia]+(len(self.positions),)+data.shape[ia+nd:]
        destData = NP.empty(destshape, dtype=data.dtype)
        destData.flat = data[mask]
        return destData
        

    def mapData(self, data, mappedAxis=0, fillValue=0):
        ia = mappedAxis
        nd = self.nbDims
        ranges = [range(d) for d in data.shape]
        ranges = ranges[:ia] + [self.positions] + ranges[ia+1:]
        coords = cartesian(*ranges)
        coords = flattenElements(coords)
        #TODO: remove explicit loop!
        mask = [NP.array(a) for a in NP.array(coords).transpose()]        
        destshape = data.shape[:ia] + self.targetShape + data.shape[ia+1:]
        destData = NP.zeros(destshape, dtype=data.dtype) + fillValue
        destData[mask] = data.flat
        return destData



class SpatialMapping:
    """
    Interface specification for the handling of a mapping between integer indexes
    and positions in a 3D space.
    """

    def getNbVoxels(self):
        """
        Return the total number of mapped position
        """
        raise NotImplementedError

    def getNdArrayMask(self):
        """
        Return the set of mapped 3D coordinates in a tuple usable as a mask for
        numpy.ndarray
        """
        raise NotImplementedError

    def getRoiMask(self):
        """
        Return a binary or n-ary mask which has the shape of the target data
        """
        raise NotImplementedError


    def getMapping(self):
        """
        Return a mapping object (list or dict) which maps an integer index to
        its 3D coordinates.
        """
        raise NotImplementedError

    def getIndex(coord):
        """
        Return index mapped with 'coord'
        """
        raise NotImplementedError

    def getCoord(index):
        """
        Return coord mapped with 'index'
        """
        raise NotImplementedError


    def getNeighboursIndexes(self, idVoxel):
        """
        @param idVoxel: index of the voxel
        @return: the list of integer indexes corresponding to the neighbours
        of the specified voxel.
        """
        raise NotImplementedError

    def getNeighboursIndexLists(self):
        """
        Get lists of neighbours for all positions
        @param idVoxel: index of the voxel
        @return: a mapping object (list or dict) which maps each integer index
        to a list of integer indexes (the neighbours).
        """
        raise NotImplementedError

    def getNeighboursCoords(self, idVoxel):
        """
        @param idVoxel: index of the voxel
        @return: the list of 3D coordinates corresponding to the neighbours
        of the specified voxel.
        """
        raise NotImplementedError


    def getNeighboursCoordLists(self):
        """
        Get lists of neighbours for all positions
        @param idVoxel: index of the voxel
        @return: a mapping object (list or dict) which maps each integer index
        to a list of 3D coordinates (the neighbours).
        """
        raise NotImplementedError


class UnboundSpatialMapping(SpatialMapping):
    """
    Convinient class to provide an implementation of SpatialMapping when there
    is no mapping.
    """

##    P_NBVOX = 'nbVoxels'

##    variables = SortedDictionary(
##        (P_NBVOX, { 'type': 'int', 'defaultValue': 100, 'optional': False, 'display': True }),
##    )

    def __init__(self,nbVoxels=100):
##        initClass(self, kwargs)
##        nbVoxels = getattr(self,self.P_NBVOX)
        self.nbVoxels = nbVoxels
        self.neighboursIndexLists = [ [] for i in xrange(nbVoxels) ]
        self.neighboursCoordLists = [ [] for i in xrange(nbVoxels) ]

    def __getinitkwargs__(self):
        return getInit(self, self.variables)

    def getNbVoxels(self):
        return self.nbVoxels

    def getMapping(self):
        return None # TODO check usability

    def getNdArrayMask(self):
        return None # TODO check usability

    def getRoiMask(self):
        """
        Return a binary or n-ary 3D mask which has the shape of the target data
        """
        return NP.ones((1,1,self.nbVoxels), dtype=int)
    
    def getIndex(self, coord):
        return None

    def getCoord(self, index):
        return None

    def getNeighboursIndexes(self, idVoxel):
        return []

    def getNeighboursCoords(self, idVoxel):
        return []

    def getNeighboursIndexLists(self):
        return self.neighboursIndexLists

    def getNeighboursCoordLists(self):
        return self.neighboursCoordLists

    @staticmethod
    def createFromGUI(GUIobject):
        """
        Creates the actual object based on the parameters
        """
        params = {}
        params[UnboundSpatialMapping.P_NBVOX] = getattr(GUIobject,UnboundSpatialMapping.P_NBVOX)
        return UnboundSpatialMapping(**params)




class RegularLatticeMapping(SpatialMapping):
    """
    Define a SpatialMapping on a 3D regular lattice.
    """
    
    order1Mask = NP.array( [ [1,0,0]  ,  [0,1,0] , [0,0,1],
                             [-1,0,0] , [0,-1,0] , [0,0,-1] ] )
    nbNeighboursOrder1 = 6

    order2Mask = NP.array([c for c in cartesian([0,-1,1],[0,-1,1],[0,-1,1])][1:])
    nbNeighboursOrder2 = 26


##    P_XSIZE= 'Xsize'
##    P_YSIZE= 'Ysize'
##    P_ZSIZE= 'Zsize'
##    P_ORDER= 'order'
##    P_DEPTH= 'depth'
##    P_MAPPING= 'mapping'


##    variables = SortedDictionary(
##        (P_XSIZE, { 'type': 'int', 'defaultValue': 1, 'optional': True, 'display': True }),
##        (P_YSIZE, { 'type': 'int', 'defaultValue': 10, 'optional': True, 'display': True }),
##        (P_ZSIZE, { 'type': 'int', 'defaultValue': 10, 'optional': True, 'display': True }),
##        (P_ORDER, { 'type': 'int', 'defaultValue': 1, 'optional': True, 'display': True }),
##        (P_DEPTH, { 'type': 'int', 'defaultValue': 1, 'optional': True, 'display': True }),
##        (P_MAPPING, { 'type': 'list', 'defaultValue': None, 'optional':True, 'display':False }),
##        ### TODO AL: handle mapping better
##        ### TODO AL: hide advanced options
##    )

    # TODO : add handling of periodic coordinate systems like a torus ...
    # TODO : check diff between order and depth ...
    def __init__(self, shape=None, mapping=None, order=1, depth=1): #, **kwargs):
        """
        Construct a spatial configuration corresponding to a regular 3D lattice.
        'shape' is a 3-item sequence defining the X, Y and Z sizes.
        Note on aims dimension conventions :
        (axial, coronal, sagittal)
        'mapping' is a list of integer coordinates defining which positions
                are valid in the regular lattice.
        'order' is the order of the neighbour system in 3D. Examples :
                - if order=1 => 6 neighbour system (with depth=1)
                - if order=2 => 26 neighbour system (with depth=1)
        'depth' is the depth of the neighbour system.

        If 'mapping'=None, the default lattice is full and fill order is X->Y->Z.
        """

        if shape!=None:
            assert len(shape)==3
        else:
            shape = (1,10,10)
##            kwargs[self.P_XSIZE] = shape[0]
##            kwargs[self.P_YSIZE] = shape[1]
##            kwargs[self.P_ZSIZE] = shape[2]

##        if mapping!=None:
##            kwargs[self.P_MAPPING] = mapping

##        if order!=1:
##            kwargs[self.P_ORDER] = order

##        if depth!=1:
##            kwargs[self.P_DEPTH] = depth

##        initClass(self, kwargs)
##        shape = (getattr(self,self.P_XSIZE),getattr(self,self.P_YSIZE),getattr(self,self.P_ZSIZE))
##        order = getattr(self,self.P_ORDER)
##        mapping = getattr(self,self.P_MAPPING)

        assert len(shape)==3
        assert (order==1 or order==2)

        self.shape = tuple(shape)
        #print 'self.shape 1:', self.shape
        self.depth = depth


        if order==1 :
            self.neighbourMask = self.order1Mask
            self.nbNeighboursMax = self.nbNeighboursOrder1
        else:
            self.neighbourMask = self.order2Mask
            self.nbNeighboursMax = self.nbNeighboursOrder2

        if mapping == None :
            #print 'no mapping ... create full mapping from shape ', shape
            # If no mapping is defined then the lattice is "full" :
            self.nbVoxels = shape[0]*shape[1]*shape[2]
            # Default fill order of the lattice is : X->Y->Z :
            lxly = shape[0]*shape[1]
            lx = shape[0]
            ly = shape[1]
            self.mapping = NP.array([ [(i%(lxly))/ly, (i%(lxly))%ly, i/(lxly)]          #creates a mapping by affecting a voxel position to each number corresponding to a voxel
                                      for i in xrange(self.nbVoxels) ])
        else:
            # Else valid positions are defined with 'mask' :
            self.mapping = mapping
            self.nbVoxels = len(mapping)

        mask = NP.array(self.mapping)
        #print "self.shape"
        #print self.shape
        #print 'mask:'
        #print mask
        # Center working origin on the "upper left corner" :
        self.correctedMask = mask - mask.min(0)
        #TODO : assert that corrected mask fits in volume with given shape

        self.ndarrayMask = ( NP.array([c[0] for c in self.mapping]),
                             NP.array([c[1] for c in self.mapping]),
                             NP.array([c[2] for c in self.mapping]) )
        self.ndarrayCorMask = ( NP.array([c[0] for c in self.correctedMask]),
                                NP.array([c[1] for c in self.correctedMask]),
                                NP.array([c[2] for c in self.correctedMask]) )

        # Build the lattice, positions with '-1' are not valid :
        self.lattice = NP.zeros((self.shape), dtype=int)-1

        # Fill lattice with integer indexes corresponding to valid positions :
        for iv in xrange(self.nbVoxels):
            x = self.correctedMask[iv,0]
            y = self.correctedMask[iv,1]
            z = self.correctedMask[iv,2]
            # Fill lattice with index corresponding to a valid position :
            self.lattice[x,y,z] = iv

##        print 'self.lattice:'
##        print self.lattice
        self.neighboursCoordLists = None
        self.neighboursIndexLists = None
        self.buildNeighboursIndexLists()
        self.buildNeighboursCoordLists()

    def getNbVoxels(self):
        return self.nbVoxels

    def getTargetAxesNames(self):
        return ['axial','coronal', 'sagittal'] #TODO make less static

    def getMapping(self):
        return self.mapping

    def getRoiMask(self):
        """
        Return a binary or n-ary 3D mask which has the shape of the target data
        """
        return NP.ones(self.shape, dtype=int) # every position marked as 1

    def getNdArrayMask(self):
        return self.ndarrayMask

    def getCoord(self, index):
        return self.mapping[index]

    def getIndex(self, coord):
        return self.lattice[coord]

    def getNeighboursCoords(self, idvoxel):
        if self.neighboursCoordLists != None :
            return self.neighboursCoordLists[idvoxel]
        else:
            return [self.mapping[i]
                    for i in self.getNeighboursIndexes(idvoxel)]

    def getClosestNeighboursIndexes(self, idVoxel):
        cv = self.correctedMask[idVoxel]
        neighbours = []
        # translate center of neighbours mask to voxel position :
        neighboursTmp = self.neighbourMask + cv
        for cn in neighboursTmp:
            if NP.bitwise_and(cn>=0, cn<self.shape).all():
                neighbours.append(cn.tolist())

        return [self.lattice[n[0],n[1],n[2]] for n in neighbours
                if self.lattice[n[0],n[1],n[2]]!=-1]


    def getNeighboursIndexes(self, idVoxel):
        if self.neighboursIndexLists != None :
            return self.neighboursIndexLists[idVoxel]
        else:
            neighbours = set([idVoxel])
            for d in xrange(self.depth):
                newNeighbours = []
                for i in neighbours:
                    newNeighbours.extend(self.getClosestNeighboursIndexes(i))
                neighbours.update(newNeighbours)
            neighbours.remove(idVoxel)
            return list(neighbours)

    def buildNeighboursIndexLists(self):
        neighboursListsTmp = range(self.nbVoxels)
        for i in xrange(self.nbVoxels):
            neighboursListsTmp[i] = self.getNeighboursIndexes(i)
        
        self.neighboursIndexLists = neighboursListsTmp
        
    def buildNeighboursCoordLists(self):
        neighboursListsTmp = range(self.nbVoxels)
        for i in xrange(self.nbVoxels):
            neighboursListsTmp[i] = self.getNeighboursCoords(i)
        self.neighboursCoordLists = neighboursListsTmp

    def getNeighboursIndexLists(self):
        return self.neighboursIndexLists

    def getNeighboursCoordLists(self):
        return self.neighboursCoordLists

    def getNbCliques(self):
        if not hasattr(self,'nbCliques') or self.nbCliques is None:
            self.nbCliques = sum( [len(x) for x in self.neighboursIndexLists])/2.
        return self.nbCliques

    def mapVoxData(self, data, fillValue=0):
        ##print 'self.mapping :', self.mapping
        ##print 'self.correctedMask :', self.correctedMask
        #print 'self.ndarrayMask:',self.ndarrayMask[0].shape
        #print 'self.shape:', self.shape
        if len(data.shape) == 1:
            mappedData = NP.zeros(self.shape, dtype=data.dtype) + fillValue
            mappedData[self.ndarrayMask] = data
        elif len(data.shape) == 2: # assume (time, nbVox)
            sh = tuple([data.shape[0]]) + self.shape
            mappedData = NP.zeros(sh, dtype=data.dtype) + fillValue
            m = self.ndarrayCorMask
            mappedData[:, m[0], m[1], m[2]] = data
        #print 'mappedData.shape : ',mappedData.shape
        #print 'mappedData[0,0,:]' , len(mappedData[0,0,:])

        return mappedData

    @staticmethod
    def createFromGUI(GUIobject):
        """
        Creates the actual object based on the parameters
        """
        params = {}
        params[RegularLatticeMapping.P_XSIZE] = getattr(GUIobject,RegularLatticeMapping.P_XSIZE)
        params[RegularLatticeMapping.P_YSIZE] = getattr(GUIobject,RegularLatticeMapping.P_YSIZE)
        params[RegularLatticeMapping.P_ZSIZE] = getattr(GUIobject,RegularLatticeMapping.P_ZSIZE)
        params[RegularLatticeMapping.P_ORDER] = getattr(GUIobject,RegularLatticeMapping.P_ORDER)
        params[RegularLatticeMapping.P_DEPTH] = getattr(GUIobject,RegularLatticeMapping.P_DEPTH)
        return RegularLatticeMapping(**params)


##################
# Spatial Fields #
##################


class StateField:
    """
    Class handling a field of states : a set of integers (ie labels) whose
    ranks can be spatially mapped to 3D coordinates. Each label refers to a
    class wich is identified by an ID and a name.
    """

    def __init__(self, classNames, spConf, initProps=None):
        """
        Initialise a StateField object.
        """
        self.classNames = classNames
        self.classIds = {} # will map each class name to an integer class ID.
        ic = 0
        for cn in classNames :
            self.classIds[cn] = ic
            ic += 1

        self.nbClasses = len(self.classNames)
        self.spatialConfig = spConf
        self.initProportions = initProps
        self.size = spConf.getNbVoxels()
        #print 'statefield size:', self.size
        self.data = NP.zeros(self.size, dtype=int) # will store label values
        self.generate() # generate label values
        self.classCounts = range(self.nbClasses)
        self.updateClassCounts()

    def getSize(self):
        """
        Return to size of the field.
        """
        return self.size

    def generate(self):
        """
        Generate values for every states. By default : if initProportions is set,
        generate values according to it. State values will be ordered by class ID
        """
        if self.initProportions != None :
            self.data = NP.array([], dtype=int)
            ## Init labels according to init proportions :
            for ic in xrange(self.nbClasses-1):
                cs = NP.round(self.size*self.initProportions[ic])
                self.data = NP.hstack((self.data, NP.zeros(cs)+ic))
            # Calculate last class size from the sum of the others :
            cs = self.size - len(self.data)
            self.data = NP.hstack((self.data, NP.zeros(cs)+self.nbClasses-1))
            # Update last proportions accordingly :
            ps = cs/(self.size+0.)
            if len(self.initProportions) == self.nbClasses:
                self.initProportions[-1] = ps
            elif len(self.initProportions) == self.nbClasses-1:
                self.initProportions.append(ps)
            else:
                #Error !
                pass
        # if self.initProportions != None :

    def getNbClasses(self):
        return self.nbClasses

    def updateClassCounts(self):
        """
        Compute the size of every classes.
        """
        for ic in xrange(self.nbClasses):
            self.classCounts[ic] = (self.data == ic).sum()

    def randomize(self):
        """
        Randomize state values with a ramdom permutation.
        """
        self.data = NP.random.permutation(self.data)

    def getFieldValues(self):
        """
        Return all field values.
        """
        return self.data

    def getMappedFieldValues(self):
        return self.spatialConfig.mapVoxData(self.data)

    def setFieldValues0(self, values, mask=None):
        """
        Copy the content of 'values' to state values masked by 'mask'.
        """
        #print 'setFieldValues ...'
        if mask == None :
            #print 'mask is None'
            #print len(values), len(self.data)
            assert len(values) == len(self.data)
            self.data[:] = values[:]
        else:
            #print 'mask present'
            #assert len(values) == sum(self.data[mask])
            self.data[mask] = values[:]
        #print 'self.data :', self.data
    
    def setFieldValues(self, values, mask=None):
        """
        Copy the content of 'values' to state values masked by 'mask'.
        """
        #print 'setFieldValues ...'
        if mask == None :
            #print 'mask is None'
            #print len(values), len(self.data)
            assert len(values) == len(self.data)
            self.data[:] = values[:]
        else:
            #print 'mask present'
            #print len(values), len(self.data[mask[:,:]]), mask.shape
            #assert len(values) == len(self.data[mask])
            self.data[mask] = values[mask]
        #print 'self.data :', self.data
    def getClassName(self, classId):
        """
        Return the class name corresponding to the integer 'classId'
        """
        assert type(classId)==int
        if classId < self.nbClasses and classId >= 0:
            return self.classNames[classId]
        else: return None

    def getClassNames(self):
        """
        Return all the class names
        """
        return self.classNames

    def getClassId(self, className):
        """
        Return the class id corresponding to the string 'className'
        """
        assert type(className)==str
        return self.classIds[className]


class PottsField(StateField):
    pass
    

################ OLD #################

##class NoVoxelMappingConfiguration(SpatialVoxelConfiguration):
##    """

##    @author: Thomas VINCENT
##    """

##    def __init__(self, nbVoxels=0):
##        """
##        Construct the voxel mapping, with the given number of voxels

##        @param nbVoxels : Total number of voxels
##        @since
##        @author: Thomas VINCENT
##        """
##        self.nbVoxels = nbVoxels
##        self.voxelsIndexes = NP.arange(nbVoxels, dtype=int)

##    def getNbVoxels(self):
##        """

##        @return: the number of voxels (int)
##        @since
##        @author
##        """
##        return self.nbVoxels

##    def getNeighbours(self, idVoxel):
##        """


##        @param idVoxel :
##        @return:
##        @since
##        @author: Thomas VINCENT
##        """
##        return NP.array([], dtype=int)


##    def getNeighboursLists(self):
##        return [ [] for i in xrange(self.nbVoxels) ] # empty lists

##    def getVoxelIndexList(self):
##        """


##        @return:
##        @since
##        @author: Thomas VINCENT
##        """
##        return self.voxelsIndexes

def maskToMapping(m):
    return NP.array(NP.where(m!=0)).transpose()

def mask_to_coords(m):
    #print 'm:', m
    return NP.array(NP.where(m!=0)).transpose()
