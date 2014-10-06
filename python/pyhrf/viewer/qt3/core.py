# -*- coding: utf-8 -*-
import numpy as _N
from tools import *
import copy as copyModule
from os.path import splitext
import string

"""
This module provides classes and functions to handle multi-dimensionnal numpy
array (ndimarray) objects and extend them with axes labels and axes real domains.
Available features :
    - a multidimensional view system by means of projections and slice handling.
      See xndarrayView class.
    - several views of the same ndimarray can be synchronised over their common
      axes (ie making indexes be the same) via the xndarrayViewWatcher class.
"""


########
# Core #
########

debug = 0

#TODO: maybe make it recursive to manage different flatten level?
def flattenElements(l):
    dest = []
    for le in l:
        dle = []
        for e in le:
            if _N.iterable(e):
                dle.extend(e)
            else:
                dle.append(e)
        dest.append(dle)
    return dest

def buildMappingMask(destxndarray, srcData, srcAxesNames, axisName,
                     mappedAxesNames, mapping):
    destAxes = destxndarray.getAxesNames()
    for ma in mappedAxesNames:
        assert ma in destAxes    
    ia = srcAxesNames.index(axisName)
    nd = len(mappedAxesNames)
    if debug: print 'srcData.shape :', srcData.shape
    targetRanges = [range(d) for d in srcData.shape]
    targetRanges = targetRanges[:ia] + [mapping] + targetRanges[ia+nd:]
    if debug: print 'targetRanges :', targetRanges
    if debug: print 'launching cartesian ...'
    targetCoords = cartesian(*targetRanges)
    targetCoords = flattenElements(targetCoords)
    if debug: print 'flatten targetCoords: ', list(targetCoords)
    if debug: print '_N.array(targetCoords) :', _N.array(targetCoords)
    return [_N.array(a) for a in _N.array(targetCoords).transpose()]


def duplicateArray(a, shape):
    if len(shape) == 0:
        return a
    newSh = a.shape + shape[0:1]
    newA = _N.repeat(a, shape[0]).reshape(newSh)
    if len(shape) > 1:
        return duplicateArray(newA, shape[1:])
    else:
        return newA

def moldxndarrays(cuboids, names=None):

    if names is not None:
        assert len(names) == len(cuboids)
    # assert that there is at least one common axes:
    axes = list(cuboids[0].getAxesNames())
    axesToxndarray = dict(zip(axes,[[cuboids[0]]]*len(axes)))
    intersectionAxes = set(axes)
    destShape = cuboids[0].getShape()
    for c in cuboids[1:]:
        ans = set(c.getAxesNames())
        intersectionAxes.intersection_update(ans)
        newAxes = ans.difference(axes)
        axes += list(newAxes)
        for an in ans:
            if axesToxndarray.has_key(an):
                axesToxndarray[an].append(c)
            else:
                axesToxndarray[an] = [c]
        
        for newAxis in newAxes:
            destShape += (c.getAxisDim(newAxis),)
            
    assert len(intersectionAxes) > 0
    # assert that common axes have the same domain, 
    # and resolve value label:
    domains = {}

    for ax,cubs in axesToxndarray.iteritems():
        if len(cubs) > 1:
            refDomain = cubs[0].getAxisDomain(ax)
            refValueLabel = cubs[0].getValueLabel()
            for c in cubs[1:]:
                if (c.getAxisDomain(ax) is not None):
                    cd = c.getAxisDomain(ax) #.astype(refDomain.dtype)
                    if cd.shape != refDomain.shape or (cd!=refDomain).any():
                        sexp = 'Domain inconsistency for axis %s \n' %ax
                        if names is not None:
                            sexp += 'reference cuboid: %s\n' %names[0]
                        else:
                            sexp += 'reference cuboid: \n' 
                        sexp += cubs[0].descrip()
                        sexp += c.descrip()
                        raise Exception(sexp)
                if refValueLabel != c.getValueLabel():
                    refValueLabel = 'value'
            domains[ax] = refDomain

    # create expanded cuboids:
    moldedxndarrays = []
    for cub in cuboids:
        moldedxndarrays.append(cub.expandTo(list(axes), destShape))
    return moldedxndarrays

def joinxndarrays(cuboids, names=None, stackAxis=None):
    if debug: print 'joinxndarrays ...'
    if debug: print 'names:',names
    moldedxndarrays = moldxndarrays(cuboids, names)
    if stackAxis is None:
        i = 0
        while 'stackAxis%d'%i in moldedxndarrays.getAxesNames():
            i += 1
        stackAxis = 'stackAxis%d'%i
    return stackxndarrays(moldedxndarrays, stackAxis, names)

def moldViews(views):
    """
    Create new xndarrayView objects from 'views' which share exactly the same axes,
    domains and mask.
    """
    view0 = views[0]
    moldedxndarrays = moldxndarrays([v.cuboid for v in views])
    #cropState = ([v.cropOn() for v in views]).all()
    mask, maskName = None, None
    for v in views:
        if v.isMasked():
            mask,maskName = v.getMask(),v.getMaskName()
    moldedViews = []
    for c in moldedxndarrays:
        v = xndarrayView(c)
        v.setView(view0.mode, view0.getCurrentAxesNames())
        if mask != None:
            v.applyMask(mask, maskName)
        moldedViews.append(v)
        
    return moldedViews
    

def stackxndarrays(cuboidList, axisName, axisValues=None):
    size = len(cuboidList)
    cub0 = cuboidList[0]
    sh = (size,) + cub0.data.shape
    stackedData = _N.zeros(sh, cub0.data.dtype)
    if cub0.errors is not None:
        stackedErrors = _N.zeros((size,)+cub0.errors.shape, cub0.errors.dtype)
    else:
        stackedErrors = None
    newDomains = cub0.getAxesDomainsDict()
    if axisValues is not None:
        newDomains[axisName] = axisValues
    targetCub = xndarray(stackedData, errors=stackedErrors,
                       axesNames=[axisName]+cub0.axesNames,
                       axesDomains=newDomains, valueLabel=cub0.valueLabel)
    for i, cuboid in enumerate(cuboidList):
        if debug: print 'targetCub.data[i] :', targetCub.data[i].shape
        if debug: print 'cuboid', cuboid.descrip()
        
        targetCub.data[i] = cuboid.data
        if targetCub.errors is not None and cuboid.errors is not None:
            targetCub.errors[i] = cuboid.errors
        
    return targetCub

class xndarray:
    """
    Handles a multidimensional numpy array of which axes are labeled and mapped
    to their respective real domains.
    Ex :
    a = numpy.array([ [4,5,6],[8,10,12] ])
    c = xndarray( a, ['time','position'], {'time':[0.1,0.2]} )
    # Will represent the following situation:

    position
    ------->
    4  5  6 | t=0.1   |time
    8 10 12 | t=0.2   v
    
    """

    def __init__(self, narray, errors=None, axesNames=None, axesDomains=None,
                 valueLabel="value"):
        """
        Initialize a new xndarray object from 'narray'. With 'axesNames' as axes
        labels and 'axesDomains' as domains mapped to integer slices. 
        By default :
        axesNames = [\"dim0\", \"dim1\", ...]
        axesDomains = slice ranges -> dict(axisId:axisRange).
        """
        if debug : print 'xndarray.__init__ ...'
        if errors!=None:
            assert narray.shape == errors.shape
            self.errors = errors
        else:
            self.errors = None
        self.data = narray
        self.valueLabel = valueLabel
        nbDims = len(self.data.shape)
        
        if axesNames is None:
            self.axesNames = ['axis'+str(i) for i in xrange(nbDims)]
        else:
            assert type(axesNames) == list
            if len(axesNames) != nbDims:
                raise Exception("length of axesNames (%d) different from nb of"\
                                    " dimensions (%d). \n Got names: %s \n" \
                                    " xndarray shape: %s" \
                                    %(len(axesNames), nbDims, str(axesNames),
                                      str(self.data.shape)))
            
            self.axesNames = axesNames
        
        self.axesIds = dict( [(self.axesNames[i],i) for i in xrange(nbDims)] )

        # By default: slices of axis <=> domain of axis
        self.axesDomains = [_N.arange(self.data.shape[d])
                            for d in xrange(nbDims)]
        if axesDomains is not None:
            assert type(axesDomains) == dict 
            for ia,dv in axesDomains.items():
                self.setAxisDomain(ia,dv)
    #__init__

    def descrip(self):
        """
        Return a printable string describing the current object.
        """
        s = ''
        s += '* shape : %s\n' %str(self.data.shape)
        s += '* axesNames : %s \n' %str(self.axesNames)
        s += '* valueLabel : %s\n' %self.valueLabel
        s += '* axesDomains: \n'
        s += str(self.axesDomains) + '\n'

        return s

    def has_axis(self, name):
        #print 'name',name
        #print self.axesNames
        #print '->', name in self.axesNames
        #print ''
        return name in self.axesNames

    def getMinValue(self):
        return self.data.min()

    def getMaxValue(self):
        return self.data.max()
    
    def getValueLabel(self):
        """
        Return the label associated to values in the cuboid. 
        """
        return self.valueLabel

    def getNbDims(self):
        return len(self.data.shape)
    #getNbDims

    def getAxisDim(self, axisId):
        axisId = self.getAxisId(axisId)
        return self.data.shape[axisId] if axisId is not None else None
    #getAxisDim

    def getShape(self):
        return self.data.shape
    #getShape
    
    def getAxisDomain(self, axisId):
        """
        Return the real domain of the given axis. 'axisId' can be a string or
        int defining te axis.
        """
        axisId = self.getAxisId(axisId)
        return self.axesDomains[axisId] if axisId is not None else None
    #getAxisDomain

    def setAxisDomain(self, axisId, domain):
        
        axisId = self.getAxisId(axisId)
        if axisId !=None :
            if debug: print 'setting domain of axis %s with %s' \
                    %(str(axisId),str(domain)) 
            if len(domain) != self.data.shape[axisId]:
                raise Exception('length of domain values (%d) does not ' \
                                ' match length of data (%d) for axis %s' \
                                %(len(domain),self.data.shape[axisId],
                                  self.getAxisName(axisId)))
            self.axesDomains[axisId] = _N.array(domain)
    #setAxisDomain
    
    def getAxesDomains(self):
        """
        Return the real domain of the given axis. 'axisId' can be a string or
        int defining te axis.
        """
        return [ self.getAxisDomain(i) for i in xrange(self.getNbDims()) ]
    #getAxisDomain

    def getAxesDomainsDict(self):
        return dict(zip(self.axesNames, self.axesDomains))

    def getAxisName(self, axisId):
        """
        Return the name of an axis from the given id.
        """
        if debug : print 'type(axisId) :', type(axisId), axisId
        assert _N.isreal(axisId) and _N.round(axisId) == axisId
        if debug : print 'core.cuboid ... getting name of ', axisId
        if debug : print 'from :', self.axesNames
        if axisId>=0 and axisId<self.getNbDims():
            return self.axesNames[axisId]
        else: return None
    #getAxisName

    def getAxesNames(self):
        """
        Return names of all axes.
        """
        return self.axesNames
    #getAxesNames

    def getAxisId(self, axisName):
        """
        Return the id of an axis from the given name.
        """
        if debug : print 'core.cuboid ... getting id of ', axisName
        if debug : print 'from :', self.axesIds
        if type(axisName) == str : # axisName is a string axis name
            if axisName in self.axesIds.keys():
                return self.axesIds[axisName]
            else: return None
        else: # axisName is an integer axis index 
            if axisName>=0 and axisName<self.getNbDims():
                return axisName
            else: return None
    #getAxisId

    def expandTo(self, destAxes, destShape, domains=None):
        """
        Repeat the cuboid accoding to destShape and return a new cuboid whose
        axes are self.axesNames + destAxes
        """
        axesToAdd = []
        shapeToAdd = tuple()
        for iax, dax in enumerate(destAxes):
            if dax not in self.getAxesNames():
                axesToAdd.append(dax)
                shapeToAdd += (destShape[iax],)
        if debug:
            print 'self.data :', self.data.shape
            print 'self.axesNames :', self.getAxesNames()
            print ' -> duplicate to:', destAxes, destShape
            print ' axesToAdd :', axesToAdd
            print ' shapeToAdd :', shapeToAdd
        newData = duplicateArray(self.data, shapeToAdd)
        if debug: print 'newData :', newData.shape
        newErrors = None
        if self.errors != None:
            newErrors = duplicateArray(self.errors, shapeToAdd)

        temp = self.getAxesNames() + axesToAdd
        ia = 0
        while temp != destAxes:
            if debug: print ' - temp axes :', temp
            if debug: print ' - target axes :', destAxes
            if debug: print ' temp[%d]: %s ,destAxes[%d]: %s' \
               %(ia, temp[ia], ia, destAxes[ia])
            if temp[ia] != destAxes[ia]:
                a1 = temp[ia]
                a2 = destAxes[ia]
                ia2 = temp.index(a2)
                temp[ia] = a2
                temp[ia2] = a1
                if debug: print ' swap %s<->%s => %d<->%d' %(a1, a2, ia, ia2)
                newData = newData.swapaxes(ia,ia2)
                if newErrors != None:
                    newErrors = newErrors.swapaxes(ia,ia2)
            ia += 1

        newDomains = self.getAxesDomainsDict()
        if domains != None:
            newDomains.update(domains)
            
        return xndarray(newData,newErrors,destAxes,newDomains,self.getValueLabel())

    def set_orientation(self, orientation):
        if orientation == self.axesNames:
            return
        newAd = range(self.getNbDims())
        #print 'be4 roll - self.data:', self.data.shape
        #print self.axesNames
        #print 'target orientation:', orientation
        for di,an in enumerate(orientation):
            i = self.axesNames.index(an)
            self.data = _N.rollaxis(self.data, i, di)
            self.axesDomains.insert(di,self.axesDomains.pop(i))
            self.axesNames.insert(di,self.axesNames.pop(i))
        #print 'after roll - self.data:', self.data.shape
        
    def is_nii_writable(self):
        from pyhrf.tools._io import MRI3Daxes
        return set(MRI3Daxes).issubset(self.axesNames)

    def save(self, filename):
        """ Save to data file(s). The type of data file is dertermined by the
        extension of 'filename'. Supported types: nii, csv, h5. 
        For the nii format, if the dimension set does not correspond to an MRI
        3D volume (axial, coronal, sagittal) or to an fMRI 4D volume (axial,
        coronal, sagittal, time) then multiple data files will be created along with
        an xml file holding the joint information.
        """
        base, ext = splitext(filename)
        if ext == '.nii':
            from pyhrf.tools._io import writexndarrayToNiftii
            from pyhrf.tools._io import MRI3Daxes
            return writexndarrayToNiftii(self, filename)            
        elif ext == '.csv':
            #print 'dumping an array of shape %s to csv' %(str(self.data.shape))
            #print '-> file:', filename
            _N.savetxt(filename, self.data, fmt="%12.9G")
            return [filename]
        elif ext == '.h5':
            import tables
            h5file = tables.openFile(filename, mode='w', title='array')
            root = h5file.root
            #print 'dumping an array of shape %s to h5' %(str(self.data.shape))
            #print '-> file:', filename
            h5file.createArray(root, "array", self.data)
            h5file.close()
            return [filename]
        else:
            raise Exception('Unsupported file type - extension: ' + ext)

    def load(filename):
        pass

#xndarray class


class xndarrayView:
    """
    Class managing projections and axis views on a xndarray object.
    """
    #TODO : maybe replace mode stuff with just specifying the number or
    #       dimensions in current view ...
    #      OR maybe it is implicit with the size of currentAxes ...
    MODE_1D = 0
    MODE_2D = 1
    MODE_3D = 2
    MODE_4D = 3
    
    def __init__(self, cuboid,  mode=None, currentAxes=None, crop=False):
        """
        Initialize a new xndarrayView on 'cuboid'. A mask can be further specified
        by calling method 'applyMask'. 'mode' indicates the slicing
        mode used in the 'getView' method and 'currentAxes' the list of axes
        ids (by string name) involved in this projection.
        'crop' is a boolean indicating wether to discard masked positions or not
        while getting the current view (with method 'getView').
        Examples:
            from numpy import *
            a = numpy.random.randn(10,10)
            c = xndarray(a)
            v = xndarrayView(c)
        """
        if debug : print 'Init of xndarrayView from cuboid :'
        if debug : print cuboid.descrip()
        self.cuboid = cuboid
        self.cropFlag = crop
        
        self.outMask = None # will be used as display mask in data returned
                            # by getView
        self.maskLabels = None
        self.maskLabel = 0
        self.maskName = None
        
        self.data = cuboid.data # shortcut to the viewed data

        if cuboid.errors != None:
            self.errors = cuboid.errors # idem
        else:
            self.errors = None
            
        if mode == None: # default mode
            if self.cuboid.getNbDims()>1:
                mode = self.MODE_2D
            else:
                mode = self.MODE_1D
        else: mode = mode
        if currentAxes == None: # default current axes : first axes by index
            if len(self.data.shape) > 1:
                currentAxes = [self.cuboid.getAxisName(0),
                               self.cuboid.getAxisName(1)]
            else:
                currentAxes = [self.cuboid.getAxisName(0)]
        self.currentMaskAxes = None
        if debug: print 'cuboidview.__init__: call setview ...'
        if debug: print ' -> ca = ', currentAxes

        self.slices = [0]*self.cuboid.getNbDims()
        self.setView(mode, currentAxes)
    #__init__

    def copy(self):
        niewView = xndarrayView(self.cuboid, self.mode, self.currentAxes,
                              self.cropFlag)
        if self.isMasked():
            niewView.applyMask(self.outMask, self.maskName)
        return niewView

    def descrip(self):
        s = 'Current View:'
        s += '* viewed cuboid:'
        s += self.cuboid.descrip()
        s += '* current slices:'
        s += str(self.slices)
        s += '*current axes:'
        s += str(self.getCurrentAxesNames())
        return s

    def getValueLabel(self):
        """
        Return the label associated to values in the viewed cuboid.
        """
        return self.cuboid.getValueLabel()
    #def getValueLabel

    def getNbDims(self):
        """
        Return the number of dimensions of the viewed cuboid
        """
        return self.cuboid.getNbDims()
    
    def getMinValue(self):
        """
        Return the minimum value of data in the viewed cuboid
        """
        return self.cuboid.getMinValue()
    #getMinValue

    def getMaxValue(self):
        """
        Return the minimum value of data in the viewed cuboid
        """
        return self.cuboid.getMaxValue()
    #getMaxValue

    
    def isMasked(self):
        """
        Return True is current view has a mask 
        """
        return self.outMask != None
    #isMasked
    
    def getMaskName(self):
        return self.maskName
    #getMaskName

    def cropOn(self):
        """
        Return True if current is cropped to mask values different from 0.
        """
        return self.cropFlag
    #cropOn

    def setCrop(self, flag):
        """
        Choose if the view is cropped or not
        """
        self.cropFlag = flag
        if flag:
            self.cropAxesRanges()
        else:
            self.buildAxesRanges()
        self.buildMask()
    #setCrop
    
    def getMaskLabels(self):
        """
        Return all unique values in current mask
        """
        return self.maskLabels
    #getMaskLabels
    
    def applyMask(self, mask, maskName='mask'):
        """
        Apply the given mask which should be a xndarray instance filled with
        integer values (labels). Label with value=0 stands for the background
        (masked values).
        """
        if debug : print 'core.cuboidView.applyMask: checking axes names ... '
        # Check if axis names in mask are found in axes of current view :
        #if set(self.getAxesNames()) == set(mask.getAxesNames()):
        if all([ax in self.getAxesNames() for ax in mask.getAxesNames()]):
            # Check if shape in mask is consistent with data shape
            # in current view :
            for a in mask.getAxesNames():
                assert mask.getAxisDim(a) == self.cuboid.getAxisDim(a)
            self.currentMaskAxes = []
            for ca in self.getCurrentAxesNames():
                icam = mask.getAxisId(ca)
                if icam is not None:
                    self.currentMaskAxes.append(icam)
            if debug: print 'self.currentMaskAxes:',self.currentMaskAxes
            self.outMask = mask
            self.maskLabels = _N.unique(_N.concatenate(([0],mask.data.ravel())))
            self.maskName = maskName
            self.buildAxesRanges()
            self.cropAxesRanges()
            self.buildMask()
        else:
            raise Exception("axis names in mask do not match axes in view")
    #def applyMask

    def getMask(self):
        """
        Return the mask applied to the current view (a xndarray instance).
        """
        return self.outMask
    #def getMask
    
    def setView(self, mode, curAxes):
        """
        Set the current view to either be 1D, 2D or 3D with 'mode'.
        Specify wich axes are involved in current view with 'curAxes' (which must
        contain string designations).
        Note : every slice is set to 0.
        """
        if debug : print 'core.cuboidView ... setting mode :', mode

        self.mode = mode
        
        if not ( type(curAxes)==list or type(curAxes)==tuple ) :
            # Then it is a single element:
            curAxes = [curAxes]

        if self.mode == self.MODE_2D:
            assert len(curAxes) == 2
        elif self.mode == self.MODE_1D:
            assert len(curAxes) == 1
        elif self.mode == self.MODE_3D:
            assert len(curAxes) == 3
        elif self.mode == self.MODE_4D:
            assert len(curAxes) == 4
        else:
            raise Exception('Mode "%s" not understood' %str(mode))
        
        self.currentAxes = [ self.cuboid.getAxisId(ca) for ca in curAxes ]
        while None in self.currentAxes:
            # Discard unknown axes:
            self.currentAxes.remove(None)
        if self.isMasked():
            self.currentMaskAxes = [ self.outMask.getAxisId(ca)
                                     for ca in curAxes ]
            while None in self.currentMaskAxes:
                # Discard unknown axes:
                self.currentMaskAxes.remove(None)
        
        if debug : print 'core.cuboidView.setView: new currentAxes: '
        if debug : print self.currentAxes
        if self.isMasked():
            if debug : print 'core.cuboidView.setView: new currentMaskAxes: '
            if debug : print self.currentMaskAxes
        self.buildAxesRanges()
        self.cropAxesRanges()
        self.buildMask()
    #setView

    def cropAxesRanges(self):
        if debug: print 'cropAxesRanges ...'
        if self.isMasked() and self.cropFlag:
            if debug: print '**** Treating cropped view ******'
            if debug: print 'self.maskLabel :', self.maskLabel
            if self.maskLabel != 0:
                # Select only positions which match mask label
                om = _N.where(self.outMask.data == self.maskLabel)
            else:
                # Discard background
                om = _N.where(self.outMask.data != 0)
            #if debug: print 'om :', len(om), om
            for can in self.getCurrentAxesNames():
                icam = self.outMask.getAxisId(can)
                if icam != None and (icam in self.currentMaskAxes):
                    ica = self.cuboid.getAxisId(can)
                    omcar = om[icam]
                    r = range(omcar.min(), omcar.max()+1)
                    #Warn : axesRanges and maskAxesRanges share
                    # the same reference...
                    self.axesRanges[ica] = r
                    self.maskAxesRanges[icam] = r
                    
            if debug: self.printAxesRanges()

    def cropView(self):
        if debug: print "crop View called ..."
        if self.isMasked() and self.cropFlag:
            self.cropAxesRanges()
            self.buildMask()
        
    def buildAxesRanges(self):
        """
        build ranges for current axes and set ranges of every other to 0
        (default slice position).
        """
        shape = self.data.shape
        self.axesRanges = [[sl] for sl in self.slices]

        for ca in self.currentAxes:
            self.axesRanges[ca] = range(shape[ca])

        if self.isMasked():
            mshape = self.outMask.data.shape
            self.maskAxesRanges = []
            for man in self.outMask.getAxesNames():
                ia = self.cuboid.getAxisId(man)
                self.maskAxesRanges.append([self.slices[ia]])
            for ca in self.currentMaskAxes:
                self.maskAxesRanges[ca] = range(mshape[ca])
                
        if debug:  self.printAxesRanges()
    #buildAxesRanges
    
    def buildMask(self):
        """
        Build the current slicing mask to be applied on a numpy array.
        """
        # Build numpy compatible mask :
        axr = self.axesRanges
        #print 'axr :', axr
        if debug: print 'core.cuboidView.buildMask: ...'
        if debug: print 'with arg:', axr
        if not self.cropFlag and \
               len(self.currentAxes) == self.getNbDims():
            self.mask = _N.where(_N.ones(self.data.shape,dtype=int))
        else:
            lc = list(cartesian(*axr))
            self.mask = _N.array(lc, dtype=int).transpose().tolist()
        if debug:
            print 'self.mask :'
            for i in xrange(len(self.mask)):
                print i, ':', min(self.mask[i]), max(self.mask[i])
        if self.isMasked():
            axrm = self.maskAxesRanges
            if debug: print '*****self.maskAxesRanges:',self.maskAxesRanges
            lcm = list(cartesian(*axrm))            
            self.maskOfOutMask = _N.array(lcm, dtype=int).transpose().tolist()
        #if debug: print 'core.cuboidView.buildMask: mask:'
        #if debug: print self.mask
    #buildMask

    def isCurrentAxis(self, id):
        return self.cuboid.getAxisId(id) in self.currentAxes

    def getCurrentAxes(self):
        """
        Return current axes (as integer ids) involved in the current view
        """
        return self.currentAxes
    #getCurrentAxes

    def getCurrentAxesNames(self):
        """
        Return current axes (as string ids) involved in the current view
        """
        return [self.cuboid.getAxisName(a) for a in self.currentAxes]
    #getCurrentAxesNames
        
    def getCurrentMaskAxesNames(self):
        """
        Return current axes (as string ids) involved in the current view
        """
        if not self.isMasked(): return None
        return [self.outMask.getAxisName(a) for a in self.currentMaskAxes]
    #getCurrentMaskAxesNames


    def getAxisDomain(self, axis):
        """
        Wrapper for xndarray.getAxisDomain method
        """
        return self.cuboid.getAxisDomain(axis)

    def setMaskLabel(self, label):
        if debug: print 'setMaskLabel - to:', label
        self.maskLabel = label
        self.cropView()
            
    def computeOutMaskView(self, viewVals, viewAxesNames):
        if not self.isMasked():
            return None

        canm = self.getCurrentMaskAxesNames()
        if debug:
            print 'self.maskOfOutMask :', len(self.maskOfOutMask), \
                  len(self.maskOfOutMask[0])
            for m in self.maskOfOutMask:
                print '-', len(m)
            print 'self.outMask.data :', self.outMask.data.shape
        omv = self.outMask.data[self.maskOfOutMask]
        if debug:
            print 'viewAxesNames :', viewAxesNames
            print 'currentMaskAxesNames : ', canm
        if omv.shape != viewVals.shape:
            nExtent = 1
            for van in viewAxesNames:
                if van not in canm:
                    nExtent *= viewVals.shape[viewAxesNames.index(van)]
            if debug:
                print 'nExtent :', nExtent
                print 'viewVals.shape :', viewVals.shape
                print 'omv.shape :', omv.shape
            omv = omv.repeat(nExtent).reshape(viewVals.shape)
                
        return omv

    def getCurrentShape(self):
        return tuple([len(self.axesRanges[ia]) for ia in self.currentAxes])
    
    def getView(self):
        """
        Return the current view over the cuboid :
        """
        #TODO : treat zero length array (?)
        
        if debug: print 'core.xndarrayView.getView ... mode=%d' %self.mode
        if debug: print 'data shape: %s -> %d' %(str(self.data.shape),
                                                 self.data.size)
        if debug: print 'mask size: %d' %(len(self.mask[0]))
        err = None

        sh = ()
        axn = []
        for ia in _N.sort(self.currentAxes):
            sh += (len(self.axesRanges[ia]),)
            if debug: print 'ia :', ia
            axn += [self.cuboid.getAxisName(ia),]
        if debug: print 'sh = ', sh
        if debug: print 'axn :', axn
        temp = _N.array(self.getAxesNames())[_N.sort(self.currentAxes)].tolist()
        dom = []
        for ica in self.currentAxes:
            d = self.cuboid.getAxisDomain(ica)
            if debug:
                print '*****ica :', ica
                print 'self.axesRanges :', self.axesRanges
                print 'd :', type(d)
            dom.append(d[self.axesRanges[ica]])
        if debug:
            print 'self.mask :', len(self.mask), len(self.mask[0])
            print 'sh :', sh , '-> ', _N.prod(sh)
        val = self.data[self.mask].reshape(sh)
        if self.errors != None:
            err = self.errors[self.mask].reshape(sh)
        outMask = self.computeOutMaskView(val, axn)
        ia = 0
        #if debug: print 'current axes :', temp
        #if debug: print 'target axes :', axn
        if debug: print  'Swapping if necessary ...'
        caxn = self.getCurrentAxesNames()
        while temp != caxn:
            if debug: print ' - temp axes :', temp
            if debug: print ' - target axes :', caxn
            if debug: print ' temp[%d]: %s ,axn[%d]: %s' \
               %(ia, temp[ia], ia, caxn[ia])
            if temp[ia] != caxn[ia]:
                a1 = temp[ia]
                a2 = caxn[ia]
                ia2 = temp.index(a2)
                temp[ia] = a2
                temp[ia2] = a1
                if debug: print ' swap %s<->%s => %d<->%d' %(a1, a2, ia, ia2)
                val = val.swapaxes(ia,ia2)
                if self.isMasked():
                    outMask = outMask.swapaxes(ia, ia2)
                if self.errors != None:
                    err = err.swapaxes(ia,ia2)
            ia += 1
        currentSlices = {}
        for ia in xrange(self.data.ndim):
            if ia not in self.currentAxes:
                d = self.cuboid.getAxisDomain(ia)
                currentSlices[self.cuboid.getAxisName(ia)] = d[self.slices[ia]]
        #if self.isMasked():
        #    currentSlices[self.maskName] = self.maskLabel
        r = {'values': val, 'errors': err,'domainNames':caxn,'domainValues':dom,
             'mask':outMask, 'maskLabel':self.maskLabel, 'slices':currentSlices}
             #'sliceMapping': [self.axesRanges[ia] for ia in self.currentAxes] }
        if debug: print 'core.xndarrayView.getView: return:'
        if outMask != None:
            if debug: print ' mask %s :' %str(r['mask'].shape)
            #if debug: print ' ->', _N.bincount(r['mask'].ravel())
        else:
            if debug: print ' Outmask None '
        if debug: print ' values %s :' %str(r['values'].shape)
        if debug: print ' values %1.3g(%1.3g) [%1.3g - %1.3g] :' \
           %(r['values'].mean(), r['values'].std(), r['values'].min(),
             r['values'].max())
        if debug: print ' domainNames :', axn
        #if debug: print 'domainValues :', dom
        return r
    
    #getView

    def get_sliced_axes(self):
        ans, cans = self.getAxesNames(), self.getCurrentAxesNames()
        return list(set(ans).difference(cans))

    def getAllViews(self):
        """
        Return a dictionary of all possible views along axes which are in the
        current axes.
        """
        slicedAxes = self.get_sliced_axes()
        if debug: print 'slicedAxes :', slicedAxes
        previousSlices = copyModule.copy(self.slices)
        allViews = {'root':None}
        self.appendAllViews(slicedAxes, allViews, 'root')
        self.slices = previousSlices
        return allViews
    #getAllViews
    
    def appendAllViews(self, axes, views, currentKey):

        if debug: print 'appendAllViews with currentKey :', currentKey
        if len(axes) == 0:
            if debug: print 'nothing to append next, appending the view'
            views[currentKey] = self.getView()['values']
            return
        if debug: print 'we got things to append next, adding a new store ...'
        views[currentKey] = {}
        views = views[currentKey]
        if debug: '%%%%% appendAllViews for :', axes
        ax = axes[0]
        if debug: '%%%%% Currently treating :', ax
        axd = self.getAxisDomain(ax)
        if debug: ' -> axd = ', axd
        for sl, dv in enumerate(axd):
            self.setSlice(ax, sl)
            if debug: print 'Appending to :', str((ax,sl))
            self.appendAllViews(axes[1:], views, (ax,dv))
    #appendAllViews
        
    def swapAxes(self, axis1, axis2):
        if debug: print 'core.xndarrayView.swapAxes: ax1: %s, ax2: %s' \
              %(str(axis1), str(axis2))
        can = self.getCurrentAxesNames()
        if (len(can)==1) or (axis1 not in can) or (axis2 not in can):
            if debug:
                print ' currentAxes not concerned'
        
        ica1 = can.index(axis1)
        ica2 = can.index(axis2)
        ca1 = self.currentAxes[ica1]
        self.currentAxes[ica1] = self.currentAxes[ica2]
        self.currentAxes[ica2] = ca1
    #swapAxes
    
    def setSlices(self, axisSlices):
        if debug: print 'core.xndarrayView.setSlice ', axisSlices
        
        for a,s in axisSlices.iteritems():
            self.setSlice(a,s)
            
    def printAxesRanges(self):
        if debug: print '**axesRanges:'
        i = 0
        for axr in self.axesRanges:
            if debug: print " %s : [%d - %d]" %(self.cuboid.getAxisName(i),
                                      min(axr),max(axr))
            i += 1
        if self.isMasked():
            if debug: print '**MaskaxesRanges:'
            i = 0
            for axr in self.maskAxesRanges:
                if debug: print " %s : [%d - %d]" %(self.outMask.getAxisName(i),
                                                    min(axr),max(axr))
                i += 1
            

    def getSliceIndex(self, axis, vSlice):
        """ Return the index associated to value 'vSlice'
        """
        if debug: print 'vSlice:', type(vSlice)
        iAxis = self.cuboid.getAxisId(axis)
        if type(vSlice) != int :
            dom = self.getAxisDomain(iAxis)
            if dom.dtype == _N.float64:
                dom = dom.astype(_N.float32)
            if debug: 
                print 'dom:', dom.dtype
                print dom==vSlice
                if vSlice == 0.7:
                    print dom[7]
                    print vSlice
                    print dom[7] - vSlice
            iSlice = _N.where(dom==vSlice)[0]
            if debug: print 'iSlice:', _N.where(dom==vSlice)
            if iSlice < 0 :
                raise('value not found for domain of axis %s : %s' \
                          (str(axis),str(vSlice)))
        else:
            iSlice = vSlice
        if debug: print 'getSliceIndex ... iSlice:', iSlice
        assert iSlice>=0
        assert iSlice<self.data.shape[iAxis]
        return iSlice
    
    def setSlice(self, axis, iSlice):
        """
        Set the current slice index for the given axis
        """
        if debug: print 'core.xndarrayView.setSlice: ax: %s, islice: %s' \
           %(str(axis), str(iSlice))
            
        iAxis = self.cuboid.getAxisId(axis)
        if self.isMasked():
            iAxisMask = self.outMask.getAxisId(axis)

        # if axis does not concern data:
        if iAxis == None or iAxis >= len(self.data.shape):
            return
        iSlice = self.getSliceIndex(axis, iSlice)
        if iAxis not in self.currentAxes:
            iSlice = _N.clip(iSlice,0,self.data.shape[iAxis]-1)
            if debug : print 'iSlice :', iSlice
            self.slices[iAxis] = iSlice
            self.axesRanges[iAxis] = [iSlice]
            if debug: self.printAxesRanges()
            ma = _N.zeros_like(self.mask[self.currentAxes[0]]) + iSlice
            self.mask[iAxis] = ma
            if self.isMasked() and iAxisMask!=None:
                self.maskAxesRanges[iAxisMask] = [iSlice]
                ma = _N.zeros_like(self.maskOfOutMask[self.currentMaskAxes[0]]) \
                     + iSlice
                self.maskOfOutMask[iAxisMask] = ma
        else:
            if debug:
                print 'core.xndarrayView.setSlice: axis in currentAxes ->'\
                      ' no effect'
            return

    #setSlice
            
    def getAxesNames(self):
        """
        Return the name of all axes.
        """
        return self.cuboid.getAxesNames()
    #getAxesNames    
#xndarrayView




class xndarrayViewNoMask:
    """
    Class managing projections and axis views on a xndarray object.
    """
    #TODO : maybe replace mode stuff with just specifying the number or
    #       dimensions in current view ...
    #      OR maybe it is implicit with the size of currentAxes ...
    MODE_1D = 0
    MODE_2D = 1
    MODE_3D = 2
    MODE_4D = 3
    
    def __init__(self, cuboid,  mode=None, currentAxes=None):
        """
        Initialize a new xndarrayView on 'cuboid'. 'mode' indicates the slicing
        mode used in the 'getView' method and 'currentAxes' the list of axes
        ids (by string name) involved in this projection.
        Examples:
            from numpy import *
            a = numpy.random.randn(10,10)
            c = xndarray(a)
            v = xndarrayView(c)
        """
        if debug : print 'Init of xndarrayView from cuboid :'
        if debug : print cuboid.descrip()
        self.cuboid = cuboid
        
        self.data = cuboid.data # shortcut to the viewed data

        if mode == None: # default mode
            if self.cuboid.getNbDims()>1:
                mode = self.MODE_2D
            else:
                mode = self.MODE_1D
        else: mode = mode
        if currentAxes == None: # default current axes : first axes by index
            if len(self.data.shape) > 1:
                currentAxes = [self.cuboid.getAxisName(0),
                               self.cuboid.getAxisName(1)]
            else:
                currentAxes = [self.cuboid.getAxisName(0)]

        if debug: print 'cuboidview.__init__: call setview ...'
        if debug: print ' -> ca = ', currentAxes

        self.slices = [0]*self.cuboid.getNbDims()
        self.setView(mode, currentAxes)
    #__init__

    def copy(self):
        niewView = xndarrayView(self.cuboid, self.mode, self.currentAxes,
                              self.cropFlag)
        return niewView

    def descrip(self):
        s = 'Current View:'
        s += '* viewed cuboid:'
        s += self.cuboid.descrip()
        s += '* current slices:'
        s += str(self.slices)
        s += '*current axes:'
        s += str(self.getCurrentAxesNames())
        return s

    def getValueLabel(self):
        """
        Return the label associated to values in the viewed cuboid.
        """
        return self.cuboid.getValueLabel()
    #def getValueLabel

    def getNbDims(self):
        """
        Return the number of dimensions of the viewed cuboid
        """
        return self.cuboid.getNbDims()
    
    def getMinValue(self):
        """
        Return the minimum value of data in the viewed cuboid
        """
        return self.cuboid.getMinValue()
    #getMinValue

    def getMaxValue(self):
        """
        Return the minimum value of data in the viewed cuboid
        """
        return self.cuboid.getMaxValue()
    #getMaxValue

    def setView(self, mode, curAxes):
        """
        Set the current view to either be 1D, 2D or 3D with 'mode'.
        Specify wich axes are involved in current view with 'curAxes' (which must
        contain string designations).
        Note : every slice is set to 0.
        """
        if debug : print 'core.cuboidView ... setting mode :', mode

        self.mode = mode
        
        if not ( type(curAxes)==list or type(curAxes)==tuple ) :
            # Then it is a single element:
            curAxes = [curAxes]

        if self.mode == self.MODE_2D:
            assert len(curAxes) == 2
        elif self.mode == self.MODE_1D:
            assert len(curAxes) == 1
        elif self.mode == self.MODE_3D:
            assert len(curAxes) == 3
        elif self.mode == self.MODE_4D:
            assert len(curAxes) == 4
        else:
            raise Exception('Mode "%s" not understood' %str(mode))
        
        self.currentAxes = [ self.cuboid.getAxisId(ca) for ca in curAxes ]
        while None in self.currentAxes:
            # Discard unknown axes:
            self.currentAxes.remove(None)
        
        if debug : print 'core.cuboidView.setView: new currentAxes: '
        if debug : print self.currentAxes

        self.buildAxesRanges()
    #setView

    def buildAxesRanges(self):
        """
        build ranges for current axes and set ranges of every other to 0
        (default slice position).
        """
        shape = self.data.shape
        self.axesRanges = [str(sl) for sl in self.slices]
        
        for ca in self.currentAxes:
            self.axesRanges[ca] = ':'
        
        #self.axesRanges = string.join(ar, ',')

    #buildAxesRanges
    
    def isCurrentAxis(self, id):
        return self.cuboid.getAxisId(id) in self.currentAxes

    def getCurrentAxes(self):
        """
        Return current axes (as integer ids) involved in the current view
        """
        return self.currentAxes
    #getCurrentAxes

    def getCurrentAxesNames(self):
        """
        Return current axes (as string ids) involved in the current view
        """
        return [self.cuboid.getAxisName(a) for a in self.currentAxes]
    #getCurrentAxesNames
        
    def getAxisDomain(self, axis):
        """
        Wrapper for xndarray.getAxisDomain method
        """
        return self.cuboid.getAxisDomain(axis)

    def getCurrentShape(self):
        return tuple([len(self.axesRanges[ia]) for ia in self.currentAxes])
    
    def getView(self):
        """
        Return the current view over the cuboid :
        """
        #TODO : treat zero length array (?)
        
        if debug: print 'core.xndarrayView.getView ... mode=%d' %self.mode
        if debug: print 'data shape: %s -> %d' %(str(self.data.shape),
                                                 self.data.size)
        err = None

        sh = ()
        axn = []
        for ia in _N.sort(self.currentAxes):
            sh += (len(self.axesRanges[ia]),)
            if debug: print 'ia :', ia
            axn += [self.cuboid.getAxisName(ia),]
        if debug: print 'sh = ', sh
        if debug: print 'axn :', axn
        temp = _N.array(self.getAxesNames())[_N.sort(self.currentAxes)].tolist()
        dom = []
        for ica in self.currentAxes:
            d = self.cuboid.getAxisDomain(ica)
            if debug:
                print '*****ica :', ica
                print 'self.axesRanges :', self.axesRanges
                print 'd :', type(d)
            dom.append(eval('d[%s]'%self.axesRanges[ica]))
        if debug:
            print 'sh :', sh , '-> ', _N.prod(sh)
        val = eval('self.data[%s]' %string.join(self.axesRanges, ','))
        ia = 0
        #if debug: print 'current axes :', temp
        #if debug: print 'target axes :', axn
        if debug: print  'Swapping if necessary ...'
        caxn = self.getCurrentAxesNames()
        while temp != caxn:
            if debug: print ' - temp axes :', temp
            if debug: print ' - target axes :', caxn
            if debug: print ' temp[%d]: %s ,axn[%d]: %s' \
               %(ia, temp[ia], ia, caxn[ia])
            if temp[ia] != caxn[ia]:
                a1 = temp[ia]
                a2 = caxn[ia]
                ia2 = temp.index(a2)
                temp[ia] = a2
                temp[ia2] = a1
                if debug: print ' swap %s<->%s => %d<->%d' %(a1, a2, ia, ia2)
                val = val.swapaxes(ia,ia2)
            ia += 1
        currentSlices = {}
        for ia in xrange(self.data.ndim):
            if ia not in self.currentAxes:
                d = self.cuboid.getAxisDomain(ia)
                currentSlices[self.cuboid.getAxisName(ia)] = d[self.slices[ia]]

        r = {'values': val, 'domainNames':caxn,'domainValues':dom,
             'slices':currentSlices}
             #'sliceMapping': [self.axesRanges[ia] for ia in self.currentAxes] }
        if debug: print 'core.xndarrayView.getView: return:'
        if debug: print ' values %s :' %str(r['values'].shape)
        if debug: print ' values %1.3g(%1.3g) [%1.3g - %1.3g] :' \
           %(r['values'].mean(), r['values'].std(), r['values'].min(),
             r['values'].max())
        if debug: print ' domainNames :', axn
        #if debug: print 'domainValues :', dom
        return r
    
    #getView

    def get_sliced_axes(self):
        ans, cans = self.getAxesNames(), self.getCurrentAxesNames()
        return list(set(ans).difference(cans))

    def getAllViews(self):
        """
        Return a dictionary of all possible views along axes which are in the
        current axes.
        """
        slicedAxes = self.get_sliced_axes()
        if debug: print 'slicedAxes :', slicedAxes
        previousSlices = copyModule.copy(self.slices)
        allViews = {'root':None}
        self.appendAllViews(slicedAxes, allViews, 'root')
        self.slices = previousSlices
        return allViews
    #getAllViews
    
    def appendAllViews(self, axes, views, currentKey):

        if debug: print 'appendAllViews with currentKey :', currentKey
        if len(axes) == 0:
            if debug: print 'nothing to append next, appending the view'
            views[currentKey] = self.getView()['values']
            return
        if debug: print 'we got things to append next, adding a new store ...'
        views[currentKey] = {}
        views = views[currentKey]
        if debug: '%%%%% appendAllViews for :', axes
        ax = axes[0]
        if debug: '%%%%% Currently treating :', ax
        axd = self.getAxisDomain(ax)
        if debug: ' -> axd = ', axd
        for sl, dv in enumerate(axd):
            self.setSlice(ax, sl)
            if debug: print 'Appending to :', str((ax,sl))
            self.appendAllViews(axes[1:], views, (ax,dv))
    #appendAllViews
        
    def swapAxes(self, axis1, axis2):
        if debug: print 'core.xndarrayView.swapAxes: ax1: %s, ax2: %s' \
              %(str(axis1), str(axis2))
        can = self.getCurrentAxesNames()
        if (len(can)==1) or (axis1 not in can) or (axis2 not in can):
            if debug:
                print ' currentAxes not concerned'
        
        ica1 = can.index(axis1)
        ica2 = can.index(axis2)
        ca1 = self.currentAxes[ica1]
        self.currentAxes[ica1] = self.currentAxes[ica2]
        self.currentAxes[ica2] = ca1
    #swapAxes
    
    def setSlices(self, axisSlices):
        if debug: print 'core.xndarrayView.setSlice ', axisSlices
        
        for a,s in axisSlices.iteritems():
            self.setSlice(a,s)
            
    def printAxesRanges(self):
        if debug: print '**axesRanges:'
        i = 0
        for axr in self.axesRanges:
            #if debug: print " %s : [%d - %d]" %(self.cuboid.getAxisName(i),
            #                          min(axr),max(axr))
            if debug: print " ",self.cuboid.getAxisName(i),": [",min(axr),"-",max(axr),"]"
            i += 1

    def getSliceIndex(self, axis, vSlice):
        """ Return the index associated to value 'vSlice'
        """
        if debug: print 'vSlice:', type(vSlice)
        iAxis = self.cuboid.getAxisId(axis)
        if type(vSlice) != int :
            dom = self.getAxisDomain(iAxis)
            if dom.dtype == _N.float64:
                dom = dom.astype(_N.float32)
            if debug: 
                print 'dom:', dom.dtype
                print dom==vSlice
                if vSlice == 0.7:
                    print dom[7]
                    print vSlice
                    print dom[7] - vSlice
            iSlice = _N.where(dom==vSlice)[0]
            if debug: print 'iSlice:', _N.where(dom==vSlice)
            if iSlice < 0 :
                raise('value not found for domain of axis %s : %s' \
                          (str(axis),str(vSlice)))
        else:
            iSlice = vSlice
        if debug: print 'getSliceIndex ... iSlice:', iSlice
        assert iSlice>=0
        assert iSlice<self.data.shape[iAxis]
        return iSlice
    
    def setSlice(self, axis, iSlice):
        """
        Set the current slice index for the given axis
        """
        if debug: print 'core.xndarrayView.setSlice: ax: %s, islice: %s' \
           %(str(axis), str(iSlice))
            
        iAxis = self.cuboid.getAxisId(axis)

        # if axis does not concern data:
        if iAxis == None or iAxis >= len(self.data.shape):
            return
        iSlice = self.getSliceIndex(axis, iSlice)
        if iAxis not in self.currentAxes:
            iSlice = _N.clip(iSlice,0,self.data.shape[iAxis]-1)
            if debug : print 'iSlice :', iSlice
            self.slices[iAxis] = iSlice
            self.axesRanges[iAxis] = str(iSlice)
            if debug: self.printAxesRanges()
        else:
            if debug:
                print 'core.xndarrayView.setSlice: axis in currentAxes ->'\
                      ' no effect'
            return

    #setSlice
            
    def getAxesNames(self):
        """
        Return the name of all axes.
        """
        return self.cuboid.getAxesNames()
    #getAxesNames    
#xndarrayViewNoMask




class xndarrayViewWatcher:

    def __init__(self, views=[]):
        """
        Initialise a new xndarrayViewWatcher object with the given list of views. 
        """
        self.axesViewMap = {}
        for v in views:
            self.addView(v)
    #__init__

    
    def registerView(self,view):
        """
        Add a view to be watched, i.e. create a mapping between this view and
        the names of its axes.
        """
        axes = view.getAxesNames()
        for a in axes:
            if not self.axesViewMap.has_key(a) :
                self.axesViewMap[a] = {}
            self.axesViewMap[a][view] = True
        view.addWatcher(self)
    #addView

    def linkAxis(self, view, axis):
        """
        Link 'axis' in the given view to others with the same name in other
        views 
        """
        #TODO : test for unknown mapping
        self.axesViewMap[axis][view] = True

    def unlinkAxis(self, view, axis):
        """
        Link 'axis' in the given view to others with the same name in other
        views 
        """
        #TODO : test for unknown mapping
        self.axesViewMap[axis][view] = False
    #unlinkAxis
    
    def removeView(self, view):
        """
        """
        axes = view.getAxesNames()
        for a in axes:
            self.axesViewMap[a].pop(view)
    #removeView
    
    def sliceChanged(self, view, axis, iSlice):
        """
        Spred slice changes among linked views
        """
        linkedViews = self.axesViewMap[axis].values()
        for lv in linkedViews:
            if lv != view:
                lv.setSlice(axis,iSlice)
    #sliceChanged

