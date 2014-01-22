# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import string
import numpy as np
import pyhrf
import sys
import os
import os.path as op
import pprint
import re
import csv
import copy as copyModule
import tempfile
from collections import defaultdict

from nibabel import gifti
import nibabel

from pyhrf.graph import graph_from_mesh, sub_graph, graph_is_sane
from pyhrf.ndarray import xndarray, MRI3Daxes, MRI4Daxes, TIME_AXIS
from pyhrf.ndarray import tree_to_cuboid
from pyhrf.tools import treeBranches, add_suffix, distance

PICKLE_DUMPED_FILE_EXTENSION = 'pck'
VOL_NII_EXTENSION = 'nii'
TEXTURE_EXTENSION = 'gii'

from _zip import *


def read_volume(fileName, remove_nans=True):
    """
    Read the content of 'filename' using nibabel.
    Return a tuple: (data numpy array, meta_data)
    Orientation:  nibabel convention
    """
    pyhrf.verbose(6, 'read_volume: %s' %fileName)
    #print 'filename:', fileName
    nim = nibabel.load(fileName)
    arr = np.array(nim.get_data()) #avoid memmapping
    # when array is memmapped from disk image:
    #Exception AttributeError: AttributeError("'NoneType' object has no attribute 'tell'",) in <bound method memmap.__del__ of memmap(
    # TODO: check and maybe fix this because this can be useful with big volumes
    if remove_nans:
        arr[np.isnan(arr)] = 0


    #print 'arr.shape:', arr.shape
    #if arr.ndim == 4:
    #    arr = np.rollaxis(arr, 3, 0)

    return arr, (nim.get_affine(), nim.get_header())

def cread_volume(fileName):
    return xndarray.load(fileName)

def read_spatial_resolution(fileName):
    nim = nibabel.load(fileName)
    return nim.get_header()['pixdim'][1:4]

def write_volume(data, fileName, meta_data=None):
    """
    Write the numpy array 'data' into a file 'filename' in nifti format,
    """
    pyhrf.verbose(5,'creating nifti image from data : ' + \
    str(data.shape) + str(data.dtype))

    if data.dtype == np.bool:
        data = data.astype(np.int8)

    if meta_data is None:
        affine = np.eye(4)
        header = None
    else:
        affine, header = meta_data
        header.set_data_dtype(data.dtype)

    img = nibabel.Nifti1Image(data, affine, header)
    pyhrf.verbose(5, 'Saving nifti Image to:' + fileName)
    img.to_filename(fileName)


def readImageWithPynifti(fileName, remove_nans=True):
    raise DeprecationWarning('Use read_volume instead ...')


def writeImageWithPynifti(data, fileName, header={}):
    raise DeprecationWarning('Use write_volume instead ...')




def hasNiftiExtension(fn):
    sfn = fn.split('.')
    return sfn[-1]=='nii' or (sfn[-1]=='gz' and sfn[-2]=='nii')

def splitFileName(fn, ext='nii'):

    sfn = fn.split('.')
    if sfn[-1] == ext:
        return (string.join(sfn[:-1],'.'), sfn[-1])
    elif sfn[-1] == 'gz' and sfn[-2] == ext:
        return string.join(sfn[:-2],'.'), string.join(sfn[-2:],'.')
    else:
        print 'SplitFileName : unrecognised extension!'
        return (fn,None)

def splitNiftiFileName(fn):
    sfn = fn.split('.')
    if sfn[-1] == 'nii':
        return (string.join(sfn[:-1],'.'), sfn[-1])
    elif sfn[-1] == 'gz' and sfn[-2] == 'nii':
        return string.join(sfn[:-2],'.'), string.join(sfn[-2:],'.')
    else:
        print 'SplitNiftiFileName : unrecognised nifti ext!'
        return (fn,None)

def writeDictVolumesNiftii(vols, fileName, meta_data=None, sep='_', slice=None):
    pyhrf.verbose(4, 'writeDictVolumesNiftii, fn:' + fileName)
    if vols is None:
        return
    if slice is None:
        slice = tuple()

    if type(vols)==dict:
        (baseFileName, ext) = splitFileName(fileName,'nii')
        filenames = []
        for k,v in vols.iteritems():
            #print '%%% k:', k
            #print '%%% v:', v
            if k != 'root':
                newFn = baseFileName+sep+k[0]+sep+str(k[1])+'.'+ext
                if np.isreal(k[1]):
                    sl = (k[1],)
                else:
                    sl = (str(k[1]),)
            else:
                newFn = fileName
                sl = tuple()
            #print 'newFn :', newFn
            #print 'slice :', slice
            filenames.extend(writeDictVolumesNiftii(v, newFn, meta_data,
                                                    sep, slice+sl))
        return filenames
    else: # I got a numpy.ndarray volume
        if 0:
            print '~~~~~~~~~~~~~~~~~~'
            print 'Got ndarray ... writing it ...'
            print 'vols.shape', vols.shape
            print '~~~~~~~~~~~~~~~~~~'
        write_volume(vols, fileName, meta_data)
        #print '->slice:,', slice
        #print '->fn:', fileName
        return [(slice,fileName)]


def writexndarrayToNiftii(cuboid, fileName, meta_data=None, sep='_'):
    raise DeprecationWarning('writexndarrayToNiftii should not be used any more')
    axns = cuboid.getAxesNames()
    pyhrf.verbose(3, 'writexndarrayToNiftii, fn:' + fileName)
    pyhrf.verbose(3, '-> got axes :' + str(axns))

    if 'sagittal' not in axns or 'coronal' not in axns or 'axial' not in axns:
        raise Exception('xndarray not writable as nifti image')

    #target_axes = MRI3Daxes + list(set(axns).difference(MRI3Daxes))
    #print 'target_axes:', target_axes

    if 'iteration' in axns and 'time' not in axns :
        v = xndarrayViewNoMask(cuboid, mode=xndarrayViewNoMask.MODE_4D,
                             currentAxes=MRI3Daxes+['iteration'])
    elif 'time' in axns:
        v = xndarrayViewNoMask(cuboid, mode=xndarrayViewNoMask.MODE_4D,
                             currentAxes=MRI3Daxes+['time'])

    else : # 3D volumes to write
        v = xndarrayViewNoMask(cuboid, mode=xndarrayViewNoMask.MODE_3D,
                             currentAxes=MRI3Daxes)

    #print 'cuboid axes:', cuboid.axes_names
    #print 'view currentAxes:', v.currentAxes

    #print 'writexndarrayToNiftii, base fileName :', fileName
    #print '%%%%%%%%%%% getting all views ....'

    #meta_obj['volume_dimension'] = v.getCurrentShape()
    #meta_obj['data_type'] = 'DOUBLE'
    #meta_obj['disk_data_type'] = 'DOUBLE'
    vols = v.getAllViews()
    eaxes = v.get_sliced_axes()
    if 0:
        print '--------------------'
        print 'got allViews :'
        print vols
        print '--------------------'
    fns = dict(writeDictVolumesNiftii(vols, fileName, meta_data, sep=sep))
    return eaxes,fns


def writeDictVolumesTex(vols, fileName):
    pyhrf.verbose(4, 'writeDictVolumesTex, fn:' + fileName)
    if vols == None:
        return
    if type(vols)==dict:
        (baseFileName, ext) = splitFileName(fileName, TEXTURE_EXTENSION)
        for k,v in vols.iteritems():
            #print '%%% k:', k
            #print '%%% v:', v
            if k != 'root':
                newFn = baseFileName+'_'+k[0]+'_'+str(k[1])+'.'+ext
            else:
                newFn = fileName
            #print 'newFn :', newFn
            writeDictVolumesTex(v, newFn)
    else: # I got a numpy.ndarray volume
        #print 'Got ndarray ... writing it ...'
        #tex = Texture(fileName, data=vols)
        #tex.write()
        write_texture(vols, fileName)

def writexndarrayToTex(cuboid, fileName):
    axns = cuboid.getAxesNames()
    if 'voxel' not in axns:
        return
    if 'time' in axns:
        v = xndarrayViewNoMask(cuboid, mode=xndarrayViewNoMask.MODE_2D,
                       currentAxes=['voxel', 'time'])
    else:
        v = xndarrayViewNoMask(cuboid, mode=xndarrayViewNoMask.MODE_1D,
                       currentAxes=['voxel'])
    texs = v.getAllViews()
    writeDictVolumesTex(texs, fileName)

if 0:
    ### old ndarray <-> xml feature
    from pyhrf.xmlio import XMLParamDrivenClass
    from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
    from pyhrf.tools import cartesian, set_leaf


    class xndarrayXml(XMLParamDrivenClass):

        defaultParameters = {
            'domains' : {},
            'orientation' : [],
            'explodedAxes' : [],
            'name' : 'cname',
            'value_label' : 'value',
            'dataFiles' : [],
            }

        @staticmethod
        def slice_idx_from_filename(fn, cname):

            i = 1
            while '_'*i in fn:
                i += 1
            sep = '_'*(i-1)
            #print 'sep:', sep
            #print 'cname:', cname
            #print 'fn:', fn
            sfn = fn[len(cname)+len(sep):]
            #print 'sfn:', sfn
            l = sfn.split(sep)
            return l[::2], l[1::2]


        def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                     xmlLabel=None, xmlComment=None):
            XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel,
                                         xmlComment)

            clabel = self.parameters['name']
            ad = self.parameters['domains']
            an = self.parameters['orientation']
            vlab = self.parameters['value_label']
            fl = self.parameters['dataFiles']
            eaxes = self.parameters['explodedAxes']
            self.outDir = './'
            #print ''
            #print 'clabel:', clabel
            #print 'eaxes :', eaxes
            #print 'an:', an

            if len(fl) > 0:
                ed = dict(zip(eaxes,[ad[ea] for ea in eaxes]))
                if len(eaxes) > 0:
                    iaxes = list(set(an).difference(eaxes))
                else:
                    iaxes = an
                idomains = dict(zip(iaxes,[ad[ia] for ia in iaxes]))
                ctree = {}
                if 0:
                    print 'fl:', fl
                    print 'id:', idomains
                    print 'iaxes:', iaxes
                    print 'ad:', ad
                    print 'ed:', ed
                    print 'eaxes:', eaxes
                if len(eaxes) > 0:
                    for fn in fl:
                        n = os.path.splitext(os.path.basename(fn))[0]
                        blabels, bvals = self.slice_idx_from_filename(n, clabel)
                        if 0:
                            print 'blabels:', blabels
                            print 'bvals:', bvals
                            print 'fn:', fn
                        cdata = self.load_cuboid(fn, iaxes, vlab, idomains)
                        #print 'ctree:'
                        #print ctree
                        #print 'cdata:', cdata
                        set_leaf(ctree, bvals, cdata)
                    self.cuboid = tree_to_cuboid(ctree, branchLabels=blabels)
                else:
                    self.cuboid = self.load_cuboid(fl[0], iaxes, vlab, idomains)

                self.cuboid.set_orientation(an)
            else:
                self.cuboid = None
            #    self.cuboid = xndarray(np.array([]),axes_names=an,
            #                         axes_domains=ad, value_label=vlab)

        def load_cuboid(self, fn, iaxes, vlab, idomains):
            if 0:
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print 'load_cuboid:'
                print 'iaxes:', iaxes
            base, ext = os.path.splitext(fn)
            if ext == '.nii':
                data, meta_data = read_volume(fn)

                if 'iteration' in iaxes:
                    iaxes = MRI3Daxes + 'iteration'
                elif 'time' in iaxes:
                    iaxes = list(MRI4Daxes)
                else:
                    iaxes = list(MRI3Daxes)

                if ('iteration' in iaxes or 'time' in iaxes) and data.ndim == 3:
                    data = data.reshape(1,data.shape[0],data.shape[1],
                                        data.shape[2])

            elif ext == '.h5':
                import tables
                h5file = tables.openFile(fn, mode='r')
                data = h5file.root.array.read()
                h5file.close()
            elif ext == '.csv':
                data = np.loadtxt(fn)
            #print 'data:', data.shape, data.ndim
            #print 'iaxes:', iaxes
            #print 'id:', idomains
            #print 'vlab:', vlab
            if data.ndim == 0:
                data.shape = tuple([1]*len(iaxes))
            elif len(iaxes) == 2 and data.ndim == 1:
                if len(idomains[iaxes[0]]) == 1:
                    data.shape = (1,data.shape[0])
                else:
                    data.shape = (data.shape[0],1)
            assert data.ndim == len(iaxes)
            return xndarray(data, axes_names=iaxes, axes_domains=idomains,
                          value_label=vlab)


        @staticmethod
        def fromxndarray(c, label='cname', outDir='./'):
            p = {
                'domains' : c.getAxesDomainsDict(),
                'orientation' : c.axes_names,
                'value_label' : c.value_label,
                'name' : label,
                }
            cxml = xndarrayXml(p)
            cxml._set_cuboid(c)
            cxml._set_output_dir(outDir)
            return cxml

        def _set_cuboid(self, c):
            self.cuboid = c

        def _set_output_dir(self, d):
            self.outDir = d


        @staticmethod
        def longest_underscore(s):
            return max(map(len,re.compile('[^_]+').subn('-',s)[0].split('-')))

        def appendParametersToDOMTree(self, doc, node):
            if self.cuboid is not None:
                if 0:
                    print ''
                    print 'Appending to DOM tree:'
                    print '- parameters :'
                    print self.parameters
                    print '- cuboid descrip:'
                    print self.cuboid.descrip()
                lu = max([self.longest_underscore(an) for an in self.cuboid.getAxesNames()])
                lud = 0
                for d in self.cuboid.getAxesDomains():
                    if 'str' in d.dtype.name:
                        lud = max([self.longest_underscore(v) for v in d])
                sep = '_'*(max([lud, lu])+1)

                if self.cuboid.is_nii_writable():
                    fn0 = op.join(self.outDir, self.parameters['name']+'.nii')
                    eaxes, fns = writexndarrayToNiftii(self.cuboid, fn0, sep=sep)
                    fns = fns.values()
                elif self.cuboid.getNbDims() <= 2:
                    eaxes = []
                    fns = [op.join(self.outDir,self.parameters['name'] + '.csv')]
                    self.cuboid.save(fns[0])
                else:
                    eaxes = []
                    fns = [op.join(self.outDir,self.parameters['name'] + '.h5')]
                    self.cuboid.save(fns[0])

                self.parameters['dataFiles'] = fns
                self.parameters['explodedAxes'] = eaxes
                #print 'eaxes:', eaxes
                XMLParamDrivenClass.appendParametersToDOMTree(self, doc, node)

        def cleanFiles(self):
            for f in self.parameters['dataFiles']:
                if os.path.exists(f):
                    os.remove(f)


    class xndarrayXml2(XMLParamDrivenClass):
        loadxndarray = True
        savexndarray = True

        defaultParameters = {
            'domains' : {},
            'orientation' : [],
            'explodedAxes' : [],
            'name' : 'cname',
            'value_label' : 'value',
            'dataFiles' : {},
            }

        @staticmethod
        def slice_idx_from_filename(fn, cname):

            i = 1
            while '_'*i in fn:
                i += 1
            sep = '_'*(i-1)
            #print 'sep:', sep
            #print 'cname:', cname
            #print 'fn:', fn
            sfn = fn[len(cname)+len(sep):]
            #print 'sfn:', sfn
            l = sfn.split(sep)
            return l[::2], l[1::2]


        def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                     xmlLabel=None, xmlComment=None):
            if 0 : print 'xndarrayXml2.__init__ ...'
            XMLParamDrivenClass.__init__(self, parameters, xmlHandler, xmlLabel,
                                         xmlComment)

            clabel = self.parameters['name']
            ad = self.parameters['domains']
            an = self.parameters['orientation']
            vlab = self.parameters['value_label']
            fl = self.parameters['dataFiles']
            eaxes = self.parameters['explodedAxes']
            self.outDir = './'
            self.meta_data = None
    #         print ''
    #         print 'clabel:', clabel
    #         print 'eaxes :', eaxes
    #         print 'an:', an

            if len(fl) > 0 and self.loadxndarray:
                ed = dict(zip(eaxes,[ad[ea] for ea in eaxes]))
                if len(eaxes) > 0:
                    iaxes = []
                    for a in an:
                        if a not in eaxes:
                            iaxes.append(a)
                    #iaxes = list(set(an).difference(eaxes))
                else:
                    iaxes = an
                idomains = dict(zip(iaxes,[ad[ia] for ia in iaxes]))
                ctree = {}
                if 0:
                    print 'fl:', fl
                    print 'id:', idomains
                    print 'iaxes:', iaxes
                    print 'ad:', ad
                    print 'ed:', ed
                    print 'eaxes:', eaxes
                if len(eaxes) > 0:
                    for slice,fn in fl.iteritems():
                        slice = list(slice)
                        if 0:
                            print 'slice:', slice
                            print 'fn:', fn
                        if slice is not None:
                            if '_' in slice:
                                slice.remove('_')
                                if '_' in slice:
                                    slice.remove('_')
                        cdata = self.load_cuboid(fn, iaxes, vlab, idomains)
                        #print 'ctree:'
                        #print ctree
                        #print 'cdata:', cdata
                        set_leaf(ctree, list(slice), cdata)
                    self.cuboid = tree_to_cuboid(ctree, branchLabels=eaxes)
                else:
                    self.cuboid = self.load_cuboid(fl.values()[0], iaxes, vlab, idomains)

                self.cuboid.set_orientation(an)
            else:
                self.cuboid = None

            self.useRelativePath = False
            #    self.cuboid = xndarray(np.array([]),axes_names=an,
            #                         axes_domains=ad, value_label=vlab)

        def load_cuboid(self, fn, iaxes, vlab, idomains):
            base, ext = os.path.splitext(fn)
            if 0:
                print 'xndarrayXml2.load_cuboid ...'
                print 'input iaxes:', iaxes
            #print 'fn :', fn
            if ext == '.nii':
                data, self.meta_data = read_volume(fn)
                if 'iteration' in iaxes:
                    iaxes = MRI3Daxes + ['iteration']
                elif 'time' in iaxes:
                    iaxes = list(MRI4Daxes)
                else:
                    iaxes = list(MRI3Daxes)


                # print '-> iaxes', iaxes
                # print 'MRI4Daxes:', MRI4Daxes

                if ('iteration' in iaxes or 'time' in iaxes) and data.ndim == 3:
                    data = data.reshape(1,data.shape[0],data.shape[1],data.shape[2])
            elif ext == '.h5':
                import tables
                h5file = tables.openFile(fn, mode='r')
                data = h5file.root.array.read()
                h5file.close()
            elif ext == '.csv':
                data = np.loadtxt(fn)
            if 0:
                print '--------------------'
                print 'before cuboid creation ...'
                print 'data:', data.shape, data.ndim
                print 'iaxes:', iaxes
                #print 'id:', idomains
                print 'vlab:', vlab
                print '--------------------'
            if data.ndim == 0:
                data.shape = tuple([1]*len(iaxes))
            elif len(iaxes) == 2 and data.ndim == 1:
                if len(idomains[iaxes[0]]) == 1:
                    data.shape = (1,data.shape[0])
                else:
                    data.shape = (data.shape[0],1)
            assert data.ndim == len(iaxes)

            return xndarray(data, axes_names=iaxes, axes_domains=idomains,
                          value_label=vlab)


        @staticmethod
        def fromxndarray(c, label='cname', outDir='./', relativePath=True,
                       meta_data=None):
            p = {
                'domains' : c.getAxesDomainsDict(),
                'orientation' : c.axes_names,
                'value_label' : c.value_label,
                'name' : label,
                }
            cxml = xndarrayXml2(p)
            cxml._set_cuboid(c)
            cxml._set_output_dir(outDir)
            cxml.set_relative_path(relativePath)
            cxml.meta_data = meta_data
            return cxml

        def _set_cuboid(self, c):
            self.cuboid = c

        def _set_output_dir(self, d):
            self.outDir = d

        def set_relative_path(self, p):
            self.useRelativePath = p

        def make_relative_path(self):
            """
            Make data file pathes be relative. Simply replace
            every file name with its basename since by convention
            all results are in the same directory as the output.xml
            """
            self.set_relative_path(True)
            cfl = self.parameters['dataFiles']
            for slice,fn in cfl.iteritems():
                #print 'ofn:', cfl[slice]
                cfl[slice] = op.basename(fn)
                #print 'nfn:', cfl[slice]

        def make_absolute_path(self,root='./'):
            #print 'make_absolute_path ...'
            self.set_relative_path(False)
            cfl = self.parameters['dataFiles']
            for slice,fn in cfl.iteritems():
                #print 'fn:', fn
                cfl[slice] = op.abspath(op.join(root,fn))
                #print 'nfn:', cfl[slice]

        @staticmethod
        def longest_underscore(s):
            return max(map(len,re.compile('[^_]+').subn('-',s)[0].split('-')))

        @staticmethod
        def stack(clist, axisLabel, axisDomain):
            pyhrf.verbose(6, 'xndarrayXml2.stack ...')
            pyhrf.verbose(6, 'axisLabel: %s, axisDomain: %s' %(axisLabel, str(axisDomain)))
            pyhrf.verbose(6, 'clist:')
            pyhrf.verbose(6, str(clist))

            cxml = copyModule.deepcopy(clist[0])
            cxml.parameters['domains'][axisLabel] = axisDomain
            #print 'orientation:',cxml.parameters['orientation']
            cxml.parameters['orientation'].append(axisLabel)
            cxml.parameters['explodedAxes'].append(axisLabel)
            fl = {}


            for c,dv in zip(clist, axisDomain):
                if 0:
                    print 'c:', c
                    print 'dv:', dv
                cfl = c.parameters['dataFiles']
                for slice,fn in cfl.iteritems():
                    if 0:
                        print 'slice:', slice
                        print 'fn:', fn
                    if slice is not None:
                        newSlice = slice + (dv,)
                    else:
                        newSlice = (dv,)
                    fl[newSlice] = fn
            cxml.parameters['dataFiles'] = fl
            return cxml

        def appendParametersToDOMTree(self, doc, node):
            if self.savexndarray:
                if self.cuboid is not None :
                    if 0:
                        print ''
                        print 'Appending to DOM tree:'
                        print '- parameters :'
                        print self.parameters
                        print '- cuboid descrip:'
                        print self.cuboid.descrip()
                    sep = '_'

                    if self.cuboid.is_nii_writable():
                        fn0 = op.join(self.outDir, self.parameters['name']+'.nii')
                        pyhrf.verbose(3, 'Writing cuboid for %s ...' \
                                          %self.parameters['name'])
                        eaxes, slicefns = writexndarrayToNiftii(self.cuboid, fn0,
                                                              sep=sep,
                                                              meta_data=self.meta_data)
                        pyhrf.verbose(3, 'Written cuboid for %s' \
                                          %self.parameters['name'])
                    elif self.cuboid.getNbDims() <= 2:
                        eaxes = []
                        fn = op.join(self.outDir,self.parameters['name'] + '.csv')
                        #slicefns = { ('_','_') : fn }
                        slicefns = { None : fn }
                        self.cuboid.save(fn)
                    else:
                        eaxes = []
                        fn = op.join(self.outDir,self.parameters['name'] + '.h5')
                        #slicefns = { ('_','_') : fn }
                        slicefns = { None : fn }
                        self.cuboid.save(fn)


                    for s,f in slicefns.iteritems():
                        if self.useRelativePath:
                            slicefns[s] = op.join('./',op.basename(f))
                        else:
                            slicefns[s] = op.abspath(f)

                    self.parameters['dataFiles'] = slicefns
                    self.parameters['explodedAxes'] = eaxes
                    #print 'eaxes:', eaxes
                    XMLParamDrivenClass.appendParametersToDOMTree(self, doc, node)
                else:
                    raise Exception('No cuboid to save')
            else:
                XMLParamDrivenClass.appendParametersToDOMTree(self, doc, node)

        def cleanFiles(self):
            for f in self.parameters['dataFiles'].itervalues():
                if os.path.exists(f):
                    os.remove(f)

    def tree_to_cuboidXml2(tree, branchLabels=None):
        # print 'tree_to_cuboidXml2 recieve:'
        # pprint.pprint(tree)
        if branchLabels is None:
            branchLabels = ['b%d'%i for i in xrange(len(treeBranches(tree).next()))]
        assert len(branchLabels) == len(treeBranches(tree).next())

        if isinstance(tree[tree.keys()[0]], dict):
            for k in tree.iterkeys():
                tree[k] = tree_to_cuboidXml2(tree[k], branchLabels[1:])
        else:
            kiter = sorted(tree.keys())
            allLeaves = []
            #print 'axis domain for stacking:'
            #print kiter
            for k in kiter:
                allLeaves.append(tree[k])
            return xndarrayXml2.stack(allLeaves, branchLabels[0], kiter)

        return tree_to_cuboidXml2(tree, [branchLabels[0]])

### End of old ndarray <-> xml feature

def sub_sample_vol(image_file, dest_file, dsf, interpolation='continuous',
                   verb_lvl=0):

    from nipy.labs.datasets.volumes.volume_img import VolumeImg #,CompositionError

    pyhrf.verbose(verb_lvl, 'Subsampling at dsf=%d, %s -> %s' \
                  %(dsf, image_file, dest_file))

    interp = interpolation
    data_src, src_meta = read_volume(image_file)
    affine = src_meta[0]
    daffine = dsf*affine

    original_dtype = data_src.dtype
    if np.issubdtype(np.int, data_src.dtype):
        # to avoid error "array type 5 not supported" on int arrays ...
        #data_src = np.asarray(np.asfarray(data_src), data_src.dtype)
        data_src = np.asfarray(data_src)

    if data_src.ndim == 4:
        ref_vol = data_src[:,:,:,0]
    else:
        ref_vol = data_src
    #print 'ref_vol:', ref_vol.shape

    img_src = VolumeImg(ref_vol, affine , 'mine')
    img_dest = img_src.as_volume_img(daffine, interpolation=interpolation)

    # setup dest geometry:
    dest_header = src_meta[1].copy()

    #dest_header['sform'] = img_src.affine
    dpixdim = np.array(src_meta[1]['pixdim'])
    dpixdim[1:4] *= dsf
    # print 'pixdim:', ','.join(map(str,src_meta[1]['pixdim']))
    #print 'dpixdim:', ','.join(map(str,dpixdim))

    dest_header['pixdim'] = list(dpixdim)
    sh = ref_vol[::dsf, ::dsf, ::dsf, ...].shape
    #print 'sh:', sh
    dest_meta = (daffine, dest_header)

    # do the resampling:
    if data_src.ndim == 3:
        vol = img_dest.get_data()[:sh[0],:sh[1],:sh[2]]
        if ref_vol.dtype == np.int32 or \
                np.allclose(np.round(ref_vol), ref_vol): #if input is int
            vol = np.round(vol).astype(np.int32)
        write_volume(vol , dest_file, dest_meta)

    elif data_src.ndim == 4:
        imgs = [VolumeImg(i, affine, 'mine') \
                    for i in np.rollaxis(data_src,3,0)]
        dvols = [i.as_volume_img(daffine,interpolation=interp).get_data() \
                     for i in imgs]
        #print 'dvols[0]:', dvols[0].shape
        dvols = np.array(dvols)
        #print 'dvols:', dvols.shape
        sub_vols = np.rollaxis(dvols[:,:sh[0],:sh[1],:sh[2]], 0, 4)
        #print 'sub_vols:', sub_vols.shape
        write_volume(sub_vols, dest_file, dest_meta)

    else:
        raise Exception('Nb of dims (%d) not handled. Only 3D or 4D' \
                            %data_src.ndim)



def concat3DVols(files, output):
    """ Concatenate 3D volumes given by a list a file to 4D volume (output)
    """
    for i,f in enumerate(files):
        img, meta_data = read_volume(f)
        img = img.squeeze() #remove dim of length=1
        if i == 0:
            pyhrf.verbose(2, "Image meta_obj: \n" + \
                              pprint.pformat(meta_data))
            pyhrf.verbose(1, "Volume size: %s, type: %s" \
                              %(str(img.shape), str(img.dtype)))
            img4D = np.zeros(img.shape + (len(files),), img.dtype)
            print 'img4D.shape:', img4D.shape
        img4D[:,:,:,i] = img
    write_volume(img4D, output, meta_data=meta_data)


def split_ext_safe(fn):
    root,ext = op.splitext(fn)
    if ext == '.gz':
        root,n_ext = op.splitext(root)
        ext = n_ext + ext
    return root,ext

def split4DVol(boldFile, output_dir=None):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='pyhrf',
                                      dir=pyhrf.cfg['global']['tmp_path'])
    root, ext = split_ext_safe(boldFile)
    img, meta_data = read_volume(boldFile)
    #print '4D BOLD header:'
    #pprint.pprint(h)

    img = img.squeeze() #remove dim of length=1
    fns = []
    for i in xrange(img.shape[TIME_AXIS]):
        fn = op.join(output_dir,op.basename(root)) + '_%06d'%i + ext
        if TIME_AXIS == 0:
            write_volume(img[i,:,:,:], fn, meta_data)
        else:
            write_volume(img[:,:,:,i], fn, meta_data)
        fns.append(fn)
    return fns

from pyhrf.graph import parcels_to_graphs, kerMask3D_6n

# DATA sanity checks:
# - volume shapes across sessions



def load_paradigm_from_csv(csvFile, delim=' '):
    """
    Load an experimental paradigm stored in the CSV file *csvFile*,
    with delimiter *delim*.

    Column must correspond to (5th is optional):

    col 1      |  col 2       |  col 3          |  col 4         |  [col 5]
    sess. idx  |  cond. name  |  onset in sec.  |  dur. in sec.  |  [amplitude]

    Actually, the amplitude info is ignored. TODO: retrieve amplitude


    Returns
    -------
    returned tuple:
        (list of onsets, list of session durations).

    Each list is actually a list of lists, eg: onsets[session][event]
    """

    reader = csv.reader(open(csvFile, "rb"), delimiter=delim,
                        skipinitialspace=True)

    onsets = defaultdict(lambda : defaultdict(list))
    durations = defaultdict(lambda : defaultdict(list))

    for row in reader:
        #print 'row:', row
        isession = int(row[0])
        cond = row[1]
        onset = float(row[2])

        amplitude = 1.
        duration = 0.0
        if len(row) > 3: duration = float(row[3])
        if len(row) > 4: amplitude = float(row[4])

        onsets[cond][isession].append(onset)
        durations[cond][isession].append(duration)


    onsets = dict( [(cn,[np.array(o[s])for s in sorted(o.keys())]) \
                        for cn,o in onsets.iteritems()] )

    durations = dict( [(cn,[np.array(dur[s]) for s in sorted(dur.keys())]) \
                           for cn,dur in durations.iteritems()] )

    return onsets, durations


# def load_paradigm_from_csv(paradigmFile):
#     """
#     Load paradigm from a CSV file which columns are:
#     Task, stimulation onset, stimulation duration, session
#     """
#     onsets = {}
#     durations = {}
#     reader = csv.reader(open(paradigmFile).readlines(),delimiter=' ')
#     for row in reader:
#         cond = row[1]
#         onset = float(row[2])
#         #duration = float(row[2])
#         duration = 0.0
#         isession = int(row[0])
#         if not onsets.has_key(isession):
#             onsets[isession] = {}
#             durations[isession] = {}
#         if not onsets[isession].has_key(cond):
#             onsets[isession][cond] = []
#             durations[isession][cond] = []
#         onsets[isession][cond].append(onset)
#         durations[isession][cond].append(duration)

#     for isession in onsets.iterkeys():
#         for cond in onsets[isession].iterkeys():
#             onsets[isession][cond] = np.array(onsets[isession][cond])
#             durations[isession][cond] = np.array(durations[isession][cond])
#             onsorder = onsets[isession][cond].argsort()
#             onsets[isession][cond] = onsets[isession][cond][onsorder]
#             durations[isession][cond] = durations[isession][cond][onsorder]

#     obak = copyModule.deepcopy(onsets)
#     dbak = copyModule.deepcopy(durations)
#     onsets = range(len(onsets))
#     durations = range(len(onsets))
#     for isession in obak.iterkeys():
#         onsets[isession] = obak[isession]
#         durations[isession] = dbak[isession]

#     return onsets, durations



def load_fmri_vol_data(boldFiles, roiMaskFile, keepBackground=False,
                       build_graph=True):
    """Load FMRI BOLD volumic data from files.

    Arguments:
    boldFiles --  pathes to 4D nifti data files (ie multisession data).
    roiMaskFile -- a path a 3D nifti data file.
    keepBackground -- extract background data: bold and graph which
                      may be heavy to compute

    Return:
    A tuple (graphs, bolds, sessionScans, roiMask, dataHeader)
    graphs -- a dict {roiId:roiGraph}
    bold -- a dict {roiId:bold}
    sessionScans -- a list of scan indexes for every session
    roiMask -- a 3D numpy int array defining ROIs (0 stands for the background)
    dataHeader -- the header from the BOLD data file
    """
    for boldFile in boldFiles:
        if not os.path.exists(boldFile):
            raise Exception('File not found: ' + boldFile)
        #assert hasNiftiExtension(boldFile)

    if not os.path.exists(roiMaskFile):
        raise Exception('File not found: ' + roiMaskFile)
    #assert hasNiftiExtension(roiMaskFile)

    #Load ROIs:
    try:
        roiMask, mask_meta_data = read_volume(roiMaskFile)
    except Exception, e:
        print 'Could not read file %s, error was:\n %s' %(roiMaskFile, str(e))
        raise e
    pyhrf.verbose(1,'Assuming orientation for mask files: ' + \
                      string.join(MRI3Daxes, ','))
    pyhrf.verbose(1,'Loaded a mask of shape %s' %str(roiMask.shape))
    mshape = roiMask.shape
    if roiMask.min() == -1:
        roiMask += 1

    #Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    pyhrf.verbose(1,'Assuming orientation for BOLD files: ' + \
                      string.join(MRI4Daxes, ','))
    for boldFile in boldFiles:
        try:
            b, b_meta = read_volume(boldFile)
        except Exception, e:
            print 'Could not read file %s, error was:\n %s' %(boldFile, str(e))
        bolds.append(b)
        pyhrf.verbose(1,'Loaded a BOLD file of shape %s' %str(b.shape))

        if (TIME_AXIS == 0 and mshape != b.shape[1:]) or \
                (TIME_AXIS == 3 and mshape != b.shape[:-1]):
            raise Exception('Error: BOLD shape is different from mask shape')

        sessionScans.append(np.arange(lastScan, lastScan+b.shape[TIME_AXIS],
                                      dtype=int))
        lastScan += b.shape[TIME_AXIS]
    bold = np.concatenate(tuple(bolds), axis=TIME_AXIS)

    discard_bad_data(bold, roiMask)

    if build_graph:
        # Build graph

        if len(np.unique(roiMask))==1 or keepBackground:
            graphs = parcels_to_graphs(roiMask.astype(int), kerMask3D_6n)
        else:
            if (roiMask==0).any():
                toDiscard = [0]
                pyhrf.verbose(1,'Discarding background (label=0)')
            else:
                toDiscard = None
            graphs = parcels_to_graphs(roiMask.astype(int), kerMask3D_6n,
                                       toDiscard=toDiscard)
        pyhrf.verbose(1,'Graph built !')
    else:
        graphs = None

    #Split bold into rois:
    roiBold = {}
    for roiId in np.unique(roiMask):
        mroi = np.where(roiMask==roiId)
        if TIME_AXIS==0:
            roiBold[roiId] = bold[:, mroi[0], mroi[1], mroi[2]]
        else:
            roiBold[roiId] = bold[mroi[0], mroi[1], mroi[2], :]

    return graphs, roiBold, sessionScans, roiMask, mask_meta_data

def discard_bad_data(bold, roiMask, time_axis=TIME_AXIS):
    """ Discard positions in 'roiMask' where 'bold' has zero variance or
    contains NaNs
    """
    m = roiMask
    #print 'roiMask:', roiMask.shape
    #print 'bold.shape:', bold.shape
    var_bold = (bold**2).sum(time_axis)/(bold.shape[time_axis]-1) - \
        bold.mean(time_axis)**2
    #zeroVarianceVoxels = np.bitwise_and(m>=1, bold.std(0)==0.) #TODO check MemoryError ?
    zeroVarianceVoxels = np.bitwise_and(m>=1, var_bold==0.)
    #print 'zeroVarianceVoxels:', zeroVarianceVoxels.shape
    if zeroVarianceVoxels.any():
        pyhrf.verbose(6, 'discarded voxels (std=0):')
        pyhrf.verbose.printDict(6, np.where(zeroVarianceVoxels==True))
        pyhrf.verbose(1, '!! discarded voxels (std=0): %d/%d' \
                          %(zeroVarianceVoxels.sum(),(m!=0).sum()) )

        #HACK
        zvv = zeroVarianceVoxels
        mzvv = np.where(zeroVarianceVoxels)
        bsh = bold.shape
        if bold.ndim == 2:
            bold[:,mzvv[0]] = np.random.randn(bsh[0],zvv.sum()) * \
                var_bold.max()**.5
        else:
            bold[mzvv[0],mzvv[1],mzvv[2],:] = np.random.randn(zvv.sum(),bsh[3])*\
                var_bold.max()**.5

        zeroVarianceVoxels = np.zeros_like(zeroVarianceVoxels)


    nanVoxels = np.bitwise_and(m>=1,np.isnan(bold.sum(time_axis)))
    #print 'nanVoxels:', nanVoxels.shape
    if nanVoxels.any():
        pyhrf.verbose(6, 'discarded voxels (nan values):')
        pyhrf.verbose.printDict(6, np.where(nanVoxels==True))
        pyhrf.verbose(1, '!! discarded voxels (nan values): %d/%d' \
                          %(nanVoxels.sum(), (m!=0).sum()) )


    toDiscard = np.bitwise_or(zeroVarianceVoxels, nanVoxels)
    #print toDiscard.sum()
    #print 'np.bitwise_not(toDiscard):', (np.bitwise_not(toDiscard)).shape
    #print 'np.bitwise_not(toDiscard):', np.bitwise_not(toDiscard).sum()
    roiMask *= np.bitwise_not(toDiscard)


def has_ext_gzsafe(fn, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    return op.splitext(fn)[1] == ext or \
        ( op.splitext(fn)[1] == '.gz' and \
              op.splitext(op.splitext(fn)[0])[1]==ext)

def read_mesh(filename):
    if has_ext_gzsafe(filename, 'gii'):
        mesh_gii = gifti.read(filename)
        cor,tri = (mesh_gii.darrays[0].data, mesh_gii.darrays[1].data)
        return cor,tri,mesh_gii.darrays[0].coordsys
    else:
        raise Exception('Unsupported file format (%s)' %filename)

def write_mesh(cor, tri, filename):
    from nibabel.gifti import GiftiImage, GiftiDataArray
    nimg = GiftiImage()
    intent = 'NIFTI_INTENT_POINTSET'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(cor,intent))
    intent = 'NIFTI_INTENT_TRIANGLE'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(tri,intent))
    gifti.write(nimg, filename)


def is_texture_file(fn):
    return has_ext_gzsafe(fn, 'gii') or has_ext_gzsafe(fn, 'tex')

def is_volume_file(fn):
    return has_ext_gzsafe(fn, 'nii') or has_ext_gzsafe(fn, 'img')

def read_texture(tex):

    if has_ext_gzsafe(tex, 'gii'):
        #from gifti import loadImage
        #texture = loadImage(tex).arrays[0].data
        tex_gii = gifti.read(tex)
        if len(tex_gii.darrays) > 1: #2D texture ... #TODO: check
            texture = np.vstack([a.data for a in tex_gii.darrays])
        else:
            texture = tex_gii.darrays[0].data
        return texture, tex_gii
    elif has_ext_gzsafe(tex, 'tex'):
        from pyhrf.tools.io.tio import Texture
        texture = Texture.read(tex).data
        return texture, None
    else:
        raise NotImplementedError('Unsupported %s extension' \
                                      %op.splitext(tex)[1])

def write_texture(tex_data, filename, intent=None, meta_data=None):
    """ Write the n-dimensional numpy array 'text_data' into filename
    using gifti (.gii) or tex format.
    """
    if tex_data.dtype == np.bool:
        tex_data = tex_data.astype(np.int16)

    if has_ext_gzsafe(filename, 'gii'):
        #tex = np.arange(len(cor), dtype=int)
        if intent is None:
            if np.issubdtype(tex_data.dtype, np.int):
                intent = 'NIFTI_INTENT_LABEL'
                dtype = None
            elif np.issubdtype(tex_data.dtype, np.float):
                intent = 'NIFTI_INTENT_NONE'
                #intent = 'dimensionless' #?fg
                #dtype = 'NIFTI_TYPE_FLOAT32'
                tex_data.astype(np.float32)
                if pyhrf.cfg['global']['write_texture_minf']:
                    s = "attributes = {'data_type' : 'FLOAT'}"
                    f = open(filename + '.minf', 'w')
                    f.write(s)
                    f.close()
            else:
                raise NotImplementedError("Unsupported dtype %s" \
                                              %str(tex_data.dtype))

        gii_array = gifti.GiftiDataArray.from_array(tex_data, intent)
        if meta_data is not None:
            md = {'pyhrf_cuboid_data':meta_data}
            gmeta_data = gifti.GiftiMetaData.from_dict(md)
        else:
            gmeta_data = None
        tex_gii = gifti.GiftiImage(darrays=[gii_array,], meta=gmeta_data)
        pyhrf.verbose(3, 'Write texture to %s'%filename)
        gifti.write(tex_gii, filename)

    elif has_ext_gzsafe(filename, 'tex'):
        if meta_data is not None:
            print 'Warning: meta ignored when saving to tex format'

        from pyhrf.tools.io.tio import Texture
        tex = Texture(filename, data=tex_data)
        tex.write()
    else:
        raise NotImplementedError('Unsupported extension for file %s' \
                                      %filename)


def load_fmri_surf_data(boldFiles, meshFile, roiMaskFile=None):
    """ Load FMRI BOLD surface data from files.

    Arguments:
    boldFiles --  pathes to 2D texture data files (ie multisession data).
    roiMaskFile -- a path to a 1D texture data file.
    meshFile -- a path to a mesh file

    Return:
    A tuple (graphs, bolds, sessionScans, roiMask, dataHeader)
    graphs -- a dict {roiId:roiGraph}. Each roiGraph is a list of neighbours list
    bold -- a dict {roiId:bold}. Each bold is a 2D numpy float array
            with axes [time,position]
    sessionScans -- a list of scan indexes for every session
    roiMask -- a 1D numpy int array defining ROIs (0 stands for the background)
    dataHeader -- the header from the BOLD data file
    """

    #Load ROIs:
    pyhrf.verbose(1, 'load roi mask: ' + roiMaskFile)
    #roiMask = Texture.read(roiMaskFile).data.astype(int)
    roiMask,_ = read_texture(roiMaskFile)
    pyhrf.verbose(1, 'roi mask shape: ' + str(roiMask.shape))

    #Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    for boldFile in boldFiles:
        pyhrf.verbose(1, 'load bold: ' + boldFile)
        b,_ = read_texture(boldFile)
        pyhrf.verbose(1, 'bold shape: ' + str(b.shape))
        bolds.append(b)
        sessionScans.append(np.arange(lastScan, lastScan+b.shape[0], dtype=int))
        lastScan += b.shape[0]
    bold = np.concatenate(tuple(bolds))

    # discard bad data (bold with var=0 and nan values):
    discard_bad_data(bold, roiMask, time_axis=0)

    #Load mesh:
    pyhrf.verbose(1, 'load mesh: ' + meshFile)
    coords,triangles,coord_sys = read_mesh(meshFile)
    #from soma import aims
    # mesh = aims.read(meshFile)
    # triangles = [t.arraydata() for t in mesh.polygon().list()]

    pyhrf.verbose(3, 'building graph ... ')
    wholeGraph = graph_from_mesh(triangles)
    assert graph_is_sane(wholeGraph)
    pyhrf.verbose(1, 'Mesh has %d nodes' %len(wholeGraph))
    assert len(roiMask) == len(wholeGraph)

    assert bold.shape[1] == len(wholeGraph)

    pyhrf.verbose(3, 'Computing length of edges ... ')
    edges_l = np.array([np.array([distance(coords[i],coords[n],coord_sys.xform) \
                                      for n in nl]) \
                            for i,nl in enumerate(wholeGraph)], dtype=object)

    #Split bold and graph into rois:
    roiBold = {}
    graphs = {}
    edge_lengthes = {}
    for roiId in np.unique(roiMask):
        mroi = np.where(roiMask==roiId)
        g, nm = sub_graph(wholeGraph, mroi[0])
        edge_lengthes[roiId] = edges_l[mroi]
        graphs[roiId] = g
        roiBold[roiId] = bold[:, mroi[0]].astype(np.float32)

    return graphs, roiBold, sessionScans, roiMask, edge_lengthes


def remote_mkdir(host, user, path):
    import paramiko
    pyhrf.verbose(1, 'Make remote dir %s@%s:%s ...' %(user,host,path))
    ssh = paramiko.SSHClient()
    known_hosts_file = os.path.join("~", ".ssh", "known_hosts")
    ssh.load_host_keys(os.path.expanduser(known_hosts_file))
    ssh.connect(host, username=user)
    sftp = ssh.open_sftp()
    sftp.mkdir(path)


def rexists(host, user, path):
    """os.path.exists for paramiko's SCP object
    """
    import paramiko
    pyhrf.verbose(1, 'Make remote dir %s@%s:%s ...' %(user,host,path))
    ssh = paramiko.SSHClient()
    known_hosts_file = os.path.join("~", ".ssh", "known_hosts")
    ssh.load_host_keys(os.path.expanduser(known_hosts_file))
    ssh.connect(host, username=user)
    sftp = ssh.open_sftp()
    try:
        sftp.stat(path)
    except IOError, e:
        if e[0] == 2:
            return False
    else:
        return True

def remote_copy(files, host, user, path, transfer_tool='ssh'):
    if transfer_tool == 'paramiko':
        import paramiko
        pyhrf.verbose(1, 'Copying files to remote destination %s@%s:%s ...' \
                          %(host,user,path))
        ssh = paramiko.SSHClient()
        known_hosts_file = os.path.join("~", ".ssh", "known_hosts")
        ssh.load_host_keys(os.path.expanduser(known_hosts_file))
        ssh.connect(host, username=user)
        sftp = ssh.open_sftp()
        for f in files:
            remotepath = op.join(path,op.basename(f))
            pyhrf.verbose(2, f + ' -> ' + remotepath + ' ...')
            flocal = open(f)
            remote_file = sftp.file(remotepath, "wb")
            remote_file.set_pipelined(True)
            remote_file.write(flocal.read())
            flocal.close()
            remote_file.close()
        sftp.close()
        ssh.close()
    else:
        sfiles = string.join(['"%s"'%f for f in files], ' ')

        scp_cmd = 'scp -C %s "%s@%s:%s"' %(sfiles, user, host, path)
        pyhrf.verbose(1, 'Data files transfer with scp ...')
        pyhrf.verbose(2, scp_cmd)
        if os.system(scp_cmd) != 0:
            raise Exception('Error while scp ...')

    pyhrf.verbose(1, 'Copy done!')

    return [op.join(path,op.basename(f)) for f in files]

from nibabel.nifti1 import Nifti1Extension, extension_codes
from pyhrf.tools import now

json_is_available = True
try:
    import json
except ImportError:
    json_is_available = False


if json_is_available:
    def append_process_info(img_fn, proc_name, proc_inputs, proc_date=None,
                            proc_version=None, proc_id=None, img_output_fn=None):

        i,(aff,header) = read_volume(img_fn)
        ecodes = header.extensions.get_codes()
        if extension_codes['workflow_fwds'] in ecodes:
            ic = ecodes.index(extension_codes['workflow_fwds'])
            econtent = header.extensions[ic].get_content()
            try:
                proc_info = json.loads(econtent)
            except:
                raise IOError("Cannot safely overwrite Extension in "\
                                  "Header. It already has a readable "\
                                  "'workflow_fwds' extension, but it's "\
                                  "not in JSON." )
        else:
            proc_info = []

        if proc_date is None:
            proc_date = str(now())

        proc_info.append({'process_name':proc_name,
                          'process_inputs':proc_inputs,
                          'process_id':proc_id,
                          'process_version':proc_version,
                          'process_date':proc_date})

        e = Nifti1Extension('workflow_fwds', json.dumps(proc_info))
        header.extensions.append(e)

        if img_output_fn is None:
            img_output_fn = img_fn

        write_volume(i, img_output_fn, (aff, header))


    def get_process_info(img_fn):

        proc_info = None
        i,(aff,header) = read_volume(img_fn)
        ecodes = header.extensions.get_codes()
        if extension_codes['workflow_fwds'] in ecodes:
            ic = ecodes.index(extension_codes['workflow_fwds'])
            econtent = header.extensions[ic].get_content()
            try:
                proc_info = json.loads(econtent)
            except:
                pass

        return proc_info

from pyhrf.tools import crop_array

def flip_data_file_in_parcels(data_fn, mask_fn, parcels, output_data_fn):

    data,h = read_volume(data_fn)
    mask,mh = read_volume(mask_fn)

    if data.ndim > mask.ndim:
        mask = mask[:,:,:,np.newaxis] * np.ones(data.shape, dtype=int)
    elif mask.ndim > data.ndim:
        raise Exception('Mask has more dims (%d) than data (%d).' \
                        'Maybe data and mask were switched?' \
                        %(mask.ndim, data.ndim))

    for p_id in parcels:
        data[np.where(mask==p_id)] *= -1

    print 'Saved flipped data to:', output_data_fn
    write_volume(data, output_data_fn, h)

def remove_parcels_in_file(mask_fn, parcels, output_data_fn):

    mask,mh = read_volume(mask_fn)


    for p_id in parcels:
        mask[np.where(mask==p_id)] = 0

    print 'Saved flipped data to:', output_data_fn
    write_volume(mask, output_data_fn, mh)



def crop_data_file(data_fn, mask_fn, output_data_fn):

    data,h = read_volume(data_fn)
    mask,mh = read_volume(mask_fn)

    if data.ndim > mask.ndim:
        mask = mask[:,:,:,np.newaxis] * np.ones(data.shape, dtype=int)
    elif mask.ndim > data.ndim:
        raise Exception('Mask has more dims (%d) than data (%d).' \
                        'Maybe data and mask were switched?' \
                        %(mask.ndim, data.ndim))

    cropped_data = crop_array(data, mask)

    if 1:
        print 'cropped_data:', cropped_data.shape
        print 'input_data_fn:', data_fn
        print 'output_data_fn:', output_data_fn

    if len(h[1].extensions) > 0:
        #remove cuboid nifti extension
        h[1].extensions.pop()

    write_volume(cropped_data, output_data_fn, h)

