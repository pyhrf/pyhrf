# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import os, sys
import os.path as op
import string
import cPickle
import tempfile

import numpy as _np
from pkg_resources import Requirement, resource_filename, resource_listdir
import pyhrf
from pyhrf.ndarray import MRI3Daxes, MRI4Daxes, expand_array_in_mask, TIME_AXIS
from nipy.labs import compute_mask_files
from pyhrf.tools import stack_trees, distance
from pyhrf.tools.io import read_volume, discard_bad_data, write_volume, \
    read_mesh, read_texture, write_texture
from pyhrf.graph import parcels_to_graphs, kerMask3D_6n, \
    graph_from_mesh, graph_is_sane, sub_graph

from pyhrf.ndarray import xndarray

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

#__all__ = ['FmriData', 'FmriRoiData', 'AttrClass', 'Condition',
           #'get_data_file_name',
           #'list_data_file_names', 'onsets_loc_av_fonc1', 'onsets_loc_av_fonc2', 'onsets_loc',
           #'durations_loc_av_fonc1', 'durations_loc_av_fonc1', 'durations_loc',]

# __all__ = ['FmriData', 'get_data_file_name',
#            'list_data_file_names', 'onsets_loc_av', 'onsets_loc',
#            'durations_loc_av', 'durations_loc']

def _pickle_method(method):
    """
    Allow to safely pickle classmethods. To be fed to copy_reg.pickle
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    """
    Allow to safely unpickle classmethods. To be fed to copy_reg.pickle
    """
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None # if func_name is classmethod
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType,
                _pickle_method,
                _unpickle_method)


def get_data_file_name(fn):
    req = Requirement.parse('pyhrf')
    pyhrfDataSubPath = 'pyhrf/datafiles'
    fn2 = os.path.join(pyhrfDataSubPath, fn)
    return resource_filename(req, fn2)

def list_data_file_names():
    req = Requirement.parse('pyhrf')
    pyhrfDataSubPath = 'pyhrf/datafiles'
    return sorted(filter(lambda x: '.svn' not in x,
                         resource_listdir(req, pyhrfDataSubPath)))

def get_tmp_path(tag='pyhrf_'):
    tmp_dir = tempfile.mkdtemp(prefix=tag, dir=pyhrf.cfg['global']['tmp_path'])
    return tmp_dir

def get_src_path():
    import pyhrf
    return os.path.join(os.path.dirname(pyhrf.__file__),'../../')

def get_src_doc_path():
    return op.join(get_src_path(), 'doc/sphinx/source')

class AttrClass:
    def __init__(self, **kwargs):
        for k,w in kwargs.iteritems():
            setattr(self,k,w)

    def __repr__(self):
        r = self.__class__.__name__ + '('
        a = [k+'='+repr(getattr(self,k)) for k in dir(self) \
                 if not k.startswith('_')]
        r += ','.join(a) + ')'

        return r

class Condition(AttrClass): pass

from pyhrf.tools import PickleableStaticMethod

## PARADIGM STUFFS ##
from paradigm import *

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

DEFAULT_SIMULATION_FILE = get_data_file_name('simu.pck')

## Default localizer onsets ##
# session-specific onsets here, so onset arrays are not encapsulated
# in lists over all sessions
DEFAULT_ONSETS = OrderedDict(
    [('audio' , _np.array([ 15.,20.7,29.7,35.4,44.7,48.,83.4,89.7,108.,
                            119.4, 135., 137.7, 146.7, 173.7, 191.7, 236.7,
                            251.7, 284.4, 293.4, 296.7])),
     ('video' , _np.array([ 0., 2.4, 8.7,33.,39.,41.7, 56.4, 59.7, 75., 96.,
                            122.7, 125.4, 131.4, 140.4, 149.4, 153., 156., 159.,
                           164.4, 167.7, 176.7, 188.4, 195., 198., 201., 203.7,
                            207., 210., 218.7, 221.4, 224.7, 234., 246., 248.4,
                            260.4, 264., 266.7, 269.7, 278.4, 288. ]))
     ]
    )

DEFAULT_STIM_DURATIONS = OrderedDict(
    [('audio', _np.array([])),
     ('video', _np.array([]))
     ]
    )

fn = 'paradigm_loc_av.csv'
DEFAULT_PARADIGM_CSV = get_data_file_name(fn)

## Default Volumic data ##
dataFn = 'subj0_bold_session0.nii.gz'
maskFn = 'subj0_parcellation.nii.gz'
DEFAULT_BOLD_VOL_FILE = get_data_file_name(dataFn)
DEFAULT_BOLD_VOL_TR = 2.4
DEFAULT_MASK_VOL_FILE = get_data_file_name(maskFn)
maskFn = 'subj0_parcellation_small.nii.gz'
DEFAULT_MASK_SMALL_VOL_FILE = get_data_file_name(maskFn)
DEFAULT_OUT_MASK_VOL_FILE = './roi_mask.nii'

maskFn = 'real_data_vol_4_regions_mask.nii.gz'
REALISTIC_REAL_DATA_MASK_VOL_FILE = get_data_file_name(maskFn)
boldFn = 'real_data_vol_4_regions_BOLD.nii.gz'
REALISTIC_REAL_DATA_BOLD_VOL_FILE = get_data_file_name(boldFn)

## Default Surfacic data ##
dataFn = 'real_data_surf_tiny_bold.gii'
maskFn = 'real_data_surf_tiny_parcellation.gii'
meshFn = 'real_data_surf_tiny_mesh.gii'
DEFAULT_BOLD_SURF_FILE = get_data_file_name(dataFn)
DEFAULT_BOLD_SURF_TR = 2.4
DEFAULT_MASK_SURF_FILE = get_data_file_name(maskFn)
DEFAULT_MESH_FILE = get_data_file_name(meshFn)
DEFAULT_OUT_MASK_SURF_FILE = './roiMask.nii'

from pyhrf.xmlio import XmlInitable


class FMRISessionVolumicData(XmlInitable):

    parametersComments = {
        'onsets' : 'Onsets of experimental simtuli in seconds. \n'\
            'Dictionnary mapping stimulus name to '\
            'the actual list of onsets.',
        'durations' : 'Durations of experimental simtuli in seconds.\n'\
            'It has to consistent with the definition of onsets',
        'bold_file' : 'Data file containing the 3D+time BOLD signal (nifti format)'
            }

    def __init__(self, onsets=DEFAULT_ONSETS,
                 durations=DEFAULT_STIM_DURATIONS,
                 bold_file=DEFAULT_BOLD_VOL_FILE):

        XmlInitable.__init__(self)
        assert isinstance(onsets, (dict, OrderedDict))
        assert durations is None or isinstance(durations, (dict, OrderedDict))
        #assert bold_file.endswith('nii') or bold_file.endswith('nii.gz')

        self.onsets = onsets
        self.durations = durations
        self.bold_file = bold_file
        #print 'bold file:', bold_file
        #print 'DEFAULT_BOLD_VOL_FILE:', DEFAULT_BOLD_VOL_FILE

    def to_dict(self):
        return {'onsets' : self.onsets,
                'durations' : self.durations,
                'bold_file' : self.bold_file}


class FMRISessionSurfacicData(XmlInitable):
    def __init__(self, onsets=DEFAULT_ONSETS,
                 durations=DEFAULT_STIM_DURATIONS,
                 bold_file=DEFAULT_BOLD_SURF_FILE):

        XmlInitable.__init__(self)
        assert isinstance(onsets, dict)
        assert durations is None or isinstance(durations, dict)
        #assert bold_file.endswith('gii') or bold_file.endswith('gii.gz')

        self.onsets = onsets
        self.durations = durations
        self.bold_file = bold_file

    def to_dict(self):
        return {'onsets' : self.onsets,
                'durations' : self.durations,
                'bold_file' : self.bold_file}


class FMRISessionSimulationData(XmlInitable):
    def __init__(self, onsets=DEFAULT_ONSETS,
                 durations=DEFAULT_STIM_DURATIONS,
                 simulation_file=DEFAULT_SIMULATION_FILE):

        XmlInitable.__init__(self)
        assert isinstance(onsets, dict)
        assert durations is None or isinstance(durations, dict)

        self.onsets = onsets
        self.durations = durations
        self.simulation_file = simulation_file

    def to_dict(self):
        return {'onsets' : self.onsets,
                'durations' : self.durations,
                'bold_file' : self.bold_file}


def load_vol_bold_and_mask(bold_files, mask_file):

    # Handle mask
    if not op.exists(mask_file):
        pyhrf.verbose(1,'Mask file %s does not exist. Mask is '\
                          ' computed from BOLD ...' %mask_file)
        bf = 'subj0_bold_session0.nii.gz'
        # HACK
        if bold_files[0] == pyhrf.get_data_file_name(bf):
            max_frac = .99999 #be sure to keep non zero voxels
            cc = 0 # default BOLD vol has 2 ROIs
        else:
            max_frac = .9
            cc = 1
        compute_mask_files(bold_files[0], mask_file, False, .4,
                           max_frac, cc=cc)
        mask_loaded_from_file = False
    else:
        mask_loaded_from_file = True
        pyhrf.verbose(1,'Assuming orientation for mask file: ' + \
                          string.join(MRI3Daxes, ','))

    pyhrf.verbose(2,'Read mask from: %s' %mask_file)
    mask, mask_meta_obj = read_volume(mask_file)

    if not np.allclose(np.round(mask),mask):
        raise Exception("Mask is not n-ary (%s)" %mask_file)
    mask = mask.astype(np.int32)

    pyhrf.verbose(1,'Mask has shape %s' %str(mask.shape))
    pyhrf.verbose(1,'Mask min value: %d' %mask.min())
    pyhrf.verbose(1,'Mask max value: %d' %mask.max())
    mshape = mask.shape
    if mask.min() == -1:
        mask += 1

    #Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    pyhrf.verbose(1,'Assuming orientation for BOLD files: ' + \
                      string.join(MRI4Daxes, ','))

    #print 'type of bold_files[0]:', type(bold_files[0])
    #print 'bold_files:', bold_files
    if type(bold_files[0]) is list:
        bold_files = bold_files[0]

    for bold_file in bold_files:
        if not op.exists(bold_file):
            raise Exception('File not found: ' + bold_file)

        b, bold_meta = read_volume(bold_file)
        bolds.append(b)
        sessionScans.append(np.arange(lastScan,
                                      lastScan+b.shape[TIME_AXIS],
                                      dtype=int))
        lastScan += b.shape[TIME_AXIS]

    bold = np.concatenate(tuple(bolds), axis=TIME_AXIS)

    pyhrf.verbose(1,'BOLD has shape %s' %str(bold.shape))
    discard_bad_data(bold, mask)

    # #HACK
    # if mask_file != DEFAULT_MASK_VOL_FILE:
    #     write_volume(mask,'./treated_mask.nii')
    #     sys.exit(0)


    return mask, mask_meta_obj, mask_loaded_from_file, bold, sessionScans


def load_surf_bold_mask(bold_files, mesh_file, mask_file=None):

    pyhrf.verbose(1, 'Load mesh: ' + mesh_file)
    coords,triangles,coord_sys = read_mesh(mesh_file)
    pyhrf.verbose(2, 'Build graph ... ')
    fgraph = graph_from_mesh(triangles)
    assert graph_is_sane(fgraph)

    pyhrf.verbose(1, 'Mesh has %d nodes' %len(fgraph))

    pyhrf.verbose(2, 'Compute length of edges ... ')
    edges_l = np.array([np.array([distance(coords[i],
                                           coords[n],
                                           coord_sys.xform) \
                                      for n in nl]) \
                            for i,nl in enumerate(fgraph)],dtype=object)

    if mask_file is None or not op.exists(mask_file):
        pyhrf.verbose(1,'Mask file %s does not exist. Taking '\
                          ' all nodes ...' %mask_file)
        mask = np.ones(len(fgraph))
        mask_meta_obj = None
        mask_loaded_from_file = False
    else:
        mask, mask_meta_obj = read_texture(mask_file)
        mask_loaded_from_file = True

    if not (np.round(mask) == mask).all():
        raise Exception("Mask is not n-ary")


    if len(mask) != len(fgraph):
        raise Exception('Size of mask (%d) is different from size '\
                        'of graph (%d)' %(len(mask),len(fgraph)))

    mask = mask.astype(np.int32)
    if mask.min() == -1:
        mask += 1

    #Split graph into rois:
    graphs = {}
    edge_lengths = {}
    for roiId in np.unique(mask):
        mroi = np.where(mask==roiId)
        g, nm = sub_graph(fgraph, mroi[0])
        edge_lengths[roiId] = edges_l[mroi]
        graphs[roiId] = g


    #Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    for boldFile in bold_files:
        pyhrf.verbose(1, 'load bold: ' + boldFile)
        b,_ = read_texture(boldFile)
        pyhrf.verbose(1, 'bold shape: ' + str(b.shape))
        bolds.append(b)
        sessionScans.append(np.arange(lastScan, lastScan+b.shape[0],
                                      dtype=int))
        lastScan += b.shape[0]

    bold = np.concatenate(tuple(bolds))
    if len(fgraph) != bold.shape[1]:
        raise Exception('Nb positions not consistent between BOLD (%d) '\
                        'and mesh (%d)' %(bold.shape[1],len(fgraph)))

    # discard bad data (bold with var=0 and nan values):
    discard_bad_data(bold, mask, time_axis=0)

    return mask, mask_meta_obj, mask_loaded_from_file, bold, sessionScans, \
        graphs, edge_lengths


def merge_fmri_sessions(fmri_data_sets):
    """

    fmri_data_sets: list of FmriData objects.
    Each FmriData object is assumed to contain only one session
    """

    all_onsets = stack_trees([fd.paradigm.stimOnsets for fd in fmri_data_sets])
    all_onsets = apply_to_leaves(all_onsets, lambda l: [e[0] for e in l])
    durations = stack_trees([fd.paradigm.stimDurations for fd in fmri_data_sets])
    durations = apply_to_leaves(durations, lambda x: [e[0] for e in x])

    bold = np.concatenate([fd.bold for fd in fmri_data_sets])

    ss = []
    lastScan = 0
    for fd in fmri_data_sets:
        nscans = fd.bold.shape[0]
        ss.append(np.arange(lastScan,lastScan+nscans,dtype=int))
        lastScan += nscans

    if fd.simulation is not None:
        simu = [fd.simulation[0] for fd in fmri_data_sets]
    else:
        simu=None

    return FmriData(all_onsets, bold, fmri_data_sets[0].tr, ss,
                    fmri_data_sets[0].roiMask, fmri_data_sets[0].graphs,
                    durations, fmri_data_sets[0].meta_obj, simu,
                    backgroundLabel=fmri_data_sets[0].backgroundLabel,
                    data_files=fmri_data_sets[0].data_files,
                    data_type=fmri_data_sets[0].data_type,
                    edge_lengths=fmri_data_sets[0].edge_lengths,
                    mask_loaded_from_file=fmri_data_sets[0].mask_loaded_from_file)


class Object(object):
    pass

def merge_fmri_subjects(fmri_data_sets, roiMask, backgroundLabel=0):
    """

    fmri_data_sets: list of FmriData objects, for different subjects.
    In case of multisession data, merging of fmri data over sessions must
    be done for each subject before using this function.
    roiMask: multi_subject parcellation (nparray)
    """

    Fmri_subjects = Object()

    #simple list containing information for each subject
    # if backgroundLabel is None:
    #         roi_ids = np.unique(roiMask.flat)
    #         if len(roi_ids) == 1:
    #             backgroundLabel = 0
    #         else:
    #             backgroundLabel = roiMask.min()
    # else:
    #     backgroundLabel = backgroundLabel

    np_roi_mask = np.where(roiMask != backgroundLabel)
    roi_ids_in_mask = roiMask[np_roi_mask]
    #nb_voxels_in_mask = len(roi_ids_in_mask)
    #spatial_shape = roiMask.shape
    unique_roi_ids = np.unique(roi_ids_in_mask)

    Fmri_subjects.subjects_data = np.append([f for f in fmri_data_sets])
    Fmri_subjects.roiMask = roiMask
    Fmri_subjects.roiId = unique_roi_ids

    return Fmri_subjects



class FmriGroupData(XmlInitable):
    '''
    Used for group level hemodynamic analysis
    Encapsulates FmriData objects for all subjects
    All subjects must habe the same number of ROIs

    Inputs:
        list_subjects: contains list of FmriData object for each subject

    '''

    def __init__(self, list_subjects):
        self.data_subjects = list_subjects
        self.nbSubj = len(self.data_subjects)

        self.spatial_shape = list_subjects[0].spatial_shape
        self.meta_obj = list_subjects[0].meta_obj

    def getSummary(self, long=False):
        s = 'Number of subjects: %d\n' %self.nbSubj
        return s + '\n'.join([ds.getSummary(long=long) \
                              for ds in self.data_subjects])

    def roi_split(self):
        '''
        Retrieve a list of FmriGroupData object,
        each containing the data for all subject, in one ROI
        '''
        rfd = [ds.roi_split() for ds in self.data_subjects]
        nbROIs = len(rfd[0])
        Rall=[]
        for rid in xrange(nbROIs):
            r=[]
            for s in xrange(self.nbSubj):
                r.append(rfd[s][rid])
            Rall.append(FmriGroupData(r))
        return Rall

    def build_graphs(self, force=False):
        return [sd.build_graphs(force=force) for sd in self.data_subjects]

    def get_roi_id(self):
        return self.data_subjects[0].get_roi_id()
        #each FmriData object in self.data_subjetcs contains data for
        #all ROIs, for one subject


def get_roi_simulation(simu_sessions, mask, roi_id):
    """
    Extract the ROI from the given simulation dict.
    Args:
        - simu (dict): dictionnary of simulated quantities
        - mask (np.ndarray): binary mask defining the spatial extent of the ROI
        - roi_id (int) : the id of the roi to extract
    Return:
         dict of roi-specific simulation items
    """
    if simu_sessions is None:
        return None

    simus = []
    for simu in simu_sessions:
        roi_simu = {}
        m = np.where(mask==roi_id)
        # for now, just extract nrls, labels, hrf, prf,
        # prls, brls, brf if available
        roi_simu['labels'] = simu['labels'][:, m[0]]
        if simu.has_key('hrf'):
            roi_simu['hrf'] = simu['hrf'][:, m[0]]
        if simu.has_key('brf'):
            roi_simu['brf'] = simu['brf'][:, m[0]]
        if simu.has_key('prf'):
            roi_simu['prf'] = simu['prf'][:, m[0]]
        if simu.has_key('nrls'):
            roi_simu['nrls'] = simu['nrls'][:, m[0]]
        if simu.has_key('brls'):
            roi_simu['brls'] = simu['brls'][:, m[0]]
        if simu.has_key('prls'):
            roi_simu['prls'] = simu['prls'][:, m[0]]
        if simu.has_key('noise'):
            roi_simu['noise'] = simu['noise'][:, m[0]]
        if simu.has_key('drift'):
            roi_simu['drift'] = simu['drift'][:, m[0]]
        if simu.has_key('bold'):
                roi_simu['bold'] = simu['bold'][:, m[0]]
        if simu.has_key('stim_induced_signal'):
                roi_simu['stim_induced_signal'] = simu['stim_induced_signal'][:, m[0]]

        if simu.has_key('primary_hrf'):
            #TOCHECK: not ROI-specific?
            roi_simu['primary_hrf'] = simu['primary_hrf']
        if simu.has_key('var_subject_hrf'):
            roi_simu['var_subject_hrf'] = simu['var_subject_hrf']

        if simu.has_key('var_hrf_group'):
            roi_simu['var_hrf_group'] = simu['var_hrf_group']

        roi_simu['condition_defs'] = simu['condition_defs']
        roi_simu['tr'] = simu['tr']
        roi_simu['dt'] = simu['dt']


        simus.append(roi_simu)
    return simus

class FmriData(XmlInitable):
    """
    Attributes:
    onsets -- a dictionary mapping a stimulus name to a list of session onsets.
              Each item of this list is a 1D numpy float array of onsets
              for a given session.
    stimDurations -- same as 'onsets' but stores durations of stimuli
    roiMask -- numpy int array of roi labels (0 stands for the background).
               shape depends on the data form (3D volumic or 1D surfacic)
    bold -- either a 4D numpy float array with axes [sag,cor,ax,scan] and then
            spatial axes must have the same shape as roiMask,
            Or a 2D numpy float array with axes [scan, position] and position
            axis must have the same length as the number of positions within
            roiMask (without background).
            Sessions are stacked in the scan axis
    sessionsScans -- a list of session indexes along scan axis.
    tr -- Time of repetition of the BOLD signal
    simulation -- if not None then it should be a list of simulation instance.
    meta_obj -- extra information associated to data
    """

    parametersComments = {
        'tr' : 'repetition time in seconds',
        'sessions_data' : 'List of data definition for all sessions',
        'mask_file' : 'Input n-ary mask file (= parcellation). '\
                      'Only positive integers are allowed. \n'
                      'All zeros are treated as background positions.'
        }

    parametersToShow = ['tr', 'sessions_data', 'mask_file']

    if pyhrf.__usemode__ == 'devel':
        parametersToShow += ['background_label']

    def __init__(self, onsets, bold, tr, sessionsScans, roiMask, graphs=None,
                 stimDurations=None, meta_obj=None, simulation=None,
                 backgroundLabel=0, data_files=None, data_type=None,
                 edge_lengths=None, mask_loaded_from_file=False):

        #pyhrf.verbose.set_verbosity(5)
        pyhrf.verbose(3, 'Creation of FmriData object ...')
        if isinstance(bold, xndarray):
            pyhrf.verbose(3, 'bold shape: %s' %str(bold.data.shape))
        else:
            pyhrf.verbose(3, 'bold shape: %s' %str(bold.shape))
        pyhrf.verbose(3, 'roi mask shape: %s' %str(roiMask.shape))
        pyhrf.verbose(3, 'unique(mask): %s' %str(np.unique(roiMask)))

        sessionsDurations = [len(ss)*tr for ss in sessionsScans]
        #print 'onsets!:', onsets
        self.paradigm = Paradigm(onsets, sessionsDurations, stimDurations)

        self.tr = tr
        self.sessionsScans = sessionsScans

        # if backgroundLabel is None:
        #     roi_ids = np.unique(roiMask.flat)
        #     if len(roi_ids) == 1 or data_type == 'surface':
        #         self.backgroundLabel = 0
        #         #dummy backgroundLabel:
        #         #self.backgroundLabel = max(roi_ids) + 1
        #         #print 'dummy backgroundLabel !!!'
        #     else:
        #         if simulation is not None:
        #             self.backgroundLabel = 0
        #         else:
        #             self.backgroundLabel = roiMask.min()
        # else:
        #     self.backgroundLabel = backgroundLabel
        if backgroundLabel is None:
            backgroundLabel = 0

        self.backgroundLabel = backgroundLabel


        pyhrf.verbose(3, 'backgroundLabel: %d' %self.backgroundLabel)
        #print 'test'

        self.store_mask_sparse(roiMask)

        m = self.np_roi_mask
        if isinstance(bold, xndarray):
            if bold.data.ndim > 2: #volumic data
                print '----------------'
                if data_type is None: data_type = 'volume'
                assert roiMask.ndim == 3
                if TIME_AXIS == 0:
                    bold = bold.data[:, m[0], m[1], m[2]]
                else: # TIME_AXIS = 3
                    bold = bold.data[m[0], m[1], m[2], :].transpose()

            else: # surfacic or already flatten data
                assert bold.data.ndim == 2
                # print 'len(m[0]):', len(m[0])
                # print 'bold.shape[1]:', bold.shape[1]

                #assert len(m[0]) == bold.shape[1]
                if len(m[0]) != bold.data.shape[1]:
                    bold = bold.data[:, m[0]]
                else:
                    bold = bold.data
        else:
            if bold.ndim > 2: #volumic data
                if data_type is None: data_type = 'volume'
                assert roiMask.ndim == 3
                if TIME_AXIS == 0:
                    bold = bold[:, m[0], m[1], m[2]]
                else: # TIME_AXIS = 3
                    bold = bold[m[0], m[1], m[2], :].transpose()

            else: # surfacic or already flatten data
                assert bold.ndim == 2
                # print 'len(m[0]):', len(m[0])
                # print 'bold.shape[1]:', bold.shape[1]

                #assert len(m[0]) == bold.shape[1]
                if len(m[0]) != bold.shape[1]:
                    bold = bold[:, m[0]]

        self.data_type = data_type

        pyhrf.verbose(3, 'extracted bold shape (without background): %s' \
                          %str(bold.shape))

        #self.roiMask = roiMask
        #self.roiMask_full = roiMask

        self.bold = bold
        self.bold_full = bold
        self.bold_avg = None

        self.graphs = graphs
        self.edge_lengths = edge_lengths
        self.graphs_full = graphs
        self.graphs_avg = None
        self.meta_obj = meta_obj
        self.simulation = simulation

        self.nbConditions = len(self.paradigm.stimOnsets)
        self.nbSessions = len(sessionsScans)

        self.mask_loaded_from_file = mask_loaded_from_file
        if data_files is None: data_files = []
        self.data_files = data_files


    def get_condition_names(self):
        return self.paradigm.stimOnsets.keys()

    def store_mask_sparse(self, roiMask):

        self.np_roi_mask = np.where(roiMask != self.backgroundLabel)
        self.roi_ids_in_mask = roiMask[self.np_roi_mask]
        self.nb_voxels_in_mask = len(self.roi_ids_in_mask)
        self.spatial_shape = roiMask.shape

    def get_roi_mask(self):
        roi_mask = np.zeros(self.spatial_shape,
                            dtype=self.roi_ids_in_mask.dtype) + \
                       self.backgroundLabel
        roi_mask[self.np_roi_mask] = self.roi_ids_in_mask
        return roi_mask
    roiMask = property(get_roi_mask)

    # def __getstate__(self):
    #     return dict((k, v) for (k, v) in self.__dict__.iteritems() \
    #                     if k != 'roiMask')

    def get_nb_rois(self):
        """ Return the number of parcels (background id is discarded) """
        s = set(np.unique(self.roi_ids_in_mask)).discard(0)
        return len(s)

    @PickleableStaticMethod
    def from_vol_files(self, mask_file=DEFAULT_MASK_VOL_FILE,
                       paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                       bold_files=[DEFAULT_BOLD_VOL_FILE],
                       tr=DEFAULT_BOLD_VOL_TR,
                       background_label=None):
        paradigm = Paradigm.from_csv(paradigm_csv_file)
        durations = paradigm.stimDurations
        onsets = paradigm.stimOnsets

        m, mmo, mlf, b, ss = load_vol_bold_and_mask(bold_files, mask_file)
        mask = m
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        bold = b
        sessionScans = ss

        fd = FmriData(onsets, bold, tr, sessionScans, mask,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files+[mask_file,paradigm_csv_file],
                      data_type='volume',
                      mask_loaded_from_file=mask_loaded_from_file,
                      backgroundLabel=background_label)
        fd.set_init(FmriData.from_vol_files, mask_file=mask_file,
                    paradigm_csv_file=paradigm_csv_file,
                    bold_files=bold_files, tr=tr,
                    background_label=background_label)
        return fd

    @PickleableStaticMethod
    def from_vol_files_rel(self, mask_file, paradigm_csv_file, bold_files,
                           tr, rel_conditions):
        paradigm = Paradigm.from_csv(paradigm_csv_file)
        durations = OrderedDict()
        onsets = OrderedDict()
        for i in xrange(len(rel_conditions)):
            durations[rel_conditions[i]] = paradigm.stimDurations[rel_conditions[i]]
            onsets[rel_conditions[i]] = paradigm.stimOnsets[rel_conditions[i]]
        m, mmo, mlf, b, ss = load_vol_bold_and_mask(bold_files, mask_file)
        mask = m
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        bold = b
        sessionScans = ss

        fd = FmriData(onsets, bold, tr, sessionScans, mask,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files+[mask_file,paradigm_csv_file],
                      data_type='volume',
                      mask_loaded_from_file=mask_loaded_from_file)
        fd.set_init(FmriData.from_vol_files, mask_file=mask_file,
                    paradigm_csv_file=paradigm_csv_file,
                    bold_files=bold_files, tr=tr)
        return fd

    @PickleableStaticMethod
    def from_vol_ui(self, sessions_data=[FMRISessionVolumicData()],
                    tr=DEFAULT_BOLD_VOL_TR,
                    mask_file=DEFAULT_MASK_VOL_FILE, background_label=None):
        """
        Convenient creation function intended to be used for XML I/O.
        'session_data' is a list of FMRISessionVolumicData objects.
        'tr' is the repetition time.
        'mask_file' is a path to a functional mask file.

        This represents the following hierarchy:
           - FMRIData:
              - list of session data:
                  [ * data for session 1:
                           - onsets for session 1,
                           - durations for session 1,
                           - fmri data file for session 1 (nii)
                    * data for session 2:
                           - onsets for session 2,
                           - durations for session 2,
                           - fmri data file for session 2 (nii)
                  ],
              - repetition time
              - mask file
        """
        pyhrf.verbose(1,'Load volumic data ...')
        pyhrf.verbose(3,'Input sessions data:')
        pyhrf.verbose.printDict(3,sessions_data)
        #print 'sessions_data:', sessions_data
        #print 'Session Data:', [sd.to_dict() for sd in sessions_data]
        # for sd in sessions_data:
        #     print 'sd:', sd.to_dict()
        sd = stack_trees([sd.to_dict() for sd in sessions_data])
        #print 'sd dict:', sd
        onsets = sd['onsets']
        #print '~~~ onsets:', onsets
        durations = sd['durations']
        #print '~~~ durations:', durations
        bold_files = sd['bold_file']
        #print 'onsets!:', onsets.values()[0]
        #print type(onsets.values()[0][0])==list
        #print onsets[onsets.keys()[0]][0]
        #print 'durations!', durations

        #HACKS!!
        if type(onsets.values()[0][0])==list:
            for i in xrange(len(onsets.keys())):
                onsets[onsets.keys()[i]] = onsets[onsets.keys()[i]][0]

        if type(durations.values()[0][0])==list:
            for i in xrange(len(durations.keys())):
                durations[durations.keys()[i]] = durations[durations.keys()[i]][0]

        if len(durations.values()[0][0].shape)>1:
            for i in xrange(len(durations.keys())):
                durations.values()[i][0] = durations.values()[i][0][0]

        if len(onsets.keys())==1:
            onsets[onsets.keys()[0]] = onsets[onsets.keys()[0]][0]
            durations[onsets.keys()[0]] = durations[onsets.keys()[0]][0]
        if isinstance(durations,list) and durations[0] is None:
            durations = None


        m, mmo, mlf, b, ss = load_vol_bold_and_mask(bold_files, mask_file)
        mask = m
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        bold = b
        sessionScans = ss
        #print onsets, durations
        fd = FmriData(onsets, bold, tr, sessionScans, mask,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files+[mask_file],
                      backgroundLabel=background_label,
                      data_type='volume',
                      mask_loaded_from_file=mask_loaded_from_file)

        fd.set_init(FmriData.from_vol_ui, sessions_data=sessions_data,
                    tr=tr, mask_file=mask_file, background_label=background_label)
        return fd


    @classmethod
    def from_surf_files(self, paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                        bold_files=[DEFAULT_BOLD_SURF_FILE],
                        tr=DEFAULT_BOLD_SURF_TR, mesh_file=DEFAULT_MESH_FILE,
                        mask_file=None):

        paradigm = Paradigm.from_csv(paradigm_csv_file)
        durations = paradigm.stimDurations
        onsets = paradigm.stimOnsets

        m, mmo, mlf, b, ss, g, el = load_surf_bold_mask(bold_files, mesh_file,
                                                        mask_file)
        mask = m
        mask_meta_obj = mmo
        bold = b
        sessionScans = ss
        graphs = g
        edge_lengths = el

        fd = FmriData(onsets, bold, tr, sessionScans, mask, graphs,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files+[mask_file,mesh_file],
                      data_type='surface', edge_lengths=edge_lengths)
        fd.set_init(FmriData.from_surf_files, paradigm_csv_file=paradigm_csv_file,
                    bold_files=bold_files, tr=tr, mesh_file=mesh_file,
                    mask_file=mask_file)
        return fd


    @classmethod
    def from_surf_ui(self, sessions_data=[FMRISessionSurfacicData()],
                     tr=DEFAULT_BOLD_SURF_TR,
                     mask_file=DEFAULT_MASK_SURF_FILE,
                     mesh_file=DEFAULT_MESH_FILE):
        """
        Convenient creation function intended to be used for XML I/O.
        'session_data' is a list of FMRISessionVolumicData objects.
        'tr' is the time of repetition.
        'mask_file' is a path to a functional mask file.

        This represents the following hierarchy:
           - FMRIData:
              - list of session data:
                  [ * data for session 1:
                           - onsets for session 1,
                           - durations for session 1,
                           - fmri data file for session 1 (gii)
                    * data for session 2:
                           - onsets for session 2,
                           - durations for session 2,
                           - fmri data file for session 2 (gii)
                  ],
              - time of repetition
              - mask file
              - mesh file
        """

        pyhrf.verbose(1,'Load surfacic data ...')

        sd = stack_trees([sd.to_dict() for sd in sessions_data])
        onsets = sd['onsets']
        durations = sd['durations']
        bold_files = sd['bold_file']

        if isinstance(durations,list) and durations[0] is None:
            durations = None

        m, mmo, mlf, b, ss, g, el = load_surf_bold_mask(bold_files, mesh_file,
                                                        mask_file)
        mask = m
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        bold = b
        sessionScans = ss
        graphs = g
        edge_lengths = el

        fd = FmriData(onsets, bold, tr, sessionScans, mask, graphs,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files+[mask_file,mesh_file],
                      data_type='surface', edge_lengths=edge_lengths)
        fd.set_init(FmriData.from_surf_ui, sessions_data=sessions_data,
                     tr=tr, mask_file=mask_file, mesh_file=mesh_file)
        return fd


    @classmethod
    def from_simu_ui(self, sessions_data=[FMRISessionSimulationData()]):
        #load simu.pck -> b
        pyhrf.verbose(4, 'from_simu_ui ...')
        pyhrf.verbose(4, 'simulation file: ' + \
                          sessions_data[0].simulation_file)
        bs = [cPickle.load(file(sd.simulation_file)) for sd in sessions_data]
        #print 'condName:',b.conditions
        #print 'onsets:',b.onsets
        #print 'durations:',b.durations
        #print 'b:',b
        #print 'b type is :', type(b)
        fds = []
        for b in bs:
            fd = FmriData.from_simulation_dict(b)
            fds.append(fd)

        fds = merge_fmri_sessions(fds)

        fds.set_init(FmriData.from_simu_ui, sessions_data=sessions_data)
        return fds

    @classmethod
    def from_simulation_dict(self, simulation, mask=None):
        bold = simulation['bold']
        if isinstance(bold, xndarray):
            bold = bold.reorient(['time','voxel'])
            nvox = bold.data.shape[1]
            ss = [range(bold.data.shape[0])] #one session
            print 'BOLD SHAPE=',bold.data.shape
        else:
            nvox = bold.shape[1]
            ss = [range(bold.shape[0])] #one session

        onsets = simulation['paradigm'].stimOnsets
        #onsets = dict(zip(ons.keys(),ons.values()]))
        durations = simulation['paradigm'].stimDurations
        #durations = dict(zip(dur.keys(),[o for o in dur.values()]))

        # print ''
        # print 'onsets:'
        # print onsets
        # print ''
        tr = simulation['tr']

        s = simulation
        labelsVol = s.get('labels_vol',np.ones((1,nvox,)))

        if mask is None:
            if isinstance(labelsVol, xndarray):
                default_mshape = labelsVol.data.shape[1:]
            else:
                default_mshape = labelsVol.shape[1:]
                #print 'default_mask_shape:', default_mshape
                roiMask = simulation.get('mask', np.ones(default_mshape,
                                                         dtype=np.int32))
        else:
            roiMask = read_volume(mask)[0]

        #print 'roiMask:', roiMask.shape
        if len(roiMask.shape) == 3:
            data_type = 'volume'
        else:
            data_type = 'surface' #simulation ??

        return FmriData(onsets, bold, tr, ss, roiMask, stimDurations=durations,
                        simulation=[simulation], data_type=data_type)

    def get_data_files(self):
        return self.data_files

    def save(self, output_dir):
        """
        Save paradigm to output_dir/paradigm.csv,
        BOLD to output_dir/bold.nii, mask to output_dir/mask.nii
        #TODO: handle multi-session

        Return: tuple of file names in this order: (paradigm, bold, mask)
        """
        paradigm_file = op.join(output_dir,'paradigm.csv')
        self.paradigm.save_csv(paradigm_file)
        if self.data_type == 'volume':
            # unflatten bold
            bold_vol = expand_array_in_mask(self.bold, self.roiMask, 1)
            bold_vol = np.rollaxis(bold_vol, 0, 4)
            bold_file = op.join(output_dir, 'bold.nii')
            write_volume(bold_vol, bold_file, self.meta_obj)

            mask_file = op.join(output_dir, 'mask.nii')
            write_volume(self.roiMask, mask_file, self.meta_obj)

        elif self.data_type == 'surface': #TODO surface
            bold_file = op.join(output_dir, 'bold.gii')
            write_texture(self.bold_vol, bold_file, self.meta_obj)
            pass

        return paradigm_file, bold_file, mask_file

    def build_graphs(self, force=False):
        pyhrf.verbose(5,'FmriData.build_graphs (self.graphs is None ? %s ) ...' \
                          %str(self.graphs is None))
        pyhrf.verbose(5,'data_type: %s' %self.data_type)
        if self.graphs is None or force:
            if self.data_type == 'volume':
                pyhrf.verbose(2,'Building graph from volume ...')
                to_discard = [self.backgroundLabel]
                self.graphs = parcels_to_graphs(self.roiMask,
                                                kerMask3D_6n,
                                                toDiscard=to_discard)
                pyhrf.verbose(2,'Graph built (%d rois)!' \
                                  %len(self.graphs.keys()))
                self.edge_lentghs = dict([(i,[[1]*len(nl) for nl in g]) \
                                              for i,g in self.graphs.items()])
            elif self.data_type == 'surface':
                if self.graphs is not None:
                    return self.graphs
                else:
                    raise Exception('Graph is not set for surface data!')


    def get_roi_id(self):
        """ In case of FMRI data containing only one
        ROI, return the id of this ROI. If data contains several ROIs
        then raise an exception
        """
        # roi_ids_in_mask = set(self.roiMask.flatten())
        # if self.backgroundLabel in roi_ids_in_mask:
        #     roi_ids_in_mask.remove(self.backgroundLabel)
        # roi_ids_in_mask = list(roi_ids_in_mask)
        unique_roi_ids = np.unique(self.roi_ids_in_mask)
        if len(unique_roi_ids) > 1:
            raise Exception('FMRI data contains more than one ROI')
        return unique_roi_ids[0]

    def get_nb_vox_in_mask(self):
        return self.nb_voxels_in_mask
        #return (self.roiMask != self.backgroundLabel).sum()

    def get_graph(self):
        self.build_graphs(force=True)
        if len(self.graphs) == 1:
            return self.graphs[self.graphs.keys()[0]]
        else:
            return self.graphs

    def roi_split(self, mask=None):
        if mask is None:
            mask = self.roiMask

        assert mask.shape == self.roiMask.shape

        onsets = self.paradigm.stimOnsets
        durations = self.paradigm.stimDurations

        in_mask = mask[np.where(mask != self.backgroundLabel)]
        data_rois = []
        for roiId in np.unique(mask):
            if roiId != self.backgroundLabel:
                mroi = np.where(mask==roiId)
                roiMask = np.zeros_like(mask) + self.backgroundLabel
                roiMask[mroi] = roiId
                mroi_bold = np.where(in_mask==roiId)
                roiBold = self.bold[:, mroi_bold[0]]
                roiGraph = None
                if self.graphs is not None:
                    roiGraph = self.graphs[roiId]
                    pyhrf.verbose(3,'graph for roi %s has %d nodes'  \
                                      %(roiId,len(roiGraph)))
                else:
                    pyhrf.verbose(3,'graph for roi %s is None' %(roiId))


                simulation = get_roi_simulation(self.simulation, in_mask, roiId)

                data_rois.append(FmriData(onsets, roiBold, self.tr,
                                          self.sessionsScans, roiMask,
                                          {roiId:roiGraph},
                                          durations, self.meta_obj,
                                          simulation, self.backgroundLabel,
                                          self.data_files, self.data_type))
        return data_rois


    def discard_small_rois(self, min_size):
        too_small_rois = [i for i in np.unique(self.roi_ids_in_mask) \
                              if (self.roi_ids_in_mask == i).sum() < min_size]
        pyhrf.verbose(1," %d too small ROIs are discarded (size < %d)" \
                          %(len(too_small_rois), min_size))
        self.discard_rois(too_small_rois)


    def discard_rois(self, roi_ids):
        roiMask = self.roiMask
        for roi_id in roi_ids:
            mroi = np.where(roiMask==roiId)
            roiMask[mroi] = self.backgroundLabel
        self.store_mask_sparse(roiMask)
        if self.graphs is not None:
            self.build_graphs(force=True)


    def keep_only_rois(self, roiIds):
        roiToKeep = set(np.unique(roiIds))
        roiToRemove = set(np.unique(self.roi_ids_in_mask)).difference(roiToKeep)
        self.discard_rois(roiToRemove)

    def get_joined_onsets(self):
        return self.paradigm.get_joined_onsets()

    #TODO : fix average computing -> handle sparse representation of mask
    def compute_average(self):
        # raise NotImplementedError("TODO : fix average computing -> "\
        #                               "handle sparse representation of mask")
        in_mask = self.roiMask[np.where(self.roiMask != self.backgroundLabel)]
        bg_lab = self.backgroundLabel
        roi_ids_in_mask = set(self.roiMask.flatten())
        roi_ids_in_mask.remove(bg_lab)
        self.mask_avg = np.array(sorted(roi_ids_in_mask))
        self.bold_avg = np.zeros((self.bold.shape[0], self.mask_avg.size))
        for i,roi_id in enumerate(self.mask_avg):
            mroi = np.where(in_mask==roi_id)
            self.bold_avg[:,i] = self.bold[:,mroi[0]].mean(1)
        if self.graphs_avg is None and self.graphs is not None:
            # average scenario -> only one position per roi:
            self.graphs_avg = [[]] *  len(self.graph)
        self.graph = self.graphs_avg

    def average(self, flag=True):
        # raise NotImplementedError("TODO : fix average computing -> "\
        #                               "handle sparse representation of mask")
        self.average_flag = flag
        if self.average_flag:
            if self.bold_avg is None:
                self.compute_average()

            self.bold = self.bold_avg
            self.graphs = self.graphs_avg
            #self.roiMask = self.mask_avg
        else:
            self.bold = self.bold_full
            self.graphs = self.graphs_full
            #self.roiMask = self.roiMask_full

    def getSummary(self,long=False):
        """
        """
        bg_lab = self.backgroundLabel

        parcel_sizes = np.bincount(self.roiMask.flatten())

        s = ''
        s += self.paradigm.get_info(long=long)
        roiIds = np.unique(self.roiMask)
        s += ' - geometry: %s [%s]\n' %(str(self.roiMask.shape),
                                      string.join(MRI3Daxes,','))
        s += ' - graph computed: %s\n' %str(self.graphs is not None)
        s += ' - %d rois - [%d' %(len(roiIds),roiIds[0])
        if len(roiIds) > 1:
            s+= '...%d]\n' %(sorted(roiIds)[-1])
        else:
            s+= ']\n'
        s += ' - background label: %d\n' %(self.backgroundLabel)
        bg_size = (self.roiMask==bg_lab).sum()
        s += ' - background size: %d\n' %bg_size
        s += ' - nb voxels (without background): %d\n' \
            %(self.roiMask.size - bg_size)
        if self.meta_obj is not None:
            if isinstance(self.meta_obj, dict) and \
                    self.meta_obj.has_key('voxel_size'):
                s += ' - voxel size : %s\n' %str(self.meta_obj['voxel_size'])
        biggest_parcel = np.argmax(parcel_sizes)
        #print 'biggest_parcel:', biggest_parcel
        s += ' - biggest parcel : %d (size=%d)\n' \
            %(biggest_parcel, parcel_sizes[biggest_parcel])
        s += ' - nb sessions: %d\n' %self.nbSessions
        s += ' - scans per session:\n'
        for si, ss in enumerate(self.sessionsScans):
            s += '   sess %d -> [%d...%d] (%d)\n' %(si,ss[0],ss[-1],len(ss))
        #print 'type tr:', self.tr, type(self.tr)
        s += ' - tr : %1.2f\n' %self.tr
        if long:
            in_mask = self.roiMask[np.where(self.roiMask != bg_lab)]
            #print 'in_mask:', in_mask.size
            #print roiIds
            #print self.bold.keys()
            for i in np.unique(in_mask):
                mroi = np.where(in_mask==i)
                bold_roi = self.bold[:,mroi[0]]
                s += '  .. roid %d: size=%d \n' \
                    %(i, parcel_sizes[i])
                s += '     bold: %1.2f(s%1.2f)[%1.2f;%1.2f]\n' \
                    %(bold_roi.mean(), bold_roi.std(),
                      bold_roi.min(), bold_roi.max())
        return s

    def __str__(self):
        """
        Return a readable string representation of the object.
        Ensures that if 2 objects are instanciated with the same parameters
        they yield the same representation.
        """
        r = self.__class__.__name__ + '('
        r += 'onsets=' + repr(self.paradigm.stimOnsets) + ','
        r += 'bold=' + repr(self.bold) + ','
        r += 'tr=' + repr(self.tr) + ','
        r += 'sessionsScans=' + repr(self.sessionsScans) + ','
        r += 'roiMask=' + repr(self.roiMask) + ','
        r += 'graph=' + repr(self.graphs) + ','
        r += 'stimDurations=' + repr(self.paradigm.stimDurations) + ','
        r += 'meta_obj=' + repr(self.meta_obj) #+ ','
        #r += 'simulation=' + repr(self.simulation)
        #TODO: make proper 'readable' __repr__ function for BoldModel object
        r += ')'
        return r


