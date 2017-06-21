# -*- coding: utf-8 -*-
import os
import os.path as op
import string
import cPickle
import tempfile
import logging

import numpy as np
from pkg_resources import Requirement, resource_filename, resource_listdir
import pyhrf
from pyhrf.ndarray import MRI3Daxes, MRI4Daxes, expand_array_in_mask, TIME_AXIS
from nipy.labs import compute_mask_files
from pyhrf.tools import stack_trees, distance
from pyhrf.graph import parcels_to_graphs, kerMask3D_6n, \
    graph_from_mesh, graph_is_sane, sub_graph
from pyhrf.ndarray import xndarray
from pyhrf.xmlio import XmlInitable

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


logger = logging.getLogger(__name__)


def _pickle_method(method):
    """
    Allow to safely pickle classmethods. To be fed to copy_reg.pickle
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        # deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Allow to safely unpickle classmethods. To be fed to copy_reg.pickle
    """
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None  # if func_name is classmethod
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


def get_data_file_name(filename):
    """Return the path of a given filename."""
    req = Requirement.parse('pyhrf')
    pyhrf_data_sub_path = 'pyhrf/datafiles'
    filename = os.path.join(pyhrf_data_sub_path, filename)
    return resource_filename(req, filename)


def list_data_file_names():
    """List all the data filenames."""
    req = Requirement.parse('pyhrf')
    pyhrf_data_sub_path = 'pyhrf/datafiles'
    return sorted(resource_listdir(req, pyhrf_data_sub_path))


def get_tmp_path(tag='pyhrf_'):
    """Return a temporary path."""
    tmp_dir = tempfile.mkdtemp(prefix=tag, dir=pyhrf.cfg['global']['tmp_path'])
    return tmp_dir


def get_src_path():
    """Return the source path of pyhrf."""
    return os.path.join(os.path.dirname(pyhrf.__file__), '../../')


def get_src_doc_path():
    """Return the documentation path of pyhrf."""
    return op.join(get_src_path(), 'doc/sphinx/source')


# FIXME: do we need the two following classes?
class AttrClass(object):
    """Base class to display attributes."""

    def __init__(self, **kwargs):
        for k, w in kwargs.iteritems():
            setattr(self, k, w)

    def __repr__(self):
        r = self.__class__.__name__ + '('
        a = [k + '=' + repr(getattr(self, k)) for k in dir(self)
             if not k.startswith('_')]
        r += ','.join(a) + ')'

        return r


class Condition(AttrClass):
    """Represents an activation condition"""
    pass

from pyhrf.tools import PickleableStaticMethod

# PARADIGM STUFFS
from pyhrf.paradigm import Paradigm

DEFAULT_SIMULATION_FILE = get_data_file_name('simu.pck')

# Default localizer onsets
# session-specific onsets here, so onset arrays are not encapsulated
# in lists over all sessions
DEFAULT_ONSETS = OrderedDict(
    [('audio', np.array([15., 20.7, 29.7, 35.4, 44.7, 48., 83.4, 89.7, 108.,
                         119.4, 135., 137.7, 146.7, 173.7, 191.7, 236.7,
                         251.7, 284.4, 293.4, 296.7])),
     ('video', np.array([0., 2.4, 8.7, 33., 39., 41.7, 56.4, 59.7, 75., 96.,
                         122.7, 125.4, 131.4, 140.4, 149.4, 153., 156., 159.,
                         164.4, 167.7, 176.7, 188.4, 195., 198., 201., 203.7,
                         207., 210., 218.7, 221.4, 224.7, 234., 246., 248.4,
                         260.4, 264., 266.7, 269.7, 278.4, 288.]))]
)

DEFAULT_STIM_DURATIONS = OrderedDict(
    [('audio', np.array([])),
     ('video', np.array([]))]
)

fn = 'paradigm_loc_av.csv'
DEFAULT_PARADIGM_CSV = get_data_file_name(fn)

# Default Volumic data
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

# Default Surfacic data
dataFn = 'real_data_surf_tiny_bold.gii'
maskFn = 'real_data_surf_tiny_parcellation.gii'
meshFn = 'real_data_surf_tiny_mesh.gii'
DEFAULT_BOLD_SURF_FILE = get_data_file_name(dataFn)
DEFAULT_BOLD_SURF_TR = 2.4
DEFAULT_MASK_SURF_FILE = get_data_file_name(maskFn)
DEFAULT_MESH_FILE = get_data_file_name(meshFn)
DEFAULT_OUT_MASK_SURF_FILE = './roiMask.nii'


class FMRISessionVolumicData(XmlInitable):

    parametersComments = {
        'onsets': 'Onsets of experimental simtuli in seconds. \n'
                  'Dictionnary mapping stimulus name to '
                  'the actual list of onsets.',
        'durations': 'Durations of experimental simtuli in seconds.\n'
                     'It has to consistent with the definition of onsets',
        'bold_file': 'Data file containing the 3D+time BOLD signal (nifti format)'
    }

    def __init__(self, onsets=DEFAULT_ONSETS,
                 durations=DEFAULT_STIM_DURATIONS,
                 bold_file=DEFAULT_BOLD_VOL_FILE):

        XmlInitable.__init__(self)
        assert isinstance(onsets, (dict, OrderedDict))
        assert durations is None or isinstance(durations, (dict, OrderedDict))

        self.onsets = onsets
        self.durations = durations
        self.bold_file = bold_file

    def to_dict(self):
        return {'onsets': self.onsets,
                'durations': self.durations,
                'bold_file': self.bold_file}


class FMRISessionSurfacicData(XmlInitable):
    def __init__(self, onsets=DEFAULT_ONSETS,
                 durations=DEFAULT_STIM_DURATIONS,
                 bold_file=DEFAULT_BOLD_SURF_FILE):

        XmlInitable.__init__(self)
        assert isinstance(onsets, dict)
        assert durations is None or isinstance(durations, dict)

        self.onsets = onsets
        self.durations = durations
        self.bold_file = bold_file

    def to_dict(self):
        return {'onsets': self.onsets,
                'durations': self.durations,
                'bold_file': self.bold_file}


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
        return {'onsets': self.onsets,
                'durations': self.durations,
                'bold_file': self.simulation_file}


def load_vol_bold_and_mask(bold_files, mask_file):

    from pyhrf.tools._io import read_volume, discard_bad_data

    # Handle mask
    if not op.exists(mask_file):
        logger.warning('Mask file %s does not exist. Mask is '
                       'computed from BOLD ...', mask_file)

        bold_file = 'subj0_bold_session0.nii.gz'
        # HACK
        if bold_files[0] == pyhrf.get_data_file_name(bold_file):
            max_frac = .99999  # be sure to keep non zero voxels
            connect_component = False  # default BOLD vol has 2 ROIs
        else:
            max_frac = .9
            connect_component = True
        compute_mask_files(bold_files[0], mask_file, False, .4,
                           max_frac, cc=connect_component)
        mask_loaded_from_file = False
    else:
        mask_loaded_from_file = True
        logger.info('Assuming orientation for mask file: ' +
                    string.join(MRI3Daxes, ','))

    logger.info('Read mask from: %s', mask_file)
    mask, mask_meta_obj = read_volume(mask_file)

    if not np.allclose(np.round(mask), mask):
        raise Exception("Mask is not n-ary (%s)" % mask_file)
    mask = np.round(mask).astype(np.int32)

    logger.info('Mask has shape %s\nMask min value: %d\n'
                'Mask max value: %d\nMask has %d parcels',
                str(mask.shape), mask.min(), mask.max(), len(np.unique(mask)))

    if mask.min() == -1:
        mask += 1

    # Load BOLD:
    last_scan = 0
    session_scans = []
    bolds = []
    logger.info('Assuming orientation for BOLD files: ' +
                string.join(MRI4Daxes, ','))

    if type(bold_files[0]) is list:
        bold_files = bold_files[0]

    for bold_file in bold_files:
        if not op.exists(bold_file):
            raise Exception('File not found: ' + bold_file)

        bold, _ = read_volume(bold_file)
        bolds.append(bold)
        session_scans.append(np.arange(last_scan,
                                       last_scan + bold.shape[TIME_AXIS],
                                       dtype=int))
        last_scan += bold.shape[TIME_AXIS]

    bold = np.concatenate(tuple(bolds), axis=TIME_AXIS)

    logger.info('BOLD has shape %s', str(bold.shape))
    discard_bad_data(bold, mask)

    return mask, mask_meta_obj, mask_loaded_from_file, bold, session_scans


def load_surf_bold_mask(bold_files, mesh_file, mask_file=None):
    from pyhrf.tools._io import read_mesh, read_texture, discard_bad_data

    logger.info('Load mesh: ' + mesh_file)
    coords, triangles, coord_sys = read_mesh(mesh_file)
    logger.debug('Build graph...')
    fgraph = graph_from_mesh(triangles)
    assert graph_is_sane(fgraph)

    logger.info('Mesh has %d nodes', len(fgraph))

    logger.debug('Compute length of edges ... ')
    edges_l = np.array([np.array([distance(coords[i],
                                           coords[n],
                                           coord_sys.xform)
                                  for n in nl])
                        for i, nl in enumerate(fgraph)], dtype=object)

    if mask_file is None or not op.exists(mask_file):
        logger.warning('Mask file %s does not exist. Taking '
                       'all nodes ...', mask_file)
        mask = np.ones(len(fgraph))
        mask_meta_obj = None
        mask_loaded_from_file = False
    else:
        mask, mask_meta_obj = read_texture(mask_file)
        mask_loaded_from_file = True

    if not (np.round(mask) == mask).all():
        raise Exception("Mask is not n-ary")

    if len(mask) != len(fgraph):
        raise Exception('Size of mask (%d) is different from size '
                        'of graph (%d)' % (len(mask), len(fgraph)))

    mask = mask.astype(np.int32)
    if mask.min() == -1:
        mask += 1

    # Split graph into rois:
    graphs = {}
    edge_lengths = {}
    for roi_id in np.unique(mask):
        mroi = np.where(mask == roi_id)
        graph, _ = sub_graph(fgraph, mroi[0])
        edge_lengths[roi_id] = edges_l[mroi]
        graphs[roi_id] = graph

    # Load BOLD:
    last_scan = 0
    session_scans = []
    bolds = []
    for bold_file in bold_files:
        logger.info('load bold: %s', bold_file)
        bold, _ = read_texture(bold_file)
        logger.info('bold shape: %s', str(bold.shape))
        bolds.append(bold)
        session_scans.append(np.arange(last_scan, last_scan + bold.shape[0],
                                       dtype=int))
        last_scan += bold.shape[0]

    bold = np.concatenate(tuple(bolds))
    if len(fgraph) != bold.shape[1]:
        raise Exception('Nb positions not consistent between BOLD (%d) '
                        'and mesh (%d)' % (bold.shape[1], len(fgraph)))

    # discard bad data (bold with var=0 and nan values):
    discard_bad_data(bold, mask, time_axis=0)

    return mask, mask_meta_obj, mask_loaded_from_file, bold, session_scans, \
        graphs, edge_lengths


def merge_fmri_sessions(fmri_data_sets):
    """

    fmri_data_sets: list of FmriData objects.
    Each FmriData object is assumed to contain only one session
    """
    from pyhrf.tools import apply_to_leaves

    all_onsets = stack_trees([fd.paradigm.stimOnsets for fd in fmri_data_sets])
    all_onsets = apply_to_leaves(all_onsets, lambda l: [e[0] for e in l])
    durations = stack_trees(
        [fd.paradigm.stimDurations for fd in fmri_data_sets])
    durations = apply_to_leaves(durations, lambda x: [e[0] for e in x])

    bold = np.concatenate([fd.bold for fd in fmri_data_sets])

    session_scan = []
    last_scan = 0
    for fmri_data in fmri_data_sets:
        nscans = fmri_data.bold.shape[0]
        session_scan.append(
            np.arange(last_scan, last_scan + nscans, dtype=int))
        last_scan += nscans

    # FIXME: define fmri_data instead of loop indice
    if fmri_data.simulation is not None:
        simu = [fmri_data.simulation[0] for fmri_data in fmri_data_sets]
    else:
        simu = None

    return FmriData(all_onsets, bold, fmri_data_sets[0].tr, session_scan,
                    fmri_data_sets[0].roiMask, fmri_data_sets[0].graphs,
                    durations, fmri_data_sets[0].meta_obj, simu,
                    backgroundLabel=fmri_data_sets[0].backgroundLabel,
                    data_files=fmri_data_sets[0].data_files,
                    data_type=fmri_data_sets[0].data_type,
                    edge_lengths=fmri_data_sets[0].edge_lengths,
                    mask_loaded_from_file=fmri_data_sets[0].mask_loaded_from_file)



# FIXME: remove the following class and rewrite the next function
class Object(object):
    pass


def merge_fmri_subjects(fmri_data_sets, roiMask, backgroundLabel=0):
    """

    fmri_data_sets: list of FmriData objects, for different subjects.
    In case of multisession data, merging of fmri data over sessions must
    be done for each subject before using this function.
    roiMask: multi_subject parcellation (nparray)
    """

    fmri_subjects = Object()

    np_roi_mask = np.where(roiMask != backgroundLabel)
    roi_ids_in_mask = roiMask[np_roi_mask]
    unique_roi_ids = np.unique(roi_ids_in_mask)

    fmri_subjects.subjects_data = np.append([f for f in fmri_data_sets])
    fmri_subjects.roiMask = roiMask
    fmri_subjects.roiId = unique_roi_ids

    return fmri_subjects


class FmriGroupData(XmlInitable):
    """
    Used for group level hemodynamic analysis
    Encapsulates FmriData objects for all subjects
    All subjects must habe the same number of ROIs

    Inputs:
        list_subjects: contains list of FmriData object for each subject

    """

    def __init__(self, list_subjects):
        self.data_subjects = list_subjects
        self.nbSubj = len(self.data_subjects)

        self.spatial_shape = list_subjects[0].spatial_shape
        self.meta_obj = list_subjects[0].meta_obj

    def getSummary(self, long=False):
        sout = 'Number of subjects: %d\n' % self.nbSubj
        return sout + '\n'.join([ds.getSummary(long=long)
                                 for ds in self.data_subjects])

    def roi_split(self):
        '''
        Retrieve a list of FmriGroupData object,
        each containing the data for all subject, in one ROI
        '''
        rfd = [ds.roi_split() for ds in self.data_subjects]
        nb_rois = len(rfd[0])
        all_rois = []
        for roi_id in xrange(nb_rois):
            rois = []
            for subject_id in xrange(self.nbSubj):
                rois.append(rfd[subject_id][roi_id])
            all_rois.append(FmriGroupData(rois))
        return all_rois

    def build_graphs(self, force=False):
        return [sd.build_graphs(force=force) for sd in self.data_subjects]

    def get_roi_id(self):
        return self.data_subjects[0].get_roi_id()
        # each FmriData object in self.data_subjetcs contains data for
        # all ROIs, for one subject


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
        m = np.where(mask == roi_id)

        def duplicate(label):
            if simu.has_key(label):
                roi_simu[label] = simu[label]

        def extract_1D_masked(label):
            if simu.has_key(label):
                roi_simu[label] = simu[label][m[0]]

        def extract_2D_masked(label):
            if simu.has_key(label):
                roi_simu[label] = simu[label][:, m[0]]

        [extract_2D_masked(l) for l in ['labels', 'hrf', 'brf', 'prf', 'nrls',
                                        'prls', 'brls', 'noise', 'drift',
                                        'bold', 'stim_induced_signal',
                                        'drift_coeffs', 'bold_stim_induced',
                                        'perf_stim_induced',
                                        'perf_baseline']]
        #[extract_1D_masked(l) for l in ['perf_baseline']]
        [duplicate(l) for l in ['perf_baseline_var', 'drift_var',
                                'primary_hrf', 'var_subject_hrf',
                                'var_hrf_group', 'condition_defs', 'tr', 'dt',
                                'bold_mixt_params', 'perf_mixt_params']]

        simus.append(roi_simu)
    return simus


class FmriData(XmlInitable):
    """
    Attributes:
        onsets: a dictionary mapping a stimulus name to a list of session onsets. Each item of this list is a 1D numpy 
            float array of onsets for a given session.
        stimDurations: same as 'onsets' but stores durations of stimuli
        roiMask: numpy int array of roi labels (0 stands for the background). Shape depends on the data form 
            (3D volumic or 1D surfacic)
        bold: either a 4D numpy float array with axes [sag,cor,ax,scan] and then
            spatial axes must have the same shape as roiMask, or a 2D numpy float array with axes [scan, position] and 
            position axis must have the same length as the number of positions within roiMask (without background). 
            Sessions are stacked in the scan axis
        sessionsScans: a list of session indexes along scan axis.
        tr: Time of repetition of the BOLD signal
        simulation: if not None then it should be a list of simulation instance.
        meta_obj: extra information associated to data
    """

    parametersComments = {
        'tr': 'repetition time in seconds',
        'sessions_data': 'List of data definition for all sessions',
        'mask_file': 'Input n-ary mask file (= parcellation). '
        'Only positive integers are allowed. \n'
        'All zeros are treated as background positions.'
    }

    parametersToShow = ['tr', 'sessions_data', 'mask_file']

    if pyhrf.__usemode__ == 'devel':
        parametersToShow += ['background_label']

    def __init__(self, onsets, bold, tr, sessionsScans, roiMask, graphs=None,
                 stimDurations=None, meta_obj=None, simulation=None,
                 backgroundLabel=0, data_files=None, data_type=None,
                 edge_lengths=None, mask_loaded_from_file=False,
                 extra_data=None):

        logger.info('Creation of FmriData object ...')
        if isinstance(bold, xndarray):
            logger.info('bold shape: %s', str(bold.data.shape))
        else:
            logger.info('bold shape: %s', str(bold.shape))
        logger.info('roi mask shape: %s', str(roiMask.shape))
        logger.info('unique(mask): %s', str(np.unique(roiMask)))

        sessionsDurations = [len(ss) * tr for ss in sessionsScans]
        self.paradigm = Paradigm(onsets, sessionsDurations,
                                 stimDurations)

        self.tr = tr
        self.sessionsScans = sessionsScans

        self.extra_data = extra_data or {}

        if backgroundLabel is None:
            backgroundLabel = 0

        self.backgroundLabel = backgroundLabel

        logger.info('backgroundLabel: %d', self.backgroundLabel)

        self.store_mask_sparse(roiMask)

        m = self.np_roi_mask
        if isinstance(bold, xndarray):
            if bold.data.ndim > 2:  # volumic data
                print '----------------'
                if data_type is None:
                    data_type = 'volume'
                assert roiMask.ndim == 3
                if TIME_AXIS == 0:
                    bold = bold.data[:, m[0], m[1], m[2]]
                else:  # TIME_AXIS = 3
                    bold = bold.data[m[0], m[1], m[2], :].transpose()

            else:  # surfacic or already flatten data
                assert bold.data.ndim == 2
                if len(m[0]) != bold.data.shape[1]:
                    bold = bold.data[:, m[0]]
                else:
                    bold = bold.data
        else:
            if bold.ndim > 2:  # volumic data
                if data_type is None:
                    data_type = 'volume'
                assert roiMask.ndim == 3
                if TIME_AXIS == 0:
                    bold = bold[:, m[0], m[1], m[2]]
                else:  # TIME_AXIS = 3
                    bold = bold[m[0], m[1], m[2], :].transpose()

            else:  # surfacic or already flatten data
                assert bold.ndim == 2
                if len(m[0]) != bold.shape[1]:
                    bold = bold[:, m[0]]

        self.data_type = data_type

        logger.info('extracted bold shape (without background): %s',
                    str(bold.shape))

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
        if data_files is None:
            data_files = []
        self.data_files = data_files

    def get_extra_data(self, label, default):
        return self.extra_data.get(label, default)

    def set_extra_data(self, label, value):
        self.extra_data[label] = value

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
                       tr=DEFAULT_BOLD_VOL_TR, background_label=None,
                       paradigm_csv_delim=None):
        paradigm = Paradigm.from_csv(paradigm_csv_file,
                                     delim=paradigm_csv_delim)
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
                      data_files=bold_files + [mask_file, paradigm_csv_file],
                      data_type='volume',
                      mask_loaded_from_file=mask_loaded_from_file,
                      backgroundLabel=background_label)
        fd.set_init(FmriData.from_vol_files, mask_file=mask_file,
                    paradigm_csv_file=paradigm_csv_file,
                    bold_files=bold_files, tr=tr,
                    background_label=background_label,
                    paradigm_csv_delim=paradigm_csv_delim)
        return fd

    @PickleableStaticMethod
    def from_vol_files_rel(self, mask_file, paradigm_csv_file, bold_files,
                           tr, rel_conditions):
        paradigm = Paradigm.from_csv(paradigm_csv_file)
        durations = OrderedDict()
        onsets = OrderedDict()
        for i in xrange(len(rel_conditions)):
            durations[rel_conditions[i]] = paradigm.stimDurations[
                rel_conditions[i]]
            onsets[rel_conditions[i]] = paradigm.stimOnsets[rel_conditions[i]]
        m, mmo, mlf, b, ss = load_vol_bold_and_mask(bold_files, mask_file)
        mask = m
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        bold = b
        sessionScans = ss

        fd = FmriData(onsets, bold, tr, sessionScans, mask,
                      stimDurations=durations, meta_obj=mask_meta_obj,
                      data_files=bold_files + [mask_file, paradigm_csv_file],
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
        logger.info('Load volumic data ...')
        logger.info('Input sessions data:')
        logger.debug(sessions_data)
        sda = stack_trees([sda.to_dict() for sda in sessions_data])
        onsets = sda['onsets']
        durations = sda['durations']
        bold_files = sda['bold_file']

        # FIXME: HACKS!!
        if isinstance(onsets.values()[0][0], list):
            for i in xrange(len(onsets.keys())):
                onsets[onsets.keys()[i]] = onsets[onsets.keys()[i]][0]

        if isinstance(durations.values()[0][0], list):
            for i in xrange(len(durations.keys())):
                durations[durations.keys()[i]] = \
                    durations[durations.keys()[i]][0]

        if len(durations.values()[0][0].shape) > 1:
            for i in xrange(len(durations.keys())):
                durations.values()[i][0] = durations.values()[i][0][0]

        if len(onsets.keys()) == 1:
            onsets[onsets.keys()[0]] = onsets[onsets.keys()[0]][0]
            durations[onsets.keys()[0]] = durations[onsets.keys()[0]][0]
        if isinstance(durations, list) and durations[0] is None:
            durations = None

        mask, mmo, mlf, bold, session_scans = \
            load_vol_bold_and_mask(bold_files, mask_file)
        mask_meta_obj = mmo
        mask_loaded_from_file = mlf
        fmri_data = FmriData(onsets, bold, tr, session_scans, mask,
                             stimDurations=durations, meta_obj=mask_meta_obj,
                             data_files=bold_files + [mask_file],
                             backgroundLabel=background_label,
                             data_type='volume',
                             mask_loaded_from_file=mask_loaded_from_file)

        fmri_data.set_init(FmriData.from_vol_ui, sessions_data=sessions_data,
                           tr=tr, mask_file=mask_file,
                           background_label=background_label)
        return fmri_data

    @classmethod
    def from_surf_files(self, paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                        bold_files=None, tr=DEFAULT_BOLD_SURF_TR,
                        mesh_file=DEFAULT_MESH_FILE, mask_file=None):
        """Return FmriData representation from surf files"""
        if bold_files is None:
            bold_files = [DEFAULT_BOLD_SURF_FILE]

        paradigm = Paradigm.from_csv(paradigm_csv_file)
        durations = paradigm.stimDurations
        onsets = paradigm.stimOnsets

        mask, mask_meta_obj, _, bold, session_scans, graphs, edge_lengths = \
            load_surf_bold_mask(bold_files, mesh_file, mask_file)

        fmri_data = FmriData(onsets, bold, tr, session_scans, mask, graphs,
                             stimDurations=durations, meta_obj=mask_meta_obj,
                             data_files=bold_files + [mask_file, mesh_file],
                             data_type='surface', edge_lengths=edge_lengths)
        fmri_data.set_init(FmriData.from_surf_files,
                           paradigm_csv_file=paradigm_csv_file,
                           bold_files=bold_files, tr=tr, mesh_file=mesh_file,
                           mask_file=mask_file)
        return fmri_data

    @classmethod
    def from_surf_ui(cls, sessions_data=None, tr=DEFAULT_BOLD_SURF_TR,
                     mask_file=DEFAULT_MASK_SURF_FILE,
                     mesh_file=DEFAULT_MESH_FILE):
        """
        Convenient creation function intended to be used for XML I/O.
        'session_data' is a list of FMRISessionVolumicData objects.
        'tr' is the time of repetition.
        'mask_file' is a path to a functional mask file.

        This represents the following hierarchy:
        
        .. code:: 
        
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

        if sessions_data is None:
            sessions_data = [FMRISessionSurfacicData()]

        logger.info('Load surfacic data...')

        sda = stack_trees([sda.to_dict() for sda in sessions_data])
        onsets = sda['onsets']
        durations = sda['durations']
        bold_files = sda['bold_file']

        if isinstance(durations, list) and durations[0] is None:
            durations = None

        (mask, mask_meta_obj, _, bold, session_scans,
         graphs, edge_lengths) = load_surf_bold_mask(bold_files, mesh_file,
                                                     mask_file)

        fmri_data = FmriData(onsets, bold, tr, session_scans, mask, graphs,
                             stimDurations=durations, meta_obj=mask_meta_obj,
                             data_files=bold_files + [mask_file, mesh_file],
                             data_type='surface', edge_lengths=edge_lengths)
        fmri_data.set_init(FmriData.from_surf_ui, sessions_data=sessions_data,
                           tr=tr, mask_file=mask_file, mesh_file=mesh_file)
        return fmri_data

    @classmethod
    def from_simu_ui(cls, sessions_data=None):
        if sessions_data is None:
            sessions_data = [FMRISessionSimulationData()]
        # load simu.pck -> b
        logger.debug('from_simu_ui\n'
                     'simulation file: %s', sessions_data[0].simulation_file)
        bs = [cPickle.load(file(sd.simulation_file)) for sd in sessions_data]
        # print 'condName:',b.conditions
        # print 'onsets:',b.onsets
        # print 'durations:',b.durations
        # print 'b:',b
        # print 'b type is :', type(b)
        fds = []
        for b in bs:
            fd = FmriData.from_simulation_dict(b)
            fds.append(fd)

        fds = merge_fmri_sessions(fds)

        fds.set_init(FmriData.from_simu_ui, sessions_data=sessions_data)
        return fds

    @classmethod
    def from_simulation_dict(self, simulation, mask=None):
        from pyhrf.tools._io import read_volume

        bold = simulation['bold']

        if isinstance(bold, xndarray):
            bold = bold.reorient(['time', 'voxel'])
            nvox = bold.data.shape[1]
            ss = [range(bold.data.shape[0])]  # one session
            print 'BOLD SHAPE=', bold.data.shape
        else:
            nvox = bold.shape[1]
            ss = [range(bold.shape[0])]  # one session

        onsets = simulation['paradigm'].stimOnsets
        # onsets = dict(zip(ons.keys(),ons.values()]))
        durations = simulation['paradigm'].stimDurations
        #durations = dict(zip(dur.keys(),[o for o in dur.values()]))

        # print ''
        # print 'onsets:'
        # print onsets
        # print ''
        tr = simulation['tr']

        s = simulation
        labelsVol = s.get('labels_vol', np.ones((1, nvox,)))

        if mask is None:
            if isinstance(labelsVol, xndarray):
                default_mshape = labelsVol.data.shape[1:]
            else:
                default_mshape = labelsVol.shape[1:]
                # print 'default_mask_shape:', default_mshape
                roiMask = simulation.get('mask', np.ones(default_mshape,
                                                         dtype=np.int32))
        else:
            roiMask = read_volume(mask)[0]

        # print 'roiMask:', roiMask.shape
        if len(roiMask.shape) == 3:
            data_type = 'volume'
        else:
            data_type = 'surface'  # simulation ??

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
        from pyhrf.tools._io import write_volume, write_texture
        paradigm_file = op.join(output_dir, 'paradigm.csv')
        self.paradigm.save_csv(paradigm_file)
        if self.data_type == 'volume':
            # unflatten bold
            bold_vol = expand_array_in_mask(self.bold, self.roiMask, 1)
            bold_vol = np.rollaxis(bold_vol, 0, 4)
            bold_file = op.join(output_dir, 'bold.nii')
            write_volume(bold_vol, bold_file, self.meta_obj)

            mask_file = op.join(output_dir, 'mask.nii')
            write_volume(self.roiMask, mask_file, self.meta_obj)

        elif self.data_type == 'surface':  # TODO surface
            bold_file = op.join(output_dir, 'bold.gii')
            write_texture(self.bold_vol, bold_file, self.meta_obj)
            pass

        return paradigm_file, bold_file, mask_file

    def build_graphs(self, force=False):
        logger.debug('FmriData.build_graphs (self.graphs is None ? %s ) ...',
                     str(self.graphs is None))
        logger.debug('data_type: %s', self.data_type)
        if self.graphs is None or force:
            if self.data_type == 'volume':
                logger.info('Building graph from volume ...')
                to_discard = [self.backgroundLabel]
                self.graphs = parcels_to_graphs(self.roiMask,
                                                kerMask3D_6n,
                                                toDiscard=to_discard)
                logger.info('Graph built (%d rois)!', len(self.graphs.keys()))
                self.edge_lentghs = dict([(i, [[1] * len(nl) for nl in g])
                                          for i, g in self.graphs.items()])
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
        # return (self.roiMask != self.backgroundLabel).sum()

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
                mroi = np.where(mask == roiId)
                roiMask = np.zeros_like(mask) + self.backgroundLabel
                roiMask[mroi] = roiId
                mroi_bold = np.where(in_mask == roiId)
                roiBold = self.bold[:, mroi_bold[0]]
                roiGraph = None
                if self.graphs is not None:
                    roiGraph = self.graphs[roiId]
                    logger.info('graph for roi %s has %d nodes',
                                roiId, len(roiGraph))
                else:
                    logger.info('graph for roi %s is None', (roiId))

                simulation = get_roi_simulation(
                    self.simulation, in_mask, roiId)

                data_rois.append(FmriData(onsets, roiBold, self.tr,
                                          self.sessionsScans, roiMask,
                                          {roiId: roiGraph},
                                          durations, self.meta_obj,
                                          simulation, self.backgroundLabel,
                                          self.data_files, self.data_type))
        return data_rois

    def discard_small_rois(self, min_size):
        too_small_rois = [i for i in np.unique(self.roi_ids_in_mask)
                          if (self.roi_ids_in_mask == i).sum() < min_size]
        logger.info(" %d too small ROIs are discarded (size < %d)",
                    len(too_small_rois), min_size)
        self.discard_rois(too_small_rois)

    def discard_rois(self, roi_ids):
        roiMask = self.roiMask
        for roi_id in roi_ids:
            mroi = np.where(roiMask == roi_id)
            roiMask[mroi] = self.backgroundLabel
        self.store_mask_sparse(roiMask)
        if self.graphs is not None:
            self.build_graphs(force=True)

    def keep_only_rois(self, roiIds):
        roiToKeep = set(np.unique(roiIds))
        roiToRemove = set(
            np.unique(self.roi_ids_in_mask)).difference(roiToKeep)
        self.discard_rois(roiToRemove)

    def get_joined_onsets(self):
        return self.paradigm.get_joined_onsets()

    def get_joined_durations(self):
        return self.paradigm.get_joined_durations()

    # TODO : fix average computing -> handle sparse representation of mask
    def compute_average(self):
        # raise NotImplementedError("TODO : fix average computing -> "\
        #                               "handle sparse representation of mask")
        in_mask = self.roiMask[np.where(self.roiMask != self.backgroundLabel)]
        bg_lab = self.backgroundLabel
        roi_ids_in_mask = set(self.roiMask.flatten())
        roi_ids_in_mask.remove(bg_lab)
        self.mask_avg = np.array(sorted(roi_ids_in_mask))
        self.bold_avg = np.zeros((self.bold.shape[0], self.mask_avg.size))
        for i, roi_id in enumerate(self.mask_avg):
            mroi = np.where(in_mask == roi_id)
            self.bold_avg[:, i] = self.bold[:, mroi[0]].mean(1)
        if self.graphs_avg is None and self.graphs is not None:
            # average scenario -> only one position per roi:
            self.graphs_avg = [[]] * len(self.graph)
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

    def getSummary(self, long=False):
        """
        """
        bg_lab = self.backgroundLabel

        parcel_sizes = np.bincount(self.roiMask.flatten())

        s = ''
        s += self.paradigm.get_info(long=long)
        roiIds = np.unique(self.roiMask)
        s += ' - geometry: %s [%s]\n' % (str(self.roiMask.shape),
                                         string.join(MRI3Daxes, ','))
        s += ' - graph computed: %s\n' % str(self.graphs is not None)
        s += ' - %d rois - [%d' % (len(roiIds), roiIds[0])
        if len(roiIds) > 1:
            s += '...%d]\n' % (sorted(roiIds)[-1])
        else:
            s += ']\n'
        s += ' - background label: %d\n' % (self.backgroundLabel)
        bg_size = (self.roiMask == bg_lab).sum()
        s += ' - background size: %d\n' % bg_size
        s += ' - nb voxels (without background): %d\n' \
            % (self.roiMask.size - bg_size)
        if self.meta_obj is not None:
            if isinstance(self.meta_obj, dict) and \
                    self.meta_obj.has_key('voxel_size'):
                s += ' - voxel size : %s\n' % str(self.meta_obj['voxel_size'])
        biggest_parcel = np.argmax(parcel_sizes)
        # print 'biggest_parcel:', biggest_parcel
        s += ' - biggest parcel : %d (size=%d)\n' \
            % (biggest_parcel, parcel_sizes[biggest_parcel])
        s += ' - nb sessions: %d\n' % self.nbSessions
        s += ' - scans per session:\n'
        for si, ss in enumerate(self.sessionsScans):
            s += '   sess %d -> [%d...%d] (%d)\n' % (si,
                                                     ss[0], ss[-1], len(ss))
        # print 'type tr:', self.tr, type(self.tr)
        s += ' - tr : %1.2f\n' % self.tr
        if long:
            in_mask = self.roiMask[np.where(self.roiMask != bg_lab)]
            # print 'in_mask:', in_mask.size
            # print roiIds
            # print self.bold.keys()
            for i in np.unique(in_mask):
                mroi = np.where(in_mask == i)
                bold_roi = self.bold[:, mroi[0]]
                s += '  .. roid %d: size=%d \n' \
                    % (i, parcel_sizes[i])
                s += '     bold: %1.2f(s%1.2f)[%1.2f;%1.2f]\n' \
                    % (bold_roi.mean(), bold_roi.std(),
                       bold_roi.min(), bold_roi.max())
        return s

    def __repr__(self):
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
        r += 'meta_obj=' + repr(self.meta_obj)  # + ','
        #r += 'simulation=' + repr(self.simulation)
        # TODO: make proper 'readable' __repr__ function for BoldModel object
        r += ')'
        return r
