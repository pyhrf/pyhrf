# -*- coding: utf-8 -*-

import string
import numpy as np
import re
import csv
import tempfile
import os
import os.path as op
import shutil
import logging

from itertools import chain
from collections import defaultdict

import nibabel

from nibabel import gifti

import pyhrf

from pyhrf.tools import distance


logger = logging.getLogger(__name__)


PICKLE_DUMPED_FILE_EXTENSION = 'pck'
VOL_NII_EXTENSION = 'nii'
TEXTURE_EXTENSION = 'gii'


def find_duplicates(l):
    """ Find the index of duplicate elements in the given list.
    Complexity is O(n*log(d)) where d is the number of unique duplicate
    elements.

    Args:
        - l (list): the list where to search for duplicates

    Return: groups of indexes corresponding to duplicated element:
            -> list of list of elements

    Example:
    >>> find_duplicates([1,1,2,1,4,5,7,2])
    [[0, 1, 3], [2, 7]]
    """
    duplicates = defaultdict(list)
    for i, e in enumerate(l):
        duplicates[e].append(i)
    return [dups for dups in duplicates.values() if len(dups) > 1]


class MissingTagError(Exception):
    pass


class DuplicateTargetError(Exception):
    pass


def rx_copy(src, src_folder, dest_basename, dest_folder, dry=False,
            replacements=None, callback=None):
    """
    Copy all file names matching the regexp *src* in folder *src_folder* to
    targets defined by format strings.
    The association between source and target file names must be injective.
    If several file names matched by *src* are associated with the same target,
    an exception will be raised.

    Args:
        - src (str): regular expression matching file names in the given folder
                     *src_folder* where group names are mapped to
                     format argument names in *dest_folder* and *dest_basename*
        - src_folder (str): path where to search for files matched by *src*
        - dest_basename (str): format string with named arguments
                               (eg 'file_{mytag}_{mytag2}.txt') used to form
                               target file basenames where named arguments are
                               substituted with group values caught by *src*.
        - dest_folder (list of str):
                      list of format strings with named arguments,
                      eg ('folder_{mytag}','folder_{mytag2}'),  to be joined
                      to form target directories.
                      Named arguments are substituted with group values
                      extracted by *src*.
        - dry (bool): if True then do not perform any copy
        - replacements (list of tuple of str):
                      list of replacements, ie [(old,new), (old,new), ...],
                      to be applied to target file names by calling the function
                      string.replace(old,new) on them
        - callback (func): callback function called on final filenames to
                           modify target name or filter a copy operation.
                           -> callback(src_fn, dest_fn)
                              Return: None | str
                                      If None then no copy is performed
                                      If str then it replaces the target filename
    Return: None
    """
    replacements = replacements or []
    callback = callback or (lambda x, y: y)
    # check consistency between src group names and dest items:
    src_tags = set(re.compile(src).groupindex.keys())

    re_named_args = re.compile('\{(.*?)\}')
    folder_dtags = set(chain(*[re_named_args.findall(d)
                               for d in dest_folder]))
    bn_dest_tags = set(re_named_args.findall(dest_basename))

    if not folder_dtags.issubset(src_tags):
        raise MissingTagError('Tags in dest_folder not defined in src: %s'
                              % ', '.join(folder_dtags.difference(src_tags)))
    if not bn_dest_tags.issubset(src_tags):
        raise MissingTagError('Tags in dest_basename not defined in src: %s'
                              % ', '.join(bn_dest_tags.difference(src_tags)))

    # resolve file names
    input_files, output_files = [], []
    re_src = re.compile(op.join(re.escape(src_folder), src))

    for root, dirs, files in os.walk(src_folder):
        for fn in files:
            input_fn = op.join(root, fn)
            ri = re_src.match(input_fn)
            if ri is not None:
                subs = ri.groupdict()
                output_subs = [df.format(**subs) for df in dest_folder]
                output_file = op.join(*(output_subs +
                                        [dest_basename.format(**subs)]))
                for old, new in replacements:
                    output_file = output_file.replace(old, new)
                output_file = callback(input_fn, output_file)
                if output_file is not None:
                    input_files.append(input_fn)
                    output_files.append(output_file)

    # check injectivity:
    duplicate_indexes = find_duplicates(output_files)
    if len(duplicate_indexes) > 0:
        sduplicates = '\n'.join(*[['\n'.join(sorted([input_files[i]
                                                     for i in dups])) +
                                   '\n' + '-> ' + output_files[dups[0]] + '\n']
                                  for dups in duplicate_indexes])
        raise DuplicateTargetError('Copy is not injective, the following copy'
                                   ' operations have the same destination:\n'
                                   '%s' % sduplicates)

    msg = '\n'.join(['\n-> '.join((ifn, ofn))
                     for ifn, ofn in zip(input_files, output_files)])
    logger.info(msg)

    if not dry:  # do the copy
        for ifn, ofn in zip(input_files, output_files):
            # Create sub directories if not existing:
            output_item_dir = op.dirname(ofn)
            if not op.exists(output_item_dir):
                os.makedirs(output_item_dir)
            shutil.copy(ifn, ofn)
    return


def read_volume(fileName, remove_nans=True):
    """
    Read the content of 'filename' using nibabel.
    Return a tuple: (data numpy array, meta_data)
    Orientation:  nibabel convention
    """
    logger.debug('read_volume: %s', fileName)
    nim = nibabel.load(fileName)
    arr = np.array(nim.get_data())  # avoid memmapping
    # when array is memmapped from disk image:
    # Exception AttributeError: AttributeError("'NoneType' object has no attribute 'tell'",) in <bound method memmap.__del__ of memmap(
    # TODO: check and maybe fix this because this can be useful with big
    # volumes
    if remove_nans:
        arr[np.isnan(arr)] = 0

    # print 'arr.shape:', arr.shape
    # if arr.ndim == 4:
    #    arr = np.rollaxis(arr, 3, 0)

    return arr, (nim.get_affine(), nim.get_header())


def cread_volume(fileName):
    from pyhrf.ndarray import xndarray
    return xndarray.load(fileName)


def read_spatial_resolution(fileName):
    nim = nibabel.load(fileName)
    return nim.get_header()['pixdim'][1:4]


def write_volume(data, fileName, meta_data=None):
    """
    Write the numpy array 'data' into a file 'filename' in nifti format,
    """
    logger.debug('creating nifti image from data : %s, %s', str(data.shape),
                 str(data.dtype))

    if data.dtype == np.bool:
        data = data.astype(np.int8)

    if meta_data is None:
        affine = np.eye(4)
        header = None
    else:
        affine, header = meta_data
        header.set_data_dtype(data.dtype)

    img = nibabel.Nifti1Image(data, affine, header)
    logger.debug('Saving nifti Image to: %s', fileName)
    img.to_filename(fileName)


def readImageWithPynifti(fileName, remove_nans=True):
    raise DeprecationWarning('Use read_volume instead ...')


def writeImageWithPynifti(data, fileName, header={}):
    raise DeprecationWarning('Use write_volume instead ...')


def hasNiftiExtension(fn):
    sfn = fn.split('.')
    return sfn[-1] == 'nii' or (sfn[-1] == 'gz' and sfn[-2] == 'nii')


def splitFileName(fn, ext='nii'):

    sfn = fn.split('.')
    if sfn[-1] == ext:
        return (string.join(sfn[:-1], '.'), sfn[-1])
    elif sfn[-1] == 'gz' and sfn[-2] == ext:
        return string.join(sfn[:-2], '.'), string.join(sfn[-2:], '.')
    else:
        print 'SplitFileName : unrecognised extension!'
        return (fn, None)


def splitNiftiFileName(fn):
    sfn = fn.split('.')
    if sfn[-1] == 'nii':
        return (string.join(sfn[:-1], '.'), sfn[-1])
    elif sfn[-1] == 'gz' and sfn[-2] == 'nii':
        return string.join(sfn[:-2], '.'), string.join(sfn[-2:], '.')
    else:
        print 'SplitNiftiFileName : unrecognised nifti ext!'
        return (fn, None)


def writeDictVolumesNiftii(vols, fileName, meta_data=None, sep='_', slice=None):
    logger.info('writeDictVolumesNiftii, fn: %s', fileName)
    if vols is None:
        return
    if slice is None:
        slice = tuple()

    if type(vols) == dict:
        (baseFileName, ext) = splitFileName(fileName, 'nii')
        filenames = []
        for k, v in vols.iteritems():
            # print '%%% k:', k
            # print '%%% v:', v
            if k != 'root':
                newFn = baseFileName + sep + k[0] + sep + str(k[1]) + '.' + ext
                if np.isreal(k[1]):
                    sl = (k[1],)
                else:
                    sl = (str(k[1]),)
            else:
                newFn = fileName
                sl = tuple()
            # print 'newFn :', newFn
            # print 'slice :', slice
            filenames.extend(writeDictVolumesNiftii(v, newFn, meta_data,
                                                    sep, slice + sl))
        return filenames
    else:  # I got a numpy.ndarray volume
        if 0:
            print '~~~~~~~~~~~~~~~~~~'
            print 'Got ndarray ... writing it ...'
            print 'vols.shape', vols.shape
            print '~~~~~~~~~~~~~~~~~~'
        write_volume(vols, fileName, meta_data)
        # print '->slice:,', slice
        # print '->fn:', fileName
        return [(slice, fileName)]


def writeDictVolumesTex(vols, fileName):
    logger.info('writeDictVolumesTex, fn: %s', fileName)
    if vols == None:
        return
    if type(vols) == dict:
        (baseFileName, ext) = splitFileName(fileName, TEXTURE_EXTENSION)
        for k, v in vols.iteritems():
            # print '%%% k:', k
            # print '%%% v:', v
            if k != 'root':
                newFn = baseFileName + '_' + k[0] + '_' + str(k[1]) + '.' + ext
            else:
                newFn = fileName
            # print 'newFn :', newFn
            writeDictVolumesTex(v, newFn)
    else:  # I got a numpy.ndarray volume
        # print 'Got ndarray ... writing it ...'
        #tex = Texture(fileName, data=vols)
        # tex.write()
        write_texture(vols, fileName)


def sub_sample_vol(image_file, dest_file, dsf, interpolation='continuous',
                   verb_lvl=0):

    # ,CompositionError
    from nipy.labs.datasets.volumes.volume_img import VolumeImg

    logger.log(verb_lvl, 'Subsampling at dsf=%d, %s -> %s', dsf, image_file,
               dest_file)

    interp = interpolation
    data_src, src_meta = read_volume(image_file)
    affine = src_meta[0]
    daffine = dsf * affine

    original_dtype = data_src.dtype
    if np.issubdtype(np.int, data_src.dtype):
        # to avoid error "array type 5 not supported" on int arrays ...
        #data_src = np.asarray(np.asfarray(data_src), data_src.dtype)
        data_src = np.asfarray(data_src)

    if data_src.ndim == 4:
        ref_vol = data_src[:, :, :, 0]
    else:
        ref_vol = data_src

    img_src = VolumeImg(ref_vol, affine, 'mine')
    img_dest = img_src.as_volume_img(daffine, interpolation=interpolation)

    # setup dest geometry:
    dest_header = src_meta[1].copy()

    dpixdim = np.array(src_meta[1]['pixdim'])
    dpixdim[1:4] *= dsf

    dest_header['pixdim'] = list(dpixdim)
    sh = ref_vol[::dsf, ::dsf, ::dsf, ...].shape
    dest_meta = (daffine, dest_header)

    # do the resampling:
    if data_src.ndim == 3:
        vol = img_dest.get_data()[:sh[0], :sh[1], :sh[2]]
        if ref_vol.dtype == np.int32 or \
                np.allclose(np.round(ref_vol), ref_vol):  # if input is int
            vol = np.round(vol).astype(np.int32)
        write_volume(vol, dest_file, dest_meta)

    elif data_src.ndim == 4:
        imgs = [VolumeImg(i, affine, 'mine')
                for i in np.rollaxis(data_src, 3, 0)]
        dvols = [i.as_volume_img(daffine, interpolation=interp).get_data()
                 for i in imgs]
        # print 'dvols[0]:', dvols[0].shape
        dvols = np.array(dvols)
        # print 'dvols:', dvols.shape
        sub_vols = np.rollaxis(dvols[:, :sh[0], :sh[1], :sh[2]], 0, 4)
        # print 'sub_vols:', sub_vols.shape
        write_volume(sub_vols, dest_file, dest_meta)

    else:
        raise Exception('Nb of dims (%d) not handled. Only 3D or 4D'
                        % data_src.ndim)


def concat3DVols(files, output):
    """ Concatenate 3D volumes given by a list a file to 4D volume (output)
    """
    img4D = None
    for i, f in enumerate(files):
        img, meta_data = read_volume(f)
        img = img.squeeze()  # remove dim of length=1
        if i == 0:
            logger.info("Image meta_obj: %s", meta_data)
            logger.info("Volume size: %s, type: %s", str(img.shape),
                        str(img.dtype))
            img4D = np.zeros(img.shape + (len(files),), img.dtype)
            # print 'img4D.shape:', img4D.shape
        img4D[:, :, :, i] = img
    if img4D is not None:
        write_volume(img4D, output, meta_data=meta_data)
    else:
        raise Exception('No 4D output produced from files: %s' % str(files))


def split_ext_safe(fn):
    root, ext = op.splitext(fn)
    if ext == '.gz':
        root, n_ext = op.splitext(root)
        ext = n_ext + ext
    return root, ext


def split4DVol(boldFile, output_dir=None):
    from pyhrf.ndarray import TIME_AXIS

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='pyhrf',
                                      dir=pyhrf.cfg['global']['tmp_path'])
    root, ext = split_ext_safe(boldFile)
    img, meta_data = read_volume(boldFile)

    img = img.squeeze()  # remove dim of length=1
    fns = []
    for i in xrange(img.shape[TIME_AXIS]):
        fn = op.join(output_dir, op.basename(root)) + '_%06d' % i + ext
        if TIME_AXIS == 0:
            write_volume(img[i, :, :, :], fn, meta_data)
        else:
            write_volume(img[:, :, :, i], fn, meta_data)
        fns.append(fn)
    return fns

# DATA sanity checks:
# - volume shapes across sessions


# def load_paradigm_from_csv(csvFile, delim=' '):
def load_paradigm_from_csv(csvFile, delim=None):
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

    onsets = defaultdict(lambda: defaultdict(list))
    durations = defaultdict(lambda: defaultdict(list))

    with open(csvFile, "rb") as csv_file:
        if delim is None:
            dialect = csv.Sniffer().sniff(csv_file.read())
            csv_file.seek(0)
            delim = dialect.delimiter
        reader = csv.reader(csv_file, delimiter=delim, skipinitialspace=True)

        for row in reader:
            # print 'row:', row
            isession = int(row[0])
            cond = row[1]
            onset = float(row[2])

            amplitude = 1.
            duration = 0.0
            if len(row) > 3:
                duration = float(row[3])
            if len(row) > 4:
                amplitude = float(row[4])

            onsets[cond][isession].append(onset)
            durations[cond][isession].append(duration)

    onsets = dict([(cn, [np.array(o[s]) for s in sorted(o.keys())])
                   for cn, o in onsets.iteritems()])

    durations = dict([(cn, [np.array(dur[s]) for s in sorted(dur.keys())])
                      for cn, dur in durations.iteritems()])

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
    from pyhrf.ndarray import MRI3Daxes, MRI4Daxes, TIME_AXIS
    from pyhrf.graph import parcels_to_graphs, kerMask3D_6n

    for boldFile in boldFiles:
        if not os.path.exists(boldFile):
            raise Exception('File not found: ' + boldFile)
        #assert hasNiftiExtension(boldFile)

    if not os.path.exists(roiMaskFile):
        raise Exception('File not found: ' + roiMaskFile)
    #assert hasNiftiExtension(roiMaskFile)

    # Load ROIs:
    try:
        roiMask, mask_meta_data = read_volume(roiMaskFile)
    except Exception, e:
        print 'Could not read file %s, error was:\n %s' % (roiMaskFile, str(e))
        raise e
    logger.info('Assuming orientation for mask files: %s',
                string.join(MRI3Daxes, ','))
    logger.info('Loaded a mask of shape %s', str(roiMask.shape))
    mshape = roiMask.shape
    if roiMask.min() == -1:
        roiMask += 1

    # Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    logger.info('Assuming orientation for BOLD files: %s',
                string.join(MRI4Daxes, ','))
    for boldFile in boldFiles:
        try:
            b, b_meta = read_volume(boldFile)
        except Exception, e:
            logger.error('Could not read file %s, error was:\n %s', boldFile,
                         str(e))
        bolds.append(b)
        logger.info('Loaded a BOLD file of shape %s', str(b.shape))

        if (TIME_AXIS == 0 and mshape != b.shape[1:]) or \
                (TIME_AXIS == 3 and mshape != b.shape[:-1]):
            raise Exception('Error: BOLD shape is different from mask shape')

        sessionScans.append(np.arange(lastScan, lastScan + b.shape[TIME_AXIS],
                                      dtype=int))
        lastScan += b.shape[TIME_AXIS]
    bold = np.concatenate(tuple(bolds), axis=TIME_AXIS)

    discard_bad_data(bold, roiMask)

    if build_graph:
        # Build graph

        if len(np.unique(roiMask)) == 1 or keepBackground:
            graphs = parcels_to_graphs(roiMask.astype(int), kerMask3D_6n)
        else:
            if (roiMask == 0).any():
                toDiscard = [0]
                logger.info('Discarding background (label=0)')
            else:
                toDiscard = None
            graphs = parcels_to_graphs(roiMask.astype(int), kerMask3D_6n,
                                       toDiscard=toDiscard)
        logger.info('Graph built !')
    else:
        graphs = None

    # Split bold into rois:
    roiBold = {}
    for roiId in np.unique(roiMask):
        mroi = np.where(roiMask == roiId)
        if TIME_AXIS == 0:
            roiBold[roiId] = bold[:, mroi[0], mroi[1], mroi[2]]
        else:
            roiBold[roiId] = bold[mroi[0], mroi[1], mroi[2], :]

    return graphs, roiBold, sessionScans, roiMask, mask_meta_data


def discard_bad_data(bold, roiMask, time_axis=None):
    """ Discard positions in 'roiMask' where 'bold' has zero variance or
    contains NaNs
    """
    from pyhrf.ndarray import TIME_AXIS

    if time_axis is None:
        time_axis = TIME_AXIS

    m = roiMask
    # print 'roiMask:', roiMask.shape
    # print 'bold.shape:', bold.shape
    var_bold = (bold ** 2).sum(time_axis) / (bold.shape[time_axis] - 1) - \
        bold.mean(time_axis) ** 2
    # zeroVarianceVoxels = np.bitwise_and(m>=1, bold.std(0)==0.) #TODO check
    # MemoryError ?
    zeroVarianceVoxels = np.bitwise_and(m >= 1, var_bold == 0.)
    # print 'zeroVarianceVoxels:', zeroVarianceVoxels.shape
    if zeroVarianceVoxels.any():
        logger.debug('discarded voxels (std=0):')
        logger.debug(np.where(zeroVarianceVoxels == True))
        logger.info('!! discarded voxels (std=0): %d/%d',
                    zeroVarianceVoxels.sum(), (m != 0).sum())

        # HACK
        zvv = zeroVarianceVoxels
        mzvv = np.where(zeroVarianceVoxels)
        bsh = bold.shape
        if bold.ndim == 2:
            bold[:, mzvv[0]] = np.random.randn(bsh[0], zvv.sum()) * \
                var_bold.max() ** .5
        else:
            bold[mzvv[0], mzvv[1], mzvv[2], :] = np.random.randn(zvv.sum(), bsh[3]) *\
                var_bold.max() ** .5

        zeroVarianceVoxels = np.zeros_like(zeroVarianceVoxels)

    nanVoxels = np.bitwise_and(m >= 1, np.isnan(bold.sum(time_axis)))
    # print 'nanVoxels:', nanVoxels.shape
    if nanVoxels.any():
        logger.debug('discarded voxels (nan values):')
        logger.debug(np.where(nanVoxels == True))
        logger.info('!! discarded voxels (nan values): %d/%d', nanVoxels.sum(),
                    (m != 0).sum())

    toDiscard = np.bitwise_or(zeroVarianceVoxels, nanVoxels)
    # print toDiscard.sum()
    # print 'np.bitwise_not(toDiscard):', (np.bitwise_not(toDiscard)).shape
    # print 'np.bitwise_not(toDiscard):', np.bitwise_not(toDiscard).sum()
    roiMask *= np.bitwise_not(toDiscard)


def has_ext_gzsafe(fn, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    return op.splitext(fn)[1] == ext or \
        (op.splitext(fn)[1] == '.gz' and
         op.splitext(op.splitext(fn)[0])[1] == ext)


def read_mesh(filename):
    if has_ext_gzsafe(filename, 'gii'):
        mesh_gii = gifti.read(filename)
        cor, tri = (mesh_gii.darrays[0].data, mesh_gii.darrays[1].data)
        return cor, tri, mesh_gii.darrays[0].coordsys
    else:
        raise Exception('Unsupported file format (%s)' % filename)


def write_mesh(cor, tri, filename):
    from nibabel.gifti import GiftiImage, GiftiDataArray
    nimg = GiftiImage()
    intent = 'NIFTI_INTENT_POINTSET'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(cor, intent))
    intent = 'NIFTI_INTENT_TRIANGLE'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(tri, intent))
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
        if len(tex_gii.darrays) > 1:  # 2D texture ... #TODO: check
            texture = np.vstack([a.data for a in tex_gii.darrays])
        else:
            texture = tex_gii.darrays[0].data
        return texture, tex_gii
    elif has_ext_gzsafe(tex, 'tex'):
        from pyhrf.tools._io.tio import Texture
        texture = Texture.read(tex).data
        return texture, None
    else:
        raise NotImplementedError('Unsupported %s extension'
                                  % op.splitext(tex)[1])


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
                # intent = 'dimensionless' #?fg
                #dtype = 'NIFTI_TYPE_FLOAT32'
                tex_data.astype(np.float32)
                if pyhrf.cfg['global']['write_texture_minf']:
                    s = "attributes = {'data_type' : 'FLOAT'}"
                    f = open(filename + '.minf', 'w')
                    f.write(s)
                    f.close()
            else:
                raise NotImplementedError("Unsupported dtype %s"
                                          % str(tex_data.dtype))

        gii_array = gifti.GiftiDataArray.from_array(tex_data, intent)
        if meta_data is not None:
            md = {'pyhrf_cuboid_data': meta_data}
            gmeta_data = gifti.GiftiMetaData.from_dict(md)
        else:
            gmeta_data = None
        tex_gii = gifti.GiftiImage(darrays=[gii_array, ], meta=gmeta_data)
        logger.info('Write texture to %s', filename)
        gifti.write(tex_gii, filename)

    elif has_ext_gzsafe(filename, 'tex'):
        if meta_data is not None:
            print 'Warning: meta ignored when saving to tex format'

        from pyhrf.tools._io.tio import Texture
        tex = Texture(filename, data=tex_data)
        tex.write()
    else:
        raise NotImplementedError('Unsupported extension for file %s'
                                  % filename)


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
    from pyhrf.graph import graph_from_mesh, sub_graph, graph_is_sane

    # Load ROIs:
    logger.info('load roi mask: %s', roiMaskFile)
    #roiMask = Texture.read(roiMaskFile).data.astype(int)
    roiMask, _ = read_texture(roiMaskFile)
    logger.info('roi mask shape: %s', str(roiMask.shape))

    # Load BOLD:
    lastScan = 0
    sessionScans = []
    bolds = []
    for boldFile in boldFiles:
        logger.info('load bold: %s', boldFile)
        b, _ = read_texture(boldFile)
        logger.info('bold shape: %s', str(b.shape))
        bolds.append(b)
        sessionScans.append(
            np.arange(lastScan, lastScan + b.shape[0], dtype=int))
        lastScan += b.shape[0]
    bold = np.concatenate(tuple(bolds))

    # discard bad data (bold with var=0 and nan values):
    discard_bad_data(bold, roiMask, time_axis=0)

    # Load mesh:
    logger.info('load mesh: %s', meshFile)
    coords, triangles, coord_sys = read_mesh(meshFile)
    #from soma import aims
    # mesh = aims.read(meshFile)
    # triangles = [t.arraydata() for t in mesh.polygon().list()]

    logger.info('building graph ... ')
    wholeGraph = graph_from_mesh(triangles)
    assert graph_is_sane(wholeGraph)
    logger.info('Mesh has %d nodes', len(wholeGraph))
    assert len(roiMask) == len(wholeGraph)

    assert bold.shape[1] == len(wholeGraph)

    logger.info('Computing length of edges ... ')
    edges_l = np.array([np.array([distance(coords[i], coords[n], coord_sys.xform)
                                  for n in nl])
                        for i, nl in enumerate(wholeGraph)], dtype=object)

    # Split bold and graph into rois:
    roiBold = {}
    graphs = {}
    edge_lengthes = {}
    for roiId in np.unique(roiMask):
        mroi = np.where(roiMask == roiId)
        g, nm = sub_graph(wholeGraph, mroi[0])
        edge_lengthes[roiId] = edges_l[mroi]
        graphs[roiId] = g
        roiBold[roiId] = bold[:, mroi[0]].astype(np.float32)

    return graphs, roiBold, sessionScans, roiMask, edge_lengthes


def remote_mkdir(host, user, path):
    import paramiko
    logger.info('Make remote dir %s@%s:%s ...', user, host, path)
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
    logger.info('Make remote dir %s@%s:%s ...', user, host, path)
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
        logger.info('Copying files to remote destination %s@%s:%s ...',
                    host, user, path)
        ssh = paramiko.SSHClient()
        known_hosts_file = os.path.join("~", ".ssh", "known_hosts")
        ssh.load_host_keys(os.path.expanduser(known_hosts_file))
        ssh.connect(host, username=user)
        sftp = ssh.open_sftp()
        for f in files:
            remotepath = op.join(path, op.basename(f))
            logger.info(f + ' -> ' + remotepath + ' ...')
            flocal = open(f)
            remote_file = sftp.file(remotepath, "wb")
            remote_file.set_pipelined(True)
            remote_file.write(flocal.read())
            flocal.close()
            remote_file.close()
        sftp.close()
        ssh.close()
    else:
        sfiles = string.join(['"%s"' % f for f in files], ' ')

        scp_cmd = 'scp -C %s "%s@%s:%s"' % (sfiles, user, host, path)
        logger.info('Data files transfer with scp ...')
        logger.info(scp_cmd)
        if os.system(scp_cmd) != 0:
            raise Exception('Error while scp ...')

    logger.info('Copy done!')

    return [op.join(path, op.basename(f)) for f in files]

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

        i, (aff, header) = read_volume(img_fn)
        ecodes = header.extensions.get_codes()
        if extension_codes['workflow_fwds'] in ecodes:
            ic = ecodes.index(extension_codes['workflow_fwds'])
            econtent = header.extensions[ic].get_content()
            try:
                proc_info = json.loads(econtent)
            except:
                raise IOError("Cannot safely overwrite Extension in "
                              "Header. It already has a readable "
                              "'workflow_fwds' extension, but it's "
                              "not in JSON.")
        else:
            proc_info = []

        if proc_date is None:
            proc_date = str(now())

        proc_info.append({'process_name': proc_name,
                          'process_inputs': proc_inputs,
                          'process_id': proc_id,
                          'process_version': proc_version,
                          'process_date': proc_date})

        e = Nifti1Extension('workflow_fwds', json.dumps(proc_info))
        header.extensions.append(e)

        if img_output_fn is None:
            img_output_fn = img_fn

        write_volume(i, img_output_fn, (aff, header))

    def get_process_info(img_fn):

        proc_info = None
        i, (aff, header) = read_volume(img_fn)
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

    data, h = read_volume(data_fn)
    mask, mh = read_volume(mask_fn)

    if data.ndim > mask.ndim:
        mask = mask[:, :, :, np.newaxis] * np.ones(data.shape, dtype=int)
    elif mask.ndim > data.ndim:
        raise Exception('Mask has more dims (%d) than data (%d).'
                        'Maybe data and mask were switched?'
                        % (mask.ndim, data.ndim))

    for p_id in parcels:
        data[np.where(mask == p_id)] *= -1

    print 'Saved flipped data to:', output_data_fn
    write_volume(data, output_data_fn, h)


def remove_parcels_in_file(mask_fn, parcels, output_data_fn):

    mask, mh = read_volume(mask_fn)

    for p_id in parcels:
        mask[np.where(mask == p_id)] = 0

    print 'Saved flipped data to:', output_data_fn
    write_volume(mask, output_data_fn, mh)


def crop_data_file(data_fn, mask_fn, output_data_fn):

    data, h = read_volume(data_fn)
    mask, mh = read_volume(mask_fn)

    if data.ndim > mask.ndim:
        mask = mask[:, :, :, np.newaxis] * np.ones(data.shape, dtype=int)
    elif mask.ndim > data.ndim:
        raise Exception('Mask has more dims (%d) than data (%d).'
                        'Maybe data and mask were switched?'
                        % (mask.ndim, data.ndim))

    cropped_data = crop_array(data, mask)

    if 1:
        print 'cropped_data:', cropped_data.shape
        print 'input_data_fn:', data_fn
        print 'output_data_fn:', output_data_fn

    if len(h[1].extensions) > 0:
        # remove cuboid nifti extension
        h[1].extensions.pop()

    write_volume(cropped_data, output_data_fn, h)
