# -*- coding: utf-8 -*-

import os
import os.path as op
import tempfile
import logging

import numpy as np

import pyhrf

from pyhrf.graph import graph_from_mesh, bfs_set_label, bfs_sub_graph
from pyhrf.tools import add_suffix, non_existent_file
from pyhrf.tools._io import (read_texture, write_texture, read_spatial_resolution,
                             read_volume, write_volume)


logger = logging.getLogger(__name__)


def mesh_contour(coords, triangles, labels):
    bridge_triangles = []
    selected_nodes = set()
    for tri in triangles:
        l = labels[tri[0]]
        if (labels[tri[1:]] != l).any():
            bridge_triangles.append(tri)
            selected_nodes.update(tri)

    selected_nodes = sorted(list(selected_nodes))
    reorder = np.zeros(len(coords))
    reorder[selected_nodes] = range(len(selected_nodes))
    bridge_triangles = [reorder[t] for t in bridge_triangles]

    return coords[selected_nodes], np.array(bridge_triangles, dtype=int)


def mesh_contour_with_files(input_mesh, input_labels, output_mesh=None,
                            output_labels=None):
    """ TODO: use nibabel here """
    #from gifti import loadImage, saveImage, GiftiDataArray, GiftiImage
    #from gifti import GiftiImage_fromarray, GiftiImage_fromTriangles
    #from gifti import GiftiIntentCode, GiftiEncoding

    labels = loadImage(input_labels).arrays[0].data.astype(int)
    cor, triangles = loadImage(input_mesh).arrays

    contour_cor, contour_tri = mesh_contour(cor.data, triangles.data.astype(int),
                                            labels)
    k = GiftiImage_fromarray(contour_cor)
    k.arrays[0].intentString = "NIFTI_INTENT_POINTSET"
    k.addDataArray_fromarray(
        contour_tri, GiftiIntentCode.NIFTI_INTENT_TRIANGLE)
    for a in k.arrays:
        a.encoding = GiftiEncoding.GIFTI_ENCODING_ASCII

    if output_mesh is None:
        output_mesh = non_existent_file(add_suffix(input_mesh, '_contour'))
    logger.info('saving to %s', output_mesh)
    k.save(output_mesh)


def extract_sub_mesh(cor, tri, center_node, radius):
    g = graph_from_mesh(tri)
    subg, kept_nodes = bfs_sub_graph(g, center_node, radius)
    subcor = cor[kept_nodes]
    n_idx = np.zeros(len(g), dtype=int)
    keep = np.zeros(len(g), dtype=bool)
    keep[kept_nodes] = True
    n_idx[kept_nodes] = range(len(kept_nodes))

    new_tri = np.vstack([n_idx[t] for t in tri if keep[t].all()])
    # new_tri = []
    # for t in tri.data:
    #     if keep[t].all():
    #         new_tri.append(n_idx[t])
    #new_tri = np.vstack(new_tri)

    return subcor, new_tri


def extract_sub_mesh_with_files(input_mesh, center_node, radius,
                                output_mesh=None):
    from nibabel import gifti
    from nibabel.gifti import GiftiImage, GiftiDataArray
    from pyhrf.tools._io import read_mesh
    cor, tri, coord_sys = read_mesh(input_mesh)
    sub_cor, sub_tri = extract_sub_mesh(cor, tri, center_node, radius)

    #nimg = GiftiImage_fromTriangles(sub_cor, sub_tri)
    nimg = GiftiImage()
    intent = 'NIFTI_INTENT_POINTSET'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(sub_cor, intent))
    intent = 'NIFTI_INTENT_TRIANGLE'
    nimg.add_gifti_data_array(GiftiDataArray.from_array(sub_tri, intent))

    if output_mesh is None:
        output_mesh = non_existent_file(add_suffix(input_mesh, '_sub'))
    logger.info('Saving extracted mesh to %s', output_mesh)
    gifti.write(nimg, output_mesh)
    return sub_cor, sub_tri, coord_sys


# Projection onto surface

def create_projection_kernels(input_mesh, output_kernel, resolution,
                              geod_decay=5., norm_decay=2., size=7):
    logger.info('Create projection kernels (%s) ...', output_kernel)
    logger.info('Call AimsFunctionProjection -op 0 ...')

    projection = ['AimsFunctionProjection',
                  '-m', input_mesh,
                  '-o', output_kernel,
                  '-i', size,
                  #'-I', 801,
                  #'--debugLevel', 10,
                  #'--verbose', 10,
                  '-vx', resolution[0],
                  '-vy', resolution[1],
                  '-vz', resolution[2],
                  '-g', geod_decay,
                  '-n', norm_decay,
                  '-op', '0',
                  '-t', 0
                  ]
    os.system(' '.join(map(str, projection)))


def project_fmri(input_mesh, data_file, output_tex_file,
                 output_kernels_file=None, data_resolution=None,
                 geod_decay=5., norm_decay=2., kernel_size=7,
                 tex_bin_threshold=None):

    if output_kernels_file is None:
        tmp_dir = tempfile.mkdtemp(prefix='pyhrf_surf_proj',
                                   dir=pyhrf.cfg['global']['tmp_path'])

        kernels_file = op.join(tmp_dir, add_suffix(op.basename(data_file),
                                                   '_kernels'))
        tmp_kernels_file = True
    else:
        kernels_file = output_kernels_file
        tmp_kernels_file = False

    if data_resolution is not None:
        resolution = data_resolution
    else:
        resolution = read_spatial_resolution(data_file)

    logger.info('Data resolution: %s', resolution)
    logger.info('Projection parameters:')
    logger.info('   - geodesic decay: %f mm', geod_decay)
    logger.info('   - normal decay: %f mm', norm_decay)
    logger.info('   - kernel size: %f voxels', kernel_size)

    create_projection_kernels(input_mesh, kernels_file, resolution,
                              geod_decay, norm_decay, kernel_size)

    project_fmri_from_kernels(input_mesh, kernels_file, data_file,
                              output_tex_file, tex_bin_threshold)

    if tmp_kernels_file:
        os.remove(kernels_file)


def project_fmri_from_kernels(input_mesh, kernels_file, fmri_data_file,
                              output_tex, bin_threshold=None, ):

    logger.info('Project data onto mesh using kernels ...')

    if 0:
        print 'Projecting ...'
        print 'func data:', fmri_data_file
        print 'Mesh file:', input_mesh
        print 'Save as:', output_tex

    logger.info('Call AimsFunctionProjection -op 1 ...')

    data_files = []
    output_texs = []
    p_ids = None
    if bin_threshold is not None:
        d, h = read_volume(fmri_data_file)
        if np.allclose(d.astype(int), d):
            tmp_dir = pyhrf.get_tmp_path()
            p_ids = np.unique(d)
            logger.info('bin threshold: %f', bin_threshold)
            logger.info(
                'pids(n=%d): %d...%d', len(p_ids), min(p_ids), max(p_ids))
            for i, p_id in enumerate(p_ids):
                if p_id != 0:
                    new_p = np.zeros_like(d)
                    new_p[np.where(d == p_id)] = i + 1  # 0 is background
                    ifn = op.join(tmp_dir, 'pmask_%d.nii' % p_id)
                    write_volume(new_p, ifn, h)
                    data_files.append(ifn)
                    ofn = op.join(tmp_dir, 'ptex_%d.gii' % p_id)
                    output_texs.append(ofn)
        else:
            data_files.append(fmri_data_file)
            output_texs.append(output_tex)
    else:
        data_files.append(fmri_data_file)
        output_texs.append(output_tex)

    logger.info('input data files: %s', str(data_files))
    logger.info('output data files: %s', str(output_texs))

    for data_file, o_tex in zip(data_files, output_texs):
        projection = [
            'AimsFunctionProjection',
            '-op', '1',
            '-d', kernels_file,
            '-d1', data_file,
            '-m', input_mesh,
            '-o', o_tex
        ]

        cmd = ' '.join(map(str, projection))
        logger.info('cmd: %s', cmd)
        os.system(cmd)

    if bin_threshold is not None:
        logger.info('Binary threshold of texture at %f', bin_threshold)
        o_tex = output_texs[0]
        data, data_gii = read_texture(o_tex)
        data = (data > bin_threshold).astype(np.int32)
        print 'data:', data.dtype
        if p_ids is not None:
            for pid, o_tex in zip(p_ids[1:], output_texs[1:]):
                pdata, pdata_gii = read_texture(o_tex)
                data += (pdata > bin_threshold).astype(np.int32) * pid

        #assert (np.unique(data) == p_ids).all()
        write_texture(data, output_tex, intent='NIFTI_INTENT_LABEL')
