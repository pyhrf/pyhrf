"""
Anatomical locations from literature
"""
# TODO: factorize abbreviation and mni parameters
# TODO: add function mapping harvard oxford regions to their functions
import numpy as np

import nibabel
from nilearn import datasets
from nipy.labs.viz_tools.coord_tools import coord_transform
from nipy.labs.viz_tools.anat_cache import mni_sform_inv


def anat_auditory(abbreviate=True, mni=True):
# TODO : check for meta-analyses coordinates
    """ Non-primary auditory field areas from: Spectral and Temporal Processing
    in Human Auditory Cortex. Deborah A Hall et al., Cereb. Cortex (2002).
    Primary auditory areas from: Event-Related fMRI of the Auditory Cortex.
    Pascal Belin et al. NeuroImage (1999).
    """
    whole_names = ['primary auditory cortex left',
                   'primary auditory cortex right',
                   'non-primary auditroy left anterior area',
                   'non-primary auditroy right anterior area',
                   'non-primary auditroy left posterior area',
                   'non-primary auditroy right posterior area',
                   'non-primary auditroy left lateral area',
                   'non-primary auditroy right lateral area',
                   'non-primary auditroy left medial area',
                   'non-primary auditroy right medial area',
                   'non-primary auditroy left superior temporal area',
                   'non-primary auditroy right superior temporal area']
    abbrevs = ['L A1', 'R A1', 'L AA', 'R AA', 'L PA', 'R PA', 'L LA', 'R LA',
               'L MA', 'R MA', 'L STA', 'R STA']
# TODO: check if mni or talairach for non-primary regions and convert
    coords_talairach = [(-41, -28, 13), (44, -22, 11),
                        (-51.1, -1.8, 5.8), (56.3, -4.1, 3.7),
                        (-41.5, -30.3, 7.8), (41.8, -28.1, 12.3),
                        (-58.6, -25.3, 6.8), (62.8, -22.1, 7.7),
                        (-44.1, -17.3, 2.3), (47.8, -14.6, -1.3),
                        (-63.6, -26.8, -0.2), (68.3, -25.1, 8.7)]
    coords_mni = [(-43, -28, 11), (46, -22, 8),
                  (-51.1, -1.8, 5.8), (56.3, -4.1, 3.7),
                  (-41.5, -30.3, 7.8), (41.8, -28.1, 12.3),
                  (-58.6, -25.3, 6.8), (62.8, -22.1, 7.7),
                  (-44.1, -17.3, 2.3), (47.8, -14.6, -1.3),
                  (-63.6, -26.8, -0.2), (68.3, -25.1, 8.7)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def anat_motor(abbreviate=True, mni=True):
# ToDO complete for right motor regions
    """Motor left regions from: Breakdown of functional connectivity in
    frontoparietal networks under-lies behavioral deficits in spatial neglect.
    He BJ et al. Neuron (2007).
    Broca area from: Mapping syntax using imaging: prospects and
    problems for the study of neurolinguistic computation. Embick D et al.
    Encyclopedia of language and linguistics (2006).
    """
    whole_names = ['left second somatosensory area',
                   'left primary motor cortex', 'Broca area']
    abbrevs = [' LSII', ' L motor', ' Broca']
    coords_talairach = [(-57, -27, 21), (-39, -27, 48), (-42, 13, 14)]
    coords_mni = [(-60, -28, 20), (-39, -30, 52), (-44, 15, 13)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def locate_harvard_oxford(coords, atlas='cort-maxprob-thr25-1mm'):
    """Locates coordinates in MNI space in Harvard-Oxford atlas"""
    atlas_file, labels = datasets.fetch_harvard_oxford(atlas)
#   TODO: check in demo_plot of nipy.labs.viz_tools
    atlas_img = nibabel.load(atlas_file)
    data = atlas_img.get_data()
#    affine = atlas_img.get_affine()
#    _, affine = _xyz_order(data, affine)
#    x_map, y_map, z_map = [int(np.round(c)) for c in
#                           coord_transform(coords[0],
#                                           coords[1],
#                                           coords[2],
#                                           np.linalg.inv(affine))]
    x_map, y_map, z_map = [int(np.round(c)) for c in
                           coord_transform(coords[0],
                                           coords[1],
                                           coords[2],
                                           mni_sform_inv)]
    label_value = data[x_map, y_map, z_map]

    # Deal with missing label values
    label_values = np.unique(data.ravel())
    label = labels[label_values == label_value][0]
    return label


def get_maxima(cluster, remove_background=True, min_distance=0):
# TODO: return the function of the region and not the anatomical name
    """
    Parameters
    ==========
    clusters : dict {'size': float, 'maxima': array, 'depth': array}
        Cluster, first output of nipy.labs.statistical_mapping.cluster_stats.
        Local maxima are sorted by descending depth order.

    remove_background : bool, optional
        If True, regions located in background are not included.

    min_distance : float, optional
        Minimal distance below which maxima with the same labels are
        merged and only coordinates of the most superficial maximum are
        returned.

    Returns
    =======
    names : list of str
        The names of the cluster local maxima

    coords : list of 3-tuples
        The xyz coordinates of the local maxima
    """
    regions = []
    coords = []
    for maximum in cluster['maxima']:
        xyz = tuple(maximum)
        region = locate_harvard_oxford(xyz)
        add_region = True
        if region == 'Background':
            region = locate_harvard_oxford(
                xyz, atlas='sub-maxprob-thr25-1mm')
            if 'Cerebral' in region:
                region = locate_harvard_oxford(
                    xyz, atlas='cort-maxprob-thr0-1mm')
        if region == 'Background' and remove_background:
            add_region = False
        if region in regions and min_distance > 0:
            previous_xyz = coords[regions == region]
            distance = np.linalg.norm(np.array(xyz) - np.array(previous_xyz))
            if distance < min_distance:
                add_region = False

        if add_region:
            regions.append(region)
            coords.append(xyz)
    return regions, coords
