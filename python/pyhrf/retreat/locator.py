"""
Anatomical locations from literature
"""
# TODO: factorize abbreviation and mni parameters
# TODO: add function mapping harvard oxford regions to their functions
# TODO: rely on Broadman areas and their functions
# TODO: add spheres masks generator and plotter from alex's functions
import numpy as np

import nibabel
from nilearn import datasets
from nipy.labs.viz_tools.coord_tools import coord_transform
from nipy.labs.viz_tools.maps_3d import mni_sform_inv


def mni_to_tal(x, y, z):
    """ Transforms coordinates from MNI space to Talairach space"""
# (z >=0): x' = 0.9900x, y' = 0.9688y + 0.0460z, z' = -0.0485y + 0.9189z
# (Duncan et al., 2000;
# http://www.mrc-cbu.cam.ac.uk/Imaging/Common/mnispace.shtml
    pass


def meta_auditory(abbreviate=True, mni=True):
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


def meta_attention(abbreviate=True, mni=True):
# TODO: get sepearately DAN and VAN from 
# Breakdown of Functional Connectivity in Frontoparietal Networks
# Underlies Behavioral Deficits in Spatial Neglect, Biyu J. He1, Abraham Z.
# Snyder1, 2, Justin L. Vincent1, Adrian Epstein1, Gordon L. Shulman2, Maurizio
# Corbetta. Neuron 2007
    whole_names = ['left ventral intra-parietal sulcus',
                   'right ventral intra-parietal sulcus',
                   'right temporoparietal junction',
                   'right dorsolateral prefrontal cortex',
                   'left posterior intraparietal sulcus',
                   'right posterior intraparietal sulcus',
                   'left middle temporal region',
                   'right middle temporal region',
                   'left frontal eye field',
                   'right frontal eye field']
    abbrevs = [' L vIPS', ' R vIPS', ' R TPJ', ' R DLPFC',
               ' L pIPS', ' R pIPS', ' L MT', ' R MT', ' L FEF', ' R FEF']
    coords_talairach = [(-24, -69, 30), (30, -80, 16), (49, -50, 28),
                        (43, 22, 34), (-25, -63, 47), (23, -65, 48),
                        (-43, -70, -3), (42, -68, -6),
                        (-26, -9, 48), (32, -9, 48)]
    coords_mni = [(-24, -73, 31), (29, -83, 14), (49, -52, 28),
                  (44, 21, 34), (-25, -67, 52), (22, -70, 54),
                  (-44, -70, -10), (41, -68, -13),
                  (-26, -12, 52), (32, -13, 52)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def meta_default_mode(abbreviate=True, mni=True):
    """Default mode regions from Scale-Free Properties of the Functional
    Magnetic Resonance Imaging Signal during Rest and Task, He BJ. The Journal
    of Neuroscience (2011).
    """
    whole_names = ['left angular gyrus',
                   'right angular gyrus',
                   'left superior frontal gyrus',
                   'right superior frontal gyrus',
                   'Posterior cingulate cortex',
                   'Medial prefrontal cortex',
                   'Frontopolar cortex']
    abbrevs = [' L AG', ' R AG', ' L SFG', ' R SFG', ' PCC', ' MPFC', ' FP']
    coords_talairach = [(-51, -54, 30), (45, -66, 27),
                        (-15, 33, 48), (18, 27, 48), (-6, -45, 33),
                        (-6, 51, -9), (-3, 45, 36)]
    coords_mni = [(-52, -56, 30), (44, -69, 27),
                  (-16, 31, 53), (18, 24, 52), (-7, -48, 35),
                  (-6, 57, -8), (-3, 45, 40)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def meta_non_cortical(abbreviate=True, mni=True):
    whole_names = ['left thalamus',
                   'right thalamus',
                   'right Cerebellum',
                   'left hippocampal formation',
                   'right hippocampal formation']
    abbrevs = [' L thalamus', ' R thalamus', ' R Cerebellum', ' L HF', ' R HF']
    coords_talairach = [(-15, -21, 6), (9, -18, 9), (21, -54, -21),
                        (-21, -25, -14), (23, -23, -14)]
    coords_mni = [(-16, -20, 3), (10, -18, 7), (21, -53, -30),
                  (-22, -24, -21), (24, -22, -21)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def meta_saliency(abbreviate=True, mni=True):
    whole_names = ['right frontoinsular cortex',
                   'Dorsal anterior cingulate cortex']
    abbrevs = [' R FI', ' dACC']
    coords_talairach = [(36, 21, -6), (-1, 10, 46)]
    coords_mni = [(38, 25, -12), (-2, 7, 50)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def meta_visual(abbreviate=True, mni=True):
    whole_names = ['left ventral primary visual cortex',
                   'right ventral primary visual cortex',
                   'left dorsal primary visual cortex',
                   'right dorsal primary visual cortex']
    abbrevs = [' L vRetino', ' R vRetino', ' L dRetino', ' R dRetino']
    coords_talairach = [(-15, -75, -9), (15, -75, -9),
                        (-6, -75, 9), (9, -75, 12)]
    coords_mni = [(-16, -77, -16), (14, -77, -16),
                  (-7, -78, 6), (9, -78, 10)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def meta_motor0(abbreviate=True, mni=True):
# ToDO complete for right motor regions
# motor regions: Biswal, B., Yetkin, F.Z., Haughton, V.M., Hyde, J.S., 1995.
# Functional connectivity in the motor cortex of resting human brain using
# echo-planar MRI. Magn. Reson. Med. 34, 537-54
# left and right motor cortex (-42, -25,63) and (42, -25, 63)
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


def meta_motor(abbreviate=True, mni=True):
    """Motor regions from: Three-dimensional locations and boundaries of
    motor and premotor cortices as defined by functional brain imaging:
    A meta-analysis Mary A. Mayka, Daniel M. Corcos, Sue E. Leurgans, and David
    E. Vaillancou. Neuroimage 2006.
    """
    whole_names = ['mesial premotor cortex',
                   'pre-supplementary motor area',
                   'supplementary motor area proper',
                   'lateral premotor cortex',
                   'premotor dorsal', 'premotor ventral',
                   'sensorimotor cortex', 'primary motor cortex',
                   'primary somatosensory cortices']
    abbrevs = [' MPMC', ' pre-SMA', ' SMA proper', ' LPMC', ' PMd', ' PMv',
               ' SMC', ' M1', ' S1']
    coords_talairach = [(-2, -1, 54), (-3, -6, 53), (-2, -7, 55),
                        (-26, -6, 56), (-30, -4, 58), (-50, 5, 22),
                        (-39, -21, 54), (-37, -21, 58), (-40, -24, 50)]
    coords_mni = [(-3, -5, 60), (-4, -10, 59), (-3, -11, 61),
                  (-26, -10, 62), (-30, -8, 64), (-52, 7, 22),
                  (-39, -24, 59), (-37, -25, 64), (-40, -27, 54)]

    if abbreviate:
        names = abbrevs
    else:
        names = whole_names

    if mni:
        coords = coords_mni
    else:
        coords = coords_talairach

    return names, coords


def func_working_memory(abbreviate=True, mni=True):
    """Working memory coordinates from servier Placebo study"""
# TODO: check conversion: here done with wfu pickatlas
    whole_names = ['left inferior parietal lobule',
                   'left middle frontal gyrus',
                   'right middle frontal gyrus',
                   'left cerebellum posterior Lobe',
                   'right cerebellum posterior Lobe peak1',
                   'right cerebellum posterior Lobe peak2',
                   'left Thalamus']
    abbrevs = [' L IPL', ' L MFG', ' R MFG', ' L CPL', ' R CPL1', ' R CPL2',
               ' L Th']
    coords_talairach = [(-26.73, -60.16, 39.86),
                        (-26.73, 10.41, 52.9), (5.94, 24.26, 38.4),
                        (32.67, -63.34, -23.74), (11.88, -77.75, -20.5),
                        (-35.64, -63.22, -21.23), (-11.88, -17.81, 12.86)]
    coords_mni = [(-27, -64, 40),
                  (-27, 8, 58), (6, 23, 43),
                  (33, -64, -32), (12, -79, -29),
                  (-36, -64, -29), (-12, -19, 13)]

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
