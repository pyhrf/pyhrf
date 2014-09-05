# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import numpy as np
import os.path as op
import pyhrf
#from pottsfield import pottsfield_c
from random import randrange

from pyhrf.tools import buildPolyMat
from field import genPotts
from pyhrf.graph import graph_from_lattice, kerMask2D_4n , kerMask3D_6n

from numpy.random import randn

from scipy.signal.signaltools import lfilter
from pyhrf import paradigm
from pyhrf.boldsynth import hrf as shrf
import pyhrf.paradigm as mpar
from pyhrf.tools._io import write_volume
from pyhrf.graph import bfs_set_label

from pyhrf.tools import add_prefix
from pyhrf.ndarray import xndarray, expand_array_in_mask
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


def create_varying_hrf(hrf_duration=25., dt=.5):
    timeaxis = np.arange(0,hrf_duration,dt)
    Picw = randrange(2,4,1)
    picp = [5,1]
    #print Picw
    h = shrf.genBezierHRF(timeAxis=timeaxis, pic=picp, picw=Picw)
    #print np.array(h)[0].shape
    #raw_input('')
    return h[1]

def create_hrf(picw,pic,under=2,hrf_duration=25., dt=.5):
    timeaxis = np.arange(0,hrf_duration,dt,)
    #Picw = randrange(2,4,1)
    #picp = [5,1]
    #print Picw
    Picp = [pic,1]
    upos = [under,-.2]
    h = shrf.genBezierHRF(timeAxis=timeaxis, pic=Picp, picw=picw,ushoot=upos)
    #print np.array(h)[0].shape

    #print  hrf_duration,dt,pic,picw
    #raw_input('')
    return h[1]


def create_bold_from_stim_induced_RealNoise(stim_induced_signal, dsf, noise,
                                            drift):
    """
    Downsample stim_induced signal according to downsampling factor 'dsf' and
    add noise and drift (nuisance signals) which has to be at downsampled
    temporal resolution.
    """
    bold = stim_induced_signal[0:-1:dsf,:]
    bold += drift
    TR = bold.shape[0]
    #nbvox = bold.shape[1]
    noise = np.reshape(noise[0:TR,:,:,:],bold.shape)
    #print bold.shape
    #print noise.shape
    bold += noise

    return bold



def create_labels_Potts(condition_defs,beta,nb_voxels):
    labels = []
    nb_voxels = nb_voxels**.5 #assume square shape
    shape = (nb_voxels,nb_voxels)
    mask = np.ones(shape, dtype=int)
    graph = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)
    shape = (1,nb_voxels,nb_voxels)
    for c in condition_defs:
	tmp = genPotts(graph, beta,2 )
        tmp = np.reshape(tmp,shape)
        labels.append(tmp)
    return np.array(labels)

def create_3Dlabels_Potts(condition_defs,beta,dims,mask):
    labels = []
    graph = graph_from_lattice(mask, kerMask=kerMask3D_6n,toroidal=True)
    for c in condition_defs:
	tmp = genPotts(graph, beta,2 )
        tmp = np.reshape(tmp,dims)
        labels.append(tmp)
    return np.array(labels)

def create_AR_noise(bold_shape, v_noise, order=2, v_corr=0.1):
    noise = np.random.randn(*bold_shape)
    nbVox = bold_shape[1]
    length = bold_shape[0]
    noiseMean = np.zeros( nbVox, dtype=float)
    noiseVar = v_noise * np.random.rand(nbVox)
    ARorder = order
    noiseARp = v_corr*(np.ones((nbVox,ARorder), dtype=float) + \
                       np.random.rand(nbVox,ARorder))
    for v in xrange(nbVox):
	tmp = np.sqrt(noiseVar[v]) * np.random.randn(length) + noiseMean[v]
	ARArray = np.concatenate( ( [1], -noiseARp[v,:] ) )
	noise[:,v] = lfilter([1], ARArray,tmp)
    return noise


def create_localizer_paradigm(condition_defs, paradigm_label='av'):
    ons = eval('paradigm.onsets_loc_%s' %paradigm_label)
    cnames = [c.name for c in condition_defs]
    onsets = OrderedDict(zip(cnames, [ons[c] for c in cnames]))
    par = mpar.Paradigm(onsets, sessionDurations=[300.])
    return par

def create_localizer_paradigm_avd(condition_defs):
    ons = mpar.onsets_loc_av_d
    cnames = [c.name for c in condition_defs]
    onsets = OrderedDict(zip(cnames, [ons[c] for c in cnames]))
    par = mpar.Paradigm(onsets, sessionDurations=[300.])
    return par

def create_localizer_paradigm_a(condition_defs):
    ons = mpar.onsets_loc_a
    dur = paradigm.durations_loc_a
    cnames = [c.name for c in condition_defs]
    onsets = OrderedDict(zip(cnames, [ons[c] for c in cnames]))
    durations = OrderedDict(zip(cnames, [dur[c] for c in cnames]))

    p = mpar.Paradigm(onsets, stimDurations=durations, sessionDurations=[300.])
    return p

def create_paradigm_un_evnt(condition_defs):
    ons = mpar.onsets_un_evnt
    cnames = [c.name for c in condition_defs]
    onsets = OrderedDict(zip(cnames, [ons[c] for c in cnames]))
    paradigm = mpar.Paradigm(onsets)
    return paradigm

def create_language_paradigm(condition_defs):
    ons = mpar.onsets_language_sess1
    dur = mpar.durations_language_sess1
    cnames = [c.name for c in condition_defs]
    onsets = OrderedDict(zip(cnames, [ons[c] for c in cnames]))
    durations = OrderedDict(zip(cnames, [dur[c] for c in cnames]))
    paradigm = mpar.Paradigm(onsets, stimDurations=durations)
    return paradigm

def rasterize_paradigm(paradigm, dt, condition_defs):
    """ Return binary sequences of onsets approximated on temporal grid of
    temporal resolution dt, for all conditions. 'paradigm' is expected to be
    an instance of 'pyhrf.paradigm.mpar.Paradigm'
    """
    rparadigm = paradigm.get_rastered(dt)
    return np.vstack( [rparadigm[c.name][0] for c in condition_defs] )


def load_drawn_labels(name):
    fn = pyhrf.get_data_file_name('simu_labels_%s.png' %name)
    if not op.exists(fn):
        raise Exception('Unknown label map %s (%s)' %(name,fn))

    from scipy.misc import fromimage
    from PIL import Image
    labels = fromimage(Image.open(fn))
    return labels[np.newaxis,:,:]

def create_labels_vol(condition_defs):
    """ Create a seet labels from the field "label_map" in *condition_defs*
    Available choices for the field label_map:
    - 'random_small' : binary labels are randomly generated with shape (1,5,5)
    - a tag (str) : corresponds to a png file in pyhrf data files
    - a 3D np containing the labels
    """
    labels = []
    for c in condition_defs:
        if isinstance(c.label_map, np.ndarray):
            labels.append(c.label_map)
            if c.label_map.ndim < 3:
                raise Exception('For cond %s, label map must be 3D, ' \
                                    'got %d dim(s)' %(c.name,c.label_map.ndim))
        elif isinstance(c.label_map, str):
            if c.label_map == 'random_small':
                shape = (1,5,5)
                labels.append(np.random.randint(0,2,shape))
            else:
                labels.append(load_drawn_labels(c.label_map))
        else:
            raise Exception("label_map of type %s not supported" \
                                %c.label_map.__class__)

    return np.array(labels)

def create_connected_label_clusters(condition_defs, activ_label_graph):
    graph = activ_label_graph
    labels = np.zeros((len(condition_defs), len(graph)), dtype=int)
    for ic,c in enumerate(condition_defs):
        for center in c.centers:
            bfs_set_label(graph, center, labels[ic,:], 1, c.radius)

    return labels

def flatten_labels_vol(labels_vol):
    vol_mask = np.ones_like(labels_vol[0])
    mask = np.where(vol_mask)
    return np.vstack([labels_vol[ic][mask] for ic in range(labels_vol.shape[0])])

def create_bigaussian_nrls(labels, mean_act, var_act, var_inact):
    """
    Simulate bi-Gaussian NRLs (zero-centered inactive component)
    """
    mask_activ = np.where(labels)
    nrls = np.random.randn(*labels.shape) * var_inact**.5
    nrls[mask_activ] = np.random.randn(labels.sum()) * var_act**.5 + mean_act

    return nrls

def create_time_invariant_gaussian_nrls(condition_defs, labels):
    nrls = []
    for ic,c in enumerate(condition_defs):
        labels_c = labels[ic]
        mask_activ = np.where(labels_c)
        nrls_c = randn(labels_c.size) * c.v_inact**.5 + 0
        nrls_c[mask_activ] = randn(labels_c.sum()) * c.v_act**.5 + c.m_act
        #print 'm_act:', c.m_act
        nrls.append(nrls_c)

    return np.vstack(nrls)

def create_time_invariant_gaussian_brls(condition_defs, labels):
    """ BOLD response levels for ASL """

    nrls = []
    for ic,c in enumerate(condition_defs):
        labels_c = labels[ic]
        mask_activ = np.where(labels_c)
        nrls_c = randn(labels_c.size) * c.bold_v_inact**.5 + 0
        nrls_c[mask_activ] = randn(labels_c.sum()) * c.bold_v_act**.5 + \
          c.bold_m_act
        nrls.append(nrls_c)
    return np.vstack(nrls)

def create_time_invariant_gaussian_prls(condition_defs, labels):
    """ Perfusion response levels for ASL """

    nrls = []
    for ic,c in enumerate(condition_defs):
        labels_c = labels[ic]
        mask_activ = np.where(labels_c)
        nrls_c = randn(labels_c.size) * c.perf_v_inact**.5 + 0
        nrls_c[mask_activ] = randn(labels_c.sum()) * c.perf_v_act**.5 + \
          c.perf_m_act
        nrls.append(nrls_c)
    return np.vstack(nrls)



def create_gaussian_nrls_sessions_and_mean(nrls, condition_defs, labels,
                                           var_sess):
    '''
    Creation of nrls by session (and by voxel and cond) - for one session n° sess
    The nrls by session vary around an nrl mean (nrls_bar) defined by voxel and cond
    (var_sess corresponds to the variation of session defined nrls around the nrl_bar)
    Here "nrls" is nrls_bar, mean over subjects!
    '''
    nrls_sess=[]
    for ic,c in enumerate(condition_defs):
        labels_c = labels[ic]
        #print c
        nrls_s = randn(labels_c.size) *var_sess**.5 + nrls[ic]
        nrls_sess.append(nrls_s)
        #print nrls_sess
    return np.vstack(nrls_sess)

def createBiGaussCovarNRL(condition_defs, labels, covariance):
    mean_for_corr = [c.m_act for c in condition_defs]
    var = [c.v_act for c in condition_defs]
    sigma = np.diag(var) + covariance
    #print "sigma:', sigma
    nrls = create_time_invariant_gaussian_nrls(condition_defs, labels)

    #Definition of covariance structure between conditions,
    # for voxels at the intersection
    mask_intersection = np.prod(labels, axis=0)
    Pos_intersection  = np.where(mask_intersection)
    size = len(Pos_intersection[0])
    nrls[:,Pos_intersection[0]]  = np.random.multivariate_normal(mean_for_corr,
                                                                 sigma, size).T
    return nrls

def create_gaussian_noise(bold_shape, v_noise, m_noise=0.):
    return np.random.randn(*bold_shape) * v_noise**.5 + m_noise

def create_gaussian_noise_asl(asl_shape, v_gnoise, m_noise=0.):
    return create_gaussian_noise(asl_shape, v_gnoise, m_noise)


def create_drift_coeffs(bold_shape, drift_order, drift_coeff_var):

    return drift_coeff_var**.5 * randn(drift_order+1,bold_shape[1])


def create_polynomial_drift_from_coeffs(bold_shape, tr, drift_order,
                                        drift_coeffs, drift_mean=0.,
                                        drift_amplitude=1.):
    p = buildPolyMat(drift_order, bold_shape[0], tr)
    if 0:
        print '%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'p:', p
        print '%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'drift_coeffs:', drift_coeffs.shape
    drift = np.dot(p, drift_coeffs) * drift_amplitude + drift_mean
    pyhrf.verbose(3, 'Drift shape: %s' %str(drift.shape))
    return drift

def create_drift_coeffs_asl(asl_shape, drift_order, drift_var):
    return create_drift_coeffs(asl_shape, drift_order, drift_var)

def create_polynomial_drift_from_coeffs_asl(asl_shape, tr, drift_order,
                                            drift_coeffs):
    return create_polynomial_drift_from_coeffs(asl_shape, tr, drift_order,
                                               drift_coeffs)


def create_polynomial_drift(bold_shape, tr, drift_order, drift_var):
    p = buildPolyMat(drift_order, bold_shape[0], tr)
    nvox = bold_shape[1]
    coeff = drift_var**.5 * randn(p.shape[1],nvox)
    drift = np.dot(p, coeff)
    pyhrf.verbose(3, 'Drift shape: %s' %str(drift.shape))
    return drift

def create_null_drift(bold_shape):
    return np.zeros(bold_shape)

def get_bold_shape(stim_induced_signal, dsf):
    return stim_induced_signal[0:-1:dsf,:].shape

def calc_asl_shape(bold_stim_induced, dsf):
    return bold_stim_induced[0:-1:dsf,:].shape


def load_hrf_territories(nb_hrf_territories=0, hrf_territories_name=None):
    if hrf_territories_name is not None:
        fn = pyhrf.get_data_file_name('simu_hrf_%s.png' %hrf_territories_name)
        if not op.exists(fn): #HACK #TODO improve this
            fn = pyhrf.get_data_file_name(hrf_territories_name)

    else:
        fn = pyhrf.get_data_file_name('simu_hrf_%d_territories.png' \
                                      %nb_hrf_territories)
    assert op.exists(fn)
    from scipy.misc import fromimage
    from PIL import Image
    territories = fromimage(Image.open(fn))
    territories = territories[np.newaxis,:,:]
    t=territories[np.where(np.ones_like(territories))]
    labels=np.unique(territories)
    for ilab, lab in enumerate(labels):
        t[np.where(t==lab)]=ilab
    return t

def load_many_hrf_territories(nb_hrf_territories):
    from pyhrf.tools._io import read_volume
    fn = pyhrf.get_data_file_name('simu_hrf_%d_territories.nii' \
                        %nb_hrf_territories)
    assert op.exists(fn)
    territories = read_volume(fn)[0]
    return territories[np.where(np.ones_like(territories))]

def create_hrf_from_territories(hrf_territories, primary_hrfs):

    pyhrf.verbose(1,'create_hrf_from_territories ...')
    pyhrf.verbose(1,' inputs: hrf_territories %s, primary_hrfs (%d,%d)' \
                      %(str(hrf_territories.shape), len(primary_hrfs),
                        len(primary_hrfs[0][0])))
    assert hrf_territories.ndim == 1
    hrfs = np.zeros((hrf_territories.size,primary_hrfs[0][0].size))
    territories = np.unique(hrf_territories)
    territories.sort()
    if territories.min() == 1:
        territories = territories - 1

    assert territories.min() == 0
    assert territories.max() <= len(primary_hrfs)
    #print hrfs.shape

    #sm = ','.join(['m[%d]'%d for d in range(hrf_territories.ndim)] + [':'])
    for territory in territories:
        #TODO: test consitency in hrf lengths
        m = np.where(hrf_territories==territory)[0]
        # print 'm:',m
        # print 'hrfs[m,:].shape:', hrfs[m,:].shape
        # print 'primary_hrfs[territory][1].shape:', \
        #     primary_hrfs[territory][1].shape
        # print primary_hrfs[territory][1]
        #print hrfs[m,:].shape
        #exec('hrfs[%s] = primary_hrfs[territory][1]' %sm)
        hrfs[m,:] = primary_hrfs[territory][1]
        #print hrfs[m,:]
    return hrfs.transpose()

def duplicate_hrf(nb_voxels, primary_hrf):
    """
    Duplicate hrf over all voxels.
    Return an array of shape (nb_voxels, len(hrf))
    """
    hrfs = np.zeros((nb_voxels,len(primary_hrf)))
    hrfs[:,:] = primary_hrf
    return hrfs.transpose()

def create_alpha_for_hrfgroup(alpha_var):
    '''
    Create alpha from a normal distribution, for one subject
    '''
    alpha = np.random.randn(1) * alpha_var**.5 + 0.
    return alpha[0]

def create_gaussian_hrf_subject(hrf_group, var_subject_hrf, dt, alpha=0.0):
    '''
    Creation of hrf by subject.
    Use group level hrf and variance for each subject
    (var_subjects_hrfs must be a list)
    Simulated hrfs must be smooth enough: correlation between temporal coeffcients
    '''

    nb_coeff=hrf_group.shape[0]

    varR = shrf.genGaussianSmoothHRF(False, nb_coeff, dt, var_subject_hrf)[1]
    Matvar = varR/var_subject_hrf
    D1 = np.eye(nb_coeff, k=1) - np.eye(nb_coeff, k=-1)
    hrf_group = hrf_group + alpha*np.dot(D1, hrf_group)

    #Matvar = np.eye(nb_coeff)/var_subject_hrf #do not regularize in this case

    hrf_s = np.random.multivariate_normal(hrf_group,np.linalg.inv(Matvar))
    #hrf_s = randn(nb_coeff) *var_subject_hrf**.5 + hrf_group
    normHRF = (hrf_s**2).sum()**(0.5)
    hrf_s /= normHRF

    return hrf_s


def duplicate_brf(nb_voxels, primary_brf):
    """
    Duplicate brf over all voxels.
    Return an array of shape (nb_voxels, len(brf))
    """
    brfs = np.zeros((nb_voxels,len(primary_brf)))
    brfs[:,:] = primary_brf
    return brfs.transpose()


def duplicate_prf(nb_voxels, primary_prf):
    """
    Duplicate prf over all voxels.
    Return an array of shape (nb_voxels, len(prf))
    """
    prfs = np.zeros((nb_voxels,len(primary_prf)))
    prfs[:,:] = primary_prf
    return prfs.transpose()


def duplicate_noise_var(nb_voxels, v_gnoise):
    """
    Duplicate variance of noise over all voxels.
    Return an array of shape (nb_voxels, var noise)
    """
    vars_n = np.zeros((nb_voxels,np.size(v_gnoise)))
    vars_n[:,:] = v_gnoise
    return vars_n.transpose()[0]

def create_canonical_hrf(hrf_duration=25., dt=.5):
    return shrf.getCanoHRF(hrf_duration, dt)[1]

def create_gsmooth_hrf(hrf_duration=25., dt=.5, order=2, hrf_var=1., zc=True,
                       normalize_hrf=True):
    """ Create a smooth HRF according to the multivariate gaussian prior
    used in JDE
    *hrf_duration* and *dt* are the HRF duration and temporal resolution,
    respectively (in sec.).
    *order* is derivative order constraining the covariance matrix.
    *hrf_var* is the HRF variance.
    *zc* is a flag to impose zeros at the begining and the end of the HRF

    return:
    a np array of HRF coefficients
    """
    tAxis = np.arange(0, hrf_duration+dt, dt)
    prcov = len(tAxis) - 2*zc
    matQ = shrf.buildFiniteDiffMatrix(order, prcov)
    matQ = np.divide(np.dot(matQ.transpose(),matQ), dt**(order**2))
    matL = np.array(np.transpose(np.linalg.cholesky(matQ/hrf_var)))

    hrf = np.linalg.solve(matL, np.random.randn(prcov,1))
    #hrf = np.sqrt(rh)*hrf
    if zc :
        hrf = np.concatenate(([[0]],hrf,[[0]]))

    if normalize_hrf:
        hrf /= (hrf**2).sum()**.5

    return hrf.squeeze()


def create_prf(prf_duration=25., dt=.5):
    tAxis = np.arange(0,prf_duration+dt, dt)
    tPic = 5.
    tus = 16.
    prf = shrf.genBezierHRF(timeAxis=tAxis, pic=[tPic,1], picw=2.5,
                            ushoot=[tus, 0.],
                       normalize=True)[1]
    assert len(prf) == len(tAxis)
    return prf

# def create_canonical_hrf(nb_voxels, dt):
#     primary_hrf = shrf.getCanoHRF(dt=dt)
#     hrfs = np.zeros((nb_voxels,len(primary_hrf)))
#     hrfs[:,:] = primary_hrf
#     return hrfs.transpose()

def create_stim_induced_signal(nrls, rastered_paradigm, hrf, dt):
    """
    Create a stimulus induced signal from neural response levels, paradigm and
    HRF (sum_{m=1}^M a^m X^m h)
    For each condition, compute the convolution of the paradigm binary sequence
    'rastered_paradigm' with the given HRF and multiply by nrls. Finally
    compute the sum over conditions.

    Return a bold array of shape (nb scans, nb voxels)
    """

    #print 'hrf:', hrf
    #print 'paradigm:', rastered_paradigm
    npos = nrls.shape[1]
    duration_dt = hrf.shape[0] + len(rastered_paradigm[0])-1
    bold = np.zeros((duration_dt , npos))
    for ipos in xrange(npos):
        if hrf.ndim == 2:
            h = hrf[:,ipos]
        else:
            h = hrf #single HRF
        for ic in xrange(nrls.shape[0]):
            activity = np.array(rastered_paradigm[ic,:], dtype=float)
            activity[np.where(activity==1)] = nrls[ic,ipos]
            #print 'activity:', activity.shape
            #print 'h:', h.shape
            bold[:,ipos] += np.convolve(activity, h)

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.plot(bold[::4,0])
    #plt.show()
    return bold

def create_Xh(nrls, rastered_paradigm, hrf, condition_defs,
                               dt, hrf_territories=None):
    """
    Retrieve the product X.h
    """
    npos = nrls.shape[1]
    duration_dt = len(hrf[:,0])+len(rastered_paradigm[0])-1
    Xh = np.zeros((duration_dt , npos))
    for ipos in xrange(npos):
        h = hrf[:,ipos]
        for ic,c in enumerate(condition_defs):
            activity = np.array(rastered_paradigm[ic,:], dtype=float)
            Xh[:,ipos] += np.convolve(activity, h)

    return Xh


def create_multisess_stim_induced_signal(nrls_session, rastered_paradigm,
                                         hrf, condition_defs,
                                         dt, hrf_territories=None):
    """
    Create a stimulus induced signal from neural response levels, paradigm and
    HRF (sum_{m=1}^M a^m X^m h)
    For each condition, compute the convolution of the paradigm binary sequence
    'rastered_paradigm' with the given HRF and multiply by nrls. Finally
    compute the sum over conditions.

    Return a bold array of shape (nb scans, nb voxels)
    """
    #print 'hrf:', hrf
    #print 'paradigm:', rastered_paradigm
    npos = nrls_session.shape[1]
    duration_dt = len(hrf[:,0])+len(rastered_paradigm[0])-1
    bold = np.zeros((duration_dt , npos))
    for ipos in xrange(npos):
        h = hrf[:,ipos]
        for ic,c in enumerate(condition_defs):
            activity = np.array(rastered_paradigm[ic,:], dtype=float)
            activity[np.where(activity==1)] = nrls_session[ic,ipos]
            #print 'activity:', activity.shape
            #print 'h:', h.shape
            bold[:,ipos] += np.convolve(activity, h)

    return bold


def create_bold_stim_induced_signal(brls, rastered_paradigm, brf, condition_defs,
                                    dt, hrf_territories=None):
    """
    Create a stimulus induced signal for ASL from BOLD response levels,
     paradigm and BRF
    (sum_{m=1}^M a^m X^m h + sum_{m=1}^M c^m W X^m g)
    For each condition, compute the convolution of the paradigm binary sequence
    'rastered_paradigm' with the given BRF and multiply by brls.
    Finally compute the sum over conditions.

    Return a asl array of shape (nb scans, nb voxels)
    """
    #print 'hrf:', hrf
    #print 'paradigm:', rastered_paradigm
    npos = brls.shape[1]
    duration_dt = len(brf[:,0]) + len(rastered_paradigm[0]) - 1
    asl = np.zeros((duration_dt , npos))

    for ipos in xrange(npos):
        h = brf[:,ipos]
        for ic,c in enumerate(condition_defs):
            activity = np.array(rastered_paradigm[ic,:], dtype=float)
            activity[np.where(activity==1)] = brls[ic,ipos]

            #print 'activity:', activity.shape
            #print 'h:', h.shape
            asl[:,ipos] += np.convolve(activity, h)
    return asl

def create_perf_baseline(asl_shape, perf_baseline_var, perf_baseline_mean=0.):
    n = asl_shape[0]
    nvox = asl_shape[1]
    rnd = np.random.randn(nvox)*perf_baseline_var**.5 + perf_baseline_mean
    return np.abs(np.tile(rnd, (n,1)))

def create_perf_stim_induced_signal(prls, rastered_paradigm, prf, condition_defs,
                                    dt, hrf_territories=None):
    """
    Create a stimulus induced signal for ASL from perfusion response levels,
     paradigm and PRF
    (sum_{m=1}^M c^m X^m g)
    For each condition, compute the convolution of the paradigm binary sequence
    'rastered_paradigm' with the given PRF and multiply by prls.
    Finally compute the sum over conditions.

    Return a asl array of shape (nb scans, nb voxels)
    """
    #print 'hrf:', hrf
    #print 'paradigm:', rastered_paradigm
    npos = prls.shape[1]
    duration_dt = len(prf[:,0])+len(rastered_paradigm[0])-1
    asl = np.zeros((duration_dt , npos))

    for ipos in xrange(npos):
        g = prf[:,ipos]
        for ic,c in enumerate(condition_defs):
            activity = np.array(rastered_paradigm[ic,:], dtype=float)
            activity[np.where(activity==1)] = prls[ic,ipos]

            #print 'activity:', activity.shape
            #print 'h:', h.shape
            asl[:,ipos] += np.convolve(activity, g)
    return asl



def create_stim_induced_signal_Parsi(nrls, rastered_paradigm, hrf,
                                     condition_defs, dt, w):
    """
    Create a stimulus induced signal from neural response levels, paradigm and
    HRF (sum_{m=1}^M a^m w^m X^m h)
    For each condition, compute the convolution of the paradigm binary sequence
    'rastered_paradigm' with the given HRF and multiply by nrls and W. Finally
    compute the sum over conditions.

    Return a bold array of shape (nb scans, nb voxels)
    """
    npos = nrls.shape[1]
    duration_dt = len(hrf[:,0])+len(rastered_paradigm[0])-1
    bold = np.zeros((duration_dt , npos))
    for ipos in xrange(npos):
        h = hrf[:,ipos]
        for ic,c in enumerate(condition_defs):
            if w[ic]==1:
                activity = np.array(rastered_paradigm[ic,:], dtype=float)
            else:
                activity = np.zeros(rastered_paradigm[ic,:].shape, dtype=float)

            activity[np.where(activity==1)] = nrls[ic,ipos]
            bold[:,ipos] += np.convolve(activity, h)

            return bold

def create_outliers(bold_shape, stim_induced_signal, nb_outliers, outlier_scale=5.):

    #print 'test create outliers'
    nscans, nvoxels = bold_shape
    #print 'bold_shape:', bold_shape
    outliers = np.zeros(bold_shape, dtype=stim_induced_signal.dtype)
    outliers_idx = np.arange(nscans)

    max_value = stim_induced_signal.max() * 5.
    for ivoxel in xrange(nvoxels):
        np.random.shuffle(outliers_idx)
        mask_outliers = outliers_idx[:nb_outliers]
        outliers[mask_outliers, ivoxel] = max_value

    return outliers

def create_bold(stim_induced_signal, dsf, noise, drift=None,
                outliers=None):
    """
    a short-cut for function create_bold_from_stim_induced
    """
    return create_bold_from_stim_induced(stim_induced_signal, dsf, noise, drift,
                                         outliers)

def create_bold_from_stim_induced(stim_induced_signal, dsf, noise, drift=None,
                                  outliers=None):
    """
    Downsample stim_induced signal according to downsampling factor 'dsf' and
    add noise and drift (nuisance signals) which has to be at downsampled
    temporal resolution.
    """
    bold = stim_induced_signal[0:-1:dsf,:].copy()
    pyhrf.verbose(3, 'create_bold_from_stim_induced ...')
    pyhrf.verbose(3, 'bold shape: %s, noise shape: %s' %(str(bold.shape),
                                                         str(noise.shape)))
    if drift is not None:
        pyhrf.verbose(3, 'drift shape: %s' %str(drift.shape))
        bold += drift
    if outliers is not None:
        bold += outliers
    bold += noise

    return bold

def create_bold_controlled_variance(stim_induced_signal, alpha, nb_voxels, dsf, nrls,
                                    Xh, drift=None, outliers=None):
    """
    Create BOLD with controlled explained variance
    alpha: percentage of explained variance on total variance
    """
    bold = stim_induced_signal[0:-1:dsf,:].copy()
    bold_shape = bold.shape

    var_stim_induced = stim_induced_signal.var(0)

    if drift is not None:
        var_drift = drift.var(0)
        var_noise = (1-alpha)/alpha * var_stim_induced - var_drift
        vars_n = np.zeros((nb_voxels, np.size(var_noise)))
        vars_n[:,:] = var_noise
        vars_noise = vars_n.transpose()[0]
        noise = np.random.randn(*bold_shape) * vars_noise**.5
        bold+=drift
    else:
        var_noise = (1-alpha)/alpha * var_stim_induced
        vars_n = np.zeros((nb_voxels, np.size(var_noise)))
        vars_n[:,:] = var_noise
        vars_noise = vars_n.transpose()[0]
        noise = np.random.randn(*bold_shape) * vars_noise**.5

    if outliers is not None:
        bold += outliers

    bold += noise

    return bold

def build_ctrl_tag_matrix(asl_shape):
    units = np.ones(asl_shape[0]) * -1
    units[::2] = 1
    #self.w[::2] = 1/2.
    return np.diag(units)

def create_asl_from_stim_induced(bold_stim_induced, perf_stim_induced,
                                 ctrl_tag_mat, dsf,
                                 perf_baseline, noise, drift=None, outliers=None):
    """
    Downsample stim_induced signal according to downsampling factor 'dsf' and
    add noise and drift (nuisance signals) which has to be at downsampled
    temporal resolution.
    """
    bold = bold_stim_induced[0:-1:dsf,:].copy()
    perf = np.dot(ctrl_tag_mat, (perf_stim_induced[0:-1:dsf,:].copy() + \
                                 perf_baseline))

    pyhrf.verbose(3, 'create_asl_from_stim_induced ...')
    pyhrf.verbose(3, 'bold shape: %s, perf_shape: %s, noise shape: %s, '\
                  'drift shape: %s' %(str(bold.shape), str(perf.shape),
                                      str(noise.shape), str(drift.shape)))

    asl = bold + perf
    if drift is not None:
        asl += drift
    if outliers is not None:
        asl += outliers
    asl += noise

    return asl


def save_simulation(simulation, output_dir):
    """ short-hand for simulation_save_vol_outputs
    """
    simulation_save_vol_outputs(simulation, output_dir)

def simulation_save_vol_outputs(simulation, output_dir, bold_3D_vols_dir=None,
                                simulation_graph_output=None, prefix=None,
                                vol_meta=None):
    """ simulation_graph_output : None, 'simple', 'thumbnails' #TODO
    """

    if simulation.has_key('paradigm'):
        fn = add_prefix(op.join(output_dir, 'paradigm.csv'), prefix)
        simulation['paradigm'].save_csv(fn)

    # Save all volumes in nifti format:
    if simulation.has_key('labels_vol'):
        mask_vol = np.ones_like(simulation['labels_vol'][0])
    elif simulation.has_key('mask'):
        mask_vol = simulation.get('mask', None)
    elif simulation.has_key('labels'):
        mask_vol = np.ones_like(simulation['labels'][0])
    else:
        raise Exception('Dunno where to get mask')

    pyhrf.verbose(3,'Vol mask of shape %s' %str(mask_vol.shape))

    fn_mask = add_prefix(op.join(output_dir, 'mask.nii'), prefix)
    write_volume(mask_vol.astype(np.int32), fn_mask, vol_meta)

    if simulation.has_key('hrf_territories'):
        fn_h_territories = add_prefix(op.join(output_dir, 'hrf_territories.nii'),
                                      prefix)

        ht = expand_array_in_mask(simulation['hrf_territories']+1, mask_vol)
        write_volume(ht, fn_h_territories, vol_meta)

    if simulation.has_key('hrf'):
        from pyhrf.ndarray import MRI3Daxes
        fn_hrf = add_prefix(op.join(output_dir, 'hrf.nii'), prefix)
        pyhrf.verbose(3,'hrf flat shape %s' %str(simulation['hrf'].shape))
        if simulation['hrf'].ndim == 1:
            hrf = (np.ones(mask_vol.size) * simulation['hrf'][:,np.newaxis])
        else:
            hrf = simulation['hrf']

        hrfs_vol = expand_array_in_mask(hrf, mask_vol, flat_axis=1)
        dt = simulation['dt']
        chrfs = xndarray(hrfs_vol, axes_names=['time',]+MRI3Daxes,
                       axes_domains={'time':np.arange(hrfs_vol.shape[0])*dt})
        #write_volume(np.rollaxis(hrfs_vol,0,4), fn_hrf, vol_meta)
        chrfs.save(fn_hrf, vol_meta)

        ttp_vol = hrfs_vol.argmax(0)
        fn_ttp = add_prefix(op.join(output_dir, 'ttp.nii'), prefix)
        write_volume(ttp_vol, fn_ttp, vol_meta)

    if simulation.has_key('brf'):
        from pyhrf.ndarray import MRI3Daxes
        fn_brf = add_prefix(op.join(output_dir, 'brf.nii'), prefix)
        pyhrf.verbose(3,'brf flat shape %s' %str(simulation['brf'].shape))
        brfs_vol = expand_array_in_mask(simulation['brf'], mask_vol, flat_axis=1)
        dt = simulation['dt']
        cbrfs = xndarray(brfs_vol, axes_names=['time',]+MRI3Daxes,
                       axes_domains={'time':np.arange(brfs_vol.shape[0])*dt})
        #write_volume(np.rollaxis(hrfs_vol,0,4), fn_hrf, vol_meta)
        cbrfs.save(fn_brf, vol_meta)

    if simulation.has_key('prf'):
        from pyhrf.ndarray import MRI3Daxes
        fn_brf = add_prefix(op.join(output_dir, 'prf.nii'), prefix)
        pyhrf.verbose(3,'prf flat shape %s' %str(simulation['prf'].shape))
        brfs_vol = expand_array_in_mask(simulation['prf'], mask_vol, flat_axis=1)
        dt = simulation['dt']
        cbrfs = xndarray(brfs_vol, axes_names=['time',]+MRI3Daxes,
                       axes_domains={'time':np.arange(brfs_vol.shape[0])*dt})
        #write_volume(np.rollaxis(hrfs_vol,0,4), fn_hrf, vol_meta)
        cbrfs.save(fn_brf, vol_meta)


    if simulation.has_key('drift'):
        fn_drift = add_prefix(op.join(output_dir, 'drift.nii'), prefix)
        pyhrf.verbose(3,'drift flat shape %s' %str(simulation['drift'].shape))
        drift_vol = expand_array_in_mask(simulation['drift'], mask_vol,
                                         flat_axis=1)
        #write_volume(drift_vol, fn_drift, vol_meta)
        write_volume(np.rollaxis(drift_vol,0,4), fn_drift)

    if simulation.has_key('drift_coeffs'):
        fn_drift = add_prefix(op.join(output_dir, 'drift_coeffs.nii'), prefix)
        pyhrf.verbose(3,'drift flat shape %s' %str(simulation['drift_coeffs'].shape))
        drift_vol = expand_array_in_mask(simulation['drift'], mask_vol,
                                         flat_axis=1)
        #write_volume(drift_vol, fn_drift, vol_meta)
        write_volume(np.rollaxis(drift_vol,0,4), fn_drift)


    if simulation.has_key('noise'):
        fn_noise = add_prefix(op.join(output_dir, 'noise.nii'), prefix)
        pyhrf.verbose(3,'noise flat shape %s' %str(simulation['noise'].shape))
        noise_vol = expand_array_in_mask(simulation['noise'], mask_vol,
                                         flat_axis=1)
        write_volume(np.rollaxis(noise_vol,0,4), fn_noise, vol_meta)

        fn_noise = add_prefix(op.join(output_dir, 'noise_emp_var.nii'), prefix)
        noise_vol = expand_array_in_mask(simulation['noise'].var(0), mask_vol)
        write_volume(noise_vol, fn_noise, vol_meta)


    if simulation.has_key('noise_var'):
        fn_noise_var = add_prefix(op.join(output_dir, 'noise_var.nii'), prefix)
        pyhrf.verbose(3,'noise_var flat shape %s' \
                      %str(simulation['noise_var'].shape))
        noise_var_vol = expand_array_in_mask(simulation['noise_var'], mask_vol)
        write_volume(noise_var_vol, fn_noise_var, vol_meta)



    if simulation.has_key('stim_induced_signal'):
        fn_stim_induced = add_prefix(op.join(output_dir, 'stim_induced.nii'),
                                     prefix)
        pyhrf.verbose(3,'stim_induced flat shape %s' \
                          %str(simulation['stim_induced_signal'].shape))
        stim_induced_vol = expand_array_in_mask(simulation['stim_induced_signal'],
                                           mask_vol, flat_axis=1)
        write_volume(np.rollaxis(stim_induced_vol,0,4), fn_stim_induced)

    if simulation.has_key('perf_stim_induced'):
        fn_stim_induced = add_prefix(op.join(output_dir, 'perf_stim_induced.nii'),
                                     prefix)
        pyhrf.verbose(3,'asl_stim_induced flat shape %s' \
                          %str(simulation['perf_stim_induced'].shape))
        stim_induced_vol = expand_array_in_mask(simulation['perf_stim_induced'],
                                                mask_vol, flat_axis=1)
        write_volume(np.rollaxis(stim_induced_vol,0,4), fn_stim_induced)

        fn_stim_induced = add_prefix(op.join(output_dir,
                                             'perf_stim_induced_ct.nii'),
                                     prefix)
        pyhrf.verbose(3,'asl_stim_induced flat shape %s' \
                          %str(simulation['perf_stim_induced'].shape))

        dsf = simulation['dsf']
        perf = np.dot(simulation['ctrl_tag_mat'],
                      simulation['perf_stim_induced'][0:-1:dsf])
        stim_induced_vol = expand_array_in_mask(perf, mask_vol, flat_axis=1)
        write_volume(np.rollaxis(stim_induced_vol,0,4), fn_stim_induced)

    if simulation.has_key('perf_baseline'):
        fn = add_prefix(op.join(output_dir, 'perf_baseline.nii'), prefix)
        pb = np.zeros_like(simulation['bold']) + simulation['perf_baseline']
        write_volume(expand_array_in_mask(pb[0], mask_vol), fn, vol_meta)




    if simulation.has_key('bold_stim_induced'):
        fn_stim_induced = add_prefix(op.join(output_dir, 'bold_stim_induced.nii'),
                                     prefix)
        pyhrf.verbose(3,'asl_stim_induced flat shape %s' \
                          %str(simulation['bold_stim_induced'].shape))
        stim_induced_vol = expand_array_in_mask(simulation['bold_stim_induced'],
                                                mask_vol, flat_axis=1)
        write_volume(np.rollaxis(stim_induced_vol,0,4), fn_stim_induced)


    m = np.where(mask_vol)
    labels_and_mask = mask_vol.copy()[m]


    for ic in xrange(simulation['labels'].shape[0]):
        if simulation.has_key('condition_defs'):
            c_name = simulation['condition_defs'][ic].name
        else:
            c_name = 'cond%d' %ic
        fn_labels = add_prefix(op.join(output_dir, 'labels_%s.nii' %c_name),
                               prefix)
        if simulation.has_key('labels'):
            labels_c = simulation['labels'][ic]
            labels_and_mask[np.where(labels_c)] = ic+2
            write_volume(expand_array_in_mask(labels_c,mask_vol).astype(np.int32),
                         fn_labels, vol_meta)
        elif simulation.has_key('labels_vol'):
            labels_c = simulation['labels_vol'][ic]
            labels_and_mask[np.where(labels_c[m])] = ic+2
            write_volume(labels_c.astype(np.int32), fn_labels, vol_meta)

        if simulation.has_key('nrls'):
            nrls_c = simulation['nrls'][ic]
            fn = add_prefix(op.join(output_dir, 'nrls_%s.nii' %c_name), prefix)
            write_volume(expand_array_in_mask(nrls_c,mask_vol) , fn, vol_meta)
        if simulation.has_key('nrls_session'):
            nrls_session_c = simulation['nrls_session'][ic]
            fn = add_prefix(op.join(output_dir, 'nrls_session_%s.nii' \
                                    %(c_name)), prefix)
            write_volume(expand_array_in_mask(nrls_session_c,mask_vol) ,
                         fn, vol_meta)

        if simulation.has_key('brls'):
            brls_c = simulation['brls'][ic]
            fn = add_prefix(op.join(output_dir, 'brls_%s.nii' %c_name), prefix)
            write_volume(expand_array_in_mask(brls_c,mask_vol) , fn, vol_meta)
        if simulation.has_key('prls'):
            prls_c = simulation['prls'][ic]
            fn = add_prefix(op.join(output_dir, 'prls_%s.nii' %c_name), prefix)
            write_volume(expand_array_in_mask(prls_c,mask_vol) , fn, vol_meta)

        if simulation.has_key('neural_efficacies'):
            ne_c = simulation['neural_efficacies'][ic]
            fn = add_prefix(op.join(output_dir, 'neural_efficacies_%s.nii' \
                                    %c_name), prefix)
            write_volume(expand_array_in_mask(ne_c,mask_vol) , fn, vol_meta)


    fn_labels_and_mask = add_prefix(op.join(output_dir, 'mask_and_labels.nii'),
                                    prefix)

    write_volume(expand_array_in_mask(labels_and_mask,mask_vol).astype(int),
                 fn_labels_and_mask, vol_meta)


    if simulation.has_key('bold_full_vol') or simulation.has_key('bold'):
        fn = add_prefix(op.join(output_dir, 'bold.nii'), prefix)
        if simulation.has_key('bold_full_vol'):
            bold4D = simulation['bold_full_vol']
        else:
            bold = simulation['bold']
            bold4D = expand_array_in_mask(bold, mask_vol, flat_axis=1)

        write_volume(np.rollaxis(bold4D,0,4), fn, vol_meta)

    def save_time_series(k):
        if simulation.has_key(k):
            fn_stim_induced = add_prefix(op.join(output_dir, k+'.nii'), prefix)
            pyhrf.verbose(3,'%s flat shape %s' %(k,str(simulation[k].shape)))
            vol = expand_array_in_mask(simulation[k], mask_vol, flat_axis=1)
            write_volume(np.rollaxis(vol,0,4), fn_stim_induced)
    save_time_series('flow_induction')
    save_time_series('cbv')
    save_time_series('hbr')
    save_time_series('bold_stim_induced_rescaled')

    if simulation.has_key('asl'):
        fn = add_prefix(op.join(output_dir, 'asl.nii'), prefix)
        asl4D = expand_array_in_mask(simulation['asl'], mask_vol, flat_axis=1)
        write_volume(np.rollaxis(asl4D,0,4), fn, vol_meta)

    if simulation.has_key('outliers'):
        fn = add_prefix(op.join(output_dir, 'outliers.nii'), prefix)
        outliers = expand_array_in_mask(simulation['outliers'], mask_vol,
                                        flat_axis=1)

        write_volume(np.rollaxis(outliers,0,4), fn, vol_meta)

    if simulation.has_key('hrf_group'):
        hrfgroup = simulation['hrf_group']
        nb_vox = mask_vol.size
        fn_hrf = add_prefix(op.join(output_dir, 'hrf_group.nii'), prefix)
        pyhrf.verbose(3,'hrf group shape %s' %str(simulation['hrf_group'].shape))
        hrfGd = duplicate_hrf(nb_vox, hrfgroup)
        hrfs_vol = expand_array_in_mask(hrfGd, mask_vol, flat_axis=1)
        dt = simulation['dt']
        chrfs = xndarray(hrfs_vol, axes_names=['time',]+MRI3Daxes,
        axes_domains={'time':np.arange(hrfs_vol.shape[0])*dt})
        chrfs.save(fn_hrf, vol_meta)


    if bold_3D_vols_dir is not None:
        assert op.exists(bold_3D_vols_dir)
        for iscan, bscan in enumerate(bold4D):
            fnout = add_prefix('bold_%06d.nii'%(iscan), prefix)
            #print fnout
            write_volume(bscan, op.join(bold_3D_vols_dir,fnout), vol_meta)

from pyhrf.core import Condition
import pyhrf.tools as ptools
import pyhrf.core as pcore

def create_small_bold_simulation(snr="high", output_dir=None, simu_items=None):
    """
    """

    conditions_def = [Condition(name='audio', m_act=3.8, v_act=.5,
                                v_inact=.5,
                                label_map=np.array([[[0,1],[1,0]]],dtype=int)),
                      Condition(name='video', m_act=3.8, v_act=.5,
                                v_inact=.5,
                                label_map=np.array([[[1,0],[1,0]]],dtype=int))]

    # noive variance
    v_noise = [1.5, .2][snr=="high"]

    simulation_steps = {
        'condition_defs' : conditions_def,
        'nb_voxels' : 4,
        # Labels
        'labels_vol' : create_labels_vol, # 4D shape
        'labels' : flatten_labels_vol, # 2D shape (flatten spatial axes)
        # Nrls
        'nrls' : create_time_invariant_gaussian_nrls,
        'dt' : .5,
        'dsf' : 2,
        'tr' : 1.,
        # Paradigm
        'paradigm' : create_localizer_paradigm,
        'rastered_paradigm' : rasterize_paradigm,
        # HRF
        'hrf_duration': 25,
        'primary_hrf' : create_canonical_hrf, # one hrf
        'hrf' : duplicate_hrf, # duplicate all HRF along all voxels
        # Stim induced
        'stim_induced_signal' : create_stim_induced_signal,
        # Noise
        'v_noise' : v_noise,
        'noise' : create_gaussian_noise, #requires bold_shape, v_gnoise
        # Drift
        'drift_order' : 4,
        'drift_var' : 11.,
        'drift' : create_polynomial_drift,
        # Bold
        'bold_shape' : get_bold_shape,
        'bold' : create_bold_from_stim_induced,
        }

    if simu_items is not None:
        simulation_steps.update(simu_items)

    simu_graph = ptools.Pipeline(simulation_steps)

    simu_graph.resolve()

    if output_dir is not None:
        # Output of the simulation graph:
        simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))

    simulation = simu_graph.get_values()

    if output_dir is not None:
        simulation_save_vol_outputs(simulation, output_dir=output_dir,
                                    bold_3D_vols_dir=None,)

    return pcore.FmriData.from_simulation_dict(simulation)
