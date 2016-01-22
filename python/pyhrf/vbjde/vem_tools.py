# -*- coding: utf-8 -*-

"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

import time
import logging

import numpy as np
import scipy as sp

from numpy.matlib import *
from scipy.linalg import toeplitz
from scipy.optimize import fmin_slsqp

import pyhrf
import pyhrf.vbjde.UtilsC as UtilsC

from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.ndarray import xndarray
from pyhrf.paradigm import restarize_events
from pyhrf.tools import format_duration
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


logger = logging.getLogger(__name__)


# Tools
##############################################################

eps = np.spacing(1)


def mult_old(v1, v2):
    matrix = np.zeros((len(v1), len(v2)), dtype=float)
    for i in xrange(len(v1)):
        for j in xrange(len(v2)):
            matrix[i, j] += v1[i] * v2[j]
    return matrix


def mult(v1, v2):
    """Multiply two vectors.

    The first vector is made vertical and the second one horizontal. The result
    will be a matrix of size len(v1), len(v2).

    Parameters
    ----------
    v1 : ndarray
        unidimensional
    v2 : ndarray
        unidimensional

    Returns
    -------
    x : ndarray, shape (len(v1), len(v2))
    """

    v1 = v1.reshape(len(v1), 1)
    v2 = v2.reshape(1, len(v2))
    return v1.dot(v2)


def maximum_old(a):
    maxx = a[0]
    maxx_ind = 0
    for i in xrange(len(a)):
        if a[i] > maxx:
            maxx = a[i]
            maxx_ind = i

    return maxx, maxx_ind


def maximum(iterable):
    """Return the maximum and the indice of the maximum of an iterable.

    Parameter
    ---------
    iterable : iterable or numpy array

    Returns
    tuple :
        iter_max : the maximum
        iter_max_indice : the indice of the maximum
    """

    iter_max = max(iterable)

    try:
        # this is an iterable (tuple or list)
        iter_max_indice = iterable.index(iter_max)
    except AttributeError:
        # this is an numpy array
        iter_max_indice = iterable.argmax()

    return iter_max, iter_max_indice


def normpdf(x, mu, sigma):
    u = (x - mu) / np.abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-u * u / 2)
    return y


def polyFit(signal, tr, order, p):
    n = len(signal)
    ptp = np.dot(p.transpose(), p)
    invptp = np.linalg.inv(ptp)
    invptppt = np.dot(invptp, p.transpose())
    l = np.dot(invptppt, signal)
    return l


def poly_fit(signal, drift_basis):
    """# TODO

    Parameters
    ----------
    signal : ndarray, shape (nb_scans, nb_voxels)
    drift_basis : ndarray, shape (nb_scans, int)

    Returns
    -------
    drift_coeffs : ndarray, shape
    """

    return np.linalg.inv(drift_basis.T.dot(drift_basis)).dot(drift_basis.T).dot(signal)


def PolyMat(Nscans, paramLFD, tr):
    '''Build polynomial basis'''
    regressors = tr * np.arange(0, Nscans)
    timePower = np.arange(0, paramLFD + 1, dtype=int)
    regMat = np.zeros((len(regressors), paramLFD + 1), dtype=np.float64)

    for v in xrange(paramLFD + 1):
        regMat[:, v] = regressors[:]
    #tPowerMat = np.matlib.repmat(timePower, Nscans, 1)
    tPowerMat = repmat(timePower, Nscans, 1)
    lfdMat = np.power(regMat, tPowerMat)
    lfdMat = np.array(sp.linalg.orth(lfdMat))
    return lfdMat


def compute_mat_X_2(nbscans, tr, lhrf, dt, onsets, durations=None):
    if durations is None:  # assume spiked stimuli
        durations = np.zeros_like(onsets)
    osf = tr / dt  # over-sampling factor
    # construction will only work if dt is a multiple of tr
    if int(osf) != osf:
        raise Exception('OSF (%f) is not an integer' % osf)

    x = np.zeros((nbscans, lhrf), dtype=float)
    tmax = nbscans * tr  # total session duration
    lgt = int((nbscans + 2) * osf)  # nb of scans if tr=dt
    paradigm_bins = restarize_events(onsets, np.zeros_like(onsets), dt, tmax)
    firstcol = np.concatenate(
        (paradigm_bins, np.zeros(lgt - len(paradigm_bins))))
    firstrow = np.concatenate(
        ([paradigm_bins[0]], np.zeros(lhrf - 1, dtype=int)))
    x_tmp = np.array(toeplitz(firstcol, firstrow), dtype=int)
    os_indexes = [(np.arange(nbscans) * osf).astype(int)]
    x = x_tmp[os_indexes]
    return x


def compute_mat_X_2_block(nbscans, tr, lhrf, dt, onsets, durations=None):
    if durations is None:  # assume spiked stimuli
        durations = np.zeros_like(onsets)
    osf = tr / dt  # over-sampling factor
    # construction will only work if dt is a multiple of tr
    if int(osf) != osf:
        raise Exception('OSF (%f) is not an integer' % osf)

    x = np.zeros((nbscans, lhrf), dtype=float)
    tmax = nbscans * tr  # total session duration
    lgt = (nbscans + 2) * osf  # nb of scans if tr=dt
    print 'onsets = ', onsets
    print 'durations = ', durations
    print 'dt = ', dt
    print 'tmax = ', tmax
    paradigm_bins = restarize_events(onsets, durations, dt, tmax)
    firstcol = np.concatenate(
        (paradigm_bins, np.zeros(lgt - len(paradigm_bins))))
    firstrow = np.concatenate(
        ([paradigm_bins[0]], np.zeros(lhrf - 1, dtype=int)))
    x_tmp = np.array(toeplitz(firstcol, firstrow), dtype=int)
    x_tmp2 = np.zeros_like(x_tmp)
    #for ix in np.arange(0, firstrow.shape[0], 1): #tr / dt):
    for ix in np.arange(0, firstrow.shape[0], tr / dt):
        x_tmp2[:, ix] = x_tmp[:, ix]       
    os_indexes = [(np.arange(nbscans) * osf).astype(int)]
    x = x_tmp2[os_indexes]
    if 0:
        import matplotlib.pyplot as plt
        from matplotlib.pylab import *
        plt.matshow(x_tmp[:300, :])
        plt.show()
        plt.matshow(x_tmp2[:300, :])
        plt.show()
        plt.matshow(x[:300, :])
        plt.show()
    return x


def buildFiniteDiffMatrix(order, size, regularization=None):
    """Build the finite difference matrix used for the hrf regularization prior.

    Parameters
    ----------
    order : int
        difference order (see numpy.diff function)
    size : int
        size of the matrix
    regularization : array like, optional
        one dimensional vector of factors used for regularizing the hrf

    Returns
    -------
    diffMat : ndarray, shape (size, size)
        the finite difference matrix"""

    a = np.diff(np.concatenate((np.zeros(order), [1], np.zeros(order))),
                n=order)
    b = a[len(a)//2:]
    diffMat = toeplitz(np.concatenate((b, np.zeros(size - len(b)))))
    if regularization is not None:
        regularization = np.array(regularization)
        if regularization.shape != (size,):
            raise Exception("regularization shape ({}) must be (size,) ({},)".format(regularization.shape, size))
        if not all(regularization > 0):
            raise Exception("All values of regularization must be stricly positive")
        diffMat = (np.triu(diffMat, 1) * regularization +
                   np.tril(diffMat, -1) * regularization[:, np.newaxis] +
                   np.diag(diffMat.diagonal() * regularization))
        # diffMat = (np.triu(diffMat, 1) + np.tril(diffMat, -1) +
                   # np.diag(diffMat.diagonal() * regularization))
        # diffMat = diffMat * regularization
    return diffMat


def _trap_area(p1, p2):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2

    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs(p2[0] - p1[0])
    avg_ht = (p1[1] + p2[1]) / 2.0

    return base * avg_ht


def roc_curve(dvals, labels, rocN=None, normalize=True):
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1}

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1:
        rocN = int(rocN * np.sum(np.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives

    fpc = [0.0]  # fp coordinates
    tpc = [0.0]  # tp coordinates
    dv_prev = - np.inf  # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    # num_pos = labels.count(1)  # num pos labels np.sum(labels)
    # num_neg = labels.count(0)  # num neg labels np.prod(labels.shape)-num_pos

    # if num_pos == 0 or num_pos == len(labels):
    #    raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = np.argsort(dvals)[::-1]

    for idx in indices:
        # increment associated TP/FP count
        if labels[idx] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN:
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[idx] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1]:
                tpc[-1] = TP
            else:
                fpc.append(FP)
                tpc.append(TP)
            dv_prev = dvals[idx]
            area += _trap_area((FP_prev, TP_prev), (FP, TP))
            FP_prev = FP
            TP_prev = TP

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )
    if normalize:
        fpc = [np.float64(x) / FP for x in fpc]
        if TP > 0:
            tpc = [np.float64(x) / TP for x in tpc]
        if area > 0:
            area /= (TP * FP)

    return fpc, tpc, area


def norm1_constraint(function, variance):
    """Returns the function constrained with optimization strategy.

    Parameters
    ----------
    function : array_like
        function to optimize under norm1 constraint
    variance : array_like
        variance of the `function`, must be the same size

    Returns
    -------
    optimized_function : numpy array

    Raises
    ------
    ValueError
        If `len(variance) != len(function)`

    """

    variance_inv = np.linalg.inv(variance)

    current_level = logger.getEffectiveLevel()
    if current_level >= logging.WARNING:
        disp = 0
    elif current_level >= logging.INFO:
        disp = 1
    else:
        disp = 2

    def minimized_function(fct):
        """Function to minimize"""
        return np.dot(np.dot((fct - function).T, variance_inv), (fct - function))

    def norm1_constraint_equation(fct):
        """Norm2(fct) == 1"""
        return np.linalg.norm(fct, 2) - 1

    def first_element_constraint(fct):
        """fct[0] == 0"""
        return fct[0]

    def last_element_constraint(fct):
        """fct[-1] == 0"""
        return fct[-1]

    return fmin_slsqp(minimized_function, function,
                      eqcons=[norm1_constraint_equation,
                              first_element_constraint,
                              last_element_constraint],
                      bounds=[(None, None)] * (len(function)), disp=disp)


# Expectation functions
##############################################################


def expectation_A(Y, Sigma_H, m_H, m_A, X, Gamma, PL, sigma_MK, q_Z, mu_MK, D, N, J, M, K, y_tilde, Sigma_A, sigma_epsilone, zerosJMD):
    X_tilde = zerosJMD.copy()  # np.zeros((Y.shape[1],M,D),dtype=float)
    J = Y.shape[1]
    for i in xrange(0, J):
        m = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Sigma_A[m, m2, i] = np.dot(np.dot(np.dot(np.dot(m_H.transpose(), X[
                                           k1].transpose()), Gamma / max(sigma_epsilone[i], eps)), X[k2]), m_H)
                Sigma_A[m, m2, i] += (np.dot(np.dot(np.dot(
                    Sigma_H, X[k1].transpose()), Gamma / max(sigma_epsilone[i], eps)), X[k2])).trace()
                m2 += 1
            X_tilde[i, m, :] = np.dot(
                np.dot(Gamma / max(sigma_epsilone[i], eps), y_tilde[:, i]).transpose(), X[k1])
            m += 1
        tmp = np.dot(X_tilde[i, :, :], m_H)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_MK[:, k] + eps))
            tmp += np.dot(Delta, mu_MK[:, k])
            Sigma_A[:, :, i] += Delta
        # print Sigma_A[:,:,i]
        tmp2 = np.linalg.inv(Sigma_A[:, :, i])
        Sigma_A[:, :, i] = tmp2
        m_A[i, :] = np.dot(Sigma_A[:, :, i], tmp)
    return Sigma_A, m_A


def nrls_expectation(hrf_mean, nrls_mean, occurence_matrix, noise_struct,
                     labels_proba, nrls_class_mean, nrls_class_var,
                     nb_conditions, y_tilde, nrls_covar,
                     hrf_covar, noise_var):
    """Computes the E-A step of the JDE-VEM algorithm.

    p_A = argmax_h(E_pc,pq,ph,pg[log p(a|y, h, c, g, q; theta)])
        \propto exp(E_pc,ph,pg[log p(y|h, a, c, g; theta)] \
                  + E_pq[log p(a|q; mu_Ma, sigma_Ma)])

    # TODO: add formulas using reST

    Parameters
    ----------
    hrf_mean : ndarray, shape (hrf_len,)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    occurence_matrix : ndarray, shape (nb_conditions, nb_scans, hrf_len)
    noise_struct : ndarray, shape (nb_scans, nb_scans)
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
    hrf_len : int
    nb_voxels : int
    nb_conditions : int
    nb_classes : int
    y_tilde : ndarray, shape (nb_scans, nb_voxels)
        BOLD data minus drifts
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
    noise_var : ndarray, shape (nb_voxels,)

    Returns
    -------
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    """

    # Pre-compute some matrix products
    om_hm_prod = occurence_matrix.dot(hrf_mean).T
    hc_om_prod = occurence_matrix.dot(hrf_covar.T)
    ns_om_prod = np.tensordot(noise_struct, occurence_matrix, axes=(1, 1))

    ## nrls_covar computation
    # first term of Sigma_A: XH.T*Gamma*XH / sigma_eps
    nrls_covar = om_hm_prod.T.dot(noise_struct).dot(om_hm_prod)[..., np.newaxis]
    # second term of Sigma_A: tr(X.T*Gamma*X*Sigma_H / sigma_eps)
    nrls_covar += np.einsum('ijk, jlk', hc_om_prod, ns_om_prod)[..., np.newaxis]
    nrls_covar = nrls_covar / noise_var

    # third term of nrls_covar: part of p(a|q; theta_A)
    delta_k = (labels_proba / nrls_class_var[:, :, np.newaxis])
    delta = delta_k.sum(axis=1)         # sum across classes K
    nrls_covar = nrls_covar.transpose(2, 0, 1) + delta.T[:, np.newaxis, :] * np.eye(nb_conditions)
    nrls_covar = np.linalg.inv(nrls_covar).transpose(1, 2, 0)

    ## m_A computation
    ns_yt_prod = noise_struct.dot(y_tilde).T
    x_tilde = ns_yt_prod.dot(om_hm_prod) / noise_var[:, np.newaxis] \
            + (delta_k * nrls_class_mean[:, :, np.newaxis]).sum(axis=1).T
    # dot product across voxels of nrls_covar and x_tilde
    nrls_mean = np.einsum('ijk,kj->ki', nrls_covar, x_tilde)

    return nrls_mean, nrls_covar


def expectation_H(Y, Sigma_A, m_A, X, Gamma, PL, D, R, sigmaH, J, N, y_tilde, zerosND, sigma_epsilone, scale, zerosDD, zerosD):
    Y_bar_tilde = zerosD.copy()  # np.zeros((D),dtype=float)
    Q_bar = scale * R / sigmaH
    Q_bar2 = scale * R / sigmaH
    for i in xrange(0, J):
        m = 0
        tmp = zerosND.copy()  # np.zeros((N,D),dtype=float)
        for k in X:  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            m += 1
        Y_bar_tilde += np.dot(np.dot(tmp.transpose(),
                                     Gamma / max(sigma_epsilone[i], eps)), y_tilde[:, i])
        Q_bar += np.dot(np.dot(tmp.transpose(), Gamma /
                               max(sigma_epsilone[i], eps)), tmp)
        Q_bar2[:, :] = Q_bar[:, :]
        m1 = 0
        for k1 in X:  # Loop over the M conditions
            m2 = 0
            for k2 in X:  # Loop over the M conditions
                Q_bar += Sigma_A[m1, m2, i] * np.dot(
                    np.dot(X[k1].transpose(), Gamma / max(sigma_epsilone[i], eps)), X[k2])
                m2 += 1
            m1 += 1
    Sigma_H = np.linalg.inv(Q_bar)
    m_H = np.dot(Sigma_H, Y_bar_tilde)
    m_H[0] = 0
    m_H[-1] = 0
    return Sigma_H, m_H


def hrf_expectation(nrls_covar, nrls_mean, occurence_matrix, noise_struct,
                    hrf_regu_prior_inv, sigmaH, nb_voxels, y_tilde, noise_var,
                    prior_mean_term=0., prior_cov_term=0.):

    """Computes the E-H step of the JDE-VEM algorithm.

    Expectation-H step:
    p_H = argmax_h(E_pa[log p(h|y, a ; theta)])
        \propto exp(E_pa[log p(y|h, a; theta) + log p(h; sigmaH)])

    Parameters
    ----------
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    occurence_matrix : ndarray, shape (nb_conditions, nb_scans, hrf_len)
    noise_struct : ndarray, shape (nb_scans, nb_scans)
    hrf_regu_prior_inv : ndarray, shape (hrf_len, hrf_len)
        inverse of the hrf regularization prior matrix `R`
    sigmaH : float
    nb_voxels : int
    y_tilde : ndarray, shape (nb_scans, nb_voxels)
    noise_var : ndarray, shape (nb_voxels,)
    prior_mean_term : float, optional
    prior_cov_term : float, optional

    Returns
    -------
    hrf_mean : ndarray, shape (hrf_len,)
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
    """

    # Pre-compute some matrix products
    nm_om_prod = np.tensordot(nrls_mean, occurence_matrix, axes=(1, 0))
    ns_om_prod = np.tensordot(noise_struct, occurence_matrix, axes=(1, 1))
    om_ns_om_prod = np.tensordot(occurence_matrix.T, ns_om_prod, axes=(1, 0))
    cov_noise = np.maximum(noise_var, eps)[:, np.newaxis, np.newaxis]
    nm_om_ns_prod = np.tensordot(nm_om_prod, noise_struct, axes=(1, 0))/cov_noise

    ## Sigma_H computation
    # first term: part of the prior -> R^-1 / sigmaH
    hrf_covar_inv = hrf_regu_prior_inv/sigmaH

    # second term: E_pa[Saj.T*noise_struct*Saj] op1
    # sum_{m, m'} Sigma_a(m,m') X_m.T noise_struct_i X_m'
    hrf_covar_inv += (np.einsum('ijk,lijm->klm', nrls_covar, om_ns_om_prod)/cov_noise).sum(0)

    # third term: E_pa[Saj.T*noise_struct*Saj] op2
    # (sum_m m_a X_m).T noise_struct_i (sum_m m_a X_m)
    for i in xrange(nb_voxels):
        hrf_covar_inv += nm_om_ns_prod[i, :, :].dot(nm_om_prod[i, :, :])

    # forth term (depends on prior type):
    # we sum the term that corresponds to the prior
    hrf_covar_inv += prior_cov_term

    # Sigma_H = S_a^-1
    hrf_covar = np.linalg.inv(hrf_covar_inv)

    ## m_H
    # (sum_m m_a X_m).T noise_struct_i y_tildeH
    y_bar_tilde = np.einsum('ijk,ki->j', nm_om_ns_prod, y_tilde)
    # we sum the term that corresponds to the prior
    y_bar_tilde += prior_mean_term

    # m_H = S_a^-1 y_bar_tilde
    hrf_mean = hrf_covar.dot(y_bar_tilde)

    return hrf_mean, hrf_covar


def expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z, graph, M, J, K, zerosK):
    energy = zerosK.copy()
    Gauss = zerosK.copy()
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = -0.5 * Sigma_A[m, m, i] / (sigma_M[m, :] + eps)
            alpha /= np.mean(alpha) + eps
            tmp = sum(Z_tilde[m, :, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[
                    k] + max(np.log(normpdf(m_A[i, m], mu_M[m, k], np.sqrt(sigma_M[m, k])) + eps), -100)
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
            Emax = max(energy)
            Probas = np.exp(energy - Emax)
            Sum = sum(Probas)
            Z_tilde[m, :, i] = Probas / (Sum + eps)
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = -0.5 * Sigma_A[m, m, i] / (sigma_M[m, :] + eps)
            alpha /= np.mean(alpha) + eps
            tmp = sum(Z_tilde[m, :, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[k]
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
                Gauss[k] = normpdf(
                    m_A[i, m], mu_M[m, k], np.sqrt(sigma_M[m, k]))
            Emax = max(energy)
            Probas = np.exp(energy - Emax)
            Sum = sum(Probas)
            q_Z[m, :, i] = Gauss * Probas / Sum
            SZ = sum(q_Z[m, :, i])
            q_Z[m, :, i] /= SZ
    return q_Z, Z_tilde


def labels_expectation(nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean,
                       beta, labels_proba, neighbours_indexes, nb_conditions, nb_classes):
    """Computes the E-Z (or E-Q) step of the JDE-VEM algorithm.

    Parameters
    ----------
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
    beta : ndarray, shape
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    neighbours_indexes : ndarray, shape (nb_voxels, max(len(a) for a in graph))
        This is the version of graph array where arrays from graph smaller than
        the maximum ones are filled with -1
    nb_conditions : int
    nb_classes : int

    Returns
    -------
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    """

    ########################### DEBUG START#####################################
    #  import matplotlib.pyplot as plt
    ########################### DEBUG END#######################################

    alpha = -0.5 * np.diagonal(nrls_covar)[:, :, np.newaxis] / (nrls_class_var[np.newaxis, :, :])

    ########################### DEBUG START#####################################
    #  for i in xrange(4):
        #  print "min diff alpha", i, (np.abs(alpha[:, i, 1] - alpha[:, i, 0])).max()
        #  print "qdsf", i, alpha[:, i, :].flatten().min(), alpha[:, i, :].flatten().mean(), alpha[:, i, :].flatten().max()

    #  f, ax = plt.subplots(2, 4, num=0)
    #  #  ax[0, 0].imshow(nrls_covar[0, 0, :].reshape(20, 20))
    #  ax[0, 0].imshow(nrls_covar[0, 0, :].reshape(20, 20))
    #  ax[0, 1].imshow(nrls_covar[1, 1, :].reshape(20, 20))
    #  ax[1, 0].imshow(nrls_covar[2, 2, :].reshape(20, 20))
    #  ax[1, 1].imshow(nrls_covar[3, 3, :].reshape(20, 20))
    #  ax[0, 2].imshow(nrls_mean[:, 0].reshape(20, 20))
    #  ax[0, 3].imshow(nrls_mean[:, 1].reshape(20, 20))
    #  ax[1, 2].imshow(nrls_mean[:, 2].reshape(20, 20))
    #  ax[1, 3].imshow(nrls_mean[:, 3].reshape(20, 20))
    #  #  plt.colorbar(plt1)
    #  f.show()
    #  plt.draw()
    #  print nrls_covar[0, 0, :].min(), nrls_covar[0, 0, :].mean(), np.median(nrls_covar[0, 0, :]), nrls_covar[0, 0, :].max()
    #  print nrls_covar[1, 1, :].min(), nrls_covar[1, 1, :].mean(), np.median(nrls_covar[1, 1, :]), nrls_covar[1, 1, :].max()
    #  print nrls_covar[2, 2, :].min(), nrls_covar[2, 2, :].mean(), np.median(nrls_covar[2, 2, :]), nrls_covar[2, 2, :].max()
    #  print nrls_covar[3, 3, :].min(), nrls_covar[3, 3, :].mean(), np.median(nrls_covar[3, 3, :]), nrls_covar[3, 3, :].max()
    ########################### DEBUG END#######################################

    alpha -= alpha.mean(axis=2)[:, :, np.newaxis]
    gauss = normpdf(nrls_mean[...,np.newaxis], nrls_class_mean, np.sqrt(nrls_class_var))

    # Update Ztilde ie the quantity which is involved in the a priori
    # Potts field [by solving for the mean-field fixed point Equation]
    beta_lprob = beta[..., np.newaxis, np.newaxis] * labels_proba
    beta_lprob = np.concatenate((beta_lprob,
                                 np.zeros((nb_conditions, nb_classes, 1),
                                          dtype=beta_lprob.dtype)), axis=2)
    local_energy = beta_lprob[:, :, neighbours_indexes].sum(axis=3).transpose(2, 0, 1)
    energy = alpha + local_energy
    #  energy -= energy.max()
    labels_proba = (np.exp(energy) * gauss).transpose(1, 2, 0)
    labels_proba = labels_proba / labels_proba.sum(axis=1)[:, np.newaxis, :]

    return labels_proba

def labels_expectation_soustraction(nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean,
                                    beta, labels_proba, neighbours_indexes, nb_conditions, nb_classes):
    """Computes the E-Z (or E-Q) step of the JDE-VEM algorithm.

    Parameters
    ----------
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
    beta : ndarray, shape
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    neighbours_indexes : ndarray, shape (nb_voxels, max(len(a) for a in graph))
        This is the version of graph array where arrays from graph smaller than
        the maximum ones are filled with -1
    nb_conditions : int
    nb_classes : int

    Returns
    -------
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    """

    alpha = - 0.5 * np.diagonal(nrls_covar)[:, :, np.newaxis] / (nrls_class_var[np.newaxis, :, :])
    #alpha -= alpha.mean(axis=2)[:, :, np.newaxis]
    gauss = normpdf(nrls_mean[...,np.newaxis], nrls_class_mean, np.sqrt(nrls_class_var))

    # Update Ztilde ie the quantity which is involved in the a priori
    # Potts field [by solving for the mean-field fixed point Equation]
    beta_lprob = beta[..., np.newaxis, np.newaxis] * labels_proba
    beta_lprob = np.concatenate((beta_lprob,
                                 np.zeros((nb_conditions, nb_classes, 1),
                                          dtype=beta_lprob.dtype)), axis=2)
    local_energy = beta_lprob[:, :, neighbours_indexes].sum(axis=3).transpose(2, 0, 1)
    energy = alpha + local_energy
    #energy -= energy.max()
    labels_proba = (np.exp(energy) * gauss).transpose(1, 2, 0)
    aux = labels_proba.sum(axis=1)[:, np.newaxis, :]
    aux[np.where(aux==0)] = eps
    labels_proba = labels_proba / aux 
    #/ labels_proba.sum(axis=1)[:, np.newaxis, :]
    return labels_proba

"""def expectation_Q(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, p_q_t, p_Q, neighbours_indexes, graph, M, J, K):
    # between ASL and BOLD just alpha and Gauss_mat change!!!
    alpha = - 0.5 * np.diagonal(Sigma_A)[:, :, np.newaxis] / (sigma_Ma[np.newaxis, :, :]) \
            - 0.5 * np.diagonal(Sigma_C)[:, :, np.newaxis] / (sigma_Mc[np.newaxis, :, :])  # (J, M, K)
    Gauss_mat = vt.normpdf(m_A[...,np.newaxis], mu_Ma, np.sqrt(sigma_Ma)) * \
                vt.normpdf(m_C[...,np.newaxis], mu_Mc, np.sqrt(sigma_Mc))

    # Update Ztilde ie the quantity which is involved in the a priori 
    # Potts field [by solving for the mean-field fixed point Equation]
    # TODO: decide if we take out the computation of p_q_t or Ztilde
    B_pqt = Beta[:, np.newaxis, np.newaxis] * p_q_t.copy()
    B_pqt = np.concatenate((B_pqt, np.zeros((M, K, 1), dtype=B_pqt.dtype)), axis=2)
    local_energy = B_pqt[:, :, neighbours_indexes].sum(axis=3).transpose(2, 0, 1)
    energy = (alpha + local_energy)
    energy -= energy.max()
    Probas = (np.exp(energy) * Gauss_mat).transpose(1, 2, 0)
    aux = Probas.sum(axis=1)[:, np.newaxis, :]
    aux[np.where(aux==0)] = eps
    p_q_t = Probas / aux
        
    return p_q_t, p_q_t"""


def labels_expectation_division(nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean,
                                beta, labels_proba, neighbours_indexes, nb_conditions, nb_classes):
    """Computes the E-Z (or E-Q) step of the JDE-VEM algorithm.

    Parameters
    ----------
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
    beta : ndarray, shape
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    neighbours_indexes : ndarray, shape (nb_voxels, max(len(a) for a in graph))
        This is the version of graph array where arrays from graph smaller than
        the maximum ones are filled with -1
    nb_conditions : int
    nb_classes : int

    Returns
    -------
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
    """

    alpha = -0.5 * np.diagonal(nrls_covar)[:, :, np.newaxis] / (nrls_class_var[np.newaxis, :, :])

    alpha /= alpha.mean(axis=2)[:, :, np.newaxis]
    gauss = normpdf(nrls_mean[...,np.newaxis], nrls_class_mean, np.sqrt(nrls_class_var))

    # Update Ztilde ie the quantity which is involved in the a priori
    # Potts field [by solving for the mean-field fixed point Equation]
    beta_lprob = beta[..., np.newaxis, np.newaxis] * labels_proba
    beta_lprob = np.concatenate((beta_lprob,
                                 np.zeros((nb_conditions, nb_classes, 1),
                                          dtype=beta_lprob.dtype)), axis=2)
    local_energy = beta_lprob[:, :, neighbours_indexes].sum(axis=3).transpose(2, 0, 1)
    energy = alpha + local_energy
    #  energy -= energy.max()
    labels_proba = (np.exp(energy) * gauss).transpose(1, 2, 0)
    labels_proba = labels_proba / labels_proba.sum(axis=1)[:, np.newaxis, :]

    return labels_proba


# Maximization functions
##############################################################

def maximization_mu_sigma(Mu, Sigma, q_Z, m_A, K, M, Sigma_A):
    for m in xrange(0, M):
        for k in xrange(0, K):
            #S = sum( q_Z[m,k,:] ) + eps
            S = sum(q_Z[m, k, :])
            if S == 0.:
                S = eps
            Sigma[m, k] = sum(
                q_Z[m, k, :] * (pow(m_A[:, m] - Mu[m, k], 2) + Sigma_A[m, m, :])) / S
            if Sigma[m, k] < eps:
                Sigma[m, k] = eps
            if k != 0:  # mu_0 = 0 a priori
                #Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
                Mu[m, k] = sum(q_Z[m, k, :] * m_A[:, m]) / S
            else:
                Mu[m, k] = 0.
    return Mu, Sigma


def maximization_class_proba(labels_proba, nrls_mean, nrls_covar):

    labels_proba_sum = labels_proba.sum(axis=2)
    nrls_class_mean = np.zeros_like(labels_proba_sum)
    nrls_class_mean[:, 1] = ((labels_proba[:, 1, :] * nrls_mean.T).sum(axis=1)
                             / labels_proba_sum[:, 1])
    nm_minus_ncm = (nrls_mean[..., np.newaxis]
                    - nrls_class_mean[np.newaxis, ...]).transpose(1, 2, 0)**2
    nrls_covar_diag = np.diagonal(nrls_covar).T
    nrls_class_var = (
        (labels_proba* (nm_minus_ncm + nrls_covar_diag[:, np.newaxis, :])).sum(axis=2)
        /labels_proba_sum
    )

    return nrls_class_mean, nrls_class_var


def maximization_L(Y, m_A, X, m_H, L, P, zerosP):
    J = Y.shape[1]
    for i in xrange(0, J):
        S = zerosP.copy()
        m = 0
        for k in X:
            S += m_A[i, m] * np.dot(X[k], m_H)
            m += 1
        L[:, i] = np.dot(P.transpose(), Y[:, i] - S)
    return L


def maximization_drift_coeffs(data, nrls_mean, occurence_matrix, hrf_mean,
                              noise_struct, drift_basis):
    # Precomputations
    db_ns_db = drift_basis.T.dot(noise_struct).dot(drift_basis)

    data_s = data - nrls_mean.dot(occurence_matrix.dot(hrf_mean)).T
    return np.linalg.inv(db_ns_db).dot(drift_basis.T).dot(noise_struct).dot(data_s)


def maximization_sigmaH(D, Sigma_H, R, m_H):
    sigmaH = (np.dot(mult(m_H, m_H) + Sigma_H, R)).trace()
    sigmaH /= D
    return sigmaH


def maximization_sigmaH_prior(D, Sigma_H, R, m_H, gamma_h):
    alpha = (np.dot(mult(m_H, m_H) + Sigma_H, R)).trace()
    #sigmaH = (D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)
    sigmaH = (-D + sqrt(D * D + 8 * gamma_h * alpha)) / (4 * gamma_h)

    return sigmaH


def maximization_sigma_noise(Y, X, m_A, m_H, Sigma_H, Sigma_A, PL, sigma_epsilone, M, zerosMM):
    N = PL.shape[0]
    J = Y.shape[1]
    Htilde = zerosMM.copy()  # np.zeros((M,M),dtype=float)
    for i in xrange(0, J):
        S = np.zeros((N), dtype=float)
        m = 0
        for k in X:
            m2 = 0
            for k2 in X:
                Htilde[m, m2] = np.dot(
                    np.dot(np.dot(m_H.transpose(), X[k].transpose()), X[k2]), m_H)
                Htilde[m, m2] += (np.dot(np.dot(Sigma_H, X[k].transpose()), X[k2])).trace()
                m2 += 1
            S += m_A[i, m] * np.dot(X[k], m_H)
            m += 1
        sigma_epsilone[i] = np.dot(-2 * S, Y[:, i] - PL[:, i])
        sigma_epsilone[i] += (np.dot(Sigma_A[:, :, i], Htilde)).trace()
        sigma_epsilone[i] += np.dot(np.dot(m_A[i, :].transpose(), Htilde), m_A[i, :])
        sigma_epsilone[i] += np.dot((Y[:, i] - PL[:, i]).transpose(), Y[:, i] - PL[:, i])
        sigma_epsilone[i] /= N
    return sigma_epsilone


def maximization_noise_var(occurence_matrix, hrf_mean, hrf_covar, nrls_mean,
                           nrls_covar, noise_struct, data_drift, nb_scans):
    """Computes the M-sigma_epsilone step of the JDE-VEM algorithm.

    """

    # Precomputations
    om_hm = occurence_matrix.dot(hrf_mean)
    ns_om = np.tensordot(noise_struct, occurence_matrix, axes=(1, 1))
    nm_om_hm = nrls_mean.dot(om_hm)

    hm_om_ns_om_hm = ns_om.transpose(1, 0, 2).dot(hrf_mean).dot(om_hm.T)
    hc_om_ns_om = np.einsum('ijk,ljk->il', occurence_matrix.dot(hrf_covar.T),
                            ns_om.transpose(1, 0, 2))

    hm_om_nm_ns_nm_om_hm = np.einsum('ij,ij->i', nrls_mean.dot(hm_om_ns_om_hm +
                                                               hc_om_ns_om),
                                     nrls_mean)

    # trace(Sigma_A (X.T H.T H X + SH X.T X) ) in each voxel
    tr_nc_om_ns_om = np.einsum('ijk,ji->k', nrls_covar, hm_om_ns_om_hm + hc_om_ns_om)

    ns_df = noise_struct.dot(data_drift)
    df_ns_df = np.einsum('ij,ij->j', data_drift, ns_df)

    nm_om_hm_ns_df = np.einsum('ij,ji->i', nm_om_hm, ns_df)

    return (hm_om_nm_ns_nm_om_hm + tr_nc_om_ns_om +
            df_ns_df - 2 * nm_om_hm_ns_df) / nb_scans


def gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma):
    Gr = gamma
    for i in xrange(0, J):
        tmp2 = beta * sum(Z_tilde[m, :, graph[i]], 0)
        Emax = max(tmp2)
        Sum = sum(np.exp(tmp2 - Emax))
        for k in xrange(0, K):
            tmp = sum(Z_tilde[m, k, graph[i]], 0)
            energy = beta * tmp
            Pzmi = np.exp(energy - Emax)
            Pzmi /= (Sum + eps)
            Gr += tmp * (-q_Z[m, k, i] + Pzmi)
    return Gr


def maximization_beta(beta, q_Z, Z_tilde, J, K, m, graph, gamma, neighbour, maxNeighbours):
    Gr = 100
    step = 0.005
    ni = 1
    while ((abs(Gr) > 0.0001) and (ni < 200)):
        Gr = gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma)
        beta -= step * Gr
        ni += 1
    return max(beta, eps)


def beta_maximization(beta, labels_proba, neighbours_indexes, gamma, nb_classes,
                      it_max_grad=200, gradient_step=0.003):
    """
    update beta for one condition

    Parameters
    ----------
    beta : float
    labels_proba : ndarray, shape(nb_classes, nb_voxels)
    neighbours_indexes : ndarray, # TODO
    gamma : float
    it_max_grad : int, optional
    gradient_step : float, optional
    """

    #  if beta < 0:
        #  import ipdb; ipdb.set_trace() #################### Breakpoint ####################

    labels_neigh = np.concatenate((labels_proba, np.zeros((nb_classes, 1), dtype=labels_proba.dtype)), axis=1)
    labels_neigh = labels_neigh[:, neighbours_indexes].sum(2)
    gradient = 1
    it = 1
    while abs(gradient) > 10**-4 and it < it_max_grad:
        beta_labels_neigh = beta * labels_neigh
        energy = np.exp(beta_labels_neigh)
        #  energy = np.exp(beta_labels_neigh - beta_labels_neigh.max(axis=0))
        energy /= energy.sum(axis=0)

        energy_neigh = np.concatenate((energy, np.zeros((nb_classes, 1), dtype=energy.dtype)), axis=1)
        energy_neigh = energy_neigh[:, neighbours_indexes].sum(axis=2)

        #  try:
        gradient = gamma + (energy*energy_neigh - labels_proba*labels_neigh).sum()/2.
        #  gradient = gamma + ((energy-labels_proba)*labels_neigh).sum()
        #  except FloatingPointError:
            #  import ipdb; ipdb.set_trace() #################### Breakpoint ####################

        #  if beta - gradient_step*gradient < 0:
            #  import ipdb; ipdb.set_trace() #################### Breakpoint ####################

        beta -= gradient_step * gradient

        it += 1

    return beta


#  def beta_maximization_scipy(beta, labels_proba, neighbours_indexes, gamma):

    #  def beta_func(beta, )
# Entropy functions
##############################################################

eps_FreeEnergy = 0.00000001
eps_freeenergy = 0.00000001


# def A_Entropy(Sigma_A, M, J):

    # logger.info('Computing NRLs Entropy ...')
    # Det_Sigma_A_j = np.zeros(J, dtype=np.float64)
    # Entropy = 0.0
    # for j in xrange(0, J):
        # Det_Sigma_A_j = np.linalg.det(Sigma_A[:, :, j])
        # Const = (2 * np.pi * np.exp(1)) ** M
        # Entropy_j = np.sqrt(Const * Det_Sigma_A_j)
        # Entropy += np.log(Entropy_j + eps_FreeEnergy)
    # Entropy = - Entropy

    # return Entropy


def nrls_entropy(nrls_covar, nb_conditions):
    """Compute the entropy of neural response levels.

    Parameters
    ----------
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
        Covariance of the NRLs
    nb_conditions : int
    nb_voxels : int

    Returns
    -------
    entropy : float
    """

    logger.info("Computing neural response levels entropy")
    const = (2*np.pi)**nb_conditions * np.exp(nb_conditions)

    det_nrls_covar = np.linalg.det(nrls_covar.transpose((2, 0, 1)))

    return np.sum(np.log(np.sqrt(const*det_nrls_covar)))

def A_Entropy(Sigma_A, M, J):
    import warnings
    warnings.warn("The A_Entropy function is deprecated, use nrls_entropy instead",
                  DeprecationWarning)
    return -nrls_entropy(Sigma_A, M, J)


# def H_Entropy(Sigma_H, D):

    # logger.info('Computing HRF Entropy ...')
    # Det_Sigma_H = np.linalg.det(Sigma_H)
    # Const = (2 * np.pi * np.exp(1)) ** D
    # Entropy = np.sqrt(Const * Det_Sigma_H)
    # Entropy = - np.log(Entropy + eps_FreeEnergy)

    # return Entropy


def hrf_entropy(hrf_covar, hrf_len):
    """Compute the entropy of the heamodynamic response function.

    Parameters
    ----------
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
        Covariance matrix of the HRF
    hrf_len : int
        size of the HRF

    Returns
    -------
    entropy : float
    """

    logger.info("Computing heamodynamic response function entropy")
    const = (2*np.pi)**hrf_len * np.exp(hrf_len)

    return np.log(np.sqrt(const*np.linalg.det(hrf_covar)))

def H_Entropy(Sigma_H, D):
    import warnings
    warnings.warn("The H_Entropy function is deprecated, use hrf_entropy instead",
                  DeprecationWarning)
    return -hrf_entropy(Sigma_H, D)


# def Z_Entropy(q_Z, M, J):

    # logger.info('Computing Z Entropy ...')
    # Entropy = 0.0
    # for j in xrange(0, J):
        # for m in xrange(0, M):
            # Entropy += q_Z[m, 1, j] * np.log(q_Z[m, 1, j] + eps_FreeEnergy) + q_Z[
                # m, 0, j] * np.log(q_Z[m, 0, j] + eps_FreeEnergy)

    # return Entropy


def labels_entropy(labels_proba):
    """Compute the labels entropy.

    Parameters
    ----------
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
        Probability of each voxel to be in one class
    nb_conditions : int
    nb_voxels : int

    Returns
    -------
    entropy : float
    """

    logger.info("Computing labels entropy")

    labels_proba_control_0 = labels_proba.copy()
    labels_proba_control_0[np.where(labels_proba==0)] = eps_FreeEnergy

    return -(labels_proba * np.log(labels_proba_control_0)).sum()


def Z_Entropy(q_Z, M, J):
    import warnings
    warnings.warn("The Z_Entropy function is deprecated, use labels_entropy instead",
                  DeprecationWarning)
    return -labels_entropy(q_Z)

# Other functions
##############################################################

def computeFit(m_H, m_A, X, J, N):
    # print 'Computing Fit ...'
    stimIndSignal = np.zeros((N, J), dtype=np.float64)
    for i in xrange(0, J):
        m = 0
        for k in X:
            stimIndSignal[:, i] += m_A[i, m] * np.dot(X[k], m_H)
            m += 1
    return stimIndSignal


def Compute_FreeEnergy(y_tilde, m_A, Sigma_A, mu_M, sigma_M, m_H, Sigma_H,
                       R, Det_invR, sigmaH, p_Wtilde, tau1, tau2, q_Z,
                       neighboursIndexes, maxNeighbours, Beta, sigma_epsilone,
                       XX, Gamma, Det_Gamma, XGamma, J, D, M, N, K, S, Model):

        # First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z, M, J)

    # if Model=="CompMod":
    Total_Entropy = EntropyA + EntropyH + EntropyZ
    # print 'Total Entropy =', Total_Entropy

    # Second Part (likelihood)
    EPtildeLikelihood = UtilsC.expectation_Ptilde_Likelihood(y_tilde, m_A, m_H, XX.astype(
        int32), Sigma_A, sigma_epsilone, Sigma_H, Gamma, p_Wtilde, XGamma, J, D, M, N, Det_Gamma)
    EPtildeA = UtilsC.expectation_Ptilde_A(
        m_A, Sigma_A, p_Wtilde, q_Z, mu_M, sigma_M, J, M, K)
    EPtildeH = UtilsC.expectation_Ptilde_H(
        R, m_H, Sigma_H, D, sigmaH, Det_invR)
    EPtildeZ = UtilsC.expectation_Ptilde_Z(
        q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)
    ##EPtildeZ = UtilsC.expectation_Ptilde_Z_MF_Begin(q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)
    # if Model=="CompMod":
    logger.debug("Computing Free Energy for CompMod")
    EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ

    FreeEnergy = EPtilde - Total_Entropy

    return FreeEnergy


def Compute_FreeEnergy2(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,
                       R,Det_invR,sigmaH,p_Wtilde,q_Z,neighboursIndexes,
                       maxNeighbours,Beta,sigma_epsilone,XX,Gamma,
                       Det_Gamma,XGamma,J,D,M,N,K,S,Model):
    ### Compute Free Energy

    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    if Model=="CompMod":
        Total_Entropy = EntropyA + EntropyH + EntropyZ

    ### Second Part (likelihood)
    ELikelihood = UtilsC.expectation_Ptilde_Likelihood(y_tilde,m_A,m_H,XX.astype(int32),Sigma_A,sigma_epsilone,Sigma_H,Gamma,p_Wtilde,XGamma,J,D,M,N,Det_Gamma)
    EA = UtilsC.expectation_A(m_A,Sigma_A,q_Z,mu_M,sigma_M,J,M,K)
    EH = UtilsC.expectation_H(R, m_H, Sigma_H, D, sigmaH, Det_invR)
    EZ = UtilsC.expectation_Z(q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)

    if Model=="CompMod":
        logger.debug("Computing Free Energy for CompMod")
        E = ELikelihood + EA + EH + EZ

    FreeEnergy = E - Total_Entropy

    return FreeEnergy


def expectation_ptilde_likelyhood(data_drift, nrls_mean, nrls_covar, hrf_mean,
                                  hrf_covar, occurence_matrix, noise_var,
                                  noise_struct, nb_voxels, nb_scans):
    """likelyhood
    # TODO

    Parameters
    ----------
    data_drift : ndarray, shape (nb_scans, nb_voxels)
        This is the BOLD data minus the drifts (y_tilde in the paper)
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
    hrf_mean : ndarray, shape (hrf_len,)
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
    occurence_matrix : ndarray, shape (nb_conditions, nb_scans, hrf_len)
    noise_var : ndarray, shape (nb_voxels,)
    noise_struct : ndarray, shape (nb_scans, nb_scans)
    nb_voxels : int
    nb_scans : int

    Returns
    -------
    ptilde_likelyhood : float
    """

    noise_var_tmp = maximization_noise_var(occurence_matrix, hrf_mean, hrf_covar, nrls_mean,
                                           nrls_covar, noise_struct, data_drift, nb_scans)
    return - (nb_scans*nb_voxels*np.log(2*np.pi) - nb_voxels*np.log(np.linalg.det(noise_struct))
              + 2*nb_scans*np.log(np.absolute(noise_var)).sum()
              + nb_scans*(noise_var_tmp / noise_var).sum()) / 2.


def expectation_ptilde_hrf(hrf_mean, hrf_covar, sigma_h, hrf_regu_prior,
                           hrf_regu_prior_inv, hrf_len):
    #logger.info('Computing hrf_regu_priorF expectation Ptilde ...')
    const = -(hrf_len*np.log(2*np.pi) + hrf_len*np.log(2*sigma_h)
             + np.log(np.linalg.det(hrf_regu_prior)))
    s = -(np.dot(np.dot(hrf_mean.T, hrf_regu_prior_inv), hrf_mean)
         + np.dot(hrf_covar, hrf_regu_prior_inv).trace()) / sigma_h

    return (const + s) / 2.


def expectation_ptilde_labels(labels_proba, neighbours_indexes, beta,
                              nb_conditions, nb_classes):
    labels_neigh = np.concatenate((labels_proba, np.zeros((nb_conditions, nb_classes, 1), dtype=labels_proba.dtype)), axis=2)
    labels_neigh = labels_neigh[:, :, neighbours_indexes].sum(axis=3)
    beta_labels_neigh = beta[:, np.newaxis, np.newaxis] * labels_neigh

    energy = np.exp(beta_labels_neigh - beta_labels_neigh.max(axis=0))
    energy /= energy.sum(axis=0)

    energy_neigh = np.concatenate((energy, np.zeros((nb_conditions, nb_classes, 1), dtype=energy.dtype)), axis=2)
    energy_neigh = energy[:, :, neighbours_indexes].sum(axis=3)

    return (-np.log(np.exp(beta_labels_neigh).sum(axis=1)).sum()
            + (beta*(labels_proba*labels_neigh/2.
                     + energy*(labels_neigh-energy_neigh/2.)).sum(axis=(1, 2))).sum())


def expectation_ptilde_nrls(labels_proba, nrls_class_mean, nrls_class_var,
                            nrls_mean, nrls_covar):
    diag_nrls_covar = np.diagonal(nrls_covar)[:, :, np.newaxis]
    s = -labels_proba.transpose(2, 0, 1)* (
        np.log(2*np.pi*nrls_class_var)
        + ((nrls_mean[:, :, np.newaxis] - nrls_class_mean[np.newaxis, :, :])**2
           + diag_nrls_covar) / (2*nrls_class_var[np.newaxis, :, :])
    )

    return s.sum()


def free_energy_computation(nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len,
                            labels_proba, data_drift, occurence_matrix, noise_var,
                            noise_struct, nb_conditions, nb_voxels, nb_scans, nb_classes,
                            nrls_class_mean, nrls_class_var, neighbours_indexes,
                            beta, sigma_h, hrf_regu_prior, hrf_regu_prior_inv,
                            gamma, gamma_h):
    """Compute the free energy.

    Parameters
    ----------

    Returns
    -------
    free_energy : float
    """

    total_entropy = (nrls_entropy(nrls_covar, nb_conditions) +
                     hrf_entropy(hrf_covar, hrf_len) +
                     labels_entropy(labels_proba))
    total_expectation = (
        expectation_ptilde_likelyhood(data_drift, nrls_mean, nrls_covar,
                                      hrf_mean, hrf_covar, occurence_matrix,
                                      noise_var, noise_struct, nb_voxels, nb_scans)
        + expectation_ptilde_nrls(labels_proba, nrls_class_mean, nrls_class_var,
                                  nrls_mean, nrls_covar)
        + expectation_ptilde_labels(labels_proba, neighbours_indexes, beta,
                                    nb_conditions, nb_classes)
        + expectation_ptilde_hrf(hrf_mean, hrf_covar, sigma_h, hrf_regu_prior,
                                 hrf_regu_prior_inv, hrf_len)
    )

    total_prior = 0
    if gamma:
        total_prior += (nb_conditions*np.log(gamma) - gamma*beta).sum()
    if gamma_h:
        total_prior += log(gamma_h) - gamma_h*sigma_h


    return total_expectation + total_entropy + total_prior


# MiniVEM
#########################################################

def MiniVEM_CompMod(Thrf, TR, dt, beta, Y, K, gamma, gradientStep, MaxItGrad, D, M, N, J, S, maxNeighbours, neighboursIndexes, XX, X, R, Det_invR, Gamma, Det_Gamma, p_Wtilde, scale, Q_barnCond, XGamma, tau1, tau2, Nit, sigmaH, estimateHRF):

    # print 'InitVar =',InitVar,',    InitMean =',InitMean,',     gamma_h
    # =',gamma_h

    Init_sigmaH = sigmaH

    IM_val = np.array([-5., 5.])
    IV_val = np.array([0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512])
    #IV_val = np.array([0.01,0.05,0.1,0.5])
    gammah_val = np.array([1000])
    MiniVemStep = IM_val.shape[0] * IV_val.shape[0] * gammah_val.shape[0]

    Init_mixt_p_gammah = []

    logger.info("Number of tested initialisation is %s", MiniVemStep)

    t1_MiniVEM = time.time()
    FE = []
    for Gh in gammah_val:
        for InitVar in IV_val:
            for InitMean in IM_val:
                Init_mixt_p_gammah += [[InitVar, InitMean, Gh]]
                sigmaH = Init_sigmaH
                sigma_epsilone = np.ones(J)
                if 0:
                    logger.info(
                        "Labels are initialized by setting active probabilities to zeros ...")
                    q_Z = np.ones((M, K, J), dtype=np.float64)
                    q_Z[:, 1, :] = 0
                if 0:
                    logger.info("Labels are initialized randomly ...")
                    q_Z = np.zeros((M, K, J), dtype=np.float64)
                    nbVoxInClass = J / K
                    for j in xrange(M):
                        if J % 2 == 0:
                            l = []
                        else:
                            l = [0]
                        for c in xrange(K):
                            l += [c] * nbVoxInClass
                        q_Z[j, 0, :] = np.random.permutation(l)
                        q_Z[j, 1, :] = 1. - q_Z[j, 0, :]
                if 1:
                    logger.info(
                        "Labels are initialized by setting active probabilities to ones ...")
                    q_Z = np.zeros((M, K, J), dtype=np.float64)
                    q_Z[:, 1, :] = 1

                # TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
                TT, m_h = getCanoHRF(Thrf, dt)  # TODO: check
                m_h = m_h[:D]
                m_H = np.array(m_h).astype(np.float64)
                if estimateHRF:
                    Sigma_H = np.ones((D, D), dtype=np.float64)
                else:
                    Sigma_H = np.zeros((D, D), dtype=np.float64)

                Beta = beta * np.ones((M), dtype=np.float64)
                P = PolyMat(N, 4, TR)
                L = polyFit(Y, TR, 4, P)
                PL = np.dot(P, L)
                y_tilde = Y - PL
                Ndrift = L.shape[0]

                gamma_h = Gh
                sigma_M = np.ones((M, K), dtype=np.float64)
                sigma_M[:, 0] = 0.1
                sigma_M[:, 1] = 1.0
                mu_M = np.zeros((M, K), dtype=np.float64)
                for k in xrange(1, K):
                    mu_M[:, k] = InitMean
                Sigma_A = np.zeros((M, M, J), np.float64)
                for j in xrange(0, J):
                    Sigma_A[:, :, j] = 0.01 * np.identity(M)
                m_A = np.zeros((J, M), dtype=np.float64)
                for j in xrange(0, J):
                    for m in xrange(0, M):
                        for k in xrange(0, K):
                            m_A[j, m] += np.random.normal(
                                mu_M[m, k], np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]

                for ni in xrange(0, Nit + 1):
                    logger.info("------------------------------ Iteration n° " +
                                str(ni + 1) + " ------------------------------")
                    UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                                         Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(int32), J, D, M, N, K)
                    val = np.reshape(m_A, (M * J))
                    val[np.where((val <= 1e-50) & (val > 0.0))] = 0.0
                    val[np.where((val >= -1e-50) & (val < 0.0))] = 0.0
                    m_A = np.reshape(val, (J, M))

                    if estimateHRF:
                        UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R, Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(
                            int32), J, D, M, N, scale, sigmaH)
                        m_H[0] = 0
                        m_H[-1] = 0

                    UtilsC.expectation_Z_ParsiMod_3(
                        Sigma_A, m_A, sigma_M, Beta, p_Wtilde, mu_M, q_Z, neighboursIndexes.astype(int32), M, J, K, maxNeighbours)
                    val = np.reshape(q_Z, (M * K * J))
                    val[np.where((val <= 1e-50) & (val > 0.0))] = 0.0
                    q_Z = np.reshape(val, (M, K, J))

                    if estimateHRF:
                        if gamma_h > 0:
                            sigmaH = maximization_sigmaH_prior(
                                D, Sigma_H, R, m_H, gamma_h)
                        else:
                            sigmaH = maximization_sigmaH(D, Sigma_H, R, m_H)
                    mu_M, sigma_M = maximization_mu_sigma(
                        mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)
                    UtilsC.maximization_L(
                        Y, m_A, m_H, L, P, XX.astype(int32), J, D, M, Ndrift, N)
                    PL = np.dot(P, L)
                    y_tilde = Y - PL
                    for m in xrange(0, M):
                        Beta[m] = UtilsC.maximization_beta(beta, q_Z[m, :, :].astype(float64), q_Z[m, :, :].astype(
                            float64), J, K, neighboursIndexes.astype(int32), gamma, maxNeighbours, MaxItGrad, gradientStep)
                    UtilsC.maximization_sigma_noise(
                        Gamma, PL, sigma_epsilone, Sigma_H, Y, m_A, m_H, Sigma_A, XX.astype(int32), J, D, M, N)

                FreeEnergy = Compute_FreeEnergy(y_tilde, m_A, Sigma_A, mu_M, sigma_M, m_H, Sigma_H, R, Det_invR, sigmaH, p_Wtilde, tau1,
                                                tau2, q_Z, neighboursIndexes, maxNeighbours, Beta, sigma_epsilone, XX, Gamma, Det_Gamma, XGamma, J, D, M, N, K, S, "CompMod")
                FE += [FreeEnergy]

    max_FE, max_FE_ind = maximum(FE)
    InitVar = Init_mixt_p_gammah[max_FE_ind][0]
    InitMean = Init_mixt_p_gammah[max_FE_ind][1]
    Initgamma_h = Init_mixt_p_gammah[max_FE_ind][2]

    t2_MiniVEM = time.time()
    logger.info(
        "MiniVEM duration is %s", format_duration(t2_MiniVEM - t1_MiniVEM))
    logger.info("Choosed initialisation is : var = %s,  mean = %s,  gamma_h = %s",
                InitVar, InitMean, Initgamma_h)

    return InitVar, InitMean, Initgamma_h


def MiniVEM_CompMod2(Thrf,TR,dt,beta,Y,K,gamma,gradientStep,MaxItGrad,
                    D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,
                    Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,
                    Nit,sigmaH,estimateHRF):
    # Mini VEM to have a goo initialization

    Init_sigmaH = sigmaH
    IM_val = np.array([-5.,5.])
    IV_val = np.array([0.008,0.016,0.032,0.064,0.128,0.256,0.512])
    gammah_val = np.array([1000])
    MiniVemStep = IM_val.shape[0]*IV_val.shape[0]*gammah_val.shape[0]
    Init_mixt_p_gammah = []

    logger.info("Number of tested initialisation is %s" %MiniVemStep)

    t1_MiniVEM = time.time()
    FE = []
    for Gh in gammah_val:
        for InitVar in IV_val:
            for InitMean in IM_val:
                Init_mixt_p_gammah += [[InitVar,InitMean,Gh]]
                sigmaH = Init_sigmaH
                sigma_epsilone = np.ones(J)
                if 0:
                    logger.info("Labels are initialized by setting active probabilities to zeros ...")
                    q_Z = np.ones((M,K,J),dtype=np.float64)
                    q_Z[:,1,:] = 0
                if 0:
                    logger.info("Labels are initialized randomly ...")
                    q_Z = np.zeros((M,K,J),dtype=np.float64)
                    nbVoxInClass = J/K
                    for j in xrange(M) :
                        if J%2==0:
                            l = []
                        else:
                            l = [0]
                        for c in xrange(K) :
                            l += [c] * nbVoxInClass
                        q_Z[j,0,:] = np.random.permutation(l)
                        q_Z[j,1,:] = 1. - q_Z[j,0,:]
                if 1:
                    logger.info("Labels are initialized by setting active probabilities to ones ...")
                    q_Z = np.zeros((M,K,J),dtype=np.float64)
                    q_Z[:,1,:] = 1

                #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
                TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
                m_h = m_h[:D]
                m_H = np.array(m_h).astype(np.float64)
                if estimateHRF:
                    Sigma_H = np.ones((D,D),dtype=np.float64)
                else:
                    Sigma_H = np.zeros((D,D),dtype=np.float64)

                Beta = beta * np.ones((M),dtype=np.float64)
                P = PolyMat( N , 4 , TR)
                L = polyFit(Y, TR, 4,P)
                PL = np.dot(P,L)
                y_tilde = Y - PL
                Ndrift = L.shape[0]

                gamma_h = Gh
                sigma_M = np.ones((M,K),dtype=np.float64)
                sigma_M[:,0] = 0.1
                sigma_M[:,1] = 1.0
                mu_M = np.zeros((M,K),dtype=np.float64)
                for k in xrange(1,K):
                    mu_M[:,k] = InitMean
                Sigma_A = np.zeros((M,M,J),np.float64)
                for j in xrange(0,J):
                    Sigma_A[:,:,j] = 0.01*np.identity(M)
                m_A = np.zeros((J,M),dtype=np.float64)
                for j in xrange(0,J):
                    for m in xrange(0,M):
                        for k in xrange(0,K):
                            m_A[j,m] += np.random.normal(mu_M[m,k], np.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

                for ni in xrange(0,Nit+1):
                    logger.info("------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
                    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
                    val = np.reshape(m_A,(M*J))
                    val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
                    val[ np.where((val>=-1e-50) & (val<0.0)) ] = 0.0
                    m_A = np.reshape(val, (J,M))

                    if estimateHRF:
                        UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                        m_H[0] = 0
                        m_H[-1] = 0

                    UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                    val = np.reshape(q_Z,(M*K*J))
                    val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
                    q_Z = np.reshape(val, (M,K,J))

                    if estimateHRF:
                        if gamma_h > 0:
                            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                        else:
                            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
                    UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
                    PL = np.dot(P,L)
                    y_tilde = Y - PL
                    for m in xrange(0,M):
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

                FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
                FE += [FreeEnergy]

    max_FE, max_FE_ind = maximum(FE)
    InitVar = Init_mixt_p_gammah[max_FE_ind][0]
    InitMean = Init_mixt_p_gammah[max_FE_ind][1]
    Initgamma_h = Init_mixt_p_gammah[max_FE_ind][2]

    t2_MiniVEM = time.time()
    logger.info("MiniVEM duration is %s" %format_duration(t2_MiniVEM-t1_MiniVEM))
    logger.info("Choosed initialisation is : var = %s,  mean = %s,  gamma_h = %s" %(InitVar,InitMean,Initgamma_h))

    return InitVar, InitMean, Initgamma_h
