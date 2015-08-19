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

eps = 1e-6


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
    iterable : iterable

    Returns
    tuple :
        iter_max : the maximum
        iter_max_indice : the indice of the maximum
    """

    iter_max = max(iterable)

    return iter_max, iterable.index(iter_max)


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
    lgt = (nbscans + 2) * osf  # nb of scans if tr=dt
    paradigm_bins = restarize_events(onsets, np.zeros_like(onsets), dt, tmax)
    firstcol = np.concatenate(
        (paradigm_bins, np.zeros(lgt - len(paradigm_bins))))
    firstrow = np.concatenate(
        ([paradigm_bins[0]], np.zeros(lhrf - 1, dtype=int)))
    x_tmp = np.array(toeplitz(firstcol, firstrow), dtype=int)
    os_indexes = [(np.arange(nbscans) * osf).astype(int)]
    x = x_tmp[os_indexes]
    return x


def buildFiniteDiffMatrix(order, size):
    a = np.diff(np.concatenate((np.zeros(order), [1], np.zeros(order))),
                n=order)
    b = a[len(a) / 2:]
    diffMat = toeplitz(np.concatenate((b, np.zeros(size - len(b)))))
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
                Htilde[
                    m, m2] += (np.dot(np.dot(Sigma_H, X[k].transpose()), X[k2])).trace()
                m2 += 1
            S += m_A[i, m] * np.dot(X[k], m_H)
            m += 1
        sigma_epsilone[i] = np.dot(-2 * S, Y[:, i] - PL[:, i])
        sigma_epsilone[i] += (np.dot(Sigma_A[:, :, i], Htilde)).trace()
        sigma_epsilone[
            i] += np.dot(np.dot(m_A[i, :].transpose(), Htilde), m_A[i, :])
        sigma_epsilone[
            i] += np.dot((Y[:, i] - PL[:, i]).transpose(), Y[:, i] - PL[:, i])
        sigma_epsilone[i] /= N
    return sigma_epsilone


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


def nrls_entropy(nrls_sigma, nb_conditions, nb_voxels):
    """Compute the entropy of neural response levels.

    Parameters
    ----------
    nrls_sigma : ndarray
        TODO
    nb_conditions : int
    nb_voxels : int

    Returns
    -------
    entropy : float
    """

    logger.info("Computing neural response levels entropy")
    entropy = 0.
    const = (2*np.pi)**nb_conditions * np.exp(nb_conditions)

    det_sigma_nrls = np.linalg.det(nrls_sigma.transpose((2, 0, 1)))
    entropy = -np.sum(np.log(np.sqrt(const*det_sigma_nrls) + eps_freeenergy))

    return entropy

def A_Entropy(Sigma_A, M, J):
    import warnings
    warnings.warn("The A_Entropy function is deprecated, use nrls_entropy instead",
                  DeprecationWarning)
    return nrls_entropy(Sigma_A, M, J)


# def H_Entropy(Sigma_H, D):

    # logger.info('Computing HRF Entropy ...')
    # Det_Sigma_H = np.linalg.det(Sigma_H)
    # Const = (2 * np.pi * np.exp(1)) ** D
    # Entropy = np.sqrt(Const * Det_Sigma_H)
    # Entropy = - np.log(Entropy + eps_FreeEnergy)

    # return Entropy


def hrf_entropy(hrf_sigma, hrf_len):
    """Compute the entropy of the heamodynamic response function.

    Parameters
    ----------
    hrf_sigma : ndarray
        TODO
    hrf_len : int

    Returns
    -------
    entropy : float
    """

    logger.info("Computing heamodynamic response function entropy")
    const = (2*np.pi)**hrf_len * np.exp(hrf_len)
    hrf_sigma_det = np.linalg.det(hrf_sigma)

    return -np.log(np.sqrt(const*hrf_sigma_det) + eps_freeenergy)

def H_Entropy(Sigma_H, D):
    import warnings
    warnings.warn("The H_Entropy function is deprecated, use hrf_entropy instead",
                  DeprecationWarning)
    return hrf_entropy(Sigma_H, D)


# def Z_Entropy(q_Z, M, J):

    # logger.info('Computing Z Entropy ...')
    # Entropy = 0.0
    # for j in xrange(0, J):
        # for m in xrange(0, M):
            # Entropy += q_Z[m, 1, j] * np.log(q_Z[m, 1, j] + eps_FreeEnergy) + q_Z[
                # m, 0, j] * np.log(q_Z[m, 0, j] + eps_FreeEnergy)

    # return Entropy


def labels_entropy(labels):
    """Compute the labels entropy.

    Parameters
    ----------
    labels : ndarray
        TODO
    nb_conditions : int
    nb_voxels : int

    Returns
    -------
    entropy : float
    """

    logger.info("Computing labels entropy")
    entropy = np.sum(
        labels[:, 1, :] * np.log(labels[:, 1, :] + eps_freeenergy)
        + labels[:, 0, :] * np.log(labels[:, 0, :] + eps_freeenergy))

    return entropy

def Z_Entropy(q_Z, M, J):
    import warnings
    warnings.warn("The Z_Entropy function is deprecated, use labels_entropy instead",
                  DeprecationWarning)
    return labels_entropy(q_Z)

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

def free_energy_computation(nrls_sigma, hrf_sigma, labels, nb_voxels, hrf_len,
                            nb_conditions, expectation_nrls, expectation_hrf,
                            expectation_labels):
    """Compute the free energy.

    Parameters
    ----------

    Returns
    -------
    free_energy : float
    """

    total_entropy = (nrls_entropy(nrls_sigma, nb_conditions, nb_voxels) +
                     hrf_entropy(hrf_sigma, hrf_len) +
                     labels_entropy(labels))
    total_expectation = expectation_nrls + expectation_hrf + expectation_labels

    return total_expectation - total_entropy


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
