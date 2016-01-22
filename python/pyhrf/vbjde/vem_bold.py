# -*- coding: utf-8 -*-

"""This module implements the VEM for BOLD data.

The function uses the C extension for expectation and maximization steps (see
src/pyhrf/vbjde/utilsmodule.c file).


See Also
--------
pyhrf.ui.analyser_ui, pyhrf.ui.treatment, pyhrf.ui.jde, pyhrf.ui.vb_jde_analyser

Notes
-----
TODO: add some refs?

Attributes
----------
eps : float
    mimics the mechine epsilon to avoid zero values
logger : logger
    logger instance identifying this module to log informations

"""

import os
import time
import logging

from collections import OrderedDict

import numpy as np

try:
    os.environ["DISPLAY"]
except KeyError:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt

import pyhrf.vbjde.UtilsC as UtilsC
import pyhrf.vbjde.vem_tools as vt
import pyhrf.vbjde.vem_tools_asl as EM
import pyhrf.vbjde.vem_tools_asl_fast as EMf

from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.tools._io import read_volume
from pyhrf.stats.misc import cpt_ppm_a_norm, cpt_ppm_g_norm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eps = np.spacing(1)


#  @profile
def jde_vem_bold(graph, bold_data, onsets, hrf_duration, nb_classes, tr, beta,
                 dt, scale=1, estimate_sigma_h=True, sigma_h=0.05, it_max=-1,
                 it_min=0, estimate_beta=True, plot=False, contrasts=None,
                 compute_contrasts=False, gamma_h=0, estimate_hrf=True,
                 true_hrf_flag=False, hrf_filename='hrf.nii',
                 estimate_labels=True, labels_filename='labels.nii',
                 constrained=False, seed=6537546):
    """This is the main function that computes the VEM analysis on BOLD data.

    Parameters
    ----------
    graph : # TODO
        # TODO
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    onsets : dict
        dictionnary of onsets
    hrf_duration : float
        hrf total time duration (in s)
    nb_classes : # TODO
        # TODO
    tr : float
        time of repetition
    beta : # TODO
        # TODO
    dt : float
        hrf temporal precision
    scale : float, optional
        scale factor for datas ? # TODO: check
    estimate_sigma_h : bool, optional
        toggle estimation of sigma H
    sigma_h : float, optional
        initial or fixed value of sigma H
    it_max : int, optional
        maximal computed iteration number
    it_min : int, optional
        minimal computed iteration number
    estimate_beta : bool, optional
        toggle the estimation of Beta
    plot : bool, optional
        if True, plot some images of some variables (# TODO: describe, or better,
        remove)
    contrasts : OrderedDict, optional
        dict of contrasts to compute
    compute_contrasts : bool, optional
        if True, compute the contrasts defined in contrasts
    gamma_h : float (# TODO: check)
        # TODO
    estimate_hrf : bool, optional
        if True, estimate the HRF for each parcel, if False use the canonical
        HRF
    true_hrf_flag : bool, optional
        if True, use the hrf_filename to provide the true HRF that will be used
        (imply estimate_hrf=False) # TODO: check
    hrf_filename : string, optional
        specify the file name for the true HRF (used only if true_hrf_flag is
        set to True)
    estimate_labels : bool, optional
        if True the labels are estimated
    seed : int, optional
        seed used by numpy to initialize random generator number

    Returns
    -------
    loop : int
        number of iterations before convergence
    m_A : ndarray, shape (nb_voxels, nb_conditions)
        Neural response level mean value
    m_H : ndarray, shape (hrf_len,)
        Hemodynamic response function mean value
    hrf_covariance : ndarray, shape (hrf_len, hrf_len)
        Covariance matrix of the HRF
    q_Z : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
        # TODO
    sigma_epsilone : ndarray, shape (nb_voxels,)
        # TODO
    mu_M : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    sigma_M : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    Beta : ndarray, shape (nb_conditions,)
        # TODO
    drift_coeffs : ndarray, shape (# TODO)
        # TODO
    drift : ndarray, shape (# TODO)
        # TODO
    CONTRAST : ndarray, shape (nb_voxels, len(contrasts))
        Contrasts computed from NRLs
    CONTRASTVAR : ndarray, shape (nb_voxels, len(contrasts))
        Variance of the contrasts
    cA : list
        NRL criteria (# TODO: explain)
    cH : list
        HRF criteria (# TODO: explain)
    cZ : list
        Z criteria (# TODO: explain)
    cAH : list
        NRLs HRF product criteria
    compute_time : list
        computation time of each iteration
    compute_time_mean : float
        computation mean time over iterations
    Sigma_A : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
        # TODO
    StimulusInducedSignal : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    density_ratio : float
        # TODO
    density_ratio_cano : float
        # TODO
    density_ratio_diff : float
        # TODO
    density_ratio_prod : float
        # TODO
    ppm_a_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_a_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    variation_coeff : float
        coefficient of variation of the HRF

    Notes
    -----
        See `A novel definition of the multivariate coefficient of variation <http://onlinelibrary.wiley.com/doi/10.1002/bimj.201000030/abstract>`_
        article for more information about the coefficient of variation.
    """

    logger.info("Fast EM with C extension started.")

    if not contrasts:
        contrasts = OrderedDict()

    np.random.seed(seed)

    Nb2Norm = 1
    NormFlag = False
    regularizing = False
    estimate_free_energy = False

    if it_max < 0:
        it_max = 100
    gamma = 7.5
    gradient_step = 0.003
    it_max_grad = 200
    thresh = 1e-5
    thresh_free_energy = 1e-5
    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_conditions = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    condition_names = []

    max_neighbours = max([len(nl) for nl in graph])
    #  neighbours_indexes = np.zeros((nb_voxels, max_neighbours), dtype=np.int32)
    #  neighbours_indexes -= 1
    #  for i in xrange(nb_voxels):
        #  neighbours_indexes[i, :len(graph[i])] = graph[i]
    neighbours_indexes = [np.concatenate((arr, np.zeros(max_neighbours-len(arr))-1))
                          for arr in graph]
    neighbours_indexes = np.asarray(neighbours_indexes, dtype=int)

    X = OrderedDict()
    for condition, onset in onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(nb_scans, tr, hrf_len, dt, onset)
        condition_names.append(condition)
    XX = np.zeros((nb_conditions, nb_scans, hrf_len), dtype=np.int32)
    for nc, condition in enumerate(onsets.iterkeys()):
        XX[nc, :, :] = X[condition]

    if regularizing:
        regularization = np.ones(hrf_len)
        regularization[hrf_len//3:hrf_len//2] = 2
        regularization[hrf_len//2:2*hrf_len//3] = 5
        regularization[2*hrf_len//3:3*hrf_len//4] = 7
        regularization[3*hrf_len//4:] = 10
        # regularization[hrf_len//2:] = 10
    else:
        regularization = None
    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, hrf_len, regularization)
    hrf_regularization_prior = np.dot(D2, D2) / pow(dt, 2 * order)

    Gamma = np.identity(nb_scans)
    Det_Gamma = np.linalg.det(Gamma)

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = np.zeros((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)
    AH1 = np.zeros((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)
    free_energy_crit = [1.]
    compute_time = []
    cA = []
    cH = []
    cZ = []
    cAH = []

    SUM_q_Z = [[] for m in xrange(nb_conditions)]
    mu1 = [[] for m in xrange(nb_conditions)]
    h_norm = []

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    Q_barnCond = np.zeros((nb_conditions, nb_conditions, hrf_len, hrf_len), dtype=np.float64)
    XGamma = np.zeros((nb_conditions, hrf_len, nb_scans), dtype=np.float64)
    m1 = 0
    for k1 in X:  # Loop over the nb_conditions conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1, m2, :, :] = np.dot(
                np.dot(X[k1].transpose(), Gamma), X[k2])
            m2 += 1
        XGamma[m1, :, :] = np.dot(X[k1].transpose(), Gamma)
        m1 += 1

    sigma_epsilone = np.ones(nb_voxels)

    logger.info(
        "Labels are initialized by setting active probabilities to ones ...")
    q_Z = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    q_Z[:, 1, :] = 1

    q_Z1 = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    Z_tilde = q_Z.copy()

    m_h = getCanoHRF(hrf_duration, dt)[1][:hrf_len]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    sigmaH1 = sigma_h
    if estimate_hrf:
        hrf_covariance = np.ones((hrf_len, hrf_len), dtype=np.float64)
    else:
        hrf_covariance = np.zeros((hrf_len, hrf_len), dtype=np.float64)

    Beta = beta * np.ones((nb_conditions), dtype=np.float64)
    drift_basis = vt.PolyMat(nb_scans, 4, tr)
    drift_coeffs = vt.polyFit(bold_data, tr, 4, drift_basis)
    drift = np.dot(drift_basis, drift_coeffs)
    y_tilde = bold_data - drift
    Ndrift = drift_coeffs.shape[0]

    sigma_M = np.ones((nb_conditions, nb_classes), dtype=np.float64)
    sigma_M[:, 0] = 0.5
    sigma_M[:, 1] = 0.6
    mu_M = np.zeros((nb_conditions, nb_classes), dtype=np.float64)
    for k in xrange(1, nb_classes):
        mu_M[:, k] = 1  # init_mean
        Sigma_A = 0.01 * (np.identity(nb_conditions)[:, :, np.newaxis]
                          + np.zeros((1, 1, nb_voxels)))
    m_A = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    m_A1 = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    for j in xrange(0, nb_voxels):
        for m in xrange(0, nb_conditions):
            for k in xrange(0, nb_classes):
                m_A[j, m] += np.random.normal(
                    mu_M[m, k], np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]
    m_A1 = m_A

    start_time = time.time()
    free_energy = [0.]
    loop = 0
    while (loop < it_min + 1) or (free_energy_crit[-1] > thresh_free_energy and
                                  Crit_AH > thresh and loop < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(loop+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: m_A = %s, Sigma_A = %s", m_A, Sigma_A)
        nrls_expectation = UtilsC.expectation_A(q_Z, mu_M, sigma_M, drift,
                                                sigma_epsilone, Gamma, hrf_covariance,
                                                bold_data, y_tilde, m_A, m_H,
                                                Sigma_A, XX.astype(np.int32),
                                                nb_voxels, hrf_len, nb_conditions,
                                                nb_scans, nb_classes)
        logger.debug("After: m_A = %s, Sigma_A = %s", m_A, Sigma_A)
        if estimate_hrf:
            logger.info("Expectation H step...")
            logger.debug("Before: m_H = %s, hrf_covariance = %s", m_H, hrf_covariance)
            hrf_expectation = UtilsC.expectation_H(XGamma, Q_barnCond,
                                                   sigma_epsilone, Gamma, hrf_regularization_prior,
                                                   hrf_covariance, bold_data, y_tilde,
                                                   m_A, m_H, Sigma_A,
                                                   XX.astype(np.int32),
                                                   nb_voxels, hrf_len,
                                                   nb_conditions, nb_scans, scale,
                                                   sigma_h)
            if constrained:
                m_H = vt.norm1_constraint(m_H, hrf_covariance)
                hrf_covariance[:] = 0
            else:
                m_H[0] = 0
                m_H[-1] = 0
            logger.debug("After: m_H = %s, hrf_covariance = %s", m_H, hrf_covariance)
            h_norm.append(np.linalg.norm(m_H))
            # Normalizing H at each Nb2Norm iterations:
            if not constrained and NormFlag:
                # Normalizing is done before sigma_h, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (loop + 1) % Nb2Norm == 0:
                    Norm = np.linalg.norm(m_H)
                    m_H /= Norm
                    hrf_covariance /= Norm ** 2
                    m_A *= Norm
                    Sigma_A *= Norm ** 2
            # Plotting HRF
            if plot and loop >= 0:
                plt.figure(nb_conditions + 1)
                plt.plot(m_H)
                plt.hold(True)
        else:
            if true_hrf_flag:
                TrueVal, head = read_volume(hrf_filename)[:, 0, 0, 0]
                m_H = TrueVal

        DIFF = np.reshape(m_A - m_A1, (nb_conditions * nb_voxels))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        if np.linalg.norm(np.reshape(m_A1, (nb_conditions * nb_voxels))) > 0:
            Crit_A = (
                np.linalg.norm(DIFF) /
                np.linalg.norm(np.reshape(m_A1, (nb_conditions * nb_voxels)))) ** 2
        else:
            # TODO: norm shouldn't be 0
            Crit_A = None
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        Crit_H = (np.linalg.norm(m_H - m_H1) / np.linalg.norm(m_H1)) ** 2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0, hrf_len):
            AH[:, :, d] = m_A[:, :] * m_H[d]
        DIFF = np.reshape(AH - AH1, (nb_conditions * nb_voxels * hrf_len))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        Crit_AH = (np.linalg.norm(DIFF) /
                   (np.linalg.norm(np.reshape(AH1, (nb_conditions * nb_voxels * hrf_len)))
                    + eps)) ** 2
        if np.linalg.norm(np.reshape(AH1, (nb_conditions * nb_voxels * hrf_len))) == 0:
            # TODO: norm shouldn't be 0
            logger.warning("AH norm should not be zero")

        logger.info("Convergence criteria: %f (Threshold = %f)",
                    Crit_AH, thresh)
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        if estimate_labels:
            logger.info("Expectation Z step...")
            logger.debug("Before: Z_tilde = %s, q_Z = %s", Z_tilde, q_Z)
            labels_expectation = UtilsC.expectation_Z(
                Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z,
                neighbours_indexes.astype(np.int32), nb_conditions, nb_voxels,
                nb_classes, max_neighbours)
            logger.debug("After: Z_tilde = %s, q_Z = %s", Z_tilde, q_Z)
        else:
            logger.info("Using True Z ...")
            TrueZ = read_volume(labels_filename)
            for m in xrange(nb_conditions):
                q_Z[m, 1, :] = np.reshape(TrueZ[0][:, :, :, m], nb_voxels)
                q_Z[m, 0, :] = 1 - q_Z[m, 1, :]

        DIFF = np.reshape(q_Z - q_Z1, (nb_conditions * nb_classes * nb_voxels))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        if np.linalg.norm(np.reshape(q_Z1, (nb_conditions * nb_classes * nb_voxels))) > 0:
            Crit_Z = (np.linalg.norm(DIFF) /
                      (np.linalg.norm(np.reshape(q_Z1, (nb_conditions * nb_classes * nb_voxels))) + eps)) ** 2
        else:
            # TODO: norm shouldn't be 0
            Crit_Z = None
        cZ += [Crit_Z]
        q_Z1[:, :, :] = q_Z[:, :, :]

        if estimate_hrf and estimate_sigma_h:
            logger.info("Maximization sigma_H step...")
            logger.debug("Before: sigma_h = %s", sigma_h)
            if gamma_h > 0:
                sigma_h = vt.maximization_sigmaH_prior(hrf_len, hrf_covariance,
                                                       hrf_regularization_prior,
                                                       m_H, gamma_h)
            else:
                sigma_h = vt.maximization_sigmaH(hrf_len, hrf_covariance,
                                                 hrf_regularization_prior, m_H)
            logger.debug("After: sigma_h = %s", sigma_h)

        logger.info("Maximization (mu,sigma) step...")
        logger.debug("Before: mu_M = %s, sigma_M = %s", mu_M, sigma_M)
        mu_M, sigma_M = vt.maximization_mu_sigma(mu_M, sigma_M, q_Z, m_A,
                                                 nb_classes, nb_conditions, Sigma_A)
        logger.debug("After: mu_M = %s, sigma_M = %s", mu_M, sigma_M)

        for m in xrange(nb_conditions):
            SUM_q_Z[m] += [sum(q_Z[m, 1, :])]
            mu1[m] += [mu_M[m, 1]]

        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        UtilsC.maximization_L(bold_data, m_A, m_H, drift_coeffs, drift_basis,
                              XX.astype(np.int32), nb_voxels, hrf_len,
                              nb_conditions, Ndrift, nb_scans)
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = np.dot(drift_basis, drift_coeffs)
        y_tilde = bold_data - drift
        if estimate_beta:
            logger.info("Maximization beta step...")
            for m in xrange(0, nb_conditions):
                Beta[m] = UtilsC.maximization_beta(
                    Beta[m], q_Z[m, :, :].astype(np.float64),
                    Z_tilde[m, :, :].astype(np.float64), nb_voxels, nb_classes,
                    neighbours_indexes.astype(np.int32), gamma, max_neighbours,
                    it_max_grad, gradient_step)
            logger.debug("Beta = %s", str(Beta))

        logger.info("Maximization sigma noise step...")
        UtilsC.maximization_sigma_noise(Gamma, drift, sigma_epsilone, hrf_covariance,
                                        bold_data, m_A, m_H, Sigma_A,
                                        XX.astype(np.int32), nb_voxels, hrf_len,
                                        nb_conditions, nb_scans)

        #### Computing Free Energy ####
        if estimate_hrf and estimate_labels and estimate_free_energy:
            free_energy_prev = free_energy[-1]
            free_energy.append(vt.free_energy_computation(Sigma_A, hrf_covariance, q_Z,
                                                          nb_voxels, hrf_len,
                                                          nb_conditions,
                                                          nrls_expectation,
                                                          hrf_expectation,
                                                          labels_expectation))
            free_energy_crit.append((free_energy_prev - free_energy) /
                                    free_energy_prev)

        loop += 1
        compute_time.append(time.time() - start_time)

    SUM_q_Z_array = np.zeros((nb_conditions, loop), dtype=np.float64)
    mu1_array = np.zeros((nb_conditions, loop), dtype=np.float64)
    h_norm_array = np.zeros((loop), dtype=np.float64)
    for m in xrange(nb_conditions):
        for i in xrange(loop):
            SUM_q_Z_array[m, i] = SUM_q_Z[m][i]
            mu1_array[m, i] = mu1[m][i]
    h_norm_array = np.array(h_norm)

    if plot:
        font = {'size': 15}
        matplotlib.rc('font', **font)
        plt.figure(1)
        plt.plot(cAH[1:], 'lightblue')
        plt.hold(True)
        plt.legend(('CAH'))
        plt.grid(True)
        plt.savefig('./Crit_CompMod.png')

    compute_time_mean = compute_time[-1] / loop

    density_ratio = np.nan
    density_ratio_cano = np.nan
    density_ratio_diff = np.nan
    density_ratio_prod = np.nan
    variation_coeff = np.nan

    if estimate_hrf and not constrained and not NormFlag:
        Norm = np.linalg.norm(m_H)
        m_H /= Norm
        hrf_covariance /= Norm ** 2
        sigma_h /= Norm ** 2
        m_A *= Norm
        Sigma_A *= Norm ** 2
        mu_M *= Norm
        sigma_M *= Norm ** 2
        density_ratio = -(m_H.T.dot(np.linalg.inv(hrf_covariance)).dot(m_H)/2.)
        density_ratio_cano = -((m_H-m_h).T.dot(np.linalg.inv(hrf_covariance)).dot(m_H-m_h)/2.)
        density_ratio_diff = density_ratio_cano - density_ratio
        density_ratio_prod = density_ratio_cano * density_ratio
        variation_coeff = np.sqrt((m_H.T.dot(hrf_covariance).dot(m_H))/(m_H.T.dot(m_H))**2)
    sigma_M = np.sqrt(np.sqrt(sigma_M))

    ppm_a_nrl = np.zeros((nb_voxels, nb_conditions))
    ppm_g_nrl = np.zeros((nb_voxels, nb_conditions))
    for condition in range(nb_conditions):
        ppm_a_nrl[:, condition] = cpt_ppm_a_norm(m_A[:, condition], Sigma_A[condition, condition, :], 0)
        ppm_g_nrl[:, condition] = cpt_ppm_g_norm(m_A[:, condition], Sigma_A[condition, condition, :], 0.95)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#

    ppm_a_contrasts = np.zeros((nb_voxels, len(contrasts)))
    ppm_g_contrasts = np.zeros((nb_voxels, len(contrasts)))

    if compute_contrasts:
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            nrls_conds = dict([(str(cn), m_A[:, ic])
                               for ic, cn in enumerate(condition_names)])

            for n, cname in enumerate(contrasts):
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(nb_conditions, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, nb_conditions):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, nb_conditions):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(nb_conditions, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, nb_voxels):
                    S_tmp = Sigma_A[:, :, j]
                    CONTRASTVAR[j, n] = np.dot(np.dot(AC, S_tmp), AC)
                #------------ variance -------------#
                logger.info('Done contrasts computing.')

                ppm_a_contrasts[:, n] = cpt_ppm_a_norm(contrast, CONTRASTVAR[:, n], 0)
                ppm_g_contrasts[:, n] = cpt_ppm_g_norm(contrast, CONTRASTVAR[:, n], 0.95)

        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    logger.info("Nb iterations to reach criterion: %d", loop)
    logger.info("Computational time = %s min %s s",
                *(str(int(x)) for x in divmod(compute_time[-1], 60)))
    logger.debug('mu_M: %s', mu_M)
    logger.debug('sigma_M: %s', sigma_M)
    logger.debug("sigma_H = %s", str(sigma_h))
    logger.debug("Beta = %s", str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, nb_voxels, nb_scans)
    SNR = 20 * np.log(
        np.linalg.norm(bold_data) / np.linalg.norm(bold_data - StimulusInducedSignal - drift))
    SNR /= np.log(10.)
    logger.info('SNR comp = %f', SNR)
    # ,FreeEnergyArray
    return (loop, m_A, m_H, hrf_covariance, q_Z, sigma_epsilone, mu_M, sigma_M,
            Beta, drift_coeffs, drift, CONTRAST, CONTRASTVAR, cA[2:], cH[2:],
            cZ[2:], cAH[2:], compute_time[2:], compute_time_mean, Sigma_A,
            StimulusInducedSignal, density_ratio, density_ratio_cano,
            density_ratio_diff, density_ratio_prod, ppm_a_nrl, ppm_g_nrl,
            ppm_a_contrasts, ppm_g_contrasts, variation_coeff)


#  @profile
def jde_vem_bold_fast_python(graph, bold_data, onsets, hrf_duration, nb_classes,
                             tr, beta, dt, estimate_sigma_h=True, sigma_h=0.05,
                             it_max=-1, it_min=0, estimate_beta=True, contrasts=None,
                             compute_contrasts=False, hrf_hyperprior=0, estimate_hrf=True,
                             constrained=False, seed=6537546):
    """This is the main function that computes the VEM analysis on BOLD data.
    This function uses optimized python functions.

    Parameters
    ----------
    graph : # TODO
        # TODO
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    onsets : dict
        dictionnary of onsets
    hrf_duration : float
        hrf total time duration (in s)
    nb_classes : # TODO
        # TODO
    tr : float
        time of repetition
    beta : # TODO
        # TODO
    dt : float
        hrf temporal precision
    estimate_sigma_h : bool, optional
        toggle estimation of sigma H
    sigma_h : float, optional
        initial or fixed value of sigma H
    it_max : int, optional
        maximal computed iteration number
    it_min : int, optional
        minimal computed iteration number
    estimate_beta : bool, optional
        toggle the estimation of beta
    contrasts : OrderedDict, optional
        dict of contrasts to compute
    compute_contrasts : bool, optional
        if True, compute the contrasts defined in contrasts
    hrf_hyperprior : float (# TODO: check)
        # TODO
    estimate_hrf : bool, optional
        if True, estimate the HRF for each parcel, if False use the canonical
        HRF
    constrained : bool, optional
        if True, constrains the l2 norm of the HRF to 1
    seed : int, optional
        seed used by numpy to initialize random generator number

    Returns
    -------
    loop : int
        number of iterations before convergence
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
        Neural response level mean value
    hrf_mean : ndarray, shape (hrf_len,)
        Hemodynamic response function mean value
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
        Covariance matrix of the HRF
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
        # TODO
    noise_var : ndarray, shape (nb_voxels,)
        # TODO
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    beta : ndarray, shape (nb_conditions,)
        # TODO
    drift_coeffs : ndarray, shape (# TODO)
        # TODO
    drift : ndarray, shape (# TODO)
        # TODO
    CONTRAST : ndarray, shape (nb_voxels, len(contrasts))
        Contrasts computed from NRLs
    CONTRASTVAR : ndarray, shape (nb_voxels, len(contrasts))
        Variance of the contrasts
    nrls_criteria : list
        NRL criteria (# TODO: explain)
    hrf_criteria : list
        HRF criteria (# TODO: explain)
    labels_criteria : list
        Z criteria (# TODO: explain)
    nrls_hrf_criteria : list
        NRLs HRF product criteria
    compute_time : list
        computation time of each iteration
    compute_time_mean : float
        computation mean time over iterations
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
        # TODO
    stimulus_induced_signal : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    density_ratio : float
        # TODO
    density_ratio_cano : float
        # TODO
    density_ratio_diff : float
        # TODO
    density_ratio_prod : float
        # TODO
    ppm_a_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_a_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    variation_coeff : float
        coefficient of variation of the HRF

    Notes
    -----
        See `A novel definition of the multivariate coefficient of variation <http://onlinelibrary.wiley.com/doi/10.1002/bimj.201000030/abstract>`_
        article for more information about the coefficient of variation.
    """

    logger.info("Fast EM with C extension started.")

    if not contrasts:
        contrasts = OrderedDict()

    np.random.seed(seed)

    nb_2_norm = 1
    normalizing = False
    regularizing = False
    estimate_free_energy = False

    if it_max < 0:
        it_max = 100
    gamma = 7.5
    gradient_step = 0.003
    it_max_grad = 200
    thresh = 1e-5
    thresh_free_energy = 1e-5

    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_conditions = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    condition_names = []

    max_neighbours = max([len(nl) for nl in graph])
    neighbours_indexes = [np.concatenate((arr, np.zeros(max_neighbours-len(arr))-1))
                          for arr in graph]
    neighbours_indexes = np.asarray(neighbours_indexes, dtype=int)

    X = OrderedDict()
    for condition, onset in onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(nb_scans, tr, hrf_len, dt, onset)
        condition_names.append(condition)
    occurence_matrix = np.zeros((nb_conditions, nb_scans, hrf_len), dtype=np.int32)
    for nc, condition in enumerate(onsets.iterkeys()):
        occurence_matrix[nc, :, :] = X[condition]

    order = 2
    if regularizing:
        regularization = np.ones(hrf_len)
        regularization[hrf_len//3:hrf_len//2] = 2
        regularization[hrf_len//2:2*hrf_len//3] = 5
        regularization[2*hrf_len//3:3*hrf_len//4] = 7
        regularization[3*hrf_len//4:] = 10
        # regularization[hrf_len//2:] = 10
    else:
        regularization = None
    D2 = vt.buildFiniteDiffMatrix(order, hrf_len, regularization)
    hrf_regu_prior_inv = D2.dot(D2) / pow(dt, 2 * order)

    noise_struct = np.identity(nb_scans)

    hrf_criteria = [1.]
    labels_criteria = [1.]
    nrls_criteria = [1.]
    nrls_hrf_criteria = [1.]
    free_energy_crit = [1.]
    compute_time = []

    #  nrls_hrf_prev = np.zeros((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)
    # initialized to ones to avoid numerical isssues
    nrls_hrf_prev = np.ones((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    #  Q_barnCond = np.zeros((nb_conditions, nb_conditions, hrf_len, hrf_len), dtype=np.float64)
    #  XGamma = np.zeros((nb_conditions, hrf_len, nb_scans), dtype=np.float64)
    #  m1 = 0
    #  for k1 in X:  # Loop over the nb_conditions conditions
        #  m2 = 0
        #  for k2 in X:
            #  Q_barnCond[m1, m2, :, :] = np.dot(
                #  np.dot(X[k1].transpose(), noise_struct), X[k2])
            #  m2 += 1
        #  XGamma[m1, :, :] = np.dot(X[k1].transpose(), noise_struct)
        #  m1 += 1
    #  XGamma = np.tensordot(occurence_matrix.T, noise_struct, axes=(1, 0)).transpose(1, 0, 2)
    #  Q_barnCond = np.tensordot(XGamma, occurence_matrix, axes=(2, 1)).transpose(0, 2, 1, 3)

    noise_var = np.ones(nb_voxels)

    logger.info("Labels are initialized by setting active probabilities to ones ...")
    labels_proba = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    labels_proba[:, 1, :] = 1

    labels_proba_prev = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    #  Z_tilde = labels_proba.copy()

    # TODO: replace every variable/variable_prev by collections.deque
    m_h = getCanoHRF(hrf_duration, dt)[1][:hrf_len]
    hrf_mean = np.array(m_h).astype(np.float64)
    hrf_mean_prev = np.array(m_h)
    if estimate_hrf:
        hrf_covar = np.ones((hrf_len, hrf_len), dtype=np.float64)
    else:
        hrf_covar = np.zeros((hrf_len, hrf_len), dtype=np.float64)

    beta = beta * np.ones((nb_conditions), dtype=np.float64)
    drift_basis = vt.PolyMat(nb_scans, 4, tr)
    drift_coeffs = vt.poly_fit(bold_data, drift_basis)
    drift = drift_basis.dot(drift_coeffs)
    y_tilde = bold_data - drift

    nrls_class_var = np.ones((nb_conditions, nb_classes), dtype=np.float64)
    nrls_class_var[:, 0] = 0.5
    nrls_class_var[:, 1] = 0.6
    #  nrls_class_mean = np.zeros((nb_conditions, nb_classes), dtype=np.float64)
    #  for k in xrange(1, nb_classes):
        #  nrls_class_mean[:, k] = 1  # init_mean
    nrls_covar = 0.01 * (np.identity(nb_conditions)[:, :, np.newaxis]
                         + np.zeros((1, 1, nb_voxels)))
    nrls_class_mean = 14 * np.ones((nb_conditions, nb_classes))
    nrls_class_mean[:, 0] = 0
    #  nrls_mean = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    #  nrls_mean_prev = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    #  for j in xrange(0, nb_voxels):
        #  for m in xrange(0, nb_conditions):
            #  for k in xrange(0, nb_classes):
                #  nrls_mean[j, m] += np.random.normal(
                    #  nrls_class_mean[m, k], np.sqrt(nrls_class_var[m, k])) * labels_proba[m, k, j]
    nrls_mean = (np.random.normal(
        nrls_class_mean, nrls_class_var)[:, :, np.newaxis] * labels_proba).sum(axis=1).T
    nrls_mean = np.zeros((nb_voxels, nb_conditions))
    #  nrls_mean_prev = nrls_mean.copy()
    # initialized to ones to avoid numerical issues
    nrls_mean_prev = np.ones((nb_voxels, nb_conditions))

    start_time = time.time()
    free_energy = [0.]
    loop = 0
    while (loop < it_min + 1) or (free_energy_crit[-1] > thresh_free_energy and
                                  nrls_hrf_criteria[-1] > thresh and loop < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(loop+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        #  nrls_expectation = UtilsC.expectation_A(labels_proba, nrls_class_mean, nrls_class_var, drift,
                                                #  noise_var, noise_struct, hrf_covar,
                                                #  bold_data, y_tilde, nrls_mean, hrf_mean,
                                                #  nrls_covar, occurence_matrix.astype(np.int32),
                                                #  nb_voxels, hrf_len, nb_conditions,
                                                #  nb_scans, nb_classes)
        nrls_mean, nrls_covar = vt.nrls_expectation(
            hrf_mean, nrls_mean, occurence_matrix, noise_struct, labels_proba,
            nrls_class_mean, nrls_class_var, nb_conditions, y_tilde, nrls_covar,
            hrf_covar, noise_var)
        logger.debug("After: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        if estimate_hrf:
            logger.info("Expectation H step...")
            logger.debug("Before: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            #  hrf_expectation = UtilsC.expectation_H(XGamma, Q_barnCond,
                                                   #  noise_var, noise_struct, hrf_regu_prior_inv,
                                                   #  hrf_covar, bold_data, y_tilde,
                                                   #  nrls_mean, hrf_mean, nrls_covar,
                                                   #  occurence_matrix.astype(np.int32),
                                                   #  nb_voxels, hrf_len,
                                                   #  nb_conditions, nb_scans, scale,
                                                   #  sigma_h)
            hrf_mean, hrf_covar = vt.hrf_expectation(
                nrls_covar, nrls_mean, occurence_matrix, noise_struct,
                hrf_regu_prior_inv, sigma_h, nb_voxels, y_tilde, noise_var)
            if constrained:
                hrf_mean = vt.norm1_constraint(hrf_mean, hrf_covar)
                hrf_covar[:] = 0
            else:
                hrf_mean[0] = 0
                hrf_mean[-1] = 0
            logger.debug("After: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            # Normalizing H at each nb_2_norm iterations:
            if not constrained and normalizing:
                # Normalizing is done before sigma_h, nrls_class_mean and nrls_class_var estimation
                # we should not include them in the normalisation step
                if (loop + 1) % nb_2_norm == 0:
                    hrf_norm = np.linalg.norm(hrf_mean)
                    hrf_mean /= hrf_norm
                    hrf_covar /= hrf_norm ** 2
                    nrls_mean *= hrf_norm
                    nrls_covar *= hrf_norm ** 2

        DIFF = np.reshape(nrls_mean - nrls_mean_prev, (nb_conditions * nb_voxels))
        diff = np.linalg.norm((nrls_mean - nrls_mean_prev).flatten())
        nrls_mean_prev_norm = np.linalg.norm(nrls_mean_prev.flatten())
        nrls_criteria.append(diff / nrls_mean_prev_norm**2
                             if nrls_mean_prev_norm else None)
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  if np.linalg.norm(np.reshape(nrls_mean_prev, (nb_conditions * nb_voxels))) > 0:
            #  nrls_criteria.append((
                #  np.linalg.norm(DIFF) /
                #  np.linalg.norm(np.reshape(nrls_mean_prev, (nb_conditions * nb_voxels)))) ** 2)
        #  else:
            #  # TODO: norm shouldn't be 0
            #  nrls_criteria.append(None)
        nrls_mean_prev[:, :] = nrls_mean[:, :]

        hrf_criteria.append((np.linalg.norm(hrf_mean - hrf_mean_prev) / np.linalg.norm(hrf_mean_prev)) ** 2)
        hrf_mean_prev[:] = hrf_mean[:]

        #  for d in xrange(0, hrf_len):
            #  nrls_hrf[:, :, d] = nrls_mean[:, :] * hrf_mean[d]
        nrls_hrf = nrls_mean[:, :, np.newaxis] * hrf_mean

        #  DIFF = np.reshape(nrls_hrf - nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len))
        diff = np.linalg.norm((nrls_hrf - nrls_hrf_prev).flatten())
        nrls_hrf_prev_norm = np.linalg.norm(nrls_hrf_prev.flatten())
        nrls_hrf_criteria.append(diff / nrls_hrf_prev_norm**2
                                 if nrls_hrf_prev_norm else None)
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  nrls_hrf_criteria.append((np.linalg.norm(DIFF) /
                   #  (np.linalg.norm(np.reshape(nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len)))
                    #  + eps)) ** 2)
        #  if np.linalg.norm(np.reshape(nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len))) == 0:
            #  # TODO: norm shouldn't be 0
            #  logger.warning("nrls_hrf norm should not be zero")
            #  nrls_hrf_criteria.append(None)

        logger.info("Convergence criteria: %f (Threshold = %f)",
                    nrls_hrf_criteria[-1], thresh)
        #  nrls_hrf_prev[:, :, :] = nrls_hrf[:, :, :]
        nrls_hrf_prev = nrls_hrf.copy()

        logger.info("Expectation Z step...")
        logger.debug("Before: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)
        #  labels_expectation = UtilsC.expectation_Z(
            #  nrls_covar, nrls_mean, nrls_class_var, beta, labels_proba, nrls_class_mean, labels_proba,
            #  neighbours_indexes.astype(np.int32), nb_conditions, nb_voxels,
            #  nb_classes, max_neighbours)
        labels_proba = vt.labels_expectation(
            nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean, beta,
            labels_proba, neighbours_indexes, nb_conditions, nb_classes)
        logger.debug("After: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)

        diff = np.linalg.norm((labels_proba - labels_proba_prev).flatten())
        labels_proba_prev_norm = np.linalg.norm(labels_proba_prev.flatten())
        labels_criteria.append(diff / labels_proba_prev_norm**2
                               if labels_proba_prev_norm else None)
        #  DIFF = np.reshape(labels_proba - labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  if np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) > 0:
            #  labels_criteria.append((np.linalg.norm(DIFF) /
                      #  (np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) + eps)) ** 2)
        #  else:
            #  # TODO: norm shouldn't be 0
            #  labels_criteria.append(None)
        #  labels_proba_prev[:, :, :] = labels_proba[:, :, :]

        if estimate_hrf and estimate_sigma_h:
            logger.info("Maximization sigma_H step...")
            logger.debug("Before: sigma_h = %s", sigma_h)
            if hrf_hyperprior > 0:
                sigma_h = vt.maximization_sigmaH_prior(hrf_len, hrf_covar,
                                                       hrf_regu_prior_inv,
                                                       hrf_mean, hrf_hyperprior)
            else:
                sigma_h = vt.maximization_sigmaH(hrf_len, hrf_covar,
                                                 hrf_regu_prior_inv, hrf_mean)
            logger.debug("After: sigma_h = %s", sigma_h)

        logger.info("Maximization (mu,sigma) step...")
        logger.debug("Before: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)
        nrls_class_mean, nrls_class_var = vt.maximization_mu_sigma(
            nrls_class_mean, nrls_class_var, labels_proba, nrls_mean,
            nb_classes, nb_conditions, nrls_covar)
        logger.debug("After: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)

        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        UtilsC.maximization_L(bold_data, nrls_mean, hrf_mean, drift_coeffs, drift_basis,
                              occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                              nb_conditions, drift_coeffs.shape[0], nb_scans)
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = drift_basis.dot(drift_coeffs)
        y_tilde = bold_data - drift
        if estimate_beta:
            logger.info("Maximization beta step...")
            for m in xrange(0, nb_conditions):
                beta[m] = UtilsC.maximization_beta(
                    beta[m], labels_proba[m, :, :].astype(np.float64),
                    labels_proba[m, :, :].astype(np.float64), nb_voxels, nb_classes,
                    neighbours_indexes.astype(np.int32), gamma, max_neighbours,
                    it_max_grad, gradient_step)
                #  beta[m] = UtilsC.maximization_beta_CB(
                    #  beta[m], labels_proba[m, :, :].astype(np.float64),
                    #  nb_voxels, nb_classes, neighbours_indexes.astype(np.int32),
                    #  gamma, max_neighbours, it_max_grad, gradient_step)
            logger.debug("beta = %s", str(beta))

        logger.info("Maximization sigma noise step...")
        UtilsC.maximization_sigma_noise(noise_struct, drift, noise_var, hrf_covar,
                                        bold_data, nrls_mean, hrf_mean, nrls_covar,
                                        occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                                        nb_conditions, nb_scans)

        #### Computing Free Energy ####
        if estimate_hrf and estimate_free_energy:
            free_energy_prev = free_energy[-1]
            nrls_expectation = None
            hrf_expectation = None
            labels_expectation = None
            free_energy.append(vt.free_energy_computation(nrls_covar, hrf_covar, labels_proba,
                                                          nb_voxels, hrf_len,
                                                          nb_conditions,
                                                          nrls_expectation,
                                                          hrf_expectation,
                                                          labels_expectation))
            free_energy_crit.append((free_energy_prev - free_energy) /
                                    free_energy_prev)

        loop += 1
        compute_time.append(time.time() - start_time)

    compute_time_mean = compute_time[-1] / loop

    density_ratio = np.nan
    density_ratio_cano = np.nan
    density_ratio_diff = np.nan
    density_ratio_prod = np.nan
    variation_coeff = np.nan

    if estimate_hrf and not constrained and not normalizing:
        hrf_norm = np.linalg.norm(hrf_mean)
        hrf_mean /= hrf_norm
        hrf_covar /= hrf_norm ** 2
        sigma_h /= hrf_norm ** 2
        nrls_mean *= hrf_norm
        nrls_covar *= hrf_norm ** 2
        nrls_class_mean *= hrf_norm
        nrls_class_var *= hrf_norm ** 2
        density_ratio = -(hrf_mean.T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean)/2.)
        density_ratio_cano = -((hrf_mean-m_h).T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean-m_h)/2.)
        density_ratio_diff = density_ratio_cano - density_ratio
        density_ratio_prod = density_ratio_cano * density_ratio
        variation_coeff = np.sqrt((hrf_mean.T.dot(hrf_covar).dot(hrf_mean))/(hrf_mean.T.dot(hrf_mean))**2)
    nrls_class_var = np.sqrt(np.sqrt(nrls_class_var))

    ppm_a_nrl = np.zeros((nb_voxels, nb_conditions))
    ppm_g_nrl = np.zeros((nb_voxels, nb_conditions))
    for condition in xrange(nb_conditions):
        #  ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0)
        ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 3*nrls_class_var[condition, 0]**.5)
        ppm_g_nrl[:, condition] = cpt_ppm_g_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0.95)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#

    ppm_a_contrasts = np.zeros((nb_voxels, len(contrasts)))
    ppm_g_contrasts = np.zeros((nb_voxels, len(contrasts)))

    if compute_contrasts:
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            nrls_conds = dict([(str(cn), nrls_mean[:, ic])
                               for ic, cn in enumerate(condition_names)])

            for n, cname in enumerate(contrasts):
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(nb_conditions, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, nb_conditions):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, nb_conditions):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(nb_conditions, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, nb_voxels):
                    S_tmp = nrls_covar[:, :, j]
                    CONTRASTVAR[j, n] = AC.dot(S_tmp).dot(AC)
                #------------ variance -------------#
                logger.info('Done contrasts computing.')

                ppm_a_contrasts[:, n] = cpt_ppm_a_norm(contrast, CONTRASTVAR[:, n], 0)
                ppm_g_contrasts[:, n] = cpt_ppm_g_norm(contrast, CONTRASTVAR[:, n], 0.95)

        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    logger.info("Nb iterations to reach criterion: %d", loop)
    logger.info("Computational time = %s min %s s",
                *(str(int(x)) for x in divmod(compute_time[-1], 60)))
    logger.debug('nrls_class_mean: %s', nrls_class_mean)
    logger.debug('nrls_class_var: %s', nrls_class_var)
    logger.debug("sigma_H = %s", str(sigma_h))
    logger.debug("beta = %s", str(beta))

    stimulus_induced_signal = vt.computeFit(hrf_mean, nrls_mean, X, nb_voxels, nb_scans)
    snr = 20 * np.log(
        np.linalg.norm(bold_data) / np.linalg.norm(bold_data - stimulus_induced_signal - drift))
    snr /= np.log(10.)
    logger.info('snr comp = %f', snr)
    # ,FreeEnergyArray
    return (loop, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var,
            nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,
            CONTRAST, CONTRASTVAR, nrls_criteria[2:], hrf_criteria[2:],
            labels_criteria[2:], nrls_hrf_criteria[2:], compute_time[2:],
            compute_time_mean, nrls_covar, stimulus_induced_signal, density_ratio,
            density_ratio_cano, density_ratio_diff, density_ratio_prod, ppm_a_nrl,
            ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff)

def jde_vem_bold_soustraction(graph, bold_data, onsets, durations, hrf_duration, nb_classes,
                              tr, beta, dt, estimate_sigma_h=True, sigma_h=0.05,
                              it_max=-1, it_min=0, estimate_beta=True, contrasts=None,
                              compute_contrasts=False, hrf_hyperprior=0, estimate_hrf=True,
                              constrained=False, seed=6537546, labels_proba_filename=None):
    """This is the main function that computes the VEM analysis on BOLD data.
    This function uses optimized python functions.

    Parameters
    ----------
    graph : # TODO
        # TODO
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    onsets : dict
        dictionnary of onsets
    hrf_duration : float
        hrf total time duration (in s)
    nb_classes : # TODO
        # TODO
    tr : float
        time of repetition
    beta : # TODO
        # TODO
    dt : float
        hrf temporal precision
    estimate_sigma_h : bool, optional
        toggle estimation of sigma H
    sigma_h : float, optional
        initial or fixed value of sigma H
    it_max : int, optional
        maximal computed iteration number
    it_min : int, optional
        minimal computed iteration number
    estimate_beta : bool, optional
        toggle the estimation of beta
    contrasts : OrderedDict, optional
        dict of contrasts to compute
    compute_contrasts : bool, optional
        if True, compute the contrasts defined in contrasts
    hrf_hyperprior : float (# TODO: check)
        # TODO
    estimate_hrf : bool, optional
        if True, estimate the HRF for each parcel, if False use the canonical
        HRF
    constrained : bool, optional
        if True, constrains the l2 norm of the HRF to 1
    seed : int, optional
        seed used by numpy to initialize random generator number

    Returns
    -------
    loop : int
        number of iterations before convergence
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
        Neural response level mean value
    hrf_mean : ndarray, shape (hrf_len,)
        Hemodynamic response function mean value
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
        Covariance matrix of the HRF
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
        # TODO
    noise_var : ndarray, shape (nb_voxels,)
        # TODO
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    beta : ndarray, shape (nb_conditions,)
        # TODO
    drift_coeffs : ndarray, shape (# TODO)
        # TODO
    drift : ndarray, shape (# TODO)
        # TODO
    CONTRAST : ndarray, shape (nb_voxels, len(contrasts))
        Contrasts computed from NRLs
    CONTRASTVAR : ndarray, shape (nb_voxels, len(contrasts))
        Variance of the contrasts
    nrls_criteria : list
        NRL criteria (# TODO: explain)
    hrf_criteria : list
        HRF criteria (# TODO: explain)
    labels_criteria : list
        Z criteria (# TODO: explain)
    nrls_hrf_criteria : list
        NRLs HRF product criteria
    compute_time : list
        computation time of each iteration
    compute_time_mean : float
        computation mean time over iterations
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
        # TODO
    stimulus_induced_signal : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    density_ratio : float
        # TODO
    density_ratio_cano : float
        # TODO
    density_ratio_diff : float
        # TODO
    density_ratio_prod : float
        # TODO
    ppm_a_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_a_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    variation_coeff : float
        coefficient of variation of the HRF
    free_energy : list
        # TODO

    Notes
    -----
        See `A novel definition of the multivariate coefficient of variation <http://onlinelibrary.wiley.com/doi/10.1002/bimj.201000030/abstract>`_
        article for more information about the coefficient of variation.
    """

    #import ipdb
    #def p(): pass
    #ipdb.set_trace = p
    logger.info("Fast EM with C extension started.")

    if not contrasts:
        contrasts = OrderedDict()

    np.random.seed(seed)

    nb_2_norm = 1
    normalizing = False
    regularizing = False
    estimate_free_energy = True

    if it_max <= 0:
        it_max = 100
    gamma = 7.5
    gradient_step = 0.003
    it_max_grad = 200
    thresh = 1e-5
    thresh_free_energy = 1e-5

    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_conditions = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    X, occurence_matrix, condition_names = EM.create_conditions_block(onsets,
                        durations, nb_conditions, nb_scans, hrf_len, tr, dt)
    #X, occurence_matrix, condition_names = EM.create_conditions(onsets,
    #                            nb_conditions, nb_scans, hrf_len, tr, dt)
    max_neighbours = max([len(nl) for nl in graph])
    neighbours_indexes = [np.concatenate((arr, np.zeros(max_neighbours-len(arr))-1))
                          for arr in graph]
    neighbours_indexes = np.asarray(neighbours_indexes, dtype=int)

    order = 2
    if regularizing:
        regularization = np.ones(hrf_len)
        regularization[hrf_len//3:hrf_len//2] = 2
        regularization[hrf_len//2:2*hrf_len//3] = 5
        regularization[2*hrf_len//3:3*hrf_len//4] = 7
        regularization[3*hrf_len//4:] = 10
        # regularization[hrf_len//2:] = 10
    else:
        regularization = None
    D2 = vt.buildFiniteDiffMatrix(order, hrf_len, regularization)
    hrf_regu_prior_inv = D2.dot(D2) / pow(dt, 2 * order)

    noise_struct = np.identity(nb_scans)

    hrf_criteria = [1.]
    labels_criteria = [1.]
    nrls_criteria = [1.]
    nrls_hrf_criteria = [1.]
    free_energy = [1.]
    free_energy_crit = [1.]
    compute_time = []

    #  nrls_hrf_prev = np.zeros((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)
    # initialized to ones to avoid numerical isssues
    nrls_hrf_prev = np.ones((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    
    noise_var = np.ones(nb_voxels)

    labels_proba = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    labels_proba_prev = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    if not labels_proba_filename:
        #  logger.info("Labels are initialized by setting active probabilities to ones ...")
        #  labels_proba[:, 1, :] = 1
        logger.info("Labels are initialized by setting everything to 0.5")
        labels_proba[:, :, :] = 0.5
        #labels_proba[:, 1, :] = 1
    else:
        if isinstance(labels_proba_filename, dict):
            logger.info("Labels are initialized by using true labels file")
            for condition_name, label_proba_filename in labels_proba_filename.iteritems():
                true_label_proba = read_volume(labels_proba_filename[condition_name])
                labels_proba[condition_names.index(condition_name), 1, :] = true_label_proba[0].flatten()
        elif isinstance(label_proba_filename, str):
            logger.info("Labels are initialized by using true labels file")
            true_labels_proba = read_volume(labels_proba_filename)
            for condition_nb, condition_name in enumerate(condition_names):
                labels_proba[condition_nb, 1, :] = true_labels_proba.sub_cuboid(condition=condition_name).data
        else:
            logger.error('Unknown labels file format.\n'
                         'Labels are initialized by setting active probabilities to ones...')

    #  Z_tilde = labels_proba.copy()

    # TODO: replace every variable/variable_prev by collections.deque
    m_h = getCanoHRF(hrf_duration, dt)[1][:hrf_len]
    hrf_mean = np.array(m_h).astype(np.float64)
    hrf_mean_prev = np.array(m_h)
    if estimate_hrf:
        #  hrf_covar = np.ones((hrf_len, hrf_len), dtype=np.float64)
        hrf_covar = np.identity(hrf_len, dtype=np.float64)
    else:
        hrf_covar = np.zeros((hrf_len, hrf_len), dtype=np.float64)

    beta = beta * np.ones((nb_conditions), dtype=np.float64)
    beta_list = []
    beta_list.append(beta.copy())
    drift_basis = vt.PolyMat(nb_scans, 4, tr)
    drift_coeffs = vt.poly_fit(bold_data, drift_basis)
    drift = drift_basis.dot(drift_coeffs)
    bold_data_drift = bold_data - drift

    # Parameters Gaussian mixtures
    nrls_class_var = 0.3 * np.ones((nb_conditions, nb_classes), dtype=np.float64)
    #nrls_covar = (np.identity(nb_conditions)[:, :, np.newaxis] + np.zeros((1, 1, nb_voxels)))
    nrls_covar = np.ones((nb_conditions, nb_conditions, nb_voxels)) * np.identity(nb_conditions)[:, :, np.newaxis]
    nrls_class_mean = 2.2 * np.ones((nb_conditions, nb_classes))
    nrls_class_mean[:, 0] = 0
    nrls_mean = (np.random.normal(
        nrls_class_mean, np.sqrt(nrls_class_var))[:, :, np.newaxis] * labels_proba).sum(axis=1).T
    nrls_mean_prev = np.ones_like(nrls_mean)

    logger.info("After: nrls_class_mean = %s, nrls_class_var = %s", nrls_class_mean, nrls_class_var)
    
    start_time = time.time()
    loop = 0
    while (loop < it_min + 1) or (free_energy_crit[-1] > thresh_free_energy and loop < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(loop+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)

        nrls_mean, nrls_covar = vt.nrls_expectation(
            hrf_mean, nrls_mean, occurence_matrix, noise_struct, labels_proba,
            nrls_class_mean, nrls_class_var, nb_conditions, bold_data_drift, nrls_covar,
            hrf_covar, noise_var)
        free_energy_tmp = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_tmp < free_energy[-1] and loop > 0:
            logger.info("free energy has decreased after nrls computation from %f to %f", free_energy[-1], free_energy_tmp)        
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        logger.debug("After: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        if estimate_hrf:
            logger.info("Expectation H step...")
            logger.debug("Before: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            hrf_mean, hrf_covar = vt.hrf_expectation(
                nrls_covar, nrls_mean, occurence_matrix, noise_struct,
                hrf_regu_prior_inv, sigma_h, nb_voxels, bold_data_drift, noise_var)
            free_energy_tmp = vt.free_energy_computation(
                    nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                    bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                    nb_voxels, nb_scans, nb_classes,  nrls_class_mean, nrls_class_var, neighbours_indexes,
                    beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
                )
            if free_energy_tmp < free_energy[-1] and loop > 0:
                logger.info("free energy has decreased after hrf computation from %f to %f", free_energy[-1], free_energy_tmp)        
            #if free_energy_tmp < free_energy[-1] and loop > 0:
            #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
            if constrained:
                hrf_mean = vt.norm1_constraint(hrf_mean, hrf_covar)
                hrf_covar[:] = 0
            else:
                hrf_mean[0] = 0
                hrf_mean[-1] = 0
            logger.debug("After: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            # Normalizing H at each nb_2_norm iterations:
            if not constrained and normalizing:
                # Normalizing is done before sigma_h, nrls_class_mean and nrls_class_var estimation
                # we should not include them in the normalisation step
                if (loop + 1) % nb_2_norm == 0:
                    hrf_norm = np.linalg.norm(hrf_mean)
                    hrf_mean /= hrf_norm
                    hrf_covar /= hrf_norm ** 2
                    nrls_mean *= hrf_norm
                    nrls_covar *= hrf_norm ** 2

        diff = np.linalg.norm((nrls_mean - nrls_mean_prev).flatten())
        nrls_mean_prev_norm = np.linalg.norm(nrls_mean_prev.flatten())
        nrls_criteria.append(diff / nrls_mean_prev_norm**2 if nrls_mean_prev_norm else None)
        nrls_mean_prev[:, :] = nrls_mean[:, :]

        hrf_criteria.append((np.linalg.norm(hrf_mean - hrf_mean_prev) / np.linalg.norm(hrf_mean_prev)) ** 2)
        hrf_mean_prev[:] = hrf_mean[:]
        nrls_hrf = nrls_mean[:, :, np.newaxis] * hrf_mean

        diff = np.linalg.norm((nrls_hrf - nrls_hrf_prev).flatten())
        nrls_hrf_prev_norm = np.linalg.norm(nrls_hrf_prev.flatten())
        nrls_hrf_criteria.append(diff / nrls_hrf_prev_norm**2
                                 if nrls_hrf_prev_norm else None)
        nrls_hrf_prev = nrls_hrf.copy()

        logger.info("Expectation Z step...")
        logger.debug("Before: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)
        #  labels_expectation = UtilsC.expectation_Z(
            #  nrls_covar, nrls_mean, nrls_class_var, beta, labels_proba, nrls_class_mean, labels_proba,
            #  neighbours_indexes.astype(np.int32), nb_conditions, nb_voxels,
            #  nb_classes, max_neighbours)
        labels_proba, _ = vt.expectation_Z(nrls_covar, nrls_mean, nrls_class_var, beta, \
                            labels_proba, nrls_class_mean, labels_proba, neighbours_indexes, \
                            nb_conditions, nb_voxels, nb_classes, np.zeros((nb_classes)))
        #labels_proba = vt.labels_expectation_soustraction(
        #    nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean, beta,
        #    labels_proba, neighbours_indexes, nb_conditions, nb_classes)
        logger.debug("After: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)
        logger.info("labels_proba = %s", labels_proba)

        diff = np.linalg.norm((labels_proba - labels_proba_prev).flatten())
        labels_proba_prev_norm = np.linalg.norm(labels_proba_prev.flatten())
        labels_criteria.append(diff / labels_proba_prev_norm**2
                               if labels_proba_prev_norm else None)
        #  if np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) > 0:
            #  labels_criteria.append((np.linalg.norm(DIFF) /
                      #  (np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) + eps)) ** 2)
        #  else:
            #  # TODO: norm shouldn't be 0
            #  labels_criteria.append(None)
        #  labels_proba_prev[:, :, :] = labels_proba[:, :, :]
        free_energy_E = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_E < free_energy[-1] and loop > 0:
            logger.info("free energy has decreased after expectation computation from %f to %f", free_energy[-1], free_energy_E)        
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################

        if estimate_hrf and estimate_sigma_h:
            logger.info("Maximization sigma_H step...")
            logger.debug("Before: sigma_h = %s", sigma_h)
            if hrf_hyperprior > 0:
                sigma_h = vt.maximization_sigmaH_prior(hrf_len, hrf_covar,
                                                       hrf_regu_prior_inv,
                                                       hrf_mean, hrf_hyperprior)
            else:
                sigma_h = vt.maximization_sigmaH(hrf_len, hrf_covar,
                                                 hrf_regu_prior_inv, hrf_mean)
            logger.debug("After: sigma_h = %s", sigma_h)

        free_energy_vh = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_vh < free_energy_E and loop > 0:
            logger.info("free energy has decreased after vh computation from %f to %f", free_energy_E, free_energy_vh)        
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        logger.info("Maximization (mu,sigma) step...")
        logger.debug("Before: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)
        nrls_class_mean, nrls_class_var = vt.maximization_class_proba(labels_proba, nrls_mean, nrls_covar)
        #nrls_class_mean, nrls_class_var = vt.maximization_mu_sigma(
        #    nrls_class_mean, nrls_class_var, labels_proba, nrls_mean, nb_classes,
        #    nb_conditions, nrls_covar)
        #  nrls_class_mean, nrls_class_var = vt.maximization_class_proba(
            #  labels_proba, nrls_mean, nrls_covar)
        logger.info("After: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)

        free_energy_GMM = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_GMM < free_energy_vh and loop > 0:
            logger.info("free energy has decreased after GMM params computation from %f to %f", free_energy_vh, free_energy_GMM)        
        #if free_energy_tmp < free_energy[-1] and loop > 0:
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        #  UtilsC.maximization_L(bold_data, nrls_mean, hrf_mean, drift_coeffs, drift_basis,
                              #  occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                              #  nb_conditions, drift_coeffs.shape[0], nb_scans)
        drift_coeffs = vt.maximization_drift_coeffs(
            bold_data, nrls_mean, occurence_matrix, hrf_mean, noise_struct, drift_basis
        )
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = drift_basis.dot(drift_coeffs)
        bold_data_drift = bold_data - drift
        free_energy_L = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_L < free_energy_vh and loop > 0:
            logger.info("free energy has decreased after drifts computation from %f to %f", free_energy_vh, free_energy_L)
        #if free_energy_tmp < free_energy[-1] and loop > 0:
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        if estimate_beta:
            logger.info("Maximization beta step...")
            Qtilde = np.concatenate((labels_proba, np.zeros((nb_conditions, nb_classes, 1), dtype=labels_proba.dtype)), axis=2)
            Qtilde_sumneighbour = labels_proba[:, :, neighbours_indexes].sum(axis=3)
            for m in xrange(0, nb_conditions):
                #  beta[m] = UtilsC.maximization_beta(
                    #  beta[m], labels_proba[m, :, :].astype(np.float64),
                    #  labels_proba[m, :, :].astype(np.float64), nb_voxels, nb_classes,
                    #  neighbours_indexes.astype(np.int32), gamma, max_neighbours,
                    #  it_max_grad, gradient_step)
                #  beta[m] = UtilsC.maximization_beta_CB(
                    #  beta[m], labels_proba[m, :, :].astype(np.float64),
                    #  nb_voxels, nb_classes, neighbours_indexes.astype(np.int32),
                    #  gamma, max_neighbours, it_max_grad, gradient_step)
                beta[m] = vt.beta_maximization(beta[m], labels_proba[m, :, :],
                                               neighbours_indexes, gamma, nb_classes,
                                               it_max_grad, gradient_step)
                #beta0 = beta.copy()
                """beta[m] = EMf.maximization_beta_m2(beta[m], labels_proba[m, :, :],
                                                   Qtilde_sumneighbour[m, :, :], Qtilde[m, :, :],
                                                   neighbours_indexes, max_neighbours, gamma,
                                                   it_max_grad, gradient_step)"""

            #  beta[beta < 0] = 0.1
            #  if (beta < 0).any():
                #  import ipdb; ipdb.set_trace() #################### Breakpoint ####################
            beta_list.append(beta.copy())
            #logger.info("beta0 = %s", str(beta0))
            logger.info("beta = %s", str(beta))

            free_energy_beta = vt.free_energy_computation(
                    nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                    bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                    nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                    beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
                )
            if free_energy_beta < free_energy_L and loop > 0:
                logger.info("free energy has decreased after beta computation from %f to %f", free_energy_L, free_energy_beta)
        else:
            free_energy_beta = free_energy_L
            #if free_energy_tmp < free_energy[-1] and loop > 0:
            #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        logger.info("Maximization sigma noise step...")
        #  UtilsC.maximization_sigma_noise(noise_struct, drift, noise_var, hrf_covar,
                                        #  bold_data, nrls_mean, hrf_mean, nrls_covar,
                                        #  occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                                        #  nb_conditions, nb_scans)
        noise_var = vt.maximization_noise_var(
            occurence_matrix, hrf_mean, hrf_covar, nrls_mean, nrls_covar,
            noise_struct, bold_data_drift, nb_scans
        )

        free_energy_n = vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            )
        if free_energy_n < free_energy_beta and loop > 0:
            logger.info("free energy has decreased after noise computation from %f to %f", free_energy_beta, free_energy_n)

        #if free_energy_tmp < free_energy[-1] and loop > 0:
        #    import ipdb; ipdb.set_trace() #################### Breakpoint ####################
        #### Computing Free Energy ####
        if estimate_hrf and estimate_free_energy:
            free_energy.append(vt.free_energy_computation(
                nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
                bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
                nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
                beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
            ))
            free_energy_crit.append(abs(abs(free_energy[-2] - free_energy[-1]) /
                                    free_energy[-2]))

        logger.info("Convergence criteria: %f (Threshold = %f)",
                    free_energy_crit[-1], thresh_free_energy)
        loop += 1
        compute_time.append(time.time() - start_time)

    compute_time_mean = compute_time[-1] / loop

    density_ratio = np.nan
    density_ratio_cano = np.nan
    density_ratio_diff = np.nan
    density_ratio_prod = np.nan
    variation_coeff = np.nan

    if estimate_hrf and not constrained and not normalizing:
        hrf_norm = np.linalg.norm(hrf_mean)
        hrf_mean /= hrf_norm
        hrf_covar /= hrf_norm ** 2
        sigma_h /= hrf_norm ** 2
        nrls_mean *= hrf_norm
        nrls_covar *= hrf_norm ** 2
        nrls_class_mean *= hrf_norm
        nrls_class_var *= hrf_norm ** 2
        density_ratio = -(hrf_mean.T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean)/2.)
        density_ratio_cano = -((hrf_mean-m_h).T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean-m_h)/2.)
        density_ratio_diff = density_ratio_cano - density_ratio
        density_ratio_prod = density_ratio_cano * density_ratio
        variation_coeff = np.sqrt((hrf_mean.T.dot(hrf_covar).dot(hrf_mean))/(hrf_mean.T.dot(hrf_mean))**2)
    nrls_class_var = np.sqrt(np.sqrt(nrls_class_var))

    ppm_a_nrl = np.zeros((nb_voxels, nb_conditions))
    ppm_g_nrl = np.zeros((nb_voxels, nb_conditions))
    for condition in xrange(nb_conditions):
        #  ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0)
        ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 3*nrls_class_var[condition, 0]**.5)
        ppm_g_nrl[:, condition] = cpt_ppm_g_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0.95)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#

    ppm_a_contrasts = np.zeros((nb_voxels, len(contrasts)))
    ppm_g_contrasts = np.zeros((nb_voxels, len(contrasts)))

    if compute_contrasts:
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            nrls_conds = dict([(str(cn), nrls_mean[:, ic])
                               for ic, cn in enumerate(condition_names)])

            for n, cname in enumerate(contrasts):
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(nb_conditions, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, nb_conditions):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, nb_conditions):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(nb_conditions, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, nb_voxels):
                    S_tmp = nrls_covar[:, :, j]
                    CONTRASTVAR[j, n] = AC.dot(S_tmp).dot(AC)
                #------------ variance -------------#
                logger.info('Done contrasts computing.')

                ppm_a_contrasts[:, n] = cpt_ppm_a_norm(contrast, CONTRASTVAR[:, n], 0)
                ppm_g_contrasts[:, n] = cpt_ppm_g_norm(contrast, CONTRASTVAR[:, n], 0.95)

        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    logger.info("Nb iterations to reach criterion: %d", loop)
    logger.info("Computational time = %s min %s s",
                *(str(int(x)) for x in divmod(compute_time[-1], 60)))
    logger.debug('nrls_class_mean: %s', nrls_class_mean)
    logger.debug('nrls_class_var: %s', nrls_class_var)
    logger.debug("sigma_H = %s", str(sigma_h))
    logger.debug("beta = %s", str(beta))

    stimulus_induced_signal = vt.computeFit(hrf_mean, nrls_mean, X, nb_voxels, nb_scans)
    snr = 20 * np.log(
        np.linalg.norm(bold_data) / np.linalg.norm(bold_data - stimulus_induced_signal - drift))
    snr /= np.log(10.)
    logger.info('snr comp = %f', snr)
    # ,FreeEnergyArray
    return (loop, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var,
            nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,
            CONTRAST, CONTRASTVAR, nrls_criteria[2:], hrf_criteria[2:],
            labels_criteria[2:], nrls_hrf_criteria[2:], compute_time[2:],
            compute_time_mean, nrls_covar, stimulus_induced_signal, density_ratio,
            density_ratio_cano, density_ratio_diff, density_ratio_prod, ppm_a_nrl,
            ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff,
            free_energy, free_energy_crit, beta_list)

def jde_vem_bold_division(graph, bold_data, onsets, hrf_duration, nb_classes,
                          tr, beta, dt, estimate_sigma_h=True, sigma_h=0.05,
                          it_max=-1, it_min=0, estimate_beta=True, contrasts=None,
                          compute_contrasts=False, hrf_hyperprior=0, estimate_hrf=True,
                          constrained=False, seed=6537546, labels_proba_filename=None):
    """This is the main function that computes the VEM analysis on BOLD data.
    This function uses optimized python functions.

    Parameters
    ----------
    graph : # TODO
        # TODO
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    onsets : dict
        dictionnary of onsets
    hrf_duration : float
        hrf total time duration (in s)
    nb_classes : # TODO
        # TODO
    tr : float
        time of repetition
    beta : # TODO
        # TODO
    dt : float
        hrf temporal precision
    estimate_sigma_h : bool, optional
        toggle estimation of sigma H
    sigma_h : float, optional
        initial or fixed value of sigma H
    it_max : int, optional
        maximal computed iteration number
    it_min : int, optional
        minimal computed iteration number
    estimate_beta : bool, optional
        toggle the estimation of beta
    contrasts : OrderedDict, optional
        dict of contrasts to compute
    compute_contrasts : bool, optional
        if True, compute the contrasts defined in contrasts
    hrf_hyperprior : float (# TODO: check)
        # TODO
    estimate_hrf : bool, optional
        if True, estimate the HRF for each parcel, if False use the canonical
        HRF
    constrained : bool, optional
        if True, constrains the l2 norm of the HRF to 1
    seed : int, optional
        seed used by numpy to initialize random generator number

    Returns
    -------
    loop : int
        number of iterations before convergence
    nrls_mean : ndarray, shape (nb_voxels, nb_conditions)
        Neural response level mean value
    hrf_mean : ndarray, shape (hrf_len,)
        Hemodynamic response function mean value
    hrf_covar : ndarray, shape (hrf_len, hrf_len)
        Covariance matrix of the HRF
    labels_proba : ndarray, shape (nb_conditions, nb_classes, nb_voxels)
        # TODO
    noise_var : ndarray, shape (nb_voxels,)
        # TODO
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
        # TODO
    beta : ndarray, shape (nb_conditions,)
        # TODO
    drift_coeffs : ndarray, shape (# TODO)
        # TODO
    drift : ndarray, shape (# TODO)
        # TODO
    CONTRAST : ndarray, shape (nb_voxels, len(contrasts))
        Contrasts computed from NRLs
    CONTRASTVAR : ndarray, shape (nb_voxels, len(contrasts))
        Variance of the contrasts
    nrls_criteria : list
        NRL criteria (# TODO: explain)
    hrf_criteria : list
        HRF criteria (# TODO: explain)
    labels_criteria : list
        Z criteria (# TODO: explain)
    nrls_hrf_criteria : list
        NRLs HRF product criteria
    compute_time : list
        computation time of each iteration
    compute_time_mean : float
        computation mean time over iterations
    nrls_covar : ndarray, shape (nb_conditions, nb_conditions, nb_voxels)
        # TODO
    stimulus_induced_signal : ndarray, shape (nb_scans, nb_voxels)
        # TODO
    density_ratio : float
        # TODO
    density_ratio_cano : float
        # TODO
    density_ratio_diff : float
        # TODO
    density_ratio_prod : float
        # TODO
    ppm_a_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_nrl : ndarray, shape (nb_voxels,)
        # TODO
    ppm_a_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    ppm_g_contrasts : ndarray, shape (nb_voxels,)
        # TODO
    variation_coeff : float
        coefficient of variation of the HRF

    Notes
    -----
        See `A novel definition of the multivariate coefficient of variation <http://onlinelibrary.wiley.com/doi/10.1002/bimj.201000030/abstract>`_
        article for more information about the coefficient of variation.
    """

    logger.info("Fast EM with C extension started.")

    if not contrasts:
        contrasts = OrderedDict()

    np.random.seed(seed)

    nb_2_norm = 1
    normalizing = False
    regularizing = False
    estimate_free_energy = False

    if it_max < 0:
        it_max = 100
    gamma = 7.5
    gradient_step = 0.003
    it_max_grad = 200
    thresh = 1e-5
    thresh_free_energy = 1e-5

    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_conditions = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    condition_names = []

    max_neighbours = max([len(nl) for nl in graph])
    neighbours_indexes = [np.concatenate((arr, np.zeros(max_neighbours-len(arr))-1))
                          for arr in graph]
    neighbours_indexes = np.asarray(neighbours_indexes, dtype=int)

    X = OrderedDict()
    for condition, onset in onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(nb_scans, tr, hrf_len, dt, onset)
        condition_names.append(condition)
    occurence_matrix = np.zeros((nb_conditions, nb_scans, hrf_len), dtype=np.int32)
    for nc, condition in enumerate(onsets.iterkeys()):
        occurence_matrix[nc, :, :] = X[condition]

    order = 2
    if regularizing:
        regularization = np.ones(hrf_len)
        regularization[hrf_len//3:hrf_len//2] = 2
        regularization[hrf_len//2:2*hrf_len//3] = 5
        regularization[2*hrf_len//3:3*hrf_len//4] = 7
        regularization[3*hrf_len//4:] = 10
        # regularization[hrf_len//2:] = 10
    else:
        regularization = None
    D2 = vt.buildFiniteDiffMatrix(order, hrf_len, regularization)
    hrf_regu_prior_inv = D2.dot(D2) / pow(dt, 2 * order)

    noise_struct = np.identity(nb_scans)

    hrf_criteria = [1.]
    labels_criteria = [1.]
    nrls_criteria = [1.]
    nrls_hrf_criteria = [1.]
    free_energy_crit = [1.]
    compute_time = []

    #  nrls_hrf_prev = np.zeros((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)
    # initialized to ones to avoid numerical isssues
    nrls_hrf_prev = np.ones((nb_voxels, nb_conditions, hrf_len), dtype=np.float64)

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    #  Q_barnCond = np.zeros((nb_conditions, nb_conditions, hrf_len, hrf_len), dtype=np.float64)
    #  XGamma = np.zeros((nb_conditions, hrf_len, nb_scans), dtype=np.float64)
    #  m1 = 0
    #  for k1 in X:  # Loop over the nb_conditions conditions
        #  m2 = 0
        #  for k2 in X:
            #  Q_barnCond[m1, m2, :, :] = np.dot(
                #  np.dot(X[k1].transpose(), noise_struct), X[k2])
            #  m2 += 1
        #  XGamma[m1, :, :] = np.dot(X[k1].transpose(), noise_struct)
        #  m1 += 1
    #  XGamma = np.tensordot(occurence_matrix.T, noise_struct, axes=(1, 0)).transpose(1, 0, 2)
    #  Q_barnCond = np.tensordot(XGamma, occurence_matrix, axes=(2, 1)).transpose(0, 2, 1, 3)

    noise_var = np.ones(nb_voxels)

    labels_proba = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    labels_proba_prev = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    if not labels_proba_filename:
        logger.info("Labels are initialized by setting active probabilities to ones ...")
        labels_proba[:, 1, :] = 1
    else:
        if isinstance(labels_proba_filename, dict):
            logger.info("Labels are initialized by using true labels file")
            for condition_name, label_proba_filename in labels_proba_filename.iteritems():
                true_label_proba = read_volume(labels_proba_filename[condition_name])
                labels_proba[condition_names.index(condition_name), 1, :] = true_label_proba[0].flatten()
        elif isinstance(label_proba_filename, str):
            logger.info("Labels are initialized by using true labels file")
            true_labels_proba = read_volume(labels_proba_filename)
            for condition_nb, condition_name in enumerate(condition_names):
                labels_proba[condition_nb, 1, :] = true_labels_proba.sub_cuboid(condition=condition_name).data
        else:
            logger.error('Unknown labels file format.\n'
                         'Labels are initialized by setting active probabilities to ones...')
    #  Z_tilde = labels_proba.copy()

    # TODO: replace every variable/variable_prev by collections.deque
    m_h = getCanoHRF(hrf_duration, dt)[1][:hrf_len]
    hrf_mean = np.array(m_h).astype(np.float64)
    hrf_mean_prev = np.array(m_h)
    if estimate_hrf:
        hrf_covar = np.ones((hrf_len, hrf_len), dtype=np.float64)
    else:
        hrf_covar = np.zeros((hrf_len, hrf_len), dtype=np.float64)

    beta = beta * np.ones((nb_conditions), dtype=np.float64)
    drift_basis = vt.PolyMat(nb_scans, 4, tr)
    drift_coeffs = vt.poly_fit(bold_data, drift_basis)
    drift = drift_basis.dot(drift_coeffs)
    y_tilde = bold_data - drift

    nrls_class_var = np.ones((nb_conditions, nb_classes), dtype=np.float64)
    nrls_class_var[:, 0] = 0.5
    nrls_class_var[:, 1] = 0.6
    #  nrls_class_mean = np.zeros((nb_conditions, nb_classes), dtype=np.float64)
    #  for k in xrange(1, nb_classes):
        #  nrls_class_mean[:, k] = 1  # init_mean
    nrls_covar = 0.01 * (np.identity(nb_conditions)[:, :, np.newaxis]
                         + np.zeros((1, 1, nb_voxels)))
    nrls_class_mean = np.ones((nb_conditions, nb_classes))
    #  nrls_mean = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    #  nrls_mean_prev = np.zeros((nb_voxels, nb_conditions), dtype=np.float64)
    #  for j in xrange(0, nb_voxels):
        #  for m in xrange(0, nb_conditions):
            #  for k in xrange(0, nb_classes):
                #  nrls_mean[j, m] += np.random.normal(
                    #  nrls_class_mean[m, k], np.sqrt(nrls_class_var[m, k])) * labels_proba[m, k, j]
    nrls_mean = (np.random.normal(
        nrls_class_mean, nrls_class_var)[:, :, np.newaxis] * labels_proba).sum(axis=1).T
    nrls_mean = np.zeros((nb_voxels, nb_conditions))
    #  nrls_mean_prev = nrls_mean.copy()
    # initialized to ones to avoid numerical issues
    nrls_mean_prev = np.ones((nb_voxels, nb_conditions))

    start_time = time.time()
    free_energy = [0.]
    loop = 0
    while (loop < it_min + 1) or (free_energy_crit[-1] > thresh_free_energy and
                                  nrls_hrf_criteria[-1] > thresh and loop < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(loop+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        #  nrls_expectation = UtilsC.expectation_A(labels_proba, nrls_class_mean, nrls_class_var, drift,
                                                #  noise_var, noise_struct, hrf_covar,
                                                #  bold_data, y_tilde, nrls_mean, hrf_mean,
                                                #  nrls_covar, occurence_matrix.astype(np.int32),
                                                #  nb_voxels, hrf_len, nb_conditions,
                                                #  nb_scans, nb_classes)
        nrls_mean, nrls_covar = vt.nrls_expectation(
            hrf_mean, nrls_mean, occurence_matrix, noise_struct, labels_proba,
            nrls_class_mean, nrls_class_var, nb_conditions, y_tilde, nrls_covar,
            hrf_covar, noise_var)
        logger.debug("After: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        if estimate_hrf:
            logger.info("Expectation H step...")
            logger.debug("Before: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            #  hrf_expectation = UtilsC.expectation_H(XGamma, Q_barnCond,
                                                   #  noise_var, noise_struct, hrf_regu_prior_inv,
                                                   #  hrf_covar, bold_data, y_tilde,
                                                   #  nrls_mean, hrf_mean, nrls_covar,
                                                   #  occurence_matrix.astype(np.int32),
                                                   #  nb_voxels, hrf_len,
                                                   #  nb_conditions, nb_scans, scale,
                                                   #  sigma_h)
            hrf_mean, hrf_covar = vt.hrf_expectation(
                nrls_covar, nrls_mean, occurence_matrix, noise_struct,
                hrf_regu_prior_inv, sigma_h, nb_voxels, y_tilde, noise_var)
            if constrained:
                hrf_mean = vt.norm1_constraint(hrf_mean, hrf_covar)
                hrf_covar[:] = 0
            else:
                hrf_mean[0] = 0
                hrf_mean[-1] = 0
            logger.debug("After: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            # Normalizing H at each nb_2_norm iterations:
            if not constrained and normalizing:
                # Normalizing is done before sigma_h, nrls_class_mean and nrls_class_var estimation
                # we should not include them in the normalisation step
                if (loop + 1) % nb_2_norm == 0:
                    hrf_norm = np.linalg.norm(hrf_mean)
                    hrf_mean /= hrf_norm
                    hrf_covar /= hrf_norm ** 2
                    nrls_mean *= hrf_norm
                    nrls_covar *= hrf_norm ** 2

        DIFF = np.reshape(nrls_mean - nrls_mean_prev, (nb_conditions * nb_voxels))
        diff = np.linalg.norm((nrls_mean - nrls_mean_prev).flatten())
        nrls_mean_prev_norm = np.linalg.norm(nrls_mean_prev.flatten())
        nrls_criteria.append(diff / nrls_mean_prev_norm**2
                             if nrls_mean_prev_norm else None)
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  if np.linalg.norm(np.reshape(nrls_mean_prev, (nb_conditions * nb_voxels))) > 0:
            #  nrls_criteria.append((
                #  np.linalg.norm(DIFF) /
                #  np.linalg.norm(np.reshape(nrls_mean_prev, (nb_conditions * nb_voxels)))) ** 2)
        #  else:
            #  # TODO: norm shouldn't be 0
            #  nrls_criteria.append(None)
        nrls_mean_prev[:, :] = nrls_mean[:, :]

        hrf_criteria.append((np.linalg.norm(hrf_mean - hrf_mean_prev) / np.linalg.norm(hrf_mean_prev)) ** 2)
        hrf_mean_prev[:] = hrf_mean[:]

        #  for d in xrange(0, hrf_len):
            #  nrls_hrf[:, :, d] = nrls_mean[:, :] * hrf_mean[d]
        nrls_hrf = nrls_mean[:, :, np.newaxis] * hrf_mean

        #  DIFF = np.reshape(nrls_hrf - nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len))
        diff = np.linalg.norm((nrls_hrf - nrls_hrf_prev).flatten())
        nrls_hrf_prev_norm = np.linalg.norm(nrls_hrf_prev.flatten())
        nrls_hrf_criteria.append(diff / nrls_hrf_prev_norm**2
                                 if nrls_hrf_prev_norm else None)
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  nrls_hrf_criteria.append((np.linalg.norm(DIFF) /
                   #  (np.linalg.norm(np.reshape(nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len)))
                    #  + eps)) ** 2)
        #  if np.linalg.norm(np.reshape(nrls_hrf_prev, (nb_conditions * nb_voxels * hrf_len))) == 0:
            #  # TODO: norm shouldn't be 0
            #  logger.warning("nrls_hrf norm should not be zero")
            #  nrls_hrf_criteria.append(None)

        #  nrls_hrf_prev[:, :, :] = nrls_hrf[:, :, :]
        nrls_hrf_prev = nrls_hrf.copy()

        logger.info("Expectation Z step...")
        logger.debug("Before: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)
        #  labels_expectation = UtilsC.expectation_Z(
            #  nrls_covar, nrls_mean, nrls_class_var, beta, labels_proba, nrls_class_mean, labels_proba,
            #  neighbours_indexes.astype(np.int32), nb_conditions, nb_voxels,
            #  nb_classes, max_neighbours)
        labels_proba = vt.labels_expectation_division(
            nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean, beta,
            labels_proba, neighbours_indexes, nb_conditions, nb_classes)
        logger.debug("After: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)

        diff = np.linalg.norm((labels_proba - labels_proba_prev).flatten())
        labels_proba_prev_norm = np.linalg.norm(labels_proba_prev.flatten())
        labels_criteria.append(diff / labels_proba_prev_norm**2
                               if labels_proba_prev_norm else None)
        #  DIFF = np.reshape(labels_proba - labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        #  # To avoid numerical problems
        #  DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        #  if np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) > 0:
            #  labels_criteria.append((np.linalg.norm(DIFF) /
                      #  (np.linalg.norm(np.reshape(labels_proba_prev, (nb_conditions * nb_classes * nb_voxels))) + eps)) ** 2)
        #  else:
            #  # TODO: norm shouldn't be 0
            #  labels_criteria.append(None)
        #  labels_proba_prev[:, :, :] = labels_proba[:, :, :]

        if estimate_hrf and estimate_sigma_h:
            logger.info("Maximization sigma_H step...")
            logger.debug("Before: sigma_h = %s", sigma_h)
            if hrf_hyperprior > 0:
                sigma_h = vt.maximization_sigmaH_prior(hrf_len, hrf_covar,
                                                       hrf_regu_prior_inv,
                                                       hrf_mean, hrf_hyperprior)
            else:
                sigma_h = vt.maximization_sigmaH(hrf_len, hrf_covar,
                                                 hrf_regu_prior_inv, hrf_mean)
            logger.debug("After: sigma_h = %s", sigma_h)

        logger.info("Maximization (mu,sigma) step...")
        logger.debug("Before: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)
        nrls_class_mean, nrls_class_var = vt.maximization_class_proba(
            nrls_class_mean, nrls_class_var, labels_proba, nrls_mean,
            nb_classes, nb_conditions, nrls_covar)
        logger.debug("After: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)

        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        UtilsC.maximization_L(bold_data, nrls_mean, hrf_mean, drift_coeffs, drift_basis,
                              occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                              nb_conditions, drift_coeffs.shape[0], nb_scans)
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = drift_basis.dot(drift_coeffs)
        y_tilde = bold_data - drift
        if estimate_beta:
            logger.info("Maximization beta step...")
            for m in xrange(0, nb_conditions):
                beta[m] = UtilsC.maximization_beta(
                    beta[m], labels_proba[m, :, :].astype(np.float64),
                    labels_proba[m, :, :].astype(np.float64), nb_voxels, nb_classes,
                    neighbours_indexes.astype(np.int32), gamma, max_neighbours,
                    it_max_grad, gradient_step)
                #  beta[m] = UtilsC.maximization_beta_CB(
                    #  beta[m], labels_proba[m, :, :].astype(np.float64),
                    #  nb_voxels, nb_classes, neighbours_indexes.astype(np.int32),
                    #  gamma, max_neighbours, it_max_grad, gradient_step)
            logger.debug("beta = %s", str(beta))

        logger.info("Maximization sigma noise step...")
        UtilsC.maximization_sigma_noise(noise_struct, drift, noise_var, hrf_covar,
                                        bold_data, nrls_mean, hrf_mean, nrls_covar,
                                        occurence_matrix.astype(np.int32), nb_voxels, hrf_len,
                                        nb_conditions, nb_scans)

        #### Computing Free Energy ####
        if estimate_hrf and estimate_free_energy:
            free_energy_prev = free_energy[-1]
            nrls_expectation = None
            hrf_expectation = None
            labels_expectation = None
            free_energy.append(vt.free_energy_computation(nrls_covar, hrf_covar, labels_proba,
                                                          nb_voxels, hrf_len,
                                                          nb_conditions,
                                                          nrls_expectation,
                                                          hrf_expectation,
                                                          labels_expectation))
            free_energy_crit.append((free_energy_prev - free_energy) /
                                    free_energy_prev)

        logger.info("Convergence criteria: %f (Threshold = %f)",
                    free_energy_crit[-1], thresh_free_energy)
        loop += 1
        compute_time.append(time.time() - start_time)

    compute_time_mean = compute_time[-1] / loop

    density_ratio = np.nan
    density_ratio_cano = np.nan
    density_ratio_diff = np.nan
    density_ratio_prod = np.nan
    variation_coeff = np.nan

    if estimate_hrf and not constrained and not normalizing:
        hrf_norm = np.linalg.norm(hrf_mean)
        hrf_mean /= hrf_norm
        hrf_covar /= hrf_norm ** 2
        sigma_h /= hrf_norm ** 2
        nrls_mean *= hrf_norm
        nrls_covar *= hrf_norm ** 2
        nrls_class_mean *= hrf_norm
        nrls_class_var *= hrf_norm ** 2
        density_ratio = -(hrf_mean.T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean)/2.)
        density_ratio_cano = -((hrf_mean-m_h).T.dot(np.linalg.inv(hrf_covar)).dot(hrf_mean-m_h)/2.)
        density_ratio_diff = density_ratio_cano - density_ratio
        density_ratio_prod = density_ratio_cano * density_ratio
        variation_coeff = np.sqrt((hrf_mean.T.dot(hrf_covar).dot(hrf_mean))/(hrf_mean.T.dot(hrf_mean))**2)
    nrls_class_var = np.sqrt(np.sqrt(nrls_class_var))

    ppm_a_nrl = np.zeros((nb_voxels, nb_conditions))
    ppm_g_nrl = np.zeros((nb_voxels, nb_conditions))
    for condition in xrange(nb_conditions):
        #  ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0)
        ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 3*nrls_class_var[condition, 0]**.5)
        ppm_g_nrl[:, condition] = cpt_ppm_g_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0.95)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#

    ppm_a_contrasts = np.zeros((nb_voxels, len(contrasts)))
    ppm_g_contrasts = np.zeros((nb_voxels, len(contrasts)))

    if compute_contrasts:
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            nrls_conds = dict([(str(cn), nrls_mean[:, ic])
                               for ic, cn in enumerate(condition_names)])

            for n, cname in enumerate(contrasts):
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(nb_conditions, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, nb_conditions):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, nb_conditions):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(nb_conditions, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, nb_voxels):
                    S_tmp = nrls_covar[:, :, j]
                    CONTRASTVAR[j, n] = AC.dot(S_tmp).dot(AC)
                #------------ variance -------------#
                logger.info('Done contrasts computing.')

                ppm_a_contrasts[:, n] = cpt_ppm_a_norm(contrast, CONTRASTVAR[:, n], 0)
                ppm_g_contrasts[:, n] = cpt_ppm_g_norm(contrast, CONTRASTVAR[:, n], 0.95)

        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    logger.info("Nb iterations to reach criterion: %d", loop)
    logger.info("Computational time = %s min %s s",
                *(str(int(x)) for x in divmod(compute_time[-1], 60)))
    logger.debug('nrls_class_mean: %s', nrls_class_mean)
    logger.debug('nrls_class_var: %s', nrls_class_var)
    logger.debug("sigma_H = %s", str(sigma_h))
    logger.debug("beta = %s", str(beta))

    stimulus_induced_signal = vt.computeFit(hrf_mean, nrls_mean, X, nb_voxels, nb_scans)
    snr = 20 * np.log(
        np.linalg.norm(bold_data) / np.linalg.norm(bold_data - stimulus_induced_signal - drift))
    snr /= np.log(10.)
    logger.info('snr comp = %f', snr)
    # ,FreeEnergyArray
    return (loop, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var,
            nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,
            CONTRAST, CONTRASTVAR, nrls_criteria[2:], hrf_criteria[2:],
            labels_criteria[2:], nrls_hrf_criteria[2:], compute_time[2:],
            compute_time_mean, nrls_covar, stimulus_induced_signal, density_ratio,
            density_ratio_cano, density_ratio_diff, density_ratio_prod, ppm_a_nrl,
            ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff)
