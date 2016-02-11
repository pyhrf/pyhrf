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

import pyhrf.vbjde.vem_tools as vt

from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.tools._io import read_volume
from pyhrf.stats.misc import cpt_ppm_a_norm, cpt_ppm_g_norm

logger = logging.getLogger(__name__)
eps = np.spacing(1)


def jde_vem_bold(graph, bold_data, onsets, durations, hrf_duration, nb_classes,
                 tr, beta, dt, estimate_sigma_h=True, sigma_h=0.05,
                 it_max=-1, it_min=0, estimate_beta=True, contrasts=None,
                 compute_contrasts=False, hrf_hyperprior=0, estimate_hrf=True,
                 constrained=False, seed=6537546):
    """This is the main function that computes the VEM analysis on BOLD data.
    This function uses optimized python functions.

    Parameters
    ----------
    graph : ndarray of lists
        represents the neighbours indexes of each voxels index
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        raw data
    onsets : dict
        dictionnary of onsets
    durations : # TODO
        # TODO
    hrf_duration : float
        hrf total time duration (in s)
    nb_classes : int
        the number of classes to classify the nrls. This parameter is provided
        for development purposes as most of the algorithm implies two classes
    tr : float
        time of repetition
    beta : float
        the initial value of beta
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
    hrf_hyperprior : float
        # TODO
    estimate_hrf : bool, optional
        if True, estimate the HRF for each parcel, if False use the canonical HRF
    constrained : bool, optional
        if True, add a constrains the l2 norm of the HRF to 1
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
        probability of voxels being in one class
    noise_var : ndarray, shape (nb_voxels,)
        estimated noise variance
    nrls_class_mean : ndarray, shape (nb_conditions, nb_classes)
        estimated mean value of the gaussians of the classes
    nrls_class_var : ndarray, shape (nb_conditions, nb_classes)
        estimated variance of the gaussians of the classes
    beta : ndarray, shape (nb_conditions,)
        estimated beta
    drift_coeffs : ndarray, shape (# TODO)
        estimated coefficient of the drifts
    drift : ndarray, shape (# TODO)
        estimated drifts
    CONTRAST : ndarray, shape (nb_voxels, len(contrasts))
        Contrasts computed from NRLs
    CONTRASTVAR : ndarray, shape (nb_voxels, len(contrasts))
        Variance of the contrasts
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
        See `A novel definition of the multivariate coefficient of variation
        <http://onlinelibrary.wiley.com/doi/10.1002/bimj.201000030/abstract>`_
        article for more information about the coefficient of variation.
    """

    logger.info("VEM started.")

    if not contrasts:
        contrasts = OrderedDict()

    np.random.seed(seed)

    nb_2_norm = 1
    normalizing = False
    regularizing = False

    if it_max <= 0:
        it_max = 100
    gamma = 7.5
    thresh_free_energy = 1e-5

    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_conditions = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    X, occurence_matrix, condition_names = vt.create_conditions(
        onsets, durations, nb_conditions, nb_scans, hrf_len, tr, dt
    )
    neighbours_indexes = vt.create_neighbours(graph)

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
    d2 = vt.buildFiniteDiffMatrix(order, hrf_len, regularization)
    hrf_regu_prior_inv = d2.dot(d2) / pow(dt, 2 * order)

    noise_struct = np.identity(nb_scans)

    free_energy = [1.]
    free_energy_crit = [1.]
    compute_time = []

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)

    noise_var = np.ones(nb_voxels)

    labels_proba = np.zeros((nb_conditions, nb_classes, nb_voxels), dtype=np.float64)
    logger.info("Labels are initialized by setting everything to {}".format(1./nb_classes))
    labels_proba[:, :, :] = 1./nb_classes

    m_h = getCanoHRF(hrf_duration, dt)[1][:hrf_len]
    hrf_mean = np.array(m_h).astype(np.float64)
    if estimate_hrf:
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
    nrls_class_mean = 2 * np.ones((nb_conditions, nb_classes))
    nrls_class_mean[:, 0] = 0
    nrls_class_var = 0.3 * np.ones((nb_conditions, nb_classes), dtype=np.float64)

    nrls_mean = (np.random.normal(
        nrls_class_mean, nrls_class_var)[:, :, np.newaxis] * labels_proba).sum(axis=1).T
    nrls_covar = (np.identity(nb_conditions)[:, :, np.newaxis] + np.zeros((1, 1, nb_voxels)))

    start_time = time.time()
    loop = 0
    while (loop <= it_min) or (free_energy_crit[-1] > thresh_free_energy and loop < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(loop+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)
        nrls_mean, nrls_covar = vt.nrls_expectation(
            hrf_mean, nrls_mean, occurence_matrix, noise_struct, labels_proba,
            nrls_class_mean, nrls_class_var, nb_conditions, bold_data_drift, nrls_covar,
            hrf_covar, noise_var)
        logger.debug("After: nrls_mean = %s, nrls_covar = %s", nrls_mean, nrls_covar)

        if estimate_hrf:
            logger.info("Expectation H step...")
            logger.debug("Before: hrf_mean = %s, hrf_covar = %s", hrf_mean, hrf_covar)
            hrf_mean, hrf_covar = vt.hrf_expectation(
                nrls_covar, nrls_mean, occurence_matrix, noise_struct,
                hrf_regu_prior_inv, sigma_h, nb_voxels, bold_data_drift, noise_var)
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

        logger.info("Expectation Z step...")
        logger.debug("Before: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)
        labels_proba = vt.labels_expectation(
            nrls_covar, nrls_mean, nrls_class_var, nrls_class_mean, beta,
            labels_proba, neighbours_indexes, nb_conditions, nb_classes,
            nb_voxels, parallel=True)
        logger.debug("After: labels_proba = %s, labels_proba = %s", labels_proba, labels_proba)

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
            labels_proba, nrls_mean, nrls_covar
        )
        logger.debug("After: nrls_class_mean = %s, nrls_class_var = %s",
                     nrls_class_mean, nrls_class_var)

        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        drift_coeffs = vt.maximization_drift_coeffs(
            bold_data, nrls_mean, occurence_matrix, hrf_mean, noise_struct, drift_basis
        )
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = drift_basis.dot(drift_coeffs)
        bold_data_drift = bold_data - drift
        if estimate_beta:
            logger.info("Maximization beta step...")
            for cond_nb in xrange(0, nb_conditions):
                beta[cond_nb], success = vt.beta_maximization(
                    beta[cond_nb]*np.ones((1,)), labels_proba[cond_nb, :, :],
                    neighbours_indexes, gamma
                )
            beta_list.append(beta.copy())
            logger.debug("beta = %s", str(beta))

        logger.info("Maximization sigma noise step...")
        noise_var = vt.maximization_noise_var(
            occurence_matrix, hrf_mean, hrf_covar, nrls_mean, nrls_covar,
            noise_struct, bold_data_drift, nb_scans
        )

        #### Computing Free Energy ####
        free_energy.append(vt.free_energy_computation(
            nrls_mean, nrls_covar, hrf_mean, hrf_covar, hrf_len, labels_proba,
            bold_data_drift, occurence_matrix, noise_var, noise_struct, nb_conditions,
            nb_voxels, nb_scans, nb_classes, nrls_class_mean, nrls_class_var, neighbours_indexes,
            beta, sigma_h, np.linalg.inv(hrf_regu_prior_inv), hrf_regu_prior_inv, gamma, hrf_hyperprior
        ))
        free_energy_crit.append(abs((free_energy[-2] - free_energy[-1]) /
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

    nb_contrasts = len(contrasts)
    if compute_contrasts and nb_contrasts > 0:
        logger.info('Compute contrasts ...')
        (contrasts_mean,
         contrasts_var,
         contrasts_class_mean,
         contrasts_class_var) = vt.contrasts_mean_var_classes(
             contrasts, condition_names, nrls_mean, nrls_covar,
             nrls_class_mean, nrls_class_var, nb_contrasts, nb_classes, nb_voxels
         )
        ppm_a_contrasts, ppm_g_contrasts = vt.ppm_contrasts(
            contrasts_mean, contrasts_var, contrasts_class_mean, contrasts_class_var
        )
        logger.info('Done contrasts computing.')

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
        np.linalg.norm(bold_data.astype(np.float))
        / np.linalg.norm((bold_data - stimulus_induced_signal - drift).astype(np.float))
    )
    snr /= np.log(10.)
    logger.info('snr comp = %f', snr)
    # ,FreeEnergyArray
    return (loop, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var,
            nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,
            contrasts_mean, contrasts_var, compute_time[2:], compute_time_mean,
            nrls_covar, stimulus_induced_signal, density_ratio,
            density_ratio_cano, density_ratio_diff, density_ratio_prod, ppm_a_nrl,
            ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff,
            free_energy[1:], free_energy_crit[1:], beta_list[1:])

