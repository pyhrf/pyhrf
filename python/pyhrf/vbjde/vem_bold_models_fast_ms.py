# -*- coding: utf-8 -*-
"""VEM BOLD Constrained

File that contains function for BOLD data analysis with positivity
and l2-norm=1 constraints.

It imports functions from vem_tools.py in pyhrf/vbjde
"""

import time
import copy
import logging
import os

import os.path as op
import numpy as np

import pyhrf
import pyhrf.vbjde.vem_tools as vt

from pyhrf.boldsynth.hrf import getCanoHRF, genGaussianSmoothHRF, \
                                genGaussianSmoothHRF_cust
from pyhrf.sandbox.physio_params import PHY_PARAMS_KHALIDOV11, \
                                        linear_rf_operator,\
                                        create_physio_brf, \
                                        create_physio_prf
from pyhrf.stats.misc import cpt_ppm_a_norm, cpt_ppm_g_norm

"""
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
"""


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eps = 1e-6

#@profile
def Main_vbjde_physio(graph, Y, Onsets, durations, Thrf, K, TR, beta, dt,
                      scale=1, estimateSigmaH=True, estimateSigmaG=True,
                      sigmaH=0.05, sigmaG=0.05, gamma_h=0, gamma_g=0,
                      NitMax=-1, NitMin=1, estimateBeta=True, PLOT=False,
                      contrasts=[], compute_contrasts=False,
                      idx_first_tag=0, simulation=None, sigmaMu=None,
                      estimateH=True, estimateG=True, estimateA=True,
                      estimateC=True, estimateZ=True, estimateNoise=True,
                      estimateMP=True, estimateLA=True, use_hyperprior=False,
                      positivity=False, constraint=False,
                      phy_params=PHY_PARAMS_KHALIDOV11, prior='omega', zc=False):

    logger.info("EM for ASL!")
    np.random.seed(6537540)
    logger.info("data shape: ")
    logger.info(Y.shape)

    Thresh = 1e-4
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    #D, M = np.int(np.ceil(Thrf / dt)), len(Onsets)
    n_sess, N, J = Y.shape[0], Y.shape[1], Y.shape[2]

    Crit_AH, Crit_CG, cTime, rerror, FE = 1, 1, [], [], []
    EP, EPlh, Ent = [],[],[]
    Crit_H, Crit_G, Crit_Z, Crit_A, Crit_C = 1, 1, 1, 1, 1
    cAH, cCG, AH1, CG1 = [], [], [], []
    cA, cC, cH, cG, cZ = [], [], [], [], []
    h_norm, g_norm = [], []
    SUM_q_Z = [[] for m in xrange(M)]
    mua1 = [[] for m in xrange(M)]
    muc1 = [[] for m in xrange(M)]
    sigmaH = sigmaH * J / 100
    print sigmaH
    gamma_h = gamma_h * 100 / J
    print gamma_h

    # Beta data
    MaxItGrad = 200
    gradientStep = 0.005
    gamma = 7.5
    print 'gamma = ', gamma
    print 'voxels = ', J
    maxNeighbours, neighboursIndexes = vt.create_neighbours(graph, J)
    print 'graph.shape = ', graph.shape

    # Conditions
    print 'Onsets: ', Onsets
    print 'durations = ', durations
    print 'creating conditions...'
    X, XX, condition_names = vt.create_conditions_block_ms(Onsets, durations,
                                                    M, N, D, n_sess, TR, dt)

    # Covariance matrix
    #R = vt.covariance_matrix(2, D, dt)
    _, R_inv = genGaussianSmoothHRF(zc, D, dt, 1., 2)
    R = np.linalg.inv(R_inv)
    if zc:
        XX = XX[:, :, :, 1:-1]    # XX shape (S, M, N, D)
        D = D - 2
    AH1, CG1 = np.zeros((J, M, D)), np.zeros((J, M, D))

    print 'HRF length = ', D
    print 'Condition number = ', M
    print 'Number of scans = ', N
    print 'Number of voxels = ', J
    print 'Number of sessions = ', n_sess
    print 'XX.shape = ', XX.shape

    # Noise matrix
    Gamma = np.identity(N)
    # Noise initialization
    sigma_eps = np.ones((n_sess, J))
    # Labels
    logger.info("Labels are initialized by setting active probabilities "
                "to ones ...")
    q_Z = np.ones((M, K, J), dtype=np.float64) / 2.
    #q_Z = np.zeros((M, K, J), dtype=np.float64)
    #q_Z[:, 1, :] = 1
    q_Z1 = copy.deepcopy(q_Z)
    Z_tilde = copy.deepcopy(q_Z)

    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    H = np.array(m_h[:D]).astype(np.float64)
    H /= np.linalg.norm(H)
    Hb = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
    Hb /= np.linalg.norm(Hb)
    if prior=='balloon':
        H = Hb.copy()
    H1 = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)

    # Initialize model parameters
    Beta = beta * np.ones((M), dtype=np.float64)
    n_drift = 4
    P = np.zeros((n_sess, N, n_drift+1), dtype=np.float64)
    L = np.zeros((n_drift+1, J, n_sess), dtype=np.float64)
    for s in xrange(0, n_sess):
        P[s, :, :] = vt.PolyMat(N, n_drift, TR)
        L[:, :, s] = vt.polyFit(Y[s, :, :], TR, n_drift, P[s, :, :])
    print 'P shape = ', P.shape
    print 'L shape = ', L.shape
    WP = P.copy()
    AL = L.copy()
    PL = np.einsum('ijk,kli->ijl', P, L)
    y_tilde = Y - PL

    # Parameters Gaussian mixtures
    mu_Ma = np.append(np.zeros((M, 1)), np.ones((M, 1)), axis=1).astype(np.float64)
    sigma_Ma = np.ones((M, K), dtype=np.float64) * 0.3

    # Params RLs
    m_A = np.zeros((n_sess, J, M), dtype=np.float64)
    for s in xrange(0, n_sess):
        for j in xrange(0, J):
            m_A[s, j, :] = (np.random.normal(mu_Ma, np.sqrt(sigma_Ma)) * q_Z[:, :, j]).sum(axis=1).T
    m_A1 = m_A.copy()
    Sigma_A = np.ones((M, M, J, n_sess)) * np.identity(M)[:, :, np.newaxis, np.newaxis]

    G = np.zeros_like(H)
    m_C = np.zeros_like(m_A)
    Sigma_G = np.zeros_like(Sigma_H)
    Sigma_C = np.zeros_like(Sigma_A)
    mu_Mc = np.zeros_like(mu_Ma)
    sigma_Mc = np.ones_like(sigma_Ma)
    W = np.zeros_like(Gamma)            # (N, N)

    # Precomputations
    print 'W shape is ', W.shape
    WX = W.dot(XX).transpose(1, 2, 0, 3)                                       # shape (S, M, N, D)
    Gamma_X = np.zeros((N, n_sess, M, D), dtype=np.float64)                    # shape (N, S, M, D)
    X_Gamma_X = np.zeros((D, M, n_sess, M, D), dtype=np.float64)               # shape (D, M, S, M, D)
    Gamma_WX = np.zeros((N, n_sess, M, D), dtype=np.float64)                   # shape (N, S, M, D)
    XW_Gamma_WX = np.zeros((D, M, n_sess, M, D), dtype=np.float64)             # shape (D, M, S, M, D)
    Gamma_WP = np.zeros((N, n_sess, n_drift+1), dtype=np.float64)              # shape (N, S, M, D)
    WP_Gamma_WP = np.zeros((n_sess, n_drift+1, n_drift+1), dtype=np.float64)   # shape (D, M, S, M, D)
    for s in xrange(0, n_sess):
        Gamma_X[:, s, :, :] = np.tensordot(Gamma, XX[s, :, :, :], axes=(1, 1))
        X_Gamma_X[:, :, s, :, :] = np.tensordot(XX[s, :, :, :].T, Gamma_X[:, s, :, :], axes=(1, 0))
        Gamma_WX[:, s, :, :] = np.tensordot(Gamma, WX[s, :, :, :], axes=(1, 1))
        XW_Gamma_WX[:, :, s, :, :] = np.tensordot(WX[s, :, :, :].T, Gamma_WX[:, s, :, :], axes=(1, 0))
        Gamma_WP[:, s, :] = Gamma.dot(WP[s, :, :])                             # (N, n_drift)
        WP_Gamma_WP[s, :, :] = WP[s, :, :].T.dot(Gamma_WP[:, s, :])            # (n_drift, n_drift)
    sigma_eps_m = np.maximum(sigma_eps, eps)                                   # (n_sess, J)
    cov_noise = sigma_eps_m[:, :, np.newaxis, np.newaxis]                      # (n_sess, J, 1, 1)


    ###########################################################################
    #############################################             VBJDE

    t1 = time.time()
    ni = 0

    #while ((ni < NitMin + 1) or (((Crit_AH > Thresh) or (Crit_CG > Thresh)) \
    #        and (ni < NitMax))):
    #while ((ni < NitMin + 1) or (((Crit_AH > Thresh)) \
    #        and (ni < NitMax))):
    while ((ni < NitMin + 1) or (((Crit_FE > Thresh * np.ones_like(Crit_FE)).any()) \
            and (ni < NitMax))):

        logger.info("-------- Iteration n° " + str(ni + 1) + " --------")

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            logger.info("Plotting HRF and PRF for current iteration")
            vt.plot_response_functions_it(ni, NitMin, M, H, G)


        # Managing types of prior
        priorH_cov_term = np.zeros_like(R_inv)
        matrix_covH = R_inv.copy()
        if prior=='balloon':
            logger.info("   prior balloon")
            #matrix_covH = np.eye(R_inv.shape[0], R_inv.shape[1])
            priorH_mean_term = np.dot(matrix_covH / sigmaH, Hb)
        else:
            logger.info("   NO prior")
            priorH_mean_term = np.zeros_like(H)
            priorG_mean_term = np.zeros_like(G)


        #####################
        # EXPECTATION
        #####################


        # HRF H
        if estimateH:
            logger.info("E H step ...")
            Ht, Sigma_H = vt.expectation_H_ms(Sigma_A, m_A, m_C, G, XX, W, Gamma,
                                            Gamma_X, X_Gamma_X, J, y_tilde,
                                            cov_noise, matrix_covH, sigmaH,
                                            priorH_mean_term, priorH_cov_term, N, M, D, n_sess)

            if constraint:
                if not np.linalg.norm(Ht)==1:
                    logger.info("   constraint l2-norm = 1")
                    H = vt.constraint_norm1_b(Ht, Sigma_H)
                    #H = Ht / np.linalg.norm(Ht)
                else:
                    logger.info("   l2-norm already 1!!!!!")
                    H = Ht.copy()
                Sigma_H = np.zeros_like(Sigma_H)
            else:
                H = Ht.copy()
                h_norm = np.append(h_norm, np.linalg.norm(H))
                print 'h_norm = ', h_norm

            Crit_H = (np.linalg.norm(H - H1) / np.linalg.norm(H1)) ** 2
            cH += [Crit_H]
            H1[:] = H[:]


        # A
        if estimateA:
            logger.info("E A step ...")
            m_A, Sigma_A = vt.expectation_A_ms(m_A, Sigma_A, H, G, m_C, W, XX,
                                             Gamma, Gamma_X, q_Z,
                                             mu_Ma, sigma_Ma, J, y_tilde,
                                             Sigma_H, sigma_eps_m, N, M, D, n_sess)

            cA += [(np.linalg.norm(m_A - m_A1) / np.linalg.norm(m_A1)) ** 2]
            m_A1[:, :, :] = m_A[:, :, :]

        # Q labels
        if estimateZ:
            logger.info("E Q step ...")
            q_Z, Z_tilde = vt.expectation_Q_ms(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, neighboursIndexes, graph, M, J, K, n_sess)

            if 0:
                import matplotlib.pyplot as plt
                plt.close('all')
                fig = plt.figure(1)
                for m in xrange(M):
                    ax = fig.add_subplot(2, M, m + 1)
                    im = ax.matshow(m_A[:, :, m].mean(0).reshape(20, 20))
                    plt.colorbar(im, ax=ax)
                    ax = fig.add_subplot(2, M, m + 3)
                    im = ax.matshow(q_Z[m, 1, :].reshape(20, 20))
                    plt.colorbar(im, ax=ax)
                fig = plt.figure(2)
                for m in xrange(M):
                    for s in xrange(n_sess):
                        ax = fig.add_subplot(M, n_sess, n_sess * m + s + 1)
                        im = ax.matshow(m_A[s, :, m].reshape(20, 20))
                        plt.colorbar(im, ax=ax)
                plt.show()

            cZ += [(np.linalg.norm(q_Z - q_Z1) / (np.linalg.norm(q_Z1) + eps)) ** 2]
            q_Z1 = q_Z

        if ni > 0:
            free_energyE = 0
            for s in xrange(n_sess):
                free_energyE += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s],
                                             mu_Ma, sigma_Ma, H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                             m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                             AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                             gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :],
                                             W, J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :],
                                             Gamma_WX[:, s, :, :], bold=True, S=n_sess)
            if free_energyE < free_energy:
                logger.info("free energy has decreased after E step from %f to %f", free_energy, free_energyE)


        # crit. AH and CG
        logger.info("crit. AH and CG")
        AH = m_A[:, :, :, np.newaxis] * H[np.newaxis, np.newaxis, :]

        Crit_AH = (np.linalg.norm(AH - AH1) / (np.linalg.norm(AH1) + eps)) ** 2
        cAH += [Crit_AH]
        AH1 = AH.copy()
        logger.info("Crit_AH = " + str(Crit_AH))


        #####################
        # MAXIMIZATION
        #####################

        if prior=='balloon':
            logger.info("   prior balloon")
            AuxH = H - Hb
            AuxG = G - Gb
        else:
            logger.info("   NO prior")
            AuxH = H.copy()
            AuxG = G.copy()

        # Variance HRF: sigmaH
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            sigmaH = vt.maximization_sigma_asl(D, Sigma_H, matrix_covH, AuxH, use_hyperprior, gamma_h)
            logger.info('sigmaH = ' + str(sigmaH))

        if ni > 0:
            free_energyVh = 0
            for s in xrange(n_sess):
                free_energyVh += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], mu_Ma, sigma_Ma,
                                             H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                             m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                             AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                             gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :], W,
                                             J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :], bold=True, S=n_sess)
            if free_energyVh < free_energyE:
                logger.info("free energy has decreased after v_h computation from %f to %f", free_energyE, free_energyVh)


        # (mu,sigma)
        if estimateMP:
            logger.info("M (mu,sigma) a and c step ...")
            mu_Ma, sigma_Ma = vt.maximization_mu_sigma_ms(q_Z, m_A, Sigma_A, M, J, n_sess, K)

        if ni > 0:
            free_energyMP = 0
            for s in xrange(n_sess):
                free_energyMP += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], mu_Ma, sigma_Ma,
                                             H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                             m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                             AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                             gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :], W,
                                             J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :], bold=True, S=n_sess)
            if free_energyMP < free_energyVh:
                logger.info("free energy has decreased after GMM parameters computation from %f to %f", free_energyVh, free_energyMP)


        # Drift L, alpha
        if estimateLA:
            logger.info("M L, alpha step ...")
            for s in xrange(n_sess):
                AL[:, :, s] = vt.maximization_LA_asl(Y[s, :, :], m_A[s, :, :], m_C[s, :, :], XX[s, :, :, :],
                                                     WP[s, :, :], W, WP_Gamma_WP[s, :, :], H, G, Gamma)
            PL = np.einsum('ijk,kli->ijl', WP, AL)
            y_tilde = Y - PL

        if ni > 0:
            free_energyLA = 0
            for s in xrange(n_sess):
                free_energyLA += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], mu_Ma, sigma_Ma,
                                                 H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                                 m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                                 AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                                 gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :], W,
                                                 J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :], bold=True, S=n_sess)
            if free_energyLA < free_energyMP:
                logger.info("free energy has decreased after drifts computation from %f to %f", free_energyMP, free_energyLA)


        # Beta
        if estimateBeta:
            logger.info("M beta step ...")
            """Qtilde = np.concatenate((Z_tilde, np.zeros((M, K, 1), dtype=Z_tilde.dtype)), axis=2)
            Qtilde_sumneighbour = Qtilde[:, :, neighboursIndexes].sum(axis=3)
            Beta = vt.maximization_beta_m2(Beta.copy(), q_Z, Qtilde_sumneighbour,
                                             Qtilde, neighboursIndexes, maxNeighbours,
                                             gamma, MaxItGrad, gradientStep)
            logger.info(Beta)
            """
            logger.info("M beta step ...")
            Qtilde = np.concatenate((Z_tilde, np.zeros((M, K, 1), dtype=Z_tilde.dtype)), axis=2)
            Qtilde_sumneighbour = Qtilde[:, :, neighboursIndexes].sum(axis=3)
            for m in xrange(0, M):
                Beta[m] = vt.maximization_beta_m2_scipy_asl(Beta[m].copy(), q_Z[m, :, :], Qtilde_sumneighbour[m, :, :],
                                                   Qtilde[m, :, :], neighboursIndexes, maxNeighbours,
                                                   gamma, MaxItGrad, gradientStep)
            logger.info(Beta)
        if ni > 0:
            free_energyB = 0
            for s in xrange(n_sess):
                free_energyB += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], mu_Ma, sigma_Ma,
                                             H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                             m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                             AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                             gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :], W,
                                             J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :], bold=True, S=n_sess)
            if free_energyB < free_energyLA:
                logger.info("free energy has decreased after Beta computation from %f to %f", \
                                free_energyLA, free_energyB)
        if 0 and ni < 5:
            plt.close('all')
            for m in xrange(0, M):
                range_b = np.arange(-10., 20., 0.1)
                beta_plotting = np.zeros_like(range_b)
                grad_plotting = np.zeros_like(range_b)
                for ib, b in enumerate(range_b):
                    beta_plotting[ib] = vt.fun(b, q_Z[m, :, :], Qtilde_sumneighbour[m, :, :],
                                                          neighboursIndexes, gamma)
                    grad_plotting[ib] = vt.grad_fun(b, q_Z[m, :, :], Qtilde_sumneighbour[m, :, :],
                                                     neighboursIndexes, gamma)
                #print beta_plotting
                plt.figure(1)
                plt.hold('on')
                plt.plot(range_b, beta_plotting)
                plt.figure(2)
                plt.hold('on')
                plt.plot(range_b, grad_plotting)
            plt.show()


        # Sigma noise
        if estimateNoise:
            logger.info("M sigma noise step ...")
            for s in xrange(n_sess):
                sigma_eps[s, :] = vt.maximization_sigma_noise_asl(XX[s, :, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], H, m_C[s, :, :], Sigma_C[:, :, :, s], \
                                                    G, Sigma_H, Sigma_G, W, y_tilde[s, :, :], Gamma, \
                                                    Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :], N)

        if PLOT:
            for m in xrange(M):
                SUM_q_Z[m] += [q_Z[m, 1, :].sum()]
                mua1[m] += [mu_Ma[m, 1]]

        free_energy = 0
        for s in xrange(n_sess):
            if s==n_sess-1:
                plotFE = True
            else:
                plotFE = False
            free_energy += vt.Compute_FreeEnergy(y_tilde[s, :, :], m_A[s, :, :], Sigma_A[:, :, :, s], mu_Ma, sigma_Ma,
                                             H, Sigma_H, AuxH, R, R_inv, sigmaH, sigmaG,
                                             m_C[s, :, :], Sigma_C[:, :, :, s], mu_Mc, sigma_Mc, G, Sigma_G,
                                             AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                                             gamma, gamma_h, gamma_g, sigma_eps[s, :], XX[s, :, :, :], W,
                                             J, D, M, N, K, use_hyperprior, Gamma_X[:, s, :, :], Gamma_WX[:, s, :, :],
                                             plot=plotFE, bold=True, S=n_sess)
        if ni > 0:
            if free_energy < free_energyB:
                logger.info("free energy has decreased after Noise computation from %f to %f", free_energyB, free_energy)

        if ni > 0:
            if free_energy < FE[-1]:
                logger.info("WARNING! free energy has decreased in this iteration from %f to %f", FE[-1], free_energy)

        FE += [free_energy]

        if ni > 5:
            #Crit_FE = np.abs((FE[-1] - FE[-2]) / FE[-2])
            FE0 = np.array(FE)
            Crit_FE = np.abs((FE0[-5:] - FE0[-6:-1]) / FE0[-6:-1])
            print Crit_FE
            print (Crit_FE > Thresh * np.ones_like(Crit_FE)).any()
        else:
            Crit_FE = 100

        ni += 1
        cTime += [time.time() - t1]

        logger.info("Computing reconstruction error")
        StimulusInducedSignal = vt.computeFit_asl(H, m_A[s, :, :], G, m_C[s, :, :], W, XX[s, :, :, :])
        rerror = np.append(rerror, \
                           np.mean(((Y[s, :, :] - StimulusInducedSignal) ** 2).sum(axis=0)) \
                           / np.mean((Y[s, :, :] ** 2).sum(axis=0)))

    CompTime = time.time() - t1


    # Normalize if not done already
    if not constraint: # or not normg:
        logger.info("l2-norm of H and G to 1 if not constraint")
        Hnorm = np.linalg.norm(H)
        H /= Hnorm
        Sigma_H /= Hnorm**2
        sigmaH /= Hnorm**2
        m_A *= Hnorm
        Sigma_A *= Hnorm**2
        mu_Ma *= Hnorm
        sigma_Ma *= Hnorm**2


    if zc:
        H = np.concatenate(([0], H, [0]))

    
    ppm_a_nrl = np.zeros((J, M))
    ppm_g_nrl = np.zeros((J, M))
    nrls_mean = m_A.mean(0)
    nrls_covar = Sigma_A.mean(3)
    A_std = np.zeros_like(nrls_mean)
    for condition in xrange(M):
        #  ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0)
        ppm_a_nrl[:, condition] = cpt_ppm_a_norm(nrls_mean[ :, condition], nrls_covar[condition, condition, :], sigma_Ma[condition, 0]**.5)
        ppm_g_nrl[:, condition] = cpt_ppm_g_norm(nrls_mean[:, condition], nrls_covar[condition, condition, :], 0.95)
        A_std[:, condition] = nrls_mean[:, condition] / np.sqrt(nrls_covar[condition, condition, :])

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#

    #if not contrasts:
    #    contrasts = OrderedDict()
    nb_contrasts = len(contrasts)
    if compute_contrasts and nb_contrasts > 0:
        logger.info('Compute contrasts ...')
        (contrasts_mean,
         contrasts_var,
         contrasts_class_mean,
         contrasts_class_var) = vt.contrasts_mean_var_classes(
             contrasts, condition_names, m_A, Sigma_A,
             mu_Ma, sigma_Ma, nb_contrasts, K, J
         )
        ppm_a_contrasts, ppm_g_contrasts = vt.ppm_contrasts(
            contrasts_mean, contrasts_var, contrasts_class_mean, contrasts_class_var
        )
        logger.info('Done contrasts computing.')
    else:
        (contrasts_mean, contrasts_var, contrasts_class_mean, contrasts_class_var,
         ppm_a_contrasts, ppm_g_contrasts) = (None, None, None, None, None, None)

    ###########################################################################
    ##########################################    PLOTS and SNR computation

    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s",
                str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))
    logger.info("Iteration time = %s min %s s",
                str(np.int((CompTime // ni) // 60)), str(np.int((CompTime / ni) % 60)))

    logger.info("perfusion baseline mean = %f", np.mean(AL[0, :, s]))
    logger.info("perfusion baseline var = %f", np.var(AL[0, :, s]))
    logger.info("drifts mean = %f", np.mean(AL[1:, :, s]))
    logger.info("drifts var = %f", np.var(AL[1:, :, s]))
    logger.info("noise mean = %f", np.mean(sigma_eps[s, :]))
    logger.info("noise var = %f", np.var(sigma_eps[s, :]))

    SNR10 = 20 * (np.log10(np.linalg.norm(Y[s, :, :]) / \
                np.linalg.norm(Y[s, :, :] - StimulusInducedSignal - PL[s, :, :])))
    logger.info("SNR = %d",  SNR10)

    return ni, m_A.mean(0), A_std, H, m_C.mean(0), G, Z_tilde, sigma_eps[s, :], \
           mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, AL[:, :, s], PL[s, :, :], \
           np.zeros_like(AL[0, :, s]), Sigma_A.mean(3), Sigma_C[:, :, :, s], Sigma_H, Sigma_G, rerror, \
           contrasts_mean, contrasts_var, ppm_a_nrl, ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, \
           cA[:], cH[2:], cC[2:], cG[2:], cZ[2:], cAH[2:], cCG[2:], cTime, FE

