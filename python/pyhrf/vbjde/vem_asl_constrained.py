# -*- coding: utf-8 -*-
"""VEM BOLD Constrained

File that contains function for BOLD data analysis with positivity
and l2-norm=1 constraints.

It imports functions from vem_tools.py in pyhrf/vbjde
"""

import time
import copy
import logging

import numpy as np

import pyhrf
import pyhrf.vbjde.UtilsC as UtilsC
import pyhrf.vbjde.vem_tools as vt
import pyhrf.vbjde.vem_tools_asl as EM

from pyhrf.boldsynth.hrf import getCanoHRF, genGaussianSmoothHRF

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eps = 1e-4


def Main_vbjde_c_constrained(graph, Y, Onsets, Thrf, K, TR, beta, dt, scale=1,
                             estimateSigmaH=True, estimateSigmaG=True,
                             sigmaH=0.05, sigmaG=0.05, gamma_h=0, gamma_g=0,
                             NitMax=-1, NitMin=1, estimateBeta=True,
                             PLOT=False, idx_first_tag=0, simulation=None,
                             estimateH=True, estimateG=True, estimateA=True,
                             estimateC=True, estimateZ=True, M_step=True,
                             estimateNoise=True, estimateMP=True,
                             estimateLA=True):
    """ Version modified by Lofti from Christine's version """
    logger.info("Fast EM with C extension started ... "
                "Here is the stable version !")
    np.random.seed(6537546)

    #Initialize parameters
    #gamma_h = 7.5
    #gamma_g = 7.5
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    N, J = Y.shape[0], Y.shape[1]
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5

    # Neighbours
    maxNeighbours, neighboursIndexes = EM.create_neighbours(graph, J)
    # Conditions
    X, XX = EM.create_conditions(Onsets, M, N, D, TR, dt)
    # Covariance matrix
    R = EM.covariance_matrix(2, D, dt)
    # Noise matrix
    Gamma = np.identity(N)
    # Noise initialization
    sigma_eps = np.ones(J)

    Crit_AH, Crit_CG = 1, 1
    Crit_H, Crit_G, Crit_Z, Crit_A, Crit_C = 1, 1, 1, 1, 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    CG = np.zeros((J, M, D), dtype=np.float64)
    CG1 = np.zeros((J, M, D), dtype=np.float64)
    cTime = []
    cAH, cCG = [], []
    cA, cC, cH, cG, cZ = [], [], [], [], []
    h_norm, g_norm = [], []

    #Labels
    logger.info("Labels are initialized by setting active probabilities "
                "to ones ...")
    p_Q = np.zeros((M, K, J), dtype=np.float64)
    p_Q[:, 1, :] = 1
    p_Q1 = np.zeros((M, K, J), dtype=np.float64)
    p_q_t = copy.deepcopy(p_Q)
    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    m_h = m_h[:D]
    H = np.array(m_h).astype(np.float64)
    H1 = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)
    G = copy.deepcopy(H)
    G1 = copy.deepcopy(H)
    Sigma_G = copy.deepcopy(Sigma_H)
    #m_H = np.array(m_h).astype(np.float64)
    #m_H1 = np.array(m_h)

    # others
    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    Ndrift = L.shape[0]
    PL = np.dot(P, L)
    alpha = np.zeros((J), dtype=np.float64)
    w = np.ones((N))
    w[idx_first_tag::2] = -1
    w = - w
    W = np.diag(w)
    wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
    y_tilde = Y - PL - wa

    # Parameters Gaussian mixtures
    sigma_Ma = np.ones((M, K), dtype=np.float64)
    sigma_Ma[:, 0] = 0.5
    sigma_Ma[:, 1] = 0.6
    mu_Ma = np.zeros((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_Ma[:, k] = 1
    sigma_Mc = copy.deepcopy(sigma_Ma)
    mu_Mc = copy.deepcopy(mu_Ma)
    # Params RLs
    Sigma_A = np.zeros((M, M, J), np.float64)
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
    m_A = np.zeros((J, M), dtype=np.float64)
    m_A1 = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(mu_Ma[m, k], \
                                np.sqrt(sigma_Ma[m, k])) * p_Q[m, k, j]
    m_A1 = m_A
    Sigma_C = copy.deepcopy(Sigma_A)
    m_C1 = m_C = copy.deepcopy(m_A)

    Q_barnCond = np.zeros((M, M, D, D), dtype=np.float64)
    XGamma = np.zeros((M, D, N), dtype=np.float64)
    XWGamma = np.zeros((M, D, N), dtype=np.float64)
    for m1, k1 in enumerate(X):            # Loop over the M conditions
        for m2, k2 in enumerate(X):
            Q_barnCond[m1, m2, :, :] = np.dot(np.dot(X[k1].T, \
                                                     Gamma), X[k2])
        XGamma[m1, :, :] = np.dot(X[k1].T, Gamma)
        XWGamma[m1, :, :] = np.dot(np.dot(X[k1].T, W), Gamma)

    sigma_eps = np.ones(J)

    if simulation is not None:
        # simulated values
        if not estimateH:
            H = H1 = simulation['primary_brf']
            sigmaH = 20.
        if not estimateG:
            G = G1 = simulation['primary_prf']
            sigmaG = 40.
        if not estimateA:
            A = simulation['brls'].T
            print 'shape BRLs: ', A.shape
            m_A = A
        if not estimateC:
            C = simulation['prls'].T
            print 'shape PRLs: ', C.shape
            m_C = C
        if not estimateZ:
            Z = np.reshape(simulation['labels_vol'], [2, 1, 400])
            Z = np.append(Z, np.ones_like(Z), axis=1)
            print np.reshape(Z[0, 0, :], [20, 20])
        if not estimateLA:
            alpha = np.mean(simulation['perf_baseline'], 0)
            L = simulation['drift_coeffs']
            PL = np.dot(P, L)
            wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
            y_tilde = Y - PL - wa
        if not estimateNoise:
            sigma_eps = np.mean(simulation['noise'], 0)
        if not estimateMP:
            mu_Ma = np.array([[0, 2.2], [0, 2.2]])
            sigma_Ma = np.array([[.3, .3], [.3, .3]])
            mu_Mc = np.array([[0, 1.6], [0, 1.6]])
            sigma_Mc = np.array([[.3, .3], [.3, .3]])

    t1 = time.time()

    ##########################################################################
    #############################################    VBJDE num. iter. minimum

    ni = 0

    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) and (ni < NitMax))):

        logger.info("------------------------------ Iteration n° " + \
                    str(ni + 1) + " ------------------------------")

        #####################
        # EXPECTATION
        #####################

        # A
        if estimateA:
            logger.info("E A step ...")
            UtilsC.expectation_A(p_Q, mu_Mc, sigma_Mc, PL, sigma_eps,
                                 Gamma, Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, K)

        # crit. A
        DIFF = np.reshape(m_A - m_A1, (M * J))
        Crit_A = (np.linalg.norm(DIFF) / \
                    np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        # C
        if estimateC:
            logger.info("E C step ...")
            UtilsC.expectation_C(p_Q, mu_Mc, sigma_Mc, PL, sigma_eps,
                                 Gamma, Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, K)

        # crit. C
        DIFF = np.reshape(m_C - m_C1, (M * J))
        Crit_C = (np.linalg.norm(DIFF) / \
                            np.linalg.norm(np.reshape(m_C1, (M * J)))) ** 2
        cC += [Crit_C]
        m_C1[:, :] = m_C[:, :]

        # HRF h
        if estimateH:
            logger.info("E H step ...")
            UtilsC.expectation_H(XGamma, Q_barnCond, sigma_eps, Gamma, R,
                                 Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, scale,
                                 sigmaH)
            #H = EM.constraint_norm1(m_H, Sigma_H)
            H = H / np.linalg.norm(H)
            if simulation is not None:
                print 'BRF ERROR = ', EM.error(H, simulation['primary_brf'])
            h_norm = np.append(h_norm, np.linalg.norm(H))
            print 'h_norm = ', h_norm

        # crit. h
        Crit_H = (np.linalg.norm(H - H1) / np.linalg.norm(H1)) ** 2
        cH += [Crit_H]
        H1[:] = H[:]

        # PRF g
        if estimateG:
            logger.info("E G step ...")
            UtilsC.expectation_G(XGamma, Q_barnCond, sigma_eps, Gamma, R,
                                 Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, scale,
                                 sigmaH)
            G = EM.constraint_norm1(G, Sigma_G)
            if simulation is not None:
                print 'PRF ERROR = ', EM.error(G, simulation['primary_prf'])
            g_norm = np.append(g_norm, np.linalg.norm(G))
            print 'g_norm = ', g_norm

        # crit. g
        Crit_G = (np.linalg.norm(G - G1) / np.linalg.norm(G1)) ** 2
        cG += [Crit_G]
        G1[:] = G[:]

        # crit. AH
        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * H[d]
        DIFF = np.reshape(AH - AH1, (M * J * D))
        Crit_AH = (np.linalg.norm(DIFF) / \
                  (np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        # crit. CG
        for d in xrange(0, D):
            CG[:, :, d] = m_C[:, :] * G[d]
        DIFF = np.reshape(CG - CG1, (M * J * D))
        Crit_CG = (np.linalg.norm(DIFF) / \
                  (np.linalg.norm(np.reshape(CG1, (M * J * D))) + eps)) ** 2
        cCG += [Crit_CG]
        CG1[:, :, :] = CG[:, :, :]

        # Z labels
        if estimateZ:
            logger.info("E Z step ...")
            UtilsC.expectation_Q(Sigma_A, m_A, sigma_M, Beta, p_q_t, mu_M,
                                 p_Q, neighboursIndexes.astype(np.int32), M,
                                 J, K, maxNeighbours)

        # crit. Z
        DIFF = np.reshape(p_Q - p_Q1, (M * K * J))
        Crit_Z = (np.linalg.norm(DIFF) / \
                 (np.linalg.norm(np.reshape(p_Q1, (M * K * J))) + eps)) ** 2
        cZ += [Crit_Z]
        p_Q1[:, :, :] = p_Q[:, :, :]

        #####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = EM.maximization_sigma_prior(D, Sigma_H, R, H,
                                                      gamma_h)
            else:
                sigmaH = EM.maximization_sigma(D, Sigma_H, R, H)
            logger.info('sigmaH = %s', str(sigmaH))

        # PRF: Sigma_g
        if estimateSigmaG:
            logger.info("M sigma_G step ...")
            if gamma_g > 0:
                sigmaG = vt.maximization_sigma_prior(D, Sigma_G, R, G,
                                                      gamma_g)
            else:
                sigmaG = vt.maximization_sigma(D, Sigma_G, R, G)
            logger.info('sigmaG = %s',  str(sigmaG))

        # (mu_a,sigma_a)
        if estimateMP:
            logger.info("M (mu_a,sigma_a) step ...")
            mu_Ma, sigma_Ma = vt.maximization_mu_sigma(mu_Ma, sigma_Ma, p_Q,
                                                       m_A, K, M, Sigma_A)
            # (mu_c,sigma_c)
            logger.info("M (mu_c,sigma_c) step ...")
            mu_Mc, sigma_Mc = vt.maximization_mu_sigma(mu_Mc, sigma_Mc, p_Q,
                                                     m_A, K, M, Sigma_A)

        # Drift L
        if estimateLA:
            UtilsC.maximization_L(Y, m_A, H, L, P, XX.astype(np.int32), J, D,
                                  M, Ndrift, N)
            PL = np.dot(P, L)
            y_tilde = Y - PL

        # Beta
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                Beta[m] = UtilsC.maximization_beta(beta, \
                            p_Q[m, :, :].astype(np.float64),
                            p_q_t[m, :, :].astype(np.float64), J, K,
                            neighboursIndexes.astype(np.int32), gamma,
                            maxNeighbours, MaxItGrad, gradientStep)
            logger.info("End estimating beta")
            logger.info(Beta)

        # Sigma noise
        if estimateNoise:
            logger.info("M sigma noise step ...")
            UtilsC.maximization_sigma_noise(Gamma, PL, sigma_eps, Sigma_H,
                                            Y, m_A, H, Sigma_A,
                                            XX.astype(np.int32), J, D, M, N)

        t02 = time.time()
        cTime += [t02 - t1]

    t2 = time.time()

    ###########################################################################
    ###########################################    PLOTS and SNR computation

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    if 0:
        Norm = np.linalg.norm(H)
        H /= Norm
        Sigma_H /= Norm ** 2
        sigmaH /= Norm ** 2
        m_A *= Norm
        Sigma_A *= Norm ** 2
        mu_Ma *= Norm
        sigma_Ma *= Norm ** 2
        sigma_Ma = np.sqrt(np.sqrt(sigma_Ma))
        Norm = np.linalg.norm(G)
        G /= Norm
        Sigma_G /= Norm ** 2
        sigmaG /= Norm ** 2
        m_C *= Norm
        Sigma_C *= Norm ** 2
        mu_Mc *= Norm
        sigma_Mc *= Norm ** 2
        sigma_Mc = np.sqrt(np.sqrt(sigma_Mc))

    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s", str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))

    StimulusInducedSignal = vt.computeFit(H, m_A, X, J, N)
    SNR = 20 * np.log(np.linalg.norm(Y) / \
               np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)

    logger.info('mu_Ma: %f', mu_Ma)
    logger.info('sigma_Ma: %f', sigma_Ma)
    logger.info("sigma_H = %s" + str(sigmaH))
    logger.info("Beta = %s" + str(Beta))
    logger.info('SNR comp = %f', SNR)

    return ni, m_A, H, p_Q, sigma_eps, mu_Ma, sigma_Ma, Beta, L, PL, \
           cA[2:], cH[2:], cZ[2:], cAH[2:], cTime[2:], \
           cTimeMean, Sigma_A, StimulusInducedSignal


def Main_vbjde_constrained(graph, Y, Onsets, Thrf, K, TR, beta, dt, scale=1,
                           estimateSigmaH=True, estimateSigmaG=True,
                           sigmaH=0.05, sigmaG=0.05, gamma_h=0, gamma_g=0,
                           NitMax=-1, NitMin=1, estimateBeta=True, PLOT=False,
                           idx_first_tag=0, simulation=None,
                           estimateH=True, estimateG=True, estimateA=True,
                           estimateC=True, estimateZ=True, estimateNoise=True,
                           estimateMP=True, estimateLA=True):
    """ Version modified by Lofti from Christine's version """
    logger.info("EM for ASL!")
    np.random.seed(6537546)

    # Initialization
    gamma_h = 10000000000  #7.5 1000000000
    gamma_g = 100000000  #7.5 1000000000
    gamma = 7.5
    beta = 1.
    Thresh = 1e-5
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    N, J = Y.shape[0], Y.shape[1]
    Crit_AH, Crit_CG = 1, 1
    Crit_H, Crit_G, Crit_Z, Crit_A, Crit_C = 1, 1, 1, 1, 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    CG = np.zeros((J, M, D), dtype=np.float64)
    CG1 = np.zeros((J, M, D), dtype=np.float64)
    cTime = []
    cAH, cCG = [], []
    cA, cC, cH, cG, cZ = [], [], [], [], []
    h_norm, g_norm = [], []
    SUM_q_Z = [[] for m in xrange(M)]
    mua1 = [[] for m in xrange(M)]
    muc1 = [[] for m in xrange(M)]
    
    # Neighbours
    maxNeighbours, neighboursIndexes = EM.create_neighbours(graph, J)
    # Conditions
    X, XX = EM.create_conditions(Onsets, M, N, D, TR, dt)
    # Covariance matrix
    #R = EM.covariance_matrix(2, D, dt)
    _, R = genGaussianSmoothHRF(False, D, dt, 1., 2)
    # Noise matrix
    Gamma = np.identity(N)
    # Noise initialization
    sigma_eps = np.ones(J)
    #Labels
    logger.info("Labels are initialized by setting active probabilities "
                "to ones ...")
    p_Q = np.zeros((M, K, J), dtype=np.float64)
    p_Q[:, 1, :] = 1
    p_Q1 = copy.deepcopy(p_Q)
    p_q_t = copy.deepcopy(p_Q)
    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    m_h = m_h[:D]
    H = np.array(m_h).astype(np.float64)
    H1 = np.array(m_h).astype(np.float64)
    Ht = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)
    #Omega = linear_rf_operator(len(H), phy_params, dt, calculating_brf=False)
    #G = np.dot(Omega, Mu)
    G = copy.deepcopy(H)
    G1 = copy.deepcopy(H)
    Gt = copy.deepcopy(H)
    Sigma_G = copy.deepcopy(Sigma_H)
    # others
    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    alpha = np.zeros((J), dtype=np.float64)
    w = np.ones((N))
    w[idx_first_tag::2] = -1
    w = - w
    W = np.diag(w)
    wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
    y_tilde = Y - PL - wa
    # Parameters Gaussian mixtures
    sigma_Ma = np.ones((M, K), dtype=np.float64)
    sigma_Ma[:, 0] = 0.5
    sigma_Ma[:, 1] = 0.6
    mu_Ma = np.zeros((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_Ma[:, k] = 1
    sigma_Mc = copy.deepcopy(sigma_Ma)
    mu_Mc = copy.deepcopy(mu_Ma)
    # Params RLs
    Sigma_A = np.zeros((M, M, J), np.float64)
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
    m_A = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(mu_Ma[m, k], \
                                np.sqrt(sigma_Ma[m, k])) * p_Q[m, k, j]
    Sigma_C = copy.deepcopy(Sigma_A)
    m_C = copy.deepcopy(m_A)
    m_C1 = copy.deepcopy(m_C)
    m_A1 = copy.deepcopy(m_A)

    if simulation is not None:
        #print simulation
        # simulated values
        if not estimateH:
            H = Ht = simulation['brf'][:, 0]
        if not estimateSigmaH:
            sigmaH = sigmaH
        if not estimateG:
            G = Gt = simulation['prf'][:, 0]
        if not estimateSigmaG:
            sigmaG = sigmaG
        A = simulation['brls'].T
        if not estimateA:
            m_A = A
        C = simulation['prls'].T
        if not estimateC:
            m_C = C
        Z = simulation['labels']
        Z = np.append(1-Z[:, np.newaxis, :], Z[:, np.newaxis, :], axis=1)
        #Z[:, 1, :] = 1
        if not estimateZ:
            p_Q = copy.deepcopy(Z)
            p_q_t = copy.deepcopy(Z)
        if not estimateLA:
            alpha = np.mean(simulation['perf_baseline'], 0)
            L = simulation['drift_coeffs']
            PL = np.dot(P, L)
            wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
            y_tilde = Y - PL - wa
        if not estimateNoise:
            sigma_eps = np.var(simulation['noise'], 0)
        if not estimateMP:
            #print simulation['condition_defs'][0]
            mu_Ma = np.array([[0, 2.2], [0, 2.2]])
            sigma_Ma = np.array([[.3, .3], [.3, .3]])
            mu_Mc = np.array([[0, 1.8], [0, 1.8]])
            sigma_Mc = np.array([[.3, .3], [.3, .3]])
    #print simulation['condition_defs'][0]
    #print simulation['condition_defs'][0]

    ###########################################################################
    #############################################             VBJDE

    t1 = time.time()
    ni = 0
    
    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) and (Crit_CG > Thresh) \
            and (ni < NitMax))):
        
        logger.info("-------- Iteration n° " + str(ni + 1) + " ---------")

        #####################
        # EXPECTATION
        #####################

        # HRF H
        logger.info("E H step ...")
        if estimateH:
            logger.info("estimation")
            Ht, Sigma_H = EM.expectation_H(Sigma_A, m_A, m_C, G, X, W, Gamma,
                                           D, J, N, y_tilde, sigma_eps, scale,
                                           R, sigmaH)
            H = EM.constraint_norm1_b(Ht, Sigma_H)
            #H = Ht / np.linalg.norm(Ht)
            if simulation is not None:
                print 'BRF ERROR = ', EM.error(H, simulation['brf'][:, 0])
            h_norm = np.append(h_norm, np.linalg.norm(H))
            print 'h_norm = ', h_norm
            # crit. h
            Crit_H = (np.linalg.norm(H - H1) / np.linalg.norm(H1)) ** 2
            cH += [Crit_H]
            H1 = H

        # PRF G
        logger.info("E G step ...")
        if estimateG:
            logger.info("estimation")
            Gt, Sigma_G = EM.expectation_G(Sigma_C, m_C, m_A, H, X, W, Gamma,
                                           D, J, N, y_tilde, sigma_eps, scale,
                                           R, sigmaG)
            G = EM.constraint_norm1_b(Gt, Sigma_G, positivity=False)
            #G = Gt / np.linalg.norm(Gt)
            if simulation is not None:
                print 'PRF ERROR = ', EM.error(G, simulation['prf'][:, 0])
            g_norm = np.append(g_norm, np.linalg.norm(G))
            print 'g_norm = ', g_norm
            # crit. g
            Crit_G = (np.linalg.norm(G - G1) / np.linalg.norm(G1)) ** 2
            cG += [Crit_G]
            G1 = G

        # A
        logger.info("E A step ...")
        if estimateA:
            logger.info("estimation")
            m_A, Sigma_A = EM.expectation_A(H, m_A, G, m_C, W, X, Gamma, q_Z,
                                            mu_Ma, sigma_Ma, D, J, M, K,
                                            y_tilde, Sigma_A, sigma_eps)
            if simulation is not None:
                print 'BRLS ERROR = ', EM.error(m_A, A)
            # crit. A
            DIFF = np.reshape(m_A - m_A1, (M * J))
            Crit_A = (np.linalg.norm(DIFF) / \
                        np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
            cA += [Crit_A]
            m_A1[:, :] = m_A[:, :]
        
        # C
        logger.info("E C step ...")
        if estimateC:
            logger.info("estimation")
            m_C, Sigma_C = EM.expectation_C(G, m_C, H, m_A, W, X, Gamma, q_Z,
                                            mu_Mc, sigma_Mc, D, J, M, K,
                                            y_tilde, Sigma_C, sigma_eps)
            #print 'true values: ', C
            #print 'estimated values: ', m_C
            if simulation is not None:
                print 'PRLS ERROR = ', EM.error(m_C, C)
            # crit. C
            DIFF = np.reshape(m_C - m_C1, (M * J))
            Crit_C = (np.linalg.norm(DIFF) / \
                        np.linalg.norm(np.reshape(m_C1, (M * J)))) ** 2
            cC += [Crit_C]
            m_C1[:, :] = m_C[:, :]

        # Q labels
        logger.info("E Z step ...")
        if estimateZ:
            logger.info("estimation")
            q_Z, Z_tilde = EM.expectation_Z(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, graph, M, J, K)
            if simulation is not None:
                print 'LABELS ERROR = ', EM.error(q_Z, Z)
                print 'Z =   ', (Z.flatten()).astype(np.int32)
                print 'q_Z = ', (q_Z.flatten()*100).astype(np.int32)
                print 'Z_t = ', (Z_tilde.flatten()*100).astype(np.int32)
            # crit. Z
            Crit_Z = (np.linalg.norm((p_Q - p_Q1).flatten()) / \
                         (np.linalg.norm(p_Q1).flatten() + eps)) ** 2
            cZ += [Crit_Z]
            p_Q1 = p_Q

        # crit. AH and CG
        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * H[d]
            CG[:, :, d] = (m_C[:, :] * G[d])
        Crit_AH = (np.linalg.norm(np.reshape(AH - AH1, (M * J * D))) / \
                  (np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
        cAH += [Crit_AH]
        AH1 = AH
        Crit_CG = (np.linalg.norm(np.reshape(CG - CG1, (M * J * D))) / \
                  (np.linalg.norm(np.reshape(CG1, (M * J * D))) + eps)) ** 2
        cCG += [Crit_CG]
        CG1 = CG
        print Crit_AH
        print Crit_CG

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            import matplotlib.pyplot as plt
            if ni==0:
                plt.close('all')
            plt.figure(M + 1)
            plt.plot(H)
            plt.hold(True)
            plt.figure(M + 2)
            plt.plot(G)
            plt.hold(True)

        #####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            print gamma_h
            #sigmaH = EM.maximization_sigma_prior(D, R, H, gamma_h)
            #print sigmaH
            sigmaH = EM.maximization_sigma(D, R, H)
            print sigmaH
            logger.info("sigmaH = " + str(sigmaH))
        # PRF: Sigma_g
        if estimateSigmaG:
            logger.info("M sigma_G step ...")
            print gamma_g
            #sigmaG = EM.maximization_sigma_prior(D, R, G, gamma_g)
            #print sigmaG
            sigmaG = EM.maximization_sigma(D, R, G)
            print 'sigmaG', sigmaG
            logger.info('sigmaG = ' + str(sigmaG))
        # (mu,sigma)
        if estimateMP:
            logger.info("M (mu,sigma) a and c step ...")
            mu_Ma, sigma_Ma = EM.maximization_mu_sigma(mu_Ma, sigma_Ma,
                                                   p_Q, m_A, K, M, Sigma_A)
            mu_Mc, sigma_Mc = EM.maximization_mu_sigma(mu_Mc, sigma_Mc,
                                                   p_Q, m_C, K, M, Sigma_C)
            logger.info("(mu,sigma)_a = ")
            logger.info(mu_Ma)
            logger.info(sigma_Ma)
            logger.info("(mu,sigma)_c = ")
            logger.info(mu_Mc)
            logger.info(sigma_Mc)
        # Drift L, alpha
        if estimateLA:
            L, alpha = EM.maximization_L_alpha(Y, m_A, m_C, X, W, w, H, \
                                               G, L, P, alpha)
            if simulation is not None:
                print 'ALPHA ERROR = ', EM.error(alpha, np.mean(\
                                            simulation['perf_baseline'], 0))
                print 'DRIFT ERROR = ', EM.error(L, simulation['drift_coeffs'])
            #alpha = np.zeros_like(np.mean(simulation['perf_baseline'], 0))
            PL = np.dot(P, L)
            wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
            y_tilde = Y - PL - wa

        # Beta
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                Beta[m] = EM.maximization_beta(Beta[m], p_Q, p_q_t,
                                        J, K, m, graph, gamma,
                                        neighboursIndexes, maxNeighbours)
            print Beta
            logger.info("End estimating beta")
            logger.info(Beta)

        # Sigma noise
        if estimateNoise:
            logger.info("M sigma noise step ...")
            sigma_eps = EM.maximization_sigma_noise(Y, X, m_A, Sigma_A, H,
                          m_C, Sigma_C, G, W, M, N, J, y_tilde, sigma_eps)
            if simulation is not None:
                print 'NOISE ERROR = ', EM.error(sigma_eps,
                                             np.var(simulation['noise'], 0))
            
        for m in xrange(M):
            SUM_p_Q[m] += [sum(p_Q[m, 1, :])]
            mua1[m] += [mu_Ma[m, 1]]
            muc1[m] += [mu_Mc[m, 1]]

        ni += 1
        t02 = time.time()
        cTime += [t02 - t1]

    t2 = time.time()
    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    ###########################################################################
    ##########################################    PLOTS and SNR computation

    SUM_p_Q_array = np.zeros((M, ni), dtype=np.float64)
    mua1_array = np.zeros((M, ni), dtype=np.float64)
    muc1_array = np.zeros((M, ni), dtype=np.float64)
    #h_norm_array = np.zeros((ni), dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_p_Q_array[m, i] = SUM_p_Q[m][i]
            mua1_array[m, i] = mua1[m][i]
            muc1_array[m, i] = muc1[m][i]
            #h_norm_array[i] = h_norm[i]

    if PLOT:
        font = {'size': 15}
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rc('font', **font)
        plt.figure(M + 1)
        plt.savefig('./info/BRF_Iter_ASL.png')
        plt.figure(M + 2)
        plt.savefig('./info/PRF_Iter_ASL.png')
        plt.hold(False)
        plt.figure(M + 3)
        plt.plot(cAH[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cCG[1:-1], 'm')
        plt.hold(False)
        plt.legend(('CAH', 'CCG'))
        plt.grid(True)
        plt.savefig('./info/Crit_ASL.png')
        plt.figure(7)
        plt.plot(cA[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cC[1:-1], 'm')
        plt.plot(cH[1:-1], 'green')
        plt.plot(cG[1:-1], 'red')
        plt.hold(False)
        plt.legend(('CA', 'CC', 'CH', 'CG'))
        plt.grid(True)
        plt.savefig('./info/Crit_all.png')
        plt.figure(4)
        for m in xrange(M):
            plt.plot(SUM_p_Q_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./info/Sum_q_Z_Iter_ASL.png')
        plt.figure(5)
        for m in xrange(M):
            plt.plot(mua1_array[m])
            plt.hold(True)
            plt.plot(muc1_array[m])
        plt.hold(False)
        plt.legend(('mu_a', 'mu_c'))
        plt.savefig('./info/mu1_Iter_ASL.png')
        
    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s",
                str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))

    StimulusInducedSignal = EM.computeFit(H, m_A, G, m_C, W, X, J, N)
    SNR = 20 * (np.log(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - PL))) / np.log(10.)
    """
    logger.info('mu_Ma: %f', mu_Ma)
    logger.info('sigma_Ma: %f', sigma_Ma)
    logger.info("sigma_H = %s" + str(sigmaH))
    logger.info('mu_Mc: %f', mu_Mc)
    logger.info('sigma_Mc: %f', sigma_Mc)
    logger.info("sigma_G = %s" + str(sigmaG))
    logger.info("Beta = %s" + str(Beta))
    logger.info('SNR comp = %f', SNR)"""

    return ni, m_A, H, m_C, G, q_Z, sigma_eps, \
           mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, L, PL, \
           Sigma_A, Sigma_C
           #cA[2:], cH[2:], cZ[2:], \
           #cAH[2:], cCG[2:], cTime[2:], cTimeMean[2:], \
 
