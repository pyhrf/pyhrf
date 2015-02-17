# -*- coding: utf-8 -*-
"""VEM BOLD Constrained

File that contains function for BOLD data analysis with positivity
and l2-norm=1 constraints.

It imports functions from vem_tools.py in pyhrf/vbjde
"""

#import os.path as op
import numpy as np
import time
import UtilsC
import pyhrf
#from pyhrf.tools._io import read_volume
from pyhrf.boldsynth.hrf import getCanoHRF
#from pyhrf.ndarray import xndarray
import vem_tools as vt
import vem_tools_asl as EM
import copy
#try:
#    from collections import OrderedDict
#except ImportError:
#    from pyhrf.tools.backports import OrderedDict

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
    pyhrf.verbose(1, "Fast EM with C extension started ... Here is the \
                      stable version !")
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
    pyhrf.verbose(3, "Labels are initialized by setting active probabilities \
                        to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = np.zeros((M, K, J), dtype=np.float64)
    Z_tilde = copy.deepcopy(q_Z)
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
                                np.sqrt(sigma_Ma[m, k])) * q_Z[m, k, j]
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

        pyhrf.verbose(1, "------------------------------ Iteration n° " + \
                         str(ni + 1) + " ------------------------------")

        #####################
        # EXPECTATION
        #####################

        # A
        if estimateA:
            pyhrf.verbose(3, "E A step ...")
            UtilsC.expectation_A(q_Z, mu_Mc, sigma_Mc, PL, sigma_eps,
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
            pyhrf.verbose(3, "E C step ...")
            UtilsC.expectation_C(q_Z, mu_Mc, sigma_Mc, PL, sigma_eps,
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
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H(XGamma, Q_barnCond, sigma_eps, Gamma, R,
                                 Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, scale,
                                 sigmaH)
            #H = EM.constraint_norm1(m_H, Sigma_H)
            H = H / np.linalg.norm(H)
            print 'BRF ERROR = ', EM.error(H, simulation['primary_brf'])
            h_norm = np.append(h_norm, np.linalg.norm(H))
            print 'h_norm = ', h_norm

        # crit. h
        Crit_H = (np.linalg.norm(H - H1) / np.linalg.norm(H1)) ** 2
        cH += [Crit_H]
        H1[:] = H[:]

        # PRF g
        if estimateG:
            pyhrf.verbose(3, "E G step ...")
            UtilsC.expectation_G(XGamma, Q_barnCond, sigma_eps, Gamma, R,
                                 Sigma_H, Y, y_tilde, m_A, H, Sigma_A,
                                 XX.astype(np.int32), J, D, M, N, scale,
                                 sigmaH)
            G = EM.constraint_norm1(G, Sigma_G)
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
            pyhrf.verbose(3, "E Z step ...")
            UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M,
                                 q_Z, neighboursIndexes.astype(np.int32), M,
                                 J, K, maxNeighbours)

        # crit. Z
        DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
        Crit_Z = (np.linalg.norm(DIFF) / \
                 (np.linalg.norm(np.reshape(q_Z1, (M * K * J))) + eps)) ** 2
        cZ += [Crit_Z]
        q_Z1[:, :, :] = q_Z[:, :, :]

        #####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateSigmaH:
            pyhrf.verbose(3, "M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = EM.maximization_sigma_prior(D, Sigma_H, R, H,
                                                      gamma_h)
            else:
                sigmaH = EM.maximization_sigma(D, Sigma_H, R, H)
            pyhrf.verbose(3, 'sigmaH = ' + str(sigmaH))

        # PRF: Sigma_g
        if estimateSigmaG:
            pyhrf.verbose(3, "M sigma_G step ...")
            if gamma_g > 0:
                sigmaG = vt.maximization_sigma_prior(D, Sigma_G, R, G,
                                                      gamma_g)
            else:
                sigmaG = vt.maximization_sigma(D, Sigma_G, R, G)
            pyhrf.verbose(3, 'sigmaG = ' + str(sigmaG))

        # (mu_a,sigma_a)
        if estimateMP:
            pyhrf.verbose(3, "M (mu_a,sigma_a) step ...")
            mu_Ma, sigma_Ma = vt.maximization_mu_sigma(mu_Ma, sigma_Ma, q_Z,
                                                       m_A, K, M, Sigma_A)
            # (mu_c,sigma_c)
            pyhrf.verbose(3, "M (mu_c,sigma_c) step ...")
            mu_Mc, sigma_Mc = vt.maximization_mu_sigma(mu_Mc, sigma_Mc, q_Z,
                                                     m_A, K, M, Sigma_A)

        # Drift L
        if estimateLA:
            UtilsC.maximization_L(Y, m_A, H, L, P, XX.astype(np.int32), J, D,
                                  M, Ndrift, N)
            PL = np.dot(P, L)
            y_tilde = Y - PL

        # Beta
        if estimateBeta:
            pyhrf.verbose(3, "estimating beta")
            for m in xrange(0, M):
                Beta[m] = UtilsC.maximization_beta(beta, \
                            q_Z[m, :, :].astype(np.float64),
                            Z_tilde[m, :, :].astype(np.float64), J, K,
                            neighboursIndexes.astype(np.int32), gamma,
                            maxNeighbours, MaxItGrad, gradientStep)
            pyhrf.verbose(3, "End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        # Sigma noise
        if estimateNoise:
            pyhrf.verbose(3, "M sigma noise step ...")
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

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" % ni)
    pyhrf.verbose(1, "Computational time = " + str(np.int(CompTime // 60)) + \
                        " min " + str(np.int(CompTime % 60)) + " s")

    StimulusInducedSignal = vt.computeFit(H, m_A, X, J, N)
    SNR = 20 * np.log(np.linalg.norm(Y) / \
               np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)

    if pyhrf.verbose.verbosity > 1:
        print 'mu_Ma:', mu_Ma
        print 'sigma_Ma:', sigma_Ma
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
        print 'SNR comp =', SNR

    return ni, m_A, H, q_Z, sigma_eps, mu_Ma, sigma_Ma, Beta, L, PL, \
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
    pyhrf.verbose(1, "EM for ASL!")
    np.random.seed(6537546)

    # Initialization
    gamma_h = 1000000000  #7.5
    gamma_g = 1000000000  #7.5
    Thresh = 1e-5
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    N, J = Y.shape[0], Y.shape[1]
    Crit_AH, Crit_CG = 1, 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    CG = np.zeros((J, M, D), dtype=np.float64)
    CG1 = np.zeros((J, M, D), dtype=np.float64)
    cTime = []
    cZ = []
    cAH = []
    cCG = []
    h_norm = []
    g_norm = []
    SUM_q_Z = [[] for m in xrange(M)]
    mua1 = [[] for m in xrange(M)]
    muc1 = [[] for m in xrange(M)]

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
    #Labels
    pyhrf.verbose(3, "Labels are initialized by setting active probabilities \
                        to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = copy.deepcopy(q_Z)
    Z_tilde = copy.deepcopy(q_Z)
    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    m_h = m_h[:D]
    H = np.array(m_h).astype(np.float64)
    Ht = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)
    G = copy.deepcopy(H)
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
                                np.sqrt(sigma_Ma[m, k])) * q_Z[m, k, j]
    Sigma_C = copy.deepcopy(Sigma_A)
    m_C = copy.deepcopy(m_A)

    if simulation is not None:
        #print simulation
        # simulated values
        if not estimateH:
            H = Ht = simulation['brf'][:, 0]
            sigmaH = 20.
        if not estimateG:
            G = Gt = simulation['prf'][:, 0]
            sigmaG = 40.
        A = simulation['brls'].T
        if not estimateA:
            m_A = A
        C = simulation['prls'].T
        if not estimateC:
            m_C = C
        Z = simulation['labels']
        Z = np.append(Z[:, np.newaxis, :], Z[:, np.newaxis, :], axis=1)
        #Z[:, 1, :] = 1
        if not estimateZ:
            q_Z = copy.deepcopy(Z)
            Z_tilde = copy.deepcopy(Z)
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
            mu_Mc = np.array([[0, 1.6], [0, 1.6]])
            sigma_Mc = np.array([[.3, .3], [.3, .3]])
    #print simulation['condition_defs'][0]
    #print simulation['condition_defs'][0]

    #sigmaH = 0.0001
    #sigmaG = 0.0001


    ###########################################################################
    #############################################             VBJDE

    t1 = time.time()
    ni = 0

    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) and (Crit_CG > Thresh) \
            and (ni < NitMax))):

        pyhrf.verbose(1, "-------- Iteration n° " + str(ni + 1) + " ---------")

        #####################
        # EXPECTATION
        #####################

        # HRF H
        pyhrf.verbose(3, "E H step ...")
        if estimateH:
            pyhrf.verbose(3, "estimation")
            #sigmaH = 0.0001            
            print sigmaH
            Ht, Sigma_H = EM.expectation_H(Sigma_A, m_A, m_C, G, X, W, Gamma,
                                           D, J, N, y_tilde, sigma_eps, scale,
                                           R, sigmaH)
            H = EM.constraint_norm1_b(Ht, Sigma_H)
            #H = Ht / np.linalg.norm(Ht)
            print 'BRF ERROR = ', EM.error(H, simulation['brf'][:, 0])
            h_norm = np.append(h_norm, np.linalg.norm(H))
            print 'h_norm = ', h_norm

        # PRF G
        pyhrf.verbose(3, "E G step ...")
        if estimateG:
            pyhrf.verbose(3, "estimation")
            Gt, Sigma_G = EM.expectation_G(Sigma_C, m_C, m_A, H, X, W, Gamma,
                                           D, J, N, y_tilde, sigma_eps, scale,
                                           R, sigmaG)
            G = EM.constraint_norm1_b(Gt, Sigma_G, positivity=True)
            #G = Gt / np.linalg.norm(Gt)
            print 'PRF ERROR = ', EM.error(G, simulation['prf'][:, 0])
            g_norm = np.append(g_norm, np.linalg.norm(G))
            print 'g_norm = ', g_norm

        # A
        pyhrf.verbose(3, "E A step ...")
        if estimateA:
            pyhrf.verbose(3, "estimation")
            m_A, Sigma_A = EM.expectation_A(H, m_A, G, m_C, W, X, Gamma, q_Z,
                                            mu_Ma, sigma_Ma, D, J, M, K,
                                            y_tilde, Sigma_A, sigma_eps)
            print 'BRLS ERROR = ', EM.error(m_A, A)

        # C
        pyhrf.verbose(3, "E C step ...")
        if estimateC:
            pyhrf.verbose(3, "estimation")
            m_C, Sigma_C = EM.expectation_C(G, m_C, H, m_A, W, X, Gamma, q_Z,
                                            mu_Mc, sigma_Mc, D, J, M, K,
                                            y_tilde, Sigma_C, sigma_eps)
            #print 'true values: ', C
            #print 'estimated values: ', m_C
            print 'PRLS ERROR = ', EM.error(m_C, C)

        # Q labels
        pyhrf.verbose(3, "E Z step ...")
        if estimateZ:
            pyhrf.verbose(3, "estimation")
            q_Z, Z_tilde = EM.expectation_Z(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, graph, M, J, K)
            #print 'LABELS ERROR = ', EM.error(q_Z, Z)
            # crit. Z
            Crit_Z = (np.linalg.norm((q_Z - q_Z1).flatten()) / \
                         (np.linalg.norm(q_Z1).flatten() + eps)) ** 2
            cZ += [Crit_Z]
            q_Z1 = q_Z

        

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

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            import matplotlib.pyplot as plt
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
            pyhrf.verbose(3, "M sigma_H step ...")
            print gamma_h
            sigmaH = EM.maximization_sigma_prior(D, R, H, gamma_h)
            pyhrf.verbose(3, 'sigmaH = ' + str(sigmaH))
        # PRF: Sigma_g
        if estimateSigmaG:
            pyhrf.verbose(3, "M sigma_G step ...")
            sigmaG = EM.maximization_sigma_prior(D, R, G, gamma_g)
            pyhrf.verbose(3, 'sigmaG = ' + str(sigmaG))
        # (mu,sigma)
        if estimateMP:
            pyhrf.verbose(3, "M (mu,sigma) a and c step ...")
            Mu_Ma, sigma_Ma = EM.maximization_mu_sigma(mu_Ma, sigma_Ma,
                                                   q_Z, m_A, K, M, Sigma_A)
            mu_Mc, sigma_Mc = EM.maximization_mu_sigma(mu_Mc, sigma_Mc,
                                                   q_Z, m_C, K, M, Sigma_C)

        # Drift L, alpha
        if estimateLA:
            L, alpha = EM.maximization_L_alpha(Y, m_A, m_C, X, W, w, H, \
                                               G, L, P, alpha)
            print 'ALPHA ERROR = ', EM.error(alpha, np.mean(\
                                            simulation['perf_baseline'], 0))
            print 'DRIFT ERROR = ', EM.error(L, simulation['drift_coeffs'])
            #alpha = np.zeros_like(np.mean(simulation['perf_baseline'], 0))
            PL = np.dot(P, L)
            wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
            y_tilde = Y - PL - wa

        # Beta
        if estimateBeta:
            pyhrf.verbose(3, "estimating beta")
            for m in xrange(0, M):
                Beta[m] = EM.maximization_beta(Beta[m], q_Z, Z_tilde,
                                        J, K, m, graph, gamma_h,
                                        neighboursIndexes, maxNeighbours)
            pyhrf.verbose(3, "End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        # Sigma noise
        if estimateNoise:
            pyhrf.verbose(3, "M sigma noise step ...")
            sigma_eps = EM.maximization_sigma_noise(Y, X, m_A, Sigma_A, H,
                          m_C, Sigma_C, G, W, M, N, J, y_tilde, sigma_eps)
            print 'NOISE ERROR = ', EM.error(sigma_eps,
                                             np.var(simulation['noise'], 0))
            #print '  - est var noise: ', sigma_eps
            #print '  - sim var noise: ', np.var(simulation['noise'], 0)

        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m, 1, :])]
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

    SUM_q_Z_array = np.zeros((M, ni), dtype=np.float64)
    mua1_array = np.zeros((M, ni), dtype=np.float64)
    muc1_array = np.zeros((M, ni), dtype=np.float64)
    #h_norm_array = np.zeros((ni), dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_q_Z_array[m, i] = SUM_q_Z[m][i]
            mua1_array[m, i] = mua1[m][i]
            muc1_array[m, i] = muc1[m][i]
            #h_norm_array[i] = h_norm[i]

    if PLOT:
        font = {'size': 15}
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rc('font', **font)
        plt.figure(M + 1)
        plt.savefig('./BRF_Iter_ASL.png')
        plt.figure(M + 2)
        plt.savefig('./PRF_Iter_ASL.png')
        plt.hold(False)
        plt.figure(2)
        plt.plot(cAH[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cCG[1:-1], 'm')
        plt.hold(False)
        plt.legend(('CAH', 'CCG'))
        plt.grid(True)
        plt.savefig('./Crit_ASL.png')
        plt.figure(4)
        for m in xrange(M):
            plt.plot(SUM_q_Z_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./Sum_q_Z_Iter_ASL.png')
        """plt.figure(5)
        for m in xrange(M):
            plt.plot(mua1_array[m])
            plt.hold(True)
            plt.plot(muc1_array[m])
        plt.hold(False)
        plt.legend(('mu_a', 'mu_c'))
        plt.savefig('./mu1_Iter_ASL.png')
        plt.figure(6)
        plt.plot(h_norm_array)
        plt.savefig('./HRF_Norm_ASL.png')"""

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" % ni)
    pyhrf.verbose(1, "Computational time = " + str(np.int(CompTime // 60)) + \
                        " min " + str(np.int(CompTime % 60)) + " s")

    StimulusInducedSignal = EM.computeFit(H, m_A, G, m_C, W, X, J, N)
    SNR = 20 * (np.log(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - PL))) / np.log(10.)

    if pyhrf.verbose.verbosity > 1:
        print 'mu_Ma:', mu_Ma
        print 'sigma_Ma:', sigma_Ma
        print "sigma_H = " + str(sigmaH)
        print 'mu_Mc:', mu_Mc
        print 'sigma_Mc:', sigma_Mc
        print "sigma_G = " + str(sigmaG)
        print "Beta = " + str(Beta)
        print 'SNR comp =', SNR

    return ni, m_A, H, m_C, G, q_Z, sigma_eps, cZ[2:],\
           cTime, cTimeMean, mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, L, PL, \
           Sigma_A, Sigma_C, StimulusInducedSignal
           #cA[2:], cH[2:],  cAH[2:],\