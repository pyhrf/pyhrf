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
import pyhrf.vbjde.UtilsC as UtilsC
import pyhrf.vbjde.vem_tools as vt
import pyhrf.vbjde.vem_tools_asl as EM

from pyhrf.boldsynth.hrf import getCanoHRF, genGaussianSmoothHRF, \
                                genGaussianSmoothHRF_cust
from pyhrf.sandbox.physio_params import PHY_PARAMS_KHALIDOV11, \
                                        linear_rf_operator,\
                                        create_physio_brf, \
                                        create_physio_prf
from pyhrf.tools.aexpression import ArithmeticExpression as AExpr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eps = 1e-6


def Main_vbjde_physio(graph, Y, Onsets, durations, Thrf, K, TR, beta, dt,
                      scale=1, estimateSigmaH=True, estimateSigmaG=True,
                      sigmaH=0.05, sigmaG=0.05, gamma_h=0, gamma_g=0,
                      NitMax=-1, NitMin=1, estimateBeta=True, PLOT=False,
                      contrasts=[], computeContrast=False,
                      idx_first_tag=0, simulation=None, sigmaMu=None,
                      estimateH=True, estimateG=True, estimateA=True,
                      estimateC=True, estimateZ=True, estimateNoise=True,
                      estimateMP=True, estimateLA=True, use_hyperprior=False,
                      positivity=False, constraint=False,
                      phy_params=PHY_PARAMS_KHALIDOV11,
                      prior_balloon=False, prior_omega=False,
                      prior_hierarchical=False):
    """ Version modified by Lofti from Christine's version """
    logger.info("EM for ASL!")
    np.random.seed(6537546)
    #NitMax=NitMin=0
    logger.info("data shape: ")
    logger.info(Y.shape)
        
    # Initialization
    gamma_h = 1  # 10000000000  # 7.5
    gamma_g = 1  # 7.5
    
    #Parameters to get very smooth RFs
    #gamma_h = 100000000000
    #gamma_g = 7.5

    #Parameters to get close RFs but a biiiit noisy
    #gamma_h = 1000000
    #gamma_g = 1000000

    #Parameters to get smooth but close RFs
    #gamma_h = 6000000
    #gamma_g = 6000000
    
    gamma = 7.5
    beta = 1.
    Thresh = 1e-5
    print Thrf / dt
    print Thrf
    print dt
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    print 'D = ', D
    #D, M = np.int(np.ceil(Thrf / dt)), len(Onsets)
    N, J = Y.shape[0], Y.shape[1]
    Crit_AH, Crit_CG = 1, 1
    Crit_H, Crit_G, Crit_Z, Crit_A, Crit_C = 1, 1, 1, 1, 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    CG = np.zeros((J, M, D), dtype=np.float64)
    CG1 = np.zeros((J, M, D), dtype=np.float64)
    cTime = []
    rerror = []
    cAH, cCG = [], []
    cA, cC, cH, cG, cZ = [], [], [], [], []
    h_norm, g_norm = [], []
    SUM_q_Z = [[] for m in xrange(M)]
    mua1 = [[] for m in xrange(M)]
    muc1 = [[] for m in xrange(M)]
    CONTRAST_A = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRASTVAR_A = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRAST_C = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRASTVAR_C = np.zeros((J, len(contrasts)), dtype=np.float64)
    
    # Neighbours
    maxNeighbours, neighboursIndexes = EM.create_neighbours(graph, J)
    # Control-tag
    w = np.ones((N))
    w[idx_first_tag::2] = -1
    w = - w
    W = np.diag(w)
    # Conditions
    #X, XX, condition_names = EM.create_conditions_block(Onsets, durations, M, N, D, TR, dt)
    X, XX, condition_names = EM.create_conditions(Onsets, M, N, D, TR, dt)
    
    # Covariance matrix
    #R = EM.covariance_matrix(2, D, dt)
    _, R_inv = genGaussianSmoothHRF(False, D, dt, 1., 2)
    #_, R = genGaussianSmoothHRF_cust(False, D, dt, 1., 2)
    R = np.linalg.inv(R_inv)
    # Noise matrix
    Gamma = np.identity(N)
    # Noise initialization
    sigma_eps = np.ones(J)
    #Labels
    logger.info("Labels are initialized by setting active probabilities "
                "to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = copy.deepcopy(q_Z)
    Z_tilde = copy.deepcopy(q_Z)
    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    m_h = m_h[:D]
    H = np.array(m_h).astype(np.float64)

    #prior_balloon = False
    #constraint = False
    #prior_omega = False
    Hb = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
    Gb = create_physio_prf(phy_params, response_dt=dt, response_duration=Thrf)
    H = copy.deepcopy(H)
    H1 = copy.deepcopy(H)
    Ht = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)
    G = copy.deepcopy(H)
    G1 = copy.deepcopy(H)
    Gt = copy.deepcopy(H)
    Sigma_G = copy.deepcopy(Sigma_H)
    normg1 = False
    if prior_omega:
        Omega0 = linear_rf_operator(len(H), phy_params, dt, calculating_brf=False)
        OmegaH = np.dot(Omega0, H)
        Omega = Omega0 
        normg1 = True
        if normg1:
            Omega /= np.linalg.norm(OmegaH)
            OmegaH /=np.linalg.norm(OmegaH)
        G = np.dot(Omega, H)

    # Initialize model parameters 
    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    alpha = np.zeros((J), dtype=np.float64)
    wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
    y_tilde = Y - PL - wa
    # Parameters Gaussian mixtures
    sigma_Ma = np.ones((M, K), dtype=np.float64)
    sigma_Ma[:, 0] = 0.5
    sigma_Ma[:, 1] = 0.6
    print sigma_Ma
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
    m_C1 = copy.deepcopy(m_C)
    m_A1 = copy.deepcopy(m_A)

    if simulation is not None:
        # simulated values
        if not estimateH:
            if dt==simulation['brf'][:, 0].shape[0]:
                H = Ht = simulation['brf'][:, 0]
            sigmaH = sigmaH
        if not estimateG:
            if dt==simulation['prf'][:, 0].shape[0]:
                G = Gt = simulation['prf'][:, 0]
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
            mu_Ma = np.array([[0, 2.2], [0, 2.2], [0, 2.2], [0, 2.2]])
            sigma_Ma = np.array([[.3, .3], [.3, .3], [.3, .3], [.3, .3]])
            mu_Mc = np.array([[0, 1.6], [0, 1.6], [0, 1.6], [0, 1.6]])
            sigma_Mc = np.array([[.3, .3], [.3, .3], [.3, .3], [.3, .3]])
            #mu_Ma = np.array([[0, 15.], [0, 10.], [0, 15.], [0, 10.]])
            #sigma_Ma = np.array([[.2, .1], [.2, .1], [.2, .1], [.2, .1]])
            #mu_Mc = np.array([[0, 14.], [0, 11.], [0, 14.], [0, 11.]])
            #sigma_Mc = np.array([[.21, .11], [.21, .11], [.21, .11], [.21, .11]])


    ###########################################################################
    #############################################             VBJDE

    t1 = time.time()
    ni = 0
    
    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) or (Crit_CG > Thresh) \
            and (ni < NitMax))):
        
        logger.info("-------- Iteration n° " + str(ni + 1) + " --------")

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            EM.plot_response_functions_it(ni, NitMin, M, H, G)
            
        #####################
        # EXPECTATION
        #####################

        # HRF H
        logger.info("E H step ...")
        if estimateH:
            logger.info("estimation")
            print np.linalg.norm(H)
            priorH_cov_term = 0
            matrix_cov = R_inv
            if prior_balloon:
                priorH_mean_term = np.dot(R_inv / sigmaH, Hb)
            else if prior_omega:
                priorH_mean_term = np.dot(np.dot(Omega.T, R_inv / sigmaG), G)
                priorH_cov_term = np.dot(np.dot(Omega.T, R_inv / sigmaG), Omega)    
            else if prior_hierarchical:
                priorH_mean_term = Mu / sigmaH
                matrix_cov = np.eye(R_inv.shape)
            else:
                priorH_mean_term = np.zeros_like(Hb)
            Ht, Sigma_H = EM.expectation_H_prior(Sigma_A, m_A, m_C, G, X, W,
                                                  Gamma, D, J, N, y_tilde,
                                                  sigma_eps, scale, matrix_cov,
                                                  sigmaH, sigmaG, priorH_mean_term,
                                                  priorH_cov_term)
            if constraint:
                logger.info("constraint l2-norm = 1")
                print np.linalg.norm(Ht)
                H = EM.constraint_norm1_b(Ht, Sigma_H)
                Sigma_H = np.zeros_like(Sigma_H)
                #H = Ht / np.linalg.norm(Ht)
            else:
                H = Ht
            if simulation is not None and H.shape==simulation['brf'][:, 0].shape:
                print 'BRF ERROR = ', EM.error(H, simulation['brf'][:, 0])
            h_norm = np.append(h_norm, np.linalg.norm(H))
            print 'h_norm = ', h_norm
            # crit. h
            Crit_H = (np.linalg.norm(H - H1) / np.linalg.norm(H1)) ** 2
            cH += [Crit_H]
            H1 = H
            if prior_omega:
                OmegaH = np.dot(Omega0, H)
                Omega = Omega0 
                if normg1:
                    Omega /= np.linalg.norm(OmegaH)
                    OmegaH /= np.linalg.norm(OmegaH)

        # PRF G
        logger.info("E G step ...")
        if estimateG:
            logger.info("estimation")
            priorG_cov_term = 0
            matrix_cov = R_inv
            if prior_balloon:
                priorG_mean_term = np.dot(R_inv / sigmaG, Gb)
            else if prior_omega:
                priorG_mean_term = np.dot(R_inv / sigmaG, OmegaH)
            else if prior_hierarchical:
                priorG_mean_term = np.dot(Omega, Mu / sigmaG) 
                matrix_cov = np.eye(R_inv.shape)
            else:
                priorG_mean_term = np.zeros_like(Gb)
            Gt, Sigma_G = EM.expectation_G_prior(Sigma_C, m_C, m_A, H, X, W,
                                          Gamma, D, J, N, y_tilde, sigma_eps,
                                          scale, matrix_cov, sigmaG, priorG_mean_term,
                                          priorG_cov_term)
            if constraint and normg1:
                logger.info("constraint l2-norm = 1")
                G = EM.constraint_norm1_b(Gt, Sigma_G, positivity=positivity)
                Sigma_G = np.zeros_like(Sigma_G)
            else:
                G = Gt
            #G = Gt / np.linalg.norm(Gt)
            if simulation is not None and G.shape==simulation['prf'][:, 0].shape:
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
            m_A, Sigma_A = EM.expectation_A_s(H, m_A, G, m_C, W, X, Gamma, q_Z,
                                            mu_Ma, sigma_Ma, D, J, M, K,
                                            y_tilde, Sigma_A, Sigma_H, sigma_eps)
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
            m_C, Sigma_C = EM.expectation_C_s(G, m_C, H, m_A, W, X, Gamma, q_Z,
                                            mu_Mc, sigma_Mc, D, J, M, K,
                                            y_tilde, Sigma_C, Sigma_G, sigma_eps)
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
            q_Z, Z_tilde = EM.expectation_Q(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, graph, M, J, K)
            if simulation is not None:
                print 'LABELS ERROR = ', EM.error(q_Z, Z)

            # crit. Z
            logger.info("crit. Z")
            Crit_Z = (np.linalg.norm((q_Z - q_Z1).flatten()) / \
                         (np.linalg.norm(q_Z1).flatten() + eps)) ** 2
            cZ += [Crit_Z]
            q_Z1 = q_Z

        # crit. AH and CG
        logger.info("crit. AH and CG")
        print 'm_A shape = ', m_A.shape
        print 'm_C shape = ', m_C.shape
        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * H[d]
            CG[:, :, d] = (m_C[:, :] * G[d])

        Crit_AH = (np.linalg.norm(np.reshape(AH - AH1, (M * J * D))) / \
                  (np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
        cAH += [Crit_AH]
        AH1 = AH.copy()
        Crit_CG = (np.linalg.norm(np.reshape(CG - CG1, (M * J * D))) / \
                  (np.linalg.norm(np.reshape(CG1, (M * J * D))) + eps)) ** 2
        cCG += [Crit_CG]
        CG1 = CG.copy()
        logger.info("Crit_AH = " + str(Crit_AH))
        logger.info("Crit_CG = " + str(Crit_CG))


        #####################
        # MAXIMIZATION
        #####################

        # Variances
        # HRF: Sigma_h
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            if prior_balloon:
                Aux = H - Hb
            else:
                Aux = H
            if use_hyperprior:
                logger.info("   ... with prior on sigma_H")
                sigmaH = vt.maximization_sigmaH_prior(D-1, Sigma_H, R_inv, Aux, gamma_h)
            else:
                logger.info("   ... without prior on sigma_H")
                gamma_h = 0
                sigmaH = vt.maximization_sigmaH(D-1, Sigma_H, R_inv, Aux)
            logger.info('sigmaH = ' + str(sigmaH))
        # PRF: Sigma_g
        if estimateSigmaG:
            logger.info("M sigma_G step ...")
            if prior_balloon:
                Aux = G - Gb
            else if prior_omega:
                Aux = G - np.dot(Omega, H)
            else:
                Aux = G
            if use_hyperprior:
                logger.info("   ... with prior on sigma_G")
                sigmaG = vt.maximization_sigmaH_prior(D-1, Sigma_G, R_inv, Aux, gamma_g)
            else:
                logger.info("   ... without prior on sigma_G")
                sigmaG = vt.maximization_sigmaH(D-1, Sigma_G, R_inv, Aux)
            logger.info('sigmaG = ' + str(sigmaG))
        # (mu,sigma)
        if estimateMP:
            logger.info("M (mu,sigma) a and c step ...")
            mu_Ma, sigma_Ma = EM.maximization_mu_sigma(mu_Ma, sigma_Ma,
                                                   q_Z, m_A, K, M, Sigma_A)
            mu_Mc, sigma_Mc = EM.maximization_mu_sigma(mu_Mc, sigma_Mc,
                                                   q_Z, m_C, K, M, Sigma_C)
            logger.info('mu Ma : ' + str(mu_Ma))
            logger.info('sigma Ma : ' + str(sigma_Ma))
            logger.info('mu Mc : ' + str(mu_Mc))
            logger.info('sigma Mc : ' + str(sigma_Mc))
        # Drift L, alpha
        if estimateLA:
            logger.info("M L, alpha step ...")
            L, alpha = EM.maximization_LA(Y, m_A, m_C, X, W, w, H, \
                                          G, L, P, alpha, Gamma, sigma_eps)
            logger.info("  estimation done")
            if simulation is not None:
                print 'ALPHA ERROR = ', EM.error(alpha, np.mean(\
                                            simulation['perf_baseline'], 0))
                print 'DRIFT ERROR = ', EM.error(L, simulation['drift_coeffs'])
            PL = np.dot(P, L)
            wa = np.dot(w[:, np.newaxis], alpha[np.newaxis, :])
            y_tilde = Y - PL - wa

        # Beta
        if estimateBeta:
            MaxItGrad = 200
            gradientStep = 0.003 
            logger.info("M estimating beta")
            for m in xrange(0, M):
                Beta[m] = EM.maximization_beta(Beta[m], q_Z, Z_tilde,
                                        J, K, m, graph, gamma,
                                        neighboursIndexes, maxNeighbours)

            logger.info("End estimating beta")
            logger.info(Beta)

        # Sigma noise
        if estimateNoise:
            logger.info("M sigma noise step ...")
            sigma_eps = EM.maximization_sigma_noise(Y, X, m_A, Sigma_A, H,
                                                    m_C, Sigma_C, G, W, M,
                                                    N, J, y_tilde, sigma_eps)
            if simulation is not None:
                print 'NOISE ERROR = ', EM.error(sigma_eps,
                                             np.var(simulation['noise'], 0))

        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m, 1, :])]
            mua1[m] += [mu_Ma[m, 1]]
            muc1[m] += [mu_Mc[m, 1]]
        
        logger.info("end of iteration")
        ni += 1
        t02 = time.time()
        cTime += [t02 - t1]
        
        jde_fit = EM.computeFit(H, m_A, G, m_C, W, X, J, N)
        r = Y - jde_fit
        rec_error_j = np.sum(r ** 2, 0)
        Y2 = np.sum(Y ** 2, 0)
        rec_error = np.mean(rec_error_j) / np.mean(Y2)
        rerror = np.append(rerror, rec_error)
    
    t2 = time.time()
    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    # Normalize if not done already
    if not constraint:
        Hnorm = np.linalg.norm(H)
        H /= Hnorm
        A *= Hnorm
        Gnorm = np.linalg.norm(G)
        G /= Gnorm
        C *= Gnorm


    ## Compute contrast maps and variance
    if computeContrast:
        CONTRAST_A, CONTRASTVAR_A, \
        CONTRAST_C, CONTRASTVAR_C = EM.compute_contrasts(condition_names, 
                                                         contrasts, m_A, m_C)

    
    ###########################################################################
    ##########################################    PLOTS and SNR computation

    if PLOT:
        EM.plot_convergence(ni, M, cA, cC, cH, cG, cAH, cCG)

    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s",
                str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))

    StimulusInducedSignal = EM.computeFit(H, m_A, G, m_C, W, X, J, N)
    SNR = 20 * (np.log(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - PL - wa))) / np.log(10.)
    logger.info("SNR = %d",  SNR)
    logger.info("ASL signal shape = %d", Y.shape)

    return ni, m_A, H, m_C, G, Z_tilde, sigma_eps, \
           mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, L, PL, \
           alpha, Sigma_A, Sigma_C, Sigma_H, Sigma_G, rerror, \
           CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C, \
           cA[2:], cH[2:], cC[2:], cG[2:], cZ[2:], cAH[2:], cCG[2:]

