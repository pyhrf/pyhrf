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
                      positivity=False, phy_params=PHY_PARAMS_KHALIDOV11,
                      prior_balloon=False, prior_omega=False,
                      prior_hierarchical=False):
    """ Version modified by Lofti from Christine's version """
    logger.info("EM for ASL!")
    np.random.seed(6537546)
    #NitMax=NitMin=0
    logger.info("data shape: ")
    logger.info(Y.shape)
        
    # Initialization
    gamma_h = 50000  # 10000000000  # 7.5
    gamma_g = 50000  # 7.5
    
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
    
    block = True
    # Conditions
    if block:
        X, XX, condition_names = EM.create_conditions_block(Onsets, durations, M, N, D, TR, dt)
    else:
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

    prior_balloon = True
    constraint = False

    if prior_balloon:
        Hb = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
        Gb = create_physio_prf(phy_params, response_dt=dt, response_duration=Thrf)
    else:
        Hb = np.zeros_like(H)
        Gb = np.zeros_like(H)
    H = copy.deepcopy(H)
    H1 = copy.deepcopy(H)
    Ht = copy.deepcopy(H)
    print 'H.shape = ', H.shape
    print 'Hb.shape = ', Hb.shape
    Sigma_H = np.zeros((D, D), dtype=np.float64)
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
    
    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) and (Crit_CG > Thresh) \
            and (ni < NitMax))):
        
        logger.info("-------- Iteration n° " + str(ni + 1) + " --------")

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            logger.info("Plotting HRF and PRF for current iteration")
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            jet = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=0, vmax=NitMin + 1)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            if ni == 0:
                plt.close('all')
            colorVal = scalarMap.to_rgba(ni)
            plt.figure(M + 1)
            plt.plot(H, color=colorVal)
            plt.hold(True)
            plt.figure(M + 2)
            plt.plot(G, color=colorVal)
            plt.hold(True)

        #####################
        # EXPECTATION
        #####################

        # HRF H
        logger.info("E H step ...")
        if estimateH:
            logger.info("estimation")
            print np.linalg.norm(H)
            Ht, Sigma_H = EM.expectation_H_balloon(Sigma_A, m_A, m_C, G, X, W,
                                                  Gamma, D, J, N, y_tilde,
                                                  sigma_eps, scale, R_inv,
                                                  sigmaH, sigmaG, Hb)
            if constraint:
                logger.info("constraint l2-norm = 1")
                print np.linalg.norm(Ht)
                H = EM.constraint_norm1_b(Ht, Sigma_H)
                Sigma_H = np.zeros_like(Sigma_H)
                #H = Ht / np.linalg.norm(Ht)
            else:
                H = Ht
            if 0:
                print H
                print simulation['brf'][:, 0]
                print H.shape
                print simulation['brf'][:, 0].shape
            if simulation is not None and H.shape==simulation['brf'][:, 0].shape:
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
            Gt, Sigma_G = EM.expectation_G_balloon(Sigma_C, m_C, m_A, H, X, W,
                                          Gamma, D, J, N, y_tilde, sigma_eps,
                                          scale, R_inv, sigmaG, Gb)
            if constraint:
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
        if 0: 
            import matplotlib.pyplot as plt
            #plt.matshow(AH[:, 0, 5].reshape(20, 20))
            #plt.show()
            #plt.matshow(AH[:, 1, 5].reshape(20, 20))
            #plt.show()
            plt.figure()
            plt.plot(AH[0, 0, :])
            plt.plot(AH[0, 1, :])
            plt.show()
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
            if use_hyperprior:
                logger.info("   ... with prior on sigma_H")
                sigmaH = vt.maximization_sigmaH_prior(D-1, Sigma_H, R_inv, H - Hb, gamma_h)
                # maximization_sigmaH_prior(D, Sigma_H, R, m_H, gamma_h)
                #sigmaH = EM.maximization_sigma_prior(D-1, R_inv, H - Hb, gamma_h)
            else:
                logger.info("   ... without prior on sigma_H")
                gamma_h = 0
                sigmaH = vt.maximization_sigmaH(D-1, Sigma_H, R_inv, H - Hb)
                # maximization_sigmaH(D, Sigma_H, R, m_H)
                #sigmaH = EM.maximization_sigma(D-1, R_inv, H - Hb)
            logger.info('sigmaH = ' + str(sigmaH))
        # PRF: Sigma_g
        if estimateSigmaG:
            logger.info("M sigma_G step ...")
            if use_hyperprior:
                logger.info("   ... with prior on sigma_G")
                #sigmaG = EM.maximization_sigma_prior(D-1, R_inv, G - Gb, gamma_g)
                sigmaG = vt.maximization_sigmaH_prior(D-1, Sigma_G, R_inv, G - Gb, gamma_g)
            else:
                logger.info("   ... without prior on sigma_G")
                #sigmaG = EM.maximization_sigma(D-1, R_inv, G - Gb)
                sigmaG = vt.maximization_sigmaH(D-1, Sigma_G, R_inv, G - Gb)
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
            #L, alpha = EM.maximization_L_alpha(Y, m_A, m_C, X, W, w, H, \
            #                              G, L, P, alpha, Gamma, sigma_eps)
            logger.info("  estimation done")
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
            MaxItGrad = 200
            gradientStep = 0.003 
            logger.info("M estimating beta")
            for m in xrange(0, M):
                Beta[m] = EM.maximization_beta(Beta[m], q_Z, Z_tilde,
                                        J, K, m, graph, gamma,
                                        neighboursIndexes, maxNeighbours)
            #for m in xrange(0, M):
            #    Beta[m] = UtilsC.maximization_beta(Beta[m], q_Z[m, :, :].astype(np.float64),
            #                                       Z_tilde[m, :, :].astype(np.float64), J, K,
            #                                       neighboursIndexes.astype(np.int32), gamma,
            #                                       maxNeighbours, MaxItGrad, gradientStep)
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


    #+++++++++++++++++++++++ calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        # CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C = EM.compute_contrasts(condition_names, contrasts, m_A, m_C)
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            brls_conds = dict([(str(cn), m_A[:, ic])
                              for ic, cn in enumerate(condition_names)])
            prls_conds = dict([(str(cn), m_C[:, ic])
                              for ic, cn in enumerate(condition_names)])
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **brls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST_A[:, n] = contrast
                contrast_expr = AExpr(contrasts[cname], **prls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST_C[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(M, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, M):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, M):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(M, dtype=float)
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, J):
                    Sa_tmp = Sigma_A[:, :, j]
                    CONTRASTVAR_A[j, n] = np.dot(np.dot(AC, Sa_tmp), AC)
                    Sc_tmp = Sigma_C[:, :, j]
                    CONTRASTVAR_C[j, n] = np.dot(np.dot(AC, Sc_tmp), AC)
                #------------ variance -------------#
                n += 1
                logger.info('Done contrasts computing.')
    #+++++++++++++++++++++++ calculate contrast maps and variance ++++++++++++++++++roc_curve+++++#

    ###########################################################################
    ##########################################    PLOTS and SNR computation

    logger.info("prepare to plot")

    SUM_p_Q_array = np.zeros((M, ni), dtype=np.float64)
    mua1_array = np.zeros((M, ni), dtype=np.float64)
    muc1_array = np.zeros((M, ni), dtype=np.float64)
    #h_norm_array = np.zeros((ni), dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_p_Q_array[m, i] = SUM_q_Z[m][i]
            mua1_array[m, i] = mua1[m][i]
            muc1_array[m, i] = muc1[m][i]
            #h_norm_array[i] = h_norm[i]
    logger.info("plots")

    if PLOT:
        font = {'size': 15}
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rc('font', **font)
        label = '_vh' + str(sigmaH) + '_vg' + str(sigmaG)
        if not op.exists('./plots'):
            os.makedirs('./plots')
        plt.figure(M + 1)
        plt.savefig('./plots/BRF_Iter_ASL' + label + '.png')
        plt.figure(M + 2)
        plt.savefig('./plots/PRF_Iter_ASL' + label + '.png')
        plt.hold(False)
        plt.figure(M + 4)
        plt.plot(cAH[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cCG[1:-1], 'm')
        plt.hold(False)
        plt.legend(('CAH', 'CCG'))
        plt.grid(True)
        plt.savefig('./plots/Crit_ASL.png')
        plt.figure(M + 5)
        plt.plot(cA[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cC[1:-1], 'm')
        plt.plot(cH[1:-1], 'green')
        plt.plot(cG[1:-1], 'red')
        plt.hold(False)
        plt.legend(('CA', 'CC', 'CH', 'CG'))
        plt.grid(True)
        plt.savefig('./plots/Crit_all.png')
        plt.figure(M + 6)
        for m in xrange(M):
            plt.plot(SUM_p_Q_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./plots/Sum_p_Q_Iter_ASL.png')
        """
        plt.figure(M + 7)
        for m in xrange(M):
            plt.plot(mua1_array[m])
            plt.hold(True)
            plt.plot(muc1_array[m])
        plt.hold(False)
        plt.legend(('mu_a', 'mu_c'))
        plt.savefig('./info/mu1_Iter_ASL.png')
        """

    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s",
                str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))

    StimulusInducedSignal = EM.computeFit(H, m_A, G, m_C, W, X, J, N)
    SNR = 20 * (np.log(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - PL - wa))) / np.log(10.)
    print 'SNR = ', SNR
    print 'ASL signal shape = ', Y.shape

    #logger.info('mu_Ma: %f', mu_Ma)
    #logger.info('sigma_Ma: %f', sigma_Ma)
    #logger.info("sigma_H = %s" + str(sigmaH))
    #logger.info('mu_Mc: %f', mu_Mc)
    #logger.info('sigma_Mc: %f', sigma_Mc)
    #logger.info("sigma_G = %s" + str(sigmaG))
    #logger.info("Beta = %s" + str(Beta))
    #logger.info('SNR comp = %f', SNR)

    return ni, m_A, H, m_C, G, Z_tilde, sigma_eps, \
           mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, L, PL, \
           alpha, Sigma_A, Sigma_C, Sigma_H, Sigma_G, rerror, \
           CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C, \
           cA[2:], cH[2:], cC[2:], cG[2:], cZ[2:], cAH[2:], cCG[2:]

