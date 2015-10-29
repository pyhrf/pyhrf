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
import pyhrf.vbjde.vem_tools_asl_fast as EMf

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

#@profile
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
                      phy_params=PHY_PARAMS_KHALIDOV11, prior='omega'):

    logger.info("EM for ASL!")
    np.random.seed(6537540)
    logger.info("data shape: ")
    logger.info(Y.shape)
    
    Thresh = 1e-4
    D, M = np.int(np.ceil(Thrf / dt)) + 1, len(Onsets)
    #D, M = np.int(np.ceil(Thrf / dt)), len(Onsets)
    N, J = Y.shape[0], Y.shape[1]
    Crit_AH, Crit_CG, cTime, rerror, FE = 1, 1, [], [], []
    Crit_H, Crit_G, Crit_Z, Crit_A, Crit_C = 1, 1, 1, 1, 1
    cAH, cCG, AH1, CG1 = [], [], [], []
    cA, cC, cH, cG, cZ = [], [], [], [], []
    h_norm, g_norm = [], []
    SUM_q_Z = [[] for m in xrange(M)]
    mua1 = [[] for m in xrange(M)]
    muc1 = [[] for m in xrange(M)]
    AH1, CG1 = np.zeros((J, M, D)), np.zeros((J, M, D))

    # Beta data
    MaxItGrad = 200
    gradientStep = 0.005
    gamma = 7.5
    maxNeighbours, neighboursIndexes = EM.create_neighbours(graph, J)

    # Control-tag
    w = np.ones((N))
    w[idx_first_tag + 1::2] = -1
    W = np.diag(w)
    # Conditions
    X, XX, _ = EM.create_conditions_block(Onsets, durations, M, N, D, TR, dt)
    #X, XX, _ = EM.create_conditions(Onsets, M, N, D, TR, dt)
    
    # Covariance matrix
    #R = EM.covariance_matrix(2, D, dt)
    _, R_inv = genGaussianSmoothHRF(False, D, dt, 1., 2)
    R = np.linalg.inv(R_inv)
    # Noise matrix
    Gamma = np.identity(N)
    # Noise initialization
    sigma_eps = np.ones(J)
    # Labels
    logger.info("Labels are initialized by setting active probabilities "
                "to ones ...")
    #q_Z = np.ones((M, K, J), dtype=np.float64) / 2.
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = copy.deepcopy(q_Z)
    Z_tilde = copy.deepcopy(q_Z)

    # H and G
    TT, m_h = getCanoHRF(Thrf, dt)
    m_h = m_h[:D]
    if prior=='balloon' or 1:
        Hb = create_physio_brf(phy_params, response_dt=dt, response_duration=Thrf)
        Hb /= np.linalg.norm(Hb)
        Gb = create_physio_prf(phy_params, response_dt=dt, response_duration=Thrf)
        Gb /= np.linalg.norm(Gb)
    H = np.array(m_h).astype(np.float64)
    H /= np.linalg.norm(H)
    #H = Hb.copy()
    H1 = copy.deepcopy(H)
    Sigma_H = np.zeros((D, D), dtype=np.float64)
    G = copy.deepcopy(H)
    G1 = copy.deepcopy(G)
    Sigma_G = copy.deepcopy(Sigma_H)
    normOh = False
    normg = True
    if prior=='omega':
        Omega0 = linear_rf_operator(len(H), phy_params, dt, calculating_brf=False)
        OmegaH = np.dot(Omega0, H)
        Omega = Omega0.copy()
        G = np.dot(Omega, H)
        if normOh or normg:
            Omega /= np.linalg.norm(OmegaH)
            OmegaH /=np.linalg.norm(OmegaH)
            G /= np.linalg.norm(G)

    # Initialize model parameters 
    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    alpha = np.zeros((J), dtype=np.float64)
    WP = np.append(w[:, np.newaxis], P, axis=1)
    AL = np.append(alpha[np.newaxis, :], L, axis=0)
    y_tilde = Y - WP.dot(AL)

    # Parameters Gaussian mixtures
    mu_Ma = np.append(np.zeros((M, 1)), np.ones((M, 1)), axis=1).astype(np.float64)
    mu_Mc = mu_Ma.copy()
    sigma_Ma = np.ones((M, K), dtype=np.float64) * 0.5
    sigma_Mc = sigma_Ma.copy()

    # Params RLs
    m_A = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        m_A[j, :] = (np.random.normal(mu_Ma, np.sqrt(sigma_Ma)) * q_Z[:, :, j]).sum(axis=1).T
    m_A1 = m_A.copy()
    Sigma_A = np.ones((M, M, J)) * np.identity(M)[:, :, np.newaxis]
    m_C = m_A.copy()
    m_C1 = m_C.copy()
    Sigma_C = Sigma_A.copy()

    # Precomputations
    WX = W.dot(XX).transpose(1, 0, 2)
    Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    X_Gamma_X = np.tensordot(XX.T, Gamma_X, axes=(1, 0))    # shape (D, M, M, D)
    Gamma_WX = np.tensordot(Gamma, WX, axes=(1, 1))
    XW_Gamma_WX = np.tensordot(WX.T, Gamma_WX, axes=(1, 0))    # shape (D, M, M, D)
    Gamma_WP = Gamma.dot(WP)
    WP_Gamma_WP = WP.T.dot(Gamma_WP)
    sigma_eps_m = np.maximum(sigma_eps, eps)
    cov_noise = sigma_eps_m[:, np.newaxis, np.newaxis]

    ###########################################################################
    #############################################             VBJDE

    t1 = time.time()
    ni = 0
    
    #while ((ni < NitMin + 1) or (((Crit_AH > Thresh) or (Crit_CG > Thresh)) \
    #        and (ni < NitMax))):
    while ((ni < NitMin + 1) or (((Crit_AH > Thresh)) \
            and (ni < NitMax))):

        logger.info("-------- Iteration n° " + str(ni + 1) + " --------")

        if PLOT and ni >= 0:  # Plotting HRF and PRF
            logger.info("Plotting HRF and PRF for current iteration")
            EM.plot_response_functions_it(ni, NitMin, M, H, G)


        #####################
        # EXPECTATION
        #####################


        # Managing types of prior
        priorH_cov_term = np.zeros_like(R_inv)
        priorG_cov_term = 0
        matrix_cov = R_inv
        if prior=='balloon':
            logger.info("   prior balloon")
            priorH_mean_term = np.dot(R_inv / sigmaH, Hb)
            priorG_mean_term = np.dot(R_inv / sigmaG, Gb)
        elif prior=='omega':
            logger.info("   prior omega")
            priorH_mean_term = np.dot(np.dot(Omega.T, R_inv / sigmaG), G)
            priorH_cov_term = np.dot(np.dot(Omega.T, R_inv / sigmaG), Omega)
            priorG_mean_term = np.dot(R_inv / sigmaG, OmegaH)
        elif prior=='hierarchical':
            logger.info("   prior hierarchical")
            priorH_mean_term = Mu / sigmaH
            priorG_mean_term = np.dot(Omega, Mu / sigmaG)
            matrix_cov = np.eye(R_inv.shape)
        else:
            logger.info("   NO prior")
            priorH_mean_term = np.zeros_like(H)
            priorG_mean_term = np.zeros_like(G)


        # HRF H
        if estimateH:
            logger.info("E H step ...")
            
            Ht, Sigma_H = EMf.expectation_H(Sigma_A, m_A, m_C, G, XX, W, Gamma,
                                            Gamma_X, X_Gamma_X, J, y_tilde,
                                            cov_noise, matrix_cov, sigmaH, 
                                            priorH_mean_term, priorH_cov_term)
            """
            Ht, Sigma_H = EM.expectation_H_physio(Sigma_A, m_A, m_C, G, X, W, Gamma,
                                          D, J, N, y_tilde, sigma_eps, scale, R_inv,
                                          sigmaH, sigmaG, Omega)
            
            """
            if constraint: 
                if not np.linalg.norm(Ht)==1:
                    logger.info("   constraint l2-norm = 1")
                    H = EM.constraint_norm1_b(Ht, Sigma_H)
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
            if prior=='omega':
                OmegaH = np.dot(Omega0, H)
                Omega = Omega0 
                if normOh:
                    Omega /= np.linalg.norm(OmegaH)
                    OmegaH /= np.linalg.norm(OmegaH)

        # PRF G
        if estimateG:
            logger.info("E G step ...")
            
            Gt, Sigma_G = EMf.expectation_G(Sigma_C, m_C, m_A, H, XX, W, WX, Gamma, Gamma_WX,
                                            XW_Gamma_WX, J, y_tilde, cov_noise, matrix_cov, sigmaG,
                                            priorG_mean_term, priorG_cov_term)
            """
            Gt, Sigma_G = EM.expectation_G_physio(Sigma_C, m_C, m_A, H, X, W,
                                          Gamma, D, J, N, y_tilde, sigma_eps,
                                          scale, R_inv, sigmaG, OmegaH)
            """
            if constraint and normg:
                if not np.linalg.norm(Gt)==1:
                    logger.info("   constraint l2-norm = 1")
                    G = EM.constraint_norm1_b(Gt, Sigma_G, positivity=positivity)
                    #G = Gt / np.linalg.norm(Gt)
                else:
                    logger.info("   l2-norm already 1!!!!!")
                    G = Gt.copy()
                Sigma_G = np.zeros_like(Sigma_G)
            else:
                G = Gt.copy()
                g_norm = np.append(g_norm, np.linalg.norm(G))
                print 'g_norm = ', g_norm
            cG += [(np.linalg.norm(G - G1) / np.linalg.norm(G1)) ** 2]
            G1[:] = G[:]

        # A
        if estimateA:
            logger.info("E A step ...")
            m_A, Sigma_A = EMf.expectation_A(H, G, m_C, W, XX, Gamma, Gamma_X, q_Z,
                                             mu_Ma, sigma_Ma, J, y_tilde,
                                             Sigma_H, sigma_eps_m)

            cA += [(np.linalg.norm(m_A - m_A1) / np.linalg.norm(m_A1)) ** 2]
            m_A1[:, :] = m_A[:, :]

        # C
        if estimateC:
            logger.info("E C step ...")
            
            m_C, Sigma_C = EMf.expectation_C(G, H, m_A, W, XX, Gamma, Gamma_X, q_Z,
                                             mu_Mc, sigma_Mc, J, y_tilde,
                                             Sigma_G, sigma_eps_m)
            """
            m_C, Sigma_C = EM.expectation_C(G, m_C, H, m_A, W, X, Gamma, q_Z,
                                            mu_Mc, sigma_Mc, D, J, M, K,
                                            y_tilde, Sigma_C, sigma_eps)
            """
            cC += [(np.linalg.norm(m_C - m_C1) / np.linalg.norm(m_C1)) ** 2]
            m_C1[:, :] = m_C[:, :]

        # Q labels
        if estimateZ:
            logger.info("E Q step ...")
            
            q_Z, Z_tilde = EMf.expectation_Q(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, neighboursIndexes, graph, M, J, K)    
            """
            q_Z, Z_tilde = EM.expectation_Q(Sigma_A, m_A, Sigma_C, m_C,
                                            sigma_Ma, mu_Ma, sigma_Mc, mu_Mc,
                                            Beta, Z_tilde, q_Z, graph, M, J, K)        
            """
            cZ += [(np.linalg.norm(q_Z - q_Z1) / (np.linalg.norm(q_Z1) + eps)) ** 2]
            q_Z1 = q_Z
        
        


        # crit. AH and CG
        logger.info("crit. AH and CG")
        AH = m_A[:, :, np.newaxis] * H[np.newaxis, np.newaxis, :]
        CG = m_C[:, :, np.newaxis] * G[np.newaxis, np.newaxis, :]
        
        Crit_AH = (np.linalg.norm(AH - AH1) / (np.linalg.norm(AH1) + eps)) ** 2
        cAH += [Crit_AH]
        AH1 = AH.copy()
        Crit_CG = (np.linalg.norm(CG - CG1) / (np.linalg.norm(CG1) + eps)) ** 2
        cCG += [Crit_CG]
        CG1 = CG.copy()
        logger.info("Crit_AH = " + str(Crit_AH))
        logger.info("Crit_CG = " + str(Crit_CG))


        #####################
        # MAXIMIZATION
        #####################

        if prior=='balloon':
            AuxH = H - Hb
            AuxG = G - Gb
        elif prior=='omega':
            logger.info("   prior omega")
            AuxH = H.copy()
            AuxG = G - np.dot(Omega, H) #/np.linalg.norm(np.dot(Omega, H))
        else:
            logger.info("   prior NOT balloon NOR omega")
            AuxH = H.copy()
            AuxG = G.copy()

        # Variance HRF: sigmaH
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            sigmaH = EMf.maximization_sigma(D-1, Sigma_H, R_inv, AuxH, use_hyperprior, gamma_h)
            logger.info('sigmaH = ' + str(sigmaH))

        # Variance PRF: sigmaG
        if estimateSigmaG:
            logger.info("M sigma_G step ...")
            sigmaG = EMf.maximization_sigma(D-1, Sigma_G, R_inv, AuxG, use_hyperprior, gamma_g)
            logger.info('sigmaG = ' + str(sigmaG))

        # (mu,sigma)
        if estimateMP:
            logger.info("M (mu,sigma) a and c step ...")
            mu_Ma, sigma_Ma = EMf.maximization_mu_sigma(mu_Ma, q_Z, m_A, Sigma_A)
            mu_Mc, sigma_Mc = EMf.maximization_mu_sigma(mu_Mc, q_Z, m_C, Sigma_C)
        
        # Drift L, alpha
        if estimateLA:
            logger.info("M L, alpha step ...")
            
            AL = EMf.maximization_LA(Y, m_A, m_C, XX, WP, W, WP_Gamma_WP, H, G, Gamma)
            """
            AL = EM.maximization_LA(Y, m_A, m_C, X, W, w, H, \
                                          G, AL[1:, :], P, AL[0, :], Gamma, sigma_eps)
            """
            y_tilde = Y - WP.dot(AL)

        # Beta
        if estimateBeta:
            logger.info("M beta step ...")
            
            Qtilde = np.concatenate((Z_tilde, np.zeros((M, K, 1), dtype=Z_tilde.dtype)), axis=2)
            Qtilde_sumneighbour = Qtilde[:, :, neighboursIndexes].sum(axis=3)
            for m in xrange(0, M):
                Beta[m] = EMf.maximization_beta_m4(Beta[m].copy(), q_Z[m, :, :], Qtilde_sumneighbour[m, :, :],
                                                   Qtilde[m, :, :], neighboursIndexes, maxNeighbours, 
                                                   gamma, MaxItGrad, gradientStep)
            """
            logger.info("M estimating beta")
            for m in xrange(0, M):
                Beta[m] = EM.maximization_beta(Beta[m], q_Z, Z_tilde,
                                        J, K, m, graph, gamma,
                                        neighboursIndexes, maxNeighbours)
            """
        # Sigma noise
        if estimateNoise:
            logger.info("M sigma noise step ...")
            
            sigma_eps = EMf.maximization_sigma_noise(XX, m_A, Sigma_A, H, m_C, Sigma_C, \
                                                    G, Sigma_H, Sigma_G, W, y_tilde, Gamma, \
                                                    Gamma_X, Gamma_WX, N)
            """
            sigma_eps = EM.maximization_sigma_noise(Y, X, m_A, Sigma_A, H,
                          m_C, Sigma_C, G, W, M, N, J, y_tilde, sigma_eps)
            """
        if PLOT:
            for m in xrange(M):
                SUM_q_Z[m] += [q_Z[m, 1, :].sum()]
                mua1[m] += [mu_Ma[m, 1]]
                muc1[m] += [mu_Mc[m, 1]]
        
        """
        free_energy = EMf.Compute_FreeEnergy(y_tilde, m_A, Sigma_A, mu_Ma, sigma_Ma,
                                             H, Sigma_H, R, R_inv, sigmaH, sigmaG,
                                             m_C, Sigma_C, mu_Mc, sigma_Mc, G, Sigma_G,
                                             q_Z, neighboursIndexes, Beta, Gamma,
                                             sigma_eps, XX, W, J, D, M, N, K)
        FE += [free_energy]
        """
        ni += 1
        cTime += [time.time() - t1]
        
        logger.info("Computing reconstruction error")
        StimulusInducedSignal = EMf.computeFit(H, m_A, G, m_C, W, XX)
        rerror = np.append(rerror, \
                           np.mean(((Y - StimulusInducedSignal) ** 2).sum(axis=0)) \
                           / np.mean((Y ** 2).sum(axis=0)))
        
    CompTime = time.time() - t1


    # Normalize if not done already
    if not constraint or not normg:
        logger.info("l2-norm of H and G to 1 if not constraint")
        Hnorm = np.linalg.norm(H)
        H /= Hnorm
        m_A *= Hnorm
        Gnorm = np.linalg.norm(G)
        G /= Gnorm
        m_C *= Gnorm


    ## Compute contrast maps and variance
    if computeContrast and len(contrasts) > 0:
        logger.info("Computing contrasts ... ")
        CONTRAST_A, CONTRASTVAR_A, \
        CONTRAST_C, CONTRASTVAR_C = EM.compute_contrasts(condition_names, 
                                                         contrasts, m_A, m_C)
    else:
        CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C = 0, 0, 0, 0


    ###########################################################################
    ##########################################    PLOTS and SNR computation

    if PLOT:
        logger.info("plotting...")
        print 'FE = ', FE
        EM.plot_convergence(ni, M, cA, cC, cH, cG, cAH, cCG, SUM_q_Z, mua1, muc1, FE)

    logger.info("Nb iterations to reach criterion: %d",  ni)
    logger.info("Computational time = %s min %s s",
                str(np.int(CompTime // 60)), str(np.int(CompTime % 60)))
    logger.info("Iteration time = %s min %s s",
                str(np.int((CompTime // ni) // 60)), str(np.int((CompTime / ni) % 60)))

    SNR = 20 * (np.log(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - WP.dot(AL)))) / np.log(10.)
    logger.info("SNR = %d",  SNR)

    SNR10 = 20 * (np.log10(np.linalg.norm(Y) / \
                np.linalg.norm(Y - StimulusInducedSignal - WP.dot(AL))))
    logger.info("SNR10 = %d",  SNR10)

    return ni, m_A, H, m_C, G, Z_tilde, sigma_eps, \
           mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, AL[1:, :], np.dot(P, AL[1:, :]), \
           AL[0, :], Sigma_A, Sigma_C, Sigma_H, Sigma_G, rerror, \
           CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C, \
           cA[2:], cH[2:], cC[2:], cG[2:], cZ[2:], cAH[2:], cCG[2:], cTime

