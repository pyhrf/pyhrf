# -*- coding: utf-8 -*-
"""VEM BOLD Constrained

File that contains function for BOLD data analysis with positivity
and l2-norm=1 constraints.

It imports functions from vem_tools.py in pyhrf/vbjde
"""

import time
import logging

import numpy as np

import pyhrf.vbjde.UtilsC as UtilsC
import pyhrf.vbjde.vem_tools as vt

from pyhrf.tools._io import read_volume
from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.ndarray import xndarray
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


logger = logging.getLogger(__name__)


def Main_vbjde_Extension_constrained(graph, Y, Onsets, Thrf, K, TR, beta,
                                     dt, scale=1, estimateSigmaH=True,
                                     sigmaH=0.05, NitMax=-1,
                                     NitMin=1, estimateBeta=True,
                                     PLOT=False, contrasts=[],
                                     computeContrast=False,
                                     gamma_h=0, estimateHRF=True,
                                     TrueHrfFlag=False,
                                     HrfFilename='hrf.nii',
                                     estimateLabels=True,
                                     LabelsFilename='labels.nii',
                                     MFapprox=False, InitVar=0.5,
                                     InitMean=2.0, MiniVEMFlag=False,
                                     NbItMiniVem=5):
    # VBJDE Function for BOLD with contraints

    logger.info("Fast EM with C extension started ...")
    np.random.seed(6537546)

    ##########################################################################
    # INITIALIZATIONS
    # Initialize parameters
    tau1 = 0.0
    tau2 = 0.0
    S = 100
    Init_sigmaH = sigmaH
    Nb2Norm = 1
    NormFlag = False
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    #gamma_h = 1000
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5
    estimateLabels = True  # WARNING!! They should be estimated

    # Initialize sizes vectors
    D = int(np.ceil(Thrf / dt)) + 1  # D = int(np.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(np.sqrt(J))
    condition_names = []

    # Neighbours
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]
    # Conditions
    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    # Covariance matrix
    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    invR = np.linalg.inv(R)
    Det_invR = np.linalg.det(invR)

    Gamma = np.identity(N)
    Det_Gamma = np.linalg.det(Gamma)

    p_Wtilde = np.zeros((M, K), dtype=np.float64)
    p_Wtilde1 = np.zeros((M, K), dtype=np.float64)
    p_Wtilde[:, 1] = 1

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    Crit_FreeEnergy = 1

    cA = []
    cH = []
    cZ = []
    cAH = []
    FreeEnergy_Iter = []
    cTime = []
    cFE = []

    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]
    h_norm = []
    h_norm2 = []

    CONTRAST = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((J, len(contrasts)), dtype=np.float64)
    Q_barnCond = np.zeros((M, M, D, D), dtype=np.float64)
    XGamma = np.zeros((M, D, N), dtype=np.float64)
    m1 = 0
    for k1 in X:  # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1, m2, :, :] = np.dot(
                np.dot(X[k1].transpose(), Gamma), X[k2])
            m2 += 1
        XGamma[m1, :, :] = np.dot(X[k1].transpose(), Gamma)
        m1 += 1

    if MiniVEMFlag:
        logger.info("MiniVEM to choose the best initialisation...")
        """InitVar, InitMean, gamma_h = MiniVEM_CompMod(Thrf,TR,dt,beta,Y,K,
                                                     gamma,gradientStep,
                                                     MaxItGrad,D,M,N,J,S,
                                                     maxNeighbours,
                                                     neighboursIndexes,
                                                     XX,X,R,Det_invR,Gamma,
                                                     Det_Gamma,
                                                     scale,Q_barnCond,XGamma,
                                                     NbItMiniVem,
                                                     sigmaH,estimateHRF)"""

        InitVar, InitMean, gamma_h = vt.MiniVEM_CompMod(Thrf, TR, dt, beta, Y, K, gamma, gradientStep, MaxItGrad, D, M, N, J, S, maxNeighbours,
                                                        neighboursIndexes, XX, X, R, Det_invR, Gamma, Det_Gamma, p_Wtilde, scale, Q_barnCond, XGamma, tau1, tau2, NbItMiniVem, sigmaH, estimateHRF)

    sigmaH = Init_sigmaH
    sigma_epsilone = np.ones(J)
    logger.info(
        "Labels are initialized by setting active probabilities to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = np.zeros((M, K, J), dtype=np.float64)
    Z_tilde = q_Z.copy()

    # TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT, m_h = getCanoHRF(Thrf, dt)  # TODO: check
    m_h = m_h[:D]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    sigmaH1 = sigmaH
    if estimateHRF:
        Sigma_H = np.ones((D, D), dtype=np.float64)
    else:
        Sigma_H = np.zeros((D, D), dtype=np.float64)

    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    y_tilde = Y - PL
    Ndrift = L.shape[0]

    sigma_M = np.ones((M, K), dtype=np.float64)
    sigma_M[:, 0] = 0.5
    sigma_M[:, 1] = 0.6
    mu_M = np.zeros((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_M[:, k] = InitMean
    Sigma_A = np.zeros((M, M, J), np.float64)
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
    m_A = np.zeros((J, M), dtype=np.float64)
    m_A1 = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(mu_M[m, k],
                                              np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]
    m_A1 = m_A

    t1 = time.time()

    ##########################################################################
    # VBJDE num. iter. minimum

    ni = 0

    while ((ni < NitMin) or (((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax))):

        logger.info("------------------------------ Iteration n° " +
                    str(ni + 1) + " ------------------------------")

        #####################
        # EXPECTATION
        #####################

        # A
        logger.info("E A step ...")
        UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                             Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N, K)
        val = np.reshape(m_A, (M * J))
        val[np.where((val <= 1e-50) & (val > 0.0))] = 0.0
        val[np.where((val >= -1e-50) & (val < 0.0))] = 0.0

        # crit. A
        DIFF = np.reshape(m_A - m_A1, (M * J))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        Crit_A = (
            np.linalg.norm(DIFF) / np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        # HRF h
        if estimateHRF:
            ################################
            #  HRF ESTIMATION
            ################################
            UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R, Sigma_H, Y,
                                 y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N, scale, sigmaH)

            import cvxpy as cvx
            m, n = Sigma_H.shape
            Sigma_H_inv = np.linalg.inv(Sigma_H)
            zeros_H = np.zeros_like(m_H[:, np.newaxis])

            # Construct the problem. PRIMAL
            h = cvx.Variable(n)
            expression = cvx.quad_form(h - m_H[:, np.newaxis], Sigma_H_inv)
            objective = cvx.Minimize(expression)
            #constraints = [h[0] == 0, h[-1]==0, h >= zeros_H, cvx.square(cvx.norm(h,2))<=1]
            constraints = [
                h[0] == 0, h[-1] == 0, cvx.square(cvx.norm(h, 2)) <= 1]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(verbose=0, solver=cvx.CVXOPT)

            # Now we update the mean of h
            m_H_old = m_H
            Sigma_H_old = Sigma_H
            m_H = np.squeeze(np.array((h.value)))
            Sigma_H = np.zeros_like(Sigma_H)

            h_norm += [np.linalg.norm(m_H)]
            # print 'h_norm = ', h_norm

            # Plotting HRF
            if PLOT and ni >= 0:
                import matplotlib.pyplot as plt
                plt.figure(M + 1)
                plt.plot(m_H)
                plt.hold(True)
        else:
            if TrueHrfFlag:
                #TrueVal, head = read_volume(HrfFilename)
                TrueVal, head = read_volume(HrfFilename)[:, 0, 0, 0]
                print TrueVal
                print TrueVal.shape
                m_H = TrueVal

        # crit. h
        Crit_H = (np.linalg.norm(m_H - m_H1) / np.linalg.norm(m_H1)) ** 2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        # crit. AH
        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * m_H[d]
        DIFF = np.reshape(AH - AH1, (M * J * D))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        if np.linalg.norm(np.reshape(AH1, (M * J * D))) == 0:
            Crit_AH = 1000000000.
        else:
            Crit_AH = (
                np.linalg.norm(DIFF) / np.linalg.norm(np.reshape(AH1, (M * J * D)))) ** 2
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        # Z labels
        if estimateLabels:
            logger.info("E Z step ...")
            # WARNING!!! ParsiMod gives better results, but we need the other
            # one.
            if MFapprox:
                UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z, neighboursIndexes.astype(
                    np.int32), M, J, K, maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_RVM_and_CompMod(
                    Sigma_A, m_A, sigma_M, Beta, mu_M, q_Z, neighboursIndexes.astype(np.int32), M, J, K, maxNeighbours)
        else:
            logger.info("Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m, 1, :] = np.reshape(TrueZ[0][:, :, :, m], J)
                q_Z[m, 0, :] = 1 - q_Z[m, 1, :]

        # crit. Z
        val = np.reshape(q_Z, (M * K * J))
        val[np.where((val <= 1e-50) & (val > 0.0))] = 0.0

        DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
        # To avoid numerical problems
        DIFF[np.where((DIFF < 1e-50) & (DIFF > 0.0))] = 0.0
        # To avoid numerical problems
        DIFF[np.where((DIFF > -1e-50) & (DIFF < 0.0))] = 0.0
        if np.linalg.norm(np.reshape(q_Z1, (M * K * J))) == 0:
            Crit_Z = 1000000000.
        else:
            Crit_Z = (
                np.linalg.norm(DIFF) / np.linalg.norm(np.reshape(q_Z1, (M * K * J)))) ** 2
        cZ += [Crit_Z]
        q_Z1 = q_Z

        #####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateHRF:
            if estimateSigmaH:
                logger.info("M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = vt.maximization_sigmaH_prior(
                        D, Sigma_H_old, R, m_H_old, gamma_h)
                else:
                    sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)
                logger.info('sigmaH = %s', str(sigmaH))

        # (mu,sigma)
        logger.info("M (mu,sigma) step ...")
        mu_M, sigma_M = vt.maximization_mu_sigma(
            mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)
        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m, 1, :])]
            mu1[m] += [mu_M[m, 1]]

        # Drift L
        UtilsC.maximization_L(
            Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M, Ndrift, N)
        PL = np.dot(P, L)
        y_tilde = Y - PL

        # Beta
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta, q_Z[m, :, :].astype(np.float64), Z_tilde[m, :, :].astype(
                        np.float64), J, K, neighboursIndexes.astype(np.int32), gamma, maxNeighbours, MaxItGrad, gradientStep)
                if not MFapprox:
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(np.float64),q_Z[m,:,:].astype(np.float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta, q_Z[m, :, :].astype(
                        np.float64), J, K, neighboursIndexes.astype(np.int32), gamma, maxNeighbours, MaxItGrad, gradientStep)
            logger.info("End estimating beta")
            logger.info(Beta)

        # Sigma noise
        logger.info("M sigma noise step ...")
        UtilsC.maximization_sigma_noise(
            Gamma, PL, sigma_epsilone, Sigma_H, Y, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy

        """FreeEnergy = vt.Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,
                                           m_H,Sigma_H,R,Det_invR,sigmaH,
                                           p_Wtilde,q_Z,neighboursIndexes,
                                           maxNeighbours,Beta,sigma_epsilone,
                                           XX,Gamma,Det_Gamma,XGamma,J,D,M,
                                           N,K,S,"CompMod")"""
        FreeEnergy = vt.Compute_FreeEnergy(y_tilde, m_A, Sigma_A, mu_M, sigma_M, m_H, Sigma_H, R, Det_invR, sigmaH, p_Wtilde, tau1,
                                           tau2, q_Z, neighboursIndexes, maxNeighbours, Beta, sigma_epsilone, XX, Gamma, Det_Gamma, XGamma, J, D, M, N, K, S, "CompMod")

        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        # Update index
        ni += 1

        t02 = time.time()
        cTime += [t02 - t1]

    t2 = time.time()

    ##########################################################################
    # PLOTS and SNR computation

    FreeEnergyArray = np.zeros((ni), dtype=np.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]

    SUM_q_Z_array = np.zeros((M, ni), dtype=np.float64)
    mu1_array = np.zeros((M, ni), dtype=np.float64)
    h_norm_array = np.zeros((ni), dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_q_Z_array[m, i] = SUM_q_Z[m][i]
            mu1_array[m, i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]

    if PLOT and 0:
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'size': 15}
        matplotlib.rc('font', **font)
        plt.savefig('./HRF_Iter_CompMod.png')
        plt.hold(False)
        plt.figure(2)
        plt.plot(cAH[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cFE[1:-1], 'm')
        plt.hold(False)
        #plt.legend( ('CA','CH', 'CZ', 'CAH', 'CFE') )
        plt.legend(('CAH', 'CFE'))
        plt.grid(True)
        plt.savefig('./Crit_CompMod.png')
        plt.figure(3)
        plt.plot(FreeEnergyArray)
        plt.grid(True)
        plt.savefig('./FreeEnergy_CompMod.png')

        plt.figure(4)
        for m in xrange(M):
            plt.plot(SUM_q_Z_array[m])
            plt.hold(True)
        plt.hold(False)
        #plt.legend( ('m=0','m=1', 'm=2', 'm=3') )
        #plt.legend( ('m=0','m=1') )
        plt.savefig('./Sum_q_Z_Iter_CompMod.png')

        plt.figure(5)
        for m in xrange(M):
            plt.plot(mu1_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./mu1_Iter_CompMod.png')

        plt.figure(6)
        plt.plot(h_norm_array)
        plt.savefig('./HRF_Norm_CompMod.png')

        Data_save = xndarray(h_norm_array, ['Iteration'])
        Data_save.save('./HRF_Norm_Comp.nii')

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    sigma_M = np.sqrt(np.sqrt(sigma_M))
    logger.info("Nb iterations to reach criterion: %d", ni)
    logger.info("Computational time = %s min %s s", str(
        int(CompTime // 60)), str(int(CompTime % 60)))
    # print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    # print "sigma_H = " + str(sigmaH)
    logger.info('mu_M: %f', mu_M)
    logger.info('sigma_M: %f', sigma_M)
    logger.info("sigma_H = %s" + str(sigmaH))
    logger.info("Beta = %s" + str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, J, N)
    SNR = 20 * \
        np.log(
            np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)
    logger.info("SNR = %d", SNR)
    return ni, m_A, m_H, q_Z, sigma_epsilone, mu_M, sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:], cH[2:], cZ[2:], cAH[2:], cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray


def Main_vbjde_Python_constrained(graph, Y, Onsets, Thrf, K, TR, beta, dt, scale=1, estimateSigmaH=True, sigmaH=0.1, NitMax=-1, NitMin=1, estimateBeta=False, PLOT=False):
    logger.info("EM started ...")
    np.random.seed(6537546)

    ##########################################################################
    # INITIALIZATIONS
    # Initialize parameters
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.005
    MaxItGrad = 120
    #Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    # Initialize sizes vectors
    D = int(np.ceil(Thrf / dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(np.sqrt(J))
    sigma_epsilone = np.ones(J)

    # Neighbours
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]
    # Conditions
    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    # Sigma and mu
    mu_M = np.zeros((M, K), dtype=np.float64)
    sigma_M = 0.5 * np.ones((M, K), dtype=np.float64)
    sigma_M0 = 0.5 * np.ones((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_M[:, k] = 2.0
    # Covariance matrix
    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)

    Gamma = np.identity(N)

    q_Z = np.zeros((M, K, J), dtype=np.float64)
    # for k in xrange(0,K):
    q_Z[:, 1, :] = 1
    q_Z1 = np.zeros((M, K, J), dtype=np.float64)
    Z_tilde = q_Z.copy()

    Sigma_A = np.zeros((M, M, J), np.float64)
    m_A = np.zeros((J, M), dtype=np.float64)
    TT, m_h = getCanoHRF(Thrf - dt, dt)  # TODO: check
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(mu_M[m, k],
                                              np.sqrt(sigma_M[m, k])) * Z_tilde[m, k, j]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    Sigma_H = np.ones((D, D), dtype=np.float64)
    Beta = beta * np.ones((M), dtype=np.float64)

    zerosDD = np.zeros((D, D), dtype=np.float64)
    zerosD = np.zeros((D), dtype=np.float64)
    zerosND = np.zeros((N, D), dtype=np.float64)
    zerosMM = np.zeros((M, M), dtype=np.float64)
    zerosJMD = np.zeros((J, M, D), dtype=np.float64)
    zerosK = np.zeros(K)

    P = vt.PolyMat(N, 4, TR)
    zerosP = np.zeros((P.shape[0]), dtype=np.float64)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    cA = []
    cH = []
    cZ = []
    Ndrift = L.shape[0]
    Crit_FreeEnergy = 1

    t1 = time.time()

    ##########################################################################
    # VBJDE num. iter. minimum

    ni = 0

    while ((ni < NitMin) or (((Crit_FreeEnergy > Thresh_FreeEnergy) or ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh))) and (ni < NitMax))):

        logger.info("------------------------------ Iteration n° " +
                    str(ni + 1) + " ------------------------------")

        #####################
        # EXPECTATION
        #####################

        # A
        logger.info("E A step ...")
        Sigma_A, m_A = vt.expectation_A(
            Y, Sigma_H, m_H, m_A, X, Gamma, PL, sigma_M, q_Z, mu_M, D, N, J, M, K, y_tilde, Sigma_A, sigma_epsilone, zerosJMD)
        m_A1 = m_A

        # crit A
        DIFF = np.abs(np.reshape(m_A, (M * J)) - np.reshape(m_A1, (M * J)))
        Crit_A = sum(DIFF) / len(np.where(DIFF != 0))
        cA += [Crit_A]

        # H
        logger.info("E H step ...")
        Sigma_H, m_H = vt.expectation_H(
            Y, Sigma_A, m_A, X, Gamma, PL, D, R, sigmaH, J, N, y_tilde, zerosND, sigma_epsilone, scale, zerosDD, zerosD)
        m_H[0] = 0
        m_H[-1] = 0
        m_H1 = m_H

        # crit H
        Crit_H = np.abs(np.mean(m_H - m_H1) / np.mean(m_H))
        cH += [Crit_H]

        # Z
        logger.info("E Z step ...")
        q_Z, Z_tilde = vt.expectation_Z(
            Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z, graph, M, J, K, zerosK)

        # crit Z
        DIFF = np.abs(
            np.reshape(q_Z, (M * K * J)) - np.reshape(q_Z1, (M * K * J)))
        Crit_Z = sum(DIFF) / len(np.where(DIFF != 0))
        cZ += [Crit_Z]
        q_Z1 = q_Z

        # Plotting HRF
        if PLOT and ni >= 0:
            import matplotlib.pyplot as plt
            plt.figure(M + 1)
            plt.plot(m_H)
            plt.hold(True)

        ####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            sigmaH = (np.dot(vt.mult(m_H, m_H) + Sigma_H, R)).trace()
            sigmaH /= D

        # (mu,sigma)
        logger.info("M (mu,sigma) step ...")
        mu_M, sigma_M = vt.maximization_mu_sigma(
            mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)

        # Drift L
        L = vt.maximization_L(Y, m_A, X, m_H, L, P, zerosP)
        PL = np.dot(P, L)
        y_tilde = Y - PL

        # Beta
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                Beta[m] = vt.maximization_beta(
                    Beta[m], q_Z, Z_tilde, J, K, m, graph, gamma, neighboursIndexes, maxNeighbours)
            logger.info("End estimating beta")
            # logger.info(Beta)

        # Sigma noise
        logger.info("M sigma noise step ...")
        sigma_epsilone = vt.maximization_sigma_noise(
            Y, X, m_A, m_H, Sigma_H, Sigma_A, PL, sigma_epsilone, M, zerosMM)

        #### Computing Free Energy ####
        """
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        """

        # Update index
        ni += 1

    t2 = time.time()

    ##########################################################################
    # PLOTS and SNR computation

    """FreeEnergyArray = np.zeros((ni),dtype=np.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]

    SUM_q_Z_array = np.zeros((M,ni),dtype=np.float64)
    mu1_array = np.zeros((M,ni),dtype=np.float64)
    h_norm_array = np.zeros((ni),dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]
    sigma_M = np.sqrt(np.sqrt(sigma_M))
    """

    Norm = np.linalg.norm(m_H)
    m_H /= Norm
    m_A *= Norm
    mu_M *= Norm
    sigma_M *= Norm
    sigma_M = np.sqrt(sigma_M)

    if PLOT and 0:
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'size': 15}
        matplotlib.rc('font', **font)
        plt.savefig('./HRF_Iter_CompMod.png')
        plt.hold(False)
        plt.figure(2)
        plt.plot(cAH[1:-1], 'lightblue')
        plt.hold(True)
        plt.plot(cFE[1:-1], 'm')
        plt.hold(False)
        #plt.legend( ('CA','CH', 'CZ', 'CAH', 'CFE') )
        plt.legend(('CAH', 'CFE'))
        plt.grid(True)
        plt.savefig('./Crit_CompMod.png')
        plt.figure(3)
        plt.plot(FreeEnergyArray)
        plt.grid(True)
        plt.savefig('./FreeEnergy_CompMod.png')

        plt.figure(4)
        for m in xrange(M):
            plt.plot(SUM_q_Z_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./Sum_q_Z_Iter_CompMod.png')

        plt.figure(5)
        for m in xrange(M):
            plt.plot(mu1_array[m])
            plt.hold(True)
        plt.hold(False)
        plt.savefig('./mu1_Iter_CompMod.png')

        plt.figure(6)
        plt.plot(h_norm_array)
        plt.savefig('./HRF_Norm_CompMod.png')

        Data_save = xndarray(h_norm_array, ['Iteration'])
        Data_save.save('./HRF_Norm_Comp.nii')

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    logger.info("Nb iterations to reach criterion: %d", ni)
    logger.info("Computational time = %s min %s s", str(
        int(CompTime // 60)), str(int(CompTime % 60)))
    # print "Computational time = " + str(int( CompTime//60 ) ) + " min " +
    # str(int(CompTime%60)) + " s"
    logger.info('mu_M: %f', mu_M)
    logger.info('sigma_M: %f', sigma_M)
    logger.info("sigma_H = %s" + str(sigmaH))
    logger.info("Beta = %s" + str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, J, N)
    SNR = 20 * \
        np.log(
            np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)
    logger.info('SNR comp = %f', SNR)
    return m_A, m_H, q_Z, sigma_epsilone, mu_M, sigma_M, Beta, L, PL


def Main_vbjde_Extension_constrained_stable(graph, Y, Onsets, Thrf, K, TR, beta,
                                            dt, scale=1, estimateSigmaH=True,
                                            sigmaH=0.05, NitMax=-1,
                                            NitMin=1, estimateBeta=True,
                                            PLOT=False, contrasts=[],
                                            computeContrast=False,
                                            gamma_h=0):
    """ Version modified by Lofti from Christine's version """
    logger.info(
        "Fast EM with C extension started ... Here is the stable version !")

    np.random.seed(6537546)

    # Initialize parameters
    S = 100
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5  # 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5

    # Initialize sizes vectors
    D = np.int(np.ceil(Thrf / dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = np.int(np.sqrt(J))
    condition_names = []

    # Neighbours
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]
    # Conditions
    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    # Covariance matrix
    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    invR = np.linalg.inv(R)
    Det_invR = np.linalg.det(invR)

    Gamma = np.identity(N)
    Det_Gamma = np.linalg.det(Gamma)

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    Crit_FreeEnergy = 1
    cTime = []
    cA = []
    cH = []
    cZ = []
    cAH = []

    CONTRAST = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((J, len(contrasts)), dtype=np.float64)
    Q_barnCond = np.zeros((M, M, D, D), dtype=np.float64)
    XGamma = np.zeros((M, D, N), dtype=np.float64)
    m1 = 0
    for k1 in X:  # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1, m2, :, :] = np.dot(
                np.dot(X[k1].transpose(), Gamma), X[k2])
            m2 += 1
        XGamma[m1, :, :] = np.dot(X[k1].transpose(), Gamma)
        m1 += 1

    sigma_epsilone = np.ones(J)
    logger.info(
        "Labels are initialized by setting active probabilities to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1
    q_Z1 = np.zeros((M, K, J), dtype=np.float64)
    Z_tilde = q_Z.copy()

    TT, m_h = getCanoHRF(Thrf, dt)  # TODO: check
    m_h = m_h[:D]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    sigmaH1 = sigmaH
    Sigma_H = np.ones((D, D), dtype=np.float64)

    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    y_tilde = Y - PL
    Ndrift = L.shape[0]

    sigma_M = np.ones((M, K), dtype=np.float64)
    sigma_M[:, 0] = 0.5
    sigma_M[:, 1] = 0.6
    mu_M = np.zeros((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_M[:, k] = 1  # InitMean
    Sigma_A = np.zeros((M, M, J), np.float64)
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
    m_A = np.zeros((J, M), dtype=np.float64)
    m_A1 = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(mu_M[m, k],
                                              np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]
    m_A1 = m_A

    t1 = time.time()

    ##########################################################################
    # VBJDE num. iter. minimum

    ni = 0

    while ((ni < NitMin + 1) or ((Crit_AH > Thresh) and (ni < NitMax))):

        logger.info("------------------------------ Iteration n° " +
                    str(ni + 1) + " ------------------------------")

        #####################
        # EXPECTATION
        #####################

        # A
        logger.info("E A step ...")
        UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                             Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N, K)

        # crit. A
        DIFF = np.reshape(m_A - m_A1, (M * J))
        Crit_A = (
            np.linalg.norm(DIFF) / np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        # HRF h
        UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R, Sigma_H, Y,
                             y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N, scale, sigmaH)
        #m_H[0] = 0
        #m_H[-1] = 0
        # Constrain with optimization strategy
        import cvxpy as cvx
        m, n = Sigma_H.shape
        Sigma_H_inv = np.linalg.inv(Sigma_H)
        zeros_H = np.zeros_like(m_H[:, np.newaxis])
        # Construct the problem. PRIMAL
        h = cvx.Variable(n)
        expression = cvx.quad_form(h - m_H[:, np.newaxis], Sigma_H_inv)
        objective = cvx.Minimize(expression)
        #constraints = [h[0] == 0, h[-1]==0, h >= zeros_H, cvx.square(cvx.norm(h,2))<=1]
        constraints = [h[0] == 0, h[-1] == 0, cvx.square(cvx.norm(h, 2)) <= 1]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(verbose=0, solver=cvx.CVXOPT)
        # Now we update the mean of h
        m_H_old = m_H
        Sigma_H_old = Sigma_H
        m_H = np.squeeze(np.array((h.value)))
        Sigma_H = np.zeros_like(Sigma_H)
        # and the norm
        h_norm += [np.linalg.norm(m_H)]

        # crit. h
        Crit_H = (np.linalg.norm(m_H - m_H1) / np.linalg.norm(m_H1)) ** 2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        # crit. AH
        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * m_H[d]
        DIFF = np.reshape(AH - AH1, (M * J * D))
        Crit_AH = (np.linalg.norm(
            DIFF) / (np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        # Z labels
        logger.info("E Z step ...")
        UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M,
                             q_Z, neighboursIndexes.astype(np.int32), M, J, K, maxNeighbours)

        # crit. Z
        DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
        Crit_Z = (np.linalg.norm(DIFF) /
                  (np.linalg.norm(np.reshape(q_Z1, (M * K * J))) + eps)) ** 2
        cZ += [Crit_Z]
        q_Z1[:, :, :] = q_Z[:, :, :]

        #####################
        # MAXIMIZATION
        #####################

        # HRF: Sigma_h
        if estimateSigmaH:
            logger.info("M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = vt.maximization_sigmaH_prior(
                    D, Sigma_H, R, m_H, gamma_h)
            else:
                sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)
            logger.info('sigmaH = %s', str(sigmaH))

        # (mu,sigma)
        logger.info("M (mu,sigma) step ...")
        mu_M, sigma_M = vt.maximization_mu_sigma(
            mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)

        # Drift L
        UtilsC.maximization_L(
            Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M, Ndrift, N)
        PL = np.dot(P, L)
        y_tilde = Y - PL

        # Beta
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                Beta[m] = UtilsC.maximization_beta(beta, q_Z[m, :, :].astype(np.float64), Z_tilde[m, :, :].astype(
                    np.float64), J, K, neighboursIndexes.astype(np.int32), gamma, maxNeighbours, MaxItGrad, gradientStep)
            logger.info("End estimating beta")
            logger.info(Beta)

        # Sigma noise
        logger.info("M sigma noise step ...")
        UtilsC.maximization_sigma_noise(
            Gamma, PL, sigma_epsilone, Sigma_H, Y, m_A, m_H, Sigma_A, XX.astype(np.int32), J, D, M, N)

        t02 = time.time()
        cTime += [t02 - t1]

    t2 = time.time()

    ##########################################################################
    # PLOTS and SNR computation

    if PLOT and 0:
        font = {'size': 15}
        matplotlib.rc('font', **font)
        savefig('./HRF_Iter_CompMod.png')
        hold(False)
        figure(2)
        plot(cAH[1:-1], 'lightblue')
        hold(True)
        plot(cFE[1:-1], 'm')
        hold(False)
        legend(('CAH', 'CFE'))
        grid(True)
        savefig('./Crit_CompMod.png')
        figure(3)
        plot(FreeEnergyArray)
        grid(True)
        savefig('./FreeEnergy_CompMod.png')

        figure(4)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        savefig('./Sum_q_Z_Iter_CompMod.png')

        figure(5)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        savefig('./mu1_Iter_CompMod.png')

        figure(6)
        plot(h_norm_array)
        savefig('./HRF_Norm_CompMod.png')

        Data_save = xndarray(h_norm_array, ['Iteration'])
        Data_save.save('./HRF_Norm_Comp.nii')

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    """
    Norm = np.linalg.norm(m_H)
    m_H /= Norm
    Sigma_H /= Norm**2
    sigmaH /= Norm**2
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = np.sqrt(np.sqrt(sigma_M))
    """
    logger.info("Nb iterations to reach criterion: %d", ni)
    logger.info("Computational time = %s min %s s", str(
        np.int(CompTime // 60)), str(np.int(CompTime % 60)))
    logger.info('mu_M: %f', mu_M)
    logger.info('sigma_M: %f', sigma_M)
    logger.info("sigma_H = %s", str(sigmaH))
    logger.info("Beta = %s", str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, J, N)
    SNR = 20 * \
        np.log(
            np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)
    logger.info('SNR comp = %f', SNR)
    return ni, m_A, m_H, q_Z, sigma_epsilone, mu_M, sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:], cH[2:], cZ[2:], cAH[2:], cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal
