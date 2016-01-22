# -*- coding: utf-8 -*-
"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

import numpy as np
import scipy as sp
import logging

import vem_tools as vt
from collections import OrderedDict

eps = 1e-8

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def expectation_H_asl(Sigma_A, m_A, m_C, G, XX, W, Gamma, Gamma_X, X_Gamma_X, J, y_tilde,
                  cov_noise, R_inv, sigmaH, prior_mean_term, prior_cov_term):
    """
    Expectation-H step: 
    p_H = argmax_h(E_pa,pc,pg[log p(h|y, a, c, g; theta)])
        \propto exp(E_pa,pc,pg[log p(y|h, a, c, g; theta) + log p(h; sigmaH)])

    Returns:
    m_H, Sigma_H of probability distribution p_H of the current iteration
    """

    ## Precomputations
    WXG = W.dot(XX.dot(G).T)
    mAX = np.tensordot(m_A, XX, axes=(1, 0))                # shape (J, N, D)
    #Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    #X_Gamma_X = np.tensordot(XX.T, Gamma_X, axes=(1, 0))    # shape (D, M, M, D)
    #cov_noise = np.maximum(sigma_eps, eps)[:, np.newaxis, np.newaxis]
    mAX_Gamma = (np.tensordot(mAX, Gamma, axes=(1, 0)) / cov_noise) # shape (J, D, N)

    ## Sigma_H computation
    # first summand: part of the prior -> R^-1 / sigmaH + prior_cov_term
    S_a = R_inv / sigmaH + prior_cov_term
    # second summand: E_pa[Saj.T*Gamma*Saj]
    # sum_{m, m'} Sigma_a(m,m') X_m.T Gamma_i X_m'
    S_a += (np.einsum('ijk,lijm->klm', Sigma_A, X_Gamma_X) / cov_noise).sum(0)
    # third summand: E_pa[Saj.T*Gamma*Saj]
    # (sum_m m_a X_m).T Gamma_i (sum_m m_a X_m)
    for i in xrange(0, J):
        S_a += mAX_Gamma[i, :, :].dot(mAX[i, :, :])  #option 1 faster 13.4
    #S_a += np.einsum('ijk,ikl->ijl', mAX_Gamma, mAX).sum(0) # option 2 second 8.8
    #S_a += np.einsum('ijk,ikl->jl', mAX_Gamma, mAX) # option 3 slower 7.5

    # Sigma_H = S_a^-1
    Sigma_H = np.linalg.inv(S_a)

    ## m_H
    # Y_bar_tilde computation: (sum_m m_a X_m).T Gamma_i y_tildeH
    # y_tildeH = yj - sum_m m_C WXG - w alphaj - P lj
    y_tildeH = y_tilde - WXG.dot(m_C.T)
    #Y_bar_tilde = np.tensordot(mAX_Gamma, y_tildeH, axes=([0, 2], [1, 0])) # slower
    Y_bar_tilde = np.einsum('ijk,ki->j', mAX_Gamma, y_tildeH)
    # we sum the term that corresponds to the prior
    Y_bar_tilde += prior_mean_term
    
    # m_H = S_a^-1 y_bar_tilde
    m_H = np.dot(np.linalg.inv(S_a), Y_bar_tilde)

    return m_H, Sigma_H


def expectation_G_asl(Sigma_C, m_C, m_A, H, XX, W, WX, Gamma, Gamma_WX,
                  XW_Gamma_WX, J, y_tilde, cov_noise, R_inv, sigmaG,
                  prior_mean_term, prior_cov_term):
    """
    Expectation-G step: 
    p_G = argmax_g(E_pa,pc,ph[log p(g|y, a, c, h; theta)])
        \propto exp(E_pa,pc,ph[log p(y|h, a, c, g; theta) + log p(g; sigmaG)])

    Returns:
    m_G, Sigma_G of probability distribution p_G of the current iteration
    """

    ## Precomputations
    XH = XX.dot(H).T
    #WX = W.dot(XX).transpose(1, 0, 2)
    mCWX = np.tensordot(m_C, WX, axes=(1, 0))                    # shape (J, N, D)
    #Gamma_WX = np.tensordot(Gamma, WX, axes=(1, 1))
    #XW_Gamma_WX = np.tensordot(WX.T, Gamma_WX, axes=(1, 0))     # shape (D, M, M, D)
    #cov_noise = np.maximum(sigma_eps, eps)[:, np.newaxis, np.newaxis]
    mCWX_Gamma = np.tensordot(mCWX, Gamma, axes=(1, 0)) / cov_noise # shape (J, D, N)

    ## Sigma_H computation
    # first summand: part of the prior -> R^-1 / sigmaH + prior_cov_term
    S_c = R_inv / sigmaG + prior_cov_term
    # second summand: E_pc[Scj.T*Gamma*Scj] op1
    # sum_{m, m'} Sigma_c(m,m') X_m.T W.T Gamma_i W X_m'
    S_c += (np.einsum('ijk,lijm->klm', Sigma_C, XW_Gamma_WX) / cov_noise).sum(0)
    # third summand: E_pc[Scj.T*Gamma*Scj] op2
    # (sum_m m_c X_m).T Gamma_i (sum_m m_c W X_m)
    for i in xrange(0, J):
        S_c += mCWX_Gamma[i, :, :].dot(mCWX[i, :, :])  # option 1 faster 13.4
    #S_c += np.einsum('ijk,ikl->ijl', mCWX_Gamma, mCWX).sum(0) # option 2 second 8.8
    #S_c += np.einsum('ijk,ikl->jl', mCWX_Gamma, mCWX) # option 3 slower 7.5
    
    # Sigma_G = S_c^-1
    Sigma_G = np.linalg.inv(S_c)

    ## m_G
    # Y_bar_tilde computation: (sum_m m_c W X_m).T Gamma_i y_tildeG
    # y_tildeG = yj - sum_m m_A XH - w alphaj - P lj
    y_tildeG = y_tilde - XH.dot(m_A.T)
    #Y_bar_tilde = np.tensordot(mCWX_Gamma, y_tildeG, axes=([0, 2], [1, 0]))  # slower
    Y_bar_tilde = np.einsum('ijk,ki->j', mCWX_Gamma, y_tildeG)
    # we sum the term that corresponds to the prior
    Y_bar_tilde += prior_mean_term
    
    # m_H = S_c^-1 y_bar_tilde
    m_G = np.dot(Sigma_G, Y_bar_tilde)

    return m_G, Sigma_G



def expectation_A_asl(H, G, m_C, W, XX, Gamma, Gamma_X, q_Z, mu_Ma, sigma_Ma,
                  J, y_tilde, Sigma_H, sigma_eps_m):
    """
    Expectation-A step: 
    p_A = argmax_h(E_pc,pq,ph,pg[log p(a|y, h, c, g, q; theta)])
        \propto exp(E_pc,ph,pg[log p(y|h, a, c, g; theta)] \
                  + E_pq[log p(a|q; mu_Ma, sigma_Ma)])

    Returns:
    m_A, Sigma_A of probability distribution p_A of the current iteration
    """

    ## Pre-compute XH, X*Sigma_H, XG, WXG, Gamma*X
    XH = XX.dot(H).T
    Sigma_H_X = XX.dot(Sigma_H.T).T
    XG = XX.dot(G).T
    WXG = W.dot(XG)
    #Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    #sigma_eps_m = np.maximum(sigma_eps, eps)

    ## Sigma_A computation
    # first summand of Sigma_A: XH.T*Gamma*XH / sigma_eps
    Sigma_A = XH.T.dot(Gamma).dot(XH)[..., np.newaxis] / sigma_eps_m
    # second summand of Sigma_A: tr(X.T*Gamma*X*Sigma_H / sigma_eps)
    second_summand = np.einsum('ijk, jli', Sigma_H_X, Gamma_X)
    Sigma_A += second_summand[..., np.newaxis] / sigma_eps_m
    # third summand of Sigma_A: part of p(a|q; theta_A)
    Delta_k = (q_Z / np.maximum(sigma_Ma[:, :, np.newaxis], eps))
    Delta = Delta_k.sum(axis=1)         # sum across classes K
    for i in xrange(0, J):
        Sigma_A[:, :, i] = np.linalg.inv(Sigma_A[:, :, i] + \
                                         np.diag(Delta[:, i]))
    
    ## m_A computation
    # adding m_C*WXG to y_tilde
    y_tildeH = y_tilde - WXG.dot(m_C.T)
    Gamma_y_tildeH = Gamma.dot(y_tildeH).T
    X_tildeH = Gamma_y_tildeH.dot(XH) / sigma_eps_m[:, np.newaxis] \
               + (Delta_k * mu_Ma[:, :, np.newaxis]).sum(axis=1).T
    # dot product across voxels of Sigma_A and X_tildeH
    m_A = np.einsum('ijk,kj->ki', Sigma_A, X_tildeH)

    return m_A, Sigma_A


def expectation_C_asl(G, H, m_A, W, XX, Gamma, Gamma_X, q_Z, mu_Mc, sigma_Mc,
                  J, y_tilde, Sigma_G, sigma_eps_m):
    """
    Expectation-C step: 
    p_C = argmax_h(E_pa,pq,ph,pg[log p(a|y, h, a, g, q; theta)])
        \propto exp(E_pa,ph,pg[log p(y|h, a, c, g; theta)] \
                  + E_pq[log p(c|q; mu_Mc, sigma_Mc)])

    Returns:
    m_C, Sigma_C of probability distribution p_C of the current iteration
    """
    
    ## Pre-compute XH, X*Sigma_H, XG, WXG, Gamma*X
    XH = XX.dot(H).T
    Sigma_G_X = XX.dot(Sigma_G.T).T
    XG = XX.dot(G).T
    WXG = W.dot(XG)
    #Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    #sigma_eps_m = np.maximum(sigma_eps, eps)
    
    ## Sigma_C computation
    # first summand of Sigma_C: WXG.T*Gamma*WXG / sigma_eps
    Sigma_C = WXG.T.dot(Gamma).dot(WXG)[..., np.newaxis] / sigma_eps_m
    # second summand of Sigma_C: tr(X.T*Gamma*X*Sigma_G / sigma_eps)
    second_summand = np.einsum('ijk, jli', Sigma_G_X, Gamma_X)
    Sigma_C += second_summand[..., np.newaxis] / sigma_eps_m
    # third summand of Sigma_C: part of p(c|q; theta_C)
    Delta_k = (q_Z / np.maximum(sigma_Mc[:, :, np.newaxis], eps))
    Delta = Delta_k.sum(axis=1)          # sum across classes K
    for i in xrange(0, J):
        Sigma_C[:, :, i] = np.linalg.inv(Sigma_C[:, :, i] + \
                                         np.diag(Delta[:, i]))

    ## m_C computation
    # adding m_A*XH to y_tilde
    y_tildeG = y_tilde - XH.dot(m_A.T)
    Gamma_y_tildeG_WXG = Gamma.dot(y_tildeG).T.dot(WXG) 
    X_tildeG = Gamma_y_tildeG_WXG / sigma_eps_m[:, np.newaxis] + \
                    (Delta_k * mu_Mc[:, :, np.newaxis]).sum(axis=1).T
    # dot product across voxels of Sigma_C and X_tildeG
    m_C = np.einsum('ijk,kj->ki', Sigma_C, X_tildeG)

    return m_C, Sigma_C


def expectation_Q_asl(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, p_q_t, p_Q, neighbours_indexes, graph, M, J, K):
    # between ASL and BOLD just alpha and Gauss_mat change!!!
    alpha = - 0.5 * np.diagonal(Sigma_A)[:, :, np.newaxis] / (sigma_Ma[np.newaxis, :, :]) \
            - 0.5 * np.diagonal(Sigma_C)[:, :, np.newaxis] / (sigma_Mc[np.newaxis, :, :])  # (J, M, K)
    Gauss_mat = vt.normpdf(m_A[...,np.newaxis], mu_Ma, np.sqrt(sigma_Ma)) * \
                vt.normpdf(m_C[...,np.newaxis], mu_Mc, np.sqrt(sigma_Mc))

    # Update Ztilde ie the quantity which is involved in the a priori 
    # Potts field [by solving for the mean-field fixed point Equation]
    # TODO: decide if we take out the computation of p_q_t or Ztilde
    B_pqt = Beta[:, np.newaxis, np.newaxis] * p_q_t.copy()
    B_pqt = np.concatenate((B_pqt, np.zeros((M, K, 1), dtype=B_pqt.dtype)), axis=2)
    local_energy = B_pqt[:, :, neighbours_indexes].sum(axis=3).transpose(2, 0, 1)
    energy = (alpha + local_energy)
    energy -= energy.max()
    Probas = (np.exp(energy) * Gauss_mat).transpose(1, 2, 0)
    aux = Probas.sum(axis=1)[:, np.newaxis, :]
    aux[np.where(aux==0)] = eps
    p_q_t = Probas / aux
        
    return p_q_t, p_q_t


def expectation_Q_async_asl(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, p_q_t, p_Q, neighbours_indexes, graph, M, J, K):
    # between ASL and BOLD just alpha and Gauss_mat change!!!
    alpha = (- 0.5 * np.diagonal(Sigma_A)[:, :, np.newaxis] / (sigma_Ma[np.newaxis, :, :]) \
             - 0.5 * np.diagonal(Sigma_C)[:, :, np.newaxis] / (sigma_Mc[np.newaxis, :, :])).transpose(1, 2, 0)  # (J, M, K)
    Gauss_mat = (vt.normpdf(m_A[...,np.newaxis], mu_Ma, np.sqrt(sigma_Ma)) * \
                 vt.normpdf(m_C[...,np.newaxis], mu_Mc, np.sqrt(sigma_Mc))).transpose(1, 2, 0)

    energy = np.zeros_like(p_q_t)
    local_energy = np.zeros_like(p_q_t)
    Probas = np.zeros_like(p_q_t)
    for vox in xrange(J):
        B_pqt = Beta[:, np.newaxis, np.newaxis] * p_q_t.copy()
        B_pqt = np.concatenate((B_pqt, np.zeros((M, K, 1), dtype=B_pqt.dtype)), axis=2)
        local_energy[:, :, vox] = B_pqt[:, :, neighbours_indexes[vox, :]].sum(axis=2)
        energy[:, :, vox] = (alpha[:, :, vox] + local_energy[:, :, vox])
        Probas[:, :, vox] = (np.exp(energy[:, :, vox]) * Gauss_mat[:, :, vox])
        aux = Probas[:, :, vox].sum(axis=1)[:, np.newaxis]
        aux[np.where(aux==0)] = eps
        p_q_t[:, :, vox] = Probas[:, :, vox] / aux
        
        #local_energy[:, :, vox] = sum_over_neighbours(neighbours_indexes[vox, :], beta[..., np.newaxis, np.newaxis] * labels_proba)
        #energy[:, :, vox] = alpha[:, :, vox] + local_energy[:, :, vox]
        #labels_proba[:, :, vox] = (np.exp(energy[:, :, vox])*gauss[:, :, vox])
        #labels_proba[:, :, vox] = labels_proba[:, :, vox]/labels_proba[:, :, vox].sum(axis=1)[:, np.newaxis]

    return p_q_t, p_q_t
    

def maximization_mu_sigma_asl(q_Z, m_X, Sigma_X):

    qZ_sumvox = q_Z.sum(axis=2)                              # (M, K)
    Mu = np.zeros_like(qZ_sumvox)
    Mu[:, 1] = (q_Z[:, 1, :] * m_X.T).sum(axis=1) / qZ_sumvox[:, 1]

    mX_minus_Mu_2 = (m_X[..., np.newaxis] - Mu[np.newaxis, ...]).transpose(1, 2, 0)**2
    SigmaX_diag = np.diagonal(Sigma_X).T
    Sigma = (q_Z * (mX_minus_Mu_2 + SigmaX_diag[:, np.newaxis, :])).sum(axis=2) / qZ_sumvox

    return Mu, Sigma


def maximization_sigma_asl(D, Sigma_H, R_inv, m_H, use_hyp, gamma_h):
    alpha = (np.dot(m_H[:, np.newaxis] * m_H[np.newaxis, :] + Sigma_H, R_inv)).trace()
    if use_hyp:
        sigma = (-(D) + np.sqrt((D) * (D) + 8 * gamma_h * alpha)) / (4*gamma_h)
    else:
        sigma = alpha / (D)
    if np.isnan(sigma) or sigma==0:
        print 'WARNING!!!'
        print 'gamma_h = ', gamma_h
        print 'D = ', D
        print 'alpha = ', alpha
        print m_H
    return sigma


def maximization_Mu_asl(H, G, matrix_covH, matrix_covG,
                    sigmaH, sigmaG, sigmaMu, Omega, R_inv):
    Aux = matrix_covH / sigmaH + np.dot(Omega.T, Omega) / sigmaG + R_inv / sigmaMu
    Mu = np.dot(sp.linalg.inv(Aux), (H / sigmaH + np.dot(Omega.T, G) / sigmaG))
    return Mu


def maximization_beta_m4_asl(beta, p_Q, Qtilde_sumneighbour, Qtilde, neighboursIndexes, 
                              maxNeighbours, gamma, MaxItGrad, gradientStep):
    # Method 4 in Christine's thesis:
    # - sum_j sum_k (sum_neighbors p_Q) (p_Q_MF - p_Q) - gamma
    Gr = 100
    ni = 1
    while ((abs(Gr) > 0.0001) and (ni < MaxItGrad)):
        
        # p_Q_MF according to new beta
        beta_Qtilde_sumneighbour = beta * Qtilde_sumneighbour
        E = np.exp(beta_Qtilde_sumneighbour - beta_Qtilde_sumneighbour.max(axis=0))  # (K, J)
        p_Q_MF = E / E.sum(axis=0)
        
        # Gradient computation
        Gr = gamma + (Qtilde_sumneighbour * (p_Q_MF - p_Q)).sum()

        # Update of beta: beta - gradientStep * Gradient
        beta -= gradientStep * Gr

        ni += 1
    return beta


def maximization_beta_m2_asl(beta, p_Q, Qtilde_sumneighbour, Qtilde, neighboursIndexes, 
                              maxNeighbours, gamma, MaxItGrad, gradientStep):
    # Method 2 in Christine's thesis:
    # - 1/2 sum_j sum_k sum_neighbors (p_Q_MF p_Q_MF_sumneighbour - p_Q pQ_sumneighbour) - gamma
    Gr = 100
    ni = 1
    
    while ((abs(Gr) > 0.0001) and (ni < MaxItGrad)):
        
        # p_Q_MF according to new beta
        beta_Qtilde_sumneighbour = beta * Qtilde_sumneighbour
        E = np.exp(beta_Qtilde_sumneighbour - beta_Qtilde_sumneighbour.max(axis=0))  # (K, J)
        p_Q_MF = E / E.sum(axis=0)

        # sum_neighbours p_Q_MF(neighbour) according to new beta
        p_Q_MF_sumneighbour = p_Q_MF[:, neighboursIndexes].sum(axis=2)

        # Gradient computation
        Gr = gamma + ( p_Q_MF * p_Q_MF_sumneighbour - p_Q * Qtilde_sumneighbour ).sum() / 2.

        # Update of beta: beta - gradientStep * Gradient
        beta -= gradientStep * Gr

        ni += 1
    return beta


def fun(Beta, p_Q, Qtilde_sumneighbour, neighboursIndexes, gamma):
    'function to minimize'
    # p_Q_MF according to new beta
    if (p_Q).ndim == 3:
        beta_Qtilde_sumneighbour = (Beta[:, np.newaxis, np.newaxis] * \
                                Qtilde_sumneighbour).squeeze()
    else:
        beta_Qtilde_sumneighbour = (Beta * Qtilde_sumneighbour).squeeze()
    # Mean field approximation
    E = np.exp(beta_Qtilde_sumneighbour - beta_Qtilde_sumneighbour.max(axis=0))  # (K, J)
    aux = E.sum(axis=0)
    aux[np.where(aux==0)] = eps
    p_Q_MF = E / aux
    if (p_Q).ndim == 3:
        p_Q_MF_sumneighbour = p_Q_MF[:, :, neighboursIndexes].sum(axis=3)  # (M, K, J)
        function = - np.log(np.exp(beta_Qtilde_sumneighbour).sum(axis=1)).sum() \
                    + (Beta * (p_Q * Qtilde_sumneighbour / 2. + \
                        p_Q_MF * (Qtilde_sumneighbour - p_Q_MF_sumneighbour / 2.) \
                        ).sum(axis=(1, 2))) - gamma * Beta
        gradient = - gamma * np.ones_like(Beta) - (p_Q_MF * p_Q_MF_sumneighbour - \
                                   p_Q * Qtilde_sumneighbour).sum(axis=(1, 2)) / 2.
    else:
        p_Q_MF_sumneighbour = p_Q_MF[:, neighboursIndexes].sum(axis=2)  # (M, K, J)
        function = - np.log(np.exp(beta_Qtilde_sumneighbour).sum(axis=0)).sum() \
                    + (Beta * (p_Q * Qtilde_sumneighbour / 2. + \
                          p_Q_MF * (Qtilde_sumneighbour - p_Q_MF_sumneighbour / 2.) \
                          ).sum()) - gamma * Beta
        gradient = - gamma * np.ones_like(Beta) - (p_Q_MF * p_Q_MF_sumneighbour - \
                                                   p_Q * Qtilde_sumneighbour).sum() / 2.
    return np.asfortranarray([-function])#, np.asfortranarray([ - gradient])


def grad_fun(Beta, p_Q, Qtilde_sumneighbour, neighboursIndexes, gamma):
    'function to minimize'
    # p_Q_MF according to new beta
    beta_Qtilde_sumneighbour = (Beta * \
                                Qtilde_sumneighbour).squeeze()
    # Mean field approximation
    E = np.exp(beta_Qtilde_sumneighbour - beta_Qtilde_sumneighbour.max(axis=0))  # (K, J)
    aux = E.sum(axis=0)
    aux[np.where(aux==0)] = eps
    p_Q_MF = E / aux
    if (p_Q_MF).ndim == 3:
        p_Q_MF_sumneighbour = p_Q_MF[:, :, neighboursIndexes].sum(axis=3)  # (M, K, J)
        gradient = - gamma * np.ones_like(Beta) - (p_Q_MF * p_Q_MF_sumneighbour - \
                                                   p_Q * Qtilde_sumneighbour).sum(axis=(1, 2)) / 2.
    else:
        p_Q_MF_sumneighbour = p_Q_MF[:, neighboursIndexes].sum(axis=2)  # (M, K, J)
        gradient = - gamma * np.ones_like(Beta) - (p_Q_MF * p_Q_MF_sumneighbour - \
                                                  p_Q * Qtilde_sumneighbour).sum() / 2.
    return np.asfortranarray([-gradient])


def maximization_beta_m2_scipy_asl(Beta, p_Q, Qtilde_sumneighbour, Qtilde, neighboursIndexes, 
                              maxNeighbours, gamma, MaxItGrad, gradientStep):
    """ Maximize beta """
    from scipy.optimize import fmin_l_bfgs_b, minimize, fmin_bfgs, fmin_cg, brentq

    try:
        beta_new, res = brentq(
            grad_fun, 0., 10, args=(p_Q, Qtilde_sumneighbour, neighboursIndexes, gamma),
            full_output=True
        )
    except ValueError:
        beta_new = Beta
        class res(): pass
        res.converged = False
    if not res.converged:
        logger.warning("max beta did not converge: %s (beta=%s)", res, beta_new)

    return beta_new #, res.converged  

    """
    #beta_new = fmin_l_bfgs_b(fun, Beta, \
    #                args=(p_Q, Qtilde_sumneighbour, neighboursIndexes, gamma),
    #                bounds=[(0, 10)] * (len(np.array([Beta]))))
    try:
        n = len(Beta)
    except:
        n = 1
    print n
    beta_new = fmin_l_bfgs_b(fun, np.array([Beta]), \
                    args=(p_Q, Qtilde_sumneighbour, neighboursIndexes, gamma),
                    fprime=grad_fun, 
                    bounds=[(0, None)]*n)
    print beta_new
    # You can add at the end fprime=grad_fun to give the gradient separately
    return beta_new[0]"""


def maximization_LA_asl(Y, m_A, m_C, XX, WP, W, WP_Gamma_WP, H, G, Gamma):
    # Precomputations
    #WP = np.append(w[:, np.newaxis], P, axis=1)
    #Gamma_WP = Gamma.dot(WP)
    #WP_Gamma_WP = WP.T.dot(Gamma_WP)
    
    # TODO: for BOLD, just take out mCWXG term
    mAXH = m_A.dot(XX.dot(H)).T                             # shape (N, J)
    mCWXG = XX.dot(G).dot(W).T.dot(m_C.T)                   # shape (N, J)
    S = Y - (mAXH + mCWXG)
    
    AL = np.linalg.inv(WP_Gamma_WP).dot(WP.T).dot(Gamma).dot(S)
    return AL   ##AL[1:, :], AL[0, :], AL


def maximization_sigma_noise_asl(XX, m_A, Sigma_A, H, m_C, Sigma_C, G, \
                             Sigma_H, Sigma_G, W, y_tilde, Gamma, \
                             Gamma_X, Gamma_WX, N):
    """
    Maximization sigma_noise
    """
    # Precomputations
    XH = XX.dot(H)                                          # shape (N, M)
    XG = XX.dot(G)                                          # shape (N, M)
    WX = W.dot(XX).transpose(1, 0, 2)                       # shape (M, N, D)
    WXG = W.dot(XG.T)
    mAXH = m_A.dot(XH)                                      # shape (J, N)
    mAXH_Gamma = mAXH.dot(Gamma)                            # shape (J, N)
    mCWXG = m_C.dot(WXG.T)                                  # shape (J, N)
    mCWXG_Gamma = mCWXG.dot(Gamma)                          # shape (J, N)
    #Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    #Gamma_WX = np.tensordot(Gamma, WX, axes=(1, 1))

    HXXH = XH.dot(Gamma_X.dot(H))       # (,D)*(D,N,M)*(M,N,D)*(D,) -> (M, M)
    Sigma_H_X_X = np.einsum('ijk,jli->kl', XX.dot(Sigma_H.T).T, Gamma_X)
    GXWWXG = WXG.T.dot(Gamma_WX.dot(G)) # (,D)*(D,N,M)*(M,N,D)*(D,) -> (M, M)
    Sigma_G_XW_WX = np.einsum('ijk,jlk->il', WX.dot(Sigma_G.T), Gamma_WX)
    
    # mA.T (X.T H.T H X + SH X.T X) mA
    HXAAXH = np.einsum('ij,ij->i', m_A.dot(HXXH + Sigma_H_X_X), m_A)
    
    # mC.T (W.T X.T G.T G X W + SH W.T X.T X W) mC
    GXWCCWXG = np.einsum('ij,ij->i', m_C.dot(GXWWXG + Sigma_G_XW_WX), m_C)

    # trace(Sigma_A (X.T H.T H X + SH X.T X) ) in each voxel
    tr_SA_HXXH = np.einsum('ijk,ji->k', Sigma_A, (HXXH + Sigma_H_X_X))

    # trace(Sigma_A (W.T X.T G.T G X W + SH W.T X.T X W) ) in each voxel
    tr_SC_HXXH = np.einsum('ijk,ji->k', Sigma_C, (GXWWXG + Sigma_G_XW_WX))

    # mC.T W.T X.T G.T H X mA
    CWXGHXA = np.einsum('ij,ij->i', mCWXG_Gamma, mAXH)
    
    # y_tilde.T y_tilde
    Gamma_ytilde = Gamma.dot(y_tilde)
    ytilde_ytilde = np.einsum('ij,ij->j', y_tilde, Gamma_ytilde)
    
    # (mA X H + mC W X G) y_tilde
    AXH_CWXG_ytilde = np.einsum('ij,ji->i', (mAXH + mCWXG), Gamma_ytilde)
    
    return (HXAAXH + GXWCCWXG + tr_SA_HXXH + tr_SC_HXXH \
            + 2 * CWXGHXA + ytilde_ytilde - 2 * AXH_CWXG_ytilde) / N



def computeFit_asl(H, m_A, G, m_C, W, XX):
    """ Compute Fit """
    # Precomputations
    XH = XX.dot(H)                                          # shape (N, M)
    XG = XX.dot(G)                                          # shape (N, M)
    WX = W.dot(XX).transpose(1, 0, 2)                       # shape (M, N, D)
    WXG = W.dot(XG.T)
    mAXH = m_A.dot(XH).T                                    # shape (N, J)
    mCWXG = m_C.dot(WXG.T).T                                # shape (N, J)
    
    return mAXH + mCWXG



# Entropy functions
##############################################################

eps_FreeEnergy = 0.0000000001


def RL_Entropy(Sigma_RL, M, J):
    #logger.info('Computing RLs Entropy ...')
    Det_Sigma_RL_j = np.zeros(J, dtype=np.float64)
    Const = (2 * np.pi * np.exp(1)) ** M
    Entropy = 0.0
    for j in xrange(0, J):
        Det_Sigma_RL_j = np.linalg.det(Sigma_RL[:, :, j])
        Entropy_j = np.sqrt(Const * Det_Sigma_RL_j)
        Entropy += np.log(Entropy_j)
    return Entropy


def RF_Entropy(Sigma_RF, D):
    #logger.info('Computing RF Entropy ...')
    if Sigma_RF.sum()==0:
        Entropy = 0
    else:
        _, log_Det_Sigma_RF = np.linalg.slogdet(Sigma_RF)
        Const = (2 * np.pi * np.exp(1)) ** D
        Entropy = (np.log(Const) + log_Det_Sigma_RF) /2
    return Entropy


def Q_Entropy(q_Z, M, J):
    #logger.info('Computing Z Entropy ...')
    q_Z_control_0 = q_Z.copy()
    q_Z_control_0[np.where(q_Z==0)] = eps_FreeEnergy
    return - (q_Z * np.log(q_Z_control_0)).sum()


def RL_expectation_Ptilde(m_X, Sigma_X, mu_Mx, sigma_Mx, q_Z):
    #logger.info('Computing RLs expectation Ptilde ...')
    diag_Sigma_X = np.diagonal(Sigma_X)[:, :, np.newaxis]
    S = - q_Z.transpose(2, 0, 1) * (np.log(2 * np.pi * sigma_Mx) + \
                 ((m_X[:, :, np.newaxis] - mu_Mx[np.newaxis, :, :]) ** 2 \
                    + diag_Sigma_X) / sigma_Mx[np.newaxis, :, :]) / 2.
    return S.sum()


def RF_expectation_Ptilde(m_X, Sigma_X, sigmaX, R, R_inv, D):
    #logger.info('Computing RF expectation Ptilde ...')
    _, logdetR = np.linalg.slogdet(R)
    const = D * np.log(2 * np.pi) + logdetR
    S = (np.dot(np.dot(m_X.T, R_inv), m_X) + np.dot(Sigma_X, R_inv).trace()) / sigmaX \
        + D * np.log(sigmaX)
    return - (const + S) / 2.


def Q_expectation_Ptilde(q_Z, neighboursIndexes, Beta, gamma, K, M):
    #Qtilde = np.concatenate((q_Z, np.zeros((M, K, 1), dtype=q_Z.dtype)), axis=2)
    Qtilde_sumneighbour = q_Z[:, :, neighboursIndexes].sum(axis=3) # (M, K, J)
    beta_Qtilde_sumneighbour = Beta[:, np.newaxis, np.newaxis] * Qtilde_sumneighbour

    # Mean field approximation
    E = np.exp(beta_Qtilde_sumneighbour - beta_Qtilde_sumneighbour.max(axis=0)) # (K, J)
    aux = E.sum(axis=0)
    aux[np.where(aux==0)] = eps
    p_Q_MF = E / aux
    
    # sum_neighbours p_Q_MF(neighbour) according to new beta
    p_Q_MF_sumneighbour = p_Q_MF[:, :, neighboursIndexes].sum(axis=3)  # (M, K, J)

    return - np.log(np.exp(beta_Qtilde_sumneighbour).sum(axis=1)).sum() \
           + (Beta * (q_Z * Qtilde_sumneighbour / 2. + \
                      p_Q_MF * (Qtilde_sumneighbour - p_Q_MF_sumneighbour / 2.)\
                      ).sum(axis=(1,2))).sum() #- (Beta * gamma).sum()


def expectation_Ptilde_Likelihood(y_tilde, m_A, Sigma_A, H, Sigma_H, m_C,
                                  Sigma_C, G, Sigma_G, XX, W, sigma_eps,
                                  Gamma, J, D, M, N, Gamma_X, Gamma_WX):
    #print sigma_eps[np.where(np.isnan(np.log(sigma_eps)))]
    sigma_eps_1 = maximization_sigma_noise(XX, m_A, Sigma_A, H, m_C, Sigma_C, \
                                           G, Sigma_H, Sigma_G, W, y_tilde, Gamma, \
                                           Gamma_X, Gamma_WX, N)
    return  - (N * J * np.log(2 * np.pi) - J * np.log(np.linalg.det(Gamma)) \
            + N * np.log(sigma_eps).sum() + N * (sigma_eps_1 / sigma_eps).sum()) / 2.


def Compute_FreeEnergy_asl(y_tilde, m_A, Sigma_A, mu_Ma, sigma_Ma, m_H, Sigma_H, AuxH,
                       R, R_inv, sigmaH, sigmaG, m_C, Sigma_C, mu_Mc, sigma_Mc,
                       m_G, Sigma_G, AuxG, q_Z, neighboursIndexes, Beta, Gamma,
                       gamma, gamma_h, gamma_g, sigma_eps, XX, W, 
                       J, D, M, N, K, hyp, Gamma_X, Gamma_WX, plot=False, 
                       bold=False):
    # Entropy
    EntropyA = RL_Entropy(Sigma_A, M, J)
    EntropyC = RL_Entropy(Sigma_C, M, J)
    EntropyH = RF_Entropy(Sigma_H, D)
    EntropyG = RF_Entropy(Sigma_G, D)
    EntropyQ = Q_Entropy(q_Z, M, J)
    if bold:
        Total_Entropy = EntropyA + EntropyH + EntropyQ        
    else:
        Total_Entropy = EntropyA + EntropyH + EntropyC + EntropyG + EntropyQ        
    
    # Likelihood
    EPtildeLikelihood = expectation_Ptilde_Likelihood(y_tilde, m_A, Sigma_A, m_H, Sigma_H, m_C,
                                                      Sigma_C, m_G, Sigma_G, XX, W, sigma_eps,
                                                      Gamma, J, D, M, N, Gamma_X, Gamma_WX)
    EPtildeA = RL_expectation_Ptilde(m_A, Sigma_A, mu_Ma, sigma_Ma, q_Z)
    EPtildeC = RL_expectation_Ptilde(m_C, Sigma_C, mu_Mc, sigma_Mc, q_Z)
    EPtildeH = RF_expectation_Ptilde(AuxH, Sigma_H, sigmaH, R, R_inv, D)
    EPtildeG = RF_expectation_Ptilde(AuxG, Sigma_G, sigmaG, R, R_inv, D)
    EPtildeQ = Q_expectation_Ptilde(q_Z, neighboursIndexes, Beta, gamma, K, M)
    EPtildeBeta = M * np.log(gamma) - gamma * Beta.sum()
    if hyp:
        EPtildeVh = np.log(gamma_h) - gamma_h * sigmaH
        EPtildeVg = np.log(gamma_g) - gamma_g * sigmaG
    else:
        EPtildeVh = 0.
        EPtildeVg = 0.
    if bold:
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeQ \
                + EPtildeBeta + EPtildeVh
    else:
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeC + EPtildeG + EPtildeQ \
                + EPtildeBeta + EPtildeVh + EPtildeVg
    if plot:
        print 'Total_Entropy = ', Total_Entropy
        print 'EA = ', EntropyA, 'EH = ', EntropyH, 'EC = ', EntropyC, \
                'EG = ', EntropyG, 'EQ = ', EntropyQ
        print 'Total_EPtilde = ', EPtilde
        print 'ELklh = ', EPtildeLikelihood, 'EPtA = ', EPtildeA, 'EPtC = ', EPtildeC, \
                'EPtH = ', EPtildeH, 'EPtG = ', EPtildeG, 'EPtQ = ', EPtildeQ, \
                'EPtBeta = ', EPtildeBeta, 'EPtVh = ', EPtildeVh, 'EPtVg = ', EPtildeVg

    return EPtilde, EPtildeLikelihood, Total_Entropy, EPtilde + Total_Entropy

