# -*- coding: utf-8 -*-
"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

import numpy as np
import vem_tools as vt
from collections import OrderedDict

eps = 1e-6


def expectation_H(Sigma_A, m_A, m_C, G, XX, W, Gamma, D, J, N, y_tilde,
                  sigma_eps, scale, R_inv, sigmaH, sigmaG, 
                  prior_mean_term, prior_cov_term):
    """
    Expectation-H step: 
    p_H = argmax_h(E_pa,pc,pg[log p(h|y, a, c, g; theta)])
        \propto exp(E_pa,pc,pg[log p(y|h, a, c, g; theta) + log p(h; sigmaH)])

    Returns:
    m_H, Sigma_H of probability distribution p_H of the current iteration
    """

    ## Precomputations
    mAX = np.tensordot(m_A, XX, axes=(1, 0))                # shape (J, N, D)
    Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    X_Gamma_X = np.tensordot(XX.T, Gamma_X, axes=(1, 0))    # shape (D, M, M, D)
    cov_noise = np.maximum(sigma_eps, eps)[:, np.newaxis, np.newaxis]
    mAX_Gamma = (np.tensordot(mAX, Gamma, axes=(1, 0)) / cov_noise) # shape (J, D, N)

    ## Sigma_H computation
    # first summand: part of the prior -> R^-1 / sigmaH
    S_a = R_inv / sigmaH
    # second summand: E_pa[Saj.T*Gamma*Saj]
    # sum_{m, m'} Sigma_a(m,m') X_m.T Gamma_i X_m'
    S_a += (np.einsum('ijk,lijm->klm', Sigma_A, X_Gamma_X) / cov_noise).sum(0)
    # third summand: E_pa[Saj.T*Gamma*Saj]
    # (sum_m m_a X_m).T Gamma_i (sum_m m_a X_m)
    for i in xrange(0, J):
        S_a += mAX_Gamma[i, :, :].dot(mAX[i, :, :])  #option 1 faster 13.4
    #S_a += np.einsum('ijk,ikl->ijl', mAX_Gamma, mAX).sum(0) # option 2 second 8.8
    #S_a += np.einsum('ijk,ikl->jl', mAX_Gamma, mAX) # option 3 slower 7.5
    # forth summand (depends on prior type): 
    # we sum the term that corresponds to the prior
    S_a += prior_cov_term

    # Sigma_H = S_a^-1
    Sigma_H = np.linalg.inv(S_a)

    ## m_H
    # Y_bar_tilde computation: (sum_m m_a X_m).T Gamma_i y_tildeH
    Y_bar_tilde = np.einsum('ijk,ki->j', mAX_Gamma, y_tilde)
    # we sum the term that corresponds to the prior
    Y_bar_tilde += prior_mean_term
    
    # m_H = S_a^-1 y_bar_tilde
    m_H = np.dot(np.linalg.inv(S_a), Y_bar_tilde)

    return m_H, Sigma_H


def expectation_A(H, m_A, G, m_C, W, XX, Gamma, q_Z, mu_Ma, sigma_Ma,
                      D, J, M, K, y_tilde, Sigma_A, Sigma_H, sigma_eps):
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
    Sigma_H_X = XX.dot(Sigma_H.T)
    Gamma_X = np.tensordot(Gamma, XX, axes=(1, 1))
    sigma_eps_m = np.maximum(sigma_eps, eps)

    ## Sigma_A computation
    # first summand of Sigma_A: XH.T*Gamma*XH / sigma_eps
    Sigma_A = XH.T.dot(Gamma).dot(XH)[..., np.newaxis] / sigma_eps_m
    # second summand of Sigma_A: tr(X.T*Gamma*X*Sigma_H / sigma_eps)
    second_summand = np.einsum('ijk, jlk', Sigma_H_X, Gamma_X)
    Sigma_A += second_summand[..., np.newaxis] / sigma_eps_m
    # third summand of Sigma_A: part of p(a|q; theta_A)
    Delta_k = (q_Z / np.maximum(sigma_Ma[:, :, np.newaxis], eps))
    Delta = Delta_k.sum(axis=1)         # sum across classes K
    for i in xrange(0, J):
        Sigma_A[:, :, i] = np.linalg.inv(Sigma_A[:, :, i] + np.diag(Delta[:, i]))
    
    ## m_A computation
    Gamma_y_tildeH = Gamma.dot(y_tilde).T
    X_tildeH = Gamma_y_tildeH.dot(XH) / sigma_eps_m[:, np.newaxis] \
               + (Delta_k * mu_Ma[:, :, np.newaxis]).sum(axis=1).T
    # dot product across voxels of Sigma_A and X_tildeH
    m_A = np.einsum('ijk,kj->ki', Sigma_A, X_tildeH)

    return m_A, Sigma_A


def expectation_Z(Sigma_A, m_A, sigma_M, mu_M, Beta0, Z_tilde, q_Z, graph, M, J, K):
    local_energy = np.zeros((J, M, K),dtype=float)
    
    alpha = -0.5 * np.diagonal(Sigma_A)[:, :, np.newaxis] / (sigma_M[np.newaxis, :, :])   # (J, M, K)
    alpha -= alpha.max(axis=2)[:, :, np.newaxis]    
    Gauss_mat = normpdf(m_A[...,np.newaxis], mu_M, np.sqrt(sigma_M))

    # Update Ztilde ie the quantity which is involved in the a priori 
    # Potts field [by solving for the mean-field fixed point Equation]

    # TODO: decide if we take out the computation of p_q_t or Ztilde
    B_Ztilde = Beta[..., np.newaxis] * Z_tilde.copy()
    for i in xrange(0, J):
        local_energy[i, :, :] = B_Ztilde[:, :, graph[i]].sum(axis=2)
    energy = (alpha + local_energy)
    Probas = (np.exp(energy) * Gauss_mat).transpose(1, 2, 0)
    Z_tilde = Probas / Probas.sum(axis=1)[:, np.newaxis, :]
    
    B_Ztilde = Beta[..., np.newaxis] * Z_tilde.copy()
    for i in xrange(0, J):
        local_energy[i, :, :] = B_Ztilde[:, :, graph[i]].sum(axis=2)
    energy = (alpha + local_energy)
    Probas = (np.exp(energy) * Gauss_mat).transpose(1, 2, 0)
    q_Z = Probas / Probas.sum(axis=1)[:, np.newaxis, :]
    
    return q_Z, Z_tilde


def maximization_sigma_noise(XX, m_A, m_H, Sigma_H, Sigma_A, y_tilde):
    """
    Maximization sigma_noise
    """
    # Precomputations
    XH = XX.dot(m_H)                                        # shape (N, M)
    mAXH = m_A.dot(XH)                                      # shape (J, N)
    HXXH = XH.dot(XH.T)          # (,D)*(D, N, M)*(M, N, D)*(D,) -> (M, M)
    Sigma_H_X = XX.dot(Sigma_H.T)                           # shape (M, N, D)
    Sigma_H_X_X = np.einsum('ijk, jlk', Sigma_H_X, XX.transpose(1, 0, 2))
    
    Htilde = HXXH + Sigma_H_X_X                             # shape (M, M)

    # trace(Sigma_A (H.T X.T X H + Sigma_H X.T X)) in each voxel
    tr_SA_Htilde = np.einsum('ijk,ji->k', Sigma_A, Htilde)
    
    # m_A * (H.T X.T X H + Sigma_H X.T X)
    mA_Htilde = m_A.dot(Htilde)

    # m_A.T * (H.T X.T X H + Sigma_H X.T X) * m_A
    mA_Htilde_mA = np.einsum('ij,ij->i', m_A.dot(Htilde), m_A)

    # m_A X H y_tilde
    mAXH_ytilde = np.einsum('ij,ji->i', mAXH, y_tilde)

    # y_tilde.T y_tilde
    ytilde_ytilde = np.einsum('ij,ij->j',y_tilde, y_tilde)

    sigma_eps = mA_Htilde_mA + tr_SA_Htilde + ytilde_ytilde - 2 * mAXH_ytilde

    return (mA_Htilde_mA + tr_SA_Htilde + ytilde_ytilde - 2 * mAXH_ytilde) / N

