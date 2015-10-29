# -*- coding: utf-8 -*-
"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

#import os.path as op
import numpy as np
from numpy.matlib import *
#import scipy as sp
#from scipy.linalg import toeplitz
#import time
#import UtilsC
#import pyhrf
#from pyhrf.boldsynth.hrf import getCanoHRF
#from pyhrf.paradigm import restarize_events
#from pyhrf.tools import format_duration
import vem_tools as vt
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

# Tools
##############################################################

eps = 1e-6


def error(estimated, simulated):
    e = np.mean(np.mean((estimated - simulated) ** 2) / \
                    np.mean(simulated ** 2))
    return e


def mult(v1, v2):
    matrix = np.zeros((len(v1), len(v2)), dtype=float)
    for i in xrange(len(v1)):
        for j in xrange(len(v2)):
                matrix[i, j] += v1[i] * v2[j]
    return matrix


def normpdf(x, mu, sigma):
    u = (x - mu) / np.abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-u * u / 2)
    return y


def covariance_matrix(order, D, dt):
    D2 = vt.buildFiniteDiffMatrix_s(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    return R


def create_conditions(Onsets, M, N, D, TR, dt):
    condition_names = []
    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
        
        if 0:
            import matplotlib.pyplot as plt
            print 'printing matrices... '
            plt.matshow(np.array(X[condition]))        
            #print condition+', TR = '+str(TR)+', dt = '+str(dt)
            #plt.title('cond '+str(i)+', dt = '+str(dt))
            #print 'title fait'
            plt.savefig('./'+condition+'_dt'+str(dt)+'_old_block.png')
            plt.close()
            #plt.show()
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    return X, XX, condition_names


def create_conditions_block(Onsets, durations, M, N, D, TR, dt):
    condition_names = []
    X = OrderedDict([])
    i = 0
    for condition, Ons in Onsets.iteritems():
        Dur = durations[condition]
        X[condition] = vt.compute_mat_X_2_block(N, TR, D, dt,
                                                Ons, durations=Dur)
        if 0:
            import matplotlib.pyplot as plt
            from matplotlib.pylab import *
            plt.matshow(np.array(X[condition]))
            plt.title('cond '+str(i)+', dt = '+str(dt))
            plt.savefig('./'+condition+'_dt'+str(dt)+'_old.png')
            plt.close()
            #plt.show()
            i = i + 1
        condition_names += [condition]
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    return X, XX, condition_names


def create_neighbours(graph, J):
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]
    return maxNeighbours, neighboursIndexes

def _trap_area( p1, p2 ):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2

    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs( p2[ 0 ] - p1[ 0 ] )
    avg_ht = ( p1[ 1 ] + p2[ 1 ] ) / 2.0

    return base * avg_ht
    
def roc_curve(dvals, labels, rocN=None, normalize=True):
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1}

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1:
        rocN = int(rocN * np.sum(np.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives

    fpc = [0.0]  # fp coordinates
    tpc = [0.0]  # tp coordinates
    dv_prev = - np.inf  # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    #num_pos = labels.count(1)  # num pos labels np.sum(labels)
    #num_neg = labels.count(0)  # num neg labels np.prod(labels.shape)-num_pos

    #if num_pos == 0 or num_pos == len(labels):
    #    raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = np.argsort(dvals)[::-1]

    for idx in indices:
        # increment associated TP/FP count
        if labels[idx] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN:
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[idx] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1]:
                tpc[-1] = TP
            else:
                fpc.append(FP)
                tpc.append(TP)
            dv_prev = dvals[idx]
            area += _trap_area((FP_prev, TP_prev), (FP, TP))
            FP_prev = FP
            TP_prev = TP

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )
    if normalize:
        fpc = [np.float64(x) / FP for x in fpc]
        if TP > 0:
            tpc = [np.float64(x) / TP for x in tpc]
        if area > 0:
            area /= (TP * FP)

    return fpc, tpc, area


# Expectation functions
##############################################################

def expectation_A(H, m_A, G, m_C, W, X, Gamma, q_Z, mu_Ma, sigma_Ma,
                  D, J, M, K, y_tilde, Sigma_A, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            # from X, we take k1=cond_name, m=index_cond
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k1]), G)

            for m2, k2 in enumerate(X):
                Sigma_A[m, m2, i] = np.dot(np.dot(np.dot(np.dot(H.T, \
                                                 X[k1].T), Gamma_i), X[k2]), H)
            X_tilde[i, m, :] = np.dot(np.dot(Gamma_i, y_tildeH[:, i]).T, X[k1])
        tmp = np.dot(X_tilde[i, :, :], H)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Ma[:, k] + eps))
            tmp += np.dot(Delta, mu_Ma[:, k])
            Sigma_A[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_A[:, :, i])
        Sigma_A[:, :, i] = tmp2
        m_A[i, :] = np.dot(Sigma_A[:, :, i], tmp)
    return m_A, Sigma_A


def expectation_A_s(H, m_A, G, m_C, W, X, Gamma, q_Z, mu_Ma, sigma_Ma,
                  D, J, M, K, y_tilde, Sigma_A, Sigma_H, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            # from X, we take k1=cond_name, m=index_cond
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k1]), G)

            for m2, k2 in enumerate(X):
                Sigma_A[m, m2, i] = np.dot(np.dot(np.dot(np.dot(H.T, \
                                                 X[k1].T), Gamma_i), X[k2]), H)
                Sigma_A[m, m2, i] += (np.dot(np.dot(np.dot(Sigma_H, X[k1].T),
                                                    Gamma_i), X[k2])).trace()
            X_tilde[i, m, :] = np.dot(np.dot(Gamma_i, y_tildeH[:, i]).T, X[k1])
        tmp = np.dot(X_tilde[i, :, :], H)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Ma[:, k] + eps))
            tmp += np.dot(Delta, mu_Ma[:, k])
            Sigma_A[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_A[:, :, i])
        Sigma_A[:, :, i] = tmp2
        m_A[i, :] = np.dot(Sigma_A[:, :, i], tmp)
    return m_A, Sigma_A


def expectation_A_s(H, m_A, G, m_C, W, X, Gamma, q_Z, mu_Ma, sigma_Ma,
                  D, J, M, K, y_tilde, Sigma_A, Sigma_H, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            # from X, we take k1=cond_name, m=index_cond
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k1]), G)

            for m2, k2 in enumerate(X):
                Sigma_A[m, m2, i] = np.dot(np.dot(np.dot(np.dot(H.T, \
                                                 X[k1].T), Gamma_i), X[k2]), H)
                Sigma_A[m, m2, i] += (np.dot(np.dot(np.dot(Sigma_H, X[k1].T),
                                                    Gamma_i), X[k2])).trace()
            X_tilde[i, m, :] = np.dot(np.dot(Gamma_i, y_tildeH[:, i]).T, X[k1])
        tmp = np.dot(X_tilde[i, :, :], H)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Ma[:, k] + eps))
            tmp += np.dot(Delta, mu_Ma[:, k])
            Sigma_A[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_A[:, :, i])
        Sigma_A[:, :, i] = tmp2
        m_A[i, :] = np.dot(Sigma_A[:, :, i], tmp)
    return m_A, Sigma_A


def expectation_C(G, m_C, H, m_A, W, X, Gamma, q_Z, mu_Mc, sigma_Mc,
                  D, J, M, K, y_tilde, Sigma_C, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k1], H)
            for m2, k2 in enumerate(X):
                Sigma_C[m, m2, i] = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(\
                                  G.T, X[k1].T), W.T), Gamma_i), W), X[k2]), G)
            X_tilde[i, m, :] = np.dot(np.dot(np.dot(Gamma_i, \
                                    y_tildeG[:, i]).T, W) ,X[k1])
        tmp = np.dot(X_tilde[i, :, :], G)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Mc[:, k] + eps))
            tmp += np.dot(Delta, mu_Mc[:, k])
            Sigma_C[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_C[:, :, i])
        Sigma_C[:, :, i] = tmp2
        m_C[i, :] = np.dot(Sigma_C[:, :, i], tmp)
    return m_C, Sigma_C


def expectation_C_s(G, m_C, H, m_A, W, X, Gamma, q_Z, mu_Mc, sigma_Mc,
                  D, J, M, K, y_tilde, Sigma_C, Sigma_G, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k1], H)
            for m2, k2 in enumerate(X):
                Sigma_C[m, m2, i] = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(\
                                  G.T, X[k1].T), W.T), Gamma_i), W), X[k2]), G)
                Sigma_C[m, m2, i] += (np.dot(np.dot(np.dot(Sigma_G, X[k1].T),
                                                    Gamma_i), X[k2])).trace()
            X_tilde[i, m, :] = np.dot(np.dot(np.dot(Gamma_i, \
                                    y_tildeG[:, i]).T, W) ,X[k1])
        tmp = np.dot(X_tilde[i, :, :], G)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Mc[:, k] + eps))
            tmp += np.dot(Delta, mu_Mc[:, k])
            Sigma_C[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_C[:, :, i])
        Sigma_C[:, :, i] = tmp2
        m_C[i, :] = np.dot(Sigma_C[:, :, i], tmp)
    return m_C, Sigma_C



def expectation_H(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R, sigmaH, mu_term=0):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R / sigmaH
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_a += Sigma_A[m1, m2, i] * np.dot(np.dot(X[k1].T,
                                                          Gamma_i), X[k2])
    Sigma_H = np.linalg.inv(S_a)
    m_H = np.dot(Sigma_H, Y_bar_tilde + mu_term)
    return m_H, Sigma_H


def expectation_H_physiob(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R, sigmaH, sigmaG, Omega):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R / sigmaH
    y_tildeH = y_tilde.copy()
    print m_A.shape
    print Sigma_A.shape
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        #S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)
        tmp = np.zeros((N, D), dtype=float)
        tmp2 = np.zeros((N, D), dtype=float)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                tmp += m_A[i, m1] * X[k1]
                tmp2 += m_A[i, m2] * X[k2]
                S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp2)
                S_a += Sigma_A[m1, m2, i] * np.dot(np.dot(X[k1].T,
                                                          Gamma_i), X[k2])
    S_a += np.dot(np.dot(Omega.T, scale * R / sigmaG), Omega)
    Y_bar_tilde += np.dot(np.dot(Omega.T, scale * R / sigmaG), G)
    Sigma_H = np.linalg.inv(S_a)
    m_H = np.dot(Sigma_H, Y_bar_tilde)
    return m_H, Sigma_H


def expectation_H_physio(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R_inv, sigmaH, sigmaG, Omega):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R_inv / sigmaH
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_a += Sigma_A[m, m2, i] * np.dot(np.dot(X[k].T, Gamma_i), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)            
    S_a += np.dot(np.dot(Omega.T, scale * R_inv / sigmaG), Omega)
    Y_bar_tilde += np.dot(np.dot(Omega.T, scale * R_inv / sigmaG), G)
    return np.dot(np.linalg.inv(S_a), Y_bar_tilde), np.linalg.inv(S_a)


def expectation_H_balloon(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R_inv, sigmaH, sigmaG, Hb):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R_inv / sigmaH
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_a += Sigma_A[m, m2, i] * np.dot(np.dot(X[k].T,
                                                          Gamma_i), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)      
    Y_bar_tilde += np.dot(scale * R_inv / sigmaH, Hb)
    return np.dot(np.linalg.inv(S_a), Y_bar_tilde), np.linalg.inv(S_a)


def expectation_G_balloon(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R_inv, sigmaG, Gb):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R_inv / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_C[i, m] * np.dot(W, X[k])
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_c += Sigma_C[m, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k].T, W.T), Gamma_i), W), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(tmp.T, Gamma_i), tmp)            
    Y_bar_tilde += np.dot(R_inv / sigmaG, Gb)
    return np.dot(np.linalg.inv(S_c), Y_bar_tilde), np.linalg.inv(S_c)

#@profile
def expectation_H_prior_copy(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                        sigma_epsilone, scale, R_inv, sigmaH, sigmaG, 
                        prior_mean_term, prior_cov_term):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R_inv / sigmaH
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_a += Sigma_A[m, m2, i] * np.dot(np.dot(X[k].T,
                                                          Gamma_i), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)      
    S_a += prior_cov_term
    Y_bar_tilde += prior_mean_term
    return np.dot(np.linalg.inv(S_a), Y_bar_tilde), np.linalg.inv(S_a)


#@profile
def expectation_H_prior(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                        sigma_epsilone, scale, R_inv, sigmaH, sigmaG, 
                        prior_mean_term, prior_cov_term):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = R_inv / sigmaH
    y_tildeH = y_tilde.copy()
    y_tildeH -= np.dot(np.dot(W, np.dot(X, G).T), m_C.T)
    #print 'y_tildeH shape is ', y_tildeH.shape
    
    Gamma1 = Gamma[:, :, None] * sigma_epsilone[None, None, :]
    print 'Gamma1 shape is ', Gamma1.shape
    
    print 'X shape is ', X.shape
    Xs = np.swapaxes(X, 1, 2).T
    aux0 = np.tensordot(Xs.T, Gamma1, axes=([-1], [0]))
    print aux0.shape
    aux1 = np.swapaxes(np.swapaxes(aux0, 3, 2), 2, 1)
    aux2 = np.swapaxes(np.tensordot(aux1, Xs, axes=([3], [0])), 1, 4)

    S_a += np.sum(np.sum(np.sum(Sigma_A[:, :, None, None, :] * aux2, 0), 0), 2)
    
    maX = np.sum(m_A[:, :, None, None] * X[None, :, :, :], 1)
    #print 'maX shape is ', maX.shape
    #print np.tensordot(maX.T, Gamma1.T, axes=([2, 1],[0, 1])).shape
    #maXs = np.swapaxes(maX, 1, 2)
    #print 'mAXs shape is ', maXs.shape
    #maXG = np.tensordot(maXs, Gamma1, axes=([-1],[0]))
    #print 'mAXG shape is ', maXG.shape
    #print np.tensordot(maXG, maXs.T, axes=([-1],[0])).shape
    #S_a += np.sum(np.tensordot(maXG, maXs.T, axes=([-1],[0])), 2) 
    #print (maX.T[:, None, :, :] * Gamma1[None, :, :, :]).shape
    #print np.tensordot(maX, Gamma1, axes=([1],[0])).shape
    #print np.tensordot(np.tensordot(maX, Gamma1, axes=([1],[0])), maX, axes=([],[]))
    #S_a += np.tensordot(maX, Gamma1, axes=([1],[0]))
    """import time
    t0 = time.time()
    #aaa =  np.sum(maX.T[:, None, :, :] * Gamma1[None, :, :, :], 2)
    aaa =  np.tensordot(maX.T, Gamma1.T, axes=([-1, -2], [0, 1]))
    print time.time() - t0
    t0 = time.time()
    bbb = np.tensordot( aaa, maXs.T, axes=([-1], [0]))
    print time.time() - t0"""
    S_a0 = np.tensordot(np.sum(maX.T[:, None, :, :] * Gamma1[None, :, :, :], 2),
                        maX, axes=([-1, -2], [0, 1]))
    #S_a += np.sum(np.tensordot(aaa, maXs.T, axes=([-1], [0])), 2)
    #print np.sum(maX.T[:, None, :, :] * Gamma1[None, :, :, :], 2).shape
    #print np.sum(np.sum(maX.T[:, None, :, :] * Gamma1[None, :, :, :], 2)* maX.T, 1).shape
    #S_a += np.sum(np.sum(maX.T[:, None, :, :] * Gamma1[None, :, :, :], 2)* maX.T, 1)
    #print 'Y_bar_tilde shape is ', Y_bar_tilde.shape
    #print 'y_tildeH shape is ', y_tildeH.shape
    Y_bar_tilde += np.sum(np.sum(maX * y_tildeH.T[:, :, None], 0), 0)
    S_a1 = np.zeros((D, D))
    for i in xrange(0, J):
        Y_bar_tilde += np.dot(np.dot(maX[i, :, :].T, Gamma1[:, :, i]), y_tildeH[:, i])
        S_a1 += np.dot(np.dot(maX[i, :, :].T, Gamma1[:, :, i]), maX[i, :, :])
        print maX.shape, Gamma1.shape
        #assert 0
    print 'error = ', np.abs(S_a0 - S_a1).max()
    print 'error =', np.sum(S_a0 - S_a1)
    S_a += S_a1
    S_a += prior_cov_term
    Y_bar_tilde += prior_mean_term
    return np.dot(np.linalg.inv(S_a), Y_bar_tilde), np.linalg.inv(S_a)


def expectation_G_prior(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                        sigma_epsilone, scale, R_inv, sigmaG, prior_mean_term,
                        prior_cov_term):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R_inv / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_C[i, m] * np.dot(W, X[k])
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_c += Sigma_C[m, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k].T, W.T), Gamma_i), W), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(tmp.T, Gamma_i), tmp)            
    S_c += prior_cov_term
    Y_bar_tilde += prior_mean_term
    return np.dot(np.linalg.inv(S_c), Y_bar_tilde), np.linalg.inv(S_c)

    
def expectation_G(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R, sigmaG, mu_term=0):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp0 = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            #tmp += m_C[i, m] * np.dot(W, X[k])
            tmp0 += m_C[i, m] * X[k]
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
        Y_bar_tilde += np.dot(np.dot(np.dot(tmp0.T, W.T), Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(np.dot(np.dot(tmp0.T, W.T), Gamma_i), W), tmp0)
        #tmp = np.zeros((N, D), dtype=float)
        #tmp2 = np.zeros((N, D), dtype=float)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                #tmp = m_C[i, m1] * X[k1]
                #tmp2 = m_C[i, m2] * X[k2]
                #S_c += np.dot(np.dot(np.dot(np.dot(tmp.T, W.T), Gamma_i), W), tmp2)
                S_c += Sigma_C[m1, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k1].T, W.T), Gamma_i), W), X[k2])
    Sigma_G = np.linalg.inv(S_c)
    m_G = np.dot(Sigma_G, Y_bar_tilde + mu_term)
    return m_G, Sigma_G


def expectation_G_b(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R_inv, sigmaG):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R_inv / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            #tmp += m_C[i, m] * np.dot(W, X[k])
            tmp += m_C[i, m] * X[k]
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
        Y_bar_tilde += np.dot(np.dot(np.dot(tmp.T, W.T), Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(np.dot(np.dot(tmp.T, W.T), Gamma_i), W), tmp)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_c += Sigma_C[m1, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k1].T, W.T), Gamma_i), W), X[k2])
    Y_bar_tilde += np.dot(scale * R / sigmaG, OmegaH)
    Sigma_G = np.linalg.inv(S_c)
    m_G = np.dot(Sigma_G, Y_bar_tilde)
    return m_G, Sigma_G
    

def expectation_G_physio(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R_inv, sigmaG, OmegaH):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R_inv / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_C[i, m] * np.dot(W, X[k])
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_c += Sigma_C[m, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k].T, W.T), Gamma_i), W), X[k2])
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(tmp.T, Gamma_i), tmp)            
    #print 'Y_bar_tilde part signal: ', Y_bar_tilde
    Y_bar_tilde += np.dot(scale * R_inv / sigmaG, OmegaH)
    #print 'Y_bar_tilde part prior: ', Y_bar_tilde
    return np.dot(np.linalg.inv(S_c), Y_bar_tilde), np.linalg.inv(S_c)


def constraint_norm1_b(Ftilde, Sigma_F, positivity=False, perfusion=None):
    """ Constrain with optimization strategy """
    from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp
    Sigma_F_inv = np.linalg.inv(Sigma_F)
    zeros_F = np.zeros_like(Ftilde)
    
    def fun(F):
        'function to minimize'
        return np.dot(np.dot((F - Ftilde).T, Sigma_F_inv), (F - Ftilde))

    def grad_fun(F):
        'gradient of the function to minimize'
        return np.dot(Sigma_F_inv, (F - Ftilde))

    def fung(F):
        'function to minimize'
        mean = np.dot(np.dot((F - Ftilde).T, Sigma_F_inv), (F - Ftilde)) #* 0.5
        Sigma = np.dot(Sigma_F_inv, (F - Ftilde))
        return mean #, Sigma

    def ec1(F):
        'Norm2(F)==1'
        return - 1 + np.linalg.norm(F, 2)
    
    def ec2(F):
        'F(0)==0'
        return F[0]
    
    def ec3(F):
        'F(end)==0'
        return F[-1]
    
    if positivity:
        def ec0(F):
            'F>=0 ?? or F>=-baseline??'
            return F
        y = fmin_slsqp(fun, Ftilde, eqcons=[ec1, ec2, ec3], ieqcons=[ec0],
                       bounds=[(None, None)] * (len(zeros_F)))
        #y = fmin_slsqp(fun, zeros_F, eqcons=[ec1], ieqcons=[ec2],
        #               bounds=[(None, None)] * (len(zeros_F)))
        #y = fmin_l_bfgs_b(fung, zeros_F, bounds=[(-1, 1)] * (len(zeros_F)))
    else:
        #y = fmin_slsqp(fun, Ftilde, eqcons=[ec1],
        #               bounds=[(None, None)] * (len(zeros_F)))
        y = fmin_slsqp(fun, Ftilde, eqcons=[ec1, ec2, ec3], 
                       bounds=[(None, None)] * (len(zeros_F)))
        #print fmin_l_bfgs_b(fung, Ftilde, bounds=[(-1, 1)] * (len(zeros_F)), approx_grad=True)
        #y = fmin_l_bfgs_b(fun, Ftilde, fprime=grad_fun, 
        #                  bounds=[(-1, 1)] * (len(zeros_F)))[0] #, approx_grad=True)[0]
    return y


def expectation_Q(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, p_q_t, p_Q, graph, M, J, K):
    energy = np.zeros(K)
    Gauss = energy.copy()
    # Compute p_q_t
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            alpha /= alpha.mean()
            p_q_neighb = sum(p_q_t[m, :, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[k] \
                            + max(np.log(normpdf(m_A[i, m], mu_Ma[m, k],
                                         sigma_Ma[m, k]) + eps), -100)\
                            + max(np.log(normpdf(m_C[i, m], mu_Mc[m, k],
                                         sigma_Mc[m, k]) + eps), -100)
                local_energy = Beta[m] * p_q_neighb[k]
                energy[k] = extern_field + local_energy
            Probas = np.exp(energy - max(energy))
            p_q_t[m, :, i] = Probas / (sum(Probas) + eps)
    # Compute p_Q
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            alpha /= alpha.mean()
            p_q_neighb = sum(p_q_t[m, :, graph[i]], 0)
            for k in xrange(0, K):
                # variances term
                extern_field = alpha[k]
                # Beta depndent term
                local_energy = Beta[m] * p_q_neighb[k]
                energy[k] = extern_field + local_energy
                Gauss[k] = normpdf(m_A[i, m], mu_Ma[m, k], \
                                    np.sqrt(sigma_Ma[m, k])) \
                         * normpdf(m_C[i, m], mu_Mc[m, k], \
                                    np.sqrt(sigma_Mc[m, k]))
            Probas = np.exp(energy - max(energy))
            p_Q[m, :, i] = Gauss * Probas / (sum(Probas) + eps)
            p_Q[m, :, i] /= (sum(p_Q[m, :, i]) + eps)
    return p_Q, p_q_t


# Maximization functions
##############################################################

def maximization_mu_sigma_original(Mu, Sigma, q_Z, m_X, K, M, Sigma_X):
    for m in xrange(0, M):
        for k in xrange(0, K):
            #S = sum( q_Z[m,k,:] ) + eps
            S = sum(q_Z[m, k, :])
            if S == 0.:
                S = eps
            Sigma[m, k] = sum(q_Z[m, k, :] * (pow(m_X[:, m] - Mu[m, k], 2) +
                                    Sigma_X[m, m, :])) / S
            #Sigma[m, k] = sum(q_Z[m, k, :] * (pow(m_X[:, m] - Mu[m, k], 2))) / S
            if Sigma[m, k] < eps:
                Sigma[m, k] = eps
            if k != 0:          # mu_0 = 0 a priori
                # Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
                Mu[m, k] = sum(q_Z[m, k, :] * m_X[:, m]) / S
            else:
                Mu[m, k] = 0.
    return Mu, Sigma


def maximization_mu_sigma(Mu, Sigma, q_Z, m_X, K, M, Sigma_X):
    for m in xrange(0, M):
        for k in xrange(0, K):
            S = sum(q_Z[m, k, :]) + eps
            Sigma[m, k] = eps + sum(q_Z[m, k, :] * (pow(m_X[:, m] - Mu[m, k], 2) +
                                    Sigma_X[m, m, :])) / (S + eps)
            if k != 0:          # mu_0 = 0 a priori
                Mu[m, k] = eps + sum(q_Z[m, k, :] * m_X[:, m]) / (S + eps)
            else:
                Mu[m, k] = 0.
    return Mu, Sigma


def maximization_L_alpha(Y, m_A, m_C, X, W, w, Ht, Gt, L, P, alpha, Gamma,
                         sigma_eps):
    # WARNING! Noise missing, but if Gamma = Identity, it is equivalent
    for i in xrange(0, Y.shape[1]):
        Gamma_i = Gamma / max(sigma_eps[i], eps)
        S = np.zeros((P.shape[0]), dtype=np.float64)  # zerosP.copy()
        S1 = S.copy()
        S2 = S.copy()
        for m, k in enumerate(X):
            S += m_A[i, m] * np.dot(X[k], Ht)
            S += m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        S1 += S + np.dot(w, alpha[i])
        S2 += S + np.dot(P, L[:, i])
        term = np.linalg.inv(np.dot(np.dot(P.T, Gamma_i), P))
        print term.shape
        print P.T.shape
        print Y[:, i].shape
        print S1.shape
        L[:, i] = np.dot(np.dot(np.dot(term, P.T), Gamma_i), Y[:, i] - S1)
        #L[:, i] = np.dot(P.T, Y[:, i] - S1)
        alpha[i] = np.dot(w.T, Y[:, i] - S2) / (np.dot(w.T, w))
        #AL[:, i] = np.dot(np.dot(np.dot(term, WP.T), Gamma_i), Y[:, i] - S)
    return L, alpha


def maximization_LA(Y, m_A, m_C, X, W, w, Ht, Gt, L, P, alpha, Gamma,
                    sigma_eps):
    AL = np.append(alpha[np.newaxis, :], L, axis=0)
    WP = np.append(w[:, np.newaxis], P, axis=1)
    for i in xrange(0, Y.shape[1]):
        Gamma_i = Gamma / max(sigma_eps[i], eps)
        S = np.zeros((WP.shape[0]), dtype=np.float64)  # zerosP.copy()
        for m, k in enumerate(X):
            S += m_A[i, m] * np.dot(X[k], Ht)
            S += m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        term = np.linalg.inv(np.dot(np.dot(WP.T, Gamma_i), WP))
        AL[:, i] = np.dot(np.dot(np.dot(term, WP.T), Gamma_i), Y[:, i] - S)
        L[:, i] = AL[1:, i]
        alpha[i] = AL[0, i]
    return AL #L, alpha


def maximization_sigma(D, R_inv, m_X):
    sigmaX = (np.dot(mult(m_X, m_X), R_inv)).trace()
    sigmaX /= (D - 1)
    return sigmaX


def maximization_sigma_prior(D, R_inv, m_X, gamma_x):
    alpha = (np.dot(mult(m_X, m_X), R_inv)).trace()
    #sigmaH = (D + sqrt(D * D + 8 * gamma_h * alpha)) / (4* gamma_h)
    print 'trace value = ', alpha
    sigmaX = (1 - D + sqrt((D - 1) * (D - 1) + 8 * gamma_x * alpha)) / (4 * gamma_x)
    return sigmaX


def maximization_sigma_noise(Y, X, m_A, Sigma_A, Ht, m_C, Sigma_C, Gt, W, \
                             M, N, J, y_tilde, sigma_eps):
    hXXh = np.zeros((M, M), dtype=float)
    gXXg = hXXh.copy()
    gXXh = hXXh.copy()
    for i in xrange(0, J):
        S = np.zeros((N), dtype=float)
        for m, k in enumerate(X):
            for m2, k2 in enumerate(X):
                hXXh[m, m2] = np.dot(np.dot(np.dot(Ht.T, X[k].T), X[k2]), Ht)
                gXXg[m, m2] = np.dot(np.dot(np.dot(np.dot(np.dot(Gt.T, X[k].T),
                                                          W.T), W), X[k2]), Gt)
                gXXh[m, m2] = np.dot(np.dot(np.dot(np.dot( \
                                               Gt.T, X[k].T), W.T), X[k2]), Ht)
            S += m_A[i, m] * np.dot(X[k], Ht) + \
                 m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        sigma_eps[i] = np.dot(np.dot(m_A[i, :].T, hXXh), m_A[i, :])
        sigma_eps[i] += (np.dot(Sigma_A[:, :, i], hXXh)).trace()
        sigma_eps[i] += np.dot(np.dot(m_C[i, :].T, gXXg), m_C[i, :])
        sigma_eps[i] += (np.dot(Sigma_C[:, :, i], gXXg)).trace()
        sigma_eps[i] += np.dot(y_tilde[:, i].T, y_tilde[:, i])
        sigma_eps[i] -= 2 * np.dot(np.dot(m_C[i, :].T, gXXh), m_A[i, :])
        sigma_eps[i] -= 2 * np.dot(S.T, y_tilde[:, i])
        sigma_eps[i] /= N
    return sigma_eps


def maximization_sigma_noise_s(Y, X, m_A, Sigma_A, Ht, m_C, Sigma_C, Gt, W, \
                             M, N, J, y_tilde, Sigma_H, Sigma_G, sigma_eps):
    hXXh = np.zeros((M, M), dtype=float)
    gXXg = hXXh.copy()
    gXXh = hXXh.copy()
    for i in xrange(0, J):
        S = np.zeros((N), dtype=float)
        for m, k in enumerate(X):
            for m2, k2 in enumerate(X):
                hXXh[m, m2] = np.dot(np.dot(np.dot(Ht.T, X[k].T), X[k2]), Ht)
                hXXh[m, m2] += (np.dot(np.dot(Sigma_H, X[k].transpose()), X[k2])).trace()
                gXXg[m, m2] = np.dot(np.dot(np.dot(np.dot(np.dot(Gt.T, X[k].T),
                                                          W.T), W), X[k2]), Gt)
                gXXg[m, m2] += (np.dot(np.dot(np.dot(np.dot(Sigma_G, X[k].T), W.T),
                                                            W), X[k2])).trace()
                gXXh[m, m2] = np.dot(np.dot(np.dot(np.dot( \
                                               Gt.T, X[k].T), W.T), X[k2]), Ht)
            S += m_A[i, m] * np.dot(X[k], Ht) + \
                 m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        sigma_eps[i] = np.dot(np.dot(m_A[i, :].T, hXXh), m_A[i, :])
        sigma_eps[i] += (np.dot(Sigma_A[:, :, i], hXXh)).trace()
        sigma_eps[i] += np.dot(np.dot(m_C[i, :].T, gXXg), m_C[i, :])
        sigma_eps[i] += (np.dot(Sigma_C[:, :, i], gXXg)).trace()
        sigma_eps[i] += np.dot(y_tilde[:, i].T, y_tilde[:, i])
        sigma_eps[i] -= 2 * np.dot(np.dot(m_C[i, :].T, gXXh), m_A[i, :])
        sigma_eps[i] -= 2 * np.dot(S.T, y_tilde[:, i])
        sigma_eps[i] /= N
    return sigma_eps


def gradient0(q_Z, Z_tilde, J, m, K, graph, beta, gamma):
    Gr = gamma
    for i in xrange(0, J):
        Pmf_i = np.exp(beta * sum(Z_tilde[m, :, graph[i]], 1) \
                 - max(beta * sum(Z_tilde[m, :, graph[i]], 0))) / \
                sum(np.exp(beta * sum(Z_tilde[m, :, graph[i]], 0) \
                     - max(beta * sum(Z_tilde[m, :, graph[i]], 0))))
        print Pmf_i.shape
        print q_Z[m, :, i].shape
        print sum(Z_tilde[m, :, graph[i]])
        Gr += sum( sum(Z_tilde[m, :, graph[i]]) * (-q_Z[m, :, i] + Pmf_i), 0)
        print Gr
    return Gr


def gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma):
    Gr = gamma
    for i in xrange(0, J):
        tmp2 = beta * sum(Z_tilde[m, :, graph[i]], 0)
        Emax = max(tmp2)
        Sum = sum(np.exp(tmp2 - Emax))
        for k in xrange(0, K):
            tmp = sum(Z_tilde[m, k, graph[i]], 0)
            energy = beta * tmp
            Pzmi = np.exp(energy - Emax) / (Sum + eps)
            Gr += tmp * (-q_Z[m, k, i] + Pzmi)
    return Gr


def maximization_beta(beta, q_Z, Z_tilde, J, K, m, graph, gamma, neighbour,
                      maxNeighbours):
    Gr = 200
    step = 0.003
    ni = 1
    while ((abs(Gr) > 0.0001) and (ni < 200)):
        Gr = gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma)
        beta -= step * Gr
        ni += 1
    if beta < eps:
        beta = 0.01
    return beta


def maximization_mu(Omega, R, H, G, sigmaH, sigmaG, sigmaM):
    I = np.eye(R.shape[0])
    R_inv = np.linalg.inv(R)
    aux = np.linalg.inv(I / sigmaH + np.dot(Omega.T, Omega) / sigmaG \
                        + R_inv / sigmaM)
    Mu = np.dot(aux, H / sigmaH + np.dot(Omega.T, G) / sigmaG)
    return Mu


# Other functions
##############################################################

def computeFit(m_H, m_A, m_G, m_C, W, X, J, N):
    # print 'Computing Fit ...'
    stimIndSignal = np.zeros((N, J), dtype=np.float64)
    for i in xrange(0, J):
        m = 0
        for k in X:
            stimIndSignal[:, i] += m_A[i, m] * np.dot(X[k], m_H) \
                                 + m_C[i, m] * np.dot(np.dot(W, X[k]), m_G)
            m += 1
    return stimIndSignal





# Contrasts
##########################

def compute_contrasts(condition_names, contrasts, m_A, m_C):
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

        n += 1
        logger.info('Done contrasts computing.')
    return CONTRAST_A, CONTRASTVAR_A, CONTRAST_C, CONTRASTVAR_C


# Plots
####################################

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os.path as op
import os

def plot_response_functions_it(ni, NitMin, M, H, G):
    if not op.exists('./plots'):
        os.makedirs('./plots')
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NitMin + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    font = {'size': 15}
    matplotlib.rc('font', **font)
    if ni == 0:
        plt.close('all')
    colorVal = scalarMap.to_rgba(ni)
    plt.figure(M + 1)
    plt.plot(H, color=colorVal)
    plt.hold(True)
    plt.savefig('./plots/BRF_Iter_ASL.png')
    plt.figure(M + 2)
    plt.plot(G, color=colorVal)
    plt.hold(True)
    plt.savefig('./plots/PRF_Iter_ASL.png')
    return


def plot_convergence(ni, M, cA, cC, cH, cG, cAH, cCG,
                     SUM_q_Z, mua1, muc1, FE):
    SUM_p_Q_array = np.zeros((M, ni), dtype=np.float64)
    mua1_array = np.zeros((M, ni), dtype=np.float64)
    muc1_array = np.zeros((M, ni), dtype=np.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_p_Q_array[m, i] = SUM_q_Z[m][i]
            mua1_array[m, i] = mua1[m][i]
            muc1_array[m, i] = muc1[m][i]
    
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'size': 15}
    matplotlib.rc('font', **font)
    if not op.exists('./plots'):
        os.makedirs('./plots')
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
    plt.figure(M + 7)
    plt.plot(FE, label='Free energy')
    plt.legend()
    plt.savefig('./plots/free_energy.png')
    return

