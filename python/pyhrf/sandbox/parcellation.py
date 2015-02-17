# -*- coding: utf-8 -*-

# modified hierachical.py from scikit learn
"""Hierarchical Agglomerative Clustering

These routines perform some hierachical agglomerative clustering of some
input data. Currently, only Ward's algorithm is implemented.

Authors : Vincent Michel, Bertrand Thirion, Alexandre Gramfort,
          Gael Varoquaux
Modified: Aina Frau
License: BSD 3 clause
"""

import warnings
import logging

from heapq import heapify, heappop, heappush, heappushpop

import numpy as np

from scipy import sparse
from scipy.cluster import hierarchy
try:
    from scipy.maxentropy import logsumexp
except ImportError:
    from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse import cs_graph_components
from sklearn.externals.joblib import Memory
from sklearn.metrics import euclidean_distances
from sklearn.utils import array2d
from sklearn.cluster._feature_agglomeration import AgglomerationTransform

import pyhrf

from pyhrf.tools import add_prefix


logger = logging.getLogger(__name__)


# Functions for Ward algorithm
def loglikelihood_computation(fm, mu0, v0, mu1, v1, a):
    # log likelihood computation
    r2 = np.logaddexp(np.log(a / v1 ** (n / 2.)) - (fm - mu1) ** 2 / (2 * v1),
                      np.log((1 - a) / v0 ** (n / 2.)) - (fm - mu0) ** 2 / (2 * v0)).sum()
    return r2


def informedGMM(features, alphas):
    """
    Given a set of features, parameters (mu, v, lambda), and alphas:
    updates the parameters
    WARNING: only works for nb features = 1
    """
    if len(features.shape) > 1:
        n = features.shape[1]
        fm = features
    else:
        n = 1
        fm = features[:, np.newaxis]
    am = alphas
    eps = 1e-6
    # parameters estimates
    if am.sum(0) > 0:
        mu1 = (fm.T * am).T.sum(0) / am.sum(0)
        v1 = (((fm - mu1) ** 2).T * am).T.sum(0) / (n * am.sum(0))
        v1 = max(eps, v1)
    else:
        mu1 = v1 = eps
    if (1 - am).sum(0) > 0:
        mu0 = (fm.T * (1 - am)).T.sum(0) / (1 - am).sum(0)
        v0 = (((fm - mu0) ** 2).T * (1 - am)).T.sum(0) / (n * (1 - am).sum(0))
        v0 = max(eps, v0)
    else:
        mu0 = v0 = eps
    a = am.mean()
    if 0:
        print 'Updated parameters'
        print 'mu1:', mu1, 'mu0:', mu0
        print 'v1:', v1, 'v0:', v0
        print 'lambda:', 1 - a
        print ''

    return np.array([mu0, mu1]).squeeze(), np.array([v0, v1]).squeeze(), \
        np.array([1 - a, a]).squeeze()


def norm2_bc(a, b):
    """
    broadcast the computation of ||a-b||^2
    where size(a) = (m,n), size(b) = n
    """
    return (a ** 2).sum(1) + (b ** 2).sum() - 2 * (a * b).sum(1)


def informedGMM_MV(fm, am, cov_type='spherical'):
    """
    Given a set of multivariate features, parameters (mu, v, lambda), and alphas:
    fit a GMM where posterior weights are known (alphas)
    """
    n = fm.shape[1]
    eps = 1e-6
    #eps_ = np.zeros(n) + eps

    # parameters estimates
    if am.sum(0) > 0:
        mu1 = (fm.T * am).T.sum(0) / am.sum(0)
        #v1 = np.dot(np.dot((fm - mu1),(fm - mu1).T).T , am).T.sum(0) / (am.sum()*n)
        v1 = 0.0
        for ns in xrange(0, fm.shape[0]):
            v1 += (fm[ns, :] - mu1) * (fm[ns, :] - mu1).T * am[ns]
        v1 = v1 / (am.sum() * n)
        for nf in xrange(0, len(v1)):
            v1[nf] = max(eps, v1[nf])
        # old spherical covariance:
        #v1b = (norm2_bc(fm, mu1).T * am).T.sum(0) / (am.sum()*n)
    else:
        mu1 = fm.mean(0)
        v1m = fm.var()
        v1 = np.array([v1m, v1m])

    if (1 - am).sum(0) > 0:
        mu0 = (fm.T * (1 - am)).T.sum(0) / (1 - am).sum()
        #v0 = np.dot(np.dot((fm - mu0),(fm - mu0).T).T , am).T.sum(0) / (am.sum()*n)
        v0 = 0.0
        for ns in xrange(0, fm.shape[0]):
            v0 += (fm[ns, :] - mu1) * (fm[ns, :] - mu1).T * (1 - am)[ns]
        v0 = v0 / ((1 - am).sum() * n)
        for nf in xrange(0, len(v0)):
            v0[nf] = max(eps, v0[nf])
        # old spherical covariance:
        #v0b = (norm2_bc(fm, mu0).T * (1-am)).T.sum(0) / ((1-am).sum()*n)
    else:
        mu0 = fm.mean(0)
        v0m = fm.var()
        v0 = np.array([v0m, v0m])

    a = am.mean()

    if cov_type == 'spherical':
        v0m = np.mean(v0)
        v0 = np.array([v0m, v0m])
        v1m = np.mean(v1)
        v1 = np.array([v1m, v1m])

    if 0:
        print 'mu1:', mu1, 'mu0:', mu0
        print 'v1:', v1, 'v0:', v0
        print 'lambda:', a
        print ''

    # log likelihood computation
    aux1 = 0.0
    for ns in xrange(0, fm.shape[0]):
        # /n
        aux1 += np.dot(np.dot((fm[ns, :] - mu1),
                              np.linalg.inv(np.diag(v1))), (fm[ns, :] - mu1).T)
    aux0 = 0.0
    for ns in xrange(0, fm.shape[0]):
        # /n
        aux0 += np.dot(np.dot((fm[ns, :] - mu0),
                              np.linalg.inv(np.diag(v0))), (fm[ns, :] - mu0).T)
    loglh0 = np.logaddexp(np.log(a / (np.linalg.det(np.diag(v1)) * 2 * np.pi) ** (n / 2.)) - aux1 / 2,
                          np.log((1 - a) / (np.linalg.det(np.diag(v0)) * 2 * np.pi) ** (n / 2.)) - aux0 / 2)
    # old spherical log-likelihood
    # loglh0b = np.logaddexp(np.log(a/(v1[0]*2*np.pi)**(n/2.)) - norm2_bc(fm,mu1)/(2*v1[0]),
    # np.log((1-a)/(v0[0]*2*np.pi)**(n/2.)) - norm2_bc(fm,mu0)/(2*v0[0]))
    loglh = np.log(np.sum(np.exp(loglh0)))  # logsumexp(lpr)

    if np.isnan(loglh):
        print 'mu1:', mu1, 'mu0:', mu0
        print 'v1:', v1, 'v0:', v0
        print 'lambda:', a
        print ''
        raise Exception('log-likelihood is nan')

    return mu0, mu1, v0, v1, a, loglh

"""
def informedGMM_MV_old(fm, am, cov_type = 'spherical'):

    n = fm.shape[1]
    eps =  1e-6
    a = am.mean()
    #eps_ = np.zeros(n) + eps
    print 'n_features = ', n
    print 'n_samples = ', fm.shape[0]
    # parameters estimates
    if am.sum(0) > 0:
        mu1 = (fm.T * am).T.sum(0) / am.sum(0)
        #cov1 = np.dot(a.T, fm*fm)/am - 2*mu1*fm/am + mu1**2 # covar a*||fm-mu||^2/am
        #lpr = -0.5 * ( n * np.log(2*np.pi) + np.sum(np.log(cov1),1) + \
        #               np.sum((mu1**2)/cov1,1) - 2 * np.dot(fm, (mu1/cov1).T) + \
        #               np.dot(fm**2,(1.0/cov1).T))
        if cov_type == 'diag':
            v1 = np.dot(np.dot((fm - mu1),(fm - mu1).T).T , am).T.sum(0) / (am.sum()*n)
            if v1 == 0.0:
                print 'zero'
                v1 = np.ones(n)*eps
                print v1
                print v1.shape
            else:
                print 'max'
                for nf in xrange(0, len(v1)):
                    v1[nf] = max(eps,v1[nf])
        else:        # cov_type == 'spherical'
            v1b = (norm2_bc(fm, mu1).T * am).T.sum(0) / (am.sum()*n)
            v1b = max(eps, v1b)
            v1 = np.array([v1b, v1b])
    else:
        mu1 = fm.mean(0)
        v1b = fm.var()
        v1 = np.array([v1b, v1b])
    if (1-am).sum(0) > 0:
        mu0 = (fm.T * (1-am)).T.sum(0) / (1-am).sum()
        if cov_type == 'diag':
            v0 = np.dot(np.dot((fm - mu0),(fm - mu0).T).T , am).T.sum(0) / (am.sum()*n)
            for nf in xrange(0, len(v0)):
                v0[nf] = max(eps,v0[nf])
        else:       # cov_type == 'spherical'
            v0b = (norm2_bc(fm, mu0).T * (1-am)).T.sum(0) / ((1-am).sum()*n)
            v0b = max(eps, v0b)
            v0 = np.array([v0b, v0b])
    else:
        mu0 = fm.mean(0)
        v0b = fm.var()
        v0 = np.array([v0b, v0b])

    if 1:
        print 'mu1:', mu1, 'mu0:', mu0
        print 'v1:', v1, 'v0:', v0, ', ', v0b
        print 'lambda:', a
        print ''

    # log likelihood computation
    #loglh = np.logaddexp(np.log(a/(v1*2*np.pi)**(n/2.)) - norm2_bc(fm,mu1)/(2*v1),
    #                     np.log((1-a)/(v0*2*np.pi)**(n/2.)) - norm2_bc(fm,mu0)/(2*v0)).sum()
    if 1: #cov_type == 'diag':
        loglh0b = np.logaddexp(np.log(a/(v1[0]*2*np.pi)**(n/2.)) - norm2_bc(fm,mu1)/(2*v1[0]),
                              np.log((1-a)/(v0[0]*2*np.pi)**(n/2.)) - norm2_bc(fm,mu0)/(2*v0[0]))
    #loglh = np.log(np.sum(np.exp(loglh0)))
    loglhb = np.log(np.sum(np.exp(loglh0b))) #logsumexp(loglh0, axis = 1)

    if np.isnan(loglhb):
                print 'mu1:', mu1, 'mu0:', mu0
                print 'v1:', v1, 'v0:', v0
                print 'lambda:', a
                print ''
                raise Exception('log-likelihood is nan')

    return mu0, mu1, v0, v1, a, loglhb
"""


def compute_mixt_dist(features, alphas, coord_row, coord_col, cluster_masks,
                      moments, cov_type, res):
    """
    within one given territory: bi-Gaussian mixture model with known posterior
    weights:
        phi ~ \sum_i \lambda_i N(mu_i, v_i)
        p(q_j = i | phi_j) is an input (alphas)

    Estimation:
        \lambda_1 = 1 - \lambda_0 is the mean of posterior weights
        mu_i is estimated by weighted sample mean
        v_i is  estimated by weighted sample variance

    Args:
        - features (np.array((nsamples, nfeatures), float)):
            the feature to parcellate
        - alphas (np.array(nsamples, float)):
            confidence levels on the features -> identified to posterior
            weights of class activating in the GMM fit
        - coord_row  (list of int): row candidates for merging
        - coord_col (list of int): col candidates for merging
        - cluster_masks ():
        - moments: !!!
        - res:

    Return:


    """
    size_max = coord_row.shape[0]

    def mixt_dist(mom1, mom2, merge_mask, cov_type):
        def log_phy(c, cov_type):
            """
            c : cluster mask
            """

            # print c
            m = np.where(c)
            # print m
            fm = features[m[0], :]
            am = alphas[m]
            #mu0, mu1, v0, v1, a, loglh = informedGMM_MV(fm, am)
            _, _, _, _, _, loglh = informedGMM_MV(fm, am, cov_type)

            return loglh

        #merge_mask = np.bitwise_or(c0, c1)
        # + merge_mask.sum() is used to penalize too big clusters
        # return log_phy(c0) + log_phy(c1) - log_phy(merge_mask) #+ .7 *
        # merge_mask.sum()
        return (mom1 + mom2 - log_phy(merge_mask, cov_type)) + merge_mask.sum()

    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        #res[i] = mixt_dist(cluster_masks[row], cluster_masks[col])
        cmr, cmc = cluster_masks[row], cluster_masks[col]
        merge_mask = np.bitwise_or(cmr, cmc)
        if merge_mask.sum() == 2:  # weighted euclidian distance
            res[i] = (
                (features[cmr[0], :] - features[cmc[0], :]) ** 2).sum() ** .5
        else:
            res[i] = mixt_dist(
                moments[row], moments[col], merge_mask, cov_type)
        logger.info('dist %d <-> %d = %f', row, col, res[i])
    return res


from sklearn.mixture import GMM


def compute_mixt_dist_skgmm(features, alphas, coord_row, coord_col,
                            cluster_masks, moments, cov_type, res):
    size_max = coord_row.shape[0]

    def mixt_dist(mom1, mom2, merge_mask, cov_type):
        def log_phy(c, cov_type):
            """
            c : cluster mask
            """
            m = np.where(c)
            if len(m[0]) == 1:  # singleton
                return 0
            else:
                fm = features[m[0], :]
                #n = features.shape[1]
                am = alphas[m]
                #g = GMM(n_states=2, cvtype='diag')
                g = GMM(n_components=2, covariance_type=cov_type)
                return g.fit(fm).score(fm).sum()

        if merge_mask.sum() == 2:  # euclidian distance
            return ((mom1 - mom2) ** 2).sum() ** .5
        else:
            return mom1 + mom2 - log_phy(merge_mask, cov_type) + merge_mask.sum()

    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        merge_mask = np.bitwise_or(cluster_masks[row], cluster_masks[col])
        res[i] = mixt_dist(moments[row], moments[col], merge_mask, cov_type)
        # print 'row %d, col %d -> %f' %(row, col, res[i])
    return res


# def compute_ward_dist_sk(np.ndarray[DOUBLE, ndim=1, mode='c'] m_1,
#                       np.ndarray[DOUBLE, ndim=2, mode='c'] m_2,
#                       np.ndarray[INTP, ndim=1, mode='c'] coord_row,
#                       np.ndarray[INTP, ndim=1, mode='c'] coord_col,
#                       np.ndarray[DOUBLE, ndim=1, mode='c'] res):
#     cdef INTP size_max = coord_row.shape[0]
#     cdef INTP n_features = m_2.shape[1]
#     cdef INTP i, j, row, col
#     cdef DOUBLE pa, n

#     for i in range(size_max):
#         row = coord_row[i]
#         col = coord_col[i]
#         n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col])
#         pa = 0.
#         for j in range(n_features):
#             pa += (m_2[row, j] / m_1[row] - m_2[col, j] / m_1[col]) ** 2
#         res[i] = pa * n
#     return res


def compute_uward_dist(m_1, m_2, coord_row, coord_col, variance, actlev, res):
    """ Function computing Ward distance:
    inertia = !!!!0

    Parameters
    ----------
    m_1,m_2,coord_row,coord_col: clusters' parameters
    variance: uncertainty
    actlev: activation level

    Returns
    -------
    res: Ward distance

    Modified: Aina Frau
    """
    size_max = coord_row.shape[0]
    n_features = m_2.shape[1]
    logger.info(
        'ward dist with uncertainty -> %d computations to do', size_max)

    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col])  # size cluster
        pa = 0.
        for j in range(n_features):
            pa += (m_2[row, j] - m_2[col, j]) ** 2
        logger.info('pa  %d <-> %d = %f', row, col, pa)
        aux = actlev[row] * actlev[col]
        logger.info('aux %d <-> %d = %f', row, col, aux)
        res[i] = pa * aux  # * n
        logger.info('dist %d <-> %d = %f', row, col, res[i])
        if np.isnan(res[i]):
            raise Exception('inertia is nan')

    return res


def compute_uward_dist2(m_1, features, alphas, coord_row, coord_col, cluster_masks, res):
    """ Function computing Ward distance:
    In this case we are using the model-based definition to compute the inertia

    Parameters
    ----------
    m_1,m_2,coord_row,coord_col: clusters' parameters
    variance: uncertainty
    actlev: activation level

    Returns
    -------
    res: Ward distance

    Modified: Aina Frau
    """
    size_max = coord_row.shape[0]
    n_features = features.shape[1]
    logger.info(
        'ward dist with uncertainty -> %d computations to do', size_max)

    def ward_dist(m0, m1, merge_mask):
        def ess(c):
            """
            c : cluster mask
            """
            m = np.where(merge_mask)
            if len(m[0]) == 1:  # singleton
                return 0
            else:
                fm = features[m[0], :]
                n = features.shape[1]
                am = alphas[m]
                # parameters estimates
                mu0 = fm.mean()
                inertia = (am * ((fm - mu0) ** 2).T).T.sum()
            return inertia

        if merge_mask.sum() == 2:  # euclidian distance
            return ((m0 - m1) ** 2).sum() ** .5
        else:
            return ess(merge_mask) - m0 - m1  # + .7 * merge_mask.sum()

    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        merge_mask = np.bitwise_or(cluster_masks[row], cluster_masks[col])
        res[i] = ward_dist(m_1[row], m_1[col], merge_mask)
        logger.info('dist %d <-> %d = %f', row, col, res[i])

    return res


def _hc_get_descendent(node, children, n_leaves):
    """
    Function returning all the descendent leaves of a set of nodes in the tree.

    Parameters
    ----------
    node : int
    The node for which we want the descendents.

    children : list of pairs. Length of n_nodes
    The children of each non-leaf node. Values less than `n_samples` refer
    to leaves of the tree. A greater value `i` indicates a node with
    children `children[i - n_samples]`.

    n_leaves : int
    Number of leaves.

    Returns
    -------
    descendent : list of int
    """
    ind = [node]
    if node < n_leaves:
        return ind
    descendent = []
    # It is actually faster to do the accounting of the number of
    # elements is the list ourselves: len is a lengthy operation on a
    # chained list
    n_indices = 1
    while n_indices:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
            n_indices -= 1
        else:
            ind.extend(children[i - n_leaves])
            n_indices += 1
    return descendent


def hc_get_heads(parents, copy=True):
    """ Return the heads of the forest, as defined by parents
    Parameters
    ===========
    parents: array of integers
    The parent structure defining the forest (ensemble of trees)
    copy: boolean
    If copy is False, the input 'parents' array is modified inplace

    Returns
    =======
    heads: array of integers of same shape as parents
    The indices in the 'parents' of the tree heads
    """
    if copy:
        parents = np.copy(parents)
    size = parents.size
    for node0 in range(size):
        # Start from the top of the tree and go down
        node0 = size - node0 - 1
        node = node0
        parent = parents[node]
        while parent != node:
            parents[node0] = parent
            node = parent
            parent = parents[node]
    return parents


def _get_parents(nodes, heads, parents, not_visited):
    """ Return the heads of the given nodes, as defined by parents
    Modifies in-place 'heads' and 'not_visited'

    Parameters
    ===========
    nodes: list of integers
    The nodes to start from
    heads: list of integers
    A list to hold the results (modified inplace)
    parents: array of integers
    The parent structure defining the tree
    not_visited:
    The tree nodes to consider (modified inplace)
    """
    for node in nodes:
        parent = parents[node]
        while parent != node:
            node = parent
            parent = parents[node]
        if not_visited[node]:
            not_visited[node] = 0
            heads.append(node)
    return heads


###############################################################################
# Ward's algorithm

def ward_tree(X, connectivity=None, n_components=None, copy=True, n_clusters=None,
              var=None, act=None, var_ini=None, act_ini=None,
              dist_type='uward', cov_type='spherical', save_history=False):
    """Ward clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account a some topological
    structure between samples.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered

    connectivity : sparse matrix.
        connectivity matrix. Defines for each sample the neigbhoring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.

    copy : bool (optional)
        Make a copy of connectivity or work inplace. If connectivity
        is not of LIL type there will be a copy in any case.

    n_clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    dist_type : str -> uward | mixt

    Returns
    -------
    children : 2D array, shape (n_nodes, 2)
        list of the children of each nodes.
        Leaves of the tree have empty list of children.

    n_components : sparse matrix.
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree

    parents : 1D array, shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    Modified: Aina Frau
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        if n_clusters is not None:
            warnings.warn('Early stopping is implemented only for '
                          'structured Ward clustering (i.e. with '
                          'explicit connectivity.', stacklevel=2)
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.int)
        return children_, 1, n_samples, None, None, None

    # Compute the number of nodes
    if n_components is None:
        n_components, labels = cs_graph_components(connectivity)

    # Convert connectivity matrix to LIL with a copy if needed
    if sparse.isspmatrix_lil(connectivity) and copy:
        connectivity = connectivity.copy()
    elif not sparse.isspmatrix(connectivity):
        connectivity = sparse.lil_matrix(connectivity)
    else:
        connectivity = connectivity.tolil()

    if n_components > 1:
        warnings.warn("the number of connected components of the "
                      "connectivity matrix is %d > 1. Completing it to avoid "
                      "stopping the tree early." % n_components)
        connectivity = _fix_connectivity(X, connectivity, n_components, labels)
        n_components = 1

    if n_clusters is None:
        n_nodes = 2 * n_samples - n_components
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters

    if (connectivity.shape[0] != n_samples
            or connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s '
                         'when X is %s' % (connectivity.shape, X.shape))

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.int)
    coord_col = np.array(coord_col, dtype=np.int)

    # compute initial cluster masks
    dummy_idx = np.arange(n_samples)
    cluster_masks = [dummy_idx == i for i in range(n_nodes)]

    inertia = np.empty(len(coord_row), dtype=np.float)
    variance = np.zeros((n_nodes, n_features))
    if var is None:
        variance[:n_samples] = 1
    else:
        variance[:n_samples] = var

    act_level = np.zeros(n_nodes)
    if act is None:
        act_level[:n_samples] = 1
    else:
        act_level[:n_samples] = act

    if 0:  # normalize act levels
        min_a, max_a = min(act), max(act)
        eps = 1e-4
        if max_a != min_a:
            act_level[:n_samples] = (
                act - max_a) / (max_a - min_a) * (1 - eps) + 1 - eps
        else:
            act_level[:n_samples] = 1 - eps

    if (dist_type == 'mixt') or (dist_type == 'mixt_skgmm'):
        act_level = act_level[:n_samples]

    # build moments
    moments_1 = np.zeros(n_nodes)
    if (dist_type == 'uward') or (dist_type == 'uward2'):
        # moments_1: cluster size
        moments_1[:n_samples] = 1
        # moments_2: cluster inertia
        moments_2 = np.zeros((n_nodes, n_features))
        moments_2[:n_samples] = X
    else:  # dist_type == 'mixt'
        # moments_1: label likelihood
        moments_1[:n_samples] = 0
        # moments_2: mixture parameters for each cluster
        moments_2 = np.zeros((n_nodes, n_features * 3 + 3))
        moments_2[:n_samples] = np.hstack((X, X, np.zeros((X.shape[0], n_features)),
                                           np.zeros((X.shape[0], n_features)),
                                           act_level[:, np.newaxis]))

    variance_ini = np.zeros((n_nodes, n_features))
    if var_ini is None:
        variance_ini[:n_samples] = 1
    else:
        variance_ini[:n_samples] = var_ini
    activation_ini = np.zeros(n_nodes)
    if act_ini is None:
        activation_ini[:n_samples] = 1
    else:
        activation_ini[:n_samples] = act_ini

    logger.info('Initialization')
    logger.info('A:')
    logger.info(A)
    logger.info('moments_1: %s', moments_1)
    logger.info('moments_2: %s', moments_2)
    logger.info('coord_row: %s', coord_row)
    logger.info('coord_col: %s', coord_col)
    if (dist_type == 'mixt') or (dist_type == 'mixt_skgmm'):
        logger.info('cluster_masks:')
        logger.info(cluster_masks)

    # Compute initial distances:
    if dist_type == 'uward':
        compute_uward_dist(moments_1, moments_2, coord_row, coord_col,
                           variance_ini, activation_ini, inertia)
    elif dist_type == 'uward2':
        compute_uward_dist2(moments_1, X, act_level,
                            coord_row, coord_col,
                            cluster_masks, inertia)
    elif dist_type == 'mixt':
        compute_mixt_dist(X, act_level, coord_row, coord_col, cluster_masks,
                          moments_1, cov_type, inertia)
    else:  # dist_type == 'mixt_skgmm'
        compute_mixt_dist_skgmm(X, act_level, coord_row, coord_col,
                                cluster_masks, moments_1, cov_type, inertia)

    logger.info('initial inertia:')
    logger.info(inertia)

    inertia = zip(inertia, coord_row, coord_col)
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.int)
    heights = np.zeros(n_nodes)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []
    not_visited = np.empty(n_nodes, dtype=np.int8)

    n_its = n_nodes - n_samples
    history = np.zeros((n_its, n_samples), dtype=np.uint16)
    if save_history:
        history_choices = []
        history_choices_inertia = []
    cur_parc = np.arange(n_samples, dtype=int)

    # recursive merge loop
    for it, k in enumerate(xrange(n_samples, n_nodes)):
        logger.info('iteration: %d', it)
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break

        logger.info('group (%d, %d) has min inertia (%f) -> group %d', i, j,
                    inert, k)

        inertia.sort()
        logger.info('remainging inertia heap:')
        logger.info(inertia)

        if save_history:
            choices = []
            choices_inertia = []
            # first choice is kept one:
            choices.append(cluster_masks[i] + cluster_masks[j] * 2)
            choices_inertia.append(inert)
            # other choices:
            for iner, u, v in inertia:
                choices.append(cluster_masks[u] + cluster_masks[v] * 2)
                choices_inertia.append(iner)
            history_choices.append(np.array(choices, dtype=np.int16))
            history_choices_inertia.append(np.array(choices_inertia))

        # print 'i='+str(i)+',j='+str(j)+',k='+str(k)
        parent[i], parent[j], heights[k] = k, k, inert
        children.append([i, j])
        used_node[i] = used_node[j] = False
        new_cluster_mask = np.bitwise_or(cluster_masks[i], cluster_masks[j])
        cluster_masks[k] = new_cluster_mask
        # print 'new_cluster_mask:', new_cluster_mask.astype(int)

        # save history:
        cur_parc[np.where(new_cluster_mask)] = k
        # if save_history:
        history[it] = cur_parc[:]

        # update the moments
        if (dist_type == 'uward'):
            sact = act_level[i] + act_level[j]
            moments_1[k] = moments_1[i] + moments_1[j]
            if sact > 0:
                moments_2[k] = (moments_2[i] * act_level[i] +
                                moments_2[j] * act_level[j]) / sact
                # variance[k] = (variance[i] * act_level[i] + \
                #                variance[j] * act_level[j]) / sact
            else:
                moments_2[k] = moments_2[i] + moments_2[j]
                #variance[k] = variance[i] + variance[j]

            act_level[k] = (act_level[i] + act_level[j])  # /2.
            # act_level[k] = max(act_level[i], act_level[j]) # OK for this
            # shouldnt we update in the same way as moments??
            variance[k] = variance[i] + variance[j]
        elif dist_type == 'uward2':
            m = np.where(new_cluster_mask)
            fm = X[m[0], :]
            am = act_level[m]
            mu0 = fm.mean()
            moments_1[k] = (am * ((fm - mu0) ** 2).T).T.sum()
        elif dist_type == 'mixt':
            m = np.where(new_cluster_mask)
            fm = X[m[0], :]
            am = act_level[m]

            mu0, mu1, v0, v1, l, loglh = informedGMM_MV(fm, am, cov_type)

            moments_1[k] = loglh
            moments_2[k] = np.hstack([mu0, mu1, v0, v1, l])

        elif dist_type == 'mixt_skgmm':
            m = np.where(new_cluster_mask)
            fm = X[m[0], :]
            am = act_level[m]

            #g0 = GMM(n_components=2, covariance_type='spherical')
            # g0.fit(fm)

            g = GMM(n_components=2, covariance_type=cov_type)
            g.fit(fm)

            if cov_type == 'spherical' and not (g.covars_[0, 0] == g.covars_[0, 1]):
                raise Exception('Variances different!!')

            moments_1[k] = g.score(fm).sum()
            moments_2[k] = np.hstack([g.means_[0], g.means_[1],
                                      g.covars_[0, :], g.covars_[1, :],
                                      g.weights_[0]])
        else:
            raise Exception('error unknown dist type: %s' % dist_type)

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _get_parents(A[i], coord_col, parent, not_visited)
        _get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[l].append(k) for l in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.int)
        coord_row = np.empty_like(coord_col)
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float)

        logger.info('moments_2 of new cluster: %s', moments_2[k])
        logger.info('coord_row: %s', coord_row)
        logger.info('coord_col: %s', coord_col)

        if dist_type == 'uward':
            compute_uward_dist(moments_1, moments_2, coord_row, coord_col,
                               variance, act_level, ini)
        elif dist_type == 'uward2':
            compute_uward_dist2(moments_1, X, act_level,
                                coord_row, coord_col,
                                cluster_masks, ini)
        elif dist_type == 'mixt':
            compute_mixt_dist(X, act_level, coord_row, coord_col,
                              cluster_masks, moments_1, cov_type, ini)

        else:  # dist_type == 'mixt_skgmm'
            compute_mixt_dist_skgmm(X, act_level, coord_row, coord_col,
                                    cluster_masks, moments_1, cov_type, ini)

        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx]))
            for idx in xrange(n_additions)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    children = np.array(children)  # return numpy array for efficient caching

    if save_history:
        # pack history of choices into numpy array
        nmax_choices = max([len(choices) for choices in history_choices])
        print 'size of history_choices:', str((n_its, nmax_choices, n_samples))
        history_choices_a = np.zeros((n_its, nmax_choices, n_samples),
                                     dtype=np.int16) - 1
        #history_choices_a = np.zeros((n_its, 1, n_samples), dtype=np.uint16) - 1
        history_choices_i = np.zeros((n_its, nmax_choices)) - 1
        for it in xrange(n_its):
            nchoices = len(history_choices[it])
            history_choices_a[it, :nchoices, :] = history_choices[it]
            history_choices_i[it, :nchoices] = history_choices_inertia[it]
    else:
        history_choices_a, history_choices_i = None, None

    return children, n_components, n_leaves, parent, heights, moments_2, \
        history, history_choices_a, history_choices_i


###############################################################################
# For non fully-connected graphs

def _fix_connectivity(X, connectivity, n_components, labels):
    """
    Warning: modifies connectivity in place
    """
    for i in range(n_components):
        idx_i = np.where(labels == i)[0]
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.where(labels == j)[0]
            Xj = X[idx_j]
            D = euclidean_distances(Xi, Xj)
            ii, jj = np.where(D == np.min(D))
            ii = ii[0]
            jj = jj[0]
            connectivity[idx_i[ii], idx_j[jj]] = True
            connectivity[idx_j[jj], idx_i[ii]] = True
    return connectivity

###############################################################################
# Functions for cutting  hierarchical clustering tree


def _hc_cut(n_clusters, children, n_leaves):
    """Function cutting the ward tree for a given number of clusters.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.

    children : list of pairs. Length of n_nodes
        List of the children of each nodes.
        Leaves have empty list of children and are not stored.

    n_leaves : int
        Number of leaves of the tree.

    Returns
    -------
    labels : array [n_samples]
        cluster labels for each point

    """
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree
    nodes = [-(max(children[-1]) + 1)]
    for i in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.int)
    for i, node in enumerate(nodes):
        label[_hc_get_descendent(-node, children, n_leaves)] = i
    return label


###############################################################################
# Class for Ward hierarchical clustering

class Ward(BaseEstimator, ClusterMixin):
    """Ward hierarchical clustering: constructs a tree and cuts it.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to find.

    connectivity : sparse matrix.
        Connectivity matrix. Defines for each sample the neigbhoring
        samples following a given structure of the data.
        Default is None, i.e, the hiearchical clustering algorithm is
        unstructured.

    memory : Instance of joblib.Memory or string
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    copy : bool
        Copy the connectivity matrix or work inplace.

    n_components : int (optional)
        The number of connected components in the graph defined by the \
        connectivity matrix. If not set, it is estimated.

    compute_full_tree: bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of cluster and using caching, it may
        be advantageous to compute the full tree.


    Attributes
    ----------
    `children_` : array-like, shape = [n_nodes, 2]
        List of the children of each nodes.  Leaves of the tree do not appear.

    `labels_` : array [n_samples]
        cluster labels for each point

    `n_leaves_` : int
        Number of leaves in the hiearchical tree.

    `n_components_` : sparse matrix.
        The estimated number of connected components in the graph.

    """

    def __init__(self, n_clusters=2, memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, copy=True, n_components=None,
                 compute_full_tree='auto', dist_type='uward', cov_type='spherical',
                 save_history=False):
        self.n_clusters = n_clusters
        self.memory = memory
        self.copy = copy
        self.n_components = n_components
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.dist_type = dist_type
        self.cov_type = cov_type
        self.save_history = save_history

    def fit(self, X, var=None, act=None, var_ini=None, act_ini=None):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.

        Returns
        -------
        self
        """
        memory = self.memory
        X = array2d(X)
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory, verbose=0)

        if not self.connectivity is None:
            if not sparse.issparse(self.connectivity):
                raise TypeError("`connectivity` should be a sparse matrix or "
                                "None, got: %r" % type(self.connectivity))

            if (self.connectivity.shape[0] != X.shape[0] or
                    self.connectivity.shape[1] != X.shape[0]):
                raise ValueError("`connectivity` does not have shape "
                                 "(n_samples, n_samples)")

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            # Early stopping is likely to give a speed up only for
            # a large number of clusters. The actual threshold
            # implemented here is heuristic
            compute_full_tree = self.n_clusters > max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        self.children_, self.n_components_, self.n_leaves_, parents, \
            self.heights, self.moments, self.history, self.history_choices, \
            self.history_choices_inertia = \
            memory.cache(ward_tree)(X, self.connectivity,
                                    n_components=self.n_components,
                                    copy=self.copy, n_clusters=n_clusters,
                                    var=var, act=act,
                                    var_ini=var_ini, act_ini=act_ini,
                                    dist_type=self.dist_type,
                                    cov_type=self.cov_type,
                                    save_history=self.save_history)

        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters, self.children_,
                                   self.n_leaves_)
        else:
            labels = hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reasign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self


###############################################################################
# Ward-based feature agglomeration

class WardAgglomeration(AgglomerationTransform, Ward):
    """Feature agglomeration based on Ward hierarchical clustering

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters.

    connectivity : sparse matrix
        connectivity matrix. Defines for each feature the neigbhoring
        features following a given structure of the data.
        Default is None, i.e, the hiearchical agglomeration algorithm is
        unstructured.

    memory : Instance of joblib.Memory or string
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    copy : bool
        Copy the connectivity matrix or work inplace.

    n_components : int (optional)
        The number of connected components in the graph defined by the
        connectivity matrix. If not set, it is estimated.

    compute_full_tree: bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of cluster and using caching, it may
        be advantageous to compute the full tree.

    variance: array with variances of all samples (default: None)
        Injected in the calculation of the inertia.

    activation: level of activation detected (default: None)
        Used to weight voxels depending on level of activation in inertia computation.
        A non-activ voxel will not estimate the HRF correctly,
        so features will not be correct either.

    Attributes
    ----------
    `children_` : array-like, shape = [n_nodes, 2]
        List of the children of each nodes.
        Leaves of the tree do not appear.

    `labels_` : array [n_samples]
        cluster labels for each point

    `n_leaves_` : int
        Number of leaves in the hiearchical tree.

    """

    def fit(self, X, y=None, **params):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The data

        Returns
        -------
        self
        """
        return Ward.fit(self, X.T, **params)

# END OF modified hierachical.py from scikit learn

# Feature extraction

import scipy as sp


def calculate_uncertainty(dm, g):
    # beta values. dm: design matrix, g: my glm
    #beta_vars = dict.fromkeys(dm.names)
    uncertainty = np.zeros((g.nvbeta.shape[0], g.s2.shape[0]))
    for ib, bname in enumerate(dm.names):
        # sp.diag(g.nvbeta)[ib]      #variance: diag of cov matrix
        # sig2 = g.s2                #ResMS
        # beta_vars[bname] = sp.diag(g.nvbeta)[ib]*g.s2   #variance for all
        # voxels, condition ib
        uncertainty[ib, :] = sp.diag(g.nvbeta)[ib] * g.s2
    return uncertainty


def hrf_canonical_derivatives(tr, oversampling=2., time_length=25.):
    # Canonical HRF and derivatives
    # build dummy design matrix with a single stim event (1 condition) to get
    # exactly the same hrf, dhrf and d2hrf as in the nipy GLM
    try:
        from collections import OrderedDict
    except ImportError:
        from pyhrf.tools.backports import OrderedDict
    from pyhrf.paradigm import Paradigm
    n_scans = time_length * oversampling
    frametimes = np.linspace(0, (n_scans - 1) * tr / oversampling, n_scans)
    ons = dict([('Pulse', np.array([1]))])
    cnames = ['Pulse']
    onsets = OrderedDict(zip(cnames, [[ons[c]] for c in cnames]))
    paradigm0 = Paradigm(onsets)
    paradigm1 = paradigm0.to_nipy_paradigm()
    from nipy.modalities.fmri import design_matrix as dm
    design_matrix = dm.make_dmtx(frametimes, paradigm1,
                                 # 'spm_time_dispersion',
                                 hrf_model='spm_time_dispersion',
                                 drift_model='Cosine', hfcut=128,
                                 fir_delays=[0])
    return design_matrix.matrix


def compute_hrf(method, my_glm, can, ndelays, i):
    can0 = can[:, 0]
    dcan0 = can[:, 1]
    d2can0 = can[:, 2]
    if method == 'glm':
        hrf = can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i, :])
    if method == 'glm_deriv':
        hrf = can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 3 + 0, :]) + \
            dcan0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 3 + 1, :]) + \
            d2can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 3 + 2, :])
    if method == 'glm_deriv5':
        d3can0 = can[:, 3]
        d4can0 = can[:, 4]
        hrf = can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 5 + 0, :]) + \
            dcan0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 5 + 1, :]) + \
            d2can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 5 + 2, :]) + \
            d3can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 5 + 3, :]) + \
            d4can0[:, np.newaxis] * (my_glm.beta[np.newaxis, i * 5 + 4, :])
    if method == 'fir':
        hrf = my_glm.beta[i * ndelays:(i + 1) * ndelays, :]
    return hrf


def squared_error(n, m):  # it needs to enter stim_induced!!!
    error = np.zeros((n.shape[1], 1))
    for i, x in enumerate(error):
        error[i] = np.mean(np.square(m[:, i] - n[:, i]))
    return error


def FWHM(Y):
    import math
    half_max = max(Y) / 2.
    max_ind = np.argmax(Y)
    if max(Y) > 0 and (abs(max(Y)) - abs(min(Y))) > 0:
        max_sign = math.copysign(1., half_max)
        d1 = max_sign * (half_max - np.array(Y[0:max_ind]))
        sd1 = np.array(sorted(d1))
        sd1_idx = np.argsort(d1)
        lval = sd1[sd1 > 0][0]
        li = sd1_idx[sd1 > 0][0]
        rval = sd1[sd1 < 0][-1]
        ri = sd1_idx[sd1 < 0][-1]
        if abs(lval) > abs(rval):
            left_val = rval
            left_idx = ri
        else:
            left_val = lval
            left_idx = li
        d2 = -max_sign * (half_max - np.array(Y[max_ind:]))
        sd2 = np.array(sorted(d2))
        sd2_idx = np.argsort(d2)
        lval = sd2[sd2 > 0][0]
        li = sd2_idx[sd2 > 0][0]
        rval = sd2[sd2 < 0][-1]
        ri = sd2_idx[sd2 < 0][-1]
        if abs(lval) > abs(rval):
            right_val = rval
            right_idx = ri + d1.shape[0]
        else:
            right_val = lval
            right_idx = li + d1.shape[0]
        fwhm = abs(right_idx - left_idx)
    else:
        fwhm = 0.
    return fwhm


def compute_fwhm(F, dt, a=0):
    # full width-at-half-maximum (FWHM):
    F_max = F.max(a)
    fwhm = np.zeros_like(F_max)
    if a == 2:
        for vx in range(F_max.shape[0]):
            for vy in range(F_max.shape[1]):
                fwhm[vx, vy] = FWHM(F[vx, vy, :]) * dt
    else:
        for v, iv in enumerate(F[0, :]):
            fwhm[v] = FWHM(F[:, v]) * dt
    fwhm[np.where((abs(F.max(a)) - abs(F.min(a))) < 0.)] = 0.
    return fwhm.flatten()


from pyhrf.glm import glm_nipy


def GLM_method(name, data0, ncond, dt=.5, time_length=25., ndelays=0):
    if name == 'glm':
        hrf_model = 'Canonical'
        nfeat = 1
    elif name == 'glm_deriv5':
        hrf_model = 'canonical with 5 derivatives'
        nfeat = 5
    elif name == 'fir':
        hrf_model = 'fir'
        nfeat = ndelays
    elif name == 'glm_deriv':
        hrf_model = 'spm_time_dispersion'
        nfeat = 3
    else:
        raise AssertionError("Method not implemented")
    # GLM, features: my_glm.beta
    oversampling = 1. / dt
    fdelays = range(ndelays)  # FIR duration = nrange*TR
    can = hrf_canonical_derivatives(data0.tr, oversampling, time_length)
    [my_glm, design_matrix, c] = glm_nipy(
        data0, hrf_model=hrf_model, fir_delays=fdelays, contrasts={"activation_contrast": "audio"})
    if 0:
        import matplotlib.pyplot as plt
        import pyhrf.plot as plt2
        design_matrix.show()
        plt.savefig('design_matrix.png')
        plt2.autocrop('design_matrix.png')
        plt.close()
    ca = c['activation_contrast']
    Xb = np.dot(design_matrix.matrix, my_glm.beta)   # fit = X*beta
    # output all betas (see pyhrf.glm.glm_nipy_from_files)
    beta_vars = calculate_uncertainty(design_matrix, my_glm)
    hrf = np.zeros((oversampling * time_length, my_glm.beta.shape[1], ncond))
    ttp = np.zeros((my_glm.beta.shape[1], ncond))
    fwhm = np.zeros((my_glm.beta.shape[1], ncond))
    for i in range(ncond):
        hrf[:, :, i] = compute_hrf(name, my_glm, can, ndelays, i)     # HRFs
        ttp[:, i] = my_glm.beta.argmax(0) * dt
        fwhm[:, i] = compute_fwhm(hrf[:, :, i], dt)
    #mse = squared_error(Xb,stim_induced[0:-1:2,:])
    # , mse
    return Xb, my_glm.beta, hrf, design_matrix, beta_vars, ca.pvalue(), ttp, nfeat, fwhm


def generate_features(parcellation, act_labels, feat_levels, noise_var=0.):
    """
    Generate noisy features with different levels across positions
    depending on parcellation and activation clusters.

    Args:
        - parcellation (flat np.ndarray of integers in [1, nb_parcels]):
            the input parcellation
        - act_labels (flat binary np.ndarray):
            define which positions are active (1) and non-active (0)
        - feat_levels (dict of (int[1,N] : (array((n_features,), float),
                                            array((n_features)), float))):
            -> (dict of (parcel_idx : (feat_lvl_inact, feat_lvl_act))
            map a parcel labels to feature levels in non-active and active-pos
            eg: {1: ([1., .5], [10., 15])} indicates that features in parcel
                1 have values [1., .5] in non-active positions (2 features per
                position) and value 10. in active-positions
        - noise_var (float>0):
            variance of additive Gaussian noise

    Return:
        np.array((n_positions, n_features), float)
        The simulated the features.

    """

    n_features = len(feat_levels[feat_levels.keys()[0]][0])
    n_positions = parcellation.size

    def is_binary(a):
        labels = np.unique(a)
        return (len(labels) == 1 and (labels[0] == 0 or labels[0] == 1) or
                (labels == [0, 1]).all())

    assert is_binary(act_labels)

    features = np.zeros((n_positions, n_features))
    for p in np.unique(parcellation):
        for l in np.unique(act_labels):
            features[np.bitwise_and(parcellation == p, act_labels == l), :] = \
                feat_levels[p][l]

    return features + np.random.randn(n_positions, n_features) * noise_var ** .5


def represent_features(features, labels, ampl, territories, t, fn):
    """
    Generate chart with features represented.

    Args:
        - features: features to be represented
        - labels: territories
        - ampl: amplitude of the positions

    Return:
        features represented in 2D: the size of the spots depends on ampl,
                                    and the color on labels

    """
    import matplotlib.pyplot as plt
    import pyhrf.plot as plt2
    act = (ampl - min(ampl)) / (max(ampl) - min(ampl))  # activation labels
    plt.figure()
    ecolors = ['none', 'black']
    acolors = ['blue', 'red']
    colors = ['blue', 'cyan', 'yellow', 'red',
              'green', 'magenta', 'black', 'white']
    for i in range(len(labels)):
        if t == 1:
            e = ecolors[labels[i]]
            c = colors[territories[i]]
        else:
            e = acolors[labels[i]]
            c = 'none'
        s0 = act[i] * 100
        plt.scatter(features[i, 0], features[i, 1], color=c, s=s0, edgecolor=e)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig(fn)
    plt.close()


def feature_extraction(fmri_data, method, dt=.5, time_length=25., ncond=1):
    """
    fmri_data (pyhrf.core.FmriData): single ROI fMRI data
    """
    import matplotlib.pyplot as plt
    import pyhrf.plot as plt2
    Xb, f, hrf, dm, beta_vars, pval, ttp, nfeat, fwhm = GLM_method(method, fmri_data, ncond,
                                                                   dt, time_length)
    amplitude0 = f[0, :]  # / beta_vars[0,:]
    a, A = amplitude0.min(), amplitude0.max()
    ampl_dynrange = (amplitude0 - a) / (A - a)
    # For more than 1 cond we sum the beta 0 of the conditions:
    # amplitude = np.zeros((f.shape[1],1))
    # for i in range(ncond):
    #     amplitude += f[i*nfeat,:]
    features = np.zeros((f.shape[1], nfeat - 1))
    uncertainty = np.zeros_like(features)
    # TODO: fix for multiple
    for i, v in enumerate(np.arange(1, nfeat * ncond, 1)):
                                                      # conditions
        features[:, i] = f[v, :]  # / beta_vars[v,:]
        if 1:  # normalize features
            m, M = features[:, i].min(), features[:, i].max()
            features[:, i] = (features[:, i] - M) / (M - m) + 1

        uncertainty[:, i] = beta_vars[v, :]
    #features[:,nfeat-1] = ttp[:,cond]

    return ampl_dynrange, pval, features, uncertainty


# Parcellation

import pyhrf.graph as pgraph


from sklearn.cluster import Ward as WardSK


def spatial_ward_sk(features, graph, nb_clusters=0):
    connectivity2 = pgraph.graph_to_sparse_matrix(graph)
    ward = WardSK(n_clusters=nb_clusters,
                  connectivity=connectivity2).fit(features)
    ward.labels_ += 1
    return ward


def spatial_ward(features, graph, nb_clusters=0):
    # Spatial Ward parcellation. Returns distance and parcels
    connectivity2 = pgraph.graph_to_sparse_matrix(graph)
    ward = Ward(n_clusters=nb_clusters, connectivity=connectivity2).fit(
        features)  # control transpose
    ward.labels_ += 1
    ward.features = features
    ward.graph = graph

    return ward


def spatial_ward_with_uncertainty(features, graph, variance, activation,
                                  var_ini=None, act_ini=None, nb_clusters=0,
                                  dist_type='uward', cov_type='spherical', save_history=False):
    """
    Parcellation the given features with the spatial Ward algorithm, taking into
    account uncertainty on features (variance) and activation level:
        - the greater the variance of a given sample,
          the lower its importance in the distance.
        - the lower the activation level of a given sample,
          the lower its distance to any other sample.

    Args:

        - feature (np.ndarray):
             observations to parcellate - size: (nsamples, nfeatures)
        - graph (listof (listof PositionIndex)) :
             spatial dependency between positions
        - variance (np.ndarray):
             variance of features - size: (nsamples, nfeatures)
        - activation (np.ndarray):
             activation level associated with observation.
             size: (nsamples)
        - var_ini: !!!
        - act_ini: !!!
        - nb_clusters (int): number of clusters.
        - dist_type (str): 'ward' | 'mixt'

    """
    # Spatial Ward parcellation with uncertainty. Returns distance and parcels
    connectivity2 = pgraph.graph_to_sparse_matrix(graph)
    ward = Ward(n_clusters=nb_clusters, connectivity=connectivity2,
                dist_type=dist_type, cov_type=cov_type, save_history=save_history)
    # control transpose
    ward.fit(features, variance, activation, var_ini, act_ini)
    ward.features = features
    ward.variance = variance
    ward.activation = activation
    ward.graph = graph
    ward.labels_ += 1
    return ward


def parcellation_hemodynamics(fmri_data, feature_extraction_method,
                              parcellation_method, nb_clusters):
    """
    Perform a hemodynamic-driven parcellation on masked fMRI data

    Args:
        - fmri_data (pyhrf.core.FmriData): input fMRI data
        - feature_extraction_method (str): one of
          'glm_hderiv', 'glm_hdisp' ...
        - parcellation_method (str): one of
          'spatial_ward', 'spatial_ward_uncertainty', ...

    Return:
         parcellation array (numpy array of integers) with flatten
         spatial axes

    Examples #TODO
    """
    fmri_data.build_graphs()  # ensures that graphs are built
    roi_ids = np.unique(fmri_data.roi_ids_in_mask)
    if len(roi_ids) == 0:
        # glm
        amplitude, pvalues, features, uncertainty = feature_extraction(fmri_data,
                                                                       feature_extraction_method)
        # ncond=1, cond=0, dt=.5, time_length=25.)

        # parcellation process
        #graph = pgraph.graph_from_lattice(np.ones((5,5)), pgraph.kerMask2D_4n)
        if parcellation_method == 'spatial_ward':
            parcellation = spatial_ward(features, graph, nb_clusters)
        else:
            parcellation = spatial_ward_with_uncertainty(features, graph,
                                                         uncertainty, 1. - pvalues, nb_clusters)

    else:  # parcellate each ROI separately
        nb_voxels_all = fmri_data.nb_voxels_in_mask
        parcellation = np.zeros(fmri_data.nb_voxels_all, dtype=int)
        for rfd in fmri_data.roi_split():
            # multiply nb_clusters by the fraction of the roi size
            nb_clusters_roi = round(nb_clusters * rfd.nb_voxels_in_mask /
                                    nb_voxels_all)
            p_roi = parcellation_hemodynamics(rfd, feature_extraction_method,
                                              parcellation_method,
                                              nb_clusters_roi)
            parcellation += p_roi + parcellation.max()

    return parcellation


from pyhrf.parcellation import parcellation_dist


def assert_parcellation_equal(p1, p2, mask=None, tol=0, tol_pos=None):
    pdist = parcellation_dist(p1, p2, mask)

    if (tol_pos is not None and (tol_pos[np.where(pdist[1] == 0)] != 1).any()) or \
            (tol_pos is None and pdist[0] > tol):
        msg = 'Parcellation are not equal. %d differing positions:\n' \
            % pdist[0]
        msg += str(pdist[1])
        raise AssertionError(msg)


def align_parcellation(p1, p2, mask=None):
    """
    Align two parcellation p1 and p2 as the minimum
    number of positions to remove in order to obtain equal partitions.
    Return:
        (p2 aligned to p1)
    """
    assert np.issubdtype(p1.dtype, np.int)
    assert np.issubdtype(p2.dtype, np.int)
    from munkres import Munkres
    from pyhrf.cparcellation import compute_intersection_matrix
    if mask is None:
        mask = (p1 != 0)
    m = np.where(mask)
    logger.debug('Nb pos inside mask: %d', len(m[0]))
    fp1 = p1[m].astype(np.int32)
    fp2 = p2[m].astype(np.int32)
    cost_matrix = np.zeros((fp1.max() + 1, fp2.max() + 1), dtype=np.int32)
    logger.debug('Cost matrix : %s', str(cost_matrix.shape))
    compute_intersection_matrix(fp1, fp2, cost_matrix)
    # discard 0-labelled parcels (background)
    cost_matrix = cost_matrix[1:, 1:]
    # solve the assignement problem:
    indexes = np.array(Munkres().compute((cost_matrix * -1).tolist()))
    p2_aligned = np.zeros_like(p2)
    for i in range(indexes.shape[0]):
        p2_aligned[np.where(p2 == indexes[i, 1] + 1)] = indexes[i, 0] + 1
    return p2_aligned


def mixtp_to_str(mp):
    n_feat = (len(mp) - 1) / 2
    return 'mu0: %s, mu1: %s, a: %1.2f' % (str(mp[:n_feat]), str(mp[n_feat:2 * n_feat]),
                                           mp[-1])

from pyhrf.ndarray import xndarray, MRI3Daxes
import os.path as op


def ward_tree_save(tree, output_dir, mask):

    def expand_and_save(x, anames, adoms, fn):
        prefix = 'parcellation_%s_' % tree.dist_type
        fn = add_prefix(op.join(output_dir, fn), prefix)
        c = xndarray(x, anames, adoms)
        if 'voxel' in anames:
            c.expand(mask, 'voxel', MRI3Daxes[:mask.ndim]).save(fn)
        else:
            c.save(fn)

    # save inputs:
    expand_and_save(tree.features, ['voxel', 'feature'], {}, 'features.nii')
    if hasattr(tree, 'variance'):
        expand_and_save(
            tree.variance, ['voxel', 'feature'], {}, 'variances.nii')
        expand_and_save(tree.activation, ['voxel'], {}, 'activations.nii')

    # save history
    expand_and_save(tree.history, ['iteration', 'voxel'], {}, 'history.nii')
    if tree.history_choices is not None:
        expand_and_save(tree.history_choices, ['iteration', 'candidate', 'voxel'],
                        {}, 'choice_history.nii')
        expand_and_save(tree.history_choices_inertia, ['iteration', 'candidate'],
                        {}, 'choice_history_inertia.nii')

    # save moments:
    expand_and_save(tree.moments, ['iteration', 'moment'], {}, 'moments.nii')

    # save final output:
    expand_and_save(tree.labels_, ['voxel'], {}, 'labels.nii')


def render_ward_tree(tree, fig_fn, leave_colors=None):

    leave_labels = ['%d: f%s - v%s - a%1.2f' % (i, str(f), str(v), a)
                    for i, (f, v, a) in enumerate(zip(tree.features,
                                                      tree.variance,
                                                      tree.activation))]

    n_leaves = len(leave_labels)
    leave_colors = leave_colors or ["black"] * n_leaves

    import pygraphviz as pgv
    g = pgv.AGraph(directed=True)

    def format_item(i):
        if i < n_leaves:
            return '%s' % leave_labels[i]
        else:  # group node
            return '%d' % i

    # add spatial associations as undirected groups:
    for l, ln in enumerate(tree.graph):
        for n in ln:
            if n > l:
                g.add_edge(format_item(l), format_item(n), dir="none")

    for l in xrange(n_leaves):
        n = g.get_node(format_item(l))
        n.attr['shape'] = 'box'
        n.attr['color'] = leave_colors[l]

    # add agglomerative associations as directed groups:
    for ic, c in enumerate(tree.children_):
        g.add_edge(format_item(c[0]), format_item(n_leaves + ic), color="blue")
        g.add_edge(format_item(c[1]), format_item(n_leaves + ic), color="blue")

    for l in xrange(n_leaves, len(tree.heights)):
        n = g.get_node(format_item(l))
        n.attr['color'] = 'blue'

    # render group info in another component:
    if 1:
        for ic in xrange(len(tree.children_)):
            if ic > 0:
                if tree.dist_type == 'uward':
                    smom0 = str(tree.moments[n_leaves + ic - 1])
                    smom1 = str(tree.moments[n_leaves + ic])
                else:  # dist_type == mixt
                    smom0 = mixtp_to_str(
                        tuple(tree.moments[n_leaves + ic - 1]))
                    smom1 = mixtp_to_str(tuple(tree.moments[n_leaves + ic]))
                g.add_edge(format_item(n_leaves + ic - 1) + ", iner=%1.3f, mom=%s"
                           % (tree.heights[n_leaves + ic - 1], smom0),
                           format_item(n_leaves + ic) + ", iner=%1.3f, mom=%s"
                           % (tree.heights[n_leaves + ic], smom1))

    g.layout('dot')
    g.draw(fig_fn)
