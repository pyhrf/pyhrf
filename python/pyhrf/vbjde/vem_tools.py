# -*- coding: utf-8 -*-
"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

import os.path as op
import numpy as np
from numpy.matlib import *
import scipy as sp
from scipy.linalg import toeplitz
import np
from pyhrf.ndarray import xndarray
import time
from pyhrf.paradigm import restarize_events
import UtilsC
import pyhrf
from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.tools.io import read_volume,write_volume
from pyhrf.tools import format_duration
from pyhrf.boldsynth.hrf import genBezierHRF
from pyhrf.boldsynth.hrf import getCanoHRF


try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


# Tools
##############################################################

eps = 1e-4

def polyFit(signal, tr, order,p):
    n = len(signal)
    ptp = np.dot(p.transpose(),p)
    invptp = np.linalg.inv(ptp)
    invptppt = np.dot(invptp, p.transpose())
    l = np.dot(invptppt,signal)
    return l

def PolyMat( Nscans , paramLFD , tr):
    '''Build polynomial basis'''
    regressors = tr * np.arange(0, Nscans)
    timePower = np.arange(0,paramLFD+1, dtype=int)
    regMat = np.zeros((len(regressors),paramLFD+1),dtype=np.float64)

    for v in xrange(paramLFD+1):
        regMat[:,v] = regressors[:]
    #tPowerMat = np.matlib.repmat(timePower, Nscans, 1)
    tPowerMat = repmat(timePower, Nscans, 1)
    lfdMat = np.power(regMat,tPowerMat)
    lfdMat = np.array(sp.linalg.orth(lfdMat))
    return lfdMat

def compute_mat_X_2(nbscans, tr, lhrf, dt, onsets, durations=None):
    if durations is None: #assume spiked stimuli
        durations = np.zeros_like(onsets)
    osf = tr/dt # over-sampling factor
    if int(osf) != osf: #construction will only work if dt is a multiple of tr
        raise Exception('OSF (%f) is not an integer' %osf)

    x = np.zeros((nbscans,lhrf),dtype=float)
    tmax = nbscans * tr #total session duration
    lgt = (nbscans + 2) * osf # nb of scans if tr=dt
    paradigm_bins = restarize_events(onsets, np.zeros_like(onsets), dt, tmax)
    firstcol = np.concatenate( (paradigm_bins, np.zeros(lgt-len(paradigm_bins))) )
    firstrow = np.concatenate( ([paradigm_bins[0]], np.zeros(lhrf-1, dtype=int)) )
    x_tmp = np.array(toeplitz( firstcol, firstrow), dtype=int)
    os_indexes = [(np.arange(nbscans)*osf).astype(int)]
    x = x_tmp[os_indexes]
    return x

def buildFiniteDiffMatrix(order, size):
    o = order
    a = np.diff(np.concatenate((np.zeros(o),[1],np.zeros(o))),n=o)
    b = a[len(a)/2:]
    diffMat = toeplitz(np.concatenate((b, np.zeros(size-len(b)))))
    return diffMat

# Maximization functions
##############################################################

def maximization_mu_sigma(Mu,Sigma,q_Z,m_A,K,M,Sigma_A):
    for m in xrange(0,M):
        for k in xrange(0,K):
            #S = sum( q_Z[m,k,:] ) + eps
            S = sum( q_Z[m,k,:] )
            if S == 0.: 
                S = eps
            Sigma[m,k] = sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + Sigma_A[m,m,:] ) ) / S
            if Sigma[m,k] < eps:
                Sigma[m,k] = eps
            if k != 0 : # mu_0 = 0 a priori
                #Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
                Mu[m,k] = sum( q_Z[m,k,:] * m_A[:,m] ) / S
            else:
                Mu[m,k] = 0. 
    return Mu , Sigma

def maximization_L(Y,m_A,X,m_H,L,P,zerosP):
    J = Y.shape[1]
    for i in xrange(0,J):
        S = zerosP.copy()
        m = 0
        for k in X:
            S += m_A[i,m]*np.dot(X[k],m_H)
            m +=1
        L[:,i] = np.dot(P.transpose(), Y[:,i] - S)
    return L

def maximization_sigmaH(D,Sigma_H,R,m_H):
    sigmaH = (np.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
    sigmaH /= D
    return sigmaH

def maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h):
    alpha = (np.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
    #sigmaH = (D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)
    sigmaH = (-D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)

    return sigmaH
       
       
# Other functions
##############################################################

def computeFit(m_H, m_A, X, J, N): 
  #print 'Computing Fit ...'
  stimIndSignal = np.zeros((N,J), dtype=np.float64)
  for i in xrange(0,J):
    m = 0
    for k in X:
      stimIndSignal[:,i] += m_A[i,m] * np.dot(X[k],m_H)
      m += 1
  return stimIndSignal


def Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,Model):
    ### Compute Free Energy  
    
    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    if Model=="CompMod":
        Total_Entropy = EntropyA + EntropyH + EntropyZ

    ## Second Part (likelihood)
    EPtildeLikelihood = UtilsC.expectation_Ptilde_Likelihood(y_tilde,m_A,m_H,XX.astype(int32),Sigma_A,sigma_epsilone,Sigma_H,Gamma,p_Wtilde,XGamma,J,D,M,N,Det_Gamma)
    EPtildeA = UtilsC.expectation_Ptilde_A(m_A,Sigma_A,p_Wtilde,q_Z,mu_M,sigma_M,J,M,K)
    EPtildeH = UtilsC.expectation_Ptilde_H(R, m_H, Sigma_H, D, sigmaH, Det_invR)
    EPtildeZ = UtilsC.expectation_Ptilde_Z(q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)
    #EPtildeZ = UtilsC.expectation_Ptilde_Z_MF_Begin(q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)
    if Model=="CompMod":
        pyhrf.verbose(5,"Computing Free Energy for CompMod")
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ

    FreeEnergy = EPtilde - Total_Entropy

    return FreeEnergy

    