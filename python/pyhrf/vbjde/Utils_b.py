# -*- coding: utf-8 -*-
"""For the Utils documentation

"""
try:
    from pywt import dwt,idwt,dwt2,idwt2
except ImportError:
    pass

#from scipy.spatial import *
import os.path as op
from scipy.linalg import toeplitz,norm,inv
import numpy
from pyhrf.ndarray import xndarray
import time
import scipy

from pyhrf.paradigm import restarize_events
#from nifti import NiftiImage
import csv
#from pyhrf import *
from pyhrf.boldsynth.hrf import getCanoHRF
import UtilsC
import pyhrf
#import UtilsModule
from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.tools.io import read_volume,write_volume
from pyhrf.boldsynth.hrf import genBezierHRF
import numpy as np
import scipy as sp
import scipy.sparse
from pyhrf.tools import format_duration

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


# Tools
##############################################################

def mult(v1,v2):
    """For the Utils documentation

    """
    matrix = numpy.zeros((len(v1),len(v2)),dtype=float)
    for i in xrange(len(v1)):
        for j in xrange(len(v2)):
                matrix[i,j] += v1[i]*v2[j]
    return matrix

eps = 1e-4

def polyFit(signal, tr, order,p):
    """For the Utils documentation

    """
    n = len(signal)
    ptp = numpy.dot(p.transpose(),p)
    invptp = numpy.linalg.inv(ptp)
    invptppt = numpy.dot(invptp, p.transpose())
    l = numpy.dot(invptppt,signal)
    return l

from numpy.matlib import *
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
    lfdMat = np.array(scipy.linalg.orth(lfdMat))
    return lfdMat

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(numpy.sqrt(2*numpy.pi)*abs(sigma)))*numpy.exp(-u*u/2)
    return y

def compute_mat_X_2(nbscans, tr, lhrf, dt, onsets, durations=None):
    if durations is None: #assume spiked stimuli
        durations = numpy.zeros_like(onsets)
    osf = tr/dt # over-sampling factor
    if int(osf) != osf: #construction will only work if dt is a multiple of tr
        raise Exception('OSF (%f) is not an integer' %osf)

    x = numpy.zeros((nbscans,lhrf),dtype=float)
    tmax = nbscans * tr #total session duration
    lgt = (nbscans + 2) * osf # nb of scans if tr=dt
    paradigm_bins = restarize_events(onsets, numpy.zeros_like(onsets), dt, tmax)
    firstcol = numpy.concatenate( (paradigm_bins, numpy.zeros(lgt-len(paradigm_bins))) )
    firstrow = numpy.concatenate( ([paradigm_bins[0]], numpy.zeros(lhrf-1, dtype=int)) )
    x_tmp = numpy.array(toeplitz( firstcol, firstrow), dtype=int)
    os_indexes = [(numpy.arange(nbscans)*osf).astype(int)]
    x = x_tmp[os_indexes]
    return x

def maximum(a):
    maxx = a[0]
    maxx_ind = 0
    for i in xrange(len(a)):
        if a[i] > maxx :
            maxx = a[i]
            maxx_ind = i

    return maxx, maxx_ind


# Expectation functions
##############################################################

def expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_MK,q_Z,mu_MK,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD):
    X_tilde = zerosJMD.copy()#numpy.zeros((Y.shape[1],M,D),dtype=float)
    J = Y.shape[1]
    for i in xrange(0,J):
        m = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Sigma_A[m,m2,i] = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H)
                Sigma_A[m,m2,i] += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
                m2 += 1
            X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
            m += 1
        tmp = numpy.dot(X_tilde[i,:,:],m_H)
        for k in xrange(0,K):
            Delta = numpy.diag( q_Z[:,k,i]/(sigma_MK[:,k] + eps) )
            tmp += numpy.dot(Delta,mu_MK[:,k])
            Sigma_A[:,:,i] += Delta
        tmp2 = inv(Sigma_A[:,:,i])
        Sigma_A[:,:,i] = tmp2
        m_A[i,:] = numpy.dot(Sigma_A[:,:,i],tmp)
    return Sigma_A, m_A

def expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD):
    Y_bar_tilde = zerosD.copy()#numpy.zeros((D),dtype=float)
    Q_bar = scale * R/sigmaH
    Q_bar2 = scale * R/sigmaH
    for i in xrange(0,J):
        m = 0
        tmp =  zerosND.copy() #numpy.zeros((N,D),dtype=float)
        for k in X: # Loop over the M conditions
            tmp += m_A[i,m] * X[k]
            m += 1
        Y_bar_tilde += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),y_tilde[:,i])
        Q_bar += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),tmp)
        Q_bar2[:,:] = Q_bar[:,:]
        m1 = 0
        for k1 in X: # Loop over the M conditions
            m2 = 0
            for k2 in X: # Loop over the M conditions
                Q_bar += Sigma_A[m1,m2,i] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[i],eps)),X[k2])
                m2 +=1
            m1 +=1
    Sigma_H = inv(Q_bar)
    m_H = numpy.dot(Sigma_H,Y_bar_tilde)
    m_H[0] = 0
    m_H[-1] = 0
    return Sigma_H, m_H

def expectation_H_constrained(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD):
    Y_bar_tilde = zerosD.copy() #numpy.zeros((D),dtype=float)
    Q_bar = scale * R/sigmaH
    Q_bar2 = scale * R/sigmaH
    for i in xrange(0,J):
        m = 0
        tmp =  zerosND.copy() #numpy.zeros((N,D),dtype=float)
        for k in X: # Loop over the M conditions
            tmp += m_A[i,m] * X[k]
            m += 1
        Y_bar_tilde += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),y_tilde[:,i])
        Q_bar += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),tmp)
        Q_bar2[:,:] = Q_bar[:,:]
        m1 = 0
        for k1 in X: # Loop over the M conditions
            m2 = 0
            for k2 in X: # Loop over the M conditions
                Q_bar += Sigma_A[m1,m2,i] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[i],eps)),X[k2])
                m2 +=1
            m1 +=1
    Sigma_H = inv(Q_bar)
    m_H = numpy.dot(Sigma_H,Y_bar_tilde)
    m_H[0] = 0
    m_H[-1] = 0
    return Sigma_H, m_H

def expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK):
    energy = zerosK.copy()
    Gauss = zerosK.copy()
    for i in xrange(0,J):
        for m in xrange(0,M):
            alpha = -0.5*Sigma_A[m,m,i] / (sigma_M[m,:] + eps)
            Malpha = alpha.mean()
            alpha /= Malpha
            tmp = sum(Z_tilde[m,:,graph[i]],0)
            for k in xrange(0,K):
                extern_field = alpha[k] + max(numpy.log( normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k])) + eps) ,-100 )
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
            Emax = max(energy)
            Probas = numpy.exp(energy - Emax)
            Sum = sum(Probas)
            Z_tilde[m,:,i] = Probas/ (Sum + eps)
    for i in xrange(0,J):
        for m in xrange(0,M):
            alpha = -0.5*Sigma_A[m,m,i] / (sigma_M[m,:] + eps)
            Malpha = alpha.mean()
            alpha /= Malpha
            tmp = sum(Z_tilde[m,:,graph[i]],0)
            for k in xrange(0,K):
                extern_field = alpha[k]
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
                Gauss[k] = normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k]))
            Emax = max(energy)
            Probas = numpy.exp(energy - Emax)
            Sum = sum(Probas)
            q_Z[m,:,i] = Gauss * Probas / Sum
            SZ = sum(q_Z[m,:,i])
            q_Z[m,:,i] /= SZ
    return q_Z, Z_tilde



# Maximization functions
##############################################################

def maximization_mu_sigma(Mu,Sigma,q_Z,m_A,K,M,Sigma_A):
    for m in xrange(0,M):
        for k in xrange(0,K):
            #S = sum( q_Z[m,k,:] ) + eps
            S = sum( q_Z[m,k,:] )
            if S == 0.: 
                S = eps
                #raise Exception('PROBLEEEEEEEEEM : Divising by Zeeerooooooooo ....')

            Sigma[m,k] = sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + Sigma_A[m,m,:] ) ) / S
            if Sigma[m,k] < eps:
                Sigma[m,k] = eps
                #raise Exception('PROBLEEEEEEEEEM : Very Low Variance ....')
                
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
            S += m_A[i,m]*numpy.dot(X[k],m_H)
            m +=1
        L[:,i] = numpy.dot(P.transpose(), Y[:,i] - S)
    return L

def gradient(q_Z,Z_tilde,J,m,K,graph,beta,gamma):
    Gr = gamma
    for i in xrange(0,J):
        tmp2 = beta * sum(Z_tilde[m,:,graph[i]],0)
        Emax = max(tmp2)
        Sum = sum( numpy.exp( tmp2 - Emax ) )
        for k in xrange(0,K):
            tmp = sum(Z_tilde[m,k,graph[i]],0)
            energy = beta * tmp
            Pzmi = numpy.exp(energy - Emax)
            Pzmi /= (Sum + eps)
            Gr += tmp * (-q_Z[m,k,i] + Pzmi)
    return Gr

def maximization_beta(beta,q_Z,Z_tilde,J,K,m,graph,gamma,neighbour,maxNeighbours):
    Gr = 100
    step = 0.005
    ni = 1
    while ((abs(Gr) > 0.0001) and (ni < 200)):
        Gr = gradient(q_Z,Z_tilde,J,m,K,graph,beta,gamma)
        beta -= step * Gr
        ni+= 1
    return max(beta,eps)

def maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM):
    N = PL.shape[0]
    J = Y.shape[1]
    Htilde = zerosMM.copy() #numpy.zeros((M,M),dtype=float)
    for i in xrange(0,J):
        S = numpy.zeros((N),dtype=float)
        m = 0
        for k in X:
            m2 = 0
            for k2 in X:
                Htilde[m,m2] =  numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k].transpose()),X[k2]),m_H)
                Htilde[m,m2] += (numpy.dot(numpy.dot(Sigma_H,X[k].transpose()),X[k2])).trace()
                m2 += 1
            S += m_A[i,m]*numpy.dot(X[k],m_H)
            m += 1
        sigma_epsilone[i] = numpy.dot( -2*S, Y[:,i] - PL[:,i] )
        sigma_epsilone[i] += (numpy.dot(Sigma_A[:,:,i],Htilde)).trace()
        sigma_epsilone[i] += numpy.dot( numpy.dot(m_A[i,:].transpose(), Htilde),m_A[i,:] )
        sigma_epsilone[i] += numpy.dot((Y[:,i] - PL[:,i]).transpose(), Y[:,i] - PL[:,i] )
        sigma_epsilone[i] /= N
    return sigma_epsilone

#def ReadNII(fname):
    #nim = NiftiImage(fname)
    #D = nim.data
    #return D

def buildFiniteDiffMatrix(order, size):
    o = order
    a = numpy.diff(numpy.concatenate((numpy.zeros(o),[1],numpy.zeros(o))),n=o)
    b = a[len(a)/2:]
    diffMat = toeplitz(numpy.concatenate((b, numpy.zeros(size-len(b)))))
    return diffMat

def maximization_sigmaH(D,Sigma_H,R,m_H):
    sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
    sigmaH /= D
    return sigmaH

def maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h):
    alpha = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
    #sigmaH = (D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)
    sigmaH = (-D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)

    return sigmaH
       
def computeFit(m_H, m_A, X, J, N): 
  #print 'Computing Fit ...'
  stimIndSignal = numpy.zeros((N,J), dtype=numpy.float64)

  for i in xrange(0,J):
    m = 0
    for k in X:
      stimIndSignal[:,i] += m_A[i,m] * numpy.dot(X[k],m_H)
      m += 1

  return stimIndSignal

eps_FreeEnergy = 0.00000001


# Entropy functions 
##############################################################

def A_Entropy(Sigma_A, M, J):
    # BRLS entropy
    pyhrf.verbose(3,'Computing NRLs Entropy ...')
    Det_Sigma_A_j = numpy.zeros(J,dtype=numpy.float64)
    Entropy = 0.0
    for j in xrange(0,J):
        Det_Sigma_A_j = numpy.linalg.det(Sigma_A[:,:,j])
        Const = (2*numpy.pi*numpy.exp(1))**M
        Entropy_j = numpy.sqrt( Const * Det_Sigma_A_j)
        Entropy += numpy.log(Entropy_j + eps_FreeEnergy)
    Entropy = - Entropy

    return Entropy

def H_Entropy(Sigma_H, D):
    # HRF entropy
    pyhrf.verbose(3,'Computing HRF Entropy ...')
    Det_Sigma_H = numpy.linalg.det(Sigma_H)
    Const = (2*numpy.pi*numpy.exp(1))**D
    Entropy = numpy.sqrt( Const * Det_Sigma_H)
    Entropy = - numpy.log(Entropy + eps_FreeEnergy)

    return Entropy

def Z_Entropy(q_Z, M, J):
    # Labels entropy
    pyhrf.verbose(3,'Computing Z Entropy ...')
    Entropy = 0.0
    for j in xrange(0,J):
        for m in xrange(0,M):
            Entropy += q_Z[m,1,j] * numpy.log(q_Z[m,1,j] + eps_FreeEnergy) + q_Z[m,0,j] * numpy.log(q_Z[m,0,j] + eps_FreeEnergy)
    return Entropy

##############################################################

def Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,Model):
    ## Free Energy  
    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    if Model=="CompMod":
        Total_Entropy = EntropyA + EntropyH + EntropyZ
    #print 'Total Entropy =', Total_Entropy

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



#VBJDE Functions
##############################################################

def Main_vbjde_Extension(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):    
    #def Main_vbjde_Extension(HRFDict,graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0):    
    pyhrf.verbose(1,"Fast EM with C extension started ...")

    numpy.random.seed(6537546)

    tau1 = 0.0
    tau2 = 0.0
    S = 100
    Init_sigmaH = sigmaH

    Nb2Norm = 1
    NormFlag = False    
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5#7.5
    #gamma_h = 1000
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5
    
    # Initialize sizes vectors
    #D = int(numpy.ceil(Thrf/dt)) ##############################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    condition_names = []

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
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
    
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        m1 += 1
    
    if MiniVEMFlag: 
        pyhrf.verbose(1,"MiniVEM to choose the best initialisation...")
        InitVar, InitMean, gamma_h = MiniVEM_CompMod(Thrf,TR,dt,beta,Y,K,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,p_Wtilde,scale,Q_barnCond,XGamma,tau1,tau2,NbItMiniVem,sigmaH,estimateHRF)

    sigmaH = Init_sigmaH
    sigma_epsilone = numpy.ones(J)
    if 0:
        pyhrf.verbose(3,"Labels are initialized by setting active probabilities to zeros ...")
        q_Z = numpy.ones((M,K,J),dtype=numpy.float64)
        q_Z[:,1,:] = 0
    if 0:
        pyhrf.verbose(3,"Labels are initialized randomly ...")
        q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
        nbVoxInClass = J/K
        for j in xrange(M) :
            if J%2==0:
                l = []
            else:
                l = [0]
            for c in xrange(K) :
                l += [c] * nbVoxInClass
            q_Z[j,0,:] = numpy.random.permutation(l)
            q_Z[j,1,:] = 1. - q_Z[j,0,:]
    if 1:
        pyhrf.verbose(3,"Labels are initialized by setting active probabilities to ones ...")
        q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
        q_Z[:,1,:] = 1
        
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)   
    Z_tilde = q_Z.copy()
    
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    m_h = m_h[:D]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    sigmaH1 = sigmaH
    if estimateHRF:
        Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
        Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)
    
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    Ndrift = L.shape[0]

    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.5
    sigma_M[:,1] = 0.6
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)    
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)    
    for j in xrange(0,J):
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += np.random.normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
            
    t1 = time.time()
    
    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        #t01 = time.time()
        UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        
        val = reshape(m_A,(M*J))
        val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ np.where((val>=-1e-50) & (val<0.0)) ] = 0.0
        #m_A = reshape(val, (J,M))
        
        if estimateHRF:
            UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            h_norm += [norm(m_H)]
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
            # Plotting HRF
            if PLOT and ni >= 0:
                import matplotlib.pyplot as plt
                plt.figure(M+1)
                plt.plot(m_H)
                plt.hold(True)
        else:
            if TrueHrfFlag:
                #TrueVal, head = read_volume(HrfFilename)
                TrueVal, head = read_volume(HrfFilename)[:,0,0,0]
                print TrueVal
                print TrueVal.shape
                m_H = TrueVal
                
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_RVM_and_CompMod(Sigma_A,m_A,sigma_M,Beta,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                #UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]            
        
        val = reshape(q_Z,(M*K*J))
        val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
        #q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        
        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            mu1[m] += [mu_M[m,1]]
        
        UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                if not MFapprox:
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
        
        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]
        
        #print 'sigma_noise =',sigma_epsilone
        #t02 = time.time()
        #cTime += [t02-t1]
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]

    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

    val = reshape(m_A,(M*J))
    val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ np.where((val>=-1e-50) & (val<0.0)) ] = 0.0
    #m_A = reshape(val, (J,M))

    if estimateHRF:
      UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
      m_H[0] = 0
      m_H[-1] = 0
      h_norm += [norm(m_H)]
      # Normalizing H at each Nb2Norm iterations:
      if NormFlag:
          if (ni+2)%Nb2Norm == 0:
              Norm = norm(m_H)
              m_H /= Norm
              Sigma_H /= Norm**2
              m_A *= Norm
              Sigma_A *= Norm**2
      # Plotting HRF        
      if PLOT and ni >= 0:
          import matplotlib.pyplot as plt
          plt.figure(M+1)
          plt.plot(m_H)
          plt.hold(True)
    
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)[:,0,0,0]
            m_H = TrueVal
    
    #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    #Crit_A = sum(DIFF) / len(find(DIFF != 0))
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]    
        
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        if MFapprox:
            UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_RVM_and_CompMod(Sigma_A,m_A,sigma_M,Beta,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
    #q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)

    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        mu1[m] += [mu_M[m,1]]
        
    UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:    
                #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)
    UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
    
    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2

    
    if ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            #t01 = time.time()
            UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ np.where((val>=-1e-50) & (val<0.0)) ] = 0.0
            #m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                h_norm += [norm(m_H)]
                if NormFlag:
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                # Plotting HRF        
                if PLOT and ni >= 0:
                    import matplotlib.pyplot as plt
                    plt.figure(M+1)
                    plt.plot(m_H)
                    plt.hold(True)
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)[:,0,0,0]
                    m_H = TrueVal
            
            #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            #Crit_A = sum(DIFF) / len(find(DIFF != 0))
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]       
                    
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]
            
            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_RVM_and_CompMod(Sigma_A,m_A,sigma_M,Beta,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            #ion()
            #figure(6).clf()
            #for m in range(0,M):
                #for k in range(0,K):
                    #z1 = q_Z[m,k,:]
                    #z2 = reshape(z1,(l,l))
                    #figure(6)
                    #subplot(M,K,1 + m*K + k)
                    #imshow(z2,interpolation='nearest')
                    #title("m = " + str(m) +"k = " + str(k))
                    #colorbar()
                    #hold(False)
            #draw()

            val = reshape(q_Z,(M*K*J))
            val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
            #q_Z = reshape(val, (M,K,J))

            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]

            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
                    
            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            
            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                mu1[m] += [mu_M[m,1]]
                
            UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                        Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)
            UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
            
            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]
            
            ni +=1
            t02 = time.time()
            cTime += [t02-t1]
    t2 = time.time()
    
    #FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    FreeEnergyArray = numpy.zeros((ni),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    #for i in xrange(ni-1,NitMax+1):
        #FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    h_norm_array = numpy.zeros((ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]
        #for i in xrange(ni-1,NitMax+1):
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]

    
    if PLOT:
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'size'   : 15}
        matplotlib.rc('font', **font)
        plt.savefig('./HRF_Iter_CompMod.png')
        plt.hold(False)
        plt.figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        plt.plot(cAH[1:-1],'lightblue')
        plt.hold(True)
        plt.plot(cFE[1:-1],'m')
        plt.hold(False)
        #plt.legend( ('CA','CH', 'CZ', 'CAH', 'CFE') )
        plt.legend( ('CAH', 'CFE') )
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
    cTimeMean = CompTime/ni
    
    if not NormFlag:
        Norm = norm(m_H)
        m_H /= Norm
        Sigma_H /= Norm**2
        sigmaH /= Norm**2
        m_A *= Norm
        Sigma_A *= Norm**2
        mu_M *= Norm
        sigma_M *= Norm**2
        
    sigma_M = sqrt(sqrt(sigma_M))
    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            pyhrf.verbose(3, 'Compute contrasts ...')
            nrls_conds = dict([(str(cn), m_A[:,ic]) \
                                   for ic,cn in enumerate(condition_names)] )
            n = 0
            #print contrasts
            #print nrls_conds
            #raw_input('')
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                print 
                CONTRAST[:,n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = numpy.zeros(M,dtype=float)
                ind_conds0 = {}
                for m in xrange(0,M):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0,M):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname],ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * numpy.ones(M,dtype=float)
                #print ContrastCoef
                #print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    #print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    #print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
        
    StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR comp =', SNR
    return ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cAH[2:],cTime[2:],cTimeMean,Sigma_A,StimulusInducedSignal,FreeEnergyArray



def Main_vbjde_Extension_constrained(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):    
    pyhrf.verbose(1,"Fast EM with C extension started ...")
    numpy.random.seed(6537546)

    #######################################################################################################################
    #####################################################################################        INITIALIZATIONS
    #Initialize parameters
    tau1 = 0.0
    tau2 = 0.0
    S = 100
    Init_sigmaH = sigmaH
    Nb2Norm = 1
    NormFlag = False    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5#7.5
    #gamma_h = 1000
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5
    estimateLabels=True #WARNING!! They should be estimated
    
    # Initialize sizes vectors
    D = int(numpy.ceil(Thrf/dt)) + 1 #D = int(numpy.ceil(Thrf/dt)) 
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    condition_names = []

    # Neighbours
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    # Conditions
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    # Covariance matrix
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
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
    
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        m1 += 1
    
    if MiniVEMFlag: 
        pyhrf.verbose(1,"MiniVEM to choose the best initialisation...")
        InitVar, InitMean, gamma_h = MiniVEM_CompMod(Thrf,TR,dt,beta,Y,K,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,p_Wtilde,scale,Q_barnCond,XGamma,tau1,tau2,NbItMiniVem,sigmaH,estimateHRF)

    sigmaH = Init_sigmaH
    sigma_epsilone = numpy.ones(J)
    pyhrf.verbose(3,"Labels are initialized by setting active probabilities to ones ...")
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)   
    Z_tilde = q_Z.copy()
    
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    m_h = m_h[:D]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    sigmaH1 = sigmaH
    if estimateHRF:
        Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
        Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)
    
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    Ndrift = L.shape[0]

    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.5
    sigma_M[:,1] = 0.6
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)    
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)    
    for j in xrange(0,J):
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += np.random.normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
            
    t1 = time.time()
    
    #######################################################################################################################
    ####################################################################################    VBJDE num. iter. minimum

    ni = 0
    
    while ((ni < NitMin) or (((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax))):
        
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        
        #####################
        # EXPECTATION
        #####################
        
        # A 
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        val = reshape(m_A,(M*J))
        val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ np.where((val>=-1e-50) & (val<0.0)) ] = 0.0
        
        # crit. A
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        # HRF h
        if estimateHRF:
            ################################
            #  HRF ESTIMATION
            ################################
            UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            
            import cvxpy as cvx
            # data: Sigma_H, m_H
            m,n = Sigma_H.shape    
            Sigma_H_inv = np.linalg.inv(Sigma_H)
            zeros_H = np.zeros_like(m_H[:,np.newaxis])
            
            # Construct the problem. PRIMAL
            h = cvx.Variable(n)
            expression = cvx.quad_form(h - m_H[:,np.newaxis], Sigma_H_inv) 
            objective = cvx.Minimize(expression)
            constraints = [h[0] == 0, h[-1]==0, h >= zeros_H, cvx.square(cvx.norm(h,2))<=1]    
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(verbose=1,solver=cvx.CVXOPT)            

            # Now we update the mean of h 
            m_H_old = m_H  
            Sigma_H_old = Sigma_H
            m_H = np.squeeze(np.array((h.value)))            
            Sigma_H = np.zeros_like(Sigma_H)    
            
            h_norm += [norm(m_H)]
            print 'h_norm = ', h_norm
            
            # Plotting HRF
            if PLOT and ni >= 0:
                import matplotlib.pyplot as plt
                plt.figure(M+1)
                plt.plot(m_H)
                plt.hold(True)
        else:
            if TrueHrfFlag:
                #TrueVal, head = read_volume(HrfFilename)
                TrueVal, head = read_volume(HrfFilename)[:,0,0,0]
                print TrueVal
                print TrueVal.shape
                m_H = TrueVal
        
        # crit. h
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        # crit. AH
        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        # Z labels
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_RVM_and_CompMod(Sigma_A,m_A,sigma_M,Beta,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                #UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]            
        
        # crit. Z 
        val = reshape(q_Z,(M*K*J))
        val[ np.where((val<=1e-50) & (val>0.0)) ] = 0.0
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ np.where( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ np.where( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #####################
        # MAXIMIZATION
        #####################
        
        # HRF: Sigma_h
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H_old,R,m_H_old,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))

        # (mu,sigma)
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M, sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            mu1[m] += [mu_M[m,1]]
        
        # Drift L
        UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        # Beta
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                if not MFapprox:
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)
        
        # Sigma noise
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
        
        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        # Update index
        ni += 1
        
        t02 = time.time()
        cTime += [t02-t1]
    
    t2 = time.time()
    
    #######################################################################################################################
    ####################################################################################    PLOTS and SNR computation
    
    FreeEnergyArray = numpy.zeros((ni),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]

    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    h_norm_array = numpy.zeros((ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]

    if PLOT:
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'size'   : 15}
        matplotlib.rc('font', **font)
        plt.savefig('./HRF_Iter_CompMod.png')
        plt.hold(False)
        plt.figure(2)
        plt.plot(cAH[1:-1],'lightblue')
        plt.hold(True)
        plt.plot(cFE[1:-1],'m')
        plt.hold(False)
        #plt.legend( ('CA','CH', 'CZ', 'CAH', 'CFE') )
        plt.legend( ('CAH', 'CFE') )
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
    cTimeMean = CompTime/ni

    sigma_M = sqrt(sqrt(sigma_M))
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    #print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    #print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
        
    StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR comp =', SNR
    return ni,m_A,m_H, q_Z,sigma_epsilone,mu_M,sigma_M,Beta,L,PL,CONTRAST,CONTRASTVAR,cA[2:],cH[2:],cZ[2:],cAH[2:],cTime[2:],cTimeMean,Sigma_A,StimulusInducedSignal,FreeEnergyArray


    