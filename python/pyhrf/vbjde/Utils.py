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
import pyhrf
#import UtilsModule
import UtilsC
from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.tools._io import read_volume,write_volume
from pyhrf.boldsynth.hrf import genBezierHRF
import numpy as np
import scipy as sp
import scipy.sparse
from pyhrf.tools import format_duration

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

def tridiag(l,autocorr,rho):
    diag_rows = numpy.array([-rho*numpy.ones(l),
                autocorr*numpy.ones(l),
                -rho*numpy.ones(l)])
    positions = [-1, 0, 1]
    return sp.sparse.spdiags(diag_rows, positions, l, l).todense()


def tridiag2(l,autocorr,rho1,rho2):
    diag_rows = numpy.array([rho2*numpy.ones(l),
                rho1*numpy.ones(l),
                autocorr*numpy.ones(l),
                rho1*numpy.ones(l),
                rho2*numpy.ones(l)])
    positions = [-2, -1, 0, 1,2]
    return sp.sparse.spdiags(diag_rows, positions, l, l).todense()

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

def roc_curve( dvals, labels, rocN=None, normalize=True ) :
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1}

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1 :
        rocN = int(rocN * numpy.sum(numpy.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives

    fpc = [ 0.0 ]  # fp coordinates
    tpc = [ 0.0 ]  # tp coordinates
    dv_prev = -numpy.inf # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    num_pos = labels.count( 1 )  # number of pos labels  numpy.sum(labels)
    num_neg = labels.count( 0 ) # number of neg labels  numpy.prod(labels.shape) - num_pos

    if num_pos == 0 or num_pos == len(labels) :
        raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = numpy.argsort( dvals )[ ::-1 ]

    for idx in indices:
        # increment associated TP/FP count
        if labels[ idx ] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN :
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[ idx ] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1] :
                tpc[-1] = TP
            else :
                fpc.append( FP  )
                tpc.append( TP  )
            dv_prev = dvals[ idx ]
            area += _trap_area( ( FP_prev, TP_prev ), ( FP, TP ) )
            FP_prev = FP
            TP_prev = TP

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )
    if normalize :
        fpc = [ float( x ) / FP for x in fpc ]
        if TP > 0:
            tpc = [ float( x ) / TP for x in tpc ]
        if area > 0:
            area /= ( TP * FP )

    return fpc, tpc, area

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

def normpdfMultivariate(x, mu, sigma):
    l = x.shape[0]
    u = (x-mu)
    #print 'det =' + str(numpy.linalg.det(sigma))
    y = (exp(-0.5*numpy.dot(u,numpy.dot(sigma,u)) ) * numpy.linalg.det(sigma)**(0.5))  / ((2*numpy.pi)**(l/2.) )
    return y

def compute_MSE(nrl , nrlref):
    e = nrl - nrlref
    e = reshape(e,(prod(e.shape),1))
    mse = mean(pow(e,2))
    return mse


def mean_HRF(m_H,Pr):
    #D = m_H.shape[0]
    sh = numpy.dot(m_H,Pr)/(sum(Pr) + eps)
    return sh

def hrf_porte(LEN):
    porte = numpy.ones(LEN)
    porte[0:LEN/10] = 0
    porte[LEN/2:] = -1
    porte[LEN-LEN/10:] = 0
    return porte


def hrf_triang(LEN):
    triang = numpy.zeros(LEN)
    for i in xrange(0,LEN/4):
        triang[i] = i
    for i in xrange(0,3*LEN/4):
        triang[LEN/4+i] = LEN/4 -i
    for i in xrange(0,LEN/4):
        triang[3*LEN/4+i] = i - LEN/4 + 1
    return triang

def mean_HRF_covar(Sigma_H,Pr,zerosDD):
    D = Sigma_H.shape[0]
    J = Sigma_H.shape[2]
    sh = zerosDD.copy()
    for j in xrange(0,J):
        sh += 0.5*(Pr[0,j]  * Sigma_H[:,:,j] / (eps + sum(Pr[0,:])) + Pr[0,j]  * Sigma_H[:,:,j] / (eps + sum(Pr[1,:])) )

    #sh = numpy.dot(m_H,Pr)/(sum(Pr) + eps)
    return sh

#def normpdfMultivariate(x, mu, sigma):
    #k = x.shape[0]
    #part1 = numpy.exp(-0.5*k*numpy.log(2*numpy.pi))
    #part2 = numpy.power(numpy.linalg.det(sigma),0.5)
    #dev = x-mu
    #part3 = numpy.exp(numpy.dot(numpy.dot(dev,cov),dev.transpose()))
    #y = part1*part2*part3
    #return y


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

def compute_mat_X(nbscans, lhrf, tr, tmax, onsets, durations=None):
    x = numpy.zeros((nbscans,lhrf),dtype=float)
    x_tmp = numpy.zeros(nbscans,dtype=float)
    x_tmp[xrange(0,tmax)] = restarize_events(onsets,numpy.zeros_like(onsets),tr,tmax)
    for i in xrange(0,lhrf):
        for j in xrange(0,len(x)):
            x[j,i] = x_tmp[j-i]
    return x

def maximum(a):
    maxx = a[0]
    maxx_ind = 0
    for i in xrange(len(a)):
        if a[i] > maxx :
            maxx = a[i]
            maxx_ind = i

    return maxx, maxx_ind

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

def expectation_AP2(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_MK,q_Z,mu_MK,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD):
    X_tilde = zerosJMD.copy()#numpy.zeros((Y.shape[1],M,D),dtype=float)
    J = Y.shape[1]
    for i in xrange(0,J):
        m = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Sigma_A[m,m2,i] = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H[:,i].transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H[:,i])
                Sigma_A[m,m2,i] += (numpy.dot(numpy.dot(numpy.dot(Sigma_H[:,:,i],X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
                m2 += 1
            X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
            m += 1
        tmp = numpy.dot(X_tilde[i,:,:],m_H[:,i])
        for k in xrange(0,K):
            Delta = numpy.diag( q_Z[:,k,i]/(sigma_MK[:,k] + eps) )
            tmp += numpy.dot(Delta,mu_MK[:,k])
            Sigma_A[:,:,i] += Delta
        tmp2 = inv(Sigma_A[:,:,i])
        Sigma_A[:,:,i] = tmp2
        m_A[i,:] = numpy.dot(Sigma_A[:,:,i],tmp)
    return Sigma_A, m_A

def expectation_AP(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_MK,q_Z,mu_MK,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,Pmask):
    X_tilde = zerosJMD.copy()#numpy.zeros((Y.shape[1],M,D),dtype=float)
    J = Y.shape[1]
    for i in xrange(0,J):
        m = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Sigma_A[m,m2,i] = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H[:,i].transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H[:,i])
                Sigma_A[m,m2,i] += (numpy.dot(numpy.dot(numpy.dot(Sigma_H[:,:,i],X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
                m2 += 1
            X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
            m += 1
        tmp = numpy.dot(X_tilde[i,:,:],m_H[:,i])
        for k in xrange(0,K):
            Delta = numpy.diag( q_Z[:,k,i]/(sigma_MK[:,k,Pmask[i]] + eps) )
            tmp += numpy.dot(Delta,mu_MK[:,k,Pmask[i]])
            Sigma_A[:,:,i] += Delta
        tmp2 = inv(Sigma_A[:,:,i])
        Sigma_A[:,:,i] = tmp2
        m_A[i,:] = numpy.dot(Sigma_A[:,:,i],tmp)
    return Sigma_A, m_A

#def expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone):
    #for j in xrange(0,J):
	#Sigma_bar_j_1 = q_Q[0,j]*HRFDictCovar[0]
	#Sum_Sigma_h_k = q_Q[0,j]*numpy.dot(HRFDictCovar[0],HRFDict[0])
	#for l in xrange(1,I):
	    #Sigma_bar_j_1 += q_Q[l,j]*HRFDictCovar[l]
	    #Sum_Sigma_h_k += q_Q[l,j]*numpy.dot(HRFDictCovar[l],HRFDict[l])
	##mu_bar_j = numpy.dot(Sigma_bar_j_1,Sum_Sigma_h_k)
	##mu_bar_j = numpy.dot(inv(Sigma_bar_j_1),Sum_Sigma_h_k)
	#m = 0
	#tmp =  zerosND.copy()
	#for k in X: # Loop over the M conditions
	    #tmp += m_A[j,m] * X[k]
	    #m += 1
	#Y_bar_tilde = numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[j],eps)),y_tilde[:,j])
	#Q_bar = numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[j],eps)),tmp)
	#m1 = 0
	#for k1 in X: # Loop over the M conditions
	    #m2 = 0
	    #for k2 in X: # Loop over the M conditions
		#Q_bar += Sigma_A[m1,m2,j] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[j],eps)),X[k2])
		#m2 += 1
	    #m1 += 1
	#Sigma_H[:,:,j] = inv(Q_bar + Sigma_bar_j_1)
	##tmp3 = Sum_Sigma_h_k#numpy.dot(Sigma_bar_j_1,mu_bar_j)
	##tmp2 = Y_bar_tilde#numpy.dot(inv(Q_bar),numpy.dot(Q_bar,Y_bar_tilde))
	#m_H[:,j] = numpy.dot(Sigma_H[:,:,j],Y_bar_tilde + Sum_Sigma_h_k)
    #return Sigma_H, m_H

def expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone):
    for j in xrange(0,J):
        Sigma_bar_j_1 = q_Q[0,j]*HRFDictCovar[0]
        Sum_Sigma_h_k = q_Q[0,j]*numpy.dot(HRFDictCovar[0],HRFDict[0])
        for l in xrange(1,I):
            Sigma_bar_j_1 += q_Q[l,j]*HRFDictCovar[l]
            Sum_Sigma_h_k += q_Q[l,j]*numpy.dot(HRFDictCovar[l],HRFDict[l])
        m = 0
        tmp =  zerosND.copy()
        for k in X: # Loop over the M conditions
            tmp += m_A[j,m] * X[k]
            m += 1
        Y_bar_tilde = numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[j],eps)),y_tilde[:,j])
        Q_bar = numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[j],eps)),tmp)
        m1 = 0
        for k1 in X: # Loop over the M conditions
            m2 = 0
            for k2 in X: # Loop over the M conditions
                Q_bar += Sigma_A[m1,m2,j] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[j],eps)),X[k2])
                m2 += 1
            m1 += 1
        Sigma_H[:,:,j] = inv(Q_bar + Sigma_bar_j_1)
        m_H[:,j] = numpy.dot(Sigma_H[:,:,j],Y_bar_tilde + Sum_Sigma_h_k)
    return Sigma_H, m_H

def maximization_hk_Sigmak(R,J,q_Q,Sigma_H,m_H,HRFDict,HRFDictCovar,I,zerosDD,zerosD):
    for i in xrange(0,I):
        S = sum( q_Q[i,:] )
        tmp = zerosDD.copy()
        tmpD = zerosD.copy()
        for j in xrange(0,J):
            tp = m_H[:,j]- HRFDict[i]
            tmp += q_Q[i,j]*(Sigma_H[:,:,j] + mult(tp,tp))
            tmpD +=  q_Q[i,j]*m_H[:,j]
        tmpf = numpy.dot(R,tmp / S)
        #print tmpf
        print tmpf.min(),tmpf.max()
        HRFDictCovar[i] = numpy.dot(inv(tmpf),R)
        #HRFDictCovar[i] = inv(tmp / S)
        HRFDict[i] = tmpD / S
    return HRFDictCovar,HRFDict

def maximization_hk(J,q_Q,m_H,HRFDict,I,zerosD):
    for i in xrange(0,I):
        #S = sum( q_Q[i,:] )
        S = 200.
        tmp =  q_Q[i,0] *m_H[:,0]/S
        for j in xrange(0,J):
            #print tmp
            #raw_input('')
            tmp +=  q_Q[i,j] *m_H[:,j] /S
            #figure(55)
            #plot(m_H[:,j])
            #draw()


        HRFDict[i] = tmp
        #HRFDict[i] = m_H[:,47]
    return HRFDict

def classify(Pr):
    nbClass = Pr.shape[0]
    J = Pr.shape[1]
    #CL = (nbClass-1)*numpy.ones((J))
    CL = numpy.zeros((J))
    #print 'J:', J
    #print 'Pr.shape:', Pr.shape
    #print 'Pr:', Pr
    #print 'np.where(Pr>0):', np.where(Pr>0)
    #print 'np.where(Pr=="NaN"):', np.where(Pr=='NaN')
    #print 'CL.shape:', CL.shape
    for j in xrange(0,J):
        #print 'j:', j
        tmp = Pr[:,j]
        #print 'Pr[:,j]', Pr[:,j]
        #print max(tmp)
        ind = find(tmp == max(tmp))
        #print 'ind:', ind
        #print 'CL:', CL
        #print 'ind.shape:', ind.shape
        CL[j] = ind[0]
    return CL

def expectation_Q(m_H,Sigma_H,sigma_M,beta,Q_tilde,HRFDict,HRFDictCovar,q_Q,graph,J,I,zerosI):
    energy = zerosI.copy()
    alpha = zerosI.copy()
    Gauss = zerosI.copy()
    for k in xrange(0,1):
        for j in xrange(0,J):
            tmp = sum(Q_tilde[:,graph[j]],1)
            for i in xrange(0,I):
                alpha[i] = -0.5*numpy.dot(Sigma_H[:,:,j],HRFDictCovar[i]).trace()
            Malpha = alpha.mean()
            alpha /= Malpha
            maxGauss = 1
            for i in xrange(0,I):
                u = m_H[:,j]-HRFDict[i]
                Gauss[i] = exp(-0.5*numpy.dot(u,numpy.dot(HRFDictCovar[i],u))) * (numpy.linalg.det(HRFDictCovar[i])**0.5)
                if ( Gauss[i]>maxGauss ):
                    maxGauss = Gauss[i]
                extern_field = alpha[i]# + gauss
                local_energy = beta * tmp[i]
                energy[i] = extern_field + local_energy
            Emax = energy.mean()
            Gauss /= maxGauss
            Probas = numpy.exp(energy - Emax) * Gauss
            Sum = sum(Probas)
            Q_tilde[:,j] = Probas/Sum

    for j in xrange(0,J):
        tmp = sum(Q_tilde[:,graph[j]],1)
        for i in xrange(0,I):
            alpha[i] = -0.5*numpy.dot(Sigma_H[:,:,j],HRFDictCovar[i]).trace()
        Malpha = alpha.mean()
        alpha /= Malpha
        maxGauss = 1
        for i in xrange(0,I):
            local_energy = beta * tmp[i]
            u = m_H[:,j]-HRFDict[i]
            extern_field = alpha[i]
            Gauss[i] = exp(-0.5*numpy.dot(u,numpy.dot(HRFDictCovar[i],u))) * (numpy.linalg.det(HRFDictCovar[i])**0.5)
            if ( Gauss[i]>maxGauss):
                maxGauss = Gauss[i]
            energy[i] = extern_field + local_energy
        Emax = energy.mean()
        Gauss /= maxGauss
        Probas = numpy.exp(energy - Emax) * Gauss
        Sum = sum(Probas)
        q_Q[:,j] = Probas/Sum
    return q_Q, Q_tilde

def thresholding(x,thresh = 0.5):
    ind = find(x>thresh).tolist()
    x *= 0
    for i in ind:
        x[i] = 1
    return x

def labelling(x,label,thresh = 0.5):
    ind = find(x>thresh).tolist()
    x *= 0
    for i in ind:
        x[i] = label
    return x

def expectation_H_Wavelet(Dw,Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD):
    Y_bar_tilde = zerosD.copy()
    Q_bar = 0 * R
    Q_bar2 = 0 * R
    for i in xrange(0,J):
        m = 0
        tmp =  zerosND.copy()
        for k in X:
            tmp += m_A[i,m] * X[k]
            m += 1
        Y_bar_tilde += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),y_tilde[:,i])
        Q_bar += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),tmp)
        m1 = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Q_bar += Sigma_A[m1,m2,i] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[i],eps)),X[k2])
                m2 +=1
            m1 +=1
    tmp = scale * R
    Q_bar = Q_bar + tmp
    Sigma_H = inv(Q_bar)
    cA, cD = dwt(Y_bar_tilde,'db8','per')
    Y_bar_tilde[0:Dw] = cA
    Y_bar_tilde[Dw:] = cD
    m_H = numpy.dot(Sigma_H,Y_bar_tilde)
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

def maximization_mu_sigma_P2(Mu,Sigma,q_Z,m_A,K,M,Sigma_A):
    for m in xrange(0,M):
        for k in xrange(0,K):
            S = sum( q_Z[m,k,:] ) + eps
            Sigma[m,k] = eps + sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + \
                                    Sigma_A[m,m,:] ) ) / S
            if k != 0 : # mu_0 = 0 a priori
                Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
            else:
                Mu[m,k] = 0.
    return Mu , Sigma


def maximization_mu_sigma_P(Mu,Sigma,q_Z,m_A,K,M,Sigma_A,Pmask,I):

    for i in xrange(0,I):
        #ind = find(Pmask == i)
        ind = find(Pmask < 100)
        for m in xrange(0,M):
            for k in xrange(0,K):
                S = sum( q_Z[m,k,ind] ) + eps
                Sigma[m,k,i] = eps + sum( q_Z[m,k,ind] * ( pow(m_A[ind,m] - Mu[m,k,i] ,2) + \
                                                        Sigma_A[m,m,ind] ) ) / S
                if k != 0 : # mu_0 = 0 a priori
                    Mu[m,k,i] = eps + sum( q_Z[m,k,ind] * m_A[ind,m] ) / S
                else:
                    Mu[m,k,i] = 0.

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

def maximization_LP(Y,m_A,X,m_H,L,P,zerosP):
    J = Y.shape[1]
    for i in xrange(0,J):
        S = zerosP.copy()
        m = 0
        for k in X:
            S += m_A[i,m]*numpy.dot(X[k],m_H[:,i])
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

def maximization_sigma_noiseP(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM):
    N = PL.shape[0]
    J = Y.shape[1]
    Htilde = zerosMM.copy() #numpy.zeros((M,M),dtype=float)
    for i in xrange(0,J):
        S = numpy.zeros((N),dtype=float)
        m = 0
        for k in X:
            m2 = 0
            for k2 in X:
                Htilde[m,m2] =  numpy.dot(numpy.dot(numpy.dot(m_H[:,i].transpose(),X[k].transpose()),X[k2]),m_H[:,i])
                Htilde[m,m2] += (numpy.dot(numpy.dot(Sigma_H[:,:,i],X[k].transpose()),X[k2])).trace()
                m2 += 1
            S += m_A[i,m]*numpy.dot(X[k],m_H[:,i])
            m += 1
        sigma_epsilone[i] = numpy.dot( -2*S, Y[:,i] - PL[:,i] )
        sigma_epsilone[i] += (numpy.dot(Sigma_A[:,:,i],Htilde)).trace()
        sigma_epsilone[i] += numpy.dot( numpy.dot(m_A[i,:].transpose(), Htilde),m_A[i,:] )
        sigma_epsilone[i] += numpy.dot((Y[:,i] - PL[:,i]).transpose(), Y[:,i] - PL[:,i] )
        sigma_epsilone[i] /= N
    return sigma_epsilone


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

def maximization_sigmaH_P(D,J,I,Sigma_H,R,m_H,HRFDict,HRFDictCovar,q_Q,gamma_h):
    for i in xrange(0,I):
        S = sum( q_Q[i,:] )
        #sigmaH = 0
        alpha = 0
        for j in xrange(0,J):
            tp = m_H[:,j]- HRFDict[i]
            alpha += q_Q[i,j]*(numpy.dot(mult(tp,tp) + Sigma_H[:,:,j] , R )).trace()
        sigmaH = (D*S + sqrt(D*D*S*S + 8*gamma_h*alpha)) / (4*gamma_h)
            #sigmaH += q_Q[i,j]*(numpy.dot(mult(tp,tp) + Sigma_H[:,:,j] , R )).trace()
        #sigmaH /= (S * D)
        #sigmaH = 1
        print sigmaH
        HRFDictCovar[i] = R / sigmaH
    return HRFDictCovar


  #def mean_HRF(m_H,Pr):
    ##D = m_H.shape[0]
    #sh = numpy.dot(m_H,Pr)/(sum(Pr) + eps)
#mean_HRF(m_H,Pr)

def maximization_h_k_prior2(m_H,Pr,Sigma_k,R):
    J = Pr.shape[0]
    D = m_H.shape[0]
    S = sum(Pr)
    Mat = inv( numpy.dot( R,inv(Sigma_k) )/S + numpy.identity(D) )
    tmp = mean_HRF(m_H,Pr)
    h_k = numpy.dot( Mat, tmp)

    return h_k

def maximization_h_k_prior(m_H,Pr,Sigma_k,R):
    J = Pr.shape[0]
    S = sum(Pr)
    D = m_H.shape[0]
    tmp = 0*m_H[:,0]
    for j in xrange(0,J):
        tmp += m_H[:,j] * Pr[j]
    #h_k = numpy.dot( inv( numpy.dot( R,inv(Sigma_k) ) + S*numpy.identity(D) ) ,numpy.dot(m_H,Pr))
    h_k = numpy.dot( inv( numpy.dot( R,inv(Sigma_k) ) + S*numpy.identity(D) ) ,tmp)
    return h_k

def maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h):
    alpha = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
    #sigmaH = (D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)
    sigmaH = (-D + sqrt(D*D + 8*gamma_h*alpha)) / (4*gamma_h)

    return sigmaH

def maximization_alphaRVM(k_RVM,lam_RVM,Mw,Vw,M,alpha_RVM):
    
    for m in xrange(0,M):
        alpha_RVM[m] = ( k_RVM - 0.5 ) / ( lam_RVM + 0.5*(Mw[m]**2 + Vw[m,m]) + eps)

    return alpha_RVM
        

def expectation_A_ParsiMod(Sigma_H,m_H,m_A,X,Gamma,sigma_MK,q_Z,mu_MK,J,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,p_Wtilde,M):
    X_tilde = zerosJMD.copy()#numpy.zeros((Y.shape[1],M,D),dtype=float)

    for i in xrange(0,J):
        m = 0
        for k1 in X:
            m2 = 0
            for k2 in X:
                Sigma_A[m,m2,i] = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H)
                Sigma_A[m,m2,i] += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
                Sigma_A[m,m2,i] = Sigma_A[m,m2,i] * p_Wtilde[m,1] * p_Wtilde[m2,1]
                m2 += 1
            X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
            X_tilde[i,m,:] *= p_Wtilde[m,1]
            m += 1
        tmp = numpy.dot(X_tilde[i,:,:],m_H)

        # Computing of Delta Matrix in case of 2 classes, with W matrix in activation and inactivation case
        # Have not the same forme so we can't do a loop to build Delta (as in complet model).
        Delta_0 = numpy.diag( (1 - q_Z[:,1,i] * p_Wtilde[:,1])/sigma_MK[:,0] )
        Sigma_A[:,:,i] += Delta_0
        tmp += numpy.dot(Delta_0,mu_MK[:,0])

        Delta_1 = numpy.diag( (q_Z[:,1,i] * p_Wtilde[:,1])/sigma_MK[:,1] )
        Sigma_A[:,:,i] += Delta_1
        tmp += numpy.dot(Delta_1,mu_MK[:,1])

        tmp2 = inv(Sigma_A[:,:,i])
        Sigma_A[:,:,i] = tmp2

        m_A[i,:] = numpy.dot(Sigma_A[:,:,i],tmp)
    return Sigma_A, m_A

def expectation_H_ParsiMod(Sigma_A,m_A,X,Gamma,R,sigmaH,J,y_tilde,zerosND,sigma_epsilone,scale,zerosD,p_Wtilde):
    Y_bar_tilde = zerosD.copy()#numpy.zeros((D),dtype=float)
    Q_bar = scale * R/sigmaH
    Q_bar2 = scale * R/sigmaH
    for i in xrange(0,J):
        m = 0
        tmp =  zerosND.copy() #numpy.zeros((N,D),dtype=float)
        for k in X: # Loop over the M conditions
            tmp += p_Wtilde[m,1] * m_A[i,m] * X[k]
            m += 1
        Y_bar_tilde += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),y_tilde[:,i])
        Q_bar += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),tmp)
        Q_bar2[:,:] = Q_bar[:,:]
        m1 = 0
        for k1 in X: # Loop over the M conditions
            m2 = 0
            for k2 in X: # Loop over the M conditions
                Q_bar += p_Wtilde[m1,1] * p_Wtilde[m2,1] * Sigma_A[m1,m2,i] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[i],eps)),X[k2])
                m2 +=1
            m1 +=1
    Sigma_H = inv(Q_bar)
    m_H = numpy.dot(Sigma_H,Y_bar_tilde)
    m_H[0] = 0
    m_H[-1] = 0
    return Sigma_H, m_H

def expectation_Z_ParsiMod(tau1,tau2,Sigma_A,m_A,J,M,sigma_M,mu_M,V,K,Beta,graph,p_Wtilde,zerosV,zerosK,q_Z):
  #P_mv = zerosV.copy()
  #part_3 = zerosV.copy()
  #tmp = zerosK.copy()
  #part_4 = zerosK.copy()

  #print 'Expectation Z Python function ...'

  for m in xrange(0,M):
    P_mv = zerosV.copy()
    part_3 = zerosV.copy()
    tmp = zerosK.copy()
    part_4 = zerosK.copy()
    for i in xrange(0,J):
      for v in xrange(0,V):
        part_2 = 1.
        part_1 = ((-1)**(v+1))*numpy.exp(((v+1)*tau1*tau2)/J)/(v+1)
        for j in xrange(0,J):
            if j != i:
                tmp = numpy.exp(((v+1)*tau1*tau2)/J)*(q_Z[m,0,j] + numpy.exp(-(v+1)*tau1)*q_Z[m,1,j])
                part_2 *= tmp
        P_mv[v] = part_1 * part_2
        part_3[v] = numpy.exp(-(v+1)*tau1) * P_mv[v]

        for k in xrange(0,K):
            tmp[k] = sum(q_Z[m,k,graph[i]],0)
            part_4[k] = normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k]))

      #numpy.seterr (all='ignore')
      #q_Z[m,0,i] = min( numpy.exp(sum(P_mv) + Beta[m]*tmp[0]), numpy.exp(700) )

      #part_5 = (part_4[1] / max(part_4[0],eps) )**p_Wtilde[m,1]
      #part_5 = numpy.exp( p_Wtilde[m,1] * ( numpy.log(part_4[1]+eps) - numpy.log(part_4[0]+eps) ) )
      part_5 = p_Wtilde[m,1] * ( numpy.log(part_4[1]+eps) - numpy.log(part_4[0]+eps) )

      #VAR = 1.
      #for v in xrange(0,V):
        #VAR *= numpy.exp(part_3[v])
      #part_6 = VAR * numpy.exp(Beta[m]*tmp[1])
      #part_7 = numpy.exp(0.5*Sigma_A[m,m,i]*p_Wtilde[m,1]*( 1./sigma_M[m,0] - 1./sigma_M[m,1] ) + tau1*(p_Wtilde[m,1]-1.))*part_6

      part_6 = sum(part_3) + Beta[m]*tmp[1]
      #numpy.seterr (all='ignore')
      #part_7 = min( numpy.exp(0.5*Sigma_A[m,m,i]*p_Wtilde[m,1]*( 1./sigma_M[m,0] - 1./sigma_M[m,1] ) + tau1*(p_Wtilde[m,1]-1.) + part_6), numpy.exp(700) )
      #q_Z[m,1,i] = part_5 * part_7

      part_7 = 0.5*Sigma_A[m,m,i]*p_Wtilde[m,1]*( 1./sigma_M[m,0] - 1./sigma_M[m,1] ) + tau1*(p_Wtilde[m,1]-1.) + part_6

      #numpy.seterr (all='ignore')
      #q_Z[m,1,i] = 1 + (1./max(part_5,eps))* min( numpy.exp(sum(P_mv) + Beta[m]*tmp[0] - part_7), numpy.exp(700) )
      if (sum(P_mv) + Beta[m]*tmp[0] - part_7 - part_5) < 700.:
        q_Z[m,1,i] = 1 + numpy.exp(sum(P_mv) + Beta[m]*tmp[0] - part_7 - part_5)
      else:
        q_Z[m,1,i] = 1 + numpy.exp(700.)

      q_Z[m,1,i] = 1./q_Z[m,1,i]
      q_Z[m,0,i] = 1. - q_Z[m,1,i]

      #SZ = sum(q_Z[m,:,i])
      #q_Z[m,:,i] /= SZ

  return q_Z

def MC_step_log(tau1,tau2,m,i,q_Zi,q_Z,M,J,S,K):

  #print 'Begining MC Step ...'

  sum_log_term = 0.

  for s in xrange(0,S):
    labels = zeros((M,J), dtype=int)
    labels_samples = numpy.random.rand(M,J)
    for j in xrange(0,J):
      if j != i:
        lab = K - 1
        for k in xrange(0,K-1):
            if labels_samples[m,j] <= q_Z[m,k,j]:
                lab = 0
        labels[m,j] = lab
    SUM = sum(labels[m,:])

    log_term = numpy.log( 1 + numpy.exp( - tau1 * (SUM + q_Zi - tau2) ) )
    sum_log_term += log_term

  E_log_term = sum_log_term/S

  #print 'Finished MC Step ...'

  return E_log_term

def expectation_Z_ParsiMod2(tau1,tau2,Sigma_A,m_A,J,M,sigma_M,mu_M,S,K,Beta,graph,p_Wtilde,zerosK,q_Z):

  print 'Begining Expectation Z (2) Step ...'

  for m in xrange(0,M):
    tmp = zerosK.copy()
    part_4 = zerosK.copy()
    E_log_term = zerosK.copy()
    for i in xrange(0,J):
      for k in xrange(0,K):
        tmp[k] = sum(q_Z[m,k,graph[i]],0)
        part_4[k] = normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k]))
        E_log_term[k] = MC_step_log(tau1,tau2,m,i,k,q_Z,M,J,S,K)

      part_5 = p_Wtilde[m,1] * ( numpy.log(part_4[1]+eps) - numpy.log(part_4[0]+eps) )


      part_6 = Beta[m]*tmp[1]

      part_7 = 0.5*Sigma_A[m,m,i]*p_Wtilde[m,1]*( 1./sigma_M[m,0] - 1./sigma_M[m,1] ) + tau1*(p_Wtilde[m,1]-1.) + part_6

      #numpy.seterr (all='ignore')
      if (Beta[m]*tmp[0] - E_log_term[0] - part_7 - part_5 + E_log_term[1]) < 700. :
        q_Z[m,1,i] = 1 + numpy.exp(Beta[m]*tmp[0] - E_log_term[0] - part_7 - part_5 + E_log_term[1])
      else:
        q_Z[m,1,i] = 1 + numpy.exp(700.)

      q_Z[m,1,i] = 1./q_Z[m,1,i]
      q_Z[m,0,i] = 1. - q_Z[m,1,i]

  print 'Finished Expectation Z (2) Step ...'

  return q_Z

def expectation_W_ParsiMod(tau1,tau2,Sigma_A,m_A,X,Gamma,J,M,y_tilde,sigma_epsilone,sigma_M,mu_M,p_Wtilde,q_Z,zerosK,K,m_H,Sigma_H,zerosJMD):
  part_1 = zerosK.copy()
  X_tilde = zerosJMD.copy()

  #print 'Expectation W Python function ...'

  for m,k1 in enumerate(X):

    part_2 = 0.
    part_3 = 0.
    part_5 = 0.
    part_8 = 0.

    part = - tau1 * (sum(q_Z[m,1,:]) - tau2)

    for i in xrange(0,J):
      for k in xrange(0,K):
        part_1[k] = normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k]))

      #part_2 *= ( part_1[1] / max(part_1[0],eps) )**q_Z[m,1,i]
      #part_2 *= numpy.exp( q_Z[m,1,i] * ( numpy.log (part_1[1]+eps) - numpy.log(part_1[0]+eps) ) )
      part_2 += q_Z[m,1,i] * ( numpy.log(part_1[1]+eps) - numpy.log(part_1[0]+eps) )

      part_3 += 0.5*q_Z[m,1,i]*Sigma_A[m,m,i]*( 1./sigma_M[m,0] - 1./sigma_M[m,1] )

      part_4 = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k1]),m_H)

      part_4 += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k1])).trace()

      part_4 *= (m_A[i,m]**2 +  Sigma_A[m,m,i])
      part_5 -= 0.5*part_4

      part_7 = 0.
      for m2,k2 in enumerate(X):

        if m2 != m:
            part_6 = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H)
            part_6 += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
            part_6 *= (m_A[i,m]*m_A[i,m2] +  Sigma_A[m,m2,i])
            part_6 *= p_Wtilde[m2,1]
        part_7 += part_6

      part_8 += part_7
      X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
    tmp = numpy.dot(X_tilde[:,m,:],m_H)
    part_9 = numpy.dot(m_A[:,m],tmp)

    #p_Wtilde[m,1] = 1 + ( 1./ max(part_2,eps) ) * min(numpy.exp( part - (part_3 + part_5 - part_8 + part_9) ), numpy.exp(700))
    if ( part - (part_2 + part_3 + part_5 - part_8 + part_9) ) < 700.:
      p_Wtilde[m,1] = 1 + numpy.exp( part - (part_2 + part_3 + part_5 - part_8 + part_9) )
    else:
      p_Wtilde[m,1] = 1 + numpy.exp(700.)

    p_Wtilde[m,1] = 1. / p_Wtilde[m,1]
    p_Wtilde[m,0] = 1. - p_Wtilde[m,1]

  #for m in xrange(0,M):
    #p_Wtilde[m,0] = numpy.exp(- tau1 * (sum(q_Z[m,1,:]) - tau2))

  #for m,k1 in enumerate(X):
    #part_2 = 1.
    #part_3 = 1.
    #part_5 = 1.
    #part_7 = 1.
    #part_8 = 1.
    #part = 0.

    #for i in xrange(0,J):
      #for k in xrange(0,K):
        #part_1[k] = normpdf(m_A[i,m], mu_M[m,k], numpy.sqrt(sigma_M[m,k]))
      #part_2 *= (part_1[1] / part_1[0])**q_Z[m,1,i]
      #part_3 *= numpy.exp(0.5*q_Z[m,1,i]*Sigma_A[m,m,i]*(1./sigma_M[m,0] - 1./sigma_M[m,1]))
      #part_4 = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k1]),m_H)
      #part_4 += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k1])).trace()
      #part_4 *= (m_A[i,m]**2 +  Sigma_A[m,m,i])
      #part_5 *= numpy.exp(-0.5*part_4)
      #for m2,k2 in enumerate(X):
        #if m2 != m:
        #part_6 = numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2]),m_H)
        #part_6 += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k1].transpose()),Gamma/max(sigma_epsilone[i],eps)),X[k2])).trace()
        #part_6 *= (m_A[i,m]*m_A[i,m2] +  Sigma_A[m,m2,i])
        #part_6 *= p_Wtilde[m2,1]
        ##print 'm2',m2
        ##print 'p_Wtilde[m2,1]',p_Wtilde[m2,1]
        #part += part_6
        #part_7 *= numpy.exp(part_6)
      #part_8 *= part_7
      #X_tilde[i,m,:] = numpy.dot(numpy.dot(Gamma/max(sigma_epsilone[i],eps),y_tilde[:,i]).transpose(),X[k1])
    #tmp = numpy.dot(X_tilde[:,m,:],m_H)
    #part_9 = numpy.exp(numpy.dot(m_A[:,m],tmp))
    #p_Wtilde[m,1] = part_2*part_3*part_5*part_8*part_9

    #SW = sum(p_Wtilde[m,:])
    #p_Wtilde[m,:] /= SW

  return p_Wtilde

def maximization_mu_sigma_ParsiMod(Mu,Sigma,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J):
    for m in xrange(0,M):
        for k in xrange(0,K):
            S = sum( q_Z[m,k,:] )
            if S == 0.: 
                S = eps
            if k != 0 :
                Sigma[m,k] = sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + \
                                                       Sigma_A[m,m,:] ) ) / S
            else:
                Sigma[m,k] = sum( ( 1 - p_Wtilde[m,1] * q_Z[m,1,:] ) * ( pow(m_A[:,m] ,2) + \
                                                       Sigma_A[m,m,:] ) ) / (sum ( 1 - p_Wtilde[m,1] * q_Z[m,1,:] ) + eps)
            if Sigma[m,k] < eps:
                Sigma[m,k] = eps

            if k != 0 : # mu_0 = 0 a priori
                Mu[m,k] = sum( q_Z[m,k,:] * m_A[:,m] ) / S
            else:
                Mu[m,k] = 0.

    return Mu , Sigma

def gradient_mu1(q_Z_Mk,p_Wtilde_Mk,Mu_Mk,Sigma_Mk,m_A_M,tau1,tau2,J):

    A = 0.0
    B = 0.0
    for j in xrange(0,J):
        A += q_Z_Mk[j]
        B += q_Z_Mk[j] * m_A_M[j]

    #print 'sum Q =',A
    #print 'sum Q*Ma =',B

    A = A * p_Wtilde_Mk / Sigma_Mk
    B = ( B * p_Wtilde_Mk / Sigma_Mk ) + ( p_Wtilde_Mk * tau1 )

    Droite = A * Mu_Mk - B
    if (-tau1*(Mu_Mk-tau2)) > 700.:
        Sigmoid = 1.0 / (1.0 + numpy.exp(700.) )
    else:
        Sigmoid = 1.0 / (1.0 + numpy.exp(-tau1*(Mu_Mk-tau2)) )

    Gr_mu1 = Droite + ( tau1 * Sigmoid )

    #print 'Mu_1 =',Mu_Mk
    #print 'Sigma_1 =',Sigma_Mk
    #print 'A =',A
    #print 'B =',B
    #print 'Droite=',Droite
    #print 'Sigmoid=',Sigmoid
    ##print '-tau1*(Mu_Mk-tau2) =',-tau1*(Mu_Mk-tau2)
    #print ' Gr_mu1 =',Gr_mu1

    return Gr_mu1
    
def Function_Dichotomie_square(A,B,tau1,tau2,x,estimateW):
    
    D = A * x + B
    if estimateW:
        if (tau1*(x**2 - tau2)) >= 0.0:
            S = 1.0 / (1.0 + numpy.exp(-tau1*(x**2 - tau2)) )
        if (tau1*(x**2 - tau2)) < 0.0:
            S = numpy.exp(tau1*(x**2 - tau2)) / (1.0 + numpy.exp(tau1*(x**2 - tau2)) )
    if not estimateW:
        S = 1.
        
    F = D - (2. * tau1 * x * S )
    
    return F
        
def Dichotomie_square(q_Z_Mk,p_Wtilde_Mk,Mu_Mk,Sigma_Mk,m_A_M,tau1,tau2,J,m,Iter,estimateW):
    
    '''
    Method to find zeros of complicated functions
    
    '''
    
    A1 = 0.0
    B1 = 0.0
    for j in xrange(0,J):
        A1 += q_Z_Mk[j]
        B1 += q_Z_Mk[j] * m_A_M[j]

    A = (2. * tau1 * p_Wtilde_Mk) - (A1 * p_Wtilde_Mk / Sigma_Mk)
    B = B1 * p_Wtilde_Mk / Sigma_Mk

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.1
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_square(A,B,tau1,tau2,xx1,estimateW)
    vxx2 = - 0.1
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_square(A,B,tau1,tau2,xx2,estimateW)
    while Fxx1 * Fxx2 > 0.:
        vxx1 *= 2.0
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_square(A,B,tau1,tau2,xx1,estimateW) 
        vxx2 *= 2.0
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_square(A,B,tau1,tau2,xx2,estimateW)
      
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_square(A,B,tau1,tau2,x3,estimateW)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    #if ((x3 < -0.005) or (x3 > 0.005)):
        ##### Plotting F To verify is computed zero is right ####
        #Mu_test = numpy.arange(-20,20.,0.1)
        #F_test = numpy.zeros(size(Mu_test), dtype=float)
        #D = numpy.zeros(size(Mu_test), dtype=float)
        #S = numpy.zeros(size(Mu_test), dtype=float)
        #for i in xrange(size(Mu_test)):
            #D[i] = A * Mu_test[i] + B
            #if (- tau1*(Mu_test[i]**2 - tau2)) > 700.:
                #S[i] = 1.0 / (1.0 + numpy.exp(700.) )
            #else:
                #S[i] = tau1*Mu_test[i]*( 1.0 / (1.0 + numpy.exp(-tau1*(Mu_test[i]**2 - tau2)) ))
            ##F_test[i] = Function_Dichotomie_square(A,B,tau1,tau2,Mu_test[i],estimateW)
            
        #print 'n =',n,',    x3 =',x3    
        #figure(Iter)
        #plot(Mu_test,D,'r')
        #plot(Mu_test,S,'b')
        #show()
    
    #print 'm =',m,',    n =',n,',   dist =',dist,',     mu1 =',x3
    
    return x3

def Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,x,Sigma,p_Wtilde,m):
    
    dKL = 0.5 * (x**2) * (1./Sigma[m,1] + 1./Sigma[m,0]) + ( (Sigma[m,1] - Sigma[m,0])**2 )/( 2. * Sigma[m,1] * Sigma[m,0] )
    
    if (tau1*(dKL - tau2)) >= 0.0:
        S = 1.0 / (1.0 + numpy.exp(-tau1*(dKL - tau2)) )
    if (tau1*(dKL - tau2)) < 0.0:
        S = numpy.exp(tau1*(dKL - tau2)) / (1.0 + numpy.exp(tau1*(dKL - tau2)) )
    
    F = B1 - x * A + (p_Wtilde[m,1] - S) * tau1 * x * ( 1. + Sigma[m,1]/Sigma[m,0] )

    return F

def Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,x,Sigma,Mu,p_Wtilde,m):
    
    dKL = 0.5 * (Mu[m,1]**2) * (1./x + 1./Sigma[m,0]) + ( (x - Sigma[m,0])**2 )/( 2. * x * Sigma[m,0] )
    
    if (tau1*(dKL - tau2)) >= 0.0:
        S = 1.0 / (1.0 + numpy.exp(-tau1*(dKL - tau2)) )
    if (tau1*(dKL - tau2)) < 0.0:
        S = numpy.exp(tau1*(dKL - tau2)) / (1.0 + numpy.exp(tau1*(dKL - tau2)) )
       
    F = B2 - x * A + (p_Wtilde[m,1] - S) * tau1 * ( -(Mu[m,1])**2 + (x**3/Sigma[m,0]) - Sigma[m,0])
    
    return F
    
def Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,x,Sigma,Mu,p_Wtilde,m):
    
    dKL = 0.5 * (Mu[m,1]**2) * (1./Sigma[m,1] + 1./x) + ( (Sigma[m,1] - x)**2 )/( 2. * Sigma[m,1] * x )
    
    if (tau1*(dKL - tau2)) >= 0.0:
        S = 1.0 / (1.0 + numpy.exp(-tau1*(dKL - tau2)) )
    if (tau1*(dKL - tau2)) < 0.0:
        S = numpy.exp(tau1*(dKL - tau2)) / (1.0 + numpy.exp(tau1*(dKL - tau2)) )
        
    F = B3 - x * A + (p_Wtilde[m,1] - S) * tau1 * ( -(Mu[m,1])**2 + (x**3/Sigma[m,1]) - Sigma[m,1])
    
    return F    

def dichotomie_mu1_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,J):
    
    At = 0.0
    Bt = 0.0
    for j in xrange(0,J):
        At += q_Z[m,1,j]
        Bt += q_Z[m,1,j] * m_A[j,m]

    A = At * p_Wtilde[m,1]
    B1 = Bt * p_Wtilde[m,1]

    ##  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 1.0
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,xx1,Sigma,p_Wtilde,m)
    vxx2 = - 1.0
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,xx2,Sigma,p_Wtilde,m)
    while Fxx1 * Fxx2 > 0.:
        vxx1 *= 2.0
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,xx1,Sigma,p_Wtilde,m)
        vxx2 *= 2.0
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,xx2,Sigma,p_Wtilde,m) 
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,x3,Sigma,p_Wtilde,m)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        #x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')

    ##### Plotting F To verify is computed zero is right ####
    #Mu_test = numpy.arange(-20,20.,0.1)
    #F_test = numpy.zeros(size(Mu_test), dtype=float)
    #for i in xrange(size(Mu_test)):
        #F_test[i] = Function_Dichotomie_mu1_Parsi4(A,B1,tau1,tau2,Mu_test[i],Sigma,p_Wtilde,m)   
    #figure(13)
    #plot(Mu_test,F_test)
    #savefig('./M1_test.png')

    return x3
    
def dichotomie_v1_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,Sigma_A,J):
    
    At = 0.0
    Bt = 0.0
    for j in xrange(0,J):
        At += q_Z[m,1,j]
        Bt += q_Z[m,1,j] * ( (m_A[j,m] - Mu[m,1])**2 + Sigma_A[m,m,j] )

    A = At * p_Wtilde[m,1]
    B2 = Bt * p_Wtilde[m,1]

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x)     
    vxx1 = 0.0001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,xx1,Sigma,Mu,p_Wtilde,m)
    vxx2 = 0.1
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,xx2,Sigma,Mu,p_Wtilde,m)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.0001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,xx1,Sigma,Mu,p_Wtilde,m)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,xx2,Sigma,Mu,p_Wtilde,m)
        
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,x3,Sigma,Mu,p_Wtilde,m)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE VARIANCE :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #V_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(V_test), dtype=float)
    #for i in xrange(size(V_test)):
        #F_test[i] = Function_Dichotomie_v1_Parsi4(A,B2,tau1,tau2,V_test[i],Sigma,Mu,p_Wtilde,m)
    #figure(14)
    #plot(V_test,F_test)
    #savefig('./V1_test.png')
    
    return x3
   
def dichotomie_v0_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,Sigma_A,J):

    At = 0.0
    Bt = 0.0
    for j in xrange(0,J):
        At += q_Z[m,1,j]
        Bt += ( 1. - p_Wtilde[m,1] * q_Z[m,1,j] ) * ( (m_A[j,m])**2 + Sigma_A[m,m,j] )

    A = J - At * p_Wtilde[m,1]
    B3 = Bt

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.0001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,xx1,Sigma,Mu,p_Wtilde,m)
    vxx2 = 0.1
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,xx2,Sigma,Mu,p_Wtilde,m)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.0001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,xx1,Sigma,Mu,p_Wtilde,m)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,xx2,Sigma,Mu,p_Wtilde,m)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,x3,Sigma,Mu,p_Wtilde,m)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE VARIANCE :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #V_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(V_test), dtype=float)
    #for i in xrange(size(V_test)):
        #F_test[i] = Function_Dichotomie_v0_Parsi4(A,B3,tau1,tau2,V_test[i],Sigma,Mu,p_Wtilde,m)
    #figure(15)
    #plot(V_test,F_test)
    #savefig('./V0_test.png')
    
    return x3

def Function_Dichotomie_tau2_Parsi4(x,Sigma,Mu,p_Wtilde,M,alpha,lam,p0):
    
    c = numpy.log((1.-p0)/p0)
    
    val = 0.0
    for m in xrange(M):
        dKL = 0.5 * (Mu[m,1]**2) * (1./Sigma[m,1] + 1./Sigma[m,0]) + ( (Sigma[m,1] - Sigma[m,0])**2 )/( 2. * Sigma[m,1] * Sigma[m,0] )
        if ((c/x)*(dKL - x)) >= 0.0:
            S = 1.0 / (1.0 + numpy.exp(-(c/x)*(dKL - x)) )
        if ((c/x)*(dKL - x)) < 0.0:
            S = numpy.exp((c/x)*(dKL - x)) / (1.0 + numpy.exp((c/x)*(dKL - x)) )
            
        val +=  dKL * (S - p_Wtilde[m,1])
         
    F = ( (c*val)/(x**2) ) - lam + (alpha - 1.0)/x   
   
    return F    
    
def Function_Dichotomie_tau2_Parsi4_FixedTau1(x,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1):
    
    val = 0.0
    for m in xrange(M):
        dKL = 0.5 * (Mu[m,1]**2) * (1./Sigma[m,1] + 1./Sigma[m,0]) + ( (Sigma[m,1] - Sigma[m,0])**2 )/( 2. * Sigma[m,1] * Sigma[m,0] )
        if (tau1*(dKL - x)) >= 0.0:
            S = 1.0 / (1.0 + numpy.exp(-tau1*(dKL - x)) )
        if (tau1*(dKL - x)) < 0.0:
            S = numpy.exp(tau1*(dKL - x)) / (1.0 + numpy.exp(tau1*(dKL - x)) )
            
        val +=  (S - p_Wtilde[m,1])
         
    F = (tau1*val) - lam + (alpha - 1.0)/x   
   
    return F        
    
def Function_Dichotomie_tau2_Parsi3(x,Sigma,Mu,p_Wtilde,M,alpha,lam,c):
    
    val = 0.0
    for m in xrange(M):
        if ((c/x)*((Mu[m,1]**2) - x)) >= 0.0:
            S = 1.0 / (1.0 + numpy.exp(-(c/x)*((Mu[m,1]**2) - x)) )
        if ((c/x)*((Mu[m,1]**2) - x)) < 0.0:
            S = numpy.exp((c/x)*((Mu[m,1]**2) - x)) / (1.0 + numpy.exp((c/x)*((Mu[m,1]**2) - x)) )
            
        val +=  (Mu[m,1]**2) * (S - p_Wtilde[m,1])
    
    F = ( (c*val)/(x**2) ) - lam + (alpha - 1.0)/x    
    
    return F 
    
    
def Function_Dichotomie_tau2_Parsi3_FixedTau1(x,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1):
    
    val = 0.0
    for m in xrange(M):
        if (tau1*((Mu[m,1]**2) - x)) >= 0.0:
            S = 1.0 / (1.0 + numpy.exp(-tau1*((Mu[m,1]**2) - x)) )
        if (tau1*((Mu[m,1]**2) - x)) < 0.0:
            S = numpy.exp(tau1*((Mu[m,1]**2) - x)) / (1.0 + numpy.exp(tau1*((Mu[m,1]**2) - x)) )
            
        val +=  (S - p_Wtilde[m,1])
    
    F = (tau1*val) - lam + (alpha - 1.0)/x    
    
    return F     

def Function_Dichotomie_tau2_Parsi3_Cond(x,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m):
    
    if ((c/x)*((Mu[m,1]**2) - x)) >= 0.0:
        S = 1.0 / (1.0 + numpy.exp(-(c/x)*((Mu[m,1]**2) - x)) )
    if ((c/x)*((Mu[m,1]**2) - x)) < 0.0:
        S = numpy.exp((c/x)*((Mu[m,1]**2) - x)) / (1.0 + numpy.exp((c/x)*((Mu[m,1]**2) - x)) )
        
    val =  (Mu[m,1]**2) * (S - p_Wtilde[m,1])
    
    F = ( (c*val)/(x**2) ) - lam + (alpha - 1.0)/x    
    
    return F
    
def Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(x,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m):
    
    if (tau1*((Mu[m,1]**2) - x)) >= 0.0:
        S = 1.0 / (1.0 + numpy.exp(-tau1*((Mu[m,1]**2) - x)) )
    if (tau1*((Mu[m,1]**2) - x)) < 0.0:
        S = numpy.exp(tau1*((Mu[m,1]**2) - x)) / (1.0 + numpy.exp(tau1*((Mu[m,1]**2) - x)) )
        
    val =  (S - p_Wtilde[m,1])
    
    F = (tau1*val) - lam + (alpha - 1.0)/x    
    
    return F    

def dichotomie_tau2_Parsi4(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,p0):

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi4(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi4(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi4(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi4(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi4(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #tau2_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(tau2_test), dtype=float)
    #for i in xrange(size(tau2_test)):
    #F_test[i] = Function_Dichotomie_tau2_Parsi4(tau2_test[i],Sigma,Mu,p_Wtilde,M,alpha,lam,p0)
    #figure(15)
    #plot(tau2_test,F_test)
    #savefig('./V0_test.png')
    
    return x3

def dichotomie_tau2_Parsi4_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1):

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi4_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi4_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi4_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi4_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi4_FixedTau1(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #tau2_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(tau2_test), dtype=float)
    #for i in xrange(size(tau2_test)):
    #F_test[i] = Function_Dichotomie_tau2_Parsi4_FixedTau1(tau2_test[i],Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    #figure(15)
    #plot(tau2_test,F_test)
    #savefig('./V0_test.png')
    
    return x3

def dichotomie_tau2_Parsi3(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c):

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi3(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,c)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi3(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,c)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi3(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,c)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi3(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,c)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi3(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,c)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #tau2_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(tau2_test), dtype=float)
    #for i in xrange(size(tau2_test)):
    #F_test[i] = Function_Dichotomie_tau2_Parsi3(au2_test[i],Sigma,Mu,p_Wtilde,M,alpha,lam,c)
    #figure(15)
    #plot(tau2_test,F_test)
    #savefig('./V0_test.png')
    
    return x3    

def dichotomie_tau2_Parsi3_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1):

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi3_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi3_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi3_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi3_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi3_FixedTau1(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    ##### Plotting F To verify is computed zero is right ####
    #tau2_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(tau2_test), dtype=float)
    #for i in xrange(size(tau2_test)):
    #F_test[i] = Function_Dichotomie_tau2_Parsi3_FixedTau1(au2_test[i],Sigma,Mu,p_Wtilde,M,alpha,lam,tau1)
    #figure(15)
    #plot(tau2_test,F_test)
    #savefig('./V0_test.png')
    
    return x3    

def dichotomie_tau2_Parsi3_Cond(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c,m):

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi3_Cond(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi3_Cond(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi3_Cond(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi3_Cond(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi3_Cond(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,c,m)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    return x3    

def dichotomie_tau2_Parsi3_Cond_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1,m):

    ##### Plotting F To verify is computed zero is right ####
    #tau2_test = numpy.arange(0.0001,20.,0.1)
    #F_test = numpy.zeros(size(tau2_test), dtype=float)
    #for i in xrange(size(tau2_test)):
        #F_test[i] = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(tau2_test[i],Sigma,Mu,p_Wtilde,M,alpha,lam,1,m)
    ##print 'F =',F_test[i]
    #figure(15)
    #plot(tau2_test,F_test)
    #savefig('./Posterior_tau2_curve.png')

    #  Choice of x1, x2 : F(x1)*F(x2) < 0.0 with F(x) 
    vxx1 = 0.001
    xx1 = vxx1 * numpy.random.rand()
    Fxx1 = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m)
    vxx2 = 0.05
    xx2 = vxx2 * numpy.random.rand()
    Fxx2 = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m)
    while Fxx1 * Fxx2 > 0.:
        vxx1 += 0.001
        xx1 = vxx1 * numpy.random.rand()
        Fxx1 = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(xx1,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m)
        vxx2 += 0.01
        xx2 = vxx2 * numpy.random.rand()
        Fxx2 = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(xx2,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m)
    
    x1 = xx1
    x2 = xx2
    
    F1 = Fxx1
    F2 = Fxx2
    
    n = 0
    x3 = (x1+x2)/2.0
    
    if F1 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x1
    
    if F2 == 0.:
        pyhrf.verbose(6,"Zeros found directly ...")
        x3 = x2
    
    dist = abs(x1-x2)
    
    if F1*F2 < 0.:
        while (dist > 1e-10 and n < 500):
            x3 = (x1+x2)/2.0
            F3 = Function_Dichotomie_tau2_Parsi3_Cond_FixedTau1(x3,Sigma,Mu,p_Wtilde,M,alpha,lam,tau1,m)
            if F3*F1 < 0.:
                F1 = F1
                F2 = F3
                x1 = x1
                x2 = x3
            if F3*F2 < 0.:
                F1 = F3
                F2 = F2
                x1 = x3
                x2 = x2
            dist = abs(x1-x2)
            n += 1

        x3 = (x1+x2)/2.0

    if F1*F2 > 0.:
        raise Exception(' ********* F(x1)*F(x2) > 0.  ->  Choice of x1,x2 in Dichotomie is NOT OK ********* ')
    
    if x3 < 0.:
        raise Exception(' ********* NEGATIVE TAU2 :((((  ********* ')
    
    return x3    


def maximization_tau2_ParsiMod4(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,p0):

    tau2 = dichotomie_tau2_Parsi4(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,p0)    
    return tau2
    
def maximization_tau2_ParsiMod4_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1):

    tau2 = dichotomie_tau2_Parsi4_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1)    
    return tau2    
    
def maximization_tau2_ParsiMod3(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c):
    
    tau2 = dichotomie_tau2_Parsi3(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c)  
    return tau2
    
def maximization_tau2_ParsiMod3_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1):
    
    tau2 = dichotomie_tau2_Parsi3_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1)  
    return tau2    
    
def maximization_tau2_ParsiMod3_Cond(tau2,q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c):
    
    for m in xrange(0,M):
        tau2[m] = dichotomie_tau2_Parsi3_Cond(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,c,m)  
    
    return tau2
    
def maximization_tau2_ParsiMod3_Cond_FixedTau1(tau2,q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1):
    
    for m in xrange(0,M):
        tau2[m] = dichotomie_tau2_Parsi3_Cond_FixedTau1(q_Z,p_Wtilde,Mu,Sigma,M,alpha,lam,tau1[m],m)  
    
    return tau2    

def maximization_mu_sigma_ParsiMod4(Mu,Sigma,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2):
    
    # TO DO: 
    # Fixed point equation --> iterating this bloc many times to
    # perform mixt-p estimation
    
    for m in xrange(0,M):
        for t in xrange(0,20):
            for k in xrange(0,K):

                if k != 0 :
                    Sigma[m,k] = dichotomie_v1_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,Sigma_A,J)
                    if Sigma[m,k] < eps:
                        Sigma[m,k] = eps
                        #raise Exception('PROBLEEEEEEEEEM : Very Low Active Variance ....')
                else:
                    Sigma[m,k] = dichotomie_v0_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,Sigma_A,J)
                    if Sigma[m,k] < eps:
                        Sigma[m,k] = eps
                        #raise Exception('PROBLEEEEEEEEEM : Very Low Inactive Variance ....')

                if k != 0 :  
                    Mu[m,k] = dichotomie_mu1_Parsi4(q_Z,p_Wtilde,tau1,tau2,Mu,Sigma,m,m_A,J)
                else:
                    Mu[m,k] = 0.

    return Mu , Sigma

def maximization_mu_sigma_ParsiMod3(Mu,Sigma,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,Iter,estimateW):
    for m in xrange(0,M):
        for k in xrange(0,K):

            S = sum( q_Z[m,k,:] )
            if S == 0.: 
                S = eps
                #raise Exception('PROBLEEEEEEEEEM : Divising by Zeeerooooooooo :( ....')
            S2 = sum ( 1. - p_Wtilde[m,1] * q_Z[m,1,:] )
            if S2 == 0.: 
                S2 = eps
                #raise Exception('PROBLEEEEEEEEEM : Divising by Zeeerooooooooo :( ....')

            if k != 0 :
                Sigma[m,k] = sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + Sigma_A[m,m,:] ) ) / S
                if Sigma[m,k] < eps:
                    Sigma[m,k] = eps
                    #raise Exception('PROBLEEEEEEEEEM : Very Low Active Variance ....')
                
            else:
                Sigma[m,k] = sum( ( 1. - p_Wtilde[m,1] * q_Z[m,1,:] ) * ( pow(m_A[:,m] ,2) + Sigma_A[m,m,:] ) ) / S2
                if Sigma[m,k] < eps:
                    Sigma[m,k] = eps
                    #raise Exception('PROBLEEEEEEEEEM : Very Low Inactive Variance ....')

            if k != 0 :                
                Mu[m,k] = Dichotomie_square(q_Z[m,k,:],p_Wtilde[m,k],Mu[m,k],Sigma[m,k],m_A[:,m],tau1,tau2,J,m,Iter,estimateW)
            else:
                Mu[m,k] = 0.

    return Mu , Sigma

def maximization_mu_sigma_ParsiMod3_Cond(Mu,Sigma,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,Iter,estimateW):
    for m in xrange(0,M):
        for k in xrange(0,K):

            S = sum( q_Z[m,k,:] )
            if S == 0.: 
                S = eps
                #raise Exception('PROBLEEEEEEEEEM : Divising by Zeeerooooooooo :( ....')
            S2 = sum ( 1. - p_Wtilde[m,1] * q_Z[m,1,:] )
            if S2 == 0.: 
                S2 = eps
                #raise Exception('PROBLEEEEEEEEEM : Divising by Zeeerooooooooo :( ....')
            
            if k != 0 :
                Sigma[m,k] = sum( q_Z[m,k,:] * ( pow(m_A[:,m] - Mu[m,k] ,2) + Sigma_A[m,m,:] ) ) / S
                if Sigma[m,k] < eps:
                    Sigma[m,k] = eps
                    #raise Exception('PROBLEEEEEEEEEM : Very Low Active Variance ....')
                
            else:
                Sigma[m,k] = sum( ( 1. - p_Wtilde[m,1] * q_Z[m,1,:] ) * ( pow(m_A[:,m] ,2) + Sigma_A[m,m,:] ) ) / S2
                if Sigma[m,k] < eps:
                    Sigma[m,k] = eps
                    #raise Exception('PROBLEEEEEEEEEM : Very Low Inactive Variance ....')

            if k != 0 :                
                Mu[m,k] = Dichotomie_square(q_Z[m,k,:],p_Wtilde[m,k],Mu[m,k],Sigma[m,k],m_A[:,m],tau1[m],tau2[m],J,m,Iter,estimateW)
            else:
                Mu[m,k] = 0.
            

    return Mu , Sigma

def maximization_sigma_noise_ParsiMod(Y,X,m_A,m_H,Sigma_H,Sigma_A,sigma_epsilone,zerosMM,N,J,p_Wtilde,Gamma):
    Htilde = zerosMM.copy() #numpy.zeros((M,M),dtype=float)
    for i in xrange(0,J):
        S = numpy.zeros((N),dtype=float)
        m = 0
        for k in X:
            m2 = 0
            for k2 in X:
                Htilde[m,m2] =  numpy.dot(numpy.dot(numpy.dot(numpy.dot(m_H.transpose(),X[k].transpose()),Gamma),X[k2]),m_H)
                Htilde[m,m2] += (numpy.dot(numpy.dot(numpy.dot(Sigma_H,X[k].transpose()),Gamma),X[k2])).trace()
                Htilde[m,m2] = Htilde[m,m2] * p_Wtilde[m,1] * p_Wtilde[m2,1]
                m2 += 1
            S += p_Wtilde[m,1] * m_A[i,m] * numpy.dot(numpy.dot(X[k],m_H),Gamma)
            m += 1
        sigma_epsilone[i] = numpy.dot( -2*S, Y[:,i])
        sigma_epsilone[i] += (numpy.dot(Sigma_A[:,:,i],Htilde)).trace()
        sigma_epsilone[i] += numpy.dot( numpy.dot(m_A[i,:].transpose(), Htilde),m_A[i,:] )
        sigma_epsilone[i] += numpy.dot(numpy.dot(Y[:,i].transpose(),Gamma), Y[:,i])
        sigma_epsilone[i] /= N

    return sigma_epsilone

def computeFit(m_H, m_A, X, J, N):
    
  #print 'Computing Fit ...'
  stimIndSignal = numpy.zeros((N,J), dtype=numpy.float64)

  for i in xrange(0,J):
    m = 0
    for k in X:
      stimIndSignal[:,i] += m_A[i,m] * numpy.dot(X[k],m_H)
      m += 1

  return stimIndSignal
  
def computeParsiFit(w, m_H, m_A, X, J, N):

  #print 'Computing Parsimonious Fit ...'
  #print 'w =',w
  stimIndSignal = numpy.zeros((N,J), dtype=numpy.float64)

  for i in xrange(0,J):
    m = 0
    for k in X:
      stimIndSignal[:,i] += w[m] * m_A[i,m] * numpy.dot(X[k],m_H)
      m += 1
    
  return stimIndSignal  

def MeanUpdate(stimIndSignal,Y, N, J):

    Y_MeanUpdated = numpy.zeros((N,J), dtype=numpy.float64)
    for i in xrange(0,J):
        Y_MeanUpdated[:,i] = Y[:,i] - Y[:,i].mean(0)
        Y_MeanUpdated[:,i] = Y_MeanUpdated[:,i] + stimIndSignal[:,i].mean(0)

    return Y_MeanUpdated

def LikeLihood(Y, m_H, m_A, X, sigma_epsilone, J, N, Gamma):

  StimulusInducedSignal = numpy.zeros((N,J), dtype=numpy.float64)
  StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)

  Diff = numpy.zeros((N,J), dtype=numpy.float64)
  Diff = Y - StimulusInducedSignal

  Li_final = 1.

  Const1 = (2*numpy.pi)**(N/2.)
  Const3 = (numpy.linalg.det(Gamma))**(-0.5)

  for i in xrange(0,J):
    var_noise_i= pow(sigma_epsilone[i],2)
    Const2 = pow(sigma_epsilone[i],N)
    Const = Const1 * Const2 * Const3
    Li = (1./Const) * numpy.exp(- (0.5/var_noise_i) * numpy.dot(numpy.dot(Diff[:,i].transpose(), Gamma), Diff[:,i]))
    Li_final *= Li

  print 'Li_final =',Li_final

def Log_LikeLihood(Y, m_H, m_A, X, sigma_epsilone, J, N, Gamma):

  StimulusInducedSignal = numpy.zeros((N,J), dtype=numpy.float64)
  StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)

  Diff = numpy.zeros((N,J), dtype=numpy.float64)
  Diff = Y - StimulusInducedSignal

  Li_final = 0.

  Const1 = 0.5 * N * numpy.log(2*numpy.pi)
  Const3 = - 0.5 * numpy.log(numpy.linalg.det(Gamma))

  for i in xrange(0,J):
    Const2 = N * numpy.log(sigma_epsilone[i])
    Const = Const1 + Const2 + Const3
    var_noise_i = sigma_epsilone[i] ** 2
    Li = - Const - ( (0.5/var_noise_i) * numpy.dot(numpy.dot(Diff[:,i].transpose(), Gamma), Diff[:,i]))
    Li_final += Li

  print 'Li_final =',Li_final

eps_FreeEnergy = 0.00000001

def A_Entropy(Sigma_A, M, J):

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

    pyhrf.verbose(3,'Computing HRF Entropy ...')
    Det_Sigma_H = numpy.linalg.det(Sigma_H)
    Const = (2*numpy.pi*numpy.exp(1))**D
    Entropy = numpy.sqrt( Const * Det_Sigma_H)
    Entropy = - numpy.log(Entropy + eps_FreeEnergy)

    return Entropy

def Z_Entropy(q_Z, M, J):

    pyhrf.verbose(3,'Computing Z Entropy ...')
    Entropy = 0.0
    for j in xrange(0,J):
        for m in xrange(0,M):
            Entropy += q_Z[m,1,j] * numpy.log(q_Z[m,1,j] + eps_FreeEnergy) + q_Z[m,0,j] * numpy.log(q_Z[m,0,j] + eps_FreeEnergy)

    return Entropy

def W_Entropy(p_Wtilde, M):

    pyhrf.verbose(3,'Computing W Entropy ...')
    Entropy = 0.0
    for m in xrange(0,M):
        Entropy += p_Wtilde[m,1] * numpy.log(p_Wtilde[m,1] + eps_FreeEnergy) + p_Wtilde[m,0] * numpy.log(p_Wtilde[m,0] + eps_FreeEnergy)

    return Entropy

def W_Entropy_RVM(V_Wtilde, M):

    pyhrf.verbose(3,'Computing W Entropy ...')
    Entropy = 0.0
    for m in xrange(0,M):
        Const = (2*numpy.pi*numpy.exp(1))
        Entropy_m = numpy.sqrt( Const * V_Wtilde[m,m])
        Entropy += numpy.log(Entropy_m + eps_FreeEnergy)
    Entropy = - Entropy
    
    return Entropy

def Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,Model):

    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    if Model=="ParsiMod1" or Model=="ParsiMod3" or Model=="ParsiMod4":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod4")
        EntropyW = W_Entropy(p_Wtilde,M)
        Total_Entropy = EntropyA + EntropyH + EntropyZ + EntropyW
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
    if Model=="ParsiMod1":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod1")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod1(q_Z, p_Wtilde, M, S, J, K, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW
    if Model=="ParsiMod3":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod3")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod3(p_Wtilde, mu_M, M, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW
    if Model=="ParsiMod4":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod4")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod4(p_Wtilde, mu_M, sigma_M, M, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW

    FreeEnergy = EPtilde - Total_Entropy

    return FreeEnergy

def Compute_FreeEnergy_RVM(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,m_Wtilde,V_Wtilde,alpha_RVM,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K):

    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    EntropyW = W_Entropy_RVM(V_Wtilde,M)
    Total_Entropy = EntropyA + EntropyH + EntropyZ + EntropyW

    ## Second Part (likelihood)
    EPtildeLikelihood = UtilsC.expectation_Ptilde_Likelihood_RVM(y_tilde,m_A,m_H,XX.astype(int32),Sigma_A,sigma_epsilone,Sigma_H,Gamma,m_Wtilde,V_Wtilde,XGamma,J,D,M,N,Det_Gamma)
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    EPtildeA = UtilsC.expectation_Ptilde_A(m_A,Sigma_A,p_Wtilde,q_Z,mu_M,sigma_M,J,M,K)
    
    EPtildeH = UtilsC.expectation_Ptilde_H(R, m_H, Sigma_H, D, sigmaH, Det_invR)
    EPtildeZ = UtilsC.expectation_Ptilde_Z(q_Z, neighboursIndexes.astype(int32), Beta, J, K, M, maxNeighbours)
    EPtildeW = UtilsC.expectation_Ptilde_W_RVM(q_Z, m_Wtilde, V_Wtilde, M, alpha_RVM)
    EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW

    FreeEnergy = EPtilde - Total_Entropy

    return FreeEnergy

def Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,Model):

    ## First part (Entropy):
    EntropyA = A_Entropy(Sigma_A, M, J)
    EntropyH = H_Entropy(Sigma_H, D)
    EntropyZ = Z_Entropy(q_Z,M,J)
    if Model=="ParsiMod1" or Model=="ParsiMod3" or Model=="ParsiMod4":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod4")
        EntropyW = W_Entropy(p_Wtilde,M)
        Total_Entropy = EntropyA + EntropyH + EntropyZ + EntropyW
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
    if Model=="ParsiMod1":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod1")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod1(q_Z, p_Wtilde, M, S, J, K, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW
    if Model=="ParsiMod3":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod3")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod3_Cond(p_Wtilde, mu_M, M, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW
    if Model=="ParsiMod4":
        pyhrf.verbose(5,"Computing Free Energy for ParsiMod4")
        EPtildeW = UtilsC.expectation_Ptilde_W_ParsiMod4(p_Wtilde, mu_M, sigma_M, M, tau1, tau2)
        EPtilde = EPtildeLikelihood + EPtildeA + EPtildeH + EPtildeZ + EPtildeW

    FreeEnergy = EPtilde - Total_Entropy

    return FreeEnergy

def Main_vbjde_Fast(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=0):
    pyhrf.verbose(1,"Fast EM started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 10
    gamma_h = 10
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=float)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    zerosDD = numpy.zeros((D,D),dtype=float)
    zerosD = numpy.zeros((D),dtype=float)
    zerosND = numpy.zeros((N,D),dtype=float)
    zerosMM = numpy.zeros((M,M),dtype=float)
    zerosJMD = numpy.zeros((J,M,D),dtype=float)
    zerosK = numpy.zeros(K)
    P = PolyMat( N , 4 , TR)
    zerosP = numpy.zeros((P.shape[0]),dtype=float)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    t1 = time.time()
    for ni in xrange(0,NitMin):
        #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
        pyhrf.verbose(3, "E H step ...")
        Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
        pyhrf.verbose(3, "E Z step ...")
        q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
            sigmaH /= D
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)

    m_H1[:] = m_H[:]
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
    Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
    Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    m_H1[:] = m_H[:]
    q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
        sigmaH /= D
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
    L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
    ni += 2
    if (Crit_H > 5e-3):
        while (Crit_H > 5e-3) and (ni < NitMax):
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
            Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
            Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            m_H1[:] = m_H[:]
            q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
                sigmaH /= D
            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
            ni +=1
    t2 = time.time()
    CompTime = t2 - t1
    Norm = norm(m_H)
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print "Beta = " + str(Beta)
    return m_A, m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Sigma_A


def MiniVEM_CompMod(Thrf,TR,dt,beta,Y,K,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,p_Wtilde,scale,Q_barnCond,XGamma,tau1,tau2,Nit,sigmaH,estimateHRF):
    
    #print 'InitVar =',InitVar,',    InitMean =',InitMean,',     gamma_h =',gamma_h

    Init_sigmaH = sigmaH

    IM_val = np.array([-5.,5.])
    IV_val = np.array([0.008,0.016,0.032,0.064,0.128,0.256,0.512])
    #IV_val = np.array([0.01,0.05,0.1,0.5])
    gammah_val = np.array([1000])
    MiniVemStep = IM_val.shape[0]*IV_val.shape[0]*gammah_val.shape[0]
    
    Init_mixt_p_gammah = []
                
    pyhrf.verbose(1,"Number of tested initialisation is %s" %MiniVemStep)
    
    t1_MiniVEM = time.time()
    FE = []
    for Gh in gammah_val:
        for InitVar in IV_val:
            for InitMean in IM_val:
                Init_mixt_p_gammah += [[InitVar,InitMean,Gh]]
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
                
                #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
                TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
                m_h = m_h[:D]
                m_H = numpy.array(m_h).astype(numpy.float64)
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
                
                gamma_h = Gh
                sigma_M = numpy.ones((M,K),dtype=numpy.float64)
                sigma_M[:,0] = 0.1
                sigma_M[:,1] = 1.0
                mu_M = numpy.zeros((M,K),dtype=numpy.float64)
                for k in xrange(1,K):
                    mu_M[:,k] = InitMean
                Sigma_A = numpy.zeros((M,M,J),numpy.float64)
                for j in xrange(0,J):
                    Sigma_A[:,:,j] = 0.01*numpy.identity(M)    
                m_A = numpy.zeros((J,M),dtype=numpy.float64)
                for j in xrange(0,J):
                    for m in xrange(0,M):
                        for k in xrange(0,K):
                            m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

                for ni in xrange(0,Nit+1):
                    pyhrf.verbose(3,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
                    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
                    val = reshape(m_A,(M*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
                    m_A = reshape(val, (J,M))
                    
                    if estimateHRF:
                        UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                        m_H[0] = 0
                        m_H[-1] = 0
                    
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                    val = reshape(q_Z,(M*K*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    q_Z = reshape(val, (M,K,J))
                    
                    if estimateHRF:
                        if gamma_h > 0:
                            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                        else:
                            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
                    UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
                    PL = numpy.dot(P,L)
                    y_tilde = Y - PL
                    for m in xrange(0,M):
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
                    
                FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"CompMod")
                FE += [FreeEnergy]
    
    max_FE, max_FE_ind = maximum(FE)
    InitVar = Init_mixt_p_gammah[max_FE_ind][0]
    InitMean = Init_mixt_p_gammah[max_FE_ind][1]
    Initgamma_h = Init_mixt_p_gammah[max_FE_ind][2]
    
    t2_MiniVEM = time.time()
    pyhrf.verbose(1,"MiniVEM duration is %s" %format_duration(t2_MiniVEM-t1_MiniVEM))
    pyhrf.verbose(1,"Choosed initialisation is : var = %s,  mean = %s,  gamma_h = %s" %(InitVar,InitMean,Initgamma_h))
    
    return InitVar, InitMean, Initgamma_h

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
    
    #D = int(numpy.ceil(Thrf/dt)) ##############################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    condition_names = []

    #print '======================'
    #print M
    #print '======================'
    #-----------------------------------------------------------------------#
    # put neighb#figure(1)
    #plot(m_H,'r')
    #hold(False)
    #draw()
    #show()our lists into a 2D numpy array so that it will be easily
    # passed to C-code
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
    
    #test_mu1 = [[] for m in xrange(M)]
    
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
            
    t1 = time.time()
    #ion()
    #timeAxis = numpy.arange(0, 25., 0.5)
    #hrf0 = genBezierHRF(timeAxis=timeAxis, pic=[4,1], picw=3)[1]
    
    #print 'InitVar =',InitVar
    #print 'InitMean =',InitMean
    
    for ni in xrange(0,NitMin):
        #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        #t01 = time.time()
        UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        
        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
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
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
                
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        #q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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

    #print "------------------------------ Iteration n° " + str(ni+2) + " ------------------------------"
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    #t01 = time.time()
    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
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
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
    
    #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    #Crit_A = sum(DIFF) / len(find(DIFF != 0))
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    #q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh):
        #while ( (Crit_AH > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) )) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            #t01 = time.time()
            UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
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
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            #Crit_A = sum(DIFF) / len(find(DIFF != 0))
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            #q_Z = reshape(val, (M,K,J))

            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
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
        #plt.legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        #plt.legend( ('m=0','m=1') ) 
        plt.savefig('./mu1_Iter_CompMod.png')
        
        plt.figure(6)
        plt.plot(h_norm_array)
        plt.savefig('./HRF_Norm_CompMod.png')
        
        Data_save = xndarray(h_norm_array, ['Iteration'])
        Data_save.save('./HRF_Norm_Comp.nii')
        
        
    #for m in xrange(M):
        #plt.figure(4+m)
        #plt.plot(test_mu1[m])
        #plt.savefig('./mu1_CompMod_Cond_%s.png' %m)
         

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



    #if computeContrast:
	#if len(contrasts) >0:
            #pyhrf.verbose(3, 'Compute contrasts ...')
	    #nrls_conds = dict([(str(cn), m_A[:,ic]) \
                                   #for ic,cn in enumerate(condition_names)] )
	    #n = 0
	    #for cname in contrasts:
		##------------ contrasts ------------#
		#contrast_expr = AExpr(contrasts[cname], **nrls_conds)
		#contrast_expr.check()
		#contrast = contrast_expr.evaluate()
		#CONTRAST[:,n] = contrast
		##------------ contrasts ------------#

		##------------ variance -------------#
		#ContrastCoef = numpy.zeros(M,dtype=float)
		#ind_conds0 = {}
		#for m in xrange(0,M):
		    #ind_conds0[condition_names[m]] = 0.0
		#for m in xrange(0,M):
		    #ind_conds = ind_conds0.copy()
		    #ind_conds[condition_names[m]] = 1.0
		    #ContrastCoef[m] = eval(contrasts[cname],ind_conds)
		#ActiveContrasts = (ContrastCoef != 0) * numpy.ones(M,dtype=float)
		##CovM = numpy.ones(M,dtype=float)
		#for j in xrange(0,J):
		    #CovM = numpy.ones(M,dtype=float)
		    #for m in xrange(0,M):
			#if ActiveContrasts[m]:
			    #CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
			    #for m2 in xrange(0,M):
				#if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
				    #CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
				    #CovM[m2] = 0
				    #CovM[m] = 0
		##------------ variance -------------#
		#n +=1
            #pyhrf.verbose(3, 'Done contrasts computing.')
	##+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
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

def Main_vbjpde(graph,Y,Onsets,Thrf,Pmask,TR,dt,K=2,v_h=0.1,beta=0.8,beta_Q=0.8,NitMax = -1,NitMin = 1,estimateBeta=True,outDir = './'):
    pyhrf.verbose(1,"Fast EM with C extension for JPDE started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-8
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    I = uint(Pmask.max()+1)
    #Par0 = reshape(Pmask,(l,l))
    #print 'I:', I
    #print 'J:', J
    condition_names = []
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1

    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    RR = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    # ----------- activation class --------------#
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    # ----------- activation class --------------#
    # --------- Parcellisation class ------------#
    q_Q = numpy.zeros((I,J),dtype=numpy.float64)
    Z_tilde = q_Z.copy()
    for j in xrange(0,J):
        ind = Pmask[j]
        q_Q[ind,j] = 1
    Q_tilde = q_Q.copy()
    q_Q0 = q_Q.copy()
    ion()
    # --------- Parcellisation class ------------#
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    m_H = numpy.ones((D,J),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    mu_bar = numpy.zeros((D),dtype=numpy.float64)
    Sigma_bar = numpy.zeros((D,D),dtype=numpy.float64)
    Sum_Sigma_h_k = numpy.zeros((D),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    for j in xrange(0,J):
        m_H[:,j] = numpy.array(m_h).astype(numpy.float64)
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], sigma_M[m,k])*Z_tilde[m,k,j]
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    #print PL.shape()
    y_tilde = Y - PL
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_Q = 1
    cA = []
    cH = []
    cZ = []
    cQ = []
    Ndrift = L.shape[0]
    t1 = time.time()
    ni = 0
    m_H1 = m_H.copy()
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    q_Q1 = q_Q.copy()
    zerosI = numpy.zeros((I),dtype=float)
    zerosK = numpy.zeros((K),dtype=float)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosDD = 0 * numpy.identity(D)#numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosP = numpy.zeros((P.shape[0]),dtype=float)
    zerosMM = 0 * numpy.identity(M)
    HRFDict = []
    HRFDictCovar = []
    for i in xrange(0,I):
        #tmp = HRFDict0[i]
        HRFDict.append(numpy.array(m_h).astype(numpy.float64))
        tmp2 = numpy.identity((D))
        HRFDictCovar.append(tmp2) #HRFDictCovar.append(tmp2/sigmaH)
        #print HRFDictCovar[i]
        #print numpy.linalg.det(HRFDictCovar[i])
    #raw_input('')

    while ((Crit_Q > Thresh) or(Crit_H > Thresh) or (Crit_Z > Thresh) or (Crit_A > Thresh) or (ni < NitMin)) and (ni < NitMax) :
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(2, "E A step ...")
        #tt1 = time.time()

        Sigma_A, m_A = expectation_AP2(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
        #UtilsC.expectation_AP(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

        #tt2 = time.time()
        #print tt2-tt1
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        pyhrf.verbose(2, "E H step ...")
        Sigma_H, m_H = expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone)
        m_H[0,:] = 0
        m_H[-1,:] = 0
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:,:] = m_H[:,:]
        pyhrf.verbose(2, "E Z step ...")
        #q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
        UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)

        Crit_Z = (numpy.linalg.norm( reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)) ) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        pyhrf.verbose(2, "E Q step ...")
        q_Q,Q_tilde = expectation_Q(m_H,Sigma_H,sigma_M,beta_Q,Q_tilde,HRFDict,HRFDictCovar,q_Q,graph,J,I,zerosI)
        Crit_Q = (numpy.linalg.norm( reshape(q_Q,(I*J)) - reshape(q_Q1,(I*J)) ) / numpy.linalg.norm( reshape(q_Q1,(I*J)) ))**2
        cQ += [Crit_Q]
        q_Q1[:,:] = q_Q[:,:]

        Pmask = classify(q_Q)
	#plt.figure(10)
	#plt.imshow(reshape(Pmask,(20,20)))
	#plt.colorbar() 
	#plt.figure(11)
	#plt.imshow(reshape(q_Q[0,:],(20,20)))
	#plt.colorbar() 
	
	#plt.draw()
	
        for i in range(0,I):
            HRFDict[i] = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],RR/v_h)
        pyhrf.verbose(2,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_P2(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)

        #print m_A.shape,m_H.transpose().shape
        #raw_input('')
        #UtilsC.maximization_LP(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
        L = maximization_LP(Y,m_A,X,m_H,L,P,zerosP)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(2,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(2,"End estimating beta")
            pyhrf.verbose.printNdarray(2, Beta)
        pyhrf.verbose(2,"M sigma noise step ...")
        UtilsC.maximization_sigma_noiseP(PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        sigma_epsilone = maximization_sigma_noiseP(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
        ni += 1
    t2 = time.time()
    CompTime = t2 - t1
    for i in range(0,I):
        HRFDict[i] = HRFDict[i]/norm(HRFDict[i])
    Mnorm = 0
    for j in xrange(0,J):
        Norm = norm(m_H[:,j])
        m_H[:,j] /= Norm
        m_A[j,:] *= Norm
        Mnorm += Norm/J
    mu_M *= Mnorm
    sigma_M *= Mnorm
    sigma_M = sqrt(sigma_M)
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "Beta = " + str(Beta)
    #write_volume(Pmask,op.join('/home/chaari/Boulot/Data/JPDE/simuls','Estimated_Pmask.nii'))
    #write_volume(numpy.array(HRFDict),'Estimated_HRFs.nii')
    return m_A,m_H, q_Z, q_Q, Pmask, HRFDict, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, cA[2:],cH[2:],cZ[2:],cQ[2:]


def Main_vbjpde2(NRL0,graph,Y,Onsets,Thrf,K,I,Pmask,HRFDict0,HRFDictCovar,TR,dt,v_h=0.1,beta=0.8,beta_Q=0.8,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False):
    pyhrf.verbose(1,"Fast EM with C extension for JPDE started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-8
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    Par0 = reshape(Pmask,(l,l))
    condition_names = []
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1

    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    RR = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    # ----------- activation class --------------#
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    # ----------- activation class --------------#
    # --------- Parcellisation class ------------#
    q_Q = numpy.zeros((I,J),dtype=numpy.float64)
    from scipy.misc import fromimage
    import Image
    dataDir = '/home/chaari/Logiciels/pyhrf/pyhrf-free/trunk/python/pyhrf/datafiles/'
    fn = op.join(dataDir,'simu_par20_3p_0b.png')# simu_par20_3p   simu_labels_carre1.png
    label = fromimage(Image.open(fn))
    Par_im = label
    label = reshape(label,(J))
    Par = label
    #ind = find(Par==0)
    #Par[ind] = 3
    #ind = find(Par==1)
    #Par[ind] = 0
    #ind = find(Par==2)
    #Par[ind] = 1
    #ind = find(Par==3)
    #Par[ind] = 2
    #for j in xrange(0,J):
        ##print label[j]
        #if (label[j]==1):
            ##print "cnd: 1"
            #q_Z[0,1,j] = 1
        #else:
            #q_Z[0,0,j] = 1
        #if (label[j]==2):
            ##print "cnd: 2"
            #q_Z[1,1,j] = 1
        #else:
            #q_Z[1,0,j] = 1

    #fn = op.join(dataDir,'simu_par20_3p_c1b.png')#simu_labels_carre1.png
    #label1 = fromimage(Image.open(fn))
    #q_Z[0,1,:] = reshape(label1,(J))
    #q_Z[0,0,:] = 1- reshape(label1,(J))

    #fn = op.join(dataDir,'simu_par20_3p_c2.png')#simu_labels_carre1.png
    #label1 = fromimage(Image.open(fn))
    #q_Z[1,1,:] = reshape(label1,(J))
    #q_Z[1,0,:] = 1- reshape(label1,(J))
    Z_tilde = q_Z.copy()
    for j in xrange(0,J):
        ind = label[j]
        q_Q[ind,j] = 1

    #for j in xrange(0,J):
        #ind = random.randint(0,I-1)
        #q_Q[ind,j] = 1
        ##q_Q[ind,j] = 1

    Q_tilde = q_Q.copy()
    q_Q0 = q_Q.copy()
    ion()
    # --------- Parcellisation class ------------#
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    m_H = numpy.ones((D,J),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    mu_bar = numpy.zeros((D),dtype=numpy.float64)
    Sigma_bar = numpy.zeros((D,D),dtype=numpy.float64)
    Sum_Sigma_h_k = numpy.zeros((D),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    for j in xrange(0,J):
        m_H[:,j] = numpy.array(m_h).astype(numpy.float64)
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], sigma_M[m,k])*Z_tilde[m,k,j]
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_Q = 1
    cA = []
    cH = []
    cZ = []
    cQ = []
    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    #XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    #m1 = 0
    #for k1 in X: # Loop over the M conditions
        #m2 = 0
        #for k2 in X:
            #Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            #m2 += 1
        #XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        #m1 += 1
    t1 = time.time()
    ni = 0
    m_H1 = m_H.copy()
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    q_Q1 = q_Q.copy()
    zerosI = numpy.zeros((I),dtype=float)
    zerosK = numpy.zeros((K),dtype=float)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosDD = 0 * numpy.identity(D)#numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosP = numpy.zeros((P.shape[0]),dtype=float)
    zerosMM = 0 * numpy.identity(M)
    HRFDict = []
    #Par = numpy.zeros((l,l),dtype=numpy.float64)
    for i in xrange(0,I):
        tmp = HRFDict0[i]
        HRFDict.append(tmp)
        #z1 = q_Q[i,:]
        #z2 = reshape(z1,(l,l))
        #Par += i*reshape(thresholding(z1.copy(),0.5),(l,l))

    while ((Crit_H > Thresh) or (Crit_Z > Thresh) or (Crit_A > Thresh) or (ni < NitMin)) and (ni < NitMax) :

        #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(2, "E A step ...")
        Sigma_A, m_A = expectation_AP2(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        pyhrf.verbose(2, "E H step ...")
        Sigma_H, m_H = expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone)
        m_H[0,:] = 0
        m_H[-1,:] = 0
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:,:] = m_H[:,:]
        pyhrf.verbose(2, "E Z step ...")
        q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
        #UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        pyhrf.verbose(2, "E Q step ...")
        q_Q,Q_tilde = expectation_Q(m_H,Sigma_H,sigma_M,beta_Q,Q_tilde,HRFDict,HRFDictCovar,q_Q,graph,J,I,zerosI)
        figure(3).clf()

        figure(6).clf()
        Par_im = classify(q_Q)
        Par_im = reshape(Par_im,(l,l))
        Par00 = reshape(Par0,J)
        ind1 = 50
        ind2 = 204
        ind3 = 215
        ind = [50,215,204]
        #Par00[ind1] = 5
        #Par00[ind3] = 6
        #Par00[ind2] = 7
        DVect = numpy.zeros((2,J))
        DVect[0,:] = Par00
        DVect[1,:] = reshape(Par_im,(J))
        dice = sp.spatial.distance.pdist(DVect,'dice')
        DIf = abs(Par00 - reshape(Par_im,(J)))
        pourcentage_error = 100.0 * len( argwhere(DIf >0) ) / J
        
        plt.figure(5).clf()
        plt.figure(5)
        plt.subplot(1,2,2)
        plt.imshow(Par_im,interpolation='nearest')
        plt.xlabel('error = %.4f ' %pourcentage_error + "%")
        plt.title('Estimated Parcel mask')
        plt.colorbar()
        plt.subplot(1,2,1)
        plt.imshow(reshape(Par00,(l,l)),interpolation='nearest')
        plt.xlabel('DICE= %.4f' %dice)

        #plt.imshow(Par0,interpolation='nearest')
        plt.title('Reference Parcel mask')
        plt.colorbar()
        plt.draw()
        for i in range(0,I):
            z1 = q_Q[i,:]
            z2 = reshape(z1,(l,l))
            plt.figure(3)
            plt.subplot(1,I,1+i)
            plt.imshow(z2,interpolation='nearest')
            plt.title('Parcel ' + str(i))
            plt.colorbar()
            plt.hold(False)
        plt.figure(7).clf()
        for m in range(0,M):
            hh = HRFDict[i]#mean_HRF(m_H,q_Q[i,:])
            z1 = m_A[:,m]
            z2 = reshape(z1,(l,l))
            plt.figure(7)
            plt.subplot(M,M,1 + m)
            plt.imshow(z2,interpolation='nearest')
            #plt.imshow(z2*norm(hh),interpolation='nearest')
            plt.title("Est: m = " + str(m))
            plt.colorbar()

            z1 = m_A[:,m]
            z2 = reshape(NRL0[m,:],(l,l))
            plt.figure(7)
            plt.subplot(M,M,1 + m + 2)
            plt.imshow(z2,interpolation='nearest')
            plt.title("Ref: m = " + str(m))
            plt.colorbar()
            #plt.hold(False)
            for k in range(0,K):
                z1 = q_Z[m,k,:]
                z2 = reshape(z1,(l,l))
                plt.figure(6)
                plt.subplot(M,K,1 + m*K + k)
                plt.imshow(z2,interpolation='nearest')
                plt.title("m = " + str(m) +"k = " + str(k))
                plt.colorbar()
                plt.hold(False)
        plt.draw()
        R = RR
        R2 = RR
        print v_h
        #print Par.shape
        for i in range(0,I):
            #hh = zeros(D)
            #Nvox = 0
            ##Pr = 0
            #for j in xrange(0,J):
                #if (Par[j] == i):
                    #Nvox += 1
                    #hh += m_H[:,j]
                    ##hh += m_H[:,j]*q_Q[i,j]
                    ##Pr += q_Q[i,j]
            ##hh /= Pr
            #hh /= Nvox
            #HRFDict[i] = hh
            #print sum(mean_HRF(m_H,q_Q[i,:]) - hh)
            #HRFDict[i] = mean_HRF(m_H,q_Q[i,:])
            #HRFDict[i] = hh/Nvox#mean_HRF(m_H,q_Q[i,:])
            HRFDict[i] = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],R2/v_h)
            #HRFDict[i] = maximization_h_k_prior2(m_H,q_Q[i,:],HRFDictCovar[i],R2/v_h)
            #if (i<I):
                #HRFDict[i] = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],R2/v_h)
            hh = mean_HRF(m_H,q_Q[i,:])
            hh2 = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],R2/v_h)
            #hh2 = hh
            #hh = HRFDict[i]
            plt.figure(2)
            plt.subplot(1,I,1+i)
            #plt.plot(hh)
            plt.plot(hh/(norm(hh)+eps))
            plt.grid(True)
            plt.hold(True)
            plt.plot(hh2/(norm(hh2)+eps),'r')
            plt.grid(True)
            plt.hold(True)
            plt.plot(HRFDict0[i],'k')
            plt.hold(True)
            plt.plot(numpy.array(m_h),'y')
            plt.legend(("Est","Est Prior","Ref","Init"))
            plt.title('mse c1 = %.11f' %compute_MSE(hh/(norm(hh)),HRFDict0[i]) )
            plt.xlabel('parcel %d' %i)
            #plt.xlabel('mse c2 = %.11f' %compute_MSE(hh,HRFDict0[1]))
            plt.hold(False)


            #figure(12)
            #subplot(1,I,1+i)
            ##plot(hh)
            #plot(m_H[:,ind[i]]/(norm(m_H[:,ind[i]])+eps))
            #grid(True)
            #hold(True)
            #plot(HRFDict0[i],'k')
            #legend(("Est","Ref"))
            #title("voxel%d" %ind[i] )
            #xlabel('parcel %d' %i)
            ##xlabel('mse c2 = %.11f' %compute_MSE(hh,HRFDict0[1]))
            #hold(False)


        show()
        pyhrf.verbose(2,"M (mu,sigma) step ...")
        #mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        mu_M , sigma_M = maximization_mu_sigma_P2(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        #mu_M[0,0] = 0
        #mu_M[0,1] = 3.2
        #mu_M[1,0] = 0
        #mu_M[1,1] = 3.2

        #sigma_M[0,0] = 0.5
        #sigma_M[0,1] = 0.5
        #sigma_M[1,0] = 0.5
        #sigma_M[1,1] = 0.5


        #print 'mu_M:', mu_M[:,1,:]
        #print 'sigma_M:', sigma_M[:,1,:]
        print 'mu_M:', mu_M[:,:]
        print 'sigma_M:', sqrt(sigma_M[:,:])
        print '------------------------------'
        L = maximization_LP(Y,m_A,X,m_H,L,P,zerosP)
        #UtilsC.maximization_L(Y,m_A,(mean_HRF(m_H,q_Q[0,:]) + mean_HRF(m_H,q_Q[1,:])+ mean_HRF(m_H,q_Q[2,:]))/3.0,L,P,XX.astype(int32),J,D,M,Ndrift,N)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(2,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(2,"End estimating beta")
            pyhrf.verbose.printNdarray(2, Beta)
        pyhrf.verbose(2,"M sigma noise step ...")
        sigma_epsilone = maximization_sigma_noiseP(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
        #sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
        #UtilsC.maximization_sigma_noise(PL,sigma_epsilone,mean_HRF_covar(Sigma_H,q_Q,zerosDD),Y,m_A,(mean_HRF(m_H,q_Q[0,:]) + mean_HRF(m_H,q_Q[1,:])+ mean_HRF(m_H,q_Q[2,:]))/3.0,Sigma_A,XX.astype(int32),J,D,M,N)
        ni += 1
    t2 = time.time()
    draw()
    show()
    CompTime = t2 - t1
    PLOT = 0
    if PLOT:
        for m in range(0,M):
            for k in range(0,K):
                z1 = q_Z[m,k,:];
                z2 = reshape(z1,(l,l));
                figure(2).add_subplot(M,K,1 + m*K + k)
                imshow(z2)
                title("m = " + str(m) +"k = " + str(k))
                print "plot"
        draw()
        show()
    Mnorm = 0
    for j in xrange(0,J):
        Norm = norm(m_H[:,j])
        m_H[:,j] /= Norm
        m_A[j,:] *= Norm
        Mnorm += Norm/J
    mu_M *= Mnorm
    sigma_M *= Mnorm
    sigma_M = sqrt(sigma_M)
    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            pyhrf.verbose(3, 'Compute contrasts ...')
            nrls_conds = dict([(str(cn), m_A[:,ic]) \
                                   for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
            pyhrf.verbose(3, 'Done contrasts computing.')
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "Beta = " + str(Beta)
    return m_A,m_H, q_Z, q_Q, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],Par_im,HRFDict


def Main_vbjpde3(graph,Y,Onsets,Thrf,Pmask,TR,dt,K=2,I=1,sigmaH=0.1,v_h=0.1,beta=0.8,beta_Q=0.8,NitMax = -1,NitMin = 1,estimateBeta=True):
    """
    JPDE model

    `Pmask` - Initialization of the parcellization mask
    `HRFDict` - Mean HRFs of parcels
    `HRFDictCovar` - HRF precision matrices (one by class of HRF)
    `I` - Number of HRF classes
    """
    pyhrf.verbose(1,"Fast EM with C extension for JPDE started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-8
    D = int(numpy.ceil(Thrf/dt))
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
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1

    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    RR = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    # ----------- activation class --------------#
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    # ----------- activation class --------------#

    # --------- Parcellisation class ------------#
    q_Q = numpy.zeros((I,J),dtype=numpy.float64)
    for j in xrange(0,J):
        ind = Pmask[j]
        q_Q[ind,j] = 1

    Q_tilde = q_Q.copy()
    q_Q0 = q_Q.copy()
    # --------- Parcellisation class ------------#
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    m_H = numpy.ones((D,J),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    mu_bar = numpy.zeros((D),dtype=numpy.float64)
    Sigma_bar = numpy.zeros((D,D),dtype=numpy.float64)
    Sum_Sigma_h_k = numpy.zeros((D),dtype=numpy.float64)
    Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    for j in xrange(0,J):
        m_H[:,j] = numpy.array(m_h).astype(numpy.float64)
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
                for k in xrange(0,K):
                    m_A[j,m] += normal(mu_M[m,k], sigma_M[m,k])*Z_tilde[m,k,j]
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_Q = 1
    cA = []
    cH = []
    cZ = []
    cQ = []
    Ndrift = L.shape[0]
    #CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    #CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    t1 = time.time()
    ni = 0
    m_H1 = m_H.copy()
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    q_Q1 = q_Q.copy()
    zerosI = numpy.zeros((I),dtype=float)
    zerosK = numpy.zeros((K),dtype=float)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosDD = 0 * numpy.identity(D)#numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosP = numpy.zeros((P.shape[0]),dtype=float)
    zerosMM = 0 * numpy.identity(M)
    HRFDict = []
    HRFDictCovar = []
    for i in range(0,I):
        HRFDict.append(numpy.array(m_h).astype(numpy.float64))
        tmp = numpy.identity((D)) / sigmaH
        HRFDictCovar.append(tmp)
    while ((Crit_H > Thresh) or (Crit_Z > Thresh) or (Crit_A > Thresh) or (ni < NitMin)) and (ni < NitMax) :
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(2, "E A step ...")
        Sigma_A, m_A = expectation_AP2(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        pyhrf.verbose(2, "E H step ...")
        Sigma_H, m_H = expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone)
        m_H[0,:] = 0
        m_H[-1,:] = 0
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:,:] = m_H[:,:]
        pyhrf.verbose(2, "E Z step ...")
        UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        pyhrf.verbose(2, "E Q step ...")
        q_Q,Q_tilde = expectation_Q(m_H,Sigma_H,sigma_M,beta_Q,Q_tilde,HRFDict,HRFDictCovar,q_Q,graph,J,I,zerosI)
        for i in range(0,I):
            HRFDict[i] = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],RR/v_h)
        pyhrf.verbose(2,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_P2(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        L = maximization_LP(Y,m_A,X,m_H,L,P,zerosP)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(2,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(2,"End estimating beta")
            pyhrf.verbose.printNdarray(2, Beta)
        pyhrf.verbose(2,"M sigma noise step ...")
        sigma_epsilone = maximization_sigma_noiseP(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
        ni += 1
    t2 = time.time()
    CompTime = t2 - t1
    Pmask = classify(q_Q)
    #Pmask = reshape(Pmask,(l,l))
    Mnorm = 0
    for j in xrange(0,J):
        Norm = norm(m_H[:,j])
        m_H[:,j] /= Norm
        m_A[j,:] *= Norm
        Mnorm += Norm/J
    mu_M *= Mnorm
    sigma_M *= Mnorm
    sigma_M = sqrt(sigma_M)
    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    #if computeContrast:
	#if len(contrasts) >0:
            #pyhrf.verbose(3, 'Compute contrasts ...')
	    #nrls_conds = dict([(str(cn), m_A[:,ic]) \
                                   #for ic,cn in enumerate(condition_names)] )
	    #n = 0
	    #for cname in contrasts:
		##------------ contrasts ------------#
		#contrast_expr = AExpr(contrasts[cname], **nrls_conds)
		#contrast_expr.check()
		#contrast = contrast_expr.evaluate()
		#CONTRAST[:,n] = contrast
		##------------ contrasts ------------#

		##------------ variance -------------#
		#ContrastCoef = numpy.zeros(M,dtype=float)
		#ind_conds0 = {}
		#for m in xrange(0,M):
		    #ind_conds0[condition_names[m]] = 0.0
		#for m in xrange(0,M):
		    #ind_conds = ind_conds0.copy()
		    #ind_conds[condition_names[m]] = 1.0
		    #ContrastCoef[m] = eval(contrasts[cname],ind_conds)
		#ActiveContrasts = (ContrastCoef != 0) * numpy.ones(M,dtype=float)
		#CovM = numpy.ones(M,dtype=float)
		#for j in xrange(0,J):
		    #CovM = numpy.ones(M,dtype=float)
		    #for m in xrange(0,M):
			#if ActiveContrasts[m]:
			    #CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
			    #for m2 in xrange(0,M):
				#if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
				    #CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
				    #CovM[m2] = 0
				    #CovM[m] = 0
		##------------ variance -------------#
		#n +=1
            #pyhrf.verbose(3, 'Done contrasts computing.')
	##+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "Beta = " + str(Beta)
    #return m_A,m_H, q_Z, q_Q, Pmask, HRFDict, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:]
    return m_A,m_H, q_Z, q_Q, Pmask, HRFDict, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, cA[2:],cH[2:],cZ[2:]




#def Main_vbjpde(NRL0,graph,Y,Onsets,Thrf,K,I,Pmask,Pact,HRFDict,HRFDictCovar,TR,dt,v_h=0.1,beta=0.8,beta_Q=0.8,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False):
    #pyhrf.verbose(1,"Fast EM with C extension for JPDE started ...")
    #if NitMax < 0:
        #NitMax = 100
    #gamma = 7.5
    #gradientStep = 0.003
    #MaxItGrad = 200
    #Thresh = 1e-8
    #autocorr = 600
    #rho = 200
    ##v_h = 0.1
    #D = int(numpy.ceil(Thrf/dt))
    #M = len(Onsets)
    #N = Y.shape[0]
    #J = Y.shape[1]
    #l = int(sqrt(J))

    #Par0 = reshape(Pmask,(l,l))
    ##HRFDictCovar.append(0.001*numpy.identity((D)))
    ##hh = 0*hrf_triang(D)
    ####multivariate_normal(tmp[1], inv(tmp2)/sigmaH, 1)[0]
    ##HRFDict.append(hh)
    ##HRFDict[I] = hh
    ##I += 1
    ##R = inv(array(tridiag(D,autocorr,rho)))
    #condition_names = []
    #maxNeighbours = max([len(nl) for nl in graph])
    #neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    #neighboursIndexes -= 1
    #for i in xrange(J):
        #neighboursIndexes[i,:len(graph[i])] = graph[i]
    #sigma_epsilone = numpy.ones(J)
    #X = OrderedDict([])
    #for condition,Ons in Onsets.iteritems():
        #X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        #condition_names += [condition]
    #XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    #nc = 0
    #for condition,Ons in Onsets.iteritems():
        #XX[nc,:,:] = X[condition]
        #nc += 1

    #mu_M = numpy.zeros((M,K,I),dtype=numpy.float64)
    #sigma_M = 0.5 * numpy.ones((M,K,I),dtype=numpy.float64)
    #sigma_M0 = 0.5*numpy.ones((M,K,I),dtype=numpy.float64)
    #for k in xrange(1,K):
        #mu_M[:,k] = 2.0
    #order = 2
    #D2 = buildFiniteDiffMatrix(order,D)
    #RR = numpy.dot(D2,D2) / pow(dt,2*order)
    ##R = 0.01*RR + 20*numpy.identity((D),dtype=float)
    #Gamma = numpy.identity(N)
    ## ----------- activation class --------------#
    #q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    #q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    #q_Z[:,1,:] = 1
    #Z_tilde = q_Z.copy()
    ## ----------- activation class --------------#
    ## --------- Parcellisation class ------------#
    #q_Q = numpy.zeros((I,J),dtype=numpy.float64)
    ##q_Q1 = numpy.zeros((I,J),dtype=numpy.float64)
    ##q_Q[1,:] = 1
    ##for j in xrange(0,J):
	##ind = random.randint(0,I-1)
	##q_Q[0,j] = 1
	###q_Q[ind,j] = 1

    ###q_Q0 = q_Q.copy()
    #import os.path as op
    #from scipy.misc import fromimage
    #import Image
    #dataDir = '/home/chaari/Logiciels/pyhrf/pyhrf-free/trunk/python/pyhrf/datafiles/'
    #fn = op.join(dataDir,'simu_par20.png')#simu_labels_carre1.png
    ##Pmask = fromimage(Image.open(fn)) + 1
    ##Pmask = reshape(Pmask,(J))
    #fn = op.join(dataDir,'simu_par20_c1.png')#simu_labels_carre1.png
    #label = fromimage(Image.open(fn))
    #q_Z[0,1,:] = reshape(label,(J))
    #q_Z[0,0,:] = 1- reshape(label,(J))

    #fn = op.join(dataDir,'simu_par20_c2.png')#simu_labels_carre1.png
    #label = fromimage(Image.open(fn))
    #q_Z[1,1,:] = reshape(label,(J))
    #q_Z[1,0,:] = 1- reshape(label,(J))
    #Z_tilde = q_Z.copy()
    #Pact = reshape(Pact,(J))
    ##print Pact.max()
    #for j in xrange(0,J):
	#ind = Pmask[j] - 1
	#q_Q[ind,j] = 1#*Pact[j]

    ##fn = op.join(dataDir,'simu_labels_p50_2.png')#simu_labels_carre1.png
    ##label = fromimage(Image.open(fn))
    ##q_Q[0,:] = reshape(label,(J))
    ##q_Q[1,:] = 1-reshape(label,(J))
    ##q_Q[2,:] = 1 - q_Q[0,:]-q_Q[1,:]
    ##print q_Q[0,:] * Pact
    ##print q_Q[0,:].shape,Pact.shape

    ##raw_input('')
    ##q_Q[0,:] *= Pact
    #Q_tilde = q_Q.copy()
    #q_Q0 = q_Q.copy()
    #ion()
    ##for i in xrange(0,I):
	##z1 = q_Q[i,:]
	##z2 = reshape(z1,(l,l))
	##figure(3)
	##subplot(2,2,1+i)
	##imshow(z2)
	##title(str(i))
	##colorbar()
	##hold(False)
    ##show()
    ##raw_input('')

    ## --------- Parcellisation class ------------#
    #Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    #m_A = numpy.zeros((J,M),dtype=numpy.float64)
    #m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    ##timeAxis = numpy.arange(0, Thrf, dt)
    ##TT,m_h = genBezierHRF(timeAxis=timeAxis, pic=[5,1], picw=3)
    #m_H = numpy.ones((D,J),dtype=numpy.float64)
    ##m_H1 = numpy.ones((D,J),dtype=numpy.float64)
    #Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    #mu_bar = numpy.zeros((D),dtype=numpy.float64)
    #Sigma_bar = numpy.zeros((D,D),dtype=numpy.float64)
    #Sum_Sigma_h_k = numpy.zeros((D),dtype=numpy.float64)
    ##for i in xrange(0,I):
        ###tmp = numpy.array(HRFDictCovar[i])#.astype(numpy.float64)
        ###tmp2 = numpy.array(HRFDict[i])#.astype(numpy.float64)
        ###tmp = HRFDictCovar[i]
        ###tmp2 = HRFDict[i]
        ##Sum_Sigma_h_k += numpy.dot(HRFDictCovar[i],HRFDict[i])
    #Sigma_H = numpy.ones((D,D,J),dtype=numpy.float64)
    #for j in xrange(0,J):
	##m_H[:,j] = numpy.array(m_h).astype(numpy.float64)
	##m_H[:,j] = numpy.array(HRFDict[1]).astype(numpy.float64)
	#m_H[:,j] = hrf_triang(D)
	##m_H1[:,j] = numpy.array(HRFDict[0]).astype(numpy.float64)
	##Sigma_H[:,:,j] = numpy.array(HRFDictCovar[0]).astype(numpy.float64)
	#Sigma_A[:,:,j] = 0.01*numpy.identity(M)
	#for i in xrange(0,I):
	    #for m in xrange(0,M):
		#for k in xrange(0,K):
		    #m_A[j,m] += normal(mu_M[m,k,i], numpy.sqrt(sigma_M[m,k,i]))*Z_tilde[m,k,j]
    #m_A = NRL0.copy()
    #Beta = beta * numpy.ones((M),dtype=numpy.float64)
    #P = PolyMat( N , 4 , TR)
    #L = polyFit(Y, TR, 4,P)
    #PL = numpy.dot(P,L)
    #y_tilde = Y - PL
    #Crit_H = 1
    #Crit_Z = 1
    #Crit_A = 1
    #Crit_Q = 1
    #cA = []
    #cH = []
    #cZ = []
    #cQ = []
    #Ndrift = L.shape[0]
    #CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    #CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    #Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    #XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    #m1 = 0
    #for k1 in X: # Loop over the M conditions
	#m2 = 0
	#for k2 in X:
	    #Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
	    #m2 += 1
	#XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
	#m1 += 1
    #t1 = time.time()
    #ni = 0
    #m_H1 = m_H.copy()
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #q_Q1 = q_Q.copy()
    #zerosI = numpy.zeros((I),dtype=float)
    #zerosK = numpy.zeros((K),dtype=float)
    #zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    #zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    #zerosDD = 0 * numpy.identity(D)#numpy.zeros((D,D),dtype=numpy.float64)
    #zerosD = numpy.zeros((D),dtype=numpy.float64)
    #zerosP = numpy.zeros((P.shape[0]),dtype=float)
    ##Par0 = numpy.zeros((l,l),dtype=numpy.float64)
    #zerosMM = 0 * numpy.identity(M)
    ##for i in range(0,I):
	##z1 = q_Q0[i,:]
	##Par0 += reshape(labelling(z1.copy(),i+1),(l,l))
    ##Par0 = reshape(Pmask,(l,l))
    #HRFDict0 = []
    #Par = numpy.zeros((l,l),dtype=numpy.float64)
    #for i in xrange(0,I):
	##print i
	#tmp = HRFDict[i]
	#HRFDict0.append(tmp)
	#z1 = q_Q[i,:]
	#z2 = reshape(z1,(l,l))
	#Par += i*reshape(thresholding(z1.copy(),0.5),(l,l))

    #while ((Crit_H > Thresh) or (Crit_Z > Thresh) or (Crit_A > Thresh) or (ni < NitMin)) and (ni < NitMax) :

	##print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
	#pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        #pyhrf.verbose(2, "E A step ...")
        #Sigma_A, m_A = expectation_AP(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,uint(reshape(Par0-1,J)))
	##m_A = NRL0.copy()
	#DIFF = reshape( m_A - m_A1,(M*J) )
	#Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
	#cA += [Crit_A]
	#m_A1[:,:] = m_A[:,:]
	#pyhrf.verbose(2, "E H step ...")
	#Sigma_H, m_H = expectation_HP(Y,Sigma_A,Sigma_H,m_A,m_H,X,I,q_Q,HRFDictCovar,HRFDict,Gamma,D,J,N,y_tilde,zerosND,sigma_epsilone)
	#m_H[0,:] = 0
        #m_H[-1,:] = 0
        #Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
	#cH += [Crit_H]
	#m_H1[:,:] = m_H[:,:]
        #pyhrf.verbose(2, "E Z step ...")
        ##UtilsC.expectation_Z(Sigma_A,m_A,sigma_M[:,:,0],Beta,Z_tilde,mu_M[:,:,0],q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        ##q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK,uint(reshape(Par0-1,J)))

        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
	#Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
	#cZ += [Crit_Z]
	#q_Z1[:,:,:] = q_Z[:,:,:]
	#pyhrf.verbose(2, "E Q step ...")
	##q_Q,Q_tilde = expectation_Q(m_H,Sigma_H,sigma_M,beta_Q,Q_tilde,HRFDict,HRFDictCovar,q_Q,graph,J,I,zerosI)
	##q_Q = numpy.zeros((I,J),dtype=numpy.float64)
	##for j in xrange(0,J):
	    ##ind = random.randint(0,3)
	    ##q_Q[ind,j] = 1

	#figure(3).clf()
	##figure(4).clf()
	##pyplot.clf()
	##Par = numpy.zeros((l,l),dtype=numpy.float64)
	#Par = numpy.zeros((J),dtype=numpy.float64)
	#for j in range(0,J):
	    #mm = max(q_Q[:,j])
	    #Par[j] = find(q_Q[:,j] == mm)[0]
	#Par = reshape(Par,(l,l))
	#for i in range(0,I):
	    #z1 = q_Q[i,:]
	    #z2 = reshape(z1,(l,l))
	    ##Par += i*reshape(thresholding(z1.copy(),0.5),(l,l))
	    #figure(3)
	    #subplot(1,I,1+i)
	    #imshow(z2,interpolation='nearest')
	    #title('Parcel ' + str(i))
	    #colorbar()
	    #hold(False)
	##figure(6).clf()
	#figure(7).clf()
	##figure(5).clf()
	##figure(5)
	##subplot(1,2,2)
	#Par = classify(q_Q)
	#Par = reshape(Par,(l,l))
	##imshow(Par,interpolation='nearest')
	##title('Estimated Parcel mask')
	##colorbar()
	##subplot(1,2,1)
	##imshow(Par0-1,interpolation='nearest')
	##title('Reference Parcel mask')
	##colorbar()


	##for i in range(0,I):
	    ##hh = mean_HRF(m_H,q_Q[i,:])
	    ###HRFDict[i] = mean_HRF(m_H,q_Q[i,:])
	    ##figure(2)
	    ##subplot(1,I,1+i)
	    ##plot(hh/(norm(hh)+eps))
	    ###plot(hh)

	    ##grid(True)
	    ##hold(True)
	    ##plot(HRFDict[i]/(norm(HRFDict[i]) + eps),'k')
	    ###plot(HRFDict[i],'k')
	    ##legend(("Est","Ref"))
	    ###title('Parcel ' + str(i) + ' -- mse = %.11f' %compute_MSE(hh/norm(hh),HRFDict[i]) )
	    ###title('mse = %.11f' %compute_MSE(hh,HRFDict[i]) )
	    ##hold(False)

	#for m in range(0,M):
	    #hh = mean_HRF(m_H,q_Q[i,:])
	    #z1 = m_A[:,m]
	    #z2 = reshape(z1,(l,l))
	    #figure(7)
	    #subplot(M,1,1 + m)
	    #imshow(z2*norm(hh),interpolation='nearest')
	    ##imshow(z2 * norm(hh),interpolation='nearest')

	    #title("m = " + str(m))
	    #colorbar()
	    #hold(False)
	    ##for k in range(0,K):
		##z1 = q_Z[m,k,:]
		##z2 = reshape(z1,(l,l))
		##figure(6)
		##subplot(M,K,1 + m*K + k)
		##imshow(z2,interpolation='nearest')
		##title("m = " + str(m) +"k = " + str(k))
		##colorbar()
		##hold(False)

	#draw()
	##R = 0.002*RR + 30*numpy.identity((D),dtype=float)
	##R = array(tridiag2(D,96,-40,10))
	##R = array(tridiag2(D,96,-64,16))
	#R = RR
	#R2 = RR
	##R2 = numpy.identity(D)
	##v_h = 0.1
	##v_h = 0.001
	##sigmaH = 0.0000001
	##sigmaH = 0.05

	##v_h = 0
	##for j in xrange(0,J):
	    ##v_h += (numpy.dot(mult(m_H[:,j],m_H[:,j]) + Sigma_H[:,:,j] , R2 )).trace()
	##v_h /= (D*J)
	##v_h = 20.0/100.5
	#print v_h
	#for i in range(0,I):
	    ##HRFDict[i] = mean_HRF(m_H,q_Q[i,:])
	    ##if (i<I):
		##HRFDict[i] = maximization_h_k_prior(m_H,q_Q[i,:],HRFDictCovar[i],R2/v_h)

	    ##HRFDictCovar[i] = R/sigmaH#numpy.identity(D)#
	    ##hh = mean_HRF(m_H,q_Q[i,:])
	    #hh = sum(m_H,1)
	    ##hh = HRFDict[i]
	    ##hh = mean_HRF(m_H,q_Q[i,:])
	    #figure(2)
	    #subplot(1,I,1+i)
	    ##plot(hh)
	    #plot(hh/(norm(hh)+eps))
	    #grid(True)
	    #hold(True)
	    ##plot(hh2/(norm(hh2)+eps),'g')
	    ##hold(True)
	    #plot(HRFDict0[i],'k')
	    ##plot(HRFDict0[i]/(norm(HRFDict0[i]) + eps),'k')
	    #hold(True)
	    #plot(numpy.array(m_h),'y')
	    ##legend(("Est","Est2","Ref","Init"))
	    #legend(("Est","Ref","Init"))
	    #title('mse c1 = %.11f' %compute_MSE(hh,HRFDict0[0]) )
	    #xlabel('mse c2 = %.11f' %compute_MSE(hh,HRFDict0[1]))

	    ##title('mse c1 = %.11f' %compute_MSE(hh/(norm(hh)+eps),HRFDict0[0]) )
	    ##xlabel('mse c2 = %.11f' %compute_MSE(hh/(norm(hh)+eps),HRFDict0[1]))
	    ###xlabel("PV=" + str(max(hh/(norm(hh)+eps))))
	    #hold(False)
	##HRFDictCovar = maximization_sigmaH_P(D,J,I,Sigma_H,R,m_H,HRFDict,HRFDictCovar,q_Q,80)
	##HRFDict = maximization_hk(J,q_Q,m_H,HRFDict,I,zerosD)
	##HRFDict[0] = mean_HRF(m_H,q_Q[0,:])
	##HRFDict[1] = mean_HRF(m_H,q_Q[1,:])

	##HRFDictCovar[0] = maximization_sigmaH_P(D,Sigma_H,R,m_H,HRFDict[0],J)

	##for j in xrange(0,J):
	    ##sigmah = maximization_sigmaH(D,Sigma_H[:,:,j],R,m_H[:,j])
	    ##Sigma_H[:,:,j] = R / sigmah


	##HRFDictCovar,HRFDict1 = maximization_hk_Sigmak(R,J,q_Q,Sigma_H,m_H,HRFDict,HRFDictCovar,I,zerosDD,zerosD)
	#pyhrf.verbose(2,"M (mu,sigma) step ...")
	#mu_M , sigma_M = maximization_mu_sigma_P(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,uint(reshape(Par0-1,J)),I)
	##mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
	#print 'mu_M:', mu_M[:,1,:]
        #print 'sigma_M:', sigma_M[:,1,:]
        #print '------------------------------'
        ##print 'mu_M:', mu_M[:,0,:]
        ##print 'sigma_M:', sigma_M[:,0,:]
	#L = maximization_LP(Y,m_A,X,m_H,L,P,zerosP)
	##UtilsC.maximization_L(Y,m_A,0.5*(mean_HRF(m_H,q_Q[0,:]) + mean_HRF(m_H,q_Q[1,:])),L,P,XX.astype(int32),J,D,M,Ndrift,N)
	##UtilsC.maximization_L(Y,m_A,m_H[1,:],L,P,XX.astype(int32),J,D,M,Ndrift,N)
	#PL = numpy.dot(P,L)
	#y_tilde = Y - PL
	#if estimateBeta:
	    #pyhrf.verbose(2,"estimating beta")
	    #for m in xrange(0,M):
		#Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
	    #pyhrf.verbose(2,"End estimating beta")
	    #pyhrf.verbose.printNdarray(2, Beta)
	#pyhrf.verbose(2,"M sigma noise step ...")

	#sigma_epsilone = maximization_sigma_noiseP(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
	##UtilsC.maximization_sigma_noise(PL,sigma_epsilone,mean_HRF_covar(Sigma_H,q_Q,zerosDD),Y,m_A,0.5*(mean_HRF(m_H,q_Q[0,:]) + mean_HRF(m_H,q_Q[1,:])),Sigma_A,XX.astype(int32),J,D,M,N)
	##UtilsC.maximization_sigma_noise(PL,sigma_epsilone,Sigma_H[:,:,60],Y,m_A,m_H[:,60],Sigma_A,XX.astype(int32),J,D,M,N)

	#ni += 1
    #t2 = time.time()
    #draw()
    #show()
    #CompTime = t2 - t1
    #PLOT = 0
    #if PLOT:
	#for m in range(0,M):
	    #for k in range(0,K):
		#z1 = q_Z[m,k,:];
		#z2 = reshape(z1,(l,l));
		#figure(2).add_subplot(M,K,1 + m*K + k)
		#imshow(z2)
		#title("m = " + str(m) +"k = " + str(k))
		#print "plot"
		##raw_input('')

	##figure(1)
	##plot(cA[1:-1],'r')
	##hold(True)
	##plot(cH[1:-1],'b')
	##hold(True)
	##plot(cZ[1:-1],'k')
	##hold(False)
	##legend( ('CA','CH', 'CZ') )
	##grid(True)
	#draw()
	#show()
    #for j in xrange(0,J):
	#Norm = norm(m_H[:,j])
	#m_H[:,j] /= Norm
	#m_A[j,:] *= Norm
	##m_A[Pmask==i] *= Norm
    #mu_M *= Norm
    #sigma_M *= Norm
    #sigma_M = sqrt(sqrt(sigma_M))
    ##+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    #if computeContrast:
	#if len(contrasts) >0:
            #pyhrf.verbose(3, 'Compute contrasts ...')
	    #nrls_conds = dict([(str(cn), m_A[:,ic]) \
                                   #for ic,cn in enumerate(condition_names)] )
	    #n = 0
	    #for cname in contrasts:
		##------------ contrasts ------------#
		#contrast_expr = AExpr(contrasts[cname], **nrls_conds)
		#contrast_expr.check()
		#contrast = contrast_expr.evaluate()
		#CONTRAST[:,n] = contrast
		##------------ contrasts ------------#

		##------------ variance -------------#
		#ContrastCoef = numpy.zeros(M,dtype=float)
		#ind_conds0 = {}
		#for m in xrange(0,M):
		    #ind_conds0[condition_names[m]] = 0.0
		#for m in xrange(0,M):
		    #ind_conds = ind_conds0.copy()
		    #ind_conds[condition_names[m]] = 1.0
		    #ContrastCoef[m] = eval(contrasts[cname],ind_conds)
		#ActiveContrasts = (ContrastCoef != 0) * numpy.ones(M,dtype=float)
		#CovM = numpy.ones(M,dtype=float)
		#for j in xrange(0,J):
		    #CovM = numpy.ones(M,dtype=float)
		    #for m in xrange(0,M):
			#if ActiveContrasts[m]:
			    #CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
			    #for m2 in xrange(0,M):
				#if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
				    #CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
				    #CovM[m2] = 0
				    #CovM[m] = 0
		##------------ variance -------------#
		#n +=1
            #pyhrf.verbose(3, 'Done contrasts computing.')
	##+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
    #pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    #pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    ##print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    #if pyhrf.verbose.verbosity > 1:
        #print 'mu_M:', mu_M
        #print 'sigma_M:', sigma_M
        #print "Beta = " + str(Beta)
    #return m_A,m_H, q_Z, q_Q, sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:]


def Main_vbjde_Extension_Wavelet(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False):
    from pywt import dwt,idwt,dwt2,idwt2
    pyhrf.verbose(1,"Fast EM with C extension started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gamma_h = 900
    gradientStep = 0.005
    MaxItGrad = 150
    Thresh = 5e-4
    D = int(numpy.ceil(Thrf/dt))
    Dw = int(numpy.ceil(0.5*Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    condition_names = []
    #-----------------------------------------------------------------------#
    # put neighbour lists into a 2D numpy array so that it will be easily
    # passed to C-code
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    #R = numpy.dot(D2,D2) / pow(dt,2*order)
    #print R.shape
    R = numpy.identity(D)
    sigmaG_D = 0.00001
    sigmaG_A = 0.00001
    for i in xrange(0,Dw):
        R[i,i] = 1.0/sigmaG_A
        R[i+Dw,i+Dw] = 1.0/sigmaG_D
    #print R
    Gamma = numpy.identity(N)
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    cA = []
    cH = []
    cZ = []
    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    zerosDD = numpy.zeros((D,D),dtype=float)
    zerosD = numpy.zeros((D),dtype=float)
    zerosND = numpy.zeros((N,D),dtype=float)
    cA = numpy.zeros((Dw),dtype=float)
    cD = numpy.zeros((Dw),dtype=float)
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
    t1 = time.time()
    for ni in xrange(0,NitMin):
        print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        #UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        Sigma_H, m_H = expectation_H_Wavelet(Dw,Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
        #Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
        cA[:] = m_G[0:Dw]
        cD[:] = m_G[Dw:]
        #m_H = idwt(cA, cD, 'db8','per')
        m_H = m_G
        figure(2)
        plot(cA)
        figure(3)
        plot(cD)

        figure(1)
        plot(m_H,'r')
        hold(False)
        draw()
        show()

        m_H[0] = 0
        m_H[-1] = 0
        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            #sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
            #sigmaH = (mult(m_G,m_G) + Sigma_H).trace()
            #sigmaG_A = (numpy.dot(mult(m_G[0:Dw],m_G[0:Dw]) + Sigma_H[0:Dw,0:Dw] , R[0:Dw,0:Dw] )).trace()/Dw
            #sigmaG_D = (numpy.dot(mult(m_G[Dw:],m_G[Dw:]) + Sigma_H[Dw:,Dw:] , R[Dw:,Dw:] )).trace()/Dw
            #sigmaH = (mult(m_H,m_H) + Sigma_H).trace()
            #sigmaH /= D
            sigmaG_D = 1
            sigmaG_A = 1
            for i in xrange(0,Dw):
                R[i,i] = 1.0/sigmaG_A
                R[i+Dw,i+Dw] = 1.0/sigmaG_D
            print sigmaG_A,sigmaG_D
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise(PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
    m_H1[:] = m_H[:]
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    #print "------------------------------ Iteration n° " + str(ni+2) + " ------------------------------"
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    Crit_A = sum(DIFF) / len(find(DIFF != 0))
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    #UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
    Sigma_H, m_H = expectation_H_Wavelet(Dw,Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)

    cA = m_H[0:Dw-1]
    cD = m_H[Dw:-1]
    m_H = idwt(cA, cD, 'db8')

    m_H[0] = 0
    m_H[-1] = 0
    Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = sum(DIFF) / len(find(DIFF != 0))
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
        sigmaH /= D
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
    UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    UtilsC.maximization_sigma_noise(PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
    ni += 2
    if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax)):# or (ni < 50):
            #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            Crit_A = sum(DIFF) / len(find(DIFF != 0))
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            cA = m_H[0:Dw-1]
            cD = m_H[Dw:-1]
            m_H = idwt(cA, cD, 'db2')

            m_H[0] = 0
            m_H[-1] = 0
            Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = sum(DIFF) / len(find(DIFF != 0))
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
                sigmaH /= D
            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            UtilsC.maximization_L(Y,m_A,m_H,L,P,XX.astype(int32),J,D,M,Ndrift,N)
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            UtilsC.maximization_sigma_noise(PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
            ni +=1
    t2 = time.time()
    
    if PLOT:
        figure(1)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(False)
        legend( ('CA','CH', 'CZ') )
        grid(True)
        draw()
        show()
    CompTime = t2 - t1
    Norm = norm(m_H)
    m_H /= Norm
    m_A *= Norm
    mu_M *= Norm
    sigma_M *= Norm
    sigma_M = sqrt(sigma_M)
    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print "Beta = " + str(Beta)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR



def Main_vbjde_Extension_NoDrifts(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True):
    pyhrf.verbose(1,"Fast EM with C extension started ...")
    print "Fast EM with C extension started without Drifts..."
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    #D = int(numpy.ceil(Thrf/dt))

    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    P = PolyMat( N , 4 , TR)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    Gamma = Gamma - numpy.dot(P,P.transpose())
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    #for k in xrange(0,K):
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    #Norm1 = norm(m_h)
    #print 'Norm1 =',Norm1

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    PL = numpy.zeros((N,J),dtype=numpy.float64)
    y_tilde = Y
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1

    cA = []
    cH = []
    cZ = []

    cTime = []
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

    StimulusInducedSignal = numpy.zeros((N,J), dtype=numpy.float64)

    t1 = time.time()
    for ni in xrange(0,NitMin):
        #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            ## 1) Changing sigmaH (Multiplying with |h|^2):
            #print 'sigmaH before =',sigmaH
            #print 'HRF Norm =', norm(m_H)
            #sigmaH = 0.0001 * (norm(m_H)**2)
            #print 'sigmaH after =',sigmaH
            ## 2) Changing sigmaH (Multiplying with |h|):
            #print 'sigmaH before =',sigmaH
            #print 'HRF Norm =', norm(m_H)
            #sigmaH = 0.0001 * norm(m_H)
            #print 'sigmaH after =',sigmaH
            UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            print 'HRF Norm =', norm(m_H)
            ## 3) Normalizing HRF each n=50 iterations
            #if (ni+1)%50 ==0:
                ##print 'HRF Normalisation at iteration =', ni+1
                #Norm = norm(m_H)
                ##print 'HRF Norm =', Norm
                #m_H /= Norm
                #Sigma_H /= Norm**2
                #m_A *= Norm
                #Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
            ##4) Normalizing HRF each iteration:
            #print 'HRF Normalisation at iteration =', ni+1
            #Norm = norm(m_H)
            #print 'HRF Norm =', Norm
            #m_H /= Norm
            #Sigma_H /= Norm**2
            #m_A *= Norm
            #Sigma_A *= Norm**2
            #mu_M *= Norm
            #sigma_M *= Norm**2
            if PLOT and ni >= 50:
                figure(1)
                plot(m_H)
                hold(True)
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        y_tilde = Y
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            #print Beta
            #print 'Beta =',Beta
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
        #print 'sigma_noise =',sigma_epsilone
        t02 = time.time()
        cTime += [t02-t1]
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #print "------------------------------ Iteration n° " + str(ni+2) + " ------------------------------"
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    #Crit_A = sum(DIFF) / len(find(DIFF != 0))
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    if estimateHRF:
      pyhrf.verbose(3, "E H step ...")
      ## 1) Changing sigmaH (Multiplying with |h|^2):
      #print 'sigmaH before =',sigmaH
      #print 'HRF Norm =', norm(m_H)
      #sigmaH = 0.0001 * (norm(m_H)**2)
      #print 'sigmaH after =',sigmaH
      ## 2) Changing sigmaH (Multiplying with |h|):
      #print 'sigmaH before =',sigmaH
      #print 'HRF Norm =', norm(m_H)
      #sigmaH = 0.0001 * norm(m_H)
      #print 'sigmaH after =',sigmaH
      UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
      m_H[0] = 0
      m_H[-1] = 0
      #print 'HRF Norm =', norm(m_H)
      ## 3) Normalizing HRF each n=50 iterations
      #if (ni+1)%50 ==0:
        ##print 'HRF Normalisation at iteration =', ni+1
        #Norm = norm(m_H)
        ##print 'HRF Norm =', Norm
        #m_H /= Norm
        #Sigma_H /= Norm**2
        #m_A *= Norm
        #Sigma_A *= Norm**2
        #mu_M *= Norm
        #sigma_M *= Norm**2
      ## 4) Normalizing HRF each iteration:
      #print 'HRF Normalisation at iteration =', ni+1
      #Norm = norm(m_H)
      #print 'HRF Norm =', Norm
      #m_H /= Norm
      #Sigma_H /= Norm**2
      #m_A *= Norm
      #Sigma_A *= Norm**2
      #mu_M *= Norm
      #sigma_M *= Norm**2
      if PLOT and ni >= 50:
        plot(m_H)
        hold(True)
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        if gamma_h > 0:
            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
        else:
            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
    y_tilde = Y
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        #print Beta
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2

    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax)):# or (ni < 50):
    if (Crit_H > Thresh) or (Crit_A > Thresh):
        while ( ( (Crit_H > Thresh) or (Crit_A > Thresh) ) and (ni < NitMax) ):# or (ni < 50):
    # Replacing Crit on H by Crit on H_norm:
    #if (Crit_A > Thresh) or (norm(m_H) > 0.1):
        #while ( ( (Crit_A > Thresh) or (norm(m_H) > 0.1) ) and (ni < NitMax) ):# or (ni < 50):
            #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A(q_Z,mu_M,sigma_M,PL,sigma_epsilone,Gamma,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            #Crit_A = sum(DIFF) / len(find(DIFF != 0))
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            if estimateHRF:
                ## 1) Changing sigmaH (Multiplying with |h|^2):
                #print 'sigmaH before =',sigmaH
                #print 'HRF Norm =', norm(m_H)
                #sigmaH = 0.0001 * (norm(m_H)**2)
                #print 'sigmaH after =',sigmaH
                ## 2) Changing sigmaH (Multiplying with |h|):
                #print 'sigmaH before =',sigmaH
                #print 'HRF Norm =', norm(m_H)
                #sigmaH = 0.0001 * norm(m_H)
                #print 'sigmaH after =',sigmaH
                UtilsC.expectation_H(XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,Y,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                #print 'HRF Norm =', norm(m_H)
                ## 3) Normalizing HRF each n=50 iterations:
                #if (ni+1)%50 ==0:
                    ##print 'HRF Normalisation at iteration =', ni+1
                    #Norm = norm(m_H)
                    ##print 'HRF Norm =', Norm
                    #m_H /= Norm
                    #Sigma_H /= Norm**2
                    #m_A *= Norm
                    #Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2
                ## 4) Normalizing HRF each iteration:
                #print 'HRF Normalisation at iteration =', ni+1
                #Norm = norm(m_H)
                #print 'HRF Norm =', Norm
                #m_H /= Norm
                #Sigma_H /= Norm**2
                #m_A *= Norm
                #Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
                if PLOT and ni >= 50:
                    plot(m_H)
                    hold(True)
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            UtilsC.expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            y_tilde = Y
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                #print Beta
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            UtilsC.maximization_sigma_noise(Gamma,PL,sigma_epsilone,Sigma_H,Y,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
            ni +=1
            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()


    if PLOT:
        savefig('./HRF_Iter.png')
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(False)
        legend( ('CA','CH', 'CZ') )
        grid(True)
        savefig('./Crit.png')
        #draw()
        #show()
    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    #print 'HRF Norm =', Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)
    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    #print "Beta = " + str(Beta)
    StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cTime[2:], cTimeMean,Sigma_A, StimulusInducedSignal

def Main_vbjde_NoDrifts_ParsiMod_Python(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True):
    pyhrf.verbose(1,"No fast EM without Drifts and with Parsimonious Model started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.005
    MaxItGrad = 120
    Thresh = 1e-8

    tau1 = 1.
    tau2 = 15.5
    #print 'type tau1 =',type(tau1)
    #print 'type tau2 =',type(tau2)
    V = 10

    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)
    zerosV = numpy.zeros(V)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    #XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    #nc = 0
    #for condition,Ons in Onsets.iteritems():
        #XX[nc,:,:] = X[condition]
        #nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    P = PolyMat( N , 4 , TR)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    Gamma = Gamma - numpy.dot(P,P.transpose())
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    #for k in xrange(0,K):
    q_Z[:,1,:] = 1

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64) #####
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64) #####
    p_Wtilde[:,1] = 1 #####

    #Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                #m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j] #####

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)


    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    #PL = numpy.zeros((N,J),dtype=numpy.float64)
    y_tilde = Y
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1

    Crit_W = 1 #####

    cA = []
    cH = []
    cZ = []

    cW = [] #####

    cTime = []

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    #Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    #XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    #m1 = 0
    #for k1 in X: # Loop over the M conditions
        #m2 = 0
        #for k2 in X:
            #Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            #m2 += 1
        #XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        #m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        Sigma_A, m_A = expectation_A_ParsiMod(Sigma_H,m_H,m_A,X,Gamma,sigma_M,q_Z,mu_M,J,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,p_Wtilde,M)
        #print 'AFTER ...'
        #print 'm_A[:,:]=',m_A[:,:]
        #print 'Sigma_A[:,:,:]=',Sigma_A[:,:,:]
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            Sigma_H, m_H = expectation_H_ParsiMod(Sigma_A,m_A,X,Gamma,R,sigmaH,J,y_tilde,zerosND,sigma_epsilone,scale,zerosD,p_Wtilde)
            m_H[0] = 0
            m_H[-1] = 0

        pyhrf.verbose(3, "E Z step ...")
        q_Z = expectation_Z_ParsiMod(tau1,tau2,Sigma_A,m_A,J,M,sigma_M,mu_M,V,K,Beta,graph,p_Wtilde,zerosV,zerosK,q_Z)

        pyhrf.verbose(3, "E W step ...")
        p_Wtilde = expectation_W_ParsiMod(tau1,tau2,Sigma_A,m_A,X,Gamma,J,M,y_tilde,sigma_epsilone,sigma_M,mu_M,p_Wtilde,q_Z,zerosK,K,m_H,Sigma_H,zerosJMD)

        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
        y_tilde = Y
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            print 'Beta =',Beta
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        sigma_epsilone = maximization_sigma_noise_ParsiMod(Y,X,m_A,m_H,Sigma_H,Sigma_A,sigma_epsilone,zerosMM,N,J,p_Wtilde,Gamma)
    m_H1[:] = m_H[:]
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    #print "------------------------------ Iteration n° " + str(ni+2) + " ------------------------------"
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    Sigma_A, m_A = expectation_A_ParsiMod(Sigma_H,m_H,m_A,X,Gamma,sigma_M,q_Z,mu_M,J,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,p_Wtilde,M)
    #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    #Crit_A = sum(DIFF) / len(find(DIFF != 0))
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    if estimateHRF:
      pyhrf.verbose(3, "E H step ...")
      Sigma_H, m_H = expectation_H_ParsiMod(Sigma_A,m_A,X,Gamma,R,sigmaH,J,y_tilde,zerosND,sigma_epsilone,scale,zerosD,p_Wtilde)
      m_H[0] = 0
      m_H[-1] = 0
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    pyhrf.verbose(3, "E Z step ...")
    q_Z = expectation_Z_ParsiMod(tau1,tau2,Sigma_A,m_A,J,M,sigma_M,mu_M,V,K,Beta,graph,p_Wtilde,zerosV,zerosK,q_Z)
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    pyhrf.verbose(3, "E W step ...")
    p_Wtilde = expectation_W_ParsiMod(tau1,tau2,Sigma_A,m_A,X,Gamma,J,M,y_tilde,sigma_epsilone,sigma_M,mu_M,p_Wtilde,q_Z,zerosK,K,m_H,Sigma_H,zerosJMD)
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = sum(DIFF) / len(find(DIFF != 0))
    DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        if gamma_h > 0:
            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
        else:
            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
    y_tilde = Y
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        #print Beta
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    sigma_epsilone = maximization_sigma_noise_ParsiMod(Y,X,m_A,m_H,Sigma_H,Sigma_A,sigma_epsilone,zerosMM,N,J,p_Wtilde,Gamma)
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
            #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            Sigma_A, m_A = expectation_A_ParsiMod(Sigma_H,m_H,m_A,X,Gamma,sigma_M,q_Z,mu_M,J,y_tilde,Sigma_A,sigma_epsilone,zerosJMD,p_Wtilde,M)
            #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            #Crit_A = sum(DIFF) / len(find(DIFF != 0))
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            if estimateHRF:
                Sigma_H, m_H = expectation_H_ParsiMod(Sigma_A,m_A,X,Gamma,R,sigmaH,J,y_tilde,zerosND,sigma_epsilone,scale,zerosD,p_Wtilde)
                m_H[0] = 0
                m_H[-1] = 0
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            q_Z = expectation_Z_ParsiMod(tau1,tau2,Sigma_A,m_A,J,M,sigma_M,mu_M,V,K,Beta,graph,p_Wtilde,zerosV,zerosK,q_Z)
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            p_Wtilde = expectation_W_ParsiMod(tau1,tau2,Sigma_A,m_A,X,Gamma,J,M,y_tilde,sigma_epsilone,sigma_M,mu_M,p_Wtilde,q_Z,zerosK,K,m_H,Sigma_H,zerosJMD)
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = sum(DIFF) / len(find(DIFF != 0))
            DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
            y_tilde = Y
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                #print Beta
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            sigma_epsilone = maximization_sigma_noise_ParsiMod(Y,X,m_A,m_H,Sigma_H,Sigma_A,sigma_epsilone,zerosMM,N,J,p_Wtilde,Gamma)
            ni +=1
            t02 = time.time()
            cTime += [t02-t1]
    t2 = time.time()

    if PLOT:
        figure(1)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(False)
        legend( ('CA','CH', 'CZ') )
        grid(True)
        draw()
        show()
    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    #print 'Norm =',Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    #print 'mu_M before normalize =',mu_M[:,:]
    mu_M *= Norm
    #print 'mu_M after normalize =',mu_M[:,:]
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)

    #print "Beta = " + str(Beta)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:], p_Wtilde,cTime[2:],cTimeMean,Sigma_A

def Main_vbjde_NoDrifts_ParsiMod_C_1(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,tau1=1.,tau2=0.,S=100,InitVar=0.5,InitMean=2.0):
    pyhrf.verbose(1,"Fast EM for Parsimonious Model (( Definition 1 ---> W-Q)) without Drifts estimation and with C Extension started...")
    
    numpy.random.seed(6537546)
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    MC_mean = numpy.zeros((M,J,S,K),dtype=numpy.float64)

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)
    #zerosV = numpy.zeros(V)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    #sigma_M = InitVar * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
    #sigma_M0 = InitVar * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.1
    sigma_M0[:,1] = 1.0
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
    
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    P = PolyMat( N , 4 , TR)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    #print 'Det_invR =', Det_invR
    Gamma = numpy.identity(N)
    Gamma = Gamma - numpy.dot(P,P.transpose())
    Det_Gamma = numpy.linalg.det(Gamma)
    #print 'Det_Gamma =',Det_Gamma
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    #for k in xrange(0,K):
    q_Z[:,1,:] = 1

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    #Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                #m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)

    #PL = numpy.zeros((N,J),dtype=numpy.float64)
    y_tilde = Y
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1

    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []

    cTime = []
    W_Iter = [[] for m in xrange(M)]
    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            ## 1) Changing sigmaH (Multiplying with |h|^2):
            #print 'sigmaH before =',sigmaH
            #print 'HRF Norm =', norm(m_H)
            #sigmaH = 0.0001 * (norm(m_H)**2)
            #print 'sigmaH after =',sigmaH
            ## 2) Changing sigmaH (Multiplying with |h|):
            #print 'sigmaH before =',sigmaH
            #print 'HRF Norm =', norm(m_H)
            #sigmaH = 0.0001 * norm(m_H)
            #print 'sigmaH after =',sigmaH
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            #print 'HRF Norm =', norm(m_H)
            ## 3) Normalizing HRF each n=50 iterations:
            #if (ni+1)%50 ==0:
                ##print 'HRF Normalisation at iteration =', ni+1
                #Norm = norm(m_H)
                ##print 'HRF Norm =', Norm
                #m_H /= Norm
                #Sigma_H /= Norm**2
                #m_A *= Norm
                #Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
            ## 4) Normalizing HRF each iteration:
            #print 'HRF Normalisation at iteration =', ni+1
            #Norm = norm(m_H)
            #print 'HRF Norm =', Norm
            #m_H /= Norm
            #Sigma_H /= Norm**2
            #m_A *= Norm
            #Sigma_A *= Norm**2
            #mu_M *= Norm
            #sigma_M *= Norm**2
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
            #print 'HXGamma=',HXGamma
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]

        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        pyhrf.verbose(3, "E W step ...")
        print 'Iteration =',ni
        UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
        DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
        
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
       
        for m in xrange(M):
           SUM_q_Z[m] += [sum(q_Z[m,1,:])]
           W_Iter[m] += [p_Wtilde[m,1]]
           mu1[m] += [mu_M[m,1]]
       
        y_tilde = Y
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        t02 = time.time()
        cTime += [t02-t1]

    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    #Crit_A = sum(DIFF) / len(find(DIFF != 0))
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    if estimateHRF:
      pyhrf.verbose(3, "E H step ...")
      ## 1) Changing sigmaH (Multiplying with |h|^2):
      #print 'sigmaH before =',sigmaH
      #print 'HRF Norm =', norm(m_H)
      #sigmaH = 0.0001 * (norm(m_H)**2)
      #print 'sigmaH after =',sigmaH
      ## 2) Changing sigmaH (Multiplying with |h|):
      #print 'sigmaH before =',sigmaH
      #print 'HRF Norm =', norm(m_H)
      #sigmaH = 0.0001 * norm(m_H)
      #print 'sigmaH after =',sigmaH
      UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
      m_H[0] = 0
      m_H[-1] = 0
      #print 'HRF Norm =', norm(m_H)
      ## 3) Normalizing HRF each n=50 iterations:
      #if (ni+1)%50 ==0:
        ##print 'HRF Normalisation at iteration =', ni+1
        #Norm = norm(m_H)
        ##print 'HRF Norm =', Norm
        #m_H /= Norm
        #Sigma_H /= Norm**2
        #m_A *= Norm
        #Sigma_A *= Norm**2
        #mu_M *= Norm
        #sigma_M *= Norm**2
      ## 4) Normalizing HRF each iteration:
      #print 'HRF Normalisation at iteration =', ni+1
      #Norm = norm(m_H)
      #print 'HRF Norm =', Norm
      #m_H /= Norm
      #Sigma_H /= Norm**2
      #m_A *= Norm
      #Sigma_A *= Norm**2
      #mu_M *= Norm
      #sigma_M *= Norm**2
      if PLOT and ni >= 0:
        figure(M+1)
        plot(m_H)
        hold(True)
      #Update HXGamma
      m1 = 0
      for k1 in X: # Loop over the M conditions
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    
    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    pyhrf.verbose(3, "E Z step ...")
    UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    pyhrf.verbose(3, "E W step ...")
    UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
    #print 'p_Wtilde =',p_Wtilde
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = sum(DIFF) / len(find(DIFF != 0))
    DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        if gamma_h > 0:
            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
        else:
            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
    
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
    
    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        W_Iter[m] += [p_Wtilde[m,1]]
        mu1[m] += [mu_M[m,1]]
    
    y_tilde = Y
    
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    
    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if Crit_FreeEnergy > Thresh_FreeEnergy:
        #while ((Crit_FreeEnergy > Thresh_FreeEnergy) and (ni < NitMax)):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            #DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            #Crit_A = sum(DIFF) / len(find(DIFF != 0))
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            if estimateHRF:
                ## 1) Changing sigmaH (Multiplying with |h|^2):
                #print 'sigmaH before =',sigmaH
                #print 'HRF Norm =', norm(m_H)
                #sigmaH = 0.0001 * (norm(m_H)**2)
                #print 'sigmaH after =',sigmaH
                ## 2) Changing sigmaH (Multiplying with |h|):
                #print 'sigmaH before =',sigmaH
                #print 'HRF Norm =', norm(m_H)
                #sigmaH = 0.0001 * norm(m_H)
                #print 'sigmaH after =',sigmaH
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                #print 'HRF Norm =', norm(m_H)
                ## 3) Normalizing HRF each n=50 iterations:
                #if (ni+1)%50 ==0:
                    ##print 'HRF Normalisation at iteration =', ni+1
                    #Norm = norm(m_H)
                    ##print 'HRF Norm =', Norm
                    #m_H /= Norm
                    #Sigma_H /= Norm**2
                    #m_A *= Norm
                    #Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2
                ## 4) Normalizing HRF each iteration:
                #print 'HRF Normalisation at iteration =', ni+1
                #Norm = norm(m_H)
                #print 'HRF Norm =', Norm
                #m_H /= Norm
                #Sigma_H /= Norm**2
                #m_A *= Norm
                #Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
                if PLOT and ni >= 0:
                    figure(1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            #Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            
            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]
            
            UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = sum(DIFF) / len(find(DIFF != 0))
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            #print 'p_Wtilde =',p_Wtilde
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = sum(DIFF) / len(find(DIFF != 0))
            DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
            
            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                W_Iter[m] += [p_Wtilde[m,1]]
                mu1[m] += [mu_M[m,1]]
            
            y_tilde = Y
            
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            
            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
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
        
    #W_Iter_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    W_Iter_array = numpy.zeros((M,ni),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            W_Iter_array[m,i] = W_Iter[m][i]
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
        #for i in xrange(ni-1,NitMax+1):
            #W_Iter_array[m,i] = W_Iter[m][ni-1]
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]      

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        #plot(cW[1:-1],'g')
        #hold(True)
        plot(cAH[1:-1],'g')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        #legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        legend( ('CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        #draw()
        #show()

        figure(3)
        plot(FreeEnergy_Iter)
        savefig('./FreeEnergy.png')

        figure(4)
        for m in xrange(M):
            plot(W_Iter_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') )
        legend( ('m=0','m=1') ) 
        savefig('./W_Iter.png')
        
        figure(5)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./Sum_q_Z_Iter.png')
        
        figure(6)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./mu1_Iter.png')

    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    print 'Norm =',Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print 'p_Wtilde =',p_Wtilde
    
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    #SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) ) # There's no PL in noDrifts version !!
    #SNR /= np.log(10.)
    #print 'SNR parsi =', SNR
    
    return ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:],p_Wtilde,cTime[2:],cTimeMean,Sigma_A,MC_mean, StimulusInducedSignal, FreeEnergyArray

def Main_vbjde_Extension_ParsiMod_C_1(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,tau1=28.,tau2=0.5,S=100,estimateLabels=True,LabelsFilename='labels.nii',InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model (( Definition 1 ---> W-Q)) with C Extension started...")
    print 'Fast EM for Parsimonious Model (( Definition 1 ---> W-Q)) with C Extension started...'
    
    HRF_Normalization = False
    
    numpy.random.seed(6537546)
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    tau2 = int(J * tau2) # This allows the user to give only a percentage of the ROI size
    #tau2 = 24
    print 'tau2 =',tau2
    print 'tau1 =',tau1

    MC_mean = numpy.zeros((M,J,S,K),dtype=numpy.float64)

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.5
    sigma_M[:,1] = 0.6
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.5
    sigma_M0[:,1] = 0.6
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    #print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)
    #print 'Det_Gamma =',Det_Gamma
    
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

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)

    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1

    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []

    cTime = []
    W_Iter = [[] for m in xrange(M)]
    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]
    h_norm = []

    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        print '------------------------------ Iteration n° ',str(ni+1),'------------------------------'
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            h_norm += [norm(m_H)]
            #4) Normalizing HRF each iteration:
            if HRF_Normalization:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2     
                m_A *= Norm
                Sigma_A *= Norm**2

            #print 'HRF Norm =', norm(m_H)
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]

        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]            
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            #print 'Iteration =',ni
            UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            print 'p_Wtilde =',p_Wtilde[:,1]
        DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
                
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
        
        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            W_Iter[m] += [p_Wtilde[m,1]]
            mu1[m] += [mu_M[m,1]]
        
        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
            
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        h_norm += [norm(m_H)]
        #4) Normalizing HRF each iteration:
        if HRF_Normalization:
            Norm = norm(m_H)
            m_H /= Norm
            Sigma_H /= Norm**2
            m_A *= Norm
            Sigma_A *= Norm**2

        #print 'HRF Norm =', norm(m_H)
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    
    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]    
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
    DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
    
    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        W_Iter[m] += [p_Wtilde[m,1]]
        mu1[m] += [mu_M[m,1]]
    
    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
        
    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if Crit_FreeEnergy > Thresh_FreeEnergy:
        #while ((Crit_FreeEnergy > Thresh_FreeEnergy) and (ni < NitMax)):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                h_norm += [norm(m_H)]
                #4) Normalizing HRF each iteration:
                if HRF_Normalization:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2

                #print 'HRF Norm =', norm(m_H)
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            
            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                UtilsC.expectation_Z_ParsiMod_1(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_1(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
            DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))   
                        
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
            
            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                W_Iter[m] += [p_Wtilde[m,1]]
                mu1[m] += [mu_M[m,1]]
            
            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
            
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
                
            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
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

    #W_Iter_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    W_Iter_array = numpy.zeros((M,ni),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    h_norm_array = numpy.zeros((ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            W_Iter_array[m,i] = W_Iter[m][i]
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]
        #for i in xrange(ni-1,NitMax+1):
            #W_Iter_array[m,i] = W_Iter[m][ni-1]
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]

    if PLOT:
        savefig('./HRF_Iter_Parsi1.png')
        hold(False)
        figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        #plot(cW[1:-1],'y')
        #hold(False)
        plot(cAH[1:-1],'g')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        #legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        legend( ('CAH', 'CFE') )
        grid(True)
        savefig('./Crit_Parsi1.png')
        #draw()
        #show()

        figure(3)
        plot(FreeEnergy_Iter)
        savefig('./FreeEnergy_Parsi1.png')
        
        figure(4)
        for m in xrange(M):
            plot(W_Iter_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') )
        legend( ('m=0','m=1') ) 
        savefig('./W_Iter_Parsi1.png')
        
        figure(5)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./Sum_q_Z_Iter_Parsi1.png')
        
        figure(6)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./mu1_Iter_Parsi1.png')
        
        figure(8)
        plot(h_norm_array)
        savefig('./HRF_Norm_Parsi1.png')
        
    Data_save = xndarray(h_norm_array, ['Iteration'])
    Data_save.save('./HRF_Norm_Parsi1.nii')

    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    print 'Norm =',Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "Beta = " + str(Beta)
        print 'p_Wtilde =', p_Wtilde
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    print 'StimulusInducedSignal.sum =',StimulusInducedSignal.sum()
    print 'Y.sum =',Y.sum()
    print 'PL.sum =', PL.sum()
    print 'result =',np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL)
    print 'log (result) =',np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR parsi 1 =', SNR

    return tau2,ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:],p_Wtilde,cTime[2:],cTimeMean,Sigma_A,MC_mean,StimulusInducedSignal,FreeEnergyArray

def Main_vbjde_Extension_ParsiMod_C_1_MeanLabels(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,tau1=28.,tau2=0.5,S=100,estimateLabels=True,LabelsFilename='labels.nii',InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model (( Definition 1 ---> W-Q *Mean Labels*)) with C Extension started...")
    print 'Fast EM for Parsimonious Model (( Definition 1 ---> W-Q *Mean Labels*)) with C Extension started...'
    
    HRF_Normalization = False
    
    numpy.random.seed(6537546)
    
    #p0 = 0.001
    #c = numpy.log((1.-p0)/p0)
    #tau1 = c/tau2
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    MC_mean = numpy.zeros((M,J,S,K),dtype=numpy.float64)

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.1
    sigma_M0[:,1] = 1.0
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    #print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)
    #print 'Det_Gamma =',Det_Gamma
    
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

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)

    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1

    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []

    cTime = []
    W_Iter = [[] for m in xrange(M)]
    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]

    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            #4) Normalizing HRF each iteration:
            if HRF_Normalization:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
            #print 'HRF Norm =', norm(m_H)
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]

        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            UtilsC.expectation_Z_ParsiMod_1_MeanLabels(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]            
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            #print 'Iteration =',ni
            UtilsC.expectation_W_ParsiMod_1_MeanLabels(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            #print 'p_Wtilde =',p_Wtilde
        DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
                
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
        
        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            W_Iter[m] += [p_Wtilde[m,1]]
            mu1[m] += [mu_M[m,1]]
        
        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
            
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        #4) Normalizing HRF each iteration:
    if HRF_Normalization:
            Norm = norm(m_H)
            m_H /= Norm
            Sigma_H /= Norm**2
            #sigmaH /= Norm**2
            m_A *= Norm
            Sigma_A *= Norm**2
            #mu_M *= Norm
            #sigma_M *= Norm**2
        #print 'HRF Norm =', norm(m_H)
    if PLOT and ni >= 0:
        figure(M+1)
        plot(m_H)
        hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    
    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z_ParsiMod_1_MeanLabels(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]    
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_1_MeanLabels(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
    DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
    
    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        W_Iter[m] += [p_Wtilde[m,1]]
        mu1[m] += [mu_M[m,1]]
    
    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
        
    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if Crit_FreeEnergy > Thresh_FreeEnergy:
        #while ((Crit_FreeEnergy > Thresh_FreeEnergy) and (ni < NitMax)):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                #4) Normalizing HRF each iteration:
                if HRF_Normalization:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2
                #print 'HRF Norm =', norm(m_H)
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            
            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                UtilsC.expectation_Z_ParsiMod_1_MeanLabels(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,S,K,maxNeighbours,tau1,tau2,MC_mean)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_1_MeanLabels(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
            DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))   
                        
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
            
            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                W_Iter[m] += [p_Wtilde[m,1]]
                mu1[m] += [mu_M[m,1]]
            
            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
            
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    Beta[m] = UtilsC.maximization_beta_CB(beta,q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
                
            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod1")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
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

    #W_Iter_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    W_Iter_array = numpy.zeros((M,ni),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            W_Iter_array[m,i] = W_Iter[m][i]
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
        #for i in xrange(ni-1,NitMax+1):
            #W_Iter_array[m,i] = W_Iter[m][ni-1]
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        #plot(cW[1:-1],'y')
        #hold(False)
        plot(cAH[1:-1],'g')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        #legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        legend( ('CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        #draw()
        #show()

        figure(3)
        plot(FreeEnergy_Iter)
        savefig('./FreeEnergy.png')
        
        figure(4)
        for m in xrange(M):
            plot(W_Iter_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') )
        legend( ('m=0','m=1') ) 
        savefig('./W_Iter.png')
        
        figure(5)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./Sum_q_Z_Iter.png')
        
        figure(6)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        legend( ('m=0','m=1') ) 
        savefig('./mu1_Iter.png')

    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    print 'Norm =',Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "Beta = " + str(Beta)
        print 'p_Wtilde =', p_Wtilde
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    print 'StimulusInducedSignal.sum =',StimulusInducedSignal.sum()
    print 'Y.sum =',Y.sum()
    print 'PL.sum =', PL.sum()
    print 'result =',np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL)
    print 'log (result) =',np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR parsi =', SNR

    return tau2,ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:],p_Wtilde,cTime[2:],cTimeMean,Sigma_A,MC_mean,StimulusInducedSignal,FreeEnergyArray


def Main_vbjde_NoDrifts_ParsiMod_C_2(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,tau1=1.,tau2=0.,S=100,alpha=0.5,alpha_0=0.5):
    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 2)) without Drifts and with C Extension started...")

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    MC_mean = numpy.zeros((M,J,S,K),dtype=numpy.float64)

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)
    #zerosV = numpy.zeros(V)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    P = PolyMat( N , 4 , TR)
    R = numpy.dot(D2,D2) / pow(dt,2*order)

    Gamma = numpy.identity(N)
    Gamma = Gamma - numpy.dot(P,P.transpose())

    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    #for k in xrange(0,K):
    q_Z[:,1,:] = 1

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    #Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                #m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)

    #PL = numpy.zeros((N,J),dtype=numpy.float64)
    y_tilde = Y
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1

    Crit_W = 1

    cA = []
    cH = []
    cZ = []

    cW = []

    cTime = []

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            #print 'HRF Norm =', norm(m_H)
            if PLOT and ni >= 0:
                figure(1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
            #print 'HXGamma=',HXGamma
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        pyhrf.verbose(3, "E Z step ...")
        UtilsC.expectation_Z_ParsiMod_2(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours,alpha_0)
        DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_2(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,Beta,neighboursIndexes.astype(int32),J,D,M,N,K,maxNeighbours,alpha,alpha_0)
        #print 'p_Wtilde =',p_Wtilde
        DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
        y_tilde = Y
        #### Computation for Beta is not done yet for this model, Beta is not estimated now
        #if estimateBeta:
            #pyhrf.verbose(3,"estimating beta")
            #for m in xrange(0,M):
                #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            #pyhrf.verbose(3,"End estimating beta")
            #pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]
    ####

    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    if estimateHRF:
      pyhrf.verbose(3, "E H step ...")
      UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
      m_H[0] = 0
      m_H[-1] = 0
      #print 'HRF Norm =', norm(m_H)
      if PLOT and ni >= 0:
        figure(1)
        plot(m_H)
        hold(True)
      #Update HXGamma
      m1 = 0
      for k1 in X: # Loop over the M conditions
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    pyhrf.verbose(3, "E Z step ...")
    UtilsC.expectation_Z_ParsiMod_2(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours,alpha_0)
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    pyhrf.verbose(3, "E W step ...")
    UtilsC.expectation_W_ParsiMod_2(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,Beta,neighboursIndexes.astype(int32),J,D,M,N,K,maxNeighbours,alpha,alpha_0)
    #print 'p_Wtilde =',p_Wtilde
    DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        if gamma_h > 0:
            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
        else:
            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
    y_tilde = Y
    #### Computation for Beta is not done yet for this model, Beta is not estimated now
    #if estimateBeta:
        #pyhrf.verbose(3,"estimating beta")
        #for m in xrange(0,M):
            #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        #pyhrf.verbose(3,"End estimating beta")
        #pyhrf.verbose(3,Beta)
    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
    if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
            #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                #print 'HRF Norm =', norm(m_H)
                if PLOT and ni >= 0:
                    figure(1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            UtilsC.expectation_Z_ParsiMod_2(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours,alpha_0)
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            UtilsC.expectation_W_ParsiMod_2(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,Beta,neighboursIndexes.astype(int32),J,D,M,N,K,maxNeighbours,alpha,alpha_0)
            #print 'p_Wtilde =',p_Wtilde
            DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J)
            y_tilde = Y
            #### Computation for Beta is not done yet for this model, Beta is not estimated now
            #if estimateBeta:
                #pyhrf.verbose(3,"estimating beta")
                #for m in xrange(0,M):
                    #Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                #pyhrf.verbose(3,"End estimating beta")
                #pyhrf.verbose(3,Beta)
            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)
            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    if PLOT:
        savefig('./HRF_Iter.png')
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW') )
        grid(True)
        savefig('./Crit.png')

    CompTime = t2 - t1
    cTimeMean = CompTime/ni
    Norm = norm(m_H)
    print 'Norm =',Norm
    m_H /= Norm
    m_A *= Norm
    Sigma_A *= Norm**2
    mu_M *= Norm
    sigma_M *= Norm**2
    sigma_M = sqrt(sigma_M)

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) >0:
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print 'p_Wtilde =',p_Wtilde
    StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)
    #print "Beta = " + str(Beta)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:], p_Wtilde,cTime[2:],cTimeMean,Sigma_A,MC_mean, StimulusInducedSignal

def Main_vbjde_Extension_ParsiMod_C_3(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateW=True,tau1=1.,tau2=0.1,alpha=3,lam=4,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3 ---> W-mu1, Fixed tau2)) with C extension started ...")

    numpy.random.seed(6537546)
    
    NormFlag = False
    Nb2Norm = 1

    #p0 = 0.001
    #print 'tau2 =',tau2
    #print 'p0 =',p0
    #tau1 = (1./tau2)*numpy.log((1.-p0)/p0)
    
    print 'tau1 =',tau1
    print 'tau2 =',tau2
    #tau1 = 1.

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
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

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    test_tau2 = []
    cTime = []
    W_Iter = [[] for m in xrange(M)]
    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]

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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
    
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1   
    
    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        
        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
        m_A = reshape(val, (J,M))
        
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            # Plotting HRF
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        else:
            if TrueHrfFlag:
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
        
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        test_tau2 += [tau2]
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            W_Iter[m] += [p_Wtilde[m,1]]
            mu1[m] += [mu_M[m,1]]

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
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
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        #FreeEnergy, Total_Entropy, EPtilde, EPtildeLikelihood, EPtildeA, EPtildeH, EPtildeZ, EPtildeW = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    
    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
    m_A = reshape(val, (J,M))
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        # Plotting HRF
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
    
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    test_tau2 += [tau2]
    
    #if estimateMixtParam:
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        W_Iter[m] += [p_Wtilde[m,1]]
        mu1[m] += [mu_M[m,1]]

    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
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

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
            m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                # Plotting HRF
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
                    
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))        
             
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:] 
             
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            test_tau2 += [tau2]
            
            #if estimateMixtParam:
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                W_Iter[m] += [p_Wtilde[m,1]]
                mu1[m] += [mu_M[m,1]]

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

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

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
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

    #W_Iter_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    W_Iter_array = numpy.zeros((M,ni),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            W_Iter_array[m,i] = W_Iter[m][i]
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
        #for i in xrange(ni-1,NitMax+1):
            #W_Iter_array[m,i] = W_Iter[m][ni-1]
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]

    if PLOT:
        savefig('./HRF_Iter_Parsi3.png')
        hold(False)
        figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        #plot(cW[1:-1],'g')
        #hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        #legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        legend( ('CAH', 'CFE') )
        grid(True)
        savefig('./Crit_Parsi3.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy_Parsi3.png')
        
        figure(4)
        plot(test_tau2)
        savefig('./tau2_Parsi3.png')
        
        figure(5)
        for m in xrange(M):
            plot(W_Iter_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') )
        #legend( ('m=0','m=1') ) 
        axis([0, ni, 0, 1.2])
        savefig('./W_Iter_Parsi3.png')
        
        figure(6)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        #legend( ('m=0','m=1') ) 
        savefig('./Sum_q_Z_Iter_Parsi3.png')
        
        figure(7)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        #legend( ('m=0','m=1') ) 
        savefig('./mu1_Iter_Parsi3.png')


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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    pyhrf.verbose(1, "sigma_H = " + str(sigmaH))
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR parsi 3 =', SNR
    
    return ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray, tau2


def MiniVEM_ParsiMod_C_3_tau2(Thrf,TR,dt,beta,Y,K,alpha,lam,c,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,tau1,tau2,Nit,sigmaH,estimateHRF):

    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    Init_tau2 = tau2
    Init_tau1 = tau1
    Init_sigmaH = sigmaH
    
    IM_val = np.array([-5.,5.])
    #IV_val = np.array([0.008,0.016,0.032,0.064,0.128,0.256,0.512)]
    IV_val = np.array([0.01,0.05,0.1,0.5])
    gammah_val = np.array([1000,5000,10000])
    MiniVemStep = IM_val.shape[0]*IV_val.shape[0]*gammah_val.shape[0]
    
    Init_mixt_p_gammah = []
    
    pyhrf.verbose(1,"Number of tested initialisation is %s" %MiniVemStep)
    
    t1_MiniVEM = time.time()
    FE = []
    for Gh in gammah_val:
        for InitVar in IV_val:
            for InitMean in IM_val:
                Init_mixt_p_gammah += [[InitVar,InitMean,Gh]]
                tau2 = Init_tau2
                tau1 = c/tau2
                #print 'tau1 =',tau1,',  tau2 =',tau2
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
                
                p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
                p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
                p_Wtilde[:,1] = 1
                
                #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
                TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
                m_h = m_h[:D]
                m_H = numpy.array(m_h).astype(numpy.float64)
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
                
                gamma_h = Gh
                sigma_M = numpy.ones((M,K),dtype=numpy.float64)
                sigma_M[:,0] = 0.1
                sigma_M[:,1] = 1.0
                mu_M = numpy.zeros((M,K),dtype=numpy.float64)
                for k in xrange(1,K):
                    mu_M[:,k] = InitMean
                Sigma_A = numpy.zeros((M,M,J),numpy.float64)
                for j in xrange(0,J):
                    Sigma_A[:,:,j] = 0.01*numpy.identity(M)    
                m_A = numpy.zeros((J,M),dtype=numpy.float64)
                for j in xrange(0,J):
                    for m in xrange(0,M):
                        for k in xrange(0,K):
                            m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1

                for ni in xrange(0,Nit+1):
                    pyhrf.verbose(3,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
                    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
                    val = reshape(m_A,(M*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
                    m_A = reshape(val, (J,M))

                    if estimateHRF:
                        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                        m_H[0] = 0
                        m_H[-1] = 0
                        #Update HXGamma
                        m1 = 0
                        for k1 in X: # Loop over the M conditions
                            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                            m1 += 1
                    
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                    val = reshape(q_Z,(M*K*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    q_Z = reshape(val, (M,K,J))

                    UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                    val = reshape(p_Wtilde,(M*K))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    p_Wtilde = reshape(val, (M,K))
                    
                    if estimateHRF:
                        if gamma_h > 0:
                            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                        else:
                            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    tau2 = maximization_tau2_ParsiMod3(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
                    # updating tau1
                    tau1 = c/tau2
                    #print 'tau1 =',tau1,',  tau2 =',tau2
                    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)
                    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
                    PL = numpy.dot(P,L)
                    y_tilde = Y - PL
                    for m in xrange(0,M):
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

                FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
                FE += [FreeEnergy]
    
    max_FE, max_FE_ind = maximum(FE)
    InitVar = Init_mixt_p_gammah[max_FE_ind][0]
    InitMean = Init_mixt_p_gammah[max_FE_ind][1]
    Initgamma_h = Init_mixt_p_gammah[max_FE_ind][2]
    
    t2_MiniVEM = time.time()
    pyhrf.verbose(1,"MiniVEM duration is %s" %format_duration(t2_MiniVEM-t1_MiniVEM))
    pyhrf.verbose(1,"Choosed initialisation is : var = %s,  mean = %s,  gamma_h = %s" %(InitVar,InitMean,Initgamma_h))
    
    return InitVar, InitMean, Initgamma_h
    

def Main_vbjde_Extension_ParsiMod_C_3_tau2(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateW=True,tau1=28.,tau2=0.5,alpha=5,lam=7,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3)) with C extension started ...")

    numpy.random.seed(6537546)
    
    Nb2Norm = 1
    NormFlag = False

    p0 = 0.001
    c = numpy.log((1.-p0)/p0)
    tau1 = c/tau2
    Init_sigmaH = sigmaH
    Init_tau2 = tau2

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
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

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    test_tau2 = []
    cFE = []
    cTime = []

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
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
        InitVar, InitMean, gamma_h = MiniVEM_ParsiMod_C_3_tau2(Thrf,TR,dt,beta,Y,K,alpha,lam,c,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,tau1,tau2,NbItMiniVem,sigmaH,estimateHRF)

    tau2 = Init_tau2
    tau1 = c/Init_tau2
    #print 'tau1 =',tau1,',  tau2 =',tau2
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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    
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
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
    
    m1 = 0
    for k1 in X: # Loop over the M conditions
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1
    
    t1 = time.time()

    for ni in xrange(0,NitMin):
        
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        
        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
        m_A = reshape(val, (J,M))

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing HRF each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2
            # Plotting HRF
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        else:
            if TrueHrfFlag:
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
        
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        if estimateW:
            tau2 = maximization_tau2_ParsiMod3(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
            # updating tau1
            tau1 = c/tau2
            test_tau2 += [tau2]
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep) 
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)   
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        #FreeEnergy, Total_Entropy, EPtilde, EPtildeLikelihood, EPtildeA, EPtildeH, EPtildeZ, EPtildeW = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    
    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
    m_A = reshape(val, (J,M))
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        # Plotting HRF
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
 
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
 
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    if estimateW:
        tau2 = maximization_tau2_ParsiMod3(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
        # Updating tau1
        tau1 = c/tau2
        #print 'tau1 =',tau1,',  tau2 =',tau2
        test_tau2 += [tau2]
    
    #if estimateMixtParam:
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    #FreeEnergy, Total_Entropy, EPtilde, EPtildeLikelihood, EPtildeA, EPtildeH, EPtildeZ, EPtildeW = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
            m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                # Plotting HRF
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
                    
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))        
             
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:] 
             
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            if estimateW:
                tau2 = maximization_tau2_ParsiMod3(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
                # Updating tau1
                tau1 = c/tau2
                #print 'tau2 =',tau2,',  tau1 =',tau1
                test_tau2 += [tau2]
            
            #if estimateMixtParam:
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')
        
        figure(4)
        plot(test_tau2)
        savefig('./tau2.png')

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    pyhrf.verbose(1, "sigma_H = " + str(sigmaH))
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray, tau2

def Main_vbjde_Extension_ParsiMod_C_3_tau2_FixedTau1(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateW=True,tau1=1.,tau2=0.1,alpha=3,lam=4,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3 ---> W-mu1, Glob tau2, Fixed tau1)) with C extension started ...")

    numpy.random.seed(6537546)
    
    NormFlag = False
    Nb2Norm = 1

    p0 = 0.001
    #print 'p0 =',p0
    val = (alpha-1.)/lam
    tau1 = (1./val)*numpy.log((1.-p0)/p0)
    #tau1 = 1.
    #print 'tau1 =',tau1

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
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

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    test_tau2 = []
    cTime = []
    W_Iter = [[] for m in xrange(M)]
    SUM_q_Z = [[] for m in xrange(M)]
    mu1 = [[] for m in xrange(M)]
    h_norm = []

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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
    
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1   
    
    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        #print '------------------------------ Iteration n° ',str(ni+1),'------------------------------'
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        
        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
        m_A = reshape(val, (J,M))
        
        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
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
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        else:
            if TrueHrfFlag:
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
        
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            #print 'p_wtilde =',p_Wtilde[:,1]
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        if estimateW:
            tau2 = maximization_tau2_ParsiMod3_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
            test_tau2 += [tau2]
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

        for m in xrange(M):
            SUM_q_Z[m] += [sum(q_Z[m,1,:])]
            W_Iter[m] += [p_Wtilde[m,1]]
            mu1[m] += [mu_M[m,1]]

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
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
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        #FreeEnergy, Total_Entropy, EPtilde, EPtildeLikelihood, EPtildeA, EPtildeH, EPtildeZ, EPtildeW = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    
    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
    m_A = reshape(val, (J,M))
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
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
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
    
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    if estimateW:
        tau2 = maximization_tau2_ParsiMod3_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
        test_tau2 += [tau2]
    
    #if estimateMixtParam:
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

    for m in xrange(M):
        SUM_q_Z[m] += [sum(q_Z[m,1,:])]
        W_Iter[m] += [p_Wtilde[m,1]]
        mu1[m] += [mu_M[m,1]]

    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
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

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
            m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
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
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
                    
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))        
             
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:] 
             
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            if estimateW:
                tau2 = maximization_tau2_ParsiMod3_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
                test_tau2 += [tau2]
            
            #if estimateMixtParam:
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            for m in xrange(M):
                SUM_q_Z[m] += [sum(q_Z[m,1,:])]
                W_Iter[m] += [p_Wtilde[m,1]]
                mu1[m] += [mu_M[m,1]]

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

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

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
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

    #W_Iter_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #SUM_q_Z_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    #mu1_array = numpy.zeros((M,NitMax+1),dtype=numpy.float64)
    W_Iter_array = numpy.zeros((M,ni),dtype=numpy.float64)
    SUM_q_Z_array = numpy.zeros((M,ni),dtype=numpy.float64)
    mu1_array = numpy.zeros((M,ni),dtype=numpy.float64)
    h_norm_array = numpy.zeros((ni),dtype=numpy.float64)
    for m in xrange(M):
        for i in xrange(ni):
            W_Iter_array[m,i] = W_Iter[m][i]
            SUM_q_Z_array[m,i] = SUM_q_Z[m][i]
            mu1_array[m,i] = mu1[m][i]
            h_norm_array[i] = h_norm[i]
        #for i in xrange(ni-1,NitMax+1):
            #W_Iter_array[m,i] = W_Iter[m][ni-1]
            #SUM_q_Z_array[m,i] = SUM_q_Z[m][ni-1]
            #mu1_array[m,i] = mu1[m][ni-1]

    font = {'size'   : 15}
    matplotlib.rc('font', **font)

    if PLOT:
        savefig('./HRF_Iter_Parsi3.png')
        hold(False)
        figure(2)
        #plot(cA[1:-1],'r')
        #hold(True)
        #plot(cH[1:-1],'b')
        #hold(True)
        #plot(cZ[1:-1],'k')
        #hold(True)
        #plot(cW[1:-1],'g')
        #hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        #legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        legend( ('CAH', 'CFE') )
        grid(True)
        savefig('./Crit_Parsi3.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy_Parsi3.png')
        
        figure(4)
        plot(test_tau2)
        savefig('./tau2_Parsi3.png')
        
        figure(5)
        for m in xrange(M):
            plot(W_Iter_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') )
        #legend( ('m=0','m=1') ) 
        #legend( ('Calcul','Phrase','Clic','Damier'),loc='upper left')
        axis([0, ni, 0, 1.2])
        savefig('./W_Iter_Parsi3.png')
        
        figure(6)
        for m in xrange(M):
            plot(SUM_q_Z_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        #legend( ('m=0','m=1') ) 
        #legend( ('Calcul','Phrase','Clic','Damier'),loc='upper left')
        savefig('./Sum_q_Z_Iter_Parsi3.png')
        
        figure(7)
        for m in xrange(M):
            plot(mu1_array[m])
            hold(True)
        hold(False)
        #legend( ('m=0','m=1', 'm=2', 'm=3') ) 
        #legend( ('m=0','m=1') ) 
        #legend( ('Calcul','Phrase','Clic','Damier'),loc='upper left')
        savefig('./mu1_Iter_Parsi3.png')
        
        figure(8)
        plot(h_norm_array)
        savefig('./HRF_Norm_Parsi3.png')
        
        Data_save = xndarray(h_norm_array, ['Iteration'])
        Data_save.save('./HRF_Norm_Parsi3.nii')

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    pyhrf.verbose(1, "sigma_H = " + str(sigmaH))
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    SNR = 20 * np.log( np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL) )
    SNR /= np.log(10.)
    print 'SNR parsi 3 =', SNR
    
    return ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray, tau2


def MiniVEM_ParsiMod_C_3_tau2_Cond(Thrf,TR,dt,beta,Y,K,alpha,lam,c,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,Nit,sigmaH,estimateHRF):

    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    Init_sigmaH = sigmaH
    
    IM_val = np.array([-5.,5.])
    IV_val = np.array([0.008,0.016,0.032,0.064,0.128,0.256,0.512])
    #IV_val = np.array([0.01,0.05,0.1,0.5])
    gammah_val = np.array([1000])
    MiniVemStep = IM_val.shape[0]*IV_val.shape[0]*gammah_val.shape[0]
    
    Init_mixt_p_gammah = []
    
    pyhrf.verbose(1,"Number of tested initialisation is %s" %MiniVemStep)
    
    t1_MiniVEM = time.time()
    FE = []
    for Gh in gammah_val:
        for InitVar in IV_val:
            for InitMean in IM_val:
                Init_mixt_p_gammah += [[InitVar,InitMean,Gh]]
                tau2 = 0.1*numpy.ones(M,dtype=numpy.float64)
                tau1 = c/tau2
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
                
                p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
                p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
                p_Wtilde[:,1] = 1
                
                #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
                TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
                m_h = m_h[:D]
                m_H = numpy.array(m_h).astype(numpy.float64)
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
                
                gamma_h = Gh
                sigma_M = numpy.ones((M,K),dtype=numpy.float64)
                sigma_M[:,0] = 0.1
                sigma_M[:,1] = 1.0
                mu_M = numpy.zeros((M,K),dtype=numpy.float64)
                for k in xrange(1,K):
                    mu_M[:,k] = InitMean
                Sigma_A = numpy.zeros((M,M,J),numpy.float64)
                for j in xrange(0,J):
                    Sigma_A[:,:,j] = 0.01*numpy.identity(M)    
                m_A = numpy.zeros((J,M),dtype=numpy.float64)
                for j in xrange(0,J):
                    for m in xrange(0,M):
                        for k in xrange(0,K):
                            m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1

                for ni in xrange(0,Nit+1):
                    pyhrf.verbose(3,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
                    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
                    val = reshape(m_A,(M*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
                    m_A = reshape(val, (J,M))

                    if estimateHRF:
                        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                        m_H[0] = 0
                        m_H[-1] = 0
                        #Update HXGamma
                        m1 = 0
                        for k1 in X: # Loop over the M conditions
                            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                            m1 += 1
                    
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                    val = reshape(q_Z,(M*K*J))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    q_Z = reshape(val, (M,K,J))

                    UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                    val = reshape(p_Wtilde,(M*K))
                    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
                    p_Wtilde = reshape(val, (M,K))
                    
                    if estimateHRF:
                        if gamma_h > 0:
                            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                        else:
                            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    tau2 = maximization_tau2_ParsiMod3_Cond(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
                    # updating tau1
                    tau1 = c/tau2
                    
                    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni)
                    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
                    PL = numpy.dot(P,L)
                    y_tilde = Y - PL
                    for m in xrange(0,M):
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

                FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
                FE += [FreeEnergy]
    
    max_FE, max_FE_ind = maximum(FE)
    InitVar = Init_mixt_p_gammah[max_FE_ind][0]
    InitMean = Init_mixt_p_gammah[max_FE_ind][1]
    Initgamma_h = Init_mixt_p_gammah[max_FE_ind][2]
    
    t2_MiniVEM = time.time()
    pyhrf.verbose(1,"MiniVEM duration is %s" %format_duration(t2_MiniVEM-t1_MiniVEM))
    pyhrf.verbose(1,"Choosed initialisation is : var = %s,  mean = %s,  gamma_h = %s" %(InitVar,InitMean,Initgamma_h))
    
    return InitVar, InitMean, Initgamma_h

def Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateW=True,alpha=5,lam=7,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3 with tau2/condition)) with C extension started ...")

    numpy.random.seed(6537546)

    NormFlag = False
    Nb2Norm = 1

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    p0 = 0.001
    c = numpy.log((1.-p0)/p0)
    
    Init_sigmaH = sigmaH

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

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    cTime = []

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
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
        InitVar, InitMean, gamma_h = MiniVEM_ParsiMod_C_3_tau2_Cond(Thrf,TR,dt,beta,Y,K,alpha,lam,c,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,NbItMiniVem,sigmaH,estimateHRF)

    tau2 = 0.1*numpy.ones(M,dtype=numpy.float64)
    tau1 = c/tau2
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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    
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
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
    
    m1 = 0
    for k1 in X: # Loop over the M conditions
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1
    
    t1 = time.time()

    for ni in xrange(0,NitMin):

        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
        m_A = reshape(val, (J,M))

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            # Plotting HRF
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        else:
            if TrueHrfFlag:
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
        
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        tau2 = maximization_tau2_ParsiMod3_Cond(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
        # updating tau1
        tau1 = c/tau2
        
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    
    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
    m_A = reshape(val, (J,M))
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        # Plotting HRF
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
    
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    tau2 = maximization_tau2_ParsiMod3_Cond(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
    # Updating tau1
    tau1 = c/tau2
    
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
            m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                # Plotting HRF
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
                    
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))        
             
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            tau2 = maximization_tau2_ParsiMod3_Cond(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,c)
            # Updating tau1
            tau1 = c/tau2

            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    pyhrf.verbose(1, "sigma_H = " + str(sigmaH))
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray, tau2

def Main_vbjde_Extension_ParsiMod_C_3_tau2_Cond_FixedTau1(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,TrueHrfFlag=False,HrfFilename='hrf.nii',estimateW=True,alpha=5,lam=7,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0,MiniVEMFlag=False,NbItMiniVem=5):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3 with tau2/condition)) with C extension started ...")

    numpy.random.seed(6537546)

    NormFlag = False
    Nb2Norm = 1

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    p0 = 0.001
    c = numpy.log((1.-p0)/p0)
    
    Init_sigmaH = sigmaH

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

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    cTime = []
    
    test_W = [[] for m in xrange(M)]
    test_tau2 = [[] for m in xrange(M)]
    #test_mu1 =  [[] for m in xrange(M)]

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
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
        InitVar, InitMean, gamma_h = MiniVEM_ParsiMod_C_3_tau2_Cond(Thrf,TR,dt,beta,Y,K,alpha,lam,c,gamma,gradientStep,MaxItGrad,D,M,N,J,S,maxNeighbours,neighboursIndexes,XX,X,R,Det_invR,Gamma,Det_Gamma,scale,Q_barnCond,XGamma,NbItMiniVem,sigmaH,estimateHRF)

    tau2 = 0.1*numpy.ones(M,dtype=numpy.float64)
    tau1 = c/tau2
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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1
    
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
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A        
    
    m1 = 0
    for k1 in X: # Loop over the M conditions
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1
    
    t1 = time.time()

    for ni in xrange(0,NitMin):

        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

        val = reshape(m_A,(M*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
        #m_A = reshape(val, (J,M))

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            # Plotting HRF
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        else:
            if TrueHrfFlag:
                TrueVal, head = read_volume(HrfFilename)
                m_H = TrueVal
        
        DIFF = reshape( m_A - m_A1,(M*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        #q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        tau2 = maximization_tau2_ParsiMod3_Cond_FixedTau1(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
        
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)
        
        for m in xrange(M):
            test_W[m] += [p_Wtilde[m,1]]
            test_tau2[m] += [tau2[m]]
            test_mu1[m] += [mu_M[m,1]]

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    
    val = reshape(m_A,(M*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
    #m_A = reshape(val, (J,M))
    
    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        # Plotting HRF
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    else:
        if TrueHrfFlag:
            TrueVal, head = read_volume(HrfFilename)
            m_H = TrueVal
    
    DIFF = reshape( m_A - m_A1,(M*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    #q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    #p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    tau2 = maximization_tau2_ParsiMod3_Cond_FixedTau1(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
    
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

    for m in xrange(M):
        test_W[m] += [p_Wtilde[m,1]]
        test_tau2[m] += [tau2[m]]
        #test_mu1[m] += [mu_M[m,1]]

    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            
            val = reshape(m_A,(M*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            val[ find((val>=-1e-50) & (val<0.0)) ] = 0.0
            #m_A = reshape(val, (J,M))
            
            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                # Plotting HRF
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            else:
                if TrueHrfFlag:
                    TrueVal, head = read_volume(HrfFilename)
                    m_H = TrueVal
            
            DIFF = reshape( m_A - m_A1,(M*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
                    
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            #q_Z = reshape(val, (M,K,J))        
             
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_3_Cond(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                    
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            #p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            tau2 = maximization_tau2_ParsiMod3_Cond_FixedTau1(tau2,q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)

            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3_Cond(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            for m in xrange(M):
                test_W[m] += [p_Wtilde[m,1]]
                test_tau2[m] += [tau2[m]]
                #test_mu1[m] += [mu_M[m,1]]

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy_Cond(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]



    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')

        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')

    for m in xrange(M):
        figure(4+m)
        plot(test_W[m])
        savefig('./W_%s.png' %m)
        figure(4+M+m)
        plot(test_tau2[m])
        savefig('./tau2_%s.png' %m)

    #for m in xrange(M):
        #figure(4+2*M+m)
        #plot(test_mu1[m])
        #savefig('./mu1_ParsiMod_Cond_%s.png' %m)

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    pyhrf.verbose(1, "sigma_H = " + str(sigmaH))
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    
    return ni,m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray, tau2


def Main_vbjde_Extension_ParsiMod_C_4(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,tau1=28.,tau2=0.5,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 4)) with C extension started ...")
    
    numpy.random.seed(6537546)
    
    NormFlag = False
    Nb2Norm = 1    
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)

    condition_names = []

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.1
    sigma_M0[:,1] = 1.0
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)
    print 'Det_Gamma =',Det_Gamma
    
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
    
    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check #########################
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    
    test_W = [[] for m in xrange(M)]
    test_dKL = [[] for m in xrange(M)]
    test_mu2 = [[] for m in xrange(M)]
    
    cTime = []

    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    print 'InitVar =',InitVar
    print 'InitMean =',InitMean

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        ## Computing KL Distance:
        #m0 = 0.0
        #for m in xrange(M):
            #v0 = sigma_M[m,0]
            #v1 = sigma_M[m,1]
            #m1 = mu_M[m,1]
            #d = 0.5 * (m1-m0)**2 * (1./v1 + 1./v0) + ( (v1-v0)**2 )/( 2. * v1 * v0 )
            #print 'm =',m,',   mu1 =',m1,',     v0 =',v0,',     v1 =',v1,',    D =',d
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
            
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            print 'p_Wtilde =',p_Wtilde
            for m in xrange(M):
                test_W[m] += [p_Wtilde[m,1]]
                dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                test_dKL[m] += [dKL_val]
                test_mu2[m] += [mu_M[m,1]**2]
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        for t in xrange(0,1):
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)
        
        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep) 
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]

    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
            
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]        
            
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
        for m in xrange(M):
            test_W[m] += [p_Wtilde[m,1]]
            dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
            test_dKL[m] += [dKL_val]
            test_mu2[m] += [mu_M[m,1]**2]
            
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]

    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    #if estimateMixtParam:
    for t in xrange(0,1):
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)
    
    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]

            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))
            
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
                for m in xrange(M):
                    test_W[m] += [p_Wtilde[m,1]]
                    dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                    test_dKL[m] += [dKL_val]
                    test_mu2[m] += [mu_M[m,1]**2]
                    
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            #if estimateMixtParam:
            for t in xrange(0,1):
                mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)
            
            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')
        
        for m in xrange(M):
            figure(6+m)
            plot(test_W[m])
            savefig('./W_%s.png' %m)
            figure(6+M+m)
            plot(test_dKL[m])
            savefig('./dKL_%s.png' %m)
            figure(6+M+M+m)
            plot(test_mu2[m])
            savefig('./mu²_%s.png' %m)

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    #print 'StimIndSign mean =', StimulusInducedSignal.mean()
    
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:], p_Wtilde,cTime[2:], cTimeMean, Sigma_A, StimulusInducedSignal, FreeEnergyArray

def Main_vbjde_Extension_ParsiMod_C_4_tau2(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,tau1=28.,tau2=0.5,alpha=5,lam=7,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 4)) with C extension started ...")
        
    p0 = 0.001
    c = numpy.log((1.-p0)/p0)
    tau1 = c/tau2
    
    NormFlag = True
    Nb2Norm = 1
    
    numpy.random.seed(6537546)
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)

    condition_names = []

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.1
    sigma_M0[:,1] = 1.0
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)
    print 'Det_Gamma =',Det_Gamma
    
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

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check #########################
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    
    test_W = [[] for m in xrange(M)]
    test_dKL = [[] for m in xrange(M)]
    test_mu2 = [[] for m in xrange(M)]
    test_v0 = [[] for m in xrange(M)]
    test_v1 = [[] for m in xrange(M)]
    test_tau2 = []
    
    cTime = []

    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    print 'InitVar =',InitVar
    print 'InitMean =',InitMean

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        ## Computing KL Distance:
        #m0 = 0.0
        #for m in xrange(M):
            #v0 = sigma_M[m,0]
            #v1 = sigma_M[m,1]
            #m1 = mu_M[m,1]
            #d = 0.5 * (m1-m0)**2 * (1./v1 + 1./v0) + ( (v1-v0)**2 )/( 2. * v1 * v0 )
            #print 'm =',m,',   mu1 =',m1,',     v0 =',v0,',     v1 =',v1,',    D =',d
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
            
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            #print 'p_Wtilde =',p_Wtilde
            for m in xrange(M):
                test_W[m] += [p_Wtilde[m,1]]
                dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                test_dKL[m] += [dKL_val]
                test_mu2[m] += [mu_M[m,1]**2]
                test_v0[m] +=[sigma_M[m,0]]
                test_v1[m] +=[sigma_M[m,1]]
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        if estimateW:
            tau2 = maximization_tau2_ParsiMod4(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,p0)
            # Updating tau1
            tau1 = c/tau2
            #print 'tau2 =',tau2,',  tau1 =',tau1
            test_tau2 += [tau2]
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        for t in xrange(0,1):
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep) 
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]

    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
        for m in xrange(M):
            test_W[m] += [p_Wtilde[m,1]]
            dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
            test_dKL[m] += [dKL_val]
            test_mu2[m] += [mu_M[m,1]**2]
            test_v0[m] +=[sigma_M[m,0]]
            test_v1[m] +=[sigma_M[m,1]]
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    if estimateW:
        tau2 = maximization_tau2_ParsiMod4(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,p0)
        # Updating tau1
        tau1 = c/tau2
        #print 'tau2 =',tau2,',  tau1 =',tau1
        test_tau2 += [tau2]
    
    #if estimateMixtParam:
    for t in xrange(0,1):
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)
    
    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]

            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))
            
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
                for m in xrange(M):
                    test_W[m] += [p_Wtilde[m,1]]
                    dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                    test_dKL[m] += [dKL_val]
                    test_mu2[m] += [mu_M[m,1]**2]
                    test_v0[m] +=[sigma_M[m,0]]
                    test_v1[m] +=[sigma_M[m,1]]
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            if estimateW:
                tau2 = maximization_tau2_ParsiMod4(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,p0)
                # Updating tau1
                tau1 = c/tau2
                #print 'tau2 =',tau2,',  tau1 =',tau1
                test_tau2 += [tau2]
            
            #if estimateMixtParam:
            for t in xrange(0,1):
                mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')
        
        figure(4)
        plot(test_tau2)
        savefig('./tau2.png')
        
        for m in xrange(M):
            figure(6+m)
            plot(test_W[m])
            savefig('./W_%s.png' %m)
            figure(6+M+m)
            plot(test_dKL[m])
            savefig('./dKL_%s.png' %m)
            figure(6+M+M+m)
            plot(test_mu2[m])
            savefig('./mu²_%s.png' %m)
            
        for m in xrange(M):
            figure(6+M+M+M+m)
            plot(test_v0[m],'r')
            hold(True)
            plot(test_v1[m],'b')
            hold(False)
            savefig('./v_%s.png' %m)

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    #print 'StimIndSign mean =', StimulusInducedSignal.mean()
    
    return m_A,m_H,q_Z,sigma_epsilone,mu_M,sigma_M,Beta,L,PL,CONTRAST,CONTRASTVAR,cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:],p_Wtilde,cTime[2:],cTimeMean,Sigma_A,StimulusInducedSignal,FreeEnergyArray,tau2

def Main_vbjde_Extension_ParsiMod_C_4_tau2_FixedTau1(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,tau1=28.,tau2=0.5,alpha=5,lam=7,S=100,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 4)) with C extension started ...")
        
    p0 = 0.001
    c = numpy.log((1.-p0)/p0)
    tau1 = c/tau2
    
    NormFlag = False
    Nb2Norm = 1
    
    numpy.random.seed(6537546)
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)

    condition_names = []

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M[:,0] = 0.1
    sigma_M[:,1] = 1.0
    sigma_M0 = numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0[:,0] = 0.1
    sigma_M0[:,1] = 1.0
    
    for k in xrange(1,K):
        mu_M[:,k] = InitMean
        
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Det_Gamma = numpy.linalg.det(Gamma)
    print 'Det_Gamma =',Det_Gamma
    
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

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check #########################
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    AH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []
    
    test_W = [[] for m in xrange(M)]
    test_dKL = [[] for m in xrange(M)]
    test_mu2 = [[] for m in xrange(M)]
    test_v0 = [[] for m in xrange(M)]
    test_v1 = [[] for m in xrange(M)]
    test_tau2 = []
    
    cTime = []

    Ndrift = L.shape[0]
    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    print 'InitVar =',InitVar
    print 'InitMean =',InitMean

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        ## Computing KL Distance:
        #m0 = 0.0
        #for m in xrange(M):
            #v0 = sigma_M[m,0]
            #v1 = sigma_M[m,1]
            #m1 = mu_M[m,1]
            #d = 0.5 * (m1-m0)**2 * (1./v1 + 1./v0) + ( (v1-v0)**2 )/( 2. * v1 * v0 )
            #print 'm =',m,',   mu1 =',m1,',     v0 =',v0,',     v1 =',v1,',    D =',d
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2 
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        else:
            pyhrf.verbose(3, "Using True Z ...")
            TrueZ = read_volume(LabelsFilename)
            for m in xrange(M):
                q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                q_Z[m,0,:] = 1 - q_Z[m,1,:]
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]
            
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            #print 'p_Wtilde =',p_Wtilde
            for m in xrange(M):
                test_W[m] += [p_Wtilde[m,1]]
                dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                test_dKL[m] += [dKL_val]
                test_mu2[m] += [mu_M[m,1]**2]
                test_v0[m] +=[sigma_M[m,0]]
                test_v1[m] +=[sigma_M[m,1]]
        
        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]
        
        if estimateHRF:
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
        
        if estimateW:
            tau2 = maximization_tau2_ParsiMod4_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
            test_tau2 += [tau2]
        
        #if estimateMixtParam:    
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)

        UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep) 
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]
        
        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    ####

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]

    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        # Normalizing H at each Nb2Norm iterations:
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2 
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    else:
        pyhrf.verbose(3, "Using True Z ...")
        TrueZ = read_volume(LabelsFilename)
        for m in xrange(M):
            q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
            q_Z[m,0,:] = 1 - q_Z[m,1,:]
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
        #print 'p_Wtilde =',p_Wtilde
        for m in xrange(M):
            test_W[m] += [p_Wtilde[m,1]]
            dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
            test_dKL[m] += [dKL_val]
            test_mu2[m] += [mu_M[m,1]**2]
            test_v0[m] +=[sigma_M[m,0]]
            test_v1[m] +=[sigma_M[m,1]]
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    if estimateW:
        tau2 = maximization_tau2_ParsiMod4_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
        test_tau2 += [tau2]
    
    #if estimateMixtParam:
    mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)
    
    UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]
    
    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_AH > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_AH > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]

            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                # Normalizing H at each Nb2Norm iterations:
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2 
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if estimateLabels:
                if MFapprox:
                    UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
                if not MFapprox:
                    UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            else:
                pyhrf.verbose(3, "Using True Z ...")
                TrueZ = read_volume(LabelsFilename)
                for m in xrange(M):
                    q_Z[m,1,:] = reshape(TrueZ[0][:,:,:,m],J)
                    q_Z[m,0,:] = 1 - q_Z[m,1,:]
            
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))
            
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_4(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
                #print 'p_Wtilde =',p_Wtilde
                for m in xrange(M):
                    test_W[m] += [p_Wtilde[m,1]]
                    dKL_val = 0.5 * (mu_M[m,1]**2) * (1./sigma_M[m,1] + 1./sigma_M[m,0]) + ( (sigma_M[m,1] - sigma_M[m,0])**2 )/( 2. * sigma_M[m,1] * sigma_M[m,0] )
                    test_dKL[m] += [dKL_val]
                    test_mu2[m] += [mu_M[m,1]**2]
                    test_v0[m] +=[sigma_M[m,0]]
                    test_v1[m] +=[sigma_M[m,1]]
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
            
            if estimateW:
                tau2 = maximization_tau2_ParsiMod4_FixedTau1(q_Z,p_Wtilde,mu_M,sigma_M,M,alpha,lam,tau1)
                test_tau2 += [tau2]
            
            #if estimateMixtParam:
            mu_M , sigma_M = maximization_mu_sigma_ParsiMod4(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2)

            UtilsC.maximization_L_ParsiMod(Y,m_A,m_H,L,P,XX.astype(int32),p_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod4")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        hold(False)
        figure(2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')
        
        figure(4)
        plot(test_tau2)
        savefig('./tau2.png')
        
        for m in xrange(M):
            figure(6+m)
            plot(test_W[m])
            savefig('./W_%s.png' %m)
            figure(6+M+m)
            plot(test_dKL[m])
            savefig('./dKL_%s.png' %m)
            figure(6+M+M+m)
            plot(test_mu2[m])
            savefig('./mu²_%s.png' %m)
            
        for m in xrange(M):
            figure(6+M+M+M+m)
            plot(test_v0[m],'r')
            hold(True)
            plot(test_v1[m],'b')
            hold(False)
            savefig('./v_%s.png' %m)

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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)
    
    w = np.zeros(M,dtype=int)
    for m in xrange(M):
        if p_Wtilde[m,1] > 0.5:
            w[m] = 1   
    StimulusInducedSignal = computeParsiFit(w, m_H, m_A, X, J, N)
    #print 'StimIndSign mean =', StimulusInducedSignal.mean()
    
    return m_A,m_H,q_Z,sigma_epsilone,mu_M,sigma_M,Beta,L,PL,CONTRAST,CONTRASTVAR,cA[2:],cH[2:],cZ[2:],cW[2:],cAH[2:],p_Wtilde,cTime[2:],cTimeMean,Sigma_A,StimulusInducedSignal,FreeEnergyArray,tau2


def Main_vbjde_Extension_ParsiMod_C_RVM(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.05,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,estimateW=True,estimateLabels=True,LabelsFilename='labels.nii',MFapprox=False,estimateMixtParam=True,InitVar=0.5,InitMean=2.0):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((RVM)) with C extension started ...")
    
    numpy.random.seed(6537546)
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5

    #D = int(numpy.ceil(Thrf/dt))  ##################################
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
    
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AWH = 1
    AW = numpy.zeros((J,M),dtype=numpy.float64)
    AWH = numpy.zeros((J,M,D),dtype=numpy.float64)
    AWH1 = numpy.zeros((J,M,D),dtype=numpy.float64)
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAWH = []
    FreeEnergy_Iter = []
    cFE = []
    cTime = []
    
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

    if estimateW:
        m_Wtilde = numpy.ones((M),dtype=numpy.float64)
        m_Wtilde1 = numpy.ones((M),dtype=numpy.float64)
        V_Wtilde = numpy.identity(M)
        alpha_RVM = 0.1*numpy.ones((M),dtype=numpy.float64)
        # with zeros we have a noninformative prior
        k_RVM = 3. #1.
        lam_RVM = 0.05 #0.2
    else:
        m_Wtilde = numpy.ones((M),dtype=numpy.float64)
        m_Wtilde1 = numpy.ones((M),dtype=numpy.float64)
        V_Wtilde = numpy.zeros((M,M),dtype=numpy.float64)
    
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check #########################
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
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]
    m_A1 = m_A

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod_RVM(m_Wtilde,V_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod_RVM(m_Wtilde,V_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0       
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
        
        if estimateLabels:
            pyhrf.verbose(3, "E Z step ...")
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
        
        if estimateW:
            pyhrf.verbose(3, "E W step ...")
            UtilsC.expectation_W_ParsiMod_RVM(m_Wtilde,V_Wtilde,alpha_RVM,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N)
            print 'm_W =',m_Wtilde,',   V_W =',V_Wtilde,',      alpha_RVM =',alpha_RVM
        for cond in xrange(M):
            AW[:,m] = m_A[:,m]*m_Wtilde[m]
        for d in xrange(0,D):
            AWH[:,:,d] = AW[:,:]*m_H[d]
        DIFF = reshape( AWH - AWH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AWH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AWH1,(M*J*D)) ))**2
        cAWH += [Crit_AWH]
        AWH1[:,:,:] = AWH[:,:,:]
        
        if estimateW:
            alpha_RVM = maximization_alphaRVM(k_RVM,lam_RVM,m_Wtilde,V_Wtilde,M,alpha_RVM)
        
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
        
        UtilsC.maximization_L_ParsiMod_RVM(Y,m_A,m_H,L,P,XX.astype(int32),m_Wtilde,J,D,M,Ndrift,N)
        
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose.printNdarray(3, Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod_RVM(m_Wtilde,V_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy_RVM(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,m_Wtilde,V_Wtilde,alpha_RVM,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K)
        if ni > 0:
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        t02 = time.time()
        cTime += [t02-t1]

    pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod_RVM(m_Wtilde,V_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K) 

    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod_RVM(m_Wtilde,V_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    
    if estimateLabels:
        pyhrf.verbose(3, "E Z step ...")
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
    
    if estimateW:
        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_RVM(m_Wtilde,V_Wtilde,alpha_RVM,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N)
    
    for cond in xrange(M):
        AW[:,m] = m_A[:,m]*m_Wtilde[m]
    for d in xrange(0,D):
        AWH[:,:,d] = AW[:,:]*m_H[d]
    DIFF = reshape( AWH - AWH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AWH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AWH1,(M*J*D)) ))**2
    cAWH += [Crit_AWH]
    AWH1[:,:,:] = AWH[:,:,:]
    
    if estimateW:
        alpha_RVM = maximization_alphaRVM(k_RVM,lam_RVM,m_Wtilde,V_Wtilde,M,alpha_RVM)
    
    if estimateHRF:
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))
    
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
    
    UtilsC.maximization_L_ParsiMod_RVM(Y,m_A,m_H,L,P,XX.astype(int32),m_Wtilde,J,D,M,Ndrift,N)
    
    PL = numpy.dot(P,L)
    y_tilde = Y - PL

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose.printNdarray(3, Beta)

    UtilsC.maximization_sigma_noise_ParsiMod_RVM(m_Wtilde,V_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy_RVM(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,m_Wtilde,V_Wtilde,alpha_RVM,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K)
    Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2

    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AWH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AWH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(1,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod_RVM(m_Wtilde,V_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)

            if estimateHRF:
                UtilsC.expectation_H_ParsiMod_RVM(m_Wtilde,V_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                if PLOT and ni >= 0:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1

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
            
            if estimateW:
                UtilsC.expectation_W_ParsiMod_RVM(m_Wtilde,V_Wtilde,alpha_RVM,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N)

            for cond in xrange(M):
                AW[:,m] = m_A[:,m]*m_Wtilde[m]
            for d in xrange(0,D):
                AWH[:,:,d] = AW[:,:]*m_H[d]
            DIFF = reshape( AWH - AWH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AWH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AWH1,(M*J*D)) ))**2
            cAWH += [Crit_AWH]
            AWH1[:,:,:] = AWH[:,:,:]
            
            if estimateW:
                alpha_RVM = maximization_alphaRVM(k_RVM,lam_RVM,m_Wtilde,V_Wtilde,M,alpha_RVM)
            
            if estimateHRF:
                if estimateSigmaH:
                    pyhrf.verbose(3,"M sigma_H step ...")
                    if gamma_h > 0:
                        sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                    else:
                        sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                    pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))

            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            
            UtilsC.maximization_L_ParsiMod_RVM(Y,m_A,m_H,L,P,XX.astype(int32),m_Wtilde,J,D,M,Ndrift,N)

            PL = numpy.dot(P,L)
            y_tilde = Y - PL

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose.printNdarray(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod_RVM(m_Wtilde,V_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy_RVM(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,m_Wtilde,V_Wtilde,alpha_RVM,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K)
            Crit_FreeEnergy = (FreeEnergy1 - FreeEnergy) / FreeEnergy1
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        print i
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    font = {'size'   : 15}
    matplotlib.rc('font', **font)

    if PLOT:
        savefig('./HRF_Iter_RVM.png')
        hold(False)
        figure(2)
        plot(cAWH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CAWH', 'CFE') )
        grid(True)
        savefig('./Crit_RVM.png')
        
        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy_RVM.png')

    CompTime = t2 - t1
    cTimeMean = CompTime/ni
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
                print ContrastCoef
                print ActiveContrasts
                AC = ActiveContrasts*ContrastCoef
                for j in xrange(0,J):
                    S_tmp = Sigma_A[:,:,j]
                    CONTRASTVAR[j,n] = numpy.dot(numpy.dot(AC,S_tmp),AC)
                #------------ variance -------------#
                n +=1
                pyhrf.verbose(3, 'Done contrasts computing.')

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "sigma_H = " + str(sigmaH)
    if pyhrf.verbose.verbosity > 1:
        print 'mu_M:', mu_M
        print 'sigma_M:', sigma_M
        print "sigma_H = " + str(sigmaH)
        print "Beta = " + str(Beta)

    return ni,m_A,m_H,q_Z,sigma_epsilone,mu_M,sigma_M,Beta,L,PL,CONTRAST,CONTRASTVAR,cA[2:],cH[2:],cZ[2:],cW[2:],m_Wtilde,V_Wtilde,alpha_RVM,cTime[2:],cTimeMean,Sigma_A


def Main_vbjde_NoDrifts_ParsiMod_C_3(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=True,PLOT=False,contrasts=[],computeContrast=False,gamma_h=0,estimateHRF=True,tau1=28.,tau2=0.5,S=100):

    pyhrf.verbose(1,"Fast EM for Parsimonious Model ((Definition 3)) without Drifts and with C Extension started...")
    
    Nb2Norm = 1
    NormFlag = False
    
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5
    Thresh_FreeEnergy = 1e-5
    
    #D = int(numpy.ceil(Thrf/dt))
    D = int(numpy.ceil(Thrf/dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))

    MC_mean = numpy.zeros((M,J,S,K),dtype=numpy.float64)

    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)
    #zerosV = numpy.zeros(V)

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    condition_names = []
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    P = PolyMat( N , 4 , TR)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    invR = numpy.linalg.inv(R)
    Det_invR = numpy.linalg.det(invR)
    print 'Det_invR =', Det_invR
    
    Gamma = numpy.identity(N)
    Gamma = Gamma - numpy.dot(P,P.transpose())
    Det_Gamma = numpy.linalg.det(Gamma)
    print 'Det_Gamma =',Det_Gamma
    
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z1 = numpy.zeros((M,K,J),dtype=numpy.float64)
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()

    p_Wtilde = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde1 = numpy.zeros((M,K),dtype=numpy.float64)
    p_Wtilde[:,1] = 1

    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    m_A1 = numpy.zeros((J,M),dtype=numpy.float64)
    #TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    TT,m_h = getCanoHRF(Thrf,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                #m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*q_Z[m,k,j]

    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)

    if estimateHRF:
      Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    else:
      Sigma_H = numpy.zeros((D,D),dtype=numpy.float64)

    Beta = beta * numpy.ones((M),dtype=numpy.float64)

    #PL = numpy.zeros((N,J),dtype=numpy.float64)
    y_tilde = Y
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_W = 1
    Crit_AH = 1
    Crit_FreeEnergy = 1
    
    cA = []
    cH = []
    cZ = []
    cW = []
    cAH = []
    FreeEnergy_Iter = []
    cFE = []

    cTime = []

    CONTRAST = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    CONTRASTVAR = numpy.zeros((J,len(contrasts)),dtype=numpy.float64)
    Q_barnCond = numpy.zeros((M,M,D,D),dtype=numpy.float64)
    XGamma = numpy.zeros((M,D,N),dtype=numpy.float64)
    HXGamma = numpy.zeros((M,N),dtype=numpy.float64)
    m1 = 0
    for k1 in X: # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1,m2,:,:] = numpy.dot(numpy.dot(X[k1].transpose(),Gamma),X[k2])
            m2 += 1
        XGamma[m1,:,:] = numpy.dot(X[k1].transpose(),Gamma)
        HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
        m1 += 1

    t1 = time.time()

    for ni in xrange(0,NitMin):
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
        DIFF = reshape( m_A - m_A1,(M*J) )
        Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
        cA += [Crit_A]
        m_A1[:,:] = m_A[:,:]

        if estimateHRF:
            pyhrf.verbose(3, "E H step ...")
            UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
            m_H[0] = 0
            m_H[-1] = 0
            # Normalizing H at each Nb2Norm iterations:
            if NormFlag:
                # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                # we should not include them in the normalisation step
                if (ni+1)%Nb2Norm == 0:
                    Norm = norm(m_H)
                    m_H /= Norm
                    Sigma_H /= Norm**2
                    #sigmaH /= Norm**2
                    m_A *= Norm
                    Sigma_A *= Norm**2
                    #mu_M *= Norm
                    #sigma_M *= Norm**2
            if PLOT and ni >= 0:
                figure(M+1)
                plot(m_H)
                hold(True)
            #Update HXGamma
            m1 = 0
            for k1 in X: # Loop over the M conditions
                HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                m1 += 1
            #print 'HXGamma=',HXGamma
        Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0,D):
            AH[:,:,d] = m_A[:,:]*m_H[d]
        DIFF = reshape( AH - AH1,(M*J*D) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
        cAH += [Crit_AH]
        AH1[:,:,:] = AH[:,:,:]

        pyhrf.verbose(3, "E Z step ...")
        if MFapprox:
            UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        if not MFapprox:
            UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
        
        val = reshape(q_Z,(M*K*J))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        q_Z = reshape(val, (M,K,J))
        
        DIFF = reshape( q_Z - q_Z1,(M*K*J) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
        cZ += [Crit_Z]
        q_Z1[:,:,:] = q_Z[:,:,:]
        
        #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
        #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cZ += [Crit_Z]
        #q_Z1[:,:,:] = q_Z[:,:,:]

        pyhrf.verbose(3, "E W step ...")
        UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)

        val = reshape(p_Wtilde,(M*K))
        val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
        p_Wtilde = reshape(val, (M,K))
        
        DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
        DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
        DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
        Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
        cW += [Crit_W]
        p_Wtilde1[:,:] = p_Wtilde[:,:]

        #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
        #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
        #cW += [Crit_W]
        #p_Wtilde1[:,:] = p_Wtilde[:,:]

        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            if gamma_h > 0:
                sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
            else:
                sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
            pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))

        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)
        y_tilde = Y
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                if MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep) 
                if not MFapprox:
                    Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)

        pyhrf.verbose(3,"M sigma noise step ...")
        UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

        #### Computing Free Energy ####
        if ni > 0:
            FreeEnergy1 = FreeEnergy
        FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
            
        if ni > 0:
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
        FreeEnergy_Iter += [FreeEnergy]
        cFE += [Crit_FreeEnergy]

        t02 = time.time()
        cTime += [t02-t1]

    #### If no Convergence Criterion in Min Iterations
    #m_H1[:] = m_H[:]
    #q_Z1[:,:,:] = q_Z[:,:,:]
    #m_A1[:,:] = m_A[:,:]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]
    ####

    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    pyhrf.verbose(3, "E A step ...")
    UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
    DIFF = reshape( m_A - m_A1,(M*J) )
    Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]

    if estimateHRF:
        pyhrf.verbose(3, "E H step ...")
        UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
        m_H[0] = 0
        m_H[-1] = 0
        if NormFlag:
            # Normalizing is done before sigmaH, mu_M and sigma_M estimation
            # we should not include them in the normalisation step
            if (ni+1)%Nb2Norm == 0:
                Norm = norm(m_H)
                m_H /= Norm
                Sigma_H /= Norm**2
                #sigmaH /= Norm**2
                m_A *= Norm
                Sigma_A *= Norm**2
                #mu_M *= Norm
                #sigma_M *= Norm**2
        if PLOT and ni >= 0:
            figure(M+1)
            plot(m_H)
            hold(True)
        #Update HXGamma
        m1 = 0
        for k1 in X: # Loop over the M conditions
            HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
            m1 += 1
    Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
    cH += [Crit_H]
    m_H1[:] = m_H[:]

    for d in xrange(0,D):
        AH[:,:,d] = m_A[:,:]*m_H[d]
    DIFF = reshape( AH - AH1,(M*J*D) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
    cAH += [Crit_AH]
    AH1[:,:,:] = AH[:,:,:]
    
    pyhrf.verbose(3, "E Z step ...")
    if MFapprox:
        UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    if not MFapprox:
        UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
    
    val = reshape(q_Z,(M*K*J))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    q_Z = reshape(val, (M,K,J))
    
    DIFF = reshape( q_Z - q_Z1,(M*K*J) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    
    #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cZ += [Crit_Z]
    #q_Z1[:,:,:] = q_Z[:,:,:]

    pyhrf.verbose(3, "E W step ...")
    UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
    
    val = reshape(p_Wtilde,(M*K))
    val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
    p_Wtilde = reshape(val, (M,K))
    
    DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
    DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
    DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
    Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
    cW += [Crit_W]
    p_Wtilde1[:,:] = p_Wtilde[:,:]
    
    #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
    #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
    #cW += [Crit_W]
    #p_Wtilde1[:,:] = p_Wtilde[:,:]

    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        if gamma_h > 0:
            sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
        else:
            sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
        pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))

    mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)
    y_tilde = Y

    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            if MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
            if not MFapprox:
                Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)

    UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

    #### Computing Free Energy ####
    FreeEnergy1 = FreeEnergy
    FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
    DIFF = FreeEnergy1 - FreeEnergy
    Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
    FreeEnergy_Iter += [FreeEnergy]
    cFE += [Crit_FreeEnergy]

    t02 = time.time()
    cTime += [t02-t1]
    ni += 2
    #if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh):
        #while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (Crit_W > Thresh) and (ni < NitMax)):# or (ni < 50):
    #if (Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh):
        #while ( (((Crit_H > Thresh) or (Crit_A > Thresh) or (Crit_W > Thresh))) and (ni < NitMax) ):# or (ni < 50):
    if (Crit_FreeEnergy > Thresh_FreeEnergy or Crit_AH > Thresh):
        while ( ((Crit_FreeEnergy > Thresh_FreeEnergy) or (Crit_AH > Thresh)) and (ni < NitMax) ):
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            UtilsC.expectation_A_ParsiMod(p_Wtilde,q_Z,mu_M,sigma_M,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,K)
            DIFF = reshape( m_A - m_A1,(M*J) )
            Crit_A = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(m_A1,(M*J)) ))**2
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]

            if estimateHRF:
                UtilsC.expectation_H_ParsiMod(p_Wtilde,XGamma,Q_barnCond,sigma_epsilone,Gamma,R,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N,scale,sigmaH)
                m_H[0] = 0
                m_H[-1] = 0
                if NormFlag:
                    # Normalizing is done before sigmaH, mu_M and sigma_M estimation
                    # we should not include them in the normalisation step
                    if (ni+1)%Nb2Norm == 0:
                        Norm = norm(m_H)
                        m_H /= Norm
                        Sigma_H /= Norm**2
                        #sigmaH /= Norm**2
                        m_A *= Norm
                        Sigma_A *= Norm**2
                        #mu_M *= Norm
                        #sigma_M *= Norm**2
                if PLOT and ni >= 10:
                    figure(M+1)
                    plot(m_H)
                    hold(True)
                #Update HXGamma
                m1 = 0
                for k1 in X: # Loop over the M conditions
                    HXGamma[m1,:] = numpy.dot(numpy.dot(m_H.transpose(),X[k1].transpose()),Gamma)
                    m1 += 1
            Crit_H = (numpy.linalg.norm( m_H - m_H1 ) / numpy.linalg.norm( m_H1 ))**2
            cH += [Crit_H]
            m_H1[:] = m_H[:]

            for d in xrange(0,D):
                AH[:,:,d] = m_A[:,:]*m_H[d]
            DIFF = reshape( AH - AH1,(M*J*D) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_AH = (numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(AH1,(M*J*D)) ))**2
            cAH += [Crit_AH]
            AH1[:,:,:] = AH[:,:,:]

            if MFapprox:
                UtilsC.expectation_Z_MF_ParsiMod_3(p_Wtilde,Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            if not MFapprox:
                UtilsC.expectation_Z_ParsiMod_3(Sigma_A,m_A,sigma_M,Beta,p_Wtilde,mu_M,q_Z,neighboursIndexes.astype(int32),M,J,K,maxNeighbours)
            
            val = reshape(q_Z,(M*K*J))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            q_Z = reshape(val, (M,K,J))
            
            DIFF = reshape( q_Z - q_Z1,(M*K*J) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_Z = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(q_Z1,(M*K*J)) ))**2
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            
            #DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            #Crit_Z = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cZ += [Crit_Z]
            #q_Z1[:,:,:] = q_Z[:,:,:]

            UtilsC.expectation_W_ParsiMod_3(p_Wtilde,q_Z,HXGamma,sigma_epsilone,Gamma,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),mu_M,sigma_M,J,D,M,N,K,tau1,tau2)
            
            val = reshape(p_Wtilde,(M*K))
            val[ find((val<=1e-50) & (val>0.0)) ] = 0.0
            p_Wtilde = reshape(val, (M,K))
            
            DIFF = reshape( p_Wtilde - p_Wtilde1,(M*K) )
            DIFF[ find( (DIFF<1e-50) & (DIFF>0.0) ) ] = 0.0 #### To avoid numerical problems
            DIFF[ find( (DIFF>-1e-50) & (DIFF<0.0) ) ] = 0.0 #### To avoid numerical problems
            Crit_W = ( numpy.linalg.norm(DIFF) / numpy.linalg.norm( reshape(p_Wtilde1,(M*K)) ))**2
            cW += [Crit_W]
            p_Wtilde1[:,:] = p_Wtilde[:,:]
            
            #DIFF = abs(reshape(p_Wtilde,(M*K)) - reshape(p_Wtilde1,(M*K)))
            #Crit_W = (sum(DIFF) / len(find(DIFF != 0)))**2
            #cW += [Crit_W]
            #p_Wtilde1[:,:] = p_Wtilde[:,:]

            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                if gamma_h > 0:
                    sigmaH = maximization_sigmaH_prior(D,Sigma_H,R,m_H,gamma_h)
                else:
                    sigmaH = maximization_sigmaH(D,Sigma_H,R,m_H)
                pyhrf.verbose(3,'sigmaH = ' + str(sigmaH))

            mu_M , sigma_M = maximization_mu_sigma_ParsiMod3(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A,p_Wtilde,J,tau1,tau2,ni,estimateW)

            y_tilde = Y

            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    if MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),Z_tilde[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                    if not MFapprox:
                        Beta[m] = UtilsC.maximization_beta(beta,q_Z[m,:,:].astype(float64),q_Z[m,:,:].astype(float64),J,K,neighboursIndexes.astype(int32),gamma,maxNeighbours,MaxItGrad,gradientStep)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)

            UtilsC.maximization_sigma_noise_ParsiMod(p_Wtilde,Gamma,sigma_epsilone,Sigma_H,y_tilde,m_A,m_H,Sigma_A,XX.astype(int32),J,D,M,N)

            #### Computing Free Energy ####
            FreeEnergy1 = FreeEnergy
            FreeEnergy = Compute_FreeEnergy(y_tilde,m_A,Sigma_A,mu_M,sigma_M,m_H,Sigma_H,R,Det_invR,sigmaH,p_Wtilde,tau1,tau2,q_Z,neighboursIndexes,maxNeighbours,Beta,sigma_epsilone,XX,Gamma,Det_Gamma,XGamma,J,D,M,N,K,S,"ParsiMod3")
            DIFF = FreeEnergy1 - FreeEnergy
            Crit_FreeEnergy = DIFF / (FreeEnergy1**2)
            FreeEnergy_Iter += [FreeEnergy]
            cFE += [Crit_FreeEnergy]

            ni +=1

            t02 = time.time()
            cTime += [t02-t1]

    t2 = time.time()

    FreeEnergyArray = numpy.zeros((NitMax+1),dtype=numpy.float64)
    for i in xrange(ni):
        FreeEnergyArray[i] = FreeEnergy_Iter[i]
    for i in xrange(ni-1,NitMax+1):
        FreeEnergyArray[i] = FreeEnergy_Iter[ni-1]

    if PLOT:
        savefig('./HRF_Iter.png')
        figure(M+2)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(True)
        plot(cW[1:-1],'g')
        hold(True)
        plot(cAH[1:-1],'lightblue')
        hold(True)
        plot(cFE[1:-1],'m')
        hold(False)
        legend( ('CA','CH', 'CZ', 'CW', 'CAH', 'CFE') )
        grid(True)
        savefig('./Crit.png')

        figure(3)
        plot(FreeEnergyArray)
        savefig('./FreeEnergy.png')

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
            nrls_conds = dict([(cn, m_A[:,ic]) for ic,cn in enumerate(condition_names)] )
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
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
                CovM = numpy.ones(M,dtype=float)
                for j in xrange(0,J):
                    CovM = numpy.ones(M,dtype=float)
                    for m in xrange(0,M):
                        if ActiveContrasts[m]:
                            CONTRASTVAR[j,n] += (ContrastCoef[m]**2) * Sigma_A[m,m,j]
                            for m2 in xrange(0,M):
                                if ( (ActiveContrasts[m2]) and (CovM[m2]) and (m2 != m)):
                                    CONTRASTVAR[j,n] += 2*ContrastCoef[m] * ContrastCoef[m2] * Sigma_A[m,m2,j]
                                    CovM[m2] = 0
                                    CovM[m] = 0
                #------------ variance -------------#
                n +=1
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print 'p_Wtilde =',p_Wtilde
    StimulusInducedSignal = computeFit(m_H, m_A, X, J, N)
    #print "Beta = " + str(Beta)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, CONTRAST, CONTRASTVAR, cA[2:],cH[2:],cZ[2:],cW[2:], p_Wtilde,cTime[2:],cTimeMean,Sigma_A,MC_mean, StimulusInducedSignal, FreeEnergyArray         


def Main_vbjde_Python(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,NitMax = -1,NitMin = 1,estimateBeta=False,PLOT=False):
    pyhrf.verbose(1,"EM started ...")
    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.005
    MaxItGrad = 120
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    #-----------------------------------------------------------------------#
    # put neighbour lists into a 2D numpy array so that it will be easily
    # passed to C-code
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = numpy.zeros((J, maxNeighbours), dtype=numpy.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i,:len(graph[i])] = graph[i]
    #-----------------------------------------------------------------------#
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
    XX = numpy.zeros((M,N,D),dtype=numpy.int32)
    nc = 0
    for condition,Ons in Onsets.iteritems():
        XX[nc,:,:] = X[condition]
        nc += 1
    mu_M = numpy.zeros((M,K),dtype=numpy.float64)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=numpy.float64)
    sigma_M0 = 0.5*numpy.ones((M,K),dtype=numpy.float64)
    for k in xrange(1,K):
        mu_M[:,k] = 2.0
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
    #for k in xrange(0,K):
    q_Z[:,1,:] = 1
    Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),numpy.float64)
    m_A = numpy.zeros((J,M),dtype=numpy.float64)
    TT,m_h = getCanoHRF(Thrf-dt,dt) #TODO: check
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
    m_H = numpy.array(m_h).astype(numpy.float64)
    m_H1 = numpy.array(m_h)
    Sigma_H = numpy.ones((D,D),dtype=numpy.float64)
    Beta = beta * numpy.ones((M),dtype=numpy.float64)
    zerosDD = numpy.zeros((D,D),dtype=numpy.float64)
    zerosD = numpy.zeros((D),dtype=numpy.float64)
    zerosND = numpy.zeros((N,D),dtype=numpy.float64)
    zerosMM = numpy.zeros((M,M),dtype=numpy.float64)
    zerosJMD = numpy.zeros((J,M,D),dtype=numpy.float64)
    zerosK = numpy.zeros(K)
    P = PolyMat( N , 4 , TR)
    zerosP = numpy.zeros((P.shape[0]),dtype=numpy.float64)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH
    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    cA = []
    cH = []
    cZ = []
    Ndrift = L.shape[0]
    t1 = time.time()
    for ni in xrange(0,NitMin):
        print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
        Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
        pyhrf.verbose(3, "E Z step ...")
        q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
        figure(1)
        plot(m_H,'r')
        hold(False)
        draw()
        show()

        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
            sigmaH /= D
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
        #print mu_M , sigma_M

        L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
        PL = numpy.dot(P,L)
        #print L.shape
        #for j in xrange(0,J):
            #print j
            #print '--------------------------'
            #print L[:,j]
            #print '--------------------------'
            #raw_input('')
        #print L
        #raw_input('')
        y_tilde = Y - PL
        if estimateBeta:
            pyhrf.verbose(3,"estimating beta")
            for m in xrange(0,M):
                Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma,neighboursIndexes,maxNeighbours)
            print Beta
            pyhrf.verbose(3,"End estimating beta")
            pyhrf.verbose(3,Beta)
        pyhrf.verbose(3,"M sigma noise step ...")
        sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
    m_H1[:] = m_H[:]
    q_Z1[:,:,:] = q_Z[:,:,:]
    m_A1[:,:] = m_A[:,:]
    pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+2) + " ------------------------------")
    Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
    DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
    Crit_A = sum(DIFF) / len(find(DIFF != 0))
    cA += [Crit_A]
    m_A1[:,:] = m_A[:,:]
    Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
    m_H[0] = 0
    m_H[-1] = 0
    Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
    cH += [Crit_H]
    m_H1[:] = m_H[:]
    q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
    DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
    Crit_Z = sum(DIFF) / len(find(DIFF != 0))
    cZ += [Crit_Z]
    q_Z1[:,:,:] = q_Z[:,:,:]
    if estimateSigmaH:
        pyhrf.verbose(3,"M sigma_H step ...")
        sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
        sigmaH /= D
    mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
    L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    if estimateBeta:
        pyhrf.verbose(3,"estimating beta")
        for m in xrange(0,M):
            Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma,neighboursIndexes,maxNeighbours)
        pyhrf.verbose(3,"End estimating beta")
        pyhrf.verbose(3,Beta)
    sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
    ni += 2
    if (Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh):
        while ((Crit_H > Thresh) and (Crit_Z > Thresh) and (Crit_A > Thresh) and (ni < NitMax)):# or (ni < 50):
            pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
            Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone,zerosJMD)
            DIFF = abs(reshape(m_A,(M*J)) - reshape(m_A1,(M*J)))
            Crit_A = sum(DIFF) / len(find(DIFF != 0))
            m_A1[:,:] = m_A[:,:]
            cA += [Crit_A]
            Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale,zerosDD,zerosD)
            m_H[0] = 0
            m_H[-1] = 0
            Crit_H = abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))
            cH += [Crit_H]
            m_H1[:] = m_H[:]
            q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K,zerosK)
            DIFF = abs(reshape(q_Z,(M*K*J)) - reshape(q_Z1,(M*K*J)))
            Crit_Z = sum(DIFF) / len(find(DIFF != 0))
            cZ += [Crit_Z]
            q_Z1[:,:,:] = q_Z[:,:,:]
            if estimateSigmaH:
                pyhrf.verbose(3,"M sigma_H step ...")
                sigmaH = (numpy.dot(mult(m_H,m_H) + Sigma_H , R )).trace()
                sigmaH /= D
            mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M,Sigma_A)
            L = maximization_L(Y,m_A,X,m_H,L,P,zerosP)
            PL = numpy.dot(P,L)
            y_tilde = Y - PL
            if estimateBeta:
                pyhrf.verbose(3,"estimating beta")
                for m in xrange(0,M):
                    Beta[m] = maximization_beta(Beta[m],q_Z,Z_tilde,J,K,m,graph,gamma,neighboursIndexes,maxNeighbours)
                pyhrf.verbose(3,"End estimating beta")
                pyhrf.verbose(3,Beta)
            sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M,zerosMM)
            ni +=1
    t2 = time.time()
    CompTime = t2 - t1
    
    if PLOT:
        figure(1)
        plot(cA[1:-1],'r')
        hold(True)
        plot(cH[1:-1],'b')
        hold(True)
        plot(cZ[1:-1],'k')
        hold(False)
        legend( ('CA','CH', 'CZ') )
        grid(True)
        draw()
        show()
    Norm = norm(m_H)
    m_H /= Norm
    m_A *= Norm
    mu_M *= Norm
    sigma_M *= Norm
    sigma_M = sqrt(sigma_M)
    pyhrf.verbose(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    print "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s"
    print 'mu_M:', mu_M
    print 'sigma_M:', sigma_M
    print "sigma_H = " + str(sigmaH)
    print "Beta = " + str(Beta)
    return m_A,m_H, q_Z , sigma_epsilone, mu_M , sigma_M, Beta, L, PL


def Main_vbjde(graph,Y,Onsets,Thrf,K,TR,beta,dt,scale=1,estimateSigmaH=True,sigmaH = 0.1,PLOT = False,NitMax = -1,NitMin = 1,hrf = None):
    pyhrf.verbose(2,"EM started ...")
    if NitMax < 0:
        NitMax = 100
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    sigma_epsilone = numpy.ones(J)
    X = OrderedDict([])
    for condition,Ons in Onsets.iteritems():
        X[condition] = compute_mat_X_2(N, TR, D, dt, Ons)
    mu_M = numpy.zeros((M,K),dtype=float)
    sigma_M = 0.5 * numpy.ones((M,K),dtype=float)
    mu_M0 = numpy.zeros((M,K),dtype=float)
    sigma_M0 = numpy.zeros((M,K),dtype=float)
    for k in xrange(0,K):
        mu_M[:,0] = 2.0
    mu_M0[:,:] = mu_M[:,:]
    sigma_M0[:,:] = sigma_M[:,:]
    #sigmaH = 0.005
    order = 2
    D2 = buildFiniteDiffMatrix(order,D)
    R = numpy.dot(D2,D2) / pow(dt,2*order)
    Gamma = numpy.identity(N)
    q_Z = numpy.zeros((M,K,J),dtype=float)
    for k in xrange(0,K):
        q_Z[:,1,:] = 1
    q_Z1 = q_Z.copy()
    Z_tilde = q_Z.copy()
    Sigma_A = numpy.zeros((M,M,J),float)
    m_A = numpy.zeros((J,M),dtype=float)
    TT,m_h = getCanoHRF(Thrf-dt,dt)
    for j in xrange(0,J):
        Sigma_A[:,:,j] = 0.01*numpy.identity(M)
        for m in xrange(0,M):
            for k in xrange(0,K):
                m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
    m_H = numpy.array(m_h)
    m_H1 = numpy.array(m_h)
    Sigma_H = numpy.ones((D,D),dtype=float)
    #Sigma_H = 0.1 * numpy.identity(D)
    Beta = beta * numpy.ones((M),dtype=float)
    m_A1 = numpy.zeros((J,M),dtype=float)
    m_A1[:,:] = m_A[:,:]
    Crit_H = [0]
    Crit_Z = [0]
    Crit_sigmaH = [0]
    Hist_sigmaH = []
    ni = 0
    Y_bar_tilde = numpy.zeros((D),dtype=float)
    zerosND = numpy.zeros((N,D),dtype=float)
    X_tilde = numpy.zeros((Y.shape[1],M,D),dtype=float)
    Q_bar = numpy.zeros(R.shape)
    P = PolyMat( N , 4 , TR)
    L = polyFit(Y, TR, 4,P)
    PL = numpy.dot(P,L)
    y_tilde = Y - PL
    sigmaH1 = sigmaH

    t1 = time.time()
    while (( (ni < NitMin) or (Crit_sigmaH[-1] > 5e-3) or (Crit_H[-1] > 5e-3) or (Crit_Z[-1] > 5e-3))) \
            and (ni < NitMax):
        #if PLOT:
            #print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
        pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
        pyhrf.verbose(3, "E A step ...")
        Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone)
        pyhrf.verbose(3,"E H step ...")
        Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,scale)
        Crit_H += [abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))]
        m_H1[:] = m_H[:]
        pyhrf.verbose(3,"E Z step ...")
        q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K)
        DIFF = abs(numpy.reshape(q_Z,(M*K*J)) - numpy.reshape(q_Z1,(M*K*J)))
        Crit_Z += [numpy.mean(DIFF) / (DIFF != 0).sum()]
        q_Z1[:,:,:] = q_Z[:,:,:]
        pyhrf.verbose(3,"M (mu,sigma) step ...")
        mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M)
        if estimateSigmaH:
            pyhrf.verbose(3,"M sigma_H step ...")
            sigmaH = numpy.dot(numpy.dot(m_H.transpose(),R) , m_H ) + (numpy.dot(Sigma_H,R)).trace()
            sigmaH /= D
            Crit_sigmaH += [abs((sigmaH - sigmaH1) / sigmaH)]
            Hist_sigmaH += [sigmaH]
            sigmaH1 = sigmaH
        pyhrf.verbose(3,"M L step ...")
        L = maximization_L(Y,m_A,X,m_H,L,P)
        PL = numpy.dot(P,L)
        y_tilde = Y - PL
        pyhrf.verbose(3,"M sigma_epsilone step ...")
        sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M)
        #if ( (ni+1)% 1) == 0:
        if PLOT:
            from matplotlib import pyplot
            m_Htmp = m_H / norm(m_H)
            hrftmp = hrf / norm(hrf)
            snrH = 20*numpy.log(1 / norm(m_Htmp - hrftmp))
            #print snrH
            pyplot.clf()
            pyplot.figure(1)
            pyplot.plot(m_H/norm(m_H),'r')
            pyplot.hold(True)
            pyplot.plot(hrf/norm(hrf),'b')
            pyplot.legend( ('Est','Ref') )
            pyplot.title(str(snrH))
            pyplot.hold(False)
            pyplot.draw()
            pyplot.show()
            #figure(2)
            #plot(Hist_sigmaH)
            #title(str(sigmaH))
            ##hold(False)
            #draw()
            #show()
            #for m in range(0,M):
                #for k in range(0,K):
                    #z1 = q_Z[m,k,:];
                    #z2 = reshape(z1,(l,l));
                    #figure(2).add_subplot(M,K,1 + m*K + k)
                    #imshow(z2)
                    #title("m = " + str(m) +"k = " + str(k))
            #draw()
            #show()
        ni +=1
    t2 = time.time()
    CompTime = t2 - t1
    Norm = norm(m_H)
    #print Norm
    m_H /= Norm
    m_A *= Norm
    pyhrf.veborse(1, "Nb iterations to reach criterion: %d" %ni)
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    return m_A, m_H, q_Z , sigma_epsilone, (numpy.array(Hist_sigmaH)).transpose()
