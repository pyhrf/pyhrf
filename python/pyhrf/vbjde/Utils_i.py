

#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.linalg import toeplitz,norm,inv
import numpy
from matplotlib import pyplot
import time
import scipy
from pylab import *
from pylab import normal,inv
from pyhrf.paradigm import restarize_events
from nifti import NiftiImage
import csv
#from pyhrf import *
from pyhrf.boldsynth.hrf import getCanoHRF
import pyhrf


eps = 1e-4
def polyFit(signal, tr, order,p):
    n = len(signal)
    ptp = numpy.dot(p.transpose(),p)
    invptp = numpy.linalg.inv(ptp)
    invptppt = numpy.dot(invptp, p.transpose())
    l = numpy.dot(invptppt,signal)
    return l

def PolyMat( Nscans , paramLFD , tr):
    '''Build polynomial basis'''
    regressors = tr * numpy.arange(0, Nscans)
    timePower = numpy.arange(0,paramLFD+1, dtype=int)
    regMat = numpy.zeros((len(regressors),paramLFD+1),dtype=float)

    for v in xrange(paramLFD+1):
        regMat[:,v] = regressors[:]    
    tPowerMat = numpy.matlib.repmat(timePower, Nscans, 1)
    lfdMat = numpy.power(regMat,tPowerMat)
    lfdMat = numpy.array(scipy.linalg.orth(lfdMat))
    return lfdMat

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(numpy.sqrt(2*numpy.pi)*abs(sigma)))*numpy.exp(-u*u/2)
    return y 
    
def compute_mat_X_2(nbscans, tr, lhrf, dt, onsets, durations=None):
    
    if durations is None: #assume spiked stimuli
	durations = numpy.zeros_like(onsets)

    osf = tr/dt # over-sampling factor
    assert( int(osf) == osf) #construction will only work if dt is a multiple of tr
    
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

def expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_MK,q_Z,mu_MK,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone):
    X_tilde = numpy.zeros((Y.shape[1],M,D),dtype=float)
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

def expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,Sigma_H,m_H):
    Q_bar = numpy.zeros((D,D))
    Y_bar_tilde = numpy.zeros((D),dtype=float)
    for i in xrange(0,J):
	m = 0
	tmp =  zerosND.copy() #numpy.zeros((N,D),dtype=float)
	for k in X: # Loop over the M conditions
	    tmp += m_A[i,m] * X[k]
	    m += 1
	Y_bar_tilde = numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),y_tilde[:,i])
	Q_bar += numpy.dot(numpy.dot(tmp.transpose(),Gamma/max(sigma_epsilone[i],eps)),tmp)
	m1 = 0
	for k1 in X: # Loop over the M conditions
	    m2 = 0
	    for k2 in X: # Loop over the M conditions
		Q_bar += Sigma_A[m1,m2,i] * numpy.dot(numpy.dot(X[k1].transpose(),Gamma/max(sigma_epsilone[i],eps)),X[k2])
		m2 +=1
	    m1 +=1	
	Q_bar += R/sigmaH
	Sigma_H[:,:,i] = inv(Q_bar)
	m_H[:,i] = numpy.dot(Sigma_H[:,:,i],Y_bar_tilde)
	m_H[0,i] = 0
	m_H[-1,i] = 0
    return Sigma_H, m_H
    
def expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K):
    energy = numpy.zeros(K)
    Gauss = numpy.zeros(K)
    for i in xrange(0,J):
	for m in xrange(0,M):
	    alpha = Sigma_A[m,m,i] / (sigma_M[m,:] + eps)
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
	    alpha = Sigma_A[m,m,i] / (sigma_M[m,:] + eps)
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

def maximization_mu_sigma(Mu,Sigma,q_Z,m_A,K,M):
    for m in xrange(0,M):
	for k in xrange(0,K):
	    S = sum( q_Z[m,k,:] )
	    Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
	    Sigma[m,k] = eps + sum( q_Z[m,k,:] * pow(m_A[:,m] - Mu[m,k] ,2)   ) / S
    return Mu , Sigma

def maximization_L(Y,m_A,X,m_H,L,P):
    J = Y.shape[1]
    for i in xrange(0,J):
	S = numpy.zeros((P.shape[0]),dtype=float)
	m = 0
	for k in X:
	    S += m_A[i,m]*numpy.dot(X[k],m_H[:,i])
	    m +=1
	L[:,i] = numpy.dot(P.transpose(), Y[:,i] - S )
    return L

def maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M):
    N = PL.shape[0]
    J = Y.shape[1]
    Htilde = numpy.zeros((M,M),dtype=float)
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
    
def maximization_sigmaH(m_H,R,J,D,Sigma_H,sigmaH):
    sigmaH = 0.0
    for i in xrange(0,J):
	sigmaH += numpy.dot(numpy.dot(m_H[:,i].transpose(),R) , m_H[:,i] ) 
	sigmaH += (numpy.dot(Sigma_H[:,:,i],R)).trace()
	#sigmaH[i] /= D
    sigmaH /= D*J
    return sigmaH

def ReadNII(fname):
    nim = NiftiImage(fname)
    D = nim.data
    return D
    
def buildFiniteDiffMatrix(order, size):
    o = order
    a = numpy.diff(numpy.concatenate((numpy.zeros(o),[1],numpy.zeros(o))),n=o)
    b = a[len(a)/2:]
    diffMat = toeplitz(numpy.concatenate((b, numpy.zeros(size-len(b)))))
    return diffMat

 
def Main_vbjde(graph,Y,Onsets,Thrf,K,TR,beta,dt,hrf,NitMax = -1, hrf = None):    
    if NitMax < 0:
	NitMax = 100
    D = int(numpy.ceil(Thrf/dt))
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    l = int(sqrt(J))
    sigma_epsilone = numpy.ones(J)
    X = {}
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
    #sigmaH = numpy.ones((J),dtype=float)
    #sigmaH1 = numpy.ones((J),dtype=float)
    sigmaH = 0.1
    sigmaH1 = 0.1
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
    m_H = numpy.zeros((D,J),dtype=float)
    m_H1 = numpy.zeros((D,J),dtype=float)
    m_H2 = numpy.zeros((D,J),dtype=float)
    TT,m_h = getCanoHRF(Thrf-dt,dt)
    for j in xrange(0,J):
	Sigma_A[:,:,j] = 0.01*numpy.identity(M)
	for m in xrange(0,M):
	    for k in xrange(0,K):
		m_A[j,m] += normal(mu_M[m,k], numpy.sqrt(sigma_M[m,k]))*Z_tilde[m,k,j]
	m_H[:,j] = numpy.array(m_h)
	m_H1[:,j] = numpy.array(m_h)
    #m_H = numpy.array(m_h)
    #m_H1 = numpy.array(m_h)
    Sigma_H = numpy.ones((D,D,J),dtype=float)
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
    t1 = time.time()
    Norm = numpy.zeros((J),dtype=float)
    while (( (ni < 15) or (Crit_sigmaH[-1] > 5e-3) or (Crit_H[-1] > 5e-3) or (Crit_Z[-1] > 5e-3))) \
	    and (ni < NitMax):
	print "------------------------------ Iteration n° " + str(ni+1) + " ------------------------------"
	pyhrf.verbose(2,"------------------------------ Iteration n° " + str(ni+1) + " ------------------------------")
	pyhrf.verbose(3, "E A step ...")
	Sigma_A, m_A = expectation_A(Y,Sigma_H,m_H,m_A,X,Gamma,PL,sigma_M,q_Z,mu_M,D,N,J,M,K,y_tilde,Sigma_A,sigma_epsilone)
	pyhrf.verbose(3,"E H step ...")
	Sigma_H, m_H = expectation_H(Y,Sigma_A,m_A,X,Gamma,PL,D,R,sigmaH,J,N,y_tilde,zerosND,sigma_epsilone,Sigma_H,m_H)
	Crit_H += [abs(numpy.mean(m_H - m_H1) / numpy.mean(m_H))]
	m_H1[:,:] = m_H[:,:]
	m_H2[:,:] = m_H[:,:]
	pyhrf.verbose(3,"E Z step ...")
	q_Z,Z_tilde = expectation_Z(Sigma_A,m_A,sigma_M,Beta,Z_tilde,mu_M,q_Z,graph,M,J,K)
	DIFF = abs(numpy.reshape(q_Z,(M*K*J)) - numpy.reshape(q_Z1,(M*K*J)))
	Crit_Z += [numpy.mean(DIFF) / (DIFF != 0).sum()]
	q_Z1[:,:] = q_Z[:,:]
	pyhrf.verbose(3,"M (mu,sigma) step ...")
	mu_M , sigma_M = maximization_mu_sigma(mu_M,sigma_M,q_Z,m_A,K,M)
	pyhrf.verbose(3,"M sigma_H step ...")
	#print "M sigma_H step ..."
	sigmaH = maximization_sigmaH(m_H,R,J,D,Sigma_H,sigmaH)
	#print sigmaH
	    #sigmaH = numpy.dot(numpy.dot(m_H.transpose(),R) , m_H ) #+ (numpy.dot(Sigma_H,R)).trace() 
	Crit_sigmaH += [abs((sigmaH - sigmaH1) / sigmaH)]
	Hist_sigmaH += [sigmaH]
	sigmaH1 = sigmaH
	pyhrf.verbose(3,"M L step ...")
	L = maximization_L(Y,m_A,X,m_H,L,P)
	PL = numpy.dot(P,L)
	y_tilde = Y - PL
	pyhrf.verbose(3,"M sigma_epsilone step ...")
	sigma_epsilone = maximization_sigma_noise(Y,X,m_A,m_H,Sigma_H,Sigma_A,PL,sigma_epsilone,M)
	for i in xrange(0,J):
	    Norm[i] = norm(m_H[:,i])	
	    m_H2[:,i] /= Norm[i]
	if ( (ni+1)% 1) == 0:
	    pyplot.clf()
	    figure(1)
	    plot(m_H[:,10],'r')
	    hold(True)
	    plot(hrf/norm(hrf),'b')
	    legend( ('Est','Ref') )
	    title(str(sigmaH))
	    hold(False)
	    draw()
	    show()
	    for m in range(0,M):
		for k in range(0,K):		
		    z1 = q_Z[m,k,:];
		    z2 = reshape(z1,(l,l));
		    figure(2).add_subplot(M,K,1 + m*K + k)
		    imshow(z2)
		    title("m = " + str(m) +"k = " + str(k))
	    draw()
	    show()
	ni +=1
    t2 = time.time()
    CompTime = t2 - t1
    
    for i in xrange(0,J):
	Norm[i] = norm(m_H[:,i])	
	m_H[:,i] /= Norm[i]
	m_A[i,:] *= Norm[i]
    pyhrf.verbose(1, "Computational time = " + str(int( CompTime//60 ) ) + " min " + str(int(CompTime%60)) + " s")
    return m_A, m_H, q_Z , sigma_epsilone, sigmaH

