# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

from scipy.linalg import toeplitz
from scipy import diff

from pyhrf.tools import resampleToGrid

hCano = np.array([0. , 0.00025487, 0.00447602, 0.01865396, 0.04314079,
                     0.07225387, 0.0986708 , 0.11704072, 0.12522419, 0.12381613,
                     0.11500752, 0.10150135, 0.08578018, 0.0697497 , 0.05466205,
                     0.0412003 , 0.02962771, 0.01994009, 0.01199021, 0.00557441,
                     0.00048408,-0.00346985,-0.00644989,-0.00859454,-0.01002438,
                     -0.01084815,-0.01116695,-0.01107628,-0.0106662 ,-0.01002018,
                     -0.0092136 ,-0.00831224,-0.00737127,-0.00643486,-0.00553647,
                     -0.00469968,-0.00393938,-0.00326324,-0.00267318,-0.0021669 ,
                     -0.00173909,-0.00138262, 0.])
dtHCano = 0.6
tAxisHCano = np.arange(0., len(hCano)*dtHCano, dtHCano)

def genPriorCov(zc, pprcov, dt):
    #TODO : comment
    if zc :
        a = np.array([-4,6,-4,1], dtype=float)
        b = np.array([5,-4,1], dtype=float)
        matQ = np.zeros((pprcov,pprcov), dtype=float)
        matQ[0,0:3] = b
        matQ[pprcov-1,:] = [matQ[0,i] for i in range(len(matQ)-1,-1,-1)]
        matQ[1,0:4] = a
        matQ[pprcov-2,:] = [matQ[1,i] for i in range(len(matQ)-1,-1,-1)]
        c = np.concatenate(([1],np.zeros(pprcov-5)))
        r0 = np.concatenate(([1], a))
        r1 = np.zeros(pprcov-5)
        r = np.concatenate((r0,r1))
        matQ[2:pprcov-2,:] = toeplitz(c,r)
        matQ = np.divide(matQ,dt**4)
    else :
        edgePrior = np.zeros((pprcov, pprcov), dtype=float)
        edgePrior[0,0] = 1
        edgePrior[pprcov-1, pprcov-1] = 1
        b = np.array([1,-1], dtype=float)
        c = np.concatenate(([1],np.zeros((pprcov-2))))
        r1 = np.zeros(pprcov-2)
        r = np.concatenate((b,r1))
        varD1 = toeplitz(c,r)
        gradPrior = np.dot(varD1.transpose(),varD1)

        edgeGradPrior = np.zeros((pprcov, pprcov), dtype=float)
        edgeGradPrior[(0,pprcov-1),:] = gradPrior[(1,pprcov-1),:]
        epsilon = 0; #1e-8
        a = np.array([1,-2,1],dtype=float)
        c = np.concatenate(([1], np.zeros(pprcov-3)))
        r = np.concatenate((a , np.zeros(pprcov-3)))
        varD2 = toeplitz(c,r)

        matQ = edgePrior + edgeGradPrior + np.dot(varD2.transpose(),varD2)
        matQ = matQ + epsilon*np.eye(pprcov)
        matQ = np.divide(matQ, dt**4)

    return matQ


def buildFiniteDiffMatrix(order, size):

    o = order
    a = np.diff(np.concatenate((np.zeros(o),[1],np.zeros(o))),n=o)
    b = a[len(a)/2:]
    diffMat = toeplitz(np.concatenate((b, np.zeros(size-len(b)))))
    #print 'diffMat :'
    #print diffMat
    return diffMat



def genGaussianSmoothHRF(zc, length, eventdt, rh, order=2):

    prcov = length-(2*zc)
    matQ = buildFiniteDiffMatrix(order, prcov)
    matQ = np.divide(np.dot(matQ.transpose(),matQ), eventdt**(order**2))
    matL = np.array(np.transpose(np.linalg.cholesky(matQ/rh)))

    hrf = np.linalg.solve(matL, np.random.randn(prcov,1))
    #hrf = np.sqrt(rh)*hrf
    if zc :
        hrf = np.concatenate(([[0]],hrf,[[0]]))

    return (hrf[:,0],matQ)


def genExpHRF(timeAxis=np.arange(0,25,0.5), ttp=6, pa=1, pw=0.2,
              ttu=11, ua=0.2, uw=0.01):
    pic = pa*np.exp(-pw*(timeAxis-ttp)**2)
    uShoot = ua*np.exp(-uw*(timeAxis-ttu)**2)
    # calculate value at origin :
    valAtOrig = pa*np.exp(-pw*ttp**2)-ua*np.exp(-uw*ttu**2)
    # correct origin value to be zero :
    return pic-uShoot-valAtOrig


def getCanoHRF(duration=25, dt=.6, hrf_from_spm=True, delay_of_response=6.,
               delay_of_undershoot=16., dispersion_of_response=1.,
               dispersion_of_undershoot=1., ratio_resp_under=6., delay=0.):
    """Compute the canonical HRF.

    Parameters
    ----------
    duration : int or float, optional
        time lenght of the HRF in seconds
    dt : float, optional
        time resolution of the HRF in seconds
    hrf_from_spm : bool, optional
        if True, use the SPM formula to compute the HRF, if False, use the hard
        coded values and resample if necessary. It is strongly advised to use
        True.
    delay_of_response : float, optional
        delay of the first peak response in seconds
    delay_of_undershoot : float, optional
        delay of the second undershoot peak in seconds
    dispersion_of_response : float, optional
    dispersion_of_undershoot : float, optional
    ratio_resp_under : float, optional
        ratio between the response peak and the undershoot peak
    delay : float, optional
        delay of the HRF

    Returns
    -------
    time_axis : ndarray, shape (round(duration/dt)+1,)
        time axis of the HRF in seconds
    HRF : ndarray, shape (round(duration/dt)+1,)

    """

    time_axis = np.arange(0, duration+dt, dt)
    axis = np.arange(len(time_axis))

    if hrf_from_spm:
        hrf_cano = (scipy.stats.gamma.pdf(axis - delay/dt,
                                          delay_of_response/dispersion_of_response, 0,
                                          dispersion_of_response/dt) -
                    scipy.stats.gamma.pdf(axis - delay/dt,
                                          delay_of_undershoot/dispersion_of_undershoot,
                                          0, dispersion_of_undershoot/dt)/ratio_resp_under)
        hrf_cano[-1] = 0
        hrf_cano /= np.linalg.norm(hrf_cano)
    else:
        hrf_cano = resampleToGrid(time_axisHCano, hCano.copy(), time_axis)
        hrf_cano[-1] = 0.
        assert len(hrf_cano) == len(time_axis)
        hrf_cano /= (hrf_cano**2).sum()**.5

    return time_axis, hrf_cano


def getCanoHRF_tderivative(duration=25., dt=.5):

    hcano = getCanoHRF()
    hcano[-1] = 0.
    tAxis = np.arange(0, duration+dt, dt)
    hderiv = diff(hcano[1], n=1)/dt
    hderiv = np.hstack(([0], hderiv))
    return tAxis, hderiv

def genCanoBezierHRF(duration=25., dt=.6, normalize=False):

    tAxis = np.arange(0,duration+dt, dt)
    tPic = 5
    if tPic >= duration:
        tPic = duration * .3
    tus = np.round(tPic+(duration-tPic)*0.6)

    # for 15sec:
    #pic : [5, 1]
    #picw: 2
    #ushoot: [11.0, -0.20000000000000001]
    #ushootw: 3
    h = genBezierHRF(timeAxis=tAxis, pic=[tPic,1], ushoot=[tus, -0.2],
                     normalize=normalize)
    assert len(h) == len(tAxis)
    return h


def genBezierHRF(timeAxis=np.arange(0,25.5,0.6), pic=[6,1], picw=2,
                 ushoot=[15,-0.2], ushootw=3, normalize=False):
##    print 'genBezierHRF ...'
##    print 'timeAxis :', timeAxis
##    print 'pic :', pic
##    print 'picw:', picw
##    print 'ushoot:', ushoot
##    print 'ushootw:', ushootw
    prec = (timeAxis[1]-timeAxis[0])/20.
    partToPic = bezierCurve([0.,0.], [2.,0.], pic, [pic[0]-picw,pic[1]], prec)
    partToUShoot = bezierCurve(pic, [pic[0]+picw,pic[1]], ushoot,
                               [ushoot[0]-ushootw, ushoot[1]], prec)

    partToEnd = bezierCurve(ushoot, [ushoot[0]+ushootw, ushoot[1]],
                            [timeAxis[-1],0.], [timeAxis[-1]-1,0.], prec)
    hrf = range(2)
    hrf[0] = partToPic[0]+partToUShoot[0]+partToEnd[0]
    hrf[1] = partToPic[1]+partToUShoot[1]+partToEnd[1]

    # check if bezier parameters are well set to generate an injective curve
    #print 'hrf[0]:', hrf[0]
    #print 'diff :', np.diff(np.array(hrf[0]))
    assert (np.diff(np.array(hrf[0]))>=0).all()

    #return hrf
    # Resample time axis :
    iSrc = 0
    resampledHRF = [[],[]]
    for itrgT in xrange(0,len(timeAxis)):
        t = timeAxis[itrgT]
        while (iSrc+1 < len(hrf[0])) and      \
                  (np.abs(t-hrf[0][iSrc]) > np.abs(t-hrf[0][iSrc+1])):
            iSrc += 1

        resampledHRF[0].append(t)
        resampledHRF[1].append(hrf[1][iSrc])
        iSrc += 1

    resampledHRF[0] = np.array(resampledHRF[0])
    hvals = np.array(resampledHRF[1])
    if normalize:
        resampledHRF[1] = hvals / (hvals**2).sum()**.5
    return resampledHRF

def bezierCurve(p1, pc1, p2, pc2, xPrecision):

    bezierCtrlPoints = []
    bezierPoints = []

    bezierCtrlPoints.append(pc1)
    bezierCtrlPoints.append(pc2)
    bezierPoints.append(p1)
    bezierPoints.append(p2)

    precisionCriterion = 0
    nbPoints = 1

    while( precisionCriterion != nbPoints):
        precisionCriterion = 0
        nbPoints = len(bezierPoints)-1
        ip = 0
        ipC = 0
        nbP = 0
        #print 'new pass ...'
        #print 'nbPoints = ', nbPoints
        while(nbP<nbPoints) :
            e1 = bezierPoints[ip]
            e2 = bezierPoints[ip+1]
            c1 = bezierCtrlPoints[ipC]
            c2 = bezierCtrlPoints[ipC+1]

            div2 = [2.,2.]
            m1 = np.divide(np.add(c1,e1),div2)
            m2 = np.divide(np.add(c2,e2),div2)
            m3 = np.divide(np.add(c1,c2),div2)
            m4 = np.divide(np.add(m1,m3),div2)
            m5 = np.divide(np.add(m2,m3),div2)
            m = np.divide(np.add(m4,m5),div2)

            bezierCtrlPoints[ipC] = m1
            bezierCtrlPoints.insert(ipC+1, m5)
            bezierCtrlPoints.insert(ipC+1, m4)
            bezierCtrlPoints[ipC+3] = m2

            bezierPoints.insert(ip+1, m)

            # Stop criterion :
            #print 'deltax :'
            #print m[0]-bezierPoints[ip][0]
            if abs(m[0]-bezierPoints[ip][0]) < xPrecision :
                precisionCriterion += 1
                #print 'precision met !'

            nbP += 1
            ip += 2
            ipC += 4

    return ([p[0] for p in bezierPoints],[p[1] for p in bezierPoints])

