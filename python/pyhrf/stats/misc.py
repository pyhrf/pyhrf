# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import trapz
from pyhrf.tools import resampleToGrid


def compute_T_Pvalue(betas, stds_beta, mask_file, null_hyp=True):
    '''
    Compute Tvalues statistic and Pvalue based upon estimates
    and their standard deviation
    beta and std_beta for all voxels
    beta: shape (nb_vox, 1)
    std: shape (1)
    Assume null hypothesis if null_hyp is True
    '''
    from pyhrf.ndarray import xndarray

    import sys
    sys.path.append("/home/i2bm/BrainVisa/source/pyhrf/pyhrf-free/trunk/script/WIP/Scripts_IRMf_Adultes_Solv/Scripts_divers_utiles/Scripts_utiles/")
    from Functions_fit import Permutation_test, stat_mean, stat_Tvalue, stat_Wilcoxon

    mask = xndarray.load(mask_file).data #to save P and Tval on a map

    BvalC = xndarray(betas, axes_names=['sagittal', 'coronal', 'axial'])
    Betasval = BvalC.flatten(mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data

    Stdsval = stds_beta

    Tval = xndarray(Betasval/Stdsval, axes_names=['position']).data

    nb_vox = Betasval.shape[0]
    nb_reg = betas.shape[1]
    dof = nb_vox - nb_reg #degrees of freedom for STudent distribution
    assert dof>0

    Probas=np.zeros(Betasval.shape)
    for i in xrange(nb_vox):
        if null_hyp:
            #STudent distribution
            from scipy.stats import t
            fmix = lambda x: t.pdf(x, dof)
        else:
            fmix = lambda t:  1/np.sqrt(2*np.pi*Stdsval[i]**2)*np.exp(- (t - Betasval[i])**2 / (2*Stdsval[i]**2) )
        Probas[i] = quad(fmix, Tval[i], float('inf'))[0]

    Tvalues_ = xndarray(Tval, axes_names=['position'])
    Pvalues_ = xndarray(Probas, axes_names=['position'])
    Tvalues = Tvalues_.expand(mask, 'position', ['sagittal','coronal','axial'])
    Pvalues = Pvalues_.expand(mask, 'position', ['sagittal','coronal','axial'])

    #Computation of Pvalue using permutations
    #not possible to do this actually...it was used for group level stats
    #Pvalue_t = np.zeros(Betasval.shape)
    #for i in xrange(nb_vox):
        #Pvalue_t[i] = Permutation_test(Betasval[i], n_permutations=10000, \
                    #stat = stat_Tvalue, two_tailed=False, plot_histo=False)

    return Tvalues.data, Pvalues.data

def compute_roc_labels_scikit(e_labels, true_labels):
    from scikits.learn.metrics import roc_curve
    from scikits.learn.metrics import auc as compute_auc
    from scipy import interp
    auc = []
    sensData = []
    specData = []
    grid_fpr = np.linspace(0,1,100)
    for j in xrange(true_labels.shape[0]):
        evalues = e_labels[j,:]
        tvalues = true_labels[j,:]
        # print 'evalues:', evalues.shape
        # print evalues
        # print 'tvalues:', tvalues.shape
        # print tvalues
        fpr, tpr, thresh = roc_curve(tvalues, evalues)
        # print 'fpr:'
        # print fpr
        # print 'tpr:'
        # print tpr
        # print 'thresh:'
        # print thresh
        auc.append(compute_auc(fpr, tpr))
        # print 'auc:', auc[-1]
        sensData.append(interp(grid_fpr, fpr, tpr))
        specData.append(grid_fpr)
        sensData[-1][0] = 0.
        # print 'sensData:'
        # print sensData[-1]
        # print 'specData:'
        # print specData[-1]
    sensData = np.array(sensData)
    specData = np.array(specData)
    auc = np.array(auc)
    return sensData, specData, auc

def threshold_labels(labels, thresh=None, act_class=1):
#def threshold_labels(labels, thresh, act_class=1):
    """
    Threshold input labels which are assumed being of shape
    (nb classes, nb conds, nb vox).
    If thresh is None then take the argmax over classes.
    Else use it on labels for activating class (act_class), suitable
    for the 2class case only.
    """
    finalLabels = np.zeros(labels.shape[1:], dtype=np.int32)
    for j in xrange(finalLabels.shape[0]):
        if thresh is None:
            finalLabels[j,:] = np.argmax(labels[:,j,:], axis=0)
        else:
            actVox = np.where(labels[act_class,j,:]>=thresh)
            finalLabels[j,actVox] = act_class
    return finalLabels

def mark_wrong_labels(labels, true_labels, lab_ca=1, lab_ci=0,false_pos=2,
                      false_neg=3):

    new_labels = labels.copy()
    for j in xrange(labels.shape[0]):
        #print 'labels :'
        #print labels[j,:]
        #print 'trueLabels '
        #print self.trueLabels[j,:]
        el = (labels[j,:] == lab_ca)
        tl = (true_labels[j,:] == lab_ca)
        nel = np.bitwise_not(el)
        ntl = np.bitwise_not(tl)
        new_labels[j, np.bitwise_and(el,tl)] = lab_ca
        new_labels[j, np.bitwise_and(nel,ntl)] = lab_ci
        new_labels[j, np.bitwise_and(el,ntl)] = false_pos
        new_labels[j, np.bitwise_and(nel,tl)] = false_neg
        #print '-> marked :'
        #print labels[j,:]

    return new_labels

def compute_roc_labels(mlabels, true_labels, dthres=0.005, lab_ca=1, lab_ci=0,
                false_pos=2, false_neg=3):
    thresholds = np.arange(0,1/dthres) * dthres
    nconds = true_labels.shape[0]
    oneMinusSpecificity = np.zeros((nconds, len(thresholds)))
    sensitivity = np.zeros((nconds, len(thresholds)))
    for it,thres in enumerate(thresholds):
	#print 'thres:',thres
        labs = threshold_labels(mlabels,thres)
        tlabels = labs.copy()
        labs = mark_wrong_labels(labs,true_labels,lab_ca, lab_ci, false_pos, false_neg)
        for cond in xrange(nconds):
            if 0 and cond == 1:
                print "**cond %d **" %cond
                print 'thresh:', thres
                print 'mlabels (activ class):'
                print mlabels[1,cond,:80]
                print 'thresholded labels:'
                print tlabels[cond,:80]
                print 'simulated labels:'
                print true_labels[cond,:80]
                print 'marked labels:'
                print labs[cond,:80]
            counts = np.bincount(labs[cond,:])
            nbTrueNeg = counts[0]
            nbTruePos = counts[1] if len(counts)>1 else 0

            fp = false_pos
            nbFalsePos = counts[fp] if len(counts)>fp else 0

            fn = false_neg
            nbFalseNeg = counts[fn] if len(counts)>fn else 0

            #if cond == 1 or cond == 2:
	      #print 'cond ', cond
	      #print 'nbTrueNeg=',nbTrueNeg
	      #print 'nbTruePos=',nbTruePos
	      #print 'nbFalsePos=',nbFalsePos
	      #print 'nbFalseNeg=',nbFalseNeg

            if 0 and cond == 1:
                print 'TN :', nbTrueNeg
                print 'TP :', nbTruePos
                print 'FP :', nbFalsePos
                print 'FN :', nbFalseNeg
            if nbTruePos == 0:
                sensitivity[cond,it] = 0
            else:
                sensitivity[cond,it] = nbTruePos /                    \
                                       (nbTruePos+nbFalseNeg+0.0)
            spec = nbTrueNeg/(nbTrueNeg+nbFalsePos+0.0)
            oneMinusSpecificity[cond,it] = 1-spec
            if 0 and cond == 1:
                print '-> se = ', sensitivity[cond, it]
                print '-> 1-sp = ', oneMinusSpecificity[cond,it]

    if 1:
        spGrid = np.arange(0.,1.,0.0005)
        omspec = np.zeros((nconds, len(spGrid)))
        sens = np.zeros((nconds, len(spGrid)))
        for cond in xrange(nconds):
            order = np.argsort(oneMinusSpecificity[cond,:])
            if oneMinusSpecificity[cond,order][0] != 0.:
                osp = np.concatenate(([0.],oneMinusSpecificity[cond,order]))
                se = np.concatenate(([0.],sensitivity[cond,order]))
            else:
                osp = oneMinusSpecificity[cond,order]
                se = sensitivity[cond,order]

            if osp[-1] != 1.:
                osp = np.concatenate((osp,[1.]))
                se = np.concatenate((se,[1.]))

            sens[cond,:] = resampleToGrid(osp, se, spGrid)
            omspec[cond, :] = spGrid
            if 0 and cond == 1:
                print '-> (se,1-spec) :'
                print zip(se,osp)
                print '-> se resampled :'
                print sens[cond,:]
                print 'spec grid :'
                print spGrid


    else:
        sens = sensitivity
        omspec = oneMinusSpecificity

    auc = np.array([trapz(sens[j,:], omspec[j,:])
                    for j in xrange(nconds)])

    #Computing the area under ROC curve (with John Walkenbach formula) (SAME AS auc)
    #area_under_ROC_curve = np.zeros(nconds, dtype=float)
    #for cond in xrange(nconds):
      #for i in xrange(len(spGrid)-1):
	#if sens[cond,i]*sens[cond,i+1] >= 0.:
	  #area_under_ROC_curve[cond] += ((sens[cond,i+1] + sens[cond,i])/2.) * (omspec[cond,i+1] - omspec[cond,i])
	#else:
	  #area_under_ROC_curve[cond] += ((sens[cond,i+1]**2 + sens[cond,i]**2)/((sens[cond,i+1] - sens[cond,i])/2.)) * (omspec[cond,i+1] - omspec[cond,i])

    return sens, omspec, auc #, area_under_ROC_curve


def cumFreq(data, thres=None):

    nBins = data.ptp()/(3.5*data.std()/(len(data)**.33)) # Scott's choice
    #nBins = np.around(np.log2(len(data))+1) # Sturges' formula

    h,b = np.histogram(data, bins=nBins,normed=True)
    ccBins = (b[:-1]+b[1:])/2. # class center approximation
##    print 'ccBins :'
##    print ccBins

    if thres != None:
        # Find bin where contrast <= thres :
        infT = (ccBins <= thres)+0
##        print 'infT :'
##        print infT
        if infT.sum() == len(ccBins):
            iThres = -1 # take all
        else:
            iThres = infT.tolist().index(0)
    else:
        iThres = -1 # take all
##    print 'iThres :', iThres
    # cut ccH at given threshold :
    h = h[:iThres]
    ccBins = ccBins[:iThres]

##    print 'H :'
##    print h
##    print 'ccBins :'
##    print ccBins


    # Append zeros at begining and end to finalize approximation:
    h = np.concatenate(([0],h,[0]))
    ccBins = np.concatenate(([b[0]-(b[1]-b[0])/2.0],
                             ccBins,
                             [b[iThres]+(b[iThres]-b[iThres-1])/2.0]))

    # Approximate the cumulative density function under given threshold:
    return np.trapz(h, ccBins)



#Functions to manipulate CDF, PDF, ...
from scipy.integrate import quad
from scipy.stats import norm

def gm_cdf(x, means, variances, props):
    """ Compute the cumulative density function of gaussian mixture,
    ie: p(x<a) = \sum_i \Nc(mean_i, variance_i)
    """
    variances[np.where(variances==0.)] = 1e-6
    return (props * norm.cdf(x, means, variances**.5)).sum(0)

def gm_mean(means, variances, props):
    if isinstance(means,list):
        means = np.array(means)

    if isinstance(props,list):
        props = np.array(props)

    return (props*means).sum()

def gm_var(means, variances, props):
    if isinstance(means,list):
        means = np.array(means)

    if isinstance(variances,list):
        variances = np.array(variances)

    if isinstance(props,list):
        props = np.array(props)

    return (props*(means**2+variances)).sum() - gm_mean(means,variances,props)**2


from scipy.stats.mstats import mquantiles
def cpt_ppm_a_mcmc(samples, alpha=0.05):
    """ Compute a Posterior Probability Map (fixed alpha) from NRL MCMC samples.
    Expected shape of 'samples': (sample, voxel)
    """
    return mquantiles(samples, prob=[1-alpha], axis=0)[0]

def cpt_ppm_g_mcmc(samples, gamma=0.):
    """ Compute a Posterior Probability Map (fixed gamma) from NRL MCMC samples.
    Expected shape of 'samples': (sample, voxel)
    """
    return (samples > gamma).mean(0)

def cpt_ppm_g_apost(means, variances, props, gamma=0.):
    """ Compute a Posterior Probability Map (fixed gamma) from posterior
    gaussian mixture components estimates.
    Expected shape of 'means', 'variances' and 'probs': (nb_classes, voxel)
    """
    return 1 - gm_cdf(gamma, means, variances, props)

def cpt_ppm_a_apost(means, variances, props, alpha=0.05):
    raise NotImplementedError('PPM with fixed alpha from posterior mixture '\
                                  'components is not implemented')

def cpt_ppm_a_norm(mean, variance, alpha=0.):
    """ Compute a Posterior Probability Map (fixed alpha) by assuming a Gaussian
    distribution.

    Parameters
    ----------
    mean : array_like
        mean value(s) of the Gaussian distribution(s)
    variance : array_like
        variance(s) of the Gaussian distribution(s)
    alpha : array_like, optional
        quantile value(s) (default=0)

    Returns
    -------
    ppm : array_like
        Posterior Probability Map evaluated at alpha
    """

    return norm.sf(alpha, mean, variance**.5)

def cpt_ppm_g_norm(mean, variance, gamma=0.95):
    """ Compute a Posterior Probability Map (fixed gamma) by assuming a Gaussian
    distribution.

    Parameters
    ----------
    mean : array_like
        mean value(s) of the Gaussian distribution(s)
    variance : array_like
        variance(s) of the Gaussian distribution(s)
    gamma : array_like, optional
        upper tail probability (default=0.95)

    Returns
    -------
    ppm : ndarray or scalar
        Posterior Probability Map corresponding to the upper tail probability gamma
    """

    return norm.isf(gamma, mean, variance**.5)


# def compute_bigaussian_ppm(threshold, mean_c1, var_c1, prop_c1,
#                            mean_c2, var_c2, prop_c2, thresh_type='value'):
#     """
#     Calculate a posterior probability map p(A>gamma)=alpha where A follows a
#     bigaussian mixture defined by the input parameters:
#     *prop_c1* Normal(*mean_c1*,*var_c1*) +
#     *prop_c2* Normal(*mean_c2*,*var_c2*).

#     *thesh_type* can be either 'value' or 'proba'.
#     If *thesh_type*='value' then the returned PPM is p(X>threshold).
#     If *thesh_type*='proba' then the returned PPM is gamma so that
#        p(X>gamma)=threshold
#     """
#     ppm = np.zeros_like(mean_c1)

#     if thresh_type == 'value':
#         def fmix(t, m1, v1, p1, m2, v2, p2):
#             val =  p1 * 1/np.sqrt(2*np.pi*v1) * \
#                 np.exp(- (t - m1)**2 / (2*v1) )  + \
#                 p2 * 1/np.sqrt(2*np.pi*v2) * \
#                 np.exp(- (t - m2)**2 / (2*v2) )
#             return val
#         for pos in xrange(len(ppm)):
#             ppm[pos] = quad(fmix, threshold, float('inf'),
#                             (mean_c1[pos], var_c1[pos], prop_c1[pos],
#                              mean_c2[pos], var_c2[pos], prop_c2[pos],)
#                             )[0]

#     elif thresh_type == 'proba':
#         raise NotImplementedError('PPM computation with proba threshold')
#     else:
#         raise Exception('Wrong thresh_type "%s", has to be '\
#                             '"value" or "proba"' %thresh_type)

#     return ppm


def acorr(x, maxlags=10, scale='var'):
    #print 'acorr ..., input x:', x.shape
    n = x.shape[0]

    maxlags = min(n-1, maxlags)

    mean = x.mean(0)
    v = ((x - mean) ** 2).sum(0) / float(n)

    #print 'c0:', v.shape

    def r(h):
        return ((x[:n - h] - mean) * (x[h:] - mean)).sum(0)

    lags = np.arange(maxlags) + 1
    ac = np.array(map(r, lags))

    #print 'ac:', ac.shape
    if scale == 'coeff':
        ac /= ac[0]
    elif scale == 'var':
        ac /= float(n) * v


    m_cst_signal = np.where(v==0.)
    for a in ac:
        a[m_cst_signal] = 1.

    return ac



    # x = detrend_mean(np.asarray(x))
    # c = np.correlate(x, x, mode=2)
    # c /= np.sqrt(np.dot(x,x)**2)


    # if maxlags is None: maxlags = Nx - 1

    # if maxlags >= Nx or maxlags < 1:
    #     raise ValueError('maxlags must be None or strictly '
    #                      'positive < %d'%Nx)

    # lags = np.arange(-maxlags,maxlags+1)
    # c = c[Nx-1-maxlags:Nx+maxlags]

