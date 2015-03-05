# -*- coding: utf-8 -*-
import numpy
try:
    from scipy.io.mio import loadmat
except:
    from scipy.io.matlab import loadmat

def load_regnames(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return [s[0] for s in spm['xX'][0,0]['name'][0,0][0]]

def loadOnsets(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']

    return get_onsets_from_spm_dict(spm)

def load_paradigm_from_mat(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return (get_onsets_from_spm_dict(spm), get_tr_from_spm_dict(spm))

def load_contrasts(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return get_contrasts(spm)

def load_scalefactor_from_mat(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return float(get_field(get_field(spm,'xGX')[0][0],'gSF')[0][0][0][0])

def get_field(a, f):
    if isinstance(a, numpy.void):
        return eval('a["%s"]' %f)
    elif isinstance(a, numpy.ndarray):
        return eval('a["%s"]' %f)
    else:
        return eval('a.%s' %f)


def get_tr_from_spm_dict(spm):
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    return float(get_field(get_field(spm,'xY')[0][0],'RT'))

def get_contrasts(spm):
    
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]

    if isinstance(spm, numpy.void):
        if len(spm['xCon']) > 0:
            cons = spm['xCon'][0]
            return [ (c['name'], c['c'].squeeze(), c['STAT']) for c in cons ]
        else:
            return []
    elif isinstance(spm, numpy.ndarray):
        cons = spm.xCon[0]
        return [ (c.name, c.c.squeeze(), c.STAT) for c in cons ]
    else:
        raise Exception("Type of input (%s) is unsupported" %str(spm.__class__))

def get_onsets_from_spm_dict(spm):
    """
    Read paradigm from SPM structure loaded by loadmat from scipy.


    Args:
        - spm (dict): SPM structure loaded from an SPM.mat

    Return:
        - the paradigm (dict), such as:
            { <session> : { 'onsets': { <condition> : array of stim onsets },
                            'stimulusLength': { <condition> :
                                                     array of stim durations}
                          }
            }

    TODO: unit test
    """

    allOnsets = {}
    #if isinstance(spm, numpy.ndarray):
        #spm = spm[0]
    #if isinstance(spm, numpy.ndarray):
        #spm = spm[0]
    #if isinstance(spm, numpy.void):
        #if isinstance(spm['Sess'], numpy.ndarray) and \
                #len(spm['Sess'].shape) == 2:
            #sessions = spm['Sess'][0]
        #else:
            #sessions = spm['Sess']
    #else:
        #if isinstance(spm.Sess, numpy.ndarray) and len(spm.Sess.shape) == 2:
            #sessions = spm.Sess[0]
        #else:
            #sessions = spm.Sess

    #if numpy.iterable(sessions):
        #sessions = sessions
    #else:
        #sessions = [sessions]

    sessions=get_field(spm, 'Sess').item()[0]

    #'sessions:', sessions
    for iSess,sess in enumerate(sessions):
        #print 'sess :', dir(sess)
        #Bprint 'sess.U:'
        #print sess.U
        ons = {}
        lgth = {}
        fU = get_field(sess,'U')
        #print 'fU:', fU
        if isinstance(fU, numpy.ndarray) and len(fU.shape) == 2:
            su = fU[0]
        else:
            su = fU

        if numpy.iterable(su):
            su = su
        else:
            su = [su]
        #print 'su', su
        for u in su:
            if isinstance(u, numpy.ndarray):
                u = u[0]
            if isinstance(u, numpy.ndarray):
                u = u[0]
            xBF = get_field(spm,'xBF')
            if isinstance(xBF, numpy.ndarray):
                xBF = xBF[0]
            if isinstance(xBF, numpy.ndarray):
                xBF = xBF[0]
            if get_field(xBF,'UNITS') == 'scans':
                tFactor = get_field(get_field(spm,'xY'),'RT')
            else:
                tFactor = 1. # assume onsets in seconds
            #print 'u.name :', u.name
            #print 'u.ons:', u.ons
            name = get_field(u,'name')
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if name.startswith('sess'):
                iSess = int(name[4]) -1
                name = name[6:]
            #print 'name:', name
            #print 'u.ons:'
            #print u.ons.shape
            uons = get_field(u,'ons')
            #print 'uons:', uons
            if len(uons.shape) == 2:
                o = uons[:,0]
            else:
                o = uons
            udur = get_field(u,'dur')
            #print 'udur:', udur, udur.shape
            if len(udur.shape) == 2:
                d = udur[:,0]
            else:
                d = udur
            #print 'd:', d
            ons[str(name)] = numpy.array(o,dtype=float)*tFactor
            lgth[str(name)] = numpy.array(d,dtype=float)*tFactor
            #print 'ons :', ons[u.name]
        allOnsets['session'+str(iSess+1)] = {'onsets':ons,
                                             'stimulusLength':lgth}

    return allOnsets


def get_onsets_from_spm_dict_child(spm):
    allOnsets = {}

    sessions=get_field(spm, 'Sess').item()[0]

    print 'sessions:', sessions
    for iSess,sess in enumerate(sessions):
        print 'sess :', dir(sess)
        #Bprint 'sess.U:'
        #print sess.U
        ons = {}
        lgth = {}
        fU = get_field(sess[0],'u')
        name = get_field(sess[0], 'name')

        uons = get_field(sess[0],'ons')
        print 'uons:', uons
        if len(uons.shape) == 2:
            o = uons[:,0]
        else:
            o = uons
        udur = get_field(u,'dur')
        print 'udur:', udur, udur.shape
        if len(udur.shape) == 2:
            d = udur[:,0]
        else:
            d = udur
        xBF = get_field(spm,'xBF')
        if isinstance(xBF, numpy.ndarray):
            xBF = xBF[0]
        if isinstance(xBF, numpy.ndarray):
            xBF = xBF[0]
        if get_field(xBF,'UNITS') == 'scans':
            tFactor = get_field(get_field(spm,'xY'),'RT')
        else:
            tFactor = 1. # assume onsets in seconds
            ons[str(name)] = numpy.array(o,dtype=float)*tFactor
            lgth[str(name)] = numpy.array(d,dtype=float)*tFactor
            #print 'ons :', ons[u.name]
        allOnsets['session'+str(iSess+1)] = {'onsets':ons,
                                             'stimulusLength':lgth}

        #HACK: take only first session
        #print 'kept session', iSess+1
        #break

    return allOnsets
