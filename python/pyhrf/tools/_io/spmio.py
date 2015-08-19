# -*- coding: utf-8 -*-

import string

from collections import OrderedDict

import numpy
try:
    from scipy.io.mio import loadmat
except:
    from scipy.io.matlab import loadmat


def load_regnames(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return [s[0] for s in spm['xX'][0, 0]['name'][0, 0][0]]


def loadOnsets(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return get_onsets_from_spm_dict(spm)


def load_paradigm_from_mat(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return (get_onsets_from_spm_dict(spm), get_tr_from_spm_dict(spm))


def load_contrasts_from_mat(spm_mat_file):
    """Returns a dictionary from an SPM.mat file of contrasts and the string
    linear combination of conditions for the corresponding contrast.

    Parameter
    ---------
    spm_mat_file : string
        path to the SPM.mat file to load contrasts from

    Returns
    -------
    contrasts : dict
        dictionary of {contrast_name: linear_condition_combination}
    """

    spm = loadmat(spm_mat_file)
    spm = spm["SPM"]

    conditions = get_conditions_list_from_mat(spm)
    contrasts_vect = get_contrasts(spm)

    return OrderedDict((clean_string_for_xml(str(contrast[0][0])),
                        contrast_vec_to_name(contrast[1][:len(conditions)], conditions))
                       for contrast in contrasts_vect if str(contrast[2][0]) == "T")


def get_conditions_list_from_mat(spm):
    """Returns a list of conditions names from an SPM.mat file with the same
    order.

    Parameter
    ---------
    spm : the SPM field of SPM structure from loadmat

    Returns
    -------
    conditions : list
        list of SPM.mat ordered conditions names
    """

    return [str(condition[0][0][0]) for condition in
                  get_field(get_field(get_field(spm, 'Sess').item()[0][0], "U"),
                            "name")[0]]


def contrast_vec_to_name(contrast_vector, conditions):
    """Returns a string of linear combination of `conditions` strings in respect
    of `contrast_vector` factors.

    Parameters
    ----------
    contrast_vector : array like
        this is a vector of factors of size `len(conditions)`
    conditions : list
        list of conditions names

    Returns
    -------
    linear_combination : string
    """

    return "".join([((str(numpy.sign(factor))[0:-1]+ "+" * bool(factor>0) + conditions[i])
                     * abs(factor)) for i, factor in
                    enumerate(contrast_vector)])


def clean_string_for_xml(name):
    """Return an equivalent of name without any other characters than
    alphanumerics ones, underscores and dashes.

    Parameter
    --------
    name : str

    Returns
    -------
    cleaned_name : str
    """

    return "".join(c for c in "_".join(name.split()) if c in string.letters + string.digits +"-_")


def load_scalefactor_from_mat(spmMatFile):
    d = loadmat(spmMatFile)
    spm = d['SPM']
    return float(get_field(get_field(spm, 'xGX')[0][0], 'gSF')[0][0][0][0])


def get_field(a, f):
    if isinstance(a, (numpy.void, numpy.ndarray)):
        return a[f]
    else:
        return getattr(a, f)


def get_tr_from_spm_dict(spm):
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    return float(get_field(get_field(spm, 'xY')[0][0], 'RT'))


def get_contrasts(spm):

    if isinstance(spm, numpy.ndarray):
        spm = spm[0]
    if isinstance(spm, numpy.ndarray):
        spm = spm[0]

    if isinstance(spm, numpy.void):
        if len(spm['xCon']) > 0:
            cons = spm['xCon'][0]
            return [(c['name'], c['c'].squeeze(), c['STAT']) for c in cons]
        else:
            return []
    elif isinstance(spm, numpy.ndarray):
        cons = spm.xCon[0]
        return [(c.name, c.c.squeeze(), c.STAT) for c in cons]
    else:
        raise Exception("Type of input (%s) is unsupported" %
                        str(spm.__class__))


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
    sessions = get_field(spm, 'Sess').item()[0]

    for iSess, sess in enumerate(sessions):
        ons = {}
        lgth = {}
        fU = get_field(sess, 'U')
        if isinstance(fU, numpy.ndarray) and len(fU.shape) == 2:
            su = fU[0]
        else:
            su = fU

        if numpy.iterable(su):
            su = su
        else:
            su = [su]
        for u in su:
            if isinstance(u, numpy.ndarray):
                u = u[0]
            if isinstance(u, numpy.ndarray):
                u = u[0]
            xBF = get_field(spm, 'xBF')
            if isinstance(xBF, numpy.ndarray):
                xBF = xBF[0]
            if isinstance(xBF, numpy.ndarray):
                xBF = xBF[0]
            if get_field(xBF, 'UNITS') == 'scans':
                tFactor = get_field(get_field(spm, 'xY'), 'RT')
            else:
                tFactor = 1.  # assume onsets in seconds
            name = get_field(u, 'name')
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if isinstance(name, numpy.ndarray):
                name = name[0]
            if name.startswith('sess'):
                iSess = int(name[4]) - 1
                name = name[6:]
            uons = get_field(u, 'ons')
            if len(uons.shape) == 2:
                o = uons[:, 0]
            else:
                o = uons
            udur = get_field(u, 'dur')
            if len(udur.shape) == 2:
                d = udur[:, 0]
            else:
                d = udur
            ons[str(name)] = numpy.array(o, dtype=float) * tFactor
            lgth[str(name)] = numpy.array(d, dtype=float) * tFactor
        allOnsets['session' + str(iSess + 1)] = {'onsets': ons,
                                                 'stimulusLength': lgth}
    return allOnsets


def get_onsets_from_spm_dict_child(spm):
    allOnsets = {}

    sessions = get_field(spm, 'Sess').item()[0]

    print 'sessions:', sessions
    for iSess, sess in enumerate(sessions):
        print 'sess :', dir(sess)
        ons = {}
        lgth = {}
        fU = get_field(sess[0], 'u')
        name = get_field(sess[0], 'name')

        uons = get_field(sess[0], 'ons')
        print 'uons:', uons
        if len(uons.shape) == 2:
            o = uons[:, 0]
        else:
            o = uons
        udur = get_field(u, 'dur')
        print 'udur:', udur, udur.shape
        if len(udur.shape) == 2:
            d = udur[:, 0]
        else:
            d = udur
        xBF = get_field(spm, 'xBF')
        if isinstance(xBF, numpy.ndarray):
            xBF = xBF[0]
        if isinstance(xBF, numpy.ndarray):
            xBF = xBF[0]
        if get_field(xBF, 'UNITS') == 'scans':
            tFactor = get_field(get_field(spm, 'xY'), 'RT')
        else:
            tFactor = 1.  # assume onsets in seconds
            ons[str(name)] = numpy.array(o, dtype=float) * tFactor
            lgth[str(name)] = numpy.array(d, dtype=float) * tFactor
        allOnsets['session' + str(iSess + 1)] = {'onsets': ons,
                                                 'stimulusLength': lgth}
    return allOnsets
