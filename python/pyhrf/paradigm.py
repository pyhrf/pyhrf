# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import pyhrf
import numpy as np
from pyhrf.verbose import dictToString
from pyhrf.tools import apply_to_leaves
from pyhrf.tools import stack_trees, unstack_trees, PickleableStaticMethod
from pyhrf.tools.io.spmio import load_paradigm_from_mat
from pyhrf.tools.io import load_paradigm_from_csv

from numpy.testing import assert_array_equal
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm, BlockParadigm
from scipy.io.matlab import savemat

def merge_onsets(onsets, new_condition, criterion=None, durations=None,
                 discard=None):
    """
     Convention for definition of onsets or durations:
     OrderedDict({
         'condition_name' : [ <array of timings for sess1>,
                              <array of timings for sess2>,
                              ...]
      }
    """

    if discard is None: discard = []

    if criterion is None: criterion = lambda x: new_condition in x

    new_onsets = onsets.__class__() #might be dict or OrderedDict
    if durations is not None: new_durations = onsets.__class__()

    nb_sessions = len(onsets[onsets.keys()[0]])
    merged_onsets = [np.array([]) for s in range(nb_sessions)]
    merged_durations = [np.array([]) for s in range(nb_sessions)]

    for cname, all_ons in onsets.iteritems():
        for isess, ons in enumerate(all_ons):
            if durations is not None:
                dur = durations[cname][isess]
            if criterion(cname):
                merged_onsets[isess] = np.concatenate((merged_onsets[isess],ons))
                if durations is not None:
                    md = np.concatenate((merged_durations[isess],dur))
                    merged_durations[isess] = md
            elif cname not in discard:
                if isess==0:
                    new_onsets[cname] = []
                    if durations is not None:
                        new_durations[cname] = []
                new_onsets[cname].append(ons)
                if durations is not None:
                    new_durations[cname].append(dur)

    for isess in range(nb_sessions):
        sorting = np.argsort(merged_onsets[isess])
        merged_onsets[isess] = merged_onsets[isess][sorting]
        if durations is not None:
            merged_durations[isess] = merged_durations[isess][sorting]

    new_onsets[new_condition] = merged_onsets

    if durations is not None:
        new_durations[new_condition] = merged_durations
        return new_onsets, new_durations
    else:
        return new_onsets

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


builtin_paradigms = []


# Convention for paradigm definition:
   # OrderedDict({
   #     'condition_name' : [ <array of timings for sess1>,
   #                          <array of timings for sess2>,
   #                          ...]
   #  }

onsets_loc = OrderedDict([
        ('calculaudio', [np.array([  35.4,   44.7, 48. , 83.4,  108., 135.,
                                    137.7,  173.7,
                                    191.7,  251.7])]
            ),#0
        ('calculvideo', [np.array([   0.,    2.4, 125.4, 153., 164.4,  198. ,
                                     201. ,  221.4,
                                     234. ,  260.4])]
            ),#1
        ('clicDaudio', [np.array([  11.4,   87. ,  143.4,  162. ,  230. ])]
            ),#2
        ('clicDvideo', [np.array([  18. ,   69. ,  227.7,  275.4,  291. ])]
            ),#3
        ('clicGaudio', [np.array([  23.7,   62.4,  170.4,  254.7,  257.4])]
            ),#4
        ('clicGvideo', [np.array([  26.7,   71.4,  116.7,  212.7,  215.7])]
            ),#5
        ('damier_H', [np.array([   8.7,   59.7,  149.4,  176.7,  188.4,  218.7,
                                  224.7, 248.4, 266.7,  269.7])]
            ),#6
        ('damier_V', [np.array([  33. ,   96. ,  122.7,  140.4,  156. ,  203.7,
                                 207. , 210. , 264. ,  278.4])]
            ),#7
        ('phraseaudio', [np.array([  15. ,   20.7,   29.7,   89.7,  119.4,  146.7,
                                    236.7,  284.4,
                                293.4,  296.7])]
            ),#8
        ('phrasevideo', [np.array([  39. ,   41.7,   56.4,   75. ,  131.4,  159. ,
                                    167.7,  195. ,
                                    246. ,  288. ])]
            ) #9
        ])

onsets_loc = OrderedDict([
        ('calculaudio', [np.array([  35.4,   44.7, 48., 83.4,  108., 135. ,  #0
                                    137.7,  173.7,
                                    191.7,  251.7])]
                                    ),
        ('calculvideo', [np.array([   0. ,    2.4,  125.4,  153. ,  164.4,  198. ,  #1
                                     201. ,  221.4,
                                     234. ,  260.4])]),
        ('clicDaudio', [np.array([  11.4,   87. ,  143.4,  162. ,  230. ])]),#2
        ('clicDvideo', [np.array([  18. ,   69. ,  227.7,  275.4,  291. ])]),#3
        ('clicGaudio', [np.array([  23.7,   62.4,  170.4,  254.7,  257.4])]),#4
        ('clicGvideo', [np.array([  26.7,   71.4,  116.7,  212.7,  215.7])]),#5
        ('damier_H', [np.array([   8.7,   59.7,  149.4,  176.7,  188.4,  218.7,
                                  224.7, 248.4, 266.7,  269.7])]),#6
        ('damier_V', [np.array([  33. ,   96. ,  122.7,  140.4,  156. ,  203.7,
                                 207. , 210. , 264. ,  278.4])]),#7
        ('phraseaudio', [np.array([  15. ,   20.7,   29.7,   89.7,  119.4,  146.7,
                                    236.7,  284.4,
                                293.4,  296.7])]),#8
        ('phrasevideo', [np.array([  39. ,   41.7,   56.4,   75. ,  131.4,  159. ,
                                    167.7,  195. ,
                                    246. ,  288. ])]),#9
        ])

builtin_paradigms.append('loc')

onsets_language_sess1 = dict([
    ('LangForeign', [np.array([  24.026,   72.026,  120.026,  144.026])]),
    ('LangForeignRep',  [np.array([  36.026,   84.026,  132.026,  156.026])]),
    ('LangMat',  [np.array([  2.60000000e-02,   4.80260000e+01,   9.60260000e+01,
                                                1.68026000e+02])]),
    ('LangMatRep',  [np.array([  12.026,   60.026,  108.026,  180.026])]),
    ])

durations_language_sess1 = dict([
    ('LangForeign',  [np.array([ 3.021,  2.831,  2.587,  2.325])]),
    ('LangForeignRep',  [np.array([ 3.021,  2.831,  2.587,  2.325])]),
    ('LangMat',  [np.array([ 2.916,  2.753,  2.602,  2.501])]),
    ('LangMatRep',  [np.array([ 2.916,  2.753,  2.602,  2.501])]),
    ])


onsets_language_sess2 = dict([
    ('LangForeign',  [np.array([  48.022,   72.022,   96.022,  120.022])]),
    ('LangForeignRep',  [np.array([  60.022,   84.022,  108.022,  132.022])]),
    ('LangMat',  [np.array([  2.20000000e-02,   2.40220000e+01,   1.44022000e+02,
                                                1.68022000e+02])]),
    ('LangMatRep',  [np.array([  12.022,   36.022,  156.022,  180.022])]),
    ])

durations_language_sess2 = dict([
    ('LangForeign',  [np.array([ 2.362,  2.662,  2.643,  2.769])]),
    ('LangForeignRep',  [np.array([ 2.362,  2.662,  2.643,  2.769])]),
    ('LangMat',  [np.array([ 2.839,  2.688,  2.973,  2.774])]),
    ('LangMatRep',  [np.array([ 2.839,  2.688,  2.973,  2.774])]),
    ])


onsets_language_sess3 = dict([
    ('LangForeign',  [np.array([  2.90000000e-02,   2.40290000e+01,   7.20290000e+01,
                                                1.44029000e+02])]),
    ('LangForeignRep',  [np.array([  12.029,   36.029,   84.029,  156.029])]),
    ('LangMat',  [np.array([  48.029,   96.029,  120.029,  168.029])]),
    ('LangMatRep',  [np.array([  60.029,  108.029,  132.029,  180.029])]),
    ])

durations_language_sess3 = dict([
    ('LangForeign',  [np.array([ 2.766,  2.625,  2.831,  2.327])]),
    ('LangForeignRep',  [np.array([ 2.766,  2.625,  2.831,  2.327])]),
    ('LangMat',  [np.array([ 2.75 ,  2.622,  2.847,  2.67 ])]),
    ('LangMatRep',  [np.array([ 2.75 ,  2.622,  2.847,  2.67 ])]),
    ])


onsets_language_sess4 = dict([
    ('LangForeign',  [np.array([  3.10000000e-02,   7.20310000e+01,   1.20031000e+02,
                                                1.68031000e+02])]),
    ('LangForeignRep',  [np.array([  12.031,   84.031,  132.031,  180.031])]),
    ('LangMat',  [np.array([  24.031,   48.031,   96.031,  144.031])]),
    ('LangMatRep',  [np.array([  36.031,   60.031,  108.031,  156.031])]),
    ])

durations_language_sess4 = dict([
    ('LangForeign',  [np.array([ 3.292,  2.53 ,  2.945,  2.775])]),
    ('LangForeignRep',  [np.array([ 3.292,  2.53 ,  2.945,  2.775])]),
    ('LangMat',  [np.array([ 2.819,  2.771,  2.736,  2.492])]),
    ('LangMatRep',  [np.array([ 2.819,  2.771,  2.736,  2.492])]),
    ])

builtin_paradigms.extend('language_sess%d'%i for i in range(1,5))


#Errors here... ??
#onsets_language_sess1 = OrderedDict([
        #('LangMat', np.array([ 0.025, 48.025, 120.025, 168.025]))
        #('LangMatRep', np.array([ 12.025, 60.025, 132.025, 180.025])),
        #('LangForeign', np.array([  24.025, 72.025, 96.025, 144.025])),#2
        #('LangForeignRep', np.array([  36.025, 84.025, 108.025, 156.025]))
        #])

#durations_language_sess1 = OrderedDict([
    #('LangMat', np.array([ 2.625, 2.662, 3.292, 3.224])),
    #('LangMatRep', np.array([ 2.625, 2.662, 3.292, 3.224])),
    #('LangForeign', np.array([ 2.819, 2.534, 2.602, 2.67])),#2
    #('LangForeignRep', np.array([ 2.819, 2.534, 2.602, 2.67]))
    #])


#onsets_language_sess2 = OrderedDict([
        #('LangMat', np.array([ 24.038, 72.038, 96.038, 120.038])),
        #('LangMatRep', np.array([ 36.038, 84.038, 108.038, 132.038 ])),
        #('LangForeign', np.array([ 0.038, 48.038, 144.038, 168.038 ])),#2
        #('LangForeignRep', np.array([ 12.038, 60.038, 156.038, 180.038 ]))
        #])

#durations_language_sess2 = OrderedDict([
    #('LangMat', np.array([ 3.021, 2.025, 2.769, 2.945])),
    #('LangMatRep', np.array([ 3.021, 2.025, 2.769, 2.945])),
    #('LangForeign', np.array([2.688, 2.501, 2.839, 2.75 ])),#2
    #('LangForeignRep', np.array([ 2.688, 2.501, 2.839, 2.75]))
    #])


#onsets_language_sess3 = OrderedDict([
        #('LangMat', np.array([24.03, 120.03, 144.03, 168.03 ])),
        #('LangMatRep', np.array([ 36.03, 132.03, 156.03, 180.03 ])),
        #('LangForeign', np.array([ 0.03, 48.03, 72.03, 96.03 ])),#2
        #('LangForeignRep', np.array([ 12.03, 60.03, 84.03, 108.03 ]))
        #])

#durations_language_sess3 = OrderedDict([
    #('LangMat', np.array([ 2.53, 2.662, 2.831, 2.766])),
    #('LangMatRep', np.array([ 2.53, 2.662, 2.831, 2.766])),
    #('LangForeign', np.array([ 2.973, 2.622, 2.783, 2.7])),#2
    #('LangForeignRep', np.array([ 2.973, 2.622, 2.783, 2.7]))
    #])


#onsets_language_sess4 = OrderedDict([
        #('LangMat', np.array([ 0.036, 48.036, 72.036, 120.036])),
        #('LangMatRep', np.array([ 12.036, 60.036, 84.036, 132.036])),
        #('LangForeign', np.array([ 24.036, 96.036, 144.036, 168.036 ])),#2
        #('LangForeignRep', np.array([ 36.036, 108.036, 156.036, 180.036]))
        #])

#durations_language_sess4 = OrderedDict([
    #('LangMat', np.array([ 2.831, 2.327, 2.587, 2.936])),
    #('LangMatRep', np.array([ 2.831, 2.327, 2.587, 2.936])),
    #('LangForeign', np.array([ 2.771, 2.753, 2.715, 2.774])),#2
    #('LangForeignRep', np.array([ 2.771, 2.753, 2.715, 2.774]))
    #])


durations_loc = apply_to_leaves(onsets_loc, lambda x: 2. + np.zeros_like(x))

default_contrasts_loc = {
    "left_click" : "clicGaudio + clicGvideo",
    "right_click" : "clicDaudio + clicDvideo",
    "checkerboard_H-V" : "damier_H - damier_V",
    "checkerboard_V-H" : "damier_V - damier_H",
    "left_click-right_click" : "clicGaudio-clicDaudio + clicGvideo-clicDvideo",
    "right_click-left_click" : "clicDaudio-clicGaudio + clicDvideo-clicGvideo",
    "computation-sentences" : "calculaudio-phraseaudio + calculvideo-phrasevideo",
    "sentences-computation" : "phraseaudio-calculaudio + phrasevideo-calculvideo",
    "video_computation-sentences" : "calculvideo-phrasevideo",
    "audio_computation-sentences" : "calculaudio-phraseaudio",
    "motor-cognitive" : "clicGaudio + clicDaudio + clicGvideo + clicDvideo -" \
        "calculaudio - phraseaudio - calculvideo - phrasevideo",
    "cognitive-motor" : "calculaudio + phraseaudio + calculvideo + phrasevideo -"\
        "clicGaudio - clicDaudio - clicGvideo - clicDvideo",
    "audio-video" : "calculaudio + phraseaudio + clicGaudio + clicDaudio -"\
        "calculvideo - phrasevideo - clicGvideo - clicDvideo",
    "video-audio" : "calculvideo + phrasevideo + clicGvideo + clicDvideo -"\
        "calculaudio - phraseaudio - clicGaudio - clicDaudio",
}

def contrasts_to_spm_vec(condition_list, contrasts):
    res = {}
    cond_d = {}
    for ic,c in enumerate(condition_list):
        cond_d[ic] = c
    for con_name,con in contrasts.iteritems():
        res[con_name] = con
        res[con_name] = res[con_name].replace(' ','')
        if not con.startswith('+') and not con.startswith('-'):
            res[con_name] = '+' + res[con_name]

        cdict = dict( (c,i) for i,c in enumerate(condition_list) )

        for c in reversed(sorted(condition_list, key=len)):
            #print 'res[con_name]:', res[con_name]
            #print 'c:', c
            res[con_name] = res[con_name].replace(c,'%02d'%cdict[c])

    for con_name, con in res.iteritems():
        v = np.zeros(len(condition_list))
        for i in range(0,len(con),3):
            v[int(con[i+1:i+3])] = eval('%s1'%con[i])
        res[con_name] = v
    return res

onsets_un_evnt = OrderedDict([
        ('audio', [np.array([  35.])])])
durations_un_evnt = OrderedDict([
        ('audio', [np.array([  0.])])])
                                    
                                    
# audio, video only (video does not comprise damier)
to_discard = ['damier_H','damier_V']
o,d = onsets_loc, durations_loc
for c in ['audio','video']:
    o,d = merge_onsets(o, c, durations=d, discard=to_discard)
onsets_loc_av, durations_loc_av = o,d
default_contrasts_loc_av = {
    'audio-video' : 'audio-video',
    'video-audio' : 'video-audio',
    }

# audio, video full (video comprises damier)
o,d = onsets_loc, durations_loc
o,d = merge_onsets(o, 'audio', durations=d)
o,d = merge_onsets(o, 'video',criterion=lambda x: 'video' in x or 'damier' in x,
                   durations=d)
onsets_loc_avd, durations_loc_avd = o,d
default_contrasts_loc_avd = {
    'audio-video' : 'audio-video',
    'video-audio' : 'video-audio',
    }


# audio, video, damier
o,d = onsets_loc, durations_loc
o,d = merge_onsets(o, 'audio', durations=d)
o,d = merge_onsets(o, 'video', durations=d)
o,d = merge_onsets(o, 'damier', durations=d)
onsets_loc_av_d, durations_loc_av_d = o,d
default_contrasts_loc_av_d = {
    'audio-video' : 'audio-video',
    'video-audio' : 'video-audio',
    }
builtin_paradigms.append('loc_av_d')

#only one condition (audio)
o,d = onsets_loc, durations_loc
to_discard = ['damier_H','damier_V','clicDvideo','clicGvideo', 'phrasevideo', 'calculvideo']
onsets_loc_a, durations_loc_a = merge_onsets(o, 'audio', durations=d, discard=to_discard)
builtin_paradigms.append('loc_a')

# calcul, phrase only
to_discard = ['damier_H','damier_V','clicDaudio',
              'clicDvideo','clicGvideo','clicGaudio']
o,d = onsets_loc, durations_loc
for c in ['calcul','phrase']:
    o,d = merge_onsets(o, c, durations=d, discard=to_discard)
onsets_loc_cp_only, durations_loc_cp_only = o,d
default_contrasts_loc_cp_only = {
    "computation-sentences" : "calcul - phrase",
    "sentences-computation" : "phrase - calcul",
    }
builtin_paradigms.append('loc_cp_only')

# calcul only
onsets_loc_c_only = { 'calcul' : onsets_loc_cp_only['calcul'] }
durations_loc_c_only = { 'calcul' : durations_loc_cp_only['calcul'] }
default_contrasts_loc_c_only = {}
builtin_paradigms.append('loc_c_only')

# calcul, phrase, clic, damier
o,d = onsets_loc, durations_loc
for c in ['calcul','phrase','clic','damier']:
    o,d = merge_onsets(o, c, durations=d)
onsets_loc_cpcd, durations_loc_cpcd = o,d
default_contrasts_loc_cpcd = {
    "computation-sentences" : "calcul - phrase",
    "sentences-computation" : "phrase - calcul",
    "motor-cognitive" : "calcul + phrase - clic",
    "cognitive-motor" : "clic - calcul - phrase",
    }
builtin_paradigms.append('loc_cpcd')

onsets_loc_ainsi = OrderedDict([
    ('clicGaudio', [np.array([14.6, 33.3, 89.1, 197.7, 209.0, 235.2, 306.8, 344.3, 347.7, 355.9])]),#0
    ('calculaudio', [np.array([47.9, 59.6, 63.7, 122.7, 153.5, 191.0, 194.3, 239.3, 265.5, 340.5])]),#1
    ('clicGvideo', [np.array([37.1, 104.0, 168.1, 175.6, 217.2, 243.1, 280.5, 291.8, 295.5, 359.3])]),#2
    ('clicDaudio', [np.array([17.9, 127.3, 138.5, 201.4, 224.7, 261.4, 284.7, 313.4, 336.4, 363.0])]),#3
    ('clicDvideo', [np.array([26.2, 44.9, 85.7, 101.0, 288.4, 299.3, 310.5, 373.8, 377.6, 397.1])]),#4
    ('phraseaudio', [np.array([22.5, 29.6, 40.8, 130.6, 171.4, 205.6, 321.8, 388.8, 400.1, 404.2])]),#5
    ('calculvideo', [np.array([0.0, 3.0, 179.0, 213.4, 227.7, 273.4, 277.1, 302.7, 318.4, 351.4])]),#6
    ('phrasevideo', [np.array([52.5, 55.8, 81.6, 108.6, 186.4, 221.0, 231.8, 269.7, 333.4, 393.3])]),#7
        ])
durations_loc_ainsi = apply_to_leaves(onsets_loc_ainsi, np.zeros_like)

builtin_paradigms.append('loc_ainsi')

# calcul, phrase, clicD, clicG for Ainsi
o,d = onsets_loc_ainsi, durations_loc_ainsi
for c in ['calcul','phrase','clicD','clicG']:
    o,d = merge_onsets(o, c, durations=d)
    onsets_loc_ainsi_cpcd, durations_loc_ainsi_cpcd = o,d
builtin_paradigms.append('loc_ainsi_cpcd')

def check_stim_durations(stimOnsets, stimDurations):
    """ If no durations specified (stimDurations is None or empty np.array)
    then assume spiked stimuli: return a sequence of zeros with same
    shape as onsets sequence.
    Check that durations have same shape as onsets.

    """
    nbc = len(stimOnsets)
    nbs = len(stimOnsets[stimOnsets.keys()[0]])
    if stimDurations is None or \
            (type(stimDurations)==list and \
                 all([d is None for d in stimDurations])):
        durSeq = [[np.array([]) for s in xrange(nbs)] for i in xrange(nbc)]
        stimDurations = OrderedDict(zip(stimOnsets.keys(), durSeq))

    if stimDurations.keys() != stimOnsets.keys():
        raise Exception('Conditions in stimDurations (%s) differ '\
                            'from stimOnsets (%s)' %(stimDurations.keys(),
                                                     stimOnsets.keys()))

    for cn, sdur in stimDurations.iteritems():
        for i,dur in enumerate(sdur):
            if (dur is None):
                stimDurations[cn][i] = np.zeros_like(stimOnsets[cn][i])
            elif (hasattr(dur, 'len') and len(dur)==0):
                stimDurations[cn][i] = np.zeros_like(stimOnsets[cn][i])
            elif  (hasattr(dur, 'size') and dur.size== 0):
                stimDurations[cn][i] = np.zeros_like(stimOnsets[cn][i])
            else:
                if not isinstance(stimDurations, np.ndarray):
                    stimDurations[cn][i] = np.array(dur)

                assert len(stimDurations[cn][i]) == len(stimOnsets[cn][i])

    return stimDurations



def extend_sampled_events(sampledEvents, sampledDurations):
    """ Add events to encode stimulus duration
    """
    if 0:
        print 'sampledEvents:', len(sampledEvents)
        print sampledEvents
        print 'sampledDurations:', len(sampledDurations)
        print sampledDurations
    extendedEvents = set(sampledEvents)
    for io,o in enumerate(sampledEvents):
        # print 'io:', io
        # print 'o:', o
        extendedEvents.update(range(o + 1, o + sampledDurations[io]))
        # print 'extendedEvents : ', extendedEvents

    return np.array(sorted(list(extendedEvents)), dtype=int)


def restarize_events(events, durations, dt, tMax):
    """ build a binary sequence of events. Each event start is approximated
    to the nearest time point on the time grid defined by dt and tMax.
    """
    smplEvents = np.array(np.round_(np.divide(events, dt)), dtype=int)
    # print 'smplEvents : ', smplEvents
    smplDurations = np.array(np.round_(np.divide(durations,dt)),dtype=int)
    # print 'smplDurations : ', smplDurations
    smplEvents = extend_sampled_events(smplEvents, smplDurations)
    # print 'smplEvents[-1] : ', smplEvents[-1]
    pyhrf.verbose(6, 'sampledOnsets :' + str(smplEvents.shape))
    pyhrf.verbose.printNdarray(6, smplEvents)
    if np.allclose(tMax%dt,0):
        binSeq = np.zeros(tMax/dt + 1)
    else:
        binSeq = np.zeros(np.round((tMax + dt)/ dt))
    binSeq[smplEvents] = 1
    pyhrf.verbose(6, 'bins :')
    pyhrf.verbose.printNdarray(6, binSeq)

    return binSeq

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

class Paradigm:
    """
    """
    def __init__(self, stimOnsets, sessionDurations=None, stimDurations=None):
        """

        Args:
            *stimOnsets* (dict of list) :
                dictionary mapping a condition name to a list of session
                stimulus time arrivals.
                eg:
                {'cond1' : [<session 1 onsets>, <session 2 onsets>]
                 'cond2' : [<session 1 onsets>, <session 2 onsets>]
                 }
            *sessionDurations* (1D numpy float array): durations for all sessions
            *stimDurations* (dict of list) : same structure as stimOnsets.
                             If None, spiked stimuli are assumed (ie duration=0).

        """

        # print 'stimOnsets;', stimOnsets.__class__
        # print 'stimDurations:', stimDurations
        assert isinstance(stimOnsets, dict)
        c = stimOnsets.keys()[0]
        assert isinstance(stimOnsets[c], list)
        #print stimOnsets[c][0]
        assert isinstance(stimOnsets[c][0], np.ndarray)
        assert stimOnsets[c][0].ndim == 1

        if sessionDurations is not None:
            assert isinstance(sessionDurations, np.ndarray) or \
                isinstance(sessionDurations, list)
            assert len(sessionDurations) == len(stimOnsets[c])

        self.stimOnsets = stimOnsets
        # for k,v in stimOnsets.items():
        #     for isess,vsess in enumerate(v):
        #         if not isinstance(vsess, np.ndarray):
        #             self.stimOnsets[k][isess] = np.array(vsess)

        self.stimDurations = check_stim_durations(stimOnsets, stimDurations)

        self.nbSessions = len(self.stimOnsets[self.stimOnsets.keys()[0]])
        if sessionDurations is None:
            # print 'compute sessionDurations ...'
            sessionDurations = []
            max_onset = 0.
            for s in xrange(self.nbSessions):
                for cn,v in stimOnsets.items():
                    argmax_onset = v[s].argmax()
                    if v[s][argmax_onset] > max_onset:
                        max_onset = v[s][argmax_onset]
                        max_durations = self.stimDurations[cn][s][argmax_onset]
                sessionDurations.append(max_onset + max_durations)

            #sessionDurations = [max([max(v[s]) for v in stimOnsets.values()]) \
                                    #for s in xrange(self.nbSessions)]
        self.sessionDurations = sessionDurations

    def delete_condition(self, cond):
        self.stimOnsets.pop(cond, None)
        self.stimDurations.pop(cond, None)

    @classmethod
    def from_session_dict(self, d, sessionDurations=None):
        nd = stack_trees([d[sess] for sess in sorted(d.keys())])
        return Paradigm(nd['onsets'], sessionDurations, nd['stimulusLength'])


    def __repr__(self):
        s = 'Paradigm('
        s += 'stimOnsets=%s,' %repr(self.stimOnsets)
        s += 'sessionDurations=%s,' %repr(self.sessionDurations)
        s += 'stimDurations=%s)' %repr(self.stimDurations)
        return s

    @classmethod
    def from_csv(self, csvFile, delim=' '):
        """
        Create a Paradigm object from a CSV file which columns are:
        session, task name, stimulation onset, stimulation duration, [amplitude]
        """
        onsets, durations = load_paradigm_from_csv(csvFile, delim)
        return Paradigm(onsets, stimDurations=durations)

    @classmethod
    def from_spm_mat(self, spm_mat_file):
        """
        TODO: handle session durations
        """
        par_data,_ = load_paradigm_from_mat(spm_mat_file)
        par_data = stack_trees([par_data[s] for s in sorted(par_data.keys())])
        return Paradigm(par_data['onsets'],
                        stimDurations=par_data['stimulusLength'])

    def save_spm_mat_for_1st_level_glm(self, mat_file, session=0):
        ordered_names = sorted(self.stimOnsets.keys())
        to_save = {
            'onsets' : np.array([self.stimOnsets[n][session] \
                                     for n in  ordered_names],
                                dtype=object),
            'names' : np.array(ordered_names, dtype=object),
            'durations' : np.array([self.stimDurations[n][session] \
                                        for n in  ordered_names], dtype=object),
            }

        # print 'to_save:'
        # print to_save
        # print 'stimOnsets:', self.stimOnsets
        savemat(mat_file, to_save, oned_as='row')

    def join_sessions(self):
        if self.nbSessions == 1:
            return self

        ons = OrderedDict((n,[o]) for n,o in self.get_joined_onsets().iteritems())
        dur = OrderedDict((n,[d]) for n,d in self.get_joined_durations().iteritems())
        return Paradigm(ons, self.sessionDurations[-1:], dur)



    def to_nipy_paradigm(self):

        p = self.join_sessions()
        sorted_conds = sorted(p.stimOnsets.keys())
        onsets = unstack_trees(p.stimOnsets)
        durations = unstack_trees(p.stimDurations)


        cond_ids = [ np.hstack([[c]*len(o) \
                                    for c,o in sorted(sess_ons.items())]) \
                         for sess_ons in onsets ]
        #print 'cond_ids:', cond_ids
        onsets = [ np.hstack([sess_ons[c] for c in sorted_conds]) \
                       for sess_ons in onsets ]
        durations = [ np.hstack([sess_dur[c] for c in sorted_conds]) \
                          for sess_dur in durations ]
        # print 'onsets:'
        # print onsets
        # print 'durations'
        # print durations
        # print '[(dur>0.).any() for dur in durations]:'
        # print [(dur>0.).any() for dur in durations]

        if any([(dur>0.).any() for dur in durations]):
            # Block paradigm
            if len(onsets) > 1:
                dd = [('session%02d'%i, BlockParadigm(d[0], d[1], d[2])) \
                          for i,d in enumerate(zip(cond_ids, onsets, durations))]
                return dict(dd)
            else:
                return BlockParadigm(cond_ids[0], onsets[0], durations[0],
                                     amplitude=None)
        else:
            if len(onsets) > 1:

                dd = [('session%02d'%i, EventRelatedParadigm(d[0], d[1])) \
                          for i,d in enumerate(zip(cond_ids,onsets))]
                return dict(dd)
            else:
                return EventRelatedParadigm(cond_ids[0], onsets[0],
                                            amplitude=None)

    def save_csv(self, csvFile):

        s = ''
        for cn, sessOns in self.stimOnsets.iteritems():
            sessDur = self.stimDurations[cn]
            for iSess,ons in enumerate(sessOns):
                durs = sessDur[iSess]
                for on,dur in zip(ons,durs):
                    s += '%d "%s" %f %f 1.\n' %(iSess,cn,on,dur)
        f = open(csvFile,'w')
        f.write(s)
        f.close()


    def get_info(self, long=True):
        s = ''
        s += 'sessionDurations: %s\n' %str(self.sessionDurations)
        allOns = []
        for oc in self.stimOnsets.values(): #parse conditions
            for o in oc: #parse sessions
                allOns.extend(o)

        allOns = np.array(allOns)
        allOns.sort()
        last = allOns.max()
        meanISI = np.diff(allOns).mean()
        stdISI = np.diff(allOns).std()
        s += ' - onsets : ISI=%1.2f(%1.2f)sec' \
             ' - last event: %1.2fsec\n' %(meanISI, stdISI, last)
        for stimName in self.stimOnsets.keys():
            ntrials = [len(o) for o in self.stimOnsets[stimName]]
            s += '     >> %s, trials per session: %s\n' %(stimName,str(ntrials))
            if long:
                s += dictToString(self.stimOnsets[stimName], prefix='     ')
                s += dictToString(self.stimDurations[stimName], prefix='     ')
        return s


    def get_joined_onsets(self):
        """ For each condition, join onsets of all sessions.
        """
        jOnsets = OrderedDict([])

        for cn, sessOns in self.stimOnsets.iteritems():
            scanStartTime = 0.
            jons = np.array([])
            for iSess,ons in enumerate(sessOns):
                shiftedOns = ons + scanStartTime
                jons = np.concatenate((jons, shiftedOns))
                scanStartTime = self.sessionDurations[iSess]
            jOnsets[cn] = jons
        #print 'jOnsets:'
        #print jOnsets
        return jOnsets


    def get_nb_trials(self):
        pass

    def get_joined_durations(self):
        """ For each condition, join stimulus durations of all sessions.
        """
        jDurations = OrderedDict([])
        for cn, sessDur in self.stimDurations.iteritems():
            #print 'sessDur:', sessDur
            jDurations[cn] = np.concatenate(tuple(sessDur))

        return jDurations


    def get_stimulus_names(self):
        return self.stimOnsets.keys()

    def get_t_max(self):
        ns = len(self.sessionDurations)
        return max([self.sessionDurations[i] for i in xrange(ns)])

    def get_rastered(self, dt, tMax=None):
        """ Return binary sequences of stimulus arrivals. Each stimulus event
        is approximated to the closest time point on the time grid defined
        by dt. eg return:
        { 'cond1' : [np.array([ 0 0 0 1 0 0 1 1 1 0 1]),
                     np.array([ 0 1 1 1 0 0 1 0 1 0 0])] },
          'cond2' : [np.array([ 0 0 0 1 0 0 1 1 1 0 0]),
                     np.array([ 1 1 0 1 0 1 0 0 0 0 0])] },

        Arg:
            - dt (float): temporal resolution of the target grid
            - tMax (float): total duration of the paradigm
                            If None, then use the session lengths
        """
        assert dt > 0.
        rasteredParadigm = OrderedDict({})
        # print 'self.stimOnsets:'
        # print self.stimOnsets
        if tMax is None:
            tMax = self.get_t_max()
        for cn in self.stimOnsets.iterkeys():
            par = []
            for iSess, ons in enumerate(self.stimOnsets[cn]):
                dur = self.stimDurations[cn][iSess]
                if 0:
                    print '-----------------'
                    print 'dur:', dur
                    print 'ons:', ons
                    print 'tMax:', tMax
                    print 'dt:', dt
                    print '-----------------'
                binaryEvents = restarize_events(ons, dur, dt, tMax)
                par.append(binaryEvents)
            rasteredParadigm[cn] = np.vstack(par)

        return rasteredParadigm

    def get_joined_and_rastered(self, dt):
        rpar = self.get_rastered(dt)
        for cn in rpar.iterkeys():
            rpar[cn] = np.concatenate(rpar[cn])

        return rpar
