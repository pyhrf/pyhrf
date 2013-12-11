

# -*- coding: utf-8 -*-
import sys
import inspect
import string
import types
import numpy as _np


# VERB_NONE = 0
# VERB_SHORT = 1
# VERB_MAIN_STEPS = 2
# VERB_MAIN_FUNC_CALLS = 3
# VERB_MAIN_VAR_VALS = 4
# VERB_DETAIL_VAR_VALS = 5
# VERB_DEBUG = 6

verboseLevels = {
    0 : 'no verbose',
    1 : 'minimal verbose',
    2 : 'main steps',
    3 : 'main function calls',
    4 : 'main variable values',
    5 : 'detailled variable values',
    6 : 'debug (everything)',
    }

debug = False

def unPrintable(o):
    return inspect.ismethod(o) or inspect.ismethoddescriptor(o) or \
           type(o) == types.BuiltinFunctionType or type(o) == types.MethodType \
           or type(o) == types.FunctionType or type(o) == _np.ufunc \
           or type(o) == type

def dictToString(d, prefix='',
                 numpyArrayFormat={'precision':3,'max_line_width':100},
                 visited=None, exclude=None):
    if debug:
        print 'dictToString recieved :', d
        print '    ->:', type(d)
        print ' unPrintable ?:', unPrintable(d)
    if visited == None:
        visited = []
    if exclude == None:
        exclude = []
    
    if (id(d) in visited) and not (type(d)==int or type(d)==str \
                                       or type(d)==float or _np.isscalar(d)):
        if debug: print 'already seen :', d, type(d)
        return prefix+' ### \n'

    visited += [id(d)]
    space = '  '
    if type(d) == str or type(d) == unicode or _np.isscalar(d):
        return prefix+repr(d)+'\n'
    elif type(d) == dict:
        sOut = prefix+'{\n'
        for k,v in d.iteritems():
            if type(k) == dict :
                sOut += '\n'
                sOut += dictToString(k, prefix=prefix+space,
                                     numpyArrayFormat=numpyArrayFormat,
                                     visited=visited, exclude=exclude)
            else:
                sOut += prefix+str(k)
                if debug: print 'k:', k
            sOut += ' : '+'\n'
            sOut += dictToString(v, prefix=prefix+space,
                                 numpyArrayFormat=numpyArrayFormat,
                                 visited=visited, exclude=exclude)
            if sOut[-1] != '\n': sOut += '\n'
        sOut+= prefix+'}'
        if sOut[-1] != '\n': sOut += '\n'
    elif type(d) == set:
        sOut = prefix+str(d)
    elif type(d) == list or \
         (type(d) == _np.ndarray and d.dtype == _np.object) :
        sOut = prefix+'[\n'
        for e in d:
            sOut += dictToString(e, prefix=prefix+space,
                                 numpyArrayFormat=numpyArrayFormat,
                                 visited=visited, exclude=exclude)
            sOut += prefix+',\n'
        sOut+= prefix+']\n'
    elif type(d) == tuple :
        sOut = prefix+'(\n'
        for e in d:
            sOut += dictToString(e, prefix=prefix+space,
                                 numpyArrayFormat=numpyArrayFormat,
                                 visited=visited, exclude=exclude)
            sOut += prefix+',\n'
        sOut+= prefix+')\n'
    elif type(d) == _np.ndarray:
        if d.size < 1000:
            sv = _np.array2string(d, **numpyArrayFormat)
            sOut = string.join([prefix+s for s in sv.split('\n')],'\n')
            #print '%% sOut[-1] :', sOut[-1]
        else:
            sOut = prefix + '%s -- %1.3g(%1.3g)[%1.3g;%1.3g]' \
                   %(str(d.shape), d.mean(), d.std(), d.min(), d.max())
        if not sOut.endswith('\n') : sOut += '\n'
    else: #type(d) == types.InstanceType :
        #print 'no direct policy for ', str(d)
        sOut = ''
        for attr in dir(d):
            attrObj = getattr(d,attr)
            if attr[0] != '_' and not unPrintable(attrObj) and \
                   attr not in exclude:
                
                if debug: print 'attr :', attr
                if debug: print ' -> ', type(attrObj)
                sOut += prefix+'.'+attr+':'+'\n'
                sOut += dictToString(getattr(d,attr), prefix=prefix+space,
                                     numpyArrayFormat=numpyArrayFormat,
                                     visited=visited, exclude=exclude)
        #sOut += '\n'
    #else:
    #    print 'unknown type for printing:', type(d)
    #    sOut = str(d)

    return sOut



class Verbose:

    default_state = 0
    line_to_be_continued = 1

    def __init__(self, verbosity=0, log=sys.stdout):
        self.verbosity = verbosity
        self.log = log
        self.state = self.default_state

    def set_log(self,log):
        self.log = log

    def setVerbosity(self, verbosity):
        self.verbosity = verbosity
    
    def __call__(self, verbosity, msg, new_line=True):
        if self.verbosity >= verbosity :
            if isinstance(self.log, file):
                for line in msg.split('\n'):
                    if self.state == self.default_state:
                        if new_line:
                            print >>self.log, '*' * verbosity, line
                            self.state = self.default_state
                        else:
                            print >>self.log, '*' * verbosity, line,
                            self.state = self.line_to_be_continued
                    elif self.state == self.line_to_be_continued:
                        if new_line:
                            print >>self.log, line
                            self.state = self.default_state
                        else:
                            print >>self.log, line,
                            self.state = self.line_to_be_continued
                if new_line:
                    self.log.flush()
            else:
                for line in msg.split('\n'):
                    s = ('*' * verbosity) + ' ' + line
                    self.log.write(s)

    def printNdarray(self, verbosity, a):
        if self.verbosity >= verbosity :
            if type(a) == _np.ndarray:
                self.printDict(verbosity, a),
            else:
                self.__call__(verbosity, repr(a))

    def printDict(self, verbosity, d, subprefix='', exclude=None):
        if self.verbosity >= verbosity :
            print dictToString(d, prefix='*' * verbosity+' '+subprefix,
                               exclude=exclude),
