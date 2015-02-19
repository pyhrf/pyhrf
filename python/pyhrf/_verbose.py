# -*- coding: utf-8 -*-

"""WARNING: DEPRECATED!
This module implements a logging facility."""

import sys
import inspect
import string
import types
import numpy as _np
import logging

logger = logging.getLogger(__name__)


# VERB_NONE = 0
# VERB_SHORT = 1
# VERB_MAIN_STEPS = 2
# VERB_MAIN_FUNC_CALLS = 3
# VERB_MAIN_VAR_VALS = 4
# VERB_DETAIL_VAR_VALS = 5
# VERB_DEBUG = 6

verboseLevels = {
    0: 'no verbose',
    1: 'minimal verbose',
    2: 'main steps',
    3: 'main function calls',
    4: 'main variable values',
    5: 'detailled variable values',
    6: 'debug (everything)',
}


def unprintable(obj):
    """Determine if object can be printed."""
    return inspect.ismethod(obj) or inspect.ismethoddescriptor(obj) or \
        isinstance(obj, (types.BuiltinFunctionType, types.MethodType,
                         types.FunctionType, _np.ufunc, type))


def dict_to_string(dikt, prefix='', numpy_array_format=None,
                   visited=None, exclude=None):
    """Convert dictionnary to displayable string."""
    if numpy_array_format is None:
        numpy_array_format = {'precision': 3, 'max_line_width': 100}
    logger.debug('dict_to_string received: %s\n'
                 '    ->: %s\n'
                 'unprintable: %s', dikt, type(dikt), unprintable(dikt))
    if visited is None:
        visited = []
    if exclude is None:
        exclude = []

    if (id(dikt) in visited) and not (isinstance(dikt, (int, str, float)) or
                                      _np.isscalar(dikt)):
        logger.debug('Already seen: %s %s', dikt, type(dikt))
        return prefix + ' ### \n'

    visited += [id(dikt)]
    space = '  '
    if isinstance(dikt, (str, unicode)) or _np.isscalar(dikt):
        return prefix + repr(dikt) + '\n'
    elif isinstance(dikt, dict):
        sout = prefix + '{\n'
        for k, v in dikt.iteritems():
            if isinstance(k, dict):
                sout += '\n'
                sout += dict_to_string(k, prefix=prefix + space,
                                       numpy_array_format=numpy_array_format,
                                       visited=visited, exclude=exclude)
            else:
                sout += prefix + str(k)
                logger.debug('k: %s', k)
            sout += ' : ' + '\n'
            sout += dict_to_string(v, prefix=prefix + space,
                                   numpy_array_format=numpy_array_format,
                                   visited=visited, exclude=exclude)
            if sout[-1] != '\n':
                sout += '\n'
        sout += prefix + '}'
        if sout[-1] != '\n':
            sout += '\n'
    elif isinstance(dikt, set):
        sout = prefix + str(dikt)
    elif isinstance(dikt, list) or \
            (isinstance(dikt, _np.ndarray) and dikt.dtype == _np.object):
        sout = prefix + '[\n'
        for e in dikt:
            sout += dict_to_string(e, prefix=prefix + space,
                                   numpy_array_format=numpy_array_format,
                                   visited=visited, exclude=exclude)
            sout += prefix + ',\n'
        sout += prefix + ']\n'
    elif isinstance(dikt, tuple):
        sout = prefix + '(\n'
        for e in dikt:
            sout += dict_to_string(e, prefix=prefix + space,
                                   numpy_array_format=numpy_array_format,
                                   visited=visited, exclude=exclude)
            sout += prefix + ',\n'
        sout += prefix + ')\n'
    elif isinstance(dikt, _np.ndarray):
        if dikt.size < 1000:
            sva = _np.array2string(dikt, **numpy_array_format)
            sout = string.join([prefix + s for s in sva.split('\n')], '\n')
        else:
            sout = prefix + '%s -- %1.3g(%1.3g)[%1.3g;%1.3g]' \
                % (str(dikt.shape), dikt.mean(), dikt.std(),
                   dikt.min(), dikt.max())
        if not sout.endswith('\n'):
            sout += '\n'
    else:
        sout = ''
        for attr in dir(dikt):
            attrobj = getattr(dikt, attr)
            if attr[0] != '_' and not unprintable(attrobj) and \
                    attr not in exclude:

                logger.debug('attr: %s\n'
                             '    -> %s', attr, type(attr))
                sout += prefix + '.' + attr + ':' + '\n'
                sout += dict_to_string(getattr(dikt, attr), prefix=prefix + space,
                                       numpy_array_format=numpy_array_format,
                                       visited=visited, exclude=exclude)
    return sout


class Verbose(object):
    """Do not use. Implement a deprecated logging class. Use logging module."""
    default_state = 0
    line_to_be_continued = 1

    def __init__(self, verbosity=0, log=sys.stdout):
        self.verbosity = verbosity
        self.log = log
        self.state = self.default_state

    def set_log(self, log):
        """Set the log file/stream to use."""
        self.log = log

    def set_verbosity(self, verbosity):
        """Define verbosity level."""
        self.verbosity = verbosity

    def __call__(self, verbosity, msg, new_line=True):
        if self.verbosity >= verbosity:
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
                    slog = ('*' * verbosity) + ' ' + line
                    self.log.write(slog)

    def print_ndarray(self, verbosity, ndarray):
        """Displays a representation of an ndarray instance."""
        if self.verbosity >= verbosity:
            if isinstance(ndarray, _np.ndarray):
                self.print_dict(verbosity, ndarray),
            else:
                self.__call__(verbosity, repr(ndarray))

    def print_dict(self, verbosity, dikt, subprefix='', exclude=None):
        """Displays a representation of a dictionnary."""
        if self.verbosity >= verbosity:
            print dict_to_string(dikt, prefix='*' * verbosity + ' ' + subprefix,
                                 exclude=exclude),
