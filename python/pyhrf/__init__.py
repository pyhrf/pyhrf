# -*- coding: utf-8 -*-

#Resolve current version
from version import __version__
_VERSION_STRING = 'pyhrf ' + __version__

from configuration import cfg, useModesStr
from configuration import DEVEL, ENDUSER

use_modes = useModesStr.keys()
if cfg['global']['use_mode'] not in use_modes:
    raise Exception('Wrong use mode "%s" in configuration, '\
                    'available choices: %s' %(cfg['global']['use_mode'],
                                              ', '.join(use_modes)))

__usemode__ = useModesStr[cfg['global']['use_mode']]

import numpy
numpy.seterr(cfg['global']['numpy_floating_point_error'])

for pname,pval in configuration.cfg['global'].iteritems():
    setattr(configuration,pname,pval)

import _verbose
from _verbose import verboseLevels
verbose = _verbose.Verbose(verbosity=cfg['global']['verbosity'])

from core import *

