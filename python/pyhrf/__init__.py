# -*- coding: utf-8 -*-
"""PyHRF is a set of tools for within-subject fMRI data analysis,
which focuses on the characterization of the hemodynamics.

Within the chain of fMRI data processing, these tools provide alternatives
to the classical within-subject GLM estimation step.
The inputs are preprocessed within-subject data and the outputs are
statistical maps and/or fitted HRFs.

The package is mainly written in Python and provides the
implementation of the two following methods:

- The joint-detection estimation (JDE) approach, which divides the brain
  into functionnaly homogeneous regions and provides one HRF estimate per
  region as well as response levels specific to each voxel and each experimental
  condition. This method embeds a temporal regularization on the estimated HRFs
  and an adaptive spatial regularization on the response levels.


- The Regularized Finite Impulse Response (RFIR) approach, which provides
  HRF estimates for each voxel and experimental conditions. This method embeds
  a temporal regularization on the HRF shapes, but proceeds independently across
  voxels (no spatial model).


Check website for details: www.pyhrf.org

"""

import os
import logging
import warnings
import sys

import numpy

import pyhrf._verbose

from pyhrf.version import __version__
from pyhrf.configuration import cfg, useModesStr
from pyhrf.configuration import DEVEL, ENDUSER


# Resolve current version
_VERSION_STRING = 'pyhrf ' + __version__
use_modes = useModesStr.keys()
if cfg['global']['use_mode'] not in use_modes:
    raise Exception('Wrong use mode "%s" in configuration, '
                    'available choices: %s' % (cfg['global']['use_mode'],
                                               ', '.join(use_modes)))

__usemode__ = useModesStr[cfg['global']['use_mode']]

verbose_levels = {logging.getLevelName(value): value
                  for value in dir(logging)
                  if value.isupper() and
                  isinstance(logging.getLevelName(value), int)}

numpy.seterr(cfg['global']['numpy_floating_point_error'])

for pname, pval in pyhrf.configuration.cfg['global'].iteritems():
    setattr(pyhrf.configuration, pname, pval)

# logging configuration
# warnings.filterwarnings("always", message=".*verbos.*",
#                         category=DeprecationWarning)
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
verb = cfg['log']['level']
try:
    logger.setLevel(verb.upper())
except ValueError, e:
    verb = logging.WARNING
    logger.setLevel(verb)
    logger.error(e.message + '. Check your configuration file.'
                 ' Falling back to default WARNING level')
if cfg['log']['log_to_file']:
    file_handler = logging.FileHandler(os.path.join(cfg['global']['tmp_path'],
                                                    'pyhrf.log'))
    formatter = logging.Formatter(
        '%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')
    file_handler.setLevel(verb)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class Verbose(pyhrf._verbose.Verbose):
    """This is a dummy class implementing the original Verbose class.

    This is only to be able to raise a warning when one uses this old
    implementation.

    """

    old_to_new_log_dict = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.INFO,
        3: logging.INFO,
        4: logging.INFO,
        5: logging.DEBUG,
        6: logging.DEBUG,
    }

    def __call__(self, verbosity, msg, new_line=True):
        warnings.warn('The homemade verbose module is deprecated,'
                      ' use logging instead', DeprecationWarning)
        logger.log(Verbose.old_to_new_log_dict[verbosity], msg)

verbose = Verbose(verbosity=verb)

# FIXME: The following are two things we want to avoid:
# import not at the beginning and wildcard import
# and furthermore it erase the logger variable in the namespace
tmplogger = logger
from pyhrf.core import *
logger = tmplogger
