# -*- coding: utf-8 -*-
import unittest
import numpy as np

import tempfile
import shutil

import pyhrf
from pyhrf import FmriData
from pyhrf.ui.treatment import jde_vol_from_files, jde_surf_from_files
from pyhrf.parcellation import parcellation_report, parcellation_for_jde

from pyhrf.jde.models import BOLDGibbsSampler as BG
from pyhrf.jde.beta import BetaSampler as BS
from pyhrf.jde.nrl.bigaussian import NRLSampler as NS
from pyhrf.jde.nrl.bigaussian import BiGaussMixtureParamsSampler as BGMS
from pyhrf.jde.hrf import RHSampler as HVS
from pyhrf.jde.hrf import HRFSampler as HS
from pyhrf.jde.models import simulate_bold
from pyhrf.jde.noise import NoiseVarianceSampler

import pyhrf.sandbox.physio as phym
from pyhrf import tools





