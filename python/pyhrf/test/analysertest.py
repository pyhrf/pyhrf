# -*- coding: utf-8 -*-

import unittest
import numpy as _np

from pyhrf.jde.beta import *
from pyhrf.boldsynth.field import genPotts, count_homo_cliques
from pyhrf.graph import *


class BetaEstimESTest(unittest.TestCase):

    def test_obs_3Dfield_MAP(self):
        """ Test estimation of beta with an observed field: a small 3D case.
        Partition function estimation method : extrapolation scheme.
        Use the MAP on p(beta|label).
        """
        shape = (5, 5, 5)
        mask = _np.ones(shape, dtype=int)  # full mask
        g = graph_from_lattice(mask, kerMask=kerMask3D_6n)
        # generate a field:
        beta = 0.4
        nbClasses = 2
        labels = genPotts(g, beta, nbClasses)
        # partition function estimation
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
        # beta estimation
        be, pb = beta_estim_obs_field(g, labels, gridLnz)

    def test_obs_2Dfield_MAP(self):
        """ Test estimation of beta with an observed 2D field.
        Partition function estimation method : extrapolation scheme.
        Use the MAP on p(beta|label).
        """
        shape = (6, 6)
        mask = _np.ones(shape, dtype=int)  # full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n)
        # generate a field:
        beta = 0.4
        nbClasses = 2
        labels = genPotts(g, beta, nbClasses)
        # partition function estimation
        old_settings = np.seterr('raise')
        gridLnz = Cpt_Vec_Estim_lnZ_Graph(g, nbClasses)
        gridPace = gridLnz[1][1] - gridLnz[1][0]
        # beta estimation
        be, pb = beta_estim_obs_field(g, labels, gridLnz)
        np.seterr(**old_settings)

    def test_obs_field_ML(self):
        """ Test estimation of beta with an observed field: a small 2D case.
        Partition function estimation method : extrapolation scheme.
        Use the ML  on p(label|beta). PF estimation: Onsager
        """
        shape = (6, 6)
        mask = _np.ones(shape, dtype=int)  # full mask
        g = graph_from_lattice(mask, kerMask=kerMask2D_4n, toroidal=True)
        # generate a field:
        beta = 0.4
        nbClasses = 2
        labels = genPotts(g, beta, nbClasses)
        # partition function estimation
        dbeta = 0.05
        betaGrid = _np.arange(0, 1.5, dbeta)
        lpf = logpf_ising_onsager(labels.size, betaGrid)
        betaML = beta_estim_obs_field(g, labels, (lpf, betaGrid), 'ML')
