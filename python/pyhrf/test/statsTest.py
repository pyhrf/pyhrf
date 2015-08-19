# -*- coding: utf-8 -*-


import unittest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np

from pyhrf.tools import stack_trees
from pprint import pprint

from pyhrf.stats import gm_cdf, cpt_ppm_a_mcmc, gm_mean, gm_var, \
    cpt_ppm_g_mcmc, cpt_ppm_g_apost, cpt_ppm_a_norm, cpt_ppm_g_norm
from pyhrf.stats.random import rpnorm, gm_sample

class RPNormTest(unittest.TestCase):

    def testSimple(self):

        rnd = rpnorm(1000000,3.,2.)
        assert (rnd>0).all()
        #print rnd.mean()
        #import matplotlib.pyplot as plt
        #plt.hist(rnd)
        #plt.show()

class PPMTest(unittest.TestCase):

    def setUp(self):
        def mixt(m,v,p):
            assert len(m) == len(v)
            assert len(m) == len(p)
            return {'means':np.array(m), 'variances':np.array(v),
                    'props':np.array(p)}

        self.mixture_inactive = mixt([0.,0.], [1., 1.], [1., 0.])

        self.mixture_active = mixt([0.,100.], [1., 1.], [0., 1.])


        self.mixture_half = mixt([-1.,1.], [1., 1.], [.5, .5])

        def stack_comp(l):
            return np.array(l).T
        self.mixt_stack = (self.mixture_inactive, self.mixture_active,
                           self.mixture_half)
        self.mixt_map = stack_trees(self.mixt_stack, join_func=stack_comp)

        self.n_pos = len(self.mixt_stack)

        if 0:
            print 'self.mixture_inactive:'
            print self.mixture_inactive

            print 'self.mixture_active:'
            print self.mixture_active

            print 'self.mixture_half:'
            print self.mixture_half


            print 'self.mixt_map:'
            pprint(self.mixt_map)


    def _test_gm(self, means, variances, props, n):
        samples = gm_sample(means, variances, props, n)
        # assert_allclose(samples.mean(), gm_mean(means, variances, props),
        #                 rtol=1e-2, atol=5e-3)

        assert_almost_equal(samples.mean(), gm_mean(means, variances, props),
                            decimal=2)

    def test_gm_sample_inactive(self):
        self._test_gm(n=100000, **self.mixture_inactive)

    def test_gm_sample_active(self):
        self._test_gm(n=100000, **self.mixture_active)

    def test_gm_sample_half(self):
        self._test_gm(n=500000, **self.mixture_half)

    def test_gm_cdf(self):
        vthresh = 0.
        if 0:
            print 'gm_cdf P(X<vthresh):'
            print gm_cdf(vthresh, **self.mixt_map)

        assert_array_equal(gm_cdf(vthresh, **self.mixt_map), [0.5,0.,0.5])

    def test_ppm_g_apost(self):
        gamma = 0.
        assert_array_equal(cpt_ppm_g_apost(gamma=gamma, **self.mixt_map),
                           [0.5,1.,0.5])

    def test_ppm_a_mcmc(self):
        nsamples = 500000
        mixt_map_samples = np.zeros((nsamples,self.n_pos))
        for i,mixt in enumerate(self.mixt_stack):
            mixt_map_samples[:,i] = gm_sample(n=nsamples, **mixt)

        alpha = .05
        ppm = cpt_ppm_a_mcmc(mixt_map_samples, alpha)
        #print 'ppm_a_mcmc:', ppm
        #assert_allclose(ppm[0], [1.645], rtol=1e-2)
        assert_almost_equal(ppm[0], [1.645], decimal=2)
        #todo theoretical calculus for mixt_active and mixt_half

    def test_ppm_g_mcmc(self):

        nsamples = 500000
        mixt_map_samples = np.zeros((nsamples,self.n_pos))
        for i,mixt in enumerate(self.mixt_stack):
            mixt_map_samples[:,i] = gm_sample(n=nsamples, **mixt)

        gamma = 0.
        ppm = cpt_ppm_g_mcmc(mixt_map_samples, gamma)
        #print 'ppm_g_mcmc:', ppm
        #assert_allclose(ppm, [0.5,1.,0.5], rtol=1e-2)
        assert_almost_equal(ppm, [0.5,1.,0.5], decimal=2)

    #def test_ppm_a_apost()

    def test_ppm_a_norm(self):

        means = np.array([0., 1., 10.])
        variances = np.array([1., 1., 1.])

        alpha = 0.

        # assert_allclose(cpt_ppm_a_norm(means, variances, alpha),
        #                 means + 1.645, rtol=1e-2)
        assert_almost_equal(cpt_ppm_a_norm(means, variances, alpha),
                        [0.5, .841, 1.], decimal=2)

    def test_ppm_g_norm(self):

        means = np.array([0., 1., 10.])
        variances = np.array([1., 1., 1.])

        gamma = 0.95

        # assert_allclose(cpt_ppm_g_norm(means, variances, gamma),
        #                 [0.5, .841, 1.], rtol=1e-2)
        assert_almost_equal(cpt_ppm_g_norm(means, variances, gamma),
                        means - 1.645, decimal=2)
