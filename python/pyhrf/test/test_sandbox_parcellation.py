import os.path as op
import unittest
import numpy as np
import pyhrf
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal, assert_array_less, assert_equal
import shutil

import pyhrf.graph as pgraph
import pyhrf.sandbox.parcellation as pm
from pyhrf.tools import assert_file_exists
from scipy.sparse import cs_graph_components
import math

from pyhrf.ndarray import expand_array_in_mask, xndarray, MRI3Daxes

from pyhrf import Condition
import pyhrf.boldsynth.scenarios as simu
from pyhrf.tools import Pipeline
from pyhrf.core import FmriData

from pyhrf.parcellation import parcellation_dist

# launch all the tests in here:
#  pyhrf_maketests -v test_sandbox_parcellation
# to see real data:
#  anatomist cortex_occipital_* hrf_territory
#  anatomist cortex_occipital_*

def my_func_to_test(p, output_path):
    return p


class StatTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_norm_bc(self):
        a = np.array([[1.,2.],
                      [2.,4.],
                      [5.,10.]])
        b = np.array([.5,.5])

        expected_norm = np.array([np.linalg.norm(x-b)**2 for x in a])
        assert_almost_equal(pm.norm2_bc(a,b), expected_norm)

    def test_gmm_known_weights_simu_1D(self):
        """
        Test biGMM fit with known post weights, from biGMM samples (no noise)
        1D case.
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = np.array([2., 5., 10.])
        v0 = 3.

        mu1 = np.array([12., 15., 20.])
        v1 = 3.

        l = .3 #weight for component 1

        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0[:,np.newaxis]
        card_c1 = int(l*nsamples)
        samples[:, :card_c1] = np.random.randn(card_c1) * v1**.5 + \
          mu1[:, np.newaxis]
        samples = samples.T

        # compute sample-specific posterior weights:
        nfeats = len(mu0)
        print nfeats
        post_weights = 1 / (1 + (1-l)/l * (v0/v1)**(-nfeats/2.) * \
            np.exp(-(pm.norm2_bc(samples,mu0)/v0 - \
                     pm.norm2_bc(samples,mu1)/v1)/2.))

        # fit GMM knowing post weights:
        mu0_est, mu1_est, v0_est, v1_est, l_est, llh = \
          pm.informedGMM_MV(samples, post_weights)

        # Some outputs:
        mus = np.array([[mu0,mu1], [mu0_est, mu1_est]]).transpose()
        vs = np.array([[v0,v1], [v0_est, v1_est]]).T
        ls = np.array([[1-l,l], [1-l_est, l_est]]).T

        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        for f in xrange(samples.shape[1]):
            pyhrf.verbose(1, get_2Dtable_string(mus[f,:,:], ['C0','C1'],
                                                ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            f = 1 #check this feature

            # plt.vlines(samples[:,f], 0, post_weights, 'r')
            # plt.vlines(samples[:,f], 0, 1-post_weights, 'b')

            plot_gaussian_mixture(np.array([[mu0[f],mu1[f]], [v0,v1]]),
                                  props=[1-l,l], color='r')

            plot_gaussian_mixture(np.array([[mu0_est[f], mu1_est[f]],
                                            [v0_est, v1_est]]),
                                  props=[1-l_est, l_est], color='b')

            plt.hist(samples[:,f], color='g',normed=True)
            plt.show()

        assert_array_almost_equal([mu0_est, mu1_est], [mu0,mu1], decimal=1)
        assert_array_almost_equal([v0_est, v1_est], [v0,v1], decimal=1)
        assert_array_almost_equal([1-l_est,l_est], [1-l,l], decimal=1)

    def test_gmm_known_weights_difvars_noise(self):
        """
        Test biGMM fit with known post weights, from biGMM samples (no noise)
        1D case.
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = np.array([2., 5., 10.])
        v0 = 3.

        mu1 = np.array([12., 15., 20.])
        v1 = v0/50.

        l = .3 #weight for component 1
        v_noise = 1.5
        
        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0[:,np.newaxis]
        card_c1 = int(l*nsamples)
        samples[:, :card_c1] = np.random.randn(card_c1) * v1**.5 + \
          mu1[:, np.newaxis]
        samples = samples.T
        noise = np.random.randn(samples.shape[0], samples.shape[1]) * v_noise
        #print 'noise shape = ', str(noise.shape)
        noisy_samples = samples + noise

        # compute sample-specific posterior weights:
        nfeats = len(mu0)
        print nfeats
        post_weights = 1 / (1 + (1-l)/l * (v0/v1)**(-nfeats/2.) * \
            np.exp(-(pm.norm2_bc(samples,mu0)/v0 - \
                     pm.norm2_bc(samples,mu1)/v1)/2.))

        # fit GMM knowing post weights:
        mu0_est, mu1_est, v0_est, v1_est, l_est, llh = \
          pm.informedGMM_MV(noisy_samples, post_weights)

        # Some outputs:
        mus = np.array([[mu0,mu1], [mu0_est, mu1_est]]).transpose()
        vs = np.array([[v0,v1], [v0_est, v1_est]]).T
        ls = np.array([[1-l,l], [1-l_est, l_est]]).T

        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        for f in xrange(samples.shape[1]):
            pyhrf.verbose(1, get_2Dtable_string(mus[f,:,:], ['C0','C1'],
                                                ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            f = 1 #check this feature

            # plt.vlines(samples[:,f], 0, post_weights, 'r')
            # plt.vlines(samples[:,f], 0, 1-post_weights, 'b')

            plot_gaussian_mixture(np.array([[mu0[f],mu1[f]], [v0,v1]]),
                                  props=[1-l,l], color='r')

            plot_gaussian_mixture(np.array([[mu0_est[f], mu1_est[f]],
                                            [v0_est, v1_est]]),
                                  props=[1-l_est, l_est], color='b')

            plt.hist(samples[:,f], color='g',normed=True)
            plt.show()

        assert_array_almost_equal([mu0_est, mu1_est], [mu0,mu1], decimal=1)
        assert_array_almost_equal([v0_est, v1_est], [v0,v1], decimal=1)
        assert_array_almost_equal([1-l_est,l_est], [1-l,l], decimal=1)
    
    
    def test_gmm_known_weights_difvars_noisea(self):
        """
        Test biGMM fit with known post weights, from biGMM samples (no noise)
        1D case.
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = np.array([2., 5., 10.])
        v0 = 3.

        mu1 = np.array([12., 15., 20.])
        v1 = v0/50.

        l = .3 #weight for component 1
        v_noise = 1.5
        
        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0[:,np.newaxis]
        card_c1 = int(l*nsamples)
        samples[:, :card_c1] = np.random.randn(card_c1) * v1**.5 + \
          mu1[:, np.newaxis]
        samples = samples.T
        noise = np.random.randn(samples.shape[0], samples.shape[1]) * v_noise
        #print 'noise shape = ', str(noise.shape)
        noisy_samples = samples + noise

        # compute sample-specific posterior weights:
        nfeats = len(mu0)
        print nfeats
        post_weights = 1 / (1 + (1-l)/l * (v0/v1)**(-nfeats/2.) * \
            np.exp(-(pm.norm2_bc(noisy_samples,mu0)/v0 - \
                     pm.norm2_bc(noisy_samples,mu1)/v1)/2.))

        # fit GMM knowing post weights:
        mu0_est, mu1_est, v0_est, v1_est, l_est, llh = \
          pm.informedGMM_MV(samples, post_weights)

        # Some outputs:
        mus = np.array([[mu0,mu1], [mu0_est, mu1_est]]).transpose()
        vs = np.array([[v0,v1], [v0_est, v1_est]]).T
        ls = np.array([[1-l,l], [1-l_est, l_est]]).T

        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        for f in xrange(samples.shape[1]):
            pyhrf.verbose(1, get_2Dtable_string(mus[f,:,:], ['C0','C1'],
                                                ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            f = 1 #check this feature

            # plt.vlines(samples[:,f], 0, post_weights, 'r')
            # plt.vlines(samples[:,f], 0, 1-post_weights, 'b')

            plot_gaussian_mixture(np.array([[mu0[f],mu1[f]], [v0,v1]]),
                                  props=[1-l,l], color='r')

            plot_gaussian_mixture(np.array([[mu0_est[f], mu1_est[f]],
                                            [v0_est, v1_est]]),
                                  props=[1-l_est, l_est], color='b')

            plt.hist(samples[:,f], color='g',normed=True)
            plt.show()

        assert_array_almost_equal([mu0_est, mu1_est], [mu0,mu1], decimal=1)
        assert_array_almost_equal([v0_est, v1_est], [v0,v1], decimal=1)
        assert_array_almost_equal([1-l_est,l_est], [1-l,l], decimal=1)


    def test_gmm_known_weights_noisea(self):
        """
        Test biGMM fit with known post weights, from biGMM samples (no noise)
        1D case.
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = np.array([2., 5., 10.])
        v0 = 3.

        mu1 = np.array([12., 15., 20.])
        v1 = 3.

        l = .3 #weight for component 1
        v_noise = 1.5
        
        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0[:,np.newaxis]
        card_c1 = int(l*nsamples)
        samples[:, :card_c1] = np.random.randn(card_c1) * v1**.5 + \
          mu1[:, np.newaxis]
        samples = samples.T
        noise = np.random.randn(samples.shape[0], samples.shape[1]) * v_noise
        #print 'noise shape = ', str(noise.shape)
        noisy_samples = samples + noise

        # compute sample-specific posterior weights:
        nfeats = len(mu0)
        print nfeats
        post_weights = 1 / (1 + (1-l)/l * (v0/v1)**(-nfeats/2.) * \
            np.exp(-(pm.norm2_bc(samples,mu0)/v0 - \
                     pm.norm2_bc(samples,mu1)/v1)/2.))

        # fit GMM knowing post weights:
        mu0_est, mu1_est, v0_est, v1_est, l_est, llh = \
          pm.informedGMM_MV(noisy_samples, post_weights)

        # Some outputs:
        mus = np.array([[mu0,mu1], [mu0_est, mu1_est]]).transpose()
        vs = np.array([[v0,v1], [v0_est, v1_est]]).T
        ls = np.array([[1-l,l], [1-l_est, l_est]]).T

        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        for f in xrange(samples.shape[1]):
            pyhrf.verbose(1, get_2Dtable_string(mus[f,:,:], ['C0','C1'],
                                                ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            f = 1 #check this feature

            # plt.vlines(samples[:,f], 0, post_weights, 'r')
            # plt.vlines(samples[:,f], 0, 1-post_weights, 'b')

            plot_gaussian_mixture(np.array([[mu0[f],mu1[f]], [v0,v1]]),
                                  props=[1-l,l], color='r')

            plot_gaussian_mixture(np.array([[mu0_est[f], mu1_est[f]],
                                            [v0_est, v1_est]]),
                                  props=[1-l_est, l_est], color='b')

            plt.hist(samples[:,f], color='g',normed=True)
            plt.show()

        assert_array_almost_equal([mu0_est, mu1_est], [mu0,mu1], decimal=1)
        assert_array_almost_equal([v0_est, v1_est], [v0,v1], decimal=1)
        assert_array_almost_equal([1-l_est,l_est], [1-l,l], decimal=1)

        
    def test_gmm_known_weights_noise(self):
        """
        Test biGMM fit with known post weights, from biGMM samples (no noise)
        1D case.
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = np.array([2., 5., 10.])
        v0 = 3.

        mu1 = np.array([12., 15., 20.])
        v1 = 3.

        l = .3 #weight for component 1
        v_noise = 1.5
        
        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0[:,np.newaxis]
        card_c1 = int(l*nsamples)
        samples[:, :card_c1] = np.random.randn(card_c1) * v1**.5 + \
          mu1[:, np.newaxis]
        samples = samples.T
        print 'samples shape = ', str(samples.shape)
        noise = np.random.randn(samples.shape[0], samples.shape[1]) * v_noise
        print 'noise shape = ', str(noise.shape)
        noisy_samples = samples + noise

        # compute sample-specific posterior weights:
        nfeats = len(mu0)
        print nfeats
        post_weights = 1 / (1 + (1-l)/l * (v0/v1)**(-nfeats/2.) * \
            np.exp(-(pm.norm2_bc(samples,mu0)/v0 - \
                     pm.norm2_bc(samples,mu1)/v1)/2.))

        # fit GMM knowing post weights:
        mu0_est, mu1_est, v0_est, v1_est, l_est, llh = \
          pm.informedGMM_MV(noisy_samples, post_weights)

        # Some outputs:
        mus = np.array([[mu0,mu1], [mu0_est, mu1_est]]).transpose()
        vs = np.array([[v0,v1], [v0_est, v1_est]]).T
        ls = np.array([[1-l,l], [1-l_est, l_est]]).T

        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        for f in xrange(samples.shape[1]):
            pyhrf.verbose(1, get_2Dtable_string(mus[f,:,:], ['C0','C1'],
                                                ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            f = 1 #check this feature
            plot_gaussian_mixture(np.array([[mu0[f],mu1[f]], [v0,v1]]),
                                  props=[1-l,l], color='r')
            plot_gaussian_mixture(np.array([[mu0_est[f], mu1_est[f]],
                                            [v0_est, v1_est]]),
                                  props=[1-l_est, l_est], color='b')
            plt.hist(samples[:,f], color='g',normed=True)
            plt.show()

        assert_array_almost_equal([mu0_est, mu1_est], [mu0,mu1], decimal=1)
        assert_array_almost_equal([v0_est, v1_est], [v0,v1], decimal=1)
        assert_array_almost_equal([1-l_est,l_est], [1-l,l], decimal=1)


    def test_informedGMM_parameters(self):
        """
        Check that merge is in favour of non-activ at the same feature level,
        starting from singleton clusters.
        """
        pyhrf.verbose.setVerbosity(0)
        n_samples = 10000000

        # Generation of bi-Gaussian distribution
        mu0, v0 = 1., 0.6 # mean and standard deviation
        g0 = np.random.normal(mu0, v0, n_samples)
        l0 = 0.3
        print 'Gaussian 0: mu0 = %d, v0 = %d, lambda0 = %d' % (mu0, v0, l0)

        mu1, v1 = 5., 1.3 # mean and standard deviation
        g1 = np.random.normal(mu1, v1, n_samples)
        l1 = 0.7
        print 'Gaussian 1: mu1 = %d, v1 = %d, lambda1 = %d' % (mu1, v1, l1)

        features = g0.copy()
        features[:l1*len(features)] = g1[:l1*len(features)]

        alphas = 1 / (1 + (1-l0)/l0 * (v0/v1)**.5 * \
                np.exp(-((features-mu0)**2/v0 - (features-mu1)**2/v1)/2.))

        print 'N samples: ', n_samples
        print ''
        print 'Original parameters'
        print 'mu1:', mu1, 'mu0:', mu0
        print 'v1:', v1, 'v0:', v0
        print 'lambda:', l0
        print ''

        umu, uv, ul = pm.informedGMM_MV(features, alphas)

        print 'Updated parameters'
        print 'mu1:', umu[1], 'mu0:', umu[0]
        print 'v1:', uv[1], 'v0:', uv[0]
        print 'lambda:', ul[0]
        print ''

        assert_array_almost_equal(umu, [mu0, mu1])
        assert_array_almost_equal( uv, [v0, v1])


    def test_gmm_known_alpha0(self):
        """
        Test biGMM update with posterior weights equal to 0
        """
        plot = False
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = 2.
        v0 = 5.
        l = 0.
        mu1 = 0.
        v1 = 0.

        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0
        card_c1 = int(l*nsamples)
        samples[:card_c1] = np.random.randn(card_c1) * v1**.5 + mu1
        post_weights = np.zeros_like(samples)

        # fit GMM knowing post weights:
        mu_est, v_est, l_est = pm.informedGMM(samples, post_weights)
        assert_array_almost_equal(mu_est, [mu0,mu1], decimal=1)
        assert_array_almost_equal(v_est, [v0,v1], decimal=1)
        assert_array_almost_equal(l_est, [1-l,l], decimal=1)

        # Some outputs:
        mus = np.array([[mu0,mu1], mu_est]).T
        vs = np.array([[v0,v1], v_est]).T
        ls = np.array([[1-l,l], l_est]).T
        from pyhrf.tools import get_2Dtable_string
        pyhrf.verbose(1, 'means:')
        pyhrf.verbose(1, get_2Dtable_string(mus, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'vars:')
        pyhrf.verbose(1, get_2Dtable_string(vs, ['C0','C1'], ['true', 'estim']))
        pyhrf.verbose(1, 'weights:')
        pyhrf.verbose(1, get_2Dtable_string(ls, ['C0','C1'], ['true', 'estim']))

        if plot:
            import matplotlib.pyplot as plt
            from pyhrf.plot import plot_gaussian_mixture
            #plt.vlines(samples, 0, post_weights, 'r')
            #plt.vlines(samples, 0, 1-post_weights, 'b')
            plot_gaussian_mixture(np.array([[mu0,mu1], [v0,v1]]),
                                  props=[1-l,l], color='r')
            plot_gaussian_mixture(np.array([mu_est, v_est]),
                                  props=l_est, color='b')
            plt.hist(samples, color='g',normed=True)
            plt.show()

    def test_gmm_likelihood(self):
        """
        Test the log likelihood computation
        """
        pyhrf.verbose.setVerbosity(0)
        np.random.seed(2354) #make reproducible results

        # biGMM parameters:
        mu0 = 2.
        v0 = 5.
        mu1 = 10.
        v1 = 5.
        l = .4 #weight for component 1

        # generate samples from biGMM:
        nsamples = 10000
        samples = np.random.randn(nsamples) * v0**.5 + mu0
        card_c1 = int(l*nsamples)
        samples[:card_c1] = np.random.randn(card_c1) * v1**.5 + mu1

        # Calculate loglikelihood
        d = loglikelihood_computation(samples, mu0, v0, mu1, v1, l)
        assert_array_almost_equal(d, 0.0, decimal=1)
        ## fit GMM knowing post weights:
        #mu_est, v_est, l_est = informedGMM(samples, post_weights)
        #d2 = loglikelihood_computation(samples, mu0, v0, mu1, v1, l)


class FeatureExtractionTest(unittest.TestCase):

    def setUp(self):
        # called before any unit test of the class
        self.my_param = "OK"
        self.tmp_path = pyhrf.get_tmp_path() #create a temporary folder
        self.clean_tmp = True

    def tearDown(self):
        # called after any unit test of the class
        if self.clean_tmp:
            pyhrf.verbose(1, 'clean tmp path')
            shutil.rmtree(self.tmp_path)

    def test_new_obj(self):
        # a unit test
        result = my_func_to_test(self.my_param, output_path=self.tmp_path)
        assert result == "OK"


    def test_generate_features(self):
        pyhrf.verbose.setVerbosity(0)
        p = np.array([1,1,1,1,1,2,2,2,2,2], dtype=int)
        act_labels = np.array([0,0,1,1,0,0,0,1,1,1], dtype=int)
        feat_levels = {1: (np.array([.4, .3]),    #feats non-activ for parc 1
                           np.array([5., 7.]),),  #feats activ for parc 1
                       2: (np.array([.1, .2]),    #feats non-activ for parc 2
                           np.array([9., 8.]),)}  #feats activ for parc 2

        expected_features = np.array([[.4, .3],
                                      [.4, .3],
                                      [5., 7.],
                                      [5., 7.],
                                      [.4, .3],
                                      [.1, .2],
                                      [.1, .2],
                                      [9., 8.],
                                      [9., 8.],
                                      [9., 8.]])

        features = pm.generate_features(p, act_labels, feat_levels, 0.)
        assert_array_equal(features, expected_features)

    # Test feature extraction previous to parcellation
    def test_feature_extraction(self):
        pyhrf.verbose.setVerbosity(0)
        method = 'glm_deriv'
        data0 = simulate_fmri_data()
        dt = data0.simulation[0]['dt']
        time_length = data0.simulation[0]['duration']
        ncond = len(data0.simulation[0]['condition_defs'])
        labels = data0.simulation[0]['labels']
        territories = data0.simulation[0]['hrf_territories']
        ampl, pvalues, feats, bvars = pm.feature_extraction(data0, method, dt, time_length, ncond)
        assert_array_equal(feats.shape,bvars.shape)
        assert_equal(feats.shape[0],ampl.shape[0])
        fn1 = op.join(self.tmp_path, 'features_representation1.png')
        fn2 = op.join(self.tmp_path, 'features_representation2.png')
        fn3 = op.join(self.tmp_path, 'features_representation3.png')
        fn4 = op.join(self.tmp_path, 'features_representation4.png')
        name = pm.represent_features(feats, labels.T, 1.-pvalues, territories, 1, fn1)
        name = pm.represent_features(feats, labels.T, 1.-pvalues, territories, 0, fn2)
        mask = np.where(labels.flatten()==0)
        f = feats[mask,:]
        name = pm.represent_features(f[0,:,:], labels.T[mask], 1.-pvalues[mask], territories[mask], 1, fn3)
        mask = np.where(labels.flatten()==1)
        f = feats[mask,:]
        name = pm.represent_features(f[0,:,:], labels.T[mask], 1.-pvalues[mask], territories[mask], 1, fn4)
        print fn1
        #res = pm.represent_features(feats, labels.T, bvars[:,0])
        self.assertTrue(op.exists(fn1), msg='%s does not exist'%fn1)

        self.clean_tmp = False #HACK
    """
    # Test feature extraction previous to parcellation
    def test_feature_extraction(self):
        pyhrf.verbose.setVerbosity(0)
        method = 'glm_deriv'
        data0 = simulate_fmri_data()
        dt = data0.simulation[0]['dt']
        time_length = data0.simulation[0]['duration']
        ncond = len(data0.simulation[0]['condition_defs'])
        labels = data0.simulation[0]['labels']
        territories = data0.simulation[0]['hrf_territories']
        ampl, feats, bvars = pm.feature_extraction(data0, method, dt, time_length, ncond)
        assert_array_equal(feats.shape,bvars.shape)
        assert_equal(feats.shape[0],ampl.shape[0])
        fn = op.join(self.tmp_path, 'features_representation.png')
        name = pm.represent_features(feats, labels.T, ampl, territories, 1, fn)
        name = pm.represent_features(feats, labels.T, ampl, territories, 0, fn)
        mask = np.where(labels.flatten()==0)
        f = feats[mask,:]
        name = pm.represent_features(f[0,:,:], labels.T[mask], ampl[mask], territories[mask], 1, fn)
        mask = np.where(labels.flatten()==1)
        f = feats[mask,:]
        name = pm.represent_features(f[0,:,:], labels.T[mask], ampl[mask], territories[mask], 1, fn)
        print fn
        #res = pm.represent_features(feats, labels.T, bvars[:,0])
        self.assertTrue(op.exists(fn), msg='%s does not exist'%fn)
        self.clean_tmp = False #HACK
"""

def simulate_fmri_data(scenario='high_snr', output_path=None):
        ## Scenarios
        # low SNR level:
        if (scenario=='low_snr'):
            m_act= 3.8          # activation magnitude
            v_noise = 1.        # noise variance
            # 1.5 normally, 0,2 no noise
        else:  # high SNR level:
            m_act= 1.8          # activation magnitude
            v_noise = 1.5       # noise variance
        v_act=.25
        v_inact=.25

        conditions_def = [Condition(name='audio', m_act= m_act, v_act=v_act,
                                    v_inact=v_inact, label_map='house_sun'),
                          #Condition(name='audio', m_act= 5.8, v_act=.25,
                          #           v_inact=.25, label_map='squares_8x8'),
                          #Condition(name='video', m_act=2, v_act=.5,
                                    #v_inact=.5, label_map='activated'),
                        ]

        # HRFs mapped to the 3 parcels:
        duration = 25.
        dt = .5
        import pyhrf.boldsynth.scenarios as simu
        primary_hrfs = [
            simu.genBezierHRF(np.arange(0,duration+dt,dt), pic=[3,1],
                              normalize=True),
            simu.genBezierHRF(np.arange(0,duration+dt,dt), pic=[5,1],
                              normalize=True),
            simu.genBezierHRF(np.arange(0,duration+dt,dt), pic=[7,1],
                              normalize=True),
            simu.genBezierHRF(np.arange(0,duration+dt,dt), pic=[10,1],
                              picw = 6,ushoot=[22,-0.2], normalize=True),
        ]
        # Dictionnary mapping an item label to a quantity or a function that generates
        # this quantity
        simulation_steps = {
            'condition_defs' : conditions_def,
            'nb_voxels' : 400,
            # Labels
            'labels_vol' : simu.create_labels_vol, # 3D shape
            'labels' : simu.flatten_labels_vol, # 1D shape (flatten)
            # Nrls
            'nrls' : simu.create_time_invariant_gaussian_nrls,
            'dt': dt,
            'duration': duration,
            'dsf' : 2,
            'tr' : 1.,
            # Paradigm
            'paradigm' : simu.create_localizer_paradigm,
            'rastered_paradigm' : simu.rasterize_paradigm,
            # HRF
            'hrf_duration': duration,
            'primary_hrfs' : primary_hrfs, # derivative of several hrfs
            'nb_hrf_territories' : 4,
            #'hrf_territories_name' : '3_territories_8x8',
            'hrf_territories_name' : '4_territories',
            'hrf_territories' : simu.load_hrf_territories,
            'hrf' : simu.create_hrf_from_territories, # duplicate all HRF along all voxels
            # Stim induced
            'stim_induced_signal' : simu.create_stim_induced_signal,
            # Noise
            'v_gnoise' : v_noise,
            'noise' : simu.create_gaussian_noise, #requires bold_shape, v_gnoise
            # Drift
            'drift_order' : 4,
            'drift_var' : 11.,
            'drift' : simu.create_polynomial_drift,
            # Bold
            'bold_shape' : simu.get_bold_shape,
            'bold' : simu.create_bold_from_stim_induced,
            }
        simu_graph = Pipeline(simulation_steps)
        simu_graph.resolve()
        simu_vals = simu_graph.get_values()

        if output_path is not None:
            simu.simulation_save_vol_outputs(simu_vals, output_path)

        return FmriData.from_simulation_dict(simu_vals)



def create_features(size='2D', feat_contrast='high', noise_var=0.,
                    n_features=2):

    if size == '2D':
        true_parcellation = np.array([[1,1,1,1],
                                      [1,1,1,2],
                                      [1,1,2,2],
                                      [2,2,2,2]])
        act_clusters = np.array([[1,1,0,0],
                                 [1,1,0,0],
                                 [0,0,1,1],
                                 [1,0,1,1]])
        ker_mask = pgraph.kerMask2D_4n
    elif size == '1D':
        ker_mask = None

        true_parcellation = np.array([1,1,1,1,1,1,1,1,1,2,2,2])
        act_clusters =      np.array([0,1,1,0,0,0,0,1,0,1,1,1])
    else:
        raise Exception('Unsupported size scenario: %s' %str(size))


    mask = true_parcellation>0
    true_parcellation_flat = true_parcellation[np.where(mask)]
    act_clusters_flat = act_clusters[np.where(mask)]


    if feat_contrast == 'high':
        if n_features == 2:
            #                   non-act  act
            feat_levels = {1 : ([1., 1.],    [10, 17.]),
                           2 : ([1., 1.],    [40, 13.]),}
        elif n_features == 1:
            #                   non-act  act
            feat_levels = {1 : ([1.],    [10]),
                           2 : ([1.],    [40]),}
        else:
            raise Exception('nb features should be <= 2')

        #                  non-act  act
        act_levels = {1 : ([.1],    [.9]),
                      2 : ([.1],    [.9]),}

    elif feat_contrast == 'low':
        if n_features == 2:
            #                   non-act      act
            feat_levels = {1 : ([1., 1.],    [8.,  7.]),
                           2 : ([1., 1.],    [8.,  6.]),}
        elif n_features == 1:
            #                   non-act  act
            feat_levels = {1 : ([1.],    [8.]),
                           2 : ([1.],    [8.]),}
        else:
            raise Exception('nb features should be <= 2')


        #                  non-act  act
        act_levels = {1 : ([.1],    [.9]),
                      2 : ([.1],    [.9]),}

    else:
        raise Exception('Unsupported feature contrast scenario: %s' \
                        %str(feat_contrast))

    features = pm.generate_features(true_parcellation_flat,
                                    act_clusters_flat,
                                    feat_levels, noise_var=noise_var)

    var = np.ones_like(features)

    act = pm.generate_features(true_parcellation_flat, act_clusters_flat,
                               act_levels).squeeze()

    pyhrf.verbose(1, 'features:')
    pyhrf.verbose.printNdarray(1, features.T)

    pyhrf.verbose(1, 'activation levels:')
    pyhrf.verbose.printNdarray(1, act.T)

    pyhrf.verbose(1, 'variances:')
    pyhrf.verbose.printNdarray(1, var.T)

    n_samples, n_features = features.shape

    graph = pgraph.graph_from_lattice(mask, ker_mask)

    return true_parcellation_flat, features, graph, var, act, \
      act_clusters_flat, mask



class ParcellationTest(unittest.TestCase):

    def setUp(self):
        # called before any unit test of the class
        self.my_param = "OK"
        self.tmp_path = pyhrf.get_tmp_path() #create a temporary folder
        self.clean_tmp = True

    def tearDown(self):
        # called after any unit test of the class
        if self.clean_tmp:
            pyhrf.verbose(1, 'clean tmp path')
            shutil.rmtree(self.tmp_path)

    def test_new_obj(self):
        # a unit test
        result = my_func_to_test(self.my_param, output_path=self.tmp_path)
        assert result == "OK"


    def test_spatialward_against_ward_sk(self):
        """
        Check that pyhrf's spatial Ward parcellation is giving the same
        results as scikit's spatial Ward parcellation
        """
        pyhrf.verbose.setVerbosity(0)
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("1D", "high", 0.)

        nbc = len(np.unique(true_parcellation))

        act = np.zeros_like(act) + 1. #no activation levels
        ww = pm.spatial_ward(features, graph, nb_clusters=nbc)

        ww_sk = pm.spatial_ward_sk(features, graph, nb_clusters=nbc)

        assert_array_equal(ww.labels_, ww_sk.labels_)


    def test_spatialward_against_modelbasedspatialward(self):
        """
        Check that pyhrf's spatial Ward parcellation is giving the same
        results as scikit's spatial Ward parcellation
        """
        pyhrf.verbose.setVerbosity(0)
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", "high", 0.)

        nbc = len(np.unique(true_parcellation))

        act = np.zeros_like(act) + 1. #no activation levels
        ww = pm.spatial_ward(features, graph, nb_clusters=nbc)

        ww_mb = pm.spatial_ward_with_uncertainty(features, graph, var, act, \
                                                 nb_clusters = nbc, \
                                                 dist_type='uward2')
        p0 = pm.align_parcellation(ww.labels_, ww_mb.labels_)
        assert_array_equal(ww.labels_, p0)


    def test_uspatialward_formula(self):
        """
        Check that pyhrf's Uncertain spatial Ward parcellation is giving the same
        results as Uncertain spatial Ward parcellation modified formula
        """
        pyhrf.verbose.setVerbosity(0)
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("1D", "high", 0.)

        nbc = len(np.unique(true_parcellation))

        act = np.zeros_like(act) + 1. #no activation levels

        ww = pm._with_uncertainty(features, graph, var, act, nb_clusters=nbc,
                                  dist_type='uward')

        ww_formula = pm._with_uncertainty(features, graph, var, act,
                                          nb_clusters=nbc, dist_type='uward2')

        assert_array_equal(ww.labels_, ww_formula.labels_)


    def save_parcellation_outputs(self, pobj, mask):
        pm.ward_tree_save(pobj, self.tmp_path, mask)

    def test_spatialward_from_forged_features(self):
        """
        Test spatial Ward on forged features
        """
        pyhrf.verbose.setVerbosity(0)
        self.tmp_path = './' #hack
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", "high", 10.)

        nbc = len(np.unique(true_parcellation))

        act = np.zeros_like(act) + 1. #no activation levels
        ww = pm.spatial_ward(features, graph, nb_clusters=nbc)

        if 1:
            self.save_parcellation_outputs(ww, mask)
            tp = expand_array_in_mask(true_parcellation, mask)
            fn_tp = op.join(self.tmp_path, 'true_parcellation.nii')
            xndarray(tp, axes_names=MRI3Daxes[:mask.ndim]).save(fn_tp)

            print 'pyhrf_view %s/*' %self.tmp_path
            self.clean_tmp = False #hack

        pdist, common_parcels = parcellation_dist(ww.labels_,
                                                  true_parcellation)
        assert_array_equal(pdist, 3) #non-regression

    def test_wpu_from_forged_features(self):
        """
        Test spatial Ward with uncertainty on forged features
        """
        pyhrf.verbose.setVerbosity(0)
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", "high", 0.)

        nbc = len(np.unique(true_parcellation))

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=nbc, save_history=False)

        if 1:
            self.save_parcellation_outputs(ww, mask)
            tp = expand_array_in_mask(true_parcellation, mask)
            fn_tp = op.join(self.tmp_path, 'true_parcellation.nii')
            xndarray(tp, axes_names=MRI3Daxes[:mask.ndim]).save(fn_tp)

            print 'pyhrf_view %s/*' %self.tmp_path
            self.clean_tmp = False #hack

        pdist, common_parcels = parcellation_dist(ww.labels_,
                                                  true_parcellation)
        assert_array_equal(pdist, 3) #non-regression

    def test_gmm_from_forged_features(self):
        """
        Test spatial Ward with uncertainty on forged features
        """
        pyhrf.verbose.setVerbosity(0)
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", "high", 0.)

        nbc = len(np.unique(true_parcellation))

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=nbc, save_history=False,
                                              dist_type='mixt')

        if 1:
            self.save_parcellation_outputs(ww, mask)
            tp = expand_array_in_mask(true_parcellation, mask)
            fn_tp = op.join(self.tmp_path, 'true_parcellation.nii')
            xndarray(tp, axes_names=MRI3Daxes[:mask.ndim]).save(fn_tp)

            print 'pyhrf_view %s/*' %self.tmp_path
            self.clean_tmp = False #hack

        pdist, common_parcels = parcellation_dist(ww.labels_,
                                                  true_parcellation)
        assert_array_equal(pdist, 3) #non-regression


    def test_parcellation_spatialWard_2(self):
        """
        Test WPU on a simple case.
        """

        pyhrf.verbose.setVerbosity(0)
        features = np.array([[100.,100.,1.,1.],
                             [100.,100.,1.,1.],
                             [100.,100.,1.,1.]]).T
        n_samples, n_features = features.shape
        graph = pgraph.graph_from_lattice(np.ones((2,2)), pgraph.kerMask2D_4n)

        var = np.ones_like(features)
        var_ini = np.ones_like(features)
        act = np.ones_like(features[:,0])
        act_ini = np.ones_like(features[:,0])
        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act, var_ini,
                                              act_ini, nb_clusters=2)
        p = ww.labels_
        assert_array_equal(p, np.array([1,1,2,2]))


    def test_parcellation_spatialWard_act_level_1D(self):
        """
        Test the ability of WPU to 'jump' non-activating positions (1D case).
        """
        pyhrf.verbose.setVerbosity(0)
        np.seterr('raise')
        true_parcellation = np.array([1,1,1,1,1,1,1,1,1,3,3,3])
        act_labels =        np.array([0,1,1,0,0,0,0,1,0,1,1,1])
        #                      non-act  act
        feat_levels = {1 : ([1., 5.],    [10, 7.]),
                       3 : ([1., 2.],    [40, 3.]),}

        features = pm.generate_features(true_parcellation, act_labels,
                                        feat_levels)

        var = np.ones_like(features)

        #                  non-act  act
        act_levels = {1 : ([.3],    [4.]),
                      3 : ([.3],    [4.]),}

        act = pm.generate_features(true_parcellation, act_labels, act_levels)
        act = act.squeeze()

        pyhrf.verbose(1, 'features:')
        pyhrf.verbose.printNdarray(1, features.T)

        pyhrf.verbose(1, 'activation levels:')
        pyhrf.verbose.printNdarray(1, act)

        pyhrf.verbose(1, 'variances:')
        pyhrf.verbose.printNdarray(1, var.T)

        n_samples, n_features = features.shape
        graph = pgraph.graph_from_lattice(np.ones((1,n_samples)),
                                          pgraph.kerMask2D_4n)

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=2)
        p = ww.labels_
        if 0:
            fn = op.join(self.tmp_path, 'parcellation_tree.png')
            pyhrf.verbose(1, 'fig parcellation tree: %s' %fn)
            lab_colors = [('black','red')[l] for l in act_labels]
            pm.render_ward_tree(ww, fn, leave_colors=lab_colors)
            self.clean_tmp = False

        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation, tol=2)



    def test_parcellation_mmp_act_level_1D(self):
        """
        Test the ability of MMP to 'jump' non-activating positions (1D case).
        """
        pyhrf.verbose.setVerbosity(0)
        np.seterr('raise')

        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("1D", "high", 0.)

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=2,
                                              dist_type='mixt')
        p = ww.labels_
        if 0:
            fn = op.join(self.tmp_path, 'parcellation_tree.png')
            pyhrf.verbose(1, 'fig parcellation tree: %s' %fn)
            lab_colors = [('black','red')[l] for l in act_labels]
            pm.render_ward_tree(ww, fn, leave_colors=lab_colors)
            self.clean_tmp = False #hack

        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation, tol=2)



    def test_parcellation_spatialWard_act_level_2D(self):
        """
        Test the ability of WPU to 'jump' non-activating positions (2D case).
        """
        pyhrf.verbose.setVerbosity(0)

        feat_contrast = "high"
        noise_var = 0.
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", feat_contrast, noise_var)

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=2)
        p = ww.labels_
        if pyhrf.verbose.verbosity > 0:
            print 'true parcellation:'
            print expand_array_in_mask(true_parcellation, mask)
            print 'WPU parcellation:'
            print expand_array_in_mask(p, mask)
            print 'act labels:'
            print  act_labels

        if 0:
            fn = op.join(self.tmp_path, 'parcellation_tree.png')
            pyhrf.verbose(1, 'fig parcellation tree: %s' %fn)
            lab_colors = [('black','red')[l] for l in act_labels]
            pm.render_ward_tree(ww, fn, leave_colors=lab_colors)
            self.clean_tmp = False #HACK


        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation,
                                     tol_pos=act_labels==0)



    def test_parcellation_mmp_act_level_2D(self):
        """
        Test the ability of MMP to 'jump' non-activating positions (2D case).
        """
        pyhrf.verbose.setVerbosity(0)

        feat_contrast = "high"
        noise_var = 3.
        true_parcellation, features, graph, var, act, act_labels, mask = \
          create_features("2D", feat_contrast, noise_var)

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=2, dist_type='mixt')
        p = ww.labels_
        if pyhrf.verbose.verbosity > 0:
            print 'true parcellation:'
            print expand_array_in_mask(true_parcellation, mask)
            print 'MMP parcellation:'
            print expand_array_in_mask(p, mask)
            print 'act labels:'
            print  act_labels

        if 0:
            fn = op.join(self.tmp_path, 'parcellation_tree.png')
            pyhrf.verbose(1, 'fig parcellation tree: %s' %fn)
            lab_colors = [('black','red')[l] for l in act_labels]
            pm.render_ward_tree(ww, fn, leave_colors=lab_colors)
            self.clean_tmp = False #HACK


        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation,
                                     tol_pos=act_labels==0)



    def test_parcellation_spatialWard_variance_1D(self):
        """
        Test the ability of WPU to 'jump' non-activating positions (1D case).
        """
        pyhrf.verbose.setVerbosity(0)
        np.seterr('raise')
        true_parcellation = np.array([1,1,1,1,1,1,1,1,1,3,3,3])
        act_labels =        np.array([1,1,1,1,1,1,1,1,1,1,1,1])

        n = 0.5
        var = (np.random.randn(*true_parcellation.shape))[:,np.newaxis] * n
        features = (true_parcellation + var)
        act = act_labels.squeeze()

        pyhrf.verbose(1, 'features:')
        pyhrf.verbose.printNdarray(1, features.T)

        pyhrf.verbose(1, 'activation levels:')
        pyhrf.verbose.printNdarray(1, act)

        pyhrf.verbose(1, 'variances:')
        pyhrf.verbose.printNdarray(1, var.T)

        n_samples, n_features = features.shape
        graph = pgraph.graph_from_lattice(np.ones((1,n_samples)),
                                          pgraph.kerMask2D_4n)

        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                             nb_clusters=2)
        p = ww.labels_

        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation, tol=2)


    def test_parcellation_spatialWard_variance_2D(self):
        """
        Test the sensibility to variance (2D case).
        """
        pyhrf.verbose.setVerbosity(0)

        true_parcellation = np.array([[1,1,1,1],
                                      [1,1,1,2],
                                      [1,1,2,2],
                                      [2,2,2,2]])
        var = np.array([[0.1,-0.1,-0.1,-0.1],
                        [0.1,0.1,-0.1,-0.1],
                        [0.1,0.1,0.1,-0.1],
                        [0.1,0.1,0.1,0.1]])

        act_clusters = np.array([[1,1,1,1],
                                 [1,1,1,1],
                                 [1,1,1,1],
                                 [1,1,1,1]])

        mask = true_parcellation>0
        true_parcellation_flat = true_parcellation[np.where(mask)]
        act_clusters_flat = act_clusters[np.where(mask)]
        features = (true_parcellation+var).flatten()
        act = np.ones_like(act_clusters)

        pyhrf.verbose(1, 'features %s:' %str(features.shape))
        pyhrf.verbose.printNdarray(1, features.T)

        pyhrf.verbose(1, 'activation levels:')
        pyhrf.verbose.printNdarray(1, act.T)

        pyhrf.verbose(1, 'variances :')
        pyhrf.verbose.printNdarray(1, var.T)

        n_samples, n_features = features[:,np.newaxis].shape
        graph = pgraph.graph_from_lattice(mask, pgraph.kerMask2D_4n)
        if 0:
            print graph.shape
            print features.shape
            print var.flatten().shape
            print act_clusters_flat.shape
        ww = pm.spatial_ward_with_uncertainty(features[:,np.newaxis], graph,
                                              var.flatten()[:,np.newaxis],
                                              act_clusters_flat[:,np.newaxis],
                                              nb_clusters=2)
        p = ww.labels_
        if pyhrf.verbose.verbosity > 0:
            print 'true parcellation:'
            print true_parcellation
            print 'WPU parcellation:'
            print expand_array_in_mask(p, mask)
            print 'act labels:'
            print  act_clusters

        if 0:
            fn = op.join(self.tmp_path, 'parcellation_tree.png')
            print 'fig parcellation tree:', fn
            lab_colors = [('black','red')[l] for l in act_clusters_flat]
            pm.render_ward_tree(ww, fn, leave_colors=lab_colors)
            #self.clean_tmp = False #HACK


        # tolerate 2 differing positions, correspond to 2 non-active
        # positions in between two different clusters
        pm.assert_parcellation_equal(p, true_parcellation_flat,
                                     tol_pos=act_clusters_flat==0)


    def test_render_ward_tree(self):

        pyhrf.verbose.setVerbosity(0)

        features = np.array([[1.,2.,4.,8.,16,32]]).T
        act = np.array([1,1,1,0,0,0])
        var = np.ones_like(features)

        n_samples, n_features = features.shape
        graph = pgraph.graph_from_lattice(np.ones((1,n_samples)),
                                          pgraph.kerMask2D_4n)

        n_clusters = 1
        ww = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                              nb_clusters=n_clusters)

        fn = op.join(self.tmp_path, 'parcellation_tree.png')
        pyhrf.verbose(1, 'fig of parcellation tree: %s' %fn)
        item_colors = [['black','red'][l] for l in act]
        pm.render_ward_tree(ww, fn, leave_colors=item_colors)
        self.assertTrue(op.exists(fn), msg='%s does not exist'%fn)

        #self.clean_tmp = False #HACK


    def test_ward_distance_1D_v1(self):
        # Test: inertia is high between clusters and 0 in the same cluster
        pyhrf.verbose.setVerbosity(0)
        features = np.array([[10.,10.,10.,5.,5.,5.]]).T
        var = np.ones_like(features)
        mom_1 = np.array([1.,1.,1.,1.,1.,1.])
        c_r = np.array([ 1, 2, 3, 4, 5])
        c_c = np.array([ 0, 1, 2, 3, 4])
        var = np.ones_like(features)
        ini = np.array([0.,0.,0.,0.,0.])
        act = np.array([1.,1.,1.,1.,1.,1.])
        i1 = pm.compute_ward_dist(mom_1, features, c_r, c_c, var, act, ini)
        pyhrf.verbose(1, 'inertia:')
        pyhrf.verbose.printNdarray(1, i1)
        assert_equal(len(np.array(np.where(i1>0))),1)


    def test_ward_distance_1D_v2(self):
        # Test effect non activation in limit between clusters
        pyhrf.verbose.setVerbosity(0)
        features = np.array([[10.3,10.1,10.7,5.1,5.3,5.2]]).T
        var = np.ones_like(features)
        mom_1 = np.array([1.,1.,1.,1.,1.,1.])
        c_r = np.array([ 1, 2, 3, 4, 5])
        c_c = np.array([ 0, 1, 2, 3, 4])
        var = np.ones_like(features)
        ini = np.array([0.,0.,0.,0.,0.])
        act = np.array([1.,1.,0.1,0.1,1.,1.])
        i1 = pm.compute_ward_dist(mom_1, features, c_r, c_c, var, act, ini)
        pyhrf.verbose(1, 'inertia:')
        pyhrf.verbose.printNdarray(1, i1)
        assert_equal(np.argmax(i1),2)


    def test_ward_distance_2D(self):
        # Test
        pyhrf.verbose.setVerbosity(0)
        true_parcellation = np.array([[1,1,1,1],
                                      [1,1,1,2],
                                      [1,1,2,2],
                                      [2,2,2,2]])
        act_clusters = np.array([[1,1,0,0],
                                 [1,1,0,0],
                                 [1,1,1,0],
                                 [1,1,1,0]])

        mask = true_parcellation>0
        true_parcellation_flat = true_parcellation[np.where(mask)]
        act_clusters_flat = act_clusters[np.where(mask)]
        #                   non-act  act
        feat_levels = {1 : ([1.],    [10]),
                       2 : ([3.],    [40]),}
        features = pm.generate_features(true_parcellation_flat, act_clusters_flat,
                                        feat_levels,noise_var=0.)
        pyhrf.verbose.printNdarray(1, features[:,0].T)
        var = np.ones_like(features)
        #                  non-act  act
        act_levels = {1 : ([.3],    [4.]),
                      2 : ([.3],    [4.]),}
        act = pm.generate_features(true_parcellation_flat, act_clusters_flat,
                                   act_levels).squeeze()
        pyhrf.verbose.printNdarray(1, act)
        mom_1 = np.ones_like(act)
        c_r = np.array([ 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9,10,10,11,11,12,13,13,14,14,15,15])
        c_c = np.array([ 0, 1, 2, 0, 1, 4, 2, 5, 3, 6, 4, 5, 8, 6, 9, 7,10, 8, 9,12,10,13,11,14])
        ini = np.zeros((c_r.shape[0], 1))
        i1 = pm.compute_ward_dist(mom_1, features, c_r, c_c, var, act, ini)
        pyhrf.verbose(1, 'inertia:')
        pyhrf.verbose.printNdarray(1, i1.T)
        #assert_array_almost_equal(act_clusters(np.where(act_clusters==0)), inertia())

    def test_parcellation_spatialWard_5_sklearn(self):
        pyhrf.verbose.setVerbosity(0)
        features0 = np.ones((25,1))
        features0[10:] = 2
        n_samples, n_features = features0.shape
        noise = 0
        var = np.random.rand(n_samples,n_features)*noise
        var_ini = np.random.rand(n_samples,n_features)*noise
        act = np.ones_like(features0[:,0])
        act_ini = np.ones_like(features0[:,0])
        features = features0 + var
        graph = pgraph.graph_from_lattice(np.ones((5,5)), pgraph.kerMask2D_4n)
        p2 = pm.spatial_ward(features, graph, nb_clusters=2)
        assert_array_equal(p2, features0.squeeze())


    def test_parcellation_spatialWard_400_nonoise(self):
        pyhrf.verbose.setVerbosity(0)
        n_samples = 400.
        n_features = 1.
        im = np.concatenate((np.zeros(math.ceil(n_samples/2))+1, \
                np.zeros(math.floor(n_samples/2))+2)).reshape(n_samples,1).astype(np.int)
        n = 0
        features = im + np.random.randn(*im.shape) * n
        graph = pgraph.graph_from_lattice(np.ones((20,20)), pgraph.kerMask2D_4n)
        var = np.ones_like(features)
        var_ini = np.ones_like(features)
        act = np.ones_like(features[:,0])
        act_ini = np.ones_like(features[:,0])
        p0 = pm.spatial_ward(features, graph, nb_clusters=2)
        p = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                             var_ini, act_ini, nb_clusters=2)

        dist_total, common_parcels = parcellation_dist(p.labels_, im.squeeze()+1)
        assert_array_equal(dist_total, 0)


    def test_hemodynamic_parcellation_wpu_2D_high_SNR(self):
        """
        test WPU on features extracted from a 2D artificial fMRI data set,
        at high SNR
        """
        pyhrf.verbose.setVerbosity(0)
        data0 = simulate_fmri_data()
        method = 'glm_deriv'
        dt = data0.simulation[0]['dt']
        if pyhrf.verbose.verbosity > 1:
            print 'fdata:'
            print data0
        time_length = data0.simulation[0]['duration']
        ncond = len(data0.simulation[0]['condition_defs'])
        ampl, feats, bvars = pm.feature_extraction(data0, method, dt,
                                                   time_length, ncond)
        data0.build_graphs()
        ww = pm.spatial_ward_with_uncertainty(feats, data0.graphs[1], bvars, ampl,
                                              nb_clusters=4)
        print 'ww: '
        print ww
        p = ww.labels_
        print p
        true_parcellation_flat = data0.simulation[0]['hrf_territories']

        if len(data0.simulation[0]['labels']) > 1:
            act_clusters_flat = np.bitwise_or(*(a for a in data0.simulation[0]['labels']))
        else:
            act_clusters_flat = data0.simulation[0]['labels'][0]


        if pyhrf.verbose.verbosity > 0:
            mask = data0.roiMask
            print 'true parcellation:'
            print expand_array_in_mask(true_parcellation_flat, mask)
            print 'WPU parcellation:'
            print expand_array_in_mask(p, mask)
            print 'act labels:'
            print expand_array_in_mask(act_clusters_flat, mask)
        pm.assert_parcellation_equal(p, true_parcellation_flat.astype(int),
                                     tol_pos=act_clusters_flat==0)


    def test_hemodynamic_parcellation_GMM_2D_high_SNR(self):
        """
        test GMM-based parcellation on features extracted from a
        2D artificial fMRI data set, at high SNR
        """
        np.random.seed(5438)
        data0 = simulate_fmri_data('high_snr', self.tmp_path)
        pyhrf.verbose.setVerbosity(0)

        method = 'glm_deriv'
        dt = data0.simulation[0]['dt']
        if pyhrf.verbose.verbosity > 5:
            print 'fdata:'
            print data0
        time_length = data0.simulation[0]['duration']
        ncond = len(data0.simulation[0]['condition_defs'])
        ampl, pvals, feats, bvars = pm.feature_extraction(data0, method, dt,
                                                          time_length, ncond)

        data0.build_graphs()
        ww = pm.spatial_ward_with_uncertainty(feats, data0.graphs[1], bvars,
                                              1-pvals, nb_clusters=3,
                                              dist_type='mixt',
                                              save_history=False)


        p = ww.labels_
        true_parcellation_flat = data0.simulation[0]['hrf_territories']

        if len(data0.simulation[0]['labels']) > 1:
            labs = data0.simulation[0]['labels']
            act_clusters_flat = np.bitwise_or(*(a for a in labs))
        else:
            act_clusters_flat = data0.simulation[0]['labels'][0]


        if pyhrf.verbose.verbosity > 0:
            mask = data0.roiMask
            print 'true parcellation:'
            print expand_array_in_mask(true_parcellation_flat, mask)
            print 'MMP parcellation:'
            print expand_array_in_mask(p, mask)
            print 'act labels:'
            print expand_array_in_mask(act_clusters_flat, mask)

        if 1:
            self.save_parcellation_outputs(ww, mask)
            tp = expand_array_in_mask(true_parcellation_flat, mask)
            fn_tp = op.join(self.tmp_path, 'true_parcellation.nii')
            xndarray(tp, axes_names=MRI3Daxes[:mask.ndim]).save(fn_tp)


            # fn = op.join(self.tmp_path, 'parcellation_tree.png')
            # pyhrf.verbose(1, 'fig parcellation tree: %s' %fn)
            # lab_colors = [('black','red')[l] \
            #               for l in data0.simulation[0]['labels'][0]]
            # pm.render_ward_tree(ww, fn, leave_colors=lab_colors)

            print 'pyhrf_view %s/*' %self.tmp_path
            self.clean_tmp = False #hack

        pm.assert_parcellation_equal(p, true_parcellation_flat.astype(int),
                                     tol_pos=act_clusters_flat==0)



    def test_parcellation_spatialWard_400_variance(self):
        pyhrf.verbose.setVerbosity(0)
        n_samples = 400.
        n_features = 1.
        im = np.concatenate((np.zeros(math.ceil(n_samples/2))+1, \
                np.zeros(math.floor(n_samples/2))+2)).reshape(n_samples,1).astype(np.int)
        n = 0.5
        var = np.random.randn(*im.shape) * n
        features = im + var
        var_ini = np.ones_like(features)
        act = np.ones_like(features[:,0])
        act_ini = np.ones_like(features[:,0])
        graph = pgraph.graph_from_lattice(np.ones((20,20)), pgraph.kerMask2D_4n)
        p0 = pm.spatial_ward(features, graph, nb_clusters=2)
        p = pm.spatial_ward_with_uncertainty(features, graph, var, act,
                                             var_ini, act_ini, nb_clusters=2)
        from pyhrf.parcellation import parcellation_dist
        dist_total1, common_parcels = parcellation_dist(p0,
                                                        im.squeeze()+1)
        dist_total2, common_parcels = parcellation_dist(p.labels_,
                                                        im.squeeze()+1)
        print dist_total1
        print dist_total2
        assert_array_less(dist_total2, dist_total1)

        #p2 = pm.align_parcellation(im.squeeze(), p)
        #assert_array_equal(im.squeeze(), p2)


    def test_parcellation_history(self):

        tp, feats, g, v, act, act_labs, mask = create_features()
        nc = len(np.unique(tp))
        ww = pm.spatial_ward_with_uncertainty(feats, g, v, act,
                                              nb_clusters=nc)
        nvoxels = feats.shape[0]
        self.assertEqual(ww.history.shape, (nvoxels-nc, nvoxels))
        self.assertEqual(ww.history_choices.shape[0], nvoxels-nc)
        self.assertEqual(ww.history_choices.shape[2], nvoxels)

        #c_hist = ww.history.expand(mask, 'voxel', target_axes=MRI3Daxes)


    def test_uward_tree_save(self):
        pyhrf.verbose.setVerbosity(0)
        tp, feats, g, v, act, act_labs, mask = create_features()
        nc = len(np.unique(tp))
        ww = pm.spatial_ward_with_uncertainty(feats, g, v, act,
                                              nb_clusters=nc)
        pm.ward_tree_save(ww, self.tmp_path, mask)
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_features.nii'))
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_variances.nii'))

        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_activations.nii'))

        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_history.nii'))

        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_choice_history.nii'))
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_history.nii'))

        if 0:
            self.clean_tmp = False #hack
            print 'pyhrf_view %s/*nii' %self.tmp_path


    def test_ward_tree_save(self):
        pyhrf.verbose.setVerbosity(0)
        tp, feats, g, v, act, act_labs, mask = create_features()
        nc = len(np.unique(tp))
        ww = pm.spatial_ward_with_uncertainty(feats, g, v, act,
                                              nb_clusters=nc)
        pm.ward_tree_save(ww, self.tmp_path, mask)
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_features.nii'))
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_history.nii'))

        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_choice_history.nii'))
        assert_file_exists(op.join(self.tmp_path,
                                   'parcellation_uward_history.nii'))

        if 0:
            self.clean_tmp = False #hack
            print 'pyhrf_view %s/*nii' %self.tmp_path


    def test_mixtdist(self):
        """
        Check that merge is in favour of non-activ at the same feature level,
        starting from singleton clusters.
        """
        pyhrf.verbose.setVerbosity(0)

        features = np.array([[1.1, 9, 1.1]]).T
        alphas = np.array([ .1,.9,.9])

        c0 = np.array([1,0,0], dtype=int)
        c1 = np.array([0,1,0], dtype=int)
        c2 = np.array([0,0,1], dtype=int)
        cmasks = [c0, c1, c2]

        #merge 0,1
        dist_0_1 = pm.compute_mixt_dist(features, alphas, np.array([0]),
                                        np.array([1]), cmasks, [None])[0]
        pyhrf.verbose(1, 'merge dist(0,1): %f' %dist_0_1)

        #merge 1,2
        c1 = np.array([0,1,0], dtype=int)
        c2 = np.array([0,0,1], dtype=int)
        dist_1_2 =  pm.compute_mixt_dist(features, alphas, np.array([1]),
                                         np.array([2]), cmasks, [None])[0]
        pyhrf.verbose(1, 'merge dist(1,2): %f'%dist_1_2)

        assert_array_less(dist_0_1, dist_1_2)
