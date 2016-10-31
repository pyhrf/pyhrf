# -*- coding: utf-8 -*-
import numpy as np
from pyhrf.sandbox.stats import GibbsSampler, GSVariable

class GSVar_X(GSVariable):

    def __init__(self):
        """
        Create a variable object handling the sampling of variable 'x'.
        The sampling init value is set to 0.
        """
        GSVariable.__init__(self, 'x', initialization=0.)

    def init_sampling(self):
        # The tracked quantities have to inited here so that
        # their references remain the same during sampling
        self.post_mean = np.zeros_like(self.current_value)
        self.post_var = np.zeros_like(self.current_value)

        self.track_sampled_quantity('smpl_post_mean', self.post_mean)
        self.track_sampled_quantity('smpl_post_var', self.post_var)

    def sample(self):
        """
        Sample variable X according to:
        N( \sum_N y_n / (N * (1/v_x + \sigma^2)),
           v_x \sigma^2 / (N (v_x + \sigma^2) )
        """

        # retrieve other quantities:
        y = self.get_variable_value('y')
        s2 = self.get_variable_value('noise_var')
        x_prior_var = self.get_variable_value('x_prior_var')

        # Do the sampling:
        self.post_var[:] = 1 / (y.size * (1/s2 + 1/x_prior_var))
        self.post_mean[:] = y.sum() * self.post_var / s2

        return np.random.randn() * self.post_var + self.post_mean


class MyGS(GibbsSampler):

    def __init__(self, sample_hist_pace, obs_hist_pace):
        GibbsSampler.__init__(self, [GSVar_X()], nb_its_max=10,
                              sample_hist_pace=sample_hist_pace,
                              obs_hist_pace=obs_hist_pace)

# Generate some data
x_true = 1.
y = x_true + np.random.randn(500) * .2

# Instanciate & run the Gibbs Sampler
gs = MyGS(sample_hist_pace=1, obs_hist_pace=1)
gs.set_variables({'y':y, 'noise_var': .04, 'x_prior_var':1000.})
gs.run()

# Grab tracked quantity:
outputs = gs.get_outputs()
print 'Tracking of post mean:'
print outputs['smpl_post_mean']

print 'Tracking of post var:'
print outputs['smpl_post_var']
