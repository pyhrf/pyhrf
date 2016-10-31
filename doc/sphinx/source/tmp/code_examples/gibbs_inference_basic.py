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
        post_var = 1 / (y.size * (1/s2 + 1/x_prior_var))
        post_mean = y.sum() * post_var / s2
        return np.random.randn() * post_var + post_mean


class MyGS(GibbsSampler):

    def __init__(self):
        GibbsSampler.__init__(self, [GSVar_X()], nb_its_max=100)

# Generate some data
x_true = 1.
y = x_true + np.random.randn(500) * .2

# Instanciate & run the Gibbs Sampler
gs = MyGS()
gs.set_variables({'y':y, 'noise_var': .04, 'x_prior_var':1000.})
gs.run()

# Grab outputs:
outputs = gs.get_outputs()
x_pm = outputs['x_obs_mean']
print 'Posterior Mean estimate of x:', x_pm
