# -*- coding: utf-8 -*-
import numpy as np
from pyhrf.sandbox.stats import GSVariable, GibbsSampler

class GSVar_X(GSVariable):

    def __init__(self):
        """
        Create a variable object handling the sampling of variable 'x'.
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

    def get_custom_init(self):
        """ Function called at Gibbs Sampling initialization
        if initialization scenario = 'custom'.
        Here, the mean of input data is a 'good guess'.
        """
        y = self.get_variable('y')
        return y.mean()

    def get_random_init(self):
        """ Function called at Gibbs Sampling initialization
        if initialization scenario = 'random'.
        Here, return a sample of the prior x ~ N(0,v_x)
        """
        v_x = self.get_variable_value('x_prior_var')
        return np.random.randn() * v_x**.5


class MyGS(GibbsSampler):

    def __init__(self):
        GibbsSampler.__init__(self, [GSVar_X()], nb_its_max=100)

# Generate some data
x_true = 1.
y = x_true + np.random.randn(500) * .2

# default init of variable x (zero):
gs = MyGS()
gs.set_variables({'y':y, 'noise_var': .04, 'x_prior_var':1000.})
gs.set_true_value('x', x_true)
gs.run()
x_pm = gs.get_outputs()['x_obs_mean']

print 'x PM with init to 0.:', x_pm

# custom init of variable x:
gs.reset()
gs.set_initialization('x', 'custom')
gs.run()
x_pm = gs.get_outputs()['x_obs_mean']

print 'x PM with custom init:', x_pm

# init of variable x to its true value:
gs.reset()
gs.set_initialization('x', 'truth')
gs.run()
x_pm = gs.get_outputs()['x_obs_mean']

print 'x PM with init to truth:', x_pm
