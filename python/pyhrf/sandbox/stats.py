# -*- coding: utf-8 -*-

import logging

import numpy as np

import pyhrf

from pyhrf.ndarray import xndarray
from pyhrf.tools import array_summary


logger = logging.getLogger(__name__)


class Trajectory:
    """ Keep track of a numpy array that is modified _inplace_ iteratively
    TODO: when mature, should be moved to pyhrf.ndarray should replace pyhrf.jde.samplerbase.Trajectory
    """

    def __init__(self, variable, history_pace, history_start, max_iterations,
                 init_iteration=None, axes_names=None, axes_domains=None):
        """
        Args:
            *variable* is the tracked numpy array that has to be
            modified _inplace_ else the reference to it is lost.
            *history_pace* is the pace at which each value of *variable* is saved.
            *history_start* is the iteration when to start saving values of
                            *variable*
            *max_iterations* is the total maximal number of iterations
                           (for allocating memory)
            *init_iteration* sets the integer value of the first
                             saved iteration. -1 is useful to mark the init
                             value which is prior to the Oth iteration.
                             If None then the value of *variable* at the creation
                             of the Trajectory is no saved.
            *axes_names* is a list of string for the axes of *variable*.
                         Only used in *to_cuboid*
            *axes_domains* is a dict of numpy arrays for the domains of the axes.
                         Only used in *to_cuboid*
        """
        self.variable = variable
        self.axes_names = axes_names
        self.axes_domains = axes_domains
        self.hist_pace = history_pace
        self.hist_start = history_start

        if history_pace > 0:
            nsamples_max = (max_iterations - history_start) / history_pace + \
                           (((max_iterations - history_start) %
                             history_pace) > 0)
        else:
            nsamples_max = 0

        if init_iteration is not None:
            nsamples_max += 1  # +1 because of init

        self.history = np.zeros((nsamples_max,) + variable.shape,
                                dtype=variable.dtype)

        self.sample_count = 0
        self.saved_iterations = []
        if init_iteration is not None:
            #-1 is initialisation.
            self.history[0] = self.variable[:]
            self.saved_iterations.append(init_iteration)
            self.sample_count = 1

    def update(self, iteration):
        """ Record the current variable value """
        if self.hist_pace > 0 and iteration >= self.hist_start and \
                ((iteration - self.hist_start) % self.hist_pace) == 0:
            logger.debug('-> record: %s (sample_count: %d, hist shape: %s)',
                         str(self.variable), self.sample_count,
                         str(self.history.shape))
            self.history[self.sample_count] = self.variable[:]
            self.saved_iterations.append(iteration)
            self.sample_count += 1
        else:
            logger.debug("-> don't record (self.hist_pace: %d, "
                         "self.hist_start: %d)", self.hist_pace, self.hist_start)

    def get_last(self):
        """ Return the last saved element """
        if self.sample_count > 0:
            return self.history[self.sample_count - 1]
        else:
            raise Exception('No recorded value')

    def to_cuboid(self):
        """ Pack the current trajectory in a xndarray
        """
        if self.sample_count > 0:
            if self.axes_names is not None:
                an = ['iteration'] + self.axes_names
            else:
                an = ['iteration'] + \
                    ['axis_%02d' % i for i in xrange(self.variable.ndim)]

            if self.axes_domains is not None:
                ad = self.axes_domains.copy()
            else:
                ad = {}
            ad['iteration'] = np.array(self.saved_iterations, dtype=int)
            c = xndarray(self.history, axes_names=an, axes_domains=ad)
            return c
        else:
            return None  # raise Exception ?


### Gibbs Sampler ###
# When mature, the following should replace pyhrf.jde.samplerbase
# and be moved to pyhrf.stats.gibbs


class GSVariable:

    def __init__(self, name, initialization, do_sampling=True,
                 axes_names=None, axes_domains=None):
        """
        Create a variable to be used in a Gibbs Sampler.

        Args:
            *name* (str): the label of the variable. Must not be the same
                as another variable in the model.
            *initialization* (str or numpy.ndarray):
                If numpy.ndarray then this value will be used as init value
                for the sampling.
                If str then available choices are:
                    'random': init will be defined by method self.get_random_init
                    'custom': init will be defined by method self.get_custom_init
                    'truth': init will be set to true value
                             (see method self.set_true_value)
            *do_sampling* (bool): flag to enable sampling.
                If False then the variable will be constant and set to its
                init value.
            *axes_names* (list): names for the axes. Used for outputs
            *axes_domains* (dict): domains associated to axes. Used for outputs
        """
        self.name = name
        self.check_initialization_arg(initialization)
        self.initialization = initialization
        self.do_sampling = do_sampling
        self.axes_names = axes_names
        self.axes_domains = axes_domains
        if self.axes_domains is None:
            self.axes_domains = {}

        self.true_value = None

    def check_initialization_arg(self, ia):
        if not (isinstance(ia, (str, np.ndarray)) or
                np.isscalar(ia)):
            raise Exception('Wrong type %s for initialization argument.'
                            'Allowed: str, npumpy.ndarray, scalar' % ia.__class__)

        if isinstance(ia, str):
            choices = ['random', 'custom', 'truth']
            if not ia in choices:
                raise Exception('Wrong choice for initialization argument: %s. '
                                'Allowed: %s' % (ia, ','.join(choices)))

    def enable_sampling(self, flag=True):
        self.do_sampling = flag

    def set_initialization(self, init):
        self.check_initialization_arg(init)
        self.initialization = init

    def set_true_value(self, true_value):
        if self.axes_names is not None:
            assert self.true_value.ndim == len(self.axes_names)

        self.true_value = true_value

    def get_variable(self, vname):
        return self.sampler.get_variable(vname)

    def get_variable_value(self, vname):
        """
        Short-hand to get variable among all those defined
        in the parent sampler
        """
        return self.sampler.get_variable_value(vname)

    # Initialization methods

    def set_init_value(self):
        """ Set the initial value of self.current_value, depending on the
        *initialization* scenario (random, custom, truth).
        """

        if isinstance(self.initialization, str):
            if self.initialization == 'random':
                self.current_value = self.get_random_init()
            elif self.initialization == 'custom':
                self.current_value = self.get_custom_init()
            elif self.initialization == 'truth':
                if self.true_value is None:
                    raise Exception('Missing true value to init variable %s'
                                    % (self.name))
                self.current_value = self.true_value
        elif isinstance(self.initialization, np.ndarray) or \
                np.isscalar(self.initialization):
            self.current_value = self.initialization

        if np.isscalar(self.current_value):
            self.current_value = np.array([self.current_value])

        if not isinstance(self.current_value, np.ndarray):
            raise Exception('Initialization of variable "%s" '
                            'with scenario "%s" did not returned a '
                            'a numpy.ndarray, but: %s'
                            % (self.name, self.initialization,
                               str(self.current_value)))

        if self.current_value.ndim == 0:  # don't handle 0-d arrays
            self.current_value = self.current_value[np.newaxis]

        if self.axes_names is not None:
            assert self.current_value.ndim == len(self.axes_names)

    def get_random_init(self):
        """
        Must return a random numpy.ndarray that will then be used as
        init value for sampling. For example, it can be a sample from
        the prior distribution.
        This function will also be used to test for the sensitivity to
        initialization.
        """
        raise NotImplementedError()

    def get_custom_init(self):
        """
        Must return a numpy.ndarray. Consider initializing with a good
        guess so that sampling converges more quickly.
        """
        raise NotImplementedError()

    def _init_sampling(self, sampler):

        self.sampler = sampler
        self.set_init_value()

        if self.do_sampling:
            # init default observables (mean, variance):
            self.cumul = np.zeros(self.current_value.shape)
            self.cumul_disp = np.zeros(self.current_value.shape, dtype=float)
            self.obs_mean = np.zeros(self.current_value.shape, dtype=float)
            self.obs_var = np.zeros(self.current_value.shape, dtype=float)
            self.nb_its_obs = 0.  # to compute obs mean and var during sampling

            self.init_observables()

            # Enable tracking of samples, mean and variance:
            self.track_sampled_quantity(self.name + '_hist_smpl',
                                        self.current_value)
            self.track_obs_quantity(
                self.name + '_hist_obs_mean', self.obs_mean)
            self.track_obs_quantity(self.name + '_hist_obs_var', self.obs_var)

            self.init_sampling()

    def track_sampled_quantity(self, name, quantity, history_pace=None,
                               axes_names=None, axes_domains=None):

        self.sampler.track_sampled_quantity(name, quantity, history_pace,
                                            axes_names, axes_domains)

    def track_obs_quantity(self, name, quantity, history_pace=None,
                           axes_names=None, axes_domains=None):

        self.sampler.track_obs_quantity(name, quantity, history_pace,
                                        axes_names, axes_domains)

    def reset(self):
        pass

    def init_observables(self):
        pass

    def init_sampling(self):
        pass

    # Sampling methods
    def _sample_next(self):
        """ Internal method called by parent GibbsSampler to perform a sampling
        iteration
        """

        if self.do_sampling:
            v = self.sample()
            if np.isscalar(v):
                v = np.array([v])
            if v.shape != self.current_value.shape:
                raise Exception('Method *sample* returned an array of shape '
                                'inconsistent with previous value: got %s,'
                                'expected %s' % (str(v.shape),
                                                 str(self.current_value.shape)))
            # inplace modification so that current_value can be trackable
            # by Trajectory object
            self.current_value[:] = v
            logger.debug(
                'sample of %s: %s', self.name, str(self.current_value))

    def sample(self):
        """
        Draw a sample conditionally to the current Gibbs Sampler state.
        Must return a numpy.ndarray.

        Variables which have been registered in the parent GibbsSampler object
        can be retrieved via methods self.get_variable(var_name) and
        self.get_variable_value(var_name)
        """
        raise NotImplementedError()

    def _update_observables(self):
        self.nb_its_obs += 1.
        self.cumul += self.current_value
        np.divide(self.cumul, self.nb_its_obs, self.obs_mean)
        self.cumul_disp += (self.current_value - self.obs_mean) ** 2
        np.divide(self.cumul_disp, self.nb_its_obs, self.obs_var)

        self.update_observables()

    def update_observables(self):
        """ Update quantities after the burnin period """
        pass

    # End of sampling

    def _finalize_sampling(self):
        pass

    def get_estim_value_for_check(self):
        return self.obs_mean

    def get_true_value_for_check(self):
        return self.true_value

    def get_accuracy_against_truth(self, abs_error, rel_error, fv, tv,
                                   atol, rtol):
        """ Return the accuray of the estimate *fv*, compared to the true
        value *tv*
        """
        # same criterion as np.allclose:
        acc = abs_error <= (atol + rtol * np.maximum(np.abs(tv),
                                                     np.abs(fv)))
        return self.axes_names, acc

    def check_against_truth(self, atol, rtol, inaccuracy_handling='print'):

        fv = self.get_estim_value_for_check()
        tv = self.get_true_value_for_check()

        if tv is None:
            logger.info(
                'Warning: no true val to check against for %s', self.name)
        elif self.do_sampling:  # when sampling is off, assume there is no need
                               # to check against truth

            abs_error = np.abs(tv - fv)
            rel_error = abs_error / np.maximum(np.abs(tv), np.abs(fv))

            acc = self.get_accuracy_against_truth(abs_error, rel_error, fv, tv,
                                                  atol, rtol)
            is_accurate = acc[1].all()
            logger.info('Check error for "%s" -> '
                        'estim: %s, truth: %s, atol=%f, rtol=%f', self.name,
                        str(fv), str(tv), atol, rtol)

            logger.info('Fit error for "%s": avg aerr=%f, avg rerr=%f, '
                        'is_accurate=%s', self.name, abs_error.mean(),
                        rel_error.mean(), is_accurate)
            if not is_accurate:
                m = "Final value of %s is not close to " \
                    "true value.\n -> aerror: %s\n -> rerror: %s\n" \
                    " Final value:\n %s\n True value:\n %s\n" \
                    % (self.name, array_summary(abs_error),
                       array_summary(rel_error), str(fv), str(tv))
                if inaccuracy_handling == 'raise':
                    raise Exception(m)
                elif inaccuracy_handling == 'print':
                    print '\n'.join(['!! ' + s for s in m.split('\n')])

            self.truth_checking_report = {
                'abs_error': abs_error,
                'rel_error': rel_error}

    def _get_outputs(self, output_type='ndarray'):
        """
        output_type : 'ndarray' or 'cuboid'
        """

        outputs = {}

        # Default outputs: post. mean, post. var, trajectories.
        on = self.name + '_obs_mean'
        if output_type == 'cuboid':
            outputs[on] = xndarray(self.obs_mean, axes_names=self.axes_names,
                                   axes_domains=self.axes_domains)
        else:
            outputs[on] = self.obs_mean

        on = self.name + '_obs_var'
        if output_type == 'cuboid':
            outputs[on] = xndarray(self.obs_var, axes_names=self.axes_names,
                                   axes_domains=self.axes_domains)
        else:
            outputs[on] = self.obs_var

        # Custom outputs:
        self.set_outputs(outputs, output_type)

        return outputs

    def set_outputs(self, outputs, output_type='ndarray'):
        """
        Args:
            - outputs (dict): dictionary to be updated with custom outputs.
            - output_type (str): 'ndarray' or 'cuboid'
        Return: None
        """
        pass


class GibbsSampler:

    def __init__(self, sampled_variables, nb_its_max, obs_pace=1, burnin=.3,
                 sample_hist_pace=-1, obs_hist_pace=-1,):

        self.variables = {}
        self.sampled_variables = sampled_variables

        for v in sampled_variables:
            self.set_variable(v.name, v)

        def get_fraction_or_nb(nb, tot):
            if nb > 0. and nb < 1.:
                return int(round(tot * nb))
            else:
                return nb

        self.nb_its_max = nb_its_max
        self.burnin = get_fraction_or_nb(burnin, nb_its_max)
        self.smpl_hist_pace = get_fraction_or_nb(sample_hist_pace, nb_its_max)
        self.obs_hist_pace = get_fraction_or_nb(obs_hist_pace, nb_its_max)
        self.tracked_quantities = {}

        logger.info('GibbsSampler init. Burnin: %d, nb_its_max: %d, '
                    'smpl_hist_pace: %d, obs_hist_pace: %d,', self.burnin,
                    self.nb_its_max, self.smpl_hist_pace, self.obs_hist_pace)

        # TODO: global hist pace

    def set_variable(self, name, var):
        if not (isinstance(var, (np.ndarray, GSVariable)) or
                np.isscalar(var)):
            raise Exception('Wrong variable type: %s. Allowed: numpy.ndarray, '
                            'GSVariable or scalar' % var.__class__)

        if self.variables.has_key(name):
            raise Exception('Variable %s already registered')

        self.variables[name] = var

    def set_variables(self, var_dict):
        for k, v in var_dict.iteritems():
            self.set_variable(k, v)

    def get_variable(self, vname):
        v = self.variables.get(vname, None)
        if v is None:
            raise Exception('Unregistered variable %s' % vname)
        else:
            return v

    def get_variable_value(self, vname):

        v = self.get_variable(vname)
        if isinstance(v, np.ndarray) or np.isscalar(v):
            return v
        elif isinstance(v, GSVariable):
            return v.current_value

    def set_true_values(self, true_values):
        for vname, true_value in true_values.iteritems():
            self.get_variable(vname).set_true_value(true_value)

    def set_true_value(self, vname, true_value):
        self.get_variable(vname).set_true_value(true_value)

    def set_initialization(self, vname, init):
        self.get_variable(vname).set_initialization(init)

    def reset(self):
        """
        Reset the Gibbs Sampler:
            - remove all previous history of quantities (trajectories)
            - call reset method of all variables
        """
        self.tracked_quantities = {}
        for v in self.sampled_variables:
            v.reset()

    def stop_criterion(self, iteration):
        return False

    def iterate_sampling(self):
        it = 0
        while it < self.nb_its_max and not self.stop_criterion(it):
            yield it
            it += 1

    def _track_quantity(self, q, name, axes_names, axes_domains,
                        history_pace, hist_start):
        if not self.tracked_quantities.has_key(name):
            trajectory = Trajectory(q, history_pace, hist_start, self.nb_its_max,
                                    axes_names, axes_domains)
            self.tracked_quantities[name] = trajectory
        else:
            raise Exception('Quantity %s already tracked' % name)

    def track_sampled_quantity(self, name, q, history_pace=None, axes_names=None,
                               axes_domains=None):
        if history_pace is None:
            history_pace = self.smpl_hist_pace

        self._track_quantity(q, name, axes_names, axes_domains, history_pace,
                             hist_start=0)

    def track_obs_quantity(self, name, q, history_pace=None, axes_names=None,
                           axes_domains=None):

        if history_pace is None:
            history_pace = self.obs_hist_pace

        self._track_quantity(q, name, axes_domains, axes_domains,
                             history_pace, hist_start=self.burnin)

    def run(self):

        for v in self.sampled_variables:
            v._init_sampling(self)

        for it in self.iterate_sampling():

            for v in self.sampled_variables:
                v._sample_next()

                if it >= self.burnin:
                    v._update_observables()

            for tname, trajectory in self.tracked_quantities.iteritems():
                logger.debug('Update Trajectory "%s" (it=%d) ...', tname, it)
                trajectory.update(it)

            #self.callback(it, self)

        for v in self.sampled_variables:
            v._finalize_sampling()

    def check_against_truth(self, default_atol=0.1, default_rtol=0.1,
                            var_specific_atol=None, var_specific_rtol=None,
                            inaccuracy_handling='print'):
        """
        """
        if var_specific_rtol is None:
            var_specific_rtol = {}
        if var_specific_atol is None:
            var_specific_atol = {}

        for v in self.sampled_variables:
            atol = var_specific_atol.get(v.name, default_atol)
            rtol = var_specific_rtol.get(v.name, default_rtol)
            v.check_against_truth(atol=atol, rtol=rtol,
                                  inaccuracy_handling=inaccuracy_handling)

    def get_outputs(self, output_type='ndarray'):
        """
        output_type : 'ndarray' or 'cuboid'
        """

        outputs = {}

        for v in self.sampled_variables:
            self._update_outputs(outputs, v._get_outputs(output_type))

        for qname, q in self.tracked_quantities.iteritems():
            if output_type == 'cuboid':
                ctrajectory = q.to_cuboid()
                if ctrajectory is not None:
                    self._add_output(outputs, qname, ctrajectory)
                else:
                    logger.info("Trajectory '%s' -> no output")
            else:
                self._add_output(outputs, qname, q.history)

        return outputs

    def _add_output(self, cur_outputs, new_output_name, new_output_value):
        if cur_outputs.has_key(new_output_name):
            raise Exception('Output %s already defined' % new_output_name)
        else:
            cur_outputs[new_output_name] = new_output_value

    def _update_outputs(self, cur_outputs, new_outputs):
        for k, v in new_outputs.iteritems():
            self._add_output(cur_outputs, k, v)
