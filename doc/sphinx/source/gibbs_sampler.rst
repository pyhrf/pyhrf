.. _gibbs_sampling:

=======================================
Gibbs Sampling implementation (sandbox)
=======================================

pyhrf provides tools to implement Gibbs Sampling for a prototyping purpose. 
The following features are available: 

    * control of initialization (section :ref:`gibbs_sampler_init`)
    * trajectory monitoring (section :ref:`gibbs_sampler_trajectory`)
    * online mean and variance computation 
    * checking of error w.r.t. ground-truth (in the context of inference)
    * short profiling of sample steps

..    * stopping criterion (in the context of inference)

The implementation requires the subclassing of two classes:

    * The top-level Gibbs Sampler: GibbsSampler class
    * Variables: GSVariable class. One for each model variable.

..
    Direct samling example:
    
    The goal is to sample the following bivariate normal distribution:
    
    .. math::
    
       (x,y) \sim \mathcal{N}\left(0, 
                                  \left [
                                  \begin{matrix}
                                  1 &\rho \\ 
                                  \rho & 1  
                                  \end{matrix} \right] \right)
    
    The conditional distributions for the Gibbs sampler are:
    
    .. math::
    
       (y | x) \sim \mathcal{N} (\rho x, 1 - \rho^2)
       (x | y) \sim \mathcal{N} (\rho y, 1 - \rho^2)
    
    First we subclass a GSVariable for each variable::
    
      from pyhrf.sandbox.stats import GSVariable
    
      class GSVar_X(GSVariable):
      
          def __init__(self):
              GSVariable.__init__(self, 'X', init_value=0.)
    
          def sample(self):
              y = self.get_variable_value('Y')
              rho = self.get_variable_value('rho')
              self.current_value[:] = np.random.randn(rho * y, 1 - rho**2)
    
      class GSVar_Y(GSVariable):
      
          def __init__(self):
              GSVariable.__init__(self, 'Y', init_value=0.)
    
          def sample(self):
              x = self.get_variable_value('X')
              rho = self.get_parameter_value('rho')
              self.current_value[:] = np.random.randn(rho * x, 1 - rho**2)          
    
      class BivariateNormalSampler(GibbsSampler):
    
          def __init__(self, rho):
              GibbsSampler.__init__(self, variables=[GSVar_X(), GSVar_Y()],
                                    parameters={'rho':rho})
    
      binormal_sampler = BivariateNormalSampler(rho=2.)
      x,y = binormal_sampler.sample(size=1000, burnin=100)
      
      import matplotlib.pyplot as plt
      plt.plot(x,y)
    
In the following, a :ref:`gibbs_sampler_inference_example` is presented where the Gibbs Sampler is
used to fit the trivial model :math:`y  = I_n x + noise` where :math:`y` is a noisy data vector and :math:`x` an unknown scalar.

This example is then complexified to illustrate the available features.

.. _gibbs_sampler_efficiency:

Computational efficiency
########################

"For a prototyping purpose" means that code is not fully optimal w.r.t. to 
computation speed and memory. In the case of direct sampling, the time spent 
during the overall sampling loop is critical, hence the proposed tools
are suitable only for testing. Once testing is done, one should consider 
implementing the validated sampler in a compiled language.
In the case of inference, the time spent during the overall sampling loop may be negligible compared with the time spent during each sampling step. In this case, 
the critical sampling steps can be implemented in C-code and the pyhrf 
Gibbs implementation can be relatively fast. 

.. _gibbs_sampler_inference_example:

Starting example
################

Consider the following model, where :math:`y` is the noisy observed data of size :math:`N` and we want to recover the unknown scalar value :math:`x`:

.. math:: 

   y = x I_N + b \\

   b \sim \mathcal{N}(0,\sigma^2 I_N), 
          \sigma^2=0.04 \\
   x \sim \mathcal{N}(0,v_x), v_x=1000.\\
   
The variance of :math:`x` is set to a high value to get a flat prior. The
variance of the noise is also assumed to be known.

..   y \in \mathcal{R}^N \\
   
Conditional posterior for :math:`x`:

.. math::

   (x | \sigma^2, v_x, y) \sim \mathcal{N}\left( \frac{\sum_N y_n}
                                                 {N (1/\sigma^2 + 1/v_x)},
                                            \frac{\sigma^2 v_x}
                                                 {N(v_x + \sigma^2)} \right)

                                                 
The sampling of "x" is handled by creating a specific subclass of 
pyhrf.sandbox.stats.GSVariable.
The basic required elements are:

    - the declaration of the name of the variable
    - the setting of its initialization (here set to 0.)
    - the implementation of the method *GSVariable.sample* that generates
      a new sample conditionally to the current other variable samples and data.

.. literalinclude:: code_examples/gibbs_inference_basic.py
   :language: python
   :linenos:


.. _gibbs_sampler_init:

Initialization
##############

Several initialization scenarios are available and are specified by the argument
*initialization* of GSVariable.__init__:

    - Using a custom value: either a scalar or a numpy array.
    - Random initialization. The method *GSVariable.get_random_init* must
      be implemented.
    - Initialization to the true value (in the context of inference). The
      true value must be set by the method *GSVariable.set_true_value*
    - Custom initialization. The method *GSVariable.get_custom_init* must
      be implemented.

.. literalinclude:: code_examples/gibbs_init.py
   :language: python
   :emphasize-lines: 28-42,65,73
   :linenos:


Sampling history
################

The history of a given numpy.ndarray quantity can be "automatically" tracked provided that it is modified *inplace* during sampling (the quantity is tracked via its reference).

Two different type of quantities are considered:

    - *sampled quantities*: their life cycle is the same as the gibbs samples.
      For example, it can be an intermediate result within a given 
      sampling step. The tracking starts at the begining of the gibbs sampling.
    - *observable quantities*: they are computed *after the burnin period*
      and thus do not take part in sampling. These computation are derived
      from the gibbs samples, hence the appelation "observable".
      For example, a posterior mean estimate is an observable. It is computed
      by default (attribute obs_mean of a GSVariable object).
      The tracking starts after the burnin period of the gibbs sampling.


Correpsonding to these two types, the initialization of a quantity tracking is done via:

    - track_sampled_quantity(quantity_name, quantity_numpy_ndarray, pace)
      The tracking of "quantity_numpy.ndarray" is associated to the name 
      "quantity_name". Recording starts at the beginning of the gibbs sampling
      and is incremented every "pace" iterations.
    - track_obs_quantity(quantity_name, quantity_numpy_ndarray, pace)      
      The tracking of "quantity_numpy.ndarray" is associated to the name 
      "quantity_name" and recording is performed every "pace" iterations.
      Recording starts after the burnin period of the gibbs sampling and is 
      incremented every "pace" iterations.

.. literalinclude:: code_examples/gibbs_tracking.py
   :language: python
   :emphasize-lines: 28-42,65,73
   :linenos:


Error w.r.t. ground truth 
##########################


When Gibbs Sampling is used in the context of inference and a ground truth (eg simulated value) is available for a given variable, one wants to compare the final estimate to this ground truth.


.. literalinclude:: code_examples/gibbs_truth_check.py
   :language: python
   :emphasize-lines: 28-42,65,73
   :linenos:
