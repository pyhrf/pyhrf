# -*- coding: utf-8 -*-
#
"""
Create dummy artificial data: very short paradigm, very few voxels.
"""
from pyhrf import get_tmp_path
import numpy as np
import pyhrf.boldsynth.scenarios as simu
from pyhrf.tools import Pipeline

simulation_steps = {
  'dt' : 0.6,
  'dsf' : 4, #downsampling factor -> tr = dt * dsf = 2.4
  'mask' : np.array([[[1,1,1,1,1,1,1]]]),
  'labels' : np.array([[0,0,1,1,0,1,1]]),
  'mean_act' : 3.,
  'var_act' : 0.5,
  'mean_inact' : 0.,
  'var_inact' : 0.5,
  'nrls' : simu.create_bigaussian_nrls,
  'rastered_paradigm' : np.array([[0,0,1,0,0,0,1,0]]),
  'hrf' : simu.create_canonical_hrf,
  'v_noise' : 1.,
  'bold_shape' : simu.get_bold_shape,
  'noise' : simu.create_gaussian_noise,
  'stim_induced_signal' : simu.create_stim_induced_signal,
  'bold' : simu.create_bold,
  }

simulation = Pipeline(simulation_steps)
simulation.resolve()
simulation_items = simulation.get_values()
output_dir = get_tmp_path()
print 'Save simulation to:', output_dir
simu.save_simulation(simulation_items, output_dir=output_dir)
