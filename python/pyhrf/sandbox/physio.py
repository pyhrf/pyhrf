import os.path as op
import numpy as np

from pyhrf import Condition
from pyhrf.paradigm import Paradigm
from pyhrf.tools import Pipeline

import pyhrf.boldsynth.scenarios as simbase

PHY_PARAMS_FRISTON00 = {
    'model_name' : 'Friston00',
    'tau_s' : 1/.8,
    'eps' : .5,
    'eps_max': 10., #TODO: check this
    'tau_m' : 1.,
    'tau_f' : 1/.4,
    'alpha_w' : .2,
    'E0' : .8,
    'V0' : .02,
    'k1' : 7 * .8,
    'k2' : 2.,
    'k3' : 2 * .8 - .2}


PHY_PARAMS_FMRII = {
    'model_name' : 'fmrii',
    'tau_s' : 1/.65,
    'eps' : 1.,
    'eps_max': 10., #TODO: check this
    'tau_m' : .98,
    'tau_f' : 1/.41,
    'alpha_w' : .5,
    'E0' : .4,
    'V0' : .01,}

PHY_PARAMS_KHALIDOV11 = {
    'model_name' : 'Khalidov11',
    'tau_s' : 1.54,
    'eps' : .54,
    'eps_max': 10., #TODO: check this
    'tau_m' : 0.98,
    'tau_f' : 2.46,
    'alpha_w' : .33,
    'E0' : .34,
    'V0' : 1,
    'k1' : 7 * .34,
    'k2' : 2.,
    'k3' : 2 * .34 - .2}


#TODO: Donnet, Deuneux

from scipy.stats import truncnorm

def create_tbg_neural_efficacies(physiological_params, condition_defs, labels):
    """
    Create neural efficacies from a truncated bi-Gaussian mixture.

    Ars:
        - physiological_params (dict (<param_name> : <param_value>):
            parameters of the physiological model
        - condition_defs (list of pyhrf.Condition):
            list of condition definitions. Each item should have the following
            fields (moments of the mixture):
                - m_act (0<=float<eff_max): mean of activating component
                - v_act (0<float): variance of activating component
                - v_inact (0<float): variance of non-activating component
        - labels (np.array((nb_cond, nb_vox), int)): binary activation states

    Return:
        np.array(np.array((nb_cond, nb_vox), float))
        -> the generated neural efficacies

    TODO: settle how to relate brls and prls to neural efficacies
    """

    eff_max = physiological_params['eps_max']
    eff = []
    for ic,c in enumerate(condition_defs):
        labels_c = labels[ic]
        mask_activ = np.where(labels_c)
        eff_c = truncnorm.rvs(0, eff_max, loc=0., scale=c.v_inact**.5,
                              size=labels_c.size)
        # truncnorm -> loc is mean, scale is std_dev
        eff_c[mask_activ] = truncnorm.rvs(0, eff_max, loc=c.m_act,
                                          scale=c.v_act**.5, size=labels_c.sum())

        eff.append(eff_c)
    return np.vstack(eff)


def phy_integrate_euler(phy_params, tstep, stim, epsilon, Y0=None):
    """
    Integrate the ODFs of the physiological model with the Euler method.

    Args:
        - phy_params (dict (<param_name> : <param_value>):
            parameters of the physiological model
        - tstep (float): time step of the integration, in seconds.
        - stim (np.array(nb_steps, float)): stimulation sequence with a temporal
            resolution equal to the time step of the integration
        - epsilon (float): neural efficacy
        - Y0 (np.array(4, float) | None): initial values for the physiological
                                          signals.
                                          If None: [0, 1,   1, 1.]
                                                    s  f_in q  v
    Result:
        - np.array((4, nb_steps), float)
          -> the integrated physiological signals, where indexes of the first
          axis correspond to:
              0 : flow inducing
              1 : inflow
              2 : HbR
              3 : blood volume

    TODO: should the output signals be rescaled wrt their value at rest?
    """

    tau_s = phy_params['tau_s']
    tau_f = phy_params['tau_f']
    tau_m = phy_params['tau_m']
    alpha_w = phy_params['alpha_w']
    E0 = phy_params['E0']

    def cpt_phy_model_deriv(y, s, epsi, dest):
        N, f_in, v, q = y
        if f_in < 0.:
            #raise Exception('Negative f_in (%f) at t=%f' %(f_in, ti))
            #HACK
            print 'Warning: Negative f_in (%f) at t=%f' %(f_in, ti)
            f_in = 1e-4

        dest[0] = epsi*s - (N/tau_s)-((f_in - 1)/tau_f) #dNdt
        dest[1] = N #dfidt
        dest[2] = (1/tau_m)*(f_in-v**(1/alpha_w)) #dvdt
        dest[3] = (1/tau_m)*((f_in/E0)*(1-(1-E0)**(1/f_in)) - \
                             (q/v)*(v**(1/alpha_w))) #dqdt
        return dest

    res = np.zeros((stim.size+1,4))
    res[0,:] = Y0 or np.array([0., 1., 1., 1.])

    for ti in xrange(1, stim.size+1):
        cpt_phy_model_deriv(res[ti-1], stim[ti-1], epsilon, dest=res[ti])
        res[ti] *= tstep
        res[ti] += res[ti-1]

    return res[1:,:].T



def create_evoked_physio_signals(physiological_params, paradigm,
                                 neural_efficacies, dt, integration_step=.05):
    """
    Generate evoked hemodynamics signals by integrating a physiological model.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
             parameters of the physiological model.
             In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII ...
        - paradigm (pyhrf.paradigm.Paradigm) :
             the experimental paradigm
        - neural_efficacies (np.ndarray (nb_conditions, nb_voxels, float)):
             neural efficacies involved in flow inducing signal.
        - dt (float):
             temporal resolution of the output signals, in second
        - integration_step (float):
             time step used for integration, in second

    Returns:
        - np.array((nb_signals, nb_scans, nb_voxels), float)
          -> All generated signals, indexes of the first axis correspond to:
              - 0: flow inducing
              - 1: inflow
              - 2: blood volume
              - 3: [HbR]
    """

    #TODO: handle multiple conditions
    # -> create input activity signal [0, 0, eff_c1, eff_c1, 0, 0, eff_c2, ...]
    # for now, take only first condition
    first_cond = paradigm.get_stimulus_names()[0]
    stim = paradigm.get_rastered(integration_step)[first_cond][0]
    neural_efficacies = neural_efficacies[0]

    # response matrix intialization
    integrated_vars = np.zeros((4, neural_efficacies.shape[0], stim.shape[0]))
    for i, epsilon in enumerate(neural_efficacies):
        integrated_vars[:,i,:] = phy_integrate_euler(physiological_params,
                                                     integration_step, stim,
                                                     epsilon)

    #downsampling:
    nb_scans = stim.size
    dsf = int(dt/integration_step)
    return np.swapaxes(integrated_vars[:,:,::dsf][:,:,:nb_scans], 1, 2)

def create_bold_from_hbr_and_cbv(physiological_params, hbr, cbv):
    """
    Compute BOLD signal from HbR and blood volume variations obtained
    by a physiological model


    """
    # physiological parameters
    V0 = physiological_params['V0']
    k1 = physiological_params['k1']
    k2 = physiological_params['k2']
    k3 = physiological_params['k3']

    return V0 *( k1*(1-hbr) + k2*(1-hbr/cbv) + k3*(1-cbv) )


def create_physio_brf(physiological_params, response_dt=.5,
                      response_duration=25.,return_brf_q_v=False):
    """
    Generate a BOLD response function by integrating a physiological model and
    setting its driving input signal to a single impulse.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
            parameters of the physiological model.
            In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII ...
        - response_dt (float): temporal resolution of the response, in second
        - response_duration (float): duration of the response, in second

    Return:
        - np.array(nb_time_coeffs, float)
          -> the BRF (normalized)
        - also return brf_not_normalized, q, v when return_prf_q_v=True
          (for error checking of v and q generation in calc_hrfs)
    """

    p = Paradigm({'c':[np.array([0.])]}, [response_duration],
                 {'c':[np.array([1.])]})
    n = np.array([[1.]])
    s,f,v,q = create_evoked_physio_signals(physiological_params, p, n,
                                           response_dt)
    brf = create_bold_from_hbr_and_cbv(physiological_params, q[:,0], v[:,0])
    if return_brf_q_v:
        return  brf/ (brf**2).sum()**.5, q, v
    else:
        return  brf / (brf**2).sum()**.5


def create_physio_prf(physiological_params, response_dt=.5,
                      response_duration=25.,return_prf_q_v=False):
    """
    Generate a perfusion response function by setting the input driving signal
    of the given physiological model with a single impulse.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
            parameters of the physiological model.
            In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII ...
        - response_dt (float): temporal resolution of the response, in second
        - response_duration (float): duration of the response, in second

    Return:
        - np.array(nb_time_coeffs, float)
          -> the PRF
        - also return brf_not_normalized, q, v when return_prf_q_v=True
          (for error checking of v and q generation in calc_hrfs)
    """
    p = Paradigm({'c':[np.array([0.])]}, [response_duration],
                 {'c':[np.array([1.])]}) # response_dt to match convention
                                         # in JDE analysis
    n = np.array([[1.]])
    s,f,v,q = create_evoked_physio_signals(physiological_params, p, n,
                                           response_dt)
    prf = f[:,0] - f[0,0] #remove y-intercept
    if return_prf_q_v:
        return prf/ (prf**2).sum()**.5, q, v
    else:
        return prf / (prf**2).sum()**.5

def rescale_bold_over_perf(bold_stim_induced, perf_stim_induced,
                           bold_perf_ratio=5.):

    return bold_stim_induced/bold_stim_induced.max() * bold_perf_ratio * \
      perf_stim_induced.max()

def create_asl_from_stim_induced(bold_stim_induced_rescaled, perf_stim_induced,
                                 ctrl_tag_mat, dsf, perf_baseline, noise,
                                 drift=None, outliers=None):
    """
    Downsample stim_induced signal according to downsampling factor 'dsf' and
    add noise and drift (nuisance signals) which has to be at downsampled
    temporal resolution.
    """
    bold = bold_stim_induced_rescaled[0:-1:dsf,:].copy()
    perf = np.dot(ctrl_tag_mat, (perf_stim_induced[0:-1:dsf,:].copy() + \
                                 perf_baseline))

    asl = bold + perf
    if drift is not None:
        asl += drift
    if outliers is not None:
        asl += outliers
    asl += noise

    return asl



def simulate_asl_full_physio(output_dir=None, noise_scenario='high_snr',
                             spatial_size='tiny'):
    """
    Generate ASL data by integrating a physiological dynamical system.

    Ags:
        - output_dir (str|None): path where to save outputs as nifti files.
                                 If None: no output files
        - noise_scenario ("high_snr"|"low_snr"): scenario defining the SNR
        - spatial_size  ("tiny"|"normal") : scenario for the size of the map
                                            - "tiny" produces 2x2 maps
                                            - "normal" produces 20x20 maps

    Result:
        dict (<item_label (str)> : <simulated_item (np.ndarray)>)
        -> a dictionary mapping names of simulated items to their values

        WARNING: in this dict the 'bold' item is in fact the ASL signal.
                 This name was used to be compatible with JDE which assumes
                 that the functional time series is named "bold".
                 TODO: rather use the more generic label 'fmri_signal'.

    TODO: use magnetization model to properly simulate final ASL signal
    """

    drift_var = 10.
    dt = .5
    dsf = 2 #down sampling factor

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    elif spatial_size == 'random_small':
        lmap1, lmap2, lmap3 = 'random_small', 'random_small', 'random_small'
    else:
        lmap1, lmap2, lmap3 = 'icassp13', 'ghost', 'house_sun'

    if noise_scenario == 'high_snr':
        v_noise = 0.05
        conditions = [
            Condition(name='audio', m_act=10., v_act=.1, v_inact=.2,
                      label_map=lmap1),
            Condition(name='video', m_act=11., v_act=.11, v_inact=.21,
                      label_map=lmap2),
            Condition(name='damier', m_act=12., v_act=.12, v_inact=.22,
                      label_map=lmap3),
                      ]
    else: #low_snr
        v_noise = 2.
        conditions = [
            Condition(name='audio', m_act=1.6, v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', m_act=1.6, v_act=.3, v_inact=.3,
                      label_map=lmap2),
                      ]

    simulation_steps = {
        'dt' : dt,
        'dsf' : dsf,
        'tr' : dt * dsf,
        'condition_defs' : conditions,
        # Paradigm
        'paradigm' : simbase.create_localizer_paradigm_avd,
        # Labels
        'labels_vol' : simbase.create_labels_vol,
        'labels' : simbase.flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # Neural efficacy
        'neural_efficacies' : create_tbg_neural_efficacies,
        # BRF
        'primary_brf' : create_physio_brf,
        'brf' : simbase.duplicate_brf,
        # PRF
        'primary_prf' : create_physio_prf,
        'prf' : simbase.duplicate_prf,
        # Physiological model
        'physiological_params' : PHY_PARAMS_FRISTON00,
        ('flow_induction','perf_stim_induced','cbv','hbr') :
            create_evoked_physio_signals,
        'bold_stim_induced' : create_bold_from_hbr_and_cbv,
        # Noise
        'v_gnoise' : v_noise,
        'noise' : simbase.create_gaussian_noise_asl,
        # Drift
        'drift_order' : 4,
        'drift_var' : drift_var,
        'drift_coeffs': simbase.create_drift_coeffs_asl,
        'drift' : simbase.create_polynomial_drift_from_coeffs_asl,
        # ASL
        'ctrl_tag_mat' : simbase.build_ctrl_tag_matrix,
        'asl_shape' : simbase.calc_asl_shape,
        # Perf baseline #should be the inflow at rest ... #TODO
        'perf_baseline' : simbase.create_perf_baseline,
        'perf_baseline_mean' : 0.,
        'perf_baseline_var': 0.,
        # maybe rename to ASL (should be also modified in JDE)#TODO
        'bold' : simbase.create_asl_from_stim_induced,
        }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    simulation = simu_graph.get_values()

    if output_dir is not None:
        #simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))
        simbase.simulation_save_vol_outputs(simulation, output_dir)

        # f = open(op.join(output_dir, 'simulation.pck'), 'w')
        # cPickle.dump(simulation, f)
        # f.close()

    return simulation




def simulate_asl_physio_rfs(output_dir=None, noise_scenario='high_snr',
                           spatial_size='tiny'):
    """
    Generate ASL data according to a LTI system, with PRF and BRF generated
    from a physiological model.

    Args:
        - output_dir (str|None): path where to save outputs as nifti files.
                                 If None: no output files
        - noise_scenario ("high_snr"|"low_snr"): scenario defining the SNR
        - spatial_size  ("tiny"|"normal") : scenario for the size of the map
                                            - "tiny" produces 2x2 maps
                                            - "normal" produces 20x20 maps

    Result:
        dict (<item_label (str)> : <simulated_item (np.ndarray)>)
        -> a dictionary mapping names of simulated items to their values

        WARNING: in this dict the 'bold' item is in fact the ASL signal.
                 This name was used to be compatible with JDE which assumes
                 that the functional time series is named "bold".
                 TODO: rather use the more generic label 'fmri_signal'.
    """

    drift_var = 10.
    dt = .5
    dsf = 2 #down sampling factor

    if spatial_size == 'tiny':
        lmap1, lmap2, lmap3 = 'tiny_1', 'tiny_2', 'tiny_3'
    elif spatial_size == 'random_small':
        lmap1, lmap2, lmap3 = 'random_small', 'random_small', 'random_small'
    else:
        lmap1, lmap2, lmap3 = 'icassp13', 'ghost', 'house_sun'

    if noise_scenario == 'high_snr':
        v_noise = 0.05
        conditions = [
            Condition(name='audio', perf_m_act=5., perf_v_act=.1,
                      perf_v_inact=.2,
                      bold_m_act=15., bold_v_act=.1, bold_v_inact=.2,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=5., perf_v_act=.11,
                      perf_v_inact=.21,
                      bold_m_act=14., bold_v_act=.11, bold_v_inact=.21,
                      label_map=lmap2),
            Condition(name='damier', perf_m_act=12.,
                      perf_v_act=.12, perf_v_inact=.22,
                      bold_m_act=20., bold_v_act=.12, bold_v_inact=.22,
                      label_map=lmap3),
                      ]
    elif noise_scenario == 'low_snr_low_prl':
        v_noise = 7.
        scale = .3
        print 'noise_scenario: low_snr_low_prl'
        conditions = [
            Condition(name='audio', perf_m_act=1.6*scale, perf_v_act=.1,
                      perf_v_inact=.1,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=1.6*scale, perf_v_act=.1,
                      perf_v_inact=.1,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap2),
                      ]

    else: #low_snr
        v_noise = 2.
        conditions = [
            Condition(name='audio', perf_m_act=1.6, perf_v_act=.3,
                      perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', perf_m_act=1.6, perf_v_act=.3,
                      perf_v_inact=.3,
                      bold_m_act=2.2, bold_v_act=.3, bold_v_inact=.3,
                      label_map=lmap2),
                      ]

    simulation_steps = {
        'dt' : dt,
        'dsf' : dsf,
        'tr' : dt * dsf,
        'condition_defs' : conditions,
        # Paradigm
        'paradigm' : simbase.create_localizer_paradigm_avd,
        'rastered_paradigm' : simbase.rasterize_paradigm,
        # Labels
        'labels_vol' : simbase.create_labels_vol,
        'labels' : simbase.flatten_labels_vol,
        'nb_voxels': lambda labels: labels.shape[1],
        # Physiological model (for generation of RFs)
        'physiological_params' : PHY_PARAMS_FRISTON00,
        # Brls
        'brls' : simbase.create_time_invariant_gaussian_brls,
        # Prls
        'prls' : simbase.create_time_invariant_gaussian_prls,
        # BRF
        'primary_brf' : create_physio_brf,
        'brf' : simbase.duplicate_brf,
        # PRF
        'primary_prf' : create_physio_prf,
        'prf' : simbase.duplicate_prf,
        # Perf baseline
        'perf_baseline' : simbase.create_perf_baseline,
        'perf_baseline_mean' : 1.5,
        'perf_baseline_var': .4,
        # Stim induced
        'bold_stim_induced' : simbase.create_bold_stim_induced_signal,
        'perf_stim_induced' : simbase.create_perf_stim_induced_signal,
        # Noise
        'v_gnoise' : v_noise,
        'noise' : simbase.create_gaussian_noise_asl,
        # Drift
        'drift_order' : 4,
        'drift_var' : drift_var,
        'drift_coeffs':simbase.create_drift_coeffs_asl,
        'drift' : simbase.create_polynomial_drift_from_coeffs_asl,
        # Bold # maybe rename as ASL (should be handled afterwards ...
        'ctrl_tag_mat' : simbase.build_ctrl_tag_matrix,
        'asl_shape' : simbase.calc_asl_shape,
        'bold' : simbase.create_asl_from_stim_induced,
        }
    simu_graph = Pipeline(simulation_steps)

    # Compute everything
    simu_graph.resolve()
    simulation = simu_graph.get_values()

    if output_dir is not None:
        #simu_graph.save_graph_plot(op.join(output_dir, 'simulation_graph.png'))
        simbase.simulation_save_vol_outputs(simulation, output_dir)

        # f = open(op.join(output_dir, 'simulation.pck'), 'w')
        # cPickle.dump(simulation, f)
        # f.close()

    return simulation


#### Linearized system to characterize BRF - PRF relationship ####

# def  buildOrder1FiniteDiffMatrix_central_alternate(size,dt):
 #    """
 #    returns a toeplitz matrix
 #    for central differences
 #    """
 #    #instability in the first few data points when calculating prf (not seen when old form is used)
 #    from scipy.linalg import toeplitz

 #    r = np.zeros(size)
 #    c = np.zeros(size)
 #    r[1] = .5
 #    r[size-1] = -.5
 #    c[1] = -.5
 #    c[size-1] = .5
 #    # to fix the last grid point
 #    D = toeplitz(r,c).T
 #    D[0,size-1]=0
 #    D[size-1,0]=0
 #    D[size-1,size-2]=-1
 #    D[size-1,size-1]=1
 #    return D/(2*dt)

def  buildOrder1FiniteDiffMatrix_central(size,dt):
    """
    returns a toeplitz matrix
    for central differences

    to correct for errors on the first and last points:
    (due to the fact that there is no rf[-1] or rf[size] to average with)
    - uses the last point to calcuate the first and vis-versa
    - this is acceptable bc the rf is assumed to begin & end at steady state
      (thus the first and last points should both be zero)
    """
    from scipy.linalg import toeplitz

    r = np.zeros(size)
    c = np.zeros(size)
    r[1] = .5
    r[size-1] = -.5
    c[1] = -.5
    c[size-1] = .5
    return toeplitz(r,c).T/(2*dt)


def plot_calc_hrf(hrf1_simu, hrf1_simu_name, hrf1_calc, hrf1_calc_name,
                  hrf2_simu, hrf2_simu_name, dt):

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(121)
    t = np.arange(hrf1_simu.size) * dt  #TODO: find non-dt method to do this
    simu1 = plt.plot(t, hrf1_simu, label=hrf1_simu_name)
    calc1 = plt.plot(t, hrf1_calc, label=hrf1_calc_name)
    plt.legend()
    plt.title(hrf1_calc_name)
    plt.subplot(122)
    simu2 = plt.plot(t, hrf2_simu, label=hrf2_simu_name)
    plt.plot(t, hrf1_simu, label=hrf1_simu_name)
    plt.legend()
    plt.title(hrf2_simu_name)
    plt.show()

    return None

def linear_rf_operator(rf_size, phy_params, dt, calculating_brf=False):
    """
    Calculates the linear operator A needed to convert brf to prf & vis-versa
      prf = (A^{-1})brf
      brf = (A)prf

    Inputs:
      - size of the prf and/or brf (assumed to be same)
      - physiological parameters
      - time resolution of data:
      - if you wish to calculate brf (return A), or prf (return inverse of A)

    Outputs:
      - np.array of size (hrf_size,1) linear operator to convert hrfs
    """
    import numpy as np

    tau_m_inv = 1./phy_params['tau_m']
    alpha_w = phy_params['alpha_w']
    alpha_w_inv = 1./phy_params['alpha_w']
    E0 = phy_params['E0']
    V0 = phy_params['V0']
    k1 = phy_params['k1']
    k2 = phy_params['k2']
    k3 = phy_params['k3']
    c = tau_m_inv * ( 1 + (1-E0)*np.log(1-E0)/E0 )

    from pyhrf.sandbox.physio import buildOrder1FiniteDiffMatrix_central
    D = buildOrder1FiniteDiffMatrix_central(rf_size,dt) #numpy matrix
    eye = np.matrix(np.eye(rf_size))  #numpy matrix

    A3 = tau_m_inv*( (D + (alpha_w_inv*tau_m_inv)*eye).I )
    A4 = c * (D+tau_m_inv*eye).I - (D+tau_m_inv*eye).I*((1-alpha_w)*alpha_w_inv* tau_m_inv**2)* (D+alpha_w_inv*tau_m_inv*eye).I
    A = V0 * ( (k1+k2)*A4 + (k3-k2)* A3 )

    if (calculating_brf):
        return -A.A
    else: #calculating_prf
        return -(A.I).A

def calc_linear_rfs(simu_brf, simu_prf, phy_params, dt, normalized_rfs=True):
    """
    Calculate 'prf given brf' and 'brf given prf' based on the a linearization
    around steady state of the physiological model as described in Friston 2000.

    Input:
      - simu_brf, simu_prf: brf and prf from the physiological simulation
                            from which you wish to calculate the respective
                            prf and brf.
                            Assumed to be of size (1,hrf.size)
      - phy_params
      - normalized_rfs: set to True if simu_hrfs are normalized

    Output:
      - calc_brf, calc_prf: np.arrays of shape (hrf.size, 1)
      - q_linear, v_linear: q and v calculated according to the linearized model

    Note:
    These calculations do not account for any rescaling between brf and prf.
    This means the input simu_brf, simu_prf should NOT be rescaled.

   ** Warning**:
      - this function assumes prf.size == brf.size and uses this to build D, I
      - if making modifications:
        calc_brf, calc_prf have a truncation error (due to the finite difference        matrix used) on the order of O(dt)^2. If for any reason a hack is later         implemented to set the y-intecepts of brf_calc, prf_calc to zero by
        setting the first row of X4, X3 = 0, this will raise a singular matrix
        error in the calculation of calc_prf (due to X.I command), so this error        is helpful in this case
    """

    D = buildOrder1FiniteDiffMatrix_central(simu_prf.size,dt) #numpy matrix
    I = np.matrix(np.eye(simu_prf.size))  #numpy matrix
    #TODO: elimlinate prf.size dependency

    tau_m = phy_params['tau_m']
    tau_m_inv = 1./tau_m #when tau_m=1, singular matrix formed by (D+tau_m_inv*I)
    alpha_w = phy_params['alpha_w']
    alpha_w_inv = 1./phy_params['alpha_w']
    E0 = phy_params['E0']
    V0 = phy_params['V0']
    k1 = phy_params['k1']
    k2 = phy_params['k2']
    k3 = phy_params['k3']

    c = tau_m_inv * ( 1 + (1-E0)*np.log(1-E0)/E0 )

    #transform to (hrf.size,1) matrix for calcs
    simu_prf = np.matrix(simu_prf).transpose()
    simu_brf = np.matrix(simu_brf).transpose()

    X3 = tau_m_inv*( (D + (alpha_w_inv*tau_m_inv)*I).I )

    X4= c *(D+tau_m_inv*I).I - (D+tau_m_inv*I).I*((1-alpha_w)*alpha_w_inv*\
                               tau_m_inv**2)* (D+alpha_w_inv*tau_m_inv*I).I

    X = V0 * ( (k1+k2)*X4 + (k3-k2)* X3 )

    #for error checking
    q_linear = 1-X4*(-simu_prf)
    v_linear = 1-X3*(-simu_prf)

    calc_brf = X*(-simu_prf)

    calc_prf = -X.I*simu_brf

    #convert to np.arrays
    calc_prf = calc_prf.A
    calc_brf = calc_brf.A
    q_linear  = q_linear.A
    v_linear  = v_linear.A

    if normalized_rfs:
        calc_prf /= (calc_prf**2).sum()**.5
        calc_brf /= (calc_brf**2).sum()**.5

    return calc_brf, calc_prf, q_linear, v_linear

def run_calc_linear_rfs():
    """
    Choose physio parameters
    Choose to generate simu_rfs from multiple or single stimulus

    TODO:
    - figure out why there is an issue that perf_stim_induced is much greater than bold_stim_induced
    - figure out why when simu_brf=bold_stim_induced_rescaled,
      calc_brf is so small it appears to be 0
    """

    phy_params = PHY_PARAMS_FRISTON00
    #phy_params = PHY_PARAMS_KHALIDOV11

    multiple_stimulus_rf=False #to test calculations using a single stimulus rf
                                 #else, tests on a single stimulus rf
    if multiple_stimulus_rf:

        simu_items = simulate_asl_full_physio()
        #for rfs, rows are rfs, columns are different instances
        choose_rf = 1 # choose any number between 0 and simu_rf.shape[1]
        simu_prf = simu_items['perf_stim_induced'][:,choose_rf].T - \
                   simu_items['perf_stim_induced'][0,choose_rf]
        simu_brf = simu_items['bold_stim_induced'][:,choose_rf].T
        dt = simu_items['dt']
        q_dynamic = simu_items['hbr'][:,choose_rf]
        v_dynamic = simu_items['cbv'][:,choose_rf]

        normalized_rfs = False
    # if normalized simulated brfs and prfs are being used, then the comparison     between v and q, linear and dynamic, is no longer valid. Disregard the plot.
    else:
        dt = .05
        duration = 25.

        simu_prf, q_unused, v_unused = create_physio_prf(phy_params,
                                     response_dt=dt, response_duration=duration,
                                                        return_prf_q_v=True)

        simu_brf, q_dynamic, v_dynamic = create_physio_brf(phy_params,
                                     response_dt=dt, response_duration=duration,
                                                        return_brf_q_v=True)
        normalized_rfs = True

## deletable - no use for rescaling here
    #rescaling irrelevant to this simulation
    #simu_brf_rescale = rescale_bold_over_perf(simu_brf, simu_prf)
    #simu_brf = simu_brf_rescale

    #in testing: assert( simu_brf.shape == simu_prf_shape)?
##

    calc_brf, calc_prf, q_linear, v_linear = calc_linear_rfs(simu_brf, simu_prf,
                                                             phy_params, dt,
                                                             normalized_rfs)

    plot_results=True
    if  plot_results:
        plot_calc_hrf(simu_brf, 'simulated brf', calc_brf, 'calculated brf',
                      simu_prf, 'simulated prf', dt)

        plot_calc_hrf(simu_prf, 'simulated prf', calc_prf, 'calculated prf',
                      simu_brf, 'simulated brf', dt)

        #for debugging
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        t = np.arange(v_linear.size) * dt  #TODO: find non-dt method to do this
        plt.plot(t,v_linear, label='v linear')
        plt.plot(t, v_dynamic, label='v dynamic')
        plt.legend()
        plt.title('v')
        plt.subplot(122)
        plt.plot(t,q_linear, label='q linear')
        plt.plot(t, q_dynamic, label='q dynamic')
        plt.legend()
        plt.title('q')
        plt.show()

        # to see calc_brf and calc_prf on same plot (if calculating both)
        plt.figure()
        plt.plot(t, calc_brf, label='calculated brf')
        plt.plot(t, calc_prf, label='calculated prf')
        plt.legend()
        plt.title('calculated hrfs')

    return None

