# -*- coding: utf-8 -*-

import numpy as np

from pyhrf.paradigm import Paradigm
from scipy.stats import truncnorm


PHY_PARAMS_FRISTON00 = {
    'model_name': 'Friston00',
    'tau_s': 1 / .8,
    'eps': .5,
    'eps_max': 10.,
    'tau_m': 1.,
    'tau_f': 1 / .4,
    'alpha_w': .2,
    'E0': .8,
    'V0': .02,
    'TE': 0.04,
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 0.4,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}         # 2 * E_0 - .2

PHY_PARAMS_DONNET06 = {
    'model_name': 'Donnet06',
    'tau_s': 1.54,
    'eps': .5,
    'eps_max': 10.,  # TODO: check this
    'tau_m': 0.98,
    'tau_f': 2.5,
    'alpha_w': .33,
    'E0': .34,
    'V0': 0.05,   # obata 0.04, Friston and others 0.02, Griffeth13 0.05
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 1.43,  # 0.4 or 1
    'TE': 0.04,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}

PHY_PARAMS_DENEUX06 = {
    'model_name': 'Deneux06',
    'tau_s': .49,
    'eps': .89,        # new alpha
    'eps_max': 10.,
    'tau_m': .27,      # new tau_0
    'tau_f': .19,
    'alpha_w': .63,    # new w
    'E0': .33,
    'V0': .05,
    'TE': 0.04,
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 0.4,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}         # 2 * E_0 - .2

PHY_PARAMS_HAVLICEK11 = {
    'model_name': 'Havlicek11',
    'tau_s': 1.54,
    'eps': .54,
    'eps_max': 10.,  # TODO: check this
    'tau_m': 0.98,
    'tau_f': 2.63,
    'alpha_w': .34,
    'E0': .32,
    'V0': .04,   # obata 0.04, Friston and others 0.02, Griffeth13 0.05
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 1.43,  # 0.4 or 1
    'TE': 0.04,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}

PHY_PARAMS_KHALIDOV11 = {
    'model_name': 'Khalidov11',
    'tau_s': 1.54,
    'eps': .54,
    'eps_max': 10.,  # TODO: check this
    'tau_m': 0.98,
    'tau_f': 2.46,
    'alpha_w': .33,
    'E0': .34,
    'V0': 1,   # obata 0.04, Friston and others 0.02, Griffeth13 0.05
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 1.43,  # 0.4 or 1
    'TE': 0.04,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}

PHY_PARAMS_DUARTE12 = {
    'model_name': 'Duarte12',
    'tau_s': 0.59,
    'eps': 1.,
    'eps_max': 10.,  # TODO: check this
    'tau_m': 0.91,
    'tau_f': 0.4,
    'alpha_w': .32,
    'E0': .34,
    'V0': .02,   # obata 0.04, Friston and others 0.02, Griffeth13 0.05
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 1.43,  # 0.4 or 1
    'TE': 0.04,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}


def buildOrder1FiniteDiffMatrix_central(size, dt):
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
    r[size - 1] = -.5
    c[1] = -.5
    c[size - 1] = .5
    return toeplitz(r, c).T / dt  # WARNING!! Modified. Before: (2*dt)


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
    for ic, c in enumerate(condition_defs):
        labels_c = labels[ic]
        mask_activ = np.where(labels_c)
        eff_c = truncnorm.rvs(0, eff_max, loc=0., scale=c.v_inact ** .5,
                              size=labels_c.size)
        # truncnorm -> loc is mean, scale is std_dev
        eff_c[mask_activ] = truncnorm.rvs(0, eff_max, loc=c.m_act,
                                          scale=c.v_act ** .5,
                                          size=labels_c.sum())

        eff.append(eff_c)
    return np.vstack(eff)


def phy_integrate_euler(phy_params, tstep, stim, epsilon, Y0=None):
    """
    Integrate the ODFs of the physiological model with the Euler method.

    Args:
        - phy_params (dict (<param_name> : <param_value>):
            parameters of the physiological model
        - tstep (float): time step of the integration, in seconds.
        - stim (np.array(nb_steps, float)): stimulation sequence with temporal
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

    epsilon = phy_params['eps']  # WARNING!! Added to compute figures
    tau_s = phy_params['tau_s']
    tau_f = phy_params['tau_f']
    tau_m = phy_params['tau_m']
    alpha_w = phy_params['alpha_w']
    E0 = phy_params['E0']

    def cpt_phy_model_deriv(y, s, epsi, dest):
        N, f_in, v, q = y
        if f_in < 0.:
            #raise Exception('Negative f_in (%f) at t=%f' %(f_in, ti))
            # HACK
            print 'Warning: Negative f_in (%f) at t=%f' % (f_in, ti)
            f_in = 1e-4

        dest[0] = epsi * s - (N / tau_s) - ((f_in - 1) / tau_f)  # dNdt
        dest[1] = N  # dfidt
        dest[2] = (1 / tau_m) * (f_in - v**(1 / alpha_w))  # dvdt
        dest[3] = (1 / tau_m) * ((f_in / E0) * (1 - (1 - E0)**(1 / f_in)) -
                                 (q / v) * (v**(1 / alpha_w)))  # dqdt
        return dest

    res = np.zeros((stim.size + 1, 4))
    res[0, :] = Y0 or np.array([0., 1., 1., 1.])

    for ti in xrange(1, stim.size + 1):
        cpt_phy_model_deriv(res[ti - 1], stim[ti - 1], epsilon, dest=res[ti])
        res[ti] *= tstep
        res[ti] += res[ti - 1]

    return res[1:, :].T


def create_evoked_physio_signals(physiological_params, paradigm,
                                 neural_efficacies, dt, integration_step=.05):
    """
    Generate evoked hemodynamics signals by integrating a physiological model.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
             parameters of the physiological model.
             In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII
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
    # TODO: handle multiple conditions
    # -> create input activity signal [0, 0, eff_c1, eff_c1, 0, 0, eff_c2, ...]
    # for now, take only first condition
    first_cond = paradigm.get_stimulus_names()[0]
    stim = paradigm.get_rastered(integration_step)[first_cond][0]
    neural_efficacies = neural_efficacies[0]

    # response matrix intialization
    integrated_vars = np.zeros((4, neural_efficacies.shape[0], stim.shape[0]))
    for i, epsilon in enumerate(neural_efficacies):
        integrated_vars[:, i, :] = phy_integrate_euler(physiological_params,
                                                       integration_step, stim,
                                                       epsilon)

    # downsampling:
    nb_scans = paradigm.get_rastered(dt)[first_cond][0].size
    dsf = int(dt / integration_step)
    return np.swapaxes(integrated_vars[:, :, ::dsf][:, :, :nb_scans], 1, 2)


def create_k_parameters(physiological_params):
    """ Create field strength dependent parameters k1, k2, k3
    """
    # physiological parameters
    V0 = physiological_params['V0']
    E0 = physiological_params['E0']
    TE = physiological_params['TE']
    r0 = physiological_params['r0']
    vt0 = physiological_params['vt0']
    e = physiological_params['e']
    if physiological_params['model'] == 'RBM':  # RBM
        k1 = 4.3 * vt0 * E0 * TE
        k2 = e * r0 * E0 * TE
        k3 = 1 - physiological_params['e']
    elif physiological_params['buxton']:
        k1 = 7 * E0
        k2 = 2
        k3 = 2 * E0 - 0.2
    else:   # CBM
        k1 = (1 - V0) * 4.3 * vt0 * E0 * TE
        k2 = 2. * E0
        k3 = 1 - physiological_params['e']
    physiological_params['k1'] = k1
    physiological_params['k2'] = k2
    physiological_params['k3'] = k3

    return k1, k2, k3


def create_bold_from_hbr_and_cbv(physiological_params, hbr, cbv):
    """
    Compute BOLD signal from HbR and blood volume variations obtained
    by a physiological model
    """

    # physiological parameters
    V0 = physiological_params['V0']
    k1, k2, k3 = create_k_parameters(physiological_params)

    # linear vs non-linear
    if physiological_params['linear']:
        sign = 1.
        if physiological_params['obata']:  # change of sign
            sign = -1
        bold = V0 * ((k1 + k2) * (1 - hbr) + sign * (k3 - k2) * (1 - cbv))
    else:  # non-linear
        bold = V0 * (k1 * (1 - hbr) + k2 * (1 - hbr / cbv) + k3 * (1 - cbv))

    return bold


def create_physio_brf(physiological_params, response_dt=.5,
                      response_duration=25., return_brf_q_v=False):
    """
    Generate a BOLD response function by integrating a physiological model and
    setting its driving input signal to a single impulse.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
            parameters of the physiological model.
            In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII...
        - response_dt (float): temporal resolution of the response, in second
        - response_duration (float): duration of the response, in second

    Return:
        - np.array(nb_time_coeffs, float)
          -> the BRF (normalized)
        - also return brf_not_normalized, q, v when return_prf_q_v=True
          (for error checking of v and q generation in calc_hrfs)
    """

    p = Paradigm({'c': [np.array([0.])]}, [response_duration],
                 {'c': [np.array([1.])]})
    n = np.array([[1.]])
    s, f, v, q = create_evoked_physio_signals(physiological_params, p, n,
                                              response_dt)
    brf = create_bold_from_hbr_and_cbv(physiological_params, q[:, 0], v[:, 0])
    if return_brf_q_v:
        # WARNING!! Added to compute figures
        return brf / (brf**2).sum()**.5, q, v, s, f
        # return  brf, q, v, s, f  #WARNING!! Added to compute figures
    else:
        return brf  # / (brf**2).sum()**.5


def create_physio_prf(physiological_params, response_dt=.5,
                      response_duration=25., return_prf_q_v=False):
    """
    Generate a perfusion response function by setting the input driving signal
    of the given physiological model with a single impulse.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
            parameters of the physiological model.
            In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII...
        - response_dt (float): temporal resolution of the response, in second
        - response_duration (float): duration of the response, in second

    Return:
        - np.array(nb_time_coeffs, float)
          -> the PRF
        - also return brf_not_normalized, q, v when return_prf_q_v=True
          (for error checking of v and q generation in calc_hrfs)
    """
    p = Paradigm({'c': [np.array([0.])]}, [response_duration],
                 {'c': [np.array([1.])]})  # response_dt to match convention
    # in JDE analysis
    n = np.array([[1.]])
    s, f, v, q = create_evoked_physio_signals(physiological_params, p, n,
                                              response_dt)
    prf = f[:, 0] - f[0, 0]  # remove y-intercept
    if return_prf_q_v:
        return prf / (prf**2).sum()**.5  # , q, v
        # return prf, q, v
    else:
        return prf  # / (prf**2).sum()**.5


def create_omega_prf(primary_brf, dt, phy_params):
    """ create prf from omega and brf
    """
    omega = linear_rf_operator(primary_brf.shape[0], phy_params, dt)
    prf = np.dot(omega, primary_brf)
    prf = prf / (prf**2).sum()**.5
    return prf


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

    tau_m_inv = 1. / phy_params['tau_m']
    alpha_w = phy_params['alpha_w']
    alpha_w_inv = 1. / phy_params['alpha_w']
    E0 = phy_params['E0']
    V0 = phy_params['V0']
    k1, k2, k3 = create_k_parameters(phy_params)
    c = tau_m_inv * (1 + (1 - E0) * np.log(1 - E0) / E0)

    from pyhrf.sandbox.physio2 import buildOrder1FiniteDiffMatrix_central
    D = buildOrder1FiniteDiffMatrix_central(rf_size, dt)  # numpy matrix
    eye = np.matrix(np.eye(rf_size))  # numpy matrix

    A3 = tau_m_inv * ((D + (alpha_w_inv * tau_m_inv) * eye).I)
    A4 = c * (D + tau_m_inv * eye).I - (D + tau_m_inv * eye).I * ((1 - alpha_w)
          * alpha_w_inv * tau_m_inv**2) * (D + alpha_w_inv * tau_m_inv * eye).I
    # A = V0 * ( (k1+k2)*A4 + (k3-k2)* A3 ) # A = h x2^{-1} = Omega^{-1}

    # linear vs non-linear
    if phy_params['linear']:
        sign = 1.
        if phy_params['obata']:  # change of sign
            sign = -1
        # A = h x2^{-1} = Omega^{-1}
        A = V0 * ((k1 + k2) * A4 + sign * (k3 - k2) * A3)
    else:  # non-linear
        A = V0 * \
            (k1 * A4 + k2 *
             (eye - (eye - A4) * np.linalg.inv(eye - A3)) + k3 * A3)

    if (calculating_brf):
        return -A.A
    else:  # calculating_prf
        return -(A.I).A


def calc_linear_rfs(simu_brf, simu_prf, phy_params, dt, normalized_rfs=True):
    """
    Calculate 'prf given brf' and 'brf given prf' based on the a linearization
    around steady state of the physiological model as described in Friston 2000

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
        calc_brf, calc_prf have a truncation error (due to the finite difference        
        matrix used) on the order of O(dt)^2. If for any reason a hack is later         
        implemented to set the y-intecepts of brf_calc, prf_calc to zero by
        setting the first row of X4, X3 = 0, this will raise a singular matrix
        error in the calculation of calc_prf (due to X.I command), so this error        
        is helpful in this case
    """

    D = buildOrder1FiniteDiffMatrix_central(simu_prf.size, dt)  # numpy matrix
    I = np.matrix(np.eye(simu_prf.size))  # numpy matrix
    # TODO: elimlinate prf.size dependency

    tau_m = phy_params['tau_m']
    # when tau_m=1, singular matrix formed by (D+tau_m_inv*I)
    tau_m_inv = 1. / tau_m
    alpha_w = phy_params['alpha_w']
    alpha_w_inv = 1. / phy_params['alpha_w']
    E0 = phy_params['E0']
    V0 = phy_params['V0']
    k1, k2, k3 = create_k_parameters(phy_params)
    c = tau_m_inv * (1 + (1 - E0) * np.log(1 - E0) / E0)

    # transform to (hrf.size,1) matrix for calcs
    simu_prf = np.matrix(simu_prf).transpose()
    simu_brf = np.matrix(simu_brf).transpose()

    X3 = tau_m_inv * ((D + (alpha_w_inv * tau_m_inv) * I).I)

    X4 = c * (D + tau_m_inv * I).I - (D + tau_m_inv * I).I * ((1 - alpha_w) 
          * alpha_w_inv * tau_m_inv**2) * (D + alpha_w_inv * tau_m_inv * I).I

    # linear vs non-linear
    if phy_params['linear']:
        sign = 1.
        if phy_params['obata']:  # change of sign
            sign = -1
        # A = h x2^{-1} = Omega^{-1}
        X = V0 * ((k1 + k2) * X4 + sign * (k3 - k2) * X3)
    else:  # non-linear
        # print X4
        # print X3
        X = V0 * (k1 * X4 + k2 * (X4 - X3) * np.linalg.inv(I - X3) + k3 * X3)

    # for error checking
    q_linear = 1 - X4 * (-simu_prf)
    v_linear = 1 - X3 * (-simu_prf)

    calc_brf = X * (-simu_prf)

    calc_prf = -X.I * simu_brf

    # convert to np.arrays
    calc_prf = calc_prf.A
    calc_brf = calc_brf.A
    q_linear = q_linear.A
    v_linear = v_linear.A

    if normalized_rfs:
        calc_prf /= (calc_prf**2).sum()**.5
        calc_brf /= (calc_brf**2).sum()**.5

    return calc_brf, calc_prf, q_linear, v_linear
