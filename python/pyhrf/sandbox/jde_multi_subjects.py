import numpy as np
from pyhrf.sandbox.stats import GibbsSampler, GSVariable

class HrfSubjectSampler(GSVariable):

    def __init__(self, duration, dt, zero_constraint):
        """
        Create a variable object handling the sampling of variable 'x'.
        The sampling init value is set to 0.
        """

        GSVariable.__init__(self, 'hrf_subjects', initialization='custom')

        self.duration = duration
        self.dt = dt
        self.zc = zero_constraint
        self.nb_coeffs = self.duration / self.dt

    def init_sampling(self):

        paradigm = self.get_variable_value('paradigm')
        ny = self.get_variable_value('y').shape[1]# shape(y)=(subj,scan,vox)
        self.X = build_paradigm_matX(paradigm, ny, self.coeffs)

        (useless, self.varR) = genGaussianSmoothHRF(self.zc, self.nb_coeffs,
                                                    self.eventdt, 1.)

    def sample(self):
        """
        """

        # retrieve other quantities:
        y = self.get_variable_value('y')
        v_hs = self.get_variable_value('hrf_subj_var')
        s2 = self.get_variable_value('noise_var')
        h_group = self.get_variable_value('hrf_group')

        snrl = self.get_variable('nrl')


        # Do the sampling:
        for s in xrange(self.nbSubj):
            (self.varDeltaS, self.varDeltaY) = self.computeStDS_StDY_one_subject(rb, nrls, snrl.aa, s)
            h[s] = sampleHRF_single_hrf(self.varDeltaS, self.varDeltaY,
                                        self.R, s2[s], self.nb_col_X,
                                        y.shape[2], h_group)
        return np.random.randn() * self.post_var + self.post_mean


class JdeMultiSubjectSampler(GibbsSampler):

    def __init__(self, ):
        sampled_variables = [HrfSubjectSampler()]
        GibbsSampler.__init__(self, sampled_variables, nb_its_max=10,
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
