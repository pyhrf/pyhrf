# -*- coding: utf-8 -*-

import unittest
import pyhrf
import numpy as np
import os.path as op
import shutil

from pyhrf.ndarray import MRI3Daxes
import pyhrf.boldsynth.scenarios as simu
from pyhrf.rfir import rfir

class RFIRTest(unittest.TestCase):
    """
    Test the Regularized FIR (RFIR)-based methods implemented in pyhrf.rfir
    """

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_rfir_on_small_simulation(self):
        """ Check if pyhrf.rfir runs properly and that returned outputs
        contains the expected items """
        fdata = simu.create_small_bold_simulation()
        outputs = rfir(fdata, nb_its_max=2)

        assert isinstance(outputs, dict)
        for k in ["fir", "fir_error", "fit", "drift"]:
            assert outputs.has_key(k)


    def test_results_small_simulation(self):
        """

        TODO: move to validation
        """
        #output_dir = pyhrf.get_tmp_path()
        output_dir = './'

        fdata = simu.create_small_bold_simulation(output_dir=output_dir)

        pyhrf.verbose.set_verbosity(0)
        outputs = rfir(fdata, nb_its_max=500, nb_its_min=100)

        m = fdata.roiMask
        for k,o in outputs.iteritems():
            output_fn = op.join(output_dir, k + '.nii')
            o.expand(m, "voxel", target_axes=MRI3Daxes).save(output_fn)
