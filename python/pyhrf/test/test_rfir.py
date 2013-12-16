import unittest
import pyhrf
import shutil

from pyhrf import tools
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


    @unittest.skipIf(not tools.is_importable('scipy.misc','fromimage'),
                     'scipy.misc.fromimage (optional dep) is N/A')        
    def test_rfir_on_small_simulation(self):
        """ Check if pyhrf.rfir runs properly and that returned outputs
        contains the expected items """
        fdata = simu.create_small_bold_simulation()
        outputs = rfir(fdata, nb_its_max=2)

        assert isinstance(outputs, dict)
        for k in ["fir", "fir_error", "fit", "drift"]:
            assert outputs.has_key(k)

        #TODO: test shape consistency
