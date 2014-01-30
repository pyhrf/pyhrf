import unittest
import numpy as np

import pyhrf

if 1:
    from core_test import *
    from boldsynthTest import *
    from toolsTest import *
    from commandTest import *
    from test_xml import *
    from statsTest import *
    from graphtest import *
    from analysertest import *
    from jdetest import *
    from iotest import *
    from test_parcellation import *

if 1:
    from test_plot import *
    from test_treatment import *
    from test_glm import *
    from test_parallel import *
    from test_jde_multi_subj import *
    from test_paradigm import *
    from test_rfir import *
    from test_xml import *

from test_ndarray import *

if pyhrf.__usemode__ == pyhrf.DEVEL:
    from test_sandbox import *
    from test_sandbox_parcellation import *
    from test_sandbox_physio import *

if __name__ == "__main__":
    unittest.main()
