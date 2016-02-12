# -*- coding: utf-8 -*-
import os
import os.path as op
import unittest
import tempfile
import shutil
import numpy as np

import pyhrf
from pyhrf.core import FmriData, merge_fmri_sessions


class FMRIDataTest(unittest.TestCase):

    def test_from_vol_ui_default(self):
        fmri_data = FmriData.from_vol_ui()

    def test_multisession_simu(self):
        fd1 = FmriData.from_simu_ui()
        fd2 = FmriData.from_simu_ui()

        fd_msession = merge_fmri_sessions([fd1, fd2])
        self.assertEqual(fd_msession.nbSessions, 2)
