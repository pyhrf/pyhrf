

import sys
import StringIO
import copy as copyModule
import traceback

import pyhrf
from pyhrf import xmlio

from pyhrf.rfir import RFIREstim
from pyhrf.ui.analyser_ui import FMRIAnalyser

DEFAULT_CFG_FILE = './rfir.xml'

class RFIRAnalyser(FMRIAnalyser):

    parametersToShow = ['HrfEstimator']

    def __init__(self, HrfEstimator=RFIREstim(), outputPrefix='hrf_'):
        xmlio.XmlInitable.__init__(self)
        FMRIAnalyser.__init__(self, outputPrefix='rfir_')

        self.hEstimator = HrfEstimator


    def analyse_roi(self, atomData):
        hEstimator = copyModule.deepcopy(self.hEstimator)
        hEstimator.linkToData(atomData)
        hEstimator.run()
        return hEstimator


