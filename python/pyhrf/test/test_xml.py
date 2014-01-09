import unittest
import pyhrf
import os
import numpy as np
import os.path as op
import shutil

from pyhrf.xmlio import XmlInitable, to_xml, from_xml

class A(XmlInitable):

    def __init__(self, p=1, c='a'):
        XmlInitable.__init__(self)
        self.param = p

class TestXML(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path() #'./'

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def test_simple_bijection(self):
        a = A(c='e')
        a_xml = to_xml(a)
        self.assertEqual(a, from_xml(a_xml))


