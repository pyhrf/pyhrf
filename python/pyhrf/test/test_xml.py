



import unittest
import pyhrf
import os
import numpy as np
import os.path as op
import shutil


class A(XMLable):
    
    def __init__(self, p=1):
        XMLable.__init__(self)
        self.param = p


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path() #'./'

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)


    def test_simple_bijection():
        
        a = A()
        a_xml = toXML(a)
        self.assertEqual(a, fromXML(a))

    

