# Launch all unit tests and doc test in this script with code coverage:
# nosetests indev.py --with-coverage --cover-html --with-doctest --cover-package indev --cover-branches --cover-erase -v

# Launch only unittest:
# python indev.py

# Generate automatic code doc with sphinx (in folder pyhrf/doc/sphinx):
# sphinx-apidoc -o ./source/autodoc/ ../../python/ ../../python/ ../../python/pyhrf.egg-info/ -f

"""
"""
import os
import unittest
import os.path as op
import numpy as np
import numpy.testing as npt

#import matplotlib.pyplot as plt
import shutil

import pyhrf

rx_py_identifier = '[^\d\W]\w*'

class Class:
    """
    #TODO: comment
    """
    
def func(self):
    """
    #TODO: comment
    #TODO: test
    #TODO: implement
    """


class Test(unittest.TestCase):

    def setUp(self,):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def _create_tmp_files(self, fns):
        for fn in [op.join(self.tmp_dir,fn) for fn in fns]:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
            open(fn, 'a').close()

    def assert_file_exists(self, fn):
        if not op.exists(fn):
            raise Exception('File %s does not exist' %fn)


if __name__ == '__main__':
    unittest.main()       
