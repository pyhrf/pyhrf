# -*- coding: utf-8 -*-


# Let setuptools manage package version
# The "base" version is uniquely defined in setup.py 
import pkg_resources
__version__ = pkg_resources.require("pyhrf")[0].version
