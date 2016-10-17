#! /usr/bin/env python2

import sys

# The current version of pyhrf works with python 2.7
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2]:
    raise RuntimeError("Python version 2.7 required.")

# setuptools is used to build and distribute pyhrf
from  ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, Extension

from glob import glob
from importlib import import_module

try:
    import numpy
except ImportError:
   raise ImportError("PyHRF requires numpy")

try:
   import scipy
except ImportError:
   raise ImportError("PyHRF requires scipy")

# Get the long description from the README file
with open('README.rst') as f:
    long_description = f.read()

# Dependencies
build_requires = ["numpy>=1.6",
                  "scipy>=0.9",
                  "nibabel>=1.1, <2.1.0",
                  "sympy>=0.7",
                  "nipy>=0.3.0"]

# Including C code
cExtensions = [Extension(ext_name,
                         sources=['src/pyhrf/'+filepath],
                         include_dirs=[numpy.get_include()])
               for (ext_name,filepath) in [('pyhrf.jde.intensivecalc','jde/intensivecalc.c'),
                                           ('pyhrf.boldsynth.pottsfield.pottsfield_c', 'boldsynth/pottsfield/pottsField.c'),
                                           ('pyhrf.vbjde.UtilsC', 'vbjde/utilsmodule.c'),
                                           ('pyhrf.cparcellation','cparcellation.c')]]

setup(
    name = 'pyhrf',
    version = '0.4.3',
    description = 'Set of tools to analyze fMRI data focused on the study of hemodynamics',
    long_description = long_description,
    author = ("Thomas VINCENT, Philippe CIUCIU, Solveig BADILLO, Florence "
              "FORBES, Aina FRAU, Thomas PERRET"),
    author_email = "thomas.tv.vincent@gmail.com",
    maintainer = 'Jaime Arias',
    maintainer_email = 'jaime.arias@inria.fr',
    url = 'http://pyhrf.org',
    license = 'CeCILLv2',
    download_url = 'https://github.com/pyhrf/pyhrf',
    package_dir = {'' : 'python'},
    packages = find_packages("python"),
    include_package_data = True,
    scripts = glob('./bin/*'),
    ext_modules = cExtensions,
    setup_requires = build_requires,
    install_requires = build_requires,
    extras_require = {"Ward": ["scikit-learn>=0.10"],
                      "parallel": ["joblib>=0.5"],
                      "cluster": ["soma-workflow"],
                      "simulation": ["Pillow>=2.3"],
                      "parcellation": ["munkres>=1.0"],
                      "pipelines": ["pygraphviz>=1.1"],
                      "graph": ["python-graph-core>=1.8"]},
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2 :: Only",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    platforms = ["Linux"],
    zip_safe = False,
    )

if sys.argv[1] == "install":
    # optional deps and description of associated feature:
    optional_deps = {
        "sklearn": "(scikit-learn) -- spatial ward parcellation",
        "joblib": "local parallel feature (eg pyhrf_jde_estim -x local)",
        "soma_workflow": "cluster parallel feature (eg pyhrf_jde_estim -x cluster)",
        "PIL": "loading of image file as simulation maps",
        "munkres": "computation of distance between parcellations",
        "pygraph": "(python-graph-core) -- save plot of simulation pipelines",
        "pygraphviz": "optimized graph operations and outputs",
        }

    def check_opt_dep(dep_name, dep_descrip):
        """
        Return a message telling if dependency *dep_name* is available
        with an import
        """
        try:
            __import__(dep_name)
        except ImportError:
            return "%s *NOT IMPORTABLE*, %s will *NOT* be available" %(dep_name,
                                                                       dep_descrip)
        return "%s is importable, %s will be available" %(dep_name, dep_descrip)

    print "Optional dependencies:"
    print "\n".join(["- "+ check_opt_dep(dn, dd) for dn, dd in optional_deps.items()])


    print ("\nIf the installation was successfull, you may run "
           '"pyhrf_maketests" to run package tests.\n')
