#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import sys

from glob import glob
from importlib import import_module

install_requires = [("numpy", "1.6"),
                    ("scipy", "0.9"),
                    ("nibabel", "1.1"),
                    ("sympy", "0.7"),
                    ("nipy", "0.3.0")]
missing_package = []
for package in install_requires:
    try:
        import_module(package[0])
    except ImportError:
        missing_package.append(package)

if missing_package != []:
    print "Package(s) {0} must be installed prior PyHRF installation.".\
            format(", ".join([">=".join(package) for package in missing_package]))
    sys.exit(1)

try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()

from setuptools import setup, find_packages, Extension

import numpy as np


cExtensions = [
    Extension('pyhrf.jde.intensivecalc',
              ['src/pyhrf/jde/intensivecalc.c'],
              [np.get_include()]),
    Extension('pyhrf.boldsynth.pottsfield.pottsfield_c',
              ['src/pyhrf/boldsynth/pottsfield/pottsField.c'],
              [np.get_include()]),
    Extension('pyhrf.vbjde.UtilsC',
              ['src/pyhrf/vbjde/utilsmodule.c'],
              [np.get_include()]),
    Extension('pyhrf.cparcellation',
              ['src/pyhrf/cparcellation.c'],
              [np.get_include()]),
    ]

setup(
    name = "pyhrf",
    version = "0.4.2",
    description = ("PyHRF is a set of tools to analyze fMRI data and "
                   "specifically study hemodynamics."),
    long_description = open("README.rst").read(),
    author = ("Thomas VINCENT, Philippe CIUCIU, Solveig BADILLO, Florence "
              "FORBES, Aina FRAU, Thomas PERRET"),
    author_email = "thomas.tv.vincent@gmail.com",
    maintainer = "Thomas PERRET",
    maintainer_email = "thomas.perret@inria.fr",
    url = "http://pyhrf.org",
    packages = find_packages("python"),
    setup_requires = ["numpy>=1.0",
                      "scipy>=0.9",
                      "nibabel>=1.1",
                      "sympy>=0.7"],
    include_package_data = True,
    scripts = glob('./bin/*'),
    install_requires = ["numpy>=1.6",
                        "scipy>=0.9",
                        "matplotlib>=1.1",
                        "nibabel>=1.1",
                        "sympy>=0.7",
                        "nipy>=0.3.0"],
    extras_require = {"Ward": ["scikit-learn>=0.10"],
                      "parallel": ["joblib>=0.5"],
                      "cluster": ["soma-workflow"],
                      "simulation": ["Pillow>=2.3"],
                      "parcellation": ["munkres>=1.0"],
                      "pipelines": ["pygraphviz>=1.1"],
                      "graph": ["python-graph-core>=1.8"]},
    package_dir = {'' : 'python'},
    # include_dirs = [np.get_include()],
    ext_modules = cExtensions,
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
    license = "CeCILLv2",
    platforms = ["linux"],
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
