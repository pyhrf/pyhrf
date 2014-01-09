#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from glob import glob

try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()

from setuptools import setup, find_packages, Extension


#######################
#  Pre-install checks #
#######################

def dependCheck(checkers):
    dflags = dict(zip( checkers.keys(), [True]*len(checkers) ))
    for depName, checker in checkers.iteritems():
        print 'Checking %s ... ' %depName,
        if checker():print 'ok'
        else:
            print 'NOT OK!'
            dflags[depName] = False
    return dflags


def checkPyXML():
    try:
        import xml
    except ImportError:
        return False
    return True


def checkPyhrf():
    try:
        import pyhrf
    except ImportError,e :
        print e
        return False
    return True

dependCheckers = {
    'pyxml': checkPyXML,
    }


dependFlags = dependCheck(dependCheckers)


cExtensions = [
    Extension('pyhrf.jde.intensivecalc',
              ['src/pyhrf/jde/intensivecalc.c']),
    Extension('pyhrf.boldsynth.pottsfield.pottsfield_c',
              ['src/pyhrf/boldsynth/pottsfield/pottsField.c']),
    Extension('pyhrf.vbjde.UtilsC',
              ['src/pyhrf/vbjde/utilsmodule.c']),
    Extension('pyhrf.cparcellation',
              ['src/pyhrf/cparcellation.c']),
## used to sample the GIG (not maintained)
##     Extension('pyhrf.stats.cRandom',
##               ['src/pyhrf/stats/cRandom.c'],
##               libraries=['unuran', 'prng'],
##               ),
    ]

try:
    import numpy as np
except ImportError:
    print 'Numpy should be installed prior to pyhrf installation'
    sys.exit(1)

setup(
    name="pyhrf", author='Thomas VINCENT, Philippe CIUCIU, Solveig BADILLO',
    author_email='thomas.tv.vincent@gmail.com',
    version='0.3',
    setup_requires=['numpy>=1.0'],
    install_requires=['numpy>=1.0','matplotlib>=0.90.1','scipy>=0.7',
                      'nibabel', 'nipy'], #, 'PyXML>=0.8.4'],
    dependency_links = [],
    package_dir = {'' : 'python'},
    packages=find_packages('python'),
    include_package_data=True,
    include_dirs = [np.get_include()],
    package_data={'pyhrf':['datafiles/*']},
    ext_modules=cExtensions,
    scripts=glob('./bin/*'),
    platforms=['linux'],
    zip_safe=False,
    )


# optional deps and description of associated feature:
optional_deps = {
    'sklearn' : '(scikit-learn) -- spatial ward parcellation',
    'joblib' : 'local parallel feature (eg pyhrf_jde_estim -x local)',
    'soma.workflow' : 'cluster parallel feature (eg pyhrf_jde_estim -x cluster)',
    'PIL' : 'loading of image file as simulation maps',
    'munkres' : 'computation of distance between parcellations',
    'pygraphviz' : '(python-graph-core) -- save plot of simulation pipelines',
    'PyQt4': 'viewer and xml editor',
    }

def check_opt_dep(dep_name, dep_descrip):
    """
    Return a message telling if dependency *dep_name* is available
    with an import
    """
    try:
        __import__(dep_name)
    except ImportError:
        return '%s *NOT IMPORTABLE*, %s will *NOT* be available' %(dep_name,
                                                               dep_descrip)
    return '%s is importable, %s will be available' %(dep_name, dep_descrip)

print 'Optional dependencies:'
print '\n'.join(['- '+ check_opt_dep(dn, dd) for dn, dd in optional_deps.items()])


print '\nIf the installation was successfull, you may run '\
    '"pyhrf_maketests" to run package tests.\n'
print 'Report on installation:'

installCheckers = {
    'Pyhrf main installation' : checkPyhrf,
    }

dependCheck(installCheckers)
