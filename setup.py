#!/usr/bin/env python2
"""Setup script for PyHRF"""
from __future__ import print_function
from setuptools import setup
import sys


# The current version of PyHRF works with python 2.7.X
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2]:
    raise RuntimeError("Python version 2.7 required.")


def parse_setuppy_commands():
    """Check the commands and respond appropriately. Disable broken commands.

    Returns:
        bool: Return a boolean value for whether or not to run the build (avoid installing numpy and other dependencies)
    """
    if len(sys.argv) < 2:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    good_commands = ('develop', 'install', 'sdist', 'build', 'build_ext',
                     'build_py', 'build_clib', 'build_scripts', 'bdist_wheel',
                     'bdist_rpm', 'bdist_wininst', 'bdist_msi', 'bdist_mpkg',
                     'build_sphinx')

    for command in good_commands:
        if command in sys.argv[1:]:
            return False

    return True


def setup_package():
    """Configuration of the setup"""

    if parse_setuppy_commands():
        extra_setuptools_args = dict()
    else:
        from setuptools import find_packages, Extension
        from glob import glob
        try: # for pip >= 10
            from pip._internal.req import parse_requirements
        except ImportError: # for pip <= 9.0.3
            from pip.req import parse_requirements

        # Dependencies for building C Extensions
        try:
            dependencies = list(parse_requirements('requirements.txt'))
        except TypeError:
            # new versions of pip requires a session
            dependencies = list(parse_requirements('requirements.txt', session=pip.download.PipSession()))

        dependencies = [str(package.req) for package in dependencies]

        # Installing the required packages to build C extensions
        for package in dependencies:
            pip.main(['install', package])

        import numpy

        c_extensions = [Extension(ext_name, sources=['src/pyhrf/' + filepath])
                        for (ext_name, filepath) in
                        [('pyhrf.jde.intensivecalc', 'jde/intensivecalc.c'),
                         ('pyhrf.boldsynth.pottsfield.pottsfield_c', 'boldsynth/pottsfield/pottsField.c'),
                         ('pyhrf.vbjde.UtilsC', 'vbjde/utilsmodule.c'),
                         ('pyhrf.cparcellation', 'cparcellation.c')]]

        extra_setuptools_args = dict(
            package_dir={'': 'python'},
            packages=find_packages("python"),
            include_package_data=True,
            scripts=glob('./bin/*'),
            zip_safe=False,  # pyhrf has C/C++ extensions, so it's not zip safe.
            ext_modules=c_extensions,
            include_dirs=[numpy.get_include()],
            setup_requires=dependencies,
            install_requires=dependencies,
            extras_require={"cluster": ["soma-workflow"],
                            "simulation": ["Pillow>=2.3"],
                            "parcellation": ["munkres>=1.0"],
                            "pipelines": ["pygraphviz>=1.1"],
                            "graph": ["python-graph-core>=1.8"]})

    # Get the long description from the README file
    with open('README.rst') as readme_file:
        long_description = readme_file.read()

    metadata = dict(
        name='pyhrf',
        version='0.5.0',
        description='Analysis of fMRI data based on the study of hemodynamics',
        long_description=long_description,
        maintainer='Jaime Arias',
        maintainer_email='jaime.arias@inria.fr',
        url='http://pyhrf.org',
        license='CeCILLv2',
        download_url='https://github.com/pyhrf/pyhrf',
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: C",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
        ],
        **extra_setuptools_args
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()

    if 'install' in sys.argv[1]:
        import colorama

        colorama.init(autoreset=True)

        def red(text):
            """Return a red terminal text"""
            return "{}{}".format(colorama.Fore.RED, text)


        def green(text):
            """Return a green terminal text"""
            return "{}{}".format(colorama.Fore.GREEN, text)


        def yellow(text):
            """Return a yellow terminal text"""
            return "{}{}".format(colorama.Fore.YELLOW, text)


        def check_opt_dep(dep_name, dep_descrip):
            """
            Return a message telling if dependency *dep_name* is available
            with an import
            """
            try:
                __import__(dep_name)
            except ImportError:
                return red(
                    "{} is *NOT IMPORTABLE*, {} will *NOT* be available".format(
                        dep_name,
                        dep_descrip))

            return green(
                "{} is importable, {} will be available".format(dep_name,
                                                                dep_descrip))


        # Optional deps and description of associated feature:
        OPTIONAL_DEPS = {
            "soma_workflow": 'cluster parallel feature (eg pyhrf_jde_estim -x cluster)',
            'PIL': "loading of image file as simulation maps",
            'munkres': 'computation of distance between parcellations',
            'pygraph': '(python-graph-core) -- save plot of simulation pipelines',
            'pygraphviz': 'optimized graph operations and outputs',
        }

        print(yellow('\nOptional dependencies:'))
        print('\n'.join(
            ['- ' + check_opt_dep(dn, dd) for dn, dd in OPTIONAL_DEPS.items()]))

        print(yellow('\nExecute pyhrf_maketests to run package tests.\n'))
