.. _installation_venv:


=====================================
 Installation in virtual environment
=====================================

Required dependencies
#####################

A ``pyhrf`` virtual environment. The procedure to install and configure a virtual environement are not described here. We recommend to use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/en/latest/>`_.

You also need to install developement header of the following libraries (refer to your distribution package manager):
    - libopenblas
    - liblapack
    - libpng
    - libfreetype

Once your virtualenv is ready, activate it and run:

.. code:: bash

    (pyhrf) $ pip install numpy
    (pyhrf) $ pip install scipy
    (pyhrf) $ pip install nibabel
    (pyhrf) $ pip install sympy
    (pyhrf) $ pip install nipy

It is not trivial (and not described here) to install PyQt4 in an virtualenv so no viewer will be available in virtualenv.
Optional dependencies:

.. code:: bash

    (pyhrf) $ pip install joblib
    (pyhrf) $ pip install scikit-learn
    (pyhrf) $ pip install sphinx
    (pyhrf) $ pip install pygraphviz
    (pyhrf) $ pip install Pillow

Install pyhrf itself:

.. code:: bash

    (pyhrf) $ pip install pyhrf

If you want to install the last development version, replace the previous command by:

.. code:: bash

    $ pip install [-e] git+https://github.com/pyhrf/pyhrf#egg=pyhrf

The ``-e`` optional flag installs pyhrf as develop mode.
