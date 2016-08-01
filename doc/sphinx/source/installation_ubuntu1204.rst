.. _installation_ubuntu1204:


===============================
 Installation for Ubuntu 12.04
===============================

These instructions are specific to Ubuntu 12.04.

Required dependencies
#####################

First, be sure that your system is up-to-date:

.. code:: bash

    $ sudo apt-get update
    $ sudo apt-get upgrade

Then install the following packages:

.. code:: bash

    $ sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-pip python-sympy gcc

The version of `nibabel` and `nipy` packages available in the Ubuntu 12.04 repositories are too old for PyHRF, install newer versions with:

.. code:: bash

    $ pip install --user nibabel
    $ pip install --user nipy

Optional dependencies
#####################

Install the following packages:

.. code:: bash

    $ sudo apt-get install python-scikits-learn python-joblib python-pygraph python-pygraphviz python-PIL python-munkres python-paramiko

This dependency is too old on the packages system manager::

    $ pip install --user sphinx

if you already installed sphinx with the packages manager, add the flag ``--upgrade`` to the previous command


If you plan to use our specific viewer (pyhrf_viewer), run:

.. code:: bash

    $ sudo apt-get install python-qt4

.. include:: pyhrf_installation.rst
