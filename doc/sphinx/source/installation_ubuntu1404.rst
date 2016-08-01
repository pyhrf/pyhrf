.. _installation_ubuntu1404:


===============================
 Installation for Ubuntu 14.04
===============================

These instructions are specific to Ubuntu 14.04.

Required dependencies
#####################

First, be sure that your system is up-to-date:

.. code:: bash

    $ sudo apt-get update
    $ sudo apt-get upgrade

Then install the following packages:

.. code:: bash

    $ sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-nibabel python-nipy gcc

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
