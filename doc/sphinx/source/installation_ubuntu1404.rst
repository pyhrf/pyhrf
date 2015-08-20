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

    $ sudo apt-get install python-scikits-learn python-joblib python-sphinx python-pygraph python-pygraphviz python-PIL python-munkres python-paramiko

If you plan to use our specific viewer (pyhrf_viewer), run:

.. code:: bash

    $ sudo apt-get install python-qt4

.. include:: pyhrf_installation.rst
