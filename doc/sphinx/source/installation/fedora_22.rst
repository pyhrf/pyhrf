.. _installation_fedora22:


==========================
Installation for Fedora 22
==========================

These instructions are specific to **Fedora 22**.

Required dependencies
#####################

First, be sure that your system is up-to-date:

.. code:: bash

    $ sudo yum update

Then install the following packages:

.. code:: bash

    $ sudo yum install python-devel numpy scipy python-matplotlib python-pip sympy gcc

These dependencies are not available as system packages, they have to be installed
by hand:

.. code:: bash

    $ pip install --user nibabel
    $ pip install --user nipy

Optional dependencies
#####################

Install the following packages:

.. code:: bash

    $ sudo yum install graphviz-python python-sphinx python-scikit-learn python-pillow python-joblib python-paramiko
    $ pip install --user munkres

These dependencies are too old on the packages system manager:

.. code:: bash

    $ pip install --user --upgrade sphinx

If you plan to use our specific viewer (pyhrf_viewer), run:

.. code:: bash

    $ sudo yum install PyQt4 python-matplotlib-qt4

.. include:: pyhrf_installation.rst
