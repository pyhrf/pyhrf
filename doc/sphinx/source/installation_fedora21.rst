.. _installation_fedora21:


============================
 Installation for Fedora 21
============================

These instructions are specific to Fedora 21.

Required dependencies
#####################

First, be sure that your system is up-to-date:

.. code:: bash

    $ sudo yum update

Then install the following packages:

.. code:: bash

    $ sudo yum install python-devel numpy scipy python-matplotlib python-pip sympy gcc

These dependencies are not available as system packages, they have to be installed
by hand ::

    $ pip install --user nibabel
    $ pip install --user nipy

Optional dependencies
#####################

Install the following packages:

.. code:: bash

    $ sudo yum install graphviz-python python-sphinx python-scikit-learn python-pillow python-joblib python-paramiko
    $ pip install --user munkres

If you plan to also install pyhrf_viewer (recommended), run:

.. code:: bash

    $ sudo yum install PyQt4 python-matplotlib-qt4 

PyHRF Installation
##################

Install package from `PYPI <https://pypi.python.org/pypi/pyhrf>`_
It is recommended to install the package in user mode (the ``--user`` option).

.. code:: bash

    $ pip install --user pyhrf

If you install in user mode, you need to add ``$HOME/.local/bin`` to your ``$PATH`` environment variable by adding the following to your ``$HOME/.profile`` (or ``$HOME/.bashrc``) file:

.. code:: bash

    if [ -d "$HOME/.local/bin" ]; then
        PATH="$HOME/.local/bin:$PATH"
    fi

You will need to logout/login to be able to use the correct ``$PATH`` environment (unless you added it in your ``.bashrc`` file in which case you must open a new terminal)

Then you can run the unit tests:

.. code:: bash

     $ pyhrf_maketests
