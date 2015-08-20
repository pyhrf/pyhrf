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

If you plan to also install pyhrf_viewer (recommended), run:

.. code:: bash

    $ sudo apt-get install python-qt4

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
