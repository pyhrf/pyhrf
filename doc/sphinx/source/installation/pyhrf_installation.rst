.. _pyhrf_installation:

PyHRF Installation
##################

Install package from `PyPI <https://pypi.python.org/pypi/pyhrf>`_.
It is recommended to install the package in user mode (the ``--user`` option).

.. code:: bash

    $ pip install --user pyhrf

If you install in user mode, you need to concatenate ``$HOME/.local/bin`` to
your ``$PATH`` environment variable by adding the following to your
``$HOME/.bashrc`` file:

.. code:: bash

    if [ -d "$HOME/.local/bin" ]; then
        PATH="$HOME/.local/bin:$PATH"
    fi

Update your ``$PATH`` environment variable by sourcing your ``.bashrc`` file:

.. code:: bash

    $ source ~/.bashrc

Then you can run the unit tests:

.. code:: bash

     $ pyhrf_maketests

PyHRF update
############

To update the package, use pip:

.. code:: bash

    $ pip install --upgrade --user pyhrf


PyHRF uninstallation
####################

To remove pyhrf package, use pip:

.. code:: bash

    $ pip uninstall pyhrf
