.. _installation:

============
Installation
============

Instructions are given for **Linux** and **Python 2.7**. The package is
mainly developed and tested under Ubuntu 14.04, we then recommend this
distribution. We officially support Ubuntu 12.04, Ubuntu 14.04,
Debian stable (jessie), Fedora 21 and 22, and Arch Linux. See
`below <#linux-based>`_ for instructions for each distribution.

For **windows users**, a linux distribution can be installed easily inside a
virtual machine such as Virtual Box. This wiki page explains
`how to install ubuntu from scratch in Virtual Box <http://www.wikihow.com/Install-Ubuntu-on-VirtualBox>`_
or you can get some `free VirtualBox images <http://virtualboxes.org/images/ubuntu/>`_.

Requirements / Dependencies
###########################

Dependencies are:

- `python <http://www.python.org>`_ 2.7 **with development headers**
- `numpy <http://docs.scipy.org/doc/numpy/user/install.html>`_ >= 1.0
- `scipy <http://www.scipy.org/install.html>`_ >= 0.7
- `matplotlib <http://matplotlib.org/users/installing.html>`_
- `nibabel <http://nipy.sourceforge.net/nibabel/>`_
- `sympy <http://sympy.sourceforge.net>`_ (required by nipy)
- `nipy <http://nipy.sourceforge.net/nipy/stable/users/installation.html>`_
- a C compiler

Optional dependencies:

- `PyQt4 <https://www.riverbankcomputing.com/software/pyqt/download>`_ (viewer and XML editor)
- `joblib <https://pythonhosted.org/joblib>`_ (local distributed computing)
- `paramiko <http://www.paramiko.org>`_ (local network distributed computing)
- `soma-workflow <https://github.com/neurospin/soma-workflow>`_ (remote distributed computing)
- `scikit-learn <http://scikit-learn.org/>`_ (clustering)
- `sphinx <http://www.sphinx-doc.org>`_ (to generate documentation)
- `pygraphviz <https://pygraphviz.github.io>`_ (for optimized graph operations and outputs)
- `munkres <http://software.clapper.org/munkres>`_ (parcellation operation)

Linux Installation
##################

Please refer to the page corresponding to your distribution.

.. toctree::
   :maxdepth: 1

   installation/ubuntu_12_04.rst
   installation/ubuntu_14_04.rst
   installation/ubuntu_16_04.rst
   installation/debian_stable.rst
   installation/fedora_21.rst
   installation/fedora_22.rst

If you have another distribution or you are an advanced user, consider
:doc:`installing pyhrf in a virtual environment <installation/venv>`.

