.. _installation:

.. format of titles:

   =====
   lvl 1
   =====

   lvl2
   ####

   lvl3
   ****

   lvl4
   ====

   lvl5
   ----


==============
 Installation
==============

Instructions are given for Linux and Python 2.7. The package is mainly developed and tested under Ubuntu 14.04, we then recommend this distribution.
We officially support Ubuntu 12.04, Ubuntu 14.04 Debian stable (jessie) and Fedora 21 and 22. See `below <#linux-based>`_ for instructions for each distribution.

For **windows users**, a linux distribution can be installed easily inside a virtual machine such as Virtual Box. This wiki page explains `how to install ubuntu from scratch in Virtual Box <http://www.wikihow.com/Install-Ubuntu-on-VirtualBox>`_ or you can get some `free VirtualBox images <http://virtualboxes.org/images/ubuntu/>`_.

Requirements / Dependencies
###########################

Dependencies are:
    - `python <http://www.python.org>`_ 2.5, 2.6 or 2.7 **with development headers**
    - `numpy <http://docs.scipy.org/doc/numpy/user/install.html>`_ >= 1.0
    - `scipy <http://www.scipy.org/install.html>`_ >= 0.7
    - `matplotlib <http://matplotlib.org/users/installing.html>`_
    - `nibabel <http://nipy.sourceforge.net/nibabel/>`_
    - `sympy <http://sympy.sourceforge.net>`_ (required by nipy)
    - `nipy <http://nipy.sourceforge.net/nipy/stable/users/installation.html>`_
    - a C compiler

Optional dependencies:
    - PyQt4 (viewer and XML editor)
    - joblib (local distributed computing)
    - paramiko (local network distributed computing)
    - soma-workflow (remote distributed computing)
    - scikit-learn (clustering)
    - sphinx (to generate documentation)
    - pygraphviz (for optimized graph operations and outputs)
    - munkres (parcellation operation)

Linux-based
***********

Please refer to the page corresponding to your distribution.

:doc:`Ubuntu 12.04 <installation_ubuntu1204>`
=============================================

:doc:`Ubuntu 14.04 <installation_ubuntu1404>`
=============================================

:doc:`Debian stable (jessie) <installation_debianstable>`
=========================================================

:doc:`Fedora 21 <installation_fedora21>`
========================================

:doc:`Fedora 22 <installation_fedora22>`
========================================

If you have another distribution or are an advanced user, consider :doc:`installing pyhrf in a virtual environment <installation_venv>`
