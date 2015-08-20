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

Instructions are given for linux and python2.7. The package is mainly developed and tested under Ubuntu 14.04, we then recommend this distribution.
We officially support Ubuntu 12.04, Ubuntu 14.04 Debian stable (jessie) and Fedora 21 and 22.

For **windows users**, a linux distribution can be installed easely inside a virtual machine such as Virtual Box. This wiki page explains `how to install ubuntu from scratch in Virtual Box <http://www.wikihow.com/Install-Ubuntu-on-VirtualBox>`_ or you can get some `free VirtualBox images <http://virtualboxes.org/images/ubuntu/>`_.

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
    - PyQt4 (viewer and xml editor)
    - joblib (local distributed computation)
    - soma-workflow (remote distributed computation)
    - scikit-learn (clustering)
    - sphinx (to generate documentation)
    - pygraphviz (for optimized graph operations and outputs)
    - munkres (parcellation operation)

Linux-based
***********

Please refer to the page corresponding to your distribution.

`Ubuntu 12.04 <installation_ubuntu1204>`
========================================

`Ubuntu 14.04 <installation_ubuntu1404>`
========================================

`Debian stable (jessie) <installation_debianstable>`
====================================================

`Fedora 21 <installation_fedora21>`
===================================

The following has been tested on a new installed Fedora 20.

Main dependencies::

    $ sudo yum install python-devel numpy scipy python-matplotlib python-pip sympy git

These dependencies are not available as system packages, they have to be installed
by hand ::

    $ pip install --user nibabel
    $ pip install --user nipy

Optional dependencies::

    $ sudo yum install PyQt4 python-matplotlib-qt4 graphviz-python python-sphinx python-scikit-learn python-pillow python-joblib
    $ pip install --user munkres

`Fedora 22 <installation_fedora22>`
===================================

If you have another distribution or are an advanced user, consider `installing pyhrf in a virtual environment <installation_venv>`

.. _Pyhrf download:

PyHRF download
##############

Release tarball
***************

The latest pyhrf release (v0.3) is available `here <http://www.pyhrf.org/dist/pyhrf-0.3.tar.gz>`_.
Extract it and follow the pyHRF installation guide above.


Source repository
*****************

First, if git is not installed (``git --version`` does not work), install it from your system packages::

    $ sudo apt-get install git # on Ubuntu and Debian
    $ sudo yum install git # on Fedora as root

The bleeding edge version of pyhrf is available via github. In a folder where you want to create the pyhrf repository, use the command::

    $ git clone https://github.com/pyhrf/pyhrf.git pyhrf

Then, to get the latest changes afterwards::

    $ cd pyhrf
    $ git pull

.. _Pyhrf installation:

PyHRF Installation
##################

In the directory where the pyhrf tarball has been decompressed or in the pyhrf git repository, you can install it globally or locally:

- global installation::

    $ python setup.py install

 This will attempt to write in the Python site-packages directory and will fail if you don't have the appropriate permissions (you usually need root privilege).

- local installation::

    $ python setup.py install --user

- local installation in develop mode (only links to the source files are installed)::

    $ python setup.py develop --user

On Ubuntu and Debian, you need to add the commands folder to your ``PATH`` environement variable by adding the following to your ``$HOME/.profile``::

    if [ -d "$HOME/.local/bin"  ]; then
        PATH="$HOME/.local/bin:$PATH"
    fi

*** Run tests to check installation**::

    $ pyhrf_maketests

Configuration
#############

Package options are stored in ``$HOME/.pyhrf/config.cfg``, which is created after the installation. It handles global package options and the setup of parallel processing. Here is the default content of this file (section order may change)::


    [global]
    write_texture_minf = False          ; compatibility with Anatomist for texture file
    tmp_prefix = pyhrftmp               ; prefix used for temporary folders in tmp_path
    verbosity = 0                       ; default of verbosity, can be changed with option -v
    tmp_path = /tmp/                    ; where to write file
    use_mode = enduser                  ; "enduser": stable features only, "devel": +indev features
    spm_path = None                     ; path to the SPM matlab toolbox (indev feature)


    [parallel-cluster]                  ; Distributed computation on a cluster.
                                        ; Soma-workflow is required.
                                        ; Authentification by ssh keys must be
                                        ; configured in both ways (remote <-> local)
                                        ; -> eg copy content of ~/.ssh/id_rsa.pub (local machine)
                                        ;    at the end of ~/.ssh/authorized_keys (remote machine)
                                        ;    Also do the converse:
                                        ;    copy content of ~/.ssh/id_rsa.pub (remote machine)
                                        ;    at the end of ~/.ssh/authorized_keys (local machine)

    server_id = None                    ; ID of the soma-workflow-engine server
    server = None                       ; hostname or IP adress of the server
    user = None                         ; user name to log in the server
    remote_path = None                  ; path on the server where data will be stored

    [parallel-local]                    ; distributed computation on the local cpu
    niceness = 10                       ; niceness of remote jobs
    nb_procs = 1                        ; number of distruted jobs, better not over
                                        ; the total number of CPU
                                        ; 'cat /proc/cpuinfo | grep processor | wc -l' on linux
                                        ; 'sysctl hw.ncpu' on MAC OS X

    [parallel-LAN]                      ; Distributed computation on a LAN
                                        ; Authentification by ssh keys must be
                                        ; configured
    remote_host = None                  ; hostname or IP address of a host on the LAN
    niceness = 10                       ; niceness for distributed jobs
    hosts = $HOME/.pyhrf/hosts_LAN      ; plain text file containing coma-separated list of hostnames on the LAN
    user = None                         ; user name used to log in on any machine
                                        ; on the LAN
    remote_path = None                  ; path readable from the machine where
                                        ; pyhrf is launched (to directly retrieve
                                        ; results)

.. see :ref:`Parallel Computation <manual_parallel>`

Documentation
#############

Sphinx is used to build the document. You get it `here <http://sphinx-doc.org/install.html>`_.

To build the pyhrf documentation, launch the following command in the folder ``doc/sphinx`` located in the pyhrf repository::

   $ make html

This will create a folder ``html`` with all the documentation (start page: ``html/index.html``.
