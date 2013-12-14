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

Instructions are given for linux and python2.7. The package is mainly developed and tested under ubuntu (>10.04), we then recommend this distribution.

For **windows users**, a linux distribution can be installed easely inside a virtual machine such as Virtual Box. This wiki page explains `how to install ubuntu from scratch in Virtual Box <http://www.wikihow.com/Install-Ubuntu-on-VirtualBox>`_ or you can get some `free VirtualBox images <http://virtualboxes.org/images/ubuntu/>`_.

**Requirements / Dependencies**
###############################

Dependencies are:
    - `python <http://www.python.org>`_ 2.5, 2.6 or 2.7 **with development headers**
    - `numpy <http://docs.scipy.org/doc/numpy/user/install.html>`_ >= 1.0
    - `scipy <http://www.scipy.org/install.html>`_ >= 0.7
    - `matplotlib <http://matplotlib.org/users/installing.html>`_ 
    - `PyXML <http://pyxml.sourceforge.net/topics/index.html>`_ >= 0.8.4
    - `nibabel <http://nipy.sourceforge.net/nibabel/>`_
    - `nipy <http://nipy.sourceforge.net/nipy/stable/users/installation.html>`_
    - a C compiler 

Optional dependencies:
    - PyQt4 (viewer and xml editor)
    - joblib (local distributed computation)
    - soma-workflow (remote distributed computation)
    - scikit-learn (clustering)


Linux-based
***********

It is advised to install python (with development headers) and all above-mentioned dependencies through distribution tools such as apt-get or urpmi. Eg, for a Debian-based linux::

    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-nibabel python-nipy

Optional dependencies::

    sudo apt-get install python2.7-qt4 python-scikits-learn python-joblib

MAC OS X
********

For the C compiler, if you don't have X-code installed, you can install the `"Command Line Tools for X-code" <https://developer.apple.com/downloads/index.action>`_. Else, if X-code is installed, make sure that Command Line Tools are installed in the menue "preferences" -> "download".

It is recommended to install part of the "scipy stack" from `individual binary source packages <http://www.scipy.org/install.html#individual-binary-and-source-packages>`_. In this page, follow the links and install binary packages in the following order: 

 * python (Even if a python installation is already available on your system, it is preferable to get the "official" installation. Otherwise, other python package may not install correctly).
 * numpy
 * scipy
 * matplotlib

Install PyQt4 from the following `link <http://sourceforge.net/projects/pyqtx/files/latest/download>`_.

Install nibabel via easy_install:

    if you want a system installation (require root privilege)::

      $ sudo easy_install nibabel

    If you want a local installation in a custom directory 
    (see "Setup a local python installation" before running this)::

      $ easy_install --prefix=/custom/path nibabel

Sympy (required by nipy) is not available as .dmg, but can be installed via easy_install:

    if you want a system installation (require root privilege)::

      $ sudo easy_install sympy

    If you want a local installation in a custom directory 
    (see "Setup a local python installation" before running this)::

      $ easy_install --prefix=/custom/path sympy


It is recommended to install the bleeding edge version of nipy. 

    Grab the code::

      $ git clone https://github.com/nipy/nipy.git nipy-dev
      $ cd nipy-dev

    Then, if you want a system installation (require root privilege)::

      $ sudo python setup.py install

    If you want a local installation in a custom directory 
    (see "Setup a local python installation" before running this) ::

      $ python setup.py install --prefix=~/.local



Dependency troubleshooting
**************************

The pyhrf installation process relies on distribute (overlay of distutils), 
therefore all python dependencies (numpy, scipy, PyXML, matplotlib,
PyQt, nipy, nibabel) should be *"egg installed"*. 
Python packages installed by the system might not compatible with the setuptools egg system. Special installation locations can be added to the ``'setup.cfg'`` file at the root directory of the pyhrf decompressed tarball. Simply append a line such as::
``site-dirs=/path/to/installed/package``

.. If dependencies are not found on the system, the installation process tries to download (therefore needing an internet connection), compile and install them
   automatically. For the compilation step, the following dependencies are
   required (specifically for numpy):

   - C compiler
   - fortran 95 compiler

Setup a local python installation
#################################

To setup a local python installation, first create a directory in your home folder where all "manually" installed python packages will go. Here we use "~/.local" but this can be replaced with any other suitable name::

  $ mkdir ~/.local

Create a folder for installed binaries::

  $ mkdir ~/.local/bin

Get the current python version, which will be used afterwards::

  $ python -c "import distutils.sysconfig as ds; print ds.get_python_version()"`/site-packages/"

Create a folder for python packages, **replace XX with the current python version**::

  $ mkdir -p ~/.local/lib/pythonXX/site-packages/

Add a new entry in the PYTHONPATH environment variable  for the previous folder to be searchable by python. Also, add a new entre in the PATH environment variable for executable to be available.
Edit your startup script (~/.profile or ~/.bashrc) and add the following lines, **replace XX with the current python version**::

  export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/pythonXX/site-packages/
  export PATH=$PATH:$HOME/.local/bin/


.. _Pyhrf download:

**PyHRF download**
##################

Release tarball
***************

The latest pyhrf release (v0.3) is available `here <http://www.pyhrf.org/dist/pyhrf-0.3.tar.gz>`_


Source repository
*****************

The bleeding edge version of pyhrf is available via github. In a folder where you want to create the pyhrf repository, use the command::

    $ git clone https://github.com/pyhrf/pyhrf.git pyhrf
  
Then, to get the latest changes afterwards::

    $ cd pyhrf
    $ git pull  
                  
.. _Pyhrf installation:

**PyHRF Installation**
######################

In the directory where the pyhrf tarball has been decompressed or in the pyhrf git repository, you can install it globally or locally:

- global installation::

     $python setup.py install 
    
 This will attempt to write in the Python site-packages directory and will fail if you don't have the appropriate permissions (you usually need root privilege).
    
- local installation::

     $python setup.py install --prefix=/local/installation/path/

 Note: /local/installation/path/lib/python2.x/site-packages must exist and be in your ``PYTHONPATH`` environment variable. Pyhrf executables will be installed in /local/installation/bin/ and the latter should then be in the ``PATH`` environment variable (see "Setup a local installation").

*** Run tests to check installation**::

    pyhrf_maketests

*** Develop mode:**

Installation in development mode (only links to the source files are installed)::

        $python setup.py develop --prefix=/local/installation/path/


Prior to the install, the installation path should have the following folders  :
 /local/installation/path/bin/
 /local/installation/path/lib/pythonXX/site-packages/ # XX is you python version
 
Add /local/installation/path/bin/ to the PATH environment variable. It will contain commands.
Add /local/installation/path/lib/pythonXX/site-packages/ to the PYTHONPATH environment. For example, you can add the following line in ~/.bash_profile or ~/.bashrc::

    export PATH=$PATH:/local/installation/path/bin/
    export PYTHONPATH=$PYTHONPATH:/local/installation/path/lib/pythonXX/site-packages/




**Configuration**
#################

Package options are stored in $HOME/.pyhrf/config.cfg, which is created after the installation. It handles global package options and the setup of parallel processing. Here is the default content of this file (section order may change)::

    [parallel-cluster]
    server_id = None
    server = None
    user = None
    remote_path = None
    
    [parallel-local]
    niceness = 10
    nb_procs = 1
    
    [global]
    write_texture_minf = False
    tmp_prefix = pyhrftmp
    verbosity = 0
    tmp_path = /tmp/
    use_mode = enduser
    spm_path = None
    
    [parallel-LAN]
    remote_host = None
    niceness = 10
    hosts = /home/tom/.pyhrf/hosts_LAN
    user = None
    remote_path = None
    

In the **global** section, parameters are used for:

   * *tmp_path*: path where to store temporary data
   * *tmp_prefix*: label used for temporary folders
   * *use_mode* (enduser/devel): define the user level. 'enduser' implies simpler and ready-to-use default configuration steps. 'devel' enables all in-dev features and provides default configurations mainly used for testing. 
   * *write_texture_minf* (True/False): enables writing extra header information in a minf file for texture output (Brainvisa format).

All **parallel-XXX** sections concern an in-dev feature which enables distributed analyses across machines in a local network or on a multi-cores cluster. This is not yet documented (but soon will be ...).

.. see :ref:`Parallel Computation <manual_parallel>`

.. 
   ** Installation from source
   
   
   bashrc : 
   export PYTHONPATH=/local/lib/site-pacakges ...
   export PATH=$HOMELOCAL/bin/:$PATH
   mkdir -p /local/lib/site ...
   
   grab nibabel
   easy-install --prefix=~/local nibabel
   
   sympy (dep of nipy): issue easy_install installs ver python3.2 rather than py2.7
   -> grab a tarball
   untargz
   python setup.py install --prefix=~/local
   
   easy_install --prefix=~/local nipy
   is direcoty issue -> grab tarball, uncompress, python setup.py install ...
   install may not work, try develop
   
   
   Grab sources of pyhrf:
   
   login: brainvisa
   password: Soma2009
   svn co https://bioproj.extra.cea.fr/neurosvn/brainvisa/pyhrf/pyhrf-free/trunk pyhrf-free_trunk
   
   svn co https://bioproj.extra.cea.fr/neurosvn/brainvisa/pyhrf/pyhrf-free/trunk pyhrf-gpl_trunk
   
   
   cd pyhrf-free_trunk
   python setup.py develop --prefix ...
   #TODO: remove import of pyhrf at the end or remove creating tmp path at import
