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

It is advised to install python (with development headers) and all above-mentioned dependencies through distribution tools such as apt-get, urpmi or yum. Eg, for a Debian-based linux::

    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-nibabel python-nipy

Optional dependencies::

    sudo apt-get install python-qt4 python-scikits-learn python-joblib

Fedora
******

Main dependecies, as root::
  
  # yum install python-devel numpy scipy python-matplotlib python-pip

These dependencies are not available as system packages, they have to be installed
by hand ::

  $ pip install --user nibabel sympy nipy

Optional dependencies::

  # yum install PyQt4 python-matplotlib-qt4 graphviz-python # as root
  
  $ pip install --user joblib sphinx scikit-learn soma-workflow 


MAC OS X
********
.. note:: between each step of the installation, you may have to reopen a new shell or source again your shell init file, eg ``source ~/.profile``. If something is not found, try to do this.

For the C compiler, if you don't have X-code installed, you can install the `"Command Line Tools for X-code" <https://developer.apple.com/downloads/index.action>`_. Else, if X-code is installed, make sure that "Command Line Tools" are installed in the menu "preferences" -> "download".


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


A Fortran compile is required 

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
therefore all python dependencies (numpy, scipy, matplotlib,
PyQt, nipy, nibabel) should be *"egg installed"*. 
Python packages installed by the system might not compatible with the setuptools egg system. Special installation locations can be added to the ``'setup.cfg'`` file at the root directory of the pyhrf decompressed tarball. Simply append a line such as::
``site-dirs=/path/to/installed/package``

.. If dependencies are not found on the system, the installation process tries to download (therefore needing an internet connection), compile and install them
   automatically. For the compilation step, the following dependencies are
   required (specifically for numpy):

   - C compiler
   - fortran 95 compiler


Optional dependency

 $ easy_install --prefix=~/.local python-graph-core

 $ easy_install --prefix=~/.local joblib

 $ easy_install --prefix=~/.local sphinx

Setup a local python installation
#################################

To setup a local python installation, first create a directory in your home folder where all "manually" installed python packages will go. Here we use "~/.local" but this can be replaced with any other suitable name::

  $ mkdir ~/.local

Create a folder for installed binaries::

  $ mkdir ~/.local/bin

Get the current python version number, which will be used afterwards::

  $ python -c "import distutils.sysconfig as ds; print ds.get_python_version()"

Create a folder for python packages, **replace XX with the current python version number**::

  $ mkdir -p ~/.local/lib/pythonXX/site-packages/

Add a new entry in the PYTHONPATH environment variable  for the previous folder to be searchable by python. Also, add a new entre in the PATH environment variable for executable to be available.
Edit your startup script (~/.profile or ~/.bashrc) and add the following lines, **replace XX with the current python version number**::

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

First, if git is not installed (``git --version`` does not work), you can install it from here TODO.

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

     $python setup.py install --prefix=~/.local/

- local installation in develop mode (only links to the source files are installed)::

        $python setup.py develop --prefix=~/.local/

 Note: /local/installation/path/lib/python2.x/site-packages must exist and be in your ``PYTHONPATH`` environment variable. Pyhrf executables will be installed in /local/installation/bin/ and the latter should then be in the ``PATH`` environment variable (see "Setup a local installation").

*** Run tests to check installation**::

    pyhrf_maketests

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
    hosts = /home/tom/.pyhrf/hosts_LAN  ; plain text file containing coma-separated list of hostnames on the LAN
    user = None                         ; user name used to log in on any machine
                                        ; on the LAN
    remote_path = None                  ; path readable from the machine where
                                        ; pyhrf is launched (to directly retrieve
                                        ; results) 
    
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


Documentation
#############

Sphinx is used to build the document. You get it `here <http://sphinx-doc.org/install.html>`_.

To build the pyhrf documentation, launch the following command in the folder ``doc/sphinx`` located in the pyhrf repository::
 
   $ make html

This will create a folder ``html`` with all the documentation (start page: ``html/index.html``.


Troubleshooting
***************

On MAC, you can get the following error::

  $ make html
  ...
  File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/locale.py", line 375, in _parse_localename
  raise ValueError, 'unknown locale: %s' % localename
  ValueError: unknown locale: UTF-8
 
   
To fix this, add the following lines to your shell init file (``~/.profile``)::

  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8
