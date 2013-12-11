.. pyhrf documentation master file, created by
   sphinx-quickstart on Thu Jul 21 16:39:42 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

    <h1 style='text-align: center; font-size: 230%;'>
    PyHRF: a python package to study hemodynamics in fMRI.
    </h1>

Overview
--------

PyHRF is a set of tools for within-subject fMRI data analysis, focused on the characterization of the hemodynamics. 

Within the chain of fMRI data processing, these tools provide alternatives to the classical within-subject GLM estimation step. The inputs are preprocessed within-subject data and the outputs are statistical maps and/or fitted HRFs.

The package is mainly written in Python and provides the implementation of the two following methods:

      * **The joint-detection estimation (JDE)** approach, which divides the brain into functionnaly homogeneous regions and provides one HRF estimate per region as well as response levels specific to each voxel and each experimental condition. This method embeds a temporal regularization on the estimated HRFs and an adaptive spatial regularization on the response levels.

      * **The Regularized Finite Impulse Response (RFIR)** approach, which provides HRF estimates for each voxel and experimental conditions. This method embeds a temporal regularization on the HRF shapes, but proceeds independently across voxels (no spatial model).

See :ref:`introduction` for a more detailed overview.

.. Developpment status
.. -------------------

Site content:
-------------
    .. toctree::
       :maxdepth: 2     
    
       introduction.rst
       installation.rst
       manual.rst

..     code_documentation.rst
    
    
..       
    Indices and tables
    ==================
    
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`



Licence and authors
-------------------

PyHRF is currently under the `CeCILL licence version 2 <http://www.cecill.info>`_. It is mainly developed at the `LNAO <http://www.lnao.fr>`_ (Neurospin, CEA) and is also involved in a collaboration with the `MISTIS team <http://mistis.inrialpes.fr/>`_ (INRIA Rhones-Alpes).

Authors are:
         Thomas Vincent\ :sup:`(1)`, Philippe Ciuciu\ :sup:`(1)`, Lotfi Chaari\ :sup:`(2)`, Solveig Badillo\ :sup:`(1)`, Christine Bakhous\ :sup:`(2)`

         1. CEA/DSV/I2BM/Neurospin, LNAO, Gif-Sur-Yvette, France
         2. INRIA, MISTIS, Grenoble, France

Contacts
++++++++

thomas.vincent@cea.fr, philippe.ciuciu@cea.fr        
