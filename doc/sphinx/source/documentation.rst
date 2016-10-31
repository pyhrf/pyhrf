Documentation
=============

PyHRF is a set of tools for within-subject fMRI data analysis, focused on the
characterization of the **hemodynamics**.

Within the chain of fMRI data processing, these tools provide alternatives to
the classical within-subject GLM fitting procedure. The inputs are preprocessed
images (except spatial smoothing) and the outputs are the contrast maps and the
HRF estimates.

The package is mainly written in Python and provides the implementation of the
two following methods:

- **The joint-detection estimation (JDE)** approach, which divides the brain
  into functionally homogeneous regions and provides one HRF estimate per
  region as well as response levels specific to each voxel and each
  experimental condition. This method embeds a temporal regularization on the
  estimated HRFs and an adaptive spatial regularization on the response levels.

- **The Regularized Finite Impulse Response (RFIR)** approach, which provides
  HRF estimates for each voxel and experimental conditions. This method embeds
  a temporal regularization on the HRF shapes, but proceeds independently
  across voxels (no spatial model). See :ref:`introduction` for a more detailed
  overview.

To cite PyHRF and get a comprehensive description, please refer to:

   Vincent, T., Badillo, S., Risser, L., Chaari, L., Bakhous, C., Forbes, F., &
   Ciuciu, P. (2014). *Flexible multivariate hemodynamics fMRI data analyses and
   simulations with PyHRF*. Frontiers in Neuroscience, 8. https://doi.org/10.3389/fnins.2014.00067

Site content
------------

.. toctree::
   :maxdepth: 2

   introduction.rst
   installation.rst
   manual.rst
   API Reference <autodoc/pyhrf.rst>
   changelog.rst

Licence and authors
-------------------

PyHRF is currently under the `CeCILL licence version 2 <http://www.cecill.info>`_.
Originally developed by the former `LNAO <http://www.lnao.fr>`_ (Neurospin, CEA),
pyHRF is now entering (since Sep 2014) in a new era under the joint
collaboration of the the `Parietal team <http://parietal.inria.fr/>`_ (Inria Saclay)
and the `MISTIS team <http://mistis.inrialpes.fr/>`_ (Inria Rhones-Alpes).

People who have significantly contributed to the development are
(by chronological order): Thomas Vincent\ :sup:`(1,3)`,
Philippe Ciuciu\ :sup:`(1,2)`, Lotfi Chaari\ :sup:`(3)`,
Solveig Badillo\ :sup:`(1,2)`, Christine Bakhous\ :sup:`(3)`,
Aina Frau-Pascual\ :sup:`(2,3)`, Thomas Perret\ :sup:`(3)`, and
Jaime Arias\ :sup:`(3)`.

1. CEA/DRF/I2BM/NeuroSpin, Gif-Sur-Yvette, France
2. Inria/CEA Parietal, Gif-Sur-Yvette, France
3. Inria, MISTIS, Grenoble, France

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
