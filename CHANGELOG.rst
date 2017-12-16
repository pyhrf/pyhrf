Changelog
=========

0.4.4 (2017/12/08)
++++++++++++++++++

Fixes
-----

- Fix syntax problems with the rst files of the documentation
- Fix docstrings of some python methods
- Fix version of the dependencies
- Fix the float index error from the ``vem_tools.py`` script
- Fix the cosine drift problem caused by the latest version of ``numpy``
- Fix the error caused by trying to compute the `log(0)` in the entropy function
- Fix the bug of the `free_energy_computation` function when the HRF is not estimated.


Enhancements
------------

- Requirements file (``requirements.txt``) for developing and installation
- Improvement of the installation script (``setup.py``)

  + Auto-install of dependencies using ``pip`` and the ``requirements.txt`` file.
  + Python version 2.7.x required
  + C extensions are only compiled when PyHRF is built or installed. The same for the installation of the dependencies
  + The outputs of the installation script are colored in order to highlight important information

- Documentation

  + Improvement readability of the ``README`` and ``CHANGELOG`` files.
  + Improvement of the documentation with detailed VEM formulas
  + Extension of the documentation with the NeuroElf based visualization tool
  + Extension of the documentation with the installation steps for different GNU/Linux distributions
  + Extension of the documentation with a developer section
  + Deprecated sections of the documentation are hidden


New
---

- PyHRF visualization tool based on the `NeuroElf <http://neuroelf.net>`_ tool
- Creation of a `Dockerfile <https://github.com/pyhrf/docker-pyhrf>`_

  + Sharing the docker image on Docker Hub (`pyrhf/pyhrf`)
  + The CI service (``travis``) runs the tests and the coveralls using the docker image
  + The releases are now generated using and creating a docker image

- Documentation

  + Using a responsive, friendly and beautiful Sphinx theme (``sphinx_bootstrap_theme``)
  + Using `readthedocs <https://readthedocs.org/projects/pyhrf/>`_ to build automatically the documentation


Optimizations
-------------

- Optimization of some VEM methods

  + ``pyhrf.vbjde.vem_tools.create_conditions``
  + ``pyhrf.vbjde.vem_tools.create_neighbours``
  + ``pyhrf.vbjde.vem_tools.poly_drifts_basis``
  + ``pyhrf.vbjde.vem_tools.cosine_drifts_basis``
  + ``pyhrf.vbjde.vem_tools.norm1_constraint``
  + ``pyhrf.vbjde.vem_tools.nrls_expectation``
  + ``pyhrf.vbjde.vem_tools.hrf_entropy``
  + ``pyhrf.vbjde.vem_tools.two_gamma_hrf``
  + ``pyhrf.vbjde.vem_tools.ppms_computation``
  + ``pyhrf.vbjde.vem_tools.computeFit``
  + ``pyhrf.vbjde.vem_tools.contrasts_mean_var_classes``
  + ``pyhrf.vbjde.vem_bold.jde_vem_bold``


0.4.3 (2016/07/08)
++++++++++++++++++

Fixes
-----

- Remove non-existing tests in devel mode
- Correct typo in documentation
- Correct list display in documentation
- Fix bug for ``numpy >= 1.11.1`` version
- Fix bug in contrasts computation
- Clean ``tmp`` folders after some unitary tests
- Fix VEM script example

0.4.2 (2016/07/07)
++++++++++++++++++

Fixes
-----

- Fix VEM algorithm

  + Fix convergence criteria computation
  + Fix underflow and overflow in labels expectation (set labels to previous
    value if necessary)

- Continue to clean ``setup.py``
- Fix some ``DeprecationWarning`` that will become ``Exceptions`` in the future
- Fix detection of parcellation files
- Fix for ``scikit-learn`` version >= 0.17
- Fix bugs with ``matplotlib`` versions >1.4
- Fix bug with ``Pillow`` latest version (see `#146 <https://github.com/pyhrf/pyhrf/issues/146>`_)
- Fix bug with ``numpy`` when installing in virtual environment (see commit `a971656 <https://github.com/pyhrf/pyhrf/commit/a971656>`_)
- Fix the zero constraint on HRF borders

Enhancements
------------

- Optimize some functions in vem_tools
- Rewrite and optimize all VEM steps
- Remove old calls to verbose module and replaced them by logging standard library module
- Update website documentation

New
---

- Updating documentation

  + Updating theme
  + Fixing some reST and display errors

- Auto-detect CPUs number (mainly to use on cluster and not yet documented)
- Add covariance regularization matrix
- Load contrasts from ``SPM.mat``
- Save contrasts in the same order that the xml configuration file
- Compute and save PPMs
- Add multi-session support for VEM BOLD and ASL
- Add cosine drifts to VEM
- Add command-line for VEM
- Add `Stanford Willard Parcellation <http://findlab.stanford.edu/functional_ROIs.html>`_

0.4.1.post1
+++++++++++

Fixes
-----

- Missing function (see `#135 <https://github.com/pyhrf/pyhrf/issues/135>`_)
- ``nipy`` version required for installation (see `#134 <https://github.com/pyhrf/pyhrf/issues/134>`_)

0.4.1 (2015/08/19)
++++++++++++++++++

Fixes
-----

- Logging level not set by command line (see `#113 <https://github.com/pyhrf/pyhrf/issues/113>`_)
- Error with VEM algorithm (see `#115 <https://github.com/pyhrf/pyhrf/issues/115>`_)

Enhancements
------------

- Clean and update setup.py (see `#84 <https://github.com/pyhrf/pyhrf/issues/84>`_)
- Update travis configuration file (see `#123 <https://github.com/pyhrf/pyhrf/issues/123>`_)


0.4 (2015/07/22)
++++++++++++++++

API Changes
-----------

- Deprecate verbose module and implements logging module instead

Fixes
-----

- Clean up setup.py

-----------------------------------

*No changelog for previous versions*
