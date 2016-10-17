Changelog
=========

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
