Changelog
=========

Current development
+++++++++++++++++++

Fixes
-----

- Fix VEM algorithm
    + fix convergence criteria computation
    + fix underflow and overflow in labels expectation (set labels to previous
      value if necessary)
- Continue to clean setup.py
- fix some DeprecationWarning that will become Exceptions in the future
- fix detection of parcellation files
- fix for scikit-learn version >= 0.17
- fix bugs with matplotlib versions >1.4
- fix bug with Pillow latest version (see #146)
- fix bug with numpy when installing in virtual environment (see commit a971656)
- fix the zero constraint on HRF borders

Enhancements
------------

- optimize some functions in vem_tools
- rewrite and optimize all VEM steps
- remove old calls to verbose module and replaced them by logging standard library module

New
---

- Updating documentation
    + updating theme
    + fixing some reST and display errors
- autodetect cpus number (mainly to use on cluster and not yet documented)
- add covariance regularization matrix
- load contrasts from SPM.mat
- save contrasts in the same order that the xml configuration file
- compute and save PPMs
- Add multisession support for VEM BOLD and ASL
- add cosine drifts to VEM

-----------------------------------

Release 0.4.1.post1
+++++++++++++++++++

Fixes:
------

- missing function (#135)
- nipy version required for installation (#134)

Release 0.4.1
+++++++++++++

Fixes:
------

- logging level not set by command line (#113)
- error with VEM algorithm (#115)

Enhancements:
-------------

- clean and update setup.py (#84)
- update travis configuration file (#123)


Release 0.4
+++++++++++

2015/03/19

API Changes:
------------

- Deprecate verbose module and implements logging module instead

Fixes:
------

- clean up setup.py

Enhancements:
-------------

-----------------------------------

*No changelog for previous version*
