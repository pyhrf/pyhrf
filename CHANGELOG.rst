Changelog
=========

Current development
+++++++++++++++++++

- Fix VEM algorithm
    + fix convergence criteria computation
- Continue to clean setup.py
- Updating documentation
    + updating theme
    + fixing some reST and display errors
- autodetect cpus number (mainly to use on cluster)
- fix some DeprecationWarning that will become Exceptions in the future
- add covariance regularization matrix
- load contrasts from SPM.mat
- save contrasts in the same order that the xml configuration file
- compute and save PPMs
- optimize some functions in vem_tools
- fix detection of parcellation files

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
