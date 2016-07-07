.. _multiprocessing_computation:

Multicore computing
###################

Pyhrf can make computation on several cores using the `joblib <https://pythonhosted.org/joblib/>`_ python package. Joblib is an optional dependency of pyhrf so if you plan to use multiprocessing computation you need to install it (refer to the :doc:`installation <installation>` page)

Configuration
-------------

You can configure the number of used cores by setting the ``nb_procs`` option in the :doc:`pyhrf configuration file <pyhrf_configuration>`.
If you want to auto-detect the number of processors and cores that the system can use, set this option to ``0``.

Use
---

When calling the ``pyhrf_jde_vem_analysis`` script, the multiprocessing mode is active by default.
If you want to disable it use ``--no-parallel`` command line option.
