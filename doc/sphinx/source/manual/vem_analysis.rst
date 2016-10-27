.. _vem_analysis:

Variational (VEM) analysis
**************************

To run a **JDE-VEM analysis** you need:

- a :doc:`paradigm csv file <paradigm>` corresponding to your experimental design
- the pre-processed (spatially **unsmoothed**) BOLD data associated with a single
  run (*4D-Nifti file*)
- the TR (in s) used at acquisition time
- a parcellation mask (if you normalize your data into MNI space, you can use
  the one provided with pyhrf)

If you want to use provided parcellation mask, check the path of the file by
running:

.. code:: bash

    $ pyhrf_list_datafiles

and check for the ``stanford_willard_parcellation_3x3x3mm.nii.gz`` file.

Then you can run:

.. code:: bash

    $ pyhrf_jde_vem_analysis 2.5 stanford_willard_parcellation_3x3x3mm.nii.gz paradigm.csv bold_data_session1.nii

If you want to tune the parameters, check the command line options (see
:doc:`commands <commands>`).
It is advised to set the ``dt`` parameter (the temporal resolution of the HRF)
and the output folder.

*The multirun extension is currently under development and testing*.

Example
=======

Check out the path of the files ``subj0_bold_session0.nii.gz``,
``subj0_parcellation.nii.gz`` and ``paradigm_loc.csv`` using:

.. code:: bash

    $ pyhrf_list_datafiles

then run:

.. code:: bash

    $ pyhrf_jde_vem_analysis 2.4 subj0_parcellation.nii.gz paradigm_loc.csv subj0_bold_session0.nii.gz

replacing the files by their full path. This will create output files in the
current folder. Check :doc:`visualizer` to check the outputs file.