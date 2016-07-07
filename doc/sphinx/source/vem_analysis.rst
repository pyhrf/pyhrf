.. _vem_analysis:

Variational (VEM) analysis
**************************

To run a JDE-VEM analysis you need:

  - :doc:`paradigm csv file <paradigm>` corresponding to your experimental design
  - the preprocessed (spatially UNsmoothed) BOLD data associated with a single run [4D-Nifti file]
  - the TR (in s) used at acquisition time
  - a parcellation mask (if you normalize your data into MNI space, you can use the one provided with pyhrf)

If you want to use provided parcellation mask, check the path of the file by running::

    $ pyhrf_list_datafiles

and check for the ``stanford_willard_parcellation_3x3x3mm.nii.gz`` file.

Then you can run::

    $ pyhrf_jde_vem_analysis 2.5 stanford_willard_parcellation_3x3x3mm.nii.gz paradigm.csv bold_data_session1.nii

If you want to tune the parameters, check command line options (see :doc:`commands <commands>`).
This is adviced to set the ``dt`` parameter (the temporal resolution of the HRF) and the output folder.

* The multirun extension is currently under development and testing
