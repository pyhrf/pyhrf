.. _manual_commands:


PyHRF commands
**************

fMRI data analysis
==================

pyhrf_jde_vem_analysis
----------------------

Usage:
^^^^^^
::

    pyhrf_jde_vem_analysis [options] TR parcels_file onsets_file [bold_data_run1 [bold_data_run2 ...]]

Description:
^^^^^^^^^^^^

Run an fMRI analysis using the JDE-VEM framework All arguments (even
positional ones) are optional and default values are defined in the beginning
of the script

Options:
^^^^^^^^
::

    positional arguments:
      tr                    Repetition time of the fMRI data
      parcels_file          Nifti integer file which define the parcels
      onsets_file           CSV onsets file
      bold_data_file        4D-Nifti file of BOLD data

    optional arguments:
      -h, --help            show this help message and exit
      -c DEF_CONTRASTS_FILE, --def_contrasts_file DEF_CONTRASTS_FILE
                            JSON file defining the contrasts
      -d DT, --dt DT        time resolution of the HRF (if not defined here or in
                            the script, it is automatically computed)
      -l HRF_DURATION, --hrf-duration HRF_DURATION
                            time lenght of the HRF (in seconds, default: 25.0)
      -o OUTPUT_DIR, --output OUTPUT_DIR
                            output directory (created if needed, default:
                            /home/tperret/code/pyhrf)
      --nb-iter-min NB_ITER_MIN
                            minimum number of iterations of the VEM algorithm
                            (default: 5)
      --nb-iter-max NB_ITER_MAX
                            maximum number of iterations of the VEM algorithm
                            (default: 100)
      --beta BETA           (default: 1.0)
      --hrf-hyperprior HRF_HYPERPRIOR
                            (default: 1000)
      --sigma-h SIGMA_H     (default: 0.1)
      --estimate-hrf        (default: True)
      --no-estimate-hrf     explicitly disable HRFs estimation
      --zero-constraint     (default: True)
      --no-zero-constraint  explicitly disable zero constraint (default enabled)
      --drifts-type DRIFTS_TYPE
                            set the drifts type, can be 'poly' or 'cos' (default:
                            poly)
      -v {DEBUG,10,INFO,20,WARNING,30,ERROR,40,CRITICAL,50}, --log-level {DEBUG,10,INFO,20,WARNING,30,ERROR,40,CRITICAL,50}
                            (default: WARNING)
      -p, --parallel        (default: True)
      --no-parallel         explicitly disable parallel computation
      --save-processing     (default: True)
      --no-save-processing


Parcellation tools
==================

pyhrf_parcellate_glm
--------------------

pyhrf_parcellate_spatial
------------------------

pyhrf_parcellation_extract
--------------------------

Usage:
^^^^^^
::

        pyhrf_parcellation_extract [options] PARCELLATION_FILE PARCEL_ID1 [PARCEL_ID2 ...]



Description:
^^^^^^^^^^^^

Extract a sub parcellation comprising only ``PARCEL_ID1``, ``PARCEL_ID2``, ... from the input parcellation ``PARCELLATION_FILE``.

Options:
^^^^^^^^
::

  -h, --help            show this help message and exit
  -v VERBOSELEVEL, --verbose=VERBOSELEVEL
                        { 0 :    'no verbose' 1 :    'minimal verbose' 2 :
                        'main steps' 3 :    'main function calls' 4 :    'main
                        variable values' 5 :    'detailled variable values' 6
                        :    'debug (everything)' }
  -o FILE, --output=FILE
                        Output parcellation file. Default is
                        <input_file>_<PARCEL_IDS>.(nii|gii)
  -c, --contiguous      Make output parcel ids be contiguous

Example
^^^^^^^

The input parcellation ``parcellation.nii`` looks like:

    .. image:: figs/pyhrf_parcellation_extract_input.png
       :width: 100pt

::

        pyhrf_parcellation_extract parcellation.nii 36 136 -o parcellation_sub.nii

The output parcellation ``parcellation_sub.nii`` will look like:

    .. image:: figs/pyhrf_parcellation_extract_output.png
       :width: 100pt


Misc tools
==========

pyhrf_list_datafiles
--------------------

Usage:
^^^^^^
::

   pyhrf_list_datafiles [options]

Description:
^^^^^^^^^^^^
        This command lists all data files included in the package.

Options
^^^^^^^
::

  -h, --help       show this help message and exit
  -b, --base-name  Display only basenames

Examples:
^^^^^^^^^

::

   pyhrf_list_datafiles

    /home/user/software/pyhrf/python/pyhrf/datafiles/SPM_v12.mat.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/SPM_v5.mat.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/SPM_v8.mat.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/cortex_occipital_hrf_territories_3mm.nii
    /home/user/software/pyhrf/python/pyhrf/datafiles/cortex_occipital_hrf_territories_convex_hull.tgz
    /home/user/software/pyhrf/python/pyhrf/datafiles/cortex_occipital_right_GWmask_3mm.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/cortex_occipital_white_surf.gii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/dummySmallBOLD.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/dummySmallMask.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_V4.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_a.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_av.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_av_comma.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_av_d.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_c_only.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_cp_only.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/paradigm_loc_cpcd.csv
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_surf_tiny_bold.gii
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_surf_tiny_mesh.gii
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_surf_tiny_parcellation.gii
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_vol_4_regions_BOLD.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_vol_4_regions_anatomy.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/real_data_vol_4_regions_mask.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu.pck
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_hrf_3_territories.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_hrf_3_territories_8x8.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_hrf_4_territories.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_activated.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_ghost.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_house_sun.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_icassp13.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_invader.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_pacman.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_small_spots_1.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_small_spots_2.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_stretched_1.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_template.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_tiny_1.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_tiny_2.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/simu_labels_tiny_3.png
    /home/user/software/pyhrf/python/pyhrf/datafiles/stanford_willard_parcellation_3x3x3mm.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/subj0_anatomy.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/subj0_bold_session0.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/subj0_parcellation.nii.gz
    /home/user/software/pyhrf/python/pyhrf/datafiles/subj0_single_roi.nii.gz

::

   pyhrf_list_datafiles -b

       SPM_v12.mat.gz
       SPM_v5.mat.gz
       SPM_v8.mat.gz
       cortex_occipital_hrf_territories_3mm.nii
       cortex_occipital_hrf_territories_convex_hull.tgz
       cortex_occipital_right_GWmask_3mm.nii.gz
       cortex_occipital_white_surf.gii.gz
       dummySmallBOLD.nii.gz
       dummySmallMask.nii.gz
       paradigm_V4.csv
       paradigm_loc.csv
       paradigm_loc_a.csv
       paradigm_loc_av.csv
       paradigm_loc_av_comma.csv
       paradigm_loc_av_d.csv
       paradigm_loc_c_only.csv
       paradigm_loc_cp_only.csv
       paradigm_loc_cpcd.csv
       real_data_surf_tiny_bold.gii
       real_data_surf_tiny_mesh.gii
       real_data_surf_tiny_parcellation.gii
       real_data_vol_4_regions_BOLD.nii.gz
       real_data_vol_4_regions_anatomy.nii.gz
       real_data_vol_4_regions_mask.nii.gz
       simu.pck
       simu_hrf_3_territories.png
       simu_hrf_3_territories_8x8.png
       simu_hrf_4_territories.png
       simu_labels_activated.png
       simu_labels_ghost.png
       simu_labels_house_sun.png
       simu_labels_icassp13.png
       simu_labels_invader.png
       simu_labels_pacman.png
       simu_labels_small_spots_1.png
       simu_labels_small_spots_2.png
       simu_labels_stretched_1.png
       simu_labels_template.png
       simu_labels_tiny_1.png
       simu_labels_tiny_2.png
       simu_labels_tiny_3.png
       stanford_willard_parcellation_3x3x3mm.nii.gz
       subj0_anatomy.nii.gz
       subj0_bold_session0.nii.gz
       subj0_parcellation.nii.gz
       subj0_single_roi.nii.gz
