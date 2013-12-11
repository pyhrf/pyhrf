.. _manual_commands:


PyHRF commands
**************

fMRI data analyses
==================

pyhrf_jde_buildcfg
------------------

pyhrf_jde_estim
---------------

pyhrf_glm_buildcfg
------------------

pyhrf_glm_estim
---------------

pyhrf_rfir_buildcfg
-------------------

pyhrf_rfir_estim
----------------

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

fMRI data processing
====================

pyhrf_sur_sub
-------------

pyhrf_undrift
-------------

pyhrf_subtract
--------------

pyhrf_mean
----------

pyhrf_var
---------

pyhrf_std
---------

pyhrf_apply_mask_to_bold
------------------------

pyhrf_check_bold
----------------

SPM related
===========

pyhrf_spm_fixnames
------------------

pyhrf_spm_paradigm_from_csv
---------------------------

Misc tools
==========

pyhrf_plot_mip
--------------

pyhrf_script_shortcut
---------------------

Usage:
^^^^^^
::
        
        pyhrf_script_shortcut [options] OUTPUT_SHORTCUT

Description:
^^^^^^^^^^^^
        
        Generate an executable shortcut to a script in pyhrf's script directory.
        With no option specified, OUTPUT_SHORTCUT will contain::
        
            #! /usr/bin/python
            import pyhrf, os
            execfile(os.path.join(pyhrf.get_src_path(),"script/"))

        With the "-f" option, a specific script can be specified. Note that the
        specified script must exist in pyhrf's script folder. Example ::
        
            pyhrf_script_shortcut launchme.py -f jde_from_real_data.py
            
        OUTPUT_SHORTCUT will contain::
        
            #! /usr/bin/python
            import pyhrf, os
            execfile(os.path.join(pyhrf.get_src_path(),"script/WIP/example/jde_from_real_data.py"))
            
        
            

Options
^^^^^^^
::
       -h, --help            show this help message and exit
       -f PYHRF_SCRIPT_FILE, --script-file=PYHRF_SCRIPT_FILE
                             Script path. If it does not exist then attemptto
                             search for it within the script of pyhrf
       -v VERBOSELEVEL, --verbose=VERBOSELEVEL
                           { 0 :    'no verbose' 1 :    'minimal verbose' 2 :
                             'main steps' 3 :    'main function calls' 4 :    'main
                           variable values' 5 :    'detailled variable values' 6
                           :    'debug (everything)' }
       -s, --shell           Make a shell script rather than a python script

        

pyhrf_gls
---------

Usage:
^^^^^^
::
        
        pyhrf_gls [options] PATH

Description:
^^^^^^^^^^^^
        This commands provides compact views of data file hierarchies.

        It lists files in ``PATH`` and group file sequences: 

             - files sharing the same prefix with a number at the end 
             - files differing only by their extension
             - according to optional regular expressions 
               (Perl-style as used in the 
               `re python module <http://docs.python.org/library/re.html>`_)


        All files in a given folder are sorted in alphabetical order.
        In list mode (default), files are displayed first and then directories.
        In tree mode (``-t`` or ``--tree`` option), directories are displayed 
        first and then files.

Options
^^^^^^^
::

  -h, --help            show this help message and exit
  -v VERBOSELEVEL, --verbose=VERBOSELEVEL
                        { 0 :    'no verbose' 1 :    'minimal verbose' 2 :
                        'main steps' 3 :    'main function calls' 4 :    'main
                        variable values' 5 :    'detailled variable values' 6
                        :    'debug (everything)' }
  -g REGEXP, --group-rule=REGEXP
                        Regular expression to group specific file names. Must
                        contain a symbolic group name labeled as "group_name".
  -r, --recursive       List subdirectories recursively
  -t, --tree            Display in tree-like format
  -c COLORS, --colors=COLORS
                        If "on", display colors (using ANSI escape sequences)
                        only on TTY. If "always", display colors even if not
                        on TTY. If "off", no colors. Default is "on"

Examples:
^^^^^^^^^

Assume the following file structure::

       /subject1
       /subject1/fmri
           paradigm.csv
       /subject1/fmri/analysis
           analysis_result_1.nii
           analysis_result_2.csv
           analysis_summary.txt
       /subject1/fmri/run1
           bold_scan_0001.nii
           bold_scan_0002.nii
           bold_scan_0003.nii
       /subject1/fmri/run2
           bold_scan_0001.nii
           bold_scan_0002.nii
           bold_scan_0003.nii
       /subject1/t1mri
           anatomy.hdr
           anatomy.img


* List files recursively in tree-like format::
  
        pyhrf_gls . -rt

  .. image:: figs/pyhrf_gls_output_1.png

  'bold_scan_[1...3].nii' represents the sequence of files ranging 
  from bold_scan_0001.nii to bold_scan_0003.nii.

  The files 'anatomy.*' are grouped because they differ only by their
  extension.

* Group files starting with \'analysis\_\'::

        pyhrf_gls . -rt -g '(?P<group_name>analysis)_.*'

  .. image:: figs/pyhrf_gls_output_2.png

  All files matching the regular expression ``(?P<group_name>analysis)_.*``
  are displayed as a single label followed by three dots.
  This label is defined by the symbolic group name ``group_name`` within the
  regular expression.


pyhrf_list_datafiles
--------------------

pyhrf_info
----------

pyhrf_script_shortcut
---------------------

