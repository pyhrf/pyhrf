.. _visualizer:

===================
Visualization tools
===================

This page sums up some advices on how to visualize PyHRF results.

NeuroElf
########

For exploration of results, you can use the `NeuroElf viewer <http://neuroelf.net/>`_
with small modifications of the code to allow better visualization of HRFs and results.
NeuroElf is a Matlab toolbox so you'll need to have Matlab installed on your system.

Installation
------------

Go to the `NeuroElf page <http://neuroelf.net/>`_ and follow instructions for NeuroElf installation.
Prefer the *flat folder snapshot*, extract it and add the path to your MATLAB's path.

For easier use of the tool, you can modify part of the code for better HRF and PPMs visualizations.
The modifications for PyHRF are available `here for download <_static/NeuroElf_modifications.tar.xz>`_ (just replace the original files).

If you prefer (for advanced use) you can alternatively download the `patchs version <_static/NeuroElf_modifications_patch.tar.xz>`_

Use
---

    #. Run NeuroElf from Matlab: ``neuroelf_gui``.
    #. (Optional) From the ``File`` menu, click on ``Open...`` and select the Anatomical file from the subject.
    #. From the ``File`` menu, open the ``jde_vem_hrf_mapped.nii`` file.
    #. From the ``File`` menu, select the ``Set underlay object`` item and select either the loaded anatomical file or the one provided by NeuroElf
    #. From the ``File`` menu, select ``Open file as stats`` item and load all the ``jde_vem_ppm_a_nrl_[condition].nii`` files at once.
    #. You can then select from the left menu the condition you want to visualize (you can tweak the threshold if you need to on the right panel).


Nilearn
#######

For global visualization of PyHRF results, we advise to use the `Nilearn <https://nilearn.github.io/>`_ plot functions.

We will provide some examples to plot PPMs maps as well as HRFs from active parcels.


PyHRF Viewer
############

There exists an `old viewer <https://github.com/pyhrf/pyhrf_viewer>`_ for pyhrf you can try (which is not supported nor recommended anymore).
