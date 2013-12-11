
.. _surface_treatment:

======================================================= 
Analysis of fMRI data projected on the cortical surface
======================================================= 

This analysis comprises the following steps:
     
     1. :ref:`Extract the cortical mesh <cortex_segmentation>`
     2. :ref:`Project data onto the cortical surface <data_projection>`
     3. :ref:`Run a pyhrf treatment on the projected data <pyhrf_surface_treatment>`

Data inputs are:

     * Anatomical MRI (3D)
     * fMRI data (3D+time)
     * paradigm (condition names, event timings)

.. _cortex_segmentation:

Extraction of the cortical surface. 
======================================================

The objective is to extract the grey/white matter interface (deep cortical or "white" surface) as a mesh from the anatomical MRI. It is worth noting that we would not want the gyral surface (superficial cortical or "pial" surface) nor the average cortical surface, because of the principle of the subsequent data projection. Indeed, the latter takes into account the orientation of the cortical fold and take the white surface as input.  

Several tools are available to extract the cortical surface:
        
        * `Freesurfer <http://surfer.nmr.mgh.harvard.edu/fswiki/RecommendedReconstruction>`_
        * `Brainsuite <http://users.loni.ucla.edu/~shattuck/brainsuite/corticalsurface/>`_
        * `T1 MRI toolbox of Brainvisa <http://brainvisa.info/doc/axon/en/processes/morphologist.html>`_

The T1 MRI toolbox of Brainvisa (Morphologist treatment) is recommended here but any procedure may work as long as it provides a file in `gifti format <http://www.nitrc.org/projects/gifti>`_. The library used to read meshes in Pyhrf is `Nibabel <http://nipy.sourceforge.net/nibabel/>`_.

Here is an illustration of the process used to obtain the real data set shipped with PyHRF:

.. image:: figs/segmentation_pipeline.png

The default setup of the morphologist treatment produces meshes in a Brainvisa-specific format (''.mesh'') and not gifti. You may change the file format to gifti in the sub-treatment "Grey white Interface" or use the following command to convert the resulting mesh files::

    AimsFileConvert -i subject_Xwhite.mesh -o subject_Xwhite.gii  


.. _data_projection:

.. _pyhrf_surface_treatment:


