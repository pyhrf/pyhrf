.. _parcelattion_mask:

Parcellation mask
*****************

The JDE framework estimates HRF parcels-wide. This means that you need a parcellation mask to compute the estimation-detection.
PyHRF provides some :doc:`tools <commands>` to generate parcellation masks and provides also a parcellation mask (see `next section <#willard-atlas>`_)

Willard atlas
+++++++++++++

The Willard atlas comes from `Stanford <http://findlab.stanford.edu/functional_ROIs.html>`_

.. image:: /figs/willard_fROIs_image.jpg

To use it check where it is installed by issuing::

    $ pyhrf_list_datafiles

and check for ``stanford_willard_parcellation_3x3x3mm.nii.gz`` file.

If you use this parcellation mask, please cite the following paper:

The citation for the 499 fROI atlas, nicknamed the "Willard" atlas after the two
creators, **Will** iam Shirer and Bern **ard** Ng, is the following publication:

`Richiardi J, et al.: Correlated gene expression supports synchronous activity
in brain networks. Science (2015). <http://science.sciencemag.org/content/348/6240/1241>`_
