.. _introduction:

==============
 Introduction
==============

Scientific issues
#################

PyHRF aims to provide advanced tools for within-subject analysis of functional Magnetic Resonance Imaging (fMRI) data acquired
during an experimental paradigm (i.e. not resting-state). The core idea of PyHRF is to estimate the dynamics of brain activation by recovering the so-called Hemodynamic Response Function (HRF) in a spatially varying manner. To this end, PyHRF implements two
different approaches: 
 (1) a voxel-wise and condition-wise HRF estimation [1];
 (2) a parcel-wise spatially adaptive joint detection-estimation (JDE) algorithm [2,3]. 

The second approach is more powerful since it jointly addresses (i) the localization of evoked brain activity in response to external stimulation and (ii) 
(ii) the estimation of the parcelwise hemodynamic filters. To this end, a parcellation (either functional or anatomical) has to be provided as input parameter.


This tool hence provides interesting perspectives for understanding the HRF differences between different populations (infants, children, adults, patients ...). Within the classical workflow of fMRI data analysis (preprocessings, estimation, statistical inference), PyHRF takes place at the estimation stage. However, we
have also implemented posterior probability maps (PPMs) to allow the user performing statistical inference at the subject-level.

For the sake of computational efficiency, the variational expectation-maximization (VEM) [3] is used as default algorithm for computing the solution. The results
that are generated in the context are the following:
 -- the 3D effect size maps associated with each experimental condition;
 -- the 3D contrast maps specified by the user;
 -- the time-series describing the HRFs for the set of all brain regions (4D volume). The analysis can also be performed on the cortical surface from projected BOLD signals and
then produces functional textures to be displayed on the input cortical mesh. 

[1] P. Ciuciu, J.-B. Poline, G. Marrelec, J. Idier, Ch. Pallier, and H. Benali, "Unsupervised robust non-parametric estimation of the
hemodynamic response function for any fMRI experiment", IEEE Trans. Medical Imaging; 22(10):1235-1251, Oct 2003.

[2] T. Vincent, L. Risser and P. Ciuciu, "Spatially adaptive mixture modeling for analysis of within-subject
	fMRI time series", IEEE Trans. Medical Imaging; 29(4):1059-1074, Apr 2010.

[3] L. Chari, T. Vincent, F. Forbes, M. Dojat and P. Ciuciu, "Fast joint detection-estimation of evoked brain activity in event-related fMRI using a variational approach", IEEE Trans. Medical Imaging; 32(5):821-837, May 2013.


Package overview
################
pyhrf is mainly written in Python, with some C-extension that handle computationally intensive parts of the
algorithms. The package relies on classical scientific libraries: numpy, scipy, matplotlib as well as Nibabel to
handle input/outputs and NiPy which provides tools for functional data analysis. 

pyhrf can be used in a stand-
alone fashion and provides a set of simple commands in a modular fashion. The setup process is handled through
XML files which can be adapted by the user from a set of templates. This format was chosen for its hierarchical
organisation which suits the nested nature of the algorithm parametrisations. A dedicated XML editor is provided
with a PyQt graphical interface for a quicker edition and also a better review of the treatment parameters. When
such an XML setup file is generated ab initio, it defines a default analysis which involves a small real data set shipped
with the package. This allows for a quick testing of the algorithm and is also used for demonstration purpose.

An articifial fMRI data generator is provided where the user can test the behaviour of the algorithms with different activation configurations, HRF shapes, nuisance types (noise, low frequency drifts) and
paradigms (slow/fast event-related or bloc designs). 

Concerning the analysis process, which can be computationally
intensive, pyhrf handles parallel computing through the python software soma-workflow for the exploitation of
cluster units as well as multiple core computers. 

Finally, results can be browsed by a dedicated viewer based on
PyQt and matplotib which handles n-dimensionnal images and provide suitable features for the exploration of
whole brain hymodynamics results.
