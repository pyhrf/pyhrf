.. _introduction:

==============
 Introduction
==============

Scientific issues
#################

PyHRF aims at providing advanced tools for within-subject analysis in
event-related functional Magnetic Resonance Imaging (fMRI). Actually, our goal
is to detect and localize brain activations while estimating the HRF time course
in a spatially adaptive (ie parcel-based) GLM.

More precisely, this software addresses the two main tasks involved in fMRI analysis: 
(i) detect and localize
which cerebral regions are activated by a given experimental paradigm 
(ii) estimate the underlying dynamics of activation by recovering the so-called Hemodynamic Response Function (HRF). 

The main outputs are then a set of 3D statistical maps of cerebral activations along with the time-series describing the HRFs for the set of all brain
regions (4D volume). The analysis can also be performed on the cortical surface from projected BOLD signals and
then produces functional textures to be displayed on the input cortical mesh. To this end, pyhrf implements two
different approaches: a voxel-wise and condition-wise HRF estimation [1] and a parcel-wise spatially adaptive joint
detection-estimation algorithm [2,3]. This tool provides interesting perspectives so as to understand the differences
in the HRFs of different populations (infants, children, adults, patients ...). Within the treatment pipeline of an
fMRI analysis, pyhrf steps in after data preprocessings (slice-timing, realignment, normalization).

[1] P. Ciuciu, J.-B. Poline, G. Marrelec, J. Idier, Ch. Pallier, and H. Benali, "Unsupervised robust non-parametric estimation of the
hemodynamic response function for any fMRI experiment," IEEE Trans. Med. Imag., vol. 22, no. 10, pp. 1235-1251, oct. 2003.

[2] T. Vincent, L. Risser, J. Idier, and P. Ciuciu, "Spatially adaptive mixture modelling for analysis of fMRI time series," in Proc.
15th HBM, San Francisco, CA, USA, juin 2009.

[3] L. Chari, F. Forbes, P. Ciuciu, T. Vincent, and M. Dojat, "Bayesian variational approximation for the joint detection-estimation
of brain activity in fMRI," in IEEE Workshop on Statistical Signal Processing (SSP 2011), Nice, France, juin 2011.


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
