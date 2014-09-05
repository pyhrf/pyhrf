.. _data_management:

Data Management
***************

Data import/export with files (no DB)
=====================================

In fMRI data analyses, files are often organised hierarchically in folders that
has to be compatible with the analysis tools. Quite often, data files 
originate from data importer specific to some MRI scanner or to an online
platform such as `Shanoir <https://shanoir.irisa.fr:8443/Shanoir/>`_. 
These tools provide archives where data files are stored in a flat structure.

Here is a tool to export data files from such flat structure into a hierarchicy
of folder. It can also be used to import data files from a specific hierarchy 
to another.

To do so, the function :py:meth:`pyhrf.tools.io.rx_copy` copies files with patterned names into a directory structure where:
    - a source is defined by a regular expressions with named groups
    - a target is defined by python format strings whose named arguments match 
      group names in the source regexp.

Example:
Folder ./raw_data contains the following files::

  AC0832_anat.nii
  AC0832_asl.nii
  AC0832_bold.nii
  PK0612_asl.nii
  PK0612_bold.nii

I want to export these files into the following directory structure:
./export/<subject>/<modality>/data.nii
where <subject> and <modality> have to be replaced by chunks extracted
from the input files

To do so, define a regexp to catch useful chunks (or tags) in input files and
also format strings that will be used to create target file names:

.. code-block:: python

       # regexp to capture values of subject and modality:
       src = '(?P<subject>[A-Z]{2}[0-9]{4})_(?P<modality>[a-zA-Z]+).nii'
       # definition of targets:
       src_folder = './raw_data/'
       dest_folder = ('./export', '{subject}', '{modality}')
       dest_basename = 'data.nii'
       # do the thing:
       rx_copy(src, src_folder, dest_basename, dest_folder):
    
Should result in the following copied files::

  ./export/AC0832/bold/data.nii
  ./export/AC0832/anat/data.nii
  ./export/AC0832/asl/data.nii 
  ./export/PK0612/bold/data.nii 
  ./export/PK0612/asl/data.nii 
   
