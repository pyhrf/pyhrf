"""
Import data obtained from a Shanoir query into a hierarchical folder structure.

This script should be called in the targer data directory
"""
import os.path as op
import pyhrf
from pyhrf.tools.io import rx_copy

data_dir = './'
shanoir_dir = op.join(data_dir, 'FromShanoir')

# regexp to catch source files and define tags:
rx_nii = r'ASL[f]* mt (?P<subject>[A-Z]+)\ *\^(?P<modality>3DMPRAGE|PASLFONCTTR38COUPES7MM)s(?P<session>[0-9]{3})a[0-9]{3,4}.nii'
src_folder = shanoir_dir
# define targets (tag between brackets {} will be replaced with group values 
#                 caught bby rx_nii)
dest_folder = (data_dir, '{subject}', '{modality}')
dest_basename = '{modality}_session_{session}.nii'

# string replacements in output file names:
replacements = [('PASLFONCTTR38COUPES7MM', 'aslf'),
                ('3DMPRAGE' , 'anat')]

if 0: # test the importation
    pyhrf.verbose.set_verbosity(4)
    rx_copy(rx_nii, src_folder, dest_basename, dest_folder, dry=True, 
            replacements=replacements)
else: # do the importation
    rx_copy(rx_nii, src_folder, dest_basename, dest_folder, dry=False, 
            replacements=replacements)
