"""
Performs parcellation to a list of subjects

Directory structure:
- subject:
    - preprocessed_data --> GM+WM mask, functional data, normalised tissue masks
    - t_maps --> T-maps previously computed with GLM (nipy, SALMA)
    - parcellation --> Output
"""

import os
import os.path as op
import numpy as np
from glob import glob


def make_parcellation(subject, dest_dir='parcellation', roi_mask_file=None):
    """
    Perform a functional parcellation from input fmri data
    
    Return: parcellation file name (str)
    """
    # Loading names for folders and files
    # - T maps (input)
    func_files = glob(op.join(op.join(op.join('./', subject), \
                        't_maps'), 'BOLD*nii'))
    # - Mask (input)
    #spm_mask_file = op.join(spm_maps_dir, 'mask.img')
    mask_file = op.join(op.join(op.join('./', subject), \
                        'preprocessed_data'), 'cut_tissue_mask.nii')
    # - parcellation (output)
    parcellation_dir = op.join('./', subject, dest_dir)
    if not op.exists(parcellation_dir): os.makedirs(parcellation_dir)
    pfile = op.join(parcellation_dir, 'parcellation_func.nii')


    # Parcellation
    from pyhrf.parcellation import make_parcellation_from_files 
    make_parcellation_from_files(func_files, mask_file, pfile, 
                                 nparcels=200, method='ward_and_gkm')


    # Masking with a ROI so we just consider parcels inside 
    # a certain area of the brain
    if roi_mask_file is not None:
        print 'Masking parcellation with roi_mask_file: ', roi_mask_file
        pfile_masked = op.join(parcellation_dir, 'parcellation_func_masked.nii')

        from pyhrf.ndarray import xndarray
        parcellation = xndarray.load(pfile)
        m = xndarray.load(roi_mask_file)
        parcels_to_keep = np.unique(parcellation.data * m.data)
        masked_parcellation = xndarray.xndarray_like(parcellation)
        for ip in parcels_to_keep:
            masked_parcellation.data[np.where(parcellation.data==ip)] = ip
        masked_parcellation.save(pfile_masked)

    return pfile


if __name__ == '__main__':
    
    subjects = ['RG130377', 'CD110147']

    for subject in subjects:
        
        print 'Treating subject:', subject
        pfile = make_parcellation(subject)

        print 'parcellation result in file ', pfile

