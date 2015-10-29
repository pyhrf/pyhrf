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
    #func_files = glob(op.join(op.join(op.join('./', subject), \
    #                    't_maps'), 'BOLD*nii'))
    func_files = glob(op.join('./', subject, 'ASLf', 'spm_analysis', \
                                'Tmaps*img'))
    # - Mask (input)
    #spm_mask_file = op.join(spm_maps_dir, 'mask.img')
    mask_dir = op.join('./', subject, 'preprocessed_data')
    if not op.exists(mask_dir): os.makedirs(mask_dir)
    mask_file = op.join(mask_dir, 'mask.nii')
    mask = op.join(mask_dir, 'rcut_tissue_mask.nii')
    volume = op.join('./', subject, 'ASLf', 'funct', 'coregister', \
                   'mean'+subject+'_ASLf_correctionT1_0001.nii')
    make_mask(mask, volume, mask_file)

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
    if 1: #roi_mask_file is not None:
        print 'Masking parcellation with roi_mask_file: ', roi_mask_file
        pfile_masked = op.join(parcellation_dir, 'parcellation_func_masked_roi200.nii')

        from pyhrf.ndarray import xndarray
        parcellation = xndarray.load(pfile)
        #m = xndarray.load(roi_mask_file)
        #parcels_to_keep = np.unique(parcellation.data * m.data)
        masked_parcellation = xndarray.xndarray_like(parcellation)
        #for ip in parcels_to_keep:
        #    masked_parcellation.data[np.where(parcellation.data==ip)] = ip
        ip = 200
        masked_parcellation.data[np.where(parcellation.data==ip)] = ip
        masked_parcellation.save(pfile_masked)

    return pfile


def make_mask(mask, volume, mask_file):
    from pyhrf.ndarray import xndarray
    m = xndarray.load(mask)
    v = xndarray.load(volume)
    
    mask = m.copy()
    #mask.data = np.zeros_like(m.data)
    mask.data[np.where(v.data == 0.)] = 0
    mask.save(mask_file)
    return 


if __name__ == '__main__':
    
    #subjects = ['RG130377', 'CD110147']
    subjects = ['AINSI_001_GC', 'AINSI_005_SB', 'AINSI_010_TV']
    #subjects = ['AINSI_010_TV']

    for subject in subjects:
        
        print 'Treating subject:', subject
        pfile = make_parcellation(subject)

        print 'parcellation result in file ', pfile

