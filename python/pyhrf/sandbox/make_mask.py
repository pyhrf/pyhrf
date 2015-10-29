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
                                'spmT*img'))
    # - Mask (input)
    #spm_mask_file = op.join(spm_maps_dir, 'mask.img')
    mask_dir = op.join('./', subject, 'preprocessed_data')
    if not op.exists(mask_dir): os.makedirs(mask_dir)
    mask_file = op.join(mask_dir, 'cut_tissue_mask2.nii')
    gm_file = op.join(op.join(op.join('./', subject), \
                        'anat'), 'c1'+subject+'_anat-0001.nii')
    wm_file = op.join(op.join(op.join('./', subject), \
                        'anat'), 'c2'+subject+'_anat-0001.nii')
    csf_file = op.join(op.join(op.join('./', subject), \
                        'anat'), 'c3'+subject+'_anat-0001.nii')
    make_tissue_mask(gm_file, wm_file, csf_file, mask_file)
    #make_tissue_mask2(func_files, mask_file)
    return mask_file
    

def make_tissue_mask(gm_file, wm_file, csf_file, mask_file):
    from pyhrf.ndarray import xndarray
    gm = xndarray.load(gm_file)
    wm = xndarray.load(gm_file)
    csf = xndarray.load(gm_file)
    
    mask = csf.copy()
    mask.data = np.zeros_like(csf.data)
    mask.data[np.where(gm.data+wm.data > 0.0000000000001)] = 1
    mask.save(mask_file)
    return 

def make_tissue_mask2(func_files, mask_file):
    from pyhrf.ndarray import xndarray
    for i, ifile in enumerate(func_files):
        ff = xndarray.load(ifile)
        if i==0:
            f = ff.data 
        else:
            f += ff.data
    
    mask = ff.copy()
    mask.data = np.zeros_like(ff.data)
    mask.data[np.where(ff.data > 0.)] = 1
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

