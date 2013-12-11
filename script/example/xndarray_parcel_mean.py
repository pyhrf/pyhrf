"""
Compute the mean of BOLD signal within parcels.

This is an example of several operations for xndarray:
- explosion of data according to a parcellation mask
- mean over voxel
- merge of several xndarray objects
"""
import os.path as op
from pyhrf import get_data_file_name, get_tmp_path
from pyhrf.ndarray import xndarray, merge

func_data = xndarray.load(get_data_file_name('subj0_bold_session_0.nii.gz'))
parcellation = xndarray.load(get_data_file_name('subj0_parcellation.nii.gz'))
parcel_fdata = func_data.explode(parcellation)
parcel_means = [d.copy().fill(d.mean('position')) for d in parcel_fdata]
parcel_means = merge(parcel_means, parcellation, axis='position')
output_fn = op.join(get_tmp_path(), './subj0_bold_parcel_means.nii')
print 'File saved to:', output_fn
parcel_means.save(output_fn)
