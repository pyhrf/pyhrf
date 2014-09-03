"""
Script to run a JDE ASL analysis on the data from the block motor protocol acquired at Rennes in 2010.
Preprocessings and GLM analysis must have been run prior to this script.

The script can run tests or real data analysis, such as:
    $ python jde_asl.py test #will run all the doctests
    $ python jde_asl.py test my_func #will run the doctest of my_func
    $ python jde_asl.py # will run the real data analysis

Input: GIN data structure :
 - archives
      |- <subject>
           |- ASLf
              |- funct
                  |- normalise #contains normalized & realigned time series
           |- anat #contains noramlized anatomy
                        
Output: results are stored within this structure:
 - archives
      |- <subject>
           |- ASLf
              |- funct
                  |- jde_analysis

#TODO: get t-map like for perf & bold
#compute var from samples that correspond to MAP label
"""
import sys
import os.path as op
import os
from glob import glob
import numpy as np
from scipy import ndimage

import pyhrf
from pyhrf.core import FmriData
from pyhrf.ndarray import xndarray, stack_cuboids
from pyhrf.tools.io import concat3DVols
from pyhrf.tools import add_suffix

import pyhrf.jde.asl as jasl
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.treatment import FMRITreatment

def main():
    do_analyses = True
    do_result_packing = False

    pyhrf.verbose.set_verbosity(1)
    # subjects = ['AM', 'EP', 'AS', 'HR', 'IR', 'JLJ', 'LY', 'MB', 
    #             'MH', 'OM', 'SH', 'ZK', 'TG']
    subjects = ['OM'] #works well (JDE and SPM)

    if do_analyses:
        # Run analyses (parcellation + JDE):
        for subject in subjects:
            print 'Treating subject:', subject
            # set to None for full brain analysis:
            mask_file = None #get_motor_mask_fn() 
            pfile = make_parcellation(subject, mask_file=mask_file)
            jde_analysis(get_fmri_data(subject, pfile, smoothed=True), 
                         get_output_dir(subject, 'jde_analysis'),
                         nb_iterations=2000, parallel='local')

    if do_result_packing: # Stack results of all subjects
        output_names = ['prf_pm','prl_pm','brf_pm', 'brl_pm']
        for on in output_names:
            outputs = [get_jde_output(subj, on) for subj in subjects]
            stack_cuboids(outputs,'subject',subjects).save(get_group_output(on))
            

def jde_analysis(fdata, output_dir, nb_iterations, osf=4., parallel=None):
    """

    Example:
    >>> import shutil
    >>> jde_analysis(get_fmri_data('AM'), get_output_dir('AM', 'doctest'), \
                     nb_iterations=3)
    >>> op.exists('./gin_struct/archives/AM/ASLf/'\
                    'doctest/jde_asl_mcmc_prf_pm.nii')
    True
    >>> shutil.rmtree('./gin_struct/archives/AM/ASLf/doctest')
    """
    params = {
            'nb_iterations' : nb_iterations,
            'smpl_hist_pace' : -1,
            'obs_hist_pace' : -1,
            'brf' :  jasl.BOLDResponseSampler(do_sampling=True,
                                              normalise=1.,
                                              zero_constraint=True),
            'brf_var' :  \
                jasl.BOLDResponseVarianceSampler(do_sampling=False,
                                                 val_ini=np.array([.1])),
            'prf' :  jasl.PerfResponseSampler(do_sampling=True,
                                              normalise=1.,
                                              zero_constraint=True),
            'prf_var' :  \
                jasl.PerfResponseVarianceSampler(do_sampling=False,
                                                 val_ini=np.array([.1])),
            'bold_response_levels' : \
                jasl.BOLDResponseLevelSampler(do_sampling=True),
            'perf_response_levels' : \
                jasl.PerfResponseLevelSampler(do_sampling=True),
        }

    sampler = jasl.ASLSampler(**params)

    analyser = JDEMCMCAnalyser(sampler=sampler, dt=fdata.tr/osf,  driftParam=4, 
                               driftType='polynomial', 
                               outputPrefix='jde_asl_mcmc_', pass_error=False)

    treatment = FMRITreatment(fmri_data=fdata, analyser=analyser,
                              output_dir=output_dir)

    treatment.run(parallel=parallel)


def make_parcellation(subject, dest_dir='parcellation', force=False,
                      mask_file=None):
    """
    Perform a functional parcellation from input fmri data
    
    Args:
        - fdata (pyhrf.core.FmriData): masked functional data

    Rerturn: path to the parcellation file (str)
    Example:
    >>> import shutil
    >>> from pyhrf.ndarray import xndarray
    >>> make_parcellation('TG', dest_dir='parc_doctest')
    './gin_struct/archives/TG/ASLf/parc_doctest/parcellation_func.nii'
    >>> xndarray.load(make_parcellation('TG')).data.ndim == 3
    True
    >>> make_parcellation('TG', dest_dir='parc_doctest', \
                          mask_file=get_motor_mask_fn())
    './gin_struct/archives/TG/ASLf/parc_doctest/parcellation_func_masked.nii'
    >>> shutil.rmtree('./gin_struct/archives/TG/ASLf/parc_doctest/')
    """
    pfile = op.join(get_output_dir(subject, dest_dir), 'parcellation_func.nii')
    if force or not op.exists(pfile):
        from pyhrf.parcellation import make_parcellation_from_files
        spm_dir = op.join(get_subj_dir(subject), 'SPMTop_block')
        func_files = glob(op.join(spm_dir, 'spmT*img'))
        spm_mask_file = op.join(spm_dir, 'mask.img')
        make_parcellation_from_files(func_files, spm_mask_file, pfile, 
                                     nparcels=200, method='ward_and_gkm')
    if mask_file is not None:
        # see function get_motor_mask_fn() to build a motor mask
        parcellation = xndarray.load(pfile)
        m = xndarray.load(mask_file)
        parcels_to_keep = np.unique(parcellation.data * m.data)
        masked_parcellation = xndarray.xndarray_like(parcellation)
        for ip in parcels_to_keep:
            masked_parcellation.data[np.where(parcellation.data==ip)] = ip
        pfile = add_suffix(pfile, '_masked')
        masked_parcellation.save(pfile)

    return pfile


def get_motor_mask_fn():
    """
    Return the path to the roi motor mask built by thresholding
    SPM contrast results.
    
    Example:
    >>> from pyhrf.ndarray import xndarray
    >>> m = xndarray.load(get_motor_mask_fn())
    >>> m.data.shape #should be normalized shape
    (61, 73, 61)
    >>> m.sum() #total nb of positions in the mask
    1356
    """
    output_fn = op.join(get_common_dir(), 'mask_roi_motor_from_spmT.nii')
    if not op.exists(output_fn):
        ref_subj = 'AM'
        tmap = xndarray.load(op.join(get_subj_dir(ref_subj), 'SPMTop_block', 
                                     'spmT_0002.img')).squeeze()
        # Threshold map:
        mdata = tmap.data>7 

        # Increase size of ROI:
        for i in range(5):
            mdata = ndimage.binary_dilation(mdata)
        
        # Make output:
        mask = xndarray.xndarray_like(tmap, data=mdata.astype(np.int32))    
        mask.save(output_fn)

    return output_fn

def get_common_dir():
    d = op.join('./gin_struct', 'common')
    if not op.exists(d):
        os.makedirs(d)
    return d


def get_fmri_data(subject, parcellation_file=None, smoothed=False):
    """
    Load functional data from files stored in the GIN hiearchy
    
    Args:
        - subject (str): subject's code name
        - parcellation_file (str|None): 
               path to the parcellation file.
               If None, take mask from GLM analysis.
    Return: functional data (pyhrf.core.FmriData)
    Example:
    >>> fd = get_fmri_data('AM')
    >>> len(fd.paradigm.stimOnsets['motorRight'][0]) #nb of stimulations
    6
    >>> fd.bold.shape[0] #nb of scans
    142
    >>> fd.tr
    3
    """
    subj_dir = get_subj_dir(subject)
    pfile = parcellation_file or op.join(subj_dir, 'SPMTop_block', 'mask.img')
    paradigm_file = './paradigm.csv'
    func_file = get_4D_fdata_file(subject, smoothed)
    return FmriData.from_vol_files(pfile, paradigm_file, [func_file], tr=3,
                                   paradigm_csv_delim=',')

def get_4D_fdata_file(subject, smoothed=False):
    """
    Example:
    >>> from pyhrf.ndarray import xndarray
    >>> get_4D_fdata_file('AM')
    './gin_struct/archives/AM/ASLf/funct/normalise/wrAM_ASLf_correctionT1.nii'
    >>> xndarray.load('./gin_struct/archives/AM/ASLf/funct/' \
                      'normalise/wrAM_ASLf_correctionT1.nii').data.shape
    (61, 73, 61, 142)
    >>> get_4D_fdata_file('AM', smoothed=True)
    './gin_struct/archives/AM/ASLf/funct/smooth/swrAM_ASLf_correctionT1.nii'
    >>> op.exists(get_4D_fdata_file('AM', smoothed=True))
    True
    """
    if smoothed:
        subd = 'smooth'
        prefix = 'swr'
    else:
        subd = 'normalise'
        prefix = 'wr'
    data4D_fn = op.join(get_subj_dir(subject), 'funct', subd, 
                        '%s%s_ASLf_correctionT1.nii'%(prefix, subject))
    if not op.exists(data4D_fn):
        concat3DVols(sorted(glob(add_suffix(data4D_fn, "_*"))), data4D_fn)
    return data4D_fn

def get_output_dir(subject, subdir):    
    """
    Example:
    >>> get_output_dir('AM', 'output_doctest')
    './gin_struct/archives/AM/ASLf/output_doctest'
    >>> os.rmdir('./gin_struct/archives/AM/ASLf/output_doctest')
    """
    d = op.join(get_subj_dir(subject), subdir)
    if not op.exists(d):
        os.makedirs(d)
    return d

def get_subj_dir(subject):
    """
    Example:
    >>> get_subj_dir('AM')
    './gin_struct/archives/AM/ASLf'
    """
    d = op.join('./gin_struct', 'archives', 
                '{subject}','ASLf').format(subject=subject)
    if not op.exists(d):
        msg = 'Data dir for subject {subject} '\
              'does exists {path}'.format(subject=subject, path=d)
        raise Exception(msg)                   
    return d

def get_jde_output(subject, output_name, jde_dir='jde_analysis'):
    """
    Return a JDE output for a specific that is stackable with other outputs
    of other subjects. Especially, map parcel-specific RFs onto voxel-specific
    RFs.
    Example:
    >>> import shutil
    >>> parc_file = op.join('gin_struct','archives','parcellation_test.nii')
    >>> jde_analysis(get_fmri_data('AM', parc_file),  \
                     get_output_dir('AM', 'jde_analysis_doctest'), \
                     nb_iterations=3, parallel='local', osf=4.) 
    >>> get_jde_output('AM','prl_pm',jde_dir='jde_analysis_doctest').data.shape
    (61, 73, 61, 1)
    >>> get_jde_output('AM','prf_pm',jde_dir='jde_analysis_doctest').data.shape
    (61, 73, 61, 35)
    >>> shutil.rmtree(get_output_dir('AM', 'jde_analysis_doctest'))
    """
    o = xndarray.load(op.join(get_output_dir(subject, jde_dir), 
                              'jde_asl_mcmc_%s.nii' %output_name))
    if not o.has_axes(['axial', 'sagittal', 'coronal']):
        o = o.map_onto(get_jde_output(subject, 'roi_mapping', jde_dir))
    return o

def get_group_output(output_name):
    d = op.join('./gin_struct', 'archives', 'group_results')
    if not op.exists(d):
        os.makedirs(d)
    return op.join(d, '%s.nii'%output_name)
     
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        import doctest
        if len(sys.argv) > 2: #test specific doctests
            for e in sys.argv[2:]: 
                doctest.run_docstring_examples(eval(e), globals())
        else:
            doctest.testmod()
    else:
        main()
