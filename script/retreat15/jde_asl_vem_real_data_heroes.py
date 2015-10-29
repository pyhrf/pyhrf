#import sys
import os
import os.path as op
import numpy as np

import pyhrf
import pyhrf.paradigm
from pyhrf.core import FmriData
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.vb_jde_analyser_asl_fast import JDEVEMAnalyser
from pyhrf.ndarray import xndarray
from pyhrf import logging

from pyhrf.jde.asl_2steps import jde_analyse_2steps_v1
from pyhrf.sandbox.physio_params import PHY_PARAMS_FRISTON00, PHY_PARAMS_DONNET06
from pyhrf.sandbox.physio_params import PHY_PARAMS_DENEUX06, PHY_PARAMS_HAVLICEK11
from pyhrf.sandbox.physio_params import PHY_PARAMS_KHALIDOV11, PHY_PARAMS_DUARTE12


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():

    #np.random.seed(np.random.randint(0, 1000000))
    np.random.seed(48258)

    te = 0.011
    tr = 2.5
    dt = tr / 2.
    n_scans = 164
    frametimes = np.arange(0, n_scans * tr, tr)

    phy_params = PHY_PARAMS_KHALIDOV11
    phy_params['TE'] = te

    archives = './Data/ASLf_analysis_GIN/archives'
    subjects = ['RG130377']
    prior_types = np.array(['omega', 'balloon', 'no'])
    prior = prior_types[2]

    for subject in subjects:

        print 'Subject ', subject        
        output_dir = op.join('./', subject, 'jde_results')
        if not op.exists(output_dir): os.makedirs(output_dir)

        # Load data for condition cond and subject subject
        data_dir = op.join(archives, subject)
        fdata = load_data(archives, data_dir, subject, tr)
        vem_output_dir = op.join(output_dir, 'vem_jde_analysis')

        print 'JDE VEM analysis on real data ...'
        jde_analyse_vem(vem_output_dir, fdata, dt=dt, nItMin=2,
                        use_hyperprior=True, positivity=False,
                        phy_params=phy_params, prior=prior)
        print 'JDE VEM analysis on real data done!'

        del fdata


def bold_mean_and_range(bold_fn, gm_fn):
    """
    Returns signal mean and range
    """
    #gm = xndarray.load(gm_fn).data
    #print gm.shape
    bold = xndarray.load(bold_fn).data
    print 'BOLD shape ', bold.shape
    bold_mean = np.mean(bold) #[np.where(gm > 0)])
    bold_range = (np.max(bold) - np.min(bold))
    del bold #, gm
    print 'BOLD mean ', bold_mean
    print 'BOLD range ', bold_range
    return bold_mean, bold_range


def load_data(archives, data_dir, subject, tr):
    """
    Load data and create 
    """
    # Folder and file names
    data_fn = op.join(data_dir, 'ASLf', 'funct', 'smooth', \
                      'swr' + subject + '_ASLf_correctionT1.nii')
    gm_fn = op.join(data_dir, 'anat', 'c1' + subject + '_anat-0001.nii')
    parcel_dir = op.join(data_dir, 'ASLf', 'parcellation')

    paradigm_fn = op.join(archives, 'paradigm_bilateral_v2_no_final_rest.csv')
    #roi_mask_fn = op.join(parcel_dir, 'parcellation_func.nii')
    roi_mask_fn = op.join(parcel_dir, 'parcellation_to_analyse.nii')

    # Loading and scaling data
    data_mean, data_range = bold_mean_and_range(data_fn, gm_fn)            
    fdata = FmriData.from_vol_files(roi_mask_fn, paradigm_fn, [data_fn], tr)
    fdata.bold = (fdata.bold - data_mean) * 100 / data_range    
    print 'mean bold = ', np.mean(fdata.bold)
    print 'shape bold = ', fdata.bold.shape

    return fdata


def jde_analyse_vem(output_dir, fmri_data, dt=0.5, physio=True, nItMin=nItMin,
                    use_hyperprior=False, positivity=False,
                    phy_params=PHY_PARAMS_KHALIDOV11, prior='no'):
    """
    Runs JDE VEM sampler
    """
    if not op.exists(output_dir): os.makedirs(output_dir)
    contrasts = {"audio-video": "audio-video",
                 "video-audio": "video-audio"}
    vmu = 100.
    vh = 0.001 #0.0001
    vg = 0.001 #0.0001
    gamma_h = 1000000000  # 10000000000  # 7.5 #100000
    gamma_g = 1000000000                  #10000000
    jde_vem_analyser = JDEVEMAnalyser(beta=1., dt=dt, hrfDuration=25.,
                            nItMax=50, nItMin=nItMin, PLOT=True,
                            sigmaH=vh, gammaH=gamma_h, sigmaMu=vmu,
                            sigmaG=vg, gammaG=gamma_g, physio=physio,
                            positivity=positivity, use_hyperprior=use_hyperprior, 
                            prior=prior, contrasts=contrasts)
    # - sigmaM = 10.
    # - sigmaH=0.00000001
    # - sigmaG=0.0001
    tjde_vem = FMRITreatment(fmri_data=fmri_data, analyser=jde_vem_analyser,
                             output_dir=output_dir)
    tjde_vem.run() #parallel='local', n_jobs=16)
    return


if __name__ == '__main__':
    main()
