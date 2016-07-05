#!/usr/bin/env python2
# -*- coding: utf-8 -*-


############################## Model Definition ############################
tr = 1.  # Repetition time in seconds
dt = None  # Time resolution of the hrf, if None, choose the nearest
           # integer part of tr to 1 second
           # /!\ WARNING /!\ : dt MUST be an integer part of tr !!!
hrf_duration = 25.  # Total time of the HRF in seconds

bold_data_file = [
    "./bold.nii",  # path to the Nifti 4D bold file(s)
]                  # /!\ MUST be a list of path(s) (for later multisession
                   # implementation)
parcels_file = "./parcels.nii"  # path to the integer 3D Nifti parcellation file
#  spm_file = None  # NOT YET IMPLEMENTED If not None, path to the SPM.mat file that configure the
                 #  # GLM analysis else you need (at least) to give the csv
                 #  # file of the # onsets and possibly the contrasts definition
onsets_file = "./onsets.csv"  # csv file of the onsets ignored if spm_file is given
def_contrasts_file = None  # json file of the contrasts definition
                           # ignored if spm_file if given and
                           # contains contrasts definition
                           # /!\ Be aware that all conditions in
                           # contrasts definitions MUST match
                           # (with case sensitive) the conditions
                           # defined in the onsets CSV file

output_dir = "."  # path for output files

################### Pyhrf parameters (advanced user only) ##################

save_processing_config = True

# VEM parameters
sigma_h = 0.1
nb_iter_max = 100
nb_iter_min = 5
beta = 1.0
estimate_hrf = True
hrf_hyperprior = 1000
zero_constraint = True
drifts_type = "cos"  # "cos" or "poly"

# Interface parameters
log_level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
parallel = True  # If True, use local multiprocessing

##################### DO NOT EDIT BELOW THIS LINE !!! #########################


import os
import argparse
import sys
import json
import time
import datetime

from ConfigParser import _Chainmap as ChainMap

import numpy as np

import pyhrf

from pyhrf.core import FmriData
from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
from pyhrf.ui.treatment import FMRITreatment
from pyhrf.tools import format_duration


description = """Run an fMRI analysis using the JDE-VEM framework


All arguments (even positional ones) are optional and default values are defined
in the beginning of the script"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("parcels_file", nargs="?", default=argparse.SUPPRESS,
                    help="Nifti integer file which define the parcels")
parser.add_argument("onsets_file", nargs="?", default=argparse.SUPPRESS,
                    help="CSV onsets file")
parser.add_argument("bold_data_file", nargs="*", default=argparse.SUPPRESS,
                    help="list of Nifti file(s) of bold data")
parser.add_argument("-c", "--def-contrasts-file", nargs="?", default=argparse.SUPPRESS,
                    help="JSON file defining the contrasts")
parser.add_argument("-t", "--tr", default=argparse.SUPPRESS, type=float,
                    help="Repetition time of the fMRI data")
parser.add_argument("-d", "--dt", default=argparse.SUPPRESS, type=float,
                    help=("time resolution of the HRF (if not defined here or"
                          " in the script, it is automatically computed)"))
parser.add_argument("-l", "--hrf-duration", default=argparse.SUPPRESS, type=float,
                    help="time lenght of the HRF (in seconds)")
parser.add_argument("-o", "--output", dest="output_dir", default=argparse.SUPPRESS,
                    help="output directory (created if needed)")
#  parser.add_argument("-s", "--spm", dest="spm_file", default=argparse.SUPPRESS)
parser.add_argument("--nb-iter-max", default=argparse.SUPPRESS, type=int,
                    help="maximum number of iterations of the VEM algorithm")
parser.add_argument("--nb-iter-min", default=argparse.SUPPRESS, type=int,
                    help="minimum number of iterations of the VEM algorithm")
parser.add_argument("--beta", default=argparse.SUPPRESS, type=float)
parser.add_argument("--hrf-hyperprior", default=argparse.SUPPRESS, type=int)
parser.add_argument("--sigma-h", default=argparse.SUPPRESS, type=float)
parser.add_argument("--estimate-hrf", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--no-estimate-hrf", action="store_false", dest="estimate_hrf",
                    help="explicitly disable HRFs estimation", default=argparse.SUPPRESS)
parser.add_argument("--zero-constraint", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--no-zero-constraint", action="store_false", dest="zero_constraint",
                    help="explicitly disable zero constraint (default enabled)",
                    default=argparse.SUPPRESS)
parser.add_argument("--drifts-type", default=argparse.SUPPRESS, type=str)
parser.add_argument("-v", "--log-level", default=argparse.SUPPRESS,
                    choices=("DEBUG", "10", "INFO", "20", "WARNING", "30",
                             "ERROR", "40", "CRITICAL", "50"))
parser.add_argument("-p", "--parallel", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--no-parallel", action="store_false", dest="parallel",
                    help="explicitly disable parallel computation", default=argparse.SUPPRESS)
parser.add_argument("--save-processing", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--no-save-processing", action="store_false", dest="save_processing_config")
args = parser.parse_args()

local_config = {
    "tr": tr,
    "dt": dt,
    "hrf_duration": hrf_duration,
    "bold_data_file": bold_data_file,
    "parcels_file": parcels_file,
    "spm_file": spm_file,
    "onsets_file": onsets_file,
    "def_contrasts_file": def_contrasts_file,
    "output_dir": output_dir,
    "log_level": log_level,
    "parallel": parallel,
    "sigma_h": sigma_h,
    "nb_iter_max": nb_iter_max,
    "nb_iter_min": nb_iter_min,
    "beta": beta,
    "estimate_hrf": estimate_hrf,
    "hrf_hyperprior": hrf_hyperprior,
    "zero_constraint": zero_constraint,
    "save_processing_config": save_processing_config,
    "drifts_type": drifts_type,
}

del (tr, dt, hrf_duration, bold_data_file, parcels_file, spm_file, onsets_file,
     def_contrasts_file, output_dir, log_level, parallel, sigma_h, nb_iter_max,
     nb_iter_min, beta, estimate_hrf, hrf_hyperprior, zero_constraint,
     drifts_type)

command_config = vars(args)

# If parameters are defined by command line, overwrite ones from above by using
# a ChainMap (undocumented in python2, see python3 documentation) object

config = ChainMap({}, command_config, local_config)
if not config["dt"]:
    tr = config["tr"]
    possible_dts = tr/np.arange(int(tr)-1, int(tr)+2)
    local_config["dt"] = possible_dts[np.abs(possible_dts-1.).argmin()]

try:
    pyhrf.logger.setLevel(config["log_level"])
except ValueError:
    try:
        pyhrf.logger.setLevel(int(config["log_level"]))
    except ValueError:
        print("Can't set log level to {}".format(config["log_level"]))
        sys.exit(1)


def load_contrasts_definitions(contrasts_file):
    """Loads contrasts from a json file defining the contrasts with linear
    combinations of conditions.

    Parameters
    ----------
    contrasts_file : str
        the path to the json file
    Returns
    -------
    compute_contrasts : bool
        if everything gone well, tell the JDEVEMAnalyser to compute the contrasts
    contrasts_def : dict
        each key is a contrast and each corresponding value is the contrast
        definition

    """

    try:
        with open(contrasts_file) as contrasts_file:
            contrasts_def = json.load(contrasts_file)
        compute_contrasts = bool(contrasts_def)
    except (IOError, TypeError):
        compute_contrasts = False
        contrasts_def = None

    return compute_contrasts, contrasts_def


def main():
    """Run when calling the script"""

    start_time = time.time()

    if not os.path.isdir(config["output_dir"]):
        try:
            os.makedirs(config["output_dir"])
        except OSError as e:
            print("Ouput directory could not be created.\n"
                  "Error was: {}".format(e.strerror))
            sys.exit(1)

    bold_data = FmriData.from_vol_files(
        mask_file=config["parcels_file"], paradigm_csv_file=config["onsets_file"],
        bold_files=config["bold_data_file"], tr=config["tr"]
    )

    compute_contrasts, contrasts_def = load_contrasts_definitions(config["def_contrasts_file"])

    jde_vem_analyser = JDEVEMAnalyser(
        hrfDuration=config["hrf_duration"], sigmaH=config["sigma_h"], fast=True,
        computeContrast=compute_contrasts, nbClasses=2, PLOT=False,
        nItMax=config["nb_iter_max"], nItMin=config["nb_iter_min"], scale=False,
        beta=config["beta"], estimateSigmaH=True, estimateHRF=config["estimate_hrf"],
        TrueHrfFlag=False, HrfFilename='hrf.nii', estimateDrifts=True,
        hyper_prior_sigma_H=config["hrf_hyperprior"], dt=config["dt"], estimateBeta=True,
        contrasts=contrasts_def, simulation=False, estimateLabels=True,
        LabelsFilename=None, MFapprox=False, estimateMixtParam=True,
        constrained=False, InitVar=0.5, InitMean=2.0, MiniVemFlag=False, NbItMiniVem=5,
        zero_constraint=config["zero_constraint"], drifts_type=config["drifts_type"]
    )

    processing_jde_vem = FMRITreatment(
        fmri_data=bold_data, analyser=jde_vem_analyser,
        output_dir=config["output_dir"], make_outputs=True
    )

    if not config["parallel"]:
        processing_jde_vem.run()
    else:
        processing_jde_vem.run(parallel="local")


    if config["save_processing_config"]:
        # Let's canonicalize all paths
        config_save = dict(config)
        for file_nb, bold_file in enumerate(config_save["bold_data_file"]):
            config_save["bold_data_file"][file_nb] = os.path.abspath(bold_file)
        config_save["parcels_file"] = os.path.abspath(config_save["parcels_file"])
        config_save["onsets_file"] = os.path.abspath(config_save["onsets_file"])
        if config_save["def_contrasts_file"]:
            config_save["def_contrasts_file"] = os.path.abspath(config_save["def_contrasts_file"])
        config_save["output_dir"] = os.path.abspath(config_save["output_dir"])
        config_save_filename = "{}_processing.json".format(
            datetime.datetime.today()
        ).replace(" ", "_")
        config_save_path = os.path.join(config["output_dir"], config_save_filename)
        with open(config_save_path, 'w') as json_file:
            json.dump(config_save, json_file, sort_keys=True, indent=4)

    print("")
    print("Total computation took: {} seconds".format(format_duration(time.time() - start_time)))

if __name__ == "__main__":
    main()
