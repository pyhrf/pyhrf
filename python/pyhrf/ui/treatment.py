# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import os
import os.path as op
from os.path import splitext
import sys
import shutil
import gzip
#import signal
import time
#import copy as copyModule
from copy import deepcopy
import string
import cPickle
import numpy as np
import itertools
from optparse import OptionParser
import cProfile
from pyhrf.tools import format_duration
from pyhrf.tools.io import remote_copy, load_paradigm_from_csv
from pyhrf.tools.io.spmio import load_paradigm_from_mat
import tempfile

import pyhrf
from pyhrf.configuration import cfg
from pyhrf.core import FmriData, FMRISessionVolumicData, \
    FMRISessionSurfacicData, FMRISessionSimulationData

from pyhrf import xmlio, DEFAULT_ONSETS, DEFAULT_STIM_DURATIONS, \
    DEFAULT_BOLD_VOL_FILE, DEFAULT_MASK_VOL_FILE, DEFAULT_BOLD_SURF_FILE, \
    DEFAULT_MESH_FILE, DEFAULT_MASK_SURF_FILE, DEFAULT_OUT_MASK_VOL_FILE, \
    DEFAULT_OUT_MASK_SURF_FILE, DEFAULT_MASK_SMALL_VOL_FILE, \
    REALISTIC_REAL_DATA_BOLD_VOL_FILE, REALISTIC_REAL_DATA_MASK_VOL_FILE, \
    DEFAULT_PARADIGM_CSV

from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
from pyhrf.xmlio import XMLable2
from pyhrf.verbose import dictToString
from pyhrf.tools import unstack_trees #stack_trees
from pyhrf.tools.io import *
from pyhrf.tools.io._zip import *
from pyhrf.ui.jde import JDEMCMCAnalyser

# Setup default parameters for a treatment:
if cfg['treatment-default']['save_result_dump']:
    DEFAULT_DUMP_FILE = './result.pck'
else:
    DEFAULT_DUMP_FILE = None

MIN_NB_VOX_IN_ROI = 10

def exec_t(t):
    return t.execute()



class FMRITreatment(XMLable2):


    parametersComments = {
        'fmri_data' : 'FMRI data definition',
        'output_dir' : "Output directory where to store analysis results",
        'analyser' : \
            'Define parameters of the analysis which will be applied to '\
            ' the previously defined data',
        'result_dump_file' : 'File to save the analyser result (uses pickle).',
        'make_outputs' : 'Make outputs from analysis results',
        }

    xmlComment = "Group all parameters for a within-subject analysis."\
        "\nTwo main parts:\n - data definition ('fmri_data')\n"\
        " - analysis parameters ('analyser')."

    parametersToShow = ['fmri_data', 'result_dump_file',
                        'output_dir', 'make_outputs', 'analyser']

    def __init__(self, fmri_data=FmriData.from_vol_ui(),
                 analyser=JDEMCMCAnalyser(), output_dir='./',
                 make_outputs=True, result_dump_file=DEFAULT_DUMP_FILE):

        XMLable2.__init__(self)

        self.analyser = analyser
        self.output_dir = output_dir

        if result_dump_file is None:
            if self.output_dir is not None and DEFAULT_DUMP_FILE is not None:
                self.result_dump_file = op.join(self.output_dir,
                                                DEFAULT_DUMP_FILE)
            else:
                self.result_dump_file = None
        else:
            self.result_dump_file = op.join(self.output_dir, result_dump_file)

        self.data = fmri_data

        #self.result_dump_file = op.join(self.output_dir, 'result.pck')


        #print 'test:', op.join(self.output_dir, 'result.pck')
        #print 'self.result_dump_file:', self.result_dump_file
        self.make_outputs = make_outputs

    def enable_draft_testing(self):
        self.analyser.enable_draft_testing()
        self.output_dir = None

    def dump_roi_datasets(self, dry=False, output_dir=None):
        pyhrf.verbose(1,'Loading data ...')
        # if no file to dump (dry), assume it's only to get file names,
        # then don't build the graph (could take some time ...)
        if not dry:
            self.data.build_graphs()

        explData = self.analyser.split_data(self.data)
        files = []
        roiIds = []
        if output_dir is not None:
            roi_data_out_dir = output_dir
        else:
            if self.output_dir is not None:
                roi_data_out_dir = op.join(self.output_dir, 'ROI_datasets')
            else:
                roi_data_out_dir = op.join(pyhrf.get_tmp_path(), 'ROI_datasets')
            if not op.exists(roi_data_out_dir): os.makedirs(roi_data_out_dir)

        assert op.exists(roi_data_out_dir)

        if not dry:
            pyhrf.verbose(1,'Dump roi data in dir %s' %roi_data_out_dir)


        #data_order = sorted([d.get_nb_vox_in_mask() for d in explData])
        pyhrf.verbose(1,'Dump of roi data, ordering by size ...')
        cmp_size = lambda e1,e2:cmp(e1.get_nb_vox_in_mask(),
                                    e2.get_nb_vox_in_mask())
        for edata in sorted(explData, cmp=cmp_size, reverse=True):
            roiId = edata.get_roi_id()
            fn = op.abspath(op.join(roi_data_out_dir,
                                    "roidata_%04d.pck" %roiId))
            roiIds.append(roiId)
            if not dry:
                f = open(fn ,'w')
                cPickle.dump(edata, f)
                f.close()
            files.append(fn)

        pyhrf.verbose(1,'Dump of roi data done.')

        return files, roiIds


    def get_data_files(self):
        return self.data.get_data_files()

    def replace_data_dir(self, d):
        print self._init_parameters['fmri_data']
        print '->', self._init_parameters['fmri_data']._init_parameters

        p_fmri_data = self._init_parameters['fmri_data']._init_parameters
        for k,v in p_fmri_data.iteritems():
            if isinstance(v,str) and op.exists(v):
                print 'Replace dir for param', k
                p_fmri_data[k] = op.join(d,op.basename(v))
                print '->', p_fmri_data[k]
            elif k=='sessions_data':
                for sd in v:
                    for kk,vv in sd._init_parameters.iteritems():
                        if isinstance(vv,str) and op.exists(vv):
                            print 'Replace dir for param', kk
                            sd._init_parameters[kk] = op.join(d,op.basename(vv))
                            print '->', sd._init_parameters[kk]
            elif isinstance(v,list): #maybe list of bold files
                for iv,vv in enumerate(v):
                    if isinstance(vv,str) and op.exists(vv):
                        print 'Replace dir for param %s[%d]'%(k,iv)
                        v[iv] = op.join(d,op.basename(vv))
                        print '->', v[iv]

    def already_done(self):
        pyhrf.verbose(1,'Checking if already done (%s) ' %self.result_dump_file)
        return self.result_dump_file is not None and \
            op.exists(self.result_dump_file)

    def execute(self):

        if pyhrf.verbose.verbosity >= 2:
            pyhrf.verbose(2, 'Input data description:')
            pyhrf.verbose(2, self.data.getSummary(long=True))
        else:
            pyhrf.verbose(1, 'Input data description:')
            pyhrf.verbose(1, self.data.getSummary(long=False))

        pyhrf.verbose(1,'All data loaded !')
        pyhrf.verbose(1,'running estimation ...')
        #TODO : print summary of analyser setup.
        pyhrf.verbose(1,'Estimation start date is : %s'
                      %time.strftime('%c'))
        tIni = time.time()
        result = self.analyser.analyse(self.data, self.output_dir)

        pyhrf.verbose(1,'Estimation done, total time : %s'
                      %format_duration(time.time()-tIni))
        pyhrf.verbose(1,'End date is : '+time.strftime('%c'))

        return result


    def run(self, parallel=None, n_jobs=None):
        """
        Run the the analysis: load data, run estimation, output results
        """
        if parallel is None:
            result = self.execute()
        elif parallel == 'local':
            cfg_parallel = pyhrf.cfg['parallel-local']
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise Exception('Can not import joblib. It is required to '\
                                'enable parallel processing on a local machine.')

            parallel_verb = pyhrf.verbose.verbosity
            if pyhrf.verbose.verbosity == 6:
                parallel_verb = 10

            if n_jobs is None:
                n_jobs = cfg_parallel['nb_procs']

            p = Parallel(n_jobs=n_jobs, verbose=parallel_verb)
            result = p(delayed(exec_t)(t) for t in self.split(output_dir=None))
            # join list of lists:
            result = list(itertools.chain.from_iterable(result))

        elif parallel == 'LAN':

            from pyhrf import grid
            cfg_parallel = pyhrf.cfg['parallel-LAN']
            remoteUser = cfg_parallel['user']

            #1. Some checks on input/output directory
            remoteDir = cfg_parallel['remote_path']
            # At the end, results will be retrieved direclty from remoteDir,
            # which has to be readable
            if remoteDir is None or not op.exists(remoteDir):
                raise Exception('Remote directory is not readable (%s).' \
                                'Consider mounting it with sshfs.'
                                %remoteDir)

            # Try if remoteDir is writeable, so that we don't need to upload
            # data via ssh
            remote_writeable = False
            if os.access(remoteDir, os.W_OK):
                remote_writeable = True
                tmpDir = remoteDir
            else:
                pyhrf.verbose(1, 'Remote dir is not writeable -> using tmp ' \
                                  'dir to store splitted data & then upload.')

            #2. split roi data
            pyhrf.verbose(1, 'Path to store sub treatments: %s' %tmpDir)
            treatments_dump_files = []
            self.split(dump_sub_results=True, output_dir=tmpDir,
                       make_sub_outputs=False,
                       output_file_list=treatments_dump_files)

            #3. copy data to remote directory
            if not remote_writeable:
                host = cfg_parallel['remote_host']
                pyhrf.verbose(1, 'Uploading data to %s ...' %(remoteDir))
                remote_input_files = remote_copy(treatments_dump_files,
                                                 host, remoteUser, remoteDir)

            #4. create job list
            tasks_list = []
            for f in treatments_dump_files:
                f = op.join(remoteDir,op.basename(f))
                nice = cfg_parallel['niceness']
                tasks_list.append('nice -n %d %s -v%d -t "%s"' \
                                      %(nice,'pyhrf_jde_estim',
                                        pyhrf.verbose.verbosity,f))
            mode = 'dispatch'
            tasks = grid.read_tasks(';'.join(tasks_list), mode)
            timeslot = grid.read_timeslot('allday')
            hosts = grid.read_hosts(cfg_parallel['hosts'])
            brokenfile = op.join(tmpDir, 'pyhrf-broken_cmd.batch')

            odir = self.output_dir or pyhrf.get_tmp_path()
            logfile = op.join(odir, 'pyhrf-parallel.log')
            pyhrf.verbose(1, 'Log file for process dispatching: %s' \
                          %logfile)

            #3. launch them
            pyhrf.verbose(1, 'Dispatching processes ...')
            try:
                grid.run_grid(mode, hosts, 'rsa', tasks, timeslot, brokenfile,
                              logfile, user=remoteUser)
                grid.kill_threads()
            except KeyboardInterrupt:
                grid.quit(None, None)

            if len(open(brokenfile).readlines()) > 0:
                pyhrf.verbose(1, 'There are some broken commands, '\
                                  'trying again ...')
                try:
                    tasks = grid.read_tasks(brokenfile, mode)
                    grid.run_grid(mode, hosts, 'rsa', tasks, timeslot, brokenfile,
                                  logfile, user=remoteUser)
                    grid.kill_threads()
                except KeyboardInterrupt:
                    grid.quit(None, None)

            #3.1 grab everything back ??
            #try:
                # "scp %s@%s:%s %s" %(remoteUser,host,
                #                     op.join(remoteDir,'result*'),
                #                     op.abspath(op.dirname(options.cfgFile))))
            #TODO : test if everything went fine

            #4. merge all results and create outputs
            result = []
            #if op.exists(remoteDir): TODO :scp if remoteDir not readable
            nb_treatments = len(treatments_dump_files)
            remote_result_files = [op.join(remoteDir, 'result_%04d.pck' %i) \
                                    for i in range(nb_treatments)]
            pyhrf.verbose(1,'remote_result_files: %s' %str(remote_result_files))
            nres = len(filter(op.exists,remote_result_files))
            if nres == nb_treatments:
                pyhrf.verbose(1, 'Grabbing results ...')
                for fnresult in remote_result_files:
                    fresult = open(fnresult)
                    result.append(cPickle.load(fresult)[0])
                    fresult.close()
            else:
                print 'Found only %d result files (expected %d)' \
                    %(nres, nb_treatments)
                print 'Something went wrong, check the log files'
            if not remote_writeable:
                pyhrf.verbose(1, 'Cleaning tmp dir (%s)...' %tmpDir)
                shutil.rmtree(tmpDir)
                pyhrf.verbose(1, 'Cleaning up remote dir (%s) through ssh ...' \
                                %remoteDir)
                cmd = 'ssh %s@%s rm -f "%s" "%s" "%s"' \
                    %(remoteUser, host, ' '.join(remote_result_files),
                      ' '.join(remote_input_files), remote_fcfg)
                pyhrf.verbose(2, cmd)
                os.system(cmd)
            else:
                if 0:
                    pyhrf.verbose(1, 'Cleaning up remote dir (%s)...' %remoteDir)
                    for f in os.listdir(remoteDir):
                        os.remove(op.join(remoteDir,f))

        elif parallel == 'cluster':

            from pyhrf.parallel import run_soma_workflow
            cfg = pyhrf.cfg['parallel-cluster']
            #create tmp remote path:
            date_now = time.strftime('%c').replace(' ','_').replace(':','_')
            remote_path = op.join(cfg['remote_path'], date_now)
            pyhrf.verbose(1,'Create tmp remote dir: %s' %remote_path)
            remote_mkdir(cfg['server'], cfg['user'], remote_path)
            #if self.result_dump_file
            t_name = 'default_treatment'
            tmp_dir = pyhrf.get_tmp_path()
            label_for_cluster = self.analyser.get_label()
            if self.output_dir is None:
                out_dir = pyhrf.get_tmp_path()
            else:
                out_dir = self.output_dir
            result = run_soma_workflow({t_name:self}, 'pyhrf_jde_estim',
                                       {t_name:tmp_dir}, cfg['server_id'],
                                       cfg['server'], cfg['user'],
                                       {t_name:remote_path},
                                       {t_name:op.abspath(out_dir)},
                                       label_for_cluster, wait_ending=True)

        else:
            raise Exception('Parallel mode "%s" not available' %parallel)

        pyhrf.verbose(1, 'Retrieved %d results' %len(result))
        return self.output(result, (self.result_dump_file is not None),
                           self.make_outputs)


    def split(self, dump_sub_results=None, make_sub_outputs=None,
              output_dir=None, output_file_list=None):

        if dump_sub_results is None:
            dump_sub_results = (self.result_dump_file is not None)
        if make_sub_outputs is None:
            make_sub_outputs = self.make_outputs

        if output_dir is None:
            output_dir = self.output_dir

        sub_treatments = [FMRITreatment(d, deepcopy(self.analyser),
                                        make_outputs=make_sub_outputs,
                                        output_dir=output_dir) \
                              for d in self.analyser.split_data(self.data)]

        if output_dir is not None:
            pyhrf.verbose(1, 'Dump sub treatments in: %s ...' %output_dir)
            cmp_size = lambda t1,t2:cmp(t1.data.get_nb_vox_in_mask(),
                                        t2.data.get_nb_vox_in_mask())

            for it, sub_t in enumerate(sorted(sub_treatments, cmp=cmp_size,
                                              reverse=True)):
                if dump_sub_results:
                    sub_t.result_dump_file = op.join(output_dir,
                                                     'result_%04d.pck' %it)
                fn = op.join(output_dir, 'treatment_%04d.pck' %it)
                fout = open(fn, 'w')
                cPickle.dump(sub_t, fout)
                fout.close()
                if output_file_list is not None:
                    output_file_list.append(fn)

        return sub_treatments

    def output(self, result, dump_result=True, outputs=True):
        if dump_result and self.result_dump_file is not None:
            self.pickle_result(result)

        if outputs:
            pyhrf.verbose(1, 'Output of results to %s ...' %self.output_dir)
            try:
                tIni = time.time()
                r = self.analyser.outputResults(result, self.output_dir)
                pyhrf.verbose(1,'Creation of outputs took : %s'
                              %format_duration(time.time()-tIni))
                return r
            except NotImplementedError :
                pyhrf.verbose(1,'No output function defined')


    def pickle_result(self, result):
        if self.result_dump_file is not None :
            t0 = time.time()

            if splitext(self.result_dump_file)[1] == '.gz' :
                res_file = splitext(self.result_dump_file)[0]
            else:
                res_file = self.result_dump_file

            pyhrf.verbose(1,'Dumping results uncompressed...')
            f = open(res_file,'w')
            cPickle.dump(result, f)
            f.close()
            if splitext(self.result_dump_file)[1] == '.gz':
                pyhrf.verbose(1,'Gzip results ...')
                os.system('gzip "%s"' %res_file)

            pyhrf.verbose(2,'Dumping done ... time spent: '\
                          + str(time.time()-t0)+ ' sec')

    def clean_output_files(self):
        if self.result_dump_file is not None and \
                op.exists(self.result_dump_file):
            os.remove(self.result_dump_file)
        self.analyser.clean_output_files(self.output_dir)


def append_common_treatment_options(parser):
    parser.add_option('-s','--spm-mat-file', metavar='MATFILE', dest='spmFile',
                      default=None,
                      help='SPM.mat from which to extract paradigm data '\
                      '(onsets and stimulus durations) and TR. Note: if '\
                      'the option "-p/--paradigm" is also provided, then'\
                      ' the latter is ignored.')

    paradigms = ['loc_av', 'loc', 'loc_cp_only', 'loc_cpcd', 'language',
                 'loc_ainsi', 'loc_ainsi_cpcd']
    parser.add_option('-p','--paradigm', dest='paradigm', default=paradigms[0],
                      metavar='STRING', type='choice', choices=paradigms,
                      help='Paradigm to use, choices are: '+ \
                          string.join(paradigms,',') + '. Default is %default.'\
                          ' Note: ignored if option "-s/--spm-mat-file" is '\
                          'provided.')

    parser.add_option('-r','--paradigm-csv', dest='paradigm_csv',
                      default=None,
                      metavar='CSVFILE',
                      help='Paradigm CSV file input')
    parser.add_option('-I','--time-repetition', dest='tr',
                        default=None,
                        metavar='FLOAT',type='float',
                        help='Repetition time')


    inputTypes = ['volume', 'surface', 'simulation']
    parser.add_option('-d','--data-type', type='choice', choices=inputTypes,
                      metavar='STRING', dest='inputDataType', default='volume',
                      help='Define the type of input data, choices are: ' + \
                          string.join(inputTypes,',') + ' default is %default')

    data_choices = ['default', 'small', 'realistic']
    parser.add_option('-t', choices=data_choices, dest='data_scenario',
                      metavar='STRING', default='default',
                      help='Scenario for default data set: %s.' \
                          %', '.join(data_choices))


    parser.add_option('-v','--verbose',dest='verbose',metavar='INTEGER',
                      type='int',default=0,
                      help=dictToString(pyhrf.verboseLevels))

    parser.add_option('-f','--func-data-file', action='append',
                      dest='func_data_file',
                      metavar='FILE', default=None,
                      help='Functional data file (BOLD signal).')

    parser.add_option('-k','--mask-file',dest='mask_file',
                      metavar='FILE', default=None,
                      help='Functional mask file '\
                          '(n-ary, may be a parcellation).')

    parser.add_option('-g','--mesh-file',dest='mesh_file',
                      metavar='FILE', default=None,
                      help='Mesh file (only for surface analysis)')


def parse_data_options(options):

    from pyhrf.core import DEFAULT_BOLD_SURF_FILE, DEFAULT_BOLD_VOL_FILE, \
        DEFAULT_SIMULATION_FILE
    # If SPM.mat is provided, retrieve paradigm from it for all sessions.
    # Leave data file pathes to unknown.
    if options.spmFile is not None:
        if options.inputDataType == 'volume':
            SessDataClass = FMRISessionVolumicData
            if options.func_data_file is None:
                data_fns = DEFAULT_BOLD_VOL_FILE
            else:
                data_fns = options.func_data_file
            if options.mask_file is None:
                options.mask_file = DEFAULT_MASK_VOL_FILE
            fmriDataInit = FmriData.from_vol_ui
        elif options.inputDataType == 'surface':
            SessDataClass = FMRISessionSurfacicData
            if options.func_data_file is None:
                data_fns = DEFAULT_BOLD_SURF_FILE
            else:
                data_fns = options.func_data_file
            if options.mask_file is None:
                options.mask_file = DEFAULT_MASK_SURF_FILE

            fmriDataInit = FmriData.from_surf_ui
        elif options.inputDataType == 'simulation':
            SessDataClass = FMRISessionSimulationData
            data_fns = DEFAULT_SIMULATION_FILE
            fmriDataInit = FmriData.from_simu_ui


        if not isinstance(data_fns, list):
            data_fns = [data_fns]

        paradigm,tr = load_paradigm_from_mat(options.spmFile)
        sessions_data = []

        # print 'paradigm:', paradigm.keys()
        # print 'data_fns:', len(data_fns)
        # print data_fns
        #TODO: check nb of sessions and nb of data files
        for isess, sess in enumerate(sorted(paradigm.keys())):
            #print len(paradigm.keys())
            sessions_data.append(SessDataClass(paradigm[sess]['onsets'],
                                               paradigm[sess]['stimulusLength'],
                                               data_fns[isess]))

        if options.inputDataType == 'surface':

            return fmriDataInit(sessions_data=sessions_data, tr=tr,
                                mask_file=options.mask_file,
                                mesh_file=options.mesh_file)

        return fmriDataInit(sessions_data=sessions_data, tr=tr,
                            mask_file=options.mask_file)


    #unstack & take 1st set of onsets for each condition to get only one session
    onsets = unstack_trees(eval('pyhrf.paradigm.onsets_%s' %options.paradigm))[0]
    durations = unstack_trees(eval('pyhrf.paradigm.durations_%s' \
                                   %options.paradigm))[0]
    #print options.paradigm

    if options.paradigm_csv is not None:
        onsets, durations = load_paradigm_from_csv(options.paradigm_csv)
        from pyhrf.tools import apply_to_leaves
        onsets = apply_to_leaves(onsets, lambda x: x[0])
        durations = apply_to_leaves(durations, lambda x: x[0])



    ### Set data type:
    if options.inputDataType == 'volume' :
        if options.data_scenario == 'default':
            SessDataClass = FMRISessionVolumicData
            sd = SessDataClass(onsets, durations)
            if options.mask_file is None:
                options.mask_file = DEFAULT_MASK_VOL_FILE
        elif options.data_scenario == 'small':
            sd = SessDataClass(onsets, durations)
            if options.mask_file is None:
                options.mask_file = DEFAULT_MASK_SMALL_VOL_FILE
        elif options.data_scenario == 'realistic':
            sd = FMRISessionVolumicData(onsets, durations,
                                        REALISTIC_REAL_DATA_BOLD_VOL_FILE)
            if options.mask_file is None:
                options.mask_file = REALISTIC_REAL_DATA_MASK_VOL_FILE
        else:
            raise Exception("Uknown data scenario: %s" %options.data_scenario)
        #print 'sd!!:', sd
        #print 'mask:', options.mask_file



        if options.func_data_file is not None:
            #print 'options.func_data_file:', options.func_data_file
            sessions_data = []
            sessions_data.append(SessDataClass(onsets, durations, options.func_data_file))
        else:
            sessions_data = [sd]
        #print options
        if hasattr(options, 'tr') and options.tr is not None:
            tr = options.tr
            res = FmriData.from_vol_ui(sessions_data=sessions_data, tr=tr,
                                    mask_file=options.mask_file)
        else:
            res = FmriData.from_vol_ui(sessions_data=sessions_data,
                                       mask_file=options.mask_file)

        return res

    elif options.inputDataType == 'surface':
        mask_fn = DEFAULT_MASK_SURF_FILE
        mesh_fn = DEFAULT_MESH_FILE

        if options.data_scenario == 'default':
            #TODO: create a bigger surface default dataset
            sd = FMRISessionSurfacicData(onsets, durations)
        if options.data_scenario == 'small':
            sd = FMRISessionSurfacicData(onsets, durations)
        elif options.data_scenario == 'realistic':
            raise NotImplementedError('Realistic surfacic dataset not yet '\
                                          'available (TODO)')

        return FmriData.from_surf_ui(sessions_data=[sd], mask_file=mask_fn,
                                     mesh_file=mesh_fn)

    elif options.inputDataType == 'simulation':
        if options.data_scenario == 'default':
            sd = FMRISessionSimulationData(onsets,durations)
        elif options.data_scenario == 'small':
            raise NotImplementedError('Small artificial dataset not yet '\
                                          'available (TODO)')
        if options.data_scenario == 'realistic':
            raise NotImplementedError('Realistic artificial dataset not yet '\
                                          'available (TODO)')


        return FmriData.from_simu_ui(sessions_data=[sd])



def run_pyhrf_cmd_treatment(cfg_cmd, exec_cmd, default_cfg_file,
                            default_profile_file, label_for_cluster):


    usage = 'usage: %%prog [options]'

    description = 'Manage a joint detection-estimation treatment of fMRI data.' \
                'This command runs the treatment defined in an xml '\
                'parameter file. See pyhrf_jde_buildcfg command to build a'\
                'template of such a file. If no xml file found, then runs a '\
                'default example analysis.'

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-c','--input-cfg-file', metavar='XMLFILE', dest='cfgFile',
                    default=default_cfg_file,
                    help='Configuration file: XML file containing parameters'\
                    ' defining input data and analysis to perform.')

    parser.add_option('-r','--roi-data', metavar='PICKLEFILE', dest='roidata',
                    default=None, help='Input fMRI ROI data. The data '\
                    'definition part in the config file is ignored.')

    parser.add_option('-t','--treatment_pck',
                      metavar='PICKLEFILE', dest='treatment_pck',
                      default=None, help='Input treatment as a pickle dump.' \
                          'The XML cfg file is ignored')

    parser.add_option('-s','--stop-on-error', dest='stop_on_error',
                      action='store_true',
                    default=False, help='For debug: do not continue if error' \
                          ' during one ROI analysis')


    parser.add_option('-v','--verbose',dest='verbose',metavar='INTEGER',
                    type='int',default=0,
                    help=dictToString(pyhrf.verboseLevels))

    parser.add_option('-p','--profile',action='store_true', default=False,
                    help='Enable profiling of treatment. Store profile data in '\
                        '%s. NOTE: not avalaible in parallel mode.'\
                    %default_profile_file)

    parallel_choices = ['LAN','local','cluster']
    parser.add_option('-x','--parallel', choices=parallel_choices,
                    help='Parallel processing. Choices are %s'\
                        %string.join(parallel_choices,', '))


    (options,args) = parser.parse_args()

    pyhrf.verbose.setVerbosity(options.verbose)

    t0 = time.time()

    if options.treatment_pck is not None:
        f = open(options.treatment_pck)
        treatment = cPickle.load(f)
        f.close()
    else:
        if not os.path.exists(options.cfgFile):
            print 'Error: could not find default configuration file "%s"\n'\
                'Consider running "%s" to generate it.' \
                %(options.cfgFile, cfg_cmd)
            sys.exit(1)
        else:
            pyhrf.verbose(1, 'Loading configuration from: "%s" ...' \
                              %options.cfgFile)
            f = open(options.cfgFile, 'r')
            sXml = string.join(f.readlines())
            f.close()
            treatment = xmlio.fromXML(sXml)
            if 0:
                sXml = xmlio.toXML(treatment,
                                   handler=xmlio.xmlnumpy.NumpyXMLHandler())
                f = './treatment_cmd.xml'
                fOut = open(f,'w')
                fOut.write(sXml)
                fOut.close()
            #f = open(fOut, 'w')
            #cPickle.dump(treatment, f)
            #f.close()


    treatment.analyser.set_pass_errors(not options.stop_on_error)

    if options.parallel is not None:

        # tmpDir = tempfile.mkdtemp(prefix='pyhrf',
        #                           dir=pyhrf.cfg['global']['tmp_path'])
        # pyhrf.verbose(1, 'Tmpdir: %s' %tmpDir)

        treatment.run(parallel=options.parallel)

    else:
        if options.roidata is not None:
            #treatment.set_roidata(options.roidata)
            pyhrf.verbose(1, 'Loading ROI data from: "%s" ...' \
                              %options.roidata)

            roidata = cPickle.load(open(options.roidata))
            roidata.verbosity = pyhrf.verbose.verbosity
            if pyhrf.verbose > 1:
                print roidata.getSummary()
            #TODO: enable profiling
            pyhrf.verbose(1, 'Launching analysis ...')
            if options.profile:
                cProfile.runctx("result = treatment.analyser(roidata)",
                                globals(),
                                {'treatment':treatment,'roidata': roidata},
                                default_profile_file)
            else:
                result = treatment.analyser(roidata)
            outPath = op.dirname(op.abspath(options.roidata))
            fOut = op.join(outPath,"result_%04d.pck" %roidata.get_roi_id())
            pyhrf.verbose(1, 'Dumping results to %s ...' %fOut)
            f = open(fOut, 'w')
            cPickle.dump(result, f)
            f.close()
        else:
            pyhrf.verbose(1, 'ROI data is none')
            if options.profile:
                cProfile.runctx("treatment.run()", globals(),
                                {'treatment':treatment}, default_profile_file)
            else:
                #print 'treatment:', treatment
                treatment.run()

    pyhrf.verbose(1, 'Estimation done, took %s' %format_duration(time.time() - t0))

from pyhrf.tools import add_prefix, add_suffix
from pyhrf.ui.jde import JDEAnalyser as JDE
from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser as VBJDE

from pyhrf.ui.jde import DEFAULT_CFG_FILE as DEFAULT_CFG_FILE_JDE
from pyhrf.ui.jde import DEFAULT_OUTPUT_FILE as DEFAULT_OUTPUT_FILE_JDE

from pyhrf.jde.models import BOLDGibbsSampler as BG
from pyhrf.jde.models import GGG_BOLDGibbsSampler as BG3

from pyhrf.jde.beta import BetaSampler as BS
from pyhrf.jde.nrl.bigaussian import NRLSampler as NS
from pyhrf.jde.nrl.trigaussian import GGGNRLSampler as NS3
from pyhrf.jde.hrf import RHSampler as HVS
from pyhrf.jde.hrf import HRFSampler as HS

#VT = FMRIVolumeTreatment
#ST = FMRISurfaceTreatment

DEFAULT_JDE_OUT_MASK_VOL = add_prefix(DEFAULT_OUT_MASK_VOL_FILE, 'jde_')
DEFAULT_JDE_OUT_MASK_SURF = add_prefix(DEFAULT_OUT_MASK_SURF_FILE, 'jde_')

def make_outfile(fn, path, pre='', suf=''):
    if fn is None or path is None:
        return None
    ofn = op.join(path, fn)
    return add_prefix(add_suffix(ofn, suf), pre)

def create_treatment(boldFiles, parcelFile, dt, tr, paradigmFile,
                     nbIterations=4000,
                     writeXmlSetup=True, parallelize=False,
                     outputDir=None, outputSuffix=None, outputPrefix=None,
                     contrasts=None, beta=.6, estimBeta=True,
                     pfMethod='ps', estimHrf=True, hrfVar=.01,
                     roiIds=None,
                     nbClasses=2,gzip_rdump=False, make_outputs=True,
                     vbjde=False, simulation_file=None):
    if roiIds is None:
        roiIds = np.array([],dtype=int)


    if make_outputs:
        outFile = make_outfile(DEFAULT_OUTPUT_FILE_JDE, outputDir,
                               outputPrefix,
                               outputSuffix)
    else:
        outFile = None

    outMask = make_outfile(DEFAULT_JDE_OUT_MASK_VOL, outputDir, outputPrefix,
                           outputSuffix)

    outDump = make_outfile(DEFAULT_DUMP_FILE, outputDir, outputPrefix,
                           outputSuffix)
    if gzip_rdump:
        outDump += '.gz'

    fmri_data = FmriData.from_vol_files(parcelFile, paradigmFile, boldFiles, tr)

    if simulation_file is not None:
        f_simu = open(simulation_file)
        simulation = cPickle.load(f_simu)
        f_simu.close()
        fmri_data.simulation = simulation

    if contrasts is not None:
        cons = dict( ("con_%d"%i, ce) \
                         for i,ce in enumerate(";".split(contrasts)) )
    else:
        cons = {}

    if(vbjde):
        analyser = JDEVEMAnalyser(dt=dt)
    else:
        if nbClasses == 2:
            sampler = BG({
                            BG.P_NB_ITERATIONS : nbIterations,
                            # level of spatial correlation = beta
                            BG.P_BETA : BS({
                                    BS.P_VAL_INI : np.array([beta]),
                                    BS.P_SAMPLE_FLAG : estimBeta,
                                    BS.P_PARTITION_FUNCTION_METH : pfMethod,
                                    }),
                            # HRF
                            BG.P_HRF : HS({
                                    HS.P_SAMPLE_FLAG : estimHrf,
                                    }),
                            # HRF variance
                            BG.P_RH : HVS({
                                    HVS.P_SAMPLE_FLAG : False,
                                    HVS.P_VAL_INI : np.array([hrfVar]),
                                    }),
                            # neural response levels (stimulus-induced effects)
                            BG.P_NRLS : NS({
                                    NS.P_CONTRASTS : cons,
                                    }),
                            })

            # paramt = {
            #     VT.P_SESSIONS : psess,
            #     VT.P_TR : tr,
            #     VT.P_ROI_MASK : parcelFile,
            #     VT.P_ROI_OUT_MASK : outMask,
            #     VT.P_ROI_IDS : roiIds,
            #     VT.P_RESULT_DUMP_FILE : outDump,
            #     VT.P_ANALYSER : JDE(sampler=sampler, outputFile=outFile, dt=dt),
            #     }



        elif nbClasses == 3:
            sampler = BG3({
                            BG.P_NB_ITERATIONS : nbIterations,
                            # level of spatial correlation = beta
                            BG.P_BETA : BS({
                                    BS.P_VAL_INI : np.array([beta]),
                                    BS.P_SAMPLE_FLAG : estimBeta,
                                    BS.P_PARTITION_FUNCTION_METH : pfMethod,
                                    }),
                            # HRF
                            BG.P_HRF : HS({
                                    HS.P_SAMPLE_FLAG : estimHrf,
                                    }),
                            # HRF variance
                            BG.P_RH : HVS({
                                    HVS.P_SAMPLE_FLAG : False,
                                    HVS.P_VAL_INI : np.array([hrfVar]),
                                    }),
                            # neural response levels (stimulus-induced effects)
                            BG.P_NRLS : NS3({
                                    NS.P_CONTRASTS : cons,
                                    }),
                            })

            # paramt = {
            #     VT.P_SESSIONS : psess,
            #     VT.P_TR : tr,
            #     VT.P_ROI_MASK : parcelFile,
            #     VT.P_ROI_OUT_MASK : outMask,
            #     VT.P_ROI_IDS : roiIds,
            #     VT.P_RESULT_DUMP_FILE : outDump,
            #     VT.P_ANALYSER : JDE(sampler=sampler, outputFile=outFile, dt=dt),
            #     }
        analyser = JDEMCMCAnalyser(sampler, dt=dt)

    tjde = FMRITreatment(fmri_data, analyser, outputDir)

    sxml = xmlio.toXML(tjde, handler=NumpyXMLHandler())
    if writeXmlSetup and outputDir is not None:
        outSetupXml = make_outfile(DEFAULT_CFG_FILE_JDE, outputDir,
                                   outputPrefix, outputSuffix)
        pyhrf.verbose(1, "Writing XML setup to: " + outSetupXml )
        f = open(outSetupXml, 'w')
        f.write(sxml)
        f.close()
    else:
        outSetupXml = None

    return tjde, outSetupXml



def create_treatment_surf(boldFiles, parcelFile, meshFile, dt, tr, paradigmFile,
                          nbIterations=4000,
                          writeXmlSetup=True, parallelize=False,
                          outputDir=None, outputSuffix=None,
                          outputPrefix=None,
                          contrasts=';', beta=.6, estimBeta=True,
                          pfMethod='ps', estimHrf=True, hrfVar=.01,
                          roiIds=None,
                          nbClasses=2,gzip_rdump=False,
                          simulation_file=None, make_outputs=True):
    if roiIds is None:
        roiIds = np.array([],dtype=int)

    outFile = make_outfile(DEFAULT_OUTPUT_FILE_JDE, outputDir, outputPrefix,
                           outputSuffix)

    outDump = make_outfile(DEFAULT_DUMP_FILE, outputDir, outputPrefix,
                           outputSuffix)
    if gzip_rdump:
        outDump += '.gz'


    if contrasts is not None:
            cons = dict( ("con_%d"%i, ce) \
                             for i,ce in enumerate(";".split(contrasts)) )
    else:
        cons = {}

    if nbClasses == 2:
        sampler = BG({
                        BG.P_NB_ITERATIONS : nbIterations,
                        # level of spatial correlation = beta
                        BG.P_BETA : BS({
                                BS.P_VAL_INI : np.array([beta]),
                                BS.P_SAMPLE_FLAG : estimBeta,
                                BS.P_PARTITION_FUNCTION_METH : pfMethod,
                                }),
                        # HRF
                        BG.P_HRF : HS({
                                HS.P_SAMPLE_FLAG : estimHrf,
                                }),
                        # HRF variance
                        BG.P_RH : HVS({
                                HVS.P_SAMPLE_FLAG : False,
                                HVS.P_VAL_INI : np.array([hrfVar]),
                                }),
                        # neural response levels (stimulus-induced effects)
                        BG.P_NRLS : NS({
                                NS.P_CONTRASTS : cons,
                                }),
                        })

    elif nbClasses == 3:
        sampler = BG3({
                        BG.P_NB_ITERATIONS : nbIterations,
                        # level of spatial correlation = beta
                        BG.P_BETA : BS({
                                BS.P_VAL_INI : np.array([beta]),
                                BS.P_SAMPLE_FLAG : estimBeta,
                                BS.P_PARTITION_FUNCTION_METH : pfMethod,
                                }),
                        # HRF
                        BG.P_HRF : HS({
                                HS.P_SAMPLE_FLAG : estimHrf,
                                }),
                        # HRF variance
                        BG.P_RH : HVS({
                                HVS.P_SAMPLE_FLAG : False,
                                HVS.P_VAL_INI : np.array([hrfVar]),
                                }),
                        # neural response levels (stimulus-induced effects)
                        BG.P_NRLS : NS3({
                                NS.P_CONTRASTS : cons,
                                }),
                        })

    analyser = JDEMCMCAnalyser(sampler, dt=dt)

    fmri_data = FmriData.from_surf_files(paradigmFile, boldFiles, tr, meshFile,
                                         parcelFile)

    if simulation_file is not None:
        f_simu = open(simulation_file)
        simulation = cPickle.load(f_simu)
        f_simu.close()
        fmri_data.simulation = simulation


    tjde = FMRITreatment(fmri_data, analyser, outputDir)
    #print 'make_outputs:', make_outputs


    sxml = xmlio.toXML(tjde, handler=NumpyXMLHandler())
    if writeXmlSetup is not None and outputDir is not None:
        outSetupXml = make_outfile(DEFAULT_CFG_FILE_JDE, outputDir,
                                   outputPrefix,
                                   outputSuffix)
        pyhrf.verbose(1, "Writing XML setup to: " + outSetupXml )
        f = open(outSetupXml, 'w')
        f.write(sxml)
        f.close()
    else:
        outSetupXml = None

    return tjde, outSetupXml



def jde_vol_from_files(boldFiles=[DEFAULT_BOLD_VOL_FILE],
                       parcelFile=DEFAULT_MASK_VOL_FILE,
                       dt=.6, tr=2.4, paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                       nbIterations=4000,
                       writeXmlSetup=True, parallelize=None,
                       outputDir=None, outputSuffix=None, outputPrefix=None,
                       contrasts=None, beta=.6, estimBeta=True,
                       pfMethod='ps', estimHrf=True, hrfVar=.01,
                       roiIds=None,force_relaunch=False,
                       nbClasses=2, gzip_rdump=False, dry=False,
                       make_outputs=True, vbjde=False,
                       simulation_file=None):

    if parallelize is not None:
        writeXmlSetup = True

    tjde, xml_file = create_treatment(boldFiles, parcelFile, dt, tr,
                                      paradigm_csv_file, nbIterations,
                                      writeXmlSetup, parallelize,
                                      outputDir, outputSuffix, outputPrefix,
                                      contrasts, beta, estimBeta,
                                      pfMethod, estimHrf, hrfVar,
                                      roiIds,nbClasses, gzip_rdump,
                                      make_outputs, vbjde,
                                      simulation_file)

    if not force_relaunch and tjde.already_done():
        pyhrf.verbose(1, 'JDE analysis already done')
        #TODO: compare written xml setup to input setup
        return tjde, xml_file
    else:
        pyhrf.verbose(1, 'JDE analysis not done, going on ...')

    if not dry:
        if parallelize is None:
            tjde.run()
        else:
            cmd_jde = 'pyhrf_jde_estim -v%d -x %s -c "%s"' \
                %(pyhrf.verbose.verbosity, parallelize, xml_file)
            exec_status = os.system(cmd_jde)
            if exec_status != 0:
                raise Exception('JDE command failed! (%s)' %cmd_jde)
    return tjde, xml_file



def jde_surf_from_files(boldFiles=[DEFAULT_BOLD_SURF_FILE],
                        parcelFile=DEFAULT_MASK_SURF_FILE,
                        meshFile=DEFAULT_MESH_FILE,
                        dt=.6, tr=2.4, paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                        nbIterations=4000,
                        writeXmlSetup=True, parallelize=None,
                        outputDir=None, outputSuffix=None,
                        outputPrefix=None,
                        contrasts=None, beta=.6, estimBeta=True,
                        pfMethod='ps', estimHrf=True, hrfVar=.01,
                        roiIds=None,force_relaunch=False,
                        nbClasses=2, gzip_rdump=False, dry=False,
                        simulation_file=None):

    if parallelize is not None:
        writeXmlSetup = True

    tjde, xml_file = create_treatment_surf(boldFiles, parcelFile, meshFile,
                                           dt, tr, paradigm_csv_file,
                                           nbIterations,
                                           writeXmlSetup, parallelize,
                                           outputDir, outputSuffix,
                                           outputPrefix,
                                           contrasts, beta, estimBeta,
                                           pfMethod, estimHrf, hrfVar,
                                           roiIds,nbClasses, gzip_rdump,
                                           simulation_file)

    if not force_relaunch and tjde.already_done():
        pyhrf.verbose(1, 'JDE analysis already done')
        return tjde, xml_file
    else:
        pyhrf.verbose(1, 'JDE analysis not done, going on ...')

    if not dry:
        result = tjde.run(parallel=parallelize)
        # if parallelize is None:
        #     tjde.run()
        # else:
        #     cmd_jde = 'pyhrf_jde_estim -v%d -x %s -c "%s"' \
        #         %(pyhrf.verbose.verbosity, parallelize, xml_file)
        #     #TODO return command status
        #     os.system(cmd_jde)
    else:
        result = None
    return tjde, xml_file, result


def jde_vol_is_done(directory, outputPrefix=None, outputSuffix=None):
    if directory is None:
        return False
    outFile = make_outfile(DEFAULT_OUTPUT_FILE_JDE, directory, outputPrefix,
                           outputSuffix)
    return op.exists(outFile)

