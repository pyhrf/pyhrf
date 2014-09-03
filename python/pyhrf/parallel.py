# -*- coding: utf-8 -*-


import os
import os.path as op
from os.path import basename, splitext
import cPickle
import shutil
import subprocess
from tempfile import mkdtemp

import pyhrf
from pyhrf.tools._io import remote_copy
from pyhrf import xmlio
try:
    from soma_workflow.client import Job, FileTransfer
    from soma_workflow.client import Group
    from soma_workflow.client import Helper
    from soma_workflow.client import Workflow, WorkflowController
except:
    pass

class RemoteException(Exception):
    pass

def save_treatment(t,f):
    sXml = xmlio.to_xml(t, handler=xmlio.xmlnumpy.NumpyXMLHandler())
    fOut = open(f,'w')
    fOut.write(sXml)
    fOut.close()

def prepare_treatment_jobs(treatment, tmp_local_dir, local_result_path,
                           local_user, local_host, remote_host, remote_user,
                           remote_path, label_for_cluster):
    """
    Prepare somaworkflow jobs to perform one treatment (ie one subject).

    Args:
        treatment (FMRITreatment): the treatment defining the analysis
        tmp_local_dir (str): a path where to store the temporary config file
                             before sending it to the remote host
        local_result_path (str): path where to store the final result
        local_user (str): the user on the local host who enables SHH connection
                          from the remote cluster
        local_host (str): local host (used to send back the result)
        remote_host (str): remote machine where the treatment will be run
        remote_user (str): user login on the remote machine.
        remote_path (str): path on the remote machine where to store ROI data and
                           analysis results
        label_for_cluster (str): label prefix to name job in somaworkflow

    Returns:
            a tuple (job_split, jobs, dependencies, mainGroup)

            job_split (Job): job handling splitting of input data into ROI data
            jobs (list of Job): all jobs except the splitting jobs
                               -> roi analyses, result merge,
                                  scp of result back to local host, data cleaning
            dependencies (list of Job pairs): define the pipeline structure
            mainGroup (Group): top-level object gathering all jobs for
                               this treatment.
    """

    # roiFiles contains the list of files that will be produced by job_split
    roiFiles, roiIds = treatment.dump_roi_datasets(dry=True)

    pyhrf.verbose(1, 'Get list of splitted data files ... %d files' \
                      %len(roiFiles))
    datafiles = treatment.get_data_files()

    # Make all path be relative in the treatment config file
    # so that data file can be found on the cluster file system
    treatment.replace_data_dir('./')
    remote_cfg_file = op.join(tmp_local_dir,'./detectestim_remote.xml')
    treatment.set_init_param('make_outputs', False)
    pyhrf.verbose(1, 'Save remote treatment to %s' %remote_cfg_file)
    save_treatment(treatment, remote_cfg_file)

    pyhrf.verbose(1, 'Upload input data')
    # All data which are the inputs of the workflow:
    data_to_upload = datafiles+[remote_cfg_file]
    remote_input_files = remote_copy(data_to_upload, remote_host,
                                     remote_user, remote_path)
    #print 'remote_input_files:'
    #print remote_input_files
    pyhrf.verbose(1, 'Remove tmp remote cfg file')
    os.remove(remote_cfg_file)

    pyhrf.verbose(1, 'Prepare jobs ...')
    pyhrf.verbose(1, 'Job split ...')
    verbosity = pyhrf.verbose.verbosity
    cmd = ["pyhrf_split_roidata","-c", basename(remote_cfg_file),
           "-v %d" %verbosity, "-d", "./"]
    pyhrf.verbose(2, '-> %s' %cmd)
    job_split = Job(cmd, working_directory=remote_path, name="roi_split")

    pyhrf.verbose(1, 'Jobs JDE ...')
    jobs_jde = [Job(["pyhrf_jde_estim","-c", basename(remote_cfg_file),
                     "-r", basename(roiFile), "-v %d" %verbosity],
                    working_directory=remote_path,
                    name="jde_r%04d" %roiId)
                for roiFile, roiId in zip(roiFiles, roiIds)]
    pyhrf.verbose(2, 'First jde job -> %s' %jobs_jde[0].command)
    # Files produced by all JDE jobs, which will be then used as input of the
    # merge job:
    resultFiles = ["result_%04d.pck" %iroi for iroi in roiIds]

    pyhrf.verbose(1, 'Job pack result ...')
    # Output of the merge job, which has to transfered back to local:
    remote_resultFile = './result.pck'
    pyhrf.verbose(1, 'Remote result file: %s' %remote_resultFile)

    cmd = ["pyhrf_pack_results",'-v1','-o',remote_resultFile]+resultFiles
    pyhrf.verbose(3, 'cmd pack result: %s' %cmd)
    job_merge = Job(cmd, working_directory=remote_path,
                    name="merge_results")


    # Retrieve result file:
    #local_host = "132.166.200.5" #HACK
    #cmd = ["pyhrf_shell_cmd", "scp","-C",remote_resultFile, "%s@%s:\"%s\"" \
               #%(local_user,local_host,local_result_path)]
    cmd = ["scp","-C",remote_resultFile, "%s@%s:\"%s\"" \
               %(local_user,local_host,local_result_path)]           
    
    pyhrf.verbose(2, 'cmd scp result: %s' %cmd)
    job_scp_result = Job(cmd, working_directory=remote_path, name="scp_result")

    # Clean everything:
    # -> all input files, splitted roi data, result for each data, merged result:
    #cmd = ["pyhrf_shell_cmd", "rm","-f", remote_resultFile] + \
      #map(basename, roiFiles) + resultFiles + remote_input_files
    #pyhrf.verbose(3, 'cmd clean: %s' %cmd)
    cmd = ["rm","-f", remote_resultFile] + \
      map(basename, roiFiles) + resultFiles + remote_input_files
    pyhrf.verbose(3, 'cmd clean: %s' %cmd)
    job_clean = Job(cmd, working_directory=remote_path, name="clean_files")

    pyhrf.verbose(1,'Setup of work flow ...')

    # Build the Job lists, dependencies and group
    clean = True
    if clean:
        nodes = [job_merge,job_scp_result,job_clean] + jobs_jde
    else:
        nodes = [job_merge,job_scp_result] + jobs_jde
    dependencies = []
    for jj in jobs_jde:
        dependencies.append((job_split,jj))
        dependencies.append((jj,job_merge))
    dependencies.append((job_merge,job_scp_result))
    if clean:
        dependencies.append((job_scp_result,job_clean))

    jjGroup = Group(elements=jobs_jde, name=label_for_cluster+'-roi_jobs')
    if clean:
        elements = [job_split,jjGroup,job_merge,
                    job_scp_result,job_clean]
    else:
        elements = [job_split,jjGroup,job_merge,
                    job_scp_result]
    mainGroup = Group(name=label_for_cluster,
                      elements=elements)

    return job_split, nodes, dependencies, mainGroup

def run_soma_workflow(treatments, exec_cmd, tmp_local_dirs, server_id,
                      remote_host, remote_user, remote_pathes,
                      local_result_pathes, label_for_cluster,
                      wait_ending=False):
    """
    Dispatch treatments using soma-workflow.
    - 'treatments' is a dict mapping a treatment name to a treatment object
    - 'exec_cmd' is the command to run on each ROI data.
    - 'tmp_local_dirs' is a dict mapping a treatment name to a local tmp dir
      (used to store a temporary configuration file)
    - 'server_id' is the server ID as expected by WorkflowController
    - 'remote_host' is the remote machine where treatments are treated in parallel
    - 'remote_user' is used to log in remote_host
    - 'remote_pathes' is a dict mapping a treatment name to an existing remote dir
      which will be used to store ROI data and result files
    - 'local_result_pathes' is a dict mapping a treatment name to a local path
      where final results will be sorted (host will send it there by scp)
    - 'label_for_cluster' is the base name used to label workflows and sub jobs
    - 'make_outputs' is a flag to tell wether to build outputs from result or not.
      -> not operational yet (#TODO)
    """

    import getpass
    from socket import gethostname

    local_user = getpass.getuser()
    local_host = gethostname()

    all_nodes = []
    all_deps = []
    all_groups = []
    split_jobs = []
    for t_id, treatment in treatments.iteritems():

        tmp_local_dir = tmp_local_dirs[t_id]
        remote_path = remote_pathes[t_id]
        local_result_path = local_result_pathes[t_id]

        sj, n, d, g = prepare_treatment_jobs(treatment, tmp_local_dir,
                                             local_result_path,
                                             local_user, local_host,
                                             remote_host,
                                             remote_user, remote_path,
                                             label_for_cluster+'-'+str(t_id))
        all_nodes.extend(n)
        all_deps.extend(d)
        all_groups.append(g)
        split_jobs.append(sj)


    # Jobs for data splitting should be done sequentially.
    # If they're done in parallel, they may flood the remote file system
    for isj in xrange(len(split_jobs)):
        if isj+1 < len(split_jobs):
            all_deps.append((split_jobs[isj],split_jobs[isj+1]))

    # # Be sure that all splitting jobs are done first:
    # # Is there a better way ?
    # for n in all_nodes:
    #     for sjob in split_jobs:
    #         all_deps.append((sjob,n))
    # Does not seem to work well -> maybe to many deps ?

    workflow = Workflow(all_nodes+split_jobs, all_deps, root_group=all_groups)

    # f = open('/tmp/workflow.pck','w')
    # cPickle.dump(workflow, f)
    # f.close()

    pyhrf.verbose(1,'Open connection ...')
    connection = WorkflowController(server_id, remote_user)

    pyhrf.verbose(1,'Submit workflow ...')
    wf_id = connection.submit_workflow( workflow=workflow,
                                        #expiration_date="",
                                        #queue="run32",
                                        name=label_for_cluster+'-' + \
                                            local_user)
    #wf = connection.workflow(wf_id)

    if wait_ending: #wait for result
        pyhrf.verbose(1,'Wait for workflow to end and make outputs ...')
        Helper.wait_workflow(wf_id, connection)

        for t_id, local_result_path in local_result_pathes.iteritems():
            treatment = treatments[t_id]
            rfilename = treatment.result_dump_file
            if rfilename is None:
                rfilename = 'result.pck'
            local_result_file = op.join(local_result_path,
                                        op.basename(rfilename))

            if not op.exists(local_result_file):
                raise Exception('Local result does not exist "%s"' \
                                    %local_result_file)

        if treatment.analyser.outFile is not None:
            #return result only for last treatment ...
            print 'Load result from %s ...' %local_result_file
            if splitext(local_result_file)[1] == '.gz':
                import gzip
                fresult = gzip.open(local_result_file)
            else:
                fresult = open(local_result_file)
            results = cPickle.load(fresult)
            fresult.close()
            #print 'Make outputs ...'
            #treatment.output(results, dump=False)
            pyhrf.verbose(1, 'Cleaning tmp dirs ...')
            for tmp_dir in tmp_local_dirs.itervalues():
                shutil.rmtree(tmp_dir)

            return results
    else:
        pyhrf.verbose(1, 'Cleaning tmp dirs ...')
        for tmp_dir in tmp_local_dirs.itervalues():
            shutil.rmtree(tmp_dir)

        pyhrf.verbose(1,'Workflow sent, returning ...')
        return []





#argv: -c, input params file, func file, output file
cfunc_marshal = "import sys;import cPickle;p=cPickle.load(open(sys.argv[1]));"\
    "import marshal,types;ff=open(sys.argv[2]);"\
    "code=marshal.loads(ff.read());ff.close();"\
    "f=types.FunctionType(code,globals());print 'p:',p;"\
    "o=f(*p[0],**p[1]);fout=open(sys.argv[3],'w');cPickle.dump(o,fout);"\
    "fout.close();"

def dump_func(func, fn):
    import marshal
    f = open(fn,'w')
    marshal.dump(func.func_code, f)
    f.close()

import inspect
def merge_default_kwargs(func, kwargs):
    args,_,_,defaults = inspect.getargspec(func)
    if defaults is not None:
        default_kwargs = dict(zip(args[len(args)-len(defaults):],defaults))
        default_kwargs.update(kwargs)
        return default_kwargs
    else:
        return kwargs

def remote_map_marshal(func, largs=None, lkwargs=None, mode='local'):

    if largs is None:
        if lkwargs is not None:
            largs = [[]] * len(lkwargs)
        else:
            largs = []

    if lkwargs is None:
        lkwargs = [{}] * len(largs)

    lkwargs = [merge_default_kwargs(func,kw) for kw in lkwargs]

    assert len(lkwargs) == len(largs)

    all_args = zip(largs, lkwargs)

    if mode=='local':
        return [func(*args,**kwargs) for args,kwargs in all_args]
    elif mode=='local_with_dumps':

        func_fn = './func.marshal'
        dump_func(func, func_fn)
        results = []
        for i,params in enumerate(all_args):
            print 'params:', params
            params_fn = 'params_%d.pck' %i
            fparams = open(params_fn,'wb')
            cPickle.dump(params, fparams)
            fparams.close()
            output_fn = 'output_%d.pck' %i
            print 'call subprocess ...'
            subprocess.call(['python','-c', cfunc_marshal, params_fn,
                             func_fn, output_fn])
            print 'Read outputs ...'
            fout = open(output_fn)
            results.append(cPickle.load(fout))
        return results
    elif mode=='remote_cluster':
        # FileTransfer creation for input files
        #data_dir = './rmap_data'
        data_dir = mkdtemp(prefix="sw_rmap")
        func_fn = op.join(data_dir ,'func.marshal')
        dump_func(func, func_fn)
        func_file = FileTransfer(is_input=True,
                                 client_path=func_fn,
                                 name="func_file")

        all_jobs = []
        param_files = []
        for i,params in enumerate(all_args):
            params_fn = op.join(data_dir,'params_%d.pck' %i)
            fparams = open(params_fn,'wb')
            cPickle.dump(params, fparams)
            fparams.close()
            param_file = FileTransfer(is_input=True,
                                      client_path=params_fn,
                                      name='params_file_%d'%i)
            param_files.append(param_file)
            output_fn = op.join(data_dir,'output_%d.pck' %i)
            output_file = FileTransfer(is_input=False,
                                       client_path=output_fn,
                                       name='output_file_%d'%i)
            job = Job(command=['python','-c', cfunc, param_file, func_file,
                               output_file],
                      name="rmap, item %d" %i,
                      referenced_input_files=[func_file, param_file],
                      referenced_output_files=[output_file])
            all_jobs.append(job)

        workflow = Workflow(jobs=all_jobs, dependencies=[])
        # submit the workflow
        cfg = pyhrf.cfg['parallel-cluster']
        controller = WorkflowController(cfg['server_id'], cfg['user'])

        #controller.transfer_files(fids_to_transfer)
        wf_id = controller.submit_workflow(workflow=workflow, name="remote_map")

        Helper.transfer_input_files(wf_id, controller)

        Helper.wait_workflow(wf_id, controller)

        Helper.transfer_output_files(wf_id, controller)

        results = []
        for i in xrange(len(all_args)):
            fout = open(op.join(data_dir,'output_%d.pck'%i))
            results.append(cPickle.load(fout))
            fout.close()
        return results


cfunc = "import sys;import cPickle;p=cPickle.load(open(sys.argv[1]));"\
    "import %s;"\
    "o=%s(*p[0],**p[1]);fout=open(sys.argv[2],'w');cPickle.dump(o,fout);"\
    "fout.close();"


def remote_map(func, largs=None, lkwargs=None, mode='serial'):
    """
    Execute a function in parallel on a list of arguments.

    Args:
        *func* (function): function to apply on each item.
                           **this function must be importable on the remote side**
        *largs* (list of tuple): each item in the list is a tuple
                                 containing all positional argument values of the
                                 function
        *lkwargs* (list of dict): each item in the list is a dict
                                  containing all named arguments of the
                                  function mapped to their value.

        *mode* (str): indicates how execution is distributed. Choices are:

            - "serial": single-thread loop on the local machine
            - "local" : use joblib to run tasks in parallel.
                        The number of simultaneous jobs is defined in
                        the configuration section ['parallel-local']['nb_procs']
                        see ~/.pyhrf/config.cfg
            - "remote_cluster: use somaworkflow to run tasks in parallel.
                               The connection setup has to be defined
                               in the configuration section ['parallel-cluster']
                               of ~/.pyhrf/config.cfg.
            - "local_with_dumps": testing purpose only, run each task serially as
                                  a subprocess.

    Returns:
         a list of results

    Raises:
         RemoteException if any remote task has failed

    Example:
    >>> from pyhrf.parallel import remote_map
    >>> def foo(a, b=2): \
        return a + b
    >>> remote_map(foo, [(2,),(3,)], [{'b':5}, {'b':7}])
    [7, 10]
    """
    if largs is None:
        if lkwargs is not None:
            largs = [tuple()] * len(lkwargs)
        else:
            largs = [tuple()]

    if lkwargs is None:
        lkwargs = [{}] * len(largs)

    lkwargs = [merge_default_kwargs(func,kw) for kw in lkwargs]

    assert len(lkwargs) == len(largs)

    all_args = zip(largs, lkwargs)
    #print 'all_args:', all_args

    fmodule = func.__module__
    fname = '.'.join([fmodule, func.__name__])

    if mode=='serial':
        return [func(*args,**kwargs) for args,kwargs in all_args]
    elif mode=='local':
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError('Can not import joblib. It is '\
                              'required to enable parallel '\
                              'processing on a local machine.')

        if pyhrf.verbose.verbosity == 6:
            parallel_verb = 10
        else:
            parallel_verb = 0
        n_jobs = pyhrf.cfg['parallel-local']['nb_procs']
        p = Parallel(n_jobs=n_jobs, verbose=parallel_verb)
        return p(delayed(func)(*args,**kwargs) \
                     for args, kwargs in all_args)

    elif mode=='local_with_dumps':
        results = []
        for i,params in enumerate(all_args):
            #print 'params:', params
            params_fn = 'params_%d.pck' %i
            fparams = open(params_fn,'wb')
            cPickle.dump(params, fparams)
            fparams.close()
            output_fn = 'output_%d.pck' %i
            #print 'call subprocess ...'
            subprocess.call(['python','-c', cfunc%(fmodule,fname),
                             params_fn, output_fn])
            #print 'Read outputs ...'
            fout = open(output_fn)
            results.append(cPickle.load(fout))
        return results
    elif mode=='remote_cluster':
        # FileTransfer creation for input files
        #data_dir = './rmap_data'
        data_dir = mkdtemp(prefix="sw_rmap")

        all_jobs = []
        param_files = []
        for i,params in enumerate(all_args):
            params_fn = op.join(data_dir,'params_%d.pck' %i)
            fparams = open(params_fn,'wb')
            cPickle.dump(params, fparams)
            fparams.close()
            param_file = FileTransfer(is_input=True,
                                      client_path=params_fn,
                                      name='params_file_%d'%i)
            param_files.append(param_file)
            output_fn = op.join(data_dir,'output_%d.pck' %i)
            output_file = FileTransfer(is_input=False,
                                       client_path=output_fn,
                                       name='output_file_%d'%i)
            job = Job(command=['pyhrf_exec_pyfunc', fmodule,fname,
                               param_file, output_file],
                      name="rmap, item %d" %i,
                      referenced_input_files=[param_file],
                      referenced_output_files=[output_file])
            all_jobs.append(job)

        workflow = Workflow(jobs=all_jobs, dependencies=[])
        # submit the workflow
        cfg = pyhrf.cfg['parallel-cluster']
        controller = WorkflowController(cfg['server_id'], cfg['user'])
        #controller.transfer_files(fids_to_transfer)
        wf_id = controller.submit_workflow(workflow=workflow, name="remote_map")

        Helper.transfer_input_files(wf_id, controller)

        Helper.wait_workflow(wf_id, controller)

        Helper.transfer_output_files(wf_id, controller)

        results = []
        for i in xrange(len(all_args)):
            fnout = op.join(data_dir, 'output_%d.pck'%i)
            fout = open(fnout)
            o = cPickle.load(fout)
            print 'file cPickle loaded:', o
            fout.close()
            os.remove(fnout)
            if isinstance(o, Exception):
                raise RemoteException('Task %d failed'%i, o)
                if o.errno != 17:   
                    raise RemoteException('Task %d failed'%i, o)
            results.append(o)
        return results
