import sys
import os
import os.path as op
import time
import itertools

import shutil
import cPickle
import numpy as np

import pyhrf
from pyhrf.ndarray import xndarray, stack_cuboids, TIME_AXIS, MRI3Daxes
from pyhrf.sandbox.design_and_ui import Initable

from pyhrf.tools import PickleableStaticMethod, stack_trees, unstack_trees, \
     format_duration
import pyhrf.tools.io as pio


from pyhrf.paradigm import Paradigm, builtin_paradigms
from pyhrf.graph import parcels_to_graphs, kerMask3D_6n

try:
    from collection import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


class FmriData(): #Initable?

    """ Hold multi-session fMRI data: paradigm, 4D time series, mask and mesh (for
    surface data).
    It can be instanciated with different methods:
        - *__init__*, which takes numpy arrays.
        - *from_vol_files* or *from_surf_files*, which take lists of data files.
    """

    def __init__(self, func_data, stim_onsets, stim_durations, tr, data_type,
                 mask, bg_label=None, sessions_scans=None,
                 simulation=None, meta_data=None, graphs=None):
        """
        Create a multi-session fMRI data object from stimulus onsets and
        durations, time of repetition, functional 4D data (3D+time) and a
        functional mask.

        Args:
            *stim_onsets* (OrderedDict of list):
                a dictionary mapping a stimulus name to a list of session onsets.
                Each item of this list should be a 1D numpy float array of
                onsets for a given session.
            *stim_durations* (OrderedDict of list):
                same data type as *stim_onsets*, but contains stimulus durations.
                If None then assume zero-length stimuli.
            *tr* (float): time of repetition in second.
            *data_type* (str): either "volume" or "surface".
            *func_data* (numpy array):
                a 4D array (for volume data) or a 2D array (for surface data).
                If None, then a data file must be given via *data_files*, eg
                data_files={'func_data':'./func_data.nii'}
            *sessions_scans* (list of numpy array):
                a list of session indexes along scan axis. If None,
                all scans of data files are taken.
            *mask* (numpy array): the functional mask as
                a 3D array (for volume data) or 1D array (for surface data).
                This array must be n-ary, ie only contain integers.
            *bg_label* (int): background label.
                Positions defined by mask==bg_label will be discarded in
                functional data.
            *simulation* (dict): a dictionary of simulation items.
        """

        self.data_type = data_type
        self.tr = tr
        if bg_label is None:
            roi_ids = np.unique(mask.flat)
            if data_type == 'surface' or \
              len(roi_ids) == 1:
                self.bg_label = 0
            else: #volume data_type
                self.bg_label = mask.min()
        else:
            self.bg_label = bg_label

        self._store_mask_sparse(mask)
        self._graphs = graphs

        self._store_flat_fdata(func_data)

        if sessions_scans is None:
            sessions_scans = [np.arange(func_data.shape[0], dtype=np.int32)]

        self.sessions_scans = sessions_scans

        sessions_durations = [len(ss)*tr for ss in sessions_scans]
        self.paradigm = Paradigm(stim_onsets, sessions_durations, stim_durations)

        self.meta_data = meta_data
        self.simulation = simulation


    def _store_flat_fdata(self, fdata):
        m = self.np_mask
        if isinstance(fdata, xndarray):
            fdata = fdata.data

        if self.data_type == 'volume':
            if len(m) != 3:
                raise Exception('Wrong mask dim for volume data (%dD), '\
                                'should be 3D' %len(m))
            if fdata.ndim != 4:
                raise Exception('Wrong func data dim for volume data (%dD), '\
                                'should be 4D' %fdata.ndim)
            if TIME_AXIS == 0:
                fdata = fdata[:, m[0], m[1], m[2]]
            else: # TIME_AXIS = 3
                fdata = fdata[m[0], m[1], m[2], :].transpose()
            self.spatial_axes = MRI3Daxes
        else: # surfacic or already flatten data
            if len(m) != 1:
                raise Exception('Wrong mask dim (%dD), '\
                                'should be 1D' %len(m))
            if fdata.ndim != 2:
                raise Exception('Wrong func data dim (%dD), '\
                                'should be 2D (time, voxel)' %fdata.ndim)
            if len(m[0]) != fdata.shape[1]:
                fdata = fdata[:, m[0]]

            self.spatial_axes = ['voxel']

        self.fdata = fdata

    def get_mask(self):
        """ Getter method for mask. Only non-zero positions are stored in the
        current object to save (lots of) memory.
        """

        mask = np.zeros(self.spatial_shape, dtype=self.roi_ids_in_mask.dtype) + \
          self.bg_label
        mask[self.np_mask] = self.roi_ids_in_mask
        return mask
    mask = property(get_mask)

    def _compute_graph(self):
        if self.data_type != 'volume':
            raise Exception('Can only compute graph for volume data')
        pyhrf.verbose(6, 'FmriData._compute_graph() ...')
        to_discard = [self.backgroundLabel]
        self._graph = parcels_to_graphs(self.roiMask, kerMask3D_6n,
                                        toDiscard=to_discard)
    def get_graph(self):
        if self._graph is None:
            self._compute_graph()
        return self._graph
    graph = property(get_graph)

    def _store_mask_sparse(self, mask):

        self.np_mask = np.where(mask != self.bg_label)
        self.roi_ids_in_mask = mask[self.np_mask]
        self.nb_voxels_in_mask = len(self.roi_ids_in_mask)
        self.spatial_shape = mask.shape



from pyhrf import get_data_file_name
from pyhrf.tools.io import read_mesh
from pyhrf.graph import graph_from_mesh
DEFAULT_DATA_TYPE = 'volume'
DEFAULT_MASK = get_data_file_name('subj0_parcellation.nii.gz')
DEFAULT_TR = 2.4
DEFAULT_ONSETS = OrderedDict(
    (('audio' , [np.array([ 15.,20.7,29.7,35.4,44.7,48.,83.4,89.7,108.,
                            119.4, 135., 137.7, 146.7, 173.7, 191.7, 236.7,
                            251.7, 284.4, 293.4, 296.7])]),
     ('video' , [np.array([ 0., 2.4, 8.7,33.,39.,41.7, 56.4, 59.7, 75., 96.,
                            122.7, 125.4, 131.4, 140.4, 149.4, 153., 156., 159.,
                           164.4, 167.7, 176.7, 188.4, 195., 198., 201., 203.7,
                            207., 210., 218.7, 221.4, 224.7, 234., 246., 248.4,
                            260.4, 264., 266.7, 269.7, 278.4, 288. ])])
     )
    )
DEFAULT_STIM_DURATIONS = OrderedDict(
    (('audio', [np.array([])]),
     ('video', [np.array([])]))

    )
DEFAULT_PARADIGM_CSV = get_data_file_name('paradigm_localizer_audio_video_only.csv')
DEFAULT_BOLD = get_data_file_name('subj0_bold_session0.nii.gz')

DEFAULT_MESH = get_data_file_name('real_data_surf_tiny_mesh.gii')
DEFAULT_SURF_MASK = get_data_file_name('real_data_surf_tiny_parcellation.gii')
DEFAULT_SURF_BOLD = get_data_file_name('real_data_surf_tiny_bold.gii')


class MultiSessionsDataUI(Initable):

    def __init__(self, paradigm_file=DEFAULT_PARADIGM_CSV,
                 func_data_files=[DEFAULT_BOLD]):
        Initable.__init__(self)
        self.paradigm_file = paradigm_file
        self.func_data_files = func_data_files

    def load_and_get_fdata_params(self, mask):

        if op.splitext(self.paradigm_file)[-1] == '.csv':
            onsets, durations = pio.load_paradigm_from_csv(self.paradigm_file)
        else:
            raise Exception('Only CSV file format support for paradigm')

        fns = self.func_data_files
        pyhrf.verbose(1, 'Load functional data from: %s' %',\n'.join(fns))
        fdata = stack_cuboids([xndarray.load(fn) for fn in fns], 'session')

        fdata = np.concatenate(fdata.data) #scan sessions along time axis
        pio.discard_bad_data(fdata, mask)
        pyhrf.verbose(1, 'Functional data shape %s' %str(fdata.shape))

        return {'stim_onsets': onsets, 'stim_durations':durations,
                'func_data': fdata}


    # Interface with command line

    @PickleableStaticMethod
    def append_cmd_options(self, parser):
        """
        """
        parser.add_option('-f','--func-data-file', action='append',
                          dest='func_data_files',
                          metavar='FILE', default=None,
                          help='Functional data file for one session.'\
                          '(duplicate for several session')

        parser.add_option('-p','--paradigm', dest='paradigm',
                          default=builtin_paradigms[0],
                          metavar='STRING|FILE',
                          help='Paradigm can be specified by: a SPM.mat or '\
                          'a CSV file or a paradigm label refering to '\
                          'a built-in paradigm (%s)' %','.join(builtin_paradigms))

    @PickleableStaticMethod
    def from_cmd_options(self, options):

        if options.func_data_files is None:
            options.func_data_files = [DEFAULT_BOLD]

        if op.splitext(options.paradigm)[-1] == '.mat' or \
          op.splitext(options.paradigm)[-1] == '.csv':
            paradigm_file = options.paradigm
        else:
            if options.paradigm not in builtin_paradigms:
                raise Exception('Paradigm label is not in built in paradigms. '\
                                'Choices are: (%s)' %','.join(builtin_paradigms))

            paradigm_file = pyhrf.get_data_file_name('paradigm_%s.csv' \
                                                     %options.paradigm)

        msdui = MultiSessionsDataUI(paradigm_file, options.func_data_files)
        return msdui

class SessionDataUI(Initable):

    def __init__(self, stim_onsets=unstack_trees(DEFAULT_ONSETS)[0],
                 stim_durations=unstack_trees(DEFAULT_STIM_DURATIONS)[0],
                 func_data_file=DEFAULT_BOLD):
        Initable.__init__(self)
        self.onsets = stim_onsets
        self.durations = stim_durations
        self.func_data_file = func_data_file


    def to_dict(self):
        return {'stim_onsets': self.onsets,
                'stim_durations' : self.durations,
                'func_data_file' : self.func_data_file
                }

    @PickleableStaticMethod
    def load_and_get_fdata_params(self, sessions_data, mask):
        params = stack_trees([sd.to_dict() for sd in sessions_data])

        fns = params.pop('func_data_file')
        pyhrf.verbose(1, 'Load functional data from: %s' %',\n'.join(fns))
        fdata = stack_cuboids([xndarray.load(fn) for fn in fns], 'session')

        fdata = np.concatenate(fdata.data) #scan sessions along time axis
        pio.discard_bad_data(fdata, mask)
        pyhrf.verbose(1, 'Functional data shape %s' %str(fdata.shape))
        params['func_data'] = fdata

        return params

class MaskUI(Initable):

    def __init__(self, mask_file=DEFAULT_MASK, mesh_file=None,
                 data_type='volume'):
        Initable.__init__(self)
        self.mask_file = mask_file
        self.mesh_file = mesh_file
        self.data_type = data_type

    @PickleableStaticMethod
    def from_vol_mask(self, mask_file=DEFAULT_MASK):
        m = MaskUI(mask_file, None, data_type='volume')
        m.set_init(MaskUI.from_vol_mask, mask_file=mask_file)
        return m

    @PickleableStaticMethod
    def from_surface_mask(self, mask_file=DEFAULT_SURF_MASK,
                          mesh_file=DEFAULT_MESH):
        m = MaskUI(mask_file, mesh_file, data_type='surface')
        m.set_init(MaskUI.from_surface_mask, mask_file=mask_file,
                   mesh_file=mesh_file)
        return m

    def load_and_get_fdata_params(self):
        pyhrf.verbose(1,'Load mask from: %s' %self.mask_file)
        if self.data_type == 'surface':
            pyhrf.verbose(2,'Read mesh from: %s' %self.mesh_file)
        p = {'mask' : xndarray.load(self.mask_file).data}
        pyhrf.verbose(1, 'Mask shape %s' %str(p['mask'].shape))

        if self.data_type == 'surface':
            p['graph'] = graph_from_mesh(read_mesh(self.mesh_file))

        return p

    # Interface with command line

    @PickleableStaticMethod
    def append_cmd_options(self, parser):
        """
        """
        parser.add_option('-k','--mask-file',dest='mask_file',
                          metavar='FILE', default=DEFAULT_MASK,
                          help='Functional mask file '\
                          '(n-ary, may be a parcellation).')

        parser.add_option('-g','--mesh-file',dest='mesh_file',
                          metavar='FILE', default=None,
                          help='Mesh file (only for surface data)')

    @PickleableStaticMethod
    def from_cmd_options(self, options):

        if pio.is_volume_file(options.mask_file):
            data_type = 'volume'
        elif pio.is_texture_file(options.mask_file):
            data_type = 'surface'
        else:
            raise Exception('Unrecognised format for mask file %s' \
                            %options.mask_file)

        return MaskUI(options.mask_file, options.mesh_file, data_type=data_type)


class FmriDataUI(Initable):

    def __init__(self, geometry=MaskUI(), sessions_data=[SessionDataUI()],
                 tr=DEFAULT_TR):

        Initable.__init__(self)
        self.mask_ui = geometry
        self.sessions_data_ui = sessions_data
        self.tr = tr
        self.data_type = geometry.data_type


    def get_fmri_data(self):
        params = {'tr' : self.tr,
                  'data_type': self.data_type,
                  }
        m_p = self.mask_ui.load_and_get_fdata_params() #mask params
        params.update(m_p)

        sd = self.sessions_data_ui
        if isinstance(sd, MultiSessionsDataUI):
            func_params = sd.load_and_get_fdata_params(m_p['mask'])
        else: #list of Sessions_data_ui
            func_params = SessionDataUI.load_and_get_fdata_params(sd, m_p['mask'])
        params.update(func_params)

        return FmriData(**params)

    def load_func_data(self, mask):
        # Load func data for all sessions and flatten them according to mask
        mask = mask != self.bg_label
        cfdata = [xndarray.load(f).flatten(mask, self.spatial_axes, 'voxel') \
                  for f in self.func_files]
        # flatten along sessions:
        cfdata = stack_cuboids(cfdata, 'session').reorient(['session','time']+\
                                                           self.spatial_axes)
        return np.concatenate(cfdata.data)

    @PickleableStaticMethod
    def from_paradigm_csv(self, mask_ui=MaskUI(),
                          paradigm_csv_file=DEFAULT_PARADIGM_CSV,
                          func_data_files=[DEFAULT_BOLD],
                          tr=DEFAULT_TR, data_type=DEFAULT_DATA_TYPE):
        """
        Interface to get multi-session paradigm data from a CSV file.
        functional data files are specified as a list of 4D data files
        """
        msdui = MultiSessionsDataUI(paradigm_csv_file, func_data_files)
        fdui = FmriDataUI(mask_ui, msdui, tr)

        fdui.set_init(FmriDataUI.from_paradigm_csv, mask_ui=mask_ui,
                      paradigm_csv_file=paradigm_csv_file,
                      func_data_files=func_data_files, tr=tr,
                      data_type=data_type)

        return fdui

    @PickleableStaticMethod
    def append_cmd_options(self, parser):
        """
        """
        MaskUI.append_cmd_options(parser)
        MultiSessionsDataUI.append_cmd_options(parser)
        parser.add_option('-r','--tr', dest='tr',metavar='FLOAT',
                          type='int', default=DEFAULT_TR,
                          help='Time of repetition in second')

    @PickleableStaticMethod
    def from_cmd_options(self, options):

        mui = MaskUI.from_cmd_options(options)
        fdui = FmriDataUI(mui, MultiSessionsDataUI.from_cmd_options(options),
                          options.tr)

        return fdui


    def to_xml(self):
        return self.to_ui_node('functional_data').to_xml()

    @PickleableStaticMethod
    def from_xml(self, xml):
        from pyhrf.sandbox.design_and_ui import UiNode
        return Initable.from_ui_node(UiNode.from_xml(xml))

def exec_t(t):
    return t.execute()


class AnalyserUI(Initable):

    def __init__(self):
        Initable.__init__(self)

class TreatmentUI(Initable):

    def __init__(self, data=FmriDataUI(), analyser=AnalyserUI(),
                 result_dump_file=None, output_dir=None):
        Initable.__init__(self)

        self.data_ui = data
        self.analyser_ui = analyser

        self.data = None
        self.analyser = None

    def force_data_dir(self, path):
        """ Change base directories of all data files to *path* """
        self.data.force_data_dir(path)
        self.analyser.force_data_dir(path)

    def get_data_files(self):
        """ Return all file names used in data and analyser definitions """
        return self.data.get_data_files() + self.analyser.get_data_files()

    def load(self):
        self.data = self.data_ui.get_fmri_data()
        self.analyser = self.analyser_ui.get_analyser()

    def execute(self):


        if self.data is None:
            self.data = self.data_ui.get_fmri_data()

        if self.analyser is None:
            self.analyser = self.analyser_ui.get_analyser()

        lg = pyhrf.verbose.verbosity >= 2
        pyhrf.verbose(2, self.data.get_summary(long=lg))

        pyhrf.verbose(1,'All data loaded !')
        pyhrf.verbose(1,'running estimation ...')
        #TODO : print summary of analyser setup.
        pyhrf.verbose(1,'Estimation start date is : %s'
                      %time.strftime('%c'))
        tIni = time.time()
        result = self.analyser.analyse(self.data)

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
                print 'Can not import joblib. It is required to enable '\
                    'parallel processing on a local machine.'
                sys.exit(1)

            parallel_verb = pyhrf.verbose.verbosity
            if pyhrf.verbose.verbosity == 6:
                parallel_verb = 10

            if n_jobs is None:
                n_jobs = cfg_parallel['nb_procs']

            p = Parallel(n_jobs=n_jobs, verbose=parallel_verb)
            result = p(delayed(exec_t)(t) for t in self.split())
            # join list of lists:
            result = list(itertools.chain.from_iterable(result))

        elif parallel == 'LAN':

            import grid
            cfg_parallel = pyhrf.cfg['parallel-LAN']
            remoteUser = cfg_parallel['user']

            #1. Some checks on input/output directory
            remoteDir = cfg_parallel['remote_path']
            # At the end, results will be retrieved direclty from remoteDir,
            # which has to be readable
            if remoteDir is None or not op.exists(remoteDir):
                raise Exception('Remote directory is not readable (%s)' \
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
                remote_input_files = pio.remote_copy(treatments_dump_files,
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
            logfile = op.join(self.output_dir, 'pyhrf-parallel.log')
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
            pyhrf.verbose(1,'remote_result_files: %s', str(remote_result_files))
            nres = len(filter(op.exists,remote_result_files))
            if nres == nb_treatments:
                pyhrf.verbose(1, 'Grabbing results ...')
                for fnresult in remote_result_files:
                    fresult = open(fnresult)
                    result.append(cPickle.load(fresult))
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
                      ' '.join(remote_input_files))
                pyhrf.verbose(2, cmd)
                os.system(cmd)
            else:
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
            pio.remote_mkdir(cfg['server'], cfg['user'], remote_path)
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

