

# -*- coding: utf-8 -*-
from os.path import join, exists, basename
from pyhrf.tools import cartesian
from itertools import chain


def apply_to_dict(d, f):
    return dict([ (k,f(k,v)) for k,v in d.iteritems()])

def safe_list(l):
    if not isinstance(l, list):
        return [l]
    else:
        return l

def safe_init(x, default):
    if x is None:
        return default
    else:
        return x

def same(x):
    """ identity function """
    return x

def check_subdirs(path, labels, tree, not_found=None):

    if not_found is None:
        not_found = []

    if not exists(path):
        not_found.append(path)
        return not_found

    if len(labels) == 0:
        return not_found

    for subdir in tree[labels[0]]:
        check_subdirs(join(path, subdir), labels[1:], tree, not_found)

    return not_found

class StructuredDataParser:


    def __init__(self, directory_labels, allowed_subdirectories,
                 directory_forgers=None, file_forgers=None, root_path=None):

        self.dir_values = allowed_subdirectories
        self.dir_labels = directory_labels
        self.root = root_path
        #self.squeezable_dirs = [ l for l,c in dir_labels if len(c)==1 ]
        self.mandatory_dirs = set([l for l,c in self.dir_values.iteritems() \
                                       if len(c)>1])

        self.dir_forgers = safe_init(directory_forgers, {})
        self.file_forgers = safe_init(file_forgers, {})

    def set_root(self, root):
        self.root = root


    def _resolve_file_defs(self, defs):
        print 'resolve_file_defs ...'
        print defs
        #return [self.file_forgers[d[0]](d[1:]) for d in safe_list(defs)]
        return self.file_forgers[defs[0]](defs[1:]) #file_forgers is a dict, file_forgers[defs[0]] is a fct (defs[0] is 'nrl' here) --> fct called with arg defs[1:] : file_forgers[defs[0]](defs[1:]), arg=the conditions here

    def _forge_sub_dir(self, dir_label, dir_value):
        """ map a directory alias to its full value """
        return [self.dir_forgers.get(dir_label,same)(dv) \
                    for dv in safe_list(dir_value) ]

    def get_files(self, file_def, **subdirs):

        # Some sanity checks:
        subdir_labels = set(subdirs.keys())
        if not self.mandatory_dirs.issubset(subdir_labels):
            #print ssubdirs
            #print self.mandatory_dirs
            diff = self.mandatory_dirs.difference(subdir_labels)
            raise Exception('\n'.join(['Missing mandatory directories:'] + \
                                          list(diff)))

        for dir_label in subdirs.iterkeys():
            assert dir_label in self.dir_labels

        #TODO check that input subdirs are all in allowed_subdirectories

        all_dirs = self.dir_values.copy()
        all_dirs.update(subdirs)
        all_dirs = apply_to_dict(all_dirs, self._forge_sub_dir)

        selected_dirs = cartesian(*all_dirs.values())
        dir_ids = dict([(i,l) for l,i in enumerate(all_dirs)])
        pathes = [join(*[d[dir_ids[l]] for l in self.dir_labels]) \
                      for d in selected_dirs]

        if self.root is None:
            raise Exception('Root is not defined!')
        if not exists(self.root):
            raise Exception('Root does not exist: %s' %self.root)

        subdirs_not_found = check_subdirs(self.root, self.dir_labels, all_dirs)
        if len(subdirs_not_found) > 0:
            raise Exception('\n'.join(['Levels not found in subdirs:'] + \
                                          list(reversed(subdirs_not_found))))

        #f_defs = chain(*self._resolve_file_defs(file_def))
        f_defs = self._resolve_file_defs(file_def)
        #print list(f_defs)
        files = [join(*[self.root]+f) for f in cartesian(pathes,f_defs)]
        print 'files:'
        print files
        files_not_found = [f for f in files if not exists(f)]

        if len(files_not_found) > 0:
            raise Exception('\n'.join(['Files not found:'] + files_not_found))

        return files


    def get_file_blocs(self, file_defs, **subdirs):
        return [self.get_files(fdef, **subdirs) for fdef in file_defs]


# functions to build a full subdir from its alias:
optimed_dir_forgers = {
    'antenna' : lambda x: 'Subjects_A%s_siemens_3D' %x,
    'acquisition' : lambda x: 'acq%s' %x,
    }

def forge_nrl_files(conditions):
    """ construct list of nrl files from list of conditions """
    print conditions
    return ['jde_pm_nrl_condition_%s.nii' %c for c in safe_list(conditions)]

# functions to build file names:
optimed_file_forgers = {
    'con' : lambda contrasts: ['jde_pm_nrl_contrast_%s.nii' %con for con in safe_list(contrasts)],
    'nrl' : forge_nrl_files,
    'beta': lambda conditions: ['jde_pm_beta_mapped_condition_%s.nii' %c for c in safe_list(conditions)],
    'hrf' : lambda x: ['jde_hrf_pm.nii'],
    'ttp' : lambda x: ['jde_ttp.nii'],
    'whM' : lambda x: ['jde_whM.nii'],
    'fit' : lambda x: ['jde_fit_type_fit.nii'],
    }

#optimed_file_forgers is a dict, optimed_file_forgers['con'] returns a fct directly callable

# labels for all available directory levels
optimed_dir_labels = ['antenna','subject','modality','acquisition','analysis',
                      'pyhrf', 'scenario']

# define available values for each level
optimed_dir_values = {
    'acquisition' : ['normal_64x64', 'R2_96x96','R4_96x96'],
    'subject' : ['RG080250', 'RK090001'],
    'antenna' : ['12C'],
    'analysis' : ['new_analysis'], ###changed here
    'pyhrf' : ['change_var_HRF_old_parcellation'], ### changed here : instead of 'change_var_HRF_old_parcellation', 'pyhrf'
    'modality' : ['fMRI'],
    'scenario': ['AllConds_USMM-ES_hsmpl_htrick', 'AllConds_USMM-ES_hcano'],
    }


def unformat_nrl_file(file_nrl):
	return basename(file_nrl).replace('jde_pm_nrl_condition_','').replace('.nii','')




optimed_parser = StructuredDataParser(optimed_dir_labels, optimed_dir_values,
									  optimed_dir_forgers, optimed_file_forgers)
