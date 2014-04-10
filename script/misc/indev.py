"""
Copy files with patterned names into a directory structure where:
    - a source is defined by a regular expressions with named groups
    - a target is defined by python format strings whose named arguments match 
      group names in the source regexp.

Example:
In folder ./raw_data, I have:
AC0832_anat.nii
AC0832_asl.nii
AC0832_bold.nii
PK0612_asl.nii
PK0612_bold.nii

I want to export these files in to the following directory structure:
./export/<subject>/<modality>/data.nii

src = '(?P<subject>[A-Z]{2}[0-9]{4}_(?P<modality>[a-zA-Z]+).nii'
src_folder = './raw_data/'
dest_folder = ('./export', '{subject}', '{modality}')
dest_basename = 'data.nii'
rx_copy(src, src_folder, dest_basename, dest_folder):

Should result in:
./export/AC0832/bold/data.nii
./export/AC0832/anat/data.nii
./export/AC0832/asl/data.nii 
./export/PK0612/bold/data.nii 
./export/PK0612/asl/data.nii 

"""
import os
import unittest
import os.path as op
import numpy as np

import matplotlib.pyplot as plt
import shutil

import pyhrf

from itertools import chain
from collections import defaultdict
import re

rx_py_identifier = '[^\d\W]\w*'

def find_duplicates(l):
    tmpset = set()
    duplicates = defaultdict(list)
    for i,e in enumerate(l):
        if e in tmpset:
            duplicates[e].append(i)
        tmpset.add(e)
    return duplicates.values()


def rx_copy(src, src_folder, dest_basename, dest_folder, dry=False):
    """
    Copy all file names matching the regexp *src* in folder *src_folder* to
    targets defined by format strings.
    The association between source and target file names must be injective.
    If several file names matched by *src* are associated with the same target,
    an exception will be raised.
    

    Args:
        - src (str): regular expression matching file names in the given folder
                     *src_folder* where group names are mapped to
                     format argument names in *targets*
        - src_folder (str): path where to search for files matched by *src*
        - dest_basename (str): format string with named arguments 
                               (eg 'file_{mytag}_{mytag2}.txt') used to form
                               target file basenames where named arguments are
                               substituted with group values extracted by *src*.
        - dest_folder (list of str): list of format strings with named 
                               arguments [eg ('folder_{mytag}', 
                               'folder_{mytag2}')] to be joined to form 
                               target directories.
                               Named arguments are substituted with group 
                               values extracted by *src*.
        - dry (bool): if True then do not perform any copy

    Return: None
    """
    # check consistency between src group names and dest items:
    src_tags = set(re.compile(src).groupindex.keys())

    re_named_args = re.compile('\{(.*)\}')
    folder_dest_tags = set(chain(*[re_named_args.findall(d) \
                                       for d in dest_folder]))
    bn_dest_tags = set(re_named_args.findall(dest_basename))
                
    if not folder_dest_tags.issubset(src_tags):
        raise Exception('Tags in dest_folder not defined in src: %s'\
                            %', '.join(folder_dest_tags.difference(src_tags)))
    if not bn_dest_tags.issubset(src_tags):
        raise Exception('Tags in dest_basename not defined in src: %s'\
                            %', '.join(bn_dest_tags.difference(src_tags)))

    # resolve file names
    input_files, output_files = [], []

    re_src = re.compile(src)
    for input_fn in os.listdir(src_folder):
        ri = re_src.match(input_fn)
        if ri is not None:
            input_files.append(op.join(src_folder, input_fn))
            subs = ri.groupdict()
            output_files.append(op.join(*([df.format(**subs) \
                                         for df in dest_folder] + \
                                        [dest_basename.format(subs)])))
            assert isinstance(output_files[-1], str)

    # check injectivity:
    duplicate_indexes = find_duplicates(output_files)
    if len(duplicate_indexes) > 0:
        sduplicates = '\n'.join(['\n'.join(input_files[i] for i in dups)+'\n'+\
                                '->' + output_files[dups[0]] + '\n'] \
                                for dups in duplicate_indexes)
        raise Exception('Copy is not injective, the following copy'
                        'operations have the same destination:\n %s' \
                        %sduplicates) 
    # do the copy:
    if not dry:
        for ifn, ofn in zip(input_files, output_files):
            # Create sub directories if not existent:
            output_item_dir = op.dirname(ofn)
            if not op.exists(output_item_dir):
                os.makedirs(output_item_dir)
            shutil.copy(ifn, ofn)
    return

class Test(unittest.TestCase):

    def setUp(self,):
        self.tmp_dir = pyhrf.get_tmp_path()

    def tearDown(self):
       shutil.rmtree(self.tmp_dir)

    def _create_files(self, fns):
        for fn in [op.join(self.tmp_dir,fn) for fn in fns]:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
            open(fn, 'a').close()

    def assert_file_exists(self, fn):
        if not op.exists(fn):
            raise Exception('File %s does not exist' %fn)

    def test_basic(self):
        self._create_files([op.join('./raw_data', f) \
                                for f in ['AC0832_anat.nii', 'AC0832_asl.nii', 
                                          'AC0832_bold.nii', 'PK0612_asl.nii',
                                          'PK0612_bold.nii']])
        src = '(?P<subject>[A-Z]{2}[0-9]{4})_(?P<modality>[a-zA-Z]+).nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = 'data.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder)

        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['AC0832/bold/data.nii',
                                 'AC0832/anat/data.nii',
                                 'AC0832/asl/data.nii', 
                                 'PK0612/bold/data.nii', 
                                 'PK0612/asl/data.nii']]:
            self.assert_file_exists(fn)

if __name__ == '__main__':
    unittest.main()       
