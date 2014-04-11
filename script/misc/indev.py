# Launch all unit tests and doc test in this script with code coverage:
# nosetests indev.py --with-coverage --cover-html --with-doctest --cover-package indev --cover-branches --cover-erase -v
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

I want to export these files into the following directory structure:
./export/<subject>/<modality>/data.nii
where <subject> and <modality> have to be replaced by chunks extracted
from the input files

To do so, define a regexp to catch useful chunks (or tags) in input files and
also format strings that will be used to create target file names::
    
    # regep to capture values of subject and modality:
    src = '(?P<subject>[A-Z]{2}[0-9]{4}_(?P<modality>[a-zA-Z]+).nii'
    # definition of targets:
    src_folder = './raw_data/'
    dest_folder = ('./export', '{subject}', '{modality}')
    dest_basename = 'data.nii'
    # do the thing:
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
#import numpy as np

#import matplotlib.pyplot as plt
import shutil

import pyhrf

from itertools import chain
from collections import defaultdict
import re

rx_py_identifier = '[^\d\W]\w*'

def find_duplicates(l):
    """ Find the index of duplicate elements in the given list. 
    Complexity is O(n*log(d)) where d is the number of unique duplicate 
    elements.

    Args:
        - l (list): the list where to search for duplicates
        
    Return: groups of indexes corresponding to duplicated element:
            -> list of list of elements

    Example:
    >>> find_duplicates([1,1,2,1,4,5,7,2])
    [[0, 1, 3], [2, 7]]
    """
    duplicates = defaultdict(list)
    for i,e in enumerate(l):
        duplicates[e].append(i)
    return [dups for dups in duplicates.values() if len(dups)>1]

class MissingTagError(Exception): pass

class DuplicateTargetError(Exception): pass

def rx_copy(src, src_folder, dest_basename, dest_folder, dry=False,
            tag_translations=None):
    """
    Copy all file names matching the regexp *src* in folder *src_folder* to
    targets defined by format strings.
    The association between source and target file names must be injective.
    If several file names matched by *src* are associated with the same target,
    an exception will be raised.
    
    Args:
        - src (str): regular expression matching file names in the given folder
                     *src_folder* where group names are mapped to
                     format argument names in *dest_folder* and *dest_basename*
        - src_folder (str): path where to search for files matched by *src*
        - dest_basename (str): format string with named arguments 
                               (eg 'file_{mytag}_{mytag2}.txt') used to form
                               target file basenames where named arguments are
                               substituted with group values caught by *src*.
        - dest_folder (list of str): 
                      list of format strings with named arguments, 
                      eg ('folder_{mytag}','folder_{mytag2}'),  to be joined 
                      to form target directories.
                      Named arguments are substituted with group values 
                      extracted by *src*.
        - dry (bool): if True then do not perform any copy

    Return: None
    """
    # check consistency between src group names and dest items:
    src_tags = set(re.compile(src).groupindex.keys())

    re_named_args = re.compile('\{(.*?)\}')
    folder_dtags = set(chain(*[re_named_args.findall(d) \
                                       for d in dest_folder]))
    bn_dest_tags = set(re_named_args.findall(dest_basename))
                
    if not folder_dtags.issubset(src_tags):
        raise MissingTagError('Tags in dest_folder not defined in src: %s'\
                              %', '.join(folder_dtags.difference(src_tags)))
    if not bn_dest_tags.issubset(src_tags):
        raise MissingTagError('Tags in dest_basename not defined in src: %s'\
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
                                        [dest_basename.format(**subs)])))
            assert isinstance(output_files[-1], str)

    # check injectivity:
    duplicate_indexes = find_duplicates(output_files)
    if len(duplicate_indexes) > 0:
        sduplicates = '\n'.join(*[['\n'.join(input_files[i] for i in dups) +  \
                                  '\n' + '-> ' + output_files[dups[0]] +'\n'] \
                                 for dups in duplicate_indexes])
        raise DuplicateTargetError('Copy is not injective, the following copy'\
                                   ' operations have the same destination:\n'\
                                   '%s' %sduplicates) 
    if not dry: # do the copy
        for ifn, ofn in zip(input_files, output_files):
            # Create sub directories if not existing:
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

    def _create_tmp_files(self, fns):
        for fn in [op.join(self.tmp_dir,fn) for fn in fns]:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
            open(fn, 'a').close()

    def assert_file_exists(self, fn):
        if not op.exists(fn):
            raise Exception('File %s does not exist' %fn)

    def test_basic(self):
        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['AC0832_anat.nii', 'AC0832_asl.nii', 
                                          'AC0832_bold.nii', 'PK0612_asl.nii',
                                          'PK0612_bold.nii', 'dummy.nii']])
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

    def test_advanced(self):
        self._create_tmp_files([op.join('./raw_data', f) \
                                for f in ['ASL mt_TG_PASL_s004a001.nii', 
                                          'ASL mt_TG_PASL_s008a001.nii', 
                                          'ASL mt_PK_PASL_s064a001.nii', 
                                          'ASL mt_PK_PASL_s003a001.nii']])
        src = 'ASL mt_(?P<subject>[A-Z]{2})_(?P<modality>[a-zA-Z]+)_'\
              's(?P<session>[0-9]{3})a[0-9]{3}.nii'
        src_folder = op.join(self.tmp_dir, 'raw_data')
        dest_folder = (self.tmp_dir, 'export', '{subject}', '{modality}')
        dest_basename = 'ASL_session_{session}.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder)

        for fn in [op.join(self.tmp_dir, 'export', f) \
                       for f in ['TG/PASL/ASL_session_004.nii',
                                 'TG/PASL/ASL_session_008.nii',
                                 'PK/PASL/ASL_session_064.nii', 
                                 'PK/PASL/ASL_session_003.nii']]:
            self.assert_file_exists(fn)
        

    def test_missing_tags_dest_folder(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '(?P<subject>[A-Z]{2}[0-9]{2})_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{study}', '{modality}', 
                       '{session}')
        dest_basename = '{subject}.nii'
        self.assertRaisesRegexp(MissingTagError, 
                                "Tags in dest_folder not defined in src: "\
                                "study, session",  rx_copy, 
                                src, src_folder, dest_basename, dest_folder)

    def test_missing_tags_dest_basename(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = '{subject}_{session}.nii'
        self.assertRaisesRegexp(MissingTagError, 
                                "Tags in dest_basename not defined in src: "\
                                "(subject, session)|(session, subject)",  
                                rx_copy, src, src_folder, 
                                dest_basename, dest_folder)

    def test_dry(self):
        self._create_tmp_files(['AK98_T1_s01.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+)'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = 'data.nii'
        rx_copy(src, src_folder, dest_basename, dest_folder, dry=True)
        fn = op.join(self.tmp_dir, 'export', 'T1', 'data.nii')
        if os.exists(fn):
            raise Exception('File %s should not exist' %fn)


    def test_duplicates_targets(self):
        self._create_tmp_files(['AK98_T1_s01.nii', 'AK98_T1_s02.nii'])
        src_folder = self.tmp_dir
        src = '[A-Z]{2}[0-9]{2}_(?P<modality>[a-zA-Z0-9]+).*nii'
        dest_folder = (self.tmp_dir, 'export', '{modality}')
        dest_basename = 'data.nii'
        error_msg = r'Copy is not injective, the following copy ' \
                    'operations have the same destination:\n'   \
                    '.*AK98_T1_s01\.nii\n.*AK98_T1_s02\.nii\n'      \
                    '-> .*export/T1/data.nii'
        self.assertRaisesRegexp(DuplicateTargetError, error_msg,
                                rx_copy, src, src_folder, 
                                dest_basename, dest_folder)

if __name__ == '__main__':
    unittest.main()       
