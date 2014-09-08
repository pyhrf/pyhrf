# -*- coding: utf-8 -*-
import re

licence = """\
"""

def fix_licence_txt(fn, dry=True, verbose=False):
    content = file(fn).read()

    #lic_re = re.compile(".*(# LICENCE BEGIN.*# LICENCE END).*")

    lcontent = content.split('\n')
    if '# LICENCE BEGIN' in lcontent and '# LICENCE END' in lcontent:
        lic_start = lcontent.index('# LICENCE BEGIN')
        lic_end = lcontent.index('# LICENCE END')

        new_content = '\n'.join(lcontent[:lic_start] + \
                                lcontent[lic_end+1:])
    else:
        new_content = content

    if verbose:
        print 'fn:', fn
        print 'old content:'
        print content
        print ''
        print 'new content:'
        print new_content

    if not dry:
        fout = open(fn, 'w')
        fout.write(new_content)
        fout.close()

import os
import os.path as op
rexpPy = re.compile(".*[.]py\Z") #selection of python files only
pyhrfRoot = '../../'
pythonRoot = '../../python'
binFolder = '../../bin'

if 1:
    #for root, dirs, files in os.walk(binFolder):
    for root, dirs, files in os.walk(pyhrfRoot):
        #for f in filter(rexpAll.match, files):
        for f in filter(rexpPy.match, files):
        #for f in files:
            if '.svn' not in root:
                fn = op.join(root, f)
                fix_licence_txt(fn, dry=False, verbose=False)

else:
    fix_licence_txt('../../python/pyhrf/configuration.py', dry=True, verbose=True)
