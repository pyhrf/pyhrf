# -*- coding: utf-8 -*-
import re
import os
from os.path import join

def replaceInFile(src, dest, inFile, outFile=None, dry=False):
#replace regular expression src by dest in file inFile
    if outFile is None:
        outFile = inFile

    fIn = open(inFile,"r")
    content = fIn.read()
    fIn.close()
    res = re.subn(src, dest, content)
    content = res[0]
    if res[1] > 0:
        print fn,': %d occurence(s) to replace' %res[1],
        if not dry:
            print '-> done!'
            fOut = open(outFile,"w")
            fOut.write(content)
            fOut.close()
        else:
            print ''

pyhrfRoot = '../../'
pythonRoot = '../../python'
binFolder = '../../bin'
exclude = ['refactor.py']#, '__init__.py']
rexpCC = re.compile(".*[.]cc\Z")
rexpPy = re.compile(".*[.]py\Z") #selection of python files only
rexpXml = re.compile(".*[.]xml\Z")
rexpXmi = re.compile(".*[.]xmi\Z")
rexpAll = re.compile(".*")

#fn_match = rexpPy.match
fn_match = rexpAll.match

if 1:
    for root, dirs, files in os.walk(binFolder):
    #for root, dirs, files in os.walk(pyhrfRoot):
        for f in filter(fn_match, files):
            if '.svn' not in root and f not in exclude:
                fn = join(root, f)
                replaceInFile('pyhrf.boldsynth.graph',
                              'pyhrf.graph',
                              fn, dry=False) #dry=True --> launch sciprt but
                                             # do nothing

