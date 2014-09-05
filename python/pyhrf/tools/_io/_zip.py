

import os
import gzip as _gzip

__all__ = ['gunzip', 'gzip_file']


def gunzip(gzFileName, outFileName=None):
    """
    Gunzip the file 'gzFileName' and store the results in 'outFileName'
    """
    if outFileName is None:
        outFileName = os.path.splitext(gzFileName)[0]
    fgz = _gzip.open(str(gzFileName),'rb')
    content = fgz.readlines()
    fgz.close()
    fOut = open(outFileName, 'w')
    fOut.writelines(content)
    fOut.close()
    
def gzip_file(fileName, outFileName=None):
    if outFileName is None:
        outFileName = fileName + '.gz'
    f = open(str(fileName))
    content = f.readlines()
    f.close()
    fOut = _gzip.open(outFileName, 'w')
    fOut.writelines(content)
    fOut.close()
