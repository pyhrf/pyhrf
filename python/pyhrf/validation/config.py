

import pyhrf

savePlots = False
plotSaveDir = './'
figext = 'svg' #'png'
useCache = False
cacheDir = '/tmp'

def figfn(fn):
    """ Append the figure file extension to fn
    """
    print 'figext:',figext
    return fn + '.' + figext

def is_tmp_file(fn):
    return fn.startswith(pyhrf.tmpPrefix)

def clean_cache():
    tmpFiles = filter(is_tmp_file, os.listdir(cacheDir))
    for f in tmpFiles:
        os.remove(os.path.join(cacheDir, f))


    
