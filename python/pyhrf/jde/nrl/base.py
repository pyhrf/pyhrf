# -*- coding: utf-8 -*-


import os

dataPath = os.getenv('PYHRFPATH', None)
if dataPath != None:
    DEFAULT_CONTRAST = 'audio-video;'
else:
    DEFAULT_CONTRAST = ';'
