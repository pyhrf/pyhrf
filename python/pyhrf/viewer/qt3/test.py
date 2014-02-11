import ndview
import os, sys, cPickle

import unittest
from pyhrf.ndarray import *
from gui.browser import *
from gui.display import *

import numpy as _N

class GuiTest(unittest.TestCase):

    def testViewer(self):

        ifn = '../../../data/pmNrlTest2.pyd'
        if not os.path.exists(ifn):
            print ifn, 'must be uncompressed first'
            sys.exit(1)
        i = cPickle.load(open(ifn))
        i = i[0]
    
        mfn = '../../../data/roiMask.pyd'
        if not os.path.exists(mfn):
            print mfn, 'must be uncompressed first'
            sys.exit(1)
        roiMask = cPickle.load(open(mfn))
        dimNames = ['axial','coronal','sagittal']
        ndview.view(i, axesNames=dimNames, valueLabel='pmNrls',
                    mask=roiMask, maskAxes=dimNames, maskName='ROI')
    
    def testLister(self):

        a1 = _N.arange(2*3*4).reshape(2,3,4)
        c1 = xndarray(a1)
        v1 = xndarrayView(c1)
        a2 = _N.arange(4*8).reshape(4,8)
        v2 = xndarrayView(xndarray(a2))
        

        a = QApplication([])
        QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"))
        lister = xndarrayViewLister({'v1':v1,'v2':v2})
        a.setMainWidget(lister)
        lister.show()
        a.exec_loop()

class xndarrayViewTester(unittest.TestCase):

    def testGetAllViews(self):
        a1 = _N.arange(2*3*4*6).reshape(2,3,4,6)
        axes = ['d0','d1','d2','d3']
        c1 = xndarray(a1, axesNames=axes)
        v1 = xndarrayView(c1, mode=xndarrayView.MODE_3D,
                        currentAxes=['d2','d1','d0'])
        vols = v1.getAllViews()
        print 'allViews :'
        print vols
        
class xndarrayTester(unittest.TestCase):

    def testJoin(self):

        a0 = _N.arange(2*3*4).reshape(2,3,4)
        axes = ['a0', 'a1', 'a2']
        c0 = xndarray(a0, axesNames=axes)
        print 'c0 :'
        print c0.descrip()
        
        a1 = _N.arange(3*5).reshape(3,5)
        axes = ['a1', 'b1']
        c1 = xndarray(a1, axesNames=axes)
        print 'c1:'
        print c1.descrip()

        jc = joinxndarrays([c0,c1], names=['c0', 'c1'])
        print 'jc:'
        print jc.descrip()
        
        cv0 = xndarrayView(c0)
        cv1 = xndarrayView(c1)
        cvj = xndarrayView(jc)

        ndview.multiView({'cv0':cv0,'cv1':cv1,'cvj':cvj})
        
if __name__ == "__main__":
    unittest.main()
