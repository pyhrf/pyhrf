from qt import *
from display import *
from browser import *
import sys
sys.path.append('../')
from pyhrf.ndarray import *


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QObject.connect(app,SIGNAL("lastWindowClosed()"),app,SLOT("quit()"))
    
    
    import numpy as _N
    import cPickle
    i = cPickle.load(open('../../Data/pmNrlTest.pyd'))
    i = i[0]
    roiMask = cPickle.load(open('../../Data/roiMask.pyd'))
    #a = _N.arange(2*3*4.*5).reshape((2,3,4,5))/4.
    #dimNamesA = ['x','y','z','t']
    #c = xndarray(a, axesNames=dimNamesA,axesDomains={'t':_N.arange(0.,0.5,0.1)})
    #v = xndarrayView(c)
    #v.setView(v.MODE_1D, ['t'])
    print i.shape
    dimNames = ['axial','coronal','sagittal']
    c = xndarray(i, axesNames=dimNames, valueLabel='pmNRL')
    v = xndarrayView(c, mode=xndarrayView.MODE_2D,
                   currentAxes=['sagittal','coronal'])

    v.applyMask(xndarray(roiMask, axesNames=['axial', 'coronal', 'sagittal']),
                maskName='ROI')
    print 'building renderer'
    fRenderer = xndarrayViewRenderer(v)
    print 'building axis browser'
    fBrowser = AxisBrowser(v)
    
    QObject.connect(fBrowser, PYSIGNAL("cuboidViewChanged"),
                    fRenderer.updateView)
    QObject.connect(fBrowser, PYSIGNAL("maskLabelChanged"),
                    fRenderer.setMaskLabel)
    app.setMainWidget(fBrowser)
    fBrowser.show()
    fRenderer.show()
    app.exec_loop()
