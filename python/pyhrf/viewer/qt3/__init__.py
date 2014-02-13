import numpy as np
from pyhrf.ndarray import xndarray, MRI3Daxes, MRI4Daxes

debug = False

class xndarrayView:

    def __init__(self, cuboid, current_axes=None):
        self.cuboid = cuboid
        if current_axes is None: # default current axes: first axes by index
            if self.cuboid.has_axis('time'):
                current_axes = ['time']
            elif self.cuboid.has_axes(MRI3Daxes):
                current_axes = ['coronal', 'sagittal']
                #current_axes = ['coronal', 'axial']
            elif self.cuboid.get_ndims() > 1:
                current_axes = self.cuboid.axes_names[:2]
            else:
                current_axes = self.cuboid.axes_names[:1]

        if debug: print 'cuboidview.__init__: call setview ...'
        if debug: print ' -> ca = ', current_axes

        self.mask_view = None
        self.mask_name = None
        self.maskLabels = None

        self.setView(current_axes)

    def setView(self, current_axes):

        if not ( type(current_axes)==list or type(current_axes)==tuple ) :
            # Then it is a single element:
            current_axes = [current_axes]

        assert set(current_axes).issubset(self.cuboid.axes_names)
        self.current_axes = current_axes

        self.slices = dict( (n,0) for n in self.cuboid.axes_names )

        if self.mask_view is not None:
            self.mask_view.setView(current_axes)


    def getView(self):
        if debug: 'xndarray.getView ...'
        cur_slices = dict((a,s) for a,s in self.slices.iteritems() \
                          if a not in self.current_axes)
        print 'slices:', cur_slices
        dslices = dict((a,self.cuboid.axes_domains[a][s]) \
                           for a,s in cur_slices.iteritems() \
                           if a in self.cuboid.axes_names)
        # currentSlices = {}
        # for ia in xrange(self.data.ndim):
        #     if ia not in self.currentAxes:
        #         d = self.cuboid.getAxisDomain(ia)
        #         currentSlices[self.cuboid.getAxisName(ia)] = d[self.slices[ia]]

        print 'self.current_axes:', self.current_axes
        c = self.cuboid.sub_cuboid_from_slices(orientation=self.current_axes,
                                               **cur_slices)
        return c, dslices

    def setSlice(self, axis, slice_idx):
        if axis in self.cuboid.axes_names:
            self.slices[axis] = slice_idx
            if self.mask_view is not None:
                self.mask_view.setSlice(axis, slice_idx)

    def swapAxes(self, a1, a2):
        ca = self.current_axes
        a, b = ca.index(a1), ca.index(a2)
        ca[b], ca[a] = ca[a], ca[b]
        if self.mask_view is not None:
            self.mask_view.swapAxes(a1, a2)

    def applyMask(self, mask, mask_name='mask'):
        if debug: print 'xndarrayview.Applymask ...'
        if set(mask.axes_names).issubset(self.cuboid.axes_names):
            self.mask_view = xndarrayView(mask, self.current_axes)
            self.mask_view.slices = dict((a,s) for a,s in self.slices.items()\
                                             if mask.has_axis(a))
            self.mask_name = mask_name

        else:
            raise Exception('Mask axes (%s) are not a subset of '\
                                'cuboid axes (%s)' \
                                %(','.join(mask.axes_names),
                                  ','.join(self.cuboid.axes_names)))

    def getMaskName(self):
        return self.mask_name

    def getMaskLabels(self):
        if self.mask_view is None:
            return None
        if self.maskLabels is None:
            self.maskLabels = np.unique(np.concatenate( \
                    (self.mask_view.cuboid.data.flatten(), []))).astype(int)
        return self.maskLabels

    def get_mask_view(self):
        if self.mask_view is None:
            return (None, None), None
        else:
            return self.mask_view.getView(), self.mask_name


def view(data, axesNames=None, axesDomains=None, errors=None,
         valueLabel='value', mask=None, maskAxes=None, maskName='mask'):

    """
    Interactively browse and display a n-dimensional numpy array. Axes can be
    labeled with a name in 'axesNames' and associated to real values in
    'axesDomains'. Errors data with the same data type and shape as 'data' can
    be given with 'errors' argument.
    Example :
    from numpy import *
    import ndview
    a = array([ [4,5,6],[8,10,12] ])
    names = ['time','position']
    domains = {'time':[0.1, 0.2]}
    ndview.view(a, axesNames=names, axesDomains=domains)
    """
    if data.dtype == np.int16:
        data = data.astype(np.int32)
    if not isinstance(data, xndarray):
        c = xndarray(data, errors=errors, axesDomains=axesDomains,
                   axesNames=axesNames,
                   value_label=valueLabel)
    else:
        c = data
    viewxndarray(c, mask=mask, maskName=maskName, maskAxes=maskAxes)





def viewxndarray(cuboid, mask=None, maskName='mask'):
    """
    Interactively browse and display a n-dimensional numpy array encapsulated in
    a xndarray object.
    Example :
    from numpy import *
    import ndview
    a = array([ [4,5,6],[8,10,12] ])
    names = ['time','position']
    domains = {'time':[0.1, 0.2]}
    c = xndarray(a, axesNames=names, axesDomains=domains)
    ndview.view(c)
    """

    from gui import browser
    from gui import display

    import qt

    #except ImportError , e :
    #   print 'Error while import:'
    #   print repr(e)

    v = xndarrayView(cuboid)
    if mask is not None:
        v.applyMask(mask, mask_name=maskName)
    if qt.qAppClass() is None:
        app = qt.QApplication([''])
    else:
        app = qt.qApp
    qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app,qt.SLOT("quit()"))

    fRenderer = display.xndarrayViewRenderer(v, '')
    fBrowser = browser.AxisBrowser(v)

    qt.QObject.connect(fBrowser, qt.PYSIGNAL("cuboidViewChanged"),
                    fRenderer.updateView)
    qt.QObject.connect(fBrowser, qt.PYSIGNAL("maskLabelChanged"),
                    fRenderer.setMaskLabel)
    qt.QObject.connect(fBrowser, qt.PYSIGNAL("windowClosed"),
                    fRenderer.close)
    qt.QObject.connect(fRenderer, qt.PYSIGNAL("windowClosed"),
                    fBrowser.close)
    app.setMainWidget(fBrowser)
    fBrowser.show()
    fRenderer.show()
    app.exec_loop()


def multiView(cuboids, mask=None, maskName='ROI'):


    if len(cuboids) == 1:
        viewxndarray(cuboids.values()[0], mask=mask, maskName=maskName)
        return

    from gui import browser
    try:
        import qt
    except ImportError:
        raise ImportError('Pyqt v3 (deprecated) is required to use viewer')

    print 'cuboids:', cuboids
    cuboidViews = dict( (n,xndarrayView(c)) for n,c in cuboids.iteritems())
    if mask is not None:
        for n,cv in cuboidViews.iteritems():
            try:
                cv.applyMask(mask, mask_name=maskName)
            except Exception:
                print 'Mask could not be applied to %s' %n


    for nv,view in cuboidViews.iteritems():
        if 'roi_mask' in nv:
            view.value_label = 'ROI'
        # if view.has_axis('time'):
        #     view.setView(view.MODE_1D, ['time'])
        # elif view.cuboid.has_axis('coronal') and view.cuboid.has_axis('sagittal'):
        #     view.setView(view.MODE_2D, ['coronal','sagittal'])


    app = qt.QApplication([''])
    qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app,qt.SLOT("quit()"))

    fObjBrowser = browser.xndarrayViewLister(cuboidViews)

    app.setMainWidget(fObjBrowser)

    fObjBrowser.show()
    app.exec_loop()

##     def multiView(cuboidViews):
##         pass
##     def view(data, axesNames=None, axesDomains=None, errors=None,
##              valueLabel='value', mask=None, maskAxes=None, maskName='mask'):
##         pass
##     def viewxndarray(cuboid, mask=None, maskAxes=None, maskName='mask'):
##         pass
