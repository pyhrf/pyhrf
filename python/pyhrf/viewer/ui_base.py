import pyhrf
import os.path as op
from collections import defaultdict
import numpy as np

from PyQt4 import QtCore, QtSvg

# DomainValue is one of:
#    - str
#    - int
#    - float
DomainValue = object

# ViewMode is one of:
# - CURVE_MODE
# - IMAGE_MODE
# the view mode for cuboid slice plots
CURVE_MODE = 'curve'
IMAGE_MODE = 'image'

RgbColor = tuple

doc_sphinx_fig_path = op.join(pyhrf.get_src_doc_path(), 'figs/pyhrf_view')

from PyQt4 import QtGui

def render_widget(w, fig_filename=None, svg_resolution=100):
    """
    Save the given widget *w* as an image.
    Used eg to render UI stuffs on-the-fly when doc is generated (no need to
    take screenshots)
    Args:
        - w: Qt widget
        - fig_filename (str): figure filename to save rendering [png, jpg, svg]
        - svg_resolution (int): resolution in dpi to use for svg rendering

    Return: None
    """
    #QPixmap pixmap(rectangle->size());

    if fig_filename is None:
        fig_filename = op.join(doc_sphinx_fig_path, './%s.png' \
                               %w.__class__.__name__)

    print '%s widget saved to:' %w.__class__.__name__, fig_filename
    ext = op.splitext(fig_filename)[1]
    if ext == '.png' or ext == '.jpg':
        fmt = ext[1:]
        p = QtGui.QPixmap.grabWidget(w)
        p.save(fig_filename, fmt)
    elif ext == '.svg':
        generator = QtSvg.QSvgGenerator()
        generator.setResolution(svg_resolution)
        generator.setFileName(fig_filename)
        painter = QtGui.QPainter()
        painter.begin(generator)
        w.render(painter)
        painter.end()
    else:
        raise Exception('Unhandled file extension: "%s"' %ext)

class Singleton(object):
    __state = {}

    def __new__(cls, *args, **kwargs):
        """ shares the state """
        parent = super(Singleton, cls)
        ob = parent.__new__(cls, *args, **kwargs)
        ob.__dict__ = cls.__state
        return ob


class SignalPrinter:
    def __init__(self, msg):
        self.msg = msg

    def __call__(self, *args):
        print self.msg + ' ' + ','.join([str(a) for a in args])


class SliceEventDispatcher(): #Singleton):

    slicers = defaultdict(list)

    # def __init__(self):
    #     Singleton.__init__(self)
    #     self.slicers = defaultdict(list)

    def register_slicer(self, aname, slicer):
        """
        Connect the given slicer to all other slicers associated with the
        same axis name.
        """
        for s in self.slicers[aname]:
            s.slice_value_changed.connect(slicer.set_slice_value)
            slicer.slice_value_changed.connect(s.set_slice_value)

        self.slicers[aname].append(slicer)

    def unregister_slicer(self, aname, slicer):
        """
        Disconnect the given slicer to all other slicers associated with the
        same axis name.
        """
        self.slicers[aname].remove(slicer)

        for s in self.slicers[aname]:
            s.slice_value_changed.disconnect(slicer.set_slice_value)
            slicer.slice_value_changed.disconnect(s.set_slice_value)


    def dispatch(self, slice_def):
        print 'dispatch slice_def:', slice_def
        for aname, dvalue in slice_def.iteritems():
            for slicer in self.slicers[aname]:
                slicer.set_slice_value(aname, dvalue)

def test():
    class Dummy(QtCore.QObject):
        slice_value_changed = QtCore.pyqtSignal(name='SliceValueChanged')

        @QtCore.pyqtSlot()
        def set_slice_value(self):
            pass

    d1 = Dummy()
    d2 = Dummy()
    d3 = Dummy()

    SliceEventDispatcher().register_slicer('a',d1)
    SliceEventDispatcher().register_slicer('a',d2)
    SliceEventDispatcher().register_slicer('b',d3)

    assert SliceEventDispatcher().slicers == {'a':[d1, d2]}

DEFAULT_COLORS = ['black', 'blue', 'red','green','orange', 'purple', 'cyan']

def get_default_color_list(size):
    l = DEFAULT_COLORS[:size]
    if len(l) < size:
        l += [tuple(a) for a in np.random.rand(size-len(l), 3)]
    return l

def qcolor_to_mpl_rgb(col):
    return tuple( c/255. for c in col.getRgb())

def format_dval(v):
    if isinstance(v, str):
        return v
    elif isinstance(v, float):
        return '%1.4f' %v
    elif isinstance(v, int):
        return '%d' %v
    elif np.issubdtype(v.dtype, np.float):
        return '%1.4f' %v
    elif np.issubdtype(v.dtype, np.int) or np.issubdtype(v.dtype, np.uint8):
        return '%d' %v
    else:
        raise Exception("unsupport type of domain value: %s (val=%s)" \
                        %(str(type(v)), str(v)))
