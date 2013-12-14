import sys
import numpy as np
from PyQt4 import QtCore, QtGui

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as \
     NavigationToolbarQt


# from matplotlib.backends.backend_qt4 import FigureCanvasQT as FigureCanvas
# from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as \
#      NavigationToolbarQt


from matplotlib.backends.qt4_compat import _getSaveFileName
from matplotlib.colors import Normalize

import pyhrf
from pyhrf.ndarray import xndarray
from pyhrf.tools import set_leaf
#from pyhrf.plot import plot_cub_as_curve, plot_cub_as_image

#from plotter_toolbar import PlotterToolbar #TODO
from plotter_toolbar_ui import Ui_PlotterToolbar
from domain_color_selector import DomainColorSelector
from mpl_cmap_editor import MplCmapEditor

import ui_base as uib


def plot_cub_as_curve(axes, c, colors=None, plot_kwargs=None, legend_prefix='',
                      show_axis_labels=True,
                      show_legend=False):
    """
    Plot a cuboid (ndims <= 2) as curve(s).
    If the input is 1D: one single curve.
    If the input is 2D:
       * multiple curves are plotted: one for each domain value on the 1st axis.
       * legends are shown to display which domain value is associated
         to which curve.

    Args:
        - colors (dict <domain value>: <matplotlib color>):
            associate domain values of the 1st axis to color curves
        - plot_kwargs (dict <arg name>:<arg value>):
            dictionary of named argument passed to the plot function
        - legend_prefix (str): prefix to prepend to legend labels.

    Return:
        None
    """
    axes = axes
    colors = colors or {}
    plot_kwargs = plot_kwargs or {}
    if c.get_ndims() == 1:
        dom = c.axes_domains[c.axes_names[0]]
        if np.issubsctype(dom.dtype, str):
            dom = np.arange(len(dom))
        axes.plot(dom, c.data, **plot_kwargs)
        axes.set_xlim(dom.min(), dom.max())
        if np.issubsctype(c.axes_domains[c.axes_names[0]], str):
            set_int_tick_labels(axes.xaxis, c.axes_domains[c.axes_names[0]],
                                rotation=30)
    elif c.get_ndims() == 2:
        for val, sub_c in c.split(c.axes_names[0]).iteritems():
            pkwargs = plot_kwargs.copy()
            col = colors.get(val, None)
            if col is not None:
                pkwargs['color'] = col
            pkwargs['label'] = legend_prefix + str(val)
            plot_cub_as_curve(axes, sub_c, plot_kwargs=pkwargs,
                              show_axis_labels=False)
        if show_legend:
            axes.legend()

    else:
        raise Exception('xndarray has too many dims (%d), expected at most 2' \
                        %c.get_ndims())

    if show_axis_labels:
        if c.get_ndims() == 1:
            axes.set_xlabel(c.axes_names[0])
        else:
            axes.set_xlabel(c.axes_names[1])
        axes.set_ylabel(c.value_label)

def set_int_tick_labels(axis, labels, fontsize=None, rotation=None):
    """
    Redefine labels of visible ticks at integer positions for the given axis.
    """
    # get the tick positions:
    tickPos = axis.get_ticklocs()#.astype(int)
    dvMax = len(labels)
    tLabels = []
    #if debug: print '%%%% tickPos :', tickPos
    for tp in tickPos:
        if tp < 0. or int(tp) != tp or tp >= dvMax:
            tLabels.append('')
        else:
            tLabels.append(labels[int(tp)])
    #if debug: print '%%%% Setting labels:', tLabels
    axis.set_ticklabels(tLabels)
    for label in axis.get_ticklabels():
        if fontsize is not None:
            label.set_fontsize(fontsize)
        if rotation is not None:
            label.set_rotation(rotation)


def plot_cub_as_image(axes, c, cmap=None, norm=None, show_axes=True,
                      show_axis_labels=True, show_tick_labels=True,
                      show_colorbar=False):
    axes = axes
    data = c.data
    if data.ndim == 1:
        data = data[:,np.newaxis]

    ms = axes.matshow(data, cmap=cmap, norm=norm)#, origin='lower')
    if show_tick_labels:
        set_int_tick_labels(axes.yaxis, c.axes_domains[c.axes_names[0]])
        if len(c.axes_domains) > 1:
            set_int_tick_labels(axes.xaxis, c.axes_domains[c.axes_names[1]])
    else:
        set_int_tick_labels(axes.yaxis,
                            [''] * len(c.axes_domains[c.axes_names[0]]))
        if len(c.axes_domains) > 1:
            set_int_tick_labels(axes.xaxis,
                                [''] * len(c.axes_domains[c.axes_names[1]]))

    if show_axis_labels:
        axes.set_ylabel(c.axes_names[0])
        if len(c.axes_domains) > 1:
            axes.set_xlabel(c.axes_names[1])
    else:
        axes.set_ylabel('')
        axes.set_xlabel('')

    if not show_axes:
        axes.set_axis_off()

    if show_colorbar:
        axes.figure.colorbar(ms)



#artist.figure.canvas.draw()

class NavigationToolbar(NavigationToolbarQt):
    """
    Wrapper for NavigationToolbar2QTAgg
    Override method save_figure so that it proposes a better default filename
    """

    def set_default_save_name(self, fn):
        """
        Args:
            - fn (list of str): list with one element encapsulating the hint
                file name. Encapsulation is used to keep a reference to a changing
                string
        """

        self.default_save_name = fn

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        default_filetype = self.canvas.get_default_filetype()

        if not hasattr(self, 'default_save_name'):
            default = 'image.'
        else:
            default = self.default_save_name[0]

        start = default + default_filetype
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname = _getSaveFileName(self, "Choose a filename to save to",
                                        start, filters, selectedFilter)
        if fname:
            try:
                self.canvas.print_figure( unicode(fname) )
            except Exception, e:
                QtGui.QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)


def is_range_domain(d):
    print 'd:', d.dtype
    if not (np.issubsctype(d.dtype, np.unicode) or \
            np.issubsctype(d.dtype, np.str)):
        delta = np.diff(d)
        return (delta == delta[0]).all()
    else:
        return False


class xndarrayPlotter(QtGui.QWidget):

    domain_colors_changed = QtCore.pyqtSignal(str, dict,
                                              name='DomainColorsChanged')
    closing = QtCore.pyqtSignal(str, name='Closing')
    focus_in = QtCore.pyqtSignal(str, name='FocusIn')

    value_label_changed = QtCore.pyqtSignal(str, name='ValueLabelChanged')
    norm_changed = QtCore.pyqtSignal(float, float, name='NormChanged')


    def __init__(self, cuboid, slice_def, orientation, cuboid_name='data',
                 parent=None):

        QtGui.QWidget.__init__(self, parent)

        self.options = {
            'view_mode' : uib.IMAGE_MODE,
            'show_axes' : True,
            'show_axis_labels' : True,
            'image' : {
                'show_colorbar' : False,
                'show_mask' : False,
                'cmaps': {},
                'norms': {},
                },
            'curve': {
                'show_legend' : False,
                'colors':{},
                'single_color': uib.DEFAULT_COLORS[0],
                }
            }
        self.cuboid_name = cuboid_name

        self.file_name_hint = [self.cuboid_name + '.']

        self.setWindowTitle('xndarray plot')
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.create_main_frame()
        self.create_color_dialogs()

        self.set_new_cuboid(cuboid, slice_def, orientation)


    def set_new_cuboid(self, c, slice_def, orientation):
        """
        Set the current cuboid, slice definition and orientation
        """
        self.slice_def = slice_def.copy()
        self.cuboid = c
        self.orientation = orientation

        # def is_range_domain(d):
        #     if not np.issubsctype(d.dtype, str):
        #         delta = np.diff(d)
        #         return (delta == delta[0]).all()
        #     else:
        #         return False

        self.is_range_domain = dict((an, is_range_domain(d)) \
                                    for an,d in self.cuboid.axes_domains.items())

        if not self.options['image']['norms'].has_key(self.cuboid.value_label):
            self.options['image']['norms'][self.cuboid.value_label] = \
              Normalize(self.cuboid.min(), self.cuboid.max())

        self.value_label_changed.emit(self.cuboid.value_label)
        self.norm_changed.emit(self.cuboid.min(), self.cuboid.max())

        self.update_current_cuboid()
        self.on_draw()


    def set_slice(self, slice_def, orientation):

        print 'set_slice ...'
        print 'recieve slice_def:', id(slice_def)
        print slice_def
        print 'current slice def:', id(self.slice_def)
        print self.slice_def

        # check that something acctually changed:
        need_update = False
        for k,v in slice_def.iteritems():
            if self.slice_def.get(k, None) != v:
                need_update = True
                break
        if orientation != self.orientation:
            need_update = True

        if need_update:
            self.slice_def = slice_def.copy()
            self.orientation = orientation
            self.update_current_cuboid()
            self.on_draw()

    def update_current_cuboid(self):
        sdef = self.slice_def.copy()
        [sdef.pop(a, None) for a in self.orientation]
        sdef.pop(self.cuboid.value_label, None)
        self.current_cuboid = self.cuboid.sub_cuboid(**sdef)
        print 'current_cuboid:', self.current_cuboid
        if np.isscalar(self.current_cuboid):
            self.current_cuboid = xndarray(np.array([self.current_cuboid]),
                                         axes_names=[''])
            self.orientation = self.current_cuboid.axes_names
        self.current_cuboid.set_orientation(self.orientation)
        doms = self.current_cuboid.axes_domains
        self.current_domains = [doms[an] for an in self.orientation]
        self.file_name_hint[0] = '_'.join([self.cuboid_name] + \
            ['_'.join([k,str(v)]) for k,v in sdef.items()]) + '.'

        self.domain_colors_changed.emit(*self.get_current_domain_colors())

    def closeEvent(self, event):
        """
        Override cloveEvent to emit a signal just before the widget is actually
        closed.
        """
        self.closing.emit(self.cuboid_name)
        QtGui.QWidget.closeEvent(self, event)

    def focusInEvent(self, event):
        """
        Override focusInEvent to emit a signal just before the widget actually
        gains focus.
        """
        self.focus_in.emit(self.cuboid_name)
        QtGui.QWidget.focusInEvent(self, event)

    def on_mouse_clicked(self, event):
        uib.SliceEventDispatcher().dispatch(self.slice_def)

    def on_mouse_move(self, event):
        """
        Handle mouse move event. Update current slice definition according
        to the position of the mouse.
        """
        def get_dval_from_pos(pos, dom_idx):
            """
            Translate a plot position into a domain value from the domain
            defined by dom_idx
            Args:
                - pos (float): the plot coordinate
                - dom_idx (int): the index of the domain

            Return:
                DomainValue (float|string), value index (int)

            """
            if self.orientation[dom_idx] == '': #no dims
                return 0,0
            nb_dom_vals = len(self.current_domains[dom_idx])
            if self.is_range_domain[self.orientation[dom_idx]]:
                search_in = self.current_domains[dom_idx]
            else: #search by major tick position -> integers
                search_in = np.arange(nb_dom_vals)
            dv_ix = (np.abs(search_in-pos)).argmin()
            # ix = search_in.searchsorted([pos])[0]
            # dv_ix  = int(np.floor(search_in[min(ix,nb_dom_vals-1)]))
            return  self.current_domains[dom_idx][min(dv_ix,nb_dom_vals-1)],\
              dv_ix

        x, y = event.xdata, event.ydata

        if x is None or y is None:
            return

        if self.options['view_mode'] == uib.CURVE_MODE:
            if self.current_cuboid.get_ndims() == 1:
                #x axis is mapped to domain values of 1st axis
                dv, dv_idx = get_dval_from_pos(x, 0)
                self.slice_def[self.orientation[0]] = dv
                val = self.current_cuboid.data[dv_idx]
            else:
                #x axis is mapped to domain values of 2nd axis
                dv = get_dval_from_pos(x, 1)
                self.slice_def[self.orientation[1]] = dv
                val = 'N/A'

        elif self.options['view_mode'] == uib.IMAGE_MODE:
            #y axis is mapped to domain values of 1st axis
            dv, dv_i = get_dval_from_pos(y, 0)
            self.slice_def[self.orientation[0]] = dv
            if self.current_cuboid.get_ndims() == 1:
                val = self.current_cuboid.data[dv_i]
            elif self.current_cuboid.get_ndims() == 2:
                dv, dv_j = get_dval_from_pos(x, 1)
                self.slice_def[self.orientation[1]] = dv
                val = self.current_cuboid.data[dv_i, dv_j]

        self.slice_def[self.cuboid.value_label] = val
        slice_items = self.slice_def.items()
        info = ' '.join('%s:%s'%(an,uib.format_dval(dval)) \
                        for an,dval in slice_items)
        self.slice_info.setText(info)


    def on_draw(self):
        """
        Redraws the figure
        """
        print 'on_draw!!'
        print 'options show_colorbar:'
        print self.options['image']['show_colorbar']

        if pyhrf.verbose.verbosity > 5:
            pyhrf.verbose(6, 'on_draw ..., cubdoid: %s' %self.cuboid.descrip())

        # Clear axes and create new subplot
        #self.fig.delaxes(self.axes)
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        #self.axes.clear()

        if self.options['view_mode'] == uib.IMAGE_MODE:
            options = self.options['image']
            print 'self.cuboid.value_label:', self.cuboid.value_label
            cm = options['cmaps'].get(self.cuboid.value_label)
            norm = options['norms'].get(self.cuboid.value_label)
            plot_cub_as_image(self.axes, self.current_cuboid, cmap=cm, norm=norm,
                              show_axis_labels=self.options['show_axis_labels'],
                              show_colorbar=options['show_colorbar'])

        else: #curve mode
            options = self.options['curve']
            single_curve_color = options['single_color']
            plot_cub_as_curve(self.axes, self.current_cuboid,
                              colors=self.get_current_domain_colors()[1],
                              show_legend=options['show_legend'],
                              show_axis_labels=self.options['show_axis_labels'],
                              plot_kwargs={'color':single_curve_color})

        if not self.options['show_axes']:
            self.axes.set_axis_off()

        self.canvas.draw()


    def get_current_domain_colors(self):
        d0 = self.current_domains[0]
        if not self.options['curve']['colors'].has_key(self.orientation[0]):
            dcols = dict(zip(d0, uib.get_default_color_list(len(d0))))
            self.options['curve']['colors'][self.orientation[0]] = dcols

        return (self.orientation[0],
                self.options['curve']['colors'][self.orientation[0]])


    def create_color_dialogs(self):

        d = self.domain_color_dialog = DomainColorSelector()
        d.domain_color_changed.connect(self.set_domain_color)
        self.domain_colors_changed.connect(d.set_domain_colors)

        # norm = self.options['image']['norms'][self.cuboid.value_label]
        # self.cmap_dialog = MplCmapEditor(self.cuboid.value_label,
        #                                  norm.vmin, norm.vmax)
        self.cmap_dialog = MplCmapEditor()
        self.cmap_dialog.range_value_changed.connect(self.set_normalize)
        self.cmap_dialog.cmap_changed.connect(self.set_cmap)
        self.value_label_changed.connect(self.cmap_dialog.set_value_label)
        self.norm_changed.connect(self.cmap_dialog.set_boundaries)

        scolor = self.options['curve']['single_color']
        self.single_color_dialog = QtGui.QColorDialog(QtGui.QColor(scolor))
        ssc = self.set_single_curve_color
        self.single_color_dialog.currentColorChanged.connect(ssc)


    def set_cmap(self, aname, cmap):
        self.options['image']['cmaps'][str(aname)] = cmap
        self.on_draw()

    def set_normalize(self, aname, min_val, max_val):
        self.options['image']['norms'][str(aname)] = Normalize(min_val, max_val)
        self.on_draw()

    def set_domain_color(self, aname, dval, color):
        self.options['curve']['colors'][str(aname)][dval] = color
        self.on_draw()

    def set_single_curve_color(self, color):
        self.options['curve']['single_color'] = uib.qcolor_to_mpl_rgb(color)
        self.on_draw()

    def create_main_frame(self):
        self.main_frame = self

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)

        self.canvas.mpl_connect('button_press_event', self.on_mouse_clicked)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        self.mpl_toolbar.set_default_save_name(self.file_name_hint)

        # Other GUI controls
        #
        # self.textbox = QtGui.QLineEdit()
        # self.textbox.setMinimumWidth(200)
        # self.textbox.editingFinished.connect(self.on_draw)

        self.toolbar = PlotterToolbar(self.options, parent=self)
        #self.toolbar.option_changed.connect(self.set_option)
        #self.toolbar.color_button_clicked.connect(self.show_color_dialog)
        #
        # Layout with box sizers
        #
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.toolbar)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)

        self.top = QtGui.QFrame(self)
        self.top.setLayout(vbox)

        self.bottom = QtGui.QFrame(self)
        self.bottom.setLayout(hbox)

        self.splitter = QtGui.QSplitter(QtCore.Qt.Vertical)

        self.splitter.addWidget(self.top)
        self.splitter.addWidget(self.bottom)
        #vbox.addWidget()
        #vbox.addLayout(hbox)
        #self.main_frame.addWidget(splitter)
        self.mainbox = QtGui.QVBoxLayout()
        self.mainbox.addWidget(self.splitter)
        self.slice_info = QtGui.QLabel('')
        self.mainbox.addWidget(self.slice_info)
        self.main_frame.setLayout(self.mainbox)
        #self.setCentralWidget(self.main_frame)

    def show_color_dialog(self):
        if self.options['view_mode'] == uib.IMAGE_MODE:
            self.domain_color_dialog.hide()
            self.single_color_dialog.hide()
            self.cmap_dialog.show()
        if self.options['view_mode'] == uib.CURVE_MODE:
            self.cmap_dialog.hide()
            if len(self.orientation) == 1:
                self.domain_color_dialog.hide()
                self.single_color_dialog.show()
            else: #2D
                self.domain_color_dialog.show()
                self.single_color_dialog.hide()


    def switch_color_dialog(self):
        if self.domain_color_dialog.isVisible() or \
            self.single_color_dialog.isVisible() or \
            self.cmap_dialog.isVisible():
          self.show_color_dialog()

    @QtCore.pyqtSlot(tuple, object)
    def set_option(self, option_def, option_value):
        set_leaf(self.options, option_def, option_value)
        if option_def == ('view_mode',):
            self.switch_color_dialog()

        self.on_draw()

class PlotterToolbar(QtGui.QWidget):

    def __init__(self, options, parent):
        QtGui.QWidget.__init__(self, parent)

        self.options = options
        self.ui = Ui_PlotterToolbar()
        self.ui.setupUi(self)



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    if 1: #3D case
        sh = (10,10,3)
        conds = np.array(['audio1','audio2', 'video'])
        c1 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['sagittal','coronal', 'condition'],
                    axes_domains={'condition':conds})

        cp = xndarrayPlotter(c1, {'sagittal':0,
                                 'condition':'video',
                                 'coronal':0},
                                 ['condition','coronal'])
        cp.show()
    else:
        sh = (3,) #1D case
        conds = np.array(['audio1','audio2', 'video'])
        c2 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['condition'],
                    axes_domains={'condition':conds})

        cp = xndarrayPlotter(c2, {'condition':'audio2'}, ['condition'])
        cp.show()

    cp.closing.connect(uib.SignalPrinter('plotter closing'))
    cp.focus_in.connect(uib.SignalPrinter('plotter gained focus'))

    if 0: #save image of widget non interactively
        uib.render_widget_in_doc(cp)
    else:
        sys.exit(app.exec_())


