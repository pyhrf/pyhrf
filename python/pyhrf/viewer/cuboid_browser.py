# -*- coding: utf-8 -*-
import sys
import numpy as np
from PyQt4 import QtCore, QtGui

from pyhrf.ndarray import xndarray

from cuboid_browser_ui import Ui_xndarrayBrowser
from axis_slicer import AxisSlicer

from ui_base import DomainValue

#pyuic4 cuboid_browser.ui -o cuboid_browser_ui.py

class LimitedFifo(list):
    """ Implement a First-In First-Out container with a limited capacity.
    If an item is added beyond max capacity, then the first added item is
    removed
    """

    def __init__(self, max_size, on_pop=None):
        """
        Create a LimitedFifo instance with max_size as the limited capacity.
        A callback can be called when an item is removed (see arg on_pop)

        Args:
            - max_size (int): the maximum capacity of the FIFO
            - on_pop (function): the function to apply on a removed item.

        """
        self.max_size = max_size
        self.on_pop = on_pop

    def append(self, o):
        list.append(self,o)
        if self.__len__() > self.max_size:
            self.pop(0)

    def pop(self, i):
        e = self.__getitem__(i)
        list.pop(self, i)
        if self.on_pop is not None:
            self.on_pop(e)

    def swap(self, i, j):
        """
        Swap the two elements defined by the given indexes, if they exist.
        Produce True if swapping occured, else False

        Args:
            - i (int>=0): the index of the first item to swap
            - j (int>=0): the index of the second item to swap

        Return:
            bool -> True if swapping occured, else False
        """
        if i < len(self) and j < len(self):
            self[i], self[j] = self[j], self[i]
            return True
        return False

class xndarrayBrowser(QtGui.QWidget):
    """
    Signals:
         - slice_changed(dict, tuple): emitted when the current slice changed                  Args are:
                 - (dict of <axis name (str)>:<DomainValue>): slice value for
                     each axis
                 - (tuple of <axis name (str)>): list of projected axes

    """

    slice_changed = QtCore.pyqtSignal(dict, tuple, name='SliceChanged')
    # projected_axes_changed = QtCore.pyqtSignal(name='ProjectedAxesChanged')
    # projected_axes_swapped = QtCore.pyqtSignal(name='ProjectedAxesSwapped')
    closing = QtCore.pyqtSignal(name='Closing')

    def __init__(self, cuboid, name, parent=None):
        """
        Args:

        """
        QtGui.QWidget.__init__(self, parent)

        self.cuboid_name = name
        self.ui = Ui_xndarrayBrowser()
        self.ui.setupUi(self)

        domains = cuboid.get_axes_domains()
        anames = cuboid.axes_names
        self.slicers = dict((an,AxisSlicer(an, domains[an])) \
                            for an in anames)

        self.slice_def = dict((an, domains[an][0]) for an in anames)

        for an in anames:
            slicer = self.slicers[an]
            self.ui.verticalLayout.addWidget(slicer)
            slicer._slice_value_changed.connect(self.handle_slice_value_changed)
            slicer.axis_state_changed.connect(self.set_axis_state)

        self.selection_fifo_max_size = 2
        self.selection_fifo = LimitedFifo(self.selection_fifo_max_size,
                                          on_pop=self.deselect_axis)

        checked = QtCore.Qt.Checked
        self.slicers[anames[0]].set_axis_selection_checkbox_state(checked)
        if len(anames) > 1:
            self.slicers[anames[1]].set_axis_selection_checkbox_state(checked)

        self.ui.swap_button.clicked.connect(self.swap_current_axes)


    def set_new_cuboid(self, c):
        """
        Set the current cuboid to c, while trying to maintain the current slice
        definition
        """
        print 'todo: set_new_cuboid:'
        print c.descrip()
        print 'current cuboid is:'
        print c.descrip()

    def get_slice_def(self):
        return self.slice_def

    def set_slice_value(self, axis, value):
        self.slicers[axis].set_slice_value(axis, value)

    def get_current_axes(self):
        return tuple(str(a) for a in self.selection_fifo)

    @QtCore.pyqtSlot(str, DomainValue)
    def handle_slice_value_changed(self, aname, dval):
        self.slice_def[str(aname)] = dval
        self.slice_changed.emit(self.slice_def, self.get_current_axes())

    @QtCore.pyqtSlot(str, bool)
    def set_axis_state(self, axis_name, state):
        print 'set_axis_state:', axis_name, '->', state
        print 'current selection_fifo:', self.selection_fifo
        state_changed = False
        if not state:
            if axis_name in self.selection_fifo:
                self.selection_fifo.remove(axis_name)
                state_changed = True
        else:
            if axis_name not in self.selection_fifo:
                self.selection_fifo.append(axis_name)
                state_changed = True

        if state_changed:
            self.update_current_axis_labels()

            print 'emit slice_changed:', self.slice_def, '|| ca:', self.get_current_axes()
            self.slice_changed.emit(self.slice_def, self.get_current_axes())

    def update_current_axis_labels(self):
        if len(self.selection_fifo) == 0:
            self.ui.label_axis_1.setText('')
            self.ui.label_axis_2.setText('')
        elif len(self.selection_fifo) == 1:
            self.ui.label_axis_1.setText(self.selection_fifo[0])
            self.ui.label_axis_2.setText('')
        else:
            self.ui.label_axis_1.setText(self.selection_fifo[0])
            self.ui.label_axis_2.setText(self.selection_fifo[1])

    def deselect_axis(self, axis_name):
        unchecked = QtCore.Qt.Unchecked
        self.slicers[str(axis_name)].set_axis_selection_checkbox_state(unchecked)

    def swap_current_axes(self):
        if self.selection_fifo.swap(0,1):
            self.slice_changed.emit(self.slice_def, self.get_current_axes())
            self.update_current_axis_labels()


    def closeEvent(self, event):
        """
        Override cloveEvent to emit a signal just before the widget is actually
        closed.
        """
        self.closing.emit()
        QtGui.QWidget.closeEvent(self, event)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    sh = (10,10,5,3)
    c1 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                axes_names=['sagittal','coronal','axial','condition'],
                axes_domains={'condition':['audio1','audio2', 'video']})

    sh = (10,10,5,4)
    c2 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                axes_names=['sagittal','coronal','axial','condition'],
                axes_domains={'condition':['video','sentence',
                                           'audio2','audio1']})

    cb1 = xndarrayBrowser(c1, 'c1')
    cb2 = xndarrayBrowser(c2, 'c2')
    cb1.show()
    cb2.show()

    from ui_base import SignalPrinter
    cb1.slice_changed.connect(SignalPrinter('cb1 slice changed ->'))
    cb2.slice_changed.connect(SignalPrinter('cb2 slice changed ->'))
    cb1.closing.connect(SignalPrinter('cb1 closing'))
    cb2.closing.connect(SignalPrinter('cb2 closing'))

    sys.exit(app.exec_())
