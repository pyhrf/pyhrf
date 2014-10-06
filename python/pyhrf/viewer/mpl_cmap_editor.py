# -*- coding: utf-8 -*-
"""

"""
import sys
from PyQt4 import QtCore, QtGui

import matplotlib.cm
import matplotlib.colors
from mpl_cmap_editor_ui import Ui_MplCmapEditor

#pyuic4 mpl_cmap_editor.ui -o mpl_cmap_editor_ui.py

ColorMap = object

class MplCmapEditor(QtGui.QWidget):

    # public signals
    # emitted when either min or max value is changed
    range_value_changed = QtCore.pyqtSignal(str, float, float,
                                            name='RangeValueChanged')
    # emitted when cmap id is changed
    cmap_changed = QtCore.pyqtSignal(str, ColorMap, name='CmapChanged')

    # private signals -> use for internal updates
    _min_value_changed = QtCore.pyqtSignal(float, name='_MinValueChanged')
    _max_value_changed = QtCore.pyqtSignal(float, name='_MaxValueChanged')


    def __init__(self, axis_name='', dom_min_val=0, dom_max_val=1, cmap_ini=None,
                 min_ini=None, max_ini=None, parent=None):
        """
        Create a MplCmapEditor that allows the user to set the cmap string
        and the value range.
        """
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MplCmapEditor()
        self.ui.setupUi(self)

        self.parent = parent


        self.dom_min_val = dom_min_val
        self.dom_max_val = dom_max_val
        self.axis_name = axis_name

        self.min_val = dom_min_val
        self.max_val = dom_max_val

        # UI initialization

        cmap_list = [str(s) for s in sorted(matplotlib.cm.cmap_d.keys())]
        self.ui.cmap_selector_box.addItems(cmap_list)
        cmap_ini = cmap_ini or 'jet'
        i = self.ui.cmap_selector_box.findText(cmap_ini)
        self.ui.cmap_selector_box.setCurrentIndex(i)

        self.ui.label_min_val.setText('| '+self.format_val(dom_min_val))
        self.ui.label_max_val.setText(self.format_val(dom_max_val)+' |')

        valid_nb_decimals = 10
        validator = QtGui.QDoubleValidator(dom_min_val, dom_max_val,
                                           valid_nb_decimals)
        self.ui.edit_min.setValidator(validator)
        self.ui.edit_max.setValidator(validator)

        min_ini = min_ini or dom_min_val
        max_ini = max_ini or dom_max_val

        # set labels showing min and max of domain:
        self.ui.edit_min.setText(self.format_val(min_ini))
        self.ui.edit_max.setText(self.format_val(max_ini))

        self.ui.slider_min.setSliderPosition(self.val_to_slider_pos(min_ini))
        self.ui.slider_max.setSliderPosition(self.val_to_slider_pos(max_ini))

        # Signal / slot connections

        # will hold last emitted range to avoid emitting 'range_value_changed'
        # twice:
        self.last_emitted_range = tuple()

        self.connect(self.ui.cmap_selector_box,
                     QtCore.SIGNAL('currentIndexChanged(const QString&)'),
                     self.emit_cmap)
        # won't work because there is two signals called currentIndexChanged
        # in QComboBox:
        #self.ui.cmap_selector_box.currentIndexChanged.connect(self.cmap_changed)

        self._min_value_changed.connect(self._set_slider_min_pos_from_val)
        self._min_value_changed.connect(self._set_edit_min_text_from_val)

        self._max_value_changed.connect(self._set_slider_max_pos_from_val)
        self._max_value_changed.connect(self._set_edit_max_text_from_val)

        self.ui.edit_min.returnPressed.connect(self.set_min_val_from_edit)
        self.ui.edit_max.returnPressed.connect(self.set_max_val_from_edit)

        self.ui.slider_min.valueChanged.connect(self._set_min_val_from_slider_pos)
        self.ui.slider_max.valueChanged.connect(self._set_max_val_from_slider_pos)


    def emit_cmap(self, cmap_id):
        self.cmap_changed.emit(self.axis_name,
                               matplotlib.cm.get_cmap(str(cmap_id)))

    def format_val(self, v):
        """ Format a domain value (float) to display it in line edits """
        return "%1.4f" %v

    def set_min_val(self, val):
        """
        Change the internal min value.
        Emit _min_value_changed to tell other internal widgets to update
             -> update slider_min position
             -> update edit_max & slider_max if min_value > max_value
        Emit range_value_changed if not already emitted by cascading signals
        """

        self.min_val = val
        self._min_value_changed.emit(val)
        if self.min_val > self.max_val:
            self.set_max_val(self.min_val)

        if (self.min_val, self.max_val) != self.last_emitted_range:
            self.range_value_changed.emit(self.axis_name, self.min_val,
                                          self.max_val)
            self.last_emitted_range = (self.min_val, self.max_val)

    def set_max_val(self, val):
        """
        Change the internal max value.
        Emit _max_value_changed to tell other internal widgets to update
             -> update slider_max position
             -> update edit_min & slider_min if min_value > max_value
        Emit range_value_changed if not already emitted by cascading signals
        """

        self.max_val = val
        self._max_value_changed.emit(val)
        if self.max_val < self.min_val:
            self.set_min_val(self.max_val)

        if (self.min_val, self.max_val) != self.last_emitted_range:
            self.range_value_changed.emit(self.axis_name, self.min_val,
                                          self.max_val)
            self.last_emitted_range = (self.min_val, self.max_val)


    def val_to_slider_pos(self, val):
        """ Convert a domain value (float) into a slider position (int)"""
        nticks = self.ui.slider_max.maximum() * 1.
        pos = nticks / (self.dom_max_val - self.dom_min_val) * \
          (val - self.dom_max_val) + nticks
        return int(round(pos))

    def slider_pos_to_val(self, pos):
        """ Convert a slider position (int) into a domain value (float)"""
        nticks = self.ui.slider_max.maximum() * 1.
        return (self.dom_max_val - self.dom_min_val) / nticks * \
          (pos - nticks) + self.dom_max_val

    def set_min_val_from_edit(self):
        """ Set the current min value by reading associated line edit"""
        txt = self.ui.edit_min.text()
        self.set_min_val(float(txt))

    def set_max_val_from_edit(self):
        """ Set the current max value by reading associated line edit"""
        txt = self.ui.edit_max.text()
        self.set_max_val(float(txt))


    @QtCore.pyqtSlot(str)
    def set_value_label(self, label):
        self.axis_name = label

    def set_boundaries(self, vmin, vmax):
        self.dom_min_val = vmin
        self.dom_max_val = vmax

        self.ui.label_min_val.setText('| '+self.format_val(vmin))
        self.ui.label_max_val.setText(self.format_val(vmax)+' |')

        valid_nb_decimals = 10
        validator = QtGui.QDoubleValidator(vmin, vmax, valid_nb_decimals)
        self.ui.edit_min.setValidator(validator)
        self.ui.edit_max.setValidator(validator)



    @QtCore.pyqtSlot(int)
    def _set_min_val_from_slider_pos(self, pos):
        """ Set the current min value from the given slider position (int) """
        if self.val_to_slider_pos(self.min_val) != pos:
            self.set_min_val(self.slider_pos_to_val(pos))


    @QtCore.pyqtSlot(int)
    def _set_max_val_from_slider_pos(self, pos):
        """ Set the current max value from the given slider position (int) """
        if self.val_to_slider_pos(self.max_val) != pos:
            self.set_max_val(self.slider_pos_to_val(pos))


    @QtCore.pyqtSlot(float)
    def _set_slider_min_pos_from_val(self, val):
        """
        Internal slot to update slider_min position
        from given val (float)
        """
        pos = self.val_to_slider_pos(val)
        self.ui.slider_min.setSliderPosition(pos)

    @QtCore.pyqtSlot(float)
    def _set_edit_min_text_from_val(self, val):
        """
        Internal slot to set the text of the line edit
        for the min at given val (float) """
        self.ui.edit_min.setText(self.format_val(val))

    @QtCore.pyqtSlot(float)
    def _set_slider_max_pos_from_val(self, val):
        """
        Internal slot to update slider_min position
        from given val (float)
        """
        pos = self.val_to_slider_pos(val)
        self.ui.slider_max.setSliderPosition(pos)

    @QtCore.pyqtSlot(float)
    def _set_edit_max_text_from_val(self, val):
        """
        Internal slot to set the text of the line edit
        for the max at given val (float) """
        self.ui.edit_max.setText(self.format_val(val))


def print_cmap(an, cm):
    print 'recieve cmap for axis %s:' %an, str(cm)

def print_range(an, vmin, vmax):
    print 'recieve range for axis %s:'%an, vmin, vmax

def main():
    app = QtGui.QApplication(sys.argv)

    editor = MplCmapEditor('nrl', 0, 299., min_ini=50., max_ini=150.)
    editor.cmap_changed.connect(print_cmap)
    editor.range_value_changed.connect(print_range)
    editor.show()
    app.exec_()


if __name__ == "__main__":
    main()
