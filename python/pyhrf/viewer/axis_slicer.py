import sys
import numpy as np
from PyQt4 import QtCore, QtGui

from axis_slicer_ui import Ui_AxisSlicer

#pyuic4 axis_slicer.ui -o axis_slicer_ui.py

from ui_base import DomainValue, SliceEventDispatcher


class AxisSlicer(QtGui.QWidget):
    """
    Signals:
        - slice_value_changed(str,str): if emission is enabled then this signal
            is emitted when slider is moved
            -> send (axis name, new domain value)
        - axis_state_changed(str, bool): emitted when axis projection is toggled
            -> send (axis name, true if projection is enabled)

    Private signal:
        - _slice_value_changed(str,str): same as slice_value_changed but ignore
            emission state. Intended to be used by xndarrayBrowser

    Slots:
        - set_slice_value(aname str, dval): if reception is enabled and
            aname is current axis and dval in current domain values
            then set current domain value to the recieved domain value.

    """

    slice_value_changed = QtCore.pyqtSignal(str, DomainValue,
                                            name='SliceValueChanged')

    _slice_value_changed = QtCore.pyqtSignal(str, DomainValue,
                                             name='_SliceValueChanged')

    axis_state_changed = QtCore.pyqtSignal(str, bool, name='AxisStateChanged')


    def __init__(self, axis_name, axis_domain, parent=None):
        """
        Create an AxisSlicer widget with initial value set to axis_domain[0].
        The created slicer is also registered in SliceEventDispatcher
        to allow synchronization with slicers sharing the same axis name

        Args:
            - axis_name (str): label of the sliced axis
            - axis_domain (np.array): 1D array of available domain values

        """
        QtGui.QWidget.__init__(self, parent)

        self.ui = Ui_AxisSlicer()
        self.ui.setupUi(self)

        self.ui.label_axis.setText(axis_name)
        self.axis_name = axis_name
        self.axis_domain = axis_domain
        self.dvals_to_int = dict((dv,i) for (i,dv) in enumerate(axis_domain))

        self.ui.slider.setMaximum(len(self.axis_domain)-1)
        self.current_value = self.axis_domain[0]
        self.ui.label_domain_value.setText(QtCore.QString(self.current_value))

        self.ui.slider.valueChanged.connect(self.set_value_from_slider_pos)
        self.ui.checkBox_current_axis.stateChanged.connect(self._set_axis_state)

        SliceEventDispatcher().register_slicer(axis_name, self)

    @QtCore.pyqtSlot(bool)
    def set_axis_state(self, state):
        """
        Set the current axis state. If state is True then current axis is
        projected, else the current axis is sliced.

        Args:
            - state (bool): the new state of the axis

        """
        if state:
            self.ui.label_domain_value.setText('<<projected>>')
            self.ui.slider.setEnabled(False)
        else:
            self.ui.label_domain_value.setText(str(self.current_value))
            self.ui.slider.setEnabled(True)
            i = self.dvals_to_int[self.current_value]
            self.ui.slider.setSliderPosition(i)

    def set_axis_selection_checkbox_state(self, state):
        self.ui.checkBox_current_axis.setChecked(state)

    @QtCore.pyqtSlot(QtCore.Qt.CheckState)
    def _set_axis_state(self, state):
        self.set_axis_state(state==QtCore.Qt.Checked)
        self.axis_state_changed.emit(self.axis_name, state)


    @QtCore.pyqtSlot(int)
    def set_value_from_slider_pos(self, index):

        if self.current_value != self.axis_domain[index]:
            self.current_value = self.axis_domain[index]
            self.ui.label_domain_value.setText(str(self.current_value))

            self._slice_value_changed.emit(self.axis_name, self.current_value)

            if self.ui.checkBox_emit.isChecked():
                self.slice_value_changed.emit(self.axis_name, self.current_value)


    @QtCore.pyqtSlot(str, DomainValue)
    def set_slice_value(self, aname, value):

        if self.ui.checkBox_recieve.isChecked() and \
            self.axis_name == aname and self.dvals_to_int.has_key(value):
            try:
                i = self.dvals_to_int[value]
            except ValueError:
                print 'could not find value %s in axis domain' %value
            self.current_value = value
            if not self.ui.checkBox_current_axis.isChecked():
                self.ui.slider.setSliderPosition(i)
                self.ui.label_domain_value.setText(str(value))
                self._slice_value_changed.emit(self.axis_name,
                                               self.current_value)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    as1 = AxisSlicer('condition', np.array(['audio','video']))
    as2 = AxisSlicer('condition', np.array(['sentence', 'audio','computation',
                                            'video', ]))

    as1.show()
    as2.show()
    sys.exit(app.exec_())

