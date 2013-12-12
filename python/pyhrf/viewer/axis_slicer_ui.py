# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'axis_slicer.ui'
#
# Created: Sat Sep  7 11:04:08 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_AxisSlicer(object):
    def setupUi(self, AxisSlicer):
        AxisSlicer.setObjectName(_fromUtf8("AxisSlicer"))
        AxisSlicer.resize(344, 82)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AxisSlicer.sizePolicy().hasHeightForWidth())
        AxisSlicer.setSizePolicy(sizePolicy)
        AxisSlicer.setMinimumSize(QtCore.QSize(320, 0))
        AxisSlicer.setMaximumSize(QtCore.QSize(16777215, 82))
        self.horizontalLayout = QtGui.QHBoxLayout(AxisSlicer)
        self.horizontalLayout.setContentsMargins(2, 0, 2, 0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.frame = QtGui.QFrame(AxisSlicer)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(320, 0))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.gridLayout = QtGui.QGridLayout(self.frame)
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.gridLayout.setContentsMargins(2, 0, 0, 0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_axis = QtGui.QLabel(self.frame)
        self.label_axis.setAlignment(QtCore.Qt.AlignCenter)
        self.label_axis.setObjectName(_fromUtf8("label_axis"))
        self.gridLayout.addWidget(self.label_axis, 0, 1, 1, 1)
        self.checkBox_emit = QtGui.QCheckBox(self.frame)
        self.checkBox_emit.setObjectName(_fromUtf8("checkBox_emit"))
        self.gridLayout.addWidget(self.checkBox_emit, 0, 2, 1, 1)
        self.checkBox_current_axis = QtGui.QCheckBox(self.frame)
        self.checkBox_current_axis.setText(_fromUtf8(""))
        self.checkBox_current_axis.setObjectName(_fromUtf8("checkBox_current_axis"))
        self.gridLayout.addWidget(self.checkBox_current_axis, 1, 0, 1, 1)
        self.slider = QtGui.QSlider(self.frame)
        self.slider.setEnabled(True)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName(_fromUtf8("slider"))
        self.gridLayout.addWidget(self.slider, 1, 1, 1, 1)
        self.label_domain_value = QtGui.QLabel(self.frame)
        self.label_domain_value.setAlignment(QtCore.Qt.AlignCenter)
        self.label_domain_value.setObjectName(_fromUtf8("label_domain_value"))
        self.gridLayout.addWidget(self.label_domain_value, 2, 1, 1, 1)
        self.checkBox_recieve = QtGui.QCheckBox(self.frame)
        self.checkBox_recieve.setChecked(True)
        self.checkBox_recieve.setObjectName(_fromUtf8("checkBox_recieve"))
        self.gridLayout.addWidget(self.checkBox_recieve, 2, 2, 1, 1)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(AxisSlicer)
        QtCore.QMetaObject.connectSlotsByName(AxisSlicer)

    def retranslateUi(self, AxisSlicer):
        AxisSlicer.setWindowTitle(QtGui.QApplication.translate("AxisSlicer", "View Slicer", None, QtGui.QApplication.UnicodeUTF8))
        self.label_axis.setText(QtGui.QApplication.translate("AxisSlicer", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_emit.setToolTip(QtGui.QApplication.translate("AxisSlicer", "<html><head/><body><p>Emit changes to other axis slicers</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_emit.setText(QtGui.QApplication.translate("AxisSlicer", "E", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_current_axis.setToolTip(QtGui.QApplication.translate("AxisSlicer", "<html><head/><body><p>Set as current axis of the graph </p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_domain_value.setText(QtGui.QApplication.translate("AxisSlicer", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_recieve.setToolTip(QtGui.QApplication.translate("AxisSlicer", "<html><head/><body><p>Recieve changes from other axis slices</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_recieve.setText(QtGui.QApplication.translate("AxisSlicer", "R", None, QtGui.QApplication.UnicodeUTF8))

