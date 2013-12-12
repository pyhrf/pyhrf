# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotter_toolbar.ui'
#
# Created: Fri Aug 16 15:01:34 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_PlotterToolbar(object):
    def setupUi(self, PlotterToolbar):
        PlotterToolbar.setObjectName(_fromUtf8("PlotterToolbar"))
        PlotterToolbar.resize(361, 90)
        self.gridLayout = QtGui.QGridLayout(PlotterToolbar)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.colorbar_checkbox = QtGui.QCheckBox(PlotterToolbar)
        self.colorbar_checkbox.setObjectName(_fromUtf8("colorbar_checkbox"))
        self.gridLayout.addWidget(self.colorbar_checkbox, 7, 2, 1, 1)
        self.mask_checkbox = QtGui.QCheckBox(PlotterToolbar)
        self.mask_checkbox.setObjectName(_fromUtf8("mask_checkbox"))
        self.gridLayout.addWidget(self.mask_checkbox, 8, 2, 1, 1)
        self.axis_labels_checkbox = QtGui.QCheckBox(PlotterToolbar)
        self.axis_labels_checkbox.setChecked(True)
        self.axis_labels_checkbox.setObjectName(_fromUtf8("axis_labels_checkbox"))
        self.gridLayout.addWidget(self.axis_labels_checkbox, 8, 1, 1, 1)
        self.axes_checkbox = QtGui.QCheckBox(PlotterToolbar)
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.setObjectName(_fromUtf8("axes_checkbox"))
        self.gridLayout.addWidget(self.axes_checkbox, 7, 1, 1, 1)
        self.groupBox = QtGui.QGroupBox(PlotterToolbar)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.curve_radio_button = QtGui.QRadioButton(self.groupBox)
        self.curve_radio_button.setObjectName(_fromUtf8("curve_radio_button"))
        self.verticalLayout.addWidget(self.curve_radio_button)
        self.image_radio_button = QtGui.QRadioButton(self.groupBox)
        self.image_radio_button.setChecked(True)
        self.image_radio_button.setObjectName(_fromUtf8("image_radio_button"))
        self.verticalLayout.addWidget(self.image_radio_button)
        self.gridLayout.addWidget(self.groupBox, 7, 0, 3, 1)
        self.color_button = QtGui.QPushButton(PlotterToolbar)
        self.color_button.setObjectName(_fromUtf8("color_button"))
        self.gridLayout.addWidget(self.color_button, 9, 1, 1, 1)

        self.retranslateUi(PlotterToolbar)
        QtCore.QMetaObject.connectSlotsByName(PlotterToolbar)

    def retranslateUi(self, PlotterToolbar):
        PlotterToolbar.setWindowTitle(QtGui.QApplication.translate("PlotterToolbar", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.colorbar_checkbox.setText(QtGui.QApplication.translate("PlotterToolbar", "show colorbar", None, QtGui.QApplication.UnicodeUTF8))
        self.mask_checkbox.setText(QtGui.QApplication.translate("PlotterToolbar", "show mask", None, QtGui.QApplication.UnicodeUTF8))
        self.axis_labels_checkbox.setText(QtGui.QApplication.translate("PlotterToolbar", "show axis labels", None, QtGui.QApplication.UnicodeUTF8))
        self.axes_checkbox.setText(QtGui.QApplication.translate("PlotterToolbar", "show axes", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("PlotterToolbar", "View mode", None, QtGui.QApplication.UnicodeUTF8))
        self.curve_radio_button.setText(QtGui.QApplication.translate("PlotterToolbar", "curve", None, QtGui.QApplication.UnicodeUTF8))
        self.image_radio_button.setText(QtGui.QApplication.translate("PlotterToolbar", "image", None, QtGui.QApplication.UnicodeUTF8))
        self.color_button.setText(QtGui.QApplication.translate("PlotterToolbar", "color", None, QtGui.QApplication.UnicodeUTF8))

