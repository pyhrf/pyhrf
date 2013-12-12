# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotter_toolbar.ui'
#
# Created: Sun Jan 27 18:43:10 2013
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
        PlotterToolbar.resize(361, 95)
        self.gridLayout = QtGui.QGridLayout(PlotterToolbar)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.color_button = QtGui.QPushButton(PlotterToolbar)
        self.color_button.setObjectName(_fromUtf8("color_button"))
        self.gridLayout.addWidget(self.color_button, 2, 1, 1, 1)
        self.image_radio_button = QtGui.QRadioButton(PlotterToolbar)
        self.image_radio_button.setChecked(True)
        self.image_radio_button.setObjectName(_fromUtf8("image_radio_button"))
        self.gridLayout.addWidget(self.image_radio_button, 2, 0, 1, 1)
        self.save_button = QtGui.QPushButton(PlotterToolbar)
        self.save_button.setObjectName(_fromUtf8("save_button"))
        self.gridLayout.addWidget(self.save_button, 3, 1, 1, 1)
        self.curve_radio_button = QtGui.QRadioButton(PlotterToolbar)
        self.curve_radio_button.setObjectName(_fromUtf8("curve_radio_button"))
        self.gridLayout.addWidget(self.curve_radio_button, 3, 0, 1, 1)

        self.retranslateUi(PlotterToolbar)
        QtCore.QMetaObject.connectSlotsByName(PlotterToolbar)

    def retranslateUi(self, PlotterToolbar):
        PlotterToolbar.setWindowTitle(QtGui.QApplication.translate("PlotterToolbar", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.color_button.setText(QtGui.QApplication.translate("PlotterToolbar", "color", None, QtGui.QApplication.UnicodeUTF8))
        self.image_radio_button.setText(QtGui.QApplication.translate("PlotterToolbar", "image", None, QtGui.QApplication.UnicodeUTF8))
        self.save_button.setText(QtGui.QApplication.translate("PlotterToolbar", "save", None, QtGui.QApplication.UnicodeUTF8))
        self.curve_radio_button.setText(QtGui.QApplication.translate("PlotterToolbar", "curve", None, QtGui.QApplication.UnicodeUTF8))

