# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cuboid_browser.ui'
#
# Created: Thu Nov 28 12:55:26 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_xndarrayBrowser(object):
    def setupUi(self, xndarrayBrowser):
        xndarrayBrowser.setObjectName(_fromUtf8("xndarrayBrowser"))
        xndarrayBrowser.resize(398, 70)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(xndarrayBrowser.sizePolicy().hasHeightForWidth())
        xndarrayBrowser.setSizePolicy(sizePolicy)
        xndarrayBrowser.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        xndarrayBrowser.setFont(font)
        self.verticalLayout_3 = QtGui.QVBoxLayout(xndarrayBrowser)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label_3 = QtGui.QLabel(xndarrayBrowser)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_axis_1 = QtGui.QLabel(xndarrayBrowser)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_axis_1.sizePolicy().hasHeightForWidth())
        self.label_axis_1.setSizePolicy(sizePolicy)
        self.label_axis_1.setText(_fromUtf8(""))
        self.label_axis_1.setObjectName(_fromUtf8("label_axis_1"))
        self.horizontalLayout_2.addWidget(self.label_axis_1)
        self.label_axis_2 = QtGui.QLabel(xndarrayBrowser)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_axis_2.sizePolicy().hasHeightForWidth())
        self.label_axis_2.setSizePolicy(sizePolicy)
        self.label_axis_2.setText(_fromUtf8(""))
        self.label_axis_2.setObjectName(_fromUtf8("label_axis_2"))
        self.horizontalLayout_2.addWidget(self.label_axis_2)
        self.swap_button = QtGui.QPushButton(xndarrayBrowser)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.swap_button.sizePolicy().hasHeightForWidth())
        self.swap_button.setSizePolicy(sizePolicy)
        self.swap_button.setObjectName(_fromUtf8("swap_button"))
        self.horizontalLayout_2.addWidget(self.swap_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_3.addLayout(self.verticalLayout)

        self.retranslateUi(xndarrayBrowser)
        QtCore.QMetaObject.connectSlotsByName(xndarrayBrowser)

    def retranslateUi(self, xndarrayBrowser):
        xndarrayBrowser.setWindowTitle(QtGui.QApplication.translate("xndarrayBrowser", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("xndarrayBrowser", "Current axes:", None, QtGui.QApplication.UnicodeUTF8))
        self.swap_button.setText(QtGui.QApplication.translate("xndarrayBrowser", "swap", None, QtGui.QApplication.UnicodeUTF8))

