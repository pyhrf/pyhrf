# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'browser_stack.ui'
#
# Created: Thu Nov 28 12:55:44 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_BrowserStack(object):
    def setupUi(self, BrowserStack):
        BrowserStack.setObjectName(_fromUtf8("BrowserStack"))
        BrowserStack.resize(355, 168)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(BrowserStack.sizePolicy().hasHeightForWidth())
        BrowserStack.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(BrowserStack)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.comboBox = QtGui.QComboBox(BrowserStack)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.verticalLayout.addWidget(self.comboBox)
        self.browser_stack = QtGui.QStackedWidget(BrowserStack)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browser_stack.sizePolicy().hasHeightForWidth())
        self.browser_stack.setSizePolicy(sizePolicy)
        self.browser_stack.setObjectName(_fromUtf8("browser_stack"))
        self.verticalLayout.addWidget(self.browser_stack)

        self.retranslateUi(BrowserStack)
        self.browser_stack.setCurrentIndex(-1)
        QtCore.QObject.connect(self.browser_stack, QtCore.SIGNAL(_fromUtf8("currentChanged(int)")), self.comboBox.setCurrentIndex)
        QtCore.QMetaObject.connectSlotsByName(BrowserStack)

    def retranslateUi(self, BrowserStack):
        BrowserStack.setWindowTitle(QtGui.QApplication.translate("BrowserStack", "Form", None, QtGui.QApplication.UnicodeUTF8))

