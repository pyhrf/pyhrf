# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'domain_color_selector.ui'
#
# Created: Fri Aug 16 09:44:00 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_DomainColorSelector(object):
    def setupUi(self, DomainColorSelector):
        DomainColorSelector.setObjectName(_fromUtf8("DomainColorSelector"))
        DomainColorSelector.resize(400, 300)
        self.verticalLayout = QtGui.QVBoxLayout(DomainColorSelector)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.tableWidget = QtGui.QTableWidget(DomainColorSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setObjectName(_fromUtf8("tableWidget"))
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout.addWidget(self.tableWidget)

        self.retranslateUi(DomainColorSelector)
        QtCore.QMetaObject.connectSlotsByName(DomainColorSelector)

    def retranslateUi(self, DomainColorSelector):
        DomainColorSelector.setWindowTitle(QtGui.QApplication.translate("DomainColorSelector", "Form", None, QtGui.QApplication.UnicodeUTF8))

