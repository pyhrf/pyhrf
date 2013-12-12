# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'viewer_main_window.ui'
#
# Created: Sat Sep  7 18:09:51 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ViewerMainWindow(object):
    def setupUi(self, ViewerMainWindow):
        ViewerMainWindow.setObjectName(_fromUtf8("ViewerMainWindow"))
        ViewerMainWindow.resize(318, 155)
        self.centralwidget = QtGui.QWidget(ViewerMainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.item_list = QtGui.QListWidget(self.centralwidget)
        self.item_list.setObjectName(_fromUtf8("item_list"))
        self.verticalLayout.addWidget(self.item_list)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        ViewerMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(ViewerMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 318, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        ViewerMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(ViewerMainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        ViewerMainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(ViewerMainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionQuit = QtGui.QAction(ViewerMainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(ViewerMainWindow)
        QtCore.QMetaObject.connectSlotsByName(ViewerMainWindow)

    def retranslateUi(self, ViewerMainWindow):
        ViewerMainWindow.setWindowTitle(QtGui.QApplication.translate("ViewerMainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("ViewerMainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setText(QtGui.QApplication.translate("ViewerMainWindow", "Open", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(QtGui.QApplication.translate("ViewerMainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))

