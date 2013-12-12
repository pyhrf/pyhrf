# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mpl_cmap_editor.ui'
#
# Created: Sat Sep  7 14:32:23 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MplCmapEditor(object):
    def setupUi(self, MplCmapEditor):
        MplCmapEditor.setObjectName(_fromUtf8("MplCmapEditor"))
        MplCmapEditor.resize(447, 108)
        self.verticalLayout = QtGui.QVBoxLayout(MplCmapEditor)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.cmap_selector_box = QtGui.QComboBox(MplCmapEditor)
        self.cmap_selector_box.setObjectName(_fromUtf8("cmap_selector_box"))
        self.verticalLayout.addWidget(self.cmap_selector_box)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_left = QtGui.QVBoxLayout()
        self.verticalLayout_left.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout_left.setObjectName(_fromUtf8("verticalLayout_left"))
        self.label_min = QtGui.QLabel(MplCmapEditor)
        self.label_min.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_min.sizePolicy().hasHeightForWidth())
        self.label_min.setSizePolicy(sizePolicy)
        self.label_min.setAlignment(QtCore.Qt.AlignCenter)
        self.label_min.setObjectName(_fromUtf8("label_min"))
        self.verticalLayout_left.addWidget(self.label_min)
        self.edit_min = QtGui.QLineEdit(MplCmapEditor)
        self.edit_min.setObjectName(_fromUtf8("edit_min"))
        self.verticalLayout_left.addWidget(self.edit_min)
        self.horizontalLayout.addLayout(self.verticalLayout_left)
        self.verticalLayout_middle = QtGui.QVBoxLayout()
        self.verticalLayout_middle.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout_middle.setObjectName(_fromUtf8("verticalLayout_middle"))
        self.slider_min = QtGui.QSlider(MplCmapEditor)
        self.slider_min.setOrientation(QtCore.Qt.Horizontal)
        self.slider_min.setObjectName(_fromUtf8("slider_min"))
        self.verticalLayout_middle.addWidget(self.slider_min)
        self.slider_max = QtGui.QSlider(MplCmapEditor)
        self.slider_max.setSliderPosition(99)
        self.slider_max.setOrientation(QtCore.Qt.Horizontal)
        self.slider_max.setObjectName(_fromUtf8("slider_max"))
        self.verticalLayout_middle.addWidget(self.slider_max)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_min_val = QtGui.QLabel(MplCmapEditor)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_min_val.sizePolicy().hasHeightForWidth())
        self.label_min_val.setSizePolicy(sizePolicy)
        self.label_min_val.setObjectName(_fromUtf8("label_min_val"))
        self.horizontalLayout_2.addWidget(self.label_min_val)
        self.label_max_val = QtGui.QLabel(MplCmapEditor)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_max_val.sizePolicy().hasHeightForWidth())
        self.label_max_val.setSizePolicy(sizePolicy)
        self.label_max_val.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_max_val.setObjectName(_fromUtf8("label_max_val"))
        self.horizontalLayout_2.addWidget(self.label_max_val)
        self.verticalLayout_middle.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_middle)
        self.verticalLayout_right = QtGui.QVBoxLayout()
        self.verticalLayout_right.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout_right.setObjectName(_fromUtf8("verticalLayout_right"))
        self.label_max = QtGui.QLabel(MplCmapEditor)
        self.label_max.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_max.sizePolicy().hasHeightForWidth())
        self.label_max.setSizePolicy(sizePolicy)
        self.label_max.setAlignment(QtCore.Qt.AlignCenter)
        self.label_max.setObjectName(_fromUtf8("label_max"))
        self.verticalLayout_right.addWidget(self.label_max)
        self.edit_max = QtGui.QLineEdit(MplCmapEditor)
        self.edit_max.setObjectName(_fromUtf8("edit_max"))
        self.verticalLayout_right.addWidget(self.edit_max)
        self.horizontalLayout.addLayout(self.verticalLayout_right)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(MplCmapEditor)
        QtCore.QMetaObject.connectSlotsByName(MplCmapEditor)

    def retranslateUi(self, MplCmapEditor):
        MplCmapEditor.setWindowTitle(QtGui.QApplication.translate("MplCmapEditor", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label_min.setText(QtGui.QApplication.translate("MplCmapEditor", "Min", None, QtGui.QApplication.UnicodeUTF8))
        self.label_min_val.setText(QtGui.QApplication.translate("MplCmapEditor", "| min_val", None, QtGui.QApplication.UnicodeUTF8))
        self.label_max_val.setText(QtGui.QApplication.translate("MplCmapEditor", "max_val |", None, QtGui.QApplication.UnicodeUTF8))
        self.label_max.setText(QtGui.QApplication.translate("MplCmapEditor", "Max", None, QtGui.QApplication.UnicodeUTF8))

