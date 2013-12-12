import sys
from PyQt4 import QtCore, QtGui

from domain_color_selector_ui import Ui_DomainColorSelector
import ui_base as uib

#pyuic4 domain_color_selector.ui -o domain_color_selector_ui.py


class DomainColorSelector(QtGui.QWidget):

    domain_color_changed = QtCore.pyqtSignal(str, uib.DomainValue, uib.RgbColor,
                                             name='DomainColorChanged')


    def __init__(self, parent=None):
        """
        Create a DomainColorSelector that allows the user to set the colors
        associated to domain values

        Args:
            - dom_colors (dict <dvalue>:<color>): mapping between a given
                domain value and its associated color

        """
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_DomainColorSelector()
        self.ui.setupUi(self)

        self.parent = parent
        
        self.ui.tableWidget.itemDoubleClicked.connect(self.edit_item_color)

    def edit_item_color(self,item):
        sdval = self.ui.tableWidget.verticalHeaderItem(item.row()).text()
        prev_col = item.backgroundColor()
        title = 'Select color for "%s"' %sdval
        flags = QtGui.QColorDialog.ShowAlphaChannel
        new_col = QtGui.QColorDialog.getColor(prev_col, self, title, flags)
        if new_col.isValid() and new_col != prev_col:
            item.setBackgroundColor(new_col)
            dval = self.sdval_to_val[sdval]
            self.domain_color_changed.emit(self.axis_name, dval,
                                           uib.qcolor_to_mpl_rgb(new_col))

    @QtCore.pyqtSlot(str, dict)
    def set_domain_colors(self, axis_name, dom_colors):
        self.axis_name = axis_name

        self.ui.tableWidget.clear()
        self.ui.tableWidget.setColumnCount(1)
        self.ui.tableWidget.setRowCount(len(dom_colors))
        item = QtGui.QTableWidgetItem()
        item.setText('Color')
        self.ui.tableWidget.setHorizontalHeaderItem(0, item)
        self.sdval_to_val = {} #mapping between displayed text and actual dom val
        for i, dval in enumerate(sorted(dom_colors.keys())):
            col = dom_colors[dval]
            item = QtGui.QTableWidgetItem()
            item.setText(str(dval))
            self.ui.tableWidget.setVerticalHeaderItem(i, item)
            self.sdval_to_val[item.text()] = dval

            item = QtGui.QTableWidgetItem()
            item.setBackgroundColor(QtGui.QColor(col))
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.ui.tableWidget.setItem(i, 0, item)

def print_dcol(aname, dval, col):
    print 'new col axis "%s" and dval "%s": %s' %(aname, str(dval), str(col))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)


    dv_colors = {'audio' : 'red',
                 'video' : 'blue',
                 'sentence' : 'green'}

    myapp = DomainColorSelector('condition', dv_colors)
    myapp.domain_color_changed.connect(print_dcol)
    myapp.show()
    sys.exit(app.exec_())
