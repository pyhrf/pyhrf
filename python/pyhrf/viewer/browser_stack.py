"""

"""

import sys
import numpy as np
from PyQt4 import QtCore, QtGui
from browser_stack_ui import Ui_BrowserStack
from cuboid_viewer import xndarrayViewer

#pyuic4 browser_stack.ui -o browser_stack_ui.py

class BrowserStack(QtGui.QWidget):


    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.ui = Ui_BrowserStack()
        self.ui.setupUi(self)

        self.ui.comboBox.currentIndexChanged.connect(self.change_stack_index)

        self.viewers = {}

    def change_stack_index(self, i):
        cw = self.ui.browser_stack.currentWidget()
        if cw is not None:
            cw.setSizePolicy(QtGui.QSizePolicy.Ignored,
                             QtGui.QSizePolicy.Ignored)

        self.ui.browser_stack.setCurrentIndex(i)
        cw = self.ui.browser_stack.currentWidget()
        if cw is not None:
            cw.setSizePolicy(QtGui.QSizePolicy.Expanding,
                             QtGui.QSizePolicy.Expanding)

            #bring associated plotter to front
            self.viewers[cw.cuboid_name].plotter.raise_()

        self.adjustSize()

    def add_cuboid(self, c, name):
        cv = xndarrayViewer(c, name, parent=self.ui.browser_stack)
        cv.plotter.closing.connect(self.remove_viewer)
        self.viewers[name] = cv

        self.ui.browser_stack.addWidget(cv.browser)
        self.ui.comboBox.addItem(name)
        cv.plotter.focus_in.connect(self.set_current_browser)
        cv.plotter.show()

    def set_new_cuboid(self, c, name):
        self.viewers[name].set_new_cuboid(c)

    def set_current_browser(self, name):
        """
        Set the current browser widget in the stack from given name

        Args:
            - name (str): name of the cuboid item
        """
        i = self.ui.browser_stack.indexOf(self.viewers[str(name)].browser)
        self.change_stack_index(i)

    def remove_viewer(self, name):

        cv = self.viewers.pop(str(name))
        # remove browser from stack
        i = self.ui.browser_stack.indexOf(cv.browser)
        self.ui.browser_stack.removeWidget(cv.browser)

        # remove browser entry in combox box
        self.ui.comboBox.removeItem(i)

def test():
    pass

if __name__ == "__main__":

    if 0:
        test()
    else:
        from pyhrf.ndarray import xndarray
        app = QtGui.QApplication(sys.argv)

        sh = (10,10,3)
        conds = np.array(['audio1','audio2', 'video'])
        c1 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['sagittal','coronal', 'condition'],
                    axes_domains={'condition':conds})

        sh = (100,3)
        conds = np.array(['audio1','audio2', 'video'])
        c2 = xndarray(np.arange(np.prod(sh)).reshape(sh),
                    axes_names=['iteration', 'condition'],
                    axes_domains={'condition':conds})


        bs = BrowserStack()
        bs.add_cuboid(c1, 'jde_mcmc_nrl_pm.nii')
        bs.add_cuboid(c2, 'jde_mcmc_hrf_pm.nii')
        bs.show()
        sys.exit(app.exec_())

