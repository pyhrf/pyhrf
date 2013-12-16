"""
Main window of the viewer, consisting of :
- a list of loaded items. Each item in the list can be reloaded from file
- a browser widget to navigate in a given item's data

Actions:
- open file
- quit application
"""
import sys
import os.path as op
from PyQt4 import QtCore, QtGui
from viewer_main_window_ui import Ui_ViewerMainWindow
import ui_base as uib
from browser_stack import BrowserStack

from pyhrf.ndarray import xndarray
from pyhrf.tools import add_suffix

#pyuic4 viewer_main_window.ui -o viewer_main_window_ui.py

class ViewerMainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.ui = Ui_ViewerMainWindow()
        self.ui.setupUi(self)

        self.item_to_filenames = {}
        self.filenames_to_items = {}

        self.main_browser = BrowserStack(self)
        self.ui.verticalLayout.addWidget(self.main_browser)

        self.pack_actions()

        self.ui.item_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.item_list.customContextMenuRequested.connect(self.item_list_popup)

    def item_list_popup(self, pos):

        sel_indexes = self.ui.item_list.selectionModel().selection().indexes()
        if len(sel_indexes) > 0:
            menu = QtGui.QMenu(self.ui.item_list)
            reloadAction = menu.addAction("Reload")
            action = menu.exec_(self.ui.item_list.mapToGlobal(pos))
            if action == reloadAction:
                fn_to_cuboid = {}
                for i in sel_indexes:
                    r, c = i.row(), i.column()
                    item = str(self.ui.item_list.itemAt(r, c).text())
                    # only load cuboids once
                    fn = self.item_to_filenames[item]
                    if not fn_to_cuboid.has_key(fn):
                        fn_to_cuboid[fn] = xndarray.load(fn)
                    self.main_browser.set_new_cuboid(fn_to_cuboid[fn], item)
                    fn_to_cuboid = None #garbage collect

    def pack_actions(self):
        """
        Associate actions trigger to functions for:
        - Open -> display file chooser, load xndarray and open new view
        - Quit -> quit the application
        """
        self.ui.actionOpen.triggered.connect(self.ask_and_load_file)
        self.ui.actionQuit.triggered.connect(QtGui.qApp.quit)

    def ask_and_load_file(self):
        """
        Display file chooser, load xndarray and open new view
        """
        # ask for file names to open:
        
        fns = QtGui.QFileDialog.getOpenFileNames(self, "Open MRI data", './',
                                                 "MRI volume (*.img *.img.gz"\
                                                 "*.nii *.nii.gz)")
        # convert QStringList to list of str:
        fns = [str(fn) for fn in fns]

        for fn in fns:
            self.add_file(fn)


    def get_unique_item_id(self, name, suffix_nb=0):
        print 'name:', name
        if suffix_nb != 0:
            suffix = '(%d)' %suffix_nb
        else:
            suffix = ''
        candidate = add_suffix(name, suffix)
        print 'candidate:', candidate
        print [str(self.ui.item_list.item(i)) \
               for i in range(self.ui.item_list.count())]
        if candidate in [str(self.ui.item_list.item(i).text()) \
            for i in range(self.ui.item_list.count())]:
            return self.get_unique_item_id(name, suffix_nb+1)

        print 'final candidate:', candidate
        return candidate

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()

    def add_file(self, filename, open_plot=True):
        """
        Load a file, place into item list and create a browser/viewer.

        Args:
            - filename (str): path to the data file to load
            - open_plot (bool): open plot window on load (TODO)

        Modifies attributes:
            - item_to_filenames: associate filename to new item id
            - filenames_to_items: associate item id to filename
        """
        if not op.exists(filename):
            QtGui.QMessageBox.critical(self, 'File not found: %s' %filename)

        item_id = self.get_unique_item_id(op.basename(filename))
        #print 'item_id:', item_id
        self.item_to_filenames[item_id] = filename
        self.filenames_to_items[filename] = item_id

        self.ui.item_list.addItem(item_id)
        self.main_browser.add_cuboid(xndarray.load(filename), item_id)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    import pyhrf
    fn_bold = pyhrf.get_data_file_name('subj0_bold_session0.nii.gz')
    fn_anatomy = pyhrf.get_data_file_name('subj0_anatomy.nii.gz')
    fn_mask = pyhrf.get_data_file_name('subj0_parcellation.nii.gz')

    fn_test_reload_changing_data = './changing_data.nii'
    fn_test_reload_changing_data_and_axes = './changing_data_and_axes.nii'


    main_win = ViewerMainWindow()
    #main_win.add_file(fn_bold)
    main_win.add_file(fn_bold)
    main_win.add_file(fn_anatomy)
    main_win.add_file(fn_mask)
    #main_win.add_file(fn_test_reload_changing_data)
    #main_win.add_file(fn_test_reload_changing_data_and_axes)

    main_win.show()

    if 0: #save image of widget non interactively
        uib.render_widget_in_doc(main_win)
    else:
        import cProfile
        cProfile.run("sys.exit(app.exec_())", "cuboid_viewer.prf")




