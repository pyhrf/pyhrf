# -*- coding: utf-8 -*-
"""
Handle view of a xndarray via 2 widgets:
    - a slicer (xndarrayBrowser):
        let the user select the current 1D or 2D slice to view
    - a plotter (xndarrayPlotter):
        plot the selected slice as an Image or as curve(s)


Behaviour:
    - when the plotter has focus, the slicer is brought forward
    - when the slicer has focus, the plotter is brought forward
    - when the potter is closed, the slicer is closed too
    - when the slicer is closed, the plotter is closed too
    - when slice is changed in slicer then plotter is updated accordingly
"""
import sys
import numpy as np

from PyQt4 import QtCore, QtGui

import pyhrf

import ui_base as uib
from cuboid_plotter import xndarrayPlotter
from cuboid_browser import xndarrayBrowser


class xndarrayViewer(QtCore.QObject):

    def __init__(self, cuboid, cuboid_name='data', parent=None):
        QtCore.QObject.__init__(self)
        
        c = cuboid.reorient(get_preferred_orientation(cuboid.axes_names))
        self.browser = xndarrayBrowser(c, cuboid_name)
        slice_def = self.browser.get_slice_def()
        orientation = self.browser.get_current_axes()
        self.plotter = xndarrayPlotter(c, slice_def, orientation, cuboid_name)

        # when slice def change, update plot
        self.browser.slice_changed.connect(self.plotter.set_slice)

    def set_new_cuboid(self, c):
        self.browser.set_new_cuboid(c)
        new_slice_def = self.browser.get_slice_def()
        new_orientation = self.browser.get_current_axes()
        self.plotter.set_new_cuboid(c, new_slice_def, new_orientation)


    def show(self):
        self.browser.show()
        self.plotter.show()

from pyhrf.ndarray import MRI3Daxes
PREFERRED_SPATIAL_ORIENTATION = ['coronal','sagittal','axial']

def get_preferred_orientation(names, rules=None):
    """
    Get the preferred orientation of the cuboid:
    if *rules* is provided and *names* in keys of rules,
    then return the associated orientation.
    Else, apply the following:
        - if 'iteration' or 'time' in names then return it as only axis
        - if 3D MRI axes in *names* then order them as
          PREFERRED_SPATIAL_ORIENTATION

    Args:
        - names (listof str): list of axis names
        - rules (dictof <tupleof str>:<tupleof str>):
            translation between a sorted list of axes names into
            its preferred orientation

    Return: listof str
        -> the reorded list of axes names


    TODO: make PREFERRED_SPATIAL_ORIENTATION be user-settable
    """
    names = names[:]
    rules = rules or {}

    if rules.has_key(tuple(sorted(names))):
        assert set(names) == set(rules[names])
        return rules[names]

    if 'time' in names:
        names.remove('time')
        return ['time'] + get_preferred_orientation(names, rules)

    if 'iteration' in names:
        names.remove('iteration')
        return ['iteration'] + get_preferred_orientation(names, rules)

    if set(names).issuperset(MRI3Daxes):
        [names.remove(a) for a in MRI3Daxes]
        return PREFERRED_SPATIAL_ORIENTATION + \
          get_preferred_orientation(names, rules)

    return names


def test_preferred_orientation():

    ori = get_preferred_orientation(['condition','coronal','sagittal','axial'])
    assert ori == PREFERRED_SPATIAL_ORIENTATION + ['condition']

    ori = get_preferred_orientation(['coronal','time','sagittal'])
    assert ori == ['time', 'coronal','sagittal']

    ori = get_preferred_orientation(['coronal','iteration','sagittal'])
    assert ori == ['iteration', 'coronal','sagittal']

    ori = get_preferred_orientation(['iteration','time','condition'])
    assert ori == ['time','iteration','condition']

def test():
    test_preferred_orientation()

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

        cv1 = xndarrayViewer(c1)
        cv1.show()

        sys.exit(app.exec_())
