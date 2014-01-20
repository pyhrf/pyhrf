import sys
import re
import os.path as op
from inspect import currentframe, getargvalues, getouterframes
from inspect import ismethod, isfunction, isclass
import pickle
import numpy as np
from PyQt4 import QtGui, QtCore, QtXml

import pyhrf
from pyhrf.xmlio import getargspec
from pyhrf.tools import PickleableStaticMethod
from copy import copy

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


debug = False
debug2 = False

def numpy_array_from_string(s, sdtype, sshape=None):
    if sdtype[:3] == 'str':
        sdtype = 'str'
    if 'unicode' in sdtype:
        sdtype = 'unicode'

    data = str(s).split()
    if sdtype != 'str' and sdtype != 'unicode':
        data = [float(e) for e in data]

    a = np.array(data, dtype=np.typeDict[sdtype])
    if sshape is not None:
        if debug: print 'sshape:', sshape
        a.shape = tuple( int(e) for e in sshape.strip('(),').split(',') )
    return a


from pyhrf.xmlio import Initable
from pyhrf.xmlio import UiNode as UiNodeBase

class UiNode(UiNodeBase):

    """
    Store data hierarchically to be used in a Qt model/tree view setting.
    Also store additional node-specific data as attributes (in a dict).
    Attributes must only contain strings.

    The resulting data structure is:

           col 0        | col 1
        |- <node_label> | <node_attributes>                   #row 0
            |
            |  col 0              | col 1
            |- <child_node_label> | <child_node_attributes>   #row 0
                |
                |...
        ...

    This structure is similar to DOM.

    Features:
    - can be instanciated from any python object that can
      be serialized into human-readable strings: bool, int, float, string,
      numpy scalars, numpy arrays.
      It also support container types (list, dict, OrderedDict) as long as their
      items are also serializabl into human-readable strings.
      See static method *from_py_object*.

      #TODO: set and tuple
      #TODO: test unicode

    - Support XML I/O through QtXml. See method *to_xml* and *from_xml*

    """

    def to_xml_qt(self):
        """
        Return an XML representation (str) of the Node and its children.
        """
        doc = QtXml.QDomDocument()

        node = doc.createElement(self.label())

        doc.appendChild(node)

        for k, v in self.serialize_attributes().iteritems():
            node.setAttribute(k, v)

        for c in self.get_children():
            c._recurse_to_xml_qt(doc, node)

        return doc.toString(indent=4)


    def _recurse_to_xml_qt(self, doc, parent):
        pyhrf.verbose(6, '_recurse_to_xml ...')

        children = self.get_children()
        if len(children) > 0:
            node = doc.createElement(self.label())
        else: #leaf node -> use QTextNode
            pyhrf.verbose(6, 'text node: %s' %self.label())
            node = doc.createTextNode(self.label())

        parent.appendChild(node)

        for k, v in self.serialize_attributes().iteritems():
            node.setAttribute(k, v)

        for c in self.get_children():
            c._recurse_to_xml_qt(doc, node)

    @PickleableStaticMethod
    def from_xml_qt(self, sxml):

        doc = QtXml.QDomDocument()

        # parser = QtXml.QXmlSimpleReader()
        # qinput = QtXml.QXmlInputSource()
        # qinput.setData(sxml)
        #doc.setContent(qinput, parser)

        doc.setContent(sxml)
        if debug:
            print 'QDomDocument:'
            print doc.toString()

        return self._recurse_from_xml_qt(doc.documentElement())

    @PickleableStaticMethod
    def _recurse_from_xml_qt(self, qnode):
        a = qnode.attributes()
        attributes = dict((str(a.item(i).nodeName()),
                           str(a.item(i).nodeValue()))\
                          for i in range(a.count()))
        attributes = self.unserialize_attributes(attributes)
        node_name = str(qnode.nodeName())
        if node_name != '#text':
            n = UiNode(node_name, attributes=attributes)
        else:
            n = UiNode(qnode.nodeValue(), attributes=attributes)

        child_nodes = qnode.childNodes()
        pyhrf.verbose(6, 'create new node %s from QDom (-> %d children)' \
                      %(node_name, child_nodes.count()))
        pyhrf.verbose(6, 'node value: %s' %str(qnode.nodeValue()))
        for i in range(child_nodes.count()):
            n.add_child(self._recurse_from_xml_qt(child_nodes.item(i)))

        return n


    ########################################
    #Methods to interact with Qt ItemModel #
    ########################################

    def insertChild(self, position, child):

        if position < 0 or position > len(self._children):
            return False

        self._children.insert(position, child)
        child._parent = self
        return True

    def removeChild(self, position):

        if position < 0 or position > len(self._children):
            return False

        child = self._children.pop(position)
        child._parent = None

        return True

    def parent(self):
        return self._parent

    def row(self):
        if self._parent is not None:
            return self._parent._children.index(self)

    def data(self, column):

        if   column is 0: return self._label
        elif column is 1: return self.attributes

    def setData(self, column, value):
        if   column is 0: self._label = value.toPyObject() #toString?
        elif column is 1: pass

    def resource(self):
        return None


    def get_actions(self, parent):
        node_type = self.get_attribute('type')
        node_meta = self.get_attribute('meta')

        if 'FILE' in node_meta:
            curFn = self.label()
            fileChooser = QtGui.QFileDialog(parent)
            fileChooser.setModal(True)
            fileChooser.setDirectory(op.dirname(curFn))


class UiModel(QtCore.QAbstractItemModel):

    """ Qt ItemModel to handle access to typed hierarchical data as stored by
    a *UiNode* instance. Intended to feed a QTreeView.
    """

    def __init__(self, root, parent=None):
        super(UiModel, self).__init__(parent)
        self._rootNode = root

        self._icons = {}
        self._default_icon  = QtGui.QPixmap('./pics/xml_element.png')

        if debug:
            print "UiModel - default icon:", self._default_icon

    def rowCount(self, parent):
        if not parent.isValid():
            parentNode = self._rootNode
        else:
            parentNode = parent.internalPointer()

        return parentNode.childCount()


    def data(self, index, role):
         if not index.isValid():
             return None

         node = index.internalPointer()

         if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if index.column() == 0: # column should always be 0,
                                    # expected in some rare cases ...
                return node.label()
         if role == QtCore.Qt.DecorationRole:
             print 'DecorationRole ...'
             if index.column() == 0:
                 icon = self._icons.get(node.type_info(), self._default_icon)
                 print 'icon:', icon
                 if icon is not None:
                     return QtGui.QIcon(icon)


    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if section == 0:
                return "UiModel" #...
            else:
                return "type_info" #...


    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
          QtCore.Qt.ItemIsEditable

    def index(self, row, column, parent):
        if not parent.isValid(): #if root node
            parentNode = self._rootNode
        else:
            parentNode = parent.internalPointer()

        childItem = parentNode.child(row)
        if childItem is not None:
            return self.createIndex(row, column, childItem)
        else:
            # may happen rarely ...
            return QtCore.QModelIndex()


    def setData(self, index, value, role=QtCore.Qt.EditRole):

        if index.isValid():
            if role == QtCore.Qt.EditRole:
                node = index.internalPointer()
                #try:
                return node.set_label(value)
                # except Exception, e:
                #     QMessageBox.critical(self,
                #                          "Could not set data. Error: %s" %str(e))
                #return True
        return False


    def parent(self, index):
        """ index: QModelIndex instance """
        #index.row()
        #index.column()

        node = index.internalPointer() #get the actual node
        parentNode = node.parent()
        if parentNode == self._rootNode:
            return QtCore.QModelIndex() #emtpy because root does not have a parent

        return self.createIndex(parentNode.row(), 0, parentNode)

    def columnCount(self, parent):
        return 1

