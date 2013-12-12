import sys, string, os
import os.path as op
from PyQt4 import QtCore, QtGui, QtXml
from PyQt4.QtGui import QFileDialog
from PyQt4.QtCore import QVariant
from domviewer_ui import Ui_MainWindow

from pkg_resources import Requirement, resource_filename, resource_listdir

debug = False
debug2 = True

def get_pic_fn(fn):
    """ Convenience to retrieve package data file name
    """
    req = Requirement.parse('pyhrf')
    pyhrfDataSubPath = 'pyhrf/xmledit/pics'
    fn2 = os.path.join(pyhrfDataSubPath, fn)
    return resource_filename(req, fn2)

####### Model : wrapper over QDomDocument ##########
class DomModel(QtCore.QAbstractItemModel):
    """
    Class wrapping a QDomDocument and exposing an editable QAbstractItemModel
    which supports item duplication (duplication of the underlying DOM subtree)
    Here is how the model provides access to data:

     - Example of Data (DOM) structure:
       <root>
        |-<tag_1 attr1='a1t1'>
        |  |-<tag_2 attr1='a1t2>
        |     |- val
        |-<tag3>

     - Corresponding structure exposed by the model:
       (root index)
        |
        |  col 0  | col 1
        |- 'root' | None                         #row=0
        |   |
        |   |  col 0   | col 1
        |   |- 'tag_1' | attribute map object      #row=0
        |       |
        |       |
        |       |  col 0  | col 1
        |       |- 'tag_2'| attribute map object    #row=0
        |           |
        |           |
        |           |  col 0  | col 1
        |           |- 'val   | None                 #row=0
        |
        |  col 0  | col 1
        |- tag3   | None                         #row=1


     The following conventions are applied:
      - There are two types of node :
         * structure nodes which have a tag name and can have attributes
         * leaf nodes which have a value and no attributes
      - The columns are fixed :
         * col 0 -> node label ( = tag name if structure node
                                 or = value if leaf node)
         * col 1 -> node attributes
    """


    def __init__(self, doc, parent, showComments=True):
        QtCore.QAbstractItemModel.__init__(self, parent)
        self.rootItem = DomItem(doc, 0)
        if debug:
            print 'DomModel.__init__ ...'
            print 'doc:'
            print doc.toString()
            print 'nb of children for root:',
            print self.rootItem.node().childNodes().length()
            print 'loading icons ...'
            print 'text icon -> ', get_pic_fn('xml_text.png')
        self.textIcon = QtGui.QIcon(get_pic_fn('xml_text.png'))
        self.commentIcon = QtGui.QIcon(get_pic_fn('xml_comment.png'))
        self.nodeIcon = QtGui.QIcon(get_pic_fn('xml_element.png'))
        self.showCommentsFlag = showComments

    def flags(self, index):
        """ Return flags for the model behaviour. All indexes are enabled,
        editable, and selectable. Indexes that concern nodes of type bool
        are also checkable.
        """
        #if debug: print 'Flags ...'
        if not index.isValid():
            return 0
        flag = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable \
            | QtCore.Qt.ItemIsEditable
        nodeItem = index.internalPointer()
        if nodeItem.is_bool_node():
            if debug: print 'checkable!'
            flag |= QtCore.Qt.ItemIsUserCheckable
        return flag

    #### READ ONLY METHODS ####
    def columnCount(self, parent):
        return 2

    def showComments(self,show=True):
        #TODO: rather handle by delegate ?
        self.showCommentsFlag = show

    def headerData(self, section, orientation, role):
        """ Return the following header, only in horizontal orientation:
        * section(=col) 0 -> "Label"
        * section(=col) 1 -> "Attributes"
        """
        if debug:
            print 'headerData - section:', section, \
                'orientation:', orientation, \
                'role:', role
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if section == 0:
                return QVariant("Label")
            elif section == 1:
                return QVariant("Attributes")
            else:
                return QVariant()

        return QVariant()

    def index(self, row, col, parent):
        """ Build a valid index for row, col and parent. It's a simple wrap
        over QDomDocument data.
        """
        if debug:
            print 'Index - row:', row
            print 'col:', col
            print 'parent:', parent
        if not self.hasIndex(row, col, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)

        if childItem is not None:
            return self.createIndex(row, col, childItem)
        else:
            return QtCore.QModelIndex()

    def rowCount(self, parent):
        if debug:
            print 'rowCount - parent:', parent
            print 'parent.column() :', parent.column()
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        cn = parentItem.node().childNodes()
        return cn.count()

    def parent(self, child):
        #print 'parent - child:', child
        if not child.isValid():
            return QtCore.QModelIndex()

        childItem = child.internalPointer()
        parentItem = childItem.parent()

        if parentItem is None or parentItem == self.rootItem:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def data(self, index, role):

        if debug:
            print 'model.data ...'
            print 'index :', index
            print 'role :', role
        if not index.isValid():
            return QVariant()


        if role != QtCore.Qt.DisplayRole and role != QtCore.Qt.EditRole \
                and role != QtCore.Qt.DecorationRole and role != QtCore.Qt.CheckStateRole:
            return QVariant()

        nodeItem = index.internalPointer()

        if role == QtCore.Qt.DecorationRole:
            if nodeItem.is_bool_node():
                return QVariant()
            if nodeItem.is_comment_node():
                return QVariant(self.commentIcon)
            elif nodeItem.is_text_node():
                return QVariant(self.textIcon)
            else:
                return QVariant(self.nodeIcon)

        if role == QtCore.Qt.CheckStateRole:
            if not nodeItem.is_bool_node():
                return QVariant()
            else:
                #print 'checkstaterole ...'
                #print '->', nodeItem.data(index.column()).toString()
                return nodeItem.data(index.column())

        if not self.showCommentsFlag and nodeItem.is_comment_node():
            return QVariant()


        if role == QtCore.Qt.DisplayRole and nodeItem.is_bool_node():
            return QVariant()

        return nodeItem.data(index.column())

    def setData(self, index, value, role):
        if debug:
            print 'model.setData ...'
            print 'role:', role
            print 'value:', value.toString()
        if role != QtCore.Qt.EditRole and role != QtCore.Qt.CheckStateRole:
            return False

        nodeItem = index.internalPointer()
        if role == QtCore.Qt.CheckStateRole:
            if nodeItem.is_bool_node():
                if debug:
                    print 'model.setData ...'
                    print 'checkstaterole with bool val'
                    print 'value:', value.toString()
                checked = (value==QtCore.Qt.Checked)
                result = nodeItem.setData(index.column(), QVariant(checked))
            else:
                result = False
        else:
            result = nodeItem.setData(index.column(), value)

        if result:
            self.emit(QtCore.SIGNAL('dataChanged(QModelIndex,QModelIndex)'),
                      index, index)
        return result

    def insertRows(self, position, rows, parent):
        if debug:
            print 'model.insertRows ...'
            print 'position:', position
            print 'rows:', rows
            print 'parent:', parent
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        self.beginInsertRows(parent, position, position + rows - 1)
        success = parentItem.insertChildren(position, rows)
        self.endInsertRows()

        return success

    def copy_node(self, index):
        if not index.isValid():
            item = self.rootItem
        else:
            item = index.internalPointer()

        return item.domNode.cloneNode()

    def insert_node(self, position, parent, node):
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        self.beginInsertRows(parent, position, position)
        success = parentItem.insert_child_node(position, node)
        self.endInsertRows()

        return success


    def take_node(self, position, parent):
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        self.beginRemoveRows(parent, position, position)
        takenNode = parentItem.take_child_node(position)
        self.endRemoveRows()
        return takenNode


    def removeRows(self, position, rows, parent):

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        self.beginRemoveRows(parent, position, position + rows - 1)
        success = parentItem.removeChildren(position, rows)
        self.endRemoveRows()

        return success

    def duplicate(self, index):
        if not index.isValid():
            return False

        domItem = index.internalPointer()
        parent = index.parent()
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        pos = index.row()
        self.beginInsertRows(parent, pos, pos)
        importedNode = domItem.domNode.cloneNode(True)
        success = parentItem.insert_child_node(pos, importedNode)
        self.endInsertRows()

        return success



class DomItem:

    DEFAULT_TAG_NAME = 'anonym'

    def __init__(self, node, row, parent=None):
        if debug: print 'Create domItem for node :', node.nodeName()
        self.domNode = node
        self.rowNumber = row
        self.parentItem = parent
        self.childItems = []

    def __del__(self):
        del self.childItems[:]

    def parent(self):
        return self.parentItem

    def row(self):
        return self.rowNumber

    def node(self):
        return self.domNode

    def is_leaf_node(self):
        n = self.domNode.nodeName()
        return n == '#comment' or n == '#text'

    def is_comment_node(self):
        n = self.domNode.nodeName()
        return n == '#comment'

    def is_text_node(self):
        n = self.domNode.nodeName()
        return n == '#text'

    def is_bool_node(self):
        """ Return True if the value of the associated DOM node is of type 'bool'
        else return False. To do so, the parent's attribute "type" is checked.
        """
        return self.is_text_node() and self.get_parent_attribute('type') == 'bool'

    def child(self, i):
        if debug:
            print 'child - i:', i
        self.ensureChildItemsSize()
        if self.childItems[i] is not None:
            return self.childItems[i]

        if i>=0 and i<self.domNode.childNodes().count():
            childNode = self.domNode.childNodes().item(i)
            childItem = DomItem(childNode, i, self)
            self.childItems[i] = childItem
            return childItem
        return False

    def ensureChildItemsSize(self):
        nc = self.domNode.childNodes().count()
        if len(self.childItems) < nc:
            self.childItems.extend([None] * nc)

    def get_parent_attribute(self, attr):
        pItem = self.parent()
        if pItem is not None:
            pNode = pItem.domNode
            attributeMap = pNode.attributes()
            #print 'parent\'s attributeMap:', attributeMap
            #print 'contains %s ?' %attr, attributeMap.contains(attr)
            return pNode.attributes().namedItem(attr).nodeValue()


    def removeChildren(self, i, count):
        if debug2:
            print 'removeChildren ...'
            print 'concerned item:'
            print self.domNode.nodeName()
            print 'i:', i
            print 'count:', count

        if i < 0 or i + count > len(self.childItems):
            return False

        if debug2:
            print 'Children Before remove:'
            for j in xrange(self.domNode.childNodes().count()):
                print 'child', j
                print '-> row:', self.childItems[j].row()
                print '->', self.childItems[j].domNode.nodeName()


        success = False
        for j in xrange(i,count):
            child = self.domNode.childNodes().item(j)

            if not self.domNode.removeChild(child).isNull():
                self.childItems.pop(j)
                success = True

        for j in xrange(0, self.domNode.childNodes().count()):
            self.childItems[j].rowNumber = j

        return success


    def take_child_node(self,i):
        if i < 0 or i >= len(self.childItems):
            return False

        child = self.domNode.childNodes().item(i)
        c = self.domNode.removeChild(child)
        if not c.isNull():
            self.childItems.pop(i)

        return c



    def insertChildren(self, i, rows):
        if debug2:
            print 'insertChildren ...'
            print 'i:', i
            print 'rows:', rows
        if i<0 or i > len(self.childItems):
            return False

        print 'Children Before insertion:'
        for j in xrange(self.domNode.childNodes().count()):
            print 'child', j
            print '-> row:', self.childItems[j].row()
            print '->', self.childItems[j].domNode.nodeName()


        print 'insertion ...'
        print 'children count:', self.domNode.childNodes().length()
        success = False
        for j in xrange(i, i+rows):
            print 'j:', j
            refChild = self.domNode.childNodes().item(j)
            print 'refChild:', refChild
            print ' ->', refChild.nodeName()
            newChild = self.domNode.ownerDocument().createElement(self.DEFAULT_TAG_NAME)
            print 'newChild:', newChild
            print ' ->', newChild.nodeName()
            if not self.domNode.insertBefore(newChild, refChild).isNull():
                newChildItem = DomItem(newChild, j, self)
                self.childItems.insert(j, newChildItem)
                success = True

        print 'Children after insertion:'
        print 'count:', self.domNode.childNodes().length()
        print 'childItems:'
        print self.childItems
        for j in xrange(self.domNode.childNodes().length()):
            print 'child', j
            print '-> row:', self.childItems[j].row()
            print '->', self.childItems[j].domNode.nodeName()

        print 'adjusting row numbers for children ...'
        for j in xrange(i, self.domNode.childNodes().count()):
            print 'j:',self.childItems[j]
            self.childItems[j].rowNumber = j

        return success


    def insert_child_node(self, i, node):
        if debug2:
            print 'insert node ...'
            print 'i:', i
            print 'node to insert:', node
            print ' ->', node.nodeName()
        if i<0 or i > len(self.childItems):
            return False

        print 'Children Before insertion:'
        for j in xrange(self.domNode.childNodes().count()):
            print 'child', j
            print '-> row:', self.childItems[j].row()
            print '->', self.childItems[j].domNode.nodeName()


        print 'insertion ...'
        print 'children count:', self.domNode.childNodes().length()

        refChild = self.domNode.childNodes().item(i)
        print 'refChild:', refChild
        print ' ->', refChild.nodeName()
        if self.domNode.insertBefore(node, refChild).isNull():
            return False
        newChildItem = DomItem(node, i, self)
        self.childItems.insert(i, newChildItem)

        print 'Children after insertion:'
        print 'count:', self.domNode.childNodes().length()
        print 'childItems:'
        print self.childItems
        for j in xrange(self.domNode.childNodes().length()):
            print 'child', j
            print '-> row:', self.childItems[j].row()
            print '->', self.childItems[j].domNode.nodeName()

        print 'adjusting row numbers for children ...'
        for j in xrange(i, self.domNode.childNodes().count()):
            print 'j:',self.childItems[j]
            self.childItems[j].rowNumber = j

        return True



    def data(self, col):
        if debug:
            print 'DomItem.data ...'
        attributeMap = self.domNode.attributes()
        attributes = QtCore.QStringList()
        #print 'col:', col
        if col == 0: # Label
            if self.is_leaf_node():
                r = self.domNode.nodeValue()#.split("\n").join(" ")
                if self.is_text_node():
                    t = self.get_parent_attribute('type')
                    if t == 'int':
                        r = QVariant(int(r))
                    elif t == 'bool':
                        if int(r) == 0:
                            r = QtCore.Qt.Unchecked
                        else:
                            r = QtCore.Qt.Checked
                    #elif t == 'float' or t == 'double':
                    #    r = QVariant(float(r))
            else:
                r = self.domNode.nodeName()
        elif col == 1: # Attributes
            for i in xrange(attributeMap.count()):
                attribute = attributeMap.item(i)
                attributes.append(attribute.nodeName() + '="' \
                                      + attribute.nodeValue() + '"')
            r = attributes.join(" ")
        else:
            r = QVariant()
        #print 'returning : ', r
        return QVariant(r)

    def get_attributes(self):
        attributeMap = self.domNode.attributes()
        names = []
        values = []
        for i in xrange(attributeMap.count()):
            attribute = attributeMap.item(i)
            names.append(attribute.nodeName())
            values.append(attribute.nodeValue())
        return names, values



    def setData(self, col, value):
        if debug:
            print 'DomItem.setData ...'
            print 'value:', type(value)
            #print dir(value)
            print 'setData ...', col
            print 'val: toString:', value.toString()
            print 'val: toBool', value.toBool()
            print 'val: toInt', value.toInt()
        if col == 0: #name
            n = self.domNode.nodeName()
            if n == '#comment' or n == '#text':
                if self.get_parent_attribute('type') == 'bool':
                    if debug:
                        print 'setting a bool value ...'
                        print str(value.toInt()[0])
                    self.domNode.setNodeValue(str(int(value.toBool())))
                else:
                    print 'value:', value
                    if isinstance(value, QtCore.QString):
                        self.domNode.setNodeValue(value)
                    else:
                        self.domNode.setNodeValue(value.toString())

                return True
            elif self.domNode.isElement():
                self.domNode.toElement().setTagName(value.toString())
                return True
        return False


##### Delegate to customize edition&view of a QTreeView ########

class XMLEditorDelegate(QtGui.QStyledItemDelegate):

    def __init__(self, parent=None, *args):
        if debug:
            print 'XMLEditorDelegate.__init__ ...'
            print 'args:', args
        QtGui.QItemDelegate.__init__(self, parent, *args)

    def createEditor(self, parent, viewItem, index):
        #editor = QtGui.QLineEdit(parent)
        if debug:
            print 'createEditor ...'
            print 'viewItem:', viewItem
            print 'index.col:', index.column()
            print 'index.isValid():', index.isValid()
        nodeItem = index.internalPointer()
        if nodeItem.is_comment_node(): # comments are not editable
            #TODO: maybe this should be specified in the model flags!
            return None
        if nodeItem.is_bool_node():
            return None
        if nodeItem.is_text_node():
            type = nodeItem.get_parent_attribute('type')
            if debug: print 'type ->', type
            meta = nodeItem.get_parent_attribute('meta')
            if debug: print 'meta ->', meta
            if 'FILE' in meta: #TODO: Also handle extension filters ...
                fileChooser = QtGui.QFileDialog(parent)
                pGeo = parent.geometry()
                if debug: print 'pgeo:', parent.geometry()
                geo = QtCore.QRect(0,0,400,400)
                fileChooser.setModal(True)
                if 'OUT' in meta:
                    fileChooser.setAcceptMode(QtGui.QFileDialog.AcceptSave)
                    fileChooser.setFileMode(QtGui.QFileDialog.AnyFile)
                else:
                    fileChooser.setFileMode(QtGui.QFileDialog.ExistingFile)
                    fileChooser.setAcceptMode(QtGui.QFileDialog.AcceptOpen)

                curFn = str(nodeItem.data(0).toString())
                print 'curFn:', curFn
                fileChooser.setDirectory(op.dirname(curFn))
                fileChooser.setMinimumSize(400,400)
                fileChooser.setMaximumSize(800,800)
                #print 'geo', fileChooser.geometry()
                self.connect(fileChooser, QtCore.SIGNAL('accepted'),
                             self.commit_and_close_editor)
                #self.connect(fileChooser, QtCore.SIGNAL('finished'),
                #             self.commit_and_close_editor)
                return fileChooser

        if debug:
            print 'comment node ?', nodeItem.is_comment_node()
            print 'text node ?', nodeItem.is_text_node()

        return QtGui.QStyledItemDelegate.createEditor(self, parent, viewItem, index)


    #def paint(self, painter, option, index):
    #    QtGui.QStyledItemDelegate.paint(self, painter, option, index)

    #def updateEditorGeometry(editor, option, index):
    #     pass

    def commit_and_close_editor(self, *args ):
        print 'commit_and_close_editor ...'
        print 'args:', args
        fchooser = self.sender()
        self.emit(QtCore.SIGNAL('commitData(editor)'), fchooser)
        self.emit(QtCore.SIGNAL('closeEditor(editor)'), fchooser)

    #def setEditorData(self, editor, index):
    #    pass

    def setModelData(self, editor, model, index):
        print 'set model data ...'
        if index.isValid():
            if isinstance(editor, QtGui.QFileDialog):
                sfiles = editor.selectedFiles()
                print 'sfiles', len(sfiles)
                if len(sfiles) > 0:
                    fn = editor.selectedFiles()[0]
                    print 'fn:', fn
                    model.setData(index, fn, QtCore.Qt.EditRole)
            else:
                QtGui.QStyledItemDelegate.setModelData(self, editor, model, index)


##### Main Window ####################
class DomViewer(QtGui.QMainWindow):

    def __init__(self, xmlFile=None, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.options = {
            'showAttributeTable' : False,
            'showComments' : True,
            }
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        if not self.options['showAttributeTable']:
            self.ui.attributeTable.hide()
        self.statusLabel = QtGui.QLabel()
        self.statusBar().addWidget(self.statusLabel)
        self.modified = False
        if debug: print 'connect ...'
        self.connect(self.ui.actionOpen, QtCore.SIGNAL('triggered()'),
                     self.openFile)
        self.connect(self.ui.actionSave, QtCore.SIGNAL('triggered()'),
                     self.saveFile)
        self.connect(self.ui.actionSave_As, QtCore.SIGNAL('triggered()'),
                     self.saveFileAs)
        self.connect(self.ui.actionQuit, QtCore.SIGNAL('triggered()'),
                     self.close)
        self.connect(self.ui.actionDuplicate, QtCore.SIGNAL('triggered()'),
                     self.duplicateSelected)
        self.connect(self.ui.actionCut, QtCore.SIGNAL('triggered()'),
                     self.cutSelected)
        self.connect(self.ui.actionCopy, QtCore.SIGNAL('triggered()'),
                     self.copySelected)
        self.connect(self.ui.actionPaste, QtCore.SIGNAL('triggered()'),
                     self.paste)
        self.connect(self.ui.actionDelete, QtCore.SIGNAL('triggered()'),
                     self.removeSelected)

        self.connect(self.ui.treeView, QtCore.SIGNAL('clicked(QModelIndex)'),
                     self.updateAttributeInfo)
        self.connect(self.ui.attributeTable,
                     QtCore.SIGNAL('cellChanged(int,int)'),
                     self.updateAttribute)
        self.connect(self.ui.actionShow_attributes,
                     QtCore.SIGNAL('triggered()'),
                     self.showAttributeTable)
        self.connect(self.ui.actionShow_comments,
                     QtCore.SIGNAL('triggered()'),
                     self.showComments)
        self.setWindowTitle(self.getCommandName())
        if debug: print 'load XML ...'
        if xmlFile is not None:
            self.loadXML(xmlFile)
        if debug: print 'XML loaded !'
        self.tmpNode = None #Used for copy/cut/paste

        self.ui.actionShow_comments.setChecked(self.options['showComments'])
        if debug: print 'create item delegate ...'
        delegate = XMLEditorDelegate(self)
        if debug: print 'setItemDelegate ...'
        self.ui.treeView.setItemDelegate(delegate)

    def openFile(self):
        fileName = str(QFileDialog.getOpenFileName(self, 'Open File ...', './',
                                                   self.tr('XML (*.xml *.html)'),
                                                   ))
        self.loadXML(fileName)

    def loadXML(self, fn):
        try:
            f = open(fn)
            content = f.read()
            f.close()
        except Exception, e:
            QtGui.QMessageBox.critical(self,self.getCommandName(),
                                       "Opening of " + fn + " failed!\n" \
                                           "Error was :\n" + e.message)
            return False
        if debug: print 'Creating DOM doc ...'
        self.domDoc = QtXml.QDomDocument() # Maybe use SAX here ?
        if debug: print 'DOM doc created!'
        (loadSuccess, errMsg, errLine, errCol) = self.domDoc.setContent(content)
        if not loadSuccess:
            msg = "XML parse in " + fn + " failed!\n" \
                "Error at line %d, column %d:\n" %(errLine, errCol) \
                + errMsg
            QtGui.QMessageBox.critical(self,self.getCommandName(),msg)
            return False
        self.fileName = fn
        if debug: print 'Creating DomModel ...'
        self.domModel = DomModel(self.domDoc, self)
        if debug: print 'DomModel created!'
        self.domModel.showComments(self.options['showComments'])
        if debug: print 'Setting model',
        self.ui.treeView.setModel(self.domModel)
        if debug: print 'Done !'
        self.ui.treeView.setColumnHidden(1, True)
        if debug: print 'Expanding ...'
        self.ui.treeView.expandAll()
        if debug: print 'Resize ...'
        self.ui.treeView.resizeColumnToContents(0)
        if debug: print 'Connect ...'
        self.connect(self.domModel,
                     QtCore.SIGNAL('dataChanged(QModelIndex, QModelIndex)'),
                     self.setModifiedState)
        self.setFileState(modified=False)
        return True

    def saveFile(self):
        if self.saveXML(self.fileName):
            self.unsetModifiedState()
            return True
        else:
            return False

    def saveFileAs(self):
        fn = str(QFileDialog.getSaveFileName(self, 'Save File As ...', './',
                                             self.tr('XML (*.xml *.html)')
                                             ))

        if self.saveXML(self.fileName):
            self.fileName = fn
            self.setFileState(False)
            return True
        else:
            return False

    def getCommandName(self):
        return 'xmledit'

    def setFileState(self, modified):
        self.modified = modified
        fn = os.path.basename(self.fileName)
        self.setWindowTitle(self.getCommandName()+' - '+fn+['','*'][modified])
        self.statusLabel.setText(fn + ['',' UNSAVED'][modified])

    def saveXML(self, fn):
        try:
            f = open(fn, 'w')
            f.write(self.domDoc.toString())
            f.close()
        except Exception, e:
            QtGui.QMessageBox.critical(self,self.getCommandName(),
                                       "Saving of "+self.fileName+" failed!\n" \
                                           "Error was :\n"+e.message)
            return False

        return True

    # SLOTS
    def showAttributeTable(self):
        if self.ui.actionShow_attributes.isChecked():
            self.options['showAttributeTable'] = True
        else:
            self.options['showAttributeTable'] = False
        if self.options['showAttributeTable']:
            self.ui.attributeTable.show()
        else:
            self.ui.attributeTable.hide()

    def showComments(self):
        if self.ui.actionShow_comments.isChecked():
            self.options['showComments'] = True
        else:
            self.options['showComments'] = False
        if self.options['showComments']:
            self.domModel.showComments()
        else:
            self.domModel.showComments(False)
        #self.ui.treeView.update(
        self.ui.treeView.expandAll()

    def setModifiedState(self, *args):
        self.setFileState(modified=True)

    def unsetModifiedState(self, *args):
        self.setFileState(modified=False)

    def resizeNameColumn(self, index):
        #print 'resizeNameColumn ...'
        self.ui.treeView.resizeColumnToContents(0)

    def updateAttributeInfo(self, index):
        #TODO : maybe use an Attribute Table Model to handle
        # access to attribute data
        # -> could be more powerfull for concurrent access
        # -> could be easier to expose in a table view
        self.disconnect(self.ui.attributeTable,
                        QtCore.SIGNAL('cellChanged(int,int)'),
                        self.updateAttribute)
        if index.isValid():
            nodeItem = index.internalPointer()
            self.selectedNode = nodeItem
            self.curAttrNames, self.curAttrValues = nodeItem.get_attributes()
            row = 0
            self.ui.attributeTable.setRowCount(len(self.curAttrValues))
            for an,av in zip(self.curAttrNames, self.curAttrValues):
                item = QtGui.QTableWidgetItem(an) #name
                self.ui.attributeTable.setItem(row,0,item)
                item = QtGui.QTableWidgetItem(av) #value
                self.ui.attributeTable.setItem(row,1,item)
                row += 1
        self.connect(self.ui.attributeTable,
                     QtCore.SIGNAL('cellChanged(int,int)'),
                     self.updateAttribute)

    def updateAttribute(self, row, col):
        #print 'updateAttribute ...', row, col
        item = self.ui.attributeTable.item(row, col)
        print item.text()
        name = self.curAttrNames[row]
        #print 'name :', name
        node = self.selectedNode.node()
        attributes = node.attributes()
        if col == 0: #name
            #print 'name changed'
            value = node.attributes().removeNamedItem(oldName).nodeValue()
            node.setAttribute(item.text(), value)
        elif col == 1: #value
            #print 'value changed'
            node.attributes().namedItem(name).setNodeValue(item.text())
        self.setModifiedState()


    def removeSelected(self):
        index = self.ui.treeView.selectionModel().currentIndex()
        model = self.ui.treeView.model()
        if model.removeRow(index.row(),index.parent()):
            self.setFileState(modified=True)

    def cutSelected(self):
        index = self.ui.treeView.selectionModel().currentIndex()
        model = self.ui.treeView.model()
        cutNode = model.take_node(index.row(),index.parent())
        if not cutNode.isNull():
            self.tmpNode = cutNode
            self.setFileState(modified=True)

    def paste(self):
        if self.tmpNode is not None:
            index = self.ui.treeView.selectionModel().currentIndex()
            model = self.ui.treeView.model()
            if model.insert_node(0, index, self.tmpNode.cloneNode()):
                self.setFileState(modified=True)

    def copySelected(self):
        index = self.ui.treeView.selectionModel().currentIndex()
        model = self.ui.treeView.model()
        copiedNode = model.copy_node(index)
        if not copiedNode.isNull():
            self.tmpNode = copiedNode

    def duplicateSelected(self):
        index = self.ui.treeView.selectionModel().currentIndex()
        model = self.ui.treeView.model()
        if model.duplicate(index):
            self.setFileState(modified=True)

    def insertChild(self):
        index = self.ui.treeView.selectionModel().currentIndex()
        model = self.ui.treeView.model()

        if not model.insertRow(0,index):
            return

        flag = QtGui.QItemSelectionModel.ClearAndSelect
        idx = model.index(0,0,index)
        self.ui.treeView.selectionModel().setCurrentIndex(idx, flag)

    # EVENTS HANDLING
    def closeEvent(self, event):
        if self.maybeSave():
            event.accept()
        else:
            event.ignore()

    def maybeSave(self):
        if self.modified:
            msg = "Document has been modified.\n" \
                "Do you want to save your changes?"
            flag = QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard \
                | QtGui.QMessageBox.Cancel
            ret = QtGui.QMessageBox.warning(self, self.getCommandName(),
                                            msg, flag)
            if ret == QtGui.QMessageBox.Save:
                return self.saveFile()
            elif ret == QtGui.QMessageBox.Cancel:
                return False

        return True

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    if len(sys.argv) > 1:
        myapp = DomViewer(sys.argv[1])
    else:
        myapp = DomViewer()
    myapp.show()
    sys.exit(app.exec_())





## TODOS :
# Add comments from Dom example
# Customize a bit:
#   X- show value & comments in name column when in leaf
#   X- expand all tree action (default : all expanded)
#   X- open file menu item
#   X- title bar shows filename + doc status (* if not saved)
#   X- direct modification of underlying DOM model
#   X- save
#   X- hide headers
#   X- do no show attributes & value columns
#   X- quit action -> ask to save if unsaved modifications
#   X- add QListWidget on the left side to display/edit current attributes
#   X- duplicate action
#   X- delete action
#   X- copy / cut / paste actions / delete
#   X- show/hide comments
#    - bug cancel action QFileDialog
#    - undo function -> easy: save a list of last domModels
#   X- add some icons to nodes / comments / leaf value
#    - keyboard shortcuts for actions
#    - create new node
#    - search function
#    - contextual menu -> new child, delete, copy, paste, duplicate
#    - drag&drop
#    - rich text to better display comments and text -> use delegate

