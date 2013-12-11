import sys
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

class Initable(object):
    """
    Abstract class to keep track of how an object is initialised.
    To do so, it stores the init function and its parameters.
    The aim is to use it in a user interface or to serialize objects.
    It also allows to add comments and meta info on init parameters.

    Should replace pyhrf.xmlio.XMLable2
    """
    def __init__(self):

        # gathers all names and values for init parameters:
        self._init_parameters = OrderedDict()

        # function used to instanciate the current object:
        self._init_obj = self.__class__

        # inspect the previous frame to get the values of the
        # the init parameters:
        frame,_,_,_,_,_ = getouterframes(currentframe())[1]
        args, _, _, values = getargvalues(frame)
        pyhrf.verbose(6, 'self.__class__: %s' %str(self.__class__))
        pyhrf.verbose(6, 'args: %s' %str(args))
        pyhrf.verbose(6, 'values: %s' %str(values))

        values.pop('self', None)
        if 'self' in args:
            args.remove('self')
        for k in args:
            self._init_parameters[k] = copy(values[k])

        pyhrf.verbose(6, 'self._init_parameters:\n%s'%str(self._init_parameters))

        if hasattr(self, 'parametersComments'):
            for pname in self.parametersComments.keys():
                ip = self._init_parameters.keys()
                if pname not in ip:
                    raise Exception('Entry "%s" in parametersComments is not in '\
                                    'init parameters (%s)' \
                                    %(pname,','.join(ip)))

        if hasattr(self, 'parametersToShow'):
            for pname in self.parametersToShow.keys():
                ip = self._init_parameters.keys()
                if pname not in ip:
                    raise Exception('Entry "%s" in parametersToShow is not in '\
                                    'init parameters (%s)' \
                                    %(pname,','.join(ip)))

        # translations of method arguments for user interface:
        self.arg_translation = {}
        self.arg_itranslation = {}



    def check_init_obj(self, params=None):
        """ check if the function used for init can be used in this API
        -> must allow **kwargs and *args.
        All arguments must have a value: either a default one or specified in
        the input dict *params*
        """
        args, varargs, varkw, defaults = getargspec(self._init_obj)
        if varkw is not None: #we don't want **kwargs arguments
            raise Exception('Keywords dict argument (eg **kwargs) ' \
                                'not supported (init function:%s).' \
                                %str(self._init_obj))
        if varargs is not None: #we don't want *args arguments
            raise Exception('Positional list argument (eg *args) ' \
                                ' not supported (init function:%s)' \
                                %str(self._init_obj))
        if args[0] == 'self' :
            args = args[1:]
        if params is not None:
            if set(params.keys()) != set(args):
                raise Exception('Some arguments do not  '\
                                    'have a value: %s' \
                                    %(str(set(args).difference(params.keys()))))

        elif len(args)-1 != len(defaults):
            pos_args = args[:len(args)-len(defaults)]
            print 'pos_args:', pos_args
            raise Exception('In init function "%s.%s", some arguments are '\
                                'not keywords arguments: %s' \
                                %(str(self.__class__),
                                  str(self._init_obj.__name__),
                                  ','.join(pos_args)))


    def set_init_param(self, param_name, param_value):
        if not self._init_parameters.has_key(param_name):
            raise Exception('"%s" is not an argument of init function %s' \
                                %(param_name, self._init_obj))
        self._init_parameters[param_name] = copy(param_value)


    def set_init(self, init_obj, **init_params):
        """ Override init function with *init_obj* and use *init_params*
        as new init parameters. *init_obj* must return an instance of the
        same class as the current object. Useful when the object is not
        instanciated via its __init__ function but eg a static method.
        """
        self._init_obj = init_obj
        args, varargs, varkw, defaults = getargspec(self._init_obj)

        self.check_init_obj(init_params)
        self._init_parameters = dict((k,copy(v)) \
                                         for k,v in init_params.iteritems())

        for ip in self._init_parameters:
            if ip not in args:
                raise Exception('Init parameter "%s" is not an argument of '\
                                    'init function "%s". Args are: %s' \
                                    %(ip,str(self._init_obj), ','.join(args)))

    def get_parameters_comments(self):
        if debug2: print 'get_parameters_comments ...'
        return getattr(self, 'parametersComments', {})


    def get_parameters_meta(self):
        if debug2: print 'get_parameters_meta ...'
        #TODO
        return {}

    def get_parameters_to_show(self):
        if debug2: print 'get_parameters_to_show ...'
        return getattr(self, 'parametersToShow', [])

    def init_new_obj(self):
        """ Creates a new instance
        """
        # if self._init_obj.__name__ == '__init__':
        #     return self._init_obj.im_self.__class__(**self._init_parameters)
        # else:

        return self._init_obj(**self._init_parameters)


    def to_ui_node(self, label, parent=None):
        pyhrf.verbose(6, 'Initable.to_ui_node(label=%s) ...' %label)
        n = UiNode(label, parent, {'init_obj':self._init_obj,'type':'Initable'})

        for pname, pval in self._init_parameters.iteritems():
            pyhrf.verbose(6, 'pname: %s, pval: %s' %(str(pname),str(pval)))
            n.add_child(UiNode.from_py_object(pname, pval))
        return n

    @PickleableStaticMethod
    def from_ui_node(self, node):
        pyhrf.verbose(6, 'Initable.from_ui_node [%s]' %node.label())
        init_obj = node.get_attribute('init_obj')
        node_type = node.get_attribute('type')
        if init_obj is not None:
            params = dict((c.label(),self.from_ui_node(c)) \
                          for c in node.get_children())
            pyhrf.verbose(6, '-> init with params: %s' %str(params))
            if init_obj.__name__ == '__init__':
                return init_obj.im_self.__class__(**params)
            else:
                return init_obj(**params)
        elif node_type is not None:
            pyhrf.verbose(6, '-> typped value (%s)' %str(node_type))
            if isinstance(node_type, str):
                if node_type == 'ndarray':
                    sdtype = node.get_attribute('dtype')
                    if sdtype.isdigit(): sdtype = int(sdtype)
                    sh = node.get_attribute('shape')
                    if pyhrf.verbose.verbosity > 2:
                        print 'node:'
                        print node.log()
                    if node.childCount() > 0:
                        array_data = node.child(0)._label
                    else: #0d array
                        array_data = ''
                    return numpy_array_from_string(array_data, sdtype, sh)
                elif node_type == 'list':
                    return list(self.from_ui_node(c) \
                                for c in node.get_children())
                elif node_type in ['odict','dict']:
                    dclass = [OrderedDict, dict][node_type == 'dict']
                    return dclass((c.label(),self.from_ui_node(c)) \
                                  for c in node.get_children())
            else:
                pyhrf.verbose(6, 'casting to node_type ...')
                return node_type(node.child(0)._label)
        else:
            l = node.label()
            pyhrf.verbose(6, '-> direct value: %s' %str(l))
            return l

    def set_arg_translation(self, a, t):
        """ Set the display name of argument *a* as *t*
        """
        self.arg_translation[a] = t
        self.arg_itranslation[t] = a

    def get_arg_for_ui(self, a):
        return self.arg_translation.get(a,a)

    def get_arg_from_ui(self, a):
        return self.arg_itranslation.get(a,a)

class UiNode(object):

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

    def __init__(self, label, parent=None, attributes=None):

        super(UiNode, self).__init__()

        pyhrf.verbose(6, 'Create new UiNode(%s,%s,%s)' \
                        %(label, str(parent), str(attributes)))

        self._label = label
        self._children = []
        self._parent = parent
        if attributes is None:
            attributes = {}

        # Check input attributes (has to be a dict of strings):
        if not isinstance(attributes, dict):
            raise Exception('Wrong type "%s" for attributes, has to be a dict'\
                            %str(type(attributes)))

        for k,v in attributes.iteritems():
            if not isinstance(k, str):
                raise Exception('Wrong type for attribute "%s" '\
                                '(has to be string)' %str(k))
            if k not in ['init_obj', 'type'] and not isinstance(v, str):
                # init_obj can be a method/function (see Initable class)
                raise Exception('Wrong type for value of attribute "%s" (%s) '\
                                '(has to be string)' %(str(k),str(v)))

        self._attributes = attributes


        if parent is not None:
            parent.add_child(self)


    def set_attribute(self, attr_name, attr_value):
        self._attributes[attr_name] = attr_value

    def get_attribute(self, attr_name):
        return self._attributes.get(attr_name)

    def has_attribute(self, attr_name):
        return self._attributes.has_key(attr_name)

    @PickleableStaticMethod
    def _serialize_init_obj(self, f):
        """ Return a dict of strings describing function f.
        """
        pyhrf.verbose(6, 'serialize init func %s' %(str(f)))

        if not isclass(f) and f.im_func.__name__ == '__init__':
            f = f.im_class

        return {'pickled_init_obj':pickle.dumps(f)}

        # if isinstance(f, PickleableStaticMethod):
        #     f = f.fn
        # return {'init_python_class' : f.__class__.__name__,


    @PickleableStaticMethod
    def _unserialize_init_obj(self, d, pop=False):
        """ Return a function from info contained in dict *d*
        """
        if pop:
            pf = d.pop('pickled_init_obj', None)
        else:
            pf = d.get('pickled_init_obj')
        if pf is not None:
            return pickle.loads(pf)
        else:
            return None

    def _serialize_type(self, t):
        """ Return a dict of strings describing type t.
        """
        return {'pickled_type':pickle.dumps(t)}

    @PickleableStaticMethod
    def _unserialize_type(self, d, pop=False):
        if pop:
            pt = d.pop('pickled_type', None)
        else:
            pt = d.get('pickled_type')
        if pt is not None:
            return pickle.loads(pt)
        else:
            return None


    def serialize_attributes(self):
        a = {}
        for k,v in self._attributes.iteritems():
            if k == 'init_obj':
                a.update(self._serialize_init_obj(v))
            elif k == 'type':
                a.update(self._serialize_type(v))
            else:
                a[k] = v
        return a

    @PickleableStaticMethod
    def unserialize_attributes(self, a):
        dest = a.copy() #make a copy 'cause things will be popping
        pyhrf.verbose(6, 'unserialize_attributes: %s' %str(a))
        for k,v in a.iteritems():
            init_obj = self._unserialize_init_obj(dest, pop=True)
            if init_obj is not None:
                dest['init_obj'] = init_obj
                dest['type'] = 'Initable'
            else:
                t = self._unserialize_type(dest, pop=True)
                if t is not None:
                    dest['type'] = t

        return dest

    def set_label(self, label):
        self._label = label
        return True

    @PickleableStaticMethod
    def _pyobj_has_leaf_type(self, o):
        """ Return true if object *o* can be stored in a leaf Node, ie
        if it can be serialized in a simple human-readable string.
        """
        return isinstance(o, (int, float, str, np.ndarray)) or \
          (isinstance(o, list) and \
           all([isinstance(e, (int, float, str)) for e in o])) or \
            np.isscalar(o) or o is None
    # or \
    #        (isinstance(o, (dict, OrderedDict)) and \
    #         all([isinstance(e, (int, float, str)) for e in o.values()]))
    

    @PickleableStaticMethod
    def from_py_object(self, label, obj, parent=None):

        pyhrf.verbose(6, 'UiNode.from_py_object(label=%s,obj=%s) ...' \
                      %(label, str(obj)))

        if isinstance(obj, Initable):
            n = obj.to_ui_node(label, parent)
        else:
            if UiNode._pyobj_has_leaf_type(obj):
                if isinstance(obj, np.ndarray):
                    dt = str(obj.dtype.name)
                    sh = str(obj.shape)
                    n = UiNode(label, attributes={'type':'ndarray',
                                                  'dtype':dt,
                                                  'shape':sh})
                    s = ' '.join( str(e) for e in obj.ravel() )
                    n.add_child(UiNode(s))
                elif obj is None:
                    n = UiNode(label, attributes={'type':'None'})
                    n.add_child(UiNode('None'))
                else:
                    n = UiNode(label, attributes={'type':obj.__class__})
                    n.add_child(UiNode(str(obj)))
            elif isinstance(obj, list):
                n = UiNode(label, attributes={'type':'list'})
                for i,sub_val in enumerate(obj):
                    n.add_child(UiNode.from_py_object('item%d'%i, sub_val))
            elif isinstance(obj, (dict, OrderedDict)):
                t = ['odict','dict'][obj.__class__ == dict]
                n = UiNode(label, attributes={'type':t})
                for k,v in obj.iteritems():
                    n.add_child(UiNode.from_py_object(k, v))
            else:
                raise Exception('In UiNode.from_py_object, unsupported object: '\
                                '%s (type: %s)' %(str(obj), str(type(obj))))
        return n


    def label(self):
        return self._label

    def type_info(self):
        return self._attributes.get('type')

    def to_xml(self):
        """
        Return an XML representation of the Node and its children.
        """
        doc = QtXml.QDomDocument()

        node = doc.createElement(self.label())

        doc.appendChild(node)

        for k, v in self.serialize_attributes().iteritems():
            node.setAttribute(k, v)

        for c in self.get_children():
            c._recurse_to_xml(doc, node)

        return doc.toString(indent=4)


    def _recurse_to_xml(self, doc, parent):
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
            c._recurse_to_xml(doc, node)



    @PickleableStaticMethod
    def from_xml(self, sxml):
        doc = QtXml.QDomDocument()

        # parser = QtXml.QXmlSimpleReader()
        # qinput = QtXml.QXmlInputSource()
        # qinput.setData(sxml)
        #doc.setContent(qinput, parser)

        doc.setContent(sxml)
        if debug:
            print 'QDomDocument:'
            print doc.toString()

        return self._recurse_from_xml(doc.documentElement())

    @PickleableStaticMethod
    def _recurse_from_xml(self, qnode):
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
            n.add_child(self._recurse_from_xml(child_nodes.item(i)))

        return n


    def log(self, tabLevel=-1):

        output     = ""
        tabLevel += 1

        for i in range(tabLevel):
            output += "\t"

        output += "|------" + self._label
        if len(self._attributes) > 0:
            output += ' |'
            for k,v in self._attributes.iteritems():
                if isfunction(v) or ismethod(v):
                    v = v.__name__
                elif hasattr(v, 'im_func'):
                    v = v.im_func
                output += ' %s:%s' %(str(k), str(v))
        output += '\n'

        for child in self._children:
            output += child.log(tabLevel)

        tabLevel -= 1
        output += "\n"

        return output


    def get_children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)
        child._parent = self

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


    def child(self, row):
        return self._children[row]

    def childCount(self):
        return len(self._children)

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

