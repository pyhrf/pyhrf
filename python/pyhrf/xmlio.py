# -*- coding: utf-8 -*-
import pickle
import re
import numpy as np
from inspect import currentframe, getargvalues, getouterframes, getargspec
from inspect import ismethod, isfunction, isclass
from copy import copy

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

import pyhrf
from pyhrf.tools import PickleableStaticMethod

NEWLINE_TAG = '##'

class DeprecatedXMLFormatException(Exception):
    pass

def protect_xml_attr(sa):
    if NEWLINE_TAG in sa:
        raise Exception('Cannot safely protect new line chars')
    return sa.replace('\n', NEWLINE_TAG)

def unprotect_xml_attr(sa):
    return sa.replace(NEWLINE_TAG, '\n')

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
        a.shape = tuple( int(e) for e in sshape.strip('(),').split(',') )
    return a


class Initable(object):
    """
    Abstract class to keep track of how an object is initialised.
    To do so, it stores the init function and its parameters.
    The aim is to use it in a user interface or to serialize objects.
    It also allows to add comments and meta info on init parameters.
    """
    _init_obj = None
    _init_parameters = None

    def __init__(self):
        if self._init_obj is None:
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

            pyhrf.verbose(6, 'self._init_parameters:\n%s' \
                          %str(self._init_parameters))

            #
            if hasattr(self, 'parametersComments'):
                ip = self._init_parameters.keys()
                pc = self.parametersComments
                pc = dict( (k,v) for k,v in pc.items() if k in ip)
                self.parametersComments = pc
            #     for pname in self.parametersComments.keys():
            #         ip = self._init_parameters.keys()
            #         if pname not in ip:
            #             raise Exception('Entry "%s" in parametersComments is '\
            #                             'not in init parameters (%s) of %s' \
            #                             %(pname,','.join(ip),
            #                               self.get_init_func()))

            if hasattr(self, 'parametersToShow'):
                ip = self._init_parameters.keys()
                for pname in self.parametersToShow:
                    if pname not in ip:
                        raise Exception('Entry "%s" in parametersToShow is not'\
                                        ' in init parameters (%s)' \
                                        %(pname,','.join(ip)))

            # translations of method arguments for user interface:
            self.arg_translation = {}
            self.arg_itranslation = {}

    def get_init_func(self):
        if isinstance(self._init_obj, PickleableStaticMethod):
            iobj = self._init_obj.fn
        else:
            iobj = self._init_obj
        return iobj

    def check_init_obj(self, params=None):
        """ check if the function used for init can be used in this API
        -> must allow **kwargs and *args.
        All arguments must have a value: either a default one or specified in
        the input dict *params*
        """
        ifunc = self.get_init_func()
        args, varargs, varkw, defaults = getargspec(ifunc)
        if varkw is not None: #we don't want **kwargs arguments
            raise Exception('Keywords dict argument (eg **kwargs) ' \
                                'not supported (init function:%s).' \
                                %str(ifunc))
        if varargs is not None: #we don't want *args arguments
            raise Exception('Positional list argument (eg *args) ' \
                                ' not supported (init function:%s)' \
                                %str(ifunc))
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
                                  str(ifunc.__name__),
                                  ','.join(pos_args)))


    def set_init_param(self, param_name, param_value):
        if not self._init_parameters.has_key(param_name):
            raise Exception('"%s" is not an argument of init function %s' \
                                %(param_name, self.get_init_func()))
        self._init_parameters[param_name] = copy(param_value)


    def set_init(self, init_obj, **init_params):
        """ Override init function with *init_obj* and use *init_params*
        as new init parameters. *init_obj* must return an instance of the
        same class as the current object. Useful when the object is not
        instanciated via its __init__ function but eg a static method.
        """
        self._init_obj = init_obj
        ifunc = self.get_init_func()
        args, varargs, varkw, defaults = getargspec(ifunc)

        self.check_init_obj(init_params)
        self._init_parameters = dict((k,copy(v)) \
                                     for k,v in init_params.iteritems())

        for ip in self._init_parameters:
            if ip not in args:
                raise Exception('Init parameter "%s" is not an argument of '\
                                    'init function "%s". Args are: %s' \
                                    %(ip,str(ifunc), ','.join(args)))

    def get_parameters_comments(self):
        pyhrf.verbose(6, 'get_parameters_comments ...')
        return getattr(self, 'parametersComments', {})


    def get_parameters_meta(self):
        pyhrf.verbose(6, 'get_parameters_meta ...')
        #TODO
        return {}

    def get_parameters_to_show(self):
        pyhrf.verbose(6, 'get_parameters_to_show ...')
        return getattr(self, 'parametersToShow', [])

    def init_new_obj(self):
        """ Creates a new instance
        """
        # if self._init_obj.__name__ == '__init__':
        #     return self._init_obj.im_self.__class__(**self._init_parameters)
        # else:

        return self._init_obj(**self._init_parameters)

    def assert_is_initialized(self):
        if self._init_obj is None or self._init_parameters is None:
            cname = self.__class__.__name__
            raise Exception('%s instance is not properly initialized for the ' \
                            'Initable API. Did you call Initable.__init__ in ' \
                            '%s.__init__ ?' %(cname, cname))

    def to_ui_node(self, label, parent=None):
        self.assert_is_initialized()
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
            if hasattr(init_obj, '__name__') and init_obj.__name__ == '__init__':
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
                    if pyhrf.verbose.verbosity >= 6:
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
                elif node_type == 'tuple':
                    return tuple(self.from_ui_node(c) \
                                for c in node.get_children())
                elif node_type in ['odict','dict']:
                    dclass = [OrderedDict, dict][node_type == 'dict']
                    return dclass((c.label(),self.from_ui_node(c)) \
                                  for c in node.get_children())
            elif node_type == bool:
                return node.child(0)._label == 'True'
            elif node_type == str and node.childCount() == 0: #emtpy string
                return ''
            else:
                pyhrf.verbose(6, 'casting "%s" to node_type %s ...' \
                              %(node.child(0)._label, str(node_type)))
                r = node_type(node.child(0)._label)
                pyhrf.verbose(6, '-> %s' %str(r))
                return r
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

XmlInitable = Initable # just an alias

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

        if not isclass(f) and hasattr(f, 'im_func') and \
          f.im_func.__name__ == '__init__':
            f = f.im_class

        return {'pickled_init_obj':protect_xml_attr(pickle.dumps(f))}

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
            return pickle.loads(unprotect_xml_attr(pf))
        else:
            return None

    def _serialize_type(self, t):
        """ Return a dict of strings describing type t.
        """
        return {'pickled_type':protect_xml_attr(pickle.dumps(t))}

    @PickleableStaticMethod
    def _unserialize_type(self, d, pop=False):
        if pop:
            pt = d.pop('pickled_type', None)
        else:
            pt = d.get('pickled_type')
        if pt is not None:
            return pickle.loads(unprotect_xml_attr(pt))
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
        for k,v in a.items():
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
          np.isscalar(o) or o is None
          # ((isinstance(o, list) or isinstance(o, list)) and \
          #  all([isinstance(e, (int, float, str)) for e in o])) or \
    # or \
    #        (isinstance(o, (dict, OrderedDict)) and \
    #         all([isinstance(e, (int, float, str)) for e in o.values()]))


    @PickleableStaticMethod
    def from_py_object(self, label, obj, parent=None):
        """
        Convert a python object into a UiNode object.
        Types that are supported for *obj*:
            - NoneType
            - int
            - float
            - str
            - list of supported types
            - dict of supported types
            - OrderedDict
            - numpy.ndarray
            - Initable
        """
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
                    #string representation of the flat array:
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
            elif isinstance(obj, tuple):
                n = UiNode(label, attributes={'type':'tuple'})
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

    def to_xml(self, pretty=False):
        """
        Return an XML representation (str) of the Node and its children.
        """
        from xml.dom.minidom import Document
        doc = Document()
        node = doc.createElement(self.label())
        doc.appendChild(node)
        for k, v in self.serialize_attributes().iteritems():
            node.setAttribute(k, v)

        for c in self.get_children():
            c._recurse_to_xml(doc, node)

        if pretty:
            return doc.toprettyxml(indent='    ') #TODO: manage encoding stuff?
        else:
            return doc.toxml()

    def is_leaf_node(self):
        return len(self.get_children()) == 0 and len(self._attributes) == 0

    def _recurse_to_xml(self, doc, parent):
        pyhrf.verbose(6, '_recurse_to_xml ...')

        if not self.is_leaf_node():
            node = doc.createElement(self.label())
        else: #leaf node -> use TextNode
            pyhrf.verbose(6, 'text node: %s' %self.label())
            node = doc.createTextNode(self.label())

        parent.appendChild(node)

        for k, v in self.serialize_attributes().iteritems():
            node.setAttribute(k, v)

        for c in self.get_children():
            c._recurse_to_xml(doc, node)

    @PickleableStaticMethod
    def from_xml(self, sxml):
        from xml.dom.minidom import parseString
        doc = parseString(str(sxml)) #TODO: ingore tab, \n
        return self._recurse_from_xml(doc.documentElement)

    @PickleableStaticMethod
    def _recurse_from_xml(self, node):
        pyhrf.verbose(6, '_recurse_from_xml node -> tag=%s, val=%s, type=%s ' \
                      %(str(getattr(node, 'tagName', 'N/A')), str(node.nodeValue),
                        str(node.nodeType)))
        if node.nodeType == node.TEXT_NODE and \
            re.match('^[ \n]*$', node.wholeText): #empty node
            return None
        a = node.attributes
        if a is not None:
            attributes = dict((str(a.item(i).nodeName),
                               str(a.item(i).nodeValue))\
                               for i in range(a.length))
            attributes = self.unserialize_attributes(attributes)
        else:
            attributes = None

        # Manage deprecated API:
        if getattr(node, 'tagName', 'N/A') == 'root' and \
          attributes is not None and attributes.has_key('pythonXMLHandlerModule'):
          raise DeprecatedXMLFormatException('Deprecated XML format (root node)')

        node_name = str(node.nodeName)
        if node_name != '#text':
            n = UiNode(node_name, attributes=attributes)
        else:
            n = UiNode(node.nodeValue, attributes=attributes)

        child_nodes = node.childNodes
        pyhrf.verbose(6, 'create new node %s (-> %d children)' \
                      %(node_name, len(child_nodes)))
        pyhrf.verbose(6, 'node value: %s' %str(node.nodeValue))
        for c in child_nodes:
            rc = self._recurse_from_xml(c)
            if rc is not None:
                n.add_child(rc)

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

    def childCount(self):
        return len(self._children)

    def child(self, row):
        return self._children[row]


def read_xml(fn):
    f = open(fn)
    xml = f.read()
    f.close()
    return from_xml(xml)

def write_xml(obj, fn):
    f = open(fn, 'w')
    f.write(to_xml(obj))
    f.close()

def to_xml(obj, label='anonym', pretty=False):
    """
    Return an XML representation of the init state of the given object *obj*.
    """
    return UiNode.from_py_object(label, obj).to_xml(pretty)

def from_xml(sxml):
    return Initable.from_ui_node(UiNode.from_xml(sxml))
