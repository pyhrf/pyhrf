# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import copy as copyModule
import re
import sys
import os
import string
import array
from copy import copy
from inspect import getmro, getsource, currentframe, getargvalues
from inspect import getargspec as _getargspec
from inspect import getouterframes, isfunction

try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict


from xml.dom.minidom import Document
from xml.dom.NodeFilter import NodeFilter
from xml.dom.ext.reader import Sax2

from pyhrf.tools import PickleableStaticMethod

debug = False
debug2 = False
debug3 = False

class FuncWrapper:
    def __init__(self, func, params=None):
        if params is None:
            params = {}
        self.func = func
        self.params = params


def getargspec(func):
    if isinstance(func, PickleableStaticMethod):
        return _getargspec(func.fn)
    else:
        return _getargspec(func)


class XMLable2(object):

    #_init_parameters = None
    #_parameters_value = None

    def __init__(self):

        self._init_parameters = {}

        self._init_func = self.__init__
        #self.check_init_func()
        frame,_,_,_,_,_ = getouterframes(currentframe())[1]
        args, _, _, values = getargvalues(frame)
        if debug:
            print 'self.__class__:', self.__class__
            print 'args:', args
            print 'values:', values
        if 'self' in values:
            values.pop('self')
        for k,v in values.iteritems():
            self._init_parameters[k] = copy(v)
        if debug:
            print 'self._init_parameters:'
            print self._init_parameters

        # args, varargs, varkw, defaults = getargspec(self._init_func)
        # args.remove('self')
        # self._init_parameters = dict( (k,v) for k,v in zip(args, defaults) )

        
        
    def check_init_func(self, params=None):
        
        args, varargs, varkw, defaults = getargspec(self._init_func)
        if varkw is not None: #we don't want **kwargs arguments
            raise Exception('Keywords dict argument (eg **kwargs) ' \
                                'not supported (init function:%s).' \
                                %str(self._init_func))
        if varargs is not None: #we don't want *args arguments
            raise Exception('Positional list argument (eg *args) ' \
                                ' not supported (init function:%s)' \
                                %str(self._init_func))
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
                                  str(self._init_func.__name__),
                                  ','.join(pos_args)))


    def set_init_param(self, param_name, param_value):
        if not self._init_parameters.has_key(param_name):
            raise Exception('"%s" is not an argument of init function %s' \
                                %(param_name, self._init_func))        
        self._init_parameters[param_name] = copy(param_value)


    def override_param_init(self, init_func,  **params):
        """ TODO (if needed)
        """
        pass

    def set_init(self, init_func, **init_params):

        self._init_func = init_func
        args, varargs, varkw, defaults = getargspec(self._init_func)

        self.check_init_func(init_params)
        self._init_parameters = dict((k,copy(v)) \
                                         for k,v in init_params.iteritems())
        
        for ip in self._init_parameters:
            if ip not in args:
                raise Exception('Init parameter "%s" is not an argument of '\
                                    'init function "%s". Args are: %s' \
                                    %(ip,str(self._init_func), ','.join(args)))

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




class XMLable:

    #_init_parameters = None
    #_parameters_value = None

    def __init__(self,**kwargs):
        #frame = currentframe()
        #if self._init_parameters is None:
        self._init_parameters = {}
        #if self._parameters_value is None:
        self._parameters_value = {}

        if debug:
            print 'XMLable.__init__ (subclass %s) ...' %self.__class__
            print 'self._init_parameters:'
            print self._init_parameters
            print 'self._parameters_value:'
            print self._parameters_value
        frame,_,_,_,_,_ = getouterframes(currentframe())[1]
        args, _, _, values = getargvalues(frame)
        if debug:
            print 'self.__class__:', self.__class__
            print 'args:', args
            print 'values:', values
        if 'self' in values:
            values.pop('self')
        self._init_parameters.update(values.copy())
        if debug:
            print 'self._init_parameters:'
            print self._init_parameters

    def set_init(self, init_func, init_params=None):
        
        if debug: 
            print '~~~~~~~~~~~~~~~~~~~~~'
            print 'set_init ......'
            print 'init_func:', init_params
            print ''

        if init_params is None:

            frame,_,_,_,_,_ = getouterframes(currentframe())[1]
            args, _, _, values = getargvalues(frame)
            # print 'args:', args
            # print 'values:', values
            init_params = dict([(k,values[k]) for k in args if k != 'self'])


        if debug: print 'init_params:', init_params

        self.__init__ = init_func
        self._init_parameters = {}
        self._init_parameters.update(init_params)
        # for k,v in init_params.iteritems():
        #     self._init_parameters[k] = v

    def override_init(self, param_name, init_obj, init_params=None):
        assert not self._parameters_value.has_key(param_name)
        if debug:
            print 'override_init of "%s" with %s for obj of class %s' \
                %(param_name, str(init_obj), self.__class__)
        self._init_parameters[param_name] = FuncWrapper(init_obj, init_params)

    def override_value(self, param_name, value):
        if debug:
            print 'override_value of "%s" with %s for obj of class %s' \
                %(param_name, str(value), self.__class__)
        #assert not self._init_parameters.has_key(param_name)
        self._parameters_value[param_name] = value

    def get_parameters_comments(self):
        return self._gather_parent_dict_parameters('parametersComments')

    def get_parameters_meta(self):
        return self._gather_parent_dict_parameters('parametersMeta')

    def get_parameters_to_show(self):
        if debug2: print 'get_parameters_to_show ...'
        #l = set()
        l = []

        if hasattr(self, "parametersToShow"):
            return self.parametersToShow
        else:
            return []
        
        # class_tree = getmro(self.__class__)
        # if debug3:
        #     print 'classTree:'
        #     print class_tree
        # for c in class_tree:
        #     if debug: print 'treating class', c
        #     if c != XMLable:
        #         if 'parametersToShow' in getsource(c): 
        #             #a bit dirty ...
        #             if debug: print 'found parametersToShow in source ...'
        #             if debug: print 'source:', getsource(c)
        #             p_to_show = getattr(c,'parametersToShow',None)
        #         else:
        #             if debug: print 'did not found parametersToShow '\
        #                     'in source ...'
        #             p_to_show = None
        #         if p_to_show is not None:
        #             for p in p_to_show:
        #                 if p not in l:
        #                     l.append(p)
        #         else: # assume we show everything
        #             print 'show all params ...'
        #             print 'c.__init__:', c.__init__
        #             print '-> argspec:', getargspec(c.__init__)
        #             l.extend(getargspec(c.__init__)[0][1:])
        #             #l.update(getargspec(c.__init__)[0])
        # return l

    

    def _gather_parent_dict_parameters(self, pname):
        if debug2: print '_gather_parent_dict_parameters ...'
        d = {}
        class_tree = getmro(self.__class__)
        if debug3:
            print 'classTree:'
            print class_tree
        for c in class_tree:
            d.update(getattr(c,pname,{}))
        return d

    def _gather_parent_list_parameters(self, pname):
        if debug2: print '_gather_parent_list_parameters ...'
        l = []
        class_tree = getmro(self.__class__)
        if debug3:
            print 'classTree:'
            print class_tree
        for c in class_tree:
            l.extend(getattr(c,pname,[]))
        return l


class TypedXMLHandler:
    """
    Class handling the xml format with the following generic document structure :
          <root>
            <tagName 'type'=tagType>
            tagContent 
            </tagName>
          </root>
    The root tag is mandatory, so is the 'type' attribute for every other tag. This class can parse an xml string and build a dictionnary of python objects according to each tag (see parseXMLString()). Conversely, it can build the xml string corresponding to a list or dictionnary of objects (see toXML()).  
    This class is based on the xml.dom python module and relies on the DOM structure to parse XML.
    XML input/output is handled via a mapping between a type attribute of a tag and a static handling function. This class handles the following basic python types : string, int, float, array.array, list, dict. One can add other type-specific read or write handlers with the functions addDOMTagReader() and addDOMWriter().
    
    Reading XML:
        - specific handlers are added with method addDOMTagReader(stype, myReadHandler) which maps the function myReadHandler to the string stype.
        - a tag reading handler must have the following prototype :
            myReadHandler(domTreeWalker):
                # interpret and process tag content
                # return built python object
            , where domTreeWalker is an instance of the _xmlplus.dom.TreeWalker.TreeWalker class.
        - usefull things to use the domTreeWalker and implement a handler:
            * node = domTreeWalker.currentNode -> current node in the tree parsing, of class Node, corresponding to the current tag.
            * node.getAttribute('my attribute') -> return the string corresponding to 'my attribute' in the tag definition
            * node.childNodes[0].data -> the tag content data (string type), to be parse and build the python object from
            * node.tagName -> the name of the tag
            * node.parentNode -> the parent tag node
            
    Writing XML :
        - handlers are added with method addDOMTagWriter(pythonType, myWriteHandler), where pythonType is of python type 'type' and myWriteHandler a function.
        - a tag writing handler must have the following prototype :
            myWriteHandler(domDocument, node, pyObj):
                where :
                    - domDocument is overall incapsulating structure (_xmlplus.dom.Document instance)
                    - node (Node instance) is the current tag node to append data to
                    - pyObj is the python object to convert into a 'human-readable' string.
        - usefull things to write handlers :
            
            
            
    
    """
    TYPE_LABEL_NONE = 'none'
    TYPE_LABEL_BOOL = 'bool'
    TYPE_LABEL_STRING = 'char'
    TYPE_LABEL_FLOAT = 'double'
    TYPE_LABEL_INT = 'int'
    TYPE_LABEL_DICT = 'struct'
    TYPE_LABEL_ODICT = 'ordered_struct'
    TYPE_LABEL_LIST = 'list'
    TYPE_LABEL_TUPLE = 'tuple'
    TYPE_LABEL_ARRAY = 'array'
    TYPE_LABEL_KEY_VAL_PAIR = 'dictItem'
    TYPE_LABEL_XML_INCLUDE = 'include'

    ATTRIBUTE_LABEL_PYTHON_CLASS = 'pythonClass'
    ATTRIBUTE_LABEL_PYTHON_CLASS_INIT_MODE = 'pythonInitMode'
    ATTRIBUTE_LABEL_PYTHON_MODULE = 'pythonModule'
    ATTRIBUTE_LABEL_PYTHON_FUNCTION = 'pythonFunction'
    ATTRIBUTE_LABEL_TYPE = 'type'
    ATTRIBUTE_LABEL_META = 'meta'

    def __init__(self, write_callback=None):
        (self.tagDOMReaders, self.objectDOMWriters) = self.packHandlers()
        
        self.write_callback = write_callback


    def __setstate__(self, dic):
        (dic['tagDOMReaders'], dic['objectDOMWriters']) = self.packHandlers()
        self.__dict__ = dic
        
    def __getstate__(self): # use for compatibilty with pickles (which doesn't support instance variable methods ...)
        #print self.__class__,'- getstate ... removing static readers/writers ...'
        # copy the __dict__ so that further changes
        # do not affect the current instance
        d = copyModule.deepcopy(self.__dict__)
        # remove the closure that cannot be pickled
        if d.has_key('tagDOMReaders'):
            del d['tagDOMReaders']
        if d.has_key('objectDOMWriters'):
            del d['objectDOMWriters']
        # return state to be pickled
        return d


    def createDocument(self):
        doc = Document()
        root = doc.createElement("root")
        root.setAttribute("pythonXMLHandlerModule",self.__class__.__module__)
        root.setAttribute("pythonXMLHandlerClass",self.__class__.__name__)
        doc.appendChild(root)
        return (doc, root)

    def parseXMLString(self, xml):

        xml = xml.replace('\t', '')
        xml = xml.replace('\n', '')

        # create Reader object
        reader = Sax2.Reader()

        # parse the document and load it in a DOM tree
        domTree = reader.fromString(xml)
        
        walker = domTree.createTreeWalker(domTree.documentElement, NodeFilter.SHOW_ELEMENT, None, 0)

        #print 'walker.currentNode.tagName :', walker.currentNode.tagName
        rootNode = walker.currentNode
        if rootNode.hasAttribute('pythonXMLHandlerModule') and rootNode.hasAttribute('pythonXMLHandlerClass'):
            modulesName = rootNode.getAttribute('pythonXMLHandlerModule')
##            print 'modulesName :', modulesName
            module = eval('__import__("'+modulesName+'", fromlist=[""])')
##            print 'module : ', module
            className = rootNode.getAttribute('pythonXMLHandlerClass')
            xmlHandler = eval('module.'+className+'()')
        else:
            raise Exception('No Typed XML Handler declared in root tag!')
        return xmlHandler.rootTagDOMReader(walker) # An XML document always starts with the root tag

    def buildXMLString(self, obj, label=None, pretty=False):
        (doc,root) = self.createDocument()
        if label is None:
            if hasattr(obj,'xml_label'):
                label = obj.xml_label
            else:
                label = 'anonymObject'
            
        self.writeDOMData(doc, root, obj, label)
        if not pretty:
            return doc.toxml()
        else:
            return doc.toprettyxml()


    def rootTagDOMReader(self, walker):
        data = {}
        self.parseTerminated = False
        try :
            while(walker.next() and not self.parseTerminated):
                name = walker.currentNode.tagName
                if debug:
                    print '-- name :', name
                data[name] = self.readDOMData(walker)
        except StopIteration:
            pass
        if debug:
            print 'rootTagDOMReader : result :'
            print data
        if len(data)==1:
            return data.values()[0]
        else:
            return data

    def readDOMData(self, walker):
        cn = walker.currentNode
        cnType = cn.getAttribute(self.ATTRIBUTE_LABEL_TYPE)
        to_override = {}
        if debug:
            if debug and 'Prefix' in cn._get_nodeName():
                print 'readDOMData ...'
                print 'cnType:', cnType
                print 'name:', cn._get_nodeName()
                print 'spa:', cn.getAttributeNS('http://www.w3.org/XML/1998/namespace','space')
                if cn.hasAttribute('xml:space'):
                    print 'xml:space found ->', cn.getAttribute('xml:space')
                if cn.hasAttribute('size'):
                    print 'size found'
        # Shade comments and empty childs:
        childsToRemove = []
        preserveSpace = cn.getAttributeNS('http://www.w3.org/XML/1998/namespace',
                                          'space') == "preserve"
        for child in cn.childNodes:

            if (child.nodeType == child.COMMENT_NODE) or \
                    hasattr(child,'data') and str(child.data).strip()=='' and \
                    not preserveSpace:
                childsToRemove.append(child)
                
        for child in childsToRemove:
            cn.removeChild(child)
        if cn.hasAttribute(self.ATTRIBUTE_LABEL_PYTHON_CLASS) or \
                cn.hasAttribute(self.ATTRIBUTE_LABEL_PYTHON_FUNCTION):
            if debug: print 'building param dict ...'
            paramDict = self.tagDOMReaders[self.TYPE_LABEL_DICT](walker, self)
            for k,v in paramDict.items():
                if isinstance(v, FuncWrapper):
                    if debug: print '%s is FuncWrapper instance' %k
                    paramDict[k] = v.func(**v.params)
                    to_override[k] = v

            #TODO: parse param dict to catch overridings

            #paramDict = dict( [(str(k),v) for k,v in paramDict.iteritems()] )
            if debug:
                print '*** param dict :'
                print paramDict
                
            #HACK
            if 'BoldEstimationModel' in paramDict:
                paramDict['sampler'] = paramDict.pop('BoldEstimationModel')

            moduleName = cn.getAttribute(self.ATTRIBUTE_LABEL_PYTHON_MODULE)
            module = __import__(moduleName, fromlist=['']) # when fromlist is
                                                           # not empty -> return
                                                           # submodule
##            print 'moduleName = ', moduleName
##            print 'module = ', module
            className = cn.getAttribute(self.ATTRIBUTE_LABEL_PYTHON_CLASS)

            creationMode = cn.getAttribute(self.ATTRIBUTE_LABEL_PYTHON_CLASS_INIT_MODE)
            func_name = cn.getAttribute(self.ATTRIBUTE_LABEL_PYTHON_FUNCTION)
            if debug: 
                print 'className:', className
                print 'creationMode:', creationMode
                print 'func_name:', func_name
            
            if creationMode == 'XMLParamDrivenClass':
                if debug:
                    print 'Building XMLParamDrivenClass ...'
                    print 'to eval :'
                    print 'module.' +className+'(parameters=paramDict, xmlHandler=self, xmlLabel=cn.tagName)'
                try:
                    obj = eval('module.' + className + \
                                   '(parameters=paramDict,' + \
                                   ' xmlHandler=self, xmlLabel=cn.tagName)')
                except TypeError, e:
                    if debug: 
                        print 'TypeError while eval ->'
                        print e
                    match_init_args(eval('module.'+className+'()'), paramDict)
                    obj = eval('module.'+className+'(**paramDict)')
            else :
                # Build parameters coma-separated list :
                if debug2: 
                    print 'Build parameters coma-separated list :'
                python_ver = sys.version_info
                if python_ver[0] == 2 and python_ver[1] <= 6 \
                      and python_ver[2] <= 4: #<python2.5
                    #print 'fix paramDict!'
                    paramDict = dict([(str(k),v) \
                                          for k,v in paramDict.iteritems()])
                if func_name is not None and func_name != '':
                    if className != '':
                        if debug: print 'func in class !!'
                        if func_name != '__init__':
                            func_str = 'module.' + className + '.' + func_name
                        else:
                            func_str = 'module.' + className
                    else:
                        if debug: print 'func !!'
                        func_str = 'module.' + func_name
                        #return eval('module.' + func_name + '(**paramDict)')
                    #obj = eval('FuncWrapper(%s,paramDict)' %(func_str))
                    if debug: print 'eval(%s(**paramDict))' %(func_str)
                    obj = eval('%s(**paramDict)' %(func_str))
                else:
                    obj = eval('module.'+className+'(**paramDict)')
##        elif cnType == self.TYPE_LABEL_DICT :
##            return self.dictTagDOMReader(walker)
        
##        elif cnType == self.TYPE_LABEL_LIST :
##            return self.listTagDOMReader(walker)
        elif self.tagDOMReaders.has_key(cnType):
            if debug: print 'using reader from catalog ...'
            obj = self.tagDOMReaders[cnType](walker, self)
        else :
            raise Exception('Unknown tag type :' + cnType)

        if len(to_override) > 0:
            for k,v in to_override.iteritems():
                if debug: print 'override init of %s for obj %s with %s' \
                    %(k,str(obj),str(v.func))
                obj.override_init(k,v.func,v.params)

        if debug:
            print 'readDOMData return:'
            print obj
        return obj

    def writeDOMData(self, doc, node, obj, label, comment=None, meta=None):
        if debug: 
            print ''
            print '---------------------------'
            print 'writeDOMData ...'
            print '---------------------------'
            
        # if override is not None and override.has_key(str(label)):
        #     if debug: print 'overriding %s with %s (original was %s)' \
        #         %(label,str(override[label]),str(obj))
        #     obj = override[label]

        if self.write_callback is not None:
            obj = self.write_callback(obj,label)

        if debug: print ' creating tag with name : ', label
        if debug: print ' type of obj : ', type(obj) 
        if debug: print ' meta:', meta
        if hasattr(obj, '__class__'):
            if debug: print ' -> class:', obj.__class__
            if debug: print '---------------------------'
        newNode = doc.createElement(label)
        if comment is not None:
            com = doc.createComment(comment)
            newNode.appendChild(com)
        if meta is not None:
            newNode.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_META,meta)
##        if type(obj) == type({}):
##            self.dictDOMWriter(doc, newNode, obj)
##        elif type(obj) == type([]):
##            self.listDOMWriter(doc, newNode, obj)  
        
        if self.inspectable(obj):
            if debug: print 'inspectable ...'
            self.inspect_and_append_to_DOM_tree(doc, newNode, obj)
        elif self.objectDOMWriters.has_key(type(obj)):
            if debug: print '  calling handler ...'
            self.objectDOMWriters[type(obj)](doc, newNode, obj, self)
        elif hasattr(obj, 'appendParametersToDOMTree'):
            if debug: print '  calling interface method ...'
            obj.appendParametersToDOMTree(doc, newNode)
        else:
            error = 'no xml policy to dump item!\n'
            error += ' tag name was : %s\n' %label
            error += ' type of obj was : %s\n' %str(type(obj))
            raise Exception(error)

            
        #TODO else: Exception ...
        node.appendChild(newNode)

    def inspectable(self, obj):
        if debug:
            print 'func inspectable? ....'
            print 'obj:', obj
            print 'obj class:', obj.__class__
        #print obj
        if isinstance(obj, FuncWrapper):
            if debug: print '-> FuncWrapper!'
            obj_init = obj.func
        elif isfunction(obj):
            if debug: print '-> function!'
            obj_init = obj
        else:
            if hasattr(obj, 'im_func'): #classmethod
                if debug: print '-> im_func found'
                obj_init = obj.im_func
            elif isinstance(obj, XMLable): # assume instance
                if debug: print '-> instance obj !!!!!'
                obj_init = obj.__init__
            elif isinstance(obj, XMLable2): # assume instance
                if debug: print '-> XMLable2 instance obj !!!!!'
                obj_init = obj._init_func
            else:
                if debug: print '-> simple instance ...'
                return False

        if debug: print 'obj_init:', obj_init
        a = getargspec(obj_init)
        if debug:
            print 'argspec:', a
        if len(a[0]) == 0:
            return True
        if debug:
            print 'a.keywords is None:', a[2] is None
            print 'a.varargs is None', a[1] is None
            print 'len(a.args)', len(a[0])
            if a[3] is not None:
                print 'len(a.defaults)', len(a[3])
            print 'a:', a
        
        if a[3] is None:
            raise Exception("No default parameters for function %s"\
                                %(obj_init.__name__))
        
        # no keywords, no varargs,  all parameters have default values
        return a[2] is None and a[1] is None and \
            (len(a[0])-1 == len(a[3]))
            
        
    def inspect_and_append_to_DOM_tree(self, doc, node, obj):
        override_for_func = None
        if debug:
            print 'inspect_and_append_to_DOM_tree ...'
            print 'Test if obj is a function :'
        
        if isinstance(obj, FuncWrapper):
            if debug: print '-> FuncWrapper instance  detected'
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_MODULE,
                              obj.func.__module__)
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_FUNCTION,
                              obj.func.__name__)
            if hasattr(obj.func, 'im_self'):
                if debug: print '-> class method  detected'
                node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_CLASS, 
                                  obj.func.im_self.__name__)

            obj_init = obj.func
            override_for_func = obj.params
        elif isfunction(obj) or hasattr(obj, 'im_func'):
            
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_MODULE,
                              obj.__module__)
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_FUNCTION,
                              obj.__name__)
            
            if hasattr(obj, 'im_self'):
                if debug: print '-> class method  detected'
                node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_CLASS, 
                                  obj.im_self.__name__)
                obj_init = obj.im_func
            else:
                if debug: print '-> function  detected'    
                obj_init = obj
        elif isinstance(obj, XMLable): 
            if debug: print 'XMLable -> instance  detected'
            obj_init = obj.__init__
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_CLASS, 
                              obj.__class__.__name__)
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_MODULE,
                              obj.__class__.__module__)

        elif isinstance(obj, XMLable2): 
            if debug: print '-> XMLable2 instance  detected'
            obj_init = obj._init_func
            if isinstance(obj_init, PickleableStaticMethod):
                obj_init = obj_init.fn
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_CLASS, 
                              obj.__class__.__name__)
            node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_MODULE,
                              obj.__class__.__module__)
            if obj_init != obj.__init__:
                attr_name = TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_FUNCTION
                try:
                    node.setAttribute(attr_name, obj_init.im_func.__name__)
                except AttributeError:
                    node.setAttribute(attr_name, obj_init.__name__)
        else:
            raise Exception("Unsupported obj: %s", str(obj))

        node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_TYPE, 
                          TypedXMLHandler.TYPE_LABEL_DICT)

        p_comments = {}
        p_to_show = []
        p_meta = {}
        if issubclass(obj.__class__, XMLable) or \
                issubclass(obj.__class__, XMLable2):
            if debug: print 'instance is of XMLable(2) subclass'
            p_comments = obj.get_parameters_comments()
            p_to_show = obj.get_parameters_to_show()
            p_meta = obj.get_parameters_meta()
            if debug:
                print 'p_comments:', p_comments
                print 'p_to_show:', p_to_show
                print 'p_meta:', p_meta

        if debug: print 'Function used for init:', obj_init
        a = getargspec(obj_init)
        if debug: print 'argspec:', a
        if len(a[0]) == 0:
            return

        if len(p_to_show) == 0:
            if debug: print 'p_to_show is empty ...'
            arg_list = a[0][1:]
            arg_defaults = a[3]
        else:
            if debug: print 'use p_to_show ...', p_to_show
            arg_list = []
            arg_defaults = []
            for p in p_to_show:
                if p in a[0][1:]:
                    arg_list.append(p)
                    arg_defaults.append(a[3][a[0][1:].index(p)])


        if debug:
            print 'arg_list :'
            print arg_list
            if hasattr(obj, '_init_parameters'):
                print 'obj._init_parameters:'
                print obj._init_parameters

        if debug:
            print 'Parsing args and wrinting them to XML ...'
        for arg, default in zip(arg_list,arg_defaults):
            override = None
            if debug:
                print 'arg:', arg
                #print 'default:', default
            comment = p_comments.get(arg,None)
            meta = p_meta.get(arg,None)
            
            if override_for_func is not None:
                if debug: print '-> override arg values provided'
                default = override_for_func.get(arg,default)
            if hasattr(obj, '_parameters_value') and \
                    obj._parameters_value is not None and \
                    obj._parameters_value.has_key(arg):
                if debug: print '-> arg provided in obj._parameters_value'
                #print obj._parameters_value
                default = obj._parameters_value.get(arg)
            elif hasattr(obj, '_init_parameters') and \
                    obj._init_parameters is not None and \
                    obj._init_parameters.has_key(arg):
                if debug:  
                    print '-> init func for arg provided in ' \
                        'obj._init_parameters'
                default = obj._init_parameters.get(arg)

            if debug2:
                print 'default:', default
            self.writeDOMData(doc, node, default, arg, comment, meta)


    def packHandlers(self):
        tagDOMReaders = {}
        objectDOMWriters = {}
        
        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_DICT] = TypedXMLHandler.dictTagDOMReader
        objectDOMWriters[dict] = TypedXMLHandler.dictDOMWriter
    

        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_ODICT] = TypedXMLHandler.odictTagDOMReader
        objectDOMWriters[OrderedDict] = TypedXMLHandler.odictDOMWriter


        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_LIST] = TypedXMLHandler.listTagDOMReader
        objectDOMWriters[list] = TypedXMLHandler.listDOMWriter

        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_TUPLE] = TypedXMLHandler.tupleTagDOMReader
        objectDOMWriters[tuple] = TypedXMLHandler.tupleDOMWriter
    
        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_STRING] = TypedXMLHandler.stringTagDOMReader
        objectDOMWriters[str] = TypedXMLHandler.stringDOMWriter
        objectDOMWriters[unicode] = TypedXMLHandler.stringDOMWriter
    
        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_FLOAT] = TypedXMLHandler.floatTagDOMReader
        objectDOMWriters[float] = TypedXMLHandler.floatDOMWriter
        
        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_INT] = TypedXMLHandler.intTagDOMReader
        objectDOMWriters[int] = TypedXMLHandler.intDOMWriter

        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_BOOL] = TypedXMLHandler.boolTagDOMReader
        objectDOMWriters[bool] = TypedXMLHandler.boolDOMWriter
        
        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_ARRAY] = TypedXMLHandler.arrayTagDOMReader
        objectDOMWriters[type(array.array('i'))] = TypedXMLHandler.arrayDOMWriter

        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_NONE] = TypedXMLHandler.noneTagDOMReader
        objectDOMWriters[type(None)] = TypedXMLHandler.noneDOMWriter

        tagDOMReaders[TypedXMLHandler.TYPE_LABEL_XML_INCLUDE] = TypedXMLHandler.includeTagDOMReader
        
        return (tagDOMReaders, objectDOMWriters)

    def mountDefaultHandlers(self):
        (self.tagDOMReaders, self.objectDOMWriters) = self.packHandlers()
        
    # xml I/O for 'char' type (python string basic type):
    def noneTagDOMReader(walker, xmlHandler):
        return None
    noneTagDOMReader = staticmethod(noneTagDOMReader)


    def noneDOMWriter(doc, node, noneObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_NONE)
    noneDOMWriter = staticmethod(noneDOMWriter)

    def includeTagDOMReader(walker, xmlHandler):
        includeFn = walker.currentNode.childNodes[0].data
        assert os.path.exists(includeFn)
        f = open(includeFn, 'r')
        includeContent = string.join(f.readlines())
        f.close()
        return fromXML(includeContent)
    
    includeTagDOMReader = staticmethod(includeTagDOMReader)


    # xml I/O for 'char' type (python string basic type):
    def stringTagDOMReader(walker, xmlHandler):
        if len(walker.currentNode.childNodes) == 0:
            return ''
        else:
            return str(walker.currentNode.childNodes[0].data)
    stringTagDOMReader = staticmethod(stringTagDOMReader)


    def stringDOMWriter(doc, node, stringObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, 
                          xmlHandler.TYPE_LABEL_STRING)
        size = str(len(stringObj))
        node.setAttribute('size', size)
        if debug:
            print 'createTextNode, from string:', stringObj
            print 'size:', size
        textData = doc.createTextNode(stringObj)
        if debug: print 'textData:', textData
        node.appendChild(textData)
    stringDOMWriter = staticmethod(stringDOMWriter)

    # xml I/O for 'bool' type:
    def boolTagDOMReader(walker, xmlHandler):
        return bool(int(walker.currentNode.childNodes[0].data))
    boolTagDOMReader = staticmethod(boolTagDOMReader)

    def boolDOMWriter(doc, node, boolObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, 
                          xmlHandler.TYPE_LABEL_BOOL)
        node.setAttribute('size', '1')
        textData = doc.createTextNode(str(int(boolObj)))
        node.appendChild(textData)
    boolDOMWriter = staticmethod(boolDOMWriter)

    # xml I/O for 'float' type :
    def floatTagDOMReader(walker, xmlHandler):
        return float(walker.currentNode.childNodes[0].data)
    floatTagDOMReader = staticmethod(floatTagDOMReader)


    def floatDOMWriter(doc, node, floatObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_FLOAT)
        node.setAttribute('size', '1')
        textData = doc.createTextNode(str(floatObj))
        node.appendChild(textData)
    floatDOMWriter = staticmethod(floatDOMWriter)

    # xml I/O for 'int' type :
    def intTagDOMReader(walker, xmlHandler):
        return int(walker.currentNode.childNodes[0].data)
    intTagDOMReader = staticmethod(intTagDOMReader)


    def intDOMWriter(doc, node, intObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_INT)
        node.setAttribute('size', '1')
        textData = doc.createTextNode(str(intObj))
        node.appendChild(textData)
    intDOMWriter = staticmethod(intDOMWriter)

    # xml I/O for 'array' type :
    def arrayTagDOMReader(walker, xmlHandler):
        return int(walker.currentNode.childNodes[0].data)
    arrayTagDOMReader = staticmethod(arrayTagDOMReader)


    def arrayDOMWriter(doc, node, arrayObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_ARRAY)
        node.setAttribute('size', str(len(array)))
        node.setAttribute('subtype', arrayObj.typecode)
        textData = doc.createTextNode(map(str,arrayObj)) ## check map(...) -> list de string & createTextNode(arg=string) ??
        node.appendChild(textData)
    arrayDOMWriter = staticmethod(arrayDOMWriter)

    # xml I/O for 'list' type:
    def listTagDOMReader(walker, xmlHandler):
        currentListNode = walker.currentNode
        #result = range(int(currentListNode.getAttribute('size')))
        result = dict()
        while 1:
            nn = walker.nextNode()
            if nn==None:
                xmlHandler.parseTerminated = True
                break
            
            if not nn.parentNode.isSameNode(currentListNode) :
                walker.previousNode()
                break
            result[int(nn.tagName[1:])] = xmlHandler.readDOMData(walker)
        return [result[R] for R in sorted(result.keys())]
    listTagDOMReader = staticmethod(listTagDOMReader)

    def listDOMWriter(doc, node, listObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_LIST)
        node.setAttribute( 'size', str(len(listObj)) )
        i = 0
        for elem in listObj :
            xmlHandler.writeDOMData(doc, node, elem, 'i'+str(i))
            i += 1
    listDOMWriter = staticmethod(listDOMWriter)



    # xml I/O for 'list' type:
    def tupleTagDOMReader(walker, xmlHandler):
        currentListNode = walker.currentNode
        result = range(int(currentListNode.getAttribute('size')))
        while 1:
            nn = walker.nextNode()
            if nn==None:
                xmlHandler.parseTerminated = True
                break
            
            if not nn.parentNode.isSameNode(currentListNode) :
                walker.previousNode()
                break
            result[int(nn.tagName[1:])] = xmlHandler.readDOMData(walker)
        return tuple(result)
    tupleTagDOMReader = staticmethod(tupleTagDOMReader)

    def tupleDOMWriter(doc, node, tupleObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, xmlHandler.TYPE_LABEL_TUPLE)
        node.setAttribute( 'size', str(len(tupleObj)) )
        i = 0
        for elem in tupleObj :
            xmlHandler.writeDOMData(doc, node, elem, 'i'+str(i))
            i += 1
    tupleDOMWriter = staticmethod(tupleDOMWriter)



    # xml I/O for 'dict' type:
    def dictTagDOMReader(walker, xmlHandler, init_class=None):
        if debug: 
            print 'dictTagDOMReader ...'
        currentDictNode = walker.currentNode
        if debug: 
            print ' currentNode : ',currentDictNode.tagName
        if init_class is None:
            result = {}
        else:
            result = init_class()
        
        if debug:
            print 'dict init:', result.__class__

        while 1:
            nn = walker.nextNode()
            if nn==None:
                xmlHandler.parseTerminated = True
                break
            if not nn.parentNode.isSameNode(currentDictNode) :
                walker.previousNode()
                break
            nnType = nn.getAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE)
            if nnType == xmlHandler.TYPE_LABEL_KEY_VAL_PAIR :
                walker.nextNode()
                key = xmlHandler.readDOMData(walker)
                walker.nextNode()
                result[key] = xmlHandler.readDOMData(walker)
            else :
                if debug: print 'nn.tagName:', nn.tagName
                result[nn.tagName] = xmlHandler.readDOMData(walker)
        
        if debug:
            print ' returning ...:', result
        return result
    dictTagDOMReader = staticmethod(dictTagDOMReader)

    def dictDOMWriter(doc, node, dictObj, xmlHandler, atype=None):

        if atype is None:
            atype = xmlHandler.TYPE_LABEL_DICT

        atypel = xmlHandler.ATTRIBUTE_LABEL_TYPE
        node.setAttribute(atypel, atype)
        for key,val in dictObj.items() :
            # if the key type is string, use the standard xml
            # association : string label <-> string data
            # if key starts with a digit -> tag name will not be well-formed
            if (type(key)==type('') or type(key)==unicode)\
                   and not key[0].isdigit(): 
                xmlHandler.writeDOMData(doc, node, val, key)
            else : # if the key type is not a string,
                   # build a key<->value pair xml structure
                keyValElem = doc.createElement("anonym")
                atypel = xmlHandler.ATTRIBUTE_LABEL_TYPE
                atype = xmlHandler.TYPE_LABEL_KEY_VAL_PAIR
                keyValElem.setAttribute(atypel, atype)
                xmlHandler.writeDOMData(doc, keyValElem, key, "key")
                xmlHandler.writeDOMData(doc, keyValElem, val, "value")
                node.appendChild(keyValElem)           
    dictDOMWriter = staticmethod(dictDOMWriter)

    def odictTagDOMReader(walker, xmlHandler):
        return xmlHandler.dictTagDOMReader(walker, xmlHandler, OrderedDict)
    odictTagDOMReader = staticmethod(odictTagDOMReader)


    def odictDOMWriter(doc, node, dictObj, xmlHandler):
        return xmlHandler.dictDOMWriter(doc, node, dictObj, xmlHandler,
                                        xmlHandler.TYPE_LABEL_ODICT)
    odictDOMWriter = staticmethod(odictDOMWriter)

def fromXML(s, handler = TypedXMLHandler()):
    return handler.parseXMLString(s)


# def loadXML(fxml):
#     f = open(fxml, 'r')
#     sXml = string.join(f.readlines())
#     f.close()
#     return fromXML(sXml)

def toXML(o, handler = TypedXMLHandler(), objName="anonymObject", pretty=False):
    if isinstance(o, XMLParamDrivenClass) or hasattr(o, 'xmlHandler'): 
        handler = o.xmlHandler
    return handler.buildXMLString(o, objName, pretty=pretty)



def read_xml(fn):
    f = open(fn)
    content = f.read()
    f.close()
    from xmlnumpy import NumpyXMLHandler
    handler = NumpyXMLHandler()
    return handler.parseXMLString(content)

def write_xml(obj, fn):    
    fOut = open(fn, 'w')
    from xmlnumpy import NumpyXMLHandler
    fOut.write(toXML(obj, handler=NumpyXMLHandler()))
    fOut.close()

class XMLParamDrivenClassInitException :

    def __init__(self):
        pass


class XMLParamDrivenClass:
    """
    Base \"abstract\" class to handle parameters with clear specification and default values.
    Recursive aggregation is availaible to handle aggregated variables which also require
    parameter specifications.
    """

    # Parameters labels should be clearly defined in derived classes, example :
    # P_PARAMETER_1 = 'labelParameter1'
    # P_PARAMETER_1 would then be the inner designation and 'labelParameter1' would be used in the outter XML specification.

    # Default parameter values :
    defaultParameters = {}

    # Comments describing parameters:
    parametersComments = {}

    # parameters meta type:
    parametersMeta = {}

    def __init__(self, parameters=None, xmlHandler=TypedXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        """
        Create a new XMLParamDrivenClass

        """ 
        self.updateParameters(parameters)
        self.xmlHandler = xmlHandler
        
        if xmlLabel is None :
            # Default tag label used in xml
            self.xmlLabel = self.__class__.__name__+"_parameters" 
        else :
            self.xmlLabel = xmlLabel 

        if xmlComment is None:
            #self.xmlComment = 'Set of parameters defining the python class : ' \
            #                  +self.__module__ +'.'+ self.__class__.__name__
            self.xmlComment = None
        else :
            self.xmlComment = xmlComment

    def fetchDefaultParameters(self):
        self.parameters = copyModule.deepcopy(self.defaultParameters)
            
    def updateParameters(self, newp):
        self.fetchDefaultParameters()

        # Extracting new parameters' values from input :
        if newp != None :
            for key in newp.keys():
                if self.parameters.has_key(key):
                    self.parameters[key] = newp[key]
                else :
                    print 'Warning - in creation of a \"'+self.__class__.__name__ +'\" instance :'
                    print '  unknown parameter designation : ', key

    def appendParametersToDOMTree(self, doc, node):
#         if tagName==None:
#             tag = self.xmlLabel
#         else :
#             tag = tagName
#         elem = doc.createElement(tag)
        node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_CLASS, 
                          self.__class__.__name__)
        node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_PYTHON_MODULE,
                          self.__class__.__module__)
        node.setAttribute(self.xmlHandler.ATTRIBUTE_LABEL_PYTHON_CLASS_INIT_MODE, 
                          "XMLParamDrivenClass")
        
        node.setAttribute(TypedXMLHandler.ATTRIBUTE_LABEL_TYPE, 
                          TypedXMLHandler.TYPE_LABEL_DICT)

        if self.xmlComment is not None:
            com = doc.createComment(self.xmlComment)
            node.appendChild(com)

        if hasattr(self, 'parametersToShow'):
            paramIter = self.parametersToShow
        else:
            paramIter = self.parameters.keys()

        for paramLabel in paramIter:
            paramVal = self.parameters[paramLabel]
            if debug: print 'treating parameter : ',paramLabel
            if debug: print 'val : ',paramVal
            #print 'self.xmlHandler :', self.xmlHandler
            comment = self.parametersComments.get(paramLabel,None)
            meta = self.parametersMeta.get(paramLabel)
            self.xmlHandler.writeDOMData(doc, node, paramVal, paramLabel,
                                         comment=comment, meta=meta)
        #node.appendChild(elem)
        
    def parametersToXml(self, tagName=None, pretty=False):
        (doc,root) = self.xmlHandler.createDocument()
        if tagName==None:
            tag = self.xmlLabel
        else :
            tag = tagName
        node = doc.createElement(tag)
        self.appendParametersToDOMTree(doc, node)
        root.appendChild(node)
        if not pretty:
            return doc.toxml()
        else:
            return doc.toprettyxml()


def match_init_args(c, argsDict):

    a = getargspec(c.__init__)
    to_pop = []
    for argname in argsDict.iterkeys():
        if argname not in a[0]:
            print 'Warning: %s unused for class %s' \
                %(argname, c.__class__)
            to_pop.append(argname)

    for tp in to_pop:
        argsDict.pop(tp)
