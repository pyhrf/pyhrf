

import numpy
from xmlbase import *

#TODO : handle numpy basic type ie numpy.float64 numpy.int32 ...

debug = False

class NumpyXMLHandler(TypedXMLHandler):

    NUMPY_ARRAY_TAG_NAME = 'numpy.ndarray'
    NUMPY_INT16_TAG_NAME = 'numpy.int16'
    NUMPY_INT32_TAG_NAME = 'numpy.int32'
    
    def __init__(self, write_callback=None):
        TypedXMLHandler.__init__(self, write_callback)

        self.tagDOMReaders[self.NUMPY_ARRAY_TAG_NAME] = self.arrayTagDOMReader
        self.objectDOMWriters[numpy.ndarray] = self.arrayDOMWriter

    def packHandlers(self):
        (tagDOMReaders, objectDOMWriters) = TypedXMLHandler.packHandlers(self)
        for n,t in numpy.typeDict.items():
            tagDOMReaders['numpy.'+str(n)] = self.numpyObjectTagDOMReader
            objectDOMWriters[t] = self.numpyObjectTagDOMWriter
            
        tagDOMReaders[self.NUMPY_ARRAY_TAG_NAME] = self.arrayTagDOMReader
        objectDOMWriters[numpy.ndarray] = self.arrayDOMWriter
        #tagDOMReaders[self.NUMPY_INT16_TAG_NAME] = self.int16TagDOMReader
        #objectDOMWriters[numpy.int16] = self.int16DOMWriter
        #tagDOMReaders[self.NUMPY_INT16_TAG_NAME] = self.int16TagDOMReader
        #objectDOMWriters[numpy.int16] = self.int16DOMWriter
        
        return (tagDOMReaders, objectDOMWriters)

    def numpyObjectTagDOMReader(walker, xmlHandler):
        wc = walker.currentNode
        nodeType = wc.getAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE)
        if debug: print 'numpyObjectTagDOMReader: nodeType:', nodeType
        typeName = nodeType.split('.')[1]
        if typeName.isdigit(): typeName = int(typeName)
        return numpy.typeDict[typeName](walker.currentNode.childNodes[0].data)
    numpyObjectTagDOMReader = staticmethod(numpyObjectTagDOMReader)
    
    def numpyObjectTagDOMWriter(doc, node, obj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE,
                          'numpy.'+str(obj.dtype.name))
        node.appendChild(doc.createTextNode(str(obj)))
    numpyObjectTagDOMWriter = staticmethod(numpyObjectTagDOMWriter)
    
    
    def arrayTagDOMReader(walker, xmlHandler):
        #TODO : handle better object of dtype other than string72, float
        # or unicode
        size = walker.currentNode.getAttribute('size')
        dtype = walker.currentNode.getAttribute('dtype')
        if dtype[:3] == 'str':
            dtype = 'str'
        if 'unicode' in dtype:
           dtype = 'unicode' 
        if debug:
            print 'numpy array reader :'
            print ' dtype = ', dtype
        if len(walker.currentNode.childNodes) > 0 and \
               walker.currentNode.childNodes[0] is not  None:
            arrayData = walker.currentNode.childNodes[0].data.split()
            if dtype != 'str' and dtype != 'unicode':
                arrayData = map(float, arrayData)
        else:
            arrayData = [] 
        if debug:
            print ' arrayData = ', arrayData
        data = numpy.array(arrayData, dtype=numpy.typeDict[dtype])
        if size != '':
            data.shape = tuple(map(int,size.split()))
        if debug:
            print ' data : ', data.dtype
            print data
        return data
    arrayTagDOMReader = staticmethod(arrayTagDOMReader)

    def arrayDOMWriter(doc, node, arrayObj, xmlHandler):
        
        #print 'arrayDOMWriter ...'
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE, 
                          xmlHandler.NUMPY_ARRAY_TAG_NAME)
        if len(arrayObj.shape) > 1:
            size = string.join(map(str,arrayObj.shape))
            node.setAttribute('size', size)

        dtype = arrayObj.dtype.name
        node.setAttribute('dtype', dtype)

        arrayData = string.join(map(str,arrayObj.ravel()))
        #print 'arrayData :'
        #print arrayData
        arrayTextData = doc.createTextNode(arrayData)
        node.appendChild(arrayTextData)
    arrayDOMWriter = staticmethod(arrayDOMWriter)

    # OBSOLETE ...
    def int16TagDOMReader(walker, xmlHandler):
        return numpy.int16(int(walker.currentNode.childNodes[0].data))
    int16TagDOMReader = staticmethod(int16TagDOMReader)

    def int16DOMWriter(doc, node, intObj, xmlHandler):
        node.setAttribute(xmlHandler.ATTRIBUTE_LABEL_TYPE,
                          xmlHandler.NUMPY_INT16_TAG_NAME)

        node.appendChild(doc.createTextNode(str(intObj)))
    int16DOMWriter = staticmethod(int16DOMWriter)
