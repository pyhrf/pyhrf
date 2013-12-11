

import numpy
from xmlbase import *


class MatlabXMLHandler(TypedXMLHandler):

    TYPE_LABEL_CELL = 'cell'
    TYPE_LABEL_DOUBLE = 'double'
    
    def __init__(self):

        TypedXMLHandler.__init__(self)

    def packHandlers(self):
        (tagDOMReaders, objectDOMWriters) = TypedXMLHandler.packHandlers(self)
        
        tagDOMReaders[self.TYPE_LABEL_CELL] = self.cellTagDOMReader
        objectDOMWriters[self.TYPE_LABEL_CELL] = self.cellDOMWriter

        tagDOMReaders[self.TYPE_LABEL_DOUBLE] = self.doubleTagDOMReader
        objectDOMWriters[self.TYPE_LABEL_DOUBLE] = self.doubleDOMWriter

    def doubleTagDOMReader(walker, xmlHandler):
        size = walker.currentNode.getAttribute('size')
        dtype = 'float' #walker.currentNode.getAttribute('dtype')
        print 'numpy array reader :'
        print ' dtype = ', dtype
        dims = tuple(map(int,size.split()))
        if 0 not in dims :
            arrayData = map(float,walker.currentNode.childNodes[0].data.split())
            print ' arrayData = ', arrayData
            data = numpy.array(arrayData, dtype=numpy.typeDict[dtype])
            data.shape = tuple(reversed(dims))
            data = data.transpose()
        else :
            data = numpy.array([], dtype=numpy.typeDict[dtype])
        return data
    doubleTagDOMReader = staticmethod(doubleTagDOMReader)


    def doubleDOMWriter(doc, node, arrayObj, xmlHandler):
        print 'xml writer for matlab double type not written yet !'
        return None
    doubleDOMWriter = staticmethod(doubleDOMWriter)

    
    def cellTagDOMReader(walker, xmlHandler):
        currentListNode = walker.currentNode
        size = int(currentListNode.getAttribute('size').split()[1])
        result = range(size)
        idx = 0
        while 1:
            nn = walker.nextNode()
            if nn==None:
                xmlHandler.parseTerminated = True
                break
            
            if not nn.parentNode.isSameNode(currentListNode) :
                walker.previousNode()
                break
            result[idx] = xmlHandler.readDOMData(walker)
            idx += 1
        return result
    cellTagDOMReader = staticmethod(cellTagDOMReader)

    
        
    def cellDOMWriter(doc, node, arrayObj, xmlHandler):
        print 'xml writer for matlab cell type not written yet !'
        return None
    cellDOMWriter = staticmethod(cellDOMWriter)
