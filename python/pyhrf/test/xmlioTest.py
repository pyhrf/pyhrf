

import os
import os.path as op

import unittest
import numpy as np

from pyhrf import xmlio
import pyhrf

class BaseTest(unittest.TestCase):
    
    def testSimple(self):

        dictObj = {'clef1' : 'val1', 'clef2': 2}
        sXml = xmlio.to_xml(dictObj)
        dictObj2 = xmlio.fromXML(sXml)
        assert dictObj2 == dictObj

        obj = ['ggg',5,(677,56),'ff']
        sXml = xmlio.to_xml(obj)
        obj2 = xmlio.fromXML(sXml)
        assert obj2 == obj
        

    def testNumpy(self):

        obj = \
            { 'array1' : np.random.randn(10),
              'val1' : np.int(5),
              'array2D' : np.ones((4,4)),
              }
        sXml = xmlio.to_xml(obj, handler=xmlio.xmlnumpy.NumpyXMLHandler())
        obj2 = xmlio.fromXML(sXml)
        assert obj2.keys() == obj.keys()
        assert np.allclose(obj2["array1"],obj["array1"])
        
    def test_ordered_dict(self):
        
        try:
            from collections import OrderedDict
        except ImportError:
            from pyhrf.tools.backports import OrderedDict

        d = OrderedDict([ ('a',1), ('b',2) ])
        sXml = xmlio.to_xml(d)
        #print 'sXml:'
        #print sXml
        d2 = xmlio.fromXML(sXml)

        self.assertEqual(d, d2)
        #for i1, i2 in zip(d.items(), d2.items):
            

# Define a class whose parameters are declared and have default values
class Dummy(xmlio.XMLParamDrivenClass):
    """
    Dummy class to illustrate the xml parametrisation system
    """
    defaultParameters = {
        'param1' : 'paramVal1',
        }

    def __init__(self, parameters=None, xmlHandler=xmlio.TypedXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        # |-> Put every parameters in a dictionary, make the synopsis of __init__
        # compatible with the __init__ of a XMLParamDrivenClass object

        # Call parent constructor (will mainly update self.parameters
        # from parameters):
        xmlio.XMLParamDrivenClass.__init__(self, parameters)

        # Retrieve parameter values:
        self.p1 = self.parameters['param1']

class DummyNumpy(xmlio.XMLParamDrivenClass):
    """
    Dummy class to illustrate the xml parametrisation system, with numpy support.
    """

    defaultParameters = {
        'param1' : 'paramVal1',
        'numArray' : np.array([56,56]),}# Beware : value associated to 'numArray'
                                        # will be shared by all instances
                                        # Eventually make a copy in __init__

    def __init__(self, parameters=None,
                 xmlHandler=xmlio.xmlnumpy.NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):
        # Call parent constructor (will mainly update self.parameters
        # from parameters):
        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)

        # Retrieve parameter values:
        self.p1 = self.parameters['param1']
        self.p2 = self.parameters['numArray'].copy() 


class DummyNested(xmlio.XMLParamDrivenClass):
    """
    Dummy class to illustrate the xml parametrisation system, with nested
    attributes also inheriting from XMLParamDrivenClass (cascade parametrisation)
    """

    defaultParameters = { \
        'objP1' : Dummy(),
        'objP2' : DummyNumpy(),
        }

    def __init__(self, parameters=None,
                 xmlHandler=xmlio.xmlnumpy.NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):

        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)

        # Retrieve parameter values:
        self.p1 = self.parameters['objP1']
        self.p2 = self.parameters['objP2']


    
class XMLParamDrivenClassTest(unittest.TestCase):


    def testSimple(self):
##        print 'Simple ..'
        obj = Dummy()
        sXml = xmlio.to_xml(obj)
##        print 'xml:'
##        print sXml
        xmlio.fromXML(sXml)
##        print 'parameters:'
##        print obj.parameters

    def testNumpy(self):
##        print 'Numpy ..'
        obj = DummyNumpy()
        sXml = xmlio.to_xml(obj)
##        print 'Xml:'
##        print sXml
        xmlio.fromXML(sXml)
##        print 'parameters:'
##        print obj.parameters

    def testNested(self):

##        print 'Nested objects ..'
        obj = DummyNested()
        sXml = xmlio.to_xml(obj)
##        print 'Xml:'
##        print sXml
        xmlio.fromXML(sXml)
##        print 'parameters:'
##        print obj.parameters
        

import inspect
class TopClass(xmlio.XMLable):
    def __init__(self, p_top='1'):
        xmlio.XMLable.__init__(self)
    
class ChildClass(xmlio.XMLable):
    def __init__(self, p_child=2):
        xmlio.XMLable.__init__(self)
        


class D(xmlio.XMLable):
    def __init__(self,p=2):
        XMLable.__init__(self)

class T(xmlio.XMLable):
    
    def __init__(self, param_a=1):
        xmlio.XMLable.__init__(self)
        self.pa = param_a

    def from_param_c(self, param_c=np.array([56])):
        return T(param_c)
    from_param_c = classmethod(from_param_c)


class A(xmlio.XMLable):
    def __init__(self, obj_t=np.array([5])):
        xmlio.XMLable.__init__(self)
        self.obj = obj_t



def create_t():
    print '#### create_t ..'
    return T(param_a=49)

def hack_a(a, label):
    # print 'hack_a ..'
    # print 'a:', a
    # print 'label:', label
    if isinstance(a, A):
        return A('blah')
    else:
        return a

def hack_treatment_data_path(o, label):
    # print 'hack_treatment_data_path ..'
    # print 'o:', o
    # print 'label:', label
    if label in ['mask_file', 'bold_file']:
        # print ''
        # print '!!!!HACK!!!!'
        # print ''
        return op.basename(o)
    else:
        return o

class XMLableTest(unittest.TestCase):


    def testDynamicParamsSingleClass(self):
        
        o = TopClass()
        #print 'init_parameters:'
        #print o._init_parameters
        assert len(o._init_parameters) == 1
        assert o._init_parameters['p_top'] == '1'

    def testDynamicParamsHierachic(self):
        
        o = ChildClass()
        #print 'init_parameters:'
        #print o._init_parameters
        assert len(o._init_parameters) == 1
        assert o._init_parameters['p_child'] == 2
        

    def test_value_override(self):
            a = A()
            a.override_init('obj_t', T.from_param_c)
            # print 'after override ...'
            # print a._init_parameters
            xml = xmlio.to_xml(a,handler=xmlio.xmlnumpy.NumpyXMLHandler(),
                              pretty=True)
            # print xml
            new_a = xmlio.fromXML(xml)
            # print ''
            # print 'new_a:'
            # print new_a


    def test_write_callback(self):
        
        a = A()
        handler = xmlio.xmlnumpy.NumpyXMLHandler(write_callback=hack_a)
        xml = xmlio.to_xml(a,handler=handler,pretty=True)
        # print xml


    # def test_treatment_path_hack(self):
    #     from pyhrf.ui.treatment import FMRITreatment
    #     from pyhrf.core import FmriData
    #     t = FMRITreatment()
    #     t.override_init('fmri_data', FmriData.from_vol_ui)
    #     handler = xmlio.xmlnumpy.NumpyXMLHandler(write_callback=hack_treatment_data_path)
    #     xml = xmlio.to_xml(t,handler=handler,pretty=True)
    #     print xml


        



class B(xmlio.XMLable2):
    def __init__(self, obj_t=np.array([5])):
        xmlio.XMLable2.__init__(self)
        self.obj = obj_t

    @classmethod
    def from_stuff(self, a=2, b=5):
        o = B(np.array([a,b]))
        o.set_init(init_func=B.from_stuff, a=a, b=b)
        return o


class XMLable2Test(unittest.TestCase):

    def test_init(self):
        b = B()
        # print 'b.obj:', b.obj

    def test_classmethod_init(self):
        b2 = B.from_stuff()
        #print 'b2.obj:', b2.obj

    def test_xml_from_init(self):
        b = B()
        #print 'b.obj:', b.obj
        xml = xmlio.to_xml(b,handler=xmlio.xmlnumpy.NumpyXMLHandler(),
                          pretty=True)
        # print 'xml:'
        # print xml
        
        b2 = xmlio.fromXML(xml)
        # print 'b2.obj:', b2.obj


    def test_xml_from_classmethod_init(self):
        b = B.from_stuff(a=4,b=3)
        # print 'b.obj:', b.obj
        np.testing.assert_equal(b.obj, [4,3])
        xml = xmlio.to_xml(b,handler=xmlio.xmlnumpy.NumpyXMLHandler(),
                          pretty=True)
        # print 'xml:'
        # print xml
        
        b2 = xmlio.fromXML(xml)
        # print 'b2.obj:', b2.obj
        np.testing.assert_equal(b2.obj, [4,3])


    def test_JDEMCMCAnalyzerXML(self):
        
        from pyhrf.ui.jde import JDEMCMCAnalyser
        from pyhrf.jde.models import BOLDGibbsSampler
        a = JDEMCMCAnalyser(sampler=BOLDGibbsSampler({'nbIterations':42}))
        # print 'a -- nbIterations:', a.sampler.nbIterations
        axml = xmlio.to_xml(a, handler=xmlio.xmlnumpy.NumpyXMLHandler(), 
                           pretty=True)
        # print 'axml:'
        # print axml

        a2 = xmlio.fromXML(axml)
        # print 'a2 -- nbIterations:', a2.sampler.nbIterations
        self.assertEqual(a2.sampler.nbIterations, 42)


    def test_TreatmentXML(self):

        #pyhrf.verbose.set_verbosity(6)
        from pyhrf.ui.jde import JDEMCMCAnalyser
        from pyhrf.jde.models import BOLDGibbsSampler
        from pyhrf.ui.treatment import FMRITreatment
        a = JDEMCMCAnalyser(sampler=BOLDGibbsSampler({'nbIterations':42}))
        t = FMRITreatment(analyser=a)
        # print 't -- nbIterations:', t.analyser.sampler.nbIterations
        txml = xmlio.to_xml(t, handler=xmlio.xmlnumpy.NumpyXMLHandler(), 
                           pretty=True)
        # print 'txml:'
        # print txml

        t2 = xmlio.fromXML(txml)
        # print 't2 -- nbIterations:', t2.analyser.sampler.nbIterations
        self.assertEqual(t2.analyser.sampler.nbIterations, 42)

        #pyhrf.verbose.set_verbosity(0)
        
