# -*- coding: utf-8 -*-

import unittest
import os
import os.path as op
import shutil
import logging

from pprint import pformat

import numpy as np
import numpy.testing as npt

import pyhrf

from pyhrf.xmlio import XmlInitable, to_xml, from_xml
from pyhrf import xmlio


logger = logging.getLogger(__name__)


class A(XmlInitable):

    def __init__(self, p=1, c='a'):
        XmlInitable.__init__(self)
        self.param1 = p
        self.param2 = c

    def __eq__(self, o):
        return isinstance(o, A) and o.param1 == self.param1 and \
            o.param2 == self.param2


class TestXML(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = pyhrf.get_tmp_path()  # './'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_simple_bijection(self):
        a = A(c='e')
        a_xml = to_xml(a)
        self.assertEqual(a, from_xml(a_xml))


class BaseTest(unittest.TestCase):

    def test_basic_types(self):

        dictObj = {'clef1': 'val1', 'clef2': 2}
        sXml = xmlio.to_xml(dictObj)
        dictObj2 = xmlio.from_xml(sXml)
        assert dictObj2 == dictObj

        obj = ['ggg', 5, (677, 56), 4.5, 'ff']
        sXml = xmlio.to_xml(obj)
        obj2 = xmlio.from_xml(sXml)
        assert obj2 == obj

    def _test_xml_bijection(self, o):
        xml = to_xml(o)
        logger.debug('xml from %s:', str(o))
        logger.debug(pformat(xml))
        o2 = from_xml(xml)
        self.assertEqual(o, o2)

    def test_list_of_str(self):
        self._test_xml_bijection(['a', 'bb', 'ccc', ''])

    def test_bool(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        self._test_xml_bijection(False)

    def test_list_of_int(self):
        self._test_xml_bijection([-1, 134, 1])

    def test_list_of_misc(self):
        self._test_xml_bijection([-1, None, 'aa', True, False])

    def test_tuple_of_misc(self):
        self._test_xml_bijection((-1, None, 'aa', True, False))

    def testNumpy(self):

        obj = \
            {'array1': np.random.randn(10),
             'val1': np.int(5),
             'array2D': np.ones((4, 4)),
             }
        sXml = xmlio.to_xml(obj)
        obj2 = xmlio.from_xml(sXml)
        assert obj2.keys() == obj.keys()
        assert np.allclose(obj2["array1"], obj["array1"])

    def test_ordered_dict(self):

        try:
            from collections import OrderedDict
        except ImportError:
            from pyhrf.tools.backports import OrderedDict

        d = OrderedDict([('a', 1), ('b', 2)])
        sXml = xmlio.to_xml(d)
        # print 'sXml:'
        # print sXml
        d2 = xmlio.from_xml(sXml)

        self.assertEqual(d, d2)
        # for i1, i2 in zip(d.items(), d2.items):


class TopClass(xmlio.XmlInitable):
    def __init__(self, p_top='1'):
        xmlio.XmlInitable.__init__(self)


class ChildClass(xmlio.XmlInitable):
    def __init__(self, p_child=2):
        xmlio.XmlInitable.__init__(self)


class D(xmlio.XmlInitable):
    def __init__(self, p=2):
        xmlio.XmlInitable.__init__(self)


class T(xmlio.XmlInitable):

    def __init__(self, param_a=1):
        xmlio.XmlInitable.__init__(self)
        self.pa = param_a

    def from_param_c(self, param_c=np.array([56])):
        return T(param_c)
    from_param_c = classmethod(from_param_c)


def create_t():
    return T(param_a=49)


class XMLableTest(unittest.TestCase):

    def testDynamicParamsSingleClass(self):
        o = TopClass()
        self.assertEqual(len(o._init_parameters), 1)
        self.assertEqual(o._init_parameters['p_top'], '1')

    def testDynamicParamsHierachic(self):

        o = ChildClass()
        self.assertEqual(len(o._init_parameters), 1)
        self.assertEqual(o._init_parameters['p_child'], 2)

    def test_set_init_param(self):
        a = A()
        a.set_init_param('c', 'new_c')
        xml = xmlio.to_xml(a, pretty=True)
        new_a = xmlio.from_xml(xml)
        self.assertEqual(new_a.param2, 'new_c')


class B(xmlio.Initable):
    def __init__(self, obj_t=np.array([5])):
        xmlio.Initable.__init__(self)
        self.obj = obj_t

    @classmethod
    def from_stuff(self, a=2, b=5):
        o = B(np.array([a, b]))
        o.set_init(B.from_stuff, a=a, b=b)
        return o

    def __eq__(self, b):
        return (self.obj == b.obj).all()


class C(xmlio.Initable):
    def __init__(self):
        xmlio.Initable.__init__(self)

    def __eq__(self, o):
        return isinstance(o, C)


class InitableTest(unittest.TestCase):

    def test_init(self):
        b = B()
        npt.assert_array_equal(b.obj, np.array([5]))

    def test_pickle_classmethod(self):
        import pickle
        f = pyhrf.core.FmriData.from_surf_files
        s = pickle.dumps(f)
        fs = pickle.loads(s)
        self.assertEqual(f, fs)

    def test_classmethod_init(self):
        b2 = B.from_stuff()
        npt.assert_array_equal(b2.obj, np.array([2, 5]))
        # print 'b2.obj:', b2.obj

    def test_xml_from_init(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        b = B()
        xml = b.to_ui_node('bobj').to_xml(pretty=True)
        import re
        pat = '.*<bobj pickled_init_obj="[^>]*test_xml[^>]*B[^>]*"[^>]*>'\
            '.*<obj_t[^>]*pickled_type="[^>]*ndarray[^>]*"[^>]*>5'\
            '</obj_t>.*</bobj>.*'
        if re.match(pat, xml, re.DOTALL) is None:
            raise Exception('Wrong XML :\n%s' % xml)

    def test_xml_from_classmethod_init(self):
        b = B.from_stuff(a=4, b=3)
        np.testing.assert_equal(b.obj, [4, 3])
        xml = b.to_ui_node('bobj').to_xml(pretty=True)
        pat = '.*<bobj pickled_init_obj="[^>]*from_stuff[^>]*"[^>]*>'\
            '.*<a[^>]*pickled_type="[^>]*int[^>]*"[^>]*>4</a>'\
            '.*<b[^>]*pickled_type="[^>]*int[^>]*"[^>]*>3</b>'\
            '.*</bobj>.*'
        import re
        if re.match(pat, xml, re.DOTALL) is None:
            raise Exception('Wrong XML :\n%s' % xml)

    def test_bijection_from_init(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        b = B()
        xml = b.to_ui_node('bobj').to_xml(pretty=True)
        b2 = xmlio.from_xml(xml)
        self.assertEqual(b, b2)

    def test_bijection_from_init_no_arg(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        o = C()
        xml = o.to_ui_node('bobj').to_xml(pretty=True)
        o2 = xmlio.from_xml(xml)
        self.assertEqual(o, o2)

    def test_bijection_from_classmethod_init(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        b = B.from_stuff(a=4, b=3)
        xml = b.to_ui_node('bobj').to_xml(pretty=True)
        b2 = xmlio.from_xml(xml)
        self.assertEqual(b, b2)

    def test_JDEMCMCAnalyzer_Uinode_bijection(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        from pyhrf.ui.jde import JDEMCMCAnalyser
        from pyhrf.jde.models import BOLDGibbsSampler
        a = JDEMCMCAnalyser(sampler=BOLDGibbsSampler(nb_iterations=42))
        a_ui = xmlio.UiNode.from_py_object('analyse', a)
        it_node = a_ui.get_children()[0].get_children()[0].get_children()[0]
        self.assertEqual(it_node.label(), str(42))

        a2 = xmlio.Initable.from_ui_node(a_ui)
        self.assertEqual(a2.sampler.nbIterations, 42)

    def test_JDEMCMCAnalyzerXML(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        from pyhrf.ui.jde import JDEMCMCAnalyser
        from pyhrf.jde.models import BOLDGibbsSampler
        a = JDEMCMCAnalyser(sampler=BOLDGibbsSampler(nb_iterations=42))
        axml = xmlio.to_xml(a, pretty=True)
        a2 = xmlio.from_xml(axml)
        self.assertEqual(a2.sampler.nbIterations, 42)

    def test_TreatmentXML(self):
        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)
        from pyhrf.ui.jde import JDEMCMCAnalyser
        from pyhrf.jde.models import BOLDGibbsSampler
        from pyhrf.ui.treatment import FMRITreatment
        a = JDEMCMCAnalyser(sampler=BOLDGibbsSampler(nb_iterations=42))
        t = FMRITreatment(analyser=a)
        txml = xmlio.to_xml(t, pretty=True)
        t2 = xmlio.from_xml(txml)
        self.assertEqual(t2.analyser.sampler.nbIterations, 42)
