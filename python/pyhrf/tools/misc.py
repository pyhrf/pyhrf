# -*- coding: utf-8 -*-

import os
import os.path as op
import sys
import string
import cPickle
import hashlib
import gzip
import datetime
import inspect
import re
import logging

from itertools import izip
from time import time
from collections import defaultdict

import numpy as np
import scipy.linalg
import scipy.signal

import pyhrf

try:
    from itertools import product as iproduct
except ImportError:
    def iproduct(*args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
            for prod in result:
                yield tuple(prod)


logger = logging.getLogger(__name__)


class PickleableStaticMethod(object):
    def __init__(self, fn, cls=None):
        self.cls = cls
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        if self.cls is None:
            return self.fn(*args, **kwargs)
        else:
            return self.fn(self.cls, *args, **kwargs)

    def __get__(self, obj, cls):
        return PickleableStaticMethod(self.fn, cls)

    def __getstate__(self):
        return (self.cls, self.fn.__name__)

    def __setstate__(self, state):
        self.cls, name = state
        self.fn = getattr(self.cls, name).fn


def is_importable(module_name, func_name=None):
    """ Return True if given *module_name* (str) is importable """
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        if func_name is None:
            return True
        else:
            return func_name in dir(module_name)


def report_arrays_in_obj(o):
    for a in dir(o):
        attr = eval('o.%s' % a)
        if isinstance(attr, np.ndarray):
            print a, '->', attr.shape, attr.dtype, '[id=%s]' % str(id(attr))
        elif (isinstance(attr, list) or isinstance(attr, tuple)) and len(attr) > 0 and \
                isinstance(attr[0], np.ndarray):
            print a, '-> list of %d arrays (first array: %s, %s), [id=%s]' \
                % (len(attr), str(attr[0].shape), str(attr[0].dtype), str(id(attr)))
        elif isinstance(attr, dict) \
                and len(attr) > 0 and isinstance(attr[attr.keys()[0]], np.ndarray):
            print a, 'is a dict of arrays comprising:'
            for k, v in attr.iteritems():
                print k, '->'
                report_arrays_in_obj(v)
            print 'end of listing dict "%s"' % a


def has_ext(fn, ext):
    if fn.endswith('gz'):
        return fn.split('.')[-2] == ext
    else:
        return fn.split('.')[-1] == ext


def replace_ext(fn, ext):
    if fn.endswith('gz'):
        return '.'.join(fn.split('.')[:-2] + [ext, 'gz'])
    else:
        return '.'.join(fn.split('.')[:-1] + [ext])


def add_suffix(fn, suffix):
    """ Add a suffix before file extension.

    >>> add_suffix('./my_file.txt', '_my_suffix')
    './my_file_my_suffix.txt'
    """
    if suffix is None:
        return fn
    sfn = op.splitext(fn)
    if sfn[1] == '.gz':
        sfn = op.splitext(fn[:-3])
        sfn = (sfn[0], sfn[1] + '.gz')
    return sfn[0] + suffix + sfn[1]


def add_prefix(fn, prefix):
    """ Add a prefix at the beginning of a file name.

    >>> add_prefix('./my_file.txt', 'my_prefix_')
    './my_prefix_my_file.txt'
    """

    if prefix is None:
        return fn
    sfn = op.split(fn)
    if sfn[1] == '.gz':
        sfn = op.splitext(fn[:-3])
        sfn = (sfn[0], sfn[1] + '.gz')

    return op.join(sfn[0], prefix + sfn[1])


def assert_path_not_in_src(p):
    p = op.realpath(p)
    src_path = op.realpath(op.join(op.dirname(pyhrf.__file__), '../../'))
    for root, _, _ in os.walk(src_path):
        if root == p:
            raise Exception('Directory %s is in source path' % p)


def assert_file_exists(fn):
    if not op.exists(fn):
        raise Exception('File %s does not exists' % fn)


def non_existent_canditate(f, start_idx=1):
    yield f
    i = start_idx
    while True:
        yield add_suffix(f, '_%d' % i)
        i += 1


def non_existent_file(f):
    for f in non_existent_canditate(f):
        if not op.exists(f):
            return f


def do_if_nonexistent_file(*dargs, **kwargs):
    force = kwargs.get('force', False)
    vlevel = kwargs.get('verbose', 20)

    def wrapper(func):
        def wrapper(*args, **kwargs):
            if force:
                return func(*args, **kwargs)

            ins_a, _, _, d = inspect.getargspec(func)
            do_func = False
            checked_fns = []
            for a in dargs:
                if a in kwargs:
                    fn = kwargs[a]
                else:
                    try:
                        iarg = func.func_code.co_varnames.index(a)
                        fn = args[iarg]
                    except (IndexError, ValueError):
                        try:
                            la, ld = len(ins_a), len(d)
                            i = ins_a[la - ld:].index(a)
                        except (IndexError, ValueError):
                            msg = 'Error when defining decorator '\
                                'do_if_nonexistent_file: '\
                                '"%s" is not a valid '\
                                'argument of func %s' \
                                % (a, func.func_name)
                            raise Exception(msg)
                        fn = d[i]
                if not isinstance(fn, str):
                    raise Exception('Arg %s should be a string, %s found'
                                    % (fn, type(fn)))
                if not op.exists(fn):
                    logger.log(vlevel, 'func %s executed because file "%s" does'
                               ' not exist', func.func_name, fn)

                    do_func = True
                    break
                checked_fns.append(fn)
            if do_func:
                return func(*args, **kwargs)
            logger.log(vlevel, 'func %s not executed because file(s) '
                       'exist(s)', func.func_name)
            logger.log(
                vlevel + 10, '\n'.join(['-> ' + f for f in checked_fns]))
            return None
        return wrapper
    return wrapper


def cartesian(*sequences):
    """
    Generate the "cartesian product" of all 'sequences'.  Each member of the
    product is a list containing an element taken from each original sequence.

    Note: equivalent to itertools.product, which is at least 2x faster !!

    """
    length = len(sequences)
    if length < 5:
        # Cases 1, 2 and 3, 4 are for speed only, these are not really
        # required.
        if length == 4:
            for first in sequences[0]:
                for second in sequences[1]:
                    for third in sequences[2]:
                        for fourth in sequences[3]:
                            yield [first, second, third, fourth]
        elif length == 3:
            for first in sequences[0]:
                for second in sequences[1]:
                    for third in sequences[2]:
                        yield [first, second, third]
        elif length == 2:
            for first in sequences[0]:
                for second in sequences[1]:
                    yield [first, second]
        elif length == 1:
            for first in sequences[0]:
                yield [first]
        else:
            yield []
    else:
        head, tail = sequences[:-1], sequences[-1]
        for result in cartesian(*head):
            for last in tail:
                yield result + [last]


def cartesian_combine_args(varying_args, fixed_args=None):
    """
    Construst the cartesian product of varying_args and append fixed_args to it.

    'varying_args': Specify varying arguments as a dict mapping
                    arg names to iterables of arg values.
                    e.g:
                      { 'my_arg1' : ['a','b','c'],
                        'my_arg2' : [2, 5, 10],
                      }
    'fixed_args' : Specify constant arguments as a dict mapping
                   arg names to arg values
                   e.g:
                     { 'my_arg3' : ['fixed_value'] }

    Example:
    >>> from pyhrf.tools import cartesian_combine_args
    >>> vargs = {'my_arg1' : ['a','b','c'],'my_arg2' : [2, 5, 10],}
    >>> fargs = { 'my_arg3' : 'fixed_value' }
    >>> res = cartesian_combine_args(vargs, fargs)
    >>> res == \
        [{'my_arg1': 'a', 'my_arg2': 2, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'b', 'my_arg2': 2, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'c', 'my_arg2': 2, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'a', 'my_arg2': 5, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'b', 'my_arg2': 5, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'c', 'my_arg2': 5, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'a', 'my_arg2': 10, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'b', 'my_arg2': 10, 'my_arg3': 'fixed_value'},
    ...  {'my_arg1': 'c', 'my_arg2': 10, 'my_arg3': 'fixed_value'}]
    True
    """

    if fixed_args is None:
        fixed_args = {}

    return [dict(zip(varying_args.keys(), vp) + fixed_args.items())
            for vp in iproduct(*varying_args.values())]


def icartesian_combine_args(varying_args, fixed_args=None):
    """
    Same as cartesian_combine_args but return an iterator over the
    list of argument combinations
    """

    if fixed_args is None:
        fixed_args = {}

    return (dict(zip(varying_args.keys(), vp) + fixed_args.items())
            for vp in iproduct(*varying_args.values()))


def cartesian_apply(varying_args, func, fixed_args=None, nb_parallel_procs=1,
                    joblib_verbose=0):
    """
    Apply function *func* iteratively on the cartesian product of *varying_args*
    with fixed args *fixed_args*. Produce a tree (nested dicts) mapping arg values    to the corresponding evaluation of function *func*

    Arg:
        - varying_args (OrderedDict): a dictionnary mapping argument names to
                                      a list of values. The Orderdict is
                                      used to keep track of argument order in
                                      the result.
                                      WARNING: all argument values must be
                                               hashable
        - func (function): the function to be applied on the cartesian product
                           of given arguments
        - fixed_args (dict): arguments that are fixed
                             (do not enter cartesian product)

    Return:
        nested dicts (tree) where each node is an argument value from varying
        args and each leaf is the result of the evaluation of the function.
        The order to the tree levels corresponds the order in the input
        OrderedDict of varying arguments.

    Example:
    >>> from pyhrf.tools import cartesian_apply
    >>> from pyhrf.tools.backports import OrderedDict
    >>> def foo(a,b,c): return a + b + c
    >>> v_args = OrderedDict( [('a',[0,1]), ('b',[1,2])] )
    >>> fixed_args = {'c': 3}
    >>> cartesian_apply(v_args, foo, fixed_args) == \
        { 0 : { 1:4, 2:5}, 1 : { 1:5, 2:6} }
    True
    """
    from pyhrf.tools.backports import OrderedDict
    assert isinstance(varying_args, OrderedDict)

    if nb_parallel_procs == 1:
        args_iter = icartesian_combine_args(varying_args, fixed_args)
        return tree([([kwargs[a] for a in varying_args.keys()], func(**kwargs))
                     for kwargs in args_iter])
    else:
        from joblib import Parallel, delayed
        p = Parallel(n_jobs=nb_parallel_procs, verbose=joblib_verbose)
        args = cartesian_combine_args(varying_args, fixed_args)
        results = p(delayed(func)(**kwargs) for kwargs in args)
        return tree([([kwargs[a] for a in varying_args.keys()], r)
                     for kwargs, r in zip(args, results)])


def format_duration(dt):
    s = ''
    if dt / 3600 >= 1:
        s += '%dH' % int(dt / 3600)
        dt = dt % 3600
    if dt / 60 >= 1:
        s += '%dmin' % int(dt / 60)
        dt = int(dt % 60)
    s += '%1.3fsec' % dt
    return s

import pyhrf.ndarray


def swapaxes(array, a1, a2):

    if isinstance(array, np.ndarray):
        return np.swapaxes(array, a1, a2)
    elif isinstance(array, pyhrf.ndarray.xndarray):
        return array.swapaxes(a1, a2)
    else:
        raise Exception('Unknown array type: %s' % str(type(array)))


def rescale_values(a, v_min=0., v_max=1., axis=None):
    a = a.astype(np.float64)  # make sure that precision is sufficient
    a_min = a.min(axis=axis)
    a_max = a.max(axis=axis)

    if axis is not None and axis != 0:
        # make target axis be the 1st to enable bcast
        a = np.swapaxes(a, 0, axis)

    res = (v_min - v_max) * 1. / (a_min - a_max) * (a - a_max) + v_max
    if axis is not None and axis != 0:
        # reposition target axis at original pos
        res = np.swapaxes(res, 0, axis)
    return res


def cartesian_params(**kwargs):
    keys = kwargs.keys()
    for p in cartesian(*kwargs.itervalues()):
        yield dict(zip(keys, p))


def cartesian_eval(func, varargs, fixedargs=None):
    resultTree = {}
    if fixedargs is None:
        fixedargs = {}
    for p in cartesian_params(**varargs):
        fargs = dict(p.items() + fixedargs.items())
        set_leaf(resultTree, [p[k] for k in varargs.iterkeys()], func(**fargs))
    return varargs.keys(), resultTree


def cuboidPrinter(c):
    print c.descrip()


def my_func(**kwargs):
    from pyhrf.ndarray import xndarray
    return xndarray(np.zeros(kwargs['shape']) + kwargs['val'])


def cartesian_test():
    branchLabels, resTree = cartesian_eval(my_func, {'shape': [(2, 5), (6, 8)],
                                                     'val': [4, 1.3]})
    pprint(resTree)
    apply_to_leaves(resTree, cuboidPrinter)


def crop_array(a, m=None, extension=0):
    """
    Return a sub array where as many zeros as possible are discarded
    Increase bounding box of mask by *extension*
    """
    if m is None:
        m = np.where(a != 0.)
    else:
        m = np.where(m != 0)
    mm = np.vstack(m).transpose()
    d = np.zeros(tuple(mm.ptp(0) + 1 + 2 * extension), dtype=a.dtype)
    d[tuple((mm - mm.min(0) + extension).transpose())] = a[m]
    return d


def buildPolyMat(paramLFD, n, dt):

    regressors = dt * np.arange(0, n)
    timePower = np.arange(0, paramLFD + 1, dtype=int)
    regMat = np.zeros((len(regressors), paramLFD + 1), dtype=float)
    logger.info('regMat: %s', str(regMat.shape))
    for v in xrange(paramLFD + 1):
        regMat[:, v] = regressors[:]

    tPowerMat = np.tile(timePower, (n, 1))
    lfdMat = np.power(regMat, tPowerMat)
    lfdMat = np.array(scipy.linalg.orth(lfdMat))
    # print 'lfdMat :', lfdMat
    return lfdMat


def polyFit(signal, tr=1., order=5):
    """
    Polynomial fit of signals.
    'signal' is a 2D matrix with first axis being time and second being position.
    'tr' is the time resolution (dt).
    'order' is the order of the polynom.
    Return the orthogonal polynom basis matrix (P) and fitted coefficients (l),
    such that P.l yields fitted polynoms.
    """
    n = len(signal)
    print 'n:', n, 'tr:', tr
    p = buildPolyMat(order, n, tr)
    ptp = np.dot(p.transpose(), p)
    invptp = np.linalg.inv(ptp)
    invptppt = np.dot(invptp, p.transpose())
    l = np.dot(invptppt, signal)
    return (p, l)


def undrift(signal, tr, order=5):
    """
    Remove the low frequency trend from 'signal' by a polynomial fit.
    Assume axis 3 of 'signal' is time.
    """
    print 'signal:', signal.shape
    m = np.where(np.ones(signal.shape[:3]))
    sm = string.join(['m[%d]' % d for d in xrange(signal.ndim - 1)], ',')
    signal_flat = eval('signal[%s,:]' % sm)
    print 'signal_flat:', signal_flat.shape
    # Least square estimate of drift
    p, l = polyFit(signal_flat.transpose(), tr, order)
    usignal_flat = signal_flat.transpose() - np.dot(p, l)
    usignal = np.zeros_like(signal)
    print 'usignal:', usignal.shape
    exec('usignal[%s,:] = usignal_flat.transpose()' % sm)
    return usignal


def root3(listCoeffs):
    length = len(listCoeffs)
    if length != 4:
        raise polyError(listCoeffs, "wrong poly order")
    if listCoeffs[0] == 0:
        raise polyError(listCoeffs[0], "wrong poly order:null coefficient")
    a = P[1] / P[0]
    b = P[2] / P[0]
    c = P[2] / P[0]

    # change of variables: z = x -a/3
    # Polynome Q(Z)=Z^3 - pZ -q
    p = a ** 2 / 3. - b
    q = (a * b) / 3. - 2. / 27. * a ** 3 - c
    if np.abs(p) < 1e-16:
        polycoeffs = np.zeros((1, 3), dtype=complex)
        polycoeffs[0] = 1
        polycoeffs[1] = (1j) ** (4 / 3.)
        polycoeffs[2] = (-1j) ** (4 / 3.)
        rp = np.multiply(polycoeffs, (np.sign(q) * q) ** (1 / 3.)) - a / 3.
    elif p < 0:
        t2 = 2 * p / 3. / q * np.sqrt(-p / 3.)
        tho = ((np.sqrt(1. + t2 ** 2) - 1) / t2) ** (1 / 3.)
        tho2 = 2. * tho / (1 - tho ** 2)
        tho3 = 2. * tho / (1 + tho ** 2)
        rp = - a / 3. * np.ones((1, 3), dtype=complex)
        fracTho2 = np.sqrt(-p / 3.) / tho2
        fracTho3 = np.sqrt(-p) / tho3
        rp[0] += -2. * fracTho2
        rp[1] += fracTho2 + 1j * fracTho3
        rp[2] += fracTho2 - 1j * fracTho3
    else:
        if np.abs((p / 3.) ** 3 - q ** 2 / 2.) < 1e-16:
            rp = - a / 3. * np.ones((1, 3), dtype=float)
            rp[0] += -3 * q / 2. / p
            rp[1] += -3 * q / 2. / p
            rp[2] += 3. * q / p


def gaussian_kernel(shape):
    """ Returns a normalized ND gauss kernel array for convolutions """
    grid = eval(
        'np.mgrid[%s]' % (string.join(['-%d:%d+1' % (s, s) for s in shape], ',')))
    k = 1. / np.prod([np.exp((grid[d] ** 2 / float(shape[d])))
                      for d in xrange(len(shape))], axis=0)
    return k / k.sum()


def gaussian_blur(a, shape):
    assert a.ndim == len(shape)
    k = gaussian_kernel(shape)
    return scipy.signal.convolve(a, k, mode='valid')


def foo(*args, **kwargs):
    pass


class polyError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def convex_hull_mask(mask):
    """
    Compute the convex hull of the positions defined by the given binary mask

    Args:
        - mask (numpy.ndarray): binary mask of positions to build the chull from

    Return:
        a numpy.ndarray binary mask of positions within the convex hull

    """
    from scipy.spatial import Delaunay

    hull = Delaunay(np.array(np.where(mask)).T)

    result = np.zeros_like(mask)
    m = np.where(np.ones_like(mask))
    result[m] = hull.find_simplex(np.array(m).T) >= 0

    return result


def peelVolume3D(volume, backgroundLabel=0):

    # Make sure that boundaries are filled with zeros
    # -> expand volume with zeros:
    vs = volume.shape
    expVol = np.zeros((vs[0] + 2, vs[1] + 2, vs[2] + 2), dtype=int)
    expVol[1:-1, 1:-1, 1:-1] = volume != backgroundLabel

    # 27-neighbourhood mask:
    neighbMask = np.array([c for c in cartesian([0, -1, 1],
                                                [0, -1, 1],
                                                [0, -1, 1])][1:])
    mask = np.where(expVol != 0)
    coords = np.array(mask).transpose()
    # For each position, compute the number of valid neighbours:
    marks = np.zeros_like(expVol)
    for iv in xrange(len(mask[0])):
        cn = (neighbMask + coords[iv]).transpose()
        curNeighbMask = (cn[0], cn[1], cn[2])
        marks[mask[0][iv], mask[1][iv], mask[2][iv]] = expVol[
            curNeighbMask].sum()

    # Let's go back to the original volume shape:
    trimmedMarks = marks[1:-1, 1:-1, 1:-1]

    # Keep only positions which have 26 neighbours (completely burried):
    peeledVolume = np.zeros_like(volume)
    validPos = np.where(trimmedMarks == 26)
    peeledVolume[validPos] = volume[validPos]

    return peeledVolume


def distance(c1, c2, coord_system=None):
    # TODO: use coordinate system (xform)
    return ((c1 - c2) ** 2).sum() ** .5


def inspect_default_args(args, defaults):
    if defaults is None:
        return {}

    kw_defaults = {}
    firstdefault = len(args) - len(defaults)
    for i in range(firstdefault, len(args)):
        kw_defaults[args[i]] = defaults[i - firstdefault]

    return kw_defaults


class Pipeline:

    THE_ROOT = 0

    def __init__(self, quantities):  # , cached=False, cache_dir='./'):
        """
        Handles a graph of quantities. A quantity can be a variable or
        a callable.
        """
        self.roots = set([])  # will store roots, ie quantities
        # which have no dependency
        self.quantities = {}  # mapping value_label => compute function
        # will hold all labels:
        self.labels = set()
        self.siblings = {}  # sibling labels associated to the same quantity
        # -> eg when a func returns multiple values

        for l, q in quantities.iteritems():
            if isinstance(l, (tuple, list)):
                for e in l:
                    self.quantities[e] = q
                    self.siblings[e] = l
                self.labels.update(l)
            else:
                self.quantities[l] = q
                self.siblings[l] = (l,)
                self.labels.add(l)

        logger.info(
            'labels at init: %s (%d)', str(self.labels), len(self.labels))
        self.dependencies = dict((l, set()) for l in self.labels)
        self.dependers = dict((l, set()) for l in self.labels)
        # virtual common root of the forest
        self.dependers[self.THE_ROOT] = set()
        self.values = {}  # will hold all values
        self.init_dependencies(quantities)

    def add_root(self, label):
        self.roots.add(label)
        # Virtually plant the forest at a common root:
        self.dependers[self.THE_ROOT].add(label)

    def init_dependencies(self, quantities):

        logger.info('Pipeline.init_dependencies ...')
        for labels, val in quantities.iteritems():
            if not isinstance(labels, (list, tuple)):
                labels = (labels,)
            logger.info('treating quantities: %s', str(labels))
            logger.info('val: %s', str(val))
            func = self.get_func(val)
            if func is not None:
                logger.info('... is a function')
                arg_spec = inspect.getargspec(func)
                args = arg_spec[0]
                for label in labels:
                    assert label not in args

                    for arg in args:
                        if self.dependers.has_key(arg):
                            self.dependencies[label].add(arg)
                            self.dependers[arg].add(label)
                        else:
                            if arg_spec[3] is None or \
                                    arg not in args[len(args) - len(arg_spec[3]):]:
                                raise Exception('arg "%s" of function "%s" '
                                                'undefined (no quantity found'
                                                ' or no default value)'
                                                % (arg, val.__name__))

                    if len(self.dependencies[label]) == 0:
                        self.add_root(label)

            else:
                logger.info('... is of type %s', val.__class__)
                for label in labels:
                    self.add_root(label)

        if 0:
            print 'dependers:'
            print self.dependers
            print 'dependencies:'
            print self.dependencies
            print 'roots:'
            print self.roots
            print 'self.quantities:'
            print self.quantities
        self.checkGraph()

    def save_graph_plot(self, image_filename, images=None):
        import pygraphviz as pgv

        g = pgv.AGraph(directed=True)
        for label in self.labels:
            for deper in self.dependers[label]:
                g.add_edge(label, deper)

        for label in self.roots:
            try:
                n = g.get_node(label)
                n.attr['shape'] = 'box'
            except Exception, e:
                print e
                pass

        if images is not None:
            blank_image = pyhrf.get_data_file_name('empty.png')
            for label in self.labels:
                n = g.get_node(label)
                if images.has_key(label):
                    n.attr['image'] = images[label]
                    n.attr['labelloc'] = 'b'
                else:
                    n.attr['image'] = blank_image

        g.layout('dot')
        g.draw(image_filename)

    def update_subgraph(self, root):
        # TODO : limit init of force_eval only to quantities involved
        #       in current subgraph
        self.force_eval = dict([(label, False) for label in self.labels])

        depths = {}
        for lab in self.labels:
            depths[lab] = -1  # mark as not visited
        self.setDepths(root, depths, 0)
        maxDepth = max(depths.values())
        levels = range(maxDepth + 1)

        levels = [[] for d in xrange(maxDepth + 1)]
        for lab, depth in depths.iteritems():
            if depth != -1:
                levels[depth].append(lab)

        updated = dict((l, False) for l in self.labels)
        for level in levels:
            for lab in level:
                self.update_quantity(lab, updated)

    def update_all(self):
        self.update_subgraph(self.THE_ROOT)

    def setDepths(self, label, depths, curDepth):
        for deper in self.dependers[label]:
            # if depth not yet set for this node
            # or
            # if a deeper path has been found to  reach this node :
            if depths[deper] == -1 or depths[deper] < curDepth:
                depths[deper] = curDepth
                self.setDepths(deper, depths, curDepth + 1)

    def resolve(self):
        self.update_subgraph(self.THE_ROOT)

    def get_value(self, label):
        """
        Return the value associated with 'label'
        """
        if len(self.values) == 0:
            self.resolve()

        return self.values[label]

    def get_values(self):
        """
        Return all computed values. Perform a full update if not done yet.
        """
        if len(self.values) == 0:
            self.resolve()

        return self.values

    def reportChange(self, rootLabel):
        """
        Trigger update of the sub graph starting at the given root
        """
        assert rootLabel in self.roots
        self.update_subgraph(rootLabel)

    def reprAllDeps(self):
        """
        Build a string representing the while graph : a concatenation of
        representations of all nodes (see reprDep)
        """
        s = ""
        for lab in self.labels:
            s += self.reprDep(lab)
            s += '\n'
        return s

    def reprDep(self, label):
        """
        Build a string representing all dependencies and dependers of the
        variable 'label'. The returned string is in the form :
               label
        depee1 <-
        depee2 <-
               -> deper1
               -> deper2
        """
        deps = self.dependencies[label]
        if len(deps) > 0:
            maxLDepee = max([len(dep) for dep in deps])
        else:
            maxLDepee = 1
        depeeMark = ' <-\n'
        s = string.join([string.ljust(' ' + dep, maxLDepee) for dep in deps] + [''],
                        depeeMark)
        deps = self.dependers[label]
        deperMark = string.rjust('-> ', maxLDepee + len(depeeMark) + 1)
        s += string.join([deperMark + dep for dep in deps], '\n')
        res = '*' + string.rjust(label, maxLDepee) + '*\n' + s
        return res

    def checkGraph(self):
        """
        Check the rightness of the builded graph (acyclicity, uniqueness and no
        short-circuits)
        """

        # check acyclicity :
        self.visited = {}
        for r in self.roots:
            self.detectCyclity([r])

        # Check there is no short-circuit:
        depths = {}
        for r in self.roots:
            for lab in self.labels:
                depths[lab] = -1
            depths[r] = 0
            self.detectShortCircuit(r, 1, depths)

    def detectShortCircuit(self, curRoot, curDepth, depths):
        """
        Recursive method which detects and corrects short-circuits
        """
        # Breadth graph walk :
        for deper in self.dependers[curRoot]:
            dDeper = depths[deper]
            # if depender was visited and its depth is smaller
            if dDeper != -1 and dDeper < curDepth:
                # Short-circuit detected -> removing it :
                self.removeShortCircuits(deper, depths)
            # Setting depth to the right one :
            depths[dDeper] = curDepth
        # Continue walking ...
        for deper in self.dependers[curRoot]:
            self.detectShortCircuit(deper, curDepth + 1, depths)

    def removeShortCircuits(self, label, depths):
        d = depths[label]
        # Look in direction leaf -> root, one level :
        for depee in self.dependencies[label]:
            dDepee = depths[depee]
            # if dependence was not visited and dependence if further than
            # depth-1 :
            if dDepee != -1 and dDepee != d - 1:
                # Remove discovered shunt :
                print 'removing shunt : %s <-> %s' \
                    % (depee, label)
                self.dependers[depee].remove(label)
                self.dependencies[label].remove(depee)

    def detectCyclity(self, viewedNodes):
        # Depth graph walk, root->leaf direction :
        # if viewedNodes
        root = viewedNodes[-1]
        for deper in self.dependers[root]:
            if deper in viewedNodes:
                msg = 'Cyclicity detected in dependency graph :\n' + \
                    ' origin is %s and dep is %s' % (root, deper)
                raise Exception(msg)
            else:
                viewedNodes.append(deper)
                self.detectCyclity(viewedNodes)
        viewedNodes.pop()

    def get_func(self, f):
        try:
            from joblib.memory import MemorizedFunc
        except ImportError:
            class MemorizedFunc:
                pass  # Dummy class

        if isinstance(f, MemorizedFunc):
            return f.func
        elif inspect.isfunction(f):
            return f
        else:
            return None

    def update_quantity(self, label, updated):

        if updated[label]:  # already updated
            return

        logger.info(" ------------- Update quantity '%s' -----------", label)
        quantity = self.quantities[label]
        siblings = self.siblings[label]
        func = self.get_func(quantity)
        if func is not None:

            logger.info(" -> %s", str(func))

            t0 = time()
            fargs = {}
            args, _, _, d = inspect.getargspec(func)
            defaults = inspect_default_args(args, d)
            for depee in args:
                fargs[depee] = self.values.get(
                    depee, defaults.get(depee, None))

            logger.info('Eval of %s ...', label)
            results = quantity(**fargs)

            logger.info('Quantity %s updated in %s', label,
                        format_duration(time() - t0))
        else:
            if isinstance(quantity, np.ndarray):
                logger.info(' -> ndarray of shape %s and type %s',
                            str(quantity.shape), str(quantity.dtype))
            else:
                logger.info(" -> %s", str(quantity))
            results = quantity

        if len(siblings) > 1:
            assert len(results) == len(siblings)

        else:
            results = (results,)

        for l, r in zip(siblings, results):
            self.values[l] = r
            updated[l] = True


def rebin(a, newshape):
    '''Rebin an array to a new shape.
    Can be used to rebin a func image onto a anat image
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new)
              for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]

    # choose the biggest smaller integer index:
    indices = coordinates.astype('i')

    return a[tuple(indices)]


def resampleToGrid(x, y, xgrid):

    i = 1
    yg = np.empty(xgrid.shape, dtype=y.dtype)
    for ig, xg in enumerate(xgrid):
        while i < len(x) and xg > x[i]:
            i += 1
        if i >= len(x):
            i -= 1
        dx = 0. if x[i] != x[i - 1] else 0.0000001
        yg[ig] = (y[i] - y[i - 1]) * (xg - x[i]) / \
            ((x[i] + dx) - (x[i - 1] - dx)) + y[i]
    return yg


def resampleSignal(s, osf):

    ls = len(s)
    timeMarks = np.arange(osf * (ls - 1), dtype=float) / osf
    prevTMarksSrc = np.array(np.floor(timeMarks), dtype=int)
    nextTMarksSrc = np.array(np.ceil(timeMarks), dtype=int)
    deltaBold = s[nextTMarksSrc, :] - s[prevTMarksSrc, :]
    deltaTime = timeMarks - prevTMarksSrc
    sr = s[prevTMarksSrc, :] + \
        (deltaBold.transpose() * deltaTime).transpose()
    return sr


def diagBlock(mats, rep=0):
    """
    Construct a diagonal block matrix from blocks which can be 1D or
    2D arrays. 1D arrays are taken as column vectors.
    If 'rep' is a non null positive number then blocks are diagonaly 'rep' times
    """
    if type(mats) == np.ndarray:
        finalMat = mats
    elif type(mats) == list and len(mats) == 1:
        finalMat = mats[0]
    elif type(mats) == list:
        m = 0  # nbLines
        n = 0  # nbCols
        for mat in mats:
            m += mat.shape[0]
            n += 1 if len(mat.shape) < 2 else mat.shape[1]
        finalMat = np.zeros((m, n), dtype=float)
        lOffset = 0
        cOffset = 0
        for mat in mats:
            m = mat.shape[0]
            n = 1 if len(mat.shape) < 2 else mat.shape[1]
            sh = finalMat[lOffset:lOffset + m, cOffset:cOffset + n].shape
            finalMat[
                lOffset:lOffset + m, cOffset:cOffset + n] = mat.reshape(sh)[:]
            lOffset += m
            cOffset += n
    else:
        raise Exception('diagBlock: unrecognised type for "mats"')

    if rep > 0:
        return diagBlock([finalMat] * rep)
    else:
        return finalMat


def extractRoiMask(nmask, roiId):

    mask = nmask == roiId
    m = np.where(mask)
    mm = np.vstack(m).transpose()
    cropMask = np.zeros(mm.ptp(0) + 1, dtype=int)
    cropMask[tuple((mm - mm.min(0)).transpose())] = mask[m]

    return cropMask


def describeRois(roiMask):

    s = 'Number of voxels : %d\n' % (roiMask != 0).sum()
    s += 'Number of regions : %d\n' % (
        len(np.unique(roiMask)) - int(0 in roiMask))
    s += 'Region sizes :\n'
    counts = np.bincount(roiMask[roiMask > 0])
    s += np.array2string(counts) + '\n'
    nbr = (counts >= 60).sum()
    s += 'Nb of region with size > 60: %d\n' % nbr

    return s


def array_summary(a, precision=4):
    return '%s -- %1.*f(%1.*f)[%1.*f...%1.*f]' % (str(a.shape),
                                                  precision, a.mean(),
                                                  precision, a.std(),
                                                  precision, a.min(),
                                                  precision, a.max())


def get_2Dtable_string(val, rownames, colnames, precision=4, col_sep='|',
                       line_end='', line_start='', outline_char=None):
    """ Return a nice tabular string representation of a 2D numeric array
    #TODO : make colnames and rownames optional
    """

    if val.ndim == 1:
        val = val.reshape(val.shape[0], 1)
    nrows, ncols = val.shape[:2]

    if np.isscalar(val[0, 0]):
        valWidth = len(str('%1.*f' % (precision, -3.141658938325)))
    else:
        if (val >= 0).all():
            valWidth = len(array_summary(val[0, 0], precision=precision))
        else:
            valWidth = len(array_summary(np.abs(val[0, 0]) * -1,
                                         precision=precision))

    rowWidth = max([len(str(rn)) for rn in rownames])
    colWidth = [max([valWidth, len(cn)]) for cn in colnames]

    sheader = line_start + ' ' * rowWidth + '  ' + col_sep
    sheader += col_sep.join([' %*s ' % (colWidth[j], cn)
                             for j, cn in enumerate(colnames)])
    sheader += line_end + '\n'

    scontent = ''
    for i in xrange(nrows):
        line = line_start + ' %*s ' % (rowWidth, str(rownames[i])) + col_sep
        for j in xrange(ncols):
            if np.isscalar(val[i, j]):
                line += ' %*.*f ' % (colWidth[j],
                                     precision, val[i, j]) + col_sep
            else:
                line += ' %*s ' % (valWidth,
                                   array_summary(val[i, j], precision)) + col_sep

        if outline_char is not None:
            outline = outline_char * (len(line) - 1 + len(line_end)) + '\n'
        else:
            outline = ''
        scontent += outline + line[:-len(col_sep)] + line_end + '\n'

    return outline + sheader + scontent + outline


def get_leaf(element, branch):
    """
    Return the nested leaf element corresponding to all dictionnary keys in
    branch from element
    """
    if isinstance(element, dict) and len(branch) > 0:
        return get_leaf(element[branch[0]], branch[1:])
    else:
        return element


def set_leaf(tree, branch, leaf, branch_classes=None):
    """
    Set the nested *leaf* element corresponding to all dictionnary keys
    defined in *branch*, within *tree*
    """
    assert isinstance(tree, dict)
    if len(branch) == 1:
        tree[branch[0]] = leaf
        return
    if not tree.has_key(branch[0]):
        if branch_classes is None:
            tree[branch[0]] = tree.__class__()
        else:
            tree[branch[0]] = branch_classes[0]()
    else:
        assert isinstance(tree[branch[0]], dict)
    if branch_classes is not None:
        set_leaf(tree[branch[0]], branch[1:], leaf, branch_classes[1:])
    else:
        set_leaf(tree[branch[0]], branch[1:], leaf)


def swap_layers(t, labels, l1, l2):
    """ Create a new tree from t where layers labeled by l1 and l2 are swapped.
    labels contains the branch labels of t.
    """
    i1 = labels.index(l1)
    i2 = labels.index(l2)
    nt = t.__class__()  # new tree init from the input tree
    # can be dict or OrderedDict
    for b, l in izip(treeBranches(t), tree_leaves(t)):
        nb = list(b)
        nb[i1], nb[i2] = nb[i2], nb[i1]
        set_leaf(nt, nb, l)

    return nt


def tree_rearrange(t, oldlabels, newlabels):
    """ Create a new tree from t where layers are rearranged following newlabels.
    oldlabels contains the branch labels of t.
    """
    order = [oldlabels.index(nl) for nl in newlabels]
    nt = t.__class__()  # new tree
    for b, l in izip(treeBranches(t), tree_leaves(t)):
        nb = [b[i] for i in order]
        set_leaf(nt, nb, l)

    return nt


def treeBranches(tree, branch=None):
    if branch is None:
        branch = []
    if isinstance(tree, dict):
        for k in tree.iterkeys():
            for b in treeBranches(tree[k], branch + [k]):
                yield b
    else:
        yield branch


def tree(branched_leaves):
    d = {}
    for branch, leaf in branched_leaves:
        set_leaf(d, branch, leaf)
    return d


def treeBranchesClasses(tree, branch=None):
    if branch is None:
        branch = []
    if isinstance(tree, dict):
        for k, v in tree.iteritems():
            for b in treeBranchesClasses(tree[k], branch + [v.__class__]):
                yield b
    else:
        yield branch


def tree_leaves(tree):
    for branch in treeBranches(tree):
        yield get_leaf(tree, branch)


def tree_items(tree):
    """
    """
    for branch in treeBranches(tree):
        yield (branch, get_leaf(tree, branch))


def stack_trees(trees, join_func=None):
    """ Stack trees (python dictionnaries) with identical structures
    into one tree
    so that one leaf of the resulting tree is a list of the corresponding leaves
    across input trees. 'trees' is a list of dict
    """
    stackedTree = trees[0].__class__()

    for branch, branch_classes in izip(treeBranches(trees[0]),
                                       treeBranchesClasses(trees[0])):
        if join_func is None:
            leaveList = [get_leaf(tree, branch) for tree in trees]
        else:
            leaveList = join_func([get_leaf(tree, branch) for tree in trees])
        set_leaf(stackedTree, branch, leaveList, branch_classes)
    return stackedTree


def unstack_trees(tree):
    """ Return a list of tree from a tree where leaves are all lists with
    the same number of items
    """

    first_list = tree_leaves(tree).next()
    tree_list = [tree.__class__() for i in xrange(len(first_list))]
    for b, l in tree_items(tree):
        for t, item in zip(tree_list, l):
            set_leaf(t, b, item)

    return tree_list


from pprint import pprint


def apply_to_leaves(tree, func, funcArgs=None, funcKwargs=None):
    """
    Apply function 'func' to all leaves in given 'tree' and return a new tree.
    """
    if funcKwargs is None:
        funcKwargs = {}
    if funcArgs is None:
        funcArgs = []

    newTree = tree.__class__()  # could be dict or OrderedDict
    for branch, leaf in tree_items(tree):
        set_leaf(newTree, branch, func(leaf, *funcArgs, **funcKwargs))
    return newTree


def map_dict(func, d):
    return d.__class__((k, func(v)) for k, v in d.iteritems())


def get_cache_filename(args, path='./', prefix=None, gz=True):
    hashArgs = hashlib.sha512(repr(args)).hexdigest()
    if prefix is not None:
        fn = os.path.join(path, prefix + '_' + hashArgs + '.pck')
    else:
        fn = os.path.join(path, hashArgs + '.pck')
    if gz:
        return fn + '.gz'
    else:
        return fn


def hash_func_input(func, args, digest_code):

    if isinstance(args, dict):  # sort keys
        to_digest = ''
        for k in sorted(args.keys()):
            v = args[k]
            to_digest += repr(k) + repr(v)
    else:
        to_digest = repr(args)

    if digest_code:
        to_digest += inspect.getsource(func)

    return hashlib.sha512(to_digest).hexdigest()


def cache_filename(func, args=None, prefix=None, path='./',
                   digest_code=False):
    if prefix is None:
        prefix = func.__name__
    else:
        prefix = prefix + '_' + func.__name__

    hashArgs = hash_func_input(func, args, digest_code)

    fn = os.path.join(path, prefix + '_' + hashArgs + '.pck.gz')
    return fn


def cache_exists(func, args=None, prefix=None, path='./',
                 digest_code=False):

    return op.exists(cache_filename(func, args=args, prefix=prefix,
                                    path=path, digest_code=digest_code))


def cached_eval(func, args=None, new=False, save=True, prefix=None,
                path='./', return_file=False, digest_code=False,
                gzip_mode='cmd'):

    fn = cache_filename(func, args=args, prefix=prefix,
                        path=path, digest_code=digest_code)

    if not os.path.exists(fn) or new:
        if args is None:
            r = func()
        elif isinstance(args, tuple) or isinstance(args, list):
            r = func(*args)
        elif isinstance(args, dict):
            r = func(**args)
        else:
            raise Exception("type of arg (%s) is not valid. Should be tuple, "
                            "list or dict" % str(args.__class__))
        if save:
            if gzip_mode == 'pygzip' and os.system('gzip -V') != 0:
                f = gzip.open(fn, 'w')
                cPickle.dump(r, f)
                f.close()
            else:
                f = open(fn[:-3], 'w')
                cPickle.dump(r, f)
                f.close()
                os.system("gzip -f %s" % fn[:-3])
        if not return_file:
            return r
        else:
            return fn
    else:
        if not return_file:
            return cPickle.load(gzip.open(fn))
        else:
            return fn


def montecarlo(datagen, festim, nbit=None):
    """Perform a Monte Carlo loop with data generator 'datagen' and estimation
    function 'festim'.
    'datagen' have to be iterable.
    'festim' must return an object on which ** and + operators can be applied.
    If 'nbit' is provided then use it for the maximum iterations else loop until
    datagen stops.
    """
    itgen = iter(datagen)

    s = itgen.next()
    e = festim(s)
    cumul = e
    cumul2 = e ** 2

    if nbit is None:
        nbit = 0
        for s in itgen:
            e = festim(s)
            cumul += e
            cumul2 += e ** 2
            nbit += 1
    else:
        for i in xrange(1, nbit):
            s = itgen.next()
            e = festim(s)
            cumul += e
            cumul2 += e ** 2

    m = cumul / float(nbit)
    v = cumul2 / float(nbit) - m ** 2
    return m, v


def closestsorted(a, val):
    i = np.searchsorted(a, val)
    if i == len(a) - 1:
        return i
    elif np.abs(a[i] - val) < np.abs(a[i + 1] - val):
        return i
    else:
        return i + 1


##################
def calc_nc2D(a, b):
    return a + b + 2 * (a - 1) * (b - 1) - 2


def nc2DGrid(maxSize):
    nc2D = np.zeros((maxSize, maxSize), dtype=int)
    for a in xrange(maxSize):
        for b in xrange(maxSize):
            nc2D[a, b] = calc_nc2D(a, b)
    return nc2D


def now():
    return datetime.datetime.fromtimestamp(time())


def time_diff_str(diff):
    return '%dH%dmin%dsec' % (diff.seconds / 3600,
                              (diff.seconds % 3600) / 60,
                              (diff.seconds % 3600) % 60)


class AnsiColorizer:
    """ Format strings with an ANSI escape sequence to encode color """

    BEGINC = '\033['
    COLORS = {
        'purple': '95',
        'blue': '94',
        'green': '92',
        'yellow': '93',
        'red': '91',
    }

    ENDC = '\033[0m'

    def __init__(self):
        self.disabled = False
        self.do_tty_check = True

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def no_tty_check(self):
        self.do_tty_check = False

    def tty_check(self):
        self.do_tty_check = True

    def __call__(self, s, color, bright=False, bold=False):
        if color not in self.COLORS:
            raise Exception('Invalid color "%s". Available colors: %s'
                            % (color, str(self.COLORS)))

        col = self.COLORS[color]
        if self.disabled or (self.do_tty_check and not sys.stdout.isatty()):
            return s
        else:
            ansi_codes = ";".join([['', '1'][bright], col])
            return '%s%sm%s%s' % (self.BEGINC, ansi_codes, s, self.ENDC)


colorizer = AnsiColorizer()


def extract_file_series(files):
    """
    group all file names sharing a common prefix followed by a number, ie:
    <prefix><number><extension>
    Return a dictionnary with two levels (<tag>,<extension>), mapped to all
    corresponding series index found.
    """
    series = {}  # will map (tag,extension) to number series
    rexpSeries = re.compile("(.*?)([0-9]+)[.]([a-zA-Z.~]*?)\Z")
    series = defaultdict(lambda: defaultdict(list))
    for f in files:
        r = rexpSeries.match(f)
        if r is not None:
            (tag, idx, ext) = r.groups()
            ext = '.' + ext
        else:
            tag, ext = op.splitext(f)
            idx = ''
        series[tag][ext].append(idx)
    return series


def format_serie(istart, iend):
    return colorizer(['[%d...%d]' % (istart, iend),
                      '[%d]' % (istart)][istart == iend],
                     'red')


def condense_series(numbers):
    if len(numbers) == 1:
        return numbers[0]
    else:
        inumbers = np.sort(np.array(map(int, numbers)))
        if (np.diff(inumbers) == 1).all():
            return format_serie(inumbers.min(), inumbers.max())
        else:
            segment_start = 0
            s = ''
            for segment_end in np.where(np.diff(inumbers) != 1)[0]:
                s += format_serie(inumbers[segment_start],
                                  inumbers[segment_end])
                segment_start = segment_end + 1
            s += format_serie(inumbers[segment_start], inumbers[-1])
            return s


def group_file_series(series, group_rules=None):
    if group_rules is None:
        group_rules = []

    groups = defaultdict(dict)
    dummy_tag = 0
    for tag, ext_data in series.iteritems():
        tag_has_been_grouped = False
        for gr in group_rules:
            r = gr.match(tag)
            if r is not None:
                gname = r.group('group_name')
                groups[gname]['...'] = gname
                tag_has_been_grouped = True
                break

        if not tag_has_been_grouped:
            for ext, numbers in ext_data.iteritems():

                if '' in numbers:
                    groups[dummy_tag][ext] = tag
                    numbers.remove('')

                if len(numbers) > 0:
                    groups[tag][ext] = tag + condense_series(numbers)
            dummy_tag += 1

    final_groups = []
    for tag, series in groups.iteritems():
        sv = series.values()
        if len(sv) > 1 and len(set(sv)) == 1:
            exts = [ext[1:] for ext in series.keys()]
            final_groups.append(
                sv[0] + colorizer('.{%s}' % ','.join(exts), 'green'))
        else:
            for ext, s in series.iteritems():
                if ext == '...':
                    ext = colorizer('...', 'purple')
                final_groups.append(s + ext)

    return sorted(final_groups)


def check_files_series(fseries, verbose=False):

    ok_status = True
    for tag, dext in fseries.iteritems():
        for ext, indexes in dext.iteritems():
            if 0:
                print 'tag', tag, 'ext', ext
                print 'indexes:', indexes
            sorted_indexes = sorted(indexes)
            last = int(sorted_indexes[-1])
            first = int(sorted_indexes[0])
            diff = last - first + 1 - len(indexes)
            if diff != 0:
                ok_status = False
                if verbose:
                    print '%d items missing for series %s[...].%s' \
                        % (diff, tag, ext)
                    print '-> Series should have %d items (%s to %s)' \
                          ' - found %d items' \
                          % (last - first + 1, first, last, len(indexes))
    return ok_status


# Factorisation of the code: functions creations
def Extract_TTP_whM_hrf(hrf, dt):
    """
    Extract TTP and whM from an hrf
    """
    from scipy.interpolate import interp1d
    from pyhrf.boldsynth.hrf import getCanoHRF
    hcano = getCanoHRF()

    Abscisses = np.arange(hrf.size) * dt

    # TTP calculus
    TTP = hrf.argmax() * dt
    print 'found TTP:', TTP

    # whM calculus
    # 1/ Round the HRF
    HRF_rounded = np.round(hrf, 5)

    # 2/ Interpolation to obtain more values
    Abscisses_round = np.arange(HRF_rounded.size) * dt
    f = interp1d(Abscisses, hrf)
    r = 0.00001
    HRF_interp = f(np.arange(0, Abscisses_round[len(Abscisses_round) - 1], r))
    HRF_interp_rounded = np.round(HRF_interp, 5)

    # To reconvert from interpolation to correct values in seconds
    len_use = len(HRF_interp)
    dt_interp = (hcano[0][len(hcano[0]) - 1]) / len(HRF_interp)

    # 3/ Where the half max is found
    Pts_1_2_h = np.where(
        abs(HRF_interp_rounded - HRF_rounded.max() / 2.) < 0.001)
    Values_pts_1_2_h = HRF_interp[Pts_1_2_h]
    if Pts_1_2_h[0].shape[0] == 0:
        print '#### No point found ####'

    # Selection of Pts of abscisse<max
    Diff1 = abs(Pts_1_2_h - HRF_interp_rounded.argmax()) * \
        (Pts_1_2_h < HRF_interp_rounded.argmax())
    Diff1_non_zeros = Diff1[0][np.where(Diff1[0] > 0)]  # retrieve positions#0
    Diff1_non_zeros.sort()  # to sort all differences
    First_diff1 = Diff1_non_zeros.mean()

    # Selection of Pts of abscisse>max
    Diff2 = abs(HRF_interp_rounded.argmax() - Pts_1_2_h) * \
        (Pts_1_2_h > HRF_interp_rounded.argmax())
    Diff2_non_zeros = Diff2[0][np.where(Diff2[0] > 0)]  # retrieve positions#0
    Diff2_non_zeros.sort()  # to sort all differences
    First_diff2 = Diff2_non_zeros.mean()

    # addition of the two differences and *dt_interp to obtain whM in seconds
    whM = (First_diff1 + First_diff2) * dt_interp
    print 'found whM:', whM

    return TTP, whM


def Extract_TTP_whM_from_group(hrfs_pck_file, dt, model, Path_data, acq):
    """
    Extract TTP and whM from a group of hrfs whose values are saved in a .pck (size nb_subjects * nb_coeff_hrf)
    """
    from scipy.interpolate import interp1d
    from pyhrf.boldsynth.hrf import getCanoHRF
    hcano = getCanoHRF()

    hrfs = cPickle.load(open(hrfs_pck_file))

    nb_subjects = hrfs.shape[0]

    TTP_tot = np.zeros((nb_subjects))
    whM_tot = np.zeros((nb_subjects))

    for isubj in np.arange(nb_subjects):

        HRF_at_max = hrfs[isubj, :]

        Abscisses = np.arange(HRF_at_max.size) * dt

        # TTP calculus
        TTP = HRF_at_max.argmax() * dt
        print 'found TTP:', TTP
        TTP_tot[isubj] = TTP

        # whM calculus
        # 1/ Round the HRF
        HRF_rounded = np.round(HRF_at_max, 5)

        # 2/ Interpolation to obtain more values
        Abscisses_round = np.arange(HRF_rounded.size) * dt
        f = interp1d(Abscisses, HRF_at_max)
        r = 0.00001
        HRF_interp = f(
            np.arange(0, Abscisses_round[len(Abscisses_round) - 1], r))
        HRF_interp_rounded = np.round(HRF_interp, 5)

        # To reconvert from interpolation to correct values in seconds
        len_use = len(HRF_interp)
        dt_interp = (hcano[0][len(hcano[0]) - 1]) / len(HRF_interp)

        # 3/ Where the half max is found
        Pts_1_2_h = np.where(
            abs(HRF_interp_rounded - HRF_rounded.max() / 2.) < 0.001)
        Values_pts_1_2_h = HRF_interp[Pts_1_2_h]
        if Pts_1_2_h[0].shape[0] == 0:
            print '#### No point found ####'

        # Selection of Pts of abscisse<max
        Diff1 = abs(Pts_1_2_h - HRF_interp_rounded.argmax()) * \
            (Pts_1_2_h < HRF_interp_rounded.argmax())
        # retrieve positions#0
        Diff1_non_zeros = Diff1[0][np.where(Diff1[0] > 0)]
        Diff1_non_zeros.sort()  # to sort all differences
        First_diff1 = Diff1_non_zeros.mean()

        # Selection of Pts of abscisse>max
        Diff2 = abs(HRF_interp_rounded.argmax() - Pts_1_2_h) * \
            (Pts_1_2_h > HRF_interp_rounded.argmax())
        # retrieve positions#0
        Diff2_non_zeros = Diff2[0][np.where(Diff2[0] > 0)]
        Diff2_non_zeros.sort()  # to sort all differences
        First_diff2 = Diff2_non_zeros.mean()

        # addition of the two differences and *dt_interp to obtain whM in
        # seconds
        whM = (First_diff1 + First_diff2) * dt_interp
        print 'found whM:', whM

        whM_tot[isubj] = np.round(whM, 1)

        cPickle.dump(TTP_tot, open(
            Path_data + '/_TTPs_at_peak_by_hand_' + '_' + model + '_' + acq + '.pck', 'w'))
        cPickle.dump(whM_tot, open(
            Path_data + '/_whMs_at_peak_by_hand_' + '_' + model + '_' + acq + '.pck', 'w'))

    return TTP_tot, whM_tot


def PPMcalculus_jde(threshold_value, apost_mean_activ_fn, apost_var_activ_fn,
                    apost_mean_inactiv_fn, apost_var_inactiv_fn, labels_activ_fn,
                    labels_inactiv_fn, nrls_fn, mask_file, null_hyp=True):
    '''
    Function to calculate the probability that the nrl in voxel j,
    condition m, is superior to a given hreshold_value
    Computation for all voxels
    Compute Tvalue according to null hypothesis
    '''
    from scipy.integrate import quad
    from pyhrf.ndarray import xndarray
    from scipy.stats import norm

    mask = xndarray.load(mask_file).data
    apost_mean_activ = xndarray.load(apost_mean_activ_fn)
    apost_mean_inactiv = xndarray.load(apost_mean_inactiv_fn)
    apost_var_activ = xndarray.load(apost_var_activ_fn)
    apost_var_inactiv = xndarray.load(apost_var_inactiv_fn)
    labels_activ = xndarray.load(labels_activ_fn)
    labels_inactiv = xndarray.load(labels_inactiv_fn)
    nrls = xndarray.load(nrls_fn)

    # flattend data
    m1 = apost_mean_activ.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    m2 = apost_mean_inactiv.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    var1 = apost_var_activ.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    var2 = apost_var_inactiv.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    perc1 = labels_activ.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    perc2 = labels_inactiv.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data
    nrls_values = nrls.flatten(
        mask, axes=['sagittal', 'coronal', 'axial'], new_axis='position').data

    Probas = np.zeros(perc1.shape[0])
    if null_hyp:
        Pvalues = np.zeros(perc1.shape[0])
        # to detect positions activ and inactiv
        Comp = perc1 - perc2
        Pos_activ = np.where(Comp[:, 0])
        Pos_inactiv = np.where(Comp[:, 0] < 0)

        Means = np.zeros(perc1.shape[0])
        Vars = np.zeros(perc1.shape[0])

    for i in xrange(perc1.shape[0]):
        # posterior probability distribution
        fmix = lambda x: perc1[
            i] * norm.pdf(x, m1[i], var1[i] ** .5) + perc2[i] * norm.pdf(x, m2[i], var2[i] ** .5)

        Probas[i] = quad(fmix, threshold_value, float('inf'))[0]

    if null_hyp:
        Means[Pos_activ] = m1[Pos_activ]
        Means[Pos_inactiv] = m2[Pos_inactiv]
        Vars[Pos_activ] = var1[Pos_activ]
        Vars[Pos_inactiv] = var2[Pos_inactiv]
        for i in xrange(perc1.shape[0]):
            nrl_val = nrls_values[i]
            fmix = lambda x: norm.pdf(x, 0, Vars[i] ** .5)
            Pvalues[i] = quad(fmix, nrl_val, float('inf'))[0]

    # deflatten to retrieve original shape
    PPM_ = xndarray(Probas, axes_names=['position'])
    PPM = PPM_.expand(mask, 'position', ['sagittal', 'coronal', 'axial'])

    PPMinvv = 1 - Probas  # to obtain more readable maps
    PPMinv_ = xndarray(PPMinvv, axes_names=['position'])
    PPMinv = PPMinv_.expand(mask, 'position', ['sagittal', 'coronal', 'axial'])

    Pval_ = xndarray(Pvalues, axes_names=['position'])
    Pval = Pval_.expand(mask, 'position', ['sagittal', 'coronal', 'axial'])

    return PPM.data, PPMinv.data, Pval.data


## HTML formating ##

def html_row(s):
    return '<tr>%s</tr>' % s


def html_table(s, border=None):
    if border is None:
        return '<table>%s</table>' % s
    else:
        return '<table border="%d">%s</table>' % (border, s)


def attrs_to_string(attrs):
    attrs = attrs or {}
    sattrs = ''
    if len(attrs) > 0:
        sattrs = ' ' + ' '.join(['%s="%s"' % (k, v) for k, v in attrs.items()])
    return sattrs


def html_img(fn, attrs=None):
    return '<img src="%s"%s>' % (fn, attrs_to_string(attrs))


def html_cell(s, cell_type='d', attrs=None):
    return '<t%s%s>%s</t%s>' % (cell_type, attrs_to_string(attrs), s, cell_type)


def html_div(s, attrs=None):
    return '<div%s>%s</div>' % (attrs_to_string(attrs), s)


def html_list_to_row(l, cell_types, attrs):
    if not isinstance(attrs, (list, tuple)):
        attrs = [attrs] * len(l)
    else:
        assert len(attrs) == len(l)

    if not isinstance(cell_types, (list, tuple)):
        cell_types = [cell_types] * len(l)
    else:
        assert len(cell_types) == len(l)

    return html_row(''.join([html_cell(e, t, a)
                             for e, t, a in zip(l, cell_types, attrs)]))


def html_doc(s):
    return '<!DOCTYPE html><html>' + s + '</html>'


def html_head(s):
    return '<head>' + s + '</head>'


def html_style(s):
    return '<style>' + s + '</style>'


def html_body(s):
    return '<body>' + s + '</body>'
