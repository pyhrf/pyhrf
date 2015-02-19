# -*- coding: utf-8 -*-
"""
This module provides classes and functions to handle multi-dimensionnal numpy
array (ndarray) objects and extend them with some semantics (axes labels and
axes domains). See xndarray class.
(TODO: make xndarray inherit numpy.ndarray?)
"""

import os.path as op
import numpy as np
import logging

from pprint import pformat

from pkg_resources import parse_version

import pyhrf
from pyhrf.tools import (treeBranches, rescale_values, has_ext, tree_items,
                         html_cell, html_list_to_row, html_row, html_table,
                         html_img, html_div)
from pyhrf.tools.backports import OrderedDict


logger = logging.getLogger(__name__)


debug = False

MRI3Daxes = ['sagittal', 'coronal', 'axial']
MRI4Daxes = MRI3Daxes + ['time']
TIME_AXIS = 3


class ArrayMappingError(Exception):
    pass


class xndarray:
    """ Handles a multidimensional numpy array with axes that are labeled
    and mapped to domain values.

    Example :
    c = xndarray( [ [4,5,6],[8,10,12] ], ['time','position'], {'time':[0.1,0.2]} )
    Will represent the following situation:

    position
    ------->
    4  5  6 | t=0.1   |time
    8 10 12 | t=0.2   v

    """

    def __init__(self, narray, axes_names=None, axes_domains=None,
                 value_label="value", meta_data=None):
        """
        Initialize a new xndarray object from 'narray' with 'axes_names'
        as axes labels and 'axes_domains' as domains mapped to integer slices.

        Args:
            - narray (numpy.ndarray): the wrapped numpy array
            - axes_names (listof str):
                labels of array axes
                if None: then axes_names = [\"dim0\", \"dim1\", ...]
            - axes_domains (dictof <str>:<numpy array>):
                domains associated to axes.
                If a domain is not specified then it defaults to
                range(size(axis))


        Return: a xndarray instance
        """
        logger.debug('xndarray.__init__ ...')

        narray = np.asarray(narray)
        self.data = narray
        self.value_label = value_label
        self.meta_data = meta_data
        self.has_deprecated_xml_header = True

        nbDims = self.data.ndim

        if axes_names is None:
            self.axes_names = ['dim' + str(i) for i in xrange(nbDims)]
        else:
            assert type(axes_names) == list
            if len(axes_names) != nbDims:
                raise Exception("length of axes_names (%d) is different "
                                "from nb of dimensions (%d).\n"
                                "Got axes names: %s"
                                % (len(axes_names), nbDims, str(axes_names)))

            self.axes_names = axes_names[:]

        self.axes_ids = dict([(self.axes_names[i], i) for i in xrange(nbDims)])

        # By default: domain of axis = array of slice indexes
        sh = self.data.shape
        self.axes_domains = dict([(axis, np.arange(sh[i]))
                                  for i, axis in enumerate(self.axes_names)])

        if axes_domains is not None:
            assert isinstance(axes_domains, dict)

            for an, dom in axes_domains.iteritems():
                if an not in self.axes_names:
                    raise Exception('Axis "%s" defined in domains not '
                                    'found in axes (%s)'
                                    % (an, ','.join(self.axes_names)))

                ia = self.axes_names.index(an)
                l = self.data.shape[ia]
                if len(dom) != l:
                    raise Exception('Length of domain for axis "%s" (%d) '
                                    'does not match length of data '
                                    'axis %d (%d) ' % (an, len(dom), ia, l))

                if len(set(dom)) != len(dom):
                    raise Exception('Domain of axis "%s" does not contain '
                                    'unique values' % an)

                axes_domains[an] = np.asarray(dom)

            self.axes_domains.update(axes_domains)

        logger.debug('Axes names: %s', str(self.axes_names))
        logger.debug('Axes domains: %s', str(self.axes_domains))

    @staticmethod
    def xndarray_like(c, data=None):
        """
        Return a new cuboid from data with axes, domains and value label
        copied from 'c'. If 'data' is provided then set it as new cuboid's data,
        else a zero array like c.data is used.

        TODO: test
        """
        if data is None:
            data = np.zeros_like(c.data)
        return xndarray(data, c.axes_names, c.axes_domains.copy(),
                        c.value_label, c.meta_data)

    def get_axes_ids(self, axes_names):
        """ Return the index of all axes in given axes_names
        """
        assert set(axes_names).issubset(self.axes_names)
        return [self.get_axis_id(an) for an in axes_names]

    def len(self, axis):
        return self.data.shape[self.get_axis_id(axis)]

    def __repr__(self):
        return 'axes: ' + str(self.axes_names) + ', ' + repr(self.data)

    def get_axis_id(self, axis_name):
        """ Return the id of an axis from the given name.
        """
        logger.debug('core.cuboid ... getting id of %s', axis_name)
        logger.debug('from : %s', str(self.axes_ids))
        if isinstance(axis_name, str):  # axis_name is a string axis name
            if axis_name in self.axes_ids.keys():
                return self.axes_ids[axis_name]
            else:
                return None
        else:  # axis_name is an integer axis index
            if axis_name >= 0 and axis_name < self.get_ndims():
                return axis_name
            else:
                return None
    # get_axis_id

    def set_axis_domain(self, axis_id, domain):
        """
        Set the value domain mapped to *axis_id* as *domain*

        Args:
            - axis_id (str): label of the axis
            - domain (numpy.ndarray): value domain
        Return:
            None
        """

        assert axis_id in self.axes_domains

        if axis_id is not None:
            logger.debug('setting domain of axis %s with %s', str(axis_id),
                         str(domain))
            if len(domain) != self.data.shape[axis_id]:
                raise Exception('length of domain values (%d) does not '
                                ' match length of data (%d) for axis %s'
                                % (len(domain), self.data.shape[axis_id],
                                   self.get_axis_name(axis_id)))
            self.axes_domains[axis_id] = np.array(domain)
    # set_axis_domain

    def get_domain(self, axis_id):
        """
        Return the domain of the axis *axis_id*

        example:
        >>> from pyhrf.ndarray import xndarray
        >>> c = xndarray(np.random.randn(10,2), axes_names=['x','y'], \
                       axes_domains={'y' : ['plop','plip']})
        >>> c.get_domain('y')  # doctest: +NORMALIZE_WHITESPACE
        array(['plop', 'plip'],
              dtype='|S4')
        >>> c.get_domain('x') #default domain made of slice indexes
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
        if axis_id in self.axes_domains:
            return self.axes_domains[axis_id]
        else:
            raise Exception('Unknow axis %s' % axis_id)
    # get_domain

    def swapaxes(self, a1, a2):
        """
        Swap axes *a1* and *a2*

        Args:
            - a1 (str|int): identifier of the 1st axis
            - a2 (str|int): identifier of the 2nd axis

        Return:
            A new cuboid wrapping a swapped view of the numpy array
        """
        an = self.axes_names[:]
        ia1, ia2 = self.get_axis_id(a1), self.get_axis_id(a2)
        an[ia2], an[ia1] = an[ia1], an[ia2]
        return xndarray(np.swapaxes(self.data, ia1, ia2), an, self.axes_domains,
                        self.value_label, self.meta_data)

    def roll(self, axis, pos=-1):
        """
        Roll xndarray by making 'axis' the last axis.
        'pos' is either 0 or -1 (first or last, respectively)
        TODO: handle all pos.
        """
        i = self.get_axis_id(axis)
        if (i == 0 and pos == 0) or (i == self.get_ndims() - 1 and pos == -1):
            return self
        if pos == 0:
            raise NotImplementedError(' pos=0 not coded yet. TODO')

        self.data = np.rollaxis(self.data, i, len(self.axes_names))
        self.axes_names = self.axes_names[:i] + self.axes_names[i + 1:] + \
            [axis]
        self.axes_ids = dict([(a, i) for i, a in enumerate(self.axes_names)])
        return self

    def repeat(self, n, new_axis, domain=None):
        """
        Return a new cuboid with self's data repeated 'n' times along a new
        axis labelled 'new_axis'. Associated 'domain' can be provided.
        """
        if new_axis is None:
            new_axis = self.get_new_axis_name()

        return stack_cuboids([self.copy() for i in xrange(n)], new_axis, domain)

    def squeeze_all_but(self, axes):
        to_squeeze = [a for i, a in enumerate(self.axes_names)
                      if self.data.shape[i] == 1 and a not in axes]
        if len(to_squeeze) == 0:
            return self
        else:
            return self.squeeze(axis=to_squeeze)

    def squeeze(self, axis=None):
        """
        Remove all dims which have length=1.
        'axis' selects a subset of the single-dimensional axes.
        """
        # print 'input axis:', axis
        sh = self.data.shape
        if axis is None:
            axis = [a for i, a in enumerate(self.axes_names) if sh[i] == 1]
        else:
            assert self.has_axes(axis)
            ssh = np.array([sh[self.get_axis_id(a)] for a in axis])
            if (ssh != 1).all():
                raise Exception('Subset axes to squeeze (%s) '
                                'are not all one-length: %s'
                                % (str(axis), str(ssh)))

        axis_ids = tuple(self.get_axis_id(a) for a in axis)
        # print 'axes to squeeze', axis
        # print 'ids :', axis_ids

        # select axes to keep:
        axes_names = [a for a in self.axes_names if a not in axis]

        axes_domains = dict((a, self.axes_domains[a]) for a in axes_names)

        if parse_version(np.__version__) >= parse_version('1.7'):
            data = self.data.squeeze(axis=axis_ids)
        else:
            sm = [':'] * len(sh)
            for i in axis_ids:
                sm[i] = '0'
            # print 'sm:', sm
            data = eval('self.data[%s]' % ','.join(sm))

        return xndarray(data, axes_names, axes_domains,
                        self.value_label, self.meta_data)

    def descrip(self):
        """ Return a printable string describing the cuboid.
        """
        s = ''
        s += '* shape : %s\n' % str(self.data.shape)
        s += '* dtype : %s\n' % str(self.data.dtype)
        s += '* orientation: %s\n' % str(self.axes_names)
        s += '* value label: %s\n' % self.value_label
        s += '* axes domains:\n'
        for dname in self.axes_names:
            dvalues = self.axes_domains[dname]
            if isinstance(dvalues, np.ndarray) and \
                    not np.issubdtype(dvalues.dtype, str) and \
                    not np.issubdtype(dvalues.dtype, unicode) and \
                    not dvalues.dtype == bool and \
                    not dvalues.dtype == np.bool_ and \
                    len(dvalues) > 1:
                delta = np.diff(dvalues)
                if (delta == delta[0]).all():
                    s += "  '%s': " % dname + 'arange(%s,%s,%s)\n' \
                        % (str(dvalues[0]), str(dvalues[-1]), str(delta[0]))
                else:
                    s += "  '%s': " % dname + pformat(dvalues) + '\n'
            else:
                s += "  '%s': " % dname + pformat(dvalues) + '\n'
        return s.rstrip('\n')

    def _html_table_headers(self, row_axes, col_axes):
        """
        Build table row and column headers corresponding to axes in *row_axes* and
        *col_axes* respectively. Headers comprises axis names and domain values.

        Return:
             tuple(list of str, list of str)
          -> tuple(list of html code for the row header (without <tr> tags),
                   list of html code for the col header (with <tr> tags))
        """
        dsh = self.get_dshape()
        nb_blank_cols = len(row_axes) * 2  # nb of blank cols preprended to
        # each line of the column header
        nb_rows = int(np.prod([dsh[a] for a in row_axes]))
        nb_cols = int(np.prod([dsh[a] for a in col_axes]))
        # col header
        if nb_blank_cols > 0:
            blank_cells = ['']
            blank_cells_attrs = [{'colspan': str(nb_blank_cols)}]
        else:
            blank_cells = []
            blank_cells_attrs = []
        col_header = []
        nb_repets = 1
        span = nb_cols
        for a in col_axes:
            dom = [str(v)
                   for v in self.get_domain(a)]  # TODO: better dv format
            span /= len(dom)
            # row showing the axis label
            col_header.append(html_list_to_row(blank_cells + [a], 'h',
                                               blank_cells_attrs +
                                               [{'colspan': nb_cols}]))
            # row showing domain values
            col_header.append(html_list_to_row(blank_cells + dom * nb_repets, 'h',
                                               blank_cells_attrs +
                                               [{'colspan': str(span)}] *
                                               len(dom) * nb_repets))
            nb_repets *= len(dom)

        # row header
        # initialization of all rows because row filling wont be sequential:
        row_header = [[] for i in range(nb_rows)]
        nb_repets = 1
        span = nb_rows
        for a in row_axes:
            # 1st row contains all axis labels:
            row_header[0].append(html_cell(html_div(a, {'class': 'rotate'}), 'h',
                                           {'rowspan': nb_rows}))

            # dispatch domain values across corresponding rows:
            dom = [str(v)
                   for v in self.get_domain(a)]  # TODO: better dv format
            span /= len(dom)
            for idv, dv in enumerate(dom * nb_repets):
                row_header[
                    idv * span].append(html_cell(dv, 'h', {'rowspan': span}))

            nb_repets *= len(dom)

        return [''.join(r) for r in row_header], col_header

    def to_html_table(self, row_axes, col_axes, inner_axes, cell_format='txt',
                      plot_dir=None, rel_plot_dir=None, plot_fig_prefix='xarray_',
                      plot_style='image', plot_args=None, tooltip=False,
                      border=None):
        """
        Render the array as an html table whose column headers correspond
        to domain values and axis names defined by *col_axes*, row headers
        defined by *row_axes* and inner cell axes defined by *inner_axes*
        Data within a cell can be render as text or as a plot figure (image
        files are produced)

        Args:
            -

        Return:
            html code (str)
        """
        import matplotlib.pyplot as plt
        import pyhrf.plot as pplot

        plot_dir = plot_dir or pyhrf.get_tmp_path()

        outer_axes = row_axes + col_axes
        plot_args = plot_args or {}

        norm = plt.normalize(self.min(), self.max())

        def protect_html_fn(fn):
            base, ext = op.splitext(fn)
            return base.replace('.', '-') + ext

        def format_cell(slice_info, cell_val):
            attrs = {}
            if tooltip:
                stooltip = '|| '.join(['%s=%s' % (a, str(s))
                                       for a, s in zip(outer_axes, slice_info)])
                attrs['title'] = stooltip

            if cell_format == 'txt':
                return html_cell(str(cell_val), attrs=attrs)
            elif cell_format == 'plot':
                # Forge figure filename
                suffix = '_'.join(['%s_%s' % (a, str(s))
                                   for a, s in zip(outer_axes, slice_info)])
                fig_fn = op.join(plot_dir, plot_fig_prefix + suffix + '.png')
                fig_fn = protect_html_fn(fig_fn)
                # Render figure
                # TODO: expose these parameters
                plt.figure(figsize=(4, 3), dpi=40)
                if plot_style == 'image':
                    pplot.plot_cub_as_image(cell_val, norm=norm, **plot_args)
                else:
                    pplot.plot_cub_as_curve(cell_val, **plot_args)
                plt.savefig(fig_fn)
                # Create html code
                html_fig_fn = fig_fn
                if rel_plot_dir is not None:
                    html_fig_fn = op.join(rel_plot_dir, op.basename(fig_fn))
                return html_cell(html_img(html_fig_fn), attrs=attrs)
            else:
                raise Exception('Wrong plot_style "%s"' % plot_style)

        logger.info('Generate html table headers ...')
        row_header, col_header = self._html_table_headers(row_axes, col_axes)
        logger.info('Convert xarray to tree ...')
        cell_vals = tree_items(self.to_tree(row_axes + col_axes, inner_axes))

        dsh = self.get_dshape()
        nb_cols = int(np.prod([dsh[a] for a in col_axes]))
        content = []
        logger.info('Generate cells ...')
        for i, r in enumerate(row_header):
            # at each row, concatenate row header + table content
            content += html_row(r + ''.join([format_cell(*cell_vals.next())
                                             for c in range(nb_cols)]))
        return html_table(''.join(col_header + content), border=border)

    def _combine_domains(self, axes):
        """
        Hierarchically combine domains of axes
        """
        def stack_dvalues(prev, x):
            if 0:
                print 'stack_dvalues ...'
                print 'prev:', prev
                print 'x:', x
            res = prev + [list(np.tile(x, len(prev) == 0 or len(prev[-1])))]
            if 0:
                print 'result:', res
                print ''
            return res
        return reduce(stack_dvalues, [self.axes_domains[a] for a in axes], [])

    def to_latex(self, row_axes=None, col_axes=None, inner_axes=None,
                 inner_separator=' | ', header_styles=None, hval_fmt=None,
                 val_fmt='%1.2f', col_align=None):

        def multicol(n, s, align='c'):
            return '\\multicolumn{%d}{%s}{%s}' % (n, align, s)

        def multirow(n, s, position='*'):
            return '\\multirow{%d}{%s}{%s}' % (n, position, s)

        if hval_fmt is None:
            hval_fmt = {}
            for a in self.axes_names:
                if np.issubdtype(self.axes_domains[a].dtype, float):
                    fmt = '%1.1f'
                elif np.issubdtype(self.axes_domains[a].dtype, int):
                    fmt = '%d'
                elif np.issubdtype(self.axes_domains[a].dtype, str):
                    fmt = '%s'
                else:
                    fmt = val_fmt

                hval_fmt[a] = fmt

        if row_axes is None:
            row_axes = []
        if col_axes is None:
            col_axes = []

        if inner_axes is None:
            inner_axes = []

        if header_styles is None:
            header_styles = {}

        hstyles = dict((a, ['split']) for a in self.axes_names)
        hstyles.update(header_styles)
        logger.debug('hstyles: %s', str(hstyles))

        if col_align is None:
            col_align = {}

        calign = dict((a, 'c') for a in self.axes_names)
        calign.update(col_align)

        def fmt_val(val):
            if isinstance(val, str):
                return val
            else:
                return val_fmt % val

        def fmt_hval(val, axis):
            if isinstance(val, str):
                return val
            else:
                return hval_fmt[axis] % val

        def fmt_hcell(axis, val):
            style = hstyles[axis]
            # print 'fmt_hcell for axis %s (style=%s), val=' %(axis,style), val
            if ('split' in style) or ('hide_name' in style):
                r = fmt_hval(val, axis)
            elif 'join' in style:
                r = a + "=" + fmt_hval(val, axis)

            if 'vertical' in style:
                return r'\rotatebox{90}{{%s}}' % r
            else:
                return r

        table_line_end = '\\\\\n'

        c_to_print = self.reorient(row_axes + col_axes + inner_axes)
        dsh = c_to_print.get_dshape()
        data = c_to_print.data.reshape(int(np.prod([dsh[a] for a in row_axes])),
                                       int(np.prod([dsh[a]
                                                    for a in col_axes])),
                                       int(np.prod([dsh[a] for a in inner_axes])))

        #data = np.atleast_2d(self.unstack(row_axes + col_axes, inner_axes).data)

        nb_rows, nb_cols = data.shape[:2]

        doms_for_col_header = self._combine_domains(col_axes)
        # print 'doms_for_col_header:'
        # print doms_for_col_header

        if len(row_axes) > 0:
            doms_for_row_header = self._combine_domains(row_axes)
            # print 'doms_for_row_header:'
            # print doms_for_row_header

            assert len(doms_for_row_header[-1]) == nb_rows
            row_header = [[] for i in range(nb_rows)]

            for a, d in zip(row_axes, doms_for_row_header):
                # print 'a:', a
                # print 'd:', d
                if 'split' in hstyles[a]:
                    row_header[0].append(multirow(nb_rows, a))
                    for i in range(1, nb_rows):
                        row_header[i].append('')

                len_d = len(d)
                # print 'row_header @start:'
                # print row_header
                # print ''
                # print 'loop over lines ...'
                j = 0
                for i in range(nb_rows):
                    # print 'i=', i
                    if i % (nb_rows / len_d) == 0:
                        if nb_rows / len_d > 1:
                            row_header[i].append(multirow(nb_rows / len_d,
                                                          fmt_hcell(a, d[j])))
                        else:
                            row_header[i].append(fmt_hcell(a, d[j]))
                        j += 1
                    else:
                        row_header[i].extend([''])
                    # print 'row_header:'
                    # print row_header
                    # print ''

            for irow, row in enumerate(row_header):
                # print 'row:', row
                row_header[irow] = ' & '.join(row) + ' & '
        else:
            row_header = [''] * nb_rows

        assert len(doms_for_col_header[-1]) == nb_cols
        hprefix = ''.join([['&', '&&']['split' in hstyles[a]]
                           for a in row_axes])
        header = ''
        for a, d in zip(col_axes[:-1], doms_for_col_header[:-1]):
            # print 'format header for axis', a
            if 'split' in hstyles[a]:
                header += hprefix + multicol(nb_cols, a) + table_line_end
            header += hprefix + ' & '.join([multicol(nb_cols / len(d),
                                                     fmt_hcell(a, e))
                                            for e in d]) + table_line_end

        a, d = col_axes[-1], doms_for_col_header[-1]
        if 'split' in hstyles[col_axes[-1]]:
            header += hprefix + \
                multicol(nb_cols, col_axes[-1]) + table_line_end
        header += hprefix + ' & '.join([fmt_hcell(a, e) for e in d]) + \
            table_line_end

        # nb_cols = len(self.axes_domains.get(col_axes[-1], '1'))
        all_col_align = [calign[col_axes[-1]]] * nb_cols
        for a, d in zip(col_axes[:-1], doms_for_col_header[:-1]):
            len_d = len(d)
            # print 'a:', a
            # print 'd:', d
            # print 'range(nb_cols/len_d, nb_cols, len_d):', \
            #   range(nb_cols/len_d-1, nb_cols, nb_cols/len_d-1)
            for i in range(nb_cols / len_d - 1, nb_cols, nb_cols / len_d):
                all_col_align[i] = calign[a]

        table_align = ' '.join([['c', 'c c']['split' in hstyles[a]]
                                for a in row_axes] +
                               ['|'] * (len(row_axes) > 0) +
                               all_col_align)

        s = '\\begin{tabular}{%s}\n' % table_align
        s += header

        def fmt_cell(c):
            if isinstance(c, xndarray):
                return inner_separator.join(map(fmt_val, c.data))
            elif isinstance(c, np.ndarray):
                return inner_separator.join(map(fmt_val, c))
            else:  # scalar
                return fmt_val(c)
        s += table_line_end.join([lh + " & ".join(map(fmt_cell, l))
                                  for lh, l in zip(row_header, data)]) + \
            table_line_end
        s += '\\end{tabular}'
        return s

    def get_orientation(self):
        return self.axes_names

    def set_MRI_orientation(self):
        """ Set orientation to sagittal,coronal,axial,[time|iteration|condition]
        Priority for the 4th axis: time > condition > iteration.
        The remaining axes are sorted in alphatical order
        """

        if self.has_axes(MRI3Daxes):
            orientation = MRI3Daxes[:]
            if self.has_axis('time'):
                orientation += ['time']
            if self.has_axis('iteration'):
                orientation += ['iteration']
            if self.has_axis('condition'):
                orientation += ['condition']

            orientation += sorted(set(self.axes_names).difference(orientation))

            self.set_orientation(orientation)

    def set_orientation(self, axes):
        """ Set the cuboid orientation (inplace) according to input axes labels
        """
        if debug:
            logger.debug('set_orientation ...')
            logger.debug('%s -> %s', str(self.axes_names), str(axes))

        if set(axes) != set(self.axes_names):
            raise Exception('Required orientation %s does not contain '
                            'all axes %s' % (str(axes), str(self.axes_names)))

        if axes == self.axes_names:  # already in the asked orientation
            return

        for i, axis in enumerate(axes):
            logger.debug('Rolling axis %s, cur pos=%d -> dest pos=%d',
                         axis, self.axes_names.index(axis), i)
            logger.debug('Shape: %s', str(self.data.shape))
            cur_i = self.axes_names.index(axis)
            self.data = np.rollaxis(self.data, cur_i, i)
            self.axes_names.pop(cur_i)
            self.axes_names.insert(i, axis)
            logger.debug('After rolling. Shape: %s, new axes: %s',
                         str(self.data.shape), str(self.axes_names))
            logger.debug('')

        self.axes_ids = dict([(a, i) for i, a in enumerate(self.axes_names)])

    def reorient(self, orientation):
        """ Return a cuboid with new orientation.
        If cuboid is already in the right orientation, then return the current
        cuboid. Else, create a new one.
        """
        if orientation == self.axes_names:
            return self
        else:
            new_c = self.copy()
            new_c.set_orientation(orientation)
            return new_c

    def cexpand(self, cmask, axis, dest=None):
        """ Same as expand but mask is a cuboid

        TODO: + unit test
        """
        return self.expand(cmask.data, axis, cmask.axes_names,
                           cmask.axes_domains, dest=dest)

    def expand(self, mask, axis, target_axes=None,
               target_domains=None, dest=None, do_checks=True, m=None):
        """ Create a new xndarray instance (or store into an existing 'dest'
        cuboid) where 'axis' is expanded and values are mapped according to
        'mask'.
        'target_axes' is a list of the names of the new axes replacing 'axis'.
        'target_domains' is a dict of domains for the new axes.

        Example:
        >>> import numpy as np
        >>> from pyhrf.ndarray import xndarray
        >>> c_flat = xndarray(np.arange(2*6).reshape(2,6).astype(np.int64), \
                              ['condition', 'voxel'], \
                              {'condition' : ['audio','video']})
        >>> print c_flat.descrip()  # doctest: +NORMALIZE_WHITESPACE
        * shape : (2, 6)
        * dtype : int64
        * orientation: ['condition', 'voxel']
        * value label: value
        * axes domains:
          'condition': array(['audio', 'video'],
              dtype='|S5')
          'voxel': arange(0,5,1)
        >>> mask = np.zeros((4,4,4), dtype=int)
        >>> mask[:3,:2,0] = 1
        >>> c_expanded = c_flat.expand(mask, 'voxel', ['x','y','z'])
        >>> print c_expanded.descrip()  # doctest: +NORMALIZE_WHITESPACE
        * shape : (2, 4, 4, 4)
        * dtype : int64
        * orientation: ['condition', 'x', 'y', 'z']
        * value label: value
        * axes domains:
          'condition': array(['audio', 'video'],
              dtype='|S5')
          'x': arange(0,3,1)
          'y': arange(0,3,1)
          'z': arange(0,3,1)
        """
        logger.debug('expand ... mask: %s -> region size=%d, '
                     'axis: %s, target_axes: %s, target_domains: %s',
                     str(mask.shape), mask.sum(dtype=int), axis,
                     str(target_axes), str(target_domains))

        if do_checks:
            if not ((mask.min() == 0 and mask.max() == 1) or
                    (mask.min() == 1 and mask.max() == 1) or
                    (mask.min() == 0 and mask.max() == 0)):
                raise Exception("Input mask is not binary (%s)"
                                % str(np.unique(mask)))

            if axis not in self.axes_names:
                raise Exception('Axes %s not found in cuboid\'s axes.' % axis)

        if target_axes is None:
            target_axes = []
            for i in xrange(mask.ndim):
                d = 0
                while 'dim%d' % d in self.axes_names + target_axes:
                    d += 1
                target_axes.append('dim%d' % d)
        else:
            target_axes = target_axes

        logger.debug('target_axes: %s', str(target_axes))

        if do_checks and len(target_axes) != 1 and \
                len(set(target_axes).intersection(self.axes_names)) != 0:
            # if len(target_axes) == 1 & target_axes[0] already in current axes
            #    -> OK, axis is mapped to itself.
            raise Exception('Error while expanding xndarray, intersection btwn'
                            ' targer axes (%s) and current axes (%s) is '
                            'not empty.'
                            % (str(target_axes), str(self.axes_names)))

        assert len(target_axes) == mask.ndim

        if target_domains is None:
            target_domains = dict([(a, range(mask.shape[i]))
                                   for i, a in enumerate(target_axes)])

        assert set(target_domains.keys()).issubset(target_axes)
        # target_domains = [target_domains.get(a,range(mask.shape[i])) \
        #                       for i,a in enumerate(target_axes)]

        logger.debug('target_domains: %s', str(target_domains))
        assert len(target_domains) == len(target_axes)

        flat_axis_idx = self.get_axis_id(axis)
        if dest is not None:
            dest_data = dest.data
        else:
            dest_data = None
        new_data = expand_array_in_mask(self.data, mask,
                                        flat_axis=flat_axis_idx, dest=dest_data,
                                        m=m)
        new_axes = self.axes_names[:flat_axis_idx] + target_axes + \
            self.axes_names[flat_axis_idx + 1:]

        new_domains = self.axes_domains.copy()
        new_domains.pop(axis)
        new_domains.update(target_domains)

        # new_domains = dict([(new_axes[i], new_domains[i]) \
        #                         for i in xrange(len(new_axes))])

        return xndarray(new_data, new_axes, new_domains, self.value_label,
                        meta_data=self.meta_data)

    def map_onto(self, xmapping):
        """
        Reshape the array by mapping the axis corresponding to
        xmapping.value_label onto the shape of xmapping.
        Args:
            - xmapping (xndarray): array whose attribute value_label
                                   matches an axis of the current array

        Return:
            - a new array (xndarray) where values from the current array
              have been mapped according to xmapping

        Example:
        >>> from pyhrf.ndarray import xndarray
        >>> import numpy as np
        >>> # data with a region axis:
        >>> data = xndarray(np.arange(2*4).reshape(2,4).T * .1,    \
                            ['time', 'region'],               \
                            {'time':np.arange(4)*.5, 'region':[2, 6]})
        >>> data
        axes: ['time', 'region'], array([[ 0. ,  0.4],
               [ 0.1,  0.5],
               [ 0.2,  0.6],
               [ 0.3,  0.7]])
        >>> # 2D spatial mask of regions:
        >>> region_map = xndarray(np.array([[2,2,2,6], [6,6,6,0], [6,6,0,0]]), \
                                  ['x','y'], value_label='region')
        >>> # expand region-specific data into region mask
        >>> # (duplicate values)
        >>> data.map_onto(region_map)
        axes: ['x', 'y', 'time'], array([[[ 0. ,  0.1,  0.2,  0.3],
                [ 0. ,  0.1,  0.2,  0.3],
                [ 0. ,  0.1,  0.2,  0.3],
                [ 0.4,  0.5,  0.6,  0.7]],
        <BLANKLINE>
               [[ 0.4,  0.5,  0.6,  0.7],
                [ 0.4,  0.5,  0.6,  0.7],
                [ 0.4,  0.5,  0.6,  0.7],
                [ 0. ,  0. ,  0. ,  0. ]],
        <BLANKLINE>
               [[ 0.4,  0.5,  0.6,  0.7],
                [ 0.4,  0.5,  0.6,  0.7],
                [ 0. ,  0. ,  0. ,  0. ],
                [ 0. ,  0. ,  0. ,  0. ]]])
        """
        mapped_axis = xmapping.value_label
        if not self.has_axis(mapped_axis):
            raise ArrayMappingError('Value label "%s" of xmapping not found '
                                    'in array axes (%s)'
                                    % (mapped_axis,
                                       ', '.join(self.axes_names)))

        if not set(xmapping.data.flat).issuperset(self.get_domain(mapped_axis)):
            raise ArrayMappingError('Domain of axis "%s" to be mapped is not a '
                                    'subset of values in the mapping array.'
                                    % mapped_axis)
        dest = None
        for mval in self.get_domain(mapped_axis):
            sub_a = self.sub_cuboid(**{mapped_axis: mval})
            sub_mapping = self.xndarray_like(
                xmapping, data=xmapping.data == mval)
            rsub_a = sub_a.repeat(sub_mapping.sum(), '__mapped_axis__')
            dest = rsub_a.cexpand(sub_mapping, '__mapped_axis__', dest=dest)
        return dest

    def flatten(self, mask, axes, new_axis):
        """ flatten cudoid.

        TODO: +unit test
        """
        if not set(axes).issubset(self.axes_names):
            raise Exception('Axes to flat (%s) are not a subset of '
                            'current axes (%s)'
                            % (str(axes), str(self.axes_names)))

        m = np.where(mask)
        # print 'mask:', mask.sum()
        # print 'flat_data:', flat_data.shape
        sm = [':'] * self.data.ndim
        for i, a in enumerate(axes):
            j = self.get_axis_id(a)
            sm[j] = 'm[%d]' % i

        new_axes = [a for a in self.axes_names if a not in axes]
        new_axes.insert(self.get_axis_id(axes[0]), new_axis)

        flat_data = eval('self.data[%s]' % ','.join(sm))

        new_domains = dict((a, self.axes_domains[a])
                           for a in new_axes if a != new_axis)

        return xndarray(flat_data, new_axes, new_domains, self.value_label,
                        self.meta_data)

    def explode_a(self, mask, axes, new_axis):
        """
        Explode array according to given n-ary *mask* so that *axes* are flatten
        into *new_axis*.

        Args:
            - mask (numpy.ndarray[int]): n-ary mask that defines "regions" used
                                         to split data
            - axes (list of str): list of axes in the current object that are
                                  mapped onto the mask
            - new_axis (str): target flat axis

        Return:
            dict of xndarray that maps a mask value to a xndarray.

        """
        return dict((i, self.flatten(mask == i, axes, new_axis))
                    for i in np.unique(mask))

    def explode(self, cmask, new_axis='position'):
        """
        Explode array according to the given n-ary *mask* so that axes matchin
        those of *mask* are flatten into *new_axis*.

        Args:
            - mask (xndarray[int]): n-ary mask that defines "regions" used
                                    to split data
            - new_axis (str): target flat axis

        Return:
            dict of xndarray that maps a mask value to a xndarray.
        """
        return dict((i, self.flatten(cmask.data == i, cmask.axes_names, new_axis))
                    for i in np.unique(cmask.data))

    def cflatten(self, cmask, new_axis):
        return self.flatten(cmask.data, cmask.axes_names, new_axis)

    def split(self, axis):
        """ Split a cuboid along given axis.
        Return an OrderedDict of cuboids.
        """
        if axis not in self.axes_names:
            raise Exception('Axis %s not found. Available axes: %s'
                            % (axis, self.axes_names))

        return OrderedDict((dv, self.sub_cuboid(**{axis: dv}))
                           for dv in self.axes_domains[axis])

    def unstack(self, outer_axes, inner_axes):
        """
        Unstack the array along outer_axes and produce a xndarray of xndarrays

        Args:
            - outer_axes (list of str): list of axis names defining the target
                                        unstacked xndarray
            - inner_axes (list of str): list of axis names of any given sub-array
                                        of the target unstacked xndarray

        Return:
            xndarray object

        Example:
        >>> from pyhrf.ndarray import xndarray
        >>> import numpy as np
        >>> c = xndarray(np.arange(4).reshape(2,2), axes_names=['a1','ia'], \
                         axes_domains={'a1':['out_dv1', 'out_dv2'], \
                                       'ia':['in_dv1', 'in_dv2']})
        >>> c.unstack(['a1'], ['ia'])
        axes: ['a1'], array([axes: ['ia'], array([0, 1]), axes: ['ia'], array([2, 3])], dtype=object)
        """

        def _unstack(xa, ans):
            if len(ans) > 0:
                return [_unstack(suba, ans[1:])
                        for suba in xa.split(ans[0]).itervalues()]
            else:
                return xa
        xarray = self.reorient(outer_axes + inner_axes)
        return xndarray(_unstack(xarray, outer_axes), axes_names=outer_axes,
                        axes_domains=dict((a, self.axes_domains[a])
                                          for a in outer_axes))

    def to_tree(self, level_axes, leaf_axes):
        """
        Convert nested dictionary mapping where each key is a domain value
        and each leaf is an array or a scalar value if *leaf_axes* is empty.

        Return:
            OrderedDict such as:
                {dv_axis1 : {dv_axis2 : {... : xndarray|scalar_type}

        Example:
        >>> from pyhrf.ndarray import xndarray
        >>> import numpy as np
        >>> c = xndarray(np.arange(4).reshape(2,2), axes_names=['a1','ia'], \
                         axes_domains={'a1':['out_dv1', 'out_dv2'], \
                                       'ia':['in_dv1', 'in_dv2']})
        >>> c.to_tree(['a1'], ['ia'])
        OrderedDict([('out_dv1', axes: ['ia'], array([0, 1])), ('out_dv2', axes: ['ia'], array([2, 3]))])
        """
        def _to_tree(xa, ans):
            if len(ans) != len(leaf_axes):
                return OrderedDict((dv, _to_tree(suba, ans[1:]))
                                   for dv, suba in xa.split(ans[0]).iteritems())
            else:
                return xa

        xarray = self.reorient(level_axes + leaf_axes)
        return _to_tree(xarray, xarray.axes_names)

    def _format_dvalues(self, axis):
        if (np.diff(self.axes_domains[axis]) == 1).all():
            ndigits = len(str(self.axes_domains[axis].max()))
            return [str(d).zfill(ndigits) for d in self.axes_domains[axis]]
        else:
            return self.axes_domains[axis]

    def get_ndims(self):
        return self.data.ndim

    def get_axes_domains(self):
        """ Return domains associated to axes as a dict (axis_name:domain array)
        """
        return self.axes_domains

    def get_axis_name(self, axis_id):
        """ Return the name of an axis from the given index 'axis_id'.
        """
        if isinstance(axis_id, str):
            if axis_id in self.axes_names:
                return axis_id
            else:
                return None
        assert np.isreal(axis_id) and np.round(axis_id) == axis_id
        if axis_id >= 0 and axis_id < self.get_ndims():
            return self.axes_names[axis_id]
        else:
            return None

    def sub_cuboid(self, orientation=None, **kwargs):
        """ Return a sub cuboid. 'kwargs' allows argument in the form:
        axis=slice_value.
        """
        if not set(kwargs.keys()).issubset(self.axes_names):
            raise Exception('Axes to slice (%s) mismatch current axes (%s)'
                            % (','.join(kwargs.keys()),
                               ','.join(self.axes_names)))
        if orientation is not None:
            assert set(orientation) == set(self.axes_names).difference(kwargs)

        new_kwargs = {}
        for axis, i in kwargs.iteritems():
            new_kwargs[axis] = self.get_domain_idx(axis, i)

        return self.sub_cuboid_from_slices(orientation, **new_kwargs)

    def sub_cuboid_from_slices(self, orientation=None, **kwargs):
        """ Return a sub cuboid. 'kwargs' allows argument in the form:
        axis=slice_index.
        """
        mask = [':'] * self.data.ndim
        for axis, i in kwargs.iteritems():
            mask[self.axes_names.index(axis)] = str(i)
        # print 'mask:', mask
        sub_data = eval('self.data[%s]' % ','.join(mask))

        sub_domains = self.axes_domains.copy()
        for a in kwargs:
            sub_domains.pop(a)

        sub_axes = [a for a in self.axes_names if a not in kwargs]

        if orientation is None:
            orientation = sub_axes

        assert set(orientation) == set(self.axes_names).difference(kwargs)

        if self.meta_data is not None:
            meta_data = (self.meta_data[0].copy(), self.meta_data[1].copy())
        else:
            meta_data = None

        if np.isscalar(sub_data):
            return sub_data

        sub_c = xndarray(sub_data, sub_axes, sub_domains, self.value_label,
                         meta_data)

        # set orientation
        sub_c.set_orientation(orientation)

        return sub_c

    def fill(self, c):
        """
        """
        sm = []
        for a in self.axes_names:
            if c.has_axis(a):
                sm.append(':')
            else:
                sm.append('np.newaxis')
        self.data[:] = eval('c.data[%s]' % ','.join(sm))
        return self

    def copy(self, copy_meta_data=False):
        """ Return copy of the current cuboid. Domains are copied with a shallow
        dictionnary copy.
        """
        if self.meta_data is not None:
            if copy_meta_data:
                new_meta_data = (self.meta_data[0].copy(),
                                 self.meta_data[1].copy())
            else:
                new_meta_data = self.meta_data
        else:
            new_meta_data = None
        return xndarray(self.data.copy(), self.axes_names[:],
                        self.axes_domains.copy(),
                        self.value_label, new_meta_data)

    def get_new_axis_name(self):
        """ Return an axis label not already in use. Format is: dim%d
        """
        i = 0
        while 'dim%d' % i in self.axes_names:
            i += 1

        return 'dim%d' % i

    def get_domain_idx(self, axis, value):
        """ Get slice index from domain value for axis 'axis'.
        """
        if debug:
            print 'get_domain_idx ... axis=%s, value=%s' % (axis, str(value))
            print 'axes_domains:', self.axes_domains
            print 'self.axes_domains[axis]:', self.axes_domains[axis]
            print 'self.axes_domains[axis] == value:', \
                self.axes_domains[axis] == value
            print type(np.where(self.axes_domains[axis] == value)[0][0])
        where_res = np.where(self.axes_domains[axis] == value)[0]
        if len(where_res) == 0:
            raise Exception('Value "%s" not found in domain of axis "%s"'
                            % (str(value), self.get_axis_name(axis)))

        return where_res[0]

    def has_axis(self, axis):
        return axis in self.axes_names

    def has_axes(self, axes):
        return set(axes).issubset(self.axes_names)

    def __eq__(self, other):
        """ Return true if other cuboid contains the same data.
        TODO: should it be irrespective of the orientation ?
        """

        for k, v in self.axes_domains.iteritems():
            if k not in other.axes_domains:
                # print '%s not in domains %s'
                # %(str(k),str(other.axes_domains))
                return False
            if isinstance(v, np.ndarray):
                if not np.allclose(v, other.axes_domains[k]).all():
                    # print 'numpy array differ for %s' %str(k)
                    return False
            else:
                if v != other.axes_domains[k]:
                    # print 'domain value differ for %s' %str(k)
                    return False

        return (self.data == other.data).all() and \
            self.axes_names == other.axes_names and \
            self.value_label == other.value_label

    def astype(self, t):
        c = self.copy()
        c.data = c.data.astype(t)
        return c

    def descrip_shape(self):
        sh = self.data.shape
        axes_info = ['%s:%d' % (a, sh[i])
                     for i, a in enumerate(self.axes_names)]
        return '(' + ','.join(axes_info) + ')'

    def get_voxel_size(self, axis):
        """ Return the size of a voxel along 'axis', only if meta data is
        available.
        """
        assert axis in MRI3Daxes
        if self.meta_data is not None:
            affine, header = self.meta_data
            return header['pixdim'][1:4][MRI3Daxes.index(axis)]
        else:
            raise Exception('xndarray does not have any meta data to get'
                            'voxel size')

    def get_dshape(self):
        """
        Return the shape of the array as dict mapping an axis name to the
        corresponding size
        """
        return dict(zip(self.axes_names, self.data.shape))

    def _prepare_for_operation(self, op_name, c):
        """ Make some checks before performing an operation between self and c.
        If c is a cuboid then it must have the same axes,
        same domains and value labels.
        If c is np.ndarray, it must have the same shape as self.data

        Return a cuboid in the same orientation as self.

        TODO: allow broadcasting
        """
        if isinstance(c, np.ndarray):
            if self.data.shape != c.shape:
                raise Exception('Cannot %s cuboid and ndarray. Shape of self'
                                ' %s different from ndarray shape %s.'
                                % (op_name, str(self.data.shape),
                                   str(c.shape)))

            c = xndarray.xndarray_like(self, data=c)
        elif np.isscalar(c):
            class Dummy:
                def __init__(self, val):
                    self.data = val
            return Dummy(c)

        if set(self.axes_names) != set(c.axes_names):
            raise Exception('Cannot %s cuboids with different axes' % op_name)

        # TODO: check axes domains ...
        if self.axes_names != c.axes_names:
            c = c.reorient(self.axes_names)

        for i, a in enumerate(self.axes_names):
            if self.data.shape[i] != c.data.shape[i]:

                raise Exception('Cannot %s cuboids, shape mismatch.'
                                ' self has shape: %s and operand has '
                                ' shape: %s'
                                % (op_name, self.descrip_shape(),
                                   c.descrip_shape()))

        return c

    def add(self, c, dest=None):
        c = self._prepare_for_operation('add', c)
        if dest is None:
            return xndarray(self.data + c.data, self.axes_names, self.axes_domains,
                            self.value_label, self.meta_data)
        else:
            dest.data += c.data
            return dest

    def multiply(self, c, dest=None):
        c = self._prepare_for_operation('multiply', c)
        if dest is None:
            return xndarray(self.data * c.data, self.axes_names, self.axes_domains,
                            self.value_label, self.meta_data)
        else:
            dest.data *= c.data
            return dest

    def divide(self, c, dest=None):
        c = self._prepare_for_operation('divide', c)
        if dest is None:
            return xndarray(self.data / c.data, self.axes_names, self.axes_domains,
                            self.value_label, self.meta_data)
        else:
            dest.data /= c.data
            return dest

    def substract(self, c, dest=None):
        c = self._prepare_for_operation('substract', c)
        if dest is None:
            return xndarray(self.data - c.data, self.axes_names, self.axes_domains,
                            self.value_label, self.meta_data)
        else:
            dest.data -= c.data
            return dest

    def __iadd__(self, c):
        return self.add(c, dest=self)

    def __add__(self, c):
        return self.add(c)

    def __radd__(self, c):
        return self.add(c)

    def __imul__(self, c):
        return self.multiply(c, dest=self)

    def __rmul__(self, c):
        return self.multiply(c)

    def __mul__(self, c):
        return self.multiply(c)

    def __idiv__(self, c):
        return self.divide(c, dest=self)

    def __div__(self, c):
        return self.divide(c)

    def __rdiv__(self, c):
        if np.isscalar(c):
            return xndarray_like(self, data=c / self.data)
        else:
            c = self._prepare_for_operation(c)
            return c.divide(self)

    def __isub__(self, c):
        return self.substract(c, dest=self)

    def __sub__(self, c):
        return self.substract(c)

    def __rsub__(self, c):
        return (self * -1).add(c)

    def __pow__(self, c):
        if np.isscalar(c):
            return xndarray_like(self, data=self.data ** c)
        else:
            raise NotImplementedError(
                'Broadcast for pow operation not available')

    def min(self, axis=None):
        if axis is None:
            return self.data.min()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            m = self.data.min(axis=ia)
            if np.isscalar(m):
                return m
            else:
                return xndarray(m, an, ad, self.value_label, self.meta_data)

    def max(self, axis=None):
        if axis is None:
            return self.data.max()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            m = self.data.max(axis=ia)
            if np.isscalar(m):
                return m
            else:
                return xndarray(m, an, ad, self.value_label, self.meta_data)

    def ptp(self, axis=None):
        if axis is None:
            return self.data.ptp()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            r = self.data.ptp(axis=ia)
            if np.isscalar(r):
                return r
            else:
                return xndarray(r, an, ad, self.value_label, self.meta_data)

    def mean(self, axis=None):
        if axis is None:
            return self.data.mean()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            return xndarray(self.data.mean(axis=ia), an, ad,
                            self.value_label, self.meta_data)

    def std(self, axis=None):
        if axis is None:
            return self.data.mean()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            return xndarray(self.data.std(axis=ia), an, ad,
                            self.value_label + '_std', self.meta_data)

    def var(self, axis=None):
        if axis is None:
            return self.data.mean()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            return xndarray(self.data.var(axis=ia), an, ad,
                            self.value_label + '_var', self.meta_data)

    def sum(self, axis=None):
        if axis is None:
            return self.data.sum()
        else:
            ia = self.get_axis_id(axis)
            na = self.get_axis_name(axis)
            if ia is None or na is None:
                raise Exception('Wrong axis %s (%d available axes: %s)'
                                % (str(axis), self.ndim,
                                   ','.join(self.axes_names)))

            an = self.axes_names[:ia] + self.axes_names[ia + 1:]
            ad = self.axes_domains.copy()
            ad.pop(na)
            return xndarray(self.data.sum(axis=ia), an, ad,
                            self.value_label, self.meta_data)

    def rescale_values(self, v_min=0., v_max=1., axis=None):
        if axis is not None:
            axis = self.get_axis_id(axis)
        new_data = rescale_values(self.data, v_min, v_max, axis)

        return xndarray(new_data, self.axes_names, self.axes_domains,
                        self.value_label, self.meta_data)

    def get_extra_info(self, fmt='dict'):
        from pyhrf.xmlio import to_xml

        info = {
            'axes_names': self.axes_names,
            'axes_domains': self.axes_domains,
            'value_label': self.value_label,
        }
        if fmt == 'dict':
            return info
        elif fmt == 'xml':
            return to_xml(info)

    def save(self, file_name, meta_data=None, set_MRI_orientation=False):
        """ Save cuboid to a file. Supported format: Nifti1.
        'meta_data' shoud be a 2-elements tuple:
        (affine matrix, Nifti1Header instance). If provided, the meta_data
        attribute of the cuboid is ignored.
        All extra axis information is stored as an extension.

        """

        from pyhrf.xmlio import from_xml, to_xml, DeprecatedXMLFormatException

        logger.debug('xndarray.save(%s)', file_name)
        ext = op.splitext(file_name)[1]
        c_to_save = self

        if has_ext(file_name, 'nii'):
            if set_MRI_orientation:
                self.set_MRI_orientation()

            extra_info = c_to_save.get_extra_info()

            logger.info('Extra info:')
            logger.info(extra_info)

            from nibabel.nifti1 import Nifti1Extension, Nifti1Header, \
                Nifti1Image, extension_codes

            if self.data.ndim > 7:
                raise Exception("Nifti format can not handle more than "
                                "7 dims. xndarray has %d dims"
                                % self.data.ndim)

            if meta_data is None:
                if c_to_save.meta_data is not None:
                    affine, header = c_to_save.meta_data
                else:
                    affine = np.eye(4)
                    header = Nifti1Header()
            else:
                affine, header = meta_data
                header = Nifti1Header()

            header = header.copy()

            ecodes = header.extensions.get_codes()
            if extension_codes['comment'] in ecodes:
                ic = ecodes.index(extension_codes['comment'])
                econtent = header.extensions[ic].get_content()
                # Check if existing extension can be safely overwritten
                try:
                    prev_extra_info = from_xml(econtent)
                except DeprecatedXMLFormatException, e:
                    from pyhrf.xmliobak import from_xml as from_xml_bak
                    prev_extra_info = from_xml_bak(econtent)
                except Exception:
                    raise IOError("Cannot safely overwrite Extension in "
                                  "Header. It already has a 'comment' "
                                  "extension "
                                  "with the following content:\n" +
                                  str(econtent))

                if not isinstance(prev_extra_info, dict) and \
                        prev_extra_info.has_key('axes_names'):
                    raise IOError("Cannot safely overwrite Extension in "
                                  "Header. It already has a readable "
                                  "XML 'comment' extension, but it's "
                                  "not related to xndarray meta info.")

                # Checks are OK, remove the previous extension:
                prev_ext = header.extensions.pop(ic).get_content()
                logger.debug('Extensions after popping previous ext:')
                logger.debug(str(header.extensions))
            else:
                prev_ext = ""

            ext_str = to_xml(extra_info)
            if len(ext_str) < len(prev_ext):
                extra_info['dummy'] = '#' * (len(prev_ext) - len(ext_str))
                ext_str = to_xml(extra_info)

            logger.debug('Length of extension string: %s', len(ext_str))
            logger.debug('Extension: \n %s', ext_str)
            e = Nifti1Extension('comment', ext_str)

            header.extensions.append(e)

            logger.info('Extensions after appending new ext:')

            header.set_data_dtype(c_to_save.data.dtype)
            i = Nifti1Image(c_to_save.data, affine, header=header)
            i.update_header()

            logger.debug('Save Nifti image to %s ...', file_name)
            i.to_filename(file_name)
            logger.debug('Save Nifti image, done!')
        elif ext == '.csv':
            np.savetxt(file_name, c_to_save.data, fmt="%12.9G")
        elif has_ext(file_name, 'gii'):
            from pyhrf.tools._io import write_texture
            logger.info('Save Gifti image (dim=%d) ...', c_to_save.get_ndims())
            logger.info('axes names: %s', str(c_to_save.axes_names))

            if c_to_save.get_ndims() == 1:
                write_texture(c_to_save.data, file_name,
                              meta_data=c_to_save.get_extra_info(fmt='xml'))
            elif c_to_save.get_ndims() == 2 and \
                    c_to_save.has_axes(['voxel', 'time']):
                saxes = ['voxel'] + \
                    list(set(c_to_save.axes_names).difference(['voxel']))
                # make sure spatial axis is the 1st one
                logger.info('reorient as %s', str(saxes))
                c_to_save = c_to_save.reorient(saxes)
                write_texture(c_to_save.data, file_name,
                              meta_data=c_to_save.get_extra_info(fmt='xml'))
            else:
                if c_to_save.has_axes(['voxel']):
                    saxes = list(set(c_to_save.axes_names).difference(['voxel',
                                                                       'time']))
                    split_c = c_to_save.split(saxes[0])
                    for dval, subc in split_c.iteritems():
                        subc.save(add_suffix(file_name,
                                             '_%s_%s' % (saxes[0], str(dval))))
                else:
                    write_texture(c_to_save.data, file_name,
                                  meta_data=c_to_save.get_extra_info(fmt='xml'))
        else:
            raise Exception('Unsupported file format (ext: "%s")' % ext)

    @staticmethod
    def load(file_name):
        """ Load cuboid from file. Supported format: nifti1.
        Extra axis information is retrieved from a nifti extension if available.
        If it's not available, label the axes as:
        (sagittal, coronal, axial[, time]).

        TODO: gifti.
        """
        from pyhrf.xmlio import from_xml, DeprecatedXMLFormatException
        has_deprecated_xml_header = False

        logger.info('xndarray.load(%s)', file_name)
        ext = op.splitext(file_name)[1]
        if ext == '.nii' or \
                (ext == '.gz' and op.splitext(file_name[:-3])[1] == '.nii') or \
                ext == '.img' or \
                (ext == '.gz' and op.splitext(file_name[:-3])[1] == '.img'):
            import nibabel
            i = nibabel.load(file_name)
            h = i.get_header()
            data = np.array(i.get_data())  # avoid memmapping
            cuboid_info = {}
            cuboid_info['axes_names'] = MRI4Daxes[:min(4, data.ndim)]
            # TODO: fill spatial domains with position in mm, and time axis
            #      according TR value.
            cuboid_info['value_label'] = 'intensity'

            # print 'extensions:', h.extensions
            if hasattr(h, 'extensions') and len(h.extensions) > 0:
                ecodes = h.extensions.get_codes()
                # print 'ecodes:', ecodes
                ccode = nibabel.nifti1.extension_codes['comment']
                if ccode in ecodes:
                    ic = ecodes.index(ccode)
                    ext_content = h.extensions[ic].get_content()
                    try:
                        cuboid_info = from_xml(ext_content)
                    except DeprecatedXMLFormatException, e:
                        has_deprecated_xml_header = True
                        try:
                            from pyhrf.xmliobak import from_xml as from_xml_bak
                            cuboid_info = from_xml_bak(ext_content)
                        except:
                            # Can't load xml -> ignore it
                            # TODO: warn?
                            cuboid_info = {}
                    except Exception, e:
                        raise IOError('Extension for xndarray meta info can not '
                                      'be read from "comment" extension. '
                                      'Content is:\n%s\n Exception was:\n%s'
                                      % (ext_content, str(e)))

                    cuboid_info.pop('dummy', None)

            logger.info('Extra info loaded from extension:')
            logger.info(cuboid_info)

            meta_data = (i.get_affine(), h)
            cuboid_info = dict((str(k), v) for k, v in cuboid_info.iteritems())
            data[np.where(np.isnan(data))] = 0
            a = xndarray(data, meta_data=meta_data, **cuboid_info)
            a.has_deprecated_xml_header = has_deprecated_xml_header
            return a
        elif ext == '.gii' or \
                (ext == '.gz' and op.splitext(file_name[:-3])[1] == '.gii'):
            from pyhrf.tools._io import read_texture
            data, gii = read_texture(file_name)
            md = gii.get_metadata().get_metadata()
            # print 'meta data loaded from gii:'
            # print md
            if md.has_key('pyhrf_cuboid_data'):
                cuboid_info = from_xml(md['pyhrf_cuboid_data'])
            else:
                cuboid_info = {}
            return xndarray(data, **cuboid_info)
        else:
            raise Exception('Unrecognised file format (ext: %s)' % ext)


def xndarray_like(c, data=None):
    return xndarray.xndarray_like(c, data)


def stack_cuboids(c_list, axis, domain=None, axis_pos='first'):
    """ Stack xndarray instances in list 'c_list' along a new axis label 'axis'.
    If 'domain' (numpy array or list) is provided, it is associated to the
    new axis.
    All cuboids in 'c_list' must have the same orientation and domains.
    'axis_pos' defines the position of the new axis: either 'first' or 'last'.

    Example:
    >>> import numpy as np
    >>> from pyhrf.ndarray import xndarray, stack_cuboids
    >>> c1 = xndarray(np.arange(4*3).reshape(4,3), ['x','y'])
    >>> c1
    axes: ['x', 'y'], array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])
    >>> c2 = xndarray(np.arange(4*3).reshape(4,3)*2, ['x','y'])
    >>> c2
    axes: ['x', 'y'], array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16],
           [18, 20, 22]])
    >>> c_stacked = stack_cuboids([c1,c2], 'stack_axis', ['c1','c2'])
    >>> print c_stacked.descrip()  # doctest: +NORMALIZE_WHITESPACE
    * shape : (2, 4, 3)
    * dtype : int64
    * orientation: ['stack_axis', 'x', 'y']
    * value label: value
    * axes domains:
      'stack_axis': array(['c1', 'c2'],
          dtype='|S2')
      'x': arange(0,3,1)
      'y': arange(0,2,1)


    TODO: enable broadcasting (?)
    """
    assert isinstance(axis, str)
    size = len(c_list)
    cub0 = c_list[0]
    axes = cub0.axes_names
    # print 'axes:', axes
    sh = (size,) + cub0.data.shape
    stackedData = np.zeros(sh, cub0.data.dtype)
    newDomains = cub0.axes_domains.copy()
    if domain is not None:
        newDomains[axis] = domain
    targetCub = xndarray(stackedData,
                         axes_names=[axis] + cub0.axes_names,
                         axes_domains=newDomains, value_label=cub0.value_label)
    # print 'c_list', c_list, c_list[0], c_list[1]
    for i, cuboid in enumerate(c_list):
        if debug:
            print 'targetCub.data[i] :', targetCub.data[i].shape
        if debug:
            print 'cuboid', cuboid.descrip()
        # print 'number:', i
        # print 'cuboid.axes:', cuboid.axes_names
        if axes != cuboid.axes_names:
            raise Exception('%dth cuboid in list does not match other cuboids'
                            'found axes: %s, should be: %s'
                            % (i, str(cuboid.axes_names), str(axes)))

        # TODO: better use numpy stacking functions (faster)
        targetCub.data[i] = cuboid.data

    if axis_pos == 'last':
        targetCub.roll(axis)

    return targetCub


def expand_array_in_mask(flat_data, mask, flat_axis=0, dest=None, m=None):
    """ Map the flat_axis of flat_data onto the region within mask.
    flat_data is then reshaped so that flat_axis is replaced with mask.shape

    *m* is the result of np.where(mask) -> can be passed to speed up if already
    done before

    Example 1
    >>> a = np.array([1,2,3])
    >>> m = np.array([[0,1,0], [0,1,1]] )
    >>> expand_array_in_mask(a,m)
    array([[0, 1, 0],
           [0, 2, 3]])

    Example 2
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> m = np.array([[0,1,0], [0,1,1]] )
    >>> expand_array_in_mask(a,m,flat_axis=1)
    array([[[0, 1, 0],
            [0, 2, 3]],
    <BLANKLINE>
           [[0, 4, 0],
            [0, 5, 6]]])

    """
    flat_sh = flat_data.shape
    mask_sh = mask.shape
    target_shape = flat_sh[:flat_axis] + mask_sh + flat_sh[flat_axis + 1:]

    logger.debug('expand_array_in_mask ... %s -> %s', str(flat_sh),
                 str(target_shape))
    if dest is None:
        dest = np.zeros(target_shape, dtype=flat_data.dtype)

    assert dest.shape == target_shape
    assert dest.dtype == flat_data.dtype

    if m is None:
        m = np.where(mask)

    n = len(m[0])
    if n != flat_data.shape[flat_axis]:
        raise Exception('Nb positions in mask (%d) different from length of '
                        'flat_data (%d)' % (n, flat_data.shape[flat_axis]))
    sm = ([':'] * len(flat_sh[:flat_axis]) +
          ['m[%d]' % i for i in range(len(mask_sh))] +
          [':'] * len(flat_sh[flat_axis + 1:]))
    exec('dest[%s] = flat_data' % ','.join(sm))

    return dest


def merge(arrays, mask, axis, fill_value=0):
    """
    Merge the given arrays into a single array according to the given
    mask, with the given axis being mapped to those of mask.
    Assume that arrays[id] corresponds to mask==id and that all arrays are in
    the same orientation.

    Arg:
        - arrays (dict of xndarrays):
        - mask (xndarray): defines the mapping between the flat axis in the
                           arrays to merge and the target expanded axes.
        - axis (str): flat axis for the
    """
    if len(arrays) == 0:
        raise Exception('Empty list of arrays')

    dest_c = None
    for i, a in arrays.iteritems():
        dest_c = a.expand(mask.data == i, axis, mask.axes_names,
                          dest=dest_c, do_checks=False)

    return dest_c


def tree_to_xndarray(tree, level_labels=None):
    """
    Stack all arrays within input tree into a single array.

    Args:
        - tree (dict): nested dictionnaries of xndarray objects.
                       Each level of the tree correspond to a target axis,
                       each key of the tree correspond to an element of the
                       domain associated to that axis.
        - level_labels (list of str): axis labels corresponding to each level
                       of the tree

    Return:
        xndarray object

    Example:
    >>> from pyhrf.ndarray import xndarray, tree_to_xndarray
    >>> d = { 1 : { .1 : xndarray([1,2], axes_names=['inner_axis']), \
                    .2 : xndarray([3,4], axes_names=['inner_axis']), \
                  },                                                 \
              2 : { .1 : xndarray([1,2], axes_names=['inner_axis']), \
                    .2 : xndarray([3,4], axes_names=['inner_axis']), \
                  }                                                  \
            }
    >>> tree_to_xndarray(d, ['level_1', 'level_2'])
    axes: ['level_1', 'level_2', 'inner_axis'], array([[[1, 2],
            [3, 4]],
    <BLANKLINE>
           [[1, 2],
            [3, 4]]])
    """
    tree_depth = len(treeBranches(tree).next())
    level_labels = level_labels or ['axis_%d' % i for i in xrange(tree_depth)]
    assert len(level_labels) == tree_depth

    def _reduce(node, level):
        if isinstance(node, xndarray):  # leaf node
            return node
        else:
            domain = sorted(node.keys())
            return stack_cuboids([_reduce(node[c], level + 1) for c in domain],
                                 level_labels[level], domain)
    return _reduce(tree, level=0)


from pyhrf.tools import add_suffix


def split_and_save(cub, axes, fn, meta_data=None, set_MRI_orientation=False,
                   output_dir=None, format_dvalues=False):

    if len(axes) == 0:
        # print 'fn:', fn
        # print 'sub_c:'
        # print cub.descrip()
        if output_dir is None:
            output_dir = op.dirname(fn)

        cub.save(op.join(output_dir, op.basename(fn)), meta_data=meta_data,
                 set_MRI_orientation=set_MRI_orientation)
        return

    axis = axes[0]

    split_res = cub.split(axis)
    split_dvals = np.array(split_res.keys())

    if format_dvalues and np.issubdtype(split_dvals.dtype, np.number) and \
            (np.diff(split_dvals) == 1).all():
        ndigits = len(str(max(split_dvals)))
        if min(split_dvals) == 0:
            o = 1
        else:
            o = 0

        split_dvals = [str(d + o).zfill(ndigits) for d in split_dvals]

    for dvalue, sub_c in zip(split_dvals, split_res.values()):
        if axis != 'error':
            newfn = add_suffix(fn, '_%s_%s' % (axis, str(dvalue)))
        else:
            if dvalue == 'error':
                newfn = add_suffix(fn, '_error')
            else:
                newfn = fn[:]

        split_and_save(sub_c, axes[1:], newfn, meta_data=meta_data,
                       set_MRI_orientation=set_MRI_orientation,
                       output_dir=output_dir)


# Html stuffs

# Define the style of cells in html table output, especially the "rotate" class to
# show axis labels in vertical orientation
# This should be put in the <head> section of the final html doc
ndarray_html_style = """<!-- td {
  border-collapse:collapse;
  border: 1px black solid;
}
.rotate {
     -moz-transform: rotate(-90.0deg);  /* FF3.5+ */
       -o-transform: rotate(-90.0deg);  /* Opera 10.5 */
  -webkit-transform: rotate(-90.0deg);  /* Saf3.1+, Chrome */
             filter:  progid:DXImageTransform.Microsoft.BasicImage(rotation=0.083);  /* IE6,IE7 */
         -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0.083)"; /* IE8 */
} -->"""
