import unittest
import os.path as op
import numpy as np

import matplotlib.pyplot as plt

import pyhrf
import pyhrf.plot as pplot
from pyhrf.ndarray import xndarray

from pyhrf.tools import tree_items


def html_row(s):
    return '<tr>%s</tr>' %s

def html_table(s):
    return '<table>%s</table>' %s

def html_img(fn):
    return '<img src="%s">' %fn

def html_cell(s, cell_type='d', attrs=None):
    attrs = attrs or {}
    sattrs = ' '.join(['%s="%s"' %(k,v) for k,v in attrs.items()])
    return '<t%s %s>%s</t%s>' %(cell_type, sattrs, s, cell_type)

def html_list_to_row(l, cell_types, attrs):
    if not isinstance(attrs, (list, tuple)):
        attrs  = [attrs] * len(l)
    else:
        assert len(attrs) == len(l)

    if not isinstance(cell_types, (list, tuple)):
        cell_types  = [cell_types] * len(l)
    else:
        assert len(cell_types) == len(l)

    return html_row(''.join([html_cell(e, t, a) \
                             for e,t,a in zip(l,cell_types,attrs)]))

class xndarray(xndarray):

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
        nb_blank_cols = len(row_axes) * 2 #nb of blank cols preprended to
                                          #each line of the column header
        nb_rows = int(np.prod([dsh[a] for a in row_axes]))
        nb_cols = int(np.prod([dsh[a] for a in col_axes]))
        # col header
        if nb_blank_cols > 0:
            blank_cells = ['']
            blank_cells_attrs = [{'colspan':str(nb_blank_cols)}]
        else:
            blank_cells = []
            blank_cells_attrs = []
        col_header = []
        nb_repets = 1
        span = nb_cols
        for a in col_axes:
            dom = [str(v) for v in self.get_domain(a)] #TODO: better dv format
            span /= len(dom)
            # row showing the axis label
            col_header.append(html_list_to_row(blank_cells + [a], 'h',
                                               blank_cells_attrs + \
                                                [{'colspan':nb_cols}]))
            # row showing domain values
            col_header.append(html_list_to_row(blank_cells + dom * nb_repets, 'h',
                                               blank_cells_attrs +
                                               [{'colspan':str(span)}] * \
                                                 len(dom) * nb_repets))
            nb_repets *= len(dom)

        # row header
        # initialization of all rows because row filling wont be sequential:
        row_header = [[] for i in range(nb_rows)]
        nb_repets = 1
        span = nb_rows
        for a in row_axes:
            # 1st row contains all axis labels:
            row_header[0].append(html_cell(a, 'h', {'rowspan':nb_rows}))

            # dispatch domain values across corresponding rows:
            dom = [str(v) for v in self.get_domain(a)] #TODO: better dv format
            span /= len(dom)
            for idv, dv in enumerate(dom * nb_repets):
                row_header[idv*span].append(html_cell(dv, 'h', {'rowspan':span}))

            nb_repets *= len(dom)

        return [''.join(r) for r in row_header], col_header

    def to_html_table(self, row_axes, col_axes, inner_axes, cell_format='txt',
                      plot_dir=None, plot_fig_prefix='xarray_',
                      plot_style='image', plot_args=None):
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
        plot_dir = plot_dir or pyhrf.get_tmp_path()
        outer_axes = row_axes + col_axes
        plot_args = plot_args or {}

        def format_cell(slice_info, cell_val):
            if cell_format == 'txt':
                return html_cell(str(cell_val))
            elif cell_format == 'plot':
                suffix = '_'.join(['_'.join(e) \
                                   for e in zip(outer_axes, slice_info)])
                fig_fn = op.join(plot_dir, plot_fig_prefix + suffix + '.png' )
                plt.figure()
                if plot_style == 'image':
                    pplot.plot_cub_as_image(cell_val, **plot_args)
                else:
                    pplot.plot_cub_as_curve(cell_val, **plot_args)
                plt.savefig(fig_fn)
                return html_cell(html_img(fig_fn))
            else:
                raise Exception('Wrong plot_style "%s"' %plot_style)


        row_header, col_header = self._html_table_headers(row_axes, col_axes)
        cell_vals = tree_items(self.to_tree(row_axes + col_axes, inner_axes))

        dsh = self.get_dshape()
        nb_cols = int(np.prod([dsh[a] for a in col_axes]))
        content = []
        for i, r in enumerate(row_header):
            content += html_row(r + ''.join([format_cell(*cell_vals.next()) \
                                             for c in range(nb_cols)]))
        return html_table(''.join(col_header + content))

class Test(unittest.TestCase):

    def test_txt_1d_col_axes_only(self):
        a = xndarray([1, 2], ['measure'], {'measure' : ['mon', 'tue']})
        html = a.to_html_table([], ['measure'], [])
        self.assertIsInstance(html, str)
        self.assertEqual(html, '<table><tr><th colspan="2">measure</th></tr>' \
                               '<tr><th colspan="1">mon</th>' \
                               '<th colspan="1">tue</th></tr>' \
                               '<tr><td >1</td><td >2</td></tr></table>')

    def test_txt_1d_row_axes_only(self):
        a = xndarray([1, 2], ['measure'], {'measure' : ['mon', 'tue']})
        html = a.to_html_table(['measure'], [], [])
        self.assertIsInstance(html, str)
        self.assertEqual(html, '<table><tr><th rowspan="2">measure</th>' \
                               '<th rowspan="1">mon</th><td >1</td></tr>' \
                               '<tr><th rowspan="1">tue</th><td >2</td></tr>' \
                               '</table>')

    def test_plot(self):
        sh = (2,3,4)
        a = xndarray(np.arange(np.prod(sh)).reshape(sh),
                     ['day', 'strength', 'position'],
                     {'day' : ['mon', 'tue'],
                      'strength':[0., .5, 1.2],
                      'position':[0, 10, 20, 30]})

        html = a.to_html_table([], ['day'],['strength', 'position'],
                               cell_format='plot', plot_style='image',
                               plot_dir='./tmp', plot_args={'show_colorbar':True})


        expected = '<table><tr><th colspan="2">day</th></tr>'\
                          '<tr><th colspan="1">mon</th>'\
                              '<th colspan="1">tue</th></tr>' \
                          '<tr><td ><img src="./tmp/xarray_day_mon.png"></td>' \
                              '<td ><img src="./tmp/xarray_day_tue.png"></td>' \
                          '</tr></table>'
        self.assertEqual(html, expected)

        f = open('./doc.html', 'w')
        f.write('<!DOCTYPE html><html><body>' + html + '</body></html>')
        f.close()

    def test_table_header(self):
        sh = (2,3,4)
        a = xndarray(np.arange(np.prod(sh)).reshape(sh),
                     ['day', 'strength', 'position'],
                     {'day' : ['mon', 'tue'],
                      'strength':[0., .5, 1.2],
                      'position':[0, 10, 20, 30]})
        rh, ch = a._html_table_headers(['position'], ['day', 'strength'])

        self.assertEqual(''.join(ch), '<tr><th colspan="2"></th>' \
                                      '<th colspan="6">day</th>' \
                                      '</tr>' \
                                      '<tr>' \
                                      '<th colspan="2"></th>' \
                                      '<th colspan="3">mon</th>' \
                                      '<th colspan="3">tue</th>' \
                                      '</tr>' \
                                      '<tr>' \
                                      '<th colspan="2"></th>' \
                                      '<th colspan="6">strength</th>' \
                                      '</tr>' \
                                      '<tr>'
                                      '<th colspan="2"></th>' \
                                      '<th colspan="1">0.0</th>' \
                                      '<th colspan="1">0.5</th>' \
                                      '<th colspan="1">1.2</th>' \
                                      '<th colspan="1">0.0</th>' \
                                      '<th colspan="1">0.5</th>' \
                                      '<th colspan="1">1.2</th>' \
                                      '</tr>')
        self.assertEqual(''.join([html_row(r) for r in rh]),
                          '<tr><th rowspan="4">position</th>' \
                          '<th rowspan="1">0</th>' \
                          '</tr>' \
                          '<tr><th rowspan="1">10</th></tr>' \
                          '<tr><th rowspan="1">20</th></tr>' \
                          '<tr><th rowspan="1">30</th></tr>')


if __name__ == '__main__':
    unittest.main()
