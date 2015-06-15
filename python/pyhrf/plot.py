# -*- coding: utf-8 -*-
import os
import numpy as np
from pyhrf.tools import rebin

fig_orientations = {
    'axial' : ['coronal','sagittal'],
    'sagittal' : ['axial','coronal'],
    'coronal' : ['axial','sagittal'],
    }


def mix_cmap(img1, cmap1, img2, cmap2, norm1=None, norm2=None, blend_r=.5):
    import matplotlib.pyplot as plt

    assert img1.shape == img2.shape
    if norm1 is None:
        norm1 = plt.Normalize()
    cols1 = cmap1(norm1(img1))
    if norm2 is None:
        norm2 = plt.Normalize()
    cols2 = cmap2(norm2(img2))

    blend_r = np.zeros(cols1.shape) + blend_r
    if isinstance(img1, np.ma.MaskedArray):
        blend_r[np.where(img1.mask)] = 0.

    if isinstance(img2, np.ma.MaskedArray):
        blend_r[np.where(img1.mask)] = 1.

    mixed_cols = blend_r * cols1 + (1.-blend_r) * cols2

    # mixed_cols = np.zeros_like(cols1)
    # for i in xrange(cols1.shape[0]):
    #     for j in xrange(cols1.shape[1]):
    #         br = blend_r[i,j]
    #         mixed_cols[i,j] = (br)*cols1[i,j] + (1.-br)*cols2[i,j]
    #         if 0:
    #             print 'mixing [%d,%d] -> %1.2f * %s + %1.2f * %s = %s' \
    #                 %(i,j, blend_r, str(cols1[i,j]), 1.-blend_r, str(cols2[i,j]),
    #                   str(mixed_cols[i,j]))
    return mixed_cols


def plot_func_slice(func_slice_data, anatomy=None, parcellation=None,
                    parcel_col='white',
                    parcels_line_width=2.5, func_cmap=None,
                    func_norm=None, anat_norm=None, func_mask=None,
                    highlighted_parcels_col=None,
                    highlighted_parcels_line_width=2.5,
                    resolution=None, crop_extension=None, blend=.5):

    import matplotlib.pyplot as plt

    if highlighted_parcels_col is None:
        highlighted_parcels_col = {}

    if func_cmap is None:
        func_cmap = plt.get_cmap('seismic')

    if anatomy is not None:
        if func_slice_data.shape != anatomy.shape:
            func_slice_data = rebin(func_slice_data, anatomy.shape)

    if parcellation is not None:
        parcellation = parcellation.astype(int)

        if anatomy is not None:
            if parcellation.shape != anatomy.shape:
                parcellation = rebin(parcellation, anatomy.shape)

        fmask = (parcellation>0).astype(bool)
    else:
        fmask = np.ones_like(func_slice_data).astype(bool)

    if func_mask is not None:
        if anatomy is not None:
            func_mask = rebin(func_mask.astype(bool), anatomy.shape)
         #print 'func_mask:', func_mask.sum()
        fmask = np.bitwise_and(fmask, func_mask)

    if crop_extension is not None:
        from pyhrf.tools import crop_array
        func_slice_data = crop_array(func_slice_data, fmask, crop_extension)
        if anatomy is not None:
            m_tmp = np.where(fmask!=0)
            mask_anat_tmp = np.zeros_like(anatomy)
            e = crop_extension
            start_i = m_tmp[0].min()-e-1
            start_j = m_tmp[1].min()-e-1
            if start_i < 0 or start_j < 0:
                raise Exception('Anatomy is not large enough to extend view.')

            mask_anat_tmp[start_i:m_tmp[0].max()+e,start_j:m_tmp[1].max()+e] = 1
            anatomy = crop_array(anatomy, mask_anat_tmp)
            # print 'func_slice_data:', func_slice_data.shape
            # print 'anatomy:', anatomy.shape
        fmask = crop_array(fmask, fmask, crop_extension)

    func_masked = np.ma.masked_where(fmask==0, func_slice_data)


    anat_cmap = plt.get_cmap('gray')

    plt.figure()
    plt.hold(True)

    if anatomy is not None:
        ax = plt.imshow(mix_cmap(func_masked, func_cmap,
                                 anatomy, anat_cmap,
                                 norm1=func_norm,
                                 norm2=anat_norm, blend_r=blend),
                        interpolation='nearest')
    else:
        ax = plt.imshow(func_masked, cmap=func_cmap,
                        norm=func_norm,
                        interpolation='nearest')
    ax.get_axes().set_axis_off()


    if resolution is not None:
        ratio = resolution[0] / resolution[1]
        print 'Set aspect ratio to:', ratio
        ax.get_axes().set_aspect(ratio)

    if parcellation is not None:
        labs = np.unique(parcellation)

        # nr, nc = parcel_rebin.shape
        # extent = [-0.5, nc-0.5, -0.5, nr-0.5]

        for lab in labs: #[484,427,540]: #
            if lab != 0:
                col = highlighted_parcels_col.get(lab, parcel_col)
                if lab in highlighted_parcels_col:
                    lw = highlighted_parcels_line_width
                else:
                    lw = parcels_line_width
                plt.contour((parcellation==lab).astype(int), 1,
                            colors=[col], linewidths=lw, alpha=.6,)
                #           extent=extent)



def plot_anat_parcel_func_fusion(anat, func, parcel, parcel_col='white',
                                 parcels_line_width=.5, func_cmap=None,
                                 func_norm=None, anat_norm=None, func_mask=None,
                                 highlighted_parcels_col=None,
                                 highlighted_parcels_line_width=1.5):

    import matplotlib.pyplot as plt

    if highlighted_parcels_col is None:
        highlighted_parcels_col = {}

    if func_cmap is None:
        func_cmap = plt.get_cmap('jet')

    parcel = parcel.astype(int)
    parcel_rebin = rebin(parcel, anat.shape)

    #parcel_rebin_ma = np.ma.masked_where(parcel_rebin==0, parcel_rebin)
    func_rebin = rebin(func, anat.shape)
    fmask = (parcel_rebin>0).astype(bool)

    if func_mask is not None:
        func_mask_rebin = rebin(func_mask.astype(bool), anat.shape)
        #print 'func_mask:', func_mask_rebin.sum()
        fmask = np.bitwise_and(fmask, func_mask_rebin)
    func_rebin_ma = np.ma.masked_where(fmask==0, func_rebin)

    anat_cmap = plt.get_cmap('gray')

    plt.figure()
    plt.hold(True)
    if 0:
        #TODO: better color mixing
        ax = plt.imshow(anat, cmap=anat_cmap, norm=anat_norm,
                        interpolation='nearest')

        plt.imshow(func_rebin_ma, interpolation='nearest',
                   alpha=.5, cmap=func_cmap, norm=func_norm, axes=ax)

    if 1:
        ax = plt.imshow(mix_cmap(func_rebin_ma, func_cmap,
                                 anat, anat_cmap,
                                 norm1=func_norm,
                                 norm2=anat_norm, blend_r=.5),
                        interpolation='nearest')
        ax.get_axes().set_axis_off()

    if 0:
        plt.imshow(func_rebin_ma*anat, interpolation='nearest',
                   cmap=func_cmap, norm=func_norm)
    if 0:
        ax = plt.imshow(func_rebin_ma, interpolation='nearest',
                        cmap=func_cmap, norm=func_norm)
        plt.colorbar()
        plt.imshow(anat, cmap=plt.get_cmap('gray'), alpha=.65,
                   interpolation='nearest', axes=ax)

    if 0:
        plt.imshow(mix_cmap(func_rebin_ma, func_cmap, anat, anat_cmap),
                   interpolation='nearest')

    labs = np.unique(parcel)

    # nr, nc = parcel_rebin.shape
    # extent = [-0.5, nc-0.5, -0.5, nr-0.5]

    for lab in labs: #[484,427,540]: #
        if lab != 0:
            col = highlighted_parcels_col.get(lab, parcel_col)
            if lab in highlighted_parcels_col:
                lw = highlighted_parcels_line_width
                alpha = .5
            else:
                lw = parcels_line_width
                alpha = .2
            plt.contour((parcel_rebin==lab).astype(int), 1,
                        colors=[col], linewidths=lw, alpha=alpha,)
    #extent=extent)


def autocrop(img_fn):
    """ Remove extra background within figure (inplace).
    Use ImageMagick (convert) """
    os.system('convert %s -trim %s' %(img_fn,img_fn))

def rotate(img_fn, angle):
    """ Rotate figure (inplace). Use ImageMagick (convert) """
    os.system('convert %s -rotate %f %s' %(img_fn,angle,img_fn))

def flip(img_fn, direction='horizontal'):
    """ Mirror the figure (inplace). Use ImageMagick (convert)
    'horizontal' direction -> use -flop.
    'vertical' direction -> use -flip.
    """
    cmd_arg = {'horizontal':'flop','vertical':'flip'}
    os.system('convert %s -%s %s' %(img_fn,cmd_arg[direction],img_fn))


def plot_palette(cmap, norm=None, fontsize=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if norm is None:
        norm = plt.Normalize()
    fig = plt.figure()
    ax1 = fig.add_axes([0.05, 0.05, 0.05, .9])
    colbar = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                              norm=norm,
                              orientation='vertical')
    if fontsize is not None:
        ticklabels = plt.getp(colbar.ax, 'yticklabels')
        plt.setp(ticklabels, 'color', 'k', fontsize=fontsize)
    return colbar

def set_ticks_fontsize(fontsize, colbar=None):
    """ Change the fontsize of the tick labels for the current figure.
    If colorbar (Colorbar instance) is provided then change the fontsize of its
    tick labels as well.
    """
    import matplotlib.pyplot as plt

    yticklabels = plt.getp(plt.gca(), 'yticklabels')
    plt.setp(yticklabels, 'color', 'k', fontsize=fontsize)
    xticklabels = plt.getp(plt.gca(), 'xticklabels')
    plt.setp(xticklabels, 'color', 'k', fontsize=fontsize)

    if colbar is not None:
        ticklabels = plt.getp(colbar.ax, 'yticklabels')
        plt.setp(ticklabels, 'color', 'k', fontsize=fontsize)


def set_xticklabels(labels, positions=None, rotation=None):
    """ Set ticks labels of the xaxis in the current figure to *labels*
    If positions is provided then enforce tick position.
    """
    import matplotlib.pyplot as plt

    if positions is None:
        positions,_ = plt.xticks()

    if len(positions) != len(labels):
        raise Exception('Wrong number of labels (%d), ticks require %s' \
                        %(len(labels), len(positions)))

    plt.xticks(positions, labels)
    labs = plt.gca().get_xticklabels()
    plt.setp(labs, rotation=rotation)
#     ca = plt.gca().axis
#     plt.setp(labels, 'color', 'k', fontsize=fontsize)



def plot_spm_mip(img_fn, mip_fn):
    pass

import scipy.stats
def plot_gaussian_pdf(bins, m, v, prop=None, plotArgs={}):
    import matplotlib.pyplot as plt

    prop = prop or 1.
    if v > 1e-2:
        pdf = scipy.stats.norm.pdf(bins, m, v)
    else:
        if v < 1e-4: v = 1e-4
        bins = np.arange(bins[0], bins[-1], 0.0001)
        pdf = scipy.stats.norm.pdf(bins, m, v)
    pdf *= prop
    plt.plot(bins, pdf, **plotArgs)
    return pdf

# lw = 1.75
# fontsize=18

def plot_gaussian_mixture(values, props=None, color='k', lw=1.75):
    """
    axes of values : (component,class)
    """
    import matplotlib.pyplot as plt

    if props is None:
        props = np.array([.5, .5])
    xMin = (values[0,:] - 5*values[1,:]**.5).min()
    xMax = (values[0,:] + 5*values[1,:]**.5).max()
    nbins = 100.
    bins = np.arange(xMin, xMax, (xMax-xMin)/nbins)
    plotArgs = {'linestyle':'--',
                'linewidth':lw,
                'color':color}
    pdf0 = plot_gaussian_pdf(bins, values[0,0], values[1,0],
                             props[0], plotArgs)
    plotArgs = {'linestyle':'-',
                'linewidth':lw,
                'color':color}
    pdf1 = plot_gaussian_pdf(bins, values[0,1], values[1,1],
                             props[1], plotArgs)

    plotArgs = {'linestyle':'-.',
                'linewidth':lw,
                'color':color}
    plt.plot(bins, pdf0 + pdf1, **plotArgs)

    if plt.ylim()[1] > 2.0:
        plt.ylim(0, 2.)
    #nyticks =  len(plt.yticks())
    #plt.yticks(np.arange(nyticks), ['']*nyticks)




def plot_cub_as_curve(c, colors=None, plot_kwargs=None, legend_prefix='',
                      show_axis_labels=True, show_legend=False, axes=None,
                      axis_label_fontsize=12):
    """
    Plot a cuboid (ndims <= 2) as curve(s).
    If the input is 1D: one single curve.
    If the input is 2D:
       * multiple curves are plotted: one for each domain value on the 1st axis.
       * legends are shown to display which domain value is associated
         to which curve.

    Args:
        - colors (dict <domain value>: <matplotlib color>):
            associate domain values of the 1st axis to color curves
        - plot_kwargs (dict <arg name>:<arg value>):
            dictionary of named argument passed to the plot function
        - legend_prefix (str): prefix to prepend to legend labels.

    Return:
        None
    """
    import matplotlib.pyplot as plt

    def protect_latex_str(s):
        return s.replace('_','\_')
        
    axes = axes or plt.gca()
    colors = colors or {}
    plot_kwargs = plot_kwargs or {}
    if c.get_ndims() == 1:
        dom = c.axes_domains[c.axes_names[0]]
        if np.issubsctype(dom.dtype, str):
            dom = np.arange(len(dom))
        axes.plot(dom, c.data, **plot_kwargs)
        if np.issubsctype(c.axes_domains[c.axes_names[0]], str):
            set_int_tick_labels(axes.xaxis, c.axes_domains[c.axes_names[0]],
                                rotation=30)
    elif c.get_ndims() == 2:
        for val, sub_c in c.split(c.axes_names[0]).iteritems():
            pkwargs = plot_kwargs.copy()
            col = colors.get(val, None)
            if col is not None:
                pkwargs['color'] = col
            pkwargs['label'] = protect_latex_str(legend_prefix + \
                                                 c.axes_names[0] + \
                                                 '=' + str(val))
            plot_cub_as_curve(sub_c, plot_kwargs=pkwargs, axes=axes,
                              show_axis_labels=False)
        if show_legend:
            axes.legend()

    else:
        raise Exception('xndarray has too many dims (%d), expected at most 2' \
                        %c.get_ndims())

    if show_axis_labels:
        if c.get_ndims() == 1:
            axes.set_xlabel(protect_latex_str(c.axes_names[0]), 
                            fontsize=axis_label_fontsize)
        else:
            axes.set_xlabel(protect_latex_str(c.axes_names[1]),
                            fontsize=axis_label_fontsize)
        axes.set_ylabel(protect_latex_str(c.value_label),
                        fontsize=axis_label_fontsize)

def set_int_tick_labels(axis, labels, fontsize=None, rotation=None):
    """
    Redefine labels of visible ticks at integer positions for the given axis.
    """
    # get the tick positions:
    tickPos = axis.get_ticklocs()#.astype(int)
    dvMax = len(labels)
    tLabels = []
    #if debug: print '%%%% tickPos :', tickPos
    for tp in tickPos:
        if tp < 0. or int(tp) != tp or tp >= dvMax:
            tLabels.append('')
        else:
            tLabels.append(labels[int(tp)])
    #if debug: print '%%%% Setting labels:', tLabels
    axis.set_ticklabels(tLabels)
    for label in axis.get_ticklabels():
        if fontsize is not None:
            label.set_fontsize(fontsize)
        if rotation is not None:
            label.set_rotation(rotation)


def plot_cub_as_image(c, cmap=None, norm=None, show_axes=True,
                      show_axis_labels=True, show_tick_labels=True,
                      show_colorbar=False, axes=None):
    import matplotlib.pyplot as plt

    axes = axes or plt.gca()
    data = c.data
    if data.ndim == 1:
        data = data[:,np.newaxis]

    ms = axes.matshow(data, cmap=cmap, norm=norm)#, origin='lower')
    if show_tick_labels:
        set_int_tick_labels(axes.yaxis, c.axes_domains[c.axes_names[0]])
        if len(c.axes_domains) > 1:
            set_int_tick_labels(axes.xaxis, c.axes_domains[c.axes_names[1]])
    else:
        set_int_tick_labels(axes.yaxis,
                            [''] * len(c.axes_domains[c.axes_names[0]]))
        if len(c.axes_domains) > 1:
            set_int_tick_labels(axes.xaxis,
                                [''] * len(c.axes_domains[c.axes_names[1]]))

    if show_axis_labels:
        axes.set_ylabel(c.axes_names[0])
        if len(c.axes_domains) > 1:
            axes.set_xlabel(c.axes_names[1])
    else:
        axes.set_ylabel('')
        axes.set_xlabel('')

    if not show_axes:
        axes.set_axis_off()

    if show_colorbar:
        axes.figure.colorbar(ms)
