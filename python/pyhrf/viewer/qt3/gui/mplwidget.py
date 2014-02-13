# -*- coding: utf-8 -*-

import sys, os, random, re
from qt import *

import matplotlib, scipy.stats
#matplotlib.use('Agg')
import pylab

import numpy as _N
from numpy import arange
import copy as copyModule
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import get_cmap
import viewModes

debug = 0

class ValuePicker:

    def __init__(self, val):
        self.val = val

    def __call__(self,line,mouseevent):
        print 'line color:', line.get_color(),
        print ' -> ', self.val
        return True,{}

def cmstring_to_mpl_cmap(s):
    lrgb = s.split('#')
    r = [float(v) for v in lrgb[0].split(';')]
    g = [float(v) for v in lrgb[1].split(';')]
    b = [float(v) for v in lrgb[2].split(';')]

    cdict = {'red':(),'green':(),'blue':()}
    for iv in xrange(0,len(r),2):
        cdict['red'] += ((r[iv],r[iv+1],r[iv+1]),)
    for iv in xrange(0,len(b),2):
        cdict['blue'] += ((b[iv],b[iv+1],b[iv+1]),)
    for iv in xrange(0,len(g),2):
        cdict['green'] += ((g[iv],g[iv+1],g[iv+1]),)

    if debug: print ' cdict:', cdict
    return LinearSegmentedColormap('mpl_colmap',cdict,256)


def _matshow(splot, Z, **kwargs):
    '''
    Plot a matrix as an image. Adapted fomr matplotlib 0.91 to be available
    from versions <0.9

    The matrix will be shown the way it would be printed,
    with the first row at the top.  Row and column numbering
    is zero-based.

    Argument:
    Z   anything that can be interpreted as a 2-D array

    kwargs: all are passed to imshow.  matshow sets defaults
    for extent, origin, interpolation, and aspect; use care
    in overriding the extent and origin kwargs, because they
    interact.  (Also, if you want to change them, you probably
    should be using imshow directly in your own version of
    matshow.)

    Returns: an AxesImage instance

    '''
    Z = _N.asarray(Z)
    nr, nc = Z.shape
    extent = [-0.5, nc-0.5, nr-0.5, -0.5]
    kw = {'extent': extent,
          'origin': 'upper',
          'interpolation': 'nearest',
          'aspect': 'equal'}          # (already the imshow default)
    kw.update(kwargs)
    im = splot.imshow(Z, **kw)
    splot.title.set_y(1.05)
    splot.xaxis.tick_top()
    splot.xaxis.set_ticks_position('both')
    splot.set_xlim((0,nc))
    splot.set_ylim((nr,0))
    #splot.xaxis.set_major_locator(MaxNLocator(nbins=9,
    #                                         steps=[1, 2, 5, 10],
    #                                         integer=True))
    #splot.yaxis.set_major_locator(MaxNLocator(nbins=9,
    #                                         steps=[1, 2, 5, 10],
    #                                         integer=True))
    return im


def setTickLabels(axis, labels):
    """
    Redefine labels of visible ticks at integer positions for the given axis.
    """
    # get the tick positions:
    tickPos = axis.get_ticklocs()#.astype(int)
    dvMax = labels.size
    tLabels = []
    #if debug: print '%%%% tickPos :', tickPos
    for tp in tickPos:
        if tp < 0. or int(tp) != tp or tp >= dvMax:
            tLabels.append('')
        else:
            tLabels.append(labels[int(tp)])
    #if debug: print '%%%% Setting labels:', tLabels
    axis.set_ticklabels(tLabels)

def plotNormPDF(ax, bins, m, v):

    if v > 1e-2:
        pdf = scipy.stats.norm.pdf(bins, m, v)
    else:
        if v < 1e-4: v = 1e-4
        bins = arange(bins[0],bins[-1],0.0001)
        pdf = scipy.stats.norm.pdf(bins, m, v)
    ax.plot(bins, pdf)


#TODO : make it handle multiple layers !!
class MatplotlibWidget(FigureCanvas):
    """
    Class handling 1D, 2D, 3D data and displaying them in a canvas as 1D curve
    graphs, 2D images or multiple 2D images.
    """

    MASK_HIDDEN = 0
    MASK_SHOWN = 1
    MASK_ONLY = 2

    def __init__(self, graphMode=None, parent=None, name=None, width=5, height=4,
                 dpi=100, bgColor=None, valueRange=None,
                 maskLabels=None):
        """
        Create matplotlib 'front-end' widget which can render 1D,2D,3D data as
        1D or 2D graphs and handle masks.
        """
        if debug : print '**xndarrayViewRenderer.__init__  ...'
        self.parent = parent

        if graphMode: self.graphMode = graphMode
        else: self.graphMode = viewModes.MODE_2D

        self.fwidth = width
        self.fheight = height
        self.dpi = dpi

        # Will define the range of the colormap associated to values:
        if debug: print 'valueRange :', valueRange
        #valueRange = [0.001, 0.2] #noise var
        #valueRange = [0.001, 0.5] #noise ARp
        #valueRange = [0, 11]
        if valueRange is not None:
            self.norm = Normalize(valueRange[0],
                                  valueRange[1]+_N.abs(valueRange[1])*.01,
                                  clip=True)
            self.backgroundValue = valueRange[0] - 100
        else:
            self.norm = None
            self.backgroundValue = 0 #?
        # Define the range of the colormap associated to the mask:
        # will be used to draw contours of mask
        self.maskCm = None
        self.maskLabels = maskLabels
        if debug: print '######### maskLabels :', maskLabels
        if maskLabels is not None:
            _N.random.seed(1) # ensure we get always the same random colors
            #TODO: put the random seed back in the same state as before!!!
            rndm = _N.random.rand(len(maskLabels),3)
            # black:
            #fixed = _N.zeros((len(maskLabels),3)) + _N.array([0.,0.,0.])
            # green:
            #fixed = _N.zeros((len(maskLabels),3)) + _N.array([0.,1.,0.])
            #white:
            fixed = _N.zeros((len(maskLabels),3)) + _N.array([1.,1.,1.])
            # Create uniform colormaps for every mask label
            # self.maskCm = dict(zip(maskLabels,
            #                       [ListedColormap([ tuple(r) ]) for r in rndm]))
            self.maskCm = dict(zip(maskLabels,
                                   [ListedColormap([tuple(r)]) for r in fixed]))
        self.displayMaskFlag = self.MASK_HIDDEN

        # Set the color of the widget background
        if self.parent:
            bgc = parent.backgroundBrush().color()
            #bgcolor = float(bgc.red())/255.0, float(bgc.green())/255.0, \
            #          float(bgc.blue())/255.0
            bgcolor = "#%02X%02X%02X" % (bgc.red(), bgc.green(), bgc.blue())
        else: bgcolor = 'w'

        # Create the matplotlib figure:
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor=bgcolor, edgecolor=bgcolor)
        # Size of the grid of plots:
        self.subplotsH = 0
        self.subplotsW = 0
        self.axes = None
        self.showAxesFlag = True
        self.showAxesLabels = True

        # Init the parent Canvas:
        FigureCanvas.__init__(self, self.fig)

        # Some QT size stuffs
        self.reparent(parent, QPoint(0, 0))
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Color bar related stuffs:
        self.showColorBar = False
        self.colorBars = None
        # color associated to position where mask=0 :
        self.bgColor = 'w'#QColor('white') if bgColor==None else bgColor

        # Default colormap (~rainbow) : black-blue-green-yellow-red
        self.colorMapString = '0;0;0.5;0.0;0.75;1.0;1.;1.0#' \
                              '0;0;0.5;1;0.75;1;1;0.#'       \
                              '0;0;0.25;1;0.5;0;1;0.'
        self.setColorMapFromString(self.colorMapString)
        self.update()

        # Signal stuffs:
        #self.mpl_connect('button_release_event', self.onRelease)
        self.mpl_connect('motion_notify_event', self.onMove)
        self.mpl_connect('button_press_event', self.onClick)

    def setBackgroundColor(self, color):
        if type(color) == QColor:
            self.bgColor = tuple([c/255. for c in color.getRgb()])
        else:
            self.bgColor = color # assume letter code/ hex color string / ...
                                 # anything supported by matplotlib.colors
                                 #TODO maybe do some checks ?
        if debug:print 'setting over/under color ', self.bgColor
        self.colorMap.set_under(color=self.bgColor)
        self.colorMap.set_over(color=self.bgColor)

    def sizeHint(self):
        w = self.fig.get_figwidth()
        h = self.fig.get_figheight()
        return QSize(w, h)

    def getColorMapString(self):
        return self.colorMapString

    def setColorMapFromString(self, scm):

        if debug : print 'MatplotlibWidget.setColorMapFromString'
        scm = str(scm)
        if debug : print ' recieved scm:', scm
        # Make sure we do not have any negative values :
        subResult = re.subn('(\+[0-9]+(\.\d*)?)','0.',scm)
        if debug and subResult[1] > 0:
            print ' !!! found negative values !!!'
            sys.exit(1)
        scm = subResult[0]
        self.colorMapString = scm
        if debug : print ' negative filtered scm :', scm

        self.colorMap = cmstring_to_mpl_cmap(scm)
        # Set colors corresponding to [minval-1 , minval] to background color:
        self.setBackgroundColor(self.bgColor)

    def setGraphMode(self, m):
        if self.graphMode == m:
            return
        if debug: print 'xndarrayViewRenderer.setGraphMode:',m
        self.graphMode = m
        self.computeFigure()

    def minimumSizeHint(self):
        return QSize(100, 100)


    def setMaskDisplay(self, flag):
        if self.displayMaskFlag == flag:
            return
        self.displayMaskFlag = flag
        self.resetGraph()
        self.refreshFigure()

    def toggleColorbar(self):
        if debug: print "MatplotlibWidget toggleColorbar..."
        self.showColorBar = not self.showColorBar
        if debug: print ' showColorBar = %s' %(['On','Off'][self.showColorBar])
        #if not self.showColorBar:
        self.resetGraph()

        #self.resetGraph()
        self.refreshFigure()
        #self.draw()

    def hideAxes(self):
        if not self.showAxesFlag:
            return
        self.showAxesFlag = False
        for ax in self.axes:
            ax.set_axis_off()
        self.adjustSubplots()
        self.draw()

    def showAxes(self):
        if self.showAxesFlag:
            return
        self.showAxesFlag = True
        for ax in self.axes:
            ax.set_axis_on()
        self.adjustSubplots()
        self.draw()

    def toggleAxesLabels(self):
        self.showAxesLabels = not self.showAxesLabels
        for a in self.axes:
            self.plotAxesLabels(a)
        self.adjustSubplots()
        self.draw()

    def setMaskLabel(self, label):
        if debug: print "MatplotlibWidget setMaskLabel ..."
        self.maskLabel = label
        self._applyMask()
        self.resetGraph()
        self.refreshFigure()


    def _applyMask(self):
        if debug: print "MatplotlibWidget applyMask ..."
        self.maskedValues = self.values.copy()
        self.maskedErrors = None if self.errors==None else self.errors.copy()
        if self.mask != None:
            if debug: print 'self.maskLabel:', self.maskLabel
            if self.maskLabel != 0:
                m = (self.mask != self.maskLabel)
            else:
                m = (self.mask == 0)
            #print 'm :', m
            if debug: print 'backgroundValue :', self.backgroundValue
            self.maskedValues[m] = self.backgroundValue
            if self.errors != None:
                self.maskedErrors[m] = 0

    def updateData(self, cub, mask=None, maskLabel=0, maskName='mask',
                   otherSlices=None):
        if debug:
            print "MatplotlibWidget update data ..."
            print "Got cuboid:"
            print cub.descrip()
            if mask is not None:
                print 'Got mask:', mask.shape
                print mask
        # Retrieve data and make everything be 3D-shaped:
        deltaDims = 3-cub.get_ndims()
        targetSh = cub.data.shape + (1,) * deltaDims
        self.domainNames = cub.axes_names + [None] * deltaDims
        self.domainValues = [cub.axes_domains[a] for a in cub.axes_names] + \
            [None] * deltaDims
        self.values = cub.data.reshape(targetSh)
        self.mask = None if mask==None else mask.reshape(targetSh)
        self.maskName = maskName
        #self.errors = None if cub.errors==None else cub.errors.reshape(targetSh)
        self.errors = None
        self.valueLabel = cub.value_label
        self.maskLabel = maskLabel
        self.otherSlices = otherSlices
        self._applyMask()
        if debug:
            print 'MatplotlibWidget.update: got view:'
            print '**Domains:'
            for i in xrange(len(self.domainNames)):
                if self.domainNames[i]!=None:
                    if _N.isreal(self.domainNames[i][0]) :
                        print '%s : [%d %d]' %(self.domainNames[i],
                                               self.domainValues[i].min(),
                                               self.domainValues[i].max())
                    else:
                        print '%s : [%s ... %s]' %(self.domainNames[i],
                                                   self.domainValues[i][0],
                                                   self.domainValues[i][-1])

            if debug:
                print 'self.maskedValues:', self.maskedValues.shape
                print self.maskedValues[:,:,0]
            #print 'self.values :', self.values.shape
            if self.errors != None: print 'self.errors:', self.maskedErrors.shape

        self.emit(PYSIGNAL('valuesChanged'),(self.getShortDataDescription(),))
        self.computeFigure()

    def getShortDataDescription(self):
        if self.mask!=None:
            if self.maskLabel > 0:
                v = self.values[self.mask == self.maskLabel]
            else: v = self.values[self.mask != 0]
        else:
            v = self.maskedValues
        if v.size > 0:
            descr = '%1.3g(%1.3g)[%1.3g,%1.3g]' %(v.mean(),v.std(),
                                                  v.min(),v.max())
        else: descr = ''
        return descr

    def resetGraph(self):
        if debug : print 'MatplotlibWidget.resetGraph ...'

        if self.colorBars != None:
            if debug: print 'self.colorbars :', self.colorBars
            for cb in self.colorBars:
                self.fig.delaxes(cb.ax)
            self.colorBars = None

        if self.graphMode == viewModes.MODE_HIST:
            if self.axes:
                for a in self.axes:
                    self.fig.delaxes(a)

            ax = self.fig.add_subplot(111)
            ax.zValue = 0
            self.axes = [ax]
            #self.axes.clear()
            self.axes[0].hold(False)

        else:
            nbZVal = self.values.shape[2]
            if debug: print 'nbZVal :', nbZVal
            h = round(nbZVal**.5)
            w = _N.ceil(nbZVal/h)
            if 1 or (h != self.subplotsH or w != self.subplotsW):
                if self.axes:
                    for a in self.axes:
                        self.fig.delaxes(a)
                self.subplotsW = w
                self.subplotsH = h
                self.axes = []
                if debug: print 'add subplots : w=%d, h=%d' %(w,h)
                for i in xrange(1,nbZVal+1):
                   #if debug: print ' add(%d, %d, %d)' %(h,w,i)
                   ax = self.fig.add_subplot(h,w,i)
                   ax.zValue = i-1
                   self.axes.append(ax)

        self.adjustSubplots()

    def adjustSubplots(self):
        if debug: print "adjustSubplots ..."
        if self.graphMode == viewModes.MODE_HIST:
            self.fig.subplots_adjust(left=0.2, right=.95,
                                     bottom=0.2, top=.9,
                                     hspace=0., wspace=0.)
            return

        if self.values.shape[2] == 1:
            if not self.showAxesFlag:
                self.fig.subplots_adjust(left=0.2, right=.8,
                                         bottom=0.2, top=.8,
                                         hspace=0.05, wspace=0.05)
            elif not self.showAxesLabels:
                if self.graphMode == viewModes.MODE_2D:
                    self.fig.subplots_adjust(left=0.05, right=.95,
                                             bottom=0.01, top=.95,
                                             hspace=0., wspace=0.)
                else:
                    self.fig.subplots_adjust(left=0.2, right=.95,
                                             bottom=0.1, top=.95,
                                             hspace=0., wspace=0.)

            else:
                self.fig.subplots_adjust(left=0.1, right=.95,
                                         bottom=0.1, top=.9,
                                         hspace=0.05, wspace=0.05)
        else:
            if not self.showAxesFlag:
                self.fig.subplots_adjust(left=0., right=1.,
                                         bottom=0., top=1.,
                                         hspace=0.05, wspace=0.05)
            elif not self.showAxesLabels:
                self.fig.subplots_adjust(left=0.1, right=.9,
                                         bottom=0.2, top=.9,
                                         hspace=0.7, wspace=0.3)
            else:
                self.fig.subplots_adjust(left=0.1, right=.9,
                                         bottom=0.1, top=.9,
                                         hspace=0.01, wspace=0.7)

    def showMask(self, splot, mask):
        if mask != None:
            nr, nc = mask.shape
            extent = [-0.5, nc-0.5, -0.5, nr-0.5]

            if self.displayMaskFlag == self.MASK_SHOWN:
                labels = _N.unique(mask)
                for il, lab in enumerate(labels):
                    if lab != 0:
                        if self.maskCm!=None : cm = self.maskCm[lab]
                        else : cm = get_cmap('binary')
                        splot.contour((mask==lab).astype(int), 1,
                                      cmap=cm, linewidths=2., extent=extent,alpha=.7)
                                     #cmap=cm, linewidths=1.5, extent=extent,alpha=.7)
            if self.displayMaskFlag == self.MASK_ONLY:
                if self.maskLabel == 0:
                    labels = _N.unique(mask)
                    for il, lab in enumerate(labels):
                        if lab != 0:
                            if self.maskCm != None:
                                cm = self.maskCm[lab]
                            else:
                                cm = get_cmap('binary')
                            ml = (mask==lab).astype(int)
                            print 'contouf of mask label %d -> %d pos' \
                                %(lab, ml.sum())
                            splot.contourf(ml, 1,
                                           cmap=cm, linewidths=1., extent=extent)
                elif (mask==self.maskLabel).sum() > 0:
                    if self.maskCm != None:
                        cm = self.maskCm[_N.where(mask==self.maskLabel)[0]]
                    else:
                        cm = get_cmap('binary')
                    ax.contourf((mask==self.maskLabel).astype(int), 1,
                                cmap=cm, linewidths=1.5, extent=extent)

    def plot1D(self):
        if debug: print 'MatplotlibWidget.computeFigure: MODE_1D'
        di2 = 0
        nbins = 100.

        d1 = self.domainValues[1]
        d0 = self.domainValues[0]
        plotPDF = False
        if d0[0] == 'mean' and len(d0)>1 and d0[1] == 'var':
            plotPDF = True
            if (self.values[1,:] < 10.).all():
                xMin = (self.values[0,:] - 6*self.values[1,:]**.5).min()
                xMax = (self.values[0,:] + 6*self.values[1,:]**.5).max()
            else:
                xMin = (self.values[0,:] - 10*self.values[1,:]**.5).min()
                xMax = (self.values[0,:] + 10*self.values[1,:]**.5).max()
            bins = _N.arange(xMin, xMax, (xMax-xMin)/nbins)

        x = d0 if _N.isreal(d0[0]) else _N.arange(len(d0))

        me = self.maskedErrors.max() if self.errors!=None else 0
        yMax = self.maskedValues.max()+me
        yMin = self.maskedValues.min()-me
        if 1 or self.errors !=None :
            dy = (yMax-yMin)*0.05
            dx = (x.max()-x.min())*0.05
        else: dx,dy = 0,0

        for ax in self.axes:
            ax.hold(True)
            #ax.set_axis_off()
            vSlice = self.maskedValues[:,:,di2]
            if self.errors != None:
                errSlice = self.maskedErrors[:,:,di2]
            for di1 in xrange(self.values.shape[1]):
                if plotPDF:
                    plotNormPDF(ax, bins, vSlice[0,di1], vSlice[1,di1])
                else:
                    print 'di1:',di1
                    print 'domainValues:', self.domainValues
                    if self.domainValues[1] is not None:
                        val = str(self.domainNames[1]) + ' = ' + \
                            str(self.domainValues[1][di1])
                    else:
                        val = 'nothing'
                    ax.plot(x, vSlice[:,di1], picker=ValuePicker(val))
                    #ax.plot(vSlice[:,di1], picker=ValuePicker(val))
                    if not _N.isreal(d0[0]):
                        setTickLabels(self.axes[0].xaxis, d0)
                    if self.errors != None and not _N.allclose(errSlice[:,di1],0.) :
                        ax.errorbar(x, vSlice[:,di1], errSlice[:,di1], fmt=None)
            if not plotPDF:
                ax.set_xlim(x.min()-dx, x.max()+dx)
                ax.set_ylim(yMin-dy, yMax+dy)
            elif ax.get_ylim()[1] > 1.0:
                ax.set_ylim(0, 1.)
            if not self.showAxesFlag:
                ax.set_axis_off()

            self.plotAxesLabels(ax)

            #ax.set_title(self.domainNames[2]+' ' \
            #             +str(self.domainValues[2][di2]))
            di2 += 1

    def plotAxesLabels(self, axis):
        if not self.showAxesLabels:
            axis.set_xlabel('')
            axis.set_ylabel('')
            return

        if self.graphMode == viewModes.MODE_1D:
            axis.set_xlabel(self.domainNames[0])
            axis.set_ylabel(self.valueLabel)
        elif self.graphMode == viewModes.MODE_2D:
            axis.set_ylabel(self.domainNames[0])

            if self.domainValues[1] != None:
                axis.set_xlabel(self.domainNames[1])
        else: #MODE_HIST
            axis.set_xlabel(self.valueLabel)
            axis.set_ylabel('density')


    def plot2D(self):
        if debug: print 'MatplotlibWidget.computeFigure: MODE_2D'
        di2 = 0
        self.colorBars = []
        for ax in self.axes:
            ax.hold(True)
            if self.mask != None:
                self.showMask(ax, self.mask[:,:,di2])
            #print 'maskedValues:', self.maskedValues.min(), self.maskedValues.max()
            if not hasattr(ax, 'matshow'): # matplotlib version < 0.9:
                ms = _matshow(ax, self.maskedValues[:,:,di2], cmap=self.colorMap,
                              norm=self.norm)
            else:
                ms = ax.matshow(self.maskedValues[:,:,di2], cmap=self.colorMap,
                                norm=self.norm)

            if self.showColorBar and len(self.axes)<2:
                if debug: print ' plot colorbar ...'
                self.colorBars.append(self.fig.colorbar(ms))

            if not self.showAxesFlag:
                ax.set_axis_off()
            else:
                setTickLabels(ax.yaxis, self.domainValues[0])
                if self.domainValues[1] != None:
                    setTickLabels(ax.xaxis, self.domainValues[1])

                self.plotAxesLabels(ax)

            #ax.set_title(self.domainNames[2]+' ' \
            #             +str(self.domainValues[2][di2]))
            di2 += 1

    def plotHist(self):
        if debug: print 'MatplotlibWidget.computeFigure: MODE_HIST'

        v = self.values
        if 0 and self.mask != None:
            if debug: print 'self.mask:', _N.unique(self.mask)
            if self.maskLabel > 0:
                #if debug: print 'self.values[self.mask == %d] :' %self.maskLabel
                #if debug: print self.values[self.mask == self.maskLabel]
                vs = [self.values[self.mask == self.maskLabel]]
            else:
                #if debug: print 'self.values[self.mask != 0] :'
                #if debug: print self.values[self.mask != 0]
                vs = [self.values[self.mask == ml] for ml in self.maskLabels
                      if ml!=0]

        else:
            vs = [self.values]
        bins = 30
        #bins = _N.arange(0,1.,0.05)
        normed = True
        colors = ['b','r','g']
        n,bins = _N.histogram(self.values, bins=bins, normed=normed)

        for iv, v in enumerate(vs):
            if v.size > 0:
                fColor = colors[iv]
                alpha = 0.5 if iv > 0 else 1.
                self.axes[0].hold(True)
                n,b,p = self.axes[0].hist(v, bins=bins, normed=normed, fc=fColor,
                                          alpha=alpha)
                #if type(bins) == int :
                #    bins = b
            else:
                print "Nothing in histogram"

        self.plotAxesLabels(self.axes[0])

    def computeFigure(self):
        if debug: print 'MatplotlibWidget.computeFigure: ...'

        # Reset subplots adjustment:
        #self.fig.subplots_adjust(left=0.15, right=.9, bottom=0.2, top=.8,
        #                         hspace=0.1, wspace=0.1)
        self.resetGraph()
        self.refreshFigure()

    def refreshFigure(self):
        if debug : print 'MatplotlibWidget.refreshFigure ...'
        if self.graphMode == viewModes.MODE_1D:
            self.plot1D()
        elif self.graphMode == viewModes.MODE_2D:
            self.plot2D()
        elif self.graphMode == viewModes.MODE_HIST:
            self.plotHist()

        if debug: print 'fig:', self.fig.get_figwidth(), self.fig.get_figheight()

        self.draw()

    def save(self, fn):
        if debug : print 'MatplotlibWidget: self.fig.savefig ...'
        self.fig.savefig(fn)


    def printVal(self, v):
        if not _N.isreal(v): # assume string
            return v
        #else assume number
        elif int(v) == v:  #integer
            return '%d' %v
        else: # float
            return '%1.3g' %v

    def onClick(self, event):
        if debug:
            print 'mpl press event !'
        if not event.inaxes:
            return
        i = round(event.ydata)
        j = round(event.xdata)
        if hasattr(event.inaxes, 'zValue'):
            #if self.graphMode == viewModes.MODE_2D:
            k = event.inaxes.zValue
        else:
            k = -1
        if debug:
            print 'click on :', (i,j,k)
            print 'self.values.shape :', self.values.shape
        if self.otherSlices is not None:
            pos = self.otherSlices.copy()
        else:
            pos = {}
        if self.graphMode == viewModes.MODE_2D:
            if i<self.values.shape[0] and j<self.values.shape[1]:
                for n in xrange(3):
                    if debug: print 'self.domainNames[n]:', self.domainNames[n]
                    if self.domainNames[n] != None:
                        dv = self.domainValues[n][[i,j,k][n]]
                        pos[self.domainNames[n]] = dv
                if self.mask is not None:
                    pos[self.maskName] = self.mask[i,j,k]
        pos[self.valueLabel] = self.values[i,j,k]
        if debug:
            print '-> ', pos
            print "emitting positionClicked ..."
        self.emit(PYSIGNAL("positionClicked"), (pos.keys(), pos.values()) )


    def onMove(self, event):
        if event.inaxes and hasattr(event.inaxes, 'zValue'):
            #print 'zVal:', event.inaxes.zValue
            k = event.inaxes.zValue
            i = round(event.ydata)
            j = round(event.xdata)
            #print 'xdata : %f, ydata : %f' %(event.xdata, event.ydata)
            #print 'i:%d, j:%d, k:%d' %(i,j,k)

            if self.graphMode == viewModes.MODE_2D:
                if i >=self.values.shape[0] or j>=self.values.shape[1]:
                    msg = ''
                else:
                    if self.mask==None or (self.mask[i,j,k] != 0 and          \
                                           (self.maskLabel==0 or          \
                                            self.mask[i,j,k]==self.maskLabel)):
                        msg = '%s: %1.3g' %(self.valueLabel, self.values[i,j,k])
                    else:
                        msg = 'background'
                    if self.mask != None:
                        msg += ', %s:%d' %(self.maskName, self.mask[i,j,k])

                if msg != '':
                    for n in xrange(3):
                        if self.domainNames[n] is not None:
                            dv = self.domainValues[n][[i,j,k][n]]
                            msg += ', %s: %s' \
                                   %(self.domainNames[n], self.printVal(dv))
                    if self.errors != None:
                        msg += ', error: %1.3g' %(self.errors[i,j,k])

            elif self.graphMode == viewModes.MODE_1D:
                msg = '%s: %1.3g, %s: %1.3g' %(self.domainNames[0],event.xdata,
                                               self.valueLabel,event.ydata)


            elif self.graphMode == viewModes.MODE_HIST:
                msg = '%s: %1.3g, %s: %1.3g' %(self.valueLabel,event.xdata,
                                               'freq',event.ydata)
        else:
            msg = ''
        self.emit(PYSIGNAL('pointerInfoChanged'), (msg,))


    def onRelease(self, event):
        if debug:print event.name
        if event.inaxes:
            self.matData = _N.random.randn(10,10)
            self.ai.set_data(self.matData)
            #self.cb.set_array(self.matData)
            #self.cb.set_clim([self.matData.min(), self.matData.max()])
            #self.cb.autoscale()
            #self.cb.changed()
            #self.cb.draw_all()
            #self.ai.changed()
            #self.cb.vmin = self.matData.min()
            #self.ai.changed()
            #self.cb.set_colorbar(self.matData,self.ai.colorbar[1])
            #self.cb.draw_all()
            #self.fig.draw(self.get_renderer())
            #self.axes = self.fig.add_subplot(111)
            #matData = _N.random.randn(10,10)
            #ai = self.axes.matshow(matData)
            #self.fig.colorbar(ai)

