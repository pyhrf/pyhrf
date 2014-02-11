# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mplwidget_tutorial.ui'
#
# Created: Thu Sep 15 15:59:51 2005
#      by: The PyQt User Interface Compiler (pyuic) 3.13
#
# WARNING! All changes made in this file will be lost!


from qt import *
from mplwidget import *
import viewModes
import numpy
from pyhrf.ndarray import *
# try:
#     from gradwidget import GradientWidget
# except ImportError , e:
#     try :
#         from soma.qtgui.gradwidget import GradientWidget
#     except ImportError , e:
#         print "Couldn't import GradientWidget class ..."\
#             "no gradient editor available"
#         GradientWidget = None

GradientWidget = None
    
debug = False

class GradientDialog(QDialog):
    def __init__(self,cmapString, vMin, vMax, parent = None,name = None, modal=0, fl = 0):
        QDialog.__init__(self,parent,name,modal,fl)
        
        if not name:
            self.setName("GradientDialog")
            
        #self.setCentralWidget(QWidget(self,"qt_central_widget"))
        self.Form1Layout = QVBoxLayout(self,11,6,"Form1Layout")
        if GradientWidget != None:
            self.gradEditor = GradientWidget(self, "gradEditor",
                                             QString(cmapString), vMin, vMax)
            self.gradEditor.setGeometry(QRect(0,0,200,200))
            self.Form1Layout.addWidget(self.gradEditor)
            self.connect(self.gradEditor, SIGNAL('gradientChanged(QString)'),
                         self.emitGradChanged)
            
        self.languageChange()

        self.resize(QSize(422,346).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)
    def emitGradChanged(self, s):
        if debug:print 'gradientChanged(String s) -> ', s
        self.emit(PYSIGNAL('gradientChanged'),(s,))

    def languageChange(self):
        #self.helloLabel.setText("blabla")
        self.setCaption(self.__tr("GradientEditor"))

    def __tr(self,s,c = None):
        return qApp.translate("GradientEditor",s,c)




class xndarrayViewRenderer(QWidget):
    def __init__(self, cuboidView, cubName, parent = None,name = None, 
                 modal=0, fl = 0):
        QWidget.__init__(self,parent,name,fl)
        #self.statusBar()
        self.cuboidName = cubName
        if not name:
            self.setName("mplViewer")
        #if type(cuboidViews) != list :
        #    cuboidViews = [cuboidViews]
        #self.cuboidViews = cuboidViews
        self.cuboidView = cuboidView
        #self.setCentralWidget(QWidget(self,"qt_central_widget"))
        Form1Layout = QVBoxLayout(self,0,0,"Form1Layout")

        self.topFrame = QFrame(self,"topFrame")
        #self.topFrame.setFrameShape(QFrame.NoFrame)
        #self.topFrame.setFrameShadow(QFrame.Plain)
        self.topFrame.setFrameShape(QFrame.StyledPanel)
        self.topFrame.setFrameShadow(QFrame.Raised)
        topFrameLayout = QHBoxLayout(self.topFrame,0,0,"topFrameLayout")

        # minVal = min(cv.getMinValue() for cv in self.cuboidViews)
        # maxVal = min(cv.getMaxValue() for cv in self.cuboidViews)        
        
        minVal = self.cuboidView.cuboid.data.min()
        maxVal = self.cuboidView.cuboid.data.max()

        #allData = numpy.array([cv.cuboid.data for cv in self.cuboidViews])
        #print 'valueRange : ', allData.min(), allData.max()
        #print 'std :', allData.std()
        #print 'mean :', allData.mean()
        #std = allData.std()
        #mean =  allData.mean()
        # Build 99% gaussian confidence range:
        #minVal = -3*std + mean
        #maxVal = 3*std + mean
        #print '95% confidence intervall:', minVal, maxVal
        #minVal = self.cuboidViews[0].getMinValue()
        #maxVal = self.cuboidViews[0].getMaxValue()

        maskLabels = self.cuboidView.getMaskLabels()
        #maskLabels = None
        self.view1D = MatplotlibWidget(viewModes.MODE_1D, self.topFrame,
                                       "view1D", valueRange=[minVal,maxVal],
                                       maskLabels=maskLabels)
        self.view1D.hide()
        self.mode = viewModes.MODE_1D
        self.updateView()
        #if 'nrl' in cubName:
        #    minVal = -100
        #    maxVal = 150
        self.view2D = MatplotlibWidget(viewModes.MODE_2D, self.topFrame,
                                       "view2D", valueRange=[minVal,maxVal],
                                       maskLabels=maskLabels)
        self.mode = viewModes.MODE_2D
        self.updateView()
        topFrameLayout.addWidget(self.view1D)
        topFrameLayout.addWidget(self.view2D)

        cm = QString(self.view2D.getColorMapString())
        self.gradDialog = GradientDialog(cm, vMin=minVal, vMax=maxVal,
                                         name="gradDialog")
        
        self.buttonFrame = QFrame(self.topFrame,"buttonFrame")
        self.buttonFrame.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Preferred,0,0,self.buttonFrame.sizePolicy().hasHeightForWidth()))
        self.buttonFrame.setFrameShape(QFrame.Panel)
        self.buttonFrame.setFrameShadow(QFrame.Raised)
        self.buttonFrame.setLineWidth(2)
        self.buttonFrame.setMargin(0)
        buttonFrameLayout = QVBoxLayout(self.buttonFrame,4,0,"buttonFrameLayout")
        
        self.saveButton = QPushButton(self.buttonFrame,"saveButton")
        ssp = self.saveButton.sizePolicy()
        self.saveButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,
                                                  QSizePolicy.Fixed,0,0,
                                                  ssp.hasHeightForWidth()))
        self.saveButton.setMinimumSize(QSize(70,27))
        self.saveButton.setMaximumSize(QSize(70,27))
        buttonFrameLayout.addWidget(self.saveButton)

        self.colorsButton = QPushButton(self.buttonFrame,"colorsButton")
        self.colorsButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.colorsButton.sizePolicy().hasHeightForWidth()))
        self.colorsButton.setMinimumSize(QSize(70,27))
        self.colorsButton.setMaximumSize(QSize(70,27))
        self.colorsMenu = QPopupMenu(self,"colorsMenu")
        #self.colorsMenu.insertItem("color map",self.editColorMap)
        self.colorsMenu.insertItem("background",self.chooseBackgroundColor)
        self.colorsButton.setPopup(self.colorsMenu)        
        buttonFrameLayout.addWidget(self.colorsButton)

        self.colbarButton = QPushButton(self.buttonFrame,"colbarButton")
        self.colbarButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.colbarButton.sizePolicy().hasHeightForWidth()))
        self.colbarButton.setMinimumSize(QSize(70,27))
        self.colbarButton.setMaximumSize(QSize(70,27))
        buttonFrameLayout.addWidget(self.colbarButton)

        self.maskDisplayButton = QPushButton(self.buttonFrame,"maskDisplayButton")
        self.maskDisplayButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.maskDisplayButton.sizePolicy().hasHeightForWidth()))
        self.maskDisplayButton.setMinimumSize(QSize(70,27))
        self.maskDisplayButton.setMaximumSize(QSize(70,27))
        self.maskDisplayMenu = QPopupMenu(self,"maskDisplayMenu")
        self.maskDisplayMenu.insertItem("hide",self.hideMask)
        self.maskDisplayMenu.insertItem("show",self.showMask)
        #self.maskDisplayMenu.insertItem("show only",self.showMaskOnly)
        self.maskDisplayButton.setPopup(self.maskDisplayMenu)        
        buttonFrameLayout.addWidget(self.maskDisplayButton)



        self.axesButton = QPushButton(self.buttonFrame,"axesButton")
        self.axesButtonMenu = QPopupMenu(self,"axesButtonMenu")
        self.axesButtonMenu.insertItem("hide",self.hideAxes)
        self.axesButtonMenu.insertItem("show",self.showAxes)
        self.axesButtonMenu.insertItem("toggle labels",self.toggleAxesLabels)
        #self.axesButtonMenu.insertItem("set labels",self.setAxesLabels)
        self.axesButton.setPopup(self.axesButtonMenu)        
        self.axesButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.axesButton.sizePolicy().hasHeightForWidth()))
        self.axesButton.setMinimumSize(QSize(70,27))
        buttonFrameLayout.addWidget(self.axesButton)

        # self.stackButton = QPushButton(self.buttonFrame,"stackButton")
        # sp = self.stackButton.sizePolicy().hasHeightForWidth()
        # self.stackButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,
        #                                            QSizePolicy.Fixed,0,0,sp))
        # self.stackButton.setMinimumSize(QSize(70,27))
        # buttonFrameLayout.addWidget(self.stackButton)

        self.graphModeButton = QPushButton(self.buttonFrame,"graphModeButton")
        self.graphModeButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.graphModeButton.sizePolicy().hasHeightForWidth()))
        self.graphModeButton.setMinimumSize(QSize(70,27))
        self.graphModeMenu = QPopupMenu(self,"graphModeMenu")
        self.graphModeMenu.insertItem("1D",self.setGraphMode1D)
        self.graphModeMenu.insertItem("2D",self.setGraphMode2D)
        self.graphModeMenu.insertItem("Hist",self.setGraphModeHist)
        self.graphModeButton.setPopup(self.graphModeMenu)
        buttonFrameLayout.addWidget(self.graphModeButton)

        topFrameLayout.addWidget(self.buttonFrame)
        Form1Layout.addWidget(self.topFrame)

        self.dimInfoLabel = QLabel(self,"dimInfoLabel")
        self.dimInfoLabel.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed,0,0,self.dimInfoLabel.sizePolicy().hasHeightForWidth()))
        Form1Layout.addWidget(self.dimInfoLabel)

        self.valueInfoLabel = QLabel(self,"valueInfoLabel")
        self.valueInfoLabel.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed,0,0,self.valueInfoLabel.sizePolicy().hasHeightForWidth()))
        self.valueInfoLabel.setText('')
        Form1Layout.addWidget(self.valueInfoLabel)
        
        self.connect(self.view2D, PYSIGNAL('pointerInfoChanged'),
                     self.updateInfoLabel)
        self.connect(self.view2D, PYSIGNAL('positionClicked'),
                     PYSIGNAL('positionClicked'))
        self.connect(self.view1D, PYSIGNAL('pointerInfoChanged'),
                     self.updateInfoLabel)
        self.connect(self.view1D, PYSIGNAL('valuesChanged'),
                     self.updateValueInfoLabel)
        self.connect(self.view2D, PYSIGNAL('valuesChanged'),
                     self.updateValueInfoLabel)
        
        self.connect(self.saveButton, SIGNAL('clicked()'), self.saveFig)
        #self.connect(self.stackButton, SIGNAL('clicked()'), self.handleStack)
        self.connect(self.colbarButton, SIGNAL('clicked()'), self.toggleColorBar)

        self.connect(self.gradDialog, PYSIGNAL('gradientChanged'),
                     self.applyColorMap)
        
        self.languageChange()

        self.resize(QSize(422,346).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)


        if self.cuboidView.current_axes == ['time']:
            self.setGraphMode1D()


    def updateValueInfoLabel(self, infoLabel):
        self.valueInfoLabel.setText(infoLabel)

    def closeEvent(self, event):
        self.emit(PYSIGNAL("windowClosed"),())
        QWidget.closeEvent(self, event)


    def applyColorMap(self, scm):
        if debug : print 'xndarrayViewRenderer.applyColormap ...'
        if debug:print 'scm:', str(scm)
        self.view2D.setColorMapFromString(scm)
        self.view2D.resetGraph()
        self.view2D.refreshFigure()
        
    def setMaskLabel(self, axn, label):
        if debug:print 'xndarrayViewRenderer.setMaskLabel ...'
        self.view1D.setMaskLabel(label)
        self.view2D.setMaskLabel(label)
        #if self.mode == viewModes.MODE_1D or self.mode == viewModes.MODE_HIST:
        #    self.view1D.computeFigure()
        #else:
        #    self.view2D.computeFigure()

    def hideAxes(self):
        if debug:print 'xndarrayViewRenderer.hideAxes ...'
        self.view1D.hideAxes()
        self.view2D.hideAxes()

    def showAxes(self):
        if debug:print 'xndarrayViewRenderer.showAxes ...'
        self.view1D.showAxes()
        self.view2D.showAxes()

    def toggleAxesLabels(self):
        if debug:print 'xndarrayViewRenderer.toggleAxesLabels ...'
        self.view1D.toggleAxesLabels()
        self.view2D.toggleAxesLabels()
        
    def updateInfoLabel(self, info):
        self.dimInfoLabel.setText(info)

    def setGraphMode1D(self):
        if self.mode == viewModes.MODE_1D:
            return
        self.mode = viewModes.MODE_1D
        self.view2D.hide()
        self.view1D.setGraphMode(viewModes.MODE_1D)
        self.view1D.show()
        self.updateView()

    def setGraphMode2D(self):
        if self.mode == viewModes.MODE_2D:
            return
        self.mode = viewModes.MODE_2D
        self.view1D.hide()
        self.view2D.show()
        self.updateView()

    def setGraphModeHist(self):
        if self.mode == viewModes.MODE_HIST:
            return
        self.mode = viewModes.MODE_HIST
        self.view2D.hide()
        self.view1D.setGraphMode(viewModes.MODE_HIST)
        self.view1D.show()
        self.updateView()

        
    def updateView(self):
        if debug:print 'cuboidViewRenderer -> updateView ...'
        if debug:print 'current mode :', self.mode

        #cubs = []
        #cv = self.cuboidViews[0]
        #for cv in self.cuboidViews:
        # view = cv.getView()
        # doms = dict(zip(view['domainNames'], view['domainValues']))
        # cubs.append(xndarray(view['values'], errors=view['errors'],
        #                    axesNames=view['domainNames'], axesDomains=doms,
        #                    valueLabel = cv.getValueLabel()))
        # if len(cubs) > 1:
        #     cub = joinxndarrays(cubs)
        # else:
        #     cub = cubs[0]

        cub, slices = self.cuboidView.getView()
        (mask, mslice), maskName = self.cuboidView.get_mask_view()

        if mask is None:
            mdata = None
        else:
            mdata = mask.data
        if self.mode == viewModes.MODE_1D or self.mode == viewModes.MODE_HIST:
            if debug:print 'updating view1D ...'
            self.view1D.updateData(cub, mask=mdata,
                                   maskLabel=0,
                                   maskName=maskName)
            #self.view1D.updateData(cub)
        elif self.mode == viewModes.MODE_2D or self.mode == viewModes.MODE_3D:
            if debug:print 'updating view2D ...'
            self.view2D.updateData(cub, mask=mdata,
                                   maskLabel=0,
                                   maskName=maskName,
                                   otherSlices=slices)
            #self.view2D.updateData(cub, otherSlices=slices)
            
    def suggest_file_name(self):
        print 'objName:', self.cuboidName
        fn = self.cuboidName
        _,slices = self.cuboidView.getView()
        for sn, sv in slices.iteritems():
            fn += '_'+sn+'_'+str(sv)
        return fn+'.png'

    def saveFig(self):
        if debug:print 'saving fig ...'
        fn = os.path.join('./',self.suggest_file_name())
        print 'file suggestion:', fn
        fileName = str(QFileDialog.getSaveFileName(fn,'Images (*.png *.eps)',None))
        if fileName:
            print "return from fileChooser: selected=", fileName
            if self.mode == viewModes.MODE_HIST or \
                   self.mode == viewModes.MODE_1D:
                self.view1D.save(fileName)
            else: # MODE_2D
                self.view2D.save(fileName)
                
    def handleStack(self):
        pass

    def chooseBackgroundColor(self):
        if debug : print 'chooseBackgroundColor ...'
        c = QColorDialog.getColor()
        if debug : print 'color:', c
        if c.isValid():
            self.view2D.setBackgroundColor(c)
            self.view2D.resetGraph()
            self.view2D.refreshFigure() #TODO don't need to replot everything ...

    def editColorMap(self):
        if debug : print 'call color map editor ...'
        self.gradDialog.exec_loop()
        
    def toggleColorBar(self):

        if self.mode == viewModes.MODE_2D:
            self.view2D.toggleColorbar()
            
    def showMask(self):
        if self.mode == viewModes.MODE_2D:
            self.view2D.setMaskDisplay(MatplotlibWidget.MASK_SHOWN)

    def showMaskOnly(self):
        if self.mode == viewModes.MODE_2D:
            self.view2D.setMaskDisplay(MatplotlibWidget.MASK_ONLY)

    def hideMask(self):
        if self.mode == viewModes.MODE_2D:
            self.view2D.setMaskDisplay(MatplotlibWidget.MASK_HIDDEN)
        
    def languageChange(self):
        self.setCaption(self.__tr("Viewer"))
        self.saveButton.setText(self.__tr("Save"))
        self.colorsButton.setText(self.__tr("Colors"))
        self.colbarButton.setText(self.__tr("Colbar"))
        self.maskDisplayButton.setText(self.__tr("Mask"))
        self.axesButton.setText(self.__tr("Axes"))
        #self.stackButton.setText(self.__tr("Stack"))
        self.graphModeButton.setText(self.__tr("Mode"))


        
    def __tr(self,s,c = None):
        return qApp.translate("Viewer",s,c)




if __name__ == "__main__":

    import sys
    sys.path.append('../')
    from core import *
    from numpy import *
    from ndview import xndarrayView

    app = QApplication(sys.argv)
    QObject.connect(app,SIGNAL("lastWindowClosed()"),app,SLOT("quit()"))
    ar = arange(2*3*4).reshape((2,3,4))
    dimNames = ['x','y','beta']
    dimDomains = {'beta':arange(0,0.8,0.2)}
    print dimDomains
    c = xndarray(ar, axesNames=dimNames, axesDomains=dimDomains)
    v = xndarrayView(c)

    f = xndarrayViewRenderer(v, 'test')
    app.setMainWidget(f)
    f.show()
    app.exec_loop()
